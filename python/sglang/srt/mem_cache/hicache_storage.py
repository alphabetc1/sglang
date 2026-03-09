import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Token ids can be plain ints or tuple-packed ids (e.g. EAGLE bigram mode).
TokenId: TypeAlias = int | tuple[int, ...]


def get_hash_str(token_ids: List[TokenId], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    # Take first 16 hex chars to get 64-bit value
    uint64_val = int(hash_str[:16], 16)
    # Convert to signed int64 range [-2^63, 2^63-1]
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


class BaseStorageMetadata:
    """Base class for storage metadata payloads."""


@dataclass
class WarmupStorageMetadata(BaseStorageMetadata):
    token_ids: List[TokenId]
    priority: int = 0


@dataclass
class StorageMetadataRequest:
    warmup: Optional[WarmupStorageMetadata] = None


@dataclass
class WarmupEntry:
    """An entry available for warmup loading from storage."""

    token_ids: List[TokenId]
    hash_chain: List[str]
    priority: int
    num_tokens: int
    extra_key: Optional[str] = None
    timestamp: float = 0.0


class WarmupEntrySelector(ABC):
    @abstractmethod
    def select_entries(
        self, entries: List[WarmupEntry], max_tokens: int = 0
    ) -> List[WarmupEntry]:
        """Select warmup entries under the given token budget."""


class DefaultWarmupEntrySelector(WarmupEntrySelector):
    """Default warmup policy.

    Priority order:
    1. higher-priority entries
    2. more recent entries
    """

    # TODO: expose this threshold through backend config once policy semantics stabilize.
    MIN_WARMUP_PREFIX_TOKENS = 256

    def _is_eligible(self, entry: WarmupEntry) -> bool:
        return entry.num_tokens >= self.MIN_WARMUP_PREFIX_TOKENS

    def select_entries(
        self, entries: List[WarmupEntry], max_tokens: int = 0
    ) -> List[WarmupEntry]:
        ranked = sorted(
            (entry for entry in entries if self._is_eligible(entry)),
            key=lambda entry: (entry.priority, entry.timestamp),
            reverse=True,
        )

        if max_tokens <= 0:
            return ranked

        selected: List[WarmupEntry] = []
        total_tokens = 0
        for entry in ranked:
            if total_tokens + entry.num_tokens > max_tokens:
                if total_tokens >= max_tokens:
                    break
                continue
            selected.append(entry)
            total_tokens += entry.num_tokens
            if total_tokens >= max_tokens:
                break
        return selected


DEFAULT_WARMUP_ENTRY_SELECTOR = DefaultWarmupEntrySelector()


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def record_warmup_metadata(
        self,
        keys: List[str],
        extra_info: Optional["HiCacheStorageExtraInfo"],
    ):
        """Record metadata for warmup after a successful batch write.

        Called by the cache controller after each successful page backup batch.
        Default implementation does nothing. Storage backends that support warmup
        should override this to persist metadata for later enumeration.
        """
        pass

    def record_storage_metadata(
        self,
        keys: List[str],
        storage_metadata: StorageMetadataRequest,
    ) -> None:
        """Generic metadata recording entrypoint for storage-owned modules.

        Default behavior routes the `warmup` payload to the legacy warmup API.
        Backends may override this method to support additional metadata fields while
        keeping `record_warmup_metadata` backward compatible.
        """
        if not isinstance(storage_metadata, StorageMetadataRequest):
            return
        warmup_metadata = storage_metadata.warmup
        if warmup_metadata is not None:
            self.record_warmup_metadata(
                keys,
                HiCacheStorageExtraInfo(
                    extra_info={
                        "token_ids": list(warmup_metadata.token_ids),
                        "priority": int(warmup_metadata.priority),
                    }
                ),
            )

    def supports_warmup_manifest(self) -> bool:
        """Whether this backend supports warmup manifest read/write.

        Backward compatibility: infer support when subclass overrides both
        `record_warmup_metadata` and `list_warmup_entries`.
        """
        cls = type(self)
        has_record = (
            cls.record_warmup_metadata is not HiCacheStorage.record_warmup_metadata
        )
        has_list = cls.list_warmup_entries is not HiCacheStorage.list_warmup_entries
        return has_record and has_list

    def list_warmup_entries(
        self,
        max_tokens: int = 0,
    ) -> List[WarmupEntry]:
        """List stored entries for warmup, ordered by priority desc then recency desc.

        Args:
            max_tokens: Max total tokens to return (0 = unlimited).

        Returns:
            List of WarmupEntry, sorted by priority desc then recency desc.
            Storage backends that don't support warmup return [].
        """
        return []

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):
    _DEFAULT_MANIFEST_SCAN_BYTES = 64 * 1024 * 1024

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        self._manifest_path = os.path.join(
            self.file_path, f"__warmup_manifest__{self.config_suffix}.jsonl"
        )
        self._manifest_lock = threading.Lock()
        extra_config = storage_config.extra_config or {}
        self._manifest_scan_bytes = self._parse_non_negative_int(
            extra_config.get(
                "warmup_manifest_scan_bytes", self._DEFAULT_MANIFEST_SCAN_BYTES
            ),
            default=self._DEFAULT_MANIFEST_SCAN_BYTES,
        )
        self._manifest_cursor_offset = 0
        self._manifest_latest_by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    @staticmethod
    def _parse_non_negative_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed < 0:
            return default
        return parsed

    def _append_manifest(
        self,
        keys: List[str],
        info: Dict[str, Any],
    ):
        """Append an entry to the warmup manifest file.

        Args:
            keys: Hash chain for the entry.
            info: Dict with keys: token_ids, priority.
        """
        token_ids = info.get("token_ids", [])
        record = {
            "hash_chain": keys,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
            "priority": info.get("priority", 0),
            "timestamp": time.time(),
        }
        try:
            with self._manifest_lock:
                with open(self._manifest_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
        except Exception:
            logger.debug("Failed to append warmup manifest entry.", exc_info=True)

    def record_warmup_metadata(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo],
    ):
        """Record metadata for warmup after a successful batch write."""
        if extra_info is None or extra_info.extra_info is None:
            return
        info = extra_info.extra_info
        if not isinstance(info.get("token_ids"), list):
            return
        self._append_manifest(keys, info)

    def _normalize_manifest_token_ids(
        self, token_ids: List[Any]
    ) -> Optional[List[Any]]:
        normalized = []
        for token in token_ids:
            if isinstance(token, int):
                normalized.append(token)
                continue
            if isinstance(token, tuple):
                if not all(isinstance(x, int) for x in token):
                    return None
                normalized.append(token)
                continue
            if isinstance(token, list):
                # JSON serializes Python tuples as lists.
                if not all(isinstance(x, int) for x in token):
                    return None
                normalized.append(tuple(token))
                continue
            return None
        return normalized

    def _parse_warmup_manifest_line(
        self, line: str, include_token_ids: bool
    ) -> Optional[Tuple[Tuple[str, ...], Dict[str, Any]]]:
        line = line.strip()
        if not line:
            return None
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None

        hash_chain = record.get("hash_chain")
        token_ids = record.get("token_ids")
        raw_priority = record.get("priority", 0)
        extra_key = record.get("extra_key")
        if not isinstance(hash_chain, list) or len(hash_chain) == 0:
            return None
        if not isinstance(token_ids, list):
            return None
        token_ids = self._normalize_manifest_token_ids(token_ids)
        if token_ids is None:
            return None

        if not all(isinstance(h, str) for h in hash_chain):
            return None
        if extra_key is not None and not isinstance(extra_key, str):
            return None
        dedup_key = tuple(hash_chain)

        try:
            priority = int(raw_priority)
            timestamp = float(record.get("timestamp", 0))
            num_tokens = int(record.get("num_tokens", len(token_ids)))
        except (TypeError, ValueError):
            return None

        if num_tokens < 0:
            return None

        num_tokens = min(num_tokens, len(token_ids))
        parsed = {
            "hash_chain": hash_chain,
            "priority": priority,
            "timestamp": timestamp,
            "num_tokens": num_tokens,
            "extra_key": extra_key,
        }
        if include_token_ids:
            parsed["token_ids"] = token_ids[:num_tokens]
        return dedup_key, parsed

    def supports_warmup_manifest(self) -> bool:
        return True

    def list_warmup_entries(
        self,
        max_tokens: int = 0,
    ) -> List[WarmupEntry]:
        """Read manifest incrementally and return deduped, selected entries."""
        try:
            with self._manifest_lock:
                if not os.path.exists(self._manifest_path):
                    self._manifest_cursor_offset = 0
                    self._manifest_latest_by_key = {}
                    return []

                file_size = os.path.getsize(self._manifest_path)
                if self._manifest_cursor_offset > file_size:
                    # Manifest might be truncated/recreated; restart incremental state.
                    self._manifest_cursor_offset = 0
                    self._manifest_latest_by_key = {}

                start_offset = self._manifest_cursor_offset
                if (
                    start_offset == 0
                    and self._manifest_scan_bytes > 0
                    and file_size > self._manifest_scan_bytes
                ):
                    # Bootstrap from tail only to avoid scanning huge manifests.
                    start_offset = file_size - self._manifest_scan_bytes
                    self._manifest_latest_by_key = {}
                skip_partial_line = start_offset > 0 and (
                    start_offset != self._manifest_cursor_offset
                )

                with open(self._manifest_path, "r") as f:
                    if start_offset > 0:
                        f.seek(start_offset)
                        if skip_partial_line:
                            # Skip a potential partial line from mid-file seek.
                            _ = f.readline()
                    for line in f:
                        parsed = self._parse_warmup_manifest_line(
                            line, include_token_ids=True
                        )
                        if parsed is None:
                            continue
                        dedup_key, record = parsed
                        self._manifest_latest_by_key[dedup_key] = record
                    self._manifest_cursor_offset = f.tell()

                latest_by_key = dict(self._manifest_latest_by_key)
        except Exception:
            logger.warning("Failed to read warmup manifest.", exc_info=True)
            return []

        if not latest_by_key:
            return []

        entries = [
            WarmupEntry(
                token_ids=record["token_ids"],
                hash_chain=record["hash_chain"],
                priority=record["priority"],
                num_tokens=record["num_tokens"],
                extra_key=record.get("extra_key"),
                timestamp=record["timestamp"],
            )
            for record in latest_by_key.values()
        ]
        return DEFAULT_WARMUP_ENTRY_SELECTOR.select_entries(
            entries, max_tokens=max_tokens
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self._manifest_cursor_offset = 0
            self._manifest_latest_by_key = {}
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
