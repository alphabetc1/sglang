import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
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
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


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

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)

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

        capacity_gb = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_CAPACITY", 0)
        if capacity_gb > 0:
            self.capacity_bytes = capacity_gb * 1e9
        else:
            self.capacity_bytes = 0
        self.ttl_seconds = int(os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_TTL", 0))
        self._lru_entries: OrderedDict[str, "_HiCacheFileEntry"] = OrderedDict()
        self._usage_bytes = 0

        self._hit_count_total = 0
        self._miss_count_total = 0
        self._evict_count_total = 0
        self._expired_count_total = 0
        self._last_reported = {
            "hit": 0,
            "miss": 0,
            "evict": 0,
            "expired": 0,
        }

        if (self.capacity_bytes > 0):
            logger.info(f"HiCacheFile storage initialized with capacity {capacity_gb} GB")
            self._load_existing_files()

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _load_existing_files(self) -> None:
        if not os.path.exists(self.file_path):
            return

        suffix = f"{self.config_suffix}.bin"
        entries = []
        for filename in os.listdir(self.file_path):
            if not filename.endswith(suffix):
                continue
            path = os.path.join(self.file_path, filename)
            if not os.path.isfile(path):
                continue
            try:
                size_bytes = os.path.getsize(path)
                mtime = os.path.getmtime(path)
            except OSError as e:
                logger.warning(f"Failed to stat HiCacheFile entry {path}: {e}")
                continue
            entries.append((mtime, filename[:-4], size_bytes))

        entries.sort(key=lambda item: item[0])
        for mtime, key, size_bytes in entries:
            self._lru_entries[key] = _HiCacheFileEntry(
                size_bytes=size_bytes,
                created_ts=mtime,
                last_access_ts=mtime,
            )
            self._usage_bytes += size_bytes

    def _touch_entry(self, key: str, now: float) -> None:
        entry = self._lru_entries.get(key)
        if entry is None:
            return
        entry.last_access_ts = now
        self._lru_entries.move_to_end(key)

    def _record_hit(self) -> None:
        self._hit_count_total += 1

    def _record_miss(self) -> None:
        self._miss_count_total += 1

    def _record_evict(self) -> None:
        self._evict_count_total += 1

    def _record_expired(self) -> None:
        self._expired_count_total += 1

    def _is_expired(self, entry: _HiCacheFileEntry, now: float) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (now - entry.created_ts) >= self.ttl_seconds

    def _remove_entry(self, key: str) -> None:
        entry = self._lru_entries.pop(key, None)
        if entry is not None:
            self._usage_bytes -= entry.size_bytes
            if self._usage_bytes < 0:
                self._usage_bytes = 0
        path = os.path.join(self.file_path, f"{key}.bin")
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to remove HiCacheFile entry {path}: {e}")

    def _ensure_entry_from_disk(self, key: str) -> Optional["_HiCacheFileEntry"]:
        entry = self._lru_entries.get(key)
        if entry is not None:
            return entry

        path = os.path.join(self.file_path, f"{key}.bin")
        if not os.path.exists(path):
            return None
        try:
            size_bytes = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError as e:
            logger.warning(f"Failed to stat HiCacheFile entry {path}: {e}")
            return None

        entry = _HiCacheFileEntry(
            size_bytes=size_bytes,
            created_ts=mtime,
            last_access_ts=mtime,
        )
        self._lru_entries[key] = entry
        self._usage_bytes += size_bytes
        return entry

    def _evict_for_capacity(self, incoming_bytes: int) -> bool:
        if self.capacity_bytes <= 0:
            return True
        if incoming_bytes > self.capacity_bytes:
            logger.warning(
                "HiCacheFile entry size exceeds capacity: "
                f"{incoming_bytes} > {self.capacity_bytes}"
            )
            return False

        while self._usage_bytes + incoming_bytes > self.capacity_bytes:
            if not self._lru_entries:
                return False
            oldest_key, _ = next(iter(self._lru_entries.items()))
            self._remove_entry(oldest_key)
            self._record_evict()
        return True

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        now = time.time()
        entry = self._ensure_entry_from_disk(key)
        if entry is not None and self._is_expired(entry, now):
            self._remove_entry(key)
            self._record_expired()
            self._record_miss()
            return None
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            self._touch_entry(key, now)
            self._record_hit()
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            self._record_miss()
            return None
        except Exception as e:
            logger.error(f"Failed to read tensor {key}: {e}")
            self._record_miss()
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
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        now = time.time()

        entry = self._ensure_entry_from_disk(key)
        if entry is not None:
            if self._is_expired(entry, now):
                self._remove_entry(key)
                self._record_expired()
            else:
                logger.debug(f"Key {key} already exists. Skipped.")
                self._touch_entry(key, now)
                return True

        size_bytes = value.numel() * value.element_size()
        if not self._evict_for_capacity(size_bytes):
            return False
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            self._lru_entries[key] = _HiCacheFileEntry(
                size_bytes=size_bytes,
                created_ts=now,
                last_access_ts=now,
            )
            self._usage_bytes += size_bytes
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
        now = time.time()
        entry = self._ensure_entry_from_disk(key)
        if entry is not None and self._is_expired(entry, now):
            self._remove_entry(key)
            self._record_expired()
            self._record_miss()
            return False

        exists = os.path.exists(tensor_path)
        if exists:
            self._touch_entry(key, now)
            self._record_hit()
            return True
        self._record_miss()
        return False

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            self._lru_entries.clear()
            self._usage_bytes = 0
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False

    def get_stats(self):
        from sglang.srt.metrics.collector import StorageMetrics

        storage_metrics = StorageMetrics()
        storage_metrics.capacity_gb = float(self.capacity_bytes / 1e9)
        storage_metrics.current_usage_gb = float(self._usage_bytes / 1e9)

        deltas = {
            "hit": self._hit_count_total - self._last_reported["hit"],
            "miss": self._miss_count_total - self._last_reported["miss"],
            "evict": self._evict_count_total - self._last_reported["evict"],
            "expired": self._expired_count_total - self._last_reported["expired"],
        }
        storage_metrics.hit_count = deltas["hit"]
        storage_metrics.miss_count = deltas["miss"]
        storage_metrics.evict_count = deltas["evict"]
        storage_metrics.expired_count = deltas["expired"]

        total_access = self._hit_count_total + self._miss_count_total
        if total_access > 0:
            storage_metrics.hit_ratio = self._hit_count_total / total_access

        self._last_reported["hit"] = self._hit_count_total
        self._last_reported["miss"] = self._miss_count_total
        self._last_reported["evict"] = self._evict_count_total
        self._last_reported["expired"] = self._expired_count_total

        return storage_metrics


@dataclass
class _HiCacheFileEntry:
    size_bytes: int
    created_ts: float
    last_access_ts: float
