# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig


def _require_cpu_tensor(t: torch.Tensor, *, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if t.is_cuda:
        raise ValueError(
            f"{name} must be a CPU tensor. PDKP integration currently supports CPU pinned memory only."
        )
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous (call .contiguous()).")


def _try_import_pdkp_sdk():
    """
    Import helper for `pdkp_sglang_sdk`.

    Users often copy `pdkp_sglang_sdk*.so` into a repo directory but forget to
    add that directory to PYTHONPATH. Here we try a small set of conventional
    locations to make the integration easier to use.
    """
    try:
        from . import pdkp_sglang_sdk  # type: ignore

        return pdkp_sglang_sdk
    except Exception:
        pass

    candidates: List[str] = []
    # User-provided override.
    if os.getenv("PDKP_SGLANG_SDK_DIR"):
        candidates.append(os.getenv("PDKP_SGLANG_SDK_DIR"))

    # Common suggested location from older integration notes:
    # python/sglang/srt/mem_cache/storage/pdkp/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_candidate = os.path.normpath(
        os.path.join(this_dir, "..", "..", "mem_cache", "storage", "pdkp")
    )
    candidates.append(repo_candidate)

    # Also try the current directory (if the .so was placed next to this file).
    candidates.append(this_dir)

    for d in candidates:
        if not d:
            continue
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
        try:
            from . import pdkp_sglang_sdk  # type: ignore

            return pdkp_sglang_sdk
        except Exception:
            continue

    raise RuntimeError(
        "PDKP backend requires `pdkp_sglang_sdk` to be importable. "
        "Place `pdkp_sglang_sdk*.so` into site-packages or a directory on PYTHONPATH. "
        "You can also set env PDKP_SGLANG_SDK_DIR to the directory containing the .so."
    )


class _PDKPClientWrapper:
    """
    A small wrapper that:
    - imports `pdkp_sglang_sdk`
    - ensures local buffers are registered once before put/get
    """

    def __init__(self, config: Dict[str, Any]):
        self._sdk = _try_import_pdkp_sdk()

        world_size = int(config.get("world_size", 1))
        rank_id = int(config.get("rank_id", 0))
        self._client = self._sdk.SGLangClient(world_size=world_size, rank_id=rank_id)

        cfg_path = config.get("client_config_path") or os.getenv(
            "PDKP_SGLANG_CLIENT_CONFIG"
        )
        if not cfg_path:
            raise ValueError(
                "Missing PDKP client config. Provide storage extra_config['client_config_path'] "
                "or set env PDKP_SGLANG_CLIENT_CONFIG."
            )
        ok = self._client.init(str(cfg_path))
        if not ok:
            raise RuntimeError(f"PDKP client init failed for config_path={cfg_path!r}")

        self._registered: Dict[tuple[int, int], int] = {}  # (addr, size)->lkey

    def _ensure_registered(self, t: torch.Tensor) -> None:
        _require_cpu_tensor(t, name="tensor")
        addr = int(t.data_ptr())
        size = int(t.numel() * t.element_size())
        key = (addr, size)
        if key in self._registered:
            return
        ok, lkey = self._client.register_memory(addr, size)
        if not ok:
            raise RuntimeError(
                "PDKP register_memory failed. Ensure the tensor is CPU pinned memory "
                "(or at least a buffer type supported by your PDKP build)."
            )
        self._registered[key] = int(lkey)

    def put_tensor(self, key: str, value: torch.Tensor) -> bool:
        self._ensure_registered(value)
        status = self._client.put(key, value)
        return status == self._sdk.PDKPStatus.OK

    def get_tensor(self, key: str, out: torch.Tensor) -> bool:
        self._ensure_registered(out)
        status = self._client.get(key, out)
        return status == self._sdk.PDKPStatus.OK

    def batch_get_tensors(self, keys: List[str], outs: List[torch.Tensor]) -> List[bool]:
        for t in outs:
            self._ensure_registered(t)
        statuses = self._client.batch_get(keys, outs)
        return [s == self._sdk.PDKPStatus.OK for s in statuses]

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))


class HiCachePDKPStorage(HiCacheStorage):
    """
    HiCache storage backend implemented on top of PDKP distributed memory pool.

    Limitation: PDKP SDK supports CPU pinned memory only. This backend requires
    `target_location` / `value` to be CPU tensors (ideally pinned), and will
    call `register_memory` automatically per unique buffer.

    Dynamic-loading note:
    StorageBackendFactory creates dynamic backends by calling:
        backend_class(storage_config, kwargs)
    so `kwargs` is provided as a dict-like second argument.
    """

    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Optional[dict] = None):
        extra = storage_config.extra_config or {}
        self._prefix = str(extra.get("key_prefix", "hicache"))
        self._client = _PDKPClientWrapper(extra)

    def _k(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        if target_location is None:
            raise ValueError(
                "HiCachePDKPStorage.get requires target_location (CPU pinned tensor)."
            )
        t = target_location
        _require_cpu_tensor(t, name="target_location")
        ok = self._client.get_tensor(self._k(key), t)
        return t if ok else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        if target_locations is None:
            raise ValueError("HiCachePDKPStorage.batch_get requires target_locations.")
        outs: List[torch.Tensor] = list(target_locations)
        ok_list = self._client.batch_get_tensors([self._k(k) for k in keys], outs)
        return [out if ok else None for out, ok in zip(outs, ok_list)]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if value is None:
            raise ValueError("HiCachePDKPStorage.set requires value (CPU pinned tensor).")
        t = value
        _require_cpu_tensor(t, name="value")
        return self._client.put_tensor(self._k(key), t)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        # PDKP SDK bindings currently do not expose batch_put.
        # Do a best-effort loop; consider adding batch_put in PDKP for performance.
        if values is None:
            raise ValueError("HiCachePDKPStorage.batch_set requires values.")
        vals: List[torch.Tensor] = list(values)
        for k, v in zip(keys, vals):
            if not self.set(k, v):
                return False
        return True

    def exists(self, key: str) -> bool:
        return self._client.exists(self._k(key))

