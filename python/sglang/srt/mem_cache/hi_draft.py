from __future__ import annotations

"""HiCache draft KV pool for speculative decoding.

When HiCache load_back remaps target KV to new device indices, the draft
KV is restored to the same indices so speculative decoding reads the
correct prefix KV.  Without this, draft model sees stale/wrong slot indices
after load_back, causing accept_len regression.
"""

import logging
from typing import TYPE_CHECKING, Dict

import torch

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
device_module = get_device_module()


class HiCacheDraftMixin:
    """Hooks a draft KV pool into HiCache write/load/evict paths.

    Used by HiRadixCache and HiMambaRadixCache via multiple inheritance.
    """

    def _init_draft_state(self) -> None:
        self._draft_pool = None  # device pool
        self._draft_host = None  # host pool
        self._draft_io = None  # io backend str
        self._draft_write_stream = None
        self._draft_load_stream = None
        self._draft_load_queue: list[CacheOperation] = []
        self._draft_ack_write: list = []
        self._draft_ack_load: list = []
        self._draft_host_map: Dict[int, torch.Tensor] = {}

    def _reset_draft_state(self) -> None:
        if self._draft_host is None:
            return
        self._draft_host_map.clear()
        self._draft_host.clear()
        self._draft_load_queue.clear()
        self._draft_ack_write.clear()
        self._draft_ack_load.clear()

    def register_draft_kv_pool(
        self,
        draft_device_pool,
        server_args: "ServerArgs",
    ) -> None:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        pool = draft_device_pool
        if isinstance(pool, SWAKVPool):
            pool = pool.full_kv_pool
        if isinstance(pool, HybridLinearKVPool):
            pool = pool.full_kv_pool

        primary = self.cache_controller.mem_pool_host
        kw = dict(
            host_to_device_ratio=primary.size / pool.size,
            host_size=0,
            page_size=self.page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=None,
        )
        if isinstance(pool, MHATokenToKVPool):
            host_pool = MHATokenToKVPoolHost(pool, **kw)
        elif isinstance(pool, NSATokenToKVPool):
            host_pool = NSATokenToKVPoolHost(pool, **kw)
        elif isinstance(pool, MLATokenToKVPool):
            host_pool = MLATokenToKVPoolHost(pool, **kw)
        else:
            raise ValueError(f"Unsupported draft pool type: {type(pool).__name__}")

        self._draft_pool = draft_device_pool
        self._draft_host = host_pool
        self._draft_io = server_args.hicache_io_backend
        self._draft_write_stream = device_module.Stream()
        self._draft_load_stream = device_module.Stream()
        logger.info(
            "HiCache draft KV registered: %s (%d host slots)",
            type(draft_device_pool).__name__,
            host_pool.size,
        )

    def _draft_move_indices(self, host_indices, device_indices):
        """Move transfer indices to the backend-specific execution device."""
        if self._draft_io == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(
                    self._draft_pool.device, non_blocking=True
                )
            return host_indices, device_indices
        if self._draft_io == "direct":
            if self._draft_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            if self._draft_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
        if self._draft_io == "kernel_ascend":
            return host_indices, device_indices.cpu()
        raise ValueError(f"Unsupported io backend: {self._draft_io}")

    def _draft_submit(self, stream, host_indices, device_indices, backup=True):
        """Submit an async device<->host transfer on *stream*, return finish event."""
        start = device_module.Event()
        finish = device_module.Event()
        start.record()
        h, d = self._draft_move_indices(host_indices, device_indices)
        with device_module.stream(stream):
            start.wait(stream)
            if backup:
                self._draft_host.backup_from_device_all_layer(
                    self._draft_pool, h, d, self._draft_io
                )
            else:
                for i in range(self._draft_host.layer_num):
                    self._draft_host.load_to_device_per_layer(
                        self._draft_pool, h, d, i, self._draft_io
                    )
            finish.record()
            for t in (h, d):
                if t.is_cuda:
                    t.record_stream(stream)
        return finish

    # --- hooks called from HiRadixCache / HiMambaRadixCache ---

    def _draft_write_backup(self, node: "TreeNode") -> None:
        if self._draft_host is None:
            return
        host_indices = self._draft_host.alloc(len(node.value))
        if host_indices is None:
            return
        ev = self._draft_submit(
            self._draft_write_stream, host_indices, node.value, backup=True
        )
        self._draft_ack_write.append(ev)
        self._draft_host_map[node.id] = host_indices

    def _draft_load_at(self, node: "TreeNode", device_indices: torch.Tensor) -> None:
        host_indices = self._draft_host_map.get(node.id)
        if self._draft_host is not None and host_indices is not None:
            self._draft_load_queue.append(
                CacheOperation(host_indices, device_indices, node.id)
            )

    def _draft_start_loading(self) -> None:
        if not self._draft_load_queue:
            return
        op = CacheOperation.merge_ops(self._draft_load_queue)
        self._draft_load_queue.clear()
        ev = self._draft_submit(
            self._draft_load_stream, op.host_indices, op.device_indices, backup=False
        )
        self._draft_ack_load.append(ev)

    def _draft_poll(self) -> None:
        for q in (self._draft_ack_write, self._draft_ack_load):
            if not q:
                continue
            n = next((i for i, ev in enumerate(q) if not ev.query()), len(q))
            del q[:n]

    def _draft_evict_host(self, node: "TreeNode") -> None:
        host_indices = self._draft_host_map.pop(node.id, None)
        if self._draft_host is not None and host_indices is not None:
            self._draft_host.free(host_indices)

    # --- L3 (storage) hooks ---

    def _draft_write_backup_storage(self, node: "TreeNode") -> None:
        if self._draft_host is None or not getattr(self, "enable_storage", False):
            return
        host_indices = self._draft_host_map.get(node.id)
        if host_indices is None or not node.hash_value:
            return
        keys = [f"d:{h}" for h in node.hash_value]
        data = [
            self._draft_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(keys))
        ]
        self.cache_controller.storage_backend.batch_set(keys, data)

    def _draft_prefetch_after_insert(
        self, last_host_node: "TreeNode", fetched_token_ids
    ) -> None:
        if self._draft_host is None or not getattr(self, "enable_storage", False):
            return
        backend = self.cache_controller.storage_backend
        node, key = last_host_node, fetched_token_ids
        while len(key) > 0:
            child_key = self.get_child_key_fn(key)
            if child_key not in node.children:
                break
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if (
                child.host_value is not None
                and child.hash_value
                and child.id not in self._draft_host_map
            ):
                ns_keys = [f"d:{h}" for h in child.hash_value]
                n_slots = len(ns_keys) * self.page_size
                hi = self._draft_host.alloc(n_slots)
                if hi is not None:
                    dummy = [
                        self._draft_host.get_dummy_flat_data_page() for _ in ns_keys
                    ]
                    pages = backend.batch_get(ns_keys, dummy)
                    if pages and all(p is not None for p in pages):
                        for i, p in enumerate(pages):
                            self._draft_host.set_from_flat_data_page(
                                hi[i * self.page_size], p
                            )
                        self._draft_host_map[child.id] = hi
                    else:
                        self._draft_host.free(hi)
            key = key[prefix_len:]
            node = child
