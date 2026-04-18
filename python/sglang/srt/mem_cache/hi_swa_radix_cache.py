from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Hierarchical radix cache for hybrid SWA (Sliding Window Attention) models.

Extends SWARadixCache with 3-tier memory management:
  L1 (GPU) → L2 (Host/CPU) → L3 (Storage)

Both full-attention KV and SWA KV are backed up to host memory.
"""

import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_swa_hybrid_stack,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import (
    LRUList,
    SWARadixCache,
    TreeNode,
    gen_swa_uuid,
    get_last_access_time,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HostSWALRUList(LRUList):
    """LRU list for SWA host data, using host_swa_prev/host_swa_next pointers."""

    def __init__(self):
        super().__init__(is_swa_list=True)
        # Override pointers to use host-specific ones
        self.prv = "host_swa_prev"
        self.nxt = "host_swa_next"
        # Re-initialize dummy head/tail with new pointers
        self.head = TreeNode()
        self.tail = TreeNode()
        setattr(self.head, self.nxt, self.tail)
        setattr(self.tail, self.prv, self.head)
        self.cache = {}

    def reset_node_mru(self, node):
        assert node.id in self.cache, f"Resetting node {node.id=} not in host SWA LRU"
        assert (
            node.swa_host_value is not None
        ), f"Resetting non-backed-up node in host SWA LRU: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def insert_mru(self, node):
        assert (
            node.swa_host_value is not None
        ), f"Inserting non-backed-up node in host SWA LRU: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in host SWA LRU"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        assert node.id in self.cache, f"Removing node {node.id=} not in host SWA LRU"
        del self.cache[node.id]
        self._remove_node(node)


class HiSWARadixCache(SWARadixCache):
    """Hierarchical cache for hybrid SWA models."""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.swa_kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(self.swa_kv_cache, SWAKVPool):
            raise ValueError(
                "HiSWARadixCache requires SWAKVPool for hybrid SWA models."
            )
        assert isinstance(
            params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), "HiSWARadixCache requires SWATokenToKVPoolAllocator."

        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )

        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        build_swa_hybrid_stack(
            self,
            params,
            server_args,
            extra_config=extra_config,
            prefetch_threshold=prefetch_threshold,
            load_cache_event=self.load_cache_event,
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}

        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        self.evictable_full_device_leaves: set[TreeNode] = set()
        self.evictable_full_host_leaves: set[TreeNode] = set()
        self.swa_host_lru_list = HostSWALRUList()

        atexit.register(self.shutdown)

        super().__init__(params=params)

    # ---- Reset ----

    def reset(self) -> None:
        TreeNode.counter = 0
        self._flush_pending_storage_backups_before_reset()
        self.cache_controller.reset()
        self.full_kv_pool_host.clear()
        self.swa_kv_pool_host.clear()
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_full_device_leaves.clear()
        self.evictable_full_host_leaves.clear()
        self.swa_host_lru_list = HostSWALRUList()
        logger.info(
            "HiSWARadixCache reset completed: host_kv_available=%s host_swa_available=%s",
            self.full_kv_pool_host.available_size(),
            self.swa_kv_pool_host.available_size(),
        )
        super().reset()

    # ---- Leaf Status Tracking ----

    def _update_leaf_status(self, node: TreeNode):
        self._update_full_device_leaf_status(node)
        self._update_full_host_leaf_status(node)

    def _update_full_device_leaf_status(self, node: TreeNode):
        if node == self.root_node or node.evicted or node.full_lock_ref > 0:
            self.evictable_full_device_leaves.discard(node)
            return
        for child in node.children.values():
            if not child.evicted:
                self.evictable_full_device_leaves.discard(node)
                return
        self.evictable_full_device_leaves.add(node)

    def _update_full_host_leaf_status(self, node: TreeNode):
        if (
            not node.evicted
            or not node.backuped
            or node == self.root_node
            or node.host_ref_counter > 0
            or node.host_swa_ref_counter > 0
            or len(node.children) > 0
        ):
            self.evictable_full_host_leaves.discard(node)
            return
        self.evictable_full_host_leaves.add(node)

    def _discard_from_leaf_sets(self, node: TreeNode):
        self.evictable_full_device_leaves.discard(node)
        self.evictable_full_host_leaves.discard(node)

    def _detach_from_parent(self, node: TreeNode, allow_missing: bool = False) -> bool:
        """Remove ``node`` from ``node.parent.children``.

        Host/device eviction candidates can stay in auxiliary leaf sets while the
        radix tree is being split by another access path. In that case the
        current ``node.key`` may no longer map to the parent's active child key
        even though ``node`` is still the intended child object. Prefer the fast
        keyed path, then fall back to an identity scan before treating the tree
        as corrupted.
        """
        parent = node.parent
        key = self.get_child_key_fn(node.key)

        if parent is None:
            if allow_missing:
                return False
            raise AssertionError(f"node {node.id} does not have a parent")

        if parent.children.get(key) is node:
            del parent.children[key]
            return True

        for child_key, child in list(parent.children.items()):
            if child is node:
                del parent.children[child_key]
                return True

        if allow_missing:
            return False
        raise AssertionError(f"parent does not have child node {node.id}, {key}")

    # ---- Host Node Protection ----

    def _protect_host_node(self, node: TreeNode, protect_swa: bool = True):
        node.protect_host()
        self.evictable_full_host_leaves.discard(node)
        if protect_swa:
            node.protect_host_swa()
            if self.swa_host_lru_list.in_list(node):
                self.swa_host_lru_list.remove_node(node)

    def _release_host_node(self, node: TreeNode, release_swa: bool = True):
        node.release_host()
        if release_swa:
            node.release_host_swa()
            if node.host_swa_ref_counter == 0 and node.swa_host_value is not None:
                if not self.swa_host_lru_list.in_list(node):
                    self.swa_host_lru_list.insert_mru(node)
        if node.host_ref_counter == 0 and node.host_swa_ref_counter == 0:
            self._update_full_host_leaf_status(node)

    # ---- SWA PoolTransfer Builders ----

    def _get_swa_device_indices(self, node: TreeNode) -> Optional[torch.Tensor]:
        """Get the SWA device indices for a non-tombstoned node."""
        if node.swa_tombstone or node.value is None:
            return None
        return self.token_to_kv_pool_allocator.full_to_swa_index_mapping[node.value]

    def swa_backup_transfers(self, node: TreeNode) -> Optional[list[PoolTransfer]]:
        """Build D→H transfer descriptor for SWA KV."""
        if node.swa_tombstone:
            return None
        swa_device_indices = self._get_swa_device_indices(node)
        if swa_device_indices is None:
            return None
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=node.swa_host_value,
                device_indices=swa_device_indices,
            )
        ]

    def swa_backup_commit(
        self, node: TreeNode, transfers: Optional[list[PoolTransfer]]
    ) -> None:
        """Store auto-allocated SWA host indices after D→H backup."""
        if not transfers:
            return
        host_indices = transfers[0].host_indices
        if node.swa_host_value is None and host_indices is not None:
            node.swa_host_value = host_indices.clone()
            self.swa_host_lru_list.insert_mru(node)

    def swa_restore_transfers(
        self, nodes_to_load: list[TreeNode]
    ) -> Optional[list[PoolTransfer]]:
        """Build H→D transfer descriptors for SWA KV."""
        swa_host_indices = []
        for n in nodes_to_load:
            if n.swa_host_value is not None:
                swa_host_indices.append(n.swa_host_value)
        if not swa_host_indices:
            return None
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.cat(swa_host_indices),
                device_indices=None,  # auto-alloc by controller
            )
        ]

    def swa_restore_commit(
        self,
        nodes_to_load: list[TreeNode],
        new_full_indices: torch.Tensor,
        transfers: Optional[list[PoolTransfer]],
    ) -> None:
        """Write back controller-allocated SWA device indices after H→D restore."""
        if not transfers or transfers[0].device_indices is None:
            return
        swa_device_indices = transfers[0].device_indices
        full_offset = 0
        swa_offset = 0
        for n in nodes_to_load:
            n_len = len(n.host_value)
            if n.swa_host_value is not None:
                swa_count = len(n.swa_host_value)
                n_full = new_full_indices[full_offset : full_offset + n_len]
                n_swa = swa_device_indices[swa_offset : swa_offset + swa_count]
                self.token_to_kv_pool_allocator.full_to_swa_index_mapping[n_full] = (
                    n_swa
                )
                n.swa_tombstone = False
                swa_offset += swa_count
            full_offset += n_len

    def swa_restore_commit_for_nodes(
        self,
        nodes_to_restore: list[TreeNode],
        full_indices_by_node: dict[int, torch.Tensor],
        transfers: Optional[list[PoolTransfer]],
    ) -> None:
        """Commit SWA host->device restores for a mixed suffix path.

        ``nodes_to_restore`` may include nodes whose full KV was restored from host
        as well as nodes whose full KV already stayed on device and only require
        SWA sidecar pages to be rebound.
        """
        if not transfers or transfers[0].device_indices is None:
            return

        swa_device_indices = transfers[0].device_indices
        swa_offset = 0
        for n in nodes_to_restore:
            if n.swa_host_value is None:
                continue
            swa_count = len(n.swa_host_value)
            full_indices = full_indices_by_node.get(n.id)
            assert full_indices is not None, f"missing full indices for node {n.id}"
            n_swa = swa_device_indices[swa_offset : swa_offset + swa_count]
            self.token_to_kv_pool_allocator.full_to_swa_index_mapping[full_indices] = (
                n_swa
            )
            n.swa_tombstone = False
            swa_offset += swa_count

    def swa_archive_transfers(
        self, node: TreeNode
    ) -> Optional[list[PoolTransfer]]:
        """Build H→Storage transfer descriptor for SWA KV."""
        if node.swa_host_value is None or not node.hash_value:
            return None
        trailing_tokens = min(len(node.swa_host_value), self.sliding_window_size)
        trailing_tokens = trailing_tokens - (trailing_tokens % self.page_size)
        if trailing_tokens == 0:
            return None
        trailing_pages = trailing_tokens // self.page_size
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=node.swa_host_value[-trailing_tokens:],
                keys=node.hash_value[-trailing_pages:],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def swa_prefetch_alloc(
        self,
        token_ids: List[int],
        last_hash: Optional[str],
    ) -> Optional[list[PoolTransfer]]:
        """Allocate a SWA host slot and build Storage→H transfer descriptor."""
        trailing_tokens = min(len(token_ids), self.sliding_window_size)
        trailing_tokens = trailing_tokens - (trailing_tokens % self.page_size)
        if trailing_tokens == 0:
            return None
        host_indices = self._alloc_with_evict(
            self.swa_kv_pool_host, trailing_tokens, self.evict_swa_host
        )
        if host_indices is None:
            return None
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=host_indices,
                keys=["__placeholder__"] * (trailing_tokens // self.page_size),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def prefetch_abort(self, pool_transfers: Optional[list[PoolTransfer]]) -> None:
        """Free any allocated SWA host slots on prefetch abort/revoke."""
        for transfer in pool_transfers or []:
            if transfer.name == PoolName.SWA:
                if transfer.host_indices is not None:
                    self.swa_kv_pool_host.free(transfer.host_indices)
                break

    # ---- Write Backup (D→H) ----

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        # If SWA host slot already exists, refresh its LRU position.
        if not node.swa_tombstone and node.swa_host_value is not None:
            if self.swa_host_lru_list.in_list(node):
                self.swa_host_lru_list.reset_node_mru(node)

        extra_pools = self.swa_backup_transfers(node)
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            extra_pools=extra_pools,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                extra_pools=extra_pools,
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            if extra_pools is not None:
                self.swa_backup_commit(node, extra_pools)
            assert len(node.host_value) > 0
            lock_result = None
            if not write_back:
                lock_result = self.inc_lock_ref(node)
            self.ongoing_write_through[node.id] = (
                node,
                DecLockRefParams(
                    swa_uuid_for_lock=lock_result.swa_uuid_for_lock
                )
                if lock_result is not None
                else None,
            )
        else:
            return 0

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        extra_pools = self.swa_archive_transfers(node)
        operation_id = self.cache_controller.write_storage(
            node.host_value,
            node.key,
            node.hash_value,
            prefix_keys,
            extra_pools=extra_pools,
        )
        swa_host_protected = extra_pools is not None
        self.ongoing_backup[operation_id] = (node, swa_host_protected)
        self._protect_host_node(node, protect_swa=swa_host_protected)

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped and node.hit_count >= self.write_through_threshold:
            self.write_backup(node)

    # ---- Load Back (H→D) ----

    def _collect_recoverable_suffix_nodes(
        self, anchor_node: TreeNode, last_hit_node: TreeNode
    ) -> list[TreeNode]:
        """Collect the matched suffix beyond ``anchor_node`` up to ``last_hit_node``."""
        if last_hit_node == anchor_node:
            return []

        suffix_nodes = []
        node = last_hit_node
        while node != self.root_node and node != anchor_node:
            suffix_nodes.insert(0, node)
            node = node.parent

        if node != anchor_node:
            return []

        return suffix_nodes

    def load_back(
        self,
        node: TreeNode,
        mem_quota: Optional[int] = None,
        anchor_node: Optional[TreeNode] = None,
    ) -> Optional[torch.Tensor]:
        last_hit_node = node
        if anchor_node is None:
            anchor_node = node
            while anchor_node.evicted:
                anchor_node = anchor_node.parent

        suffix_nodes = self._collect_recoverable_suffix_nodes(anchor_node, last_hit_node)
        if not suffix_nodes and last_hit_node.evicted:
            suffix_nodes = [last_hit_node]

        nodes_to_restore_swa = [
            n for n in suffix_nodes if n.swa_tombstone and n.swa_host_value is not None
        ]
        evicted_nodes = [n for n in suffix_nodes if n.evicted]

        result = self.inc_lock_ref(anchor_node)
        delta = result.delta

        if evicted_nodes:
            full_host_indices = torch.cat([n.host_value for n in evicted_nodes])
        else:
            full_host_indices = torch.empty((0,), dtype=torch.int64, device="cpu")

        if len(full_host_indices) > 0 and (
            len(full_host_indices) < self.load_back_threshold
            or (
                len(full_host_indices) > mem_quota + delta
                if mem_quota is not None
                else False
            )
        ):
            self.dec_lock_ref(anchor_node)
            return None

        swa_pools = self.swa_restore_transfers(nodes_to_restore_swa)
        full_device_indices = self.cache_controller.load(
            host_indices=full_host_indices,
            node_id=last_hit_node.id,
            extra_pools=swa_pools,
        )
        if full_device_indices is None:
            self.evict(
                EvictParams(
                    num_tokens=len(full_host_indices),
                    swa_num_tokens=sum(len(n.swa_host_value) for n in nodes_to_restore_swa),
                )
            )
            swa_pools = self.swa_restore_transfers(nodes_to_restore_swa)
            full_device_indices = self.cache_controller.load(
                host_indices=full_host_indices,
                node_id=last_hit_node.id,
                extra_pools=swa_pools,
            )
        self.dec_lock_ref(anchor_node)
        if full_device_indices is None:
            return None

        full_indices_by_node: dict[int, torch.Tensor] = {}
        offset = 0
        for n in evicted_nodes:
            n_len = len(n.host_value)
            restored_full = full_device_indices[offset : offset + n_len].clone()
            n.value = restored_full
            full_indices_by_node[n.id] = restored_full
            offset += n_len
            self.full_lru_list.insert_mru(n)
            self.full_evictable_size_ += n_len
            self._discard_from_leaf_sets(n)
            self._update_leaf_status(n)

        for n in suffix_nodes:
            if not n.evicted:
                full_indices_by_node[n.id] = n.value.clone()

        self.swa_restore_commit_for_nodes(
            nodes_to_restore_swa, full_indices_by_node, swa_pools
        )

        for n in suffix_nodes:
            if n.swa_tombstone:
                # Host-only tombstone segments restore full KV only. If the
                # primary device allocator implicitly paired SWA pages with
                # these full indices, drop those mappings immediately.
                self.token_to_kv_pool_allocator.free_swa(full_indices_by_node[n.id])
            elif not n.evicted and not self.swa_lru_list.in_list(n):
                self.swa_lru_list.insert_mru(n)
                self.swa_evictable_size_ += len(n.value)
            elif n.evicted:
                self.swa_lru_list.insert_mru(n)
                self.swa_evictable_size_ += len(n.value)
            self._update_leaf_status(n)

        self._update_leaf_status(anchor_node)

        load_lock_result = self.inc_lock_ref(last_hit_node)
        self.ongoing_load_back[last_hit_node.id] = (
            last_hit_node,
            DecLockRefParams(
                swa_uuid_for_lock=load_lock_result.swa_uuid_for_lock
            )
            if load_lock_result.swa_uuid_for_lock is not None
            else None,
        )

        if not suffix_nodes:
            return full_device_indices

        return torch.cat([full_indices_by_node[n.id] for n in suffix_nodes])

    def init_load_back(self, params: InitLoadBackParams):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        anchor_node = params.req.last_node if params.req is not None else None
        suffix_nodes = []
        if anchor_node is not None:
            suffix_nodes = self._collect_recoverable_suffix_nodes(anchor_node, last_node)

        if last_node.evicted or any(n.swa_tombstone for n in suffix_nodes):
            loading_values = self.load_back(
                last_node,
                mem_quota,
                anchor_node=anchor_node,
            )
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    # ---- Eviction ----

    def _free_device_swa(self, node: TreeNode):
        """Free SWA device KV for a node (tombstone it)."""
        if node.swa_tombstone:
            return
        self.token_to_kv_pool_allocator.free_swa(node.value)
        if self.swa_lru_list.in_list(node):
            self.swa_evictable_size_ -= len(node.value)
            self.swa_lru_list.remove_node(node)
        elif node.swa_lock_ref > 0:
            self.swa_protected_size_ -= len(node.value)
            node.swa_lock_ref = 0
        node.swa_tombstone = True

    def _evict_to_host(self, node: TreeNode) -> Tuple[int, int]:
        """GPU→CPU demotion: node stays in tree as evicted+backuped."""
        assert not node.evicted, f"already evicted, {node.id=}"
        assert node.backuped, f"not backuped, {node.id=}"

        num_full = len(node.value)

        # Free SWA device KV first
        swa_num = 0
        if not node.swa_tombstone:
            swa_num = len(node.value)
            self._free_device_swa(node)

        # Free full device KV
        self.cache_controller.evict_device(node.value)
        self.full_evictable_size_ -= num_full
        if self.full_lru_list.in_list(node):
            self.full_lru_list.remove_node(node)

        node.value = None
        self._update_leaf_status(node)
        self._update_full_device_leaf_status(node.parent)
        return num_full, swa_num

    def _evict_regular(self, node: TreeNode) -> Tuple[int, int]:
        """Evict a non-backuped device leaf: free GPU KV + SWA, delete from tree."""
        assert not node.evicted, f"already evicted, {node.id=}"
        assert not node.backuped, f"backuped node, {node.id=}"
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        full_num_evicted = len(node.value)
        swa_num_evicted = 0 if node.swa_tombstone else len(node.value)

        # Free both full and SWA device KV
        self.token_to_kv_pool_allocator.free(node.value)
        self.full_evictable_size_ -= full_num_evicted
        if self.full_lru_list.in_list(node):
            self.full_lru_list.remove_node(node)
        if not node.swa_tombstone and self.swa_lru_list.in_list(node):
            self.swa_evictable_size_ -= swa_num_evicted
            self.swa_lru_list.remove_node(node)

        # Free host SWA data if any
        if node.swa_host_value is not None:
            if self.swa_host_lru_list.in_list(node):
                self.swa_host_lru_list.remove_node(node)
            self.swa_kv_pool_host.free(node.swa_host_value)
            node.swa_host_value = None

        node.value = None
        self._discard_from_leaf_sets(node)

        parent = node.parent
        self._detach_from_parent(node)

        self._update_leaf_status(parent)
        _, cascade_full, cascade_swa = self._iteratively_delete_tombstone_leaf(node)
        return full_num_evicted + cascade_full, swa_num_evicted + cascade_swa

    def _evict_device_leaf(self, x: TreeNode) -> Tuple[int, int]:
        """Evict a device leaf node, choosing the right strategy."""
        if not x.backuped:
            if self.cache_controller.write_policy == "write_back":
                self.write_backup(x, write_back=True)
                self.writing_check(write_back=True)
                return self._evict_to_host(x)
            else:
                return self._evict_regular(x)
        return self._evict_to_host(x)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_tokens = params.num_tokens
        swa_num_tokens = params.swa_num_tokens
        full_num_evicted = 0
        swa_num_evicted = 0

        # Full eviction (from device)
        if full_num_tokens > 0:
            leaves = list(self.evictable_full_device_leaves)
            eviction_heap = [(n.last_access_time, n) for n in leaves]
            heapq.heapify(eviction_heap)

            while full_num_evicted < full_num_tokens and eviction_heap:
                _, x = heapq.heappop(eviction_heap)
                if x not in self.evictable_full_device_leaves:
                    continue

                evicted_full, evicted_swa = self._evict_device_leaf(x)
                full_num_evicted += evicted_full
                swa_num_evicted += evicted_swa

                parent = x.parent
                if parent in self.evictable_full_device_leaves:
                    heapq.heappush(
                        eviction_heap, (parent.last_access_time, parent)
                    )

        # SWA eviction (tombstone internal nodes or evict leaves)
        if swa_num_evicted < swa_num_tokens:
            x = self.swa_lru_list.get_lru_no_lock()
            while (
                swa_num_evicted < swa_num_tokens and self.swa_lru_list.in_list(x)
            ):
                assert not x.swa_tombstone, f"tombstone in SWA LRU, {x.id=}"
                assert x != self.root_node, f"root in SWA LRU, {x.id=}"
                assert x.swa_lock_ref == 0, f"locked in SWA LRU, {x.id=}"

                if len(x.children) > 0:
                    # Internal node: tombstone SWA only
                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    swa_num_evicted += len(x.value)
                    self.token_to_kv_pool_allocator.free_swa(x.value)
                    self.swa_lru_list.remove_node(x)
                    self.swa_evictable_size_ -= len(x.value)
                    x.swa_tombstone = True
                else:
                    # Leaf node: full evict
                    assert x.full_lock_ref == 0, (
                        f"leaf full_lock_ref mismatch {x.id=} "
                        f"{x.full_lock_ref=} {x.swa_lock_ref=}"
                    )
                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    _, evicted_swa = self._evict_device_leaf(x)
                    swa_num_evicted += evicted_swa

                    if not self.swa_lru_list.in_list(x_next):
                        x_next = self.swa_lru_list.get_lru_no_lock()

                x = x_next

        return EvictResult(
            num_tokens_evicted=full_num_evicted,
            swa_num_tokens_evicted=swa_num_evicted,
        )

    def evict_host(self, num_tokens: int):
        """Evict host-resident leaf nodes."""
        heap = [(n.last_access_time, n) for n in self.evictable_full_host_leaves]
        heapq.heapify(heap)

        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.evictable_full_host_leaves:
                continue
            num_evicted += self._evict_host_leaf(x)
            if x.parent in self.evictable_full_host_leaves:
                heapq.heappush(heap, (x.parent.last_access_time, x.parent))

    def evict_swa_host(self, num_swa_hosts: int) -> int:
        """Evict SWA host data."""
        if self.disable or num_swa_hosts <= 0:
            return 0

        x = self.swa_host_lru_list.get_lru_no_lock()
        num_evicted = 0
        while num_evicted < num_swa_hosts and self.swa_host_lru_list.in_list(x):
            x_next = self.swa_host_lru_list.get_prev_no_lock(x)
            if x in self.evictable_full_host_leaves:
                self._evict_host_leaf(x)
                num_evicted += 1
            else:
                # Internal host node: free SWA host only
                assert (
                    x.host_swa_ref_counter == 0
                ), f"evict SWA host: ref_counter != 0, {x.id=}"
                self.swa_host_lru_list.remove_node(x)
                self.swa_kv_pool_host.free(x.swa_host_value)
                x.swa_host_value = None
                num_evicted += 1
            x = x_next
        return num_evicted

    def _evict_host_leaf(self, node: TreeNode) -> int:
        """Evict a host-resident leaf: free host KV + SWA, delete from tree, cascade."""
        assert node.evicted, f"not evicted, {node.id=}"
        assert node.backuped, f"not backuped, {node.id=}"
        assert node.host_ref_counter == 0, f"host KV in use, {node.id=}"
        assert node.host_swa_ref_counter == 0, f"host SWA in use, {node.id=}"

        full_num_evicted = self.cache_controller.evict_host(node.host_value)
        node.host_value = None

        if node.swa_host_value is not None:
            if self.swa_host_lru_list.in_list(node):
                self.swa_host_lru_list.remove_node(node)
            self.swa_kv_pool_host.free(node.swa_host_value)
            node.swa_host_value = None

        self._discard_from_leaf_sets(node)
        parent = node.parent
        attached = self._detach_from_parent(node, allow_missing=True)

        if attached:
            self._update_leaf_status(parent)
            _, cascade_full, _ = self._iteratively_delete_tombstone_leaf(node)
            return full_num_evicted + cascade_full

        # This host leaf was already detached from the radix tree, but it can
        # still linger in the host-eviction leaf set. Its host buffers were
        # released above, so treat it as a stale bookkeeping entry instead of
        # crashing the scheduler.
        return full_num_evicted

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        parent = node.parent
        self._detach_from_parent(node)

        self._discard_from_leaf_sets(node)

        # Free SWA host data if any
        if node.swa_host_value is not None:
            if self.swa_host_lru_list.in_list(node):
                self.swa_host_lru_list.remove_node(node)
            self.swa_kv_pool_host.free(node.swa_host_value)
            node.swa_host_value = None

        if (
            node.backuped
            and node.host_ref_counter == 0
            and node.host_swa_ref_counter == 0
        ):
            self.cache_controller.evict_host(node.host_value)
            node.host_value = None

        self._update_leaf_status(parent)

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int, int]:
        full_num_evicted = 0
        swa_num_evicted = 0

        while len(node.parent.children) == 0:
            if node.parent == self.root_node:
                break
            if node.parent.full_lock_ref > 0 or node.parent.swa_lock_ref > 0:
                break

            parent = node.parent

            if not parent.evicted:
                full_num_evicted += len(parent.value)
                swa_evicted = 0 if parent.swa_tombstone else len(parent.value)
                swa_num_evicted += swa_evicted
                # Free device full + SWA
                self.token_to_kv_pool_allocator.free(parent.value)
                self.full_evictable_size_ -= len(parent.value)
                if self.full_lru_list.in_list(parent):
                    self.full_lru_list.remove_node(parent)
                if not parent.swa_tombstone and self.swa_lru_list.in_list(parent):
                    self.swa_evictable_size_ -= swa_evicted
                    self.swa_lru_list.remove_node(parent)

            self._discard_from_leaf_sets(parent)
            self._delete_tombstone_leaf(parent)
            node = parent

        return node, full_num_evicted, swa_num_evicted

    # ---- Lock Ref ----

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()

        swa_lock_size = 0
        swa_uuid_for_lock = None
        delta = 0
        while node != self.root_node:
            if node.evicted:
                node = node.parent
                continue

            # Lock full
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
                delta -= len(node.value)
                self.evictable_full_device_leaves.discard(node)
            node.full_lock_ref += 1

            # Lock SWA within sliding window
            if swa_lock_size < self.sliding_window_size and not node.swa_tombstone:
                if node.swa_lock_ref == 0:
                    self.swa_evictable_size_ -= len(node.value)
                    self.swa_protected_size_ += len(node.value)
                node.swa_lock_ref += 1
                swa_lock_size += len(node.value)
                if swa_lock_size >= self.sliding_window_size:
                    if node.swa_uuid is None:
                        node.swa_uuid = gen_swa_uuid()
                    swa_uuid_for_lock = node.swa_uuid

            node = node.parent
        return IncLockRefResult(delta=delta, swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None

        if self.disable:
            return DecLockRefResult()

        dec_lock_swa = True
        delta = 0
        while node != self.root_node:
            if node.evicted:
                node = node.parent
                continue

            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
                delta += len(node.value)
            node.full_lock_ref -= 1
            if node.full_lock_ref == 0:
                self._update_full_device_leaf_status(node)

            if dec_lock_swa and not node.swa_tombstone:
                if node.swa_lock_ref == 0:
                    if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                        dec_lock_swa = False
                    node = node.parent
                    continue
                if node.swa_lock_ref == 1:
                    self.swa_evictable_size_ += len(node.value)
                    self.swa_protected_size_ -= len(node.value)
                node.swa_lock_ref -= 1
                if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                    dec_lock_swa = False

            node = node.parent

        return DecLockRefResult(delta=delta)

    # ---- Match Prefix ----

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        value, best_last_node, best_value_len, deepest_host_node = (
            self._match_prefix_helper(key)
        )
        return self._match_post_processor(
            params, value, best_last_node, best_value_len, deepest_host_node
        )

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int, TreeNode]:
        """SWA prefix matching with hierarchical cache awareness.

        Extends the base SWA match to handle evicted-but-backed-up nodes.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value = []
        match_len_since_tombstone = float("inf")
        best_value_len = 0
        best_last_node = node
        deepest_host_node = node

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]

            # Stop at non-backed-up evicted nodes
            if child.evicted and not child.backuped:
                break

            if child.swa_tombstone:
                if match_len_since_tombstone >= self.sliding_window_size:
                    best_value_len = len(value)
                    best_last_node = node
                match_len_since_tombstone = 0

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                if new_node.backuped:
                    deepest_host_node = new_node
                if not new_node.swa_tombstone:
                    match_len_since_tombstone += len(new_node.key)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                if not child.swa_tombstone:
                    match_len_since_tombstone += len(child.key)
                node = child
                if node.backuped:
                    deepest_host_node = node
                key = key[prefix_len:]
                if len(key):
                    child_key = self.get_child_key_fn(key)

        if match_len_since_tombstone >= self.sliding_window_size:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len, deepest_host_node

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        best_last_node: TreeNode,
        best_value_len: int,
        deepest_host_node: TreeNode,
    ) -> MatchResult:
        access_node = (
            deepest_host_node if deepest_host_node != self.root_node else best_last_node
        )
        # LRU updates: skip evicted nodes for full_lru_list
        lru_node = access_node
        while lru_node != self.root_node and lru_node.evicted:
            lru_node = lru_node.parent
        self.full_lru_list.reset_node_and_parents_mru(lru_node, self.root_node)
        self.swa_lru_list.reset_node_and_parents_mru(lru_node, self.root_node)

        cur_time = get_last_access_time()
        node_update = access_node
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        # Keep last_device_node aligned with device_indices. deepest_host_node can
        # point at a backuped node whose SWA window is no longer reusable, in
        # which case best_value_len may be 0 even though deepest_host_node is
        # non-root. Using deepest_host_node here makes the scheduler lock a node
        # that contributes no reusable device prefix, which can incorrectly turn
        # an otherwise-evictable cache state into NO_TOKEN during admission.
        kv_host_hit_length = 0
        last_device_node = best_last_node
        last_host_node = deepest_host_node
        while last_host_node != self.root_node and not last_host_node.backuped:
            last_host_node = last_host_node.parent

        recover_node = last_host_node
        while recover_node != self.root_node and recover_node != best_last_node:
            if recover_node.evicted or (
                recover_node.swa_tombstone and recover_node.swa_host_value is not None
            ):
                kv_host_hit_length += len(recover_node.host_value)
            recover_node = recover_node.parent

        while last_device_node != self.root_node and last_device_node.evicted:
            last_device_node = last_device_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_device_node,
            last_host_node=last_host_node,
            host_hit_length=kv_host_hit_length,
        )

    # ---- Split Node ----

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        if child.evicted:
            return self._split_evicted_node(key, child, split_len)

        self.evictable_full_device_leaves.discard(child)

        new_node = super()._split_node(key, child, split_len)

        # Split host values
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        if child.swa_backuped:
            new_node.swa_host_value = child.swa_host_value[:split_len].clone()
            child.swa_host_value = child.swa_host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        self._update_leaf_status(new_node)
        self._update_leaf_status(child)

        return new_node

    def _split_evicted_node(
        self, key: RadixKey, child: TreeNode, split_len: int
    ) -> TreeNode:
        self.evictable_full_host_leaves.discard(child)

        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.value = None  # evicted
        new_node.swa_tombstone = child.swa_tombstone
        new_node.full_lock_ref = child.full_lock_ref
        new_node.swa_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.swa_uuid = child.swa_uuid
        child.swa_uuid = None

        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        if child.swa_backuped:
            new_node.swa_host_value = child.swa_host_value[:split_len].clone()
            child.swa_host_value = child.swa_host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        child.last_access_time = get_last_access_time()
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._update_full_host_leaf_status(new_node)
        self._update_full_host_leaf_status(child)

        return new_node

    # ---- Insert ----

    def cache_unfinished_req(self, req: Req, chunked: bool = False) -> None:
        if chunked or req.is_chunked > 0:
            # HiSWA tombstone/storage recovery can intentionally keep request-side
            # full KV outside the radix tree until the whole prompt finishes.
            # Re-inserting intermediate chunks would free still-in-use request
            # pages, and those freed pages can then be re-counted or re-used
            # before the request completes.
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        super().cache_unfinished_req(req, chunked=chunked)

    def _unevict_node(self, node: TreeNode, fresh_value: torch.Tensor):
        """Restore a previously evicted node with fresh device KV indices."""
        assert node.evicted, f"not evicted, {node.id=}"
        if node.swa_tombstone:
            # Unevicting a tombstoned node should only restore its full KV.
            # The request-side allocation still owns paired SWA pages, so clear
            # those mappings immediately instead of leaving orphaned SWA slots.
            self.token_to_kv_pool_allocator.free_swa(fresh_value)
        node.value = fresh_value.clone()
        self.full_lru_list.insert_mru(node)
        self.full_evictable_size_ += len(fresh_value)
        if not node.swa_tombstone:
            self.swa_lru_list.insert_mru(node)
            self.swa_evictable_size_ += len(fresh_value)
        self._update_leaf_status(node)
        if node.parent is not None:
            self._update_leaf_status(node.parent)

    def _free_request_slice_not_owned_by_subtree(
        self, node: TreeNode, request_slice: torch.Tensor, include_self: bool = True
    ) -> None:
        if request_slice.numel() == 0:
            return

        live_values = []
        if include_self and not node.evicted:
            live_values.append(node.value)

        stack = list(node.children.values())
        while stack:
            cur = stack.pop()
            if not cur.evicted:
                live_values.append(cur.value)
            stack.extend(cur.children.values())

        if not live_values:
            self.token_to_kv_pool_allocator.free(request_slice)
            return

        live_indices = torch.cat(live_values)
        keep_mask = torch.isin(request_slice, live_indices)
        to_free = request_slice[~keep_mask]
        self.token_to_kv_pool_allocator.free(to_free)

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        update_kv_after_len: int,
        swa_evicted_seqlen: int = 0,
    ) -> int:
        matched_prefix_length = 0
        cursor_length = 0
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            if not node.evicted:
                self.full_lru_list.reset_node_mru(node)
                if not node.swa_tombstone:
                    self.swa_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()

            if not node.evicted:
                self.full_lru_list.reset_node_mru(node)
                if not node.swa_tombstone:
                    self.swa_lru_list.reset_node_mru(node)

            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if node.evicted:
                # Un-evict with fresh value
                self._unevict_node(node, value[:prefix_len])
            elif update_kv_after_len < cursor_length + prefix_len:
                # Original SWA tombstone recovery logic
                if node.swa_tombstone:
                    assert (
                        node.swa_lock_ref == 0
                    ), f"tombstone swa_lock_ref should be 0, {node.id=}"
                    assert (
                        swa_evicted_seqlen % self.page_size == 0
                    ), f"swa_evicted_seqlen must be page aligned"
                    if len(node.children) > 0:
                        # In HiSWA a host load-back can restore a child suffix
                        # before this ancestor is revisited. Rebinding the
                        # internal tombstone to the request's fresh indices would
                        # alias the child pages and corrupt accounting, so keep
                        # the ancestor tombstoned and discard the request-side
                        # KV that is not already owned by a live descendant.
                        self._free_request_slice_not_owned_by_subtree(
                            node, value[:prefix_len]
                        )
                    elif swa_evicted_seqlen <= cursor_length:
                        if len(node.children) == 0:
                            self.token_to_kv_pool_allocator.free(node.value[:prefix_len])
                            node.value = value[:prefix_len].clone()
                            node.swa_tombstone = False
                            self.swa_lru_list.insert_mru(node)
                            self.swa_evictable_size_ += len(node.value)
                    elif swa_evicted_seqlen < cursor_length + prefix_len:
                        if len(node.children) == 0:
                            start_update_idx = swa_evicted_seqlen - cursor_length
                            self.token_to_kv_pool_allocator.free(
                                node.value[start_update_idx:prefix_len]
                            )
                            self._split_node(node.key, node, start_update_idx)
                            node.value = value[start_update_idx:prefix_len].clone()
                            self.token_to_kv_pool_allocator.free(
                                value[:start_update_idx]
                            )
                            node.swa_tombstone = False
                            self.swa_lru_list.insert_mru(node)
                            self.swa_evictable_size_ += len(node.value)
                    else:
                        self.token_to_kv_pool_allocator.free(value[:prefix_len])
                else:
                    if swa_evicted_seqlen <= cursor_length:
                        self._free_request_slice_not_owned_by_subtree(
                            node, value[:prefix_len]
                        )
                    elif swa_evicted_seqlen < cursor_length + prefix_len:
                        split_idx = swa_evicted_seqlen - cursor_length
                        tombstone_node = self._split_node(node.key, node, split_idx)
                        self._free_device_swa(tombstone_node)
                        self._free_request_slice_not_owned_by_subtree(
                            node, value[:prefix_len]
                        )
                    else:
                        self._free_device_swa(node)
                        self._free_request_slice_not_owned_by_subtree(
                            node, value[:prefix_len]
                        )
            else:
                matched_prefix_length += prefix_len
                self._inc_hit_count(node)

            cursor_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            if swa_evicted_seqlen == cursor_length + len(key):
                self.token_to_kv_pool_allocator.free(value)
                return matched_prefix_length

            if (
                swa_evicted_seqlen > cursor_length
                and swa_evicted_seqlen < cursor_length + len(key)
            ):
                swa_tombstone_len = swa_evicted_seqlen - cursor_length
                node = self._add_new_node(
                    node,
                    key[:swa_tombstone_len],
                    value[:swa_tombstone_len],
                    swa_tombstone=True,
                )
                key = key[swa_tombstone_len:]
                value = value[swa_tombstone_len:]

            self._add_new_node(node, key, value, swa_tombstone=False)
        return matched_prefix_length

    def _add_new_node(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        swa_tombstone: bool = False,
    ) -> TreeNode:
        assert len(key) > 0, "key should not be empty"
        new_node = TreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        new_node.swa_tombstone = swa_tombstone
        parent.children[self.get_child_key_fn(key)] = new_node
        self.full_lru_list.insert_mru(new_node)
        self.full_evictable_size_ += len(value)
        if not swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_evictable_size_ += len(value)

        # Compute hash_value for storage
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

        self._update_full_device_leaf_status(new_node)
        self._update_full_device_leaf_status(parent)

        if self.cache_controller.write_policy != "write_back":
            self._inc_hit_count(new_node)

        return new_node

    # ---- HiCache Event Handling ----

    def writing_check(self, write_back=False):
        if write_back:
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node, _ = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1

        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())

        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node, lock_params = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node, lock_params)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                break
            finish_count += 1
            for ack_id in ack_list:
                end_node, lock_params = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node, lock_params)

        del self.cache_controller.ack_load_queue[:finish_count]

    def ready_to_load_host_cache(self) -> int:
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    # ---- Storage (L3) Support ----

    def shutdown(self):
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        storage_metrics_collector = None
        if enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            storage_metrics_collector = StorageMetricsCollector(labels=labels)

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics
        if self.enable_storage_metrics:
            self.storage_metrics_collector = storage_metrics_collector
        else:
            self.storage_metrics_collector = None

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (format: {ext})"
                            )
                else:
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: "
                    f"{hicache_storage_prefetch_policy!r}. Expected one of {allowed}.",
                )
        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type
            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                return (
                    True,
                    "HiCache storage already enabled with same backend; policies updated.",
                )
            return (
                False,
                f"HiCache storage already enabled with '{current_backend}'. Detach first.",
            )

        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            return (False, f"Failed to parse storage config: {e}")

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=self.host_pool_group.entries,
            )
        except Exception as e:
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple:
        try:
            self._drain_storage_control_queues_local()
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            return False, f"Failed to detach HiCache storage backend: {e}"

        self._drain_storage_control_queues_local()
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        if hasattr(self, "storage_metrics_collector"):
            self.storage_metrics_collector = None
        return True, "Detached HiCache storage backend successfully."

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info("Hierarchical cache storage backend cleared!")
                    return True
                else:
                    logger.warning(
                        f"Storage backend "
                        f"{type(self.cache_controller.storage_backend).__name__} "
                        "does not support clear."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear storage backend: {e}")
                return False
        else:
            logger.warning("Storage backend is not enabled.")
            return False

    def drain_storage_control_queues(self):
        cc = self.cache_controller
        if self.ongoing_prefetch or self.ongoing_backup:
            qsizes = torch.tensor(
                [
                    cc.prefetch_revoke_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                ],
                dtype=torch.int,
            )
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(
                    qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
            n_revoke, n_backup = map(int, qsizes.tolist())
        else:
            n_revoke = 0
            n_backup = 0
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            # Host mem releases are local-only frees queued after storage termination.
            # Draining them without a TP collective avoids mismatched queue-size
            # collectives when scheduler ranks advance at slightly different speeds.
            n_release=None,
            log_metrics=True,
        )

    def _drain_storage_control_queues_local(self):
        self._drain_storage_control_queues_impl(
            n_revoke=None, n_backup=None, n_release=None, log_metrics=False
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, operation = info
                    self.prefetch_abort(operation.pool_transfers)
                    self._release_host_node(last_host_node)
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    node, swa_host_protected = entry
                    self._release_host_node(node, release_swa=swa_host_protected)
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _force_release_pending_storage_ops(self):
        cc = self.cache_controller
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    self.ongoing_prefetch.pop(req_id, None)
                    continue
                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )
                try:
                    self.prefetch_abort(getattr(_operation, "pool_transfers", None))
                except Exception:
                    logger.exception(
                        "Failed to release SWA host indices for prefetch %s", req_id
                    )
                try:
                    self._release_host_node(last_host_node)
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )
                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass
                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        try:
            for ack_id, entry in list(self.ongoing_backup.items()):
                try:
                    node, swa_host_protected = entry
                    self._release_host_node(node, release_swa=swa_host_protected)
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True
        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return
        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        self._protect_host_node(last_host_node, protect_swa=False)

        # Allocate host KV memory
        host_indices = self._alloc_with_evict(
            self.cache_controller.mem_pool_host,
            prefetch_length,
            self.evict_host,
        )
        if host_indices is None:
            self._release_host_node(last_host_node, release_swa=False)
            return

        # Allocate host SWA slot
        extra_pools = self.swa_prefetch_alloc(new_input_tokens, last_hash)
        if extra_pools is None:
            self.cache_controller.mem_pool_host.free(host_indices)
            self._release_host_node(last_host_node, release_swa=False)
            return

        # SWA is also being loaded, protect host SWA as well
        last_host_node.protect_host_swa()
        if self.swa_host_lru_list.in_list(last_host_node):
            self.swa_host_lru_list.remove_node(last_host_node)

        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
            extra_pools=extra_pools,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]
        if operation.host_indices is None:
            return True
        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()

        swa_host_indices = None
        swa_loaded = False
        for transfer in operation.pool_transfers or []:
            if transfer.name == PoolName.SWA:
                swa_host_indices = transfer.host_indices
                swa_loaded = (
                    operation.pool_storage_result.extra_pool_hit_pages.get(
                        PoolName.SWA, 0
                    )
                    >= 1
                )
                break

        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids,
                extra_key=last_host_node.key.extra_key if hasattr(last_host_node.key, 'extra_key') else None,
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            swa_host_indices,
            swa_loaded,
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )

        # Free SWA host slot if it wasn't inserted into the tree
        if swa_host_indices is not None:
            inserted_new = matched_length < min_completed_tokens
            if not inserted_new or not swa_loaded:
                self.swa_kv_pool_host.free(swa_host_indices)

        self._release_host_node(last_host_node)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def _insert_helper_host(
        self,
        node: TreeNode,
        key: RadixKey,
        host_value,
        hash_value,
        swa_host_value: Optional[torch.Tensor] = None,
        swa_loaded: bool = False,
    ):
        node.last_access_time = get_last_access_time()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            prefix_len = self.key_match_fn(node.key, key)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        leaf_node: Optional[TreeNode] = None
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = None  # evicted (host-only)
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            # Host-only nodes start as SWA tombstones until trailing SWA host
            # data is attached to a suffix-aligned leaf segment.
            new_node.swa_tombstone = True
            node.children[child_key] = new_node
            leaf_node = new_node
            self._update_full_host_leaf_status(new_node)
            self._update_full_host_leaf_status(node)

        # Attach SWA host data to the new leaf
        if leaf_node is not None and swa_host_value is not None and swa_loaded:
            full_len = len(leaf_node.host_value)
            swa_len = len(swa_host_value)
            if swa_len == full_len:
                leaf_node.swa_host_value = swa_host_value.clone()
                leaf_node.swa_tombstone = False
            else:
                assert (
                    0 < swa_len < full_len
                ), f"invalid partial SWA host span: {swa_len=} {full_len=}"
                assert (
                    swa_len % self.page_size == 0
                ), f"partial SWA host span must be page aligned: {swa_len=}"
                split_len = full_len - swa_len
                assert (
                    split_len % self.page_size == 0
                ), f"prefix host span must be page aligned: {split_len=}"
                # Storage only preserves trailing SWA pages. Represent that as
                # a tombstoned prefix plus a suffix leaf whose full/SWA host
                # spans have matching lengths.
                self._split_evicted_node(leaf_node.key, leaf_node, split_len)
                leaf_node.swa_host_value = swa_host_value.clone()
                leaf_node.swa_tombstone = False
            if not self.swa_host_lru_list.in_list(leaf_node):
                self.swa_host_lru_list.insert_mru(leaf_node)

        return matched_length

    def release_aborted_request(self, rid: str):
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        self._release_host_node(last_host_node)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.prefetch_abort(operation.pool_transfers)
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

    def _flush_pending_storage_backups_before_reset(self) -> None:
        if not self.enable_storage:
            return
        self.writing_check(write_back=True)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self.drain_storage_control_queues()
            backup_qsize = self.cache_controller.backup_queue.qsize()
            ack_backup_qsize = self.cache_controller.ack_backup_queue.qsize()
            ongoing_backup = len(self.ongoing_backup)
            ongoing_write = len(self.ongoing_write_through)
            if (
                backup_qsize == 0
                and ack_backup_qsize == 0
                and ongoing_backup == 0
                and ongoing_write == 0
            ):
                return
            time.sleep(0.05)
        logger.warning(
            "Timed out waiting for HiCache storage backups to drain before reset: "
            "ongoing_write=%s ongoing_backup=%s",
            len(self.ongoing_write_through),
            len(self.ongoing_backup),
        )

    def _alloc_with_evict(
        self,
        pool,
        size: int,
        evict_fn,
        lock_node: Optional[TreeNode] = None,
        error_message: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        indices = pool.alloc(size)
        if indices is None:
            if lock_node is not None:
                self.inc_lock_ref(lock_node)
            evict_fn(size)
            indices = pool.alloc(size)
            if lock_node is not None:
                self.dec_lock_ref(lock_node)
        if indices is None and error_message is not None:
            raise RuntimeError(error_message)
        return indices

    # ---- Misc ----

    def sanity_check(self):
        self.loading_check()
        if self.ongoing_load_back or self.ongoing_write_through:
            return
        super().sanity_check()

    def _collect_all_nodes(self) -> list:
        ret = []
        stack = [self.root_node]
        while stack:
            cur = stack.pop()
            if not cur.evicted:
                ret.append(cur)
            stack.extend(cur.children.values())
        return ret

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: TreeNode):
            for child in node.children.values():
                if not child.evicted:
                    values.append(child.value)
                _dfs(child)

        _dfs(self.root_node)
        return torch.cat(values) if values else torch.tensor([])
