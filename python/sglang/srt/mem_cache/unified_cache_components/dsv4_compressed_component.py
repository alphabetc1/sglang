from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

logger = logging.getLogger(__name__)


class DSV4CompressedComponent(TreeComponent):
    """Compressed-attention component for DeepSeek-V4.

    Owns three device pools (c4 / c4_indexer / c128) via the parent
    DeepSeekV4TokenToKVPool, and three host pools registered as a
    HostPoolGroup with c4 anchor + c4_indexer/c128 share-indices side pools.
    One page-id tensor per node addresses all six.

    For eviction, the component delegates to
    DeepSeekV4TokenToKVPool.free_compressed_pages / alloc_compressed_pages
    (wired to the allocator in T7 via HybridPoolAssembler).

    HiCache hooks (build_hicache_transfers, commit_hicache_transfer,
    drive_host_eviction) are stubs — Task 4 fills them in.
    """

    component_type = ComponentType.DSV4_COMPRESSED

    def __init__(self, cache, params):
        super().__init__(cache, params)
        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        self._dsv4_pool: "DeepSeekV4TokenToKVPool" = kvcache
        # Host pools are bound by hybrid_pool_assembler (T7); start as None.
        self._c4_pool_host = None
        self._c4_indexer_pool_host = None
        self._c128_pool_host = None

    # ---- match validators ----

    def create_match_validator(self) -> Callable[["UnifiedTreeNode"], bool]:
        """Node alive iff this component's device value is present OR FULL has
        a host backup (host backup is owned by FULL on the c4 anchor pool;
        DSV4_COMPRESSED is a shadow of FULL for tree bookkeeping)."""
        ct = self.component_type
        base_ct = BASE_COMPONENT_TYPE
        return lambda node: (
            node.component_data[ct].value is not None
            or node.component_data[base_ct].host_value is not None
        )

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """Walk last_host_node → last_device_node summing host_value lengths."""
        ct = self.component_type
        host_hit = 0
        node = result.last_host_node
        root = self.cache.root_node
        while node is not result.last_device_node and node is not root:
            cd = node.component_data[ct]
            if cd.host_value is not None:
                host_hit += len(cd.host_value)
            node = node.parent
        if host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, host_hit)
            )
        return result

    # ---- node split ----

    def redistribute_on_node_split(
        self, new_parent: "UnifiedTreeNode", child: "UnifiedTreeNode"
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        cd = child.component_data[ct]
        split_len = len(new_parent.key)
        if cd.value is not None:
            new_parent.component_data[ct].value = cd.value[:split_len].clone()
            cd.value = cd.value[split_len:].clone()
        if cd.host_value is not None:
            new_parent.component_data[ct].host_value = cd.host_value[:split_len].clone()
            cd.host_value = cd.host_value[split_len:].clone()

    # ---- insert helpers ----

    def update_component_on_insert_overlap(
        self,
        node: "UnifiedTreeNode",
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        """Consume the full prefix overlap; no DSV4-specific duplicate freeing needed."""
        return prefix_len

    def recover_after_unevict(
        self,
        node: "UnifiedTreeNode",
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        """No SWA-style tombstones; host_value remains attached as-is."""
        return None

    def commit_insert_component_data(
        self,
        node: "UnifiedTreeNode",
        is_new_leaf: bool,
        params: InsertParams,
        result,
    ) -> None:
        """Mirror FULL's per-node value. DSV4 compressed pools (c4 / c4_indexer
        / c128) share the same page-id namespace as FULL's value, so we just
        clone it. Insert into the device LRU and bump evictable size so the
        node is visible to drive_eviction.
        """
        if not is_new_leaf:
            return
        ct = self.component_type
        cd_full = node.component_data[BASE_COMPONENT_TYPE]
        cd_self = node.component_data[ct]
        if cd_full.value is None or cd_self.value is not None:
            return
        cd_self.value = cd_full.value.clone()
        self.cache.lru_lists[ct].insert_mru(node)
        self.cache.component_evictable_size_[ct] += len(cd_self.value)

    # ---- eviction (DEVICE) ----

    def evict_component(
        self,
        node: "UnifiedTreeNode",
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """Bookkeeping-only eviction for the DSV4_COMPRESSED shadow.

        FULL is the authoritative owner of the page-id allocation: it calls
        full_attn_allocator.free on EvictLayer.DEVICE and c4-host-pool.free on
        EvictLayer.HOST. DSV4_COMPRESSED shares the same page-id namespace, so
        calling free here would be a double-free. We just clear our cd.value
        and decrement our component_evictable_size_ counter so the tree stays
        balanced; FULL's evict path does the actual release.
        """
        cd = node.component_data[self.component_type]
        device_freed = host_freed = 0
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= device_freed
            cd.value = None
        # HOST: DSV4_COMPRESSED does not track its own host_value (FULL does);
        # nothing to do here.
        return device_freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        # Same priority as FULL so FULL's cascade picks us up on every evict.
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        """No-op: FULL drives device eviction (it owns the allocator). The
        cascade in UnifiedRadixCache._cascade_evict will call our
        evict_component to clear cd.value bookkeeping when FULL picks a node.
        """
        return

    # ---- locks ----

    def acquire_component_lock(
        self, node: "UnifiedTreeNode", result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        cd = node.component_data[ct]
        value = cd.value
        if value is not None:
            if cd.lock_ref == 0:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] -= vlen
                self.cache.component_protected_size_[ct] += vlen
            cd.lock_ref += 1
        return result

    def release_component_lock(
        self, node: "UnifiedTreeNode", params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        cd = node.component_data[ct]
        value = cd.value
        if value is not None and cd.lock_ref > 0:
            if cd.lock_ref == 1:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] += vlen
                self.cache.component_protected_size_[ct] -= vlen
            cd.lock_ref -= 1

    # ---- HiCache hooks ----
    #
    # DSV4_COMPRESSED is a "shadow" of FULL: it shares the same page-id
    # namespace, and the c4 anchor + register_hicache_anchor_kv_shared_indices_pool
    # mechanism delivers all backup/load transfers (anchor c4 + shared
    # c4_indexer). FULL.commit_hicache_transfer sets the authoritative
    # cd.host_value on FULL; DSV4_COMPRESSED's match_validator delegates to
    # FULL.host_value, so we don't track our own.
    #
    # All three hooks below are intentional no-ops that just satisfy the ABC.

    def build_hicache_transfers(
        self,
        node: "UnifiedTreeNode",
        phase: CacheTransferPhase,
        **kw,
    ) -> Optional[list[PoolTransfer]]:
        return None

    @staticmethod
    def _emit_three_pool_transfers(
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
        node: "UnifiedTreeNode",
    ) -> list[PoolTransfer]:
        return [
            PoolTransfer(
                name=pool_name,
                device_indices=device_indices,
                host_indices=host_indices,
                nodes_to_load=[node],
            )
            for pool_name in (PoolName.C4, PoolName.C4_INDEXER, PoolName.C128)
        ]

    def _build_backup_transfers(
        self, nodes: list["UnifiedTreeNode"]
    ) -> list[PoolTransfer]:
        """For each evictable node with device value but no host backup, allocate
        host page-ids on the c4 anchor pool and emit 3 PoolTransfer descriptors."""
        transfers = []
        ct = self.component_type
        for node in nodes:
            cd = node.component_data[ct]
            if cd.value is None or cd.host_value is not None or cd.lock_ref > 0:
                continue
            host_ids = self._c4_pool_host.alloc(num_pages=len(cd.value))
            if host_ids is None:
                break  # host pool full; let drive_host_eviction handle it next call.
            transfers.extend(self._emit_three_pool_transfers(cd.value, host_ids, node))
        return transfers

    def _build_load_transfers(
        self, nodes: list["UnifiedTreeNode"]
    ) -> list[PoolTransfer]:
        """For each host-only node, claim device page-ids and emit 3 H→D PoolTransfers.

        Critical: cd.value is set BEFORE the transfer completes so concurrent
        callers see the in-flight load and don't redundantly claim device slots.
        """
        transfers = []
        ct = self.component_type
        for node in nodes:
            cd = node.component_data[ct]
            if cd.host_value is None or cd.value is not None:
                continue
            device_ids = self._dsv4_pool.alloc_compressed_pages(
                num_pages=len(cd.host_value)
            )
            if device_ids is None:
                break  # device pool full; eviction must run first.
            cd.value = device_ids
            self.cache.component_evictable_size_[ct] += len(device_ids)
            self.cache.lru_lists[ct].insert_mru(node)
            if node not in self.cache.evictable_device_leaves:
                self.cache.evictable_device_leaves.add(node)
            transfers.extend(
                self._emit_three_pool_transfers(device_ids, cd.host_value, node)
            )
        return transfers

    def commit_hicache_transfer(
        self,
        node: "UnifiedTreeNode",
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
    ) -> None:
        """No-op; FULL handles host_value commit on the c4 anchor."""
        return None

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """No-op; FULL drives host eviction (it owns the c4 anchor host pool)."""
        return None
