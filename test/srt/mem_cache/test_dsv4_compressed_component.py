"""Tests for DSV4CompressedComponent.

DSV4_COMPRESSED is a passive shadow of FULL: it shares the same page-id
namespace and tree-level lifecycle, while the actual host backup / load is
handled by the c4 anchor + register_hicache_anchor_kv_shared_indices_pool
mechanism on the host pool group. Tests cover only the shadow lifecycle:
registry, match validator, eviction bookkeeping, split, insert-overlap.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    ComponentData,
    ComponentType,
    DSV4CompressedComponent,
    EvictLayer,
)


class _FakeNode:
    def __init__(self, value=None, host_value=None):
        self.component_data = {
            ComponentType.DSV4_COMPRESSED: ComponentData(
                value=value,
                lock_ref=0,
                host_value=None,
                host_lock_ref=0,
            ),
            BASE_COMPONENT_TYPE: ComponentData(
                value=value,
                lock_ref=0,
                host_value=host_value,
                host_lock_ref=0,
            ),
        }
        self.parent = None
        self.last_access_time = 0.0


class _FakeCache:
    def __init__(self):
        self.evictable_device_leaves = set()
        self.evictable_host_leaves = set()
        self.lru_lists = {ComponentType.DSV4_COMPRESSED: _FakeLRU()}
        self.host_lru_lists = {ComponentType.DSV4_COMPRESSED: _FakeLRU()}
        self.component_evictable_size_ = {ComponentType.DSV4_COMPRESSED: 0}
        self.component_protected_size_ = {ComponentType.DSV4_COMPRESSED: 0}
        self.root_node = None


class _FakeLRU:
    def __init__(self):
        self.items = []

    def insert_mru(self, node):
        if node not in self.items:
            self.items.append(node)


def _make_component():
    cache = _FakeCache()
    comp = DSV4CompressedComponent.__new__(DSV4CompressedComponent)
    comp.cache = cache
    comp._dsv4_pool = None
    comp._c4_pool_host = None
    comp._c4_indexer_pool_host = None
    comp._c128_pool_host = None
    return comp, cache


def test_component_registry_contains_dsv4_compressed():
    from sglang.srt.mem_cache.unified_radix_cache import COMPONENT_REGISTRY

    assert COMPONENT_REGISTRY[ComponentType.DSV4_COMPRESSED] is DSV4CompressedComponent


def test_component_type_class_attribute():
    assert DSV4CompressedComponent.component_type is ComponentType.DSV4_COMPRESSED


def test_match_validator_alive_when_device_value_present():
    comp, _ = _make_component()
    validator = comp.create_match_validator()
    node = _FakeNode(value=torch.arange(0, 8, dtype=torch.long))
    assert validator(node) is True


def test_match_validator_alive_when_full_host_value_present():
    """DSV4_COMPRESSED defers host-aliveness to FULL.host_value because the
    c4 anchor flow stores host indices on FULL, not on the shadow component."""
    comp, _ = _make_component()
    validator = comp.create_match_validator()
    node = _FakeNode(value=None, host_value=torch.arange(0, 4, dtype=torch.long))
    # Clear the shadow's value to simulate cascade-evict.
    node.component_data[ComponentType.DSV4_COMPRESSED].value = None
    assert validator(node) is True


def test_match_validator_dead_when_neither_value_nor_host_value():
    comp, _ = _make_component()
    validator = comp.create_match_validator()
    node = _FakeNode(value=None)
    node.component_data[ComponentType.DSV4_COMPRESSED].value = None
    node.component_data[BASE_COMPONENT_TYPE].host_value = None
    assert validator(node) is False


def test_evict_component_device_clears_value_bookkeeping_only():
    """FULL is the authoritative free-er; DSV4_COMPRESSED.evict only clears
    cd.value and decrements its evictable counter. No allocator call."""
    comp, cache = _make_component()
    node = _FakeNode(value=torch.arange(0, 8, dtype=torch.long))
    cache.component_evictable_size_[ComponentType.DSV4_COMPRESSED] = 8
    device_freed, host_freed = comp.evict_component(node, target=EvictLayer.DEVICE)
    assert device_freed == 8
    assert host_freed == 0
    cd = node.component_data[ComponentType.DSV4_COMPRESSED]
    assert cd.value is None
    assert cache.component_evictable_size_[ComponentType.DSV4_COMPRESSED] == 0


def test_redistribute_on_node_split_slices_indices():
    comp, _ = _make_component()
    parent = _FakeNode()
    parent.key = list(range(3))
    child = _FakeNode()
    child.component_data[ComponentType.DSV4_COMPRESSED].value = torch.arange(
        100, 108, dtype=torch.long
    )
    child.component_data[ComponentType.DSV4_COMPRESSED].host_value = torch.arange(
        0, 8, dtype=torch.long
    )
    comp.redistribute_on_node_split(parent, child)
    parent_cd = parent.component_data[ComponentType.DSV4_COMPRESSED]
    child_cd = child.component_data[ComponentType.DSV4_COMPRESSED]
    assert parent_cd.value.tolist() == [100, 101, 102]
    assert child_cd.value.tolist() == [103, 104, 105, 106, 107]
    assert parent_cd.host_value.tolist() == [0, 1, 2]
    assert child_cd.host_value.tolist() == [3, 4, 5, 6, 7]


def test_insert_overlap_returns_full_prefix_len():
    comp, _ = _make_component()
    node = _FakeNode()
    consumed = comp.update_component_on_insert_overlap(
        node=node,
        prefix_len=16,
        total_prefix_len=16,
        value_slice=None,
        params=None,
    )
    assert consumed == 16


def test_drive_eviction_is_noop():
    """FULL drives device eviction for DSV4 (shared allocator namespace)."""
    comp, cache = _make_component()
    node = _FakeNode(value=torch.arange(0, 8, dtype=torch.long))
    cache.evictable_device_leaves.add(node)
    cache.component_evictable_size_[ComponentType.DSV4_COMPRESSED] = 8
    tracker = {ComponentType.DSV4_COMPRESSED: 0}
    # Pass a minimal params object exposing num_tokens.
    import types

    params = types.SimpleNamespace(num_tokens=8)
    comp.drive_eviction(params, tracker)
    assert tracker[ComponentType.DSV4_COMPRESSED] == 0
    cd = node.component_data[ComponentType.DSV4_COMPRESSED]
    assert cd.value is not None  # untouched — FULL handles the actual eviction


def test_hicache_hooks_are_noops():
    comp, _ = _make_component()
    node = _FakeNode(value=torch.arange(0, 8, dtype=torch.long))
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        CacheTransferPhase,
    )

    assert comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST) is None
    assert comp.build_hicache_transfers(node, CacheTransferPhase.LOAD_BACK) is None
    assert (
        comp.commit_hicache_transfer(node, CacheTransferPhase.BACKUP_HOST, ()) is None
    )
    tracker = {ComponentType.DSV4_COMPRESSED: 0}
    assert comp.drive_host_eviction(num_tokens=8, tracker=tracker) is None
    assert tracker[ComponentType.DSV4_COMPRESSED] == 0
