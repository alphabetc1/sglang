from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class BackupAwareStrategy(EvictionStrategy):
    """Decorator that makes any strategy prefer evicting backed-up nodes first.

    Backed-up nodes (host_value != None) can be evicted from GPU at near-zero
    cost (just free device memory; data remains on host for load_back). Non-backed-up
    nodes either require a blocking GPU->Host transfer (write_back) or lose data
    permanently (write_through). Prepending a backup tier to the priority ensures
    backed-up nodes are evicted before non-backed-up ones, with the inner strategy
    as tiebreaker within each tier.
    """

    def __init__(self, inner: EvictionStrategy):
        self.inner = inner

    def get_priority(self, node: "TreeNode") -> Tuple:
        inner_priority = self.inner.get_priority(node)
        # Tier 0 = backed-up (evict first), Tier 1 = not backed-up (evict later)
        backup_tier = 0 if node.backuped else 1
        if isinstance(inner_priority, tuple):
            return (backup_tier,) + inner_priority
        return (backup_tier, inner_priority)
