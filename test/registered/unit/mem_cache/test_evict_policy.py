"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    BackupAwareStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)


def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    node.host_value = kwargs.get("host_value", None)
    # backuped property: True when host_value is not None
    type(node).backuped = property(lambda self: self.host_value is not None)
    return node


class TestLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LRUStrategy()

    def test_priority_is_last_access_time(self):
        node = _make_node(last_access_time=42.0)
        self.assertEqual(self.strategy.get_priority(node), 42.0)

    def test_older_access_evicted_first(self):
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestLFUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LFUStrategy()

    def test_priority_is_hit_count_and_time(self):
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), (5, 3.0))

    def test_lower_hit_count_evicted_first(self):
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    def test_same_hit_count_older_access_evicted_first(self):
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestFIFOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FIFOStrategy()

    def test_priority_is_creation_time(self):
        node = _make_node(creation_time=7.0)
        self.assertEqual(self.strategy.get_priority(node), 7.0)

    def test_earlier_created_evicted_first(self):
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


class TestMRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MRUStrategy()

    def test_priority_is_negated_access_time(self):
        node = _make_node(last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), -5.0)

    def test_most_recently_used_evicted_first(self):
        """MRU evicts the most recently accessed node first (lowest priority value)."""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


class TestFILOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FILOStrategy()

    def test_priority_is_negated_creation_time(self):
        node = _make_node(creation_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), -3.0)

    def test_last_created_evicted_first(self):
        """FILO evicts the most recently created node first."""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


class TestPriorityStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_priority_is_tuple(self):
        node = _make_node(priority=2, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 4.0))

    def test_lower_priority_evicted_first(self):
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    def test_same_priority_older_access_evicted_first(self):
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestSLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    def test_probationary_segment(self):
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_protected_segment(self):
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_highly_accessed_is_protected(self):
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_probationary_evicted_before_protected(self):
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    def test_same_segment_older_access_evicted_first(self):
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))
        self.assertEqual(strategy.get_priority(at), (1, 1.0))

    def test_default_threshold_is_2(self):
        default = SLRUStrategy()
        self.assertEqual(default.protected_threshold, 2)


class TestBackupAwareStrategy(unittest.TestCase):
    """Tests for BackupAwareStrategy decorator."""

    def test_backuped_node_has_lower_priority_lru(self):
        """Backed-up nodes should be evicted before non-backed-up (lower priority value)."""
        strategy = BackupAwareStrategy(LRUStrategy())
        backuped = _make_node(last_access_time=10.0, host_value="some_indices")
        not_backuped = _make_node(last_access_time=1.0)
        # Even though not_backuped has older access time (normally evicted first),
        # backuped node should have lower priority (evicted first from GPU).
        self.assertLess(
            strategy.get_priority(backuped), strategy.get_priority(not_backuped)
        )

    def test_same_backup_status_preserves_inner_order(self):
        """Within the same backup tier, inner strategy ordering is preserved."""
        strategy = BackupAwareStrategy(LRUStrategy())
        old_backuped = _make_node(last_access_time=1.0, host_value="idx")
        new_backuped = _make_node(last_access_time=10.0, host_value="idx")
        self.assertLess(
            strategy.get_priority(old_backuped),
            strategy.get_priority(new_backuped),
        )

    def test_wraps_tuple_returning_strategy(self):
        """Works correctly with strategies that return tuples (e.g., LFU, SLRU)."""
        strategy = BackupAwareStrategy(LFUStrategy())
        backuped = _make_node(hit_count=100, last_access_time=10.0, host_value="idx")
        not_backuped = _make_node(hit_count=0, last_access_time=1.0)
        # Backed-up hot node should still be evicted before cold non-backed-up node
        self.assertLess(
            strategy.get_priority(backuped), strategy.get_priority(not_backuped)
        )

    def test_wraps_float_returning_strategy(self):
        """Priority tuple is (backup_tier, float) for float-returning strategies."""
        strategy = BackupAwareStrategy(LRUStrategy())
        node = _make_node(last_access_time=5.0, host_value="idx")
        self.assertEqual(strategy.get_priority(node), (0, 5.0))

        node_no_backup = _make_node(last_access_time=5.0)
        self.assertEqual(strategy.get_priority(node_no_backup), (1, 5.0))

    def test_wraps_slru_strategy(self):
        """Priority tuple is (backup_tier, segment, time) for SLRU."""
        strategy = BackupAwareStrategy(SLRUStrategy(protected_threshold=2))
        node = _make_node(hit_count=3, last_access_time=5.0, host_value="idx")
        self.assertEqual(strategy.get_priority(node), (0, 1, 5.0))

    def test_full_ordering_mixed_nodes(self):
        """Integration: sort mixed backed-up and non-backed-up nodes."""
        strategy = BackupAwareStrategy(LRUStrategy())
        nodes = [
            _make_node(last_access_time=1.0),  # not backuped, old
            _make_node(last_access_time=10.0, host_value="idx"),  # backuped, new
            _make_node(last_access_time=5.0, host_value="idx"),  # backuped, old
            _make_node(last_access_time=3.0),  # not backuped, newer
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected_priorities = [
            (0, 5.0),  # backuped, old  -> evict first
            (0, 10.0),  # backuped, new
            (1, 1.0),  # not backuped, old
            (1, 3.0),  # not backuped, new -> evict last
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected_priorities)


class TestBackupAwareHeapSimulation(unittest.TestCase):
    """Simulate the heapq-based eviction loop to verify backed-up nodes are popped first."""

    def test_heapq_pops_backuped_before_non_backuped(self):
        """Reproduce the exact heap pattern used in HiRadixCache.evict()."""
        import heapq

        strategy = BackupAwareStrategy(LRUStrategy())
        # Mix of backed-up and non-backed-up nodes with various access times
        nodes = [
            _make_node(last_access_time=1.0),  # cold, not backuped
            _make_node(last_access_time=8.0, host_value="idx"),  # warm, backuped
            _make_node(last_access_time=3.0, host_value="idx"),  # cold, backuped
            _make_node(last_access_time=6.0),  # warm, not backuped
            _make_node(last_access_time=10.0, host_value="idx"),  # hot, backuped
        ]
        # Build heap exactly as evict() does
        heap = [(strategy.get_priority(n), n) for n in nodes]
        heapq.heapify(heap)

        popped = []
        while heap:
            _pri, node = heapq.heappop(heap)
            popped.append(node)

        # First 3 popped must all be backed-up (tier 0), in LRU order
        for node in popped[:3]:
            self.assertTrue(node.backuped, "Expected backed-up node in first tier")
        self.assertEqual([n.last_access_time for n in popped[:3]], [3.0, 8.0, 10.0])
        # Last 2 must be non-backed-up (tier 1), in LRU order
        for node in popped[3:]:
            self.assertFalse(
                node.backuped, "Expected non-backed-up node in second tier"
            )
        self.assertEqual([n.last_access_time for n in popped[3:]], [1.0, 6.0])

    def test_himamba_heap_key_pops_backuped_first(self):
        """Reproduce the exact heap pattern used in HiMambaRadixCache.evict()."""
        import heapq

        nodes = [
            _make_node(last_access_time=2.0),  # not backuped
            _make_node(last_access_time=9.0, host_value="idx"),  # backuped
            _make_node(last_access_time=4.0, host_value="idx"),  # backuped
            _make_node(last_access_time=7.0),  # not backuped
        ]
        # Build heap exactly as HiMambaRadixCache.evict() does (inline key)
        heap = [((0 if n.backuped else 1, n.last_access_time), n) for n in nodes]
        heapq.heapify(heap)

        popped = []
        while heap:
            _pri, node = heapq.heappop(heap)
            popped.append(node)

        # Backed-up nodes first
        self.assertTrue(popped[0].backuped)
        self.assertTrue(popped[1].backuped)
        self.assertFalse(popped[2].backuped)
        self.assertFalse(popped[3].backuped)
        self.assertEqual([n.last_access_time for n in popped], [4.0, 9.0, 2.0, 7.0])


class TestEvictionOrdering(unittest.TestCase):
    """Integration-style test: sort a list of nodes by eviction priority."""

    def test_lru_ordering(self):
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])

    def test_slru_ordering(self):
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # protected, old
            _make_node(hit_count=0, last_access_time=10.0),  # probationary, new
            _make_node(hit_count=0, last_access_time=2.0),  # probationary, old
            _make_node(hit_count=3, last_access_time=8.0),  # protected, new
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # probationary old
            (0, 10.0),  # probationary new
            (1, 1.0),  # protected old
            (1, 8.0),  # protected new
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
