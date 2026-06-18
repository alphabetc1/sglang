"""Unit tests for hybrid HiCache anchor-capacity resolution.

Covers resolve_anchor_capacities: the uniform scaling derived from
--hicache-size / --hicache-ratio, the two consistent capacity views, the
device_pages + 1 floor, and the over-allocation warning it emits.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    AnchorGroup,
    resolve_anchor_capacities,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ASSEMBLER = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"


def _server_args(*, hicache_size: float, hicache_ratio: float) -> SimpleNamespace:
    return SimpleNamespace(hicache_size=hicache_size, hicache_ratio=hicache_ratio)


class TestResolveAnchorCapacities(CustomTestCase):
    def test_size_mode_splits_by_device_bytes(self):
        # Two groups with different per-page bytes share one back-solved ratio.
        # device_bytes = 1000*1e6 + 2000*5e5 = 2e9; ratio = 4e9 / 2e9 = 2.0.
        groups = [
            AnchorGroup("A", 1000, 1_000_000),
            AnchorGroup("B", 2000, 500_000),
        ]
        caps = resolve_anchor_capacities(
            _server_args(hicache_size=4.0, hicache_ratio=99.0), groups
        )
        self.assertEqual([c.host_pages for c in caps], [2000, 4000])
        # The two views are arithmetically consistent and the GB view sums
        # back to the requested --hicache-size.
        self.assertAlmostEqual(sum(c.host_size_gb for c in caps), 4.0)

    def test_ratio_mode_ignores_size(self):
        groups = [AnchorGroup("A", 1000, 1_000_000)]
        caps = resolve_anchor_capacities(
            _server_args(hicache_size=0, hicache_ratio=3.0), groups
        )
        self.assertEqual(caps[0].host_pages, 3000)

    def test_floor_keeps_host_above_device(self):
        # ratio 1.0 would give host_pages == device_pages, which violates the
        # host > device protocol; the floor bumps it to device_pages + 1.
        groups = [AnchorGroup("A", 1000, 1_000_000)]
        caps = resolve_anchor_capacities(
            _server_args(hicache_size=0, hicache_ratio=1.0), groups
        )
        self.assertEqual(caps[0].host_pages, 1001)

    def test_size_mode_warns_when_floor_raises(self):
        # A tiny --hicache-size scales every group below device_pages + 1, so
        # the floor raises them and the real footprint exceeds the request.
        groups = [
            AnchorGroup("A", 1000, 1_000_000),
            AnchorGroup("B", 1000, 1_000_000),
        ]
        with mock.patch(f"{_ASSEMBLER}.logger") as logger:
            caps = resolve_anchor_capacities(
                _server_args(hicache_size=0.0005, hicache_ratio=2.0), groups
            )
        self.assertEqual([c.host_pages for c in caps], [1001, 1001])
        logger.warning.assert_called_once()
        warned_groups = logger.warning.call_args.args[1]
        self.assertEqual(warned_groups, "A, B")

    def test_ratio_mode_does_not_warn_on_floor(self):
        # Without an absolute --hicache-size budget, hitting the floor is not an
        # over-allocation, so no warning is emitted.
        groups = [AnchorGroup("A", 1000, 1_000_000)]
        with mock.patch(f"{_ASSEMBLER}.logger") as logger:
            resolve_anchor_capacities(
                _server_args(hicache_size=0, hicache_ratio=1.0), groups
            )
        logger.warning.assert_not_called()

    def test_size_mode_requires_nonzero_device_bytes(self):
        # All-logical groups (0 bytes) give no anchor to back-solve from.
        groups = [AnchorGroup("LOGICAL", 1000, 0)]
        with self.assertRaises(ValueError):
            resolve_anchor_capacities(
                _server_args(hicache_size=4.0, hicache_ratio=2.0), groups
            )


if __name__ == "__main__":
    unittest.main()
