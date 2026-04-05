"""Unit tests for SessionAwareCache."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache, SessionSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _FakeInnerCache:
    def __init__(self, page_size: int = 4):
        self.page_size = page_size
        self.disable = False
        self.metrics_collector = None

    def reset(self):
        raise AssertionError("not used in this test")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


class _FakeStreamingInnerCache(_FakeInnerCache):
    def __init__(self, page_size: int = 4):
        super().__init__(page_size=page_size)
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int32).reshape(2, 16),
            free_slots=[],
        )
        self.token_to_kv_pool_allocator = _FakeAllocator()

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        raise AssertionError("streaming requests should stay in SessionAwareCache")


def _make_req(
    *,
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    cache_protected_len: int = 0,
    swa_evicted_seqlen: int = 0,
):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        kv_committed_len=kv_committed_len,
        kv_allocated_len=kv_allocated_len,
        cache_protected_len=cache_protected_len,
        swa_evicted_seqlen=swa_evicted_seqlen,
        last_node="node",
        swa_uuid_for_lock=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
    )


def _make_streaming_req(
    *, req_pool_idx: int, kv_committed_len: int, kv_allocated_len: int
):
    req = _make_req(
        req_pool_idx=req_pool_idx,
        kv_committed_len=kv_committed_len,
        kv_allocated_len=kv_allocated_len,
        cache_protected_len=4,
    )
    req.session = SimpleNamespace(session_id="session-1", streaming=True)
    return req


class TestSessionAwareCacheBusyAccounting(unittest.TestCase):
    def setUp(self):
        self.cache = SessionAwareCache(_FakeInnerCache(page_size=4))

    def test_inactive_slot_counts_toward_session_held(self):
        self.cache.slots["sid"] = SessionSlot(
            req_pool_idx=3,
            kv_allocated_len=10,
            cache_protected_len=2,
            swa_evicted_seqlen=6,
        )

        self.assertEqual(self.cache.session_held_tokens(), 10)
        self.assertEqual(self.cache.session_held_swa_tokens(), 6)
        self.assertEqual(self.cache.session_held_req_count(), 1)

    def test_active_slot_is_excluded_from_session_held_counters(self):
        slot = SessionSlot(
            req_pool_idx=3,
            kv_committed_len=10,
            kv_allocated_len=10,
            cache_protected_len=2,
        )
        self.cache.slots["sid"] = slot

        req = _make_req(
            req_pool_idx=99,
            kv_committed_len=1,
            kv_allocated_len=1,
        )
        slot.restore_to_req(req)

        self.assertTrue(slot.active)
        self.assertEqual(req.req_pool_idx, 3)
        self.assertEqual(self.cache.session_held_tokens(), 0)
        self.assertEqual(self.cache.session_held_swa_tokens(), 0)
        self.assertEqual(self.cache.session_held_req_count(), 0)

    def test_save_from_req_marks_slot_inactive_again(self):
        slot = SessionSlot(active=True)
        self.cache.slots["sid"] = slot

        req = _make_req(
            req_pool_idx=5,
            kv_committed_len=7,
            kv_allocated_len=9,
            cache_protected_len=1,
        )
        slot.save_from_req(req, is_first=True)

        self.assertFalse(slot.active)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(self.cache.session_held_req_count(), 1)


class TestSessionAwareCacheRetractCleanup(unittest.TestCase):
    def test_retract_frees_page_aligned_overallocated_tokens_before_slot_takeover(self):
        inner = _FakeStreamingInnerCache(page_size=4)
        cache = SessionAwareCache(inner)
        req = _make_streaming_req(
            req_pool_idx=1,
            kv_committed_len=5,
            kv_allocated_len=9,
        )

        cache.cache_finished_req(req, is_insert=False)

        self.assertEqual(
            [buf.tolist() for buf in inner.token_to_kv_pool_allocator.freed],
            [[24]],
        )
        self.assertEqual(req.kv_allocated_len, 5)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(cache.slots["session-1"].kv_allocated_len, 5)

    def test_insert_path_keeps_overallocated_tokens_for_common_release_flow(self):
        inner = _FakeStreamingInnerCache(page_size=4)
        cache = SessionAwareCache(inner)
        req = _make_streaming_req(
            req_pool_idx=1,
            kv_committed_len=5,
            kv_allocated_len=9,
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(inner.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(cache.slots["session-1"].kv_allocated_len, 9)


if __name__ == "__main__":
    unittest.main()
