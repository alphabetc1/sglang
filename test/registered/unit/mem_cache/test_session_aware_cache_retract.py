"""Unit tests for SessionAwareCache retract cleanup."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


class _FakeInnerCache:
    def __init__(self, page_size: int = 4):
        self.page_size = page_size
        self.disable = False
        self.metrics_collector = None
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int32).reshape(2, 16),
            free_slots=[],
        )
        self.token_to_kv_pool_allocator = _FakeAllocator()

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        raise AssertionError("streaming requests should stay in SessionAwareCache")

    def reset(self):
        raise AssertionError("not used in this test")


def _make_streaming_req(*, req_pool_idx: int, kv_committed_len: int, kv_allocated_len: int):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        kv_committed_len=kv_committed_len,
        kv_allocated_len=kv_allocated_len,
        cache_protected_len=4,
        swa_evicted_seqlen=0,
        last_node="node",
        swa_uuid_for_lock=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
        session=SimpleNamespace(session_id="session-1", streaming=True),
    )


class TestSessionAwareCacheRetractCleanup(unittest.TestCase):
    def test_retract_frees_page_aligned_overallocated_tokens_before_slot_takeover(self):
        inner = _FakeInnerCache(page_size=4)
        cache = SessionAwareCache(inner)
        req = _make_streaming_req(
            req_pool_idx=1,
            kv_committed_len=5,
            kv_allocated_len=9,
        )

        cache.cache_finished_req(req, is_insert=False)

        self.assertEqual([buf.tolist() for buf in inner.token_to_kv_pool_allocator.freed], [[24]])
        self.assertEqual(req.kv_allocated_len, 5)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(cache.slots["session-1"].kv_allocated_len, 5)

    def test_insert_path_keeps_overallocated_tokens_for_common_release_flow(self):
        inner = _FakeInnerCache(page_size=4)
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
