import unittest

import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestEagleDraftInputOptionalTopK(unittest.TestCase):
    def test_filter_batch_without_topk(self):
        draft_input = EagleDraftInput(
            hidden_states=torch.tensor(
                [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=torch.float32
            ),
            verified_id=torch.tensor([10, 20, 30], dtype=torch.int32),
            topk_p=None,
            topk_index=None,
        )

        draft_input.filter_batch(
            new_indices=torch.tensor([2, 0], dtype=torch.int64),
            has_been_filtered=False,
        )

        self.assertIsNone(draft_input.topk_p)
        self.assertIsNone(draft_input.topk_index)
        self.assertTrue(
            torch.equal(
                draft_input.hidden_states,
                torch.tensor([[3.0, 3.5], [1.0, 1.5]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.verified_id,
                torch.tensor([30, 10], dtype=torch.int32),
            )
        )

    def test_merge_batch_without_topk(self):
        lhs = EagleDraftInput(
            hidden_states=torch.tensor([[1.0, 1.5]], dtype=torch.float32),
            verified_id=torch.tensor([10], dtype=torch.int32),
            topk_p=None,
            topk_index=None,
        )
        rhs = EagleDraftInput(
            hidden_states=torch.tensor([[2.0, 2.5]], dtype=torch.float32),
            verified_id=torch.tensor([20], dtype=torch.int32),
            topk_p=None,
            topk_index=None,
        )

        lhs.merge_batch(rhs)

        self.assertIsNone(lhs.topk_p)
        self.assertIsNone(lhs.topk_index)
        self.assertTrue(
            torch.equal(
                lhs.hidden_states,
                torch.tensor([[1.0, 1.5], [2.0, 2.5]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(lhs.verified_id, torch.tensor([10, 20], dtype=torch.int32))
        )
