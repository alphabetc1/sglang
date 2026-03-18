import unittest

import torch

from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=2, suite="stage-b-test-small-1-gpu-amd")


class TestBuildPEagleTree(unittest.TestCase):
    def test_linear_tree_layout(self):
        device = get_device()
        verified_id = torch.tensor([101], device=device, dtype=torch.int32)
        draft_tokens = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int64)
        seq_lens = torch.tensor([5], device=device, dtype=torch.int32)
        parent_list = torch.tensor([[-1, 0, 1]], device=device, dtype=torch.int64)
        top_scores_index = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

        (
            _tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            seq_lens,
            seq_lens_sum=5,
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
        )

        self.assertEqual(flat_draft_tokens.tolist(), [101, 1, 2, 3])
        self.assertEqual(positions.tolist(), [5, 6, 7, 8])
        self.assertEqual(retrive_index.tolist(), [[0, 1, 2, 3]])
        self.assertEqual(retrive_next_token.tolist(), [[1, 2, 3, -1]])
        self.assertEqual(retrive_next_sibling.tolist(), [[-1, -1, -1, -1]])


if __name__ == "__main__":
    unittest.main()
