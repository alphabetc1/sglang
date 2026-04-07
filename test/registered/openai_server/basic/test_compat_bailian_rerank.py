import unittest

from sglang.srt.entrypoints.compat.bailian_qwen3_rerank import (
    BailianQwenRerankRequest,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestCompatBailianRerank(unittest.TestCase):
    def test_return_documents_defaults_to_true(self):
        request = BailianQwenRerankRequest(
            model="qwen3-rerank",
            query="什么是文本排序模型",
            documents=["doc1", "doc2"],
        )

        self.assertTrue(request.return_documents)
        self.assertTrue(request.to_v1_rerank_request().return_documents)

    def test_return_documents_can_be_explicitly_disabled(self):
        request = BailianQwenRerankRequest(
            model="qwen3-rerank",
            query="什么是文本排序模型",
            documents=["doc1", "doc2"],
            return_documents=False,
        )

        self.assertFalse(request.return_documents)
        self.assertFalse(request.to_v1_rerank_request().return_documents)


if __name__ == "__main__":
    unittest.main(verbosity=2)
