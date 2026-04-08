import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.entrypoints.compat.bailian_qwen3_rerank import (
    BailianQwen3RerankAdapter,
    BailianQwenRerankRequest,
)
from sglang.srt.entrypoints.compat.base import CompatServiceRegistry
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestCompatBailianRerank(unittest.TestCase):
    def test_routes_include_v1_reranks(self):
        paths = {route.path for route in BailianQwen3RerankAdapter().route_specs()}
        self.assertIn("/reranks", paths)
        self.assertIn("/v1/reranks", paths)
        self.assertIn("/compatible-api/v1/reranks", paths)

    def test_return_documents_defaults_to_false(self):
        request = BailianQwenRerankRequest(
            model="qwen3-rerank",
            query="什么是文本排序模型",
            documents=["doc1", "doc2"],
        )

        self.assertFalse(request.return_documents)
        self.assertFalse(request.to_v1_rerank_request().return_documents)

    def test_return_documents_can_be_explicitly_disabled(self):
        request = BailianQwenRerankRequest(
            model="qwen3-rerank",
            query="什么是文本排序模型",
            documents=["doc1", "doc2"],
            return_documents=False,
        )

        self.assertFalse(request.return_documents)
        self.assertFalse(request.to_v1_rerank_request().return_documents)

    def test_handle_request_returns_flat_bailian_response(self):
        adapter = BailianQwen3RerankAdapter()
        raw_request = Mock()
        raw_request.headers = {"content-type": "application/json"}
        raw_request.json = AsyncMock(
            return_value={
                "model": "qwen3-rerank",
                "query": "什么是文本排序模型",
                "documents": ["doc0", "doc1", "doc2"],
            }
        )

        rerank_service = Mock()
        rerank_service.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(
                chat_template='Note that the answer can only be "yes" or "no".'
            )
        )
        rerank_service._validate_request.return_value = None
        rerank_service._score_text_reranker_request = AsyncMock(
            return_value=(
                [0.9254411498359545, 0.311868147831286, 0.7565501439174214],
                105,
            )
        )
        rerank_service._build_rerank_response.return_value = [
            SimpleNamespace(index=0, score=0.9254411498359545, document=None),
            SimpleNamespace(index=2, score=0.7565501439174214, document=None),
            SimpleNamespace(index=1, score=0.311868147831286, document=None),
        ]

        response = self._run(
            adapter.handle_request(
                raw_request,
                CompatServiceRegistry(rerank=rerank_service),
            )
        )
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["model"], "qwen3-rerank")
        self.assertIn("id", payload)
        self.assertEqual(payload["usage"]["total_tokens"], 105)
        self.assertEqual(
            payload["results"],
            [
                {"index": 0, "relevance_score": 0.9254411498359545},
                {"index": 2, "relevance_score": 0.7565501439174214},
                {"index": 1, "relevance_score": 0.311868147831286},
            ],
        )

    def _run(self, coro):
        import asyncio

        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main(verbosity=2)
