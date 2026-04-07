# Launch

## Start Bailian qwen3-rerank

The Bailian adapter is built in and loads by default.

This command was validated locally with the model at
`/models/Qwen/Qwen3-Reranker-0.6B`:

```bash
PYTHONPATH=python:. \
python -m sglang.launch_server \
  --model-path /models/Qwen/Qwen3-Reranker-0.6B \
  --served-model-name qwen3-rerank \
  --chat-template examples/chat_template/qwen3_reranker.jinja \
  --host 127.0.0.1 \
  --port 38000 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

## Test Request

Official-compatible path:

```bash
curl -X POST http://127.0.0.1:38000/compatible-api/v1/reranks \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-rerank",
    "documents": [
      "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
      "量子计算是计算科学的一个前沿领域",
      "预训练语言模型的发展给文本排序模型带来了新的进展"
    ],
    "query": "什么是文本排序模型",
    "top_n": 2,
    "instruct": "Given a web search query, retrieve relevant passages that answer the query."
  }'
```

The adapter also exposes a shorter alias:

```bash
curl -X POST http://127.0.0.1:38000/reranks \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-rerank",
    "query": "什么是文本排序模型",
    "documents": ["doc1", "doc2", "doc3"],
    "top_n": 2
  }'
```

## Local Validation

The request above returned a Bailian-style response:

```json
{
  "output": {
    "results": [
      {
        "index": 0,
        "relevance_score": 0.9924227605250054
      },
      {
        "index": 2,
        "relevance_score": 0.766293557690281
      }
    ]
  },
  "usage": {
    "total_tokens": 264
  },
  "request_id": "2335ce90-079a-4aa9-b3cb-e99bf97538bd"
}
```

This matches the Bailian `qwen3-rerank` response envelope:

- `output.results[*].index`
- `output.results[*].relevance_score`
- `usage.total_tokens`
- `request_id`

The ranking also looked correct: document `0` and document `2` are about text
reranking, while document `1` is unrelated.

`return_documents` now defaults to `true` on the Bailian adapter. If you want
to suppress returned document text, set `return_documents=false` explicitly.

When documents are returned, each result includes:

```json
{
  "document": {
    "text": "..."
  }
}
```
