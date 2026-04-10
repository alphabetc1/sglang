#!/usr/bin/env python3
"""
Benchmark script for measuring rerank service TPM (Tokens Per Minute).

Simulates N concurrent users sending rerank requests with configurable
document count and text length to match real-world RAG workloads.

Usage:
    python bench_rerank_tpm.py \
        --url http://127.0.0.1:38002 \
        --model qwen3-rerank \
        --concurrency 32 \
        --num-requests 500 \
        --num-docs 10 \
        --doc-tokens 200 \
        --query-tokens 50 \
        --duration 60

    # Use --duration for time-based runs, or --num-requests for count-based runs.
    # If both are set, whichever limit is hit first stops the benchmark.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

# Reusable word pool so generated text looks semi-realistic for tokenizers
_WORD_POOL = [
    "search",
    "query",
    "document",
    "ranking",
    "model",
    "relevance",
    "score",
    "text",
    "information",
    "retrieval",
    "neural",
    "network",
    "transformer",
    "attention",
    "embedding",
    "vector",
    "semantic",
    "similarity",
    "passage",
    "knowledge",
    "language",
    "understanding",
    "context",
    "token",
    "sequence",
    "encoder",
    "decoder",
    "pretrained",
    "finetune",
    "benchmark",
    "dataset",
    "training",
    "inference",
    "latency",
    "throughput",
    "batch",
    "parallel",
    "optimization",
    "performance",
    "accuracy",
    "recall",
    "precision",
    "metric",
    "evaluation",
    "cross",
    "rerank",
    "candidate",
    "generation",
    "response",
]


def _random_text(approx_tokens: int) -> str:
    """Generate random text with approximately `approx_tokens` tokens.

    Rough heuristic: 1 token ≈ 1 word for English-like text in most tokenizers.
    """
    words = random.choices(_WORD_POOL, k=approx_tokens)
    return " ".join(words)


def build_request_payload(
    model: str,
    num_docs: int,
    query_tokens: int,
    doc_tokens: int,
    instruct: Optional[str] = None,
    api_style: str = "v1",
) -> dict:
    query = _random_text(query_tokens)
    documents = [_random_text(doc_tokens) for _ in range(num_docs)]

    if api_style == "bailian":
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
    else:
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": False,
        }

    if instruct:
        payload["instruct"] = instruct
    return payload


# ---------------------------------------------------------------------------
# Stats collection
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    success: bool
    latency_ms: float
    prompt_tokens: int = 0
    error: str = ""


@dataclass
class BenchmarkStats:
    results: List[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    def summary(self) -> dict:
        ok = [r for r in self.results if r.success]
        fail = [r for r in self.results if not r.success]
        total_tokens = sum(r.prompt_tokens for r in ok)
        duration_min = self.duration_s / 60.0

        latencies = sorted(r.latency_ms for r in ok) if ok else [0]

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            idx = min(idx, len(data) - 1)
            return data[idx]

        return {
            "total_requests": len(self.results),
            "successful": len(ok),
            "failed": len(fail),
            "duration_s": round(self.duration_s, 2),
            "total_tokens": total_tokens,
            "TPM": round(total_tokens / duration_min, 0) if duration_min > 0 else 0,
            "RPS": round(len(ok) / self.duration_s, 2) if self.duration_s > 0 else 0,
            "avg_tokens_per_req": round(total_tokens / len(ok), 1) if ok else 0,
            "latency_ms": {
                "avg": round(sum(r.latency_ms for r in ok) / len(ok), 1) if ok else 0,
                "p50": round(percentile(latencies, 50), 1),
                "p95": round(percentile(latencies, 95), 1),
                "p99": round(percentile(latencies, 99), 1),
                "min": round(latencies[0], 1),
                "max": round(latencies[-1], 1),
            },
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def send_one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    api_style: str,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000

            if resp.status != 200:
                return RequestResult(
                    success=False,
                    latency_ms=latency_ms,
                    error=f"HTTP {resp.status}: {json.dumps(body)[:200]}",
                )

            # Extract token count from response
            prompt_tokens = 0
            if api_style == "bailian":
                # Bailian format: {"usage": {"total_tokens": N}}
                prompt_tokens = body.get("usage", {}).get("total_tokens", 0)
            else:
                # v1/rerank format: each result has meta_info.prompt_tokens
                # or we sum from individual results
                results = body if isinstance(body, list) else body.get("results", body)
                if isinstance(results, list):
                    for r in results:
                        mi = r.get("meta_info", {}) if isinstance(r, dict) else {}
                        prompt_tokens += mi.get("prompt_tokens", 0)

            return RequestResult(
                success=True,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
            )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(success=False, latency_ms=latency_ms, error=str(e))


async def worker(
    worker_id: int,
    session: aiohttp.ClientSession,
    url: str,
    args: argparse.Namespace,
    stats: BenchmarkStats,
    stop_event: asyncio.Event,
    request_counter: dict,
    counter_lock: asyncio.Lock,
):
    while not stop_event.is_set():
        # Check request count limit
        if args.num_requests:
            async with counter_lock:
                if request_counter["n"] >= args.num_requests:
                    return
                request_counter["n"] += 1

        payload = build_request_payload(
            model=args.model,
            num_docs=args.num_docs,
            query_tokens=args.query_tokens,
            doc_tokens=args.doc_tokens,
            instruct=args.instruct,
            api_style=args.api_style,
        )

        result = await send_one_request(session, url, payload, args.api_style)
        stats.results.append(result)

        if not result.success and args.verbose:
            print(f"[worker-{worker_id}] ERROR: {result.error}")


async def run_benchmark(args: argparse.Namespace):
    if args.api_style == "bailian":
        url = f"{args.url.rstrip('/')}/compatible-api/v1/reranks"
    else:
        url = f"{args.url.rstrip('/')}/v1/rerank"

    print(f"Target URL: {url}")
    print(f"Concurrency: {args.concurrency}")
    print(
        f"Docs/request: {args.num_docs}, ~query_tokens: {args.query_tokens}, ~doc_tokens: {args.doc_tokens}"
    )
    if args.num_requests:
        print(f"Total requests: {args.num_requests}")
    if args.duration:
        print(f"Duration limit: {args.duration}s")
    print()

    # Warm up with a single request
    print("Warming up...")
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        warmup_payload = build_request_payload(
            model=args.model,
            num_docs=args.num_docs,
            query_tokens=args.query_tokens,
            doc_tokens=args.doc_tokens,
            instruct=args.instruct,
            api_style=args.api_style,
        )
        warmup = await send_one_request(session, url, warmup_payload, args.api_style)
        if not warmup.success:
            print(f"Warmup failed: {warmup.error}")
            print("Aborting benchmark.")
            return
        print(f"Warmup OK: {warmup.latency_ms:.0f}ms, {warmup.prompt_tokens} tokens\n")

    # Run benchmark
    stats = BenchmarkStats()
    stop_event = asyncio.Event()
    request_counter = {"n": 0}
    counter_lock = asyncio.Lock()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        stats.start_time = time.perf_counter()

        workers = [
            asyncio.create_task(
                worker(
                    i,
                    session,
                    url,
                    args,
                    stats,
                    stop_event,
                    request_counter,
                    counter_lock,
                )
            )
            for i in range(args.concurrency)
        ]

        # Duration-based stop
        if args.duration:

            async def _timer():
                await asyncio.sleep(args.duration)
                stop_event.set()

            timer_task = asyncio.create_task(_timer())
        else:
            timer_task = None

        await asyncio.gather(*workers)
        stop_event.set()
        if timer_task:
            timer_task.cancel()

        stats.end_time = time.perf_counter()

    # Print results
    s = stats.summary()
    print("=" * 60)
    print("  Rerank Benchmark Results")
    print("=" * 60)
    print(f"  Duration:           {s['duration_s']}s")
    print(
        f"  Requests:           {s['successful']} ok / {s['failed']} failed / {s['total_requests']} total"
    )
    print(f"  Total Tokens:       {s['total_tokens']:,}")
    print(f"  TPM:                {s['TPM']:,.0f}")
    print(f"  RPS:                {s['RPS']}")
    print(f"  Avg tokens/req:     {s['avg_tokens_per_req']}")
    print()
    lat = s["latency_ms"]
    print(
        f"  Latency (ms):       avg={lat['avg']}  p50={lat['p50']}  p95={lat['p95']}  p99={lat['p99']}"
    )
    print(f"                      min={lat['min']}  max={lat['max']}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(s, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark rerank service TPM")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:38002",
        help="Base URL of the sglang server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-rerank",
        help="Model name for bailian-style requests",
    )
    parser.add_argument(
        "--api-style",
        choices=["v1", "bailian"],
        default="bailian",
        help="API style: 'v1' for /v1/rerank, 'bailian' for /compatible-api/v1/reranks",
    )
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Total number of requests to send (None = unlimited, use --duration)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (0 = no limit, use --num-requests)",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of documents per rerank request",
    )
    parser.add_argument(
        "--query-tokens",
        type=int,
        default=50,
        help="Approximate token count for query text",
    )
    parser.add_argument(
        "--doc-tokens",
        type=int,
        default=200,
        help="Approximate token count per document",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Optional instruct string for the reranker",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results JSON to this file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print individual request errors"
    )
    args = parser.parse_args()

    if not args.num_requests and not args.duration:
        parser.error("At least one of --num-requests or --duration must be set")

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
