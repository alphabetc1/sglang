"""Probe SGLang frontend overhead with comparable request shapes.

This script intentionally keeps dependencies to the packages already used by
``sglang.bench_serving``.  It records per-request client latency for OpenAI chat,
native text, and native input_ids requests.  Native ``/generate`` responses also
carry SGLang internal timestamps when the server is launched with
``--enable-metrics``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def make_prompt(i: int, target_tokens: int, tokenizer: Any) -> tuple[str, list[int]]:
    # Use unique-but-boring text so prefix cache cannot dominate if it is enabled.
    base = (
        f"Request {i}. Summarize the operational implications of moving an LLM "
        "serving frontend from Python to Rust. Include routing, tokenization, "
        "streaming, validation, and observability tradeoffs. "
    )
    text = base
    while len(tokenizer.encode(text)) < target_tokens:
        text += (
            f" Detail {len(text)}: latency budgets depend on concurrency, prompt "
            "length, JSON parsing, chat template rendering, and scheduler queueing."
        )
    ids = tokenizer.encode(text)[:target_tokens]
    # Decode back to make native text and input_ids as comparable as possible.
    return tokenizer.decode(ids), ids


def make_heavy_tools(count: int = 8, fields_per_tool: int = 8) -> list[dict[str, Any]]:
    tools = []
    for tool_idx in range(count):
        properties = {
            f"field_{field_idx}": {
                "type": "string",
                "description": (
                    "A verbose field used to stress OpenAI tool schema validation, "
                    "normalization, template rendering, and tokenizer encoding."
                ),
            }
            for field_idx in range(fields_per_tool)
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"lookup_frontend_metric_{tool_idx}",
                    "description": (
                        "Lookup detailed frontend, scheduler, tokenizer, HTTP, "
                        "validation, and serialization metrics for a serving request."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": ["field_0", "field_1", "field_2"],
                    },
                },
            }
        )
    return tools


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_mode.setdefault(rec["mode"], []).append(rec)

    summary: dict[str, Any] = {}
    for mode, rows in by_mode.items():
        ok = [r for r in rows if r["ok"]]
        mode_summary: dict[str, Any] = {
            "requests": len(rows),
            "ok": len(ok),
            "client_latency_s": stats([r["client_latency_s"] for r in ok]),
            "prompt_tokens": stats([r.get("prompt_tokens", 0) for r in ok]),
            "completion_tokens": stats([r.get("completion_tokens", 0) for r in ok]),
        }
        for key in [
            "server_e2e_s",
            "frontend_dispatch_s",
            "post_dispatch_to_finish_s",
            "queue_time_s",
            "response_send_lag_s",
        ]:
            vals = [r[key] for r in ok if r.get(key) is not None]
            if vals:
                mode_summary[key] = stats(vals)
        timing_keys = sorted(
            {
                key
                for row in ok
                for key in row
                if key.startswith("openai_frontend_timing_")
            }
        )
        for key in timing_keys:
            vals = [r[key] for r in ok if r.get(key) is not None]
            if vals:
                mode_summary[key] = stats(vals)
        summary[mode] = mode_summary
    return summary


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    mode: str,
    idx: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    rec: dict[str, Any] = {"mode": mode, "idx": idx, "ok": False}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            rec["status"] = resp.status
            rec["client_latency_s"] = time.perf_counter() - start
            if resp.status != 200:
                rec["error"] = text[:1000]
                return rec
            data = json.loads(text)
            rec["ok"] = True
            rec["response"] = data

            meta = data.get("meta_info") or {}
            response_metadata = data.get("metadata") or {}
            usage = data.get("usage") or {}
            rec["prompt_tokens"] = meta.get("prompt_tokens", usage.get("prompt_tokens"))
            rec["completion_tokens"] = meta.get(
                "completion_tokens", usage.get("completion_tokens")
            )
            rec["cached_tokens"] = meta.get("cached_tokens")
            rec["server_e2e_s"] = meta.get("e2e_latency")
            rec["queue_time_s"] = meta.get("queue_time")

            received = meta.get("request_received_ts")
            dispatched = meta.get("api_server_dispatch_finish_ts")
            finished = meta.get("request_finished_ts")
            sent = meta.get("response_sent_to_client_ts")
            if received is not None and dispatched is not None:
                rec["frontend_dispatch_s"] = dispatched - received
            if dispatched is not None and finished is not None:
                rec["post_dispatch_to_finish_s"] = finished - dispatched
            if finished is not None and sent is not None:
                rec["response_send_lag_s"] = sent - finished
            frontend_timing = response_metadata.get("sglang_frontend_timing") or {}
            for group_name, group in frontend_timing.items():
                if not isinstance(group, dict):
                    continue
                for timing_name, value in group.items():
                    if isinstance(value, (int, float)):
                        rec[f"openai_frontend_timing_{group_name}_{timing_name}_s"] = (
                            value
                        )
            return rec
    except Exception as exc:  # noqa: BLE001 - benchmark artifact should keep going.
        rec["client_latency_s"] = time.perf_counter() - start
        rec["error"] = repr(exc)
        return rec


def build_payload(
    mode: str,
    model: str,
    prompt: str,
    input_ids: list[int],
    output_len: int,
) -> tuple[str, dict[str, Any]]:
    rid = f"frontend-probe-{mode}-{uuid.uuid4().hex}"
    if mode == "native_input_ids":
        return "/generate", {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_len,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": False,
            "rid": rid,
        }
    if mode == "native_text":
        return "/generate", {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": output_len,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": False,
            "rid": rid,
        }
    if mode == "oai_chat":
        return "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": output_len,
            "temperature": 0,
            "stream": False,
            "rid": rid,
        }
    if mode == "oai_chat_tools":
        return "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": make_heavy_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_completion_tokens": output_len,
            "temperature": 0,
            "stream": False,
            "rid": rid,
        }
    if mode == "oai_completion":
        return "/v1/completions", {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0,
            "stream": False,
            "rid": rid,
        }
    raise ValueError(f"unknown mode: {mode}")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    prompts = [make_prompt(i, args.input_len, tokenizer) for i in range(args.requests)]

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * len(args.modes), 1))
    records: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for mode in args.modes:
            headers = (
                {"x-sglang-frontend-timing": "1"}
                if args.openai_frontend_timing
                and mode in {"oai_chat", "oai_chat_tools"}
                else None
            )
            warmups = min(args.warmup, len(prompts))
            for i in range(warmups):
                path, payload = build_payload(
                    mode, args.model, prompts[i][0], prompts[i][1], args.output_len
                )
                await post_json(
                    session, args.base_url + path, payload, mode, -1 - i, headers
                )

            sem = asyncio.Semaphore(args.concurrency)

            async def one(i: int) -> dict[str, Any]:
                async with sem:
                    path, payload = build_payload(
                        mode,
                        args.model,
                        prompts[i][0],
                        prompts[i][1],
                        args.output_len,
                    )
                    return await post_json(
                        session, args.base_url + path, payload, mode, i, headers
                    )

            start = time.perf_counter()
            mode_records = await asyncio.gather(*(one(i) for i in range(len(prompts))))
            elapsed = time.perf_counter() - start
            for rec in mode_records:
                rec["mode_elapsed_s"] = elapsed
            records.extend(mode_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    details_path = args.output.with_suffix(".jsonl")
    with details_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    result = {
        "config": vars(args) | {"output": str(args.output)},
        "summary": summarize(records),
        "details_jsonl": str(details_path),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31031")
    parser.add_argument("--model", default="/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default="/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--openai-frontend-timing",
        action="store_true",
        help="Request debug frontend timing metadata from OpenAI chat responses.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["native_input_ids", "native_text", "oai_completion", "oai_chat"],
        choices=[
            "native_input_ids",
            "native_text",
            "oai_completion",
            "oai_chat",
            "oai_chat_tools",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/frontend_overhead_probe/result.json"),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    random.seed(args.seed)
    return args


def main() -> None:
    result = asyncio.run(run(parse_args()))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
