"""DSV4 hicache reproducer & benchmark.

This script verifies the two acceptance gates for DSV4+hicache v1:

  1. Functional parity (GSM8K accuracy):
     # Launch DSV4 server with hicache:
     python -m sglang.launch_server --model deepseek-ai/<dsv4-model> \\
         --enable-hierarchical-cache --hicache-ratio 4

     # Then in a second terminal:
     python -m sglang.test.eval_gsm8k --base-url http://localhost:30000 \\
         --num-questions 200 --parallel 32

     # Expected: score within +-1% of the same model run WITHOUT
     # --enable-hierarchical-cache.

  2. TTFT improvement (this script):
     python python/sglang/srt/mem_cache/bench_dsv4_hicache.py \\
         --base-url http://localhost:30000 \\
         --prefix-tokens 16384 \\
         --rounds 3

What this script measures:
  - Round 1: cold TTFT (no cache).
  - Round 2: post-flush_cache TTFT — exercises host hit + reload + SWA tail-replay.
  - Round 3: device-resident hit TTFT — should be fastest.
"""

from __future__ import annotations

import argparse
import time

import requests


def _generate(base_url: str, prompt: str, max_new: int = 4) -> float:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/generate",
        json={"text": prompt, "sampling_params": {"max_new_tokens": max_new}},
    )
    resp.raise_for_status()
    return time.perf_counter() - t0


def _flush(base_url: str) -> None:
    resp = requests.post(f"{base_url}/flush_cache")
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=16384,
        help="approximate length of the synthetic shared prefix in tokens",
    )
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    prefix = " ".join([f"token{i}" for i in range(args.prefix_tokens)])
    suffix = " summarize the above."
    prompt = prefix + suffix

    print("Round 1 (cold) ...")
    t1 = _generate(args.base_url, prompt)
    print(f"  total time at max_new_tokens=4: {t1:.3f}s")

    if args.rounds >= 2:
        _flush(args.base_url)
        print("Round 2 (post-flush, host hit + reload + SWA replay) ...")
        t2 = _generate(args.base_url, prompt)
        speedup_2 = t1 / t2 if t2 > 0 else float("inf")
        print(f"  total time: {t2:.3f}s    speedup vs cold: {speedup_2:.2f}x")

    if args.rounds >= 3:
        print("Round 3 (device hit) ...")
        t3 = _generate(args.base_url, prompt)
        speedup_3 = t1 / t3 if t3 > 0 else float("inf")
        print(f"  total time: {t3:.3f}s    speedup vs cold: {speedup_3:.2f}x")


if __name__ == "__main__":
    main()
