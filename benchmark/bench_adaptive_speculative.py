"""
Benchmark adaptive speculative decoding vs static configurations.

Sends mixed traffic with varying acceptance characteristics:
- Phase 1 (easy): Repetitive/predictable text → high acceptance
- Phase 2 (hard): Creative/diverse generation → low acceptance
- Phase 3 (easy): Back to predictable text → high acceptance

Adaptive should match the best static config for each phase.

Usage:
# 1. Start server with adaptive mode:
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-eagle-topk 1 \
    --speculative-adaptive \
    --port 30000

# 2. Run benchmark:
python benchmark/bench_adaptive_speculative.py --port 30000

# 3. Compare with static baselines by restarting server without --speculative-adaptive
#    and with different --speculative-num-steps values (e.g., 2, 3, 5)
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests


# Prompts designed to produce different acceptance rates.
# "Easy" prompts: structured output, code, repetitive patterns → high draft acceptance
# "Hard" prompts: creative, open-ended, reasoning → low draft acceptance

EASY_PROMPTS = [
    "Write a Python function that computes the factorial of n using a for loop. Include type hints and a docstring.",
    "List the numbers from 1 to 50, one per line.",
    "Write a JSON schema for a User object with fields: id (integer), name (string), email (string), age (integer).",
    "Convert this Python dict to a formatted JSON string: {'name': 'Alice', 'scores': [90, 85, 92, 88, 95, 91, 87, 93, 89, 96]}",
    "Write a simple HTML page with a header, a paragraph, and a bulleted list of 10 items about programming languages.",
    "Write a SQL CREATE TABLE statement for a users table with columns: id, username, email, created_at, updated_at, is_active.",
    "Print the multiplication table from 1 to 10 in a formatted grid.",
    "Write a Python class called Rectangle with __init__, area, perimeter, and __repr__ methods.",
]

HARD_PROMPTS = [
    "Explain the philosophical implications of Gödel's incompleteness theorems on artificial intelligence. Be creative and speculative.",
    "Write a surrealist short story about a clock that can taste colors. Use unexpected metaphors.",
    "If gravity suddenly reversed for 10 seconds every hour, describe 5 non-obvious societal changes that would emerge over 100 years.",
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. Make it emotionally resonant.",
    "Design a new board game that teaches group theory concepts. Describe rules, pieces, and winning conditions in detail.",
    "What would happen if humans could photosynthesize? Analyze the biological, economic, and cultural implications.",
    "Write a dialogue between Socrates and a modern AI researcher debating whether machines can truly understand meaning.",
    "Invent a new musical instrument. Describe its physical construction, how it's played, and what it sounds like.",
]

STRUCTURED_PROMPTS = [
    EASY_PROMPTS[1],
    EASY_PROMPTS[2],
    EASY_PROMPTS[3],
    EASY_PROMPTS[5],
    EASY_PROMPTS[6],
]

CODEGEN_PROMPTS = [
    EASY_PROMPTS[0],
    EASY_PROMPTS[4],
    EASY_PROMPTS[7],
]

ANALYTICAL_PROMPTS = [
    HARD_PROMPTS[0],
    HARD_PROMPTS[2],
    HARD_PROMPTS[5],
    HARD_PROMPTS[6],
]

CREATIVE_PROMPTS = [
    HARD_PROMPTS[1],
    HARD_PROMPTS[3],
    HARD_PROMPTS[4],
    HARD_PROMPTS[7],
]

# Hand-curated prompt pools that create a wider acceptance spread on GLM-4.7-FP8
# with EAGLE speculative decoding:
# - high: highly repetitive outputs that can reach ~8 accepted tokens
# - mid: structured/codegen tasks that typically sit around ~4
# - low: long-form varied generation that can fall near ~2 on 256-token runs
EXTREME_HIGH_PROMPTS = [
    "Output exactly 256 new lines. Every line must be 1. Do not add numbering, punctuation, or commentary.",
    "Output exactly 256 new lines. Every line must be READY. Do not add numbering, punctuation, or commentary.",
]

EXTREME_MID_PROMPTS = [
    "Write a CSV with header n,double,triple and rows for n=1 through 80. Do not add any explanation.",
    "Write a JSON schema for a User object with fields: id (integer), name (string), email (string), age (integer).",
    "Implement binary search in Python and include 3 small doctest examples.",
    "Write a Python function that merges overlapping intervals. Include type hints and a short explanation.",
]

EXTREME_LOW_PROMPTS = [
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. Make it emotionally resonant.",
    "Write 100 two-sentence biographies of eccentric inventors with unique names, hometowns, and inventions.",
    "Write a long travel diary from a botanist visiting a chain of floating islands. Every paragraph should introduce new flora, customs, weather, and political tensions.",
    "Write 80 newspaper headlines and subheads from 80 different alternate-history worlds. Each headline must introduce a different place, conflict, and technology.",
]

SCENARIOS = {
    "balanced": [
        ("easy_1", EASY_PROMPTS, 1.0),
        ("hard", HARD_PROMPTS, 1.0),
        ("easy_2", EASY_PROMPTS, 1.0),
    ],
    # Longer hard span gives adaptive more time to settle on smaller steps.
    "hard_heavy": [
        ("easy_1", EASY_PROMPTS, 0.5),
        ("hard_1", HARD_PROMPTS, 2.0),
        ("hard_2", HARD_PROMPTS, 2.0),
        ("easy_2", EASY_PROMPTS, 0.5),
    ],
    # Stress the steady-state low-acceptance case directly.
    "hard_only": [
        ("hard_1", HARD_PROMPTS, 3.0),
    ],
    # Longer overall run with multiple transitions to measure switch overhead.
    "switchback": [
        ("easy_1", EASY_PROMPTS, 0.75),
        ("hard_1", HARD_PROMPTS, 1.5),
        ("easy_2", EASY_PROMPTS, 0.75),
        ("hard_2", HARD_PROMPTS, 1.5),
        ("easy_3", EASY_PROMPTS, 0.75),
    ],
    # Short alternating phases with multiple workload subtypes to stress adaptation
    # when acceptance characteristics flip frequently.
    "rapid_mix": [
        ("structured_1", STRUCTURED_PROMPTS, 0.5),
        ("creative_1", CREATIVE_PROMPTS, 0.5),
        ("codegen_1", CODEGEN_PROMPTS, 0.5),
        ("analytical_1", ANALYTICAL_PROMPTS, 0.5),
        ("structured_2", STRUCTURED_PROMPTS, 0.5),
        ("analytical_2", ANALYTICAL_PROMPTS, 0.5),
        ("codegen_2", CODEGEN_PROMPTS, 0.5),
        ("creative_2", CREATIVE_PROMPTS, 0.5),
        ("structured_3", STRUCTURED_PROMPTS, 0.5),
    ],
    # Wide acceptance ladder: roughly ~8 -> ~4 -> ~2 -> ~4 -> ~8.
    "oracle_ladder_256": [
        ("high_1", EXTREME_HIGH_PROMPTS, 0.75),
        ("mid_1", EXTREME_MID_PROMPTS, 1.0),
        ("low_1", EXTREME_LOW_PROMPTS, 2.0),
        ("mid_2", EXTREME_MID_PROMPTS, 1.0),
        ("high_2", EXTREME_HIGH_PROMPTS, 0.75),
    ],
    # Single-bucket scenarios for measuring the steady-state overhead of
    # keeping a fixed speculative step size on a stable workload.
    "oracle_high_only_256": [
        ("high_1", EXTREME_HIGH_PROMPTS, 3.0),
    ],
    "oracle_mid_only_256": [
        ("mid_1", EXTREME_MID_PROMPTS, 3.0),
    ],
    "oracle_low_only_256": [
        ("low_1", EXTREME_LOW_PROMPTS, 3.0),
    ],
    # Low-heavy variant to give adaptive more time to settle on smaller steps.
    "oracle_low_heavy_256": [
        ("high_1", EXTREME_HIGH_PROMPTS, 0.5),
        ("mid_1", EXTREME_MID_PROMPTS, 0.75),
        ("low_1", EXTREME_LOW_PROMPTS, 2.0),
        ("low_2", EXTREME_LOW_PROMPTS, 2.0),
        ("mid_2", EXTREME_MID_PROMPTS, 0.75),
        ("high_2", EXTREME_HIGH_PROMPTS, 0.5),
    ],
    # Long recovery tail so adaptive has enough time to climb back up after
    # spending a long interval on small speculative steps.
    "oracle_recovery_256": [
        ("low_1", EXTREME_LOW_PROMPTS, 2.0),
        ("low_2", EXTREME_LOW_PROMPTS, 2.0),
        ("mid_1", EXTREME_MID_PROMPTS, 1.0),
        ("high_1", EXTREME_HIGH_PROMPTS, 2.0),
        ("high_2", EXTREME_HIGH_PROMPTS, 2.0),
    ],
}


def build_phase_plan(scenario, base_requests):
    """Expand a scenario preset into concrete per-phase request counts."""
    phases = []
    for phase_name, prompts, multiplier in SCENARIOS[scenario]:
        num_requests = max(1, round(base_requests * multiplier))
        phases.append((phase_name, prompts, num_requests))
    return phases


def send_request(base_url, prompt, max_tokens=256):
    """Send a single generate request and return timing + acceptance info."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
                "return_logprob": False,
            },
            timeout=max(120, max_tokens),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e), "latency": time.perf_counter() - start}

    latency = time.perf_counter() - start
    meta = data.get("meta_info", {})
    completion_tokens = meta.get("completion_tokens", 0)
    spec_verify_ct = meta.get("spec_verify_ct", 0)

    accept_len = (
        completion_tokens / spec_verify_ct if spec_verify_ct > 0 else float("nan")
    )

    return {
        "latency": latency,
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct,
        "accept_length": accept_len,
    }


def run_phase(base_url, prompts, phase_name, num_requests, max_tokens, parallel):
    """Run a phase of the benchmark with the given prompts."""
    # Repeat prompts to fill num_requests
    expanded = (prompts * ((num_requests + len(prompts) - 1) // len(prompts)))[
        :num_requests
    ]

    print(f"\n--- Phase: {phase_name} ({num_requests} requests, parallel={parallel}) ---")
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [
            pool.submit(send_request, base_url, p, max_tokens) for p in expanded
        ]
        results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start

    # Aggregate
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if not ok:
        print(f"  All {len(errors)} requests failed!")
        return {"phase": phase_name, "error": True}

    total_tokens = sum(r["completion_tokens"] for r in ok)
    total_verify = sum(r["spec_verify_ct"] for r in ok)
    avg_latency = sum(r["latency"] for r in ok) / len(ok)
    throughput = total_tokens / elapsed
    avg_accept_len = total_tokens / total_verify if total_verify > 0 else float("nan")

    stats = {
        "phase": phase_name,
        "num_requests": len(ok),
        "num_errors": len(errors),
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(throughput, 2),
        "avg_latency_s": round(avg_latency, 3),
        "avg_accept_length": round(avg_accept_len, 3),
    }

    print(
        f"  Throughput: {throughput:.1f} tok/s | "
        f"Avg latency: {avg_latency:.3f}s | "
        f"Avg accept_len: {avg_accept_len:.2f} | "
        f"Errors: {len(errors)}"
    )
    return stats


def get_server_info(base_url):
    """Get current adaptive spec params from server info if available."""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=5)
        info = resp.json()
        return info
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive speculative decoding"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--num-requests", type=int, default=32, help="Requests per phase"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument(
        "--warmup", type=int, default=8, help="Warmup requests before benchmark"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="balanced",
        choices=sorted(SCENARIOS.keys()),
        help="Traffic mix preset",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Check server is up
    print(f"Connecting to {base_url}...")
    try:
        info = get_server_info(base_url)
        if info is None:
            print("ERROR: Cannot connect to server")
            return
        print("Server connected.")
        print(f"Scenario: {args.scenario}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Warmup
    phase_plan = build_phase_plan(args.scenario, args.num_requests)
    if args.warmup > 0:
        print(f"\nWarming up with {args.warmup} requests...")
        warmup_prompts = phase_plan[0][1] if phase_plan else EASY_PROMPTS
        run_phase(
            base_url,
            warmup_prompts,
            "warmup",
            args.warmup,
            args.max_tokens,
            args.parallel,
        )

    # Run benchmark phases
    all_stats = []
    for phase_name, prompts, num_requests in phase_plan:
        stats = run_phase(
            base_url,
            prompts,
            phase_name,
            num_requests,
            args.max_tokens,
            args.parallel,
        )
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Phase':<10} {'Throughput':>12} {'Avg Latency':>12} {'Accept Len':>12}")
    print("-" * 50)
    for s in all_stats:
        if "error" in s and s.get("error"):
            print(f"{s['phase']:<10} {'ERROR':>12}")
        else:
            print(
                f"{s['phase']:<10} "
                f"{s['throughput_tok_s']:>10.1f}/s "
                f"{s['avg_latency_s']:>10.3f}s "
                f"{s['avg_accept_length']:>11.2f}"
            )

    # Overall
    ok_stats = [s for s in all_stats if not s.get("error")]
    if ok_stats:
        total_tokens = sum(s["total_tokens"] for s in ok_stats)
        total_elapsed = sum(s["elapsed_s"] for s in ok_stats)
        overall_throughput = total_tokens / total_elapsed if total_elapsed > 0 else 0
        print("-" * 50)
        print(f"{'OVERALL':<10} {overall_throughput:>10.1f}/s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "scenario": args.scenario,
                    "num_requests": args.num_requests,
                    "max_tokens": args.max_tokens,
                    "parallel": args.parallel,
                    "warmup": args.warmup,
                    "phases": all_stats,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
