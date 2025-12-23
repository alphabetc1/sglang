import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sgl_kernel import (
    chain_speculative_sampling_target_only,
    tree_speculative_sampling_target_only,
)


@dataclass
class BenchConfig:
    bs: int
    num_draft_tokens: int
    vocab_size: int
    iters: int
    warmup: int
    threshold_single: float
    threshold_acc: float
    deterministic: bool
    seed: int


def _build_chain_inputs(cfg: BenchConfig, device: str):
    """
    Build a topk=1 "chain" structure (no siblings):
      - retrive_next_token: 0 -> 1 -> 2 -> ... -> (n-1) -> -1
      - retrive_next_sibling: all -1
      - retrive_index: maps each node to a flat output position in predicts
    """
    bs = cfg.bs
    n = cfg.num_draft_tokens
    d = cfg.vocab_size

    # candidates: [bs, n], token ids in [0, d)
    candidates = torch.randint(0, d, (bs, n), dtype=torch.int64, device=device)
    # Root token id is unused by the kernels, but keep it valid and stable.
    candidates[:, 0] = 0

    # retrive_index: [bs, n] maps to a flat predicts buffer of length bs*n
    base = (torch.arange(bs, device=device, dtype=torch.int64) * n).unsqueeze(1)
    retrive_index = base + torch.arange(n, device=device, dtype=torch.int64).unsqueeze(
        0
    )

    # next_token: [bs, n] with -1 at the tail
    next_token = (
        torch.arange(n, device=device, dtype=torch.int64).unsqueeze(0).repeat(bs, 1)
    )
    next_token[:, :-1] = next_token[:, 1:]
    next_token[:, -1] = -1
    retrive_next_token = next_token.contiguous()

    # no siblings for topk=1
    retrive_next_sibling = torch.full((bs, n), -1, dtype=torch.int64, device=device)

    # Uniform samples: [bs, n] and [bs]
    coins = torch.rand((bs, n), dtype=torch.float32, device=device)
    coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

    # target_probs: [bs, n, d] float32 probabilities
    # Keep it moderately "peaky" to resemble real logits but avoid extreme underflow.
    logits = torch.randn((bs, n, d), dtype=torch.float32, device=device)
    target_probs = F.softmax(logits, dim=-1)

    return (
        candidates.contiguous(),
        retrive_index.contiguous(),
        retrive_next_token,
        retrive_next_sibling.contiguous(),
        coins.contiguous(),
        coins_for_final_sampling.contiguous(),
        target_probs.contiguous(),
    )


def _alloc_outputs(cfg: BenchConfig, device: str):
    bs = cfg.bs
    n = cfg.num_draft_tokens
    num_spec_tokens = n  # allow at most n-1 draft tokens accepted
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


def _reset_outputs(
    predicts: torch.Tensor, accept_index: torch.Tensor, accept_token_num: torch.Tensor
):
    predicts.fill_(-1)
    accept_index.fill_(-1)
    accept_token_num.zero_()


def _time_kernel_cuda_events(fn, iters: int) -> list[float]:
    """
    Returns per-iter GPU time in milliseconds using CUDA events.
    """
    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _summarize(name: str, times_ms: list[float]) -> str:
    times_sorted = sorted(times_ms)
    mean = statistics.mean(times_ms)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(len(times_sorted) * 0.95)]
    return f"{name}: mean={mean:.3f}ms  p50={p50:.3f}ms  p95={p95:.3f}ms  iters={len(times_ms)}"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark topk=1 speculative sampling: chain vs tree (same inputs, CUDA events timing)."
    )
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num-draft-tokens", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threshold-single", type=float, default=1.0)
    parser.add_argument("--threshold-acc", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    cfg = BenchConfig(
        bs=args.bs,
        num_draft_tokens=args.num_draft_tokens,
        vocab_size=args.vocab_size,
        iters=args.iters,
        warmup=args.warmup,
        threshold_single=args.threshold_single,
        threshold_acc=args.threshold_acc,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = "cuda"

    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        coins,
        coins_for_final_sampling,
        target_probs,
    ) = _build_chain_inputs(cfg, device)

    # Pre-allocate outputs for both kernels.
    predicts_tree, accept_index_tree, accept_token_num_tree = _alloc_outputs(
        cfg, device
    )
    predicts_chain, accept_index_chain, accept_token_num_chain = _alloc_outputs(
        cfg, device
    )

    def call_tree():
        tree_speculative_sampling_target_only(
            predicts=predicts_tree,
            accept_index=accept_index_tree,
            accept_token_num=accept_token_num_tree,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            threshold_single=cfg.threshold_single,
            threshold_acc=cfg.threshold_acc,
            deterministic=cfg.deterministic,
        )

    def call_chain():
        chain_speculative_sampling_target_only(
            predicts=predicts_chain,
            accept_index=accept_index_chain,
            accept_token_num=accept_token_num_chain,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            threshold_single=cfg.threshold_single,
            threshold_acc=cfg.threshold_acc,
            deterministic=cfg.deterministic,
        )

    # Warmup (also ensures kernels are JIT-loaded and caches are populated).
    for _ in range(cfg.warmup):
        _reset_outputs(predicts_tree, accept_index_tree, accept_token_num_tree)
        call_tree()
        _reset_outputs(predicts_chain, accept_index_chain, accept_token_num_chain)
        call_chain()
    torch.cuda.synchronize()

    # Time tree
    def timed_tree():
        _reset_outputs(predicts_tree, accept_index_tree, accept_token_num_tree)
        call_tree()

    # Time chain
    def timed_chain():
        _reset_outputs(predicts_chain, accept_index_chain, accept_token_num_chain)
        call_chain()

    # Make sure clocks are stable before timing.
    torch.cuda.synchronize()
    time.sleep(0.05)

    tree_ms = _time_kernel_cuda_events(timed_tree, cfg.iters)
    chain_ms = _time_kernel_cuda_events(timed_chain, cfg.iters)

    print("=== Config ===")
    print(
        f"bs={cfg.bs} num_draft_tokens={cfg.num_draft_tokens} vocab_size={cfg.vocab_size} "
        f"warmup={cfg.warmup} iters={cfg.iters} threshold_single={cfg.threshold_single} "
        f"threshold_acc={cfg.threshold_acc} deterministic={cfg.deterministic} seed={cfg.seed}"
    )
    print("=== Results (GPU kernel time) ===")
    print(_summarize("tree", tree_ms))
    print(_summarize("chain", chain_ms))
    speedup = statistics.mean(tree_ms) / max(statistics.mean(chain_ms), 1e-12)
    print(f"speedup(tree/chain) = {speedup:.2f}x")

    # Optional correctness spot-check: outputs are expected to match for topk==1 chain structure.
    # This is not exhaustive and is deliberately outside the timed region.
    _reset_outputs(predicts_tree, accept_index_tree, accept_token_num_tree)
    _reset_outputs(predicts_chain, accept_index_chain, accept_token_num_chain)
    call_tree()
    call_chain()
    torch.cuda.synchronize()
    ok = (
        torch.equal(predicts_tree, predicts_chain)
        and torch.equal(accept_index_tree, accept_index_chain)
        and torch.equal(accept_token_num_tree, accept_token_num_chain)
    )
    print(f"spot_check_equal(tree, chain) = {ok}")


if __name__ == "__main__":
    main()
