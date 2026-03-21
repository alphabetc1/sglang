import math

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    build_default_adaptive_candidate_steps,
)


def _run_static_trace(trace, steps, base_cost=1.0, step_cost=0.2):
    total_cost = 0.0
    total_emitted_tokens = 0

    for latent_accept_len in trace:
        accepted = min(latent_accept_len, steps)
        total_emitted_tokens += accepted + 1
        total_cost += base_cost + step_cost * (steps + 1)

    return total_emitted_tokens / total_cost


def _run_adaptive_trace(
    trace,
    initial_steps=5,
    max_steps=5,
    min_steps=1,
    base_cost=1.0,
    step_cost=0.2,
    ema_alpha=0.1,
    update_interval=10,
    warmup_batches=0,
    down_hysteresis=0.0,
    up_hysteresis=0.0,
    candidate_steps=None,
):
    params = AdaptiveSpeculativeParams(
        initial_steps=initial_steps,
        max_steps=max_steps,
        min_steps=min_steps,
        ema_alpha=ema_alpha,
        update_interval=update_interval,
        warmup_batches=warmup_batches,
        down_hysteresis=down_hysteresis,
        up_hysteresis=up_hysteresis,
        candidate_steps=candidate_steps,
    )
    total_cost = 0.0
    total_emitted_tokens = 0
    step_history = []

    for latent_accept_len in trace:
        current_steps = params.current_steps
        accepted = min(latent_accept_len, current_steps)
        total_emitted_tokens += accepted + 1
        total_cost += base_cost + step_cost * (current_steps + 1)
        step_history.append(current_steps)
        params.update([accepted])

    return total_emitted_tokens / total_cost, step_history


def test_init_clamps_steps_and_tracks_topk1_invariant():
    params = AdaptiveSpeculativeParams(initial_steps=9, max_steps=5, min_steps=2)

    assert params.current_steps == 5
    assert params.current_draft_tokens == 6
    assert params.ema_accept_len == 4.0


def test_default_candidate_steps_use_sparse_power_of_two_anchors():
    assert build_default_adaptive_candidate_steps(1) == [1]
    assert build_default_adaptive_candidate_steps(3) == [2, 3]
    assert build_default_adaptive_candidate_steps(5) == [2, 4, 5]
    assert build_default_adaptive_candidate_steps(8) == [2, 4, 8]


def test_default_eagle_policy_uses_internal_tuned_defaults():
    params = AdaptiveSpeculativeParams.create_for_eagle_topk1(
        initial_steps=8,
        max_steps=8,
    )

    assert params.min_steps == 2
    assert params.candidate_steps == [2, 4, 8]
    assert params.ema_alpha == 1.0
    assert params.update_interval == 1
    assert params.warmup_batches == 1
    assert params.down_hysteresis == 0.0
    assert params.up_hysteresis == -0.25


def test_update_waits_for_interval_before_recomputing():
    params = AdaptiveSpeculativeParams(
        initial_steps=5,
        max_steps=5,
        ema_alpha=0.5,
        update_interval=2,
    )

    assert params.update([0, 0]) is False
    assert params.current_steps == 5
    assert math.isclose(params.ema_accept_len, 2.0)

    assert params.update([0, 0]) is True
    assert params.current_steps == 2
    assert params.current_draft_tokens == 3
    assert math.isclose(params.ema_accept_len, 1.0)


def test_warmup_batches_delay_first_recompute():
    params = AdaptiveSpeculativeParams(
        initial_steps=5,
        max_steps=5,
        ema_alpha=1.0,
        update_interval=1,
        warmup_batches=2,
    )

    assert params.update([0]) is False
    assert params.current_steps == 5

    assert params.update([0]) is False
    assert params.current_steps == 5

    assert params.update([0]) is True
    assert params.current_steps == 1


def test_update_clamps_to_min_and_max():
    params = AdaptiveSpeculativeParams(
        initial_steps=3,
        max_steps=5,
        min_steps=2,
        ema_alpha=1.0,
        update_interval=1,
    )

    assert params.update([0]) is True
    assert params.current_steps == 2
    assert params.current_draft_tokens == 3

    assert params.update([10]) is True
    assert params.current_steps == 5
    assert params.current_draft_tokens == 6


def test_empty_update_is_a_noop():
    params = AdaptiveSpeculativeParams(initial_steps=4, max_steps=5)

    assert params.update([]) is False
    assert params.current_steps == 4
    assert params.current_draft_tokens == 5
    assert params.ema_accept_len == 3.0


def test_half_step_threshold_does_not_probe_up_early():
    params = AdaptiveSpeculativeParams(
        initial_steps=3,
        max_steps=5,
        ema_alpha=0.5,
        update_interval=1,
    )

    assert params.update([1]) is True
    assert math.isclose(params.ema_accept_len, 1.5)
    assert params.current_steps == 2
    assert params.current_draft_tokens == 3


def test_mixed_workload_reduces_steps_for_hard_phase_and_recovers():
    easy_trace = [5, 4, 5, 4, 5, 3, 4, 5, 4, 5] * 10
    hard_trace = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0] * 10
    trace = easy_trace + hard_trace + easy_trace

    _, step_history = _run_adaptive_trace(trace)

    assert step_history[0] == 5
    assert min(step_history[len(easy_trace) : len(easy_trace) + len(hard_trace)]) == 1
    assert max(step_history[-20:]) == 5
    assert step_history[-1] == 5


def test_mixed_workload_surrogate_throughput_beats_any_static_config():
    easy_trace = [5, 4, 5, 4, 5, 3, 4, 5, 4, 5] * 10
    hard_trace = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0] * 10
    trace = easy_trace + hard_trace + easy_trace

    adaptive_throughput, _ = _run_adaptive_trace(trace)
    static_throughputs = {
        steps: _run_static_trace(trace, steps) for steps in range(1, 6)
    }

    assert adaptive_throughput > max(static_throughputs.values())


def test_short_phase_trace_prefers_more_responsive_tuning():
    structured = [3, 3, 3, 3] * 4
    creative = [1, 1, 1, 1] * 4
    codegen = [3, 3, 2, 3] * 4
    analytical = [2, 1, 2, 1] * 4
    trace = (
        structured
        + creative
        + codegen
        + analytical
        + structured
        + analytical
        + codegen
        + creative
        + structured
    )

    default_throughput, _ = _run_adaptive_trace(
        trace, initial_steps=3, max_steps=3, min_steps=2
    )
    tuned_throughput, _ = _run_adaptive_trace(
        trace,
        initial_steps=3,
        max_steps=3,
        min_steps=2,
        ema_alpha=0.3,
        update_interval=4,
    )

    assert tuned_throughput > default_throughput


def test_hysteresis_requires_stronger_signal_to_step_back_up():
    params = AdaptiveSpeculativeParams(
        initial_steps=2,
        max_steps=3,
        min_steps=2,
        down_hysteresis=0.05,
        up_hysteresis=0.15,
    )

    params.ema_accept_len = 1.6
    assert params._recompute_params() is False
    assert params.current_steps == 2

    params.ema_accept_len = 1.7
    assert params._recompute_params() is True
    assert params.current_steps == 3


def test_candidate_steps_snap_to_nearest_anchor():
    params = AdaptiveSpeculativeParams(
        initial_steps=8,
        max_steps=8,
        min_steps=2,
        update_interval=1,
        ema_alpha=1.0,
        candidate_steps=[2, 4, 8],
    )

    assert params.update([2]) is True
    assert params.current_steps == 4

    assert params.update([1]) is True
    assert params.current_steps == 2

    assert params.update([2]) is True
    assert params.current_steps == 4

    assert params.update([4]) is True
    assert params.current_steps == 8


def test_candidate_steps_surrogate_trace_beats_static_baselines():
    high_trace = [6] * 20
    mid_trace = [3] * 20
    low_trace = [1] * 40
    trace = high_trace + mid_trace + low_trace + mid_trace + high_trace

    adaptive_throughput, step_history = _run_adaptive_trace(
        trace,
        initial_steps=8,
        max_steps=8,
        min_steps=2,
        ema_alpha=1.0,
        update_interval=1,
        candidate_steps=[2, 4, 8],
    )
    static_throughputs = {
        steps: _run_static_trace(trace, steps) for steps in (2, 4, 8)
    }

    assert 2 in step_history
    assert 4 in step_history
    assert 8 in step_history
    assert adaptive_throughput > max(static_throughputs.values())
