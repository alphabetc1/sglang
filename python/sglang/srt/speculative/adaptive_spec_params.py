"""Adaptive speculative decoding parameters.

Adjust speculative runtime state from observed acceptance lengths, with optional
throughput-aware guards to reject or revert unprofitable step changes.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_adaptive_config(path: Optional[str]) -> Dict[str, Any]:
    """Load adaptive speculative config from a JSON file.

    The file may contain any subset of the following keys:
        ema_alpha, update_interval, warmup_batches,
        down_hysteresis, up_hysteresis, candidate_steps,
        throughput_aware, throughput_ema_alpha,
        upshift_history_tolerance, switch_revert_tolerance,
        switch_guard_batches

    Returns an empty dict when *path* is ``None``.
    """
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"speculative_adaptive_config must be a JSON object, got {type(cfg).__name__}"
        )
    return cfg


class AdaptiveSpeculativeParams:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - Optionally keeps a per-step throughput EMA to avoid sticky slow states
    """

    def __init__(
        self,
        initial_steps: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        # TODO: Wider range of candidate_steps (once lazy init is supported).
        self.candidate_steps = sorted(set(cfg.get("candidate_steps", [1, 3, 7])))
        assert (
            len(self.candidate_steps) >= 2
        ), "candidate_steps must have at least 2 distinct values"

        self.min_steps = self.candidate_steps[0]
        self.max_steps = self.candidate_steps[-1]
        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", 0.0)
        self.up_hysteresis = cfg.get("up_hysteresis", -0.25)
        self.min_state_dwell_updates = cfg.get("min_state_dwell_updates", 0)
        self.throughput_aware = cfg.get("throughput_aware", False)
        self.throughput_ema_alpha = cfg.get("throughput_ema_alpha", self.ema_alpha)
        self.upshift_history_tolerance = cfg.get("upshift_history_tolerance", 0.05)
        self.switch_revert_tolerance = cfg.get("switch_revert_tolerance", 0.05)
        self.switch_guard_batches = cfg.get("switch_guard_batches", 2)

        self.current_steps = min(
            self.candidate_steps,
            key=lambda step: (abs(step - initial_steps), -step),
        )

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0
        self._decision_updates_since_switch = 0
        self._last_observation_time: Optional[float] = None
        self._step_throughput_ema: Dict[int, float] = {}
        self._step_throughput_samples: Dict[int, int] = defaultdict(int)
        self._pending_revert_step: Optional[int] = None
        self._pending_revert_baseline: Optional[float] = None
        self._batches_since_switch = 0

        logger.info(
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self.candidate_steps}"
        )

    def update(self, accept_lengths: List[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            accept_lengths: Per-request accepted draft token counts from last verify.
        """
        if not accept_lengths:
            return False

        self._observe_batch_throughput(accept_lengths)

        batch_avg = sum(accept_lengths) / len(accept_lengths)
        self.ema_accept_len = (
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._maybe_revert_on_throughput_drop():
            return True

        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        changed = self._recompute_params()
        if changed:
            self._decision_updates_since_switch = 0
        else:
            self._decision_updates_since_switch += 1
        return changed

    def _observe_batch_throughput(self, accept_lengths: List[int]) -> None:
        now = time.perf_counter()
        prev_time = self._last_observation_time
        self._last_observation_time = now
        if not self.throughput_aware or prev_time is None:
            return

        elapsed = now - prev_time
        if elapsed <= 0:
            return

        # Each speculative verify emits the accepted draft tokens plus one
        # verified target token per live request.
        output_tokens = sum(accept_lengths) + len(accept_lengths)
        observed_tps = output_tokens / elapsed
        prev_tps = self._step_throughput_ema.get(self.current_steps)
        if prev_tps is None:
            self._step_throughput_ema[self.current_steps] = observed_tps
        else:
            self._step_throughput_ema[self.current_steps] = (
                (1 - self.throughput_ema_alpha) * prev_tps
                + self.throughput_ema_alpha * observed_tps
            )
        self._step_throughput_samples[self.current_steps] += 1
        if self._pending_revert_step is not None:
            self._batches_since_switch += 1

    def _maybe_revert_on_throughput_drop(self) -> bool:
        if (
            not self.throughput_aware
            or self._pending_revert_step is None
            or self._pending_revert_baseline is None
            or self._batches_since_switch < self.switch_guard_batches
        ):
            return False

        current_tps = self._step_throughput_ema.get(self.current_steps)
        if current_tps is None:
            return False

        baseline_tps = self._pending_revert_baseline
        if current_tps < baseline_tps * (1 - self.switch_revert_tolerance):
            revert_step = self._pending_revert_step
            old_steps = self.current_steps
            self.current_steps = revert_step
            self._decision_updates_since_switch = 0
            self._clear_pending_revert()
            logger.info(
                "Adaptive spec throughput revert: steps %s -> %s "
                "(current_tps=%.1f, baseline_tps=%.1f)",
                old_steps,
                revert_step,
                current_tps,
                baseline_tps,
            )
            return True

        self._clear_pending_revert()
        return False

    def _clear_pending_revert(self) -> None:
        self._pending_revert_step = None
        self._pending_revert_baseline = None
        self._batches_since_switch = 0

    def _apply_throughput_history_guard(self, old_steps: int, target: int) -> int:
        if not self.throughput_aware or target <= old_steps:
            return target

        current_tps = self._step_throughput_ema.get(old_steps)
        target_tps = self._step_throughput_ema.get(target)
        if current_tps is None or target_tps is None:
            return target

        if target_tps < current_tps * (1 - self.upshift_history_tolerance):
            logger.info(
                "Adaptive spec throughput guard keeps steps at %s instead of %s "
                "(current_tps=%.1f, target_tps=%.1f)",
                old_steps,
                target,
                current_tps,
                target_tps,
            )
            return old_steps
        return target

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop_threshold = prev_step - 0.5 + self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        while current_idx < len(self.candidate_steps) - 1:
            current_step = self.candidate_steps[current_idx]
            rise_threshold = current_step - 0.5 + self.up_hysteresis
            if self.ema_accept_len > rise_threshold:
                current_idx += 1
            else:
                break

        target = self._apply_throughput_history_guard(
            old_steps, self.candidate_steps[current_idx]
        )

        if target != old_steps:
            if self._decision_updates_since_switch < self.min_state_dwell_updates:
                return False
            self.current_steps = target
            if self.throughput_aware:
                baseline_tps = self._step_throughput_ema.get(old_steps)
                if baseline_tps is None:
                    self._clear_pending_revert()
                else:
                    self._pending_revert_step = old_steps
                    self._pending_revert_baseline = baseline_tps
                    self._batches_since_switch = 0
            logger.info(
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})"
            )
            return True
        return False
