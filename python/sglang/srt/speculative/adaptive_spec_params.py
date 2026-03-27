"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_adaptive_config(path: Optional[str]) -> Dict[str, Any]:
    """Load adaptive speculative config from a JSON file.

    The file may contain any subset of the following keys:
        ema_alpha, update_interval, warmup_batches,
        down_hysteresis, up_hysteresis, candidate_steps

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
    """

    @classmethod
    def create(
        cls,
        initial_steps: int,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> "AdaptiveSpeculativeParams":
        default_candidate_steps = [1, 3, 7]
        cfg = config_overrides or {}
        candidate_steps = cfg.get("candidate_steps", default_candidate_steps)
        return cls(
            initial_steps=initial_steps,
            max_steps=candidate_steps[-1],
            min_steps=candidate_steps[0],
            ema_alpha=cfg.get("ema_alpha", 0.2),
            update_interval=cfg.get("update_interval", 5),
            warmup_batches=cfg.get("warmup_batches", 10),
            down_hysteresis=cfg.get("down_hysteresis", 0.0),
            up_hysteresis=cfg.get("up_hysteresis", -0.25),
            candidate_steps=candidate_steps,
        )

    def __init__(
        self,
        initial_steps: int,
        max_steps: int,
        min_steps: int = 1,
        ema_alpha: float = 0.2,
        update_interval: int = 5,
        down_hysteresis: float = 0.0,
        up_hysteresis: float = -0.25,
        candidate_steps: Optional[List[int]] = None,
        warmup_batches: int = 10,
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.ema_alpha = ema_alpha
        self.update_interval = update_interval
        self.warmup_batches = warmup_batches
        self.down_hysteresis = down_hysteresis
        self.up_hysteresis = up_hysteresis
        if candidate_steps is None:
            self.candidate_steps = None
        else:
            normalized_candidate_steps = sorted(set(candidate_steps))
            assert normalized_candidate_steps, "candidate_steps must not be empty"
            assert all(
                min_steps <= step <= max_steps for step in normalized_candidate_steps
            ), "candidate_steps must be within [min_steps, max_steps]"
            self.candidate_steps = normalized_candidate_steps

        self.current_steps = min(max(initial_steps, min_steps), max_steps)
        if self.candidate_steps is not None:
            self.current_steps = min(
                self.candidate_steps,
                key=lambda step: (abs(step - self.current_steps), -step),
            )

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

        logger.info(
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, range=[{self.min_steps}, {self.max_steps}], "
            f"ema_alpha={self.ema_alpha}, update_interval={self.update_interval}, "
            f"warmup_batches={self.warmup_batches}, "
            f"down_hysteresis={self.down_hysteresis}, up_hysteresis={self.up_hysteresis}, "
            f"candidate_steps={self.candidate_steps}"
        )

    def update(self, accept_lengths: List[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            accept_lengths: Per-request accepted draft token counts from last verify.
        """
        if not accept_lengths:
            return False

        batch_avg = sum(accept_lengths) / len(accept_lengths)
        self.ema_accept_len = (
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        if self.candidate_steps is not None:
            current_idx = self.candidate_steps.index(old_steps)

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

            target = self.candidate_steps[current_idx]
        else:
            target = old_steps

            # Base boundaries come from the original "round(ema_accept_len) + 1"
            # policy:
            # - drop from N to N - 1 when EMA is at or below N - 1.5
            # - probe up from N to N + 1 when EMA is above N - 0.5
            #
            # Hysteresis shifts these thresholds asymmetrically so step reductions can
            # happen with less evidence than step increases, which reduces oscillation
            # on short mixed workloads without changing the zero-hysteresis behavior.
            while target > self.min_steps:
                drop_threshold = target - 1.5 + self.down_hysteresis
                if self.ema_accept_len <= drop_threshold:
                    target -= 1
                else:
                    break

            while target < self.max_steps:
                rise_threshold = target - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    target += 1
                else:
                    break

        if target != old_steps:
            self.current_steps = target
            logger.info(
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})"
            )
            return True
        return False
