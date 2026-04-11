import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    load_adaptive_config,
)

logger = logging.getLogger(__name__)


@dataclass
class SpecRuntimeState:
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    draft_attn_backend: Optional[object]
    draft_extend_attn_backend: Optional[object]
    cuda_graph_runner: Optional[object]
    cuda_graph_runner_for_draft_extend: Optional[object]
    target_attn_backend: object
    target_graph_runner: Optional[object]


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> Any: ...

    def apply_runtime_state(self, state: Any) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → state object
      - ``apply_runtime_state(state)`` → apply it to the worker

    The worker only needs to:
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      2. Call ``on_verify_complete(accept_lengths)`` after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: Optional[str] = None):
        self.worker = worker
        cfg = load_adaptive_config(config_path)
        default_candidate_steps = getattr(worker, "adaptive_candidate_steps", None)
        if default_candidate_steps is not None and "candidate_steps" not in cfg:
            cfg = {**cfg, "candidate_steps": default_candidate_steps}
        initial_value = getattr(worker, "adaptive_initial_value", worker.speculative_num_steps)
        self.params = AdaptiveSpeculativeParams(
            initial_steps=initial_value,
            config=cfg,
        )
        self._states: Dict[int, Any] = {}

    @property
    def candidate_steps(self):
        return self.params.candidate_steps

    @property
    def states(self) -> Dict[int, Any]:
        return self._states

    def register(self, state: Any, steps: Optional[int] = None):
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def _get_runtime_params(self, key: int) -> tuple[int, int]:
        resolver = getattr(self.worker, "get_adaptive_runtime_params", None)
        if resolver is not None:
            return resolver(key)
        return key, key + 1

    def init_states(self):
        """Build and register runtime states for all candidate steps."""
        for key in self.params.candidate_steps:
            if key in self._states:
                continue
            speculative_num_steps, speculative_num_draft_tokens = (
                self._get_runtime_params(key)
            )
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )
            self._states[key] = state
        self._activate(self.params.current_steps)

    def on_verify_complete(self, accept_lengths: List[int]) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        if self.params.update(accept_lengths):
            self._activate(self.params.current_steps)

    def _activate(self, speculative_num_steps: int):
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)


def maybe_init_adaptive_controller(
    worker: AdaptiveSpecWorker, config_path: Optional[str] = None
) -> Optional[AdaptiveController]:
    if not getattr(worker.server_args, "speculative_adaptive", False):
        worker.adaptive_controller = None
        return None

    candidate_steps = getattr(worker, "adaptive_candidate_steps", None)
    if candidate_steps is not None and len(set(candidate_steps)) < 2:
        logger.warning(
            "Disabling speculative_adaptive because fewer than 2 adaptive "
            f"candidates are available: {candidate_steps}"
        )
        worker.adaptive_controller = None
        return None

    worker.adaptive_controller = AdaptiveController(worker, config_path=config_path)
    return worker.adaptive_controller


def maybe_register_adaptive_state(worker: AdaptiveSpecWorker, state: Any) -> None:
    controller = getattr(worker, "adaptive_controller", None)
    if controller is None:
        return

    controller.register(state, steps=getattr(worker, "adaptive_initial_value", None))
    controller.init_states()
