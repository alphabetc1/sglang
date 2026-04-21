import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    load_adaptive_config,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )
    from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
        MultiLayerEagleDraftExtendCudaGraphRunner,
        MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime state hierarchy
# ---------------------------------------------------------------------------


@dataclass
class SpecRuntimeState:
    """Base runtime state for adaptive speculative decoding.

    Every worker needs at minimum the configuration (num_steps, draft_tokens)
    and target-side resources (attention backend + CUDA graph runner) that depend
    on those values.
    """

    # -- Configuration --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Verify stage: target model one-pass verification --
    target_attn_backend: "AttentionBackend | None" = None
    target_graph_runner: "CudaGraphRunner | CPUGraphRunner | None" = None


@dataclass
class EagleRuntimeState(SpecRuntimeState):
    """Runtime state for single-layer EAGLE (v1/v2) and Standalone workers."""

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None" = None
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None" = None

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None" = None
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None" = None


@dataclass
class MultiLayerEagleRuntimeState(SpecRuntimeState):
    """Runtime state for MultiLayerEagleWorker (v1) — per-step backend/graph lists."""

    draft_extend_attn_backend_list: "list[AttentionBackend | None] | None" = None
    cuda_graph_runner_for_draft_extend_list: (
        "list[MultiLayerEagleDraftExtendCudaGraphRunner] | None"
    ) = None


@dataclass
class MultiLayerEagleV2RuntimeState(SpecRuntimeState):
    """Runtime state for MultiLayerEagleWorkerV2 — per-step backends + single multi-step graph."""

    draft_extend_attn_backend_list: "list[AttentionBackend | None] | None" = None
    cuda_graph_runner_for_draft_extend: (
        "MultiLayerEagleMultiStepDraftExtendCudaGraphRunner | None"
    ) = None


# ---------------------------------------------------------------------------
# Shared helper: build target-side resources
# ---------------------------------------------------------------------------


def build_target_runtime(
    target_model_runner,
    server_args,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
) -> tuple:
    """Build target attention backend + CUDA graph runner for a given step config.

    Returns:
        (target_attn_backend, target_graph_runner)
    """
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    backup_init = target_model_runner.init_new_workspace
    try:
        target_attn_backend = target_model_runner._get_attention_backend(
            init_new_workspace=True
        )
    finally:
        target_model_runner.init_new_workspace = backup_init

    target_graph_runner = None
    if not server_args.disable_cuda_graph:
        target_graph_runner = CudaGraphRunner(
            target_model_runner,
            attn_backend=target_attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
    return target_attn_backend, target_graph_runner


# ---------------------------------------------------------------------------
# Adaptive protocol + controller
# ---------------------------------------------------------------------------


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → runtime state
      - ``apply_runtime_state(state)`` → apply it to the worker

    The worker only needs to:
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      2. Call ``on_verify_complete(accept_lengths)`` after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        cfg = load_adaptive_config(config_path)
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            config=cfg,
        )
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self) -> None:
        """Build and register runtime states for all candidate steps."""
        for steps in self.params.candidate_steps:
            if steps in self._states:
                continue
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
            )
            self._states[steps] = state
        self._activate(self.params.current_steps)

    def on_verify_complete(self, accept_lengths: list[int]) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        if self.params.update(accept_lengths):
            self._activate(self.params.current_steps)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
