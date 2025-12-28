"""Dynamic disaggregation mode controller.

Supports runtime switching between NULL, PREFILL, and DECODE modes
with snapshot-based rollback on failure.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class TransitionState(Enum):
    """Mode transition states."""

    IDLE = "idle"  # Normal running, accepts requests
    CHECKING = "checking"  # Checking if switch is possible
    SWITCHING = "switching"  # Executing switch, rejecting requests
    ROLLBACK = "rollback"  # Rolling back after failure


class ModeTransitionError(Exception):
    """Exception raised during mode transition."""

    def __init__(self, message: str, can_retry: bool = False):
        super().__init__(message)
        self.can_retry = can_retry


@dataclass
class ModeSnapshot:
    """State snapshot for rollback support."""

    mode: DisaggregationMode
    radix_cache_disabled: bool
    bootstrap_server_running: bool
    prefill_initialized: bool = False
    decode_initialized: bool = False
    extra_state: Dict[str, Any] = field(default_factory=dict)


class ModeController:
    """
    Controls dynamic switching of disaggregation modes.

    Thread-safe implementation with snapshot-based rollback.
    """

    def __init__(self, scheduler: "Scheduler"):
        self._scheduler = scheduler
        self._lock = threading.RLock()

        # Current mode from server args
        self._current_mode = DisaggregationMode(
            scheduler.server_args.disaggregation_mode
        )
        self._target_mode: Optional[DisaggregationMode] = None
        self._transition_state = TransitionState.IDLE
        self._snapshot: Optional[ModeSnapshot] = None
        self._last_error: Optional[str] = None

        # Lazy initialization flags
        self._prefill_initialized = self._current_mode == DisaggregationMode.PREFILL
        self._decode_initialized = self._current_mode == DisaggregationMode.DECODE

        # Bootstrap server reference
        self._bootstrap_server: Optional[Any] = None
        self._bootstrap_server_factory: Optional[Callable] = None
        self._bootstrap_server_shutdown: Optional[Callable] = None

        # Configure bootstrap server callbacks
        self._setup_bootstrap_callbacks()

    def _setup_bootstrap_callbacks(self) -> None:
        """Set up bootstrap server factory and shutdown callbacks."""
        from sglang.srt.managers.disagg_service import (
            create_bootstrap_server,
            shutdown_bootstrap_server,
        )

        server_args = self._scheduler.server_args
        transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

        # Factory: creates bootstrap server
        self._bootstrap_server_factory = partial(
            create_bootstrap_server,
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
            transfer_backend=transfer_backend,
            node_rank=server_args.node_rank,
        )

        # Shutdown: stops bootstrap server
        self._bootstrap_server_shutdown = shutdown_bootstrap_server

    @property
    def current_mode(self) -> DisaggregationMode:
        """Current disaggregation mode."""
        with self._lock:
            return self._current_mode

    @property
    def transition_state(self) -> TransitionState:
        """Current transition state."""
        with self._lock:
            return self._transition_state

    def can_accept_requests(self) -> bool:
        """Check if new requests can be accepted."""
        with self._lock:
            return self._transition_state == TransitionState.IDLE

    def set_bootstrap_server_callbacks(
        self,
        factory: Callable[[], Any],
        shutdown: Callable[[Any], None],
    ) -> None:
        """
        Set callbacks for bootstrap server lifecycle management.

        Args:
            factory: Function to create and start bootstrap server
            shutdown: Function to shutdown bootstrap server
        """
        with self._lock:
            self._bootstrap_server_factory = factory
            self._bootstrap_server_shutdown = shutdown

    def set_bootstrap_server(self, server: Any) -> None:
        """Set existing bootstrap server reference."""
        with self._lock:
            self._bootstrap_server = server

    def request_mode_switch(self, target_mode: DisaggregationMode) -> Tuple[bool, str]:
        """
        Request a mode switch.

        Args:
            target_mode: Target disaggregation mode

        Returns:
            (success, message) tuple
        """
        with self._lock:
            # Validate current state
            if self._transition_state != TransitionState.IDLE:
                return (
                    False,
                    f"Switch in progress, state: {self._transition_state.value}",
                )

            if self._current_mode == target_mode:
                return True, f"Already in {target_mode.value} mode"

            # Enter checking state
            self._transition_state = TransitionState.CHECKING
            self._target_mode = target_mode
            self._last_error = None

            # Check if queues are empty
            if not self._are_queues_empty():
                self._transition_state = TransitionState.IDLE
                self._target_mode = None
                return False, "Queues not empty, wait for requests to complete"

            # Execute switch
            try:
                self._execute_switch()
                return True, f"Switched to {target_mode.value} mode"
            except ModeTransitionError as e:
                self._last_error = str(e)
                return False, f"Switch failed: {e}"
            except Exception as e:
                self._last_error = str(e)
                logger.exception("Unexpected error during mode switch")
                return False, f"Switch failed (unexpected): {e}"

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        with self._lock:
            return {
                "current_mode": self._current_mode.value,
                "transition_state": self._transition_state.value,
                "target_mode": self._target_mode.value if self._target_mode else None,
                "last_error": self._last_error,
                "prefill_initialized": self._prefill_initialized,
                "decode_initialized": self._decode_initialized,
                "bootstrap_server_running": self._bootstrap_server is not None,
            }

    def _are_queues_empty(self) -> bool:
        """Check if all request queues are empty."""
        s = self._scheduler

        # Common queues
        if len(s.waiting_queue) > 0:
            logger.debug("waiting_queue not empty")
            return False

        if hasattr(s, "running_batch") and not s.running_batch.is_empty():
            logger.debug("running_batch not empty")
            return False

        # PREFILL mode queues
        if hasattr(s, "disagg_prefill_bootstrap_queue"):
            if len(s.disagg_prefill_bootstrap_queue.queue) > 0:
                logger.debug("prefill bootstrap queue not empty")
                return False

        if hasattr(s, "disagg_prefill_inflight_queue"):
            if len(s.disagg_prefill_inflight_queue) > 0:
                logger.debug("prefill inflight queue not empty")
                return False

        # DECODE mode queues
        if hasattr(s, "disagg_decode_prealloc_queue"):
            if not s.disagg_decode_prealloc_queue.is_empty():
                logger.debug("decode prealloc queue not empty")
                return False

        if hasattr(s, "disagg_decode_transfer_queue"):
            if not s.disagg_decode_transfer_queue.is_empty():
                logger.debug("decode transfer queue not empty")
                return False

        return True

    def _execute_switch(self) -> None:
        """Execute the mode switch with rollback on failure."""
        self._transition_state = TransitionState.SWITCHING

        # Create snapshot for rollback
        self._create_snapshot()

        try:
            # Handle cache strategy change
            self._handle_cache_strategy()

            # Initialize target mode components
            self._init_target_components()

            # Manage bootstrap server
            self._manage_bootstrap_server()

            # Update current mode
            old_mode = self._current_mode
            self._current_mode = self._target_mode
            self._scheduler.disaggregation_mode = self._target_mode

            # Cleanup
            self._target_mode = None
            self._snapshot = None
            self._transition_state = TransitionState.IDLE

            logger.info(
                f"Mode switched: {old_mode.value} -> {self._current_mode.value}"
            )

        except Exception as e:
            logger.error(f"Mode switch failed: {e}, initiating rollback")
            self._rollback()
            raise ModeTransitionError(str(e))

    def _create_snapshot(self) -> None:
        """Create state snapshot for potential rollback."""
        self._snapshot = ModeSnapshot(
            mode=self._current_mode,
            radix_cache_disabled=self._scheduler.server_args.disable_radix_cache,
            bootstrap_server_running=self._bootstrap_server is not None,
            prefill_initialized=self._prefill_initialized,
            decode_initialized=self._decode_initialized,
        )
        logger.debug(f"Snapshot created: mode={self._snapshot.mode.value}")

    def _handle_cache_strategy(self) -> None:
        """Handle cache strategy changes between modes."""
        # DECODE mode requires radix cache to be disabled
        if self._target_mode == DisaggregationMode.DECODE:
            if not self._scheduler.server_args.disable_radix_cache:
                # Check if tree cache is empty
                if hasattr(self._scheduler, "tree_cache"):
                    tree_cache = self._scheduler.tree_cache
                    # Check if cache has entries (implementation dependent)
                    if (
                        hasattr(tree_cache, "total_size")
                        and tree_cache.total_size() > 0
                    ):
                        raise ModeTransitionError(
                            "Radix cache not empty, cannot switch to DECODE. "
                            "Wait for cache eviction or flush manually.",
                            can_retry=True,
                        )

                self._scheduler.server_args.disable_radix_cache = True
                logger.info("Radix cache disabled for DECODE mode")

    def _init_target_components(self) -> None:
        """Initialize components for target mode."""
        if self._target_mode == DisaggregationMode.PREFILL:
            if not self._prefill_initialized:
                logger.info("Initializing PREFILL components...")
                self._scheduler._init_prefill_disaggregation()
                self._prefill_initialized = True
                logger.info("PREFILL components initialized")

        elif self._target_mode == DisaggregationMode.DECODE:
            if not self._decode_initialized:
                logger.info("Initializing DECODE components...")
                self._scheduler._init_decode_disaggregation()
                self._decode_initialized = True
                logger.info("DECODE components initialized")

    def _manage_bootstrap_server(self) -> None:
        """Start or stop bootstrap server based on mode transition."""
        # Start server when switching TO PREFILL
        if self._target_mode == DisaggregationMode.PREFILL:
            if self._bootstrap_server is None:
                if self._bootstrap_server_factory is None:
                    raise ModeTransitionError("Bootstrap server factory not configured")
                try:
                    self._bootstrap_server = self._bootstrap_server_factory()
                    logger.info("Bootstrap server started")
                except Exception as e:
                    raise ModeTransitionError(f"Failed to start bootstrap server: {e}")

        # Stop server when switching FROM PREFILL
        elif self._current_mode == DisaggregationMode.PREFILL:
            if self._bootstrap_server is not None:
                if self._bootstrap_server_shutdown is not None:
                    try:
                        self._bootstrap_server_shutdown(self._bootstrap_server)
                        logger.info("Bootstrap server stopped")
                    except Exception as e:
                        logger.warning(f"Failed to stop bootstrap server: {e}")
                self._bootstrap_server = None

    def _rollback(self) -> None:
        """Rollback to previous state after failed switch."""
        self._transition_state = TransitionState.ROLLBACK

        if self._snapshot is None:
            logger.error("Cannot rollback: no snapshot available")
            self._transition_state = TransitionState.IDLE
            return

        try:
            # Restore cache setting
            self._scheduler.server_args.disable_radix_cache = (
                self._snapshot.radix_cache_disabled
            )

            # Restore bootstrap server state
            if self._snapshot.bootstrap_server_running:
                if self._bootstrap_server is None and self._bootstrap_server_factory:
                    try:
                        self._bootstrap_server = self._bootstrap_server_factory()
                    except Exception as e:
                        logger.error(f"Failed to restore bootstrap server: {e}")
            else:
                if self._bootstrap_server is not None:
                    if self._bootstrap_server_shutdown:
                        try:
                            self._bootstrap_server_shutdown(self._bootstrap_server)
                        except Exception as e:
                            logger.warning(
                                f"Failed to stop bootstrap server during rollback: {e}"
                            )
                    self._bootstrap_server = None

            # Restore initialization flags
            self._prefill_initialized = self._snapshot.prefill_initialized
            self._decode_initialized = self._snapshot.decode_initialized

            logger.info(
                f"Rollback complete, restored to {self._snapshot.mode.value} mode"
            )

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
        finally:
            self._target_mode = None
            self._snapshot = None
            self._transition_state = TransitionState.IDLE

    def mark_prefill_initialized(self) -> None:
        """Mark PREFILL components as initialized (called during startup)."""
        with self._lock:
            self._prefill_initialized = True

    def mark_decode_initialized(self) -> None:
        """Mark DECODE components as initialized (called during startup)."""
        with self._lock:
            self._decode_initialized = True
