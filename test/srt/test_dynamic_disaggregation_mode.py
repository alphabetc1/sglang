"""
Unit tests for dynamic disaggregation mode switching.

Tests the ModeController, IO structures, and related functionality.
"""

import unittest
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.mode_controller import (
    ModeController,
    ModeSnapshot,
    ModeTransitionError,
    TransitionState,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    ModeStatusReqInput,
    ModeStatusReqOutput,
    ModeSwitchReqInput,
    ModeSwitchReqOutput,
)


class MockServerArgs:
    """Mock ServerArgs for testing."""

    def __init__(
        self,
        disaggregation_mode: str = "null",
        disable_radix_cache: bool = False,
        enable_dynamic_disaggregation: bool = True,
        disaggregation_transfer_backend: str = "mooncake",
        host: str = "localhost",
        disaggregation_bootstrap_port: int = 30100,
        node_rank: int = 0,
    ):
        self.disaggregation_mode = disaggregation_mode
        self.disable_radix_cache = disable_radix_cache
        self.enable_dynamic_disaggregation = enable_dynamic_disaggregation
        self.disaggregation_transfer_backend = disaggregation_transfer_backend
        self.host = host
        self.disaggregation_bootstrap_port = disaggregation_bootstrap_port
        self.node_rank = node_rank


class MockScheduleBatch:
    """Mock ScheduleBatch for testing."""

    def __init__(self, empty: bool = True):
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


class MockQueue:
    """Mock queue for testing."""

    def __init__(self, items: Optional[List[Any]] = None):
        self.queue = items or []

    def __len__(self):
        return len(self.queue)

    def is_empty(self) -> bool:
        return len(self.queue) == 0


class MockScheduler:
    """Mock Scheduler for testing ModeController."""

    def __init__(
        self,
        server_args: Optional[MockServerArgs] = None,
        waiting_queue_empty: bool = True,
        running_batch_empty: bool = True,
    ):
        self.server_args = server_args or MockServerArgs()
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.waiting_queue = [] if waiting_queue_empty else [MagicMock()]
        self.running_batch = MockScheduleBatch(running_batch_empty)

        # Mock methods for initialization
        self._init_prefill_disaggregation = MagicMock()
        self._init_decode_disaggregation = MagicMock()


class TestTransitionState(unittest.TestCase):
    """Test TransitionState enum."""

    def test_enum_values(self):
        """Test that all expected states exist."""
        self.assertEqual(TransitionState.IDLE.value, "idle")
        self.assertEqual(TransitionState.CHECKING.value, "checking")
        self.assertEqual(TransitionState.SWITCHING.value, "switching")
        self.assertEqual(TransitionState.ROLLBACK.value, "rollback")

    def test_enum_count(self):
        """Test that we have exactly 4 states."""
        self.assertEqual(len(TransitionState), 4)


class TestModeSnapshot(unittest.TestCase):
    """Test ModeSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a mode snapshot."""
        snapshot = ModeSnapshot(
            mode=DisaggregationMode.PREFILL,
            radix_cache_disabled=True,
            bootstrap_server_running=True,
            prefill_initialized=True,
            decode_initialized=False,
        )

        self.assertEqual(snapshot.mode, DisaggregationMode.PREFILL)
        self.assertTrue(snapshot.radix_cache_disabled)
        self.assertTrue(snapshot.bootstrap_server_running)
        self.assertTrue(snapshot.prefill_initialized)
        self.assertFalse(snapshot.decode_initialized)

    def test_snapshot_defaults(self):
        """Test snapshot default values."""
        snapshot = ModeSnapshot(
            mode=DisaggregationMode.NULL,
            radix_cache_disabled=False,
            bootstrap_server_running=False,
        )

        self.assertFalse(snapshot.prefill_initialized)
        self.assertFalse(snapshot.decode_initialized)
        self.assertEqual(snapshot.extra_state, {})


class TestModeController(unittest.TestCase):
    """Test ModeController class."""

    def setUp(self):
        """Set up test fixtures."""
        self.server_args = MockServerArgs()
        self.scheduler = MockScheduler(self.server_args)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_init_null_mode(self, mock_setup):
        """Test initialization in NULL mode."""
        controller = ModeController(self.scheduler)

        self.assertEqual(controller.current_mode, DisaggregationMode.NULL)
        self.assertEqual(controller.transition_state, TransitionState.IDLE)
        self.assertFalse(controller._prefill_initialized)
        self.assertFalse(controller._decode_initialized)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_init_prefill_mode(self, mock_setup):
        """Test initialization in PREFILL mode."""
        self.server_args.disaggregation_mode = "prefill"
        self.scheduler = MockScheduler(self.server_args)
        controller = ModeController(self.scheduler)

        self.assertEqual(controller.current_mode, DisaggregationMode.PREFILL)
        self.assertTrue(controller._prefill_initialized)
        self.assertFalse(controller._decode_initialized)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_init_decode_mode(self, mock_setup):
        """Test initialization in DECODE mode."""
        self.server_args.disaggregation_mode = "decode"
        self.scheduler = MockScheduler(self.server_args)
        controller = ModeController(self.scheduler)

        self.assertEqual(controller.current_mode, DisaggregationMode.DECODE)
        self.assertFalse(controller._prefill_initialized)
        self.assertTrue(controller._decode_initialized)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_can_accept_requests_idle(self, mock_setup):
        """Test can_accept_requests returns True when IDLE."""
        controller = ModeController(self.scheduler)
        self.assertTrue(controller.can_accept_requests())

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_can_accept_requests_switching(self, mock_setup):
        """Test can_accept_requests returns False when SWITCHING."""
        controller = ModeController(self.scheduler)
        controller._transition_state = TransitionState.SWITCHING
        self.assertFalse(controller.can_accept_requests())

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_switch_same_mode(self, mock_setup):
        """Test switching to the same mode returns success."""
        controller = ModeController(self.scheduler)
        success, msg = controller.request_mode_switch(DisaggregationMode.NULL)

        self.assertTrue(success)
        self.assertIn("Already in", msg)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_switch_in_progress(self, mock_setup):
        """Test switching when another switch is in progress."""
        controller = ModeController(self.scheduler)
        controller._transition_state = TransitionState.SWITCHING

        success, msg = controller.request_mode_switch(DisaggregationMode.PREFILL)

        self.assertFalse(success)
        self.assertIn("in progress", msg)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_switch_queues_not_empty(self, mock_setup):
        """Test switching fails when queues are not empty."""
        self.scheduler.waiting_queue = [MagicMock()]  # Non-empty queue
        controller = ModeController(self.scheduler)

        success, msg = controller.request_mode_switch(DisaggregationMode.PREFILL)

        self.assertFalse(success)
        self.assertIn("not empty", msg)
        self.assertEqual(controller.transition_state, TransitionState.IDLE)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_get_status(self, mock_setup):
        """Test get_status returns correct information."""
        controller = ModeController(self.scheduler)
        status = controller.get_status()

        self.assertEqual(status["current_mode"], "null")
        self.assertEqual(status["transition_state"], "idle")
        self.assertIsNone(status["target_mode"])
        self.assertIsNone(status["last_error"])
        self.assertFalse(status["prefill_initialized"])
        self.assertFalse(status["decode_initialized"])
        self.assertFalse(status["bootstrap_server_running"])

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_mark_initialized(self, mock_setup):
        """Test mark_prefill_initialized and mark_decode_initialized."""
        controller = ModeController(self.scheduler)

        self.assertFalse(controller._prefill_initialized)
        controller.mark_prefill_initialized()
        self.assertTrue(controller._prefill_initialized)

        self.assertFalse(controller._decode_initialized)
        controller.mark_decode_initialized()
        self.assertTrue(controller._decode_initialized)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_set_bootstrap_server_callbacks(self, mock_setup):
        """Test setting bootstrap server callbacks."""
        controller = ModeController(self.scheduler)

        factory = MagicMock()
        shutdown = MagicMock()

        controller.set_bootstrap_server_callbacks(factory, shutdown)

        self.assertEqual(controller._bootstrap_server_factory, factory)
        self.assertEqual(controller._bootstrap_server_shutdown, shutdown)

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_set_bootstrap_server(self, mock_setup):
        """Test setting existing bootstrap server."""
        controller = ModeController(self.scheduler)
        server = MagicMock()

        controller.set_bootstrap_server(server)

        self.assertEqual(controller._bootstrap_server, server)
        self.assertTrue(controller.get_status()["bootstrap_server_running"])


class TestModeTransitionError(unittest.TestCase):
    """Test ModeTransitionError exception."""

    def test_error_message(self):
        """Test error message."""
        error = ModeTransitionError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertFalse(error.can_retry)

    def test_error_can_retry(self):
        """Test error with can_retry flag."""
        error = ModeTransitionError("Retryable error", can_retry=True)
        self.assertTrue(error.can_retry)


class TestIOStructs(unittest.TestCase):
    """Test IO structures for mode switching."""

    def test_mode_switch_req_input(self):
        """Test ModeSwitchReqInput."""
        req = ModeSwitchReqInput(target_mode="prefill")
        self.assertEqual(req.target_mode, "prefill")

    def test_mode_switch_req_output(self):
        """Test ModeSwitchReqOutput."""
        output = ModeSwitchReqOutput(
            success=True,
            message="Switched to prefill mode",
            current_mode="prefill",
            transition_state="idle",
        )
        self.assertTrue(output.success)
        self.assertEqual(output.current_mode, "prefill")

    def test_mode_status_req_input(self):
        """Test ModeStatusReqInput."""
        req = ModeStatusReqInput()
        self.assertIsNotNone(req)

    def test_mode_status_req_output(self):
        """Test ModeStatusReqOutput."""
        output = ModeStatusReqOutput(
            current_mode="decode",
            transition_state="idle",
            target_mode=None,
            last_error=None,
            prefill_initialized=False,
            decode_initialized=True,
            bootstrap_server_running=False,
            dynamic_mode_enabled=True,
        )
        self.assertEqual(output.current_mode, "decode")
        self.assertTrue(output.dynamic_mode_enabled)
        self.assertTrue(output.decode_initialized)


class TestQueueEmptyCheck(unittest.TestCase):
    """Test queue empty checking logic."""

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_waiting_queue_not_empty(self, mock_setup):
        """Test detection of non-empty waiting queue."""
        scheduler = MockScheduler(waiting_queue_empty=False)
        controller = ModeController(scheduler)

        self.assertFalse(controller._are_queues_empty())

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_running_batch_not_empty(self, mock_setup):
        """Test detection of non-empty running batch."""
        scheduler = MockScheduler(running_batch_empty=False)
        controller = ModeController(scheduler)

        self.assertFalse(controller._are_queues_empty())

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_all_queues_empty(self, mock_setup):
        """Test all queues are empty."""
        scheduler = MockScheduler(waiting_queue_empty=True, running_batch_empty=True)
        controller = ModeController(scheduler)

        self.assertTrue(controller._are_queues_empty())

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_prefill_queues_not_empty(self, mock_setup):
        """Test detection of non-empty prefill queues."""
        scheduler = MockScheduler()
        scheduler.disagg_prefill_bootstrap_queue = MockQueue([MagicMock()])
        scheduler.disagg_prefill_inflight_queue = []

        controller = ModeController(scheduler)
        self.assertFalse(controller._are_queues_empty())

    @patch(
        "sglang.srt.disaggregation.mode_controller.ModeController._setup_bootstrap_callbacks"
    )
    def test_decode_queues_not_empty(self, mock_setup):
        """Test detection of non-empty decode queues."""
        scheduler = MockScheduler()
        scheduler.disagg_decode_prealloc_queue = MockQueue([MagicMock()])

        controller = ModeController(scheduler)
        self.assertFalse(controller._are_queues_empty())


class TestDisaggregationMode(unittest.TestCase):
    """Test DisaggregationMode enum."""

    def test_mode_values(self):
        """Test mode string values."""
        self.assertEqual(DisaggregationMode.NULL.value, "null")
        self.assertEqual(DisaggregationMode.PREFILL.value, "prefill")
        self.assertEqual(DisaggregationMode.DECODE.value, "decode")

    def test_mode_from_string(self):
        """Test creating mode from string."""
        self.assertEqual(DisaggregationMode("null"), DisaggregationMode.NULL)
        self.assertEqual(DisaggregationMode("prefill"), DisaggregationMode.PREFILL)
        self.assertEqual(DisaggregationMode("decode"), DisaggregationMode.DECODE)

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with self.assertRaises(ValueError):
            DisaggregationMode("invalid")


if __name__ == "__main__":
    unittest.main()
