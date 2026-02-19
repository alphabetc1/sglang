import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=2, suite="stage-b-test-small-1-gpu-amd")


class TestVerifyDoneOptionalWait(unittest.TestCase):
    @staticmethod
    def _make_batch(*, is_spec_v2=True, verify_done=None):
        return SimpleNamespace(
            is_spec_v2=is_spec_v2,
            spec_info=SimpleNamespace(verify_done=verify_done),
            device="cuda:0",
        )

    def test_synchronize_by_default(self):
        verify_done = MagicMock()
        batch = self._make_batch(verify_done=verify_done)

        with envs.SGLANG_SPEC_V2_VERIFY_DONE_STREAM_WAIT.override(False):
            ScheduleBatch.maybe_wait_verify_done(batch)

        verify_done.synchronize.assert_called_once_with()
        self.assertIsNone(batch.spec_info.verify_done)

    def test_wait_event_when_enabled(self):
        verify_done = MagicMock()
        stream = MagicMock()
        device_module = MagicMock()
        device_module.current_stream.return_value = stream
        batch = self._make_batch(verify_done=verify_done)

        with envs.SGLANG_SPEC_V2_VERIFY_DONE_STREAM_WAIT.override(True), patch(
            "sglang.srt.managers.schedule_batch.torch.get_device_module",
            return_value=device_module,
        ):
            ScheduleBatch.maybe_wait_verify_done(batch)

        stream.wait_event.assert_called_once_with(verify_done)
        verify_done.synchronize.assert_not_called()
        self.assertIsNone(batch.spec_info.verify_done)

    def test_noop_when_event_is_none(self):
        batch = self._make_batch(verify_done=None)

        with envs.SGLANG_SPEC_V2_VERIFY_DONE_STREAM_WAIT.override(False):
            ScheduleBatch.maybe_wait_verify_done(batch)

        self.assertIsNone(batch.spec_info.verify_done)

    def test_wait_event_falls_back_to_synchronize(self):
        verify_done = MagicMock()
        stream = MagicMock()
        stream.wait_event.side_effect = RuntimeError("wait_event failed")
        device_module = MagicMock()
        device_module.current_stream.return_value = stream
        batch = self._make_batch(verify_done=verify_done)

        with envs.SGLANG_SPEC_V2_VERIFY_DONE_STREAM_WAIT.override(True), patch(
            "sglang.srt.managers.schedule_batch.torch.get_device_module",
            return_value=device_module,
        ), self.assertLogs("sglang.srt.managers.schedule_batch", level="ERROR"):
            ScheduleBatch.maybe_wait_verify_done(batch)

        stream.wait_event.assert_called_once_with(verify_done)
        verify_done.synchronize.assert_called_once_with()
        self.assertIsNone(batch.spec_info.verify_done)

    def test_event_is_consumed_once(self):
        verify_done = MagicMock()
        batch = self._make_batch(verify_done=verify_done)

        with envs.SGLANG_SPEC_V2_VERIFY_DONE_STREAM_WAIT.override(False):
            ScheduleBatch.maybe_wait_verify_done(batch)
            ScheduleBatch.maybe_wait_verify_done(batch)

        verify_done.synchronize.assert_called_once_with()
        self.assertIsNone(batch.spec_info.verify_done)


if __name__ == "__main__":
    unittest.main()
