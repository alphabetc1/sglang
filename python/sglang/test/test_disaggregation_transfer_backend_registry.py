# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from types import SimpleNamespace

from sglang.srt.server_args import DISAGG_TRANSFER_BACKEND_CHOICES
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.disaggregation.utils import (
        KVClassType,
        TransferBackend,
        get_kv_class,
    )
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "torch":
        raise unittest.SkipTest("torch is not available")
    raise


class TestDisaggregationTransferBackendRegistry(CustomTestCase):
    def test_choices_include_file_and_dynamic(self):
        self.assertIn("file", DISAGG_TRANSFER_BACKEND_CHOICES)
        self.assertIn("dynamic", DISAGG_TRANSFER_BACKEND_CHOICES)

    def test_file_backend_class_mapping(self):
        from sglang.srt.disaggregation.file import FileKVSender

        cls = get_kv_class(TransferBackend.FILE, KVClassType.SENDER)
        self.assertIs(cls, FileKVSender)

    def test_dynamic_backend_resolves_provider(self):
        from sglang.srt.disaggregation.file import FileKVSender

        extra = {
            "backend_name": "unit_test",
            "module_path": "sglang.test.disaggregation_dynamic_provider",
            "class_name": "TestDisaggDynamicProvider",
        }
        server_args = SimpleNamespace(
            disaggregation_transfer_backend_extra_config=json.dumps(extra)
        )
        cls = get_kv_class(
            TransferBackend.DYNAMIC,
            KVClassType.SENDER,
            server_args=server_args,
        )
        self.assertIs(cls, FileKVSender)

    def test_dynamic_backend_requires_server_args(self):
        with self.assertRaises(ValueError):
            get_kv_class(TransferBackend.DYNAMIC, KVClassType.SENDER)


if __name__ == "__main__":
    unittest.main()
