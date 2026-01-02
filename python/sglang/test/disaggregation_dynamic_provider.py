# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Type

from sglang.srt.disaggregation.base.conn import KVArgs
from sglang.srt.disaggregation.file import (
    FileKVBootstrapServer,
    FileKVManager,
    FileKVReceiver,
    FileKVSender,
)
from sglang.srt.disaggregation.utils import KVClassType


class TestDisaggDynamicProvider:
    """A minimal dynamic backend provider used by unit tests."""

    def get_kv_class(self, class_type: KVClassType) -> Optional[Type]:
        mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FileKVManager,
            KVClassType.SENDER: FileKVSender,
            KVClassType.RECEIVER: FileKVReceiver,
            KVClassType.BOOTSTRAP_SERVER: FileKVBootstrapServer,
        }
        return mapping.get(class_type)
