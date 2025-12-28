from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.file.conn import FileKVReceiver, FileKVSender

# Use the common implementations for the file backend
FileKVBootstrapServer = CommonKVBootstrapServer
FileKVManager = CommonKVManager

__all__ = ["FileKVSender", "FileKVReceiver", "FileKVBootstrapServer", "FileKVManager"]
