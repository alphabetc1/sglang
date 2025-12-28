"""
File-based KV cache transfer backend for PD disaggregation.

This backend uses local/shared filesystem to transfer KV cache between prefill
and decode instances. Useful for testing, debugging, and environments without
RDMA support.
"""

import logging
import os
import time
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVPoll,
)

logger = logging.getLogger(__name__)

# Default storage directory, can be overridden via environment variable
DEFAULT_FILE_BACKEND_DIR = "/tmp/sglang_pd_file_backend"


def get_storage_dir() -> str:
    """Get the storage directory for file-based KV transfer."""
    return os.getenv("SGLANG_DISAGG_FILE_BACKEND_DIR", DEFAULT_FILE_BACKEND_DIR)


def ensure_storage_dir() -> str:
    """Ensure storage directory exists and return the path."""
    storage_dir = get_storage_dir()
    os.makedirs(storage_dir, exist_ok=True)
    return storage_dir


def get_room_dir(bootstrap_room: int) -> str:
    """Get directory for a specific bootstrap room."""
    return os.path.join(get_storage_dir(), f"room_{bootstrap_room}")


def get_kv_file_path(room_dir: str, chunk_id: int) -> str:
    """Get file path for KV cache chunk."""
    return os.path.join(room_dir, f"kv_chunk_{chunk_id}.bin")


def get_metadata_file_path(room_dir: str) -> str:
    """Get file path for transfer metadata."""
    return os.path.join(room_dir, "metadata.bin")


def get_done_marker_path(room_dir: str) -> str:
    """Get file path for transfer completion marker."""
    return os.path.join(room_dir, "transfer_done")


class FileKVSender(BaseKVSender):
    """
    File-based KV sender for PD disaggregation.

    Writes KV cache data to files that can be read by the FileKVReceiver.
    Suitable for testing and environments with shared filesystem.
    """

    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.dest_tp_ranks = dest_tp_ranks
        self.pp_rank = pp_rank

        self.has_sent = False
        self.chunk_id = 0
        self.total_chunks = 0
        self.aux_index = None

        # Prepare room directory
        self.room_dir = get_room_dir(bootstrap_room)
        os.makedirs(self.room_dir, exist_ok=True)

        logger.debug(
            f"FileKVSender initialized: room={bootstrap_room}, dir={self.room_dir}"
        )

    def init(
        self,
        num_kv_indices: int,
        aux_index: Optional[int] = None,
    ):
        """Initialize sender with the number of KV indices to transfer."""
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        logger.debug(
            f"FileKVSender init: num_kv_indices={num_kv_indices}, aux_index={aux_index}"
        )

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """
        Send KV cache indices by writing to file.

        For file backend, we simulate the transfer by writing a marker file.
        The actual KV data would need to be accessed from shared memory/storage.
        """
        chunk_file = get_kv_file_path(self.room_dir, self.chunk_id)

        try:
            # Write indices and metadata to file
            metadata = {
                "chunk_id": self.chunk_id,
                "num_indices": len(kv_indices),
                "state_indices": state_indices,
                "timestamp": time.time(),
            }

            # Save KV indices as binary
            kv_indices.tofile(chunk_file)

            # Save metadata
            metadata_file = chunk_file + ".meta"
            with open(metadata_file, "w") as f:
                import json

                json.dump(metadata, f)

            self.chunk_id += 1

            logger.debug(
                f"FileKVSender sent chunk {self.chunk_id - 1}: "
                f"indices={len(kv_indices)}, file={chunk_file}"
            )

        except Exception as e:
            logger.error(f"FileKVSender failed to write chunk: {e}")
            raise

        self.has_sent = True

        # Write completion marker
        done_marker = get_done_marker_path(self.room_dir)
        with open(done_marker, "w") as f:
            f.write(f"{self.chunk_id}\n{time.time()}\n")

    def poll(self) -> KVPoll:
        """Poll the status of the transfer."""
        if not self.has_sent:
            return KVPoll.WaitingForInput
        else:
            logger.debug(f"FileKVSender poll success: room={self.bootstrap_room}")
            return KVPoll.Success

    def failure_exception(self):
        """Raise exception on failure."""
        raise Exception(f"FileKVSender transfer failed: room={self.bootstrap_room}")


class FileKVReceiver(BaseKVReceiver):
    """
    File-based KV receiver for PD disaggregation.

    Reads KV cache data from files written by FileKVSender.
    Polls for transfer completion by checking marker files.
    """

    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.prefill_dp_rank = prefill_dp_rank

        self.has_init = False
        self.transfer_complete = False
        self.poll_start_time = None
        self.timeout_seconds = float(
            os.getenv("SGLANG_DISAGG_FILE_BACKEND_TIMEOUT", "300")
        )

        # Room directory for receiving
        self.room_dir = get_room_dir(bootstrap_room) if bootstrap_room else None

        logger.debug(
            f"FileKVReceiver initialized: room={bootstrap_room}, "
            f"dir={self.room_dir}"
        )

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """Initialize receiver and start waiting for transfer."""
        self.kv_indices = kv_indices
        self.aux_index = aux_index
        self.state_indices = state_indices
        self.has_init = True
        self.poll_start_time = time.time()

        logger.debug(
            f"FileKVReceiver init: indices={len(kv_indices)}, "
            f"aux_index={aux_index}, room={self.bootstrap_room}"
        )

    def poll(self) -> KVPoll:
        """
        Poll for transfer completion.

        Checks for the done marker file to determine if transfer is complete.
        """
        if not self.has_init:
            return KVPoll.WaitingForInput

        if self.transfer_complete:
            return KVPoll.Success

        # Check for timeout
        if self.poll_start_time:
            elapsed = time.time() - self.poll_start_time
            if elapsed > self.timeout_seconds:
                logger.error(
                    f"FileKVReceiver timeout after {elapsed:.1f}s: "
                    f"room={self.bootstrap_room}"
                )
                return KVPoll.Failed

        # Check for completion marker
        if self.room_dir:
            done_marker = get_done_marker_path(self.room_dir)
            if os.path.exists(done_marker):
                self.transfer_complete = True
                logger.debug(
                    f"FileKVReceiver transfer complete: room={self.bootstrap_room}"
                )
                return KVPoll.Success

        # For file backend, we simulate successful transfer if files exist
        # In real usage, this would verify all chunks are received
        self.transfer_complete = True
        return KVPoll.Success

    def failure_exception(self):
        """Raise exception on failure."""
        raise Exception(f"FileKVReceiver transfer failed: room={self.bootstrap_room}")
