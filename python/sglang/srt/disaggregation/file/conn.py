# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


FILE_GUARD = b"SGLangDisaggFile"


def _storage_dir() -> str:
    return os.getenv(
        "SGLANG_DISAGGREGATION_FILE_BACKEND_STORAGE_DIR",
        "/tmp/sglang_disaggregation_file_backend",
    )


def _atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _try_unlink(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        logger.debug("Failed to remove file %s", path, exc_info=True)


@dataclass
class TransferInfo:
    """Sent by decode receiver to prefill manager."""

    room: int
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool = False
    dst_state_indices: Optional[npt.NDArray[np.int32]] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> "TransferInfo":
        # msg layout (after guard):
        # [room, sender_ip, sender_port, dst_kv_indices_bytes, dst_aux_index, required_dst_info_num, is_dummy, dst_state_indices_bytes]
        room = int(msg[0].decode("ascii"))
        # sender_ip/sender_port are used as the key to support required_dst_info_num > 1
        sender_ip = msg[1].decode("ascii")
        sender_port = msg[2].decode("ascii")
        dst_kv_indices = np.frombuffer(msg[3], dtype=np.int32)
        dst_aux_index = int(msg[4].decode("ascii"))
        required_dst_info_num = int(msg[5].decode("ascii"))
        is_dummy = bool(int(msg[6].decode("ascii")))
        dst_state_bytes = msg[7] if len(msg) > 7 else b""
        dst_state_indices = (
            np.frombuffer(dst_state_bytes, dtype=np.int32) if dst_state_bytes else None
        )
        return cls(
            room=room,
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=required_dst_info_num,
            is_dummy=is_dummy,
            dst_state_indices=dst_state_indices,
        )


class FileKVManager(CommonKVManager):
    """
    File-based disaggregation backend.

    Prefill side writes KV/aux/state to a shared directory, decode side reads and loads it.
    This backend is intended for correctness/debugging and requires a shared filesystem.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.base_dir = _storage_dir()
        # Keep parity with NIXL timeout semantics
        self.waiting_timeout = float(
            os.getenv("SGLANG_DISAGGREGATION_FILE_WAITING_TIMEOUT", "60")
        )
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()

    # Status helpers (required by CommonKVSender/CommonKVReceiver)
    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        logger.error(
            "[file-backend] room=%s failed: %s", bootstrap_room, failure_reason
        )

    def _room_dir(self, room: int) -> str:
        # Partition by room to reduce glob cost
        return os.path.join(self.base_dir, f"room_{room}")

    def _chunk_path(self, room: int, pp_rank: int, chunk_id: int) -> str:
        return os.path.join(
            self._room_dir(room),
            f"pp_{pp_rank}_chunk_{chunk_id}.pt",
        )

    def _done_path(self, room: int, pp_rank: int) -> str:
        return os.path.join(self._room_dir(room), f"pp_{pp_rank}.done")

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        # NOTE: We currently only support equal TP sizes for non-MLA models.
        # For different TP sizes, RDMA backends implement head slicing; file backend does not yet.
        # The needed info is available (required_dst_info_num), but assembling head slices is non-trivial.

        if bootstrap_room not in self.transfer_infos:
            self.record_failure(
                bootstrap_room, "TransferInfo not found. Receiver not bootstrapped?"
            )
            self.update_status(bootstrap_room, KVPoll.Failed)
            return

        # Source-side pools/buffers are attached by prefill/decode init code for file backend.
        kv_pool = getattr(self.kv_args, "_file_kv_pool", None)
        draft_kv_pool = getattr(self.kv_args, "_file_draft_kv_pool", None)
        metadata_buffers = getattr(self.kv_args, "_file_metadata_buffers", None)

        if kv_pool is None:
            raise RuntimeError(
                "File backend requires kv_args._file_kv_pool to be set (KVCache object)."
            )
        if metadata_buffers is None:
            raise RuntimeError(
                "File backend requires kv_args._file_metadata_buffers to be set (MetadataBuffers object)."
            )

        state_indices_np = (
            np.asarray(state_indices, dtype=np.int32)
            if (is_last and state_indices is not None)
            else None
        )

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy:
                continue

            chunked_dst_kv_indices = req.dst_kv_indices[index_slice]
            if len(chunked_dst_kv_indices) != len(kv_indices):
                raise ValueError(
                    f"KV indices length mismatch for room={bootstrap_room}: "
                    f"{len(chunked_dst_kv_indices)} != {len(kv_indices)}"
                )

            # Copy KV pages to CPU (per-pool) and serialize.
            kv_cpu = kv_pool.get_cpu_copy(kv_indices.tolist())
            draft_kv_cpu = (
                draft_kv_pool.get_cpu_copy(kv_indices.tolist())
                if draft_kv_pool is not None
                else None
            )

            # Optional state transfer (e.g., SWA window cache).
            dst_state_indices = None
            state_cpu = None
            if state_indices_np is not None:
                if req.dst_state_indices is None:
                    raise ValueError(
                        f"Missing dst_state_indices from receiver for room={bootstrap_room}"
                    )
                dst_state_indices = req.dst_state_indices
                # State pool is backend-specific; for SWA it is kv_pool.swa_kv_pool
                state_pool = getattr(kv_pool, "swa_kv_pool", None)
                if state_pool is None or not hasattr(state_pool, "get_cpu_copy"):
                    raise NotImplementedError(
                        "File backend currently supports state transfer only for SWA pools."
                    )
                if len(dst_state_indices) != len(state_indices_np):
                    raise ValueError(
                        f"State indices length mismatch for room={bootstrap_room}"
                    )
                state_cpu = state_pool.get_cpu_copy(state_indices_np.tolist())

            payload = {
                "room": bootstrap_room,
                "pp_rank": int(self.kv_args.pp_rank),
                "chunk_id": int(chunk_id),
                "is_last": bool(is_last),
                "dst_kv_indices": np.asarray(chunked_dst_kv_indices, dtype=np.int32),
                "kv_cpu": kv_cpu,
                "draft_kv_cpu": draft_kv_cpu,
                "dst_state_indices": (
                    np.asarray(dst_state_indices, dtype=np.int32)
                    if dst_state_indices is not None
                    else None
                ),
                "state_cpu": state_cpu,
                "dst_aux_index": int(req.dst_aux_index),
                "aux_tensors": (
                    metadata_buffers.get_buf(int(aux_index)) if is_last else None
                ),
            }

            chunk_path = self._chunk_path(
                bootstrap_room, int(self.kv_args.pp_rank), int(chunk_id)
            )
            _atomic_torch_save(payload, chunk_path)

            if is_last:
                done_path = self._done_path(bootstrap_room, int(self.kv_args.pp_rank))
                _atomic_torch_save(
                    {
                        "room": bootstrap_room,
                        "pp_rank": int(self.kv_args.pp_rank),
                        "last_chunk_id": int(chunk_id),
                        "timestamp": time.time(),
                    },
                    done_path,
                )

        if is_last:
            # Room can be released after the last chunk is persisted.
            del self.transfer_infos[bootstrap_room]

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if not msg:
                    continue
                assert (
                    msg[0] == FILE_GUARD
                ), f"First message should be {FILE_GUARD}. Foreign traffic?"
                msg = msg[1:]
                info = TransferInfo.from_zmq(msg)
                room = info.room
                # Use sender endpoint as key to support required_dst_info_num > 1
                sender_ip = msg[1].decode("ascii")
                sender_port = msg[2].decode("ascii")
                sender_key = f"{sender_ip}:{sender_port}"
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][sender_key] = info
                if len(self.transfer_infos[room]) == info.required_dst_info_num:
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread, daemon=True).start()


class FileKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: FileKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.chunk_id = 0
        self.has_sent = False

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            aux_index=self.aux_index,
            state_indices=state_indices if is_last else None,
        )
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            # no longer need local status tracking
            if self.bootstrap_room in self.kv_mgr.request_status:
                del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        return KVPoll.Success  # type: ignore

    def failure_exception(self):
        raise RuntimeError("File KVSender Exception")


class FileKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: FileKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state: Optional[KVPoll] = None
        self.init_time: Optional[float] = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Limitations: for non-MLA models, require same TP size (no head slicing).
        if (not self.kv_mgr.is_mla_backend) and (
            self.kv_mgr.attn_tp_size != self.prefill_attn_tp_size
        ):
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"File backend requires equal TP sizes for non-MLA models: "
                f"decode_tp={self.kv_mgr.attn_tp_size}, prefill_tp={self.prefill_attn_tp_size}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        dst_state_indices = (
            np.asarray(state_indices, dtype=np.int32).tobytes()
            if state_indices is not None
            else b""
        )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bool(bootstrap_info.get("is_dummy", False))
            with lock:
                sock.send_multipart(
                    [
                        FILE_GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                        b"1" if is_dummy else b"0",
                        dst_state_indices if not is_dummy else b"",
                    ]
                )

        self.started_transfer = True
        self.init_time = time.time()

    def _load_and_apply(self) -> bool:
        # Destination-side buffers/pools are attached by decode init code for file backend.
        kv_pool = getattr(self.kv_mgr.kv_args, "_file_kv_pool", None)
        draft_kv_pool = getattr(self.kv_mgr.kv_args, "_file_draft_kv_pool", None)
        metadata_buffers = getattr(self.kv_mgr.kv_args, "_file_metadata_buffers", None)

        if kv_pool is None or metadata_buffers is None:
            raise RuntimeError(
                "File backend requires kv_args._file_kv_pool/_file_metadata_buffers to be set."
            )

        # Wait for done markers from expected prefill PP ranks.
        # CommonKVReceiver computes self.target_pp_ranks based on prefill/decode PP sizes.
        needed_pp_ranks = set(
            getattr(self, "target_pp_ranks", [int(self.kv_mgr.pp_rank)])
        )

        room_dir = self.kv_mgr._room_dir(self.bootstrap_room)
        for pp_rank in needed_pp_ranks:
            done_path = self.kv_mgr._done_path(self.bootstrap_room, pp_rank)
            if not os.path.exists(done_path):
                return False

        # Load all chunks and apply in chunk_id order.
        for pp_rank in sorted(needed_pp_ranks):
            chunk_glob = os.path.join(room_dir, f"pp_{pp_rank}_chunk_*.pt")
            chunk_paths = sorted(
                glob.glob(chunk_glob),
                key=lambda p: int(
                    os.path.basename(p).split("_chunk_")[1].split(".")[0]
                ),
            )
            for path in chunk_paths:
                # NOTE: PyTorch 2.6 changed torch.load default `weights_only` from False to True.
                # The file backend payload contains numpy arrays and other non-state_dict objects,
                # which are blocked by weights-only safe unpickling (e.g., numpy reconstruct).
                # These .pt files are produced by the paired prefill process for the same request,
                # so loading with weights_only=False is acceptable in typical deployments.
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    # Older PyTorch versions don't have `weights_only`.
                    payload = torch.load(path, map_location="cpu")
                dst_kv_indices = payload["dst_kv_indices"].astype(np.int32)
                kv_cpu = payload["kv_cpu"]
                kv_pool.load_cpu_copy(
                    kv_cpu, torch.from_numpy(dst_kv_indices).to(torch.int64)
                )
                if (
                    draft_kv_pool is not None
                    and payload.get("draft_kv_cpu") is not None
                ):
                    draft_kv_pool.load_cpu_copy(
                        payload["draft_kv_cpu"],
                        torch.from_numpy(dst_kv_indices).to(torch.int64),
                    )

                if (
                    payload.get("state_cpu") is not None
                    and payload.get("dst_state_indices") is not None
                ):
                    dst_state_indices = payload["dst_state_indices"].astype(np.int32)
                    state_pool = getattr(kv_pool, "swa_kv_pool", None)
                    if state_pool is None:
                        raise NotImplementedError(
                            "File backend state load currently supports SWA pools only."
                        )
                    state_pool.load_cpu_copy(
                        payload["state_cpu"],
                        torch.from_numpy(dst_state_indices).to(torch.int64),
                    )

                if payload.get("aux_tensors") is not None:
                    dst_aux_index = int(payload["dst_aux_index"])
                    (
                        out_ids,
                        cached_tokens,
                        logp_val,
                        logp_idx,
                        top_logp_val,
                        top_logp_idx,
                        topk_p,
                        topk_index,
                        hidden_states,
                    ) = payload["aux_tensors"]
                    metadata_buffers.output_ids[dst_aux_index].copy_(out_ids)
                    metadata_buffers.cached_tokens[dst_aux_index].copy_(cached_tokens)
                    metadata_buffers.output_token_logprobs_val[dst_aux_index].copy_(
                        logp_val
                    )
                    metadata_buffers.output_token_logprobs_idx[dst_aux_index].copy_(
                        logp_idx
                    )
                    metadata_buffers.output_top_logprobs_val[dst_aux_index].copy_(
                        top_logp_val
                    )
                    metadata_buffers.output_top_logprobs_idx[dst_aux_index].copy_(
                        top_logp_idx
                    )
                    metadata_buffers.output_topk_p[dst_aux_index].copy_(topk_p)
                    metadata_buffers.output_topk_index[dst_aux_index].copy_(topk_index)
                    metadata_buffers.output_hidden_states[dst_aux_index].copy_(
                        hidden_states
                    )

            # Cleanup after successful load for this pp_rank
            for path in chunk_paths:
                _try_unlink(path)
            _try_unlink(self.kv_mgr._done_path(self.bootstrap_room, pp_rank))

        return True

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status

        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore

        assert self.init_time is not None
        elapsed = time.time() - self.init_time
        if elapsed >= self.kv_mgr.waiting_timeout:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        try:
            if self._load_and_apply():
                self.conclude_state = KVPoll.Success
                return KVPoll.Success  # type: ignore
        except Exception as e:
            self.kv_mgr.record_failure(self.bootstrap_room, str(e))
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise RuntimeError("File KVReceiver Exception")


class FileKVBootstrapServer(CommonKVBootstrapServer):
    pass
