# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

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
from sglang.srt.disaggregation.utils import DisaggregationMode, KVClassType, kv_to_page_indices
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

GUARD = b"SGLangDisaggPDKP"


def _require_cpu_contiguous(t: torch.Tensor, *, name: str) -> None:
    if t.is_cuda:
        raise ValueError(f"{name} must be a CPU tensor (PDKP supports CPU pinned only).")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


class _PDKPClientWrapper:
    """Same wrapper logic as HiCache: import + init + register buffers."""

    def __init__(self, config: Dict[str, Any]):
        try:
            from . import pdkp_sglang_sdk  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PDKP PD-disaggregation backend requires `pdkp_sglang_sdk` to be importable."
            ) from e
        self._sdk = pdkp_sglang_sdk
        world_size = int(config.get("world_size", 1))
        rank_id = int(config.get("rank_id", 0))
        self._client = pdkp_sglang_sdk.SGLangClient(world_size=world_size, rank_id=rank_id)

        cfg_path = config.get("client_config_path") or os.getenv("PDKP_SGLANG_CLIENT_CONFIG")
        if not cfg_path:
            raise ValueError(
                "Missing PDKP client config. Provide dynamic extra_config['client_config_path'] "
                "or set env PDKP_SGLANG_CLIENT_CONFIG."
            )
        ok = self._client.init(str(cfg_path))
        if not ok:
            raise RuntimeError(f"PDKP client init failed for config_path={cfg_path!r}")

        self._registered: Dict[tuple[int, int], int] = {}

    def _ensure_registered(self, t: torch.Tensor) -> None:
        _require_cpu_contiguous(t, name="tensor")
        addr = int(t.data_ptr())
        size = int(t.numel() * t.element_size())
        key = (addr, size)
        if key in self._registered:
            return
        ok, lkey = self._client.register_memory(addr, size)
        if not ok:
            raise RuntimeError(
                "PDKP register_memory failed. Ensure buffers are CPU pinned / supported by PDKP."
            )
        self._registered[key] = int(lkey)

    def put_tensor(self, key: str, value: torch.Tensor) -> bool:
        self._ensure_registered(value)
        st = self._client.put(key, value)
        return st == self._sdk.PDKPStatus.OK

    def get_tensor(self, key: str, out: torch.Tensor) -> bool:
        self._ensure_registered(out)
        st = self._client.get(key, out)
        return st == self._sdk.PDKPStatus.OK

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))


def _key_prefix(server_args: ServerArgs) -> str:
    # Allow namespacing across multiple deployments.
    return str(
        os.getenv("SGLANG_PDKP_DISAGG_KEY_PREFIX", "pd")
    )


def _chunk_key(prefix: str, room: int, pp_rank: int, chunk_id: int) -> str:
    return f"{prefix}:room:{room}:pp:{pp_rank}:chunk:{chunk_id}"

def _chunk_meta_key(prefix: str, room: int, pp_rank: int, chunk_id: int) -> str:
    return f"{prefix}:room:{room}:pp:{pp_rank}:chunk:{chunk_id}:meta"


def _done_key(prefix: str, room: int, pp_rank: int) -> str:
    return f"{prefix}:room:{room}:pp:{pp_rank}:done"


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


class PDKPKVManager(CommonKVManager):
    """
    PDKP-based PD-disaggregation backend (方案A：中转 store).

    Prefill writes KV/aux/state payloads into PDKP by key, decode reads them back and loads into local pools.

    NOTE: This backend currently relies on:
    - CPU tensors (ideally pinned) for put/get
    - existence polling for done markers
    - server-side TTL/eviction, because the SDK does not provide delete/evict APIs.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.waiting_timeout = float(
            os.getenv("SGLANG_PDKP_DISAGG_WAITING_TIMEOUT", "300")
        )
        self._prefix = _key_prefix(server_args)

        # Load config from server args JSON string if provided.
        cfg = {}
        extra = getattr(server_args, "disaggregation_transfer_backend_extra_config", None)
        if extra:
            try:
                cfg = json.loads(extra)
            except Exception:
                cfg = {}
        self._client = _PDKPClientWrapper(cfg)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()

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
        logger.error("[pdkp-backend] room=%s failed: %s", bootstrap_room, failure_reason)

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if not msg:
                    continue
                assert msg[0] == GUARD, f"First message should be {GUARD}. Foreign traffic?"
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

        import threading

        threading.Thread(target=bootstrap_thread, daemon=True).start()

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

        if bootstrap_room not in self.transfer_infos:
            self.record_failure(
                bootstrap_room, "TransferInfo not found. Receiver not bootstrapped?"
            )
            self.update_status(bootstrap_room, KVPoll.Failed)
            return

        kv_pool = getattr(self.kv_args, "_file_kv_pool", None)
        draft_kv_pool = getattr(self.kv_args, "_file_draft_kv_pool", None)
        metadata_buffers = getattr(self.kv_args, "_file_metadata_buffers", None)
        if kv_pool is None or metadata_buffers is None:
            raise RuntimeError(
                "PDKP backend requires kv_args._file_kv_pool/_file_metadata_buffers to be set "
                "(reuse file-backend-style direct object access for CPU staging)."
            )

        state_indices_np = (
            np.asarray(state_indices, dtype=np.int32)
            if (is_last and state_indices is not None)
            else None
        )

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        for req in reqs_to_be_processed:
            if req.is_dummy:
                continue
            # NOTE: KV transfer operates at **page** granularity. Both sides should
            # exchange page indices (not per-token cache indices). If upstream code
            # accidentally passes per-token indices, attempt to convert to pages using
            # the actual KV pool page size.
            chunked_dst_kv_indices = req.dst_kv_indices[index_slice]
            if len(chunked_dst_kv_indices) != len(kv_indices):
                # Try to interpret `kv_indices` as per-token indices and convert to page indices.
                try:
                    maybe_pages = kv_to_page_indices(
                        np.asarray(kv_indices, dtype=np.int32),
                        int(self.kv_args.page_size),
                    )
                except Exception:
                    maybe_pages = None

                if maybe_pages is not None and len(maybe_pages) == len(chunked_dst_kv_indices):
                    kv_indices = maybe_pages
                else:
                    raise ValueError(
                        f"KV indices length mismatch for room={bootstrap_room}: "
                        f"{len(chunked_dst_kv_indices)} != {len(kv_indices)} "
                        f"(kv_page_size={getattr(self.kv_args, 'page_size', None)})"
                    )

            kv_cpu = kv_pool.get_cpu_copy(kv_indices.tolist())
            draft_kv_cpu = (
                draft_kv_pool.get_cpu_copy(kv_indices.tolist())
                if draft_kv_pool is not None
                else None
            )

            dst_state_indices = None
            state_cpu = None
            if state_indices_np is not None:
                if req.dst_state_indices is None:
                    raise ValueError(
                        f"Missing dst_state_indices from receiver for room={bootstrap_room}"
                    )
                dst_state_indices = req.dst_state_indices
                state_pool = getattr(kv_pool, "swa_kv_pool", None)
                if state_pool is None or not hasattr(state_pool, "get_cpu_copy"):
                    raise NotImplementedError(
                        "PDKP backend currently supports state transfer only for SWA pools."
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

            # Serialize via torch.save into a CPU tensor buffer.
            import io

            buf = io.BytesIO()
            torch.save(payload, buf)
            data_bytes = buf.getbuffer()
            data = torch.frombuffer(data_bytes, dtype=torch.uint8).clone()
            _require_cpu_contiguous(data, name="serialized_payload")

            key = _chunk_key(self._prefix, bootstrap_room, int(self.kv_args.pp_rank), int(chunk_id))
            meta_key = _chunk_meta_key(self._prefix, bootstrap_room, int(self.kv_args.pp_rank), int(chunk_id))

            # Store payload size first so the receiver can allocate an exact buffer (PDKP get requires size).
            payload_size = int(data.numel())
            meta = payload_size.to_bytes(8, byteorder="little", signed=False)
            meta_tensor = torch.from_numpy(np.frombuffer(meta, dtype=np.uint8).copy())
            _require_cpu_contiguous(meta_tensor, name="payload_size_meta")
            if not self._client.put_tensor(meta_key, meta_tensor):
                self.record_failure(bootstrap_room, f"PDKP put failed for meta_key={meta_key}")
                self.update_status(bootstrap_room, KVPoll.Failed)
                return

            if not self._client.put_tensor(key, data):
                self.record_failure(bootstrap_room, f"PDKP put failed for key={key}")
                self.update_status(bootstrap_room, KVPoll.Failed)
                return

            if is_last:
                done = {
                    "room": bootstrap_room,
                    "pp_rank": int(self.kv_args.pp_rank),
                    "last_chunk_id": int(chunk_id),
                    "timestamp": time.time(),
                }
                done_bytes = json.dumps(done).encode("utf-8")
                done_tensor = torch.from_numpy(
                    np.frombuffer(done_bytes, dtype=np.uint8).copy()
                )
                done_key = _done_key(self._prefix, bootstrap_room, int(self.kv_args.pp_rank))
                if not self._client.put_tensor(done_key, done_tensor):
                    self.record_failure(bootstrap_room, f"PDKP put failed for done_key={done_key}")
                    self.update_status(bootstrap_room, KVPoll.Failed)
                    return

        if is_last:
            del self.transfer_infos[bootstrap_room]


class PDKPKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: PDKPKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.chunk_id = 0
        self.has_sent = False

    def send(self, kv_indices: npt.NDArray[np.int32], state_indices: Optional[List[int]] = None):
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
            if self.bootstrap_room in self.kv_mgr.request_status:
                del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        return KVPoll.Success  # type: ignore

    def failure_exception(self):
        raise RuntimeError("PDKP KVSender Exception")


class PDKPKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: PDKPKVManager,
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

        dst_state_indices = (
            np.asarray(state_indices, dtype=np.int32).tobytes() if state_indices is not None else b""
        )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bool(bootstrap_info.get("is_dummy", False))
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
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
        kv_pool = getattr(self.kv_mgr.kv_args, "_file_kv_pool", None)
        draft_kv_pool = getattr(self.kv_mgr.kv_args, "_file_draft_kv_pool", None)
        metadata_buffers = getattr(self.kv_mgr.kv_args, "_file_metadata_buffers", None)
        if kv_pool is None or metadata_buffers is None:
            raise RuntimeError(
                "PDKP backend requires kv_args._file_kv_pool/_file_metadata_buffers to be set."
            )

        needed_pp_ranks = set(getattr(self, "target_pp_ranks", [int(self.kv_mgr.pp_rank)]))

        # Wait for done markers
        for pp_rank in needed_pp_ranks:
            if not self.kv_mgr._client.exists(_done_key(self.kv_mgr._prefix, self.bootstrap_room, pp_rank)):
                return False

        # Load chunks by scanning sequential chunk meta keys.
        # This requires the producer to write contiguous chunk ids starting at 0.
        for pp_rank in sorted(needed_pp_ranks):
            chunk_id = 0
            while True:
                meta_key = _chunk_meta_key(self.kv_mgr._prefix, self.bootstrap_room, pp_rank, chunk_id)
                if not self.kv_mgr._client.exists(meta_key):
                    break
                meta_buf = torch.empty((8,), dtype=torch.uint8, pin_memory=True)
                if not self.kv_mgr._client.get_tensor(meta_key, meta_buf):
                    raise RuntimeError(f"PDKP get failed for meta_key={meta_key}")
                payload_size = int.from_bytes(bytes(meta_buf.cpu().tolist()), byteorder="little", signed=False)

                key = _chunk_key(self.kv_mgr._prefix, self.bootstrap_room, pp_rank, chunk_id)
                buf = torch.empty((payload_size,), dtype=torch.uint8, pin_memory=True)
                if not self.kv_mgr._client.get_tensor(key, buf):
                    raise RuntimeError(f"PDKP get failed for key={key}")

                import io

                raw = buf.cpu().numpy().tobytes()
                bio = io.BytesIO(raw)
                try:
                    payload = torch.load(bio, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(bio, map_location="cpu")

                dst_kv_indices = payload["dst_kv_indices"].astype(np.int32)
                kv_cpu = payload["kv_cpu"]
                kv_pool.load_cpu_copy(kv_cpu, torch.from_numpy(dst_kv_indices).to(torch.int64))

                if draft_kv_pool is not None and payload.get("draft_kv_cpu") is not None:
                    draft_kv_pool.load_cpu_copy(
                        payload["draft_kv_cpu"],
                        torch.from_numpy(dst_kv_indices).to(torch.int64),
                    )

                if payload.get("state_cpu") is not None and payload.get("dst_state_indices") is not None:
                    dst_state_indices = payload["dst_state_indices"].astype(np.int32)
                    state_pool = getattr(kv_pool, "swa_kv_pool", None)
                    if state_pool is None:
                        raise NotImplementedError(
                            "PDKP backend state load currently supports SWA pools only."
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
                    metadata_buffers.output_token_logprobs_val[dst_aux_index].copy_(logp_val)
                    metadata_buffers.output_token_logprobs_idx[dst_aux_index].copy_(logp_idx)
                    metadata_buffers.output_top_logprobs_val[dst_aux_index].copy_(top_logp_val)
                    metadata_buffers.output_top_logprobs_idx[dst_aux_index].copy_(top_logp_idx)
                    metadata_buffers.output_topk_p[dst_aux_index].copy_(topk_p)
                    metadata_buffers.output_topk_index[dst_aux_index].copy_(topk_index)
                    metadata_buffers.output_hidden_states[dst_aux_index].copy_(hidden_states)

                chunk_id += 1

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

        now = time.time()
        elapsed = now - self.init_time
        if elapsed >= self.kv_mgr.waiting_timeout:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        try:
            if self._load_and_apply():
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                self.conclude_state = KVPoll.Success
                return KVPoll.Success
        except Exception as e:
            self.kv_mgr.record_failure(self.bootstrap_room, str(e))
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise RuntimeError("PDKP KVReceiver Exception")


class PDKPKVBootstrapServer(CommonKVBootstrapServer):
    pass


class PDKPDisaggBackendProvider:
    """
    Dynamic backend provider for PD disaggregation transfer: PDKP (方案A：中转 store).

    Use with:
      --disaggregation-transfer-backend dynamic
      --disaggregation-transfer-backend-extra-config '{"backend_name": "...", "module_path": "sglang.srt.integrations.pdkp.disagg_pdkp", "class_name": "PDKPDisaggBackendProvider", ...}'

    NOTE: extra_config is parsed by the factory only for locating this provider.
    Any PDKP client settings should be passed via environment variables (e.g., PDKP_SGLANG_CLIENT_CONFIG),
    or be embedded into the same JSON and parsed by PDKPKVManager.
    """

    def get_kv_class(self, class_type: KVClassType) -> Optional[Type]:
        mapping: Dict[KVClassType, Type] = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: PDKPKVManager,
            KVClassType.SENDER: PDKPKVSender,
            KVClassType.RECEIVER: PDKPKVReceiver,
            KVClassType.BOOTSTRAP_SERVER: PDKPKVBootstrapServer,
        }
        return mapping.get(class_type)


