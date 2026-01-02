# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

try:
    import numpy as np
    import torch

    from sglang.srt.disaggregation.file.conn import (
        _atomic_torch_save,
        FileKVReceiver,
        TransferInfo,
    )
except ModuleNotFoundError as e:
    # Importing `sglang` transitively imports numpy/torch in some environments.
    # Keep unit tests robust when optional deps are not installed.
    if getattr(e, "name", None) in ("torch", "numpy", "zmq"):
        raise unittest.SkipTest(f"Optional dependency is not available: {e}")
    raise


class _DummyStatePool:
    def __init__(self):
        self.data = {}

    def load_cpu_copy(self, state_cpu: torch.Tensor, indices: torch.Tensor):
        indices_list = indices.to(torch.int64).cpu().tolist()
        flat = state_cpu.detach().cpu()
        if flat.ndim == 0:
            flat = flat.view(1)
        if flat.ndim == 1:
            values = flat.tolist()
        else:
            values = [x.clone() for x in flat]
        for idx, val in zip(indices_list, values):
            self.data[int(idx)] = val


class _DummyKVPool:
    def __init__(self):
        self.data = {}
        # File backend state load currently supports SWA pools only.
        self.swa_kv_pool = _DummyStatePool()

    def load_cpu_copy(self, kv_cpu: torch.Tensor, indices: torch.Tensor):
        indices_list = indices.to(torch.int64).cpu().tolist()
        flat = kv_cpu.detach().cpu()
        if flat.ndim == 0:
            flat = flat.view(1)
        if flat.ndim == 1:
            values = flat.tolist()
        else:
            values = [x.clone() for x in flat]
        for idx, val in zip(indices_list, values):
            self.data[int(idx)] = val


class _DummyMetadataBuffers:
    """A minimal in-memory stand-in for MetadataBuffers used by the file backend."""

    def __init__(self, buf_size: int, seq_len: int = 8, topk: int = 4, hidden: int = 6):
        self.output_ids = torch.zeros((buf_size, seq_len), dtype=torch.int64)
        self.cached_tokens = torch.zeros((buf_size, seq_len), dtype=torch.int64)
        self.output_token_logprobs_val = torch.zeros((buf_size, seq_len), dtype=torch.float32)
        self.output_token_logprobs_idx = torch.zeros((buf_size, seq_len), dtype=torch.int64)
        self.output_top_logprobs_val = torch.zeros((buf_size, seq_len, topk), dtype=torch.float32)
        self.output_top_logprobs_idx = torch.zeros((buf_size, seq_len, topk), dtype=torch.int64)
        self.output_topk_p = torch.zeros((buf_size, topk), dtype=torch.float32)
        self.output_topk_index = torch.zeros((buf_size, topk), dtype=torch.int64)
        self.output_hidden_states = torch.zeros((buf_size, hidden), dtype=torch.float32)


class _DummyFileKVManager:
    """A minimal subset of FileKVManager used by FileKVReceiver._load_and_apply()."""

    def __init__(self, base_dir: str, kv_pool: _DummyKVPool, metadata_buffers: _DummyMetadataBuffers):
        self.base_dir = base_dir
        self.pp_rank = 0
        self.kv_args = SimpleNamespace(
            _file_kv_pool=kv_pool,
            _file_draft_kv_pool=None,
            _file_metadata_buffers=metadata_buffers,
        )

    def _room_dir(self, room: int) -> str:
        return os.path.join(self.base_dir, f"room_{room}")

    def _done_path(self, room: int, pp_rank: int) -> str:
        return os.path.join(self._room_dir(room), f"pp_{pp_rank}.done")


class TestDisaggregationFileBackend(CustomTestCase):
    def test_transfer_info_from_zmq_roundtrip_fields(self):
        room = 7
        sender_ip = "127.0.0.1"
        sender_port = "12345"
        dst_kv_indices = np.asarray([1, 2, 3], dtype=np.int32)
        dst_aux_index = 5
        required_dst_info_num = 2
        is_dummy = True
        dst_state_indices = np.asarray([9, 10], dtype=np.int32)

        msg = [
            str(room).encode("ascii"),
            sender_ip.encode("ascii"),
            sender_port.encode("ascii"),
            dst_kv_indices.tobytes(),
            str(dst_aux_index).encode("ascii"),
            str(required_dst_info_num).encode("ascii"),
            b"1" if is_dummy else b"0",
            dst_state_indices.tobytes(),
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertEqual(info.room, room)
        self.assertEqual(info.dst_aux_index, dst_aux_index)
        self.assertEqual(info.required_dst_info_num, required_dst_info_num)
        self.assertEqual(info.is_dummy, is_dummy)
        self.assertTrue(np.array_equal(info.dst_kv_indices, dst_kv_indices))
        self.assertTrue(np.array_equal(info.dst_state_indices, dst_state_indices))

    def test_file_receiver_load_and_apply_kv_aux_state_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            room = 123
            kv_pool = _DummyKVPool()
            metadata_buffers = _DummyMetadataBuffers(buf_size=2)
            kv_mgr = _DummyFileKVManager(tmp, kv_pool, metadata_buffers)

            room_dir = kv_mgr._room_dir(room)
            os.makedirs(room_dir, exist_ok=True)

            # Prepare a single chunk payload.
            dst_kv_indices = np.asarray([10, 11], dtype=np.int32)
            kv_cpu = torch.tensor([111, 222], dtype=torch.int64)

            dst_state_indices = np.asarray([7], dtype=np.int32)
            state_cpu = torch.tensor([999], dtype=torch.int64)

            dst_aux_index = 1
            aux_tensors = (
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64),  # out_ids
                torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.int64),  # cached_tokens
                torch.zeros((8,), dtype=torch.float32),  # logp_val
                torch.zeros((8,), dtype=torch.int64),  # logp_idx
                torch.zeros((8, 4), dtype=torch.float32),  # top_logp_val
                torch.zeros((8, 4), dtype=torch.int64),  # top_logp_idx
                torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),  # topk_p
                torch.tensor([10, 20, 30, 40], dtype=torch.int64),  # topk_index
                torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),  # hidden_states
            )

            chunk_path = os.path.join(room_dir, "pp_0_chunk_0.pt")
            _atomic_torch_save(
                {
                    "room": room,
                    "pp_rank": 0,
                    "chunk_id": 0,
                    "is_last": True,
                    "dst_kv_indices": dst_kv_indices,
                    "kv_cpu": kv_cpu,
                    "draft_kv_cpu": None,
                    "dst_state_indices": dst_state_indices,
                    "state_cpu": state_cpu,
                    "dst_aux_index": dst_aux_index,
                    "aux_tensors": aux_tensors,
                },
                chunk_path,
            )
            done_path = kv_mgr._done_path(room, 0)
            _atomic_torch_save({"room": room, "pp_rank": 0, "last_chunk_id": 0}, done_path)

            # Construct a FileKVReceiver without running the full network bootstrap path.
            recv = FileKVReceiver.__new__(FileKVReceiver)
            recv.kv_mgr = kv_mgr
            recv.bootstrap_room = room
            recv.target_pp_ranks = [0]

            ok = recv._load_and_apply()
            self.assertTrue(ok)

            # KV applied to destination indices.
            self.assertEqual(kv_pool.data[10], 111)
            self.assertEqual(kv_pool.data[11], 222)

            # State applied to SWA pool indices.
            self.assertEqual(kv_pool.swa_kv_pool.data[7], 999)

            # Aux tensors copied into the dst aux slot.
            self.assertTrue(torch.equal(metadata_buffers.output_ids[dst_aux_index], aux_tensors[0]))
            self.assertTrue(torch.equal(metadata_buffers.cached_tokens[dst_aux_index], aux_tensors[1]))
            self.assertTrue(torch.equal(metadata_buffers.output_topk_p[dst_aux_index], aux_tensors[6]))
            self.assertTrue(torch.equal(metadata_buffers.output_topk_index[dst_aux_index], aux_tensors[7]))
            self.assertTrue(torch.equal(metadata_buffers.output_hidden_states[dst_aux_index], aux_tensors[8]))

            # Cleanup after successful load.
            self.assertFalse(os.path.exists(chunk_path))
            self.assertFalse(os.path.exists(done_path))

    def test_file_receiver_returns_false_if_done_marker_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            room = 1
            kv_pool = _DummyKVPool()
            metadata_buffers = _DummyMetadataBuffers(buf_size=1)
            kv_mgr = _DummyFileKVManager(tmp, kv_pool, metadata_buffers)
            os.makedirs(kv_mgr._room_dir(room), exist_ok=True)

            recv = FileKVReceiver.__new__(FileKVReceiver)
            recv.kv_mgr = kv_mgr
            recv.bootstrap_room = room
            recv.target_pp_ranks = [0]

            ok = recv._load_and_apply()
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()



