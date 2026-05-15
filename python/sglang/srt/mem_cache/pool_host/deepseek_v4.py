from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool.hisparse import HiSparseC4DevicePool

import psutil
import torch

from sglang.srt.utils import is_cuda, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if not (_is_npu or _is_xpu or _is_mps):
    from sgl_kernel.kvcacheio import (
        transfer_kv_direct,
    )
if _is_npu:
    pass

logger = logging.getLogger(__name__)

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)


from sglang.srt.mem_cache.pool_host.base import HostKVCache, synchronized
from sglang.srt.mem_cache.pool_host.tensor_allocator import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)


class LogicalHostPool:
    """Pure-logical anchor pool for V4 HiCache.

    The pool manages page-aligned token slots but holds no KV tensor. V4
    compressed side pools use these logical FULL indices as stable page anchors.
    """

    def __init__(self, size: int, page_size: int):
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = "layer_first"
        self.dtype = torch.uint8
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.lock = threading.RLock()
        self.clear()

    @synchronized
    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_ksize_per_token(self):
        return 0


class DeepSeekV4PagedHostPool(HostKVCache):
    """Host mirror for a DeepSeek V4 paged KV/indexer sub-pool."""

    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        self.pool_name = pool_name
        self.layer_num = len(device_buffers)
        self.item_bytes = item_bytes
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = "layer_first"
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        self.gpu_device = device_buffers[0].device if device_buffers else device

        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.kv_buffer = [
            alloc_func(
                (num_host_pages, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            for _ in range(self.layer_num)
        ]
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
        )
        self.clear()

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.slot_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be page-aligned, "
                f"got numel={indices.numel()}, slot_page_size={self.slot_page_size}"
            )
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _check_io_backend(self, io_backend: str) -> None:
        if io_backend != "direct":
            raise NotImplementedError(
                f"{self.pool_name} supports only direct io_backend, got {io_backend}"
            )

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=self.device_buffers,
            dst_layers=self.data_refs,
            src_indices=device_rows,
            dst_indices=host_rows,
            page_size=1,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=[self.kv_buffer[layer_id]],
            dst_layers=[self.device_buffers[layer_id]],
            src_indices=host_rows,
            dst_indices=device_rows,
            page_size=1,
        )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.slot_page_size
        data_page = torch.stack(
            [self.kv_buffer[i][index] for i in range(self.layer_num)]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.slot_page_size
        data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
        for i in range(self.layer_num):
            self.kv_buffer[i][index].copy_(data[i])

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        for row in rows:
            for layer_id in range(self.layer_num):
                ptr = (
                    self.kv_buffer[layer_id].data_ptr()
                    + int(row) * self.item_bytes * self.dtype.itemsize
                )
                ptr_list.append(ptr)
        element_size = self.item_bytes * self.dtype.itemsize
        return ptr_list, [element_size] * len(ptr_list)


class DeepSeekV4StateHostPool(HostKVCache):
    """Host pool for V4 CompressStatePool page rows."""

    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = "layer_first"
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.ring_size = 0
        self.state_page_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self.size_per_token = self.state_page_bytes

        requested_bytes = self.layer_num * num_host_pages * self.state_page_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 state pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.kv_buffer = [
            alloc_func(
                (num_host_pages, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            for _ in range(self.layer_num)
        ]
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
        )

    def _init_device_page_views(self) -> None:
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def _check_io_backend(self, io_backend: str) -> None:
        if io_backend != "direct":
            raise NotImplementedError(
                f"{self.pool_name} supports only direct io_backend, got {io_backend}"
            )

    def get_size_per_token(self):
        return self.state_page_bytes

    def get_ksize_per_token(self):
        return self.state_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer

    def clear(self):
        pass

    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=self.device_page_views,
            dst_layers=self.data_refs,
            src_indices=device_rows,
            dst_indices=host_rows,
            page_size=1,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=[self.kv_buffer[layer_id]],
            dst_layers=[self.device_page_views[layer_id]],
            src_indices=host_rows,
            dst_indices=device_rows,
            page_size=1,
        )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.swa_page_size
        data_page = torch.stack(
            [self.kv_buffer[i][index] for i in range(self.layer_num)]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.swa_page_size
        data = data_page.view(self.dtype).reshape(self.layer_num, self.state_page_bytes)
        for i in range(self.layer_num):
            self.kv_buffer[i][index].copy_(data[i])

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        for row in rows:
            for layer_id in range(self.layer_num):
                ptr = (
                    self.kv_buffer[layer_id].data_ptr()
                    + int(row) * self.state_page_bytes * self.dtype.itemsize
                )
                ptr_list.append(ptr)
        element_size = self.state_page_bytes * self.dtype.itemsize
        return ptr_list, [element_size] * len(ptr_list)


class DeepSeekV4SingleKVPoolHost:

    def __init__(
        self,
        device_pool: HiSparseC4DevicePool,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):

        assert host_size > 0, "Host size must be specified and greater than 0"
        assert page_size == 1, "Host page size must be 1 for DeepSeekV4SingleKVPoolHost"

        self.device_pool = device_pool
        self.size = host_size
        self.page_size = page_size
        self.num_pages = (self.size + self.page_size - 1) // self.page_size
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.layer_num = device_pool.layer_num
        self.kv_cache_total_dim = device_pool.kv_cache_total_dim

        self.kv_buffer = self.init_kv_buffer()
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.clear()

    def clear(self):
        self.free_slots = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device="cpu"
        )

    def init_kv_buffer(self):
        dims = (self.layer_num, self.size + self.page_size, self.kv_cache_total_dim)
        requested_bytes = (
            self.layer_num
            * (self.size + self.page_size)
            * self.kv_cache_total_dim
            * self.dtype.itemsize
        )
        host_mem = psutil.virtual_memory()
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        host_pool = torch.empty(dims, dtype=self.dtype, device=self.device)
        assert self.pin_memory, "DeepSeekV4SingleKVPoolHost requires pin_memory=True"
        if self.pin_memory:
            torch.cuda.cudart().cudaHostRegister(
                host_pool.data_ptr(), host_pool.numel() * host_pool.element_size(), 0
            )
        return host_pool

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if io_backend != "kernel":
            raise ValueError(f"Unsupported IO backend: {io_backend}")

        from sglang.jit_kernel.deepseek_v4 import hisparse_offload_to_host

        if host_indices.device != device_indices.device:
            host_indices = host_indices.to(device=device_indices.device)
        host_indices_i64 = (
            host_indices.to(torch.int64)
            if host_indices.dtype != torch.int64
            else host_indices
        )
        device_indices_i64 = (
            device_indices.to(torch.int64)
            if device_indices.dtype != torch.int64
            else device_indices
        )
        hisparse_offload_to_host(
            gpu_ptrs=device_pool.data_ptrs,
            cpu_ptrs=self.data_ptrs,
            gpu_indices=device_indices_i64,
            cpu_indices=host_indices_i64,
        )

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)
