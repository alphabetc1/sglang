"""Host-side mirror pools for DeepSeek-V4 compressed KV components.

c4 is the anchor (primary) pool; c4_indexer and c128 are side pools that
share allocation indices with c4 via PoolEntry.share_indices_with_anchor=True
(same mechanism NSA's NSAIndexerPoolHost uses).

DeepSeekV4SingleKVPool device-side buffer layout:
- per-layer 2D tensor `kv_buffer[L]` of shape (num_pages, bytes_per_page_padded), uint8
- bytes_per_token = 584 (FP8 nope + BF16 rope + scales + pad)
- bytes_per_page_padded = ceil_div(page_size * 584, 576) * 576  (page-level alignment)
- token slot S maps to bytes `kv_buffer[L][S // page_size, (S % page_size) * 584 : ... + 584]`

The MLA host transfer kernels assume a flat (size, kv_cache_dim) device layout, so
they cannot be reused. C4HostPool and C128HostPool override `load_to_device_per_layer`
and `backup_from_device_all_layer` with page-aware torch indexing using as_strided.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4IndexerPool,
        DeepSeekV4SingleKVPool,
    )

logger = logging.getLogger(__name__)


def _dsv4_dev_view(
    layer_buffer: torch.Tensor,
    dev_page_size: int,
    bytes_per_token: int,
    bytes_per_page_padded: int,
) -> torch.Tensor:
    """View a DSV4 device layer buffer as (num_pages, dev_page_size, bytes_per_token).

    Underlying storage is (num_pages, bytes_per_page_padded). dev_page_size is
    the device pool's page size in COMPRESSED positions (64 for c4, 2 for c128).
    """
    num_pages = layer_buffer.shape[0]
    return layer_buffer.as_strided(
        size=(num_pages, dev_page_size, bytes_per_token),
        stride=(bytes_per_page_padded, bytes_per_token, 1),
    )


def _dsv4_split_indices(
    raw_indices: torch.Tensor, host_page_size: int, compress_ratio: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split allocator slot indices (in RAW token units) into the device
    buffer's (page_id, offset_in_compressed_page) coordinates.

    The radix-tree allocator hands out raw-token indices at host_page_size
    granularity (e.g. 256). Each radix page maps 1:1 to one compressed page in
    the device buffer; within that page the device stores 1 compressed slot
    per `compress_ratio` raw tokens.
    """
    page_ids = raw_indices // host_page_size
    offsets = (raw_indices % host_page_size) // compress_ratio
    return page_ids, offsets


def _dsv4_swa_split_indices(
    swa_indices: torch.Tensor, dev_page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """SWA flavour: input indices are already in SWA-allocator slot space
    (per-token, not raw-token-id), so addressing is direct paging.
    """
    return swa_indices // dev_page_size, swa_indices % dev_page_size


def _dsv4_swa_load_per_layer(
    host_pool: MLATokenToKVPoolHost,
    device_pool: "DeepSeekV4SingleKVPool",
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    layer_id: int,
) -> None:
    """H -> D for one SWA layer (no compression, simple paging)."""
    if host_pool.layout != "layer_first":
        raise NotImplementedError(
            f"DSV4 SWA host pool currently supports only layout=layer_first; "
            f"got {host_pool.layout}"
        )
    if host_indices.numel() == 0:
        return
    bytes_per_token = device_pool.kv_cache_total_dim
    dev_page_size = device_pool.page_size
    bytes_per_page_padded = device_pool.bytes_per_page_padded

    host_idx_cpu = host_indices.to("cpu", non_blocking=True)
    dev_idx_cuda = (
        device_indices
        if device_indices.is_cuda
        else device_indices.to(device_pool.kv_buffer[0].device)
    )
    dev_view = _dsv4_dev_view(
        device_pool.kv_buffer[layer_id],
        dev_page_size,
        bytes_per_token,
        bytes_per_page_padded,
    )
    page_ids, offsets = _dsv4_swa_split_indices(dev_idx_cuda, dev_page_size)
    src_chunks_cpu = host_pool.kv_buffer[layer_id, host_idx_cpu, 0, :]
    dev_view[page_ids, offsets] = src_chunks_cpu.to(dev_view.device, non_blocking=True)


def _dsv4_swa_backup_all_layer(
    host_pool: MLATokenToKVPoolHost,
    device_pool: "DeepSeekV4SingleKVPool",
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> None:
    """D -> H for all layers of the SWA pool (no compression)."""
    if host_pool.layout != "layer_first":
        raise NotImplementedError(
            f"DSV4 SWA host pool currently supports only layout=layer_first; "
            f"got {host_pool.layout}"
        )
    if host_indices.numel() == 0:
        return
    bytes_per_token = device_pool.kv_cache_total_dim
    dev_page_size = device_pool.page_size
    bytes_per_page_padded = device_pool.bytes_per_page_padded

    host_idx_cpu = host_indices.to("cpu", non_blocking=True)
    dev_idx_cuda = (
        device_indices
        if device_indices.is_cuda
        else device_indices.to(device_pool.kv_buffer[0].device)
    )
    page_ids, offsets = _dsv4_swa_split_indices(dev_idx_cuda, dev_page_size)
    for layer_id in range(host_pool.layer_num):
        dev_view = _dsv4_dev_view(
            device_pool.kv_buffer[layer_id],
            dev_page_size,
            bytes_per_token,
            bytes_per_page_padded,
        )
        chunks_cpu = dev_view[page_ids, offsets].cpu()
        host_pool.kv_buffer[layer_id, host_idx_cpu, 0, :] = chunks_cpu


def _dsv4_load_per_layer(
    host_pool: MLATokenToKVPoolHost,
    device_pool: "DeepSeekV4SingleKVPool",
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    layer_id: int,
) -> None:
    """H -> D for one DSV4 layer. Layer_first host layout only."""
    if host_pool.layout != "layer_first":
        raise NotImplementedError(
            f"DSV4 host pool currently supports only layout=layer_first; "
            f"got {host_pool.layout}"
        )
    if host_indices.numel() == 0:
        return
    bytes_per_token = device_pool.kv_cache_total_dim
    dev_page_size = device_pool.page_size  # compressed positions per page
    bytes_per_page_padded = device_pool.bytes_per_page_padded
    host_page_size = host_pool.page_size  # raw tokens per radix page
    compress_ratio = host_page_size // dev_page_size

    host_idx_cpu = host_indices.to("cpu", non_blocking=True)
    dev_idx_cuda = (
        device_indices
        if device_indices.is_cuda
        else device_indices.to(device_pool.kv_buffer[0].device)
    )
    dev_view = _dsv4_dev_view(
        device_pool.kv_buffer[layer_id],
        dev_page_size,
        bytes_per_token,
        bytes_per_page_padded,
    )
    page_ids, offsets = _dsv4_split_indices(
        dev_idx_cuda, host_page_size, compress_ratio
    )
    # host buffer layout_first shape: (layer, size, 1, kv_cache_dim)
    src_chunks_cpu = host_pool.kv_buffer[layer_id, host_idx_cpu, 0, :]
    dev_view[page_ids, offsets] = src_chunks_cpu.to(dev_view.device, non_blocking=True)


def _dsv4_backup_all_layer(
    host_pool: MLATokenToKVPoolHost,
    device_pool: "DeepSeekV4SingleKVPool",
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> None:
    """D -> H for all layers of one DSV4 single-KV pool."""
    if host_pool.layout != "layer_first":
        raise NotImplementedError(
            f"DSV4 host pool currently supports only layout=layer_first; "
            f"got {host_pool.layout}"
        )
    if host_indices.numel() == 0:
        return
    bytes_per_token = device_pool.kv_cache_total_dim
    dev_page_size = device_pool.page_size
    bytes_per_page_padded = device_pool.bytes_per_page_padded
    host_page_size = host_pool.page_size
    compress_ratio = host_page_size // dev_page_size

    host_idx_cpu = host_indices.to("cpu", non_blocking=True)
    dev_idx_cuda = (
        device_indices
        if device_indices.is_cuda
        else device_indices.to(device_pool.kv_buffer[0].device)
    )
    page_ids, offsets = _dsv4_split_indices(
        dev_idx_cuda, host_page_size, compress_ratio
    )
    for layer_id in range(host_pool.layer_num):
        dev_view = _dsv4_dev_view(
            device_pool.kv_buffer[layer_id],
            dev_page_size,
            bytes_per_token,
            bytes_per_page_padded,
        )
        chunks_cpu = dev_view[page_ids, offsets].cpu()  # explicit D->H
        host_pool.kv_buffer[layer_id, host_idx_cpu, 0, :] = chunks_cpu


class C4HostPool(MLATokenToKVPoolHost):
    """Host mirror of c4 (DeepSeekV4SingleKVPool). Anchor pool."""

    def __init__(
        self,
        device_pool: "DeepSeekV4SingleKVPool",
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ) -> None:
        # kv_cache_total_dim is set by DeepSeekV4SingleKVPool.create_buffer() and
        # equals get_bytes_per_token(): the raw byte count per token stored as uint8.
        # Since store_dtype=torch.uint8 (itemsize=1), kv_cache_dim == bytes_per_token.
        kv_cache_dim = device_pool.kv_cache_total_dim
        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            override_kv_cache_dim=kv_cache_dim,
        )
        logger.info(
            "C4HostPool size=%d pages=%d layers=%d kv_cache_dim=%d layout=%s",
            self.size,
            self.page_num,
            self.layer_num,
            kv_cache_dim,
            layout,
        )

    def get_size_per_token(self):
        # DeepSeekV4SingleKVPool does not have kv_lora_rank; override to avoid
        # AttributeError and use override_kv_cache_dim directly.
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.override_kv_cache_dim
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        _dsv4_load_per_layer(self, device_pool, host_indices, device_indices, layer_id)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _dsv4_backup_all_layer(self, device_pool, host_indices, device_indices)


class C128HostPool(MLATokenToKVPoolHost):
    """Host mirror of c128 (DeepSeekV4SingleKVPool, per-token KV).

    Sized to the c4 anchor's host pool size (NOT c128_kv_pool.size × ratio):
    c128 is registered as a share-indices side pool of c4, so the controller
    feeds it the anchor's raw-token host slot indices. We therefore need
    enough host slots to address [0, anchor_host.size). The c128 *device*
    buffer itself is small (c128_size × 1/128 of full), so 128 redundant
    raw-token slots collapse to the same c128 device entry on backup —
    wasteful but functionally a 1:1 share with anchor.
    """

    def __init__(
        self,
        device_pool: "DeepSeekV4SingleKVPool",
        anchor_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ) -> None:
        kv_cache_dim = device_pool.kv_cache_total_dim
        # Drive the parent's host_size>0 path so size === anchor_size after
        # page-alignment. size_per_token = layer_num * itemsize * kv_cache_dim;
        # multiplying by anchor_size and dividing by 1e9 yields a host_size in
        # GB that the parent then converts back to a slot count.
        size_per_token_bytes = (
            device_pool.layer_num * device_pool.store_dtype.itemsize * kv_cache_dim
        )
        host_size_gb = float(anchor_size * size_per_token_bytes) / 1e9
        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=1.0,  # ignored; host_size>0 takes precedence
            host_size=host_size_gb,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            override_kv_cache_dim=kv_cache_dim,
        )
        logger.info(
            "C128HostPool size=%d pages=%d layers=%d kv_cache_dim=%d layout=%s "
            "(sized to c4 anchor for share-indices; %.2f GB)",
            self.size,
            self.page_num,
            self.layer_num,
            kv_cache_dim,
            layout,
            self.size * size_per_token_bytes / 1e9,
        )

    def get_size_per_token(self):
        # DeepSeekV4SingleKVPool does not have kv_lora_rank; override to avoid
        # AttributeError and use override_kv_cache_dim directly.
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.override_kv_cache_dim
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        _dsv4_load_per_layer(self, device_pool, host_indices, device_indices, layer_id)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _dsv4_backup_all_layer(self, device_pool, host_indices, device_indices)


class SWAHostPool(MLATokenToKVPoolHost):
    """Host mirror of swa_kv_pool (DeepSeekV4SingleKVPool).

    Independent allocation (NOT share-indices): SWA uses its own swa_attn
    namespace, separate from full_attn. SWAComponent emits PoolTransfer with
    swa-translated device indices on backup; the controller allocates host
    slots from this pool independently. compress_ratio = 1 (per-token SWA).
    """

    def __init__(
        self,
        device_pool: "DeepSeekV4SingleKVPool",
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ) -> None:
        kv_cache_dim = device_pool.kv_cache_total_dim
        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            override_kv_cache_dim=kv_cache_dim,
        )
        logger.info(
            "SWAHostPool size=%d pages=%d layers=%d kv_cache_dim=%d layout=%s",
            self.size,
            self.page_num,
            self.layer_num,
            kv_cache_dim,
            layout,
        )

    def get_size_per_token(self):
        # Same override as C4HostPool.
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.override_kv_cache_dim
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        _dsv4_swa_load_per_layer(
            self, device_pool, host_indices, device_indices, layer_id
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _dsv4_swa_backup_all_layer(self, device_pool, host_indices, device_indices)


class C4IndexerHostPool(NSAIndexerPoolHost):
    """Host mirror of c4_indexer (DeepSeekV4IndexerPool). Side pool sharing anchor's slot space."""

    def __init__(
        self,
        device_pool: "DeepSeekV4IndexerPool",
        anchor_host: C4HostPool,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ) -> None:
        # NSAIndexerPoolHost.__init__ works for DeepSeekV4IndexerPool because
        # they share the same attribute names:
        #   - device_pool.index_k_with_scale_buffer  (per-layer list of tensors)
        #   - device_pool.index_head_dim
        #   - device_pool.quant_block_size            (class attr = 128)
        #   - type(device_pool).index_k_with_scale_buffer_dtype = torch.uint8
        #
        # The only incompatibility is the hard-coded NSA class reference:
        #   self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        # We re-bind it immediately after super().__init__ to the DSV4 dtype.
        super().__init__(
            device_pool=device_pool,
            anchor_host=anchor_host,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
        )
        # Re-bind to the DSV4 dtype in case super() hard-referenced NSA's class attribute.
        self.indexer_dtype = type(device_pool).index_k_with_scale_buffer_dtype
        # Cache derived stride params for the page-aware transfer overrides.
        self._dev_page_size = device_pool.page_size  # compressed-position page size
        self._host_page_size = anchor_host.page_size  # anchor (raw-token) page size
        self._dev_page_bytes = device_pool.index_k_with_scale_buffer[0].shape[1]
        self._host_page_stride = self.indexer_page_stride_size

    # ---- DSV4-aware indexer transfer ----
    #
    # NSAIndexerPoolHost's transfers assume host and device share the same
    # page_size and use page-aligned indices. For DSV4 the host's page size
    # (raw tokens, 256) differs from the indexer device's page size (compressed
    # positions, 64), so we override with as_strided byte-range copies that
    # work on token-level slot indices directly.
    def _indexer_dev_view(self, layer_id: int) -> torch.Tensor:
        buf = self.device_pool.index_k_with_scale_buffer[layer_id]
        num_dev_pages = buf.shape[0]
        return buf.as_strided(
            size=(num_dev_pages, self._dev_page_size, self.indexer_size_per_token),
            stride=(self._dev_page_bytes, self.indexer_size_per_token, 1),
        )

    def _indexer_host_view(self, layer_id: int) -> torch.Tensor:
        # NSAIndexerPoolHost allocates `(layer_num, indexer_page_num,
        # indexer_page_stride_size)` for layer_first.
        host_buf = self.index_k_with_scale_buffer[layer_id]
        return host_buf.as_strided(
            size=(
                self.indexer_page_num,
                self._host_page_size,
                self.indexer_size_per_token,
            ),
            stride=(self._host_page_stride, self.indexer_size_per_token, 1),
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices.numel() == 0:
            return
        if self.layout != "layer_first":
            raise NotImplementedError(
                f"DSV4 c4_indexer host pool currently supports only layer_first; "
                f"got {self.layout}"
            )
        compress_ratio = self._host_page_size // self._dev_page_size
        host_idx_cpu = host_indices.to("cpu", non_blocking=True)
        h_pages = host_idx_cpu // self._host_page_size
        h_off = host_idx_cpu % self._host_page_size
        dev_idx_cuda = (
            device_indices
            if device_indices.is_cuda
            else device_indices.to(device_pool.index_k_with_scale_buffer[0].device)
        )
        # Device indexer page_size is in COMPRESSED positions; raw indices need
        # the same translation as c4 (page_id = S // host_page_size, offset =
        # (S % host_page_size) // compress_ratio).
        d_pages = dev_idx_cuda // self._host_page_size
        d_off = (dev_idx_cuda % self._host_page_size) // compress_ratio
        host_view = self._indexer_host_view(layer_id)
        dev_view = self._indexer_dev_view(layer_id)
        chunks_cpu = host_view[h_pages, h_off]
        dev_view[d_pages, d_off] = chunks_cpu.to(dev_view.device, non_blocking=True)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices.numel() == 0:
            return
        if self.layout != "layer_first":
            raise NotImplementedError(
                f"DSV4 c4_indexer host pool currently supports only layer_first; "
                f"got {self.layout}"
            )
        compress_ratio = self._host_page_size // self._dev_page_size
        host_idx_cpu = host_indices.to("cpu", non_blocking=True)
        h_pages = host_idx_cpu // self._host_page_size
        h_off = host_idx_cpu % self._host_page_size
        dev_idx_cuda = (
            device_indices
            if device_indices.is_cuda
            else device_indices.to(device_pool.index_k_with_scale_buffer[0].device)
        )
        d_pages = dev_idx_cuda // self._host_page_size
        d_off = (dev_idx_cuda % self._host_page_size) // compress_ratio
        for layer_id in range(self.layer_num):
            dev_view = self._indexer_dev_view(layer_id)
            host_view = self._indexer_host_view(layer_id)
            chunks_cpu = dev_view[d_pages, d_off].cpu()
            host_view[h_pages, h_off] = chunks_cpu
