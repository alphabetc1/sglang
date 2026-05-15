"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.pool.base import KVCache, get_tensor_size_bytes
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = use_nsa
        self.nsa_kv_cache_store_fp8 = (
            use_nsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with nsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_nsa:
            # NSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if _is_hip and self.use_nsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            set_mla_kv_buffer_triton_fp8_quant(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.nsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices, mamba_indices=None):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )
