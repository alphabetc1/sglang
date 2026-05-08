from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
    PoolEntry,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _make_layer_mapper(
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
) -> Callable[[int], Optional[int]]:
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    server_args: ServerArgs,
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
):
    kv_host_pool_cls = MLATokenToKVPoolHost if use_mla else MHATokenToKVPoolHost
    kwargs = {}
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    return kv_host_pool_cls(
        kv_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        **kwargs,
    )


def build_pool_entry(
    *,
    name: PoolName,
    host_pool: Any,
    device_pool: Any,
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
    is_anchor: bool = False,
    share_indices_with_anchor: bool = False,
    host_evict_fn: Optional[Callable[[int], Any]] = None,
    device_evict_fn: Optional[Callable[[int], Any]] = None,
    device_alloc_fn: Optional[Callable[[int], Any]] = None,
    device_free_fn: Optional[Callable[[Any], Any]] = None,
) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num),
        is_primary_index_anchor=is_anchor,
        share_indices_with_anchor=share_indices_with_anchor,
        host_evict_fn=host_evict_fn,
        device_evict_fn=device_evict_fn,
        device_alloc_fn=device_alloc_fn,
        device_free_fn=device_free_fn,
    )


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        )
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_swa_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )

    # For SWA hybrid, the device alloc/free goes through the inner swa_attn_allocator
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=full_kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        allocator_type=server_args.hicache_storage_backend,
        layout=server_args.hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_shared_anchor_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    shared_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    shared_host_pool_factory: Callable[[Any], Any],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    shared_host_pool = shared_host_pool_factory(kv_host_pool)
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=shared_pool_name,
            host_pool=shared_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            share_indices_with_anchor=True,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_dsv4_compressed_stack(
    *,
    params: "CacheInitParams",
    server_args: "ServerArgs",
    c4_pool: Any,
    c4_indexer_pool: Any,
    c128_pool: Any,
    swa_pool: Any,
    swa_layer_mapping: dict,
    swa_attn_allocator: Any,
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> "tuple[HostPoolGroup, HybridCacheController]":
    """Build host pool group + controller for DSV4 (c4 anchor, c4_indexer
    + c128 share-indices side pools, SWA independent side pool).
    """
    from sglang.srt.mem_cache.dsv4_host_pool import (
        C4HostPool,
        C4IndexerHostPool,
        C128HostPool,
        SWAHostPool,
    )

    # c4 anchor.
    c4_host = C4HostPool(
        c4_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )

    # c128 sized to anchor for share-indices. Wasteful (128:1 slot redundancy)
    # but small in absolute terms (only 2 c128 layers in DSV4-Flash) and avoids
    # any index-translation extension to the share-indices controller path.
    c128_host = C128HostPool(
        c128_pool,
        anchor_size=c4_host.size,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )

    # c4_indexer reuses c4's slot layout (NSAIndexerPoolHost-style).
    c4_indexer_host = C4IndexerHostPool(
        c4_indexer_pool,
        anchor_host=c4_host,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )

    # SWA host pool — independent indices, sized via swa_attn_allocator's pool.
    swa_host = SWAHostPool(
        swa_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )

    c4_layer_mapping = {i: i for i in range(c4_pool.layer_num)}
    c128_layer_mapping = {i: i for i in range(c128_pool.layer_num)}
    transfer_layer_num = max(
        len(c4_layer_mapping),
        len(c128_layer_mapping),
        len(swa_layer_mapping),
    )

    entries = [
        build_pool_entry(
            name=PoolName.C4,
            host_pool=c4_host,
            device_pool=c4_pool,
            layer_mapping=c4_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.C4_INDEXER,
            host_pool=c4_indexer_host,
            device_pool=c4_indexer_pool,
            layer_mapping=c4_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            share_indices_with_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.C128,
            host_pool=c128_host,
            device_pool=c128_pool,
            layer_mapping=c128_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            share_indices_with_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host,
            device_pool=swa_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MLATokenToKVPool,
        NSATokenToKVPool,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        swa_stack = isinstance(kvcache, SWAKVPool)
        mamba_stack = isinstance(kvcache, HybridLinearKVPool)
        nsa_stack = isinstance(kvcache, NSATokenToKVPool)
        dsv4_stack = isinstance(kvcache, DeepSeekV4TokenToKVPool)

        if mamba_stack:
            full_kv_pool = kvcache.full_kv_pool
            use_mla = kvcache.use_mla
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.MAMBA,
            }, "HybridLinearKVPool currently only supports FULL + MAMBA in UnifiedRadixCache."
        elif swa_stack:
            full_kv_pool = kvcache.full_kv_pool
            use_mla = False
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.SWA,
            }, "SWAKVPool currently only supports FULL + SWA in UnifiedRadixCache."
        elif dsv4_stack:
            # DSV4 has no single full_kv_pool; use c4_kv_pool as the anchor.
            full_kv_pool = kvcache.c4_kv_pool
            use_mla = False
        else:
            full_kv_pool = kvcache
            use_mla = isinstance(kvcache, MLATokenToKVPool)
            assert set(cache.components.keys()) == {
                ComponentType.FULL
            }, "Non-hybrid KV pool currently only supports FULL-only UnifiedRadixCache."

        if mamba_stack:
            full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
            mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
            host_pool_group, cache_controller = build_hybrid_mamba_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                mamba_pool=params.req_to_token_pool.mamba_pool,
                full_layer_mapping=full_layer_mapping,
                mamba_layer_mapping=mamba_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
                device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
            cache.components[ComponentType.MAMBA]._mamba_pool_host = (
                cache.mamba_pool_host
            )
            params.req_to_token_pool.register_layer_transfer_counter(
                cache_controller.layer_done_counter
            )
            transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        elif swa_stack:
            full_layer_mapping = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if not is_swa
            }
            swa_layer_mapping = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if is_swa
            }
            host_pool_group, cache_controller = build_hybrid_swa_stack(
                params=params,
                server_args=server_args,
                full_kv_pool=full_kv_pool,
                swa_kv_pool=kvcache.swa_kv_pool,
                full_layer_mapping=full_layer_mapping,
                swa_layer_mapping=swa_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                storage_backend=None,
                use_mla=False,
                host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                device_swa_evict_fn=lambda n: cache.evict(
                    EvictParams(swa_num_tokens=n)
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.swa_kv_pool_host = host_pool_group.get_pool(PoolName.SWA)
            cache.components[ComponentType.SWA]._swa_kv_pool_host = (
                cache.swa_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
        elif nsa_stack:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_shared_anchor_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                shared_pool_name=PoolName.INDEXER,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                storage_backend=None,
                use_mla=use_mla,
                shared_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                    full_kv_pool,
                    kv_host_pool,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
                attn_cp_rank=params.attn_cp_rank,
                attn_cp_size=params.attn_cp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            # Register the NSA indexer pool as sharing anchor-KV indices so
            # HiCache backup/load emits its PoolTransfer together with KV.
            cache.register_hicache_anchor_kv_shared_indices_pool(
                PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)
        elif dsv4_stack:
            assert set(cache.components.keys()) >= {
                ComponentType.FULL,
                ComponentType.SWA,
                ComponentType.DSV4_COMPRESSED,
            }, "DSV4 stack requires (FULL, SWA, DSV4_COMPRESSED) tree components"

            # All-layers iteration; c4/c128 only fire on their respective layers
            # via the per-pool layer mapper.
            full_layer_mapping = {
                global_id: item.compress_layer_id
                for global_id, item in enumerate(kvcache.layer_mapping)
                if item.compress_ratio == 4
            }
            swa_layer_mapping = {i: i for i in range(kvcache.swa_kv_pool.layer_num)}
            allocator = params.token_to_kv_pool_allocator

            host_pool_group, cache_controller = build_dsv4_compressed_stack(
                params=params,
                server_args=server_args,
                c4_pool=kvcache.c4_kv_pool,
                c4_indexer_pool=kvcache.c4_indexer_kv_pool,
                c128_pool=kvcache.c128_kv_pool,
                swa_pool=kvcache.swa_kv_pool,
                swa_layer_mapping=swa_layer_mapping,
                swa_attn_allocator=allocator.swa_attn_allocator,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
                attn_cp_rank=params.attn_cp_rank,
                attn_cp_size=params.attn_cp_size,
            )

            # Standard cache wiring.
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.C4)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller

            # FULL component owns the host_value lifecycle on the c4 anchor.
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            # SWAComponent owns SWA host backup (independent slot space).
            cache.components[ComponentType.SWA]._swa_kv_pool_host = (
                host_pool_group.get_pool(PoolName.SWA)
            )

            # Register c4_indexer + c128 as anchor-shared so write_backup
            # includes their PoolTransfers in extra_pools; the controller fills
            # in their host/device indices from the anchor via
            # share_indices_with_anchor. c128 is sized to the anchor in v2 to
            # eliminate the v1 correctness gap (c128 layers operating on
            # uninitialized memory after host hit).
            cache.register_hicache_anchor_kv_shared_indices_pool(
                PoolName.C4_INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            cache.register_hicache_anchor_kv_shared_indices_pool(
                PoolName.C128,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )

            # Bind c4/c4_indexer/c128 host pool refs (used by DSV4_COMPRESSED's
            # bookkeeping; the actual transfers go through the controller +
            # share-indices flow above, not the component's hicache hooks).
            comp = cache.components[ComponentType.DSV4_COMPRESSED]
            comp._c4_pool_host = host_pool_group.get_pool(PoolName.C4)
            comp._c4_indexer_pool_host = host_pool_group.get_pool(PoolName.C4_INDEXER)
            comp._c128_pool_host = host_pool_group.get_pool(PoolName.C128)

            # Wire alloc/free callbacks on DeepSeekV4TokenToKVPool: c4 / c4_indexer
            # / c128 page-ids share the SWATokenToKVPoolAllocator.full_attn_allocator
            # namespace (DSV4 has SWA, so the token pool allocator wraps full_attn
            # + swa sub-allocators).
            kvcache.register_compressed_free_alloc(
                allocator.full_attn_allocator.free,
                allocator.full_attn_allocator.alloc,
            )

            transfer_layer_num = max(len(full_layer_mapping), len(swa_layer_mapping))
        else:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_kv_only_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)

        kvcache.register_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )

        if mamba_stack:
            pools_desc = "KV + MAMBA"
        elif swa_stack:
            pools_desc = "KV + SWA"
        elif nsa_stack:
            pools_desc = "KV + INDEXER"
        elif dsv4_stack:
            pools_desc = "C4 + C4_INDEXER + C128 + SWA"
        else:
            pools_desc = "KV"
        logger.info(
            "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
            pools_desc,
            transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def attach_hybrid_nsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's NSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_shared_anchor_stack(
            params=params,
            server_args=server_args,
            kv_pool=kv,
            shared_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            page_size=radix_cache.page_size,
            tp_group=radix_cache.tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=True,
            prefetch_threshold=prefetch_threshold,
            shared_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                kv,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=radix_cache.pp_rank,
            pp_size=radix_cache.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid NSA pool stack to HiRadixCache: pools=KV + INDEXER, "
            "transfer_layer_num=%s",
            len(layer_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_nsa_pool_to_hiradix_cache failed")
        raise


def attach_hybrid_pool_to_mamba_cache(
    mamba_cache: HiMambaRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Attach HostPoolGroup (KV + Mamba) + HybridCacheController for HiMambaRadixCache.

    This entrypoint is currently intended only for HiMambaRadixCache.
    """
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            server_args=server_args,
            kv_pool=kvcache,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=params.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=hybrid_kv.use_mla,
            host_mamba_evict_fn=mamba_cache.evict_mamba_host,
            device_mamba_evict_fn=mamba_cache.evict_mamba,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        mamba_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        mamba_cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
        mamba_cache.transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)
        logger.info(
            "Attached hybrid Mamba pool stack to HiMambaRadixCache: pools=KV + MAMBA, "
            "transfer_layer_num=%s",
            mamba_cache.transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_mamba_cache failed")
        raise
