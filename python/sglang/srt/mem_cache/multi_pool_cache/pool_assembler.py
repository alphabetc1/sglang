from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.multi_pool_cache.multi_pool_controller import (
    MultiPoolCacheController,
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
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
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
    )


def build_draft_host_pool(
    *,
    draft_pool: Any,
    anchor_host_pool: Any,
    page_size: int,
    server_args: "ServerArgs",
):
    """Create a host pool for a draft KV sidecar."""
    from sglang.srt.mem_cache.memory_pool import (
        MHATokenToKVPool,
        MLATokenToKVPool,
        NSATokenToKVPool,
    )

    host_to_device_ratio = anchor_host_pool.size / draft_pool.size
    common_kw = dict(
        host_to_device_ratio=host_to_device_ratio,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    if isinstance(draft_pool, NSATokenToKVPool):
        return MLATokenToKVPoolHost(
            draft_pool,
            **common_kw,
            override_kv_cache_dim=draft_pool.kv_cache_dim,
        )
    if isinstance(draft_pool, MLATokenToKVPool):
        return MLATokenToKVPoolHost(draft_pool, **common_kw)
    if isinstance(draft_pool, MHATokenToKVPool):
        return MHATokenToKVPoolHost(draft_pool, **common_kw)
    raise ValueError(f"Draft pool type {type(draft_pool).__name__} not supported")


def append_draft_entries(
    entries: list[PoolEntry],
    *,
    draft_pool: Any,
    anchor_host_pool: Any,
    current_transfer_layer_num: int,
    page_size: int,
    server_args: "ServerArgs",
) -> int:
    """Append DRAFT (+ optional DRAFT_INDEXER) sidecar entries to *entries*.

    If *draft_pool* is ``None``, the list is left unchanged and
    *current_transfer_layer_num* is returned as-is.

    Returns the (possibly updated) ``transfer_layer_num``.
    """
    if draft_pool is None:
        return current_transfer_layer_num

    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool

    if isinstance(draft_pool, HybridLinearKVPool):
        draft_pool = draft_pool.full_kv_pool

    draft_host = build_draft_host_pool(
        draft_pool=draft_pool,
        anchor_host_pool=anchor_host_pool,
        page_size=page_size,
        server_args=server_args,
    )

    draft_layer_num = draft_pool.layer_num
    transfer_layer_num = max(current_transfer_layer_num, draft_layer_num)
    draft_mapping = {i: i for i in range(draft_layer_num)}

    entries.append(
        build_pool_entry(
            name=PoolName.DRAFT,
            host_pool=draft_host,
            device_pool=draft_pool,
            layer_mapping=draft_mapping,
            transfer_layer_num=transfer_layer_num,
            share_indices_with_anchor=True,
        )
    )

    # NSA draft model also needs its own indexer sidecar
    if isinstance(draft_pool, NSATokenToKVPool):
        draft_indexer_host = NSAIndexerPoolHost(
            draft_pool,
            draft_host,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        entries.append(
            build_pool_entry(
                name=PoolName.DRAFT_INDEXER,
                host_pool=draft_indexer_host,
                device_pool=draft_pool,
                layer_mapping=draft_mapping,
                transfer_layer_num=transfer_layer_num,
                share_indices_with_anchor=True,
            )
        )

    # When draft has more layers, rebuild earlier entries with the new
    # transfer_layer_num so that their layer_mapper covers the full range.
    if transfer_layer_num > current_transfer_layer_num:
        for i, entry in enumerate(entries):
            if entry.name in (PoolName.DRAFT, PoolName.DRAFT_INDEXER):
                continue
            layer_mapping = {
                lid: lid for lid in range(entry.device_pool.layer_num)
            }
            entries[i] = build_pool_entry(
                name=entry.name,
                host_pool=entry.host_pool,
                device_pool=entry.device_pool,
                layer_mapping=layer_mapping,
                transfer_layer_num=transfer_layer_num,
                is_anchor=entry.is_primary_index_anchor,
                share_indices_with_anchor=entry.share_indices_with_anchor,
                host_evict_fn=entry.host_evict_fn,
                device_evict_fn=entry.device_evict_fn,
            )

    return transfer_layer_num


def _register_draft_counter(draft_pool, layer_done_counter):
    """Register layer transfer counter on the draft pool for prefetch sync."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    pool = draft_pool
    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool
    pool.register_layer_transfer_counter(layer_done_counter)


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
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
) -> tuple[HostPoolGroup, MultiPoolCacheController]:
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
    cache_controller = MultiPoolCacheController(
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
) -> tuple[HostPoolGroup, MultiPoolCacheController]:
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
    cache_controller = MultiPoolCacheController(
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
) -> tuple[HostPoolGroup, MultiPoolCacheController]:
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
    cache_controller = MultiPoolCacheController(
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


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
) -> None:
    """Attach HostPoolGroup + MultiPoolCacheController to UnifiedRadixCache."""
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MLATokenToKVPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kvcache, HybridLinearKVPool):
            full_kv_pool = kvcache.full_kv_pool
            use_mla = kvcache.use_mla
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.MAMBA,
            }, "HybridLinearKVPool currently only supports FULL + MAMBA in UnifiedRadixCache."
        else:
            full_kv_pool = kvcache
            use_mla = isinstance(kvcache, MLATokenToKVPool)
            assert set(cache.components.keys()) == {
                ComponentType.FULL
            }, "Non-hybrid KV pool currently only supports FULL-only UnifiedRadixCache."

        mamba_stack = isinstance(kvcache, HybridLinearKVPool)
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

        logger.info(
            "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
            "KV + MAMBA" if mamba_stack else "KV",
            transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def attach_multi_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
    draft_pool=None,
) -> None:
    """Attach HostPoolGroup + MultiPoolCacheController for HiRadixCache.

    Handles: KV+Indexer (NSA), KV+Draft, KV+Indexer+Draft+DraftIndexer (NSA+draft).
    """
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

    try:
        kv = radix_cache.kv_cache
        layer_num = kv.layer_num
        layer_mapping = {i: i for i in range(layer_num)}
        transfer_layer_num = layer_num

        is_nsa = isinstance(kv, NSATokenToKVPool)

        # --- KV anchor host pool ---
        if is_nsa:
            kv_host_pool = build_kv_host_pool(
                kv_pool=kv,
                page_size=radix_cache.page_size,
                server_args=server_args,
                use_mla=True,
                override_kv_cache_dim=kv.kv_cache_dim,
            )
        else:
            # MHA/MLA: already created in __init__
            kv_host_pool = radix_cache.token_to_kv_pool_host

        entries = [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=kv,
                layer_mapping=layer_mapping,
                transfer_layer_num=transfer_layer_num,
                is_anchor=True,
            ),
        ]

        # --- NSA/DSA indexer sidecar ---
        if is_nsa:
            indexer_host_pool = NSAIndexerPoolHost(
                kv,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
            entries.append(
                build_pool_entry(
                    name=PoolName.INDEXER,
                    host_pool=indexer_host_pool,
                    device_pool=kv,
                    layer_mapping=layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                    share_indices_with_anchor=True,
                ),
            )

        # --- Speculative draft KV sidecar ---
        transfer_layer_num = append_draft_entries(
            entries,
            draft_pool=draft_pool,
            anchor_host_pool=kv_host_pool,
            current_transfer_layer_num=transfer_layer_num,
            page_size=radix_cache.page_size,
            server_args=server_args,
        )

        host_pool_group, cache_controller = MultiPoolCacheController.from_entries(
            entries,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            page_size=radix_cache.page_size,
            transfer_layer_num=transfer_layer_num,
            tp_group=radix_cache.tp_group,
            load_cache_event=load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
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

        if draft_pool is not None:
            _register_draft_counter(draft_pool, cache_controller.layer_done_counter)

        pool_names = [str(e.name).upper() for e in entries]
        logger.info(
            "Attached multi-pool stack to HiRadixCache: pools=%s, "
            "transfer_layer_num=%s",
            " + ".join(pool_names),
            transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_multi_pool_to_hiradix_cache failed")
        raise


def attach_multi_pool_to_mamba_cache(
    mamba_cache: HiMambaRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
    draft_pool=None,
) -> None:
    """Attach HostPoolGroup + MultiPoolCacheController for HiMambaRadixCache."""
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)

        kv_host_pool = build_kv_host_pool(
            kv_pool=kvcache,
            page_size=params.page_size,
            server_args=server_args,
            use_mla=hybrid_kv.use_mla,
        )
        mamba_host_pool = MambaPoolHost(
            params.req_to_token_pool.mamba_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            allocator_type=server_args.hicache_storage_backend,
            layout=server_args.hicache_mem_layout,
        )
        entries = [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=kvcache,
                layer_mapping=full_layer_mapping,
                transfer_layer_num=transfer_layer_num,
                is_anchor=True,
            ),
            build_pool_entry(
                name=PoolName.MAMBA,
                host_pool=mamba_host_pool,
                device_pool=params.req_to_token_pool.mamba_pool,
                layer_mapping=mamba_layer_mapping,
                transfer_layer_num=transfer_layer_num,
                host_evict_fn=mamba_cache.evict_mamba_host,
                device_evict_fn=mamba_cache.evict_mamba,
            ),
        ]

        # --- Speculative draft KV sidecar ---
        transfer_layer_num = append_draft_entries(
            entries,
            draft_pool=draft_pool,
            anchor_host_pool=kv_host_pool,
            current_transfer_layer_num=transfer_layer_num,
            page_size=params.page_size,
            server_args=server_args,
        )

        host_pool_group, cache_controller = MultiPoolCacheController.from_entries(
            entries,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            page_size=params.page_size,
            transfer_layer_num=transfer_layer_num,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
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
        mamba_cache.transfer_layer_num = transfer_layer_num
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)

        if draft_pool is not None:
            _register_draft_counter(draft_pool, cache_controller.layer_done_counter)

        pool_names = [str(e.name).upper() for e in entries]
        logger.info(
            "Attached multi-pool stack to HiMambaRadixCache: pools=%s, "
            "transfer_layer_num=%s",
            " + ".join(pool_names),
            transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_multi_pool_to_mamba_cache failed")
        raise
