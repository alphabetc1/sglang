# `mem_cache/` — Layout Index

This directory holds SGLang's KV-cache and SSM-state subsystem.

## Quick lookup: `(type × scope) → file`

| type \ scope | full (MHA) | mla | nsa | mamba | swa | hybrid_linear | deepseek_v4 | hisparse |
|---|---|---|---|---|---|---|---|---|
| **allocator (unpaged, page_size=1)** | `allocator/unpaged.py` | — | — | — | — | — | — | — |
| **allocator (paged)** | `allocator/paged.py` | `allocator/paged.py` | `allocator/paged.py` | — | — | — | — | — |
| **allocator (swa-style)** | — | — | — | — | `allocator/swa.py` | — | — | — |
| **allocator (hisparse)** | — | — | — | — | — | — | `allocator/hisparse.py` | `allocator/hisparse.py` |
| **pool (device)** | `pool/mha.py` | `pool/mla.py` | `pool/nsa.py` | `pool/mamba.py` | `pool/swa.py` | `pool/hybrid_linear.py` | `pool/deepseek_v4.py` | `pool/hisparse.py` |
| **pool (host)** | `pool_host/mha.py` | `pool_host/mla.py` | `pool_host/nsa.py` | `pool_host/mamba.py` | — | — | `pool_host/deepseek_v4.py` | — |
| **req → token** | `pool/req_to_token.py` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ |
| **cross-family abc** | `pool/base.py` (`KVCache`, `BaseSWAKVPool`) · `allocator/base.py` (`BaseTokenToKVPoolAllocator`) · `pool_host/base.py` (`HostKVCache`) · `pool_host/tensor_allocator.py` (`HostTensorAllocator`) |

## Layer responsibilities

```
              upper layers (scheduler / model_runner / attention backend)
                          │
                          ▼
              ┌────────────────────────────┐
              │  hybrid_cache/             │   multi-pool router for hybrid models
              │  controller + pool_assembler│
              └────────────────────────────┘
                          │ uses
                          ▼
        ┌──────────────────────────────────────────┐
        │  allocator/                              │   indexing: "give me N slots"
        │  (Base / Unpaged / Paged / SWA / HiSparse)│
        └──────────────────────────────────────────┘
                          │ owns reference to ↓
                          ▼
        ┌──────────────────────────────────────────┐                    ┌──────────────────────────┐
        │  pool/  (device storage, L1)             │  ── hicache ───►   │  pool_host/  (L2, CPU)   │
        │  KV / SSM state physical holder          │   offload/reload   │  host mirror             │
        │  MHA/MLA/NSA/Mamba/SWA/DSv4/HiSparse     │  ◄─────────────    │  + DeepSeekV4 host       │
        └──────────────────────────────────────────┘                    └──────────────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │ storage/  (L3)   │
                                                │ aibrix/hf3fs/    │
                                                │ lmcache/mooncake │
                                                └──────────────────┘
```

| Module | Cares about | Input → Output |
|---|---|---|
| `allocator/` | "which slots are free" | `need_size` → `indices` |
| `pool/` | physical KV / SSM-state layout | `(layer_id, indices) ↔ tensor` |
| `pool_host/` | host mirror + H2D/D2H transfer | `device_indices ↔ host_indices` |
| `hybrid_cache/` | per-layer routing across pools | `layer_id → pool` |
| `storage/` | L3 KV backends (disk/remote) | external plugin system |

## Two orthogonal axes

```
allocation strategy (allocator/)              storage format (pool/)
• Unpaged / Paged / SWA / HiSparse            • MHA / MLA / NSA / Mamba / SWA / DSv4 / HiSparse
```

Any combination is valid (Paged + MHA, Paged + MLA, HiSparse + DSv4). Orthogonal axes must live in separate modules.

## Naming convention

> **Remove role affixes that don't differentiate**: keep `_backend` / `_controller` style only when same-directory files have *different* roles; drop them when all files share the role the directory already names.

Examples:
- `attention/*_backend.py` — keep (`attention_registry.py` is not a backend)
- `managers/*_controller.py` — keep (mixed roles in dir)
- `models/qwen.py` — no `_model` suffix (every file in `models/` is a model)
- `unified_cache_components/full.py` — no `_component` suffix (every file is a Component)

## HiSparse is a "horizontal feature"

HiSparse is not an attention family — it's a sparsification strategy applied across families. Rule:
1. If a class **is HiSparse** (regardless of underlying family) → `*/hisparse.py`
2. If a class is **pure X** (no HiSparse) → `*/x.py`
3. **No file = no instances of that type** (don't create empty placeholder files)

So `allocator/deepseek_v4.py` doesn't exist (no pure DSv4 allocator — DSv4 uses paged); `pool_host/hisparse.py` doesn't exist (HiSparse runs entirely on device today).

## Hardware-specific extensions

Hardware-specific allocators/pools (NPU, MLX) live in `python/sglang/srt/hardware_backend/<hw>/`, **not here**:

| Hardware backend file | Extends |
|---|---|
| `hardware_backend/npu/allocator_npu.py` :: `NPUPagedTokenToKVPoolAllocator` | `PagedTokenToKVPoolAllocator` |
| `hardware_backend/npu/memory_pool_npu.py` :: `NPUMHATokenToKVPool` | `MHATokenToKVPool` |
| `hardware_backend/npu/memory_pool_npu.py` :: `NPUMLATokenToKVPool` | `MLATokenToKVPool` |
| `hardware_backend/mlx/kv_cache/*.py` | (parallel implementation) |

Dependency direction is `hardware_backend → mem_cache`, never reverse.

## Related but not in this directory

| File | Location | Notes |
|---|---|---|
| `HiCacheController` (L1/L2/L3 transfer orchestrator) | `managers/cache_controller.py` | Heavy import of `pool/` and `pool_host/`; candidate to migrate into `mem_cache/hicache/` in a follow-up. |
| `HiSparseCoordinator` (HiSparse ↔ host pool transfer) | `managers/hisparse_coordinator.py` | File-top comment self-tags as TODO; candidate to migrate into `mem_cache/sparsity/`. |

## Tree-layer files

The following files live at the `mem_cache/` root but are **tree-layer** (cache eviction / prefix matching) — out of scope for the pool/allocator restructure (Issue #24335). They belong to the radix-tree unification track (Issue #20415):

- `base_prefix_cache.py`
- `radix_cache.py` / `radix_cache_cpp.py` / `cpp_radix_tree/`
- `swa_radix_cache.py` / `mamba_radix_cache.py` / `hi_mamba_radix_cache.py`
- `hiradix_cache.py` / `unified_radix_cache.py`
- `chunk_cache.py`
- `unified_cache_components/` (subpackage)
- `evict_policy.py`

## Other top-level files (out of restructure scope)

| File | Purpose |
|---|---|
| `cache_init_params.py` | dataclass for cache initialization parameters |
| `common.py` | shared utilities |
| `events.py` | `KVCacheEventMixin` for disaggregation events |
| `flush_cache.py` | argparse + requests CLI script |
| `hicache_storage.py` | hicache ↔ storage backend bridge |
| `multimodal_cache.py` | multimodal feature |
| `utils.py` | generic utilities |
| `sparsity/` | sparsity-feature subpackage |
| `storage/` | L3 storage backends (aibrix, hf3fs, lmcache, mooncake, nixl, simm, eic) |
