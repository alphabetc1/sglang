# RFC: `mem_cache` pool / allocator restructure

## 1. TL;DR

Reorganize `mem_cache/` from "a few catch-all files plus cross-feature pollution" into per-family one-file modules:

- `allocator.py` → `allocator/` package (5 files + `__init__.py`)
- `memory_pool.py` + 5 sibling files → `pool/` package (10 files, one per family)
- `memory_pool_host.py` → `pool_host/` package (9 files, mirrors `pool/`)
- Untangle DSv4 ↔ HiSparse class misplacement
- Naming rule: **remove role affixes that don't differentiate** — keep `_backend` / `_controller` style only when same-directory files have different roles; drop them when all files share the role the directory already names
- Add `mem_cache/README.md` (type × scope) index

---

## 2. Motivation

`mem_cache/` has four structural problems (per issue #24335 work items):

1. **Blurred type boundaries** — `memory_pool.py` is 2000 lines / 11 classes mixing req-to-slot indexing, KV abc, KV implementations, and SSM state.
2. **Feature-scope leakage** — `deepseek_v4_memory_pool.py` defines `HiSparseC4DevicePool`; `hisparse_memory_pool.py` defines three DSv4 classes including a host-side one.
3. **Same abstraction scattered** — allocator classes live in `allocator.py` / `swa_memory_pool.py` / `hisparse_memory_pool.py`.
4. **Inconsistent naming** — `hybrid_cache/{hybrid_cache_controller, hybrid_pool_assembler}.py` literally repeats the parent directory name; `unified_cache_components/{full,mamba,swa}_component.py` repeats the role that the plural directory name already states (compare `models/qwen.py` not `qwen_model.py`).

---

## 3. Overall directory structure

### 3.1 Before

```
mem_cache/
├── allocator.py                          (3 classes; SWA/HiSparse allocators live elsewhere)
├── memory_pool.py                        (2000 lines / 11 classes — catch-all)
├── memory_pool_host.py                   (1900 lines / 8 classes — host-side catch-all)
├── deepseek_v4_memory_pool.py            ❌ contains HiSparseC4DevicePool
├── deepseek_v4_compress_state.py         (DSv4 sub-module sitting next to its parent file)
├── swa_memory_pool.py                    ❌ contains SWATokenToKVPoolAllocator
├── base_swa_memory_pool.py               (cross-family base, but lives under SWA's own file)
├── hisparse_memory_pool.py               ❌ contains DeepSeekV4SingleKVPoolHost
│                                         ❌ contains HiSparseTokenToKVPoolAllocator
│                                         ❌ contains DeepSeekV4HiSparseTokenToKVPoolAllocator
├── hybrid_cache/
│   ├── hybrid_cache_controller.py        ❌ redundant prefix
│   └── hybrid_pool_assembler.py          ❌ redundant prefix
├── unified_cache_components/
│   ├── full_component.py                 ❌ `_component` redundant — dir name already says "components"
│   ├── mamba_component.py                ❌ same
│   ├── swa_component.py                  ❌ same
│   └── tree_component.py                 ❌ misnamed (actually abc + enums, not "a tree component")
└── ... (tree-layer etc., out of scope)
```

### 3.2 After

```
mem_cache/
├── README.md                             # (type × scope) index
│
├── allocator/                            # allocation strategy layer
│   ├── __init__.py
│   ├── base.py                           # BaseTokenToKVPoolAllocator (abc only)
│   ├── unpaged.py                        # TokenToKVPoolAllocator (page_size=1)
│   ├── paged.py                          # PagedTokenToKVPoolAllocator
│   ├── swa.py                            # SWATokenToKVPoolAllocator
│   └── hisparse.py                       # HiSparseTokenToKVPoolAllocator
│                                         # DeepSeekV4HiSparseTokenToKVPoolAllocator
│
├── pool/                                 # device physical storage
│   ├── __init__.py
│   ├── base.py                           # KVCache + BaseSWAKVPool (cross-family abc)
│   ├── req_to_token.py                   # ReqToTokenPool + HybridReqToTokenPool
│   ├── mha.py                            # MHA + NoOp + FP4
│   ├── mla.py                            # MLA + FP4
│   ├── nsa.py                            # NSATokenToKVPool
│   ├── mamba.py                          # MambaPool
│   ├── hybrid_linear.py                  # HybridLinearKVPool
│   ├── swa.py                            # SWAKVPool
│   ├── hisparse.py                       # HiSparseC4DevicePool + HiSparseNSATokenToKVPool
│   └── deepseek_v4.py                    # All DSv4 classes: Single + Indexer + LayerItem + TokenToKV
│                                         # + CompressStatePool + KVAndScore + helper  (~780 lines)
│
├── pool_host/                            # host-side mirrors for supported device pools
│   ├── __init__.py
│   ├── base.py                           # HostKVCache (abc) + synchronized decorator
│   ├── tensor_allocator.py               # HostTensorAllocator + alloc_with_host_register
│                                         # + alloc_with_pin_memory + get_allocator_from_storage
│   ├── mha.py                            # MHATokenToKVPoolHost
│   ├── mla.py                            # MLATokenToKVPoolHost
│   ├── nsa.py                            # NSAIndexerPoolHost
│   ├── mamba.py                          # MambaPoolHost
│   ├── group.py                          # PoolEntry + HostPoolGroup
│   └── deepseek_v4.py                    # DeepSeekV4SingleKVPoolHost
│
├── hybrid_cache/                         # multi-pool orchestrator (no __init__.py)
│   ├── controller.py                     # HybridCacheController
│   └── pool_assembler.py                 # HybridPoolAssembler
│
├── unified_cache_components/             # tree-layer subpackage (#20415 territory)
│   ├── __init__.py
│   ├── base.py                           # TreeComponent (abc) + shared enums  (was tree_component.py)
│   ├── full.py                           # FullComponent       (was full_component.py)
│   ├── mamba.py                          # MambaComponent      (was mamba_component.py)
│   └── swa.py                            # SWAComponent        (was swa_component.py)
│
└── ... (tree-layer etc., unchanged)
```

---

## 4. Per-module class relocation

### 4.1 `allocator/`

| Class | Before | After |
|---|---|---|
| `BaseTokenToKVPoolAllocator` | `allocator.py:35` | `allocator/base.py` |
| `TokenToKVPoolAllocator` | `allocator.py:121` | `allocator/unpaged.py` |
| `PagedTokenToKVPoolAllocator` | `allocator.py:362` | `allocator/paged.py` |
| `SWATokenToKVPoolAllocator` | `swa_memory_pool.py:296` | `allocator/swa.py` |
| `HiSparseTokenToKVPoolAllocator` | `hisparse_memory_pool.py:138` | `allocator/hisparse.py` |
| `DeepSeekV4HiSparseTokenToKVPoolAllocator` | `hisparse_memory_pool.py:503` | `allocator/hisparse.py` |

### 4.2 `pool/`

| Class | Before | After |
|---|---|---|
| `KVCache` (abc) | `memory_pool.py:692` | `pool/base.py` |
| `BaseSWAKVPool` (abc) | `base_swa_memory_pool.py:9` | `pool/base.py` |
| `ReqToTokenPool` | `memory_pool.py:127` | `pool/req_to_token.py` |
| `HybridReqToTokenPool` | `memory_pool.py:486` | `pool/req_to_token.py` |
| `MHATokenToKVPool` / `NoOpMHATokenToKVPool` / `MHATokenToKVPoolFP4` | `memory_pool.py:788/1135/1245` | `pool/mha.py` |
| `MLATokenToKVPool` / `MLATokenToKVPoolFP4` | `memory_pool.py:1617/1851` | `pool/mla.py` |
| `NSATokenToKVPool` | `memory_pool.py:1980` | `pool/nsa.py` |
| `MambaPool` | `memory_pool.py:194` | `pool/mamba.py` |
| `HybridLinearKVPool` | `memory_pool.py:1388` | `pool/hybrid_linear.py` |
| `SWAKVPool` | `swa_memory_pool.py:29` | `pool/swa.py` |
| `DeepSeekV4SingleKVPool` / `IndexerPool` / `TokenToKVPool` / `LayerItem` | `deepseek_v4_memory_pool.py:44/242/351/345` | `pool/deepseek_v4.py` |
| `KVAndScore` / `CompressStatePool` | `deepseek_v4_compress_state.py:14/36` | `pool/deepseek_v4.py` |
| `HiSparseC4DevicePool` ❌ | `deepseek_v4_memory_pool.py:159` | `pool/hisparse.py` |
| `HiSparseNSATokenToKVPool` | `hisparse_memory_pool.py:39` | `pool/hisparse.py` |

#### DSv4 ↔ HiSparse — what's broken and how it gets fixed

Not an import cycle (verified via `grep`). The real problem is **class misplacement**:

- `HiSparseC4DevicePool` defined at `deepseek_v4_memory_pool.py:159` → moves to `pool/hisparse.py`
- `DeepSeekV4SingleKVPoolHost` defined at `hisparse_memory_pool.py:388` → moves to `pool_host/deepseek_v4.py`

The first misplacement forces `hisparse_memory_pool.py:15` to import its own class back from the DSv4 file. After moving the misplaced classes home, the dependency becomes a clean one-way `pool/hisparse.py → pool/deepseek_v4.py`, aligning with HiSparse's OOP nature (HiSparse extends DSv4).

### 4.3 `pool_host/`

| Class / function | Before | After |
|---|---|---|
| `HostKVCache` (abc) | `memory_pool_host.py:155` | `pool_host/base.py` |
| `synchronized` (decorator) | `memory_pool_host.py:71` | `pool_host/base.py` |
| `HostTensorAllocator` (abc) | `memory_pool_host.py:80` | `pool_host/tensor_allocator.py` |
| `alloc_with_host_register`, `alloc_with_pin_memory`, `get_allocator_from_storage` | `memory_pool_host.py:113/132/94` | `pool_host/tensor_allocator.py` |
| `MHATokenToKVPoolHost` | `memory_pool_host.py:291` | `pool_host/mha.py` |
| `MLATokenToKVPoolHost` | `memory_pool_host.py:788` | `pool_host/mla.py` |
| `MambaPoolHost` | `memory_pool_host.py:1188` | `pool_host/mamba.py` |
| `NSAIndexerPoolHost` | `memory_pool_host.py:1819` | `pool_host/nsa.py` |
| `PoolEntry` / `HostPoolGroup` | `memory_pool_host.py:1660/1683` | `pool_host/group.py` |
| `DeepSeekV4SingleKVPoolHost` ❌ | `hisparse_memory_pool.py:388` | `pool_host/deepseek_v4.py` |

### 4.4 `hybrid_cache/` and `unified_cache_components/`

File renames only (no class moves):

```
hybrid_cache/hybrid_cache_controller.py       → hybrid_cache/controller.py
hybrid_cache/hybrid_pool_assembler.py         → hybrid_cache/pool_assembler.py

unified_cache_components/tree_component.py    → unified_cache_components/base.py
unified_cache_components/full_component.py    → unified_cache_components/full.py
unified_cache_components/mamba_component.py   → unified_cache_components/mamba.py
unified_cache_components/swa_component.py     → unified_cache_components/swa.py
```

Three distinct issues:
- `hybrid_cache_controller.py` literally repeats the parent directory name `hybrid_cache_` → drop prefix.
- `tree_component.py` is **misnamed**: it holds `TreeComponent` (abc) + 4 shared enums (`ComponentType`, `ComponentData`, `EvictLayer`, `CacheTransferPhase`), not "one tree component" → rename to `base.py`.
- `{full,mamba,swa}_component.py` use `_component` as a role suffix, but all files in `unified_cache_components/` share that role and the directory name (plural "components") already states it — same pattern as `models/qwen.py` (not `qwen_model.py`), `entrypoints/engine.py` (not `engine_entrypoint.py`). The suffix doesn't differentiate, so drop it.

This rename touches #20415 territory — coordinate a merge window with @hzh0425 / @yizhang2077 / @pansicheng.

---

## 5. Design rationale

### 5.1 Layer responsibilities and orthogonal axes

```
                upper layers (scheduler / model_runner / attention backend)
                            │
                            ▼
                ┌────────────────────────────┐
                │  hybrid_cache/             │   multi-pool router
                └────────────────────────────┘
                            │ uses
                            ▼
        ┌──────────────────────────────────────────┐
        │  allocator/                              │   indexing: "give me N slots"
        └──────────────────────────────────────────┘
                            │ owns reference to ↓
                            ▼
        ┌──────────────────────────────────────────┐                    ┌──────────────────────────┐
        │  pool/  (device storage)                 │  ── hicache ───►   │  pool_host/  (CPU mirror)│
        │  KV / SSM state physical holder          │   offload/reload   │  host mirror of pool/    │
        └──────────────────────────────────────────┘                    └──────────────────────────┘
```

| Module | Cares about | In → Out |
|---|---|---|
| `allocator/` | which slots are free | `need_size` → `indices` |
| `pool/` | physical KV / SSM state layout | `(layer_id, indices) ↔ tensor` |
| `pool_host/` | host mirror + H2D/D2H | `device_indices ↔ host_indices` |
| `hybrid_cache/` | per-layer routing across pools | `layer_id → pool` |

**Two orthogonal axes** justify the split:

```
allocation strategy (allocator/)              storage format (pool/)
• unpaged / paged / SWA / HiSparse           • MHA / MLA / NSA / Mamba / SWA / DSv4 / HiSparse
```

Any combination is valid (Paged + MHA, Paged + MLA, HiSparse + DSv4). Merging would produce `MHA-paged`, `MLA-paged`, etc. — paged logic only concerns page tables, independent of K/V layout. **Orthogonal axes must live in separate modules.**

### 5.2 Naming convention

> **Role affixes (`_backend` / `_controller` / `_component` etc.) earn their place only by differentiating files within the same directory. When all files in a directory share the same role — and the directory name already says so — the affix is redundant.**

| Same-dir files have different roles → keep affix | All same-dir files share one role → drop affix |
|---|---|
| `attention/*_backend.py` (`attention_registry.py` is not a backend) | `models/qwen.py`, `bert.py` (all files are models — no `_model` suffix) |
| `managers/*_controller.py` / `*_manager.py` / `*_mixin.py` (varied roles) | `entrypoints/engine.py`, `http_server.py` (no `_entrypoint`) |
| `pool_host/group.py` (peer with `mha.py`, `tensor_allocator.py` etc.) | `layers/quantization/fp8.py`, `bitsandbytes.py` (no `_quantization`) |
| ↑ keep | **`unified_cache_components/full.py`, `mamba.py`** (drop `_component` — all files are Components) |

Other rules:

| Counter-example | Correct | Why |
|---|---|---|
| `hybrid_cache/hybrid_cache_controller.py` | `hybrid_cache/controller.py` | `hybrid_cache_` is a literal parent-directory-name repeat |
| `pool/dsv4/dsv4_single.py` | `pool/deepseek_v4.py` (single file) | repeated `dsv4_` prefix; abbreviated dir adds nothing |
| `pool_host/mla_host.py` | `pool_host/mla.py` | `_host` repeats parent dir name |

`pool/mla.py` and `pool_host/mla.py` share a filename intentionally. Disambiguation via: (1) fully-qualified import paths, (2) class names (`MLATokenToKVPool` vs. `MLATokenToKVPoolHost`), (3) the README index.

### 5.3 HiSparse as a "horizontal feature"

HiSparse is not an attention family — it's a sparsification strategy applied across families. Single rule:

> 1. If a class **is HiSparse** (regardless of which underlying family it adapts) → `*/hisparse.py`
> 2. If a class is **pure X** (no HiSparse) → `*/x.py`
> 3. **No file = no instances of that type** (don't create empty placeholders)

So `allocator/deepseek_v4.py` doesn't exist (no pure DSv4 allocator — DSv4 uses paged); `pool_host/hisparse.py` doesn't exist (HiSparse runs entirely on device today). Concentrating all HiSparse code in `*/hisparse.py` keeps algorithm changes in one place and keeps primary-family files HiSparse-free.

### 5.4 Hicache tiers (L1/L2/L3) — flat, not nested

```
                 orchestrator (HiCacheController, currently in managers/)
                  │              │              │
                  ▼              ▼              ▼
              pool/          pool_host/     storage/
              (L1 GPU)       (L2 host)      (L3 disk/remote)
```

L1/L2/L3 is a runtime data flow, not a code-organization hierarchy. The three modules are peers strung together by an orchestrator. sglang's prevailing style is flat at this level.

### 5.5 `__init__.py` policy

sglang relies on Python 3 implicit namespace packages — `__init__.py` is not mandatory. Add it only when the package needs aggregate re-exports.

| Package | `__init__.py` |
|---|---|
| `allocator/` / `pool/` / `pool_host/` | ✅ explicit `__all__` re-export |
| `hybrid_cache/` | ❌ no aggregate re-export needed |
| `unified_cache_components/` | ✅ already present |

`*/base.py` holds **abstract base classes only**. Concrete impls — even the simplest one — go in their own files (e.g., `TokenToKVPoolAllocator` → `allocator/unpaged.py`, not `allocator/base.py`).

### 5.6 Class location index (`README.md` content)

| type \ scope | full(MHA) | mla | nsa | mamba | swa | hybrid_linear | deepseek_v4 | hisparse |
|---|---|---|---|---|---|---|---|---|
| allocator (unpaged, page_size=1) | `allocator/unpaged.py` | — | — | — | — | — | — | — |
| allocator (paged) | `allocator/paged.py` | `allocator/paged.py` | `allocator/paged.py` | — | — | — | — | — |
| allocator (swa-style) | — | — | — | — | `allocator/swa.py` | — | — | — |
| allocator (hisparse) | — | — | — | — | — | — | `allocator/hisparse.py` | `allocator/hisparse.py` |
| pool (device) | `pool/mha.py` | `pool/mla.py` | `pool/nsa.py` | `pool/mamba.py` | `pool/swa.py` | `pool/hybrid_linear.py` | `pool/deepseek_v4.py` | `pool/hisparse.py` |
| pool (host) | `pool_host/mha.py` | `pool_host/mla.py` | `pool_host/nsa.py` | `pool_host/mamba.py` | — | — | `pool_host/deepseek_v4.py` | — |
| req-to-token | `pool/req_to_token.py` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ |
| cross-family abc | `pool/base.py` (`KVCache`, `BaseSWAKVPool`) · `allocator/base.py` (`BaseTokenToKVPoolAllocator`) · `pool_host/base.py` (`HostKVCache`) · `pool_host/tensor_allocator.py` (`HostTensorAllocator`) |

---

## 6. Trade-offs and rejected alternatives

| Decision | Choice | Rejected | Why rejected |
|---|---|---|---|
| Hardware-specific allocators/pools | stay in `hardware_backend/<hw>/` | move into `mem_cache/` | sglang convention is per-hardware subdirs; deps go plugin → framework |
| `pool/deepseek_v4` layout | **single file `pool/deepseek_v4.py`** (~780 lines, all 7 DSv4 classes) | (a) flat 2-file `dsv4.py` + `dsv4_compress_state.py` / (b) 2-file subpackage `pool/dsv4/{kv,compress_state}.py` / (c) 4-file subpackage | (a)/(b)/(c) all over-engineered — `CompressStatePool` (110 lines) has no external consumers, it's a private helper of `DeepSeekV4TokenToKVPool`; the original split was a historical artifact, not architectural necessity; single file matches the per-family pattern of `mha.py`/`mla.py`/etc; 780 lines is well within sglang norms (`memory_pool.py` was 2251; `models/qwen2_5_vl.py` is 1500+) |
| `TokenToKVPoolAllocator` placement | own file `allocator/unpaged.py` | `allocator/base.py` (with abc) / `default.py` / `contiguous.py` | not a trivial wrapper (state machines + optimizations); `base.py` stays abc-only; `unpaged.py` pairs cleanly with `paged.py` |
| `HostTensorAllocator` placement | `pool_host/tensor_allocator.py` | `pool_host/base.py` / `tensor_factory.py` | filename matches class name; `pool_host/` scope already disambiguates from device `allocator/` |
| `pool_host/` filenames same as `pool/` | `mla.py` / `mla.py` | `pool_host/mla_host.py` suffix | violates naming convention; reintroduces the style this RFC removes |
| L1/L2/L3 in directory layout | flat siblings | `tier/{device,host,storage}/` nesting | tier hierarchy is runtime data flow, not code organization |
| Top-level `pool/` vs `device_pool/` | `pool/` (default = device) | `device_pool/` + `host_pool/` + `storage_pool/` | matches sglang repo-wide "device is default"; `storage_pool/` would be a misnomer (storage holds backend plugins, not pools) |
| Filename `dsv4` vs `deepseek_v4` | **long `deepseek_v4`** | short `dsv4` | sglang's actual file-naming convention is long form (11 files: `models/deepseek_v4.py`, `configs/deepseek_v4.py`, `layers/deepseek_v4_rope.py`, `layers/attention/deepseek_v4_backend.py`, etc.); short form `dsv4` is reserved for **multi-file subpackage directories** (`layers/attention/dsv4/` has 8 files inside) — that pattern doesn't apply to a single file; `mha`/`mla`/`nsa`/`swa` are standard attention-literature abbreviations, but `dsv4` is a specific model nickname (different category) |
| `tree_component.py` rename | `base.py` | `tree.py` | file is actually abc + shared enums |
| `{full,mamba,swa}_component.py` rename | drop suffix → `full.py` / `mamba.py` / `swa.py` | keep `_component` suffix | all files in `unified_cache_components/` share the role and the plural directory name already states it; same precedent as `models/qwen.py` not `qwen_model.py`; suffix has no differentiating value |
| Compatibility | one-shot rewrite (move + update ~94 import sites in same PR) | shim files with deprecation warnings | sglang doesn't promise import stability; shims add maintenance burden and encourage lazy migration; `__module__` changes either way |

---

## 7. Drawbacks and impact

### 7.1 Compatibility

- **One-shot rewrite**: the PR moves classes AND updates all ~94 downstream import sites. No shim files.
- **External consumers** (NPU/MLX maintainers, third-party plugins) update via release notes — `sed`-able rewrites:
  ```bash
  sed -i 's|sglang.srt.mem_cache.memory_pool|sglang.srt.mem_cache.pool|g' <files>
  sed -i 's|sglang.srt.mem_cache.memory_pool_host|sglang.srt.mem_cache.pool_host|g' <files>
  ```
- **Class `__module__` changes** — pickled checkpoints holding these classes (unlikely but possible) won't deserialize.

### 7.2 Hardware-specific extensions stay put

`hardware_backend/npu/{allocator_npu.py, memory_pool_npu.py}` and `hardware_backend/mlx/kv_cache/*` remain in their current locations. They extend mem_cache base classes via `from sglang.srt.mem_cache.* import ...`; only their import paths need updating. Dependency direction is `hardware_backend → mem_cache`, never reverse.

### 7.3 Out of scope (audited)

| File / directory | Reason |
|---|---|
| `evict_policy.py` | tree-layer eviction strategies, belongs to #20415 |
| `events.py` | cross-layer mixin |
| `flush_cache.py` | argparse + requests CLI script, no classes |
| `multimodal_cache.py` / `hicache_storage.py` / `cache_init_params.py` | independent or belong elsewhere |
| `common.py` / `utils.py` | generic utilities |
| `cpp_radix_tree/` / `sparsity/` / `storage/` | independent subsystems |
| `*_radix_cache.py` / `base_prefix_cache.py` / `chunk_cache.py` | tree-layer, #20415 |
| `managers/cache_controller.py` / `managers/hisparse_coordinator.py` | cohesion issue but exceeds #24335 scope (suggest follow-up issue to migrate into `mem_cache/hicache/` and `mem_cache/sparsity/`) |

---

## 8. Unresolved questions

1. ~~DSv4 file size~~ — **resolved**: single file `pool/deepseek_v4.py` (~780 lines, all classes); long-form filename matches sglang's existing DeepSeek file naming.
2. **`unified_cache_components/` merge window** — file rename will conflict with in-flight #20415 PRs. Owner-coordinated window?
3. **`pool/hybrid_linear.py` filename** — class is `HybridLinearKVPool` (linear attention in hybrid models). Keep `hybrid_linear.py` (matches class prefix) or rename to `linear.py` (cleaner; `pool/` scope unambiguous)?
4. **NSA host vs DSv4 indexer naming asymmetry** — device side has `NSATokenToKVPool` + `DeepSeekV4IndexerPool`; host side has `NSAIndexerPoolHost`. RFC adds a FIXME docstring only. Acceptable?
5. **Pickle risk** — any checkpoint or distributed flow that pickles these classes will break.
