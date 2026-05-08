# DeepSeek-V4 HiCache (L2) — Implementation Strategy

End-to-end summary of how DSV4 hicache (device ↔ host KV cache) is implemented
in this branch. Reflects the design **as actually shipped** — significantly
different in places from the original spec because several invariants only
surfaced during real-model integration.

This is the **v2** design. The v1 of this branch had c128 and SWA layers as
"device-only / recompute on hit", with a `SWA_REPLAY` workaround mini-batch
to refill SWA after a host hit. v2 backs up **all four** pools (c4, c4_indexer,
c128, SWA) and removes the entire replay machinery: a host hit is now a pure
H→D copy with no model re-execution required.

---

## 1. Goal & scope

Add a host-side L2 KV cache to DeepSeek-V4 so that prefixes evicted from the
GPU pool (LRU) can be recovered without recomputing them — paying only the
H→D copy.

**In scope:** TP single-node, EAGLE / DSV4 NextN.
**Out of scope (asserted at startup):** DP attention, PP > 1, HiSparse, L3
storage. Each combination raises a clear `ValueError` with a remediation
hint instead of silently degrading.

---

## 2. Architecture

### 2.1 Tree component layout

DSV4 routes through `UnifiedRadixCache` automatically when
`--enable-hierarchical-cache` is on. Each `UnifiedTreeNode` carries three
`ComponentData` slots:

| Component | device | host (L2) | counted in node aliveness |
|---|---|---|---|
| `FULL` (BASE) | logical token-ids — also serves as the c4 anchor's index list | **owns `host_value`** (anchor c4 host pool, side-shared by c4_indexer + c128) | yes |
| `SWA` | per-token SWA KV (independent slot space via `swa_attn_allocator`) | **owns its own `host_value`** on `SWAHostPool` (independent indices) | yes |
| `DSV4_COMPRESSED` (new) | shadow of FULL.value (same page-ids) | not used directly | yes — defers to FULL.host_value |

`DSV4_COMPRESSED` is intentionally a **passive shadow** of `FULL`. It
exists for two reasons:

1. So that the unified tree's per-component bookkeeping (LRU position,
   evictable counters, lock_ref) records DSV4 separately from generic FULL
   handling — without leaking DSV4-specific freeing logic into FULL.
2. To keep its `cd.value` mirrored from FULL on insert, so `match_prefix`
   has a uniform per-component validator pass.

The component is **not** the host-side authority. Its `evict_component` is
bookkeeping-only — clears `cd.value` and decrements the evictable counter,
but does **not** call any allocator/free. FULL is the single owner of the
free path on the shared `full_attn_allocator` namespace.

### 2.2 Host pool group

Four host pools live in a single `HostPoolGroup`:

| Pool | host class | role | how device indices flow |
|---|---|---|---|
| `C4HostPool` | extends `MLATokenToKVPoolHost` | **anchor** | n/a (it's the anchor) |
| `C4IndexerHostPool` | extends `NSAIndexerPoolHost` | side pool, shares anchor's indices | `share_indices_with_anchor=True` |
| `C128HostPool` | extends `MLATokenToKVPoolHost` | side pool, shares anchor's indices | `share_indices_with_anchor=True` (sized to anchor) |
| `SWAHostPool` | extends `MLATokenToKVPoolHost` | independent — uses swa_attn_allocator | `device_alloc_fn / device_free_fn` of `swa_attn_allocator` |

c4_indexer and c128 are registered via the existing
`register_hicache_anchor_kv_shared_indices_pool` mechanism — same path NSA
uses for its single side pool. SWA is the standard "independent indices"
case (the existing SWA hybrid pool flow).

**c128 sized to the anchor.** Sharing indices with c4 means c128 must use
the same host index space. c4 allocates one host slot per raw token; c128
only needs one slot per 128 raw tokens, so 127 / 128 slots per host page
go unused. In absolute terms this is small: DSV4-Flash has 2 c128 layers
out of 23, so the redundancy adds only a few hundred MB of pinned host
memory at typical hicache_ratio. The alternative — extending the
share-indices controller to support compressed-position translation per
side pool — was rejected as a much larger architectural change for v1.

**SWA on its own indices.** SWA's slot space comes from
`swa_attn_allocator` (a sub-allocator of the SWA token pool). On backup
the SWAComponent emits a `PoolTransfer(name=PoolName.SWA, device_indices=
cd.value)` — the host pool's `device_alloc_fn` is wired to
`swa_attn_allocator.alloc`, so the controller allocates fresh host slots
independently and stores their indices on `cd.host_value`.

### 2.3 Why `MLATokenToKVPoolHost` + `NSAIndexerPoolHost` (and not from
scratch)

The base `HostKVCache` ABC requires implementing
`get_size_per_token / init_kv_buffer / load_to_device_per_layer /
backup_from_device_all_layer / get_data_page / get_dummy_flat_data_page`,
plus layout-specific transfer kernels. Reusing the existing MLA / NSA
subclasses inherits the layout (`layer_first`), pinned-host allocation,
and the controller integration; we only override what diverges (see §4).

---

## 3. Index translation — the load-bearing detail

Three different "page" / "slot" units coexist in DSV4 and must not be
confused:

1. **Allocator slot index** — what `full_attn_allocator.alloc(N)` returns.
   Indexes a *raw-token* slot in `[0, full_max_total_num_tokens)`. The
   radix tree stores these in `cd.value`.
2. **Device c4 buffer index** — `kv_buffer[layer_id][page_id, offset]`
   where `page_id` runs over compressed-position pages (size
   `c4_page_size = 64`) and `offset` runs within a c4 page.
3. **Host c4 slot index** — output of `c4_host_pool.alloc(N)`; addresses
   the host buffer's `(layer, host_size, 1, kv_cache_dim)` rows.

When the radix tree hands the allocator's raw-token index to the host /
device transfer code, the translation is:

```python
# host_page_size = radix tree's page_size in raw tokens (e.g. 256)
# dev_page_size  = c4 device pool's page_size in compressed positions (e.g. 64)
# compress_ratio = host_page_size // dev_page_size       (== 4 for c4)
page_id = raw_index //  host_page_size
offset  = (raw_index %  host_page_size) // compress_ratio
```

Same formula works for c128 (`compress_ratio=128, dev_page_size=2`) and
for the c4_indexer (same ratio as c4). One radix-tree page maps 1:1 to
one device page; what changes per pool is how many compressed entries the
device page holds.

The **early bug** of using `raw_index // dev_page_size` directly (i.e.
treating raw indices as already-compressed) caused CUDA out-of-bounds at
4× the buffer size. The current code centralises the translation in
`_dsv4_split_indices()` so all three compressed pools use the same logic.

SWA does *not* need this translation: its device buffer is per-token
(`compress_ratio == 1`), and its device indices are already in the
swa_attn allocator's slot space. The SWA host pool transfers run through
the standard `_dsv4_load_per_layer` / `_dsv4_backup_all_layer` helpers
with `dev_page_size == host_page_size`.

---

## 4. Page-aware D ↔ H transfer kernels

`MLATokenToKVPoolHost`'s default transfer code assumes a flat
`(size, kv_cache_dim)` device layout. DSV4's `DeepSeekV4SingleKVPool`
stores `(num_pages, bytes_per_page_padded)` uint8 with per-page padding.
We override `load_to_device_per_layer` and `backup_from_device_all_layer`
on `C4HostPool`, `C128HostPool`, and `SWAHostPool` (and analogously the
indexer):

```python
dev_view = layer_buffer.as_strided(
    size  =(num_pages, dev_page_size, bytes_per_token),
    stride=(bytes_per_page_padded, bytes_per_token, 1),
)
page_ids, offsets = _dsv4_split_indices(device_indices, host_page_size, compress_ratio)
chunks = dev_view[page_ids, offsets]            # (N, bytes_per_token) on CUDA
host_pool.kv_buffer[layer_id, host_idx_cpu, 0, :] = chunks.cpu()
```

Two subtleties that took the longest to find:

- **CPU/CUDA index split.** `cache_controller.move_indices` puts
  `host_indices` on CUDA when the kernel io_backend is selected. The host
  buffer is pinned-CPU; indexing a CPU tensor with CUDA indices triggers
  a silent cuda-launch assert later. Always `host_indices.to('cpu')`
  before indexing the host buffer.
- **`as_strided` for byte-level slicing.** The buffer has padding per
  page. `as_strided` with `stride=(bytes_per_page_padded, …)` skips that
  padding cleanly, so advanced indexing works without copies. Direct
  arithmetic on a `view(-1)` was tried first and didn't vectorise.

The c4_indexer host pool's `_indexer_host_view / _indexer_dev_view` use the
same trick over the indexer's k-with-scale buffer (per-token width 132 B
on Flash).

---

## 5. The eight integration bugs surfaced during e2e (all fixed)

These are upstream / integration bugs, not just typos in our own code.

1. **`unified_radix_cache.match_prefix` crashes on short bigram keys.**
   The warmup probe sends a < page_size prompt; with EAGLE
   `is_bigram=True`, after `page_aligned()` the key is empty and
   `child_key()` does `t[j+1] for j in range(page_size)` → IndexError. Fix:
   guard `len(key) == 0` post-`page_aligned()`.

2. **Raw-token indices fed to compressed-position buffer.** Discussed
   in §3. The CUDA out-of-bounds was deterministic on every backup once
   the first request landed.

3. **Host index on CUDA, host buffer on CPU.** Discussed in §4.

4. **Double-free between FULL and DSV4_COMPRESSED.** Both components
   originally called `full_attn_allocator.free` on the same indices,
   corrupting the allocator's `available + evictable + protected ==
   total` invariant (caught by `scheduler_runtime_checker_mixin`'s pool
   leak check). Fix: DSV4_COMPRESSED.evict_component is bookkeeping-only;
   FULL is the single free authority on the shared namespace.

5. **DSV4_COMPRESSED match_validator returned False after eviction.**
   With FULL holding the host backup on the c4 anchor, DSV4_COMPRESSED
   has no host_value of its own. The validator now defers to
   `BASE_COMPONENT_TYPE.host_value` so a node that's host-only (still
   has FULL.host_value but lost cd.value) is correctly marked alive.

6. **(v1 only) `SWAComponent.build_hicache_transfers` emitted a transfer
   when there was no SWA host pool.** Fixed in v2 by giving SWA its own
   host pool (`SWAHostPool` registered as an independent side pool). The
   `_swa_kv_pool_host is None` guards from v1 are gone.

7. **(v1 only) `SWAComponent.create_match_validator` rejected host-only
   nodes when SWA was intentionally device-only.** Same root cause as
   above; with SWA backed up to host the standard validator
   (`cd.value is None && cd.host_value is None` → fail) is correct again.

8. **(v1 only) `_swa_replay_pending` deadlock.** v1 parked the request
   awaiting an H→D ack, then returned `None` *before*
   `tree_cache.ready_to_load_host_cache()` was called (which is what
   actually invokes `cache_controller.start_loading` and kicks off the
   transfer). With no transfer to ack, `loading_check` had nothing to
   drain and the request hung. v2 deletes the entire park-and-replay
   path: a host hit follows the standard load_back flow, no special
   readiness gate is needed.

---

## 6. Cache hit flow (load_back, no replay)

When a request matches a host-only segment of the radix tree, the flow is
the **standard** unified-radix host-hit reload — same code path as a
non-DSV4 hybrid SWA model on host hit:

```
init_load_back(req, last_host_node)             # in PrefillAdder.add_one_req
  └─ tree_cache.load_back(node)                 # builds H→D PoolTransfers:
                                                #   - FULL on c4 anchor
                                                #   - c4_indexer (anchor-shared)
                                                #   - c128       (anchor-shared)
                                                #   - SWA        (independent indices)
     └─ cache_controller.load(...)              # queues op in load_queue;
        ongoing_load_back[node.id] = (node, lock_params)

# req.prefix_indices now covers the matched prefix (device slots claimed pre-transfer)
# extend_input_len shrinks to just the new (uncached) tail tokens

# scheduler.get_new_batch_prefill loop, same tick:
new_batch.hicache_consumer_index = tree_cache.ready_to_load_host_cache()
new_batch.prepare_for_extend()                  # standard EXTEND path

# forward pass:
#   – attention reads prefix slots (filled by H→D copy by the time fwd runs)
#   – attention writes new-token slots into c4 / c4_indexer / c128 / SWA
#     (no replay; the prefix slots already hold valid KV from host)
```

`forward_mode == SWA_REPLAY` and the `_swa_replay_pending` queue **do not
exist** in v2. The `compressor.py` `forward_core_compressor` /
`forward_indexer_compressor` writes are unconditional — there is no
`is_swa_replay()` gate.

---

## 7. Wiring summary (where the changes land)

```
python/sglang/srt/
├── managers/
│   └── scheduler.py              [1] DSV4-routing branch (no
│                                     SGLANG_ENABLE_UNIFIED_RADIX_TREE flag)
│                                 [2] _assert_dsv4_hicache_v1_supported
│                                     (HiSparse / DP / PP / L3)
├── mem_cache/
│   ├── dsv4_host_pool.py         (new) C4HostPool, C128HostPool,
│   │                             C4IndexerHostPool, SWAHostPool with the
│   │                             page-aware load/backup overrides
│   ├── deepseek_v4_memory_pool.py register_compressed_free_alloc /
│   │                             free_compressed_pages /
│   │                             alloc_compressed_pages
│   ├── hicache_storage.py        PoolName.{C4, C4_INDEXER, C128, SWA}
│   ├── hybrid_cache/hybrid_pool_assembler.py
│   │                             dsv4_stack branch in
│   │                             attach_hybrid_pool_to_unified_cache;
│   │                             build_dsv4_compressed_stack (4 pools)
│   ├── unified_radix_cache.py    register DSV4_COMPRESSED in
│   │                             COMPONENT_REGISTRY; page_aligned + empty
│   │                             guard fix
│   └── unified_cache_components/
│       ├── tree_component.py     ComponentType.DSV4_COMPRESSED
│       ├── dsv4_compressed_component.py  (new) passive-shadow component
│       │                                  (defers host-aliveness to BASE)
│       └── swa_component.py      (unchanged from upstream — works as-is
│                                  once SWAHostPool is wired)
└── layers/attention/dsv4/
    └── compressor.py             (no special hicache gates)
```

Plus tests under `test/srt/mem_cache/` and a manual reproducer at
`python/sglang/srt/mem_cache/bench_dsv4_hicache.py`.

---

## 8. Configuration surface

No new server args. Reused:

- `--enable-hierarchical-cache`  (must be set; otherwise normal radix path)
- `--hicache-ratio N`            (host = N × device; v1 tested with 4 and 16)
- `--hicache-write-policy write_through` (default; v1 doesn't exercise write_back)
- `--hicache-mem-layout layer_first`     (only layout supported)
- `--hicache-io-backend kernel`          (default; works through our overrides)

Asserts at startup raise `ValueError` if any of:
`--enable-hisparse`, `--enable-dp-attention`, `pp_size > 1`,
`--hicache-storage-backend != none` is combined with DSV4+hicache.

---

## 9. Known limitations / follow-ups

1. **c128 host slots are 127/128 redundant.** Sharing the c4 anchor's
   index space wastes most of the per-page slots since c128 only needs
   one entry per 128 raw tokens. Real fix needs per-pool host index
   spaces with a compressed-position translation in
   `_resolve_pool_transfers_allocation`. Non-trivial and not justified
   for DSV4-Flash's 2 c128 layers.
2. **4× c4 host over-allocation.** The host pool allocates 1 slot per
   raw token, but each c4 KV entry serves 4 raw tokens, so 75% of host
   slots are redundant. Same fix as (1).
3. **Load-back lock window.** `loading_check` releases the device
   `lock_ref` on ack; under high pressure with concurrent eviction in
   the same tick the reloaded slots can race with eviction. Not observed
   in eviction tests, but the window exists.
4. **`host_lock_ref` not bumped during H→D transfer.** Concurrent host
   pool eviction can free the in-flight backup. NSA has the same issue
   today; tracked for a unified fix.
5. **`_maybe_register_hicache_draft` warning.** On DSV4+EAGLE startup the
   draft worker logs `Draft pool type DeepSeekV4TokenToKVPool not
   supported for HiCache, skipping` — this is the *intended* behaviour
   (draft stays device-only per spec) but the message reads as an error.
   Cosmetic.
6. **DP attention / PP > 1 / HiSparse coexistence.** Each is a substantial
   piece of work; v1 asserts each out at startup with a clear message.
