# HiCache Global Warmup: Cross-Instance KV Cache Preloading

## Overview

When a new SGLang instance starts (cold start or rolling update), its KV cache is empty. All requests must compute KV from scratch, causing a "cache stampede" until the cache is repopulated. **Global Warmup** enables new instances to pre-populate their host-level KV cache from a shared storage backend, loading entries by priority (pinned prefixes first, then most recently used).

## Quick Start

### Cold Start Warmup

Add `warmup_ratio` to your storage backend extra config:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --page-size 64 \
    --hicache-storage-backend file \
    --hicache-storage-backend-extra-config '{"warmup_ratio": 0.5}'
```

This will pre-populate up to 50% of the host memory pool from storage on startup. The server starts accepting requests immediately (warmup runs in the background).

### Runtime Attach with Warmup

When using runtime attach, include `warmup_ratio` in the extra config JSON:

```bash
curl -X PUT http://localhost:30000/hicache/storage-backend \
    -H "Authorization: Bearer <admin-key>" \
    -H "Content-Type: application/json" \
    -d '{
        "hicache_storage_backend": "file",
        "hicache_storage_backend_extra_config_json": "{\"warmup_ratio\": 0.3}"
    }'
```

By default, `warmup_ratio` is `0.0` (disabled), so runtime attach does not trigger warmup unless explicitly requested.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_ratio` | float | 0.0 | Max ratio of host memory pool to fill during warmup. Must be in `[0, 1)`. `0` disables warmup. |

The parameter is set via `--hicache-storage-backend-extra-config` (JSON string or `@file.json`).

## How It Works

### Write Path (Metadata Recording)

When KV cache pages are written to storage (backup), the system records metadata alongside the data:
- **Priority**: `1` for pinned prefixes, `0` for normal entries
- **Token IDs**: The token sequence corresponding to each batch of pages

This metadata is used by the warmup mechanism to enumerate and prioritize entries for loading.

### Warmup Path (Cold Start Loading)

1. **Enumerate entries**: Query storage for available entries, sorted by priority (desc) then recency (desc).
2. **Allocate host memory**: Reserve space from the host memory pool, respecting the `warmup_ratio` budget.
3. **Verify and load**: Check that pages exist in storage (`batch_exists`), then load KV data into host memory.
4. **Insert into radix tree**: On the main thread, insert loaded entries into the radix tree as host-level nodes.
5. **Serve requests**: When a request matches a warmed-up entry, the KV data is loaded from host to GPU (L2 hit) instead of recomputing from scratch.

### Safety Guarantees

- **No OOM**: Warmup respects the `warmup_ratio` budget. If host memory allocation fails, warmup stops immediately.
- **No IMA**: Host indices are properly tracked in radix tree nodes via `_insert_helper_host`. Matched prefix indices are freed to prevent leaks.
- **Thread safety**: The warmup thread only writes to a Queue. Radix tree mutations happen exclusively on the main scheduler thread.
- **Non-blocking**: The server starts accepting requests immediately. Warmup runs in a background thread and results are consumed in the scheduler event loop.
- **Graceful stop**: If storage is detached while warmup is in progress, the warmup thread detects the stop event and exits cleanly.

## Storage Backend Support

| Backend | Warmup Support |
|---------|---------------|
| `file` | Yes (JSONL manifest) |
| `mooncake` | Requires implementation of `record_warmup_metadata` and `list_warmup_entries` |
| `hf3fs` | Requires implementation |
| `nixl` | Requires implementation |
| Custom | Implement `record_warmup_metadata` and `list_warmup_entries` on your `HiCacheStorage` subclass |

For backends that don't implement warmup methods, the default behavior is a no-op (warmup is silently skipped).

## Backward Compatibility

- All changes are opt-in: `warmup_ratio=0.0` (default) means no warmup.
- Existing storage backends are unaffected: `record_warmup_metadata` defaults to no-op, `list_warmup_entries` defaults to `[]`.
- No changes to existing `batch_set_v1`, `batch_get_v1`, or `batch_exists` signatures.
