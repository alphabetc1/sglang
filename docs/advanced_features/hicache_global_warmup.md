# HiCache Global Warmup

Pre-populates host KV cache from storage on cold start, turning L3 misses into L2 hits.

## Usage

Warmup is **disabled by default** (`warmup_ratio=0.0`) even when a storage backend is configured. It runs in the background only after explicitly setting a positive ratio.

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend file
```

To enable (example 50% host budget): `--hicache-storage-backend-extra-config '{"warmup_ratio": 0.5}'`

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `warmup_ratio` | `0.0` | Host HiCache warmup budget ratio in `[0, 1)`. Warmup loads up to `warmup_ratio * host_hicache_capacity`. `0` disables. |
| `warmup_manifest_scan_bytes` | `67108864` | File backend only. Initial warmup scan reads only the tail window (bytes) of the manifest to avoid full-file cold-start scan. Set to `0` to force full scan. |

Warmup also stops automatically after **300 seconds**, when the host warmup budget is exhausted, or when storage is detached.

## How It Works

1. **Write path**: Backup operations persist warmup essentials: token IDs and priority.
2. **Warmup path**: On startup, a background thread reads warmup entries from storage, loads KV pages into host memory, and queues them for radix tree insertion on the main thread.
3. **Selection policy**: Storage backend decides selection policy. The built-in file backend uses a simple default: higher priority, then recency, and skips short prefixes.
4. **Load path**: Warmup only populates host HiCache. Device KV is still populated through the normal host-to-device load-back path on demand.
5. **Stop conditions**: host budget exhausted, 300s timeout, explicit stop, or storage detach.

## Semantics

- Warmup only exists when storage is attached. It is not an independent cache layer.
- Only one warmup session can be active at a time.
- Warmup restores cache identity using the same radix-key path as request-time insertion.
- TODO: extend metadata envelope for namespace fields (for example `extra_key`) once storage-key semantics are finalized.

## Storage Backend Support

| Backend | Supported | Notes |
|---|---|---|
| `file` | Yes | JSONL manifest, works out of the box |
| Custom | Override `record_storage_metadata` (or `record_warmup_metadata`), `list_warmup_entries`, and `supports_warmup_manifest` |

Backends without warmup support silently skip it (`supports_warmup_manifest=False`).
