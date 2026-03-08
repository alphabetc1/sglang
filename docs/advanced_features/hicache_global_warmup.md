# HiCache Global Warmup

Pre-populates host KV cache from storage on cold start, turning L3 misses into L2 hits.

## Usage

Warmup is **enabled by default** (`warmup_ratio=0.8`) when a storage backend is configured. It runs in the background — the server accepts requests immediately.

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend file
```

To disable: `--hicache-storage-backend-extra-config '{"warmup_ratio": 0.0}'`

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `warmup_ratio` | `0.8` | GPU KV usage threshold in `[0, 1)`. Warmup continues while GPU usage < ratio. `0` disables. |

Warmup also stops automatically after **300 seconds** or when host memory budget (`warmup_ratio × host_pool_size`) is exhausted.

## How It Works

1. **Write path**: Backup operations record token metadata (priority, token IDs) to a manifest file.
2. **Warmup path**: On startup, a background thread reads the manifest, loads KV pages from storage into host memory, and queues them for radix tree insertion on the main thread.
3. **Stop conditions**: GPU KV usage >= `warmup_ratio`, host budget exhausted, 300s timeout, or storage detach.

## Storage Backend Support

| Backend | Supported | Notes |
|---|---|---|
| `file` | Yes | JSONL manifest, works out of the box |
| Custom | Override `record_warmup_metadata` and `list_warmup_entries` |

Backends without warmup support silently skip it (default no-op).
