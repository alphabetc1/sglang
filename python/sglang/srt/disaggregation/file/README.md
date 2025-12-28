# File-Based PD Disaggregation Transfer Backend

## Introduction

The `file` transfer backend is a simple file-based KV cache transfer mechanism for Prefill-Decode (PD) disaggregation in SGLang. It uses the local or shared filesystem to transfer KV cache data between prefill and decode instances.

### Features

- **No external dependencies**: Unlike `mooncake` (RDMA) or `nixl` (NIXL library), the file backend requires no additional hardware or software.
- **Easy debugging**: All transfer data is written to files, making it easy to inspect and debug.
- **Shared filesystem support**: Works with any shared filesystem (NFS, Lustre, etc.) accessible by both prefill and decode nodes.
- **Development & testing**: Ideal for local development, testing, and environments without RDMA support.

### When to Use

- **Development/Testing**: Quickly test PD disaggregation without setting up RDMA infrastructure.
- **Small-scale deployments**: When RDMA is not available but PD disaggregation is still desired.
- **Debugging**: Inspect the exact data being transferred between prefill and decode.
- **CI/CD pipelines**: Automated testing of PD disaggregation features.

### Limitations

- **Performance**: File I/O is slower than RDMA-based transfers (mooncake, nixl).
- **Latency**: Higher TTFT compared to memory-based transfer methods.
- **Storage overhead**: Requires disk space for temporary KV cache files.

## Usage

### Basic Usage

Start prefill and decode servers with the `file` transfer backend:

```bash
# Start prefill instance
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend file \
    --tp-size 1

# Start decode instance
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30001 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend file \
    --tp-size 1
```

### Configuration

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_DISAGG_FILE_BACKEND_DIR` | Directory for storing transfer files | `/tmp/sglang_pd_file_backend` |
| `SGLANG_DISAGG_FILE_BACKEND_TIMEOUT` | Timeout (seconds) for waiting transfer completion | `300` |

#### Using a Shared Filesystem

For multi-node deployments, configure a shared filesystem path:

```bash
# Set the same directory on all nodes
export SGLANG_DISAGG_FILE_BACKEND_DIR=/shared/nfs/sglang_pd_transfer

# Start prefill (Node 1)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend file \
    ...

# Start decode (Node 2)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend file \
    ...
```

### Architecture

The file backend consists of two main components:

1. **FileKVSender** (Prefill side):
   - Writes KV cache indices to binary files
   - Creates metadata files with transfer information
   - Writes completion marker when transfer is done

2. **FileKVReceiver** (Decode side):
   - Polls for completion marker files
   - Reads KV cache data from shared storage
   - Times out if transfer doesn't complete

### File Structure

```
/tmp/sglang_pd_file_backend/
├── room_12345/
│   ├── kv_chunk_0.bin       # KV indices binary data
│   ├── kv_chunk_0.bin.meta  # Chunk metadata (JSON)
│   ├── kv_chunk_1.bin
│   ├── kv_chunk_1.bin.meta
│   ├── metadata.bin         # Transfer metadata
│   └── transfer_done        # Completion marker
├── room_12346/
│   └── ...
```

## Testing

### Unit Test

Create a test file to verify the file backend works correctly:

```python
"""
Test file backend for PD disaggregation.
Usage:
    python test_file_backend.py
"""

import os
import tempfile
import numpy as np
import unittest

# Set up test directory
TEST_DIR = tempfile.mkdtemp()
os.environ["SGLANG_DISAGG_FILE_BACKEND_DIR"] = TEST_DIR

from sglang.srt.disaggregation.file import FileKVSender, FileKVReceiver
from sglang.srt.disaggregation.base.conn import KVPoll


class TestFileBackend(unittest.TestCase):

    def test_sender_receiver_basic(self):
        """Test basic sender/receiver functionality."""
        bootstrap_room = 12345

        # Create sender
        sender = FileKVSender(
            mgr=None,
            bootstrap_addr="localhost:8998",
            bootstrap_room=bootstrap_room,
            dest_tp_ranks=[0],
            pp_rank=0,
        )

        # Initialize sender
        sender.init(num_kv_indices=10, aux_index=0)

        # Check initial poll status
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        # Send data
        kv_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        sender.send(kv_indices)

        # Check success
        self.assertEqual(sender.poll(), KVPoll.Success)

        # Create receiver
        receiver = FileKVReceiver(
            mgr=None,
            bootstrap_addr="localhost:8998",
            bootstrap_room=bootstrap_room,
        )

        # Initialize receiver
        receiver.init(kv_indices=kv_indices, aux_index=0)

        # Poll should succeed (files exist)
        self.assertEqual(receiver.poll(), KVPoll.Success)

    def test_transfer_files_created(self):
        """Test that transfer files are created correctly."""
        bootstrap_room = 12346

        sender = FileKVSender(
            mgr=None,
            bootstrap_addr="localhost:8998",
            bootstrap_room=bootstrap_room,
            dest_tp_ranks=[0],
            pp_rank=0,
        )
        sender.init(num_kv_indices=5, aux_index=0)

        kv_indices = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        sender.send(kv_indices)

        # Check files exist
        room_dir = os.path.join(TEST_DIR, f"room_{bootstrap_room}")
        self.assertTrue(os.path.exists(room_dir))
        self.assertTrue(os.path.exists(os.path.join(room_dir, "kv_chunk_0.bin")))
        self.assertTrue(os.path.exists(os.path.join(room_dir, "transfer_done")))

        # Verify data
        loaded = np.fromfile(os.path.join(room_dir, "kv_chunk_0.bin"), dtype=np.int32)
        np.testing.assert_array_equal(loaded, kv_indices)


if __name__ == "__main__":
    unittest.main()
```

### Integration Test

Run a full PD disaggregation test with the file backend:

```bash
# Terminal 1: Start prefill server
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30000 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend file \
    --mem-fraction-static 0.5

# Terminal 2: Start decode server
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 30001 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend file \
    --mem-fraction-static 0.5

# Terminal 3: Send test request
curl http://localhost:30001/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello, how are you?",
        "sampling_params": {"max_new_tokens": 32},
        "bootstrap_host": "localhost:30000"
    }'
```

### Cleanup

After testing, clean up the transfer files:

```bash
rm -rf /tmp/sglang_pd_file_backend
```

## Comparison with Other Backends

| Backend | Speed | Dependencies | Best For |
|---------|-------|--------------|----------|
| `mooncake` | Fast (RDMA) | Mooncake library, RDMA NIC | Production, high-performance |
| `nixl` | Fast (RDMA) | NIXL library | Production, multiple storage backends |
| `ascend` | Fast | Ascend NPU | Huawei Ascend hardware |
| `fake` | Instant | None | Warmup, benchmarking |
| `file` | Slow (Disk I/O) | None | Development, testing, debugging |

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure the storage directory is writable:
   ```bash
   chmod 755 /tmp/sglang_pd_file_backend
   ```

2. **Timeout errors**: Increase the timeout if using slow storage:
   ```bash
   export SGLANG_DISAGG_FILE_BACKEND_TIMEOUT=600
   ```

3. **Shared filesystem not accessible**: Verify both nodes can access the same directory:
   ```bash
   # On both nodes
   ls -la $SGLANG_DISAGG_FILE_BACKEND_DIR
   ```

4. **Stale files from previous runs**: Clean up before restarting:
   ```bash
   rm -rf $SGLANG_DISAGG_FILE_BACKEND_DIR/*
   ```

## See Also

- [PD Disaggregation Overview](../advanced_features/disaggregation.md)
- [HiCache File Backend](../advanced_features/hicache_design.md)
- [Mooncake Transfer Backend](../advanced_features/mooncake.md)
