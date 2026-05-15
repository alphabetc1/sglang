"""KVCache abstract base + cross-family abc (BaseSWAKVPool) + shared helpers.

NOTE: Stage D is the public API restructure. Concrete class definitions still
live in the legacy files (memory_pool.py, base_swa_memory_pool.py) and are
re-exported here. A follow-up PR will move the bodies in-place.
"""

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    copy_all_layer_kv_cache_tiled,
    get_tensor_size_bytes,
    move_kv_cache_native,
)

__all__ = [
    "BaseSWAKVPool",
    "KVCache",
    "copy_all_layer_kv_cache_tiled",
    "get_tensor_size_bytes",
    "move_kv_cache_native",
]
