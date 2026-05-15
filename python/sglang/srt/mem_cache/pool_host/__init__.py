"""Host-side mirrors for supported device pools.

NOTE: Stage E introduces this public namespace; class definitions still
live in memory_pool_host.py and are re-exported here. A follow-up PR
will move the bodies in-place.
"""

from sglang.srt.mem_cache.pool_host.base import HostKVCache, synchronized
from sglang.srt.mem_cache.pool_host.deepseek_v4 import DeepSeekV4SingleKVPoolHost
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.nsa import NSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.tensor_allocator import (
    HostTensorAllocator,
    alloc_with_host_register,
    alloc_with_pin_memory,
    get_allocator_from_storage,
)

__all__ = [
    "DeepSeekV4SingleKVPoolHost",
    "HostKVCache",
    "HostPoolGroup",
    "HostTensorAllocator",
    "MambaPoolHost",
    "MHATokenToKVPoolHost",
    "MLATokenToKVPoolHost",
    "NSAIndexerPoolHost",
    "PoolEntry",
    "alloc_with_host_register",
    "alloc_with_pin_memory",
    "get_allocator_from_storage",
    "synchronized",
]
