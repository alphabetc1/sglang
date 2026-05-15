"""HostTensorAllocator + host tensor allocation helpers.

This file holds the host-side tensor factory. The class name is
'HostTensorAllocator' but its interface (allocate(dims, dtype) -> Tensor)
is a tensor factory, not a slot allocator like the device-side
BaseTokenToKVPoolAllocator (alloc(n) -> indices).
"""

from sglang.srt.mem_cache.memory_pool_host import (
    HostTensorAllocator,
    alloc_with_host_register,
    alloc_with_pin_memory,
    get_allocator_from_storage,
)

__all__ = [
    "HostTensorAllocator",
    "alloc_with_host_register",
    "alloc_with_pin_memory",
    "get_allocator_from_storage",
]
