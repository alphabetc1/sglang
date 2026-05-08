def test_dsv4_host_pool_imports():
    from sglang.srt.mem_cache.dsv4_host_pool import (
        C4HostPool,
        C4IndexerHostPool,
        C128HostPool,
    )
    from sglang.srt.mem_cache.memory_pool_host import (
        HostKVCache,
        MLATokenToKVPoolHost,
        NSAIndexerPoolHost,
    )

    assert issubclass(C4HostPool, MLATokenToKVPoolHost)
    assert issubclass(C128HostPool, MLATokenToKVPoolHost)
    assert issubclass(C4IndexerHostPool, NSAIndexerPoolHost)
    assert issubclass(C4HostPool, HostKVCache)
    assert issubclass(C4IndexerHostPool, HostKVCache)
