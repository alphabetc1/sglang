import time

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig


def _make_storage(tmp_path, monkeypatch, extra_config=None):
    monkeypatch.setenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", str(tmp_path))
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        is_page_first_layout=False,
        model_name="test/model",
        extra_config=extra_config,
    )
    return HiCacheFile(config, file_path=str(tmp_path))


def test_hicache_file_no_capacity_ttl(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path, monkeypatch, extra_config=None)
    value1 = torch.zeros((4,), dtype=torch.float32)
    value2 = torch.zeros((8,), dtype=torch.float32)

    assert storage.set("k1", value1) is True
    assert storage.set("k2", value2) is True
    assert storage.exists("k1") is True
    assert storage.exists("k2") is True


def test_hicache_file_capacity_lru_eviction(tmp_path, monkeypatch):
    value = torch.zeros((4,), dtype=torch.float32)
    size_bytes = value.numel() * value.element_size()
    capacity_gb = (size_bytes + 1) / 1e9

    storage = _make_storage(
        tmp_path, monkeypatch, extra_config={"capacity_gb": capacity_gb}
    )

    assert storage.set("k1", value) is True
    assert storage.set("k2", value) is True
    assert storage.exists("k1") is False
    assert storage.exists("k2") is True


def test_hicache_file_ttl_expiration(tmp_path, monkeypatch):
    storage = _make_storage(
        tmp_path, monkeypatch, extra_config={"ttl_seconds": 0.01}
    )
    value = torch.zeros((4,), dtype=torch.float32)

    assert storage.set("k1", value) is True
    time.sleep(0.02)
    assert storage.get("k1", torch.zeros_like(value)) is None
    assert storage.exists("k1") is False


def test_hicache_file_metrics(tmp_path, monkeypatch):
    value = torch.zeros((4,), dtype=torch.float32)
    size_bytes = value.numel() * value.element_size()
    capacity_gb = (size_bytes * 2) / 1e9

    storage = _make_storage(
        tmp_path,
        monkeypatch,
        extra_config={"capacity_gb": capacity_gb, "ttl_seconds": 0},
    )

    assert storage.set("k1", value) is True
    assert storage.get("k1", torch.zeros_like(value)) is not None
    assert storage.get("missing", torch.zeros_like(value)) is None

    metrics = storage.get_stats()
    assert metrics.capacity_gb == float(capacity_gb)
    assert metrics.current_usage_gb == float(size_bytes / 1e9)
    assert metrics.hit_count == 1
    assert metrics.miss_count == 1
    assert metrics.evict_count == 0
    assert metrics.expired_count == 0
    assert metrics.hit_ratio == 0.5

    metrics = storage.get_stats()
    assert metrics.hit_count == 0
    assert metrics.miss_count == 0
