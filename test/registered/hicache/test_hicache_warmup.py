"""
E2E tests for HiCache Global Warmup functionality.

Tests that a new sglang instance can pre-populate its host-level KV cache
from a shared storage backend on cold start.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_warmup.py -v
"""

import json
import os
import random
import tempfile
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=300, suite="stage-b-test-large-2-gpu")


class TestHiCacheWarmupUnit(CustomTestCase):
    """Unit tests for WarmupEntry, record_warmup_metadata, and list_warmup_entries."""

    def test_warmup_entry_dataclass(self):
        from sglang.srt.mem_cache.hicache_storage import WarmupEntry

        entry = WarmupEntry(
            token_ids=[1, 2, 3],
            hash_chain=["abc", "def"],
            priority=1,
            num_tokens=3,
        )
        self.assertEqual(entry.token_ids, [1, 2, 3])
        self.assertEqual(entry.priority, 1)
        self.assertEqual(entry.num_tokens, 3)

    def test_hicache_file_manifest_write_and_read(self):
        """Test that record_warmup_metadata writes manifest and list_warmup_entries reads it."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            HiCacheStorageExtraInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            storage = HiCacheFile(config, file_path=tmpdir)

            # Write some warmup metadata
            extra1 = HiCacheStorageExtraInfo()
            extra1.extra_info = {
                "priority": 1,
                "token_ids": [10, 20, 30, 40],
            }
            storage.record_warmup_metadata(["h1", "h2"], extra1)

            extra2 = HiCacheStorageExtraInfo()
            extra2.extra_info = {
                "priority": 0,
                "token_ids": [50, 60],
            }
            storage.record_warmup_metadata(["h3"], extra2)

            # Read back
            entries = storage.list_warmup_entries()
            self.assertEqual(len(entries), 2)
            # Should be sorted by priority desc
            self.assertEqual(entries[0].priority, 1)
            self.assertEqual(entries[0].token_ids, [10, 20, 30, 40])
            self.assertEqual(entries[0].hash_chain, ["h1", "h2"])
            self.assertEqual(entries[1].priority, 0)

    def test_list_warmup_entries_deduplication(self):
        """Test that duplicate hash chains are deduplicated (last write wins)."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            HiCacheStorageExtraInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            storage = HiCacheFile(config, file_path=tmpdir)

            # Write same hash chain twice with different priority
            extra1 = HiCacheStorageExtraInfo()
            extra1.extra_info = {"priority": 0, "token_ids": [1, 2]}
            storage.record_warmup_metadata(["h1"], extra1)

            extra2 = HiCacheStorageExtraInfo()
            extra2.extra_info = {"priority": 1, "token_ids": [1, 2]}
            storage.record_warmup_metadata(["h1"], extra2)

            entries = storage.list_warmup_entries()
            self.assertEqual(len(entries), 1)
            # Last write wins
            self.assertEqual(entries[0].priority, 1)

    def test_list_warmup_entries_max_tokens(self):
        """Test that max_tokens budget is respected."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            HiCacheStorageExtraInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            storage = HiCacheFile(config, file_path=tmpdir)

            # Write two entries
            for i, (chain, toks) in enumerate(
                [
                    (["a1"], list(range(100))),
                    (["b1"], list(range(200))),
                ]
            ):
                extra = HiCacheStorageExtraInfo()
                extra.extra_info = {"priority": 0, "token_ids": toks}
                storage.record_warmup_metadata(chain, extra)

            # Budget of 150 tokens: should get only the first entry (100 tokens)
            # since the second (200) doesn't fit, it's skipped
            entries = storage.list_warmup_entries(max_tokens=150)
            total = sum(e.num_tokens for e in entries)
            self.assertLessEqual(total, 150)

    def test_list_warmup_entries_empty_manifest(self):
        """Test that missing manifest returns empty list."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            storage = HiCacheFile(config, file_path=tmpdir)
            entries = storage.list_warmup_entries()
            self.assertEqual(entries, [])

    def test_record_warmup_metadata_no_extra_info(self):
        """Test that record_warmup_metadata is a no-op when extra_info is None."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            HiCacheStorageExtraInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            storage = HiCacheFile(config, file_path=tmpdir)

            # No extra_info
            storage.record_warmup_metadata(["h1"], None)
            # extra_info without extra_info dict
            storage.record_warmup_metadata(["h1"], HiCacheStorageExtraInfo())
            # extra_info with empty dict (no token_ids)
            ei = HiCacheStorageExtraInfo()
            ei.extra_info = {}
            storage.record_warmup_metadata(["h1"], ei)

            entries = storage.list_warmup_entries()
            self.assertEqual(entries, [])

    def test_storage_operation_priority(self):
        """Test that StorageOperation carries priority field."""
        from sglang.srt.managers.cache_controller import StorageOperation
        from sglang.srt.mem_cache.radix_cache import RadixKey

        op = StorageOperation(
            host_indices=None,
            token_ids=[1, 2, 3],
            priority=1,
        )
        self.assertEqual(op.priority, 1)

        op_default = StorageOperation(
            host_indices=None,
            token_ids=[1, 2, 3],
        )
        self.assertEqual(op_default.priority, 0)

        op_extra_key = StorageOperation(
            host_indices=None,
            token_ids=RadixKey(token_ids=[1, 2, 3], extra_key="tenant-a"),
        )
        self.assertEqual(op_extra_key.extra_key, "tenant-a")

    def test_hicache_file_warmup_defaults_on_empty(self):
        """Test that warmup methods return safe defaults on a fresh HiCacheFile."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            s = HiCacheFile(config, file_path=tmpdir)
            # No-op when no extra_info
            s.record_warmup_metadata(["h1"], None)
            # Empty list on fresh storage
            self.assertEqual(s.list_warmup_entries(), [])
            self.assertEqual(s.list_warmup_entries(max_tokens=100), [])

    def test_list_warmup_entries_legacy_manifest_compat(self):
        """Test backward compatibility with manifest lines missing num_tokens."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            s = HiCacheFile(config, file_path=tmpdir)
            legacy_record = {
                "hash_chain": ["h1"],
                "token_ids": [1, 2, 3, 4],
                "priority": 1,
                "timestamp": time.time(),
            }
            with open(s._manifest_path, "w") as f:
                f.write(json.dumps(legacy_record) + "\n")

            entries = s.list_warmup_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].num_tokens, 4)

    def test_list_warmup_entries_skip_malformed_lines(self):
        """Test malformed manifest lines are ignored."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            s = HiCacheFile(config, file_path=tmpdir)
            valid_record = {
                "hash_chain": ["h1"],
                "token_ids": [10, 20],
                "num_tokens": 2,
                "priority": 0,
                "timestamp": time.time(),
            }
            with open(s._manifest_path, "w") as f:
                f.write("this-is-not-json\n")
                f.write(json.dumps(valid_record) + "\n")
                f.write(json.dumps({"hash_chain": [], "token_ids": [1]}) + "\n")

            entries = s.list_warmup_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].hash_chain, ["h1"])

    def test_list_warmup_entries_dedup_uses_full_hash_chain(self):
        """Entries with same tail hash but different full chains are distinct."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            s = HiCacheFile(config, file_path=tmpdir)
            records = [
                {
                    "hash_chain": ["prefix-a", "same-tail"],
                    "token_ids": [1, 2],
                    "num_tokens": 2,
                    "priority": 0,
                    "timestamp": time.time(),
                },
                {
                    "hash_chain": ["prefix-b", "same-tail"],
                    "token_ids": [3, 4],
                    "num_tokens": 2,
                    "priority": 0,
                    "timestamp": time.time() + 1,
                },
            ]
            with open(s._manifest_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            entries = s.list_warmup_entries()
            self.assertEqual(len(entries), 2)

    def test_list_warmup_entries_preserve_extra_key(self):
        """Warmup manifest preserves extra_key namespace for radix insertion."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            HiCacheStorageExtraInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            s = HiCacheFile(config, file_path=tmpdir)
            ei = HiCacheStorageExtraInfo()
            ei.extra_info = {
                "priority": 1,
                "token_ids": [1, 2, 3],
                "extra_key": "lora:adapter-a",
            }
            s.record_warmup_metadata(["h1"], ei)

            entries = s.list_warmup_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].extra_key, "lora:adapter-a")

    def test_list_warmup_entries_restore_bigram_tokens(self):
        """JSON-serialized bigram tokens should be restored as tuples."""
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            HiCacheStorageExtraInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                is_mla_model=False,
                is_page_first_layout=False,
                model_name="test-model",
            )
            s = HiCacheFile(config, file_path=tmpdir)
            ei = HiCacheStorageExtraInfo()
            ei.extra_info = {
                "priority": 0,
                "token_ids": [(10, 11), (20, 21)],
            }
            s.record_warmup_metadata(["h1"], ei)

            entries = s.list_warmup_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].token_ids, [(10, 11), (20, 21)])


class TestHiCacheWarmupE2E(CustomTestCase):
    """E2E test: write KV to storage, restart with warmup, verify cache hit."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = get_tokenizer(cls.model)

        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _get_server_args(self, warmup_ratio=0.0):
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
            "warmup_ratio": warmup_ratio,
        }
        return [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            "64",
            "--enable-cache-report",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-backend-extra-config",
            json.dumps(extra_config),
        ]

    def _launch_server(self, warmup_ratio=0.0):
        env_vars = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": self.temp_dir,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._get_server_args(warmup_ratio),
            env=env_vars,
        )
        wait_for_http_ready(
            url=f"{self.base_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=process,
        )
        return process

    def _send_request(self, prompt, max_tokens=100):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _flush_cache(self):
        r = requests.post(f"{self.base_url}/flush_cache", timeout=10)
        return r.status_code == 200

    def _gen_prompt(self, token_num):
        all_tokens = list(self.tokenizer.get_vocab().values())
        selected = random.choices(all_tokens, k=token_num)
        return self.tokenizer.decode(selected)

    def test_warmup_preloads_cache(self):
        """Test that warmup preloads KV cache from storage on cold start.

        Phase 1: Launch server, run requests to populate storage, shut down.
        Phase 2: Launch server with warmup_ratio>0, verify manifest exists
                 and cache is warmed.
        """
        # Phase 1: Populate storage
        print("\n=== Phase 1: Populate storage backend ===")
        process1 = self._launch_server(warmup_ratio=0.0)
        try:
            base_prompt = self._gen_prompt(768)

            # Run request to populate cache
            resp1 = self._send_request(base_prompt, max_tokens=50)
            self.assertIsNotNone(resp1)

            # Trigger offloading to storage
            self._send_request(self._gen_prompt(1), max_tokens=150)
            time.sleep(3)
            self._flush_cache()
            time.sleep(1)

            # Verify storage has data: re-request should get cache hit
            resp2 = self._send_request(base_prompt, max_tokens=50)
            cached = resp2.get("meta_info", {}).get("cached_tokens", 0)
            print(f"Phase 1 verification: cached_tokens={cached}")
            self.assertGreater(cached, 200, "Storage should have cached data")
        finally:
            kill_process_tree(process1.pid)

        # Verify manifest file was created
        manifest_files = [
            f for f in os.listdir(self.temp_dir) if "warmup_manifest" in f
        ]
        print(f"Manifest files: {manifest_files}")
        self.assertGreater(len(manifest_files), 0, "Warmup manifest should be created")

        # Phase 2: Restart with warmup
        print("\n=== Phase 2: Restart with warmup_ratio=0.5 ===")
        time.sleep(2)  # Ensure port is released
        process2 = self._launch_server(warmup_ratio=0.5)
        try:
            # Give warmup a moment to complete
            time.sleep(3)

            # Request with same prompt - should benefit from warmed-up cache
            resp3 = self._send_request(base_prompt, max_tokens=50)
            cached_after_warmup = resp3.get("meta_info", {}).get("cached_tokens", 0)
            print(f"Phase 2: cached_tokens after warmup={cached_after_warmup}")

            # After warmup, the cache should have some hits
            # (at least from host-level cache, even without a fresh storage prefetch)
            self.assertGreater(
                cached_after_warmup,
                100,
                "Warmup should preload some tokens into cache",
            )
        finally:
            kill_process_tree(process2.pid)


if __name__ == "__main__":
    unittest.main()
