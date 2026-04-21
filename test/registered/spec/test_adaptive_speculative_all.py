"""E2E tests for adaptive speculative decoding across all supported algorithms.

Tests:
  1. EAGLE (v1) — already covered in test_adaptive_speculative.py, quick sanity here
  2. NGRAM — target-only (no draft model), adaptive adjusts draft_token_num
  3. Standalone (v1) — separate target + draft model, inherits EAGLE adaptive
  4. Standalone V2 — overlap scheduling variant

Each test verifies:
  - Server launches successfully with --speculative-adaptive
  - Generation produces correct output
  - Adaptive controller shifts steps up/down based on acceptance patterns
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_NGRAM,
    DEFAULT_TARGET_MODEL_STANDALONE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Aggressive adaptive config for fast switching in tests
ADAPTIVE_CONFIG = {
    "candidate_steps": [1, 3],
    "ema_alpha": 1.0,
    "warmup_batches": 1,
    "update_interval": 1,
    "up_hysteresis": 0.0,
    "down_hysteresis": -0.25,
}

HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)

LOW_ACCEPT_PROMPT = (
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. "
    "Make it emotionally resonant and at least 100 words."
)

MAX_UPSHIFT_ATTEMPTS = 5
MAX_DOWNSHIFT_ATTEMPTS = 8


class AdaptiveSpecTestBase(CustomTestCase):
    """Base class for adaptive speculative decoding tests."""

    model: str
    base_url = DEFAULT_URL_FOR_TEST
    process = None
    adaptive_config_path = None

    @classmethod
    def _write_adaptive_config(cls):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(ADAPTIVE_CONFIG, f)
        f.close()
        cls.adaptive_config_path = f.name

    @classmethod
    def get_server_args(cls) -> list:
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        cls._write_adaptive_config()
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.get_server_args(),
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        if cls.adaptive_config_path and os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _get_spec_steps(self) -> int:
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["internal_states"][0]["speculative_num_steps"]

    def test_server_responds(self):
        """Verify basic generation works."""
        result = self._generate("Hello, world!", max_new_tokens=16)
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)

    def test_adaptive_upshift(self):
        """Drive acceptance high and verify steps increase."""
        for _ in range(MAX_UPSHIFT_ATTEMPTS):
            self._generate(HIGH_ACCEPT_PROMPT, max_new_tokens=128)
            if self._get_spec_steps() == 3:
                return
        self.assertEqual(self._get_spec_steps(), 3, "Never upshifted to steps=3")

    def test_adaptive_downshift(self):
        """Drive acceptance low and verify steps decrease."""
        # First upshift
        for _ in range(MAX_UPSHIFT_ATTEMPTS):
            self._generate(HIGH_ACCEPT_PROMPT, max_new_tokens=128)
            if self._get_spec_steps() == 3:
                break

        # Then downshift
        for _ in range(MAX_DOWNSHIFT_ATTEMPTS):
            self._generate(LOW_ACCEPT_PROMPT, max_new_tokens=128)
            if self._get_spec_steps() == 1:
                return
        self.assertEqual(self._get_spec_steps(), 1, "Never downshifted to steps=1")


# ---------------------------------------------------------------------------
# EAGLE (spec v1) — quick sanity (detailed test in test_adaptive_speculative.py)
# ---------------------------------------------------------------------------


class TestAdaptiveEagle(AdaptiveSpecTestBase):
    model = DEFAULT_TARGET_MODEL_EAGLE

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--attention-backend", "triton",
            "--speculative-algorithm", "EAGLE",
            "--speculative-draft-model-path", DEFAULT_DRAFT_MODEL_EAGLE,
            "--speculative-num-steps", "1",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "2",
            "--speculative-adaptive",
            "--speculative-adaptive-config", cls.adaptive_config_path,
            "--skip-server-warmup",
            "--mem-fraction-static", "0.7",
        ]


# ---------------------------------------------------------------------------
# NGRAM (no draft model)
# ---------------------------------------------------------------------------


class TestAdaptiveNgram(AdaptiveSpecTestBase):
    model = DEFAULT_TARGET_MODEL_NGRAM

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--attention-backend", "triton",
            "--speculative-algorithm", "NGRAM",
            "--speculative-num-draft-tokens", "4",
            "--speculative-adaptive",
            "--speculative-adaptive-config", cls.adaptive_config_path,
            "--skip-server-warmup",
            "--mem-fraction-static", "0.8",
            "--cuda-graph-max-bs", "8",
        ]


# ---------------------------------------------------------------------------
# Standalone (spec v1, inherits EAGLE adaptive)
# ---------------------------------------------------------------------------


class TestAdaptiveStandalone(AdaptiveSpecTestBase):
    model = DEFAULT_TARGET_MODEL_STANDALONE

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--attention-backend", "triton",
            "--speculative-algorithm", "STANDALONE",
            "--speculative-draft-model-path", DEFAULT_DRAFT_MODEL_STANDALONE,
            "--speculative-num-steps", "1",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "2",
            "--speculative-adaptive",
            "--speculative-adaptive-config", cls.adaptive_config_path,
            "--skip-server-warmup",
            "--mem-fraction-static", "0.7",
        ]


# ---------------------------------------------------------------------------
# Standalone V2 (spec v2, overlap scheduling)
# ---------------------------------------------------------------------------


class TestAdaptiveStandaloneV2(AdaptiveSpecTestBase):
    model = DEFAULT_TARGET_MODEL_STANDALONE

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(True)
        super().setUpClass()

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--attention-backend", "triton",
            "--speculative-algorithm", "STANDALONE",
            "--speculative-draft-model-path", DEFAULT_DRAFT_MODEL_STANDALONE,
            "--speculative-num-steps", "1",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "2",
            "--speculative-adaptive",
            "--speculative-adaptive-config", cls.adaptive_config_path,
            "--skip-server-warmup",
            "--mem-fraction-static", "0.7",
        ]


if __name__ == "__main__":
    unittest.main()
