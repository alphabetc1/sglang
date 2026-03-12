import unittest

from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestNemotronHConfig(CustomTestCase):
    def test_chunk_size_alias_sets_mamba_chunk_size(self):
        config = NemotronHConfig(chunk_size=128)

        self.assertEqual(config.mamba_chunk_size, 128)

    def test_explicit_mamba_chunk_size_wins_over_chunk_size_alias(self):
        config = NemotronHConfig(mamba_chunk_size=64, chunk_size=128)

        self.assertEqual(config.mamba_chunk_size, 64)


if __name__ == "__main__":
    unittest.main()
