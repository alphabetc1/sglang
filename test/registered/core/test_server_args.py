import json
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestPrepareServerArgs(CustomTestCase):
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )
        self.assertEqual(
            server_args.model_path, "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.assertEqual(
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )


class TestLoadBalanceMethod(unittest.TestCase):
    def test_non_pd_defaults_to_round_robin(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="null")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="prefill")
        self.assertEqual(server_args.load_balance_method, "follow_bootstrap_room")

    def test_pd_decode_defaults_to_round_robin(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        self.assertEqual(server_args.load_balance_method, "round_robin")


class TestHiCacheArgsNormalization(unittest.TestCase):
    def _make_args(self, **overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _assert_hicache_fields(
        self,
        args: ServerArgs,
        *,
        expected_io_backend: str,
        expected_mem_layout: str,
        expected_decode_backend: str | None = None,
    ):
        self.assertEqual(args.hicache_io_backend, expected_io_backend)
        self.assertEqual(args.hicache_mem_layout, expected_mem_layout)
        if expected_decode_backend is not None:
            self.assertEqual(args.decode_attention_backend, expected_decode_backend)

    def test_hicache_io_backend_and_mem_layout_compatibility(self):
        cases = [
            {
                "name": "kernel_with_page_first_direct",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first_direct",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "direct_with_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "page_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "mooncake_with_layer_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_storage_backend": "mooncake",
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "layer_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "fa3_kernel_with_explicit_decode_backend",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first",
                    "attention_backend": "triton",
                    "decode_attention_backend": "fa3",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                args = self._make_args(**case["overrides"])
                args._handle_hicache()
                self._assert_hicache_fields(
                    args,
                    expected_io_backend=case["expected_io_backend"],
                    expected_mem_layout=case["expected_mem_layout"],
                )

    @patch.object(ServerArgs, "use_mla_backend", return_value=False)
    @patch("sglang.srt.server_args.is_flashinfer_available", return_value=False)
    def test_decode_attention_backend_with_implicit_fa3(
        self, _mock_flashinfer, _mock_use_mla_backend
    ):
        args = self._make_args(
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel",
            attention_backend="fa3",
            decode_attention_backend=None,
        )

        args._handle_hicache()

        self.assertEqual(args.decode_attention_backend, "triton")


class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.get_free_port")
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_with_nccl_port_none(self, mock_temp_file, mock_get_free_port):
        """Test that get_free_port() is called when nccl_port is None"""
        mock_temp_file.return_value.name = "temp_file"
        mock_get_free_port.return_value = 45678  # Mock ephemeral port

        # Use MagicMock here to verify get_free_port is called
        server_args = MagicMock()
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.tokenizer_worker_num = 1

        port_args = PortArgs.init_new(server_args)

        # Verify get_free_port was called
        mock_get_free_port.assert_called_once()

        # Verify the returned port is used
        self.assertEqual(port_args.nccl_port, 45678)

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_single_node_dp_attention(self):

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_dp_rank(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        worker_ports = [25006, 25007, 25008, 25009]
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=worker_ports)

        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_malformed_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(AssertionError) as context:
            PortArgs.init_new(server_args)

        self.assertIn(
            "please provide --dist-init-addr as host:port", str(context.exception)
        )

    def test_init_new_with_malformed_ipv4_address_invalid_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError):
            PortArgs.init_new(server_args)

    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_ipv6_address(self, mock_is_valid_ipv6):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://[2001:db8::1]:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertTrue(
            port_args.detokenizer_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=False)
    def test_init_new_with_invalid_ipv6_address(self, mock_is_valid_ipv6):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[invalid-ipv6]:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid IPv6 address", str(context.exception))

    def test_init_new_with_malformed_ipv6_address_missing_bracket(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid IPv6 address format", str(context.exception))

    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_missing_port(
        self, mock_is_valid_ipv6
    ):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn(
            "a port must be specified in IPv6 address", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_invalid_port(
        self, mock_is_valid_ipv6
    ):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:abcde"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid port in IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_wrong_separator(
        self, mock_is_valid_ipv6
    ):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]#25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("expected ':' after ']'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
