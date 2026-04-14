from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.glm4_moe_nextn import resolve_glm4_nextn_draft_quant_config


def test_resolve_glm4_nextn_draft_quant_config_preserves_auto_detected_quant():
    quant_config = object()
    with patch(
        "sglang.srt.models.glm4_moe_nextn.get_global_server_args",
        return_value=SimpleNamespace(speculative_draft_model_explicit_unquant=False),
    ):
        assert resolve_glm4_nextn_draft_quant_config(quant_config) is quant_config


def test_resolve_glm4_nextn_draft_quant_config_honors_explicit_unquant():
    quant_config = object()
    with patch(
        "sglang.srt.models.glm4_moe_nextn.get_global_server_args",
        return_value=SimpleNamespace(speculative_draft_model_explicit_unquant=True),
    ):
        assert resolve_glm4_nextn_draft_quant_config(quant_config) is None


def test_resolve_glm4_nextn_draft_quant_config_keeps_unquant_for_none():
    with patch(
        "sglang.srt.models.glm4_moe_nextn.get_global_server_args",
        return_value=SimpleNamespace(speculative_draft_model_explicit_unquant=False),
    ):
        assert resolve_glm4_nextn_draft_quant_config(None) is None
