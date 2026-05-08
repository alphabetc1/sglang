"""Unit tests for the DSV4 hicache v1 startup assert function."""

import types

import pytest

from sglang.srt.managers.scheduler import _assert_dsv4_hicache_v1_supported


def _args(**kw):
    base = dict(
        enable_hierarchical_cache=True,
        enable_hisparse=False,
        enable_dp_attention=False,
        pp_size=1,
        hicache_storage_backend=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _model_config_dsv4():
    cfg = types.SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    return types.SimpleNamespace(hf_config=cfg)


def _model_config_other():
    cfg = types.SimpleNamespace(architectures=["LlamaForCausalLM"])
    return types.SimpleNamespace(hf_config=cfg)


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(enable_hisparse=True), "HiSparse"),
        (dict(enable_dp_attention=True), "dp-attention"),
        (dict(pp_size=2), "pipeline parallelism"),
        (dict(hicache_storage_backend="hf3fs"), "storage-backend"),
    ],
)
def test_assert_blocks_unsupported_combos(kw, match):
    with pytest.raises(ValueError, match=match):
        _assert_dsv4_hicache_v1_supported(_args(**kw), _model_config_dsv4())


def test_assert_passes_supported_combo():
    _assert_dsv4_hicache_v1_supported(_args(), _model_config_dsv4())


def test_assert_no_op_for_non_dsv4():
    # Even with all "bad" kwargs, non-DSV4 models must not raise.
    _assert_dsv4_hicache_v1_supported(
        _args(enable_hisparse=True, pp_size=4, hicache_storage_backend="hf3fs"),
        _model_config_other(),
    )


def test_assert_no_op_when_hicache_disabled():
    # DSV4 without hicache must not raise even if other combos look bad.
    _assert_dsv4_hicache_v1_supported(
        _args(enable_hierarchical_cache=False, enable_hisparse=True),
        _model_config_dsv4(),
    )
