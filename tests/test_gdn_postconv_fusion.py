"""Correct-by-construction contract for the A3B GDN post-conv lane."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import gdn_capture
from mtplx import runtime as runtime_module


_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class _ArraySpec:
    def __init__(self, shape, dtype=mx.bfloat16) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class _QuantProjection:
    def __init__(self, scales_shape) -> None:
        self.bits = 4
        self.group_size = 64
        self.mode = "affine"
        self.scales = _ArraySpec(scales_shape)


def _fake_a3b_config():
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(_LAYER_TYPES),
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
        },
    }


def _fake_gdn():
    return SimpleNamespace(
        sharding_group=None,
        conv_kernel_size=4,
        conv_dim=8192,
        key_dim=2048,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        A_log=_ArraySpec((32,)),
        dt_bias=_ArraySpec((32,)),
        conv1d=SimpleNamespace(weight=_ArraySpec((8192, 4, 1))),
        in_proj_qkv=_QuantProjection((8192, 32)),
        in_proj_a=_QuantProjection((32, 32)),
        in_proj_b=_QuantProjection((32, 32)),
        norm=lambda out, gate: out,
        out_proj=lambda out: out,
    )


def _fake_a3b_model():
    layers = []
    for kind in _LAYER_TYPES:
        if kind == "linear_attention":
            layers.append(SimpleNamespace(is_linear=True, linear_attn=_fake_gdn()))
        else:
            layers.append(SimpleNamespace(is_linear=False, self_attn=object()))
    inner = SimpleNamespace(layers=layers, fa_idx=3, ssm_idx=0)
    return SimpleNamespace(language_model=SimpleNamespace(model=inner))


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("MTPLX_FUSE_GDN_POST_CONV", raising=False)
    monkeypatch.delenv("MTPLX_NATIVE_GDN_TAIL", raising=False)
    gdn_capture._reset_gdn_postconv_stats_for_tests()
    yield
    gdn_capture._reset_gdn_postconv_stats_for_tests()


def test_flag_off_constructs_unchanged_stock_path() -> None:
    model = _fake_a3b_model()

    assert gdn_capture.prepare_a3b_gdn_postconv(
        model, config=_fake_a3b_config()
    ) is None
    assert all(
        not hasattr(layer.linear_attn, "_mtplx_a3b_gdn_postconv_impl")
        for layer in model.language_model.model.layers
        if layer.is_linear
    )
    assert gdn_capture.gdn_postconv_stats() == {
        "enabled": False,
        "installed": False,
        "installation_status": "disabled",
        "installation_error": None,
        "gdn_layers": 0,
        "validated_contract": None,
    }


def test_exact_a3b_contract_installs_all_30_prebound_routes_after_selfcheck(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    model = _fake_a3b_model()

    plan = gdn_capture.prepare_a3b_gdn_postconv(
        model, config=_fake_a3b_config()
    )
    assert plan is not None
    assert all(
        not hasattr(layer.linear_attn, "_mtplx_a3b_gdn_postconv_impl")
        for layer in model.language_model.model.layers
        if layer.is_linear
    )

    report = gdn_capture.install_a3b_gdn_postconv(
        plan, {"lanes": {"gdn_postconv_inline_g": "ok"}}
    )

    assert report["installed"] is True
    assert report["installation_status"] == "installed"
    assert report["gdn_layers"] == 30
    assert report["validated_contract"] == {
        "batch": 1,
        "logical_m": 2,
        "conv_shape": [1, 2, 8192],
        "gate_shapes": {"a": [1, 2, 32], "b": [1, 2, 32]},
        "state_shape": [1, 32, 128, 128],
        "output_shape": [1, 2, 32, 128],
        "captured_states_shape": [1, 2, 32, 128, 128],
        "input_dtype": "bfloat16",
        "state_dtype": "float32",
        "key_heads": 16,
        "value_heads": 32,
        "key_axis": 128,
        "value_axis": 128,
        "threadgroup": [32, 4, 1],
    }
    assert all(
        callable(
            getattr(layer.linear_attn, "_mtplx_a3b_gdn_postconv_impl", None)
        )
        for layer in model.language_model.model.layers
        if layer.is_linear
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "layer_count",
        "topology",
        "sharding",
        "conv_geometry",
        "head_geometry",
        "parameter_shape",
        "parameter_dtype",
        "projection_quantization",
    ),
)
def test_invalid_external_model_contract_prevents_installation(
    monkeypatch, mutation
) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    model = _fake_a3b_model()
    config = _fake_a3b_config()
    inner = model.language_model.model
    first = inner.layers[0].linear_attn
    if mutation == "layer_count":
        inner.layers.pop()
    elif mutation == "topology":
        inner.layers[2], inner.layers[3] = inner.layers[3], inner.layers[2]
    elif mutation == "sharding":
        first.sharding_group = object()
    elif mutation == "conv_geometry":
        first.conv_dim = 4096
    elif mutation == "head_geometry":
        first.num_k_heads = 8
    elif mutation == "parameter_shape":
        first.A_log = _ArraySpec((16,))
    elif mutation == "parameter_dtype":
        first.dt_bias = _ArraySpec((32,), mx.float32)
    elif mutation == "projection_quantization":
        first.in_proj_a.group_size = 128

    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError, match=mutation):
        gdn_capture.prepare_a3b_gdn_postconv(model, config=config)
    assert gdn_capture.gdn_postconv_stats()["installed"] is False


def test_selfcheck_failure_prevents_installation(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    model = _fake_a3b_model()
    plan = gdn_capture.prepare_a3b_gdn_postconv(
        model, config=_fake_a3b_config()
    )

    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError, match="selfcheck"):
        gdn_capture.install_a3b_gdn_postconv(
            plan, {"lanes": {"gdn_postconv_inline_g": "fallback"}}
        )
    assert all(
        not hasattr(layer.linear_attn, "_mtplx_a3b_gdn_postconv_impl")
        for layer in model.language_model.model.layers
        if layer.is_linear
    )


def test_installed_hot_route_and_exact_entrypoint_have_no_runtime_validation() -> None:
    enabled_source = inspect.getsource(
        gdn_capture._apply_enabled_a3b_gdn_postconv_m2_tgy4
    )
    entrypoint_source = inspect.getsource(
        gdn_capture._a3b_compiled_target_gdn_postconv_m2_tgy4
    )
    forward_source = inspect.getsource(gdn_capture.gdn_forward_with_capture)

    forbidden = (
        "os.environ",
        "_env_enabled",
        "lane_disabled",
        "eligible",
        "fallback",
        "try:",
        "except ",
        ".shape",
        ".dtype",
        "getattr",
        "gdn.",
        "return None",
    )
    assert all(item not in enabled_source for item in forbidden)
    assert all(item not in entrypoint_source for item in forbidden)
    assert "MTPLX_FUSE_GDN_POST_CONV" not in forward_source
    assert 'lane_disabled("gdn_postconv_inline_g")' not in forward_source
    assert "_a3b_gdn_postconv_eligible" not in forward_source
    assert "_record_gdn_postconv_fallback" not in forward_source
    assert "_mtplx_a3b_gdn_postconv_impl" in forward_source
    assert "threadgroup=(32, 4, 1)" in entrypoint_source
    assert "grid=(32, 128, 32)" in entrypoint_source
    assert "output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128)]" in entrypoint_source


def test_hot_route_does_not_validate_internal_artifacts(monkeypatch) -> None:
    sentinel = (object(), object())
    conv_out = object()
    a = object()
    b = object()
    state = object()
    A_log = object()
    dt_bias = object()
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        gdn_capture,
        "_linear_gated_delta_from_conv_inline_g_kernel",
        kernel,
    )

    assert gdn_capture._apply_enabled_a3b_gdn_postconv_m2_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    ) == sentinel
    assert calls[0]["inputs"] == [conv_out, a, b, A_log, dt_bias, state, 2]


def test_runtime_contract_propagates_only_the_postconv_enable_flag() -> None:
    from mtplx.profiles import normalize_runtime_env_overrides

    assert normalize_runtime_env_overrides(
        {"MTPLX_FUSE_GDN_POST_CONV": True}
    ) == {"MTPLX_FUSE_GDN_POST_CONV": "1"}


def test_runtime_prepares_before_selfcheck_and_installs_afterward() -> None:
    source = inspect.getsource(runtime_module.load)

    prepare = source.index("prepare_a3b_gdn_postconv(model, config=config)")
    selfcheck = source.index("maybe_run_model_selfcheck(model)")
    install = source.index("install_a3b_gdn_postconv(")
    assert prepare < selfcheck < install
