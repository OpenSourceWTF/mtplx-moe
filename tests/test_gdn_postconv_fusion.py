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
            "dtype": "bfloat16",
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
    monkeypatch.delenv("MTPLX_COMPILED_TARGET_PREFIX", raising=False)
    monkeypatch.delenv("MTPLX_NATIVE_GDN_TAIL", raising=False)
    gdn_capture._reset_gdn_postconv_stats_for_tests()
    yield
    gdn_capture._reset_gdn_postconv_stats_for_tests()


def test_flag_off_constructs_unchanged_stock_path() -> None:
    model = _fake_a3b_model()

    assert gdn_capture.prepare_a3b_gdn_postconv(
        model, config=_fake_a3b_config()
    ) is None
    assert gdn_capture.gdn_postconv_stats() == {
        "enabled": False,
        "installed": False,
        "installation_status": "disabled",
        "installation_error": None,
        "gdn_layers": 0,
        "validated_contract": None,
        "implementation": "inline_g",
    }


def test_exact_a3b_contract_installs_all_30_prebound_routes_after_selfcheck(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    model = _fake_a3b_model()

    plan = gdn_capture.prepare_a3b_gdn_postconv(
        model, config=_fake_a3b_config()
    )
    assert plan is not None

    factory = gdn_capture.install_a3b_gdn_postconv(
        plan, {"lanes": {"gdn_postconv_inline_g": "ok"}}
    )
    report = gdn_capture.gdn_postconv_stats()

    assert isinstance(factory, gdn_capture.A3BGDNPostconvFactory)
    assert len(factory.m1_implementations) == 30
    assert len(factory.m2_implementations) == 30
    assert len(factory.m3_implementations) == 30
    assert report["installed"] is True
    assert report["installation_status"] == "installed"
    assert report["gdn_layers"] == 30
    assert report["validated_contract"] == {
        "batch": 1,
        "logical_m": [1, 2, 3],
        "routes": {
            "m1_correction": {
                "conv_shape": [1, 1, 8192],
                "gate_shapes": {"a": [1, 1, 32], "b": [1, 1, 32]},
                "output_shape": [1, 1, 32, 128],
                "captured_states_shape": [1, 1, 32, 128, 128],
            },
            "m2_verify": {
                "conv_shape": [1, 2, 8192],
                "gate_shapes": {"a": [1, 2, 32], "b": [1, 2, 32]},
                "output_shape": [1, 2, 32, 128],
                "captured_states_shape": [1, 2, 32, 128, 128],
            },
            "m3_verify": {
                "conv_shape": [1, 3, 8192],
                "gate_shapes": {"a": [1, 3, 32], "b": [1, 3, 32]},
                "output_shape": [1, 3, 32, 128],
                "captured_states_shape": [1, 3, 32, 128, 128],
            },
        },
        "state_shape": [1, 32, 128, 128],
        "input_dtype": "bfloat16",
        "state_dtype": "float32",
        "key_heads": 16,
        "value_heads": 32,
        "key_axis": 128,
        "value_axis": 128,
        "threadgroup": [32, 4, 1],
    }
    gdns = tuple(
        layer.linear_attn
        for layer in model.language_model.model.layers
        if layer.is_linear
    )
    for gdn, m1_impl, m2_impl, m3_impl in zip(
        gdns,
        factory.m1_implementations,
        factory.m2_implementations,
        factory.m3_implementations,
    ):
        for implementation in (m1_impl, m2_impl, m3_impl):
            assert implementation.keywords == {
                "A_log": gdn.A_log,
                "dt_bias": gdn.dt_bias,
            }
        assert not hasattr(gdn, "_mtplx_a3b_gdn_postconv_m1_impl")
        assert not hasattr(gdn, "_mtplx_a3b_gdn_postconv_m2_impl")


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
        "config_dtype",
    ),
)
def test_invalid_external_model_contract_prevents_installation(
    monkeypatch, mutation
) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
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
    elif mutation == "config_dtype":
        config["text_config"]["dtype"] = "float16"

    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError, match=mutation):
        gdn_capture.prepare_a3b_gdn_postconv(model, config=config)
    assert gdn_capture.gdn_postconv_stats()["installed"] is False


def test_selfcheck_failure_prevents_installation(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    model = _fake_a3b_model()
    plan = gdn_capture.prepare_a3b_gdn_postconv(
        model, config=_fake_a3b_config()
    )

    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError, match="selfcheck"):
        gdn_capture.install_a3b_gdn_postconv(
            plan, {"lanes": {"gdn_postconv_inline_g": "fallback"}}
        )


def test_postconv_flag_without_compiled_target_prefix_fails_before_selfcheck(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")

    with pytest.raises(
        gdn_capture.A3BGDNPostconvConfigError,
        match="compiled_target_prefix_flag",
    ):
        gdn_capture.prepare_a3b_gdn_postconv(
            _fake_a3b_model(),
            config=_fake_a3b_config(),
        )

    report = gdn_capture.gdn_postconv_stats()
    assert report["installed"] is False
    assert report["installation_status"] == "configuration_error"


def test_fixed_m1_m2_entrypoints_encode_exact_geometry() -> None:
    m1_enabled = inspect.getsource(gdn_capture._apply_enabled_a3b_gdn_postconv_m1_tgy4)
    m2_enabled = inspect.getsource(gdn_capture._apply_enabled_a3b_gdn_postconv_m2_tgy4)
    m1_entrypoint = inspect.getsource(
        gdn_capture._a3b_compiled_target_gdn_postconv_m1_tgy4
    )
    m2_entrypoint = inspect.getsource(
        gdn_capture._a3b_compiled_target_gdn_postconv_m2_tgy4
    )
    exact_gdn_source = inspect.getsource(
        gdn_capture._a3b_gdn_forward_with_fixed_postconv
    )

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
    assert all(item not in m1_enabled for item in forbidden)
    assert all(item not in m2_enabled for item in forbidden)
    assert all(item not in m1_entrypoint for item in forbidden)
    assert all(item not in m2_entrypoint for item in forbidden)
    for projection in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        assert f"gdn.{projection}(inputs)" in exact_gdn_source
    assert "_stock_conv1d_capture(qkv, conv_state, gdn)" in exact_gdn_source
    assert "postconv_implementation(conv_out, a, b, cache[1])" in exact_gdn_source
    assert "cache[0] = mx.contiguous(conv_states[:, -1, :, :])" in exact_gdn_source
    assert "cache[1] = states[:, -1, :, :, :]" in exact_gdn_source
    assert "out = gdn.norm(out, z)" in exact_gdn_source
    assert "out = gdn.out_proj(out.reshape(B, S, -1))" in exact_gdn_source
    assert "state, 1" in m1_entrypoint
    assert "state, 2" in m2_entrypoint
    assert "threadgroup=(32, 4, 1)" in m1_entrypoint
    assert "threadgroup=(32, 4, 1)" in m2_entrypoint
    assert "grid=(32, 128, 32)" in m1_entrypoint
    assert "grid=(32, 128, 32)" in m2_entrypoint
    assert "output_shapes=[(1, 1, 32, 128), (1, 1, 32, 128, 128)]" in m1_entrypoint
    assert "output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128)]" in m2_entrypoint


def test_exact_capture_is_transitively_separate_from_generic_capture() -> None:
    exact_gdn_source = inspect.getsource(
        gdn_capture._a3b_gdn_forward_with_fixed_postconv
    )
    exact_model_source = inspect.getsource(
        gdn_capture.forward_with_a3b_gdn_postconv_capture
    )
    runtime_source = inspect.getsource(
        runtime_module.MTPLXRuntime._forward_ar_capture_a3b_postconv
    )
    generic_source = inspect.getsource(gdn_capture.forward_with_gdn_capture)

    forbidden = (
        "gdn_forward_with_capture",
        "_forward_with_gdn_capture",
        "_gdn_input_projections",
        "_maybe_contiguous_authoritative_gdn_leaf",
        "resolve_gdn_capture_backend",
        "os.environ",
        "_env_enabled",
        "lane_disabled",
        "eligible",
        "fallback",
        "return None",
        "sharding_group",
    )
    assert all(item not in exact_gdn_source for item in forbidden)
    assert all(item not in exact_model_source for item in forbidden)
    assert all(item not in runtime_source for item in forbidden)
    assert "mask" not in exact_gdn_source
    assert "forward_with_a3b_gdn_postconv_capture" in runtime_source
    assert "postconv_implementations" not in generic_source
    assert "forward_with_a3b_gdn_postconv_capture" not in generic_source


def test_exact_capture_consumes_only_linear_positions_in_proven_order() -> None:
    source = inspect.getsource(gdn_capture.forward_with_a3b_gdn_postconv_capture)

    iterator = source.index("implementation_iter = iter(postconv_implementations)")
    linear_branch = source.index('if kind == "linear_attention":')
    consume = source.index("next(implementation_iter)")
    full_attention_branch = source.index("else:", consume)
    assert iterator < linear_branch < consume < full_attention_branch
    assert "zip(inner.layers, cache, _A3B_GDN_POSTCONV_LAYER_TYPES)" in source


@pytest.mark.parametrize("logical_m", (1, 2))
def test_exact_gdn_fixed_surroundings_match_explicit_reference(
    monkeypatch,
    logical_m,
) -> None:
    inputs = mx.zeros((1, logical_m, 2048), dtype=mx.bfloat16)
    qkv = mx.full((1, logical_m, 8192), 0.25, dtype=mx.bfloat16)
    z = mx.full((1, logical_m, 4096), 0.5, dtype=mx.bfloat16)
    a = mx.full((1, logical_m, 32), 0.75, dtype=mx.bfloat16)
    b = mx.full((1, logical_m, 32), 1.0, dtype=mx.bfloat16)
    conv_state = mx.zeros((1, 3, 8192), dtype=mx.bfloat16)
    conv_out = mx.full((1, logical_m, 8192), 1.25, dtype=mx.bfloat16)
    conv_states = mx.full(
        (1, logical_m, 3, 8192),
        1.5,
        dtype=mx.bfloat16,
    )
    state = mx.zeros((1, 32, 128, 128), dtype=mx.float32)
    postconv_out = mx.full(
        (1, logical_m, 32, 128),
        2.0,
        dtype=mx.bfloat16,
    )
    states = mx.full(
        (1, logical_m, 32, 128, 128),
        2.5,
        dtype=mx.float32,
    )
    seen: list[tuple] = []
    gdn = SimpleNamespace(
        in_proj_qkv=lambda value: seen.append(("qkv", value)) or qkv,
        in_proj_z=lambda value: seen.append(("z", value)) or z,
        in_proj_b=lambda value: seen.append(("b", value)) or b,
        in_proj_a=lambda value: seen.append(("a", value)) or a,
        norm=lambda value, gate: value + gate,
        out_proj=lambda value: value * 2,
    )

    def stock_conv(projected, base_state, owner):
        seen.append(("conv", projected, base_state, owner))
        return conv_out, conv_states

    def postconv(conv, gate_a, gate_b, recurrent):
        seen.append(("postconv", conv, gate_a, gate_b, recurrent))
        return postconv_out, states

    monkeypatch.setattr(gdn_capture, "_stock_conv1d_capture", stock_conv)
    cache = [conv_state, state]

    out, captures = gdn_capture._a3b_gdn_forward_with_fixed_postconv(
        gdn,
        inputs,
        cache,
        postconv,
    )
    expected = (postconv_out + z.reshape(1, logical_m, 32, 128)).reshape(
        1, logical_m, -1
    ) * 2
    mx.eval(out, expected, *cache)

    assert mx.array_equal(out, expected)
    assert mx.array_equal(cache[0], conv_states[:, -1, :, :])
    assert mx.array_equal(cache[1], states[:, -1, :, :, :])
    assert captures["conv_states"] is conv_states
    assert captures["states"] is states
    assert [call[0] for call in seen] == ["qkv", "z", "b", "a", "conv", "postconv"]


@pytest.mark.parametrize(
    ("implementation", "logical_m"),
    (
        ("_apply_enabled_a3b_gdn_postconv_m1_tgy4", 1),
        ("_apply_enabled_a3b_gdn_postconv_m2_tgy4", 2),
    ),
)
def test_hot_route_does_not_validate_internal_artifacts(
    monkeypatch, implementation, logical_m
) -> None:
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

    assert (
        getattr(gdn_capture, implementation)(
            conv_out,
            a,
            b,
            state,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        == sentinel
    )
    assert calls[0]["inputs"] == [
        conv_out,
        a,
        b,
        A_log,
        dt_bias,
        state,
        logical_m,
    ]


def test_runtime_contract_propagates_only_the_postconv_enable_flag() -> None:
    from mtplx.profiles import normalize_runtime_env_overrides

    assert normalize_runtime_env_overrides(
        {"MTPLX_FUSE_GDN_POST_CONV": True}
    ) == {"MTPLX_FUSE_GDN_POST_CONV": "1"}


def test_runtime_finalizes_compiled_factory_only_after_postconv_install() -> None:
    source = inspect.getsource(runtime_module.load)

    prepare = source.index("prepare_a3b_gdn_postconv(model, config=config)")
    selfcheck = source.index("maybe_run_model_selfcheck(model)")
    install = source.index("install_a3b_gdn_postconv(")
    compiled_factory = source.index("prepare_a3b_compiled_target_prefix(")
    assert prepare < selfcheck < install < compiled_factory
    assert "gdn_postconv_factory=postconv_factory" in source
    assert (
        "a3b_compiled_target_prefix_factory=compiled_target_factory" in source
    )
