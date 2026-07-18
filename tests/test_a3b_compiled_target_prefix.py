from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from mtplx import a3b_compiled_target_prefix as a3b_target


LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


def _config() -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(LAYER_TYPES),
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_value_head_dim": 128,
            "linear_key_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "mtp_num_hidden_layers": 1,
        },
    }


def _model() -> SimpleNamespace:
    layers = []
    for index, layer_type in enumerate(LAYER_TYPES):
        is_linear = layer_type == "linear_attention"
        layer = SimpleNamespace(is_linear=is_linear)
        if is_linear:
            layer.linear_attn = SimpleNamespace(
                sharding_group=None,
                num_v_heads=32,
                num_k_heads=16,
                head_v_dim=128,
                head_k_dim=128,
                conv_kernel_size=4,
                conv_dim=8192,
            )
        else:
            layer.self_attn = SimpleNamespace(
                sharding_group=None,
                num_heads=16,
                num_kv_heads=2,
                head_dim=256,
            )
        layers.append(layer)
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        mtp=SimpleNamespace(layers=[SimpleNamespace()]),
    )


def test_flag_off_installs_no_model_factory(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_COMPILED_TARGET_PREFIX", raising=False)
    model = SimpleNamespace()

    assert a3b_target.prepare_a3b_compiled_target_prefix(model, config={}) is None
    assert not hasattr(model, "_mtplx_a3b_compiled_target_prefix_factory")


def test_exact_model_contract_installs_one_immutable_factory(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    model = _model()

    factory = a3b_target.prepare_a3b_compiled_target_prefix(
        model,
        config=_config(),
    )

    assert factory is not None
    assert model._mtplx_a3b_compiled_target_prefix_factory is factory
    assert factory.layer_types == LAYER_TYPES
    assert factory.gdn_layers == 30
    assert factory.full_attention_layers == 10


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("quantization", "bits"), 8),
        (("quantization", "group_size"), 32),
        (("text_config", "hidden_size"), 4096),
        (("text_config", "num_hidden_layers"), 39),
        (("text_config", "num_key_value_heads"), 4),
        (("text_config", "head_dim"), 128),
    ),
)
def test_invalid_model_contract_fails_during_load(monkeypatch, path, value) -> None:
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    config = _config()
    config[path[0]][path[1]] = value

    with pytest.raises(a3b_target.A3BCompiledTargetPrefixConfigError):
        a3b_target.prepare_a3b_compiled_target_prefix(_model(), config=config)


def test_installed_m2_dispatch_contains_no_runtime_validation_or_fallback() -> None:
    source = inspect.getsource(a3b_target.A3BK1TargetPrefixRoute._forward_m2)

    for forbidden in (
        "os.environ",
        "getenv",
        "shape",
        "dtype",
        "validate",
        "eligible",
        "promote",
        "build_verify_state_spec",
        "fallback",
        "stats",
        "try:",
        "except",
        "forward_ar",
        "_decode_length",
        "_unpack_outputs",
        "_rebuild_captures",
    ):
        assert forbidden not in source


def test_fixed_compiled_body_contains_no_dynamic_length_or_output_validation() -> None:
    source = inspect.getsource(a3b_target._make_a3b_k1_target_prefix_m2_step)

    assert "_decode_length" not in source
    assert "len(outputs)" not in source
    assert "expected" not in source
    assert "fallback" not in source


def test_final_report_derives_m2_engagement_without_hot_counters() -> None:
    route = object.__new__(a3b_target.A3BK1TargetPrefixRoute)
    route.request_max_tokens = 10_000
    route.growth_reserve_tokens = 10_002
    route.prompt_tokens = 181

    report = route.final_report(verify_calls=1_604, repair_calls=318)

    assert report["calls"] == 1_922
    assert report["compiled_calls"] == 1_922
    assert report["m2_calls"] == 1_922
    assert report["buckets"] == {"0": 1_922}
    assert report["fallback_calls"] == 0
    assert report["fallback_reasons"] == {}
    assert report["growth_demotions"] == 0


def test_generation_exact_route_has_no_per_cycle_stats_or_trim_probe() -> None:
    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    event_block = source[source.index("if graphbank is not None:") : source.index(
        "accepted_count = 0"
    )]

    assert "compiled_verify_bank.to_dict()" not in event_block
    assert "a3b_target_prefix_route.final_report" in source
    assert "if a3b_target_prefix_route is not None:" in source
    assert "committed_from_trim = rejection_correction is None" in source
