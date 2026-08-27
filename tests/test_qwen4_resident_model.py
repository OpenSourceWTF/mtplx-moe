from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from mlx.utils import tree_flatten
import pytest

import mtplx.runtime as runtime
from mtplx.runtime import _model_classes_for_config
from mtplx.mtp_patch import validate_mtp_support


def _tiny_args():
    from mtplx.models.qwen4_omlx import ModelArgs, TextArgs

    text = TextArgs(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        full_attention_interval=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=32,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        ple_embed_dim=16,
        ple_layer_ids=[2],
        ple_conv_kernel_size=2,
        eos_token_id=31,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        tie_word_embeddings=False,
    )
    return ModelArgs(text_config=asdict(text))


def test_qwen4_uses_resident_switch_glu_for_target_and_mtp() -> None:
    from mlx_lm.models.switch_layers import SwitchGLU
    from mtplx.models.qwen4_omlx import Model

    model = Model(_tiny_args())
    target_switch = model.language_model.model.layers[0].mlp.switch_mlp
    mtp_switch = model.language_model.mtp.layers[0].mlp.switch_mlp
    parameter_names = {name for name, _ in tree_flatten(model.parameters())}

    assert isinstance(target_switch, SwitchGLU)
    assert isinstance(mtp_switch, SwitchGLU)
    assert any("model.layers.0.mlp.switch_mlp" in name for name in parameter_names)
    assert any("mtp.layers.0.mlp.switch_mlp" in name for name in parameter_names)
    assert not hasattr(model, "streamed_layers")
    assert validate_mtp_support(model) is True


def test_normal_loader_owns_qwen4_model_classes() -> None:
    from mtplx.models.qwen4_omlx import Model, ModelArgs

    assert _model_classes_for_config({"model_type": "qwen4_exp"}) == (
        Model,
        ModelArgs,
    )


def test_sanitize_drops_only_ngram_payload_and_keeps_moe_weights() -> None:
    from mtplx.models.qwen4_omlx import Model

    model = Model(_tiny_args())
    expert = object()
    ngram_weight = object()
    ngram_scale = object()
    ngram_bias = object()
    ngram_metadata = object()
    weights = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": expert,
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.0.weight": ngram_weight,
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.0.scales": ngram_scale,
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.0.biases": ngram_bias,
        "language_model.model.layers.1.ple.ple_embedding.layer_multipliers": ngram_metadata,
    }

    sanitized = model.sanitize(weights)

    assert sanitized[
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    ] is expert
    assert sanitized[
        "language_model.model.layers.1.ple.ple_embedding.layer_multipliers"
    ] is ngram_metadata
    assert not any("ngram_embedding.0." in key for key in sanitized)


def test_ngram_resources_attach_to_normal_resident_model(monkeypatch) -> None:
    captured = {}
    resources = SimpleNamespace(cache=object(), artifact=object(), report={})

    monkeypatch.setattr(
        runtime,
        "_construct_qwen4_ngram_runtime",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or resources,
    )
    owners = []
    model = object()

    result = runtime._construct_qwen4_ngram_resources(
        "/model",
        model,
        {"model_type": "qwen4_exp"},
        context_tokens=17_408,
        prefill_chunk_tokens=2_048,
        payload_ceiling_bytes=10 * 1024**3,
        target_residency_bytes=75 * 1024**3,
        mx_module=object(),
        temporary_owners=owners,
    )

    assert result is resources
    assert captured["args"][:2] == ("/model", model)
    assert captured["kwargs"] == {
        "config": {"model_type": "qwen4_exp"},
        "context_tokens": 17_408,
        "prefill_chunk_tokens": 2_048,
        "payload_ceiling_bytes": 10 * 1024**3,
        "target_residency_bytes": 75 * 1024**3,
        "mx_module": captured["kwargs"]["mx_module"],
    }
    assert owners == [resources.cache, resources.artifact]


def test_load_rejects_qwen_preflight_before_constructing_mlx_model(
    tmp_path, monkeypatch
) -> None:
    config = {"model_type": "qwen4_exp"}
    base_load_called = False

    monkeypatch.setattr(runtime, "load_config", lambda _path: config)
    monkeypatch.setattr(
        runtime,
        "_preflight_qwen4_resident",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("preflight rejected before MLX")
        ),
    )

    def forbidden_base_load(*_args, **_kwargs):
        nonlocal base_load_called
        base_load_called = True
        raise AssertionError("MLX model construction must not begin")

    monkeypatch.setattr(runtime, "_load_base_model", forbidden_base_load)

    with pytest.raises(RuntimeError, match="preflight rejected before MLX"):
        runtime.load(tmp_path)

    assert base_load_called is False
