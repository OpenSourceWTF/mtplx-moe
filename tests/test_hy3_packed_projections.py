"""Construction-time packed shared gate/up projections for streamed Hy3."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mtplx.models.hy3_mlx import (
    FUSE_SHARED_GATE_UP_ENV,
    FusedSharedMLP,
    Model,
    ModelArgs,
)


def _tiny_args() -> ModelArgs:
    return ModelArgs(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
    )


def test_flag_off_keeps_original_shared_projection_layout(monkeypatch) -> None:
    monkeypatch.delenv(FUSE_SHARED_GATE_UP_ENV, raising=False)

    model = Model(_tiny_args())

    shared = model.layers[1].mlp.shared_mlp
    assert hasattr(shared, "gate_proj")
    assert hasattr(shared, "up_proj")
    assert not hasattr(shared, "gate_up_proj")


def test_shared_flag_only_packs_sparse_shared_mlp(monkeypatch) -> None:
    monkeypatch.setenv(FUSE_SHARED_GATE_UP_ENV, "true")

    model = Model(_tiny_args())

    dense = model.layers[0].mlp
    assert hasattr(dense, "gate_proj")
    assert hasattr(dense, "up_proj")
    assert not hasattr(dense, "gate_up_proj")

    shared = model.layers[1].mlp.shared_mlp
    assert isinstance(shared, FusedSharedMLP)
    assert hasattr(shared, "gate_up_proj")
    assert not hasattr(shared, "gate_proj")
    assert not hasattr(shared, "up_proj")


def _quantized_component(rows: int, marker: int) -> dict[str, mx.array]:
    return {
        "weight": mx.full((rows, 1), marker, dtype=mx.uint32),
        "scales": mx.full((rows, 1), marker, dtype=mx.bfloat16),
        "biases": mx.full((rows, 1), -marker, dtype=mx.bfloat16),
    }


def _add_component(
    weights: dict[str, mx.array],
    prefix: str,
    rows: int,
    marker: int,
) -> None:
    for suffix, value in _quantized_component(rows, marker).items():
        weights[f"{prefix}.{suffix}"] = value


def test_sanitize_replaces_quantized_sources_without_retaining_duplicates(
    monkeypatch,
) -> None:
    monkeypatch.setenv(FUSE_SHARED_GATE_UP_ENV, "1")
    model = Model(_tiny_args())
    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": mx.zeros((1, 1)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((1,)),
        "model.layers.2.self_attn.q_proj.weight": mx.zeros((1, 1)),
    }
    shared = "model.layers.1.mlp.shared_mlp"
    _add_component(weights, f"{shared}.gate_proj", 3, 4)
    _add_component(weights, f"{shared}.up_proj", 3, 5)

    packed = model.sanitize(weights)

    assert weights == {}
    assert "model.embed_tokens.weight" in packed
    assert not any("rotary_emb.inv_freq" in key for key in packed)
    assert not any("model.layers.2" in key for key in packed)
    assert not any(
        component in key
        for key in packed
        for component in ("shared_mlp.gate_proj", "shared_mlp.up_proj")
    )

    gate_up = packed["model.layers.1.mlp.shared_mlp.gate_up_proj.weight"]
    mx.eval(gate_up)
    assert tuple(gate_up.shape) == (6, 1)
    assert mx.array_equal(
        gate_up[:, 0], mx.array([4, 4, 4, 5, 5, 5], dtype=mx.uint32)
    ).item()


def test_sanitize_rejects_an_incomplete_shared_projection(monkeypatch) -> None:
    monkeypatch.setenv(FUSE_SHARED_GATE_UP_ENV, "1")
    model = Model(_tiny_args())
    weights: dict[str, mx.array] = {}
    _add_component(weights, "model.layers.1.mlp.shared_mlp.gate_proj", 3, 4)

    with pytest.raises(ValueError, match="incomplete packed projection source"):
        model.sanitize(weights)


def test_packed_q4_shared_projection_matches_separate_output_exactly(
    monkeypatch,
) -> None:
    monkeypatch.delenv(FUSE_SHARED_GATE_UP_ENV, raising=False)
    control = Model(_tiny_args())
    nn.quantize(control, group_size=64, bits=4)
    control.eval()
    source = dict(tree_flatten(control.parameters()))

    monkeypatch.setenv(FUSE_SHARED_GATE_UP_ENV, "1")
    candidate = Model(_tiny_args())
    packed = candidate.sanitize(dict(source))
    nn.quantize(
        candidate,
        group_size=64,
        bits=4,
        class_predicate=lambda path, _module: f"{path}.scales" in packed,
    )
    candidate.load_weights(list(packed.items()), strict=True)
    candidate.eval()

    inputs = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    control_shared = control.layers[1].mlp.shared_mlp(inputs)
    candidate_shared = candidate.layers[1].mlp.shared_mlp(inputs)
    mx.eval(control_shared, candidate_shared)

    assert mx.array_equal(control_shared, candidate_shared).item()
