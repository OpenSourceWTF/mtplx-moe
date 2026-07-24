"""Decode-lane submission cadence: gating and bitwise parity."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.models import hy3_mlx
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args


def _bind_execution_policy(
    monkeypatch: pytest.MonkeyPatch,
    raw_cadence: str | None,
) -> None:
    environ = (
        {}
        if raw_cadence is None
        else {hy3_mlx.SUBMIT_CADENCE_ENV: raw_cadence}
    )
    monkeypatch.setattr(
        hy3_mlx,
        "_HY3_EXECUTION_POLICY",
        hy3_mlx.bind_hy3_execution_policy(environ),
    )


def _tiny_args() -> Hy3Args:
    return Hy3Args(
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
        router_scaling_factor=2.0,
    )


class _NullMoE(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x * 0


def _fresh_model() -> Hy3Model:
    mx.random.seed(7)
    model = Hy3Model(_tiny_args())
    model.model.layers[1].mlp = _NullMoE()
    return model


def _decode_logits(model: Hy3Model) -> mx.array:
    cache = model.make_cache()
    mx.eval(model(mx.array([[3, 5, 7, 11]], dtype=mx.int32), cache=cache))
    logits = model(mx.array([[6]], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    return logits


def test_cadence_parses_and_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind_execution_policy(monkeypatch, None)
    assert hy3_mlx._decode_submit_cadence() == 0
    _bind_execution_policy(monkeypatch, "6")
    assert hy3_mlx._decode_submit_cadence() == 6
    for bad in ("", "0", "-3", "nope"):
        _bind_execution_policy(monkeypatch, bad)
        assert hy3_mlx._decode_submit_cadence() == 0


def test_cadence_submits_on_decode_not_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}
    original = mx.async_eval

    def counting(*values):
        calls["n"] += 1
        return original(*values)

    _bind_execution_policy(monkeypatch, "1")
    monkeypatch.setattr(hy3_mlx.mx, "async_eval", counting)
    model = _fresh_model()
    cache = model.make_cache()
    # Prefill (9 rows > 8) must not checkpoint.
    mx.eval(model(mx.array([list(range(9))], dtype=mx.int32), cache=cache))
    assert calls["n"] == 0
    # Single-token decode checkpoints once per layer at cadence 1.
    mx.eval(model(mx.array([[6]], dtype=mx.int32), cache=cache))
    assert calls["n"] == len(model.model.layers)


def test_cadence_is_bitwise_invisible(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind_execution_policy(monkeypatch, None)
    baseline = _decode_logits(_fresh_model())

    calls = {"n": 0}
    original = mx.async_eval

    def counting(*values):
        calls["n"] += 1
        return original(*values)

    _bind_execution_policy(monkeypatch, "1")
    monkeypatch.setattr(hy3_mlx.mx, "async_eval", counting)
    cadenced = _decode_logits(_fresh_model())
    assert calls["n"] > 0
    assert mx.array_equal(baseline, cadenced).item()
