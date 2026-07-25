"""Parity tests for construction-time MoE gate/up packing.

Everything here runs on the CPU stream: the packing itself is weight surgery
and the parity claim is about which dot products produce which outputs, so it
does not need Metal.  A GPU parity run on a real Qwen MoE artifact is tracked
separately.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.moe_packed_projections import (
    PACK_GATE_UP_ENV,
    PackedGateUpMLP,
    PackedSwitchGLU,
    _pack_pair,
    configure_moe_packed_projections,
    moe_pack_gate_up_enabled,
    moe_packed_projection_stats,
)

pytest.importorskip("mlx_lm.models.qwen3_next")

HIDDEN = 128
MOE_INTERMEDIATE = 64
SHARED_INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 4
GROUP_SIZE = 64
BITS = 4


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Keep every op in these tests off the Metal device."""
    with mx.stream(mx.cpu):
        yield


def _moe_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE_INTERMEDIATE,
        shared_expert_intermediate_size=SHARED_INTERMEDIATE,
        norm_topk_prob=True,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
    )


def _make_block(quantized: bool = True):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    mx.random.seed(1234)
    block = Qwen3NextSparseMoeBlock(_moe_args())
    if quantized:
        nn.quantize(block, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(block.parameters())
    return block


class _Wrapper(nn.Module):
    """Minimal model-shaped container so named_modules() finds the block."""

    def __init__(self, block):
        super().__init__()
        self.block = block


# --------------------------------------------------------------------------
# The core claim: concatenating output rows preserves every dot product.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quantized", [True, False])
def test_packed_linear_pair_is_bitwise_exact(quantized: bool):
    mx.random.seed(7)
    gate = nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=False)
    up = nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=False)
    if quantized:
        gate = nn.QuantizedLinear.from_linear(gate, group_size=GROUP_SIZE, bits=BITS)
        up = nn.QuantizedLinear.from_linear(up, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(gate.parameters(), up.parameters())

    x = mx.random.normal((5, HIDDEN))
    stock_gate, stock_up = gate(x), up(x)

    result = _pack_pair(gate, up, axis=0)
    assert not isinstance(result, str), result
    packed, split_at = result
    assert split_at == SHARED_INTERMEDIATE

    fused = packed(x)
    got_gate, got_up = mx.split(fused, [split_at], axis=-1)

    assert mx.array_equal(got_gate, stock_gate)
    assert mx.array_equal(got_up, stock_up)


@pytest.mark.parametrize("quantized", [True, False])
def test_packed_switch_pair_is_bitwise_exact(quantized: bool):
    from mlx_lm.models.switch_layers import SwitchLinear

    mx.random.seed(11)
    gate = SwitchLinear(HIDDEN, MOE_INTERMEDIATE, NUM_EXPERTS, bias=False)
    up = SwitchLinear(HIDDEN, MOE_INTERMEDIATE, NUM_EXPERTS, bias=False)
    if quantized:
        gate = gate.to_quantized(group_size=GROUP_SIZE, bits=BITS)
        up = up.to_quantized(group_size=GROUP_SIZE, bits=BITS)
    mx.eval(gate.parameters(), up.parameters())

    x = mx.random.normal((3, 1, 1, HIDDEN))
    indices = mx.array([[0, 5], [2, 7], [1, 3]])
    stock_gate = gate(x, indices)
    stock_up = up(x, indices)

    result = _pack_pair(gate, up, axis=1)
    assert not isinstance(result, str), result
    packed, split_at = result
    assert split_at == MOE_INTERMEDIATE

    fused = packed.gather(x, indices, False)
    got_gate, got_up = mx.split(fused, [split_at], axis=-1)

    assert mx.array_equal(got_gate, stock_gate)
    assert mx.array_equal(got_up, stock_up)


# --------------------------------------------------------------------------
# Module-level parity, including SwitchGLU's token-sorting path.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 3, 20])
@pytest.mark.parametrize("quantized", [True, False])
def test_packed_switch_glu_matches_stock(tokens: int, quantized: bool):
    block = _make_block(quantized=quantized)
    switch_mlp = block.switch_mlp

    mx.random.seed(21)
    x = mx.random.normal((tokens, HIDDEN))
    indices = mx.broadcast_to(mx.arange(TOP_K), (tokens, TOP_K))
    stock = switch_mlp(x, indices)

    # tokens=20 crosses SwitchGLU's indices.size >= 64 sorting threshold.
    assert (indices.size >= 64) is (tokens == 20)

    result = _pack_pair(switch_mlp.gate_proj, switch_mlp.up_proj, axis=1)
    assert not isinstance(result, str), result
    packed, split_at = result
    fused = PackedSwitchGLU(
        packed, switch_mlp.down_proj, switch_mlp.activation, split_at
    )

    assert mx.array_equal(fused(x, indices), stock)


@pytest.mark.parametrize("quantized", [True, False])
def test_packed_shared_expert_matches_stock(quantized: bool):
    block = _make_block(quantized=quantized)
    shared = block.shared_expert

    mx.random.seed(31)
    x = mx.random.normal((4, HIDDEN))
    stock = shared(x)

    result = _pack_pair(shared.gate_proj, shared.up_proj, axis=0)
    assert not isinstance(result, str), result
    packed, split_at = result
    fused = PackedGateUpMLP(packed, shared.down_proj, split_at)

    assert mx.array_equal(fused(x), stock)


# --------------------------------------------------------------------------
# configure_moe_packed_projections wiring.
# --------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(PACK_GATE_UP_ENV, raising=False)
    assert moe_pack_gate_up_enabled() is False

    model = _Wrapper(_make_block())
    stats = configure_moe_packed_projections(model)

    assert stats["enabled"] is False
    assert stats["packed_switch_mlp"] == 0
    assert stats["packed_shared_expert"] == 0
    # The block is untouched: stock projections still present.
    assert hasattr(model.block.switch_mlp, "gate_proj")
    assert hasattr(model.block.shared_expert, "gate_proj")


@pytest.mark.parametrize("flag", ["1", "true", "on", "yes"])
def test_flag_parsing(monkeypatch, flag: str):
    monkeypatch.setenv(PACK_GATE_UP_ENV, flag)
    assert moe_pack_gate_up_enabled() is True


def test_configure_packs_block_and_preserves_output(monkeypatch):
    model = _Wrapper(_make_block())

    mx.random.seed(41)
    x = mx.random.normal((6, HIDDEN))
    stock = model.block(x)

    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")
    stats = configure_moe_packed_projections(model)

    assert stats["enabled"] is True
    assert stats["packed_switch_mlp"] == 1
    assert stats["packed_shared_expert"] == 1
    assert stats["skip_reasons"] == []
    assert isinstance(model.block.switch_mlp, PackedSwitchGLU)
    assert isinstance(model.block.shared_expert, PackedGateUpMLP)

    assert mx.array_equal(model.block(x), stock)


def test_configure_reaches_blocks_nested_in_layer_lists(monkeypatch):
    """The real model holds its MoE blocks in a list attribute, not directly."""

    class _Layer(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.mlp = block

    class _Model(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.layers = [_Layer(block) for block in blocks]

    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")
    model = _Model([_make_block() for _ in range(3)])

    mx.random.seed(61)
    x = mx.random.normal((2, HIDDEN))
    stock = [layer.mlp(x) for layer in model.layers]

    stats = configure_moe_packed_projections(model)

    assert stats["packed_switch_mlp"] == 3
    assert stats["packed_shared_expert"] == 3
    for layer, expected in zip(model.layers, stock):
        assert isinstance(layer.mlp.switch_mlp, PackedSwitchGLU)
        assert isinstance(layer.mlp.shared_expert, PackedGateUpMLP)
        assert mx.array_equal(layer.mlp(x), expected)


def test_configure_is_idempotent(monkeypatch):
    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")
    model = _Wrapper(_make_block())

    mx.random.seed(51)
    x = mx.random.normal((2, HIDDEN))

    configure_moe_packed_projections(model)
    once = model.block(x)

    second = configure_moe_packed_projections(model)
    assert second["packed_switch_mlp"] == 0
    assert second["packed_shared_expert"] == 0
    assert mx.array_equal(model.block(x), once)


def test_configure_without_moe_blocks_is_noop(monkeypatch):
    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")

    class _Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    stats = configure_moe_packed_projections(_Dense())
    assert stats["packed_switch_mlp"] == 0
    assert stats["packed_shared_expert"] == 0
    assert stats["skipped_blocks"] == 0


def test_configure_handles_none_model(monkeypatch):
    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")
    stats = configure_moe_packed_projections(None)
    assert stats["packed_switch_mlp"] == 0


# --------------------------------------------------------------------------
# Guard rails: a pair that cannot be packed is skipped, never mispacked.
# --------------------------------------------------------------------------


def test_mismatched_output_widths_are_skipped():
    gate = nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=False)
    up = nn.Linear(HIDDEN, SHARED_INTERMEDIATE * 2, bias=False)
    mx.eval(gate.parameters(), up.parameters())

    reason = _pack_pair(gate, up, axis=0)
    assert isinstance(reason, str)
    assert "output widths" in reason


def test_mismatched_quantization_is_skipped():
    gate = nn.QuantizedLinear.from_linear(
        nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=False),
        group_size=GROUP_SIZE,
        bits=BITS,
    )
    up = nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=False)
    mx.eval(gate.parameters(), up.parameters())

    reason = _pack_pair(gate, up, axis=0)
    assert isinstance(reason, str)
    assert "quantization" in reason


def test_additive_bias_is_skipped():
    gate = nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=True)
    up = nn.Linear(HIDDEN, SHARED_INTERMEDIATE, bias=True)
    mx.eval(gate.parameters(), up.parameters())

    reason = _pack_pair(gate, up, axis=0)
    assert isinstance(reason, str)
    assert "bias" in reason


def test_skip_reason_surfaces_in_stats(monkeypatch):
    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")
    model = _Wrapper(_make_block(quantized=False))
    # Make the shared expert unpackable without touching the routed experts.
    model.block.shared_expert.up_proj = nn.Linear(
        HIDDEN, SHARED_INTERMEDIATE * 2, bias=False
    )
    mx.eval(model.block.shared_expert.parameters())

    stats = configure_moe_packed_projections(model)

    assert stats["packed_switch_mlp"] == 1
    assert stats["packed_shared_expert"] == 0
    assert stats["skipped_blocks"] == 1
    assert any("shared_expert" in reason for reason in stats["skip_reasons"])


def test_stats_snapshot_is_a_copy(monkeypatch):
    monkeypatch.setenv(PACK_GATE_UP_ENV, "1")
    configure_moe_packed_projections(_Wrapper(_make_block()))
    snapshot = moe_packed_projection_stats()
    snapshot["skip_reasons"].append("mutation")
    assert "mutation" not in moe_packed_projection_stats()["skip_reasons"]
