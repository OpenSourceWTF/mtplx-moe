"""Pin the layer-80 resident packaging to the pinned artifact's conventions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "quantize_mtp_layer80_residents.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "quantize_mtp_layer80_residents", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load_module()

_TINY_PREFIX = "model.layers.80."


def _tiny_resident_tensors() -> dict[str, mx.array]:
    """A shape-scaled resident set: hidden 64, heads 4x16, kv 2x16, moe 64."""

    def bf16(*shape):
        return mx.random.normal(shape).astype(mx.bfloat16)

    tensors = {
        _TINY_PREFIX + "self_attn.q_proj.weight": bf16(64, 64),
        _TINY_PREFIX + "self_attn.k_proj.weight": bf16(32, 64),
        _TINY_PREFIX + "self_attn.v_proj.weight": bf16(32, 64),
        _TINY_PREFIX + "self_attn.o_proj.weight": bf16(64, 64),
        _TINY_PREFIX + "mlp.shared_mlp.gate_proj.weight": bf16(64, 64),
        _TINY_PREFIX + "mlp.shared_mlp.up_proj.weight": bf16(64, 64),
        _TINY_PREFIX + "mlp.shared_mlp.down_proj.weight": bf16(64, 64),
        _TINY_PREFIX + "mlp.router.gate.weight": bf16(4, 64),
        _TINY_PREFIX + "eh_proj.weight": bf16(64, 128),
        _TINY_PREFIX + "enorm.weight": bf16(64),
        _TINY_PREFIX + "hnorm.weight": bf16(64),
        _TINY_PREFIX + "input_layernorm.weight": bf16(64),
        _TINY_PREFIX + "post_attention_layernorm.weight": bf16(64),
        _TINY_PREFIX + "final_layernorm.weight": bf16(64),
        _TINY_PREFIX + "self_attn.q_norm.weight": bf16(16),
        _TINY_PREFIX + "self_attn.k_norm.weight": bf16(16),
        _TINY_PREFIX + "mlp.expert_bias": mx.zeros((4,), dtype=mx.float32),
    }
    mx.eval(list(tensors.values()))
    return tensors


def test_resident_suffix_table_is_complete_and_disjoint() -> None:
    groups = (
        MOD.Q4_SUFFIXES,
        MOD.Q8_SUFFIXES,
        MOD.BF16_PASS_SUFFIXES,
        MOD.F32_PASS_SUFFIXES,
    )
    combined = [suffix for group in groups for suffix in group]
    # The layer-80 checkpoint has exactly 17 non-expert tensors.
    assert len(combined) == 17
    assert len(set(combined)) == 17
    assert set(MOD.RESIDENT_SUFFIXES) == set(combined)


def test_pinned_shape_table_matches_quantization_math() -> None:
    # eh_proj is [4096, 8192]; the shape table must be derivable from the
    # real geometry: packed U32 width in*bits/32, scales width in/64.
    logical_in = {
        "self_attn.q_proj": 4096,
        "self_attn.k_proj": 4096,
        "self_attn.v_proj": 4096,
        "self_attn.o_proj": 8192,
        "mlp.shared_mlp.gate_proj": 4096,
        "mlp.shared_mlp.up_proj": 4096,
        "mlp.shared_mlp.down_proj": 1536,
        "mlp.router.gate": 4096,
    }
    for base, (wshape, gshape) in MOD.PINNED_RESIDENT_Q_SHAPES.items():
        bits = 8 if base == "mlp.router.gate" else 4
        in_dims = logical_in[base]
        assert wshape[1] == in_dims * bits // 32, base
        assert gshape == (wshape[0], in_dims // MOD.GROUP_SIZE), base


def test_quantizes_projections_to_pinned_dtypes_and_shapes() -> None:
    tensors = _tiny_resident_tensors()
    out = MOD.quantize_resident_tensors(tensors)
    mx.eval(list(out.values()))

    # 8 quantized projections -> weight/scales/biases; 9 pass-through.
    assert len(out) == 8 * 3 + 9

    q4_cases = {
        "self_attn.q_proj": (64, 64),
        "self_attn.k_proj": (32, 64),
        "self_attn.v_proj": (32, 64),
        "self_attn.o_proj": (64, 64),
        "mlp.shared_mlp.gate_proj": (64, 64),
        "mlp.shared_mlp.up_proj": (64, 64),
        "mlp.shared_mlp.down_proj": (64, 64),
    }
    for base, (rows, cols) in q4_cases.items():
        weight = out[_TINY_PREFIX + base + ".weight"]
        scales = out[_TINY_PREFIX + base + ".scales"]
        biases = out[_TINY_PREFIX + base + ".biases"]
        assert weight.dtype == mx.uint32
        assert tuple(weight.shape) == (rows, cols * 4 // 32)
        assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
        assert tuple(scales.shape) == (rows, cols // 64)
        assert tuple(biases.shape) == (rows, cols // 64)

    gate_weight = out[_TINY_PREFIX + "mlp.router.gate.weight"]
    gate_scales = out[_TINY_PREFIX + "mlp.router.gate.scales"]
    assert gate_weight.dtype == mx.uint32
    assert tuple(gate_weight.shape) == (4, 64 * 8 // 32)
    assert gate_scales.dtype == mx.bfloat16
    assert tuple(gate_scales.shape) == (4, 1)


def test_pass_through_tensors_keep_source_precision() -> None:
    tensors = _tiny_resident_tensors()
    out = MOD.quantize_resident_tensors(tensors)

    for suffix in MOD.BF16_PASS_SUFFIXES:
        name = _TINY_PREFIX + suffix
        assert out[name].dtype == mx.bfloat16
        assert out[name].shape == tensors[name].shape
        assert name + ".scales" not in out and _TINY_PREFIX + suffix.replace(
            ".weight", ".scales"
        ) not in out
    bias = out[_TINY_PREFIX + "mlp.expert_bias"]
    assert bias.dtype == mx.float32


def test_quantized_residents_roundtrip_close_to_source() -> None:
    tensors = _tiny_resident_tensors()
    out = MOD.quantize_resident_tensors(tensors)
    for base, bits in (("self_attn.q_proj", 4), ("mlp.router.gate", 8)):
        orig = tensors[_TINY_PREFIX + base + ".weight"].astype(mx.float32).flatten()
        deq = mx.dequantize(
            out[_TINY_PREFIX + base + ".weight"],
            out[_TINY_PREFIX + base + ".scales"],
            out[_TINY_PREFIX + base + ".biases"],
            group_size=64,
            bits=bits,
            mode="affine",
        ).astype(mx.float32).flatten()
        cos = (
            mx.sum(orig * deq) / (mx.linalg.norm(orig) * mx.linalg.norm(deq))
        ).item()
        assert cos > (0.99 if bits == 4 else 0.999)


def test_rejects_routed_expert_tensors() -> None:
    tensors = _tiny_resident_tensors()
    tensors[_TINY_PREFIX + "mlp.experts.0.gate_proj.weight"] = mx.zeros(
        (64, 64), dtype=mx.bfloat16
    )
    with pytest.raises(ValueError, match="unexpected resident"):
        MOD.quantize_resident_tensors(tensors)
    with pytest.raises(ValueError, match="routed expert"):
        MOD.classify_resident(_TINY_PREFIX + "mlp.experts.0.gate_proj.weight")


def test_rejects_incomplete_or_unknown_resident_sets() -> None:
    tensors = _tiny_resident_tensors()
    del tensors[_TINY_PREFIX + "mlp.router.gate.weight"]
    with pytest.raises(ValueError, match="missing resident"):
        MOD.quantize_resident_tensors(tensors)

    tensors = _tiny_resident_tensors()
    tensors[_TINY_PREFIX + "mystery.weight"] = mx.zeros((4,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="unexpected resident"):
        MOD.quantize_resident_tensors(tensors)


def test_rejects_wrong_pass_through_dtype() -> None:
    tensors = _tiny_resident_tensors()
    tensors[_TINY_PREFIX + "mlp.expert_bias"] = mx.zeros((4,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="must stay"):
        MOD.quantize_resident_tensors(tensors)

    tensors = _tiny_resident_tensors()
    tensors[_TINY_PREFIX + "self_attn.q_proj.weight"] = mx.zeros(
        (64, 64), dtype=mx.float32
    )
    with pytest.raises(ValueError, match="bfloat16 source"):
        MOD.quantize_resident_tensors(tensors)
