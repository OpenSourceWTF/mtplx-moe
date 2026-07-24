"""Load-time trunk *_proj quantization (mtplx.proj_quant)."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.proj_quant import (
    ProjQuantError,
    proj_quant_covers,
    quantize_projections,
    requantize_projections,
)

H = 64


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, H, bias=False)
        self.down_proj = nn.Linear(H, H, bias=False)


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()
        self.router = _Router()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, H)
        self.layers = [_Layer() for _ in range(2)]
        self.lm_head = nn.Linear(H, 128, bias=False)


def test_scope_predicate():
    assert proj_quant_covers("model.layers.3.self_attn.q_proj")
    assert proj_quant_covers("model.layers.3.mlp.down_proj")
    assert proj_quant_covers("model.layers.3.mlp.shared_mlp.up_proj")
    assert not proj_quant_covers("model.layers.3.mlp.router.gate")
    assert not proj_quant_covers("lm_head")
    assert not proj_quant_covers("model.embed_tokens")


def test_quantize_projections_scopes_and_bits():
    model = _Model()
    touched = quantize_projections(model, "q4")
    # 2 layers x (2 attn + 2 mlp) projections
    assert len(touched) == 8
    layer = model.layers[0]
    assert isinstance(layer.self_attn.q_proj, nn.QuantizedLinear)
    assert layer.self_attn.q_proj.bits == 4
    assert layer.self_attn.q_proj.group_size == 64
    assert isinstance(layer.mlp.gate_proj, nn.QuantizedLinear)
    # router / embeddings / head untouched
    assert not isinstance(layer.router.gate, nn.QuantizedLinear)
    assert not isinstance(model.lm_head, nn.QuantizedLinear)


def test_quantize_rejects_bad_mode_and_empty_scope():
    with pytest.raises(ProjQuantError, match="mode must be one of"):
        quantize_projections(_Model(), "q2")

    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(H, 128, bias=False)

    with pytest.raises(ProjQuantError, match="matched no trunk"):
        quantize_projections(_Bare(), "q4")


def test_requantize_q8_to_q4_via_canonical_builder():
    model = _Model()
    quantize_projections(model, "q8")
    q8 = model.layers[0].self_attn.q_proj
    assert q8.bits == 8
    touched = requantize_projections(model, "q4")
    assert len(touched) == 8
    q4 = model.layers[0].self_attn.q_proj
    assert isinstance(q4, nn.QuantizedLinear) and q4.bits == 4
    # dequantized q4 stays close to the q8 dequantization it derived from
    a = mx.dequantize(q8.weight, q8.scales, q8.biases,
                      group_size=q8.group_size, bits=8).astype(mx.float32)
    b = mx.dequantize(q4.weight, q4.scales, q4.biases,
                      group_size=q4.group_size, bits=4).astype(mx.float32)
    cos = (a * b).sum() / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum()))
    assert cos.item() > 0.99
    # idempotence: nothing left above the target -> loud error, not silence
    with pytest.raises(ProjQuantError, match="matched no quantized trunk"):
        requantize_projections(model, "q4")
