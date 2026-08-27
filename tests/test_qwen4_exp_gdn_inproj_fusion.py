"""GDN in_proj fusion parity (CPU exact).

Row-axis concat of the four quantized input projections must be bit-exact
against the four separate matmuls: each output row's dot product and its
quant groups are untouched by the concat, so the fused GEMV + split is the
same arithmetic in one dispatch. Run on CPU — the exactness surface."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.models.qwen4_exp import GatedDeltaNet, _FusedGDNInProj, TextArgs


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=256,
        num_hidden_layers=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )


@pytest.fixture()
def gdn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(7)
    layer = GatedDeltaNet(_tiny_args())
    # quantize the four in_projs the way a forged pack ships them
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        lin = getattr(layer, name)
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(mx.bfloat16)
        setattr(
            layer,
            name,
            nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4),
        )
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _attach_fused(layer):
    parts = [getattr(layer, n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    rows = [p.weight.shape[0] for p in parts]
    splits = [rows[0], rows[0] + rows[1], rows[0] + rows[1] + rows[2]]
    layer.in_proj_fused = _FusedGDNInProj(
        mx.concatenate([p.weight for p in parts], axis=0),
        mx.concatenate([p.scales for p in parts], axis=0),
        mx.concatenate([p.biases for p in parts], axis=0),
        64,
        4,
        "affine",
        splits,
    )


def test_fused_projection_rows_bit_exact(gdn):
    x = (mx.random.normal((1, 3, 256)) * 0.5).astype(mx.bfloat16)
    _attach_fused(gdn)
    qkv_f, z_f, b_f, a_f = gdn.in_proj_fused(x)
    refs = [getattr(gdn, n)(x) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    for got, ref in zip((qkv_f, z_f, b_f, a_f), refs):
        assert got.shape == ref.shape
        assert mx.array_equal(got, ref).item()


def test_module_forward_identical_with_fusion(gdn):
    x = (mx.random.normal((1, 4, 256)) * 0.5).astype(mx.bfloat16)
    out_ref = gdn(x)
    _attach_fused(gdn)
    out_fused = gdn(x)
    mx.eval(out_ref, out_fused)
    assert mx.array_equal(out_fused, out_ref).item()
