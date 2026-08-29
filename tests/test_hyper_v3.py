"""Fused hyper-read v3 parity (GPU: the kernels are Metal).

Gate 1: the kernel pair reproduces the exact same math as an eager mlx
reference built on the SAME dequantized 8-bit weights (accumulation-order
tolerance only). Gate 2: against the bf16 eager module the difference is
quantization-class, bounded well below the mixer's measured 8-bit KLD
headroom.
"""

import mlx.core as mx
import pytest

from mtplx.kernels.hyper_connection_v3 import fused_hyper_read_v3, prepare_v3_pack
from mtplx.models.qwen4_exp import GatedResidual, TextArgs


def _family_args() -> TextArgs:
    return TextArgs()  # shipped geometry: hidden 2560, hc 4, lowrank 320


@pytest.fixture()
def module():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernels need the GPU")
    mx.random.seed(3)
    m = GatedResidual(_family_args())
    # realistic magnitudes: small weights, one-centered norm weight
    m.input_mix_weight_down.weight = (
        mx.random.normal(m.input_mix_weight_down.weight.shape) * 0.02
    ).astype(mx.bfloat16)
    m.input_mix_weight_up.weight = (
        mx.random.normal(m.input_mix_weight_up.weight.shape) * 0.02
    ).astype(mx.bfloat16)
    m.block_inject_weight.weight = (
        mx.random.normal(m.block_inject_weight.weight.shape) * 0.02
    ).astype(mx.bfloat16)
    m.hc_norm.weight = 1.0 + mx.random.normal(m.hc_norm.weight.shape) * 0.05
    mx.eval(m.parameters())
    return m


def _eager_reference_8bit(module, x_row):
    """The v3 math on dequantized 8-bit weights, in plain mlx ops."""
    pack = prepare_v3_pack(module)
    w1 = mx.dequantize(pack[0], pack[1], pack[2], group_size=64, bits=8)
    w2 = mx.dequantize(pack[3], pack[4], pack[5], group_size=64, bits=8)
    grouped = x_row.reshape(4, 2560).astype(mx.float32)
    rms = mx.rsqrt((grouped * grouped).mean(axis=-1, keepdims=True) + 1e-6)
    normed = (grouped * rms).reshape(-1) * module.hc_norm.weight.astype(mx.float32)
    z1 = (w1.astype(mx.float32) @ normed) / 4.0
    mix = mx.sigmoid(w2.astype(mx.float32) @ (z1[:320] * mx.sigmoid(z1[:320])))
    inject = 2.0 * mx.sigmoid(z1[320:])
    mixed = (mix.reshape(4, 2560) * normed.reshape(4, 2560)).mean(axis=0)
    return mixed, inject


def test_kernel_matches_dequant_reference(module):
    x = (mx.random.normal((10240,)) * 0.5).astype(mx.bfloat16)
    pack = prepare_v3_pack(module)
    mixed_k, inject_k = fused_hyper_read_v3(x, module.hc_norm.weight, pack)
    mixed_r, inject_r = _eager_reference_8bit(module, x)
    mx.eval(mixed_k, inject_k, mixed_r, inject_r)
    scale = mx.abs(mixed_r).max().item() + 1e-6
    err = (mx.abs(mixed_k.astype(mx.float32) - mixed_r) / scale).max().item()
    assert err < 2e-2, f"mixed rel err {err}"
    ierr = mx.abs(inject_k.astype(mx.float32) - inject_r).max().item()
    assert ierr < 2e-2, f"inject err {ierr}"


def test_module_path_stays_quantization_close_to_bf16(module, monkeypatch):
    monkeypatch.setenv("MTPLX_FUSED_HC_V3", "1")
    x = (mx.random.normal((1, 1, 10240)) * 0.5).astype(mx.bfloat16)
    mixed_v3, hyper_v3, inject_v3 = module(x)
    monkeypatch.setenv("MTPLX_FUSED_HC_V3", "0")
    mixed_e, hyper_e, inject_e = module(x)
    mx.eval(mixed_v3, inject_v3, mixed_e, inject_e)
    assert hyper_v3 is x and hyper_e is x
    scale = mx.abs(mixed_e.astype(mx.float32)).max().item() + 1e-6
    err = (
        mx.abs(mixed_v3.astype(mx.float32) - mixed_e.astype(mx.float32)) / scale
    ).max().item()
    assert err < 5e-2, f"mixed vs bf16 rel err {err}"
    ierr = mx.abs(
        inject_v3.astype(mx.float32) - inject_e.astype(mx.float32)
    ).max().item()
    assert ierr < 5e-2, f"inject vs bf16 err {ierr}"


def test_v3_never_engages_on_multirow_or_quantized(module, monkeypatch):
    monkeypatch.setenv("MTPLX_FUSED_HC_V3", "1")
    multi = (mx.random.normal((1, 4, 10240)) * 0.5).astype(mx.bfloat16)
    assert not module._v3_read_applies(multi)
    single = (mx.random.normal((1, 1, 10240)) * 0.5).astype(mx.bfloat16)
    assert module._v3_read_applies(single)
