"""Fused GDN output kernel parity (GPU: Metal).

The kernel must reproduce SigmoidRMSNormGated + QuantizedLinear on the SAME
4-bit pack, at both shipped forge group sizes, within accumulation-order
tolerance (the kernel casts the gated-normed values through bf16 exactly at
the module boundary the eager chain has)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.kernels.gdn_out_fused import fused_gdn_out
from mtplx.models.qwen4_exp import SigmoidRMSNormGated

NH, HD = 48, 128
K = NH * HD
DMODEL = 2560


@pytest.fixture(params=[32, 64], ids=["g32", "g64"])
def setup(request):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    gs = request.param
    mx.random.seed(23 + gs)
    w = (mx.random.normal((DMODEL, K)) * 0.03).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(w, group_size=gs, bits=4)
    x = (mx.random.normal((K,)) * 0.7).astype(mx.bfloat16)
    z = (mx.random.normal((K,)) * 0.8).astype(mx.bfloat16)
    norm = SigmoidRMSNormGated(HD)
    norm.weight = 1.0 + mx.random.normal((HD,)) * 0.05
    mx.eval(qw, qs, qb, x, z, norm.weight)
    return gs, x, z, norm, (qw, qs, qb)


def _reference(gs, x, z, norm, pack):
    qw, qs, qb = pack
    v = norm(x.reshape(NH, HD), z.reshape(NH, HD)).reshape(K)
    wd = mx.dequantize(qw, qs, qb, group_size=gs, bits=4).astype(mx.float32)
    return wd @ v.astype(mx.float32)


def test_kernel_matches_module_chain(setup):
    gs, x, z, norm, pack = setup
    y_k = fused_gdn_out(x, z, norm.weight.astype(mx.float32), *pack, group_size=gs)
    y_r = _reference(gs, x, z, norm, pack)
    mx.eval(y_k, y_r)
    scale = mx.abs(y_r).max().item() + 1e-6
    err = (mx.abs(y_k.astype(mx.float32) - y_r) / scale).max().item()
    assert err < 2e-2, f"g{gs} rel err {err}"


def test_rejects_bad_group_size(setup):
    gs, x, z, norm, pack = setup
    with pytest.raises(ValueError):
        fused_gdn_out(x, z, norm.weight.astype(mx.float32), *pack, group_size=80)
