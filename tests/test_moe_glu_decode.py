"""Fused routed-expert GLU decode parity (GPU: the kernels are Metal).

The kernel pair must reproduce the stock chain's math (dequant -> gate/up ->
silu*mul -> down -> router-weighted sum) on the SAME 4-bit affine packs, at
both shipped forge group sizes: g32 (bring-up forge, 2026-08-26) and g64
(the 2026-08-27 01:35 reforge that silently disarmed the g32-hardcoded first
cut). Tolerance is accumulation-order class only — the kernel casts through
the IO dtype mid-chain exactly like the stock path's bf16 intermediates.
"""

import mlx.core as mx
import pytest

from mtplx.kernels.moe_glu_decode import moe_glu_decode

K = 2560  # kernel constexpr: family hidden size
E = 4
N_INTER = 64
DMODEL = 64
TOPK = 2


def _quant_stack(w, gs):
    qs, ss, bs = [], [], []
    for e in range(w.shape[0]):
        q, s, b = mx.quantize(w[e], group_size=gs, bits=4)
        qs.append(q)
        ss.append(s)
        bs.append(b)
    return mx.stack(qs), mx.stack(ss), mx.stack(bs)


def _dequant_stack(q, s, b, gs):
    return mx.stack(
        [mx.dequantize(q[e], s[e], b[e], group_size=gs, bits=4) for e in range(q.shape[0])]
    )


@pytest.fixture(params=[32, 64], ids=["g32", "g64"])
def packs(request):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernels need the GPU")
    gs = request.param
    mx.random.seed(11 + gs)
    gu = (mx.random.normal((E, 2 * N_INTER, K)) * 0.05).astype(mx.bfloat16)
    dn = (mx.random.normal((E, DMODEL, N_INTER)) * 0.05).astype(mx.bfloat16)
    gu_q = _quant_stack(gu, gs)
    dn_q = _quant_stack(dn, gs)
    x = (mx.random.normal((K,)) * 0.5).astype(mx.bfloat16)
    experts = mx.array([1, 3], dtype=mx.uint32)
    route_w = mx.array([0.7, 0.3], dtype=mx.float32)
    mx.eval(x, *gu_q, *dn_q)
    return gs, x, gu_q, dn_q, experts, route_w


def _reference(gs, x, gu_q, dn_q, experts, route_w):
    gu = _dequant_stack(*gu_q, gs).astype(mx.float32)
    dn = _dequant_stack(*dn_q, gs).astype(mx.float32)
    xf = x.astype(mx.float32)
    y = mx.zeros((DMODEL,), dtype=mx.float32)
    for slot in range(TOPK):
        e = int(experts[slot])
        z = gu[e] @ xf
        gate, up = z[:N_INTER], z[N_INTER:]
        h = (gate * mx.sigmoid(gate)) * up
        y = y + float(route_w[slot]) * (dn[e] @ h)
    return y


def test_kernel_matches_dequant_reference(packs):
    gs, x, gu_q, dn_q, experts, route_w = packs
    y_k = moe_glu_decode(
        x, *gu_q, *dn_q, experts, route_w,
        gu_group_size=gs, dn_group_size=gs,
    )
    y_r = _reference(gs, x, gu_q, dn_q, experts, route_w)
    mx.eval(y_k, y_r)
    scale = mx.abs(y_r).max().item() + 1e-6
    err = (mx.abs(y_k.astype(mx.float32) - y_r) / scale).max().item()
    assert err < 2e-2, f"g{gs} rel err {err}"


def test_rejects_indivisible_group_size(packs):
    gs, x, gu_q, dn_q, experts, route_w = packs
    with pytest.raises(ValueError):
        moe_glu_decode(
            x, *gu_q, *dn_q, experts, route_w,
            gu_group_size=48, dn_group_size=gs,
        )
