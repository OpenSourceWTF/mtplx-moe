"""Exactness tests for the dequant-in-flight q8/q4 packed GQA kernel.

Contract: identical math to (dequantize_symmetric -> fp32 attention) on the
same quantized rows. The comparator is the bf16 packed kernel run on the
dequantized rows — same topology, so the quant kernel must land at least as
close to the fp32 reference (it skips the bf16 round-trip the comparator pays).
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric  # noqa: E402
from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail  # noqa: E402
from mtplx.kernels.sdpa_gqa_packed_quant import (  # noqa: E402
    sdpa_gqa_packed_tail_quant,
)

METAL = mx.metal.is_available()

HQ, HK, D = 24, 4, 256
SCALE = D**-0.5


def _ref_tail_causal(q, k, v, scale):
    qf = q.astype(mx.float32)
    kf = mx.repeat(k.astype(mx.float32), qf.shape[1] // k.shape[1], axis=1)
    vf = mx.repeat(v.astype(mx.float32), qf.shape[1] // v.shape[1], axis=1)
    n_kv = kf.shape[2]
    q_len = qf.shape[2]
    scores = (qf * scale) @ kf.transpose(0, 1, 3, 2)
    q_pos = mx.arange(n_kv - q_len, n_kv)[:, None]
    k_pos = mx.arange(n_kv)[None, :]
    mask = (k_pos <= q_pos)[None, None]
    scores = mx.where(mask, scores, mx.full(scores.shape, -1e30))
    return mx.softmax(scores, axis=-1) @ vf


@pytest.mark.skipif(not METAL, reason="requires Metal")
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("q_len", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("offset", [515, 2051])
def test_quant_matches_dequantized_fp32_reference(bits, q_len, offset):
    mx.random.seed(offset * 100 + q_len * 10 + bits)
    capacity = offset + 173
    q = mx.random.normal((1, HQ, q_len, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    mx.eval(q, keys, values)

    k_q, k_scale = quantize_symmetric(keys, bits=bits)
    v_q, v_scale = quantize_symmetric(values, bits=bits)
    mx.eval(k_q, k_scale, v_q, v_scale)

    # The ground truth the kernel must reproduce: fp32 attention over the
    # DEQUANTIZED rows (the quantization error itself is the quality lane's
    # business, not the kernel's).
    k_deq = dequantize_symmetric(k_q, k_scale, bits=bits, head_dim=D)
    v_deq = dequantize_symmetric(v_q, v_scale, bits=bits, head_dim=D)
    ref = _ref_tail_causal(q, k_deq[..., :offset, :], v_deq[..., :offset, :], SCALE)

    out = sdpa_gqa_packed_tail_quant(
        queries=q,
        k_q=k_q,
        k_scale=k_scale,
        v_q=v_q,
        v_scale=v_scale,
        offset=offset,
        scale=SCALE,
        bits=bits,
    )
    assert out is not None, "quant kernel bailed"
    my_diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item())

    if q_len >= 2:
        # Same-topology comparator: bf16 kernel on the dequantized rows pays
        # an extra bf16 round-trip; the quant kernel must not do worse.
        k_bf = k_deq.astype(mx.bfloat16)
        v_bf = v_deq.astype(mx.bfloat16)
        mx.eval(k_bf, v_bf)
        comp = sdpa_gqa_packed_tail(
            queries=q, keys=k_bf, values=v_bf, offset=offset, scale=SCALE
        )
        assert comp is not None
        comp_diff = float(mx.max(mx.abs(comp.astype(mx.float32) - ref)).item())
        assert my_diff <= comp_diff + 1e-3, (my_diff, comp_diff)
    else:
        assert my_diff < 2e-2, my_diff


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_quant_contract_bails():
    q = mx.random.normal((1, HQ, 4, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, HK, 600, D)).astype(mx.bfloat16)
    k_q, k_scale = quantize_symmetric(keys, bits=8)
    # wrong bits
    assert sdpa_gqa_packed_tail_quant(
        queries=q, k_q=k_q, k_scale=k_scale, v_q=k_q, v_scale=k_scale,
        offset=500, scale=SCALE, bits=2,
    ) is None
    # scale dtype contract
    assert sdpa_gqa_packed_tail_quant(
        queries=q, k_q=k_q, k_scale=k_scale.astype(mx.float16), v_q=k_q,
        v_scale=k_scale, offset=500, scale=SCALE, bits=8,
    ) is None
