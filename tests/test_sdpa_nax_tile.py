"""Correctness gate for the TensorOps wide-M SDPA kernel (GPU required).

Compares sdpa_nax_tile against the production packed kernel and a fp32
reference at real Qwen3.8-27B decode shapes. Tolerances match the packed
kernel's own acceptance (bf16 output, fp32 accumulate).
"""

import math

import mlx.core as mx
import pytest

from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail
from mtplx.kernels.sdpa_nax_tile import sdpa_nax_tile

HQ, HKV, D = 24, 4, 256


def _ref_tail_causal(q, k, v, offset, scale, q_len):
    """fp32 reference: row j attends to n <= offset - q_len + j."""
    qf = q.astype(mx.float32)
    kf = k[:, :, :offset, :].astype(mx.float32)
    vf = v[:, :, :offset, :].astype(mx.float32)
    gqa = q.shape[1] // k.shape[1]
    kf = mx.repeat(kf, gqa, axis=1)
    vf = mx.repeat(vf, gqa, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    n = mx.arange(offset)[None, None, None, :]
    j = mx.arange(q_len)[None, None, :, None]
    mask = n <= (offset - q_len + j)
    s = mx.where(mask, s, mx.array(-1e30, dtype=s.dtype))
    p = mx.softmax(s, axis=-1)
    return p @ vf


@pytest.mark.parametrize("ctx,q_len", [
    (512, 4), (2048, 4), (4096, 4), (2048, 2), (2048, 5),
    (1023, 4),   # non-tile-aligned context
    (2048, 8),   # wide-QL (M=48 > 32 -> kernel must bail, not miscompute)
])
def test_nax_tile_matches_reference(ctx, q_len):
    mx.random.seed(7)
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)

    out = sdpa_nax_tile(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    if HQ // HKV * q_len > 32:
        assert out is None, "M>32 must bail to fallback"
        return
    assert out is not None, "kernel bailed on a supported shape"
    mx.eval(out)

    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()

    packed = sdpa_gqa_packed_tail(queries=q, keys=k, values=v, offset=ctx,
                                  scale=scale)
    if packed is not None:
        mx.eval(packed)
        perr = mx.abs(packed.astype(mx.float32) - ref).max().item()
        # NAX-tile must not be materially worse than the shipped kernel's
        # own bf16 deviation from the fp32 reference.
        assert err <= max(2.5 * perr, 0.02), (err, perr)
    assert err < 0.05, err


def test_nax_tile_empty_and_bail_paths():
    mx.random.seed(3)
    q = (mx.random.normal((1, HQ, 4, D))).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, 256, D))).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, 256, D))).astype(mx.bfloat16)
    mx.eval(q, k, v)
    assert sdpa_nax_tile(queries=q, keys=k, values=v, offset=0,
                         scale=0.0625) is None      # offset_range
    assert sdpa_nax_tile(queries=q.astype(mx.float32), keys=k, values=v,
                         offset=128, scale=0.0625) is None  # dtype gate
