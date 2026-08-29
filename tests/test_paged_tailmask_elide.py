"""The dynamic-offset paged kernel's built-in visibility equals the
capacity-wide tail-causal bool mask the TensorOffset paged adapter emits.

This is the equivalence MTPLX_PAGED_TAILMASK_ELIDE relies on (the kernel
refuses array masks outright — the 147.4k verify decline, receipts
MEASUREMENTS 2026-08-26 12:58): eliding the standard tail mask must be a
no-op mathematically.
"""

import mlx.core as mx
import pytest

from mtplx.kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail_dynamic_offset

HK, GQA, D, BLOCK = 4, 6, 256, 16
HQ = HK * GQA


def _pool_from_contiguous(x, blocks):
    # (1, hk, T, d) -> (blocks, block, hk, d) with sequential block layout
    t = int(x.shape[2])
    pad = blocks * BLOCK - t
    if pad:
        x = mx.concatenate(
            [x, mx.zeros((1, HK, pad, D), dtype=x.dtype)], axis=2
        )
    return x[0].transpose(1, 0, 2).reshape(blocks, BLOCK, HK, D)


def _tail_mask(offset_pre, q_len, capacity):
    # Exactly TensorOffsetVllmMetalPagedKVCache.make_mask: row j sees keys
    # <= offset_pre + j (offset_pre = offset BEFORE the q_len rows landed).
    rinds = mx.arange(capacity)
    linds = offset_pre + mx.arange(q_len)
    return linds[:, None] >= rinds[None, :]


@pytest.mark.parametrize("offset_total,q_len", [(37, 4), (64, 4), (97, 5)])
def test_kernel_builtin_visibility_matches_adapter_tail_mask(offset_total, q_len):
    blocks = 8
    capacity = blocks * BLOCK
    assert offset_total <= capacity
    mx.random.seed(offset_total * 131 + q_len)
    keys = mx.random.normal((1, HK, offset_total, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, offset_total, D)).astype(mx.bfloat16)
    queries = mx.random.normal((1, HQ, q_len, D)).astype(mx.bfloat16)
    scale = D ** -0.5

    out = sdpa_2pass_paged_tail_dynamic_offset(
        queries=queries,
        key_cache=_pool_from_contiguous(keys, blocks),
        value_cache=_pool_from_contiguous(values, blocks),
        offset=mx.array([offset_total], dtype=mx.int32),
        block_size=BLOCK,
        scale=scale,
        mask=None,
        max_q_len=16,
        max_offset=None,
    )
    assert out is not None, "kernel declined the elide-shape call"

    # fp32 reference over the contiguous live prefix with the adapter's mask
    offset_pre = offset_total - q_len
    mask = _tail_mask(offset_pre, q_len, offset_total)
    q32 = queries.astype(mx.float32)
    k32 = mx.repeat(keys.astype(mx.float32), GQA, axis=1)
    v32 = mx.repeat(values.astype(mx.float32), GQA, axis=1)
    scores = (q32 @ k32.transpose(0, 1, 3, 2)) * scale
    scores = mx.where(mask[None, None, :, :], scores, mx.array(-1e30))
    ref = mx.softmax(scores, axis=-1) @ v32

    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err <= 2e-2, f"elide equivalence broken: max err {err}"


def test_kernel_declines_past_threadgroup_ceiling():
    # 32 x GQA(6) x QL(8) = 1536 > 1024 Metal threads: the wrapper must
    # decline, not raise (the launch used to throw ValueError).
    blocks = 8
    t = 128
    keys = mx.random.normal((1, HK, t, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, t, D)).astype(mx.bfloat16)
    queries = mx.random.normal((1, HQ, 8, D)).astype(mx.bfloat16)
    out = sdpa_2pass_paged_tail_dynamic_offset(
        queries=queries,
        key_cache=_pool_from_contiguous(keys, blocks),
        value_cache=_pool_from_contiguous(values, blocks),
        offset=mx.array([t], dtype=mx.int32),
        block_size=BLOCK,
        scale=D ** -0.5,
        mask=None,
        max_q_len=16,
        max_offset=None,
    )
    assert out is None
