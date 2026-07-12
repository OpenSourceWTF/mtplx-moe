"""Exactness + contract tests for the packed-row GQA verify kernel (Lane A)."""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.attention_split import configure_split_full_attention  # noqa: E402
from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail  # noqa: E402

METAL = mx.metal.is_available()

HQ, HK, D = 24, 4, 256
GQA = HQ // HK
SCALE = D**-0.5


def _ref_tail_causal(q, k, v, scale):
    """fp32 reference with tail-causal semantics over the live rows."""

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


def _fused_reference_tolerance(q, k_live, v_live, ref):
    out_fused = mx.fast.scaled_dot_product_attention(
        q, k_live, v_live, scale=SCALE, mask="causal"
    )
    return float(mx.max(mx.abs(out_fused.astype(mx.float32) - ref)).item())


@pytest.mark.skipif(not METAL, reason="requires Metal")
@pytest.mark.parametrize("q_len", [2, 3, 4])
@pytest.mark.parametrize("offset", [515, 2048, 2051])
def test_packed_matches_fp32_reference_with_capacity_padding(q_len, offset):
    mx.random.seed(offset * 10 + q_len)
    capacity = offset + 256  # capacity-padded buffer, live rows = offset
    q = mx.random.normal((1, HQ, q_len, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    mx.eval(q, keys, values)

    k_live = keys[..., :offset, :]
    v_live = values[..., :offset, :]
    ref = _ref_tail_causal(q, k_live, v_live, SCALE)
    fused_diff = _fused_reference_tolerance(q, k_live, v_live, ref)

    out = sdpa_gqa_packed_tail(
        queries=q,
        keys=keys,
        values=values,
        offset=offset,
        scale=SCALE,
    )
    assert out is not None
    diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item())
    # Same numeric class as the stock fused kernel (bf16 accumulation
    # ordering differences only).
    assert diff <= max(5e-3, 4.0 * fused_diff)


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_packed_array_offset_matches_int_offset():
    mx.random.seed(99)
    offset = 3072
    capacity = offset + 512
    q = mx.random.normal((1, HQ, 4, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    mx.eval(q, keys, values)

    out_int = sdpa_gqa_packed_tail(
        queries=q, keys=keys, values=values, offset=offset, scale=SCALE
    )
    out_arr = sdpa_gqa_packed_tail(
        queries=q,
        keys=keys,
        values=values,
        offset=mx.array(offset, dtype=mx.int32),
        scale=SCALE,
    )
    assert out_int is not None and out_arr is not None
    assert float(mx.max(mx.abs(out_int - out_arr)).item()) == 0.0


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_packed_ignores_garbage_beyond_offset():
    mx.random.seed(7)
    offset = 1024
    capacity = offset + 256
    q = mx.random.normal((1, HQ, 4, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, capacity, D)).astype(mx.bfloat16)
    mx.eval(q, keys, values)
    out_a = sdpa_gqa_packed_tail(
        queries=q, keys=keys, values=values, offset=offset, scale=SCALE
    )

    # Poison the dead rows; the output must be bit-identical.
    poison_k = mx.concatenate(
        [
            keys[..., :offset, :],
            mx.full((1, HK, capacity - offset, D), 1e4, dtype=keys.dtype),
        ],
        axis=2,
    )
    poison_v = mx.concatenate(
        [
            values[..., :offset, :],
            mx.full((1, HK, capacity - offset, D), -1e4, dtype=values.dtype),
        ],
        axis=2,
    )
    poison_k = mx.contiguous(poison_k)
    poison_v = mx.contiguous(poison_v)
    mx.eval(poison_k, poison_v)
    out_b = sdpa_gqa_packed_tail(
        queries=q, keys=poison_k, values=poison_v, offset=offset, scale=SCALE
    )
    assert out_a is not None and out_b is not None
    assert float(mx.max(mx.abs(out_a - out_b)).item()) == 0.0


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_packed_bails_out_of_contract():
    q4 = mx.random.normal((1, HQ, 4, D)).astype(mx.bfloat16)
    keys = mx.random.normal((1, HK, 1024, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, HK, 1024, D)).astype(mx.bfloat16)
    mx.eval(q4, keys, values)

    # q_len outside 2..4
    q1 = mx.random.normal((1, HQ, 1, D)).astype(mx.bfloat16)
    q5 = mx.random.normal((1, HQ, 5, D)).astype(mx.bfloat16)
    mx.eval(q1, q5)
    assert (
        sdpa_gqa_packed_tail(
            queries=q1, keys=keys, values=values, offset=512, scale=SCALE
        )
        is None
    )
    assert (
        sdpa_gqa_packed_tail(
            queries=q5, keys=keys, values=values, offset=512, scale=SCALE
        )
        is None
    )
    # fp32 inputs unsupported
    assert (
        sdpa_gqa_packed_tail(
            queries=q4.astype(mx.float32),
            keys=keys.astype(mx.float32),
            values=values.astype(mx.float32),
            offset=512,
            scale=SCALE,
        )
        is None
    )
    # offset out of range
    assert (
        sdpa_gqa_packed_tail(
            queries=q4, keys=keys, values=values, offset=0, scale=SCALE
        )
        is None
    )
    assert (
        sdpa_gqa_packed_tail(
            queries=q4, keys=keys, values=values, offset=4096, scale=SCALE
        )
        is None
    )


class _FakeAttn:
    def __init__(self):
        self.q_proj = None
        self.q_norm = None


def test_configure_reads_gqa_packed_env(monkeypatch):
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA_THRESHOLD", "12345")

    class _Layer:
        is_linear = False

        def __init__(self):
            self.self_attn = _FakeAttn()

    class _Inner:
        def __init__(self):
            self.layers = [_Layer()]

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()
    stats = configure_split_full_attention(model)
    assert stats["gqa_packed_sdpa_enabled"] is True
    assert stats["gqa_packed_sdpa_threshold"] == 12345
    attn = model.model.layers[0].self_attn
    assert attn._mtplx_gqa_packed_sdpa_enabled is True
    assert attn._mtplx_gqa_packed_sdpa_threshold == 12345
    assert attn._mtplx_split_full_attention_enabled is True


def test_configure_defaults_gqa_packed_off(monkeypatch):
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA_THRESHOLD", raising=False)
    # os.environ may carry profile state from the harness; be explicit.
    assert os.environ.get("MTPLX_GQA_PACKED_SDPA") is None

    class _Layer:
        is_linear = False

        def __init__(self):
            self.self_attn = _FakeAttn()

    class _Inner:
        def __init__(self):
            self.layers = [_Layer()]

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()
    stats = configure_split_full_attention(model)
    assert stats["gqa_packed_sdpa_enabled"] is False
    assert stats["gqa_packed_sdpa_threshold"] == 8192
