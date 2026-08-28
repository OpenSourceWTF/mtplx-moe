"""QSA decode gather lane parity (GPU: Metal).

Past the indexer's engage threshold (T > budget), the gather lane
(MTPLX_QSA_GATHER) must produce the same attention output as the dense
bool-mask lane: identical visible set, so the softmax differs only by
reduction-size bf16 noise. Exercised through a real Attention + QSACache
prefill long enough to engage the indexer, then several decode steps.
Below the threshold the indexer returns None on both lanes (dense == sparse
regime) and the gather lane must not activate.
"""

import mlx.core as mx
import pytest

import mtplx.models.qwen4_exp as q4
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs


@pytest.fixture()
def attn():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(23)
    layer = Attention(TextArgs())
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def _run(layer, prefill, decodes):
    cache = QSACache(compress_ratio=layer.indexer.ratio)
    out_p = layer(prefill, cache)
    outs = [layer(d, cache) for d in decodes]
    mx.eval(out_p, *outs)
    return outs


def test_gather_parity_past_engage_threshold(attn, monkeypatch):
    ratio = attn.indexer.ratio
    engage_t = attn.indexer.block_topk * ratio  # budget tokens
    T0 = engage_t + 8 * ratio  # comfortably past the dense==sparse regime
    mx.random.seed(31)
    prefill = (mx.random.normal((1, T0, 2560)) * 0.3).astype(mx.bfloat16)
    decodes = [
        (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16) for _ in range(3)
    ]

    monkeypatch.setenv("MTPLX_QSA_GATHER", "0")
    ref = _run(attn, prefill, decodes)

    calls = {"take": 0}
    orig_take = mx.take

    def counting_take(*a, **k):
        calls["take"] += 1
        return orig_take(*a, **k)

    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setattr(mx, "take", counting_take)
    got = _run(attn, prefill, decodes)
    monkeypatch.setattr(mx, "take", orig_take)
    assert calls["take"] >= 2 * len(decodes), "gather lane did not run — vacuous"

    for i, (r, g) in enumerate(zip(ref, got)):
        scale = mx.abs(r.astype(mx.float32)).max().item() + 1e-6
        err = (
            mx.abs(g.astype(mx.float32) - r.astype(mx.float32)) / scale
        ).max().item()
        assert err < 2e-2, f"decode {i} rel err {err}"


def test_gather_inactive_below_threshold(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    mx.random.seed(37)
    prefill = (mx.random.normal((1, 64, 2560)) * 0.3).astype(mx.bfloat16)
    step = (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16)
    cache = QSACache(compress_ratio=attn.indexer.ratio)
    p = attn(prefill, cache)
    sel = attn.indexer(step, cache.offset, cache)
    assert sel is None, "below the engage threshold the indexer must stay dense"
    mx.eval(p)
