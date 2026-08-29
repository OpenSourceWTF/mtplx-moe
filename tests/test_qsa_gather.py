"""QSA decode gather lane parity (GPU: Metal).

Past the indexer's engage threshold (T > budget), the S=1 gather lane
(MTPLX_QSA_GATHER_DECODE — its own dormant opt-in, FALSIFIED d6171d2c at
-5.25%, deliberately NOT armed by the MTPLX_QSA_GATHER rows-gather family
default) must produce the same attention output as the dense bool-mask
lane: identical visible set, so the softmax differs only by reduction-size
bf16 noise. Exercised through a real Attention + QSACache prefill long
enough to engage the indexer, then several decode steps. Below the
threshold the indexer returns None on both lanes (dense == sparse regime)
and the gather lane must not activate.
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

    monkeypatch.setenv("MTPLX_QSA_GATHER_DECODE", "0")
    ref = _run(attn, prefill, decodes)

    calls = {"take": 0}
    orig_take = mx.take

    def counting_take(*a, **k):
        calls["take"] += 1
        return orig_take(*a, **k)

    monkeypatch.setenv("MTPLX_QSA_GATHER_DECODE", "1")
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
    monkeypatch.setenv("MTPLX_QSA_GATHER_DECODE", "1")
    mx.random.seed(37)
    prefill = (mx.random.normal((1, 64, 2560)) * 0.3).astype(mx.bfloat16)
    step = (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16)
    cache = QSACache(compress_ratio=attn.indexer.ratio)
    p = attn(prefill, cache)
    sel = attn.indexer(step, cache.offset, cache)
    assert sel is None, "below the engage threshold the indexer must stay dense"
    mx.eval(p)


def test_family_default_env_never_routes_s1_gather(attn, monkeypatch):
    """The shipped family default (MTPLX_QSA_GATHER=1, the S>1 rows lane)
    must never re-arm the falsified S=1 decode lane (d6171d2c: clean A/B/A
    50.16 / 46.19 / 47.33 -> -5.25%). Engaged regime, one decode row: the
    indexer must return the dense bool mask, never 1-D token indices."""
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.delenv("MTPLX_QSA_GATHER_DECODE", raising=False)
    monkeypatch.delenv("MTPLX_QSA_FLASH", raising=False)
    mx.random.seed(43)
    prefill = (mx.random.normal((1, 8192, 2560)) * 0.3).astype(mx.bfloat16)
    step = (mx.random.normal((1, 1, 2560)) * 0.3).astype(mx.bfloat16)
    cache = QSACache(compress_ratio=attn.indexer.ratio)
    mx.eval(attn(prefill, cache))
    sel = attn.indexer(step, cache.offset, cache)
    assert sel is not None, "T=8192 must be past the engage threshold"
    assert isinstance(sel, mx.array) and sel.ndim == 4, (
        "S=1 under the family default must stay on the dense mask — a 1-D "
        "token-index return means the falsified decode-gather lane re-armed"
    )


def test_qsa_gather_decode_env_defaults_off(monkeypatch):
    """The S=1 decode-gather opt-in is dormant by default, and the
    rows-gather family key must not arm it (that conflation once shipped
    the falsified lane as a silent family default)."""
    monkeypatch.delenv("MTPLX_QSA_GATHER_DECODE", raising=False)
    monkeypatch.delenv("MTPLX_QSA_GATHER", raising=False)
    assert q4._qsa_gather_decode_enabled() is False
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    assert q4._qsa_gather_decode_enabled() is False
    monkeypatch.setenv("MTPLX_QSA_GATHER_DECODE", "1")
    assert q4._qsa_gather_decode_enabled() is True
