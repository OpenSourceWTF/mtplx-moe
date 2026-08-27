"""QSA attention qkv fusion parity (CPU exact).

The fused shared-input GEMV (q/k/v + indexer qk in one call) must be
bit-exact against the four separate projections through the FULL attention
module — including the indexer's selection mask past the engage threshold,
where a desynced projection would change block selection (the crash class
tests/test_qwen4_exp_qsa_cache.py pins)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs, _FusedGDNInProj


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


_PROJS = ("q_proj", "k_proj", "v_proj")


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(3)
    layer = Attention(_tiny_args())
    for name in _PROJS:
        lin = getattr(layer, name)
        lin.weight = mx.random.normal(lin.weight.shape) * 0.05
        setattr(layer, name, nn.QuantizedLinear.from_linear(lin, group_size=32, bits=4))
    ilin = layer.indexer.index_qk_proj
    ilin.weight = mx.random.normal(ilin.weight.shape) * 0.05
    layer.indexer.index_qk_proj = nn.QuantizedLinear.from_linear(
        ilin, group_size=32, bits=4
    )
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _attach_fused(layer, include_indexer=True):
    parts = [getattr(layer, n) for n in _PROJS]
    if include_indexer:
        parts.append(layer.indexer.index_qk_proj)
    rows = [p.weight.shape[0] for p in parts]
    splits = [sum(rows[: i + 1]) for i in range(len(rows) - 1)]
    layer.qkv_fused = _FusedGDNInProj(
        mx.concatenate([p.weight for p in parts], axis=0),
        mx.concatenate([p.scales for p in parts], axis=0),
        mx.concatenate([p.biases for p in parts], axis=0),
        32,
        4,
        "affine",
        splits,
    )


def _run(layer, chunks):
    cache = QSACache()
    outs = [layer(c, cache) for c in chunks]
    mx.eval(*outs)
    return outs


def test_fused_qkv_bit_exact_through_indexer_mask(attn):
    mx.random.seed(9)
    # prefill past the engage threshold (budget 8 / ratio 2 -> >8 visible),
    # then a decode-shaped step: the indexer mask is live in both runs.
    chunks = [
        mx.random.normal((1, 12, 64)).astype(mx.float32),
        mx.random.normal((1, 1, 64)).astype(mx.float32),
        mx.random.normal((1, 1, 64)).astype(mx.float32),
    ]
    ref = _run(attn, chunks)
    _attach_fused(attn)
    fused = _run(attn, chunks)
    for r, f in zip(ref, fused):
        assert r.shape == f.shape
        assert mx.array_equal(r, f).item()


def test_fused_qkv_without_indexer_member_bit_exact(attn):
    """The shipped forge keeps the indexer projection at 8-bit, so it stays
    out of the merge — the 3-way fused module plus the indexer's own
    dispatch must still be bit-exact."""
    mx.random.seed(13)
    chunks = [
        mx.random.normal((1, 12, 64)).astype(mx.float32),
        mx.random.normal((1, 1, 64)).astype(mx.float32),
    ]
    ref = _run(attn, chunks)
    _attach_fused(attn, include_indexer=False)
    fused = _run(attn, chunks)
    for r, f in zip(ref, fused):
        assert mx.array_equal(r, f).item()
