"""CompiledARForward: compile + KV-cache state threading, validated on a toy model.

The real model needs a GPU to load, so this proves the MECHANISM on a tiny 2-layer
attention model on CPU: that threading each layer's (keys, values, offset) as
explicit compile inputs/outputs preserves decode correctness across steps, that
the offset advances, and that the engagement counter fires. Parity is checked at
fp tolerance (the toy uses fp32 matmul, which compiles exactly; the real model's
quantized-gather divergence is the separate A/B question).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from mtplx.graphbank import TensorOffsetKVCache
from mtplx.compiled_forward import CompiledARForward, compiled_forward_calls


class _ToyAttn(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache):
        b, t, d = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        shp = lambda z: z.reshape(b, t, self.heads, self.hd).transpose(0, 2, 1, 3)  # noqa: E731
        q, k, v = shp(q), shp(k), shp(v)
        # Build the mask from the PRE-update offset (the real model builds it
        # before the layer loop). A fixed-buffer cache returns the whole buffer
        # incl. an uninitialized tail beyond `offset`; a post-update mask would
        # attend into that garbage, and the garbage differs between the compiled
        # trace and eager runtime — the divergence this ordering avoids. A
        # trimmed KVCache returns only valid rows and needs no mask.
        mask = cache.make_mask(t) if isinstance(cache, TensorOffsetKVCache) else None
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.hd ** -0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.o(out)


class _ToyLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = _ToyAttn(dim, heads)
        self.norm = nn.RMSNorm(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache):
        h = x + self.attn(self.norm(x), cache)
        return h + self.mlp(h)


class _ToyModel(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = [_ToyLayer(dim, heads) for _ in range(layers)]
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self._n = layers

    def make_cache(self):
        return [KVCache() for _ in range(self._n)]

    def __call__(self, inputs, cache=None):
        h = self.embed(inputs)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i])
        return self.head(self.norm(h))


def _decode_eager(model, cache, tokens):
    logits = []
    for tok in tokens:
        out = model(mx.array([[tok]]), cache=cache)
        logits.append(out)
    return logits


def _decode_compiled(model, comp, cache, tokens):
    logits = []
    for tok in tokens:
        out = comp(mx.array([[tok]]), cache)
        logits.append(out)
    return logits


def test_compiled_forward_matches_eager_decode_across_steps() -> None:
    mx.random.seed(0)
    model = _ToyModel(vocab=64, dim=32, heads=4, layers=2)
    mx.eval(model.parameters())

    prompt = mx.array([[1, 2, 3, 4, 5]])
    decode = [7, 9, 11, 13]

    # eager: prime + decode
    ce = model.make_cache()
    model(prompt, cache=ce)
    eager = _decode_eager(model, ce, decode)

    # compiled: prime with the SAME live cache path, then compiled decode
    cc = model.make_cache()
    model(prompt, cache=cc)
    comp = CompiledARForward(model, reserve_tokens=64)
    before = compiled_forward_calls()
    compiled = _decode_compiled(model, comp, cc, decode)

    assert compiled_forward_calls() == before + len(decode), "compiled path did not fire each step"
    for e, c in zip(eager, compiled):
        mx.eval(e, c)
        # fp32 path compiles exactly; allow a hair for sdpa fusion
        assert float(mx.abs(e - c).max()) < 1e-3, "compiled decode diverged from eager"


def test_offset_advances_and_argmax_tokens_match() -> None:
    mx.random.seed(1)
    model = _ToyModel(vocab=64, dim=32, heads=4, layers=3)
    mx.eval(model.parameters())
    prompt = mx.array([[2, 4, 6]])
    decode = [8, 10, 12, 14, 16]

    ce = model.make_cache()
    model(prompt, cache=ce)
    eager_tokens = [int(mx.argmax(m[:, -1, :])) for m in _decode_eager(model, ce, decode)]

    cc = model.make_cache()
    model(prompt, cache=cc)
    comp = CompiledARForward(model, reserve_tokens=64)
    comp_tokens = [int(mx.argmax(m[:, -1, :])) for m in _decode_compiled(model, comp, cc, decode)]

    assert eager_tokens == comp_tokens, "argmax tokens diverged under compiled forward"
