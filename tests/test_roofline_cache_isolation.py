"""The roofline profiler must never advance the LIVE KV cache.

Regression test for a real defect. capture_attention registered a thunk closing
over the live cache object and replayed it ~61 times (1 warm + _N=60). Attention's
forward calls cache.update_and_fetch, which APPENDS a token per call, so profiling
injected ~61 phantom tokens into layers 0-7 only (the captured ones). Consequences:

  * the run's generated text is corrupt from that point on, so tok/s and token
    hashes from any MTPLX_ROOFLINE_PROFILE=1 run cannot be trusted; and
  * the measurement is not idempotent -- offset crosses KVCache.step=256
    boundaries mid-bench, triggering whole-buffer reallocation inside the timing.

snapshot_cache returns an independent copy, or None (caller skips the capture)
when one cannot be made safely -- losing a component's number beats corrupting
the run that produced it.

These tests assert MLX's copy semantics rather than assuming them: the whole fix
rests on `arr + 0` yielding a distinct buffer, so that is verified directly.
"""

import mlx.core as mx
import pytest

from mtplx.roofline_profile import snapshot_cache

KVCache = pytest.importorskip("mlx_lm.models.cache").KVCache

HEADS, DIM = 8, 128


def _tok(n: int = 1):
    return mx.ones((1, HEADS, n, DIM), dtype=mx.bfloat16)


def _primed(tokens: int = 300):
    """A cache with real content, past one step=256 boundary."""
    cache = KVCache()
    cache.update_and_fetch(_tok(tokens), _tok(tokens))
    mx.eval(cache.keys, cache.values)
    return cache


def test_snapshot_is_independent_of_the_live_cache() -> None:
    live = _primed()
    before_offset = live.offset
    before_sum = float(live.keys.sum())

    snap = snapshot_cache(live)
    assert snap is not None

    # Replay the way the profiler does: many appends against the snapshot.
    for _ in range(61):
        snap.update_and_fetch(_tok(), _tok())
    mx.eval(snap.keys, snap.values, live.keys, live.values)

    assert live.offset == before_offset, "profiling advanced the LIVE cache"
    assert float(live.keys.sum()) == before_sum, "profiling mutated LIVE cache contents"
    assert snap.offset == before_offset + 61, "snapshot did not actually record appends"


def test_snapshot_preserves_shape_and_offset() -> None:
    """The replay must measure a REALISTIC KV length, not a fresh empty cache."""
    live = _primed(300)
    snap = snapshot_cache(live)
    assert snap.offset == live.offset == 300
    assert snap.keys.shape == live.keys.shape
    assert snap.values.shape == live.values.shape


def test_mlx_add_zero_yields_a_distinct_buffer() -> None:
    """The fix depends on this; verify it instead of trusting aliasing rules."""
    original = mx.ones((1, HEADS, 256, DIM), dtype=mx.bfloat16)
    mx.eval(original)
    copied = original + 0
    mx.eval(copied)
    copied[..., 0:1, :] = mx.zeros((1, HEADS, 1, DIM), dtype=mx.bfloat16)
    mx.eval(copied, original)
    assert float(original[0, 0, 0, 0]) == 1.0, "arr + 0 aliased the source buffer"
    assert float(copied[0, 0, 0, 0]) == 0.0


def test_none_cache_is_handled() -> None:
    assert snapshot_cache(None) is None


def test_unsupported_cache_is_skipped_not_shared() -> None:
    """Quantized caches hold TUPLES of arrays. Returning None makes the caller skip
    the capture; sharing the live buffer would corrupt the run."""

    class QuantishCache:
        def __init__(self):
            self.keys = (mx.ones((4,)), mx.ones((4,)), mx.ones((4,)))
            self.values = (mx.ones((4,)), mx.ones((4,)), mx.ones((4,)))
            self.offset = 7

    assert snapshot_cache(QuantishCache()) is None
