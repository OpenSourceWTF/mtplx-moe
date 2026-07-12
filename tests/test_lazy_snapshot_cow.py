"""Lazy (zero-copy) snapshot semantics on the paged KV cache.

The kvcache-v2 lazy design stores *references* to the live paged buffers plus
an offset instead of `_clone_tree` copies. That is only sound if MLX's
functional update semantics guarantee a retained array reference can never be
mutated by later cache writes: `mx.slice_update` must produce a new buffer
(copy-on-write) whenever the input buffer is still referenced elsewhere, and
may only donate/write in place when the cache holds the sole reference.

These tests pin that contract on the real `TensorOffsetVllmMetalPagedKVCache`
write path. If they ever fail (an MLX upgrade changing donation rules), the
lazy snapshot path must fall back to eager copies — fail loudly here, not
silently in restored sessions.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.cache_state import TensorOffsetVllmMetalPagedKVCache

BLOCKS = 4
BLOCK = 8
HEADS = 2
DIM = 4


def _make_cache() -> TensorOffsetVllmMetalPagedKVCache:
    return TensorOffsetVllmMetalPagedKVCache(
        key_cache=mx.zeros((BLOCKS, BLOCK, HEADS, DIM), dtype=mx.float16),
        value_cache=mx.zeros((BLOCKS, BLOCK, HEADS, DIM), dtype=mx.float16),
        offset=0,
        block_size=BLOCK,
        num_blocks=BLOCKS,
    )


def _step(seed: int, steps: int = 1) -> tuple[mx.array, mx.array]:
    k = mx.random.normal((1, HEADS, steps, DIM), key=mx.random.key(seed)).astype(
        mx.float16
    )
    v = mx.random.normal((1, HEADS, steps, DIM), key=mx.random.key(seed + 10_000)).astype(
        mx.float16
    )
    return k, v


def _flat_prefix(cache_array: mx.array, tokens: int) -> mx.array:
    flat = cache_array.reshape(-1, HEADS, DIM)
    return flat[:tokens]


def test_retained_reference_survives_later_appends() -> None:
    """A snapshot that is just (array ref, offset) must be immune to appends."""
    live = _make_cache()
    control = _make_cache()
    for i in range(10):
        k, v = _step(i)
        live.update_without_fetch(k, v)
        control.update_without_fetch(k, v)
    mx.eval(live.cache[0], live.cache[1])

    # Lazy snapshot: retain references, no copy.
    snap_k, snap_v = live.cache[0], live.cache[1]
    snap_offset = int(live.offset.item())

    # Live cache moves on (would overwrite rows 10..14 in place if donation
    # were allowed despite our retained reference).
    for i in range(5):
        k, v = _step(100 + i)
        live.update_without_fetch(k, v)
    mx.eval(live.cache[0], live.cache[1])

    assert snap_offset == 10
    assert mx.array_equal(
        _flat_prefix(snap_k, 10), _flat_prefix(control.cache[0], 10)
    ).item(), "retained key reference was mutated by a later append"
    assert mx.array_equal(
        _flat_prefix(snap_v, 10), _flat_prefix(control.cache[1], 10)
    ).item(), "retained value reference was mutated by a later append"


def test_restore_from_retained_reference_is_exact_after_divergence() -> None:
    """Restoring from retained refs equals a cache that never diverged."""
    live = _make_cache()
    control = _make_cache()
    for i in range(10):
        k, v = _step(i)
        live.update_without_fetch(k, v)
        control.update_without_fetch(k, v)

    snap = (live.cache[0], live.cache[1], int(live.offset.item()))
    for i in range(5):  # divergent tail on the live cache
        live.update_without_fetch(*_step(200 + i))

    restored = TensorOffsetVllmMetalPagedKVCache(
        key_cache=snap[0],
        value_cache=snap[1],
        offset=snap[2],
        block_size=BLOCK,
        num_blocks=BLOCKS,
    )
    # Both continue with the same suffix; states must stay identical.
    for i in range(4):
        k, v = _step(300 + i)
        restored.update_without_fetch(k, v)
        control.update_without_fetch(k, v)
    mx.eval(restored.cache[0], control.cache[0])

    tokens = int(restored.offset.item())
    assert tokens == 14
    assert mx.array_equal(
        _flat_prefix(restored.cache[0], tokens), _flat_prefix(control.cache[0], tokens)
    ).item()
    assert mx.array_equal(
        _flat_prefix(restored.cache[1], tokens), _flat_prefix(control.cache[1], tokens)
    ).item()


def test_inplace_trim_then_append_rewrites_tail_only() -> None:
    """Lease path: trim + re-append must preserve the kept prefix exactly."""
    live = _make_cache()
    control = _make_cache()
    for i in range(10):
        k, v = _step(i)
        live.update_without_fetch(k, v)
        control.update_without_fetch(k, v)

    trimmed = live.trim(4)
    assert trimmed == 4
    assert int(live.offset.item()) == 6

    for i in range(3):
        k, v = _step(400 + i)
        live.update_without_fetch(k, v)
    mx.eval(live.cache[0])

    assert mx.array_equal(
        _flat_prefix(live.cache[0], 6), _flat_prefix(control.cache[0], 6)
    ).item(), "trim+append corrupted the retained prefix"
    assert int(live.offset.item()) == 9


def test_snapshot_across_block_boundary_growth() -> None:
    """Snapshot taken mid-block stays exact while appends cross block edges."""
    live = _make_cache()
    control = _make_cache()
    for i in range(BLOCK - 2):  # stop 2 short of the first block edge
        k, v = _step(i)
        live.update_without_fetch(k, v)
        control.update_without_fetch(k, v)

    snap_k = live.cache[0]
    snap_tokens = int(live.offset.item())

    for i in range(BLOCK):  # cross the block boundary on the live cache
        live.update_without_fetch(*_step(500 + i))
    mx.eval(live.cache[0])

    assert mx.array_equal(
        _flat_prefix(snap_k, snap_tokens), _flat_prefix(control.cache[0], snap_tokens)
    ).item()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
