"""Attention plumbing for the ragged batch KV lane (R2).

The ragged lane reuses the existing stock attention path unchanged.  Two seams
carry the per-row information:

1. **RoPE.**  ``RaggedBatchKVCache.offset`` is an ``int32[B]`` array, so the
   existing rope call sites (``attention_split.py:202-203``,
   ``self.rope(queries, offset=cached_prefix_offset)``) already forward per-row
   start positions -- ``mx.fast.rope`` applies a ``[B]`` offset per row,
   bitwise-verified on MLX 0.31.2.  When the cache is the old scalar class the
   offset stays a Python ``int`` and that path is byte-identical.  No edit to
   ``attention_split.py`` is required; the array simply flows through, and
   ``_cache_offset_static_int`` returning ``None`` fails every custom Metal
   fast-path closed so only stock SDPA runs.

2. **Mask.**  The stock attention consumes a caller-supplied mask
   (``attention_split.py:179-183`` accepts an explicit ``mx.array``).  This
   module builds that mask from the per-row logical lengths.

Fixed-shape gate discipline: :func:`ragged_causal_mask` ALWAYS returns a
``[B, 1, q_len, key_len]`` boolean array.  Neither the shape nor the dtype ever
depends on the *content* of ``q_start`` -- only which entries are ``True`` does.
That keeps kernel dispatch shape-stable across every composition (append,
replay, ragged offsets), which is what the scheme's fixed-shape correctness
gate requires.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

__all__ = ["ragged_causal_mask", "ragged_additive_mask"]


def _q_start_vector(q_start: Any) -> mx.array:
    arr = q_start if isinstance(q_start, mx.array) else mx.array(q_start)
    return arr.astype(mx.int32).reshape(-1)


def ragged_causal_mask(
    q_start: Any,
    q_len: int,
    key_len: int,
) -> mx.array:
    """Per-row causal attention mask ``[B, 1, q_len, key_len]`` (boolean).

    Parameters
    ----------
    q_start:
        Per-row absolute position of the *first* query token (``int32[B]``).
        Query ``i`` in row ``b`` sits at absolute position ``q_start[b] + i``.
        For an append write this is the row's pre-write offset; for a REPLAY
        write it is ``offset - 1`` (the same value passed as ``write_start`` to
        ``RaggedBatchKVCache.update_and_fetch``).
    q_len:
        Number of query tokens per row (the fixed cohort width).
    key_len:
        Physical key capacity -- i.e. ``keys.shape[2]`` of the buffer returned
        by ``update_and_fetch``.  Zero-padded positions beyond a row's logical
        length are excluded automatically by the causal rule.

    Returns
    -------
    ``[B, 1, key_len]``-broadcasting boolean mask of shape
    ``(B, 1, q_len, key_len)`` where ``mask[b, 0, i, j]`` is ``True`` iff key
    position ``j`` is attendable by query ``i`` of row ``b`` -- that is,
    ``j <= q_start[b] + i``.  Each row therefore attends to exactly its own
    ``[0, q_start[b])`` history plus its own new positions, causally, and to
    nothing else.

    The shape ``(B, 1, q_len, key_len)`` and dtype ``bool`` are invariant to the
    values in ``q_start``.
    """
    starts = _q_start_vector(q_start)
    q_len = int(q_len)
    key_len = int(key_len)
    # absolute query positions: [B, q_len]
    qpos = starts[:, None] + mx.arange(q_len, dtype=mx.int32)[None, :]
    # key positions: [key_len]
    kpos = mx.arange(key_len, dtype=mx.int32)
    # [B, 1, q_len, key_len] = [1,1,1,key_len] <= [B,1,q_len,1]
    mask = kpos[None, None, None, :] <= qpos[:, None, :, None]
    return mask


def ragged_additive_mask(
    q_start: Any,
    q_len: int,
    key_len: int,
    *,
    dtype: Any = mx.float32,
) -> mx.array:
    """Additive form of :func:`ragged_causal_mask`.

    Disallowed positions get ``-inf``; allowed positions get ``0``.  Same fixed
    ``[B, 1, q_len, key_len]`` shape, ``dtype`` fixed by the caller.  Provided
    for callers/kernels that want an additive mask; the boolean form is the
    canonical one the stock SDPA path takes.
    """
    allow = ragged_causal_mask(q_start, q_len, key_len)
    neg = mx.array(float("-inf"), dtype=dtype)
    zero = mx.array(0.0, dtype=dtype)
    return mx.where(allow, zero, neg)
