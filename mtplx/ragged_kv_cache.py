"""Ragged batch KV cache for the A3B batched-decode lane (R1).

The batch-generic ``TailOwnedKVCache`` keys everything off a single *scalar*
offset: one shared logical length for every row, a contiguous shared-slice
write (``self.keys[..., prev:offset, :] = keys``), and a scalar-offset causal
mask.  That is correct for lockstep decode where all rows always sit at the
same position, but the fold-in reject scheme (scheme doc S2.2/S6) needs rows to
diverge *without bound*: an accepting stream advances 2 positions/cycle while a
rejecting stream advances 1 and rewrites a stale draft slot.

``RaggedBatchKVCache`` is the opt-in cache that supports that:

* per-row logical lengths ``offsets: int32[B]`` instead of a scalar,
* a shared physical buffer sized to a step-rounded max capacity, kept
  zero-padded (ragged *logical* lengths, not paged *physical* memory), and
* a vectorized per-row scatter write so different rows write at different
  sequence positions in a single call, including *rewriting* positions below
  another row's offset (the REPLAY precondition).

Design invariants that make it a safe drop-in (verified by the R1/R2 gates):

* ``offset`` is exposed as the ``int32[B]`` array so the existing attention
  rope call sites (``attention_split.py:202-203``) forward per-row start
  positions unchanged, and ``_cache_offset_static_int`` returns ``None`` which
  fails every custom Metal fast-path closed -> stock SDPA only.
* ``update_and_fetch`` returns the FULL zero-padded physical buffer; the
  companion ragged mask (``ragged_attention.py``) confines each row to its own
  valid range.  With uniform offsets this is bitwise-identical to the scalar
  lane's trimmed-buffer + causal-mask output (degenerate-equivalence gate).

The scalar lane in ``cache_state.py`` is left byte-for-byte untouched; this is a
sibling module, imported only when the ragged lane is explicitly selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

__all__ = [
    "RaggedBatchKVCache",
    "RaggedKVSnapshot",
    "snapshot_ragged_caches",
    "restore_ragged_caches",
    "restore_ragged_caches_masked",
]


def _as_int32_vector(value: Any, *, batch_size: int | None = None) -> mx.array:
    arr = value if isinstance(value, mx.array) else mx.array(value)
    arr = arr.astype(mx.int32).reshape(-1)
    if batch_size is not None and int(arr.size) != int(batch_size):
        raise ValueError(
            f"expected a length-{batch_size} per-row vector, got size {int(arr.size)}"
        )
    return arr


@dataclass(frozen=True)
class RaggedKVSnapshot:
    """Whole-batch snapshot of a ragged cache (owned copies)."""

    keys: Any
    values: Any
    offsets: Any
    step: int


class RaggedBatchKVCache:
    """Full-attention KV cache with per-row logical lengths.

    Parameters
    ----------
    batch_size:
        Number of rows.  Optional; inferred from the first ``update_and_fetch``
        keys tensor when ``offsets`` is not given.
    step:
        Physical growth granularity (matches ``TailOwnedKVCache.step``).
    keys, values:
        Optional pre-existing physical buffers (``[B, n_kv_heads, cap, dim]``).
    offsets:
        Optional per-row logical lengths (``int32[B]``).
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        step: int = 256,
        keys: Any | None = None,
        values: Any | None = None,
        offsets: Any | None = None,
    ) -> None:
        self.keys = keys
        self.values = values
        self.step = int(step)
        if offsets is not None:
            self.offsets = _as_int32_vector(offsets, batch_size=batch_size)
        elif batch_size is not None:
            self.offsets = mx.zeros((int(batch_size),), dtype=mx.int32)
        elif keys is not None:
            self.offsets = mx.zeros((int(keys.shape[0]),), dtype=mx.int32)
        else:
            self.offsets = None
        # telemetry
        self.ragged_updates = 0
        self.ragged_grows = 0
        # HOST-side monotone capacity upper bound (fable-main review, item 3).
        # When set, ``update_and_fetch`` grows the physical buffer off this Python
        # int and NEVER reads the device ``offsets`` for capacity -- so once
        # ``write_start`` depends on the accept mask (a device expression) the call
        # does not force a sync that would block the pipelined fold-in forward.
        # ``None`` selects the legacy one-off path (a single device read), keeping
        # every pre-existing caller byte-identical.
        self._capacity_bound: int | None = None
        # CONSTANT-capacity pin (refill/admission lane).  When set, every write
        # and mask uses exactly this capacity: the §47 fixed-shape discipline
        # extended to the KV axis, and no bound arithmetic / device read at all.
        self._frozen_capacity: int | None = None

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_scalar_cache(
        cls, entry: Any, *, batch_size: int | None = None, step: int | None = None
    ) -> "RaggedBatchKVCache":
        """Build a ragged cache seeded from a scalar batch-generic cache.

        Every row inherits the scalar cache's single offset, so the result is
        the degenerate (uniform) ragged cache equivalent to ``entry``.
        """
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        scalar_offset = int(getattr(entry, "offset", 0))
        if batch_size is None and keys is not None:
            batch_size = int(keys.shape[0])
        if batch_size is None:
            raise ValueError("batch_size is required when the source cache has no keys")
        offsets = mx.full((int(batch_size),), scalar_offset, dtype=mx.int32)
        out = cls(
            batch_size=int(batch_size),
            step=int(step or getattr(entry, "step", 256)),
            keys=keys,
            values=values,
            offsets=offsets,
        )
        # Seed the host capacity bound from the (host-known) scalar offset so the
        # fold-in decode lane is sync-free from its first ``update_and_fetch``.
        # from_scalar_cache makes every row uniform at ``scalar_offset``, so this
        # is an EXACT upper bound on the max logical offset (not the step-rounded
        # physical capacity -- seeding that would over-grow every cycle).
        out._capacity_bound = int(scalar_offset)
        return out

    # -- core contract -------------------------------------------------------

    @property
    def offset(self) -> Any:
        """Per-row logical lengths as an ``int32[B]`` array.

        Named ``offset`` (not ``offsets``) so the attention rope call sites and
        ``_cache_offset_value`` in ``attention_split.py`` pick it up unchanged;
        the array form disables every scalar-int-gated custom kernel path.
        """
        return self.offsets

    def _batch_size(self) -> int:
        if self.offsets is not None:
            return int(self.offsets.size)
        if self.keys is not None:
            return int(self.keys.shape[0])
        raise ValueError("ragged cache batch size is not yet known")

    def _ensure_offsets(self, batch_size: int) -> None:
        if self.offsets is None:
            self.offsets = mx.zeros((int(batch_size),), dtype=mx.int32)
        elif int(self.offsets.size) != int(batch_size):
            raise ValueError(
                f"batch size mismatch: cache has {int(self.offsets.size)} rows, "
                f"write has {int(batch_size)}"
            )

    def _grow_to(self, required_positions: int, template_keys: mx.array, template_values: mx.array) -> None:
        """Grow the physical buffers so index ``required_positions-1`` is valid."""
        B, n_kv_heads, _, k_dim = template_keys.shape
        v_dim = template_values.shape[3]
        current_cap = 0 if self.keys is None else int(self.keys.shape[2])
        if required_positions <= current_cap and self.keys is not None:
            return
        n_steps = (int(required_positions) + self.step - 1) // self.step
        new_cap = max(n_steps * self.step, self.step)
        add = new_cap - current_cap
        pad_k = mx.zeros((B, n_kv_heads, add, k_dim), template_keys.dtype)
        pad_v = mx.zeros((B, n_kv_heads, add, v_dim), template_values.dtype)
        if self.keys is None:
            self.keys, self.values = pad_k, pad_v
        else:
            self.keys = mx.concatenate([self.keys, pad_k], axis=2)
            self.values = mx.concatenate([self.values, pad_v], axis=2)
        self.ragged_grows += 1

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        *,
        write_start: Any | None = None,
        new_offsets: Any | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Scatter each row's ``q``-token K/V block at its own position.

        Parameters
        ----------
        keys, values:
            New tail ``[B, n_kv_heads, q, dim]`` tensors (uniform ``q`` across
            rows -- the fixed-shape cohort width).
        write_start:
            Per-row start position (``int32[B]``); the block occupies
            ``[write_start[r], write_start[r] + q)`` for row ``r``.  Defaults to
            the current per-row offsets (append at each row's own tail).  A
            REPLAY row passes ``write_start = offset - 1`` to overwrite a stale
            draft slot.
        new_offsets:
            Per-row logical length after the write (``int32[B]``).  Defaults to
            ``write_start + q``.  Passing a smaller value expresses a per-row
            advance ``adv[r] < q`` (e.g. a stalled / dummy slot that writes
            ``q`` positions for shape stability but does not commit them).

        Returns the FULL zero-padded physical buffers ``[B, n_kv_heads, cap,
        dim]``; the ragged mask restricts each row to its valid range.
        """
        B, _n_kv, q, _k_dim = keys.shape
        self._ensure_offsets(B)

        if write_start is None:
            start = self.offsets
        else:
            start = _as_int32_vector(write_start, batch_size=B)

        if new_offsets is None:
            resulting = start + int(q)
        else:
            resulting = _as_int32_vector(new_offsets, batch_size=B)

        # Capacity: prefer the HOST-side monotone bound (item 3).  Every write
        # advances a row's offset by at most ``q``, so ``_capacity_bound += q`` is
        # an upper bound on ``max(offsets)`` -- and a REPLAY write at ``offset-1``
        # touches positions ``[offset-1, offset-1+q) < _capacity_bound + q``, never
        # exceeding it.  This reads NO device offset, so a device-valued
        # ``write_start`` (the fold-in loop) never forces a sync here.  With no
        # host bound seeded we fall back to the legacy single device read
        # (byte-identical to the pre-fix behaviour for every existing caller).
        if self._frozen_capacity is not None:
            required = self._frozen_capacity
        elif self._capacity_bound is not None:
            self._capacity_bound += int(q)
            required = self._capacity_bound
        else:
            required = int(mx.max(start).item()) + int(q)
        self._grow_to(required, keys, values)

        # per-row scatter indices along the sequence axis (axis=2):
        # idx[b, h, i, d] = start[b] + i
        seq_idx = start[:, None] + mx.arange(q, dtype=mx.int32)[None, :]  # [B, q]
        n_kv_heads = int(self.keys.shape[1])
        k_dim = int(self.keys.shape[3])
        v_dim = int(self.values.shape[3])
        idx_k = mx.broadcast_to(
            seq_idx[:, None, :, None], (B, n_kv_heads, int(q), k_dim)
        ).astype(mx.int32)
        idx_v = mx.broadcast_to(
            seq_idx[:, None, :, None], (B, n_kv_heads, int(q), v_dim)
        ).astype(mx.int32)
        self.keys = mx.put_along_axis(self.keys, idx_k, keys, axis=2)
        self.values = mx.put_along_axis(self.values, idx_v, values, axis=2)

        self.offsets = resulting.astype(mx.int32)
        self.ragged_updates += 1
        return self.keys, self.values

    def make_mask(
        self,
        q_len: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
        key_len: int | None = None,
    ) -> mx.array:
        """Per-row causal mask ``[B, 1, q, key_len]`` from the current offsets.

        This is the seam ``mlx_lm.models.base.create_attention_mask`` takes: it
        delegates to ``cache.make_mask(N, return_array=..., window_size=...)``
        whenever the cache entry exposes ``make_mask``, so a full-attention layer
        gets the RAGGED mask WITHOUT any edit to the model or the custom attention
        (``attention_split`` consumes an explicit ``mx.array`` mask on its stock
        SDPA path).  The ``return_array``/``window_size`` kwargs match that call
        signature: this lane is full-attention (the GDN layers carry the sliding
        window on the recurrent side, so ``window_size`` is inapplicable) and it
        ALWAYS returns an explicit ``[B, 1, q, key_len]`` boolean array (never the
        ``"causal"`` sentinel), which is exactly ``return_array`` semantics.

        Uses this cache's *current* offsets as the per-row query-start positions
        -- for the fold-in loop those are the pre-write offsets it set before the
        forward (``offset`` for SPEC rows, ``offset-1`` for REPLAY rows).
        ``key_len`` defaults to the physical buffer capacity; the fold-in loop
        calls :meth:`reserve` before the forward so this already equals the
        post-``update_and_fetch`` capacity (no grow inside the forward).
        """
        from .ragged_attention import ragged_causal_mask

        del return_array, window_size  # honoured implicitly (see docstring)
        if key_len is None:
            key_len = 0 if self.keys is None else int(self.keys.shape[2])
        return ragged_causal_mask(
            q_start=self.offsets,
            q_len=int(q_len),
            key_len=int(key_len),
        )

    def reserve(self, additional_positions: int) -> None:
        """Pre-grow the physical buffer to hold ``additional_positions`` more.

        The fold-in decode loop calls this on every ragged entry BEFORE the
        forward so the ragged mask's ``key_len`` (read from ``keys.shape[2]``
        inside ``create_attention_mask`` -> :meth:`make_mask`) already matches the
        capacity ``update_and_fetch`` will return -- no grow happens *inside* the
        forward, so the mask and the K/V buffer stay shape-consistent and the
        fixed-shape cohort discipline holds.  Grows purely off the host bound
        (never a device read); a no-op until a host bound is seeded and the
        buffers exist (post-prefill).
        """
        if self.keys is None or self.values is None:
            return
        if self._frozen_capacity is not None:
            self._grow_to(self._frozen_capacity, self.keys, self.values)
            return
        if self._capacity_bound is None:
            return
        required = self._capacity_bound + int(additional_positions)
        self._grow_to(required, self.keys, self.values)

    def freeze_capacity(self, capacity: int) -> None:
        """Pin the physical buffer at a CONSTANT capacity (refill lane).

        After this, every mask key width and every ``update_and_fetch`` growth
        target is exactly ``capacity`` (step-rounded once by ``_grow_to``) for
        the rest of the decode — the fixed-shape cohort discipline extended to
        the KV axis, so admission events can never shift kernel dispatch
        mid-run.  The CALLER contracts that no row ever writes at or beyond the
        rounded capacity (the refill driver bounds offsets by ``prompt_len +
        max_new + lag + one admission window``, far under the frozen cap).
        Buffers that do not exist yet grow to the frozen capacity on their
        first write instead.
        """
        cap = int(capacity)
        if self.keys is not None and self.values is not None:
            self._grow_to(cap, self.keys, self.values)
        self._frozen_capacity = cap

    def size(self) -> int:
        if self.offsets is None:
            return 0
        return int(mx.max(self.offsets).item())

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Whole-batch trim: subtract ``n`` from every row's offset (clamped).

        Parity with the scalar lane's ``trim``.  Per-row rollback uses the
        masked-restore path, not this.
        """
        if self.offsets is None:
            return 0
        n = int(n)
        smallest = int(mx.min(self.offsets).item())
        n = min(n, max(smallest, 0))
        self.offsets = mx.maximum(self.offsets - n, 0).astype(mx.int32)
        return n

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return int(self.keys.nbytes) + int(self.values.nbytes)

    # -- state / meta_state (snapshot_cache compatibility) -------------------

    @property
    def state(self):
        return (self.keys, self.values, self.offsets)

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.keys = self.values = self.offsets = None
            return
        keys, values, offsets = value
        self.keys = keys
        self.values = values
        self.offsets = None if offsets is None else _as_int32_vector(offsets)

    @property
    def meta_state(self) -> tuple[str, ...]:
        return ("ragged_batch_kv", str(self.step))

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        if len(value) > 1:
            self.step = int(value[1])

    # -- filter / merge parity -----------------------------------------------

    def filter(self, batch_indices: Any) -> None:
        """Keep only the rows in ``batch_indices`` (admission vacate / cohort recompose)."""
        idx = batch_indices if isinstance(batch_indices, mx.array) else mx.array(batch_indices)
        idx = idx.astype(mx.int32).reshape(-1)
        if self.keys is not None:
            self.keys = self.keys[idx]
            self.values = self.values[idx]
        if self.offsets is not None:
            self.offsets = self.offsets[idx].astype(mx.int32)

    def extend(self, other: "RaggedBatchKVCache") -> None:
        """Concatenate ``other``'s rows after this cache's rows (cohort merge).

        Physical capacities are aligned by zero-padding the shorter buffer.
        Offsets are the per-row logical lengths; no repositioning is performed
        (rows keep their absolute positions -- correct for the zero-pad lane).
        """
        if other.keys is None:
            return
        if self.keys is None:
            self.keys = other.keys
            self.values = other.values
            self.offsets = (
                None if other.offsets is None else other.offsets.astype(mx.int32)
            )
            return
        a_k, a_v = self.keys, self.values
        b_k, b_v = other.keys, other.values
        cap = max(int(a_k.shape[2]), int(b_k.shape[2]))
        a_k, a_v = _pad_seq_to(a_k, cap), _pad_seq_to(a_v, cap)
        b_k, b_v = _pad_seq_to(b_k, cap), _pad_seq_to(b_v, cap)
        self.keys = mx.concatenate([a_k, b_k], axis=0)
        self.values = mx.concatenate([a_v, b_v], axis=0)
        self.offsets = mx.concatenate(
            [self.offsets, other.offsets], axis=0
        ).astype(mx.int32)

    # -- snapshot / restore --------------------------------------------------

    def snapshot(self) -> RaggedKVSnapshot:
        """Owned whole-batch snapshot (keys/values/offsets copied)."""
        return RaggedKVSnapshot(
            keys=None if self.keys is None else _own(self.keys),
            values=None if self.values is None else _own(self.values),
            offsets=None if self.offsets is None else _own(self.offsets),
            step=int(self.step),
        )

    def restore(self, snap: RaggedKVSnapshot) -> None:
        """Whole-batch restore from a snapshot."""
        self.keys = None if snap.keys is None else _own(snap.keys)
        self.values = None if snap.values is None else _own(snap.values)
        self.offsets = None if snap.offsets is None else _own(snap.offsets)
        self.step = int(snap.step)

    def restore_masked(self, snap: RaggedKVSnapshot, row_mask: Any) -> None:
        """Per-row masked restore.

        ``row_mask`` is a boolean ``[B]`` selector: rows where it is ``True``
        revert (both their ``offsets`` AND their buffer region) to ``snap``;
        rows where it is ``False`` keep their current advanced state.  Bitwise
        exact for the reverting rows and untouched for the advancing rows.
        """
        mask = row_mask if isinstance(row_mask, mx.array) else mx.array(row_mask)
        mask = mask.astype(mx.bool_).reshape(-1)

        if snap.offsets is not None and self.offsets is not None:
            self.offsets = mx.where(mask, snap.offsets, self.offsets).astype(mx.int32)

        if snap.keys is None or self.keys is None:
            return
        cap = max(int(self.keys.shape[2]), int(snap.keys.shape[2]))
        cur_k = _pad_seq_to(self.keys, cap)
        cur_v = _pad_seq_to(self.values, cap)
        snap_k = _pad_seq_to(snap.keys, cap)
        snap_v = _pad_seq_to(snap.values, cap)
        m = mask[:, None, None, None]
        self.keys = mx.where(m, snap_k, cur_k)
        self.values = mx.where(m, snap_v, cur_v)


def _own(value: mx.array) -> mx.array:
    """Force an independent array expression (COW-safe snapshot leaf)."""
    return value + mx.zeros((), dtype=value.dtype)


def _pad_seq_to(buf: mx.array, cap: int) -> mx.array:
    """Zero-pad a ``[B, H, seq, dim]`` buffer along the seq axis up to ``cap``."""
    seq = int(buf.shape[2])
    if seq >= cap:
        return buf
    pad = mx.zeros(
        (int(buf.shape[0]), int(buf.shape[1]), cap - seq, int(buf.shape[3])),
        buf.dtype,
    )
    return mx.concatenate([buf, pad], axis=2)


# -- cache-list helpers (parity with cache_state snapshot conventions) -------


def snapshot_ragged_caches(cache: list[Any]) -> tuple[RaggedKVSnapshot | None, ...]:
    """Snapshot every :class:`RaggedBatchKVCache` entry; ``None`` for others.

    Companion to ``cache_state.snapshot_untrimmable_cache``: that helper skips
    trimmable KV (stores ``None``) because the scalar lane rolls back by
    trimming.  The ragged lane's per-row masked restore cannot be expressed as
    a scalar trim, so this captures the ragged entries explicitly.
    """
    out: list[RaggedKVSnapshot | None] = []
    for entry in cache:
        if isinstance(entry, RaggedBatchKVCache):
            out.append(entry.snapshot())
        else:
            out.append(None)
    return tuple(out)


def restore_ragged_caches(
    cache: list[Any], snapshots: tuple[RaggedKVSnapshot | None, ...]
) -> None:
    """Whole-batch restore of every ragged entry from ``snapshots``."""
    for entry, snap in zip(cache, snapshots):
        if isinstance(entry, RaggedBatchKVCache) and snap is not None:
            entry.restore(snap)


def restore_ragged_caches_masked(
    cache: list[Any],
    snapshots: tuple[RaggedKVSnapshot | None, ...],
    row_mask: Any,
) -> None:
    """Per-row masked restore of every ragged entry (fold-in reject rollback)."""
    for entry, snap in zip(cache, snapshots):
        if isinstance(entry, RaggedBatchKVCache) and snap is not None:
            entry.restore_masked(snap, row_mask)
