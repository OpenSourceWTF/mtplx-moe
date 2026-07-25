"""A pure, ``mx.compile``-able single-token decode step for Laguna-S-2.1.

Laguna's B=1 decode step spends ~5.7 ms per token building and encoding the
graph on the host.  Almost none of that is the model: it is 48 attention blocks
and 47 MoE blocks each re-traced from Python, plus the two mlx-lm cache classes
mutating their buffers and their *Python integer* offsets between every layer.
Those integers are the blocker.  ``mx.compile`` captures Python scalars as trace
constants, so a compiled step built at offset ``P`` would keep using ``P`` for
rope and for the mask forever — the same reason ``SpecDecodeGraphBank`` refuses
to capture the stock caches (see ``mtplx/graphbank.py``).

This module removes the blocker by making the whole decode state explicit
tensors — "leaves" — that go in as arguments and come out as results:

* the KV buffer, one per layer (see "Packed KV" below);
* ``offset``, an int32 scalar ``mx.array`` shared by every layer;
* ``ring_idx``, an int32 scalar ``mx.array`` shared by every sliding layer.

Everything the step reads is then either a weight (constant) or a leaf (an
argument), so ``mx.compile`` produces one graph that is valid at every position,
and an ICB capture becomes possible on top of it.

Packed KV
---------
K and V for a layer go to the SAME slot on the same step, so they do not need
two writes.  ``packed_kv=True`` gives each layer one leaf — measured
2026-07-24 at ~0.07 ms/step SLOWER in-model than unpacked (the Select
pack plus the strided sdpa plane views cost more than the saved
dispatch), so UNPACKED is the default and the packed layout stays
available for A/B —
``[2, H, S, D]`` — plane 0 keys, plane 1 values — and writes both with a single
``mx.slice_update`` whose update is ``[2, H, 1, D]``.  The leading axis doubles
as the batch axis on the way out: ``kv[0:1]`` is exactly the ``[1, H, S, D]``
keys tensor attention wants, which is only true because this lane is B=1.

That trades two dynamic writes for one write plus one pack.  It is a win only
because of how MLX implements each piece (checked against 0.31.2):

* a dynamic ``mx.slice_update`` costs TWO dispatches, ``compute_dynamic_offset``
  and a ``gg*_dynamic_copy``, so dropping one drops two;
* the pack is :func:`pack_kv`, a ``mx.where`` — ONE ``Select`` dispatch.
  ``mx.concatenate`` would be the obvious way to build the same array and is the
  wrong one: ``concatenate_gpu`` has no kernel of its own and issues one copy
  per input, so a two-input concat would cost back exactly what the merged write
  saved;
* the reads ``kv[0:1]`` / ``kv[1:2]`` are unit-stride slices, which MLX resolves
  to a buffer offset (``shared_buffer_slice``) rather than a copy, and both
  planes stay row-contiguous, so attention sees the layout it would have seen
  from separate leaves.

``packed_kv=False`` keeps the original two-leaf layout for A/B.  It is a pure
layout change: both produce the same tokens and the same numbers.

Two shared scalars, not 48
--------------------------
At B=1 every cache advances in lockstep: ``LagunaModel.__call__`` runs one token
through all 48 layers, so all 48 caches gain exactly one position per step.  The
full-attention caches therefore always agree on ``offset``, and the sliding
caches always agree on their write slot.  ``snapshot_leaves`` verifies both
rather than assuming them, and the step returns one updated scalar of each kind.

Regime
------
Sliding layers are handled in their *steady state* only — context at least as
long as the window, which is the regime the real model serves in (window 512
against prompts that are far longer).  In that regime every ring slot holds a
live key, so sliding attention needs no mask at all, and softmax's
permutation-invariance makes the ring's rotated order irrelevant.

Fused kernels
-------------
The step calls the shipped module code for gating, MoE and the norms, so every
``laguna_fused`` installer keeps applying to it.  The fused q/k norm+rope Metal
kernel is wired the same way and needs no env var here: when
``install_kernel_qk_rope`` (``MTPLX_LAGUNA_KERNEL_QK_ROPE``) has left a
``_qk_rope_spec`` on an attention module, and the projected q/k shapes are ones
the kernel covers, the step replaces that layer's norm -> transpose -> rope
chain with one dispatch; otherwise it runs the stock chain.

Scope: T=1, B=1, greedy.  Prefill still runs through the stock python-cache
path; this lane takes over afterwards via :func:`snapshot_leaves`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx

from .kernels.laguna_decode import fused_qk_norm_rope, is_qk_norm_rope_eligible
from .models import laguna

FULL = "full"
SLIDING = "sliding"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepGeometry:
    """Everything the step needs to know that is not a tensor.

    Read off the model rather than hardcoded: the shipped checkpoint is 48
    layers with a 512 window, but the toy models the tests build are neither.
    """

    cap: int
    window: int | None
    kinds: tuple[str, ...]
    packed_kv: bool = False

    @property
    def n_layers(self) -> int:
        return len(self.kinds)

    @property
    def has_sliding(self) -> bool:
        return self.window is not None

    @property
    def leaves_per_layer(self) -> int:
        """One packed ``[2, H, S, D]`` leaf, or the keys/values pair."""

        return 1 if self.packed_kv else 2

    @property
    def n_leaves(self) -> int:
        """The leaves of every layer, in layer order."""

        return self.leaves_per_layer * len(self.kinds)

    def layer_leaves(
        self, leaves: Sequence[mx.array], index: int
    ) -> tuple[mx.array, ...]:
        """The slice of a flat leaf sequence that belongs to layer ``index``."""

        width = self.leaves_per_layer
        return tuple(leaves[width * index : width * (index + 1)])


def geometry_for(model: Any, cap: int, *, packed_kv: bool = False) -> StepGeometry:
    """Derive the leaf layout from the model's own layers.

    Uses ``self_attn.is_sliding`` — the same flag ``LagunaModel.__call__``
    dispatches its two masks on — so the lane cannot disagree with the eager
    forward about which layers rotate.
    """

    cap = int(cap)
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    inner = getattr(model, "model", model)
    kinds = tuple(
        SLIDING if layer.self_attn.is_sliding else FULL for layer in inner.layers
    )
    window = None
    if SLIDING in kinds:
        window = int(model.args.sliding_window)
        if window <= 0:
            raise ValueError(f"sliding_window must be positive, got {window}")
    return StepGeometry(
        cap=cap, window=window, kinds=kinds, packed_kv=bool(packed_kv)
    )


# ---------------------------------------------------------------------------
# the packed KV layout
# ---------------------------------------------------------------------------
def kv_plane_mask() -> mx.array:
    """The ``[2, 1, 1, 1]`` selector that defines which plane is which.

    Plane 0 is keys, plane 1 is values — everything in this module that reads or
    writes a packed leaf goes through this constant or through the slices in
    :func:`unpack_kv`, so the convention is stated once.
    """

    return mx.array([True, False]).reshape(2, 1, 1, 1)


def pack_kv(
    keys: mx.array, values: mx.array, plane: mx.array | None = None
) -> mx.array:
    """Fold a ``[1, H, T, D]`` k and v into one ``[2, H, T, D]`` array.

    A ``mx.where`` rather than the obvious ``mx.concatenate``: both write the
    same bytes, but MLX has a ``Select`` kernel and has no concatenate kernel —
    ``concatenate_gpu`` allocates the output and then issues one
    ``copy_gpu_inplace`` PER INPUT.  On the step's hot path this function stands
    in for one of the two dynamic writes packing removes, so a two-dispatch pack
    would hand the saving straight back; a one-dispatch pack keeps it.

    ``plane`` lets a caller pass a hoisted :func:`kv_plane_mask` so the constant
    is not rebuilt inside a traced body.
    """

    if keys.shape != values.shape:
        raise ValueError(
            f"k and v must have the same shape to pack, got {keys.shape} "
            f"and {values.shape}"
        )
    if int(keys.shape[0]) != 1:
        raise ValueError(
            "the packed layout spends the leading axis on the k/v plane, so it "
            f"is B=1 only; got batch {keys.shape[0]}"
        )
    return mx.where(kv_plane_mask() if plane is None else plane, keys, values)


def unpack_kv(leaf: mx.array) -> tuple[mx.array, mx.array]:
    """Read a packed leaf back as the ``[1, H, S, D]`` k and v tensors.

    Unit-stride slices, which MLX resolves to a buffer offset instead of a copy,
    and each plane is still row-contiguous — so attention is handed the same
    layout the unpacked leaves would have handed it.  The ``[0:1]`` (rather than
    ``[0]``) is what restores the batch axis attention expects.
    """

    if int(leaf.shape[0]) != 2:
        raise ValueError(
            f"a packed KV leaf has 2 planes, got shape {tuple(leaf.shape)}"
        )
    return leaf[0:1], leaf[1:2]


# ---------------------------------------------------------------------------
# state tensorization
# ---------------------------------------------------------------------------
def _copy(buf: mx.array) -> mx.array:
    """Detach a leaf from a live mlx-lm cache buffer.

    ``KVCache`` and ``RotatingKVCache`` both write with ``self.keys[..., a:b, :]
    = k``, which mutates the array the attribute points at.  Handing that array
    out as a leaf would let the eager cache keep editing state this lane owns.
    ``mx.array(x)`` copies bit-for-bit — including signed zeros, which ``x + 0``
    would flatten — so a snapshot is never a numerics change.
    """

    return mx.array(buf)


def full_state_from_cache(cache: Any, cap: int) -> tuple[mx.array, mx.array]:
    """Zero-pad a ``KVCache``'s live prefix out to the fixed leaf length.

    ``KVCache`` over-allocates in 256-position steps and returns
    ``keys[..., :offset, :]`` from ``update_and_fetch``, so its buffer length is
    an allocation detail, not state.  The leaf pins one length — ``cap`` — for
    the whole run so the compiled graph never has to retrace, and the step's
    additive mask hides the padding.
    """

    keys, values = cache.keys, cache.values
    if keys is None or values is None:
        raise ValueError(
            "full-attention cache is empty; snapshot after a prefill forward"
        )
    if int(keys.shape[0]) != 1:
        raise ValueError(
            f"the compiled step is B=1 only, cache holds batch {keys.shape[0]}"
        )
    offset = int(cache.offset)
    if offset >= cap:
        # Not "> cap": the first step writes AT index `offset`, so a snapshot
        # with no room left cannot take a step and is a caller error, not a
        # silently truncated run.
        raise ValueError(
            f"cache offset {offset} leaves no room in a cap of {cap}; "
            "re-snapshot with a larger cap"
        )
    return _padded(keys, offset, cap), _padded(values, offset, cap)


def _padded(buf: mx.array, offset: int, cap: int) -> mx.array:
    batch, heads, _, dims = buf.shape
    live = _copy(buf[..., :offset, :])
    if offset == cap:
        return live
    pad = mx.zeros((batch, heads, cap - offset, dims), dtype=buf.dtype)
    return mx.concatenate([live, pad], axis=2)


def ring_state_from_cache(
    cache: Any, window: int
) -> tuple[mx.array, mx.array, int]:
    """Normalize a ``RotatingKVCache`` into a ``[1, H, W, D]`` ring plus a slot.

    Mirrors the head of ``RotatingKVCache._update_in_place`` (mlx-lm 0.31.3) at
    ``S=1``, ``keep=0``, which is what the eager cache would do to itself on its
    very next decode step:

    * a prefill goes through ``_update_concat``, which leaves the buffer at the
      FULL prompt length with ``_idx == length`` — not window length.  The next
      in-place update computes ``trim_size = length - max_size``, keeps the last
      ``max_size`` positions, sets ``_idx = max_size``, then rotates it to
      ``keep`` (0).  Doing that here is what makes the leaf a real ring;
    * a buffer already at window length only needs the rotate, which at
      ``keep=0`` is exactly ``_idx % window``.

    The returned slot is therefore always pre-normalized: it is the index the
    NEXT write lands on, in ``[0, window)``, never the sentinel ``window``.
    """

    if int(getattr(cache, "keep", 0)) != 0:
        raise ValueError(
            f"only keep=0 rotating caches are supported, got keep={cache.keep}"
        )
    if int(cache.max_size) != int(window):
        raise ValueError(
            f"cache max_size {cache.max_size} does not match the model's "
            f"sliding_window {window}"
        )
    keys, values = cache.keys, cache.values
    if keys is None or values is None:
        raise ValueError(
            "sliding cache is empty; snapshot after a prefill forward"
        )
    if int(keys.shape[0]) != 1:
        raise ValueError(
            f"the compiled step is B=1 only, cache holds batch {keys.shape[0]}"
        )

    offset = int(cache.offset)
    if offset < window:
        # Below the window the eager cache returns a SHORT k/v (see the tail of
        # `_update_in_place`) and the sliding mask still bites.  Supporting it
        # would mean a second graph shape for a regime the served model never
        # sits in, so it is refused rather than approximated.
        raise ValueError(
            f"sliding cache is not in steady state: offset {offset} < window "
            f"{window}; the compiled step supports ctx >= window only"
        )

    length = int(keys.shape[2])
    idx = int(cache._idx)
    if length > window:
        if idx != length:
            raise ValueError(
                "an over-long sliding buffer must be in linear temporal order "
                f"(_idx {idx} != length {length}); this one is rotated and "
                "cannot be trimmed the way the eager cache would"
            )
        keys = keys[..., length - window :, :]
        values = values[..., length - window :, :]
        idx = 0
    elif length < window:
        raise ValueError(
            f"sliding buffer is {length} long, shorter than the window {window}"
        )
    else:
        idx = idx % window

    return _copy(keys), _copy(values), idx


def snapshot_leaves(
    model: Any, caches: Sequence[Any], cap: int, *, packed_kv: bool = False
) -> tuple[mx.array, mx.array, tuple[mx.array, ...]]:
    """Turn a post-prefill eager cache list into the step's tensor state.

    Returns ``(offset, ring_idx, leaves)``, where ``leaves`` is one packed
    ``[2, H, S, D]`` array per layer in layer order — or, with
    ``packed_kv=False``, the keys/values pair — and both scalars are int32
    ``mx.array``s so the compiled graph reads them as inputs rather than baking
    them in.

    The lockstep invariant that lets one ``offset`` and one ``ring_idx`` serve
    all 48 layers is checked here, not assumed: every cache must report the same
    ``offset``, and every sliding cache the same normalized slot.

    The pack goes through :func:`pack_kv` like the step's own writes do, so one
    function decides which plane is keys.  This is a once-per-seed call, so its
    dispatch count does not matter; matching the step's layout does.
    """

    geometry = geometry_for(model, cap, packed_kv=packed_kv)
    if len(caches) != geometry.n_layers:
        raise ValueError(
            f"expected {geometry.n_layers} caches, got {len(caches)}"
        )

    offsets: set[int] = set()
    slots: set[int] = set()
    leaves: list[mx.array] = []
    for index, (kind, cache) in enumerate(zip(geometry.kinds, caches)):
        try:
            if kind == FULL:
                keys, values = full_state_from_cache(cache, geometry.cap)
            else:
                keys, values, slot = ring_state_from_cache(cache, geometry.window)
                slots.add(slot)
        except ValueError as error:
            raise ValueError(f"layer {index} ({kind}): {error}") from error
        offsets.add(int(cache.offset))
        if geometry.packed_kv:
            leaves.append(pack_kv(keys, values))
        else:
            leaves.extend((keys, values))

    if len(offsets) != 1:
        raise ValueError(
            f"caches are not in lockstep; offsets {sorted(offsets)} differ. "
            "One shared offset only holds at B=1 single-token decode."
        )
    if len(slots) > 1:
        raise ValueError(
            f"sliding caches disagree on the ring slot: {sorted(slots)}. "
            "One shared ring_idx only holds when every window advances together."
        )

    offset = mx.array(offsets.pop(), dtype=mx.int32)
    ring_idx = mx.array(slots.pop() if slots else 0, dtype=mx.int32)
    mx.eval(offset, ring_idx, *leaves)
    return offset, ring_idx, tuple(leaves)


# ---------------------------------------------------------------------------
# ring arithmetic, exposed so it can be tested against the real cache
# ---------------------------------------------------------------------------
def kv_slot_write(leaf: mx.array, update: mx.array, start: mx.array) -> mx.array:
    """Write one step's KV into a state leaf at a DYNAMIC slot.

    The eager caches do ``self.keys[..., idx : idx + 1, :] = k`` with a Python
    ``idx``.  ``mx.slice_update`` takes its start indices as an ``mx.array``,
    which is the whole point: the slot stays a graph input, so one compiled
    graph serves every position instead of one per position.

    ``axes=(2,)`` pins only the slot; the update's own shape sizes every other
    axis.  That is what lets ONE call serve the packed layout: an update of
    ``[2, H, 1, D]`` covers both planes at the same slot, which is exactly where
    k and v belong, while the unpacked ``[1, H, 1, D]`` update covers one.

    It is NOT bounds-checked: a start past the end of ``leaf`` drops the write
    silently.  Nothing in the graph can catch that, which is why
    :class:`LagunaCompiledLane` tracks the position on the host and refuses the
    step instead.
    """

    return mx.slice_update(leaf, update, start, axes=(2,))


def next_ring_index(ring_idx: mx.array, window: int) -> mx.array:
    """Advance the shared ring slot, keeping it pre-normalized.

    ``RotatingKVCache`` stores the POST-write index and normalizes lazily
    (``if _idx == max_size: _idx = keep`` at the top of the next update).  Here
    the wrap is applied eagerly instead, so ``ring_idx`` is always a legal slot
    and the step never needs a branch.  The two are the same sequence of write
    positions; only where the modulo happens differs.
    """

    return (ring_idx + 1) % window


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------
def build_step(
    model: Any, cap: int, *, compiled: bool = True, packed_kv: bool = False
) -> Callable[..., tuple[mx.array, ...]]:
    """Build the pure decode step for ``model``.

    The returned callable is::

        step(token, offset, ring_idx, *leaves)
            -> (next_token, offset_next, ring_idx_next, *leaves_next)

    ``offset`` and ``ring_idx`` lead the updated leaves because they lead the
    arguments: the output tuple can be fed straight back in as the next step's
    input, which is what makes the lane a fixed point ICB capture can replay.

    ``compiled=False`` returns the traced-by-Python twin of the exact same body,
    so a divergence between the two is a compilation bug and never a rewrite.

    ``packed_kv`` decides the leaf layout, and with it the arity: one
    ``[2, H, S, D]`` leaf per layer written once, or the keys/values pair
    written twice.  See the module docstring for why one write plus a
    ``mx.where`` beats two writes.  Nothing else about the step changes, so the
    two builds are expected to agree token for token and number for number.

    Every op below mirrors a specific line of ``mtplx/models/laguna.py``; the
    only substitutions are the cache plumbing (explicit leaves instead of
    ``cache.update_and_fetch``), the full-attention mask (an in-graph additive
    mask over the padded leaf instead of ``mask=None`` over an exactly sized
    buffer), and the fused q/k norm+rope kernel below.  Gating, MoE and the
    norms call the shipped module code, so the ``laguna_fused`` installers keep
    applying — including the destructive q/k/v/g concatenation, which the
    projection block reads off the attention module rather than assuming the
    four separate projections still exist.

    The q/k kernel is likewise switched on by an installer rather than by a
    knob of this lane's own: ``laguna_fused.install_kernel_qk_rope`` (env
    ``MTPLX_LAGUNA_KERNEL_QK_ROPE``) leaves a ``_qk_rope_spec`` on every
    attention module, and the step calls
    :func:`~mtplx.kernels.laguna_decode.fused_qk_norm_rope` for every layer
    whose projected shapes that spec covers.  Without the install — or on CPU,
    or at any geometry the kernel does not implement — the stock
    norm/transpose/rope chain runs unchanged.
    """

    geometry = geometry_for(model, cap, packed_kv=packed_kv)
    inner = getattr(model, "model", model)
    layers = list(inner.layers)
    window = geometry.window
    packed = geometry.packed_kv
    tied = bool(model.args.tie_word_embeddings)

    # Loop-invariant constants live OUTSIDE the traced body so they are captured
    # once instead of re-dispatched per step; `mx.compile` treats closed-over
    # arrays as constants, and none of these ever change.
    positions = mx.arange(geometry.cap, dtype=mx.int32)
    admit = mx.array(0.0, dtype=mx.float32)
    reject = mx.array(-float("inf"), dtype=mx.float32)
    plane = kv_plane_mask()
    mx.eval(positions, admit, reject, plane)

    def _attention(
        attn: Any,
        x: mx.array,
        offset: mx.array,
        start: mx.array,
        kv_state: tuple[mx.array, ...],
        mask_for: Callable[[Any], mx.array] | None,
        gate_impl: Callable[..., mx.array],
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        """``Attention.__call__`` at T=1 with the cache replaced by leaves."""

        batch, length, _ = x.shape
        # `laguna_fused.install_fused_qkvg` concatenates q/k/v/g into one
        # projection and DROPS the four originals, so a converted layer has no
        # `q_proj` to call; reading `_qkvg` first is what keeps that install
        # applicable to this lane, exactly as the gate hook below does for the
        # attention gate.
        qkvg = getattr(attn, "_qkvg", None)
        if qkvg is not None:
            queries, keys, values, gate_logits = qkvg(x)
        else:
            queries, keys, values = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
            gate_logits = None

        values = values.reshape(batch, length, attn.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        # `install_kernel_qk_rope` (env MTPLX_LAGUNA_KERNEL_QK_ROPE) attaches a
        # `_qk_rope_spec` to every attention module; its presence is the whole
        # switch, so this lane needs no env var of its own.  The eligibility
        # check wants the PRE-transpose [B, 1, H*D] projections, which is what
        # the block above just produced, and it refuses anything the kernel
        # does not cover exactly — CPU runs and the toy head_dim the tests
        # build both land in the stock chain below.
        spec = getattr(attn, "_qk_rope_spec", None)
        if is_qk_norm_rope_eligible(
            queries, keys, attn.q_norm.weight, attn.k_norm.weight, spec
        ):
            # One dispatch for norm + transpose + rope, on q and k together,
            # instead of the five-plus the chain below costs per layer.  The
            # offset goes in as the ARRAY for the same reason the stock rope
            # takes it that way: a Python int would be a trace constant.
            queries, keys = fused_qk_norm_rope(
                queries,
                keys,
                attn.q_norm.weight,
                attn.k_norm.weight,
                float(attn.q_norm.eps),
                offset,
                spec,
            )
        else:
            queries = attn.q_norm(
                queries.reshape(batch, length, attn.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            keys = attn.k_norm(
                keys.reshape(batch, length, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)

            # The offset goes in as the int32 ARRAY, not `int(offset)`: both
            # `nn.RoPE` and `YarnRoPE` hand it straight to `mx.fast.rope`,
            # which accepts an array, and that is what keeps the position a
            # graph input.  Sliding layers rope at the ABSOLUTE position too —
            # the window bounds what is attended to, never how a key is
            # rotated.
            queries = attn.rope(queries, offset=offset)
            keys = attn.rope(keys, offset=offset)

        if packed:
            # ONE dynamic write for both planes: k and v go to the same slot, so
            # a `[2, H, 1, D]` update at `start` lands each of them where a
            # separate `slice_update` would have.  The planes come back out as
            # unit-stride slices, which cost no dispatch of their own.
            (kv_leaf,) = kv_state
            kv_leaf = kv_slot_write(
                kv_leaf, pack_kv(keys, values, plane), start
            )
            keys, values = unpack_kv(kv_leaf)
            updated: tuple[mx.array, ...] = (kv_leaf,)
        else:
            k_leaf, v_leaf = kv_state
            keys = kv_slot_write(k_leaf, keys, start)
            values = kv_slot_write(v_leaf, values, start)
            updated = (keys, values)

        # Straight to `mx.fast.scaled_dot_product_attention`: the mlx-lm wrapper
        # only branches on a quantized cache, which this lane does not have, and
        # its `mask=None` contract assumes an exactly sized buffer.
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=attn.scale,
            mask=None if mask_for is None else mask_for(queries.dtype),
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch,
            length,
            attn.n_heads * attn.head_dim,
        )

        if attn.gating:
            if gate_logits is None:
                gate_logits = attn.g_proj(x)
            if attn.gate_per_head:
                output = gate_impl(
                    output, gate_logits, attn.n_heads, attn.head_dim
                )
            else:
                gate = mx.logaddexp(
                    gate_logits.astype(mx.float32), mx.array(0.0)
                ).astype(output.dtype)
                output = output * gate

        return attn.o_proj(output), updated

    def step(
        token: mx.array,
        offset: mx.array,
        ring_idx: mx.array,
        *leaves: mx.array,
    ) -> tuple[mx.array, ...]:
        if len(leaves) != geometry.n_leaves:
            raise ValueError(
                f"expected {geometry.n_leaves} leaves, got {len(leaves)}"
            )

        # Read once per trace so an installed fused gate is honoured, exactly
        # like `Attention.__call__` reading the module-level hook.
        gate_impl = laguna.PER_HEAD_GATE_IMPL

        full_start = mx.reshape(offset, (1,))
        ring_start = mx.reshape(ring_idx, (1,))

        # The additive mask admits [0, offset], INCLUDING the key written this
        # step at index `offset` — the eager path gets the same admission for
        # free by fetching `keys[..., : offset + 1, :]`.  Everything past it is
        # zero padding whose scores would otherwise contribute exp(0) each.
        admitted = positions < (offset + 1)
        masks: dict[Any, mx.array] = {}

        def mask_for(dtype: Any) -> mx.array:
            cached = masks.get(dtype)
            if cached is None:
                cached = (
                    mx.where(admitted, admit, reject)
                    .astype(dtype)
                    .reshape(1, 1, 1, geometry.cap)
                )
                masks[dtype] = cached
            return cached

        hidden = inner.embed_tokens(token)
        updated: list[mx.array] = []
        for index, layer in enumerate(layers):
            attn = layer.self_attn
            sliding = geometry.kinds[index] == SLIDING
            attention_out, layer_updated = _attention(
                attn,
                layer.input_layernorm(hidden),
                offset,
                # Sliding layers overwrite the oldest slot; full layers append
                # at the absolute position.
                ring_start if sliding else full_start,
                geometry.layer_leaves(leaves, index),
                # Steady state means every ring slot is live, so sliding needs
                # no mask; softmax does not care that the ring is rotated.
                None if sliding else mask_for,
                gate_impl,
            )
            updated.extend(layer_updated)
            hidden = hidden + attention_out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        output = inner.norm(hidden)
        logits = (
            inner.embed_tokens.as_linear(output) if tied else model.lm_head(output)
        )
        next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]

        ring_next = (
            next_ring_index(ring_idx, window) if window is not None else ring_idx
        )
        return (next_token, offset + 1, ring_next, *updated)

    return mx.compile(step) if compiled else step


# ---------------------------------------------------------------------------
# public lane
# ---------------------------------------------------------------------------
class LagunaCompiledLane:
    """Holds the tensor state and drives :func:`build_step` one token at a time.

    Usage is prefill on the stock path, then hand the caches over::

        cache = model.make_cache()
        logits = model(prompt, cache=cache, logits_keep=1)
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]

        lane = LagunaCompiledLane(model, cap=2048)
        lane.seed(cache, token)
        for _ in range(n):
            token = lane.advance()

    ``advance`` deliberately does NOT evaluate: the caller decides where the
    sync goes (``mx.async_eval`` on the token is the point of the exercise).
    A loop that never evaluates will grow the graph without bound.

    The lane is good for ``cap - offset`` steps.  Past that the full-attention
    write would land outside the leaf, which ``mx.slice_update`` drops silently
    while the mask happily admits the stale padding — so the position is
    mirrored on the host (exact at B=1, where every step is +1) and the step is
    refused rather than allowed to corrupt state.  The mirror costs one sync at
    ``seed`` and none per step.

    ``packed_kv`` (default on) is the leaf layout — one ``[2, H, S, D]`` array
    per layer instead of a keys/values pair, halving both the arity and the
    per-layer cache writes.  It changes what ``state()``, ``capture_inputs()``
    and ``geometry.n_leaves`` describe, and nothing else: ``packed_kv=False``
    rebuilds the original layout for an A/B and generates the same tokens.
    """

    def __init__(
        self,
        model: Any,
        cap: int,
        *,
        compiled: bool = True,
        packed_kv: bool = False,
    ) -> None:
        self.model = model
        self.compiled = bool(compiled)
        self.packed_kv = bool(packed_kv)
        self.geometry = geometry_for(model, cap, packed_kv=packed_kv)
        self.step = build_step(
            model, cap, compiled=compiled, packed_kv=packed_kv
        )
        self.token: mx.array | None = None
        self.offset: mx.array | None = None
        self.ring_idx: mx.array | None = None
        self.leaves: tuple[mx.array, ...] = ()
        self._position = 0

    @property
    def cap(self) -> int:
        return self.geometry.cap

    def seed(self, caches: Sequence[Any], token: mx.array) -> "LagunaCompiledLane":
        """Adopt a post-prefill eager cache list and the token it produced."""

        self.offset, self.ring_idx, self.leaves = snapshot_leaves(
            self.model, caches, self.geometry.cap, packed_kv=self.packed_kv
        )
        self._position = int(self.offset)
        self.token = mx.array(token, dtype=mx.uint32).reshape(1, 1)
        return self

    def advance(self) -> mx.array:
        """Run one step and return the newly sampled ``[1, 1]`` uint32 token."""

        if self.token is None:
            raise ValueError("lane has no state; call seed() after a prefill")
        if self._position >= self.geometry.cap:
            raise ValueError(
                f"the leaves are full at cap {self.geometry.cap}; re-seed from "
                "eager caches with a larger cap"
            )
        outputs = self.step(
            self.token, self.offset, self.ring_idx, *self.leaves
        )
        self.token, self.offset, self.ring_idx = outputs[0], outputs[1], outputs[2]
        self.leaves = tuple(outputs[3:])
        self._position += 1
        return self.token

    def remaining_steps(self) -> int:
        """How many more steps fit before the full-attention leaves overflow.

        Reads the host-side mirror, so it costs nothing and is safe to call
        inside the decode loop.
        """

        if self.token is None:
            raise ValueError("lane has no state; call seed() after a prefill")
        return self.geometry.cap - self._position

    def state(self) -> tuple[mx.array, mx.array, tuple[mx.array, ...]]:
        """The ``(offset, ring_idx, leaves)`` triple, for comparison in tests."""

        return self.offset, self.ring_idx, self.leaves

    def capture_inputs(self) -> tuple[mx.array, ...]:
        """The flat argument tuple :attr:`step` would be called with right now.

        ``state()`` deliberately omits the token, and ``advance`` builds the
        argument list inline — neither is what an ICB capture needs.  Capture
        wants the arguments as ONE flat sequence, in the step's own order, so
        that ``capture_compiled(lane.step, *lane.capture_inputs())`` records the
        stream this lane would actually run next.

        The ordering is the contract, not a convenience: the step is a fixed
        point — its result tuple has the same arity, order, shape and dtype as
        this one — so position ``i`` of the result is the successor of position
        ``i`` here, and a replay feedback plan can blit output ``i`` straight
        into input ``i`` for every leaf, token included.  Anything that reorders
        one side has to reorder the other.
        """

        if self.token is None:
            raise ValueError("lane has no state; call seed() after a prefill")
        return (self.token, self.offset, self.ring_idx, *self.leaves)
