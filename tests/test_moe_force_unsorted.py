"""Regression tests for the B>=4 per-stream sha ROOT-CAUSE FIX.

The B>=4 batched-decode sha failure was greedy batch NON-invariance from the
MoE token-sort switch: :class:`PackedSwitchGLU` flips ``do_sort`` on at
``indices.size >= 64`` (a B>=4 verify has ``16 * B`` routed indices at
``top_k=8``, ``rows=2``: B=2 -> 32 UNSORTED, B=4 -> 64 SORTED), and the sorted
``gather_qmm`` kernel accumulates in a different float order than the unsorted
kernel a B<4 / single stream uses.  Same math, different numerical path -> a
batched stream's committed tokens diverge from the same stream run alone.  The
fix (``MTPLX_A3B_MOE_FORCE_UNSORTED``) pins the batched-decode lane to the
UNSORTED path for every row count.

Two guards live here:

* :func:`test_sentinel_expert_no_row_permutation` -- a float-noise-free proof
  that the sort/unsort round-trip never PERMUTES a row (a distinct failure mode
  the fix does not address but must never regress into); and
* :func:`test_batch_invariance_under_force_unsorted_flag` -- the invariance the
  fix actually buys, checked BITWISE on the default (Metal) device.

Unlike ``tests/test_moe_packed_projections.py`` these do NOT force the CPU
stream: the sorted/unsorted float-accumulation difference is a Metal-kernel
property, so the invariance claim is only meaningful on the default device.
The tensors are tiny (an 8-expert / 128-hidden 4-bit block), well under the
unit-test Metal budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.moe_packed_projections import (
    FORCE_UNSORTED_ENV,
    PackedSwitchGLU,
    _PackedDenseProjection,
    _pack_pair,
    moe_force_unsorted_enabled,
)

pytest.importorskip("mlx_lm.models.qwen3_next")
pytest.importorskip("mlx_lm.models.switch_layers")

HIDDEN = 128
MOE_INTERMEDIATE = 64
SHARED_INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 4
GROUP_SIZE = 64
BITS = 4


@pytest.fixture(autouse=True)
def _fresh_force_unsorted_cache():
    """``moe_force_unsorted_enabled`` is ``functools.cache``d (hot path); clear
    it around every test so each starts from a fresh env read and no cached
    value leaks between the flag-on and flag-off arms."""
    moe_force_unsorted_enabled.cache_clear()
    yield
    moe_force_unsorted_enabled.cache_clear()


def _moe_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE_INTERMEDIATE,
        shared_expert_intermediate_size=SHARED_INTERMEDIATE,
        norm_topk_prob=True,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
    )


def _make_block(quantized: bool = True):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    mx.random.seed(1234)
    block = Qwen3NextSparseMoeBlock(_moe_args())
    if quantized:
        nn.quantize(block, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(block.parameters())
    return block


# --------------------------------------------------------------------------
# (a) Row-permutation guard -- float-noise-free, catches a real permutation at
#     B>=4 forever (a distinct failure mode from the numerical-path bug).
# --------------------------------------------------------------------------
def test_sentinel_expert_no_row_permutation():
    """Expert ``e`` maps ANY input to a distinguishable constant, so a row
    permutation is directly visible.

    Construction: a ZERO packed gate/up projection makes the routed gather
    exactly ``0`` on any device and any sort order (``sum`` of zeros is exactly
    ``0.0``), and ``swiglu(0, 0) == 0``; the only surviving signal is the
    ``down_proj`` per-expert BIAS, gathered by expert id.  With
    ``indices.size >= 64`` the ``_gather_sort`` / ``_scatter_unsort`` round-trip
    is active, yet each output slot must still carry ITS OWN expert's constant.
    A broken round-trip (a genuine row permutation, NOT the float-path bug this
    PR fixes) would surface here.
    """
    from mlx_lm.models.switch_layers import SwiGLU, SwitchLinear

    experts = 64
    hid = 4
    inter = 8
    rows = 4
    top_k = 16  # rows * top_k = 64 -> crosses the do_sort threshold
    assert rows * top_k >= 64

    # ZERO packed gate/up: gather_mm(x, 0) == 0 exactly, any device / any order.
    gate_up = _PackedDenseProjection(mx.zeros((experts, 2 * inter, hid)))
    mx.eval(gate_up["weight"])

    # down_proj weight is multiplied by the (zero) activation, so only its
    # index-gathered bias survives: output slot == sentinel[expert].
    down = SwitchLinear(inter, hid, experts, bias=True)
    sentinel = mx.reshape(
        mx.arange(1, experts * hid + 1, dtype=mx.float32), (experts, hid)
    )
    down.bias = sentinel
    down.weight = mx.zeros((experts, hid, inter))
    mx.eval(down.parameters())

    fused = PackedSwitchGLU(gate_up, down, SwiGLU(), inter)

    x = mx.random.normal((rows, hid))  # arbitrary: expert output is input-free
    indices = mx.reshape(mx.arange(rows * top_k), (rows, top_k))  # all distinct
    out = fused(x, indices)  # [rows, top_k, hid]
    mx.eval(out)

    expected = sentinel[indices]  # pure gather reference: slot -> own expert
    assert out.shape == expected.shape == (rows, top_k, hid)
    assert mx.array_equal(out, expected)


# --------------------------------------------------------------------------
# (b) Batch invariance under the flag -- the property the fix buys, BITWISE on
#     the default (Metal) device.
# --------------------------------------------------------------------------
def test_batch_invariance_under_force_unsorted_flag(monkeypatch):
    """With the flag ON, a rows=4 and a rows=8 call both take the UNSORTED
    gather path (the rows=8 call's 64 indices would otherwise sort), so the
    FIRST 4 rows of the rows=8 output are BITWISE equal to the rows=4 output --
    exactly the batch-invariance the per-stream sha gate needs.

    Without the flag the rows=8 call sorts and the rows=4 call does not, so the
    two kernels' float accumulation is NOT guaranteed bitwise-equal.  On tiny
    random weights the delta may happen to be 0, so that arm is informational
    only (never asserted) to keep the test from flaking either way.
    """
    block = _make_block(quantized=True)  # default device (Metal)
    switch_mlp = block.switch_mlp
    result = _pack_pair(switch_mlp.gate_proj, switch_mlp.up_proj, axis=1)
    assert not isinstance(result, str), result
    packed, split_at = result
    fused = PackedSwitchGLU(
        packed, switch_mlp.down_proj, switch_mlp.activation, split_at
    )

    mx.random.seed(101)
    top_k = 8
    x8 = mx.random.normal((8, HIDDEN))
    # rows differ in their routing so this is a real per-row gather, not a
    # broadcast; all expert ids in [0, NUM_EXPERTS).
    indices8 = (
        mx.arange(8).reshape(8, 1) + mx.arange(top_k).reshape(1, top_k)
    ) % NUM_EXPERTS
    x4 = x8[:4]
    indices4 = indices8[:4]
    assert int(indices4.size) == 32  # < 64 -> unsorted regardless of the flag
    assert int(indices8.size) == 64  # >= 64 -> SORTED unless the flag pins it

    # Flag ON -> both unsorted -> first 4 rows bitwise identical.
    monkeypatch.setenv(FORCE_UNSORTED_ENV, "1")
    moe_force_unsorted_enabled.cache_clear()
    out4 = fused(x4, indices4)
    out8 = fused(x8, indices8)
    mx.eval(out4, out8)
    assert mx.array_equal(out8[:4], out4)  # BITWISE, default device

    # Flag OFF -> rows=8 sorts, rows=4 does not: informational only, no assert.
    monkeypatch.delenv(FORCE_UNSORTED_ENV, raising=False)
    moe_force_unsorted_enabled.cache_clear()
    out4_off = fused(x4, indices4)
    out8_off = fused(x8, indices8)
    mx.eval(out4_off, out8_off)
    off_equal = bool(mx.array_equal(out8_off[:4], out4_off))
    assert off_equal in (True, False)  # documents "may or may not differ"


# --------------------------------------------------------------------------
# Flag plumbing (truthy-set convention, matching batched_decode_enabled).
# --------------------------------------------------------------------------
def test_force_unsorted_flag_parsing(monkeypatch):
    monkeypatch.delenv(FORCE_UNSORTED_ENV, raising=False)
    moe_force_unsorted_enabled.cache_clear()
    assert moe_force_unsorted_enabled() is False

    for truthy in ("1", "true", "YES", "On", "yes"):
        monkeypatch.setenv(FORCE_UNSORTED_ENV, truthy)
        moe_force_unsorted_enabled.cache_clear()
        assert moe_force_unsorted_enabled() is True

    for falsy in ("0", "nope", ""):
        monkeypatch.setenv(FORCE_UNSORTED_ENV, falsy)
        moe_force_unsorted_enabled.cache_clear()
        assert moe_force_unsorted_enabled() is False
