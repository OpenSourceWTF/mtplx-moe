"""Contracts for the pure, compile-able Laguna decode step.

The lane replaces two mutating Python objects — ``KVCache`` and
``RotatingKVCache`` — with explicit tensor leaves, so the things that can break
are exactly: where a key gets written, which keys the softmax is allowed to see,
and whether ``mx.compile`` reproduces the eager trace.  Each is pinned here
separately rather than folded into one end-to-end token comparison, because a
token comparison passes for a while even when a ring is one slot out.

Runs on the CPU device at toy geometry (window 8, 12-token prompt, so the
sliding caches are past the window and in the steady state the lane supports).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from mtplx.laguna_compiled_step import (
    LagunaCompiledLane,
    full_state_from_cache,
    kv_slot_write,
    next_ring_index,
    ring_state_from_cache,
    snapshot_leaves,
)
from mtplx.models import laguna, laguna_fused
from mtplx.models.laguna import Model, ModelArgs

# A float32 delta this far below bf16's ~8e-3 resolution cannot change the
# model's own arithmetic.  The padded-leaf softmax reduces over `cap` terms
# instead of `offset + 1`, so a delta at this scale is expected; a token change
# is not.
BF16_SAFE_TOLERANCE = 1e-5

LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]

# 12 > sliding_window 8, so every sliding cache is in steady state after prefill.
PROMPT = mx.array([[3, 9, 14, 2, 7, 21, 5, 11, 30, 1, 18, 6]], dtype=mx.uint32)
CAP = 32
STEPS = 6


def _toy_args(**updates):
    config = dict(
        model_type="laguna",
        hidden_size=64,
        num_hidden_layers=len(LAYER_TYPES),
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=256,
        rms_norm_eps=1e-6,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[0],
        gating="per-head",
        sliding_window=8,
        layer_types=list(LAYER_TYPES),
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 500_000.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    config.update(updates)
    return ModelArgs(**config)


@pytest.fixture
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def toy_model(cpu_device):
    mx.random.seed(3)
    model = Model(_toy_args())
    mx.eval(model.parameters())
    stock_moe = laguna.LagunaSparseMoeBlock.__call__
    stock_forward = laguna.LagunaModel.__call__
    stock_attention = laguna.Attention.__call__
    try:
        yield model
    finally:
        laguna.LagunaSparseMoeBlock.__call__ = stock_moe
        laguna.LagunaModel.__call__ = stock_forward
        laguna.Attention.__call__ = stock_attention
        laguna.PER_HEAD_GATE_IMPL = laguna._stock_per_head_gate
        laguna.MOE_COMBINE_IMPL = laguna._stock_moe_combine
        laguna_fused.reset_cached_gather_indices()


def _greedy_token(logits):
    return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]


def _prefill(model, prompt=PROMPT):
    """Run the stock python-cache prefill and return ``(caches, first token)``."""

    caches = model.make_cache()
    token = _greedy_token(model(prompt, cache=caches, logits_keep=1))
    mx.eval(token)
    return caches, token


def _stock_step(model, caches, token):
    return _greedy_token(model(token, cache=caches))


def _max_delta(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _identical(a, b):
    return bool(mx.all(a == b))


# ---------------------------------------------------------------------------
# write positions, against the real cache classes
# ---------------------------------------------------------------------------
def test_ring_write_matches_rotating_kv_cache_across_two_wraps(cpu_device):
    """The tensor ring must land every write where the eager cache lands it.

    Both sides are fed the SAME k/v, so equality here is bitwise and is purely a
    statement about slot arithmetic — the one thing a model-level token
    comparison would not catch until the wrong key drifted the logits far enough
    to flip an argmax.
    """

    mx.random.seed(17)
    window = 8
    cache = RotatingKVCache(max_size=window, keep=0)
    # Prefill goes through `_update_concat`, which leaves a 12-long buffer with
    # `_idx == 12` — NOT a ring.  Normalizing that is `ring_state_from_cache`'s
    # job and is exercised here from the state the model actually produces.
    cache.update_and_fetch(
        mx.random.normal((1, 2, window + 4, 4)),
        mx.random.normal((1, 2, window + 4, 4)),
    )
    ring_k, ring_v, slot = ring_state_from_cache(cache, window)
    idx = mx.array(slot, dtype=mx.int32)

    for step in range(2 * window + 3):  # wraps twice with room to spare
        keys = mx.random.normal((1, 2, 1, 4))
        values = mx.random.normal((1, 2, 1, 4))
        cache.update_and_fetch(keys, values)

        start = mx.reshape(idx, (1,))
        ring_k = kv_slot_write(ring_k, keys, start)
        ring_v = kv_slot_write(ring_v, values, start)
        idx = next_ring_index(idx, window)
        mx.eval(ring_k, ring_v, idx, cache.keys, cache.values)

        assert ring_k.shape == cache.keys.shape
        assert _identical(ring_k, cache.keys), f"ring keys diverged at step {step}"
        assert _identical(ring_v, cache.values), f"ring values diverged at step {step}"
        # The eager cache stores the POST-write index and wraps lazily; the leaf
        # wraps eagerly.  Same slot sequence, different place for the modulo.
        assert int(idx) == cache._idx % window


def test_full_write_matches_kv_cache(cpu_device):
    """The padded leaf must hold the eager cache's live prefix and nothing else."""

    mx.random.seed(23)
    cap = 16
    cache = KVCache()
    cache.update_and_fetch(
        mx.random.normal((1, 2, 5, 4)), mx.random.normal((1, 2, 5, 4))
    )
    leaf_k, leaf_v = full_state_from_cache(cache, cap)
    offset = mx.array(int(cache.offset), dtype=mx.int32)

    for step in range(6):
        keys = mx.random.normal((1, 2, 1, 4))
        values = mx.random.normal((1, 2, 1, 4))
        live_k, live_v = cache.update_and_fetch(keys, values)

        start = mx.reshape(offset, (1,))
        leaf_k = kv_slot_write(leaf_k, keys, start)
        leaf_v = kv_slot_write(leaf_v, values, start)
        offset = offset + 1
        mx.eval(leaf_k, leaf_v, offset, live_k, live_v)

        length = int(cache.offset)
        assert _identical(live_k, leaf_k[..., :length, :]), f"keys differ at {step}"
        assert _identical(live_v, leaf_v[..., :length, :]), f"values differ at {step}"
        # Nothing may be written past the live prefix; the additive mask relies
        # on that region staying inert.
        assert _identical(leaf_k[..., length:, :], mx.zeros_like(leaf_k[..., length:, :]))


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------
def test_snapshot_reports_the_shared_offset_and_ring_slot(toy_model):
    caches, _ = _prefill(toy_model)
    offset, ring_idx, leaves = snapshot_leaves(toy_model, caches, CAP)

    assert offset.dtype == mx.int32 and offset.shape == ()
    assert ring_idx.dtype == mx.int32 and ring_idx.shape == ()
    assert int(offset) == PROMPT.shape[1]
    # A 12-token prompt into an 8-slot ring trims to the last 8 and rotates the
    # write slot back to 0 — what `_update_in_place` would do on the next step.
    assert int(ring_idx) == 0

    assert len(leaves) == 2 * len(LAYER_TYPES)
    window = toy_model.args.sliding_window
    for index, kind in enumerate(LAYER_TYPES):
        keys = leaves[2 * index]
        expected = CAP if kind == "full_attention" else window
        assert keys.shape == (1, 2, expected, 8), f"layer {index} leaf shape"


def test_snapshot_does_not_alias_the_eager_cache_buffers(toy_model):
    """The eager caches keep mutating their buffers; the leaves must not follow.

    ``KVCache`` and ``RotatingKVCache`` both write with ``self.keys[..., a:b, :]
    = k``.  Handing out ``cache.keys`` directly would leave the lane's state
    being edited from underneath it by whoever still holds the cache.
    """

    caches, token = _prefill(toy_model)
    lane = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(caches, token)
    frozen = [mx.array(leaf) for leaf in lane.leaves]
    mx.eval(frozen)

    for _ in range(3):
        token = _stock_step(toy_model, caches, token)
    mx.eval(token, *[cache.keys for cache in caches])

    for index, (leaf, expected) in enumerate(zip(lane.leaves, frozen)):
        assert _identical(leaf, expected), f"leaf {index} tracked the eager cache"


def test_snapshot_refuses_a_sliding_cache_below_the_window(toy_model):
    """Short context is a different graph shape, so it is refused, not guessed."""

    short = mx.array([[3, 9, 14]], dtype=mx.uint32)
    caches, _ = _prefill(toy_model, short)
    with pytest.raises(ValueError, match="steady state"):
        snapshot_leaves(toy_model, caches, CAP)


def test_snapshot_refuses_a_cap_with_no_room_left(toy_model):
    caches, _ = _prefill(toy_model)
    with pytest.raises(ValueError, match="no room"):
        snapshot_leaves(toy_model, caches, PROMPT.shape[1])


def test_snapshot_refuses_caches_that_are_not_in_lockstep(toy_model):
    """One shared offset is only sound while every cache advances together."""

    caches, _ = _prefill(toy_model)
    caches[0].offset -= 1
    with pytest.raises(ValueError, match="lockstep"):
        snapshot_leaves(toy_model, caches, CAP)


def test_snapshot_refuses_sliding_caches_with_different_slots(toy_model):
    """One shared ring_idx is only sound while every window writes together."""

    caches, token = _prefill(toy_model)
    _stock_step(toy_model, caches, token)
    sliding = [
        cache for cache, kind in zip(caches, LAYER_TYPES) if kind == "sliding_attention"
    ]
    sliding[0]._idx = (sliding[0]._idx + 1) % toy_model.args.sliding_window
    with pytest.raises(ValueError, match="ring slot"):
        snapshot_leaves(toy_model, caches, CAP)


# ---------------------------------------------------------------------------
# the eager twin against the stock python-cache path
# ---------------------------------------------------------------------------
def test_eager_twin_generates_what_the_stock_path_generates(toy_model):
    """The lane may reduce over a padded leaf; it may not change the tokens."""

    stock_caches, stock_token = _prefill(toy_model)
    lane_caches, lane_token = _prefill(toy_model)
    assert stock_token.tolist() == lane_token.tolist()

    lane = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(
        lane_caches, lane_token
    )

    for step in range(STEPS):
        stock_token = _stock_step(toy_model, stock_caches, stock_token)
        twin_token = lane.advance()
        mx.eval(stock_token, twin_token)
        assert twin_token.tolist() == stock_token.tolist(), (
            f"greedy token diverged at step {step}"
        )


def test_eager_twin_state_tracks_the_eager_caches(toy_model):
    """Per step, the leaves must equal what the eager caches now hold.

    Compared against ``snapshot_leaves`` of the stock caches, which is the same
    normalization the lane was seeded through — so this pins the WRITE POSITIONS
    exactly (a slot off by one moves whole keys and blows past any tolerance)
    while allowing the ulp-scale value drift the padded softmax introduces
    upstream of each layer's projections.
    """

    stock_caches, stock_token = _prefill(toy_model)
    lane_caches, lane_token = _prefill(toy_model)
    lane = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(
        lane_caches, lane_token
    )

    for step in range(STEPS):
        stock_token = _stock_step(toy_model, stock_caches, stock_token)
        lane.advance()
        offset, ring_idx, leaves = lane.state()
        mx.eval(stock_token, offset, ring_idx, *leaves)

        want_offset, want_ring, want_leaves = snapshot_leaves(
            toy_model, stock_caches, CAP
        )
        assert int(offset) == int(want_offset), f"offset drifted at step {step}"
        assert int(ring_idx) == int(want_ring), f"ring slot drifted at step {step}"

        for index, (leaf, want) in enumerate(zip(leaves, want_leaves)):
            assert leaf.shape == want.shape
            delta = _max_delta(leaf, want)
            assert delta <= BF16_SAFE_TOLERANCE, (
                f"leaf {index} diverged at step {step}: {delta:.3e}"
            )

        live = int(offset)
        for index, kind in enumerate(LAYER_TYPES):
            if kind != "full_attention":
                continue
            tail = leaves[2 * index][..., live:, :]
            assert _identical(tail, mx.zeros_like(tail)), (
                f"layer {index} wrote past the live prefix at step {step}"
            )


# ---------------------------------------------------------------------------
# compiled against eager
# ---------------------------------------------------------------------------
def test_compiled_twin_matches_the_eager_twin(toy_model):
    """Compiling the step may not change what it generates.

    The bar is greedy tokens EQUAL and leaves equal to well inside bf16
    resolution — not bitwise.  Bitwise is not available: ``mx.compile``
    substitutes its own implementations of ``mx.fast.rms_norm`` and
    ``mx.fast.rope``, and on the CPU backend those disagree with the eager
    kernels at ~1e-7 on their own, before any model is involved.  That is MLX's
    behaviour for those two primitives and nothing in the step can change it
    without moving off the ops the shipped model uses.  The observed drift here
    is ~1e-6 after six steps, three orders below the ~8e-3 the checkpoint's own
    dtype can represent.
    """

    eager_caches, eager_token = _prefill(toy_model)
    compiled_caches, compiled_token = _prefill(toy_model)
    eager = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(
        eager_caches, eager_token
    )
    compiled = LagunaCompiledLane(toy_model, CAP, compiled=True).seed(
        compiled_caches, compiled_token
    )

    for step in range(STEPS):
        a = eager.advance()
        b = compiled.advance()
        mx.eval(a, b)
        assert b.tolist() == a.tolist(), f"compiled token diverged at step {step}"

    eager_offset, eager_ring, eager_leaves = eager.state()
    offset, ring_idx, leaves = compiled.state()
    mx.eval(offset, ring_idx, *leaves, eager_offset, eager_ring, *eager_leaves)

    # The integer state IS exact: nothing about compilation may move a write.
    assert int(offset) == int(eager_offset)
    assert int(ring_idx) == int(eager_ring)
    for index, (leaf, want) in enumerate(zip(leaves, eager_leaves)):
        assert leaf.shape == want.shape
        delta = _max_delta(leaf, want)
        assert delta <= BF16_SAFE_TOLERANCE, (
            f"compiled leaf {index} diverged from the eager twin: {delta:.3e}"
        )

    # The padding the mask relies on stays untouched under compilation too.
    live = int(offset)
    for index, kind in enumerate(LAYER_TYPES):
        if kind != "full_attention":
            continue
        tail = leaves[2 * index][..., live:, :]
        assert _identical(tail, mx.zeros_like(tail)), (
            f"compiled layer {index} wrote past the live prefix"
        )


# ---------------------------------------------------------------------------
# the configurations the shipped checkpoint actually runs in
# ---------------------------------------------------------------------------
def _assert_lane_tracks_stock(model, *, compiled, steps=STEPS):
    stock_caches, stock_token = _prefill(model)
    lane_caches, lane_token = _prefill(model)
    lane = LagunaCompiledLane(model, CAP, compiled=compiled).seed(
        lane_caches, lane_token
    )
    for step in range(steps):
        stock_token = _stock_step(model, stock_caches, stock_token)
        twin_token = lane.advance()
        mx.eval(stock_token, twin_token)
        assert twin_token.tolist() == stock_token.tolist(), (
            f"greedy token diverged at step {step}"
        )


def test_lane_runs_the_quantized_layout(toy_model):
    """The admitted checkpoint is oQ4e, so the quantized modules are the case.

    Quantization swaps in ``QuantizedEmbedding``, ``QuantizedLinear`` and the
    ``gather_qmm`` expert bank, none of which the float path exercises, and all
    of which have to survive being traced by ``mx.compile``.
    """

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())
    _assert_lane_tracks_stock(toy_model, compiled=True)


def test_lane_handles_the_shipped_yarn_rope(cpu_device):
    """The admitted config ropes full-attention layers with YaRN, factor 128.

    That puts ``mscale`` at 1.485, and ``YarnRoPE.__call__`` implements a
    non-unit mscale with ``x = x[...]`` followed by an item ASSIGNMENT — the one
    mutating construct anywhere on the step's path.  It has to survive being
    traced, so the real rope shape is tested rather than the toy default one.
    """

    real_rope = {
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500_000.0,
            "factor": 128.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
        },
    }
    mx.random.seed(3)
    model = Model(_toy_args(rope_parameters=real_rope))
    mx.eval(model.parameters())
    assert model.model.layers[0].self_attn.rope.mscale != 1.0

    _assert_lane_tracks_stock(model, compiled=True)


def test_lane_handles_tied_word_embeddings(cpu_device):
    """The other ``Model.__call__`` head branch: ``embed_tokens.as_linear``."""

    mx.random.seed(3)
    model = Model(_toy_args(tie_word_embeddings=True))
    mx.eval(model.parameters())
    _assert_lane_tracks_stock(model, compiled=True)


def test_lane_handles_a_model_with_no_sliding_layers(cpu_device):
    """With no window there is no ring, and ``ring_idx`` must stay inert."""

    mx.random.seed(3)
    model = Model(
        _toy_args(
            layer_types=["full_attention"] * len(LAYER_TYPES),
            sliding_window=None,
        )
    )
    mx.eval(model.parameters())

    caches, token = _prefill(model)
    lane = LagunaCompiledLane(model, CAP, compiled=True).seed(caches, token)
    assert lane.geometry.window is None
    for _ in range(3):
        lane.advance()
    mx.eval(lane.token, lane.ring_idx)
    assert int(lane.ring_idx) == 0

    _assert_lane_tracks_stock(model, compiled=True)


def test_lane_goes_through_the_module_level_gate_hook(toy_model):
    """The step must call the shipped hooks, or every laguna_fused install dies.

    Attention, gating and the MoE block are the module code verbatim precisely
    so that ``install_compiled_attention_gate`` and friends keep applying to the
    compiled lane; a bespoke inlined gate would silently opt out of all of them.
    """

    caches, token = _prefill(toy_model)
    calls: list[int] = []

    def probe(output, gate_logits, n_heads, head_dim):
        calls.append(n_heads)
        return laguna._stock_per_head_gate(output, gate_logits, n_heads, head_dim)

    laguna.PER_HEAD_GATE_IMPL = probe  # the fixture restores it
    lane = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(caches, token)
    lane.advance()
    mx.eval(lane.token)

    assert calls == [8] * len(LAYER_TYPES)


def test_compiling_builds_the_graph_once_for_the_whole_run(toy_model):
    """The point of the exercise: Python stops re-tracing per step.

    The eager twin rebuilds the graph every step — that is the ~5.7 ms/step this
    lane exists to remove.  With ``compiled=True`` the body must run exactly once
    no matter how many tokens come out, which is also the precondition for an
    ICB capture: a graph that retraced would be a different graph each step.
    """

    traces: list[int] = []

    def probe(output, gate_logits, n_heads, head_dim):
        traces.append(n_heads)
        return laguna._stock_per_head_gate(output, gate_logits, n_heads, head_dim)

    laguna.PER_HEAD_GATE_IMPL = probe  # the fixture restores it

    eager_caches, eager_token = _prefill(toy_model)
    eager = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(
        eager_caches, eager_token
    )
    traces.clear()  # the prefill forward went through the hook too
    for _ in range(3):
        eager.advance()
    mx.eval(eager.token)
    assert len(traces) == 3 * len(LAYER_TYPES), "the eager twin must retrace"

    compiled_caches, compiled_token = _prefill(toy_model)
    compiled = LagunaCompiledLane(toy_model, CAP, compiled=True).seed(
        compiled_caches, compiled_token
    )
    traces.clear()
    for _ in range(3):
        compiled.advance()
    mx.eval(compiled.token)
    assert len(traces) == len(LAYER_TYPES), (
        f"compiled step traced {len(traces) // len(LAYER_TYPES)} times, want 1"
    )


def test_lane_refuses_to_step_past_the_end_of_its_leaves(toy_model):
    """Overflow must be an error, not a dropped write.

    ``mx.slice_update`` does not bounds-check: a start past the leaf discards
    the key silently, and the mask — which only knows ``offset`` — would then
    admit stale padding in its place.  Nothing in the graph can notice, so the
    lane counts positions on the host and stops.
    """

    cap = PROMPT.shape[1] + 2
    caches, token = _prefill(toy_model)
    lane = LagunaCompiledLane(toy_model, cap, compiled=False).seed(caches, token)

    assert lane.remaining_steps() == 2
    lane.advance()
    assert lane.remaining_steps() == 1
    lane.advance()
    assert lane.remaining_steps() == 0
    with pytest.raises(ValueError, match="leaves are full"):
        lane.advance()


# ---------------------------------------------------------------------------
# the capture surface
# ---------------------------------------------------------------------------
def test_capture_inputs_are_the_arguments_the_lane_would_step_with(toy_model):
    """``capture_inputs()`` must BE the next call's arguments, not a copy of them.

    An ICB capture records the stream produced by evaluating the step on the
    arrays it is handed, and the fork pins exactly those buffers.  If the
    accessor handed back clones, the capture would pin buffers the lane does not
    own and the first replay would read stale state, so identity is the contract
    — and calling the step with them has to land where ``advance`` lands.
    """

    caches, token = _prefill(toy_model)
    lane = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(caches, token)

    inputs = lane.capture_inputs()
    assert inputs[0] is lane.token
    assert inputs[1] is lane.offset
    assert inputs[2] is lane.ring_idx
    assert tuple(inputs[3:]) == lane.leaves

    manual = lane.step(*inputs)
    advanced = lane.advance()
    mx.eval(*manual, advanced)

    assert _identical(manual[0], advanced), "step(*capture_inputs) != advance()"
    offset, ring_idx, leaves = lane.state()
    assert _identical(manual[1], offset)
    assert _identical(manual[2], ring_idx)
    for index, (got, want) in enumerate(zip(manual[3:], leaves)):
        assert _identical(got, want), f"leaf {index} diverged"


def test_capture_inputs_and_the_step_result_are_a_fixed_point(toy_model):
    """Position ``i`` in must match position ``i`` out, in arity, shape and dtype.

    This is what makes an identity feedback plan legal: the fork's
    ``make_feedback_plan`` validates each ``(output i -> input i)`` pair by byte
    size, so a step whose result reordered, resized or re-dtyped any position
    could not advance its own state on-device.  The token pair is the one worth
    naming — it is what removes the last per-cycle host write.
    """

    caches, token = _prefill(toy_model)
    lane = LagunaCompiledLane(toy_model, CAP, compiled=False).seed(caches, token)

    inputs = lane.capture_inputs()
    outputs = lane.step(*inputs)
    mx.eval(*outputs)

    assert len(outputs) == len(inputs) == 3 + lane.geometry.n_leaves
    for index, (before, after) in enumerate(zip(inputs, outputs)):
        assert before.shape == after.shape, f"position {index} changed shape"
        assert before.dtype == after.dtype, f"position {index} changed dtype"
        assert before.nbytes == after.nbytes, f"position {index} changed size"


def test_capture_inputs_refuses_an_unseeded_lane(toy_model):
    """No state means no capture; the accessor says so instead of returning None."""

    lane = LagunaCompiledLane(toy_model, CAP, compiled=False)
    with pytest.raises(ValueError, match="no state"):
        lane.capture_inputs()
