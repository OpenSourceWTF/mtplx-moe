"""Correctness gates for the ragged batch KV lane (R1 + R2).

Gates (scheme doc S6 / build spec):

1. Degenerate equivalence  -- uniform offsets/advances => bitwise == scalar lane
   (both through the real ``attention_split.split_call`` path and a minimal toy
   stack), for prefill-shaped [B,T] and decode-shaped [B,2] calls.
2. Ragged correctness / per-row independence -- changing row j never changes
   row i's output, bitwise.
3. Rewrite-in-place (REPLAY precondition) -- rewriting stale positions yields
   bitwise the same K/V/attention as never having written them.
4. Per-row masked restore -- some rows revert, others keep advancing, bitwise.
5. Per-row RoPE regression -- array offset == per-row scalar loop, bitwise
   (banked as a regression against MLX upgrades).
6. (existing suite + ruff, run separately)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.base import scaled_dot_product_attention as sdpa
from mlx_lm.models.cache import create_causal_mask

from mtplx.attention_split import _install_split_attention_hook
from mtplx.cache_state import (
    TailOwnedKVCache,
    snapshot_cache,
    restore_cache,
)
from mtplx.ragged_attention import ragged_additive_mask, ragged_causal_mask
from mtplx.ragged_kv_cache import (
    RaggedBatchKVCache,
    restore_ragged_caches,
    restore_ragged_caches_masked,
    snapshot_ragged_caches,
)


def _bitwise_equal(a, b) -> bool:
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    if a.shape != b.shape:
        return False
    return bool(mx.all(a == b).item())


def _diff(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    n = int((a != b).sum().item())
    m = float(mx.abs(a - b).max().item()) if a.size else 0.0
    return n, m


# ---------------------------------------------------------------------------
# Gate 5: per-row RoPE regression (banked first -- everything else leans on it)
# ---------------------------------------------------------------------------


def test_gate5_nn_rope_array_offset_matches_scalar_loop():
    mx.random.seed(1)
    B, H, T, D = 4, 3, 2, 8
    rope = nn.RoPE(dims=D, traditional=False, base=10000.0)
    x = mx.random.normal((B, H, T, D))
    offs = mx.array([3, 7, 0, 11])
    out_arr = rope(x, offset=offs)
    rows = [rope(x[r : r + 1], offset=int(offs[r].item())) for r in range(B)]
    out_loop = mx.concatenate(rows, axis=0)
    assert _bitwise_equal(out_arr, out_loop), _diff(out_arr, out_loop)


def test_gate5_nn_rope_uniform_array_matches_scalar_int():
    mx.random.seed(2)
    B, H, T, D = 4, 3, 2, 8
    rope = nn.RoPE(dims=D, traditional=False, base=10000.0)
    x = mx.random.normal((B, H, T, D))
    L = 5
    out_int = rope(x, offset=L)
    out_arr = rope(x, offset=mx.array([L] * B))
    assert _bitwise_equal(out_int, out_arr), _diff(out_int, out_arr)


def test_gate5_mx_fast_rope_array_offset_regression():
    # MLX-upgrade regression: mx.fast.rope must apply a [B] offset per row.
    mx.random.seed(3)
    B, H, T, D = 4, 2, 2, 8
    x = mx.random.normal((B, H, T, D))
    offs = mx.array([2, 9, 0, 4])
    out = mx.fast.rope(x, D, traditional=False, base=10000.0, scale=1.0, offset=offs)
    rows = [
        mx.fast.rope(
            x[r : r + 1], D, traditional=False, base=10000.0, scale=1.0,
            offset=int(offs[r].item()),
        )
        for r in range(B)
    ]
    out_loop = mx.concatenate(rows, axis=0)
    assert _bitwise_equal(out, out_loop), _diff(out, out_loop)


# ---------------------------------------------------------------------------
# Mask builder: fixed-shape discipline
# ---------------------------------------------------------------------------


def test_mask_shape_and_dtype_invariant_to_offset_content():
    q_len, key_len = 2, 16
    m0 = ragged_causal_mask(mx.array([0, 0, 0, 0]), q_len, key_len)
    m1 = ragged_causal_mask(mx.array([3, 7, 0, 11]), q_len, key_len)
    m2 = ragged_causal_mask(mx.array([15, 14, 1, 9]), q_len, key_len)
    for m in (m0, m1, m2):
        assert m.shape == (4, 1, q_len, key_len)
        assert m.dtype == mx.bool_
    # content differs, shape/dtype do not
    assert not _bitwise_equal(m0.astype(mx.int32), m1.astype(mx.int32))


def test_mask_causal_semantics_per_row():
    # row b query i (abs pos start[b]+i) attends key j iff j <= start[b]+i
    start = mx.array([2, 5])
    q_len, key_len = 2, 10
    m = ragged_causal_mask(start, q_len, key_len)
    for b, s in enumerate([2, 5]):
        for i in range(q_len):
            for j in range(key_len):
                assert bool(m[b, 0, i, j].item()) == (j <= s + i)


def test_additive_mask_matches_boolean():
    start = mx.array([3, 7, 0, 5])
    m_bool = ragged_causal_mask(start, 2, 12)
    m_add = ragged_additive_mask(start, 2, 12)
    assert m_add.shape == m_bool.shape
    # allowed -> 0, disallowed -> -inf
    assert bool(mx.all((m_add == 0.0) == m_bool).item())
    assert bool(mx.all((m_add == float("-inf")) == (~m_bool)).item())


# ---------------------------------------------------------------------------
# A minimal toy attention stack (decoupled from split_call) for gates 1B/2/3
# ---------------------------------------------------------------------------


def _sdpa_attention(queries, keys, values, *, scale, mask):
    return sdpa(queries, keys, values, cache=None, scale=scale, mask=mask)


def _ragged_layer(queries_hd, k_hd, v_hd, cache: RaggedBatchKVCache, *, rope, scale):
    """One ragged attention layer given pre-projected [B,H,q,d] tensors."""
    q_start = cache.offset  # pre-update per-row offsets (append semantics)
    q = rope(queries_hd, offset=q_start)
    k = rope(k_hd, offset=q_start)
    keys, values = cache.update_and_fetch(k, v_hd)
    key_len = int(keys.shape[2])
    mask = ragged_causal_mask(q_start, int(q.shape[2]), key_len)
    return _sdpa_attention(q, keys, values, scale=scale, mask=mask)


def _scalar_layer(queries_hd, k_hd, v_hd, cache: TailOwnedKVCache, *, rope, scale):
    """Same layer on the scalar batch-generic cache."""
    prev = int(cache.offset)
    q = rope(queries_hd, offset=prev)
    k = rope(k_hd, offset=prev)
    keys, values = cache.update_and_fetch(k, v_hd)
    mask = create_causal_mask(int(q.shape[2]), prev)  # [q, prev+q]
    return _sdpa_attention(q, keys, values, scale=scale, mask=mask)


# ---------------------------------------------------------------------------
# Gate 1B: degenerate equivalence through the minimal toy stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shapes", [("prefill", 6), ("decode", 2)])
def test_gate1b_degenerate_equivalence_toy_stack(shapes):
    kind, T = shapes
    mx.random.seed(10)
    B, H, Hkv, D = 4, 4, 2, 8
    n_layers = 3
    scale = D ** -0.5
    rope = nn.RoPE(dims=D, traditional=False, base=10000.0)
    prev_offset = 5  # uniform starting offset for all rows

    # identical random inputs per layer for both lanes
    q_in = [mx.random.normal((B, H, T, D)) for _ in range(n_layers)]
    k_in = [mx.random.normal((B, Hkv, T, D)) for _ in range(n_layers)]
    v_in = [mx.random.normal((B, Hkv, T, D)) for _ in range(n_layers)]
    # identical prefilled history for both lanes (uniform offset)
    hist_k = [mx.random.normal((B, Hkv, prev_offset, D)) for _ in range(n_layers)]
    hist_v = [mx.random.normal((B, Hkv, prev_offset, D)) for _ in range(n_layers)]

    scalar_out, ragged_out = [], []
    for li in range(n_layers):
        sc = TailOwnedKVCache(step=64)
        sc.keys = hist_k[li] + 0.0
        sc.values = hist_v[li] + 0.0
        sc.offset = prev_offset
        scalar_out.append(_scalar_layer(q_in[li], k_in[li], v_in[li], sc, rope=rope, scale=scale))

        rg = RaggedBatchKVCache(step=64)
        # seed the ragged buffer with identical history and uniform offsets
        rg.keys = hist_k[li] + 0.0
        rg.values = hist_v[li] + 0.0
        rg.offsets = mx.array([prev_offset] * B, dtype=mx.int32)
        ragged_out.append(_ragged_layer(q_in[li], k_in[li], v_in[li], rg, rope=rope, scale=scale))

    for li in range(n_layers):
        n, m = _diff(scalar_out[li], ragged_out[li])
        assert n == 0, f"{kind} layer {li}: {n} elems differ, max_abs={m}"


# ---------------------------------------------------------------------------
# Gate 1A: degenerate equivalence through the REAL split_call attention path
# ---------------------------------------------------------------------------


class GatedToyAttn(nn.Module):
    """Minimal Qwen3Next-style gated attention compatible with split_call."""

    def __init__(self, dim, num_heads, num_kv, head_dim):
        super().__init__()
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, 2 * num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.rope = nn.RoPE(dims=head_dim, traditional=False, base=10000.0)

    def __call__(self, x, mask=None, cache=None):  # replaced by split_call hook
        raise AssertionError("split_call hook not installed")


def _make_gated_attn(seed):
    mx.random.seed(seed)
    dim, num_heads, num_kv, head_dim = 32, 4, 2, 8
    attn = GatedToyAttn(dim, num_heads, num_kv, head_dim)
    mx.eval(attn.parameters())
    _install_split_attention_hook(attn)
    attn._mtplx_split_full_attention_enabled = True
    attn._mtplx_split_full_attention_explicit_enabled = False
    attn._mtplx_blockwise_full_attention_enabled = False
    attn._mtplx_sdpa_2pass_enabled = False
    attn._mtplx_vllm_metal_paged_enabled = False
    attn._mtplx_gqa_packed_sdpa_enabled = False
    attn._mtplx_full_attention_index = 0
    return attn, dim


@pytest.mark.parametrize("T", [6, 2])
def test_gate1a_degenerate_equivalence_through_split_call(T):
    # Uniform-offset equivalence through the REAL attention_split.split_call:
    # scalar cache (trimmed buffer + causal mask) vs ragged cache (fixed-capacity
    # zero-padded buffer + explicit per-row mask). The mask is built before the
    # call, so the ragged buffer is pre-sized to a fixed cohort capacity CAP and
    # the mask uses key_len=CAP -- no grow happens inside update_and_fetch.
    attn, dim = _make_gated_attn(seed=21)
    B, Hkv, D = 4, 2, 8
    prev = 5
    CAP = 16
    x = mx.random.normal((B, T, dim))
    hist_k = mx.random.normal((B, Hkv, prev, D))
    hist_v = mx.random.normal((B, Hkv, prev, D))

    # scalar lane
    sc = TailOwnedKVCache(step=64)
    sc.keys, sc.values, sc.offset = hist_k + 0.0, hist_v + 0.0, prev
    scalar_mask = create_causal_mask(T, prev)
    out_scalar = attn(x, mask=scalar_mask, cache=sc)

    # ragged lane, uniform offsets, fixed capacity CAP (no grow)
    rg = RaggedBatchKVCache(step=64)
    rg_k = mx.zeros((B, Hkv, CAP, D))
    rg_v = mx.zeros((B, Hkv, CAP, D))
    rg_k[:, :, :prev, :] = hist_k
    rg_v[:, :, :prev, :] = hist_v
    rg.keys, rg.values = rg_k, rg_v
    rg.offsets = mx.array([prev] * B, dtype=mx.int32)
    ragged_mask = ragged_causal_mask(rg.offset, T, CAP)
    out_ragged = attn(x, mask=ragged_mask, cache=rg)

    n, m = _diff(out_scalar, out_ragged)
    assert n == 0, f"T={T}: {n} elems differ, max_abs={m}"


# ---------------------------------------------------------------------------
# Gate 2: ragged correctness / per-row independence
# ---------------------------------------------------------------------------


def _make_ragged_state(offsets, H, D, cap, seed):
    """Build (keys, values) buffers with random valid data per row, zeros beyond."""
    mx.random.seed(seed)
    B = len(offsets)
    k = mx.zeros((B, H, cap, D))
    v = mx.zeros((B, H, cap, D))
    for r, L in enumerate(offsets):
        if L > 0:
            k[r, :, :L, :] = mx.random.normal((H, L, D))
            v[r, :, :L, :] = mx.random.normal((H, L, D))
    return k, v


def _run_ragged_decode(offsets, q_in, k_in, v_in, hist_k, hist_v, cap, *, rope, scale):
    rg = RaggedBatchKVCache(step=cap)
    rg.keys = hist_k + 0.0
    rg.values = hist_v + 0.0
    rg.offsets = mx.array(list(offsets), dtype=mx.int32)
    return _ragged_layer(q_in, k_in, v_in, rg, rope=rope, scale=scale)


def test_gate2_per_row_independence():
    mx.random.seed(30)
    B, H, Hkv, D = 4, 4, 2, 8
    cap, T = 24, 2
    scale = D ** -0.5
    rope = nn.RoPE(dims=D, traditional=False, base=10000.0)
    offsets = [3, 7, 0, 11]

    q_in = mx.random.normal((B, H, T, D))
    k_in = mx.random.normal((B, Hkv, T, D))
    v_in = mx.random.normal((B, Hkv, T, D))
    hist_k, hist_v = _make_ragged_state(offsets, Hkv, D, cap, seed=31)

    out_ref = _run_ragged_decode(offsets, q_in, k_in, v_in, hist_k, hist_v, cap, rope=rope, scale=scale)

    # Perturb row 1: change its history content AND its offset.
    hist_k2 = hist_k + 0.0
    hist_v2 = hist_v + 0.0
    hist_k2[1, :, :, :] = mx.random.normal((Hkv, cap, D))
    hist_v2[1, :, :, :] = mx.random.normal((Hkv, cap, D))
    offsets2 = [3, 5, 0, 11]  # only row 1 changes
    # perturb row 1's decode input too
    q_in2 = q_in + 0.0
    k_in2 = k_in + 0.0
    v_in2 = v_in + 0.0
    q_in2[1] = mx.random.normal((H, T, D))
    k_in2[1] = mx.random.normal((Hkv, T, D))
    v_in2[1] = mx.random.normal((Hkv, T, D))

    out_pert = _run_ragged_decode(offsets2, q_in2, k_in2, v_in2, hist_k2, hist_v2, cap, rope=rope, scale=scale)

    # every row EXCEPT 1 must be bitwise unchanged
    for r in (0, 2, 3):
        n, m = _diff(out_ref[r], out_pert[r])
        assert n == 0, f"row {r} changed when only row 1 was perturbed: {n} elems, max_abs={m}"
    # row 1 did change (sanity that the perturbation was real)
    assert _diff(out_ref[1], out_pert[1])[0] > 0


def test_gate2_rows_attend_only_their_own_history():
    # A row at offset 0 (empty history) must attend only to its own new tokens;
    # a row at offset L attends to its L history + new tokens. Verify against a
    # single-row reference computed independently.
    mx.random.seed(32)
    H, Hkv, D = 4, 2, 8
    cap, T = 20, 2
    scale = D ** -0.5
    rope = nn.RoPE(dims=D, traditional=False, base=10000.0)
    offsets = [0, 6, 3]
    B = len(offsets)

    q_in = mx.random.normal((B, H, T, D))
    k_in = mx.random.normal((B, Hkv, T, D))
    v_in = mx.random.normal((B, Hkv, T, D))
    hist_k, hist_v = _make_ragged_state(offsets, Hkv, D, cap, seed=33)
    out = _run_ragged_decode(offsets, q_in, k_in, v_in, hist_k, hist_v, cap, rope=rope, scale=scale)

    # independent single-row reference for each row
    for r, L in enumerate(offsets):
        sc = TailOwnedKVCache(step=cap)
        sc.keys = hist_k[r : r + 1, :, :L, :] + 0.0 if L > 0 else None
        sc.values = hist_v[r : r + 1, :, :L, :] + 0.0 if L > 0 else None
        sc.offset = L
        ref = _scalar_layer(
            q_in[r : r + 1], k_in[r : r + 1], v_in[r : r + 1], sc, rope=rope, scale=scale
        )
        n, m = _diff(out[r : r + 1], ref)
        assert n == 0, f"row {r} (offset {L}) != single-row reference: {n} elems, max_abs={m}"


# ---------------------------------------------------------------------------
# Gate 3: rewrite-in-place (REPLAY precondition)
# ---------------------------------------------------------------------------


def test_gate3_rewrite_in_place_bitwise_equivalent():
    mx.random.seed(40)
    H, D = 2, 8
    cap = 16
    L = 6  # clean row reaches offset L-1 then writes at (L-1, L)

    # row 0: "clean" -- offset L-1, history 0..L-2, then writes (X,Y) at (L-1,L)
    # row 1: "stale" -- offset L, history 0..L-1 where pos L-1 is junk, then
    #        REPLAY-writes the same (X,Y) at (L-1, L)
    hist = mx.random.normal((H, L - 1, D))  # shared history positions 0..L-2
    junk_k = mx.random.normal((H, 1, D))
    junk_v = mx.random.normal((H, 1, D))
    X_k = mx.random.normal((1, H, 1, D))
    Y_k = mx.random.normal((1, H, 1, D))
    X_v = mx.random.normal((1, H, 1, D))
    Y_v = mx.random.normal((1, H, 1, D))
    new_k = mx.concatenate([X_k, Y_k], axis=2)  # [1,H,2,D]
    new_v = mx.concatenate([X_v, Y_v], axis=2)

    rg = RaggedBatchKVCache(step=cap)
    keys = mx.zeros((2, H, cap, D))
    vals = mx.zeros((2, H, cap, D))
    # row 0 history 0..L-2
    keys[0, :, : L - 1, :] = hist
    vals[0, :, : L - 1, :] = mx.random.normal((H, L - 1, D))
    # row 1 history 0..L-2 identical to row 0, plus stale junk at L-1
    keys[1, :, : L - 1, :] = hist
    vals[1, :, : L - 1, :] = vals[0, :, : L - 1, :]
    keys[1, :, L - 1 : L, :] = junk_k
    vals[1, :, L - 1 : L, :] = junk_v
    rg.keys = keys
    rg.values = vals
    rg.offsets = mx.array([L - 1, L], dtype=mx.int32)

    both_new_k = mx.concatenate([new_k, new_k], axis=0)  # [2,H,2,D]
    both_new_v = mx.concatenate([new_v, new_v], axis=0)
    # both rows REPLAY-write at position L-1
    out_k, out_v = rg.update_and_fetch(
        both_new_k, both_new_v, write_start=mx.array([L - 1, L - 1])
    )

    # both rows now bitwise-identical over their logical range [0, L+1)
    assert int(rg.offsets[0].item()) == L + 1
    assert int(rg.offsets[1].item()) == L + 1
    n, m = _diff(out_k[0:1, :, : L + 1, :], out_k[1:2, :, : L + 1, :])
    assert n == 0, f"keys diverge after rewrite: {n} elems, max_abs={m}"
    n, m = _diff(out_v[0:1, :, : L + 1, :], out_v[1:2, :, : L + 1, :])
    assert n == 0, f"values diverge after rewrite: {n} elems, max_abs={m}"

    # attention outputs of the two rows must be bitwise identical for identical
    # queries (the stale write left no trace)
    scale = D ** -0.5
    q = mx.random.normal((1, H, 2, D))
    q2 = mx.concatenate([q, q], axis=0)
    mask = ragged_causal_mask(mx.array([L - 1, L - 1]), 2, int(out_k.shape[2]))
    out = sdpa(q2, out_k, out_v, cache=None, scale=scale, mask=mask)
    n, m = _diff(out[0:1], out[1:2])
    assert n == 0, f"attention diverges after rewrite: {n} elems, max_abs={m}"


# ---------------------------------------------------------------------------
# Gate 4: per-row masked restore
# ---------------------------------------------------------------------------


def test_gate4_per_row_masked_restore():
    mx.random.seed(50)
    B, H, D = 4, 2, 8
    cap = 16
    offsets = [4, 6, 2, 5]
    keys, vals = _make_ragged_state(offsets, H, D, cap, seed=51)
    rg = RaggedBatchKVCache(step=cap)
    rg.keys, rg.values = keys, vals
    rg.offsets = mx.array(offsets, dtype=mx.int32)

    snap = rg.snapshot()
    pre_keys = rg.keys + 0.0
    pre_vals = rg.values + 0.0
    pre_offsets = rg.offsets + 0

    # advance ALL rows by writing 2 new tokens each (append)
    new_k = mx.random.normal((B, H, 2, D))
    new_v = mx.random.normal((B, H, 2, D))
    rg.update_and_fetch(new_k, new_v)
    adv_keys = rg.keys + 0.0
    adv_vals = rg.values + 0.0
    adv_offsets = rg.offsets + 0

    # restore rows 0 and 2, keep rows 1 and 3 advanced
    mask = mx.array([True, False, True, False])
    rg.restore_masked(snap, mask)

    for r in (0, 2):
        assert int(rg.offsets[r].item()) == int(pre_offsets[r].item())
        n, _ = _diff(rg.keys[r, :, : cap, :], pre_keys[r])
        assert n == 0, f"row {r} keys not reverted"
        n, _ = _diff(rg.values[r, :, : cap, :], pre_vals[r])
        assert n == 0, f"row {r} values not reverted"
    for r in (1, 3):
        assert int(rg.offsets[r].item()) == int(adv_offsets[r].item())
        n, _ = _diff(rg.keys[r], adv_keys[r])
        assert n == 0, f"row {r} keys wrongly reverted"
        n, _ = _diff(rg.values[r], adv_vals[r])
        assert n == 0, f"row {r} values wrongly reverted"


def test_gate4_masked_restore_cache_list_helpers():
    mx.random.seed(52)
    B, H, D = 3, 2, 8
    offsets = [2, 4, 1]
    keys, vals = _make_ragged_state(offsets, H, D, 12, seed=53)
    rg = RaggedBatchKVCache(step=12)
    rg.keys, rg.values, rg.offsets = keys, vals, mx.array(offsets, dtype=mx.int32)
    cache = [rg]
    snaps = snapshot_ragged_caches(cache)
    pre = rg.offsets + 0
    rg.update_and_fetch(mx.random.normal((B, H, 2, D)), mx.random.normal((B, H, 2, D)))
    restore_ragged_caches_masked(cache, snaps, mx.array([True, False, True]))
    assert int(rg.offsets[0].item()) == int(pre[0].item())
    assert int(rg.offsets[1].item()) == int(pre[1].item()) + 2
    assert int(rg.offsets[2].item()) == int(pre[2].item())
    # whole-batch restore returns everything
    restore_ragged_caches(cache, snaps)
    assert rg.offsets.tolist() == offsets


# ---------------------------------------------------------------------------
# R1 mechanics: scatter / advance / filter / merge / snapshot_cache compat
# ---------------------------------------------------------------------------


def test_update_appends_at_per_row_offsets_no_python_loop():
    mx.random.seed(60)
    B, H, D = 4, 2, 8
    rg = RaggedBatchKVCache(step=8)
    rg.offsets = mx.array([1, 3, 0, 5], dtype=mx.int32)
    rg.keys = mx.zeros((B, H, 8, D))
    rg.values = mx.zeros((B, H, 8, D))
    nk = mx.random.normal((B, H, 2, D))
    nv = mx.random.normal((B, H, 2, D))
    k, v = rg.update_and_fetch(nk, nv)
    for r, s in enumerate([1, 3, 0, 5]):
        assert _bitwise_equal(k[r, :, s : s + 2, :], nk[r])
        assert _bitwise_equal(v[r, :, s : s + 2, :], nv[r])
        assert int(rg.offsets[r].item()) == s + 2


def test_per_row_advance_differs_via_new_offsets():
    # adv[r] in {0,1,2}: write 2 tokens but commit fewer logical positions
    mx.random.seed(61)
    B, H, D = 3, 2, 8
    rg = RaggedBatchKVCache(step=8)
    rg.offsets = mx.array([2, 2, 2], dtype=mx.int32)
    rg.keys = mx.zeros((B, H, 8, D))
    rg.values = mx.zeros((B, H, 8, D))
    nk = mx.random.normal((B, H, 2, D))
    nv = mx.random.normal((B, H, 2, D))
    # row0 advance 2 (append), row1 replay advance 1, row2 stall advance 0
    rg.update_and_fetch(
        nk, nv,
        write_start=mx.array([2, 1, 0]),
        new_offsets=mx.array([4, 3, 2]),
    )
    assert rg.offsets.tolist() == [4, 3, 2]


def test_grow_preserves_existing_content():
    mx.random.seed(62)
    B, H, D = 2, 2, 8
    rg = RaggedBatchKVCache(step=4)
    rg.offsets = mx.array([3, 2], dtype=mx.int32)
    rg.keys = mx.zeros((B, H, 4, D))
    rg.values = mx.zeros((B, H, 4, D))
    seed_k = mx.random.normal((B, H, 3, D))
    rg.keys[:, :, :3, :] = seed_k
    # write near the end forcing a grow beyond cap=4
    nk = mx.random.normal((B, H, 2, D))
    nv = mx.random.normal((B, H, 2, D))
    k, _ = rg.update_and_fetch(nk, nv, write_start=mx.array([3, 4]))
    assert int(k.shape[2]) >= 6
    # original seeded content preserved
    assert _bitwise_equal(k[:, :, :3, :], seed_k)


def test_filter_keeps_selected_rows():
    mx.random.seed(63)
    H, D = 2, 8
    offs = [2, 4, 1, 5]
    keys, vals = _make_ragged_state(offs, H, D, 8, seed=64)
    rg = RaggedBatchKVCache(step=8)
    rg.keys, rg.values, rg.offsets = keys, vals, mx.array(offs, dtype=mx.int32)
    rg.filter(mx.array([0, 2]))
    assert rg.offsets.tolist() == [2, 1]
    assert _bitwise_equal(rg.keys[0], keys[0])
    assert _bitwise_equal(rg.keys[1], keys[2])


def test_extend_merges_cohorts():
    mx.random.seed(65)
    H, D = 2, 8
    a = RaggedBatchKVCache(step=4)
    a.keys, a.values = mx.random.normal((2, H, 4, D)), mx.random.normal((2, H, 4, D))
    a.offsets = mx.array([2, 3], dtype=mx.int32)
    b = RaggedBatchKVCache(step=8)
    b.keys, b.values = mx.random.normal((1, H, 8, D)), mx.random.normal((1, H, 8, D))
    b.offsets = mx.array([5], dtype=mx.int32)
    a.extend(b)
    assert a.offsets.tolist() == [2, 3, 5]
    assert int(a.keys.shape[0]) == 3
    assert int(a.keys.shape[2]) == 8  # padded to the larger cap


def test_snapshot_cache_roundtrip_compat():
    # the generic cache_state.snapshot_cache / restore_cache must handle a
    # ragged entry via its state / meta_state properties without crashing
    mx.random.seed(66)
    B, H, D = 3, 2, 8
    offs = [2, 4, 1]
    keys, vals = _make_ragged_state(offs, H, D, 8, seed=67)
    rg = RaggedBatchKVCache(step=8)
    rg.keys, rg.values, rg.offsets = keys, vals, mx.array(offs, dtype=mx.int32)
    cache = [rg]
    snap = snapshot_cache(cache)
    rg.update_and_fetch(mx.random.normal((B, H, 2, D)), mx.random.normal((B, H, 2, D)))
    assert rg.offsets.tolist() == [4, 6, 3]
    restore_cache(cache, snap)
    assert rg.offsets.tolist() == offs
    assert _bitwise_equal(rg.keys, keys)


def test_trim_whole_batch_parity():
    rg = RaggedBatchKVCache(step=8)
    rg.offsets = mx.array([5, 8, 6], dtype=mx.int32)
    rg.keys = mx.zeros((3, 2, 8, 8))
    rg.values = mx.zeros((3, 2, 8, 8))
    trimmed = rg.trim(3)
    assert trimmed == 3
    assert rg.offsets.tolist() == [2, 5, 3]


def test_from_scalar_cache_degenerate_seed():
    sc = TailOwnedKVCache(step=64)
    sc.keys = mx.random.normal((4, 2, 5, 8))
    sc.values = mx.random.normal((4, 2, 5, 8))
    sc.offset = 5
    rg = RaggedBatchKVCache.from_scalar_cache(sc)
    assert rg.offsets.tolist() == [5, 5, 5, 5]
    assert _bitwise_equal(rg.keys, sc.keys)
    # host capacity bound seeded from the (host-known) scalar offset -- EXACT, not
    # the step-rounded physical capacity (which would over-grow every cycle).
    assert rg._capacity_bound == 5


# ---------------------------------------------------------------------------
# Item 3 (fable-main review): host-side monotone capacity bound -- update_and_fetch
# performs ZERO device syncs when a host bound is seeded (a device-valued
# write_start must not block the pipelined fold-in forward).
# ---------------------------------------------------------------------------


class _MaxSpy:
    """Wrap ``mx.max`` to prove ``update_and_fetch`` never calls it for capacity."""

    def __init__(self, monkeypatch):
        self.calls = 0
        self._real = mx.max
        monkeypatch.setattr(mx, "max", self)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._real(*args, **kwargs)


def test_capacity_bound_zero_device_sync_when_seeded(monkeypatch):
    # A seeded host bound => update_and_fetch grows off the Python int only and
    # never reads the device offsets (no mx.max(...).item()).  The device-valued
    # write_start below is exactly the fold-in loop's blocking hazard.
    B, H, D = 4, 2, 8
    rg = RaggedBatchKVCache(step=64)
    rg.keys = mx.zeros((B, H, 32, D))
    rg.values = mx.zeros((B, H, 32, D))
    rg.offsets = mx.array([5, 5, 5, 5], dtype=mx.int32)
    rg._capacity_bound = 5  # seed the host bound (as the fold-in make-cache does)
    spy = _MaxSpy(monkeypatch)
    # SPEC append + a REPLAY write at offset-1, both device-valued write_start.
    rg.update_and_fetch(
        mx.random.normal((B, H, 2, D)),
        mx.random.normal((B, H, 2, D)),
        write_start=mx.array([5, 4, 5, 5]),  # row 1 replays at offset-1
    )
    assert spy.calls == 0, "seeded host bound must not read device offsets"
    assert rg._capacity_bound == 7  # += q, monotone
    assert rg.ragged_grows == 0  # capacity (32) sufficed => no allocation, no sync


def test_capacity_bound_legacy_path_unchanged(monkeypatch):
    # With NO host bound (default None), the legacy single device read runs -- the
    # byte-identical pre-fix behaviour every existing caller relies on.
    B, H, D = 3, 2, 8
    rg = RaggedBatchKVCache(step=8)
    rg.keys = mx.zeros((B, H, 8, D))
    rg.values = mx.zeros((B, H, 8, D))
    rg.offsets = mx.array([1, 3, 0], dtype=mx.int32)
    assert rg._capacity_bound is None
    spy = _MaxSpy(monkeypatch)
    rg.update_and_fetch(mx.random.normal((B, H, 2, D)), mx.random.normal((B, H, 2, D)))
    assert spy.calls == 1  # legacy path reads max(offsets) exactly once
    assert rg.offsets.tolist() == [3, 5, 2]


def test_capacity_bound_monotone_grows_off_python_int():
    # Repeated appends grow the buffer purely off _capacity_bound; a REPLAY write
    # at offset-1 never needs more capacity than a SPEC append already reserved.
    B, H, D = 2, 2, 8
    rg = RaggedBatchKVCache.from_scalar_cache(
        TailOwnedKVCache(step=4), batch_size=B
    )  # fresh: offsets 0, bound 0
    assert rg._capacity_bound == 0
    rg.keys = mx.zeros((B, H, 0, D))
    rg.values = mx.zeros((B, H, 0, D))
    for cycle in range(6):
        rg.update_and_fetch(
            mx.random.normal((B, H, 2, D)), mx.random.normal((B, H, 2, D))
        )
        assert rg._capacity_bound == 2 * (cycle + 1)
        assert int(rg.keys.shape[2]) >= rg._capacity_bound


def test_reserve_pregrows_so_no_grow_inside_update():
    # reserve() before the forward makes keys.shape[2] final so make_mask's
    # key_len matches update_and_fetch's output capacity (mask/keys consistency).
    B, H, D = 4, 2, 8
    rg = RaggedBatchKVCache(step=16)
    rg.keys = mx.zeros((B, H, 16, D))
    rg.values = mx.zeros((B, H, 16, D))
    rg.offsets = mx.array([14, 14, 14, 14], dtype=mx.int32)
    rg._capacity_bound = 14
    rg.reserve(2)  # 14 + 2 = 16 <= cap 16 => no grow yet
    assert int(rg.keys.shape[2]) == 16
    cap_before = int(rg.keys.shape[2])
    mask_key_len = int(rg.make_mask(2).shape[-1])
    keys, _ = rg.update_and_fetch(
        mx.random.normal((B, H, 2, D)), mx.random.normal((B, H, 2, D))
    )
    assert int(keys.shape[2]) == cap_before  # no grow inside update
    assert mask_key_len == int(keys.shape[2])  # mask key_len matches K buffer
    # next cycle crosses the step boundary: reserve grows, make_mask still matches.
    rg.reserve(2)  # 16 + 2 = 18 > 16 => grow to 32
    assert int(rg.keys.shape[2]) == 32
    assert int(rg.make_mask(2).shape[-1]) == 32


def test_make_mask_create_attention_mask_signature():
    # create_attention_mask calls cache.make_mask(N, return_array=..., window_size=..).
    rg = RaggedBatchKVCache(step=16)
    rg.keys = mx.zeros((3, 2, 16, 8))
    rg.values = mx.zeros((3, 2, 16, 8))
    rg.offsets = mx.array([2, 5, 0], dtype=mx.int32)
    m = rg.make_mask(2, return_array=True, window_size=None)
    assert m.shape == (3, 1, 2, 16)
    assert m.dtype == mx.bool_
    from mlx_lm.models.base import create_attention_mask

    # the real seam: create_attention_mask delegates to make_mask when present.
    mixed = mx.zeros((3, 2, 8))  # [B, N=2, H]
    delegated = create_attention_mask(mixed, rg)
    assert delegated.shape == (3, 1, 2, 16)
    assert _bitwise_equal(delegated.astype(mx.int32), m.astype(mx.int32))
