"""Fused Metal kernels for the two Laguna decode blocks that are all overhead.

The component census on the pinned oQ4e checkpoint (M5 Max, ctx 1024, batch 1)
put 21.1% of a decode step in the MoE router and 12.2% in the attention per-head
gate — a third of the step spent on arithmetic over 256 and 48 floats
respectively.  Neither is doing real work; both are chains of ten-plus tiny
elementwise kernels whose cost is launch and latency, repeated 47 and 48 times
per step.

So these two kernels do not try to beat MLX at matmul, which it would win.  They
leave every matmul on the stock path and collapse only the epilogue around it:

``fused_router_topk``
    sigmoid -> add correction bias -> top-k select -> gather the unbiased
    scores -> normalize -> scale, as one threadgroup per row.  Replaces roughly
    eleven dispatches with one.

``fused_per_head_gate``
    softplus of the gate logits, broadcast across each head's slice, in one
    pass.  Replaces four dispatches with one.

The softplus reproduces MLX's own ``LogAddExp`` (max + log1p(exp(min - max)),
with the infinity short-circuit) so the gate is bit-identical to the shipped
expression rather than merely close.

The router's top-k cannot be bit-identical by construction: ``argpartition``
leaves the order of the selected indices unspecified, and the normalizing sum
therefore accumulates in an order this kernel has no way to reproduce.  Ties
are broken toward the lower expert index here.  Callers get the divergence
measured, not assumed.

``fused_router_gemv_topk`` is the one exception to "leave every matmul alone",
and it is opt-in for exactly that reason: the router's own gemv is small enough
that MLX spends more on launching it than on running it, so it is taken over by
a kernel of ours that also absorbs the bf16 -> f32 cast of the logits — at the
price of a different fp32 reduction order and therefore an INEXACT-config path.
See its own note below.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# MLX's Metal LogAddExp, specialized to logaddexp(x, 0) — the shipped softplus.
_SOFTPLUS = """
    inline float mtplx_softplus(float x) {
        if (metal::isnan(x)) {
            return metal::numeric_limits<float>::quiet_NaN();
        }
        constexpr float inf = metal::numeric_limits<float>::infinity();
        float maxval = metal::max(x, 0.0f);
        float minval = metal::min(x, 0.0f);
        if (minval == -inf || maxval == inf) {
            return maxval;
        }
        // log1p is unqualified here on purpose: MLX's own LogAddExp calls it
        // that way, and metal:: has no such member in the JIT context.
        return maxval + log1p(metal::exp(minval - maxval));
    }
"""


# The fused router wins where the stock chain's cost is launch overhead and
# loses where it is not.  Measured on the pinned checkpoint (ctx 1024): +2.3% of
# the whole step at B=1 and +2.5% at B=2, but -3.5% at B=8, because the stock
# chain barely gets more expensive with rows (20.6 us at B=1 vs 18.8 us at B=8)
# while this kernel's ten serial reduction rounds do.  So it is row-gated rather
# than sold as a uniform win.
DEFAULT_ROUTER_MAX_ROWS = 4


def _router_max_rows() -> int:
    import os

    raw = os.environ.get("MTPLX_LAGUNA_KERNEL_ROUTER_MAX_ROWS")
    if raw is None:
        return DEFAULT_ROUTER_MAX_ROWS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_ROUTER_MAX_ROWS


def _on_metal_device() -> bool:
    """Metal being AVAILABLE is not the same as being the device in use.

    `mx.fast.metal_kernel` raises "Only supports the GPU" when the default
    device is the CPU, which happens on any CPU-device run (tests, fallbacks).
    Checking availability alone let an ineligible run reach the kernel and
    crash instead of taking the stock path.
    """

    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def _router_selection_shape_ok(experts: int, top_k: int) -> bool:
    """The expert/top-k bounds the SELECTION epilogue is compiled for.

    One thread per expert, the selection scratch sized at compile time, and the
    tree reduction stepping ``experts / 2`` downward.  Shared by both router
    entry points so the two can never disagree about what they cover.
    """

    return 0 < top_k <= 32 and 32 <= experts <= 1024 and (experts % 32) == 0


def is_router_eligible(logits: mx.array, bias: mx.array, top_k: int) -> bool:
    if not _on_metal_device():
        return False
    if logits.ndim != 2 or bias.ndim != 1:
        return False
    if logits.dtype != mx.float32 or bias.dtype != mx.float32:
        return False
    experts = int(logits.shape[1])
    if experts != int(bias.shape[0]):
        return False
    if int(logits.shape[0]) > _router_max_rows():
        return False
    return _router_selection_shape_ok(experts, top_k)


# The selection scratch and the select/normalize/scale epilogue, shared by both
# router entry points rather than copied into each.  Sharing the SOURCE is what
# makes "the gemv variant runs the same selection" a fact instead of a claim:
# there is one tie-break, one accumulation order for the normalizing sum, and
# one output contract, and neither kernel can drift from the other.
_ROUTER_SELECT_DECLS = """
        threadgroup float tg_score[NUM_EXPERTS];
        threadgroup float tg_choice[NUM_EXPERTS];
        threadgroup float red_val[NUM_EXPERTS];
        threadgroup uint  red_idx[NUM_EXPERTS];
        threadgroup uint  sel_idx[TOP_K];
        threadgroup float sel_score[TOP_K];
"""

# Entered with `score` (the sigmoid of this thread's logit) already in hand.
_ROUTER_SELECT_EPILOGUE = """
        tg_score[lid] = score;
        tg_choice[lid] = score + correction_bias[lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TOP_K; ++k) {
            red_val[lid] = tg_choice[lid];
            red_idx[lid] = lid;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = NUM_EXPERTS / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    float mine = red_val[lid];
                    float theirs = red_val[lid + stride];
                    uint mine_idx = red_idx[lid];
                    uint their_idx = red_idx[lid + stride];
                    // Ties resolve toward the lower expert index so the
                    // selection is at least deterministic run to run.
                    if (theirs > mine || (theirs == mine && their_idx < mine_idx)) {
                        red_val[lid] = theirs;
                        red_idx[lid] = their_idx;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                sel_idx[k] = red_idx[0];
                sel_score[k] = tg_score[red_idx[0]];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lid == sel_idx[k]) {
                tg_choice[lid] = -metal::numeric_limits<float>::infinity();
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            float total = 0.0f;
            for (uint k = 0; k < TOP_K; ++k) {
                total += sel_score[k];
            }
            float inv = (total == 0.0f) ? 0.0f : (1.0f / total);
            for (uint k = 0; k < TOP_K; ++k) {
                indices[row * TOP_K + k] = sel_idx[k];
                weights[row * TOP_K + k] =
                    normalize ? (sel_score[k] * inv * scale)
                              : (sel_score[k] * scale);
            }
        }
"""


@lru_cache(maxsize=None)
def _router_kernel(experts: int, top_k: int):
    header = _SOFTPLUS + f"""
        using namespace metal;
        constant constexpr int NUM_EXPERTS = {experts};
        constant constexpr int TOP_K = {top_k};
    """

    source = (
        """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
"""
        + _ROUTER_SELECT_DECLS
        + """
        float logit = logits[row * NUM_EXPERTS + lid];
        float score = 1.0f / (1.0f + metal::exp(-logit));
"""
        + _ROUTER_SELECT_EPILOGUE
    )

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_router_e{experts}_k{top_k}",
        input_names=["logits", "correction_bias", "scale", "normalize"],
        output_names=["indices", "weights"],
        header=header,
        source=source,
    )


def fused_router_topk(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool,
    scale: float,
) -> tuple[mx.array, mx.array]:
    """Return ``(indices, weights)`` for one MoE routing decision.

    Falls back to the stock op chain on any shape the kernel does not cover, so
    callers can switch it on without also owning a correctness branch.
    """

    if not is_router_eligible(logits, correction_bias, top_k):
        scores = mx.sigmoid(logits)
        choice = scores + correction_bias
        indices = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if normalize:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return indices, weights * scale

    rows, experts = int(logits.shape[0]), int(logits.shape[1])
    kernel = _router_kernel(experts, top_k)
    indices, weights = kernel(
        inputs=[logits, correction_bias, float(scale), bool(normalize)],
        grid=(experts * rows, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, weights


# ---------------------------------------------------------------------------
# the router gemv taken over from MLX, in two phases
# ---------------------------------------------------------------------------
#
# The kernel above leaves the routing matmul on the stock path, which is the
# right call for anything MLX can do well.  The router's gemv is not that: at
# [rows, 3072] x [3072, 256] in unquantized bfloat16 it reads 1.5 MB and does
# 786k MACs, and per-command ICB attribution puts it at 48.5 us against a
# ~17-20 us floor for a kernel that does nothing at all.  Most of that number is
# launch and occupancy, not arithmetic, and it is paid 47 times per decode step
# on a TRUE serial link — plus a second dispatch for the bf16 -> f32 cast of the
# logits, 47 more.  A kernel that writes float32 logits directly removes the
# cast outright and gets to choose its own decomposition for the matmul.
#
# WHAT THE FIRST ATTEMPT MEASURED, because it decided this layout.  The first
# version did the whole thing in ONE dispatch: one threadgroup per row, one
# THREAD per expert, each thread streaming its own contiguous 6 KB weight row
# against a staged copy of x.  It was correct — see the digest note below — and
# at model scale it cost +2.0 ms/step on the compiled lane (17.25/17.41 ms
# against 15.28/15.53 ms stock anchors).  A threadgroup runs on ONE GPU core, so
# that layout pulled all 1.5 MB of routing weight through one core while the
# rest of the GPU had nothing to do, and the layer's next kernel had to wait for
# it — 47 times per step.
#
# WHICH LANE SAYS SO.  The micro-benchmark had missed that, and the reason is
# worth more than the fix.  It was not cache residency: cycling 47 distinct
# weights (74 MB, past the SLC) still prices v1 at 0.4 us/layer FASTER than the
# reference.  The queued lane runs 200 INDEPENDENT iterations, so the GPU
# overlaps them and hides the fact that any one dispatch occupies a single core.
# A decode step cannot overlap them — every router dispatch sits between two
# pieces of dependent work — so the lane that predicts is one where each
# iteration depends on the previous: no host sync, one command buffer, a true
# chain.  That lane puts v1 at +41.5 us/layer, i.e. +1.95 ms/step against a
# window that measured +2.0.  `check_router_gemv_distinct` runs both lanes and
# prints them side by side for exactly that reason.
#
# MEMORY PATTERN.  So the matmul is its own kernel, decomposed for latency
# rather than for the fusion: ONE threadgroup per (row, expert) — 256 of them
# per row instead of 1 — with the expert's row split across the threadgroup's 8
# simdgroups, and each simdgroup's 32 lanes walking its slice in blocks of four.
# So a lane takes k = 4l .. 4l+3 and strides 128 within its slice, the simdgroup
# asks for 128 CONSECUTIVE weights (256 bytes) per step, `simd_sum` collapses
# each simdgroup's partial dot in registers, and the 8 partials meet in 32 bytes
# of threadgroup memory.  The unroll by four is v1's dot loop unchanged; all that
# moved is which lane owns which group of four, and it is still why the width
# must be a multiple of four.  The slices are ceil-partitioned, so a width that
# is a multiple of four but not of 32 still works: the last simdgroup gets a
# short slice, and an empty one contributes a deterministic zero.
#
# Measured on the chained lane at rows=1, against the three-dispatch reference:
# this layout is -2.6 to -3.7 us/layer across sessions (-0.12 to -0.17 ms/step
# over 47 layers), and every configuration from 2 to 16 simdgroups per expert
# lands within 0.4 us of it, while NOT splitting the row at all (one simdgroup
# per expert, 16 experts per threadgroup) is only break even.  Staging x in
# threadgroup memory first was also measured and dropped: x is 6 KB and stays in
# cache, so the barrier costs more than the re-reads it saves.  At rows=4 the
# same kernel is -38 us/layer, because MLX's matmul leaves its gemv path above
# one row and the reference arm quadruples.
#
# The second phase is `fused_router_topk`, unchanged — it already takes float32
# logits and does sigmoid, correction bias, top-k, normalize and scale in one
# dispatch.  Two dispatches replace the stock three (gemv, astype, selection),
# and the selection half of the contract is the SAME SOURCE the selection-only
# kernel runs, not a copy of it.
#
# EXACTNESS.  This path is NOT bit-exact against the stock chain and is not
# meant to be.  The dot is fp32, lane-strided inside each of the eight slices,
# each slice finished by a simd_sum tree and the eight partials added in
# ascending slice order: deterministic run to run, but a different ORDER from
# MLX's, and deliberately so — chasing gemv's blocking would give up the point.
# Callers who need bit-exactness use `fused_router_topk` and leave the matmul
# where it is.
#
# What keeps the divergence to ulp scale rather than something much bigger is
# the rounding.  The stock chain is `nn.Linear` on a bfloat16 input and a
# bfloat16 weight, so its logits come out BFLOAT16 and only then widen to float32
# — that widening cast is the dispatch this path absorbs.  Carrying a full
# float32 dot straight into the sigmoid would therefore be a different NUMBER,
# not a different last bit: bfloat16 resolves to about 4e-3 relative, and through
# the sigmoid's <= 1/4 slope that is a ~1e-3 shift in every routing score, which
# would move far more selections than reassociation ever could.  So the dot is
# rounded to T before it leaves the kernel, reproducing the precision the stock
# gemv would have produced.  What is left is the fp32 ordering difference (~1e-6
# relative over 3072 terms) surviving only where it straddles a bfloat16 rounding
# boundary — rare, and one ulp when it happens.
#
# Rare has now been counted rather than hoped at: comparing this kernel's logits
# against the stock gemv's over 47 distinct weights, ONE logit in 12032 lands a
# bfloat16 ulp away at rows=1, and 8 to 13 in 48128 at rows=4.  None of them
# moved a selection — `check_router_gemv` and `check_router_gemv_distinct` see
# zero flips over 50 draws at the real shape, with a per-expert weight delta of
# 7e-9, below the 1e-6 the SELECTION-only kernel already diverges by since its
# normalizing sum reassociates too.  The guarded window said the same thing where
# it counts: with this rounding contract in place the end-to-end token digest
# MATCHED the stock lane's, so nothing the model emitted moved.  It stays
# classified inexact — a straddled logit demonstrably happens, and on a near-tie
# one of them can reorder two experts.  The classification is a statement about
# the guarantee, not about the observed delta.

# The widest router row this decomposition is characterized at.  Nothing in the
# kernel breaks above it — the slices are strides, not unrolls — but at 4096 each
# of the eight slices is already 128 blocks of four for one float of output, and
# a wider router would want more simdgroups on the row rather than this shape
# stretched.  The gate keeps an uncharacterized width on the stock chain instead
# of guessing that the same split still holds there.
MAX_ROUTER_GEMV_DIMS = 4096

# One threadgroup per (row, expert), 8 simdgroups of 32 lanes sharing that
# expert's row.  Nothing here has to divide the expert count, so the geometry
# cannot disagree with the selection gate.  The simdgroup/lane split is
# arithmetic rather than `thread_index_in_simdgroup` because a 1-D threadgroup
# assigns simdgroup s the thread indices [32s, 32s + 32) by definition, and this
# way the kernel needs one attribute instead of three.
_ROUTER_GEMV_SIMD = 32
_ROUTER_GEMV_SPLIT = 8
_ROUTER_GEMV_THREADS = _ROUTER_GEMV_SIMD * _ROUTER_GEMV_SPLIT


def is_router_gemv_eligible(
    x: mx.array, gate_weight: mx.array, bias: mx.array, top_k: int
) -> bool:
    """Whether the fused-gemv router covers this exact shape.

    Deliberately narrow: bfloat16 only (what the oQ4e routers are — ``mlp.gate``
    carries no quantization entry), 2-D only, and the same row gate and
    expert/top-k bounds the selection-only kernel uses.
    """

    if not _on_metal_device():
        return False
    if x.ndim != 2 or gate_weight.ndim != 2 or bias.ndim != 1:
        return False
    if x.dtype != mx.bfloat16 or gate_weight.dtype != mx.bfloat16:
        return False
    if bias.dtype != mx.float32:
        return False
    rows, dims = int(x.shape[0]), int(x.shape[1])
    experts = int(gate_weight.shape[0])
    if int(gate_weight.shape[1]) != dims:
        return False
    if experts != int(bias.shape[0]):
        return False
    # The dot walks the row in blocks of four, and eight simdgroups share it, so
    # the width also has to stay inside the one this split is characterized at.
    if dims <= 0 or (dims % 4) != 0 or dims > MAX_ROUTER_GEMV_DIMS:
        return False
    if rows <= 0 or rows > _router_max_rows():
        return False
    return _router_selection_shape_ok(experts, top_k)


@lru_cache(maxsize=None)
def _router_gemv_logits_kernel(experts: int, dims: int):
    header = f"""
        using namespace metal;
        constant constexpr int NUM_EXPERTS = {experts};
        constant constexpr int DIMS = {dims};
        constant constexpr int SIMD = {_ROUTER_GEMV_SIMD};
        constant constexpr int SPLIT = {_ROUTER_GEMV_SPLIT};
        constant constexpr int BLOCKS = DIMS / 4;
        constant constexpr int PER_PART = (BLOCKS + SPLIT - 1) / SPLIT;
    """

    source = """
        uint tg = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint part = lid / uint(SIMD);
        uint lane = lid - part * uint(SIMD);

        uint row = tg / uint(NUM_EXPERTS);
        uint expert = tg - row * uint(NUM_EXPERTS);

        threadgroup float partials[SPLIT];

        // x is one short row and every threadgroup wants all of it, so it is
        // read straight from global memory: it was written by the kernel just
        // before this one and is in cache, and staging it in threadgroup memory
        // measured slower than the re-reads it saves.
        //
        // `auto`, not `const device T*`: MLX hands a small enough input to the
        // kernel in the CONSTANT address space instead of device, and a pointer
        // declared with an address space it did not pick fails to compile.  At
        // the real shape both are device; at the smallest shape the gate admits,
        // x is constant.  Inferring costs nothing and covers both.
        auto x_row = x + (size_t)row * (size_t)DIMS;
        auto w_row = gate_weight + (size_t)expert * (size_t)DIMS;

        // This simdgroup's slice of the expert's row, walked in blocks of four:
        // at every step its 32 lanes want 128 CONSECUTIVE weights, which is a
        // handful of wide reads rather than 32 scattered ones.  The slice is
        // ceil-sized, so the last one may be short and a surplus one empty.
        uint kbeg = part * uint(PER_PART) + lane;
        uint kend = metal::min((part + 1u) * uint(PER_PART), uint(BLOCKS));
        float acc = 0.0f;
        for (uint b = kbeg; b < kend; b += uint(SIMD)) {
            uint k = b * 4u;
            acc += static_cast<float>(w_row[k]) * static_cast<float>(x_row[k]);
            acc += static_cast<float>(w_row[k + 1u]) *
                   static_cast<float>(x_row[k + 1u]);
            acc += static_cast<float>(w_row[k + 2u]) *
                   static_cast<float>(x_row[k + 2u]);
            acc += static_cast<float>(w_row[k + 3u]) *
                   static_cast<float>(x_row[k + 3u]);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            partials[part] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            // Ascending slice order, so the sum is one fixed order run to run.
            float total = 0.0f;
            for (uint p = 0; p < uint(SPLIT); ++p) {
                total += partials[p];
            }
            // The stock gemv writes T and the stock chain then widens it to
            // float, so round HERE — carrying the raw float dot forward would be
            // a different number, not a different last bit.  See the note above.
            float logit = static_cast<float>(static_cast<T>(total));
            logits[(size_t)row * (size_t)NUM_EXPERTS + (size_t)expert] = logit;
        }
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_router_gemv_logits_e{experts}_d{dims}",
        input_names=["x", "gate_weight"],
        output_names=["logits"],
        header=header,
        source=source,
    )


def router_gemv_logits(x: mx.array, gate_weight: mx.array) -> mx.array:
    """The routing matmul alone: float32 logits at bfloat16 precision.

    Callers check :func:`is_router_gemv_eligible` first — there is no fallback
    here, because the only caller that needs one is
    :func:`fused_router_gemv_topk` and it owns the stock path.  The float32
    output is what makes the stock chain's separate widening cast unnecessary;
    the VALUE is still a bfloat16 one, rounded exactly where the stock gemv
    rounds.
    """

    rows, dims = int(x.shape[0]), int(x.shape[1])
    experts = int(gate_weight.shape[0])
    kernel = _router_gemv_logits_kernel(experts, dims)
    (logits,) = kernel(
        inputs=[x, gate_weight],
        template=[("T", x.dtype)],
        grid=(_ROUTER_GEMV_THREADS * experts * rows, 1, 1),
        threadgroup=(_ROUTER_GEMV_THREADS, 1, 1),
        output_shapes=[(rows, experts)],
        output_dtypes=[mx.float32],
    )
    return logits


def fused_router_gemv_topk(
    x: mx.array,
    gate_weight: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool,
    scale: float,
) -> tuple[mx.array, mx.array]:
    """Routing matmul AND the routing decision, in two dispatches.

    ``x`` is the flattened hidden state ``[rows, dims]`` and ``gate_weight`` is
    the router's ``nn.Linear`` weight ``[experts, dims]`` — the layout mlx-lm
    stores, so no transpose happens anywhere.  There is no softcap here: the
    shipped checkpoint pins ``moe_router_logit_softcapping`` to 0.0 (see
    ``models/laguna_config.py``), and the caller keeps a softcapped block on the
    stock path rather than this kernel growing a branch it can never take.

    Returns the same ``(indices, weights)`` contract as
    :func:`fused_router_topk`: uint32 ``[rows, top_k]`` and float32
    ``[rows, top_k]``.  Falls back to the two-step stock chain on any shape the
    kernel does not cover, so callers can switch it on without owning a
    correctness branch — but see the exactness note above: where the kernel DOES
    run, it is an inexact-config path.

    Eligible or not, the tail is the same call, so the two branches can only
    differ in where the logits came from.
    """

    if not is_router_gemv_eligible(x, gate_weight, correction_bias, top_k):
        logits = (x @ gate_weight.swapaxes(-1, -2)).astype(mx.float32)
    else:
        logits = router_gemv_logits(x, gate_weight)
    return fused_router_topk(
        logits, correction_bias, top_k, normalize=normalize, scale=scale
    )


def is_per_head_gate_eligible(
    output: mx.array, gate_logits: mx.array, n_heads: int, head_dim: int
) -> bool:
    if not _on_metal_device():
        return False
    if output.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if gate_logits.dtype != output.dtype:
        return False
    if output.ndim != 3 or gate_logits.ndim != 3:
        return False
    if int(output.shape[-1]) != n_heads * head_dim:
        return False
    if int(gate_logits.shape[-1]) != n_heads:
        return False
    return output.shape[:2] == gate_logits.shape[:2]


@lru_cache(maxsize=None)
def _per_head_gate_kernel(n_heads: int, head_dim: int):
    header = _SOFTPLUS + f"""
        using namespace metal;
        constant constexpr int N_HEADS = {n_heads};
        constant constexpr int HEAD_DIM = {head_dim};
    """

    source = """
        uint index = thread_position_in_grid.x;
        uint width = N_HEADS * HEAD_DIM;
        uint row = index / width;
        uint within = index - row * width;
        uint head = within / HEAD_DIM;

        float logit = static_cast<float>(gate_logits[row * N_HEADS + head]);
        // Cast the softplus back to the tensor dtype BEFORE multiplying, which
        // is what the shipped expression does.
        T gate = static_cast<T>(mtplx_softplus(logit));
        gated[index] = attention_output[index] * gate;
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_head_gate_h{n_heads}_d{head_dim}",
        input_names=["attention_output", "gate_logits"],
        output_names=["gated"],
        header=header,
        source=source,
    )


def fused_per_head_gate(
    output: mx.array, gate_logits: mx.array, n_heads: int, head_dim: int
) -> mx.array:
    """Softplus the per-head gate logits and scale each head's slice by it."""

    if not is_per_head_gate_eligible(output, gate_logits, n_heads, head_dim):
        batch, length, _ = output.shape
        gate = mx.logaddexp(
            gate_logits.astype(mx.float32), mx.array(0.0)
        ).astype(output.dtype)
        return (
            output.reshape(batch, length, n_heads, head_dim) * gate[..., None]
        ).reshape(batch, length, -1)

    batch, length, width = output.shape
    total = batch * length * width
    kernel = _per_head_gate_kernel(n_heads, head_dim)
    threadgroup = 256 if total >= 256 else 32
    (gated,) = kernel(
        inputs=[output, gate_logits],
        template=[("T", output.dtype)],
        grid=(total, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(batch, length, width)],
        output_dtypes=[output.dtype],
    )
    return gated


# ---------------------------------------------------------------------------
# fused q/k RMSNorm + rope for the single-token decode step
# ---------------------------------------------------------------------------
#
# The census put 1.21 ms/step in rope and 0.35 ms in q/k norm + transpose —
# arithmetic on 48x128 and 8x128 elements, spread over ~200 dispatches per step
# (two norms, two ropes, and on the YaRN layers a copy plus a sliced scalar
# multiply per projection).  This kernel does the whole chain for q AND k in
# ONE dispatch per layer, writing directly into the head-major layout sdpa
# reads.
#
# Bit-exactness is by construction, matching each stock stage's exact math:
#   - RMSNorm reproduces MLX's rms_single_row at axis 128: one 32-lane
#     simdgroup, four sequential squares per lane in float, simd_sum, and
#     precise::rsqrt(acc/axis + eps); the output expression is
#     w[i] * static_cast<T>(x[i] * inv), the same double rounding.
#   - The YaRN pre-scale reproduces mlx-lm's `mscale * x[..., :dims]`: the
#     scalar is rounded to the tensor dtype first, then multiplied in that
#     dtype, and it touches only the rotated dims.
#   - The rotation reproduces mx.fast.rope: theta = float(offset) * inv_freq
#     with inv_freq either exp2(-d * log2(base)) at d = p / (dims/2) or the
#     module's own float32 freqs buffer, metal::fast::cos/sin, pairs (p,
#     p + dims/2), float multiply-add, one cast back to T per element.
#
# Two callers, one switch.  `laguna_fused.install_kernel_qk_rope` (env
# MTPLX_LAGUNA_KERNEL_QK_ROPE) attaches a `_qk_rope_spec` to every attention
# module; the eager forward and `mtplx.laguna_compiled_step` both read that
# spec and call this kernel when `is_qk_norm_rope_eligible` passes.  Neither
# has an env var of its own — the presence of the spec IS the switch, so a run
# that did not install the path keeps the stock chain in both lanes.

_QK_HEAD_DIM = 128
_QK_LANES = 32


@dataclass(frozen=True)
class QkRopeSpec:
    """Per-layer constants for the fused q/k norm+rope kernel.

    Captured from the layer's own rope module at install time so the kernel
    can only ever see the exact frequencies and scale the stock path uses.
    """

    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    rot_dims: int
    freqs: Optional[mx.array]  # float32 [rot_dims // 2], or None for base form
    base_log2: Optional[float]  # log2(rope theta) when freqs is None
    mscale: Optional[float]  # YaRN attention factor, None when 1.0


def is_qk_norm_rope_eligible(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    spec: "QkRopeSpec | None",
) -> bool:
    if spec is None or not _on_metal_device():
        return False
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return False
    if keys.dtype != queries.dtype:
        return False
    if q_weight.dtype != queries.dtype or k_weight.dtype != queries.dtype:
        return False
    if spec.head_dim != _QK_HEAD_DIM or spec.rot_dims not in (64, 128):
        return False
    if spec.freqs is None and spec.base_log2 is None:
        return False
    if spec.freqs is not None:
        if spec.freqs.dtype != mx.float32:
            return False
        if int(spec.freqs.size) != spec.rot_dims // 2:
            return False
    if queries.ndim != 3 or keys.ndim != 3:
        return False
    if int(queries.shape[1]) != 1 or int(keys.shape[1]) != 1:
        return False
    if int(queries.shape[-1]) != spec.n_q_heads * spec.head_dim:
        return False
    if int(keys.shape[-1]) != spec.n_kv_heads * spec.head_dim:
        return False
    if int(q_weight.size) != spec.head_dim or int(k_weight.size) != spec.head_dim:
        return False
    return int(queries.shape[0]) == int(keys.shape[0])


@lru_cache(maxsize=None)
def _qk_norm_rope_kernel(
    n_q_heads: int,
    n_kv_heads: int,
    rot_dims: int,
    use_freqs: bool,
    base_log2: float,
    mscale: float,
):
    has_mscale = mscale != 1.0
    header = f"""
        using namespace metal;
        constant constexpr int HQ = {n_q_heads};
        constant constexpr int HKV = {n_kv_heads};
        constant constexpr int HEAD_DIM = {_QK_HEAD_DIM};
        constant constexpr int ROT_DIMS = {rot_dims};
        constant constexpr int HALF_ROT = ROT_DIMS / 2;
        constant constexpr bool USE_FREQS = {"true" if use_freqs else "false"};
        constant constexpr bool HAS_MSCALE = {"true" if has_mscale else "false"};
        constant constexpr float MSCALE_F = {mscale!r}f;
        constant constexpr float BASE_LOG2 = {base_log2!r}f;
    """

    source = """
        uint tg = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        constexpr int TOTAL_HEADS = HQ + HKV;

        uint b = tg / uint(TOTAL_HEADS);
        uint hg = tg - b * uint(TOTAL_HEADS);
        bool is_q = hg < uint(HQ);
        uint h = is_q ? hg : (hg - uint(HQ));

        // [B, 1, H*D] in and [B, H, 1, D] out are the same linear layout at
        // T == 1, so the stock transpose is free here.
        size_t head_offset = is_q
            ? ((size_t)b * (size_t)(HQ * HEAD_DIM) + (size_t)h * HEAD_DIM)
            : ((size_t)b * (size_t)(HKV * HEAD_DIM) + (size_t)h * HEAD_DIM);
        const device T* src = (is_q ? q_in : k_in) + head_offset;
        const device T* w = is_q ? q_w : k_w;
        device T* dst = (is_q ? q_out : k_out) + head_offset;

        // RMS statistic, exactly as MLX's rms_single_row lays it out for a
        // 128-wide axis: 32 lanes x 4 sequential float squares, one simd_sum.
        float acc = 0.0f;
        uint base = lane * 4;
        for (int i = 0; i < 4; ++i) {
            float xi = static_cast<float>(src[base + i]);
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        float inv = metal::precise::rsqrt(acc / float(HEAD_DIM) + eps);

        float L = float(position);

        if (ROT_DIMS == HEAD_DIM) {
            for (uint p = lane; p < uint(HALF_ROT); p += 32u) {
                float inv_freq;
                if (USE_FREQS) {
                    inv_freq = 1.0 / (freqs[p]);
                } else {
                    float d = float(p) / float(HALF_ROT);
                    inv_freq = metal::exp2(-d * BASE_LOG2);
                }
                float theta = L * inv_freq;
                float costheta = metal::fast::cos(theta);
                float sintheta = metal::fast::sin(theta);
                T v1 = w[p] * static_cast<T>(src[p] * inv);
                T v2 = w[p + uint(HALF_ROT)] *
                    static_cast<T>(src[p + uint(HALF_ROT)] * inv);
                if (HAS_MSCALE) {
                    v1 = static_cast<T>(MSCALE_F) * v1;
                    v2 = static_cast<T>(MSCALE_F) * v2;
                }
                float x1 = static_cast<float>(v1);
                float x2 = static_cast<float>(v2);
                dst[p] = static_cast<T>(x1 * costheta - x2 * sintheta);
                dst[p + uint(HALF_ROT)] =
                    static_cast<T>(x1 * sintheta + x2 * costheta);
            }
        } else {
            // Partial rotary: rotate pairs (p, p + HALF_ROT) inside the first
            // ROT_DIMS dims; the tail is normed output with NO mscale, which
            // is exactly what the stock chain produces (the YaRN pre-scale
            // slices [..., :dims]).
            if (lane < uint(HALF_ROT)) {
                uint p = lane;
                float inv_freq;
                if (USE_FREQS) {
                    inv_freq = 1.0 / (freqs[p]);
                } else {
                    float d = float(p) / float(HALF_ROT);
                    inv_freq = metal::exp2(-d * BASE_LOG2);
                }
                float theta = L * inv_freq;
                float costheta = metal::fast::cos(theta);
                float sintheta = metal::fast::sin(theta);
                T v1 = w[p] * static_cast<T>(src[p] * inv);
                T v2 = w[p + uint(HALF_ROT)] *
                    static_cast<T>(src[p + uint(HALF_ROT)] * inv);
                if (HAS_MSCALE) {
                    v1 = static_cast<T>(MSCALE_F) * v1;
                    v2 = static_cast<T>(MSCALE_F) * v2;
                }
                float x1 = static_cast<float>(v1);
                float x2 = static_cast<float>(v2);
                dst[p] = static_cast<T>(x1 * costheta - x2 * sintheta);
                dst[p + uint(HALF_ROT)] =
                    static_cast<T>(x1 * sintheta + x2 * costheta);
            }
            constexpr int TAIL = HEAD_DIM - ROT_DIMS;
            constexpr int PER_LANE = TAIL / 32;
            for (int i = 0; i < PER_LANE; ++i) {
                uint t = uint(ROT_DIMS) + lane * uint(PER_LANE) + uint(i);
                dst[t] = w[t] * static_cast<T>(src[t] * inv);
            }
        }
    """

    name = (
        f"mtplx_laguna_qk_rope_hq{n_q_heads}_hkv{n_kv_heads}_r{rot_dims}"
        f"_{'freqs' if use_freqs else 'base'}{'_ms' if has_mscale else ''}"
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=["q_in", "k_in", "q_w", "k_w", "freqs", "eps", "position"],
        output_names=["q_out", "k_out"],
        header=header,
        source=source,
    )


_QK_DUMMY_FREQS = None


def fused_qk_norm_rope(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    position: int | mx.array,
    spec: QkRopeSpec,
) -> tuple[mx.array, mx.array]:
    """RMSNorm + (partial/YaRN) rope for q and k in one dispatch, at T == 1.

    Returns ``(queries, keys)`` shaped ``[B, heads, 1, head_dim]`` — the
    layout the stock transpose+rope chain hands to attention.  Callers check
    :func:`is_qk_norm_rope_eligible` first; there is no fallback here because
    the caller owns the stock path.

    ``position`` is the rope offset and may be either a Python ``int`` or an
    int32 scalar ``mx.array``.  The distinction matters under ``mx.compile``:
    the kernel's ``position`` input is a scalar buffer, so an ARRAY stays a
    traced graph input and one compiled graph serves every position, while a
    Python int would be captured as a trace constant and freeze the rotation
    at whatever offset the graph was first built at.  An array is therefore
    forwarded untouched — no ``int()``, which would sync the stream and bake
    the value in.  ``mtplx.laguna_compiled_step`` passes its ``offset`` leaf;
    the eager forward in ``mtplx.models.laguna_fused`` passes an int.
    """

    global _QK_DUMMY_FREQS
    batch = int(queries.shape[0])
    freqs = spec.freqs
    if freqs is None:
        if _QK_DUMMY_FREQS is None:
            _QK_DUMMY_FREQS = mx.ones((1,), dtype=mx.float32)
        freqs = _QK_DUMMY_FREQS

    kernel = _qk_norm_rope_kernel(
        spec.n_q_heads,
        spec.n_kv_heads,
        spec.rot_dims,
        spec.freqs is not None,
        float(spec.base_log2) if spec.base_log2 is not None else 0.0,
        float(spec.mscale) if spec.mscale is not None else 1.0,
    )
    total_heads = spec.n_q_heads + spec.n_kv_heads
    # A traced offset goes in as-is; a host int is normalized to one.  Both
    # reach the kernel as a scalar int32 buffer, so the JIT source is the same
    # either way and the two callers share one compiled variant.
    offset = position if isinstance(position, mx.array) else int(position)
    q_out, k_out = kernel(
        inputs=[queries, keys, q_weight, k_weight, freqs, float(eps), offset],
        template=[("T", queries.dtype)],
        grid=(_QK_LANES * batch * total_heads, 1, 1),
        threadgroup=(_QK_LANES, 1, 1),
        output_shapes=[
            (batch, spec.n_q_heads, 1, spec.head_dim),
            (batch, spec.n_kv_heads, 1, spec.head_dim),
        ],
        output_dtypes=[queries.dtype, queries.dtype],
    )
    return q_out, k_out


# ---------------------------------------------------------------------------
# fused MoE weighted combine (+ shared-expert add)
# ---------------------------------------------------------------------------
#
# Stock: `(expert_out * weights[..., None]).sum(axis=-2) + shared` materializes
# a [rows, K, hidden] product (61 KB per layer per token written and re-read)
# and costs three dispatches.  This kernel reads the expert outputs once and
# writes the combined row directly.
#
# Bit-exactness means matching MLX's own strided_reduce_small order for a
# K-deep bf16 column reduction: threadgroup_y = min(8, K) partial accumulators,
# partial y summing rows {y, y+TY, ...} in ascending order, partials combined
# in ascending y with `op(partial, total)`, everything in the tensor dtype.
# The weight multiply rounds to the tensor dtype first (the stock astype), and
# the shared-expert add happens after the reduction's final rounding, exactly
# as the separate stock add does.


def is_moe_combine_eligible(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> bool:
    if not _on_metal_device():
        return False
    if expert_out.ndim != 3 or weights.ndim != 2 or shared.ndim != 2:
        return False
    if expert_out.dtype not in (mx.bfloat16, mx.float16):
        return False
    if shared.dtype != expert_out.dtype:
        return False
    if weights.dtype not in (mx.float32, expert_out.dtype):
        return False
    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    if top_k <= 0 or top_k > 32:
        return False
    if (int(weights.shape[0]), int(weights.shape[1])) != (rows, top_k):
        return False
    return (int(shared.shape[0]), int(shared.shape[1])) == (rows, hidden)


@lru_cache(maxsize=None)
def _moe_combine_kernel(top_k: int, hidden: int):
    ty = min(8, top_k)
    header = f"""
        using namespace metal;
        constant constexpr int TOP_K = {top_k};
        constant constexpr int HIDDEN = {hidden};
        constant constexpr int TY = {ty};
    """

    source = """
        uint idx = thread_position_in_grid.x;
        uint row = idx / uint(HIDDEN);
        uint c = idx - row * uint(HIDDEN);

        const device T* base_ptr =
            expert_out + (size_t)row * (size_t)(TOP_K * HIDDEN) + c;

        // Reproduce col_reduce_small's threadgroup_y=TY accumulation exactly:
        // partial y takes rows {y, y+TY, ...} in order, then partials combine
        // in ascending y as op(partial, running).
        T totals[TY];
        for (int y = 0; y < TY; ++y) {
            totals[y] = T(0);
        }
        for (int r = 0; r < TOP_K; ++r) {
            T wv = static_cast<T>(weights[(size_t)row * TOP_K + r]);
            T prod = base_ptr[(size_t)r * HIDDEN] * wv;
            totals[r % TY] = prod + totals[r % TY];
        }
        T total = totals[0];
        for (int y = 1; y < TY; ++y) {
            total = totals[y] + total;
        }
        combined[idx] = total + shared_in[idx];
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_moe_combine_k{top_k}_h{hidden}",
        input_names=["expert_out", "weights", "shared_in"],
        output_names=["combined"],
        header=header,
        source=source,
    )


def fused_moe_combine(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> mx.array:
    """Weighted expert combine plus shared-expert add, one dispatch.

    Falls back to the stock op chain on any shape the kernel does not cover,
    so callers can switch it on without owning a correctness branch.
    """

    if not is_moe_combine_eligible(expert_out, weights, shared):
        combined = (
            expert_out * weights.astype(expert_out.dtype)[..., None]
        ).sum(axis=-2)
        return combined + shared

    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    kernel = _moe_combine_kernel(top_k, hidden)
    total = rows * hidden
    (combined,) = kernel(
        inputs=[expert_out, weights, shared],
        template=[("T", expert_out.dtype)],
        grid=(total, 1, 1),
        threadgroup=(256 if total >= 256 else 32, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[expert_out.dtype],
    )
    return combined


# ---------------------------------------------------------------------------
# fused dense-MLP GLU over the concatenated gate/up projection
# ---------------------------------------------------------------------------
#
# `laguna_fused.FusedGateUpMLP` issues gate and up as ONE (quantized) matmul, so
# what comes back is a single [rows, 2H] row with gate in the first half and up
# in the second.  The stock activation then slices that row in two and evaluates
# `nn.silu(gate) * up` over the halves.  Both halves are STRIDED views of the
# same buffer, and the component census on the pinned oQ4e checkpoint prices
# what that costs per layer: the fused-elementwise sigmoid/multiply chain, plus
# an H-wide contiguity copy before `down_proj` will take the result.  That is
# ~two removable serial links per layer over 47 shared experts at H=1024 and the
# dense layer-0 block at H=12288 — 0.25-0.4 ms/step.
#
# This kernel reads the two halves at their strides and writes ONE contiguous
# activation, so neither slice materializes and the copy has nothing left to do.
#
# WHAT IT IS WORTH, measured, because the two serving lanes disagree and the
# number is smaller than the census implies.  `check_fused_glu` prices it on a
# chained spine — 47 activations back to back, each depending on the last, so
# the harness's own plumbing is amortized rather than dominating — at bf16,
# rows=1, over four sessions on an M5 Max (ranges, not a best run):
#
#   eager     stock 360-370 us | kernel 368-373 us per 47 blocks
#             -> +0.03 to +0.42 us/block, +0.002 to +0.020 ms/step.  A wash
#             to a small LOSS, and the sign never went the other way.
#   compiled  stock 250-257 us | kernel 213-230 us per 47 blocks
#             -> -0.53 to -0.68 us/block, -0.027 to -0.032 ms/step.  A win,
#             and the tightest number of the four.
#
# They disagree because they run different stock chains.  Eager, the stock
# activation is a compiled `silu` kernel plus a binary multiply — two MLX
# elementwise dispatches, ~1.3 us each — against this kernel's one; but one
# custom-kernel dispatch costs about two stock ones at this size, because
# `mx.fast.metal_kernel`'s own call validation is ~2.8 us of HOST time against
# ~2.0 us for the whole stock expression.  Removing a dispatch does not pay for
# the one that replaces it.  Under `mx.compile` the stock chain fuses to a
# single kernel, but its two inputs are strided views and MLX materializes them
# before the fused elementwise runs — and that copy is what this kernel removes.
#
# So it is the compiled lane that this is for, which is also the lane the model
# serves from.  -0.03 ms/step is an order of magnitude under the 0.25-0.4
# ms/step the in-model census attributes to this link, and the micro-lane cannot
# adjudicate that gap: it prices dispatches around a synthetic dependency, and
# the census prices them inside a decode step where each sits between two pieces
# of real work.  The guarded window decides; nothing here is sold as the answer.
#
# EXACTNESS.  Bit-exact by mirroring MLX's own Metal op structs rather than
# re-deriving the activation:
#
#   - `mtplx_sigmoid` is `struct Sigmoid` from
#     mlx/backend/metal/kernels/unary_ops.h verbatim —
#     `auto y = 1 / (1 + metal::exp(metal::abs(x))); return (x < 0) ? y : 1 - y;`
#     Written that way for the rounding, not the algebra: at T = bfloat16_t,
#     `metal::exp` resolves to bf16_math.h's overload, which is
#     `static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), ...))` — so the
#     exponential is ROUNDED TO BFLOAT16 before the reciprocal, and a version
#     that kept it in float would be a different number, not a different last
#     bit.  Custom kernels get the same preamble (`metal::utils()`, which
#     includes bf16_math.h) and the same compile options as MLX's own kernels
#     (device.cpp builds every JIT library with `setFastMathEnabled(false)`), so
#     `__METAL_MAYBE_FAST_MATH__` resolves identically on both sides.
#   - The two multiplies are `struct Multiply`'s `T operator()(T x, T y)` from
#     binary_ops.h, applied in the stock order: `silu(gate)` lands in a T
#     variable first, which is the rounding `nn.silu`'s own output carries,
#     and only then meets `up`.
#
# That order is what BOTH stock lanes evaluate.  Uncompiled, `nn.silu` is itself
# `@partial(mx.compile)` (`x * mx.sigmoid(x)`), so it writes a bfloat16 array and
# the `* up` binary op reads it back.  Under the compiled step the whole chain
# fuses, but MLX's compiled elementwise codegen declares every intermediate at
# its array's own dtype (backend/metal/compiled.cpp emits
# `{type} tmp_{name} = Op{}(...)` with `get_type_string(x.dtype())`), so the
# intermediate is rounded to bfloat16 there too.  The two lanes agree, and this
# kernel reproduces both.
#
# `mtplx_sigmoid` deliberately matches the `sigmoid_mlx_exact` helper already
# embedded in `kernels/verify_mlp_fused.py`; the difference is only that this
# one keeps MLX's `(x < 0)` spelling rather than `(x < T(0))`.

# MLX's Metal Sigmoid and Multiply, so what this kernel writes is the same bits
# the stock expression writes.  See the note above for why each cast is where
# it is.
_GLU_OPS = """
    // mlx/backend/metal/kernels/unary_ops.h, struct Sigmoid, verbatim.
    template <typename T>
    inline T mtplx_sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(metal::abs(x)));
        return (x < 0) ? y : 1 - y;
    }

    // mlx/backend/metal/kernels/binary_ops.h, struct Multiply, applied the two
    // times the stock expression applies it.  `silu` is a T variable rather
    // than an expression on purpose: that is the rounding `nn.silu`'s own
    // output carries into the second multiply.
    template <typename T>
    inline T mtplx_silu_mul(T gate, T up) {
        T silu = gate * mtplx_sigmoid<T>(gate);
        return silu * up;
    }
"""


def is_glu_eligible(fused: mx.array, hidden: int) -> bool:
    """Whether the fused GLU covers this concatenated projection.

    No row gate, unlike the router: this kernel reads 2H and writes H per row
    and does the stock arithmetic once per output element, so it is strictly
    fewer bytes and strictly fewer dispatches than the slice/activate/copy chain
    at every row count.  There is no batch at which the stock path becomes the
    better one, so there is nothing for a threshold to protect.
    """

    if not _on_metal_device():
        return False
    if fused.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if fused.ndim < 1:
        return False
    if hidden <= 0:
        return False
    return int(fused.shape[-1]) == 2 * hidden


@lru_cache(maxsize=None)
def _glu_kernel(hidden: int):
    header = _GLU_OPS + f"""
        using namespace metal;
        constant constexpr int HIDDEN = {hidden};
    """

    source = """
        uint index = thread_position_in_grid.x;
        uint row = index / uint(HIDDEN);
        uint col = index - row * uint(HIDDEN);
        // The gate/up halves at their strides — read, never materialized.
        size_t base = (size_t)row * (size_t)(2 * HIDDEN) + (size_t)col;
        activated[index] =
            mtplx_silu_mul<T>(gate_up[base], gate_up[base + (size_t)HIDDEN]);
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_glu_h{hidden}",
        input_names=["gate_up"],
        output_names=["activated"],
        header=header,
        source=source,
    )


def fused_glu(fused: mx.array, hidden: int) -> mx.array:
    """``silu(gate) * up`` over a concatenated [..., 2H] projection.

    One dispatch, one contiguous [..., H] output: the caller's ``down_proj``
    takes it as it stands.  Falls back to the stock expression verbatim on any
    shape or device the kernel does not cover, so callers do not own a
    correctness branch.

    No environment variable of its own.  This is the internals of the path
    ``MTPLX_LAGUNA_FUSED_SHARED_GATE_UP`` already installs — a run that did not
    ask for the gate/up concatenation never reaches it, and one that did gets
    the same numbers with two fewer links in the chain.
    """

    if not is_glu_eligible(fused, hidden):
        gate = fused[..., :hidden]
        up = fused[..., hidden:]
        return nn.silu(gate) * up

    # Kept to the cheapest expressions that produce the right thing: this
    # kernel replaces two dispatches worth ~1.4 us, and `mx.fast.metal_kernel`'s
    # own call validation already spends ~2.8 us of that, so a comprehension
    # over the shape is a measurable fraction of the margin rather than noise.
    shape = fused.shape
    total = fused.size // 2
    kernel = _glu_kernel(hidden)
    (activated,) = kernel(
        inputs=[fused],
        template=[("T", fused.dtype)],
        grid=(total, 1, 1),
        threadgroup=(256 if total >= 256 else 32, 1, 1),
        output_shapes=[shape[:-1] + (hidden,)],
        output_dtypes=[fused.dtype],
    )
    return activated
