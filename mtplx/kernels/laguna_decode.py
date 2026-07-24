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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import mlx.core as mx

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
    # One thread per expert, and the selection scratch is sized at compile time.
    return 0 < top_k <= 32 and 32 <= experts <= 1024 and (experts % 32) == 0


@lru_cache(maxsize=None)
def _router_kernel(experts: int, top_k: int):
    header = _SOFTPLUS + f"""
        using namespace metal;
        constant constexpr int NUM_EXPERTS = {experts};
        constant constexpr int TOP_K = {top_k};
    """

    source = """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;

        threadgroup float tg_score[NUM_EXPERTS];
        threadgroup float tg_choice[NUM_EXPERTS];
        threadgroup float red_val[NUM_EXPERTS];
        threadgroup uint  red_idx[NUM_EXPERTS];
        threadgroup uint  sel_idx[TOP_K];
        threadgroup float sel_score[TOP_K];

        float logit = logits[row * NUM_EXPERTS + lid];
        float score = 1.0f / (1.0f + metal::exp(-logit));
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
    position: int,
    spec: QkRopeSpec,
) -> tuple[mx.array, mx.array]:
    """RMSNorm + (partial/YaRN) rope for q and k in one dispatch, at T == 1.

    Returns ``(queries, keys)`` shaped ``[B, heads, 1, head_dim]`` — the
    layout the stock transpose+rope chain hands to attention.  Callers check
    :func:`is_qk_norm_rope_eligible` first; there is no fallback here because
    the caller owns the stock path.
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
    q_out, k_out = kernel(
        inputs=[queries, keys, q_weight, k_weight, freqs, float(eps), int(position)],
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
