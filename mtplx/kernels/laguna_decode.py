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

from functools import lru_cache

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
