"""Fused GDN decode conv + silu + q/k l2norm — the between-GEMVs kernel.

D1 slice of the 2-dispatch ladder, shaped by the 2026-08-27 kernel-shape
law: library GEMVs stay library calls; this kernel only replaces the eager
chain BETWEEN them (conv-state concat -> depthwise conv1d(k=4) -> silu ->
split -> per-head l2norm on q/k -> q scale), which is ~5-7 dependent
elementwise/copy dispatches per GDN layer at qL=1.

Channel layout (family): conv_dim 10240 = q 16x128 | k 16x128 | v 48x128.
Math contract (mirrors GatedDeltaNet.__call__ exactly):
  conv_out[c] = silu(sum_t w[c, t] * window[t, c]),  window = [state(3), new]
  q_head = inv_scale * x * rsqrt(sum(x^2) + 1e-6)   (fp32, cast back)
  k_head =             x * rsqrt(sum(x^2) + 1e-6)
  v      = conv_out (silu only)
  state' = window[1:4]
One threadgroup owns 1024 consecutive channels (8 heads of 128) so the
per-head reductions never leave threadgroup scope.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SRC = """
    constexpr int C = 10240;                  // conv channels
    constexpr int QK = 2048;                  // q width == k width
    constexpr int HD = 128;                   // head dim
    constexpr float INV_SCALE = 0.08838834764831845f;   // 128^-0.5

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint c = threadgroup_position_in_grid.x * 1024 + tid;
    if (c >= (uint)C) return;

    threadgroup float tg_vals[1024];
    threadgroup float tg_partial[32];

    // depthwise conv over [state0, state1, state2, new] + silu
    const float acc =
        (float)cw[c * 4 + 0] * (float)state[0 * C + c] +
        (float)cw[c * 4 + 1] * (float)state[1 * C + c] +
        (float)cw[c * 4 + 2] * (float)state[2 * C + c] +
        (float)cw[c * 4 + 3] * (float)xnew[c];
    const float sv = acc / (1.0f + metal::exp(-acc));

    // rolled conv state
    state_out[0 * C + c] = state[1 * C + c];
    state_out[1 * C + c] = state[2 * C + c];
    state_out[2 * C + c] = xnew[c];

    if (c >= (uint)(2 * QK)) {
        v_out[c - 2 * QK] = (T)sv;
        return;
    }

    // q/k: per-head l2norm. This TG holds 8 aligned heads of 128 channels;
    // 4 consecutive simdgroups own one head.
    tg_vals[tid] = sv;
    float part = simd_sum(sv * sv);
    if (lane == 0) tg_partial[sg] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint head_sg0 = (sg / 4) * 4;
    const float ssum = tg_partial[head_sg0] + tg_partial[head_sg0 + 1]
                     + tg_partial[head_sg0 + 2] + tg_partial[head_sg0 + 3];
    const float inv = metal::rsqrt(ssum + 1e-6f);
    const float normed = tg_vals[tid] * inv;
    if (c < (uint)QK) {
        q_out[c] = (T)(normed * INV_SCALE);
    } else {
        k_out[c - QK] = (T)normed;
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gdn_conv_norm",
        input_names=["xnew", "state", "cw"],
        output_names=["q_out", "k_out", "v_out", "state_out"],
        header=_HEADER,
        source=_SRC,
    )


def fused_gdn_conv_norm(qkv_row, conv_state, conv_w):
    """qkv_row [10240] (post in_proj, pre conv); conv_state [3, 10240];
    conv_w [10240, 4] (or [10240, 4, 1]). Returns (q [2048] normed+scaled,
    k [2048] normed, v [6144] silu, new_state [3, 10240])."""
    cw = conv_w.reshape(10240, 4)
    k = _kernel()
    q, kk, v, ns = k(
        inputs=[qkv_row, conv_state.reshape(-1), cw.reshape(-1)],
        template=[("T", qkv_row.dtype)],
        grid=(10 * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(2048,), (2048,), (6144,), (3, 10240)],
        output_dtypes=[qkv_row.dtype] * 4,
    )
    return q, kk, v, ns
