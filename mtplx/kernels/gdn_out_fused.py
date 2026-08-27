"""Fused GDN output: SigmoidRMSNormGated + out_proj GEMV in ONE dispatch.

Step 1 of the 2-dispatch GDN ladder (design-gdn-2dispatch.md, 2026-08-27):
the decode-path chain norm -> sigmoid gate -> bf16 cast -> quantized GEMV is
2-3 dependent dispatches per GDN layer x 36 layers. This kernel computes the
48 per-head rms values and the 6144 gated-normed bf16 values cooperatively
into threadgroup memory (12 KB), then each simdgroup dots one out_proj row
(4-bit affine, group size templated) against the threadgroup-resident
values — out_proj is streamed exactly once from DRAM and the intermediate
value vector never round-trips.

Math contract (mirrors SigmoidRMSNormGated + QuantizedLinear exactly, up to
accumulation order): per head h over D=128 dims, rms = rsqrt(mean(x^2)+eps);
normed = x * rms * w[d % 128] (norm weight is per-head-dim, shared across
heads); v = bf16(sigmoid(z) * normed); y[d] = sum_k deq(W[d,k]) * v[k].
The bf16 cast of v BEFORE the dot matches the module boundary (the norm
returns hidden dtype; the projection consumes it).

INTEGRATION DEFERRED: kernel + parity only in this change; the module wiring
(env MTPLX_FUSED_GDN_OUT, capture-trace compatibility) is next-window work.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# One threadgroup = 1024 threads = 32 simdgroups. Phase 1 (cooperative):
# per-head sum of squares -> rms[48]; gated-normed bf16 values v[6144] in
# threadgroup memory. Phase 2: simdgroup sg dots out_proj rows
# (tg * 32 + sg), lane-strided over quant groups.
_SRC = """
    constexpr int NH = 48;                    // value heads
    constexpr int HD = 128;                   // head dim
    constexpr int K = NH * HD;                // 6144
    constexpr int GS = GS_C;                  // quant group size
    constexpr int NGROUPS = K / GS;
    constexpr int WPG = GS / 8;               // u32 words per group
    constexpr int DMODEL = 2560;

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;

    threadgroup float tg_rms[NH];
    threadgroup T tg_v[K];
    threadgroup float tg_part[32];

    // Phase 1a: per-head sum of squares. Head h owned by simdgroup h%32
    // (simdgroups 0..15 own two heads).
    for (uint h = sg; h < (uint)NH; h += 32) {
        float part = 0.0f;
        for (uint d = lane; d < (uint)HD; d += 32) {
            const float xv = (float)x[h * HD + d];
            part += xv * xv;
        }
        part = simd_sum(part);
        if (lane == 0) {
            tg_rms[h] = metal::rsqrt(part / (float)HD + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1b: gated-normed values, bf16-cast through T.
    for (uint i = tid; i < (uint)K; i += 1024) {
        const uint h = i / HD;
        const uint d = i % HD;
        const float normed = (float)x[i] * tg_rms[h] * (float)wn[d];
        const float g = 1.0f / (1.0f + metal::exp(-(float)z[i]));
        tg_v[i] = (T)(g * normed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: one simdgroup per output row.
    const uint row = threadgroup_position_in_grid.x * 32 + sg;
    if (row >= (uint)DMODEL) return;
    const device uint32_t* wrow = qw + (size_t)row * (K / 8);
    const device T* srow = qs + (size_t)row * NGROUPS;
    const device T* brow = qb + (size_t)row * NGROUPS;
    float acc = 0.0f;
    for (int g = lane; g < NGROUPS; g += 32) {
        const float s = (float)srow[g];
        const float b = (float)brow[g];
        const device uint32_t* wg = wrow + g * WPG;
        const int base = g * GS;
        float qacc = 0.0f;
        float vsum = 0.0f;
        for (int wi = 0; wi < WPG; ++wi) {
            const uint32_t word = wg[wi];
            const int i0 = base + wi * 8;
            for (int nib = 0; nib < 8; ++nib) {
                const float vv = (float)tg_v[i0 + nib];
                qacc += (float)((word >> (4 * nib)) & 0xF) * vv;
                vsum += vv;
            }
        }
        acc += s * qacc + b * vsum;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = (T)acc;
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gdn_out_fused",
        input_names=["x", "z", "wn", "qw", "qs", "qb"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC,
    )


def fused_gdn_out(x_row, z_row, norm_w, qw, qs, qb, *, group_size=64):
    """x_row, z_row: [6144] (48 heads x 128) in bf16; norm_w [128];
    out_proj 4-bit affine pack qw [2560, 768] u32, qs/qb [2560, 6144/gs].
    Returns y [2560] in x dtype."""
    if 6144 % group_size:
        raise ValueError(f"group size {group_size} must divide 6144")
    k = _kernel()
    tgs = (2560 + 31) // 32
    (y,) = k(
        inputs=[x_row, z_row, norm_w, qw, qs, qb],
        template=[("T", x_row.dtype), ("GS_C", int(group_size))],
        grid=(tgs * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(2560,)],
        output_dtypes=[x_row.dtype],
    )
    return y
