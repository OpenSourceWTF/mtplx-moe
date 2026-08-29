"""Fused hyper-connection (GatedResidual) read kernel for qwen4_exp decode.

The eager read chain is ~9 tiny kernels (grouped RMS norm, down-proj qmv,
silu, up-proj, sigmoid, reshape/mul/mean, inject qmv, sigmoid) issued 96
times per decode step (2 reads x 48 layers) — measured ~7-8ms of the 20.6ms
step on the 2026-08-26 attribution campaign, almost all launch overhead and
underfilled grids (the matmuls are 10240x320). This kernel runs the whole
read for one row in ONE dispatch: one threadgroup per row, 1024 threads,
normed staged in threadgroup memory (bf16, 20KB).

Float discipline mirrors the module chain: f32 accumulation with bf16 casts
at each op boundary (norm output, each Linear output, /hc, activations, the
mul). Not bit-identical to the eager chain (dot-product lane order differs
from MLX's matmul tiling) but within bf16 ulp noise; gate adoption on the
parity probe + downstream KLD/exactness harness, not on bitwise equality.

Weights are the module's bf16 tensors (hc mixes ship unquantized in the
current packs). Quantized hc weights fall back to the eager path.
"""

from functools import lru_cache

import mlx.core as mx

HC = 4
D_HIDDEN = 2560
HCD = HC * D_HIDDEN
R_LOWRANK = 320

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SOURCE = """
    constexpr int HC = 4;
    constexpr int D = 2560;
    constexpr int HCD = HC * D;
    constexpr int R = 320;
    constexpr float EPS = 1e-6f;

    const uint row = thread_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;

    threadgroup float gsum[32];
    threadgroup float nrm_scale[HC];
    threadgroup T normed[HCD];
    threadgroup float mixv[R];

    device const T* xr = x + (size_t)row * HCD;

    // ---- 1) per-group sum of squares (group g owns threads [g*256,(g+1)*256))
    const int g = tid >> 8;
    const int t_in = tid & 255;
    float ss = 0.0f;
    for (int k = 0; k < 10; ++k) {
        const int i = g * D + t_in + k * 256;
        const float v = (float)xr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) gsum[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HC) {
        float tot = 0.0f;
        for (int j = 0; j < 8; ++j) tot += gsum[tid * 8 + j];
        nrm_scale[tid] = metal::rsqrt(tot / (float)D + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 2) normed = x * scale_g * gamma (bf16 boundary like the module)
    for (int k = 0; k < 10; ++k) {
        const int i = g * D + t_in + k * 256;
        const float nv = (float)((T)((float)xr[i] * nrm_scale[g]));
        normed[i] = (T)(nv * (float)gamma[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 3) down proj rows (320 = 32 simdgroups x 10), /HC, silu
    for (int rr = 0; rr < 10; ++rr) {
        const int orow = sg * 10 + rr;
        float acc = 0.0f;
        device const T* wrow = wd + (size_t)orow * HCD;
        for (int i = lane; i < HCD; i += 32) {
            acc += (float)normed[i] * (float)wrow[i];
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            float t0 = (float)((T)acc);        // Linear out cast
            t0 = (float)((T)(t0 * 0.25f));     // /hc_count
            mixv[orow] = (float)((T)(t0 / (1.0f + metal::exp(-t0))));  // silu
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 4) inject = 2*sigmoid((Wi @ normed)/HC) — BEFORE the mix, which
    // repurposes the `normed` staging below.
    if (HAS_INJECT && sg < HC) {
        float acc = 0.0f;
        device const T* irow = wi + (size_t)sg * HCD;
        for (int i = lane; i < HCD; i += 32) {
            acc += (float)normed[i] * (float)irow[i];
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            float t1 = (float)((T)acc);
            t1 = (float)((T)(t1 * 0.25f));
            inject[(size_t)row * HC + sg] = (T)(2.0f / (1.0f + metal::exp(-t1)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 5a) up proj + sigmoid, gated product written back over `normed`
    // (one simdgroup per output row: lanes stride R — coalesced wu reads)
    for (int pos = (int)sg; pos < HCD; pos += 32) {
        device const T* urow = wu + (size_t)pos * R;
        float a = 0.0f;
        for (int j = lane; j < R; j += 32) a += mixv[j] * (float)urow[j];
        a = simd_sum(a);
        if (lane == 0) {
            a = (float)((T)a);                                   // Linear out
            float w = 1.0f / (1.0f + metal::exp(-a));            // sigmoid
            w = (float)((T)w);
            normed[pos] = (T)(w * (float)normed[pos]);           // bf16 mul
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 5b) mean over the HC groups
    for (int d0 = tid; d0 < D; d0 += 1024) {
        const float acc_mix = (float)normed[d0] + (float)normed[D + d0]
            + (float)normed[2 * D + d0] + (float)normed[3 * D + d0];
        mixed[(size_t)row * D + d0] = (T)(acc_mix * 0.25f);
    }
"""


@lru_cache(maxsize=2)
def _kernel(has_inject: bool):
    return mx.fast.metal_kernel(
        name=f"mtplx_hyper_read_{'inj' if has_inject else 'mix'}",
        input_names=["x", "gamma", "wd", "wu", "wi"],
        output_names=["mixed", "inject"],
        header=_HEADER,
        source=_SOURCE,
    )


def fused_hyper_read(x2d, gamma, wd, wu, wi=None):
    """x2d: [S, 10240] bf16 rows. Returns (mixed [S,2560], inject [S,4]);
    inject is meaningful only when wi is given."""
    S = x2d.shape[0]
    has_inject = wi is not None
    k = _kernel(has_inject)
    outs = k(
        inputs=[x2d, gamma, wd, wu, wi if has_inject else gamma],
        template=[("T", x2d.dtype), ("HAS_INJECT", 1 if has_inject else 0)],
        grid=(1024, S, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(S, D_HIDDEN), (S, HC)],
        output_dtypes=[x2d.dtype, x2d.dtype],
    )
    return outs[0], outs[1]
