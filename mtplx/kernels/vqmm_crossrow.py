"""Crossrow multi-row QMV for 4-bit/g64 verify widths (M 2..9).

Port of ``qmv_fast_crossrow_affine4_g64`` from the public
Layr-Labs/qwen-3.8-mtp-challenge repository (vendored MLX fork,
``mlx/backend/metal/kernels/quantized.h``; see that repo's notices) —
adapted to an ``mx.fast.metal_kernel`` launch and the qwen4_exp family
shapes. The technique: small-M quantized matmul is WEIGHT-bandwidth-bound,
so one packed-weight read is dotted against TWO activation rows (float2
accumulators) — qmv-class weight traffic per PAIR of rows instead of per
row. Activations are pre-scaled by 1, 1/16, 1/256, 1/4096 at load so the
nibble dot runs on masked (unshifted) weights, and the affine bias term
rides on the raw activation sums.

Routing law (4 falsifications strong): this replaces library calls ONLY
where a microbench on the exact shape shows a win; everything else stays
stock. Use ``vqmm_crossrow_eligible`` + measured shape allowlists.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# One threadgroup = 64 threads = 2 simdgroups x 4 output rows; tg.x picks
# the input-row PAIR (ceil(M/2) groups), tg.y the 8-row output tile.
# 32 lanes x 16 values = one 512-wide k block per iteration; each 4-lane
# cluster shares one g64 scale group. Odd-M tail pairs a zeroed second row
# (identical output; the unused half rides arithmetic the weight-bound
# kernel has free).
_SRC = """
    constexpr int VPT = 16;                 // values per thread
    constexpr int BLOCK = 512;              // 32 lanes * VPT

    const uint tid_in_tg = thread_position_in_threadgroup.x;
    const uint sg = tid_in_tg / 32;
    const uint lane = tid_in_tg % 32;
    const int K = params[0];
    const int N = params[1];

    const int first_m = (int)threadgroup_position_in_grid.x * 2;
    if (first_m >= M) return;
    const int out_row = (int)threadgroup_position_in_grid.y * 8 + (int)sg * 4;
    const int Kb = K / 2;                   // packed bytes per row
    const int Kg = K / 64;                  // scale groups per row
    const bool has_pair = (first_m + 1) < M;

    const device uint8_t* wq8 = (const device uint8_t*)w;
    float2 pres[4];
    for (int r = 0; r < 4; ++r) pres[r] = float2(0.0f);

    for (int k = 0; k < K; k += BLOCK) {
        ushort pk[4][4];
        float sl[4];
        float bl[4];
        for (int r = 0; r < 4; ++r) {
            const int row = out_row + r;
            const device uint16_t* ws =
                (const device uint16_t*)(wq8 + (size_t)row * Kb + (k >> 1) + (int)lane * 8);
            for (int i = 0; i < 4; ++i) pk[r][i] = ws[i];
            const int gi = row * Kg + (k >> 6) + (int)lane / 4;
            sl[r] = (float)scales[gi];
            bl[r] = (float)biases[gi];
        }
        const device T* xm0 = x + (size_t)first_m * K + k + (int)lane * VPT;
        float x0[VPT];
        float x1[VPT];
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int i = 0; i < VPT; i += 4) {
            sum0 += (float)xm0[i] + (float)xm0[i + 1] + (float)xm0[i + 2] + (float)xm0[i + 3];
            x0[i] = (float)xm0[i];
            x0[i + 1] = (float)xm0[i + 1] / 16.0f;
            x0[i + 2] = (float)xm0[i + 2] / 256.0f;
            x0[i + 3] = (float)xm0[i + 3] / 4096.0f;
        }
        if (has_pair) {
            const device T* xm1 = xm0 + K;
            for (int i = 0; i < VPT; i += 4) {
                sum1 += (float)xm1[i] + (float)xm1[i + 1] + (float)xm1[i + 2] + (float)xm1[i + 3];
                x1[i] = (float)xm1[i];
                x1[i + 1] = (float)xm1[i + 1] / 16.0f;
                x1[i + 2] = (float)xm1[i + 2] / 256.0f;
                x1[i + 3] = (float)xm1[i + 3] / 4096.0f;
            }
        } else {
            for (int i = 0; i < VPT; ++i) x1[i] = 0.0f;
        }
        for (int r = 0; r < 4; ++r) {
            float2 acc = float2(0.0f);
            for (int i = 0; i < 4; ++i) {
                acc += float2(x0[4 * i],     x1[4 * i])     * (float)(pk[r][i] & 0x000f);
                acc += float2(x0[4 * i + 1], x1[4 * i + 1]) * (float)(pk[r][i] & 0x00f0);
                acc += float2(x0[4 * i + 2], x1[4 * i + 2]) * (float)(pk[r][i] & 0x0f00);
                acc += float2(x0[4 * i + 3], x1[4 * i + 3]) * (float)(pk[r][i] & 0xf000);
            }
            pres[r] += sl[r] * acc + float2(sum0, sum1) * bl[r];
        }
    }

    for (int r = 0; r < 4; ++r) {
        const int row = out_row + r;
        const float red0 = simd_sum(pres[r].x);
        const float red1 = simd_sum(pres[r].y);
        if (lane == 0 && row < N) {
            y[(size_t)first_m * N + row] = (T)red0;
            if (has_pair) y[(size_t)(first_m + 1) * N + row] = (T)red1;
        }
    }
"""


@lru_cache(maxsize=16)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_vqmm_crossrow_g64",
        input_names=["x", "w", "scales", "biases", "params"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC,
    )


def vqmm_crossrow_eligible(x, module) -> bool:
    if x.ndim < 2 or not (2 <= int(x.shape[-2]) <= 9):
        return False
    if getattr(module, "bits", None) != 4 or getattr(module, "group_size", None) != 64:
        return False
    w = getattr(module, "weight", None)
    if w is None or w.dtype != mx.uint32:
        return False
    k = int(x.shape[-1])
    if k % 512 != 0 or int(w.shape[1]) * 8 != k:
        return False
    if getattr(module, "scales", None) is None or module.scales.dtype != x.dtype:
        return False
    if getattr(module, "bias", None) is not None:
        return False
    return True


def vqmm_crossrow(x, module):
    """x [.., M, K] (M 2..9), module a 4-bit/g64 QuantizedLinear.
    Returns x @ W.T like mx.quantized_matmul(transpose=True)."""
    if not vqmm_crossrow_eligible(x, module):
        return mx.quantized_matmul(
            x, module.weight, module.scales, module.biases,
            transpose=True, group_size=module.group_size, bits=module.bits,
        )
    lead = x.shape[:-2]
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    x2 = x.reshape(m, k)
    params = mx.array([k, n], dtype=mx.int32)
    kern = _kernel()
    (y,) = kern(
        inputs=[x2, module.weight, module.scales, module.biases, params],
        template=[("T", x.dtype), ("M", m)],
        grid=(((m + 1) // 2) * 64, (n + 7) // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*lead, m, n)
