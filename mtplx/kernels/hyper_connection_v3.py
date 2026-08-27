"""Wide fused hyper-connection read, v3: two dispatches, 8-bit weights.

The eager hyper read is ~10 small ops over 13.2MB of bf16 weights per read
(96 reads/step = 1.27 GB/token — the single biggest bf16 stream left in the
qwen4_exp decode step). The v1 single-threadgroup kernel lost to the eager
library GEMVs (single-TG bandwidth ceiling, receipt 2026-08-27 01:00), and
routing the shipped pack's hc Linears through stock QuantizedLinear
regressed AR 51->32 (DRAM-cold packed-qmv shape pathology, receipt
2026-08-27 ~04:55). v3 keeps the module weights bf16 for prefill and bakes
a kernel-private 8-bit affine pack at prepare time:

  R1: grouped-rms norm scales + the K=10240 GEMV for down(320 rows) with
      the 4 inject rows FOLDED IN as rows 320..323 (same shape, one
      stream); silu on the down half, 2*sigmoid(x/4) on the inject half.
  R2: the K=320 up-GEMV per (hc, d) with sigmoid, then the mix*normed
      hc-mean — one simdgroup per output d owns all four hc rows.

Halves the read bytes (13.2 -> ~6.8 MB/read) and collapses ~10 dependent
ops to 2. Mixers measured KLD-clean at 8-bit (0.0003, 2026-08-26 KLD
attribution). Quantization contract: mx.quantize affine, group_size 64,
bits 8 — q u32-packed 4 values/word little-endian, w = scale*q + bias.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# One simdgroup per output row n of the folded [324, 10240] 8-bit matrix.
# The threadgroup first cooperatively computes the four grouped-rms scales
# (shared over its 32 simdgroups), then each simdgroup dots its row against
# normed(x) = x * w_norm * rms[g].
_SRC_R1 = """
    constexpr int K = 10240;
    constexpr int GROUP = 2560;          // hc group width
    constexpr int QGS = 64;              // quant group size
    constexpr int NGROUPS = K / QGS;     // 160 quant groups per row
    constexpr int N_DOWN = 320;
    constexpr int N_TOTAL = 324;

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint n = threadgroup_position_in_grid.x * 32 + sg;

    threadgroup float tg_sums[4];
    threadgroup float tg_partial[32];

    // Cooperative x^2 group sums (all 1024 threads, 10 elems each).
    float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = tid; i < K; i += 1024) {
        const float xv = (float)x[i];
        part[i / GROUP] += xv * xv;
    }
    for (int g = 0; g < 4; ++g) {
        float v = simd_sum(part[g]);
        if (lane == 0) tg_partial[sg] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float acc = (lane < 32) ? tg_partial[lane] : 0.0f;
            acc = simd_sum(acc);
            if (lane == 0) tg_sums[g] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms[4];
    for (int g = 0; g < 4; ++g) {
        rms[g] = metal::rsqrt(tg_sums[g] / (float)GROUP + 1e-6f);
    }
    if (threadgroup_position_in_grid.x == 0 && tid < 4) {
        rms_out[tid] = rms[tid];
    }
    if (n >= (uint)N_TOTAL) return;

    const device uint32_t* wrow = qw + (size_t)n * (K / 4);
    const device T* srow = qs + (size_t)n * NGROUPS;
    const device T* brow = qb + (size_t)n * NGROUPS;
    float acc = 0.0f;
    for (int g = lane; g < NGROUPS; g += 32) {
        const float s = (float)srow[g];
        const float b = (float)brow[g];
        const device uint32_t* wg = wrow + g * (QGS / 4);
        const int base = g * QGS;
        float qacc = 0.0f;
        float nsum = 0.0f;
        for (int wi = 0; wi < QGS / 4; ++wi) {
            const uint32_t word = wg[wi];
            const int i0 = base + wi * 4;
            for (int by = 0; by < 4; ++by) {
                const int i = i0 + by;
                const float nx = (float)x[i] * (float)wn[i] * rms[i / GROUP];
                qacc += (float)((word >> (8 * by)) & 0xFF) * nx;
                nsum += nx;
            }
        }
        acc += s * qacc + b * nsum;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        if (n < (uint)N_DOWN) {
            const float v = acc * 0.25f;               // / hc_count
            mix_out[n] = (T)(v / (1.0f + metal::exp(-v)));   // silu
        } else {
            const float v = acc * 0.25f;
            inject_out[n - N_DOWN] = (T)(2.0f / (1.0f + metal::exp(-v)));
        }
    }
"""

# One simdgroup per output dim d: four K=320 up-rows (d, 2560+d, ...),
# sigmoid each, hc-mean of sigmoid * normed.
_SRC_R2 = """
    constexpr int KUP = 320;
    constexpr int GROUP = 2560;
    constexpr int QGS = 64;
    constexpr int NG = KUP / QGS;        // 5 quant groups per up row

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint d = threadgroup_position_in_grid.x * 32 + sg;
    if (d >= (uint)GROUP) return;

    float acc_mix = 0.0f;
    for (int h = 0; h < 4; ++h) {
        const uint row = (uint)h * GROUP + d;
        const device uint32_t* wrow = qw + (size_t)row * (KUP / 4);
        const device T* srow = qs + (size_t)row * NG;
        const device T* brow = qb + (size_t)row * NG;
        float dot = 0.0f;
        for (int g = 0; g < NG; ++g) {
            const float s = (float)srow[g];
            const float b = (float)brow[g];
            const device uint32_t* wg = wrow + g * (QGS / 4);
            const int base = g * QGS;
            float qacc = 0.0f;
            float msum = 0.0f;
            for (int wi = (int)lane; wi < QGS / 4; wi += 32) {
                const uint32_t word = wg[wi];
                const int i0 = base + wi * 4;
                for (int by = 0; by < 4; ++by) {
                    const float mv = (float)mixv[i0 + by];
                    qacc += (float)((word >> (8 * by)) & 0xFF) * mv;
                    msum += mv;
                }
            }
            dot += s * qacc + b * msum;
        }
        dot = simd_sum(dot);
        const float m = 1.0f / (1.0f + metal::exp(-dot));   // sigmoid
        const int i = h * GROUP + (int)d;
        const float normed = (float)x[i] * (float)wn[i] * rms_in[h];
        acc_mix += m * normed;
    }
    if (lane == 0) {
        y[d] = (T)(acc_mix * 0.25f);
    }
"""


@lru_cache(maxsize=1)
def _kernel_r1():
    return mx.fast.metal_kernel(
        name="mtplx_hyper_v3_r1",
        input_names=["x", "wn", "qw", "qs", "qb"],
        output_names=["mix_out", "inject_out", "rms_out"],
        header=_HEADER,
        source=_SRC_R1,
    )


@lru_cache(maxsize=1)
def _kernel_r2():
    return mx.fast.metal_kernel(
        name="mtplx_hyper_v3_r2",
        input_names=["x", "wn", "mixv", "rms_in", "qw", "qs", "qb"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC_R2,
    )


def prepare_v3_pack(module):
    """Quantize this GatedResidual's read weights into the kernel pack.

    Returns (w1_q, w1_s, w1_b, w2_q, w2_s, w2_b) with the inject rows folded
    under the down rows in W1. bf16 module weights stay untouched (prefill
    and the >8-row path keep the eager chain).
    """
    down = module.input_mix_weight_down.weight  # [320, 10240]
    up = module.input_mix_weight_up.weight  # [10240, 320]
    inject = module.block_inject_weight.weight  # [4, 10240]
    w1 = mx.concatenate([down, inject], axis=0)  # [324, 10240]
    w1_q, w1_s, w1_b = mx.quantize(w1, group_size=64, bits=8)
    w2_q, w2_s, w2_b = mx.quantize(up, group_size=64, bits=8)
    pack = tuple(mx.contiguous(t) for t in (w1_q, w1_s, w1_b, w2_q, w2_s, w2_b))
    mx.eval(*pack)
    return pack


def fused_hyper_read_v3(x_row, wn, pack):
    """One hyper read for a single row. x_row [10240], wn [10240] bf16.

    Returns (mixed [2560], inject [4]). eps is the family's 1e-6,
    baked into the kernel."""
    w1_q, w1_s, w1_b, w2_q, w2_s, w2_b = pack
    r1 = _kernel_r1()
    n_tgs = (324 + 31) // 32
    mix, inject, rms = r1(
        inputs=[x_row, wn, w1_q, w1_s, w1_b],
        template=[("T", x_row.dtype)],
        grid=(n_tgs * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(320,), (4,), (4,)],
        output_dtypes=[x_row.dtype, x_row.dtype, mx.float32],
    )
    r2 = _kernel_r2()
    d_tgs = (2560 + 31) // 32
    (mixed,) = r2(
        inputs=[x_row, wn, mix, rms, w2_q, w2_s, w2_b],
        template=[("T", x_row.dtype)],
        grid=(d_tgs * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(2560,)],
        output_dtypes=[x_row.dtype],
    )
    return mixed, inject
