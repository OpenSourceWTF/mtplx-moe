"""Single-matrix qmv4 — the house verify-width quantized GEMV, standalone.

Port of ``_gate_up_swiglu_qmv4_kernel`` (verify_mlp_fused.py) with the swiglu
epilogue stripped: one 4-bit affine matrix, ``y[m, n] = x[m] . W[n]``, M<=6
(decode AND verify rows). The dot core is byte-identical to the proven family:
per-lane pre-scaled x registers (x/16, /256, /4096 so nibble dots need no
shifts), uint16 packed loads, 16 contiguous values/lane (per-lane scale index
= simd_lid / (GS/16)), 8 rows/TG (2 simdgroups x 4 rows), K-blocks of 512.

Constraints (checked by ``qmv4_eligible``): 4-bit affine, group_size in
{32, 64, 128}, K % 512 == 0, scales dtype == x dtype, no bias, M <= 6.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
    constant constexpr int MAX_M = 6;

    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i] * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
"""

_SRC = """
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int M = int(M_size);
    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* w_base =
      (const device uint8_t*)w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
      scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
      biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

    float result[MAX_M][RESULTS_PER_SIMDGROUP];
    float x_thread[MAX_M][VALUES_PER_THREAD];
    float x_sum[MAX_M];

    for (int m = 0; m < MAX_M; ++m) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[m][row] = 0.0f;
      }
    }

    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
      for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
          const device T* x_m =
            x + m * K + k_block + int(simd_lid) * VALUES_PER_THREAD;
          x_sum[m] = load_vector4_exact<T>(x_m, x_thread[m]);
        }
      }

      const device uint8_t* w_block =
        w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device T* scales_block = scales_base + k_block / GS;
      const device T* biases_block = biases_base + k_block / GS;

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* w_row = w_block + row * in_vec_size_w;
          const device T* sc_row = scales_block + row * in_vec_size_g;
          const device T* bs_row = biases_block + row * in_vec_size_g;
          float scale = float(sc_row[0]);
          float bias = float(bs_row[0]);

          for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
              result[m][row] += qdot4_exact(
                w_row, x_thread[m], scale, bias, x_sum[m]
              );
            }
          }
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        for (int m = 0; m < MAX_M; ++m) {
          if (m < M) {
            float total = simd_sum(result[m][row]);
            if (simd_lid == 0) {
              y[m * N + n] = T(total);
            }
          }
        }
      }
    }
"""


@lru_cache(maxsize=None)
def _qmv4_kernel(group_size: int, dtype: mx.Dtype):
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_qmv4_matmul_gs{group_size}_{dtype_tag}",
        input_names=["x", "w", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=_SRC,
        header=_HEADER,
    )


def qmv4_eligible(x: mx.array, module) -> bool:
    """``module`` needs .weight/.scales/.biases/.group_size/.bits (a
    QuantizedLinear or any fused pack carrying the same attrs)."""
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if getattr(module, "bits", None) != 4:
        return False
    if getattr(module, "mode", "affine") not in (None, "affine"):
        return False
    gs = int(getattr(module, "group_size", 0) or 0)
    if gs not in (32, 64, 128):
        return False
    w = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    if w is None or scales is None or getattr(module, "biases", None) is None:
        return False
    if w.dtype != mx.uint32 or scales.dtype != x.dtype:
        return False
    if getattr(module, "bias", None) is not None:
        return False
    m = int(x.shape[-2]) if x.ndim >= 2 else 1
    k = int(x.shape[-1])
    if m > 6 or k % 512 != 0:
        return False
    return int(w.shape[1]) * 8 == k


def qmv4_matmul(x: mx.array, module) -> mx.array:
    """``x @ module.weight.T`` (transpose=True convention) via the qmv4 core.

    x: [..., M, K] with M <= 6. Falls back to stock mx.quantized_matmul on
    ineligible shapes so callers can route unconditionally.
    """
    if not qmv4_eligible(x, module):
        return mx.quantized_matmul(
            x,
            module.weight,
            module.scales,
            module.biases,
            transpose=True,
            group_size=int(module.group_size),
            bits=int(module.bits),
        )
    leading = x.shape[:-2]
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    x2 = x.reshape(m, k)
    kernel = _qmv4_kernel(int(module.group_size), x.dtype)
    grid_y = 2 * ((n + 7) // 8)
    (y,) = kernel(
        inputs=[x2, module.weight, module.scales, module.biases, m, k, n],
        template=[("T", x.dtype), ("GS", int(module.group_size))],
        grid=(32, grid_y, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, m, n)
