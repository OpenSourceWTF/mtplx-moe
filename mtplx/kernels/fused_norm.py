"""Small exact fusion probes for verify-window RMSNorm glue.

The kernels in this module intentionally target the Qwen3.6 verify shapes
where the last dimension is 5120 and the row count is tiny. They preserve the
stock MLX order for ``h = x + residual`` followed by ``mx.fast.rms_norm(h)``:
the residual add is rounded back to the input dtype before the RMS sum, matching
the standalone bf16 add feeding MLX's RMSNorm kernel.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def is_fused_add_rmsnorm_eligible(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
) -> bool:
    """Return whether the diagnostic add+RMSNorm kernel can run."""
    if not mx.metal.is_available():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if tuple(x.shape) != tuple(residual.shape):
        return False
    if len(x.shape) < 2 or len(weight.shape) != 1:
        return False
    if int(x.shape[-1]) != int(weight.shape[0]):
        return False
    return int(x.shape[-1]) > 0


@lru_cache(maxsize=None)
def _add_rmsnorm_kernel(dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int N_READS = 4;
    """

    source = """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint lsize = threads_per_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        int axis = int(axis_size);
        size_t row_offset = size_t(row) * size_t(axis);
        float acc = 0.0f;

        for (uint r = 0; r < uint(axis); r += lsize * N_READS) {
          uint base = r + lid * N_READS;
          for (int i = 0; i < N_READS; ++i) {
            uint idx = base + uint(i);
            if (idx < uint(axis)) {
              T h_val = x[row_offset + idx] + residual[row_offset + idx];
              acc += float(h_val) * float(h_val);
            }
          }
        }

        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(axis) + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < uint(axis); r += lsize * N_READS) {
          uint base = r + lid * N_READS;
          for (int i = 0; i < N_READS; ++i) {
            uint idx = base + uint(i);
            if (idx < uint(axis)) {
              T h_val = x[row_offset + idx] + residual[row_offset + idx];
              h[row_offset + idx] = h_val;
              normed[row_offset + idx] =
                weight[idx] * static_cast<T>(float(h_val) * local_inv_mean[0]);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_add_rmsnorm_{dtype_tag}",
        input_names=["x", "residual", "weight", "eps", "axis_size"],
        output_names=["h", "normed"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _add_rmsnorm_exact_kernel(dtype: mx.Dtype, axis: int):
    """Single-pass add+RMSNorm for rows an entire threadgroup covers at once.

    The looped kernel above was tuned for a 5120-wide row; at Laguna's 3072 it
    ran a fixed 1024-wide threadgroup (25% idle lanes) and read x and residual
    TWICE — once for the statistic, once for the writeback — and measured 1.7%
    SLOWER than the stock pair.  This variant sizes the threadgroup so
    ``threads * N_READS == axis`` (768 lanes at 3072), keeps the summed values
    in registers, and writes both outputs without re-reading.

    Accumulation layout matches MLX's own rms_single_row for the same axis:
    lane i squares its four consecutive elements in float, one simd_sum per
    simdgroup, partials combined through shared memory by simdgroup 0, and
    precise::rsqrt(acc / axis + eps).
    """

    header = f"""
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int N_READS = 4;
        constant constexpr int AXIS = {axis};
    """

    source = """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        size_t row_offset = size_t(row) * size_t(AXIS);
        uint base = lid * N_READS;

        T h_vals[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
          T h_val = x[row_offset + base + i] + residual[row_offset + base + i];
          h_vals[i] = h_val;
          float f = static_cast<float>(h_val);
          acc += f * f;
        }

        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(AXIS) + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inv = local_inv_mean[0];
        for (int i = 0; i < N_READS; ++i) {
          uint idx = base + uint(i);
          T h_val = h_vals[i];
          h[row_offset + idx] = h_val;
          normed[row_offset + idx] =
            weight[idx] * static_cast<T>(float(h_val) * inv);
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_add_rmsnorm_exact_{dtype_tag}_{axis}",
        input_names=["x", "residual", "weight", "eps"],
        output_names=["h", "normed"],
        header=header,
        source=source,
    )


def _exact_fit_threads(axis: int) -> int | None:
    """Threadgroup size covering the whole row in one pass, or None."""

    if axis <= 0 or axis % (32 * 4) != 0:
        return None
    threads = axis // 4
    return threads if threads <= 1024 else None


def fused_add_rmsnorm(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float,
    *,
    threadgroup_size: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Return ``(x + residual, rms_norm(x + residual, weight, eps))``.

    Unsupported shapes fall back to stock MLX operations so callers can use the
    helper behind an environment switch without changing correctness behavior.
    With no explicit ``threadgroup_size``, rows that a threadgroup covers in a
    single pass take the register-resident exact-fit kernel; everything else
    keeps the original looped kernel at 1024 lanes.
    """
    if not is_fused_add_rmsnorm_eligible(x, residual, weight):
        h = x + residual
        return h, mx.fast.rms_norm(h, weight, eps).astype(x.dtype)

    leading = x.shape[:-1]
    axis = int(x.shape[-1])
    rows = 1
    for dim in leading:
        rows *= int(dim)
    x2 = x.reshape(rows, axis)
    residual2 = residual.reshape(rows, axis)

    exact_threads = (
        _exact_fit_threads(axis) if threadgroup_size is None else None
    )
    if exact_threads is not None:
        kernel = _add_rmsnorm_exact_kernel(x.dtype, axis)
        h, normed = kernel(
            inputs=[x2, residual2, weight, float(eps)],
            template=[("T", x.dtype)],
            grid=(exact_threads * rows, 1, 1),
            threadgroup=(exact_threads, 1, 1),
            output_shapes=[(rows, axis), (rows, axis)],
            output_dtypes=[x.dtype, x.dtype],
        )
        return h.reshape(*leading, axis), normed.reshape(*leading, axis)

    threads = 1024 if threadgroup_size is None else int(threadgroup_size)
    kernel = _add_rmsnorm_kernel(x.dtype)
    h, normed = kernel(
        inputs=[x2, residual2, weight, float(eps), axis],
        template=[("T", x.dtype)],
        grid=(threads * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, axis), (rows, axis)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h.reshape(*leading, axis), normed.reshape(*leading, axis)


def is_fused_gdn_norm_gate_eligible(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
) -> bool:
    """Return whether the fused GDN RMSNorm+SiLU gate kernel can run."""
    if not mx.metal.is_available():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if gate.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if tuple(x.shape) != tuple(gate.shape):
        return False
    if len(x.shape) < 2 or len(weight.shape) != 1:
        return False
    if int(x.shape[-1]) != int(weight.shape[0]):
        return False
    return 0 < int(x.shape[-1]) <= 1024


@lru_cache(maxsize=None)
def _gdn_norm_gate_kernel(dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int N_READS = 4;

        inline float sigmoid_stable(float x) {
          float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
          return (x < 0.0f) ? y : 1.0f - y;
        }
    """

    source = """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        int axis = int(axis_size);
        size_t row_offset = size_t(row) * size_t(axis);
        uint base = lid * N_READS;
        float acc = 0.0f;

        for (int i = 0; i < N_READS; ++i) {
          uint idx = base + uint(i);
          if (idx < uint(axis)) {
            float xi = float(x[row_offset + idx]);
            acc += xi * xi;
          }
        }

        acc = simd_sum(acc);
        if (simd_group_id == 0) {
          local_sums[simd_lane_id] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_lane_id == 0) {
          local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(axis) + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < N_READS; ++i) {
          uint idx = base + uint(i);
          if (idx < uint(axis)) {
            T normed_t = weight[idx] *
              static_cast<T>(float(x[row_offset + idx]) * local_inv_mean[0]);
            float normed = float(normed_t);
            float gate_f = float(gate[row_offset + idx]);
            float silu = gate_f * sigmoid_stable(gate_f);
            y[row_offset + idx] = static_cast<T>(silu * normed);
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_gdn_norm_gate_{dtype_tag}",
        input_names=["x", "gate", "weight", "eps", "axis_size"],
        output_names=["y"],
        header=header,
        source=source,
    )


def fused_gdn_norm_gate(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    eps: float,
) -> mx.array:
    """Return Qwen3NextRMSNormGated(x, gate) for small GDN head rows."""
    if not is_fused_gdn_norm_gate_eligible(x, gate, weight):
        normed = mx.fast.rms_norm(x, weight, eps)
        gate_f = gate.astype(mx.float32)
        return (gate_f * mx.sigmoid(gate_f) * normed.astype(mx.float32)).astype(x.dtype)

    leading = x.shape[:-1]
    axis = int(x.shape[-1])
    rows = 1
    for dim in leading:
        rows *= int(dim)
    x2 = x.reshape(rows, axis)
    gate2 = gate.reshape(rows, axis)
    threadgroup_size = 32 * ((axis + 127) // 128)
    threadgroup_size = max(32, int(threadgroup_size))
    kernel = _gdn_norm_gate_kernel(x.dtype)
    (y,) = kernel(
        inputs=[x2, gate2, weight, float(eps), axis],
        template=[("T", x.dtype)],
        grid=(threadgroup_size * rows, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(rows, axis)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, axis)
