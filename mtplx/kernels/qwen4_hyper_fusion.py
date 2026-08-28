"""Fixed M=2 Qwen4 hyper-connection D/U kernels.

The D/U ownership and launch geometry derive from ``mlxserve_hc_read_d`` and
``mlxserve_hc_read_u`` in ddalcu/mlx-serve. Copyright (c) 2026 David Dalcu;
original kernels are MIT licensed. MTPLX keeps its own q4 inject projection and
its existing combine+grouped-normalization boundary.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


ROWS = 2
HC = 4
HIDDEN = 2560
WIDTH = HC * HIDDEN
RANK = 320
GROUP_SIZE = 32
BITS = 4

_KERNELS: dict[str, Any] = {}


def _down_source() -> str:
    return r"""
        using namespace metal;
        constexpr int ROWS = 2;
        constexpr int HC = 4;
        constexpr int H = 2560;
        constexpr int K = HC * H;
        constexpr int R = 320;
        constexpr int GS = 32;
        constexpr int BITS = 4;
        constexpr int VPW = 32 / BITS;
        constexpr int K_BY_PACK = K / VPW;
        constexpr int K_BY_GROUP = K / GS;
        constexpr int SLICE = K_BY_PACK / 8;
        constexpr int ITERS = SLICE / 32;

        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        uint output_row = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        threadgroup float partial[8];

        const device bfloat* x = normed + row * K;
        device bfloat* act_row = act + row * R;
        device bfloat* inject_row = inject + row * HC;
        bool is_down = output_row < uint(R);
        uint projection_row = is_down ? output_row : output_row - uint(R);
        const device uint32_t* weight = is_down ? down_weight : inject_weight;
        const device bfloat* scale = is_down ? down_scales : inject_scales;
        const device bfloat* bias = is_down ? down_biases : inject_biases;

        size_t weight_base = size_t(projection_row) * size_t(K_BY_PACK);
        size_t metadata_base = size_t(projection_row) * size_t(K_BY_GROUP);
        int pack0 = int(simd) * SLICE + int(lane);
        uint32_t packed[ITERS];
        for (int iteration = 0; iteration < ITERS; ++iteration) {
            packed[iteration] = weight[
                weight_base + size_t(pack0 + 32 * iteration)];
        }

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int iteration = 0; iteration < ITERS; ++iteration) {
            int k_base = (pack0 + 32 * iteration) * VPW;
            int group = k_base / GS;
            float s = float(scale[metadata_base + size_t(group)]);
            float b = float(bias[metadata_base + size_t(group)]);
            for (int piece = 0; piece < VPW; piece += 4) {
                int k = k_base + piece;
                uint32_t q = packed[iteration] >> (piece * BITS);
                acc0 += float(x[k + 0]) * (float((q >> 0) & 15u) * s + b);
                acc1 += float(x[k + 1]) * (float((q >> 4) & 15u) * s + b);
                acc2 += float(x[k + 2]) * (float((q >> 8) & 15u) * s + b);
                acc3 += float(x[k + 3]) * (float((q >> 12) & 15u) * s + b);
            }
        }
        float reduced = simd_sum((acc0 + acc1) + (acc2 + acc3));
        if (lane == 0) partial[simd] = reduced;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (int group = 0; group < 8; ++group) total += partial[group];
            bfloat projected = bfloat(total);
            bfloat scaled = bfloat(float(projected) * 0.25f);
            bfloat sigmoid_value = bfloat(
                1.0f / (1.0f + metal::exp(-float(scaled))));
            if (is_down) {
                act_row[projection_row] = bfloat(
                    float(scaled) * float(sigmoid_value));
            } else {
                inject_row[projection_row] = bfloat(
                    float(sigmoid_value) * 2.0f);
            }
        }
    """


def _up_source() -> str:
    return r"""
        using namespace metal;
        constexpr int ROWS = 2;
        constexpr int HC = 4;
        constexpr int H = 2560;
        constexpr int R = 320;
        constexpr int GS = 32;
        constexpr int BITS = 4;
        constexpr int VPW = 32 / BITS;
        constexpr int R_BY_PACK = R / VPW;
        constexpr int R_BY_GROUP = R / GS;
        constexpr int RIT = (R_BY_PACK + 31) / 32;

        uint lane = thread_index_in_simdgroup;
        uint hidden = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;
        const device bfloat* x = normed + row * (HC * H);
        const device bfloat* act_row = act + row * R;
        device bfloat* mixed_row = mixed + row * H;

        float stream_sum = 0.0f;
        for (int stream = 0; stream < HC; ++stream) {
            size_t output_row = size_t(stream) * size_t(H) + size_t(hidden);
            size_t weight_base = output_row * size_t(R_BY_PACK);
            size_t metadata_base = output_row * size_t(R_BY_GROUP);
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (int iteration = 0; iteration < RIT; ++iteration) {
                int pack = int(lane) + 32 * iteration;
                if (pack < R_BY_PACK) {
                    uint32_t packed = up_weight[weight_base + size_t(pack)];
                    int k_base = pack * VPW;
                    int group = k_base / GS;
                    float s = float(up_scales[metadata_base + size_t(group)]);
                    float b = float(up_biases[metadata_base + size_t(group)]);
                    for (int piece = 0; piece < VPW; piece += 4) {
                        int k = k_base + piece;
                        uint32_t q = packed >> (piece * BITS);
                        acc0 += float(act_row[k + 0])
                            * (float((q >> 0) & 15u) * s + b);
                        acc1 += float(act_row[k + 1])
                            * (float((q >> 4) & 15u) * s + b);
                        acc2 += float(act_row[k + 2])
                            * (float((q >> 8) & 15u) * s + b);
                        acc3 += float(act_row[k + 3])
                            * (float((q >> 12) & 15u) * s + b);
                    }
                }
            }
            float reduced = simd_sum((acc0 + acc1) + (acc2 + acc3));
            bfloat projected = bfloat(reduced);
            bfloat sigmoid_value = bfloat(
                1.0f / (1.0f + metal::exp(-float(projected))));
            bfloat product = bfloat(
                float(sigmoid_value) * float(x[output_row]));
            stream_sum += float(product);
        }
        if (lane == 0) {
            mixed_row[hidden] = bfloat(float(bfloat(stream_sum)) * 0.25f);
        }
    """


def sources() -> dict[str, str]:
    return {"down": _down_source(), "up": _up_source()}


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    if name == "down":
        kernel = mx.fast.metal_kernel(
            name="mtplx_qwen4_hyper_d_m2_q4g32",
            input_names=[
                "normed",
                "down_weight",
                "down_scales",
                "down_biases",
                "inject_weight",
                "inject_scales",
                "inject_biases",
            ],
            output_names=["act", "inject"],
            source=_down_source(),
            ensure_row_contiguous=True,
        )
    else:
        kernel = mx.fast.metal_kernel(
            name="mtplx_qwen4_hyper_u_m2_q4g32",
            input_names=["normed", "act", "up_weight", "up_scales", "up_biases"],
            output_names=["mixed"],
            source=_up_source(),
            ensure_row_contiguous=True,
        )
    _KERNELS[name] = kernel
    return kernel


def bind_m2(module: Any):
    down_kernel = _kernel("down")
    up_kernel = _kernel("up")
    down = module.input_mix_weight_down
    up = module.input_mix_weight_up
    inject_projection = module.block_inject_weight
    down_static = (
        down.weight,
        down.scales,
        down.biases,
        inject_projection.weight,
        inject_projection.scales,
        inject_projection.biases,
    )
    up_static = (up.weight, up.scales, up.biases)

    def call(normed: Any):
        act, inject = down_kernel(
            inputs=[normed, *down_static],
            grid=(256, RANK + HC, ROWS),
            threadgroup=(256, 1, 1),
            output_shapes=[(ROWS, RANK), (ROWS, HC)],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
        (mixed,) = up_kernel(
            inputs=[normed, act, *up_static],
            grid=(32, HIDDEN, ROWS),
            threadgroup=(32, 8, 1),
            output_shapes=[(ROWS, HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return mixed.reshape(1, ROWS, HIDDEN), inject.reshape(1, ROWS, HC)

    return call


__all__ = ["bind_m2", "sources"]
