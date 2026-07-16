"""Row-owned fused Hy3 router: one threadgroup owns one logical row.

Issue #58 design law: correct by construction and lock/verification free.
One threadgroup is the only writer of its row's expert IDs and route weights,
so no cross-threadgroup protocol exists. The kernel contains no atomics, no
device-scope fences, no election, no readiness or checker machinery, and no
fallback dispatch; it uses only hardware threadgroup/simdgroup barriers.

Arithmetic contract (bitwise versus the #59 incumbent):

- R1 reproduces the N16/P16 MPP ``matmul2d`` cells: one ``multiply`` per
  (K-partition, 16-expert tile) with BM=8/BN=16/BK=256, FP32 activations
  against the prepared K-major BF16 weight. Padding rows live in threadgroup
  memory; no host-side row padding or padded device tensor is materialized.
- The per-expert total uses ``_balanced_splitk_reduction_source(16)``,
  the same deterministic balanced FP32 tree as the split-K incumbent.
- R2 reproduces the precise-G6 SIMD top-8 finalizer: sigmoid, expert-bias
  selection scores, later-index tie breaking, and the same reversed output
  order and ``ROUTING_SCALE / (sum + 1e-20)`` weight normalization.
"""

from __future__ import annotations

import math

import mlx.core as mx

from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    _balanced_splitk_reduction_source,
    _sigmoid_exp_call,
    hy3_router_fp32_available,
)

_EXPERTS = 192
_INPUT_WIDTH = 4096
_TOP_K = 8
_K_PARTS = 16
_K_SLICE = _INPUT_WIDTH // _K_PARTS
_N_TILE = 16
_SIMD_GROUPS = _EXPERTS // _N_TILE
_R2_SIMD_GROUPS = 6
_THREADS = _SIMD_GROUPS * 32

_KERNEL_CACHE: dict[tuple[float, str], object] = {}


def hy3_router_row_owned_source(
    *,
    scaling_factor: float = 2.826,
    sigmoid_mode: str = "precise",
) -> str:
    """Emit the complete row-owned fused R1+R2 Metal body."""

    exp_call = _sigmoid_exp_call(sigmoid_mode, "-total")
    reduction = _balanced_splitk_reduction_source(_K_PARTS)
    scaling_literal = format(float(scaling_factor), ".9g")
    if "." not in scaling_literal and "e" not in scaling_literal.lower():
        scaling_literal += ".0"
    return f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int N = {_EXPERTS};
        constexpr int TOPK = {_TOP_K};
        constexpr int P = {_K_PARTS};
        constexpr int KS = {_K_SLICE};
        constexpr int K = {_INPUT_WIDTH};
        constexpr int BM = 8;
        constexpr int BN = {_N_TILE};
        constexpr int SIMD_GROUPS = {_SIMD_GROUPS};
        constexpr int R2_SIMD_GROUPS = {_R2_SIMD_GROUPS};
        constexpr int EXPERTS_PER_SIMDGROUP = N / R2_SIMD_GROUPS;
        constexpr int CANDIDATES_PER_LANE = 1;
        constexpr int LOCAL_CANDIDATES = R2_SIMD_GROUPS * TOPK;
        constexpr int STRIDE = N;
        constexpr float ROUTING_SCALE = {scaling_literal}f;

        uint row = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tid = thread_index_in_threadgroup;

        threadgroup float a_tile[BM * KS];
        threadgroup float c_tile[SIMD_GROUPS][BM * BN];
        threadgroup float partials[P * N];
        threadgroup float local_selection[LOCAL_CANDIDATES];
        threadgroup float local_unbiased[LOCAL_CANDIDATES];
        threadgroup int local_indices[LOCAL_CANDIDATES];
        threadgroup float merged_unbiased[TOPK];
        threadgroup int merged_indices[TOPK];

        // R1: this threadgroup's padding rows are zeroed once; only the
        // owned activation slice is restaged per K partition.
        for (int offset = int(tid) + KS; offset < BM * KS;
             offset += SIMD_GROUPS * 32) {{
            a_tile[offset] = 0.0f;
        }}

        int n0 = int(simd_gid) * BN;
        constexpr auto desc = matmul2d_descriptor(
            BM,
            BN,
            KS,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc, metal::execution_simdgroup> op;

        for (int part = 0; part < P; ++part) {{
            int k0 = part * KS;
            for (int offset = int(tid); offset < KS;
                 offset += SIMD_GROUPS * 32) {{
                a_tile[offset] = x[row * K + k0 + offset];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            tensor<threadgroup float, dextents<int, 2>, tensor_inline> A(
                a_tile,
                dextents<int, 2>{{KS, BM}},
                array<int, 2>{{1, KS}});
            tensor<device bfloat, dextents<int, 2>, tensor_inline> B(
                (device bfloat*)weight + k0 * N + n0,
                dextents<int, 2>{{BN, KS}},
                array<int, 2>{{1, N}});
            tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
                c_tile[simd_gid],
                dextents<int, 2>{{BN, BM}},
                array<int, 2>{{1, BN}});
            op.run(A, B, C);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (int(lane) < BN) {{
                partials[part * N + n0 + int(lane)] =
                    c_tile[simd_gid][int(lane)];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // R2: precise-G6 top-8 on the first six SIMD groups; every thread
        // reaches every threadgroup barrier.
        float candidate_selection[CANDIDATES_PER_LANE];
        float candidate_unbiased[CANDIDATES_PER_LANE];
        int candidate_indices[CANDIDATES_PER_LANE];

        if (simd_gid < R2_SIMD_GROUPS) {{
            _Pragma("unroll")
            for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
                uint group_offset = lane + uint(slot) * 32;
                uint expert = simd_gid * EXPERTS_PER_SIMDGROUP + group_offset;
                uint index = expert;
                {reduction}
                float score = 1.0f / (1.0f + {exp_call});
                candidate_selection[slot] = score + expert_bias[expert];
                candidate_unbiased[slot] = score;
                candidate_indices[slot] = int(expert);
            }}

            _Pragma("unroll")
            for (int rank = 0; rank < TOPK; ++rank) {{
                float lane_selection = -INFINITY;
                float lane_unbiased = 0.0f;
                int lane_index = -1;
                _Pragma("unroll")
                for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
                    float selection = candidate_selection[slot];
                    int index = candidate_indices[slot];
                    bool higher = selection > lane_selection;
                    bool later_equal = selection == lane_selection
                        && index > lane_index;
                    if (higher || later_equal) {{
                        lane_selection = selection;
                        lane_unbiased = candidate_unbiased[slot];
                        lane_index = index;
                    }}
                }}

                float winner_selection = simd_max(lane_selection);
                float winner_index_value = simd_max(
                    lane_selection == winner_selection
                        ? float(lane_index)
                        : -1.0f);
                int winner_index = int(winner_index_value);
                float winner_unbiased = simd_sum(
                    lane_index == winner_index ? lane_unbiased : 0.0f);
                if (lane == 0) {{
                    int destination = int(simd_gid) * TOPK + rank;
                    local_selection[destination] = winner_selection;
                    local_unbiased[destination] = winner_unbiased;
                    local_indices[destination] = winner_index;
                }}
                _Pragma("unroll")
                for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
                    if (candidate_indices[slot] == winner_index) {{
                        candidate_selection[slot] = -INFINITY;
                    }}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {{
            int slot0 = int(lane);
            int slot1 = int(lane) + 32;
            bool valid0 = slot0 < LOCAL_CANDIDATES;
            bool valid1 = slot1 < LOCAL_CANDIDATES;
            float selection0 = valid0
                ? local_selection[slot0]
                : -INFINITY;
            float unbiased0 = valid0 ? local_unbiased[slot0] : 0.0f;
            int index0 = valid0 ? local_indices[slot0] : -1;
            float selection1 = valid1
                ? local_selection[slot1]
                : -INFINITY;
            float unbiased1 = valid1 ? local_unbiased[slot1] : 0.0f;
            int index1 = valid1 ? local_indices[slot1] : -1;

            _Pragma("unroll")
            for (int rank = 0; rank < TOPK; ++rank) {{
                bool take1 = selection1 > selection0
                    || (selection1 == selection0 && index1 > index0);
                float lane_selection = take1 ? selection1 : selection0;
                float lane_unbiased = take1 ? unbiased1 : unbiased0;
                int lane_index = take1 ? index1 : index0;

                float winner_selection = simd_max(lane_selection);
                float winner_index_value = simd_max(
                    lane_selection == winner_selection
                        ? float(lane_index)
                        : -1.0f);
                int winner_index = int(winner_index_value);
                float winner_unbiased = simd_sum(
                    lane_index == winner_index ? lane_unbiased : 0.0f);
                if (lane == 0) {{
                    merged_unbiased[rank] = winner_unbiased;
                    merged_indices[rank] = winner_index;
                }}
                if (lane_index == winner_index) {{
                    if (take1) {{
                        selection1 = -INFINITY;
                    }} else {{
                        selection0 = -INFINITY;
                    }}
                }}
            }}

            if (lane == 0) {{
                float score_sum = 0.0f;
                for (int output = 0; output < TOPK; ++output) {{
                    score_sum += merged_unbiased[TOPK - 1 - output];
                }}
                float scale = ROUTING_SCALE / (score_sum + 1e-20f);
                for (int output = 0; output < TOPK; ++output) {{
                    int source = TOPK - 1 - output;
                    uint output_index = row * TOPK + output;
                    expert_ids[output_index] = merged_indices[source];
                    router_scores[output_index] =
                        merged_unbiased[source] * scale;
                }}
            }}
        }}
    """


def _build_hy3_router_row_owned_kernel(
    scaling_factor: float,
    sigmoid_mode: str,
):
    key = (float(scaling_factor), sigmoid_mode)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name="mtplx_hy3_router_row_owned_g12_p16_precise",
        input_names=["x", "weight", "expert_bias"],
        output_names=["expert_ids", "router_scores"],
        header=("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"),
        source=hy3_router_row_owned_source(
            scaling_factor=scaling_factor,
            sigmoid_mode=sigmoid_mode,
        ),
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def hy3_router_row_owned_eligible(
    *,
    rows: int,
    input_width: int,
    experts: int,
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
    available: bool | None = None,
) -> bool:
    """Check the exact M1..M8 row-owned router contract."""

    supported = hy3_router_fp32_available() if available is None else bool(available)
    return (
        supported
        and 1 <= int(rows) <= 8
        and int(input_width) == _INPUT_WIDTH
        and int(experts) == _EXPERTS
        and input_dtype == mx.float32
        and weight_dtype == mx.bfloat16
    )


def hy3_router_row_owned_route(
    value: mx.array,
    weight: mx.array,
    expert_bias: mx.array,
    *,
    available: bool | None = None,
    top_k: int = 8,
    route_norm: bool = True,
    scaling_factor: float = 2.826,
    sigmoid_mode: str = "precise",
) -> tuple[mx.array, mx.array]:
    """Route M1..M8 rows through one fused row-owned dispatch."""

    if value.ndim < 2:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input must include rows and hidden width"
        )
    rows = math.prod(int(dimension) for dimension in value.shape[:-1])
    if weight.ndim != 2:
        raise Hy3RouterFP32Ineligible("Hy3 router weight must be rank two")
    input_width, experts = (int(dimension) for dimension in weight.shape)
    if int(value.shape[-1]) != input_width:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input and transposed-weight widths do not match"
        )
    if not hy3_router_row_owned_eligible(
        rows=rows,
        input_width=input_width,
        experts=experts,
        input_dtype=value.dtype,
        weight_dtype=weight.dtype,
        available=available,
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input is outside the exact row-owned M1..M8 lane"
        )
    if (
        expert_bias.ndim != 1
        or tuple(int(dimension) for dimension in expert_bias.shape) != (_EXPERTS,)
        or expert_bias.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router expert bias must be FP32 with shape (192,)"
        )
    if int(top_k) != _TOP_K:
        raise Hy3RouterFP32Ineligible("Hy3 router row-owned lane requires top-8")
    if not route_norm:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router row-owned lane requires normalized routes"
        )
    if not math.isfinite(float(scaling_factor)) or float(scaling_factor) <= 0.0:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router scaling factor must be finite and positive"
        )

    kernel = _build_hy3_router_row_owned_kernel(
        float(scaling_factor),
        sigmoid_mode,
    )
    expert_ids, router_scores = kernel(
        inputs=[value.reshape(rows, input_width), weight, expert_bias],
        grid=(rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, _TOP_K), (rows, _TOP_K)],
        output_dtypes=[mx.int32, mx.float32],
    )
    output_shape = (*value.shape[:-1], _TOP_K)
    return (
        expert_ids.reshape(output_shape),
        router_scores.reshape(output_shape),
    )
