"""G17 tensor-op projection for the exact Hy3 FP32 router geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import mlx.core as mx


class Hy3RouterFP32Ineligible(ValueError):
    """Raised before dispatch when a value is outside the exact router lane."""


@dataclass(frozen=True, slots=True)
class Hy3RouterFP32Tiling:
    """One exact-shape router projection scheduling candidate."""

    n_tile: Literal[16, 32, 64]
    grid_k_parts: Literal[1, 2, 4, 8, 16, 32]
    operand_mode: Literal["direct", "staged", "grouped-direct", "grouped-staged"] = (
        "direct"
    )
    k_tile: Literal[16, 32] | None = None
    simd_groups_per_threadgroup: int = 1

    def __post_init__(self) -> None:
        if self.n_tile not in (16, 32, 64):
            raise ValueError("router N tile must be 16, 32, or 64")
        if self.grid_k_parts not in (1, 2, 4, 8, 16, 32):
            raise ValueError("router grid K parts must be 1, 2, 4, 8, 16, or 32")
        if self.operand_mode not in (
            "direct",
            "staged",
            "grouped-direct",
            "grouped-staged",
        ):
            raise ValueError(
                "router operand mode must be direct, staged, grouped-direct, "
                "or grouped-staged"
            )
        if self.operand_mode == "direct" and self.k_tile is not None:
            raise ValueError("direct router operands do not accept a staged K tile")
        if self.operand_mode == "staged" and self.k_tile not in (16, 32):
            raise ValueError("staged router operands require K tile 16 or 32")
        grouped = self.operand_mode in ("grouped-direct", "grouped-staged")
        if grouped:
            if self.k_tile is not None:
                raise ValueError(
                    "grouped router operands do not accept a staged K tile"
                )
            if not 2 <= self.simd_groups_per_threadgroup <= 8:
                raise ValueError("grouped router operands require 2 to 8 SIMDgroups")
            n_tiles = 192 // self.n_tile
            if n_tiles % self.simd_groups_per_threadgroup:
                raise ValueError("router SIMDgroups must divide N tiles")
        elif self.simd_groups_per_threadgroup != 1:
            raise ValueError("ungrouped router operands require one SIMDgroup")
        if self.operand_mode == "grouped-staged" and self.grid_k_parts < 8:
            raise ValueError("grouped-staged router operands require P8, P16, or P32")

    @property
    def total_simdgroups(self) -> int:
        return (192 // self.n_tile) * self.grid_k_parts

    @property
    def stage1_threadgroups(self) -> int:
        return self.total_simdgroups // self.simd_groups_per_threadgroup

    @property
    def k_span(self) -> int:
        return 4096 // self.grid_k_parts

    @property
    def partial_bytes(self) -> int:
        return self.grid_k_parts * 8 * 192 * 4

    @property
    def staged_threadgroup_bytes(self) -> int:
        if self.operand_mode == "grouped-staged":
            return 8 * self.k_span * 4
        if self.operand_mode != "staged":
            return 0
        assert self.k_tile is not None
        activation = 8 * self.k_tile * 4
        weight = self.k_tile * self.n_tile * 2
        return activation + weight

    @property
    def modeled_activation_bytes(self) -> int:
        if self.operand_mode == "grouped-staged":
            return self.staged_threadgroup_bytes * self.stage1_threadgroups
        return 8 * self.k_span * 4 * self.total_simdgroups

    @property
    def modeled_weight_bytes(self) -> int:
        return 192 * 4096 * 2


_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _sigmoid_exp_call(sigmoid_mode: str, operand: str) -> str:
    """Return the selected Metal exponential call for fused router R2."""

    if sigmoid_mode == "precise":
        return f"exp({operand})"
    if sigmoid_mode == "fast-exp":
        return f"fast::exp({operand})"
    raise Hy3RouterFP32Ineligible("Hy3 router sigmoid mode must be precise or fast-exp")


def _balanced_splitk_reduction_source(grid_k_parts: int) -> str:
    """Emit one deterministic balanced FP32 reduction over all K partials."""

    if int(grid_k_parts) not in (1, 2, 4, 8, 16, 32):
        raise ValueError("unsupported router grid K partition count")
    terms = [f"partials[{part} * STRIDE + index]" for part in range(int(grid_k_parts))]
    lines: list[str] = []
    level = 0
    while len(terms) > 2:
        next_terms = []
        for pair in range(0, len(terms), 2):
            name = f"reduce_l{level}_{pair // 2}"
            lines.append(f"float {name} = {terms[pair]} + {terms[pair + 1]};")
            next_terms.append(name)
        terms = next_terms
        level += 1
    if len(terms) == 1:
        lines.append(f"float total = {terms[0]};")
    else:
        lines.append(f"float total = {terms[0]} + {terms[1]};")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def hy3_router_fp32_available() -> bool:
    """Report whether the installed device supports the G17 tensor-op lane."""

    try:
        from mtplx.nax_verify import nax_available

        return bool(nax_available())
    except Exception:
        return False


def hy3_router_fp32_eligible(
    *,
    rows: int,
    input_width: int,
    experts: int,
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
    available: bool | None = None,
) -> bool:
    """Check the exact K0..K7 source-Hy3 router projection contract."""

    supported = hy3_router_fp32_available() if available is None else bool(available)
    return (
        supported
        and 1 <= int(rows) <= 8
        and int(input_width) == 4096
        and int(experts) == 192
        and input_dtype == mx.float32
        and weight_dtype == mx.bfloat16
    )


def _build_hy3_router_fp32_kernel(
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
):
    key = (input_dtype, weight_dtype)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    source = """
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 8;
        constexpr int BN = 32;
        constexpr int BK = 16;
        constexpr int NSG = 6;
        constexpr int K = 4096;
        constexpr int N = 192;

        uint sg_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tid = thread_position_in_threadgroup.x;
        int n0 = int(sg_id) * BN;

        threadgroup float A_tile[BM * BK];
        threadgroup bfloat B_tile[NSG][BK * BN];

        constexpr auto desc = matmul2d_descriptor(
            BM,
            BN,
            BK,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        tensor<threadgroup float, dextents<int, 2>, tensor_inline> A(
            A_tile,
            dextents<int, 2>{BK, BM},
            array<int, 2>{1, BK});
        tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> B(
            B_tile[sg_id],
            dextents<int, 2>{BN, BK},
            array<int, 2>{1, BN});
        tensor<device float, dextents<int, 2>, tensor_inline> C(
            (device float*)y,
            dextents<int, 2>{N, BM},
            array<int, 2>{1, N});

        auto destination = op.template get_destination_cooperative_tensor<
            tensor<threadgroup float,
                   extents<int, BM, BK>, tensor_inline>,
            tensor<threadgroup bfloat,
                   extents<int, BN, BK>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < destination.get_capacity(); ++i) {
            destination[i] = 0.0f;
        }

        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int offset = int(tid); offset < BM * BK;
                 offset += NSG * 32) {
                int row = offset / BK;
                int column = offset - row * BK;
                A_tile[offset] = x[row * K + k0 + column];
            }
            _Pragma("unroll")
            for (int ki = 0; ki < BK; ++ki) {
                B_tile[sg_id][ki * BN + int(lane)] =
                    weight[(k0 + ki) * N + n0 + int(lane)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto tile_a = A.template slice<BM, BK>(0, 0);
            auto tile_b = B.template slice<BN, BK>(0, 0);
            op.run(tile_a, tile_b, destination);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto tile_c = C.template slice<BN, BM>(n0, 0);
        destination.store(tile_c);
    """
    kernel = mx.fast.metal_kernel(
        name="mtplx_hy3_router_fp32_m8_k4096_n192_coalesced",
        input_names=["x", "weight"],
        output_names=["y"],
        header=("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"),
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _build_hy3_router_fp32_direct_partial_kernel(
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
    *,
    n_tile: int,
    grid_k_parts: int,
):
    key = (
        "direct-partials",
        input_dtype,
        weight_dtype,
        int(n_tile),
        int(grid_k_parts),
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    n_tiles = 192 // n_tile
    k_span = 4096 // grid_k_parts
    weight_type = "float" if weight_dtype == mx.float32 else "bfloat"
    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 8;
        constexpr int BN = {int(n_tile)};
        constexpr int K = 4096;
        constexpr int N = 192;
        constexpr int P = {int(grid_k_parts)};
        constexpr int NT = {int(n_tiles)};
        constexpr int KS = {int(k_span)};

        uint tg = threadgroup_position_in_grid.x;
        int part = int(tg) / NT;
        int n_tile_index = int(tg) - part * NT;
        int n0 = n_tile_index * BN;
        int k0 = part * KS;

        tensor<device float, dextents<int, 2>, tensor_inline> A(
            (device float*)x + k0,
            dextents<int, 2>{{KS, BM}},
            array<int, 2>{{1, K}});
        tensor<device {weight_type}, dextents<int, 2>, tensor_inline> B(
            (device {weight_type}*)weight + k0 * N + n0,
            dextents<int, 2>{{BN, KS}},
            array<int, 2>{{1, N}});
        tensor<device float, dextents<int, 2>, tensor_inline> C(
            (device float*)partials + part * BM * N + n0,
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, N}});

        constexpr auto desc = matmul2d_descriptor(
            BM,
            BN,
            KS,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc, metal::execution_simdgroup> op;
        op.run(A, B, C);
    """
    kernel = mx.fast.metal_kernel(
        name=(f"mtplx_hy3_router_fp32_direct_m8_n{int(n_tile)}_p{int(grid_k_parts)}"),
        input_names=["x", "weight"],
        output_names=["partials"],
        header=("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"),
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _grouped_partial_source(
    tiling: Hy3RouterFP32Tiling,
    *,
    weight_dtype: mx.Dtype = mx.bfloat16,
) -> str:
    """Emit one-part-per-threadgroup MPP source for grouped N tiles."""

    if tiling.operand_mode not in ("grouped-direct", "grouped-staged"):
        raise ValueError("grouped partial source requires a grouped operand mode")
    n_tiles = 192 // tiling.n_tile
    groups_per_part = n_tiles // tiling.simd_groups_per_threadgroup
    weight_type = "float" if weight_dtype == mx.float32 else "bfloat"
    if tiling.operand_mode == "grouped-staged":
        activation_setup = """
        threadgroup float A_tile[BM * KS];
        for (int offset = int(tid); offset < BM * KS; offset += SGPTG * 32) {
            int row = offset / KS;
            int column = offset - row * KS;
            A_tile[offset] = x[row * K + k0 + column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tensor<threadgroup float, dextents<int, 2>, tensor_inline> A(
            A_tile,
            dextents<int, 2>{KS, BM},
            array<int, 2>{1, KS});
        """
    else:
        activation_setup = """
        tensor<device float, dextents<int, 2>, tensor_inline> A(
            (device float*)x + k0,
            dextents<int, 2>{KS, BM},
            array<int, 2>{1, K});
        """
    return f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 8;
        constexpr int BN = {int(tiling.n_tile)};
        constexpr int K = 4096;
        constexpr int N = 192;
        constexpr int P = {int(tiling.grid_k_parts)};
        constexpr int NT = {int(n_tiles)};
        constexpr int KS = {int(tiling.k_span)};
        constexpr int SGPTG = {int(tiling.simd_groups_per_threadgroup)};
        constexpr int GROUPS_PER_PART = {int(groups_per_part)};

        uint tg = threadgroup_position_in_grid.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint tid = thread_index_in_threadgroup;
        int part = int(tg) / GROUPS_PER_PART;
        int group_in_part = int(tg) - part * GROUPS_PER_PART;
        int n_tile_index = group_in_part * SGPTG + int(sg_id);
        int n0 = n_tile_index * BN;
        int k0 = part * KS;

        {activation_setup}
        tensor<device {weight_type}, dextents<int, 2>, tensor_inline> B(
            (device {weight_type}*)weight + k0 * N + n0,
            dextents<int, 2>{{BN, KS}},
            array<int, 2>{{1, N}});
        tensor<device float, dextents<int, 2>, tensor_inline> C(
            (device float*)partials + part * BM * N + n0,
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, N}});

        constexpr auto desc = matmul2d_descriptor(
            BM,
            BN,
            KS,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc, metal::execution_simdgroup> op;
        op.run(A, B, C);
    """


def _build_hy3_router_fp32_grouped_partial_kernel(
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
    *,
    tiling: Hy3RouterFP32Tiling,
):
    key = (
        "grouped-partials",
        input_dtype,
        weight_dtype,
        int(tiling.n_tile),
        int(tiling.grid_k_parts),
        tiling.operand_mode,
        int(tiling.simd_groups_per_threadgroup),
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    kernel = mx.fast.metal_kernel(
        name=(
            f"mtplx_hy3_router_fp32_{tiling.operand_mode.replace('-', '_')}_"
            f"m8_n{int(tiling.n_tile)}_p{int(tiling.grid_k_parts)}_"
            f"sg{int(tiling.simd_groups_per_threadgroup)}"
        ),
        input_names=["x", "weight"],
        output_names=["partials"],
        header=("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"),
        source=_grouped_partial_source(tiling, weight_dtype=weight_dtype),
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _partial_kernel_and_threads(
    input_dtype: mx.Dtype,
    weight_dtype: mx.Dtype,
    tiling: Hy3RouterFP32Tiling,
) -> tuple[Any, int]:
    if tiling.operand_mode == "direct":
        return (
            _build_hy3_router_fp32_direct_partial_kernel(
                input_dtype,
                weight_dtype,
                n_tile=tiling.n_tile,
                grid_k_parts=tiling.grid_k_parts,
            ),
            32,
        )
    if tiling.operand_mode in ("grouped-direct", "grouped-staged"):
        threads = tiling.simd_groups_per_threadgroup * 32
        return (
            _build_hy3_router_fp32_grouped_partial_kernel(
                input_dtype,
                weight_dtype,
                tiling=tiling,
            ),
            threads,
        )
    raise Hy3RouterFP32Ineligible("legacy staged Hy3 router is not implemented")


def _build_hy3_router_fp32_reduce_kernel(grid_k_parts: int):
    key = ("reduce-partials", int(grid_k_parts))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    reduction = _balanced_splitk_reduction_source(grid_k_parts)

    source = f"""
        using namespace metal;

        constexpr int ROWS = 8;
        constexpr int N = 192;
        constexpr int STRIDE = ROWS * N;

        uint index = thread_position_in_grid.x;
        if (index >= STRIDE) {{
            return;
        }}
        {reduction}
        y[index] = total;
    """
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_hy3_router_fp32_reduce_p{int(grid_k_parts)}",
        input_names=["partials"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _build_hy3_router_fp32_route_serial_kernel(
    grid_k_parts: int,
    scaling_factor: float,
    sigmoid_mode: str,
):
    """Reduce grid-K partials and finalize all top-8 router rows."""

    exp_call = _sigmoid_exp_call(sigmoid_mode, "-total")
    key = (
        "route-partials-serial",
        int(grid_k_parts),
        float(scaling_factor),
        sigmoid_mode,
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    reduction = _balanced_splitk_reduction_source(grid_k_parts)

    scaling_literal = format(float(scaling_factor), ".9g")
    if "." not in scaling_literal and "e" not in scaling_literal.lower():
        scaling_literal += ".0"
    source = f"""
        using namespace metal;

        constexpr int PADDED_ROWS = 8;
        constexpr int N = 192;
        constexpr int TOPK = 8;
        constexpr int STRIDE = PADDED_ROWS * N;
        constexpr float ROUTING_SCALE = {scaling_literal}f;

        uint row = threadgroup_position_in_grid.x;
        uint expert = thread_position_in_threadgroup.x;
        uint index = row * N + expert;

        threadgroup float selection_scores[N];
        threadgroup float unbiased_scores[N];

        {reduction}
        float score = 1.0f / (1.0f + {exp_call});
        unbiased_scores[expert] = score;
        selection_scores[expert] = score + expert_bias[expert];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (expert == 0) {{
            float top_selection[TOPK];
            float top_scores[TOPK];
            int top_indices[TOPK];
            for (int position = 0; position < TOPK; ++position) {{
                top_selection[position] = -INFINITY;
                top_scores[position] = 0.0f;
                top_indices[position] = -1;
            }}

            for (int candidate = 0; candidate < N; ++candidate) {{
                float selection = selection_scores[candidate];
                float candidate_score = unbiased_scores[candidate];
                for (int position = 0; position < TOPK; ++position) {{
                    bool higher = selection > top_selection[position];
                    bool later_equal = selection == top_selection[position]
                        && candidate > top_indices[position];
                    if (higher || later_equal) {{
                        for (
                            int shift = TOPK - 1;
                            shift > position;
                            --shift
                        ) {{
                            top_selection[shift] = top_selection[shift - 1];
                            top_scores[shift] = top_scores[shift - 1];
                            top_indices[shift] = top_indices[shift - 1];
                        }}
                        top_selection[position] = selection;
                        top_scores[position] = candidate_score;
                        top_indices[position] = candidate;
                        break;
                    }}
                }}
            }}

            float score_sum = 0.0f;
            for (int output = 0; output < TOPK; ++output) {{
                score_sum += top_scores[TOPK - 1 - output];
            }}
            float scale = ROUTING_SCALE / (score_sum + 1e-20f);
            for (int output = 0; output < TOPK; ++output) {{
                int source = TOPK - 1 - output;
                uint output_index = row * TOPK + output;
                expert_ids[output_index] = top_indices[source];
                router_scores[output_index] = top_scores[source] * scale;
            }}
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtplx_hy3_router_fp32_route_serial_p{int(grid_k_parts)}_"
            f"{sigmoid_mode.replace('-', '_')}"
        ),
        input_names=["partials", "expert_bias"],
        output_names=["expert_ids", "router_scores"],
        source=source,
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _build_hy3_router_fp32_route_simd_kernel(
    grid_k_parts: int,
    scaling_factor: float,
    sigmoid_mode: str,
    simd_groups: int = 6,
):
    """Reduce partials and select top-8 with a tuned SIMDgroup topology."""

    if not 1 <= int(simd_groups) <= 8:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router SIMDgroup count must be between 1 and 8"
        )

    exp_call = _sigmoid_exp_call(sigmoid_mode, "-total")
    key = (
        "route-partials-simd",
        int(grid_k_parts),
        float(scaling_factor),
        sigmoid_mode,
        int(simd_groups),
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    reduction = _balanced_splitk_reduction_source(grid_k_parts)

    scaling_literal = format(float(scaling_factor), ".9g")
    if "." not in scaling_literal and "e" not in scaling_literal.lower():
        scaling_literal += ".0"
    single_simd_source = f"""
        using namespace metal;

        constexpr int PADDED_ROWS = 8;
        constexpr int N = 192;
        constexpr int TOPK = 8;
        constexpr int CANDIDATES_PER_LANE = 6;
        constexpr int STRIDE = PADDED_ROWS * N;
        constexpr float ROUTING_SCALE = {scaling_literal}f;

        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        float candidate_selection[CANDIDATES_PER_LANE];
        float candidate_unbiased[CANDIDATES_PER_LANE];
        int candidate_indices[CANDIDATES_PER_LANE];
        float merged_unbiased[TOPK];
        int merged_indices[TOPK];

        _Pragma("unroll")
        for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
            uint expert = lane + uint(slot) * 32;
            uint index = row * N + expert;
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
                merged_unbiased[rank] = winner_unbiased;
                merged_indices[rank] = winner_index;
            }}
            _Pragma("unroll")
            for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
                if (candidate_indices[slot] == winner_index) {{
                    candidate_selection[slot] = -INFINITY;
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
    """
    num_simd_groups = int(simd_groups)
    experts_per_simdgroup = (192 + num_simd_groups - 1) // num_simd_groups
    candidates_per_lane = (experts_per_simdgroup + 31) // 32
    source = f"""
        using namespace metal;

        constexpr int PADDED_ROWS = 8;
        constexpr int N = 192;
        constexpr int TOPK = 8;
        constexpr int NUM_SIMDGROUPS = {num_simd_groups};
        constexpr int EXPERTS_PER_SIMDGROUP = {experts_per_simdgroup};
        constexpr int CANDIDATES_PER_LANE = {candidates_per_lane};
        constexpr int LOCAL_CANDIDATES = NUM_SIMDGROUPS * TOPK;
        constexpr int STRIDE = PADDED_ROWS * N;
        constexpr float ROUTING_SCALE = {scaling_literal}f;

        uint row = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        threadgroup float local_selection[LOCAL_CANDIDATES];
        threadgroup float local_unbiased[LOCAL_CANDIDATES];
        threadgroup int local_indices[LOCAL_CANDIDATES];
        threadgroup float merged_unbiased[TOPK];
        threadgroup int merged_indices[TOPK];

        float candidate_selection[CANDIDATES_PER_LANE];
        float candidate_unbiased[CANDIDATES_PER_LANE];
        int candidate_indices[CANDIDATES_PER_LANE];

        _Pragma("unroll")
        for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
            uint group_offset = lane + uint(slot) * 32;
            uint expert = simd_gid * EXPERTS_PER_SIMDGROUP + group_offset;
            bool valid = group_offset < EXPERTS_PER_SIMDGROUP && expert < N;
            if (valid) {{
                uint index = row * N + expert;
                {reduction}
                float score = 1.0f / (1.0f + {exp_call});
                candidate_selection[slot] = score + expert_bias[expert];
                candidate_unbiased[slot] = score;
                candidate_indices[slot] = int(expert);
            }} else {{
                candidate_selection[slot] = -INFINITY;
                candidate_unbiased[slot] = 0.0f;
                candidate_indices[slot] = -1;
            }}
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
                lane_index == winner_index
                    ? lane_unbiased
                    : 0.0f);
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
    if int(simd_groups) == 1:
        source = single_simd_source
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtplx_hy3_router_fp32_route_simd_g{int(simd_groups)}_"
            f"p{int(grid_k_parts)}_"
            f"{sigmoid_mode.replace('-', '_')}"
        ),
        input_names=["partials", "expert_bias"],
        output_names=["expert_ids", "router_scores"],
        source=source,
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def prepare_hy3_router_fp32_weight(weight: mx.array) -> mx.array:
    """Build the construction-time K-major layout used by the MPP probe."""

    if (
        weight.ndim != 2
        or tuple(weight.shape) != (192, 4096)
        or weight.dtype != mx.bfloat16
    ):
        raise Hy3RouterFP32Ineligible(
            "source Hy3 router weight must be BF16 with shape (192, 4096)"
        )
    return mx.contiguous(weight.T)


def prepare_hy3_router_fp32_exact_weight(weight: mx.array) -> mx.array:
    """Promote once while preserving stock row-major transposed dispatch."""

    if (
        weight.ndim != 2
        or tuple(weight.shape) != (192, 4096)
        or weight.dtype != mx.bfloat16
    ):
        raise Hy3RouterFP32Ineligible(
            "source Hy3 router weight must be BF16 with shape (192, 4096)"
        )
    return mx.contiguous(weight.astype(mx.float32))


def prepare_hy3_router_fp32_exact_splitk_weight(weight: mx.array) -> mx.array:
    """Promote once into the K-major FP32 layout consumed by split-K R1."""

    if (
        weight.ndim != 2
        or tuple(weight.shape) != (192, 4096)
        or weight.dtype != mx.bfloat16
    ):
        raise Hy3RouterFP32Ineligible(
            "source Hy3 router weight must be BF16 with shape (192, 4096)"
        )
    return mx.contiguous(weight.T.astype(mx.float32))


def hy3_router_fp32_exact_project(
    value: mx.array,
    weight: mx.array,
    *,
    available: bool | None = None,
) -> mx.array:
    """Run one stock MLX GEMV/GEMM projection with a pre-promoted weight."""

    if value.ndim < 2:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input must include rows and hidden width"
        )
    rows = math.prod(int(dimension) for dimension in value.shape[:-1])
    supported = hy3_router_fp32_available() if available is None else bool(available)
    if (
        not supported
        or not 1 <= rows <= 8
        or int(value.shape[-1]) != 4096
        or value.dtype != mx.float32
        or weight.ndim != 2
        or tuple(int(dimension) for dimension in weight.shape) != (192, 4096)
        or weight.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 exact router projection requires M1..M8 FP32 x [192, 4096] FP32"
        )
    logits = value.reshape(rows, 4096) @ weight.T
    return logits.reshape(*value.shape[:-1], 192)


def hy3_router_fp32_project(
    value: mx.array,
    weight: mx.array,
    *,
    available: bool | None = None,
    n_tile: int = 16,
    grid_k_parts: int = 4,
    operand_mode: str = "direct",
    k_tile: int | None = None,
    simd_groups_per_threadgroup: int = 1,
) -> mx.array:
    """Project K0..K7 target rows through one source-Hy3 router matrix."""

    tiling = Hy3RouterFP32Tiling(
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
        operand_mode=operand_mode,
        k_tile=k_tile,
        simd_groups_per_threadgroup=simd_groups_per_threadgroup,
    )
    if value.ndim < 2:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input must include rows and hidden width"
        )
    rows = 1
    for dimension in value.shape[:-1]:
        rows *= int(dimension)
    if weight.ndim != 2:
        raise Hy3RouterFP32Ineligible("Hy3 router weight must be rank two")
    input_width, experts = (int(dimension) for dimension in weight.shape)
    if int(value.shape[-1]) != input_width:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input and transposed-weight widths do not match"
        )
    if not hy3_router_fp32_eligible(
        rows=rows,
        input_width=input_width,
        experts=experts,
        input_dtype=value.dtype,
        weight_dtype=weight.dtype,
        available=available,
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input is outside the exact G17 K0..K7 lane"
        )

    execution_rows = 8
    flat = value.reshape(rows, input_width)
    if rows < execution_rows:
        flat = mx.concatenate(
            (
                flat,
                mx.zeros(
                    (execution_rows - rows, input_width),
                    dtype=value.dtype,
                ),
            ),
            axis=0,
        )
    partial_kernel, threads = _partial_kernel_and_threads(
        value.dtype,
        weight.dtype,
        tiling,
    )
    (partials,) = partial_kernel(
        inputs=[mx.contiguous(flat), weight],
        grid=(tiling.stage1_threadgroups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tiling.grid_k_parts, execution_rows, 192)],
        output_dtypes=[mx.float32],
    )
    reduce_kernel = _build_hy3_router_fp32_reduce_kernel(tiling.grid_k_parts)
    (output,) = reduce_kernel(
        inputs=[partials],
        grid=(execution_rows * 192, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(execution_rows, 192)],
        output_dtypes=[mx.float32],
    )
    return output[:rows].reshape(*value.shape[:-1], experts)


def hy3_router_fp32_finalize(
    logits: mx.array,
    expert_bias: mx.array,
    *,
    available: bool | None = None,
    top_k: int = 8,
    route_norm: bool = True,
    scaling_factor: float = 2.826,
    finalizer_mode: Literal["serial", "simd"] = "simd",
    simd_groups: Literal[1, 2, 3, 4, 5, 6, 7, 8] = 6,
    sigmoid_mode: Literal["precise", "fast-exp"] = "precise",
) -> tuple[mx.array, mx.array]:
    """Finalize stock FP32 logits with the selected sigmoid implementation."""

    if not 1 <= int(simd_groups) <= 8:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router SIMDgroup count must be between 1 and 8"
        )
    if logits.ndim < 2 or int(logits.shape[-1]) != 192:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router logits must include rows and 192 experts"
        )
    rows = 1
    for dimension in logits.shape[:-1]:
        rows *= int(dimension)
    supported = hy3_router_fp32_available() if available is None else bool(available)
    if not supported or not 1 <= rows <= 8 or logits.dtype != mx.float32:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router logits are outside the exact G17 K0..K7 lane"
        )
    if (
        expert_bias.ndim != 1
        or tuple(int(dimension) for dimension in expert_bias.shape) != (192,)
        or expert_bias.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router expert bias must be FP32 with shape (192,)"
        )
    if int(top_k) != 8:
        raise Hy3RouterFP32Ineligible("Hy3 router fused finalizer requires top-8")
    if not route_norm:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router fused finalizer requires normalized routes"
        )
    if not math.isfinite(float(scaling_factor)) or float(scaling_factor) <= 0.0:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router scaling factor must be finite and positive"
        )
    if finalizer_mode not in ("serial", "simd"):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router finalizer mode must be serial or simd"
        )
    _sigmoid_exp_call(sigmoid_mode, "-total")

    if finalizer_mode == "simd":
        route_kernel = _build_hy3_router_fp32_route_simd_kernel(
            1,
            float(scaling_factor),
            sigmoid_mode,
            simd_groups=simd_groups,
        )
        threads = int(simd_groups) * 32
    else:
        route_kernel = _build_hy3_router_fp32_route_serial_kernel(
            1,
            float(scaling_factor),
            sigmoid_mode,
        )
        threads = 192
    expert_ids, router_scores = route_kernel(
        inputs=[logits.reshape(rows, 192), expert_bias],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, 8), (rows, 8)],
        output_dtypes=[mx.int32, mx.float32],
    )
    output_shape = (*logits.shape[:-1], 8)
    return expert_ids.reshape(output_shape), router_scores.reshape(output_shape)


def hy3_router_fp32_exact_route(
    value: mx.array,
    weight: mx.array,
    expert_bias: mx.array,
    *,
    available: bool | None = None,
    top_k: int = 8,
    route_norm: bool = True,
    scaling_factor: float = 2.826,
    finalizer_mode: Literal["serial", "simd"] = "simd",
    simd_groups: Literal[1, 2, 3, 4, 5, 6, 7, 8] = 6,
    sigmoid_mode: Literal["precise", "fast-exp"] = "precise",
) -> tuple[mx.array, mx.array]:
    """Run exact stock R1 once, then the exact fused R2 finalizer once."""

    logits = hy3_router_fp32_exact_project(
        value,
        weight,
        available=available,
    )
    return hy3_router_fp32_finalize(
        logits,
        expert_bias,
        available=available,
        top_k=top_k,
        route_norm=route_norm,
        scaling_factor=scaling_factor,
        finalizer_mode=finalizer_mode,
        simd_groups=simd_groups,
        sigmoid_mode=sigmoid_mode,
    )


def hy3_router_fp32_exact_splitk_project(
    value: mx.array,
    weight: mx.array,
    *,
    available: bool | None = None,
    n_tile: int = 16,
    grid_k_parts: int = 8,
    operand_mode: str = "direct",
    k_tile: int | None = None,
    simd_groups_per_threadgroup: int = 1,
) -> mx.array:
    """Project with K-major FP32 split-K and materialize diagnostic logits."""

    tiling = Hy3RouterFP32Tiling(
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
        operand_mode=operand_mode,
        k_tile=k_tile,
        simd_groups_per_threadgroup=simd_groups_per_threadgroup,
    )
    if value.ndim < 2:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input must include rows and hidden width"
        )
    rows = math.prod(int(dimension) for dimension in value.shape[:-1])
    supported = hy3_router_fp32_available() if available is None else bool(available)
    if (
        not supported
        or not 1 <= rows <= 8
        or int(value.shape[-1]) != 4096
        or value.dtype != mx.float32
        or weight.ndim != 2
        or tuple(int(dimension) for dimension in weight.shape) != (4096, 192)
        or weight.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 exact split-K router is outside the FP32 K0..K7 lane"
        )
    execution_rows = 8
    flat = value.reshape(rows, 4096)
    if rows < execution_rows:
        flat = mx.concatenate(
            (
                flat,
                mx.zeros((execution_rows - rows, 4096), dtype=value.dtype),
            ),
            axis=0,
        )
    partial_kernel, threads = _partial_kernel_and_threads(
        value.dtype,
        weight.dtype,
        tiling,
    )
    (partials,) = partial_kernel(
        inputs=[mx.contiguous(flat), weight],
        grid=(tiling.stage1_threadgroups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tiling.grid_k_parts, execution_rows, 192)],
        output_dtypes=[mx.float32],
    )
    reduce_kernel = _build_hy3_router_fp32_reduce_kernel(tiling.grid_k_parts)
    (output,) = reduce_kernel(
        inputs=[partials],
        grid=(execution_rows * 192, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(execution_rows, 192)],
        output_dtypes=[mx.float32],
    )
    return output[:rows].reshape(*value.shape[:-1], 192)


def hy3_router_fp32_exact_splitk_route(
    value: mx.array,
    weight: mx.array,
    expert_bias: mx.array,
    *,
    available: bool | None = None,
    n_tile: int = 16,
    grid_k_parts: int = 8,
    operand_mode: str = "direct",
    k_tile: int | None = None,
    simd_groups_per_threadgroup: int = 1,
    top_k: int = 8,
    route_norm: bool = True,
    scaling_factor: float = 2.826,
    finalizer_mode: Literal["serial", "simd"] = "simd",
    simd_groups: Literal[1, 2, 3, 4, 5, 6, 7, 8] = 6,
    sigmoid_mode: Literal["precise", "fast-exp"] = "precise",
) -> tuple[mx.array, mx.array]:
    """Run K-major FP32 split-K R1 and consume its partials directly in R2.

    The source operands are promoted exactly to FP32, while split-K changes the
    non-associative reduction order relative to MLX's stock Steel GEMM. This is
    therefore a separately gated route-equivalent candidate, not a claim of
    bitwise stock-logit identity.
    """

    tiling = Hy3RouterFP32Tiling(
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
        operand_mode=operand_mode,
        k_tile=k_tile,
        simd_groups_per_threadgroup=simd_groups_per_threadgroup,
    )
    if value.ndim < 2:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input must include rows and hidden width"
        )
    rows = math.prod(int(dimension) for dimension in value.shape[:-1])
    supported = hy3_router_fp32_available() if available is None else bool(available)
    if (
        not supported
        or not 1 <= rows <= 8
        or int(value.shape[-1]) != 4096
        or value.dtype != mx.float32
        or weight.ndim != 2
        or tuple(int(dimension) for dimension in weight.shape) != (4096, 192)
        or weight.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 exact split-K router is outside the FP32 K0..K7 lane"
        )
    if (
        expert_bias.ndim != 1
        or tuple(int(dimension) for dimension in expert_bias.shape) != (192,)
        or expert_bias.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router expert bias must be FP32 with shape (192,)"
        )
    if int(top_k) != 8:
        raise Hy3RouterFP32Ineligible("Hy3 router fused finalizer requires top-8")
    if not route_norm:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router fused finalizer requires normalized routes"
        )
    if not math.isfinite(float(scaling_factor)) or float(scaling_factor) <= 0.0:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router scaling factor must be finite and positive"
        )
    if finalizer_mode not in ("serial", "simd"):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router finalizer mode must be serial or simd"
        )
    if not 1 <= int(simd_groups) <= 8:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router SIMDgroup count must be between 1 and 8"
        )
    _sigmoid_exp_call(sigmoid_mode, "-total")
    execution_rows = 8
    flat = value.reshape(rows, 4096)
    if rows < execution_rows:
        flat = mx.concatenate(
            (
                flat,
                mx.zeros((execution_rows - rows, 4096), dtype=value.dtype),
            ),
            axis=0,
        )
    partial_kernel, threads = _partial_kernel_and_threads(
        value.dtype,
        weight.dtype,
        tiling,
    )
    (partials,) = partial_kernel(
        inputs=[mx.contiguous(flat), weight],
        grid=(tiling.stage1_threadgroups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tiling.grid_k_parts, execution_rows, 192)],
        output_dtypes=[mx.float32],
    )
    if finalizer_mode == "simd":
        route_kernel = _build_hy3_router_fp32_route_simd_kernel(
            tiling.grid_k_parts,
            float(scaling_factor),
            sigmoid_mode,
            simd_groups=simd_groups,
        )
        threads = int(simd_groups) * 32
    else:
        route_kernel = _build_hy3_router_fp32_route_serial_kernel(
            tiling.grid_k_parts,
            float(scaling_factor),
            sigmoid_mode,
        )
        threads = 192
    expert_ids, router_scores = route_kernel(
        inputs=[partials, expert_bias],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, 8), (rows, 8)],
        output_dtypes=[mx.int32, mx.float32],
    )
    output_shape = (*value.shape[:-1], 8)
    return expert_ids.reshape(output_shape), router_scores.reshape(output_shape)


def hy3_router_fp32_route(
    value: mx.array,
    weight: mx.array,
    expert_bias: mx.array,
    *,
    available: bool | None = None,
    n_tile: int = 16,
    grid_k_parts: int = 4,
    operand_mode: str = "direct",
    k_tile: int | None = None,
    simd_groups_per_threadgroup: int = 1,
    top_k: int = 8,
    route_norm: bool = True,
    scaling_factor: float = 2.826,
    finalizer_mode: Literal["serial", "simd"] = "serial",
    simd_groups: Literal[1, 2, 3, 4, 5, 6, 7, 8] = 6,
    sigmoid_mode: Literal["precise", "fast-exp"] = "precise",
) -> tuple[mx.array, mx.array]:
    """Project and finalize source-Hy3 K0..K7 router rows."""

    tiling = Hy3RouterFP32Tiling(
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
        operand_mode=operand_mode,
        k_tile=k_tile,
        simd_groups_per_threadgroup=simd_groups_per_threadgroup,
    )
    if value.ndim < 2:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input must include rows and hidden width"
        )
    rows = 1
    for dimension in value.shape[:-1]:
        rows *= int(dimension)
    if weight.ndim != 2:
        raise Hy3RouterFP32Ineligible("Hy3 router weight must be rank two")
    input_width, experts = (int(dimension) for dimension in weight.shape)
    if int(value.shape[-1]) != input_width:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input and transposed-weight widths do not match"
        )
    if not hy3_router_fp32_eligible(
        rows=rows,
        input_width=input_width,
        experts=experts,
        input_dtype=value.dtype,
        weight_dtype=weight.dtype,
        available=available,
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router input is outside the exact G17 K0..K7 lane"
        )
    if (
        expert_bias.ndim != 1
        or tuple(int(dimension) for dimension in expert_bias.shape) != (192,)
        or expert_bias.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router expert bias must be FP32 with shape (192,)"
        )
    if int(top_k) != 8:
        raise Hy3RouterFP32Ineligible("Hy3 router fused finalizer requires top-8")
    if not route_norm:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router fused finalizer requires normalized routes"
        )
    if not math.isfinite(float(scaling_factor)) or float(scaling_factor) <= 0.0:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router scaling factor must be finite and positive"
        )
    if finalizer_mode not in ("serial", "simd"):
        raise Hy3RouterFP32Ineligible(
            "Hy3 router finalizer mode must be serial or simd"
        )
    if not 1 <= int(simd_groups) <= 8:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router SIMDgroup count must be between 1 and 8"
        )
    _sigmoid_exp_call(sigmoid_mode, "-total")
    execution_rows = 8
    flat = value.reshape(rows, input_width)
    if rows < execution_rows:
        flat = mx.concatenate(
            (
                flat,
                mx.zeros(
                    (execution_rows - rows, input_width),
                    dtype=value.dtype,
                ),
            ),
            axis=0,
        )
    partial_kernel, threads = _partial_kernel_and_threads(
        value.dtype,
        weight.dtype,
        tiling,
    )
    (partials,) = partial_kernel(
        inputs=[mx.contiguous(flat), weight],
        grid=(tiling.stage1_threadgroups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tiling.grid_k_parts, execution_rows, 192)],
        output_dtypes=[mx.float32],
    )
    if finalizer_mode == "simd":
        route_kernel = _build_hy3_router_fp32_route_simd_kernel(
            tiling.grid_k_parts,
            float(scaling_factor),
            sigmoid_mode,
            simd_groups=simd_groups,
        )
        threads = int(simd_groups) * 32
    else:
        route_kernel = _build_hy3_router_fp32_route_serial_kernel(
            tiling.grid_k_parts,
            float(scaling_factor),
            sigmoid_mode,
        )
        threads = 192
    expert_ids, router_scores = route_kernel(
        inputs=[partials, expert_bias],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, 8), (rows, 8)],
        output_dtypes=[mx.int32, mx.float32],
    )
    output_shape = (*value.shape[:-1], 8)
    return expert_ids.reshape(output_shape), router_scores.reshape(output_shape)
