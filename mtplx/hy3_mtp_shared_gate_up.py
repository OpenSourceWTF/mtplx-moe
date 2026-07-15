"""Data-shape-specific Hy3 MTP M=1 gate/up/SwiGLU candidates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import mlx.core as mx
import mlx.nn as nn


HY3_MTP_HIDDEN_SIZE = 4096
HY3_MTP_SHARED_INTERMEDIATE_SIZE = 1536
HY3_MTP_N_TILES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
HY3_MTP_K_VECTORS = (1, 2, 4, 8, 16)
HY3_MTP_ROWS_PER_SIMDGROUP = (1, 2, 4, 8)
HY3_MTP_ACTIVATION_MODES = ("exact", "fast")


@dataclass(frozen=True)
class Hy3MTPGateUpCandidate:
    """One fixed M=1 gate/up/SwiGLU tiling candidate."""

    n_tile: int
    k_vector: int
    rows_per_simdgroup: int = 1
    activation_mode: str = "exact"

    def __post_init__(self) -> None:
        if self.n_tile not in HY3_MTP_N_TILES:
            raise ValueError(f"n_tile must be one of {HY3_MTP_N_TILES}")
        if self.k_vector not in HY3_MTP_K_VECTORS:
            raise ValueError(f"k_vector must be one of {HY3_MTP_K_VECTORS}")
        if self.rows_per_simdgroup not in HY3_MTP_ROWS_PER_SIMDGROUP:
            raise ValueError(
                "rows_per_simdgroup must be one of "
                f"{HY3_MTP_ROWS_PER_SIMDGROUP}"
            )
        if (
            HY3_MTP_SHARED_INTERMEDIATE_SIZE
            % (self.n_tile * self.rows_per_simdgroup)
            != 0
        ):
            raise ValueError("output tile must divide the fixed Hy3 MTP width")
        if self.activation_mode not in HY3_MTP_ACTIVATION_MODES:
            raise ValueError(
                f"activation_mode must be one of {HY3_MTP_ACTIVATION_MODES}"
            )

    @property
    def threads(self) -> int:
        return self.n_tile * 32

    @property
    def name(self) -> str:
        return (
            f"n{self.n_tile}_r{self.rows_per_simdgroup}"
            f"_v{self.k_vector}_{self.activation_mode}"
        )


def hy3_mtp_gate_up_candidates(
    *,
    activation_modes: tuple[str, ...] = ("exact",),
) -> tuple[Hy3MTPGateUpCandidate, ...]:
    """Return the exhaustive shape-specific tiling frontier."""

    unknown = set(activation_modes) - set(HY3_MTP_ACTIVATION_MODES)
    if unknown:
        raise ValueError(f"unknown activation modes: {sorted(unknown)}")
    return tuple(
        Hy3MTPGateUpCandidate(
            n_tile=n_tile,
            k_vector=k_vector,
            rows_per_simdgroup=rows_per_simdgroup,
            activation_mode=activation_mode,
        )
        for activation_mode, n_tile, rows_per_simdgroup, k_vector in product(
            activation_modes,
            HY3_MTP_N_TILES,
            HY3_MTP_ROWS_PER_SIMDGROUP,
            HY3_MTP_K_VECTORS,
        )
    )


def render_hy3_mtp_gate_up_source(candidate: Hy3MTPGateUpCandidate) -> str:
    """Render one fused BF16 gate/up/SwiGLU kernel for [1,1,4096]."""

    exp_call = (
        "metal::exp(metal::abs(gate_value))"
        if candidate.activation_mode == "exact"
        else "fast::exp(metal::abs(gate_value))"
    )
    return f"""
        constexpr uint K = {HY3_MTP_HIDDEN_SIZE};
        constexpr uint N = {HY3_MTP_SHARED_INTERMEDIATE_SIZE};
        constexpr uint N_TILE = {candidate.n_tile};
        constexpr uint ROWS_PER_SIMDGROUP = {candidate.rows_per_simdgroup};
        constexpr uint K_VECTOR = {candidate.k_vector};
        constexpr uint THREADS = N_TILE * 32;

        threadgroup T activation_tile[4096];
        uint thread_id = thread_index_in_threadgroup;
        uint simd_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint n_base = (threadgroup_position_in_grid.x * N_TILE + simd_id)
            * ROWS_PER_SIMDGROUP;

        for (uint k = thread_id; k < K; k += THREADS) {{
            activation_tile[k] = input_values[k];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float gate_sum[ROWS_PER_SIMDGROUP] = {{0.0f}};
        float up_sum[ROWS_PER_SIMDGROUP] = {{0.0f}};
        for (uint k_base = 0;
             k_base < K;
             k_base += 32 * K_VECTOR) {{
            _Pragma("clang loop unroll(full)")
            for (uint offset = 0; offset < K_VECTOR; ++offset) {{
                uint k = k_base + offset * 32 + lane;
                float activation = float(activation_tile[k]);
                _Pragma("clang loop unroll(full)")
                for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {{
                    uint n = n_base + row;
                    gate_sum[row] += (
                        activation * float(gate_weight[n * K + k])
                    );
                    up_sum[row] += (
                        activation * float(up_weight[n * K + k])
                    );
                }}
            }}
        }}
        _Pragma("clang loop unroll(full)")
        for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {{
            float gate_reduced = simd_sum(gate_sum[row]);
            float up_reduced = simd_sum(up_sum[row]);
            if (lane == 0) {{
                uint n = n_base + row;
                T gate_value = T(gate_reduced);
                T up_value = T(up_reduced);
                auto sigmoid_base = 1 / (1 + {exp_call});
                T sigmoid_value = gate_value < T(0)
                    ? sigmoid_base : 1 - sigmoid_base;
                T silu_value = gate_value * sigmoid_value;
                output_values[n] = T(silu_value * up_value);
            }}
        }}
    """


def hy3_mtp_gate_up_savings(
    candidate: Hy3MTPGateUpCandidate,
    *,
    depth: int,
) -> dict[str, int]:
    """Report logical savings and the shape-pinned kernel costs explicitly."""

    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be positive")
    outputs_per_threadgroup = candidate.n_tile * candidate.rows_per_simdgroup
    threadgroups_per_depth = (
        HY3_MTP_SHARED_INTERMEDIATE_SIZE // outputs_per_threadgroup
    )
    activation_bytes = HY3_MTP_HIDDEN_SIZE * 2
    intermediate_bytes = HY3_MTP_SHARED_INTERMEDIATE_SIZE * 2
    gate_up_weight_bytes = (
        2 * HY3_MTP_SHARED_INTERMEDIATE_SIZE * HY3_MTP_HIDDEN_SIZE * 2
    )
    threadgroups = threadgroups_per_depth * depth
    return {
        "depth": depth,
        # Two GEMVs plus SwiGLU become one kernel. The down GEMV is unchanged.
        "logical_dispatches_saved": 2 * depth,
        # Both graphs remain lazy until the same caller-owned evaluation boundary.
        "host_synchronizations_saved": 0,
        "gate_up_weight_bytes_required": gate_up_weight_bytes * depth,
        # Gate/up BF16 arrays are neither written nor read by a separate SwiGLU.
        "intermediate_device_bytes_avoided": 4 * intermediate_bytes * depth,
        "threadgroup_storage_bytes": activation_bytes,
        "threadgroups": threadgroups,
        "threadgroup_barriers": threadgroups,
        # Address-space load instructions; cache residency determines DRAM traffic.
        "input_fill_load_instruction_bytes": activation_bytes * threadgroups,
        "steady_extra_weight_bytes": 0,
    }


_KERNEL_CACHE: dict[Hy3MTPGateUpCandidate, Any] = {}


def build_hy3_mtp_gate_up_kernel(candidate: Hy3MTPGateUpCandidate) -> Any:
    """Construct a lazy Metal kernel; no work is dispatched here."""

    cached = _KERNEL_CACHE.get(candidate)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_hy3_mtp_gate_up_{candidate.name}",
        input_names=["input_values", "gate_weight", "up_weight"],
        output_names=["output_values"],
        source=render_hy3_mtp_gate_up_source(candidate),
    )
    _KERNEL_CACHE[candidate] = kernel
    return kernel


def _validate_arrays(
    value: mx.array,
    gate_weight: mx.array,
    up_weight: mx.array,
) -> None:
    expected_input = (1, 1, HY3_MTP_HIDDEN_SIZE)
    expected_weight = (HY3_MTP_SHARED_INTERMEDIATE_SIZE, HY3_MTP_HIDDEN_SIZE)
    if tuple(value.shape) != expected_input:
        raise ValueError(f"Hy3 MTP fused gate/up requires input {expected_input}")
    if tuple(gate_weight.shape) != expected_weight:
        raise ValueError(f"Hy3 MTP gate weight must have shape {expected_weight}")
    if tuple(up_weight.shape) != expected_weight:
        raise ValueError(f"Hy3 MTP up weight must have shape {expected_weight}")
    if any(array.dtype != mx.bfloat16 for array in (value, gate_weight, up_weight)):
        raise ValueError("Hy3 MTP fused gate/up requires BF16 inputs and weights")


def hy3_mtp_fused_gate_up_swiglu(
    value: mx.array,
    gate_weight: mx.array,
    up_weight: mx.array,
    *,
    candidate: Hy3MTPGateUpCandidate,
) -> mx.array:
    """Run one fused shape-pinned M=1 gate/up/SwiGLU operation."""

    _validate_arrays(value, gate_weight, up_weight)
    kernel = build_hy3_mtp_gate_up_kernel(candidate)
    groups = HY3_MTP_SHARED_INTERMEDIATE_SIZE // (
        candidate.n_tile * candidate.rows_per_simdgroup
    )
    (output,) = kernel(
        inputs=[
            mx.contiguous(value.reshape(HY3_MTP_HIDDEN_SIZE)),
            mx.contiguous(gate_weight),
            mx.contiguous(up_weight),
        ],
        template=[("T", mx.bfloat16)],
        grid=(candidate.threads * groups, 1, 1),
        threadgroup=(candidate.threads, 1, 1),
        output_shapes=[(HY3_MTP_SHARED_INTERMEDIATE_SIZE,)],
        output_dtypes=[mx.bfloat16],
    )
    return output.reshape(1, 1, HY3_MTP_SHARED_INTERMEDIATE_SIZE)


class MetalFusedMTPSharedMLP(nn.Module):
    """Benchmark/runtime module retaining source weights without duplicates."""

    def __init__(
        self,
        gate_weight: mx.array,
        up_weight: mx.array,
        down_weight: mx.array,
        *,
        candidate: Hy3MTPGateUpCandidate,
    ):
        super().__init__()
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.candidate = candidate

    def activate(self, value: mx.array) -> mx.array:
        """Return the fused gate/up/SwiGLU boundary before down projection."""

        return hy3_mtp_fused_gate_up_swiglu(
            value,
            self.gate_weight,
            self.up_weight,
            candidate=self.candidate,
        )

    def __call__(self, value: mx.array) -> mx.array:
        activated = self.activate(value)
        return mx.matmul(activated, self.down_weight.T)
