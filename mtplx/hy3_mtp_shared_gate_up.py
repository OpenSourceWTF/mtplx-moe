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
HY3_MTP_REDUCTION_LAYOUTS = (
    "striped",
    "striped_tree",
    "stock_tn4",
    "stock_tn4_sum",
)
HY3_MTP_INPUT_MODES = ("threadgroup", "threadgroup_f32", "direct")
HY3_MTP_WEIGHT_LAYOUTS = ("split", "packed2")
HY3_MTP_SHARED_KERNELS = ("stock", "metal-exact")


@dataclass(frozen=True)
class Hy3MTPGateUpCandidate:
    """One fixed M=1 gate/up/SwiGLU tiling candidate."""

    n_tile: int
    k_vector: int
    rows_per_simdgroup: int = 1
    activation_mode: str = "exact"
    reduction_layout: str = "striped"
    input_mode: str = "threadgroup"
    weight_layout: str = "split"

    def __post_init__(self) -> None:
        if self.n_tile not in HY3_MTP_N_TILES:
            raise ValueError(f"n_tile must be one of {HY3_MTP_N_TILES}")
        if self.k_vector not in HY3_MTP_K_VECTORS:
            raise ValueError(f"k_vector must be one of {HY3_MTP_K_VECTORS}")
        if self.rows_per_simdgroup not in HY3_MTP_ROWS_PER_SIMDGROUP:
            raise ValueError(
                f"rows_per_simdgroup must be one of {HY3_MTP_ROWS_PER_SIMDGROUP}"
            )
        if (
            HY3_MTP_SHARED_INTERMEDIATE_SIZE % (self.n_tile * self.rows_per_simdgroup)
            != 0
        ):
            raise ValueError("output tile must divide the fixed Hy3 MTP width")
        if self.activation_mode not in HY3_MTP_ACTIVATION_MODES:
            raise ValueError(
                f"activation_mode must be one of {HY3_MTP_ACTIVATION_MODES}"
            )
        if self.reduction_layout not in HY3_MTP_REDUCTION_LAYOUTS:
            raise ValueError(
                f"reduction_layout must be one of {HY3_MTP_REDUCTION_LAYOUTS}"
            )
        if self.reduction_layout.startswith("stock_tn4") and self.k_vector != 4:
            raise ValueError("stock_tn4 reduction layouts require k_vector=4")
        if self.input_mode not in HY3_MTP_INPUT_MODES:
            raise ValueError(f"input_mode must be one of {HY3_MTP_INPUT_MODES}")
        if self.weight_layout not in HY3_MTP_WEIGHT_LAYOUTS:
            raise ValueError(f"weight_layout must be one of {HY3_MTP_WEIGHT_LAYOUTS}")

    @property
    def threads(self) -> int:
        return self.n_tile * 32

    @property
    def name(self) -> str:
        name = (
            f"n{self.n_tile}_r{self.rows_per_simdgroup}"
            f"_v{self.k_vector}_{self.activation_mode}"
        )
        if self.reduction_layout != "striped":
            name += f"_{self.reduction_layout}"
        if self.input_mode != "threadgroup":
            name += f"_{self.input_mode}"
        if self.weight_layout != "split":
            name += f"_{self.weight_layout}"
        return name


HY3_MTP_K3_EXACT_CANDIDATE = Hy3MTPGateUpCandidate(
    n_tile=24,
    k_vector=16,
    rows_per_simdgroup=2,
)


def hy3_mtp_gate_up_candidates(
    *,
    activation_modes: tuple[str, ...] = ("exact",),
    reduction_layouts: tuple[str, ...] = ("striped",),
    input_modes: tuple[str, ...] = ("threadgroup",),
    weight_layouts: tuple[str, ...] = ("split",),
) -> tuple[Hy3MTPGateUpCandidate, ...]:
    """Return the exhaustive shape-specific tiling frontier."""

    unknown = set(activation_modes) - set(HY3_MTP_ACTIVATION_MODES)
    if unknown:
        raise ValueError(f"unknown activation modes: {sorted(unknown)}")
    unknown_layouts = set(reduction_layouts) - set(HY3_MTP_REDUCTION_LAYOUTS)
    if unknown_layouts:
        raise ValueError(f"unknown reduction layouts: {sorted(unknown_layouts)}")
    unknown_input_modes = set(input_modes) - set(HY3_MTP_INPUT_MODES)
    if unknown_input_modes:
        raise ValueError(f"unknown input modes: {sorted(unknown_input_modes)}")
    unknown_weight_layouts = set(weight_layouts) - set(HY3_MTP_WEIGHT_LAYOUTS)
    if unknown_weight_layouts:
        raise ValueError(f"unknown weight layouts: {sorted(unknown_weight_layouts)}")
    return tuple(
        Hy3MTPGateUpCandidate(
            n_tile=n_tile,
            k_vector=k_vector,
            rows_per_simdgroup=rows_per_simdgroup,
            activation_mode=activation_mode,
            reduction_layout=reduction_layout,
            input_mode=input_mode,
            weight_layout=weight_layout,
        )
        for weight_layout, input_mode, reduction_layout, activation_mode, n_tile, rows_per_simdgroup, k_vector in product(
            weight_layouts,
            input_modes,
            reduction_layouts,
            activation_modes,
            HY3_MTP_N_TILES,
            HY3_MTP_ROWS_PER_SIMDGROUP,
            HY3_MTP_K_VECTORS,
        )
        if not reduction_layout.startswith("stock_tn4") or k_vector == 4
    )


def render_hy3_mtp_gate_up_source(candidate: Hy3MTPGateUpCandidate) -> str:
    """Render one fused BF16 gate/up/SwiGLU kernel for [1,1,4096]."""

    exp_call = (
        "metal::exp(metal::abs(gate_value))"
        if candidate.activation_mode == "exact"
        else "fast::exp(metal::abs(gate_value))"
    )
    uses_stock_tn4_k_order = candidate.reduction_layout.startswith("stock_tn4")
    uses_explicit_tree = candidate.reduction_layout in {
        "striped_tree",
        "stock_tn4",
    }
    k_index = (
        "k_base + lane * K_VECTOR + offset"
        if uses_stock_tn4_k_order
        else "k_base + offset * 32 + lane"
    )
    reduction = (
        """float gate_reduced = gate_sum[row];
            float up_reduced = up_sum[row];
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                gate_reduced += simd_shuffle_down(gate_reduced, delta);
                up_reduced += simd_shuffle_down(up_reduced, delta);
            }"""
        if uses_explicit_tree
        else """float gate_reduced = simd_sum(gate_sum[row]);
            float up_reduced = simd_sum(up_sum[row]);"""
    )
    if candidate.input_mode == "threadgroup":
        input_prelude = """threadgroup T activation_tile[4096];
        for (uint k = thread_id; k < K; k += THREADS) {
            activation_tile[k] = input_values[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"""
    elif candidate.input_mode == "threadgroup_f32":
        input_prelude = """threadgroup float activation_tile[4096];
        for (uint k = thread_id; k < K; k += THREADS) {
            activation_tile[k] = float(input_values[k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"""
    else:
        input_prelude = ""
    activation_source = (
        "input_values[k]" if candidate.input_mode == "direct" else "activation_tile[k]"
    )
    if candidate.weight_layout == "packed2":
        weight_prelude = (
            "device const vec<T, 2>* packed_pairs = "
            "reinterpret_cast<device const vec<T, 2>*>(packed_weight);"
        )
        weight_load = "vec<T, 2> weight_pair = packed_pairs[n * K + k];"
        gate_weight_value = "weight_pair[0]"
        up_weight_value = "weight_pair[1]"
    else:
        weight_prelude = ""
        weight_load = ""
        gate_weight_value = "gate_weight[n * K + k]"
        up_weight_value = "up_weight[n * K + k]"
    return f"""
        constexpr uint K = {HY3_MTP_HIDDEN_SIZE};
        constexpr uint N = {HY3_MTP_SHARED_INTERMEDIATE_SIZE};
        constexpr uint N_TILE = {candidate.n_tile};
        constexpr uint ROWS_PER_SIMDGROUP = {candidate.rows_per_simdgroup};
        constexpr uint K_VECTOR = {candidate.k_vector};
        constexpr uint THREADS = N_TILE * 32;

        uint thread_id = thread_index_in_threadgroup;
        uint simd_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint n_base = (threadgroup_position_in_grid.x * N_TILE + simd_id)
            * ROWS_PER_SIMDGROUP;
        {weight_prelude}

        {input_prelude}

        float gate_sum[ROWS_PER_SIMDGROUP] = {{0.0f}};
        float up_sum[ROWS_PER_SIMDGROUP] = {{0.0f}};
        for (uint k_base = 0;
             k_base < K;
             k_base += 32 * K_VECTOR) {{
            _Pragma("clang loop unroll(full)")
            for (uint offset = 0; offset < K_VECTOR; ++offset) {{
                uint k = {k_index};
                float activation = float({activation_source});
                _Pragma("clang loop unroll(full)")
                for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {{
                    uint n = n_base + row;
                    {weight_load}
                    gate_sum[row] += (
                        activation * float({gate_weight_value})
                    );
                    up_sum[row] += (
                        activation * float({up_weight_value})
                    );
                }}
            }}
        }}
        _Pragma("clang loop unroll(full)")
        for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {{
            {reduction}
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
    threadgroups_per_depth = HY3_MTP_SHARED_INTERMEDIATE_SIZE // outputs_per_threadgroup
    activation_bytes = HY3_MTP_HIDDEN_SIZE * 2
    intermediate_bytes = HY3_MTP_SHARED_INTERMEDIATE_SIZE * 2
    gate_up_weight_bytes = (
        2 * HY3_MTP_SHARED_INTERMEDIATE_SIZE * HY3_MTP_HIDDEN_SIZE * 2
    )
    threadgroups = threadgroups_per_depth * depth
    uses_threadgroup_input = candidate.input_mode != "direct"
    threadgroup_storage_bytes = (
        HY3_MTP_HIDDEN_SIZE * 4
        if candidate.input_mode == "threadgroup_f32"
        else activation_bytes
        if uses_threadgroup_input
        else 0
    )
    compute_input_repetitions = candidate.n_tile * threadgroups
    input_device_load_instruction_bytes = (
        activation_bytes * threadgroups
        if uses_threadgroup_input
        else activation_bytes * compute_input_repetitions
    )
    input_threadgroup_load_instruction_bytes = (
        threadgroup_storage_bytes * compute_input_repetitions
        if uses_threadgroup_input
        else 0
    )
    input_bf16_to_fp32_conversions = (
        HY3_MTP_HIDDEN_SIZE * threadgroups
        if candidate.input_mode == "threadgroup_f32"
        else HY3_MTP_HIDDEN_SIZE * compute_input_repetitions
    )
    return {
        "depth": depth,
        # Two GEMVs plus SwiGLU become one kernel. The down GEMV is unchanged.
        "logical_dispatches_saved": 2 * depth,
        # Both graphs remain lazy until the same caller-owned evaluation boundary.
        "host_synchronizations_saved": 0,
        "gate_up_weight_bytes_required": gate_up_weight_bytes * depth,
        # Gate/up BF16 arrays are neither written nor read by a separate SwiGLU.
        "intermediate_device_bytes_avoided": 4 * intermediate_bytes * depth,
        "threadgroup_storage_bytes": threadgroup_storage_bytes,
        "threadgroups": threadgroups,
        "threadgroup_barriers": threadgroups if uses_threadgroup_input else 0,
        # Address-space load instructions; cache residency determines DRAM traffic.
        "input_fill_load_instruction_bytes": (
            activation_bytes * threadgroups if uses_threadgroup_input else 0
        ),
        "input_device_load_instruction_bytes": input_device_load_instruction_bytes,
        "input_threadgroup_load_instruction_bytes": (
            input_threadgroup_load_instruction_bytes
        ),
        "input_bf16_to_fp32_conversions": input_bf16_to_fp32_conversions,
        "steady_extra_weight_bytes": 0,
    }


_KERNEL_CACHE: dict[Hy3MTPGateUpCandidate, Any] = {}


def build_hy3_mtp_gate_up_kernel(candidate: Hy3MTPGateUpCandidate) -> Any:
    """Construct a lazy Metal kernel; no work is dispatched here."""

    cached = _KERNEL_CACHE.get(candidate)
    if cached is not None:
        return cached
    input_names = (
        ["input_values", "packed_weight"]
        if candidate.weight_layout == "packed2"
        else ["input_values", "gate_weight", "up_weight"]
    )
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_hy3_mtp_gate_up_{candidate.name}",
        input_names=input_names,
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


def _validate_packed_arrays(value: mx.array, packed_weight: mx.array) -> None:
    expected_input = (1, 1, HY3_MTP_HIDDEN_SIZE)
    expected_packed = (
        HY3_MTP_SHARED_INTERMEDIATE_SIZE,
        HY3_MTP_HIDDEN_SIZE,
        2,
    )
    if tuple(value.shape) != expected_input:
        raise ValueError(f"Hy3 MTP fused gate/up requires input {expected_input}")
    if tuple(packed_weight.shape) != expected_packed:
        raise ValueError(
            f"Hy3 MTP packed gate/up weight must have shape {expected_packed}"
        )
    if value.dtype != mx.bfloat16 or packed_weight.dtype != mx.bfloat16:
        raise ValueError("Hy3 MTP packed gate/up requires BF16 input and weights")


def hy3_mtp_fused_gate_up_swiglu(
    value: mx.array,
    gate_weight: mx.array,
    up_weight: mx.array,
    *,
    candidate: Hy3MTPGateUpCandidate,
) -> mx.array:
    """Run one fused shape-pinned M=1 gate/up/SwiGLU operation."""

    if candidate.weight_layout != "split":
        raise ValueError("split gate/up arrays require weight_layout='split'")
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


def hy3_mtp_fused_packed_gate_up_swiglu(
    value: mx.array,
    packed_weight: mx.array,
    *,
    candidate: Hy3MTPGateUpCandidate,
) -> mx.array:
    """Run fused M=1 gate/up/SwiGLU from [N,K,2] interleaved BF16 pairs."""

    if candidate.weight_layout != "packed2":
        raise ValueError("packed gate/up array requires weight_layout='packed2'")
    _validate_packed_arrays(value, packed_weight)
    kernel = build_hy3_mtp_gate_up_kernel(candidate)
    groups = HY3_MTP_SHARED_INTERMEDIATE_SIZE // (
        candidate.n_tile * candidate.rows_per_simdgroup
    )
    (output,) = kernel(
        inputs=[
            mx.contiguous(value.reshape(HY3_MTP_HIDDEN_SIZE)),
            mx.contiguous(packed_weight),
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


class DepthGatedMTPSharedMLP(nn.Module):
    """Select exact Metal only for its proven fixed speculative depth.

    The wrapper owns only the original shared MLP.  The Metal path reads that
    module's existing gate/up/down arrays, so selecting it creates no duplicate
    resident weights.  ``configure_depth`` swaps a bound callable before a
    generation begins; ``__call__`` contains no per-dispatch flag branch.
    """

    def __init__(
        self,
        stock: Any,
        *,
        candidate: Hy3MTPGateUpCandidate = HY3_MTP_K3_EXACT_CANDIDATE,
        target_depth: int = 3,
    ) -> None:
        super().__init__()
        if (
            candidate.activation_mode != "exact"
            or candidate.weight_layout != "split"
        ):
            raise ValueError(
                "depth-gated runtime requires an exact split-weight candidate"
            )
        if isinstance(target_depth, bool) or not isinstance(target_depth, int):
            raise TypeError("target_depth must be an integer")
        if target_depth < 1:
            raise ValueError("target_depth must be positive")
        self.stock = stock
        self.candidate = candidate
        self.target_depth = target_depth
        self.active_mode = "stock"
        self._active_call = self._call_stock

    def _call_stock(self, value: mx.array) -> mx.array:
        return self.stock(value)

    def _call_exact(self, value: mx.array) -> mx.array:
        activated = hy3_mtp_fused_gate_up_swiglu(
            value,
            self.stock.gate_proj.weight,
            self.stock.up_proj.weight,
            candidate=self.candidate,
        )
        return self.stock.down_proj(activated)

    def configure_depth(self, depth: int | None) -> str:
        """Swap the active implementation for one complete generation."""

        if depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int)
        ):
            raise TypeError("depth must be an integer or None")
        if depth is not None and depth < 0:
            raise ValueError("depth must be non-negative")
        if depth == self.target_depth:
            self.active_mode = "metal-exact"
            self._active_call = self._call_exact
        else:
            self.active_mode = "stock"
            self._active_call = self._call_stock
        return self.active_mode

    def __call__(self, value: mx.array) -> mx.array:
        return self._active_call(value)


def install_depth_gated_mtp_shared_mlp(
    mtp: Any,
    *,
    target_depth: int = 3,
    candidate: Hy3MTPGateUpCandidate = HY3_MTP_K3_EXACT_CANDIDATE,
) -> int:
    """Wrap every Hy3 MTP shared MLP after strict weight loading."""

    layers = getattr(mtp, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise ValueError("Hy3 MTP module exposes no layers to wrap")
    installed = 0
    for layer in layers:
        shared = layer.mtp_block.mlp.shared_mlp
        if isinstance(shared, DepthGatedMTPSharedMLP):
            if shared.target_depth != target_depth or shared.candidate != candidate:
                raise ValueError("Hy3 MTP shared MLP is already wrapped differently")
            installed += 1
            continue
        required = ("gate_proj", "up_proj", "down_proj")
        if any(not hasattr(shared, name) for name in required):
            raise ValueError("Hy3 MTP shared MLP does not expose split projections")
        gate_weight = shared.gate_proj.weight
        up_weight = shared.up_proj.weight
        down_weight = shared.down_proj.weight
        expected_gate_up = (
            HY3_MTP_SHARED_INTERMEDIATE_SIZE,
            HY3_MTP_HIDDEN_SIZE,
        )
        expected_down = (
            HY3_MTP_HIDDEN_SIZE,
            HY3_MTP_SHARED_INTERMEDIATE_SIZE,
        )
        if tuple(gate_weight.shape) != expected_gate_up:
            raise ValueError(
                f"Hy3 MTP gate weight must have shape {expected_gate_up}"
            )
        if tuple(up_weight.shape) != expected_gate_up:
            raise ValueError(f"Hy3 MTP up weight must have shape {expected_gate_up}")
        if tuple(down_weight.shape) != expected_down:
            raise ValueError(f"Hy3 MTP down weight must have shape {expected_down}")
        if any(
            weight.dtype != mx.bfloat16
            for weight in (gate_weight, up_weight, down_weight)
        ):
            raise ValueError("depth-gated Hy3 MTP shared weights must be BF16")
        layer.mtp_block.mlp.shared_mlp = DepthGatedMTPSharedMLP(
            shared,
            candidate=candidate,
            target_depth=target_depth,
        )
        installed += 1
    return installed


class MetalPackedFusedMTPSharedMLP(nn.Module):
    """Fused shared MLP retaining only packed gate/up pairs and down weight."""

    def __init__(
        self,
        packed_weight: mx.array,
        down_weight: mx.array,
        *,
        candidate: Hy3MTPGateUpCandidate,
    ):
        super().__init__()
        self.packed_weight = packed_weight
        self.down_weight = down_weight
        self.candidate = candidate

    def activate(self, value: mx.array) -> mx.array:
        """Return the fused gate/up/SwiGLU boundary before down projection."""

        return hy3_mtp_fused_packed_gate_up_swiglu(
            value,
            self.packed_weight,
            candidate=self.candidate,
        )

    def __call__(self, value: mx.array) -> mx.array:
        activated = self.activate(value)
        return mx.matmul(activated, self.down_weight.T)
