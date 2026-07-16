"""Complete fixed-M4 streamed Hy3 affine-Q2 expert wave.

This module is the Issue #65 side of ``mtplx.hy3_fixed_k3``'s expert-wave
contract.  It deliberately owns neither routing nor target capture/commit:
the caller supplies authoritative assignment-aligned component-bank slots and
route weights, and this module returns the four route-reduced hidden rows.

The promoted arithmetic path retains MLX's tuned affine-Q2 ``gather_qmm`` for
all three projections.  Gate, up, exact SwiGLU, down, and route reduction are
each expressed once over the full flattened M4 x top-8 assignment axis.  The
source component banks are consumed directly and are never repacked, stacked,
or duplicated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlx_lm.models.activations import swiglu


HY3_M4_BATCH = 1
HY3_M4_ROWS = 4
HY3_M4_TOP_K = 8
HY3_M4_ASSIGNMENTS = HY3_M4_ROWS * HY3_M4_TOP_K
HY3_M4_HIDDEN_SIZE = 4096
HY3_M4_INTERMEDIATE_SIZE = 1536
HY3_M4_BITS = 2
HY3_M4_GROUP_SIZE = 64

_ASSIGNMENT_MASK = (1 << HY3_M4_ASSIGNMENTS) - 1
_ROW_MASK = (1 << HY3_M4_ROWS) - 1
_ASSIGNMENT_STAGES = (
    "gate_qmm",
    "up_qmm",
    "exact_swiglu",
    "down_qmm",
    "route_weight",
)
_QMM_STAGES = frozenset(("gate_qmm", "up_qmm", "down_qmm"))

_COMPONENT_LAYOUT: dict[str, tuple[tuple[int, ...], Any]] = {
    "gate_proj.weight": ((HY3_M4_INTERMEDIATE_SIZE, 256), mx.uint32),
    "gate_proj.scales": ((HY3_M4_INTERMEDIATE_SIZE, 64), mx.bfloat16),
    "gate_proj.biases": ((HY3_M4_INTERMEDIATE_SIZE, 64), mx.bfloat16),
    "up_proj.weight": ((HY3_M4_INTERMEDIATE_SIZE, 256), mx.uint32),
    "up_proj.scales": ((HY3_M4_INTERMEDIATE_SIZE, 64), mx.bfloat16),
    "up_proj.biases": ((HY3_M4_INTERMEDIATE_SIZE, 64), mx.bfloat16),
    "down_proj.weight": ((HY3_M4_HIDDEN_SIZE, 96), mx.uint32),
    "down_proj.scales": ((HY3_M4_HIDDEN_SIZE, 24), mx.bfloat16),
    "down_proj.biases": ((HY3_M4_HIDDEN_SIZE, 24), mx.bfloat16),
}


class Hy3M4ExpertWaveIneligible(ValueError):
    """Raised before QMM dispatch when a value leaves the fixed-M4 lane."""


@dataclass(frozen=True, slots=True)
class Hy3M4StageWork:
    """API-level dispatch and coverage evidence for one expert-wave stage.

    ``dispatches`` counts graph-operation requests, not post-compilation Metal
    command-buffer kernels: MLX may fuse elementwise work when the lazy graph is
    evaluated.  ``work_items`` plus ``coverage_mask`` proves that the fixed
    assignment or row domain is covered exactly once by that request.
    """

    name: str
    domain: str
    dispatches: int
    work_items: int
    coverage_mask: int

    @property
    def covers_all_assignments_once(self) -> bool:
        return (
            self.domain == "assignment"
            and self.dispatches == 1
            and self.work_items == HY3_M4_ASSIGNMENTS
            and self.coverage_mask == _ASSIGNMENT_MASK
            and self.coverage_mask.bit_count() == self.work_items
        )

    @property
    def covers_all_rows_once(self) -> bool:
        return (
            self.domain == "row"
            and self.dispatches == 1
            and self.work_items == HY3_M4_ROWS
            and self.coverage_mask == _ROW_MASK
            and self.coverage_mask.bit_count() == self.work_items
        )

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "name": self.name,
            "domain": self.domain,
            "dispatches": self.dispatches,
            "work_items": self.work_items,
            "coverage_mask": self.coverage_mask,
            "coverage_once": (
                self.covers_all_assignments_once
                if self.domain == "assignment"
                else self.covers_all_rows_once
            ),
        }


@dataclass(frozen=True, slots=True)
class Hy3M4ExpertWaveCounters:
    """Complete M4 x top-8 dispatch/work evidence consumed by Issue #63."""

    stages: tuple[Hy3M4StageWork, ...]

    @classmethod
    def complete(cls) -> Hy3M4ExpertWaveCounters:
        assignments = tuple(
            Hy3M4StageWork(
                name=name,
                domain="assignment",
                dispatches=1,
                work_items=HY3_M4_ASSIGNMENTS,
                coverage_mask=_ASSIGNMENT_MASK,
            )
            for name in _ASSIGNMENT_STAGES
        )
        reduction = Hy3M4StageWork(
            name="route_reduce",
            domain="row",
            dispatches=1,
            work_items=HY3_M4_ROWS,
            coverage_mask=_ROW_MASK,
        )
        return cls(stages=(*assignments, reduction))

    @property
    def assignment_count(self) -> int:
        return HY3_M4_ASSIGNMENTS

    @property
    def resident_weight_copies(self) -> int:
        return 0

    @property
    def per_row_host_loops(self) -> int:
        return 0

    @property
    def qmm_dispatches(self) -> int:
        return sum(
            stage.dispatches for stage in self.stages if stage.name in _QMM_STAGES
        )

    @property
    def total_dispatches(self) -> int:
        return sum(stage.dispatches for stage in self.stages)

    @property
    def assignments_covered_once(self) -> bool:
        assignment_stages = tuple(
            stage for stage in self.stages if stage.domain == "assignment"
        )
        return (
            tuple(stage.name for stage in assignment_stages) == _ASSIGNMENT_STAGES
            and all(stage.covers_all_assignments_once for stage in assignment_stages)
            and self.stage("route_reduce").covers_all_rows_once
        )

    def stage(self, name: str) -> Hy3M4StageWork:
        matches = tuple(stage for stage in self.stages if stage.name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_count": self.assignment_count,
            "assignments_covered_once": self.assignments_covered_once,
            "resident_weight_copies": self.resident_weight_copies,
            "per_row_host_loops": self.per_row_host_loops,
            "qmm_dispatches": self.qmm_dispatches,
            "total_dispatches": self.total_dispatches,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True, slots=True)
class Hy3M4ExpertWaveOutput:
    """Duck-compatible implementation of fixed-K3 ``Rows4ExpertWaveOutput``."""

    hidden_rows: Any
    stage_capture: Hy3M4ExpertWaveCounters

    @property
    def counters(self) -> Hy3M4ExpertWaveCounters:
        return self.stage_capture

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (HY3_M4_BATCH,)

    @property
    def rows(self) -> int:
        return HY3_M4_ROWS

    @property
    def hidden_size(self) -> int:
        return HY3_M4_HIDDEN_SIZE


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise Hy3M4ExpertWaveIneligible(f"{label} must expose a shape")
    try:
        shape = tuple(int(dimension) for dimension in raw)
    except (TypeError, ValueError) as exc:
        raise Hy3M4ExpertWaveIneligible(f"{label} has a non-integral shape") from exc
    return shape


def _require_tensor(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: Any,
    dtype_label: str,
) -> None:
    if _shape(value, label=label) != shape:
        raise Hy3M4ExpertWaveIneligible(f"{label} must have shape {list(shape)}")
    if getattr(value, "dtype", None) != dtype:
        raise Hy3M4ExpertWaveIneligible(f"{label} must use {dtype_label}")


def _validate_component_bank(bank_arrays: Mapping[str, Any]) -> int:
    expected_names = set(_COMPONENT_LAYOUT)
    if set(bank_arrays) != expected_names:
        missing = sorted(expected_names - set(bank_arrays))
        extra = sorted(set(bank_arrays) - expected_names)
        raise Hy3M4ExpertWaveIneligible(
            "fixed-M4 component-bank components differ from the Hy3-Q2 layout: "
            f"missing={missing}, extra={extra}"
        )
    weight_shape = _shape(
        bank_arrays["gate_proj.weight"],
        label="component bank gate_proj.weight",
    )
    capacity = weight_shape[0] if len(weight_shape) == 3 else 0
    if capacity <= 0:
        raise Hy3M4ExpertWaveIneligible(
            "fixed-M4 component-bank capacity must be positive"
        )
    for name, (tail, dtype) in _COMPONENT_LAYOUT.items():
        dtype_label = "uint32" if dtype == mx.uint32 else "BF16"
        _require_tensor(
            bank_arrays[name],
            label=f"component bank {name}",
            shape=(capacity, *tail),
            dtype=dtype,
            dtype_label=dtype_label,
        )
    return capacity


def _validate_resolved_slot_bounds(value: Any) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise Hy3M4ExpertWaveIneligible(
            "validated_slot_bounds must be an exact (minimum, maximum) tuple"
        )
    minimum, maximum = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise Hy3M4ExpertWaveIneligible(
            "validated_slot_bounds must contain exact integers"
        )
    if minimum > maximum:
        raise Hy3M4ExpertWaveIneligible("validated_slot_bounds minimum exceeds maximum")
    return minimum, maximum


def _tuned_qmm(
    values: Any,
    slot_indices: Any,
    bank_arrays: Mapping[str, Any],
    projection: str,
) -> Any:
    return mx.gather_qmm(
        values,
        bank_arrays[f"{projection}.weight"],
        bank_arrays[f"{projection}.scales"],
        bank_arrays[f"{projection}.biases"],
        rhs_indices=slot_indices,
        transpose=True,
        group_size=HY3_M4_GROUP_SIZE,
        bits=HY3_M4_BITS,
        mode="affine",
    )


def _execute_tuned_m4(
    hidden_rows: Any,
    slot_indices: Any,
    route_weights: Any,
    bank_arrays: Mapping[str, Any],
    combine_mode: str,
) -> Any:
    """Build one vectorized graph over the complete ordered assignment axis."""

    selected = mx.broadcast_to(
        hidden_rows.reshape((HY3_M4_ROWS, 1, HY3_M4_HIDDEN_SIZE)),
        (HY3_M4_ROWS, HY3_M4_TOP_K, HY3_M4_HIDDEN_SIZE),
    ).reshape((HY3_M4_ASSIGNMENTS, 1, 1, HY3_M4_HIDDEN_SIZE))
    qmm_indices = slot_indices.reshape((HY3_M4_ASSIGNMENTS, 1))

    gate = _tuned_qmm(selected, qmm_indices, bank_arrays, "gate_proj")
    up = _tuned_qmm(selected, qmm_indices, bank_arrays, "up_proj")
    activated = swiglu(gate, up)
    down = _tuned_qmm(activated, qmm_indices, bank_arrays, "down_proj")

    per_assignment = down.reshape(
        (HY3_M4_BATCH, HY3_M4_ROWS, HY3_M4_TOP_K, HY3_M4_HIDDEN_SIZE)
    )
    if combine_mode == "fp32":
        # Mirrors enable_moe_fp32_combine=True: FP32 products and reduction,
        # FP32 rows returned so the caller adds the shared expert before the
        # single BF16 rounding (#31 Arm-B precedent).
        return (per_assignment.astype(mx.float32) * route_weights[..., None]).sum(
            axis=-2
        )
    # Default mirrors the pinned Hy3 reference (enable_moe_fp32_combine=False):
    # BF16 weights, BF16 products, BF16 reduction — bitwise-identical to the
    # canonical lane's block combine.
    return (per_assignment * route_weights[..., None]).sum(axis=-2)


def hy3_q2_m4_expert_wave(
    hidden_rows: Any,
    slot_indices: Any,
    route_weights: Any,
    bank_arrays: Mapping[str, Any],
    *,
    validated_slot_bounds: tuple[int, int],
    combine_mode: str = "bf16",
) -> Hy3M4ExpertWaveOutput:
    """Execute the complete fixed-M4 Q2 expert wave.

    Args:
        hidden_rows: BF16 ``[1, 4, 4096]`` authoritative target rows.
        slot_indices: int32 ``[1, 4, 8]`` component-bank rows in router order.
        route_weights: ``[1, 4, 8]`` authoritative routing weights — BF16 in the default ``bf16`` combine mode (bitwise-matches the pinned reference), FP32 in ``fp32`` mode (mirrors enable_moe_fp32_combine; returns FP32 rows so the caller adds the shared expert before one BF16 rounding).
        bank_arrays: The original nine component-major resident arrays.  Their
            exact Hy3 affine-Q2 shapes are required and they are passed directly
            to MLX without a replacement or duplicate packing layout.
        validated_slot_bounds: Inclusive ``(minimum, maximum)`` bank indices
            already observed while the streaming runtime resolved and pinned
            the 32 assignments.  Reusing that host evidence avoids a new device
            synchronization or a per-row host scan in this arithmetic path.

    Returns:
        ``Hy3M4ExpertWaveOutput`` with BF16 ``hidden_rows`` of shape
        ``[1, 4, 4096]`` and stage counters proving complete assignment work.

    Raises:
        Hy3M4ExpertWaveIneligible: Before QMM dispatch for every unsupported
            batch/row/top-k shape, dtype, bank layout, or slot range.
    """

    _require_tensor(
        hidden_rows,
        label="fixed-M4 hidden rows",
        shape=(HY3_M4_BATCH, HY3_M4_ROWS, HY3_M4_HIDDEN_SIZE),
        dtype=mx.bfloat16,
        dtype_label="BF16",
    )
    _require_tensor(
        slot_indices,
        label="fixed-M4 slot indices",
        shape=(HY3_M4_BATCH, HY3_M4_ROWS, HY3_M4_TOP_K),
        dtype=mx.int32,
        dtype_label="int32",
    )
    if combine_mode not in ("bf16", "fp32"):
        raise Hy3M4ExpertWaveIneligible(
            "fixed-M4 combine mode must be 'bf16' or 'fp32'"
        )
    weights_dtype, weights_label = (
        (mx.float32, "FP32") if combine_mode == "fp32" else (mx.bfloat16, "BF16")
    )
    _require_tensor(
        route_weights,
        label="fixed-M4 route weights",
        shape=(HY3_M4_BATCH, HY3_M4_ROWS, HY3_M4_TOP_K),
        dtype=weights_dtype,
        dtype_label=weights_label,
    )
    if not isinstance(bank_arrays, Mapping):
        raise Hy3M4ExpertWaveIneligible(
            "fixed-M4 component bank must be a component mapping"
        )
    capacity = _validate_component_bank(bank_arrays)
    minimum_slot, maximum_slot = _validate_resolved_slot_bounds(validated_slot_bounds)
    if minimum_slot < 0:
        raise Hy3M4ExpertWaveIneligible("fixed-M4 slot indices must be non-negative")
    if maximum_slot >= capacity:
        raise Hy3M4ExpertWaveIneligible(
            f"fixed-M4 slot index exceeds component-bank capacity {capacity}"
        )

    output = _execute_tuned_m4(
        hidden_rows,
        slot_indices,
        route_weights,
        bank_arrays,
        combine_mode,
    )
    if _shape(output, label="fixed-M4 expert-wave output") != (
        HY3_M4_BATCH,
        HY3_M4_ROWS,
        HY3_M4_HIDDEN_SIZE,
    ) or getattr(output, "dtype", None) != (
        mx.float32 if combine_mode == "fp32" else mx.bfloat16
    ):
        raise RuntimeError(
            "fixed-M4 expert-wave graph violated its [1, 4, 4096] BF16 output contract"
        )
    return Hy3M4ExpertWaveOutput(
        hidden_rows=output,
        stage_capture=Hy3M4ExpertWaveCounters.complete(),
    )
