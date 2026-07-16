from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import mlx.core as mx
import pytest

from mtplx import hy3_expert_wave_m4 as wave_m4


@dataclass(frozen=True, slots=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: Any
    label: str

    def reshape(self, *raw_shape: Any) -> FakeTensor:
        shape = raw_shape[0] if len(raw_shape) == 1 else raw_shape
        shape = tuple(int(value) for value in shape)
        if shape.count(-1) > 1:
            raise ValueError("only one inferred dimension is supported")
        if -1 in shape:
            known = prod(value for value in shape if value != -1)
            shape = tuple(
                prod(self.shape) // known if value == -1 else value for value in shape
            )
        assert prod(shape) == prod(self.shape)
        return FakeTensor(shape, self.dtype, f"reshape({self.label})")

    def __getitem__(self, key: Any) -> FakeTensor:
        if key == (Ellipsis, None):
            return FakeTensor((*self.shape, 1), self.dtype, f"expand({self.label})")
        raise AssertionError(f"unexpected fake-tensor index: {key!r}")

    def __mul__(self, other: FakeTensor) -> FakeTensor:
        assert len(self.shape) == len(other.shape)
        assert all(
            left == right or left == 1 or right == 1
            for left, right in zip(self.shape, other.shape, strict=True)
        )
        assert self.dtype == other.dtype
        shape = tuple(
            max(left, right)
            for left, right in zip(self.shape, other.shape, strict=True)
        )
        return FakeTensor(shape, self.dtype, f"mul({self.label},{other.label})")

    def sum(self, *, axis: int) -> FakeTensor:
        resolved = axis if axis >= 0 else len(self.shape) + axis
        return FakeTensor(
            self.shape[:resolved] + self.shape[resolved + 1 :],
            self.dtype,
            f"sum({self.label})",
        )

    def astype(self, dtype: Any) -> FakeTensor:
        return FakeTensor(self.shape, dtype, f"cast({self.label})")


def _bank(capacity: int = 5) -> dict[str, FakeTensor]:
    return {
        "gate_proj.weight": FakeTensor((capacity, 1536, 256), mx.uint32, "gw"),
        "gate_proj.scales": FakeTensor((capacity, 1536, 64), mx.bfloat16, "gs"),
        "gate_proj.biases": FakeTensor((capacity, 1536, 64), mx.bfloat16, "gb"),
        "up_proj.weight": FakeTensor((capacity, 1536, 256), mx.uint32, "uw"),
        "up_proj.scales": FakeTensor((capacity, 1536, 64), mx.bfloat16, "us"),
        "up_proj.biases": FakeTensor((capacity, 1536, 64), mx.bfloat16, "ub"),
        "down_proj.weight": FakeTensor((capacity, 4096, 96), mx.uint32, "dw"),
        "down_proj.scales": FakeTensor((capacity, 4096, 24), mx.bfloat16, "ds"),
        "down_proj.biases": FakeTensor((capacity, 4096, 24), mx.bfloat16, "db"),
    }


def _inputs() -> tuple[FakeTensor, FakeTensor, FakeTensor]:
    return (
        FakeTensor((1, 4, 4096), mx.bfloat16, "hidden"),
        FakeTensor((1, 4, 8), mx.int32, "slots"),
        FakeTensor((1, 4, 8), mx.float32, "weights"),
    )


def test_fixed_m4_geometry_and_stage_capture_cover_every_assignment_once() -> None:
    counters = wave_m4.Hy3M4ExpertWaveCounters.complete()

    assert wave_m4.HY3_M4_ROWS == 4
    assert wave_m4.HY3_M4_TOP_K == 8
    assert wave_m4.HY3_M4_ASSIGNMENTS == 32
    assert counters.assignment_count == 32
    assert counters.assignments_covered_once
    assert counters.resident_weight_copies == 0
    assert counters.per_row_host_loops == 0
    assert counters.qmm_dispatches == 3
    assert counters.total_dispatches == 6
    assert counters.stage("route_reduce").work_items == 4
    for stage_name in (
        "gate_qmm",
        "up_qmm",
        "exact_swiglu",
        "down_qmm",
        "route_weight",
    ):
        stage = counters.stage(stage_name)
        assert stage.dispatches == 1
        assert stage.work_items == 32
        assert stage.coverage_mask == (1 << 32) - 1
        assert stage.covers_all_assignments_once


def test_m4_runtime_uses_three_tuned_qmms_without_copying_source_banks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_rows, slot_indices, route_weights = _inputs()
    bank = _bank()
    calls: list[dict[str, Any]] = []

    def fake_broadcast(value: FakeTensor, shape: tuple[int, ...]) -> FakeTensor:
        assert value.shape == (4, 1, 4096)
        assert shape == (4, 8, 4096)
        return FakeTensor(shape, value.dtype, f"broadcast({value.label})")

    projection_by_weight = {
        id(bank["gate_proj.weight"]): ("gate", 1536),
        id(bank["up_proj.weight"]): ("up", 1536),
        id(bank["down_proj.weight"]): ("down", 4096),
    }

    def fake_qmm(
        values: FakeTensor,
        weight: FakeTensor,
        scales: FakeTensor,
        biases: FakeTensor,
        **kwargs: Any,
    ) -> FakeTensor:
        projection, width = projection_by_weight[id(weight)]
        calls.append(
            {
                "projection": projection,
                "values": values,
                "weight": weight,
                "scales": scales,
                "biases": biases,
                "kwargs": kwargs,
            }
        )
        assert values.shape[0] == 32
        assert kwargs == {
            "rhs_indices": slot_indices.reshape(32, 1),
            "transpose": True,
            "group_size": 64,
            "bits": 2,
            "mode": "affine",
        }
        return FakeTensor((32, 1, 1, width), mx.bfloat16, projection)

    def fake_swiglu(gate: FakeTensor, up: FakeTensor) -> FakeTensor:
        assert gate.label == "gate"
        assert up.label == "up"
        return FakeTensor(gate.shape, gate.dtype, "exact-swiglu")

    monkeypatch.setattr(wave_m4.mx, "broadcast_to", fake_broadcast)
    monkeypatch.setattr(wave_m4.mx, "gather_qmm", fake_qmm)
    monkeypatch.setattr(wave_m4, "swiglu", fake_swiglu)
    result = wave_m4.hy3_q2_m4_expert_wave(
        hidden_rows,
        slot_indices,
        route_weights,
        bank,
        validated_slot_bounds=(0, 4),
    )

    assert result.hidden_rows.shape == (1, 4, 4096)
    assert result.hidden_rows.dtype == mx.bfloat16
    assert result.batch_shape == (1,)
    assert result.rows == 4
    assert result.hidden_size == 4096
    assert result.stage_capture.assignments_covered_once
    assert [call["projection"] for call in calls] == ["gate", "up", "down"]
    for projection, call in zip(("gate_proj", "up_proj", "down_proj"), calls):
        assert call["weight"] is bank[f"{projection}.weight"]
        assert call["scales"] is bank[f"{projection}.scales"]
        assert call["biases"] is bank[f"{projection}.biases"]


@pytest.mark.parametrize(
    ("input_index", "replacement", "match"),
    (
        (0, FakeTensor((1, 3, 4096), mx.bfloat16, "hidden"), r"\[1, 4, 4096\]"),
        (0, FakeTensor((1, 4, 4096), mx.float32, "hidden"), "BF16"),
        (1, FakeTensor((1, 4, 7), mx.int32, "slots"), r"\[1, 4, 8\]"),
        (1, FakeTensor((1, 4, 8), mx.uint32, "slots"), "int32"),
        (2, FakeTensor((1, 4, 7), mx.float32, "weights"), r"\[1, 4, 8\]"),
        (2, FakeTensor((1, 4, 8), mx.bfloat16, "weights"), "FP32"),
    ),
)
def test_m4_runtime_fails_closed_on_noncontract_inputs(
    input_index: int,
    replacement: FakeTensor,
    match: str,
) -> None:
    values = list(_inputs())
    values[input_index] = replacement

    with pytest.raises(wave_m4.Hy3M4ExpertWaveIneligible, match=match):
        wave_m4.hy3_q2_m4_expert_wave(
            *values,
            _bank(),
            validated_slot_bounds=(0, 4),
        )


def test_m4_runtime_fails_closed_on_bank_layout_or_slot_bounds() -> None:
    hidden_rows, slot_indices, route_weights = _inputs()
    missing = _bank()
    missing.pop("down_proj.weight")
    with pytest.raises(wave_m4.Hy3M4ExpertWaveIneligible, match="components"):
        wave_m4.hy3_q2_m4_expert_wave(
            hidden_rows,
            slot_indices,
            route_weights,
            missing,
            validated_slot_bounds=(0, 4),
        )

    wrong_shape = _bank()
    wrong_shape["down_proj.weight"] = FakeTensor((5, 4096, 95), mx.uint32, "wrong-down")
    with pytest.raises(wave_m4.Hy3M4ExpertWaveIneligible, match="down_proj.weight"):
        wave_m4.hy3_q2_m4_expert_wave(
            hidden_rows,
            slot_indices,
            route_weights,
            wrong_shape,
            validated_slot_bounds=(0, 4),
        )

    with pytest.raises(wave_m4.Hy3M4ExpertWaveIneligible, match="slot indices"):
        wave_m4.hy3_q2_m4_expert_wave(
            hidden_rows,
            slot_indices,
            route_weights,
            _bank(),
            validated_slot_bounds=(-1, 4),
        )

    with pytest.raises(wave_m4.Hy3M4ExpertWaveIneligible, match="capacity"):
        wave_m4.hy3_q2_m4_expert_wave(
            hidden_rows,
            slot_indices,
            route_weights,
            _bank(capacity=5),
            validated_slot_bounds=(0, 5),
        )
