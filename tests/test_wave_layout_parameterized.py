"""Parameterized fixed-M4 expert-wave layout + end-to-end parity (CPU-only).

Covers the generalization of ``mtplx.hy3_expert_wave_m4`` from hardcoded
Hy3/gs64 constants to a spec-derived component layout:

* ``component_layout`` shape derivation for Hy3-gs64 (byte-identical to the old
  hardcoded dict), Hy3-gs128 (the oQ2e checkpoint), and GLM dims.
* Divisibility fail-closed.
* End-to-end wave parity vs a straightforward per-assignment reference at both
  gs64 and gs128 on real quantized banks.
* Group-size eligibility fail-closed (a bank shaped for one group size is
  rejected under the other).

Every test pins the default device to the CPU: the campaign forbids taking the
GPU lock from the test lane.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_lm.models.activations import swiglu

from mtplx import hy3_expert_wave_m4 as wave_m4
from mtplx.hy3_expert_wave_m4 import (
    Hy3M4ExpertWaveIneligible,
    component_layout,
    hy3_q2_m4_expert_wave,
)

mx.set_default_device(mx.cpu)


# The exact tail layout the module hardcoded before parameterization, kept here
# as an independent golden so a regression in the derivation is caught
# field-by-field rather than against the function under test.
_OLD_HARDCODED_LAYOUT: dict[str, tuple[tuple[int, ...], object]] = {
    "gate_proj.weight": ((1536, 256), mx.uint32),
    "gate_proj.scales": ((1536, 64), mx.bfloat16),
    "gate_proj.biases": ((1536, 64), mx.bfloat16),
    "up_proj.weight": ((1536, 256), mx.uint32),
    "up_proj.scales": ((1536, 64), mx.bfloat16),
    "up_proj.biases": ((1536, 64), mx.bfloat16),
    "down_proj.weight": ((4096, 96), mx.uint32),
    "down_proj.scales": ((4096, 24), mx.bfloat16),
    "down_proj.biases": ((4096, 24), mx.bfloat16),
}


def test_component_layout_gs64_reproduces_old_hardcoded_dict() -> None:
    layout = component_layout(4096, 1536, 64, 2)

    assert set(layout) == set(_OLD_HARDCODED_LAYOUT)
    # Order must be preserved (validators and call sites iterate the dict).
    assert list(layout) == list(_OLD_HARDCODED_LAYOUT)
    for name, expected in _OLD_HARDCODED_LAYOUT.items():
        assert layout[name] == expected, name


def test_component_layout_gs128_changes_only_scale_bias_columns() -> None:
    layout = component_layout(4096, 1536, 128, 2)

    # 2-bit packing is group-size-independent: weight columns are unchanged.
    assert layout["gate_proj.weight"] == ((1536, 256), mx.uint32)
    assert layout["up_proj.weight"] == ((1536, 256), mx.uint32)
    assert layout["down_proj.weight"] == ((4096, 96), mx.uint32)
    # Scales/biases halve their columns (input_size // 128).
    assert layout["gate_proj.scales"] == ((1536, 32), mx.bfloat16)
    assert layout["gate_proj.biases"] == ((1536, 32), mx.bfloat16)
    assert layout["up_proj.scales"] == ((1536, 32), mx.bfloat16)
    assert layout["up_proj.biases"] == ((1536, 32), mx.bfloat16)
    assert layout["down_proj.scales"] == ((4096, 12), mx.bfloat16)
    assert layout["down_proj.biases"] == ((4096, 12), mx.bfloat16)


def test_component_layout_glm_dims() -> None:
    layout = component_layout(6144, 2048, 64, 2)

    assert layout["gate_proj.weight"] == ((2048, 384), mx.uint32)
    assert layout["up_proj.weight"] == ((2048, 384), mx.uint32)
    assert layout["down_proj.weight"] == ((6144, 128), mx.uint32)
    assert layout["gate_proj.scales"] == ((2048, 96), mx.bfloat16)
    assert layout["up_proj.biases"] == ((2048, 96), mx.bfloat16)
    assert layout["down_proj.scales"] == ((6144, 32), mx.bfloat16)
    assert layout["down_proj.biases"] == ((6144, 32), mx.bfloat16)


@pytest.mark.parametrize(
    ("hidden", "intermediate", "group_size", "bits"),
    (
        (4096, 1536, 100, 2),  # hidden not divisible by group size
        (4096, 1500, 64, 2),  # intermediate not divisible by group size
        (4096, 1536, 0, 2),  # non-positive group size
        (4096, 1536, 64, 0),  # non-positive bits
        (24, 1536, 64, 2),  # hidden*bits does not fill whole uint32 lanes
    ),
)
def test_component_layout_fails_closed_on_bad_geometry(
    hidden: int, intermediate: int, group_size: int, bits: int
) -> None:
    with pytest.raises(Hy3M4ExpertWaveIneligible):
        component_layout(hidden, intermediate, group_size, bits)


# --------------------------------------------------------------------------- #
# End-to-end parity fixtures                                                   #
# --------------------------------------------------------------------------- #

_HY3_HIDDEN = 4096
_HY3_INTERMEDIATE = 1536
_ROWS = 4
_TOP_K = 8
_CAPACITY = 3


def _quantized_bank(
    hidden: int, intermediate: int, group_size: int, bits: int, capacity: int
) -> dict[str, mx.array]:
    """Quantize random experts into the resident-bank format the wave consumes.

    gate/up read the token (in=hidden, out=intermediate); down reads the
    activation (in=intermediate, out=hidden). ``mx.quantize`` groups along the
    trailing input axis, exactly matching ``component_layout``.
    """

    bank: dict[str, mx.array] = {}
    projection_dims = {
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    for projection, (out_dim, in_dim) in projection_dims.items():
        dense = mx.random.normal((capacity, out_dim, in_dim)) * 0.1
        weight, scales, biases = mx.quantize(dense, group_size=group_size, bits=bits)
        bank[f"{projection}.weight"] = weight
        bank[f"{projection}.scales"] = scales.astype(mx.bfloat16)
        bank[f"{projection}.biases"] = biases.astype(mx.bfloat16)
    mx.eval(list(bank.values()))
    return bank


def _reference_wave(
    hidden_rows: mx.array,
    slot_list: list[list[int]],
    route_weights: mx.array,
    bank: dict[str, mx.array],
    hidden: int,
    group_size: int,
    bits: int,
) -> mx.array:
    """Straightforward per-assignment reference: single-expert quantized_matmul
    projections, exact SwiGLU, then the identical BF16 route reduction."""

    def project(x: mx.array, projection: str, expert: int) -> mx.array:
        return mx.quantized_matmul(
            x,
            bank[f"{projection}.weight"][expert],
            bank[f"{projection}.scales"][expert],
            bank[f"{projection}.biases"][expert],
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )

    per_assignment = []
    for row in range(_ROWS):
        for slot in range(_TOP_K):
            expert = slot_list[row][slot]
            token = hidden_rows[0, row][None, :]  # (1, hidden)
            gate = project(token, "gate_proj", expert)
            up = project(token, "up_proj", expert)
            activated = swiglu(gate, up)  # (1, intermediate)
            down = project(activated, "down_proj", expert)  # (1, hidden)
            per_assignment.append(down[0])
    stacked = mx.stack(per_assignment).reshape((1, _ROWS, _TOP_K, hidden))
    return (stacked * route_weights[..., None]).sum(axis=-2)


@pytest.mark.parametrize("group_size", (64, 128))
def test_wave_matches_reference_end_to_end(group_size: int) -> None:
    mx.random.seed(1234 + group_size)
    bits = 2
    bank = _quantized_bank(
        _HY3_HIDDEN, _HY3_INTERMEDIATE, group_size, bits, _CAPACITY
    )

    hidden_rows = (mx.random.normal((1, _ROWS, _HY3_HIDDEN)) * 0.5).astype(mx.bfloat16)
    slot_list = [
        [(row * _TOP_K + slot) % _CAPACITY for slot in range(_TOP_K)]
        for row in range(_ROWS)
    ]
    slot_indices = mx.array(slot_list, dtype=mx.int32).reshape((1, _ROWS, _TOP_K))
    route_weights = (mx.random.uniform(shape=(1, _ROWS, _TOP_K))).astype(mx.bfloat16)
    flat = [expert for row in slot_list for expert in row]
    bounds = (min(flat), max(flat))

    result = hy3_q2_m4_expert_wave(
        hidden_rows,
        slot_indices,
        route_weights,
        bank,
        validated_slot_bounds=bounds,
        combine_mode="bf16",
        hidden_size=_HY3_HIDDEN,
        intermediate_size=_HY3_INTERMEDIATE,
        group_size=group_size,
        bits=bits,
    )
    reference = _reference_wave(
        hidden_rows, slot_list, route_weights, bank, _HY3_HIDDEN, group_size, bits
    )

    assert tuple(result.hidden_rows.shape) == (1, _ROWS, _HY3_HIDDEN)
    assert result.hidden_rows.dtype == mx.bfloat16
    assert result.hidden_size == _HY3_HIDDEN
    got = result.hidden_rows.astype(mx.float32)
    want = reference.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(got - want)))
    assert bool(mx.allclose(got, want, rtol=1e-2, atol=1e-2)), (
        f"wave/reference diverged at gs{group_size}: max_abs={max_abs}"
    )


def _zero_bank(
    hidden: int, intermediate: int, group_size: int, bits: int, capacity: int
) -> dict[str, mx.array]:
    """A shape/dtype-correct bank for the given geometry, contents irrelevant."""

    layout = component_layout(hidden, intermediate, group_size, bits)
    return {
        name: mx.zeros((capacity, *tail), dtype=dtype)
        for name, (tail, dtype) in layout.items()
    }


@pytest.mark.parametrize(
    ("bank_group_size", "spec_group_size"),
    ((64, 128), (128, 64)),
)
def test_wave_rejects_bank_shaped_for_other_group_size(
    bank_group_size: int, spec_group_size: int
) -> None:
    bits = 2
    bank = _zero_bank(
        _HY3_HIDDEN, _HY3_INTERMEDIATE, bank_group_size, bits, _CAPACITY
    )
    hidden_rows = mx.zeros((1, _ROWS, _HY3_HIDDEN), dtype=mx.bfloat16)
    slot_indices = mx.zeros((1, _ROWS, _TOP_K), dtype=mx.int32)
    route_weights = mx.zeros((1, _ROWS, _TOP_K), dtype=mx.bfloat16)

    with pytest.raises(Hy3M4ExpertWaveIneligible):
        hy3_q2_m4_expert_wave(
            hidden_rows,
            slot_indices,
            route_weights,
            bank,
            validated_slot_bounds=(0, 0),
            combine_mode="bf16",
            hidden_size=_HY3_HIDDEN,
            intermediate_size=_HY3_INTERMEDIATE,
            group_size=spec_group_size,
            bits=bits,
        )


def test_wave_accepts_bank_matching_declared_group_size() -> None:
    """Control for the rejection test: a gs128 bank under a gs128 spec runs."""

    bits = 2
    group_size = 128
    bank = _zero_bank(
        _HY3_HIDDEN, _HY3_INTERMEDIATE, group_size, bits, _CAPACITY
    )
    hidden_rows = mx.zeros((1, _ROWS, _HY3_HIDDEN), dtype=mx.bfloat16)
    slot_indices = mx.zeros((1, _ROWS, _TOP_K), dtype=mx.int32)
    route_weights = mx.zeros((1, _ROWS, _TOP_K), dtype=mx.bfloat16)

    output = hy3_q2_m4_expert_wave(
        hidden_rows,
        slot_indices,
        route_weights,
        bank,
        validated_slot_bounds=(0, 0),
        combine_mode="bf16",
        hidden_size=_HY3_HIDDEN,
        intermediate_size=_HY3_INTERMEDIATE,
        group_size=group_size,
        bits=bits,
    )
    assert tuple(output.hidden_rows.shape) == (1, _ROWS, _HY3_HIDDEN)
    assert output.stage_capture.assignments_covered_once
