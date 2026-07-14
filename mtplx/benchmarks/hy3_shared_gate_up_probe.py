"""Benchmark-only packed shared gate/up projection probe for issue #31."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class PackedQuantizedGateUp:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    split_at: int
    group_size: int
    bits: int
    mode: str
    packed_bytes: int


def pack_quantized_gate_up(
    gate_proj: nn.QuantizedLinear,
    up_proj: nn.QuantizedLinear,
) -> PackedQuantizedGateUp:
    """Materialize compatible gate/up Q4 arrays for one combined QMM."""

    compatible = (
        int(gate_proj.weight.shape[0]) == int(up_proj.weight.shape[0])
        and tuple(gate_proj.weight.shape[1:]) == tuple(up_proj.weight.shape[1:])
        and tuple(gate_proj.scales.shape[1:]) == tuple(up_proj.scales.shape[1:])
        and tuple(gate_proj.biases.shape[1:]) == tuple(up_proj.biases.shape[1:])
        and int(gate_proj.group_size) == int(up_proj.group_size)
        and int(gate_proj.bits) == int(up_proj.bits)
        and str(gate_proj.mode) == str(up_proj.mode)
        and "bias" not in gate_proj
        and "bias" not in up_proj
    )
    if not compatible:
        raise ValueError(
            "gate/up projections require matching widths and quantization without bias"
        )
    weight = mx.concatenate((gate_proj.weight, up_proj.weight), axis=0)
    scales = mx.concatenate((gate_proj.scales, up_proj.scales), axis=0)
    biases = mx.concatenate((gate_proj.biases, up_proj.biases), axis=0)
    mx.eval(weight, scales, biases)
    return PackedQuantizedGateUp(
        weight=weight,
        scales=scales,
        biases=biases,
        split_at=int(gate_proj.weight.shape[0]),
        group_size=int(gate_proj.group_size),
        bits=int(gate_proj.bits),
        mode=str(gate_proj.mode),
        packed_bytes=sum(int(array.nbytes) for array in (weight, scales, biases)),
    )


def run_packed_gate_up(
    inputs: mx.array,
    packed: PackedQuantizedGateUp,
) -> tuple[mx.array, mx.array]:
    """Run one packed QMM and split gate/up outputs."""

    output = mx.quantized_matmul(
        inputs,
        packed.weight,
        scales=packed.scales,
        biases=packed.biases,
        transpose=True,
        group_size=packed.group_size,
        bits=packed.bits,
        mode=packed.mode,
    )
    gate, up = mx.split(output, [packed.split_at], axis=-1)
    return gate, up
