from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from mtplx.benchmarks.hy3_shared_gate_up_probe import (
    pack_quantized_gate_up,
    run_packed_gate_up,
)


def _quantized_linear() -> nn.QuantizedLinear:
    linear = nn.Linear(64, 32, bias=False)
    linear.weight = mx.random.normal((32, 64), dtype=mx.float32).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(
        linear,
        group_size=64,
        bits=4,
        mode="affine",
    )


def test_packed_shared_gate_up_matches_two_qmms_through_swiglu() -> None:
    mx.random.seed(312)
    gate_proj = _quantized_linear()
    up_proj = _quantized_linear()
    x = mx.random.normal((1, 1, 64), dtype=mx.float32).astype(mx.bfloat16)

    expected = swiglu(gate_proj(x), up_proj(x))
    packed = pack_quantized_gate_up(gate_proj, up_proj)
    gate, up = run_packed_gate_up(x, packed)
    actual = swiglu(gate, up)
    mx.eval(expected, actual)

    assert packed.split_at == 32
    assert packed.packed_bytes == sum(
        int(array.nbytes)
        for module in (gate_proj, up_proj)
        for array in (module.weight, module.scales, module.biases)
    )
    assert mx.array_equal(expected, actual).item()


def test_packed_shared_gate_up_rejects_mismatched_output_widths() -> None:
    gate_proj = _quantized_linear()
    up_linear = nn.Linear(64, 16, bias=False)
    up_proj = nn.QuantizedLinear.from_linear(
        up_linear,
        group_size=64,
        bits=4,
        mode="affine",
    )

    try:
        pack_quantized_gate_up(gate_proj, up_proj)
    except ValueError as error:
        assert "matching widths" in str(error)
    else:
        raise AssertionError("mismatched gate/up widths were accepted")
