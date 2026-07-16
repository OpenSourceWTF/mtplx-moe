"""Row-owned Hy3 router: source contract, dispatch shape, and hardware parity."""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from mtplx import hy3_router_row_owned as row_owned
from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_weight,
)

_ROWS = range(1, 9)
_HARDWARE = os.environ.get("MTPLX_ROW_OWNED_HARDWARE") == "1"

_BANNED_TOKENS = (
    "atomic_",
    "atomic<",
    "thread_scope_device",
    "memory_order",
    "volatile",
    "epoch",
    "ready",
    "elected",
    "checksum",
    "scratch",
    "while (true)",
)


def test_source_is_row_owned_and_device_protocol_free() -> None:
    source = row_owned.hy3_router_row_owned_source()
    assert "uint row = threadgroup_position_in_grid.x;" in source
    assert f"constexpr int SIMD_GROUPS = {row_owned._SIMD_GROUPS};" in source
    assert "threadgroup float partials[P * N];" in source
    assert "threadgroup float a_tile[BM * KS];" in source
    assert "matmul2d_descriptor::mode::multiply" in source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in source
    for token in _BANNED_TOKENS:
        assert token not in source, f"banned device-protocol token: {token}"


def test_source_reproduces_incumbent_r2_semantics() -> None:
    source = row_owned.hy3_router_row_owned_source()
    assert "float score = 1.0f / (1.0f + exp(-total));" in source
    assert "score + expert_bias[expert]" in source
    assert "&& index > lane_index" in source
    assert "ROUTING_SCALE / (score_sum + 1e-20f)" in source
    assert "int source = TOPK - 1 - output;" in source
    assert "constexpr float ROUTING_SCALE = 2.826f;" in source


def test_source_uses_sixteen_part_balanced_reduction() -> None:
    source = row_owned.hy3_router_row_owned_source()
    assert "constexpr int P = 16;" in source
    assert "partials[15 * STRIDE + index]" in source
    assert "constexpr int STRIDE = N;" in source


class _FakeKernel:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def __call__(self, *, inputs, grid, threadgroup, output_shapes, output_dtypes):
        self._captured.update(
            inputs=inputs,
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )
        return tuple(
            mx.zeros(shape, dtype=dtype)
            for shape, dtype in zip(output_shapes, output_dtypes)
        )


@pytest.mark.parametrize("rows", _ROWS)
def test_dispatch_launches_one_threadgroup_per_row(rows, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        row_owned,
        "_build_hy3_router_row_owned_kernel",
        lambda *_args: _FakeKernel(captured),
    )
    value = mx.zeros((1, rows, 4096), dtype=mx.float32)
    weight = mx.zeros((4096, 192), dtype=mx.bfloat16)
    bias = mx.zeros((192,), dtype=mx.float32)
    ids, scores = row_owned.hy3_router_row_owned_route(
        value, weight, bias, available=True
    )
    assert captured["grid"] == (rows * 384, 1, 1)
    assert captured["threadgroup"] == (384, 1, 1)
    assert captured["output_shapes"] == [(rows, 8), (rows, 8)]
    assert captured["output_dtypes"] == [mx.int32, mx.float32]
    assert len(captured["inputs"]) == 3
    assert tuple(ids.shape) == (1, rows, 8)
    assert tuple(scores.shape) == (1, rows, 8)


@pytest.mark.parametrize(
    "value_shape, weight_shape, bias_shape, message",
    [
        ((1, 9, 4096), (4096, 192), (192,), "row-owned M1..M8"),
        ((1, 4, 4095), (4096, 192), (192,), "widths do not match"),
        ((1, 4, 4096), (4096, 191), (192,), "row-owned M1..M8"),
        ((1, 4, 4096), (4096, 192), (191,), "expert bias"),
    ],
)
def test_dispatch_rejects_out_of_contract_shapes(
    value_shape, weight_shape, bias_shape, message
) -> None:
    value = mx.zeros(value_shape, dtype=mx.float32)
    weight = mx.zeros(weight_shape, dtype=mx.bfloat16)
    bias = mx.zeros(bias_shape, dtype=mx.float32)
    with pytest.raises(Hy3RouterFP32Ineligible, match=message):
        row_owned.hy3_router_row_owned_route(value, weight, bias, available=True)


def test_dispatch_rejects_unavailable_device() -> None:
    value = mx.zeros((1, 4, 4096), dtype=mx.float32)
    weight = mx.zeros((4096, 192), dtype=mx.bfloat16)
    bias = mx.zeros((192,), dtype=mx.float32)
    with pytest.raises(Hy3RouterFP32Ineligible):
        row_owned.hy3_router_row_owned_route(value, weight, bias, available=False)


def _reference_route(value: mx.array, weight: mx.array, bias: mx.array):
    return hy3_router_fp32_route(
        value,
        weight,
        bias,
        n_tile=16,
        grid_k_parts=16,
        operand_mode="direct",
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode="simd",
        sigmoid_mode="precise",
    )


def _assert_bitwise_parity(value, weight, bias) -> None:
    expected_ids, expected_scores = _reference_route(value, weight, bias)
    ids, scores = row_owned.hy3_router_row_owned_route(value, weight, bias)
    assert mx.array_equal(ids, expected_ids).item()
    assert mx.array_equal(scores, expected_scores).item()


@pytest.mark.skipif(
    not _HARDWARE,
    reason="row-owned parity requires MTPLX_ROW_OWNED_HARDWARE=1",
)
@pytest.mark.parametrize("rows", _ROWS)
def test_hardware_parity_random(rows) -> None:
    mx.random.seed(51 + rows)
    value = mx.random.normal((1, rows, 4096)).astype(mx.float32)
    weight = prepare_hy3_router_fp32_weight(
        mx.random.normal((192, 4096)).astype(mx.bfloat16)
    )
    bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    _assert_bitwise_parity(value, weight, bias)


@pytest.mark.skipif(
    not _HARDWARE,
    reason="row-owned parity requires MTPLX_ROW_OWNED_HARDWARE=1",
)
@pytest.mark.parametrize("rows", (1, 4, 8))
def test_hardware_parity_adversarial_ties(rows) -> None:
    weight = prepare_hy3_router_fp32_weight(mx.zeros((192, 4096), dtype=mx.bfloat16))
    zero_value = mx.zeros((1, rows, 4096), dtype=mx.float32)

    # Full tie: identical logits and identical bias; later index must win.
    _assert_bitwise_parity(zero_value, weight, mx.zeros((192,), dtype=mx.float32))

    # Near-monotonic bias with a duplicated selected value at the boundary.
    ramp = (mx.arange(192).astype(mx.float32)) * 1e-3
    duplicated = mx.concatenate([ramp[:-1], ramp[-2:-1]])
    _assert_bitwise_parity(zero_value, weight, duplicated)

    # Selected tie inside the top-8 band only.
    band = mx.concatenate(
        [
            mx.zeros((184,), dtype=mx.float32),
            mx.ones((8,), dtype=mx.float32) * 0.25,
        ]
    )
    _assert_bitwise_parity(zero_value, weight, band)


@pytest.mark.skipif(
    not _HARDWARE,
    reason="row-owned parity requires MTPLX_ROW_OWNED_HARDWARE=1",
)
def test_hardware_repeated_execution_is_deterministic() -> None:
    mx.random.seed(58)
    value = mx.random.normal((1, 4, 4096)).astype(mx.float32)
    weight = prepare_hy3_router_fp32_weight(
        mx.random.normal((192, 4096)).astype(mx.bfloat16)
    )
    bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    first_ids, first_scores = row_owned.hy3_router_row_owned_route(value, weight, bias)
    for _ in range(16):
        ids, scores = row_owned.hy3_router_row_owned_route(value, weight, bias)
        assert mx.array_equal(ids, first_ids).item()
        assert mx.array_equal(scores, first_scores).item()
