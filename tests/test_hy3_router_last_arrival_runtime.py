from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.hy3_router_last_arrival as last_arrival
import mtplx.models.hy3_mlx as hy3_mlx
from mtplx.attention_context import attention_phase
from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_weight,
)


_SELECTOR = "mpp-r1-last-arrival-fused-r2"


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    hardware_parity = os.environ.get("MTPLX_RUN_ISSUE58_HARDWARE_PARITY") == "1"
    if not hardware_parity:
        mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _router_args() -> hy3_mlx.ModelArgs:
    return hy3_mlx.ModelArgs(
        model_type="hy_v3",
        hidden_size=4096,
        num_hidden_layers=1,
        intermediate_size=8192,
        moe_intermediate_size=1536,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts=192,
        num_experts_per_tok=8,
        num_shared_experts=1,
        first_k_dense_replace=0,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=128,
        route_norm=True,
        router_scaling_factor=2.826,
    )


def _router() -> hy3_mlx.Router:
    router = hy3_mlx.Router(_router_args())
    router.gate.weight = mx.zeros((192, 4096), dtype=mx.bfloat16)
    return router


def _runtime_inputs(rows: int = 4):
    return (
        mx.zeros((1, rows, 4096), dtype=mx.float32),
        mx.zeros((4096, 192), dtype=mx.bfloat16),
        mx.zeros((192,), dtype=mx.float32),
    )


def test_last_arrival_source_is_the_complete_authoritative_m4_boundary() -> None:
    source = last_arrival.hy3_router_last_arrival_source(
        scaling_factor=2.826,
        sigmoid_mode="precise",
    )

    assert "constexpr int ROWS = 4;" in source
    assert "constexpr int PADDED_ROWS = 8;" in source
    assert "constexpr int P = 16;" in source
    assert "constexpr int SGPTG = 4;" in source
    assert "constexpr int THREADGROUPS = 48;" in source
    assert "threadgroup float A_tile[PADDED_ROWS * KS];" in source
    assert "row < ROWS" in source
    assert "partials[15 * STRIDE + index]" in source
    assert "uint row = simd_gid;" in source
    assert "atomic_store_explicit(&ready[tg], tag" in source
    assert "atomic_store_explicit(&checks[tg], ~tag" in source
    assert "atomic_compare_exchange_weak_explicit(" in source
    assert "memory_order_seq_cst" in source
    assert "thread_scope_device" in source
    assert "atomic_fetch_add" not in source
    assert "while (atomic_load" not in source


def test_last_arrival_runtime_calls_one_kernel_and_returns_rows4_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, weight, expert_bias = _runtime_inputs()
    captured: dict[str, object] = {}

    class FakeKernel:
        def __call__(self, **kwargs: object):
            assert not captured
            captured.update(kwargs)
            return (
                mx.zeros((4, 8), dtype=mx.int32),
                mx.ones((4, 8), dtype=mx.float32),
                mx.zeros((24_672,), dtype=mx.float32),
            )

    monkeypatch.setattr(
        last_arrival,
        "_build_hy3_router_last_arrival_kernel",
        lambda *_args, **_kwargs: FakeKernel(),
    )
    monkeypatch.setattr(last_arrival, "_next_router_epoch", lambda: 73)

    output = last_arrival.hy3_router_last_arrival_route(
        value,
        weight,
        expert_bias,
        available=True,
        sigmoid_mode="precise",
    )

    assert tuple(output.expert_ids.shape) == (1, 4, 8)
    assert output.expert_ids.dtype == mx.int32
    assert tuple(output.route_weights.shape) == (1, 4, 8)
    assert output.route_weights.dtype == mx.float32
    assert output.dispatch_count == 1
    assert output.batch_shape == (1,)
    assert output.rows == 4
    assert output.top_k == 8
    assert output.assignment_count == 32
    assert captured["grid"] == (48 * 128, 1, 1)
    assert captured["threadgroup"] == (128, 1, 1)
    assert captured["output_shapes"] == [(4, 8), (4, 8), (24_672,)]
    assert captured["output_dtypes"] == [mx.int32, mx.float32, mx.float32]
    assert "init_value" not in captured
    inputs = captured["inputs"]
    assert isinstance(inputs, list)
    assert inputs[1] is weight
    assert inputs[2] is expert_bias


def test_resident_rows4_callable_structurally_satisfies_fixed_k3_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, weight, expert_bias = _runtime_inputs()
    expected = last_arrival.Hy3RouterLastArrivalOutput(
        expert_ids=mx.zeros((1, 4, 8), dtype=mx.int32),
        route_weights=mx.ones((1, 4, 8), dtype=mx.float32),
    )
    calls: list[tuple[object, object, object, dict[str, object]]] = []

    def fake_route(hidden_rows, resident_weight, resident_bias, **kwargs):
        calls.append((hidden_rows, resident_weight, resident_bias, kwargs))
        return expected

    monkeypatch.setattr(last_arrival, "hy3_router_last_arrival_route", fake_route)
    router = last_arrival.Hy3RouterLastArrival(
        weight=weight,
        expert_bias=expert_bias,
        available=True,
    )

    observed = router(value)

    assert observed is expected
    assert calls == [
        (
            value,
            weight,
            expert_bias,
            {
                "available": True,
                "top_k": 8,
                "route_norm": True,
                "scaling_factor": 2.826,
                "sigmoid_mode": "precise",
            },
        )
    ]


def test_last_arrival_runtime_fails_before_dispatch_outside_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("an ineligible route must not build or dispatch a kernel")

    monkeypatch.setattr(
        last_arrival,
        "_build_hy3_router_last_arrival_kernel",
        forbidden_kernel,
    )
    value, weight, expert_bias = _runtime_inputs()

    with pytest.raises(Hy3RouterFP32Ineligible, match=r"\[1, 4, 4096\]"):
        last_arrival.hy3_router_last_arrival_route(
            mx.zeros((1, 3, 4096), dtype=mx.float32),
            weight,
            expert_bias,
            available=True,
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="K-major BF16"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            mx.zeros((192, 4096), dtype=mx.bfloat16),
            expert_bias,
            available=True,
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="precise"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            weight,
            expert_bias,
            available=True,
            sigmoid_mode="fast",
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="top-8"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            weight,
            expert_bias,
            available=True,
            top_k=4,
        )


def test_last_arrival_runtime_epochs_do_not_repeat_with_reused_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(last_arrival, "_ROUTER_EPOCH", 580_000)

    assert [last_arrival._next_router_epoch() for _ in range(3)] == [
        580_001,
        580_002,
        580_003,
    ]


@pytest.mark.skipif(
    os.environ.get("MTPLX_RUN_ISSUE58_HARDWARE_PARITY") != "1",
    reason="Issue #58 parity requires an explicitly locked Metal hardware gate",
)
def test_last_arrival_runtime_matches_issue59_authoritative_m4_routes() -> None:
    mx.random.seed(58_590_004)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    resident_weight = prepare_hy3_router_fp32_weight(source_weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    value = mx.random.normal((1, 4, 4096)).astype(mx.float32)

    expected_ids, expected_weights = hy3_router_fp32_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
        n_tile=16,
        grid_k_parts=16,
        operand_mode="grouped-direct",
        simd_groups_per_threadgroup=4,
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode="simd",
        simd_groups=1,
        sigmoid_mode="precise",
    )
    first = last_arrival.hy3_router_last_arrival_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
    )
    second = last_arrival.hy3_router_last_arrival_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
    )
    mx.eval(
        expected_ids,
        expected_weights,
        first.expert_ids,
        first.route_weights,
        second.expert_ids,
        second.route_weights,
    )

    assert bool(mx.array_equal(first.expert_ids, expected_ids).item())
    assert bool(mx.array_equal(first.route_weights, expected_weights).item())
    assert bool(mx.array_equal(second.expert_ids, first.expert_ids).item())
    assert bool(mx.array_equal(second.route_weights, first.route_weights).item())


def test_issue58_selector_reuses_one_issue59_weight_and_dispatches_only_m4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router()
    source_weight = router.gate.weight
    calls: list[tuple[object, object, object, dict[str, object]]] = []

    def fake_route(value, weight, expert_bias, **kwargs):
        calls.append((value, weight, expert_bias, kwargs))
        return last_arrival.Hy3RouterLastArrivalOutput(
            expert_ids=mx.zeros((1, 4, 8), dtype=mx.int32),
            route_weights=mx.ones((1, 4, 8), dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "hy3_router_last_arrival_route", fake_route)

    report = router.configure_kernel(_SELECTOR, available=True)
    with attention_phase("decode_verify"):
        indices, weights = router(mx.zeros((1, 4, 4096), dtype=mx.bfloat16))

    assert report["selector"] == _SELECTOR
    assert report["dispatch_count"] == 1
    assert report["supported_rows"] == 4
    assert report["sigmoid_mode"] == "precise"
    assert report["topology"] == "n16-p16-sg4-in-kernel-pad"
    assert report["threadgroups"] == 48
    assert report["attention_phase"] == "decode_verify"
    assert report["prepared_weight_bytes"] == 192 * 4096 * 2
    assert report["incremental_bytes"] == 192 * 4096 * 2
    assert router.gate.weight is source_weight
    assert tuple(indices.shape) == (1, 4, 8)
    assert tuple(weights.shape) == (1, 4, 8)
    assert len(calls) == 1
    value, resident_weight, resident_bias, kwargs = calls[0]
    assert value.dtype == mx.float32
    assert resident_weight is router._mtplx_router_kernel_state.prepared_weight
    assert tuple(resident_weight.shape) == (4096, 192)
    assert resident_weight.dtype == mx.bfloat16
    assert resident_bias is router.expert_bias
    assert kwargs == {
        "top_k": 8,
        "route_norm": True,
        "scaling_factor": 2.826,
        "sigmoid_mode": "precise",
    }


def test_issue58_selector_falls_back_to_stock_without_hybrid_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingGate(nn.Linear):
        def __init__(self) -> None:
            super().__init__(4096, 192, bias=False)
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return mx.zeros((*value.shape[:-1], 192), dtype=mx.float32)

    router = _router()
    gate = RecordingGate()
    gate.weight = router.gate.weight
    router.gate = gate
    router.configure_kernel(_SELECTOR, available=True)

    def forbidden_route(*_args, **_kwargs):
        raise AssertionError("unsupported Rows3 must fail closed to stock")

    monkeypatch.setattr(hy3_mlx, "hy3_router_last_arrival_route", forbidden_route)
    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_fp32_route",
        lambda *_args, **_kwargs: pytest.fail("#59 checker must not double-run"),
    )

    indices, weights = router(mx.zeros((1, 3, 4096), dtype=mx.bfloat16))

    assert gate.calls == 1
    assert tuple(indices.shape) == (1, 3, 8)
    assert tuple(weights.shape) == (1, 3, 8)


def test_issue58_selector_falls_back_to_stock_outside_decode_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingGate(nn.Linear):
        def __init__(self) -> None:
            super().__init__(4096, 192, bias=False)
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return mx.zeros((*value.shape[:-1], 192), dtype=mx.float32)

    router = _router()
    gate = RecordingGate()
    gate.weight = router.gate.weight
    router.gate = gate
    router.configure_kernel(_SELECTOR, available=True)
    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_last_arrival_route",
        lambda *_args, **_kwargs: pytest.fail("prefill must stay on stock routing"),
    )

    with attention_phase("prefill"):
        indices, weights = router(mx.zeros((1, 4, 4096), dtype=mx.bfloat16))

    assert gate.calls == 1
    assert tuple(indices.shape) == (1, 4, 8)
    assert tuple(weights.shape) == (1, 4, 8)


def test_issue58_selector_does_not_admit_the_issue60_fast_mode() -> None:
    router = _router()

    with pytest.raises(ValueError, match="Hy3 router kernel"):
        router.configure_kernel("mpp-r1-last-arrival-fast-fused-r2", available=True)
