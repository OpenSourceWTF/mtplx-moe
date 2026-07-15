from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.models.hy3_mlx as hy3_mlx
from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    hy3_router_fp32_available,
)


def _exact_router_args() -> hy3_mlx.ModelArgs:
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


def _exact_router() -> hy3_mlx.Router:
    router = hy3_mlx.Router(_exact_router_args())
    router.gate.weight = mx.zeros((192, 4096), dtype=mx.bfloat16)
    return router


def test_router_steel_r1_selector_prepares_once_and_dispatches_exact_r2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _exact_router()
    calls = []

    def fake_exact_route(value, weight, expert_bias, **kwargs):
        calls.append((value, weight, expert_bias, kwargs))
        output_shape = (*value.shape[:-1], 8)
        return (
            mx.zeros(output_shape, dtype=mx.int32),
            mx.ones(output_shape, dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "hy3_router_fp32_exact_route", fake_exact_route)

    report = router.configure_kernel("steel-r1-fused-r2", available=True)
    indices, weights = router(mx.zeros((1, 2, 4096), dtype=mx.bfloat16))

    assert report == {
        "selector": "steel-r1-fused-r2",
        "enabled": True,
        "source_weight_bytes": 192 * 4096 * 2,
        "prepared_weight_bytes": 192 * 4096 * 4,
        "incremental_bytes": 192 * 4096 * 2,
    }
    assert router.gate.weight.dtype == mx.float32
    assert tuple(indices.shape) == (1, 2, 8)
    assert tuple(weights.shape) == (1, 2, 8)
    assert len(calls) == 1
    value, prepared_weight, expert_bias, kwargs = calls[0]
    assert value.dtype == mx.float32
    assert prepared_weight is router.gate.weight
    assert expert_bias is router.expert_bias
    assert kwargs == {
        "top_k": 8,
        "route_norm": True,
        "scaling_factor": 2.826,
        "finalizer_mode": "simd",
    }


def test_router_mpp_r1_selector_prepares_once_and_dispatches_precise_r2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _exact_router()
    source_weight = router.gate.weight
    calls = []

    def fake_mpp_route(value, weight, expert_bias, **kwargs):
        calls.append((value, weight, expert_bias, kwargs))
        output_shape = (*value.shape[:-1], 8)
        return (
            mx.zeros(output_shape, dtype=mx.int32),
            mx.ones(output_shape, dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "hy3_router_fp32_route", fake_mpp_route)

    report = router.configure_kernel("mpp-r1-fused-r2", available=True)
    indices, weights = router(mx.zeros((1, 3, 4096), dtype=mx.bfloat16))

    assert report == {
        "selector": "mpp-r1-fused-r2",
        "enabled": True,
        "source_weight_bytes": 192 * 4096 * 2,
        "prepared_weight_bytes": 192 * 4096 * 2,
        "incremental_bytes": 192 * 4096 * 2,
    }
    assert router.gate.weight is source_weight
    assert router.gate.weight.dtype == mx.bfloat16
    assert tuple(indices.shape) == (1, 3, 8)
    assert tuple(weights.shape) == (1, 3, 8)
    assert len(calls) == 1
    value, prepared_weight, expert_bias, kwargs = calls[0]
    assert value.dtype == mx.float32
    assert prepared_weight is not router.gate.weight
    assert tuple(prepared_weight.shape) == (4096, 192)
    assert prepared_weight.dtype == mx.bfloat16
    assert expert_bias is router.expert_bias
    assert kwargs == {
        "n_tile": 16,
        "grid_k_parts": 8,
        "operand_mode": "direct",
        "top_k": 8,
        "route_norm": True,
        "scaling_factor": 2.826,
        "finalizer_mode": "simd",
        "sigmoid_mode": "precise",
    }


def test_issue59_rejects_issue60_fast_r2_selector() -> None:
    router = _exact_router()

    with pytest.raises(ValueError, match="Hy3 router kernel"):
        router.configure_kernel("mpp-r1-fast-fused-r2", available=True)


def test_router_mpp_fp32_splitk_selector_dispatches_m4_precise_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _exact_router()
    source_weight = router.gate.weight
    calls = []

    def fake_splitk_route(value, weight, expert_bias, **kwargs):
        calls.append((value, weight, expert_bias, kwargs))
        output_shape = (*value.shape[:-1], 8)
        return (
            mx.zeros(output_shape, dtype=mx.int32),
            mx.ones(output_shape, dtype=mx.float32),
        )

    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_fp32_exact_splitk_route",
        fake_splitk_route,
    )

    report = router.configure_kernel(
        "mpp-fp32-splitk-r1-fused-r2",
        available=True,
    )
    indices, weights = router(mx.zeros((1, 4, 4096), dtype=mx.bfloat16))

    assert report == {
        "selector": "mpp-fp32-splitk-r1-fused-r2",
        "enabled": True,
        "source_weight_bytes": 192 * 4096 * 2,
        "prepared_weight_bytes": 192 * 4096 * 4,
        "incremental_bytes": 192 * 4096 * 4,
        "m1_policy": "stock",
        "m4_grid_k_parts": 32,
        "other_grid_k_parts": 16,
    }
    assert router.gate.weight is source_weight
    assert router.gate.weight.dtype == mx.bfloat16
    assert tuple(indices.shape) == (1, 4, 8)
    assert tuple(weights.shape) == (1, 4, 8)
    assert len(calls) == 1
    value, prepared_weight, expert_bias, kwargs = calls[0]
    assert value.dtype == mx.float32
    assert tuple(prepared_weight.shape) == (4096, 192)
    assert prepared_weight.dtype == mx.float32
    assert expert_bias is router.expert_bias
    assert kwargs == {
        "n_tile": 32,
        "grid_k_parts": 32,
        "operand_mode": "direct",
        "top_k": 8,
        "route_norm": True,
        "scaling_factor": 2.826,
        "finalizer_mode": "simd",
        "sigmoid_mode": "precise",
    }


def test_router_mpp_fp32_splitk_selector_keeps_stock_m1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingGate(nn.Linear):
        def __init__(self) -> None:
            super().__init__(4096, 192, bias=False)
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return mx.zeros((*value.shape[:-1], 192), dtype=mx.float32)

    router = _exact_router()
    gate = RecordingGate()
    gate.weight = router.gate.weight
    router.gate = gate
    router.configure_kernel("mpp-fp32-splitk-r1-fused-r2", available=True)

    def forbidden_splitk_path(*_args, **_kwargs):
        raise AssertionError("M1 must stay on the stock router")

    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_fp32_exact_splitk_route",
        forbidden_splitk_path,
    )

    indices, weights = router(mx.zeros((1, 1, 4096), dtype=mx.bfloat16))

    assert router.gate is gate
    assert gate.calls == 1
    assert tuple(indices.shape) == (1, 1, 8)
    assert tuple(weights.shape) == (1, 1, 8)


def test_configure_router_kernels_uses_splitk_m1_only_for_mtp_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MTPContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = _exact_router()

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target_router = _exact_router()
            self.mtp = MTPContainer()

    root = Root()
    calls = []

    def fake_splitk_route(value, weight, expert_bias, **kwargs):
        calls.append((value, weight, expert_bias, kwargs))
        output_shape = (*value.shape[:-1], 8)
        return (
            mx.zeros(output_shape, dtype=mx.int32),
            mx.ones(output_shape, dtype=mx.float32),
        )

    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_fp32_exact_splitk_route",
        fake_splitk_route,
    )

    report = hy3_mlx.configure_hy3_router_kernels(
        root,
        "mpp-fp32-splitk-r1-fused-r2",
        available=True,
    )

    assert report["router_count"] == 2
    assert report["m1_splitk_count"] == 1
    assert root.target_router._mtplx_router_kernel_state.splitk_m1 is False
    assert root.mtp.router._mtplx_router_kernel_state.splitk_m1 is True

    root.mtp.router(mx.zeros((1, 1, 4096), dtype=mx.bfloat16))
    assert len(calls) == 1
    assert tuple(calls[0][0].shape) == (1, 1, 4096)


def test_authoritative_mpp_selector_is_shared_by_target_and_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MTPContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = _exact_router()

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target_router = _exact_router()
            self.mtp = MTPContainer()

    root = Root()
    calls = []

    def fake_mpp_route(value, weight, expert_bias, **kwargs):
        calls.append(kwargs)
        output_shape = (*value.shape[:-1], 8)
        return (
            mx.zeros(output_shape, dtype=mx.int32),
            mx.ones(output_shape, dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "hy3_router_fp32_route", fake_mpp_route)

    report = hy3_mlx.configure_hy3_router_kernels(
        root,
        "mpp-r1-fused-r2",
        available=True,
    )
    root.target_router(mx.zeros((1, 4, 4096), dtype=mx.bfloat16))
    root.mtp.router(mx.zeros((1, 1, 4096), dtype=mx.bfloat16))

    assert report["selector"] == "mpp-r1-fused-r2"
    assert root.target_router._mtplx_router_kernel_state.selector == ("mpp-r1-fused-r2")
    assert root.mtp.router._mtplx_router_kernel_state.selector == ("mpp-r1-fused-r2")
    assert [call["sigmoid_mode"] for call in calls] == ["precise", "precise"]


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_router_mpp_fp32_splitk_selector_matches_stock_routes_on_g17(
    rows: int,
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(521_000 + rows)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    value = mx.random.normal((1, rows, 4096)).astype(mx.bfloat16)

    stock = _exact_router()
    stock.gate.weight = source_weight
    stock.expert_bias = expert_bias
    expected_ids, expected_weights = stock(value)

    candidate = _exact_router()
    candidate.gate.weight = source_weight
    candidate.expert_bias = expert_bias
    candidate.configure_kernel(
        "mpp-fp32-splitk-r1-fused-r2",
        available=True,
    )
    observed_ids, observed_weights = candidate(value)
    mx.eval(
        expected_ids,
        expected_weights,
        observed_ids,
        observed_weights,
    )

    assert bool(mx.array_equal(observed_ids, expected_ids).item())
    assert float(mx.max(mx.abs(observed_weights - expected_weights)).item()) <= 5e-4


@pytest.mark.parametrize(
    ("selector", "route_name"),
    (
        ("steel-r1-fused-r2", "hy3_router_fp32_exact_route"),
        ("mpp-r1-fused-r2", "hy3_router_fp32_route"),
        (
            "mpp-fp32-splitk-r1-fused-r2",
            "hy3_router_fp32_exact_splitk_route",
        ),
    ),
)
def test_router_optimized_selectors_dispatch_native_m8(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    route_name: str,
) -> None:
    router = _exact_router()
    calls = []

    def fake_route(value, weight, expert_bias, **kwargs):
        calls.append((value, weight, expert_bias, kwargs))
        output_shape = (*value.shape[:-1], 8)
        return (
            mx.zeros(output_shape, dtype=mx.int32),
            mx.ones(output_shape, dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, route_name, fake_route)
    router.configure_kernel(selector, available=True)

    indices, weights = router(mx.zeros((1, 8, 4096), dtype=mx.bfloat16))

    assert tuple(indices.shape) == (1, 8, 8)
    assert tuple(weights.shape) == (1, 8, 8)
    assert len(calls) == 1
    assert tuple(calls[0][0].shape) == (1, 8, 4096)
    if selector == "mpp-fp32-splitk-r1-fused-r2":
        assert calls[0][3]["n_tile"] == 32
        assert calls[0][3]["grid_k_parts"] == 16


@pytest.mark.parametrize(
    "selector",
    (
        "steel-r1-fused-r2",
        "mpp-r1-fused-r2",
        "mpp-fp32-splitk-r1-fused-r2",
    ),
)
def test_router_optimized_selectors_reject_wrappers_and_nonexact_contracts(
    selector: str,
) -> None:
    class Wrapper(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def __call__(self, value):
            return self.base(value)

    wrapped = _exact_router()
    wrapped.gate = Wrapper(wrapped.gate)
    with pytest.raises(Hy3RouterFP32Ineligible, match="unwrapped"):
        wrapped.configure_kernel(selector, available=True)

    unnormalized = _exact_router()
    unnormalized.route_norm = False
    with pytest.raises(Hy3RouterFP32Ineligible, match="exact top-8"):
        unnormalized.configure_kernel(selector, available=True)


def test_router_optimized_mode_uses_stock_path_outside_m1_to_m8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingGate(nn.Module):
        def __init__(self, base: nn.Linear) -> None:
            super().__init__()
            self.base = base
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return mx.zeros((*value.shape[:-1], 192), dtype=mx.float32)

    router = _exact_router()
    router.configure_kernel("mpp-r1-fused-r2", available=True)
    gate = RecordingGate(router.gate)
    router.gate = gate

    def forbidden_fast_path(*_args, **_kwargs):
        raise AssertionError("large-M input must not enter the MPP router")

    monkeypatch.setattr(hy3_mlx, "hy3_router_fp32_route", forbidden_fast_path)

    indices, weights = router(mx.zeros((1, 9, 4096), dtype=mx.bfloat16))

    assert gate.calls == 1
    assert tuple(indices.shape) == (1, 9, 8)
    assert tuple(weights.shape) == (1, 9, 8)


def test_configure_hy3_router_kernels_reports_incremental_memory() -> None:
    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = _exact_router()

    root = Root()

    report = hy3_mlx.configure_hy3_router_kernels(
        root,
        "steel-r1-fused-r2",
        available=True,
    )

    assert report == {
        "selector": "steel-r1-fused-r2",
        "router_count": 1,
        "enabled_count": 1,
        "source_weight_bytes": 192 * 4096 * 2,
        "prepared_weight_bytes": 192 * 4096 * 4,
        "incremental_bytes": 192 * 4096 * 2,
    }


@pytest.mark.parametrize(
    ("selector", "expected"),
    (
        ("stock", 0),
        ("steel-r1-fused-r2", 4 * 192 * 4096 * 2),
        ("mpp-r1-fused-r2", 4 * 192 * 4096 * 2),
        ("mpp-fp32-splitk-r1-fused-r2", 4 * 192 * 4096 * 4),
    ),
)
def test_router_kernel_memory_estimate_counts_sparse_trunk_and_mtp(
    selector: str,
    expected: int,
) -> None:
    config = {
        "model_type": "hy_v3",
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
    }

    assert (
        hy3_mlx.estimate_hy3_router_kernel_incremental_bytes(
            config,
            selector,
            include_mtp=True,
        )
        == expected
    )


def test_router_kernel_memory_estimate_honors_explicit_layer_types() -> None:
    config = {
        "model_type": "hy_v3",
        "num_hidden_layers": 4,
        "first_k_dense_replace": 0,
        "mlp_layer_types": ["dense", "sparse", "dense", "sparse"],
    }

    assert (
        hy3_mlx.estimate_hy3_router_kernel_incremental_bytes(
            config,
            "mpp-r1-fused-r2",
            include_mtp=False,
        )
        == 2 * 192 * 4096 * 2
    )
