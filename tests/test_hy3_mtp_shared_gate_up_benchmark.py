"""Regression tests for the Issue #69 benchmark evidence harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _benchmark_module():
    path = Path(__file__).parents[1] / "benchmarks" / "hy3_mtp_shared_gate_up.py"
    spec = importlib.util.spec_from_file_location("issue69_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_device_metadata_uses_the_current_mlx_api() -> None:
    calls = []
    core = SimpleNamespace(device_info=lambda: calls.append("called") or {"gpu": "M"})

    assert _benchmark_module()._device_info(core) == {"gpu": "M"}
    assert calls == ["called"]


def test_module_names_include_a_non_stock_control_once() -> None:
    benchmark = _benchmark_module()

    assert benchmark._module_names(("fast",), control_candidate="exact") == (
        "exact",
        "fast",
    )
    assert benchmark._module_names(
        ("exact", "fast"),
        control_candidate="exact",
    ) == ("exact", "fast")
    assert benchmark._module_names(("fast",), control_candidate="stock") == ("fast",)


def test_metal_candidate_map_includes_separate_stock_tn4_exact_arms() -> None:
    benchmark = _benchmark_module()

    candidates = benchmark._metal_candidate_map()
    stock = candidates["metal_n24_r2_v4_exact_stock_tn4"]

    assert stock.reduction_layout == "stock_tn4"
    assert stock.k_vector == 4
    assert "metal_n24_r2_v16_exact" in candidates
    assert "metal_n24_r2_v4_fast_stock_tn4" not in candidates
    assert "metal_n24_r2_v4_exact_stock_tn4_direct" in candidates
    assert "metal_n24_r2_v16_exact_direct" in candidates
    assert "metal_n24_r2_v16_exact_threadgroup_f32" in candidates
    assert "metal_n24_r2_v16_exact_packed2" in candidates
    assert "metal_n24_r2_v16_exact_direct_packed2" in candidates
    assert "metal_n24_r2_v16_exact_threadgroup_f32_packed2" in candidates
    assert "metal_n24_r2_v4_exact_stock_tn4_packed2" in candidates
    assert "metal_n24_r2_v16_exact_striped_tree" in candidates
    assert "metal_n24_r2_v4_exact_stock_tn4_sum" in candidates


def test_k3_refinement_frontier_factors_one_axis_at_a_time() -> None:
    benchmark = _benchmark_module()

    assert benchmark.K3_REFINEMENT_CANDIDATES == (
        "metal_n24_r2_v16_exact",
        "metal_n24_r2_v16_exact_direct",
        "metal_n24_r2_v16_exact_threadgroup_f32",
        "metal_n24_r2_v16_exact_striped_tree",
        "metal_n24_r2_v4_exact_stock_tn4",
        "metal_n24_r2_v4_exact_stock_tn4_sum",
        "metal_n24_r2_v16_exact_packed2",
        "metal_n24_r2_v16_exact_direct_packed2",
        "metal_n24_r2_v16_exact_threadgroup_f32_packed2",
        "metal_n24_r2_v4_exact_stock_tn4_packed2",
    )
    assert set(benchmark.K3_REFINEMENT_CANDIDATES) <= set(
        benchmark._metal_candidate_map()
    )


def test_packed2_candidate_interleaves_weights_once_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark_module()
    gate, up, down, packed = object(), object(), object(), object()
    observed = {}

    def fake_stack(values, *, axis):
        observed["stack"] = (values, axis)
        return packed

    class FakePackedModule:
        def __init__(self, packed_weight, down_weight, *, candidate):
            observed["module"] = (packed_weight, down_weight, candidate)

        def parameters(self):
            return ()

    monkeypatch.setattr(benchmark.mx, "stack", fake_stack)
    monkeypatch.setattr(benchmark.mx, "eval", lambda *_args: None)
    monkeypatch.setattr(benchmark, "MetalPackedFusedMTPSharedMLP", FakePackedModule)

    module = benchmark._candidate_shared(
        "metal_n24_r2_v16_exact_packed2",
        SimpleNamespace(),
        gate,
        up,
        down,
    )

    assert isinstance(module, FakePackedModule)
    assert observed["stack"] == ((gate, up), -1)
    packed_weight, down_weight, candidate = observed["module"]
    assert packed_weight is packed
    assert down_weight is down
    assert candidate.weight_layout == "packed2"


def test_candidate_extra_bytes_distinguishes_benchmark_only_packing() -> None:
    benchmark = _benchmark_module()

    assert benchmark._candidate_extra_bytes("block", gate_up_bytes=1234) == 1234
    assert (
        benchmark._candidate_extra_bytes(
            "metal_n24_r2_v16_exact_packed2",
            gate_up_bytes=1234,
        )
        == 1234
    )
    assert (
        benchmark._candidate_extra_bytes(
            "metal_n24_r2_v16_exact",
            gate_up_bytes=1234,
        )
        == 0
    )


def test_candidate_modules_share_one_packed_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark_module()
    gate, up, down, packed = object(), object(), object(), object()
    stack_calls = []
    module_calls = []

    def fake_stack(values, *, axis):
        stack_calls.append((values, axis))
        return packed

    def fake_candidate_shared(
        name,
        args,
        gate_weight,
        up_weight,
        down_weight,
        *,
        packed_weight=None,
    ):
        module_calls.append(
            (name, args, gate_weight, up_weight, down_weight, packed_weight)
        )
        return name

    monkeypatch.setattr(benchmark.mx, "stack", fake_stack)
    monkeypatch.setattr(benchmark, "_candidate_shared", fake_candidate_shared)
    names = (
        "metal_n24_r2_v16_exact_packed2",
        "metal_n24_r2_v16_exact_direct_packed2",
        "metal_n24_r2_v16_exact",
    )

    modules = benchmark._candidate_modules(
        names,
        SimpleNamespace(),
        gate,
        up,
        down,
    )

    assert modules == {name: name for name in names}
    assert stack_calls == [((gate, up), -1)]
    assert [call[-1] for call in module_calls] == [packed, packed, None]
