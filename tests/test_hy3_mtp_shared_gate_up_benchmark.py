"""Regression tests for the Issue #69 benchmark evidence harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
