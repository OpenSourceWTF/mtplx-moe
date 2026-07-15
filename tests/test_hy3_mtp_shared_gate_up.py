"""Data-shape-specific shared gate/up projection tests for Hy3 MTP."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.hy3_mtp_shared_gate_up as shared_gate_up
from mtplx.hy3_mtp_shared_gate_up import (
    DepthGatedMTPSharedMLP,
    Hy3MTPGateUpCandidate,
    hy3_mtp_gate_up_savings,
    hy3_mtp_gate_up_candidates,
    install_depth_gated_mtp_shared_mlp,
    render_hy3_mtp_gate_up_source,
)


def test_depth_gated_exact_shared_mlp_switches_once_per_configured_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class Projection:
        def __init__(self, name: str) -> None:
            self.weight = f"{name}-weight"
            self.name = name

        def __call__(self, value):
            calls.append((self.name, value))
            return (self.name, value)

    class StockShared:
        def __init__(self) -> None:
            self.gate_proj = Projection("gate")
            self.up_proj = Projection("up")
            self.down_proj = Projection("down")

        def __call__(self, value):
            calls.append(("stock", value))
            return ("stock", value)

    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=16,
        rows_per_simdgroup=2,
    )

    def fake_fused(value, gate_weight, up_weight, *, candidate):
        calls.append(("metal", (value, gate_weight, up_weight, candidate.name)))
        return ("activated", value)

    monkeypatch.setattr(
        shared_gate_up,
        "hy3_mtp_fused_gate_up_swiglu",
        fake_fused,
    )
    module = DepthGatedMTPSharedMLP(
        StockShared(),
        candidate=candidate,
        minimum_depth=3,
    )

    assert module.active_mode == "stock"
    assert module("d1") == ("stock", "d1")
    module.configure_depth(3)
    assert module.active_mode == "metal-exact"
    assert module("d3") == ("down", ("activated", "d3"))
    module.configure_depth(1)
    assert module.active_mode == "stock"
    assert module("d1-again") == ("stock", "d1-again")
    assert calls == [
        ("stock", "d1"),
        (
            "metal",
            ("d3", "gate-weight", "up-weight", "n24_r2_v16_exact"),
        ),
        ("down", ("activated", "d3")),
        ("stock", "d1-again"),
    ]


def test_depth_gated_shared_mlp_rejects_approximate_or_packed_runtime_arms() -> None:
    with pytest.raises(ValueError, match="exact split-weight"):
        DepthGatedMTPSharedMLP(
            object(),
            candidate=Hy3MTPGateUpCandidate(
                n_tile=24,
                k_vector=16,
                rows_per_simdgroup=2,
                activation_mode="fast",
            ),
            minimum_depth=3,
        )
    with pytest.raises(ValueError, match="exact split-weight"):
        DepthGatedMTPSharedMLP(
            object(),
            candidate=Hy3MTPGateUpCandidate(
                n_tile=24,
                k_vector=16,
                rows_per_simdgroup=2,
                weight_layout="packed2",
            ),
            minimum_depth=3,
        )
    with pytest.raises(ValueError, match="minimum_depth"):
        DepthGatedMTPSharedMLP(
            object(),
            candidate=Hy3MTPGateUpCandidate(
                n_tile=24,
                k_vector=16,
                rows_per_simdgroup=2,
            ),
            minimum_depth=0,
        )


def test_depth_gated_install_reuses_the_loaded_projection_arrays() -> None:
    class Weight:
        def __init__(self, shape) -> None:
            self.shape = shape
            self.dtype = mx.bfloat16

    stock = SimpleNamespace(
        gate_proj=SimpleNamespace(weight=Weight((1536, 4096))),
        up_proj=SimpleNamespace(weight=Weight((1536, 4096))),
        down_proj=SimpleNamespace(weight=Weight((4096, 1536))),
    )
    mtp = SimpleNamespace(
        layers=[
            SimpleNamespace(
                mtp_block=SimpleNamespace(
                    mlp=SimpleNamespace(shared_mlp=stock),
                )
            )
        ]
    )

    assert install_depth_gated_mtp_shared_mlp(mtp, minimum_depth=3) == 1
    wrapped = mtp.layers[0].mtp_block.mlp.shared_mlp
    assert isinstance(wrapped, DepthGatedMTPSharedMLP)
    assert wrapped.stock is stock
    assert wrapped.stock.gate_proj.weight is stock.gate_proj.weight
    assert wrapped.stock.up_proj.weight is stock.up_proj.weight
    assert wrapped.stock.down_proj.weight is stock.down_proj.weight
    assert not hasattr(wrapped, "gate_weight")
    assert not hasattr(wrapped, "up_weight")
    assert not hasattr(wrapped, "down_weight")


def test_mtp_gate_up_frontier_is_pinned_to_the_real_m1_shape() -> None:
    candidates = hy3_mtp_gate_up_candidates()

    assert len(candidates) == 200
    assert {candidate.n_tile for candidate in candidates} == {
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
    }
    assert {candidate.k_vector for candidate in candidates} == {1, 2, 4, 8, 16}
    assert {candidate.rows_per_simdgroup for candidate in candidates} == {1, 2, 4, 8}
    assert {candidate.activation_mode for candidate in candidates} == {"exact"}
    assert len({candidate.name for candidate in candidates}) == len(candidates)


def test_mtp_gate_up_candidate_rejects_invalid_tiling() -> None:
    with pytest.raises(ValueError, match="n_tile"):
        Hy3MTPGateUpCandidate(n_tile=5, k_vector=4)
    with pytest.raises(ValueError, match="k_vector"):
        Hy3MTPGateUpCandidate(n_tile=4, k_vector=3)
    with pytest.raises(ValueError, match="rows_per_simdgroup"):
        Hy3MTPGateUpCandidate(n_tile=4, k_vector=4, rows_per_simdgroup=3)
    with pytest.raises(ValueError, match="activation_mode"):
        Hy3MTPGateUpCandidate(n_tile=4, k_vector=4, activation_mode="approx")
    with pytest.raises(ValueError, match="reduction_layout"):
        Hy3MTPGateUpCandidate(n_tile=4, k_vector=4, reduction_layout="unknown")
    with pytest.raises(ValueError, match="k_vector=4"):
        Hy3MTPGateUpCandidate(
            n_tile=4,
            k_vector=16,
            reduction_layout="stock_tn4",
        )
    with pytest.raises(ValueError, match="input_mode"):
        Hy3MTPGateUpCandidate(n_tile=4, k_vector=4, input_mode="unknown")
    with pytest.raises(ValueError, match="weight_layout"):
        Hy3MTPGateUpCandidate(n_tile=4, k_vector=4, weight_layout="unknown")


def test_mtp_gate_up_source_caches_m1_input_and_selects_exact_math() -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=4,
        k_vector=4,
        rows_per_simdgroup=4,
    )

    source = render_hy3_mtp_gate_up_source(candidate)

    assert "threadgroup T activation_tile[4096]" in source
    assert "metal::exp(metal::abs(gate_value))" in source
    assert "precise::exp" not in source
    assert "fast::exp" not in source
    assert "simd_sum" in source
    assert "N_TILE = 4" in source
    assert "K_VECTOR = 4" in source
    assert "ROWS_PER_SIMDGROUP = 4" in source
    assert "float gate_sum[ROWS_PER_SIMDGROUP]" in source
    assert "float activation = float(activation_tile[k]);" in source
    assert "if (n < N)" not in source
    assert "? sigmoid_base : 1 - sigmoid_base" in source
    assert "uint k_base = 0;" in source
    assert "uint k = k_base + offset * 32 + lane;" in source
    assert "lane * K_VECTOR" not in source
    assert "constexpr uint K = 4096" in source
    assert "constant constexpr" not in source


def test_mtp_gate_up_stock_tn4_source_matches_mlx_gemv_reduction_order() -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=4,
        rows_per_simdgroup=2,
        reduction_layout="stock_tn4",
    )

    source = render_hy3_mtp_gate_up_source(candidate)

    assert candidate.name == "n24_r2_v4_exact_stock_tn4"
    assert "uint k = k_base + lane * K_VECTOR + offset;" in source
    assert "k_base += 32 * K_VECTOR" in source
    assert "for (ushort delta = 16; delta >= 1; delta >>= 1)" in source
    assert "simd_shuffle_down(gate_reduced, delta)" in source
    assert "simd_shuffle_down(up_reduced, delta)" in source
    assert "simd_sum" not in source


def test_mtp_gate_up_reduction_factor_arms_separate_k_order_from_tree() -> None:
    striped_tree = render_hy3_mtp_gate_up_source(
        Hy3MTPGateUpCandidate(
            n_tile=24,
            k_vector=16,
            rows_per_simdgroup=2,
            reduction_layout="striped_tree",
        )
    )
    stock_sum = render_hy3_mtp_gate_up_source(
        Hy3MTPGateUpCandidate(
            n_tile=24,
            k_vector=4,
            rows_per_simdgroup=2,
            reduction_layout="stock_tn4_sum",
        )
    )

    assert "uint k = k_base + offset * 32 + lane;" in striped_tree
    assert "simd_shuffle_down(gate_reduced, delta)" in striped_tree
    assert "simd_sum" not in striped_tree
    assert "uint k = k_base + lane * K_VECTOR + offset;" in stock_sum
    assert "simd_sum(gate_sum[row])" in stock_sum
    assert "simd_shuffle_down" not in stock_sum


def test_mtp_gate_up_direct_input_source_removes_threadgroup_cache_and_barrier() -> (
    None
):
    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=4,
        rows_per_simdgroup=2,
        reduction_layout="stock_tn4",
        input_mode="direct",
    )

    source = render_hy3_mtp_gate_up_source(candidate)

    assert candidate.name == "n24_r2_v4_exact_stock_tn4_direct"
    assert "threadgroup T activation_tile[4096]" not in source
    assert "threadgroup_barrier" not in source
    assert "float activation = float(input_values[k]);" in source


def test_mtp_gate_up_fp32_cache_converts_input_once_during_fill() -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=16,
        rows_per_simdgroup=2,
        input_mode="threadgroup_f32",
    )

    source = render_hy3_mtp_gate_up_source(candidate)
    savings = hy3_mtp_gate_up_savings(candidate, depth=3)

    assert candidate.name == "n24_r2_v16_exact_threadgroup_f32"
    assert "threadgroup float activation_tile[4096]" in source
    assert "activation_tile[k] = float(input_values[k]);" in source
    assert "float activation = float(activation_tile[k]);" in source
    assert savings["threadgroup_storage_bytes"] == 16_384
    assert savings["threadgroup_barriers"] == 96
    assert savings["input_device_load_instruction_bytes"] == 786_432
    assert savings["input_threadgroup_load_instruction_bytes"] == 37_748_736
    assert savings["input_bf16_to_fp32_conversions"] == 393_216


def test_mtp_gate_up_packed2_source_vector_loads_interleaved_gate_up() -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=16,
        rows_per_simdgroup=2,
        weight_layout="packed2",
    )

    source = render_hy3_mtp_gate_up_source(candidate)

    assert candidate.name == "n24_r2_v16_exact_packed2"
    assert "device const vec<T, 2>* packed_pairs" in source
    assert "vec<T, 2> weight_pair = packed_pairs[n * K + k];" in source
    assert "float(weight_pair[0])" in source
    assert "float(weight_pair[1])" in source
    assert "gate_weight" not in source
    assert "up_weight" not in source


def test_mtp_gate_up_packed2_kernel_declares_one_weight_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=16,
        rows_per_simdgroup=2,
        weight_layout="packed2",
    )
    captured = {}

    def fake_metal_kernel(**kwargs):
        captured.update(kwargs)
        return "kernel"

    shared_gate_up._KERNEL_CACHE.clear()
    monkeypatch.setattr(shared_gate_up.mx.fast, "metal_kernel", fake_metal_kernel)

    assert shared_gate_up.build_hy3_mtp_gate_up_kernel(candidate) == "kernel"
    assert captured["input_names"] == ["input_values", "packed_weight"]


def test_mtp_gate_up_savings_are_explicit_at_k3() -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=4,
        k_vector=4,
        rows_per_simdgroup=4,
    )

    savings = hy3_mtp_gate_up_savings(candidate, depth=3)

    assert savings == {
        "depth": 3,
        "logical_dispatches_saved": 6,
        "host_synchronizations_saved": 0,
        "gate_up_weight_bytes_required": 75_497_472,
        "intermediate_device_bytes_avoided": 36_864,
        "threadgroup_storage_bytes": 8_192,
        "threadgroups": 288,
        "threadgroup_barriers": 288,
        "input_fill_load_instruction_bytes": 2_359_296,
        "input_device_load_instruction_bytes": 2_359_296,
        "input_threadgroup_load_instruction_bytes": 9_437_184,
        "input_bf16_to_fp32_conversions": 4_718_592,
        "steady_extra_weight_bytes": 0,
    }


def test_mtp_gate_up_direct_input_savings_report_no_tg_storage_or_barriers() -> None:
    candidate = Hy3MTPGateUpCandidate(
        n_tile=24,
        k_vector=4,
        rows_per_simdgroup=2,
        reduction_layout="stock_tn4",
        input_mode="direct",
    )

    savings = hy3_mtp_gate_up_savings(candidate, depth=3)

    assert savings["threadgroup_storage_bytes"] == 0
    assert savings["threadgroup_barriers"] == 0
    assert savings["input_fill_load_instruction_bytes"] == 0
    assert savings["input_device_load_instruction_bytes"] == 18_874_368
    assert savings["input_threadgroup_load_instruction_bytes"] == 0
    assert savings["input_bf16_to_fp32_conversions"] == 9_437_184
