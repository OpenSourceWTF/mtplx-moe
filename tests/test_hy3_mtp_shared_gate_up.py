"""Data-shape-specific shared gate/up projection tests for Hy3 MTP."""

from __future__ import annotations

import pytest

from mtplx.hy3_mtp_shared_gate_up import (
    Hy3MTPGateUpCandidate,
    hy3_mtp_gate_up_savings,
    hy3_mtp_gate_up_candidates,
    render_hy3_mtp_gate_up_source,
)


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
