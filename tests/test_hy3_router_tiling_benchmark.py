from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "hy3_router_tiling.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("hy3_router_tiling", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_tiling_benchmark_covers_k0_through_k7() -> None:
    module = _load_script()

    assert module.ROUTER_ROWS == tuple(range(1, 9))


def test_router_tiling_candidates_exhaust_supported_direct_n_p_frontier() -> None:
    module = _load_script()

    candidates = module.router_tiling_candidates()

    assert tuple(candidates) == (
        "n16_p1_tg12",
        "n16_p2_tg24",
        "n16_p4_tg48",
        "n16_p8_tg96",
        "n16_p16_tg192",
        "n16_p32_tg384",
        "n32_p1_tg6",
        "n32_p2_tg12",
        "n32_p4_tg24",
        "n32_p8_tg48",
        "n32_p16_tg96",
        "n32_p32_tg192",
        "n64_p1_tg3",
        "n64_p2_tg6",
        "n64_p4_tg12",
        "n64_p8_tg24",
        "n64_p16_tg48",
        "n64_p32_tg96",
    )
    assert candidates["n16_p4_tg48"].stage1_threadgroups == 48
    assert candidates["n32_p8_tg48"].partial_bytes == 48 * 1024
    assert candidates["n64_p16_tg48"].partial_bytes == 96 * 1024
    assert candidates["n16_p32_tg384"].partial_bytes == 192 * 1024


def test_mpp_frontier_pairs_compare_every_schedule_to_retained_incumbent() -> None:
    module = _load_script()

    candidates = module.router_tiling_candidates()
    pairs = module.mpp_frontier_comparison_pairs()

    assert len(pairs) == len(candidates) - 1
    assert {control for control, _candidate in pairs} == {"n16_p8_tg96"}
    assert {candidate for _control, candidate in pairs} == set(candidates) - {
        "n16_p8_tg96"
    }
    assert ("n16_p8_tg96", "n16_p2_tg24") in pairs
    assert ("n16_p8_tg96", "n16_p32_tg384") in pairs


def test_grouped_frontier_covers_direct_and_staged_controls() -> None:
    module = _load_script()

    candidates = module.grouped_router_tiling_candidates()

    assert "n16_p8_sg4_grouped_direct" in candidates
    assert "n16_p8_sg4_staged" in candidates
    assert "n32_p16_sg6_staged" in candidates
    assert "n64_p32_sg3_staged" in candidates
    assert len(candidates) == 48
    assert {tiling.operand_mode for tiling in candidates.values()} == {
        "grouped-direct",
        "grouped-staged",
    }


def test_candidate_failure_is_recorded_without_aborting_other_arms() -> None:
    module = _load_script()

    def raises_resource() -> object:
        raise RuntimeError("threadgroup memory resource limit")

    observed = module.evaluate_candidate_arms(
        {"bad": raises_resource, "good": lambda: object()},
        evaluator=lambda _value: None,
    )

    assert observed["bad"]["failure_phase"] == "resource"
    assert observed["bad"]["status"] == "failed"
    assert observed["good"] == {"status": "ok"}


def test_rotated_arm_order_balances_first_and_last_positions() -> None:
    module = _load_script()
    names = ("stock", "a", "b")

    observed = [module.rotated_arm_order(names, repeat) for repeat in range(6)]

    assert observed == [
        ("stock", "a", "b"),
        ("a", "b", "stock"),
        ("b", "stock", "a"),
        ("b", "a", "stock"),
        ("stock", "b", "a"),
        ("a", "stock", "b"),
    ]


def test_paired_comparison_reports_control_over_candidate_speedup() -> None:
    module = _load_script()

    comparison = module.paired_comparison(
        [2.0, 2.1, 1.9, 2.0],
        [1.0, 1.1, 0.9, 1.0],
        bootstrap_resamples=2_000,
        seed=51,
    )

    assert comparison["paired_ratio_mean"] == pytest.approx(2.0, rel=0.03)
    assert comparison["bootstrap_mean_ratio_95_ci"][0] > 1.0


def test_named_pairwise_comparisons_preserve_the_requested_boundary() -> None:
    module = _load_script()

    observed = module.named_pairwise_comparisons(
        {
            "mlx": [2.0, 2.1, 1.9, 2.0],
            "simd": [1.0, 1.1, 0.9, 1.0],
        },
        pairs=(("mlx", "simd"),),
        bootstrap_resamples=2_000,
        seed=52,
    )

    assert tuple(observed) == ("mlx_over_simd",)
    assert observed["mlx_over_simd"]["ratio_of_means"] == pytest.approx(2.0)
    assert observed["mlx_over_simd"]["bootstrap_mean_ratio_95_ci"][0] > 1.0


@pytest.mark.parametrize(
    (
        "candidate_topk_exact",
        "candidate_weights_valid",
        "within_mode_deterministic",
        "ci",
        "expected",
    ),
    (
        (True, True, True, (1.01, 1.08), True),
        (False, True, True, (1.01, 1.08), False),
        (True, False, True, (1.01, 1.08), False),
        (True, True, False, (1.01, 1.08), False),
        (True, True, True, (0.99, 1.08), False),
    ),
)
def test_authoritative_mpp_decision_uses_own_logits_determinism_and_speed(
    candidate_topk_exact: bool,
    candidate_weights_valid: bool,
    within_mode_deterministic: bool,
    ci: tuple[float, float],
    expected: bool,
) -> None:
    module = _load_script()

    assert (
        module.router_candidate_passes(
            candidate_topk_exact=candidate_topk_exact,
            candidate_weights_valid=candidate_weights_valid,
            within_mode_deterministic=within_mode_deterministic,
            bootstrap_mean_ratio_95_ci=ci,
        )
        is expected
    )


def test_authoritative_mpp_contract_uses_candidate_relative_r2_and_normalization() -> (
    None
):
    module = _load_script()

    contract = module.authoritative_candidate_contract(
        candidate_topk_exact=True,
        max_candidate_route_weight_abs_error=8.94e-8,
        weights_finite=True,
        max_normalized_sum_abs_error=2.4e-7,
        repeated_ids_exact=True,
        repeated_weights_exact=True,
    )

    assert contract == {
        "candidate_topk_exact": True,
        "candidate_weights_valid": True,
        "within_mode_deterministic": True,
    }


def test_authoritative_mpp_contract_fails_nonfinite_or_nondeterministic_output() -> (
    None
):
    module = _load_script()

    nonfinite = module.authoritative_candidate_contract(
        candidate_topk_exact=True,
        max_candidate_route_weight_abs_error=0.0,
        weights_finite=False,
        max_normalized_sum_abs_error=0.0,
        repeated_ids_exact=True,
        repeated_weights_exact=True,
    )
    nondeterministic = module.authoritative_candidate_contract(
        candidate_topk_exact=True,
        max_candidate_route_weight_abs_error=0.0,
        weights_finite=True,
        max_normalized_sum_abs_error=0.0,
        repeated_ids_exact=False,
        repeated_weights_exact=True,
    )

    assert nonfinite["candidate_weights_valid"] is False
    assert nondeterministic["within_mode_deterministic"] is False
