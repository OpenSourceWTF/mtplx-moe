from __future__ import annotations

from copy import deepcopy

import pytest

from mtplx.qwen27b_mtp_cohort import (
    validate_qwen27b_mtp_cohort_selfcheck_report,
)


def _comparison(
    path: str,
    shape: list[int],
    *,
    tolerance: float,
) -> dict[str, object]:
    return {
        "path": path,
        "candidate_shape": shape,
        "reference_shape": shape,
        "dmax": 0.0,
        "tolerance": tolerance,
    }


def _cache_layers(*, path: str, tolerance: float) -> list[dict[str, object]]:
    layers: list[dict[str, object]] = []
    attention_layers = set(range(3, 64, 4))
    for row in range(2):
        for layer_index in range(64):
            kind = (
                "attention"
                if layer_index in attention_layers
                else "recurrent"
            )
            layer: dict[str, object] = {
                "row": row,
                "layer_index": layer_index,
                "kind": kind,
                "state_comparisons": [
                    _comparison(
                        f"{path}.row{row}.layer{layer_index}.state.0",
                        [1, 64],
                        tolerance=tolerance,
                    ),
                ],
            }
            if kind == "attention":
                layer.update(
                    {
                        "candidate_offset": 1100,
                        "reference_offset": 1100,
                    }
                )
            layers.append(layer)
    return layers


def _passing_report() -> dict[str, object]:
    attention_layers = set(range(3, 64, 4))
    recurrent_layers = [
        layer_index
        for layer_index in range(64)
        if layer_index not in attention_layers
    ]
    return {
        "schema": "qwen27b-mtp-cohort-selfcheck-v1",
        "status": "pass",
        "prefill_chunk_tokens": 1024,
        "prefill_prompt_tokens": {"0": 1100, "1": 1101},
        "prefill_spans": {
            "0": [[0, 1024], [1024, 1100]],
            "1": [[0, 1024], [1024, 1101]],
        },
        "target_cache_reference": "two_owned_clones_per_prefilled_row",
        "qlinear": {
            "reference": "mx.quantized_matmul_transpose_q4_group64",
            "expected_module_count": 2,
            "tested_module_count": 2,
            "expected_shapes": [
                {"k": 64, "n": 32, "module_count": 1},
                {"k": 128, "n": 64, "module_count": 1},
            ],
            "tested_shapes": [
                {"k": 64, "n": 32, "module_count": 1},
                {"k": 128, "n": 64, "module_count": 1},
            ],
            "routes": [
                {
                    "module_path": "model.layers.0.self_attn.q_proj",
                    "k": 64,
                    "n": 32,
                    "input_shape": [2, 3, 64],
                    "output_shape": [2, 3, 32],
                    "dmax": 0.0,
                },
                {
                    "module_path": "model.layers.0.mlp.up_proj",
                    "k": 128,
                    "n": 64,
                    "input_shape": [2, 3, 128],
                    "output_shape": [2, 3, 64],
                    "dmax": 0.0,
                },
            ],
        },
        "target_cycle": {
            "input_shape": [2, 3],
            "acceptance_source": "GenerationOutput.stats.accepted_drafts",
            "output_comparisons": [
                _comparison("logits", [2, 3, 256], tolerance=1.0),
                _comparison("hidden", [2, 3, 64], tolerance=1.0),
                *[
                    _comparison(
                        f"captures.row{row}.{layer_index}.gate",
                        [1, 3, 64],
                        tolerance=1.0,
                    )
                    for row in range(2)
                    for layer_index in recurrent_layers
                ],
            ],
            "starting_cache_layers": _cache_layers(
                path="starting_cache",
                tolerance=0.0,
            ),
            "starting_cache_aliasing": [
                {
                    "row": row,
                    "candidate_reference_aliasing": False,
                    "candidate_sibling_aliasing": False,
                    "reference_sibling_aliasing": False,
                }
                for row in range(2)
            ],
            "cache_layers": _cache_layers(path="cache", tolerance=1.0),
            "commit_order_layers": _cache_layers(
                path="commit_order",
                tolerance=0.0,
            ),
            "rows": [
                {
                    "row": 0,
                    "candidate_tokens": [11, 12],
                    "reference_tokens": [11, 12],
                    "candidate_accepted_drafts": 1,
                    "reference_accepted_drafts": 1,
                },
                {
                    "row": 1,
                    "candidate_tokens": [21, 22],
                    "reference_tokens": [21, 22],
                    "candidate_accepted_drafts": 0,
                    "reference_accepted_drafts": 0,
                },
            ],
            "isolation": [
                {
                    "row": 0,
                    "sibling_row": 1,
                    "extracted_aliasing": False,
                    "sibling_unchanged_after_mutation": True,
                    "sibling_unchanged_after_commit": True,
                },
                {
                    "row": 1,
                    "sibling_row": 0,
                    "extracted_aliasing": False,
                    "sibling_unchanged_after_mutation": True,
                    "sibling_unchanged_after_commit": True,
                },
            ],
        },
    }


def test_selfcheck_report_accepts_complete_passing_receipt() -> None:
    report = _passing_report()

    assert validate_qwen27b_mtp_cohort_selfcheck_report(report) is report


def test_selfcheck_report_rejects_missing_qlinear_shape() -> None:
    report = _passing_report()
    report["qlinear"]["tested_shapes"].pop()  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="qlinear shape coverage"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_qmm_dmax_over_turbo_tolerance() -> None:
    report = _passing_report()
    report["qlinear"]["routes"][0]["dmax"] = 0.10001  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match=r"qlinear.*dmax.*0\.1"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_non_laguna_prefill_chunk_geometry() -> None:
    report = _passing_report()
    report["prefill_chunk_tokens"] = 512

    with pytest.raises(ValueError, match="prefill_chunk_tokens must remain 1024"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_short_prefill_that_never_crosses_chunk() -> None:
    report = _passing_report()
    report["prefill_prompt_tokens"]["1"] = 700  # type: ignore[index]
    report["prefill_spans"]["1"] = [[0, 700]]  # type: ignore[index]

    with pytest.raises(ValueError, match=r"prefill row 1.*full 1024"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_noncontiguous_prefill_coverage() -> None:
    report = _passing_report()
    report["prefill_spans"]["0"][1] = [1025, 1100]  # type: ignore[index]

    with pytest.raises(ValueError, match=r"prefill row 0.*contiguous"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_noncloned_target_reference() -> None:
    report = _passing_report()
    report["target_cache_reference"] = "independent_prefill"

    with pytest.raises(ValueError, match="owned target-cache clones"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


@pytest.mark.parametrize(
    "path",
    ["logits", "hidden", "captures.row0.0.gate"],
)
def test_selfcheck_report_rejects_target_output_shape_mismatch(path: str) -> None:
    report = _passing_report()
    comparisons = report["target_cycle"]["output_comparisons"]  # type: ignore[index]
    comparison = next(item for item in comparisons if item["path"] == path)
    comparison["reference_shape"] = [9, 9, 9]

    with pytest.raises(ValueError, match=rf"{path}.*shape"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


@pytest.mark.parametrize(
    ("path", "inflated"),
    [
        ("logits", 1.1),
        ("hidden", 1.1),
        ("captures.row0.0.gate", 1.1),
    ],
)
def test_selfcheck_report_rejects_inflated_target_tolerance(
    path: str,
    inflated: float,
) -> None:
    report = _passing_report()
    comparisons = report["target_cycle"]["output_comparisons"]  # type: ignore[index]
    comparison = next(item for item in comparisons if item["path"] == path)
    comparison["tolerance"] = inflated

    with pytest.raises(ValueError, match=rf"{path}.*tolerance"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_requires_capture_coverage_for_every_recurrent_layer() -> None:
    report = _passing_report()
    comparisons = report["target_cycle"]["output_comparisons"]  # type: ignore[index]
    comparisons.remove(
        next(item for item in comparisons if item["path"] == "captures.row1.62.gate")
    )

    with pytest.raises(ValueError, match=r"capture layer coverage.*row 1"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_attention_offset_mismatch() -> None:
    report = _passing_report()
    attention = next(
        item
        for item in report["target_cycle"]["cache_layers"]  # type: ignore[index]
        if item["row"] == 0 and item["layer_index"] == 3
    )
    attention["reference_offset"] = 1099

    with pytest.raises(ValueError, match="attention.*offset"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_recurrent_state_mismatch() -> None:
    report = _passing_report()
    recurrent = next(
        item
        for item in report["target_cycle"]["cache_layers"]  # type: ignore[index]
        if item["row"] == 0 and item["layer_index"] == 0
    )
    recurrent["state_comparisons"][0]["dmax"] = 1.1

    with pytest.raises(
        ValueError,
        match=r"cache\.row0\.layer0\.state\.0.*dmax",
    ):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_missing_cache_layer_coverage() -> None:
    report = _passing_report()
    report["target_cycle"]["cache_layers"].pop()  # type: ignore[index]

    with pytest.raises(ValueError, match="target cache layer coverage"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_starting_cache_aliasing() -> None:
    report = _passing_report()
    report["target_cycle"]["starting_cache_aliasing"][0][  # type: ignore[index]
        "candidate_reference_aliasing"
    ] = True

    with pytest.raises(ValueError, match=r"starting cache row 0.*aliases"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_extracted_row_aliasing() -> None:
    report = _passing_report()
    report["target_cycle"]["isolation"][0]["extracted_aliasing"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="row 0.*aliases.*row 1"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("candidate_tokens", [11, 99], "row 0 token parity"),
        ("candidate_accepted_drafts", 0, "row 0 acceptance parity"),
    ],
)
def test_selfcheck_report_rejects_token_or_acceptance_mismatch(
    field: str,
    value: object,
    match: str,
) -> None:
    report = deepcopy(_passing_report())
    report["target_cycle"]["rows"][0][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=match):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


@pytest.mark.parametrize(
    "field",
    [
        "sibling_unchanged_after_mutation",
        "sibling_unchanged_after_commit",
    ],
)
def test_selfcheck_report_rejects_sibling_contamination(field: str) -> None:
    report = _passing_report()
    report["target_cycle"]["isolation"][1][field] = False  # type: ignore[index]

    with pytest.raises(ValueError, match=r"row 1.*changed.*row 0"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_report_rejects_duplicate_isolation_pair() -> None:
    report = _passing_report()
    report["target_cycle"]["isolation"][1].update(  # type: ignore[index]
        {"row": 0, "sibling_row": 1}
    )

    with pytest.raises(ValueError, match="exact row pairs"):
        validate_qwen27b_mtp_cohort_selfcheck_report(report)


def test_selfcheck_shape_diagnostic_names_shapes_delta_and_tolerance() -> None:
    report = _passing_report()
    comparison = report["target_cycle"]["output_comparisons"][0]  # type: ignore[index]
    comparison["reference_shape"] = [9, 9, 9]
    comparison["dmax"] = 0.25

    with pytest.raises(ValueError) as captured:
        validate_qwen27b_mtp_cohort_selfcheck_report(report)

    message = str(captured.value)
    assert "candidate_shape=" in message
    assert "reference_shape=" in message
    assert "dmax=0.25" in message
    assert "tolerance=1.0" in message
