from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mtplx.benchmarks.issue51 import (
    A1_CANDIDATES,
    A1_PROCESS_CONFIG,
    ISSUE51_PRIORITY,
    CampaignCell,
    build_abba_schedule,
    decide_performance,
    pair_abba_rows,
    paired_decode_statistics,
    validate_a1_child,
)


_ROOT = Path(__file__).resolve().parent.parent
_RUNNER = _ROOT / "scripts" / "run_issue51_hy3_q2.py"
_SUMMARIZER = _ROOT / "scripts" / "summarize_issue51_hy3_q2.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compiled_evidence(arm: str) -> dict[str, object] | None:
    mode, _strategy = A1_PROCESS_CONFIG[arm]
    if mode == "off":
        return None
    return {
        "calls": 4,
        "compiled_calls": 4,
        "fallback_calls": 0,
        "fallback_reasons": {},
        "mode": mode,
    }


def _child_payload(
    arm: str,
    *,
    decode_tok_s: float = 16.0,
    end_to_end_tok_s: float = 8.0,
    depths: tuple[int, ...] = (1,),
    resource_telemetry: bool = False,
) -> dict[str, object]:
    mode, strategy = A1_PROCESS_CONFIG[arm]
    observations = []
    for context in (1024, 2048):
        observations.append(
            {
                "context_tokens": context,
                "requested_depth": 0,
                "generated_tokens": 128,
                "decode_tok_s": 15.0,
                "end_to_end_tok_s": 7.0,
                "expert_resource_telemetry": (
                    {
                        "decode": {
                            "reader_pool": {
                                "worker_capacity": 32,
                                "mean_active_readers": 3.0,
                            }
                        }
                    }
                    if resource_telemetry
                    else None
                ),
                "gates": {"output_tokens_exact": True},
            }
        )
        for depth in depths:
            observations.append(
                {
                    "context_tokens": context,
                    "requested_depth": depth,
                    "generated_tokens": 128,
                    "decode_tok_s": decode_tok_s + depth + context / 2048,
                    "end_to_end_tok_s": end_to_end_tok_s + depth + context / 4096,
                    "expert_resource_telemetry": (
                        {
                            "decode": {
                                "reader_pool": {
                                    "worker_capacity": 32,
                                    "mean_active_readers": 4.0,
                                }
                            }
                        }
                        if resource_telemetry
                        else None
                    ),
                    "compiled_verify": _compiled_evidence(arm),
                    "final_state_contract": {
                        "safe_to_commit": True,
                        "generated_token_ids_match": True,
                        "finish_reason_match": True,
                        "prompt_mtp_history_tokens": context - 1,
                        "mtp_history_position_base": 0,
                        "target_cache_offsets": [context + 128],
                        "committed_mtp_cache_offsets": [context + 127],
                    },
                    "gates": {
                        "prompt_length_exact": True,
                        "new_prefill_tokens_exact": True,
                        "output_tokens_exact": True,
                        "generated_count_consistent": True,
                        "length_finish": True,
                        "requested_depth_exact": True,
                        "effective_depth_exact": True,
                        "committed_history": True,
                        "guards_disabled": True,
                        "decode_expert_cache_metrics": True,
                        "speculative_event_contract": True,
                        "final_state_contract": True,
                        "compiled_verify_evidence": True,
                    },
                }
            )
    return {
        "schema": "mtplx-q2-bf16-mtp-depth-matrix-v3",
        "status": "passed",
        "passed": True,
        "configuration": {
            "contexts": [1024, 2048],
            "output_tokens": 128,
            "candidate": {
                "verify_strategy": strategy,
                "compiled_verify_mode": mode,
                "trace_routes": False,
            },
            "measurement_lane": (
                "diagnostic-resource-instrumented"
                if resource_telemetry
                else "headline-uninstrumented"
            ),
        },
        "models": [
            {
                "model": "hy3-q2",
                "model_key": "hy3-expert-q2",
                "depths": list(depths),
                "passed": True,
                "observations": observations,
            }
        ],
    }


def _write_child(
    path: Path,
    arm: str,
    *,
    speed: float,
    resource_telemetry: bool = False,
) -> str:
    path.write_text(
        json.dumps(
            _child_payload(
                arm,
                decode_tok_s=speed,
                resource_telemetry=resource_telemetry,
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_entry(path: Path, arm: str, index: int, digest: str) -> dict[str, object]:
    schedule = build_abba_schedule(
        control="capture-eager",
        candidate="capture-compiled",
        retained_pairs=2,
    )[index]
    return {
        "path": path.name,
        "sha256": digest,
        "arm": arm,
        "schedule_index": index,
        "block": schedule.block,
        "pair_slot": schedule.pair_slot,
    }


def _valid_index(tmp_path: Path) -> dict[str, object]:
    qualification = []
    for arm in A1_CANDIDATES:
        path = tmp_path / f"qual-{arm}.json"
        qualification.append(
            {
                "path": path.name,
                "sha256": _write_child(path, arm, speed=16.0),
                "arm": arm,
            }
        )

    schedule = build_abba_schedule(
        control="capture-eager",
        candidate="capture-compiled",
        retained_pairs=2,
    )
    artifacts = []
    for row in schedule:
        path = tmp_path / f"perf-{row.index:02d}-{row.arm}.json"
        digest = _write_child(
            path,
            row.arm,
            speed=16.0 if row.arm == "capture-eager" else 18.0,
        )
        artifacts.append(_artifact_entry(path, row.arm, row.index, digest))
    diagnostics = []
    for index in range(4):
        path = tmp_path / f"diagnostic-{index:02d}-capture-compiled.json"
        diagnostics.append(
            {
                "path": path.name,
                "sha256": _write_child(
                    path,
                    "capture-compiled",
                    speed=18.0,
                    resource_telemetry=True,
                ),
                "arm": "capture-compiled",
            }
        )
    return {
        "schema": "mtplx-issue51-a1-campaign-v1",
        "stage": "a1",
        "status": "passed",
        "configuration": {
            "contexts": [1024, 2048],
            "depths": [1],
            "output_tokens": 128,
            "retained_pairs": 2,
            "diagnostic_repeats": 4,
        },
        "qualifications": qualification,
        "comparisons": [
            {
                "name": "eager-vs-compiled",
                "control": "capture-eager",
                "candidate": "capture-compiled",
                "schedule": [
                    {
                        "index": row.index,
                        "block": row.block,
                        "arm": row.arm,
                        "pair_slot": row.pair_slot,
                    }
                    for row in schedule
                ],
                "artifacts": artifacts,
            }
        ],
        "diagnostics": diagnostics,
    }


def test_priority_candidates_and_process_modes_are_fixed() -> None:
    assert ISSUE51_PRIORITY == (
        "compiled_whole_window",
        "mtp_hint_only_prediction",
        "q2_nax_grouping",
    )
    assert A1_CANDIDATES == (
        "batched-stock",
        "capture-eager",
        "capture-compiled-parity",
        "capture-compiled",
    )
    assert A1_PROCESS_CONFIG == {
        "batched-stock": ("off", "batched"),
        "capture-eager": ("off", "capture_commit"),
        "capture-compiled-parity": ("parity", "capture_commit"),
        "capture-compiled": ("on", "capture_commit"),
    }


def test_abba_schedule_is_balanced_and_pairs_in_temporal_order() -> None:
    schedule = build_abba_schedule(
        control="capture-eager",
        candidate="capture-compiled",
        retained_pairs=8,
    )

    assert [row.arm for row in schedule] == [
        "capture-eager",
        "capture-compiled",
        "capture-compiled",
        "capture-eager",
    ] * 4
    pairs = pair_abba_rows(schedule)
    assert len(pairs) == 8
    assert [(left.index, right.index) for left, right in pairs[:2]] == [(0, 1), (3, 2)]
    assert all(left.arm == "capture-eager" for left, _right in pairs)
    assert all(right.arm == "capture-compiled" for _left, right in pairs)


@pytest.mark.parametrize("retained_pairs", [0, -2, 1, 3])
def test_abba_schedule_rejects_nonpositive_or_odd_pairs(retained_pairs: int) -> None:
    with pytest.raises(ValueError, match="positive even"):
        build_abba_schedule(control="a", candidate="b", retained_pairs=retained_pairs)


def test_pairing_rejects_incomplete_or_mutated_abba_blocks() -> None:
    schedule = list(build_abba_schedule(control="a", candidate="b", retained_pairs=2))
    with pytest.raises(ValueError, match="complete ABBA"):
        pair_abba_rows(schedule[:-1])
    schedule[1] = type(schedule[1])(
        index=1,
        block=0,
        arm="a",
        pair_slot=1,
    )
    with pytest.raises(ValueError, match="ABBA"):
        pair_abba_rows(schedule)


def test_child_validation_returns_k0_and_k1_metrics() -> None:
    cells = validate_a1_child(
        _child_payload("capture-compiled"),
        arm="capture-compiled",
        depths=(1,),
    )

    assert set(cells) == {
        CampaignCell(1024, 0),
        CampaignCell(1024, 1),
        CampaignCell(2048, 0),
        CampaignCell(2048, 1),
    }
    assert cells[CampaignCell(1024, 1)]["decode_tok_s"] == 17.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(status="failed"), "passed"),
        (lambda payload: payload.update(passed=False), "passed"),
        (
            lambda payload: payload["configuration"].update(contexts=[1024]),
            "contexts",
        ),
        (
            lambda payload: payload["configuration"].update(output_tokens=127),
            "128",
        ),
        (lambda payload: payload["models"][0].update(depths=[2]), "depths"),
        (
            lambda payload: payload["models"][0]["observations"][1]["gates"].pop(
                "final_state_contract"
            ),
            "gate",
        ),
        (
            lambda payload: payload["models"][0]["observations"][1].update(
                decode_tok_s=float("nan")
            ),
            "finite",
        ),
    ],
)
def test_child_validation_fails_closed(mutation, message: str) -> None:
    payload = _child_payload("capture-compiled")
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_a1_child(payload, arm="capture-compiled", depths=(1,))


def test_child_validation_rejects_candidate_mismatch_and_compiled_fallback() -> None:
    with pytest.raises(ValueError, match="candidate"):
        validate_a1_child(
            _child_payload("capture-eager"),
            arm="capture-compiled",
            depths=(1,),
        )

    payload = _child_payload("capture-compiled")
    payload["models"][0]["observations"][1]["compiled_verify"]["fallback_calls"] = 1
    with pytest.raises(ValueError, match="fallback"):
        validate_a1_child(payload, arm="capture-compiled", depths=(1,))


def test_child_validation_requires_the_fixed_32_reader_utilization_metric() -> None:
    payload = _child_payload("capture-compiled", resource_telemetry=True)
    payload["models"][0]["observations"][0]["expert_resource_telemetry"]["decode"][
        "reader_pool"
    ]["worker_capacity"] = 31

    with pytest.raises(ValueError, match="32"):
        validate_a1_child(payload, arm="capture-compiled", depths=(1,))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_cache_offsets", [1151], "target cache"),
        ("committed_mtp_cache_offsets", [1150], "MTP cache"),
        ("prompt_mtp_history_tokens", 1022, "prompt MTP history"),
        ("mtp_history_position_base", 1, "position base"),
        ("generated_token_ids_match", False, "generated token"),
        ("finish_reason_match", False, "finish reason"),
    ],
)
def test_child_validation_recomputes_terminal_mtp_cache_contract(
    field: str, value: object, message: str
) -> None:
    payload = _child_payload("capture-compiled")
    contract = payload["models"][0]["observations"][1]["final_state_contract"]
    contract.update(
        {
            "target_cache_offsets": [1152],
            "committed_mtp_cache_offsets": [1151],
            "prompt_mtp_history_tokens": 1023,
            "mtp_history_position_base": 0,
            "generated_token_ids_match": True,
            "finish_reason_match": True,
        }
    )
    contract[field] = value

    with pytest.raises(ValueError, match=message):
        validate_a1_child(payload, arm="capture-compiled", depths=(1,))


def test_paired_statistics_are_deterministic_and_performance_is_fail_closed() -> None:
    rows = [
        {
            "control_decode_tok_s": 100.0,
            "candidate_decode_tok_s": value,
            "control_end_to_end_tok_s": 50.0,
            "candidate_end_to_end_tok_s": value / 2,
        }
        for value in (106.0, 108.0, 107.0, 109.0, 106.0, 108.0, 107.0, 109.0)
    ]

    first = paired_decode_statistics(rows)
    second = paired_decode_statistics(rows)
    assert first == second
    assert first["samples"] == 8
    assert len(first["paired_fractional_decode_gains"]) == 8
    assert first["bootstrap_samples"] == 10_000
    assert first["bootstrap_95_interval"][0] > 0.05
    assert first["end_to_end_bootstrap_95_interval"][0] > 0.0
    assert decide_performance(first)["promote"] is True

    too_small = deepcopy(first)
    too_small["mean_fractional_decode_gain"] = 0.049
    assert decide_performance(too_small)["promote"] is False
    no_e2e = deepcopy(first)
    no_e2e["end_to_end_bootstrap_95_interval"] = (-0.01, 0.1)
    assert decide_performance(no_e2e)["promote"] is False


@pytest.mark.parametrize(
    "row",
    [
        {
            "control_decode_tok_s": 0.0,
            "candidate_decode_tok_s": 1.0,
            "control_end_to_end_tok_s": 1.0,
            "candidate_end_to_end_tok_s": 1.0,
        },
        {
            "control_decode_tok_s": 1.0,
            "candidate_decode_tok_s": float("inf"),
            "control_end_to_end_tok_s": 1.0,
            "candidate_end_to_end_tok_s": 1.0,
        },
    ],
)
def test_paired_statistics_reject_nonfinite_and_nonpositive_values(row) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        paired_decode_statistics([row])


def test_runner_builds_exact_one_candidate_child_command(tmp_path: Path) -> None:
    runner = _load_script(_RUNNER, "run_issue51")
    output = tmp_path / "child.json"

    command, environment = runner.build_a1_child_invocation(
        arm="capture-compiled-parity",
        contexts=(1024, 2048),
        depths=(1,),
        output_tokens=128,
        output_path=output,
        python_executable="python",
    )

    assert command[0] == "python"
    assert command.count("--model") == 1
    assert command[command.index("--model") + 1] == "hy3-q2"
    assert command[command.index("--contexts") + 1] == "1024,2048"
    assert command[command.index("--hy3-depths") + 1] == "1"
    assert command[command.index("--verify-strategy") + 1] == "capture_commit"
    assert command[command.index("--compiled-verify-mode") + 1] == "parity"
    assert command[command.index("--output-json") + 1] == str(output)
    assert environment["MTPLX_COMPILED_VERIFY"] == "parity"
    assert environment["MTPLX_COMPILED_VERIFY_FORCE"] == "1"
    assert environment["MTPLX_SUSTAINED_PREFILL"] == "1"


def test_runner_forces_only_q2_compiled_candidates(tmp_path: Path, monkeypatch) -> None:
    runner = _load_script(_RUNNER, "run_issue51_force")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_FORCE", "inherited-host-value")

    _command, stock_environment = runner.build_a1_child_invocation(
        arm="batched-stock",
        contexts=(1024, 2048),
        depths=(1,),
        output_tokens=128,
        output_path=tmp_path / "stock.json",
    )
    _command, compiled_environment = runner.build_a1_child_invocation(
        arm="capture-compiled",
        contexts=(1024, 2048),
        depths=(1,),
        output_tokens=128,
        output_path=tmp_path / "compiled.json",
    )

    assert "MTPLX_COMPILED_VERIFY_FORCE" not in stock_environment
    assert compiled_environment["MTPLX_COMPILED_VERIFY_FORCE"] == "1"


def test_runner_refuses_to_replace_existing_child_artifact(tmp_path: Path) -> None:
    runner = _load_script(_RUNNER, "run_issue51_existing")
    path = tmp_path / "child.json"
    path.write_text("owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_a1_child(
            arm="batched-stock",
            contexts=(1024, 2048),
            depths=(1,),
            output_tokens=128,
            output_path=path,
            run_process=lambda *_args, **_kwargs: None,
        )


def test_summarizer_validates_digests_schedule_and_renders_decisions(
    tmp_path: Path,
) -> None:
    summarizer = _load_script(_SUMMARIZER, "summarize_issue51")
    index = _valid_index(tmp_path)

    summary = summarizer.summarize_a1_index(index, base_dir=tmp_path)
    markdown = summarizer.render_markdown(summary)

    assert summary["qualification"]["passed"] is True
    assert summary["comparisons"][0]["cells"]
    assert summary["next_k_gate"]["advance_to_k2"] is True
    assert "A1 correctness" in markdown
    assert "A1 performance" in markdown
    assert "capture-compiled" in markdown
    assert "K=2 gate: GO" in markdown


def test_runner_rejects_combined_k1_k2_campaign(tmp_path: Path) -> None:
    runner = _load_script(_RUNNER, "run_issue51_combined")

    with pytest.raises(ValueError, match="one depth at a time"):
        runner.build_a1_child_invocation(
            arm="batched-stock",
            contexts=(1024, 2048),
            depths=(1, 2),
            output_tokens=128,
            output_path=tmp_path / "combined.json",
        )


def test_k2_requires_a_passing_k1_speed_and_utilization_summary() -> None:
    runner = _load_script(_RUNNER, "run_issue51_k2_gate")

    with pytest.raises(ValueError, match="passing K=1 summary"):
        runner.validate_depth_authorization((2,), None)

    failed = {
        "schema": "mtplx-issue51-a1-summary-v2",
        "next_k_gate": {
            "tested_depth": 1,
            "max_depth": 2,
            "advance_to_k2": False,
        },
    }
    with pytest.raises(ValueError, match="does not authorize"):
        runner.validate_depth_authorization((2,), failed)

    passed = deepcopy(failed)
    passed["next_k_gate"]["advance_to_k2"] = True
    assert runner.validate_depth_authorization((2,), passed) == {
        "tested_depth": 1,
        "max_depth": 2,
        "advance_to_k2": True,
    }

    with pytest.raises(ValueError, match="K=1 or K=2"):
        runner.validate_depth_authorization((3,), passed)


def test_summarizer_rejects_duplicate_paths_digest_drift_and_schedule_drift(
    tmp_path: Path,
) -> None:
    summarizer = _load_script(_SUMMARIZER, "summarize_issue51_invalid")

    duplicate = _valid_index(tmp_path)
    duplicate["qualifications"][1]["path"] = duplicate["qualifications"][0]["path"]
    with pytest.raises(ValueError, match="duplicate artifact path"):
        summarizer.summarize_a1_index(duplicate, base_dir=tmp_path)


def test_summarizer_rejects_digest_and_schedule_drift(tmp_path: Path) -> None:
    summarizer = _load_script(_SUMMARIZER, "summarize_issue51_drift")
    digest_dir = tmp_path / "digest"
    digest_dir.mkdir()
    index = _valid_index(digest_dir)
    index["qualifications"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        summarizer.summarize_a1_index(index, base_dir=digest_dir)

    schedule_dir = tmp_path / "schedule"
    schedule_dir.mkdir()
    index = _valid_index(schedule_dir)
    index["comparisons"][0]["schedule"][1]["arm"] = "capture-eager"
    with pytest.raises(ValueError, match="schedule"):
        summarizer.summarize_a1_index(index, base_dir=schedule_dir)
