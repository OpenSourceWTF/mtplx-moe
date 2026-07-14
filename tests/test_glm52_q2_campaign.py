from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mtplx.benchmarks.glm52_q2_campaign import (
    summarize_glm52_q2_campaign,
)


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "summarize_glm52_q2_campaign.py"
)
_GIB = 1024**3


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "summarize_glm52_q2_campaign", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(
    kind: str,
    position: int,
    lane: str,
    *,
    tps: float,
    token_ids: list[int] | None = None,
    text: str | None = None,
) -> dict:
    assert kind in {"resource", "headline"}
    assert lane in {"q4", "q2"}
    telemetry = kind == "resource"
    model_key = "glm52-q4" if lane == "q4" else "glm52-expert-q2"
    slots = 64 if lane == "q4" else 116
    token_ids = token_ids or [101, 102, 103]
    text = text if text is not None else f"output-{lane}"
    read_bytes = ((100 if lane == "q4" else 60) + position * 10) * 1024**2
    read_gib_s = (5.0 if lane == "q4" else 7.0) + position / 10
    miss_wait = (0.4 if lane == "q4" else 0.2) + position / 100

    runtime_config = {
        "model_key": model_key,
        "memory_limit_bytes": 160 * _GIB,
        "max_live_kv_tokens": 8192,
        "runtime_reserve_bytes": 16 * _GIB,
        "expert_cache_limit_bytes": 96 * _GIB,
        "cache_policy": "frequency",
        "cache_scope": "layer",
        "slot_layout": "component-banks",
        "max_read_chunk_bytes": 8 * 1024**2,
        "bypass_page_cache": True,
        "verify_record_hashes": True,
        "resource_telemetry": telemetry,
    }
    performance_settings = {
        "runtime_config": runtime_config,
        "sampler": {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
        "seed": 0,
        "prompt_identity": {
            "content_sha256": "a" * 64,
            "content_bytes": 123,
            "token_sha256": "b" * 64,
            "token_count": 17,
        },
        "prompt_options": {
            "chat": False,
            "system_prompt": "fixed",
            "prompt_style": "coding-agent",
            "enable_thinking": False,
            "reasoning_effort": None,
            "prompt_metadata": None,
        },
        "generation": {
            "generation_profile": "deterministic",
            "max_tokens": 128,
            "context_tokens": None,
            "window_tokens": 32,
            "window_telemetry": False,
            "repeats": 1,
            "reset_between": True,
        },
        "scheduler": {
            "requested_concurrency": 1,
            "max_prefills_per_step": 1,
            "execution_lane": "reference-ar",
            "workload_shape": "static",
            "join_after_step": None,
        },
        "mtp": {"enabled": False, "precision": "bf16", "artifact_identity": None},
        "model_artifact": {
            "method": "manifest_plus_executable_resident_content_v1",
            "manifest": {
                "model_key": model_key,
                "content_sha256": ("c" if lane == "q4" else "d") * 64,
            },
            "expert_payload": {
                "sha256": ("e" if lane == "q4" else "f") * 64,
                "size": 407_686_348_800 if lane == "q4" else 226_492_416_000,
            },
            "harness_source": {
                "source_sha256": "1" * 64,
                "dependency_versions": {"mlx": "0.31.0"},
                "dirty": False,
            },
        },
    }
    label = f"glm52-q2-{kind}-{position}-{lane}"
    configuration_summary = {
        "run_label": label,
        "configuration_label": f"derived-{position}-{lane}-{kind}",
        "configuration_fingerprint": f"{position:016x}",
        "cache_scope": "layer",
        "slot_layout": "component-banks",
        "concurrency": 1,
        "requested_concurrency": 1,
        "execution_lane": "reference-ar",
        "performance_settings": performance_settings,
    }
    run = {
        "run_label": label,
        "execution_lane": "reference-ar",
        "completion_tokens": len(token_ids),
        "completion_tokens_per_second": tps,
        "token_ids": token_ids,
        "text": text,
        "finish_reason": "length",
        "streaming_after": {
            "memory_plan": {
                "total_limit_bytes": 160 * _GIB,
                "fixed_bytes": 32 * _GIB,
                "persistent_cache_bytes": 96 * _GIB,
                "slots_per_layer": slots,
                "cache_scope": "layer",
                "transient_slots": 8,
                "allocated_bytes": 144 * _GIB,
                "unallocated_bytes": 16 * _GIB,
            },
            "integrity": {
                "valid": True,
                "model_key": model_key,
                "checked_shards": 1,
                "checked_records": 19_200,
                "sidecar_verified": False,
            },
            "memory_cap": {"applied": True, "limit": 144 * _GIB},
        },
    }
    if telemetry:
        run["diagnostic_run"] = True
        run["resource_telemetry"] = {
            "schema": "mtplx-resource-telemetry-v2",
            "interval_count": 10,
            "elapsed_seconds": 10.0,
            "throughput": {
                "completion_tokens": len(token_ids),
                "completion_tokens_per_second": tps - 0.25,
                "expert_requests": 1000,
                "expert_requests_per_second": 100.0,
            },
            "storage": {
                "io_cache_modes": ["f-nocache"],
                "reader_read_bytes": read_bytes,
                "reader_read_operations": 200,
                "mean_gib_per_second": read_gib_s,
            },
            "expert_pipeline": {
                "potentially_blocking_next_miss_fraction": miss_wait,
                "coverage": {
                    "attribution": "measured",
                    "decode_phase": "measured",
                    "potentially_blocking_next_miss_step": "measured_upper_bound",
                    "generation_expert_input_wait": "unavailable",
                    "gpu_expert_wait": "unavailable",
                },
            },
            "coverage": {
                "runtime_occupancy": "measured",
                "storage_reads": "uncached_reader_bytes",
                "ssd_ceiling": "supplied",
                "gpu": "unavailable",
                "dram_bandwidth": "unavailable",
                "generation_thread_cpu": "measured",
            },
        }
    return {
        "schema": "mtplx-streamed-generation-benchmark-v1",
        "model_key": model_key,
        "seed": 0,
        "generation_profile": "deterministic",
        "run_label": label,
        "cache_scope": "layer",
        "slot_layout": "component-banks",
        "execution_lane": "reference-ar",
        "configuration_summary": configuration_summary,
        "generation": {
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
        "reset_between": True,
        "runs": [run],
    }


def _campaigns() -> tuple[list[dict], list[dict]]:
    resource = [
        _payload("resource", 1, "q4", tps=9.0),
        _payload("resource", 2, "q2", tps=11.0),
        _payload("resource", 3, "q2", tps=13.0),
        _payload("resource", 4, "q4", tps=10.0),
    ]
    headline = [
        _payload("headline", 1, "q4", tps=10.0, token_ids=[1, 2, 3]),
        _payload("headline", 2, "q2", tps=12.0, token_ids=[1, 2, 4]),
        _payload("headline", 3, "q2", tps=18.0, token_ids=[5, 6, 7]),
        _payload("headline", 4, "q4", tps=15.0, token_ids=[5, 6, 7]),
        _payload("headline", 5, "q4", tps=20.0, token_ids=[8, 9, 10]),
        _payload("headline", 6, "q2", tps=30.0, token_ids=[8, 9, 10]),
    ]
    return resource, headline


@pytest.mark.parametrize(
    ("path", "replacement", "expected_error"),
    [
        (
            ("prompt_identity", "content_sha256"),
            "9" * 64,
            "configuration mismatch",
        ),
        (
            ("generation", "generation_profile"),
            "model-default",
            "configuration mismatch",
        ),
        (("sampler", "temperature"), 0.5, "configuration mismatch"),
        (("seed",), 1, "configuration mismatch"),
        (
            ("runtime_config", "expert_cache_limit_bytes"),
            95 * _GIB,
            "configuration mismatch",
        ),
        (
            ("runtime_config", "memory_limit_bytes"),
            159 * _GIB,
            "configuration mismatch",
        ),
        (("runtime_config", "max_live_kv_tokens"), 4096, "configuration mismatch"),
        (
            ("runtime_config", "runtime_reserve_bytes"),
            15 * _GIB,
            "configuration mismatch",
        ),
        (("runtime_config", "cache_policy"), "lru", "configuration mismatch"),
        (("runtime_config", "cache_scope"), "global", "configuration mismatch"),
        (("runtime_config", "slot_layout"), "direct-slots", "configuration mismatch"),
        (
            ("runtime_config", "max_read_chunk_bytes"),
            4 * 1024**2,
            "configuration mismatch",
        ),
        (("runtime_config", "bypass_page_cache"), False, "configuration mismatch"),
    ],
)
def test_rejects_any_comparability_change(
    path: tuple[str, ...], replacement: object, expected_error: str
) -> None:
    resource, headline = _campaigns()
    settings = headline[-1]["configuration_summary"]["performance_settings"]
    target = settings
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert any(expected_error in error for error in summary["errors"])


def test_harness_identity_is_not_removed_with_model_artifact_identity() -> None:
    resource, headline = _campaigns()
    headline[-1]["configuration_summary"]["performance_settings"]["model_artifact"][
        "harness_source"
    ]["source_sha256"] = "9" * 64

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert any("configuration mismatch" in error for error in summary["errors"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 1),
        ("generation_profile", "model-default"),
        ("generation.max_tokens", 64),
        ("generation.temperature", 0.5),
        ("reset_between", False),
        ("cache_scope", "global"),
        ("slot_layout", "direct-slots"),
        ("configuration_resource_telemetry", True),
    ],
)
def test_rejects_top_level_or_lane_telemetry_declaration_drift(
    field: str, replacement: object
) -> None:
    resource, headline = _campaigns()
    target = headline[-1]
    if field.startswith("generation."):
        target["generation"][field.split(".", 1)[1]] = replacement
    elif field == "configuration_resource_telemetry":
        target["configuration_summary"]["performance_settings"]["runtime_config"][
            "resource_telemetry"
        ] = replacement
    else:
        target[field] = replacement

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert summary["errors"]


def test_malformed_lane_pair_is_reported_instead_of_crashing() -> None:
    resource, headline = _campaigns()
    headline[1]["model_key"] = "glm52-q4"

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert any("order mismatch" in error for error in summary["errors"])


@pytest.mark.parametrize(
    "location",
    ["payload", "configuration_summary", "scheduler", "run"],
)
def test_requires_reference_ar_at_every_exported_level(location: str) -> None:
    resource, headline = _campaigns()
    payload = headline[-1]
    if location == "payload":
        payload["execution_lane"] = "continuous-batch-ar"
    elif location == "configuration_summary":
        payload["configuration_summary"]["execution_lane"] = "continuous-batch-ar"
    elif location == "scheduler":
        payload["configuration_summary"]["performance_settings"]["scheduler"][
            "execution_lane"
        ] = "continuous-batch-ar"
    else:
        payload["runs"][0]["execution_lane"] = "continuous-batch-ar"

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert any("reference-ar" in error for error in summary["errors"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_resource_payload",
        "missing_headline_payload",
        "missing_run",
        "extra_run",
        "resource_order",
        "headline_order",
        "wrong_label",
    ],
)
def test_rejects_missing_runs_and_wrong_declared_order(mutation: str) -> None:
    resource, headline = _campaigns()
    if mutation == "missing_resource_payload":
        resource.pop()
    elif mutation == "missing_headline_payload":
        headline.pop()
    elif mutation == "missing_run":
        headline[0]["runs"] = []
    elif mutation == "extra_run":
        headline[0]["runs"].append(deepcopy(headline[0]["runs"][0]))
    elif mutation == "resource_order":
        resource[0], resource[1] = resource[1], resource[0]
    elif mutation == "headline_order":
        headline[0], headline[1] = headline[1], headline[0]
    elif mutation == "wrong_label":
        headline[0]["run_label"] = "glm52-q2-headline-99-q4"

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert any(
        phrase in " ".join(summary["errors"])
        for phrase in ("exactly", "order", "run label")
    )


@pytest.mark.parametrize(("lane", "slots"), [("q4", 63), ("q2", 115)])
def test_requires_exact_96_gib_cache_slots(lane: str, slots: int) -> None:
    resource, headline = _campaigns()
    target = next(
        payload for payload in resource if payload["run_label"].endswith(lane)
    )
    target["runs"][0]["streaming_after"]["memory_plan"]["slots_per_layer"] = slots

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert summary["resource_gate"]["passed"] is False
    assert any("slots_per_layer" in error for error in summary["errors"])


def test_keeps_resource_rates_separate_and_summarizes_all_pairs() -> None:
    resource, headline = _campaigns()

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is True
    assert summary["errors"] == []
    assert summary["cache_slots_per_layer"] == {"q4": 64, "q2": 116}

    assert summary["headline"]["q4"] == {
        "samples": 3,
        "mean_completion_tokens_per_second": 15.0,
        "median_completion_tokens_per_second": 15.0,
        "min_completion_tokens_per_second": 10.0,
        "max_completion_tokens_per_second": 20.0,
        "range_completion_tokens_per_second": 10.0,
        "relative_range": pytest.approx(2 / 3),
    }
    assert summary["headline"]["q2"]["samples"] == 3
    assert summary["headline"]["q2"]["mean_completion_tokens_per_second"] == 20.0
    assert summary["headline"]["q2"]["median_completion_tokens_per_second"] == 18.0
    assert summary["headline"]["pooled_mean_percent_change"] == pytest.approx(100 / 3)
    assert summary["headline"]["pooled_median_percent_change"] == pytest.approx(20.0)
    assert summary["headline"]["pairs"]["percent_changes"] == pytest.approx(
        [20.0, 20.0, 50.0]
    )
    assert summary["headline"]["pairs"]["mean_percent_change"] == pytest.approx(30.0)
    assert summary["headline"]["pairs"]["median_percent_change"] == pytest.approx(20.0)
    assert summary["headline"]["order_splits"]["q4_first"][
        "mean_percent_change"
    ] == pytest.approx(35.0)
    assert summary["headline"]["order_splits"]["q2_first"][
        "mean_percent_change"
    ] == pytest.approx(20.0)

    assert "resource_telemetry_completion_tokens_per_second" not in summary["headline"]
    assert "completion_tokens_per_second" not in summary["resource"]["q4"]
    assert summary["resource"]["q4"]["diagnostic_samples"] == 2
    assert summary["resource"]["q4"]["reader_read_bytes_total"] > 0
    assert summary["resource"]["q2"]["mean_reader_gib_per_second"] > 0
    assert (
        summary["resource"]["q4"]["mean_potentially_blocking_next_miss_fraction"]
        > summary["resource"]["q2"]["mean_potentially_blocking_next_miss_fraction"]
    )
    assert "coverage.gpu=unavailable" in summary["resource"]["coverage_gaps"]
    assert (
        "expert_pipeline.coverage.generation_expert_input_wait=unavailable"
        in summary["resource"]["coverage_gaps"]
    )

    divergence = summary["headline"]["output_divergence"]
    assert divergence["pairs"] == 3
    assert divergence["token_divergent_pairs"] == 1
    assert divergence["text_divergent_pairs"] == 3
    assert divergence["details"][0]["first_token_divergence_index"] == 2
    assert divergence["details"][1]["first_token_divergence_index"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "resource_missing_telemetry",
        "headline_has_telemetry",
        "nonfinite_tps",
        "nonfinite_storage",
        "bad_integrity",
        "bad_memory_cap",
        "missing_coverage",
    ],
)
def test_resource_and_numeric_failures_reject_campaign(mutation: str) -> None:
    resource, headline = _campaigns()
    if mutation == "resource_missing_telemetry":
        resource[0]["runs"][0].pop("resource_telemetry")
    elif mutation == "headline_has_telemetry":
        headline[0]["runs"][0]["resource_telemetry"] = deepcopy(
            resource[0]["runs"][0]["resource_telemetry"]
        )
    elif mutation == "nonfinite_tps":
        headline[0]["runs"][0]["completion_tokens_per_second"] = float("nan")
    elif mutation == "nonfinite_storage":
        resource[0]["runs"][0]["resource_telemetry"]["storage"][
            "mean_gib_per_second"
        ] = float("inf")
    elif mutation == "bad_integrity":
        resource[0]["runs"][0]["streaming_after"]["integrity"]["valid"] = False
    elif mutation == "bad_memory_cap":
        resource[0]["runs"][0]["streaming_after"]["memory_cap"]["applied"] = False
    elif mutation == "missing_coverage":
        resource[0]["runs"][0]["resource_telemetry"].pop("coverage")

    summary = summarize_glm52_q2_campaign(resource, headline)

    assert summary["valid"] is False
    assert summary["resource_gate"]["passed"] is False


def test_cli_writes_invalid_json_before_exit_two(tmp_path: Path) -> None:
    module = _load_script()
    resource, headline = _campaigns()
    headline.pop()
    resource_paths = []
    headline_paths = []
    for index, payload in enumerate(resource):
        path = tmp_path / f"resource-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        resource_paths.append(path)
    for index, payload in enumerate(headline):
        path = tmp_path / f"headline-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        headline_paths.append(path)
    output = tmp_path / "summary.json"

    status = module.main(
        [
            "--resource",
            *map(str, resource_paths),
            "--headline",
            *map(str, headline_paths),
            "--output-json",
            str(output),
        ]
    )

    assert status == 2
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["valid"] is False
    assert written["errors"]


def test_cli_refuses_tracked_output_path(tmp_path: Path) -> None:
    module = _load_script()
    resource, headline = _campaigns()
    resource_path = tmp_path / "resource.json"
    headline_path = tmp_path / "headline.json"
    resource_path.write_text(json.dumps(resource[0]), encoding="utf-8")
    headline_path.write_text(json.dumps(headline[0]), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the Git worktree"):
        module.validate_external_output_path(
            Path(__file__).resolve().parent / "tracked-summary.json"
        )
