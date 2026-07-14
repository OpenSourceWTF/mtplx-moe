"""Strict comparison and aggregation for the GLM expert-Q2 campaign."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from mtplx.expert_streaming_models import get_model_spec


GIB = 1024**3
RESOURCE_ORDER = ("q4", "q2", "q2", "q4")
HEADLINE_ORDER = ("q4", "q2", "q2", "q4", "q4", "q2")
MODEL_KEYS = {"q4": "glm52-q4", "q2": "glm52-expert-q2"}
EXPECTED_SLOTS = {"q4": 64, "q2": 116}
EXPECTED_RUNTIME = {
    "memory_limit_bytes": 160 * GIB,
    "max_live_kv_tokens": 8192,
    "runtime_reserve_bytes": 16 * GIB,
    "expert_cache_limit_bytes": 96 * GIB,
    "cache_policy": "frequency",
    "cache_scope": "layer",
    "slot_layout": "component-banks",
    "max_read_chunk_bytes": 8 * 1024**2,
    "bypass_page_cache": True,
    "verify_record_hashes": True,
    "transient_slots": None,
    "io_staging_bytes": 0,
    "execution_workspace_bytes": 0,
}
EXPECTED_GLOBAL_COVERAGE = frozenset(
    {
        "runtime_occupancy",
        "storage_reads",
        "ssd_ceiling",
        "gpu",
        "dram_bandwidth",
        "generation_thread_cpu",
        "timeline",
    }
)
EXPECTED_PIPELINE_COVERAGE = frozenset(
    {
        "attribution",
        "decode_phase",
        "sampler_window_backend",
        "potentially_blocking_next_miss_step",
        "generation_expert_input_wait",
        "operation_credit",
        "byte_credit",
        "authoritative_reserve",
        "slot_capacity_admission",
        "outer_split_executor_queue",
        "eligible_unsubmitted_cause",
        "admitted_read_ranges",
        "scheduled_read_ranges",
        "physical_device_operations",
        "physical_device_bytes",
        "physical_device_queue_depth",
        "gpu_expert_wait",
        "gpu_idle_time",
        "future_layer_eligibility",
        "speculative_record_accounting",
        "python_preadv_when_native_reader",
    }
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nonfinite_numeric_paths(value: object, *, path: str) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_nonfinite_numeric_paths(item, path=f"{path}.{key}"))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_nonfinite_numeric_paths(item, path=f"{path}[{index}]"))
        return paths
    return []


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lane(payload: Mapping[str, Any]) -> str | None:
    model_key = payload.get("model_key")
    for lane, expected in MODEL_KEYS.items():
        if model_key == expected:
            return lane
    return None


def _normalized_configuration(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only per-run, model-artifact, and telemetry-presence identity."""

    normalized = deepcopy(dict(summary))
    for key in ("run_label", "configuration_label", "configuration_fingerprint"):
        normalized.pop(key, None)

    settings = normalized.get("performance_settings")
    if not isinstance(settings, dict):
        return normalized
    runtime = settings.get("runtime_config")
    if isinstance(runtime, dict):
        runtime.pop("model_key", None)
        runtime.pop("resource_telemetry", None)

    # Executable model bytes intentionally differ. The harness source nested in
    # that identity does not, and remains a comparability requirement.
    artifact = settings.get("model_artifact")
    if isinstance(artifact, dict):
        settings["model_artifact"] = {
            "harness_source": deepcopy(artifact.get("harness_source"))
        }
    return normalized


def _stats(values: Sequence[float], *, sample_label: str) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            f"mean_{sample_label}": None,
            f"median_{sample_label}": None,
            f"min_{sample_label}": None,
            f"max_{sample_label}": None,
            f"range_{sample_label}": None,
            "relative_range": None,
        }
    mean = statistics.fmean(values)
    low = min(values)
    high = max(values)
    return {
        "samples": len(values),
        f"mean_{sample_label}": mean,
        f"median_{sample_label}": statistics.median(values),
        f"min_{sample_label}": low,
        f"max_{sample_label}": high,
        f"range_{sample_label}": high - low,
        "relative_range": (high - low) / mean if mean else None,
    }


def _percent_change(q4: float, q2: float) -> float:
    return (q2 / q4 - 1.0) * 100.0


def _pair_summary(
    values: Sequence[tuple[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    changes: list[float] = []
    order_changes = {"q4_first": [], "q2_first": []}
    pair_details = []
    for pair_index in range(0, len(values), 2):
        first_lane, first_value = values[pair_index]
        second_lane, second_value = values[pair_index + 1]
        lane_values = {first_lane: first_value, second_lane: second_value}
        change = _percent_change(lane_values["q4"], lane_values["q2"])
        changes.append(change)
        order_name = f"{first_lane}_first"
        order_changes[order_name].append(change)
        pair_details.append(
            {
                "pair": pair_index // 2 + 1,
                "order": [first_lane, second_lane],
                "q4": lane_values["q4"],
                "q2": lane_values["q2"],
                "percent_change": change,
            }
        )

    pairs = {
        "samples": len(changes),
        "percent_changes": changes,
        "mean_percent_change": statistics.fmean(changes) if changes else None,
        "median_percent_change": statistics.median(changes) if changes else None,
        "min_percent_change": min(changes) if changes else None,
        "max_percent_change": max(changes) if changes else None,
        "range_percent_change": max(changes) - min(changes) if changes else None,
        "details": pair_details,
    }
    order_splits = {
        name: {
            "samples": len(split),
            "mean_percent_change": statistics.fmean(split) if split else None,
            "median_percent_change": statistics.median(split) if split else None,
            "percent_changes": split,
        }
        for name, split in order_changes.items()
    }
    return pairs, order_splits


def _first_token_divergence(left: Sequence[Any], right: Sequence[Any]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _output_divergence(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    details = []
    token_divergent = 0
    text_divergent = 0
    for pair_index in range(0, len(entries), 2):
        first_lane, first_run = entries[pair_index]
        second_lane, second_run = entries[pair_index + 1]
        runs = {first_lane: first_run, second_lane: second_run}
        q4_tokens = list(_sequence(runs["q4"].get("token_ids")))
        q2_tokens = list(_sequence(runs["q2"].get("token_ids")))
        q4_text = runs["q4"].get("text")
        q2_text = runs["q2"].get("text")
        first_divergence = _first_token_divergence(q4_tokens, q2_tokens)
        token_equal = first_divergence is None
        text_equal = isinstance(q4_text, str) and q4_text == q2_text
        token_divergent += int(not token_equal)
        text_divergent += int(not text_equal)
        details.append(
            {
                "pair": pair_index // 2 + 1,
                "order": [first_lane, second_lane],
                "token_ids_equal": token_equal,
                "first_token_divergence_index": first_divergence,
                "q4_token_count": len(q4_tokens),
                "q2_token_count": len(q2_tokens),
                "q4_token_sha256": _hash_json(q4_tokens),
                "q2_token_sha256": _hash_json(q2_tokens),
                "text_equal": text_equal,
                "q4_text_sha256": (
                    hashlib.sha256(q4_text.encode("utf-8")).hexdigest()
                    if isinstance(q4_text, str)
                    else None
                ),
                "q2_text_sha256": (
                    hashlib.sha256(q2_text.encode("utf-8")).hexdigest()
                    if isinstance(q2_text, str)
                    else None
                ),
            }
        )
    return {
        "pairs": len(details),
        "token_divergent_pairs": token_divergent,
        "text_divergent_pairs": text_divergent,
        "details": details,
    }


def _coverage_gaps(telemetry: Mapping[str, Any]) -> list[str]:
    gaps: list[str] = []
    sections = (
        (
            "coverage",
            _mapping(telemetry.get("coverage")),
            EXPECTED_GLOBAL_COVERAGE,
        ),
        (
            "expert_pipeline.coverage",
            _mapping(_mapping(telemetry.get("expert_pipeline")).get("coverage")),
            EXPECTED_PIPELINE_COVERAGE,
        ),
    )
    for prefix, coverage, expected_keys in sections:
        for key in sorted(expected_keys - set(coverage)):
            gaps.append(f"{prefix}.{key}=missing")
        for key, value in coverage.items():
            status = str(value)
            if status.startswith("measured") or status in {
                "complete",
                "supplied",
                "uncached_reader_bytes",
            }:
                continue
            gaps.append(f"{prefix}.{key}={status}")
    return gaps


def _append_expected_error(
    errors: list[str], label: str, actual: object, expected: object
) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _validate_payloads(
    payloads: Sequence[dict],
    *,
    kind: str,
    order: Sequence[str],
    errors: list[str],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    if len(payloads) != len(order):
        errors.append(
            f"{kind} campaign requires exactly {len(order)} payloads; got {len(payloads)}"
        )
    entries: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for index, raw_payload in enumerate(payloads):
        label = f"{kind}[{index + 1}]"
        if not isinstance(raw_payload, Mapping):
            errors.append(f"{label} must be an object")
            continue
        payload = raw_payload
        lane = _lane(payload)
        expected_lane = order[index] if index < len(order) else None
        if lane is None:
            errors.append(f"{label} has an unsupported model_key")
            continue
        if lane != expected_lane:
            errors.append(
                f"{kind} order mismatch at position {index + 1}: "
                f"expected {expected_lane}, got {lane}"
            )
        expected_label = f"glm52-q2-{kind}-{index + 1}-{expected_lane}"
        if payload.get("run_label") != expected_label:
            errors.append(
                f"{label} run label must be {expected_label!r}; "
                f"got {payload.get('run_label')!r}"
            )
        _append_expected_error(
            errors,
            f"{label} schema",
            payload.get("schema"),
            "mtplx-streamed-generation-benchmark-v1",
        )
        runs = _sequence(payload.get("runs"))
        if len(runs) != 1:
            errors.append(f"{label} requires exactly one run; got {len(runs)}")
            continue
        run = _mapping(runs[0])
        if not run:
            errors.append(f"{label} run must be an object")
            continue
        if run.get("run_label") != expected_label:
            errors.append(f"{label} nested run label does not match declared order")
        configuration_label = _mapping(payload.get("configuration_summary")).get(
            "configuration_label"
        )
        if (
            not isinstance(configuration_label, str)
            or payload.get("configuration_label") != configuration_label
            or run.get("configuration_label") != configuration_label
        ):
            errors.append(
                f"{label} configuration_label differs across payload, summary, and run"
            )
        _append_expected_error(
            errors,
            f"{label} run execution_lane",
            run.get("execution_lane"),
            "reference-ar",
        )
        for field, expected in (
            ("cache_scope", "layer"),
            ("slot_layout", "component-banks"),
            ("concurrency", 1),
            ("requested_concurrency", 1),
            ("achieved_peak_concurrency", 1),
            ("saturation_valid", True),
            ("undersubscribed", False),
        ):
            _append_expected_error(
                errors,
                f"{label} run {field}",
                run.get(field),
                expected,
            )
        if payload.get("error") or run.get("error"):
            errors.append(f"{label} reports an operational error")
        finish_reason = run.get("finish_reason")
        if not isinstance(finish_reason, str) or any(
            marker in finish_reason.lower()
            for marker in ("error", "crash", "exception", "failed")
        ):
            errors.append(f"{label} has invalid finish_reason {finish_reason!r}")
        tps = _finite_number(run.get("completion_tokens_per_second"))
        if tps is None or tps <= 0:
            errors.append(f"{label} has invalid completion_tokens_per_second")
        token_ids = run.get("token_ids")
        if not isinstance(token_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in token_ids
        ):
            errors.append(f"{label} token_ids must be a list of integers")
        completion_tokens = run.get("completion_tokens")
        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
            or not isinstance(token_ids, list)
            or completion_tokens != len(token_ids)
        ):
            errors.append(f"{label} completion_tokens does not match token_ids")
        text = run.get("text")
        if not isinstance(text, str):
            errors.append(f"{label} text must be present for divergence reporting")
        elif completion_tokens and not text:
            errors.append(f"{label} text evidence is empty for a non-empty completion")

        telemetry = run.get("resource_telemetry")
        if kind == "resource":
            if not isinstance(telemetry, Mapping):
                errors.append(f"{label} is missing resource_telemetry")
            if run.get("diagnostic_run") is not True:
                errors.append(f"{label} must be declared diagnostic_run")
        elif telemetry is not None or run.get("diagnostic_run"):
            errors.append(f"{label} headline run must not contain resource telemetry")
        entries.append((lane, payload, run))
    return entries


def _validate_configuration(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    errors: list[str],
) -> None:
    reference: str | None = None
    reference_prompt: str | None = None
    for index, (lane, payload, run) in enumerate(entries):
        label = f"payload[{index + 1}]"
        summary = _mapping(payload.get("configuration_summary"))
        settings = _mapping(summary.get("performance_settings"))
        runtime = _mapping(settings.get("runtime_config"))
        prompt = _mapping(settings.get("prompt_identity"))
        generation = _mapping(settings.get("generation"))
        sampler = _mapping(settings.get("sampler"))
        scheduler = _mapping(settings.get("scheduler"))
        prompt_options = _mapping(settings.get("prompt_options"))
        mtp = _mapping(settings.get("mtp"))
        payload_mtp = _mapping(payload.get("mtp"))
        model_artifact = _mapping(settings.get("model_artifact"))
        top_generation = _mapping(payload.get("generation"))
        if not summary or not settings:
            errors.append(f"{label} is missing configuration_summary")
            continue
        if not prompt or not model_artifact.get("harness_source"):
            errors.append(f"{label} is missing prompt or harness identity")
        else:
            prompt_identity = _canonical(prompt)
            if reference_prompt is None:
                reference_prompt = prompt_identity
            elif prompt_identity != reference_prompt:
                errors.append(f"{label} prompt identity mismatch")

        normalized = _canonical(_normalized_configuration(summary))
        if reference is None:
            reference = normalized
        elif normalized != reference:
            errors.append(f"{label} configuration mismatch after allowed normalization")

        _append_expected_error(
            errors, f"{label} model_key", runtime.get("model_key"), MODEL_KEYS[lane]
        )
        _append_expected_error(
            errors,
            f"{label} runtime_config.resource_telemetry",
            runtime.get("resource_telemetry"),
            run.get("diagnostic_run") is True,
        )
        _append_expected_error(
            errors,
            f"{label} payload execution_lane",
            payload.get("execution_lane"),
            "reference-ar",
        )
        _append_expected_error(
            errors,
            f"{label} configuration execution_lane",
            summary.get("execution_lane"),
            "reference-ar",
        )
        _append_expected_error(
            errors,
            f"{label} scheduler execution_lane",
            scheduler.get("execution_lane"),
            "reference-ar",
        )
        for field, expected in (
            ("cache_scope", "layer"),
            ("slot_layout", "component-banks"),
            ("concurrency", 1),
            ("requested_concurrency", 1),
        ):
            _append_expected_error(
                errors,
                f"{label} configuration {field}",
                summary.get(field),
                expected,
            )
        _append_expected_error(
            errors,
            f"{label} scheduler requested_concurrency",
            scheduler.get("requested_concurrency"),
            1,
        )
        _append_expected_error(
            errors,
            f"{label} scheduler max_prefills_per_step",
            scheduler.get("max_prefills_per_step"),
            1,
        )
        _append_expected_error(
            errors,
            f"{label} scheduler workload_shape",
            scheduler.get("workload_shape"),
            "static",
        )
        _append_expected_error(
            errors,
            f"{label} MTP enabled",
            mtp.get("enabled"),
            False,
        )
        _append_expected_error(
            errors,
            f"{label} payload MTP enabled",
            payload_mtp.get("enabled"),
            False,
        )
        _append_expected_error(
            errors,
            f"{label} thinking enabled",
            prompt_options.get("enable_thinking"),
            False,
        )
        _append_expected_error(
            errors,
            f"{label} payload thinking enabled",
            payload.get("enable_thinking"),
            False,
        )
        _append_expected_error(
            errors,
            f"{label} prompt chat",
            prompt_options.get("chat"),
            False,
        )
        _append_expected_error(
            errors,
            f"{label} payload chat",
            payload.get("chat"),
            False,
        )
        for key, expected in EXPECTED_RUNTIME.items():
            _append_expected_error(
                errors, f"{label} runtime_config.{key}", runtime.get(key), expected
            )
        _append_expected_error(errors, f"{label} seed", settings.get("seed"), 0)
        _append_expected_error(
            errors,
            f"{label} generation_profile",
            generation.get("generation_profile"),
            "deterministic",
        )
        _append_expected_error(
            errors, f"{label} max_tokens", generation.get("max_tokens"), 128
        )
        _append_expected_error(errors, f"{label} repeats", generation.get("repeats"), 1)
        _append_expected_error(
            errors, f"{label} reset_between", generation.get("reset_between"), True
        )
        _append_expected_error(
            errors,
            f"{label} window_telemetry",
            generation.get("window_telemetry"),
            False,
        )
        _append_expected_error(
            errors, f"{label} temperature", sampler.get("temperature"), 0.0
        )
        _append_expected_error(errors, f"{label} top_p", sampler.get("top_p"), 1.0)
        _append_expected_error(errors, f"{label} top_k", sampler.get("top_k"), 1)
        _append_expected_error(errors, f"{label} payload seed", payload.get("seed"), 0)
        _append_expected_error(
            errors,
            f"{label} payload generation_profile",
            payload.get("generation_profile"),
            "deterministic",
        )
        _append_expected_error(
            errors,
            f"{label} payload max_tokens",
            top_generation.get("max_tokens"),
            128,
        )
        _append_expected_error(
            errors,
            f"{label} payload temperature",
            top_generation.get("temperature"),
            0.0,
        )
        _append_expected_error(
            errors, f"{label} payload top_p", top_generation.get("top_p"), 1.0
        )
        _append_expected_error(
            errors, f"{label} payload top_k", top_generation.get("top_k"), 1
        )
        _append_expected_error(
            errors,
            f"{label} payload reset_between",
            payload.get("reset_between"),
            True,
        )
        _append_expected_error(
            errors,
            f"{label} payload cache_scope",
            payload.get("cache_scope"),
            "layer",
        )
        _append_expected_error(
            errors,
            f"{label} payload slot_layout",
            payload.get("slot_layout"),
            "component-banks",
        )
        for field, expected in (
            ("concurrency", 1),
            ("requested_concurrency", 1),
            ("achieved_peak_concurrency", 1),
            ("saturation_valid", True),
            ("undersubscribed", False),
        ):
            _append_expected_error(
                errors,
                f"{label} payload {field}",
                payload.get(field),
                expected,
            )


def _validate_run_gate(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    errors: list[str],
) -> dict[str, set[int]]:
    slots: dict[str, set[int]] = {"q4": set(), "q2": set()}
    for index, (lane, _payload, run) in enumerate(entries):
        label = f"run[{index + 1}]"
        spec = get_model_spec(MODEL_KEYS[lane])
        after = _mapping(run.get("streaming_after"))
        plan = _mapping(after.get("memory_plan"))
        integrity = _mapping(after.get("integrity"))
        cap = _mapping(after.get("memory_cap"))
        slot_value = plan.get("slots_per_layer")
        if isinstance(slot_value, int) and not isinstance(slot_value, bool):
            slots[lane].add(slot_value)
        _append_expected_error(
            errors,
            f"{label} slots_per_layer",
            slot_value,
            EXPECTED_SLOTS[lane],
        )
        _append_expected_error(
            errors,
            f"{label} memory plan total_limit_bytes",
            plan.get("total_limit_bytes"),
            EXPECTED_RUNTIME["memory_limit_bytes"],
        )
        _append_expected_error(
            errors,
            f"{label} memory plan cache_scope",
            plan.get("cache_scope"),
            "layer",
        )
        _append_expected_error(
            errors,
            f"{label} memory plan transient_slots",
            plan.get("transient_slots"),
            spec.top_k,
        )
        expected_fixed = (
            spec.resident_bytes
            + EXPECTED_RUNTIME["max_live_kv_tokens"] * spec.kv_bytes_per_token
            + spec.transient_scratch_bytes
            + EXPECTED_RUNTIME["runtime_reserve_bytes"]
        )
        expected_persistent = spec.persistent_cache_bytes(EXPECTED_SLOTS[lane])
        expected_allocated = expected_fixed + expected_persistent
        expected_unallocated = (
            EXPECTED_RUNTIME["memory_limit_bytes"] - expected_allocated
        )
        _append_expected_error(
            errors,
            f"{label} memory plan fixed_bytes",
            plan.get("fixed_bytes"),
            expected_fixed,
        )
        _append_expected_error(
            errors,
            f"{label} memory plan persistent_cache_bytes",
            plan.get("persistent_cache_bytes"),
            expected_persistent,
        )
        _append_expected_error(
            errors,
            f"{label} memory plan allocated_bytes",
            plan.get("allocated_bytes"),
            expected_allocated,
        )
        _append_expected_error(
            errors,
            f"{label} memory plan unallocated_bytes",
            plan.get("unallocated_bytes"),
            expected_unallocated,
        )
        fixed = plan.get("fixed_bytes")
        persistent = plan.get("persistent_cache_bytes")
        allocated = plan.get("allocated_bytes")
        unallocated = plan.get("unallocated_bytes")
        if (
            isinstance(allocated, bool)
            or not isinstance(allocated, int)
            or allocated < 0
            or allocated > EXPECTED_RUNTIME["memory_limit_bytes"]
        ):
            errors.append(f"{label} has an invalid fixed memory plan allocation")
        if (
            isinstance(unallocated, bool)
            or not isinstance(unallocated, int)
            or unallocated < 0
        ):
            errors.append(f"{label} has an invalid fixed memory plan remainder")
        if (
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (fixed, persistent, allocated)
            )
            and fixed + persistent != allocated
        ):
            errors.append(
                f"{label} memory plan violates fixed + persistent == allocated"
            )
        total = plan.get("total_limit_bytes")
        if (
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (allocated, unallocated, total)
            )
            and allocated + unallocated != total
        ):
            errors.append(
                f"{label} memory plan violates allocated + unallocated == total"
            )
        if (
            integrity.get("valid") is not True
            or integrity.get("model_key") != MODEL_KEYS[lane]
        ):
            errors.append(f"{label} manifest integrity gate failed")
        checked_records = integrity.get("checked_records")
        if checked_records != 19_200:
            errors.append(f"{label} did not verify all 19200 expert records")
        _append_expected_error(
            errors, f"{label} MLX memory cap", cap.get("applied"), True
        )
        _append_expected_error(
            errors, f"{label} MLX memory cap limit", cap.get("limit"), 144 * GIB
        )
    return slots


def _resource_summary(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    errors: list[str],
) -> dict[str, Any]:
    per_lane: dict[str, dict[str, list[float] | list[int]]] = {
        lane: {"diagnostic_tps": [], "bytes": [], "gib_s": [], "miss_wait": []}
        for lane in MODEL_KEYS
    }
    pair_values: list[tuple[str, float]] = []
    gaps: set[str] = set()
    for index, (lane, _payload, run) in enumerate(entries):
        label = f"resource[{index + 1}]"
        telemetry = _mapping(run.get("resource_telemetry"))
        if not telemetry:
            continue
        _append_expected_error(
            errors,
            f"{label} telemetry schema",
            telemetry.get("schema"),
            "mtplx-resource-telemetry-v2",
        )
        nonfinite_paths = _nonfinite_numeric_paths(telemetry, path="resource_telemetry")
        if nonfinite_paths:
            errors.append(
                f"{label} has non-finite resource telemetry at "
                + ", ".join(nonfinite_paths)
            )
        interval_count = telemetry.get("interval_count")
        sample_count = telemetry.get("sample_count")
        elapsed = _finite_number(telemetry.get("elapsed_seconds"))
        if (
            isinstance(interval_count, bool)
            or not isinstance(interval_count, int)
            or interval_count <= 0
        ):
            errors.append(f"{label} interval_count must be a positive integer")
        if elapsed is None or elapsed <= 0:
            errors.append(f"{label} elapsed_seconds must be positive and finite")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 2
        ):
            errors.append(f"{label} sample_count must be at least two")
        elif interval_count != sample_count - 1:
            errors.append(f"{label} interval_count must equal sample_count minus one")
        if telemetry.get("samples_dropped") != 0:
            errors.append(f"{label} samples_dropped must be zero")
        if (
            telemetry.get("sampling_failures") != 0
            or telemetry.get("sampling_failure") is not None
        ):
            errors.append(f"{label} resource sampling failure was reported")
        throughput = _mapping(telemetry.get("throughput"))
        storage = _mapping(telemetry.get("storage"))
        pipeline = _mapping(telemetry.get("expert_pipeline"))
        coverage = _mapping(telemetry.get("coverage"))
        pipeline_coverage = _mapping(pipeline.get("coverage"))
        if not coverage or not pipeline_coverage:
            errors.append(
                f"{label} is missing required telemetry coverage declarations"
            )
        missing_global = EXPECTED_GLOBAL_COVERAGE - set(coverage)
        missing_pipeline = EXPECTED_PIPELINE_COVERAGE - set(pipeline_coverage)
        if missing_global or missing_pipeline:
            errors.append(
                f"{label} missing coverage keys: "
                f"global={sorted(missing_global)}, "
                f"expert_pipeline={sorted(missing_pipeline)}"
            )
        if coverage.get("timeline") != "complete":
            errors.append(f"{label} telemetry timeline coverage is not complete")

        run_completion = run.get("completion_tokens")
        for field in ("completion_tokens", "final_completion_tokens"):
            if throughput.get(field) != run_completion:
                errors.append(
                    f"{label} telemetry completion token count {field} "
                    "does not match the run"
                )

        diagnostic_tps = _finite_number(throughput.get("completion_tokens_per_second"))
        read_bytes = _finite_number(storage.get("reader_read_bytes"))
        read_gib_s = _finite_number(storage.get("mean_gib_per_second"))
        miss_wait = _finite_number(
            pipeline.get("potentially_blocking_next_miss_fraction")
        )
        for metric, value in (
            ("diagnostic throughput", diagnostic_tps),
            ("reader_read_bytes", read_bytes),
            ("mean_gib_per_second", read_gib_s),
            ("potentially_blocking_next_miss_fraction", miss_wait),
        ):
            if value is None or value < 0:
                errors.append(f"{label} has invalid {metric}")
        if miss_wait is not None and miss_wait > 1:
            errors.append(f"{label} miss-wait fraction exceeds one")
        if coverage.get("storage_reads") != "uncached_reader_bytes":
            errors.append(f"{label} does not prove uncached physical reader bytes")
        if storage.get("io_cache_modes") != ["f-nocache"]:
            errors.append(f"{label} io_cache_modes must be exactly ['f-nocache']")
        if (
            pipeline_coverage.get("potentially_blocking_next_miss_step")
            != "measured_upper_bound"
        ):
            errors.append(f"{label} does not cover next-miss wait attribution")

        if all(
            value is not None
            for value in (diagnostic_tps, read_bytes, read_gib_s, miss_wait)
        ):
            lane_values = per_lane[lane]
            lane_values["diagnostic_tps"].append(diagnostic_tps)  # type: ignore[union-attr]
            lane_values["bytes"].append(int(read_bytes))  # type: ignore[union-attr]
            lane_values["gib_s"].append(read_gib_s)  # type: ignore[union-attr]
            lane_values["miss_wait"].append(miss_wait)  # type: ignore[union-attr]
            pair_values.append((lane, diagnostic_tps))
        gaps.update(_coverage_gaps(telemetry))

    result: dict[str, Any] = {}
    for lane, values in per_lane.items():
        diagnostic_tps = values["diagnostic_tps"]
        read_bytes = values["bytes"]
        gib_s = values["gib_s"]
        miss_wait = values["miss_wait"]
        result[lane] = {
            "diagnostic_samples": len(diagnostic_tps),
            "mean_diagnostic_completion_tokens_per_second": (
                statistics.fmean(diagnostic_tps) if diagnostic_tps else None
            ),
            "median_diagnostic_completion_tokens_per_second": (
                statistics.median(diagnostic_tps) if diagnostic_tps else None
            ),
            "reader_read_bytes_total": sum(read_bytes),
            "mean_reader_read_bytes": statistics.fmean(read_bytes)
            if read_bytes
            else None,
            "mean_reader_gib_per_second": statistics.fmean(gib_s) if gib_s else None,
            "median_reader_gib_per_second": statistics.median(gib_s) if gib_s else None,
            "mean_potentially_blocking_next_miss_fraction": (
                statistics.fmean(miss_wait) if miss_wait else None
            ),
            "median_potentially_blocking_next_miss_fraction": (
                statistics.median(miss_wait) if miss_wait else None
            ),
        }
    if (
        len(pair_values) == len(RESOURCE_ORDER)
        and tuple(lane for lane, _value in pair_values) == RESOURCE_ORDER
    ):
        result["pairs"], result["order_splits"] = _pair_summary(pair_values)
    else:
        result["pairs"] = {"samples": 0, "percent_changes": []}
        result["order_splits"] = {}
    gaps.add("swap_pressure=unavailable")
    result["coverage_gaps"] = sorted(gaps)
    result["swap_pressure"] = {
        "status": "unavailable",
        "gate_passed": False,
        "reason": "current benchmark payload does not measure swap pressure",
    }
    errors.append(
        "resource campaign swap-pressure evidence unavailable; "
        "the current benchmark payload cannot prove a pressure-free run"
    )
    return result


def _headline_summary(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    lane_values: dict[str, list[float]] = {"q4": [], "q2": []}
    pair_values: list[tuple[str, float]] = []
    output_entries: list[tuple[str, Mapping[str, Any]]] = []
    for lane, _payload, run in entries:
        tps = _finite_number(run.get("completion_tokens_per_second"))
        if tps is None or tps <= 0:
            continue
        lane_values[lane].append(tps)
        pair_values.append((lane, tps))
        output_entries.append((lane, run))
    q4 = _stats(lane_values["q4"], sample_label="completion_tokens_per_second")
    q2 = _stats(lane_values["q2"], sample_label="completion_tokens_per_second")
    result: dict[str, Any] = {"q4": q4, "q2": q2}
    q4_mean = q4["mean_completion_tokens_per_second"]
    q2_mean = q2["mean_completion_tokens_per_second"]
    q4_median = q4["median_completion_tokens_per_second"]
    q2_median = q2["median_completion_tokens_per_second"]
    result["pooled_mean_percent_change"] = (
        _percent_change(q4_mean, q2_mean) if q4_mean and q2_mean is not None else None
    )
    result["pooled_median_percent_change"] = (
        _percent_change(q4_median, q2_median)
        if q4_median and q2_median is not None
        else None
    )
    if (
        len(pair_values) == len(HEADLINE_ORDER)
        and tuple(lane for lane, _value in pair_values) == HEADLINE_ORDER
    ):
        result["pairs"], result["order_splits"] = _pair_summary(pair_values)
        result["output_divergence"] = _output_divergence(output_entries)
    else:
        result["pairs"] = {"samples": 0, "percent_changes": []}
        result["order_splits"] = {}
        result["output_divergence"] = {
            "pairs": 0,
            "token_divergent_pairs": 0,
            "text_divergent_pairs": 0,
            "details": [],
        }
    return result


def summarize_glm52_q2_campaign(
    resource_payloads: Sequence[dict],
    headline_payloads: Sequence[dict],
) -> dict[str, Any]:
    """Validate one fixed campaign and summarize diagnostic/headline evidence."""

    comparability_errors: list[str] = []
    resource_errors: list[str] = []
    resource_entries = _validate_payloads(
        resource_payloads,
        kind="resource",
        order=RESOURCE_ORDER,
        errors=comparability_errors,
    )
    headline_entries = _validate_payloads(
        headline_payloads,
        kind="headline",
        order=HEADLINE_ORDER,
        errors=comparability_errors,
    )
    all_entries = [*resource_entries, *headline_entries]
    _validate_configuration(all_entries, errors=comparability_errors)
    slots = _validate_run_gate(all_entries, errors=resource_errors)
    resource = _resource_summary(resource_entries, errors=resource_errors)
    headline = _headline_summary(headline_entries)

    slot_summary = {
        lane: next(iter(values)) if len(values) == 1 else None
        for lane, values in slots.items()
    }
    unique_comparability_errors = list(dict.fromkeys(comparability_errors))
    unique_resource_errors = list(dict.fromkeys(resource_errors))
    unique_errors = list(
        dict.fromkeys([*unique_comparability_errors, *unique_resource_errors])
    )
    valid = not unique_errors
    return {
        "schema": "mtplx-glm52-expert-q2-campaign-summary-v1",
        "valid": valid,
        "errors": unique_errors,
        "expected_order": {
            "resource": list(RESOURCE_ORDER),
            "headline": list(HEADLINE_ORDER),
        },
        "comparability": {
            "passed": not unique_comparability_errors,
            "errors": unique_comparability_errors,
            "normalization": [
                "run_and_derived_labels",
                "model_key",
                "model_artifact_except_harness_source",
                "resource_telemetry_presence",
            ],
        },
        "resource_gate": {
            "passed": not unique_comparability_errors and not unique_resource_errors,
            "errors": unique_resource_errors,
            "requires_uncached_reader_bytes": True,
            "requires_full_record_integrity": True,
            "requires_fixed_memory_plan": True,
            "requires_swap_pressure_evidence": True,
        },
        "cache_slots_per_layer": slot_summary,
        "resource": resource,
        "headline": headline,
    }


__all__ = [
    "HEADLINE_ORDER",
    "RESOURCE_ORDER",
    "summarize_glm52_q2_campaign",
]
