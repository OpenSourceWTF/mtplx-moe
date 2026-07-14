"""Strict evidence aggregation for the paired Hy3 expert-Q2 campaign."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory


GIB = 1024**3
RESOURCE_ORDER = ("q4", "q2", "q2", "q4")
HEADLINE_ORDER = ("q4", "q2", "q2", "q4", "q4", "q2", "q2", "q4")
MODEL_KEYS = {"q4": "hy3-expert-only-q4", "q2": "hy3-expert-q2"}
EXPECTED_CAPACITIES = {"q4": 7_821, "q2": 14_077}
EXPECTED_RUNTIME = {
    "memory_limit_bytes": 112 * GIB,
    "max_live_kv_tokens": 18_888,
    "runtime_reserve_bytes": 8 * GIB,
    "expert_cache_limit_bytes": 83_034_243_072,
    "cache_policy": "lru",
    "cache_scope": "global",
    "slot_layout": "component-banks",
    "max_read_chunk_bytes": 64 * 1024**2,
    "bypass_page_cache": True,
    "prefer_sidecar": True,
    "verify_record_hashes": False,
    "verify_sidecar_hash_at_open": False,
    "transient_slots": 32,
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
_OPERATIONAL_ARTIFACT_FIELDS = frozenset(
    {
        "verification_elapsed_seconds",
        "verification_method",
        "identity_method",
        "verification_level",
        "page_cache_bypassed",
        "receipt_reused",
        "receipt",
    }
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lane(payload: Mapping[str, Any]) -> str | None:
    model_key = payload.get("model_key")
    return next((lane for lane, key in MODEL_KEYS.items() if key == model_key), None)


def _nonfinite_paths(value: object, path: str) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, Mapping):
        return [
            nested
            for key, item in value.items()
            for nested in _nonfinite_paths(item, f"{path}.{key}")
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            nested
            for index, item in enumerate(value)
            for nested in _nonfinite_paths(item, f"{path}[{index}]")
        ]
    return []


def _append_expected(
    errors: list[str], label: str, actual: object, expected: object
) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _stable_artifact(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _stable_artifact(item)
            for key, item in value.items()
            if key not in _OPERATIONAL_ARTIFACT_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_artifact(item) for item in value]
    return value


def _normalized_configuration(summary: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(summary))
    for key in ("run_label", "configuration_label", "configuration_fingerprint"):
        normalized.pop(key, None)
    settings = normalized.get("performance_settings")
    if isinstance(settings, dict):
        runtime = settings.get("runtime_config")
        if isinstance(runtime, dict):
            runtime.pop("model_key", None)
            runtime.pop("resource_telemetry", None)
        artifact = settings.get("model_artifact")
        if isinstance(artifact, dict):
            settings["model_artifact"] = {
                "harness_source": deepcopy(artifact.get("harness_source"))
            }
    return normalized


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "mean_completion_tokens_per_second": None,
            "median_completion_tokens_per_second": None,
            "min_completion_tokens_per_second": None,
            "max_completion_tokens_per_second": None,
            "range_completion_tokens_per_second": None,
        }
    low, high = min(values), max(values)
    return {
        "samples": len(values),
        "mean_completion_tokens_per_second": statistics.fmean(values),
        "median_completion_tokens_per_second": statistics.median(values),
        "min_completion_tokens_per_second": low,
        "max_completion_tokens_per_second": high,
        "range_completion_tokens_per_second": high - low,
    }


def _pair_summary(values: Sequence[tuple[str, float]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    deltas: list[float] = []
    order_splits: dict[str, list[float]] = {"q4_first": [], "q2_first": []}
    for offset in range(0, len(values), 2):
        first_lane, first_value = values[offset]
        second_lane, second_value = values[offset + 1]
        by_lane = {first_lane: first_value, second_lane: second_value}
        delta = (by_lane["q2"] / by_lane["q4"] - 1.0) * 100.0
        deltas.append(delta)
        order = f"{first_lane}_first"
        order_splits[order].append(delta)
        details.append(
            {
                "pair": offset // 2 + 1,
                "order": [first_lane, second_lane],
                "q4": by_lane["q4"],
                "q2": by_lane["q2"],
                "percent_change": delta,
            }
        )
    return {
        "samples": len(deltas),
        "percent_changes": deltas,
        "mean_percent_change": statistics.fmean(deltas) if deltas else None,
        "median_percent_change": statistics.median(deltas) if deltas else None,
        "min_percent_change": min(deltas) if deltas else None,
        "max_percent_change": max(deltas) if deltas else None,
        "range_percent_change": max(deltas) - min(deltas) if deltas else None,
        "all_positive": bool(deltas) and all(delta > 0 for delta in deltas),
        "order_splits": {
            name: {
                "samples": len(items),
                "mean_percent_change": statistics.fmean(items) if items else None,
                "median_percent_change": statistics.median(items) if items else None,
                "percent_changes": items,
            }
            for name, items in order_splits.items()
        },
        "details": details,
    }


def _validate_payloads(
    payloads: Sequence[dict],
    *,
    kind: str,
    order: Sequence[str],
    comparability_errors: list[str],
    gate_errors: list[str],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    if len(payloads) != len(order):
        comparability_errors.append(
            f"{kind} campaign requires exactly {len(order)} payloads; got {len(payloads)}"
        )
    entries: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for index, raw_payload in enumerate(payloads):
        label = f"{kind}[{index + 1}]"
        if not isinstance(raw_payload, Mapping):
            comparability_errors.append(f"{label} must be an object")
            continue
        payload = raw_payload
        lane = _lane(payload)
        expected_lane = order[index] if index < len(order) else None
        if lane is None:
            comparability_errors.append(f"{label} has an unsupported model_key")
            continue
        if lane != expected_lane:
            comparability_errors.append(
                f"{kind} order mismatch at position {index + 1}: "
                f"expected {expected_lane}, got {lane}"
            )
        expected_label = f"hy3-q2-{kind}-{index + 1}-{expected_lane}"
        _append_expected(
            comparability_errors,
            f"{label} run label",
            payload.get("run_label"),
            expected_label,
        )
        _append_expected(
            comparability_errors,
            f"{label} schema",
            payload.get("schema"),
            "mtplx-streamed-generation-benchmark-v1",
        )
        runs = _sequence(payload.get("runs"))
        if len(runs) != 1:
            comparability_errors.append(
                f"{label} requires exactly one run; got {len(runs)}"
            )
            continue
        run = _mapping(runs[0])
        if not run:
            comparability_errors.append(f"{label} run must be an object")
            continue
        _append_expected(
            comparability_errors,
            f"{label} nested run label",
            run.get("run_label"),
            expected_label,
        )
        summary = _mapping(payload.get("configuration_summary"))
        configuration_label = summary.get("configuration_label")
        if (
            not isinstance(configuration_label, str)
            or payload.get("configuration_label") != configuration_label
            or run.get("configuration_label") != configuration_label
        ):
            comparability_errors.append(
                f"{label} configuration_label differs across payload, summary, and run"
            )
        for where, value in (
            ("payload", payload.get("execution_lane")),
            ("configuration", summary.get("execution_lane")),
            ("run", run.get("execution_lane")),
        ):
            _append_expected(
                comparability_errors,
                f"{label} {where} execution lane must be reference-ar",
                value,
                "reference-ar",
            )
        for field, expected in (
            ("cache_scope", "global"),
            ("slot_layout", "component-banks"),
            ("concurrency", 1),
            ("requested_concurrency", 1),
            ("achieved_peak_concurrency", 1),
            ("saturation_valid", True),
            ("undersubscribed", False),
        ):
            _append_expected(
                comparability_errors, f"{label} run {field}", run.get(field), expected
            )
        if payload.get("error") or run.get("error"):
            gate_errors.append(f"{label} reports an operational error")
        finish_reason = run.get("finish_reason")
        if not isinstance(finish_reason, str) or any(
            marker in finish_reason.lower()
            for marker in ("error", "crash", "exception", "failed")
        ):
            gate_errors.append(f"{label} has invalid finish_reason {finish_reason!r}")
        tps = _finite(run.get("completion_tokens_per_second"))
        if tps is None or tps <= 0:
            gate_errors.append(
                f"{label} completion_tokens_per_second must be positive and finite"
            )
        token_ids = run.get("token_ids")
        completion_tokens = run.get("completion_tokens")
        valid_tokens = isinstance(token_ids, list) and all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in token_ids
        )
        if not valid_tokens:
            gate_errors.append(f"{label} token_ids must be a list of integers")
        if completion_tokens != 512 or not valid_tokens or len(token_ids) != 512:
            gate_errors.append(f"{label} must contain exactly 512 completion tokens")
        if not isinstance(run.get("text"), str):
            gate_errors.append(f"{label} must contain completion text")
        telemetry = run.get("resource_telemetry")
        if kind == "resource":
            if not isinstance(telemetry, Mapping):
                gate_errors.append(f"{label} is missing resource_telemetry")
            if run.get("diagnostic_run") is not True:
                gate_errors.append(f"{label} must be declared diagnostic_run")
        elif telemetry is not None or run.get("diagnostic_run"):
            gate_errors.append(
                f"{label} headline run must not contain resource telemetry"
            )
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
        options = _mapping(settings.get("prompt_options"))
        mtp = _mapping(settings.get("mtp"))
        payload_mtp = _mapping(payload.get("mtp"))
        top_generation = _mapping(payload.get("generation"))
        if not summary or not settings:
            errors.append(f"{label} is missing configuration_summary")
            continue
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

        _append_expected(
            errors, f"{label} model_key", runtime.get("model_key"), MODEL_KEYS[lane]
        )
        _append_expected(
            errors,
            f"{label} runtime_config.resource_telemetry",
            runtime.get("resource_telemetry"),
            run.get("diagnostic_run") is True,
        )
        _append_expected(
            errors,
            f"{label} scheduler execution lane must be reference-ar",
            scheduler.get("execution_lane"),
            "reference-ar",
        )
        for field, expected in (
            ("cache_scope", "global"),
            ("slot_layout", "component-banks"),
            ("concurrency", 1),
            ("requested_concurrency", 1),
        ):
            _append_expected(
                errors, f"{label} configuration {field}", summary.get(field), expected
            )
        for field, expected in (
            ("requested_concurrency", 1),
            ("max_prefills_per_step", 1),
            ("workload_shape", "static"),
        ):
            _append_expected(
                errors, f"{label} scheduler {field}", scheduler.get(field), expected
            )
        for field, expected in EXPECTED_RUNTIME.items():
            _append_expected(
                errors, f"{label} runtime_config.{field}", runtime.get(field), expected
            )
        for field, actual, expected in (
            ("MTP enabled", mtp.get("enabled"), False),
            ("payload MTP enabled", payload_mtp.get("enabled"), False),
            ("thinking enabled", options.get("enable_thinking"), False),
            ("payload thinking enabled", payload.get("enable_thinking"), False),
            ("chat", options.get("chat"), True),
            ("payload chat", payload.get("chat"), True),
            ("seed", settings.get("seed"), 0),
            (
                "generation_profile",
                generation.get("generation_profile"),
                "deterministic",
            ),
            ("max_tokens", generation.get("max_tokens"), 512),
            ("repeats", generation.get("repeats"), 1),
            ("reset_between", generation.get("reset_between"), True),
            ("window_telemetry", generation.get("window_telemetry"), False),
            ("temperature", sampler.get("temperature"), 0.0),
            ("top_p", sampler.get("top_p"), 1.0),
            ("top_k", sampler.get("top_k"), 1),
            ("payload seed", payload.get("seed"), 0),
            (
                "payload generation_profile",
                payload.get("generation_profile"),
                "deterministic",
            ),
            ("payload max_tokens", top_generation.get("max_tokens"), 512),
            ("payload temperature", top_generation.get("temperature"), 0.0),
            ("payload top_p", top_generation.get("top_p"), 1.0),
            ("payload top_k", top_generation.get("top_k"), 1),
            ("payload reset_between", payload.get("reset_between"), True),
            ("payload workload_shape", payload.get("workload_shape"), "static"),
            ("payload cache_scope", payload.get("cache_scope"), "global"),
            ("payload slot_layout", payload.get("slot_layout"), "component-banks"),
        ):
            _append_expected(errors, f"{label} {field}", actual, expected)


def _expected_plan(lane: str) -> dict[str, Any]:
    plan = plan_expert_memory(
        get_model_spec(MODEL_KEYS[lane]),
        total_limit_bytes=EXPECTED_RUNTIME["memory_limit_bytes"],
        context_tokens=EXPECTED_RUNTIME["max_live_kv_tokens"],
        runtime_reserve_bytes=EXPECTED_RUNTIME["runtime_reserve_bytes"],
        expert_cache_limit_bytes=EXPECTED_RUNTIME["expert_cache_limit_bytes"],
        transient_slots=EXPECTED_RUNTIME["transient_slots"],
        io_staging_bytes=EXPECTED_RUNTIME["io_staging_bytes"],
        execution_workspace_bytes=EXPECTED_RUNTIME["execution_workspace_bytes"],
        cache_scope="global",
    )
    payload = asdict(plan)
    payload["fixed_bytes"] = plan.fixed_bytes
    payload["allocated_bytes"] = plan.allocated_bytes
    return payload


def _validate_run_integrity(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    errors: list[str],
) -> None:
    for index, (lane, _payload, run) in enumerate(entries):
        label = f"run[{index + 1}]"
        after = _mapping(run.get("streaming_after"))
        plan = _mapping(after.get("memory_plan"))
        expected = _expected_plan(lane)
        for field, value in expected.items():
            _append_expected(
                errors, f"{label} memory plan {field}", plan.get(field), value
            )
        _append_expected(
            errors,
            f"{label} persistent_slots",
            plan.get("persistent_slots"),
            EXPECTED_CAPACITIES[lane],
        )
        integrity = _mapping(after.get("integrity"))
        for field, value in (
            ("valid", True),
            ("model_key", MODEL_KEYS[lane]),
            ("checked_shards", 19),
            ("checked_records", 15_168),
        ):
            _append_expected(
                errors,
                f"{label} manifest integrity {field}",
                integrity.get(field),
                value,
            )
        cap = _mapping(after.get("memory_cap"))
        _append_expected(
            errors, f"{label} MLX memory cap applied", cap.get("applied"), True
        )
        _append_expected(
            errors,
            f"{label} MLX memory cap limit",
            cap.get("limit"),
            EXPECTED_RUNTIME["memory_limit_bytes"]
            - EXPECTED_RUNTIME["runtime_reserve_bytes"],
        )


def _small_file_map(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(artifact.get("small_files")):
        item = _mapping(raw)
        name = item.get("name")
        if isinstance(name, str):
            result[name] = item
    return result


def _resident_shards(artifact: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("shard"))
        for raw in _sequence(artifact.get("resident_tensors"))
        if (item := _mapping(raw)) and isinstance(item.get("shard"), str)
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_flat_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _resident_range_map(
    values: object,
    *,
    label: str,
    errors: list[str],
) -> dict[tuple[str, str, int, int], str]:
    result: dict[tuple[str, str, int, int], str] = {}
    rows = _sequence(values)
    if not rows:
        errors.append(f"{label} has no authenticated resident ranges")
        return result
    for index, raw in enumerate(rows):
        item = _mapping(raw)
        tensor = item.get("tensor")
        shard = item.get("shard")
        offset = item.get("offset")
        length = item.get("length")
        digest = item.get("sha256")
        row_label = f"{label}[{index + 1}]"
        if not isinstance(tensor, str) or not tensor:
            errors.append(f"{row_label} has an invalid tensor name")
            continue
        if not isinstance(shard, str) or not shard:
            errors.append(f"{row_label} has an invalid shard name")
            continue
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            errors.append(f"{row_label} has an invalid offset")
            continue
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            errors.append(f"{row_label} has an invalid length")
            continue
        if item.get("size", length) != length:
            errors.append(f"{row_label} size does not match its range length")
        if not _valid_sha256(digest):
            errors.append(f"{row_label} has an invalid SHA-256")
            continue
        key = (tensor, shard, offset, length)
        if key in result:
            errors.append(f"{row_label} repeats resident range {key!r}")
            continue
        result[key] = digest
    return result


def _validate_artifact_binding(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    quality: Mapping[str, Any],
    *,
    errors: list[str],
) -> dict[str, Any]:
    quality_lanes = _mapping(quality.get("lanes"))
    quality_range_maps: dict[str, dict[tuple[str, str, int, int], str]] = {}
    quality_shards: dict[str, set[str]] = {}
    for lane in MODEL_KEYS:
        quality_lane = _mapping(quality_lanes.get(lane))
        quality_artifact = _mapping(quality_lane.get("artifact"))
        residents = _mapping(quality_artifact.get("residents"))
        _append_expected(
            errors,
            f"quality {lane} resident file algorithm",
            residents.get("algorithm"),
            "sha256-name-size-and-verified-file-digest-v1",
        )
        _append_expected(
            errors,
            f"quality {lane} resident range algorithm",
            residents.get("range_algorithm"),
            "sha256-tensor-shard-offset-length-and-content-v1",
        )
        file_names: set[str] = set()
        file_receipts: list[tuple[str, int, str]] = []
        files = _sequence(residents.get("files"))
        if not files:
            errors.append(f"quality {lane} has no authenticated resident files")
        for index, raw in enumerate(files):
            item = _mapping(raw)
            name = item.get("name")
            size = item.get("bytes")
            actual = item.get("sha256")
            declared = item.get("declared_sha256")
            label = f"quality {lane} resident file[{index + 1}]"
            if not _valid_flat_name(name) or name in file_names:
                errors.append(f"{label} has an invalid or repeated name")
                continue
            file_names.add(name)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                errors.append(f"{label} has invalid bytes")
            if item.get("page_cache_bypassed") is not True:
                errors.append(f"{label} did not prove F_NOCACHE hashing")
            if not _valid_sha256(actual) or not _valid_sha256(declared):
                errors.append(f"{label} has an invalid actual or declared SHA-256")
            elif actual != declared:
                errors.append(f"{label} actual SHA-256 does not match declared SHA-256")
            if (
                isinstance(size, int)
                and not isinstance(size, bool)
                and size > 0
                and _valid_sha256(actual)
            ):
                file_receipts.append((name, size, actual))
        composite = hashlib.sha256()
        for name, size, actual in sorted(file_receipts):
            encoded_name = name.encode("utf-8")
            composite.update(len(encoded_name).to_bytes(4, "big"))
            composite.update(encoded_name)
            composite.update(size.to_bytes(8, "big"))
            composite.update(bytes.fromhex(actual))
        declared_composite = residents.get("sha256")
        if not _valid_sha256(declared_composite):
            errors.append(f"quality {lane} resident composite has an invalid SHA-256")
        elif (
            len(file_receipts) != len(files)
            or composite.hexdigest() != declared_composite
        ):
            errors.append(
                f"quality {lane} resident composite does not match file receipts"
            )
        quality_shards[lane] = file_names
        quality_range_maps[lane] = _resident_range_map(
            residents.get("ranges"),
            label=f"quality {lane} resident ranges",
            errors=errors,
        )
        range_shards = {key[1] for key in quality_range_maps[lane]}
        if range_shards != file_names:
            errors.append(
                f"quality {lane} resident range and file shard inventories differ"
            )
    lane_references: dict[str, str] = {}
    harness_reference: str | None = None
    resident_reference: str | None = None
    checked = 0
    for index, (lane, payload, _run) in enumerate(entries):
        label = f"artifact[{index + 1}]"
        artifact = _mapping(_mapping(payload.get("artifact_verification")).get("model"))
        config_artifact = _mapping(
            _mapping(
                _mapping(payload.get("configuration_summary")).get(
                    "performance_settings"
                )
            ).get("model_artifact")
        )
        if not artifact:
            errors.append(f"{label} is missing full model artifact verification")
            continue
        checked += 1
        _append_expected(
            errors,
            f"{label} method",
            artifact.get("method"),
            "manifest_plus_executable_resident_content_v1",
        )
        _append_expected(
            errors,
            f"{label} verification level",
            artifact.get("verification_level"),
            "full_small_files_and_authenticated_resident_content",
        )
        expert = _mapping(artifact.get("expert_payload"))
        for field, value in (
            ("method", "verified_sidecar_sha256"),
            ("verification_level", "actual_full_file_digest_matches_manifest"),
            ("page_cache_bypassed", True),
        ):
            _append_expected(
                errors, f"{label} expert payload {field}", expert.get(field), value
            )
        if not isinstance(expert.get("sha256"), str) or len(expert["sha256"]) != 64:
            errors.append(f"{label} expert payload lacks an actual SHA-256")
        if not isinstance(expert.get("size"), int) or expert.get("size", 0) <= 0:
            errors.append(f"{label} expert payload lacks an actual size")
        if _stable_artifact(artifact) != config_artifact:
            errors.append(f"{label} full identity does not bind performance settings")

        identity = _canonical(_stable_artifact(artifact))
        previous = lane_references.setdefault(lane, identity)
        if identity != previous:
            errors.append(f"{label} {lane} model artifact changed across campaign runs")
        harness = _canonical(_mapping(artifact.get("harness_source")))
        if harness_reference is None:
            harness_reference = harness
        elif harness != harness_reference:
            errors.append(
                f"{label} harness source identity differs across campaign runs"
            )
        residents = _canonical(_stable_artifact(artifact.get("resident_tensors")))
        if resident_reference is None:
            resident_reference = residents
        elif residents != resident_reference:
            errors.append(f"{label} resident tensor bytes differ between Q4 and Q2")

        quality_lane = _mapping(quality_lanes.get(lane))
        quality_manifest = _mapping(quality_lane.get("manifest"))
        quality_artifact = _mapping(quality_lane.get("artifact"))
        manifest = _mapping(artifact.get("manifest"))
        _append_expected(
            errors,
            f"{label} artifact manifest model_key",
            manifest.get("model_key"),
            MODEL_KEYS[lane],
        )
        spec = get_model_spec(MODEL_KEYS[lane])
        _append_expected(
            errors,
            f"{label} artifact source revision",
            manifest.get("source_revision"),
            spec.source_revision,
        )
        _append_expected(
            errors,
            f"{label} artifact source repository",
            manifest.get("source_repo"),
            spec.source_model,
        )
        _append_expected(
            errors,
            f"{label} quality manifest file SHA-256",
            quality_manifest.get("file_sha256"),
            manifest.get("content_sha256"),
        )
        _append_expected(
            errors,
            f"{label} quality artifact manifest SHA-256",
            quality_artifact.get("manifest_file_sha256"),
            manifest.get("content_sha256"),
        )
        _append_expected(
            errors,
            f"{label} quality declared manifest SHA-256",
            quality_manifest.get("declared_sha256"),
            manifest.get("declared_manifest_sha256"),
        )
        files = _small_file_map(artifact)
        index = files.get("model.safetensors.index.json", {})
        _append_expected(
            errors,
            f"{label} quality index SHA-256",
            _mapping(quality_artifact.get("index")).get("sha256"),
            index.get("sha256"),
        )
        for raw in _sequence(_mapping(quality_lane.get("tokenizer")).get("files")):
            item = _mapping(raw)
            name = item.get("name")
            benchmark_file = files.get(str(name), {})
            _append_expected(
                errors,
                f"{label} tokenizer {name} SHA-256",
                item.get("sha256"),
                benchmark_file.get("sha256"),
            )
        if quality_shards[lane] != _resident_shards(artifact):
            errors.append(f"{label} quality resident shard inventory mismatch")
        benchmark_ranges = _resident_range_map(
            artifact.get("resident_tensors"),
            label=f"{label} benchmark resident ranges",
            errors=errors,
        )
        if benchmark_ranges != quality_range_maps[lane]:
            errors.append(
                f"{label} benchmark resident range content does not match "
                f"the quality receipt for {lane}"
            )
    return {"passed": not errors, "checked_artifacts": checked, "errors": list(errors)}


def _validate_quality(quality: Mapping[str, Any], *, errors: list[str]) -> None:
    _append_expected(
        errors, "quality schema", quality.get("schema"), "mtplx-streamed-quality-v1"
    )
    for field, expected in (
        ("passed", True),
        ("quality_passed", True),
        ("finite", True),
        ("nan_count", 0),
        ("nonfinite_count", 0),
        ("errors", []),
    ):
        _append_expected(errors, f"quality {field}", quality.get(field), expected)
    regression = _finite(quality.get("relative_perplexity_regression"))
    threshold = _finite(quality.get("max_relative_perplexity_regression"))
    if regression is None or regression > 0.05:
        errors.append(
            "quality relative perplexity regression must be finite and <= 0.05"
        )
    if threshold is None or threshold > 0.05:
        errors.append("quality threshold must be finite and <= 0.05")
    lanes = _mapping(quality.get("lanes"))
    for lane, model_key in MODEL_KEYS.items():
        lane_payload = _mapping(lanes.get(lane))
        _append_expected(
            errors,
            f"quality {lane} model_key",
            lane_payload.get("model_key"),
            model_key,
        )
        if _mapping(lane_payload.get("loss")).get("finite") is not True:
            errors.append(f"quality {lane} loss is not finite")
    q4_residents = _mapping(
        _mapping(_mapping(lanes.get("q4")).get("artifact")).get("residents")
    ).get("sha256")
    q2_residents = _mapping(
        _mapping(_mapping(lanes.get("q2")).get("artifact")).get("residents")
    ).get("sha256")
    if not isinstance(q4_residents, str) or q4_residents != q2_residents:
        errors.append("quality Q4/Q2 resident artifact identity mismatch")


def _resource_summary(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    errors: list[str],
) -> dict[str, Any]:
    lane_values: dict[str, dict[str, list[float]]] = {
        lane: {
            "tps": [],
            "bytes": [],
            "tokens": [],
            "gib_s": [],
            "miss_wait": [],
            "hits": [],
            "misses": [],
            "evictions": [],
            "fences": [],
            "active_fraction": [],
            "queued_fraction": [],
        }
        for lane in MODEL_KEYS
    }
    pair_values: list[tuple[str, float]] = []
    coverage_gaps: set[str] = set()
    for index, (lane, _payload, run) in enumerate(entries):
        label = f"resource[{index + 1}]"
        telemetry = _mapping(run.get("resource_telemetry"))
        if not telemetry:
            continue
        _append_expected(
            errors,
            f"{label} telemetry schema",
            telemetry.get("schema"),
            "mtplx-resource-telemetry-v2",
        )
        paths = _nonfinite_paths(telemetry, "resource_telemetry")
        if paths:
            errors.append(f"{label} has non-finite telemetry at {', '.join(paths)}")
        sample_count = telemetry.get("sample_count")
        interval_count = telemetry.get("interval_count")
        elapsed = _finite(telemetry.get("elapsed_seconds"))
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 2
        ):
            errors.append(f"{label} sample_count must be at least two")
        if (
            not isinstance(interval_count, int)
            or isinstance(interval_count, bool)
            or interval_count <= 0
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or interval_count != sample_count - 1
        ):
            errors.append(f"{label} interval_count must equal sample_count minus one")
        if elapsed is None or elapsed <= 0:
            errors.append(f"{label} elapsed_seconds must be positive and finite")
        if telemetry.get("samples_dropped") != 0:
            errors.append(f"{label} samples_dropped must be zero")
        if (
            telemetry.get("sampling_failures") != 0
            or telemetry.get("sampling_failure") is not None
        ):
            errors.append(f"{label} resource sampling failure was reported")

        throughput = _mapping(telemetry.get("throughput"))
        storage = _mapping(telemetry.get("storage"))
        reader_pool = _mapping(telemetry.get("reader_pool"))
        fences = _mapping(telemetry.get("completion_fences"))
        pipeline = _mapping(telemetry.get("expert_pipeline"))
        coverage = _mapping(telemetry.get("coverage"))
        pipeline_coverage = _mapping(pipeline.get("coverage"))
        missing_global = EXPECTED_GLOBAL_COVERAGE - set(coverage)
        missing_pipeline = EXPECTED_PIPELINE_COVERAGE - set(pipeline_coverage)
        if missing_global or missing_pipeline:
            errors.append(
                f"{label} missing coverage keys: global={sorted(missing_global)}, "
                f"expert_pipeline={sorted(missing_pipeline)}"
            )
        if coverage.get("timeline") != "complete":
            errors.append(f"{label} telemetry timeline coverage is not complete")
        for key, status in coverage.items():
            if status not in {
                "measured",
                "complete",
                "supplied",
                "uncached_reader_bytes",
            }:
                coverage_gaps.add(f"coverage.{key}={status}")
        for key, status in pipeline_coverage.items():
            if not str(status).startswith("measured"):
                coverage_gaps.add(f"expert_pipeline.coverage.{key}={status}")
        if storage.get("io_cache_modes") != ["f-nocache"]:
            errors.append(f"{label} io_cache_modes must be exactly ['f-nocache']")
        if coverage.get("storage_reads") != "uncached_reader_bytes":
            errors.append(
                f"{label} storage read coverage must prove uncached reader bytes"
            )
        if (
            pipeline_coverage.get("potentially_blocking_next_miss_step")
            != "measured_upper_bound"
        ):
            errors.append(f"{label} coverage lacks next-miss wait attribution")
        for field in ("completion_tokens", "final_completion_tokens"):
            if throughput.get(field) != run.get("completion_tokens"):
                errors.append(f"{label} telemetry {field} does not match the run")

        cache_totals = {"hits": 0, "misses": 0, "evictions": 0, "bytes": 0}
        cache_by_layer = _mapping(telemetry.get("cache_by_layer"))
        if not cache_by_layer:
            errors.append(f"{label} cache_by_layer telemetry is missing")
        for raw in cache_by_layer.values():
            row = _mapping(raw)
            for target, source in (
                ("hits", "expert_hits"),
                ("misses", "expert_misses"),
                ("evictions", "evictions"),
                ("bytes", "bytes_read"),
            ):
                value = row.get(source)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"{label} cache_by_layer {source} is invalid")
                else:
                    cache_totals[target] += value
        storage_bytes = _finite(storage.get("reader_read_bytes"))
        if storage_bytes is not None and cache_totals["bytes"] != storage_bytes:
            errors.append(f"{label} cache bytes do not match storage reader_read_bytes")
        metric_values = {
            "tps": _finite(throughput.get("completion_tokens_per_second")),
            "bytes": storage_bytes,
            "tokens": _finite(run.get("completion_tokens")),
            "gib_s": _finite(storage.get("mean_gib_per_second")),
            "miss_wait": _finite(
                pipeline.get("potentially_blocking_next_miss_fraction")
            ),
            "fences": _finite(fences.get("synchronous_fences")),
            "active_fraction": _finite(reader_pool.get("active_capacity_fraction")),
            "queued_fraction": _finite(reader_pool.get("queue_nonempty_fraction")),
        }
        for name, value in metric_values.items():
            if value is None or value < 0:
                errors.append(f"{label} has invalid {name} telemetry")
        if metric_values["miss_wait"] is not None and metric_values["miss_wait"] > 1:
            errors.append(f"{label} miss-wait fraction exceeds one")
        if all(value is not None for value in metric_values.values()):
            values = lane_values[lane]
            for name, value in metric_values.items():
                values[name].append(value)  # type: ignore[arg-type]
            values["hits"].append(float(cache_totals["hits"]))
            values["misses"].append(float(cache_totals["misses"]))
            values["evictions"].append(float(cache_totals["evictions"]))
            if metric_values["tps"] is not None:
                pair_values.append((lane, metric_values["tps"]))

    result: dict[str, Any] = {}
    for lane, values in lane_values.items():
        total_bytes = sum(values["bytes"])
        total_tokens = sum(values["tokens"])
        result[lane] = {
            "diagnostic_samples": len(values["tps"]),
            "mean_diagnostic_completion_tokens_per_second": (
                statistics.fmean(values["tps"]) if values["tps"] else None
            ),
            "expert_reader_bytes_total": total_bytes,
            "expert_bytes_per_token": total_bytes / total_tokens
            if total_tokens
            else None,
            "mean_reader_gib_per_second": (
                statistics.fmean(values["gib_s"]) if values["gib_s"] else None
            ),
            "mean_potentially_blocking_next_miss_fraction": (
                statistics.fmean(values["miss_wait"]) if values["miss_wait"] else None
            ),
            "cache_hits_total": int(sum(values["hits"])),
            "cache_misses_total": int(sum(values["misses"])),
            "cache_evictions_total": int(sum(values["evictions"])),
            "mean_synchronous_fences": (
                statistics.fmean(values["fences"]) if values["fences"] else None
            ),
            "mean_reader_active_capacity_fraction": (
                statistics.fmean(values["active_fraction"])
                if values["active_fraction"]
                else None
            ),
            "mean_reader_queue_nonempty_fraction": (
                statistics.fmean(values["queued_fraction"])
                if values["queued_fraction"]
                else None
            ),
        }
    result["pairs"] = (
        _pair_summary(pair_values)
        if len(pair_values) == len(RESOURCE_ORDER)
        and tuple(lane for lane, _ in pair_values) == RESOURCE_ORDER
        else {"samples": 0, "percent_changes": [], "all_positive": False}
    )
    result["coverage_gaps"] = sorted(coverage_gaps)
    for name, reason in (
        ("swap_pressure", "current telemetry has no run-scoped swap-pressure evidence"),
        ("page_pressure", "current telemetry has no run-scoped page-pressure evidence"),
    ):
        result[name] = {"status": "unavailable", "gate_passed": False, "reason": reason}
        errors.append(
            f"resource campaign {name.replace('_', '-')} evidence unavailable"
        )
    return result


def _headline_summary(
    entries: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    lane_values: dict[str, list[float]] = {"q4": [], "q2": []}
    pair_values: list[tuple[str, float]] = []
    for lane, _payload, run in entries:
        tps = _finite(run.get("completion_tokens_per_second"))
        if tps is not None and tps > 0:
            lane_values[lane].append(tps)
            pair_values.append((lane, tps))
    result: dict[str, Any] = {
        lane: _stats(values) for lane, values in lane_values.items()
    }
    result["pairs"] = (
        _pair_summary(pair_values)
        if len(pair_values) == len(HEADLINE_ORDER)
        and tuple(lane for lane, _ in pair_values) == HEADLINE_ORDER
        else {"samples": 0, "percent_changes": [], "all_positive": False}
    )
    return result


def summarize_hy3_q2_campaign(
    resource_payloads: Sequence[dict],
    headline_payloads: Sequence[dict],
    *,
    quality_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the fixed campaign and return one fail-closed evidence summary."""

    comparability_errors: list[str] = []
    quality_errors: list[str] = []
    integrity_errors: list[str] = []
    headline_errors: list[str] = []
    resource_errors: list[str] = []
    resource_entries = _validate_payloads(
        resource_payloads,
        kind="resource",
        order=RESOURCE_ORDER,
        comparability_errors=comparability_errors,
        gate_errors=resource_errors,
    )
    headline_entries = _validate_payloads(
        headline_payloads,
        kind="headline",
        order=HEADLINE_ORDER,
        comparability_errors=comparability_errors,
        gate_errors=headline_errors,
    )
    entries = [*resource_entries, *headline_entries]
    _validate_configuration(entries, errors=comparability_errors)
    _validate_run_integrity(entries, errors=integrity_errors)
    _validate_quality(quality_payload, errors=quality_errors)
    artifact_errors: list[str] = []
    artifact_binding = _validate_artifact_binding(
        entries, quality_payload, errors=artifact_errors
    )
    integrity_errors.extend(artifact_errors)
    resource = _resource_summary(resource_entries, errors=resource_errors)
    headline = _headline_summary(headline_entries)

    if headline["q4"]["samples"] != 4 or headline["q2"]["samples"] != 4:
        headline_errors.append(
            "headline campaign must contain four valid samples per lane"
        )
    if headline["pairs"].get("samples") != 4:
        headline_errors.append(
            "headline campaign must contain exactly four valid pairs"
        )
    if headline["pairs"].get("all_positive") is not True:
        headline_errors.append(
            "all four headline Q2-minus-Q4 pair deltas must be positive"
        )
    q4_bytes = resource.get("q4", {}).get("expert_bytes_per_token")
    q2_bytes = resource.get("q2", {}).get("expert_bytes_per_token")
    bytes_nonincrease = bool(
        isinstance(q4_bytes, (int, float))
        and isinstance(q2_bytes, (int, float))
        and math.isfinite(float(q4_bytes))
        and math.isfinite(float(q2_bytes))
        and q2_bytes <= q4_bytes
    )
    if not bytes_nonincrease:
        resource_errors.append("Q2 expert bytes per token increased relative to Q4")

    comparability_errors = list(dict.fromkeys(comparability_errors))
    quality_errors = list(dict.fromkeys(quality_errors))
    integrity_errors = list(dict.fromkeys(integrity_errors))
    headline_errors = list(dict.fromkeys(headline_errors))
    resource_errors = list(dict.fromkeys(resource_errors))
    comparable = not comparability_errors
    quality_passed = comparable and not quality_errors
    integrity_passed = comparable and not integrity_errors
    headline_passed = comparable and not headline_errors
    resource_passed = comparable and not resource_errors and bytes_nonincrease
    eligible = all((quality_passed, integrity_passed, headline_passed, resource_passed))
    errors = list(
        dict.fromkeys(
            [
                *comparability_errors,
                *quality_errors,
                *integrity_errors,
                *headline_errors,
                *resource_errors,
            ]
        )
    )
    observed_runs = sum(
        1
        for _lane_name, _payload, run in entries
        if run.get("completion_tokens") == 512
        and isinstance(run.get("token_ids"), list)
        and len(run["token_ids"]) == 512
    )
    return {
        "schema": "mtplx-hy3-expert-q2-campaign-summary-v1",
        "valid": eligible,
        "comparable": comparable,
        "errors": errors,
        "expected_order": {
            "resource": list(RESOURCE_ORDER),
            "headline": list(HEADLINE_ORDER),
        },
        "completion_tokens": {"expected": 512, "observed_runs": observed_runs},
        "cache_records": dict(EXPECTED_CAPACITIES),
        "comparability": {
            "passed": comparable,
            "errors": comparability_errors,
            "normalization": [
                "run_and_derived_labels",
                "model_key",
                "model_artifact_except_harness_source",
                "resource_telemetry_presence",
            ],
        },
        "quality_gate": {"passed": quality_passed, "errors": quality_errors},
        "integrity_gate": {
            "passed": integrity_passed,
            "errors": integrity_errors,
            "requires_full_record_integrity": True,
            "requires_fixed_memory_plan": True,
        },
        "headline_gate": {
            "passed": headline_passed,
            "errors": headline_errors,
            "requires_four_positive_pairs": True,
        },
        "resource_gate": {
            "passed": resource_passed,
            "errors": resource_errors,
            "expert_bytes_per_token_nonincrease": bytes_nonincrease,
            "requires_uncached_reader_bytes": True,
            "requires_run_scoped_swap_pressure": True,
            "requires_run_scoped_page_pressure": True,
        },
        "artifact_binding": artifact_binding,
        "decision": {
            "eligible": eligible,
            "requires": ["quality", "integrity", "headline", "resource"],
        },
        "resource": resource,
        "headline": headline,
    }


__all__ = ["HEADLINE_ORDER", "RESOURCE_ORDER", "summarize_hy3_q2_campaign"]
