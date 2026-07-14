from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from mtplx.benchmarks.hy3_q2_campaign import summarize_hy3_q2_campaign
from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "summarize_hy3_q2_campaign.py"
)
_GIB = 1024**3
_MEMORY_LIMIT = 112 * _GIB
_RUNTIME_RESERVE = 8 * _GIB
_EXPERT_CACHE_LIMIT = 83_034_243_072
_MAX_LIVE_KV_TOKENS = 18_888
_READ_CHUNK = 64 * 1024**2
_MODEL_KEYS = {"q4": "hy3-expert-only-q4", "q2": "hy3-expert-q2"}
_CAPACITIES = {"q4": 7_821, "q2": 14_077}
_RESOURCE_ORDER = ("q4", "q2", "q2", "q4")
_HEADLINE_ORDER = ("q4", "q2", "q2", "q4", "q4", "q2", "q2", "q4")


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_hy3_q2_campaign", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_identity(lane: str) -> dict:
    model_key = _MODEL_KEYS[lane]
    manifest_sha256 = ("a" if lane == "q4" else "b") * 64
    sidecar_sha256 = ("c" if lane == "q4" else "d") * 64
    sidecar_bytes = 161_036_107_776 if lane == "q4" else 89_464_504_320
    return {
        "method": "manifest_plus_executable_resident_content_v1",
        "verification_level": "full_small_files_and_authenticated_resident_content",
        "manifest": {
            "content_sha256": manifest_sha256,
            "declared_manifest_sha256": ("e" if lane == "q4" else "f") * 64,
            "model_key": model_key,
            "source_revision": "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
            "source_repo": "tencent/Hy3",
        },
        "expert_payload": {
            "method": "verified_sidecar_sha256",
            "verification_level": "actual_full_file_digest_matches_manifest",
            "sha256": sidecar_sha256,
            "size": sidecar_bytes,
            "page_cache_bypassed": True,
            "verification_method": "fresh_full_file_nocache_sha256",
        },
        "small_files": [
            {"name": "config.json", "sha256": "1" * 64, "size": 10},
            {
                "name": "model.safetensors.index.json",
                "sha256": "2" * 64,
                "size": 20,
            },
            {"name": "tokenizer.json", "sha256": "3" * 64, "size": 30},
            {
                "name": "tokenizer_config.json",
                "sha256": "4" * 64,
                "size": 40,
            },
        ],
        "resident_tensors": [
            {
                "tensor": "model.embed_tokens.weight",
                "shard": "model-00001-of-00018.safetensors",
                "offset": 4096,
                "length": 8192,
                "sha256": "5" * 64,
                "size": 8192,
                "page_cache_bypassed": True,
            }
        ],
        "harness_source": {
            "source_sha256": "6" * 64,
            "dependency_versions": {"mlx": "0.31.0"},
            "dirty": False,
        },
    }


def _stable_artifact(identity: dict) -> dict:
    operational = {
        "verification_elapsed_seconds",
        "verification_method",
        "identity_method",
        "verification_level",
        "page_cache_bypassed",
        "receipt_reused",
        "receipt",
    }

    def normalize(value):
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in value.items()
                if key not in operational
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(identity)


def _memory_plan(lane: str) -> dict:
    plan = plan_expert_memory(
        get_model_spec(_MODEL_KEYS[lane]),
        total_limit_bytes=_MEMORY_LIMIT,
        context_tokens=_MAX_LIVE_KV_TOKENS,
        runtime_reserve_bytes=_RUNTIME_RESERVE,
        expert_cache_limit_bytes=_EXPERT_CACHE_LIMIT,
        transient_slots=32,
        cache_scope="global",
    )
    assert plan.persistent_slots == _CAPACITIES[lane]
    payload = asdict(plan)
    payload["fixed_bytes"] = plan.fixed_bytes
    payload["allocated_bytes"] = plan.allocated_bytes
    return payload


def _telemetry(lane: str, position: int) -> dict:
    q4 = lane == "q4"
    read_bytes = (4_000_000_000 if q4 else 3_000_000_000) + position * 1_000
    hits = (2000 if q4 else 3000) + position
    misses = (1000 if q4 else 700) + position
    return {
        "schema": "mtplx-resource-telemetry-v2",
        "sample_count": 11,
        "samples_dropped": 0,
        "sampling_failures": 0,
        "sampling_failure": None,
        "interval_count": 10,
        "elapsed_seconds": 10.0,
        "throughput": {
            "completion_tokens": 512,
            "final_completion_tokens": 512,
            "completion_tokens_per_second": 8.0 if q4 else 10.0,
            "expert_requests": hits + misses,
            "expert_requests_per_second": (hits + misses) / 10.0,
        },
        "storage": {
            "io_cache_modes": ["f-nocache"],
            "reader_read_bytes": read_bytes,
            "reader_read_operations": 200 + position,
            "mean_gib_per_second": 5.0 if q4 else 6.0,
        },
        "reader_pool": {
            "worker_capacity": 32,
            "mean_active_readers": 8.0 if q4 else 7.0,
            "mean_queued_reads": 3.0 if q4 else 2.0,
            "active_capacity_fraction": 0.25 if q4 else 0.21875,
            "queue_nonempty_fraction": 0.6 if q4 else 0.4,
            "lifetime_queue_depth_peak": 12,
            "lifetime_active_readers_peak": 16,
        },
        "completion_fences": {
            "registrations": 400,
            "registered_slots": 3200,
            "fallbacks": 0,
            "failures": 0,
            "synchronous_fences": 20 if q4 else 15,
            "synchronous_fence_slots": 160 if q4 else 120,
        },
        "cache_by_layer": {
            "1": {
                "expert_hits": hits,
                "expert_misses": misses,
                "evictions": 100 if q4 else 50,
                "bytes_read": read_bytes,
                "hit_rate": hits / (hits + misses),
            }
        },
        "expert_pipeline": {
            "potentially_blocking_next_miss_fraction": 0.4 if q4 else 0.25,
            "coverage": {
                "attribution": "measured",
                "decode_phase": "measured",
                "sampler_window_backend": "measured_all_phases",
                "potentially_blocking_next_miss_step": "measured_upper_bound",
                "generation_expert_input_wait": "unavailable",
                "operation_credit": "unavailable",
                "byte_credit": "unavailable",
                "authoritative_reserve": "unavailable",
                "slot_capacity_admission": "unavailable",
                "outer_split_executor_queue": "unavailable",
                "eligible_unsubmitted_cause": "unattributed",
                "admitted_read_ranges": "unavailable",
                "scheduled_read_ranges": "unavailable",
                "physical_device_operations": "unavailable",
                "physical_device_bytes": "unavailable",
                "physical_device_queue_depth": "unavailable",
                "gpu_expert_wait": "unavailable",
                "gpu_idle_time": "unavailable",
                "future_layer_eligibility": "unavailable",
                "speculative_record_accounting": "unavailable",
                "python_preadv_when_native_reader": "unavailable",
            },
        },
        "coverage": {
            "runtime_occupancy": "measured",
            "storage_reads": "uncached_reader_bytes",
            "ssd_ceiling": "supplied",
            "gpu": "unavailable",
            "dram_bandwidth": "unavailable",
            "generation_thread_cpu": "measured",
            "timeline": "complete",
        },
    }


def _payload(kind: str, position: int, lane: str, *, tps: float) -> dict:
    assert kind in {"resource", "headline"}
    diagnostic = kind == "resource"
    model_key = _MODEL_KEYS[lane]
    artifact = _artifact_identity(lane)
    label = f"hy3-q2-{kind}-{position}-{lane}"
    runtime_config = {
        "model_key": model_key,
        "memory_limit_bytes": _MEMORY_LIMIT,
        "max_live_kv_tokens": _MAX_LIVE_KV_TOKENS,
        "runtime_reserve_bytes": _RUNTIME_RESERVE,
        "expert_cache_limit_bytes": _EXPERT_CACHE_LIMIT,
        "cache_policy": "lru",
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "max_read_chunk_bytes": _READ_CHUNK,
        "bypass_page_cache": True,
        "prefer_sidecar": True,
        "verify_record_hashes": False,
        "verify_sidecar_hash_at_open": False,
        "transient_slots": 32,
        "io_staging_bytes": 0,
        "execution_workspace_bytes": 0,
        "resource_telemetry": diagnostic,
    }
    performance_settings = {
        "runtime_config": runtime_config,
        "sampler": {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
        "seed": 0,
        "prompt_identity": {
            "content_sha256": "7" * 64,
            "content_bytes": 505,
            "token_sha256": "8" * 64,
            "token_count": 74,
        },
        "prompt_options": {
            "chat": True,
            "system_prompt": None,
            "prompt_style": None,
            "enable_thinking": False,
            "reasoning_effort": None,
            "prompt_metadata": None,
        },
        "generation": {
            "generation_profile": "deterministic",
            "max_tokens": 512,
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
        "model_artifact": _stable_artifact(artifact),
    }
    configuration_summary = {
        "run_label": label,
        "configuration_label": f"derived-{position}-{lane}-{kind}",
        "configuration_fingerprint": f"{position:016x}",
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "concurrency": 1,
        "requested_concurrency": 1,
        "execution_lane": "reference-ar",
        "performance_settings": performance_settings,
    }
    token_ids = [position] * 512
    run = {
        "run_label": label,
        "configuration_label": f"derived-{position}-{lane}-{kind}",
        "execution_lane": "reference-ar",
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "concurrency": 1,
        "requested_concurrency": 1,
        "achieved_peak_concurrency": 1,
        "saturation_valid": True,
        "undersubscribed": False,
        "completion_tokens": 512,
        "completion_tokens_per_second": tps,
        "token_ids": token_ids,
        "text": f"output-{kind}-{position}-{lane}",
        "finish_reason": "length",
        "streaming_after": {
            "memory_plan": _memory_plan(lane),
            "integrity": {
                "valid": True,
                "model_key": model_key,
                "checked_shards": 19,
                "checked_records": 15_168,
                "sidecar_verified": False,
            },
            "memory_cap": {
                "applied": True,
                "limit": _MEMORY_LIMIT - _RUNTIME_RESERVE,
            },
        },
    }
    if diagnostic:
        run["diagnostic_run"] = True
        run["resource_telemetry"] = _telemetry(lane, position)
    return {
        "schema": "mtplx-streamed-generation-benchmark-v1",
        "model_key": model_key,
        "seed": 0,
        "generation_profile": "deterministic",
        "run_label": label,
        "configuration_label": f"derived-{position}-{lane}-{kind}",
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "execution_lane": "reference-ar",
        "workload_shape": "static",
        "concurrency": 1,
        "requested_concurrency": 1,
        "achieved_peak_concurrency": 1,
        "saturation_valid": True,
        "undersubscribed": False,
        "enable_thinking": False,
        "chat": True,
        "mtp": {"enabled": False, "artifacts": None, "precision": None},
        "configuration_summary": configuration_summary,
        "artifact_verification": {"model": artifact, "mtp": None},
        "generation": {
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        },
        "reset_between": True,
        "runs": [run],
    }


def _quality() -> dict:
    lanes = {}
    for lane in ("q4", "q2"):
        artifact = _artifact_identity(lane)
        resident_name = "model-00001-of-00018.safetensors"
        resident_bytes = 16_384
        resident_sha256 = "5" * 64
        resident_digest = hashlib.sha256()
        encoded_name = resident_name.encode("utf-8")
        resident_digest.update(len(encoded_name).to_bytes(4, "big"))
        resident_digest.update(encoded_name)
        resident_digest.update(resident_bytes.to_bytes(8, "big"))
        resident_digest.update(bytes.fromhex(resident_sha256))
        lanes[lane] = {
            "model_key": _MODEL_KEYS[lane],
            "manifest": {
                "file_sha256": artifact["manifest"]["content_sha256"],
                "declared_sha256": artifact["manifest"]["declared_manifest_sha256"],
            },
            "tokenizer": {
                "sha256": "9" * 64,
                "files": [
                    {"name": "tokenizer.json", "sha256": "3" * 64},
                    {"name": "tokenizer_config.json", "sha256": "4" * 64},
                ],
            },
            "artifact": {
                "manifest_file_sha256": artifact["manifest"]["content_sha256"],
                "index": {"sha256": "2" * 64},
                "residents": {
                    "algorithm": "sha256-name-size-and-verified-file-digest-v1",
                    "sha256": resident_digest.hexdigest(),
                    "range_algorithm": (
                        "sha256-tensor-shard-offset-length-and-content-v1"
                    ),
                    "files": [
                        {
                            "name": resident_name,
                            "bytes": resident_bytes,
                            "sha256": resident_sha256,
                            "declared_sha256": resident_sha256,
                            "page_cache_bypassed": True,
                        }
                    ],
                    "ranges": [
                        {
                            "tensor": "model.embed_tokens.weight",
                            "shard": "model-00001-of-00018.safetensors",
                            "offset": 4096,
                            "length": 8192,
                            "sha256": "5" * 64,
                        }
                    ],
                },
            },
            "loss": {"finite": True, "perplexity": 10.0 if lane == "q4" else 10.2},
        }
    return {
        "schema": "mtplx-streamed-quality-v1",
        "passed": True,
        "quality_passed": True,
        "finite": True,
        "relative_perplexity_regression": 0.02,
        "max_relative_perplexity_regression": 0.05,
        "nan_count": 0,
        "nonfinite_count": 0,
        "lanes": lanes,
        "errors": [],
    }


def _campaigns() -> tuple[list[dict], list[dict], dict]:
    resource = [
        _payload("resource", 1, "q4", tps=8.0),
        _payload("resource", 2, "q2", tps=10.0),
        _payload("resource", 3, "q2", tps=11.0),
        _payload("resource", 4, "q4", tps=9.0),
    ]
    headline = [
        _payload("headline", 1, "q4", tps=10.0),
        _payload("headline", 2, "q2", tps=12.0),
        _payload("headline", 3, "q2", tps=15.0),
        _payload("headline", 4, "q4", tps=12.5),
        _payload("headline", 5, "q4", tps=20.0),
        _payload("headline", 6, "q2", tps=21.0),
        _payload("headline", 7, "q2", tps=27.0),
        _payload("headline", 8, "q4", tps=25.0),
    ]
    return resource, headline, _quality()


def test_complete_campaign_is_comparable_but_resource_ineligible() -> None:
    resource, headline, quality = _campaigns()

    summary = summarize_hy3_q2_campaign(
        resource,
        headline,
        quality_payload=quality,
    )

    assert summary["comparable"] is True
    assert summary["quality_gate"]["passed"] is True
    assert summary["integrity_gate"]["passed"] is True
    assert summary["headline_gate"]["passed"] is True
    assert summary["resource_gate"]["passed"] is False
    assert summary["decision"]["eligible"] is False
    assert summary["valid"] is False
    assert summary["cache_records"] == _CAPACITIES
    assert summary["resource"]["swap_pressure"]["status"] == "unavailable"
    assert summary["resource"]["page_pressure"]["status"] == "unavailable"
    assert (
        summary["resource"]["q2"]["expert_bytes_per_token"]
        <= summary["resource"]["q4"]["expert_bytes_per_token"]
    )
    assert summary["headline"]["q4"]["samples"] == 4
    assert summary["headline"]["q2"]["samples"] == 4
    assert summary["headline"]["pairs"]["samples"] == 4
    assert summary["headline"]["pairs"]["all_positive"] is True
    assert summary["completion_tokens"] == {"expected": 512, "observed_runs": 12}


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        ("prompt_identity", "content_sha256", "f" * 64),
        ("generation", "generation_profile", "model-default"),
        ("sampler", "temperature", 0.5),
        ("settings", "seed", 1),
        ("runtime_config", "memory_limit_bytes", _MEMORY_LIMIT - 1),
        ("runtime_config", "runtime_reserve_bytes", _RUNTIME_RESERVE - 1),
        ("runtime_config", "expert_cache_limit_bytes", _EXPERT_CACHE_LIMIT - 1),
        ("runtime_config", "max_live_kv_tokens", 8192),
        ("runtime_config", "cache_policy", "frequency"),
        ("runtime_config", "cache_scope", "layer"),
        ("runtime_config", "slot_layout", "direct-slots"),
        ("runtime_config", "transient_slots", 31),
        ("runtime_config", "max_read_chunk_bytes", 8 * 1024**2),
        ("runtime_config", "bypass_page_cache", False),
        ("runtime_config", "verify_record_hashes", True),
    ),
)
def test_rejects_exact_contract_drift(
    section: str,
    field: str,
    replacement: object,
) -> None:
    resource, headline, quality = _campaigns()
    settings = headline[-1]["configuration_summary"]["performance_settings"]
    target = settings if section == "settings" else settings[section]
    target[field] = replacement

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["comparable"] is False
    assert summary["decision"]["eligible"] is False
    assert summary["comparability"]["errors"]


def test_rejects_top_level_workload_shape_drift() -> None:
    resource, headline, quality = _campaigns()
    headline[-1]["workload_shape"] = "dynamic"

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["comparable"] is False
    assert any("workload_shape" in error for error in summary["errors"])


@pytest.mark.parametrize(
    "location",
    ("payload", "configuration", "scheduler", "run"),
)
def test_requires_reference_ar_at_every_exported_level(location: str) -> None:
    resource, headline, quality = _campaigns()
    payload = headline[-1]
    if location == "payload":
        payload["execution_lane"] = "continuous-batch-ar"
    elif location == "configuration":
        payload["configuration_summary"]["execution_lane"] = "continuous-batch-ar"
    elif location == "scheduler":
        payload["configuration_summary"]["performance_settings"]["scheduler"][
            "execution_lane"
        ] = "continuous-batch-ar"
    else:
        payload["runs"][0]["execution_lane"] = "continuous-batch-ar"

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["comparable"] is False
    assert any("reference-ar" in error for error in summary["errors"])


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_resource",
        "missing_headline",
        "extra_run",
        "resource_order",
        "headline_order",
        "wrong_label",
    ),
)
def test_requires_exact_payload_counts_runs_and_orders(mutation: str) -> None:
    resource, headline, quality = _campaigns()
    if mutation == "missing_resource":
        resource.pop()
    elif mutation == "missing_headline":
        headline.pop()
    elif mutation == "extra_run":
        headline[0]["runs"].append(deepcopy(headline[0]["runs"][0]))
    elif mutation == "resource_order":
        resource[0], resource[1] = resource[1], resource[0]
    elif mutation == "headline_order":
        headline[0], headline[1] = headline[1], headline[0]
    else:
        headline[0]["run_label"] = "hy3-q2-headline-99-q4"

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["comparable"] is False
    assert summary["errors"]


@pytest.mark.parametrize(("lane", "capacity"), (("q4", 7820), ("q2", 14076)))
def test_requires_exact_global_cache_capacity(lane: str, capacity: int) -> None:
    resource, headline, quality = _campaigns()
    payload = next(item for item in resource if item["model_key"] == _MODEL_KEYS[lane])
    payload["runs"][0]["streaming_after"]["memory_plan"]["persistent_slots"] = capacity

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["integrity_gate"]["passed"] is False
    assert any("persistent_slots" in error for error in summary["errors"])
    assert summary["artifact_binding"]["passed"] is True


@pytest.mark.parametrize(
    "location",
    (
        "manifest",
        "index",
        "tokenizer",
        "expert_payload",
        "harness",
        "quality_resident_hash",
        "quality_resident_whole_file_hash",
        "benchmark_resident_range_hash",
    ),
)
def test_quality_receipt_must_bind_every_benchmark_artifact(location: str) -> None:
    resource, headline, quality = _campaigns()
    payload = headline[-1]
    model = payload["artifact_verification"]["model"]
    if location == "manifest":
        model["manifest"]["content_sha256"] = "f" * 64
    elif location == "index":
        next(
            item
            for item in model["small_files"]
            if item["name"] == "model.safetensors.index.json"
        )["sha256"] = "f" * 64
    elif location == "tokenizer":
        next(item for item in model["small_files"] if item["name"] == "tokenizer.json")[
            "sha256"
        ] = "f" * 64
    elif location == "expert_payload":
        model["expert_payload"]["verification_level"] = "declared_only"
    elif location == "harness":
        model["harness_source"]["source_sha256"] = "f" * 64
    elif location == "quality_resident_hash":
        quality["lanes"]["q2"]["artifact"]["residents"]["files"][0]["sha256"] = "f" * 64
    elif location == "quality_resident_whole_file_hash":
        receipt = quality["lanes"]["q2"]["artifact"]["residents"]["files"][0]
        receipt["sha256"] = "f" * 64
        receipt["declared_sha256"] = "f" * 64
    else:
        for candidate in [*resource, *headline]:
            full = candidate["artifact_verification"]["model"]
            stable = candidate["configuration_summary"]["performance_settings"][
                "model_artifact"
            ]
            full["resident_tensors"][0]["sha256"] = "f" * 64
            stable["resident_tensors"][0]["sha256"] = "f" * 64

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["integrity_gate"]["passed"] is False
    assert summary["decision"]["eligible"] is False
    assert summary["artifact_binding"]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    ("quality_failed", "nonfinite", "too_high", "lane_key", "resident_mismatch"),
)
def test_quality_gate_is_bound_and_fail_closed(mutation: str) -> None:
    resource, headline, quality = _campaigns()
    if mutation == "quality_failed":
        quality["quality_passed"] = False
        quality["passed"] = False
    elif mutation == "nonfinite":
        quality["finite"] = False
    elif mutation == "too_high":
        quality["relative_perplexity_regression"] = 0.0500001
    elif mutation == "lane_key":
        quality["lanes"]["q2"]["model_key"] = "hy3-expert-only-q4"
    else:
        quality["lanes"]["q2"]["artifact"]["residents"]["sha256"] = "f" * 64

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["quality_gate"]["passed"] is False
    assert summary["decision"]["eligible"] is False


@pytest.mark.parametrize("tokens", (0, 511, 513))
def test_every_run_must_reach_exactly_512_completion_tokens(tokens: int) -> None:
    resource, headline, quality = _campaigns()
    run = headline[-1]["runs"][0]
    run["completion_tokens"] = tokens
    run["token_ids"] = [1] * tokens

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["headline_gate"]["passed"] is False
    assert any("512" in error for error in summary["errors"])


def test_all_four_headline_pair_deltas_must_be_positive() -> None:
    resource, headline, quality = _campaigns()
    headline[1]["runs"][0]["completion_tokens_per_second"] = 10.0

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["headline"]["pairs"]["all_positive"] is False
    assert summary["headline_gate"]["passed"] is False
    assert summary["decision"]["eligible"] is False


def test_q2_expert_bytes_per_token_must_not_increase() -> None:
    resource, headline, quality = _campaigns()
    for payload in resource:
        if payload["model_key"] == _MODEL_KEYS["q2"]:
            telemetry = payload["runs"][0]["resource_telemetry"]
            telemetry["storage"]["reader_read_bytes"] = 8_000_000_000
            telemetry["cache_by_layer"]["1"]["bytes_read"] = 8_000_000_000

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["resource_gate"]["expert_bytes_per_token_nonincrease"] is False
    assert summary["decision"]["eligible"] is False


def test_resource_reader_bytes_must_match_cache_attribution() -> None:
    resource, headline, quality = _campaigns()
    resource[0]["runs"][0]["resource_telemetry"]["cache_by_layer"]["1"][
        "bytes_read"
    ] -= 1

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert any("cache bytes" in error.lower() for error in summary["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("nonfinite_tps", "finite"),
        ("missing_telemetry", "resource_telemetry"),
        ("headline_telemetry", "must not contain"),
        ("buffered", "f-nocache"),
        ("sample_failure", "sampling"),
        ("coverage", "coverage"),
        ("integrity", "integrity"),
        ("memory_cap", "memory cap"),
    ),
)
def test_strict_numeric_telemetry_coverage_and_integrity_gates(
    mutation: str,
    expected_error: str,
) -> None:
    resource, headline, quality = _campaigns()
    if mutation == "nonfinite_tps":
        headline[0]["runs"][0]["completion_tokens_per_second"] = float("nan")
    elif mutation == "missing_telemetry":
        resource[0]["runs"][0].pop("resource_telemetry")
    elif mutation == "headline_telemetry":
        headline[0]["runs"][0]["resource_telemetry"] = deepcopy(
            resource[0]["runs"][0]["resource_telemetry"]
        )
    elif mutation == "buffered":
        resource[0]["runs"][0]["resource_telemetry"]["storage"]["io_cache_modes"] = [
            "buffered"
        ]
    elif mutation == "sample_failure":
        resource[0]["runs"][0]["resource_telemetry"]["sampling_failures"] = 1
    elif mutation == "coverage":
        resource[0]["runs"][0]["resource_telemetry"]["coverage"].pop("timeline")
    elif mutation == "integrity":
        resource[0]["runs"][0]["streaming_after"]["integrity"]["valid"] = False
    else:
        resource[0]["runs"][0]["streaming_after"]["memory_cap"]["applied"] = False

    summary = summarize_hy3_q2_campaign(resource, headline, quality_payload=quality)

    assert summary["decision"]["eligible"] is False
    assert any(expected_error.lower() in error.lower() for error in summary["errors"])


def _write_payloads(tmp_path: Path, payloads: list[dict], prefix: str) -> list[Path]:
    paths = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"{prefix}-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    return paths


def test_cli_writes_complete_rejection_before_exit_two(tmp_path: Path) -> None:
    module = _load_script()
    resource, headline, quality = _campaigns()
    resource_paths = _write_payloads(tmp_path, resource, "resource")
    headline_paths = _write_payloads(tmp_path, headline, "headline")
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    output_dir = tmp_path / "external"
    output_dir.mkdir()
    output = output_dir / "summary.json"

    status = module.main(
        [
            "--resource",
            *map(str, resource_paths),
            "--headline",
            *map(str, headline_paths),
            "--quality",
            str(quality_path),
            "--output-json",
            str(output),
        ]
    )

    assert status == 2
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["comparable"] is True
    assert written["resource_gate"]["passed"] is False
    assert written["resource"]["swap_pressure"]["status"] == "unavailable"


def test_cli_exit_zero_is_reserved_for_an_eligible_summary(tmp_path: Path) -> None:
    module = _load_script()
    resource, headline, quality = _campaigns()
    resource_paths = _write_payloads(tmp_path, resource, "resource")
    headline_paths = _write_payloads(tmp_path, headline, "headline")
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    output_dir = tmp_path / "external"
    output_dir.mkdir()
    output = output_dir / "summary.json"

    def eligible_summary(*args, **kwargs):
        return {"decision": {"eligible": True}, "valid": True}

    status = module.main(
        [
            "--resource",
            *map(str, resource_paths),
            "--headline",
            *map(str, headline_paths),
            "--quality",
            str(quality_path),
            "--output-json",
            str(output),
        ],
        _summarize=eligible_summary,
    )

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True


def test_cli_returns_one_without_output_for_malformed_input(tmp_path: Path) -> None:
    module = _load_script()
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    output_dir = tmp_path / "external"
    output_dir.mkdir()
    output = output_dir / "summary.json"

    status = module.main(
        [
            "--resource",
            str(bad),
            "--headline",
            str(bad),
            "--quality",
            str(bad),
            "--output-json",
            str(output),
        ]
    )

    assert status == 1
    assert not output.exists()


def test_cli_refuses_tracked_or_existing_output(tmp_path: Path) -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="outside the Git worktree"):
        module.validate_external_output_path(
            Path(__file__).resolve().parent / "tracked-summary.json"
        )
    existing = tmp_path / "summary.json"
    existing.write_text("attacker", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        module.validate_external_output_path(existing)


def test_exclusive_writer_rejects_parent_substitution_without_deleting_attacker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    parent = tmp_path / "evidence"
    moved = tmp_path / "evidence-original"
    parent.mkdir()
    output = parent / "summary.json"
    real_link = module.os.link

    def substitute_parent_after_link(*args, **kwargs):
        result = real_link(*args, **kwargs)
        parent.rename(moved)
        parent.mkdir()
        output.write_text("attacker", encoding="utf-8")
        return result

    monkeypatch.setattr(module.os, "link", substitute_parent_after_link)

    with pytest.raises(RuntimeError, match="parent.*substitut"):
        module._write_json_exclusive(output, {"valid": False})

    assert output.read_text(encoding="utf-8") == "attacker"
    assert not (moved / output.name).exists()
