from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from mtplx.expert_streaming_models import (
    GLM52_Q4,
    HY3_EXPERT_ONLY_Q4,
    HY3_EXPERT_Q2,
    HY3_Q4,
    MODEL_SPECS,
    get_model_spec,
    plan_expert_memory,
)


GIB = 1 << 30
ROOT = Path(__file__).resolve().parents[1]


def test_hy3_q4_exact_expert_layout() -> None:
    spec = get_model_spec("hy3-q4")

    assert MODEL_SPECS["hy3-q4"] is spec
    assert spec.total_layers == 80
    assert spec.total_tensor_bytes == 165_988_461_824
    assert spec.routed_layer_indices == tuple(range(1, 80))
    assert spec.expert_count == 192
    assert spec.top_k == 8
    assert spec.hidden_size == 4096
    assert spec.expert_hidden_size == 1536
    assert spec.quant_bits == 4
    assert spec.quant_group_size == 64
    assert spec.quant_parameter_bytes == 2
    assert spec.expert_record_bytes == 10_616_832
    assert spec.routed_expert_bytes == 161_036_107_776
    assert spec.cold_expert_bytes_per_token == 6_709_837_824
    assert spec.transient_scratch_bytes == 84_934_656
    assert spec.persistent_cache_bytes(0) == 0
    assert spec.persistent_cache_bytes(8) == 6_709_837_824
    assert spec.persistent_cache_bytes(32) == 26_839_351_296
    assert spec.resident_bytes == 4_952_354_048
    assert spec.routed_expert_bytes + spec.resident_bytes == spec.total_tensor_bytes
    assert spec.router_bytes == 66_071_808
    assert spec.router_storage == "affine-q8 with fp32 correction bias"
    assert spec.router_matmul_dtype == "activation_dtype"
    assert spec.source_revision == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert spec.quant_revision == "160619d3f96c8470350b6dac0ef033a8381551e3"
    assert spec.kv_bytes_per_token == 327_680
    assert spec.mtp_layer_index == 80
    assert spec.mtp_included is False
    assert spec.full_indexer_layers == ()


def test_hy3_expert_only_q4_control_exact_layout() -> None:
    q4 = get_model_spec("hy3-expert-only-q4")

    assert q4.quant_model == "local/hy3-expert-only-mlx-q4"
    assert q4.quant_revision == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert q4.quant_bits == 4
    assert q4.expert_record_bytes == 10_616_832
    assert q4.routed_expert_bytes == 161_036_107_776
    assert q4.resident_bytes == 17_494_289_664
    assert q4.total_tensor_bytes == 178_530_397_440
    assert q4.router_storage == "source bfloat16 with fp32 correction bias"
    assert q4.router_bytes == 124_316_928

    cache = 83_034_243_072
    assert cache // q4.expert_record_bytes == 7_821


def test_hy3_expert_q2_exact_layout() -> None:
    q2 = get_model_spec("hy3-expert-q2")

    assert q2.source_model == "tencent/Hy3"
    assert q2.source_revision == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert q2.quant_model == "local/hy3-expert-only-mlx-q4"
    assert q2.quant_revision == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert q2.quant_bits == 2
    assert q2.expert_record_bytes == 5_898_240
    assert q2.routed_expert_bytes == 89_464_504_320
    assert q2.resident_bytes == 17_494_289_664
    assert q2.total_tensor_bytes == 106_958_793_984
    assert q2.mtp_included is False
    assert q2.router_storage == "source bfloat16 with fp32 correction bias"
    assert q2.router_bytes == 124_316_928

    cache = 83_034_243_072
    assert cache // q2.expert_record_bytes == 14_077


def test_hy3_q4_is_unchanged_by_hy3_expert_q2_registry_expansion() -> None:
    expected = {
        "key": "hy3-q4",
        "display_name": "Tencent Hy3 affine Q4",
        "source_model": "tencent/Hy3",
        "source_revision": "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
        "quant_model": "pipenetwork/Hy3-4bit",
        "quant_revision": "160619d3f96c8470350b6dac0ef033a8381551e3",
        "total_tensor_bytes": 165_988_461_824,
        "total_layers": 80,
        "routed_layer_start": 1,
        "routed_layer_count": 79,
        "expert_count": 192,
        "top_k": 8,
        "hidden_size": 4096,
        "expert_hidden_size": 1536,
        "quant_bits": 4,
        "quant_group_size": 64,
        "quant_parameter_bytes": 2,
        "router_storage": "affine-q8 with fp32 correction bias",
        "router_matmul_dtype": "activation_dtype",
        "router_bytes": 66_071_808,
        "kv_bytes_per_token": 327_680,
        "mtp_layer_index": 80,
        "mtp_included": False,
        "full_indexer_layers": (),
        "island_pin_order": (),
        "expert_codec": "affine",
    }
    before = asdict(HY3_Q4)

    get_model_spec("hy3-expert-only-q4")
    get_model_spec("hy3-expert-q2")

    assert before == expected
    assert asdict(HY3_Q4) == expected
    assert MODEL_SPECS["hy3-q4"] is HY3_Q4


def test_glm52_q4_exact_expert_and_indexshare_layout() -> None:
    spec = get_model_spec("glm52-q4")

    assert MODEL_SPECS["glm52-q4"] is spec
    assert spec.total_layers == 78
    assert spec.total_tensor_bytes == 418_320_895_488
    assert spec.routed_layer_indices == tuple(range(3, 78))
    assert spec.expert_count == 256
    assert spec.top_k == 8
    assert spec.hidden_size == 6144
    assert spec.expert_hidden_size == 2048
    assert spec.quant_bits == 4
    assert spec.quant_group_size == 64
    assert spec.quant_parameter_bytes == 2
    assert spec.expert_record_bytes == 21_233_664
    assert spec.routed_expert_bytes == 407_686_348_800
    assert spec.cold_expert_bytes_per_token == 12_740_198_400
    assert spec.transient_scratch_bytes == 169_869_312
    assert spec.persistent_cache_bytes(0) == 0
    assert spec.persistent_cache_bytes(8) == 12_740_198_400
    assert spec.persistent_cache_bytes(32) == 50_960_793_600
    assert spec.persistent_cache_bytes(64) == 101_921_587_200
    assert spec.resident_bytes == 10_634_546_688
    assert spec.routed_expert_bytes + spec.resident_bytes == spec.total_tensor_bytes
    assert spec.router_bytes == 236_006_400
    assert spec.router_storage == "bfloat16 with fp32 correction bias"
    assert spec.router_matmul_dtype == "float32"
    assert spec.source_revision == "b4734de4facf877f85769a911abafc5283eab3d9"
    assert spec.quant_revision == "6b347a6472d46bf55de65ee34032136a3929d778"
    assert spec.kv_bytes_per_token == 95_232
    assert spec.mtp_layer_index == 78
    assert spec.mtp_included is False
    assert spec.full_indexer_layers == (0, 1, 2, *range(6, 75, 4))
    assert len(spec.full_indexer_layers) == 21


def test_glm52_expert_q2_exact_expert_and_indexshare_layout() -> None:
    spec = get_model_spec("glm52-expert-q2")

    assert MODEL_SPECS["glm52-expert-q2"] is spec
    assert spec.display_name == "GLM-5.2 expert-only affine Q2"
    assert spec.source_model == "zai-org/GLM-5.2"
    assert spec.source_revision == "b4734de4facf877f85769a911abafc5283eab3d9"
    assert spec.quant_model == "mlx-community/GLM-5.2-4bit"
    assert spec.quant_revision == "6b347a6472d46bf55de65ee34032136a3929d778"
    assert spec.total_layers == 78
    assert spec.total_tensor_bytes == 237_126_962_688
    assert spec.routed_layer_indices == tuple(range(3, 78))
    assert spec.expert_count == 256
    assert spec.top_k == 8
    assert spec.hidden_size == 6144
    assert spec.expert_hidden_size == 2048
    assert spec.quant_bits == 2
    assert spec.quant_group_size == 64
    assert spec.quant_parameter_bytes == 2
    assert spec.packed_weight_bytes == 9_437_184
    assert spec.scale_bias_bytes == 2_359_296
    assert spec.expert_record_bytes == 11_796_480
    assert spec.routed_layer_count * spec.expert_count == 19_200
    assert spec.routed_expert_bytes == 226_492_416_000
    assert spec.resident_bytes == 10_634_546_688
    assert spec.routed_expert_bytes + spec.resident_bytes == spec.total_tensor_bytes
    assert spec.router_bytes == 236_006_400
    assert spec.router_storage == "bfloat16 with fp32 correction bias"
    assert spec.router_matmul_dtype == "float32"
    assert spec.kv_bytes_per_token == 95_232
    assert spec.mtp_layer_index == 78
    assert spec.mtp_included is False
    assert spec.full_indexer_layers == (0, 1, 2, *range(6, 75, 4))
    assert len(spec.full_indexer_layers) == 21


def test_existing_descriptors_are_unchanged_by_glm52_expert_q2_registry_expansion() -> (
    None
):
    expected_digests = {
        # Digests track the intentional field set: island_pin_order
        # (auto-census #98) and expert_codec (q1 lane, issue #51) are
        # spec fields now; a digest change without a matching field-set
        # change is the drift this test exists to catch.
        "hy3-q4": "bc121154d4d6286e9499995e06632e10007ef5c9b343254018a60bff59ef344c",
        # 2026-07-21: island_pin_order shared across hy3 banks (a3300f0)
        "hy3-expert-only-q4": "831c990eaa6b87c0ff5fe434a8d79ee91d198501c1dfe3b38220808af8d7ad40",
        "hy3-expert-q2": "286bc48306801005db9b32d96362ac2553bbe4ce5e53112503b1a12c9a6a78ea",
        "glm52-q4": "6372e17bf28658526de0b2150cda8fad486f077849b5d0c5be23ff19e6b770b1",
    }
    existing = (HY3_Q4, HY3_EXPERT_ONLY_Q4, HY3_EXPERT_Q2, GLM52_Q4)

    actual_digests = {
        spec.key: hashlib.sha256(
            json.dumps(
                asdict(spec),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for spec in existing
    }

    assert actual_digests == expected_digests


def test_model_registry_contains_all_streaming_targets() -> None:
    assert {
        "hy3-q4",
        "hy3-expert-only-q4",
        "hy3-expert-q2",
        "glm52-q4",
        "glm52-expert-q2",
    } <= MODEL_SPECS.keys()
    with pytest.raises(ValueError, match="unknown model"):
        get_model_spec("unknown-model")


@pytest.mark.parametrize("model_key", ["hy3-q4", "glm52-q4"])
def test_memory_plan_turns_a_total_limit_into_whole_per_layer_slots(
    model_key: str,
) -> None:
    spec = get_model_spec(model_key)
    context_tokens = 32_768
    runtime_reserve = 2 * GIB
    fixed_bytes = (
        spec.resident_bytes
        + context_tokens * spec.kv_bytes_per_token
        + spec.transient_scratch_bytes
        + runtime_reserve
    )
    total_limit = fixed_bytes + spec.persistent_cache_bytes(20) + 12_345

    plan = plan_expert_memory(
        spec,
        total_limit_bytes=total_limit,
        context_tokens=context_tokens,
        runtime_reserve_bytes=runtime_reserve,
    )

    assert plan.fits_fixed is True
    assert plan.total_limit_bytes == total_limit
    assert plan.context_tokens == context_tokens
    assert plan.runtime_reserve_bytes == runtime_reserve
    assert plan.resident_bytes == spec.resident_bytes
    assert plan.kv_bytes == context_tokens * spec.kv_bytes_per_token
    assert plan.transient_bytes == spec.transient_scratch_bytes
    assert plan.persistent_budget_bytes == spec.persistent_cache_bytes(20) + 12_345
    assert plan.slots_per_layer == 20
    assert plan.persistent_cache_bytes == spec.persistent_cache_bytes(20)
    assert plan.unallocated_bytes == 12_345
    assert plan.allocated_bytes <= plan.total_limit_bytes


def test_memory_plan_charges_external_mtp_head_as_resident_mlx_bytes() -> None:
    spec = get_model_spec("glm52-expert-q2")
    mtp_head_bytes = 19_905_841_664
    arguments = {
        "total_limit_bytes": 112 * GIB,
        "context_tokens": 4096,
        "runtime_reserve_bytes": 12 * GIB,
        "expert_cache_limit_bytes": 64 * GIB,
        "transient_slots": 8,
    }
    base = plan_expert_memory(spec, **arguments)
    with_mtp = plan_expert_memory(
        spec,
        **arguments,
        additional_resident_bytes=mtp_head_bytes,
    )

    assert with_mtp.resident_bytes == base.resident_bytes + mtp_head_bytes
    assert with_mtp.fixed_bytes == base.fixed_bytes + mtp_head_bytes
    assert with_mtp.persistent_budget_bytes == min(
        64 * GIB,
        max(0, with_mtp.total_limit_bytes - with_mtp.fixed_bytes),
    )
    assert with_mtp.unallocated_bytes == base.unallocated_bytes - mtp_head_bytes
    assert with_mtp.allocated_bytes <= with_mtp.total_limit_bytes


def test_explicit_expert_cache_limit_caps_slots_below_available_memory() -> None:
    spec = get_model_spec("glm52-q4")
    per_layer_bank_slot = spec.persistent_cache_bytes(1)
    fixed_bytes = spec.resident_bytes + spec.transient_scratch_bytes
    total_limit = fixed_bytes + spec.persistent_cache_bytes(40)
    expert_cache_limit = spec.persistent_cache_bytes(12) + per_layer_bank_slot - 1

    plan = plan_expert_memory(
        spec,
        total_limit_bytes=total_limit,
        context_tokens=0,
        expert_cache_limit_bytes=expert_cache_limit,
    )

    assert plan.fits_fixed is True
    assert plan.persistent_budget_bytes == expert_cache_limit
    assert plan.slots_per_layer == 12
    assert plan.persistent_cache_bytes == spec.persistent_cache_bytes(12)
    assert plan.unallocated_bytes == spec.persistent_cache_bytes(28)


def test_global_memory_plan_uses_record_granularity_not_uniform_layer_rounding() -> (
    None
):
    spec = get_model_spec("hy3-q4")
    expert_cache_limit = 80 * GIB
    fixed = spec.resident_bytes + spec.transient_scratch_bytes

    layer_plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + expert_cache_limit,
        context_tokens=0,
        expert_cache_limit_bytes=expert_cache_limit,
        cache_scope="layer",
    )
    global_plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + expert_cache_limit,
        context_tokens=0,
        expert_cache_limit_bytes=expert_cache_limit,
        cache_scope="global",
    )

    assert layer_plan.persistent_slots == 79 * 102
    assert (
        global_plan.persistent_slots == expert_cache_limit // spec.expert_record_bytes
    )
    assert global_plan.persistent_slots == 8_090
    assert global_plan.persistent_cache_bytes > layer_plan.persistent_cache_bytes
    assert 0 <= global_plan.unallocated_bytes < spec.expert_record_bytes


def test_memory_plan_reports_when_fixed_footprint_does_not_fit() -> None:
    spec = get_model_spec("hy3-q4")
    context_tokens = 65_536
    runtime_reserve = 4 * GIB
    fixed_bytes = (
        spec.resident_bytes
        + context_tokens * spec.kv_bytes_per_token
        + spec.transient_scratch_bytes
        + runtime_reserve
    )

    plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed_bytes - 1,
        context_tokens=context_tokens,
        runtime_reserve_bytes=runtime_reserve,
    )

    assert plan.fits_fixed is False
    assert plan.persistent_budget_bytes == 0
    assert plan.slots_per_layer == 0
    assert plan.persistent_cache_bytes == 0
    assert plan.unallocated_bytes == -1


@pytest.mark.parametrize("model_key", ["hy3-q4", "glm52-q4"])
def test_memory_plan_exact_fixed_one_slot_and_full_residency(model_key: str) -> None:
    spec = get_model_spec(model_key)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes

    fixed_only = plan_expert_memory(
        spec,
        total_limit_bytes=fixed,
        context_tokens=0,
    )
    one_slot = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        context_tokens=0,
    )
    full = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.routed_expert_bytes + GIB,
        context_tokens=0,
    )

    assert fixed_only.fits_fixed is True
    assert fixed_only.slots_per_layer == 0
    assert fixed_only.unallocated_bytes == 0
    assert one_slot.slots_per_layer == 1
    assert one_slot.unallocated_bytes == 0
    assert full.slots_per_layer == spec.expert_count
    assert full.persistent_budget_bytes == spec.routed_expert_bytes
    assert full.persistent_cache_bytes == spec.routed_expert_bytes
    assert full.unallocated_bytes == GIB
    assert all(
        plan.allocated_bytes <= plan.total_limit_bytes
        for plan in (fixed_only, one_slot, full)
    )


def test_zero_cache_cap_and_explicit_fixed_workspaces_are_accounted() -> None:
    spec = get_model_spec("glm52-q4")
    transient_slots = 12
    io_staging = 512 * 2**20
    execution_workspace = 256 * 2**20
    fixed = (
        spec.resident_bytes
        + transient_slots * spec.expert_record_bytes
        + io_staging
        + execution_workspace
    )

    plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + 10 * GIB,
        context_tokens=0,
        transient_slots=transient_slots,
        io_staging_bytes=io_staging,
        execution_workspace_bytes=execution_workspace,
        expert_cache_limit_bytes=0,
    )

    assert plan.transient_slots == transient_slots
    assert plan.transient_bytes == transient_slots * spec.expert_record_bytes
    assert plan.io_staging_bytes == io_staging
    assert plan.execution_workspace_bytes == execution_workspace
    assert plan.fixed_bytes == fixed
    assert plan.persistent_budget_bytes == 0
    assert plan.slots_per_layer == 0
    assert plan.unallocated_bytes == 10 * GIB


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"total_limit_bytes": 100.5}, TypeError),
        ({"context_tokens": 1.5}, TypeError),
        ({"transient_slots": 8.5}, TypeError),
        ({"expert_cache_limit_bytes": True}, TypeError),
        ({"transient_slots": 7}, ValueError),
    ],
)
def test_memory_plan_rejects_lossy_or_undersized_inputs(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    spec = get_model_spec("hy3-q4")
    arguments: dict[str, object] = {
        "total_limit_bytes": 128 * GIB,
        "context_tokens": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(error):
        plan_expert_memory(spec, **arguments)  # type: ignore[arg-type]


def test_model_spec_rejects_float_dimensions_and_misaligned_projections() -> None:
    spec = get_model_spec("hy3-q4")

    with pytest.raises(TypeError, match="total_layers must be an integer"):
        replace(spec, total_layers=80.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="projection input"):
        replace(spec, expert_hidden_size=1537)


def test_glm_model_key_drives_trace_simulator_geometry(tmp_path: Path) -> None:
    trace = tmp_path / "glm-routes.jsonl"
    trace.write_text(
        json.dumps(
            {
                "phase": "decode",
                "layer": 3,
                "experts": list(range(8)),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_expert_cache.py"),
            str(trace),
            "--model",
            "glm52-q4",
            "--persistent-slots-per-layer",
            "0",
        ],
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model_key"] == "glm52-q4"
    assert payload["expert_misses"] == 8
    assert payload["bytes_read"] == 8 * 21_233_664
    assert payload["transient_scratch_bytes"] == 8 * 21_233_664
    assert payload["allocated_layer_count"] == 75
    assert payload["persistent_cache_scope"] == "configured_model"


def test_memory_planner_cli_runs_from_a_clean_checkout_environment() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plan_expert_memory.py"),
            "--model",
            "glm52-q4",
            "--memory-limit-gib",
            "128",
            "--context-tokens",
            "131072",
            "--runtime-reserve-gib",
            "16",
        ],
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model"]["key"] == "glm52-q4"
    assert payload["plan"]["persistent_slots_per_layer"] == 60
    assert payload["plan"]["fits_fixed"] is True
    assert payload["plan"]["context_tokens"] == 131072
    assert payload["plan"]["transient_slots"] == 8
    assert payload["plan"]["expert_cache_limit_bytes"] is None
    assert payload["plan"]["fixed_bytes"] < payload["plan"]["accounted_bytes"]
    assert payload["plan"]["accounted_bytes"] <= payload["plan"]["memory_limit_bytes"]


def test_memory_planner_cli_returns_json_and_exit_two_for_fixed_deficit() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plan_expert_memory.py"),
            "--model",
            "glm52-q4",
            "--memory-limit-gib",
            "1",
            "--context-tokens",
            "0",
            "--runtime-reserve-gib",
            "0",
        ],
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["plan"]["fits_fixed"] is False
    assert payload["plan"]["persistent_slots_per_layer"] == 0
    assert payload["plan"]["unallocated_bytes"] < 0
