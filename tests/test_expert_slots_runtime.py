from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mtplx.expert_io import (
    ExpertIOCancelled,
    ExpertIOIntegrityError,
    ExpertIOShortRead,
    PositionalExpertReader,
)
from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
    save_expert_manifest,
)
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
    apply_mlx_memory_cap,
    partition_route_waves,
    reconcile_mlx_memory_cap,
)
from mtplx.expert_slots import ExpertSlotError, ExpertSlotPool
from mtplx.expert_streaming import (
    LayerExpertSlotBank,
    RoutePlan,
    RoutingPhase,
    SlotLoad,
)
from mtplx.expert_streaming_models import ExpertStreamingModelSpec, plan_expert_memory


COMPONENTS = (
    ("gate_proj.weight", 2_048, "U32", (64, 8)),
    ("gate_proj.scales", 128, "BF16", (64, 1)),
    ("gate_proj.biases", 128, "BF16", (64, 1)),
    ("up_proj.weight", 2_048, "U32", (64, 8)),
    ("up_proj.scales", 128, "BF16", (64, 1)),
    ("up_proj.biases", 128, "BF16", (64, 1)),
    ("down_proj.weight", 2_048, "U32", (64, 8)),
    ("down_proj.scales", 128, "BF16", (64, 1)),
    ("down_proj.biases", 128, "BF16", (64, 1)),
)


def _spec() -> ExpertStreamingModelSpec:
    record_bytes = sum(item[1] for item in COMPONENTS)
    return ExpertStreamingModelSpec(
        key="tiny-q4",
        display_name="Tiny Q4",
        source_model="test/tiny",
        source_revision="source-revision",
        quant_model="test/tiny-q4",
        quant_revision="quant-revision",
        total_tensor_bytes=2 * record_bytes + 1,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="bfloat16",
        router_matmul_dtype="float32",
        router_bytes=0,
        kv_bytes_per_token=16,
        mtp_layer_index=2,
        mtp_included=False,
    )


def _artifact(
    tmp_path: Path,
) -> tuple[Path, ExpertStreamingModelSpec, ExpertManifest, dict[int, bytes]]:
    root = tmp_path / "artifact"
    root.mkdir()
    spec = _spec()
    raw = bytearray()
    records: list[ExpertRecord] = []
    expected: dict[int, bytes] = {}
    for expert in range(spec.expert_count):
        segments: list[TensorSegment] = []
        record_payload = bytearray()
        for component_index, (component, length, dtype, shape) in enumerate(COMPONENTS):
            payload = bytes([expert * 16 + component_index + 1]) * length
            offset = len(raw)
            raw.extend(payload)
            record_payload.extend(payload)
            segments.append(
                TensorSegment(
                    component=component,
                    tensor=f"model.layers.1.mlp.switch_mlp.{component}",
                    shard="source.bin",
                    offset=offset,
                    length=length,
                    dtype=dtype,
                    shape=shape,
                )
            )
        expected[expert] = bytes(record_payload)
        records.append(
            ExpertRecord(
                layer=1,
                expert=expert,
                logical_bytes=len(record_payload),
                segments=tuple(segments),
                sha256=hashlib.sha256(record_payload).hexdigest(),
            )
        )
    resident_offset = len(raw)
    raw.append(123)
    (root / "source.bin").write_bytes(raw)
    manifest = ExpertManifest(
        model_key=spec.key,
        source_repo=spec.quant_model,
        source_revision=spec.quant_revision,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        artifact_tensor_bytes=spec.total_tensor_bytes,
        resident_tensor_bytes=1,
        routed_expert_bytes=spec.routed_expert_bytes,
        shards=(
            ShardInfo(
                name="source.bin",
                size=len(raw),
                header_bytes=1,
                header_sha256="fixture-header",
            ),
        ),
        resident_tensors=(
            ResidentTensor(
                tensor="model.norm.flag",
                shard="source.bin",
                offset=resident_offset,
                length=1,
                dtype="U8",
                shape=(1,),
            ),
        ),
        records=tuple(records),
    ).with_digest()
    manifest.validate_structure()
    return root, spec, manifest, expected


def _plan(spec: ExpertStreamingModelSpec):
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    return plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        context_tokens=0,
        runtime_reserve_bytes=0,
    )


def test_positional_reader_fills_and_hashes_source_record(tmp_path: Path) -> None:
    root, _spec_value, manifest, expected = _artifact(tmp_path)
    destination = bytearray(manifest.records[0].logical_bytes)

    with PositionalExpertReader(root, max_open_files=1, use_native=False) as reader:
        digest = reader.read_record_into(manifest, manifest.records[0], destination)
        metrics = reader.metrics.as_dict()

    assert bytes(destination) == expected[0]
    assert digest == manifest.records[0].sha256
    assert metrics["source_record_requests"] == 1
    assert metrics["read_operations"] == 9
    assert metrics["read_bytes"] == len(destination)
    assert metrics["open_files_peak"] == 1


def test_reader_cancellation_integrity_and_short_read_fail_closed(
    tmp_path: Path,
) -> None:
    root, _spec_value, manifest, _expected = _artifact(tmp_path)
    destination = bytearray(manifest.records[0].logical_bytes)
    cancel = threading.Event()
    cancel.set()
    with PositionalExpertReader(root, use_native=False) as reader:
        with pytest.raises(ExpertIOCancelled):
            reader.read_record_into(
                manifest,
                manifest.records[0],
                destination,
                cancel_event=cancel,
            )

    corrupt = bytearray((root / "source.bin").read_bytes())
    corrupt[0] ^= 0xFF
    (root / "source.bin").write_bytes(corrupt)
    with PositionalExpertReader(root, use_native=False) as reader:
        with pytest.raises(ExpertIOIntegrityError):
            reader.read_record_into(manifest, manifest.records[0], destination)

    (root / "source.bin").write_bytes(b"short")
    with PositionalExpertReader(root, use_native=False) as reader:
        with pytest.raises(ExpertIOShortRead):
            reader.read_record_into(manifest, manifest.records[0], destination)


def test_slot_pool_loads_hits_replaces_and_preserves_component_views(
    tmp_path: Path,
) -> None:
    root, spec, manifest, expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    bank = LayerExpertSlotBank(
        expert_count=2,
        persistent_slots=1,
        transient_slots=1,
    )
    try:
        first_plan = bank.plan([0], phase="decode")
        first = pool.ensure_route(1, first_plan)
        assert bytes(first.bindings[0].buffer) == expected[0]
        assert len(first.bindings[0].component_view("gate_proj.weight")) == 2_048
        first_generation = first.generations[0]
        first.release(synchronize=False)

        hit_plan = bank.plan([0], phase="decode")
        hit = pool.ensure_route(1, hit_plan)
        assert hit.plan.loads == ()
        assert hit.generations[0] == first_generation
        hit.release(synchronize=False)

        cold_plan = bank.plan([1], phase="decode")
        assert all(not load.persistent for load in cold_plan.loads)
        cold = pool.ensure_route(1, cold_plan)
        assert bytes(cold.bindings[0].buffer) == expected[1]
        cold.release(synchronize=False)

        admitted_plan = bank.plan([1], phase="decode")
        assert any(load.persistent for load in admitted_plan.loads)
        admitted = pool.ensure_route(1, admitted_plan)
        assert admitted.generations[0] > first_generation
        admitted.release(synchronize=False)

        snapshot = pool.snapshot()
        assert snapshot["allocated_bytes"] == 2 * spec.expert_record_bytes
        assert snapshot["metrics"]["generation_replacements"] == 1
        assert snapshot["io"]["record_requests"] == 3
    finally:
        pool.close()


def _manual_plan(expert: int, slot: int) -> RoutePlan:
    return RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(expert,),
        slots=(slot,),
        hits=(),
        misses=(expert,),
        loads=(SlotLoad(expert=expert, slot=slot, persistent=False),),
        evictions=(),
    )


def test_transient_slot_waits_for_pinned_generation_before_overwrite(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    transient_slot = plan.slots_per_layer
    first = pool.ensure_route(1, _manual_plan(0, transient_slot))
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(pool.ensure_route, 1, _manual_plan(1, transient_slot))
        time.sleep(0.05)
        assert pending.done() is False
        first.release(synchronize=False)
        second = pending.result(timeout=2)
    assert second.bindings[0].expert == 1
    second.release(synchronize=False)
    pool.close()


def test_runtime_handles_kv_admission_routes_waves_and_reset(tmp_path: Path) -> None:
    root, spec, manifest, expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=plan.total_limit_bytes + 4 * spec.kv_bytes_per_token,
        max_live_kv_tokens=4,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
        max_inflight_io_bytes=spec.expert_record_bytes,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        admission = runtime.admit_kv_tokens(4)
        with pytest.raises(ExpertStreamingConfigurationError, match="exceeds"):
            runtime.admit_kv_tokens(1)
        admission.release()

        ready = runtime.ensure_route(1, [0], phase="decode")
        assert bytes(ready.bindings[0].buffer) == expected[0]
        ready.release(synchronize=False)

        waves = runtime.route_waves([0, 1, 0, 1])
        assert len(waves) == 2
        assert waves[0].positions == (0, 2)
        assert waves[1].positions == (1, 3)
        snapshot = runtime.snapshot(mx_module=object())
        assert snapshot["cache"]["expert_requests"] == 1
        assert snapshot["slots"]["pins"] == 0
        runtime.reset()
        assert runtime.snapshot(mx_module=object())["slots"]["states"]["empty"] == 2
    finally:
        runtime.close()


def test_runtime_rolls_back_policy_mapping_after_integrity_failure(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=plan.total_limit_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    payload = bytearray((root / "source.bin").read_bytes())
    payload[0] ^= 0xFF
    (root / "source.bin").write_bytes(payload)
    try:
        with pytest.raises(ExpertSlotError, match="hash mismatch"):
            runtime.ensure_route(1, [0], phase="decode")
        assert runtime._banks[1].occupancy == 0
        assert runtime.snapshot(mx_module=object())["slots"]["states"]["empty"] == 2
    finally:
        runtime.close()


def test_memory_cap_reconciliation_and_fake_mlx_application() -> None:
    spec = _spec()
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=100_000,
        context_tokens=0,
        runtime_reserve_bytes=10_000,
        io_staging_bytes=5_000,
    )
    expected = 85_000
    assert reconcile_mlx_memory_cap(plan, env={}) == expected
    with pytest.raises(ExpertStreamingConfigurationError, match="conflicts"):
        reconcile_mlx_memory_cap(plan, env={"MTPLX_MEMORY_LIMIT_BYTES": "84kb"})

    class FakeMX:
        value = 0

        @classmethod
        def set_memory_limit(cls, value: int) -> None:
            cls.value = value

    env: dict[str, str] = {}
    report = apply_mlx_memory_cap(plan, mx_module=FakeMX, env=env)
    assert report == {"applied": True, "limit": expected}
    assert FakeMX.value == expected
    assert env["MTPLX_MEMORY_LIMIT_BYTES"] == str(expected)


def test_partition_route_waves_rejects_non_integral_ids() -> None:
    with pytest.raises(TypeError, match="exact integers"):
        partition_route_waves([0, 1.5], max_unique_experts=1)
