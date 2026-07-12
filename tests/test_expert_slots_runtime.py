from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_io import (
    ExpertIOCancelled,
    ExpertIOError,
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
    PendingSplitRoute,
    apply_mlx_memory_cap,
    partition_route_waves,
    reconcile_mlx_memory_cap,
)
from mtplx.expert_slots import ExpertSlotError, ExpertSlotPool, ReadyRoute
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


class _CloseTrackingResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ObservedLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquire_attempted = threading.Event()

    def acquire(self, *args, **kwargs) -> bool:
        self.acquire_attempted.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


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
    *,
    expert_count: int = 2,
) -> tuple[Path, ExpertStreamingModelSpec, ExpertManifest, dict[int, bytes]]:
    root = tmp_path / "artifact"
    root.mkdir()
    spec = _spec()
    if expert_count != spec.expert_count:
        spec = replace(
            spec,
            expert_count=expert_count,
            total_tensor_bytes=expert_count * spec.expert_record_bytes + 1,
        )
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


def _global_artifact(
    tmp_path: Path,
    *,
    expert_count: int = 2,
) -> tuple[
    Path,
    ExpertStreamingModelSpec,
    ExpertManifest,
    dict[tuple[int, int], bytes],
]:
    root = tmp_path / "global-artifact"
    root.mkdir()
    record_bytes = sum(item[1] for item in COMPONENTS)
    spec = replace(
        _spec(),
        key="tiny-global-q4",
        display_name="Tiny Global Q4",
        total_tensor_bytes=2 * expert_count * record_bytes + 1,
        total_layers=3,
        routed_layer_count=2,
        expert_count=expert_count,
        mtp_layer_index=3,
    )
    raw = bytearray()
    records: list[ExpertRecord] = []
    expected: dict[tuple[int, int], bytes] = {}
    for layer in spec.routed_layer_indices:
        for expert in range(spec.expert_count):
            segments: list[TensorSegment] = []
            record_payload = bytearray()
            for component_index, (component, length, dtype, shape) in enumerate(
                COMPONENTS
            ):
                payload = (
                    bytes([layer * 64 + expert * 16 + component_index + 1]) * length
                )
                offset = len(raw)
                raw.extend(payload)
                record_payload.extend(payload)
                segments.append(
                    TensorSegment(
                        component=component,
                        tensor=(f"model.layers.{layer}.mlp.switch_mlp.{component}"),
                        shard="source.bin",
                        offset=offset,
                        length=length,
                        dtype=dtype,
                        shape=shape,
                    )
                )
            expected[(layer, expert)] = bytes(record_payload)
            records.append(
                ExpertRecord(
                    layer=layer,
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


def _global_policy_state(runtime: ExpertStreamingRuntime) -> dict[str, object]:
    bank = runtime._global_bank
    assert bank is not None
    return {
        "decode_epoch": bank._decode_epoch,
        "slot_to_key": tuple(bank._slot_to_key),
        "key_to_slot": dict(bank._key_to_slot),
        "directory": tuple(
            sorted(
                (key, entry.slot, entry.generation, entry.state)
                for key, entry in bank._directory.items()
            )
        ),
        "slot_generations": tuple(bank._slot_generations),
        "free_slots": tuple(bank._free_slots),
        "free_slot_set": set(bank._free_slot_set),
        "lru": tuple(bank._lru.items()),
        "history": tuple(
            sorted(
                (key, value.score, value.score_epoch, value.last_used)
                for key, value in bank._history.items()
            )
        ),
        "layer_occupancy": dict(bank._layer_occupancy),
        "evictions": bank._evictions,
        "cross_layer_evictions": bank._cross_layer_evictions,
        "prefill_seed_candidates": tuple(
            (layer, frozenset(experts))
            for layer, experts in sorted(bank._prefill_seed_candidates.items())
        ),
    }


def _layer_policy_state(
    runtime: ExpertStreamingRuntime, layer: int
) -> dict[str, object]:
    bank = runtime._banks[layer]
    return {
        "decode_epoch": bank._decode_epoch,
        "slot_to_expert": tuple(bank._slot_to_expert),
        "expert_to_slot": dict(bank._expert_to_slot),
        "history": tuple(
            (value.score, value.score_epoch, value.last_used) for value in bank._history
        ),
        "prefill_seed_candidates": frozenset(bank._prefill_seed_candidates),
    }


def _plan(spec: ExpertStreamingModelSpec):
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    return plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        context_tokens=0,
        runtime_reserve_bytes=0,
    )


def _global_plan(spec: ExpertStreamingModelSpec, *, persistent_slots: int = 2):
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    return plan_expert_memory(
        spec,
        total_limit_bytes=fixed + persistent_slots * spec.expert_record_bytes,
        context_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
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


def test_native_backend_failure_is_normalized_and_counted(tmp_path: Path) -> None:
    root, _spec_value, manifest, _expected = _artifact(tmp_path)
    destination = bytearray(manifest.records[0].logical_bytes)
    reader = PositionalExpertReader(root, use_native=False)

    def fail_native(_fd: int, _offset: int, _destination: memoryview) -> int:
        raise RuntimeError("pread failed: injected EIO")

    reader._native_read_into = fail_native
    try:
        with pytest.raises(ExpertIOError, match="native positional read failed"):
            reader.read_record_into(manifest, manifest.records[0], destination)
        metrics = reader.metrics.as_dict()
        assert metrics["io_errors"] == 1
        assert metrics["read_bytes"] == 0
    finally:
        reader.close()


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


def test_completion_fence_holds_generation_until_consumer_finishes(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    transient_slot = plan.slots_per_layer
    completed = threading.Event()
    first = pool.ensure_route(1, _manual_plan(0, transient_slot))
    first_generation = first.generations[0]

    assert first.defer_bindings_until(first.bindings, completed.wait) is True
    first.release(synchronize=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        replacement = executor.submit(
            pool.ensure_route,
            1,
            _manual_plan(1, transient_slot),
        )
        time.sleep(0.05)
        assert replacement.done() is False
        slot = pool._physical(1, transient_slot)
        with slot.condition:
            assert slot.pins == 1
        completed.set()
        second = replacement.result(timeout=2)

    assert second.bindings[0].expert == 1
    assert second.generations[0] > first_generation
    second.release(synchronize=False)
    snapshot = pool.snapshot()
    assert snapshot["pins"] == 0
    assert snapshot["metrics"]["completion_fences"] == 1
    assert snapshot["metrics"]["completion_fence_slots"] == 1
    pool.close()


def test_completion_fence_registration_rolls_back_non_runtime_submit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    ready = pool.ensure_route(1, _manual_plan(0, plan.slots_per_layer))

    def reject_submit(*_args, **_kwargs):
        raise ValueError("injected non-runtime completion submit rejection")

    monkeypatch.setattr(pool._completion_executor, "submit", reject_submit)
    try:
        with pytest.raises(ValueError, match="non-runtime completion submit"):
            ready.defer_bindings_until(ready.bindings, lambda: None)
        assert ready._scheduled_slots == set()

        ready.release(synchronize=False)
        slot = pool._physical(1, plan.slots_per_layer)
        with slot.condition:
            assert slot.pins == 0
        assert pool.metrics.as_dict()["active_routes"] == 0
    finally:
        with ready._release_lock:
            ready._scheduled_slots.clear()
        ready.release(synchronize=False)
        pool.close(timeout=2)


def test_completion_fence_release_waits_for_concurrent_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    ready = pool.ensure_route(1, _manual_plan(0, plan.slots_per_layer))
    registration_submitted = threading.Event()
    finish_registration = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("concurrent registration fence failure")
    original_submit = pool._submit_completion_fence

    def wait_then_fail() -> None:
        assert fail_completion.wait(timeout=2)
        raise fence_error

    def pause_after_submit(*args, **kwargs):
        future = original_submit(*args, **kwargs)
        registration_submitted.set()
        assert finish_registration.wait(timeout=2)
        return future

    monkeypatch.setattr(pool, "_submit_completion_fence", pause_after_submit)
    with ThreadPoolExecutor(max_workers=2) as executor:
        registration = executor.submit(
            ready.defer_bindings_until,
            ready.bindings,
            wait_then_fail,
        )
        assert registration_submitted.wait(timeout=2)
        release = executor.submit(ready.release)
        try:
            with pytest.raises(TimeoutError):
                release.result(timeout=0.05)
            finish_registration.set()
            assert registration.result(timeout=2) is True
            fail_completion.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                release.result(timeout=2)
            assert exc.value.__cause__ is fence_error
        finally:
            finish_registration.set()
            fail_completion.set()
            try:
                registration.result(timeout=2)
            except BaseException:
                pass
            try:
                release.result(timeout=2)
            except BaseException:
                pass

    slot = pool._physical(1, plan.slots_per_layer)
    with slot.condition:
        assert slot.pins == 0
    assert ready._scheduled_slots == set()
    assert pool.metrics.as_dict()["active_routes"] == 0
    with pytest.raises(ExpertSlotError, match="completion fence failed"):
        pool.close(timeout=2)


def test_completion_fence_failure_releases_pin_and_fails_next_route(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    transient_slot = plan.slots_per_layer
    first = pool.ensure_route(1, _manual_plan(0, transient_slot))

    def fail_completion() -> None:
        raise RuntimeError("injected Metal completion failure")

    first.defer_bindings_until(first.bindings, fail_completion)
    first.release(synchronize=False)
    pool._drain_completion_fences()

    slot = pool._physical(1, transient_slot)
    with slot.condition:
        assert slot.pins == 0
    assert pool.metrics.as_dict()["completion_fence_failures"] == 1
    with pytest.raises(ExpertSlotError, match="completion fence failed"):
        pool.ensure_route(1, _manual_plan(1, transient_slot))
    with pytest.raises(ExpertSlotError, match="completion fence failed"):
        pool.close()


def test_completion_fence_failure_stops_replacement_already_waiting_on_pin(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    transient_slot = plan.slots_per_layer
    first = pool.ensure_route(1, _manual_plan(0, transient_slot))
    slot = pool._physical(1, transient_slot)
    first_generation = first.generations[0]
    read_bytes = reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("injected Metal completion failure after pin wait")
    replacement_ready = None

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    try:
        first.defer_bindings_until(first.bindings, wait_then_fail)
        first.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            replacement = executor.submit(
                pool.ensure_route,
                1,
                _manual_plan(1, transient_slot),
            )
            deadline = time.monotonic() + 2
            while pool.metrics.as_dict()["pin_waits"] == 0:
                assert time.monotonic() < deadline, "replacement did not wait on pin"
                time.sleep(0.001)
            fail_completion.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                replacement_ready = replacement.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.expert == 0
            assert slot.generation == first_generation
        assert reader.metrics.as_dict()["read_bytes"] == read_bytes
    finally:
        fail_completion.set()
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            pool.close(timeout=2)
        except ExpertSlotError:
            pass


def test_runtime_completion_fence_failure_preserves_waiting_victim(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    bank = runtime._banks[1]
    policy_before = (
        bank.resident_experts,
        bank._decode_epoch,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in bank._history
        ),
    )
    slot = runtime.slots._physical(1, 0)
    generation = first.generations[0]
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("runtime victim fence failure")
    replacement_ready = None

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    try:
        first.defer_bindings_until(first.bindings, wait_then_fail)
        first.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            replacement = executor.submit(
                runtime.ensure_route,
                1,
                [1],
                phase="decode",
            )
            deadline = time.monotonic() + 2
            while runtime.slots.metrics.as_dict()["pin_waits"] == 0:
                assert time.monotonic() < deadline, "replacement did not wait on pin"
                time.sleep(0.001)
            fail_completion.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                replacement_ready = replacement.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        assert (
            bank.resident_experts,
            bank._decode_epoch,
            tuple(
                (history.score, history.score_epoch, history.last_used)
                for history in bank._history
            ),
        ) == policy_before
    finally:
        fail_completion.set()
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_failure_while_waiting_runtime_lock_skips_policy_plan(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    bank = runtime._banks[1]
    policy_before = (
        bank.resident_experts,
        bank._decode_epoch,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in bank._history
        ),
    )
    slot = runtime.slots._physical(1, 0)
    generation = first.generations[0]
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("fence failure while waiting on runtime layer lock")
    observed_lock = _ObservedLock()
    observed_lock._lock.acquire()
    runtime._layer_locks[1] = observed_lock
    replacement_ready = None
    lock_held = True

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    try:
        first.defer_bindings_until(first.bindings, wait_then_fail)
        first.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                runtime.ensure_route,
                1,
                [1],
                phase="decode",
            )
            assert observed_lock.acquire_attempted.wait(timeout=2)
            fail_completion.set()
            runtime.slots._drain_completion_fences()
            observed_lock._lock.release()
            lock_held = False
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                replacement_ready = pending.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        assert (
            bank.resident_experts,
            bank._decode_epoch,
            tuple(
                (history.score, history.score_epoch, history.last_used)
                for history in bank._history
            ),
        ) == policy_before
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        fail_completion.set()
        if lock_held:
            observed_lock._lock.release()
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_failure_after_layer_lock_precheck_blocks_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    bank = runtime._banks[1]
    policy_before = (
        bank.resident_experts,
        bank._decode_epoch,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in bank._history
        ),
    )
    slot = runtime.slots._physical(1, 0)
    generation = first.generations[0]
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    precheck_complete = threading.Event()
    fence_error = RuntimeError("fence failure while replacement waits on layer lock")
    replacement_ready = None
    layer_lock = runtime.slots._ensure_locks[1]
    lock_held = False

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    original_raise = runtime.slots._raise_completion_error

    def observe_precheck() -> None:
        precheck_complete.set()
        original_raise()

    monkeypatch.setattr(runtime.slots, "_raise_completion_error", observe_precheck)

    try:
        first.defer_bindings_until(first.bindings, wait_then_fail)
        first.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        layer_lock.acquire()
        lock_held = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            replacement = executor.submit(
                runtime.ensure_route,
                1,
                [1],
                phase="decode",
            )
            assert precheck_complete.wait(timeout=2)
            fail_completion.set()
            runtime.slots._drain_completion_fences()
            with slot.condition:
                assert slot.pins == 0
            layer_lock.release()
            lock_held = False
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                replacement_ready = replacement.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        assert (
            bank.resident_experts,
            bank._decode_epoch,
            tuple(
                (history.score, history.score_epoch, history.last_used)
                for history in bank._history
            ),
        ) == policy_before

        with pytest.raises(ExpertSlotError, match="completion fence failed") as visible:
            runtime.snapshot(mx_module=object())
        assert visible.value.__cause__ is fence_error
        with pytest.raises(ExpertSlotError, match="completion fence failed") as closed:
            runtime.close(timeout=2)
        assert closed.value.__cause__ is fence_error
        assert runtime.reader._closed is True
    finally:
        fail_completion.set()
        if lock_held:
            layer_lock.release()
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_failure_after_post_lock_check_blocks_all_hit_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    bank = runtime._banks[1]
    policy_before = (
        bank.resident_experts,
        bank._decode_epoch,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in bank._history
        ),
    )
    slot = runtime.slots._physical(1, 0)
    generation = first.generations[0]
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    entered_locked_route = threading.Event()
    continue_locked_route = threading.Event()
    fence_error = RuntimeError("fence failure after all-hit post-lock check")
    all_hit_ready = None

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    original_ensure_locked = runtime.slots._ensure_route_locked

    def block_after_post_lock_check(
        layer: int,
        route: RoutePlan,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
    ):
        entered_locked_route.set()
        assert continue_locked_route.wait(timeout=2)
        return original_ensure_locked(
            layer,
            route,
            cancel_event=cancel_event,
            deadline_ns=deadline_ns,
        )

    monkeypatch.setattr(
        runtime.slots, "_ensure_route_locked", block_after_post_lock_check
    )

    try:
        first.defer_bindings_until(first.bindings, wait_then_fail)
        first.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                runtime.try_all_hit_route,
                1,
                [0],
                phase="decode",
            )
            assert entered_locked_route.wait(timeout=2)
            fail_completion.set()
            runtime.slots._drain_completion_fences()
            with slot.condition:
                assert slot.pins == 0
            continue_locked_route.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                all_hit_ready = pending.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        assert (
            bank.resident_experts,
            bank._decode_epoch,
            tuple(
                (history.score, history.score_epoch, history.last_used)
                for history in bank._history
            ),
        ) == policy_before

        with pytest.raises(ExpertSlotError, match="completion fence failed") as visible:
            runtime.snapshot(mx_module=object())
        assert visible.value.__cause__ is fence_error
        with pytest.raises(ExpertSlotError, match="completion fence failed") as closed:
            runtime.close(timeout=2)
        assert closed.value.__cause__ is fence_error
        assert runtime.reader._closed is True
    finally:
        fail_completion.set()
        continue_locked_route.set()
        if all_hit_ready is not None:
            all_hit_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_all_hit_generic_pin_failure_rolls_back_policy_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    first.release(synchronize=False)
    bank = runtime._banks[1]
    policy_before = (
        bank._decode_epoch,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in bank._history
        ),
    )

    def reject_pin(*_args, **_kwargs):
        raise ValueError("injected all-hit pin rejection")

    monkeypatch.setattr(runtime.slots, "ensure_route", reject_pin)
    try:
        with pytest.raises(ValueError, match="all-hit pin rejection"):
            runtime.try_all_hit_route(1, [0], phase="decode")
        assert (
            bank._decode_epoch,
            tuple(
                (history.score, history.score_epoch, history.last_used)
                for history in bank._history
            ),
        ) == policy_before
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        runtime.close()


def test_global_runtime_success_publishes_ready_generation(tmp_path: Path) -> None:
    root, spec, manifest, expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _global_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
            cache_scope="global",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        first = runtime.ensure_route(1, [0], phase="decode")
        assert bytes(first.bindings[0].buffer) == expected[(1, 0)]
        bank = runtime._global_bank
        assert bank is not None
        entry = bank._directory[(1, 0)]
        assert entry.state == "ready"
        generation = entry.generation
        first.release(synchronize=False)

        second = runtime.ensure_route(1, [0], phase="decode")
        assert second.plan.hits == (0,)
        assert second.plan.loads == ()
        assert bank._directory[(1, 0)].generation == generation
        assert bank._directory[(1, 0)].state == "ready"
        second.release(synchronize=False)
        assert runtime.counters.as_dict()["route_calls"] == 2
        assert runtime.counters.as_dict()["expert_hits"] == 1
    finally:
        runtime.close()


def test_global_safe_fence_failure_restores_cross_layer_policy_exactly(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _global_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
            cache_scope="global",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    held = runtime.ensure_route(1, [0], phase="decode")
    other = runtime.ensure_route(2, [0], phase="decode")
    other.release(synchronize=False)
    slot = runtime.slots._physical(1, held.slots[0])
    generation = held.generations[0]
    policy_before = _global_policy_state(runtime)
    counters_before = (
        runtime.counters.as_dict(),
        runtime._layer_counters[2].as_dict(),
        runtime._phase_counters[RoutingPhase.DECODE].as_dict(),
    )
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("global cross-layer victim fence failure")
    replacement_ready = None

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    try:
        held.defer_bindings_until(held.bindings, wait_then_fail)
        held.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            replacement = executor.submit(
                runtime.ensure_route,
                2,
                [1],
                phase="decode",
            )
            deadline = time.monotonic() + 2
            while runtime.slots.metrics.as_dict()["pin_waits"] == 0:
                assert time.monotonic() < deadline, "global victim did not wait on pin"
                time.sleep(0.001)
            fail_completion.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                replacement_ready = replacement.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        assert _global_policy_state(runtime) == policy_before
        assert (
            runtime.counters.as_dict(),
            runtime._layer_counters[2].as_dict(),
            runtime._phase_counters[RoutingPhase.DECODE].as_dict(),
        ) == counters_before
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.layer == 1
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        fail_completion.set()
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_multi_load_prepare_failure_restores_earlier_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.slots.ensure_route(1, _manual_plan(0, 0))
    first.release(synchronize=False)
    held = runtime.slots.ensure_route(1, _manual_plan(1, 1))
    first_slot = runtime.slots._physical(1, 0)
    held_slot = runtime.slots._physical(1, 1)
    before = (
        (first_slot.state, first_slot.expert, first_slot.generation),
        (held_slot.state, held_slot.expert, held_slot.generation),
    )
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("multi-load preparation fence failure")

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    route = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(1, 0),
        slots=(0, 1),
        hits=(),
        misses=(1, 0),
        loads=(
            SlotLoad(expert=1, slot=0, persistent=True),
            SlotLoad(expert=0, slot=1, persistent=False),
        ),
        evictions=(),
    )
    monkeypatch.setattr(runtime, "_plan_route", lambda *_args, **_kwargs: route)

    try:
        held.defer_bindings_until(held.bindings, wait_then_fail)
        held.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                runtime.ensure_route,
                1,
                [1, 0],
                phase="decode",
            )
            deadline = time.monotonic() + 2
            while runtime.slots.metrics.as_dict()["pin_waits"] == 0:
                assert time.monotonic() < deadline, "second load did not wait on pin"
                time.sleep(0.001)
            fail_completion.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                pending.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        with first_slot.condition, held_slot.condition:
            after = (
                (first_slot.state, first_slot.expert, first_slot.generation),
                (held_slot.state, held_slot.expert, held_slot.generation),
            )
            assert after == before
            assert first_slot.pins == 0
            assert held_slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        fail_completion.set()
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_failure_after_prepare_stops_io_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    first = pool.ensure_route(1, _manual_plan(0, 0))
    target_slot = pool._physical(1, plan.slots_per_layer)
    read_bytes = reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    prepare_complete = threading.Event()
    continue_after_prepare = threading.Event()
    fence_error = RuntimeError("fence failure after slot preparation")

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    original_can_batch = pool._can_batch_component_sidecar

    def block_after_prepare(*args, **kwargs) -> bool:
        prepare_complete.set()
        assert continue_after_prepare.wait(timeout=2)
        return original_can_batch(*args, **kwargs)

    monkeypatch.setattr(pool, "_can_batch_component_sidecar", block_after_prepare)

    try:
        first.defer_bindings_until(first.bindings, wait_then_fail)
        first.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                pool.ensure_route,
                1,
                _manual_plan(1, plan.slots_per_layer),
            )
            assert prepare_complete.wait(timeout=2)
            with target_slot.condition:
                assert target_slot.state.value == "loading"
                assert target_slot.expert == 1
            fail_completion.set()
            pool._drain_completion_fences()
            continue_after_prepare.set()
            with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
                pending.result(timeout=2)

        assert exc.value.__cause__ is fence_error
        with target_slot.condition:
            assert target_slot.state.value == "empty"
            assert target_slot.layer is None
            assert target_slot.expert is None
            assert target_slot.generation == 0
            assert target_slot.pins == 0
        assert reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert pool.metrics.as_dict()["active_routes"] == 0
    finally:
        fail_completion.set()
        continue_after_prepare.set()
        try:
            pool.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_failure_is_visible_to_snapshot_and_close(
    tmp_path: Path,
) -> None:
    class CloseTrackingAllocator:
        backend = "test-close-tracking"

        def __init__(self) -> None:
            self.closed = False

        def __call__(self, size: int, _label: str) -> bytearray:
            return bytearray(size)

        def close(self) -> None:
            self.closed = True

    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    allocator = CloseTrackingAllocator()
    pool = ExpertSlotPool(
        spec,
        plan,
        manifest,
        reader,
        buffer_allocator=allocator,
    )
    transient_slot = plan.slots_per_layer
    first = pool.ensure_route(1, _manual_plan(0, transient_slot))
    fence_error = RuntimeError("sticky Metal completion failure")

    def fail_completion() -> None:
        raise fence_error

    try:
        first.defer_bindings_until(first.bindings, fail_completion)
        first.release(synchronize=False)

        with pytest.raises(ExpertSlotError, match="completion fence failed") as one:
            pool.snapshot()
        with pytest.raises(ExpertSlotError, match="completion fence failed") as two:
            pool.snapshot()
        assert one.value.__cause__ is fence_error
        assert two.value.__cause__ is fence_error

        with pytest.raises(ExpertSlotError, match="completion fence failed") as closed:
            pool.close(timeout=2)
        assert closed.value.__cause__ is fence_error
        assert reader._closed is True
        assert allocator.closed is True
        assert pool._closed is True
        assert pool._closing is False
        assert all(
            slot.state.value == "closed"
            for slot in (*pool._persistent.values(), *pool._transient)
        )
    finally:
        try:
            pool.close(timeout=2)
        except ExpertSlotError:
            pass


def test_completion_fence_synchronous_failure_releases_route_and_blocks_replacement(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    fence_error = RuntimeError("injected synchronous Metal fence failure")

    def fail_synchronize() -> None:
        raise fence_error

    pool = ExpertSlotPool(
        spec,
        plan,
        manifest,
        reader,
        device_synchronize=fail_synchronize,
    )
    transient_slot = plan.slots_per_layer
    ready = pool.ensure_route(1, _manual_plan(0, transient_slot))
    slot = pool._physical(1, transient_slot)
    generation = ready.generations[0]
    read_bytes = reader.metrics.as_dict()["read_bytes"]

    try:
        with pytest.raises(RuntimeError, match="synchronous Metal fence") as released:
            ready.release()
        assert released.value is fence_error
        with slot.condition:
            assert slot.pins == 0
            assert slot.expert == 0
            assert slot.generation == generation
        assert pool.metrics.as_dict()["active_routes"] == 0

        with pytest.raises(ExpertSlotError, match="completion fence failed") as blocked:
            pool.ensure_route(1, _manual_plan(1, transient_slot))
        assert blocked.value.__cause__ is fence_error
        with slot.condition:
            assert slot.expert == 0
            assert slot.generation == generation
        assert reader.metrics.as_dict()["read_bytes"] == read_bytes

        with pytest.raises(ExpertSlotError, match="completion fence failed") as closed:
            pool.close(timeout=2)
        assert closed.value.__cause__ is fence_error
        assert reader._closed is True
    finally:
        with slot.condition:
            leaked_pin = slot.pins > 0
        if leaked_pin:
            ready._finish_slots((slot,))
        try:
            pool.close(timeout=2)
        except ExpertSlotError:
            pass


def test_slot_pool_close_timeout_blocks_admission_and_can_be_retried(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    transient_slot = plan.slots_per_layer
    active = pool.ensure_route(1, _manual_plan(0, transient_slot))

    try:
        with pytest.raises(TimeoutError, match="active expert routes"):
            pool.close(timeout=0)
        assert pool._closed is False
        assert pool._closing is True
        with pytest.raises(ExpertSlotError, match="closing"):
            pool.ensure_route(1, _manual_plan(1, transient_slot))

        active.release(synchronize=False)
        pool.close(timeout=2)
        assert pool._closed is True
        assert pool._closing is False
        assert reader._closed is True
    finally:
        active.release(synchronize=False)
        try:
            pool.close(timeout=2)
        except ExpertSlotError:
            pass


def test_slot_pool_concurrent_close_lock_honors_timeout(tmp_path: Path) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    plan = _plan(spec)
    reader = PositionalExpertReader(root, use_native=False)
    pool = ExpertSlotPool(spec, plan, manifest, reader)
    active = pool.ensure_route(1, _manual_plan(0, plan.slots_per_layer))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_close = executor.submit(pool.close)
        deadline = time.monotonic() + 2
        while not pool._closing:
            assert time.monotonic() < deadline, (
                "first close did not enter draining state"
            )
            time.sleep(0.001)
        started = time.monotonic()
        second_close = executor.submit(pool.close, timeout=0.01)
        try:
            with pytest.raises(TimeoutError, match="close.*progress|deadline"):
                second_close.result(timeout=0.2)
            assert time.monotonic() - started < 0.2
        finally:
            active.release(synchronize=False)
            first_close.result(timeout=2)
            try:
                second_close.result(timeout=2)
            except TimeoutError:
                pass

    pool.close(timeout=2)
    assert pool._closed is True
    assert reader._closed is True


def test_runtime_close_timeout_is_retryable_after_route_release(
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
    active = runtime.ensure_route(1, [0], phase="decode")
    mapped = _CloseTrackingResource()
    runtime._mapped_expert_store = mapped

    try:
        with pytest.raises(TimeoutError, match="active expert routes"):
            runtime.close(timeout=0)
        assert runtime._closed is False
        assert runtime._closing is True
        assert runtime._split_executor._shutdown is False
        assert mapped.closed is False
        assert runtime._mapped_expert_store is mapped
        with pytest.raises(ExpertSlotError, match="closing"):
            runtime.ensure_route(1, [1], phase="decode")

        active.release(synchronize=False)
        runtime.close(timeout=2)
        assert runtime._closed is True
        assert runtime._closing is False
        assert runtime._split_executor._shutdown is True
        assert mapped.closed is True
        assert runtime._mapped_expert_store is None
        assert runtime.reader._closed is True
        assert all(
            slot.state.value == "closed"
            for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient)
        )
    finally:
        active.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        finally:
            if not runtime.reader._closed:
                runtime.slots.close(timeout=2)


def test_runtime_concurrent_close_lock_honors_timeout(tmp_path: Path) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    active = runtime.ensure_route(1, [0], phase="decode")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_close = executor.submit(runtime.close)
        deadline = time.monotonic() + 2
        while not runtime._closing:
            assert time.monotonic() < deadline, (
                "first close did not enter draining state"
            )
            time.sleep(0.001)
        started = time.monotonic()
        second_close = executor.submit(runtime.close, timeout=0.01)
        try:
            with pytest.raises(TimeoutError, match="close.*progress|deadline"):
                second_close.result(timeout=0.2)
            assert time.monotonic() - started < 0.2
        finally:
            active.release(synchronize=False)
            first_close.result(timeout=2)
            try:
                second_close.result(timeout=2)
            except TimeoutError:
                pass

    runtime.close(timeout=2)
    assert runtime._closed is True
    assert runtime.reader._closed is True


def test_runtime_close_timeout_does_not_wait_for_running_split_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    mapped = _CloseTrackingResource()
    runtime._mapped_expert_store = mapped
    read_started = threading.Event()
    finish_read = threading.Event()
    original_read = runtime.reader.read_record_into

    def blocking_read(*args, **kwargs):
        read_started.set()
        assert finish_read.wait(timeout=2)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(runtime.reader, "read_record_into", blocking_read)
    pending = runtime.begin_split_route(1, [0], phase="decode")
    assert read_started.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        started = time.monotonic()
        closing = executor.submit(runtime.close, timeout=0.01)
        try:
            with pytest.raises(TimeoutError, match="active expert routes"):
                closing.result(timeout=0.2)
            assert time.monotonic() - started < 0.2
            assert runtime._closed is False
            assert runtime._closing is True
            assert runtime._split_executor._shutdown is False
            assert mapped.closed is False
            assert runtime._mapped_expert_store is mapped
        finally:
            finish_read.set()
            pending.close()
            try:
                closing.result(timeout=2)
            except TimeoutError:
                pass

    runtime.close(timeout=2)
    assert runtime._closed is True
    assert runtime._split_executor._shutdown is True
    assert mapped.closed is True
    assert runtime._mapped_expert_store is None
    assert runtime.reader._closed is True


def test_runtime_finite_close_does_not_wait_for_preadmission_split_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    mapped = _CloseTrackingResource()
    runtime._mapped_expert_store = mapped
    worker_started = threading.Event()
    release_worker = threading.Event()
    original_ensure = runtime.slots.ensure_route_part

    def block_before_admission(*args, **kwargs):
        worker_started.set()
        assert release_worker.wait(timeout=2)
        return original_ensure(*args, **kwargs)

    monkeypatch.setattr(runtime.slots, "ensure_route_part", block_before_admission)
    pending = runtime.begin_split_route(1, [0], phase="decode")
    assert worker_started.wait(timeout=2)
    assert runtime.slots.metrics.as_dict()["active_routes"] == 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        started = time.monotonic()
        closing = executor.submit(runtime.close, timeout=0.01)
        try:
            closing.result(timeout=0.2)
            assert time.monotonic() - started < 0.2
            assert runtime._closed is True
            assert runtime._closing is False
            assert runtime.slots._closed is True
            assert runtime._split_executor._shutdown is True
            assert mapped.closed is True
            assert runtime._mapped_expert_store is None
            assert runtime.reader._closed is True
            assert all(
                slot.state.value == "closed"
                for slot in (
                    *runtime.slots._persistent.values(),
                    *runtime.slots._transient,
                )
            )
        finally:
            release_worker.set()
            pending.close()
            closing.result(timeout=2)

    runtime.close(timeout=2)


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


def test_runtime_snapshot_splits_cache_counters_by_phase(tmp_path: Path) -> None:
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
    try:
        first = runtime.ensure_route(1, [0], phase="decode")
        first.release(synchronize=False)
        second = runtime.ensure_route(1, [0], phase="decode")
        second.release(synchronize=False)
        third = runtime.ensure_route(1, [1], phase="prefill")
        third.release(synchronize=False)

        snapshot = runtime.snapshot(mx_module=object())
        by_phase = snapshot["cache_by_phase"]
        assert set(by_phase) == {"prefill", "decode"}
        decode = by_phase["decode"]
        prefill = by_phase["prefill"]
        assert decode["route_calls"] == 2
        assert decode["expert_hits"] == 1
        assert decode["expert_misses"] == 1
        assert prefill["route_calls"] == 1
        assert prefill["expert_hits"] == 0
        assert prefill["expert_misses"] == 1
        aggregate = snapshot["cache"]
        for key in aggregate:
            if key == "hit_rate":
                continue
            assert aggregate[key] == decode[key] + prefill[key]

        # The split-route observation path must feed the same phase buckets.
        runtime.reset()
        assert all(
            counters["route_calls"] == 0
            for counters in runtime.snapshot(mx_module=object())[
                "cache_by_phase"
            ].values()
        )
        with runtime.begin_split_route(1, [0], phase="decode") as pending:
            pending.finish_misses()
        split_snapshot = runtime.snapshot(mx_module=object())
        assert split_snapshot["cache_by_phase"]["decode"]["route_calls"] == 1
        assert split_snapshot["cache_by_phase"]["decode"]["expert_misses"] == 1
        assert split_snapshot["cache_by_phase"]["prefill"]["route_calls"] == 0
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


def test_runtime_generic_io_failure_does_not_restore_overwritten_victim(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    first_generation = first.generations[0]
    first.release(synchronize=False)
    corrupt_offset = manifest.records[1].segments[0].offset
    payload = bytearray((root / "source.bin").read_bytes())
    payload[corrupt_offset] ^= 0xFF
    (root / "source.bin").write_bytes(payload)

    try:
        with pytest.raises(ExpertSlotError, match="hash mismatch"):
            runtime.ensure_route(1, [1], phase="decode")

        bank = runtime._banks[1]
        slot = runtime.slots._physical(1, 0)
        assert bank.resident_experts == ()
        with slot.condition:
            assert slot.state.value == "empty"
            assert slot.layer is None
            assert slot.expert is None
            assert slot.generation > first_generation
            assert slot.pins == 0
    finally:
        runtime.close()


def test_layer_first_io_submit_rejection_restores_policy_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    first.release(synchronize=False)
    slot = runtime.slots._physical(1, 0)
    generation = slot.generation
    policy_before = _layer_policy_state(runtime, 1)
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]

    def reject_submit(*_args, **_kwargs):
        raise ValueError("injected first expert I/O submit rejection")

    monkeypatch.setattr(runtime.slots._executor, "submit", reject_submit)
    try:
        with pytest.raises(ValueError, match="first expert I/O submit rejection"):
            runtime.ensure_route(1, [1], phase="decode")

        assert _layer_policy_state(runtime, 1) == policy_before
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.layer == 1
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        runtime.close()


def test_global_first_io_submit_rejection_restores_policy_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _global_plan(spec, persistent_slots=1)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
            cache_scope="global",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    first.release(synchronize=False)
    slot = runtime.slots._physical(1, 0)
    generation = slot.generation
    policy_before = _global_policy_state(runtime)
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]

    def reject_submit(*_args, **_kwargs):
        raise ValueError("injected first global expert I/O submit rejection")

    monkeypatch.setattr(runtime.slots._executor, "submit", reject_submit)
    try:
        with pytest.raises(
            ValueError, match="first global expert I/O submit rejection"
        ):
            runtime.ensure_route(2, [1], phase="decode")

        assert _global_policy_state(runtime) == policy_before
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.layer == 1
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        runtime.close()


def test_partial_io_submit_rejection_cleans_every_prepared_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    original_submit = runtime.slots._executor.submit
    first_started = threading.Event()
    release_first = threading.Event()
    second_rejected = threading.Event()
    submit_count = 0

    def controlled_submit(fn, *args, **kwargs):
        nonlocal submit_count
        submit_count += 1
        if submit_count == 1:

            def gated_first():
                first_started.set()
                assert release_first.wait(timeout=2)
                return fn(*args, **kwargs)

            return original_submit(gated_first)
        second_rejected.set()
        raise ValueError("injected second expert I/O submit rejection")

    monkeypatch.setattr(runtime.slots._executor, "submit", controlled_submit)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                runtime.ensure_route,
                1,
                [0, 1],
                phase="decode",
            )
            assert first_started.wait(timeout=2)
            assert second_rejected.wait(timeout=2)
            try:
                assert not pending.done(), "route returned before accepted I/O drained"
            finally:
                release_first.set()
            with pytest.raises(ValueError, match="second expert I/O submit rejection"):
                pending.result(timeout=2)

        slots = (
            runtime.slots._physical(1, 0),
            runtime.slots._physical(1, plan.slots_per_layer),
        )
        for slot in slots:
            with slot.condition:
                assert slot.state.value == "empty"
                assert slot.layer is None
                assert slot.expert is None
                assert slot.pins == 0
        assert runtime._banks[1].resident_experts == ()
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0

        monkeypatch.setattr(runtime.slots._executor, "submit", original_submit)
        retry = runtime.ensure_route(1, [0, 1], phase="decode")
        retry.release(synchronize=False)
    finally:
        release_first.set()
        runtime.close(timeout=2)


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

    ordered = partition_route_waves(
        [3, 1, 2, 3, 0],
        max_unique_experts=2,
        sort_unique=True,
    )
    assert ordered[0].experts == (1, 0)
    assert ordered[1].experts == (3, 2, 3)


def test_prefill_seeds_only_empty_persistent_slots_by_frequency() -> None:
    bank = LayerExpertSlotBank(
        expert_count=6,
        persistent_slots=2,
        transient_slots=2,
    )
    assert bank.prepare_prefill_seed([3, 3, 2, 1, 3, 2]) == (3, 2)

    first = bank.plan([3, 1], phase="prefill")
    assert [(load.expert, load.persistent) for load in first.loads] == [
        (3, True),
        (1, False),
    ]
    second = bank.plan([2], phase="prefill")
    assert second.loads[0].persistent is True
    assert set(bank.resident_experts) == {2, 3}

    assert bank.prepare_prefill_seed([4, 4, 4]) == ()
    third = bank.plan([4], phase="prefill")
    assert third.loads[0].persistent is False
    assert third.evictions == ()
    assert set(bank.resident_experts) == {2, 3}


def test_reader_reports_unverified_digest_when_hashing_disabled(
    tmp_path: Path,
) -> None:
    root, _spec_value, manifest, expected = _artifact(tmp_path)
    destination = bytearray(manifest.records[0].logical_bytes)
    with PositionalExpertReader(root, use_native=False) as reader:
        digest = reader.read_record_into(
            manifest,
            manifest.records[0],
            destination,
            verify_hash=False,
        )
    assert digest == "unverified"
    assert bytes(destination) == expected[manifest.records[0].expert]


def test_config_rejects_unsafe_trust_combinations() -> None:
    spec = _spec()
    base = dict(
        model_key=spec.key,
        memory_limit_bytes=1 << 30,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    with pytest.raises(ValueError, match="requires prefer_sidecar"):
        ExpertStreamingConfig(
            **base,
            verify_sidecar_hash_at_open=True,
            prefer_sidecar=False,
        )
    with pytest.raises(ValueError, match="requires verify_sidecar_hash_at_open"):
        ExpertStreamingConfig(**base, slot_layout="metal-mmap")


def test_begin_split_route_rolls_back_when_executor_rejects(
    tmp_path: Path,
) -> None:
    root, spec, manifest, expected = _artifact(tmp_path)
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
    try:
        runtime._split_executor.shutdown(wait=True, cancel_futures=True)
        with pytest.raises(RuntimeError):
            runtime.begin_split_route(1, [0], phase="decode")
        # Without rollback the bank would keep mapping expert 0 to a
        # never-loaded slot, wedging every later route on this layer.
        assert runtime._banks[1].occupancy == 0
        ready = runtime.ensure_route(1, [0], phase="decode")
        assert bytes(ready.bindings[0].buffer) == expected[0]
        ready.release(synchronize=False)
    finally:
        runtime.close()


def test_begin_split_constructs_pending_before_accepting_miss_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first = runtime.ensure_route(1, [0], phase="decode")
    first.release(synchronize=False)
    slot = runtime.slots._physical(1, 0)
    generation = slot.generation
    policy_before = _layer_policy_state(runtime, 1)
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    original_submit = runtime._split_executor.submit
    submitted_ready: list[ReadyRoute] = []
    submit_calls = 0

    def complete_before_return(fn, *args, **kwargs):
        nonlocal submit_calls
        submit_calls += 1
        future = original_submit(fn, *args, **kwargs)
        submitted_ready.append(future.result(timeout=2))
        return future

    def reject_pending(*_args, **_kwargs):
        raise RuntimeError("injected pending split construction failure")

    monkeypatch.setattr(runtime._split_executor, "submit", complete_before_return)
    monkeypatch.setattr("mtplx.expert_runtime.PendingSplitRoute", reject_pending)
    try:
        with pytest.raises(RuntimeError, match="pending split construction failure"):
            runtime.begin_split_route(1, [1], phase="decode")

        assert submit_calls == 0
        assert submitted_ready == []
        assert _layer_policy_state(runtime, 1) == policy_before
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.layer == 1
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        for ready in submitted_ready:
            ready.release(synchronize=False)
        runtime.close(timeout=2)


def test_split_safe_fence_failure_rolls_back_policy_without_observation(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    held = runtime.ensure_route(1, [0], phase="decode")
    bank = runtime._banks[1]
    policy_before = (
        bank.resident_experts,
        bank._decode_epoch,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in bank._history
        ),
    )
    counters_before = (
        runtime.counters.as_dict(),
        runtime._layer_counters[1].as_dict(),
        runtime._phase_counters[RoutingPhase.DECODE].as_dict(),
    )
    slot = runtime.slots._physical(1, 0)
    generation = held.generations[0]
    read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]
    completion_started = threading.Event()
    fail_completion = threading.Event()
    fence_error = RuntimeError("split miss victim fence failure")
    pending: PendingSplitRoute | None = None

    def wait_then_fail() -> None:
        completion_started.set()
        assert fail_completion.wait(timeout=2)
        raise fence_error

    try:
        held.defer_bindings_until(held.bindings, wait_then_fail)
        held.release(synchronize=False)
        assert completion_started.wait(timeout=2)
        pending = runtime.begin_split_route(1, [1], phase="decode")
        deadline = time.monotonic() + 2
        while runtime.slots.metrics.as_dict()["pin_waits"] == 0:
            assert time.monotonic() < deadline, "split miss did not wait on pin"
            time.sleep(0.001)
        fail_completion.set()
        with pytest.raises(ExpertSlotError, match="completion fence failed") as exc:
            pending.finish_misses()

        assert exc.value.__cause__ is fence_error
        assert (
            bank.resident_experts,
            bank._decode_epoch,
            tuple(
                (history.score, history.score_epoch, history.last_used)
                for history in bank._history
            ),
        ) == policy_before
        assert (
            runtime.counters.as_dict(),
            runtime._layer_counters[1].as_dict(),
            runtime._phase_counters[RoutingPhase.DECODE].as_dict(),
        ) == counters_before
        with slot.condition:
            assert slot.state.value == "ready"
            assert slot.expert == 0
            assert slot.generation == generation
            assert slot.pins == 0
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes
    finally:
        fail_completion.set()
        if pending is not None:
            pending.close()
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_split_success_commits_and_observes_exactly_once(tmp_path: Path) -> None:
    root, spec, manifest, expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        pending = runtime.begin_split_route(1, [0], phase="decode")
        first = pending.finish_misses()
        second = pending.finish_misses()
        assert first is not None
        assert second is first
        assert bytes(first.bindings[0].buffer) == expected[0]
        assert runtime._banks[1].resident_experts == (0,)
        assert runtime.counters.as_dict()["route_calls"] == 1
        assert runtime._layer_counters[1].as_dict()["route_calls"] == 1
        assert (
            runtime._phase_counters[RoutingPhase.DECODE].as_dict()["route_calls"] == 1
        )
        pending.close()
    finally:
        runtime.close()


def test_split_route_subsets_preserve_global_generations() -> None:
    plan = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(0, 1),
        slots=(3, 4),
        hits=(0,),
        misses=(1,),
        loads=(SlotLoad(expert=1, slot=4, persistent=True, generation=12),),
        evictions=(),
        generations=(7, 12),
    )

    hit_plan = ExpertStreamingRuntime._subset_route_plan(plan, hits=True)
    miss_plan = ExpertStreamingRuntime._subset_route_plan(plan, hits=False)

    assert hit_plan is not None
    assert miss_plan is not None
    assert hit_plan.generations == (7,)
    assert miss_plan.generations == (12,)


def test_pending_split_route_reports_only_unfinished_miss_io() -> None:
    future: Future[ReadyRoute] = Future()
    layer_lock = threading.Lock()
    layer_lock.acquire()
    plan = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(0,),
        slots=(0,),
        hits=(),
        misses=(0,),
        loads=(),
        evictions=(),
    )
    pending = PendingSplitRoute(
        runtime=object(),
        layer=1,
        plan=plan,
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={future: plan},
    )

    assert pending.misses_pending is True
    future.set_exception(RuntimeError("test future completed"))
    assert pending.misses_pending is False
    pending.close()
    assert layer_lock.acquire(blocking=False) is True
    layer_lock.release()


def test_decode_split_route_yields_each_miss_in_completion_order(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=(spec.resident_bytes + 2 * spec.expert_record_bytes),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        expert_cache_limit_bytes=0,
        transient_slots=2,
        max_inflight_io_bytes=2 * spec.expert_record_bytes,
        verify_artifact_headers=False,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    release_slow = threading.Event()
    original_read = runtime.reader.read_record_into

    def ordered_read(manifest, record, destination, **kwargs):
        if record.expert == 0:
            assert release_slow.wait(timeout=2)
        return original_read(manifest, record, destination, **kwargs)

    runtime.reader.read_record_into = ordered_read
    try:
        with runtime.begin_split_route(1, [0, 1], phase="decode") as pending:
            assert len(pending._miss_futures) == 2
            ready_iter = pending.iter_ready_misses()
            first = next(ready_iter)
            assert tuple(binding.expert for binding in first.bindings) == (1,)

            release_slow.set()
            second = next(ready_iter)
            assert tuple(binding.expert for binding in second.bindings) == (0,)
            assert runtime.counters.as_dict()["route_calls"] == 1
            assert runtime.snapshot(mx_module=object())["incremental_misses"] == {
                "routes": 1,
                "parts": 2,
            }

            combined = pending.finish_misses()
            assert combined is not None
            assert tuple(binding.expert for binding in combined.bindings) == (0, 1)
    finally:
        release_slow.set()
        snapshot = runtime.snapshot(mx_module=object())
        runtime.close()

    assert snapshot["incremental_misses"] == {"routes": 1, "parts": 2}
    assert runtime.slots.snapshot()["metrics"]["active_routes"] == 0


def test_incremental_miss_failure_cancels_running_sibling_without_blocking_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    primary_error = RuntimeError("injected primary incremental miss failure")
    primary: Future[ReadyRoute] = Future()
    primary.set_exception(primary_error)
    sibling: Future[ReadyRoute] = Future()
    assert sibling.set_running_or_notify_cancel()
    captured_cancels: list[object] = []
    submitted = 0

    def controlled_submit(_fn, *_args, **kwargs):
        nonlocal submitted
        captured_cancels.append(kwargs["cancel_event"])
        submitted += 1
        return primary if submitted == 1 else sibling

    class ReleaseCounter:
        def __init__(self) -> None:
            self.releases = 0

        def release(self, *, synchronize: bool = True) -> None:
            assert synchronize is False
            self.releases += 1

    abandoned = ReleaseCounter()
    caller_cancel = threading.Event()
    monkeypatch.setattr(runtime._split_executor, "submit", controlled_submit)
    pending = runtime.begin_split_route(
        1,
        [0, 1],
        phase="decode",
        cancel_event=caller_cancel,
    )
    observed: list[BaseException] = []
    consume_started = threading.Event()

    def consume_failure() -> None:
        consume_started.set()
        try:
            next(pending.iter_ready_misses())
        except BaseException as exc:
            observed.append(exc)

    worker = threading.Thread(target=consume_failure, daemon=True)
    worker.start()
    assert consume_started.wait(timeout=2)
    worker.join(timeout=0.25)
    pending_closed = False
    try:
        assert not worker.is_alive(), "primary failure waited for a running sibling"
        assert observed == [primary_error]
        assert len(captured_cancels) == 2
        assert captured_cancels[0] is captured_cancels[1]
        assert captured_cancels[0] is not caller_cancel
        assert captured_cancels[0].is_set()
        pending.close()
        pending_closed = True
        layer_lock = runtime._layer_locks[1]
        assert not layer_lock.acquire(blocking=False), (
            "failed split released its lifecycle before sibling cleanup"
        )
    finally:
        sibling.set_result(abandoned)  # type: ignore[arg-type]
        worker.join(timeout=2)
        if not pending_closed:
            pending.close()
        runtime.close(timeout=2)
    assert not worker.is_alive()
    assert abandoned.releases == 1
    layer_lock = runtime._layer_locks[1]
    assert layer_lock.acquire(blocking=False)
    layer_lock.release()


def test_incremental_submit_failure_releases_pinned_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    warm = runtime.ensure_route(1, [0], phase="decode")
    warm.release(synchronize=False)
    policy_before = _layer_policy_state(runtime, 1)
    counters_before = runtime.snapshot(mx_module=object())["incremental_misses"]

    def reject_submit(*_args, **_kwargs):
        raise RuntimeError("injected incremental route submit rejection")

    monkeypatch.setattr(runtime._split_executor, "submit", reject_submit)
    try:
        with pytest.raises(RuntimeError, match="route submit rejection"):
            runtime.begin_split_route(1, [0, 1], phase="decode")

        assert _layer_policy_state(runtime, 1) == policy_before
        assert (
            runtime.snapshot(mx_module=object())["incremental_misses"]
            == counters_before
        )
        slot = runtime.slots._physical(1, 0)
        with slot.condition:
            assert slot.expert == 0
            assert slot.pins == 0
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        lock = runtime._layer_locks[1]
        assert lock.acquire(blocking=False)
        lock.release()
    finally:
        runtime.close(timeout=2)


def test_incremental_second_submit_failure_drains_accepted_part_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path, expert_count=5)
    spec = replace(base_spec, top_k=3)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.persistent_cache_bytes(3),
        context_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    warm = runtime.ensure_route(1, [0, 1, 2], phase="decode")
    warm.release(synchronize=False)
    bank = runtime._banks[1]
    assert tuple(bank._slot_to_expert) == (0, 1, 2)
    policy_before = _layer_policy_state(runtime, 1)
    untouched_victim = runtime.slots._physical(1, 2)
    with untouched_victim.condition:
        untouched_generation = untouched_victim.generation
    cache_before = runtime.counters.as_dict()
    incremental_before = runtime.snapshot(mx_module=object())["incremental_misses"]
    original_submit = runtime._split_executor.submit
    original_ensure = runtime.slots.ensure_route_part
    submit_error = RuntimeError("injected second outer split submit rejection")
    first_part_completed = threading.Event()
    submitted_parts: list[tuple[int, int]] = []
    successful_parts: list[ReadyRoute] = []
    submit_count = 0

    def observe_real_part(layer, part, **kwargs):
        ready = original_ensure(layer, part, **kwargs)
        successful_parts.append(ready)
        first_part_completed.set()
        return ready

    def reject_second_submit(fn, layer, part, **kwargs):
        nonlocal submit_count
        submit_count += 1
        submitted_parts.append((part.loads[0].expert, part.loads[0].slot))
        if submit_count == 1:
            future = original_submit(fn, layer, part, **kwargs)
            assert first_part_completed.wait(timeout=2)
            assert future.result(timeout=2) is successful_parts[0]
            return future
        raise submit_error

    monkeypatch.setattr(runtime.slots, "ensure_route_part", observe_real_part)
    monkeypatch.setattr(runtime._split_executor, "submit", reject_second_submit)
    try:
        with pytest.raises(RuntimeError, match="second outer split") as failed:
            runtime.begin_split_route(1, [0, 3, 4], phase="decode")

        assert failed.value is submit_error
        assert submit_count == 2
        assert submitted_parts == [(3, 1), (4, 2)]
        assert len(successful_parts) == 1
        assert successful_parts[0]._released is True
        assert tuple(bank._slot_to_expert) == (0, None, 2)
        assert bank._expert_to_slot == {0: 0, 2: 2}
        policy_after = _layer_policy_state(runtime, 1)
        assert policy_after["decode_epoch"] == policy_before["decode_epoch"]
        assert policy_after["history"] == policy_before["history"]
        assert (
            policy_after["prefill_seed_candidates"]
            == (policy_before["prefill_seed_candidates"])
        )
        assert runtime.counters.as_dict() == cache_before
        assert runtime.snapshot(mx_module=object())["incremental_misses"] == (
            incremental_before
        )
        slots = (*runtime.slots._persistent.values(), *runtime.slots._transient)
        for slot in slots:
            with slot.condition:
                assert slot.state.value != "loading"
                assert slot.pins == 0
        restored = runtime.slots._physical(1, 2)
        with restored.condition:
            assert restored.state.value == "ready"
            assert restored.expert == 2
            assert restored.generation == untouched_generation
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        layer_lock = runtime._layer_locks[1]
        assert layer_lock.acquire(blocking=False)
        layer_lock.release()

        monkeypatch.setattr(runtime._split_executor, "submit", original_submit)
        monkeypatch.setattr(runtime.slots, "ensure_route_part", original_ensure)
        restored_hit = runtime.ensure_route(1, [2], phase="decode")
        assert restored_hit.plan.hits == (2,)
        assert restored_hit.plan.loads == ()
        restored_hit.release(synchronize=False)
        with runtime.begin_split_route(1, [0, 3, 4], phase="decode") as retry:
            ready = retry.finish_misses()
            assert ready is not None
            assert tuple(binding.expert for binding in ready.bindings) == (3, 4)
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
    finally:
        monkeypatch.setattr(runtime._split_executor, "submit", original_submit)
        monkeypatch.setattr(runtime.slots, "ensure_route_part", original_ensure)
        runtime.close(timeout=2)


@pytest.mark.parametrize("warm_victims", [False, True], ids=["empty", "evicted"])
def test_global_partial_submit_failure_preserves_physical_generation_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm_victims: bool,
) -> None:
    root, base_spec, manifest, _expected = _global_artifact(
        tmp_path,
        expert_count=5,
    )
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _global_plan(spec, persistent_slots=2)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
            cache_scope="global",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    if warm_victims:
        warm = runtime.ensure_route(1, [0, 1], phase="decode")
        warm.release(synchronize=False)

    bank = runtime._global_bank
    assert bank is not None
    accepted_slot = 0
    accepted_physical = runtime.slots._physical(1, accepted_slot)
    with accepted_physical.condition:
        physical_generation_before = accepted_physical.generation
    policy_generation_before = bank._slot_generations[accepted_slot]
    assert policy_generation_before == physical_generation_before

    original_submit = runtime._split_executor.submit
    original_ensure = runtime.slots.ensure_route_part
    submit_error = RuntimeError("injected second global outer submit rejection")
    first_part_completed = threading.Event()
    successful_parts: list[ReadyRoute] = []
    submit_count = 0

    def observe_real_part(layer, part, **kwargs):
        ready = original_ensure(layer, part, **kwargs)
        successful_parts.append(ready)
        first_part_completed.set()
        return ready

    def reject_second_submit(fn, layer, part, **kwargs):
        nonlocal submit_count
        submit_count += 1
        if submit_count == 1:
            future = original_submit(fn, layer, part, **kwargs)
            assert first_part_completed.wait(timeout=2)
            assert future.result(timeout=2) is successful_parts[0]
            return future
        raise submit_error

    monkeypatch.setattr(runtime.slots, "ensure_route_part", observe_real_part)
    monkeypatch.setattr(runtime._split_executor, "submit", reject_second_submit)
    try:
        with pytest.raises(RuntimeError, match="second global outer") as failed:
            runtime.begin_split_route(1, [2, 3], phase="decode")

        assert failed.value is submit_error
        assert submit_count == 2
        assert len(successful_parts) == 1
        assert successful_parts[0]._released is True
        accepted_load = successful_parts[0].plan.loads[0]
        assert accepted_load.slot == accepted_slot
        assert accepted_load.generation == physical_generation_before + 1
        with accepted_physical.condition:
            physical_generation_after = accepted_physical.generation
            assert accepted_physical.state.value == "empty"
            assert accepted_physical.layer is None
            assert accepted_physical.expert is None
        assert physical_generation_after == accepted_load.generation
        policy_generation_after = bank._slot_generations[accepted_slot]

        if warm_victims:
            assert bank._slot_to_key == [None, (1, 1)]
            untouched = runtime.slots._physical(1, 1)
            with untouched.condition:
                assert untouched.state.value == "ready"
                assert untouched.layer == 1
                assert untouched.expert == 1

        monkeypatch.setattr(runtime._split_executor, "submit", original_submit)
        monkeypatch.setattr(runtime.slots, "ensure_route_part", original_ensure)
        try:
            retry = runtime.ensure_route(1, [4], phase="decode")
        except ExpertSlotError as exc:
            pytest.fail(
                "global slot generation rewound below physical generation: "
                f"policy={policy_generation_after}, "
                f"physical={physical_generation_after}; retry failed: {exc}"
            )
        assert policy_generation_after == physical_generation_after
        assert retry.plan.loads[0].generation == physical_generation_after + 1
        retry.release(synchronize=False)
    finally:
        monkeypatch.setattr(runtime._split_executor, "submit", original_submit)
        monkeypatch.setattr(runtime.slots, "ensure_route_part", original_ensure)
        runtime.close(timeout=2)


def test_incremental_partial_submit_defers_accepted_running_part_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path, expert_count=5)
    spec = replace(base_spec, top_k=3)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.persistent_cache_bytes(3),
        context_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
            cache_policy="lru",
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    warm = runtime.ensure_route(1, [0, 1, 2], phase="decode")
    warm.release(synchronize=False)
    cache_before = runtime.counters.as_dict()
    original_submit = runtime._split_executor.submit
    original_read = runtime.reader.read_record_into
    read_started = threading.Event()
    accepted = threading.Event()
    release_read = threading.Event()
    submit_error = RuntimeError("injected running-part outer submit rejection")
    first_future: Future[ReadyRoute] | None = None
    submit_count = 0

    def blocking_read(manifest, record, destination, **kwargs):
        if record.expert == 3:
            read_started.set()
            assert release_read.wait(timeout=2)
        return original_read(manifest, record, destination, **kwargs)

    def reject_after_running_accept(fn, layer, part, **kwargs):
        nonlocal first_future, submit_count
        submit_count += 1
        if submit_count == 1:
            admission = kwargs["io_admission"]
            original_mark = admission.mark_accepted

            def observe_acceptance() -> None:
                original_mark()
                accepted.set()

            admission.mark_accepted = observe_acceptance
            first_future = original_submit(fn, layer, part, **kwargs)
            assert read_started.wait(timeout=2)
            assert accepted.wait(timeout=2)
            assert not first_future.done()
            return first_future
        assert accepted.is_set()
        assert first_future is not None and not first_future.done()
        raise submit_error

    monkeypatch.setattr(runtime.reader, "read_record_into", blocking_read)
    monkeypatch.setattr(runtime._split_executor, "submit", reject_after_running_accept)
    try:
        with pytest.raises(RuntimeError, match="running-part outer") as failed:
            runtime.begin_split_route(1, [0, 3, 4], phase="decode")

        assert failed.value is submit_error
        assert first_future is not None and not first_future.done()
        assert tuple(runtime._banks[1]._slot_to_expert) == (0, 3, 4)

        release_read.set()
        with pytest.raises(ExpertSlotError):
            first_future.result(timeout=2)
        layer_lock = runtime._layer_locks[1]
        assert layer_lock.acquire(timeout=2)
        layer_lock.release()

        bank = runtime._banks[1]
        assert tuple(bank._slot_to_expert) == (0, None, 2)
        assert bank._expert_to_slot == {0: 0, 2: 2}
        assert runtime.counters.as_dict() == cache_before
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient):
            with slot.condition:
                assert slot.state.value != "loading"
                assert slot.pins == 0

        monkeypatch.setattr(runtime._split_executor, "submit", original_submit)
        restored = runtime.ensure_route(1, [2], phase="decode")
        assert restored.plan.hits == (2,)
        restored.release(synchronize=False)
    finally:
        release_read.set()
        monkeypatch.setattr(runtime._split_executor, "submit", original_submit)
        runtime.close(timeout=2)


def test_incremental_part_observes_sticky_completion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    original_ensure = runtime.slots.ensure_route_part
    completion_lock = threading.Lock()
    both_parts_completed = threading.Event()
    completion_count = 0

    def track_completion(*args, **kwargs):
        nonlocal completion_count
        ready = original_ensure(*args, **kwargs)
        with completion_lock:
            completion_count += 1
            if completion_count == 2:
                both_parts_completed.set()
        return ready

    monkeypatch.setattr(runtime.slots, "ensure_route_part", track_completion)
    pending = runtime.begin_split_route(1, [0, 1], phase="decode")
    assert both_parts_completed.wait(timeout=2)
    assert all(future.done() for future in pending._miss_futures)
    parts = pending.iter_ready_misses()
    first = next(parts)
    assert first is not None
    pending.release_miss(first)
    completion_error = RuntimeError("injected sticky completion failure between parts")
    runtime.slots._record_completion_error(completion_error)
    try:
        with pytest.raises(ExpertSlotError, match="completion fence failed") as failed:
            next(parts)
        assert failed.value.__cause__ is completion_error
        assert pending._io_admission is not None
        assert pending._io_admission.any_accepted
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        slots = (*runtime.slots._persistent.values(), *runtime.slots._transient)
        for slot in slots:
            with slot.condition:
                assert slot.pins == 0
                assert slot.state.value == "empty"
    finally:
        pending.close()
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_incremental_final_commit_is_atomic_with_completion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    bank = runtime._banks[1]
    policy_before = _layer_policy_state(runtime, 1)
    counters_before = runtime.counters.as_dict()
    incremental_before = (
        runtime._incremental_miss_routes,
        runtime._incremental_miss_parts,
    )
    pending = runtime.begin_split_route(1, [0], phase="decode")

    fence_started = threading.Event()
    fail_fence = threading.Event()
    final_health_checked = threading.Event()
    continue_after_fence = threading.Event()
    fence_error = RuntimeError("injected final-commit completion fence failure")

    def wait_then_fail() -> None:
        fence_started.set()
        assert fail_fence.wait(timeout=2)
        raise fence_error

    fence_future = runtime.slots._submit_completion_fence(
        wait_then_fail,
        lambda: None,
        slot_count=1,
    )
    assert fence_future is not None
    assert fence_started.wait(timeout=2)

    original_health_check = runtime.slots.raise_if_unhealthy
    health_checks = 0

    def pause_after_final_health_check() -> None:
        nonlocal health_checks
        original_health_check()
        health_checks += 1
        if health_checks == 1:
            final_health_checked.set()
            assert continue_after_fence.wait(timeout=2)

    monkeypatch.setattr(
        runtime.slots,
        "raise_if_unhealthy",
        pause_after_final_health_check,
    )
    finish_executor = ThreadPoolExecutor(max_workers=1)
    finish_call: Future[object] | None = None
    finish_result: object | None = None
    finish_error: BaseException | None = None
    close_error: ExpertSlotError | None = None
    try:
        finish_call = finish_executor.submit(pending.finish_misses)
        assert final_health_checked.wait(timeout=2)
        fail_fence.set()
        runtime.slots._drain_completion_fences()
        with pytest.raises(RuntimeError, match="final-commit completion") as fence:
            fence_future.result(timeout=2)
        assert fence.value is fence_error
        with runtime.slots._completion_error_lock:
            assert runtime.slots._completion_error is fence_error

        continue_after_fence.set()
        try:
            finish_result = finish_call.result(timeout=2)
        except BaseException as exc:
            finish_error = exc
        pending.close()

        policy_after = _layer_policy_state(runtime, 1)
        counters_after = runtime.counters.as_dict()
        incremental_after = (
            runtime._incremental_miss_routes,
            runtime._incremental_miss_parts,
        )
        pins_after_failure = 0
        physical_states: list[str] = []
        for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient):
            with slot.condition:
                pins_after_failure += slot.pins
                physical_states.append(slot.state.value)

        with pytest.raises(ExpertSlotError, match="completion fence failed") as retry:
            runtime.ensure_route(1, [0], phase="decode")
        with pytest.raises(ExpertSlotError, match="completion fence failed") as admit:
            runtime.begin_split_route(1, [0], phase="decode")
        with pytest.raises(
            ExpertSlotError, match="completion fence failed"
        ) as snapshot:
            runtime.snapshot(mx_module=object())
        try:
            runtime.close(timeout=2)
        except ExpertSlotError as exc:
            close_error = exc

        sticky_errors = (retry.value, admit.value, snapshot.value, close_error)
        assert close_error is not None
        assert all(error.__cause__ is fence_error for error in sticky_errors)
        evidence = (
            f"finish_result={type(finish_result).__name__}, "
            f"finish_error={finish_error!r}, "
            f"policy_restored={policy_after == policy_before}, "
            f"counter_delta={counters_after['route_calls'] - counters_before['route_calls']}, "
            f"incremental_before={incremental_before}, "
            f"incremental_after={incremental_after}, "
            f"pins={pins_after_failure}, states={physical_states}, "
            f"resident={bank.resident_experts}"
        )
        assert isinstance(finish_error, ExpertSlotError), evidence
        assert finish_error.__cause__ is fence_error, evidence
        assert finish_result is None, evidence
        assert policy_after == policy_before, evidence
        assert counters_after == counters_before, evidence
        assert incremental_after == incremental_before, evidence
        assert pins_after_failure == 0, evidence
        assert all(state == "empty" for state in physical_states), evidence
    finally:
        fail_fence.set()
        continue_after_fence.set()
        if finish_call is not None:
            try:
                finish_call.result(timeout=2)
            except BaseException:
                pass
        pending.close()
        finish_executor.shutdown(wait=True, cancel_futures=True)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_incremental_failure_keeps_lifecycle_until_deferred_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    sibling_started = threading.Event()
    release_sibling = threading.Event()
    original_ensure = runtime.slots.ensure_route_part

    def gate_second_part(layer, part, **kwargs):
        if part.misses == (1,):
            sibling_started.set()
            assert release_sibling.wait(timeout=2)
        return original_ensure(layer, part, **kwargs)

    monkeypatch.setattr(runtime.slots, "ensure_route_part", gate_second_part)
    pending = runtime.begin_split_route(1, [0, 1], phase="decode")
    parts = pending.iter_ready_misses()
    try:
        assert sibling_started.wait(timeout=2)
        first = next(parts)
        assert tuple(binding.expert for binding in first.bindings) == (0,)
        pending.release_miss(first)
        assert runtime.slots.metrics.as_dict()["active_routes"] == 1

        with pytest.raises(TimeoutError, match="active expert routes"):
            runtime.close(timeout=0.01)
        assert runtime._closed is False
        assert runtime.slots._closed is False
        assert all(
            slot.state.value != "closed"
            for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient)
        )

        release_sibling.set()
        with pytest.raises(ExpertSlotError, match="closing"):
            next(parts)
        pending.close()
        runtime.close(timeout=2)
        assert runtime.slots._closed is True
    finally:
        release_sibling.set()
        pending.close()
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_incremental_miss_parts_preserve_first_use_and_duplicate_order() -> None:
    plan = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(2, 0, 2, 1),
        slots=(5, 3, 5, 4),
        hits=(),
        misses=(2, 0, 1),
        loads=(
            SlotLoad(expert=2, slot=5, persistent=False, generation=7),
            SlotLoad(expert=0, slot=3, persistent=False, generation=8),
            SlotLoad(expert=1, slot=4, persistent=False, generation=9),
        ),
        evictions=(),
        generations=(7, 8, 7, 9),
    )

    parts = ExpertStreamingRuntime._miss_route_parts(plan)

    assert tuple(part.misses for part in parts) == ((2,), (0,), (1,))
    assert tuple(part.experts for part in parts) == ((2, 2), (0,), (1,))
    assert tuple(part.slots for part in parts) == ((5, 5), (3,), (4,))
    assert tuple(part.generations for part in parts) == ((7, 7), (8,), (9,))


def test_pending_split_close_releases_all_parts_after_first_release_error() -> None:
    first_error = RuntimeError("injected first part release failure")

    class FailingPart:
        def __init__(self, error: BaseException | None = None) -> None:
            self.error = error
            self.releases = 0

        def release(self, *, synchronize: bool = True) -> None:
            assert synchronize is False
            self.releases += 1
            if self.error is not None:
                raise self.error

    first = FailingPart(first_error)
    second = FailingPart()
    layer_lock = threading.Lock()
    layer_lock.acquire()
    pending = PendingSplitRoute(
        runtime=object(),
        layer=1,
        plan=RoutePlan(
            phase=RoutingPhase.DECODE,
            experts=(0, 1),
            slots=(0, 1),
            hits=(),
            misses=(0, 1),
            loads=(),
            evictions=(),
        ),
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={},
    )
    pending._policy_observed = True
    pending._miss_ready_parts = {  # type: ignore[assignment]
        0: first,
        1: second,
    }

    with pytest.raises(RuntimeError, match="first part release failure") as failed:
        pending.close()

    assert failed.value is first_error
    assert first.releases == 1
    assert second.releases == 1
    assert layer_lock.acquire(blocking=False)
    layer_lock.release()


def test_pending_split_abort_claims_future_completing_during_partition() -> None:
    part = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(0,),
        slots=(0,),
        hits=(),
        misses=(0,),
        loads=(),
        evictions=(),
    )

    class ReleaseCounter:
        def __init__(self) -> None:
            self.releases = 0

        def release(self, *, synchronize: bool = True) -> None:
            assert synchronize is False
            self.releases += 1

    ready = ReleaseCounter()

    class CompletingFuture(Future[ReadyRoute]):
        def __init__(self) -> None:
            super().__init__()
            self.done_calls = 0
            assert self.set_running_or_notify_cancel()

        def done(self) -> bool:
            self.done_calls += 1
            return self.done_calls > 1 or super().done()

        def add_done_callback(self, fn) -> None:
            if not super().done():
                self.set_result(ready)  # type: ignore[arg-type]
            super().add_done_callback(fn)

    future = CompletingFuture()
    failures: list[BaseException] = []

    class FakeRuntime:
        def _handle_split_route_failure(
            self,
            _layer,
            _plan,
            _policy_txn,
            error,
            **_kwargs,
        ) -> None:
            failures.append(error)

    layer_lock = threading.Lock()
    layer_lock.acquire()
    pending = PendingSplitRoute(
        runtime=FakeRuntime(),
        layer=1,
        plan=part,
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={future: part},
        miss_parts=(part,),
    )
    primary_error = RuntimeError("injected partition failure")

    pending.abort(primary_error)
    pending.close()

    assert failures == [primary_error]
    assert ready.releases == 1
    assert layer_lock.acquire(blocking=False)
    layer_lock.release()


def test_pending_split_abort_preclaims_callbacks_before_concurrent_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    original_read = runtime.reader.read_record_into
    original_failure = runtime._handle_split_route_failure
    reads_started = threading.Event()
    release_reads = threading.Event()
    read_lock = threading.Lock()
    started_experts: set[int] = set()
    failure_calls: list[BaseException] = []

    def block_accepted_reads(manifest, record, destination, **kwargs):
        with read_lock:
            started_experts.add(record.expert)
            if started_experts == {0, 1}:
                reads_started.set()
        assert release_reads.wait(timeout=2)
        kwargs["cancel_event"] = None
        return original_read(manifest, record, destination, **kwargs)

    def observe_failure(
        layer,
        route_plan,
        policy_txn,
        error,
        **kwargs,
    ):
        failure_calls.append(error)
        return original_failure(
            layer,
            route_plan,
            policy_txn,
            error,
            **kwargs,
        )

    monkeypatch.setattr(runtime.reader, "read_record_into", block_accepted_reads)
    monkeypatch.setattr(runtime, "_handle_split_route_failure", observe_failure)
    pending = runtime.begin_split_route(1, [0, 1], phase="decode")
    layer_lock = runtime._layer_locks[1]
    assert reads_started.wait(timeout=2)
    with pending._state_lock:
        futures = tuple(
            sorted(
                pending._miss_futures,
                key=lambda future: pending._miss_ordinals[future],
            )
        )
    assert len(futures) == 2

    abort_claimed_futures = threading.Event()
    continue_abort_setup = threading.Event()
    original_cancel = futures[0].cancel

    def pause_first_cancel() -> bool:
        abort_claimed_futures.set()
        assert continue_abort_setup.wait(timeout=2)
        return original_cancel()

    monkeypatch.setattr(futures[0], "cancel", pause_first_cancel)
    primary_error = RuntimeError("injected concurrent abort failure")
    abort_executor = ThreadPoolExecutor(max_workers=1)
    abort_call: Future[None] | None = None
    late_readies: list[ReadyRoute] = []
    layer_was_released = False
    try:
        abort_call = abort_executor.submit(pending.abort, primary_error)
        assert abort_claimed_futures.wait(timeout=2)
        with pending._state_lock:
            assert pending._failure is primary_error
            assert pending._miss_futures == {}
            callbacks_preclaimed = pending._failure_callbacks

        pending.close()
        with pending._state_lock:
            prematurely_finalized = pending._failure_finalized
            lifecycle_retained = pending._lifecycle_release is not None
        rollback_deferred = failure_calls == []
        active_routes_after_close = runtime.slots.metrics.as_dict()["active_routes"]
        layer_was_released = layer_lock.acquire(blocking=False)
        if layer_was_released:
            layer_lock.release()

        continue_abort_setup.set()
        abort_call.result(timeout=2)

        callbacks_completed = threading.Event()
        callback_lock = threading.Lock()
        completed_callbacks = 0

        def observe_callback(_future: Future[ReadyRoute]) -> None:
            nonlocal completed_callbacks
            with callback_lock:
                completed_callbacks += 1
                if completed_callbacks == len(futures):
                    callbacks_completed.set()

        for future in futures:
            future.add_done_callback(observe_callback)
        release_reads.set()
        assert callbacks_completed.wait(timeout=2)
        late_readies = [future.result(timeout=2) for future in futures]
        late_routes_released = tuple(ready._released for ready in late_readies)
        pins_after_callbacks = 0
        physical_states: list[str] = []
        for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient):
            with slot.condition:
                pins_after_callbacks += slot.pins
                physical_states.append(slot.state.value)
        assert layer_lock.acquire(timeout=2)
        layer_lock.release()

        assert failure_calls == [primary_error]
        evidence = (
            f"callbacks={callbacks_preclaimed}, "
            f"premature_finalized={prematurely_finalized}, "
            f"lifecycle_retained={lifecycle_retained}, "
            f"rollback_deferred={rollback_deferred}, "
            f"active_routes={active_routes_after_close}, "
            f"layer_released={layer_was_released}, "
            f"late_released={late_routes_released}, "
            f"pins={pins_after_callbacks}, states={physical_states}"
        )
        assert callbacks_preclaimed == len(futures), evidence
        assert prematurely_finalized is False, evidence
        assert lifecycle_retained is True, evidence
        assert rollback_deferred is True, evidence
        assert active_routes_after_close == 3, evidence
        assert layer_was_released is False, evidence
        assert late_routes_released == (True, True), evidence
        assert pins_after_callbacks == 0, evidence
    finally:
        continue_abort_setup.set()
        release_reads.set()
        if abort_call is not None:
            try:
                abort_call.result(timeout=2)
            except BaseException:
                pass
        for ready in late_readies:
            if not ready._released:
                ready.release(synchronize=False)
        pending.close()
        abort_executor.shutdown(wait=True, cancel_futures=True)
        runtime.close(timeout=2)


@pytest.mark.parametrize(
    "cleanup_kind",
    ["release", "rollback", "lifecycle"],
)
def test_deferred_split_cleanup_failure_becomes_sticky_runtime_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_kind: str,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=1,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    primary_error = RuntimeError(f"injected primary miss failure before {cleanup_kind}")
    cleanup_error = RuntimeError(f"injected deferred {cleanup_kind} cleanup failure")
    existing_kv = runtime.admit_kv_tokens(1)

    class DeferredReady:
        def __init__(self, error: BaseException | None = None) -> None:
            self.error = error
            self.releases = 0

        def release(self, *, synchronize: bool = True) -> None:
            assert synchronize is False
            self.releases += 1
            if self.error is not None:
                raise self.error

    full_plan = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(0, 1),
        slots=(0, 1),
        hits=(),
        misses=(0, 1),
        loads=(),
        evictions=(),
    )
    parts = tuple(
        RoutePlan(
            phase=RoutingPhase.DECODE,
            experts=(expert,),
            slots=(expert,),
            hits=(),
            misses=(expert,),
            loads=(),
            evictions=(),
        )
        for expert in (0, 1)
    )
    primary: Future[ReadyRoute] = Future()
    primary.set_exception(primary_error)
    sibling: Future[ReadyRoute] = Future()
    assert sibling.set_running_or_notify_cancel()
    ready = DeferredReady(cleanup_error if cleanup_kind == "release" else None)
    lifecycle_releases = 0

    def release_lifecycle() -> None:
        nonlocal lifecycle_releases
        lifecycle_releases += 1
        if cleanup_kind == "lifecycle":
            raise cleanup_error

    if cleanup_kind == "rollback":

        def fail_rollback(*_args, **_kwargs) -> None:
            raise cleanup_error

        monkeypatch.setattr(runtime, "_handle_split_route_failure", fail_rollback)

    layer_lock = runtime._layer_locks[1]
    layer_lock.acquire()
    pending = PendingSplitRoute(
        runtime=runtime,
        layer=1,
        plan=full_plan,
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={primary: parts[0], sibling: parts[1]},
        lifecycle_release=release_lifecycle,
        miss_parts=parts,
    )
    routes = pending.iter_ready_misses()
    ensured = None
    admitted = None
    snapshot_result = None
    ensure_error: ExpertSlotError | None = None
    admission_error: ExpertSlotError | None = None
    snapshot_error: ExpertSlotError | None = None
    reset_error: ExpertSlotError | None = None
    kv_error: ExpertSlotError | None = None
    close_error: ExpertSlotError | None = None
    try:
        with pytest.raises(RuntimeError, match="primary miss failure") as primary_seen:
            next(routes)
        assert primary_seen.value is primary_error
        assert sibling.done() is False
        pending.close()

        sibling.set_result(ready)  # type: ignore[arg-type]
        assert ready.releases == 1
        assert pending._cleanup_error is cleanup_error
        assert lifecycle_releases == 1
        assert layer_lock.acquire(blocking=False)
        layer_lock.release()
        existing_kv.release()
        kv_before = (runtime._live_kv_tokens, runtime._live_kv_peak)

        new_kv = None
        try:
            new_kv = runtime.admit_kv_tokens(1)
        except ExpertSlotError as exc:
            kv_error = exc
        kv_after = (runtime._live_kv_tokens, runtime._live_kv_peak)
        if new_kv is not None:
            new_kv.release()

        try:
            ensured = runtime.ensure_route(1, [0], phase="decode")
        except ExpertSlotError as exc:
            ensure_error = exc
        if ensured is not None:
            ensured.release(synchronize=False)
        try:
            admitted = runtime.begin_split_route(1, [0], phase="decode")
        except ExpertSlotError as exc:
            admission_error = exc
        if admitted is not None:
            admitted.close()
        try:
            snapshot_result = runtime.snapshot(mx_module=object())
        except ExpertSlotError as exc:
            snapshot_error = exc
        try:
            runtime.reset()
        except ExpertSlotError as exc:
            reset_error = exc
        try:
            runtime.close(timeout=2)
        except ExpertSlotError as exc:
            close_error = exc

        errors = (
            ensure_error,
            admission_error,
            snapshot_error,
            reset_error,
            kv_error,
            close_error,
        )
        evidence = (
            f"kind={cleanup_kind}, "
            f"ensure_error={ensure_error!r}, "
            f"admission_error={admission_error!r}, "
            f"snapshot_succeeded={snapshot_result is not None}, "
            f"reset_error={reset_error!r}, "
            f"kv_error={kv_error!r}, kv_before={kv_before}, kv_after={kv_after}, "
            f"close_error={close_error!r}, "
            f"runtime_closed={runtime._closed}, slots_closed={runtime.slots._closed}"
        )
        assert all(isinstance(error, ExpertSlotError) for error in errors), evidence
        assert all(error.__cause__ is cleanup_error for error in errors), evidence
        assert snapshot_result is None, evidence
        assert kv_after == kv_before, evidence
        assert runtime._closed is True, evidence
        assert runtime.slots._closed is True, evidence
        assert runtime.reader._closed is True, evidence
    finally:
        if not sibling.done():
            sibling.set_exception(RuntimeError("test cleanup"))
        pending.close()
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


@pytest.mark.parametrize("runtime_state", ["closing", "closed"])
def test_kv_admission_rejects_closing_and_closed_runtime_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_state: str,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=1,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    original_close = runtime.slots.close
    close_entered = threading.Event()
    continue_close = threading.Event()

    def gate_slot_close(*, timeout=None) -> None:
        close_entered.set()
        assert continue_close.wait(timeout=2)
        original_close(timeout=timeout)

    close_executor = ThreadPoolExecutor(max_workers=1)
    close_call: Future[None] | None = None
    admitted = None
    admission_error: ExpertSlotError | None = None
    try:
        if runtime_state == "closing":
            monkeypatch.setattr(runtime.slots, "close", gate_slot_close)
            close_call = close_executor.submit(runtime.close, timeout=2)
            assert close_entered.wait(timeout=2)
            assert runtime._closing is True
        else:
            runtime.close(timeout=2)
            assert runtime._closed is True

        kv_before = (runtime._live_kv_tokens, runtime._live_kv_peak)
        try:
            admitted = runtime.admit_kv_tokens(1)
        except ExpertSlotError as exc:
            admission_error = exc
        kv_after = (runtime._live_kv_tokens, runtime._live_kv_peak)
        if admitted is not None:
            admitted.release()

        evidence = (
            f"state={runtime_state}, error={admission_error!r}, "
            f"kv_before={kv_before}, kv_after={kv_after}"
        )
        assert isinstance(admission_error, ExpertSlotError), evidence
        assert str(admission_error) == (
            f"expert streaming runtime is {runtime_state}"
        ), evidence
        assert kv_after == kv_before, evidence
    finally:
        continue_close.set()
        if close_call is not None:
            close_call.result(timeout=2)
        close_executor.shutdown(wait=True, cancel_futures=True)
        runtime.close(timeout=2)


def test_claimed_ready_future_cannot_yield_after_concurrent_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    part = _manual_plan(0, 0)
    ready = runtime.slots.ensure_route(1, part)
    result_entered = threading.Event()
    continue_result = threading.Event()

    class GatedDoneFuture(Future[ReadyRoute]):
        def result(self, timeout=None):
            result_entered.set()
            assert continue_result.wait(timeout=2)
            return super().result(timeout=timeout)

    future = GatedDoneFuture()
    future.set_result(ready)
    release_count = 0
    original_release = ready.release

    def count_release(*, synchronize: bool = True) -> None:
        nonlocal release_count
        release_count += 1
        original_release(synchronize=synchronize)

    monkeypatch.setattr(ready, "release", count_release)
    lifecycle_releases = 0

    def release_lifecycle() -> None:
        nonlocal lifecycle_releases
        lifecycle_releases += 1

    layer_lock = runtime._layer_locks[1]
    layer_lock.acquire()
    pending = PendingSplitRoute(
        runtime=runtime,
        layer=1,
        plan=part,
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={future: part},
        lifecycle_release=release_lifecycle,
        miss_parts=(part,),
    )
    yielded: list[ReadyRoute] = []
    consumer_errors: list[BaseException] = []

    def consume() -> None:
        try:
            yielded.append(next(pending.iter_ready_misses()))
        except BaseException as exc:
            consumer_errors.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    layer_released_early = False
    try:
        assert result_entered.wait(timeout=2)
        with pending._state_lock:
            assert pending._miss_futures == {}
        pending.close()
        with pending._state_lock:
            finalized_before_result = pending._failure_finalized
        lifecycle_before_result = lifecycle_releases
        layer_released_early = layer_lock.acquire(blocking=False)
        if layer_released_early:
            layer_lock.release()

        continue_result.set()
        consumer.join(timeout=2)
        assert not consumer.is_alive()
        with pending._state_lock:
            owned_parts = tuple(pending._miss_ready_parts)
        pins = 0
        loading = 0
        for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient):
            with slot.condition:
                pins += slot.pins
                loading += int(slot.state.value == "loading")
        active_routes = runtime.slots.metrics.as_dict()["active_routes"]
        assert layer_lock.acquire(timeout=2)
        layer_lock.release()

        evidence = (
            f"yielded={len(yielded)}, errors={consumer_errors!r}, "
            f"release_count={release_count}, owned={owned_parts}, pins={pins}, "
            f"routes={active_routes}, loading={loading}, "
            f"finalized_before_result={finalized_before_result}, "
            f"lifecycle_before_result={lifecycle_before_result}, "
            f"layer_released_early={layer_released_early}"
        )
        assert yielded == [], evidence
        assert len(consumer_errors) == 1, evidence
        assert release_count == 1, evidence
        assert owned_parts == (), evidence
        assert pins == 0, evidence
        assert active_routes == 0, evidence
        assert loading == 0, evidence
        assert finalized_before_result is False, evidence
        assert lifecycle_before_result == 0, evidence
        assert lifecycle_releases == 1, evidence
        assert layer_released_early is False, evidence
    finally:
        continue_result.set()
        consumer.join(timeout=2)
        if not ready._released:
            ready.release(synchronize=False)
        pending.close()
        runtime.close(timeout=2)


def test_yielded_miss_lease_survives_abort_and_close_until_owner_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    part = _manual_plan(0, 0)
    ready = runtime.slots.ensure_route(1, part)
    future: Future[ReadyRoute] = Future()
    future.set_result(ready)
    release_count = 0
    original_release = ready.release

    def count_release(*, synchronize: bool = True) -> None:
        nonlocal release_count
        release_count += 1
        original_release(synchronize=synchronize)

    monkeypatch.setattr(ready, "release", count_release)
    layer_lock = runtime._layer_locks[1]
    layer_lock.acquire()
    pending = PendingSplitRoute(
        runtime=runtime,
        layer=1,
        plan=part,
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={future: part},
        miss_parts=(part,),
    )
    primary_error = RuntimeError("injected failure while consumer owns miss")
    layer_released_early = False
    owner_release_error: BaseException | None = None
    try:
        yielded = next(pending.iter_ready_misses())
        assert yielded is ready
        physical = runtime.slots._physical(1, 0)
        with physical.condition:
            assert physical.pins == 1

        pending.abort(primary_error)
        pending.close()
        with physical.condition:
            pins_during_compute = physical.pins
        release_count_during_compute = release_count
        layer_released_early = layer_lock.acquire(blocking=False)
        if layer_released_early:
            layer_lock.release()

        try:
            pending.release_miss(ready)
        except BaseException as exc:
            owner_release_error = exc
        with physical.condition:
            pins_after_release = physical.pins
            loading_after_release = physical.state.value == "loading"
        active_routes = runtime.slots.metrics.as_dict()["active_routes"]
        with pending._state_lock:
            owned_parts = tuple(pending._miss_ready_parts)
        assert layer_lock.acquire(timeout=2)
        layer_lock.release()

        evidence = (
            f"release_during_compute={release_count_during_compute}, "
            f"pins_during_compute={pins_during_compute}, "
            f"layer_released_early={layer_released_early}, "
            f"owner_release_error={owner_release_error!r}, "
            f"release_count={release_count}, pins_after={pins_after_release}, "
            f"routes={active_routes}, loading={loading_after_release}, "
            f"owned={owned_parts}"
        )
        assert pending._failure is primary_error, evidence
        assert release_count_during_compute == 0, evidence
        assert pins_during_compute == 1, evidence
        assert layer_released_early is False, evidence
        assert owner_release_error is None, evidence
        assert release_count == 1, evidence
        assert pins_after_release == 0, evidence
        assert active_routes == 0, evidence
        assert loading_after_release is False, evidence
        assert owned_parts == (), evidence
    finally:
        if not ready._released:
            ready.release(synchronize=False)
        pending.close()
        runtime.close(timeout=2)


def test_finish_misses_releases_internal_lease_after_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_spec, manifest, _expected = _artifact(tmp_path)
    spec = replace(base_spec, top_k=2)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=plan.total_limit_bytes,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            verify_artifact_headers=False,
        ),
        spec=spec,
        apply_memory_cap=False,
    )
    first_part = _manual_plan(0, 0)
    second_part = _manual_plan(1, 1)
    full_plan = RoutePlan(
        phase=RoutingPhase.DECODE,
        experts=(0, 1),
        slots=(0, 1),
        hits=(),
        misses=(0, 1),
        loads=first_part.loads + second_part.loads,
        evictions=(),
    )
    first_ready = runtime.slots.ensure_route(1, first_part)
    first_result_seen = threading.Event()

    class ObservedFuture(Future[ReadyRoute]):
        def result(self, timeout=None):
            first_result_seen.set()
            return super().result(timeout=timeout)

    first: Future[ReadyRoute] = ObservedFuture()
    first.set_result(first_ready)
    second: Future[ReadyRoute] = Future()
    assert second.set_running_or_notify_cancel()
    release_count = 0
    original_release = first_ready.release

    def count_release(*, synchronize: bool = True) -> None:
        nonlocal release_count
        release_count += 1
        original_release(synchronize=synchronize)

    monkeypatch.setattr(first_ready, "release", count_release)
    lifecycle = runtime.slots.retain_split_lifecycle()
    layer_lock = runtime._layer_locks[1]
    layer_lock.acquire()
    pending = PendingSplitRoute(
        runtime=runtime,
        layer=1,
        plan=full_plan,
        layer_lock=layer_lock,
        hit_ready=None,
        miss_futures={first: first_part, second: second_part},
        lifecycle_release=lifecycle.release,
        miss_parts=(first_part, second_part),
    )
    primary_error = RuntimeError("injected later finish_misses failure")
    finish_executor = ThreadPoolExecutor(max_workers=1)
    finish_call = finish_executor.submit(pending.finish_misses)
    layer_released = False
    try:
        assert first_result_seen.wait(timeout=2)
        second.set_exception(primary_error)
        with pytest.raises(RuntimeError, match="later finish_misses") as failed:
            finish_call.result(timeout=2)
        assert failed.value is primary_error
        pending.close()

        with pending._state_lock:
            leases = set(pending._consumer_leases)
            owned_parts = tuple(pending._miss_ready_parts)
            failure_finalized = pending._failure_finalized
        physical = runtime.slots._physical(1, 0)
        with physical.condition:
            pins = physical.pins
            loading = physical.state.value == "loading"
        active_routes = runtime.slots.metrics.as_dict()["active_routes"]
        layer_released = layer_lock.acquire(blocking=False)
        if layer_released:
            layer_lock.release()

        evidence = (
            f"release_count={release_count}, leases={leases}, owned={owned_parts}, "
            f"failure_finalized={failure_finalized}, active={active_routes}, "
            f"pins={pins}, loading={loading}, layer_released={layer_released}"
        )
        assert release_count == 1, evidence
        assert leases == set(), evidence
        assert owned_parts == (), evidence
        assert failure_finalized is True, evidence
        assert active_routes == 0, evidence
        assert pins == 0, evidence
        assert loading is False, evidence
        assert layer_released is True, evidence
    finally:
        if not second.done():
            second.set_exception(RuntimeError("test cleanup"))
        if not first_ready._released:
            try:
                pending.release_miss(first_ready)
            except ExpertSlotError:
                first_ready.release(synchronize=False)
        pending.close()
        finish_executor.shutdown(wait=True, cancel_futures=True)
        runtime.close(timeout=2)
