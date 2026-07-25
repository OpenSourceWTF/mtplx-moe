"""Fix (B) resident-first same-layer overlap (issue #130).

Covers the ``overlap_miss_reads`` knob end to end: config validation, the
batched single-future decode miss submission in ``begin_split_route``, the
run-coalesced scatter reads in the slot pool, fail-closed short-read and
cancellation behavior on the batched path, the knob-off behavior lock, and
the overlap telemetry counters the acceptance window reports.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from mtplx.expert_io import (
    ExpertIOCancelled,
    ExpertIOError,
)
from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
    build_expert_sidecar,
    save_expert_manifest,
)
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
)
from mtplx.expert_slots import ExpertSlotError
from mtplx.expert_streaming_models import (
    ExpertStreamingModelSpec,
    plan_expert_memory,
)
from mtplx.models.expert_mlx import (
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)


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


def _spec(*, expert_count: int, top_k: int) -> ExpertStreamingModelSpec:
    record_bytes = sum(item[1] for item in COMPONENTS)
    return ExpertStreamingModelSpec(
        key="tiny-overlap-q4",
        display_name="Tiny Overlap Q4",
        source_model="test/tiny",
        source_revision="source-revision",
        quant_model="test/tiny-q4",
        quant_revision="quant-revision",
        total_tensor_bytes=expert_count * record_bytes + 1,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=expert_count,
        top_k=top_k,
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


def _overlap_artifact(
    tmp_path: Path,
    *,
    expert_count: int,
    top_k: int,
) -> tuple[Path, ExpertStreamingModelSpec, ExpertManifest, Path, dict[int, bytes]]:
    """Sidecar-backed tiny artifact whose records are strictly adjacent.

    ``alignment=1`` keeps record N's sidecar offset exactly at the end of
    record N-1, so consecutive expert ids form scatter-preadv-coalescible
    runs the same way the real hy3 sidecar's 16 KiB-aligned records do.
    """

    root = tmp_path / "artifact"
    root.mkdir()
    spec = _spec(expert_count=expert_count, top_k=top_k)
    raw = bytearray()
    records: list[ExpertRecord] = []
    expected: dict[int, bytes] = {}
    for expert in range(spec.expert_count):
        segments: list[TensorSegment] = []
        record_payload = bytearray()
        for component_index, (component, length, dtype, shape) in enumerate(COMPONENTS):
            payload = bytes([(expert * 16 + component_index + 1) % 251]) * length
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
    manifest = build_expert_sidecar(
        manifest, root, root / "experts.bin", alignment=1
    )
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, spec, manifest, manifest_path, expected


def _open_overlap_runtime(
    root: Path,
    spec: ExpertStreamingModelSpec,
    manifest_path: Path,
    *,
    overlap: bool,
) -> ExpertStreamingRuntime:
    from mtplx.expert_manifest import load_expert_manifest

    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        context_tokens=0,
        runtime_reserve_bytes=0,
    )
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=plan.total_limit_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
        cache_scope="layer",
        slot_layout="component-banks",
        overlap_miss_reads=overlap,
    )
    return ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            config.memory_plan(spec),
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )


def _drain_split_route(pending) -> list:
    readies = []
    if pending.hit_ready is not None:
        pending.release_hits()
    for ready in pending.iter_ready_misses():
        readies.append(ready)
        pending.release_miss(ready)
    return readies


# ---------------------------------------------------------------------------
# config knob


def test_overlap_miss_reads_config_defaults_off_and_validates() -> None:
    base = dict(
        model_key="tiny-overlap-q4",
        memory_limit_bytes=1 << 30,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )

    defaults = ExpertStreamingConfig(**base)
    assert defaults.overlap_miss_reads is False
    assert defaults.to_dict()["overlap_miss_reads"] is False

    enabled = ExpertStreamingConfig(
        **base,
        slot_layout="component-banks",
        overlap_miss_reads=True,
    )
    assert enabled.overlap_miss_reads is True
    assert enabled.to_dict()["overlap_miss_reads"] is True

    with pytest.raises(ValueError, match="component-banks"):
        ExpertStreamingConfig(**base, overlap_miss_reads=True)
    with pytest.raises((TypeError, ValueError), match="overlap_miss_reads"):
        ExpertStreamingConfig(
            **base,
            slot_layout="component-banks",
            overlap_miss_reads="yes",
        )


# ---------------------------------------------------------------------------
# canonical accumulation: split partitions must reproduce the fused wave
# bitwise (real shapes: hidden 4096, expert_hidden 1536, gs64 affine q4)


def _real_shape_bank(seed: int) -> dict[str, mx.array]:
    rng = np.random.default_rng(seed)
    experts = 8
    hidden = 4096
    expert_hidden = 1536

    def leaf(rows: int, cols: int) -> dict[str, np.ndarray]:
        return {
            "weight": rng.integers(
                0, 2**32, size=(experts, rows, cols // 8), dtype=np.uint64
            ).astype(np.uint32),
            "scales": rng.uniform(
                0.005, 0.02, size=(experts, rows, cols // 64)
            ).astype(np.float32),
            "biases": rng.uniform(
                -0.1, 0.0, size=(experts, rows, cols // 64)
            ).astype(np.float32),
        }

    arrays: dict[str, mx.array] = {}
    for projection, rows, cols in (
        ("gate_proj", expert_hidden, hidden),
        ("up_proj", expert_hidden, hidden),
        ("down_proj", hidden, expert_hidden),
    ):
        leaves = leaf(rows, cols)
        arrays[f"{projection}.weight"] = mx.array(leaves["weight"])
        arrays[f"{projection}.scales"] = mx.array(leaves["scales"]).astype(
            mx.bfloat16
        )
        arrays[f"{projection}.biases"] = mx.array(leaves["biases"]).astype(
            mx.bfloat16
        )
    mx.eval(arrays)
    return arrays


def _gather_rows(
    arrays: dict[str, mx.array],
    x_rows: mx.array,
    expert_rows: tuple[int, ...],
) -> np.ndarray:
    """Mirror of ``_gather_component_bank``: [rows, 1, 1, K] in, rows out."""

    from mtplx.models.expert_mlx import swiglu

    rows = int(x_rows.shape[0])
    hidden = int(x_rows.shape[-1])
    selected = x_rows.reshape((rows, 1, 1, hidden))
    indices = mx.array(list(expert_rows), dtype=mx.int32).reshape((-1, 1))

    def qmm(values: mx.array, projection: str) -> mx.array:
        result = mx.gather_qmm(
            values,
            arrays[f"{projection}.weight"],
            arrays[f"{projection}.scales"],
            arrays[f"{projection}.biases"],
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=4,
            mode="affine",
        )
        # The [rows, 1, K] -> [rows, 1, N] calling-convention trap: a wrong
        # input shape silently does 8x the work and fakes wins.
        assert result.shape[:3] == (rows, 1, 1), result.shape
        return result

    gate = qmm(selected, "gate_proj")
    up = qmm(selected, "up_proj")
    output = qmm(swiglu(gate, up), "down_proj")
    output = output.reshape((rows, int(output.shape[-1])))
    mx.eval(output)
    return np.asarray(output.astype(mx.float32))


def test_split_partitions_reproduce_fused_wave_bitwise() -> None:
    arrays = _real_shape_bank(seed=7)
    mx.random.seed(11)
    tokens = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    routed = (0, 3, 1, 7, 2, 6, 5, 4)
    x_rows = mx.broadcast_to(tokens, (len(routed), 4096))
    mx.eval(x_rows)

    fused = _gather_rows(arrays, x_rows, routed)

    partitions = (
        ((0, 1, 2, 3, 4, 5, 6, 7), ()),
        ((0, 1, 2, 3, 4, 5), (6, 7)),
        ((), (0, 1, 2, 3, 4, 5, 6, 7)),
        ((1, 4, 6), (0, 2, 3, 5, 7)),
    )
    for hit_positions, miss_positions in partitions:
        reassembled = np.zeros_like(fused)
        for positions in (hit_positions, miss_positions):
            if not positions:
                continue
            subset_x = mx.take(
                x_rows, mx.array(list(positions), dtype=mx.int32), axis=0
            )
            subset = _gather_rows(
                arrays,
                subset_x,
                tuple(routed[position] for position in positions),
            )
            for row, position in enumerate(positions):
                reassembled[position] = subset[row]
        assert np.array_equal(fused, reassembled), (
            hit_positions,
            miss_positions,
        )

    # Run-to-run determinism: a rebuilt bank and inputs from the same seeds
    # must reproduce the fused wave bit for bit.
    arrays_again = _real_shape_bank(seed=7)
    mx.random.seed(11)
    tokens_again = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    x_again = mx.broadcast_to(tokens_again, (len(routed), 4096))
    mx.eval(x_again)
    assert np.array_equal(fused, _gather_rows(arrays_again, x_again, routed))


# ---------------------------------------------------------------------------
# batched single-future decode miss submission


def test_split_route_overlap_on_submits_single_batched_miss_part(
    tmp_path: Path,
) -> None:
    root, spec, _manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    try:
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 1
            readies = _drain_split_route(pending)
        assert len(readies) == 1
        assert set(readies[0].plan.misses) == {0, 1, 2, 3}
        metrics = runtime.slots.metrics.as_dict()
        assert metrics["batched_miss_parts"] == 1
        assert metrics["batched_miss_records"] == 4
        assert metrics["load_failures"] == 0
    finally:
        runtime.close()


def test_split_route_overlap_off_keeps_per_expert_parts(tmp_path: Path) -> None:
    root, spec, _manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=False)
    try:
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 4
            readies = _drain_split_route(pending)
        assert len(readies) == 4
        metrics = runtime.slots.metrics.as_dict()
        assert metrics["batched_miss_parts"] == 0
        assert metrics["batched_miss_records"] == 0
        assert metrics["overlap_split_routes"] == 0
    finally:
        runtime.close()


def test_decode_batch_reads_coalesce_adjacent_runs(tmp_path: Path) -> None:
    root, spec, _manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=8, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    try:
        before = runtime.reader.metrics.as_dict()["read_operations"]
        with runtime.begin_split_route(1, [1, 2, 3], phase="decode") as pending:
            _drain_split_route(pending)
        adjacent_ops = runtime.reader.metrics.as_dict()["read_operations"] - before
        # Records 1..3 are strictly adjacent in the sidecar: one scatter call.
        assert adjacent_ops == 1

        before = runtime.reader.metrics.as_dict()["read_operations"]
        with runtime.begin_split_route(1, [5, 7], phase="decode") as pending:
            _drain_split_route(pending)
        scattered_ops = runtime.reader.metrics.as_dict()["read_operations"] - before
        # Records 5 and 7 are not adjacent: two independent reads, exactly
        # as the unbatched path would issue, preserving read concurrency.
        assert scattered_ops == 2

        metrics = runtime.slots.metrics.as_dict()
        assert metrics["load_failures"] == 0
        assert metrics["batched_miss_parts"] == 2
        assert metrics["batched_miss_records"] == 5
    finally:
        runtime.close()


def test_decode_batch_read_short_read_fails_closed(tmp_path: Path) -> None:
    root, spec, manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    assert manifest.sidecar is not None
    record = manifest.record(1, 3)
    assert record.sidecar_offset is not None
    sidecar_path = root / manifest.sidecar.file
    sidecar_path.write_bytes(
        sidecar_path.read_bytes()[: record.sidecar_offset + 100]
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    try:
        with pytest.raises((ExpertSlotError, ExpertIOError)):
            with runtime.begin_split_route(1, [2, 3], phase="decode") as pending:
                _drain_split_route(pending)
        assert runtime.slots.metrics.as_dict()["load_failures"] >= 1
    finally:
        runtime.close()


def test_decode_batch_read_cancellation_fails_closed(tmp_path: Path) -> None:
    root, spec, _manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    cancel_event = threading.Event()
    cancel_event.set()
    try:
        with pytest.raises((ExpertIOCancelled, ExpertSlotError, ExpertIOError)):
            with runtime.begin_split_route(
                1,
                [1, 2],
                phase="decode",
                cancel_event=cancel_event,
            ) as pending:
                _drain_split_route(pending)
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# overlap telemetry: the acceptance window reports measured coactivity, not
# tok/s inference


def _warm_persistent_expert(runtime: ExpertStreamingRuntime, expert: int) -> None:
    # First decode touch loads the record transient; the second touch admits
    # it persistent; after both, routing the expert again is a planned hit.
    for _ in range(2):
        with runtime.begin_split_route(1, [expert], phase="decode") as pending:
            _drain_split_route(pending)


def _slow_reader(runtime: ExpertStreamingRuntime, delay_s: float) -> None:
    reader = runtime.reader
    original_single = reader.read_record_into
    original_batch = reader.read_component_records_into

    def slow_single(*args, **kwargs):
        time.sleep(delay_s)
        return original_single(*args, **kwargs)

    def slow_batch(*args, **kwargs):
        time.sleep(delay_s)
        return original_batch(*args, **kwargs)

    reader.read_record_into = slow_single
    reader.read_component_records_into = slow_batch


def test_overlap_telemetry_records_dispatch_and_exposed_wait(
    tmp_path: Path,
) -> None:
    root, spec, _manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=4, top_k=2
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    try:
        _warm_persistent_expert(runtime, 0)
        _slow_reader(runtime, 0.05)
        switch = HotExpertSwitchGLU(runtime, 1)
        mx.random.seed(3)
        x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
        indices = mx.array([[[0, 1]]], dtype=mx.uint32)
        output = switch(x, indices)
        mx.eval(output)
        assert output.shape == (1, 1, 2, 64)

        metrics = runtime.slots.metrics.as_dict()
        assert metrics["overlap_split_routes"] >= 1
        assert metrics["overlap_gpu_dispatch_ns"] > 0
        assert metrics["overlap_exposed_wait_ns"] > 0
    finally:
        runtime.close()


def test_overlap_telemetry_stays_zero_with_knob_off(tmp_path: Path) -> None:
    root, spec, _manifest, manifest_path, _expected = _overlap_artifact(
        tmp_path, expert_count=4, top_k=2
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=False)
    try:
        _warm_persistent_expert(runtime, 0)
        switch = HotExpertSwitchGLU(runtime, 1)
        mx.random.seed(3)
        x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
        indices = mx.array([[[0, 1]]], dtype=mx.uint32)
        output = switch(x, indices)
        mx.eval(output)

        metrics = runtime.slots.metrics.as_dict()
        assert metrics["overlap_split_routes"] == 0
        assert metrics["overlap_gpu_dispatch_ns"] == 0
        assert metrics["overlap_exposed_wait_ns"] == 0
    finally:
        runtime.close()
