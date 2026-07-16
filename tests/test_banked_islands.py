"""Banked mmap island layers (issue #51, C6): page-cache-backed full banks.

A banked island layer serves all of its experts from one component-major
region of a repacked sidecar mapped into Metal without copies. Bank row
index IS the expert id (the C5 island contract), so router indices drive
``gather_qmm`` directly; physical residency belongs to the pager, not to
the slot pool. These tests pin the banked sidecar format, the repack
byte layout, the memory-plan carve-out (unwired), the config surface,
and bitwise parity between mmap-backed and wired island dispatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_banked import (
    BANKED_ALIGNMENT,
    BANKED_FORMAT,
    BankedManifestError,
    load_banked_manifest,
    write_banked_expert_banks,
)
from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_manifest import build_expert_sidecar
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_slots import ExpertSlotPool
from mtplx.expert_streaming_models import plan_expert_memory
from tests.test_dense_islands import _config, _plan_kwargs, _two_layer_spec
from tests.test_expert_slots_runtime import _global_artifact


# ---------------------------------------------------------------- plan math


def test_plan_mmap_islands_stay_off_the_fixed_side() -> None:
    spec = _two_layer_spec()
    base = plan_expert_memory(spec, **_plan_kwargs())
    plan = plan_expert_memory(spec, mmap_island_layer_count=1, **_plan_kwargs())
    band_bytes = spec.expert_count * spec.expert_record_bytes
    assert plan.mmap_island_layer_count == 1
    assert plan.mmap_island_bytes == band_bytes
    assert plan.island_bytes == 0
    # Unwired pages are the pager's, not MLX's: fixed budget must not move.
    assert plan.fixed_bytes == base.fixed_bytes
    # Slot math spans only the one remaining streamed layer.
    assert plan.slots_per_layer == spec.expert_count
    assert plan.persistent_slots == spec.expert_count


def test_plan_wired_plus_mmap_islands_cover_all_layers() -> None:
    spec = _two_layer_spec()
    plan = plan_expert_memory(
        spec,
        island_layer_count=1,
        mmap_island_layer_count=1,
        **_plan_kwargs(),
    )
    assert plan.slots_per_layer == 0
    assert plan.persistent_slots == 0
    assert plan.persistent_cache_bytes == 0
    assert plan.island_bytes == spec.expert_count * spec.expert_record_bytes
    assert plan.mmap_island_bytes == spec.expert_count * spec.expert_record_bytes


def test_plan_mmap_island_validation() -> None:
    spec = _two_layer_spec()
    with pytest.raises(ValueError, match="exceed routed layer"):
        plan_expert_memory(
            spec,
            island_layer_count=1,
            mmap_island_layer_count=2,
            **_plan_kwargs(),
        )
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        plan_expert_memory(
            spec,
            mmap_island_layer_count=1,
            cache_scope="global",
            **_plan_kwargs(),
        )


# ------------------------------------------------------------ config surface


def test_config_normalizes_mmap_island_layers(tmp_path) -> None:
    manifest_path = tmp_path / "banked.json"
    manifest_path.write_text("{}")
    config = _config(
        mmap_island_layers=[2, 1, 2],
        banked_manifest=str(manifest_path),
    )
    assert config.mmap_island_layers == (1, 2)
    assert _config().mmap_island_layers == ()
    assert _config().banked_codec == "none"


def test_config_mmap_island_validation(tmp_path) -> None:
    manifest_path = tmp_path / "banked.json"
    manifest_path.write_text("{}")
    with pytest.raises(ValueError, match="banked_manifest"):
        _config(mmap_island_layers=(1,))
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        _config(
            mmap_island_layers=(1,),
            banked_manifest=str(manifest_path),
            cache_scope="global",
        )
    with pytest.raises(ValueError, match="component-banks"):
        _config(
            mmap_island_layers=(1,),
            banked_manifest=str(manifest_path),
            slot_layout="direct-slots",
        )
    with pytest.raises(ValueError, match="trace_routes"):
        _config(
            mmap_island_layers=(1,),
            banked_manifest=str(manifest_path),
            trace_routes=True,
        )
    with pytest.raises(ValueError, match="disjoint"):
        _config(
            island_layers=(1,),
            mmap_island_layers=(1, 2),
            banked_manifest=str(manifest_path),
        )
    with pytest.raises(ValueError, match="banked_codec"):
        _config(
            mmap_island_layers=(1,),
            banked_manifest=str(manifest_path),
            banked_codec="rans32x-v1",
        )
    with pytest.raises(ValueError, match="banked_codec"):
        _config(banked_codec="gzip")


# ------------------------------------------------------------- repack writer


def _sidecar_artifact(tmp_path):
    root, spec, manifest, expected = _global_artifact(tmp_path)
    manifest = build_expert_sidecar(manifest, root, root / "experts.sidecar")
    return root, spec, manifest, expected


def _component_slices(record) -> dict[str, tuple[int, int]]:
    slices = {}
    cursor = 0
    for segment in record.segments:
        slices[segment.component] = (cursor, segment.length)
        cursor += segment.length
    return slices


def _write_banked(tmp_path, root, manifest, layers):
    out_bin = tmp_path / "banked" / "experts-banked.bin"
    out_manifest = tmp_path / "banked" / "experts-banked-manifest.json"
    banked = write_banked_expert_banks(
        manifest,
        root,
        layers,
        output_bin=out_bin,
        output_manifest=out_manifest,
    )
    return banked, out_bin, out_manifest


def test_write_banked_banks_layout_and_bytes(tmp_path) -> None:
    root, spec, manifest, expected = _sidecar_artifact(tmp_path)
    layers = tuple(spec.routed_layer_indices)
    banked, out_bin, out_manifest = _write_banked(tmp_path, root, manifest, layers)

    assert banked.format == BANKED_FORMAT
    assert banked.codec == "none"
    assert banked.expert_count == spec.expert_count
    assert banked.alignment == BANKED_ALIGNMENT
    assert tuple(entry.layer for entry in banked.layers) == layers

    payload = out_bin.read_bytes()
    reference = next(iter(manifest.records))
    slices = _component_slices(reference)
    for entry in banked.layers:
        assert entry.offset % BANKED_ALIGNMENT == 0
        region = payload[entry.offset : entry.offset + entry.length]
        assert hashlib.sha256(region).hexdigest() == entry.sha256
        cursor = 0
        for component in entry.components:
            start, length = slices[component.component]
            assert component.offset == cursor
            assert component.length == spec.expert_count * length
            want = b"".join(
                expected[(entry.layer, expert)][start : start + length]
                for expert in range(spec.expert_count)
            )
            got = region[component.offset : component.offset + component.length]
            assert got == want, f"layer {entry.layer} {component.component}"
            cursor += component.length
        assert cursor == entry.length

    reloaded = load_banked_manifest(out_manifest)
    assert reloaded.layers[0].components[0].component == (
        banked.layers[0].components[0].component
    )


def test_write_banked_requires_every_expert(tmp_path) -> None:
    from dataclasses import replace

    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    layer = spec.routed_layer_indices[0]
    pruned = replace(
        manifest,
        records=tuple(
            record
            for record in manifest.records
            if not (record.layer == layer and record.expert == 0)
        ),
    )
    with pytest.raises(BankedManifestError, match="no record"):
        _write_banked(tmp_path, root, pruned, (layer,))


def test_load_banked_manifest_rejects_tampering(tmp_path) -> None:
    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    _banked, _out_bin, out_manifest = _write_banked(
        tmp_path, root, manifest, tuple(spec.routed_layer_indices)
    )
    obj = json.loads(out_manifest.read_text())

    bad_format = dict(obj, format="mtplx-banked-expert-banks-v0")
    bad_path = tmp_path / "bad-format.json"
    bad_path.write_text(json.dumps(bad_format))
    with pytest.raises(BankedManifestError, match="format"):
        load_banked_manifest(bad_path)

    bad_codec = dict(obj, codec="gzip")
    bad_path = tmp_path / "bad-codec.json"
    bad_path.write_text(json.dumps(bad_codec))
    with pytest.raises(BankedManifestError, match="codec"):
        load_banked_manifest(bad_path)

    truncated = json.loads(out_manifest.read_text())
    truncated["layers"][0]["components"][0]["length"] -= 1
    bad_path = tmp_path / "bad-length.json"
    bad_path.write_text(json.dumps(truncated))
    with pytest.raises(BankedManifestError, match="length"):
        load_banked_manifest(bad_path)


# ------------------------------------------------------------------ pool


def test_pool_excludes_mmap_island_layers(tmp_path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    plan = plan_expert_memory(spec, mmap_island_layer_count=1, **_plan_kwargs())
    reader = PositionalExpertReader(root)
    banked_layer = spec.routed_layer_indices[0]
    pool = ExpertSlotPool(
        spec,
        plan,
        manifest,
        reader,
        island_layers=(banked_layer,),
    )
    try:
        labels = {slot.label for slot in pool._persistent.values()}
        assert not any(f"layer-{banked_layer}-" in label for label in labels)
        assert pool.allocated_bytes == (
            plan.persistent_cache_bytes + plan.transient_bytes
        )
    finally:
        pool.close()
        reader.close()


def test_pool_island_count_check_spans_both_kinds(tmp_path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    plan = plan_expert_memory(
        spec,
        island_layer_count=1,
        mmap_island_layer_count=1,
        **_plan_kwargs(),
    )
    reader = PositionalExpertReader(root)
    try:
        pool = ExpertSlotPool(
            spec,
            plan,
            manifest,
            reader,
            island_layers=tuple(spec.routed_layer_indices),
        )
        pool.close()
        with pytest.raises(ValueError, match="island"):
            ExpertSlotPool(
                spec,
                plan,
                manifest,
                reader,
                island_layers=tuple(spec.routed_layer_indices)[:1],
            )
    finally:
        reader.close()


# ------------------------------------------------------------- runtime open


def test_runtime_open_guards_mmap_island_entrypoints(tmp_path) -> None:
    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    from mtplx.expert_manifest import save_expert_manifest

    manifest_path = tmp_path / "manifest.json"
    save_expert_manifest(manifest, manifest_path)
    banked_layer = spec.routed_layer_indices[0]
    streamed_layer = spec.routed_layer_indices[1]
    _banked, _bin, banked_manifest = _write_banked(
        tmp_path, root, manifest, (banked_layer,)
    )
    config = _config(
        mmap_island_layers=(banked_layer,),
        banked_manifest=str(banked_manifest),
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
        assert banked_layer not in runtime._banks
        assert streamed_layer in runtime._banks
        assert banked_layer in runtime.island_layer_set
        with pytest.raises(
            ExpertStreamingConfigurationError, match="island layer"
        ):
            runtime.try_all_hit_route(banked_layer, (0,), phase="decode")
        with pytest.raises(
            ExpertStreamingConfigurationError, match="island layer"
        ):
            runtime.begin_split_route(banked_layer, (0,), phase="decode")
    finally:
        runtime.close()


def test_runtime_open_rejects_uncovered_mmap_layer(tmp_path) -> None:
    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    from mtplx.expert_manifest import save_expert_manifest

    manifest_path = tmp_path / "manifest.json"
    save_expert_manifest(manifest, manifest_path)
    covered = spec.routed_layer_indices[0]
    uncovered = spec.routed_layer_indices[1]
    _banked, _bin, banked_manifest = _write_banked(
        tmp_path, root, manifest, (covered,)
    )
    config = _config(
        mmap_island_layers=(uncovered,),
        banked_manifest=str(banked_manifest),
        verify_artifact_headers=False,
    )
    with pytest.raises(
        ExpertStreamingConfigurationError, match="banked manifest"
    ):
        ExpertStreamingRuntime.open(
            root,
            manifest_path,
            config,
            spec=spec,
            apply_memory_cap=False,
        )


# ------------------------------------------------------- benchmark gate


def _load_benchmark_module():
    import importlib.util
    import sys

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_q2_mtp_depth_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("bench_depth_matrix_c6", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_depth_matrix_c6", module)
    spec.loader.exec_module(module)
    return module


def test_decode_cache_gate_understands_full_island_coverage() -> None:
    bench = _load_benchmark_module()
    idle = {"decode": {"expert_hits": 0, "expert_misses": 0, "hit_rate": 0.0}}
    counters, hit_rate = bench._require_decode_cache_metrics(
        idle, model="hy3-q2", depth=3, fully_islanded=True
    )
    assert hit_rate is None
    assert counters["expert_hits"] == 0
    # Streamed traffic on a fully-islanded model is a wiring bug.
    busy = {"decode": {"expert_hits": 5, "expert_misses": 0, "hit_rate": 1.0}}
    with pytest.raises(bench.BenchmarkGateError, match="full island coverage"):
        bench._require_decode_cache_metrics(
            busy, model="hy3-q2", depth=3, fully_islanded=True
        )
    # Default behavior is unchanged: zero traffic still fails.
    with pytest.raises(bench.BenchmarkGateError, match="no routed assignments"):
        bench._require_decode_cache_metrics(idle, model="hy3-q2", depth=3)


# ---------------------------------------------------------- metal store


@pytest.fixture
def mlx():
    return pytest.importorskip("mlx.core")


def test_banked_store_arrays_match_wired_fill(tmp_path, mlx) -> None:
    import mlx.core as mx

    from mtplx.models.expert_mlx import BankedMmapIslandStore, DenseIslandStore

    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    layers = tuple(spec.routed_layer_indices)
    _banked, _bin, banked_manifest = _write_banked(tmp_path, root, manifest, layers)

    mmap_store = BankedMmapIslandStore(
        banked_manifest,
        layers,
        expert_count=spec.expert_count,
    )
    wired_store = DenseIslandStore(
        manifest,
        layers,
        expert_count=spec.expert_count,
    )
    reader = PositionalExpertReader(root)
    try:
        mmap_store.prepare()
        assert mmap_store.prefetch_all() >= 0
        wired_store.fill(manifest, reader, verify_hash=True)
        reference = next(iter(manifest.records))
        for layer in layers:
            mapped = mmap_store.bank_for_layer(layer)
            wired = wired_store.bank_for_layer(layer)
            for segment in reference.segments:
                component = segment.component
                assert mapped.arrays[component].shape == (
                    wired.arrays[component].shape
                )
                assert mapped.arrays[component].dtype == (
                    wired.arrays[component].dtype
                )
                assert mx.array_equal(
                    mapped.arrays[component], wired.arrays[component]
                ).item(), f"layer {layer} {component}"
        snapshot = mmap_store.snapshot()
        assert snapshot["backend"] == "banked-mmap-island-banks"
        assert snapshot["layers"] == list(layers)
    finally:
        mmap_store.close()
        wired_store.close()
        reader.close()


def test_banked_store_detects_corrupt_region(tmp_path, mlx) -> None:
    from mtplx.models.expert_mlx import BankedMmapIslandStore

    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    layers = tuple(spec.routed_layer_indices)
    banked, out_bin, banked_manifest = _write_banked(
        tmp_path, root, manifest, layers
    )
    payload = bytearray(out_bin.read_bytes())
    payload[banked.layers[0].offset] ^= 0xFF
    out_bin.write_bytes(payload)
    store = BankedMmapIslandStore(
        banked_manifest,
        layers,
        expert_count=spec.expert_count,
    )
    try:
        with pytest.raises(BankedManifestError, match="hash"):
            store.prepare()
    finally:
        store.close()


def test_banked_island_switch_bitwise_parity(tmp_path, mlx) -> None:
    """The mmap-backed island dispatch must be bitwise-identical to both the
    wired island dispatch and the streamed all-hit dispatch."""

    import mlx.core as mx

    from mtplx.models.expert_mlx import (
        BankedMmapIslandStore,
        DenseIslandStore,
        DenseIslandSwitchGLU,
        MlxComponentSlot,
        _run_component_bank_q4,
    )

    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    island_layer = spec.routed_layer_indices[0]
    _banked, _bin, banked_manifest = _write_banked(
        tmp_path, root, manifest, (island_layer,)
    )
    mmap_store = BankedMmapIslandStore(
        banked_manifest,
        (island_layer,),
        expert_count=spec.expert_count,
    )
    wired_store = DenseIslandStore(
        manifest,
        (island_layer,),
        expert_count=spec.expert_count,
    )
    reader = PositionalExpertReader(root)
    try:
        mmap_store.prepare()
        wired_store.fill(manifest, reader, verify_hash=True)
        runtime = SimpleNamespace(spec=spec)

        def make_switch(store):
            switch = DenseIslandSwitchGLU.__new__(DenseIslandSwitchGLU)
            switch.runtime = runtime
            switch.layer_index = island_layer
            switch.group_size = spec.quant_group_size
            switch.bits = spec.quant_bits
            switch._bank = store.bank_for_layer(island_layer)
            return switch

        rows = 3
        mx.random.seed(7)
        x = mx.random.normal((1, rows, spec.hidden_size)).astype(mx.bfloat16)
        indices = mx.array([[[1], [0], [1]]], dtype=mx.uint32)

        mapped_output = make_switch(mmap_store)(x, indices)
        wired_output = make_switch(wired_store)(x, indices)
        assert mx.array_equal(mapped_output, wired_output).item()

        tokens = x.reshape(-1, spec.hidden_size)
        assignment_inputs = mx.broadcast_to(
            tokens[:, None, :],
            (rows, spec.top_k, spec.hidden_size),
        ).reshape(-1, spec.hidden_size)
        wired_bank = wired_store.bank_for_layer(island_layer)
        bindings = tuple(
            SimpleNamespace(
                buffer=MlxComponentSlot(wired_bank, expert, label=f"ref-{expert}")
            )
            for expert in (1, 0, 1)
        )
        reference = _run_component_bank_q4(
            assignment_inputs,
            bindings,
            group_size=spec.quant_group_size,
            bits=spec.quant_bits,
        ).reshape((1, rows, spec.top_k, spec.hidden_size))
        assert mx.array_equal(mapped_output, reference).item()
    finally:
        mmap_store.close()
        wired_store.close()
        reader.close()


# ------------------------------------------------------------- source scan


def test_banked_store_hot_path_is_protocol_free() -> None:
    import inspect

    from mtplx.models import expert_mlx

    source = inspect.getsource(expert_mlx.BankedMmapIslandStore.bank_for_layer)
    for banned in (
        "mx.eval",
        "tolist",
        "try_all_hit_route",
        "begin_split_route",
        "threading.Lock",
        "atomic",
    ):
        assert banned not in source, f"banked island path must not use {banned}"
