"""Dense island layers (issue #63, C5): full-resident per-layer expert banks.

An island layer holds all of its experts in a component bank whose row index
IS the expert id, so router indices drive ``gather_qmm`` directly — no
expert-to-slot translation, no residency probe, no pins. These tests pin the
memory-plan math, the pool carve-out, the config surface, the store fill
path, and bitwise parity between the island dispatch and the streamed
component-bank dispatch.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_manifest import build_expert_sidecar
from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_slots import ExpertSlotPool
from mtplx.expert_streaming_models import plan_expert_memory
from tests.test_expert_slots_runtime import _global_artifact, _spec


def _two_layer_spec():
    base = _spec()
    return replace(
        base,
        routed_layer_count=2,
        total_layers=3,
        mtp_layer_index=3,
        total_tensor_bytes=2 * base.expert_count * base.expert_record_bytes + 1,
    )


def _plan_kwargs(**overrides):
    kwargs = {
        "total_limit_bytes": 1 << 30,
        "context_tokens": 16,
        "runtime_reserve_bytes": 0,
        "transient_slots": 2,
    }
    kwargs.update(overrides)
    return kwargs


def test_plan_island_bytes_land_on_fixed_side() -> None:
    spec = _two_layer_spec()
    base = plan_expert_memory(spec, **_plan_kwargs())
    plan = plan_expert_memory(spec, island_layer_count=1, **_plan_kwargs())
    island_bytes = spec.expert_count * spec.expert_record_bytes
    assert plan.island_layer_count == 1
    assert plan.island_bytes == island_bytes
    assert plan.fixed_bytes == base.fixed_bytes + island_bytes
    # Slot math spans only the streamed layer.
    assert plan.slots_per_layer == spec.expert_count
    assert plan.persistent_slots == spec.expert_count
    assert (
        plan.persistent_cache_bytes
        == spec.expert_count * spec.expert_record_bytes
    )


def test_plan_all_layers_islanded_needs_no_uniform_slots() -> None:
    spec = _two_layer_spec()
    plan = plan_expert_memory(spec, island_layer_count=2, **_plan_kwargs())
    assert plan.slots_per_layer == 0
    assert plan.persistent_slots == 0
    assert plan.persistent_cache_bytes == 0
    assert plan.island_bytes == 2 * spec.expert_count * spec.expert_record_bytes


def test_plan_islands_count_toward_fits_fixed() -> None:
    spec = _two_layer_spec()
    tight = (
        spec.resident_bytes
        + 16 * spec.kv_bytes_per_token
        + 2 * spec.expert_record_bytes
        + spec.expert_count * spec.expert_record_bytes
    )
    fits = plan_expert_memory(
        spec, island_layer_count=1, **_plan_kwargs(total_limit_bytes=tight)
    )
    assert fits.fits_fixed
    over = plan_expert_memory(
        spec, island_layer_count=1, **_plan_kwargs(total_limit_bytes=tight - 1)
    )
    assert not over.fits_fixed


def test_plan_island_validation() -> None:
    spec = _two_layer_spec()
    with pytest.raises(ValueError, match="exceeds routed layer"):
        plan_expert_memory(spec, island_layer_count=3, **_plan_kwargs())
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        plan_expert_memory(
            spec,
            island_layer_count=1,
            cache_scope="global",
            **_plan_kwargs(),
        )


def _config(**overrides) -> ExpertStreamingConfig:
    kwargs = {
        "model_key": "tiny-global-q4",
        "memory_limit_bytes": 1 << 30,
        "max_live_kv_tokens": 16,
        "runtime_reserve_bytes": 0,
        "transient_slots": 2,
        "slot_layout": "component-banks",
    }
    kwargs.update(overrides)
    return ExpertStreamingConfig(**kwargs)


def test_config_normalizes_island_layers() -> None:
    config = _config(island_layers=[2, 1, 2])
    assert config.island_layers == (1, 2)
    assert _config().island_layers == ()


def test_config_island_validation() -> None:
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        _config(island_layers=(1,), cache_scope="global")
    with pytest.raises(ValueError, match="component-banks"):
        _config(island_layers=(1,), slot_layout="direct-slots")
    with pytest.raises(ValueError, match="trace_routes"):
        _config(island_layers=(1,), trace_routes=True)
    with pytest.raises(TypeError, match="island_layers"):
        _config(island_layers="1,2")


def test_pool_excludes_island_layers_and_keeps_invariant(tmp_path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    plan = plan_expert_memory(spec, island_layer_count=1, **_plan_kwargs())
    reader = PositionalExpertReader(root)
    island_layer = spec.routed_layer_indices[0]
    pool = ExpertSlotPool(
        spec,
        plan,
        manifest,
        reader,
        island_layers=(island_layer,),
    )
    try:
        labels = {
            slot.label for slot in pool._persistent.values()
        }
        assert not any(f"layer-{island_layer}-" in label for label in labels)
        streamed_layer = spec.routed_layer_indices[1]
        assert sum(
            1 for label in labels if f"layer-{streamed_layer}-" in label
        ) == plan.slots_per_layer
        assert pool.allocated_bytes == (
            plan.persistent_cache_bytes + plan.transient_bytes
        )
    finally:
        pool.close()
        reader.close()


def test_pool_island_layers_must_match_plan(tmp_path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    plan = plan_expert_memory(spec, **_plan_kwargs())
    reader = PositionalExpertReader(root)
    try:
        with pytest.raises(ValueError, match="island_layer_count"):
            ExpertSlotPool(
                spec,
                plan,
                manifest,
                reader,
                island_layers=(spec.routed_layer_indices[0],),
            )
    finally:
        reader.close()


def test_runtime_open_guards_island_route_entrypoints(tmp_path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    from mtplx.expert_manifest import save_expert_manifest

    save_expert_manifest(manifest, manifest_path)
    island_layer = spec.routed_layer_indices[0]
    streamed_layer = spec.routed_layer_indices[1]
    config = _config(island_layers=(island_layer,), verify_artifact_headers=False)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        assert island_layer not in runtime._banks
        assert streamed_layer in runtime._banks
        with pytest.raises(
            ExpertStreamingConfigurationError, match="dense island layer"
        ):
            runtime.try_all_hit_route(
                island_layer, (0,), phase="decode"
            )
        with pytest.raises(
            ExpertStreamingConfigurationError, match="dense island layer"
        ):
            runtime.begin_split_route(island_layer, (0,), phase="decode")
        with pytest.raises(
            ExpertStreamingConfigurationError, match="dense island layer"
        ):
            runtime.prepare_prefill_seed(island_layer, (0,))
    finally:
        runtime.close()


def test_runtime_open_rejects_unrouted_island_layer(tmp_path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    from mtplx.expert_manifest import save_expert_manifest

    save_expert_manifest(manifest, manifest_path)
    config = _config(island_layers=(99,), verify_artifact_headers=False)
    with pytest.raises(
        ExpertStreamingConfigurationError, match="island_layers must be routed"
    ):
        ExpertStreamingRuntime.open(
            root,
            manifest_path,
            config,
            spec=spec,
            apply_memory_cap=False,
        )


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_q2_mtp_depth_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("bench_depth_matrix", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_depth_matrix", module)
    spec.loader.exec_module(module)
    return module


def test_parse_island_layers_ranges() -> None:
    bench = _load_benchmark_module()
    assert bench.parse_island_layers("") == ()
    assert bench.parse_island_layers("1-3,7, 9-10") == (1, 2, 3, 7, 9, 10)
    assert bench.parse_island_layers("5") == (5,)
    assert bench.parse_island_layers("3, 1-3") == (1, 2, 3)
    with pytest.raises(ValueError, match="inverted"):
        bench.parse_island_layers("9-7")


@pytest.fixture
def mlx():
    return pytest.importorskip("mlx.core")


def _sidecar_artifact(tmp_path):
    root, spec, manifest, expected = _global_artifact(tmp_path)
    manifest = build_expert_sidecar(manifest, root, root / "experts.sidecar")
    return root, spec, manifest, expected


def test_island_store_fill_places_expert_bytes_in_expert_rows(
    tmp_path, mlx
) -> None:
    from mtplx.models.expert_mlx import DenseIslandStore

    root, spec, manifest, expected = _sidecar_artifact(tmp_path)
    store = DenseIslandStore(
        manifest,
        spec.routed_layer_indices,
        expert_count=spec.expert_count,
    )
    reader = PositionalExpertReader(root)
    try:
        store.fill(manifest, reader, verify_hash=True)
        for layer in spec.routed_layer_indices:
            bank = store.bank_for_layer(layer)
            for expert in range(spec.expert_count):
                payload = bytearray()
                record = next(
                    r
                    for r in manifest.records
                    if r.layer == layer and r.expert == expert
                )
                for segment in record.segments:
                    payload.extend(
                        bank.component_view(expert, segment.component)
                    )
                assert bytes(payload) == expected[(layer, expert)]
        snapshot = store.snapshot()
        assert snapshot["backend"] == "dense-island-banks"
        assert snapshot["filled_layers"] == len(spec.routed_layer_indices)
        assert snapshot["island_bytes"] == (
            len(spec.routed_layer_indices)
            * spec.expert_count
            * spec.expert_record_bytes
        )
    finally:
        store.close()
        reader.close()


def test_island_store_requires_complete_layer(tmp_path, mlx) -> None:
    from mtplx.models.expert_mlx import DenseIslandStore

    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    pruned = replace(
        manifest,
        records=tuple(
            record
            for record in manifest.records
            if not (
                record.layer == spec.routed_layer_indices[0]
                and record.expert == 0
            )
        ),
    )
    with pytest.raises(ValueError, match="no record for island layer"):
        DenseIslandStore(
            pruned,
            spec.routed_layer_indices,
            expert_count=spec.expert_count,
        )


def test_island_store_bank_requires_fill_before_use(tmp_path, mlx) -> None:
    from mtplx.models.expert_mlx import DenseIslandStore

    _root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    store = DenseIslandStore(
        manifest,
        spec.routed_layer_indices[:1],
        expert_count=spec.expert_count,
    )
    try:
        with pytest.raises(RuntimeError, match="has not been filled"):
            store.bank_for_layer(spec.routed_layer_indices[0])
    finally:
        store.close()


def test_island_switch_matches_component_bank_dispatch(tmp_path, mlx) -> None:
    """Raw-index island dispatch is bitwise-identical to the streamed
    all-hit dispatch over the same bank contents and routes."""

    import mlx.core as mx

    from mtplx.models.expert_mlx import (
        DenseIslandStore,
        DenseIslandSwitchGLU,
        MlxComponentSlot,
        _run_component_bank_q4,
    )

    root, spec, manifest, _expected = _sidecar_artifact(tmp_path)
    island_layer = spec.routed_layer_indices[0]
    store = DenseIslandStore(
        manifest,
        (island_layer,),
        expert_count=spec.expert_count,
    )
    reader = PositionalExpertReader(root)
    try:
        store.fill(manifest, reader, verify_hash=True)
        bank = store.bank_for_layer(island_layer)
        runtime = SimpleNamespace(spec=spec)
        switch = DenseIslandSwitchGLU.__new__(DenseIslandSwitchGLU)
        switch.runtime = runtime
        switch.layer_index = island_layer
        switch.group_size = spec.quant_group_size
        switch.bits = spec.quant_bits
        switch._bank = bank

        rows = 3
        mx.random.seed(7)
        x = mx.random.normal((1, rows, spec.hidden_size)).astype(mx.bfloat16)
        indices = mx.array(
            [[[1], [0], [1]]],
            dtype=mx.uint32,
        )
        island_output = switch(x, indices)
        assert island_output.shape == (1, rows, spec.top_k, spec.hidden_size)

        tokens = x.reshape(-1, spec.hidden_size)
        assignment_inputs = mx.broadcast_to(
            tokens[:, None, :],
            (rows, spec.top_k, spec.hidden_size),
        ).reshape(-1, spec.hidden_size)
        bindings = tuple(
            SimpleNamespace(
                buffer=MlxComponentSlot(bank, expert, label=f"ref-{expert}")
            )
            for expert in (1, 0, 1)
        )
        reference = _run_component_bank_q4(
            assignment_inputs,
            bindings,
            group_size=spec.quant_group_size,
            bits=spec.quant_bits,
        ).reshape((1, rows, spec.top_k, spec.hidden_size))
        assert mx.array_equal(island_output, reference).item()
    finally:
        store.close()
        reader.close()


def test_scatter_read_survives_more_views_than_iov_max(tmp_path) -> None:
    """A full island layer scatters into 192 records x 9 views = 1728
    iovecs; preadv rejects vectors above IOV_MAX (1024 on macOS) with
    EINVAL, so the reader must slice the vector per syscall."""

    import mtplx.expert_io as expert_io_module

    payload = bytes(range(256)) * 64  # 16 KiB
    source = tmp_path / "artifact"
    source.mkdir()
    (source / "record.bin").write_bytes(payload)
    reader = PositionalExpertReader(source)
    try:
        view_count = expert_io_module._IOV_MAX + 704
        chunk = len(payload) // view_count
        assert chunk > 0
        buffer = bytearray(chunk * view_count)
        destinations = tuple(
            memoryview(buffer)[index * chunk : (index + 1) * chunk]
            for index in range(view_count)
        )
        reader._readv_range_into(
            "record.bin",
            0,
            destinations,
            cancel_event=None,
            deadline_ns=None,
        )
        assert bytes(buffer) == payload[: len(buffer)]
    finally:
        reader.close()


def test_island_switch_source_is_protocol_free() -> None:
    import inspect

    from mtplx.models import expert_mlx

    source = inspect.getsource(expert_mlx.DenseIslandSwitchGLU) + inspect.getsource(
        expert_mlx.DenseIslandStore
    )
    for banned in (
        "mx.eval",
        "tolist",
        "try_all_hit_route",
        "begin_split_route",
        "threading.Lock",
        "release(",
        "atomic",
    ):
        assert banned not in source, f"island path must not use {banned}"


def test_island_layer_count_resolves_from_pin_order() -> None:
    from mtplx.expert_streaming_models import get_model_spec

    spec = get_model_spec("hy3-expert-q2")
    assert len(spec.island_pin_order) == spec.routed_layer_count
    assert set(spec.island_pin_order) == set(spec.routed_layer_indices)
    config = ExpertStreamingConfig(
        model_key="hy3-expert-q2",
        memory_limit_bytes=1 << 40,
        max_live_kv_tokens=16,
        slot_layout="component-banks",
        island_layer_count=4,
    )
    assert config.island_layers == tuple(sorted(spec.island_pin_order[:4]))
    assert config.island_layer_count == 4


def test_island_layer_count_validation() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(island_layers=(1,), island_layer_count=2)
    with pytest.raises(ValueError, match="pin order"):
        ExpertStreamingConfig(
            model_key="glm52-expert-q2",
            memory_limit_bytes=1 << 30,
            max_live_kv_tokens=16,
            slot_layout="component-banks",
            island_layer_count=1,
        )
    with pytest.raises(ValueError, match="island_layer_count"):
        ExpertStreamingConfig(
            model_key="hy3-expert-q2",
            memory_limit_bytes=1 << 40,
            max_live_kv_tokens=16,
            slot_layout="component-banks",
            island_layer_count=100,
        )


def test_island_layer_count_survives_benchmark_option_pipeline() -> None:
    """End-to-end through the benchmark's own plumbing: CLI vector ->
    parser -> runtime options -> config factory. Guards against option
    keys silently dropping between argparse and ExpertStreamingConfig
    (the count knob parsed but vanished in the 2026-07-17 sub-90 run)."""

    from types import SimpleNamespace

    from mtplx.expert_runtime import parse_memory_bytes

    bench = _load_benchmark_module()
    parser = bench._build_parser() if hasattr(bench, "_build_parser") else None
    if parser is None:
        parser = bench.build_parser() if hasattr(bench, "build_parser") else None
    if parser is None:
        import argparse as _argparse

        for name in dir(bench):
            fn = getattr(bench, name)
            if callable(fn) and name.endswith("parser"):
                candidate = fn()
                if isinstance(candidate, _argparse.ArgumentParser):
                    parser = candidate
                    break
    assert parser is not None, "benchmark parser factory not found"
    args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
            "--island-layer-count", "40",
        ]
    )
    options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(args),
    }
    assert options["island_layer_count"] == 40
    apis = SimpleNamespace(
        config_factory=ExpertStreamingConfig,
        parse_memory_bytes=parse_memory_bytes,
    )
    config = bench._runtime_config(apis, "hy3-expert-q2", options)
    assert config.island_layer_count == 40
    assert len(config.island_layers) == 40


def test_island_wave_call_matches_external_combine(mlx) -> None:
    """The fused K3 wave must be bitwise-identical to the classic island
    dispatch followed by the block's BF16 combine, at real Hy3 shapes."""

    import mlx.core as mx
    import numpy as np

    from types import SimpleNamespace

    from mtplx.models.expert_mlx import DenseIslandSwitchGLU

    hidden, expert_hidden, group = 4096, 1536, 64
    capacity, rows, top_k = 16, 4, 8
    rng = np.random.default_rng(65)

    def quantized(shape_out, shape_in):
        weight = mx.array(
            rng.integers(0, 2**32, size=(capacity, shape_out, shape_in // 16),
                         dtype=np.uint64).astype(np.uint32)
        )
        scales = mx.array(
            (rng.standard_normal((capacity, shape_out, shape_in // group)) * 0.01)
            .astype(np.float32)
        ).astype(mx.bfloat16)
        biases = mx.array(
            (rng.standard_normal((capacity, shape_out, shape_in // group)) * 0.01)
            .astype(np.float32)
        ).astype(mx.bfloat16)
        return weight, scales, biases

    arrays = {}
    for projection, (o, i) in (
        ("gate_proj", (expert_hidden, hidden)),
        ("up_proj", (expert_hidden, hidden)),
        ("down_proj", (hidden, expert_hidden)),
    ):
        w, s, b = quantized(o, i)
        arrays[f"{projection}.weight"] = w
        arrays[f"{projection}.scales"] = s
        arrays[f"{projection}.biases"] = b
    mx.eval(*arrays.values())

    spec = SimpleNamespace(
        top_k=top_k, hidden_size=hidden, quant_group_size=group,
        quant_bits=2, expert_count=capacity,
    )
    switch = DenseIslandSwitchGLU.__new__(DenseIslandSwitchGLU)
    switch.runtime = SimpleNamespace(spec=spec)
    switch.layer_index = 1
    switch.group_size = group
    switch.bits = 2
    switch._bank = SimpleNamespace(arrays=arrays, capacity=capacity)

    mx.random.seed(65)
    x = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
    indices = mx.array(
        rng.integers(0, capacity, size=(1, rows, top_k)).astype(np.uint32)
    )
    scores = mx.softmax(
        mx.random.normal((1, rows, top_k)), axis=-1
    ).astype(mx.bfloat16)

    fused = switch.wave_call(x, indices, scores)
    assert fused is not None, "wave declined an eligible K3 shape"

    routed = switch(x, indices)
    classic = (routed * scores[..., None]).sum(axis=-2)
    assert fused.shape == classic.shape
    assert mx.array_equal(fused, classic).item(), "wave combine diverged"

    # Ineligible shapes must decline, never raise.
    x1 = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    i1 = mx.array(rng.integers(0, capacity, size=(1, 1, top_k)).astype(np.uint32))
    s1 = mx.softmax(mx.random.normal((1, 1, top_k)), axis=-1).astype(mx.bfloat16)
    assert switch.wave_call(x1, i1, s1) is None
