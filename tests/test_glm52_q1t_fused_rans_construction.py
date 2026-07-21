from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
)
from mtplx.expert_streaming_models import GLM52_EXPERT_Q1T


def _strict_fused_manifest(tmp_path: Path):
    from mtplx.expert_rans import LANES, RANS_GUARD_BYTES, _HEADER_DTYPE
    from mtplx.glm52_q1t_over10 import GLM52_Q1T_BASE_MANIFEST_SHA256
    from mtplx.glm52_q1t_rans_artifact import (
        COMPONENT_ALIGNMENT,
        FUSED_RANS_FORMAT,
        FUSED_RANS_MODEL_KEY,
        FUSED_RANS_SOURCE_CODEC,
        FUSED_RANS_UNIFORM_PACKED_CODEC,
        FusedRansComponent,
        FusedRansLayer,
        Glm52Q1TFusedRansManifest,
    )

    geometries = (
        ("gate_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
        ("gate_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
        ("up_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
        ("up_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
        ("down_proj.packed", "U8", (6144, 416), 2048, 6144, 416),
        ("down_proj.scales", "U16", (6144, 32), 2048, 6144, 64),
    )
    offset = 0
    layers = []
    for layer in GLM52_EXPERT_Q1T.routed_layer_indices:
        components = []
        for name, dtype, shape, in_dim, out_dim, row_bytes in geometries:
            tiles = out_dim // LANES
            directory_offset = _HEADER_DTYPE.itemsize + 256 * 2
            payload_offset = directory_offset + 256 * tiles * LANES * 4
            length = payload_offset + 8 + RANS_GUARD_BYTES
            mapped_length = -(-length // COMPONENT_ALIGNMENT) * COMPONENT_ALIGNMENT
            components.append(
                FusedRansComponent(
                    component=name,
                    dtype=dtype,
                    shape=shape,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    row_bytes=row_bytes,
                    offset=offset,
                    length=length,
                    mapped_length=mapped_length,
                    raw_length=256 * out_dim * row_bytes,
                    sha256="c" * 64,
                    header_bytes=_HEADER_DTYPE.itemsize,
                    frequency_offset=_HEADER_DTYPE.itemsize,
                    directory_offset=directory_offset,
                    payload_offset=payload_offset,
                    payload_length=8,
                    guard_bytes=RANS_GUARD_BYTES,
                    record_count=256 * tiles,
                    lanes=LANES,
                    per_lane=row_bytes,
                )
            )
            offset += mapped_length
        layers.append(FusedRansLayer(layer=layer, components=tuple(components)))
    return Glm52Q1TFusedRansManifest(
        format=FUSED_RANS_FORMAT,
        model_key=FUSED_RANS_MODEL_KEY,
        codec=FUSED_RANS_UNIFORM_PACKED_CODEC,
        source_codec=FUSED_RANS_SOURCE_CODEC,
        source_model_key="glm52-expert-q2-q1t158",
        source_manifest_sha256=GLM52_Q1T_BASE_MANIFEST_SHA256,
        source_q1_parent_manifest_sha256="d" * 64,
        source_q1_manifest_sha256="e" * 64,
        file="experts-glm52-q1t-fused-rans.bin",
        file_bytes=offset,
        file_sha256="f" * 64,
        alignment=COMPONENT_ALIGNMENT,
        output_tile=LANES,
        expert_count=256,
        routed_layers=GLM52_EXPERT_Q1T.routed_layer_indices,
        layers=tuple(layers),
        path=tmp_path / "expert-manifest-glm52-q1t-fused-rans.json",
    )


def _fused_config(**overrides) -> ExpertStreamingConfig:
    values = {
        "model_key": "glm52-expert-q1t",
        "memory_limit_bytes": 96 * 1024**3,
        "runtime_reserve_bytes": 12 * 1024**3,
        "max_live_kv_tokens": 4096,
        "expert_cache_limit_bytes": 72 * 1024**3,
        "transient_slots": 48,
        "slot_layout": "fused-rans",
        "banked_manifest": "/tmp/glm52-q1t-fused-rans.json",
        "banked_codec": "rans32x-uniform-packed-v1",
        "streamed_codec": "none",
        "streamed_codec_manifest": None,
        "streamed_codec_verify": False,
        "cache_policy": "frequency",
        "cache_scope": "layer",
        "deferred_pin_release": True,
        "split_route_release": "deferred",
        "prefetch_slots": 0,
        "route_census": False,
        "resource_telemetry": False,
        "trace_routes": False,
        "bypass_page_cache": True,
        "verify_record_hashes": False,
        "verify_sidecar_hash_at_open": False,
    }
    values.update(overrides)
    return ExpertStreamingConfig(**values)


def test_fused_lane_memory_plan_exactly_matches_cache72_t158_control() -> None:
    config = _fused_config()
    plan = config.memory_plan(GLM52_EXPERT_Q1T)

    assert "execution_lane" not in config.__dataclass_fields__
    assert plan.total_limit_bytes == 96 * 1024**3
    assert plan.runtime_reserve_bytes == 12 * 1024**3
    assert plan.expert_cache_limit_bytes == 72 * 1024**3
    assert plan.cache_scope == "layer"
    assert plan.slots_per_layer == 116
    assert plan.persistent_slots == 116 * len(GLM52_EXPERT_Q1T.routed_layer_indices)
    assert plan.persistent_cache_bytes == (
        plan.persistent_slots * GLM52_EXPERT_Q1T.expert_record_bytes
    )
    assert plan.transient_slots == 48
    assert plan.transient_bytes == 48 * GLM52_EXPERT_Q1T.expert_record_bytes
    assert config.cache_policy == "frequency"
    assert config.deferred_pin_release is True
    assert config.split_route_release == "deferred"


def test_fused_rans_cache_policy_matches_the_t158_frequency_cache() -> None:
    from mtplx.expert_streaming import LayerExpertSlotBank, RoutingPhase
    from mtplx.models.glm52_q1t_fused_rans import _Glm52Q1TFrequencyCache

    control = LayerExpertSlotBank(
        expert_count=256,
        persistent_slots=116,
        transient_slots=48,
        frequency_decay=0.995,
        cache_policy="frequency",
    )
    candidate = _Glm52Q1TFrequencyCache(
        persistent_slots=116,
        transient_slots=48,
    )
    prefill = tuple((index * 37 + index // 5) % 256 for index in range(1024 * 8))
    control.prepare_prefill_seed(prefill)
    candidate.prepare_prefill(prefill)
    ordered = tuple(sorted(set(prefill)))
    waves = tuple(ordered[start : start + 48] for start in range(0, 256, 48))
    for experts in waves:
        expected = control.plan(experts, phase=RoutingPhase.PREFILL)
        actual = candidate.plan(experts, decode=False)
        assert actual.experts == expected.experts
        assert actual.slots == expected.slots
        assert actual.hits == expected.hits
        assert actual.misses == expected.misses
        assert tuple((load.expert, load.slot) for load in actual.loads) == tuple(
            (load.expert, load.slot) for load in expected.loads
        )
    for step in range(256):
        experts = tuple((step * 11 + lane * 29) % 256 for lane in range(24))
        expected = control.plan(experts, phase=RoutingPhase.DECODE)
        actual = candidate.plan(experts, decode=True)
        assert actual.experts == expected.experts
        assert actual.slots == expected.slots
        assert actual.hits == expected.hits
        assert actual.misses == expected.misses
        assert tuple((load.expert, load.slot) for load in actual.loads) == tuple(
            (load.expert, load.slot) for load in expected.loads
        )


def test_cached_fused_hot_path_has_no_invariant_validation_or_instrumentation() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    source = "\n".join(
        (
            inspect.getsource(fused.Glm52Q1TFusedRansCachedSwitchGLU._route_inputs),
            inspect.getsource(fused.Glm52Q1TFusedRansCachedSwitchGLU.__call__),
            inspect.getsource(
                fused.Glm52Q1TFusedRansCachedSwitchGLU.run_with_shared_overlap
            ),
            inspect.getsource(fused.Glm52Q1TFusedRansRuntime._execute_layer),
            inspect.getsource(fused.Glm52Q1TFusedRansRuntime.execute_layer),
            inspect.getsource(
                fused.Glm52Q1TFusedRansRuntime.execute_layer_with_shared
            ),
            inspect.getsource(fused._Glm52Q1TFrequencyCache.plan),
        )
    )
    for forbidden in (
        "validate",
        "eligible",
        "fallback",
        "retry",
        "counter",
        "os.environ",
        "getenv",
        "decode_component",
        "decode_container",
        "MlxComponentBank",
        "mx.zeros",
    ):
        assert forbidden not in source


def test_fused_lane_applies_allocator_and_wired_caps_from_total_plan() -> None:
    import mtplx.glm52_q1t_over10 as contract

    calls: list[tuple[str, int]] = []
    mx = SimpleNamespace(
        set_memory_limit=lambda value: calls.append(("memory", int(value))),
        set_wired_limit=lambda value: calls.append(("wired", int(value))),
    )
    env: dict[str, str] = {}
    plan = _fused_config().memory_plan(GLM52_EXPERT_Q1T)

    result = contract.apply_glm52_q1t_fused_rans_memory_caps(
        plan,
        mx_module=mx,
        env=env,
    )

    expected_mlx_limit = 84 * 1024**3
    assert calls == [
        ("memory", expected_mlx_limit),
        ("wired", expected_mlx_limit),
    ]
    assert result == {
        "total_limit_bytes": 96 * 1024**3,
        "runtime_reserve_bytes": 12 * 1024**3,
        "external_residency_bytes": 0,
        "mlx_memory_limit_bytes": expected_mlx_limit,
        "mlx_wired_limit_bytes": expected_mlx_limit,
    }
    assert env == {
        "MTPLX_MEMORY_LIMIT_BYTES": str(expected_mlx_limit),
        "MTPLX_WIRED_LIMIT_BYTES": str(expected_mlx_limit),
    }


def test_fused_lane_charges_the_complete_native_compressed_cache_to_96gib() -> None:
    import mtplx.glm52_q1t_over10 as contract

    calls: list[tuple[str, int]] = []
    mx = SimpleNamespace(
        set_memory_limit=lambda value: calls.append(("memory", int(value))),
        set_wired_limit=lambda value: calls.append(("wired", int(value))),
    )
    base_mlx_limit = 84 * 1024**3
    env = {
        "MTPLX_MEMORY_LIMIT_BYTES": str(base_mlx_limit),
        "MTPLX_WIRED_LIMIT_BYTES": str(base_mlx_limit),
    }
    plan = _fused_config().memory_plan(GLM52_EXPERT_Q1T)
    compressed_cache_bytes = plan.persistent_cache_bytes + plan.transient_bytes

    result = contract.apply_glm52_q1t_fused_rans_memory_caps(
        plan,
        mx_module=mx,
        env=env,
        external_residency_bytes=compressed_cache_bytes,
    )

    bounded_mlx_limit = base_mlx_limit - compressed_cache_bytes
    assert calls == [
        ("memory", bounded_mlx_limit),
        ("wired", bounded_mlx_limit),
    ]
    assert result["external_residency_bytes"] == compressed_cache_bytes
    assert result["mlx_memory_limit_bytes"] == bounded_mlx_limit
    assert result["mlx_wired_limit_bytes"] == bounded_mlx_limit
    assert env == {
        "MTPLX_MEMORY_LIMIT_BYTES": str(bounded_mlx_limit),
        "MTPLX_WIRED_LIMIT_BYTES": str(bounded_mlx_limit),
    }


def test_runtime_routes_memory_caps_only_for_the_fused_glm_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.runtime as runtime_module

    calls: list[tuple[str, object]] = []
    plan = object()
    mx = object()
    monkeypatch.setattr(
        "mtplx.glm52_q1t_over10.apply_glm52_q1t_fused_rans_memory_caps",
        lambda selected, *, mx_module, external_residency_bytes=0: calls.append(
            ("glm-fused", selected, external_residency_bytes)
        ),
    )
    monkeypatch.setattr(
        "mtplx.expert_runtime.apply_mlx_memory_cap",
        lambda selected, *, mx_module: calls.append(("generic", selected)),
    )

    runtime_module._apply_streaming_memory_cap(
        plan,
        SimpleNamespace(slot_layout="fused-rans"),
        mx_module=mx,
        external_residency_bytes=123,
    )
    runtime_module._apply_streaming_memory_cap(
        plan,
        SimpleNamespace(slot_layout="component-banks"),
        mx_module=mx,
    )

    assert calls == [("glm-fused", plan, 123), ("generic", plan)]


def test_runtime_applies_compressed_cache_cap_once_before_fused_runtime_open() -> None:
    import mtplx.runtime as runtime_module

    source = inspect.getsource(runtime_module._load_impl)
    planned = source.index("fused_external_residency_bytes =")
    capped = source.index(
        "fused_memory_cap_report = _apply_streaming_memory_cap("
    )
    opened = source.index("expert_runtime = _open_glm52_q1t_fused_rans_runtime(")

    assert planned < capped < opened
    assert "expert_runtime.memory_cap_report = _apply_streaming_memory_cap(" not in source


def test_runtime_uses_one_exact_cache72_plan_for_fused_cap_and_store() -> None:
    import mtplx.runtime as runtime_module

    plan = SimpleNamespace(slots_per_layer=116, transient_slots=48)
    calls = []

    class Config:
        def memory_plan(self, spec, **kwargs):
            calls.append((spec, kwargs))
            return plan

    selected = runtime_module._glm52_q1t_fused_rans_cache_plan(
        Config(),
        GLM52_EXPERT_Q1T,
    )

    assert selected is plan
    assert calls == [(GLM52_EXPERT_Q1T, {})]


def test_fused_lane_does_not_accept_the_old_record_transport_receipt() -> None:
    import mtplx.glm52_q1t_over10 as contract

    assert not hasattr(contract, "GLM52_Q1T_RANS_ARTIFACT_SHA256")
    assert not hasattr(contract, "Over10QualificationReceipt")
    assert not hasattr(contract, "QualifiedGeometryTable")


def test_fused_weight_path_has_no_decoded_record_or_component_bank_route() -> None:
    import mtplx.kernels.glm52_q1t_fused_rans as kernel
    import mtplx.models.glm52_q1t_fused_rans as model

    hot_source = "\n".join(
        (
            inspect.getsource(
                kernel.BoundGlm52Q1TFusedRansCachedBank.__call__
            ),
            inspect.getsource(
                kernel.BoundGlm52Q1TFusedRansCachedBankGateUp.__call__
            ),
            inspect.getsource(
                kernel.BoundGlm52Q1TFusedRansCachedBankDown.__call__
            ),
            inspect.getsource(model.Glm52Q1TFusedRansCachedSwitchGLU._route_inputs),
            inspect.getsource(model.Glm52Q1TFusedRansCachedSwitchGLU.__call__),
            inspect.getsource(
                model.Glm52Q1TFusedRansCachedSwitchGLU.run_with_shared_overlap
            ),
            inspect.getsource(model.Glm52Q1TFusedRansRuntime._execute_layer),
            inspect.getsource(model.Glm52Q1TFusedRansRuntime.execute_layer),
            inspect.getsource(
                model.Glm52Q1TFusedRansRuntime.execute_layer_with_shared
            ),
        )
    )
    source = "\n".join(
        (
            hot_source,
            inspect.getsource(model._prepare_fused_rans_component),
            inspect.getsource(model._glm52_q1t_rans_cache_sources),
            inspect.getsource(model._Glm52Q1TRansCacheReader.read_into),
            inspect.getsource(model.Glm52Q1TFusedRansCacheStore.prepare),
        )
    )
    for forbidden in (
        "PositionalExpertReader",
        "_read_record_decoded",
        "decode_component",
        "decode_container",
        "MlxComponentBank",
        "mx.zeros",
        "np.array(decoded)",
        "_decode_layer_banks",
        "os.environ",
        "getenv",
        "fallback",
        "retry",
        "_inactive_fused_rans_route",
        "route_slot",
    ):
        assert forbidden not in source
    for forbidden in (
        "mx.contiguous",
        ".astype(",
        "np.array(decoded)",
    ):
        assert forbidden not in hot_source
    assert "output_shapes=[(assignments, self.out_dim)]" in source
    assert "expert_ids" in hot_source
    assert "expert_slot_map" not in hot_source


def test_installed_cached_bank_call_has_no_validation_or_fallback() -> None:
    import mtplx.kernels.glm52_q1t_fused_rans as kernel

    source = inspect.getsource(
        kernel.BoundGlm52Q1TFusedRansCachedBank.__call__
    )
    for forbidden in (
        "if ",
        "_require_array",
        "isinstance",
        "raise ",
        "fallback",
        "retry",
        "getenv",
        "environ",
    ):
        assert forbidden not in source


def test_cached_switch_preserves_shared_expert_overlap_during_miss_io() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    assert hasattr(
        fused.Glm52Q1TFusedRansCachedSwitchGLU,
        "run_with_shared_overlap",
    )
    overlap = inspect.getsource(
        fused.Glm52Q1TFusedRansRuntime.execute_layer_with_shared
    )
    core = inspect.getsource(fused.Glm52Q1TFusedRansRuntime._execute_layer)
    assert "shared_work()" in overlap
    assert "self._async_eval(shared)" in overlap
    assert core.index("overlap(futures)") < core.index("future.result()")
    for forbidden in (
        "eligible",
        "fallback",
        "retry",
        "getenv",
        "environ",
        "validate",
    ):
        assert forbidden not in overlap


def test_cached_runtime_dispatches_each_ready_miss_before_remaining_reads_finish() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    source = inspect.getsource(fused.Glm52Q1TFusedRansRuntime._execute_layer)
    assert "for future in as_completed(futures):" in source
    assert "_dispatch_experts" not in source
    ready = source.index("ready_outputs, ready_positions = self._dispatch_assignments(")
    submit = source.index("self._async_eval(ready_outputs)")
    assert ready < submit


def test_cached_runtime_batches_mixed_all_hit_experts_into_one_bank_route() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    source = inspect.getsource(fused.Glm52Q1TFusedRansRuntime._dispatch_assignments)
    assert "route_for_layer" in source
    assert "expert_slot_map" not in source
    assert "dict.fromkeys" not in source
    assert "for expert in" not in source
    execute = inspect.getsource(fused.Glm52Q1TFusedRansRuntime._execute_layer)
    assert "self.store.install_route(layer, plan.experts, plan.slots)" in execute


def test_cached_runtime_forwards_evaluated_router_ids_without_rebuilding_them() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    routed = inspect.getsource(fused.Glm52Q1TFusedRansCachedSwitchGLU._route_inputs)
    dispatch = inspect.getsource(fused.Glm52Q1TFusedRansRuntime._dispatch_assignments)
    assert "routed_ids = indices.reshape(-1)" in routed
    assert "routed_ids" in dispatch
    assert "[plan.experts[index]" not in dispatch


def test_cached_runtime_returns_the_single_all_hit_bank_output_directly() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    source = inspect.getsource(fused.Glm52Q1TFusedRansRuntime._execute_layer)
    direct = source.index("return hit_outputs[0]")
    concatenated = source.index("grouped = mx.concatenate(outputs, axis=0)")
    assert direct < concatenated


def test_cached_transient_reuse_is_covered_by_the_next_router_eval() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    route_inputs = inspect.getsource(
        fused.Glm52Q1TFusedRansCachedSwitchGLU._route_inputs
    )
    switch = inspect.getsource(fused.Glm52Q1TFusedRansCachedSwitchGLU.__call__)
    assert route_inputs.index("mx.eval(indices)") < route_inputs.index(
        "return assignments, routed_ids, expert_ids, decode"
    )
    assert switch.index("self._route_inputs(x, indices)") < switch.index(
        "self.runtime.execute_layer("
    )


def test_installed_switch_keeps_fused_weight_route_out_of_model_parameters() -> None:
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansMlpRoute,
        Glm52Q1TFusedRansSwitchGLU,
    )

    payload = object()

    class Projection:
        packed_payload = payload

        def __call__(self, x, _expert_ids):
            return x

    projection = Projection()
    switch = Glm52Q1TFusedRansSwitchGLU(
        route=Glm52Q1TFusedRansMlpRoute(
            gate=projection,
            up=projection,
            down=projection,
        ),
        hidden_size=6144,
        top_k=8,
    )

    assert switch.parameters() == {}
    assert tuple(switch.items()) == ()
    assert switch.route.gate.packed_payload is payload


def test_fused_runtime_qualifies_once_at_construction_not_inline() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    construction = inspect.getsource(
        fused.construct_glm52_q1t_fused_rans_runtime
    )
    for required in (
        "validate_glm52_q1t_fused_rans_manifest(fused_manifest)",
        "verify_glm52_q1t_fused_rans_artifact(fused_manifest)",
        "run_glm52_q1t_fused_rans_self_checks(runtime)",
    ):
        assert construction.count(required) == 1

    hot_source = "\n".join(
        (
            inspect.getsource(fused.Glm52Q1TFusedRansCachedSwitchGLU.__call__),
            inspect.getsource(fused.Glm52Q1TFusedRansRuntime.execute_layer),
        )
    )
    for forbidden in (
        "validate_glm52_q1t_fused_rans_manifest",
        "verify_glm52_q1t_fused_rans_artifact",
        "run_glm52_q1t_fused_rans_self_checks",
        "_load_self_check_receipt",
    ):
        assert forbidden not in hot_source


def test_full_artifact_qualification_bypasses_the_page_cache() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    source = inspect.getsource(fused.verify_glm52_q1t_fused_rans_artifact)
    assert "fcntl.F_NOCACHE" in source

    admission = inspect.getsource(fused.Glm52Q1TFusedRansRuntime.admit_kv_tokens)
    for forbidden in ("if ", "raise ", "_kv_lock", "_live_kv_tokens"):
        assert forbidden not in admission


def test_fused_construction_binds_only_the_measured_real_shape_geometry() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    assert fused._QUALIFIED_REAL_SHAPE_REPORT_SHA256 == (
        "5eb92c6462f1ff10a8edf19347362848e65d91a9d8340616e09bee3ed00262df"
    )
    assert fused._qualified_projection_threadgroups() == {
        "gate_proj": 64,
        "up_proj": 64,
        "down_proj": 64,
    }
    assert fused._QUALIFIED_GATE_UP_THREADGROUPS == 64
    assert fused._QUALIFIED_ASSIGNMENT_COUNTS == (1, 2, 3, 8, 16, 24, 32)
    assert fused._SELF_CHECK_FORMAT == "mtplx-glm52-q1t-fused-rans-selfcheck-v3"
    assert fused._SELF_CHECK_SUFFIX == ".selfcheck-v3.json"
    assert "dict(_QUALIFIED_PROJECTION_THREADGROUPS)" in inspect.getsource(
        fused.construct_glm52_q1t_fused_rans_runtime
    )


def test_cache_slot_loader_copies_only_rans_directories_and_payloads(
    tmp_path: Path,
) -> None:
    import numpy as np

    import mtplx.models.glm52_q1t_fused_rans as fused

    binary = bytearray()
    components = []
    expected_directories = []
    expected_payloads = []
    for component_index in range(6):
        directory = np.zeros(64, dtype="<u4")
        directory[:32] = np.arange(32, dtype=np.uint32) % 8
        directory[32:] = 9 + np.arange(32, dtype=np.uint32) % 9
        payload = bytes(range(component_index * 18, component_index * 18 + 18))
        offset = len(binary)
        binary.extend(directory.tobytes())
        binary.extend(payload)
        components.append(
            SimpleNamespace(
                out_dim=32,
                record_count=2,
                offset=offset,
                directory_offset=0,
                payload_offset=directory.nbytes,
                payload_length=len(payload),
            )
        )
        expected_directories.append(directory[32:].tobytes())
        expected_payloads.append(payload[9:])
    path = tmp_path / "cache-source.bin"
    path.write_bytes(binary)
    artifact = SimpleNamespace(
        expert_count=2,
        layers=(SimpleNamespace(layer=3, components=tuple(components)),),
        bin_path=lambda: path,
    )

    sources = fused._glm52_q1t_rans_cache_sources(
        artifact,
        slot_bytes=4096,
        max_read_chunk_bytes=256,
    )
    source = sources[(3, 1)]
    assert source == fused._glm52_q1t_rans_cache_source(
        artifact,
        layer=3,
        expert=1,
        slot_bytes=4096,
        max_read_chunk_bytes=256,
    )
    destination = bytearray(4096)
    reader = fused._Glm52Q1TRansCacheReader(path)
    try:
        reader.read_into(source, memoryview(destination))
    finally:
        reader.close()

    header = np.frombuffer(destination[:72], dtype="<u4")
    for component_index in range(6):
        directory_offset, payload_offset, source_payload_offset = header[
            component_index * 3 : component_index * 3 + 3
        ]
        assert directory_offset % 4 == 0
        assert source_payload_offset == 9
        assert destination[
            directory_offset : directory_offset + 32 * 4
        ] == expected_directories[component_index]
        assert destination[
            payload_offset : payload_offset + 9
        ] == expected_payloads[component_index]
    assert source.image_bytes <= 4096 - 256 * 4


def test_cache_copy_geometry_is_rejected_before_reader_construction(
    tmp_path: Path,
) -> None:
    import numpy as np

    import mtplx.models.glm52_q1t_fused_rans as fused

    directory = np.zeros(64, dtype="<u4")
    directory[32:] = 8
    payload = bytes(range(16))
    path = tmp_path / "oversized-cache-copy.bin"
    path.write_bytes(directory.tobytes() + payload)
    component = SimpleNamespace(
        component="gate_proj.packed",
        out_dim=32,
        record_count=2,
        offset=0,
        directory_offset=0,
        payload_offset=directory.nbytes,
        payload_length=len(payload),
    )
    artifact = SimpleNamespace(
        expert_count=2,
        layers=(SimpleNamespace(layer=3, components=(component,) * 6),),
        bin_path=lambda: path,
    )

    with pytest.raises(
        fused.Glm52Q1TFusedRansConstructionError,
        match="read chunk",
    ):
        fused._glm52_q1t_rans_cache_sources(
            artifact,
            slot_bytes=4096,
            max_read_chunk_bytes=127,
        )

    reader_source = inspect.getsource(fused._Glm52Q1TRansCacheReader.read_into)
    assert "max_read_chunk" not in reader_source
    assert "copy.length >" not in reader_source


def test_glm_fused_rans_module_has_no_obsolete_mmap_store() -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    source = inspect.getsource(fused)
    for forbidden in (
        "class Glm52Q1TFusedRansStore",
        "class _Glm52Q1TFusedRansMappedBand",
        "def _map_fused_rans_component",
        "def glm52_q1t_fused_rans_mapping_band_bytes",
        "activate_region",
        "metal_u32_slice",
        "plan_region",
    ):
        assert forbidden not in source


def test_cache_construction_rejects_nonuniform_packed_table_once(
    tmp_path: Path,
) -> None:
    import numpy as np

    import mtplx.models.glm52_q1t_fused_rans as fused
    from mtplx.expert_rans import (
        RANS_CONTAINER_MAGIC,
        RANS_CONTAINER_VERSION,
        _HEADER_DTYPE,
    )

    component = _strict_fused_manifest(tmp_path).layers[0].components[0]
    metadata = bytearray(component.directory_offset)
    header = np.frombuffer(metadata, dtype=_HEADER_DTYPE, count=1)
    header["magic"] = RANS_CONTAINER_MAGIC
    header["version"] = RANS_CONTAINER_VERSION
    header["lanes"] = component.lanes
    header["expert_count"] = component.record_count
    header["seg_len"] = component.lanes * component.per_lane
    header["payload_len"] = component.payload_length
    frequency = np.full(256, 16, dtype="<u2")
    frequency[0] = 15
    frequency[1] = 17
    metadata[component.frequency_offset : component.directory_offset] = (
        frequency.tobytes()
    )
    path = tmp_path / "wrong-uniform-packed.bin"
    path.write_bytes(metadata)

    with pytest.raises(
        fused.Glm52Q1TFusedRansConstructionError,
        match="not uniform freq-16",
    ):
        fused._prepare_fused_rans_component(
            path,
            component,
            expert_count=256,
            require_uniform=True,
        )


def test_fused_manifest_requires_complete_real_glm_geometry(tmp_path: Path) -> None:
    from mtplx.models.glm52_q1t_fused_rans import (
        validate_glm52_q1t_fused_rans_manifest,
    )

    validate_glm52_q1t_fused_rans_manifest(_strict_fused_manifest(tmp_path))


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda value: replace(value, model_key="hy3-expert-q1t158"), "model"),
        (lambda value: replace(value, codec="rans32x-v1"), "codec"),
        (lambda value: replace(value, source_codec="b1"), "t158"),
        (lambda value: replace(value, source_manifest_sha256="0" * 64), "source"),
        (
            lambda value: replace(
                value,
                routed_layers=value.routed_layers[:-1],
                layers=value.layers[:-1],
            ),
            "layer",
        ),
        (
            lambda value: replace(
                value,
                layers=(
                    replace(value.layers[0], components=value.layers[0].components[:-1]),
                    *value.layers[1:],
                ),
            ),
            "component",
        ),
        (
            lambda value: replace(
                value,
                layers=(
                    replace(
                        value.layers[0],
                        components=(
                            replace(value.layers[0].components[0], shape=(2048, 416)),
                            *value.layers[0].components[1:],
                        ),
                    ),
                    *value.layers[1:],
                ),
            ),
            "geometry",
        ),
        (lambda value: replace(value, output_tile=16), "tile"),
    ),
)
def test_fused_manifest_rejects_wrong_identity_or_geometry(
    tmp_path: Path, mutation, match: str
) -> None:
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansConstructionError,
        validate_glm52_q1t_fused_rans_manifest,
    )

    with pytest.raises(Glm52Q1TFusedRansConstructionError, match=match):
        validate_glm52_q1t_fused_rans_manifest(
            mutation(_strict_fused_manifest(tmp_path))
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"model_key": "hy3-expert-q2"},
        {"model_key": "glm52-expert-q2"},
        {"model_key": "glm52-q4"},
        {"banked_codec": "none"},
        {"banked_manifest": None},
        {"expert_cache_limit_bytes": 64 * 1024**3},
        {"transient_slots": 32},
        {"cache_policy": "lru"},
        {"cache_scope": "global"},
        {"deferred_pin_release": False},
        {"split_route_release": "fenced"},
        {
            "streamed_codec": "rans32x-v1",
            "streamed_codec_manifest": "/tmp/old-record-rans.json",
        },
        {"route_census": True},
        {"resource_telemetry": True},
    ),
)
def test_fused_lane_rejects_non_glm_or_cache_transport_routes(overrides) -> None:
    with pytest.raises(
        (ExpertStreamingConfigurationError, ValueError),
        match="glm52-expert-q1t|fused-rans|model_key|banked",
    ):
        _fused_config(**overrides)


def test_stock_glm_q1t_component_banks_remain_a_separate_construction_route() -> None:
    config = _fused_config(
        slot_layout="component-banks",
        banked_codec="none",
        banked_manifest=None,
        expert_cache_limit_bytes=64 * 1024**3,
        transient_slots=8,
        verify_record_hashes=True,
        bypass_page_cache=False,
    )

    assert config.slot_layout == "component-banks"


def _fake_glm_model():
    layers = [
        SimpleNamespace(mlp=SimpleNamespace(switch_mlp=object()))
        for _index in range(78)
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_fused_runtime_binds_every_glm_q1t_routed_layer_directly() -> None:
    from mtplx.models.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_switches,
    )

    model = _fake_glm_model()
    switches = {
        layer: object() for layer in GLM52_EXPERT_Q1T.routed_layer_indices
    }
    runtime = SimpleNamespace(
        spec=GLM52_EXPERT_Q1T,
        switch_for_layer=switches.__getitem__,
    )

    bound = bind_glm52_q1t_fused_rans_switches(model, runtime)

    assert bound == 75
    for layer in GLM52_EXPERT_Q1T.routed_layer_indices:
        assert model.model.layers[layer].mlp.switch_mlp is switches[layer]


def test_resident_loader_exposes_construction_selected_switch_binder() -> None:
    from mtplx.resident_loader import construct_resident_model

    assert "switch_binder" in inspect.signature(construct_resident_model).parameters


def test_runtime_exact_lane_opener_uses_authoritative_and_fused_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.runtime as runtime_module

    base = object()
    fused = object()
    constructed = object()
    calls = []
    monkeypatch.setattr(
        "mtplx.expert_manifest.load_expert_manifest",
        lambda path: calls.append(("base", Path(path))) or base,
    )
    monkeypatch.setattr(
        "mtplx.glm52_q1t_rans_artifact.load_glm52_q1t_fused_rans_manifest",
        lambda path: calls.append(("fused", Path(path))) or fused,
    )

    def construct(**kwargs):
        calls.append(("construct", kwargs))
        return constructed

    monkeypatch.setattr(
        "mtplx.models.glm52_q1t_fused_rans.construct_glm52_q1t_fused_rans_runtime",
        construct,
    )
    config = _fused_config(
        banked_manifest="/tmp/expert-manifest-glm52-q1t-fused-rans.json"
    )
    cache_plan = object()

    result = runtime_module._open_glm52_q1t_fused_rans_runtime(
        expert_manifest="/tmp/expert-manifest.json",
        config=config,
        cache_plan=cache_plan,
    )

    assert result is constructed
    assert calls[:2] == [
        ("base", Path("/tmp/expert-manifest.json")),
        ("fused", Path(config.banked_manifest)),
    ]
    assert calls[2][1] == {
        "base_manifest": base,
        "fused_manifest": fused,
        "config": config,
        "cache_plan": cache_plan,
    }


def test_fused_runtime_constructs_compressed_cache_without_decoded_or_mmap_banks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mtplx.models.expert_mlx as expert_mlx
    import mtplx.models.glm52_q1t_fused_rans as fused

    manifest = _strict_fused_manifest(tmp_path)
    base = SimpleNamespace(
        model_key="glm52-expert-q1t",
        manifest_sha256=manifest.source_manifest_sha256,
        quant_mode="t158",
        quant_group_size=64,
        records=tuple(
            SimpleNamespace(layer=layer, expert=expert)
            for layer in GLM52_EXPERT_Q1T.routed_layer_indices
            for expert in range(GLM52_EXPERT_Q1T.expert_count)
        ),
        resident_tensors=(),
        resident_tensor_bytes=1,
    )

    class ForbiddenBank:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MlxComponentBank must not be constructed")

    class FakeStore:
        def __init__(
            self,
            artifact,
            *,
            plan,
            projection_threadgroups,
            max_read_chunk_bytes,
        ):
            assert artifact is manifest
            assert plan.slots_per_layer == 116
            assert plan.transient_slots == 48
            assert max_read_chunk_bytes == 8 * 1024**2
            assert projection_threadgroups == {
                "gate_proj": 64,
                "up_proj": 64,
                "down_proj": 64,
            }
            self.source_artifact_bytes = artifact.file_bytes
            self.table_bytes = 75 * 6 * (4096 + 256 * 4 * 2)
            self.projection_threadgroups = projection_threadgroups
            self.compressed_rans_persistent_cache_bytes = plan.persistent_cache_bytes
            self.compressed_rans_transient_bytes = plan.transient_bytes
            self.compressed_rans_allocated_bytes = (
                plan.persistent_cache_bytes + plan.transient_bytes
            )
            self.decoded_expert_cache_bytes = 0
            self.persistent_slots_per_layer = plan.slots_per_layer
            self.transient_slots = plan.transient_slots
            self.metal_buffer_count = 76
            self.metal_slot_view_count = 0
            self.max_cache_image_bytes = 7_000_000

        def prepare(self):
            return None

        def route_for_layer(self, layer):
            assert layer in GLM52_EXPERT_Q1T.routed_layer_indices
            return lambda x, _expert_ids, _expert_slots: x

        def close(self):
            return None

    monkeypatch.setattr(expert_mlx, "MlxComponentBank", ForbiddenBank)
    monkeypatch.setattr(fused, "Glm52Q1TFusedRansCacheStore", FakeStore)
    qualification_calls = []
    monkeypatch.setattr(
        fused,
        "verify_glm52_q1t_fused_rans_artifact",
        lambda artifact: qualification_calls.append(("integrity", artifact)) or 1.25,
    )

    def exact_self_check(installed):
        qualification_calls.append(("self-check", installed))
        installed.qualification_receipt_sha256 = "a" * 64
        return 2.5

    monkeypatch.setattr(
        fused,
        "run_glm52_q1t_fused_rans_self_checks",
        exact_self_check,
    )

    runtime = fused.construct_glm52_q1t_fused_rans_runtime(
        base_manifest=base,
        fused_manifest=manifest,
        config=_fused_config(),
    )
    runtime.memory_cap_report = {
        "total_limit_bytes": 96 * 1024**3,
        "external_residency_bytes": runtime.store.compressed_rans_allocated_bytes,
    }

    assert runtime.spec is GLM52_EXPERT_Q1T
    assert runtime.island_layer_set == frozenset()
    assert all(
        runtime.switch_for_layer(layer) is not None
        for layer in GLM52_EXPERT_Q1T.routed_layer_indices
    )
    snapshot = runtime.snapshot()
    assert snapshot["expert_codec"] == "rans32x-uniform-packed-v1"
    assert snapshot["memory_plan"]["decoded_expert_cache_bytes"] == 0
    assert snapshot["memory_plan"]["persistent_cache_bytes"] > 0
    assert snapshot["memory_plan"]["compressed_rans_persistent_cache_bytes"] == (
        snapshot["compressed_rans_persistent_cache_bytes"]
    )
    assert snapshot["memory_plan"]["compressed_rans_transient_bytes"] == (
        snapshot["compressed_rans_transient_bytes"]
    )
    assert snapshot["source_compressed_bytes"] == manifest.file_bytes
    assert snapshot["projection_threadgroups"] == {
        "gate_proj": 64,
        "up_proj": 64,
        "down_proj": 64,
    }
    assert snapshot["gate_up_threadgroups"] == 64
    assert snapshot["decoded_expert_cache_bytes"] == 0
    assert snapshot["persistent_slots_per_layer"] == 116
    assert snapshot["transient_slots"] == 48
    assert snapshot["metal_buffer_count"] == 76
    assert snapshot["metal_slot_view_count"] == 0
    for forbidden in (
        "mapped_compressed_bytes",
        "mapping_band_start_layers",
        "mapping_band_end_layers",
        "max_mapping_band_bytes",
        "metal_subbuffer_count",
    ):
        assert forbidden not in snapshot
    assert snapshot["memory_caps"] == runtime.memory_cap_report
    assert snapshot["construction"]["integrity_seconds"] == 1.25
    assert snapshot["construction"]["self_check_seconds"] == 2.5
    assert snapshot["construction"]["qualification_receipt_sha256"] == "a" * 64
    assert qualification_calls == [
        ("integrity", manifest),
        ("self-check", runtime),
    ]

    first = runtime.admit_kv_tokens(3000)
    second = runtime.admit_kv_tokens(1_000_000)
    assert first is second
    with first, second:
        pass
    runtime.close()


def test_fused_constructor_rejects_a_cache_plan_that_is_not_exact_cache72(
    tmp_path: Path,
) -> None:
    import mtplx.models.glm52_q1t_fused_rans as fused

    manifest = _strict_fused_manifest(tmp_path)
    base = SimpleNamespace(
        model_key="glm52-expert-q1t",
        manifest_sha256=manifest.source_manifest_sha256,
    )

    with pytest.raises(
        fused.Glm52Q1TFusedRansConstructionError,
        match="116-persistent/48-transient",
    ):
        fused.construct_glm52_q1t_fused_rans_runtime(
            base_manifest=base,
            fused_manifest=manifest,
            config=_fused_config(),
            cache_plan=SimpleNamespace(slots_per_layer=115, transient_slots=48),
        )


@pytest.mark.parametrize(
    "base",
    (
        SimpleNamespace(
            model_key="hy3-expert-q1t158",
            manifest_sha256="0" * 64,
            quant_mode="t158",
            quant_group_size=64,
            records=(),
        ),
        SimpleNamespace(
            model_key="glm52-expert-q1t",
            manifest_sha256="0" * 64,
            quant_mode="t158",
            quant_group_size=64,
            records=(),
        ),
    ),
)
def test_fused_runtime_rejects_wrong_authoritative_manifest(
    tmp_path: Path, base
) -> None:
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansConstructionError,
        construct_glm52_q1t_fused_rans_runtime,
    )

    with pytest.raises(Glm52Q1TFusedRansConstructionError, match="authoritative"):
        construct_glm52_q1t_fused_rans_runtime(
            base_manifest=base,
            fused_manifest=_strict_fused_manifest(tmp_path),
            config=_fused_config(),
        )


@pytest.mark.parametrize("model_key", ("hy3-expert-q2", "glm52-expert-q2"))
def test_fused_runtime_binder_rejects_other_model_families(model_key: str) -> None:
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansConstructionError,
        bind_glm52_q1t_fused_rans_switches,
    )

    model = _fake_glm_model()
    runtime = SimpleNamespace(
        spec=SimpleNamespace(
            key=model_key,
            routed_layer_indices=GLM52_EXPERT_Q1T.routed_layer_indices,
        ),
        switch_for_layer=lambda _layer: object(),
    )

    with pytest.raises(Glm52Q1TFusedRansConstructionError, match="glm52-expert-q1t"):
        bind_glm52_q1t_fused_rans_switches(model, runtime)


def test_construction_self_check_hashes_only_final_fused_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    import mtplx.models.glm52_q1t_fused_rans as fused

    manifest = _strict_fused_manifest(tmp_path)
    output_bits = np.arange(64, dtype=np.uint16)
    output_sha256 = hashlib.sha256(output_bits.tobytes()).hexdigest()
    vectors = [
        {
            "layer": layer,
            "seed": 61000 + layer,
            "expert_ids": list(range(GLM52_EXPERT_Q1T.top_k)),
            "output_sha256": output_sha256,
        }
        for layer in GLM52_EXPERT_Q1T.routed_layer_indices
    ]
    receipt = {
        "format": fused._SELF_CHECK_FORMAT,
        "model_key": "glm52-expert-q1t",
        "artifact_sha256": manifest.file_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "kernel_sha256": fused._self_check_kernel_sha256(),
        "route_kind": "cached-rans-t158-routed-bank",
        "qualified_report_sha256": fused._QUALIFIED_REAL_SHAPE_REPORT_SHA256,
        "qualified_assignment_counts": list(fused._QUALIFIED_ASSIGNMENT_COUNTS),
        "launch_threadgroups": {"gate_up": 64, "down": 64},
        "vectors": vectors,
    }

    def write_receipt() -> None:
        unsigned = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        receipt["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
        path = manifest.path.with_name(manifest.path.stem + fused._SELF_CHECK_SUFFIX)
        path.write_text(json.dumps(receipt))

    write_receipt()
    fake_mx = SimpleNamespace(
        array=lambda value, dtype=None: np.asarray(value),
        bfloat16=np.float32,
        int32=np.int32,
        uint16=np.uint16,
        eval=lambda *_values: None,
        view=lambda value, _dtype: value,
    )
    monkeypatch.setattr(fused, "mx", fake_mx)
    runtime = SimpleNamespace(
        fused_manifest=manifest,
        spec=GLM52_EXPERT_Q1T,
        store=SimpleNamespace(
            projection_threadgroups={
                "gate_proj": 64,
                "up_proj": 64,
                "down_proj": 64,
            }
        ),
        switch_for_layer=lambda _layer: (
            lambda _x, _indices: output_bits.copy()
        ),
    )

    assert fused.run_glm52_q1t_fused_rans_self_checks(runtime) >= 0

    receipt.pop("receipt_sha256")
    receipt["vectors"][0]["output_sha256"] = "0" * 64
    write_receipt()
    with pytest.raises(
        fused.Glm52Q1TFusedRansConstructionError,
        match="exact self-check failed",
    ):
        fused.run_glm52_q1t_fused_rans_self_checks(runtime)
