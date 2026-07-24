from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


def _load_module():
    name = "benchmark_glm52_q1t_fused_rans"
    path = Path(__file__).parents[1] / "scripts/benchmark_glm52_q1t_fused_rans.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_benchmark_entrypoint_uses_one_compressed_cache_slot_not_mmap_store() -> None:
    module = _load_module()
    source = inspect.getsource(module.main)
    bind_source = inspect.getsource(module._bind_route)

    assert "_glm52_q1t_rans_cache_source" in source
    assert "allocate_metal_u8" in source
    assert "bind_glm52_q1t_fused_rans_cached_bank" in bind_source
    assert "bind_glm52_q1t_fused_rans_cached_expert" not in bind_source
    assert "Glm52Q1TFusedRansStore" not in source
    assert "verify_glm52_q1t_fused_rans_artifact" not in source


def test_real_shape_benchmark_interleaves_candidate_and_control_samples() -> None:
    module = _load_module()
    source = inspect.getsource(module._benchmark_expert)

    assert "_time_call_pair(" in source
    assert "shadow_times = {}" not in source


def _samples(module):
    return tuple(
        module.ExpertSample(
            assignments=assignments,
            gate_up_threadgroup=gate_up,
            down_threadgroup=down,
            fused_median_ms=(abs(gate_up - 64) / 64 + abs(down - 128) / 128 + 1.0),
            fused_p95_ms=(abs(gate_up - 64) / 64 + abs(down - 128) / 128 + 1.1),
            shadow_median_ms=2.0,
            shadow_p95_ms=2.1,
            bitwise_equal=True,
        )
        for gate_up in module._CANDIDATE_THREADGROUPS
        for down in module._CANDIDATE_THREADGROUPS
        for assignments in module._ASSIGNMENT_COUNTS
    )


def test_real_shape_report_covers_full_cached_expert_geometry() -> None:
    module = _load_module()
    samples = _samples(module)

    report = module.build_real_shape_report(
        samples,
        artifact_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        codec="rans32x-uniform-packed-v1",
        layer=3,
        expert=17,
        cache_image_bytes=8_000_000,
        decoded_weight_outputs=0,
    )

    assert report["model_key"] == "glm52-expert-q1t"
    assert report["component_shapes"] == {
        "gate_proj": {"in_dim": 6144, "out_dim": 2048},
        "up_proj": {"in_dim": 6144, "out_dim": 2048},
        "down_proj": {"in_dim": 2048, "out_dim": 6144},
    }
    assert report["assignment_counts"] == list(module._ASSIGNMENT_COUNTS)
    assert report["timing_protocol"] == "interleaved-candidate-shadow-v1"
    assert report["selected_threadgroups"] == {"gate_up": 64, "down": 128}
    assert report["output_shape"] == ["assignments", 6144]
    assert report["output_count"] == 1
    assert report["dispatches_per_routed_batch"] == 2
    assert report["kernel_consumes_routed_expert_ids"] is True
    assert "dispatches_per_expert" not in report
    assert report["decoded_weight_outputs"] == 0
    assert report["bitwise_equal"] is True
    assert len(report["samples"]) == len(samples)
    assert len(report["report_sha256"]) == 64

    with pytest.raises(ValueError, match="coverage"):
        module.build_real_shape_report(
            samples[:-1],
            artifact_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            codec="rans32x-uniform-packed-v1",
            layer=3,
            expert=17,
            cache_image_bytes=8_000_000,
            decoded_weight_outputs=0,
        )
    with pytest.raises(ValueError, match="bitwise"):
        module.build_real_shape_report(
            (*samples[:-1], replace(samples[-1], bitwise_equal=False)),
            artifact_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            codec="rans32x-uniform-packed-v1",
            layer=3,
            expert=17,
            cache_image_bytes=8_000_000,
            decoded_weight_outputs=0,
        )


def test_selfcheck_v3_binds_cached_route_and_all_glm_layers() -> None:
    module = _load_module()
    vectors = tuple(
        {
            "layer": layer,
            "seed": 62000 + layer,
            "expert_ids": list(range(8)),
            "output_sha256": hashlib.sha256(str(layer).encode()).hexdigest(),
        }
        for layer in range(3, 78)
    )

    receipt = module.build_selfcheck_receipt(
        artifact_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        kernel_sha256="c" * 64,
        qualified_report_sha256="d" * 64,
        launch_threadgroups={"gate_up": 64, "down": 128},
        vectors=vectors,
    )

    assert receipt["format"] == "mtplx-glm52-q1t-fused-rans-selfcheck-v3"
    assert receipt["route_kind"] == "cached-rans-t158-routed-bank"
    assert receipt["launch_threadgroups"] == {"gate_up": 64, "down": 128}
    assert [vector["layer"] for vector in receipt["vectors"]] == list(range(3, 78))
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert (
        receipt["receipt_sha256"]
        == hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    with pytest.raises(ValueError, match="layer coverage"):
        module.build_selfcheck_receipt(
            artifact_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            kernel_sha256="c" * 64,
            qualified_report_sha256="d" * 64,
            launch_threadgroups={"gate_up": 64, "down": 128},
            vectors=vectors[:-1],
        )


def test_install_only_parser_targets_new_sibling_receipt() -> None:
    module = _load_module()
    args = module._parser().parse_args(
        [
            "--model-root",
            "/tmp/model",
            "--fused-manifest",
            "/tmp/fused.json",
            "--qualified-report",
            "/tmp/report.json",
            "--selfcheck-output",
            "/tmp/fused.selfcheck-v3.json",
        ]
    )

    assert args.qualified_report == Path("/tmp/report.json")
    assert args.output_json is None


def test_qualified_report_loader_rejects_tampering(tmp_path: Path) -> None:
    module = _load_module()
    report = module.build_real_shape_report(
        _samples(module),
        artifact_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        codec="rans32x-uniform-packed-v1",
        layer=3,
        expert=17,
        cache_image_bytes=8_000_000,
        decoded_weight_outputs=0,
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))

    loaded = module.load_qualified_real_shape_report(
        path,
        expected_report_sha256=report["report_sha256"],
        artifact_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        codec="rans32x-uniform-packed-v1",
    )
    assert loaded == report

    report["decoded_weight_outputs"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="digest"):
        module.load_qualified_real_shape_report(
            path,
            expected_report_sha256=loaded["report_sha256"],
            artifact_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            codec="rans32x-uniform-packed-v1",
        )


def _component(component, dtype, shape, in_dim, out_dim, row_bytes):
    return SimpleNamespace(
        component=component,
        dtype=dtype,
        shape=shape,
        in_dim=in_dim,
        out_dim=out_dim,
        row_bytes=row_bytes,
        per_lane=row_bytes,
        record_count=256 * (out_dim // 32),
        raw_length=256 * out_dim * row_bytes,
        lanes=32,
    )


def _probe_manifest(*, layers=(3,), codec="rans32x-uniform-packed-v1"):
    components = (
        _component("gate_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
        _component("gate_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
        _component("up_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
        _component("up_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
        _component("down_proj.packed", "U8", (6144, 416), 2048, 6144, 416),
        _component("down_proj.scales", "U16", (6144, 32), 2048, 6144, 64),
    )
    return SimpleNamespace(
        format="mtplx-glm52-q1t-fused-rans-v1",
        model_key="glm52-expert-q1t",
        codec=codec,
        source_codec="t158",
        expert_count=256,
        output_tile=32,
        alignment=16384,
        routed_layers=tuple(layers),
        layers=tuple(
            SimpleNamespace(layer=layer, components=components) for layer in layers
        ),
    )


def test_real_shape_scope_is_glm_only_and_exact_geometry() -> None:
    module = _load_module()
    assert module.validate_real_shape_artifact_scope(_probe_manifest(), layer=3) == (
        "probe"
    )
    assert (
        module.validate_real_shape_artifact_scope(
            _probe_manifest(layers=range(3, 78)), layer=40
        )
        == "full"
    )

    with pytest.raises(ValueError, match="requested layer"):
        module.validate_real_shape_artifact_scope(_probe_manifest(layers=(4,)), layer=3)
    wrong = _probe_manifest()
    wrong.model_key = "hy3-expert-q1t158"
    with pytest.raises(ValueError, match="identity"):
        module.validate_real_shape_artifact_scope(wrong, layer=3)
    wrong = _probe_manifest(codec="rans32x-v1")
    with pytest.raises(ValueError, match="identity"):
        module.validate_real_shape_artifact_scope(wrong, layer=3)
