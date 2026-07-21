#!/usr/bin/env python3
"""Benchmark one real cached GLM Q1T rANS expert against raw t158 control.

The candidate slot contains only compressed rANS directories and payloads.
Each timed candidate call performs rANS decode, t158 decode, gate/up SwiGLU,
down projection, and FP32 accumulation inside two final-output dispatches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ASSIGNMENT_COUNTS = (1, 2, 3, 8, 16, 24, 32)
_CANDIDATE_THREADGROUPS = (32, 64, 128, 256, 512, 1024)
_COMPONENT_SHAPES = {
    "gate_proj": (6144, 2048),
    "up_proj": (6144, 2048),
    "down_proj": (2048, 6144),
}
_COMPONENT_GEOMETRY = (
    ("gate_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
    ("gate_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
    ("up_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
    ("up_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
    ("down_proj.packed", "U8", (6144, 416), 2048, 6144, 416),
    ("down_proj.scales", "U16", (6144, 32), 2048, 6144, 64),
)
_ROUTED_LAYERS = tuple(range(3, 78))
_CODEC = "rans32x-uniform-packed-v1"
_REPORT_FORMAT = "mtplx-glm52-q1t-cached-fused-rans-real-shape-v4"
_TIMING_PROTOCOL = "interleaved-candidate-shadow-v1"
_SELF_CHECK_FORMAT = "mtplx-glm52-q1t-fused-rans-selfcheck-v3"
_SELF_CHECK_SUFFIX = ".selfcheck-v3.json"
_SHA256_CHARS = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _digest_payload(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_real_shape_artifact_scope(manifest: Any, *, layer: int) -> str:
    """Accept only the uniform-packed GLM artifact and its exact real geometry."""

    if int(layer) not in _ROUTED_LAYERS:
        raise ValueError("real-shape benchmark requested layer is outside 3..77")
    if (
        manifest.format != "mtplx-glm52-q1t-fused-rans-v1"
        or manifest.model_key != "glm52-expert-q1t"
        or manifest.codec != _CODEC
        or manifest.source_codec != "t158"
        or manifest.expert_count != 256
        or manifest.output_tile != 32
        or manifest.alignment != 16384
    ):
        raise ValueError("real-shape artifact identity or geometry is incompatible")
    routed_layers = tuple(manifest.routed_layers)
    if routed_layers == _ROUTED_LAYERS:
        scope = "full"
    elif routed_layers == (int(layer),):
        scope = "probe"
    elif len(routed_layers) == 1:
        raise ValueError("real-shape artifact does not contain the requested layer")
    else:
        raise ValueError(
            "real-shape artifact must contain one requested layer or all routed layers"
        )
    if tuple(entry.layer for entry in manifest.layers) != routed_layers:
        raise ValueError("real-shape artifact layer directory is incompatible")
    for entry in manifest.layers:
        if len(entry.components) != 6:
            raise ValueError("real-shape artifact component coverage is incomplete")
        for component, expected in zip(
            entry.components,
            _COMPONENT_GEOMETRY,
            strict=True,
        ):
            name, dtype, shape, in_dim, out_dim, row_bytes = expected
            if (
                component.component != name
                or component.dtype != dtype
                or component.shape != shape
                or component.in_dim != in_dim
                or component.out_dim != out_dim
                or component.row_bytes != row_bytes
                or component.per_lane != row_bytes
                or component.record_count != 256 * (out_dim // 32)
                or component.raw_length != 256 * out_dim * row_bytes
                or component.lanes != 32
            ):
                raise ValueError(
                    f"real-shape artifact component geometry is incompatible: {name}"
                )
    return scope


@dataclass(frozen=True)
class ExpertSample:
    assignments: int
    gate_up_threadgroup: int
    down_threadgroup: int
    fused_median_ms: float
    fused_p95_ms: float
    shadow_median_ms: float
    shadow_p95_ms: float
    bitwise_equal: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "assignments": self.assignments,
            "gate_up_threadgroup": self.gate_up_threadgroup,
            "down_threadgroup": self.down_threadgroup,
            "fused_median_ms": self.fused_median_ms,
            "fused_p95_ms": self.fused_p95_ms,
            "shadow_median_ms": self.shadow_median_ms,
            "shadow_p95_ms": self.shadow_p95_ms,
            "fused_vs_shadow_ratio": self.fused_median_ms / self.shadow_median_ms,
            "bitwise_equal": self.bitwise_equal,
        }


def build_real_shape_report(
    samples: Iterable[ExpertSample],
    *,
    artifact_sha256: str,
    source_manifest_sha256: str,
    codec: str,
    layer: int,
    expert: int,
    cache_image_bytes: int,
    decoded_weight_outputs: int,
) -> dict[str, Any]:
    """Build a complete immutable cached-expert geometry receipt."""

    if not _is_sha256(artifact_sha256) or not _is_sha256(source_manifest_sha256):
        raise ValueError("real-shape report requires artifact SHA-256 identities")
    if codec != _CODEC:
        raise ValueError("real-shape report codec is unsupported")
    rows = tuple(samples)
    expected = {
        (gate_up, down, assignments)
        for gate_up in _CANDIDATE_THREADGROUPS
        for down in _CANDIDATE_THREADGROUPS
        for assignments in _ASSIGNMENT_COUNTS
    }
    actual = {
        (row.gate_up_threadgroup, row.down_threadgroup, row.assignments) for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError("real-shape sample coverage is incomplete")
    for row in rows:
        timings = (
            row.fused_median_ms,
            row.fused_p95_ms,
            row.shadow_median_ms,
            row.shadow_p95_ms,
        )
        if any(not math.isfinite(value) or value <= 0 for value in timings):
            raise ValueError("real-shape timings must be finite and positive")
        if not row.bitwise_equal:
            raise ValueError("real-shape sample is not bitwise equal")
    if decoded_weight_outputs != 0:
        raise ValueError("fused dispatch emitted a decoded weight output")
    selected_gate_up, selected_down = min(
        (
            (gate_up, down)
            for gate_up in _CANDIDATE_THREADGROUPS
            for down in _CANDIDATE_THREADGROUPS
        ),
        key=lambda pair: (
            sum(
                row.fused_median_ms
                for row in rows
                if (row.gate_up_threadgroup, row.down_threadgroup) == pair
            ),
            pair,
        ),
    )
    payload = {
        "format": _REPORT_FORMAT,
        "model_key": "glm52-expert-q1t",
        "codec": codec,
        "source_codec": "t158",
        "artifact_sha256": artifact_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "layer": int(layer),
        "expert": int(expert),
        "component_shapes": {
            name: {"in_dim": dimensions[0], "out_dim": dimensions[1]}
            for name, dimensions in _COMPONENT_SHAPES.items()
        },
        "assignment_counts": list(_ASSIGNMENT_COUNTS),
        "candidate_threadgroups": list(_CANDIDATE_THREADGROUPS),
        "timing_protocol": _TIMING_PROTOCOL,
        "selected_threadgroups": {
            "gate_up": selected_gate_up,
            "down": selected_down,
        },
        "cache_representation": "rans-directories-and-payloads",
        "cache_image_bytes": int(cache_image_bytes),
        "dispatches_per_routed_batch": 2,
        "kernel_consumes_routed_expert_ids": True,
        "output_shape": ["assignments", 6144],
        "output_count": 1,
        "decoded_weight_outputs": 0,
        "bitwise_equal": True,
        "samples": [
            row.to_json()
            for row in sorted(
                rows,
                key=lambda item: (
                    item.gate_up_threadgroup,
                    item.down_threadgroup,
                    item.assignments,
                ),
            )
        ],
    }
    payload["report_sha256"] = _digest_payload(payload)
    return payload


def load_qualified_real_shape_report(
    path: Path,
    *,
    expected_report_sha256: str,
    artifact_sha256: str,
    source_manifest_sha256: str,
    codec: str,
) -> dict[str, Any]:
    """Load one immutable bitwise cached-expert geometry receipt."""

    try:
        report = json.loads(path.expanduser().resolve().read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read qualified real-shape report: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("qualified real-shape report must be an object")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    digest = _digest_payload(unsigned)
    if (
        not _is_sha256(expected_report_sha256)
        or report.get("report_sha256") != digest
        or digest != expected_report_sha256
    ):
        raise ValueError("qualified real-shape report digest mismatch")
    if (
        report.get("format") != _REPORT_FORMAT
        or report.get("model_key") != "glm52-expert-q1t"
        or report.get("artifact_sha256") != artifact_sha256
        or report.get("source_manifest_sha256") != source_manifest_sha256
        or report.get("codec") != codec
        or report.get("source_codec") != "t158"
        or report.get("layer") != 3
        or report.get("assignment_counts") != list(_ASSIGNMENT_COUNTS)
        or report.get("candidate_threadgroups") != list(_CANDIDATE_THREADGROUPS)
        or report.get("timing_protocol") != _TIMING_PROTOCOL
        or report.get("cache_representation") != "rans-directories-and-payloads"
        or report.get("dispatches_per_routed_batch") != 2
        or report.get("kernel_consumes_routed_expert_ids") is not True
        or report.get("output_shape") != ["assignments", 6144]
        or report.get("output_count") != 1
        or report.get("decoded_weight_outputs") != 0
        or report.get("bitwise_equal") is not True
    ):
        raise ValueError("qualified real-shape report identity is incompatible")
    selected = report.get("selected_threadgroups")
    if (
        not isinstance(selected, dict)
        or set(selected) != {"gate_up", "down"}
        or any(value not in _CANDIDATE_THREADGROUPS for value in selected.values())
    ):
        raise ValueError("qualified real-shape launch geometry is incompatible")
    return report


def build_selfcheck_receipt(
    *,
    artifact_sha256: str,
    source_manifest_sha256: str,
    kernel_sha256: str,
    qualified_report_sha256: str,
    launch_threadgroups: dict[str, int],
    vectors: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Bind final cached-route hashes for all 75 routed layers."""

    for name, value in (
        ("artifact_sha256", artifact_sha256),
        ("source_manifest_sha256", source_manifest_sha256),
        ("kernel_sha256", kernel_sha256),
        ("qualified_report_sha256", qualified_report_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    normalized_threads = {
        str(name): int(threads) for name, threads in launch_threadgroups.items()
    }
    if set(normalized_threads) != {"gate_up", "down"} or any(
        threads not in _CANDIDATE_THREADGROUPS
        for threads in normalized_threads.values()
    ):
        raise ValueError("self-check launch threadgroups are incompatible")
    normalized = tuple(dict(vector) for vector in vectors)
    if tuple(vector.get("layer") for vector in normalized) != _ROUTED_LAYERS:
        raise ValueError("self-check layer coverage must be exactly 3..77")
    for vector in normalized:
        if set(vector) != {"layer", "seed", "expert_ids", "output_sha256"}:
            raise ValueError("self-check vector schema is incomplete")
        ids = vector["expert_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) != 8
            or any(
                isinstance(expert, bool)
                or not isinstance(expert, int)
                or expert < 0
                or expert >= 256
                for expert in ids
            )
        ):
            raise ValueError("self-check expert IDs are incompatible")
        if not _is_sha256(vector["output_sha256"]):
            raise ValueError("self-check output hash is invalid")
    payload = {
        "format": _SELF_CHECK_FORMAT,
        "model_key": "glm52-expert-q1t",
        "artifact_sha256": artifact_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "kernel_sha256": kernel_sha256,
        "route_kind": "cached-rans-t158-routed-bank",
        "qualified_report_sha256": qualified_report_sha256,
        "qualified_assignment_counts": list(_ASSIGNMENT_COUNTS),
        "launch_threadgroups": normalized_threads,
        "vectors": list(normalized),
    }
    payload["receipt_sha256"] = _digest_payload(payload)
    return payload


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _projection_bank(source, source_map, *, layer, experts, projection):
    leaves: dict[str, list[np.ndarray]] = {"packed": [], "scales": []}
    for expert in experts:
        record = source.record(layer, expert)
        segments = {segment.component: segment for segment in record.segments}
        for leaf, dtype in (("packed", np.uint8), ("scales", np.dtype("<u2"))):
            segment = segments[f"{projection}.{leaf}"]
            value = np.ndarray(
                segment.shape,
                dtype=dtype,
                buffer=source_map,
                offset=record.offset + segment.offset,
            )
            leaves[leaf].append(np.array(value, copy=True))
    return np.stack(leaves["packed"]), np.stack(leaves["scales"])


def _raw_banks(source, source_map, *, layer, experts):
    return {
        projection: _projection_bank(
            source,
            source_map,
            layer=layer,
            experts=experts,
            projection=projection,
        )
        for projection in _PROJECTIONS
    }


def _scale_tables(path, artifact, layer_entry, mx):
    from mtplx.models.glm52_q1t_fused_rans import _prepare_fused_rans_component

    prepared = tuple(
        _prepare_fused_rans_component(
            path,
            component,
            expert_count=artifact.expert_count,
            require_uniform=component.component.endswith(".packed"),
        )
        for component in layer_entry.components
    )
    del mx
    return {
        "gate": prepared[1].transition,
        "up": prepared[3].transition,
        "down": prepared[5].transition,
    }


def _bind_route(slot, tables, *, gate_up_threads, down_threads, mx):
    from mtplx.kernels.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_cached_bank,
    )

    return bind_glm52_q1t_fused_rans_cached_bank(
        persistent_bytes=slot,
        transient_bytes=slot,
        slot_bytes=int(slot.size),
        persistent_slots=1,
        gate_scales_transition=tables["gate"],
        up_scales_transition=tables["up"],
        down_scales_transition=tables["down"],
        hidden_size=6144,
        expert_hidden_size=2048,
        gate_up_threads_per_tg=gate_up_threads,
        down_threads_per_tg=down_threads,
        dtype=mx.bfloat16,
    )


def _shadow_call(x, ids, banks, *, mx):
    from mlx_lm.models.switch_layers import swiglu
    from mtplx.kernels.shadow_gather import shadow_gather_mm

    gate = shadow_gather_mm(x, ids, *banks["gate_proj"], codec="t158")
    up = shadow_gather_mm(x, ids, *banks["up_proj"], codec="t158")
    return shadow_gather_mm(
        swiglu(gate, up),
        ids,
        *banks["down_proj"],
        codec="t158",
    )


def _time_call_pair(candidate, control, *, mx, warmups, samples):
    calls = (candidate, control)
    for index in range(warmups):
        order = calls if index % 2 == 0 else calls[::-1]
        for call in order:
            mx.eval(call())
            mx.synchronize()
    timings = ([], [])
    for index in range(samples):
        order = (0, 1) if index % 2 == 0 else (1, 0)
        for call_index in order:
            started = time.perf_counter_ns()
            mx.eval(calls[call_index]())
            mx.synchronize()
            timings[call_index].append((time.perf_counter_ns() - started) / 1_000_000.0)

    def summarize(values):
        ordered = sorted(values)
        return float(median(values)), float(
            ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        )

    return summarize(timings[0]), summarize(timings[1])


def _benchmark_expert(
    *,
    raw_source,
    raw_map,
    layer,
    expert,
    slot,
    tables,
    warmups,
    samples,
    mx,
):
    raw = _raw_banks(raw_source, raw_map, layer=layer, experts=(expert,))
    banks = {
        name: (mx.array(packed), mx.array(scales))
        for name, (packed, scales) in raw.items()
    }
    shadow_calls = {}
    inputs = {}
    references = {}
    reference_ids = {}
    routed_ids = {}
    for assignments in _ASSIGNMENT_COUNTS:
        rng = np.random.default_rng(63000 + layer * 100 + expert + assignments)
        inputs[assignments] = mx.array(
            rng.standard_normal((assignments, 6144), dtype=np.float32),
            dtype=mx.bfloat16,
        )
        reference_ids[assignments] = mx.zeros(
            (assignments,),
            dtype=mx.int32,
        )
        routed_ids[assignments] = mx.array(
            [expert] * assignments,
            dtype=mx.uint32,
        )

        def shadow(count=assignments):
            return _shadow_call(
                inputs[count],
                reference_ids[count],
                banks,
                mx=mx,
            )

        shadow_calls[assignments] = shadow
        references[assignments] = shadow()
        mx.eval(references[assignments])
    rows = []
    for gate_up_threads in _CANDIDATE_THREADGROUPS:
        for down_threads in _CANDIDATE_THREADGROUPS:
            route = _bind_route(
                slot,
                tables,
                gate_up_threads=gate_up_threads,
                down_threads=down_threads,
                mx=mx,
            )
            for assignments in _ASSIGNMENT_COUNTS:

                def candidate(count=assignments, bound=route):
                    return bound(
                        inputs[count],
                        routed_ids[count],
                    )

                output = candidate()
                mx.eval(output)
                exact = bool(mx.array_equal(output, references[assignments]).item())
                (fused_median, fused_p95), (shadow_median, shadow_p95) = (
                    _time_call_pair(
                        candidate,
                        shadow_calls[assignments],
                        mx=mx,
                        warmups=warmups,
                        samples=samples,
                    )
                )
                rows.append(
                    ExpertSample(
                        assignments=assignments,
                        gate_up_threadgroup=gate_up_threads,
                        down_threadgroup=down_threads,
                        fused_median_ms=fused_median,
                        fused_p95_ms=fused_p95,
                        shadow_median_ms=shadow_median,
                        shadow_p95_ms=shadow_p95,
                        bitwise_equal=exact,
                    )
                )
    return tuple(rows)


def _reference_selfcheck_vectors(
    *,
    raw_source,
    raw_map,
    artifact,
    slot,
    reader,
    launch_threadgroups,
    mx,
):
    from mtplx.models.glm52_q1t_fused_rans import (
        _build_glm52_q1t_rans_cache_source,
        _layer_component_payload_offsets,
    )

    path = artifact.bin_path()
    host = memoryview(slot).cast("B")
    vectors = []
    expert_ids = (255, 0, 255, 64, 32, 1, 200, 17)
    unique = tuple(dict.fromkeys(expert_ids))
    try:
        for layer_entry in artifact.layers:
            layer = layer_entry.layer
            seed = 64000 + layer
            rng = np.random.default_rng(seed)
            logical = mx.array(
                rng.standard_normal((1, 1, 6144), dtype=np.float32),
                dtype=mx.bfloat16,
            )
            x = mx.broadcast_to(logical.reshape(1, 1, 6144), (1, 8, 6144))
            x = x.reshape(8, 6144)
            tables = _scale_tables(path, artifact, layer_entry, mx)
            offsets = _layer_component_payload_offsets(artifact, layer_entry)
            outputs = []
            positions = []
            for expert in unique:
                source = _build_glm52_q1t_rans_cache_source(
                    layer_entry,
                    expert,
                    offsets,
                    slot_bytes=int(slot.size),
                    max_read_chunk_bytes=8 * 1024**2,
                )
                reader.read_into(source, host)
                route = _bind_route(
                    slot,
                    tables,
                    gate_up_threads=launch_threadgroups["gate_up"],
                    down_threads=launch_threadgroups["down"],
                    mx=mx,
                )
                selected_positions = tuple(
                    index for index, value in enumerate(expert_ids) if value == expert
                )
                selected = mx.take(
                    x,
                    mx.array(selected_positions, dtype=mx.uint32),
                    axis=0,
                )
                selected_ids = mx.array(
                    [expert] * len(selected_positions),
                    dtype=mx.uint32,
                )
                route_table = host[-256 * 4 :].cast("I")
                route_table[expert] = 0
                route_table.release()
                output = route(selected, selected_ids)
                mx.eval(output)
                outputs.append(output)
                positions.extend(selected_positions)
            grouped = mx.concatenate(outputs, axis=0)
            inverse = [0] * len(positions)
            for grouped_position, original_position in enumerate(positions):
                inverse[original_position] = grouped_position
            candidate = mx.take(
                grouped,
                mx.array(inverse, dtype=mx.uint32),
                axis=0,
            )
            raw_experts = tuple(sorted(set(expert_ids)))
            local = {expert: index for index, expert in enumerate(raw_experts)}
            local_ids = mx.array(
                [local[expert] for expert in expert_ids],
                dtype=mx.int32,
            )
            raw = _raw_banks(
                raw_source,
                raw_map,
                layer=layer,
                experts=raw_experts,
            )
            banks = {
                name: (mx.array(packed), mx.array(scales))
                for name, (packed, scales) in raw.items()
            }
            reference = _shadow_call(x, local_ids, banks, mx=mx)
            mx.eval(candidate, reference)
            if not bool(mx.array_equal(candidate, reference).item()):
                raise RuntimeError(
                    f"layer {layer} cached self-check reference diverged"
                )
            bits = np.asarray(mx.view(reference, mx.uint16))
            vectors.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "expert_ids": list(expert_ids),
                    "output_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                }
            )
            del candidate, reference, banks, raw, outputs, grouped, x, logical
            gc.collect()
    finally:
        host.release()
    return tuple(vectors)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--q1-manifest", type=Path)
    parser.add_argument("--expert-manifest", type=Path)
    parser.add_argument("--fused-manifest", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--qualified-report", type=Path)
    parser.add_argument("--selfcheck-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    install_only = args.qualified_report is not None
    if install_only and (args.output_json is not None or args.selfcheck_output is None):
        raise ValueError(
            "--qualified-report requires --selfcheck-output and forbids --output-json"
        )
    if not install_only and args.output_json is None:
        raise ValueError("real-shape benchmarking requires --output-json")
    if args.layer not in _ROUTED_LAYERS or not 0 <= args.expert < 256:
        raise ValueError("focused layer/expert is outside GLM routed geometry")
    if args.warmups < 1 or args.samples < 3:
        raise ValueError("real-shape benchmark requires warmups >= 1 and samples >= 3")

    import mlx.core as mx

    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.expert_q1 import load_q1_manifest
    from mtplx.expert_streaming_models import GLM52_EXPERT_Q1T
    from mtplx.glm52_q1t_rans_artifact import (
        load_glm52_q1t_fused_rans_manifest,
    )
    from mtplx.mmap_mlx import allocate_metal_u8
    from mtplx.models.glm52_q1t_fused_rans import (
        _Glm52Q1TRansCacheReader,
        _QUALIFIED_REAL_SHAPE_REPORT_SHA256,
        _SELF_CHECK_SUFFIX as installed_selfcheck_suffix,
        _glm52_q1t_rans_cache_source,
        _self_check_kernel_sha256,
        validate_glm52_q1t_fused_rans_manifest,
    )

    root = args.model_root.expanduser().resolve()
    q1_path = args.q1_manifest or root / "expert-manifest-q1-t158.json"
    expert_path = args.expert_manifest or root / "expert-manifest.json"
    raw_source = load_q1_manifest(q1_path)
    authoritative = load_expert_manifest(expert_path)
    artifact = load_glm52_q1t_fused_rans_manifest(args.fused_manifest)
    scope = validate_real_shape_artifact_scope(artifact, layer=args.layer)
    if scope == "full":
        validate_glm52_q1t_fused_rans_manifest(artifact)
    elif install_only:
        raise ValueError("a partial probe artifact cannot create a self-check receipt")
    if (
        authoritative.model_key != "glm52-expert-q1t"
        or authoritative.manifest_sha256 != artifact.source_manifest_sha256
        or raw_source.codec != "t158"
        or raw_source.source_manifest_sha256
        != artifact.source_q1_parent_manifest_sha256
        or hashlib.sha256(q1_path.read_bytes()).hexdigest()
        != artifact.source_q1_manifest_sha256
    ):
        raise ValueError("real-shape benchmark source identity is incompatible")

    raw_map = np.memmap(raw_source.bin_path(), mode="r", dtype=np.uint8)
    slot = allocate_metal_u8(GLM52_EXPERT_Q1T.expert_record_bytes)
    host = memoryview(slot).cast("B")
    reader = _Glm52Q1TRansCacheReader(artifact.bin_path())
    try:
        if install_only:
            if installed_selfcheck_suffix != _SELF_CHECK_SUFFIX:
                raise ValueError("installed self-check suffix is not cache-v3")
            report = load_qualified_real_shape_report(
                args.qualified_report,
                expected_report_sha256=_QUALIFIED_REAL_SHAPE_REPORT_SHA256,
                artifact_sha256=artifact.file_sha256,
                source_manifest_sha256=artifact.source_manifest_sha256,
                codec=artifact.codec,
            )
            expected_path = artifact.path.with_name(
                artifact.path.stem + _SELF_CHECK_SUFFIX
            )
            if args.selfcheck_output.expanduser().resolve() != expected_path.resolve():
                raise ValueError(
                    "--selfcheck-output must be the new fused manifest sibling receipt"
                )
            vectors = _reference_selfcheck_vectors(
                raw_source=raw_source,
                raw_map=raw_map,
                artifact=artifact,
                slot=slot,
                reader=reader,
                launch_threadgroups=report["selected_threadgroups"],
                mx=mx,
            )
            receipt = build_selfcheck_receipt(
                artifact_sha256=artifact.file_sha256,
                source_manifest_sha256=artifact.source_manifest_sha256,
                kernel_sha256=_self_check_kernel_sha256(),
                qualified_report_sha256=report["report_sha256"],
                launch_threadgroups=report["selected_threadgroups"],
                vectors=vectors,
            )
            _write_json_new(args.selfcheck_output, receipt)
            print(f"cached fused-rANS self-check receipt: {args.selfcheck_output}")
            return 0

        source = _glm52_q1t_rans_cache_source(
            artifact,
            layer=args.layer,
            expert=args.expert,
            slot_bytes=GLM52_EXPERT_Q1T.expert_record_bytes,
            max_read_chunk_bytes=8 * 1024**2,
        )
        reader.read_into(source, host)
        route_table = host[-256 * 4 :].cast("I")
        route_table[args.expert] = 0
        route_table.release()
        layer_entry = next(
            entry for entry in artifact.layers if entry.layer == args.layer
        )
        tables = _scale_tables(artifact.bin_path(), artifact, layer_entry, mx)
        rows = _benchmark_expert(
            raw_source=raw_source,
            raw_map=raw_map,
            layer=args.layer,
            expert=args.expert,
            slot=slot,
            tables=tables,
            warmups=args.warmups,
            samples=args.samples,
            mx=mx,
        )
        report = build_real_shape_report(
            rows,
            artifact_sha256=artifact.file_sha256,
            source_manifest_sha256=artifact.source_manifest_sha256,
            codec=artifact.codec,
            layer=args.layer,
            expert=args.expert,
            cache_image_bytes=source.image_bytes,
            decoded_weight_outputs=0,
        )
        _write_json_new(args.output_json, report)
        print(
            f"cached fused-rANS real-shape report: {args.output_json}; "
            f"bitwise samples={len(rows)}"
        )
        return 0
    finally:
        reader.close()
        host.release()
        del raw_map, slot
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
