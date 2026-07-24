#!/usr/bin/env python3
"""Retired component-bank benchmark for the rejected GLM Q1T route.

The helpers remain as historical control code. The CLI fails closed because it
materializes ``MlxComponentBank`` weights; use
``benchmark_glm52_q1t_fused_rans.py`` for the raw-t158 control and direct
compressed candidate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import time
from typing import Any


CENSUS_SCHEMA = "mtplx-glm52-q1t-component-shape-census-v1"
OUTPUT_SCHEMA = "mtplx-shadow-gather-geometry-v1"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DECODE_PHASES = frozenset({"ar_decode", "decode_verify"})


@dataclass(frozen=True)
class GeometrySample:
    projection: str
    rows: int
    threads_per_tg: int
    stage: bool
    median_ms: float
    bitwise_equal: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "projection": self.projection,
            "rows": self.rows,
            "threads_per_tg": self.threads_per_tg,
            "stage": self.stage,
            "median_ms": self.median_ms,
            "bitwise_equal": self.bitwise_equal,
        }


@dataclass(frozen=True)
class DispatchCase:
    rows: int
    slot_pattern: tuple[int, ...]
    dispatches: int


@dataclass(frozen=True)
class CaseGeometrySample:
    projection: str
    rows: int
    slot_pattern: tuple[int, ...]
    dispatches: int
    threads_per_tg: int
    stage: bool
    median_ms: float
    bitwise_equal: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "projection": self.projection,
            "rows": self.rows,
            "slot_pattern": list(self.slot_pattern),
            "dispatches": self.dispatches,
            "threads_per_tg": self.threads_per_tg,
            "stage": self.stage,
            "median_ms": self.median_ms,
            "bitwise_equal": self.bitwise_equal,
        }


@dataclass(frozen=True)
class QualificationRequest:
    model_root: Path
    manifest_path: Path
    codec: str
    logical_rows: tuple[int, ...]
    dispatch_cases: tuple[DispatchCase, ...]
    threads: tuple[int, ...]
    stages: tuple[bool, ...]
    warmups: int
    samples: int
    layer: int | None


@dataclass(frozen=True)
class QualificationRun:
    model_key: str
    manifest_sha256: str
    samples: tuple[GeometrySample, ...]
    case_samples: tuple[CaseGeometrySample, ...] = ()


def choose_projection_geometry(samples: list[GeometrySample]) -> GeometrySample:
    exact = [sample for sample in samples if sample.bitwise_equal]
    if not exact:
        raise RuntimeError("no bitwise-exact shadow-gather geometry")
    return min(exact, key=lambda sample: sample.median_ms)


def geometry_digest(value: Any) -> str:
    """Digest the complete canonical qualification evidence."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dispatch_cases_from_census(payload: dict[str, Any]) -> tuple[DispatchCase, ...]:
    if payload.get("schema") != CENSUS_SCHEMA:
        raise ValueError(f"unexpected component-shape census schema: {payload.get('schema')!r}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("component-shape census records must be an array")
    aggregated: dict[tuple[int, tuple[int, ...]], int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"component-shape census record {index} must be an object")
        logical_m = record.get("logical_m")
        if isinstance(logical_m, bool) or not isinstance(logical_m, int) or logical_m <= 0:
            raise ValueError(
                f"component-shape census record {index} must have positive logical_m"
            )
        dispatches = record.get("dispatches")
        if isinstance(dispatches, bool) or not isinstance(dispatches, int) or dispatches <= 0:
            raise ValueError(
                f"component-shape census record {index} must have positive dispatches"
            )
        slot_pattern = record.get("slot_pattern")
        if (
            not isinstance(slot_pattern, list)
            or len(slot_pattern) != logical_m
            or any(
                isinstance(slot, bool) or not isinstance(slot, int) or slot < 0
                for slot in slot_pattern
            )
        ):
            raise ValueError(
                f"component-shape census record {index} has invalid slot_pattern"
            )
        phase = record.get("attention_phase")
        if not isinstance(phase, str) or not phase:
            raise ValueError(
                f"component-shape census record {index} has invalid attention_phase"
            )
        if phase in DECODE_PHASES:
            pattern = tuple(slot_pattern)
            identities: dict[int, int] = {}
            normalized = tuple(
                identities.setdefault(slot, len(identities)) for slot in pattern
            )
            if normalized != pattern:
                raise ValueError(
                    f"component-shape census record {index} has non-normalized slot_pattern"
                )
            key = (logical_m, pattern)
            aggregated[key] = aggregated.get(key, 0) + dispatches
    if not aggregated:
        raise ValueError("component-shape census must contain at least one logical M")
    return tuple(
        DispatchCase(rows=rows, slot_pattern=pattern, dispatches=dispatches)
        for (rows, pattern), dispatches in sorted(aggregated.items())
    )


def logical_rows_from_census(payload: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted({case.rows for case in dispatch_cases_from_census(payload)}))


def _weighted_median_ms(samples: list[CaseGeometrySample]) -> float:
    total = sum(sample.dispatches for sample in samples)
    cumulative = 0
    for sample in sorted(samples, key=lambda item: item.median_ms):
        cumulative += sample.dispatches
        if cumulative * 2 >= total:
            return sample.median_ms
    raise RuntimeError("weighted geometry sample set is empty")


def aggregate_case_samples(
    samples: list[CaseGeometrySample],
    *,
    dispatch_cases: tuple[DispatchCase, ...],
) -> list[GeometrySample]:
    expected_by_rows: dict[int, dict[tuple[int, ...], int]] = {}
    for case in dispatch_cases:
        expected_by_rows.setdefault(case.rows, {})[case.slot_pattern] = case.dispatches
    grouped: dict[tuple[str, int, int, bool], list[CaseGeometrySample]] = {}
    seen: set[tuple[str, int, int, bool, tuple[int, ...]]] = set()
    for sample in samples:
        expected = expected_by_rows.get(sample.rows, {}).get(sample.slot_pattern)
        if expected != sample.dispatches:
            raise RuntimeError("case sample does not match the census dispatch weight")
        group = (
            sample.projection,
            sample.rows,
            sample.threads_per_tg,
            sample.stage,
        )
        identity = (*group, sample.slot_pattern)
        if identity in seen:
            raise RuntimeError("duplicate case geometry sample")
        seen.add(identity)
        grouped.setdefault(group, []).append(sample)

    aggregated: list[GeometrySample] = []
    for (projection, rows, threads_per_tg, stage), case_samples in sorted(
        grouped.items()
    ):
        observed_patterns = {sample.slot_pattern for sample in case_samples}
        expected_patterns = set(expected_by_rows[rows])
        if observed_patterns != expected_patterns:
            raise RuntimeError(
                f"incomplete case geometry samples for rows={rows} "
                f"projection={projection} threads={threads_per_tg} stage={stage}"
            )
        aggregated.append(
            GeometrySample(
                projection=projection,
                rows=rows,
                threads_per_tg=threads_per_tg,
                stage=stage,
                median_ms=_weighted_median_ms(case_samples),
                bitwise_equal=all(sample.bitwise_equal for sample in case_samples),
            )
        )
    return aggregated


def build_qualification_payload(
    samples: list[GeometrySample],
    *,
    case_samples: Sequence[CaseGeometrySample] = (),
    logical_rows: tuple[int, ...],
    model_key: str,
    manifest_sha256: str,
    codec: str,
    census_sha256: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for rows in sorted(logical_rows):
        for projection in sorted(PROJECTIONS):
            candidates = [
                sample
                for sample in samples
                if sample.rows == rows and sample.projection == projection
            ]
            if not candidates:
                raise RuntimeError(
                    f"no shadow-gather samples for rows={rows} projection={projection}"
                )
            selected.append(choose_projection_geometry(candidates).to_json_dict())
    payload = {
        "schema": OUTPUT_SCHEMA,
        "model_key": model_key,
        "manifest_sha256": manifest_sha256,
        "codec": codec,
        "census_sha256": census_sha256,
        "logical_rows": list(sorted(logical_rows)),
        "samples": [
            sample.to_json_dict()
            for sample in sorted(
                samples,
                key=lambda item: (
                    item.rows,
                    item.projection,
                    item.threads_per_tg,
                    item.stage,
                ),
            )
        ],
        "case_samples": [
            sample.to_json_dict()
            for sample in sorted(
                case_samples,
                key=lambda item: (
                    item.rows,
                    item.projection,
                    item.threads_per_tg,
                    item.stage,
                    item.slot_pattern,
                ),
            )
        ],
        "selected": selected,
    }
    payload["geometry_sha256"] = geometry_digest(payload)
    return payload


def copy_record_payload(
    payload: bytes,
    record: Any,
    component_views: Sequence[memoryview],
) -> None:
    """Copy one verified record into component-major bank rows."""

    logical_bytes = int(record.logical_bytes)
    if len(payload) != logical_bytes:
        raise ValueError(
            f"record payload length {len(payload)} differs from {logical_bytes}"
        )
    segments = tuple(record.segments)
    if len(component_views) != len(segments):
        raise ValueError("component views do not cover every record segment")
    cursor = 0
    for index, (segment, view) in enumerate(zip(segments, component_views)):
        length = int(segment.length)
        if view.readonly or len(view) != length:
            raise ValueError(
                f"component view {index} is not writable with exact length {length}"
            )
        view[:] = payload[cursor : cursor + length]
        cursor += length
    if cursor != logical_bytes:
        raise ValueError("record segments do not cover the logical payload")


def _measure_projection_ms(
    launch: Callable[[], Any],
    *,
    mx: Any,
    warmups: int,
    samples: int,
) -> float:
    for _ in range(warmups):
        mx.eval(launch())
        mx.synchronize()
    elapsed_ms: list[float] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        mx.eval(launch())
        mx.synchronize()
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return float(median(elapsed_ms))


def _benchmark_component_bank(
    bank: Any,
    request: QualificationRequest,
    *,
    mx: Any,
) -> tuple[tuple[CaseGeometrySample, ...], tuple[GeometrySample, ...]]:
    import numpy as np

    from mtplx.expert_shadow import SHADOW_GROUP
    from mtplx.kernels.shadow_gather import (
        bind_shadow_gather_mm,
        shadow_gather_mm,
    )

    case_results: list[CaseGeometrySample] = []
    for dispatch_case in request.dispatch_cases:
        rows = dispatch_case.rows
        slot_rows = mx.array(dispatch_case.slot_pattern, dtype=mx.int32)
        for projection_index, projection in enumerate(PROJECTIONS):
            packed = bank.arrays[f"{projection}.packed"]
            scales = bank.arrays[f"{projection}.scales"]
            out_dim = int(packed.shape[1])
            in_dim = int(scales.shape[2]) * SHADOW_GROUP
            pattern_seed = sum(
                (index + 1) * slot
                for index, slot in enumerate(dispatch_case.slot_pattern)
            )
            rng = np.random.default_rng(
                rows * 1009 + projection_index * 97 + pattern_seed
            )
            host_x = rng.standard_normal((rows, in_dim)).astype(np.float32)
            x = mx.array(host_x, dtype=mx.bfloat16)
            mx.eval(x, slot_rows)
            reference = shadow_gather_mm(
                x,
                slot_rows,
                packed,
                scales,
                codec=request.codec,
            )
            mx.eval(reference)
            mx.synchronize()

            for threads_per_tg in request.threads:
                for stage in request.stages:
                    bound = bind_shadow_gather_mm(
                        codec=request.codec,
                        dtype=x.dtype,
                        rows=rows,
                        in_dim=in_dim,
                        out_dim=out_dim,
                        packed_shape=packed.shape,
                        scales_shape=scales.shape,
                        threads_per_tg=threads_per_tg,
                        stage=stage,
                    )
                    candidate = bound(x, slot_rows, packed, scales)
                    exact = bool(mx.array_equal(candidate, reference).item())
                    median_ms = _measure_projection_ms(
                        lambda bound=bound: bound(x, slot_rows, packed, scales),
                        mx=mx,
                        warmups=request.warmups,
                        samples=request.samples,
                    )
                    case_results.append(
                        CaseGeometrySample(
                            projection=projection,
                            rows=rows,
                            slot_pattern=dispatch_case.slot_pattern,
                            dispatches=dispatch_case.dispatches,
                            threads_per_tg=threads_per_tg,
                            stage=stage,
                            median_ms=median_ms,
                            bitwise_equal=exact,
                        )
                    )
    samples = aggregate_case_samples(
        case_results,
        dispatch_cases=request.dispatch_cases,
    )
    return tuple(case_results), tuple(samples)


def qualify_real_geometry(request: QualificationRequest) -> QualificationRun:
    """Load distinct real Q1T records and benchmark every census-derived M."""

    import mlx.core as mx

    from mtplx.expert_manifest import load_expert_manifest, read_expert_record
    from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot

    manifest = load_expert_manifest(request.manifest_path)
    if manifest.model_key != "glm52-expert-q1t":
        raise ValueError(
            f"geometry qualification requires glm52-expert-q1t, got {manifest.model_key!r}"
        )
    if manifest.quant_mode != request.codec:
        raise ValueError(
            f"manifest quantization mode {manifest.quant_mode!r} does not match "
            f"codec {request.codec!r}"
        )
    if manifest.manifest_sha256 is None:
        raise ValueError("geometry qualification requires a digested manifest")

    available_layers = sorted({record.layer for record in manifest.records})
    layer = available_layers[0] if request.layer is None else request.layer
    records = sorted(
        (record for record in manifest.records if record.layer == layer),
        key=lambda record: record.expert,
    )
    capacity = max(
        max(dispatch_case.slot_pattern) + 1
        for dispatch_case in request.dispatch_cases
    )
    if len(records) < capacity:
        raise ValueError(
            f"layer {layer} has {len(records)} records; census requires M={capacity}"
        )
    records = records[:capacity]
    prototype = records[0]
    bank = MlxComponentBank(
        capacity=capacity,
        record=prototype,
        label=f"geometry-layer-{layer}",
    )
    try:
        for bank_index, record in enumerate(records):
            payload = read_expert_record(
                manifest,
                request.model_root,
                record.layer,
                record.expert,
                verify_hash=True,
            )
            slot = MlxComponentSlot(
                bank,
                bank_index,
                label=f"geometry-layer-{layer}-expert-{record.expert}",
            )
            views = slot.record_views(record)
            try:
                copy_record_payload(payload, record, views)
            finally:
                for view in views:
                    view.release()
        mx.synchronize()
        case_samples, samples = _benchmark_component_bank(bank, request, mx=mx)
    finally:
        bank.close()
    return QualificationRun(
        model_key=manifest.model_key,
        manifest_sha256=manifest.manifest_sha256,
        samples=samples,
        case_samples=case_samples,
    )


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON source {path} must contain an object")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
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


def _parse_positive_csv(value: str, *, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed) or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(
            f"{label} must contain unique positive comma-separated integers"
        )
    return parsed


def main(
    argv: list[str] | None = None,
    *,
    qualify_fn: Callable[[QualificationRequest], QualificationRun] | None = None,
) -> int:
    if qualify_fn is None:
        raise RuntimeError(
            "retired GLM Q1T component-bank benchmark: use the fused-rANS "
            "real-record benchmark"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--census-json", type=Path, required=True)
    parser.add_argument("--codec", choices=("t158",), default="t158")
    parser.add_argument("--threads", default="64,128,256,512")
    parser.add_argument("--stage", choices=("both", "staged", "unstaged"), default="both")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmups < 1:
        parser.error("--warmups must be at least 1")
    if args.samples < 50:
        parser.error("--samples must be at least 50")
    try:
        threads = _parse_positive_csv(args.threads, label="--threads")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    stages = {
        "both": (False, True),
        "staged": (True,),
        "unstaged": (False,),
    }[args.stage]
    census = _read_json_object(args.census_json)
    dispatch_cases = dispatch_cases_from_census(census)
    logical_rows = tuple(sorted({case.rows for case in dispatch_cases}))
    census_sha256 = hashlib.sha256(
        json.dumps(census, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    model_root = args.model_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else model_root / "expert-manifest.json"
    )
    request = QualificationRequest(
        model_root=model_root,
        manifest_path=manifest_path,
        codec=args.codec,
        logical_rows=logical_rows,
        dispatch_cases=dispatch_cases,
        threads=threads,
        stages=stages,
        warmups=args.warmups,
        samples=args.samples,
        layer=args.layer,
    )
    run = (qualify_fn or qualify_real_geometry)(request)
    payload = build_qualification_payload(
        list(run.samples),
        case_samples=run.case_samples,
        logical_rows=logical_rows,
        model_key=run.model_key,
        manifest_sha256=run.manifest_sha256,
        codec=request.codec,
        census_sha256=census_sha256,
    )
    _write_json_atomic(args.output_json, payload)
    return 0


__all__ = [
    "CaseGeometrySample",
    "DispatchCase",
    "GeometrySample",
    "QualificationRequest",
    "QualificationRun",
    "aggregate_case_samples",
    "build_qualification_payload",
    "choose_projection_geometry",
    "copy_record_payload",
    "dispatch_cases_from_census",
    "geometry_digest",
    "logical_rows_from_census",
    "main",
    "qualify_real_geometry",
]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
