"""Sweep t158 launch geometry on the retained GLM-5.2 28-row route."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mtplx.expert_shadow import _T158_BYTES_PER_GROUP, SHADOW_GROUP
from mtplx.kernels.shadow_gather import bind_shadow_gather_mm, shadow_gather_mm

ROUTE = (
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
)
PROJECTIONS = {
    "gate_up": (6144, 2048),
    "down": (2048, 6144),
}
THREAD_OPTIONS = (32, 64, 128, 256, 512)
STAGE_OPTIONS = (False, True)


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default = next(
        row for row in results if row["threads"] == 128 and row["stage"] is False
    )
    default_ms = float(default["median_ms"])
    ranked = [
        {**row, "over_default": float(row["median_ms"]) / default_ms} for row in results
    ]
    return sorted(ranked, key=lambda row: float(row["median_ms"]))


def _time_call(function: Any) -> int:
    started = time.perf_counter_ns()
    output = function()
    mx.eval(output)
    return time.perf_counter_ns() - started


def _projection_sweep(
    *,
    in_dim: int,
    out_dim: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    rows = len(ROUTE)
    capacity = max(ROUTE) + 1
    groups = in_dim // SHADOW_GROUP
    packed = mx.random.randint(
        low=0,
        high=243,
        shape=(capacity, out_dim, groups * _T158_BYTES_PER_GROUP),
        dtype=mx.uint8,
    )
    scales = mx.random.randint(
        low=0x3B00,
        high=0x3D00,
        shape=(capacity, out_dim, groups),
        dtype=mx.uint16,
    )
    values = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(rows, in_dim),
    ).astype(mx.bfloat16)
    slot_rows = mx.array(ROUTE, dtype=mx.int32)
    mx.eval(packed, scales, values, slot_rows)
    reference = shadow_gather_mm(
        values,
        slot_rows,
        packed,
        scales,
        codec="t158",
    )
    mx.eval(reference)

    launches: list[tuple[tuple[int, bool], Any]] = []
    for threads in THREAD_OPTIONS:
        for stage in STAGE_OPTIONS:
            bound = bind_shadow_gather_mm(
                codec="t158",
                dtype=mx.bfloat16,
                rows=rows,
                in_dim=in_dim,
                out_dim=out_dim,
                packed_shape=packed.shape,
                scales_shape=scales.shape,
                threads_per_tg=threads,
                stage=stage,
            )

            def launch(bound=bound) -> mx.array:
                return bound(values, slot_rows, packed, scales)

            output = launch()
            mx.eval(output)
            if not bool(mx.all(reference == output).item()):
                raise RuntimeError(
                    "geometry parity failed for "
                    f"K={in_dim}, N={out_dim}, threads={threads}, stage={stage}"
                )
            launches.append(((threads, stage), launch))

    for repeat in range(warmups):
        order = launches if repeat % 2 == 0 else reversed(launches)
        for _key, launch in order:
            _time_call(launch)

    timings = {key: [] for key, _launch in launches}
    for repeat in range(repeats):
        order = launches if repeat % 2 == 0 else reversed(launches)
        for key, launch in order:
            timings[key].append(_time_call(launch))
    results = [
        {
            "threads": threads,
            "stage": stage,
            "median_ms": statistics.median(values_ns) / 1_000_000.0,
            "mean_ms": statistics.fmean(values_ns) / 1_000_000.0,
            "min_ms": min(values_ns) / 1_000_000.0,
            "max_ms": max(values_ns) / 1_000_000.0,
        }
        for (threads, stage), values_ns in timings.items()
    ]
    ranked = _rank_results(results)
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "rows": rows,
        "parity": True,
        "warmups": warmups,
        "repeats": repeats,
        "ranked": ranked,
        "best": ranked[0],
        "default": next(
            row for row in ranked if row["threads"] == 128 and row["stage"] is False
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")

    mx.random.seed(158)
    projections = {
        name: _projection_sweep(
            in_dim=in_dim,
            out_dim=out_dim,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        for name, (in_dim, out_dim) in PROJECTIONS.items()
    }
    default_mix = (
        2 * projections["gate_up"]["default"]["median_ms"]
        + projections["down"]["default"]["median_ms"]
    )
    best_mix = (
        2 * projections["gate_up"]["best"]["median_ms"]
        + projections["down"]["best"]["median_ms"]
    )
    payload = {
        "schema": "glm52-t158-geometry-sweep-v1",
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_core_file": str(Path(mx.__file__).resolve()),
        "dispatch_census": os.environ.get("MLX_DISPATCH_CENSUS"),
        "route": {
            "rows": len(ROUTE),
            "unique_slots": len(set(ROUTE)),
        },
        "projections": projections,
        "independently_best_projection_mix": {
            "default_ms": default_mix,
            "best_ms": best_mix,
            "best_over_default": best_mix / default_mix,
            "speedup": default_mix / best_mix,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
