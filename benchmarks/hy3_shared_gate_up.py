#!/usr/bin/env python3
"""Benchmark issue #31 C2 packed shared gate/up on one exact Hy3 layer."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from mtplx.benchmarks.hy3_shared_gate_up_probe import (
    PackedQuantizedGateUp,
    pack_quantized_gate_up,
    run_packed_gate_up,
)


MODEL = Path(
    "/Users/davidtai/.cache/huggingface/hub/"
    "models--pipenetwork--Hy3-4bit/snapshots/"
    "160619d3f96c8470350b6dac0ef033a8381551e3"
)
SHARD = MODEL / "model-00001-of-00034.safetensors"
PREFIX = "model.layers.1.mlp.shared_mlp"


def _projection(weights: dict[str, mx.array], name: str) -> nn.QuantizedLinear:
    module = nn.QuantizedLinear(
        4096,
        1536,
        bias=False,
        group_size=64,
        bits=4,
        mode="affine",
    )
    base = f"{PREFIX}.{name}"
    module.weight = weights[f"{base}.weight"]
    module.scales = weights[f"{base}.scales"]
    module.biases = weights[f"{base}.biases"]
    return module


def _time_pair(
    control: Callable[[], mx.array],
    candidate: Callable[[], mx.array],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        mx.eval(control(), candidate())
    samples: dict[str, list[float]] = {"control": [], "candidate": []}
    functions = {"control": control, "candidate": candidate}
    for round_index in range(rounds):
        order = (
            ("control", "candidate")
            if round_index % 2 == 0
            else ("candidate", "control")
        )
        for name in order:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                mx.eval(functions[name]())
            elapsed = time.perf_counter_ns() - started
            samples[name].append(elapsed / iterations / 1_000.0)
    return samples["control"], samples["candidate"]


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "samples_us": samples,
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _bootstrap_mean_ci(samples: list[float], *, resamples: int) -> list[float]:
    rng = random.Random(312)
    means = sorted(
        statistics.fmean(rng.choice(samples) for _ in samples) for _ in range(resamples)
    )
    return [
        means[int(0.025 * resamples)],
        means[min(resamples - 1, int(0.975 * resamples))],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    weights = mx.load(str(SHARD))
    gate_proj = _projection(weights, "gate_proj")
    up_proj = _projection(weights, "up_proj")
    mx.eval(
        gate_proj.weight,
        gate_proj.scales,
        gate_proj.biases,
        up_proj.weight,
        up_proj.scales,
        up_proj.biases,
    )
    original_bytes = sum(
        int(array.nbytes)
        for module in (gate_proj, up_proj)
        for array in (module.weight, module.scales, module.biases)
    )
    pack_started = time.perf_counter_ns()
    packed: PackedQuantizedGateUp = pack_quantized_gate_up(gate_proj, up_proj)
    pack_us = (time.perf_counter_ns() - pack_started) / 1_000.0

    mx.random.seed(312)
    inputs = mx.random.normal((1, 1, 4096), dtype=mx.float32).astype(mx.bfloat16)
    mx.eval(inputs)

    def control() -> mx.array:
        return swiglu(gate_proj(inputs), up_proj(inputs))

    def candidate() -> mx.array:
        gate, up = run_packed_gate_up(inputs, packed)
        return swiglu(gate, up)

    expected = control()
    actual = candidate()
    mx.eval(expected, actual)
    if not mx.array_equal(expected, actual).item():
        raise RuntimeError("packed gate/up differs from two-QMM SwiGLU control")

    control_samples, candidate_samples = _time_pair(
        control,
        candidate,
        warmup=args.warmup,
        iterations=args.iterations,
        rounds=args.rounds,
    )
    speedups = [
        control_us / candidate_us
        for control_us, candidate_us in zip(
            control_samples,
            candidate_samples,
            strict=True,
        )
    ]
    result = {
        "schema": "mtplx-issue31-hy3-shared-gate-up-v1",
        "model": str(MODEL),
        "shard": str(SHARD),
        "layer": 1,
        "input_shape": [1, 1, 4096],
        "output_shape": [1, 1, 1536],
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "weights": {
            "original_bytes": original_bytes,
            "packed_bytes": packed.packed_bytes,
            "one_time_pack_us": pack_us,
            "temporary_pack_peak_bytes": original_bytes + packed.packed_bytes,
            "steady_candidate_requires_original_release": True,
        },
        "correctness": {"exact_swiglu_output": True},
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "bootstrap_resamples": args.bootstrap_resamples,
            "scope": "Python graph construction through evaluated SwiGLU output; candidate includes split",
            "control": _summary(control_samples),
            "candidate": _summary(candidate_samples),
            "paired_speedup": {
                "samples": speedups,
                "mean": statistics.fmean(speedups),
                "median": statistics.median(speedups),
                "bootstrap_mean_95_ci": _bootstrap_mean_ci(
                    speedups,
                    resamples=args.bootstrap_resamples,
                ),
            },
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
