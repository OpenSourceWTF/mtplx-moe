#!/usr/bin/env python3
"""Measure Issue #69 M=1 Hy3 MTP shared gate/up candidates.

The fast screen times the exact BF16 shared MLP from layer 80.  The recurrent
screen loads the complete BF16 MTP head and times the recurrent-hidden boundary
(enorm/hnorm/eh_proj, attention, routed+shared MLP, and final norm), excluding
only the trunk-owned embedding lookup and LM head.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from mtplx.hy3_mtp_patch import build_hy3_mtp_module
from mtplx.hy3_mtp_shared_gate_up import (
    Hy3MTPGateUpCandidate,
    MetalFusedMTPSharedMLP,
    hy3_mtp_gate_up_candidates,
    hy3_mtp_gate_up_savings,
)
from mtplx.models.expert_mlx import run_switch_with_shared_overlap
from mtplx.models.hy3_mlx import FusedSharedMLP, ModelArgs


DEFAULT_MODEL = Path("~/.cache/huggingface/hy3-expert-only-mlx-q2").expanduser()
DEFAULT_MTP = Path("~/.cache/huggingface/hy3-mtp-layer80").expanduser()
SOURCE_PREFIX = "model.layers.80.mlp.shared_mlp"


class InterleavedSharedMLP(nn.Module):
    """Benchmark-only G0,U0,G1,U1 packed row layout."""

    def __init__(self, gate: mx.array, up: mx.array, down: mx.array):
        super().__init__()
        self.width = int(gate.shape[0])
        self.gate_up_weight = mx.stack((gate, up), axis=1).reshape(
            2 * self.width, int(gate.shape[1])
        )
        self.down_weight = down

    def __call__(self, value: mx.array) -> mx.array:
        projected = mx.matmul(value, self.gate_up_weight.T).reshape(
            *value.shape[:-1], self.width, 2
        )
        return mx.matmul(
            swiglu(projected[..., 0], projected[..., 1]),
            self.down_weight.T,
        )


class BatchedSharedMLP(nn.Module):
    """Benchmark-only two-projection batched-matmul layout."""

    def __init__(self, gate: mx.array, up: mx.array, down: mx.array):
        super().__init__()
        self.gate_up_weight = mx.stack((gate, up), axis=0)
        self.down_weight = down

    def __call__(self, value: mx.array) -> mx.array:
        projected = mx.matmul(
            mx.broadcast_to(value[None, ...], (2, *value.shape)),
            self.gate_up_weight.swapaxes(-1, -2),
        )
        return mx.matmul(swiglu(projected[0], projected[1]), self.down_weight.T)


def _load_shared_weights(path: Path) -> tuple[mx.array, mx.array, mx.array]:
    tensors = mx.load(str(path))
    gate = tensors[f"{SOURCE_PREFIX}.gate_proj.weight"]
    up = tensors[f"{SOURCE_PREFIX}.up_proj.weight"]
    down = tensors[f"{SOURCE_PREFIX}.down_proj.weight"]
    mx.eval(gate, up, down)
    del tensors
    return gate, up, down


def _control_shared(
    gate: mx.array,
    up: mx.array,
    down: mx.array,
) -> nn.Module:
    class Control(nn.Module):
        def __call__(self, value: mx.array) -> mx.array:
            return mx.matmul(
                swiglu(mx.matmul(value, gate.T), mx.matmul(value, up.T)),
                down.T,
            )

    return Control()


def _block_packed_shared(
    args: ModelArgs,
    gate: mx.array,
    up: mx.array,
    down: mx.array,
) -> FusedSharedMLP:
    width = int(gate.shape[0])
    module = FusedSharedMLP(args, intermediate_size=width)
    module.gate_up_proj.weight = mx.concatenate((gate, up), axis=0)
    module.down_proj.weight = down
    mx.eval(module.parameters())
    return module


def _metal_candidate_map() -> dict[str, Hy3MTPGateUpCandidate]:
    """Expose coalesced arms plus exact MLX TN4-order arms by stable names."""

    candidates = hy3_mtp_gate_up_candidates(
        activation_modes=("exact", "fast"),
    )
    candidates += hy3_mtp_gate_up_candidates(
        activation_modes=("exact",),
        reduction_layouts=("stock_tn4",),
    )
    candidates += hy3_mtp_gate_up_candidates(
        activation_modes=("exact",),
        reduction_layouts=("stock_tn4",),
        input_modes=("direct",),
    )
    candidates += hy3_mtp_gate_up_candidates(
        activation_modes=("exact",),
        input_modes=("direct", "threadgroup_f32"),
    )
    return {f"metal_{candidate.name}": candidate for candidate in candidates}


def _candidate_shared(
    name: str,
    args: ModelArgs,
    gate: mx.array,
    up: mx.array,
    down: mx.array,
) -> nn.Module:
    if name == "block":
        return _block_packed_shared(args, gate, up, down)
    if name == "interleaved":
        module = InterleavedSharedMLP(gate, up, down)
    elif name == "batched":
        module = BatchedSharedMLP(gate, up, down)
    elif name.startswith("metal_"):
        metal_candidates = _metal_candidate_map()
        candidate = metal_candidates.get(name)
        if candidate is None:
            raise ValueError(f"unknown candidate {name!r}")
        module = MetalFusedMTPSharedMLP(
            gate,
            up,
            down,
            candidate=candidate,
        )
    else:
        raise ValueError(f"unknown candidate {name!r}")
    mx.eval(module.parameters())
    return module


def _module_names(
    candidates: tuple[str, ...],
    *,
    control_candidate: str,
) -> tuple[str, ...]:
    """Return candidate modules plus an optional non-stock control once."""

    names = candidates
    if control_candidate != "stock":
        names = (control_candidate, *names)
    return tuple(dict.fromkeys(names))


def _chain(
    module: Callable[[mx.array], mx.array], value: mx.array, depth: int
) -> mx.array:
    hidden = value
    for _ in range(depth):
        hidden = module(hidden)
    return hidden


def _recurrent_hidden(
    layer: Any,
    shared: Callable[[mx.array], mx.array],
    embedded: mx.array,
    previous: mx.array,
    depth: int,
) -> mx.array:
    hidden_state = previous
    token_state = embedded
    for _ in range(depth):
        mixed = layer.eh_proj(
            mx.concatenate(
                [layer.enorm(token_state), layer.hnorm(hidden_state)],
                axis=-1,
            )
        )
        block = layer.mtp_block
        hidden = mixed + block.self_attn(block.input_layernorm(mixed), None, None)
        mlp_input = block.post_attention_layernorm(hidden)
        mlp = block.mlp
        indices, scores = mlp.router(mlp_input)
        if not mlp.enable_moe_fp32_combine:
            scores = scores.astype(mlp_input.dtype)
        routed, shared_output = run_switch_with_shared_overlap(
            mlp.switch_mlp,
            mlp_input,
            indices,
            lambda: shared(mlp_input),
        )
        routed = (routed * scores[..., None]).sum(axis=-2)
        if mlp.enable_moe_fp32_combine:
            mlp_output = (
                routed.astype(mx.float32) + shared_output.astype(mx.float32)
            ).astype(mlp_input.dtype)
        else:
            mlp_output = routed.astype(mlp_input.dtype) + shared_output
        hidden_state = layer.final_layernorm(hidden + mlp_output)
        token_state = hidden_state
    return hidden_state


def _time_one(function: Callable[[], mx.array]) -> float:
    started = time.perf_counter_ns()
    mx.eval(function())
    return (time.perf_counter_ns() - started) / 1_000.0


def _time_pairs(
    control: Callable[[], mx.array],
    candidate: Callable[[], mx.array],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        mx.eval(control(), candidate())
    functions = {"control": control, "candidate": candidate}
    samples: dict[str, list[float]] = {"control": [], "candidate": []}
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
            samples[name].append(
                (time.perf_counter_ns() - started) / iterations / 1_000.0
            )
    return samples["control"], samples["candidate"]


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_us": samples,
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _bootstrap_mean_ci(samples: list[float], *, resamples: int) -> list[float]:
    rng = random.Random(6903)
    means = sorted(
        statistics.fmean(rng.choice(samples) for _ in samples) for _ in range(resamples)
    )
    return [
        means[int(0.025 * resamples)],
        means[min(resamples - 1, int(0.975 * resamples))],
    ]


def _device_info(core: Any) -> dict[str, Any]:
    """Return JSON-serializable metadata through MLX's current core API."""
    return json.loads(json.dumps(core.device_info(), default=str))


def _correctness(expected: mx.array, actual: mx.array) -> dict[str, Any]:
    mx.eval(expected, actual)
    difference = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    squared = difference * difference
    mx.eval(difference, squared)
    return {
        "array_equal": bool(mx.array_equal(expected, actual).item()),
        "max_abs_error": float(mx.max(difference).item()),
        "rmse": float(mx.sqrt(mx.mean(squared)).item()),
        "control_dtype": str(expected.dtype),
        "candidate_dtype": str(actual.dtype),
        "shape": list(expected.shape),
    }


def _measurement(
    control: Callable[[], mx.array],
    candidate: Callable[[], mx.array],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    control_cold_us = _time_one(control)
    candidate_cold_us = _time_one(candidate)
    expected = control()
    actual = candidate()
    correctness = _correctness(expected, actual)
    control_samples, candidate_samples = _time_pairs(
        control,
        candidate,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
    )
    speedups = [
        control_us / candidate_us
        for control_us, candidate_us in zip(
            control_samples,
            candidate_samples,
            strict=True,
        )
    ]
    return {
        "cold_us": {
            "control_first": control_cold_us,
            "candidate_second": candidate_cold_us,
            "order_caveat": "one-shot graph/compiler order; paired warm timings decide",
        },
        "correctness": correctness,
        "warm": {
            "control": _summary(control_samples),
            "candidate": _summary(candidate_samples),
            "paired_speedup": {
                "samples": speedups,
                "mean": statistics.fmean(speedups),
                "median": statistics.median(speedups),
                "bootstrap_mean_95_ci": _bootstrap_mean_ci(
                    speedups,
                    resamples=bootstrap_resamples,
                ),
            },
        },
    }


def main() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mtp-artifacts", type=Path, default=DEFAULT_MTP)
    parser.add_argument("--scope", choices=("shared", "recurrent"), default="shared")
    parser.add_argument(
        "--candidates",
        default="block,interleaved,batched",
        help="Comma-separated block, interleaved, and/or batched layouts.",
    )
    parser.add_argument(
        "--metal-frontier",
        action="store_true",
        help="Append all exact Metal M=1 n-tile/k-vector candidates.",
    )
    parser.add_argument(
        "--control-candidate",
        default="stock",
        help="Use stock or a named candidate as the paired timing control.",
    )
    parser.add_argument("--depths", default="1,3")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    config = json.loads((args.model_root.expanduser() / "config.json").read_text())
    model_args = ModelArgs.from_dict(config)
    candidates = tuple(
        part.strip() for part in args.candidates.split(",") if part.strip()
    )
    if args.metal_frontier:
        candidates += tuple(
            f"metal_{candidate.name}" for candidate in hy3_mtp_gate_up_candidates()
        )
    depths = tuple(int(part) for part in args.depths.split(",") if part.strip())
    if not candidates or not depths or any(depth < 1 for depth in depths):
        raise ValueError("at least one candidate and positive depth are required")
    module_names = _module_names(
        candidates,
        control_candidate=args.control_candidate,
    )

    gate, up, down = _load_shared_weights(
        args.mtp_artifacts.expanduser() / "layer80-bf16.safetensors"
    )
    control_shared = _control_shared(gate, up, down)
    candidate_modules = {
        name: _candidate_shared(name, model_args, gate, up, down)
        for name in module_names
    }
    mx.eval(control_shared.parameters())
    mx.random.seed(6903)
    value = mx.random.normal((1, 1, model_args.hidden_size)).astype(mx.bfloat16)
    embedded = mx.random.normal((1, 1, model_args.hidden_size)).astype(mx.bfloat16)
    previous = mx.random.normal((1, 1, model_args.hidden_size)).astype(mx.bfloat16)
    mx.eval(value, embedded, previous)

    layer = None
    if args.scope == "recurrent":
        mtp = build_hy3_mtp_module(
            args.mtp_artifacts.expanduser(),
            model_args,
            precision="bf16",
        )
        layer = mtp.layers[0]
        loaded_shared = layer.mtp_block.mlp.shared_mlp
        gate = loaded_shared.gate_proj.weight
        up = loaded_shared.up_proj.weight
        down = loaded_shared.down_proj.weight
        control_shared = loaded_shared
        candidate_modules = {
            name: _candidate_shared(name, model_args, gate, up, down)
            for name in module_names
        }

    measurement_control_shared = (
        control_shared
        if args.control_candidate == "stock"
        else candidate_modules[args.control_candidate]
    )

    measurements: dict[str, Any] = {}
    stage_correctness = {}
    expected_activation = swiglu(mx.matmul(value, gate.T), mx.matmul(value, up.T))
    for name, module in candidate_modules.items():
        activate = getattr(module, "activate", None)
        if callable(activate):
            try:
                stage_correctness[name] = _correctness(
                    expected_activation,
                    activate(value),
                )
            except Exception as exc:  # retain compiler failures in the frontier
                stage_correctness[name] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    for depth in depths:
        measurements[str(depth)] = {}
        for name in candidates:
            candidate_shared = candidate_modules[name]
            if layer is None:

                def control(d: int = depth) -> mx.array:
                    return _chain(measurement_control_shared, value, d)

                def candidate(
                    d: int = depth,
                    module: nn.Module = candidate_shared,
                ) -> mx.array:
                    return _chain(module, value, d)

            else:

                def control(d: int = depth) -> mx.array:
                    return _recurrent_hidden(
                        layer,
                        measurement_control_shared,
                        embedded,
                        previous,
                        d,
                    )

                def candidate(
                    d: int = depth,
                    module: nn.Module = candidate_shared,
                ) -> mx.array:
                    return _recurrent_hidden(
                        layer,
                        module,
                        embedded,
                        previous,
                        d,
                    )

            try:
                measurements[str(depth)][name] = _measurement(
                    control,
                    candidate,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    rounds=args.rounds,
                    bootstrap_resamples=args.bootstrap_resamples,
                )
            except Exception as exc:  # keep the exhaustive compiler frontier visible
                measurements[str(depth)][name] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

    shared_params = int(gate.size + up.size + down.size)
    gate_up_params = int(gate.size + up.size)
    metal_candidates = _metal_candidate_map()
    candidate_savings_at_k3 = {}
    for name in candidates:
        metal_candidate = metal_candidates.get(name)
        if metal_candidate is not None:
            candidate_savings_at_k3[name] = hy3_mtp_gate_up_savings(
                metal_candidate,
                depth=3,
            )
        else:
            candidate_savings_at_k3[name] = {
                "depth": 3,
                "logical_dispatches_saved": 3,
                "host_synchronizations_saved": 0,
                "steady_extra_weight_bytes": 0,
                "intermediate_device_bytes_avoided": 0,
            }
    result = {
        "schema": "mtplx-issue69-hy3-mtp-shared-gate-up-v1",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "metal_device": _device_info(mx),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip(),
        },
        "artifacts": {
            "model_root": str(args.model_root.expanduser().resolve()),
            "mtp_artifacts": str(args.mtp_artifacts.expanduser().resolve()),
            "mtp_bf16_file_bytes": (
                args.mtp_artifacts.expanduser() / "layer80-bf16.safetensors"
            )
            .stat()
            .st_size,
        },
        "scope": args.scope,
        "shape": {
            "M": 1,
            "hidden_size": model_args.hidden_size,
            "shared_intermediate_size": int(gate.shape[0]),
            "gate": list(gate.shape),
            "up": list(up.shape),
            "down": list(down.shape),
            "dtype": str(gate.dtype),
        },
        "parameters": {
            "gate_up": gate_up_params,
            "shared_mlp_total": shared_params,
            "gate_up_bytes": int(gate.nbytes + up.nbytes),
            "shared_mlp_total_bytes": int(gate.nbytes + up.nbytes + down.nbytes),
            "candidate_steady_extra_bytes_after_source_release": 0,
            "benchmark_extra_bytes_per_packed_candidate": int(gate.nbytes + up.nbytes),
            "benchmark_extra_bytes_per_metal_candidate": 0,
        },
        "dispatch_contract": {
            "control_gate_up_matmuls_per_depth": 2,
            "packed_candidate_gate_up_matmuls_per_depth": 1,
            "packed_projection_dispatches_saved_at_k3": 3,
            "metal_gate_up_swiglu_dispatches_saved_at_k3": 6,
            "candidate_savings_at_k3": candidate_savings_at_k3,
        },
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "bootstrap_resamples": args.bootstrap_resamples,
            "depths": list(depths),
            "control_candidate": args.control_candidate,
            "candidates": list(candidates),
            "gate_up_swiglu_correctness": stage_correctness,
            "scope_detail": (
                "shared gate/up, SwiGLU, and down"
                if layer is None
                else "recurrent hidden through full Hy3 MTP layer; shared trunk embedding and LM head excluded"
            ),
            "results": measurements,
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
