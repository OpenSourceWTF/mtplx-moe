#!/usr/bin/env python3
"""Isolated M1...M7 Hy3 source-BF16 router tiling benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Tiling,
    hy3_router_fp32_project,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_weight,
)
from mtplx.qwen_guard import DEFAULT_MLX_LOCK_PATH, exclusive_mlx_window


ROUTER_ROWS = tuple(range(1, 9))


def router_tiling_candidates() -> dict[str, Hy3RouterFP32Tiling]:
    """Exhaust the direct N-tile/K-part schedules supported by the kernel."""

    candidates = {}
    for n_tile in (16, 32, 64):
        for grid_k_parts in (1, 2, 4, 8, 16, 32):
            tiling = Hy3RouterFP32Tiling(n_tile, grid_k_parts, "direct")
            name = f"n{n_tile}_p{grid_k_parts}_tg{tiling.stage1_threadgroups}"
            candidates[name] = tiling
    return candidates


def grouped_router_tiling_candidates() -> dict[str, Hy3RouterFP32Tiling]:
    """Enumerate legal grouped schedules for the exact Hy3 router shape."""

    candidates = {}
    groups_by_n_tile = {
        16: (2, 3, 4, 6),
        32: (2, 3, 6),
        64: (3,),
    }
    for n_tile, group_counts in groups_by_n_tile.items():
        for grid_k_parts in (8, 16, 32):
            for groups in group_counts:
                for operand_mode, suffix in (
                    ("grouped-direct", "grouped_direct"),
                    ("grouped-staged", "staged"),
                ):
                    tiling = Hy3RouterFP32Tiling(
                        n_tile=n_tile,
                        grid_k_parts=grid_k_parts,
                        operand_mode=operand_mode,
                        simd_groups_per_threadgroup=groups,
                    )
                    name = f"n{n_tile}_p{grid_k_parts}_sg{groups}_{suffix}"
                    candidates[name] = tiling
    return candidates


def _candidate_failure_phase(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}".lower()
    if any(
        marker in message
        for marker in (
            "threadgroup memory",
            "resource limit",
            "out of memory",
            "too many threads",
        )
    ):
        return "resource"
    if any(
        marker in message
        for marker in (
            "compile",
            "compiler",
            "metal library",
            "program_source",
        )
    ):
        return "compile"
    return "dispatch"


def _candidate_failure_record(error: Exception) -> dict[str, str]:
    return {
        "status": "failed",
        "failure_phase": _candidate_failure_phase(error),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def evaluate_candidate_arms(
    functions: dict[str, Callable[[], Any]],
    *,
    evaluator: Callable[[Any], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Preflight every arm independently and retain structured failures."""

    evaluate = _evaluate if evaluator is None else evaluator
    results = {}
    for name, function in functions.items():
        try:
            evaluate(function())
        except Exception as error:
            results[name] = _candidate_failure_record(error)
        else:
            results[name] = {"status": "ok"}
    return results


def router_benchmark_candidates(
    frontier: str,
) -> dict[str, Hy3RouterFP32Tiling]:
    """Select a bounded screen while retaining the authoritative control."""

    direct = router_tiling_candidates()
    grouped = grouped_router_tiling_candidates()
    if frontier == "direct":
        return direct
    if frontier == "grouped":
        return {"n16_p8_tg96": direct["n16_p8_tg96"], **grouped}
    if frontier == "all":
        return {**direct, **grouped}
    raise ValueError("router frontier must be direct, grouped, or all")


def mpp_frontier_comparison_pairs(
    candidates: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Directly pair every supported schedule with the retained N16/P8 arm."""

    incumbent = "n16_p8_tg96"
    names = (
        tuple(router_tiling_candidates()) if candidates is None else tuple(candidates)
    )
    if incumbent not in names:
        raise ValueError("MPP frontier requires the retained N16/P8 incumbent")
    return tuple(
        (incumbent, candidate) for candidate in names if candidate != incumbent
    )


def rotated_arm_order(names: tuple[str, ...], repeat: int) -> tuple[str, ...]:
    """Rotate, then reverse by complete cycles, to balance timing position."""

    if not names:
        raise ValueError("router timing requires at least one arm")
    offset = repeat % len(names)
    order = names[offset:] + names[:offset]
    if (repeat // len(names)) % 2:
        order = tuple(reversed(order))
    return order


def sample_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize positive millisecond samples without hiding their tails."""

    samples = [float(value) for value in values]
    if not samples or any(value <= 0.0 for value in samples):
        raise ValueError("router timing samples must be nonempty and positive")
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def paired_comparison(
    control_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Report control/candidate speedup using paired repeated samples."""

    control = np.asarray(control_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if (
        control.ndim != 1
        or candidate.ndim != 1
        or control.size == 0
        or control.size != candidate.size
        or np.any(control <= 0.0)
        or np.any(candidate <= 0.0)
    ):
        raise ValueError("paired router samples must be equal positive vectors")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    ratios = control / candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        control.size,
        size=(bootstrap_resamples, control.size),
    )
    boot = control[indices].mean(axis=1) / candidate[indices].mean(axis=1)
    return {
        "paired_ratio_mean": float(ratios.mean()),
        "paired_ratio_median": float(np.median(ratios)),
        "ratio_of_means": float(control.mean() / candidate.mean()),
        "bootstrap_mean_ratio_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
    }


def named_pairwise_comparisons(
    samples: dict[str, Sequence[float]],
    *,
    pairs: Sequence[tuple[str, str]],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Compare explicitly named control/candidate timing boundaries."""

    result = {}
    for index, (control, candidate) in enumerate(pairs):
        if control not in samples or candidate not in samples:
            raise ValueError("named router comparison references a missing arm")
        name = f"{control}_over_{candidate}"
        if name in result:
            raise ValueError("named router comparisons must be unique")
        result[name] = paired_comparison(
            samples[control],
            samples[candidate],
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + index,
        )
    return result


def router_candidate_passes(
    *,
    candidate_topk_exact: bool,
    candidate_weights_valid: bool,
    within_mode_deterministic: bool,
    bootstrap_mean_ratio_95_ci: Sequence[float],
) -> bool:
    """Gate one authoritative arithmetic mode against its own contract."""

    if len(bootstrap_mean_ratio_95_ci) != 2:
        raise ValueError("router speed interval must contain two bounds")
    return (
        bool(candidate_topk_exact)
        and bool(candidate_weights_valid)
        and bool(within_mode_deterministic)
        and float(bootstrap_mean_ratio_95_ci[0]) > 1.0
    )


def authoritative_candidate_contract(
    *,
    candidate_topk_exact: bool,
    max_candidate_route_weight_abs_error: float,
    weights_finite: bool,
    max_normalized_sum_abs_error: float,
    repeated_ids_exact: bool,
    repeated_weights_exact: bool,
    route_weight_tolerance: float = 5e-4,
    normalization_tolerance: float = 5e-5,
) -> dict[str, bool]:
    """Classify MPP output against its own deterministic FP32 contract."""

    if min(route_weight_tolerance, normalization_tolerance) < 0.0:
        raise ValueError("authoritative router tolerances must be non-negative")
    weight_error = float(max_candidate_route_weight_abs_error)
    normalization_error = float(max_normalized_sum_abs_error)
    candidate_weights_valid = (
        bool(weights_finite)
        and math.isfinite(weight_error)
        and math.isfinite(normalization_error)
        and 0.0 <= weight_error <= route_weight_tolerance
        and 0.0 <= normalization_error <= normalization_tolerance
    )
    return {
        "candidate_topk_exact": bool(candidate_topk_exact),
        "candidate_weights_valid": candidate_weights_valid,
        "within_mode_deterministic": bool(repeated_ids_exact)
        and bool(repeated_weights_exact),
    }


def _load_safetensor(path: Path, key: str) -> tuple[mx.array, dict[str, Any]]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
        descriptor = header[key]
        start, end = descriptor["data_offsets"]
        handle.seek(8 + header_length + int(start))
        raw = handle.read(int(end) - int(start))
    shape = tuple(int(value) for value in descriptor["shape"])
    dtype = str(descriptor["dtype"])
    if dtype == "BF16":
        words = mx.array(np.frombuffer(raw, dtype="<u2").copy())
        value = words.view(mx.bfloat16).reshape(shape)
    elif dtype == "F32":
        value = mx.array(np.frombuffer(raw, dtype="<f4").copy()).reshape(shape)
    else:
        raise ValueError(f"unsupported router tensor dtype: {dtype}")
    return value, {
        "key": key,
        "shard": path.name,
        "shape": list(shape),
        "dtype": dtype,
        "bytes": len(raw),
    }


def _load_router(
    model: Path,
    *,
    layer: int,
) -> tuple[mx.array, mx.array, dict[str, Any]]:
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    weight_key = f"model.layers.{layer}.mlp.router.gate.weight"
    bias_key = f"model.layers.{layer}.mlp.router.expert_bias"
    weight, weight_metadata = _load_safetensor(
        model / weight_map[weight_key],
        weight_key,
    )
    bias, bias_metadata = _load_safetensor(
        model / weight_map[bias_key],
        bias_key,
    )
    return (
        weight,
        bias,
        {
            "index": str(index_path),
            "weight": weight_metadata,
            "expert_bias": bias_metadata,
        },
    )


def _evaluate(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def _measure_arms(
    functions: dict[str, Callable[[], Any]],
    *,
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
    seed: int,
    comparison_pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    if min(warmups, repeats, bootstrap_resamples) <= 0:
        raise ValueError("router measurement counts must be positive")
    failures: dict[str, dict[str, str]] = {}
    active_functions = {}
    for name, function in functions.items():
        try:
            for _ in range(warmups):
                _evaluate(function())
        except Exception as error:
            if name == "stock":
                raise RuntimeError("stock router timing arm failed") from error
            failures[name] = _candidate_failure_record(error)
        else:
            active_functions[name] = function
    names = tuple(active_functions)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(repeats):
        for name in rotated_arm_order(names, repeat):
            if name in failures:
                continue
            started = time.perf_counter_ns()
            try:
                _evaluate(active_functions[name]())
            except Exception as error:
                if name == "stock":
                    raise RuntimeError("stock router timing arm failed") from error
                failures[name] = _candidate_failure_record(error)
                continue
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)
    samples = {
        name: values
        for name, values in samples.items()
        if name not in failures and len(values) == repeats
    }
    comparisons = {
        name: paired_comparison(
            samples["stock"],
            values,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + index,
        )
        for index, (name, values) in enumerate(samples.items())
        if name != "stock"
    }
    available_pairs = tuple(
        (control, candidate)
        for control, candidate in comparison_pairs
        if control in samples and candidate in samples
    )
    return {
        "arms": {name: sample_summary(values) for name, values in samples.items()},
        "comparisons": comparisons,
        "pairwise_comparisons": named_pairwise_comparisons(
            samples,
            pairs=available_pairs,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + len(samples),
        ),
        "failures": failures,
    }


def _route(logits: mx.array, expert_bias: mx.array) -> tuple[mx.array, mx.array]:
    scores = mx.sigmoid(logits.astype(mx.float32))
    selection_scores = scores + expert_bias.astype(mx.float32)
    indices = mx.argpartition(selection_scores, kth=-8, axis=-1)[..., -8:]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return indices, weights * 2.826


def _value_rows(rows: int, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((rows, 4096)).astype(mx.float32)


def _candidate_correctness(
    *,
    stock_logits: mx.array,
    stock_indices: mx.array,
    stock_weights: mx.array,
    candidate_projection: Callable[[], mx.array],
    candidate_router: Callable[[], tuple[mx.array, mx.array]],
    candidate_serial_fused_router: Callable[[], tuple[mx.array, mx.array]],
    candidate_simd_fused_router: Callable[[], tuple[mx.array, mx.array]],
) -> dict[str, Any]:
    """Compile, execute, and validate one candidate independently."""

    candidate_logits = candidate_projection()
    reference_indices, reference_weights = candidate_router()
    serial_indices, serial_weights = candidate_serial_fused_router()
    candidate_indices, candidate_weights = candidate_simd_fused_router()
    repeated_indices, repeated_weights = candidate_simd_fused_router()
    mx.eval(
        candidate_logits,
        reference_indices,
        reference_weights,
        serial_indices,
        serial_weights,
        candidate_indices,
        candidate_weights,
        repeated_indices,
        repeated_weights,
    )
    logit_error = mx.abs(stock_logits - candidate_logits)
    serial_weight_error = mx.abs(stock_weights - serial_weights)
    weight_error = mx.abs(stock_weights - candidate_weights)
    candidate_weight_error = mx.abs(reference_weights - candidate_weights)
    normalized_sum_error = mx.abs(mx.sum(candidate_weights, axis=-1) / 2.826 - 1.0)
    finite_weights = mx.all(mx.isfinite(candidate_weights))
    mx.eval(
        logit_error,
        serial_weight_error,
        weight_error,
        candidate_weight_error,
        normalized_sum_error,
        finite_weights,
    )
    authoritative_contract = authoritative_candidate_contract(
        candidate_topk_exact=bool(
            mx.array_equal(reference_indices, candidate_indices).item()
        ),
        max_candidate_route_weight_abs_error=float(
            mx.max(candidate_weight_error).item()
        ),
        weights_finite=bool(finite_weights.item()),
        max_normalized_sum_abs_error=float(mx.max(normalized_sum_error).item()),
        repeated_ids_exact=bool(
            mx.array_equal(candidate_indices, repeated_indices).item()
        ),
        repeated_weights_exact=bool(
            mx.array_equal(candidate_weights, repeated_weights).item()
        ),
    )
    return {
        **authoritative_contract,
        "max_candidate_route_weight_abs_error": float(
            mx.max(candidate_weight_error).item()
        ),
        "weights_finite": bool(finite_weights.item()),
        "max_normalized_sum_abs_error": float(mx.max(normalized_sum_error).item()),
        "route_ids_exact": bool(
            mx.array_equal(stock_indices, candidate_indices).item()
        ),
        "serial_route_ids_exact": bool(
            mx.array_equal(stock_indices, serial_indices).item()
        ),
        "serial_simd_ids_exact": bool(
            mx.array_equal(serial_indices, candidate_indices).item()
        ),
        "max_logit_abs_error": float(mx.max(logit_error).item()),
        "max_route_weight_abs_error": float(mx.max(weight_error).item()),
        "serial_max_route_weight_abs_error": float(mx.max(serial_weight_error).item()),
    }


def run_benchmark(
    *,
    model: Path,
    layer: int,
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
    frontier: str = "all",
    router_rows: Sequence[int] = ROUTER_ROWS,
) -> dict[str, Any]:
    import mlx.nn as nn

    weight, expert_bias, tensor_metadata = _load_router(model, layer=layer)
    prepared_weight = prepare_hy3_router_fp32_weight(weight)
    mx.eval(prepared_weight, expert_bias)
    gate = nn.Linear(4096, 192, bias=False)
    gate.weight = weight
    selected_rows = tuple(int(rows) for rows in router_rows)
    if (
        not selected_rows
        or len(set(selected_rows)) != len(selected_rows)
        or any(rows not in ROUTER_ROWS for rows in selected_rows)
    ):
        raise ValueError("router rows must be unique values from M1 through M8")
    candidates = router_benchmark_candidates(frontier)
    result_rows = []
    for rows in selected_rows:
        value = _value_rows(rows, seed=515_100 + rows)

        def stock_projection() -> mx.array:
            return gate(value).astype(mx.float32)

        def stock_router() -> tuple[mx.array, mx.array]:
            return _route(stock_projection(), expert_bias)

        stock_logits = stock_projection()
        stock_indices, stock_weights = stock_router()
        mx.eval(stock_logits, stock_indices, stock_weights)
        projection_functions: dict[str, Callable[[], Any]] = {"stock": stock_projection}
        router_functions: dict[str, Callable[[], Any]] = {"stock": stock_router}
        serial_fused_router_functions: dict[str, Callable[[], Any]] = {
            "stock": stock_router
        }
        simd_fused_router_functions: dict[str, Callable[[], Any]] = {
            "stock": stock_router
        }
        r2_topology_functions: dict[str, Callable[[], Any]] = {"stock": stock_router}
        correctness = {}
        candidate_status: dict[str, dict[str, Any]] = {}
        for name, tiling in candidates.items():

            def candidate_projection(tiling=tiling) -> mx.array:
                return hy3_router_fp32_project(
                    value,
                    prepared_weight,
                    n_tile=tiling.n_tile,
                    grid_k_parts=tiling.grid_k_parts,
                    operand_mode=tiling.operand_mode,
                    k_tile=tiling.k_tile,
                    simd_groups_per_threadgroup=(tiling.simd_groups_per_threadgroup),
                )

            def candidate_router(
                candidate_projection=candidate_projection,
            ) -> tuple[mx.array, mx.array]:
                return _route(candidate_projection(), expert_bias)

            def candidate_serial_fused_router(
                tiling=tiling,
            ) -> tuple[mx.array, mx.array]:
                return hy3_router_fp32_route(
                    value,
                    prepared_weight,
                    expert_bias,
                    n_tile=tiling.n_tile,
                    grid_k_parts=tiling.grid_k_parts,
                    operand_mode=tiling.operand_mode,
                    k_tile=tiling.k_tile,
                    simd_groups_per_threadgroup=(tiling.simd_groups_per_threadgroup),
                    top_k=8,
                    route_norm=True,
                    scaling_factor=2.826,
                    finalizer_mode="serial",
                )

            def candidate_simd_fused_router(
                tiling=tiling,
            ) -> tuple[mx.array, mx.array]:
                return hy3_router_fp32_route(
                    value,
                    prepared_weight,
                    expert_bias,
                    n_tile=tiling.n_tile,
                    grid_k_parts=tiling.grid_k_parts,
                    operand_mode=tiling.operand_mode,
                    k_tile=tiling.k_tile,
                    simd_groups_per_threadgroup=(tiling.simd_groups_per_threadgroup),
                    top_k=8,
                    route_norm=True,
                    scaling_factor=2.826,
                    finalizer_mode="simd",
                )

            try:
                correctness[name] = _candidate_correctness(
                    stock_logits=stock_logits,
                    stock_indices=stock_indices,
                    stock_weights=stock_weights,
                    candidate_projection=candidate_projection,
                    candidate_router=candidate_router,
                    candidate_serial_fused_router=candidate_serial_fused_router,
                    candidate_simd_fused_router=candidate_simd_fused_router,
                )
            except Exception as error:
                candidate_status[name] = _candidate_failure_record(error)
                continue
            candidate_status[name] = {"status": "ok"}
            projection_functions[name] = candidate_projection
            router_functions[name] = candidate_router
            serial_fused_router_functions[name] = candidate_serial_fused_router
            simd_fused_router_functions[name] = candidate_simd_fused_router
            if name == "n16_p8_tg96":
                r2_topology_functions["projection_mlx"] = candidate_router
                r2_topology_functions["fused_serial"] = candidate_serial_fused_router
                r2_topology_functions["fused_simd"] = candidate_simd_fused_router
        projection = _measure_arms(
            projection_functions,
            warmups=warmups,
            repeats=repeats,
            bootstrap_resamples=bootstrap_resamples,
            seed=51_000 + rows,
            comparison_pairs=mpp_frontier_comparison_pairs(tuple(candidates)),
        )
        complete_router = _measure_arms(
            router_functions,
            warmups=warmups,
            repeats=repeats,
            bootstrap_resamples=bootstrap_resamples,
            seed=52_000 + rows,
        )
        complete_fused_router_serial = _measure_arms(
            serial_fused_router_functions,
            warmups=warmups,
            repeats=repeats,
            bootstrap_resamples=bootstrap_resamples,
            seed=53_000 + rows,
        )
        complete_fused_router_simd = _measure_arms(
            simd_fused_router_functions,
            warmups=warmups,
            repeats=repeats,
            bootstrap_resamples=bootstrap_resamples,
            seed=54_000 + rows,
            comparison_pairs=mpp_frontier_comparison_pairs(tuple(candidates)),
        )
        r2_topology = _measure_arms(
            r2_topology_functions,
            warmups=warmups,
            repeats=repeats,
            bootstrap_resamples=bootstrap_resamples,
            seed=55_000 + rows,
            comparison_pairs=(
                ("stock", "projection_mlx"),
                ("stock", "fused_serial"),
                ("stock", "fused_simd"),
                ("fused_serial", "fused_simd"),
                ("projection_mlx", "fused_simd"),
            ),
        )
        decisions = {}
        for name in candidates:
            comparison = complete_fused_router_simd["comparisons"].get(name)
            if name not in correctness or comparison is None:
                decisions[name] = False
                continue
            decisions[name] = router_candidate_passes(
                candidate_topk_exact=correctness[name]["candidate_topk_exact"],
                candidate_weights_valid=correctness[name]["candidate_weights_valid"],
                within_mode_deterministic=correctness[name][
                    "within_mode_deterministic"
                ],
                bootstrap_mean_ratio_95_ci=comparison["bootstrap_mean_ratio_95_ci"],
            )
        result_rows.append(
            {
                "mtp_depth": rows - 1,
                "m": rows,
                "candidate_status": candidate_status,
                "correctness": correctness,
                "projection": projection,
                "complete_router_with_stock_finalizer": complete_router,
                "complete_fused_router_serial": complete_fused_router_serial,
                "complete_fused_router_simd": complete_fused_router_simd,
                "r2_topology": r2_topology,
                "operator_gate": decisions,
            }
        )
        print(f"completed isolated router M={rows}", file=sys.stderr, flush=True)
    return {
        "schema": "mtplx-issue51-hy3-router-tiling-v3",
        "scope": "router only; no experts, SSD reads, cache, or full model",
        "control": "same-shape stock MLX",
        "model": str(model),
        "layer": layer,
        "frontier": frontier,
        "router_rows": list(selected_rows),
        "tensor_metadata": tensor_metadata,
        "environment": {"device": mx.metal.device_info()},
        "measurement": {
            "warmups": warmups,
            "paired_interleaved_repeats": repeats,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "candidates": {
            name: {
                "n_tile": tiling.n_tile,
                "grid_k_parts": tiling.grid_k_parts,
                "stage1_threadgroups": tiling.stage1_threadgroups,
                "k_span": tiling.k_span,
                "partial_bytes": tiling.partial_bytes,
                "operand_mode": tiling.operand_mode,
                "simd_groups_per_threadgroup": (tiling.simd_groups_per_threadgroup),
                "total_simdgroups": tiling.total_simdgroups,
                "staged_threadgroup_bytes": tiling.staged_threadgroup_bytes,
                "modeled_activation_bytes": tiling.modeled_activation_bytes,
                "modeled_weight_bytes": tiling.modeled_weight_bytes,
            }
            for name, tiling in candidates.items()
        },
        "rows": result_rows,
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _router_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "rows must be comma-separated M values"
        ) from error
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(row not in ROUTER_ROWS for row in rows)
    ):
        raise argparse.ArgumentTypeError("rows must be unique values from 1 through 8")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--warmups", type=_positive_int, default=8)
    parser.add_argument("--repeats", type=_positive_int, default=80)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=10_000)
    parser.add_argument(
        "--frontier",
        choices=("direct", "grouped", "all"),
        default="all",
    )
    parser.add_argument("--rows", type=_router_rows, default=ROUTER_ROWS)
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument("--lock-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with exclusive_mlx_window(
        plist=args.qwen_plist,
        lock_path=args.lock_path,
        lock_timeout_seconds=args.lock_timeout_seconds,
    ) as receipt:
        print(
            f"acquired exclusive MLX window at {receipt.lock_path}",
            file=sys.stderr,
            flush=True,
        )
        result = run_benchmark(
            model=args.model.expanduser().resolve(),
            layer=args.layer,
            warmups=args.warmups,
            repeats=args.repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            frontier=args.frontier,
            router_rows=args.rows,
        )
        result["exclusive_window"] = {"lock_path": str(receipt.lock_path)}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
