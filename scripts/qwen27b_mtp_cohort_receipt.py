#!/usr/bin/env python3
"""Capture Qwen 3.6 27B depth-2 MTP cohort control/candidate receipts.

The public import surface is deliberately standard-library-only.  MLX and
MTPLX are imported only by the isolated contract-probe subprocess after the
selected worktree has been validated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
import glob
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.request


MODEL_PATH = Path(
    "/Users/davidtai/.mtplx/models/"
    "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
)
MODEL_ID = "qwen3.6-27b-mtplx-optimized-speed"
HOST = "127.0.0.1"
MAX_TOKENS = 256
PROMPTS = (
    (
        "Write a compact implementation notebook about a lock-free ring buffer. "
        "Continue with concrete numbered observations until the token limit; "
        "do not conclude early."
    ),
    (
        "Write a compact implementation notebook about a persistent radix tree. "
        "Continue with concrete numbered observations until the token limit; "
        "do not conclude early."
    ),
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        "request_id",
        "completion_tokens",
        "elapsed_s",
        "ttft_s",
        "decode_tok_s",
    }
)
PROMOTION_REPEATS = 3
SOLO_MIN_RATIO = 0.99
PAIR_MIN_RATIO = 1.35
TTFT_MAX_RATIO = 1.05
LONG_PREFILL_MIN_RATIO = 0.95
REQUIRED_VALIDATION_GATES = (
    "token_parity",
    "acceptance_parity",
    "cache_isolation",
    "session_isolation",
    "streaming",
    "constraint",
    "tool",
    "cancellation",
)
REQUIRED_SCHEDULER_LANES = frozenset(
    {
        "mtp_cohort_width_1",
        "mtp_cohort_width_2",
    }
)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    sid: int
    start: str


@dataclass(frozen=True)
class OwnedProcessGroup:
    leader_pid: int
    pgid: int
    sid: int
    leader_start: str


@dataclass(frozen=True)
class EmergencyProcessGroup:
    leader_pid: int
    pgid: int
    sid: int


@dataclass(frozen=True)
class RunningReceiptServer:
    process: subprocess.Popen[str]
    owned_group: OwnedProcessGroup
    command: list[str]
    health_before: dict[str, Any]
    log_path: Path


def summarize_cell(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate simultaneously submitted request rows."""

    for row in rows:
        missing = REQUIRED_ROW_FIELDS.difference(row)
        if missing:
            raise ValueError(f"receipt row missing {sorted(missing)}")
    if not rows:
        raise ValueError("receipt cell must contain at least one row")
    wall_s = max(float(row["elapsed_s"]) for row in rows)
    if wall_s <= 0:
        raise ValueError("receipt cell elapsed_s must be positive")
    total_tokens = sum(int(row["completion_tokens"]) for row in rows)
    return {
        "aggregate_output_tok_s": total_tokens / wall_s,
        "per_request_decode_tok_s": [
            float(row["decode_tok_s"]) for row in rows
        ],
        "max_ttft_s": max(float(row["ttft_s"]) for row in rows),
    }


def evaluate_promotion(
    control: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Evaluate the fixed Qwen 27B cohort promotion contract."""

    control_metrics = dict(control.get("metrics") or {})
    candidate_metrics = dict(candidate.get("metrics") or {})

    def metric(
        metrics: dict[str, object],
        cell: str,
        field: str,
    ) -> float:
        value = dict(metrics.get(cell) or {}).get(field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"promotion metric {cell}.{field} is missing")
        return float(value)

    control_c1 = metric(control_metrics, "c1", "aggregate_output_tok_s")
    control_c2 = metric(control_metrics, "c2", "aggregate_output_tok_s")
    if control_c1 <= 0 or control_c2 <= 0:
        raise ValueError("control throughput metrics must be positive")
    solo_ratio = (
        metric(candidate_metrics, "c1", "aggregate_output_tok_s") / control_c1
    )
    pair_ratio = (
        metric(candidate_metrics, "c2", "aggregate_output_tok_s") / control_c2
    )

    failures: list[str] = []
    checks: dict[str, bool] = {
        "receipt_status": (
            control.get("status") == "complete"
            and candidate.get("status") == "complete"
        ),
        "solo_throughput": solo_ratio >= SOLO_MIN_RATIO,
        "pair_throughput": pair_ratio >= PAIR_MIN_RATIO,
    }

    repeat_counts: list[int] = []
    for metrics in (control_metrics, candidate_metrics):
        for cell in ("c1", "c2", "c2_4k", "production", "long_prefill"):
            values = dict(metrics.get(cell) or {}).get("repeat_values")
            repeat_counts.append(len(values) if isinstance(values, list) else 0)
    checks["paired_repeats"] = all(
        count == PROMOTION_REPEATS for count in repeat_counts
    )

    for cell, failure in (
        ("c2_4k", "c2_4k_ttft"),
        ("production", "production_ttft"),
    ):
        control_ttft = metric(control_metrics, cell, "max_ttft_s")
        candidate_ttft = metric(candidate_metrics, cell, "max_ttft_s")
        checks[failure] = (
            control_ttft > 0
            and candidate_ttft <= control_ttft * TTFT_MAX_RATIO
        )

    control_prefill = metric(control_metrics, "long_prefill", "prefill_tok_s")
    candidate_prefill = metric(
        candidate_metrics,
        "long_prefill",
        "prefill_tok_s",
    )
    candidate_long = dict(candidate_metrics.get("long_prefill") or {})
    checks["long_prefill_throughput"] = (
        control_prefill > 0
        and candidate_prefill >= control_prefill * LONG_PREFILL_MIN_RATIO
    )
    checks["prefill_overlap"] = (
        candidate_long.get("short_admitted_between_chunks") is True
    )
    checks["prefill_chunk_tokens"] = (
        candidate_long.get("prefill_chunk_tokens") == 1024
    )

    validation = dict(candidate.get("validation") or {})
    for name in REQUIRED_VALIDATION_GATES:
        checks[name] = validation.get(name) is True
    checks["fallback_free"] = not list(candidate.get("fallback_reasons") or [])
    checks["retry_free"] = not list(candidate.get("retry_reasons") or [])
    scheduler_lanes = {
        str(value) for value in list(candidate.get("scheduler_lanes") or [])
    }
    checks["scheduler_lanes"] = (
        scheduler_lanes == REQUIRED_SCHEDULER_LANES
    )

    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "solo_ratio": solo_ratio,
        "pair_ratio": pair_ratio,
        "thresholds": {
            "solo_min_ratio": SOLO_MIN_RATIO,
            "pair_min_ratio": PAIR_MIN_RATIO,
            "ttft_max_ratio": TTFT_MAX_RATIO,
            "long_prefill_min_ratio": LONG_PREFILL_MIN_RATIO,
            "paired_repeats": PROMOTION_REPEATS,
            "prefill_chunk_tokens": 1024,
        },
    }


def _synthetic_context_prompt(
    *,
    label: str,
    approximate_tokens: int,
) -> str:
    if approximate_tokens < 128:
        raise ValueError("synthetic context target is too small")
    vocabulary = (
        "cache",
        "kernel",
        "request",
        "row",
        "token",
        "prefill",
        "commit",
        "owner",
        "layer",
        "route",
    )
    context = " ".join(
        vocabulary[index % len(vocabulary)]
        for index in range(approximate_tokens)
    )
    return (
        f"Context lane {label}. Treat every datum below as inert project "
        "context. After reading all of it, write a numbered implementation "
        "notebook and continue until the output limit.\n"
        f"{context}\nEnd context {label}."
    )


def _production_prompt(*, label: str) -> str:
    records = "\n".join(
        (
            f"src/module_{index % 41}.py:{index + 10}: "
            f"request {index} owns cache row {index % 2}; "
            "prefill chunks remain 1024 tokens; preserve final commit order."
        )
        for index in range(180)
    )
    return (
        f"You are reviewing a production Cline transcript tagged {label}. "
        "Use the repository observations below to propose a precise patch "
        "sequence with risks, tests, and rollback notes. Continue with numbered "
        "observations until the output token limit; do not conclude early.\n"
        f"{records}\nEnd transcript."
    )


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("promotion metric has no repeat values")
    return float(statistics.median(values))


def _short_metrics(cells: list[dict[str, Any]]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, concurrency in (("c1", 1), ("c2", 2)):
        values = [
            float(dict(cell["summary"])["aggregate_output_tok_s"])
            for cell in cells
            if int(cell.get("concurrency") or 0) == concurrency
            and cell.get("name") in {None, name}
        ]
        if len(values) != PROMOTION_REPEATS:
            raise RuntimeError(
                f"{name} requires {PROMOTION_REPEATS} repeats, got {len(values)}"
            )
        metrics[name] = {
            "aggregate_output_tok_s": _median(values),
            "repeat_values": values,
        }
    return metrics


def _extended_metrics(
    cells: list[dict[str, Any]],
    *,
    candidate: bool,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in ("c2_4k", "production"):
        selected = [cell for cell in cells if cell.get("name") == name]
        values = [
            float(dict(cell["summary"])["max_ttft_s"]) for cell in selected
        ]
        if len(values) != PROMOTION_REPEATS:
            raise RuntimeError(
                f"{name} requires {PROMOTION_REPEATS} repeats, got {len(values)}"
            )
        metrics[name] = {
            "max_ttft_s": _median(values),
            "repeat_values": values,
        }

    long_cells = [cell for cell in cells if cell.get("name") == "long_prefill"]
    if len(long_cells) != PROMOTION_REPEATS:
        raise RuntimeError(
            "long_prefill requires "
            f"{PROMOTION_REPEATS} repeats, got {len(long_cells)}"
        )
    prefill_values: list[float] = []
    overlap_values: list[bool] = []
    prompt_tokens: list[int] = []
    for cell in long_cells:
        rows = list(cell["rows"])
        long_row = next(
            row for row in rows if "-long" in str(row.get("request_id"))
        )
        stats = dict(long_row.get("mtplx_stats") or {})
        prefill = stats.get("prefill_tok_s")
        if not isinstance(prefill, (int, float)) or float(prefill) <= 0:
            raise RuntimeError(
                f"long prefill row lacks prefill_tok_s: {long_row.get('request_id')}"
            )
        prefill_values.append(float(prefill))
        prompt_tokens.append(int(stats.get("prompt_tokens") or 0))
        if candidate:
            short_row = next(
                row for row in rows if "-short" in str(row.get("request_id"))
            )
            overlap_values.append(
                float(short_row["first_token_at_cell_s"])
                < float(long_row["first_token_at_cell_s"])
            )
    metrics["long_prefill"] = {
        "prefill_tok_s": _median(prefill_values),
        "repeat_values": prefill_values,
        "prompt_tokens": prompt_tokens,
        "short_admitted_between_chunks": (
            all(overlap_values) if candidate else None
        ),
        "overlap_repeat_values": overlap_values,
        "prefill_chunk_tokens": 1024,
    }
    return metrics


def _load_control_receipt(
    pattern: str,
    *,
    harness_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    paths = sorted(
        (Path(value).resolve() for value in glob.glob(pattern)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = dict(payload.get("harness_provenance") or {})
        if (
            payload.get("status") == "complete"
            and payload.get("mode") == "control"
            and provenance.get("worktree_dirty") is False
            and provenance.get("harness_sha256") == harness_sha256
        ):
            return path, payload
    raise RuntimeError(
        "no complete clean control receipt with matching harness bytes "
        f"matched {pattern!r}"
    )


def _session_reuse_messages(prompt: str) -> list[dict[str, str]]:
    """Reuse the exact committed prompt boundary for the session-cache gate."""
    return [{"role": "user", "content": str(prompt)}]


SESSION_PROBE_PROMPT = (
    "Return a JSON object with the single key status set to ready."
)
SESSION_PROBE_FOLLOWUP_PROMPT = (
    "Return a JSON object with the single key status set to complete."
)


def _session_probe_request_spec(
    *,
    request_id: str,
    extra_headers: dict[str, str],
    assistant_content: str | None = None,
) -> dict[str, Any]:
    """Build the seed or its exact committed-transcript extension."""
    prompt = SESSION_PROBE_PROMPT
    messages = _session_reuse_messages(SESSION_PROBE_PROMPT)
    if assistant_content is not None:
        prompt = SESSION_PROBE_FOLLOWUP_PROMPT
        messages.extend(
            [
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": SESSION_PROBE_FOLLOWUP_PROMPT},
            ]
        )
    return _request_spec(
        request_id=request_id,
        prompt=prompt,
        prompt_index=0,
        max_tokens=64,
        messages=messages,
        body_overrides={
            "response_format": {"type": "json_object"},
        },
        cache_mode=None,
        extra_headers=extra_headers,
        require_max_tokens=False,
    )


def _constraint_row_valid(row: dict[str, Any]) -> bool:
    """Validate completed constrained visible content, excluding reasoning."""
    stats = dict(row.get("mtplx_stats") or {})
    if (
        stats.get("constraint_active") is not True
        or stats.get("constraint_completed") is not True
    ):
        return False
    content = "".join(
        str(item.get("text") or "")
        for item in list(row.get("stream_tokens") or [])
        if item.get("kind") == "content"
    )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict)


def _git_output(worktree: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=worktree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed rc={proc.returncode}: {proc.stderr}"
        )
    return proc.stdout


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _harness_provenance(worktree: Path, harness_path: Path) -> dict[str, object]:
    """Bind a receipt to its exact harness bytes and repository state."""

    resolved_worktree = worktree.resolve()
    resolved_harness = harness_path.resolve()
    if not resolved_harness.is_relative_to(resolved_worktree):
        raise RuntimeError(
            f"harness {resolved_harness} is outside worktree {resolved_worktree}"
        )
    status = _git_output(
        resolved_worktree,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).rstrip("\n")
    working_diff = _git_output(
        resolved_worktree,
        "diff",
        "--binary",
        "HEAD",
    )
    staged_diff = _git_output(
        resolved_worktree,
        "diff",
        "--binary",
        "--cached",
        "HEAD",
    )
    relative_harness = str(resolved_harness.relative_to(resolved_worktree))
    harness_diff = _git_output(
        resolved_worktree,
        "diff",
        "--binary",
        "HEAD",
        "--",
        relative_harness,
    )
    head = _git_output(resolved_worktree, "rev-parse", "HEAD").strip()
    identity = {
        "harness_path": str(resolved_harness),
        "harness_sha256": hashlib.sha256(resolved_harness.read_bytes()).hexdigest(),
        "worktree_head": head,
        "worktree_dirty": bool(status),
        "git_status_porcelain": status,
        "working_tree_diff_sha256": _sha256_text(working_diff),
        "staged_diff_sha256": _sha256_text(staged_diff),
        "harness_diff_sha256": _sha256_text(harness_diff),
    }
    identity["provenance_identity_sha256"] = _sha256_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
    )
    return identity


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _contract_probe(worktree: Path, output: Path) -> None:
    """Load the real model and record construction facts in a short-lived PID."""

    resolved_worktree = worktree.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import mtplx

    mtplx_source = Path(mtplx.__file__).resolve()
    if not mtplx_source.is_relative_to(resolved_worktree):
        raise RuntimeError(
            f"mtplx resolved outside --worktree: {mtplx_source} "
            f"(wanted {resolved_worktree})"
        )

    import importlib.metadata
    import mlx.core as mx
    import mlx.nn as nn

    mlx_version = importlib.metadata.version("mlx")
    if not mlx_version.startswith("0.32."):
        raise RuntimeError(f"expected MLX 0.32.x, got {mlx_version}")
    if MODEL_PATH.resolve() != MODEL_PATH or not MODEL_PATH.is_dir():
        raise RuntimeError(f"exact local model is unavailable: {MODEL_PATH}")

    from mtplx.artifacts import load_config
    from mtplx.backends.descriptors import descriptor_from_runtime
    from mtplx.generation import _prefill, generate_mtpk
    from mtplx.profiles import apply_profile_env
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    apply_profile_env("turbo")
    # The profile drops detailed cycle events in measured serving.  This
    # short-lived, unmeasured construction probe retains the already-existing
    # event long enough to prove the input that was actually sent to verify.
    os.environ["MTPLX_DROP_EVENTS"] = "0"
    runtime = load(MODEL_PATH, mtp=True)
    descriptor = descriptor_from_runtime(runtime)
    if not runtime.mtp_enabled:
        raise RuntimeError("real model loaded without native MTP")

    config = load_config(MODEL_PATH)
    trunk_quant = dict(config.get("quantization") or {})
    expected_quant = {"bits": 4, "group_size": 64, "mode": "affine"}
    observed_quant = {
        "bits": int(trunk_quant.get("bits") or 0),
        "group_size": int(trunk_quant.get("group_size") or 0),
        "mode": str(trunk_quant.get("mode") or ""),
    }
    if observed_quant != expected_quant:
        raise RuntimeError(
            f"target trunk quantization mismatch: {observed_quant}, "
            f"wanted {expected_quant}"
        )

    text_model = getattr(runtime.model, "language_model", runtime.model)
    inner = getattr(text_model, "model", text_model)
    layers = list(getattr(inner, "layers", ()))
    if not layers:
        raise RuntimeError("loaded Qwen target exposes no decoder layers")
    layer_types = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        layer_types.append(
            {
                "layer_index": index,
                "layer_type": _qualified_type(layer),
                "attention_type": (
                    None if attention is None else _qualified_type(attention)
                ),
                "is_linear": bool(getattr(layer, "is_linear", False)),
            }
        )

    qlinears: list[dict[str, object]] = []
    seen_modules: set[int] = set()
    for raw_path, module in text_model.named_modules():
        path = str(raw_path)
        if id(module) in seen_modules or "mtp" in path.split("."):
            continue
        if not isinstance(module, nn.QuantizedLinear):
            continue
        seen_modules.add(id(module))
        bits = int(getattr(module, "bits", 0) or 0)
        group_size = int(getattr(module, "group_size", 0) or 0)
        mode = str(getattr(module, "mode", "affine"))
        k = int(module.weight.shape[1]) * (32 // bits)
        n = int(module.weight.shape[0])
        qlinears.append(
            {
                "path": path,
                "type": _qualified_type(module),
                "bits": bits,
                "group_size": group_size,
                "mode": mode,
                "k": k,
                "n": n,
                "weight_shape": [int(value) for value in module.weight.shape],
                "weight_dtype": str(module.weight.dtype),
            }
        )
    if not qlinears:
        raise RuntimeError("loaded Qwen target exposes no QuantizedLinear modules")
    wrong_qlinears = [
        item
        for item in qlinears
        if (
            item["bits"],
            item["group_size"],
            item["mode"],
        )
        != (4, 64, "affine")
    ]
    if wrong_qlinears:
        raise RuntimeError(
            "target QuantizedLinear contract is not uniformly q4 affine gs64: "
            + json.dumps(wrong_qlinears[:3], sort_keys=True)
        )

    prompt_ids = list(
        runtime.tokenizer.encode(
            "Prove the concrete runtime shape with a short deterministic probe.",
            add_special_tokens=False,
        )
    )
    if not prompt_ids:
        raise RuntimeError("tokenizer returned no offline probe tokens")
    cache, logits, hidden, prefill_time_s = _prefill(
        runtime,
        [int(token) for token in prompt_ids],
        return_hidden=True,
    )

    def describe_leaf(value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return {"type": _qualified_type(value)}
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            return {
                "type": _qualified_type(value),
                "shape": [int(item) for item in shape],
                "dtype": str(dtype),
            }
        if isinstance(value, (list, tuple)):
            return [describe_leaf(item, depth + 1) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": _qualified_type(value)}

    cache_receipt = []
    for index, entry in enumerate(cache):
        item: dict[str, object] = {
            "layer_index": index,
            "type": _qualified_type(entry),
        }
        for attribute in (
            "offset",
            "left_padding",
            "lengths",
            "state",
            "cache",
            "keys",
            "values",
        ):
            if hasattr(entry, attribute):
                item[attribute] = describe_leaf(getattr(entry, attribute))
        cache_receipt.append(item)

    result = generate_mtpk(
        runtime,
        [int(token) for token in prompt_ids],
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        draft_sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=2,
        seed=0,
        stop_token_ids=set(),
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )
    stats = asdict(result.stats) if is_dataclass(result.stats) else dict(result.stats)
    executed_width1_shapes = []
    for event in stats.get("events") or []:
        lazy = event.get("lazy_bonus_verify") or {}
        width = lazy.get("verify_input_tokens")
        if width:
            executed_width1_shapes.append([1, int(width)])
    if [1, 3] not in executed_width1_shapes:
        raise RuntimeError(
            "real depth-2 generation did not execute a [1, 3] target input: "
            f"{executed_width1_shapes}"
        )
    actual_width1 = mx.array([[1, 2, 3]], dtype=mx.int32)
    planned_width2 = mx.concatenate((actual_width1, actual_width1), axis=0)
    if list(planned_width2.shape) != [2, 3]:
        raise RuntimeError(
            f"two-row stacking did not produce planned [2, 3]: "
            f"{list(planned_width2.shape)}"
        )

    geometry_counts: dict[tuple[int, int], int] = {}
    for item in qlinears:
        key = (int(item["k"]), int(item["n"]))
        geometry_counts[key] = geometry_counts.get(key, 0) + 1

    receipt = {
        "probe_pid": os.getpid(),
        "mtplx_source": str(mtplx_source),
        "mlx_version": mlx_version,
        "model_path": str(runtime.model_path.resolve()),
        "backend_id": descriptor.backend_id,
        "architecture_id": descriptor.architecture_id,
        "native_mtp": {
            "enabled": bool(runtime.mtp_enabled),
            "requested_depth": 2,
            "model_depth_max": int(descriptor.draft_semantics.maximum),
            "runtime_contract": runtime.contract.to_dict(),
        },
        "target_quantization": observed_quant,
        "activation_dtype_after_real_prefill": str(hidden.dtype),
        "prefill": {
            "prompt_tokens": len(prompt_ids),
            "elapsed_s": float(prefill_time_s),
            "logits_shape": [int(value) for value in logits.shape],
            "hidden_shape": [int(value) for value in hidden.shape],
            "cache_types": cache_receipt,
        },
        "target_layer_types": layer_types,
        "target_quantized_linears": qlinears,
        "target_qlinear_geometries": [
            {"k": key[0], "n": key[1], "module_count": count}
            for key, count in sorted(geometry_counts.items())
        ],
        "target_shapes": {
            "observed_executed_k2_width1": [1, 3],
            "observed_executed_width1_shapes": executed_width1_shapes,
            "planned_width2_from_two_observed_rows": [
                int(value) for value in planned_width2.shape
            ],
            "planned_width2_construction": "mlx.concatenate two [1,3] rows axis=0",
        },
        "offline_probe_generation": {
            "completion_tokens": len(result.tokens),
            "finish_reason": result.finish_reason,
            "stats": stats,
        },
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _http_json(url: str, *, timeout_s: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_health(
    base_url: str,
    *,
    process: subprocess.Popen[str],
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited before health with rc={return_code}: {last_error}"
            )
        try:
            health = _http_json(base_url + "/health", timeout_s=10.0)
            if health.get("ok"):
                return health
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise TimeoutError(
        f"server did not become healthy in {timeout_s:.1f}s: {last_error}"
    )


def _wait_for_session_prefix(
    *,
    base_url: str,
    session_id: str,
    process: subprocess.Popen[str],
    timeout_s: float = 20.0,
    http_json: Callable[..., dict[str, Any]] = _http_json,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Wait for async postcommit to publish the exact session prefix."""
    started = monotonic()
    deadline = started + timeout_s
    polls = 0
    last_observation = "session prefix was absent"
    while True:
        now = monotonic()
        if now >= deadline:
            break
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "server exited before session postcommit completed "
                f"rc={return_code}"
            )
        polls += 1
        try:
            health = http_json(base_url + "/health", timeout_s=5.0)
            prefixes = list(
                dict(health.get("session_bank") or {}).get("prefixes") or []
            )
            for raw_prefix in prefixes:
                prefix = dict(raw_prefix)
                if (
                    prefix.get("session_id") == session_id
                    and int(prefix.get("prefix_len") or 0) > 0
                ):
                    return {
                        "session_id": session_id,
                        "prefix_len": int(prefix["prefix_len"]),
                        "polls": polls,
                        "elapsed_s": now - started,
                    }
            scheduler = dict(health.get("scheduler") or {})
            last_observation = (
                f"active_lane={scheduler.get('active_lane')!r}, "
                f"prefixes={len(prefixes)}"
            )
        except Exception as exc:
            last_observation = f"{type(exc).__name__}: {exc}"
        sleep(0.05)
    raise TimeoutError(
        f"session {session_id!r} postcommit did not publish within "
        f"{timeout_s:.1f}s: {last_observation}"
    )


def _stream_request_headers(
    *,
    cache_mode: str | None,
    extra_headers: dict[str, str] | None,
) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-MTPLX-Client": "qwen27b-cohort-receipt",
        "X-MTPLX-Allow-Client-Controls": "1",
    }
    if cache_mode is not None:
        headers["X-MTPLX-Cache-Mode"] = cache_mode
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _stream_request(
    *,
    base_url: str,
    request_id: str,
    prompt: str,
    prompt_index: int,
    repeat: int,
    cell_started: float,
    barrier: threading.Barrier | None,
    max_tokens: int = MAX_TOKENS,
    messages: list[dict[str, Any]] | None = None,
    body_overrides: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    cache_mode: str | None = "bypass",
    require_max_tokens: bool = True,
    start_delay_s: float = 0.0,
) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait(timeout=30.0)
    if start_delay_s > 0:
        time.sleep(start_delay_s)
    request_started = time.perf_counter()
    body = {
        "model": MODEL_ID,
        "messages": (
            [{"role": "user", "content": prompt}]
            if messages is None
            else messages
        ),
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 1000 + repeat * 10 + prompt_index,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "metadata": {
            "client": "qwen27b_mtp_cohort_receipt",
            "request_id": request_id,
        },
    }
    if body_overrides:
        body.update(body_overrides)
    headers = _stream_request_headers(
        cache_mode=cache_mode,
        extra_headers=extra_headers,
    )
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        # Independent performance cells pass cache_mode="bypass". Session
        # validation omits that header so construction-time cache ownership is
        # exercised without changing generation arithmetic.
        headers=headers,
    )
    done = False
    first_token_at: float | None = None
    finish_reasons: list[str] = []
    stream_tokens: list[dict[str, object]] = []
    stats: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    response_id: str | None = None
    raw_sse_chunks = 0
    with urllib.request.urlopen(request, timeout=3600) as response:
        if int(response.status) != 200:
            raise RuntimeError(
                f"{request_id} returned HTTP {response.status}"
            )
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                continue
            if not payload:
                continue
            chunk = json.loads(payload)
            raw_sse_chunks += 1
            if isinstance(chunk.get("id"), str):
                response_id = chunk["id"]
            for choice in chunk.get("choices") or []:
                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason:
                    finish_reasons.append(finish_reason)
                delta = choice.get("delta") or {}
                for kind in ("reasoning_content", "content"):
                    text = delta.get(kind)
                    if not isinstance(text, str) or not text:
                        continue
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    stream_tokens.append(
                        {
                            "kind": kind,
                            "text": text,
                            "at_s": now - request_started,
                        }
                    )
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    stream_tokens.append(
                        {
                            "kind": "tool_calls",
                            "text": json.dumps(
                                tool_calls,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "at_s": now - request_started,
                        }
                    )
            if isinstance(chunk.get("usage"), dict):
                usage = dict(chunk["usage"])
            if isinstance(chunk.get("mtplx_stats"), dict):
                stats = dict(chunk["mtplx_stats"])

    finished = time.perf_counter()
    completion_tokens = int(
        stats.get("completion_tokens") or usage.get("completion_tokens") or 0
    )
    ttft_s = None if first_token_at is None else first_token_at - request_started
    decode_tok_s = stats.get("decode_tok_s")
    if (
        not done
        or first_token_at is None
        or not finish_reasons
        or not stats
        or (
            require_max_tokens
            and completion_tokens != int(max_tokens)
        )
        or completion_tokens <= 0
        or not isinstance(decode_tok_s, (int, float))
        or float(decode_tok_s) <= 0
        or (
            cache_mode == "bypass"
            and int(stats.get("cached_tokens") or 0) != 0
        )
        or (
            cache_mode == "bypass"
            and stats.get("request_session_bank_bypass") is not True
        )
    ):
        raise RuntimeError(
            f"incomplete streaming response for {request_id}: "
            f"done={done} ttft_s={ttft_s} finish={finish_reasons} "
            f"stats={bool(stats)} completion_tokens={completion_tokens} "
            f"decode_tok_s={decode_tok_s} "
            f"cached_tokens={stats.get('cached_tokens')} "
            f"session_bank_bypass={stats.get('request_session_bank_bypass')}"
        )
    combined_text = "".join(str(item["text"]) for item in stream_tokens)
    return {
        "request_id": request_id,
        "response_id": response_id,
        "prompt_index": prompt_index,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "completion_tokens": completion_tokens,
        "elapsed_s": finished - cell_started,
        "request_elapsed_s": finished - request_started,
        "request_started_at_cell_s": request_started - cell_started,
        "first_token_at_cell_s": float(first_token_at - cell_started),
        "ttft_s": float(ttft_s),
        "decode_tok_s": float(decode_tok_s),
        "end_to_end_tok_s": (
            completion_tokens / (finished - request_started)
            if finished > request_started
            else 0.0
        ),
        "done": done,
        "finish_reasons": finish_reasons,
        "raw_sse_chunks": raw_sse_chunks,
        "stream_tokens": stream_tokens,
        "combined_text": combined_text,
        "text_sha256": hashlib.sha256(combined_text.encode("utf-8")).hexdigest(),
        "usage": usage,
        "mtplx_stats": stats,
    }


def _memory_receipt(
    pid: int,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    commands = (
        ("vmmap", ["/usr/bin/vmmap", "-summary", str(pid)]),
        ("footprint", ["/usr/bin/footprint", "-p", str(pid)]),
    )
    for tool, command in commands:
        try:
            proc = runner(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60.0,
                check=False,
            )
            attempt: dict[str, object] = {
                "tool": tool,
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except Exception as exc:
            attempt = {
                "tool": tool,
                "command": command,
                "returncode": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        attempts.append(attempt)
        if attempt["returncode"] == 0 and str(attempt.get("stdout") or "").strip():
            return {**attempt, "attempts": attempts}
    raise RuntimeError(
        "memory receipt failed: "
        + json.dumps(attempts, sort_keys=True, default=str)[:8000]
    )


def _candidate_row_failures(
    rows: list[dict[str, object]],
) -> list[str]:
    """Return enabled-path routing, fallback, and retry violations."""

    failures: list[str] = []
    for row in rows:
        request_id = str(row.get("request_id") or "<unknown>")
        stats = dict(row.get("mtplx_stats") or {})
        if stats.get("generation_mode") != "mtp":
            failures.append(
                f"{request_id}: generation_mode={stats.get('generation_mode')}"
            )
        if stats.get("scheduler_lane") != "mtp_cohort":
            failures.append(
                f"{request_id}: scheduler_lane={stats.get('scheduler_lane')}"
            )
        disabled_reason = stats.get("mtp_disabled_reason")
        if disabled_reason not in {None, ""}:
            failures.append(
                f"{request_id}: mtp_disabled_reason={disabled_reason}"
            )
        for name, value in sorted(stats.items()):
            if name.endswith("_retry_attempted") and value is True:
                failures.append(f"{request_id}: {name}=true")
    return failures


def _port_is_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _process_identities() -> list[ProcessIdentity]:
    proc = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,lstart="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ps identity scan failed: {proc.stderr}")
    identities: list[ProcessIdentity] = []
    for line in proc.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        try:
            sid = os.getsid(pid)
        except ProcessLookupError:
            continue
        identities.append(
            ProcessIdentity(
                pid=pid,
                pgid=int(fields[1]),
                sid=sid,
                start=fields[2].strip(),
            )
        )
    return identities


def _read_process_identity(pid: int) -> ProcessIdentity | None:
    return next(
        (identity for identity in _process_identities() if identity.pid == pid),
        None,
    )


def _read_group_members(pgid: int) -> list[ProcessIdentity]:
    return [
        identity for identity in _process_identities() if identity.pgid == pgid
    ]


def _capture_owned_process_group(
    process: subprocess.Popen[str],
    *,
    identity_reader: Callable[[int], ProcessIdentity | None] = (
        _read_process_identity
    ),
) -> OwnedProcessGroup:
    identity = identity_reader(process.pid)
    if identity is None:
        raise RuntimeError(
            f"cannot capture process identity for server leader {process.pid}"
        )
    if identity.pid != identity.pgid or identity.pid != identity.sid:
        raise RuntimeError(
            "dedicated server did not become its own process-group/session "
            f"leader: {identity}"
        )
    return OwnedProcessGroup(
        leader_pid=identity.pid,
        pgid=identity.pgid,
        sid=identity.sid,
        leader_start=identity.start,
    )


def _new_session_emergency_identity(
    process: subprocess.Popen[str],
) -> EmergencyProcessGroup:
    if (
        isinstance(process.pid, bool)
        or not isinstance(process.pid, int)
        or process.pid <= 0
    ):
        raise RuntimeError("spawned server has no valid process identity")
    # Popen reports success only after the child-side start_new_session setsid()
    # has succeeded. The spawned PID is therefore also the exact PGID and SID.
    return EmergencyProcessGroup(
        leader_pid=process.pid,
        pgid=process.pid,
        sid=process.pid,
    )


def _emergency_group_present(
    process: subprocess.Popen[str],
    owner: EmergencyProcessGroup,
    *,
    getpgid: Callable[[int], int],
    getsid: Callable[[int], int],
    group_members: Callable[[int], list[ProcessIdentity]],
) -> bool:
    if process.poll() is None:
        try:
            current_pgid = getpgid(owner.leader_pid)
            current_sid = getsid(owner.leader_pid)
        except ProcessLookupError:
            if process.poll() is None:
                raise RuntimeError(
                    "spawned server leader is reported alive but its emergency "
                    "identity is unavailable"
                )
        else:
            if current_pgid != owner.pgid or current_sid != owner.sid:
                raise RuntimeError(
                    "spawned server emergency identity changed; refusing to "
                    f"signal captured={owner}, current_pgid={current_pgid}, "
                    f"current_sid={current_sid}"
                )
            return True
    members = group_members(owner.pgid)
    wrong_session = [member for member in members if member.sid != owner.sid]
    if wrong_session:
        raise RuntimeError(
            "spawned server emergency process-group identity changed; refusing "
            f"to signal captured_sid={owner.sid}, members={wrong_session}"
        )
    return bool(members)


def _wait_for_emergency_group_exit(
    process: subprocess.Popen[str],
    owner: EmergencyProcessGroup,
    *,
    getpgid: Callable[[int], int],
    getsid: Callable[[int], int],
    group_members: Callable[[int], list[ProcessIdentity]],
    timeout_s: float,
    sleep: Callable[[float], None],
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if not _emergency_group_present(
            process,
            owner,
            getpgid=getpgid,
            getsid=getsid,
            group_members=group_members,
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _stop_emergency_process_group(
    process: subprocess.Popen[str],
    owner: EmergencyProcessGroup,
    *,
    getpgid: Callable[[int], int] | None = None,
    getsid: Callable[[int], int] | None = None,
    group_members: Callable[[int], list[ProcessIdentity]] | None = None,
    killpg: Callable[[int, int], None] | None = None,
    term_timeout_s: float = 30.0,
    kill_timeout_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    getpgid = os.getpgid if getpgid is None else getpgid
    getsid = os.getsid if getsid is None else getsid
    group_members = _read_group_members if group_members is None else group_members
    killpg = os.killpg if killpg is None else killpg
    signals: list[dict[str, object]] = []
    escalated = False
    leader_exited = process.poll() is not None
    present = _emergency_group_present(
        process,
        owner,
        getpgid=getpgid,
        getsid=getsid,
        group_members=group_members,
    )
    if present:
        killpg(owner.pgid, signal.SIGTERM)
        signals.append({"pgid": owner.pgid, "signal": "SIGTERM"})
        exited = _wait_for_emergency_group_exit(
            process,
            owner,
            getpgid=getpgid,
            getsid=getsid,
            group_members=group_members,
            timeout_s=term_timeout_s,
            sleep=sleep,
        )
        if not exited:
            if not _emergency_group_present(
                process,
                owner,
                getpgid=getpgid,
                getsid=getsid,
                group_members=group_members,
            ):
                exited = True
            else:
                killpg(owner.pgid, signal.SIGKILL)
                escalated = True
                signals.append({"pgid": owner.pgid, "signal": "SIGKILL"})
                exited = _wait_for_emergency_group_exit(
                    process,
                    owner,
                    getpgid=getpgid,
                    getsid=getsid,
                    group_members=group_members,
                    timeout_s=kill_timeout_s,
                    sleep=sleep,
                )
        if not exited:
            raise RuntimeError(
                f"spawned server emergency process group {owner.pgid} did not exit"
            )
    try:
        process.wait(timeout=0.0)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass
    return {
        "leader_pid": owner.leader_pid,
        "pgid": owner.pgid,
        "sid": owner.sid,
        "leader_exited_before_shutdown": leader_exited,
        "signals": signals,
        "escalated": escalated,
        "group_exited": True,
        "returncode": process.returncode,
    }


def _validate_owned_group_members(
    process: subprocess.Popen[str],
    owner: OwnedProcessGroup,
    *,
    identity_reader: Callable[[int], ProcessIdentity | None],
    group_members: Callable[[int], list[ProcessIdentity]],
) -> list[ProcessIdentity]:
    members = group_members(owner.pgid)
    scanned_leader = next(
        (member for member in members if member.pid == owner.leader_pid),
        None,
    )
    leader = identity_reader(owner.leader_pid) or scanned_leader
    if leader is not None and (
        leader.pgid != owner.pgid
        or leader.sid != owner.sid
        or leader.start != owner.leader_start
    ):
        raise RuntimeError(
            "owned server leader identity changed; refusing to signal "
            f"captured={owner}, current={leader}"
        )
    wrong_session = [member for member in members if member.sid != owner.sid]
    if wrong_session:
        raise RuntimeError(
            "owned process-group identity changed; refusing to signal "
            f"captured_sid={owner.sid}, members={wrong_session}"
        )
    if not members:
        return []
    if process.poll() is None and leader is None:
        raise RuntimeError(
            "owned server group is present but its leader identity is missing"
        )
    if process.poll() is None and not any(
        member.pid == owner.leader_pid
        and member.start == owner.leader_start
        for member in members
    ):
        raise RuntimeError(
            "owned server leader is alive but absent from its captured group"
        )
    return members


def _wait_for_owned_group_exit(
    process: subprocess.Popen[str],
    owner: OwnedProcessGroup,
    *,
    identity_reader: Callable[[int], ProcessIdentity | None],
    group_members: Callable[[int], list[ProcessIdentity]],
    timeout_s: float,
    sleep: Callable[[float], None],
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        members = _validate_owned_group_members(
            process,
            owner,
            identity_reader=identity_reader,
            group_members=group_members,
        )
        if not members:
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _stop_owned_process_group(
    process: subprocess.Popen[str],
    owner: OwnedProcessGroup,
    *,
    identity_reader: Callable[[int], ProcessIdentity | None] = (
        _read_process_identity
    ),
    group_members: Callable[[int], list[ProcessIdentity]] = _read_group_members,
    killpg: Callable[[int, int], None] = os.killpg,
    term_timeout_s: float = 30.0,
    kill_timeout_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    leader_exited = process.poll() is not None
    initial_members = _validate_owned_group_members(
        process,
        owner,
        identity_reader=identity_reader,
        group_members=group_members,
    )
    signals: list[dict[str, object]] = []
    escalated = False
    if initial_members:
        killpg(owner.pgid, signal.SIGTERM)
        signals.append({"pgid": owner.pgid, "signal": "SIGTERM"})
        exited = _wait_for_owned_group_exit(
            process,
            owner,
            identity_reader=identity_reader,
            group_members=group_members,
            timeout_s=term_timeout_s,
            sleep=sleep,
        )
        if not exited:
            # Revalidation inside the wait is the ownership gate immediately
            # preceding escalation.
            killpg(owner.pgid, signal.SIGKILL)
            escalated = True
            signals.append({"pgid": owner.pgid, "signal": "SIGKILL"})
            exited = _wait_for_owned_group_exit(
                process,
                owner,
                identity_reader=identity_reader,
                group_members=group_members,
                timeout_s=kill_timeout_s,
                sleep=sleep,
            )
        if not exited:
            raise RuntimeError(
                f"owned server process group {owner.pgid} did not exit"
            )
    try:
        process.wait(timeout=0.0)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass
    return {
        "leader_pid": owner.leader_pid,
        "pgid": owner.pgid,
        "sid": owner.sid,
        "leader_start": owner.leader_start,
        "leader_exited_before_shutdown": leader_exited,
        "signals": signals,
        "escalated": escalated,
        "group_exited": True,
        "returncode": process.returncode,
    }


def _atomic_json_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_contract_subprocess(
    *,
    worktree: Path,
    python_executable: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qwen27b-contract-") as temp_dir:
        output = Path(temp_dir) / "contract.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(worktree)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        proc = subprocess.run(
            [
                python_executable,
                str(Path(__file__).resolve()),
                "--worktree",
                str(worktree),
                "--contract-probe-output",
                str(output),
            ],
            cwd=worktree,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800.0,
            check=False,
        )
        if proc.returncode != 0 or not output.is_file():
            raise RuntimeError(
                "offline contract probe failed "
                f"rc={proc.returncode}\nstdout:\n{proc.stdout[-8000:]}\n"
                f"stderr:\n{proc.stderr[-16000:]}"
            )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        receipt["subprocess"] = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-16000:],
        }
        return receipt


def _assert_health_contract(
    health: dict[str, Any],
    port: int,
    *,
    scheduler_mode: str = "serial",
) -> None:
    scheduler = health.get("scheduler") or {}
    config = scheduler.get("config") or {}
    checks = {
        "ok": health.get("ok") is True,
        "model_path": Path(str(health.get("model_path"))).resolve()
        == MODEL_PATH.resolve(),
        "generation_mode": health.get("generation_mode") == "mtp",
        "depth": int(health.get("depth") or 0) == 2,
        "verify_strategy": health.get("verify_strategy") == "capture_commit",
        "verify_core": health.get("verify_core")
        == "linear-gdn-from-conv-tape",
        "profile": (health.get("profile") or {}).get("name") == "turbo",
        "scheduler_mode": scheduler.get("mode") == scheduler_mode,
        "max_active_requests": int(config.get("max_active_requests") or 0) == 2,
        "dedicated_port": port == 18081,
    }
    if scheduler_mode == "mtp_cohort_experimental":
        path_b = scheduler.get("path_b") or {}
        checks.update(
            {
                "decode_batch_max": int(config.get("decode_batch_max") or 0)
                == 2,
                "batch_wait_ms": float(config.get("batch_wait_ms") or 0.0)
                == 0.0,
                "prefill_chunk_tokens": int(
                    config.get("prefill_chunk_tokens") or 0
                )
                == 1024,
                "experimental_mtp_cohorts": (
                    config.get("experimental_mtp_cohorts") is True
                ),
                "scheduler_path_b": scheduler.get("path") == "path_b",
                "cohort_installed": path_b.get("installed") is True,
                "cohort_explicit": (
                    path_b.get("experimental_mtp_cohorts") is True
                ),
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"server health contract failed {failed}: "
            + json.dumps(health, sort_keys=True, default=str)[:8000]
        )


def _capture_final_health(
    base_url: str,
    port: int,
    *,
    http_json: Callable[[str], dict[str, Any]] = _http_json,
    scheduler_mode: str = "serial",
) -> dict[str, Any]:
    health = http_json(base_url + "/health")
    _assert_health_contract(
        health,
        port,
        scheduler_mode=scheduler_mode,
    )
    return health


def _server_command(
    worktree: Path,
    port: int,
    *,
    scheduler_mode: str,
) -> list[str]:
    """Build one fixed server command before the measured phase starts."""

    if not worktree.is_dir():
        raise RuntimeError(f"receipt worktree does not exist: {worktree}")
    command = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "serve",
        "--model",
        str(MODEL_PATH),
        "--model-id",
        MODEL_ID,
        "--host",
        HOST,
        "--port",
        str(port),
        "--generation-mode",
        "mtp",
        "--depth",
        "2",
        "--verify-strategy",
        "capture_commit",
        "--verify-core",
        "linear-gdn-from-conv-tape",
        "--profile",
        "turbo",
        "--scheduler-mode",
        scheduler_mode,
        "--max-active-requests",
        "2",
        "--prefill-chunk-tokens",
        "1024",
        "--warmup-tokens",
        "0",
        "--ssd-session-cache",
        "off",
        "--no-stats-footer",
    ]
    if scheduler_mode == "mtp_cohort_experimental":
        command.extend(
            [
                "--experimental-mtp-cohorts",
                "--decode-batch-max",
                "2",
                "--batch-wait-ms",
                "0",
            ]
        )
    elif scheduler_mode != "serial":
        raise ValueError(f"unsupported receipt scheduler mode {scheduler_mode!r}")
    return command


def _start_receipt_server(
    *,
    worktree: Path,
    port: int,
    scheduler_mode: str,
    log_path: Path,
    timeout_s: float,
) -> RunningReceiptServer:
    if not _port_is_free(HOST, port):
        raise RuntimeError(f"dedicated benchmark port {HOST}:{port} is not free")
    command = _server_command(
        worktree,
        port,
        scheduler_mode=scheduler_mode,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=worktree,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    emergency_group = _new_session_emergency_identity(process)
    try:
        owned_group = _capture_owned_process_group(process)
    except BaseException:
        _stop_emergency_process_group(process, emergency_group)
        raise
    base_url = f"http://{HOST}:{port}"
    try:
        health = _wait_for_health(
            base_url,
            process=process,
            timeout_s=timeout_s,
        )
        _assert_health_contract(
            health,
            port,
            scheduler_mode=scheduler_mode,
        )
    except BaseException:
        _stop_owned_process_group(process, owned_group)
        raise
    return RunningReceiptServer(
        process=process,
        owned_group=owned_group,
        command=command,
        health_before=health,
        log_path=log_path,
    )


def _stop_receipt_server(
    server: RunningReceiptServer,
    *,
    port: int,
) -> dict[str, object]:
    shutdown = _stop_owned_process_group(server.process, server.owned_group)
    deadline = time.monotonic() + 30.0
    while not _port_is_free(HOST, port) and time.monotonic() < deadline:
        time.sleep(0.1)
    if not _port_is_free(HOST, port):
        raise RuntimeError(f"dedicated benchmark port {HOST}:{port} did not clear")
    return shutdown


def _run_http_cell(
    *,
    base_url: str,
    process: subprocess.Popen[str],
    name: str,
    repeat: int,
    requests: list[dict[str, Any]],
    sample_lanes: bool = False,
    validate_candidate: bool = True,
) -> dict[str, object]:
    if not requests:
        raise ValueError("HTTP receipt cell requires at least one request")
    cell_started = time.perf_counter()
    barrier = threading.Barrier(len(requests)) if len(requests) > 1 else None
    lane_samples: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = []
        for spec in requests:
            kwargs = dict(spec)
            kwargs.update(
                {
                    "base_url": base_url,
                    "repeat": repeat,
                    "cell_started": cell_started,
                    "barrier": barrier,
                }
            )
            futures.append(executor.submit(_stream_request, **kwargs))
        while sample_lanes and not all(future.done() for future in futures):
            try:
                health = _http_json(base_url + "/health", timeout_s=5.0)
                scheduler = dict(health.get("scheduler") or {})
                active_lane = scheduler.get("active_lane")
                if active_lane in {
                    "mtp_cohort_width_1",
                    "mtp_cohort_width_2",
                }:
                    lane_samples.append(
                        {
                            "at_s": time.perf_counter() - cell_started,
                            "active_lane": active_lane,
                            "active_requests": scheduler.get("active_requests"),
                        }
                    )
            except Exception as exc:
                lane_samples.append(
                    {
                        "at_s": time.perf_counter() - cell_started,
                        "sample_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            time.sleep(0.02)
        rows = [future.result() for future in futures]
    failures = _candidate_row_failures(rows) if validate_candidate else []
    return {
        "name": name,
        "repeat": repeat + 1,
        "concurrency": len(rows),
        "rows": rows,
        "summary": summarize_cell(rows),
        "lane_samples": lane_samples,
        "row_contract_failures": failures,
        "memory": _memory_receipt(process.pid),
    }


def _finalize_receipt(
    receipt: dict[str, object],
    *,
    output: Path,
    process: subprocess.Popen[str] | None,
    owned_group: OwnedProcessGroup | None,
    host: str,
    port: int,
    stop_group: Callable[
        [subprocess.Popen[str], OwnedProcessGroup], dict[str, object]
    ] = _stop_owned_process_group,
    port_is_free: Callable[[str, int], bool] = _port_is_free,
    port_timeout_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    cleanup_failed = False
    if process is not None:
        server_receipt = dict(receipt.get("server") or {})
        if owned_group is None:
            cleanup_failed = True
            receipt["shutdown_error"] = (
                "server was launched without a captured process-group identity"
            )
        else:
            try:
                shutdown = stop_group(process, owned_group)
                server_receipt["shutdown"] = shutdown
                if shutdown.get("group_exited") is not True:
                    raise RuntimeError(
                        f"owned group did not report exit: {shutdown}"
                    )
            except BaseException as exc:
                cleanup_failed = True
                receipt["shutdown_error"] = f"{type(exc).__name__}: {exc}"
        receipt["server"] = server_receipt

    deadline = time.monotonic() + max(0.0, port_timeout_s)
    while not port_is_free(host, port) and time.monotonic() < deadline:
        sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    port_free = port_is_free(host, port)
    receipt["dedicated_port_free_after"] = port_free
    if not port_free:
        cleanup_failed = True
        receipt["port_cleanup_error"] = (
            f"{host}:{port} remained occupied after owned shutdown"
        )
    if cleanup_failed:
        receipt["status"] = "failed"
    if receipt.get("status") != "complete":
        exit_code = 2
    else:
        exit_code = 0
    _atomic_json_write(output, receipt)
    return exit_code


def _request_spec(
    *,
    request_id: str,
    prompt: str,
    prompt_index: int,
    max_tokens: int = MAX_TOKENS,
    **overrides: Any,
) -> dict[str, Any]:
    spec = {
        "request_id": request_id,
        "prompt": prompt,
        "prompt_index": prompt_index,
        "max_tokens": max_tokens,
    }
    spec.update(overrides)
    return spec


def _run_extended_http_cells(
    *,
    base_url: str,
    process: subprocess.Popen[str],
    candidate: bool,
) -> list[dict[str, object]]:
    prefix = "candidate" if candidate else "baseline"
    cells: list[dict[str, object]] = []
    four_k_prompts = (
        _synthetic_context_prompt(label="four-k-a", approximate_tokens=3800),
        _synthetic_context_prompt(label="four-k-b", approximate_tokens=3900),
    )
    production_prompts = (
        _production_prompt(label="cline-a"),
        _production_prompt(label="cline-b"),
    )
    long_prompt = _synthetic_context_prompt(
        label="eight-k-long",
        approximate_tokens=7800,
    )
    for repeat in range(PROMOTION_REPEATS):
        cells.append(
            _run_http_cell(
                base_url=base_url,
                process=process,
                name="c2_4k",
                repeat=repeat,
                requests=[
                    _request_spec(
                        request_id=f"{prefix}-r{repeat + 1}-4k-slot{index + 1}",
                        prompt=prompt,
                        prompt_index=index,
                    )
                    for index, prompt in enumerate(four_k_prompts)
                ],
                sample_lanes=candidate and repeat == 0,
                validate_candidate=candidate,
            )
        )
        cells.append(
            _run_http_cell(
                base_url=base_url,
                process=process,
                name="production",
                repeat=repeat,
                requests=[
                    _request_spec(
                        request_id=(
                            f"{prefix}-r{repeat + 1}-production-slot{index + 1}"
                        ),
                        prompt=prompt,
                        prompt_index=index,
                    )
                    for index, prompt in enumerate(production_prompts)
                ],
                sample_lanes=candidate and repeat == 0,
                validate_candidate=candidate,
            )
        )
        long_requests = [
            _request_spec(
                request_id=f"{prefix}-r{repeat + 1}-long",
                prompt=long_prompt,
                prompt_index=0,
                max_tokens=64,
            )
        ]
        if candidate:
            long_requests.append(
                _request_spec(
                    request_id=f"{prefix}-r{repeat + 1}-short",
                    prompt=PROMPTS[1],
                    prompt_index=1,
                    max_tokens=64,
                    start_delay_s=0.25,
                )
            )
        cells.append(
            _run_http_cell(
                base_url=base_url,
                process=process,
                name="long_prefill",
                repeat=repeat,
                requests=long_requests,
                sample_lanes=candidate,
                validate_candidate=candidate,
            )
        )
    return cells


def _short_parity(
    control_cells: list[dict[str, Any]],
    candidate_cells: list[dict[str, Any]],
) -> tuple[bool, bool, list[dict[str, object]]]:
    control: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cell in control_cells:
        name = "c1" if int(cell["concurrency"]) == 1 else "c2"
        for row in cell["rows"]:
            control[(name, int(cell["repeat"]), int(row["prompt_index"]))] = row
    comparisons: list[dict[str, object]] = []
    token_parity = True
    acceptance_parity = True
    for cell in candidate_cells:
        for row in cell["rows"]:
            key = (
                str(cell["name"]),
                int(cell["repeat"]),
                int(row["prompt_index"]),
            )
            reference = control[key]
            token_match = row["text_sha256"] == reference["text_sha256"]
            candidate_acceptance = dict(row["mtplx_stats"]).get("accepted_drafts")
            control_acceptance = dict(reference["mtplx_stats"]).get(
                "accepted_drafts"
            )
            acceptance_match = candidate_acceptance == control_acceptance
            token_parity = token_parity and token_match
            acceptance_parity = acceptance_parity and acceptance_match
            comparisons.append(
                {
                    "key": list(key),
                    "token_match": token_match,
                    "acceptance_match": acceptance_match,
                    "control_accepted_drafts": control_acceptance,
                    "candidate_accepted_drafts": candidate_acceptance,
                }
            )
    return token_parity, acceptance_parity, comparisons


def _cancel_after_first_output(
    *,
    base_url: str,
    request_id: str,
    prompt: str,
    barrier: threading.Barrier,
    cell_started: float,
) -> dict[str, object]:
    barrier.wait(timeout=30.0)
    started = time.perf_counter()
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 8801,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "metadata": {
            "client": "qwen27b_mtp_cohort_receipt",
            "request_id": request_id,
        },
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-MTPLX-Client": "qwen27b-cohort-receipt",
            "X-MTPLX-Cache-Mode": "bypass",
        },
    )
    chunks = 0
    first_output_at_cell_s: float | None = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            if any(
                (choice.get("delta") or {}).get(name)
                for choice in chunk.get("choices") or []
                for name in ("content", "reasoning_content", "tool_calls")
            ):
                chunks += 1
                if first_output_at_cell_s is None:
                    first_output_at_cell_s = time.perf_counter() - cell_started
                if chunks >= 2:
                    response.close()
                    break
    return {
        "request_id": request_id,
        "client_closed": True,
        "chunks_before_close": chunks,
        "first_output_at_cell_s": first_output_at_cell_s,
        "request_elapsed_s": time.perf_counter() - started,
    }


def _run_cancellation_cell(
    *,
    base_url: str,
    process: subprocess.Popen[str],
) -> dict[str, object]:
    cell_started = time.perf_counter()
    barrier = threading.Barrier(2)
    lane_samples: list[str] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        cancelled = executor.submit(
            _cancel_after_first_output,
            base_url=base_url,
            request_id="candidate-cancel-shared-row",
            prompt=PROMPTS[0],
            barrier=barrier,
            cell_started=cell_started,
        )
        survivor = executor.submit(
            _stream_request,
            base_url=base_url,
            request_id="candidate-cancel-survivor",
            prompt=PROMPTS[1],
            prompt_index=1,
            repeat=90,
            cell_started=cell_started,
            barrier=barrier,
            max_tokens=256,
        )
        while not survivor.done():
            try:
                scheduler = dict(
                    _http_json(base_url + "/health", timeout_s=5.0).get(
                        "scheduler"
                    )
                    or {}
                )
                lane = scheduler.get("active_lane")
                if lane in {
                    "mtp_cohort_width_1",
                    "mtp_cohort_width_2",
                }:
                    lane_samples.append(str(lane))
            except Exception:
                pass
            time.sleep(0.02)
        cancelled_row = cancelled.result()
        survivor_row = survivor.result()
    deadline = time.monotonic() + 30.0
    final_scheduler: dict[str, Any] = {}
    while time.monotonic() < deadline:
        final_scheduler = dict(
            _http_json(base_url + "/health", timeout_s=5.0).get("scheduler") or {}
        )
        cohort = dict(final_scheduler.get("mtp_cohort") or {})
        if int(cohort.get("active") or 0) == 0 and int(cohort.get("pending") or 0) == 0:
            break
        time.sleep(0.1)
    return {
        "name": "cancellation",
        "repeat": 1,
        "concurrency": 2,
        "cancelled_row": cancelled_row,
        "survivor_row": survivor_row,
        "lane_samples": lane_samples,
        "final_scheduler": final_scheduler,
        "memory": _memory_receipt(process.pid),
    }


def _run_candidate_main(args: argparse.Namespace) -> int:
    worktree = Path(args.worktree).resolve()
    expected_script = worktree / "scripts" / "qwen27b_mtp_cohort_receipt.py"
    if Path(__file__).resolve() != expected_script.resolve():
        raise RuntimeError(
            f"receipt script is not running from --worktree: {Path(__file__).resolve()}"
        )
    if args.server_port != 18081:
        raise ValueError("the candidate receipt must use dedicated port 18081")
    if args.repeats != PROMOTION_REPEATS:
        raise ValueError(
            f"the candidate requires exactly {PROMOTION_REPEATS} repeats"
        )
    if not args.control_glob:
        raise ValueError("--control-glob is required for candidate mode")
    if not _port_is_free(HOST, args.server_port):
        raise RuntimeError(
            f"dedicated benchmark port {HOST}:{args.server_port} is not free"
        )

    output = Path(args.output).expanduser().resolve()
    comparison_output = (
        Path(args.comparison_output).expanduser().resolve()
        if args.comparison_output
        else output.with_name(
            output.name.replace(
                "concurrency2-candidate",
                "concurrency2-comparison",
            )
        )
    )
    if comparison_output == output:
        comparison_output = output.with_suffix(output.suffix + ".comparison.json")
    provenance = _harness_provenance(worktree, Path(__file__))
    receipt: dict[str, object] = {
        "schema": "qwen27b-mtp-cohort-candidate-v1",
        "status": "running",
        "mode": "candidate",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "worktree": str(worktree),
        "worktree_commit": provenance["worktree_head"],
        "harness_provenance": provenance,
        "model_path": str(MODEL_PATH),
        "prefill_chunk_tokens": 1024,
        "comparison_output": str(comparison_output),
        "phases": {},
        "cells": [],
    }
    promotion_passed = False
    try:
        if provenance["worktree_dirty"]:
            raise RuntimeError(
                "candidate receipt requires a clean worktree: "
                f"{provenance['git_status_porcelain']}"
            )
        control_path, frozen_control = _load_control_receipt(
            args.control_glob,
            harness_sha256=str(provenance["harness_sha256"]),
        )
        receipt["control_receipt"] = str(control_path)
        receipt["contract"] = _run_contract_subprocess(
            worktree=worktree,
            python_executable=sys.executable,
        )
        phases = receipt["phases"]
        assert isinstance(phases, dict)

        baseline_log = output.with_suffix(".baseline.server.log")
        baseline_server = _start_receipt_server(
            worktree=worktree,
            port=args.server_port,
            scheduler_mode="serial",
            log_path=baseline_log,
            timeout_s=args.server_timeout_s,
        )
        baseline_phase: dict[str, object] = {
            "scheduler_mode": "serial",
            "command": baseline_server.command,
            "log_path": str(baseline_log),
            "pid": baseline_server.process.pid,
            "owned_process_group": asdict(baseline_server.owned_group),
            "health_before": baseline_server.health_before,
        }
        phases["extended_control"] = baseline_phase
        try:
            baseline_cells = _run_extended_http_cells(
                base_url=f"http://{HOST}:{args.server_port}",
                process=baseline_server.process,
                candidate=False,
            )
            baseline_phase["cells"] = baseline_cells
            baseline_phase["health_after"] = _capture_final_health(
                f"http://{HOST}:{args.server_port}",
                args.server_port,
            )
        finally:
            baseline_phase["shutdown"] = _stop_receipt_server(
                baseline_server,
                port=args.server_port,
            )
        _atomic_json_write(output, receipt)

        candidate_log = output.with_suffix(".candidate.server.log")
        candidate_server = _start_receipt_server(
            worktree=worktree,
            port=args.server_port,
            scheduler_mode="mtp_cohort_experimental",
            log_path=candidate_log,
            timeout_s=args.server_timeout_s,
        )
        candidate_phase: dict[str, object] = {
            "scheduler_mode": "mtp_cohort_experimental",
            "command": candidate_server.command,
            "log_path": str(candidate_log),
            "pid": candidate_server.process.pid,
            "owned_process_group": asdict(candidate_server.owned_group),
            "health_before": candidate_server.health_before,
        }
        phases["candidate"] = candidate_phase
        candidate_cells: list[dict[str, object]] = []
        try:
            base_url = f"http://{HOST}:{args.server_port}"
            short_cells: list[dict[str, object]] = []
            for repeat in range(PROMOTION_REPEATS):
                solo_prompt_index = repeat % 2
                short_cells.append(
                    _run_http_cell(
                        base_url=base_url,
                        process=candidate_server.process,
                        name="c1",
                        repeat=repeat,
                        requests=[
                            _request_spec(
                                request_id=(
                                    f"candidate-r{repeat + 1}-c1"
                                    f"-p{solo_prompt_index + 1}"
                                ),
                                prompt=PROMPTS[solo_prompt_index],
                                prompt_index=solo_prompt_index,
                            )
                        ],
                        sample_lanes=repeat == 0,
                    )
                )
                prompt_order = [0, 1] if repeat % 2 == 0 else [1, 0]
                short_cells.append(
                    _run_http_cell(
                        base_url=base_url,
                        process=candidate_server.process,
                        name="c2",
                        repeat=repeat,
                        requests=[
                            _request_spec(
                                request_id=(
                                    f"candidate-r{repeat + 1}-c2-slot{slot + 1}"
                                    f"-p{prompt_index + 1}"
                                ),
                                prompt=PROMPTS[prompt_index],
                                prompt_index=prompt_index,
                            )
                            for slot, prompt_index in enumerate(prompt_order)
                        ],
                        sample_lanes=repeat == 0,
                    )
                )
                _atomic_json_write(output, receipt)
            candidate_cells.extend(short_cells)
            candidate_cells.extend(
                _run_extended_http_cells(
                    base_url=base_url,
                    process=candidate_server.process,
                    candidate=True,
                )
            )

            candidate_cells.append(
                _run_http_cell(
                    base_url=base_url,
                    process=candidate_server.process,
                    name="uneven_departure",
                    repeat=0,
                    requests=[
                        _request_spec(
                            request_id="candidate-uneven-short",
                            prompt=PROMPTS[0],
                            prompt_index=0,
                            max_tokens=64,
                        ),
                        _request_spec(
                            request_id="candidate-uneven-long",
                            prompt=PROMPTS[1],
                            prompt_index=1,
                        ),
                    ],
                    sample_lanes=True,
                )
            )

            session_id = f"qwen27b-cohort-receipt-{int(time.time())}"
            session_headers = {"X-MTPLX-Session-ID": session_id}
            session_seed = _run_http_cell(
                base_url=base_url,
                process=candidate_server.process,
                name="session_seed",
                repeat=0,
                requests=[
                    _session_probe_request_spec(
                        request_id="candidate-session-seed",
                        extra_headers=session_headers,
                    )
                ],
                sample_lanes=True,
            )
            candidate_cells.append(session_seed)
            session_seed["postcommit_wait"] = _wait_for_session_prefix(
                base_url=base_url,
                session_id=session_id,
                process=candidate_server.process,
            )
            session_seed_row = dict(list(session_seed["rows"])[0])
            session_seed_content = str(
                session_seed_row.get("combined_text") or ""
            )
            if not session_seed_content:
                raise RuntimeError(
                    "session seed completed without committed assistant content"
                )
            session_followup = _run_http_cell(
                base_url=base_url,
                process=candidate_server.process,
                name="session_followup",
                repeat=0,
                requests=[
                    _session_probe_request_spec(
                        request_id="candidate-session-followup",
                        extra_headers=session_headers,
                        assistant_content=session_seed_content,
                    )
                ],
                sample_lanes=True,
            )
            candidate_cells.append(session_followup)

            constrained = _run_http_cell(
                base_url=base_url,
                process=candidate_server.process,
                name="constrained_pair",
                repeat=0,
                requests=[
                    _request_spec(
                        request_id="candidate-constrained",
                        prompt=(
                            "Return a JSON object with keys status and rows. "
                            "Set status to ok and rows to 2."
                        ),
                        prompt_index=0,
                        max_tokens=256,
                        body_overrides={
                            "response_format": {"type": "json_object"},
                        },
                        require_max_tokens=False,
                    ),
                    _request_spec(
                        request_id="candidate-constrained-peer",
                        prompt=PROMPTS[1],
                        prompt_index=1,
                        max_tokens=64,
                    ),
                ],
                sample_lanes=True,
            )
            candidate_cells.append(constrained)

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "session_status",
                        "description": "Show the current agent session status.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            tool_pair = _run_http_cell(
                base_url=base_url,
                process=candidate_server.process,
                name="tool_pair",
                repeat=0,
                requests=[
                    _request_spec(
                        request_id="candidate-tool",
                        prompt=(
                            "Call session_status exactly once with an empty "
                            "argument object."
                        ),
                        prompt_index=0,
                        max_tokens=128,
                        body_overrides={
                            "tools": tools,
                            "tool_choice": "auto",
                        },
                        require_max_tokens=False,
                    ),
                    _request_spec(
                        request_id="candidate-tool-peer",
                        prompt=PROMPTS[1],
                        prompt_index=1,
                        max_tokens=128,
                    ),
                ],
                sample_lanes=True,
            )
            candidate_cells.append(tool_pair)

            cancellation = _run_cancellation_cell(
                base_url=base_url,
                process=candidate_server.process,
            )
            candidate_phase["cancellation"] = cancellation
            candidate_phase["cells"] = candidate_cells
            candidate_phase["health_after"] = _capture_final_health(
                base_url,
                args.server_port,
                scheduler_mode="mtp_cohort_experimental",
            )
        finally:
            candidate_phase["shutdown"] = _stop_receipt_server(
                candidate_server,
                port=args.server_port,
            )

        receipt["cells"] = candidate_cells
        control_metrics = {
            **_short_metrics(list(frozen_control["cells"])),
            **_extended_metrics(baseline_cells, candidate=False),
        }
        candidate_metrics = {
            **_short_metrics(short_cells),
            **_extended_metrics(candidate_cells, candidate=True),
        }
        token_parity, acceptance_parity, parity_rows = _short_parity(
            list(frozen_control["cells"]),
            short_cells,
        )
        all_rows = [
            row
            for cell in candidate_cells
            for row in list(cell.get("rows") or [])
        ]
        row_failures = [
            failure
            for cell in candidate_cells
            for failure in list(cell.get("row_contract_failures") or [])
        ]
        lanes = {
            str(sample["active_lane"])
            for cell in candidate_cells
            for sample in list(cell.get("lane_samples") or [])
            if isinstance(sample, dict) and sample.get("active_lane")
        }
        lanes.update(
            str(value)
            for value in list(
                dict(candidate_phase["cancellation"]).get("lane_samples") or []
            )
        )
        fallback_reasons = [
            value for value in row_failures if "_retry_attempted" not in value
        ]
        retry_reasons = [
            value for value in row_failures if "_retry_attempted" in value
        ]
        session_followup_row = list(session_followup["rows"])[0]
        session_stats = dict(session_followup_row["mtplx_stats"])
        constrained_row = next(
            row
            for row in constrained["rows"]
            if row["request_id"] == "candidate-constrained"
        )
        constraint_ok = _constraint_row_valid(constrained_row)
        tool_row = next(
            row for row in tool_pair["rows"] if row["request_id"] == "candidate-tool"
        )
        tool_ok = any(
            item.get("kind") == "tool_calls"
            for item in list(tool_row.get("stream_tokens") or [])
        ) or int(dict(tool_row["mtplx_stats"]).get("tool_calls_emitted") or 0) > 0
        response_ids = [
            str(row.get("response_id"))
            for row in all_rows
            if row.get("response_id")
        ]
        streaming_ok = all(
            row.get("done") is True
            and int(row.get("raw_sse_chunks") or 0) > 0
            for row in all_rows
        )
        cancellation_row = dict(
            dict(candidate_phase["cancellation"]).get("cancelled_row") or {}
        )
        cancellation_survivor = dict(
            dict(candidate_phase["cancellation"]).get("survivor_row") or {}
        )
        validation = {
            "token_parity": token_parity,
            "acceptance_parity": acceptance_parity,
            "cache_isolation": (
                len(response_ids) == len(set(response_ids))
                and not row_failures
            ),
            "session_isolation": (
                session_stats.get("session_cache_hit") is True
                and int(session_stats.get("cached_tokens") or 0) > 0
            ),
            "streaming": streaming_ok,
            "constraint": constraint_ok,
            "tool": tool_ok,
            "cancellation": (
                cancellation_row.get("client_closed") is True
                and cancellation_survivor.get("done") is True
            ),
        }
        control_summary: dict[str, object] = {
            "status": "complete",
            "receipt_path": str(control_path),
            "metrics": control_metrics,
        }
        candidate_summary: dict[str, object] = {
            "status": "complete",
            "receipt_path": str(output),
            "metrics": candidate_metrics,
            "validation": validation,
            "scheduler_lanes": sorted(lanes),
            "fallback_reasons": fallback_reasons,
            "retry_reasons": retry_reasons,
        }
        promotion = evaluate_promotion(control_summary, candidate_summary)
        comparison = {
            "schema": "qwen27b-mtp-cohort-comparison-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "control": control_summary,
            "candidate": candidate_summary,
            "short_parity_rows": parity_rows,
            "promotion": promotion,
        }
        _atomic_json_write(comparison_output, comparison)
        receipt["comparison"] = comparison
        receipt["status"] = "complete"
        receipt["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        promotion_passed = promotion["passed"] is True
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    exit_code = _finalize_receipt(
        receipt,
        output=output,
        process=None,
        owned_group=None,
        host=HOST,
        port=args.server_port,
    )
    if exit_code == 0 and not promotion_passed:
        return 3
    return exit_code


def _run_control_main(args: argparse.Namespace) -> int:
    worktree = Path(args.worktree).resolve()
    expected_script = worktree / "scripts" / "qwen27b_mtp_cohort_receipt.py"
    if Path(__file__).resolve() != expected_script.resolve():
        raise RuntimeError(
            f"receipt script is not running from --worktree: {Path(__file__).resolve()}"
        )
    if args.server_port != 18081:
        raise ValueError("the control receipt must use dedicated port 18081")
    if args.mode != "control":
        raise ValueError("this unchanged harness currently supports only --mode control")
    if args.repeats != 3:
        raise ValueError("the frozen control requires exactly three repeats")
    if not _port_is_free(HOST, args.server_port):
        raise RuntimeError(
            f"dedicated benchmark port {HOST}:{args.server_port} is not free"
        )

    output = Path(args.output).expanduser().resolve()
    server_log = output.with_suffix(output.suffix + ".server.log")
    provenance = _harness_provenance(worktree, Path(__file__))
    receipt: dict[str, object] = {
        "schema": "qwen27b-mtp-cohort-control-v2",
        "status": "running",
        "mode": args.mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "worktree": str(worktree),
        "worktree_commit": provenance["worktree_head"],
        "harness_provenance": provenance,
        "model_path": str(MODEL_PATH),
        "server": {
            "host": HOST,
            "port": args.server_port,
            "log_path": str(server_log),
        },
        "cells": [],
    }
    process: subprocess.Popen[str] | None = None
    owned_group: OwnedProcessGroup | None = None
    try:
        if provenance["worktree_dirty"]:
            raise RuntimeError(
                "control receipt requires a clean worktree: "
                f"{provenance['git_status_porcelain']}"
            )
        receipt["contract"] = _run_contract_subprocess(
            worktree=worktree,
            python_executable=sys.executable,
        )
        command = _server_command(
            worktree,
            args.server_port,
            scheduler_mode="serial",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(worktree)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        server_log.parent.mkdir(parents=True, exist_ok=True)
        with server_log.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=worktree,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        emergency_group = _new_session_emergency_identity(process)
        server_receipt = dict(receipt["server"])
        server_receipt.update(
            {
                "command": command,
                "pid": process.pid,
                "emergency_process_group": asdict(emergency_group),
            }
        )
        receipt["server"] = server_receipt
        try:
            owned_group = _capture_owned_process_group(process)
        except BaseException:
            server_receipt["emergency_shutdown"] = (
                _stop_emergency_process_group(process, emergency_group)
            )
            process = None
            raise
        server_receipt["owned_process_group"] = asdict(owned_group)

        base_url = f"http://{HOST}:{args.server_port}"
        health = _wait_for_health(
            base_url,
            process=process,
            timeout_s=args.server_timeout_s,
        )
        _assert_health_contract(health, args.server_port)
        server_receipt["health_before"] = health

        cells = receipt["cells"]
        assert isinstance(cells, list)
        for repeat in range(args.repeats):
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited before repeat {repeat} rc={process.returncode}"
                )
            solo_prompt_index = repeat % 2
            cell_started = time.perf_counter()
            solo_row = _stream_request(
                base_url=base_url,
                request_id=f"control-r{repeat + 1}-c1-p{solo_prompt_index + 1}",
                prompt=PROMPTS[solo_prompt_index],
                prompt_index=solo_prompt_index,
                repeat=repeat,
                cell_started=cell_started,
                barrier=None,
            )
            solo_cell = {
                "repeat": repeat + 1,
                "concurrency": 1,
                "prompt_order": [solo_prompt_index],
                "rows": [solo_row],
                "summary": summarize_cell([solo_row]),
                "memory": _memory_receipt(process.pid),
            }
            cells.append(solo_cell)

            prompt_order = [0, 1] if repeat % 2 == 0 else [1, 0]
            cell_started = time.perf_counter()
            barrier = threading.Barrier(2)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        _stream_request,
                        base_url=base_url,
                        request_id=(
                            f"control-r{repeat + 1}-c2-slot{slot + 1}"
                            f"-p{prompt_index + 1}"
                        ),
                        prompt=PROMPTS[prompt_index],
                        prompt_index=prompt_index,
                        repeat=repeat,
                        cell_started=cell_started,
                        barrier=barrier,
                    )
                    for slot, prompt_index in enumerate(prompt_order)
                ]
                pair_rows = [future.result() for future in futures]
            pair_cell = {
                "repeat": repeat + 1,
                "concurrency": 2,
                "prompt_order": prompt_order,
                "rows": pair_rows,
                "summary": summarize_cell(pair_rows),
                "memory": _memory_receipt(process.pid),
            }
            cells.append(pair_cell)
            _atomic_json_write(output, receipt)

        if process.poll() is not None:
            raise RuntimeError(
                f"server exited after benchmark rc={process.returncode}"
            )
        server_receipt["health_after"] = _capture_final_health(
            base_url,
            args.server_port,
        )
        server_receipt["final_health_contract_validated"] = True
        receipt["status"] = "complete"
        receipt["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    return _finalize_receipt(
        receipt,
        output=output,
        process=process,
        owned_group=owned_group,
        host=HOST,
        port=args.server_port,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--server-port", type=int, default=18081)
    parser.add_argument(
        "--mode",
        choices=("control", "candidate"),
        default="control",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--control-glob")
    parser.add_argument("--comparison-output")
    parser.add_argument("--server-timeout-s", type=float, default=1200.0)
    parser.add_argument("--contract-probe-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.contract_probe_output and not args.output:
        parser.error("--output is required")
    return args


def _run_main(args: argparse.Namespace) -> int:
    if args.mode == "candidate":
        return _run_candidate_main(args)
    return _run_control_main(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.contract_probe_output:
        _contract_probe(
            Path(args.worktree),
            Path(args.contract_probe_output),
        )
        return 0
    return _run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
