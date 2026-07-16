#!/usr/bin/env python3
"""Paired complete-router timing for the Issue #58 M1-M8 candidate family."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mtplx.hy3_router_fp32 import (  # noqa: E402
    hy3_router_fp32_available,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_weight,
)
from mtplx.hy3_router_last_arrival import (  # noqa: E402
    hy3_router_last_arrival_route,
)
from mtplx.hy3_router_row_owned import (  # noqa: E402
    hy3_router_row_owned_route,
    prepare_hy3_router_row_owned_weight,
)

try:  # Stacked #58 API; fail loudly at benchmark construction if absent.
    from mtplx.hy3_router_last_arrival import (  # noqa: E402
        new_hy3_router_forward_epoch,
    )
except ImportError:  # pragma: no cover - exercised through the explicit API gate
    new_hy3_router_forward_epoch = None
from mtplx.qwen_guard import (  # noqa: E402
    DEFAULT_MLX_LOCK_PATH,
    MlxWindowReceipt,
    _exclusive_mlx_lock,
    qwen_stopped_for_mlx,
)


SCHEMA = "mtplx-issue58-router-paired-timing-v4"
CONTROL_ARM = "issue59-r41-topology-n16-p16-sg4-grouped-direct-precise-g6"
CANDIDATE_ARM = "issue58-m1-m8-last-arrival-one-dispatch-precise"
ROW_OWNED_ARM = "issue58-m1-m8-row-owned-tiled-one-dispatch-g6"
CANDIDATE_ARMS = ("row-owned", "last-arrival")
MIN_ROWS = 1
MAX_ROWS = 8
DEFAULT_ROWS = 4
HIDDEN_SIZE = 4096
EXPERTS = 192
TOP_K = 8
ROUTER_SCALING = 2.826
ROUTE_WEIGHT_SUM_ATOL = 1e-5
PERCENTILE_METHOD = "linear"
MIN_ABBA_BLOCKS = 30
MIN_PAIRED_REPEATS = MIN_ABBA_BLOCKS * 2
MIN_BOOTSTRAP_RESAMPLES = 10_000
NOT_CLAIMED_GATES = (
    "million-election stress gate",
    "adversarial tie and boundary parity gate",
    "full K0-K7 1024/1024 model benchmark matrix",
)
REQUIRED_PROVENANCE_SOURCES = (
    "benchmarks/hy3_router_last_arrival_timing.py",
    "mtplx/hy3_router_fp32.py",
    "mtplx/hy3_router_last_arrival.py",
    "mtplx/hy3_router_row_owned.py",
    "mtplx/nax_verify.py",
    "mtplx/qwen_guard.py",
)
_CORRECTNESS_REQUIREMENTS = (
    "ids_exact",
    "route_weights_bitwise_exact",
    "control_route_weights_finite",
    "candidate_route_weights_finite",
    "control_route_weights_normalized",
    "candidate_route_weights_normalized",
    "control_route_weights_in_range",
    "candidate_route_weights_in_range",
    "control_repeated_deterministic",
    "candidate_repeated_deterministic",
)


def paired_arm_order(repeat: int) -> tuple[str, str]:
    """Alternate AB then BA so each adjacent pair is position balanced."""

    if int(repeat) < 0:
        raise ValueError("paired repeat index must be non-negative")
    if int(repeat) % 2:
        return ("candidate", "control")
    return ("control", "candidate")


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples or any(not np.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("router timing samples must be finite, positive, and nonempty")
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.quantile(samples, 0.95, method=PERCENTILE_METHOD)),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def paired_timing_statistics(
    raw_pairs: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize paired complete-route samples and a paired bootstrap ratio."""

    if int(bootstrap_resamples) < MIN_BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"router timing requires at least {MIN_BOOTSTRAP_RESAMPLES} "
            "bootstrap resamples"
        )
    if not raw_pairs:
        raise ValueError("paired router timing requires at least one pair")
    if len(raw_pairs) % 2:
        raise ValueError("paired router timing requires complete even ABBA blocks")
    if len(raw_pairs) < MIN_PAIRED_REPEATS:
        raise ValueError(
            f"router timing requires at least {MIN_ABBA_BLOCKS} complete ABBA blocks"
        )
    pairs = [dict(pair) for pair in raw_pairs]
    for index, pair in enumerate(pairs):
        if int(pair.get("repeat", -1)) != index:
            raise ValueError(
                "paired router samples must have contiguous repeat indices"
            )
        if tuple(pair.get("order", ())) != paired_arm_order(index):
            raise ValueError("paired router samples must preserve ABBA arm order")
    control = np.asarray([pair["control_ms"] for pair in pairs], dtype=np.float64)
    candidate = np.asarray([pair["candidate_ms"] for pair in pairs], dtype=np.float64)
    if (
        np.any(~np.isfinite(control))
        or np.any(~np.isfinite(candidate))
        or np.any(control <= 0.0)
        or np.any(candidate <= 0.0)
    ):
        raise ValueError("paired router samples must be finite and positive")
    paired_ratios = control / candidate
    control_blocks = control.reshape(-1, 2)
    candidate_blocks = candidate.reshape(-1, 2)
    rng = np.random.default_rng(int(bootstrap_seed))
    block_indices = rng.integers(
        0,
        control_blocks.shape[0],
        size=(int(bootstrap_resamples), control_blocks.shape[0]),
    )
    bootstrap_ratios = control_blocks[block_indices].mean(axis=(1, 2)) / (
        candidate_blocks[block_indices].mean(axis=(1, 2))
    )
    return {
        "raw_pairs": pairs,
        "control": _sample_summary(control.tolist()),
        "candidate": _sample_summary(candidate.tolist()),
        "speed_ratio_definition": "control_ms / candidate_ms; above 1 is faster",
        "speed_ratio_control_over_candidate": float(control.mean() / candidate.mean()),
        "paired_speed_ratio_mean": float(paired_ratios.mean()),
        "paired_speed_ratio_median": float(np.median(paired_ratios)),
        "paired_speed_ratio_p95": float(
            np.quantile(
                paired_ratios,
                0.95,
                method=PERCENTILE_METHOD,
            )
        ),
        "bootstrap_speed_ratio_95_ci": [
            float(
                np.quantile(
                    bootstrap_ratios,
                    0.025,
                    method=PERCENTILE_METHOD,
                )
            ),
            float(
                np.quantile(
                    bootstrap_ratios,
                    0.975,
                    method=PERCENTILE_METHOD,
                )
            ),
        ],
        "percentile_method": "numpy.quantile(method='linear')",
        "abba_blocks": int(control_blocks.shape[0]),
        "bootstrap_unit": "complete ABBA block (2 paired repeats)",
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def candidate_advances_to_full_matrix(
    *,
    correctness: Mapping[str, Any],
    bootstrap_speed_ratio_95_ci: Sequence[float],
) -> bool:
    """Require enough evidence to justify running the full validation matrix."""

    if len(bootstrap_speed_ratio_95_ci) != 2:
        raise ValueError("router speed interval must contain two bounds")
    return (
        all(bool(correctness.get(name, False)) for name in _CORRECTNESS_REQUIREMENTS)
        and float(bootstrap_speed_ratio_95_ci[0]) > 1.0
    )


def _valid_hex_digest(value: Any, *, length: int) -> bool:
    text = str(value)
    return len(text) == int(length) and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def provenance_gate(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a clean, identified build and device were captured."""

    failures: list[str] = []
    commit = provenance.get("commit")
    if not isinstance(commit, Mapping):
        failures.append("missing_commit_provenance")
    else:
        if not _valid_hex_digest(commit.get("head"), length=40):
            failures.append("invalid_commit_head")
        if commit.get("dirty") is not False:
            failures.append("dirty_worktree")
        if commit.get("capture_error") not in (None, ""):
            failures.append("commit_capture_error")

    sources = provenance.get("sources")
    if not isinstance(sources, Mapping):
        failures.append("missing_source_provenance")
    else:
        for name in REQUIRED_PROVENANCE_SOURCES:
            metadata = sources.get(name)
            if not isinstance(metadata, Mapping) or not _valid_hex_digest(
                metadata.get("sha256"),
                length=64,
            ):
                failures.append(f"invalid_source_sha256:{name}")

    libraries = provenance.get("libraries")
    for name in ("mlx", "numpy"):
        version = libraries.get(name) if isinstance(libraries, Mapping) else None
        if not isinstance(version, str) or version in {"", "unavailable"}:
            failures.append(f"missing_{name}_version")

    device = provenance.get("device")
    if not isinstance(device, Mapping) or device.get("qualified") is not True:
        failures.append("unqualified_device")

    command = provenance.get("command")
    if not isinstance(command, Mapping) or not command.get("argv"):
        failures.append("missing_command_provenance")
    host = provenance.get("host")
    if not isinstance(host, Mapping) or not host.get("python_executable"):
        failures.append("missing_host_provenance")
    return {"passed": not failures, "failures": failures}


def timing_gate(timing: Mapping[str, Any]) -> dict[str, Any]:
    """Reject advance records that do not preserve complete ABBA blocks."""

    failures: list[str] = []
    raw_pairs = timing.get("raw_pairs")
    if (
        not isinstance(raw_pairs, Sequence)
        or isinstance(raw_pairs, (str, bytes))
        or not raw_pairs
    ):
        failures.append("missing_raw_pairs")
        pairs: list[Any] = []
    else:
        pairs = list(raw_pairs)
        if len(pairs) % 2:
            failures.append("unbalanced_abba_pairs")
        for index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                failures.append(f"invalid_pair:{index}")
                continue
            order = pair.get("order")
            if (
                pair.get("repeat") != index
                or not isinstance(order, Sequence)
                or isinstance(order, (str, bytes))
                or tuple(order) != paired_arm_order(index)
            ):
                failures.append(f"invalid_pair_order:{index}")
    if timing.get("abba_blocks") != len(pairs) // 2:
        failures.append("invalid_abba_block_count")
    if len(pairs) // 2 < MIN_ABBA_BLOCKS:
        failures.append("insufficient_abba_blocks")
    bootstrap_resamples = timing.get("bootstrap_resamples")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < MIN_BOOTSTRAP_RESAMPLES
    ):
        failures.append("insufficient_bootstrap_resamples")
    if timing.get("bootstrap_unit") != "complete ABBA block (2 paired repeats)":
        failures.append("invalid_bootstrap_unit")
    if timing.get("percentile_method") != "numpy.quantile(method='linear')":
        failures.append("invalid_percentile_method")
    return {"passed": not failures, "failures": failures}


def _candidate_arm_label(config: Mapping[str, Any]) -> str:
    """Read the configured candidate arm; legacy payloads keep the old label."""

    candidate = config.get("candidate") if isinstance(config, Mapping) else None
    if isinstance(candidate, Mapping):
        arm = candidate.get("arm")
        if isinstance(arm, str) and arm:
            return arm
    return CANDIDATE_ARM


def completed_result(
    *,
    correctness: dict[str, Any],
    timing: dict[str, Any],
    provenance: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    interval = timing["bootstrap_speed_ratio_95_ci"]
    evidence_passed = candidate_advances_to_full_matrix(
        correctness=correctness,
        bootstrap_speed_ratio_95_ci=interval,
    )
    provenance_status = provenance_gate(provenance)
    timing_status = timing_gate(timing)
    passed = (
        evidence_passed
        and bool(provenance_status["passed"])
        and bool(timing_status["passed"])
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "complete Hy3 router only; no experts, SSD reads, or full model",
        "control_arm": CONTROL_ARM,
        "candidate_arm": _candidate_arm_label(config),
        "config": config,
        "correctness": correctness,
        "timing": timing,
        "advance_to_full_matrix": {
            "passed": passed,
            "decision": "advance" if passed else "hold",
            "scope": "authorizes the subsequent full validation matrix only",
            "requirements": {
                "ids_exact": True,
                "route_weights_bitwise_exact": True,
                "both_arms_repeated_deterministic": True,
                "bootstrap_speed_ratio_95_ci_lower_gt": 1.0,
                "qualified_clean_provenance": True,
                "balanced_abba_timing_record": True,
            },
            "not_claimed": list(NOT_CLAIMED_GATES),
        },
        "timing_gate": timing_status,
        "provenance_gate": provenance_status,
        "provenance": provenance,
    }


def rejected_result(
    *,
    correctness: dict[str, Any],
    provenance: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Record a correctness rejection without contaminating it with timing."""

    reasons = [
        name for name in _CORRECTNESS_REQUIREMENTS if not correctness.get(name, False)
    ]
    if not reasons:
        raise ValueError("correctness rejection requires a failed gate")
    return {
        "schema": SCHEMA,
        "status": "rejected",
        "scope": "complete Hy3 router only; no experts, SSD reads, or full model",
        "control_arm": CONTROL_ARM,
        "candidate_arm": _candidate_arm_label(config),
        "config": config,
        "correctness": correctness,
        "rejection": {"phase": "correctness", "reasons": reasons},
        "timing": {
            "status": "not_run",
            "reason": "correctness_or_determinism_gate_failed",
        },
        "advance_to_full_matrix": {
            "passed": False,
            "decision": "hold",
            "scope": "authorizes the subsequent full validation matrix only",
            "not_claimed": list(NOT_CLAIMED_GATES),
        },
        "provenance_gate": provenance_gate(provenance),
        "provenance": provenance,
    }


def failure_result(
    *,
    phase: str,
    error: BaseException,
    provenance: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        "scope": "complete Hy3 router only; no experts, SSD reads, or full model",
        "control_arm": CONTROL_ARM,
        "candidate_arm": _candidate_arm_label(config),
        "config": config,
        "failure": {
            "phase": str(phase),
            "error_type": type(error).__name__,
            "error": str(error),
        },
        "advance_to_full_matrix": {
            "passed": False,
            "decision": "not_evaluated",
            "scope": "authorizes the subsequent full validation matrix only",
            "not_claimed": list(NOT_CLAIMED_GATES),
        },
        "provenance": provenance,
    }


def finalize_result_provenance(
    result: dict[str, Any],
    final_capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the final decision to provenance captured after a checkpoint exists."""

    prior = result.get("provenance")
    final_provenance = dict(final_capture)
    if isinstance(prior, Mapping):
        for name in ("device", "model_tensors", "activation"):
            if name in prior:
                final_provenance[name] = prior[name]
    result["provenance"] = final_provenance
    status = result.get("status")
    if status == "complete":
        provenance_status = provenance_gate(final_provenance)
        timing_status = timing_gate(result["timing"])
        evidence_passed = candidate_advances_to_full_matrix(
            correctness=result["correctness"],
            bootstrap_speed_ratio_95_ci=result["timing"]["bootstrap_speed_ratio_95_ci"],
        )
        passed = (
            evidence_passed
            and bool(provenance_status["passed"])
            and bool(timing_status["passed"])
        )
        result["provenance_gate"] = provenance_status
        result["timing_gate"] = timing_status
        result["advance_to_full_matrix"]["passed"] = passed
        result["advance_to_full_matrix"]["decision"] = "advance" if passed else "hold"
    elif status == "rejected":
        result["provenance_gate"] = provenance_gate(final_provenance)
    return result


def write_json_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one readable benchmark checkpoint."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def collect_provenance(argv: Sequence[str]) -> dict[str, Any]:
    """Capture code identity without touching the Metal device."""

    source_paths = tuple(_REPO_ROOT / name for name in REQUIRED_PROVENANCE_SOURCES)
    capture_error: str | None = None
    try:
        head = _git_output("rev-parse", "HEAD")
        status_lines = _git_output("status", "--porcelain", "--untracked-files=all")
        dirty: bool | None = bool(status_lines)
    except (OSError, subprocess.CalledProcessError) as error:
        head = "unavailable"
        status_lines = ""
        dirty = None
        capture_error = f"{type(error).__name__}: {error}"
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = "unavailable"
    command = [str(item) for item in argv]
    return {
        "commit": {
            "head": head,
            "dirty": dirty,
            "status_lines": status_lines.splitlines(),
            "capture_error": capture_error,
        },
        "sources": {
            str(path.relative_to(_REPO_ROOT)): {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        },
        "command": {
            "argv": command,
            "shell": shlex.join(command),
            "cwd": str(Path.cwd()),
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "libraries": {
            "mlx": mlx_version,
            "numpy": np.__version__,
        },
    }


def _macos_version_tuple(version: str) -> tuple[int, int, int]:
    values: list[int] = []
    for part in str(version).split(".")[:3]:
        try:
            values.append(int(part))
        except ValueError:
            values.append(0)
    while len(values) < 3:
        values.append(0)
    return values[0], values[1], values[2]


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), default=str))


def _require_qualified_device() -> dict[str, Any]:
    """Verify the real G17/macOS capability after exclusivity is acquired."""

    system = platform.system()
    macos_version = platform.mac_ver()[0]
    mlx_device = _jsonable_mapping(mx.device_info())
    metal_device = _jsonable_mapping(mx.metal.device_info())
    architecture = str(mlx_device.get("architecture", "")).lower()
    if system != "Darwin":
        raise RuntimeError("Issue #58 router timing requires Darwin")
    if _macos_version_tuple(macos_version) < (26, 2, 0):
        raise RuntimeError("Issue #58 router timing requires macOS 26.2 or newer")
    if not architecture.startswith("applegpu_g17"):
        raise RuntimeError("Issue #58 router timing requires a real G17 device")
    if not hy3_router_fp32_available():
        raise RuntimeError("Issue #58 router G17 capability probe failed")
    return {
        "qualified": True,
        "system": system,
        "macos_version": macos_version,
        "architecture": architecture,
        "mlx": mlx_device,
        "metal": metal_device,
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
        "shard": str(path),
        "shape": list(shape),
        "dtype": dtype,
        "bytes": len(raw),
        "tensor_payload_sha256": _sha256_bytes(raw),
    }


def _load_router(
    model: Path,
    *,
    layer: int,
) -> tuple[mx.array, mx.array, dict[str, Any]]:
    index_path = model / "model.safetensors.index.json"
    index_payload = index_path.read_bytes()
    index = json.loads(index_payload)
    weight_map = index["weight_map"]
    weight_key = f"model.layers.{layer}.mlp.router.gate.weight"
    bias_key = f"model.layers.{layer}.mlp.router.expert_bias"
    source_weight, weight_metadata = _load_safetensor(
        model / weight_map[weight_key],
        weight_key,
    )
    expert_bias, bias_metadata = _load_safetensor(
        model / weight_map[bias_key],
        bias_key,
    )
    if (
        tuple(int(dimension) for dimension in source_weight.shape)
        != (EXPERTS, HIDDEN_SIZE)
        or source_weight.dtype != mx.bfloat16
    ):
        raise ValueError("Issue #58 timing requires source BF16 [192, 4096] weight")
    if (
        tuple(int(dimension) for dimension in expert_bias.shape) != (EXPERTS,)
        or expert_bias.dtype != mx.float32
    ):
        raise ValueError("Issue #58 timing requires FP32 [192] expert bias")
    resident_weight = prepare_hy3_router_fp32_weight(source_weight)
    mx.eval(resident_weight, expert_bias)
    mx.synchronize()
    if (
        tuple(int(dimension) for dimension in resident_weight.shape)
        != (HIDDEN_SIZE, EXPERTS)
        or resident_weight.dtype != mx.bfloat16
    ):
        raise ValueError("prepared router weight is not K-major BF16 [4096, 192]")
    return (
        resident_weight,
        expert_bias,
        {
            "index": {
                "path": str(index_path),
                "sha256": _sha256_bytes(index_payload),
            },
            "source_weight": weight_metadata,
            "expert_bias": bias_metadata,
            "resident_weight": {
                "shape": [HIDDEN_SIZE, EXPERTS],
                "dtype": "BF16",
                "layout": "K-major",
                "shared_by_both_arms": True,
            },
        },
    )


def _validate_rows(rows: int) -> int:
    logical_rows = int(rows)
    if isinstance(rows, bool) or logical_rows != rows:
        raise ValueError("router rows must be an integer")
    if not MIN_ROWS <= logical_rows <= MAX_ROWS:
        raise ValueError(
            f"router rows must be between {MIN_ROWS} and {MAX_ROWS} inclusive"
        )
    return logical_rows


def _activation(seed: int, *, rows: int) -> tuple[mx.array, dict[str, Any]]:
    logical_rows = _validate_rows(rows)
    rng = np.random.default_rng(int(seed))
    host = rng.standard_normal((1, logical_rows, HIDDEN_SIZE), dtype=np.float32)
    value = mx.array(np.ascontiguousarray(host), dtype=mx.float32)
    mx.eval(value)
    mx.synchronize()
    return value, {
        "seed": int(seed),
        "generator": "numpy.random.default_rng.standard_normal",
        "shape": [1, logical_rows, HIDDEN_SIZE],
        "logical_rows": logical_rows,
        "mpp_descriptor_rows": 8,
        "logical_extent_rows": logical_rows,
        "dtype": "FP32",
        "payload_sha256": _sha256_bytes(host.tobytes(order="C")),
        "shared_by_both_arms": True,
    }


def _execute_route(
    function: Callable[[], tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array]:
    expert_ids, route_weights = function()
    mx.eval(expert_ids, route_weights)
    mx.synchronize()
    return expert_ids, route_weights


def _array_payload(value: mx.array) -> tuple[np.ndarray, str]:
    host = np.ascontiguousarray(np.asarray(value))
    return host, _sha256_bytes(host.tobytes(order="C"))


def _weights_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise ValueError("route-weight bitwise parity requires FP32 outputs")
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def _route_weight_health(weights: np.ndarray) -> dict[str, Any]:
    """Validate the mathematical route-weight contract without NaN loopholes."""

    values = np.asarray(weights)
    if values.dtype != np.float32 or values.ndim < 1 or values.shape[-1] != TOP_K:
        raise ValueError("route weights must be FP32 with a final top-8 dimension")
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        return {
            "finite": False,
            "normalized": False,
            "in_range": False,
            "max_sum_abs_error": None,
            "minimum": None,
            "maximum": None,
        }
    sums = np.sum(values, axis=-1, dtype=np.float64)
    max_sum_abs_error = float(np.max(np.abs(sums - ROUTER_SCALING)))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return {
        "finite": True,
        "normalized": max_sum_abs_error <= ROUTE_WEIGHT_SUM_ATOL,
        "in_range": minimum >= 0.0 and maximum <= ROUTER_SCALING,
        "max_sum_abs_error": max_sum_abs_error,
        "minimum": minimum,
        "maximum": maximum,
    }


def _correctness(
    control: Callable[[], tuple[mx.array, mx.array]],
    candidate: Callable[[], tuple[mx.array, mx.array]],
) -> dict[str, Any]:
    control_first = _execute_route(control)
    candidate_first = _execute_route(candidate)
    candidate_repeat = _execute_route(candidate)
    control_repeat = _execute_route(control)
    control_ids, control_ids_sha = _array_payload(control_first[0])
    control_weights, control_weights_sha = _array_payload(control_first[1])
    candidate_ids, candidate_ids_sha = _array_payload(candidate_first[0])
    candidate_weights, candidate_weights_sha = _array_payload(candidate_first[1])
    control_repeat_ids, _ = _array_payload(control_repeat[0])
    control_repeat_weights, _ = _array_payload(control_repeat[1])
    candidate_repeat_ids, _ = _array_payload(candidate_repeat[0])
    candidate_repeat_weights, _ = _array_payload(candidate_repeat[1])
    control_weight_health = _route_weight_health(control_weights)
    candidate_weight_health = _route_weight_health(candidate_weights)
    control_ids_repeat_exact = bool(np.array_equal(control_ids, control_repeat_ids))
    control_weights_repeat_exact = _weights_bitwise_equal(
        control_weights,
        control_repeat_weights,
    )
    candidate_ids_repeat_exact = bool(
        np.array_equal(candidate_ids, candidate_repeat_ids)
    )
    candidate_weights_repeat_exact = _weights_bitwise_equal(
        candidate_weights,
        candidate_repeat_weights,
    )
    return {
        "ids_exact": bool(np.array_equal(control_ids, candidate_ids)),
        "route_weights_bitwise_exact": _weights_bitwise_equal(
            control_weights,
            candidate_weights,
        ),
        "control_route_weights_finite": control_weight_health["finite"],
        "candidate_route_weights_finite": candidate_weight_health["finite"],
        "control_route_weights_normalized": control_weight_health["normalized"],
        "candidate_route_weights_normalized": candidate_weight_health["normalized"],
        "control_route_weights_in_range": control_weight_health["in_range"],
        "candidate_route_weights_in_range": candidate_weight_health["in_range"],
        "control_repeated_ids_exact": control_ids_repeat_exact,
        "control_repeated_weights_bitwise_exact": control_weights_repeat_exact,
        "candidate_repeated_ids_exact": candidate_ids_repeat_exact,
        "candidate_repeated_weights_bitwise_exact": (candidate_weights_repeat_exact),
        "control_repeated_deterministic": (
            control_ids_repeat_exact and control_weights_repeat_exact
        ),
        "candidate_repeated_deterministic": (
            candidate_ids_repeat_exact and candidate_weights_repeat_exact
        ),
        "max_route_weight_abs_error": (
            float(np.max(np.abs(control_weights - candidate_weights)))
            if control_weight_health["finite"] and candidate_weight_health["finite"]
            else None
        ),
        "route_weight_health": {
            "control": control_weight_health,
            "candidate": candidate_weight_health,
            "sum_target": ROUTER_SCALING,
            "sum_abs_tolerance": ROUTE_WEIGHT_SUM_ATOL,
            "required_range": [0.0, ROUTER_SCALING],
        },
        "outputs": {
            "control": {
                "ids_sha256": control_ids_sha,
                "route_weights_sha256": control_weights_sha,
            },
            "candidate": {
                "ids_sha256": candidate_ids_sha,
                "route_weights_sha256": candidate_weights_sha,
            },
        },
    }


def _router_arms(
    value: mx.array,
    resident_weight: mx.array,
    expert_bias: mx.array,
    *,
    candidate_invocations: int,
    candidate_arm: str = "row-owned",
) -> dict[str, Callable[[], tuple[mx.array, mx.array]]]:
    """Build the exact #59 r41 G6 control and the selected #58 candidate."""

    if candidate_arm not in CANDIDATE_ARMS:
        raise ValueError(f"candidate arm must be one of {CANDIDATE_ARMS}")
    invocation_count = int(candidate_invocations)
    if invocation_count <= 0:
        raise ValueError("candidate invocation count must be positive")

    def control() -> tuple[mx.array, mx.array]:
        return hy3_router_fp32_route(
            value,
            resident_weight,
            expert_bias,
            n_tile=16,
            grid_k_parts=16,
            operand_mode="grouped-direct",
            simd_groups_per_threadgroup=4,
            top_k=TOP_K,
            route_norm=True,
            scaling_factor=ROUTER_SCALING,
            finalizer_mode="simd",
            simd_groups=6,
            sigmoid_mode="precise",
        )

    if candidate_arm == "row-owned":
        tiled_weight = prepare_hy3_router_row_owned_weight(
            mx.contiguous(resident_weight.T)
        )
        mx.eval(tiled_weight)
        mx.synchronize()

        def candidate() -> tuple[mx.array, mx.array]:
            return hy3_router_row_owned_route(
                value,
                tiled_weight,
                expert_bias,
                top_k=TOP_K,
                route_norm=True,
                scaling_factor=ROUTER_SCALING,
                sigmoid_mode="precise",
            )

        return {"control": control, "candidate": candidate}

    if not callable(new_hy3_router_forward_epoch):
        raise RuntimeError("Issue #58 timing requires the explicit forward-epoch API")
    epoch_block = new_hy3_router_forward_epoch(invocation_count)
    mx.eval(epoch_block)
    mx.synchronize()
    epoch_cursor = 0

    def candidate() -> tuple[mx.array, mx.array]:
        nonlocal epoch_cursor
        if epoch_cursor >= invocation_count:
            raise RuntimeError("Issue #58 timing exhausted candidate epoch views")
        epoch = epoch_block[epoch_cursor]
        epoch_cursor += 1
        output = hy3_router_last_arrival_route(
            value,
            resident_weight,
            expert_bias,
            epoch=epoch,
            top_k=TOP_K,
            route_norm=True,
            scaling_factor=ROUTER_SCALING,
            sigmoid_mode="precise",
        )
        return output.expert_ids, output.route_weights

    return {"control": control, "candidate": candidate}


def _measure_pairs(
    functions: Mapping[str, Callable[[], tuple[mx.array, mx.array]]],
    *,
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    if set(functions) != {"control", "candidate"}:
        raise ValueError("paired timing requires exactly control and candidate arms")
    if min(int(warmups), int(repeats)) <= 0 or any(
        count % 2 for count in (int(warmups), int(repeats))
    ):
        raise ValueError(
            "paired timing warmups and repeats must be positive even ABBA blocks"
        )
    for warmup in range(int(warmups)):
        for name in paired_arm_order(warmup):
            _execute_route(functions[name])
    raw_pairs: list[dict[str, Any]] = []
    for repeat in range(int(repeats)):
        order = paired_arm_order(repeat)
        sample: dict[str, Any] = {
            "repeat": repeat,
            "order": list(order),
        }
        for name in order:
            started = time.perf_counter_ns()
            _execute_route(functions[name])
            sample[f"{name}_ms"] = (time.perf_counter_ns() - started) / 1_000_000
        raw_pairs.append(sample)
    return raw_pairs


def _measure_queued_blocks(
    functions: Mapping[str, Callable[[], tuple[mx.array, mx.array]]],
    *,
    repeats: int,
    block: int,
) -> list[dict[str, Any]]:
    """Time N back-to-back dispatches per arm with one eval/synchronize.

    Per-call host synchronization dominates the eager microsecond lane, so a
    candidate must also hold its ratio when kernels are queued back to back.
    """

    if set(functions) != {"control", "candidate"}:
        raise ValueError("queued timing requires exactly control and candidate arms")
    if min(int(repeats), int(block)) <= 0 or int(repeats) % 2:
        raise ValueError("queued timing requires a positive block and even repeats")
    for warmup in range(2):
        for name in paired_arm_order(warmup):
            outputs: list[mx.array] = []
            for _ in range(int(block)):
                outputs.extend(functions[name]())
            mx.eval(*outputs)
            mx.synchronize()
    samples: list[dict[str, Any]] = []
    for repeat in range(int(repeats)):
        order = paired_arm_order(repeat)
        sample: dict[str, Any] = {
            "repeat": repeat,
            "order": list(order),
            "block_invocations": int(block),
        }
        for name in order:
            started = time.perf_counter_ns()
            outputs = []
            for _ in range(int(block)):
                outputs.extend(functions[name]())
            mx.eval(*outputs)
            mx.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            sample[f"{name}_ms"] = elapsed_ms / int(block)
        samples.append(sample)
    return samples


def run_benchmark(
    *,
    model: Path,
    layer: int,
    rows: int,
    warmups: int,
    repeats: int,
    activation_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    provenance: dict[str, Any],
    config: dict[str, Any],
    device: dict[str, Any],
    candidate_arm: str = "row-owned",
    queued_block: int = 0,
    queued_repeats: int = 60,
) -> dict[str, Any]:
    """Execute the paired gate; the caller owns the exclusive MLX window."""

    logical_rows = _validate_rows(rows)
    if (
        min(int(warmups), int(repeats), int(bootstrap_resamples)) <= 0
        or int(warmups) % 2
        or int(repeats) % 2
        or int(repeats) < MIN_PAIRED_REPEATS
        or int(bootstrap_resamples) < MIN_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            f"router warmups must be positive/even, timing requires at least "
            f"{MIN_ABBA_BLOCKS} complete ABBA blocks, and bootstrap requires at "
            f"least {MIN_BOOTSTRAP_RESAMPLES} resamples"
        )
    if device.get("qualified") is not True:
        raise ValueError("router benchmark requires a qualified device receipt")
    resident_weight, expert_bias, tensor_metadata = _load_router(
        model,
        layer=int(layer),
    )
    value, activation_metadata = _activation(
        int(activation_seed),
        rows=logical_rows,
    )

    queued_invocations = (
        (2 + int(queued_repeats)) * int(queued_block) if int(queued_block) else 0
    )
    functions = _router_arms(
        value,
        resident_weight,
        expert_bias,
        candidate_invocations=2 + int(warmups) + int(repeats) + queued_invocations,
        candidate_arm=candidate_arm,
    )
    correctness = _correctness(functions["control"], functions["candidate"])
    complete_provenance = {
        **provenance,
        "device": device,
        "model_tensors": tensor_metadata,
        "activation": activation_metadata,
    }
    if any(not correctness.get(name, False) for name in _CORRECTNESS_REQUIREMENTS):
        return rejected_result(
            correctness=correctness,
            provenance=complete_provenance,
            config=config,
        )
    raw_pairs = _measure_pairs(
        functions,
        warmups=int(warmups),
        repeats=int(repeats),
    )
    timing = paired_timing_statistics(
        raw_pairs,
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_seed=int(bootstrap_seed),
    )
    if int(queued_block):
        queued_samples = _measure_queued_blocks(
            functions,
            repeats=int(queued_repeats),
            block=int(queued_block),
        )
        timing["queued"] = paired_timing_statistics(
            queued_samples,
            bootstrap_resamples=int(bootstrap_resamples),
            bootstrap_seed=int(bootstrap_seed),
        )
        timing["queued"]["block_invocations"] = int(queued_block)
    return completed_result(
        correctness=correctness,
        timing=timing,
        provenance=complete_provenance,
        config=config,
    )


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_even_int(value: str) -> int:
    result = _positive_int(value)
    if result % 2:
        raise argparse.ArgumentTypeError("must be positive and even for ABBA balance")
    return result


def _sufficient_paired_repeats(value: str) -> int:
    result = _positive_even_int(value)
    if result < MIN_PAIRED_REPEATS:
        raise argparse.ArgumentTypeError(
            f"must provide at least {MIN_PAIRED_REPEATS} paired repeats "
            f"({MIN_ABBA_BLOCKS} complete ABBA blocks)"
        )
    return result


def _sufficient_bootstrap_resamples(value: str) -> int:
    result = _positive_int(value)
    if result < MIN_BOOTSTRAP_RESAMPLES:
        raise argparse.ArgumentTypeError(
            f"must provide at least {MIN_BOOTSTRAP_RESAMPLES} bootstrap resamples"
        )
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _logical_rows(value: str) -> int:
    try:
        return _validate_rows(int(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--rows", type=_logical_rows, default=DEFAULT_ROWS)
    parser.add_argument("--warmups", type=_positive_even_int, default=12)
    parser.add_argument(
        "--candidate-arm",
        choices=list(CANDIDATE_ARMS),
        default="row-owned",
    )
    parser.add_argument("--queued-block", type=int, default=64)
    parser.add_argument("--queued-repeats", type=_positive_even_int, default=60)
    parser.add_argument("--repeats", type=_sufficient_paired_repeats, default=100)
    parser.add_argument("--activation-seed", type=int, default=58_590_004)
    parser.add_argument("--bootstrap-seed", type=int, default=58_051)
    parser.add_argument(
        "--bootstrap-resamples",
        type=_sufficient_bootstrap_resamples,
        default=10_000,
    )
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument(
        "--lock-timeout-seconds",
        type=_positive_float,
        default=21_600.0,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace) -> dict[str, Any]:
    rows = int(args.rows)
    return {
        "model": str(args.model.expanduser().resolve()),
        "layer": int(args.layer),
        "shape": {
            "hidden": [1, rows, HIDDEN_SIZE],
            "logical_rows": rows,
            "mpp_descriptor_rows": 8,
            "logical_extent_rows": int(args.rows),
            "resident_weight": [HIDDEN_SIZE, EXPERTS],
            "expert_bias": [EXPERTS],
            "top_k": TOP_K,
        },
        "arithmetic": {
            "input": "FP32",
            "weight": "BF16",
            "expert_bias": "FP32",
            "route_weights": "FP32",
            "sigmoid": "precise",
            "router_scaling_factor": ROUTER_SCALING,
        },
        "control": {
            "arm": CONTROL_ARM,
            "implementation": "hy3_router_fp32_route",
            "issue59_candidate": "n16_p16_sg4_grouped_direct_precise_g6",
            "n_tile": 16,
            "grid_k_parts": 16,
            "operand_mode": "grouped-direct",
            "simd_groups_per_threadgroup": 4,
            "finalizer_mode": "simd",
            "finalizer_simd_groups": 6,
        },
        "candidate": (
            {
                "arm": ROW_OWNED_ARM,
                "implementation": "hy3_router_row_owned_route",
                "dispatches": 1,
                "logical_m": rows,
                "supported_logical_m_range": [MIN_ROWS, MAX_ROWS],
                "threadgroups_per_dispatch": rows,
                "threads_per_threadgroup": 384,
            }
            if args.candidate_arm == "row-owned"
            else {
                "arm": CANDIDATE_ARM,
                "implementation": "hy3_router_last_arrival_route",
                "dispatches": 1,
                "logical_m": rows,
                "supported_logical_m_range": [MIN_ROWS, MAX_ROWS],
            }
        ),
        "measurement": {
            "order": "balanced ABBA paired interleave in complete even blocks",
            "warmups_per_arm": int(args.warmups),
            "paired_repeats": int(args.repeats),
            "activation_seed": int(args.activation_seed),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "percentile_method": "numpy.quantile(method='linear')",
            "synchronization": "mx.eval(ids, weights) then mx.synchronize for both arms",
            "queued_block_invocations": int(args.queued_block),
            "queued_paired_repeats": int(args.queued_repeats),
        },
    }


def _initial_exclusive_window_audit(lock_path: Path) -> dict[str, Any]:
    return {
        "lock_path": str(lock_path.expanduser().resolve()),
        "lock_holder_pid": None,
        "lock_acquired": False,
        "qwen_state_captured": False,
        "qwen_loaded_before": None,
        "qwen_models_before": [],
        "qwen_restore_verified": None,
        "lock_release_verified": None,
    }


@contextmanager
def _audited_exclusive_mlx_window(
    *,
    plist: Path,
    lock_path: Path,
    lock_timeout_seconds: float,
    audit: dict[str, Any],
):
    """Expose acquisition/restoration boundaries hidden by the composed guard."""

    qwen_error: BaseException | None = None
    try:
        with _exclusive_mlx_lock(
            lock_path=Path(lock_path),
            timeout_seconds=float(lock_timeout_seconds),
        ) as acquired_path:
            audit.update(
                {
                    "lock_path": str(acquired_path),
                    "lock_holder_pid": os.getpid(),
                    "lock_acquired": True,
                    "qwen_restore_verified": False,
                    "lock_release_verified": False,
                }
            )
            body_error: BaseException | None = None
            try:
                with qwen_stopped_for_mlx(plist=Path(plist)) as state:
                    audit.update(
                        {
                            "qwen_state_captured": True,
                            "qwen_loaded_before": state.loaded,
                            "qwen_models_before": list(state.models),
                        }
                    )
                    try:
                        yield MlxWindowReceipt(
                            lock_path=acquired_path,
                            qwen_state=state,
                        )
                    except BaseException as error:
                        body_error = error
                        raise
            except BaseException as error:
                qwen_error = error
                audit["qwen_restore_verified"] = (
                    body_error is not None and error is body_error
                )
                raise
            else:
                audit["qwen_restore_verified"] = True
    except BaseException as error:
        if audit["lock_acquired"]:
            audit["lock_release_verified"] = (
                qwen_error is not None and error is qwen_error
            )
        raise
    else:
        audit["lock_release_verified"] = True


def _failure_exit_code(error: BaseException) -> int:
    return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(parsed_argv)
    command_argv = [sys.executable, str(Path(__file__).resolve()), *parsed_argv]
    provenance = collect_provenance(command_argv)
    config = _config(args)
    output = args.output_json.expanduser().resolve()
    started_at = datetime.now(UTC).isoformat()
    phase = "acquire-exclusive-window"
    exclusive_window = _initial_exclusive_window_audit(args.lock_path)
    checkpoint: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "phase": phase,
        "started_at": started_at,
        "config": config,
        "provenance": provenance,
        "exclusive_window": exclusive_window,
    }
    write_json_checkpoint(output, checkpoint)
    result: dict[str, Any] | None = None
    body_error: BaseException | None = None
    body_error_phase: str | None = None
    window_error: BaseException | None = None
    qualified_device: dict[str, Any] | None = None
    try:
        with _audited_exclusive_mlx_window(
            plist=args.qwen_plist.expanduser().resolve(),
            lock_path=args.lock_path.expanduser().resolve(),
            lock_timeout_seconds=float(args.lock_timeout_seconds),
            audit=exclusive_window,
        ):
            try:
                phase = "qualify-g17-device"
                checkpoint = {
                    **checkpoint,
                    "phase": phase,
                    "exclusive_window": exclusive_window,
                }
                write_json_checkpoint(output, checkpoint)
                qualified_device = _require_qualified_device()
                phase = "correctness-then-paired-complete-router-timing"
                result = run_benchmark(
                    model=args.model.expanduser().resolve(),
                    layer=int(args.layer),
                    rows=int(args.rows),
                    warmups=int(args.warmups),
                    repeats=int(args.repeats),
                    activation_seed=int(args.activation_seed),
                    bootstrap_seed=int(args.bootstrap_seed),
                    bootstrap_resamples=int(args.bootstrap_resamples),
                    provenance=provenance,
                    config=config,
                    device=qualified_device,
                    candidate_arm=str(args.candidate_arm),
                    queued_block=int(args.queued_block),
                    queued_repeats=int(args.queued_repeats),
                )
            except BaseException as error:
                body_error = error
                body_error_phase = phase
            phase = "restore-qwen-and-release-window"
    except BaseException as error:
        window_error = error

    error = window_error if window_error is not None else body_error
    if error is not None:
        failure_provenance = dict(provenance)
        if qualified_device is not None:
            failure_provenance["device"] = qualified_device
        failed = failure_result(
            phase=(
                phase
                if window_error is not None
                else body_error_phase or "benchmark-body"
            ),
            error=error,
            provenance=failure_provenance,
            config=config,
        )
        failed["started_at"] = started_at
        failed["finished_at"] = datetime.now(UTC).isoformat()
        failed["exclusive_window"] = exclusive_window
        if window_error is not None and body_error is not None:
            failed["suppressed_body_failure"] = {
                "phase": body_error_phase,
                "error_type": type(body_error).__name__,
                "error": str(body_error),
            }
        write_json_checkpoint(output, failed)
        failed = finalize_result_provenance(
            failed,
            collect_provenance(command_argv),
        )
        write_json_checkpoint(output, failed)
        print(json.dumps(failed, indent=2, sort_keys=True), file=sys.stderr)
        return _failure_exit_code(error)

    if result is None:
        raise RuntimeError("exclusive router timing produced no result")
    result["started_at"] = started_at
    result["finished_at"] = datetime.now(UTC).isoformat()
    result["exclusive_window"] = exclusive_window
    write_json_checkpoint(output, result)
    result = finalize_result_provenance(
        result,
        collect_provenance(command_argv),
    )
    write_json_checkpoint(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["status"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
