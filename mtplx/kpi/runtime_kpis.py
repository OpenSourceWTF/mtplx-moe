"""Run-envelope extraction and gates for public MTPLX commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mtplx.benchmarks.schema import now_run_id
from mtplx.profiles import (
    LONG_RESPONSE_STAGED_ENV,
    NATIVE_MTP_60_FAST_PATH_ENV,
    get_profile,
)

EXIT_UNSUPPORTED_MODEL = 2
EXIT_EXACTNESS = 3
EXIT_QUALITY = 4
EXIT_STRICT_GATE = 5
EXIT_TELEMETRY = 6

NATIVE_MTP_FAST_PATH_ENV = dict(NATIVE_MTP_60_FAST_PATH_ENV)

PROMPT_SUITES = {
    "default": "mtplx/benchmarks/prompts/default.jsonl",
    "long_code": "mtplx/benchmarks/prompts/long_code.jsonl",
    "long-code": "mtplx/benchmarks/prompts/long_code.jsonl",
    "cold-long-code-192": "mtplx/benchmarks/prompts/long_code.jsonl",
    "long_code_uncapped": "mtplx/benchmarks/prompts/long_code_uncapped.jsonl",
    "long-code-uncapped": "mtplx/benchmarks/prompts/long_code_uncapped.jsonl",
    "python_modules_long": "mtplx/benchmarks/prompts/python_modules_long.jsonl",
    "python-modules-long": "mtplx/benchmarks/prompts/python_modules_long.jsonl",
    "flappy": "mtplx/benchmarks/prompts/flappy.jsonl",
    "calibration_coding": "mtplx/benchmarks/prompts/calibration_coding.jsonl",
    "calibration-coding": "mtplx/benchmarks/prompts/calibration_coding.jsonl",
}

DISTRIBUTION_SUITES = {
    "distribution-smoke": ["flappy", "python_modules_long", "long_code"],
    "champion-bakeoff": ["flappy", "python_modules_long", "cold-long-code-192"],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_json(path: Path | str, value: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_output_path(kind: str, run_id: str | None = None) -> Path:
    return Path("outputs") / "cli" / kind / f"{run_id or now_run_id(kind)}.json"


def prompt_suite_path(suite: str | None, *, fallback: str = "default") -> str:
    key = suite or fallback
    if key in PROMPT_SUITES:
        return PROMPT_SUITES[key]
    path = Path(key)
    if path.exists():
        return str(path)
    raise SystemExit(f"unknown prompt suite: {key}")


def distribution_suite_names(suite: str | None) -> list[str]:
    key = suite or "distribution-smoke"
    if key in DISTRIBUTION_SUITES:
        return DISTRIBUTION_SUITES[key]
    if "," in key:
        return [part.strip() for part in key.split(",") if part.strip()]
    return [key]


def apply_native_mtp_fast_path_env() -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in NATIVE_MTP_FAST_PATH_ENV}
    for key, value in NATIVE_MTP_FAST_PATH_ENV.items():
        os.environ[key] = value
    return previous


def exact_paged_attention_env(
    *,
    attention_impl: str = "mlx_vector_paged",
    block_size: int = 16,
    num_blocks: int = 1024,
    partitioned: bool = True,
    partition_threshold: int = 2048,
    partition_size: int = 512,
) -> dict[str, str]:
    env = {
        "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": str(block_size),
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": str(num_blocks),
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": str(attention_impl),
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": str(partition_threshold),
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": str(partition_size),
    }
    if partitioned:
        env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] = "1"
    return env


EXACT_PAGED_ATTENTION_ENV = exact_paged_attention_env()


def long_response_staged_env() -> dict[str, str]:
    return dict(LONG_RESPONSE_STAGED_ENV)


def public_bench_runtime_profile_env(
    *,
    suite: str,
    max_tokens: int,
    exact_paged_env: dict[str, str] | None = None,
    profile: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Return the intended public benchmark runtime profile.

    The v0.1 default is always ``stable``.  The historical 60+ tok/s
    path is exposed only through ``profile="performance-cold"`` (or its legacy
    alias), not through suite names.
    """

    _ = (suite, max_tokens)
    selected = get_profile(profile)
    env = selected.env_dict()
    if selected.name in {"stable", "exact", "max-diagnostic"} and exact_paged_env:
        env.update(exact_paged_env)
    return selected.runtime_profile, env


def fast_path_env_status() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "expected": expected,
            "observed": os.environ.get(key),
            "ok": os.environ.get(key) == expected,
        }
        for key, expected in NATIVE_MTP_FAST_PATH_ENV.items()
    }


def run_exactness_smoke(
    model: str,
    *,
    context: int = 2048,
    prompt_suite: str = "mtplx/benchmarks/prompts/flappy.jsonl",
    prompt_index: int = 0,
    output: str | Path | None = None,
    attention_impl: str = "mlx_vector_paged",
    block_size: int = 16,
    num_blocks: int = 1024,
    partitioned: bool = True,
    partition_threshold: int = 2048,
    partition_size: int = 512,
) -> dict[str, Any]:
    """Run the Phase 0H smoke gate through the source-tree script."""

    out = Path(output) if output else default_output_path("exactness-smoke")
    script = repo_root() / "scripts" / "phase0h_paged_verifier_exactness.py"
    if not script.exists():
        return {
            "passed": False,
            "returncode": EXIT_EXACTNESS,
            "error": f"missing Phase 0H script: {script}",
            "output": str(out),
        }
    cmd = [
        sys.executable,
        str(script),
        "--model",
        str(model),
        "--contexts",
        str(context),
        "--prompt-suite",
        str(prompt_suite),
        "--prompt-index",
        str(prompt_index),
        "--attention-impl",
        str(attention_impl),
        "--block-size",
        str(block_size),
        "--num-blocks",
        str(num_blocks),
        "--partition-threshold",
        str(partition_threshold),
        "--partition-size",
        str(partition_size),
        "--output",
        str(out),
    ]
    if partitioned:
        cmd.append("--partitioned")
    else:
        cmd.append("--no-partitioned")
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "elapsed_s": time.perf_counter() - started,
        "command": cmd,
        "output": str(out),
        "profile": {
            "attention_impl": attention_impl,
            "block_size": int(block_size),
            "num_blocks": int(num_blocks),
            "partitioned": bool(partitioned),
            "partition_threshold": int(partition_threshold),
            "partition_size": int(partition_size),
        },
        "stdout_tail": proc.stdout[-4000:],
    }


def _first_depth(result: dict[str, Any]) -> dict[str, Any] | None:
    depths = result.get("depths") or []
    return depths[0] if depths else None


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    depth = _first_depth(result)
    if not depth:
        return []
    return list(depth.get("rows") or [])


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    depth = _first_depth(result)
    if not depth:
        return {}
    return dict(depth.get("summary") or {})


def _quality_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        for validation in row.get("validations") or []:
            if not validation.get("passed"):
                failures.append(
                    {
                        "prompt_id": row.get("prompt_id"),
                        "validation": validation.get("name"),
                        "detail": validation.get("detail", ""),
                    }
                )
    return failures


def _acceptance_smells(summary: dict[str, Any], *, floor: float = 0.4) -> list[dict[str, Any]]:
    smells: list[dict[str, Any]] = []
    for index, value in enumerate(summary.get("mean_accept_probability_by_depth") or []):
        if value is not None and float(value) < floor:
            smells.append(
                {
                    "depth_index": index,
                    "mean_accept_probability": float(value),
                    "floor": floor,
                }
            )
    return smells


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _token_window_rate(rows: list[dict[str, Any]], *, size: int, reverse: bool = False) -> float | None:
    ordered = list(reversed(rows)) if reverse else rows
    tokens = 0.0
    elapsed = 0.0
    for row in ordered:
        delta_tokens = float(row.get("generated_tokens_delta") or 0.0)
        if delta_tokens <= 0:
            continue
        delta_elapsed = float(row.get("elapsed_s") or 0.0)
        take = min(float(size) - tokens, delta_tokens)
        fraction = take / delta_tokens if delta_tokens else 0.0
        tokens += take
        elapsed += delta_elapsed * fraction
        if tokens >= float(size):
            break
    if tokens <= 0 or elapsed <= 0:
        return None
    return tokens / elapsed


def _late_verify_ms(rows: list[dict[str, Any]], *, size: int = 10) -> float | None:
    tokens = 0.0
    weighted_ms = 0.0
    for row in reversed(rows):
        delta_tokens = float(row.get("generated_tokens_delta") or 0.0)
        if delta_tokens <= 0:
            continue
        verify_ms = row.get("verify_ms_per_call_delta")
        if verify_ms is None:
            continue
        take = min(float(size) - tokens, delta_tokens)
        tokens += take
        weighted_ms += float(verify_ms) * take
        if tokens >= float(size):
            break
    if tokens <= 0:
        return None
    return weighted_ms / tokens


def summarize_decode_trace(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "available": False}
    trace_path = Path(path)
    rows = [row for row in _read_jsonl(trace_path) if row.get("event") == "decode_trace_bucket"]
    positive = [row for row in rows if float(row.get("generated_tokens_delta") or 0.0) > 0]
    first64 = _token_window_rate(positive, size=64)
    last64 = _token_window_rate(positive, size=64, reverse=True)
    first10 = _token_window_rate(positive, size=10)
    last10 = _token_window_rate(positive, size=10, reverse=True)
    generated = int(max((row.get("generated_tokens_total") or 0 for row in rows), default=0))
    return {
        "path": str(trace_path),
        "available": bool(rows),
        "buckets": len(rows),
        "positive_buckets": len(positive),
        "generated_tokens": generated,
        "first64_tok_s": first64,
        "last64_tok_s": last64,
        "last64_over_first64": (last64 / first64) if first64 and last64 else None,
        "first10_tok_s": first10,
        "last10_tok_s": last10,
        "last10_over_first10": (last10 / first10) if first10 and last10 else None,
        "late_verify_ms": _late_verify_ms(positive, size=10),
        "cache_gib_last": (
            float(rows[-1].get("mlx_memory", {}).get("cache_memory_bytes") or 0.0) / (1024**3)
            if rows and isinstance(rows[-1].get("mlx_memory"), dict)
            else None
        ),
    }


def _runtime_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = int(summary.get("generated_tokens") or sum(int(row.get("generated_tokens") or 0) for row in rows))
    elapsed = float(summary.get("elapsed_s") or sum(float(row.get("elapsed_s") or 0.0) for row in rows))
    verify_calls = int(summary.get("verify_calls") or sum(int(row.get("verify_calls") or 0) for row in rows))
    verify_s = float(summary.get("verify_time_s") or sum(float(row.get("verify_time_s") or 0.0) for row in rows))
    draft_s = float(summary.get("draft_time_s") or sum(float(row.get("draft_time_s") or 0.0) for row in rows))
    accept_s = float(summary.get("accept_time_s") or sum(float(row.get("accept_time_s") or 0.0) for row in rows))
    return {
        "generated_tokens": generated,
        "elapsed_s": elapsed,
        "tok_s": generated / elapsed if elapsed > 0 else None,
        "mean_tok_s": summary.get("mean_tok_s"),
        "verify_calls": verify_calls,
        "verify_ms_per_call": (1000.0 * verify_s / verify_calls) if verify_calls else None,
        "verify_time_s": verify_s,
        "draft_time_s": draft_s,
        "accept_time_s": accept_s,
        "target_forward_time_s": summary.get("target_forward_time_s"),
        "peak_memory_bytes": summary.get("peak_memory_bytes"),
        "accepted_by_depth": summary.get("accepted_by_depth"),
        "drafted_by_depth": summary.get("drafted_by_depth"),
        "mean_accept_probability_by_depth": summary.get("mean_accept_probability_by_depth"),
        "acceptance_by_depth": summary.get("acceptance_by_depth"),
        "correction_tokens": summary.get("correction_tokens"),
        "bonus_tokens": summary.get("bonus_tokens"),
    }


def _strict_gates(
    runtime: dict[str, Any],
    *,
    suite: str,
    strict_cold: bool,
    strict_product: bool,
    telemetry: dict[str, Any],
    trace: dict[str, Any],
    fan_controlled: bool,
) -> dict[str, bool]:
    tok_s = runtime.get("tok_s")
    gates: dict[str, bool] = {}
    if strict_cold or suite == "cold-long-code-192":
        gates["cold_tok_s_ge_55"] = bool(tok_s is not None and float(tok_s) >= 55.0)
    if strict_product:
        last64_ratio = trace.get("last64_over_first64")
        last10_ratio = trace.get("last10_over_first10")
        late_verify = trace.get("late_verify_ms")
        gates["flappy_tok_s_ge_50"] = bool(tok_s is not None and float(tok_s) >= 50.0)
        gates["last64_over_first64_ge_0_90"] = bool(
            last64_ratio is not None and float(last64_ratio) >= 0.90
        )
        gates["last10_over_first10_ge_0_85"] = bool(
            last10_ratio is not None and float(last10_ratio) >= 0.85
        )
        gates["late_verify_le_75ms"] = bool(
            late_verify is not None and float(late_verify) <= 75.0
        )
        gates["telemetry_available"] = not bool(telemetry.get("telemetry_unavailable"))
        gates["no_fan_control"] = not fan_controlled
    return gates


def build_benchmark_envelope(
    *,
    result: dict[str, Any],
    model_inspection: dict[str, Any],
    run_id: str,
    suite: str,
    exactness_smoke: dict[str, Any] | None,
    fan_controlled: bool,
    strict: bool,
    strict_cold: bool,
    telemetry: dict[str, Any] | None = None,
    decode_trace_path: Path | str | None = None,
    runtime_profile: str | None = None,
    runtime_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = _rows(result)
    summary = _summary(result)
    runtime = _runtime_metrics(summary, rows)
    telemetry = telemetry or {"telemetry_unavailable": True}
    trace = summarize_decode_trace(decode_trace_path)
    quality_failures = _quality_failures(rows)
    acceptance_smells = _acceptance_smells(summary)
    strict_gates = _strict_gates(
        runtime,
        suite=suite,
        strict_cold=strict_cold,
        strict_product=strict,
        telemetry=telemetry,
        trace=trace,
        fan_controlled=fan_controlled,
    )
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": suite,
        "model": model_inspection,
        "fan_controlled": fan_controlled,
        "runtime_profile": runtime_profile,
        "runtime_env": runtime_env or {},
        "fast_path_env": fast_path_env_status(),
        "runtime": runtime,
        "decode_trace": trace,
        "thermal": telemetry,
        "dispatch": {"dispatch_trace_attached": False, "command_buffers_per_token": None},
        "quality": {
            "passed": not quality_failures,
            "failures": quality_failures,
            "acceptance_smells": acceptance_smells,
        },
        "correctness": {
            "exactness_smoke": exactness_smoke,
            "full_exactness": None,
            "distribution_exactness": None,
        },
        "strict_gates": strict_gates,
        "strict_passed": all(strict_gates.values()) if strict_gates else None,
    }
