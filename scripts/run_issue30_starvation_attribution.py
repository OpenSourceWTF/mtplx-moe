#!/usr/bin/env python3
"""Run the issue #30 telemetry off/on attribution campaign safely."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple


CAMPAIGN_ORDER = (
    "off-p01",
    "on-p01",
    "on-p02",
    "off-p02",
    "off-p03",
    "on-p03",
    "on-p04",
    "off-p04",
)
EXPECTED_QWEN_MODELS = ("mtplx-qwen36-27b-optimized-speed",)
EXCLUSIVE_LANE = Path("/tmp/mtplx-gpu-exclusive")
BENCHMARK_PROCESS_PATTERNS = (
    "benchmark_streamed_generation.py",
    "probe_mtp",
    "probe_paged",
)


class QwenState(NamedTuple):
    loaded: bool
    models: tuple[str, ...]


def _assert_qwen_state_restored(initial: QwenState, final: QwenState) -> None:
    if final != initial:
        raise RuntimeError(
            f"Qwen final state differs from the captured state: {final} != {initial}"
        )


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_text(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        rendered = " ".join(command)
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): {rendered}: {detail}"
        )
    return result.stdout.strip()


def _json_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_fields(
    value: Mapping[str, object],
    fields: Sequence[str],
    *,
    context: str,
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise RuntimeError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _require_mapping_field(
    value: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> Mapping[str, object]:
    item = value.get(field)
    if not isinstance(item, Mapping):
        raise RuntimeError(f"{context}.{field} must be an object")
    return item


def _require_exact_int(
    value: object,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{context} must be an integer >= {minimum}")
    return value


def _require_finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{context} must be a finite number")
    return result


def _require_nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a nonempty string")
    return value


def _require_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{context} must be bool")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{context} must be 64 lowercase hexadecimal digits")
    return value


def normalized_config_fingerprint(config: Mapping[str, object]) -> str:
    """Hash runtime settings after removing only the telemetry-local toggle."""

    normalized = dict(config)
    normalized.pop("resource_telemetry", None)
    return _json_fingerprint(normalized)


def normalized_configuration_fingerprint(
    summary: Mapping[str, object],
) -> str:
    """Hash the full resolved identity after removing only telemetry enablement."""

    _require_fields(
        summary,
        (
            "cache_scope",
            "slot_layout",
            "requested_concurrency",
            "execution_lane",
            "performance_settings",
        ),
        context="configuration summary",
    )
    _require_nonempty_string(
        summary["cache_scope"], context="configuration summary.cache_scope"
    )
    _require_nonempty_string(
        summary["slot_layout"], context="configuration summary.slot_layout"
    )
    _require_exact_int(
        summary["requested_concurrency"],
        context="configuration summary.requested_concurrency",
        minimum=1,
    )
    _require_nonempty_string(
        summary["execution_lane"], context="configuration summary.execution_lane"
    )
    settings_value = summary.get("performance_settings")
    if not isinstance(settings_value, Mapping):
        raise RuntimeError("configuration summary has no performance settings")
    required_settings = (
        "runtime_config",
        "prompt_identity",
        "generation",
        "model_artifact",
        "mtp",
        "prompt_options",
        "sampler",
        "scheduler",
        "seed",
    )
    _require_fields(
        settings_value,
        required_settings,
        context="configuration summary.performance_settings",
    )
    settings = dict(settings_value)
    runtime_value = settings.get("runtime_config")
    if not isinstance(runtime_value, Mapping):
        raise RuntimeError("performance settings have no runtime config")
    _require_fields(
        runtime_value,
        ("model_key", "resource_telemetry"),
        context="configuration summary.performance_settings.runtime_config",
    )
    _require_nonempty_string(
        runtime_value["model_key"],
        context="configuration summary.performance_settings.runtime_config.model_key",
    )
    _require_bool(
        runtime_value["resource_telemetry"],
        context=(
            "configuration summary.performance_settings.runtime_config."
            "resource_telemetry"
        ),
    )

    prompt_identity = _require_mapping_field(
        settings_value,
        "prompt_identity",
        context="configuration summary.performance_settings",
    )
    _require_fields(
        prompt_identity,
        ("content_bytes", "content_sha256", "token_count", "token_sha256"),
        context="configuration summary.performance_settings.prompt_identity",
    )
    _require_exact_int(
        prompt_identity["content_bytes"],
        context="configuration summary.performance_settings.prompt_identity.content_bytes",
    )
    _require_sha256(
        prompt_identity["content_sha256"],
        context="configuration summary.performance_settings.prompt_identity.content_sha256",
    )
    _require_exact_int(
        prompt_identity["token_count"],
        context="configuration summary.performance_settings.prompt_identity.token_count",
    )
    _require_sha256(
        prompt_identity["token_sha256"],
        context="configuration summary.performance_settings.prompt_identity.token_sha256",
    )

    generation = _require_mapping_field(
        settings_value,
        "generation",
        context="configuration summary.performance_settings",
    )
    _require_fields(
        generation,
        ("generation_profile", "max_tokens"),
        context="configuration summary.performance_settings.generation",
    )
    _require_nonempty_string(
        generation["generation_profile"],
        context=(
            "configuration summary.performance_settings.generation.generation_profile"
        ),
    )
    _require_exact_int(
        generation["max_tokens"],
        context="configuration summary.performance_settings.generation.max_tokens",
        minimum=1,
    )

    model_artifact = _require_mapping_field(
        settings_value,
        "model_artifact",
        context="configuration summary.performance_settings",
    )
    _require_fields(
        model_artifact,
        ("method", "expert_payload", "manifest", "harness_source"),
        context="configuration summary.performance_settings.model_artifact",
    )
    _require_nonempty_string(
        model_artifact["method"],
        context="configuration summary.performance_settings.model_artifact.method",
    )
    expert_payload = _require_mapping_field(
        model_artifact,
        "expert_payload",
        context="configuration summary.performance_settings.model_artifact",
    )
    _require_fields(
        expert_payload,
        ("sha256", "size"),
        context=(
            "configuration summary.performance_settings.model_artifact.expert_payload"
        ),
    )
    _require_sha256(
        expert_payload["sha256"],
        context=(
            "configuration summary.performance_settings.model_artifact."
            "expert_payload.sha256"
        ),
    )
    _require_exact_int(
        expert_payload["size"],
        context=(
            "configuration summary.performance_settings.model_artifact."
            "expert_payload.size"
        ),
        minimum=1,
    )
    manifest = _require_mapping_field(
        model_artifact,
        "manifest",
        context="configuration summary.performance_settings.model_artifact",
    )
    _require_fields(
        manifest,
        (
            "content_sha256",
            "declared_manifest_sha256",
            "model_key",
            "source_repo",
            "source_revision",
        ),
        context="configuration summary.performance_settings.model_artifact.manifest",
    )
    for field in ("content_sha256", "declared_manifest_sha256"):
        _require_sha256(
            manifest[field],
            context=(
                "configuration summary.performance_settings.model_artifact."
                f"manifest.{field}"
            ),
        )
    for field in ("model_key", "source_repo", "source_revision"):
        _require_nonempty_string(
            manifest[field],
            context=(
                "configuration summary.performance_settings.model_artifact."
                f"manifest.{field}"
            ),
        )
    harness_source = _require_mapping_field(
        model_artifact,
        "harness_source",
        context="configuration summary.performance_settings.model_artifact",
    )
    _require_fields(
        harness_source,
        ("dirty", "git_head", "source_sha256"),
        context=(
            "configuration summary.performance_settings.model_artifact.harness_source"
        ),
    )
    _require_bool(
        harness_source["dirty"],
        context=(
            "configuration summary.performance_settings.model_artifact."
            "harness_source.dirty"
        ),
    )
    _require_nonempty_string(
        harness_source["git_head"],
        context=(
            "configuration summary.performance_settings.model_artifact."
            "harness_source.git_head"
        ),
    )
    _require_sha256(
        harness_source["source_sha256"],
        context=(
            "configuration summary.performance_settings.model_artifact."
            "harness_source.source_sha256"
        ),
    )

    for field in ("mtp", "prompt_options", "sampler", "scheduler"):
        mapping = _require_mapping_field(
            settings_value,
            field,
            context="configuration summary.performance_settings",
        )
        if not mapping:
            raise RuntimeError(
                f"configuration summary.performance_settings.{field} must not be empty"
            )
    _require_exact_int(
        settings_value["seed"],
        context="configuration summary.performance_settings.seed",
    )

    runtime_config = dict(runtime_value)
    runtime_config.pop("resource_telemetry", None)
    settings["runtime_config"] = runtime_config
    return _json_fingerprint(
        {
            "cache_scope": summary.get("cache_scope"),
            "slot_layout": summary.get("slot_layout"),
            "requested_concurrency": summary.get("requested_concurrency"),
            "execution_lane": summary.get("execution_lane"),
            "performance_settings": settings,
        }
    )


def build_benchmark_command(
    *,
    repo: Path,
    model: Path,
    label: str,
    output_dir: Path,
    telemetry: bool,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "dev",
        "--extra",
        "server",
        "python",
        str(repo / "scripts" / "benchmark_streamed_generation.py"),
        str(model),
        str(model / "expert-manifest-sidecar.json"),
        "--model-key",
        "hy3-q4",
        "--memory-limit",
        "120259084288",
        "--runtime-reserve",
        "8589934592",
        "--expert-cache-limit",
        "83034243072",
        "--max-live-kv-tokens",
        "18888",
        "--cache-policy",
        "lru",
        "--cache-scope",
        "global",
        "--slot-layout",
        "component-banks",
        "--transient-slots",
        "32",
        "--read-chunk",
        "67108864",
        "--f-nocache",
        "--trust-sidecar",
        "--no-enable-mtp",
        "--chat",
        "--prompt-file",
        str(repo / "benchmarks" / "prompts" / "moe_streaming_realistic.md"),
        "--generation-profile",
        "deterministic",
        "--seed",
        "0",
        "--no-enable-thinking",
        "--max-tokens",
        "256",
        "--repeats",
        "1",
        "--concurrency",
        "1",
        "--max-prefills-per-step",
        "1",
        "--workload-shape",
        "static",
        "--no-window-telemetry",
    ]
    if telemetry:
        command.extend(
            (
                "--resource-telemetry",
                "--resource-sample-interval",
                "0.25",
                "--resource-max-samples",
                "4096",
                "--ssd-ceiling-gib-s",
                "12.47",
                "--no-powermetrics",
            )
        )
    else:
        command.append("--no-resource-telemetry")
    command.extend(
        (
            "--run-label",
            label,
            "--output-dir",
            str(output_dir),
            "--route-trace-json",
            str(output_dir / "routes.json"),
            "--output-json",
            str(output_dir / "result.json"),
        )
    )
    return command


def run_exclusive_window(
    workload: Callable[[], Any],
    *,
    acquire_lane: Callable[[], None],
    release_lane: Callable[[], None],
    capture_qwen: Callable[[], QwenState],
    stop_qwen: Callable[[QwenState], None],
    restore_qwen: Callable[[QwenState], None],
) -> Any:
    """Always restore captured Qwen state and release the exclusive lane."""

    lane_acquired = False
    state: QwenState | None = None
    try:
        with _blocked_termination_signals():
            acquire_lane()
            lane_acquired = True
        state = capture_qwen()
        stop_qwen(state)
        return workload()
    finally:
        with _blocked_termination_signals():
            try:
                if state is not None:
                    restore_qwen(state)
            finally:
                if lane_acquired:
                    release_lane()


@contextmanager
def _blocked_termination_signals():
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return
    previous = pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_sha(repo: Path) -> str:
    return _run_text(("git", "rev-parse", "HEAD"), cwd=repo)


def _assert_clean_sha(repo: Path, expected_sha: str | None = None) -> str:
    sha = _git_sha(repo)
    if expected_sha is not None and sha != expected_sha:
        raise RuntimeError(f"source SHA changed: expected {expected_sha}, found {sha}")
    dirty = _run_text(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repo,
    )
    if dirty:
        raise RuntimeError(f"benchmark requires a clean worktree:\n{dirty}")
    return sha


def _matching_processes() -> tuple[int, ...]:
    matches: set[int] = set()
    for pattern in BENCHMARK_PROCESS_PATTERNS:
        output = _run_text(("pgrep", "-f", pattern), check=False)
        for item in output.splitlines():
            try:
                pid = int(item.strip())
            except ValueError:
                continue
            if pid != os.getpid():
                matches.add(pid)
    return tuple(sorted(matches))


def _acquire_lane(*, poll_seconds: float) -> None:
    while True:
        competitors = _matching_processes()
        if EXCLUSIVE_LANE.exists():
            if not competitors:
                raise RuntimeError(
                    f"exclusive lane {EXCLUSIVE_LANE} exists without a live benchmark; "
                    "refusing to remove an ambiguous lock"
                )
            print(
                f"waiting for benchmark processes {competitors} and exclusive lane",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        if competitors:
            print(f"waiting for benchmark processes {competitors}", flush=True)
            time.sleep(poll_seconds)
            continue
        try:
            EXCLUSIVE_LANE.mkdir()
        except FileExistsError:
            continue
        competitors = _matching_processes()
        if not competitors:
            return
        EXCLUSIVE_LANE.rmdir()
        print(f"benchmark process raced lane acquisition: {competitors}", flush=True)
        time.sleep(poll_seconds)


def _release_lane() -> None:
    EXCLUSIVE_LANE.rmdir()


def _qwen_service(uid: int) -> str:
    return f"gui/{uid}/com.tea.qwen"


def _qwen_domain(uid: int) -> str:
    return f"gui/{uid}"


def _service_loaded(uid: int) -> bool:
    result = subprocess.run(
        ("launchctl", "print", _qwen_service(uid)),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _fetch_qwen_models(url: str, *, timeout: float) -> tuple[str, ...] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Qwen /v1/models response has no data list")
    models: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise RuntimeError("Qwen /v1/models response has an invalid model row")
        models.append(row["id"])
    if len(models) != len(set(models)):
        raise RuntimeError("Qwen /v1/models response contains duplicate model IDs")
    return tuple(models)


def _qwen_processes() -> tuple[int, ...]:
    output = _run_text(("pgrep", "-f", "mtplx.server.openai.*Qwen3.6"), check=False)
    return tuple(int(value) for value in output.splitlines() if value.strip().isdigit())


def _capture_qwen(
    *,
    uid: int,
    api_url: str,
    api_timeout: float,
    expected_models: tuple[str, ...],
) -> QwenState:
    loaded = _service_loaded(uid)
    models = _fetch_qwen_models(api_url, timeout=api_timeout)
    processes = _qwen_processes()
    if loaded:
        if models != expected_models or not processes:
            raise RuntimeError(
                "Qwen service state is ambiguous: loaded service must expose the "
                f"exact expected models and a server process; models={models}, "
                f"processes={processes}"
            )
        return QwenState(loaded=True, models=models)
    if models is not None or processes:
        raise RuntimeError(
            "Qwen state is ambiguous: launchd service is absent but API/process remains"
        )
    return QwenState(loaded=False, models=())


def _wait_for_qwen_stopped(
    *,
    uid: int,
    api_url: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if (
            not _service_loaded(uid)
            and _fetch_qwen_models(api_url, timeout=1.0) is None
            and not _qwen_processes()
        ):
            return
        time.sleep(poll_seconds)
    raise RuntimeError("Qwen did not fully stop before the exclusive benchmark")


def _stop_qwen(
    state: QwenState,
    *,
    uid: int,
    plist: Path,
    api_url: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    if not state.loaded:
        return
    subprocess.run(
        ("launchctl", "bootout", _qwen_domain(uid), str(plist)),
        check=True,
    )
    _wait_for_qwen_stopped(
        uid=uid,
        api_url=api_url,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def _restore_qwen(
    state: QwenState,
    *,
    uid: int,
    plist: Path,
    api_url: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    if not state.loaded:
        if (
            _service_loaded(uid)
            or _fetch_qwen_models(api_url, timeout=1.0) is not None
            or _qwen_processes()
        ):
            raise RuntimeError(
                "Qwen became loaded during an initially-unloaded campaign"
            )
        return
    if (
        _service_loaded(uid)
        or _fetch_qwen_models(api_url, timeout=1.0) is not None
        or _qwen_processes()
    ):
        raise RuntimeError("Qwen became active before controlled restoration")
    subprocess.run(
        ("launchctl", "bootstrap", _qwen_domain(uid), str(plist)),
        check=True,
    )
    deadline = time.monotonic() + timeout_seconds
    last_models: tuple[str, ...] | None = None
    while time.monotonic() < deadline:
        last_models = _fetch_qwen_models(api_url, timeout=1.0)
        if _service_loaded(uid) and last_models == state.models and _qwen_processes():
            return
        time.sleep(poll_seconds)
    raise RuntimeError(
        "Qwen did not restore its exact captured model list; "
        f"expected={state.models}, last={last_models}"
    )


def _command_observation(command: Sequence[str]) -> dict[str, object]:
    result = subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "argv": list(command),
        "exit_code": result.returncode,
        "output": result.stdout.strip(),
    }


def _sanitize_hardware_profile(payload: Mapping[str, object]) -> dict[str, object]:
    rows = payload.get("SPHardwareDataType")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
        raise RuntimeError("system_profiler returned no hardware row")
    allowed = (
        "machine_name",
        "machine_model",
        "chip_type",
        "physical_memory",
        "number_processors",
        "total_number_cores",
    )
    return {name: rows[0][name] for name in allowed if name in rows[0]}


def _hardware_snapshot() -> dict[str, object]:
    try:
        payload = json.loads(
            _run_text(("system_profiler", "SPHardwareDataType", "-json"))
        )
        if not isinstance(payload, dict):
            raise RuntimeError("system_profiler did not return a JSON object")
        return _sanitize_hardware_profile(payload)
    except (json.JSONDecodeError, RuntimeError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}


def _host_snapshot() -> dict[str, object]:
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = "unavailable"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": mlx_version,
        "hardware": _hardware_snapshot(),
        "memory_bytes": _command_observation(("sysctl", "-n", "hw.memsize")),
        "os": _command_observation(("sw_vers",)),
        "thermal": _command_observation(("pmset", "-g", "therm")),
        "fan": {
            "status": "unavailable",
            "reason": "no nonprivileged fan sensor is configured for this campaign",
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "MLX_METAL_CACHE_LIMIT",
                "MTPLX_MMAP_WORKERS",
                "MTPLX_NATIVE_EXPERT_IO",
                "PYTHONHASHSEED",
            )
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _stream_signature(payload: Mapping[str, Any]) -> str:
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("campaign requires exactly one benchmark repeat")
    streams = runs[0].get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise RuntimeError("benchmark result stream list must contain exactly one row")
    required = (
        "seed",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "token_ids",
        "text",
    )
    stable: list[dict[str, object]] = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, Mapping):
            raise RuntimeError("benchmark stream payload contains a non-object row")
        context = f"benchmark stream row {index}"
        _require_fields(stream, required, context=context)
        _require_exact_int(stream["seed"], context=f"{context}.seed")
        _require_exact_int(stream["prompt_tokens"], context=f"{context}.prompt_tokens")
        completion_tokens = _require_exact_int(
            stream["completion_tokens"], context=f"{context}.completion_tokens"
        )
        _require_nonempty_string(
            stream["finish_reason"], context=f"{context}.finish_reason"
        )
        token_ids = stream["token_ids"]
        if not isinstance(token_ids, list):
            raise RuntimeError(f"{context}.token_ids must be a list")
        for token_index, token_id in enumerate(token_ids):
            _require_exact_int(
                token_id,
                context=f"{context}.token_ids[{token_index}]",
            )
        if len(token_ids) != completion_tokens:
            raise RuntimeError(
                f"{context}.token_ids length must equal completion_tokens"
            )
        if not isinstance(stream["text"], str):
            raise RuntimeError(f"{context}.text must be a string")
        stable.append({field: stream[field] for field in required})
    return _json_fingerprint(stable)


def _cache_signature(payload: Mapping[str, Any]) -> str:
    run = payload["runs"][0]
    after = run.get("streaming_after", {})
    if not isinstance(after, dict):
        raise RuntimeError("benchmark result has no streaming_after object")
    cache = after.get("cache")
    cache_by_phase = after.get("cache_by_phase")
    incremental_misses = after.get("incremental_misses")
    slots = after.get("slots")
    if not all(
        isinstance(value, Mapping)
        for value in (cache, cache_by_phase, incremental_misses, slots)
    ):
        raise RuntimeError("benchmark result is missing deterministic cache evidence")
    io_value = slots.get("io")
    metrics = slots.get("metrics")
    states = slots.get("states")
    if not isinstance(io_value, Mapping):
        raise RuntimeError("benchmark result has no slot I/O metrics object")
    if not isinstance(metrics, Mapping) or not isinstance(states, Mapping):
        raise RuntimeError("benchmark result has no slot health metrics")
    deterministic_io_fields = (
        "record_requests",
        "source_record_requests",
        "sidecar_record_requests",
        "read_operations",
        "python_preadv_invocations",
        "preadv_bytes_returned",
        "native_positional_calls",
        "native_bytes_returned",
        "requested_bytes",
        "read_bytes",
        "short_reads",
        "integrity_errors",
        "cancellations",
        "deadline_errors",
        "io_errors",
    )
    cache_counter_fields = (
        "bytes_read",
        "evictions",
        "expert_hits",
        "expert_misses",
        "expert_requests",
        "persistent_loads",
        "route_calls",
        "shared_expert_assignments",
        "transient_loads",
        "unique_expert_requests",
    )
    _require_fields(cache, cache_counter_fields, context="benchmark aggregate cache")
    for field in cache_counter_fields:
        _require_exact_int(cache[field], context=f"benchmark aggregate cache.{field}")
    _require_finite_number(
        cache.get("hit_rate"), context="benchmark aggregate cache.hit_rate"
    )
    _require_fields(
        cache_by_phase,
        ("prefill", "decode"),
        context="benchmark cache_by_phase",
    )
    for phase in ("prefill", "decode"):
        phase_cache = _require_mapping_field(
            cache_by_phase,
            phase,
            context="benchmark cache_by_phase",
        )
        _require_fields(
            phase_cache,
            cache_counter_fields,
            context=f"benchmark cache_by_phase.{phase}",
        )
        for field in cache_counter_fields:
            _require_exact_int(
                phase_cache[field],
                context=f"benchmark cache_by_phase.{phase}.{field}",
            )
        _require_finite_number(
            phase_cache.get("hit_rate"),
            context=f"benchmark cache_by_phase.{phase}.hit_rate",
        )
    _require_fields(
        incremental_misses,
        ("routes", "parts"),
        context="benchmark incremental_misses",
    )
    for field in ("routes", "parts"):
        _require_exact_int(
            incremental_misses[field],
            context=f"benchmark incremental_misses.{field}",
        )
    _require_fields(
        io_value,
        deterministic_io_fields,
        context="benchmark slot I/O metrics",
    )
    for field in deterministic_io_fields:
        _require_exact_int(
            io_value[field], context=f"benchmark slot I/O metrics.{field}"
        )
    _require_fields(
        metrics,
        ("load_failures", "completion_fence_failures"),
        context="benchmark slot metrics",
    )
    for field in ("load_failures", "completion_fence_failures"):
        _require_exact_int(metrics[field], context=f"benchmark slot metrics.{field}")
    return _json_fingerprint(
        {
            "cache": cache,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            "io": {name: io_value.get(name) for name in deterministic_io_fields},
            "slot_failures": {
                "load_failures": metrics.get("load_failures"),
                "completion_fence_failures": metrics.get("completion_fence_failures"),
            },
        }
    )


def _final_health(payload: Mapping[str, Any]) -> dict[str, object]:
    run = payload["runs"][0]
    after = run.get("streaming_after")
    slots = after.get("slots") if isinstance(after, Mapping) else None
    metrics = slots.get("metrics") if isinstance(slots, Mapping) else None
    states = slots.get("states") if isinstance(slots, Mapping) else None
    if not isinstance(metrics, Mapping) or not isinstance(states, Mapping):
        raise RuntimeError("benchmark result has no final slot-health evidence")
    values = {
        "load_failures": metrics.get("load_failures"),
        "completion_fence_failures": metrics.get("completion_fence_failures"),
        "active_routes": metrics.get("active_routes"),
        "pins": slots.get("pins"),
        "loading_slots": states.get("loading"),
        "failed_slots": states.get("failed"),
    }
    valid = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in values.values()
    )
    return {**values, "ok": valid and all(value == 0 for value in values.values())}


def _route_evidence(route_trace: Mapping[str, object]) -> tuple[str, str]:
    entries = route_trace.get("entries")
    manifest_sha256 = route_trace.get("manifest_sha256")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("route entries must be a nonempty list")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise RuntimeError("route manifest SHA must be 64 lowercase hexadecimal digits")
    required = ("expert_ids", "layer", "phase", "token_count", "trace_epoch")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"route entry {index} must be an object")
        context = f"route entry {index}"
        _require_fields(entry, required, context=context)
        expert_ids = entry["expert_ids"]
        if not isinstance(expert_ids, list) or not expert_ids:
            raise RuntimeError(f"{context}.expert_ids must be a nonempty list")
        for expert_index, expert_id in enumerate(expert_ids):
            _require_exact_int(
                expert_id,
                context=f"{context}.expert_ids[{expert_index}]",
            )
        _require_exact_int(entry["layer"], context=f"{context}.layer")
        phase = entry["phase"]
        if phase not in {"prefill", "decode"}:
            raise RuntimeError(f"{context}.phase must be prefill or decode")
        _require_exact_int(
            entry["token_count"], context=f"{context}.token_count", minimum=1
        )
        _require_exact_int(entry["trace_epoch"], context=f"{context}.trace_epoch")
        if phase == "decode":
            _require_fields(entry, ("decode_step",), context=context)
            _require_exact_int(entry["decode_step"], context=f"{context}.decode_step")
    return _json_fingerprint(entries), manifest_sha256


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _run_benchmark_process(command: Sequence[str], *, cwd: Path) -> int:
    """Run one benchmark in its own process group and reap it on interruption."""

    process = subprocess.Popen(list(command), cwd=cwd, start_new_session=True)
    try:
        exit_code = int(process.wait())
    except BaseException:
        leader_reaped = False
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
            leader_reaped = True
        except subprocess.TimeoutExpired:
            pass
        if _process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not leader_reaped:
                try:
                    process.wait(timeout=5.0)
                    leader_reaped = True
                except subprocess.TimeoutExpired:
                    pass
        if not _wait_for_process_group_exit(process.pid, timeout_seconds=5.0):
            raise RuntimeError("benchmark process group survived SIGKILL")
        if not leader_reaped:
            process.wait(timeout=5.0)
        raise
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not _wait_for_process_group_exit(process.pid, timeout_seconds=5.0):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not _wait_for_process_group_exit(process.pid, timeout_seconds=5.0):
                raise RuntimeError("benchmark descendants survived SIGKILL")
        raise RuntimeError("benchmark descendants remained after the leader exited")
    return exit_code


def _run_record(
    *,
    repo: Path,
    model: Path,
    campaign_dir: Path,
    variant: str,
    sha: str,
    on_start: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    _assert_clean_sha(repo, sha)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"moe-runtime-issue30-{variant}-{timestamp}-{sha[:12]}"
    output_dir = campaign_dir / variant
    telemetry = variant.startswith("on-")
    command = build_benchmark_command(
        repo=repo,
        model=model,
        label=label,
        output_dir=output_dir,
        telemetry=telemetry,
    )
    started = _utc_now()
    thermal_before = _command_observation(("pmset", "-g", "therm"))
    record: dict[str, object] = {
        "variant": variant,
        "telemetry": telemetry,
        "label": label,
        "argv": command,
        "started_at_utc": started,
        "finished_at_utc": None,
        "exit_code": None,
        "thermal_before": thermal_before,
        "thermal_after": None,
        "output_dir": str(output_dir),
        "result_json": str(output_dir / "result.json"),
        "route_trace_json": str(output_dir / "routes.json"),
    }
    if on_start is not None:
        on_start(record)
    try:
        exit_code = _run_benchmark_process(command, cwd=repo)
    except BaseException as exc:
        record.update(
            {
                "finished_at_utc": _utc_now(),
                "thermal_after": _command_observation(("pmset", "-g", "therm")),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    record.update(
        {
            "finished_at_utc": _utc_now(),
            "exit_code": exit_code,
            "thermal_after": _command_observation(("pmset", "-g", "therm")),
        }
    )
    if exit_code != 0:
        return record
    payload = _load_json(output_dir / "result.json")
    route_trace = _load_json(output_dir / "routes.json")
    if payload.get("git_commit") != sha:
        raise RuntimeError(f"{variant} result used unexpected source SHA")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{variant} result has no runtime config")
    expected_toggle = telemetry
    if config.get("resource_telemetry") is not expected_toggle:
        raise RuntimeError(f"{variant} result has the wrong telemetry toggle")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError(f"{variant} result does not contain exactly one repeat")
    configuration_summary = payload.get("configuration_summary")
    if not isinstance(configuration_summary, Mapping):
        raise RuntimeError(f"{variant} result has no configuration summary")
    route_signature, manifest_sha256 = _route_evidence(route_trace)
    completion_tps = runs[0].get("aggregate_completion_tokens_per_second")
    if (
        isinstance(completion_tps, bool)
        or not isinstance(completion_tps, (int, float))
        or not math.isfinite(float(completion_tps))
        or completion_tps <= 0
    ):
        raise RuntimeError(f"{variant} result has invalid completion throughput")
    final_health = _final_health(payload)
    record.update(
        {
            "raw_configuration_fingerprint": configuration_summary.get(
                "configuration_fingerprint"
            ),
            "normalized_runtime_config_fingerprint": (
                normalized_config_fingerprint(config)
            ),
            "normalized_configuration_fingerprint": (
                normalized_configuration_fingerprint(configuration_summary)
            ),
            "stream_signature": _stream_signature(payload),
            "cache_signature": _cache_signature(payload),
            "route_signature": route_signature,
            "manifest_sha256": manifest_sha256,
            "completion_tokens_per_second": completion_tps,
            "final_health": final_health,
            "final_health_ok": final_health["ok"],
        }
    )
    _assert_clean_sha(repo, sha)
    return record


def _parity_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successful = [record for record in records if record.get("exit_code") == 0]
    fields = (
        "normalized_configuration_fingerprint",
        "stream_signature",
        "cache_signature",
        "route_signature",
        "manifest_sha256",
    )
    checks = {
        field: len({record.get(field) for record in successful}) == 1
        for field in fields
    }
    raw_by_arm = {
        arm: sorted(
            {
                str(record.get("raw_configuration_fingerprint"))
                for record in successful
                if bool(record.get("telemetry")) is enabled
            }
        )
        for arm, enabled in (("off", False), ("on", True))
    }
    raw_fingerprints_valid = (
        len(raw_by_arm["off"]) == 1
        and len(raw_by_arm["on"]) == 1
        and raw_by_arm["off"] != raw_by_arm["on"]
        and "None" not in {*raw_by_arm["off"], *raw_by_arm["on"]}
    )
    checks["raw_configuration_fingerprints"] = raw_fingerprints_valid
    checks["final_health"] = all(
        record.get("final_health_ok") is True for record in successful
    )
    return {
        "successful_runs": len(successful),
        "expected_runs": len(CAMPAIGN_ORDER),
        "checks": checks,
        "all_selected_parity": len(successful) == len(CAMPAIGN_ORDER)
        and all(checks.values()),
        "raw_configuration_fingerprints_by_arm": raw_by_arm,
        "raw_fingerprints_expected_to_differ_by_telemetry_toggle": True,
    }


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            Path.home()
            / ".cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots"
            / "160619d3f96c8470350b6dac0ef033a8381551e3"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "benchmarks" / "raw" / "moe-runtime",
    )
    parser.add_argument(
        "--qwen-api-url",
        default="http://127.0.0.1:8080/v1/models",
    )
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--qwen-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the frozen order and command templates without changing state.",
    )
    return parser


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0 or args.qwen_timeout_seconds <= 0:
        raise ValueError("poll and Qwen timeout values must be positive")
    required = (
        args.repo / "scripts" / "benchmark_streamed_generation.py",
        args.repo / "benchmarks" / "prompts" / "moe_streaming_realistic.md",
        args.model / "expert-manifest-sidecar.json",
        args.qwen_plist,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required campaign inputs are missing: {missing}")


def _plan_payload(args: argparse.Namespace) -> dict[str, object]:
    placeholder = "moe-runtime-issue30-VARIANT-YYYYMMDDTHHMMSSZ-SHORTSHA"
    output = args.output_root / "CAMPAIGN" / "VARIANT"
    return {
        "order": list(CAMPAIGN_ORDER),
        "off_argv": build_benchmark_command(
            repo=args.repo,
            model=args.model,
            label=placeholder,
            output_dir=output,
            telemetry=False,
        ),
        "on_argv": build_benchmark_command(
            repo=args.repo,
            model=args.model,
            label=placeholder,
            output_dir=output,
            telemetry=True,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.qwen_plist = args.qwen_plist.expanduser().resolve()
    _validate_inputs(args)
    if args.plan_only:
        print(json.dumps(_plan_payload(args), indent=2, sort_keys=True))
        return 0

    sha = _assert_clean_sha(args.repo)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    campaign_id = f"moe-runtime-issue30-campaign-p{os.getpid()}-{timestamp}-{sha[:12]}"
    campaign_dir = args.output_root / campaign_id
    args.output_root.mkdir(parents=True, exist_ok=True)
    campaign_dir.mkdir(exist_ok=False)
    manifest_path = campaign_dir / "campaign.json"
    manifest: dict[str, object] = {
        "schema": "mtplx-issue30-starvation-campaign-v1",
        "campaign_id": campaign_id,
        "status": "starting",
        "git_commit": sha,
        "physical_order": list(CAMPAIGN_ORDER),
        "created_at_utc": _utc_now(),
        "host": _host_snapshot(),
        "runs": [],
    }
    _atomic_write_json(manifest_path, manifest)
    uid = os.getuid()

    def capture_qwen() -> QwenState:
        state = _capture_qwen(
            uid=uid,
            api_url=args.qwen_api_url,
            api_timeout=3.0,
            expected_models=EXPECTED_QWEN_MODELS,
        )
        manifest["initial_qwen"] = state._asdict()
        manifest["status"] = "exclusive_lane_acquired"
        _atomic_write_json(manifest_path, manifest)
        return state

    def stop_qwen(state: QwenState) -> None:
        _stop_qwen(
            state,
            uid=uid,
            plist=args.qwen_plist,
            api_url=args.qwen_api_url,
            timeout_seconds=args.qwen_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        manifest["status"] = "benchmarking"
        _atomic_write_json(manifest_path, manifest)

    def restore_qwen(state: QwenState) -> None:
        _restore_qwen(
            state,
            uid=uid,
            plist=args.qwen_plist,
            api_url=args.qwen_api_url,
            timeout_seconds=args.qwen_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        final = _capture_qwen(
            uid=uid,
            api_url=args.qwen_api_url,
            api_timeout=3.0,
            expected_models=state.models or EXPECTED_QWEN_MODELS,
        )
        _assert_qwen_state_restored(state, final)
        manifest["final_qwen"] = final._asdict()
        _atomic_write_json(manifest_path, manifest)

    def workload() -> None:
        records = manifest["runs"]
        assert isinstance(records, list)
        for variant in CAMPAIGN_ORDER:

            def persist_started(record: dict[str, object]) -> None:
                records.append(record)
                _atomic_write_json(manifest_path, manifest)

            record = _run_record(
                repo=args.repo,
                model=args.model,
                campaign_dir=campaign_dir,
                variant=variant,
                sha=sha,
                on_start=persist_started,
            )
            _atomic_write_json(manifest_path, manifest)
            if record["exit_code"] != 0:
                raise RuntimeError(f"benchmark variant {variant} failed")
        parity = _parity_summary(records)
        manifest["parity"] = parity
        if not parity["all_selected_parity"]:
            raise RuntimeError("selected deterministic parity checks failed")

    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    prior_handlers = {
        signum: signal.signal(signum, interrupted)
        for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        run_exclusive_window(
            workload,
            acquire_lane=lambda: _acquire_lane(poll_seconds=args.poll_seconds),
            release_lane=_release_lane,
            capture_qwen=capture_qwen,
            stop_qwen=stop_qwen,
            restore_qwen=restore_qwen,
        )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        raise
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
