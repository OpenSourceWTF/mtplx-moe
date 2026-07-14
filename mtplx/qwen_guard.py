"""Fail-closed Qwen stop/restore guard for explicit MLX execution windows."""

from __future__ import annotations

import json
import math
import os
import plistlib
import stat
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


QWEN_LABEL = "com.tea.qwen"
QWEN_PROCESS_PATTERN = "mtplx.server.openai.*Qwen3.6"
EXPECTED_QWEN_MODELS = ("mtplx-qwen36-27b-optimized-speed",)
_MAX_PLIST_BYTES = 1024 * 1024
_MAX_API_BYTES = 1024 * 1024
_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class QwenState:
    loaded: bool
    models: tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class _QwenObservation:
    loaded: bool
    models: tuple[str, ...] | None
    processes: tuple[int, ...]


CommandRunner = Callable[[tuple[str, ...]], CommandResult]
ModelFetcher = Callable[[str, float], tuple[str, ...] | None]


def _run_command(command: tuple[str, ...]) -> CommandResult:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _fetch_models(api_url: str, timeout: float) -> tuple[str, ...] | None:
    try:
        with urllib.request.urlopen(api_url, timeout=timeout) as response:  # noqa: S310
            payload = response.read(_MAX_API_BYTES + 1)
    except (OSError, TimeoutError, urllib.error.URLError):
        return None
    if len(payload) > _MAX_API_BYTES:
        raise RuntimeError("Qwen /v1/models response exceeds its size bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Qwen /v1/models response is malformed: {exc}") from exc
    rows = value.get("data") if isinstance(value, dict) else None
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


def _validate_qwen_launcher(path_value: str, *, uid: int) -> None:
    path = Path(path_value)
    if not path.is_absolute() or "qwen" not in path.name.lower():
        raise ValueError("Qwen plist wrapper must be an absolute Qwen-named executable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Qwen plist wrapper is not a no-follow file: {exc}") from exc
    try:
        status = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ValueError("Qwen plist wrapper must be a single-link regular file")
    if status.st_uid != uid:
        raise ValueError("Qwen plist wrapper must be owned by the current user")
    if status.st_mode & 0o022:
        raise ValueError("Qwen plist wrapper must not be group- or world-writable")
    if not status.st_mode & stat.S_IXUSR:
        raise ValueError("Qwen plist wrapper must be executable by its owner")


def _read_validated_plist(plist: Path, *, uid: int) -> Path:
    path = plist.expanduser()
    if not path.is_absolute():
        raise ValueError("Qwen plist path must be absolute")
    if path.name != f"{QWEN_LABEL}.plist":
        raise ValueError(f"Qwen plist must be named {QWEN_LABEL}.plist")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Qwen plist must be a no-follow regular file: {exc}") from exc
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError("Qwen plist must be a single-link regular file")
        if status.st_uid != uid:
            raise ValueError("Qwen plist must be owned by the current user")
        if status.st_mode & 0o022:
            raise ValueError("Qwen plist must not be group- or world-writable")
        if status.st_size > _MAX_PLIST_BYTES:
            raise ValueError("Qwen plist exceeds its size bound")
        chunks: list[bytes] = []
        remaining = status.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError("Qwen plist ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    try:
        value = plistlib.loads(b"".join(chunks))
    except plistlib.InvalidFileException as exc:
        raise ValueError(f"Qwen plist is malformed: {exc}") from exc
    if not isinstance(value, dict) or value.get("Label") != QWEN_LABEL:
        raise ValueError(f"Qwen plist Label must be exactly {QWEN_LABEL}")
    arguments = value.get("ProgramArguments")
    if (
        not isinstance(arguments, list)
        or not arguments
        or any(not isinstance(item, str) or not item for item in arguments)
    ):
        raise ValueError("Qwen plist ProgramArguments must be nonempty strings")
    module_pair = any(
        arguments[index : index + 2] == ["-m", "mtplx.server.openai"]
        for index in range(len(arguments) - 1)
    )
    direct_server = module_pair and any("Qwen3.6" in item for item in arguments)
    wrapper = len(arguments) == 1
    if direct_server:
        return path
    if wrapper:
        _validate_qwen_launcher(arguments[0], uid=uid)
        return path
    else:
        raise ValueError(
            "Qwen plist ProgramArguments must launch the Qwen server directly or "
            "through one validated wrapper"
        )


def _service_loaded(
    uid: int,
    *,
    run_command: CommandRunner,
) -> bool:
    result = run_command(("launchctl", "print", f"gui/{uid}/{QWEN_LABEL}"))
    if result.returncode == 0:
        return True
    if result.returncode == 113:
        return False
    detail = result.stderr.strip() or result.stdout.strip() or "no command output"
    raise RuntimeError(
        f"launchctl could not determine Qwen service state "
        f"(exit {result.returncode}): {detail}"
    )


def _qwen_processes(*, run_command: CommandRunner) -> tuple[int, ...]:
    result = run_command(("pgrep", "-f", QWEN_PROCESS_PATTERN))
    if result.returncode == 1 and not result.stdout.strip():
        return ()
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no command output"
        raise RuntimeError(
            f"pgrep could not determine Qwen process state "
            f"(exit {result.returncode}): {detail}"
        )
    values = result.stdout.splitlines()
    try:
        processes = tuple(int(value.strip()) for value in values if value.strip())
    except ValueError as exc:
        raise RuntimeError("pgrep returned an invalid Qwen process ID") from exc
    if not processes or any(pid <= 0 for pid in processes):
        raise RuntimeError("pgrep reported success without valid Qwen process IDs")
    if len(processes) != len(set(processes)):
        raise RuntimeError("pgrep returned duplicate Qwen process IDs")
    return tuple(sorted(processes))


def _observe_qwen(
    *,
    uid: int,
    api_url: str,
    run_command: CommandRunner,
    fetch_models: ModelFetcher,
) -> _QwenObservation:
    return _QwenObservation(
        loaded=_service_loaded(uid, run_command=run_command),
        models=fetch_models(api_url, 1.0),
        processes=_qwen_processes(run_command=run_command),
    )


def _is_stopped(observation: _QwenObservation) -> bool:
    return (
        not observation.loaded
        and observation.models is None
        and not observation.processes
    )


def _is_exact_loaded(
    observation: _QwenObservation,
    *,
    models: tuple[str, ...],
) -> bool:
    return (
        observation.loaded
        and observation.models == models
        and bool(observation.processes)
    )


def _capture_qwen_state(
    *,
    uid: int,
    api_url: str,
    run_command: CommandRunner,
    fetch_models: ModelFetcher,
) -> QwenState:
    observation = _observe_qwen(
        uid=uid,
        api_url=api_url,
        run_command=run_command,
        fetch_models=fetch_models,
    )
    if _is_exact_loaded(observation, models=EXPECTED_QWEN_MODELS):
        return QwenState(loaded=True, models=EXPECTED_QWEN_MODELS)
    if _is_stopped(observation):
        return QwenState(loaded=False, models=())
    raise RuntimeError(
        "Qwen service/API/process state is ambiguous or does not expose the exact "
        f"expected model list; observation={observation}"
    )


def _run_required(command: tuple[str, ...], *, run_command: CommandRunner) -> None:
    result = run_command(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no command output"
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}: {detail}"
        )


def _wait_for(
    predicate: Callable[[_QwenObservation], bool],
    *,
    description: str,
    uid: int,
    api_url: str,
    timeout_seconds: float,
    run_command: CommandRunner,
    fetch_models: ModelFetcher,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> _QwenObservation:
    deadline = monotonic() + timeout_seconds
    last: _QwenObservation | None = None
    while True:
        last = _observe_qwen(
            uid=uid,
            api_url=api_url,
            run_command=run_command,
            fetch_models=fetch_models,
        )
        if predicate(last):
            return last
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(_POLL_SECONDS, remaining))
    raise RuntimeError(f"timed out waiting for Qwen to {description}; last={last}")


def _restore_loaded_state(
    state: QwenState,
    *,
    stopped_confirmed: bool,
    uid: int,
    plist: Path,
    api_url: str,
    timeout_seconds: float,
    run_command: CommandRunner,
    fetch_models: ModelFetcher,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    observation = _observe_qwen(
        uid=uid,
        api_url=api_url,
        run_command=run_command,
        fetch_models=fetch_models,
    )
    if not stopped_confirmed and _is_exact_loaded(observation, models=state.models):
        return
    if not _is_stopped(observation):
        if stopped_confirmed:
            raise RuntimeError(
                "Qwen became active or ambiguous before controlled restoration; "
                f"observation={observation}"
            )
        observation = _wait_for(
            lambda item: (
                _is_stopped(item) or _is_exact_loaded(item, models=state.models)
            ),
            description="settle after a failed stop",
            uid=uid,
            api_url=api_url,
            timeout_seconds=timeout_seconds,
            run_command=run_command,
            fetch_models=fetch_models,
            monotonic=monotonic,
            sleep=sleep,
        )
        if _is_exact_loaded(observation, models=state.models):
            return
    _run_required(
        ("launchctl", "bootstrap", f"gui/{uid}", str(plist)),
        run_command=run_command,
    )
    _wait_for(
        lambda item: _is_exact_loaded(item, models=state.models),
        description=f"restore its exact model list {state.models}",
        uid=uid,
        api_url=api_url,
        timeout_seconds=timeout_seconds,
        run_command=run_command,
        fetch_models=fetch_models,
        monotonic=monotonic,
        sleep=sleep,
    )


@contextmanager
def qwen_stopped_for_mlx(
    *,
    plist: Path,
    api_url: str = "http://127.0.0.1:8080/v1/models",
    timeout_seconds: float = 180.0,
    _run_command: CommandRunner = _run_command,
    _fetch_models: ModelFetcher = _fetch_models,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
    _getuid: Callable[[], int] = os.getuid,
) -> Iterator[QwenState]:
    """Stop Qwen for one MLX window and restore its exact captured state."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    uid = _getuid()
    if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
        raise ValueError("current user ID is invalid")
    validated_plist = _read_validated_plist(Path(plist), uid=uid)
    state = _capture_qwen_state(
        uid=uid,
        api_url=api_url,
        run_command=_run_command,
        fetch_models=_fetch_models,
    )
    stopped_confirmed = not state.loaded
    try:
        if state.loaded:
            _run_required(
                (
                    "launchctl",
                    "bootout",
                    f"gui/{uid}",
                    str(validated_plist),
                ),
                run_command=_run_command,
            )
            _wait_for(
                _is_stopped,
                description="fully stop",
                uid=uid,
                api_url=api_url,
                timeout_seconds=float(timeout_seconds),
                run_command=_run_command,
                fetch_models=_fetch_models,
                monotonic=_monotonic,
                sleep=_sleep,
            )
            stopped_confirmed = True
        yield state
    finally:
        if state.loaded:
            _restore_loaded_state(
                state,
                stopped_confirmed=stopped_confirmed,
                uid=uid,
                plist=validated_plist,
                api_url=api_url,
                timeout_seconds=float(timeout_seconds),
                run_command=_run_command,
                fetch_models=_fetch_models,
                monotonic=_monotonic,
                sleep=_sleep,
            )
        else:
            final = _observe_qwen(
                uid=uid,
                api_url=api_url,
                run_command=_run_command,
                fetch_models=_fetch_models,
            )
            if not _is_stopped(final):
                raise RuntimeError(
                    "Qwen became active during an initially unloaded MLX window; "
                    f"observation={final}"
                )
