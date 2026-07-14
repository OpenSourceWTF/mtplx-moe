"""Fail-closed Qwen stop/restore guard for explicit MLX execution windows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
import time
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
_FETCH_MODELS_HELPER = """
import sys
import urllib.request

response = urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2]))
payload = response.read(int(sys.argv[3]) + 1)
response.close()
sys.stdout.buffer.write(payload)
"""


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


@dataclass(frozen=True)
class _ValidatedPlist:
    path: Path
    value: dict[str, object]
    payload: bytes
    wrapper_payload: bytes | None


@dataclass(frozen=True)
class _PlistSnapshot:
    uid: int
    path: Path
    directory_fd: int
    parent_fd: int
    directory_name: str
    directory_identity: tuple[int, int]
    plist_identity: tuple[int, int]
    plist_payload: bytes
    wrapper_directory_fd: int | None
    wrapper_directory_name: str | None
    wrapper_directory_identity: tuple[int, int] | None
    wrapper_name: str | None
    wrapper_identity: tuple[int, int] | None
    wrapper_payload: bytes | None


CommandRunner = Callable[[tuple[str, ...], float], CommandResult]
ModelFetcher = Callable[[str, float], tuple[str, ...] | None]


def _run_command(command: tuple[str, ...], timeout: float) -> CommandResult:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out after {timeout:.6f}s: {command!r}"
        ) from exc
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _fetch_models(api_url: str, timeout: float) -> tuple[str, ...] | None:
    deadline = time.monotonic() + timeout
    try:
        result = subprocess.run(
            (
                sys.executable,
                "-c",
                _FETCH_MODELS_HELPER,
                api_url,
                str(timeout),
                str(_MAX_API_BYTES),
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=max(0.0, deadline - time.monotonic()),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or time.monotonic() >= deadline:
        return None
    payload = result.stdout
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


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_without_symlink_ancestors(path: Path, flags: int) -> int:
    if not path.is_absolute() or len(path.parts) < 2:
        raise ValueError("validated path must be an absolute non-root path")
    components = path.parts[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("validated path must not contain relative components")
    directory_fd = os.open("/", _directory_flags())
    try:
        for component in components[:-1]:
            next_fd = os.open(
                component,
                _directory_flags(),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            components[-1],
            flags | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(
            f"validated path has a symlink ancestor or unsafe component: {exc}"
        ) from exc
    finally:
        os.close(directory_fd)


def _read_owned_file(
    path: Path,
    *,
    uid: int,
    label: str,
    max_bytes: int,
    require_executable: bool,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = _open_without_symlink_ancestors(path, flags)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} must be a no-follow file: {exc}") from exc
    try:
        status_before = os.fstat(fd)
        if not stat.S_ISREG(status_before.st_mode) or status_before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if status_before.st_uid != uid:
            raise ValueError(f"{label} must be owned by the current user")
        if status_before.st_mode & 0o022:
            raise ValueError(f"{label} must not be group- or world-writable")
        if require_executable and not status_before.st_mode & stat.S_IXUSR:
            raise ValueError(f"{label} must be executable by its owner")
        if status_before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds its size bound")
        chunks: list[bytes] = []
        remaining = status_before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        status_after = os.fstat(fd)
        if (
            status_after.st_dev != status_before.st_dev
            or status_after.st_ino != status_before.st_ino
            or status_after.st_size != status_before.st_size
            or status_after.st_mtime_ns != status_before.st_mtime_ns
            or status_after.st_ctime_ns != status_before.st_ctime_ns
        ):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _validate_qwen_launcher(path_value: str, *, uid: int) -> bytes:
    path = Path(path_value)
    if not path.is_absolute() or "qwen" not in path.name.lower():
        raise ValueError("Qwen plist wrapper must be an absolute Qwen-named executable")
    return _read_owned_file(
        path,
        uid=uid,
        label="Qwen plist wrapper",
        max_bytes=_MAX_PLIST_BYTES,
        require_executable=True,
    )


def _load_validated_plist(plist: Path, *, uid: int) -> _ValidatedPlist:
    path = plist.expanduser()
    if not path.is_absolute():
        raise ValueError("Qwen plist path must be absolute")
    if path.name != f"{QWEN_LABEL}.plist":
        raise ValueError(f"Qwen plist must be named {QWEN_LABEL}.plist")
    payload = _read_owned_file(
        path,
        uid=uid,
        label="Qwen plist",
        max_bytes=_MAX_PLIST_BYTES,
        require_executable=False,
    )
    try:
        value = plistlib.loads(payload)
    except plistlib.InvalidFileException as exc:
        raise ValueError(f"Qwen plist is malformed: {exc}") from exc
    if not isinstance(value, dict) or value.get("Label") != QWEN_LABEL:
        raise ValueError(f"Qwen plist Label must be exactly {QWEN_LABEL}")
    if "Program" in value:
        raise ValueError(
            "Qwen plist Program is unsupported; use validated ProgramArguments only"
        )
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
        return _ValidatedPlist(path, value, payload, None)
    if wrapper:
        wrapper_payload = _validate_qwen_launcher(arguments[0], uid=uid)
        return _ValidatedPlist(path, value, payload, wrapper_payload)
    raise ValueError(
        "Qwen plist ProgramArguments must launch the Qwen server directly or "
        "through one validated wrapper"
    )


def _read_validated_plist(plist: Path, *, uid: int) -> Path:
    return _load_validated_plist(plist, uid=uid).path


def _write_snapshot_file(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
) -> tuple[int, int]:
    fd = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError("short write while creating Qwen plist snapshot")
            written += count
        os.fsync(fd)
        status = os.fstat(fd)
        return status.st_dev, status.st_ino
    finally:
        os.close(fd)


def _validate_bound_file(
    directory_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
    expected_payload: bytes,
    uid: int,
    require_executable: bool,
) -> None:
    fd = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        status = os.fstat(fd)
        identity = (status.st_dev, status.st_ino)
        payload = os.read(fd, len(expected_payload) + 1)
    finally:
        os.close(fd)
    if (
        identity != expected_identity
        or payload != expected_payload
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_uid != uid
        or bool(status.st_mode & 0o022)
        or (require_executable and not status.st_mode & stat.S_IXUSR)
    ):
        raise RuntimeError("private Qwen plist snapshot changed before launchctl")


def _open_wrapper_cache(
    parent: Path,
    *,
    parent_fd: int,
    uid: int,
) -> tuple[Path, int, tuple[int, int]]:
    name = ".mtplx-qwen-guard"
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        directory_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(
            f"Qwen wrapper cache must be a no-follow directory: {exc}"
        ) from exc
    status = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != uid
        or bool(status.st_mode & 0o077)
    ):
        os.close(directory_fd)
        raise ValueError(
            "Qwen wrapper cache must be a private directory owned by the current user"
        )
    return parent / name, directory_fd, (status.st_dev, status.st_ino)


def _cache_wrapper(
    directory_fd: int,
    payload: bytes,
    *,
    uid: int,
) -> tuple[str, tuple[int, int]]:
    name = f"qwen-wrapper-{hashlib.sha256(payload).hexdigest()}"
    try:
        identity = _write_snapshot_file(
            directory_fd,
            name,
            payload,
            mode=0o500,
        )
    except FileExistsError:
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            status = os.fstat(fd)
            identity = (status.st_dev, status.st_ino)
        finally:
            os.close(fd)
        _validate_bound_file(
            directory_fd,
            name,
            expected_identity=identity,
            expected_payload=payload,
            uid=uid,
            require_executable=True,
        )
    os.fsync(directory_fd)
    return name, identity


def _validate_snapshot(snapshot: _PlistSnapshot) -> None:
    named = os.stat(
        snapshot.directory_name,
        dir_fd=snapshot.parent_fd,
        follow_symlinks=False,
    )
    if (named.st_dev, named.st_ino) != snapshot.directory_identity:
        raise RuntimeError("private Qwen snapshot directory identity changed")
    _validate_bound_file(
        snapshot.directory_fd,
        f"{QWEN_LABEL}.plist",
        expected_identity=snapshot.plist_identity,
        expected_payload=snapshot.plist_payload,
        uid=snapshot.uid,
        require_executable=False,
    )
    if snapshot.wrapper_payload is not None:
        assert snapshot.wrapper_directory_fd is not None
        assert snapshot.wrapper_directory_name is not None
        assert snapshot.wrapper_directory_identity is not None
        assert snapshot.wrapper_name is not None
        assert snapshot.wrapper_identity is not None
        named_wrapper_directory = os.stat(
            snapshot.wrapper_directory_name,
            dir_fd=snapshot.parent_fd,
            follow_symlinks=False,
        )
        if (
            named_wrapper_directory.st_dev,
            named_wrapper_directory.st_ino,
        ) != snapshot.wrapper_directory_identity:
            raise RuntimeError("private Qwen wrapper cache identity changed")
        _validate_bound_file(
            snapshot.wrapper_directory_fd,
            snapshot.wrapper_name,
            expected_identity=snapshot.wrapper_identity,
            expected_payload=snapshot.wrapper_payload,
            uid=snapshot.uid,
            require_executable=True,
        )


@contextmanager
def _validated_plist_snapshot(plist: Path, *, uid: int) -> Iterator[_PlistSnapshot]:
    source = _load_validated_plist(plist, uid=uid)
    parent = Path.home()
    parent_fd = _open_without_symlink_ancestors(parent, _directory_flags())
    directory: Path | None = None
    directory_fd: int | None = None
    directory_identity: tuple[int, int] | None = None
    wrapper_directory_fd: int | None = None
    try:
        directory = Path(tempfile.mkdtemp(prefix=".mtplx-qwen-guard-", dir=parent))
        directory_fd = os.open(
            directory.name,
            _directory_flags(),
            dir_fd=parent_fd,
        )
        os.fchmod(directory_fd, 0o700)
        directory_status = os.fstat(directory_fd)
        directory_identity = (directory_status.st_dev, directory_status.st_ino)
        wrapper_identity: tuple[int, int] | None = None
        wrapper_payload = source.wrapper_payload
        snapshot_value = dict(source.value)
        if wrapper_payload is not None:
            (
                wrapper_directory,
                wrapper_directory_fd,
                wrapper_directory_identity,
            ) = _open_wrapper_cache(
                parent,
                parent_fd=parent_fd,
                uid=uid,
            )
            wrapper_name, wrapper_identity = _cache_wrapper(
                wrapper_directory_fd,
                wrapper_payload,
                uid=uid,
            )
            snapshot_value["ProgramArguments"] = [str(wrapper_directory / wrapper_name)]
            plist_payload = plistlib.dumps(snapshot_value)
        else:
            wrapper_directory_identity = None
            wrapper_name = None
            plist_payload = source.payload
        plist_identity = _write_snapshot_file(
            directory_fd,
            f"{QWEN_LABEL}.plist",
            plist_payload,
            mode=0o600,
        )
        os.fsync(directory_fd)
        snapshot = _PlistSnapshot(
            uid=uid,
            path=directory / f"{QWEN_LABEL}.plist",
            directory_fd=directory_fd,
            parent_fd=parent_fd,
            directory_name=directory.name,
            directory_identity=directory_identity,
            plist_identity=plist_identity,
            plist_payload=plist_payload,
            wrapper_directory_fd=wrapper_directory_fd,
            wrapper_directory_name=(
                wrapper_directory.name if wrapper_payload is not None else None
            ),
            wrapper_directory_identity=wrapper_directory_identity,
            wrapper_name=wrapper_name,
            wrapper_identity=wrapper_identity,
            wrapper_payload=wrapper_payload,
        )
        _validate_snapshot(snapshot)
        yield snapshot
    finally:
        try:
            if wrapper_directory_fd is not None:
                os.close(wrapper_directory_fd)
                wrapper_directory_fd = None
            if directory_fd is not None:
                for name in (f"{QWEN_LABEL}.plist", "qwen-wrapper"):
                    try:
                        os.unlink(name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
                os.fsync(directory_fd)
                remaining = os.listdir(directory_fd)
                os.close(directory_fd)
                directory_fd = None
                if directory is not None and directory_identity is not None:
                    try:
                        named = os.stat(
                            directory.name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        if (
                            not remaining
                            and (
                                named.st_dev,
                                named.st_ino,
                            )
                            == directory_identity
                        ):
                            os.rmdir(directory.name, dir_fd=parent_fd)
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
            os.close(parent_fd)


def _remaining_seconds(
    deadline: float,
    monotonic: Callable[[], float],
    *,
    description: str,
) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise RuntimeError(f"timed out while attempting to {description}")
    return remaining


def _service_loaded(
    uid: int,
    *,
    run_command: CommandRunner,
    deadline: float,
    monotonic: Callable[[], float],
) -> bool:
    result = run_command(
        ("launchctl", "print", f"gui/{uid}/{QWEN_LABEL}"),
        _remaining_seconds(deadline, monotonic, description="observe launchctl"),
    )
    if result.returncode == 0:
        return True
    if result.returncode == 113:
        return False
    detail = result.stderr.strip() or result.stdout.strip() or "no command output"
    raise RuntimeError(
        f"launchctl could not determine Qwen service state "
        f"(exit {result.returncode}): {detail}"
    )


def _qwen_processes(
    *,
    run_command: CommandRunner,
    deadline: float,
    monotonic: Callable[[], float],
) -> tuple[int, ...]:
    result = run_command(
        ("pgrep", "-f", QWEN_PROCESS_PATTERN),
        _remaining_seconds(deadline, monotonic, description="observe Qwen processes"),
    )
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
    deadline: float,
    monotonic: Callable[[], float],
) -> _QwenObservation:
    loaded = _service_loaded(
        uid,
        run_command=run_command,
        deadline=deadline,
        monotonic=monotonic,
    )
    api_timeout = min(
        1.0,
        _remaining_seconds(deadline, monotonic, description="observe Qwen API"),
    )
    models = fetch_models(api_url, api_timeout)
    return _QwenObservation(
        loaded=loaded,
        models=models,
        processes=_qwen_processes(
            run_command=run_command,
            deadline=deadline,
            monotonic=monotonic,
        ),
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
    deadline: float,
    monotonic: Callable[[], float],
) -> QwenState:
    observation = _observe_qwen(
        uid=uid,
        api_url=api_url,
        run_command=run_command,
        fetch_models=fetch_models,
        deadline=deadline,
        monotonic=monotonic,
    )
    if _is_exact_loaded(observation, models=EXPECTED_QWEN_MODELS):
        return QwenState(loaded=True, models=EXPECTED_QWEN_MODELS)
    if _is_stopped(observation):
        return QwenState(loaded=False, models=())
    raise RuntimeError(
        "Qwen service/API/process state is ambiguous or does not expose the exact "
        f"expected model list; observation={observation}"
    )


def _run_required(
    command: tuple[str, ...],
    *,
    run_command: CommandRunner,
    deadline: float,
    monotonic: Callable[[], float],
) -> None:
    result = run_command(
        command,
        _remaining_seconds(deadline, monotonic, description=f"run {command[1]}"),
    )
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
    deadline: float,
    run_command: CommandRunner,
    fetch_models: ModelFetcher,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> _QwenObservation:
    last: _QwenObservation | None = None
    while True:
        last = _observe_qwen(
            uid=uid,
            api_url=api_url,
            run_command=run_command,
            fetch_models=fetch_models,
            deadline=deadline,
            monotonic=monotonic,
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
    snapshot: _PlistSnapshot,
    api_url: str,
    deadline: float,
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
        deadline=deadline,
        monotonic=monotonic,
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
            deadline=deadline,
            run_command=run_command,
            fetch_models=fetch_models,
            monotonic=monotonic,
            sleep=sleep,
        )
        if _is_exact_loaded(observation, models=state.models):
            return
    _validate_snapshot(snapshot)
    _run_required(
        ("launchctl", "bootstrap", f"gui/{uid}", str(snapshot.path)),
        run_command=run_command,
        deadline=deadline,
        monotonic=monotonic,
    )
    _wait_for(
        lambda item: _is_exact_loaded(item, models=state.models),
        description=f"restore its exact model list {state.models}",
        uid=uid,
        api_url=api_url,
        deadline=deadline,
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
    with _validated_plist_snapshot(Path(plist), uid=uid) as snapshot:
        entry_deadline = _monotonic() + float(timeout_seconds)
        state = _capture_qwen_state(
            uid=uid,
            api_url=api_url,
            run_command=_run_command,
            fetch_models=_fetch_models,
            deadline=entry_deadline,
            monotonic=_monotonic,
        )
        stopped_confirmed = not state.loaded
        try:
            if state.loaded:
                _validate_snapshot(snapshot)
                _run_required(
                    (
                        "launchctl",
                        "bootout",
                        f"gui/{uid}",
                        str(snapshot.path),
                    ),
                    run_command=_run_command,
                    deadline=entry_deadline,
                    monotonic=_monotonic,
                )
                _wait_for(
                    _is_stopped,
                    description="fully stop",
                    uid=uid,
                    api_url=api_url,
                    deadline=entry_deadline,
                    run_command=_run_command,
                    fetch_models=_fetch_models,
                    monotonic=_monotonic,
                    sleep=_sleep,
                )
                stopped_confirmed = True
            yield state
        finally:
            restore_deadline = _monotonic() + float(timeout_seconds)
            if state.loaded:
                _restore_loaded_state(
                    state,
                    stopped_confirmed=stopped_confirmed,
                    uid=uid,
                    snapshot=snapshot,
                    api_url=api_url,
                    deadline=restore_deadline,
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
                    deadline=restore_deadline,
                    monotonic=_monotonic,
                )
                if not _is_stopped(final):
                    if not final.loaded:
                        raise RuntimeError(
                            "Qwen became ambiguous during an initially unloaded MLX "
                            f"window and cannot be restored safely; observation={final}"
                        )
                    _validate_snapshot(snapshot)
                    _run_required(
                        (
                            "launchctl",
                            "bootout",
                            f"gui/{uid}",
                            str(snapshot.path),
                        ),
                        run_command=_run_command,
                        deadline=restore_deadline,
                        monotonic=_monotonic,
                    )
                    _wait_for(
                        _is_stopped,
                        description="restore the initially unloaded state",
                        uid=uid,
                        api_url=api_url,
                        deadline=restore_deadline,
                        run_command=_run_command,
                        fetch_models=_fetch_models,
                        monotonic=_monotonic,
                        sleep=_sleep,
                    )
