"""Guarded benchmark-window helpers.

Every guarded perf run on this box needs the same three things before it may
touch the GPU:

1. no other benchmark already holding the device (the run-lock precheck),
2. the process-shared exclusive MLX lock,
3. the shared qwen server stopped and restored on every exit path.

Until now that logic lived in a scratchpad file on a tmpfs path
(``.../<session>/scratchpad/window_helpers.py``) which ~20 one-off drivers
imported by absolute path, and the run-lock precheck was re-derived by hand in
~24 places. A single tmpfs cleanup would have made the whole benchmark fleet
unrunnable, so it is promoted here verbatim.

``guarded_qwen`` fixes a real race seen 2026-07-17 23:08: when another lane's
window releases the lock it restores qwen, but qwen needs time to load its
model. A window that acquires the lock in that gap sees the service registered
with no model list, and the (correctly fail-closed) guard raises "Qwen
service/API/process state is ambiguous". Waiting for a settled state and
retrying ENTRY turns that transient into a short pause instead of a dead
window.

Only guard ENTRY is retried — never the window body, which must run exactly
once — and teardown always runs, so qwen is restored on every exit path.
"""

from __future__ import annotations

import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from ..qwen_guard import (
    DEFAULT_MLX_LOCK_PATH,
    _exclusive_mlx_lock,
)

__all__ = [
    "DEFAULT_MLX_LOCK_PATH",
    "GPU_RUN_LOCK_TOKEN",
    "exclusive_mlx_lock",
    "gpu_run_lock_clear",
    "guarded_qwen",
    "qwen_ready",
]

# The run-lock is advisory and keyed on a PROCESS NAME, not a file: ~24 call
# sites do `pgrep -f benchmark_streamed_generation`. Renaming or moving that
# script silently disables the kernel-panic guard everywhere, so the token is
# named here once and any move must update this constant in the same commit.
GPU_RUN_LOCK_TOKEN = "benchmark_streamed_generation"

# Public alias. The underlying context manager is private but has ~95 external
# importers, which is exactly the signal that it should not be private.
exclusive_mlx_lock = _exclusive_mlx_lock


def gpu_run_lock_clear(token: str = GPU_RUN_LOCK_TOKEN) -> bool:
    """True when no benchmark process is currently holding the GPU.

    ``pgrep`` exits 1 when nothing matches, which is the clear case.
    """

    probe = subprocess.run(["pgrep", "-f", token], capture_output=True)
    return probe.returncode != 0


def qwen_ready(timeout_s: float = 420.0, poll_s: float = 10.0) -> bool:
    """Block until qwen exposes a model id, or the timeout expires.

    True means qwen is serving. False means it never settled — callers may
    still proceed (the guard handles a cleanly-down qwen), but it is worth
    logging.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["curl", "-s", "-m", "5", "http://127.0.0.1:8080/v1/models"],
            capture_output=True,
            text=True,
        )
        if '"id"' in probe.stdout:
            return True
        time.sleep(poll_s)
    return False


@contextmanager
def guarded_qwen(guard_factory, *, attempts: int = 4, backoff_s: float = 30.0):
    """Enter the qwen guard with retries; yield its captured state.

    `guard_factory` must return a FRESH context manager per call (guard
    objects are single-use).
    """
    entered = None
    state = None
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        ready = qwen_ready()
        print(
            f"qwen readiness: {'serving' if ready else 'NOT settled'} "
            f"(guard entry attempt {attempt}/{attempts})",
            flush=True,
        )
        candidate = guard_factory()
        try:
            state = candidate.__enter__()
            entered = candidate
            break
        except RuntimeError as exc:
            last = exc
            print(f"guard entry failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(backoff_s)
    if entered is None:
        raise RuntimeError(
            f"qwen guard never settled after {attempts} attempts: {last}"
        )
    try:
        yield state
    except BaseException:
        if not entered.__exit__(*sys.exc_info()):
            raise
    else:
        entered.__exit__(None, None, None)
