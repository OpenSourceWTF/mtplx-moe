"""Tests for the promoted guarded-window harness."""

from __future__ import annotations

import pytest

from mtplx.benchmarks import window


def test_run_lock_token_matches_the_guarded_script_name() -> None:
    """The run-lock is keyed on a process name, so the token must stay exact.

    ~24 call sites pgrep for this string. If the benchmark script is renamed
    or moved without updating the constant, every guard silently passes and
    two concurrent ~95 GB runs can kernel-panic the box (2026-07-17).
    """

    assert window.GPU_RUN_LOCK_TOKEN == "benchmark_streamed_generation"


def test_gpu_run_lock_clear_reports_clear_when_pgrep_finds_nothing() -> None:
    # A token that cannot match any real process must read as clear.
    assert window.gpu_run_lock_clear("mtplx-token-that-matches-nothing-xyzzy")


def test_exclusive_mlx_lock_is_exported_publicly() -> None:
    from mtplx.qwen_guard import _exclusive_mlx_lock

    assert window.exclusive_mlx_lock is _exclusive_mlx_lock


def test_guarded_qwen_retries_entry_then_yields_state(monkeypatch) -> None:
    monkeypatch.setattr(window, "qwen_ready", lambda *a, **k: True)
    monkeypatch.setattr(window.time, "sleep", lambda _s: None)
    attempts: list[int] = []

    class Guard:
        def __init__(self, ok: bool) -> None:
            self.ok = ok
            self.exited = False

        def __enter__(self):
            attempts.append(1)
            if not self.ok:
                raise RuntimeError("ambiguous")
            return "captured"

        def __exit__(self, *exc):
            self.exited = True
            return False

    made: list[Guard] = []

    def factory() -> Guard:
        guard = Guard(ok=len(made) >= 2)
        made.append(guard)
        return guard

    with window.guarded_qwen(factory, backoff_s=0.0) as state:
        assert state == "captured"

    assert len(attempts) == 3, "entry should retry until the guard settles"
    assert made[-1].exited, "teardown must run on the successful guard"


def test_guarded_qwen_restores_qwen_when_the_body_raises(monkeypatch) -> None:
    """Teardown must run on every exit path, including exceptions."""

    monkeypatch.setattr(window, "qwen_ready", lambda *a, **k: True)
    exited: list[bool] = []

    class Guard:
        def __enter__(self):
            return "captured"

        def __exit__(self, *exc):
            exited.append(True)
            return False

    with pytest.raises(ValueError):
        with window.guarded_qwen(lambda: Guard()):
            raise ValueError("body blew up")

    assert exited == [True]


def test_guarded_qwen_raises_when_the_guard_never_settles(monkeypatch) -> None:
    monkeypatch.setattr(window, "qwen_ready", lambda *a, **k: False)
    monkeypatch.setattr(window.time, "sleep", lambda _s: None)

    class Guard:
        def __enter__(self):
            raise RuntimeError("ambiguous")

        def __exit__(self, *exc):
            return False

    with pytest.raises(RuntimeError, match="never settled after 2 attempts"):
        with window.guarded_qwen(lambda: Guard(), attempts=2, backoff_s=0.0):
            pass  # pragma: no cover - body must never run
