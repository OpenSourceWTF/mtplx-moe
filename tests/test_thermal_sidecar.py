"""Tests for the detached fan-restore sidecar.

The sidecar is the only piece of the crash-safety machinery that handles
SIGKILL and terminal-close (signal handlers can't catch those, and the
marker-file recovery only fires on the *next* MTPLX invocation). We
exercise it without spawning real subprocesses by importing the module
and driving its building blocks directly.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from mtplx import thermal_sidecar


@pytest.fixture(autouse=True)
def _default_no_daemon_socket(monkeypatch):
    """Default every test to "no ThermalForge daemon socket" so the suite never
    touches a real daemon on the dev machine. The socket-path test opts in."""
    monkeypatch.setattr(thermal_sidecar, "_daemon_socket_send", lambda *a, **k: None)


def test_parent_alive_returns_true_for_self():
    assert thermal_sidecar._parent_alive(os.getpid()) is True


def test_restore_fans_prefers_daemon_socket(monkeypatch):
    """When the daemon socket answers, restore through it and never shell out
    to sudo (which needs a password and would run the app-killing CLI)."""
    monkeypatch.setattr(
        thermal_sidecar,
        "_daemon_socket_send",
        lambda *a, **k: {"ok": True, "response": "ok", "command": ["<sock>", "auto"]},
    )

    def boom(*a, **k):
        raise AssertionError("must not shell out when the daemon socket handles it")

    monkeypatch.setattr(subprocess, "run", boom)
    assert thermal_sidecar._restore_fans("/path/to/thermalforge") == 0


def test_parent_alive_returns_false_for_dead_pid():
    # 999_999_999 is far above kernel.pid_max on macOS — definitely free.
    assert thermal_sidecar._parent_alive(999_999_999) is False


def test_parent_alive_returns_false_for_zero_or_negative():
    assert thermal_sidecar._parent_alive(0) is False
    assert thermal_sidecar._parent_alive(-1) is False


def test_clear_marker_no_op_when_missing(tmp_path):
    marker = tmp_path / "missing.json"
    thermal_sidecar._clear_marker(str(marker))  # must not raise
    assert not marker.exists()


def test_clear_marker_deletes_existing_file(tmp_path):
    marker = tmp_path / "active.json"
    marker.write_text("{}")
    thermal_sidecar._clear_marker(str(marker))
    assert not marker.exists()


def test_clear_marker_handles_none():
    thermal_sidecar._clear_marker(None)  # must not raise


def test_restore_fans_runs_sudo_thermalforge_auto(monkeypatch):
    """Sidecar must invoke ``sudo -n <binary> auto`` (passwordless),
    never a plain ``thermalforge auto`` (which fails with "Run with
    sudo"), and never an interactive ``sudo`` (which would prompt
    against /dev/null and hang)."""

    captured: list[list[str]] = []

    class _FakeProc:
        def __init__(self, returncode: int = 0) -> None:
            self.returncode = returncode

    def fake_run(cmd, *, stdin=None, stdout=None, stderr=None, timeout=None):
        captured.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar._restore_fans("/path/to/thermalforge")

    assert rc == 0
    assert captured == [["sudo", "-n", "/path/to/thermalforge", "auto"]]


def test_restore_fans_swallows_subprocess_exceptions(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(subprocess, "run", boom)
    rc = thermal_sidecar._restore_fans("/path/to/thermalforge")
    assert rc == 1


def test_main_exits_immediately_when_parent_already_dead(monkeypatch, tmp_path):
    """If the parent is gone before the sidecar's first poll, restore
    must run on iteration 1 and the sidecar must exit with 0."""

    marker = tmp_path / "active.json"
    marker.write_text("{}")
    captured: list[list[str]] = []

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: False)

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        class _P:
            returncode = 0
        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "1",
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--poll-seconds",
            "0.1",
        ]
    )

    assert rc == 0
    assert captured == [["sudo", "-n", "/path/to/thermalforge", "auto"]]
    assert not marker.exists()  # marker cleared


def test_main_keeps_marker_when_restore_command_fails(monkeypatch, tmp_path):
    marker = tmp_path / "active.json"
    marker.write_text("{}")

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: False)

    def fake_run(cmd, *args, **kwargs):
        class _P:
            returncode = 1
        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "1",
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--poll-seconds",
            "0.1",
        ]
    )

    assert rc == 1
    assert marker.exists()


def test_main_polls_until_parent_dies(monkeypatch, tmp_path):
    """Sidecar must keep polling while the parent is alive and only
    fire the restore once it's gone."""

    polls = {"n": 0}

    def fake_alive(pid: int) -> bool:
        polls["n"] += 1
        return polls["n"] < 3  # die after 2 alive polls

    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        class _P:
            returncode = 0
        return _P()

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", fake_alive)
    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "12345",
            "--binary",
            "/path/to/thermalforge",
            "--poll-seconds",
            "0.05",
        ]
    )

    assert rc == 0
    assert polls["n"] == 3
    assert len(captured) == 1


def test_main_respects_max_lifetime(monkeypatch):
    """Hard ceiling so a buggy sidecar can never live forever."""

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: True)

    started = time.time()
    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            str(os.getpid()),
            "--binary",
            "/path/to/thermalforge",
            "--poll-seconds",
            "0.1",
            "--max-lifetime-seconds",
            "0.3",
        ]
    )
    elapsed = time.time() - started
    assert rc == 0
    assert elapsed < 2.0, f"sidecar overran its lifetime ceiling ({elapsed:.2f}s)"


def test_install_max_lifecycle_hooks_spawns_sidecar(monkeypatch, tmp_path):
    """``install_max_lifecycle_hooks`` must call ``_spawn_thermal_sidecar``
    so the parent process gets crash-safety coverage on terminal close."""

    from mtplx import thermal

    monkeypatch.setattr(thermal, "MAX_MARKER_FILE", tmp_path / "max-active.json")
    spawned: list[bool] = []

    def fake_spawn():
        spawned.append(True)
        return None  # we don't care about the Popen return for this test

    monkeypatch.setattr(thermal, "_spawn_thermal_sidecar", fake_spawn)
    monkeypatch.setattr(
        "mtplx.thermal.set_thermal_profile",
        lambda profile, **kw: {"ok": True},
    )
    monkeypatch.setattr("mtplx.thermal.signal.signal", lambda *a, **kw: None)
    monkeypatch.setattr("mtplx.thermal.atexit.register", lambda *a, **kw: None)

    cleanup = thermal.install_max_lifecycle_hooks()

    assert spawned == [True], "sidecar was not spawned by install_max_lifecycle_hooks"
    cleanup()
