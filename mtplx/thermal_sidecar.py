"""Standalone fan-restore sidecar for ``--max`` sessions.

Spawned as a detached child by ``cmd_serve_public`` when MAX mode pins
the fans. The sidecar:

  1. Detaches itself from the parent's controlling terminal via ``setsid``
     so closing the terminal window or sending SIGHUP to the process
     group does NOT kill it too.
  2. Polls the parent PID every ``poll_seconds``.
  3. The moment the parent is gone (any cause: clean exit, SIGINT,
     SIGTERM, SIGHUP, SIGKILL, OOM, terminal closed, kernel panic
     followed by reboot — well, except that last one), it runs
     ``sudo -n <thermalforge-path> auto`` and exits.

This is the only piece of the crash-safety machinery that handles
SIGKILL of the parent. The signal-handler / atexit path covers
recoverable signals; the marker-file path covers "user runs MTPLX
again later"; the sidecar covers "user closes the terminal and walks
away".

The sidecar runs as the unprivileged user and relies on the
``/etc/sudoers.d/mtplx-thermalforge`` NOPASSWD rule installed by
``mtplx max --install`` to do its restore without a password.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from mtplx.thermal import _daemon_socket_send


def _detach_from_terminal() -> None:
    """Best-effort detach from the controlling terminal.

    A second fork would be the textbook double-fork daemonization, but
    ``setsid`` + closing stdio is enough for our use case: we only need
    to survive the parent's death, not to live forever as a system
    daemon.
    """

    try:
        os.setsid()
    except OSError:
        pass

    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass
    finally:
        if devnull > 2:
            try:
                os.close(devnull)
            except OSError:
                pass


def _parent_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The pid exists but isn't ours. Treat as alive — better to
        # leave fans pinned for an extra poll cycle than to restore
        # while the real owner is still running.
        return True
    except OSError:
        return False


def _restore_fans(binary: str) -> int:
    """Restore Apple-auto fans and return an exit code.

    Prefer the ThermalForge daemon socket: it resets fans as root without sudo
    and, unlike ``thermalforge auto``, never quits the menu bar app. Fall back
    to ``sudo -n <binary> auto`` when no daemon is reachable. ``sudo -n`` never
    prompts, so it succeeds immediately (NOPASSWD active) or fails fast — either
    way the sidecar exits, having made its attempt.
    """

    reset = _daemon_socket_send("auto")
    if reset is not None and reset["ok"]:
        return 0

    try:
        proc = subprocess.run(
            ["sudo", "-n", binary, "auto"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return proc.returncode
    except Exception:
        return 1


def _clear_marker(marker_path: str | None) -> None:
    if not marker_path:
        return
    try:
        if os.path.exists(marker_path):
            os.unlink(marker_path)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--binary", required=True, help="Path to thermalforge CLI")
    parser.add_argument("--marker", default=None, help="Marker file to delete after restore")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-lifetime-seconds", type=float, default=24 * 3600.0,
                        help="Hard ceiling on sidecar lifetime; ensures we eventually die even on bugs")
    args = parser.parse_args(argv)

    _detach_from_terminal()
    started_at = time.time()

    while True:
        if not _parent_alive(args.parent_pid):
            rc = _restore_fans(args.binary)
            if rc == 0:
                _clear_marker(args.marker)
            return rc
        if (time.time() - started_at) > args.max_lifetime_seconds:
            return 0
        try:
            time.sleep(max(0.5, float(args.poll_seconds)))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
