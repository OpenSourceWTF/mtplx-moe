#!/usr/bin/env python3
"""Run one explicit command while Qwen is stopped, then restore Qwen exactly."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from mtplx.qwen_guard import qwen_stopped_for_mlx


GuardFactory = Callable[..., Any]
PopenFactory = Callable[[tuple[str, ...]], Any]


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plist", required=True, type=Path)
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080/v1/models",
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=180.0)
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if "--" not in values:
        parser.parse_args(values)
        parser.error("child command must appear after an explicit -- delimiter")
    delimiter = values.index("--")
    args = parser.parse_args(values[:delimiter])
    command = tuple(values[delimiter + 1 :])
    if not command:
        parser.error("child command after -- must not be empty")
    args.command = command
    return args


class _SignalRelay:
    def __init__(self) -> None:
        self.received: int | None = None
        self.process: Any | None = None
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: FrameType | None) -> None:
        if self.received is None:
            self.received = signum
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass

    def __enter__(self) -> _SignalRelay:
        supported = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            supported.append(signal.SIGHUP)
        for signum in supported:
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def run_child(self, command: tuple[str, ...], *, popen: PopenFactory) -> int:
        if self.received is not None:
            return 128 + self.received
        process = popen(command)
        self.process = process
        try:
            if self.received is not None and process.poll() is None:
                try:
                    process.send_signal(self.received)
                except ProcessLookupError:
                    pass
            returncode = process.wait()
        finally:
            self.process = None
        if self.received is not None:
            return 128 + self.received
        if returncode < 0:
            return 128 - returncode
        return returncode


def _run_guarded(
    args: argparse.Namespace,
    *,
    guard_factory: GuardFactory,
    popen: PopenFactory,
) -> int:
    child_exit = 1
    with _SignalRelay() as relay:
        with guard_factory(
            plist=args.plist,
            api_url=args.api_url,
            timeout_seconds=args.timeout_seconds,
        ):
            child_exit = relay.run_child(args.command, popen=popen)
        if relay.received is not None:
            return 128 + relay.received
    return child_exit


def main(
    argv: Sequence[str] | None = None,
    *,
    _guard_factory: GuardFactory = qwen_stopped_for_mlx,
    _popen: PopenFactory = subprocess.Popen,
) -> int:
    args = parse_cli_args(argv)
    try:
        return _run_guarded(
            args,
            guard_factory=_guard_factory,
            popen=_popen,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"run_with_qwen_stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
