"""No-MLX argparse primitives shared by CLI command groups."""

from __future__ import annotations

import argparse
import sys

from mtplx.profiles import resolve_profile_name
from mtplx.runtime_options import normalize_paged_kv_quantization


def _profile_arg(value: str) -> str:
    try:
        return resolve_profile_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _comma_floats(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _kv_quant_arg(value: str) -> str:
    try:
        normalized = normalize_paged_kv_quantization(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert normalized is not None
    return normalized


def _explicit_cli_flags(raw_args: list[str]) -> set[str]:
    """Return long/short flag names explicitly typed on the CLI."""

    flags: set[str] = set()
    for token in raw_args:
        if not token.startswith("-") or token == "-" or token == "--":
            continue
        head = token.split("=", 1)[0]
        if head.startswith("--"):
            flags.add(head[2:])
        else:
            flags.add(head[1:])
    return flags


class _FlagRecordingArgumentParser(argparse.ArgumentParser):
    """Root parser that always records which flags were actually typed."""

    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        raw = list(sys.argv[1:]) if args is None else list(args)
        parsed = super().parse_args(raw, namespace)
        if not hasattr(parsed, "_cli_flags"):
            parsed._cli_flags = _explicit_cli_flags(raw)
        return parsed
