#!/usr/bin/env python3
"""Write one strict GLM expert-Q2 campaign summary outside the repository."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Sequence


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.glm52_q2_campaign import (  # noqa: E402
    summarize_glm52_q2_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize paired GLM Q4/Q2 benchmark evidence."
    )
    parser.add_argument(
        "--resource",
        nargs="+",
        type=Path,
        required=True,
        help="Four resource payloads in physical ABBA order.",
    )
    parser.add_argument(
        "--headline",
        nargs="+",
        type=Path,
        required=True,
        help="Six headline payloads in physical paired order.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def validate_external_output_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output == _ROOT or output.is_relative_to(_ROOT):
        raise ValueError("--output-json must remain outside the Git worktree")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite campaign evidence: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    return output


def _load_payloads(paths: Sequence[Path]) -> list[dict]:
    payloads = []
    for path in paths:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"benchmark payload must be a JSON object: {path}")
        payloads.append(payload)
    return payloads


def _write_json_exclusive(path: Path, payload: dict) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = validate_external_output_path(args.output_json)
        resource = _load_payloads(args.resource)
        headline = _load_payloads(args.headline)
        summary = summarize_glm52_q2_campaign(resource, headline)
        _write_json_exclusive(output, summary)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"campaign summary failed: {exc}", file=sys.stderr)
        return 1
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
