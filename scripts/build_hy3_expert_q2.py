#!/usr/bin/env python3
"""Phase-isolated builder for the explicit local Hy3 expert-only Q2 artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _record(value: str) -> tuple[int, int]:
    try:
        layer_text, expert_text = value.split(":", 1)
        layer = int(layer_text)
        expert = int(expert_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("record must be LAYER:EXPERT") from exc
    if layer < 0 or expert < 0:
        raise argparse.ArgumentTypeError("record coordinates must be non-negative")
    return layer, expert


def _write_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _config(args: argparse.Namespace, *, pilot: bool = False):
    from mtplx.hy3_expert_q2 import ConversionConfig

    source = args.source.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if hasattr(args, "output")
        else source.with_name("hy3-expert-only-mlx-q2")
    )
    return ConversionConfig(
        source_root=source,
        source_manifest=args.source_manifest.expanduser().resolve(),
        source_provenance=args.source_provenance.expanduser().resolve(),
        output_root=output,
        pilot_report=(args.pilot_report.expanduser().resolve() if pilot else None),
    )


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.hy3_expert_q2 import preflight_hy3_expert_q2

    return preflight_hy3_expert_q2(
        _config(args),
        deep_source_hash=args.deep_source_hash,
    )


def _run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.hy3_expert_q2 import pilot_hy3_expert_q2

    report = pilot_hy3_expert_q2(_config(args), tuple(args.record))
    _write_json(args.output_json, report)
    return report


def _run_stage(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.hy3_expert_q2 import stage_hy3_expert_q2

    work_root = stage_hy3_expert_q2(_config(args))
    return {"staged": True, "work_root": os.fspath(work_root)}


def _run_convert(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.hy3_expert_q2 import convert_expert_records

    records = convert_expert_records(_config(args), resume=args.resume)
    return {
        "converted": True,
        "record_count": len(records),
        "output": os.fspath(args.output.expanduser().resolve()),
    }


def _run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.hy3_expert_q2 import finalize_hy3_expert_q2

    output = finalize_hy3_expert_q2(_config(args, pilot=True))
    return {"published": True, "output": os.fspath(output)}


def _run_verify(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.hy3_expert_q2 import verify_hy3_expert_q2

    report = verify_hy3_expert_q2(
        args.output.expanduser().resolve(),
        deep=args.deep,
    )
    if args.output_json is not None:
        _write_json(args.output_json, report)
    return report


def _source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify the explicit local Hy3 expert-only Q2 artifact."
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    preflight = subparsers.add_parser("preflight")
    _source_options(preflight)
    preflight.add_argument("--deep-source-hash", action="store_true")
    preflight.set_defaults(handler=_run_preflight)

    pilot = subparsers.add_parser("pilot")
    _source_options(pilot)
    pilot.add_argument("--record", action="append", type=_record, required=True)
    pilot.add_argument("--output-json", type=Path, required=True)
    pilot.set_defaults(handler=_run_pilot)

    stage = subparsers.add_parser("stage")
    _source_options(stage)
    stage.add_argument("output", type=Path)
    stage.set_defaults(handler=_run_stage)

    convert = subparsers.add_parser("convert")
    _source_options(convert)
    convert.add_argument("output", type=Path)
    convert.add_argument("--resume", action="store_true")
    convert.set_defaults(handler=_run_convert)

    finalize = subparsers.add_parser("finalize")
    _source_options(finalize)
    finalize.add_argument("output", type=Path)
    finalize.add_argument("--pilot-report", type=Path, required=True)
    finalize.set_defaults(handler=_run_finalize)

    verify = subparsers.add_parser("verify")
    verify.add_argument("output", type=Path)
    verify.add_argument("--deep", action="store_true")
    verify.add_argument("--output-json", type=Path)
    verify.set_defaults(handler=_run_verify)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handler: Callable[[argparse.Namespace], dict[str, Any]] = args.handler
    try:
        report = handler(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Hy3 expert-Q2 {args.phase} failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
