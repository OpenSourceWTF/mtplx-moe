#!/usr/bin/env python3
"""Preflight, extract, or verify the external GLM-5.2 layer-78 MTP artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mtplx.glm52_mtp_artifact import (
    ArtifactError,
    Glm52MtpArtifactConfig,
    extract_glm52_mtp_layer78,
    preflight_glm52_mtp_layer78,
    verify_glm52_mtp_layer78,
)


def _add_build_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="authenticate sources and print the bounded extraction plan"
    )
    _add_build_paths(preflight)

    extract = subparsers.add_parser(
        "extract", help="copy exact tensor bits and atomically publish the artifact"
    )
    _add_build_paths(extract)

    verify = subparsers.add_parser(
        "verify", help="verify the authenticated artifact and pinned sibling source"
    )
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument(
        "--deep",
        action="store_true",
        default=True,
        help="perform the required authenticated sibling-source verification",
    )
    return parser


def _config(args: argparse.Namespace) -> Glm52MtpArtifactConfig:
    return Glm52MtpArtifactConfig(
        source_root=args.source_root,
        output_root=args.output_root,
        producer_root=args.producer_root,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            plan = preflight_glm52_mtp_layer78(_config(args))
            print(
                json.dumps(
                    {
                        "tensor_count": plan.tensor_count,
                        "payload_bytes": plan.payload_bytes,
                        "shard_distribution": plan.shard_distribution,
                        "producer_commit": plan.producer_commit,
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "extract":
            print(extract_glm52_mtp_layer78(_config(args)))
        else:
            manifest = verify_glm52_mtp_layer78(args.output_root, deep=args.deep)
            print(json.dumps(manifest, sort_keys=True))
    except ArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
