#!/usr/bin/env python3
"""Inspect or assemble the pinned Kimi K3 Q2_K-to-t158 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mtplx.kimi_k3_t158 import (
    KIMI_K3_OFFICIAL_REVISION,
    assemble_artifact,
    project_artifact,
)


def parse_layers(value: str) -> tuple[int, ...]:
    """Parse comma-separated layer numbers and inclusive ranges."""

    layers: list[int] = []
    if not value.strip():
        return ()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError("layer selection contains an empty item")
        if "-" in item:
            raw_start, separator, raw_end = item.partition("-")
            if not separator or "-" in raw_end:
                raise ValueError(f"invalid layer range {item!r}")
            try:
                start = int(raw_start)
                end = int(raw_end)
            except ValueError:
                raise ValueError(f"invalid layer range {item!r}") from None
            if start > end:
                raise ValueError(f"layer range must be ascending: {item!r}")
            layers.extend(range(start, end + 1))
        else:
            try:
                layers.append(int(item))
            except ValueError:
                raise ValueError(f"invalid layer number {item!r}") from None
    if any(not 1 <= layer <= 92 for layer in layers):
        raise ValueError("Kimi K3 layers must be in 1..92")
    if len(set(layers)) != len(layers):
        raise ValueError("Kimi K3 layer selection contains a duplicate")
    return tuple(sorted(layers))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Requantize only the pinned Kimi K3 routed Q2_K experts to MTPLX "
            "t158 and serialize a text-streaming artifact."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--official-revision",
        default=KIMI_K3_OFFICIAL_REVISION,
    )
    parser.add_argument("--official-metadata", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=parse_layers,
        help="pilot layer list/ranges, for example 1,7-9; omit for all 92",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="validate the source and print byte projections without writing",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _print_projection(projection, *, as_json: bool) -> None:
    value = projection.to_dict()
    if as_json:
        print(json.dumps(value, sort_keys=True), flush=True)
        return
    print(
        "Kimi K3 pinned projection: "
        f"source={value['source_tensor_bytes']:,} tensor bytes, "
        f"preserved residents="
        f"{value['output_preserved_resident_tensor_bytes']:,}, "
        f"t158 routed={value['output_routed_tensor_bytes']:,}, "
        f"serialized={value['output_tensor_bytes']:,}, "
        f"text runtime={value['text_runtime_tensor_bytes']:,}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        projection = project_artifact(
            args.source,
            source_revision=args.source_revision,
            official_revision=args.official_revision,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    _print_projection(projection, as_json=args.json)
    if args.inspect:
        return 0
    try:
        result = assemble_artifact(
            args.source,
            args.output,
            source_revision=args.source_revision,
            official_revision=args.official_revision,
            official_root=args.official_metadata,
            resume=args.resume,
            layers=args.layers,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True), flush=True)
    else:
        status = "complete" if result.complete else "pilot/incomplete"
        print(
            f"Kimi K3 assembly {status}: {result.output_root} "
            f"(layers={list(result.converted_layers)})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
