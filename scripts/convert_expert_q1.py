#!/usr/bin/env python
"""Convert an affine expert artifact into a q1 (shadow-codec) artifact.

Minimal conversion lane for the issue #51 q1 burn: dequantizes each
source record (the exact affine math the runtime executes), re-encodes it
with a shadow codec, and writes a self-describing q1 artifact
(``experts-q1-<codec>.bin`` + ``expert-manifest-q1-<codec>.json``).

Probe verdict (research/glm52-shadow-codec-probe-*.json): burn **t158**;
b1 is a no-go from Q2 sources (the Q2 grid's exact-zero level carries
~50% of the weight mass and a 1-bit sign code cannot represent it).

Smoke slices: ``--layers 3 --experts 0,1,2`` or ``--limit N`` bound the
record set so the lane can be validated in seconds; ``--verify-sample``
re-reads written records and requires bitwise equality against a fresh
encode of the source. The full burn is the unbounded invocation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_ints(text: str) -> tuple[int, ...] | None:
    text = text.strip()
    if not text:
        return None
    return tuple(int(value) for value in text.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Source expert manifest (default: <source-root>/expert-manifest.json).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=("b1", "t158"), required=True)
    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--experts", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="sha256-verify every source record read (slower).",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=3,
        help="Re-encode this many written records and require bitwise "
        "equality (0 disables).",
    )
    args = parser.parse_args()

    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.expert_q1 import convert_expert_q1, verify_q1_against_source
    from mtplx.expert_streaming_models import get_model_spec

    manifest_path = args.manifest or args.source_root / "expert-manifest.json"
    source_manifest = load_expert_manifest(manifest_path)
    spec = get_model_spec(source_manifest.model_key)
    started = time.perf_counter()
    manifest = convert_expert_q1(
        source_manifest,
        args.source_root,
        args.output_dir,
        codec=args.codec,
        spec=spec,
        layers=_parse_ints(args.layers),
        experts=_parse_ints(args.experts),
        limit=args.limit,
        verify_source_hashes=args.verify_source_hashes,
        progress=True,
    )
    elapsed = time.perf_counter() - started
    total_bytes = sum(record.length for record in manifest.records)
    source_bytes = len(manifest.records) * source_manifest.records[0].logical_bytes
    print(
        f"q1[{args.codec}] wrote {len(manifest.records)} records, "
        f"{total_bytes / 1024**2:.1f} MiB "
        f"({source_bytes / max(total_bytes, 1):.3f}x smaller than source), "
        f"{elapsed:.1f}s -> {manifest.bin_path()}"
    )
    if args.verify_sample:
        for layer, expert in verify_q1_against_source(
            manifest,
            source_manifest,
            args.source_root,
            sample=args.verify_sample,
        ):
            print(f"verified bitwise: layer {layer} expert {expert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
