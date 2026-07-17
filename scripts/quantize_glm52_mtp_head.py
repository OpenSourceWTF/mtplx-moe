#!/usr/bin/env python3
"""Quantize the GLM-5.2 layer-78 MTP head's expert bank to affine Q4.

The trained NextN head is packaged BF16 by ``extract_glm52_mtp_layer78.py``:
18.54 GiB of which 18.0 GiB (97%) is the head's own 256 routed experts
(3 x 24 MiB BF16 each).  This tool streams that verified BF16 artifact and
publishes a sibling ``layer78-q4.safetensors`` whose routed experts are the
pinned trunk expert recipe (affine Q4 group-size 64, U32 packed weight with
BF16 scales/biases) while every trunk-side tensor (norms, eh_proj, indexer,
attention, shared experts, router gate/bias) is copied bit-exact.  The head
shrinks from 18.54 GiB to ~5.6 GiB resident, freeing ~12.9 GiB of the fixed
streamed budget.

The head is speculative-only: a worse head costs draft acceptance, never
correctness, and GLM conditional acceptance (0.93/0.91/0.85 at d1-d3) leaves
headroom.  A Q4 acceptance-rate validation run gates making it the serving
default; until then BF16 stays the default and Q4 is selectable.

Memory: conversion is bounded — trunk tensors are copied in fixed-size
chunks and each routed expert projection is read, quantized, and written
before the next is touched (never the whole 18.5 GiB at once).  Publication
is atomic and the staged artifact is fully re-verified before it is visible.

The 18.5 GiB burn is a coordinator-scheduled disk window, not a unit test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run standalone regardless of whether mtplx is installed: prefer this
# checkout's package over any stale site-packages sibling.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mtplx.glm52_mtp_artifact import (  # noqa: E402
    ArtifactError,
    Glm52MtpQ4Config,
    quantize_glm52_mtp_layer78_q4,
    verify_glm52_mtp_layer78_q4,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize = subparsers.add_parser(
        "quantize",
        help="stream the verified BF16 head and publish the Q4 sibling artifact",
    )
    quantize.add_argument(
        "--bf16-root",
        type=Path,
        required=True,
        help="published BF16 head directory (layer78-bf16.safetensors + manifest)",
    )
    quantize.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new sibling directory to publish the Q4 artifact into",
    )
    quantize.add_argument(
        "--producer-root",
        type=Path,
        required=True,
        help="authenticated clean Git worktree stamped into the manifest",
    )

    verify = subparsers.add_parser(
        "verify", help="verify a published Q4 head artifact from its receipt"
    )
    verify.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "quantize":
            published = quantize_glm52_mtp_layer78_q4(
                Glm52MtpQ4Config(
                    bf16_root=args.bf16_root,
                    output_root=args.output_root,
                    producer_root=args.producer_root,
                )
            )
            manifest = verify_glm52_mtp_layer78_q4(published, deep=True)
            print(
                json.dumps(
                    {
                        "published": str(published),
                        "payload_bytes": manifest["inventory"]["payload_bytes"],
                        "tensor_count": manifest["inventory"]["tensor_count"],
                        "min_roundtrip_cosine": manifest["quantization"][
                            "min_roundtrip_cosine"
                        ],
                    },
                    sort_keys=True,
                )
            )
        else:
            manifest = verify_glm52_mtp_layer78_q4(args.output_root, deep=True)
            print(json.dumps(manifest, sort_keys=True))
    except ArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
