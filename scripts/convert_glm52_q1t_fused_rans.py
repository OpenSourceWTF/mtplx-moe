#!/usr/bin/env python3
"""Build the GLM-5.2 Q1T matmul-native banked rANS artifact."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_q1 import load_q1_manifest  # noqa: E402
from mtplx.glm52_q1t_rans_artifact import (  # noqa: E402
    write_glm52_q1t_fused_rans_artifact,
)


def _parse_layers(value: str) -> tuple[int, ...] | None:
    stripped = value.strip()
    if not stripped:
        return None
    return tuple(sorted({int(item) for item in stripped.split(",")}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--q1-manifest", type=Path, default=None)
    parser.add_argument("--expert-manifest", type=Path, default=None)
    parser.add_argument("--output-bin", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--layers", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--uniform-packed", action="store_true")
    parser.add_argument("--no-verify-record-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    q1_manifest_path = args.q1_manifest or source_root / "expert-manifest-q1-t158.json"
    expert_manifest_path = args.expert_manifest or source_root / "expert-manifest.json"
    source = load_q1_manifest(q1_manifest_path)
    authoritative = load_expert_manifest(expert_manifest_path)
    if (
        authoritative.model_key != "glm52-expert-q1t"
        or authoritative.quant_mode != "t158"
        or authoritative.quant_group_size != 64
        or authoritative.manifest_sha256 is None
    ):
        raise ValueError(
            "authoritative manifest must be glm52-expert-q1t t158 g64"
        )
    available_layers = tuple(sorted({record.layer for record in authoritative.records}))
    layers = _parse_layers(args.layers) or available_layers
    if not layers or not set(layers).issubset(available_layers):
        raise ValueError("selected layers are outside the authoritative Q1T manifest")
    expert_sets = {
        layer: {record.expert for record in authoritative.records if record.layer == layer}
        for layer in layers
    }
    reference_experts = set(range(len(expert_sets[layers[0]])))
    if any(experts != reference_experts for experts in expert_sets.values()):
        raise ValueError("authoritative Q1T manifest does not have rectangular experts")
    suffix = "-uniform-packed" if args.uniform_packed else ""
    output_bin = args.output_bin or source_root / (
        f"experts-glm52-q1t-fused-rans{suffix}.bin"
    )
    output_manifest = args.output_manifest or source_root / (
        f"expert-manifest-glm52-q1t-fused-rans{suffix}.json"
    )
    started = time.perf_counter()
    manifest = write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=output_bin,
        output_manifest=output_manifest,
        layers=layers,
        expected_expert_count=len(reference_experts),
        verify_record_hashes=not args.no_verify_record_hashes,
        source_expert_manifest_sha256=authoritative.manifest_sha256,
        resume=args.resume,
        uniform_packed=args.uniform_packed,
    )
    elapsed = time.perf_counter() - started
    print(
        f"fused-rANS wrote {manifest.file_bytes / 2**30:.3f} GiB "
        f"across {len(layers)} layers in {elapsed:.1f}s; "
        f"sha256={manifest.file_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
