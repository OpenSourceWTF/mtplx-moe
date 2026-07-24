#!/usr/bin/env python
"""Convert a streamed affine expert sidecar into a rANS-compressed sidecar.

Issue #113: PR #112's decoder is exact and 2.7x over the SSD-bandwidth
target, but its integration only shrank ISLAND disk footprint. This lane
produces the STREAMED compressed artifact -- one self-describing rANS
container per expert record -- so a per-wave miss read pulls ~1.31x fewer
bytes off SSD, and the reader rebuilds the raw record in-kernel between the
(smaller) read and slot residency, bitwise-identical to reading it raw.

It reads the base expert manifest's streamed records (from the authoritative
sidecar when present, else the source shards), rANS-encodes each record's
payload, and writes:

    experts-rans32x.bin              (compressed sidecar)
    expert-streamed-codec-rans32x.json  (StreamedCodecManifest)

Streams one record at a time (bounded memory) and is resumable (--resume).

Smoke slices -- ``--layers 1 --experts 0,1`` or ``--limit N`` -- bound the
record set so the lane validates in seconds. Do NOT run the unbounded burn on
a full hy3/GLM artifact without the model-scale window; the converter is
offline (pure numpy, no MLX) but the full sweep re-reads the whole sidecar.
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
        help="Base expert manifest (default: <source-root>/expert-manifest.json).",
    )
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=None,
        help="Compressed sidecar path (default: <source-root>/experts-rans32x.bin).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Streamed codec manifest path (default beside the source manifest).",
    )
    parser.add_argument("--codec", choices=("rans32x-v1",), default="rans32x-v1")
    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--experts", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alignment", type=int, default=16 * 1024)
    parser.add_argument(
        "--no-verify-source-hashes",
        action="store_true",
        help="Skip sha256 verification of every source record read (faster).",
    )
    parser.add_argument(
        "--no-verify-roundtrip",
        action="store_true",
        help="Skip the per-record numpy decode round-trip check (faster).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-written containers whose bytes still hash correctly.",
    )
    args = parser.parse_args()

    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.expert_streamed_codec import write_streamed_rans_sidecar

    source_root = args.source_root.resolve()
    manifest_path = args.manifest or source_root / "expert-manifest.json"
    base_manifest = load_expert_manifest(manifest_path)
    output_bin = args.output_bin or source_root / "experts-rans32x.bin"
    output_manifest = (
        args.output_manifest
        or manifest_path.parent / "expert-streamed-codec-rans32x.json"
    )

    started = time.perf_counter()
    manifest = write_streamed_rans_sidecar(
        base_manifest,
        source_root,
        output_bin=output_bin,
        output_manifest=output_manifest,
        codec=args.codec,
        alignment=args.alignment,
        layers=_parse_ints(args.layers),
        experts=_parse_ints(args.experts),
        limit=args.limit,
        resume=args.resume,
        verify_source_hashes=not args.no_verify_source_hashes,
        verify_roundtrip=not args.no_verify_roundtrip,
        progress=True,
    )
    elapsed = time.perf_counter() - started
    print(
        f"streamed[{args.codec}] wrote {len(manifest.records)} records, "
        f"stored {manifest.stored_bytes / 1024**2:.1f} MiB from "
        f"{manifest.raw_bytes / 1024**2:.1f} MiB raw "
        f"({manifest.compression_ratio():.3f}x smaller), "
        f"{elapsed:.1f}s -> {manifest.bin_path()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
