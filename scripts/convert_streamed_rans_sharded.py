#!/usr/bin/env python3
"""Encode streamed rANS records in parallel and publish one exact sidecar."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_streamed_codec import (  # noqa: E402
    merge_streamed_codec_shards,
)


def contiguous_ranges(
    layers: Iterable[int], *, workers: int
) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(sorted({int(layer) for layer in layers}))
    if not ordered:
        raise ValueError("rANS sharding requires at least one layer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    parts = min(workers, len(ordered))
    ranges: list[tuple[int, ...]] = []
    for index in range(parts):
        chunk = ordered[
            round(index * len(ordered) / parts) : round(
                (index + 1) * len(ordered) / parts
            )
        ]
        if chunk:
            ranges.append(chunk)
    return tuple(ranges)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-bin", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-verify-source-hashes", action="store_true")
    parser.add_argument("--no-verify-roundtrip", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    manifest_path = args.manifest or source_root / "expert-manifest.json"
    base_manifest = load_expert_manifest(manifest_path)
    layers = sorted({record.layer for record in base_manifest.records})
    ranges = contiguous_ranges(layers, workers=args.workers)
    shard_dir = (args.shard_dir or source_root / ".rans32x-shards").resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_bin = (args.output_bin or source_root / "experts-rans32x.bin").resolve()
    output_manifest = (
        args.output_manifest or source_root / "expert-streamed-codec-rans32x.json"
    ).resolve()
    print(
        f"streamed[rans32x-v1] {len(layers)} routed layers -> {len(ranges)} shards",
        flush=True,
    )

    processes = []
    started = time.time()
    for index, layer_range in enumerate(ranges):
        shard_bin = shard_dir / f"shard{index:02d}.bin"
        shard_manifest = shard_dir / f"shard{index:02d}.json"
        command = [
            args.python,
            str(_ROOT / "scripts/convert_streamed_rans.py"),
            "--source-root",
            str(source_root),
            "--manifest",
            str(manifest_path),
            "--output-bin",
            str(shard_bin),
            "--output-manifest",
            str(shard_manifest),
            "--layers",
            ",".join(str(layer) for layer in layer_range),
        ]
        if args.resume:
            command.append("--resume")
        if args.no_verify_source_hashes:
            command.append("--no-verify-source-hashes")
        if args.no_verify_roundtrip:
            command.append("--no-verify-roundtrip")
        log = (shard_dir / f"shard{index:02d}.log").open("w")
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((index, process, log, shard_bin, shard_manifest))
    print(f"launched {len(processes)} rANS shards", flush=True)

    failures = []
    for index, process, log, _shard_bin, _shard_manifest in processes:
        returncode = process.wait()
        log.close()
        if returncode != 0:
            failures.append((index, returncode))
        print(
            f"[{time.strftime('%H:%M:%S')}] shard {index:02d} rc={returncode} "
            f"({time.time() - started:.0f}s elapsed)",
            flush=True,
        )
    if failures:
        print(f"FATAL: rANS shard failures: {failures}; not merging", flush=True)
        return 1

    shard_manifests = tuple(item[4] for item in processes)
    merged = merge_streamed_codec_shards(
        shard_manifests,
        source_root,
        output_bin=output_bin,
        output_manifest=output_manifest,
        base_manifest=base_manifest,
    )
    if not args.keep_shards:
        for _index, _process, _log, shard_bin, _shard_manifest in processes:
            shard_bin.unlink()
        print("removed validated temporary shard bins", flush=True)
    print(
        f"streamed[rans32x-v1] merged {len(merged.records)} records -> "
        f"{merged.size / 2**30:.1f} GiB, {merged.compression_ratio():.3f}x "
        f"in {(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
