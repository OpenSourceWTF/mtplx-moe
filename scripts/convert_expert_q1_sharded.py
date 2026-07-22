#!/usr/bin/env python3
"""Convert a model-scale q1 artifact in parallel, then merge it exactly once."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_q1 import (  # noqa: E402
    Q1Manifest,
    Q1ManifestError,
    build_streamed_expert_manifest,
    load_q1_manifest,
    verify_q1_against_source,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402


def contiguous_ranges(
    layers: Iterable[int], *, workers: int
) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(sorted({int(layer) for layer in layers}))
    if not ordered:
        raise ValueError("q1 sharding requires at least one layer")
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


def _validate_shard(manifest: Q1Manifest) -> int:
    bin_path = manifest.bin_path()
    size = bin_path.stat().st_size
    cursor = 0
    for record in sorted(manifest.records, key=lambda item: item.offset):
        if record.offset != cursor:
            raise Q1ManifestError(
                f"q1 shard {bin_path} has a gap or overlap at "
                f"({record.layer}, {record.expert})"
            )
        cursor += record.length
    if cursor != size:
        raise Q1ManifestError(
            f"q1 shard records cover {cursor} bytes but {bin_path} is {size}"
        )
    return size


def _write_private_manifest(manifest: Q1Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_json(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def merge_q1_shards(
    shard_manifests: Sequence[Path | str],
    *,
    output_bin: Path | str,
    output_manifest: Path | str,
) -> Q1Manifest:
    loaded = [load_q1_manifest(path) for path in shard_manifests]
    if not loaded:
        raise Q1ManifestError("q1 merge requires at least one shard")
    loaded.sort(key=lambda item: min((r.layer, r.expert) for r in item.records))
    first = loaded[0]
    metadata = (
        first.format,
        first.model_key,
        first.codec,
        first.group_size,
        first.source_model_key,
        first.source_manifest_sha256,
    )
    for manifest in loaded[1:]:
        candidate = (
            manifest.format,
            manifest.model_key,
            manifest.codec,
            manifest.group_size,
            manifest.source_model_key,
            manifest.source_manifest_sha256,
        )
        if candidate != metadata:
            raise Q1ManifestError("q1 shard metadata does not match")

    output_bin = Path(output_bin)
    output_manifest = Path(output_manifest)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_bin.name}.", suffix=".tmp", dir=output_bin.parent
    )
    temporary = Path(temporary_name)
    records = []
    seen: set[tuple[int, int]] = set()
    base = 0
    try:
        with os.fdopen(descriptor, "wb") as sink:
            for manifest in loaded:
                shard_size = _validate_shard(manifest)
                with manifest.bin_path().open("rb") as source:
                    shutil.copyfileobj(source, sink, 8 * 1024 * 1024)
                for record in manifest.records:
                    key = (record.layer, record.expert)
                    if key in seen:
                        raise Q1ManifestError(f"duplicate q1 shard record {key}")
                    seen.add(key)
                    records.append(replace(record, offset=record.offset + base))
                base += shard_size
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, output_bin)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    records.sort(key=lambda item: (item.layer, item.expert))
    merged = Q1Manifest(
        format=first.format,
        model_key=first.model_key,
        codec=first.codec,
        group_size=first.group_size,
        file=output_bin.name,
        source_model_key=first.source_model_key,
        source_manifest_sha256=first.source_manifest_sha256,
        records=tuple(records),
        path=output_manifest,
    )
    _write_private_manifest(merged, output_manifest)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--codec", choices=("b1", "t158"), required=True)
    parser.add_argument("--streamed-model-key", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-sample", type=int, default=3)
    parser.add_argument("--keep-shards", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest_path = args.manifest or source_root / "expert-manifest.json"
    source_manifest = load_expert_manifest(source_manifest_path)
    layers = sorted({record.layer for record in source_manifest.records})
    ranges = contiguous_ranges(layers, workers=args.workers)
    shard_root = output_dir / f".q1-{args.codec}-shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    print(
        f"q1[{args.codec}] {len(layers)} routed layers -> {len(ranges)} shards",
        flush=True,
    )

    processes = []
    started = time.time()
    for index, layer_range in enumerate(ranges):
        shard_dir = shard_root / f"shard{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            str(_ROOT / "scripts/convert_expert_q1.py"),
            "--source-root",
            str(source_root),
            "--manifest",
            str(source_manifest_path),
            "--output-dir",
            str(shard_dir),
            "--codec",
            args.codec,
            "--layers",
            ",".join(str(layer) for layer in layer_range),
            "--verify-sample",
            "0",
        ]
        if args.resume:
            command.append("--resume")
        log = (shard_root / f"shard{index:02d}.log").open("w")
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((index, process, log, shard_dir))
    print(f"launched {len(processes)} q1 shards", flush=True)

    failures = []
    for index, process, log, _shard_dir in processes:
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
        print(f"FATAL: q1 shard failures: {failures}; not merging", flush=True)
        return 1

    shard_manifests = tuple(
        shard_dir / f"expert-manifest-q1-{args.codec}.json"
        for _index, _process, _log, shard_dir in processes
    )
    output_bin = output_dir / f"experts-q1-{args.codec}.bin"
    private_manifest_path = output_dir / f"expert-manifest-q1-{args.codec}.json"
    merged = merge_q1_shards(
        shard_manifests,
        output_bin=output_bin,
        output_manifest=private_manifest_path,
    )
    expected = {(record.layer, record.expert) for record in source_manifest.records}
    actual = {(record.layer, record.expert) for record in merged.records}
    if actual != expected:
        raise Q1ManifestError(
            f"merged q1 record set differs from source: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )

    streamed_spec = get_model_spec(args.streamed_model_key)
    authoritative = build_streamed_expert_manifest(output_dir, merged, streamed_spec)
    authoritative_path = output_dir / "expert-manifest.json"
    save_expert_manifest(authoritative, authoritative_path)
    if args.verify_sample:
        for layer, expert in verify_q1_against_source(
            merged,
            source_manifest,
            source_root,
            sample=args.verify_sample,
        ):
            print(f"verified bitwise: layer {layer} expert {expert}", flush=True)

    if not args.keep_shards:
        for _index, _process, _log, shard_dir in processes:
            (shard_dir / f"experts-q1-{args.codec}.bin").unlink()
        print("removed validated temporary shard bins", flush=True)
    print(
        f"q1[{args.codec}] merged {len(merged.records)} records -> "
        f"{output_bin.stat().st_size / 2**30:.1f} GiB in "
        f"{(time.time() - started) / 60:.1f} min; "
        f"authoritative manifest -> {authoritative_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
