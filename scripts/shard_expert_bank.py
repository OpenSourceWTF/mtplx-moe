#!/usr/bin/env python3
"""Split a single-file expert bank into Hugging-Face-publishable safetensors parts.

``experts.bin`` is 89.46 GB for hy3 q2 and 226 GB for glm52 q2, against Hugging
Face's **50 GB hard per-file limit** (~20 GB is the practical recommendation).
That single file is the only thing standing between these artifacts and being
downloadable, and the constraint was never technical: ``pread`` does not care
how many files there are.

This is the writer half of ``docs/plans/2026-07-18-safetensors-expert-banks.md``.
The reader half (parts-aware ``SidecarInfo`` / ``ExpertRecord.part``) already
shipped, and reads a one-part manifest byte-identically, so converted and
unconverted artifacts both load.

## Why this is a copy, not a re-quantization

An expert record is already exactly its 9 component tensors written back to
back with no gaps — verified on the real banks, where a record's span equals
its ``logical_bytes`` exactly (5,898,240 for hy3 q2). That is precisely what a
safetensors data section is. So conversion moves bytes and writes a header; it
never touches a weight. Per-record sha256 values carry over unchanged and are
re-verified after the move.

## Alignment falls out for free

Record sizes are exact multiples of 16384, which on this platform is
simultaneously the page size and the manifest's ``DEFAULT_ALIGNMENT``. If a
part's data section starts at a 16 KiB-aligned offset and records are written
adjacently, **every** record boundary is page-aligned with no inter-record
padding. We get that by padding the JSON header with trailing whitespace until
``8 + header_len`` is a multiple of the alignment — spec-legal, since the
header is JSON and trailing whitespace is insignificant, and it costs at most
one alignment unit per part.

Without the padding the ``pread`` path still works perfectly; only the
``metal-mmap`` slot layout would be unable to map records directly.

## On shard kind

Each part is declared ``kind="safetensors"``, not ``kind="sidecar"``. A
"sidecar" shard is by definition a raw ``.bin`` with no header
(``header_bytes == 0``), so a framed part is simply a different kind — not a
rule that needed bending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    ExpertManifest,
    ExpertRecord,
    ShardInfo,
    SidecarPart,
    load_expert_manifest,
    resolve_artifact_member,
)

DEFAULT_PART_BYTES = 15 * 1024**3  # ~15 GiB: under HF's 50 GB cap, near the 20 GB advice
HF_HARD_LIMIT_BYTES = 50 * 1024**3
_DTYPE_TO_SAFETENSORS = {"U32": "U32", "BF16": "BF16", "F32": "F32", "U8": "U8", "U16": "U16"}


class ShardConversionError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def plan_parts(
    records: Sequence[ExpertRecord],
    *,
    part_bytes: int = DEFAULT_PART_BYTES,
) -> list[list[ExpertRecord]]:
    """Group records into parts, never splitting a record across a boundary.

    Records are 5.6-11.3 MiB against a ~15 GiB part, so whole-record packing
    wastes at most one record per part. Keeping records whole is what confines
    the reader change to file selection instead of range arithmetic.
    """

    if part_bytes <= 0:
        raise ShardConversionError("part_bytes must be positive")
    largest = max((record.sidecar_length for record in records), default=0)
    if largest > part_bytes:
        raise ShardConversionError(
            f"part_bytes {part_bytes} cannot hold the largest record ({largest})"
        )
    parts: list[list[ExpertRecord]] = []
    current: list[ExpertRecord] = []
    used = 0
    for record in records:
        if current and used + record.sidecar_length > part_bytes:
            parts.append(current)
            current, used = [], 0
        current.append(record)
        used += record.sidecar_length
    if current:
        parts.append(current)
    return parts


def part_filename(stem: str, index: int, total: int) -> str:
    return f"{stem}-{index + 1:05d}-of-{total:05d}.safetensors"


# --------------------------------------------------------------------------
# safetensors framing
# --------------------------------------------------------------------------


def _aligned_header(header: dict[str, Any], alignment: int) -> bytes:
    """Serialize a safetensors header padded so the data section is aligned.

    Trailing whitespace inside the JSON header is insignificant, so padding
    there is spec-legal and keeps every record boundary page-aligned.
    """

    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    total = 8 + len(body)
    pad = (-total) % alignment
    body += b" " * pad
    return struct.pack("<Q", len(body)) + body


def build_part_header(
    records: Sequence[ExpertRecord], alignment: int
) -> tuple[bytes, list[tuple[ExpertRecord, int]]]:
    """Build the header for one part and the record offsets inside its data.

    Returns the framed header bytes and ``(record, offset_from_data_start)``
    pairs. Components of a record are emitted adjacently and in manifest order,
    which is what preserves the one-pread-per-expert fast path.
    """

    header: dict[str, Any] = {}
    placements: list[tuple[ExpertRecord, int]] = []
    cursor = 0
    for record in records:
        # Align every record start, exactly as build_expert_sidecar does.
        # On the real banks record sizes are already 16384-multiples so this
        # is a no-op and the data section stays gapless; it matters only for
        # banks whose records are not alignment multiples, where packing
        # adjacently would silently break the metal-mmap path.
        cursor += (-cursor) % alignment
        placements.append((record, cursor))
        for segment in record.segments:
            dtype = _DTYPE_TO_SAFETENSORS.get(segment.dtype)
            if dtype is None:
                raise ShardConversionError(f"unsupported dtype {segment.dtype!r}")
            header[segment.tensor] = {
                "dtype": dtype,
                "shape": list(segment.shape),
                "data_offsets": [cursor, cursor + segment.length],
            }
            cursor += segment.length
    return _aligned_header(header, alignment), placements


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------


def _copy_record(
    src_fd: int, dst, record: ExpertRecord, chunk: int, *, src_data_start: int = 0
) -> str:
    """Stream one record from the source bank, returning its sha256.

    ``src_data_start`` is the source part's framing offset; record offsets in
    the manifest are relative to it.
    """

    digest = hashlib.sha256()
    remaining = record.sidecar_length
    offset = src_data_start + record.sidecar_offset
    while remaining:
        want = min(chunk, remaining)
        blob = os.pread(src_fd, want, offset)
        if not blob:
            raise ShardConversionError(
                f"source bank truncated at offset {offset} for record "
                f"({record.layer}, {record.expert})"
            )
        dst.write(blob)
        digest.update(blob)
        offset += len(blob)
        remaining -= len(blob)
    return digest.hexdigest()


def convert(
    manifest: ExpertManifest,
    root: Path,
    destination: Path,
    *,
    part_bytes: int = DEFAULT_PART_BYTES,
    stem: str = "experts",
    chunk: int = 8 * 1024**2,
    progress: bool = True,
) -> ExpertManifest:
    """Write a sharded copy of the bank and return the new manifest.

    The source artifact is opened read-only and never modified.
    """

    if manifest.sidecar is None:
        raise ShardConversionError("manifest has no sidecar bank to shard")
    if len(manifest.sidecar.parts) != 1:
        raise ShardConversionError("source bank is already sharded")

    alignment = manifest.sidecar.alignment
    source = resolve_artifact_member(root, manifest.sidecar.parts[0].file)
    groups = plan_parts(manifest.records, part_bytes=part_bytes)
    destination.mkdir(parents=True, exist_ok=True)

    parts: list[SidecarPart] = []
    shards: list[ShardInfo] = []
    rewritten: dict[tuple[int, int], ExpertRecord] = {}

    src_fd = os.open(source, os.O_RDONLY)
    try:
        for index, group in enumerate(groups):
            name = part_filename(stem, index, len(groups))
            header, placements = build_part_header(group, alignment)
            out_path = destination / name
            file_digest = hashlib.sha256()
            file_digest.update(header)
            with open(out_path, "wb") as handle:
                handle.write(header)
                for record, offset in placements:
                    # honour the planned (aligned) offset
                    gap = len(header) + offset - handle.tell()
                    if gap > 0:
                        handle.write(b"\x00" * gap)
                    actual = _copy_record(
                        src_fd, handle, record, chunk,
                        src_data_start=manifest.sidecar.parts[0].data_start,
                    )
                    if record.sha256 and actual != record.sha256:
                        raise ShardConversionError(
                            f"record ({record.layer}, {record.expert}) digest "
                            f"changed during copy: {actual} != {record.sha256}"
                        )
                    # Rebase the record's segments onto this part, the way
                    # make_sidecar_authoritative does: each component points at
                    # the part file at data_start + its offset within the
                    # record. Without this the segments still name the source
                    # shard and validation rejects the manifest.
                    cursor = len(header) + offset
                    segments = []
                    for segment in record.segments:
                        segments.append(
                            replace(segment, shard=name, offset=cursor)
                        )
                        cursor += segment.length
                    rewritten[(record.layer, record.expert)] = replace(
                        record,
                        part=index,
                        sidecar_offset=offset,
                        segments=tuple(segments),
                    )
                size = handle.tell()
            # Digest the finished file in one pass rather than holding it.
            file_digest = hashlib.sha256()
            with open(out_path, "rb") as handle:
                for blob in iter(lambda: handle.read(chunk), b""):
                    file_digest.update(blob)
            if size > HF_HARD_LIMIT_BYTES:
                raise ShardConversionError(
                    f"{name} is {size} bytes, over Hugging Face's 50 GB limit"
                )
            parts.append(
                SidecarPart(
                    file=name,
                    size=size,
                    sha256=file_digest.hexdigest(),
                    data_start=len(header),
                )
            )
            shards.append(
                ShardInfo(
                    name=name,
                    size=size,
                    header_bytes=len(header),
                    header_sha256=hashlib.sha256(header).hexdigest(),
                    sha256=file_digest.hexdigest(),
                    kind="safetensors",
                )
            )
            if progress:
                print(
                    f"  {name}: {len(group)} records, {size / 1024**3:.2f} GiB",
                    flush=True,
                )
    finally:
        os.close(src_fd)

    sidecar = replace(manifest.sidecar, parts=tuple(parts))
    records = tuple(
        rewritten[(record.layer, record.expert)] for record in manifest.records
    )
    # Resident shards must survive: they hold attention/norms/lm_head and are
    # referenced by resident_tensors. Only the bank is being re-laid-out.
    resident_names = {tensor.shard for tensor in manifest.resident_tensors}
    kept = [shard for shard in manifest.shards if shard.name in resident_names]
    return replace(
        manifest,
        sidecar=sidecar,
        records=records,
        shards=tuple(kept) + tuple(shards),
        manifest_sha256=None,
    ).with_digest()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _human(value: str) -> int:
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    text = value.strip().upper().rstrip("B").rstrip("I")
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="artifact directory holding the bank")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="expert manifest (default: <root>/expert-manifest.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="destination directory for the sharded parts",
    )
    parser.add_argument(
        "--part-bytes",
        type=_human,
        default=DEFAULT_PART_BYTES,
        help="target bytes per part (default 15GiB; HF's hard cap is 50GB)",
    )
    parser.add_argument("--stem", default="experts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the part plan and exit without writing or reading weights",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest_path = args.manifest or (root / "expert-manifest.json")
    manifest = load_expert_manifest(manifest_path)

    groups = plan_parts(manifest.records, part_bytes=args.part_bytes)
    total = sum(record.sidecar_length for record in manifest.records)
    print(
        f"{len(manifest.records)} records, {total / 1024**3:.2f} GiB "
        f"-> {len(groups)} parts"
    )
    for index, group in enumerate(groups):
        size = sum(record.sidecar_length for record in group)
        print(
            f"  {part_filename(args.stem, index, len(groups))}: "
            f"{len(group)} records, {size / 1024**3:.2f} GiB"
        )
    if args.dry_run:
        return 0

    out = args.out.expanduser().resolve()
    sharded = convert(
        manifest, root, out, part_bytes=args.part_bytes, stem=args.stem
    )
    target = out / "expert-manifest.json"
    target.write_text(json.dumps(sharded.to_dict(), indent=2))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
