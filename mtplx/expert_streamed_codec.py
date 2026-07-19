"""Streamed compressed expert sidecar: per-record rANS containers (issue #113).

PR #112 gave us an exact, in-kernel byte-rANS decoder and wired it into the
*island* (banked) load path, where it shrinks disk footprint but not the
per-token bandwidth (island banks are resident regardless). This module is
the *streamed* analogue: it stores every streamed expert **record** as its
own self-describing rANS container so a per-wave miss read pulls ~1.31x fewer
bytes off SSD, then the reader rebuilds the raw record through the same
``expert_rans_metal`` kernel between the (now smaller) read and slot residency
-- bitwise-identical to reading the uncompressed record.

Why a separate manifest (not the strict ``ExpertManifest`` sidecar): an
authoritative ``ExpertManifest`` sidecar requires ``sidecar_length ==
logical_bytes`` and byte-contiguous component segments. A compressed record
breaks both (its on-disk length is the *compressed* container size, and its
bytes are entropy-coded, not the raw component tensors). So the compressed
sidecar carries its own record-major descriptor -- mirroring
``expert_banked.BankedManifest`` but keyed by ``(layer, expert)`` instead of a
component-major bank -- and *references* the base manifest (by digest) which
still owns record geometry, spec identity, and the raw-payload hashes the
reader verifies decoded bytes against.

Container per record: the record's raw payload (the segment-concatenated
``logical_bytes``) is zero-padded up to the 32-lane interleave and encoded as
a single-expert (``expert_count == 1``) ``expert_rans`` container. ``raw_length``
in the manifest is the true (pre-pad) length; the decoder slices the padded
decode back to it. The codec keeps ``rans32x-v1`` semantics unchanged, so the
same Metal kernel that #112 validated decodes these records with no new
device code.

Offline / open-path tooling only: the converter is pure numpy (no MLX, no
locks), streams one record at a time (bounded memory), and is resumable like
``build_expert_sidecar`` / the q1 converter.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from mtplx import expert_rans as rans
from mtplx.expert_manifest import (
    ExpertManifest,
    _align_up,
    _safe_relative_name,
    read_expert_record,
    resolve_artifact_member,
)

STREAMED_CODEC_FORMAT = "mtplx-streamed-expert-codec-v1"
STREAMED_CODECS = ("rans32x-v1",)
DEFAULT_ALIGNMENT = 16 * 1024


class StreamedCodecError(ValueError):
    """Raised when a streamed codec sidecar or its manifest is invalid."""


# ---------------------------------------------------------------- codec core


def encode_record_payload(payload: bytes | bytearray, *, codec: str = "rans32x-v1") -> bytes:
    """Encode one raw record payload into a self-describing rANS container.

    The payload is zero-padded up to the 32-lane interleave and encoded as a
    single-expert bank; the true length is recovered by the reader from the
    manifest ``raw_length`` (the container's ``seg_len`` is the padded size).
    """

    if codec != "rans32x-v1":
        raise StreamedCodecError(f"unsupported streamed codec {codec!r}")
    arr = np.frombuffer(bytes(payload), dtype=np.uint8)
    pad = (-arr.size) % rans.LANES
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    segments = arr.reshape(1, arr.size)
    try:
        table = rans.build_table(rans.histogram(arr))
        streams = rans.encode_bank(segments, table)
        return rans.serialize_component(streams, table)
    except rans.RansError as exc:
        raise StreamedCodecError(f"cannot encode record: {exc}") from exc


def decode_record_reference(blob, raw_length: int, *, codec: str = "rans32x-v1") -> bytes:
    """Pure-numpy decode of a record container back to its raw payload.

    Mirrors what ``expert_rans_metal.decode_container`` produces on device;
    used by the converter self-check and tests so parity does not require MLX.
    """

    if codec != "rans32x-v1":
        raise StreamedCodecError(f"unsupported streamed codec {codec!r}")
    decoded = rans.decode_container_reference(blob).reshape(-1)
    if int(raw_length) > decoded.size:
        raise StreamedCodecError("raw_length exceeds the decoded container")
    return decoded[: int(raw_length)].tobytes()


# ---------------------------------------------------------------- descriptor


@dataclass(frozen=True)
class StreamedCodecRecord:
    layer: int
    expert: int
    offset: int  # compressed byte offset in the codec sidecar file
    length: int  # compressed container byte length
    raw_length: int  # decoded (raw) payload length == base record logical_bytes
    sha256: str  # raw payload sha256 (== base ExpertRecord.sha256)

    def to_json(self) -> dict:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "offset": self.offset,
            "length": self.length,
            "raw_length": self.raw_length,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, obj) -> "StreamedCodecRecord":
        if not isinstance(obj, dict):
            raise StreamedCodecError("streamed codec record must be an object")
        try:
            layer = int(obj["layer"])
            expert = int(obj["expert"])
            offset = int(obj["offset"])
            length = int(obj["length"])
            raw_length = int(obj["raw_length"])
            sha256 = str(obj["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamedCodecError(
                f"streamed codec record is malformed: {exc}"
            ) from exc
        if offset < 0 or length <= 0 or raw_length <= 0:
            raise StreamedCodecError("streamed codec record range is degenerate")
        return cls(
            layer=layer,
            expert=expert,
            offset=offset,
            length=length,
            raw_length=raw_length,
            sha256=sha256,
        )


@dataclass(frozen=True)
class StreamedCodecManifest:
    format: str
    model_key: str
    codec: str
    file: str
    alignment: int
    expert_count: int
    base_manifest_sha256: str
    size: int
    sha256: str
    records: tuple[StreamedCodecRecord, ...]
    path: Path | None = field(default=None, compare=False)

    # -- lookups -----------------------------------------------------------

    def record_map(self) -> dict[tuple[int, int], StreamedCodecRecord]:
        return {(r.layer, r.expert): r for r in self.records}

    def record_for(self, layer: int, expert: int) -> StreamedCodecRecord:
        for record in self.records:
            if record.layer == int(layer) and record.expert == int(expert):
                return record
        raise StreamedCodecError(
            f"streamed codec sidecar has no record ({layer}, {expert})"
        )

    def bin_path(self) -> Path:
        if self.path is None:
            raise StreamedCodecError("streamed codec manifest has no on-disk location")
        return self.path.parent / self.file

    # -- pricing -----------------------------------------------------------

    @property
    def stored_bytes(self) -> int:
        """Total compressed record bytes actually read off SSD on full sweep."""

        return sum(record.length for record in self.records)

    @property
    def raw_bytes(self) -> int:
        """Total decoded record bytes (what codec 'none' would read)."""

        return sum(record.raw_length for record in self.records)

    def compression_ratio(self) -> float:
        stored = self.stored_bytes
        return float(self.raw_bytes) / stored if stored else 1.0

    # -- serialization -----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "format": self.format,
            "model_key": self.model_key,
            "codec": self.codec,
            "file": self.file,
            "alignment": self.alignment,
            "expert_count": self.expert_count,
            "base_manifest_sha256": self.base_manifest_sha256,
            "size": self.size,
            "sha256": self.sha256,
            "records": [record.to_json() for record in self.records],
        }

    def validate(self) -> None:
        if self.format != STREAMED_CODEC_FORMAT:
            raise StreamedCodecError(
                f"unsupported streamed codec manifest format {self.format!r}"
            )
        if self.codec not in STREAMED_CODECS:
            raise StreamedCodecError(f"unsupported streamed codec {self.codec!r}")
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)):
            raise StreamedCodecError("streamed codec alignment must be a power of two")
        if self.expert_count <= 0:
            raise StreamedCodecError("streamed codec expert_count must be positive")
        if self.size <= 0:
            raise StreamedCodecError("streamed codec sidecar size must be positive")
        if not self.records:
            raise StreamedCodecError("streamed codec manifest lists no records")
        keys = [(record.layer, record.expert) for record in self.records]
        if len(keys) != len(set(keys)):
            raise StreamedCodecError("streamed codec records are not unique")
        ranges: list[tuple[int, int]] = []
        for record in self.records:
            if record.offset % self.alignment:
                raise StreamedCodecError(
                    f"streamed codec record ({record.layer}, {record.expert}) "
                    "is not aligned"
                )
            end = record.offset + record.length
            if end > self.size:
                raise StreamedCodecError(
                    f"streamed codec record ({record.layer}, {record.expert}) "
                    "exceeds the sidecar size"
                )
            ranges.append((record.offset, end))
        ranges.sort()
        for (a_lo, a_hi), (b_lo, _b_hi) in zip(ranges, ranges[1:]):
            if a_hi > b_lo:
                raise StreamedCodecError("streamed codec records overlap")

    @classmethod
    def from_json(cls, obj, *, path: Path | None = None) -> "StreamedCodecManifest":
        if not isinstance(obj, dict):
            raise StreamedCodecError("streamed codec manifest must be an object")
        try:
            records = tuple(
                StreamedCodecRecord.from_json(item) for item in obj["records"]
            )
            manifest = cls(
                format=str(obj["format"]),
                model_key=str(obj["model_key"]),
                codec=str(obj["codec"]),
                file=_safe_relative_name(str(obj["file"]), label="streamed codec file"),
                alignment=int(obj["alignment"]),
                expert_count=int(obj["expert_count"]),
                base_manifest_sha256=str(obj["base_manifest_sha256"]),
                size=int(obj["size"]),
                sha256=str(obj["sha256"]),
                records=records,
                path=path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamedCodecError(
                f"streamed codec manifest is malformed: {exc}"
            ) from exc
        manifest.validate()
        return manifest


def load_streamed_codec_manifest(path: Path | str) -> StreamedCodecManifest:
    path = Path(path)
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise StreamedCodecError(
            f"cannot read streamed codec manifest: {exc}"
        ) from exc
    return StreamedCodecManifest.from_json(obj, path=path)


def save_streamed_codec_manifest(
    manifest: StreamedCodecManifest, path: Path | str
) -> None:
    manifest.validate()
    Path(path).write_text(json.dumps(manifest.to_json(), indent=1))


def validate_against_base(
    manifest: StreamedCodecManifest, base: ExpertManifest
) -> None:
    """Require the codec sidecar to exactly cover the base manifest records.

    The base manifest owns record geometry, identity, and the raw-payload
    hashes; the codec sidecar only re-locates each record's *bytes*. Every
    base record must have a codec record whose raw_length and hash match, so
    the reader can decode + verify against ``ExpertRecord.sha256`` unchanged.
    """

    manifest.validate()
    if base.manifest_sha256 is not None and (
        manifest.base_manifest_sha256 != base.manifest_sha256
    ):
        raise StreamedCodecError(
            "streamed codec sidecar references a different base manifest digest"
        )
    if manifest.model_key != base.model_key:
        raise StreamedCodecError(
            "streamed codec model_key does not match the base manifest"
        )
    expert_count = 1 + max(record.expert for record in base.records)
    if manifest.expert_count != expert_count:
        raise StreamedCodecError(
            f"streamed codec holds {manifest.expert_count} experts; base has "
            f"{expert_count}"
        )
    codec_map = manifest.record_map()
    for record in base.records:
        key = (record.layer, record.expert)
        codec_record = codec_map.get(key)
        if codec_record is None:
            raise StreamedCodecError(
                f"streamed codec sidecar is missing record {key}"
            )
        if codec_record.raw_length != record.logical_bytes:
            raise StreamedCodecError(
                f"streamed codec record {key} raw_length "
                f"{codec_record.raw_length} != base logical_bytes "
                f"{record.logical_bytes}"
            )
        if record.sha256 is not None and codec_record.sha256 != record.sha256:
            raise StreamedCodecError(
                f"streamed codec record {key} hash differs from the base record"
            )
    if len(codec_map) != len(base.records):
        raise StreamedCodecError(
            "streamed codec sidecar carries records absent from the base manifest"
        )


# ---------------------------------------------------------------- converter


def _relative_output(root: Path, output: Path) -> tuple[str, Path]:
    base = root.resolve()
    parent = output.parent.resolve()
    try:
        relative_parent = parent.relative_to(base)
    except ValueError as exc:
        raise StreamedCodecError(
            "streamed codec sidecar must be created inside the artifact root"
        ) from exc
    relative = (relative_parent / output.name).as_posix()
    return _safe_relative_name(relative, label="streamed codec output"), parent / output.name


def write_streamed_rans_sidecar(
    base_manifest: ExpertManifest,
    root: Path | str,
    *,
    output_bin: Path | str,
    output_manifest: Path | str,
    codec: str = "rans32x-v1",
    alignment: int = DEFAULT_ALIGNMENT,
    layers: Iterable[int] | None = None,
    experts: Iterable[int] | None = None,
    limit: int | None = None,
    resume: bool = False,
    verify_source_hashes: bool = True,
    verify_roundtrip: bool = True,
    progress: bool = False,
) -> StreamedCodecManifest:
    """Rewrite the base streamed records as per-record rANS containers.

    Streams one record at a time (bounded memory), writes each container at an
    aligned offset, and -- on completion -- publishes the compressed sidecar
    plus a ``StreamedCodecManifest``. ``layers``/``experts``/``limit`` bound a
    smoke slice; ``resume`` reuses already-written containers whose bytes still
    hash to the freshly-encoded blob.
    """

    if codec not in STREAMED_CODECS:
        raise StreamedCodecError(f"unsupported streamed codec {codec!r}")
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise StreamedCodecError("alignment must be a positive integer")
    if alignment & (alignment - 1):
        raise StreamedCodecError("alignment must be a power of two")
    artifact_root = Path(root).resolve()
    relative_output, final_path = _relative_output(artifact_root, Path(output_bin))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    layer_filter = None if layers is None else {int(v) for v in layers}
    expert_filter = None if experts is None else {int(v) for v in experts}
    selected = [
        record
        for record in base_manifest.records
        if (layer_filter is None or record.layer in layer_filter)
        and (expert_filter is None or record.expert in expert_filter)
    ]
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise StreamedCodecError("no base records selected for streamed codec build")
    expert_count = 1 + max(record.expert for record in base_manifest.records)

    prefer_sidecar = base_manifest.sidecar is not None
    partial = final_path.with_name(f".{final_path.name}.partial")
    if partial.exists() and not resume:
        partial.unlink()
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(partial, flags, 0o644)

    codec_records: list[StreamedCodecRecord] = []
    cursor = 0
    try:
        for index, record in enumerate(selected):
            payload = read_expert_record(
                base_manifest,
                artifact_root,
                record.layer,
                record.expert,
                prefer_sidecar=prefer_sidecar,
                verify_hash=verify_source_hashes and record.sha256 is not None,
            )
            raw_sha = hashlib.sha256(payload).hexdigest()
            blob = encode_record_payload(payload, codec=codec)
            if verify_roundtrip:
                if decode_record_reference(blob, len(payload), codec=codec) != payload:
                    raise StreamedCodecError(
                        f"streamed codec round-trip failed for record "
                        f"({record.layer}, {record.expert})"
                    )
            offset = _align_up(cursor, alignment)
            reusable = False
            if resume and os.fstat(fd).st_size >= offset + len(blob):
                existing = os.pread(fd, len(blob), offset)
                reusable = (
                    len(existing) == len(blob)
                    and hashlib.sha256(existing).hexdigest()
                    == hashlib.sha256(blob).hexdigest()
                )
            if not reusable:
                if offset > os.fstat(fd).st_size:
                    os.ftruncate(fd, offset)
                position = offset
                view = memoryview(blob)
                while view:
                    written = os.pwrite(fd, view, position)
                    if written <= 0:
                        raise StreamedCodecError("short streamed codec write")
                    position += written
                    view = view[written:]
            codec_records.append(
                StreamedCodecRecord(
                    layer=record.layer,
                    expert=record.expert,
                    offset=offset,
                    length=len(blob),
                    raw_length=len(payload),
                    sha256=raw_sha,
                )
            )
            cursor = offset + len(blob)
            if progress and (index + 1) % 64 == 0:
                print(
                    f"streamed[{codec}] {index + 1}/{len(selected)} records, "
                    f"{cursor / 1024**2:.1f} MiB",
                    flush=True,
                )
        os.ftruncate(fd, cursor)
        os.fsync(fd)
    finally:
        os.close(fd)

    sidecar_sha = _hash_file(partial)
    os.replace(partial, final_path)
    directory_fd = os.open(final_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    manifest = StreamedCodecManifest(
        format=STREAMED_CODEC_FORMAT,
        model_key=base_manifest.model_key,
        codec=codec,
        file=relative_output,
        alignment=alignment,
        expert_count=expert_count,
        base_manifest_sha256=base_manifest.manifest_sha256 or "",
        size=cursor,
        sha256=sidecar_sha,
        records=tuple(codec_records),
        path=output_manifest,
    )
    manifest.validate()
    save_streamed_codec_manifest(manifest, output_manifest)
    return replace(manifest, path=output_manifest)


def _hash_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        while True:
            chunk = os.read(fd, chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


__all__ = [
    "STREAMED_CODEC_FORMAT",
    "STREAMED_CODECS",
    "DEFAULT_ALIGNMENT",
    "StreamedCodecError",
    "StreamedCodecRecord",
    "StreamedCodecManifest",
    "encode_record_payload",
    "decode_record_reference",
    "load_streamed_codec_manifest",
    "save_streamed_codec_manifest",
    "validate_against_base",
    "write_streamed_rans_sidecar",
    "resolve_artifact_member",
]
