"""q1 expert artifact: shadow-codec records as the streamed format (issue #51).

Converts an affine Q2/Q4 expert artifact into a ``q1`` artifact whose
records store each projection as ``{proj}.packed`` + ``{proj}.scales`` in
a shadow codec (``b1`` or ``t158`` — see :mod:`mtplx.expert_shadow`).
Halving (b1) or nearly halving (t158: 15/22.5 of Q2) the record bytes
speeds SSD miss service and lets the persistent cache hold proportionally
more experts. Probe verdict (research/glm52-shadow-codec-probe-*.json):
encode **t158** from Q2 sources; b1 is a no-go from Q2 because the Q2
grid's exact-zero level carries ~50% of the weight mass and a 1-bit sign
code cannot represent it.

The q1 artifact is self-describing (own manifest format) and is NOT yet
loadable by the streamed runtime — the streamed path's affine assumptions
are catalogued in research/streamed-q1-codec-gap-analysis.md. This module
provides the write/read/verify lane so the conversion burn can start as
soon as the gap items land.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from mtplx.expert_manifest import ExpertManifest, read_expert_record
from mtplx.expert_shadow import (
    SHADOW_GROUP,
    _B1_WORDS_PER_GROUP,
    _T158_BYTES_PER_GROUP,
    ShadowCodecError,
    decode_shadow,
    dequantize_record_projections,
    encode_shadow,
    normalize_shadow_codec,
    shadow_record_bytes,
)

Q1_FORMAT = "mtplx-expert-q1-v1"
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_PACKED_DTYPES = {"b1": "U32", "t158": "U8"}
_DTYPE_ITEM_SIZE = {"U32": 4, "U16": 2, "U8": 1}
_NUMPY_DTYPES = {"U32": "<u4", "U16": "<u2", "U8": "u1"}


class Q1ManifestError(ValueError):
    """Raised when a q1 artifact or its manifest is invalid."""


@dataclass(frozen=True)
class Q1Segment:
    component: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    def to_json(self) -> dict:
        return {
            "component": self.component,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "length": self.length,
        }


@dataclass(frozen=True)
class Q1Record:
    layer: int
    expert: int
    offset: int
    length: int
    sha256: str
    segments: tuple[Q1Segment, ...]

    def to_json(self) -> dict:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
            "segments": [segment.to_json() for segment in self.segments],
        }


@dataclass(frozen=True)
class Q1Manifest:
    format: str
    model_key: str
    codec: str
    group_size: int
    file: str
    source_model_key: str
    source_manifest_sha256: str | None
    records: tuple[Q1Record, ...]
    path: Path | None = field(default=None, compare=False)

    def record(self, layer: int, expert: int) -> Q1Record:
        key = (int(layer), int(expert))
        for entry in self.records:
            if (entry.layer, entry.expert) == key:
                return entry
        raise Q1ManifestError(f"q1 manifest has no record for {key}")

    def bin_path(self) -> Path:
        if self.path is None:
            raise Q1ManifestError("q1 manifest has no on-disk location")
        return self.path.parent / self.file

    def to_json(self) -> dict:
        return {
            "format": self.format,
            "model_key": self.model_key,
            "codec": self.codec,
            "group_size": self.group_size,
            "file": self.file,
            "source_model_key": self.source_model_key,
            "source_manifest_sha256": self.source_manifest_sha256,
            "records": [record.to_json() for record in self.records],
        }


def _packed_shape(codec: str, rows: int, cols: int) -> tuple[int, int]:
    groups = cols // SHADOW_GROUP
    if codec == "b1":
        return (rows, groups * _B1_WORDS_PER_GROUP)
    return (rows, groups * _T158_BYTES_PER_GROUP)


def encode_q1_record(
    codec: str, dense: dict[str, np.ndarray]
) -> tuple[bytes, tuple[Q1Segment, ...]]:
    """Encode one expert's fp32 projections into a q1 record payload."""

    chunks: list[bytes] = []
    segments: list[Q1Segment] = []
    cursor = 0
    for projection in _PROJECTIONS:
        try:
            weights = dense[projection]
        except KeyError:
            raise Q1ManifestError(f"missing projection {projection!r}") from None
        packed, scales = encode_shadow(codec, weights)
        for leaf, value, dtype in (
            ("packed", packed, _PACKED_DTYPES[codec]),
            ("scales", scales, "U16"),
        ):
            payload = value.tobytes()
            segments.append(
                Q1Segment(
                    component=f"{projection}.{leaf}",
                    dtype=dtype,
                    shape=tuple(int(size) for size in value.shape),
                    offset=cursor,
                    length=len(payload),
                )
            )
            chunks.append(payload)
            cursor += len(payload)
    return b"".join(chunks), tuple(segments)


def decode_q1_record(
    manifest: Q1Manifest, record: Q1Record, blob: bytes
) -> dict[str, np.ndarray]:
    """Decode a q1 record payload back to fp32 (out, in) per projection."""

    leaves: dict[str, np.ndarray] = {}
    for segment in record.segments:
        view = memoryview(blob)[segment.offset : segment.offset + segment.length]
        leaves[segment.component] = np.frombuffer(
            view, dtype=_NUMPY_DTYPES[segment.dtype]
        ).reshape(segment.shape)
    dense: dict[str, np.ndarray] = {}
    for projection in _PROJECTIONS:
        packed = leaves[f"{projection}.packed"]
        scales = leaves[f"{projection}.scales"]
        groups = scales.shape[1]
        dense[projection] = decode_shadow(
            manifest.codec, packed, scales, groups * SHADOW_GROUP
        )
    return dense


def read_q1_record(manifest: Q1Manifest, record: Q1Record) -> bytes:
    blob = manifest.bin_path().read_bytes()[
        record.offset : record.offset + record.length
    ]
    if len(blob) != record.length:
        raise Q1ManifestError(
            f"short q1 read for ({record.layer}, {record.expert})"
        )
    if hashlib.sha256(blob).hexdigest() != record.sha256:
        raise Q1ManifestError(
            f"q1 record hash mismatch: ({record.layer}, {record.expert})"
        )
    return blob


def convert_expert_q1(
    source_manifest: ExpertManifest,
    source_root: Path | str,
    output_dir: Path | str,
    *,
    codec: str,
    spec,
    layers: Iterable[int] | None = None,
    experts: Iterable[int] | None = None,
    limit: int | None = None,
    verify_source_hashes: bool = False,
    progress: bool = False,
) -> Q1Manifest:
    """Convert (a slice of) an affine expert artifact into a q1 artifact.

    ``layers``/``experts``/``limit`` bound the record set for smoke runs;
    omitting them converts every routed record (the full burn). Streaming:
    one record is dequantized, encoded, and written at a time.
    """

    codec = normalize_shadow_codec(codec)
    if codec is None:
        raise ShadowCodecError("q1 conversion requires a codec")
    import mlx.core as mx

    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_filter = None if layers is None else {int(layer) for layer in layers}
    expert_filter = None if experts is None else {int(expert) for expert in experts}
    selected = [
        record
        for record in source_manifest.records
        if (layer_filter is None or record.layer in layer_filter)
        and (expert_filter is None or record.expert in expert_filter)
    ]
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise Q1ManifestError("q1 conversion selected no records")

    expected_record_bytes = shadow_record_bytes(
        codec, spec.expert_source_parameters
    )
    bin_name = f"experts-q1-{codec}.bin"
    manifest_name = f"expert-manifest-q1-{codec}.json"
    started = time.perf_counter()
    records: list[Q1Record] = []
    bytes_in = 0
    with (output_dir / bin_name).open("wb") as out:
        cursor = 0
        for index, record in enumerate(selected):
            blob = read_expert_record(
                source_manifest,
                source_root,
                record.layer,
                record.expert,
                verify_hash=verify_source_hashes and record.sha256 is not None,
            )
            bytes_in += len(blob)
            dense = dequantize_record_projections(
                mx,
                record,
                blob,
                bits=source_manifest.quant_bits,
                group_size=source_manifest.quant_group_size,
            )
            payload, segments = encode_q1_record(codec, dense)
            if len(payload) != expected_record_bytes:
                raise Q1ManifestError(
                    f"record ({record.layer}, {record.expert}) encoded to "
                    f"{len(payload)} bytes; spec prices {expected_record_bytes}"
                )
            out.write(payload)
            records.append(
                Q1Record(
                    layer=record.layer,
                    expert=record.expert,
                    offset=cursor,
                    length=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    segments=segments,
                )
            )
            cursor += len(payload)
            if progress and (index + 1) % 64 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"q1[{codec}] {index + 1}/{len(selected)} records "
                    f"({cursor / 1024**2:.0f} MiB, {elapsed:.0f}s)",
                    flush=True,
                )
    manifest = Q1Manifest(
        format=Q1_FORMAT,
        model_key=f"{source_manifest.model_key}-q1{codec}",
        codec=codec,
        group_size=SHADOW_GROUP,
        file=bin_name,
        source_model_key=source_manifest.model_key,
        source_manifest_sha256=source_manifest.manifest_sha256,
        records=tuple(records),
        path=output_dir / manifest_name,
    )
    (output_dir / manifest_name).write_text(json.dumps(manifest.to_json()))
    return manifest


def load_q1_manifest(path: Path | str) -> Q1Manifest:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Q1ManifestError(f"cannot read q1 manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != Q1_FORMAT:
        raise Q1ManifestError(
            f"unsupported q1 manifest format {payload.get('format')!r}"
        )
    codec = normalize_shadow_codec(payload.get("codec"))
    if codec is None:
        raise Q1ManifestError("q1 manifest lists no codec")
    records: list[Q1Record] = []
    for record_obj in payload.get("records", ()):
        segments: list[Q1Segment] = []
        cursor = 0
        for segment_obj in record_obj.get("segments", ()):
            dtype = segment_obj.get("dtype")
            if dtype not in _DTYPE_ITEM_SIZE:
                raise Q1ManifestError(f"unsupported q1 segment dtype {dtype!r}")
            shape = tuple(int(value) for value in segment_obj.get("shape", ()))
            length = segment_obj.get("length")
            offset = segment_obj.get("offset")
            elements = 1
            for size in shape:
                elements *= size
            if length != elements * _DTYPE_ITEM_SIZE[dtype]:
                raise Q1ManifestError(
                    f"q1 segment {segment_obj.get('component')!r} length "
                    f"{length} does not match shape {shape}"
                )
            if offset != cursor:
                raise Q1ManifestError(
                    f"q1 segment {segment_obj.get('component')!r} is not "
                    "densely packed"
                )
            cursor += length
            segments.append(
                Q1Segment(
                    component=str(segment_obj.get("component")),
                    dtype=str(dtype),
                    shape=shape,
                    offset=int(offset),
                    length=int(length),
                )
            )
        if record_obj.get("length") != cursor:
            raise Q1ManifestError(
                f"q1 record ({record_obj.get('layer')}, "
                f"{record_obj.get('expert')}) does not cover its segments"
            )
        records.append(
            Q1Record(
                layer=int(record_obj.get("layer")),
                expert=int(record_obj.get("expert")),
                offset=int(record_obj.get("offset")),
                length=int(record_obj.get("length")),
                sha256=str(record_obj.get("sha256")),
                segments=tuple(segments),
            )
        )
    if not records:
        raise Q1ManifestError("q1 manifest lists no records")
    return Q1Manifest(
        format=Q1_FORMAT,
        model_key=str(payload.get("model_key")),
        codec=codec,
        group_size=int(payload.get("group_size", SHADOW_GROUP)),
        file=str(payload.get("file")),
        source_model_key=str(payload.get("source_model_key")),
        source_manifest_sha256=payload.get("source_manifest_sha256"),
        records=tuple(records),
        path=path,
    )


def verify_q1_against_source(
    manifest: Q1Manifest,
    source_manifest: ExpertManifest,
    source_root: Path | str,
    *,
    sample: int = 3,
) -> Iterator[tuple[int, int]]:
    """Re-encode sampled source records and require bitwise q1 equality."""

    import mlx.core as mx

    step = max(1, len(manifest.records) // max(1, sample))
    for record in manifest.records[::step][:sample]:
        blob = read_q1_record(manifest, record)
        source_record = source_manifest.record(record.layer, record.expert)
        source_blob = read_expert_record(
            source_manifest,
            Path(source_root),
            record.layer,
            record.expert,
            verify_hash=False,
        )
        dense = dequantize_record_projections(
            mx,
            source_record,
            source_blob,
            bits=source_manifest.quant_bits,
            group_size=source_manifest.quant_group_size,
        )
        payload, _segments = encode_q1_record(manifest.codec, dense)
        if payload != blob:
            raise Q1ManifestError(
                f"q1 record ({record.layer}, {record.expert}) does not "
                "round-trip against its source"
            )
        yield (record.layer, record.expert)
