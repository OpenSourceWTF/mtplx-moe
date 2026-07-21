"""GLM-5.2 Q1T artifact whose rANS lane streams are matmul-native."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from mtplx.expert_q1 import Q1Manifest
from mtplx.expert_rans import (
    LANES,
    LaneStreams,
    RANS_GUARD_BYTES,
    RANS_L,
    _HEADER_DTYPE,
    build_table,
    encode_bank,
    histogram,
    serialize_component,
    table_from_freq,
)


FUSED_RANS_FORMAT = "mtplx-glm52-q1t-fused-rans-v1"
FUSED_RANS_MODEL_KEY = "glm52-expert-q1t"
FUSED_RANS_CODEC = "rans32x-v1"
FUSED_RANS_UNIFORM_PACKED_CODEC = "rans32x-uniform-packed-v1"
FUSED_RANS_SOURCE_CODEC = "t158"
COMPONENT_ALIGNMENT = 16384
OUTPUT_TILE = LANES
_ENCODE_RECORD_CHUNK = 2048
_COMPONENTS = (
    "gate_proj.packed",
    "gate_proj.scales",
    "up_proj.packed",
    "up_proj.scales",
    "down_proj.packed",
    "down_proj.scales",
)
_DTYPE_BYTES = {"U8": 1, "U16": 2}
_SHA256_CHARS = frozenset("0123456789abcdef")


class FusedRansArtifactError(ValueError):
    """Raised when a fused GLM Q1T artifact is incomplete or incompatible."""


def _align(value: int) -> int:
    return -(-int(value) // COMPONENT_ALIGNMENT) * COMPONENT_ALIGNMENT


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


@dataclass(frozen=True)
class FusedRansComponent:
    component: str
    dtype: str
    shape: tuple[int, int]
    in_dim: int
    out_dim: int
    row_bytes: int
    offset: int
    length: int
    mapped_length: int
    raw_length: int
    sha256: str
    header_bytes: int
    frequency_offset: int
    directory_offset: int
    payload_offset: int
    payload_length: int
    guard_bytes: int
    record_count: int
    lanes: int
    per_lane: int

    def to_json(self) -> dict:
        return {
            "component": self.component,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "row_bytes": self.row_bytes,
            "offset": self.offset,
            "length": self.length,
            "mapped_length": self.mapped_length,
            "raw_length": self.raw_length,
            "sha256": self.sha256,
            "header_bytes": self.header_bytes,
            "frequency_offset": self.frequency_offset,
            "directory_offset": self.directory_offset,
            "payload_offset": self.payload_offset,
            "payload_length": self.payload_length,
            "guard_bytes": self.guard_bytes,
            "record_count": self.record_count,
            "lanes": self.lanes,
            "per_lane": self.per_lane,
        }


@dataclass(frozen=True)
class FusedRansLayer:
    layer: int
    components: tuple[FusedRansComponent, ...]

    def to_json(self) -> dict:
        return {
            "layer": self.layer,
            "components": [component.to_json() for component in self.components],
        }


@dataclass(frozen=True)
class Glm52Q1TFusedRansManifest:
    format: str
    model_key: str
    codec: str
    source_codec: str
    source_model_key: str
    source_manifest_sha256: str
    source_q1_parent_manifest_sha256: str | None
    source_q1_manifest_sha256: str
    file: str
    file_bytes: int
    file_sha256: str
    alignment: int
    output_tile: int
    expert_count: int
    routed_layers: tuple[int, ...]
    layers: tuple[FusedRansLayer, ...]
    path: Path | None = field(default=None, compare=False)

    def bin_path(self) -> Path:
        if self.path is None:
            raise FusedRansArtifactError("fused-rANS manifest has no location")
        return self.path.parent / self.file

    def layer_entry(self, layer: int) -> FusedRansLayer:
        for entry in self.layers:
            if entry.layer == int(layer):
                return entry
        raise FusedRansArtifactError(f"fused-rANS manifest has no layer {layer}")

    def to_json(self) -> dict:
        return {
            "format": self.format,
            "model_key": self.model_key,
            "codec": self.codec,
            "source_codec": self.source_codec,
            "source_model_key": self.source_model_key,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_q1_parent_manifest_sha256": (
                self.source_q1_parent_manifest_sha256
            ),
            "source_q1_manifest_sha256": self.source_q1_manifest_sha256,
            "file": self.file,
            "file_bytes": self.file_bytes,
            "file_sha256": self.file_sha256,
            "alignment": self.alignment,
            "output_tile": self.output_tile,
            "expert_count": self.expert_count,
            "routed_layers": list(self.routed_layers),
            "layers": [layer.to_json() for layer in self.layers],
        }


def _validate_source(source: Q1Manifest) -> None:
    accepted_keys = {
        FUSED_RANS_MODEL_KEY,
        "glm52-expert-q2-q1t158",
    }
    if source.model_key not in accepted_keys:
        raise FusedRansArtifactError(
            "fused artifact source must be glm52-expert-q1t; got "
            f"{source.model_key!r}"
        )
    if source.codec != FUSED_RANS_SOURCE_CODEC:
        raise FusedRansArtifactError(
            f"fused artifact source codec must be {FUSED_RANS_SOURCE_CODEC!r}"
        )
    if source.group_size != 64:
        raise FusedRansArtifactError("GLM Q1T source group size must be 64")
    if source.path is None:
        raise FusedRansArtifactError("GLM Q1T source manifest has no location")


def _layer_records(
    source: Q1Manifest,
    layer: int,
    expert_count: int,
):
    records = sorted(
        (record for record in source.records if record.layer == layer),
        key=lambda record: record.expert,
    )
    if [record.expert for record in records] != list(range(expert_count)):
        raise FusedRansArtifactError(
            f"layer {layer} must contain experts 0..{expert_count - 1}"
        )
    reference = records[0].segments
    if tuple(segment.component for segment in reference) != _COMPONENTS:
        raise FusedRansArtifactError(
            f"layer {layer} component order is not the GLM Q1T six-component schema"
        )
    for record in records:
        if record.segments != reference:
            raise FusedRansArtifactError(
                f"layer {layer} expert {record.expert} component schema differs"
            )
    return records, reference


def _component_geometry(segment) -> tuple[int, int, int]:
    if segment.dtype not in _DTYPE_BYTES or len(segment.shape) != 2:
        raise FusedRansArtifactError(
            f"unsupported component geometry for {segment.component!r}"
        )
    out_dim, row_elements = (int(value) for value in segment.shape)
    item_size = _DTYPE_BYTES[segment.dtype]
    row_bytes = row_elements * item_size
    if out_dim <= 0 or out_dim % OUTPUT_TILE:
        raise FusedRansArtifactError(
            f"component {segment.component!r} output rows must divide {OUTPUT_TILE}"
        )
    if segment.component.endswith(".packed"):
        if segment.dtype != "U8" or row_bytes % 13:
            raise FusedRansArtifactError(
                f"component {segment.component!r} is not t158 packed bytes"
            )
        groups = row_bytes // 13
    else:
        if segment.dtype != "U16":
            raise FusedRansArtifactError(
                f"component {segment.component!r} is not t158 scale bits"
            )
        groups = row_elements
    in_dim = groups * 64
    return in_dim, out_dim, row_bytes


def _encode_bank_bounded(
    segments: np.ndarray,
    table,
) -> LaneStreams:
    """Encode independent records in bounded batches without changing bytes."""

    record_count, seg_len = segments.shape
    chunk_size = int(_ENCODE_RECORD_CHUNK)
    if chunk_size < 1:
        raise FusedRansArtifactError("rANS encode record chunk must be positive")
    directory = np.empty((record_count, LANES), dtype=np.uint32)
    payload_parts: list[np.ndarray] = []
    payload_cursor = 0
    for start in range(0, record_count, chunk_size):
        stop = min(start + chunk_size, record_count)
        chunk = encode_bank(segments[start:stop], table)
        if payload_cursor + int(chunk.payload.size) >= 2**32:
            raise FusedRansArtifactError("rANS component payload exceeds u32 offsets")
        directory[start:stop] = chunk.directory + np.uint32(payload_cursor)
        payload_parts.append(chunk.payload)
        payload_cursor += int(chunk.payload.size)
    payload = np.concatenate(payload_parts)
    return LaneStreams(
        payload=payload,
        directory=directory,
        seg_len=int(seg_len),
        per_lane=int(seg_len // LANES),
        expert_count=int(record_count),
        lanes=LANES,
        ratio=float(record_count * seg_len / max(payload.size, 1)),
    )


def _encode_uniform_bank(segments: np.ndarray) -> LaneStreams:
    """Encode the fixed freq-16 model without generic per-symbol bookkeeping."""

    segments = np.asarray(segments, dtype=np.uint8)
    if segments.ndim != 2:
        raise FusedRansArtifactError("uniform rANS segments must be two-dimensional")
    record_count, seg_len = segments.shape
    if seg_len % LANES:
        raise FusedRansArtifactError(
            f"uniform rANS segment length {seg_len} is not divisible by {LANES}"
        )
    per_lane = seg_len // LANES
    rows = segments.reshape(record_count, LANES, per_lane)
    state = np.full((record_count, LANES), RANS_L, dtype=np.uint32)
    refills = np.empty_like(rows)
    clear_low_nibble = np.uint32(0xFFFFFFF0)
    for index in range(per_lane - 1, -1, -1):
        refills[:, :, index] = (state & np.uint32(0xFF)).astype(np.uint8)
        reduced = state >> np.uint32(8)
        symbols = rows[:, :, index].astype(np.uint32)
        state = (
            ((reduced & clear_low_nibble) << np.uint32(8))
            | (symbols << np.uint32(4))
            | (reduced & np.uint32(15))
        )
    lane_bytes = np.empty((record_count, LANES, per_lane + 4), dtype=np.uint8)
    lane_bytes[:, :, 0] = (state & np.uint32(0xFF)).astype(np.uint8)
    lane_bytes[:, :, 1] = (
        (state >> np.uint32(8)) & np.uint32(0xFF)
    ).astype(np.uint8)
    lane_bytes[:, :, 2] = (
        (state >> np.uint32(16)) & np.uint32(0xFF)
    ).astype(np.uint8)
    lane_bytes[:, :, 3] = (
        (state >> np.uint32(24)) & np.uint32(0xFF)
    ).astype(np.uint8)
    lane_bytes[:, :, 4:] = refills
    lane_length = per_lane + 4
    directory = (
        np.arange(record_count * LANES, dtype=np.uint32)
        * np.uint32(lane_length)
    ).reshape(record_count, LANES)
    payload = lane_bytes.reshape(-1)
    return LaneStreams(
        payload=payload,
        directory=directory,
        seg_len=int(seg_len),
        per_lane=int(per_lane),
        expert_count=int(record_count),
        lanes=LANES,
        ratio=float(record_count * seg_len / max(payload.size, 1)),
    )


def _discard_source_pages(source_map: np.memmap) -> None:
    """Release completed file-backed source pages after each converted layer."""

    source_map._mmap.madvise(mmap.MADV_DONTNEED)


def _encode_component(
    source_map: np.memmap,
    records,
    segment,
    *,
    expert_count: int,
    uniform_packed: bool = False,
) -> tuple[bytes, FusedRansComponent]:
    in_dim, out_dim, row_bytes = _component_geometry(segment)
    raw_bank = np.empty((expert_count, segment.length), dtype=np.uint8)
    for expert, record in enumerate(records):
        start = record.offset + segment.offset
        raw_bank[expert] = source_map[start : start + segment.length]
    tiles = out_dim // OUTPUT_TILE
    segments = raw_bank.reshape(
        expert_count * tiles,
        OUTPUT_TILE * row_bytes,
    )
    table = (
        table_from_freq(np.full(256, 16, dtype=np.uint32))
        if uniform_packed and segment.component.endswith(".packed")
        else build_table(histogram(segments.reshape(-1)))
    )
    streams = (
        _encode_uniform_bank(segments)
        if uniform_packed and segment.component.endswith(".packed")
        else _encode_bank_bounded(segments, table)
    )
    blob = serialize_component(streams, table)
    header_bytes = _HEADER_DTYPE.itemsize
    frequency_offset = header_bytes
    directory_offset = frequency_offset + 256 * 2
    payload_offset = directory_offset + streams.directory.size * 4
    payload_length = int(streams.payload.size)
    if payload_offset + payload_length + RANS_GUARD_BYTES != len(blob):
        raise FusedRansArtifactError(
            f"component {segment.component!r} container layout is inconsistent"
        )
    component = FusedRansComponent(
        component=segment.component,
        dtype=segment.dtype,
        shape=tuple(int(value) for value in segment.shape),
        in_dim=in_dim,
        out_dim=out_dim,
        row_bytes=row_bytes,
        offset=0,
        length=len(blob),
        mapped_length=_align(len(blob)),
        raw_length=expert_count * segment.length,
        sha256=_sha256_bytes(blob),
        header_bytes=header_bytes,
        frequency_offset=frequency_offset,
        directory_offset=directory_offset,
        payload_offset=payload_offset,
        payload_length=payload_length,
        guard_bytes=RANS_GUARD_BYTES,
        record_count=expert_count * tiles,
        lanes=LANES,
        per_lane=row_bytes,
    )
    return blob, component


def write_glm52_q1t_fused_rans_artifact(
    source: Q1Manifest,
    *,
    output_bin: Path | str,
    output_manifest: Path | str,
    layers: Iterable[int],
    expected_expert_count: int = 256,
    verify_record_hashes: bool = True,
    source_expert_manifest_sha256: str,
    resume: bool = False,
    uniform_packed: bool = False,
) -> Glm52Q1TFusedRansManifest:
    """Write a distinct page-aligned fused-rANS artifact from Q1T records."""

    _validate_source(source)
    if not isinstance(uniform_packed, bool):
        raise FusedRansArtifactError("uniform_packed must be a bool")
    output_codec = (
        FUSED_RANS_UNIFORM_PACKED_CODEC if uniform_packed else FUSED_RANS_CODEC
    )
    if not _is_sha256(source_expert_manifest_sha256):
        raise FusedRansArtifactError(
            "authoritative GLM Q1T expert manifest digest is invalid"
        )
    output_bin = Path(output_bin)
    output_manifest = Path(output_manifest)
    if output_bin.parent.resolve() != output_manifest.parent.resolve():
        raise FusedRansArtifactError(
            "fused binary and manifest must share one artifact directory"
        )
    partial = output_bin.with_name(output_bin.name + ".partial")
    progress_path = output_manifest.with_name(output_manifest.name + ".progress")
    finalizing = (
        resume
        and output_bin.exists()
        and not output_manifest.exists()
        and progress_path.exists()
        and not partial.exists()
    )
    if output_manifest.exists() or (output_bin.exists() and not finalizing):
        raise FusedRansArtifactError("fused-rANS output already exists")
    selected_layers = tuple(sorted({int(layer) for layer in layers}))
    if not selected_layers:
        raise FusedRansArtifactError("fused-rANS conversion selected no layers")
    expert_count = int(expected_expert_count)
    if expert_count <= 0:
        raise FusedRansArtifactError("expected_expert_count must be positive")
    source_path = source.bin_path()
    source_size = source_path.stat().st_size
    source_manifest_bytes = source.path.read_bytes()
    source_q1_manifest_sha256 = _sha256_bytes(source_manifest_bytes)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    if (partial.exists() or progress_path.exists()) and not resume:
        raise FusedRansArtifactError("fused-rANS partial output already exists")
    if not finalizing and partial.exists() != progress_path.exists():
        raise FusedRansArtifactError(
            "fused-rANS resume requires both partial data and progress metadata"
        )
    source_map: np.memmap | None = np.memmap(
        source_path, mode="r", dtype=np.uint8
    )
    completed: list[tuple[int, FusedRansComponent]] = []
    file_hasher = hashlib.sha256()
    cursor = 0
    expected_prefix = tuple(
        (layer, component) for layer in selected_layers for component in _COMPONENTS
    )

    def progress_value() -> dict:
        return {
            "format": f"{FUSED_RANS_FORMAT}-progress-v1",
            "output_file": output_bin.name,
            "source_q1_manifest_sha256": source_q1_manifest_sha256,
            "source_manifest_sha256": source_expert_manifest_sha256,
            "routed_layers": list(selected_layers),
            "expert_count": expert_count,
            "verify_record_hashes": bool(verify_record_hashes),
            "codec": output_codec,
            "cursor": cursor,
            "completed": [
                {"layer": layer, "component": component.to_json()}
                for layer, component in completed
            ],
        }

    def save_progress() -> None:
        temporary = progress_path.with_name(progress_path.name + ".tmp")
        with temporary.open("w") as out:
            json.dump(progress_value(), out, indent=1)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, progress_path)

    working_path = output_bin if finalizing else partial
    if working_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
            progress_layers = tuple(int(layer) for layer in progress["routed_layers"])
            progress_expert_count = int(progress["expert_count"])
            progress_cursor = int(progress["cursor"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise FusedRansArtifactError(
                f"invalid fused-rANS resume progress: {exc}"
            ) from exc
        if (
            progress.get("format") != f"{FUSED_RANS_FORMAT}-progress-v1"
            or progress.get("output_file") != output_bin.name
            or progress.get("source_q1_manifest_sha256")
            != source_q1_manifest_sha256
            or progress.get("source_manifest_sha256")
            != source_expert_manifest_sha256
            or progress_layers != selected_layers
            or progress_expert_count != expert_count
            or progress.get("verify_record_hashes") != bool(verify_record_hashes)
            or progress.get("codec", FUSED_RANS_CODEC) != output_codec
            or progress_cursor < 0
        ):
            raise FusedRansArtifactError(
                "fused-rANS resume progress does not match this conversion"
            )
        try:
            for item in progress["completed"]:
                layer = int(item["layer"])
                component = _parse_component(
                    item["component"], expert_count=expert_count
                )
                completed.append((layer, component))
        except (KeyError, TypeError, ValueError) as exc:
            raise FusedRansArtifactError(
                f"invalid fused-rANS completed component progress: {exc}"
            ) from exc
        actual_prefix = tuple(
            (layer, component.component) for layer, component in completed
        )
        if actual_prefix != expected_prefix[: len(actual_prefix)]:
            raise FusedRansArtifactError(
                "fused-rANS completed components are not a conversion prefix"
            )
        extent = 0
        for _layer, component in completed:
            if component.offset != extent:
                raise FusedRansArtifactError(
                    "fused-rANS resume component extents are inconsistent"
                )
            extent += component.mapped_length
        if progress_cursor != extent or working_path.stat().st_size < extent:
            raise FusedRansArtifactError(
                "fused-rANS partial extent does not match progress"
            )
        if finalizing and (
            actual_prefix != expected_prefix or working_path.stat().st_size != extent
        ):
            raise FusedRansArtifactError(
                "published fused-rANS binary is not a complete conversion"
            )
        cursor = extent
        if not finalizing:
            with working_path.open("r+b") as out:
                out.truncate(cursor)
        with working_path.open("rb") as existing:
            for _layer, component in completed:
                if existing.tell() != component.offset:
                    raise FusedRansArtifactError(
                        "fused-rANS resume component extents are inconsistent"
                    )
                component_hasher = hashlib.sha256()
                remaining = component.length
                while remaining:
                    chunk = existing.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise FusedRansArtifactError(
                            "fused-rANS completed component is truncated"
                        )
                    component_hasher.update(chunk)
                    file_hasher.update(chunk)
                    remaining -= len(chunk)
                if component_hasher.hexdigest() != component.sha256:
                    raise FusedRansArtifactError(
                        f"fused-rANS completed component {component.component} "
                        "hash mismatch"
                    )
                padding = component.mapped_length - component.length
                while padding:
                    chunk = existing.read(min(padding, 8 * 1024 * 1024))
                    if not chunk or any(chunk):
                        raise FusedRansArtifactError(
                            "fused-rANS completed component padding is corrupt"
                        )
                    file_hasher.update(chunk)
                    padding -= len(chunk)
    try:
        mode = "r+b" if working_path.exists() else "xb"
        with working_path.open(mode) as out:
            out.seek(cursor)
            if not progress_path.exists():
                save_progress()
            for layer in selected_layers:
                if source_map is None:  # pragma: no cover - construction invariant
                    raise AssertionError("source mapping was not installed")
                records, segments = _layer_records(source, layer, expert_count)
                for record in records:
                    if record.offset < 0 or record.offset + record.length > source_size:
                        raise FusedRansArtifactError(
                            f"source record ({layer}, {record.expert}) is out of range"
                        )
                    if verify_record_hashes:
                        with memoryview(source_map)[
                            record.offset : record.offset + record.length
                        ] as raw:
                            actual_sha256 = hashlib.sha256(raw).hexdigest()
                        if actual_sha256 != record.sha256:
                            raise FusedRansArtifactError(
                                f"source record hash mismatch: ({layer}, {record.expert})"
                            )
                for segment in segments:
                    key = (layer, segment.component)
                    completed_index = len(completed)
                    if key in expected_prefix[:completed_index]:
                        continue
                    blob, component = _encode_component(
                        source_map,
                        records,
                        segment,
                        expert_count=expert_count,
                        uniform_packed=uniform_packed,
                    )
                    padding_before = -cursor % COMPONENT_ALIGNMENT
                    if padding_before:
                        zeros = b"\0" * padding_before
                        out.write(zeros)
                        file_hasher.update(zeros)
                        cursor += padding_before
                    component = replace(component, offset=cursor)
                    out.write(blob)
                    file_hasher.update(blob)
                    cursor += len(blob)
                    padding_after = component.mapped_length - component.length
                    if padding_after:
                        zeros = b"\0" * padding_after
                        out.write(zeros)
                        file_hasher.update(zeros)
                        cursor += padding_after
                    out.flush()
                    os.fsync(out.fileno())
                    completed.append((layer, component))
                    save_progress()
                    del blob
                _discard_source_pages(source_map)
                source_map._mmap.close()
                source_map = None
                if layer != selected_layers[-1]:
                    source_map = np.memmap(source_path, mode="r", dtype=np.uint8)
            out.flush()
            os.fsync(out.fileno())
        if working_path == partial:
            os.replace(partial, output_bin)
    except BaseException:
        if not resume:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            try:
                progress_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if source_map is not None:
            source_map._mmap.close()
    entries = tuple(
        FusedRansLayer(
            layer=layer,
            components=tuple(
                component
                for completed_layer, component in completed
                if completed_layer == layer
            ),
        )
        for layer in selected_layers
    )
    if tuple(
        (entry.layer, component.component)
        for entry in entries
        for component in entry.components
    ) != expected_prefix:
        raise FusedRansArtifactError("fused-rANS conversion did not finish every component")
    manifest = Glm52Q1TFusedRansManifest(
        format=FUSED_RANS_FORMAT,
        model_key=FUSED_RANS_MODEL_KEY,
        codec=output_codec,
        source_codec=FUSED_RANS_SOURCE_CODEC,
        source_model_key=source.model_key,
        source_manifest_sha256=source_expert_manifest_sha256,
        source_q1_parent_manifest_sha256=source.source_manifest_sha256,
        source_q1_manifest_sha256=source_q1_manifest_sha256,
        file=output_bin.name,
        file_bytes=cursor,
        file_sha256=file_hasher.hexdigest(),
        alignment=COMPONENT_ALIGNMENT,
        output_tile=OUTPUT_TILE,
        expert_count=expert_count,
        routed_layers=selected_layers,
        layers=entries,
        path=output_manifest,
    )
    temporary_manifest = output_manifest.with_name(output_manifest.name + ".partial")
    try:
        with temporary_manifest.open("x") as out:
            json.dump(manifest.to_json(), out, indent=1)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary_manifest, output_manifest)
    except BaseException:
        try:
            temporary_manifest.unlink()
        except FileNotFoundError:
            pass
        raise
    progress_path.unlink(missing_ok=True)
    return manifest


def _parse_component(value: object, *, expert_count: int) -> FusedRansComponent:
    if not isinstance(value, dict):
        raise FusedRansArtifactError("fused-rANS component must be an object")
    try:
        component = FusedRansComponent(
            component=str(value["component"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            in_dim=int(value["in_dim"]),
            out_dim=int(value["out_dim"]),
            row_bytes=int(value["row_bytes"]),
            offset=int(value["offset"]),
            length=int(value["length"]),
            mapped_length=int(value["mapped_length"]),
            raw_length=int(value["raw_length"]),
            sha256=str(value["sha256"]),
            header_bytes=int(value["header_bytes"]),
            frequency_offset=int(value["frequency_offset"]),
            directory_offset=int(value["directory_offset"]),
            payload_offset=int(value["payload_offset"]),
            payload_length=int(value["payload_length"]),
            guard_bytes=int(value["guard_bytes"]),
            record_count=int(value["record_count"]),
            lanes=int(value["lanes"]),
            per_lane=int(value["per_lane"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FusedRansArtifactError(
            f"invalid fused-rANS component: {exc}"
        ) from exc
    in_dim, out_dim, row_bytes = _component_geometry(component)
    tiles = out_dim // OUTPUT_TILE
    expected_directory = component.header_bytes + 256 * 2
    expected_payload = expected_directory + expert_count * tiles * LANES * 4
    if (
        component.in_dim != in_dim
        or component.out_dim != out_dim
        or component.row_bytes != row_bytes
        or component.header_bytes != _HEADER_DTYPE.itemsize
        or component.frequency_offset != component.header_bytes
        or component.directory_offset != expected_directory
        or component.payload_offset != expected_payload
        or component.record_count != expert_count * tiles
        or component.lanes != LANES
        or component.per_lane != row_bytes
        or component.guard_bytes != RANS_GUARD_BYTES
        or component.length
        != component.payload_offset
        + component.payload_length
        + component.guard_bytes
        or component.raw_length != expert_count * component.shape[0] * row_bytes
        or component.offset % COMPONENT_ALIGNMENT
        or component.mapped_length != _align(component.length)
        or not _is_sha256(component.sha256)
    ):
        raise FusedRansArtifactError(
            f"fused-rANS component {component.component!r} geometry is inconsistent"
        )
    return component


def load_glm52_q1t_fused_rans_manifest(
    path: Path | str,
) -> Glm52Q1TFusedRansManifest:
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise FusedRansArtifactError(f"cannot read fused-rANS manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise FusedRansArtifactError("fused-rANS manifest must be an object")
    if value.get("format") != FUSED_RANS_FORMAT:
        raise FusedRansArtifactError("unsupported fused-rANS manifest format")
    if value.get("model_key") != FUSED_RANS_MODEL_KEY:
        raise FusedRansArtifactError("fused-rANS manifest is not GLM Q1T")
    codec = value.get("codec")
    if codec not in (FUSED_RANS_CODEC, FUSED_RANS_UNIFORM_PACKED_CODEC):
        raise FusedRansArtifactError(
            "fused-rANS manifest codec is unsupported"
        )
    if value.get("source_codec") != FUSED_RANS_SOURCE_CODEC:
        raise FusedRansArtifactError("fused-rANS source codec is not t158")
    try:
        expert_count = int(value["expert_count"])
        alignment = int(value["alignment"])
        output_tile = int(value["output_tile"])
        file_bytes = int(value["file_bytes"])
        routed_layers = tuple(int(layer) for layer in value["routed_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FusedRansArtifactError(f"invalid fused-rANS manifest: {exc}") from exc
    if expert_count <= 0 or alignment != COMPONENT_ALIGNMENT or output_tile != OUTPUT_TILE:
        raise FusedRansArtifactError("fused-rANS manifest geometry is incompatible")
    layers: list[FusedRansLayer] = []
    extent_cursor = 0
    for layer_value in value.get("layers", ()):
        if not isinstance(layer_value, dict):
            raise FusedRansArtifactError("fused-rANS layer must be an object")
        components = tuple(
            _parse_component(item, expert_count=expert_count)
            for item in layer_value.get("components", ())
        )
        if tuple(component.component for component in components) != _COMPONENTS:
            raise FusedRansArtifactError("fused-rANS layer is missing components")
        for component in components:
            if component.offset != extent_cursor:
                raise FusedRansArtifactError(
                    "fused-rANS component extents are not densely page-aligned"
                )
            extent_cursor += component.mapped_length
        layers.append(
            FusedRansLayer(layer=int(layer_value["layer"]), components=components)
        )
    if tuple(layer.layer for layer in layers) != routed_layers or not layers:
        raise FusedRansArtifactError("fused-rANS routed layer coverage is inconsistent")
    file_name = value.get("file")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise FusedRansArtifactError("fused-rANS file must be a safe relative name")
    if file_bytes != extent_cursor or not _is_sha256(value.get("file_sha256")):
        raise FusedRansArtifactError("fused-rANS file extent or hash is invalid")
    source_q1_hash = value.get("source_q1_manifest_sha256")
    if not _is_sha256(source_q1_hash):
        raise FusedRansArtifactError("fused-rANS source manifest hash is invalid")
    source_manifest_hash = value.get("source_manifest_sha256")
    if not _is_sha256(source_manifest_hash):
        raise FusedRansArtifactError("fused-rANS base manifest hash is invalid")
    source_q1_parent_hash = value.get("source_q1_parent_manifest_sha256")
    if source_q1_parent_hash is not None and not _is_sha256(source_q1_parent_hash):
        raise FusedRansArtifactError("fused-rANS Q1 parent manifest hash is invalid")
    return Glm52Q1TFusedRansManifest(
        format=FUSED_RANS_FORMAT,
        model_key=FUSED_RANS_MODEL_KEY,
        codec=str(codec),
        source_codec=FUSED_RANS_SOURCE_CODEC,
        source_model_key=str(value.get("source_model_key")),
        source_manifest_sha256=str(source_manifest_hash),
        source_q1_parent_manifest_sha256=source_q1_parent_hash,
        source_q1_manifest_sha256=str(source_q1_hash),
        file=file_name,
        file_bytes=file_bytes,
        file_sha256=str(value["file_sha256"]),
        alignment=alignment,
        output_tile=output_tile,
        expert_count=expert_count,
        routed_layers=routed_layers,
        layers=tuple(layers),
        path=path,
    )
