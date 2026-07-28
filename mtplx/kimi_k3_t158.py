"""Resumable Kimi K3 Q2_K to MTPLX t158 artifact serialization.

The converter deliberately has two binary paths:

* routed merged expert tensors are decoded one projection at a time and
  immediately encoded into the existing t158 record representation;
* BF16/F32 residents are framed as safetensors without numerical conversion.

All source and artifact invariants are checked before a conversion lane is
installed.  Resume journals bind both the exact source tensor slices and each
durable output record.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Protocol

import numpy as np

from mtplx.expert_manifest import (
    EMPTY_SHA256,
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    SidecarInfo,
    SidecarPart,
    TensorSegment,
    verify_expert_manifest,
)
from mtplx.expert_shadow import encode_t158
from mtplx.kimi_k3_gguf import (
    GGML_TYPE_BF16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    GGUFFile,
    GGUFTensor,
    KIMI_K3_RESIDENT_BYTES,
    KIMI_K3_ROUTED_BYTES,
    KIMI_K3_SOURCE_REVISION,
    KimiK3Inventory,
    dequantize_q2_k,
    inspect_kimi_k3_source,
    open_verified_gguf,
    tensor_nbytes,
)

KIMI_K3_EXPERT_COUNT = 896
KIMI_K3_LAYER_COUNT = 92
KIMI_K3_GATE_SHAPE = (3072, 3584)
KIMI_K3_UP_SHAPE = (3072, 3584)
KIMI_K3_DOWN_SHAPE = (3584, 3072)
KIMI_K3_T158_RECORD_BYTES = 7_741_440
KIMI_K3_T158_LAYER_BYTES = 6_936_330_240
KIMI_K3_T158_ROUTED_BYTES = 638_142_382_080
KIMI_K3_TEXT_RESIDENT_BYTES = 113_509_540_864
KIMI_K3_TEXT_ARTIFACT_BYTES = 751_651_922_944
KIMI_K3_OFFICIAL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
KIMI_K3_SOURCE_REPO = "GrEarl/Kimi-K3-GGUF"
KIMI_K3_OFFICIAL_REPO = "moonshotai/Kimi-K3"

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_GGUF_PROJECTIONS = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}
_RESIDENT_DTYPES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_BF16: "BF16",
}
_COPY_CHUNK_BYTES = 16 * 1024 * 1024
_JOURNAL_FORMAT = "mtplx-kimi-k3-t158-layer-journal-v1"
_RESIDENT_RECEIPT_FORMAT = "mtplx-kimi-k3-resident-copy-v1"


class _Digest(Protocol):
    def update(self, payload: bytes) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class KimiK3Layout:
    """Injectable geometry; defaults are the exact production K3 layout."""

    expert_count: int = KIMI_K3_EXPERT_COUNT
    layer_count: int = KIMI_K3_LAYER_COUNT
    gate_shape: tuple[int, int] = KIMI_K3_GATE_SHAPE
    up_shape: tuple[int, int] = KIMI_K3_UP_SHAPE
    down_shape: tuple[int, int] = KIMI_K3_DOWN_SHAPE

    def __post_init__(self) -> None:
        if self.expert_count <= 0 or self.layer_count <= 0:
            raise ValueError("Kimi K3 layout counts must be positive")
        for name, shape in self.projection_shapes.items():
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError(f"{name} shape must contain two positive axes")
            if shape[1] % 64:
                raise ValueError(f"{name} input axis must divide into groups of 64")
            if shape[0] * shape[1] % 256:
                raise ValueError(f"{name} values must divide into Q2_K blocks")

    @property
    def projection_shapes(self) -> dict[str, tuple[int, int]]:
        return {
            "gate_proj": self.gate_shape,
            "up_proj": self.up_shape,
            "down_proj": self.down_shape,
        }

    @property
    def record_bytes(self) -> int:
        return sum(
            _t158_projection_bytes(shape) for shape in self.projection_shapes.values()
        )

    @property
    def layer_bytes(self) -> int:
        return self.expert_count * self.record_bytes


@dataclass(frozen=True, slots=True)
class EncodedSegment:
    component: str
    tensor: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "tensor": self.tensor,
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class EncodedExpertRecord:
    layer: int
    expert: int
    shard: str
    record_offset: int
    logical_bytes: int
    segments: tuple[EncodedSegment, ...]
    sha256: str
    payload: bytes | None = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "shard": self.shard,
            "record_offset": self.record_offset,
            "logical_bytes": self.logical_bytes,
            "segments": [segment.to_dict() for segment in self.segments],
            "sha256": self.sha256,
        }

    def metadata_only(self) -> EncodedExpertRecord:
        return EncodedExpertRecord(
            layer=self.layer,
            expert=self.expert,
            shard=self.shard,
            record_offset=self.record_offset,
            logical_bytes=self.logical_bytes,
            segments=self.segments,
            sha256=self.sha256,
            payload=None,
        )


@dataclass(frozen=True, slots=True)
class ConvertedLayer:
    layer: int
    path: Path
    journal_path: Path
    logical_bytes: int
    sha256: str
    records: tuple[EncodedExpertRecord, ...]


def _t158_projection_bytes(shape: tuple[int, int]) -> int:
    rows, cols = shape
    groups = rows * cols // 64
    return groups * 15


def _sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_prefix(path: Path, length: int) -> _Digest:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as source:
        while remaining:
            chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"truncated file while hashing prefix of {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _same_regular_inode(first: Path, second: Path) -> bool:
    first_status = first.lstat()
    second_status = second.lstat()
    return (
        stat.S_ISREG(first_status.st_mode)
        and stat.S_ISREG(second_status.st_mode)
        and (first_status.st_dev, first_status.st_ino)
        == (second_status.st_dev, second_status.st_ino)
    )


def _adopt_without_overwrite(partial: Path, final: Path) -> None:
    try:
        os.link(partial, final, follow_symlinks=False)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing output {final}") from exc
    partial.unlink()


def _pread_exact(descriptor: int, offset: int, length: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    cursor = offset
    remaining = length
    while remaining:
        chunk = os.pread(descriptor, min(remaining, _COPY_CHUNK_BYTES), cursor)
        if not chunk:
            raise ValueError(f"truncated source while reading {label}")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_projection(
    projection: str,
    weights: np.ndarray,
    *,
    layer: int,
    expert: int,
    shard: str,
    offset: int,
) -> tuple[bytes, tuple[EncodedSegment, EncodedSegment]]:
    if not np.isfinite(weights).all():
        raise ValueError(f"{projection} contains non-finite decoded values")
    packed, scales = encode_t158(weights)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype="<u2")
    scale_values = (scales.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.isfinite(scale_values).all():
        raise ValueError(f"{projection} encoded scale contains non-finite values")
    packed_bytes = packed.tobytes(order="C")
    scale_bytes = scales.tobytes(order="C")
    tensor_prefix = f"model.layers.{layer}.mlp.switch_mlp.experts.{expert}.{projection}"
    packed_segment = EncodedSegment(
        component=f"{projection}.packed",
        tensor=f"{tensor_prefix}.packed",
        shard=shard,
        offset=offset,
        length=len(packed_bytes),
        dtype="U8",
        shape=tuple(int(axis) for axis in packed.shape),
        sha256=_sha256_bytes(packed_bytes),
    )
    scale_segment = EncodedSegment(
        component=f"{projection}.scales",
        tensor=f"{tensor_prefix}.scales",
        shard=shard,
        offset=offset + len(packed_bytes),
        length=len(scale_bytes),
        dtype="BF16",
        shape=tuple(int(axis) for axis in scales.shape),
        sha256=_sha256_bytes(scale_bytes),
    )
    return packed_bytes + scale_bytes, (packed_segment, scale_segment)


def encode_expert_record(
    projections: Mapping[str, np.ndarray],
    *,
    layer: int,
    expert: int,
    shard: str,
    record_offset: int,
) -> EncodedExpertRecord:
    """Encode gate/up/down in fixed component order using the t158 codec."""

    if set(projections) != set(_PROJECTIONS):
        raise ValueError("expert record requires exactly gate_proj/up_proj/down_proj")
    if layer < 0 or expert < 0 or record_offset < 0:
        raise ValueError("record layer, expert, and offset must be non-negative")
    cursor = record_offset
    payloads: list[bytes] = []
    segments: list[EncodedSegment] = []
    for projection in _PROJECTIONS:
        weights = np.asarray(projections[projection])
        encoded, encoded_segments = _encode_projection(
            projection,
            weights,
            layer=layer,
            expert=expert,
            shard=shard,
            offset=cursor,
        )
        payloads.append(encoded)
        segments.extend(encoded_segments)
        cursor += len(encoded)
    payload = b"".join(payloads)
    return EncodedExpertRecord(
        layer=layer,
        expert=expert,
        shard=shard,
        record_offset=record_offset,
        logical_bytes=len(payload),
        segments=tuple(segments),
        sha256=_sha256_bytes(payload),
        payload=payload,
    )


def _source_identity(
    source: GGUFFile,
    tensors: Mapping[str, GGUFTensor],
    *,
    revision: str,
) -> dict[str, Any]:
    if source.identity is None:
        raise ValueError("Kimi K3 source has no inspected file identity")
    return {
        "revision": revision,
        "path": str(source.path.resolve()),
        "file_identity": {
            "device": source.identity.device,
            "inode": source.identity.inode,
            "size": source.identity.size,
            "mtime_ns": source.identity.mtime_ns,
            "ctime_ns": source.identity.ctime_ns,
        },
        "data_offset": source.data_offset,
        "tensors": [
            {
                "projection": projection,
                "name": tensor.name,
                "dims": list(tensor.dims),
                "ggml_type": tensor.ggml_type,
                "offset": tensor.offset,
                "length": tensor_nbytes(tensor),
            }
            for projection, tensor in sorted(tensors.items())
        ],
    }


def _layer_tensors(
    source: GGUFFile,
    inventory: KimiK3Inventory,
    *,
    layer: int,
    layout: KimiK3Layout,
) -> dict[str, GGUFTensor]:
    if layer not in inventory.layers:
        raise ValueError(f"layer {layer} is absent from the Kimi K3 inventory")
    if source not in inventory.files:
        raise ValueError("layer source is absent from the Kimi K3 inventory")
    result: dict[str, GGUFTensor] = {}
    for projection, gguf_projection in _GGUF_PROJECTIONS.items():
        name = f"blk.{layer}.ffn_{gguf_projection}_exps.weight"
        matches = [tensor for tensor in inventory.expert_tensors if tensor.name == name]
        if len(matches) != 1:
            raise ValueError(f"layer {layer} requires exactly one tensor {name!r}")
        tensor = matches[0]
        try:
            owned = source.tensor(name)
        except KeyError as exc:
            raise ValueError(
                f"tensor {name!r} is not owned by the selected source"
            ) from exc
        if owned != tensor:
            raise ValueError(f"inventory identity differs for tensor {name!r}")
        expected_shape = (
            layout.expert_count,
            *layout.projection_shapes[projection],
        )
        if tensor.ggml_type != GGML_TYPE_Q2_K or tensor.shape != expected_shape:
            raise ValueError(
                f"tensor {name!r} must be Q2_K with shape {expected_shape}, "
                f"got type={tensor.ggml_type}, shape={tensor.shape}"
            )
        result[projection] = tensor
    return result


def _journal_header(
    source: GGUFFile,
    tensors: Mapping[str, GGUFTensor],
    inventory: KimiK3Inventory,
    *,
    layer: int,
    output_file: str,
    layout: KimiK3Layout,
) -> dict[str, Any]:
    return {
        "format": _JOURNAL_FORMAT,
        "layer": layer,
        "output_file": output_file,
        "source": _source_identity(
            source,
            tensors,
            revision=inventory.revision,
        ),
        "layout": {
            "expert_count": layout.expert_count,
            "layer_count": layout.layer_count,
            "projection_shapes": {
                name: list(shape) for name, shape in layout.projection_shapes.items()
            },
            "record_bytes": layout.record_bytes,
            "layer_bytes": layout.layer_bytes,
        },
    }


def _record_from_dict(value: object) -> EncodedExpertRecord:
    if not isinstance(value, dict):
        raise ValueError("journal record must be an object")
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("journal record segments must be an array")
    segments: list[EncodedSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("journal segment must be an object")
        segments.append(
            EncodedSegment(
                component=str(raw["component"]),
                tensor=str(raw["tensor"]),
                shard=str(raw["shard"]),
                offset=int(raw["offset"]),
                length=int(raw["length"]),
                dtype=str(raw["dtype"]),
                shape=tuple(int(axis) for axis in raw["shape"]),
                sha256=str(raw["sha256"]),
            )
        )
    return EncodedExpertRecord(
        layer=int(value["layer"]),
        expert=int(value["expert"]),
        shard=str(value["shard"]),
        record_offset=int(value["record_offset"]),
        logical_bytes=int(value["logical_bytes"]),
        segments=tuple(segments),
        sha256=str(value["sha256"]),
        payload=None,
    )


def _parse_journal(path: Path) -> tuple[list[tuple[dict[str, Any], int]], int]:
    raw = path.read_bytes()
    parsed: list[tuple[dict[str, Any], int]] = []
    cursor = 0
    for line in raw.splitlines(keepends=True):
        next_cursor = cursor + len(line)
        if not line.endswith(b"\n"):
            return parsed, cursor
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid complete Kimi K3 journal entry") from exc
        if not isinstance(value, dict):
            raise ValueError("Kimi K3 journal entries must be objects")
        parsed.append((value, next_cursor))
        cursor = next_cursor
    return parsed, cursor


def _record_source_state(
    source_fd: int,
    source: GGUFFile,
    tensors: Mapping[str, GGUFTensor],
    *,
    expert: int,
    layout: KimiK3Layout,
) -> tuple[dict[str, Any], ...]:
    states: list[dict[str, Any]] = []
    for projection in _PROJECTIONS:
        tensor = tensors[projection]
        merged_bytes = tensor_nbytes(tensor)
        if merged_bytes % layout.expert_count:
            raise ValueError(f"tensor {tensor.name!r} is not expert-contiguous")
        expert_bytes = merged_bytes // layout.expert_count
        start, _end = source.tensor_span(tensor)
        offset = start + expert * expert_bytes
        payload = _pread_exact(
            source_fd,
            offset,
            expert_bytes,
            label=f"{projection} expert {expert}",
        )
        states.append(
            {
                "projection": projection,
                "tensor": tensor.name,
                "offset": offset,
                "length": expert_bytes,
                "sha256": _sha256_bytes(payload),
            }
        )
    return tuple(states)


def _decode_projection(
    source_fd: int,
    source: GGUFFile,
    tensor: GGUFTensor,
    *,
    projection: str,
    expert: int,
    layout: KimiK3Layout,
) -> tuple[np.ndarray, dict[str, Any]]:
    merged_bytes = tensor_nbytes(tensor)
    if merged_bytes % layout.expert_count:
        raise ValueError(f"tensor {tensor.name!r} is not expert-contiguous")
    expert_bytes = merged_bytes // layout.expert_count
    start, _end = source.tensor_span(tensor)
    offset = start + expert * expert_bytes
    raw = _pread_exact(
        source_fd,
        offset,
        expert_bytes,
        label=f"{projection} expert {expert}",
    )
    shape = layout.projection_shapes[projection]
    decoded = dequantize_q2_k(raw, value_count=shape[0] * shape[1]).reshape(shape)
    state = {
        "projection": projection,
        "tensor": tensor.name,
        "offset": offset,
        "length": expert_bytes,
        "sha256": _sha256_bytes(raw),
    }
    return decoded, state


def _validate_record_geometry(
    record: EncodedExpertRecord,
    *,
    layer: int,
    expert: int,
    shard: str,
    layout: KimiK3Layout,
) -> None:
    def is_sha256(value: str) -> bool:
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )

    expected_offset = expert * layout.record_bytes
    if (
        record.layer != layer
        or record.expert != expert
        or record.shard != shard
        or record.record_offset != expected_offset
        or record.logical_bytes != layout.record_bytes
        or len(record.segments) != 6
        or not is_sha256(record.sha256)
    ):
        raise ValueError("journal record geometry is not the expected fixed layout")
    cursor = expected_offset
    expected_segments: list[dict[str, Any]] = []
    for projection in _PROJECTIONS:
        rows, cols = layout.projection_shapes[projection]
        groups = cols // 64
        tensor_prefix = (
            f"model.layers.{layer}.mlp.switch_mlp.experts.{expert}.{projection}"
        )
        packed_shape = (rows, groups * 13)
        scale_shape = (rows, groups)
        for leaf, dtype, shape, length in (
            ("packed", "U8", packed_shape, rows * groups * 13),
            ("scales", "BF16", scale_shape, rows * groups * 2),
        ):
            expected_segments.append(
                {
                    "component": f"{projection}.{leaf}",
                    "tensor": f"{tensor_prefix}.{leaf}",
                    "shard": shard,
                    "offset": cursor,
                    "length": length,
                    "dtype": dtype,
                    "shape": shape,
                }
            )
            cursor += length
    for segment, expected in zip(
        record.segments,
        expected_segments,
        strict=True,
    ):
        actual = {
            "component": segment.component,
            "tensor": segment.tensor,
            "shard": segment.shard,
            "offset": segment.offset,
            "length": segment.length,
            "dtype": segment.dtype,
            "shape": segment.shape,
        }
        if actual != expected or not is_sha256(segment.sha256):
            raise ValueError(
                f"journal segment metadata is invalid for {expected['component']}"
            )
    if cursor != expected_offset + layout.record_bytes:
        raise ValueError("journal record segment size is invalid")


def _resume_layer(
    source_fd: int,
    source: GGUFFile,
    tensors: Mapping[str, GGUFTensor],
    *,
    partial: Path,
    journal: Path,
    header: dict[str, Any],
    layer: int,
    shard: str,
    layout: KimiK3Layout,
) -> tuple[list[EncodedExpertRecord], int]:
    entries, complete_end = _parse_journal(journal)
    if not entries or entries[0][0] != header:
        raise ValueError("Kimi K3 resume source identity or layout changed")
    record_entries = entries[1:]
    if record_entries and "completion" in record_entries[-1][0]:
        record_entries = record_entries[:-1]
    if any("completion" in entry for entry, _end in record_entries):
        raise ValueError("Kimi K3 completion journal entry is not last")
    records: list[EncodedExpertRecord] = []
    valid_journal_end = entries[0][1]
    with partial.open("r+b") as output:
        for ordinal, (entry, journal_end) in enumerate(record_entries):
            if ordinal >= layout.expert_count:
                raise ValueError("Kimi K3 journal contains too many expert records")
            if entry.get("ordinal") != ordinal:
                raise ValueError("Kimi K3 journal is not a contiguous prefix")
            record = _record_from_dict(entry.get("output"))
            _validate_record_geometry(
                record,
                layer=layer,
                expert=ordinal,
                shard=shard,
                layout=layout,
            )
            current_source = _record_source_state(
                source_fd,
                source,
                tensors,
                expert=ordinal,
                layout=layout,
            )
            if entry.get("source_components") != list(current_source):
                raise ValueError(
                    f"Kimi K3 source hash changed for completed expert {ordinal}"
                )
            output.seek(record.record_offset)
            payload = output.read(record.logical_bytes)
            if (
                len(payload) != record.logical_bytes
                or _sha256_bytes(payload) != record.sha256
            ):
                break
            if any(
                _sha256_bytes(
                    payload[
                        segment.offset - record.record_offset : segment.offset
                        - record.record_offset
                        + segment.length
                    ]
                )
                != segment.sha256
                for segment in record.segments
            ):
                break
            records.append(record)
            valid_journal_end = journal_end
        target_size = len(records) * layout.record_bytes
        if output.seek(0, os.SEEK_END) != target_size:
            output.truncate(target_size)
            output.flush()
            os.fsync(output.fileno())
    journal_size = journal.stat().st_size
    if journal_size != valid_journal_end or complete_end != journal_size:
        with journal.open("r+b") as handle:
            handle.truncate(valid_journal_end)
            handle.flush()
            os.fsync(handle.fileno())
    return records, valid_journal_end


def _adopt_completed_layer(
    source_fd: int,
    source: GGUFFile,
    tensors: Mapping[str, GGUFTensor],
    *,
    final: Path,
    journal: Path,
    header: dict[str, Any],
    layer: int,
    layout: KimiK3Layout,
) -> ConvertedLayer:
    entries, complete_end = _parse_journal(journal)
    if (
        not entries
        or entries[0][0] != header
        or complete_end != journal.stat().st_size
        or len(entries) != layout.expert_count + 2
    ):
        raise ValueError("completed Kimi K3 layer has no matching complete journal")
    completion = entries[-1][0].get("completion")
    if not isinstance(completion, dict):
        raise ValueError("completed Kimi K3 layer has no completion receipt")
    if final.stat().st_size != layout.layer_bytes:
        raise ValueError("completed Kimi K3 layer has the wrong byte size")
    records: list[EncodedExpertRecord] = []
    layer_digest = hashlib.sha256()
    with final.open("rb") as output:
        for ordinal, (entry, _end) in enumerate(entries[1:-1]):
            if entry.get("ordinal") != ordinal:
                raise ValueError("completed Kimi K3 journal is not contiguous")
            record = _record_from_dict(entry.get("output"))
            _validate_record_geometry(
                record,
                layer=layer,
                expert=ordinal,
                shard=final.name,
                layout=layout,
            )
            current_source = _record_source_state(
                source_fd,
                source,
                tensors,
                expert=ordinal,
                layout=layout,
            )
            if entry.get("source_components") != list(current_source):
                raise ValueError(
                    f"Kimi K3 source hash changed for completed expert {ordinal}"
                )
            payload = output.read(record.logical_bytes)
            if _sha256_bytes(payload) != record.sha256:
                raise ValueError(
                    f"completed Kimi K3 output hash changed for expert {ordinal}"
                )
            for segment in record.segments:
                relative = segment.offset - record.record_offset
                segment_payload = payload[relative : relative + segment.length]
                if _sha256_bytes(segment_payload) != segment.sha256:
                    raise ValueError(
                        "completed Kimi K3 segment hash changed for "
                        f"{segment.component}"
                    )
            layer_digest.update(payload)
            records.append(record)
    if completion != {
        "record_count": layout.expert_count,
        "size": layout.layer_bytes,
        "sha256": layer_digest.hexdigest(),
    }:
        raise ValueError("completed Kimi K3 whole-part hash does not match")
    return ConvertedLayer(
        layer=layer,
        path=final,
        journal_path=journal,
        logical_bytes=layout.layer_bytes,
        sha256=layer_digest.hexdigest(),
        records=tuple(records),
    )


def convert_layer(
    source: GGUFFile,
    output_dir: Path,
    inventory: KimiK3Inventory,
    *,
    layer: int,
    resume: bool,
    layout: KimiK3Layout | None = None,
) -> ConvertedLayer:
    """Convert one merged-Q2_K K3 layer into one durable t158 sidecar part."""

    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    layout = layout or KimiK3Layout()
    if not 1 <= layer <= layout.layer_count:
        raise ValueError(f"layer must be in 1..{layout.layer_count}")
    tensors = _layer_tensors(
        source,
        inventory,
        layer=layer,
        layout=layout,
    )
    output_dir = Path(output_dir)
    source_path = source.path.resolve()
    if output_dir.resolve() in {source_path, source_path.parent}:
        raise ValueError("Kimi K3 source and output must be separate paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / (
        f"experts-t158-layer-{layer:03d}-of-{layout.layer_count:03d}.bin"
    )
    partial = final.with_name(final.name + ".partial")
    journal = final.with_name(final.name + ".journal.jsonl")
    for candidate in (final, partial, journal):
        if candidate.resolve() == source_path:
            raise ValueError("Kimi K3 source and output must be separate paths")
        if _entry_exists(candidate) and not stat.S_ISREG(candidate.lstat().st_mode):
            raise ValueError(
                f"Kimi K3 conversion member is not a regular file: {candidate}"
            )
    header = _journal_header(
        source,
        tensors,
        inventory,
        layer=layer,
        output_file=final.name,
        layout=layout,
    )

    source_fd = open_verified_gguf(source)
    try:
        source_status = os.fstat(source_fd)
        if not stat.S_ISREG(source_status.st_mode):
            raise ValueError("Kimi K3 source must be a regular file")
        if source_status.st_size != source.file_size:
            raise ValueError("Kimi K3 source size changed after GGUF inspection")
        if _entry_exists(final):
            if not resume:
                raise ValueError(
                    "completed Kimi K3 layer exists and resume is disabled"
                )
            if _entry_exists(partial):
                if not _same_regular_inode(final, partial):
                    raise ValueError(
                        "completed Kimi K3 layer and partial do not share one inode"
                    )
            if not _entry_exists(journal):
                raise ValueError("completed Kimi K3 layer is missing its journal")
            adopted = _adopt_completed_layer(
                source_fd,
                source,
                tensors,
                final=final,
                journal=journal,
                header=header,
                layer=layer,
                layout=layout,
            )
            if _entry_exists(partial):
                partial.unlink()
                _fsync_directory(output_dir)
            return adopted

        if _entry_exists(partial) and not _entry_exists(journal):
            partial_status = partial.lstat()
            if (
                not resume
                or partial_status.st_size != 0
                or partial_status.st_nlink != 1
            ):
                raise ValueError(
                    "lone Kimi K3 partial is not a safe empty initial partial"
                )
            with journal.open("xb") as handle:
                handle.write(_canonical_json(header) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(output_dir)
        if _entry_exists(partial) != _entry_exists(journal):
            raise ValueError("Kimi K3 resume requires both partial and journal files")
        if _entry_exists(partial) and not resume:
            raise ValueError("partial Kimi K3 layer exists and resume is disabled")
        if not _entry_exists(partial):
            with partial.open("xb") as output:
                output.flush()
                os.fsync(output.fileno())
            with journal.open("xb") as handle:
                handle.write(_canonical_json(header) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(output_dir)
            records: list[EncodedExpertRecord] = []
            journal_offset = journal.stat().st_size
        else:
            records, journal_offset = _resume_layer(
                source_fd,
                source,
                tensors,
                partial=partial,
                journal=journal,
                header=header,
                layer=layer,
                shard=final.name,
                layout=layout,
            )
        layer_digest = _hash_prefix(partial, len(records) * layout.record_bytes)

        with partial.open("r+b") as output, journal.open("r+b") as log:
            output.seek(len(records) * layout.record_bytes)
            log.seek(journal_offset)
            for expert in range(len(records), layout.expert_count):
                cursor = expert * layout.record_bytes
                payloads: list[bytes] = []
                segments: list[EncodedSegment] = []
                source_components: list[dict[str, Any]] = []
                for projection in _PROJECTIONS:
                    decoded, source_state = _decode_projection(
                        source_fd,
                        source,
                        tensors[projection],
                        projection=projection,
                        expert=expert,
                        layout=layout,
                    )
                    encoded, encoded_segments = _encode_projection(
                        projection,
                        decoded,
                        layer=layer,
                        expert=expert,
                        shard=final.name,
                        offset=cursor,
                    )
                    # The decoded FP32 projection is released before the next
                    # source projection is read.
                    del decoded
                    payloads.append(encoded)
                    segments.extend(encoded_segments)
                    source_components.append(source_state)
                    cursor += len(encoded)
                payload = b"".join(payloads)
                record = EncodedExpertRecord(
                    layer=layer,
                    expert=expert,
                    shard=final.name,
                    record_offset=expert * layout.record_bytes,
                    logical_bytes=len(payload),
                    segments=tuple(segments),
                    sha256=_sha256_bytes(payload),
                    payload=payload,
                )
                _validate_record_geometry(
                    record,
                    layer=layer,
                    expert=expert,
                    shard=final.name,
                    layout=layout,
                )
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
                layer_digest.update(payload)
                entry = {
                    "ordinal": expert,
                    "source_components": source_components,
                    "output": record.to_dict(),
                }
                encoded_entry = _canonical_json(entry) + b"\n"
                log.write(encoded_entry)
                log.flush()
                os.fsync(log.fileno())
                journal_offset += len(encoded_entry)
                records.append(record.metadata_only())
            output.truncate(layout.layer_bytes)
            output.flush()
            os.fsync(output.fileno())
        if partial.stat().st_size != layout.layer_bytes:
            raise ValueError("Kimi K3 converted layer byte size is not exact")
        expected_part_sha256 = layer_digest.hexdigest()
        durable_part_sha256 = _hash_file(partial)
        if durable_part_sha256 != expected_part_sha256:
            raise ValueError(
                "durable Kimi K3 partial readback hash does not match completion"
            )
        completion_entry = {
            "completion": {
                "record_count": layout.expert_count,
                "size": layout.layer_bytes,
                "sha256": expected_part_sha256,
            }
        }
        with journal.open("ab") as log:
            log.write(_canonical_json(completion_entry) + b"\n")
            log.flush()
            os.fsync(log.fileno())
        _adopt_without_overwrite(partial, final)
        _fsync_directory(output_dir)
        return ConvertedLayer(
            layer=layer,
            path=final,
            journal_path=journal,
            logical_bytes=layout.layer_bytes,
            sha256=durable_part_sha256,
            records=tuple(records),
        )
    finally:
        os.close(source_fd)


def _resident_tensors(source: GGUFFile) -> tuple[GGUFTensor, ...]:
    residents = tuple(
        sorted(
            (
                tensor
                for tensor in source.tensors
                if tensor.ggml_type in _RESIDENT_DTYPES
            ),
            key=lambda tensor: tensor.name,
        )
    )
    if not residents:
        raise ValueError("GGUF source contains no BF16/F32 resident tensors")
    if any(tensor.name == "__metadata__" for tensor in residents):
        raise ValueError("GGUF resident tensor cannot use safetensors metadata name")
    return residents


def _resident_source_identity(
    source: GGUFFile,
    tensors: tuple[GGUFTensor, ...],
) -> dict[str, Any]:
    if source.identity is None:
        raise ValueError("resident source has no inspected file identity")
    return {
        "path": str(source.path.resolve()),
        "file_identity": {
            "device": source.identity.device,
            "inode": source.identity.inode,
            "size": source.identity.size,
            "mtime_ns": source.identity.mtime_ns,
            "ctime_ns": source.identity.ctime_ns,
        },
        "data_offset": source.data_offset,
        "tensors": [
            {
                "name": tensor.name,
                "dims": list(tensor.dims),
                "ggml_type": tensor.ggml_type,
                "offset": tensor.offset,
                "length": tensor_nbytes(tensor),
            }
            for tensor in tensors
        ],
    }


def _safetensors_prefix(
    tensors: tuple[GGUFTensor, ...],
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    header: dict[str, Any] = {}
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for tensor in tensors:
        end = cursor + tensor_nbytes(tensor)
        offsets[tensor.name] = (cursor, end)
        header[tensor.name] = {
            "dtype": _RESIDENT_DTYPES[tensor.ggml_type],
            "shape": list(tensor.shape),
            "data_offsets": [cursor, end],
        }
        cursor = end
    encoded = _canonical_json(header)
    padded_length = (len(encoded) + 7) & ~7
    padded = encoded + b" " * (padded_length - len(encoded))
    return padded_length.to_bytes(8, "little") + padded, offsets


def _copy_span(
    source_fd: int,
    output,
    *,
    offset: int,
    length: int,
    source_digest: _Digest,
    output_digest: _Digest,
) -> None:
    cursor = offset
    remaining = length
    while remaining:
        chunk = os.pread(source_fd, min(remaining, _COPY_CHUNK_BYTES), cursor)
        if not chunk:
            raise ValueError("resident tensor source span was truncated")
        output.write(chunk)
        source_digest.update(chunk)
        output_digest.update(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)


def _resident_source_hashes(
    source: GGUFFile,
    tensors: tuple[GGUFTensor, ...],
    *,
    output_prefix: bytes,
) -> tuple[list[dict[str, Any]], str]:
    output_digest = hashlib.sha256(output_prefix)
    components: list[dict[str, Any]] = []
    source_fd = open_verified_gguf(source)
    try:
        for tensor in tensors:
            start, end = source.tensor_span(tensor)
            digest = hashlib.sha256()
            cursor = start
            while cursor < end:
                chunk = os.pread(
                    source_fd,
                    min(end - cursor, _COPY_CHUNK_BYTES),
                    cursor,
                )
                if not chunk:
                    raise ValueError("resident tensor source span was truncated")
                digest.update(chunk)
                output_digest.update(chunk)
                cursor += len(chunk)
            components.append(
                {
                    "tensor": tensor.name,
                    "offset": start,
                    "length": end - start,
                    "sha256": digest.hexdigest(),
                }
            )
    finally:
        os.close(source_fd)
    return components, output_digest.hexdigest()


def _resident_results(
    tensors: tuple[GGUFTensor, ...],
    *,
    output_path: Path,
    prefix_length: int,
    offsets: Mapping[str, tuple[int, int]],
) -> tuple[ResidentTensor, ...]:
    return tuple(
        ResidentTensor(
            tensor=tensor.name,
            shard=output_path.name,
            offset=prefix_length + offsets[tensor.name][0],
            length=tensor_nbytes(tensor),
            dtype=_RESIDENT_DTYPES[tensor.ggml_type],
            shape=tensor.shape,
        )
        for tensor in tensors
    )


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("resident receipt is too large")
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("resident receipt is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("resident receipt must be an object")
    return value


def copy_resident_safetensors(
    source: GGUFFile,
    output_path: Path,
) -> tuple[ResidentTensor, ...]:
    """Raw-copy one GGUF shard's BF16/F32 tensors into safetensors framing."""

    output_path = Path(output_path)
    source_path = source.path.resolve()
    if output_path.resolve() == source_path:
        raise ValueError("resident source and output paths must be separate")
    verified_source_fd = open_verified_gguf(source)
    os.close(verified_source_fd)
    tensors = _resident_tensors(source)
    prefix, offsets = _safetensors_prefix(tensors)
    expected_size = len(prefix) + sum(tensor_nbytes(tensor) for tensor in tensors)
    source_identity = _resident_source_identity(source, tensors)
    partial = output_path.with_name(output_path.name + ".partial")
    receipt = output_path.with_name(output_path.name + ".receipt.json")
    receipt_partial = receipt.with_name(receipt.name + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (output_path, partial, receipt, receipt_partial):
        if _entry_exists(candidate) and not stat.S_ISREG(candidate.lstat().st_mode):
            raise ValueError(
                f"resident conversion member is not a regular file: {candidate}"
            )

    if _entry_exists(output_path):
        if _entry_exists(partial) and not _same_regular_inode(output_path, partial):
            raise ValueError(
                "resident final and partial hard-link identity does not match"
            )
        if (
            _entry_exists(receipt)
            and _entry_exists(receipt_partial)
            and not _same_regular_inode(receipt, receipt_partial)
        ):
            raise ValueError(
                "resident receipt final and partial hard-link identity does not match"
            )
        active_receipt = receipt if _entry_exists(receipt) else receipt_partial
        if not _entry_exists(active_receipt):
            raise ValueError("refusing existing resident file without a receipt")
        parsed = _load_receipt(active_receipt)
        source_components, expected_sha256 = _resident_source_hashes(
            source,
            tensors,
            output_prefix=prefix,
        )
        expected_receipt = {
            "format": _RESIDENT_RECEIPT_FORMAT,
            "source": source_identity,
            "source_components": source_components,
            "output": {
                "file": output_path.name,
                "size": expected_size,
                "sha256": expected_sha256,
                "header_sha256": _sha256_bytes(prefix),
            },
        }
        if parsed != expected_receipt:
            raise ValueError("resident receipt identity or hash does not match")
        if output_path.stat().st_size != expected_size:
            raise ValueError("resident output size does not match its receipt")
        with output_path.open("rb") as handle:
            if handle.read(len(prefix)) != prefix:
                raise ValueError(
                    "resident safetensors header does not match its receipt"
                )
        if _hash_file(output_path) != expected_sha256:
            raise ValueError("resident output hash does not match its receipt")
        if active_receipt == receipt_partial:
            _adopt_without_overwrite(receipt_partial, receipt)
            _fsync_directory(output_path.parent)
        elif _entry_exists(receipt_partial):
            receipt_partial.unlink()
            _fsync_directory(output_path.parent)
        if _entry_exists(partial):
            partial.unlink()
            _fsync_directory(output_path.parent)
        return _resident_results(
            tensors,
            output_path=output_path,
            prefix_length=len(prefix),
            offsets=offsets,
        )

    if _entry_exists(receipt):
        raise ValueError("resident receipt exists without its final output")
    if _entry_exists(partial):
        partial.unlink()
    if _entry_exists(receipt_partial):
        receipt_partial.unlink()

    source_components: list[dict[str, Any]] = []
    output_digest = hashlib.sha256(prefix)
    source_fd = open_verified_gguf(source)
    try:
        with partial.open("xb") as output:
            output.write(prefix)
            for tensor in tensors:
                start, end = source.tensor_span(tensor)
                source_digest = hashlib.sha256()
                _copy_span(
                    source_fd,
                    output,
                    offset=start,
                    length=end - start,
                    source_digest=source_digest,
                    output_digest=output_digest,
                )
                source_components.append(
                    {
                        "tensor": tensor.name,
                        "offset": start,
                        "length": end - start,
                        "sha256": source_digest.hexdigest(),
                    }
                )
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(source_fd)
    if partial.stat().st_size != expected_size:
        raise ValueError("resident safetensors final size is not exact")
    output_sha256 = output_digest.hexdigest()
    receipt_value = {
        "format": _RESIDENT_RECEIPT_FORMAT,
        "source": source_identity,
        "source_components": source_components,
        "output": {
            "file": output_path.name,
            "size": expected_size,
            "sha256": output_sha256,
            "header_sha256": _sha256_bytes(prefix),
        },
    }
    with receipt_partial.open("xb") as handle:
        handle.write(_canonical_json(receipt_value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if _entry_exists(output_path) or _entry_exists(receipt):
        raise ValueError("refusing to overwrite a concurrently created resident output")
    _adopt_without_overwrite(partial, output_path)
    # If this second adoption is interrupted, the matching partial receipt is
    # intentionally retained and validated by the completed-output path.
    _adopt_without_overwrite(receipt_partial, receipt)
    _fsync_directory(output_path.parent)
    return _resident_results(
        tensors,
        output_path=output_path,
        prefix_length=len(prefix),
        offsets=offsets,
    )


_PRODUCTION_LAYOUT = KimiK3Layout()
_ASSEMBLY_RECEIPT_FORMAT = "mtplx-kimi-k3-q2k-t158-conversion-v1"
_ORIGINAL_CONFIG_NAME = "config.kimi_k3.original.json"
_AUDITED_RUNTIME_FILES = (
    "tokenizer_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
    "encoding_k3.py",
    "generation_config.json",
    "configuration_kimi_k3.py",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
    "media_utils.py",
    "preprocessor_config.json",
)
_OPTIONAL_DOCUMENTATION_FILES = ("LICENSE", "README.md")
_MAX_OFFICIAL_METADATA_BYTES = 256 * 1024 * 1024
_OFFICIAL_FILE_SHA256 = {
    "config.json": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    "tokenizer_config.json": (
        "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e"
    ),
    "tiktoken.model": (
        "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
    ),
    "tokenization_kimi.py": (
        "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944"
    ),
    "encoding_k3.py": (
        "b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333"
    ),
    "generation_config.json": (
        "c6648c25e9705af7fba8847e243840d21b5cc63ddeb6297f750a7ddbb6a02836"
    ),
    "configuration_kimi_k3.py": (
        "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae"
    ),
    "modeling_kimi_k3.py": (
        "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2"
    ),
    "modeling_kimi_linear.py": (
        "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"
    ),
    "kimi_k3_processor.py": (
        "ec9f7e86d2ab0eee07a8e7e7c037046e77ac3c25a710ad1298ec13be3b585b54"
    ),
    "kimi_k3_vision_processing.py": (
        "d122b30bfd3a51a6f05d4bfcfda1e657827322b1353f7caefeebc2835d7736b5"
    ),
    "media_utils.py": (
        "78403540328f9847d6b7ebc5c44eb2e6a752863de0afb7d0710728bb161dc60d"
    ),
    "preprocessor_config.json": (
        "4be333605990c53a816e586dee9d5dd545afb7a59947c17f8f7ef26b4782668e"
    ),
    "LICENSE": "20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887",
    "README.md": ("57de265b5842dfa465c6e73b368b0e15a89b8793b5450528dad577da202cc6fe"),
}


@dataclass(frozen=True, slots=True)
class AssemblyProjection:
    source_revision: str
    source_tensor_bytes: int
    source_resident_tensor_bytes: int
    source_routed_tensor_bytes: int
    output_preserved_resident_tensor_bytes: int
    output_routed_tensor_bytes: int
    output_tensor_bytes: int
    text_runtime_tensor_bytes: int
    layer_count: int
    expert_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_revision": self.source_revision,
            "source_tensor_bytes": self.source_tensor_bytes,
            "source_resident_tensor_bytes": self.source_resident_tensor_bytes,
            "source_routed_tensor_bytes": self.source_routed_tensor_bytes,
            "output_preserved_resident_tensor_bytes": (
                self.output_preserved_resident_tensor_bytes
            ),
            "output_routed_tensor_bytes": self.output_routed_tensor_bytes,
            "output_tensor_bytes": self.output_tensor_bytes,
            "text_runtime_tensor_bytes": self.text_runtime_tensor_bytes,
            "layer_count": self.layer_count,
            "expert_count": self.expert_count,
        }


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    output_root: Path
    converted_layers: tuple[int, ...]
    complete: bool
    manifest_path: Path | None
    resident_tensor_bytes: int
    routed_expert_bytes: int
    artifact_tensor_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "converted_layers": list(self.converted_layers),
            "complete": self.complete,
            "manifest_path": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
            "resident_tensor_bytes": self.resident_tensor_bytes,
            "routed_expert_bytes": self.routed_expert_bytes,
            "artifact_tensor_bytes": self.artifact_tensor_bytes,
        }


def _is_production_layout(layout: KimiK3Layout) -> bool:
    return layout == _PRODUCTION_LAYOUT


def _require_pinned_revisions(
    source_revision: str,
    official_revision: str,
) -> None:
    if source_revision != KIMI_K3_SOURCE_REVISION:
        raise ValueError(
            "Kimi K3 source revision must be pinned to "
            f"{KIMI_K3_SOURCE_REVISION}, got {source_revision!r}"
        )
    if official_revision != KIMI_K3_OFFICIAL_REVISION:
        raise ValueError(
            "Kimi K3 official revision must be pinned to "
            f"{KIMI_K3_OFFICIAL_REVISION}, got {official_revision!r}"
        )


def _normalized_layers(
    layers: Collection[int] | None,
    *,
    layout: KimiK3Layout,
) -> tuple[int, ...]:
    if layers is None:
        return tuple(range(1, layout.layer_count + 1))
    normalized: list[int] = []
    for value in layers:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Kimi K3 layer selections must be exact integers")
        if not 1 <= value <= layout.layer_count:
            raise ValueError(
                f"Kimi K3 layer selections must be in 1..{layout.layer_count}"
            )
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Kimi K3 layer selections must be unique")
    return tuple(sorted(normalized))


def _check_separate_roots(source_root: Path, output_root: Path) -> None:
    source = source_root.resolve()
    output = output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Kimi K3 source and output roots must be separate")


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_OFFICIAL_METADATA_BYTES,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if metadata.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit: {path}")
    payload = path.read_bytes()
    if len(payload) != metadata.st_size:
        raise ValueError(f"{label} changed while being read: {path}")
    return payload


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in official config: {key!r}")
        result[key] = value
    return result


def _flatten_official_config(payload: bytes) -> bytes:
    try:
        top = json.loads(payload, object_pairs_hook=_strict_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("official Kimi K3 config.json is invalid") from exc
    if not isinstance(top, dict) or top.get("model_type") != "kimi_k3":
        raise ValueError("official config.json must be the pinned kimi_k3 top config")
    text = top.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "kimi_linear":
        raise ValueError("official config.json has no kimi_linear text_config")
    flattened = dict(text)
    for incompatible in ("auto_map", "quantization_config", "_name_or_path"):
        flattened.pop(incompatible, None)
    flattened["model_type"] = "kimi_linear"
    return (
        json.dumps(flattened, indent=2, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def _open_owned_regular(path: Path, flags: int) -> int:
    """Open the exact regular inode inspected at this construction boundary."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"could not inspect artifact metadata {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact metadata is not a regular file: {path}")
    try:
        descriptor = os.open(
            path,
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"could not open artifact metadata {path}") from exc
    after = os.fstat(descriptor)
    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        os.close(descriptor)
        raise ValueError(f"artifact metadata identity changed while opening {path}")
    return descriptor


def _matching_payload_prefix(
    descriptor: int,
    payload: bytes,
    *,
    label: str,
) -> int:
    size = os.fstat(descriptor).st_size
    if size > len(payload):
        raise ValueError(f"existing {label} is longer than the pinned output")
    expected = memoryview(payload)
    cursor = 0
    while cursor < size:
        length = min(_COPY_CHUNK_BYTES, size - cursor)
        try:
            chunk = os.pread(descriptor, length, cursor)
        except InterruptedError:
            continue
        if not chunk or chunk != expected[cursor : cursor + len(chunk)]:
            raise ValueError(f"existing {label} does not match the pinned output")
        cursor += len(chunk)
    if cursor != size or os.fstat(descriptor).st_size != size:
        raise ValueError(f"existing {label} changed during prefix validation")
    return size


def _validate_installed_payload(path: Path, payload: bytes) -> None:
    descriptor = _open_owned_regular(path, os.O_RDONLY)
    try:
        size = _matching_payload_prefix(
            descriptor,
            payload,
            label=path.name,
        )
        if size != len(payload):
            raise ValueError(f"existing {path.name} does not match the pinned output")
    finally:
        os.close(descriptor)


def _complete_metadata_partial(partial: Path, payload: bytes) -> None:
    descriptor = _open_owned_regular(partial, os.O_RDWR)
    try:
        if os.fstat(descriptor).st_nlink != 1:
            raise ValueError(
                f"artifact metadata partial is not exclusively owned: {partial}"
            )
        size = _matching_payload_prefix(
            descriptor,
            payload,
            label=f"{partial.name} partial",
        )
        if size < len(payload):
            if os.lseek(descriptor, 0, os.SEEK_END) != size:
                raise ValueError(
                    f"existing {partial.name} partial changed before append"
                )
            remaining = memoryview(payload)[size:]
            cursor = 0
            while cursor < len(remaining):
                try:
                    written = os.write(
                        descriptor,
                        remaining[
                            cursor : cursor
                            + min(
                                _COPY_CHUNK_BYTES,
                                len(remaining) - cursor,
                            )
                        ],
                    )
                except InterruptedError:
                    continue
                if written <= 0:
                    raise ValueError(
                        f"could not resume artifact metadata partial {partial}"
                    )
                cursor += written
            os.fsync(descriptor)
        if os.fstat(descriptor).st_size != len(payload):
            raise ValueError(
                f"resumed artifact metadata partial has wrong size: {partial}"
            )
    finally:
        os.close(descriptor)


def _install_bytes_exact(
    path: Path,
    payload: bytes,
    *,
    resume: bool,
) -> None:
    partial = path.with_name(path.name + ".partial")
    if _entry_exists(path):
        if not resume:
            raise ValueError(f"refusing to overwrite existing output {path}")
        _validate_installed_payload(path, payload)
        if _entry_exists(partial):
            descriptor = _open_owned_regular(partial, os.O_RDONLY)
            try:
                if os.fstat(descriptor).st_nlink != 1:
                    raise ValueError(
                        f"artifact metadata partial is not exclusively owned: {partial}"
                    )
                _matching_payload_prefix(
                    descriptor,
                    payload,
                    label=f"{path.name} partial",
                )
            finally:
                os.close(descriptor)
            partial.unlink()
            _fsync_directory(path.parent)
        return
    if _entry_exists(partial):
        if not resume:
            raise ValueError(f"refusing existing metadata partial {partial}")
        _complete_metadata_partial(partial, payload)
        _adopt_without_overwrite(partial, path)
        _fsync_directory(path.parent)
        return
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _adopt_without_overwrite(partial, path)
    _fsync_directory(path.parent)


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _layer_source(
    inventory: KimiK3Inventory,
    *,
    layer: int,
) -> GGUFFile:
    sources: list[GGUFFile] = []
    for projection in ("gate", "up", "down"):
        name = f"blk.{layer}.ffn_{projection}_exps.weight"
        try:
            source, _tensor = inventory.tensor_source(name)
        except KeyError as exc:
            raise ValueError(f"Kimi K3 layer {layer} is missing {name!r}") from exc
        sources.append(source)
    first = sources[0]
    if any(source is not first for source in sources[1:]):
        raise ValueError(
            f"Kimi K3 layer {layer} gate/up/down tensors must share the same GGUF shard"
        )
    return first


def _safetensors_shard_info(
    path: Path,
    *,
    kind: str = "safetensors",
) -> ShardInfo:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"resident safetensors header is truncated: {path}")
        header_bytes = 8 + int.from_bytes(raw_length, "little")
        if header_bytes > size:
            raise ValueError(f"resident safetensors header exceeds its file: {path}")
        handle.seek(0)
        header = handle.read(header_bytes)
    if len(header) != header_bytes:
        raise ValueError(f"resident safetensors header is truncated: {path}")
    return ShardInfo(
        name=path.name,
        size=size,
        header_bytes=header_bytes,
        header_sha256=_sha256_bytes(header),
        sha256=_hash_file(path),
        kind=kind,
    )


def _expert_record(
    record: EncodedExpertRecord,
    *,
    part: int,
) -> ExpertRecord:
    return ExpertRecord(
        layer=record.layer,
        expert=record.expert,
        logical_bytes=record.logical_bytes,
        segments=tuple(
            TensorSegment(
                component=segment.component,
                tensor=segment.tensor,
                shard=segment.shard,
                offset=segment.offset,
                length=segment.length,
                # The serializer carries the bfloat16 scale *bits* as uint16
                # arrays. The authoritative component-bank schema names that
                # physical representation U16, matching every other t158
                # artifact and the prebound runtime loader.
                dtype=(
                    "U16" if segment.component.endswith(".scales") else segment.dtype
                ),
                shape=segment.shape,
            )
            for segment in record.segments
        ),
        sha256=record.sha256,
        sidecar_offset=record.record_offset,
        sidecar_length=record.logical_bytes,
        part=part,
    )


def _official_payloads(
    official_root: Path,
) -> tuple[dict[str, bytes], bytes, bytes]:
    config_payload = _read_regular_bytes(
        official_root / "config.json",
        label="official config.json",
    )
    payloads = {
        name: _read_regular_bytes(
            official_root / name,
            label=f"official {name}",
        )
        for name in _AUDITED_RUNTIME_FILES
    }
    for name in _OPTIONAL_DOCUMENTATION_FILES:
        path = official_root / name
        if _entry_exists(path):
            payloads[name] = _read_regular_bytes(
                path,
                label=f"official {name}",
            )
    checked = {"config.json": config_payload, **payloads}
    for name, payload in checked.items():
        expected = _OFFICIAL_FILE_SHA256.get(name)
        if expected is None or _sha256_bytes(payload) != expected:
            raise ValueError(
                f"official {name} is not bound to pinned official revision "
                f"{KIMI_K3_OFFICIAL_REVISION}"
            )
    return payloads, config_payload, _flatten_official_config(config_payload)


def _copy_official_metadata(
    official_root: Path,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    payloads, original_config, flattened_config = _official_payloads(official_root)
    installed: dict[str, tuple[str, bytes]] = {
        "config.json": (_ORIGINAL_CONFIG_NAME, original_config),
        **{name: (name, payload) for name, payload in payloads.items()},
    }
    _install_bytes_exact(
        output_root / "config.json",
        flattened_config,
        resume=resume,
    )
    for _source_name, (output_name, payload) in installed.items():
        _install_bytes_exact(
            output_root / output_name,
            payload,
            resume=resume,
        )
    return {
        source_name: {
            "output_file": output_name,
            "size": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for source_name, (output_name, payload) in sorted(installed.items())
    }


def project_artifact(
    source_root: Path,
    *,
    source_revision: str,
    official_revision: str = KIMI_K3_OFFICIAL_REVISION,
) -> AssemblyProjection:
    """Inspect the immutable source and report byte projections without writes."""

    _require_pinned_revisions(source_revision, official_revision)
    inventory = inspect_kimi_k3_source(Path(source_root), source_revision)
    layout = KimiK3Layout()
    resident_bytes = sum(tensor_nbytes(tensor) for tensor in inventory.resident_tensors)
    source_routed_bytes = sum(
        tensor_nbytes(tensor) for tensor in inventory.expert_tensors
    )
    output_routed_bytes = layout.layer_count * layout.expert_count * layout.record_bytes
    if _is_production_layout(layout):
        expected = (
            KIMI_K3_RESIDENT_BYTES,
            KIMI_K3_ROUTED_BYTES,
            KIMI_K3_T158_ROUTED_BYTES,
        )
        actual = (resident_bytes, source_routed_bytes, output_routed_bytes)
        if actual != expected:
            raise ValueError(
                f"Kimi K3 byte projection differs from the pinned contract: {actual}"
            )
        text_runtime_bytes = KIMI_K3_TEXT_ARTIFACT_BYTES
    else:
        text_runtime_bytes = resident_bytes + output_routed_bytes
    return AssemblyProjection(
        source_revision=source_revision,
        source_tensor_bytes=resident_bytes + source_routed_bytes,
        source_resident_tensor_bytes=resident_bytes,
        source_routed_tensor_bytes=source_routed_bytes,
        output_preserved_resident_tensor_bytes=resident_bytes,
        output_routed_tensor_bytes=output_routed_bytes,
        output_tensor_bytes=resident_bytes + output_routed_bytes,
        text_runtime_tensor_bytes=text_runtime_bytes,
        layer_count=layout.layer_count,
        expert_count=layout.expert_count,
    )


def assemble_artifact(
    source_root: Path,
    output_root: Path,
    *,
    source_revision: str,
    official_revision: str,
    official_root: Path,
    resume: bool,
    layers: Collection[int] | None = None,
) -> AssemblyResult:
    """Assemble a pinned K3 text artifact; publish its manifest only when whole."""

    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    _require_pinned_revisions(source_revision, official_revision)
    source_root = Path(source_root)
    output_root = Path(output_root)
    official_root = Path(official_root)
    _check_separate_roots(source_root, output_root)
    layout = KimiK3Layout()
    selected_layers = _normalized_layers(layers, layout=layout)
    if not selected_layers:
        return AssemblyResult(
            output_root=output_root,
            converted_layers=(),
            complete=False,
            manifest_path=None,
            resident_tensor_bytes=0,
            routed_expert_bytes=0,
            artifact_tensor_bytes=0,
        )

    inventory = inspect_kimi_k3_source(source_root, source_revision)
    expected_layers = tuple(range(1, layout.layer_count + 1))
    if tuple(inventory.layers) != expected_layers:
        raise ValueError(
            "Kimi K3 inspected routed layers differ from the installed layout"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "expert-manifest.json"
    if _entry_exists(manifest_path) and not resume:
        raise ValueError(f"refusing to overwrite existing output {manifest_path}")
    converted: list[ConvertedLayer] = []
    for layer in selected_layers:
        expected_part = output_root / (
            f"experts-t158-layer-{layer:03d}-of-{layout.layer_count:03d}.bin"
        )
        if _entry_exists(expected_part) and not resume:
            raise ValueError(f"refusing to overwrite existing output {expected_part}")
        source = _layer_source(inventory, layer=layer)
        converted.append(
            convert_layer(
                source,
                output_root,
                inventory,
                layer=layer,
                resume=resume,
                layout=layout,
            )
        )

    if selected_layers != expected_layers:
        return AssemblyResult(
            output_root=output_root,
            converted_layers=selected_layers,
            complete=False,
            manifest_path=None,
            resident_tensor_bytes=0,
            routed_expert_bytes=sum(item.logical_bytes for item in converted),
            artifact_tensor_bytes=sum(item.logical_bytes for item in converted),
        )

    all_residents: list[ResidentTensor] = []
    resident_shards: list[ShardInfo] = []
    shard_count = len(inventory.files)
    for index, source in enumerate(inventory.files, 1):
        if not any(tensor.ggml_type in _RESIDENT_DTYPES for tensor in source.tensors):
            continue
        output_path = output_root / (
            f"resident-{index:05d}-of-{shard_count:05d}.safetensors"
        )
        if _entry_exists(output_path) and not resume:
            raise ValueError(f"refusing to overwrite existing output {output_path}")
        residents = copy_resident_safetensors(source, output_path)
        all_residents.extend(residents)
        shard_kind = (
            "safetensors"
            if any(tensor.tensor.startswith("language_model.") for tensor in residents)
            else "preserved-safetensors"
        )
        resident_shards.append(_safetensors_shard_info(output_path, kind=shard_kind))

    resident_names = [tensor.tensor for tensor in all_residents]
    if len(resident_names) != len(set(resident_names)):
        raise ValueError("Kimi K3 resident tensor names are not globally unique")
    all_residents.sort(key=lambda tensor: tensor.tensor)
    text_residents = tuple(
        tensor
        for tensor in all_residents
        if tensor.tensor.startswith("language_model.")
    )
    preserved_resident_bytes = sum(tensor.length for tensor in all_residents)
    text_resident_bytes = sum(tensor.length for tensor in text_residents)

    converted.sort(key=lambda item: item.layer)
    parts = tuple(
        SidecarPart(
            file=item.path.name,
            size=item.logical_bytes,
            sha256=item.sha256,
        )
        for item in converted
    )
    sidecar_shards = tuple(
        ShardInfo(
            name=item.path.name,
            size=item.logical_bytes,
            header_bytes=0,
            header_sha256=EMPTY_SHA256,
            sha256=item.sha256,
            kind="sidecar",
        )
        for item in converted
    )
    records = tuple(
        _expert_record(record, part=part)
        for part, item in enumerate(converted)
        for record in item.records
    )
    routed_expert_bytes = sum(record.logical_bytes for record in records)
    artifact_tensor_bytes = text_resident_bytes + routed_expert_bytes
    if _is_production_layout(layout):
        exact = (
            preserved_resident_bytes,
            text_resident_bytes,
            routed_expert_bytes,
            artifact_tensor_bytes,
            len(parts),
            len(records),
        )
        expected = (
            KIMI_K3_RESIDENT_BYTES,
            KIMI_K3_TEXT_RESIDENT_BYTES,
            KIMI_K3_T158_ROUTED_BYTES,
            KIMI_K3_TEXT_ARTIFACT_BYTES,
            KIMI_K3_LAYER_COUNT,
            KIMI_K3_LAYER_COUNT * KIMI_K3_EXPERT_COUNT,
        )
        if exact != expected:
            raise ValueError(
                "assembled Kimi K3 bytes or record inventory differ from the "
                f"pinned contract: expected={expected}, actual={exact}"
            )

    index = {
        "metadata": {"total_size": preserved_resident_bytes},
        "weight_map": {tensor.tensor: tensor.shard for tensor in all_residents},
    }
    official_files = _copy_official_metadata(
        official_root,
        output_root,
        resume=resume,
    )
    _install_bytes_exact(
        output_root / "model.safetensors.index.json",
        _pretty_json(index),
        resume=resume,
    )

    alignment = math.gcd(4096, layout.record_bytes)
    manifest = ExpertManifest(
        model_key="kimi-k3-q1t",
        source_repo=KIMI_K3_SOURCE_REPO,
        source_revision=source_revision,
        quant_bits=2,
        quant_group_size=64,
        quant_mode="t158",
        artifact_tensor_bytes=artifact_tensor_bytes,
        resident_tensor_bytes=text_resident_bytes,
        routed_expert_bytes=routed_expert_bytes,
        shards=tuple(resident_shards) + sidecar_shards,
        resident_tensors=text_residents,
        records=records,
        sidecar=SidecarInfo(alignment=alignment, parts=parts),
    ).with_digest()
    manifest.validate_structure()
    # Verify the exact resident index/header inventory, preserved multimodal
    # shard provenance, descriptor geometry, and sidecar sizes before the
    # publication marker or its receipt can exist.
    verify_expert_manifest(manifest, output_root)
    # This manifest contains 82,432 records in production. Compact canonical
    # JSON keeps it inside expert_manifest.MAX_MANIFEST_BYTES; indentation
    # alone would push the authoritative file beyond the bounded reader.
    manifest_payload = _canonical_json(manifest.to_dict()) + b"\n"
    receipt = {
        "format": _ASSEMBLY_RECEIPT_FORMAT,
        "source": {
            "repo": KIMI_K3_SOURCE_REPO,
            "revision": source_revision,
            "gguf_shards": len(inventory.files),
            "resident_descriptor_sha256": inventory.resident_descriptor_sha256,
            "resident_tensor_bytes": preserved_resident_bytes,
            "routed_q2_k_tensor_bytes": sum(
                tensor_nbytes(tensor) for tensor in inventory.expert_tensors
            ),
        },
        "official": {
            "repo": KIMI_K3_OFFICIAL_REPO,
            "revision": official_revision,
            "files": official_files,
        },
        "codec": {
            "mode": "t158",
            "nominal_bits": 2,
            "physical_bits_per_weight": 1.875,
            "group_size": 64,
            "record_bytes": layout.record_bytes,
            "routed_tensor_bytes": routed_expert_bytes,
        },
        "residents": {
            "preserved_tensor_bytes": preserved_resident_bytes,
            "text_tensor_bytes": text_resident_bytes,
            "non_text_tensor_bytes": (preserved_resident_bytes - text_resident_bytes),
            "shards": [shard.to_dict() for shard in resident_shards],
        },
        "artifact": {
            "text_tensor_bytes": artifact_tensor_bytes,
            "manifest_sha256": manifest.manifest_sha256,
            "sidecar_parts": [part.to_dict() for part in parts],
        },
    }
    _install_bytes_exact(
        output_root / "conversion-receipt.json",
        _pretty_json(receipt),
        resume=resume,
    )
    # The authoritative manifest is the final publication marker. Every
    # construction product above is durable and validated before this write.
    _install_bytes_exact(
        manifest_path,
        manifest_payload,
        resume=resume,
    )
    return AssemblyResult(
        output_root=output_root,
        converted_layers=selected_layers,
        complete=True,
        manifest_path=manifest_path,
        resident_tensor_bytes=text_resident_bytes,
        routed_expert_bytes=routed_expert_bytes,
        artifact_tensor_bytes=artifact_tensor_bytes,
    )
