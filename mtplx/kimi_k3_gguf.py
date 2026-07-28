"""Strict local GGUF v3 parsing for the pinned Kimi K3 Q2_K source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import struct
from types import MappingProxyType
from typing import BinaryIO

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_Q2_K = 10
GGML_TYPE_BF16 = 30

Q2_K_VALUES = 256
Q2_K_BYTES = 84

_Q2_K_BATCH_BLOCKS = 4096
# The pinned split has at most 165 tensors, four metadata entries, no arrays,
# and sub-32 KiB headers. These fixed ceilings retain headroom without exposing
# a generic untrusted-object parser surface.
_MAX_TENSOR_COUNT = 256
_MAX_METADATA_COUNT = 16
_MAX_STRING_BYTES = 4 * 1024
_MAX_ARRAY_ELEMENTS = 4 * 1024
_MAX_METADATA_OBJECTS = 8 * 1024
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_DIMENSIONS = 4
_MAX_ALIGNMENT = 1024 * 1024

KIMI_K3_SOURCE_REVISION = "0169245d3ea1473a3f9f03bca821d855df5fb2a3"
KIMI_K3_SHARDS = 96
KIMI_K3_LAYERS = tuple(range(1, 93))
KIMI_K3_EXPERTS = 896
KIMI_K3_TENSOR_COUNT = 2_904
KIMI_K3_F32_COUNT = 506
KIMI_K3_BF16_COUNT = 2_122
KIMI_K3_Q2_K_COUNT = 276
KIMI_K3_RESIDENT_BYTES = 114_404_258_816
KIMI_K3_ROUTED_BYTES = 893_399_334_912
# Generated from the pinned 96 headers by resident_descriptor_sha256(). The
# length-prefixed stream contains shard, name, GGML dims/type, and absolute
# byte span for each BF16/F32 descriptor; tensor payloads are never loaded.
KIMI_K3_RESIDENT_DESCRIPTOR_SHA256 = (
    "04227c39303d433a2586b929ac7ef9535c0a9c29d39a39a8dafcdb981beceb98"
)

_SHARD_RE = re.compile(r".*-(\d{5})-of-(\d{5})\.gguf\Z", re.IGNORECASE)
_EXPERT_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\Z")
_PROJECTION_ORDER = {"gate": 0, "up": 1, "down": 2}
_EXPECTED_EXPERT_DIMS = {
    "gate": (3584, 3072, KIMI_K3_EXPERTS),
    "up": (3584, 3072, KIMI_K3_EXPERTS),
    "down": (3072, 3584, KIMI_K3_EXPERTS),
}

_VALUE_UINT8 = 0
_VALUE_INT8 = 1
_VALUE_UINT16 = 2
_VALUE_INT16 = 3
_VALUE_UINT32 = 4
_VALUE_INT32 = 5
_VALUE_FLOAT32 = 6
_VALUE_BOOL = 7
_VALUE_STRING = 8
_VALUE_ARRAY = 9
_VALUE_UINT64 = 10
_VALUE_INT64 = 11
_VALUE_FLOAT64 = 12

_REQUIRED_METADATA_TYPES = {
    "general.alignment": _VALUE_UINT32,
    "split.no": _VALUE_UINT16,
    "split.count": _VALUE_UINT16,
    "split.tensors.count": _VALUE_UINT16,
}
_VALUE_TYPE_NAMES = {
    _VALUE_UINT16: "UINT16",
    _VALUE_UINT32: "UINT32",
}

_SCALAR_VALUE_FORMATS = {
    _VALUE_UINT8: "<B",
    _VALUE_INT8: "<b",
    _VALUE_UINT16: "<H",
    _VALUE_INT16: "<h",
    _VALUE_UINT32: "<I",
    _VALUE_INT32: "<i",
    _VALUE_FLOAT32: "<f",
    _VALUE_BOOL: "<B",
    _VALUE_UINT64: "<Q",
    _VALUE_INT64: "<q",
    _VALUE_FLOAT64: "<d",
}


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    """One GGUF tensor descriptor; ``offset`` is relative to the data section."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Row-major shape corresponding to the GGML dimension order."""

        return tuple(reversed(self.dims))

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self)

    @property
    def type_name(self) -> str:
        try:
            return {
                GGML_TYPE_F32: "F32",
                GGML_TYPE_BF16: "BF16",
                GGML_TYPE_Q2_K: "Q2_K",
            }[self.ggml_type]
        except KeyError as exc:
            raise ValueError(f"unsupported GGML tensor type {self.ggml_type}") from exc


@dataclass(frozen=True, slots=True)
class GGUFFileIdentity:
    """Stable local-file descriptors captured from the opened source inode."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class GGUFFile:
    """Immutable parsed header and validated tensor-data spans for one file."""

    path: Path
    version: int
    alignment: int
    data_offset: int
    file_size: int
    metadata: Mapping[str, object]
    tensors: tuple[GGUFTensor, ...]
    metadata_types: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    identity: GGUFFileIdentity | None = None

    def tensor(self, name: str) -> GGUFTensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def tensor_span(self, tensor: GGUFTensor | str) -> tuple[int, int]:
        descriptor = self.tensor(tensor) if isinstance(tensor, str) else tensor
        if descriptor not in self.tensors:
            raise ValueError("tensor does not belong to this GGUF file")
        start = self.data_offset + descriptor.offset
        return start, start + tensor_nbytes(descriptor)


@dataclass(frozen=True, slots=True)
class KimiK3Inventory:
    """Construction-time validated view of the pinned 96-shard source."""

    revision: str
    files: tuple[GGUFFile, ...]
    expert_tensors: tuple[GGUFTensor, ...]
    resident_tensors: tuple[GGUFTensor, ...]
    layers: tuple[int, ...]
    resident_descriptor_sha256: str = ""

    def tensor_source(self, name: str) -> tuple[GGUFFile, GGUFTensor]:
        for source in self.files:
            for tensor in source.tensors:
                if tensor.name == name:
                    return source, tensor
        raise KeyError(name)


class _Reader:
    __slots__ = ("_file", "_file_size", "position")

    def __init__(self, file: BinaryIO, file_size: int) -> None:
        self._file = file
        self._file_size = file_size
        self.position = 0

    def read_exact(self, count: int, *, what: str) -> bytes:
        if count < 0:
            raise ValueError(f"negative byte count for {what}")
        end = self.position + count
        if end > _MAX_HEADER_BYTES:
            raise ValueError("GGUF header exceeds bounded header size")
        if end > self._file_size:
            raise ValueError(f"truncated GGUF header while reading {what}")
        raw = self._file.read(count)
        if len(raw) != count:
            raise ValueError(f"truncated GGUF header while reading {what}")
        self.position = end
        return raw

    def unpack(self, fmt: str, *, what: str) -> tuple[object, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read_exact(size, what=what))

    def string(self, *, what: str) -> str:
        (length,) = self.unpack("<Q", what=f"{what} length")
        assert isinstance(length, int)
        if length > _MAX_STRING_BYTES:
            raise ValueError(
                f"GGUF string length {length} exceeds {_MAX_STRING_BYTES} bytes"
            )
        raw = self.read_exact(length, what=what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{what} is not valid UTF-8") from exc


class _MetadataBudget:
    __slots__ = ("used",)

    def __init__(self) -> None:
        self.used = 0

    def consume(self, count: int) -> None:
        if count < 0 or count > _MAX_METADATA_OBJECTS - self.used:
            raise ValueError(
                f"GGUF metadata object budget exceeds {_MAX_METADATA_OBJECTS} objects"
            )
        self.used += count


def _read_scalar(reader: _Reader, value_type: int, *, what: str) -> object:
    try:
        fmt = _SCALAR_VALUE_FORMATS[value_type]
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}") from exc
    (value,) = reader.unpack(fmt, what=what)
    if value_type == _VALUE_BOOL:
        if value not in (0, 1):
            raise ValueError(f"invalid GGUF boolean value {value}")
        return bool(value)
    return value


def _read_metadata_value(
    reader: _Reader,
    value_type: int,
    *,
    what: str,
    budget: _MetadataBudget,
) -> object:
    if value_type in _SCALAR_VALUE_FORMATS:
        budget.consume(1)
        return _read_scalar(reader, value_type, what=what)
    if value_type == _VALUE_STRING:
        budget.consume(1)
        return reader.string(what=what)
    if value_type != _VALUE_ARRAY:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")

    (element_type,) = reader.unpack("<I", what=f"{what} array element type")
    (count,) = reader.unpack("<Q", what=f"{what} array count")
    assert isinstance(element_type, int)
    assert isinstance(count, int)
    if element_type == _VALUE_ARRAY or (
        element_type not in _SCALAR_VALUE_FORMATS and element_type != _VALUE_STRING
    ):
        raise ValueError(f"unsupported GGUF array element type {element_type}")
    if count > _MAX_ARRAY_ELEMENTS:
        raise ValueError(
            f"GGUF array count {count} exceeds {_MAX_ARRAY_ELEMENTS} elements"
        )
    # One tuple plus one retained Python object per element. Charge the whole
    # expansion before allocating any of it.
    budget.consume(1 + count)
    if element_type == _VALUE_STRING:
        return tuple(
            reader.string(what=f"{what} array element {index}")
            for index in range(count)
        )
    return tuple(
        _read_scalar(reader, element_type, what=f"{what} array element {index}")
        for index in range(count)
    )


def tensor_nbytes(tensor: GGUFTensor) -> int:
    """Return exact storage bytes for the locally supported GGML types."""

    if not tensor.dims or len(tensor.dims) > _MAX_DIMENSIONS:
        raise ValueError("GGUF tensor dimension count must be between 1 and 4")
    value_count = 1
    for dimension in tensor.dims:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError("GGUF tensor dimensions must be integers")
        if dimension <= 0:
            raise ValueError("GGUF tensor dimensions must be positive")
        value_count *= dimension
    if tensor.ggml_type == GGML_TYPE_F32:
        return value_count * 4
    if tensor.ggml_type == GGML_TYPE_BF16:
        return value_count * 2
    if tensor.ggml_type == GGML_TYPE_Q2_K:
        if tensor.dims[0] % Q2_K_VALUES:
            raise ValueError(
                "Q2_K tensor first GGML dimension must be divisible by 256"
            )
        return value_count // Q2_K_VALUES * Q2_K_BYTES
    raise ValueError(f"unsupported GGML tensor type {tensor.ggml_type}")


def _checked_alignment(metadata: Mapping[str, object]) -> int:
    alignment = metadata.get("general.alignment", 32)
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
        or alignment > _MAX_ALIGNMENT
        or alignment & (alignment - 1)
    ):
        raise ValueError("GGUF general.alignment must be a bounded power of two")
    return alignment


def _file_identity(metadata: os.stat_result) -> GGUFFileIdentity:
    return GGUFFileIdentity(
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        size=int(metadata.st_size),
        mtime_ns=int(metadata.st_mtime_ns),
        ctime_ns=int(metadata.st_ctime_ns),
    )


def open_verified_gguf(source: GGUFFile) -> int:
    """Open the exact inode/version captured by :func:`read_gguf`.

    The returned descriptor, rather than a second path-based open, must be used
    for conversion reads. Replacement, truncation, or in-place modification
    after inspection fails before any bytes are consumed.
    """

    if source.identity is None:
        raise ValueError("GGUF source has no inspected file identity")
    try:
        descriptor = os.open(source.path, os.O_RDONLY)
    except OSError as exc:
        raise ValueError(f"cannot reopen inspected GGUF file {source.path}") from exc
    actual = _file_identity(os.fstat(descriptor))
    if actual != source.identity:
        os.close(descriptor)
        raise ValueError(f"GGUF source changed since inspection: {source.path}")
    return descriptor


def _update_resident_descriptor_digest(
    digest: object,
    *,
    shard_number: int,
    source: GGUFFile,
    tensor: GGUFTensor,
) -> None:
    """Hash one unambiguous, length-prefixed canonical descriptor."""

    update = getattr(digest, "update")
    encoded_name = tensor.name.encode("utf-8")
    update(struct.pack("<H", shard_number))
    update(struct.pack("<I", len(encoded_name)))
    update(encoded_name)
    update(struct.pack("<B", len(tensor.dims)))
    for dimension in tensor.dims:
        update(struct.pack("<Q", dimension))
    update(struct.pack("<I", tensor.ggml_type))
    start = source.data_offset + tensor.offset
    update(struct.pack("<QQ", start, tensor_nbytes(tensor)))


def resident_descriptor_sha256(files: tuple[GGUFFile, ...]) -> str:
    """Digest canonical BF16/F32 descriptors without retaining a manifest."""

    digest = hashlib.sha256()
    for shard_number, source in enumerate(files, start=1):
        residents = sorted(
            (
                tensor
                for tensor in source.tensors
                if tensor.ggml_type in (GGML_TYPE_BF16, GGML_TYPE_F32)
            ),
            key=lambda tensor: tensor.name,
        )
        for tensor in residents:
            _update_resident_descriptor_digest(
                digest,
                shard_number=shard_number,
                source=source,
                tensor=tensor,
            )
    return digest.hexdigest()


def read_gguf(path: Path) -> GGUFFile:
    """Parse one local GGUF v3 header and validate every tensor byte span."""

    path = Path(path)
    try:
        file = path.open("rb")
    except OSError as exc:
        raise ValueError(f"cannot open GGUF file {path}") from exc
    with file:
        identity = _file_identity(os.fstat(file.fileno()))
        file_size = identity.size
        reader = _Reader(file, file_size)
        magic = reader.read_exact(4, what="magic")
        if magic != GGUF_MAGIC:
            raise ValueError(f"bad GGUF magic {magic!r}")
        (version, tensor_count, metadata_count) = reader.unpack(
            "<IQQ", what="fixed header"
        )
        assert isinstance(version, int)
        assert isinstance(tensor_count, int)
        assert isinstance(metadata_count, int)
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}")
        if tensor_count > _MAX_TENSOR_COUNT:
            raise ValueError(
                f"GGUF tensor count {tensor_count} exceeds {_MAX_TENSOR_COUNT}"
            )
        if metadata_count > _MAX_METADATA_COUNT:
            raise ValueError(
                f"GGUF metadata count {metadata_count} exceeds {_MAX_METADATA_COUNT}"
            )

        mutable_metadata: dict[str, object] = {}
        mutable_metadata_types: dict[str, int] = {}
        metadata_budget = _MetadataBudget()
        for index in range(metadata_count):
            key = reader.string(what=f"metadata key {index}")
            metadata_budget.consume(1)
            if key in mutable_metadata:
                raise ValueError(f"duplicate GGUF metadata key {key!r}")
            (value_type,) = reader.unpack("<I", what=f"metadata type for {key!r}")
            assert isinstance(value_type, int)
            required_type = _REQUIRED_METADATA_TYPES.get(key)
            if required_type is not None and value_type != required_type:
                raise ValueError(
                    f"GGUF metadata {key!r} must use "
                    f"{_VALUE_TYPE_NAMES[required_type]} type"
                )
            mutable_metadata_types[key] = value_type
            mutable_metadata[key] = _read_metadata_value(
                reader,
                value_type,
                what=f"metadata value for {key!r}",
                budget=metadata_budget,
            )

        tensors: list[GGUFTensor] = []
        names: set[str] = set()
        for index in range(tensor_count):
            name = reader.string(what=f"tensor name {index}")
            if not name:
                raise ValueError("GGUF tensor name must not be empty")
            if name in names:
                raise ValueError(f"duplicate tensor name {name!r}")
            names.add(name)
            (dimension_count,) = reader.unpack(
                "<I", what=f"dimension count for {name!r}"
            )
            assert isinstance(dimension_count, int)
            if not 1 <= dimension_count <= _MAX_DIMENSIONS:
                raise ValueError(
                    f"invalid dimension count {dimension_count} for tensor {name!r}"
                )
            dims = reader.unpack(
                f"<{dimension_count}Q", what=f"dimensions for {name!r}"
            )
            (ggml_type, offset) = reader.unpack(
                "<IQ", what=f"type and offset for {name!r}"
            )
            assert isinstance(ggml_type, int)
            assert isinstance(offset, int)
            tensor = GGUFTensor(
                name, tuple(int(dim) for dim in dims), ggml_type, offset
            )
            # This validates dimensions, block divisibility, and the type before
            # untrusted offsets participate in span arithmetic.
            tensor_nbytes(tensor)
            tensors.append(tensor)

        declared_tensor_count = mutable_metadata.get("split.tensors.count")
        if declared_tensor_count is not None and declared_tensor_count != len(tensors):
            raise ValueError(
                "GGUF metadata 'split.tensors.count' must equal the "
                f"header tensor count {len(tensors)}, got {declared_tensor_count!r}"
            )
        alignment = _checked_alignment(mutable_metadata)
        data_offset = (reader.position + alignment - 1) & ~(alignment - 1)
        if data_offset > file_size:
            raise ValueError("GGUF tensor data section starts beyond end of file")
        if _file_identity(os.fstat(file.fileno())) != identity:
            raise ValueError(f"GGUF source changed during inspection: {path}")

    spans: list[tuple[int, int, str]] = []
    for tensor in tensors:
        if tensor.offset % alignment:
            raise ValueError(
                f"tensor {tensor.name!r} offset is not aligned to {alignment}"
            )
        start = data_offset + tensor.offset
        size = tensor_nbytes(tensor)
        if start > file_size or size > file_size - start:
            raise ValueError(f"tensor {tensor.name!r} span is outside the GGUF file")
        spans.append((start, start + size, tensor.name))
    spans.sort()
    for previous, current in zip(spans, spans[1:], strict=False):
        if current[0] < previous[1]:
            raise ValueError(
                f"tensor spans overlap: {previous[2]!r} and {current[2]!r}"
            )

    return GGUFFile(
        path=path,
        version=version,
        alignment=alignment,
        data_offset=data_offset,
        file_size=file_size,
        metadata=MappingProxyType(dict(mutable_metadata)),
        tensors=tuple(tensors),
        metadata_types=MappingProxyType(dict(mutable_metadata_types)),
        identity=identity,
    )


def inspect_kimi_k3_source(root: Path, revision: str) -> KimiK3Inventory:
    """Validate and install the exact pinned Kimi K3 source inventory."""

    if revision != KIMI_K3_SOURCE_REVISION:
        raise ValueError(
            "Kimi K3 source revision must be pinned to "
            f"{KIMI_K3_SOURCE_REVISION}, got {revision!r}"
        )
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Kimi K3 source root is not a directory: {root}")
    paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".gguf"
    )
    if len(paths) != KIMI_K3_SHARDS:
        raise ValueError(
            f"Kimi K3 source requires exactly 96 GGUF shards, found {len(paths)}"
        )

    numbered_paths: dict[int, Path] = {}
    for path in paths:
        match = _SHARD_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid Kimi K3 shard filename {path.name!r}")
        shard_number = int(match.group(1))
        shard_count = int(match.group(2))
        if shard_count != KIMI_K3_SHARDS:
            raise ValueError(f"shard {path.name!r} declares {shard_count}, expected 96")
        if shard_number in numbered_paths:
            raise ValueError(f"duplicate Kimi K3 shard number {shard_number}")
        numbered_paths[shard_number] = path
    expected_shards = set(range(1, KIMI_K3_SHARDS + 1))
    if set(numbered_paths) != expected_shards:
        missing = sorted(expected_shards - set(numbered_paths))
        extra = sorted(set(numbered_paths) - expected_shards)
        raise ValueError(
            f"Kimi K3 shard numbers must be 1..96; missing={missing}, extra={extra}"
        )

    files_list: list[GGUFFile] = []
    split_numbers: set[int] = set()
    seen_names: set[str] = set()
    expert_entries: list[tuple[int, int, GGUFTensor]] = []
    resident_tensors: list[GGUFTensor] = []
    layer_projections: dict[int, set[str]] = {}
    type_counts = {
        GGML_TYPE_F32: 0,
        GGML_TYPE_BF16: 0,
        GGML_TYPE_Q2_K: 0,
    }
    resident_bytes = 0
    routed_bytes = 0
    running_tensor_count = 0
    for shard_number in range(1, KIMI_K3_SHARDS + 1):
        source = read_gguf(numbered_paths[shard_number])
        for key in ("split.no", "split.count", "split.tensors.count"):
            if source.metadata_types.get(key) != _VALUE_UINT16:
                raise ValueError(
                    f"shard {shard_number} metadata {key!r} must use UINT16 type"
                )
        split_number = source.metadata.get("split.no")
        split_count = source.metadata.get("split.count")
        split_tensor_count = source.metadata.get("split.tensors.count")
        if (
            isinstance(split_number, bool)
            or not isinstance(split_number, int)
            or split_number != shard_number - 1
        ):
            raise ValueError(
                f"shard {shard_number} split.no must be {shard_number - 1}, "
                f"got {split_number!r}"
            )
        if (
            isinstance(split_count, bool)
            or not isinstance(split_count, int)
            or split_count != KIMI_K3_SHARDS
        ):
            raise ValueError(
                f"shard {shard_number} split.count must be 96, got {split_count!r}"
            )
        if (
            isinstance(split_tensor_count, bool)
            or not isinstance(split_tensor_count, int)
            or split_tensor_count != len(source.tensors)
        ):
            raise ValueError(
                f"shard {shard_number} split.tensors.count must be "
                f"its header tensor count {len(source.tensors)}, "
                f"got {split_tensor_count!r}"
            )
        if split_number in split_numbers:
            raise ValueError(f"duplicate Kimi K3 split number {split_number}")
        split_numbers.add(split_number)
        running_tensor_count += len(source.tensors)
        if running_tensor_count > KIMI_K3_TENSOR_COUNT:
            raise ValueError(
                f"Kimi K3 running tensor count exceeds {KIMI_K3_TENSOR_COUNT} "
                f"at shard {shard_number}: {running_tensor_count}"
            )

        for tensor in source.tensors:
            if tensor.name in seen_names:
                raise ValueError(
                    f"duplicate tensor name across Kimi K3 shards: {tensor.name!r}"
                )
            seen_names.add(tensor.name)
            if tensor.ggml_type != GGML_TYPE_Q2_K:
                if tensor.ggml_type not in (GGML_TYPE_BF16, GGML_TYPE_F32):
                    raise ValueError(
                        f"resident tensor type must be BF16 or F32: "
                        f"{tensor.name!r} has {tensor.ggml_type}"
                    )
                type_counts[tensor.ggml_type] += 1
                resident_bytes += tensor_nbytes(tensor)
                resident_tensors.append(tensor)
                continue

            match = _EXPERT_RE.fullmatch(tensor.name)
            if match is None:
                raise ValueError(
                    f"unexpected Q2_K tensor outside routed experts: {tensor.name!r}"
                )
            layer = int(match.group(1))
            projection = match.group(2)
            if layer not in KIMI_K3_LAYERS:
                raise ValueError(
                    f"Q2_K routed expert layer must be 1..92: {tensor.name!r}"
                )
            expected_dims = _EXPECTED_EXPERT_DIMS[projection]
            if tensor.dims != expected_dims:
                raise ValueError(
                    f"wrong Kimi K3 expert tensor shape for {tensor.name!r}: "
                    f"expected GGUF dims {expected_dims}, got {tensor.dims}"
                )
            layer_projections.setdefault(layer, set()).add(projection)
            type_counts[GGML_TYPE_Q2_K] += 1
            routed_bytes += tensor_nbytes(tensor)
            expert_entries.append((layer, _PROJECTION_ORDER[projection], tensor))
        files_list.append(source)

    expected_splits = set(range(KIMI_K3_SHARDS))
    if split_numbers != expected_splits:
        raise ValueError("Kimi K3 split numbers must be exactly 0..95")
    files = tuple(files_list)
    tensor_count = running_tensor_count
    if tensor_count != KIMI_K3_TENSOR_COUNT:
        raise ValueError(
            f"Kimi K3 tensor count must be {KIMI_K3_TENSOR_COUNT}, found {tensor_count}"
        )
    expected_type_counts = {
        GGML_TYPE_F32: KIMI_K3_F32_COUNT,
        GGML_TYPE_BF16: KIMI_K3_BF16_COUNT,
        GGML_TYPE_Q2_K: KIMI_K3_Q2_K_COUNT,
    }
    if type_counts != expected_type_counts:
        raise ValueError(
            "Kimi K3 tensor type counts must be "
            f"F32={KIMI_K3_F32_COUNT}, BF16={KIMI_K3_BF16_COUNT}, "
            f"Q2_K={KIMI_K3_Q2_K_COUNT}; found "
            f"F32={type_counts[GGML_TYPE_F32]}, "
            f"BF16={type_counts[GGML_TYPE_BF16]}, "
            f"Q2_K={type_counts[GGML_TYPE_Q2_K]}"
        )
    if resident_bytes != KIMI_K3_RESIDENT_BYTES:
        raise ValueError(
            f"Kimi K3 resident payload must be {KIMI_K3_RESIDENT_BYTES} bytes, "
            f"found {resident_bytes}"
        )
    if routed_bytes != KIMI_K3_ROUTED_BYTES:
        raise ValueError(
            f"Kimi K3 routed payload must be {KIMI_K3_ROUTED_BYTES} bytes, "
            f"found {routed_bytes}"
        )
    expected_descriptor_digest = KIMI_K3_RESIDENT_DESCRIPTOR_SHA256
    if re.fullmatch(r"[0-9a-f]{64}", expected_descriptor_digest) is None:
        raise ValueError("pinned Kimi K3 resident descriptor digest is invalid")
    actual_descriptor_digest = resident_descriptor_sha256(files)
    if actual_descriptor_digest != expected_descriptor_digest:
        raise ValueError(
            "Kimi K3 resident descriptor digest mismatch: expected "
            f"{expected_descriptor_digest}, found {actual_descriptor_digest}"
        )
    if len(expert_entries) != KIMI_K3_Q2_K_COUNT:
        raise ValueError(
            f"Kimi K3 source requires exactly {KIMI_K3_Q2_K_COUNT} Q2_K tensors, "
            f"found {len(expert_entries)}"
        )
    expected_projections = set(_PROJECTION_ORDER)
    if set(layer_projections) != set(KIMI_K3_LAYERS):
        missing_layers = sorted(set(KIMI_K3_LAYERS) - set(layer_projections))
        extra_layers = sorted(set(layer_projections) - set(KIMI_K3_LAYERS))
        raise ValueError(
            f"Kimi K3 routed layers must be exactly 1..92; "
            f"missing={missing_layers}, extra={extra_layers}"
        )
    for layer in KIMI_K3_LAYERS:
        if layer_projections[layer] != expected_projections:
            raise ValueError(
                f"layer {layer} must contain exactly gate/up/down merged "
                f"expert tensors, got {sorted(layer_projections[layer])}"
            )

    expert_entries.sort(key=lambda entry: (entry[0], entry[1]))
    resident_tensors.sort(key=lambda tensor: tensor.name)
    return KimiK3Inventory(
        revision=revision,
        files=files,
        expert_tensors=tuple(entry[2] for entry in expert_entries),
        resident_tensors=tuple(resident_tensors),
        layers=KIMI_K3_LAYERS,
        resident_descriptor_sha256=actual_descriptor_digest,
    )


def dequantize_q2_k(
    blob: bytes | bytearray | memoryview,
    *,
    value_count: int,
) -> np.ndarray:
    """Decode complete GGML Q2_K blocks into a flat FP32 array."""

    if value_count < 0 or value_count % Q2_K_VALUES:
        raise ValueError("Q2_K value_count must be a non-negative multiple of 256")
    block_count = value_count // Q2_K_VALUES
    expected_bytes = block_count * Q2_K_BYTES
    view = memoryview(blob)
    if view.nbytes != expected_bytes:
        raise ValueError(
            f"Q2_K byte length mismatch: expected {expected_bytes}, got {view.nbytes}"
        )
    if value_count == 0:
        return np.empty(0, dtype=np.float32)

    raw = np.frombuffer(view, dtype=np.uint8)
    decoded = np.empty(value_count, dtype=np.float32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    for first in range(0, block_count, _Q2_K_BATCH_BLOCKS):
        count = min(_Q2_K_BATCH_BLOCKS, block_count - first)
        blocks = raw[first * Q2_K_BYTES : (first + count) * Q2_K_BYTES].reshape(
            count, Q2_K_BYTES
        )
        scales = blocks[:, :16]
        qs = blocks[:, 16:80]
        d = blocks[:, 80:82].copy().view("<f2").astype(np.float32)
        dmin = blocks[:, 82:84].copy().view("<f2").astype(np.float32)

        scale = (d * (scales & 0x0F).astype(np.float32)).reshape(count, 16, 1)
        minimum = (dmin * (scales >> 4).astype(np.float32)).reshape(count, 16, 1)
        quant = ((qs.reshape(count, 2, 1, 32) >> shifts) & np.uint8(0x03)).reshape(
            count, 16, 16
        )
        batch = scale * quant.astype(np.float32) - minimum
        decoded[first * Q2_K_VALUES : (first + count) * Q2_K_VALUES] = batch.reshape(-1)
    return decoded
