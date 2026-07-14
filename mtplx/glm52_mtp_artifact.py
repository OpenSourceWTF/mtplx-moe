"""Strict exact-bit extraction for the GLM-5.2 layer-78 MTP artifact.

The extractor deliberately treats safetensors as a bounded container format.
It never materializes or decodes BF16 values: validated byte ranges are copied
from pinned source descriptors into a newly published sibling artifact.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence


SOURCE_REPO = "zai-org/GLM-5.2"
SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
SOURCE_DIRECTORY = "glm52-mtp-layer78-source"
LAYER_PREFIX = "model.layers.78."

ARTIFACT_FILE = "layer78-bf16.safetensors"
MANIFEST_FILE = "mtp-artifact-manifest.json"
MANIFEST_SCHEMA = "mtplx-glm52-mtp-layer78-v1"

EXPECTED_TENSOR_COUNT = 791
EXPECTED_PAYLOAD_BYTES = 19_905_841_664
EXPECTED_BF16_COUNT = 790
EXPECTED_F32_COUNT = 1

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
MAX_TENSORS_PER_FILE = 100_000
COPY_CHUNK_BYTES = 8 * 1024 * 1024

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


class ArtifactError(RuntimeError):
    """Base class for artifact construction failures."""


class ArtifactValidationError(ArtifactError):
    """Raised when an input or published artifact fails closed validation."""


class ArtifactPublicationError(ArtifactError):
    """Raised when an artifact cannot be published without replacement."""


@dataclass(frozen=True)
class SourceFilePin:
    size: int
    sha256: str


PINNED_SOURCE_FILES: dict[str, SourceFilePin] = {
    "config.json": SourceFilePin(
        3_732,
        "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
    ),
    "model.safetensors.index.json": SourceFilePin(
        5_408_032,
        "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
    ),
    "model-00270-of-00282.safetensors": SourceFilePin(
        5_366_430_968,
        "d74106256f061e73000e9660d157bd22254d2a5692cf9466d76dfea6985c0924",
    ),
    "model-00271-of-00282.safetensors": SourceFilePin(
        5_360_347_304,
        "90ba74c758309888b9d3f17adc189e32e77cee77ca8f7892ad12a9b38956cd43",
    ),
    "model-00272-of-00282.safetensors": SourceFilePin(
        5_360_347_320,
        "d5c9dbfba6aff2be069079cf39c8991393a2b69469a4f0ac4e246af25d519a06",
    ),
    "model-00273-of-00282.safetensors": SourceFilePin(
        5_360_347_208,
        "1344c75f27e5564baa46641ebcdf19a2a13ac1b2cfc52e01744b6dda1127aa94",
    ),
    "model-00274-of-00282.safetensors": SourceFilePin(
        5_359_997_688,
        "1943b335a5aa626389e819fa5a7339c361844c425389b35328eba0142935fbf6",
    ),
}

EXPECTED_SHARD_COUNTS = {
    "model-00270-of-00282.safetensors": 4,
    "model-00271-of-00282.safetensors": 213,
    "model-00272-of-00282.safetensors": 213,
    "model-00273-of-00282.safetensors": 213,
    "model-00274-of-00282.safetensors": 148,
}

FIXED_LAYER78: dict[str, tuple[str, tuple[int, ...]]] = {
    "eh_proj.weight": ("BF16", (6144, 12288)),
    "enorm.weight": ("BF16", (6144,)),
    "hnorm.weight": ("BF16", (6144,)),
    "input_layernorm.weight": ("BF16", (6144,)),
    "mlp.gate.e_score_correction_bias": ("F32", (256,)),
    "mlp.gate.weight": ("BF16", (256, 6144)),
    "mlp.shared_experts.down_proj.weight": ("BF16", (6144, 2048)),
    "mlp.shared_experts.gate_proj.weight": ("BF16", (2048, 6144)),
    "mlp.shared_experts.up_proj.weight": ("BF16", (2048, 6144)),
    "post_attention_layernorm.weight": ("BF16", (6144,)),
    "self_attn.indexer.k_norm.bias": ("BF16", (128,)),
    "self_attn.indexer.k_norm.weight": ("BF16", (128,)),
    "self_attn.indexer.weights_proj.weight": ("BF16", (32, 6144)),
    "self_attn.indexer.wk.weight": ("BF16", (128, 6144)),
    "self_attn.indexer.wq_b.weight": ("BF16", (4096, 2048)),
    "self_attn.kv_a_layernorm.weight": ("BF16", (512,)),
    "self_attn.kv_a_proj_with_mqa.weight": ("BF16", (576, 6144)),
    "self_attn.kv_b_proj.weight": ("BF16", (28672, 512)),
    "self_attn.o_proj.weight": ("BF16", (6144, 16384)),
    "self_attn.q_a_layernorm.weight": ("BF16", (2048,)),
    "self_attn.q_a_proj.weight": ("BF16", (2048, 6144)),
    "self_attn.q_b_proj.weight": ("BF16", (16384, 2048)),
    "shared_head.norm.weight": ("BF16", (6144,)),
}

EXPECTED_CONFIG_VALUES: dict[str, object] = {
    "hidden_size": 6144,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_hidden_layers": 78,
    "num_nextn_predict_layers": 1,
    "model_type": "glm_moe_dsa",
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "num_attention_heads": 64,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "n_shared_experts": 1,
}

MANIFEST_KEYS = frozenset(
    {"schema", "source", "inventory", "artifact", "producer", "manifest_sha256"}
)
SOURCE_MANIFEST_KEYS = frozenset({"repo", "revision", "directory", "files"})
SOURCE_FILE_MANIFEST_KEYS = frozenset({"name", "bytes", "sha256"})
INVENTORY_MANIFEST_KEYS = frozenset(
    {"tensor_count", "payload_bytes", "dtype_counts", "source_shard_distribution"}
)
ARTIFACT_MANIFEST_KEYS = frozenset(
    {
        "file",
        "file_bytes",
        "sha256",
        "header_bytes",
        "header_sha256",
        "payload_bytes",
        "tensors",
    }
)
TENSOR_MANIFEST_KEYS = frozenset(
    {
        "name",
        "dtype",
        "shape",
        "source_file",
        "source_data_offsets",
        "output_data_offsets",
        "sha256",
    }
)
PRODUCER_MANIFEST_KEYS = frozenset({"commit", "clean"})


@dataclass(frozen=True)
class TensorExpectation:
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in self.shape
        ):
            raise ValueError(f"invalid tensor shape {self.shape!r}")

    @property
    def nbytes(self) -> int:
        return _checked_tensor_bytes(self.dtype, self.shape)


@dataclass(frozen=True)
class Glm52MtpArtifactConfig:
    source_root: Path
    output_root: Path
    producer_root: Path


@dataclass(frozen=True)
class VerifiedGlm52MtpArtifact:
    """Authenticated artifact receipt plus its still-open exact file handle."""

    root: Path
    manifest: dict[str, object]
    file: BinaryIO


@dataclass(frozen=True)
class SourceFileIdentity:
    name: str
    size: int
    sha256: str
    device: int
    inode: int
    uid: int
    mode: int
    links: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class PlannedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: str
    source_data_offsets: tuple[int, int]
    source_absolute_offset: int

    @property
    def nbytes(self) -> int:
        return self.source_data_offsets[1] - self.source_data_offsets[0]


@dataclass(frozen=True)
class ExtractionPlan:
    tensors: tuple[PlannedTensor, ...]
    source_files: tuple[SourceFileIdentity, ...]
    producer_commit: str
    shard_distribution: dict[str, int]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def payload_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)


@dataclass(frozen=True)
class _ParsedTensor:
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]


@dataclass(frozen=True)
class _ParsedSafetensors:
    tensors: dict[str, _ParsedTensor]
    data_start: int
    header_bytes: bytes
    metadata: dict[str, str]


def _checked_tensor_bytes(dtype: str, shape: Sequence[int]) -> int:
    width = DTYPE_BYTES.get(dtype)
    if width is None:
        raise ArtifactValidationError(f"unsupported safetensors dtype {dtype!r}")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ArtifactValidationError(f"invalid tensor shape {tuple(shape)!r}")
    elements = math.prod(shape) if shape else 1
    if elements > (1 << 63) // width:
        raise ArtifactValidationError("tensor byte count overflows the bounded range")
    return elements * width


def expected_glm52_layer78_inventory(
    config: Mapping[str, object],
) -> dict[str, TensorExpectation]:
    """Return the one accepted GLM-5.2 layer-78 tensor inventory."""

    for key, expected in EXPECTED_CONFIG_VALUES.items():
        actual = config.get(key)
        if actual != expected or type(actual) is not type(expected):
            raise ArtifactValidationError(
                f"config field {key!r} is {actual!r}; expected pinned value {expected!r}"
            )

    inventory = {
        LAYER_PREFIX + suffix: TensorExpectation(dtype, shape)
        for suffix, (dtype, shape) in FIXED_LAYER78.items()
    }
    for expert in range(256):
        prefix = f"{LAYER_PREFIX}mlp.experts.{expert}."
        inventory[prefix + "gate_proj.weight"] = TensorExpectation("BF16", (2048, 6144))
        inventory[prefix + "up_proj.weight"] = TensorExpectation("BF16", (2048, 6144))
        inventory[prefix + "down_proj.weight"] = TensorExpectation("BF16", (6144, 2048))

    _validate_inventory_totals(inventory)
    return dict(sorted(inventory.items()))


def _validate_inventory_totals(inventory: Mapping[str, TensorExpectation]) -> None:
    tensor_count = len(inventory)
    payload_bytes = sum(tensor.nbytes for tensor in inventory.values())
    bf16_count = sum(tensor.dtype == "BF16" for tensor in inventory.values())
    f32_count = sum(tensor.dtype == "F32" for tensor in inventory.values())
    if tensor_count != EXPECTED_TENSOR_COUNT:
        raise ArtifactValidationError(
            f"layer-78 inventory has {tensor_count} tensors; expected {EXPECTED_TENSOR_COUNT}"
        )
    if payload_bytes != EXPECTED_PAYLOAD_BYTES:
        raise ArtifactValidationError(
            f"layer-78 inventory has {payload_bytes} payload bytes; "
            f"expected {EXPECTED_PAYLOAD_BYTES}"
        )
    if bf16_count != EXPECTED_BF16_COUNT or f32_count != EXPECTED_F32_COUNT:
        raise ArtifactValidationError(
            "layer-78 inventory dtype counts do not match the pinned BF16/F32 split"
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, label: str) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=_strict_object)
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"malformed JSON in {label}: {exc}") from exc


def _pread_exact(fd: int, count: int, offset: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    cursor = offset
    while remaining:
        try:
            chunk = os.pread(fd, remaining, cursor)
        except OSError as exc:
            raise ArtifactValidationError(f"failed to read {label}: {exc}") from exc
        if not chunk:
            actual = count - remaining
            raise ArtifactValidationError(
                f"short read for {label}: received {actual} of {count} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
        cursor += len(chunk)
    return b"".join(chunks)


def _pwrite_exact(fd: int, data: bytes, offset: int, *, label: str) -> None:
    view = memoryview(data)
    cursor = offset
    while view:
        try:
            written = os.pwrite(fd, view, cursor)
        except OSError as exc:
            raise ArtifactPublicationError(f"failed to write {label}: {exc}") from exc
        if written <= 0:
            raise ArtifactPublicationError(f"short write for {label}")
        view = view[written:]
        cursor += written


def _sha256_fd(fd: int, size: int, *, offset: int = 0, label: str) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < size:
        count = min(COPY_CHUNK_BYTES, size - cursor)
        digest.update(_pread_exact(fd, count, offset + cursor, label=label))
        cursor += count
    return digest.hexdigest()


def _validate_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ArtifactValidationError(f"cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ArtifactValidationError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise ArtifactValidationError(f"{label} must be a directory: {path}")
    if info.st_uid != os.geteuid():
        raise ArtifactValidationError(f"unsafe owner for {label}: {path}")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ArtifactValidationError(f"unsafe mode for {label}: {path}")
    return info


@contextlib.contextmanager
def _open_regular_nofollow(path: Path, *, label: str):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ArtifactValidationError(
                f"{label} must not be a symlink: {path}"
            ) from exc
        raise ArtifactValidationError(f"cannot open {label} {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactValidationError(f"{label} must be a regular file: {path}")
        if info.st_uid != os.geteuid():
            raise ArtifactValidationError(f"unsafe ownership for {label}: {path}")
        if info.st_nlink != 1:
            raise ArtifactValidationError(f"{label} must not be a hardlink: {path}")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ArtifactValidationError(f"unsafe mode for {label}: {path}")
        yield fd, info
    finally:
        os.close(fd)


def _identity(name: str, info: os.stat_result, sha256: str) -> SourceFileIdentity:
    return SourceFileIdentity(
        name=name,
        size=info.st_size,
        sha256=sha256,
        device=info.st_dev,
        inode=info.st_ino,
        uid=info.st_uid,
        mode=stat.S_IMODE(info.st_mode),
        links=info.st_nlink,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def _same_identity(identity: SourceFileIdentity, info: os.stat_result) -> bool:
    return (
        identity.size,
        identity.device,
        identity.inode,
        identity.uid,
        identity.mode,
        identity.links,
        identity.mtime_ns,
        identity.ctime_ns,
    ) == (
        info.st_size,
        info.st_dev,
        info.st_ino,
        info.st_uid,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _same_open_file_contents(
    expected: os.stat_result,
    actual: os.stat_result,
) -> bool:
    return (
        expected.st_dev,
        expected.st_ino,
        expected.st_size,
        expected.st_mtime_ns,
        expected.st_ctime_ns,
    ) == (
        actual.st_dev,
        actual.st_ino,
        actual.st_size,
        actual.st_mtime_ns,
        actual.st_ctime_ns,
    )


def _read_bounded_json_fd(fd: int, size: int, *, label: str) -> Mapping[str, object]:
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ArtifactValidationError(
            f"{label} size {size} is outside the bounded JSON range"
        )
    value = _parse_json(_pread_exact(fd, size, 0, label=label), label=label)
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} JSON root must be an object")
    return value


def _parse_safetensors_fd(fd: int, size: int, *, label: str) -> _ParsedSafetensors:
    if size < 10:
        raise ArtifactValidationError(f"{label} is too short to be safetensors")
    raw_length = _pread_exact(fd, 8, 0, label=f"{label} header length")
    header_size = struct.unpack("<Q", raw_length)[0]
    if header_size < 2 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
        raise ArtifactValidationError(
            f"{label} header size {header_size} is outside the bounded range"
        )
    data_start = 8 + header_size
    if data_start > size:
        raise ArtifactValidationError(f"{label} has a truncated header")
    header_bytes = _pread_exact(fd, header_size, 8, label=f"{label} header")
    value = _parse_json(header_bytes, label=f"{label} header")
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} header must be a JSON object")
    if len(value) > MAX_TENSORS_PER_FILE + 1:
        raise ArtifactValidationError(f"{label} contains too many tensor entries")

    parsed: dict[str, _ParsedTensor] = {}
    container_metadata: dict[str, str] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, metadata in value.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in metadata.items()
            ):
                raise ArtifactValidationError(f"{label} has invalid __metadata__")
            container_metadata = dict(metadata)
            continue
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError(f"{label} has an invalid tensor name")
        if not isinstance(metadata, dict) or set(metadata) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise ArtifactValidationError(
                f"{label} tensor {name!r} has invalid metadata keys"
            )
        dtype = metadata["dtype"]
        shape = metadata["shape"]
        offsets = metadata["data_offsets"]
        if not isinstance(dtype, str):
            raise ArtifactValidationError(f"{label} tensor {name!r} has invalid dtype")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            raise ArtifactValidationError(f"{label} tensor {name!r} has invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in offsets
            )
        ):
            raise ArtifactValidationError(
                f"{label} tensor {name!r} has invalid data offsets"
            )
        start, end = offsets
        if start < 0 or end < start:
            raise ArtifactValidationError(
                f"{label} tensor {name!r} has invalid data range"
            )
        expected_bytes = _checked_tensor_bytes(dtype, shape)
        if end - start != expected_bytes:
            raise ArtifactValidationError(
                f"{label} tensor {name!r} dtype/shape byte-count mismatch: "
                f"range={end - start}, expected={expected_bytes}"
            )
        if end > size - data_start:
            raise ArtifactValidationError(
                f"{label} tensor {name!r} extends beyond the file"
            )
        parsed[name] = _ParsedTensor(dtype, tuple(shape), (start, end))
        ranges.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(ranges):
        if start < cursor:
            raise ArtifactValidationError(
                f"{label} has overlapping tensor range at {name!r}"
            )
        if start != cursor:
            raise ArtifactValidationError(
                f"{label} tensor ranges are not contiguous at {name!r}"
            )
        cursor = end
    if cursor != size - data_start:
        raise ArtifactValidationError(
            f"{label} has trailing data after the final declared tensor range"
        )
    return _ParsedSafetensors(parsed, data_start, header_bytes, container_metadata)


def _producer_identity(root: Path) -> str:
    _validate_directory(root, label="producer root")

    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", os.fspath(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ArtifactValidationError(
                f"cannot authenticate producer repository: {exc}"
            ) from exc
        return result.stdout.strip()

    top_level = Path(run("rev-parse", "--show-toplevel"))
    if os.path.normcase(os.path.realpath(top_level)) != os.path.normcase(
        os.path.realpath(root)
    ):
        raise ArtifactValidationError(
            "producer root is not the authenticated Git worktree root"
        )
    commit = run("rev-parse", "--verify", "HEAD")
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ArtifactValidationError(
            "producer commit is not a full hexadecimal object ID"
        )
    dirty = run("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ArtifactValidationError("producer worktree is dirty")
    return commit


def _inspect_source(
    source_root: Path,
) -> tuple[tuple[SourceFileIdentity, ...], tuple[PlannedTensor, ...], dict[str, int]]:
    _validate_directory(source_root, label="source root")
    identities: list[SourceFileIdentity] = []
    parsed_files: dict[str, _ParsedSafetensors] = {}
    json_files: dict[str, Mapping[str, object]] = {}

    with contextlib.ExitStack() as stack:
        descriptors: dict[str, tuple[int, os.stat_result]] = {}
        for name, pin in PINNED_SOURCE_FILES.items():
            fd, info = stack.enter_context(
                _open_regular_nofollow(source_root / name, label=f"source file {name}")
            )
            if info.st_size != pin.size:
                raise ArtifactValidationError(
                    f"source file {name} pinned size mismatch: {info.st_size} != {pin.size}"
                )
            digest = _sha256_fd(fd, info.st_size, label=f"source file {name}")
            if digest != pin.sha256:
                raise ArtifactValidationError(
                    f"source file {name} pinned SHA-256 mismatch"
                )
            identities.append(_identity(name, info, digest))
            descriptors[name] = (fd, info)

        for name in ("config.json", "model.safetensors.index.json"):
            fd, info = descriptors[name]
            json_files[name] = _read_bounded_json_fd(fd, info.st_size, label=name)
        for name in EXPECTED_SHARD_COUNTS:
            fd, info = descriptors[name]
            parsed_files[name] = _parse_safetensors_fd(fd, info.st_size, label=name)

        for identity in identities:
            fd, _initial = descriptors[identity.name]
            if not _same_identity(identity, os.fstat(fd)):
                raise ArtifactValidationError(
                    f"source file {identity.name} was replaced or mutated during preflight"
                )

    config = json_files["config.json"]
    inventory = expected_glm52_layer78_inventory(config)
    _validate_inventory_totals(inventory)
    index = json_files["model.safetensors.index.json"]
    if set(index) - {"metadata", "weight_map"}:
        raise ArtifactValidationError("model index has unexpected top-level keys")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ArtifactValidationError("model index weight_map must be a string mapping")
    indexed_layer = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith(LAYER_PREFIX)
    }
    expected_names = set(inventory)
    if set(indexed_layer) != expected_names:
        missing = sorted(expected_names - set(indexed_layer))
        extra = sorted(set(indexed_layer) - expected_names)
        raise ArtifactValidationError(
            f"layer-78 inventory mismatch in index: missing={missing[:3]}, extra={extra[:3]}"
        )
    distribution = {name: 0 for name in EXPECTED_SHARD_COUNTS}
    for tensor_name, shard_name in indexed_layer.items():
        if shard_name not in distribution:
            raise ArtifactValidationError(
                f"layer-78 tensor {tensor_name!r} maps to an unpinned shard {shard_name!r}"
            )
        distribution[shard_name] += 1
    if distribution != EXPECTED_SHARD_COUNTS:
        raise ArtifactValidationError(
            f"layer-78 source shard distribution mismatch: {distribution} != {EXPECTED_SHARD_COUNTS}"
        )

    locations: dict[str, tuple[str, _ParsedTensor, int]] = {}
    for shard_name, parsed in parsed_files.items():
        for tensor_name, metadata in parsed.tensors.items():
            if not tensor_name.startswith(LAYER_PREFIX):
                continue
            if tensor_name in locations:
                raise ArtifactValidationError(
                    f"duplicate tensor {tensor_name!r} across source shards"
                )
            locations[tensor_name] = (shard_name, metadata, parsed.data_start)
    if set(locations) != expected_names:
        missing = sorted(expected_names - set(locations))
        extra = sorted(set(locations) - expected_names)
        raise ArtifactValidationError(
            f"layer-78 inventory mismatch in source headers: missing={missing[:3]}, extra={extra[:3]}"
        )

    planned: list[PlannedTensor] = []
    for name in sorted(expected_names):
        shard_name, metadata, data_start = locations[name]
        expectation = inventory[name]
        if indexed_layer[name] != shard_name:
            raise ArtifactValidationError(
                f"index/header source disagreement for tensor {name!r}"
            )
        if metadata.dtype != expectation.dtype:
            raise ArtifactValidationError(
                f"wrong dtype for {name!r}: {metadata.dtype} != {expectation.dtype}"
            )
        if metadata.shape != expectation.shape:
            raise ArtifactValidationError(
                f"wrong shape for {name!r}: {metadata.shape} != {expectation.shape}"
            )
        planned.append(
            PlannedTensor(
                name=name,
                dtype=metadata.dtype,
                shape=metadata.shape,
                source_file=shard_name,
                source_data_offsets=metadata.offsets,
                source_absolute_offset=data_start + metadata.offsets[0],
            )
        )
    return (
        tuple(sorted(identities, key=lambda item: item.name)),
        tuple(planned),
        distribution,
    )


def preflight_glm52_mtp_layer78(config: Glm52MtpArtifactConfig) -> ExtractionPlan:
    """Authenticate all inputs and return the immutable extraction plan."""

    source_root = Path(config.source_root)
    output_root = Path(config.output_root)
    producer_root = Path(config.producer_root)
    if output_root.name in {"", ".", ".."}:
        raise ArtifactPublicationError("output root must name a new sibling directory")
    _validate_directory(output_root.parent, label="output parent")
    try:
        os.lstat(output_root)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ArtifactPublicationError(f"cannot inspect output root: {exc}") from exc
    else:
        raise ArtifactPublicationError(f"output root already exists: {output_root}")

    producer_commit = _producer_identity(producer_root)
    source_files, tensors, distribution = _inspect_source(source_root)
    plan = ExtractionPlan(
        tensors=tensors,
        source_files=source_files,
        producer_commit=producer_commit,
        shard_distribution=distribution,
    )
    if (
        plan.tensor_count != EXPECTED_TENSOR_COUNT
        or plan.payload_bytes != EXPECTED_PAYLOAD_BYTES
    ):
        raise ArtifactValidationError(
            "preflight plan does not match the exact inventory totals"
        )
    return plan


def _assert_held_source_identity(
    source_root: Path,
    identity: SourceFileIdentity,
    fd: int,
) -> None:
    info = os.fstat(fd)
    if not _same_identity(identity, info):
        raise ArtifactValidationError(
            f"source file {identity.name} was replaced or mutated during extraction"
        )
    try:
        path_info = os.lstat(source_root / identity.name)
    except OSError as exc:
        raise ArtifactValidationError(
            f"source file {identity.name} was replaced during extraction"
        ) from exc
    if not _same_identity(identity, path_info):
        raise ArtifactValidationError(
            f"source file {identity.name} was replaced or mutated during extraction"
        )


def _copy_tensor_bytes(
    source_fd: int,
    output_fd: int,
    *,
    source_offset: int,
    output_offset: int,
    size: int,
    label: str,
) -> str:
    digest = hashlib.sha256()
    copied = 0
    while copied < size:
        count = min(COPY_CHUNK_BYTES, size - copied)
        chunk = _pread_exact(
            source_fd,
            count,
            source_offset + copied,
            label=f"source tensor {label}",
        )
        _pwrite_exact(
            output_fd,
            chunk,
            output_offset + copied,
            label=f"output tensor {label}",
        )
        digest.update(chunk)
        copied += count
    return digest.hexdigest()


def _output_header(
    plan: ExtractionPlan,
) -> tuple[bytes, list[tuple[PlannedTensor, tuple[int, int]]]]:
    metadata: dict[str, object] = {
        "__metadata__": {
            "schema": MANIFEST_SCHEMA,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "extracted_prefix": LAYER_PREFIX,
            "producer_commit": plan.producer_commit,
        }
    }
    rows: list[tuple[PlannedTensor, tuple[int, int]]] = []
    offset = 0
    for tensor in plan.tensors:
        output_offsets = (offset, offset + tensor.nbytes)
        metadata[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": list(output_offsets),
        }
        rows.append((tensor, output_offsets))
        offset = output_offsets[1]
    encoded = _canonical_json(metadata)
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    if len(encoded) > MAX_SAFETENSORS_HEADER_BYTES:
        raise ArtifactValidationError(
            "output safetensors header exceeds the bounded size"
        )
    return encoded, rows


def _verify_written_output(
    fd: int,
    *,
    header_bytes: bytes,
    rows: list[tuple[PlannedTensor, tuple[int, int]]],
    expected_tensor_digests: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    prefix = struct.pack("<Q", len(header_bytes)) + header_bytes
    file_digest = hashlib.sha256(prefix)
    data_start = len(prefix)
    verified: dict[str, str] = {}
    for tensor, offsets in rows:
        tensor_digest = hashlib.sha256()
        cursor = offsets[0]
        while cursor < offsets[1]:
            count = min(COPY_CHUNK_BYTES, offsets[1] - cursor)
            chunk = _pread_exact(
                fd,
                count,
                data_start + cursor,
                label=f"written tensor {tensor.name}",
            )
            file_digest.update(chunk)
            tensor_digest.update(chunk)
            cursor += count
        actual = tensor_digest.hexdigest()
        if actual != expected_tensor_digests[tensor.name]:
            raise ArtifactValidationError(
                f"tensor SHA-256 mismatch after write: {tensor.name}"
            )
        verified[tensor.name] = actual
    return file_digest.hexdigest(), verified


def _verify_artifact_digests(
    fd: int,
    parsed: _ParsedSafetensors,
    tensor_rows: Mapping[str, Mapping[str, object]],
) -> str:
    """Hash the artifact file and every tensor in one physical-order pass."""

    file_digest = hashlib.sha256(
        struct.pack("<Q", len(parsed.header_bytes)) + parsed.header_bytes
    )
    for name, metadata in sorted(
        parsed.tensors.items(), key=lambda item: item[1].offsets[0]
    ):
        tensor_digest = hashlib.sha256()
        cursor = metadata.offsets[0]
        while cursor < metadata.offsets[1]:
            count = min(COPY_CHUNK_BYTES, metadata.offsets[1] - cursor)
            chunk = _pread_exact(
                fd,
                count,
                parsed.data_start + cursor,
                label=f"artifact tensor {name}",
            )
            file_digest.update(chunk)
            tensor_digest.update(chunk)
            cursor += count
        if tensor_digest.hexdigest() != tensor_rows[name]["sha256"]:
            raise ArtifactValidationError(f"tensor SHA-256 mismatch for {name}")
    return file_digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_directory_exclusive(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    result: int
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex = libc.renamex_np
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        result = renamex(source_raw, destination_raw, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100, source_raw, -100, destination_raw, 1
        )  # RENAME_NOREPLACE
    else:
        raise ArtifactPublicationError(
            "platform lacks an atomic no-replace directory rename primitive"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArtifactPublicationError(f"output root already exists: {destination}")
    raise ArtifactPublicationError(
        f"atomic artifact publication failed: {os.strerror(error)}"
    )


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _write_new_file(path: Path, data: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ArtifactPublicationError(f"cannot create {path.name}: {exc}") from exc
    try:
        _pwrite_exact(fd, data, 0, label=path.name)
        os.fsync(fd)
    finally:
        os.close(fd)


def extract_glm52_mtp_layer78(config: Glm52MtpArtifactConfig) -> Path:
    """Extract and atomically publish the authenticated exact-bit artifact."""

    plan = preflight_glm52_mtp_layer78(config)
    source_root = Path(config.source_root)
    output_root = Path(config.output_root)
    output_parent = output_root.parent
    _validate_directory(output_parent, label="output parent")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=os.fspath(output_parent)
        )
    )
    os.chmod(staging, 0o700)
    published = False
    try:
        header_bytes, rows = _output_header(plan)
        data_start = 8 + len(header_bytes)
        artifact_path = staging / ARTIFACT_FILE
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        output_fd = os.open(artifact_path, flags, 0o600)
        try:
            total_size = data_start + plan.payload_bytes
            os.ftruncate(output_fd, total_size)
            _pwrite_exact(
                output_fd,
                struct.pack("<Q", len(header_bytes)) + header_bytes,
                0,
                label="artifact header",
            )
            identities = {identity.name: identity for identity in plan.source_files}
            with contextlib.ExitStack() as stack:
                held: dict[str, int] = {}
                for name, identity in identities.items():
                    fd, _info = stack.enter_context(
                        _open_regular_nofollow(
                            source_root / name, label=f"source file {name}"
                        )
                    )
                    _assert_held_source_identity(source_root, identity, fd)
                    held[name] = fd

                copy_digests: dict[str, str] = {}
                for tensor, output_offsets in rows:
                    copy_digests[tensor.name] = _copy_tensor_bytes(
                        held[tensor.source_file],
                        output_fd,
                        source_offset=tensor.source_absolute_offset,
                        output_offset=data_start + output_offsets[0],
                        size=tensor.nbytes,
                        label=tensor.name,
                    )
                os.fsync(output_fd)
                artifact_sha256, verified_digests = _verify_written_output(
                    output_fd,
                    header_bytes=header_bytes,
                    rows=rows,
                    expected_tensor_digests=copy_digests,
                )
                for name, fd in held.items():
                    _assert_held_source_identity(source_root, identities[name], fd)

            if _producer_identity(Path(config.producer_root)) != plan.producer_commit:
                raise ArtifactValidationError(
                    "producer commit changed during extraction"
                )
            info = os.fstat(output_fd)
            if info.st_size != total_size:
                raise ArtifactValidationError(
                    "published artifact byte count changed after write"
                )
        finally:
            os.close(output_fd)

        source_manifest = {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "directory": source_root.name,
            "files": [
                {"name": item.name, "bytes": item.size, "sha256": item.sha256}
                for item in plan.source_files
            ],
        }
        tensor_manifest = []
        for tensor, output_offsets in rows:
            tensor_manifest.append(
                {
                    "name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "source_file": tensor.source_file,
                    "source_data_offsets": list(tensor.source_data_offsets),
                    "output_data_offsets": list(output_offsets),
                    "sha256": verified_digests[tensor.name],
                }
            )
        manifest: dict[str, object] = {
            "schema": MANIFEST_SCHEMA,
            "source": source_manifest,
            "inventory": {
                "tensor_count": plan.tensor_count,
                "payload_bytes": plan.payload_bytes,
                "dtype_counts": {
                    "BF16": sum(tensor.dtype == "BF16" for tensor in plan.tensors),
                    "F32": sum(tensor.dtype == "F32" for tensor in plan.tensors),
                },
                "source_shard_distribution": plan.shard_distribution,
            },
            "artifact": {
                "file": ARTIFACT_FILE,
                "file_bytes": total_size,
                "sha256": artifact_sha256,
                "header_bytes": len(header_bytes),
                "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                "payload_bytes": plan.payload_bytes,
                "tensors": tensor_manifest,
            },
            "producer": {"commit": plan.producer_commit, "clean": True},
        }
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        _write_new_file(staging / MANIFEST_FILE, _canonical_json(manifest) + b"\n")
        _fsync_directory(staging)

        # Verify the exact bytes and manifest before making the directory visible.
        verify_glm52_mtp_layer78(staging, deep=True)
        _rename_directory_exclusive(staging, output_root)
        published = True
        _fsync_directory(output_parent)
        return output_root
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _require_exact_keys(
    value: object, keys: frozenset[str], *, label: str
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ArtifactValidationError(
            f"{label} keys mismatch: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _validate_hex_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / MANIFEST_FILE
    try:
        with _open_regular_nofollow(path, label="artifact manifest") as (fd, info):
            if info.st_size <= 0 or info.st_size > MAX_JSON_BYTES:
                raise ArtifactValidationError(
                    "artifact manifest size is outside the bounded range"
                )
            raw = _pread_exact(fd, info.st_size, 0, label="artifact manifest")
    except ArtifactValidationError as exc:
        raise ArtifactValidationError(
            f"authenticated manifest unavailable: {exc}"
        ) from exc
    value = _parse_json(raw, label="artifact manifest")
    manifest = dict(_require_exact_keys(value, MANIFEST_KEYS, label="manifest"))
    if raw != _canonical_json(manifest) + b"\n":
        raise ArtifactValidationError("artifact manifest is not in canonical JSON form")
    actual_digest = _validate_hex_digest(
        manifest["manifest_sha256"], label="manifest digest"
    )
    if _manifest_digest(manifest) != actual_digest:
        raise ArtifactValidationError("manifest SHA-256 mismatch")
    return manifest


def _verify_glm52_mtp_layer78_held(
    root: Path,
    manifest: dict[str, object],
    artifact_fd: int,
    artifact_info: os.stat_result,
    *,
    deep: bool,
) -> dict[str, object]:
    """Verify manifest, held artifact bytes, and pinned source as one receipt."""

    source = _require_exact_keys(
        manifest["source"], SOURCE_MANIFEST_KEYS, label="source"
    )
    if source["repo"] != SOURCE_REPO or source["revision"] != SOURCE_REVISION:
        raise ArtifactValidationError("manifest source repo/revision mismatch")
    if source["directory"] != SOURCE_DIRECTORY:
        raise ArtifactValidationError("manifest source staging directory mismatch")
    source_files = source["files"]
    if not isinstance(source_files, list):
        raise ArtifactValidationError("source files must be a list")
    expected_source_rows = [
        {"name": name, "bytes": pin.size, "sha256": pin.sha256}
        for name, pin in sorted(PINNED_SOURCE_FILES.items())
    ]
    for row in source_files:
        _require_exact_keys(row, SOURCE_FILE_MANIFEST_KEYS, label="source file")
    if source_files != expected_source_rows:
        raise ArtifactValidationError(
            "manifest source file identities do not match pinned inputs"
        )

    inventory = _require_exact_keys(
        manifest["inventory"], INVENTORY_MANIFEST_KEYS, label="inventory"
    )
    if inventory["tensor_count"] != EXPECTED_TENSOR_COUNT:
        raise ArtifactValidationError("manifest tensor count mismatch")
    if inventory["payload_bytes"] != EXPECTED_PAYLOAD_BYTES:
        raise ArtifactValidationError("manifest payload byte count mismatch")
    if inventory["dtype_counts"] != {
        "BF16": EXPECTED_BF16_COUNT,
        "F32": EXPECTED_F32_COUNT,
    }:
        raise ArtifactValidationError("manifest dtype counts mismatch")
    if inventory["source_shard_distribution"] != EXPECTED_SHARD_COUNTS:
        raise ArtifactValidationError("manifest source shard distribution mismatch")

    producer = _require_exact_keys(
        manifest["producer"], PRODUCER_MANIFEST_KEYS, label="producer"
    )
    commit = producer["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ArtifactValidationError("manifest producer commit is invalid")
    if producer["clean"] is not True:
        raise ArtifactValidationError("manifest producer clean-tree status is not true")

    artifact = _require_exact_keys(
        manifest["artifact"], ARTIFACT_MANIFEST_KEYS, label="artifact"
    )
    if artifact["file"] != ARTIFACT_FILE:
        raise ArtifactValidationError("manifest artifact filename mismatch")
    tensors = artifact["tensors"]
    if not isinstance(tensors, list) or len(tensors) != EXPECTED_TENSOR_COUNT:
        raise ArtifactValidationError(
            "manifest artifact tensor list has the wrong length"
        )
    tensor_rows: dict[str, Mapping[str, object]] = {}
    for row in tensors:
        checked = _require_exact_keys(
            row, TENSOR_MANIFEST_KEYS, label="artifact tensor"
        )
        name = checked["name"]
        if not isinstance(name, str) or name in tensor_rows:
            raise ArtifactValidationError(
                "manifest tensor names must be unique strings"
            )
        _validate_hex_digest(checked["sha256"], label=f"tensor {name} digest")
        tensor_rows[name] = checked
    if list(tensor_rows) != sorted(tensor_rows):
        raise ArtifactValidationError("manifest tensors must be sorted by name")

    with contextlib.nullcontext((artifact_fd, artifact_info)) as (fd, info):
        if info.st_size != artifact["file_bytes"]:
            raise ArtifactValidationError("artifact file byte count mismatch")
        parsed = _parse_safetensors_fd(fd, info.st_size, label="artifact file")
        if parsed.metadata.get("producer_commit") != commit:
            raise ArtifactValidationError("producer commit/header metadata mismatch")
        expected_header_metadata = {
            "schema": MANIFEST_SCHEMA,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "extracted_prefix": LAYER_PREFIX,
            "producer_commit": commit,
        }
        if parsed.metadata != expected_header_metadata:
            raise ArtifactValidationError(
                "artifact safetensors metadata does not match the verified receipt"
            )
        if len(parsed.header_bytes) != artifact["header_bytes"]:
            raise ArtifactValidationError("artifact header byte count mismatch")
        if hashlib.sha256(parsed.header_bytes).hexdigest() != _validate_hex_digest(
            artifact["header_sha256"], label="artifact header digest"
        ):
            raise ArtifactValidationError("artifact header SHA-256 mismatch")
        if info.st_size - parsed.data_start != artifact["payload_bytes"]:
            raise ArtifactValidationError("artifact payload byte count mismatch")
        if set(parsed.tensors) != set(tensor_rows):
            raise ArtifactValidationError("artifact header/manifest inventory mismatch")
        for name, metadata in parsed.tensors.items():
            row = tensor_rows[name]
            if metadata.dtype != row["dtype"] or list(metadata.shape) != row["shape"]:
                raise ArtifactValidationError(
                    f"artifact tensor metadata mismatch for {name}"
                )
            if list(metadata.offsets) != row["output_data_offsets"]:
                raise ArtifactValidationError(
                    f"artifact tensor offsets mismatch for {name}"
                )
        actual_file_digest = _verify_artifact_digests(fd, parsed, tensor_rows)
        if actual_file_digest != _validate_hex_digest(
            artifact["sha256"], label="artifact digest"
        ):
            raise ArtifactValidationError("artifact SHA-256 mismatch")

    if deep:
        source_root = root.parent / SOURCE_DIRECTORY
        source_identities, planned, distribution = _inspect_source(source_root)
        if distribution != EXPECTED_SHARD_COUNTS:
            raise ArtifactValidationError("deep source shard distribution mismatch")
        deep_source_rows = [
            {"name": item.name, "bytes": item.size, "sha256": item.sha256}
            for item in source_identities
        ]
        if deep_source_rows != source_files:
            raise ArtifactValidationError("deep source identity mismatch")
        if len(planned) != len(tensor_rows):
            raise ArtifactValidationError("deep source tensor count mismatch")
        identities = {item.name: item for item in source_identities}
        with contextlib.ExitStack() as stack:
            held: dict[str, int] = {}
            for shard_name in EXPECTED_SHARD_COUNTS:
                fd, _info = stack.enter_context(
                    _open_regular_nofollow(
                        source_root / shard_name,
                        label=f"deep source file {shard_name}",
                    )
                )
                _assert_held_source_identity(source_root, identities[shard_name], fd)
                held[shard_name] = fd
            for tensor in planned:
                row = tensor_rows.get(tensor.name)
                if row is None:
                    raise ArtifactValidationError(
                        f"deep source tensor missing: {tensor.name}"
                    )
                if (
                    row["dtype"] != tensor.dtype
                    or row["shape"] != list(tensor.shape)
                    or row["source_file"] != tensor.source_file
                    or row["source_data_offsets"] != list(tensor.source_data_offsets)
                ):
                    raise ArtifactValidationError(
                        f"deep source provenance mismatch for tensor {tensor.name}"
                    )
                source_digest = _sha256_fd(
                    held[tensor.source_file],
                    tensor.nbytes,
                    offset=tensor.source_absolute_offset,
                    label=f"deep source tensor {tensor.name}",
                )
                if source_digest != row["sha256"]:
                    raise ArtifactValidationError(
                        f"deep source tensor SHA-256 mismatch for {tensor.name}"
                    )
            for shard_name, fd in held.items():
                _assert_held_source_identity(source_root, identities[shard_name], fd)
    return manifest


def _prepare_artifact_verification(root: Path) -> tuple[Path, dict[str, object]]:
    root = Path(root)
    _validate_directory(root, label="artifact root")
    entries = sorted(path.name for path in root.iterdir())
    if entries != sorted([ARTIFACT_FILE, MANIFEST_FILE]):
        raise ArtifactValidationError(
            f"artifact root must contain exactly {ARTIFACT_FILE!r} and {MANIFEST_FILE!r}"
        )
    manifest = _read_manifest(root)
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ArtifactValidationError("manifest schema mismatch")
    return root, manifest


@contextlib.contextmanager
def open_verified_glm52_mtp_layer78(
    root: Path,
    *,
    deep: bool = True,
) -> Iterator[VerifiedGlm52MtpArtifact]:
    """Authenticate and hold the exact artifact inode for a runtime consumer.

    The yielded binary file object owns no independent pathname lookup. Consumers
    must pass it directly to their tensor loader and materialize all arrays before
    leaving the context.
    """

    if deep is not True:
        raise ArtifactValidationError(
            "deep=False is integrity-only and unauthenticated; "
            "runtime artifact verification requires deep=True"
        )
    root, manifest = _prepare_artifact_verification(root)
    artifact_path = root / ARTIFACT_FILE
    with _open_regular_nofollow(artifact_path, label="artifact file") as (
        artifact_fd,
        artifact_info,
    ):
        with os.fdopen(artifact_fd, "rb", closefd=False) as artifact_file:
            verified = _verify_glm52_mtp_layer78_held(
                root,
                manifest,
                artifact_fd,
                artifact_info,
                deep=deep,
            )
            artifact_file.seek(0)
            try:
                yield VerifiedGlm52MtpArtifact(root, verified, artifact_file)
            finally:
                final_info = os.fstat(artifact_fd)
                if not _same_open_file_contents(artifact_info, final_info):
                    expected_digest = _validate_hex_digest(
                        verified["artifact"]["sha256"],
                        label="artifact digest",
                    )
                    if (
                        final_info.st_size != artifact_info.st_size
                        or _sha256_fd(
                            artifact_fd,
                            final_info.st_size,
                            label="artifact file after use",
                        )
                        != expected_digest
                    ):
                        raise ArtifactValidationError(
                            "artifact file changed while in use"
                        )


def verify_glm52_mtp_layer78(root: Path, *, deep: bool = True) -> dict[str, object]:
    """Verify a published artifact and its pinned sibling source."""

    with open_verified_glm52_mtp_layer78(root, deep=deep) as verified:
        return verified.manifest


__all__ = [
    "ARTIFACT_FILE",
    "MANIFEST_FILE",
    "MANIFEST_KEYS",
    "MANIFEST_SCHEMA",
    "ArtifactError",
    "ArtifactPublicationError",
    "ArtifactValidationError",
    "ExtractionPlan",
    "Glm52MtpArtifactConfig",
    "SourceFilePin",
    "TensorExpectation",
    "VerifiedGlm52MtpArtifact",
    "expected_glm52_layer78_inventory",
    "extract_glm52_mtp_layer78",
    "open_verified_glm52_mtp_layer78",
    "preflight_glm52_mtp_layer78",
    "verify_glm52_mtp_layer78",
]
