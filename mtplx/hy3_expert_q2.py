"""Bounded conversion primitives for the explicit Hy3 expert-Q2 lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from . import expert_manifest as expert_manifest_module
from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
    resolve_artifact_member,
)


SOURCE_MODEL_KEY = "hy3-expert-only-q4"
TARGET_MODEL_KEY = "hy3-expert-q2"
SOURCE_MANIFEST_SHA256 = (
    "507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43"
)

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "scales", "biases")
_DTYPES = ("U32", "BF16", "BF16")
_GROUP_SIZE = 64
_INDEX_FILE = "model.safetensors.index.json"
_ANCILLARY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_MAX_COPY_CHUNK_BYTES = 64 * 1024**2
_MAX_INDEX_BYTES = 128 * 1024**2
_MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024**2
_MAX_RESIDENT_TENSORS = 1_000_000
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_ROUTED_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.\d+\.mlp\.(?:switch_mlp|experts)(?:\.|$)"
)
_MTP_TENSOR_RE = re.compile(
    r"(?:^|\.)layers\.80(?:\.|$)|(?:^|[._/])mtp(?:[._/]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    component: str
    cosine_q4_q2: float
    normalized_error_q4_q2: float
    finite: bool


@dataclass(frozen=True)
class ResidentReuse:
    shards: tuple[ShardInfo, ...]
    tensors: tuple[ResidentTensor, ...]
    copied_files: dict[str, str]


@dataclass(frozen=True)
class _CopyReceipt:
    sha256: str
    size: int
    source_device: int
    source_inode: int
    target_device: int
    target_inode: int


def _read_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _write_flags() -> int:
    return (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _pread_chunk(fd: int, length: int, offset: int) -> bytes:
    while True:
        try:
            return os.pread(fd, length, offset)
        except InterruptedError:
            continue


def _pread_exact(fd: int, offset: int, length: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    consumed = 0
    while consumed < length:
        chunk = _pread_chunk(fd, length - consumed, offset + consumed)
        if not chunk:
            raise ValueError(
                f"short read for {label}: got {consumed} bytes; expected {length}"
            )
        chunks.append(chunk)
        consumed += len(chunk)
    return b"".join(chunks)


def _pwrite_all(fd: int, payload: bytes, offset: int) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        try:
            count = os.pwrite(fd, view[written:], offset + written)
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError("positional write made no progress")
        written += count


def _hash_fd(fd: int, *, length: int, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    consumed = 0
    while consumed < length:
        chunk = _pread_chunk(fd, min(chunk_bytes, length - consumed), consumed)
        if not chunk:
            raise ValueError(
                f"short read while hashing: got {consumed} bytes; expected {length}"
            )
        digest.update(chunk)
        consumed += len(chunk)
    return digest.hexdigest()


def _absolute_without_symlink_ancestors(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ValueError(f"{label} ancestor is unavailable: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} has a symlinked ancestor: {current}")
    return absolute


def _require_real_directory(path: Path, *, label: str) -> Path:
    path = _absolute_without_symlink_ancestors(path, label=label)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a real directory, not a symlink")
    return path


def _assert_directory_path_identity(path: Path, fd: int, *, label: str) -> None:
    try:
        path_status = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} path identity is unavailable: {exc}") from exc
    descriptor_status = os.fstat(fd)
    if (
        stat.S_ISLNK(path_status.st_mode)
        or not stat.S_ISDIR(path_status.st_mode)
        or (path_status.st_dev, path_status.st_ino)
        != (descriptor_status.st_dev, descriptor_status.st_ino)
    ):
        raise ValueError(f"{label} path identity changed during resident staging")


def _require_flat_target_name(name: str) -> str:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or pure.is_absolute()
        or len(pure.parts) != 1
        or pure.parts[0] in {".", ".."}
    ):
        raise ValueError(f"unsafe target artifact path {name!r}")
    return name


def _copy_independent_file(
    source: Path,
    directory_fd: int,
    target_name: str,
    *,
    chunk_bytes: int,
    expected_sha256: str | None,
) -> _CopyReceipt:
    source_fd = os.open(source, _read_flags())
    target_fd: int | None = None
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError(f"source artifact is not a regular file: {source}")
        target_fd = os.open(target_name, _write_flags(), 0o644, dir_fd=directory_fd)
        copied_digest = hashlib.sha256()
        copied = 0
        while copied < source_before.st_size:
            chunk = _pread_chunk(
                source_fd,
                min(chunk_bytes, source_before.st_size - copied),
                copied,
            )
            if not chunk:
                raise ValueError(
                    f"short source read for {target_name}: got {copied} bytes; "
                    f"expected {source_before.st_size}"
                )
            _pwrite_all(target_fd, chunk, copied)
            copied_digest.update(chunk)
            copied += len(chunk)
        os.fsync(target_fd)
        source_after = os.fstat(source_fd)
        target_status = os.fstat(target_fd)
        if (
            source_after.st_size != source_before.st_size
            or target_status.st_size != source_before.st_size
        ):
            raise ValueError(
                f"source or target size changed while copying {target_name}"
            )
        if (target_status.st_dev, target_status.st_ino) == (
            source_after.st_dev,
            source_after.st_ino,
        ) or target_status.st_nlink != 1:
            raise ValueError(f"target {target_name} is not an independent inode")
        copied_sha256 = copied_digest.hexdigest()
        source_sha256 = _hash_fd(
            source_fd,
            length=source_after.st_size,
            chunk_bytes=chunk_bytes,
        )
        target_sha256 = _hash_fd(
            target_fd,
            length=target_status.st_size,
            chunk_bytes=chunk_bytes,
        )
        if len({copied_sha256, source_sha256, target_sha256}) != 1:
            raise ValueError(f"independent copy hash mismatch for {target_name}")
        if expected_sha256 is not None and target_sha256 != expected_sha256:
            raise ValueError(
                f"resident shard hash does not match manifest provenance: {target_name}"
            )
        return _CopyReceipt(
            sha256=target_sha256,
            size=target_status.st_size,
            source_device=source_after.st_dev,
            source_inode=source_after.st_ino,
            target_device=target_status.st_dev,
            target_inode=target_status.st_ino,
        )
    except BaseException:
        if target_fd is not None:
            try:
                descriptor_status = os.fstat(target_fd)
                entry_status = os.stat(
                    target_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISREG(entry_status.st_mode) and (
                    entry_status.st_dev,
                    entry_status.st_ino,
                ) == (descriptor_status.st_dev, descriptor_status.st_ino):
                    os.unlink(target_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            finally:
                os.close(target_fd)
                target_fd = None
        raise
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def _reject_resident_contamination(
    tensor_names: set[str],
    shard_names: set[str],
) -> None:
    for name in sorted(tensor_names):
        if _ROUTED_EXPERT_RE.search(name):
            raise ValueError(f"resident index contains routed expert tensor {name!r}")
        if _MTP_TENSOR_RE.search(name):
            raise ValueError(f"resident index contains MTP tensor {name!r}")
    for name in sorted(shard_names):
        if any(part.lower() == "mtp" for part in PurePosixPath(name).parts):
            raise ValueError(f"resident index contains MTP shard {name!r}")


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_from_fd(fd: int, *, name: str, max_bytes: int) -> Any:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
        raise ValueError(f"final {name} is not a bounded regular JSON file")
    payload = _pread_exact(fd, 0, metadata.st_size, label=name)
    try:
        return json.loads(payload, object_pairs_hook=_strict_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid final JSON in {name}: {exc}") from exc


def _exact_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _parse_index_fd(fd: int) -> tuple[dict[str, str], int | None]:
    value = _json_from_fd(fd, name=_INDEX_FILE, max_bytes=_MAX_INDEX_BYTES)
    if not isinstance(value, dict) or "weight_map" not in value:
        raise ValueError("final resident index must contain a weight_map object")
    if set(value) - {"weight_map", "metadata"}:
        raise ValueError("final resident index contains unknown top-level keys")
    raw_map = value["weight_map"]
    if not isinstance(raw_map, dict) or len(raw_map) > _MAX_RESIDENT_TENSORS:
        raise ValueError("final resident index weight_map is invalid or too large")
    weight_map: dict[str, str] = {}
    for tensor, shard in raw_map.items():
        if not isinstance(tensor, str) or not tensor:
            raise ValueError("final resident index has an invalid tensor name")
        if not isinstance(shard, str):
            raise ValueError(f"final resident index shard for {tensor!r} is invalid")
        weight_map[tensor] = _require_flat_target_name(shard)
    total_size = None
    if "metadata" in value:
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("final resident index metadata must be an object")
        if "total_size" in metadata:
            total_size = _exact_integer(
                metadata["total_size"],
                label="final resident index total_size",
            )
    return weight_map, total_size


def _parse_safetensors_fd(
    fd: int,
    *,
    name: str,
    sha256: str,
) -> tuple[ShardInfo, dict[str, ResidentTensor]]:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"final resident shard {name} is not regular")
    length_raw = _pread_exact(fd, 0, 8, label=f"{name} header length")
    header_length = int.from_bytes(length_raw, "little")
    if not 1 <= header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"final resident shard {name} has an invalid header length")
    header_raw = _pread_exact(fd, 8, header_length, label=f"{name} header")
    try:
        header = json.loads(header_raw, object_pairs_hook=_strict_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid final safetensors header in {name}: {exc}") from exc
    if not isinstance(header, dict) or len(header) > _MAX_RESIDENT_TENSORS:
        raise ValueError(f"final resident shard {name} has an invalid tensor header")
    data_start = 8 + header_length
    tensors: dict[str, ResidentTensor] = {}
    ranges: list[tuple[int, int, str]] = []
    for tensor_name, raw_info in header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"final resident shard {name} has an invalid tensor name")
        if not isinstance(raw_info, dict) or set(raw_info) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise ValueError(f"final tensor {tensor_name} has invalid metadata keys")
        dtype = raw_info["dtype"]
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"final tensor {tensor_name} has unsupported dtype")
        raw_shape = raw_info["shape"]
        if not isinstance(raw_shape, list):
            raise ValueError(f"final tensor {tensor_name} has an invalid shape")
        shape = tuple(
            _exact_integer(item, label=f"final tensor {tensor_name} shape", minimum=1)
            for item in raw_shape
        )
        offsets = raw_info["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"final tensor {tensor_name} has invalid offsets")
        start = _exact_integer(offsets[0], label=f"final tensor {tensor_name} start")
        end = _exact_integer(offsets[1], label=f"final tensor {tensor_name} end")
        if end <= start:
            raise ValueError(f"final tensor {tensor_name} has an empty range")
        length = end - start
        expected_length = _SAFETENSORS_DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_length *= dimension
        if length != expected_length:
            raise ValueError(f"final tensor {tensor_name} dtype/shape bytes mismatch")
        absolute_start = data_start + start
        absolute_end = data_start + end
        if absolute_end > metadata.st_size:
            raise ValueError(f"final tensor {tensor_name} exceeds shard {name}")
        tensors[tensor_name] = ResidentTensor(
            tensor=tensor_name,
            shard=name,
            offset=absolute_start,
            length=length,
            dtype=dtype,
            shape=shape,
        )
        ranges.append((absolute_start, absolute_end, tensor_name))
    if not tensors:
        raise ValueError(f"final resident shard {name} contains no tensors")
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ValueError(
                f"final resident shard {name} has overlapping tensor ranges"
            )
    if max(end for _start, end, _tensor in ranges) != metadata.st_size:
        raise ValueError(f"final resident shard {name} has trailing payload bytes")
    return (
        ShardInfo(
            name=name,
            size=metadata.st_size,
            header_bytes=data_start,
            header_sha256=hashlib.sha256(length_raw + header_raw).hexdigest(),
            sha256=sha256,
        ),
        tensors,
    )


def _open_final_files(
    directory_fd: int,
    receipts: dict[str, _CopyReceipt],
    *,
    chunk_bytes: int,
) -> dict[str, int]:
    expected_names = set(receipts)
    actual_names = set(os.listdir(directory_fd))
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "final resident directory inventory mismatch; "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    opened: dict[str, int] = {}
    try:
        for name in sorted(expected_names):
            receipt = receipts[name]
            entry_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(entry_status.st_mode) or entry_status.st_nlink != 1:
                raise ValueError(
                    f"final resident entry {name} is not a single-link regular file"
                )
            if (
                entry_status.st_size != receipt.size
                or (entry_status.st_dev, entry_status.st_ino)
                != (receipt.target_device, receipt.target_inode)
                or (entry_status.st_dev, entry_status.st_ino)
                == (receipt.source_device, receipt.source_inode)
            ):
                raise ValueError(
                    f"final resident entry {name} identity or size changed"
                )
            fd = os.open(name, _read_flags(), dir_fd=directory_fd)
            opened[name] = fd
            descriptor_status = os.fstat(fd)
            if (descriptor_status.st_dev, descriptor_status.st_ino) != (
                entry_status.st_dev,
                entry_status.st_ino,
            ):
                raise ValueError(f"final resident entry {name} swapped while opening")
            if (
                _hash_fd(fd, length=descriptor_status.st_size, chunk_bytes=chunk_bytes)
                != receipt.sha256
            ):
                raise ValueError(f"final resident entry {name} hash changed")
            final_entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(final_entry.st_mode) or (
                final_entry.st_dev,
                final_entry.st_ino,
            ) != (descriptor_status.st_dev, descriptor_status.st_ino):
                raise ValueError(
                    f"final resident entry {name} swapped during validation"
                )
        return opened
    except BaseException:
        for fd in opened.values():
            os.close(fd)
        raise


def _validate_final_index_and_shards(
    final_fds: dict[str, int],
    receipts: dict[str, _CopyReceipt],
    expected_shards: tuple[ShardInfo, ...],
    expected_tensors: tuple[ResidentTensor, ...],
) -> None:
    weight_map, declared_total_size = _parse_index_fd(final_fds[_INDEX_FILE])
    tensors_by_name = {tensor.tensor: tensor for tensor in expected_tensors}
    if set(weight_map) != set(tensors_by_name):
        raise ValueError("final resident index tensor set changed during staging")
    expected_shards_by_name = {shard.name: shard for shard in expected_shards}
    if set(weight_map.values()) != set(expected_shards_by_name):
        raise ValueError("final resident index shard set changed during staging")
    parsed_tensors: dict[str, ResidentTensor] = {}
    for name in sorted(expected_shards_by_name):
        parsed_shard, shard_tensors = _parse_safetensors_fd(
            final_fds[name],
            name=name,
            sha256=receipts[name].sha256,
        )
        if parsed_shard != expected_shards_by_name[name]:
            raise ValueError(f"final resident shard metadata changed: {name}")
        for tensor_name, tensor in shard_tensors.items():
            if tensor_name in parsed_tensors:
                raise ValueError(f"final resident tensor is duplicated: {tensor_name}")
            if weight_map.get(tensor_name) != name:
                raise ValueError(f"final resident index maps {tensor_name} incorrectly")
            parsed_tensors[tensor_name] = tensor
    if parsed_tensors != tensors_by_name:
        raise ValueError("final resident tensor metadata changed during staging")
    expected_total = sum(tensor.length for tensor in expected_tensors)
    if declared_total_size is not None and declared_total_size != expected_total:
        raise ValueError("final resident index total_size changed during staging")


def _final_recheck(
    directory_fd: int,
    final_fds: dict[str, int],
    receipts: dict[str, _CopyReceipt],
    *,
    chunk_bytes: int,
) -> None:
    if set(os.listdir(directory_fd)) != set(receipts):
        raise ValueError("final resident directory inventory changed before return")
    for name, fd in final_fds.items():
        receipt = receipts[name]
        descriptor_status = os.fstat(fd)
        entry_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry_status.st_mode)
            or entry_status.st_nlink != 1
            or (entry_status.st_dev, entry_status.st_ino)
            != (descriptor_status.st_dev, descriptor_status.st_ino)
            or (descriptor_status.st_dev, descriptor_status.st_ino)
            != (receipt.target_device, receipt.target_inode)
        ):
            raise ValueError(
                f"final resident entry {name} identity changed before return"
            )
        if (
            _hash_fd(fd, length=descriptor_status.st_size, chunk_bytes=chunk_bytes)
            != receipt.sha256
        ):
            raise ValueError(f"final resident entry {name} hash changed before return")
    if set(os.listdir(directory_fd)) != set(receipts):
        raise ValueError("final resident directory inventory changed during validation")
    for name, fd in final_fds.items():
        descriptor_status = os.fstat(fd)
        entry_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry_status.st_mode) or (
            entry_status.st_dev,
            entry_status.st_ino,
        ) != (descriptor_status.st_dev, descriptor_status.st_ino):
            raise ValueError(f"final resident entry {name} swapped before return")


def _cleanup_created_entries(
    directory_fd: int,
    receipts: dict[str, _CopyReceipt],
) -> None:
    for name, receipt in reversed(tuple(receipts.items())):
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(metadata.st_mode) and (
            metadata.st_dev,
            metadata.st_ino,
        ) == (receipt.target_device, receipt.target_inode):
            os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def stage_exact_residents(
    source_root: Path,
    source_manifest: ExpertManifest,
    work_root: Path,
    *,
    copy_chunk_bytes: int = 8 * 1024**2,
) -> ResidentReuse:
    """Copy the exact complete resident checkpoint without reserialization."""

    if (
        isinstance(copy_chunk_bytes, bool)
        or not isinstance(copy_chunk_bytes, int)
        or not 1 <= copy_chunk_bytes <= _MAX_COPY_CHUNK_BYTES
    ):
        raise ValueError(
            f"copy_chunk_bytes must be inside [1, {_MAX_COPY_CHUNK_BYTES}]"
        )
    source_root = _require_real_directory(Path(source_root), label="source root")
    work_root = _require_real_directory(Path(work_root), label="target work root")
    if (
        source_root == work_root
        or source_root in work_root.parents
        or work_root in source_root.parents
    ):
        raise ValueError("source root and target work root must not contain each other")
    source_fd = os.open(source_root, _directory_flags())
    try:
        work_fd = os.open(work_root, _directory_flags())
    except BaseException:
        os.close(source_fd)
        raise
    try:
        _assert_directory_path_identity(source_root, source_fd, label="source root")
        _assert_directory_path_identity(work_root, work_fd, label="target work root")
        if os.listdir(work_fd):
            raise ValueError("target work root must be empty before resident staging")

        source_manifest.validate_structure()
        if (
            source_manifest.manifest_sha256 is None
            or source_manifest.manifest_sha256
            != source_manifest.with_digest().manifest_sha256
        ):
            raise ValueError("source manifest digest is missing or invalid")
        try:
            index_source = resolve_artifact_member(source_root, _INDEX_FILE)
        except ExpertManifestError as exc:
            raise ValueError(f"required resident index is unavailable: {exc}") from exc
        source_shards, source_tensors = expert_manifest_module._checkpoint_inventory(
            source_root,
            hash_shards=True,
        )
        index_names = set(source_tensors)
        manifest_tensors = {
            tensor.tensor: tensor for tensor in source_manifest.resident_tensors
        }
        _reject_resident_contamination(
            index_names | set(manifest_tensors),
            {shard.name for shard in source_shards},
        )
        if index_names != set(manifest_tensors):
            missing = sorted(set(manifest_tensors) - index_names)
            extra = sorted(index_names - set(manifest_tensors))
            raise ValueError(
                "resident index tensor set does not equal the manifest allowlist; "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        for name, tensor in source_tensors.items():
            expected = manifest_tensors[name]
            if (
                tensor.shard != expected.shard
                or tensor.offset != expected.offset
                or tensor.length != expected.length
                or tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
            ):
                raise ValueError(
                    f"resident metadata does not match index headers: {name}"
                )

        manifest_shards = {
            shard.name: shard
            for shard in source_manifest.shards
            if shard.kind == "safetensors"
        }
        selected_shards = []
        source_members: list[tuple[str, Path, str | None]] = []
        for shard in source_shards:
            _require_flat_target_name(shard.name)
            expected = manifest_shards.get(shard.name)
            if expected is None or expected.sha256 is None:
                raise ValueError(
                    f"resident shard lacks manifest provenance: {shard.name}"
                )
            if (
                shard.size != expected.size
                or shard.header_bytes != expected.header_bytes
                or shard.header_sha256 != expected.header_sha256
                or shard.sha256 != expected.sha256
            ):
                raise ValueError(
                    f"resident shard header, size, or hash provenance mismatch: {shard.name}"
                )
            selected_shards.append(expected)
            source_members.append(
                (
                    shard.name,
                    resolve_artifact_member(source_root, shard.name),
                    expected.sha256,
                )
            )

        source_members.append((_INDEX_FILE, index_source, None))
        for name in _ANCILLARY_FILES:
            _require_flat_target_name(name)
            try:
                source = resolve_artifact_member(source_root, name)
            except ExpertManifestError as exc:
                raise ValueError(
                    f"required ancillary {name} is unavailable: {exc}"
                ) from exc
            source_members.append((name, source, None))
        target_names = [name for name, _source, _digest in source_members]
        if len(target_names) != len(set(target_names)):
            raise ValueError("resident staging target names are not unique")

        receipts: dict[str, _CopyReceipt] = {}
        final_fds: dict[str, int] = {}
        try:
            for name, source, expected_sha256 in source_members:
                receipts[name] = _copy_independent_file(
                    source,
                    work_fd,
                    name,
                    chunk_bytes=copy_chunk_bytes,
                    expected_sha256=expected_sha256,
                )
                os.fsync(work_fd)
            final_fds = _open_final_files(
                work_fd,
                receipts,
                chunk_bytes=copy_chunk_bytes,
            )
            _validate_final_index_and_shards(
                final_fds,
                receipts,
                tuple(selected_shards),
                source_manifest.resident_tensors,
            )
            result = ResidentReuse(
                shards=tuple(selected_shards),
                tensors=source_manifest.resident_tensors,
                copied_files={
                    name: receipt.sha256 for name, receipt in receipts.items()
                },
            )
            _assert_directory_path_identity(
                source_root,
                source_fd,
                label="source root",
            )
            _assert_directory_path_identity(
                work_root,
                work_fd,
                label="target work root",
            )
            os.fsync(work_fd)
            _final_recheck(
                work_fd,
                final_fds,
                receipts,
                chunk_bytes=copy_chunk_bytes,
            )
            _assert_directory_path_identity(
                source_root,
                source_fd,
                label="source root",
            )
            _assert_directory_path_identity(
                work_root,
                work_fd,
                label="target work root",
            )
            return result
        except BaseException:
            for fd in final_fds.values():
                os.close(fd)
            final_fds.clear()
            _cleanup_created_entries(work_fd, receipts)
            raise
        finally:
            for fd in final_fds.values():
                os.close(fd)
    finally:
        os.close(work_fd)
        os.close(source_fd)


def _byte_view(
    payload: bytes | memoryview,
    *,
    component: str,
    expected_bytes: int,
) -> memoryview:
    try:
        view = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{component} must be a contiguous byte buffer") from exc
    if view.nbytes != expected_bytes:
        qualifier = "short read" if view.nbytes < expected_bytes else "oversized read"
        raise ValueError(
            f"{qualifier} for {component}: got {view.nbytes} bytes; "
            f"expected {expected_bytes}"
        )
    return view


def _projection_shapes(
    *,
    input_size: int,
    output_size: int,
    bits: int,
    group_size: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if (
        isinstance(input_size, bool)
        or not isinstance(input_size, int)
        or input_size <= 0
    ):
        raise ValueError("input_size must be a positive integer")
    if (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size <= 0
    ):
        raise ValueError("output_size must be a positive integer")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size != _GROUP_SIZE
    ):
        raise ValueError(f"group_size must be {_GROUP_SIZE}")
    if input_size % group_size:
        raise ValueError("input_size must be divisible by group_size")
    if input_size * bits % 32:
        raise ValueError("input_size is not representable by packed uint32 weights")
    return (
        (output_size, input_size * bits // 32),
        (output_size, input_size // group_size),
        (output_size, input_size // group_size),
    )


def _shape_bytes(shape: tuple[int, int], dtype: str) -> int:
    item_size = 4 if dtype == "U32" else 2
    return shape[0] * shape[1] * item_size


def requantize_projection_q4_to_q2(
    weight_bytes: bytes | memoryview,
    scales_bytes: bytes | memoryview,
    biases_bytes: bytes | memoryview,
    *,
    projection: str,
    input_size: int,
    output_size: int,
    group_size: int = 64,
) -> tuple[tuple[bytes, bytes, bytes], ProjectionDiagnostics]:
    """Convert one affine-Q4 projection to canonical affine-Q2 bytes.

    MLX is imported only inside this numerical boundary. All MLX arrays are
    released and its cache is cleared before returning so records can be
    processed one projection at a time without retaining dense expert weights.
    """

    if projection not in _PROJECTIONS:
        raise ValueError(f"unsupported expert projection {projection!r}")
    source_shapes = _projection_shapes(
        input_size=input_size,
        output_size=output_size,
        bits=4,
        group_size=group_size,
    )
    target_shapes = _projection_shapes(
        input_size=input_size,
        output_size=output_size,
        bits=2,
        group_size=group_size,
    )
    source_views = tuple(
        _byte_view(
            payload,
            component=f"{projection}.{leaf}",
            expected_bytes=_shape_bytes(shape, dtype),
        )
        for payload, leaf, shape, dtype in zip(
            (weight_bytes, scales_bytes, biases_bytes),
            _LEAVES,
            source_shapes,
            _DTYPES,
            strict=True,
        )
    )

    import mlx.core as mx

    q4_weight = None
    q4_scales = None
    q4_biases = None
    dense = None
    q2_weight = None
    q2_scales = None
    q2_biases = None
    q2_dense = None
    source_fp32 = None
    target_fp32 = None
    try:
        q4_weight = mx.array(
            np.frombuffer(source_views[0], dtype="<u4")
            .copy()
            .reshape(source_shapes[0]),
            dtype=mx.uint32,
        )

        def decode_bf16(view: memoryview, shape: tuple[int, int]):
            words = mx.array(
                np.frombuffer(view, dtype="<u2").copy().reshape(shape),
                dtype=mx.uint16,
            )
            return words.view(mx.bfloat16)

        q4_scales = decode_bf16(source_views[1], source_shapes[1])
        q4_biases = decode_bf16(source_views[2], source_shapes[2])
        mx.eval(q4_weight, q4_scales, q4_biases)
        if not bool(
            mx.all(mx.isfinite(q4_scales)).item()
            and mx.all(mx.isfinite(q4_biases)).item()
        ):
            raise ValueError(f"projection {projection} has non-finite Q4 values")

        dense = mx.dequantize(
            q4_weight,
            q4_scales,
            q4_biases,
            bits=4,
            group_size=group_size,
            mode="affine",
        )
        mx.eval(dense)
        if not bool(mx.all(mx.isfinite(dense)).item()):
            raise ValueError(f"projection {projection} has non-finite Q4 values")

        q2_weight, q2_scales, q2_biases = mx.quantize(
            dense,
            bits=2,
            group_size=group_size,
            mode="affine",
        )
        mx.eval(q2_weight, q2_scales, q2_biases)
        if q2_weight.dtype != mx.uint32 or tuple(q2_weight.shape) != target_shapes[0]:
            raise ValueError(
                f"projection {projection} produced invalid Q2 weight metadata"
            )
        for label, value, shape in (
            ("scales", q2_scales, target_shapes[1]),
            ("biases", q2_biases, target_shapes[2]),
        ):
            if value.dtype != mx.bfloat16 or tuple(value.shape) != shape:
                raise ValueError(
                    f"projection {projection} produced invalid Q2 {label} metadata"
                )

        q2_dense = mx.dequantize(
            q2_weight,
            q2_scales,
            q2_biases,
            bits=2,
            group_size=group_size,
            mode="affine",
        )
        mx.eval(q2_dense)
        finite = bool(
            mx.all(mx.isfinite(q2_scales)).item()
            and mx.all(mx.isfinite(q2_biases)).item()
            and mx.all(mx.isfinite(q2_dense)).item()
        )
        if not finite:
            raise ValueError(f"projection {projection} produced non-finite Q2 values")

        source_fp32 = dense.astype(mx.float32).reshape(-1)
        target_fp32 = q2_dense.astype(mx.float32).reshape(-1)
        source_norm = float(mx.linalg.norm(source_fp32).item())
        target_norm = float(mx.linalg.norm(target_fp32).item())
        dot = float(mx.sum(source_fp32 * target_fp32).item())
        error_norm = float(mx.linalg.norm(source_fp32 - target_fp32).item())
        if source_norm == 0.0:
            cosine = 1.0 if target_norm == 0.0 else 0.0
            normalized_error = error_norm
        else:
            cosine = dot / (source_norm * target_norm) if target_norm else 0.0
            normalized_error = error_norm / source_norm
        if not all(math.isfinite(value) for value in (cosine, normalized_error)):
            raise ValueError(f"projection {projection} produced non-finite diagnostics")

        output = (
            np.array(q2_weight, copy=True).astype("<u4", copy=False).tobytes(),
            np.array(q2_scales.view(mx.uint16), copy=True)
            .astype("<u2", copy=False)
            .tobytes(),
            np.array(q2_biases.view(mx.uint16), copy=True)
            .astype("<u2", copy=False)
            .tobytes(),
        )
        expected_output_bytes = tuple(
            _shape_bytes(shape, dtype)
            for shape, dtype in zip(target_shapes, _DTYPES, strict=True)
        )
        if tuple(len(item) for item in output) != expected_output_bytes:
            raise ValueError(
                f"projection {projection} Q2 serialization has the wrong byte counts"
            )
        diagnostics = ProjectionDiagnostics(
            component=projection,
            cosine_q4_q2=cosine,
            normalized_error_q4_q2=normalized_error,
            finite=True,
        )
        return output, diagnostics
    finally:
        q4_weight = q4_scales = q4_biases = None
        dense = q2_weight = q2_scales = q2_biases = q2_dense = None
        source_fp32 = target_fp32 = None
        value = None
        mx.clear_cache()


def _canonical_q4_metadata(
    *,
    hidden_size: int,
    expert_hidden_size: int,
    group_size: int,
) -> tuple[tuple[str, str, tuple[int, int], int], ...]:
    dimensions = {
        "gate_proj": (hidden_size, expert_hidden_size),
        "up_proj": (hidden_size, expert_hidden_size),
        "down_proj": (expert_hidden_size, hidden_size),
    }
    expected = []
    for projection in _PROJECTIONS:
        input_size, output_size = dimensions[projection]
        shapes = _projection_shapes(
            input_size=input_size,
            output_size=output_size,
            bits=4,
            group_size=group_size,
        )
        expected.extend(
            (
                f"{projection}.{leaf}",
                dtype,
                shape,
                _shape_bytes(shape, dtype),
            )
            for leaf, dtype, shape in zip(_LEAVES, _DTYPES, shapes, strict=True)
        )
    return tuple(expected)


def requantize_expert_record_q4_to_q2(
    record: ExpertRecord,
    read_component: Callable[[TensorSegment], bytes | memoryview],
    write_component: Callable[[str, bytes], None],
    *,
    hidden_size: int,
    expert_hidden_size: int,
    group_size: int = 64,
) -> tuple[ProjectionDiagnostics, ...]:
    """Validate and convert a record in gate/up/down projection order."""

    expected = _canonical_q4_metadata(
        hidden_size=hidden_size,
        expert_hidden_size=expert_hidden_size,
        group_size=group_size,
    )
    if len(record.segments) != len(expected):
        raise ValueError("expert record must contain nine canonical Q4 components")
    for segment, (component, dtype, shape, length) in zip(
        record.segments,
        expected,
        strict=True,
    ):
        if segment.component != component:
            raise ValueError(
                "expert record component order is not canonical: "
                f"expected {component}; found {segment.component}"
            )
        if segment.dtype != dtype:
            raise ValueError(
                f"expert record component {component} dtype {segment.dtype!r} "
                f"does not match {dtype!r}"
            )
        if tuple(segment.shape) != shape:
            raise ValueError(
                f"expert record component {component} shape {segment.shape!r} "
                f"does not match {shape!r}"
            )
        if segment.length != length:
            raise ValueError(
                f"expert record component {component} length {segment.length} "
                f"does not match {length}"
            )
    expected_record_bytes = sum(item[3] for item in expected)
    if record.logical_bytes != expected_record_bytes:
        raise ValueError(
            f"expert record logical_bytes {record.logical_bytes} does not match "
            f"canonical Q4 length {expected_record_bytes}"
        )

    dimensions = {
        "gate_proj": (hidden_size, expert_hidden_size),
        "up_proj": (hidden_size, expert_hidden_size),
        "down_proj": (expert_hidden_size, hidden_size),
    }
    source_payloads = tuple(
        bytes(
            _byte_view(
                read_component(segment),
                component=segment.component,
                expected_bytes=segment.length,
            )
        )
        for segment in record.segments
    )
    diagnostics = []
    staged_outputs: list[tuple[str, bytes]] = []
    for projection_index, projection in enumerate(_PROJECTIONS):
        projection_segments = record.segments[
            projection_index * 3 : projection_index * 3 + 3
        ]
        projection_payloads = source_payloads[
            projection_index * 3 : projection_index * 3 + 3
        ]
        input_size, output_size = dimensions[projection]
        converted, projection_diagnostics = requantize_projection_q4_to_q2(
            *projection_payloads,
            projection=projection,
            input_size=input_size,
            output_size=output_size,
            group_size=group_size,
        )
        for segment, payload in zip(projection_segments, converted, strict=True):
            staged_outputs.append((segment.component, payload))
        diagnostics.append(projection_diagnostics)
        del projection_payloads, converted, projection_diagnostics
    for component, payload in staged_outputs:
        write_component(component, payload)
    return tuple(diagnostics)
