"""Bounded conversion primitives for the explicit Hy3 expert-Q2 lane."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

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


def _require_real_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a real directory, not a symlink")
    return path.resolve()


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
) -> str:
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
        return target_sha256
    except BaseException:
        if target_fd is not None:
            os.close(target_fd)
            target_fd = None
            try:
                os.unlink(target_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
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
    work_fd = os.open(work_root, _directory_flags())
    try:
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

        copied_files: dict[str, str] = {}
        created: list[str] = []
        try:
            for name, source, expected_sha256 in source_members:
                copied_files[name] = _copy_independent_file(
                    source,
                    work_fd,
                    name,
                    chunk_bytes=copy_chunk_bytes,
                    expected_sha256=expected_sha256,
                )
                created.append(name)
                os.fsync(work_fd)
            target_shards, target_tensors = (
                expert_manifest_module._checkpoint_inventory(
                    work_root,
                    hash_shards=True,
                )
            )
            if target_shards != source_shards:
                raise ValueError(
                    "copied resident shard inventory changed during staging"
                )
            for name, tensor in target_tensors.items():
                source_tensor = source_tensors.get(name)
                if source_tensor is None or tensor != source_tensor:
                    raise ValueError(
                        f"copied resident tensor metadata changed during staging: {name}"
                    )
        except BaseException:
            for name in reversed(created):
                try:
                    os.unlink(name, dir_fd=work_fd)
                except FileNotFoundError:
                    pass
            os.fsync(work_fd)
            raise
        os.fsync(work_fd)
        return ResidentReuse(
            shards=tuple(selected_shards),
            tensors=source_manifest.resident_tensors,
            copied_files=copied_files,
        )
    finally:
        os.close(work_fd)


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
