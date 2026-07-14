"""Bounded conversion primitives for the explicit Hy3 expert-Q2 lane."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from .expert_manifest import (
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
    validate_expert_manifest_spec,
)
from .expert_streaming_models import ExpertStreamingModelSpec, get_model_spec


SOURCE_MODEL_KEY = "hy3-expert-only-q4"
TARGET_MODEL_KEY = "hy3-expert-q2"
SOURCE_MANIFEST_SHA256 = (
    "507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43"
)
_SOURCE_MANIFEST_FILE_SHA256 = (
    "e7fcfd6c69486456af4261d908d95f8a84a391d6a273ff1cff02a15f73fac92d"
)
_SOURCE_PROVENANCE_SHA256 = (
    "e832743f84f09f5a548a8734b2ab6d75043e32723223e90b1c05da074b42e7f2"
)
_SOURCE_INDEX_SHA256 = (
    "b901cc98a86131b519d69294d65a20023b5ac4d5706c96bd4bf128ef7e41ef5e"
)
_SOURCE_CONFIG_SHA256 = (
    "cf58dd3aaf61b1d59495622c209680abf718d2ee8fd952b56187e57f355923b7"
)
_SOURCE_SIDECAR_SHA256 = (
    "5ba698b9b2c51bca66254e5d8d35101325e37dfe40744294d4aa233c980472ae"
)
_CONVERSION_SCHEMA = "mtplx-hy3-expert-q2-conversion-v1"
_JOURNAL_SCHEMA = "mtplx-hy3-expert-q2-journal-v1"
_SOURCE_DIRECTORY_NAME = "hy3-expert-only-mlx-q4"
_TARGET_DIRECTORY_NAME = "hy3-expert-only-mlx-q2"
_WORK_DIRECTORY_NAME = ".hy3-expert-only-mlx-q2.incomplete"
_JOURNAL_FILE = "conversion-journal.jsonl"
_SIDECAR_FILE = "experts.bin"
_DEFAULT_ALIGNMENT = 16 * 1024
_SOURCE_RECORD_BYTES = 10_616_832
_TARGET_RECORD_BYTES = 5_898_240
_RECORD_COUNT = 15_168
_SOURCE_SIDECAR_BYTES = 161_036_107_776
_TARGET_SIDECAR_BYTES = 89_464_504_320
_RESIDENT_TENSOR_BYTES = 17_494_289_664
_TARGET_TENSOR_BYTES = 106_958_793_984
_RESIDENT_SHARD_COUNT = 18
_PROJECTION_WORKING_RESERVE = 64 * 1024**2
_MAX_MANIFEST_BYTES = 256 * 1024**2
_MAX_PROVENANCE_BYTES = 16 * 1024**2
_MAX_JOURNAL_BYTES = 1024 * 1024**2

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


@dataclass(frozen=True)
class _SourceArtifact:
    name: str
    fd: int
    size: int
    device: int
    inode: int
    entry_directory_fd: int
    entry_name: str
    entry_device: int
    entry_inode: int
    link_device: int | None = None
    link_inode: int | None = None
    link_target: str | None = None


@dataclass
class _HfBlobLayout:
    source_name: str
    snapshots_fd: int
    repository_fd: int
    blobs_fd: int | None = None


@dataclass
class _PinnedPreflightInputs:
    source_root: Path
    source_fd: int
    hf_blob_layout: _HfBlobLayout | None
    output_parent: Path
    output_parent_fd: int
    artifacts: dict[str, _SourceArtifact]
    closed: bool = False

    def open_artifact(self, name: str) -> _SourceArtifact:
        if self.closed:
            raise ValueError("pinned preflight inputs are closed")
        existing = self.artifacts.get(name)
        if existing is not None:
            return existing
        artifact, self.hf_blob_layout = _open_source_artifact(
            self.source_root,
            self.source_fd,
            name,
            self.hf_blob_layout,
        )
        self.artifacts[name] = artifact
        return artifact

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for artifact in self.artifacts.values():
            os.close(artifact.fd)
        self.artifacts.clear()
        os.close(self.source_fd)
        if self.hf_blob_layout is not None:
            _close_hf_blob_layout(self.hf_blob_layout)
        os.close(self.output_parent_fd)


@dataclass(frozen=True)
class ConversionConfig:
    source_root: Path
    source_manifest: Path
    source_provenance: Path
    output_root: Path
    alignment: int = _DEFAULT_ALIGNMENT
    pilot_report: Path | None = None

    def __post_init__(self) -> None:
        paths = {
            "source_root": self.source_root,
            "source_manifest": self.source_manifest,
            "source_provenance": self.source_provenance,
            "output_root": self.output_root,
        }
        if self.pilot_report is not None:
            paths["pilot_report"] = self.pilot_report
        for label, path in paths.items():
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{label} must be an absolute pathlib.Path")
        if self.source_root.name != _SOURCE_DIRECTORY_NAME:
            raise ValueError("source_root must name the pinned Hy3 Q4 artifact")
        if self.output_root.name != _TARGET_DIRECTORY_NAME:
            raise ValueError("output_root must name the explicit Hy3 Q2 artifact")
        if self.source_root.parent != self.output_root.parent:
            raise ValueError("source and output roots must be siblings")
        if self.source_manifest != self.source_root / "expert-manifest.json":
            raise ValueError("source_manifest must be the source root manifest")
        if self.source_provenance != self.source_root / "conversion-provenance.json":
            raise ValueError("source_provenance must be the source root provenance")
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("alignment must be a positive power of two")


@dataclass(frozen=True)
class _ConversionExpectations:
    source_root: Path
    manifest_file_sha256: str
    manifest_sha256: str
    provenance_sha256: str
    index_sha256: str
    config_sha256: str
    sidecar_sha256: str
    source_sidecar_bytes: int
    record_count: int
    source_record_bytes: int
    target_record_bytes: int
    target_sidecar_bytes: int
    resident_tensor_bytes: int
    target_tensor_bytes: int
    resident_shard_count: int
    alignment: int
    resident_source_repo: str
    resident_source_revision: str
    oracle_repo: str
    oracle_revision: str


@dataclass(frozen=True)
class _PreflightContext:
    manifest: ExpertManifest
    report: dict[str, Any]
    source_descriptor: ExpertStreamingModelSpec
    target_descriptor: ExpertStreamingModelSpec
    expectations: _ConversionExpectations
    pinned: _PinnedPreflightInputs


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


def _read_write_flags() -> int:
    return os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _conversion_expectations() -> _ConversionExpectations:
    return _ConversionExpectations(
        source_root=Path("/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q4"),
        manifest_file_sha256=_SOURCE_MANIFEST_FILE_SHA256,
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        provenance_sha256=_SOURCE_PROVENANCE_SHA256,
        index_sha256=_SOURCE_INDEX_SHA256,
        config_sha256=_SOURCE_CONFIG_SHA256,
        sidecar_sha256=_SOURCE_SIDECAR_SHA256,
        source_sidecar_bytes=_SOURCE_SIDECAR_BYTES,
        record_count=_RECORD_COUNT,
        source_record_bytes=_SOURCE_RECORD_BYTES,
        target_record_bytes=_TARGET_RECORD_BYTES,
        target_sidecar_bytes=_TARGET_SIDECAR_BYTES,
        resident_tensor_bytes=_RESIDENT_TENSOR_BYTES,
        target_tensor_bytes=_TARGET_TENSOR_BYTES,
        resident_shard_count=_RESIDENT_SHARD_COUNT,
        alignment=_DEFAULT_ALIGNMENT,
        resident_source_repo="tencent/Hy3",
        resident_source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
        oracle_repo="pipenetwork/Hy3-4bit",
        oracle_revision="160619d3f96c8470350b6dac0ef033a8381551e3",
    )


def _source_descriptor() -> ExpertStreamingModelSpec:
    return get_model_spec(SOURCE_MODEL_KEY)


def _target_descriptor() -> ExpertStreamingModelSpec:
    return get_model_spec(TARGET_MODEL_KEY)


def _producer_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"could not establish producer Git state: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("producer Git commit is not a full SHA-1")
    dirty = bool(run("status", "--porcelain", "--untracked-files=all"))
    return {"git_commit": commit, "dirty": dirty}


def _mlx_version() -> str:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("MLX package version is unavailable") from exc


def _target_descriptor_state(spec: ExpertStreamingModelSpec) -> dict[str, Any]:
    if not isinstance(spec, ExpertStreamingModelSpec):
        raise TypeError("target descriptor must be an ExpertStreamingModelSpec")
    state = {field.name: getattr(spec, field.name) for field in fields(spec)}
    return json.loads(_canonical_json_bytes(state))


def _minimum_conversion_manifest(
    expectations: _ConversionExpectations,
) -> dict[str, Any]:
    return {
        "schema": _CONVERSION_SCHEMA,
        "source": {
            "model_key": SOURCE_MODEL_KEY,
            "manifest_file_sha256": expectations.manifest_file_sha256,
            "manifest_sha256": expectations.manifest_sha256,
            "conversion_provenance_sha256": expectations.provenance_sha256,
            "sidecar_sha256": expectations.sidecar_sha256,
        },
        "derivation": {
            "kind": "q4_to_q2",
            "source_bits": 4,
            "target_bits": 2,
            "group_size": 64,
            "mode": "affine",
            "external_q2_artifact_used": False,
        },
        "target": {
            "model_key": TARGET_MODEL_KEY,
            "record_count": expectations.record_count,
            "record_bytes": expectations.target_record_bytes,
            "sidecar_bytes": expectations.target_sidecar_bytes,
            "resident_tensor_bytes": expectations.resident_tensor_bytes,
            "tensor_bytes": expectations.target_tensor_bytes,
            "mtp_included": False,
        },
    }


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


def _close_hf_blob_layout(layout: _HfBlobLayout) -> None:
    if layout.blobs_fd is not None:
        os.close(layout.blobs_fd)
    os.close(layout.snapshots_fd)
    os.close(layout.repository_fd)


def _open_source_directory(
    source_root: Path,
) -> tuple[int, _HfBlobLayout | None]:
    if source_root.parent.name != "snapshots":
        return os.open(source_root, _directory_flags()), None

    snapshot_name = _require_flat_target_name(source_root.name)
    repository_root = _require_real_directory(
        source_root.parent.parent,
        label="HF repository root",
    )
    repository_fd: int | None = os.open(repository_root, _directory_flags())
    snapshots_fd: int | None = None
    source_fd: int | None = None
    try:
        _assert_directory_path_identity(
            repository_root,
            repository_fd,
            label="HF repository root",
        )
        snapshots_before = os.stat(
            "snapshots",
            dir_fd=repository_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(snapshots_before.st_mode):
            raise ValueError("HF snapshots entry must be a real directory")
        snapshots_fd = os.open(
            "snapshots",
            _directory_flags(),
            dir_fd=repository_fd,
        )
        snapshots_status = os.fstat(snapshots_fd)
        snapshots_after = os.stat(
            "snapshots",
            dir_fd=repository_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(snapshots_status.st_mode)
            or not stat.S_ISDIR(snapshots_after.st_mode)
            or (snapshots_before.st_dev, snapshots_before.st_ino)
            != (snapshots_status.st_dev, snapshots_status.st_ino)
            or (snapshots_after.st_dev, snapshots_after.st_ino)
            != (snapshots_status.st_dev, snapshots_status.st_ino)
        ):
            raise ValueError("HF snapshots directory changed while opening")
        source_before = os.stat(
            snapshot_name,
            dir_fd=snapshots_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(source_before.st_mode):
            raise ValueError("HF snapshot revision must be a real directory")
        source_fd = os.open(
            snapshot_name,
            _directory_flags(),
            dir_fd=snapshots_fd,
        )
        source_status = os.fstat(source_fd)
        source_after = os.stat(
            snapshot_name,
            dir_fd=snapshots_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(source_status.st_mode)
            or not stat.S_ISDIR(source_after.st_mode)
            or (source_before.st_dev, source_before.st_ino)
            != (source_status.st_dev, source_status.st_ino)
            or (source_after.st_dev, source_after.st_ino)
            != (source_status.st_dev, source_status.st_ino)
        ):
            raise ValueError("HF snapshot revision changed while opening")
        _assert_directory_path_identity(source_root, source_fd, label="source root")
        _assert_directory_path_identity(
            repository_root,
            repository_fd,
            label="HF repository root",
        )
        result = _HfBlobLayout(
            source_name=snapshot_name,
            snapshots_fd=snapshots_fd,
            repository_fd=repository_fd,
        )
        snapshots_fd = None
        repository_fd = None
        opened_source_fd = source_fd
        source_fd = None
        return opened_source_fd, result
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if snapshots_fd is not None:
            os.close(snapshots_fd)
        if repository_fd is not None:
            os.close(repository_fd)


def _open_hf_blob_directory(layout: _HfBlobLayout) -> int:
    if layout.blobs_fd is not None:
        return layout.blobs_fd
    blob_before = os.stat(
        "blobs",
        dir_fd=layout.repository_fd,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(blob_before.st_mode):
        raise ValueError("HF blobs entry must be a real directory")
    blob_fd: int | None = os.open(
        "blobs",
        _directory_flags(),
        dir_fd=layout.repository_fd,
    )
    try:
        blob_status = os.fstat(blob_fd)
        blob_after = os.stat(
            "blobs",
            dir_fd=layout.repository_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(blob_status.st_mode)
            or not stat.S_ISDIR(blob_after.st_mode)
            or (blob_before.st_dev, blob_before.st_ino)
            != (blob_status.st_dev, blob_status.st_ino)
            or (blob_after.st_dev, blob_after.st_ino)
            != (blob_status.st_dev, blob_status.st_ino)
        ):
            raise ValueError("HF blobs directory changed while opening")
        layout.blobs_fd = blob_fd
        blob_fd = None
        return layout.blobs_fd
    finally:
        if blob_fd is not None:
            os.close(blob_fd)


def _assert_hf_layout_identity(source_fd: int, layout: _HfBlobLayout) -> None:
    try:
        source_entry = os.stat(
            layout.source_name,
            dir_fd=layout.snapshots_fd,
            follow_symlinks=False,
        )
        snapshots_entry = os.stat(
            "snapshots",
            dir_fd=layout.repository_fd,
            follow_symlinks=False,
        )
        source_status = os.fstat(source_fd)
        snapshots_status = os.fstat(layout.snapshots_fd)
        if (
            not stat.S_ISDIR(source_entry.st_mode)
            or (source_entry.st_dev, source_entry.st_ino)
            != (source_status.st_dev, source_status.st_ino)
            or not stat.S_ISDIR(snapshots_entry.st_mode)
            or (snapshots_entry.st_dev, snapshots_entry.st_ino)
            != (snapshots_status.st_dev, snapshots_status.st_ino)
        ):
            raise ValueError("HF snapshot topology identity changed during staging")
        if layout.blobs_fd is not None:
            blobs_entry = os.stat(
                "blobs",
                dir_fd=layout.repository_fd,
                follow_symlinks=False,
            )
            blobs_status = os.fstat(layout.blobs_fd)
            if not stat.S_ISDIR(blobs_entry.st_mode) or (
                blobs_entry.st_dev,
                blobs_entry.st_ino,
            ) != (blobs_status.st_dev, blobs_status.st_ino):
                raise ValueError("HF blob directory identity changed during staging")
    except OSError as exc:
        raise ValueError(f"HF snapshot topology is unavailable: {exc}") from exc


def _open_source_artifact(
    source_root: Path,
    source_fd: int,
    name: str,
    hf_blob_layout: _HfBlobLayout | None,
) -> tuple[_SourceArtifact, _HfBlobLayout | None]:
    name = _require_flat_target_name(name)
    try:
        initial_entry = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"required source artifact {name} is unavailable: {exc}"
        ) from exc
    artifact_fd: int | None = None
    try:
        if stat.S_ISREG(initial_entry.st_mode):
            artifact_fd = os.open(name, _read_flags(), dir_fd=source_fd)
            entry_directory_fd = source_fd
            entry_name = name
            link_device = None
            link_inode = None
            link_target = None
        elif stat.S_ISLNK(initial_entry.st_mode):
            link_target = os.readlink(name, dir_fd=source_fd)
            target = PurePosixPath(link_target)
            if (
                "\\" in link_target
                or len(target.parts) != 4
                or target.parts[:3] != ("..", "..", "blobs")
            ):
                raise ValueError(
                    f"source artifact {name} is not an exact HF blob symlink"
                )
            blob_name = _require_flat_target_name(target.parts[3])
            if link_target != f"../../blobs/{blob_name}":
                raise ValueError(
                    f"source artifact {name} is not an exact HF blob symlink"
                )
            repeated_link = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            if not stat.S_ISLNK(repeated_link.st_mode) or (
                repeated_link.st_dev,
                repeated_link.st_ino,
            ) != (initial_entry.st_dev, initial_entry.st_ino):
                raise ValueError(f"source artifact {name} changed while resolving")
            if hf_blob_layout is None:
                raise ValueError(
                    "source artifact symlink is outside a pinned HF snapshot layout: "
                    f"{source_root / name}"
                )
            blob_directory_fd = _open_hf_blob_directory(hf_blob_layout)
            blob_entry = os.stat(
                blob_name,
                dir_fd=blob_directory_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(blob_entry.st_mode):
                raise ValueError(f"HF blob for {name} is not a regular file")
            artifact_fd = os.open(
                blob_name,
                _read_flags(),
                dir_fd=blob_directory_fd,
            )
            entry_directory_fd = blob_directory_fd
            entry_name = blob_name
            initial_entry = blob_entry
            link_device = repeated_link.st_dev
            link_inode = repeated_link.st_ino
        else:
            raise ValueError(f"source artifact {name} is not a regular file")
        descriptor_status = os.fstat(artifact_fd)
        repeated_entry = os.stat(
            entry_name,
            dir_fd=entry_directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or not stat.S_ISREG(repeated_entry.st_mode)
            or (descriptor_status.st_dev, descriptor_status.st_ino)
            != (initial_entry.st_dev, initial_entry.st_ino)
            or (repeated_entry.st_dev, repeated_entry.st_ino)
            != (descriptor_status.st_dev, descriptor_status.st_ino)
        ):
            raise ValueError(f"source artifact {name} changed while opening")
        artifact = _SourceArtifact(
            name=name,
            fd=artifact_fd,
            size=descriptor_status.st_size,
            device=descriptor_status.st_dev,
            inode=descriptor_status.st_ino,
            entry_directory_fd=entry_directory_fd,
            entry_name=entry_name,
            entry_device=repeated_entry.st_dev,
            entry_inode=repeated_entry.st_ino,
            link_device=link_device,
            link_inode=link_inode,
            link_target=link_target,
        )
        artifact_fd = None
        return artifact, hf_blob_layout
    except OSError as exc:
        raise ValueError(f"could not open source artifact {name}: {exc}") from exc
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)


def _recheck_source_artifacts(
    source_fd: int,
    artifacts: dict[str, _SourceArtifact],
    receipts: dict[str, _CopyReceipt],
    hf_blob_layout: _HfBlobLayout | None,
    *,
    chunk_bytes: int,
) -> None:
    if hf_blob_layout is not None:
        _assert_hf_layout_identity(source_fd, hf_blob_layout)
    for name, artifact in artifacts.items():
        receipt = receipts[name]
        descriptor_status = os.fstat(artifact.fd)
        entry_status = os.stat(
            artifact.entry_name,
            dir_fd=artifact.entry_directory_fd,
            follow_symlinks=False,
        )
        identity = (artifact.device, artifact.inode)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or not stat.S_ISREG(entry_status.st_mode)
            or descriptor_status.st_size != artifact.size
            or (descriptor_status.st_dev, descriptor_status.st_ino) != identity
            or (entry_status.st_dev, entry_status.st_ino)
            != (artifact.entry_device, artifact.entry_inode)
            or (artifact.entry_device, artifact.entry_inode) != identity
            or (receipt.source_device, receipt.source_inode) != identity
            or receipt.size != artifact.size
        ):
            raise ValueError(f"source artifact {name} identity changed during staging")
        if artifact.link_target is not None:
            link_status = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            if (
                not stat.S_ISLNK(link_status.st_mode)
                or (link_status.st_dev, link_status.st_ino)
                != (artifact.link_device, artifact.link_inode)
                or os.readlink(name, dir_fd=source_fd) != artifact.link_target
            ):
                raise ValueError(f"source HF link {name} changed during staging")
        if (
            _hash_fd(
                artifact.fd,
                length=descriptor_status.st_size,
                chunk_bytes=chunk_bytes,
            )
            != receipt.sha256
        ):
            raise ValueError(f"source artifact {name} contents changed during staging")


def _copy_independent_file(
    source_fd: int,
    directory_fd: int,
    target_name: str,
    *,
    chunk_bytes: int,
    expected_sha256: str | None,
) -> _CopyReceipt:
    target_fd: int | None = None
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError(f"source artifact is not a regular file: {target_name}")
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
        if (source_after.st_dev, source_after.st_ino, source_after.st_size) != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
        ) or target_status.st_size != source_before.st_size:
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
    source_fd, hf_blob_layout = _open_source_directory(source_root)
    try:
        work_fd = os.open(work_root, _directory_flags())
    except BaseException:
        if hf_blob_layout is not None:
            _close_hf_blob_layout(hf_blob_layout)
        os.close(source_fd)
        raise
    source_artifacts: dict[str, _SourceArtifact] = {}
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
        index_source, hf_blob_layout = _open_source_artifact(
            source_root,
            source_fd,
            _INDEX_FILE,
            hf_blob_layout,
        )
        source_artifacts[_INDEX_FILE] = index_source
        weight_map, declared_total_size = _parse_index_fd(index_source.fd)
        shard_names = sorted(set(weight_map.values()))
        if not shard_names:
            raise ValueError("resident index contains no safetensors shards")
        reserved_names = {_INDEX_FILE, *_ANCILLARY_FILES}
        if reserved_names.intersection(shard_names):
            raise ValueError("resident index uses a reserved artifact as a shard")
        manifest_tensors = {
            tensor.tensor: tensor for tensor in source_manifest.resident_tensors
        }
        _reject_resident_contamination(
            set(weight_map) | set(manifest_tensors),
            set(shard_names),
        )
        manifest_shards = {
            shard.name: shard
            for shard in source_manifest.shards
            if shard.kind == "safetensors"
        }
        selected_shards: list[ShardInfo] = []
        source_tensors: dict[str, ResidentTensor] = {}
        source_members: list[tuple[str, _SourceArtifact, str | None]] = []
        for shard_name in shard_names:
            shard_source, hf_blob_layout = _open_source_artifact(
                source_root,
                source_fd,
                shard_name,
                hf_blob_layout,
            )
            source_artifacts[shard_name] = shard_source
            shard_sha256 = _hash_fd(
                shard_source.fd,
                length=shard_source.size,
                chunk_bytes=copy_chunk_bytes,
            )
            shard, shard_tensors = _parse_safetensors_fd(
                shard_source.fd,
                name=shard_name,
                sha256=shard_sha256,
            )
            for tensor_name, tensor in shard_tensors.items():
                if tensor_name in source_tensors:
                    raise ValueError(
                        f"resident tensor appears in multiple shards: {tensor_name}"
                    )
                if weight_map.get(tensor_name) != shard_name:
                    raise ValueError(
                        f"resident index maps {tensor_name} to the wrong shard"
                    )
                source_tensors[tensor_name] = tensor
            expected = manifest_shards.get(shard_name)
            if expected is None or expected.sha256 is None:
                raise ValueError(
                    f"resident shard lacks manifest provenance: {shard_name}"
                )
            if (
                shard.size != expected.size
                or shard.header_bytes != expected.header_bytes
                or shard.header_sha256 != expected.header_sha256
                or shard.sha256 != expected.sha256
            ):
                raise ValueError(
                    "resident shard header, size, or hash provenance mismatch: "
                    f"{shard_name}"
                )
            selected_shards.append(expected)
            source_members.append((shard_name, shard_source, expected.sha256))

        index_names = set(source_tensors)
        if set(weight_map) != index_names:
            missing = sorted(set(weight_map) - index_names)
            extra = sorted(index_names - set(weight_map))
            raise ValueError(
                "resident index/header tensor mismatch; "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        if declared_total_size is not None and declared_total_size != sum(
            tensor.length for tensor in source_tensors.values()
        ):
            raise ValueError(
                "resident index total_size does not match header inventory"
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
            if tensor != expected:
                raise ValueError(
                    f"resident metadata does not match index headers: {name}"
                )

        source_members.append((_INDEX_FILE, index_source, None))
        for name in _ANCILLARY_FILES:
            source, hf_blob_layout = _open_source_artifact(
                source_root,
                source_fd,
                name,
                hf_blob_layout,
            )
            source_artifacts[name] = source
            source_members.append((name, source, None))
        target_names = [name for name, _source, _digest in source_members]
        if len(target_names) != len(set(target_names)):
            raise ValueError("resident staging target names are not unique")

        receipts: dict[str, _CopyReceipt] = {}
        final_fds: dict[str, int] = {}
        try:
            for name, source, expected_sha256 in source_members:
                receipts[name] = _copy_independent_file(
                    source.fd,
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
            _recheck_source_artifacts(
                source_fd,
                source_artifacts,
                receipts,
                hf_blob_layout,
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
        for source in source_artifacts.values():
            os.close(source.fd)
        if hf_blob_layout is not None:
            _close_hf_blob_layout(hf_blob_layout)
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


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def _sha256_path(path: Path, *, expected_size: int | None = None) -> tuple[str, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"source file is not regular: {path}")
        if expected_size is not None and metadata.st_size != expected_size:
            raise ValueError(
                f"source file size mismatch for {path.name}: "
                f"{metadata.st_size} != {expected_size}"
            )
        fd = os.open(path, _read_flags())
    except OSError as exc:
        raise ValueError(f"source file is unavailable: {path}: {exc}") from exc
    try:
        descriptor = os.fstat(fd)
        repeated = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor.st_mode)
            or (descriptor.st_dev, descriptor.st_ino)
            != (metadata.st_dev, metadata.st_ino)
            or (repeated.st_dev, repeated.st_ino)
            != (descriptor.st_dev, descriptor.st_ino)
        ):
            raise ValueError(f"source file changed while opening: {path}")
        digest = _hash_fd(
            fd,
            length=descriptor.st_size,
            chunk_bytes=8 * 1024**2,
        )
        final = os.fstat(fd)
        if (final.st_dev, final.st_ino, final.st_size) != (
            descriptor.st_dev,
            descriptor.st_ino,
            descriptor.st_size,
        ):
            raise ValueError(f"source file changed while hashing: {path}")
        return digest, descriptor.st_size
    finally:
        os.close(fd)


def _read_json_path(path: Path, *, max_bytes: int, label: str) -> tuple[Any, str, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise ValueError(f"{label} must be a bounded regular file")
        fd = os.open(path, _read_flags())
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {exc}") from exc
    try:
        descriptor = os.fstat(fd)
        if (descriptor.st_dev, descriptor.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ValueError(f"{label} changed while opening")
        payload = _pread_exact(fd, 0, descriptor.st_size, label=label)
        repeated = os.fstat(fd)
        if (repeated.st_dev, repeated.st_ino, repeated.st_size) != (
            descriptor.st_dev,
            descriptor.st_ino,
            descriptor.st_size,
        ):
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(fd)
    try:
        value = json.loads(payload, object_pairs_hook=_strict_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    return value, hashlib.sha256(payload).hexdigest(), len(payload)


def _open_pinned_preflight_inputs(config: ConversionConfig) -> _PinnedPreflightInputs:
    source_root = _require_real_directory(config.source_root, label="source root")
    output_parent = _require_real_directory(
        config.output_root.parent,
        label="output parent",
    )
    output_parent_fd: int | None = None
    source_fd: int | None = None
    try:
        output_parent_fd = os.open(output_parent, _directory_flags())
        _assert_directory_path_identity(
            output_parent,
            output_parent_fd,
            label="output parent",
        )
        source_entry = os.stat(
            source_root.name,
            dir_fd=output_parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(source_entry.st_mode):
            raise ValueError("source root must be a real directory")
        source_fd = os.open(
            source_root.name,
            _directory_flags(),
            dir_fd=output_parent_fd,
        )
        source_descriptor = os.fstat(source_fd)
        source_repeated = os.stat(
            source_root.name,
            dir_fd=output_parent_fd,
            follow_symlinks=False,
        )
        source_identity = (source_descriptor.st_dev, source_descriptor.st_ino)
        if (
            not stat.S_ISDIR(source_descriptor.st_mode)
            or (source_entry.st_dev, source_entry.st_ino) != source_identity
            or (source_repeated.st_dev, source_repeated.st_ino) != source_identity
        ):
            raise ValueError("source root identity changed while opening")
        _assert_directory_path_identity(source_root, source_fd, label="source root")

        try:
            final_entry = os.stat(
                config.output_root.name,
                dir_fd=output_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            final_entry = None
        if final_entry is not None:
            raise ValueError("final output root already exists")
        try:
            work_entry = os.stat(
                _WORK_DIRECTORY_NAME,
                dir_fd=output_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            work_entry = None
        if work_entry is not None and not stat.S_ISDIR(work_entry.st_mode):
            raise ValueError("conversion work root must be a real directory")

        result = _PinnedPreflightInputs(
            source_root=source_root,
            source_fd=source_fd,
            hf_blob_layout=None,
            output_parent=output_parent,
            output_parent_fd=output_parent_fd,
            artifacts={},
        )
        source_fd = None
        output_parent_fd = None
        return result
    except OSError as exc:
        raise ValueError(f"could not pin conversion directories: {exc}") from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if output_parent_fd is not None:
            os.close(output_parent_fd)


def _assert_source_artifact_identity(
    pinned: _PinnedPreflightInputs,
    artifact: _SourceArtifact,
) -> None:
    try:
        descriptor = os.fstat(artifact.fd)
        entry = os.stat(
            artifact.entry_name,
            dir_fd=artifact.entry_directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(
            f"source artifact {artifact.name} identity is unavailable: {exc}"
        ) from exc
    identity = (artifact.device, artifact.inode)
    if (
        not stat.S_ISREG(descriptor.st_mode)
        or not stat.S_ISREG(entry.st_mode)
        or descriptor.st_size != artifact.size
        or (descriptor.st_dev, descriptor.st_ino) != identity
        or (entry.st_dev, entry.st_ino) != identity
        or (artifact.entry_device, artifact.entry_inode) != identity
    ):
        raise ValueError(f"source artifact {artifact.name} identity changed")
    if artifact.link_target is not None:
        try:
            link = os.stat(
                artifact.name,
                dir_fd=pinned.source_fd,
                follow_symlinks=False,
            )
            link_target = os.readlink(artifact.name, dir_fd=pinned.source_fd)
        except OSError as exc:
            raise ValueError(
                f"source HF link {artifact.name} is unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISLNK(link.st_mode)
            or (link.st_dev, link.st_ino) != (artifact.link_device, artifact.link_inode)
            or link_target != artifact.link_target
        ):
            raise ValueError(f"source HF link {artifact.name} identity changed")


def _assert_pinned_preflight_identities(pinned: _PinnedPreflightInputs) -> None:
    if pinned.closed:
        raise ValueError("pinned preflight inputs are closed")
    _assert_directory_path_identity(
        pinned.output_parent,
        pinned.output_parent_fd,
        label="output parent",
    )
    _assert_directory_path_identity(
        pinned.source_root,
        pinned.source_fd,
        label="source root",
    )
    try:
        source_entry = os.stat(
            pinned.source_root.name,
            dir_fd=pinned.output_parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(f"source root identity is unavailable: {exc}") from exc
    source_descriptor = os.fstat(pinned.source_fd)
    if not stat.S_ISDIR(source_entry.st_mode) or (
        source_entry.st_dev,
        source_entry.st_ino,
    ) != (source_descriptor.st_dev, source_descriptor.st_ino):
        raise ValueError("source root identity changed under output parent")
    if pinned.hf_blob_layout is not None:
        _assert_hf_layout_identity(pinned.source_fd, pinned.hf_blob_layout)
    for artifact in pinned.artifacts.values():
        _assert_source_artifact_identity(pinned, artifact)


def _hash_pinned_source_artifact(
    pinned: _PinnedPreflightInputs,
    name: str,
    *,
    expected_size: int | None = None,
) -> tuple[str, int]:
    artifact = pinned.open_artifact(name)
    if expected_size is not None and artifact.size != expected_size:
        raise ValueError(
            f"source file size mismatch for {name}: {artifact.size} != {expected_size}"
        )
    digest = _hash_fd(
        artifact.fd,
        length=artifact.size,
        chunk_bytes=8 * 1024**2,
    )
    _assert_source_artifact_identity(pinned, artifact)
    return digest, artifact.size


def _read_pinned_json_artifact(
    pinned: _PinnedPreflightInputs,
    name: str,
    *,
    max_bytes: int,
    label: str,
) -> tuple[Any, str, int]:
    artifact = pinned.open_artifact(name)
    if artifact.size > max_bytes:
        raise ValueError(f"{label} must be a bounded regular file")
    payload = _pread_exact(artifact.fd, 0, artifact.size, label=label)
    _assert_source_artifact_identity(pinned, artifact)
    try:
        value = json.loads(payload, object_pairs_hook=_strict_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    return value, hashlib.sha256(payload).hexdigest(), len(payload)


def _close_preflight_context(context: _PreflightContext) -> None:
    context.pinned.close()


def _work_root(config: ConversionConfig) -> Path:
    work_root = config.output_root.with_name(_WORK_DIRECTORY_NAME)
    if work_root.parent != config.output_root.parent:
        raise ValueError("work root is not the stable output sibling")
    return work_root


def _descriptor_component_metadata(
    spec: ExpertStreamingModelSpec,
) -> tuple[tuple[str, str, tuple[int, int], int], ...]:
    result: list[tuple[str, str, tuple[int, int], int]] = []
    for projection in _PROJECTIONS:
        input_size = (
            spec.hidden_size if projection != "down_proj" else spec.expert_hidden_size
        )
        output_size = (
            spec.expert_hidden_size if projection != "down_proj" else spec.hidden_size
        )
        shapes = (
            (output_size, input_size * spec.quant_bits // 32),
            (output_size, input_size // spec.quant_group_size),
            (output_size, input_size // spec.quant_group_size),
        )
        for leaf, dtype, shape in zip(_LEAVES, _DTYPES, shapes, strict=True):
            item_size = 4 if dtype == "U32" else 2
            result.append(
                (
                    f"{projection}.{leaf}",
                    dtype,
                    shape,
                    shape[0] * shape[1] * item_size,
                )
            )
    return tuple(result)


def _validate_upstream_provenance(
    value: Any,
    expectations: _ConversionExpectations,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("source provenance must be an object")
    source = value.get("source")
    oracle = value.get("oracle")
    if not isinstance(source, dict) or not isinstance(oracle, dict):
        raise ValueError("source provenance lacks source or oracle identity")
    if (
        source.get("repo") != expectations.resident_source_repo
        or source.get("revision") != expectations.resident_source_revision
    ):
        raise ValueError("source provenance resident identity mismatch")
    if (
        oracle.get("repo") != expectations.oracle_repo
        or oracle.get("revision") != expectations.oracle_revision
    ):
        raise ValueError("source provenance Q4 oracle identity mismatch")


def _validate_preflight_context(
    config: ConversionConfig,
    *,
    deep_source_hash: bool,
    pinned: _PinnedPreflightInputs,
) -> _PreflightContext:
    if not isinstance(config, ConversionConfig):
        raise TypeError("config must be a ConversionConfig")
    if not isinstance(deep_source_hash, bool):
        raise TypeError("deep_source_hash must be a bool")
    expectations = _conversion_expectations()
    if config.source_root != expectations.source_root:
        raise ValueError("source path does not match the pinned Hy3 Q4 artifact")
    if config.alignment != expectations.alignment:
        raise ValueError("conversion alignment does not match the pinned descriptor")
    if expectations.target_record_bytes % config.alignment:
        raise ValueError("target record bytes are not exactly alignment-divisible")
    if (
        expectations.record_count * expectations.target_record_bytes
        != expectations.target_sidecar_bytes
    ):
        raise ValueError("target record count and bytes do not equal sidecar bytes")
    if (
        expectations.target_sidecar_bytes + expectations.resident_tensor_bytes
        != expectations.target_tensor_bytes
    ):
        raise ValueError("target routed and resident bytes do not equal tensor bytes")
    source_root = pinned.source_root

    manifest_value, manifest_file_sha256, manifest_file_bytes = (
        _read_pinned_json_artifact(
            pinned,
            config.source_manifest.name,
            max_bytes=_MAX_MANIFEST_BYTES,
            label="source expert manifest",
        )
    )
    if manifest_file_sha256 != expectations.manifest_file_sha256:
        raise ValueError("source manifest file hash mismatch")
    try:
        manifest = ExpertManifest.from_dict(manifest_value)
    except ValueError as exc:
        raise ValueError(f"source manifest is invalid: {exc}") from exc
    if manifest.manifest_sha256 != expectations.manifest_sha256:
        raise ValueError("source canonical manifest hash mismatch")
    source_descriptor = _source_descriptor()
    target_descriptor = _target_descriptor()
    validate_expert_manifest_spec(manifest, source_descriptor)
    if (
        manifest.model_key != SOURCE_MODEL_KEY
        or manifest.quant_bits != 4
        or manifest.quant_group_size != 64
        or manifest.quant_mode != "affine"
    ):
        raise ValueError("source manifest is not the pinned affine Q4 model")
    if (
        target_descriptor.key != TARGET_MODEL_KEY
        or target_descriptor.quant_bits != 2
        or target_descriptor.quant_group_size != 64
        or target_descriptor.mtp_included
    ):
        raise ValueError(
            "target descriptor is not the explicit AR-only affine Q2 model"
        )
    if source_descriptor.expert_record_bytes != expectations.source_record_bytes:
        raise ValueError("source descriptor record bytes mismatch")
    if target_descriptor.expert_record_bytes != expectations.target_record_bytes:
        raise ValueError("target descriptor record bytes mismatch")
    if len(manifest.records) != expectations.record_count:
        raise ValueError("source record count mismatch")
    expected_keys = tuple(
        (layer, expert)
        for layer in source_descriptor.routed_layer_indices
        for expert in range(source_descriptor.expert_count)
    )
    if (
        tuple((record.layer, record.expert) for record in manifest.records)
        != expected_keys
    ):
        raise ValueError("source record Cartesian product mismatch")
    for ordinal, record in enumerate(manifest.records):
        if (
            record.logical_bytes != expectations.source_record_bytes
            or record.sidecar_offset != ordinal * expectations.source_record_bytes
            or record.sidecar_length != expectations.source_record_bytes
            or record.sha256 is None
        ):
            raise ValueError("source record metadata is not canonical Q4")
    if (
        manifest.resident_tensor_bytes != expectations.resident_tensor_bytes
        or target_descriptor.total_tensor_bytes != expectations.target_tensor_bytes
    ):
        raise ValueError("resident or target tensor byte accounting mismatch")
    if manifest.sidecar is None or (
        manifest.sidecar.file != _SIDECAR_FILE
        or manifest.sidecar.alignment != expectations.alignment
        or manifest.sidecar.size != expectations.source_sidecar_bytes
        or manifest.sidecar.sha256 != expectations.sidecar_sha256
    ):
        raise ValueError("source sidecar metadata mismatch")

    provenance, provenance_sha256, provenance_bytes = _read_pinned_json_artifact(
        pinned,
        config.source_provenance.name,
        max_bytes=_MAX_PROVENANCE_BYTES,
        label="source conversion provenance",
    )
    if provenance_sha256 != expectations.provenance_sha256:
        raise ValueError("source conversion provenance hash mismatch")
    _validate_upstream_provenance(provenance, expectations)

    index_sha256, index_bytes = _hash_pinned_source_artifact(pinned, _INDEX_FILE)
    config_sha256, config_bytes = _hash_pinned_source_artifact(
        pinned,
        "config.json",
    )
    if index_sha256 != expectations.index_sha256:
        raise ValueError("source resident index hash mismatch")
    if config_sha256 != expectations.config_sha256:
        raise ValueError("source config hash mismatch")
    sidecar_artifact = pinned.open_artifact(_SIDECAR_FILE)
    if sidecar_artifact.size != expectations.source_sidecar_bytes:
        raise ValueError("source sidecar size mismatch")
    if deep_source_hash:
        actual_sidecar_sha256, _size = _hash_pinned_source_artifact(
            pinned,
            _SIDECAR_FILE,
            expected_size=expectations.source_sidecar_bytes,
        )
        if actual_sidecar_sha256 != expectations.sidecar_sha256:
            raise ValueError("source sidecar hash mismatch")

    manifest_shards = {shard.name: shard for shard in manifest.shards}
    resident_shard_names = sorted(
        {tensor.shard for tensor in manifest.resident_tensors}
    )
    if len(resident_shard_names) != expectations.resident_shard_count:
        raise ValueError("resident shard count mismatch")
    resident_files: list[dict[str, Any]] = []
    resident_physical_bytes = 0
    for name in resident_shard_names:
        shard = manifest_shards.get(name)
        if shard is None or shard.kind != "safetensors" or shard.sha256 is None:
            raise ValueError(f"resident shard provenance is incomplete: {name}")
        resident_artifact = pinned.open_artifact(name)
        if resident_artifact.size != shard.size:
            raise ValueError(f"resident shard size mismatch: {name}")
        if deep_source_hash:
            actual_sha256, _size = _hash_pinned_source_artifact(
                pinned,
                name,
                expected_size=shard.size,
            )
            if actual_sha256 != shard.sha256:
                raise ValueError(f"resident shard hash mismatch: {name}")
        resident_physical_bytes += shard.size
        resident_files.append(
            {"file": name, "size": shard.size, "sha256": shard.sha256}
        )

    ancillary_files: list[dict[str, Any]] = [
        {"file": _INDEX_FILE, "size": index_bytes, "sha256": index_sha256},
        {"file": "config.json", "size": config_bytes, "sha256": config_sha256},
    ]
    ancillary_physical_bytes = index_bytes + config_bytes
    for name in _ANCILLARY_FILES:
        if name == "config.json":
            continue
        digest, size = _hash_pinned_source_artifact(pinned, name)
        ancillary_files.append({"file": name, "size": size, "sha256": digest})
        ancillary_physical_bytes += size

    producer = _producer_state()
    if producer.get("dirty") is not False:
        raise ValueError("producer Git state is dirty")
    commit = producer.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("producer Git commit is invalid")
    mlx_version = _mlx_version()
    if not isinstance(mlx_version, str) or not mlx_version:
        raise ValueError("MLX version is invalid")
    pilot_sha256 = None
    pilot_bytes = 0
    if config.pilot_report is not None:
        pilot_sha256, pilot_bytes = _sha256_path(config.pilot_report)

    source_fingerprint_value = {
        "source_root": os.fspath(source_root),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "provenance_sha256": provenance_sha256,
        "index_sha256": index_sha256,
        "config_sha256": config_sha256,
        "sidecar_sha256": expectations.sidecar_sha256,
        "sidecar_bytes": expectations.source_sidecar_bytes,
        "resident_files": resident_files,
        "ancillary_files": ancillary_files,
    }
    source_fingerprint_sha256 = hashlib.sha256(
        _canonical_json_bytes(source_fingerprint_value)
    ).hexdigest()
    report = _minimum_conversion_manifest(expectations)
    report["source"].update(
        {
            "path": os.fspath(source_root),
            "record_count": len(manifest.records),
            "record_bytes": expectations.source_record_bytes,
            "sidecar_bytes": expectations.source_sidecar_bytes,
            "index_sha256": index_sha256,
            "config_sha256": config_sha256,
            "fingerprint_sha256": source_fingerprint_sha256,
            "resident_files": resident_files,
            "ancillary_files": ancillary_files,
        }
    )
    report["producer"] = dict(producer)
    report["mlx_version"] = mlx_version
    report["target_descriptor"] = _target_descriptor_state(target_descriptor)
    report["alignment"] = config.alignment
    report["resident_copy_policy"] = "exact-independent-whole-file"
    report["pilot_report_sha256"] = pilot_sha256
    manifest_header_overhead = (
        manifest_file_bytes
        + provenance_bytes
        + pilot_bytes
        + expectations.record_count * expectations.alignment
    )
    base_bytes = (
        expectations.target_sidecar_bytes
        + resident_physical_bytes
        + ancillary_physical_bytes
        + manifest_header_overhead
    )
    required_bytes = (base_bytes * 105 + 99) // 100 + _PROJECTION_WORKING_RESERVE
    try:
        free_bytes = shutil.disk_usage(pinned.output_parent_fd).free
    except OSError as exc:
        raise ValueError(f"could not determine output free space: {exc}") from exc
    report["space"] = {
        "target_sidecar_bytes": expectations.target_sidecar_bytes,
        "resident_file_bytes": resident_physical_bytes,
        "ancillary_file_bytes": ancillary_physical_bytes,
        "manifest_header_overhead_bytes": manifest_header_overhead,
        "base_bytes": base_bytes,
        "safety_margin_percent": 5,
        "projection_working_reserve_bytes": _PROJECTION_WORKING_RESERVE,
        "required_bytes": required_bytes,
        "free_bytes": free_bytes,
    }
    if free_bytes < required_bytes:
        raise ValueError(
            f"insufficient free space: {free_bytes} available; {required_bytes} required"
        )
    _assert_pinned_preflight_identities(pinned)
    return _PreflightContext(
        manifest=manifest,
        report=report,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        expectations=expectations,
        pinned=pinned,
    )


def _preflight_context(
    config: ConversionConfig,
    *,
    deep_source_hash: bool,
) -> _PreflightContext:
    if not isinstance(config, ConversionConfig):
        raise TypeError("config must be a ConversionConfig")
    if not isinstance(deep_source_hash, bool):
        raise TypeError("deep_source_hash must be a bool")
    pinned = _open_pinned_preflight_inputs(config)
    try:
        return _validate_preflight_context(
            config,
            deep_source_hash=deep_source_hash,
            pinned=pinned,
        )
    except BaseException:
        pinned.close()
        raise


def preflight_hy3_expert_q2(
    config: ConversionConfig,
    *,
    deep_source_hash: bool,
) -> dict[str, Any]:
    """Fail closed on every pinned source, producer, target, and space gate."""

    context = _preflight_context(
        config,
        deep_source_hash=deep_source_hash,
    )
    try:
        return context.report
    finally:
        _close_preflight_context(context)


def _journal_header(
    config: ConversionConfig,
    context: _PreflightContext,
) -> dict[str, Any]:
    report = context.report
    body = {
        "kind": "header",
        "schema": _JOURNAL_SCHEMA,
        "conversion_schema": _CONVERSION_SCHEMA,
        "source": report["source"],
        "derivation": report["derivation"],
        "target": report["target"],
        "producer": report["producer"],
        "mlx_version": report["mlx_version"],
        "target_descriptor": report["target_descriptor"],
        "alignment": config.alignment,
        "resident_copy_policy": report["resident_copy_policy"],
        "pilot_report_sha256": report["pilot_report_sha256"],
    }
    return {
        **body,
        "header_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }


def _source_record_state(
    sidecar_fd: int,
    record: ExpertRecord,
) -> tuple[bytes, dict[str, Any]]:
    if (
        record.sidecar_offset is None
        or record.sidecar_length is None
        or record.sha256 is None
    ):
        raise ValueError(
            f"source record ({record.layer}, {record.expert}) lacks sidecar provenance"
        )
    payload = _pread_exact(
        sidecar_fd,
        record.sidecar_offset,
        record.sidecar_length,
        label=f"source record ({record.layer}, {record.expert})",
    )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record.sha256:
        raise ValueError(
            f"source record hash mismatch: ({record.layer}, {record.expert})"
        )
    components = []
    for segment in record.segments:
        relative = segment.offset - record.sidecar_offset
        if relative < 0 or relative + segment.length > len(payload):
            raise ValueError("source component escapes its sidecar record")
        component_payload = payload[relative : relative + segment.length]
        components.append(
            {
                "component": segment.component,
                "tensor": segment.tensor,
                "offset": segment.offset,
                "length": segment.length,
                "dtype": segment.dtype,
                "shape": list(segment.shape),
                "sha256": hashlib.sha256(component_payload).hexdigest(),
            }
        )
    return payload, {
        "offset": record.sidecar_offset,
        "length": record.sidecar_length,
        "sha256": digest,
        "components": components,
    }


def _output_record_metadata(
    source_record: ExpertRecord,
    target_descriptor: ExpertStreamingModelSpec,
    *,
    output_offset: int,
) -> tuple[tuple[TensorSegment, ...], int]:
    metadata = _descriptor_component_metadata(target_descriptor)
    if len(source_record.segments) != len(metadata):
        raise ValueError("source record does not have nine target components")
    cursor = output_offset
    segments = []
    for source_segment, (component, dtype, shape, length) in zip(
        source_record.segments,
        metadata,
        strict=True,
    ):
        if source_segment.component != component:
            raise ValueError("source and target component order differ")
        segments.append(
            TensorSegment(
                component=component,
                tensor=source_segment.tensor,
                shard=_SIDECAR_FILE,
                offset=cursor,
                length=length,
                dtype=dtype,
                shape=shape,
            )
        )
        cursor += length
    return tuple(segments), cursor - output_offset


def _diagnostics_json(
    diagnostics: tuple[ProjectionDiagnostics, ...],
) -> list[dict[str, Any]]:
    if tuple(item.component for item in diagnostics) != _PROJECTIONS:
        raise ValueError("conversion diagnostics are not projection-complete")
    result = []
    for item in diagnostics:
        if (
            not item.finite
            or not math.isfinite(item.cosine_q4_q2)
            or not math.isfinite(item.normalized_error_q4_q2)
        ):
            raise ValueError("conversion diagnostics are non-finite")
        result.append(
            {
                "component": item.component,
                "cosine_q4_q2": item.cosine_q4_q2,
                "normalized_error_q4_q2": item.normalized_error_q4_q2,
                "finite": True,
            }
        )
    return result


def _convert_one_record(
    source_record: ExpertRecord,
    source_payload: bytes,
    target_descriptor: ExpertStreamingModelSpec,
    *,
    output_offset: int,
) -> tuple[ExpertRecord, bytes, dict[str, Any], list[dict[str, Any]]]:
    target_segments, target_length = _output_record_metadata(
        source_record,
        target_descriptor,
        output_offset=output_offset,
    )
    source_by_component = {
        segment.component: segment for segment in source_record.segments
    }
    outputs: dict[str, bytes] = {}

    def read_component(segment: TensorSegment) -> bytes:
        expected = source_by_component.get(segment.component)
        if expected != segment or source_record.sidecar_offset is None:
            raise ValueError("converter requested an unexpected source component")
        relative = segment.offset - source_record.sidecar_offset
        return source_payload[relative : relative + segment.length]

    def write_component(component: str, payload: bytes) -> None:
        if component in outputs:
            raise ValueError(f"converter emitted duplicate component {component}")
        outputs[component] = bytes(payload)

    diagnostics = requantize_expert_record_q4_to_q2(
        source_record,
        read_component,
        write_component,
        hidden_size=target_descriptor.hidden_size,
        expert_hidden_size=target_descriptor.expert_hidden_size,
        group_size=target_descriptor.quant_group_size,
    )
    if tuple(outputs) != tuple(segment.component for segment in target_segments):
        raise ValueError("converter output component order is not canonical")
    component_states = []
    chunks = []
    for segment in target_segments:
        payload = outputs[segment.component]
        if len(payload) != segment.length:
            raise ValueError(
                f"converter output length mismatch for {segment.component}"
            )
        chunks.append(payload)
        component_states.append(
            {
                "component": segment.component,
                "tensor": segment.tensor,
                "offset": segment.offset,
                "length": segment.length,
                "dtype": segment.dtype,
                "shape": list(segment.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    output_payload = b"".join(chunks)
    if len(output_payload) != target_length:
        raise ValueError("converted record output bytes are inconsistent")
    output_sha256 = hashlib.sha256(output_payload).hexdigest()
    output_record = ExpertRecord(
        layer=source_record.layer,
        expert=source_record.expert,
        logical_bytes=target_length,
        segments=target_segments,
        sha256=output_sha256,
        sidecar_offset=output_offset,
        sidecar_length=target_length,
    )
    output_state = {
        "offset": output_offset,
        "length": target_length,
        "sha256": output_sha256,
        "components": component_states,
    }
    return output_record, output_payload, output_state, _diagnostics_json(diagnostics)


def _journal_entry(
    *,
    ordinal: int,
    source_record: ExpertRecord,
    source_state: dict[str, Any],
    output_state: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    previous_sha256: str,
) -> dict[str, Any]:
    body = {
        "kind": "record",
        "ordinal": ordinal,
        "layer": source_record.layer,
        "expert": source_record.expert,
        "source": source_state,
        "output": output_state,
        "diagnostics": diagnostics,
        "previous_sha256": previous_sha256,
    }
    return {
        **body,
        "entry_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }


def _parse_journal(
    journal_fd: int,
) -> tuple[list[tuple[dict[str, Any], int, int]], int, bool]:
    metadata = os.fstat(journal_fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_JOURNAL_BYTES:
        raise ValueError("conversion journal is not a bounded regular file")
    payload = _pread_exact(
        journal_fd,
        0,
        metadata.st_size,
        label="conversion journal",
    )
    last_newline = payload.rfind(b"\n")
    if last_newline < 0:
        raise ValueError("conversion journal has no durable header")
    complete_end = last_newline + 1
    partial_tail = complete_end != len(payload)
    parsed: list[tuple[dict[str, Any], int, int]] = []
    cursor = 0
    for raw_line in payload[:complete_end].splitlines(keepends=True):
        end = cursor + len(raw_line)
        line_payload = raw_line[:-1]
        if not line_payload:
            raise ValueError("conversion journal contains an empty line")
        try:
            value = json.loads(
                line_payload,
                object_pairs_hook=_strict_json_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"conversion journal JSON is invalid: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("conversion journal line must be an object")
        parsed.append((value, cursor, end))
        cursor = end
    return parsed, complete_end, partial_tail


def _validate_journal_chain(
    parsed: list[tuple[dict[str, Any], int, int]],
    expected_header: dict[str, Any],
    source_records: tuple[ExpertRecord, ...],
) -> None:
    if not parsed or parsed[0][0] != expected_header:
        raise ValueError("resume journal header fingerprint mismatch")
    previous_sha256 = expected_header["header_sha256"]
    expected_keys = {
        "kind",
        "ordinal",
        "layer",
        "expert",
        "source",
        "output",
        "diagnostics",
        "previous_sha256",
        "entry_sha256",
    }
    if len(parsed) - 1 > len(source_records):
        raise ValueError("conversion journal contains excess records")
    for ordinal, (line, _start, _end) in enumerate(parsed[1:]):
        if set(line) != expected_keys or line.get("kind") != "record":
            raise ValueError("conversion journal record schema mismatch")
        source_record = source_records[ordinal]
        if line.get("ordinal") != ordinal or (
            line.get("layer"),
            line.get("expert"),
        ) != (source_record.layer, source_record.expert):
            raise ValueError("conversion journal is not a contiguous record prefix")
        if line.get("previous_sha256") != previous_sha256:
            raise ValueError("conversion journal hash chain mismatch")
        body = {key: value for key, value in line.items() if key != "entry_sha256"}
        entry_sha256 = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        if line.get("entry_sha256") != entry_sha256:
            raise ValueError("conversion journal entry hash mismatch")
        diagnostics = line.get("diagnostics")
        if not isinstance(diagnostics, list) or len(diagnostics) != len(_PROJECTIONS):
            raise ValueError("conversion journal diagnostics are incomplete")
        for projection, item in zip(_PROJECTIONS, diagnostics, strict=True):
            if (
                not isinstance(item, dict)
                or set(item)
                != {
                    "component",
                    "cosine_q4_q2",
                    "normalized_error_q4_q2",
                    "finite",
                }
                or item.get("component") != projection
                or item.get("finite") is not True
                or not isinstance(item.get("cosine_q4_q2"), (int, float))
                or not isinstance(item.get("normalized_error_q4_q2"), (int, float))
                or not math.isfinite(item["cosine_q4_q2"])
                or not math.isfinite(item["normalized_error_q4_q2"])
            ):
                raise ValueError("conversion journal diagnostics are invalid")
        previous_sha256 = entry_sha256


def _record_from_output_state(
    source_record: ExpertRecord,
    target_descriptor: ExpertStreamingModelSpec,
    output_state: Any,
    *,
    ordinal: int,
    target_record_bytes: int,
) -> ExpertRecord:
    output_offset = ordinal * target_record_bytes
    segments, expected_length = _output_record_metadata(
        source_record,
        target_descriptor,
        output_offset=output_offset,
    )
    if not isinstance(output_state, dict) or set(output_state) != {
        "offset",
        "length",
        "sha256",
        "components",
    }:
        raise ValueError("journal output state schema mismatch")
    if (
        output_state["offset"] != output_offset
        or output_state["length"] != expected_length
        or not isinstance(output_state["sha256"], str)
        or not isinstance(output_state["components"], list)
        or len(output_state["components"]) != len(segments)
    ):
        raise ValueError("journal output record metadata mismatch")
    for state, segment in zip(output_state["components"], segments, strict=True):
        expected = {
            "component": segment.component,
            "tensor": segment.tensor,
            "offset": segment.offset,
            "length": segment.length,
            "dtype": segment.dtype,
            "shape": list(segment.shape),
        }
        if (
            not isinstance(state, dict)
            or {key: state.get(key) for key in expected} != expected
            or set(state) != {*expected, "sha256"}
            or not isinstance(state.get("sha256"), str)
        ):
            raise ValueError("journal output component metadata mismatch")
    return ExpertRecord(
        layer=source_record.layer,
        expert=source_record.expert,
        logical_bytes=expected_length,
        segments=segments,
        sha256=output_state["sha256"],
        sidecar_offset=output_offset,
        sidecar_length=expected_length,
    )


def _output_state_matches(
    output_fd: int,
    output_state: dict[str, Any],
) -> bool:
    output_offset = output_state["offset"]
    output_length = output_state["length"]
    metadata = os.fstat(output_fd)
    if metadata.st_size < output_offset + output_length:
        return False
    payload = _pread_exact(
        output_fd,
        output_offset,
        output_length,
        label="resumed output record",
    )
    if hashlib.sha256(payload).hexdigest() != output_state["sha256"]:
        return False
    for component in output_state["components"]:
        relative = component["offset"] - output_offset
        component_payload = payload[relative : relative + component["length"]]
        if (
            len(component_payload) != component["length"]
            or hashlib.sha256(component_payload).hexdigest() != component["sha256"]
        ):
            return False
    return True


def _open_conversion_files(
    work_fd: int,
    header: dict[str, Any],
    *,
    resume: bool,
) -> tuple[int, int, bool]:
    def entry_exists(name: str) -> bool:
        try:
            os.stat(name, dir_fd=work_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    output_exists = entry_exists(_SIDECAR_FILE)
    journal_exists = entry_exists(_JOURNAL_FILE)
    if not resume and (output_exists or journal_exists):
        raise ValueError("conversion output exists and resume is disabled")
    if output_exists != journal_exists:
        raise ValueError("resume requires both output sidecar and journal")
    if output_exists:
        output_fd: int | None = None
        try:
            output_fd = os.open(
                _SIDECAR_FILE,
                _read_write_flags(),
                dir_fd=work_fd,
            )
            journal_fd = os.open(
                _JOURNAL_FILE,
                _read_write_flags(),
                dir_fd=work_fd,
            )
            result = (output_fd, journal_fd, False)
            output_fd = None
            return result
        except OSError as exc:
            raise ValueError(f"could not open conversion resume files: {exc}") from exc
        finally:
            if output_fd is not None:
                os.close(output_fd)

    output_fd: int | None = None
    journal_fd: int | None = None
    try:
        output_fd = os.open(
            _SIDECAR_FILE,
            _write_flags(),
            0o644,
            dir_fd=work_fd,
        )
        journal_fd = os.open(
            _JOURNAL_FILE,
            _write_flags(),
            0o644,
            dir_fd=work_fd,
        )
        header_payload = _canonical_json_bytes(header) + b"\n"
        _pwrite_all(journal_fd, header_payload, 0)
        os.fsync(journal_fd)
        os.fsync(work_fd)
        result = (output_fd, journal_fd, True)
        output_fd = None
        journal_fd = None
        return result
    except BaseException:
        if journal_fd is not None:
            os.close(journal_fd)
        if output_fd is not None:
            os.close(output_fd)
        raise


def _assert_conversion_file_identity(
    work_fd: int,
    name: str,
    fd: int,
) -> None:
    try:
        entry = os.stat(name, dir_fd=work_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"conversion file {name} is unavailable: {exc}") from exc
    descriptor = os.fstat(fd)
    if (
        not stat.S_ISREG(entry.st_mode)
        or not stat.S_ISREG(descriptor.st_mode)
        or entry.st_nlink != 1
        or (entry.st_dev, entry.st_ino) != (descriptor.st_dev, descriptor.st_ino)
    ):
        raise ValueError(f"conversion file {name} identity changed")


def _open_or_create_work_root(config: ConversionConfig, parent_fd: int) -> int:
    work_fd: int | None = None
    try:
        _assert_directory_path_identity(
            config.output_root.parent,
            parent_fd,
            label="output parent",
        )
        try:
            os.stat(
                config.output_root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError("final output root already exists")
        try:
            entry = os.stat(
                _WORK_DIRECTORY_NAME,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            os.mkdir(_WORK_DIRECTORY_NAME, 0o755, dir_fd=parent_fd)
            os.fsync(parent_fd)
            entry = os.stat(
                _WORK_DIRECTORY_NAME,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        if not stat.S_ISDIR(entry.st_mode):
            raise ValueError("conversion work root must be a real directory")
        work_fd = os.open(
            _WORK_DIRECTORY_NAME,
            _directory_flags(),
            dir_fd=parent_fd,
        )
        descriptor = os.fstat(work_fd)
        repeated = os.stat(
            _WORK_DIRECTORY_NAME,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (entry.st_dev, entry.st_ino) != (descriptor.st_dev, descriptor.st_ino) or (
            repeated.st_dev,
            repeated.st_ino,
        ) != (descriptor.st_dev, descriptor.st_ino):
            raise ValueError("conversion work root changed while opening")
        result = work_fd
        work_fd = None
        return result
    finally:
        if work_fd is not None:
            os.close(work_fd)


def _assert_work_root_identity(parent_fd: int, work_fd: int) -> None:
    try:
        entry = os.stat(
            _WORK_DIRECTORY_NAME,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(f"conversion work root is unavailable: {exc}") from exc
    descriptor = os.fstat(work_fd)
    if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (
        descriptor.st_dev,
        descriptor.st_ino,
    ):
        raise ValueError("conversion work root identity changed")


def _prepare_resume_prefix(
    output_fd: int,
    journal_fd: int,
    source_fd: int,
    context: _PreflightContext,
    header: dict[str, Any],
) -> tuple[list[ExpertRecord], int]:
    parsed, complete_end, partial_tail = _parse_journal(journal_fd)
    source_records = context.manifest.records
    _validate_journal_chain(parsed, header, source_records)

    for ordinal, (line, _start, _end) in enumerate(parsed[1:]):
        _payload, current_source = _source_record_state(
            source_fd,
            source_records[ordinal],
        )
        if line.get("source") != current_source:
            raise ValueError("resume source record metadata or hash mismatch")

    prefix: list[ExpertRecord] = []
    invalid_ordinal: int | None = None
    for ordinal, (line, _start, _end) in enumerate(parsed[1:]):
        record = _record_from_output_state(
            source_records[ordinal],
            context.target_descriptor,
            line.get("output"),
            ordinal=ordinal,
            target_record_bytes=context.expectations.target_record_bytes,
        )
        if not _output_state_matches(output_fd, line["output"]):
            invalid_ordinal = ordinal
            break
        prefix.append(record)

    target_size = len(prefix) * context.expectations.target_record_bytes
    journal_size = (
        parsed[len(prefix) + 1][1] if invalid_ordinal is not None else complete_end
    )
    output_size = os.fstat(output_fd).st_size
    journal_actual_size = os.fstat(journal_fd).st_size
    needs_truncation = (
        invalid_ordinal is not None
        or partial_tail
        or output_size != target_size
        or journal_actual_size != journal_size
    )
    if needs_truncation:
        os.ftruncate(output_fd, target_size)
        os.fsync(output_fd)
        os.ftruncate(journal_fd, journal_size)
        os.fsync(journal_fd)
    previous_sha256 = (prefix and parsed[len(prefix)][0]["entry_sha256"]) or header[
        "header_sha256"
    ]
    return prefix, previous_sha256


def convert_expert_records(
    config: ConversionConfig,
    *,
    resume: bool = True,
) -> tuple[ExpertRecord, ...]:
    """Convert and durably journal the exact sorted source record bank."""

    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    context: _PreflightContext | None = None
    output_fd: int | None = None
    journal_fd: int | None = None
    work_fd: int | None = None
    try:
        context = _preflight_context(config, deep_source_hash=True)
        work_root = _work_root(config)
        header = _journal_header(config, context)
        _assert_pinned_preflight_identities(context.pinned)
        parent_fd = context.pinned.output_parent_fd
        work_fd = _open_or_create_work_root(config, parent_fd)
        output_fd, journal_fd, created = _open_conversion_files(
            work_fd,
            header,
            resume=resume,
        )
        _assert_conversion_file_identity(work_fd, _SIDECAR_FILE, output_fd)
        _assert_conversion_file_identity(work_fd, _JOURNAL_FILE, journal_fd)
        source_fd = context.pinned.artifacts[_SIDECAR_FILE].fd
        source_status = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_status.st_mode)
            or source_status.st_size != context.expectations.source_sidecar_bytes
        ):
            raise ValueError("source sidecar identity or size changed")
        if created:
            output_records: list[ExpertRecord] = []
            previous_sha256 = header["header_sha256"]
        else:
            output_records, previous_sha256 = _prepare_resume_prefix(
                output_fd,
                journal_fd,
                source_fd,
                context,
                header,
            )
        journal_offset = os.fstat(journal_fd).st_size
        for ordinal in range(len(output_records), len(context.manifest.records)):
            source_record = context.manifest.records[ordinal]
            source_payload, source_state = _source_record_state(
                source_fd,
                source_record,
            )
            output_offset = ordinal * context.expectations.target_record_bytes
            output_record, output_payload, output_state, diagnostics = (
                _convert_one_record(
                    source_record,
                    source_payload,
                    context.target_descriptor,
                    output_offset=output_offset,
                )
            )
            if output_record.logical_bytes != context.expectations.target_record_bytes:
                raise ValueError(
                    "converted record bytes do not match target descriptor"
                )
            _pwrite_all(output_fd, output_payload, output_offset)
            os.ftruncate(
                output_fd,
                output_offset + context.expectations.target_record_bytes,
            )
            os.fsync(output_fd)
            _assert_conversion_file_identity(work_fd, _SIDECAR_FILE, output_fd)
            entry = _journal_entry(
                ordinal=ordinal,
                source_record=source_record,
                source_state=source_state,
                output_state=output_state,
                diagnostics=diagnostics,
                previous_sha256=previous_sha256,
            )
            entry_payload = _canonical_json_bytes(entry) + b"\n"
            _pwrite_all(journal_fd, entry_payload, journal_offset)
            journal_offset += len(entry_payload)
            os.ftruncate(journal_fd, journal_offset)
            os.fsync(journal_fd)
            _assert_conversion_file_identity(work_fd, _JOURNAL_FILE, journal_fd)
            output_records.append(output_record)
            previous_sha256 = entry["entry_sha256"]
        if len(output_records) != context.expectations.record_count:
            raise ValueError("conversion did not produce the exact record count")
        if os.fstat(output_fd).st_size != context.expectations.target_sidecar_bytes:
            raise ValueError("conversion output sidecar size mismatch")
        _assert_conversion_file_identity(work_fd, _SIDECAR_FILE, output_fd)
        _assert_conversion_file_identity(work_fd, _JOURNAL_FILE, journal_fd)
        _assert_pinned_preflight_identities(context.pinned)
        _assert_work_root_identity(parent_fd, work_fd)
        _assert_directory_path_identity(
            work_root, work_fd, label="conversion work root"
        )
        return tuple(output_records)
    finally:
        if journal_fd is not None:
            os.close(journal_fd)
        if output_fd is not None:
            os.close(output_fd)
        if work_fd is not None:
            os.close(work_fd)
        if context is not None:
            _close_preflight_context(context)
