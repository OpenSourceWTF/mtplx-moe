#!/usr/bin/env python3
"""Compare sequential Q4/Q2 streamed quality with deterministic gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCHEMA = "mtplx-streamed-quality-v1"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_MAX_EVIDENCE_FILE_BYTES = 512 * 1024 * 1024
_MAX_QUALITY_CEILING_DECIMAL = Decimal("0.05")
_HF_REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_HF_BLOB_TARGET_RE = re.compile(r"\.\./\.\./blobs/([0-9a-f]{40}(?:[0-9a-f]{24})?)")


@dataclass(frozen=True)
class LaneConfig:
    """Immutable runtime identity and memory contract for one quality lane."""

    label: str
    model_root: Path
    manifest_path: Path
    model_key: str
    memory_limit: str | None = None
    expert_cache_limit: str | None = None
    runtime_reserve: str = "16GiB"
    max_live_kv_tokens: int = 8192
    cache_policy: str = "frequency"
    cache_scope: str = "layer"
    slot_layout: str = "direct-slots"
    transient_slots: int | None = None
    read_chunk: str = "8MiB"
    f_nocache: bool = False
    trust_sidecar: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_root", Path(self.model_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        if self.label not in {"q4", "q2"}:
            raise ValueError("lane label must be 'q4' or 'q2'")
        if not self.model_key:
            raise ValueError("lane model_key must be non-empty")
        if self.cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        if self.cache_scope not in {"layer", "global"}:
            raise ValueError("cache_scope must be 'layer' or 'global'")
        if self.slot_layout not in {
            "direct-slots",
            "component-banks",
            "metal-mmap",
        }:
            raise ValueError("unsupported slot_layout")
        if self.transient_slots is not None and (
            isinstance(self.transient_slots, bool)
            or not isinstance(self.transient_slots, int)
            or self.transient_slots <= 0
        ):
            raise ValueError("transient_slots must be a positive integer")
        for name in ("f_nocache", "trust_sidecar"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class QualityLane:
    """A lane loader plus its mandatory post-close MLX cache cleanup."""

    config: LaneConfig
    load_runtime: Callable[[], Any]
    clear_cache: Callable[[], None]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_flags(*, directory: bool = False) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("secure evidence reads require O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    if directory:
        if not hasattr(os, "O_DIRECTORY"):
            raise RuntimeError("secure evidence reads require O_DIRECTORY")
        flags |= os.O_DIRECTORY
    return flags


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_child_directory(parent_fd: int, name: str, *, label: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"invalid {label} directory member {name!r}")
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} must be a directory")
    descriptor = os.open(
        name,
        _open_flags(directory=True),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not (
            _object_identity(before)
            == _object_identity(opened)
            == _object_identity(current)
        ):
            raise ValueError(f"{label} changed while being opened")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_anchored_directory(path: Path, *, label: str) -> int:
    absolute = _absolute_lexical_path(path)
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError(f"{label} must be an absolute directory")
    descriptor = os.open(absolute.anchor, _open_flags(directory=True))
    try:
        for component in absolute.parts[1:]:
            child = _open_child_directory(descriptor, component, label=label)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_stable_descriptor(descriptor: int, *, label: str, path: Path) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if before.st_size > _MAX_EVIDENCE_FILE_BYTES:
        raise ValueError(f"{label} exceeds its size bound: {path}")
    chunks: list[bytes] = []
    remaining = before.st_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError(f"{label} ended before its declared size: {path}")
        chunks.append(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    if _stable_identity(before) != _stable_identity(after):
        raise ValueError(f"{label} changed while being read: {path}")
    return b"".join(chunks)


def _validate_hf_blob_content_address(
    blob_name: str,
    payload: bytes,
    *,
    label: str,
) -> None:
    if len(blob_name) == 64:
        computed = hashlib.sha256(payload).hexdigest()
    elif len(blob_name) == 40:
        header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
        computed = hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()
    else:
        raise ValueError(f"{label} has an unsupported blob content address")
    if computed != blob_name:
        raise ValueError(f"{label} blob content address does not match its bytes")


def _open_bound_regular_file(parent_fd: int, name: str, *, label: str) -> int:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    descriptor = os.open(name, _open_flags(), dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not (
            _object_identity(before)
            == _object_identity(opened)
            == _object_identity(current)
        ):
            raise ValueError(f"{label} changed while being opened")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _stable_hf_snapshot_symlink(
    source: Path,
    *,
    label: str,
    expected_revision_fd: int,
    expected_member: os.stat_result,
) -> tuple[Path, bytes]:
    revision_dir = source.parent
    snapshots_dir = revision_dir.parent
    repository = snapshots_dir.parent
    if snapshots_dir.name != "snapshots" or not repository.name.startswith("models--"):
        raise ValueError(f"{label} symlink is not under a Hugging Face model snapshot")
    if _HF_REVISION_RE.fullmatch(revision_dir.name) is None:
        raise ValueError(f"{label} symlink requires a pinned revision")

    descriptors: list[int] = []
    try:
        repository_fd = _open_anchored_directory(
            repository,
            label="Hugging Face repository",
        )
        descriptors.append(repository_fd)
        snapshots_fd = _open_child_directory(
            repository_fd,
            "snapshots",
            label="Hugging Face snapshots",
        )
        descriptors.append(snapshots_fd)
        revision_fd = _open_child_directory(
            snapshots_fd,
            revision_dir.name,
            label="Hugging Face pinned revision",
        )
        descriptors.append(revision_fd)
        if _object_identity(os.fstat(expected_revision_fd)) != _object_identity(
            os.fstat(revision_fd)
        ):
            raise ValueError(f"{label} snapshot ancestor changed while being opened")
        blobs_fd = _open_child_directory(
            repository_fd,
            "blobs",
            label="Hugging Face blobs",
        )
        descriptors.append(blobs_fd)

        link_before = os.stat(
            source.name,
            dir_fd=revision_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISLNK(link_before.st_mode):
            raise ValueError(f"{label} snapshot member stopped being a symlink")
        if _stable_identity(expected_member) != _stable_identity(link_before):
            raise ValueError(f"{label} snapshot member changed while being opened")
        target = os.readlink(source.name, dir_fd=revision_fd)
        match = _HF_BLOB_TARGET_RE.fullmatch(target)
        if match is None:
            raise ValueError(
                f"{label} symlink target must be exact ../../blobs/<flat-name>"
            )
        blob_name = match.group(1)
        blob_fd = _open_bound_regular_file(blobs_fd, blob_name, label=label)
        descriptors.append(blob_fd)
        link_after = os.stat(
            source.name,
            dir_fd=revision_fd,
            follow_symlinks=False,
        )
        if _stable_identity(link_before) != _stable_identity(link_after):
            raise ValueError(f"{label} snapshot member changed while being opened")
        payload = _read_stable_descriptor(blob_fd, label=label, path=source)
        _validate_hf_blob_content_address(blob_name, payload, label=label)
        link_final = os.stat(
            source.name,
            dir_fd=revision_fd,
            follow_symlinks=False,
        )
        if _stable_identity(link_before) != _stable_identity(link_final):
            raise ValueError(f"{label} snapshot member changed while being read")
        return source, payload
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _stable_file_bytes(path: Path, *, label: str) -> tuple[Path, bytes]:
    source = _absolute_lexical_path(path)
    parent_fd = _open_anchored_directory(source.parent, label=f"{label} parent")
    try:
        member = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(member.st_mode):
            return _stable_hf_snapshot_symlink(
                source,
                label=label,
                expected_revision_fd=parent_fd,
                expected_member=member,
            )
        descriptor = _open_bound_regular_file(parent_fd, source.name, label=label)
        try:
            return source, _read_stable_descriptor(
                descriptor,
                label=label,
                path=source,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _manifest_receipt(path: Path) -> dict[str, Any]:
    resolved, payload = _stable_file_bytes(path, label="expert manifest")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"expert manifest is malformed: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expert manifest must be an object: {resolved}")
    declared = value.get("manifest_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"expert manifest has no valid manifest_sha256: {resolved}")
    return {
        "path": str(resolved),
        "file_sha256": _sha256(payload),
        "declared_sha256": declared,
    }


def _flat_artifact_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} must be a flat artifact name")
    if value in {".", ".."} or "\x00" in value:
        raise ValueError(f"{label} must be a flat artifact name")
    return value


def _resident_file_size(root: Path, name: str) -> tuple[Path, int]:
    resolved_root = _absolute_lexical_path(root)
    root_fd = _open_anchored_directory(resolved_root, label="model root")
    try:
        descriptor = _open_bound_regular_file(
            root_fd,
            name,
            label=f"resident shard {name}",
        )
        try:
            first = os.fstat(descriptor)
            second = os.fstat(descriptor)
            if _stable_identity(first) != _stable_identity(second):
                raise ValueError(f"resident shard {name} changed while being inspected")
            return resolved_root / name, first.st_size
        finally:
            os.close(descriptor)
    finally:
        os.close(root_fd)


def _artifact_receipt(
    root: Path,
    manifest_path: Path,
    *,
    expected_model_key: str,
) -> dict[str, Any]:
    """Bind a lane to its physical index and manifest-proven resident shards."""

    manifest_resolved, manifest_payload = _stable_file_bytes(
        manifest_path,
        label="expert manifest",
    )
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"expert manifest is malformed: {manifest_resolved}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"expert manifest must be an object: {manifest_resolved}")
    if manifest.get("model_key") != expected_model_key:
        raise ValueError(
            "lane model key does not exactly match the expert manifest: "
            f"{expected_model_key!r}"
        )
    raw_residents = manifest.get("resident_tensors")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_residents, list) or not isinstance(raw_shards, list):
        raise ValueError("expert manifest has no resident shard provenance")
    resident_names: set[str] = set()
    for item in raw_residents:
        if not isinstance(item, dict):
            raise ValueError("expert manifest resident tensor must be an object")
        resident_names.add(
            _flat_artifact_name(item.get("shard"), label="resident shard")
        )
    if not resident_names:
        raise ValueError("expert manifest has no resident tensors")
    shard_by_name: dict[str, dict[str, Any]] = {}
    for item in raw_shards:
        if not isinstance(item, dict):
            raise ValueError("expert manifest shard must be an object")
        name = _flat_artifact_name(item.get("name"), label="manifest shard")
        if name in shard_by_name:
            raise ValueError(f"expert manifest repeats shard {name!r}")
        shard_by_name[name] = item

    resident_files = []
    resident_digest = hashlib.sha256()
    for name in sorted(resident_names):
        shard = shard_by_name.get(name)
        if shard is None or shard.get("kind", "safetensors") != "safetensors":
            raise ValueError(f"resident shard provenance is incomplete: {name}")
        size = shard.get("size")
        declared_sha256 = shard.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"resident shard has invalid size: {name}")
        if (
            not isinstance(declared_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None
        ):
            raise ValueError(f"resident shard has invalid SHA-256 provenance: {name}")
        resolved, physical_size = _resident_file_size(root, name)
        if physical_size != size:
            raise ValueError(f"resident shard size does not match provenance: {name}")
        encoded_name = name.encode("utf-8")
        resident_digest.update(len(encoded_name).to_bytes(4, "big"))
        resident_digest.update(encoded_name)
        resident_digest.update(size.to_bytes(8, "big"))
        resident_digest.update(bytes.fromhex(declared_sha256))
        resident_files.append(
            {
                "name": name,
                "path": str(resolved),
                "bytes": size,
                "declared_sha256": declared_sha256,
            }
        )

    index_path = Path(root) / "model.safetensors.index.json"
    index_resolved, index_payload = _stable_file_bytes(
        index_path,
        label="resident index",
    )
    try:
        index = json.loads(index_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"resident index is malformed: {index_resolved}: {exc}"
        ) from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("resident index has no weight map")
    index_shards = {
        _flat_artifact_name(value, label="resident index shard")
        for value in weight_map.values()
    }
    if index_shards != resident_names:
        raise ValueError("resident index and manifest shard inventories differ")
    return {
        "manifest_file_sha256": _sha256(manifest_payload),
        "index": {
            "path": str(index_resolved),
            "bytes": len(index_payload),
            "sha256": _sha256(index_payload),
        },
        "residents": {
            "algorithm": "sha256-name-size-and-declared-file-digest-v1",
            "sha256": resident_digest.hexdigest(),
            "files": resident_files,
        },
    }


def _tokenizer_receipt_changed(detail: str) -> ValueError:
    return ValueError(f"{detail} changed during tokenizer receipt")


def _revalidate_directory_binding(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: tuple[int, ...],
    *,
    label: str,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        held = os.fstat(descriptor)
    except OSError as exc:
        raise _tokenizer_receipt_changed(label) from exc
    if (
        _object_identity(current) != _object_identity(held)
        or _stable_identity(held) != expected
    ):
        raise _tokenizer_receipt_changed(label)


def _optional_member_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _revalidate_missing_members(parent_fd: int, names: Sequence[str]) -> None:
    for name in names:
        if _optional_member_stat(parent_fd, name) is not None:
            raise _tokenizer_receipt_changed(f"tokenizer {name}")


def _hf_snapshot_layout(root: Path) -> tuple[Path, Path, Path] | None:
    revision_dir = root
    snapshots_dir = revision_dir.parent
    repository = snapshots_dir.parent
    if snapshots_dir.name != "snapshots" or not repository.name.startswith("models--"):
        return None
    if _HF_REVISION_RE.fullmatch(revision_dir.name) is None:
        raise ValueError("tokenizer snapshot requires a pinned revision")
    return repository, snapshots_dir, revision_dir


def _read_regular_tokenizer_set(root: Path) -> list[tuple[str, Path, bytes]]:
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int, tuple[int, ...], str]] = []
    members: list[tuple[str, int, tuple[int, ...]]] = []
    missing: list[str] = []
    items: list[tuple[str, Path, bytes]] = []
    try:
        parent_fd = _open_anchored_directory(
            root.parent,
            label="tokenizer root parent",
        )
        descriptors.append(parent_fd)
        root_fd = _open_child_directory(parent_fd, root.name, label="tokenizer root")
        descriptors.append(root_fd)
        bindings.append(
            (
                parent_fd,
                root.name,
                root_fd,
                _stable_identity(os.fstat(root_fd)),
                "tokenizer root",
            )
        )
        for name in _TOKENIZER_FILES:
            initial = _optional_member_stat(root_fd, name)
            if initial is None:
                missing.append(name)
                continue
            if not stat.S_ISREG(initial.st_mode):
                raise ValueError(f"tokenizer {name} must be a regular file")
            descriptor = _open_bound_regular_file(
                root_fd,
                name,
                label=f"tokenizer {name}",
            )
            descriptors.append(descriptor)
            expected = _stable_identity(os.fstat(descriptor))
            payload = _read_stable_descriptor(
                descriptor,
                label=f"tokenizer {name}",
                path=root / name,
            )
            members.append((name, descriptor, expected))
            items.append((name, root / name, payload))

        for name, descriptor, expected in members:
            try:
                current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                held = os.fstat(descriptor)
            except OSError as exc:
                raise _tokenizer_receipt_changed(f"tokenizer {name}") from exc
            if not (
                _object_identity(current) == _object_identity(held)
                and _stable_identity(held) == expected
            ):
                raise _tokenizer_receipt_changed(f"tokenizer {name}")
        _revalidate_missing_members(root_fd, missing)
        for binding in reversed(bindings):
            _revalidate_directory_binding(*binding[:4], label=binding[4])
        return items
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_hf_tokenizer_set(
    root: Path,
    layout: tuple[Path, Path, Path],
) -> list[tuple[str, Path, bytes]]:
    repository, _snapshots_dir, revision_dir = layout
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int, tuple[int, ...], str]] = []
    members: list[tuple[str, tuple[int, ...], str, int, tuple[int, ...]]] = []
    missing: list[str] = []
    items: list[tuple[str, Path, bytes]] = []
    try:
        repository_parent_fd = _open_anchored_directory(
            repository.parent,
            label="Hugging Face repository parent",
        )
        descriptors.append(repository_parent_fd)
        repository_fd = _open_child_directory(
            repository_parent_fd,
            repository.name,
            label="Hugging Face repository",
        )
        descriptors.append(repository_fd)
        bindings.append(
            (
                repository_parent_fd,
                repository.name,
                repository_fd,
                _stable_identity(os.fstat(repository_fd)),
                "Hugging Face repository",
            )
        )
        snapshots_fd = _open_child_directory(
            repository_fd,
            "snapshots",
            label="Hugging Face snapshots",
        )
        descriptors.append(snapshots_fd)
        bindings.append(
            (
                repository_fd,
                "snapshots",
                snapshots_fd,
                _stable_identity(os.fstat(snapshots_fd)),
                "Hugging Face snapshots",
            )
        )
        revision_fd = _open_child_directory(
            snapshots_fd,
            revision_dir.name,
            label="Hugging Face pinned revision",
        )
        descriptors.append(revision_fd)
        bindings.append(
            (
                snapshots_fd,
                revision_dir.name,
                revision_fd,
                _stable_identity(os.fstat(revision_fd)),
                "Hugging Face pinned revision",
            )
        )
        blobs_fd = _open_child_directory(
            repository_fd,
            "blobs",
            label="Hugging Face blobs",
        )
        descriptors.append(blobs_fd)
        bindings.append(
            (
                repository_fd,
                "blobs",
                blobs_fd,
                _stable_identity(os.fstat(blobs_fd)),
                "Hugging Face blobs",
            )
        )

        for name in _TOKENIZER_FILES:
            link = _optional_member_stat(revision_fd, name)
            if link is None:
                missing.append(name)
                continue
            if not stat.S_ISLNK(link.st_mode):
                raise ValueError(f"tokenizer {name} must be a snapshot symlink")
            target = os.readlink(name, dir_fd=revision_fd)
            match = _HF_BLOB_TARGET_RE.fullmatch(target)
            if match is None:
                raise ValueError(
                    f"tokenizer {name} symlink target must be exact "
                    "../../blobs/<flat-name>"
                )
            blob_name = match.group(1)
            descriptor = _open_bound_regular_file(
                blobs_fd,
                blob_name,
                label=f"tokenizer {name}",
            )
            descriptors.append(descriptor)
            expected_blob = _stable_identity(os.fstat(descriptor))
            payload = _read_stable_descriptor(
                descriptor,
                label=f"tokenizer {name}",
                path=root / name,
            )
            _validate_hf_blob_content_address(
                blob_name,
                payload,
                label=f"tokenizer {name}",
            )
            members.append(
                (name, _stable_identity(link), blob_name, descriptor, expected_blob)
            )
            items.append((name, root / name, payload))

        for name, expected_link, blob_name, descriptor, expected_blob in members:
            try:
                current_link = os.stat(
                    name,
                    dir_fd=revision_fd,
                    follow_symlinks=False,
                )
                current_blob = os.stat(
                    blob_name,
                    dir_fd=blobs_fd,
                    follow_symlinks=False,
                )
                held_blob = os.fstat(descriptor)
            except OSError as exc:
                raise _tokenizer_receipt_changed(f"tokenizer {name}") from exc
            if not (
                _stable_identity(current_link) == expected_link
                and _object_identity(current_blob) == _object_identity(held_blob)
                and _stable_identity(held_blob) == expected_blob
            ):
                raise _tokenizer_receipt_changed(f"tokenizer {name}")
        _revalidate_missing_members(revision_fd, missing)
        for binding in reversed(bindings):
            _revalidate_directory_binding(*binding[:4], label=binding[4])
        return items
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _tokenizer_receipt(root: Path) -> dict[str, Any]:
    root = _absolute_lexical_path(root)
    layout = _hf_snapshot_layout(root)
    if layout is None:
        items = _read_regular_tokenizer_set(root)
    else:
        items = _read_hf_tokenizer_set(root, layout)
    entries = []
    digest = hashlib.sha256()
    for name, resolved, payload in items:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        entries.append(
            {
                "name": name,
                "path": str(resolved),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    if not entries or entries[0]["name"] != "tokenizer.json":
        raise ValueError(f"tokenizer.json is required under {root}")
    return {
        "algorithm": "sha256-name-and-length-prefixed-tokenizer-files-v1",
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def _corpus_receipt(paths: Sequence[Path]) -> tuple[dict[str, Any], list[str]]:
    if not paths:
        raise ValueError("at least one corpus file is required")
    digest = hashlib.sha256()
    entries = []
    texts = []
    for index, path in enumerate(paths):
        resolved, payload = _stable_file_bytes(path, label="corpus file")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus file is not UTF-8: {resolved}: {exc}") from exc
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        entries.append(
            {
                "index": index,
                "path": str(resolved),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
        texts.append(text)
    return (
        {
            "algorithm": "sha256-length-prefixed-file-bytes-v1",
            "sha256": digest.hexdigest(),
            "file_order": [entry["path"] for entry in entries],
            "files": entries,
        },
        texts,
    )


def _load_prompts(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    resolved, payload = _stable_file_bytes(path, label="prompt JSONL")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"prompt JSONL is not UTF-8: {resolved}: {exc}") from exc
    prompts = []
    names = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{resolved}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{resolved}:{line_number}: prompt must be an object")
        row = {}
        for field in ("name", "category", "prompt"):
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{resolved}:{line_number}: {field} must be a non-empty string"
                )
            row[field] = item
        if row["name"] in names:
            raise ValueError(f"{resolved}:{line_number}: duplicate prompt name")
        names.add(row["name"])
        prompts.append(row)
    if not prompts:
        raise ValueError("prompt JSONL is empty")
    return (
        {
            "path": str(resolved),
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "prompt_count": len(prompts),
            "categories": sorted({prompt["category"] for prompt in prompts}),
        },
        prompts,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, np.ndarray):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise TypeError("tokenizer.encode must return a token sequence")
    result = []
    for token in encoded:
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
            raise TypeError("tokenizer returned a non-integer token")
        value = int(token)
        if value < 0:
            raise ValueError("tokenizer returned a negative token")
        result.append(value)
    return result


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _tokenize_corpus(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    evaluation_tokens: int,
) -> tuple[list[int], list[int]]:
    per_file = [_encode(tokenizer, text) for text in texts]
    all_tokens = [token for tokens in per_file for token in tokens]
    selected = all_tokens[:evaluation_tokens]
    if len(selected) < 2:
        raise ValueError("quality corpus must yield at least two evaluation tokens")
    return selected, [len(tokens) for tokens in per_file]


def _runtime_input(runtime: Any, token_ids: Sequence[int]) -> Any:
    maker = getattr(runtime, "quality_input_array", None)
    if callable(maker):
        return maker([int(token) for token in token_ids])
    import mlx.core as mx

    return mx.array([[int(token) for token in token_ids]], dtype=mx.int32)


def _logits_array(logits: Any, *, expected_tokens: int) -> np.ndarray:
    module_name = type(logits).__module__
    if module_name == "mlx.core" or module_name.startswith("mlx."):
        import mlx.core as mx

        logits = logits.astype(mx.float32)
        mx.eval(logits)
    array = np.asarray(logits, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != expected_tokens or array.shape[1] <= 0:
        raise ValueError(
            "runtime logits must have shape [1, token_count, vocabulary_size]"
        )
    return array


def _attention_phase(name: str):
    try:
        from mtplx.attention_context import attention_phase
    except ImportError:
        return nullcontext()
    return attention_phase(name)


def _admission(runtime: Any, tokens: int):
    admit = getattr(runtime, "admit_kv_tokens", None)
    return admit(tokens) if callable(admit) else nullcontext()


def _reset_streaming(runtime: Any) -> None:
    streaming = getattr(runtime, "expert_streaming", None)
    reset = getattr(streaming, "reset", None)
    if callable(reset):
        reset()


def teacher_forced_loss(
    runtime: Any,
    token_ids: Sequence[int],
    *,
    chunk_tokens: int,
) -> dict[str, Any]:
    """Score every next token with float32 logits through one sequential cache."""

    tokens = [int(token) for token in token_ids]
    if len(tokens) < 2:
        raise ValueError("teacher-forced evaluation requires at least two tokens")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    inputs = tokens[:-1]
    targets = tokens[1:]
    cache = runtime.make_cache()
    nll_sum = 0.0
    nan_count = 0
    nonfinite_count = 0
    nll_valid = True
    chunks = 0
    _reset_streaming(runtime)
    with _admission(runtime, len(tokens)), _attention_phase("prefill"):
        for start in range(0, len(inputs), chunk_tokens):
            input_chunk = inputs[start : start + chunk_tokens]
            target_chunk = np.asarray(
                targets[start : start + chunk_tokens], dtype=np.int64
            )
            logits = runtime.forward_ar(
                _runtime_input(runtime, input_chunk), cache=cache
            )
            rows = _logits_array(logits, expected_tokens=len(input_chunk))
            if np.any(target_chunk >= rows.shape[1]):
                raise ValueError("teacher-forced target token exceeds vocabulary")
            row_nan_count = int(np.isnan(rows).sum())
            row_nonfinite_count = int((~np.isfinite(rows)).sum())
            nan_count += row_nan_count
            nonfinite_count += row_nonfinite_count
            if row_nonfinite_count:
                nll_valid = False
                chunks += 1
                continue
            maximum = np.max(rows, axis=-1).astype(np.float32, copy=False)
            shifted = (rows - maximum[:, None]).astype(np.float32, copy=False)
            exponentials = np.exp(shifted).astype(np.float32, copy=False)
            totals = np.sum(exponentials, axis=-1, dtype=np.float32)
            logsumexp = (maximum + np.log(totals)).astype(np.float32, copy=False)
            selected = rows[np.arange(len(target_chunk)), target_chunk]
            losses = (logsumexp - selected).astype(np.float32, copy=False)
            loss_nan_count = int(np.isnan(losses).sum())
            loss_nonfinite_count = int((~np.isfinite(losses)).sum())
            nan_count += loss_nan_count
            nonfinite_count += loss_nonfinite_count
            if loss_nonfinite_count:
                nll_valid = False
            else:
                nll_sum += float(np.sum(losses, dtype=np.float64))
            chunks += 1
    token_count = len(targets)
    mean_nll = nll_sum / token_count if nll_valid else None
    perplexity = None
    if mean_nll is not None and math.isfinite(mean_nll):
        try:
            candidate_perplexity = math.exp(mean_nll)
        except OverflowError:
            candidate_perplexity = math.inf
        if math.isfinite(candidate_perplexity):
            perplexity = candidate_perplexity
        else:
            nll_valid = False
            nonfinite_count += 1
    finite = bool(
        nll_valid
        and nan_count == 0
        and nonfinite_count == 0
        and math.isfinite(nll_sum)
        and mean_nll is not None
        and math.isfinite(mean_nll)
        and perplexity is not None
        and math.isfinite(perplexity)
    )
    error = None
    if not finite:
        error = {
            "type": "NonFiniteQualityEvidence",
            "message": "teacher-forced loss produced nonfinite numeric evidence",
        }
    return {
        "input_token_count": len(tokens),
        "token_count": token_count,
        "chunk_tokens": chunk_tokens,
        "chunk_count": chunks,
        "nll_sum": nll_sum if finite else None,
        "mean_nll": mean_nll if finite else None,
        "perplexity": perplexity if finite else None,
        "finite": finite,
        "nan_count": nan_count,
        "nonfinite_count": nonfinite_count,
        "error": error,
    }


def _stop_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [] if value is None else [value]
    if isinstance(values, (int, np.integer)):
        values = [values]
    return {
        int(value)
        for value in values
        if not isinstance(value, bool) and isinstance(value, (int, np.integer))
    }


def greedy_outputs(
    runtime: Any,
    prompts: Sequence[dict[str, str]],
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Generate bounded greedy token arrays for diagnostics only."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    tokenizer = runtime.tokenizer
    stops = _stop_token_ids(tokenizer)
    results = []
    for index, prompt in enumerate(prompts):
        prompt_ids = _encode(tokenizer, prompt["prompt"])
        if not prompt_ids:
            raise ValueError(f"prompt {prompt['name']!r} tokenized to an empty input")
        _reset_streaming(runtime)
        cache = runtime.make_cache()
        generated: list[int] = []
        nan_count = 0
        nonfinite_count = 0
        finish_reason = "length"
        with _admission(runtime, len(prompt_ids) + max_tokens):
            with _attention_phase("prefill"):
                logits = runtime.forward_ar(
                    _runtime_input(runtime, prompt_ids), cache=cache
                )
            row = _logits_array(logits, expected_tokens=len(prompt_ids))[-1]
            for step in range(max_tokens):
                nan_count += int(np.isnan(row).sum())
                nonfinite_count += int((~np.isfinite(row)).sum())
                if not bool(np.all(np.isfinite(row))):
                    finish_reason = "nonfinite"
                    break
                token = int(np.argmax(row))
                generated.append(token)
                if token in stops:
                    finish_reason = "eos"
                    break
                if step + 1 < max_tokens:
                    with _attention_phase("ar_decode"):
                        logits = runtime.forward_ar(
                            _runtime_input(runtime, [token]), cache=cache
                        )
                    row = _logits_array(logits, expected_tokens=1)[-1]
        try:
            text = tokenizer.decode(generated)
        except TypeError:
            text = tokenizer.decode(generated, skip_special_tokens=False)
        results.append(
            {
                "prompt_index": index,
                "name": prompt["name"],
                "category": prompt["category"],
                "prompt_sha256": _sha256(prompt["prompt"].encode("utf-8")),
                "prompt_token_count": len(prompt_ids),
                "token_ids": generated,
                "generated_token_count": len(generated),
                "text": str(text),
                "finish_reason": finish_reason,
                "finite": nan_count == 0 and nonfinite_count == 0,
                "nan_count": nan_count,
                "nonfinite_count": nonfinite_count,
            }
        )
    return results


def greedy_diagnostics(
    q4_outputs: Sequence[dict[str, Any]],
    q2_outputs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare deterministic greedy arrays without making them a quality gate."""

    if len(q4_outputs) != len(q2_outputs):
        raise ValueError("greedy output prompt counts differ")
    prompt_results = []
    agreement_tokens = 0
    compared_positions = 0
    first_divergence = None
    for index, (q4, q2) in enumerate(zip(q4_outputs, q2_outputs, strict=True)):
        if q4["name"] != q2["name"]:
            raise ValueError("greedy output prompt order differs")
        q4_tokens = [int(token) for token in q4["token_ids"]]
        q2_tokens = [int(token) for token in q2["token_ids"]]
        positions = max(len(q4_tokens), len(q2_tokens))
        agreements = 0
        prompt_first = None
        for token_index in range(positions):
            q4_token = q4_tokens[token_index] if token_index < len(q4_tokens) else None
            q2_token = q2_tokens[token_index] if token_index < len(q2_tokens) else None
            if q4_token == q2_token:
                agreements += 1
                continue
            if prompt_first is None:
                prompt_first = {
                    "token_index": token_index,
                    "q4_token": q4_token,
                    "q2_token": q2_token,
                }
            if first_divergence is None:
                first_divergence = {
                    "prompt_index": index,
                    "prompt_name": q4["name"],
                    **prompt_first,
                }
        agreement_tokens += agreements
        compared_positions += positions
        prompt_results.append(
            {
                "prompt_index": index,
                "prompt_name": q4["name"],
                "agreement_tokens": agreements,
                "compared_positions": positions,
                "agreement_fraction": agreements / positions if positions else 1.0,
                "first_divergence": prompt_first,
            }
        )
    return {
        "agreement_tokens": agreement_tokens,
        "compared_positions": compared_positions,
        "agreement_fraction": (
            agreement_tokens / compared_positions if compared_positions else 1.0
        ),
        "first_divergence": first_divergence,
        "prompts": prompt_results,
    }


def quality_gate(
    q4_perplexity: float | None,
    q2_perplexity: float | None,
    *,
    finite: bool,
    max_relative_perplexity_regression: float,
) -> dict[str, Any]:
    threshold, threshold_decimal = _bounded_quality_ceiling(
        max_relative_perplexity_regression
    )
    relative_decimal = None
    relative = None
    error = None
    try:
        q4_value = float(q4_perplexity) if q4_perplexity is not None else math.nan
        q2_value = float(q2_perplexity) if q2_perplexity is not None else math.nan
    except (TypeError, ValueError):
        q4_value = math.nan
        q2_value = math.nan
    if (
        math.isfinite(q4_value)
        and q4_value > 0.0
        and math.isfinite(q2_value)
        and q2_value > 0.0
    ):
        relative_decimal = Decimal(str(q2_value)) / Decimal(str(q4_value)) - Decimal(1)
        try:
            candidate_relative = float(relative_decimal)
        except (OverflowError, ValueError):
            candidate_relative = math.inf
        if math.isfinite(candidate_relative):
            relative = candidate_relative
        else:
            error = {
                "type": "NonFiniteQualityEvidence",
                "message": "relative perplexity regression is not JSON-safe",
            }
    else:
        error = {
            "type": "NonFiniteQualityEvidence",
            "message": "perplexity inputs must be finite and positive",
        }
    quality_passed = bool(
        finite
        and relative_decimal is not None
        and relative_decimal.is_finite()
        and relative is not None
        and relative_decimal <= threshold_decimal
    )
    return {
        "relative_perplexity_regression": relative,
        "max_relative_perplexity_regression": threshold,
        "quality_passed": quality_passed,
        "error": error,
    }


def _error(
    stage: str, exc: BaseException, *, lane: str | None = None
) -> dict[str, str]:
    result = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if lane is not None:
        result["lane"] = lane
    return result


def _evaluate_lane(
    lane: QualityLane,
    *,
    corpus_texts: Sequence[str],
    prompts: Sequence[dict[str, str]],
    evaluation_tokens: int,
    chunk_tokens: int,
    greedy_max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, str]], bool]:
    config = lane.config
    result: dict[str, Any] = {
        "label": config.label,
        "model_root": str(config.model_root.expanduser().resolve()),
        "model_key": config.model_key,
    }
    errors: list[dict[str, str]] = []
    runtime = None
    load_attempted = False
    evaluation_ok = False
    cleanup_ok = True
    try:
        result["manifest"] = _manifest_receipt(config.manifest_path)
        result["tokenizer"] = _tokenizer_receipt(config.model_root)
        artifact = _artifact_receipt(
            config.model_root,
            config.manifest_path,
            expected_model_key=config.model_key,
        )
        if artifact["manifest_file_sha256"] != result["manifest"]["file_sha256"]:
            raise ValueError("expert manifest changed between artifact receipts")
        result["artifact"] = artifact
        load_attempted = True
        runtime = lane.load_runtime()
        token_ids, per_file_token_counts = _tokenize_corpus(
            runtime.tokenizer,
            corpus_texts,
            evaluation_tokens=evaluation_tokens,
        )
        result["corpus"] = {
            "token_count": len(token_ids),
            "per_file_token_counts_before_truncation": per_file_token_counts,
            "token_ids_sha256": _token_ids_sha256(token_ids),
        }
        result["loss"] = teacher_forced_loss(
            runtime,
            token_ids,
            chunk_tokens=chunk_tokens,
        )
        result["greedy_outputs"] = greedy_outputs(
            runtime,
            prompts,
            max_tokens=greedy_max_tokens,
        )
        evaluation_ok = True
    except Exception as exc:
        errors.append(_error("lane_evaluation", exc, lane=config.label))
        result["error"] = errors[-1]
    finally:
        if runtime is not None:
            try:
                runtime.close(timeout=10.0)
            except Exception as exc:
                cleanup_ok = False
                errors.append(_error("runtime_close", exc, lane=config.label))
        if load_attempted:
            runtime = None
            gc.collect()
            try:
                lane.clear_cache()
            except Exception as exc:
                cleanup_ok = False
                errors.append(_error("mlx_cache_clear", exc, lane=config.label))
    return result, errors, evaluation_ok and cleanup_ok


def compare_quality(
    q4_lane: QualityLane,
    q2_lane: QualityLane,
    *,
    corpus_files: Sequence[Path],
    prompt_file: Path,
    evaluation_tokens: int,
    chunk_tokens: int,
    greedy_max_tokens: int,
    max_relative_perplexity_regression: float = 0.05,
) -> dict[str, Any]:
    """Evaluate Q4 to completion, close it, then evaluate Q2."""

    for name, value in (
        ("evaluation_tokens", evaluation_tokens),
        ("chunk_tokens", chunk_tokens),
        ("greedy_max_tokens", greedy_max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    max_relative_perplexity_regression, _threshold_decimal = _bounded_quality_ceiling(
        max_relative_perplexity_regression
    )
    corpus, corpus_texts = _corpus_receipt(corpus_files)
    prompt_receipt, prompts = _load_prompts(prompt_file)
    errors: list[dict[str, str]] = []
    q4_result, q4_errors, q4_lane_ok = _evaluate_lane(
        q4_lane,
        corpus_texts=corpus_texts,
        prompts=prompts,
        evaluation_tokens=evaluation_tokens,
        chunk_tokens=chunk_tokens,
        greedy_max_tokens=greedy_max_tokens,
    )
    errors.extend(q4_errors)
    if q4_lane_ok:
        q2_result, q2_errors, _q2_cleanup_ok = _evaluate_lane(
            q2_lane,
            corpus_texts=corpus_texts,
            prompts=prompts,
            evaluation_tokens=evaluation_tokens,
            chunk_tokens=chunk_tokens,
            greedy_max_tokens=greedy_max_tokens,
        )
        errors.extend(q2_errors)
    else:
        q2_result = {
            "label": q2_lane.config.label,
            "model_root": str(q2_lane.config.model_root.expanduser().resolve()),
            "model_key": q2_lane.config.model_key,
            "skipped": "q4 lane did not complete safely",
        }

    q4_corpus = q4_result.get("corpus")
    q2_corpus = q2_result.get("corpus")
    if isinstance(q4_corpus, dict) and isinstance(q2_corpus, dict):
        corpus["token_count"] = q4_corpus["token_count"]
        if q4_corpus["token_ids_sha256"] != q2_corpus["token_ids_sha256"]:
            errors.append(
                {
                    "stage": "token_alignment",
                    "type": "TokenizerMismatch",
                    "message": "Q4 and Q2 corpus token IDs differ",
                }
            )
    q4_tokenizer = q4_result.get("tokenizer")
    q2_tokenizer = q2_result.get("tokenizer")
    if isinstance(q4_tokenizer, dict) and isinstance(q2_tokenizer, dict):
        if q4_tokenizer["sha256"] != q2_tokenizer["sha256"]:
            errors.append(
                {
                    "stage": "tokenizer_identity",
                    "type": "TokenizerMismatch",
                    "message": "Q4 and Q2 tokenizer artifact hashes differ",
                }
            )
    q4_artifact = q4_result.get("artifact")
    q2_artifact = q2_result.get("artifact")
    if isinstance(q4_artifact, dict) and isinstance(q2_artifact, dict):
        if q4_artifact["index"]["sha256"] != q2_artifact["index"]["sha256"]:
            errors.append(
                {
                    "stage": "resident_index_identity",
                    "type": "ArtifactMismatch",
                    "message": "Q4 and Q2 resident index hashes differ",
                }
            )
        if q4_artifact["residents"]["sha256"] != q2_artifact["residents"]["sha256"]:
            errors.append(
                {
                    "stage": "resident_identity",
                    "type": "ArtifactMismatch",
                    "message": "Q4 and Q2 resident shard hashes differ",
                }
            )

    greedy = None
    if "greedy_outputs" in q4_result and "greedy_outputs" in q2_result:
        try:
            greedy = greedy_diagnostics(
                q4_result["greedy_outputs"], q2_result["greedy_outputs"]
            )
        except Exception as exc:
            errors.append(_error("greedy_diagnostics", exc))

    lane_values = (q4_result, q2_result)
    nan_count = sum(
        int(lane.get("loss", {}).get("nan_count", 0))
        + sum(int(row.get("nan_count", 0)) for row in lane.get("greedy_outputs", []))
        for lane in lane_values
    )
    nonfinite_count = sum(
        int(lane.get("loss", {}).get("nonfinite_count", 0))
        + sum(
            int(row.get("nonfinite_count", 0)) for row in lane.get("greedy_outputs", [])
        )
        for lane in lane_values
    )
    finite = bool(
        not errors
        and nan_count == 0
        and nonfinite_count == 0
        and all(bool(lane.get("loss", {}).get("finite")) for lane in lane_values)
        and all(
            bool(row.get("finite"))
            for lane in lane_values
            for row in lane.get("greedy_outputs", [])
        )
    )
    q4_perplexity = q4_result.get("loss", {}).get("perplexity")
    q2_perplexity = q2_result.get("loss", {}).get("perplexity")
    gate = quality_gate(
        q4_perplexity,
        q2_perplexity,
        finite=finite,
        max_relative_perplexity_regression=max_relative_perplexity_regression,
    )
    passed = bool(gate["quality_passed"] and not errors)
    return {
        "schema": _SCHEMA,
        "passed": passed,
        "quality_passed": gate["quality_passed"],
        "finite": finite,
        "relative_perplexity_regression": gate["relative_perplexity_regression"],
        "max_relative_perplexity_regression": gate[
            "max_relative_perplexity_regression"
        ],
        "nan_count": nan_count,
        "nonfinite_count": nonfinite_count,
        "gate_error": gate["error"],
        "corpus": corpus,
        "prompt_file": prompt_receipt,
        "lanes": {"q4": q4_result, "q2": q2_result},
        "greedy_diagnostics": greedy,
        "errors": errors,
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _bounded_quality_ceiling(value: object) -> tuple[float, Decimal]:
    if isinstance(value, bool):
        raise ValueError("quality ceiling must be numeric")
    try:
        decimal_value = Decimal(str(value))
        result = float(decimal_value)
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("quality ceiling must be numeric") from exc
    if not decimal_value.is_finite() or not math.isfinite(result) or result < 0.0:
        raise ValueError("quality ceiling must be finite and non-negative")
    if decimal_value > _MAX_QUALITY_CEILING_DECIMAL:
        raise ValueError("quality ceiling must not exceed 0.05")
    return result, decimal_value


def _quality_ceiling_argument(value: str) -> float:
    try:
        result, _decimal = _bounded_quality_ceiling(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4-root", type=Path, required=True)
    parser.add_argument("--q4-manifest", type=Path, required=True)
    parser.add_argument("--q4-model-key", default="glm52-q4")
    parser.add_argument("--q2-root", type=Path, required=True)
    parser.add_argument("--q2-manifest", type=Path, required=True)
    parser.add_argument("--q2-model-key", default="glm52-expert-q2")
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--expert-cache-limit", required=True)
    parser.add_argument("--runtime-reserve", default="16GiB")
    parser.add_argument("--max-live-kv-tokens", type=_positive_int, required=True)
    parser.add_argument(
        "--cache-policy", choices=("frequency", "lru"), default="frequency"
    )
    parser.add_argument("--cache-scope", choices=("layer", "global"), default="layer")
    parser.add_argument(
        "--slot-layout",
        choices=("direct-slots", "component-banks", "metal-mmap"),
        default="direct-slots",
    )
    parser.add_argument("--transient-slots", type=_positive_int)
    parser.add_argument("--read-chunk", default="8MiB")
    parser.add_argument("--f-nocache", action="store_true")
    parser.add_argument(
        "--trust-sidecar",
        action="store_true",
        help="Use a separately deep-verified sidecar without per-record hashes.",
    )
    parser.add_argument(
        "--corpus-file", type=Path, action="append", required=True, dest="corpus_files"
    )
    parser.add_argument("--evaluation-tokens", type=_positive_int, required=True)
    parser.add_argument("--chunk-tokens", type=_positive_int, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--greedy-max-tokens", type=_positive_int, required=True)
    parser.add_argument(
        "--max-relative-perplexity-regression",
        type=_quality_ceiling_argument,
        default=0.05,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _load_lane_runtime(config: LaneConfig) -> Any:
    if config.memory_limit is None or config.expert_cache_limit is None:
        raise ValueError("runtime lane memory limits are required")
    from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes
    from mtplx.runtime import load

    streaming = ExpertStreamingConfig(
        model_key=config.model_key,
        memory_limit_bytes=parse_memory_bytes(config.memory_limit),
        expert_cache_limit_bytes=parse_memory_bytes(config.expert_cache_limit),
        runtime_reserve_bytes=parse_memory_bytes(config.runtime_reserve),
        max_live_kv_tokens=config.max_live_kv_tokens,
        cache_policy=config.cache_policy,
        cache_scope=config.cache_scope,
        slot_layout=config.slot_layout,
        transient_slots=config.transient_slots,
        max_read_chunk_bytes=parse_memory_bytes(config.read_chunk),
        bypass_page_cache=config.f_nocache,
        verify_record_hashes=not config.trust_sidecar,
    )
    return load(
        config.model_root,
        mtp=False,
        expert_streaming_config=streaming,
        expert_manifest=config.manifest_path,
    )


def _clear_mlx_cache() -> None:
    import mlx.core as mx

    mx.synchronize()
    mx.clear_cache()


def _lane(config: LaneConfig) -> QualityLane:
    return QualityLane(
        config=config,
        load_runtime=lambda: _load_lane_runtime(config),
        clear_cache=_clear_mlx_cache,
    )


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("short write while saving quality evidence")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(
    argv: Sequence[str] | None = None,
    *,
    _compare_quality: Callable[..., dict[str, Any]] = compare_quality,
) -> int:
    args = build_parser().parse_args(argv)
    q4_config = LaneConfig(
        "q4",
        args.q4_root,
        args.q4_manifest,
        args.q4_model_key,
        memory_limit=args.memory_limit,
        expert_cache_limit=args.expert_cache_limit,
        runtime_reserve=args.runtime_reserve,
        max_live_kv_tokens=args.max_live_kv_tokens,
        cache_policy=args.cache_policy,
        cache_scope=args.cache_scope,
        slot_layout=args.slot_layout,
        transient_slots=args.transient_slots,
        read_chunk=args.read_chunk,
        f_nocache=args.f_nocache,
        trust_sidecar=args.trust_sidecar,
    )
    q2_config = LaneConfig(
        "q2",
        args.q2_root,
        args.q2_manifest,
        args.q2_model_key,
        memory_limit=args.memory_limit,
        expert_cache_limit=args.expert_cache_limit,
        runtime_reserve=args.runtime_reserve,
        max_live_kv_tokens=args.max_live_kv_tokens,
        cache_policy=args.cache_policy,
        cache_scope=args.cache_scope,
        slot_layout=args.slot_layout,
        transient_slots=args.transient_slots,
        read_chunk=args.read_chunk,
        f_nocache=args.f_nocache,
        trust_sidecar=args.trust_sidecar,
    )
    try:
        payload = _compare_quality(
            _lane(q4_config),
            _lane(q2_config),
            corpus_files=args.corpus_files,
            prompt_file=args.prompt_file,
            evaluation_tokens=args.evaluation_tokens,
            chunk_tokens=args.chunk_tokens,
            greedy_max_tokens=args.greedy_max_tokens,
            max_relative_perplexity_regression=(
                args.max_relative_perplexity_regression
            ),
        )
    except Exception as exc:
        payload = {
            "schema": _SCHEMA,
            "passed": False,
            "quality_passed": False,
            "relative_perplexity_regression": None,
            "errors": [_error("operation", exc)],
        }
        try:
            _write_json_once(args.output_json, payload)
        except Exception as write_exc:
            print(f"compare_streamed_quality: {write_exc}", file=sys.stderr)
        return 1
    try:
        _write_json_once(args.output_json, payload)
    except Exception as exc:
        print(f"compare_streamed_quality: {exc}", file=sys.stderr)
        return 1
    if payload.get("errors"):
        return 1
    return 0 if payload.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
