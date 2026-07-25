"""One-time integrity admission for SSD-streamed expert artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    load_expert_manifest,
    resolve_artifact_member,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import get_model_spec


RECEIPT_SCHEMA = 1
DEFAULT_RECEIPT_ROOT = Path("~/.mtplx/receipts").expanduser()
_MAX_RECEIPT_BYTES = 1024 * 1024
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrustedFileDigest:
    """A digest bound to the exact file identity hashed by the downloader."""

    sha256: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int


def _resolve_receipt_root(receipt_root: Path | str | None) -> Path:
    if receipt_root is not None:
        return Path(receipt_root).expanduser().resolve()
    configured = os.environ.get("MTPLX_RECEIPT_DIR")
    selected = Path(configured).expanduser() if configured else DEFAULT_RECEIPT_ROOT
    return selected.resolve()


def _resolved_artifact_root(artifact_root: Path | str) -> Path:
    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise ExpertManifestError(f"expert artifact root is not a directory: {root}")
    return root


def _receipt_path(root: Path, receipt_root: Path) -> Path:
    resolved_receipt_root = receipt_root.resolve()
    try:
        resolved_receipt_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("expert admission receipts must be outside the artifact root")
    key = hashlib.sha256(os.fsencode(root)).hexdigest()
    return receipt_root / f"{key}.json"


def admission_receipt_path(
    artifact_root: Path | str,
    *,
    receipt_root: Path | str | None = None,
) -> Path:
    """Return the external receipt path assigned to an artifact directory."""

    root = _resolved_artifact_root(artifact_root)
    resolved_receipt_root = _resolve_receipt_root(receipt_root)
    return _receipt_path(root, resolved_receipt_root)


def _authoritative_manifest(root: Path) -> ExpertManifest:
    manifest_path = resolve_artifact_member(root, "expert-manifest.json")
    manifest = load_expert_manifest(manifest_path)
    spec = get_model_spec(manifest.model_key)
    if spec is None:
        raise ExpertManifestError(
            f"unknown model {manifest.model_key!r} in expert manifest"
        )
    validate_expert_manifest_spec(manifest, spec)
    verify_expert_manifest(
        manifest,
        root,
        verify_records=False,
        verify_shard_hashes=False,
        verify_sidecar_hash=False,
    )
    if manifest.sidecar is None:
        raise ExpertManifestError(
            "authoritative streamed manifest requires sidecar metadata"
        )
    return manifest


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _trusted_matches(
    trusted: TrustedFileDigest,
    metadata: os.stat_result,
) -> bool:
    return (
        trusted.st_dev,
        trusted.st_ino,
        trusted.st_size,
        trusted.st_mtime_ns,
        trusted.st_ctime_ns,
    ) == _identity(metadata)


def _hash_bank_descriptor(
    descriptor: int,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        try:
            chunk = os.read(descriptor, chunk_bytes)
        except InterruptedError:
            continue
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _inspect_bank(
    root: Path,
    *,
    relative_name: str,
    expected_size: int,
    expected_sha256: str,
    trusted: TrustedFileDigest | None,
) -> dict[str, Any]:
    path = resolve_artifact_member(root, relative_name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExpertManifestError(
                f"expert bank part {relative_name} is not a regular file"
            )
        if before.st_size != expected_size:
            raise ExpertManifestError(
                f"expert bank part {relative_name} size mismatch: "
                f"expected {expected_size}, got {before.st_size}"
            )
        if (
            trusted is not None
            and trusted.sha256 == expected_sha256
            and _trusted_matches(trusted, before)
        ):
            digest = trusted.sha256
        else:
            digest = _hash_bank_descriptor(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise ExpertManifestError(
            f"expert bank part {relative_name} changed during verification"
        )
    if digest != expected_sha256:
        raise ExpertManifestError(
            f"expert bank part {relative_name} SHA-256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    return {
        "file": relative_name,
        "sha256": digest,
        "st_dev": after.st_dev,
        "st_ino": after.st_ino,
        "st_size": after.st_size,
        "st_mtime_ns": after.st_mtime_ns,
        "st_ctime_ns": after.st_ctime_ns,
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_receipt(path: Path) -> dict[str, Any] | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_RECEIPT_BYTES
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            return None
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
            except InterruptedError:
                continue
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(metadata) != _identity(after):
        return None
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _receipt_bank_matches(
    root: Path,
    bank: Any,
    *,
    relative_name: str,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    if not isinstance(bank, dict):
        return False
    expected = {
        "file": relative_name,
        "sha256": expected_sha256,
        "st_size": expected_size,
    }
    if any(bank.get(key) != value for key, value in expected.items()):
        return False
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        isinstance(bank.get(field), bool)
        or not isinstance(bank.get(field), int)
        for field in identity_fields
    ):
        return False
    try:
        path = resolve_artifact_member(root, relative_name)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, ExpertManifestError):
        return False
    if not stat.S_ISREG(metadata.st_mode):
        return False
    return tuple(bank[field] for field in identity_fields) == _identity(metadata)


def load_valid_admission_receipt(
    artifact_root: Path | str,
    *,
    revision: str | None = None,
    receipt_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load a receipt only when its manifest, revision, and file identities match."""

    try:
        root = _resolved_artifact_root(artifact_root)
        resolved_receipt_root = _resolve_receipt_root(receipt_root)
        path = _receipt_path(root, resolved_receipt_root)
        receipt = _read_receipt(path)
        if receipt is None:
            return None
        if (
            type(receipt.get("schema")) is not int
            or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("artifact_root") != str(root)
            or receipt.get("receipt_path") != str(path)
            or (
                revision is not None
                and receipt.get("revision") != revision
            )
        ):
            return None
        manifest = _authoritative_manifest(root)
        if receipt.get("manifest_sha256") != manifest.manifest_sha256:
            return None
        banks = receipt.get("banks")
        if not isinstance(banks, list) or len(banks) != len(manifest.sidecar.parts):
            return None
        if not all(
            _receipt_bank_matches(
                root,
                bank,
                relative_name=part.file,
                expected_size=part.size,
                expected_sha256=part.sha256,
            )
            for bank, part in zip(banks, manifest.sidecar.parts, strict=True)
        ):
            return None
        return receipt
    except (OSError, TypeError, ValueError):
        return None


def admit_expert_artifact(
    artifact_root: Path | str,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    receipt_root: Path | str | None = None,
    trusted_bank_digests: Mapping[str, TrustedFileDigest] | None = None,
) -> dict[str, Any]:
    """Verify an authoritative expert artifact and atomically publish its receipt."""

    root = _resolved_artifact_root(artifact_root)
    resolved_receipt_root = _resolve_receipt_root(receipt_root)
    path = _receipt_path(root, resolved_receipt_root)
    manifest = _authoritative_manifest(root)
    trusted = trusted_bank_digests or {}
    banks = [
        _inspect_bank(
            root,
            relative_name=part.file,
            expected_size=part.size,
            expected_sha256=part.sha256,
            trusted=trusted.get(part.file),
        )
        for part in manifest.sidecar.parts
    ]
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "artifact_root": str(root),
        "repo_id": repo_id,
        "revision": revision,
        "manifest_sha256": manifest.manifest_sha256,
        "banks": banks,
        "receipt_path": str(path),
    }
    _write_receipt(path, receipt)
    return receipt


def ensure_expert_admitted(
    artifact_root: Path | str,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    receipt_root: Path | str | None = None,
) -> dict[str, Any]:
    """Reuse a valid receipt, or verify the expert banks once and create one."""

    resolved_receipt_root = _resolve_receipt_root(receipt_root)
    receipt = load_valid_admission_receipt(
        artifact_root,
        revision=revision,
        receipt_root=resolved_receipt_root,
    )
    if (
        receipt is not None
        and (repo_id is None or receipt.get("repo_id") == repo_id)
    ):
        _LOGGER.info("expert admission receipt reused; bank SHA-256 skipped")
        return receipt
    _LOGGER.info("expert admission receipt missing or stale; verifying bank")
    return admit_expert_artifact(
        artifact_root,
        repo_id=repo_id,
        revision=revision,
        receipt_root=resolved_receipt_root,
    )
