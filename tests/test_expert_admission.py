from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from mtplx.expert_admission import (
    TrustedFileDigest,
    admit_expert_artifact,
    admission_receipt_path,
    ensure_expert_admitted,
    load_valid_admission_receipt,
)
from mtplx.expert_manifest import (
    ExpertManifestError,
    save_expert_manifest,
)
from mtplx.expert_streaming_models import MODEL_SPECS

from test_expert_manifest import _make_authoritative_checkpoint
from test_hf_loader_expert_artifacts import _split_authoritative_sidecar


@pytest.fixture
def authoritative_expert_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "model"
    spec, manifest = _make_authoritative_checkpoint(root)
    saved = save_expert_manifest(manifest, root / "expert-manifest.json")
    monkeypatch.setitem(MODEL_SPECS, spec.key, spec)
    assert saved.sidecar is not None
    return root, saved, root / saved.sidecar.file


def test_admission_writes_revision_bound_external_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, _bank = authoritative_expert_artifact
    receipt = admit_expert_artifact(
        root,
        repo_id="owner/model",
        revision="a" * 40,
        receipt_root=tmp_path / "receipts",
    )

    assert receipt["revision"] == "a" * 40
    assert receipt["manifest_sha256"] == manifest.manifest_sha256
    assert receipt["banks"][0]["sha256"] == manifest.sidecar.sha256
    assert Path(receipt["receipt_path"]).parent == tmp_path / "receipts"
    assert not (root / ".mtplx_admission.json").exists()
    assert (os.stat(receipt["receipt_path"]).st_mode & 0o777) == 0o600


def test_same_size_bank_mutation_invalidates_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, _manifest, bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    admit_expert_artifact(
        root,
        repo_id="owner/model",
        revision="b" * 40,
        receipt_root=receipt_root,
    )
    original = bank.read_bytes()
    bank.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))

    assert load_valid_admission_receipt(root, receipt_root=receipt_root) is None


def test_truncated_bank_invalidates_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, _manifest, bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    admit_expert_artifact(
        root,
        revision="b" * 40,
        receipt_root=receipt_root,
    )
    bank.write_bytes(bank.read_bytes()[:-1])

    assert load_valid_admission_receipt(root, receipt_root=receipt_root) is None


def test_wrong_manifest_digest_invalidates_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, _manifest, _bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    admit_expert_artifact(
        root,
        revision="c" * 40,
        receipt_root=receipt_root,
    )
    manifest_path = root / "expert-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_valid_admission_receipt(root, receipt_root=receipt_root) is None


def test_wrong_revision_invalidates_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, _manifest, _bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    admit_expert_artifact(
        root,
        revision="d" * 40,
        receipt_root=receipt_root,
    )

    assert (
        load_valid_admission_receipt(
            root,
            revision="e" * 40,
            receipt_root=receipt_root,
        )
        is None
    )


def test_unsafe_sidecar_member_is_rejected(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, _manifest, _bank = authoritative_expert_artifact
    manifest_path = root / "expert-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sidecar"]["file"] = "../experts.bin"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExpertManifestError, match="unsafe sidecar file"):
        admit_expert_artifact(root, receipt_root=tmp_path / "receipts")


def test_missing_resident_shard_is_rejected(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, _bank = authoritative_expert_artifact
    resident = next(shard for shard in manifest.shards if shard.kind == "safetensors")
    (root / resident.name).unlink()

    with pytest.raises(ExpertManifestError, match="resident"):
        admit_expert_artifact(root, receipt_root=tmp_path / "receipts")


def test_interrupted_atomic_write_leaves_no_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _manifest, _bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    receipt_path = admission_receipt_path(root, receipt_root=receipt_root)

    def fail_replace(_source, _destination):
        raise OSError("interrupted")

    monkeypatch.setattr("mtplx.expert_admission.os.replace", fail_replace)

    with pytest.raises(OSError, match="interrupted"):
        admit_expert_artifact(root, receipt_root=receipt_root)

    assert not receipt_path.exists()
    assert list(receipt_root.glob("*.tmp")) == []


def test_valid_receipt_skips_bank_hasher(
    authoritative_expert_artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, _manifest, _bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    first = admit_expert_artifact(
        root,
        repo_id="owner/model",
        revision="f" * 40,
        receipt_root=receipt_root,
    )

    def fail_hash(_descriptor: int) -> str:
        raise AssertionError("valid receipt must not rehash expert banks")

    monkeypatch.setattr("mtplx.expert_admission._hash_bank_descriptor", fail_hash)
    with caplog.at_level(logging.INFO, logger="mtplx.expert_admission"):
        reused = ensure_expert_admitted(
            root,
            repo_id="owner/model",
            revision="f" * 40,
            receipt_root=receipt_root,
        )

    assert reused == first
    assert (
        "expert admission receipt reused; bank SHA-256 skipped"
        in caplog.messages
    )


def test_stale_downloader_identity_forces_bank_hash(
    authoritative_expert_artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, bank = authoritative_expert_artifact
    metadata = bank.stat()
    trusted = TrustedFileDigest(
        sha256=manifest.sidecar.sha256,
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_size=metadata.st_size,
        st_mtime_ns=metadata.st_mtime_ns - 1,
    )
    calls = 0
    import mtplx.expert_admission as expert_admission

    real_hash = expert_admission._hash_bank_descriptor

    def count_hash(descriptor: int) -> str:
        nonlocal calls
        calls += 1
        return real_hash(descriptor)

    monkeypatch.setattr(expert_admission, "_hash_bank_descriptor", count_hash)

    admit_expert_artifact(
        root,
        receipt_root=tmp_path / "receipts",
        trusted_bank_digests={manifest.sidecar.file: trusted},
    )

    assert calls == 1


def test_downloader_digest_with_exact_identity_skips_bank_hash(
    authoritative_expert_artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, bank = authoritative_expert_artifact
    metadata = bank.stat()
    trusted = TrustedFileDigest(
        sha256=manifest.sidecar.sha256,
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_size=metadata.st_size,
        st_mtime_ns=metadata.st_mtime_ns,
    )

    def fail_hash(_descriptor: int) -> str:
        raise AssertionError("matching downloader digest should be trusted")

    monkeypatch.setattr("mtplx.expert_admission._hash_bank_descriptor", fail_hash)

    receipt = admit_expert_artifact(
        root,
        receipt_root=tmp_path / "receipts",
        trusted_bank_digests={manifest.sidecar.file: trusted},
    )

    assert receipt["banks"][0]["sha256"] == manifest.sidecar.sha256


def test_admission_covers_every_sidecar_part(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, _bank = authoritative_expert_artifact
    split = _split_authoritative_sidecar(root, manifest)
    split = save_expert_manifest(split, root / "expert-manifest.json")

    receipt = admit_expert_artifact(
        root,
        revision="1" * 40,
        receipt_root=tmp_path / "receipts",
    )

    assert split.sidecar is not None
    assert [bank["file"] for bank in receipt["banks"]] == [
        part.file for part in split.sidecar.parts
    ]
