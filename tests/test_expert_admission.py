from __future__ import annotations

import json
import logging
import os
import time
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
from mtplx.expert_io import (
    ADMITTED_DESCRIPTOR_SECURITY_BOUNDARY,
    ExpertIOError,
    ExpertIOIntegrityError,
    PositionalExpertReader,
)
from mtplx.expert_streaming_models import MODEL_SPECS

import mtplx.runtime as runtime_module

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
    root, manifest, bank = authoritative_expert_artifact
    receipt = admit_expert_artifact(
        root,
        repo_id="owner/model",
        revision="a" * 40,
        receipt_root=tmp_path / "receipts",
    )

    assert receipt["revision"] == "a" * 40
    assert receipt["manifest_sha256"] == manifest.manifest_sha256
    assert receipt["banks"][0]["sha256"] == manifest.sidecar.sha256
    assert receipt["banks"][0]["st_ctime_ns"] == bank.stat().st_ctime_ns
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


def test_mtime_restored_same_inode_mutation_invalidates_receipt(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, _manifest, bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    admit_expert_artifact(
        root,
        revision="2" * 40,
        receipt_root=receipt_root,
    )
    before = bank.stat()
    original = bank.read_bytes()
    time.sleep(0.002)
    with bank.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[-1] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(
        bank,
        ns=(before.st_atime_ns, before.st_mtime_ns),
    )

    after = bank.stat()
    assert after.st_ino == before.st_ino
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns
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
        st_ctime_ns=metadata.st_ctime_ns,
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
        st_ctime_ns=metadata.st_ctime_ns,
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


def test_reader_rejects_bank_replaced_after_admission(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, bank = authoritative_expert_artifact
    receipt = admit_expert_artifact(
        root,
        revision="3" * 40,
        receipt_root=tmp_path / "receipts",
    )
    original = bank.read_bytes()
    bank.replace(root / "original-experts.bin")
    bank.write_bytes(original)

    with pytest.raises(ExpertIOError, match="admission receipt identity"):
        PositionalExpertReader(
            root,
            use_native=False,
            expert_admission_receipt=receipt,
        )


def test_reader_uses_pinned_bank_after_pathname_replacement(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, bank = authoritative_expert_artifact
    receipt = admit_expert_artifact(
        root,
        revision="4" * 40,
        receipt_root=tmp_path / "receipts",
    )
    reader = PositionalExpertReader(
        root,
        use_native=False,
        expert_admission_receipt=receipt,
    )
    try:
        original = bank.read_bytes()
        replacement = root / "replacement.bin"
        replacement.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
        os.replace(replacement, bank)
        record = manifest.records[0]
        destination = bytearray(record.logical_bytes)

        digest = reader.read_record_into(
            manifest,
            record,
            destination,
            verify_hash=True,
        )

        assert digest == record.sha256
    finally:
        reader.close()


def test_pinned_reader_hash_check_detects_same_inode_mutation(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, bank = authoritative_expert_artifact
    receipt = admit_expert_artifact(
        root,
        revision="6" * 40,
        receipt_root=tmp_path / "receipts",
    )
    reader = PositionalExpertReader(
        root,
        use_native=False,
        expert_admission_receipt=receipt,
    )
    try:
        assert "outside the local artifact threat model" in (
            ADMITTED_DESCRIPTOR_SECURITY_BOUNDARY
        )
        record = manifest.records[0]
        assert manifest.sidecar is not None
        part = manifest.sidecar.parts[record.part]
        offset = part.data_start + int(record.sidecar_offset or 0)
        with bank.open("r+b") as handle:
            handle.seek(offset)
            original = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([original[0] ^ 0xFF]))
            handle.flush()
            os.fsync(handle.fileno())

        with pytest.raises(ExpertIOIntegrityError, match="hash mismatch"):
            reader.read_record_into(
                manifest,
                record,
                bytearray(record.logical_bytes),
                verify_hash=True,
            )
    finally:
        reader.close()


def test_multipart_admitted_descriptors_are_never_evicted_and_close(
    authoritative_expert_artifact,
    tmp_path: Path,
) -> None:
    root, manifest, _bank = authoritative_expert_artifact
    split = _split_authoritative_sidecar(root, manifest)
    split = save_expert_manifest(split, root / "expert-manifest.json")
    receipt = admit_expert_artifact(
        root,
        revision="5" * 40,
        receipt_root=tmp_path / "receipts",
    )
    assert split.sidecar is not None
    reader = PositionalExpertReader(
        root,
        max_open_files=1,
        use_native=False,
        expert_admission_receipt=receipt,
    )
    pinned: dict[str, int] = {}
    for part in split.sidecar.parts:
        with reader._lease(part.file) as descriptor:
            pinned[part.file] = descriptor
    # Exercise the ordinary one-entry LRU; admitted descriptors are outside it.
    resident = next(
        shard.name for shard in split.shards if shard.kind == "safetensors"
    )
    with reader._lease(resident):
        pass
    for part in split.sidecar.parts:
        with reader._lease(part.file) as descriptor:
            assert descriptor == pinned[part.file]

    reader.close()

    for descriptor in pinned.values():
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_runtime_load_forwards_expert_admission_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {"schema": 1, "banks": []}
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_load_impl(*_args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(runtime_module, "_load_impl", fake_load_impl)

    result = runtime_module.load(
        tmp_path,
        expert_admission_receipt=receipt,
    )

    assert result is sentinel
    assert captured["expert_admission_receipt"] is receipt
