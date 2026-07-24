"""Authoritative completeness checks for published streamed-expert artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_manifest import (
    EMPTY_SHA256,
    ExpertManifest,
    ShardInfo,
    SidecarInfo,
    SidecarPart,
    save_expert_manifest,
    validate_expert_manifest_spec,
)
from mtplx.expert_streaming_models import MODEL_SPECS, ExpertStreamingModelSpec
from mtplx.hf_loader import (
    MAX_RUNTIME_CONTRACT_BYTES,
    expert_artifact_status,
)

from test_expert_manifest import _make_authoritative_checkpoint


def _install_authoritative_model(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ExpertStreamingModelSpec, ExpertManifest]:
    root.parent.mkdir(parents=True, exist_ok=True)
    spec, manifest = _make_authoritative_checkpoint(root)
    saved = save_expert_manifest(manifest, root / "expert-manifest.json")
    monkeypatch.setitem(MODEL_SPECS, spec.key, spec)
    return spec, saved


@pytest.fixture
def authoritative_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, ExpertStreamingModelSpec, ExpertManifest]:
    root = tmp_path / "model"
    spec, manifest = _install_authoritative_model(root, monkeypatch)
    return root, spec, manifest


def test_authoritative_streamed_artifact_is_complete(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
) -> None:
    root, _spec, manifest = authoritative_model

    status = expert_artifact_status(root)

    assert manifest.sidecar is not None
    assert status["streamed_experts"] is True
    assert status["ok"] is True
    assert status["sidecar_file"] == manifest.sidecar.file
    assert status["expected_bytes"] == manifest.sidecar.size
    assert status["actual_bytes"] == manifest.sidecar.size
    assert status["reason"] is None


def test_unknown_manifest_model_key_fails_closed(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
) -> None:
    root, _spec, manifest = authoritative_model
    save_expert_manifest(
        replace(manifest, model_key="unknown-streaming-model"),
        root / "expert-manifest.json",
    )

    status = expert_artifact_status(root)

    assert status["ok"] is False
    assert "unknown model" in status["reason"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (
            lambda manifest: replace(manifest, source_repo="attacker/wrong-source"),
            "source identity",
        ),
        (
            lambda manifest: replace(manifest, records=manifest.records[:-1]),
            "record",
        ),
        (
            lambda manifest: replace(
                manifest,
                resident_tensors=(
                    replace(
                        manifest.resident_tensors[0],
                        tensor="model.layers.1.mlp.experts.0.gate_proj.weight",
                    ),
                ),
            ),
            "resident",
        ),
        (
            lambda manifest: replace(
                manifest,
                shards=tuple(
                    shard for shard in manifest.shards if shard.kind != "sidecar"
                ),
            ),
            "authoritative",
        ),
        (
            lambda manifest: replace(
                manifest,
                records=(
                    replace(
                        manifest.records[0],
                        sidecar_offset=manifest.records[0].sidecar_offset + 1,
                    ),
                    *manifest.records[1:],
                ),
            ),
            "aligned",
        ),
    ),
    ids=("identity", "records", "resident", "shards", "layout"),
)
def test_manifest_identity_and_layout_mismatches_fail_closed(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
    mutation,
    reason: str,
) -> None:
    root, _spec, manifest = authoritative_model
    save_expert_manifest(mutation(manifest), root / "expert-manifest.json")

    status = expert_artifact_status(root)

    assert status["ok"] is False
    assert reason in status["reason"]


def test_sidecar_bank_must_meet_authoritative_part_size(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
) -> None:
    root, _spec, manifest = authoritative_model
    assert manifest.sidecar is not None
    bank = root / manifest.sidecar.file
    bank.write_bytes(bank.read_bytes()[:-1])

    status = expert_artifact_status(root)

    assert status["ok"] is False
    assert "truncated" in status["reason"]


def _split_authoritative_sidecar(
    root: Path,
    manifest: ExpertManifest,
) -> ExpertManifest:
    assert manifest.sidecar is not None
    blob = (root / manifest.sidecar.file).read_bytes()
    parts: list[SidecarPart] = []
    records = []
    sidecar_shards: list[ShardInfo] = []
    for index, record in enumerate(manifest.records):
        assert record.sidecar_offset is not None
        assert record.sidecar_length is not None
        payload = blob[
            record.sidecar_offset : record.sidecar_offset + record.sidecar_length
        ]
        name = f"experts-{index + 1:05d}-of-{len(manifest.records):05d}.bin"
        (root / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        parts.append(SidecarPart(file=name, size=len(payload), sha256=digest))
        sidecar_shards.append(
            ShardInfo(
                name=name,
                size=len(payload),
                header_bytes=0,
                header_sha256=EMPTY_SHA256,
                sha256=digest,
                kind="sidecar",
            )
        )
        cursor = 0
        segments = []
        for segment in record.segments:
            segments.append(replace(segment, shard=name, offset=cursor))
            cursor += segment.length
        records.append(
            replace(
                record,
                segments=tuple(segments),
                sidecar_offset=0,
                part=index,
            )
        )
    (root / manifest.sidecar.file).unlink()
    resident_shards = tuple(
        shard for shard in manifest.shards if shard.kind == "safetensors"
    )
    return replace(
        manifest,
        sidecar=SidecarInfo(
            alignment=manifest.sidecar.alignment,
            parts=tuple(parts),
        ),
        shards=(*resident_shards, *sidecar_shards),
        records=tuple(records),
        manifest_sha256=None,
    ).with_digest()


def test_every_authoritative_sidecar_part_must_exist(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
) -> None:
    root, spec, manifest = authoritative_model
    split = _split_authoritative_sidecar(root, manifest)
    validate_expert_manifest_spec(split, spec)
    save_expert_manifest(split, root / "expert-manifest.json")
    assert split.sidecar is not None

    complete = expert_artifact_status(root)
    assert complete["ok"] is True
    assert complete["expected_bytes"] == sum(
        part.size for part in split.sidecar.parts
    )

    (root / split.sidecar.parts[-1].file).unlink()

    status = expert_artifact_status(root)

    assert status["ok"] is False
    assert split.sidecar.parts[-1].file in status["reason"]
    assert "missing" in status["reason"]


def test_repository_local_hf_blob_symlink_is_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models--owner--repo" / "snapshots" / "revision"
    _spec, manifest = _install_authoritative_model(root, monkeypatch)
    assert manifest.sidecar is not None
    bank = root / manifest.sidecar.file
    blob_root = root.parents[1] / "blobs"
    blob_root.mkdir()
    blob = blob_root / "bank-digest"
    bank.replace(blob)
    bank.symlink_to(Path("..") / ".." / "blobs" / blob.name)

    status = expert_artifact_status(root)

    assert status["ok"] is True
    assert status["actual_bytes"] == manifest.sidecar.size


def test_sidecar_symlink_outside_artifact_is_rejected(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
    tmp_path: Path,
) -> None:
    root, _spec, manifest = authoritative_model
    assert manifest.sidecar is not None
    bank = root / manifest.sidecar.file
    outside = tmp_path / "outside.bin"
    outside.write_bytes(bank.read_bytes())
    bank.unlink()
    bank.symlink_to(outside)

    status = expert_artifact_status(root)

    assert status["ok"] is False
    assert "escapes root" in status["reason"]


def test_manifest_symlink_is_not_followed(
    authoritative_model: tuple[Path, ExpertStreamingModelSpec, ExpertManifest],
    tmp_path: Path,
) -> None:
    root, _spec, _manifest = authoritative_model
    manifest_path = root / "expert-manifest.json"
    outside = tmp_path / "outside-manifest.json"
    manifest_path.replace(outside)
    manifest_path.symlink_to(outside)

    status = expert_artifact_status(root)

    assert status["streamed_experts"] is True
    assert status["ok"] is False
    assert "could not read" in status["reason"]


def test_runtime_contract_symlink_is_not_followed(tmp_path: Path) -> None:
    model = tmp_path / "plain"
    model.mkdir()
    outside = tmp_path / "outside-runtime.json"
    outside.write_text(
        '{"expert_manifest_file":"expert-manifest.json"}',
        encoding="utf-8",
    )
    (model / "mtplx_runtime.json").symlink_to(outside)

    status = expert_artifact_status(model)

    assert status["streamed_experts"] is False
    assert status["ok"] is True


def test_runtime_contract_read_is_bounded(tmp_path: Path) -> None:
    model = tmp_path / "plain"
    model.mkdir()
    (model / "mtplx_runtime.json").write_bytes(
        b'{"expert_streaming":"' + b"x" * MAX_RUNTIME_CONTRACT_BYTES + b'"}'
    )

    status = expert_artifact_status(model)

    assert status["streamed_experts"] is False
    assert status["ok"] is True


def test_plain_model_is_unaffected(tmp_path: Path) -> None:
    model = tmp_path / "plain"
    model.mkdir()

    status = expert_artifact_status(model)

    assert status["streamed_experts"] is False
    assert status["ok"] is True
