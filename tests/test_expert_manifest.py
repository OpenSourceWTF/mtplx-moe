from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    build_expert_manifest,
    build_expert_sidecar,
    load_expert_manifest,
    read_expert_record,
    save_expert_manifest,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import ExpertStreamingModelSpec


COMPONENTS = (
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)


def _tiny_spec(*, resident_bytes: int = 8) -> ExpertStreamingModelSpec:
    record_bytes = 6_912
    return ExpertStreamingModelSpec(
        key="tiny-q4",
        display_name="Tiny affine Q4",
        source_model="test/tiny",
        source_revision="source-revision",
        quant_model="test/tiny-q4",
        quant_revision="quant-revision",
        total_tensor_bytes=2 * record_bytes + resident_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="bfloat16",
        router_matmul_dtype="float32",
        router_bytes=0,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )


def _component_info(component: str, *, stacked: bool) -> tuple[str, list[int], int]:
    leaf = component.rsplit(".", 1)[1]
    if leaf == "weight":
        dtype = "U32"
        per_expert_shape = [64, 8]
        per_expert_bytes = 2_048
    else:
        dtype = "BF16"
        per_expert_shape = [64, 1]
        per_expert_bytes = 128
    shape = [2, *per_expert_shape] if stacked else per_expert_shape
    return dtype, shape, per_expert_bytes


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _make_checkpoint(
    root: Path, *, numbered: bool = False
) -> tuple[ExpertStreamingModelSpec, dict[int, bytes]]:
    root.mkdir()
    spec = _tiny_spec()
    shards: list[list[tuple[str, str, list[int], bytes]]] = [[], []]
    expected: dict[int, bytearray] = {0: bytearray(), 1: bytearray()}
    weight_map: dict[str, str] = {}
    for component_index, component in enumerate(COMPONENTS):
        dtype, shape, per_expert_bytes = _component_info(
            component, stacked=not numbered
        )
        projection, leaf = component.split(".")
        if numbered:
            for expert in range(2):
                name = f"model.layers.1.mlp.experts.{expert}.{projection}.{leaf}"
                raw = bytes([component_index * 2 + expert + 1]) * per_expert_bytes
                shard_index = (component_index + expert) % 2
                shards[shard_index].append((name, dtype, shape, raw))
                weight_map[name] = f"model-{shard_index + 1:05d}-of-00002.safetensors"
                expected[expert].extend(raw)
        else:
            name = f"model.layers.1.mlp.switch_mlp.{component}"
            raw_parts = [
                bytes([component_index * 2 + expert + 1]) * per_expert_bytes
                for expert in range(2)
            ]
            raw = b"".join(raw_parts)
            shard_index = component_index % 2
            shards[shard_index].append((name, dtype, shape, raw))
            weight_map[name] = f"model-{shard_index + 1:05d}-of-00002.safetensors"
            for expert, part in enumerate(raw_parts):
                expected[expert].extend(part)
    resident_name = "model.embed_tokens.weight"
    resident_raw = bytes(range(8))
    shards[0].append((resident_name, "F32", [2], resident_raw))
    weight_map[resident_name] = "model-00001-of-00002.safetensors"
    for index, tensors in enumerate(shards, 1):
        _write_safetensors(root / f"model-{index:05d}-of-00002.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return spec, {expert: bytes(value) for expert, value in expected.items()}


@pytest.mark.parametrize("numbered", [False, True])
def test_build_manifest_for_stacked_and_numbered_experts(
    tmp_path: Path,
    numbered: bool,
) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root, numbered=numbered)

    manifest = build_expert_manifest(
        root,
        spec,
        hash_records=True,
        hash_shards=True,
    )

    assert manifest.artifact_tensor_bytes == spec.total_tensor_bytes
    assert manifest.resident_tensor_bytes == 8
    assert manifest.routed_expert_bytes == spec.routed_expert_bytes
    assert len(manifest.shards) == 2
    assert len(manifest.resident_tensors) == 1
    assert len(manifest.records) == 2
    assert all(shard.sha256 for shard in manifest.shards)
    for expert in range(2):
        record = manifest.record(1, expert)
        assert tuple(segment.component for segment in record.segments) == COMPONENTS
        assert record.logical_bytes == spec.expert_record_bytes
        assert record.sha256 == hashlib.sha256(expected[expert]).hexdigest()
        assert read_expert_record(manifest, root, 1, expert) == expected[expert]

    report = verify_expert_manifest(
        manifest,
        root,
        verify_records=True,
        verify_shard_hashes=True,
    )
    assert report["valid"] is True
    assert report["checked_records"] == 2


def test_manifest_roundtrip_digest_and_unknown_fields_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    path = root / "expert-manifest.json"

    saved = save_expert_manifest(manifest, path)
    loaded = load_expert_manifest(path)

    assert loaded == saved
    tampered = loaded.to_dict()
    tampered["model_key"] = "wrong"
    with pytest.raises(ExpertManifestError, match="digest mismatch"):
        ExpertManifest.from_dict(tampered)
    unknown = loaded.to_dict()
    unknown["surprise"] = True
    with pytest.raises(ExpertManifestError, match="unknown keys"):
        ExpertManifest.from_dict(unknown, verify_digest=False)
    escaped = loaded.to_dict()
    escaped["records"][0]["segments"][0]["shard"] = "../escape"
    with pytest.raises(ExpertManifestError, match="unsafe"):
        ExpertManifest.from_dict(escaped, verify_digest=False)


def test_aligned_sidecar_is_readable_and_verifiable(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)

    updated = build_expert_sidecar(manifest, root, root / "experts.bin")

    assert updated.sidecar is not None
    assert updated.sidecar.file == "experts.bin"
    assert updated.sidecar.alignment == 16_384
    assert updated.records[0].sidecar_offset == 0
    assert updated.records[1].sidecar_offset == 16_384
    assert updated.sidecar.size == 16_384 + spec.expert_record_bytes
    assert read_expert_record(updated, root, 1, 0) == expected[0]
    assert read_expert_record(updated, root, 1, 1) == expected[1]
    report = verify_expert_manifest(updated, root, verify_sidecar_hash=True)
    assert report["sidecar_verified"] is True


def test_corrupt_payload_and_truncated_sidecar_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    record = manifest.records[0]
    segment = record.segments[0]
    shard = root / segment.shard
    fd = os.open(shard, os.O_RDWR)
    try:
        original = os.pread(fd, 1, segment.offset)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), segment.offset)
    finally:
        os.close(fd)
    with pytest.raises(ExpertManifestError, match="record hash mismatch"):
        read_expert_record(manifest, root, record.layer, record.expert)

    # Rebuild a clean model before testing the independent sidecar failure.
    clean_root = tmp_path / "clean"
    clean_spec, _ = _make_checkpoint(clean_root)
    clean_manifest = build_expert_manifest(clean_root, clean_spec)
    updated = build_expert_sidecar(
        clean_manifest, clean_root, clean_root / "experts.bin"
    )
    assert updated.sidecar is not None
    (clean_root / "experts.bin").write_bytes(b"short")
    with pytest.raises(ExpertManifestError, match="sidecar size mismatch"):
        verify_expert_manifest(updated, clean_root)


def test_missing_component_and_index_mismatch_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing_key = "model.layers.1.mlp.switch_mlp.down_proj.biases"
    del index["weight_map"][missing_key]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ExpertManifestError, match="wrong shard|mismatch"):
        build_expert_manifest(root, spec)


def test_sidecar_must_stay_inside_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)

    with pytest.raises(ExpertManifestError, match="inside the artifact root"):
        build_expert_sidecar(manifest, root, tmp_path / "outside.bin")


def test_manifest_builder_can_skip_pinned_total_only_for_fixtures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    wrong_total = replace(spec, total_tensor_bytes=spec.total_tensor_bytes + 4)

    with pytest.raises(ExpertManifestError, match="pinned"):
        build_expert_manifest(root, wrong_total)
    manifest = build_expert_manifest(
        root,
        wrong_total,
        require_pinned_tensor_bytes=False,
    )
    assert manifest.artifact_tensor_bytes == spec.total_tensor_bytes
