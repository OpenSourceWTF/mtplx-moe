from __future__ import annotations

import filecmp
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_manifest import (
    EMPTY_SHA256,
    MAX_INDEX_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_MANIFEST_RECORDS,
    MAX_MANIFEST_RESIDENT_TENSORS,
    MAX_MANIFEST_SHARDS,
    ExpertManifest,
    ExpertManifestError,
    ResidentTensor,
    ShardInfo,
    _classify_expert_name,
    build_expert_manifest,
    build_expert_sidecar,
    load_expert_manifest,
    make_sidecar_authoritative,
    read_expert_record,
    resolve_artifact_member,
    save_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import ExpertStreamingModelSpec, get_model_spec
from mtplx.qwen4_oq4 import externalize_source_residents

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

PINNED_HY3_EXPERT_ONLY_Q4_ROOT = Path(
    "/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q4"
)
PINNED_HY3_EXPERT_ONLY_Q4_FILE_SHA256 = (
    "e7fcfd6c69486456af4261d908d95f8a84a391d6a273ff1cff02a15f73fac92d"
)
PINNED_HY3_EXPERT_ONLY_Q4_CANONICAL_SHA256 = (
    "507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43"
)


def _tiny_spec(
    *,
    resident_bytes: int = 8,
    bits: int = 4,
    key: str | None = None,
) -> ExpertStreamingModelSpec:
    record_bytes = 3 * (64 * 64 * bits // 8 + 2 * 64 * 2)
    return ExpertStreamingModelSpec(
        key=key or f"tiny-q{bits}",
        display_name=f"Tiny affine Q{bits}",
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
        quant_bits=bits,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="bfloat16",
        router_matmul_dtype="float32",
        router_bytes=0,
        kv_bytes_per_token=0,
        mtp_layer_index=80 if key and key.startswith("hy3-") else 2,
        mtp_included=False,
    )


def test_qwen4_native_mtp_experts_map_to_synthetic_streaming_layer() -> None:
    tiny = _tiny_spec(key="qwen38-flash-next-q4")
    spec = replace(
        tiny,
        total_tensor_bytes=49 * 2 * tiny.expert_record_bytes + 8,
        total_layers=49,
        routed_layer_start=0,
        routed_layer_count=49,
        mtp_layer_index=None,
        mtp_included=True,
    )
    assert _classify_expert_name(
        "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
        spec,
    ) == (48, None, "gate_proj.weight")
    assert (
        _classify_expert_name(
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            spec,
        )
        == (0, None, "gate_proj.weight")
    )
    assert (
        _classify_expert_name(
            "language_model.model.layers.48.mlp.switch_mlp.gate_proj.weight",
            spec,
        )
        is None
    )


def test_qwen4_source_native_manifest_pins_published_oq4_identity(
    tmp_path: Path,
) -> None:
    spec, _expected = _make_checkpoint(tmp_path / "source")
    manifest = build_expert_manifest(tmp_path / "source", spec)
    qwen_spec = replace(
        spec,
        key="qwen38-flash-next-q4",
        source_model="Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP",
        source_revision="published-revision",
        quant_model="user/Qwen3.8-Flash-Next-MTPLX-oQ4-MTP",
        quant_revision="upload-revision",
    )
    source_native = replace(
        manifest,
        model_key=qwen_spec.key,
        source_repo=qwen_spec.source_model,
        source_revision=qwen_spec.source_revision,
        manifest_sha256=None,
    ).with_digest()

    validate_expert_manifest_spec(source_native, qwen_spec)


def test_source_native_external_residents_are_removed_from_runtime_bytes(
    tmp_path: Path,
) -> None:
    spec, _expected = _make_checkpoint(tmp_path / "source")
    manifest = build_expert_manifest(tmp_path / "source", spec)
    resident = manifest.resident_tensors[0]
    external = replace(
        resident,
        tensor="external.ngram.weight",
        length=4,
        dtype="U8",
        shape=(4,),
    )
    kept = replace(
        resident,
        tensor="model.embed_tokens.weight",
        offset=resident.offset + 4,
        length=4,
        dtype="U8",
        shape=(4,),
    )
    source = replace(
        manifest,
        resident_tensors=(external, kept),
        manifest_sha256=None,
    ).with_digest()
    target_spec = replace(
        spec,
        total_tensor_bytes=spec.total_tensor_bytes - 4,
    )

    result = externalize_source_residents(
        source,
        external_tensor_names=frozenset({external.tensor}),
        external_bytes=4,
        target_spec=target_spec,
    )

    assert result.artifact_tensor_bytes == target_spec.total_tensor_bytes
    assert result.resident_tensor_bytes == target_spec.resident_bytes
    assert result.resident_tensors == (kept,)


def _component_info(
    component: str, *, stacked: bool, bits: int
) -> tuple[str, list[int], int]:
    leaf = component.rsplit(".", 1)[1]
    if leaf == "weight":
        dtype = "U32"
        per_expert_shape = [64, 64 * bits // 32]
        per_expert_bytes = 64 * 64 * bits // 8
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
    root: Path,
    *,
    numbered: bool = False,
    bits: int = 4,
    key: str | None = None,
    separate_resident: bool = False,
) -> tuple[ExpertStreamingModelSpec, dict[int, bytes]]:
    root.mkdir()
    spec = _tiny_spec(bits=bits, key=key)
    shard_count = 3 if separate_resident else 2
    shards: list[list[tuple[str, str, list[int], bytes]]] = [
        [] for _ in range(shard_count)
    ]
    expected: dict[int, bytearray] = {0: bytearray(), 1: bytearray()}
    weight_map: dict[str, str] = {}
    for component_index, component in enumerate(COMPONENTS):
        dtype, shape, per_expert_bytes = _component_info(
            component, stacked=not numbered, bits=bits
        )
        projection, leaf = component.split(".")
        if numbered:
            for expert in range(2):
                name = f"model.layers.1.mlp.experts.{expert}.{projection}.{leaf}"
                raw = bytes([component_index * 2 + expert + 1]) * per_expert_bytes
                shard_index = (component_index + expert) % 2
                shards[shard_index].append((name, dtype, shape, raw))
                weight_map[name] = (
                    f"model-{shard_index + 1:05d}-of-{shard_count:05d}.safetensors"
                )
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
            weight_map[name] = (
                f"model-{shard_index + 1:05d}-of-{shard_count:05d}.safetensors"
            )
            for expert, part in enumerate(raw_parts):
                expected[expert].extend(part)
    resident_name = "model.embed_tokens.weight"
    resident_raw = bytes(range(8))
    resident_shard = 2 if separate_resident else 0
    shards[resident_shard].append((resident_name, "F32", [2], resident_raw))
    weight_map[resident_name] = (
        f"model-{resident_shard + 1:05d}-of-{shard_count:05d}.safetensors"
    )
    for index, tensors in enumerate(shards, 1):
        _write_safetensors(
            root / f"model-{index:05d}-of-{shard_count:05d}.safetensors", tensors
        )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return spec, {expert: bytes(value) for expert, value in expected.items()}


def _make_authoritative_checkpoint(
    root: Path, *, bits: int = 2
) -> tuple[ExpertStreamingModelSpec, ExpertManifest]:
    key = "hy3-expert-q2" if bits == 2 else "hy3-expert-only-q4"
    spec, _expected = _make_checkpoint(
        root,
        bits=bits,
        key=key,
        separate_resident=True,
    )
    manifest = build_expert_manifest(
        root,
        spec,
        hash_records=True,
        hash_shards=True,
    )
    with_sidecar = build_expert_sidecar(manifest, root, root / "experts.bin")
    authoritative = make_sidecar_authoritative(with_sidecar, spec)

    resident_names = {tensor.tensor for tensor in authoritative.resident_tensors}
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"] = {
        tensor: shard
        for tensor, shard in index["weight_map"].items()
        if tensor in resident_names
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")
    retained = {shard.name for shard in authoritative.shards}
    for path in root.glob("model*.safetensors"):
        if path.name not in retained:
            path.unlink()
    return spec, authoritative


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


def test_hugging_face_snapshot_blob_symlink_is_allowed_but_other_escapes_fail(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "models--org--model"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / ("b" * 64)
    blob.write_bytes(b"trusted content-addressed payload")
    member = snapshot / "model-00001-of-00001.safetensors"
    member.symlink_to(Path("..") / ".." / "blobs" / blob.name)

    assert resolve_artifact_member(snapshot, member.name) == blob

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    escaped = snapshot / "escaped.bin"
    escaped.symlink_to(outside)
    with pytest.raises(ExpertManifestError, match="escapes root"):
        resolve_artifact_member(snapshot, escaped.name)

    # A blobs-looking symlink is not trusted outside the exact snapshots/
    # sibling-blobs repository layout.
    ordinary_root = tmp_path / "ordinary"
    ordinary_root.mkdir()
    ordinary = ordinary_root / "model.bin"
    ordinary.symlink_to(blob)
    with pytest.raises(ExpertManifestError, match="escapes root"):
        resolve_artifact_member(ordinary_root, ordinary.name)


def test_manifest_builder_rejects_external_index_symlink(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    index = root / "model.safetensors.index.json"
    external = tmp_path / "external-index.json"
    index.replace(external)
    index.symlink_to(external)

    with pytest.raises(ExpertManifestError, match="escapes root"):
        build_expert_manifest(root, spec)


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


def test_legacy_q4_shard_kind_omission_and_digest_are_frozen(tmp_path: Path) -> None:
    raw_shard = {
        "name": "model-00001-of-00001.safetensors",
        "size": 128,
        "header_bytes": 64,
        "header_sha256": "0" * 64,
        "sha256": "1" * 64,
    }

    shard = ShardInfo.from_dict(raw_shard)

    assert shard.kind == "safetensors"
    assert shard.to_dict() == raw_shard

    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(
        root,
        spec,
        hash_records=False,
        hash_shards=False,
    )
    assert manifest.manifest_sha256 == (
        "f4a6fd4faebd7fe09aa81741d623c1a4d83f0da4f10ad033c92d392d04f6ab5c"
    )
    assert manifest.quant_bits == 4
    assert manifest.quant_group_size == 64
    assert manifest.records[0].logical_bytes == 6_912
    assert all("kind" not in item for item in manifest.to_dict()["shards"])


def test_pinned_current_hy3_q4_manifest_identity_and_reserialization(
    tmp_path: Path,
) -> None:
    root = PINNED_HY3_EXPERT_ONLY_Q4_ROOT
    manifest_path = root / "expert-manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"pinned Hy3 Q4 artifact is unavailable at {root}")

    with manifest_path.open("rb") as handle:
        file_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    assert file_sha256 == PINNED_HY3_EXPERT_ONLY_Q4_FILE_SHA256

    manifest = load_expert_manifest(manifest_path)

    assert manifest.manifest_sha256 == PINNED_HY3_EXPERT_ONLY_Q4_CANONICAL_SHA256
    assert manifest.model_key == "hy3-expert-only-q4"
    assert manifest.quant_bits == 4
    assert manifest.quant_group_size == 64
    assert len(manifest.records) == 15_168
    assert len(manifest.shards) == 52
    assert len(manifest.resident_tensors) == 1_041
    assert all(record.logical_bytes == 10_616_832 for record in manifest.records)
    assert all(shard.kind == "safetensors" for shard in manifest.shards)
    assert all("kind" not in shard for shard in manifest.to_dict()["shards"])

    validate_expert_manifest_spec(
        manifest,
        get_model_spec("hy3-expert-only-q4"),
    )
    report = verify_expert_manifest(manifest, root)
    assert report["valid"] is True
    assert report["checked_shards"] == 52

    reserialized_path = tmp_path / "expert-manifest.json"
    saved = save_expert_manifest(manifest, reserialized_path)
    assert saved == manifest
    assert filecmp.cmp(manifest_path, reserialized_path, shallow=False)
    with reserialized_path.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == PINNED_HY3_EXPERT_ONLY_Q4_FILE_SHA256
        )


def test_affine_q2_descriptor_geometry_roundtrips_and_validates(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root, bits=2, key="hy3-expert-q2")

    manifest = build_expert_manifest(
        root,
        spec,
        hash_records=True,
        hash_shards=True,
    )
    validate_expert_manifest_spec(manifest, spec)
    saved = save_expert_manifest(manifest, root / "expert-manifest.json")
    loaded = load_expert_manifest(root / "expert-manifest.json")

    assert loaded == saved
    assert loaded.quant_bits == 2
    assert loaded.records[0].logical_bytes == 3_840
    assert loaded.records[0].segments[0].shape == (64, 4)
    assert loaded.records[0].segments[1].shape == (64, 1)
    assert read_expert_record(loaded, root, 1, 0) == expected[0]


@pytest.mark.parametrize("bits", [1, 3, 8])
def test_manifest_rejects_unsupported_affine_bits(tmp_path: Path, bits: int) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)

    with pytest.raises(ExpertManifestError, match="bits|affine Q"):
        replace(manifest, quant_bits=bits).validate_structure()


def test_descriptor_validation_rejects_mode_group_key_and_precision_mismatches(
    tmp_path: Path,
) -> None:
    q2_root = tmp_path / "q2"
    q2_spec, _expected = _make_checkpoint(q2_root, bits=2, key="hy3-expert-q2")
    q2 = build_expert_manifest(q2_root, q2_spec)
    q4_root = tmp_path / "q4"
    q4_spec, _expected = _make_checkpoint(q4_root, bits=4, key="hy3-expert-only-q4")
    q4 = build_expert_manifest(q4_root, q4_spec)

    with pytest.raises(ExpertManifestError, match="affine"):
        replace(q2, quant_mode="symmetric").validate_structure()
    with pytest.raises(ExpertManifestError, match="group"):
        validate_expert_manifest_spec(
            q2,
            replace(
                q2_spec,
                quant_group_size=32,
                total_tensor_bytes=2 * 4_608 + 8,
            ),
        )
    with pytest.raises(ExpertManifestError, match="model key"):
        validate_expert_manifest_spec(q2, replace(q2_spec, key="hy3-expert-only-q4"))
    with pytest.raises(ExpertManifestError, match="bits"):
        validate_expert_manifest_spec(q2, q4_spec)
    with pytest.raises(ExpertManifestError, match="bits"):
        validate_expert_manifest_spec(q4, q2_spec)
    with pytest.raises(ExpertManifestError, match="source"):
        validate_expert_manifest_spec(
            q2,
            replace(q2_spec, quant_revision="different-revision"),
        )


def test_authoritative_q2_sidecar_is_explicit_contiguous_and_verifiable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, authoritative = _make_authoritative_checkpoint(root)

    sidecar_shards = [
        shard for shard in authoritative.shards if shard.kind == "sidecar"
    ]
    resident_shards = [
        shard for shard in authoritative.shards if shard.kind == "safetensors"
    ]
    assert authoritative.sidecar is not None
    assert len(sidecar_shards) == 1
    assert len(resident_shards) == 1
    assert sidecar_shards[0].name == authoritative.sidecar.file
    assert sidecar_shards[0].size == authoritative.sidecar.size
    assert sidecar_shards[0].sha256 == authoritative.sidecar.sha256
    assert sidecar_shards[0].header_bytes == 0
    assert sidecar_shards[0].header_sha256 == EMPTY_SHA256
    assert sidecar_shards[0].to_dict()["kind"] == "sidecar"
    assert all("kind" not in shard.to_dict() for shard in resident_shards)
    for record in authoritative.records:
        assert record.sidecar_offset is not None
        cursor = record.sidecar_offset
        for segment in record.segments:
            assert segment.shard == authoritative.sidecar.file
            assert segment.offset == cursor
            cursor += segment.length
        assert cursor == record.sidecar_offset + record.logical_bytes

    validate_expert_manifest_spec(authoritative, spec)
    report = verify_expert_manifest(
        authoritative,
        root,
        verify_records=True,
        verify_shard_hashes=True,
        verify_sidecar_hash=True,
    )
    assert report == {
        "valid": True,
        "model_key": "hy3-expert-q2",
        "checked_shards": 2,
        "checked_records": 2,
        "sidecar_verified": True,
    }

    saved = save_expert_manifest(authoritative, root / "expert-manifest.json")
    assert load_expert_manifest(root / "expert-manifest.json") == saved


def test_make_sidecar_authoritative_requires_resident_hashes(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(
        root,
        bits=2,
        key="hy3-expert-q2",
        separate_resident=True,
    )
    manifest = build_expert_manifest(root, spec, hash_shards=False)
    with_sidecar = build_expert_sidecar(manifest, root, root / "experts.bin")

    with pytest.raises(ExpertManifestError, match="hash"):
        make_sidecar_authoritative(with_sidecar, spec)


def test_make_sidecar_authoritative_rejects_stale_input_digest(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(
        root,
        bits=2,
        key="hy3-expert-q2",
        separate_resident=True,
    )
    manifest = build_expert_manifest(root, spec, hash_shards=True)
    with_sidecar = build_expert_sidecar(manifest, root, root / "experts.bin")
    assert with_sidecar.sidecar is not None
    tampered = replace(
        with_sidecar,
        sidecar=replace(with_sidecar.sidecar, sha256="f" * 64),
    )

    with pytest.raises(ExpertManifestError, match="manifest digest"):
        make_sidecar_authoritative(tampered, spec)


def test_authoritative_sidecar_rejects_metadata_and_component_range_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, authoritative = _make_authoritative_checkpoint(root)
    assert authoritative.sidecar is not None
    sidecar_index = next(
        index
        for index, shard in enumerate(authoritative.shards)
        if shard.kind == "sidecar"
    )
    bad_shard = replace(
        authoritative.shards[sidecar_index],
        sha256="f" * 64,
    )
    shards = list(authoritative.shards)
    shards[sidecar_index] = bad_shard

    with pytest.raises(ExpertManifestError, match="sidecar.*mismatch"):
        validate_expert_manifest_spec(
            replace(authoritative, shards=tuple(shards)), spec
        )

    record = authoritative.records[0]
    segments = list(record.segments)
    segments[1] = replace(segments[1], offset=segments[1].offset + 1)
    records = list(authoritative.records)
    records[0] = replace(record, segments=tuple(segments))
    with pytest.raises(ExpertManifestError, match="contiguous"):
        validate_expert_manifest_spec(
            replace(authoritative, records=tuple(records)), spec
        )


@pytest.mark.parametrize(
    "tensor_name",
    [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.80.self_attn.q_proj.weight",
    ],
)
def test_authoritative_residents_reject_routed_and_mtp_tensors(
    tmp_path: Path,
    tensor_name: str,
) -> None:
    root = tmp_path / "model"
    spec, authoritative = _make_authoritative_checkpoint(root)
    resident = replace(authoritative.resident_tensors[0], tensor=tensor_name)

    with pytest.raises(ExpertManifestError, match="resident.*routed|MTP"):
        validate_expert_manifest_spec(
            replace(authoritative, resident_tensors=(resident,)), spec
        )


def test_authoritative_resident_inventory_rejects_missing_index_and_extra_shard(
    tmp_path: Path,
) -> None:
    no_index_root = tmp_path / "no-index"
    _spec, no_index = _make_authoritative_checkpoint(no_index_root)
    (no_index_root / "model.safetensors.index.json").unlink()
    with pytest.raises(ExpertManifestError, match="resident index"):
        verify_expert_manifest(no_index, no_index_root)

    missing_root = tmp_path / "missing"
    _spec, missing = _make_authoritative_checkpoint(missing_root)
    index_path = missing_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"] = {}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ExpertManifestError, match="inventory|mismatch|safetensors"):
        verify_expert_manifest(missing, missing_root)

    extra_root = tmp_path / "extra"
    _spec, extra = _make_authoritative_checkpoint(extra_root)
    _write_safetensors(
        extra_root / "model-extra.safetensors",
        [("extra.weight", "F32", [1], b"\0\0\0\0")],
    )
    with pytest.raises(ExpertManifestError, match="unreferenced"):
        verify_expert_manifest(extra, extra_root)


def test_authoritative_resident_inventory_accepts_non_model_shard_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resident-names"
    _spec, authoritative = _make_authoritative_checkpoint(root)
    old_shard = next(
        shard for shard in authoritative.shards if shard.kind == "safetensors"
    )
    new_name = "resident-00001-of-00001.safetensors"
    (root / old_shard.name).rename(root / new_name)

    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"] = {tensor: new_name for tensor in index["weight_map"]}
    index_path.write_text(json.dumps(index), encoding="utf-8")

    renamed = replace(
        authoritative,
        shards=tuple(
            replace(shard, name=new_name) if shard.name == old_shard.name else shard
            for shard in authoritative.shards
        ),
        resident_tensors=tuple(
            replace(tensor, shard=new_name)
            if tensor.shard == old_shard.name
            else tensor
            for tensor in authoritative.resident_tensors
        ),
        manifest_sha256=None,
    ).with_digest()

    report = verify_expert_manifest(renamed, root)

    assert report["checked_shards"] == 2


def test_authoritative_filter_retains_18_residents_and_drops_34_q4_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    source_spec, _expected = _make_checkpoint(
        root,
        bits=2,
        key="hy3-expert-q2",
        separate_resident=True,
    )
    source = build_expert_manifest(
        root,
        source_spec,
        hash_records=True,
        hash_shards=True,
    )
    source = build_expert_sidecar(source, root, root / "experts.bin")
    spec = _tiny_spec(resident_bytes=18, bits=2, key="hy3-expert-q2")
    expert_shard_names = {
        segment.shard for record in source.records for segment in record.segments
    }
    expert_shards = [
        shard for shard in source.shards if shard.name in expert_shard_names
    ]
    while len(expert_shards) < 34:
        index = len(expert_shards)
        expert_shards.append(
            ShardInfo(
                name=f"q4-source-{index:02d}.safetensors",
                size=1,
                header_bytes=1,
                header_sha256="2" * 64,
                sha256="3" * 64,
            )
        )
    resident_shards = tuple(
        ShardInfo(
            name=f"model-{index:05d}-of-00018.safetensors",
            size=2,
            header_bytes=1,
            header_sha256="4" * 64,
            sha256="5" * 64,
        )
        for index in range(1, 19)
    )
    residents = tuple(
        ResidentTensor(
            tensor=f"resident.{index:02d}",
            shard=shard.name,
            offset=1,
            length=1,
            dtype="U8",
            shape=(1,),
        )
        for index, shard in enumerate(resident_shards)
    )
    expanded = replace(
        source,
        artifact_tensor_bytes=spec.total_tensor_bytes,
        resident_tensor_bytes=18,
        shards=tuple(expert_shards) + resident_shards,
        resident_tensors=residents,
        manifest_sha256=None,
    ).with_digest()

    authoritative = make_sidecar_authoritative(expanded, spec)

    assert len(expert_shards) == 34
    assert (
        len([shard for shard in authoritative.shards if shard.kind == "safetensors"])
        == 18
    )
    assert len(authoritative.shards) == 19
    source_shard_names = {shard.name for shard in expert_shards}
    assert all(shard.name not in source_shard_names for shard in authoritative.shards)


def test_manifest_declared_counts_reject_booleans_and_array_ceilings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)

    boolean_count = manifest.to_dict()
    boolean_count["artifact"]["shard_count"] = True
    with pytest.raises(ExpertManifestError, match="exact integer"):
        ExpertManifest.from_dict(boolean_count, verify_digest=False)

    too_many_shards = manifest.to_dict()
    too_many_shards["shards"] = [too_many_shards["shards"][0]] * (
        MAX_MANIFEST_SHARDS + 1
    )
    too_many_shards["artifact"]["shard_count"] = MAX_MANIFEST_SHARDS + 1
    with pytest.raises(ExpertManifestError, match="shard.*limit|too many shards"):
        ExpertManifest.from_dict(too_many_shards, verify_digest=False)

    too_many_records = manifest.to_dict()
    too_many_records["records"] = [too_many_records["records"][0]] * (
        MAX_MANIFEST_RECORDS + 1
    )
    too_many_records["artifact"]["record_count"] = MAX_MANIFEST_RECORDS + 1
    with pytest.raises(ExpertManifestError, match="record.*limit|too many records"):
        ExpertManifest.from_dict(too_many_records, verify_digest=False)

    too_many_residents = manifest.to_dict()
    too_many_residents["resident_tensors"] = [
        too_many_residents["resident_tensors"][0]
    ] * (MAX_MANIFEST_RESIDENT_TENSORS + 1)
    with pytest.raises(ExpertManifestError, match="resident.*limit|too many resident"):
        ExpertManifest.from_dict(too_many_residents, verify_digest=False)


def test_manifest_and_index_file_size_limits_fail_before_json_parse(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "oversized-manifest.json"
    with manifest_path.open("wb") as handle:
        handle.truncate(MAX_MANIFEST_BYTES + 1)
    with pytest.raises(ExpertManifestError, match="exceeds.*limit"):
        load_expert_manifest(manifest_path)

    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    index_path = root / "model.safetensors.index.json"
    with index_path.open("wb") as handle:
        handle.truncate(MAX_INDEX_BYTES + 1)
    with pytest.raises(ExpertManifestError, match="exceeds.*limit"):
        build_expert_manifest(root, spec)


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "nested\\escape"])
def test_manifest_rejects_all_unsafe_shard_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    raw = manifest.to_dict()
    raw["shards"][0]["name"] = unsafe

    with pytest.raises(ExpertManifestError, match="unsafe|POSIX"):
        ExpertManifest.from_dict(raw, verify_digest=False)


def test_sparse_oversized_segment_range_fails_structurally(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    raw = manifest.to_dict()
    raw["records"][0]["segments"][0]["offset"] = 10**100

    with pytest.raises(ExpertManifestError, match="exceeds its shard"):
        ExpertManifest.from_dict(raw, verify_digest=False)


def test_external_authoritative_sidecar_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, authoritative = _make_authoritative_checkpoint(root)
    sidecar = root / "experts.bin"
    external = tmp_path / "external.bin"
    sidecar.replace(external)
    sidecar.symlink_to(external)

    with pytest.raises(ExpertManifestError, match="escapes root"):
        verify_expert_manifest(authoritative, root, verify_sidecar_hash=True)


def test_header_resident_shard_and_sidecar_corruption_fail_independently(
    tmp_path: Path,
) -> None:
    header_root = tmp_path / "header"
    _spec, header_manifest = _make_authoritative_checkpoint(header_root)
    resident_shard = next(
        shard for shard in header_manifest.shards if shard.kind == "safetensors"
    )
    shard_path = header_root / resident_shard.name
    raw = shard_path.read_bytes()
    header_length = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_length])
    tensor_name, tensor_info = next(iter(header.items()))
    rewritten = json.dumps(
        {tensor_name: dict(reversed(tuple(tensor_info.items())))},
        separators=(",", ":"),
    ).encode()
    assert len(rewritten) == header_length
    shard_path.write_bytes(raw[:8] + rewritten + raw[8 + header_length :])
    with pytest.raises(ExpertManifestError, match="provenance|header"):
        verify_expert_manifest(header_manifest, header_root)

    shard_root = tmp_path / "shard"
    _spec, shard_manifest = _make_authoritative_checkpoint(shard_root)
    resident = shard_manifest.resident_tensors[0]
    shard_path = shard_root / resident.shard
    fd = os.open(shard_path, os.O_RDWR)
    try:
        original = os.pread(fd, 1, resident.offset)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), resident.offset)
    finally:
        os.close(fd)
    with pytest.raises(ExpertManifestError, match="shard hash mismatch"):
        verify_expert_manifest(
            shard_manifest,
            shard_root,
            verify_shard_hashes=True,
        )

    sidecar_root = tmp_path / "sidecar"
    _spec, sidecar_manifest = _make_authoritative_checkpoint(sidecar_root)
    sidecar_path = sidecar_root / "experts.bin"
    fd = os.open(sidecar_path, os.O_RDWR)
    try:
        original = os.pread(fd, 1, 0)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), 0)
    finally:
        os.close(fd)
    with pytest.raises(ExpertManifestError, match="sidecar hash mismatch"):
        verify_expert_manifest(
            sidecar_manifest,
            sidecar_root,
            verify_sidecar_hash=True,
        )
