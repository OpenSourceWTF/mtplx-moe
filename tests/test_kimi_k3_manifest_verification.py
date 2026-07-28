from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import mtplx.expert_manifest as manifest_module
from mtplx.expert_manifest import (
    ExpertManifestError,
    ResidentTensor,
    _read_safetensors_header,
    _validate_authoritative_resident_inventory,
    build_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import get_model_spec
from test_expert_manifest import (
    _make_authoritative_checkpoint,
    _make_checkpoint,
    _write_safetensors,
)


KIMI_K3_SPEC = get_model_spec("kimi-k3-q1t")


def _resident(name: str, *, offset: int = 16) -> ResidentTensor:
    return ResidentTensor(
        tensor=name,
        shard="residents.safetensors",
        offset=offset,
        length=4,
        dtype="F32",
        shape=(1,),
    )


def test_kimi_authoritative_inventory_allows_only_preserved_non_text_prefixes() -> None:
    text = _resident("language_model.model.embed_tokens.weight")
    inventory = {
        text.tensor: text,
        "vision_tower.patch_embed.proj.weight": _resident(
            "vision_tower.patch_embed.proj.weight",
            offset=20,
        ),
        "mm_projector.proj.0.weight": _resident(
            "mm_projector.proj.0.weight",
            offset=24,
        ),
    }

    _validate_authoritative_resident_inventory(
        {text.tensor: text},
        inventory,
        trusted_spec=KIMI_K3_SPEC,
    )


@pytest.mark.parametrize(
    "extra",
    (
        "language_model.unlisted.weight",
        "audio_tower.weight",
        "vision_towerish.weight",
        "mm_projectorish.weight",
    ),
)
def test_kimi_authoritative_inventory_rejects_unlisted_extra_prefixes(
    extra: str,
) -> None:
    text = _resident("language_model.model.embed_tokens.weight")

    with pytest.raises(
        ExpertManifestError,
        match="authoritative resident index/header inventory mismatch",
    ):
        _validate_authoritative_resident_inventory(
            {text.tensor: text},
            {text.tensor: text, extra: _resident(extra, offset=20)},
            trusted_spec=KIMI_K3_SPEC,
        )


def test_kimi_authoritative_inventory_requires_every_text_resident() -> None:
    text = _resident("language_model.model.embed_tokens.weight")

    with pytest.raises(
        ExpertManifestError,
        match="authoritative resident index/header inventory mismatch",
    ):
        _validate_authoritative_resident_inventory(
            {text.tensor: text},
            {"vision_tower.weight": _resident("vision_tower.weight")},
            trusted_spec=KIMI_K3_SPEC,
        )


def test_kimi_authoritative_manifest_allowlist_must_be_text_only() -> None:
    vision = _resident("vision_tower.weight")

    with pytest.raises(
        ExpertManifestError,
        match="Kimi K3 resident allowlist must be text-only",
    ):
        _validate_authoritative_resident_inventory(
            {vision.tensor: vision},
            {vision.tensor: vision},
            trusted_spec=KIMI_K3_SPEC,
        )


def test_kimi_authoritative_inventory_still_checks_text_metadata_exactly() -> None:
    text = _resident("language_model.model.embed_tokens.weight")

    with pytest.raises(
        ExpertManifestError,
        match="authoritative resident metadata mismatch",
    ):
        _validate_authoritative_resident_inventory(
            {text.tensor: text},
            {text.tensor: replace(text, offset=32)},
            trusted_spec=KIMI_K3_SPEC,
        )


def test_other_authoritative_models_still_require_exact_inventory_equality() -> None:
    text = _resident("model.embed_tokens.weight")
    vision = _resident("vision_tower.weight", offset=20)

    with pytest.raises(
        ExpertManifestError,
        match="authoritative resident index/header inventory mismatch",
    ):
        _validate_authoritative_resident_inventory(
            {text.tensor: text},
            {text.tensor: text, vision.tensor: vision},
        )


def test_spec_validation_rejects_preserved_shards_for_non_kimi_descriptor(
    tmp_path,
) -> None:
    source_spec, authoritative = _make_authoritative_checkpoint(tmp_path / "model")
    resident_shard = next(
        shard for shard in authoritative.shards if shard.kind == "safetensors"
    )
    preserved = replace(
        resident_shard,
        name="unreferenced-preserved.safetensors",
        kind="preserved-safetensors",
    )
    manifest = replace(
        authoritative,
        shards=authoritative.shards + (preserved,),
    ).with_digest()

    with pytest.raises(
        ExpertManifestError,
        match="trusted Kimi K3 descriptor",
    ):
        validate_expert_manifest_spec(manifest, source_spec)


def test_spec_validation_requires_preserved_shards_to_be_authoritative(
    tmp_path,
) -> None:
    source_spec, _expected = _make_checkpoint(tmp_path / "model")
    manifest = build_expert_manifest(
        tmp_path / "model",
        source_spec,
        hash_shards=True,
    )
    resident_shard = next(
        shard for shard in manifest.shards if shard.kind == "safetensors"
    )
    preserved = replace(
        resident_shard,
        name="unreferenced-preserved.safetensors",
        kind="preserved-safetensors",
    )
    manifest = replace(
        manifest,
        shards=manifest.shards + (preserved,),
    ).with_digest()

    with pytest.raises(
        ExpertManifestError,
        match="authoritative manifest",
    ):
        validate_expert_manifest_spec(manifest, KIMI_K3_SPEC)


def test_verify_kimi_authoritative_inventory_accepts_trusted_text_subset(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "model"
    source_spec, authoritative = _make_authoritative_checkpoint(root)
    resident_shard = next(
        shard for shard in authoritative.shards if shard.kind == "safetensors"
    )
    resident_path = root / resident_shard.name
    text_name = "language_model.model.embed_tokens.weight"
    vision_name = "vision_tower.patch_embed.proj.weight"
    _write_safetensors(
        resident_path,
        [(text_name, "F32", [2], bytes(range(8)))],
    )
    text_shard, tensors = _read_safetensors_header(
        resident_path,
        relative_name=resident_shard.name,
    )
    text_shard = replace(
        text_shard,
        sha256=hashlib.sha256(resident_path.read_bytes()).hexdigest(),
    )
    vision_path = root / "preserved-vision.safetensors"
    _write_safetensors(
        vision_path,
        [(vision_name, "F32", [1], bytes(range(4)))],
    )
    vision_shard, _vision_tensors = _read_safetensors_header(
        vision_path,
        relative_name=vision_path.name,
    )
    vision_shard = replace(
        vision_shard,
        sha256=hashlib.sha256(vision_path.read_bytes()).hexdigest(),
        kind="preserved-safetensors",
    )
    by_name = {tensor.name: tensor for tensor in tensors}
    text = by_name[text_name]
    text_resident = ResidentTensor(
        tensor=text.name,
        shard=text.shard,
        offset=text.offset,
        length=text.length,
        dtype=text.dtype,
        shape=text.shape,
    )
    index = {
        "metadata": {"total_size": 12},
        "weight_map": {
            text_name: resident_shard.name,
            vision_name: vision_path.name,
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    shards = tuple(
        text_shard if item.kind == "safetensors" else item
        for item in authoritative.shards
    ) + (vision_shard,)
    text_bytes = 8
    total_bytes = authoritative.routed_expert_bytes + text_bytes
    trusted_spec = replace(
        source_spec,
        key="kimi-k3-q1t",
        total_tensor_bytes=total_bytes,
    )
    trusted_manifest = replace(
        authoritative,
        model_key=trusted_spec.key,
        artifact_tensor_bytes=total_bytes,
        resident_tensor_bytes=text_bytes,
        shards=shards,
        resident_tensors=(text_resident,),
    ).with_digest()
    monkeypatch.setattr(
        manifest_module,
        "get_model_spec",
        lambda key: trusted_spec if key == trusted_spec.key else get_model_spec(key),
    )

    report = verify_expert_manifest(
        trusted_manifest,
        root,
        verify_shard_hashes=True,
    )

    assert report["valid"] is True
    assert report["model_key"] == "kimi-k3-q1t"


def test_spoofed_kimi_model_key_cannot_unlock_subset_verification(
    tmp_path,
) -> None:
    root = tmp_path / "model"
    _spec, authoritative = _make_authoritative_checkpoint(root)
    spoofed = replace(authoritative, model_key="kimi-k3-q1t").with_digest()

    with pytest.raises(
        ExpertManifestError,
        match="do not match descriptor|descriptor codec|source identity|record keys",
    ):
        verify_expert_manifest(spoofed, root)
