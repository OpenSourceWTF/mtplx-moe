"""CPU-only manifests for the unchanged Vontra Qwen3.8 oQ4-MTP artifact."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    ShardInfo,
    build_expert_manifest,
    validate_expert_manifest_spec,
)
from .expert_streaming_models import (
    QWEN38_FLASH_NEXT_Q4,
    ExpertStreamingModelSpec,
)
from .qwen4_ngram import (
    NGramComponent,
    NGramManifest,
    NGramManifestError,
    qwen38_ngram_manifest,
    segmented_ngram_shard,
)

_NGRAM_ROWS_PER_SHARD = 2_500_012
_NGRAM_SHARD_COUNT = 128
_NGRAM_PREFIX = (
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding."
)
_NGRAM_RE = re.compile(
    re.escape(_NGRAM_PREFIX)
    + r"shard_(?P<index>0|[1-9][0-9]*)\.(?P<leaf>weight|scales|biases)$"
)
_NGRAM_LAYOUT = {
    "weight": ("U32", (_NGRAM_ROWS_PER_SHARD, 20), 80),
    "scales": ("BF16", (_NGRAM_ROWS_PER_SHARD, 5), 10),
    "biases": ("BF16", (_NGRAM_ROWS_PER_SHARD, 5), 10),
}


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise NGramManifestError(
            f"{label} must be an exact integer of at least {minimum}"
        )
    return value


def _tensor_field(tensor: Any, field: str) -> Any:
    try:
        return getattr(tensor, field)
    except AttributeError as exc:
        if field == "name":
            try:
                return tensor.tensor
            except AttributeError:
                pass
        raise NGramManifestError(
            f"n-gram tensor metadata is missing {field!r}"
        ) from exc


def build_qwen4_oq4_ngram_manifest(
    shards: Sequence[ShardInfo],
    tensors: Mapping[str, Any],
) -> NGramManifest:
    """Build exact segmented row routes from inspected safetensors headers."""

    if not isinstance(shards, Sequence) or not isinstance(tensors, Mapping):
        raise TypeError("shards and tensors must be inspected checkpoint collections")
    shard_by_name: dict[str, ShardInfo] = {}
    for shard in shards:
        if type(shard) is not ShardInfo:
            raise TypeError("shards must contain exact ShardInfo values")
        if shard.name in shard_by_name:
            raise NGramManifestError(f"duplicate source shard {shard.name}")
        if shard.sha256 is None:
            raise NGramManifestError(
                f"source shard {shard.name} requires a complete SHA-256"
            )
        shard_by_name[shard.name] = shard

    grouped: dict[tuple[int, str], Any] = {}
    malformed: list[str] = []
    for name, tensor in tensors.items():
        if type(name) is not str:
            raise NGramManifestError("tensor names must be exact strings")
        if _NGRAM_PREFIX not in name:
            continue
        match = _NGRAM_RE.fullmatch(name)
        if match is None:
            malformed.append(name)
            continue
        index = int(match.group("index"))
        leaf = match.group("leaf")
        if not 0 <= index < _NGRAM_SHARD_COUNT:
            malformed.append(name)
            continue
        if _tensor_field(tensor, "name") != name:
            raise NGramManifestError(f"tensor mapping key/name mismatch for {name}")
        key = (index, leaf)
        if key in grouped:
            raise NGramManifestError(f"duplicate published n-gram component {key}")
        grouped[key] = tensor
    if malformed:
        raise NGramManifestError(
            f"malformed published n-gram tensors: {sorted(malformed)[:4]}"
        )

    expected = {
        (index, leaf)
        for index in range(_NGRAM_SHARD_COUNT)
        for leaf in _NGRAM_LAYOUT
    }
    missing = sorted(expected - set(grouped))
    if missing:
        raise NGramManifestError(
            f"missing {len(missing)} published n-gram components: {missing[:4]}"
        )
    logical_shards = []
    for index in range(_NGRAM_SHARD_COUNT):
        prefix = f"{_NGRAM_PREFIX}shard_{index}"
        components: list[NGramComponent] = []
        for leaf in ("weight", "scales", "biases"):
            tensor = grouped[(index, leaf)]
            expected_dtype, expected_shape, row_bytes = _NGRAM_LAYOUT[leaf]
            dtype = _tensor_field(tensor, "dtype")
            shape = _tensor_field(tensor, "shape")
            if dtype != expected_dtype:
                raise NGramManifestError(
                    f"{prefix}.{leaf} dtype {dtype!r} does not match {expected_dtype}"
                )
            if type(shape) is not tuple or shape != expected_shape:
                raise NGramManifestError(
                    f"{prefix}.{leaf} shape {shape!r} does not match published shape"
                )
            source_name = _tensor_field(tensor, "shard")
            source = shard_by_name.get(source_name)
            if source is None:
                raise NGramManifestError(
                    f"{prefix}.{leaf} references unknown shard {source_name!r}"
                )
            offset = _exact_int(
                _tensor_field(tensor, "offset"),
                label=f"{prefix}.{leaf} offset",
            )
            length = _exact_int(
                _tensor_field(tensor, "length"),
                label=f"{prefix}.{leaf} length",
                minimum=1,
            )
            expected_length = _NGRAM_ROWS_PER_SHARD * row_bytes
            if length != expected_length:
                raise NGramManifestError(
                    f"{prefix}.{leaf} byte length does not match published shape"
                )
            if offset + length > source.size:
                raise NGramManifestError(f"{prefix}.{leaf} exceeds {source.name}")
            assert source.sha256 is not None
            components.append(
                NGramComponent(
                    component=leaf,  # type: ignore[arg-type]
                    name=source.name,
                    tensor=f"{prefix}.{leaf}",
                    data_offset=offset,
                    row_bytes=row_bytes,
                    data_bytes=length,
                    file_size=source.size,
                    file_sha256=source.sha256,
                    dtype=dtype,
                    shape=shape,
                )
            )
        logical_shards.append(
            segmented_ngram_shard(
                name=f"ngram-shard-{index:03d}",
                tensor=prefix,
                start_row=index * _NGRAM_ROWS_PER_SHARD,
                row_count=_NGRAM_ROWS_PER_SHARD,
                components=tuple(components),
            )
        )
    manifest = qwen38_ngram_manifest("affine-q4-g32", tuple(logical_shards))
    if sum(shard.data_bytes for shard in manifest.shards) != 32_000_153_600:
        raise NGramManifestError("published n-gram byte total changed")
    return manifest


def externalize_source_residents(
    manifest: ExpertManifest,
    *,
    external_tensor_names: frozenset[str],
    external_bytes: int,
    target_spec: ExpertStreamingModelSpec,
) -> ExpertManifest:
    """Remove explicitly streamed tensors from the resident runtime footprint."""

    if type(manifest) is not ExpertManifest:
        raise TypeError("manifest must be an exact ExpertManifest")
    if type(external_tensor_names) is not frozenset or not external_tensor_names:
        raise TypeError("external_tensor_names must be a non-empty exact frozenset")
    if type(external_bytes) is not int or external_bytes <= 0:
        raise ValueError("external_bytes must be an exact positive integer")
    if type(target_spec) is not ExpertStreamingModelSpec:
        raise TypeError("target_spec must be an exact ExpertStreamingModelSpec")
    manifest.validate_structure()
    if (
        manifest.manifest_sha256 is None
        or manifest.manifest_sha256 != manifest.with_digest().manifest_sha256
    ):
        raise ExpertManifestError("source expert manifest digest mismatch")
    residents = tuple(
        tensor
        for tensor in manifest.resident_tensors
        if tensor.tensor not in external_tensor_names
    )
    removed = {
        tensor.tensor: tensor
        for tensor in manifest.resident_tensors
        if tensor.tensor in external_tensor_names
    }
    missing = external_tensor_names - set(removed)
    if missing:
        raise ExpertManifestError(
            f"external resident tensor set is missing {sorted(missing)[:4]}"
        )
    removed_bytes = sum(tensor.length for tensor in removed.values())
    if removed_bytes != external_bytes:
        raise ExpertManifestError(
            f"external resident bytes {removed_bytes} do not match {external_bytes}"
        )
    result = replace(
        manifest,
        artifact_tensor_bytes=manifest.artifact_tensor_bytes - removed_bytes,
        resident_tensor_bytes=manifest.resident_tensor_bytes - removed_bytes,
        resident_tensors=residents,
        manifest_sha256=None,
    ).with_digest()
    validate_expert_manifest_spec(result, target_spec)
    return result


@dataclass(frozen=True)
class Qwen4OQ4Manifests:
    expert: ExpertManifest
    ngram: NGramManifest


def build_qwen4_oq4_manifests(root: Path | str) -> Qwen4OQ4Manifests:
    """Inspect and hash the unchanged checkpoint once, then build both manifests."""

    spec = QWEN38_FLASH_NEXT_Q4
    source_spec = replace(
        spec,
        total_tensor_bytes=spec.disk_tensor_bytes,
        external_backing_bytes=0,
        quant_model=spec.source_model,
        quant_revision=spec.source_revision,
    )
    source = build_expert_manifest(
        root,
        source_spec,
        source_repo=spec.source_model,
        source_revision=spec.source_revision,
        hash_records=False,
        hash_shards=True,
        require_pinned_tensor_bytes=True,
    )
    ngram_tensors = {
        tensor.tensor: tensor
        for tensor in source.resident_tensors
        if _NGRAM_PREFIX in tensor.tensor
    }
    ngram = build_qwen4_oq4_ngram_manifest(source.shards, ngram_tensors)
    external_names = frozenset(ngram_tensors)
    expert = externalize_source_residents(
        source,
        external_tensor_names=external_names,
        external_bytes=spec.external_backing_bytes,
        target_spec=spec,
    )
    return Qwen4OQ4Manifests(expert=expert, ngram=ngram)


__all__ = [
    "Qwen4OQ4Manifests",
    "build_qwen4_oq4_manifests",
    "build_qwen4_oq4_ngram_manifest",
    "externalize_source_residents",
]
