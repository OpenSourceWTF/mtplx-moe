from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from mtplx.expert_manifest import ShardInfo
from mtplx.qwen4_oq4 import build_qwen4_oq4_ngram_manifest


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]


def _inventory() -> tuple[tuple[ShardInfo, ...], dict[str, TensorInfo]]:
    rows = 2_500_012
    tensors: dict[str, TensorInfo] = {}
    offset = 4096
    shard_name = "model-00001-of-00022.safetensors"
    for index in range(128):
        prefix = (
            "language_model.model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{index}"
        )
        for leaf, dtype, shape, length in (
            ("weight", "U32", (rows, 20), rows * 80),
            ("scales", "BF16", (rows, 5), rows * 10),
            ("biases", "BF16", (rows, 5), rows * 10),
        ):
            name = f"{prefix}.{leaf}"
            tensors[name] = TensorInfo(
                name=name,
                shard=shard_name,
                offset=offset,
                length=length,
                dtype=dtype,
                shape=shape,
            )
            offset += length
    shards = (
        ShardInfo(
            name=shard_name,
            size=offset,
            header_bytes=4096,
            header_sha256="0" * 64,
            sha256="1" * 64,
        ),
    )
    return shards, tensors


def test_build_qwen4_oq4_ngram_manifest_pins_source_native_layout() -> None:
    shards, tensors = _inventory()

    manifest = build_qwen4_oq4_ngram_manifest(shards, tensors)

    assert manifest.storage == "affine-q4-g32"
    assert manifest.row_bytes == 100
    assert manifest.padded_rows == 320_001_536
    assert len(manifest.shards) == 128
    assert sum(shard.data_bytes for shard in manifest.shards) == 32_000_153_600
    first = manifest.shards[0]
    assert tuple(component.component for component in first.components) == (
        "weight",
        "scales",
        "biases",
    )
    assert tuple(component.row_bytes for component in first.components) == (80, 10, 10)
    assert all(component.name == shards[0].name for component in first.components)
    assert all(component.file_sha256 == "1" * 64 for component in first.components)


def test_build_qwen4_oq4_ngram_manifest_rejects_missing_or_drifted_tensor() -> None:
    shards, tensors = _inventory()
    missing = dict(tensors)
    missing.pop(
        "language_model.model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_127.biases"
    )
    with pytest.raises(ValueError, match="missing"):
        build_qwen4_oq4_ngram_manifest(shards, missing)

    drifted = dict(tensors)
    name = (
        "language_model.model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_0.weight"
    )
    drifted[name] = replace(drifted[name], shape=(2_500_012, 19))
    with pytest.raises(ValueError, match="shape"):
        build_qwen4_oq4_ngram_manifest(shards, drifted)
