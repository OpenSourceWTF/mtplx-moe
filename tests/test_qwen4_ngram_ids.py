from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from mtplx.qwen4_ngram import (
    QWEN38_FLASH_NEXT_REVISION,
    NGramGeometry,
    NGramManifest,
    NGramManifestError,
    NGramShard,
    load_ngram_manifest,
    qwen38_ngram_manifest,
    save_ngram_manifest,
    verify_ngram_manifest,
)

EXPECTED_SIZES = (
    20_000_003,
    20_000_023,
    20_000_033,
    20_000_047,
    20_000_059,
    20_000_063,
    20_000_069,
    20_000_077,
    20_000_081,
    20_000_093,
    20_000_107,
    20_000_147,
    20_000_153,
    20_000_159,
    20_000_161,
    20_000_171,
)
EXPECTED_OFFSETS = (
    0,
    20_000_003,
    40_000_026,
    60_000_059,
    80_000_106,
    100_000_165,
    120_000_228,
    140_000_297,
    160_000_374,
    180_000_455,
    200_000_548,
    220_000_655,
    240_000_802,
    260_000_955,
    280_001_114,
    300_001_275,
)


def test_qwen38_geometry_is_exact_and_frozen() -> None:
    geometry = NGramGeometry.qwen38()

    assert geometry.vocab_size == 248_320
    assert geometry.eos_token_id == 248_044
    assert geometry.ngram_size == 3
    assert geometry.heads_per_ngram == 8
    assert geometry.ple_layer_index == 0
    assert geometry.seed == 1234
    assert geometry.multipliers == (
        23_703_573_157_769,
        20_109_073_645_365,
        8_052_911_324_071,
    )
    assert geometry.head_vocab_sizes == EXPECTED_SIZES
    assert geometry.head_offsets == EXPECTED_OFFSETS
    assert geometry.total_vocab_size == 320_001_446
    assert geometry.padded_rows == 320_001_536
    with pytest.raises(FrozenInstanceError):
        geometry.seed = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 0},
        {"vocab_size": 10},
        {"eos_token_id": 1, "ngram_size": 2, "heads_per_ngram": 1},
    ],
)
def test_geometry_rejects_public_parameterization(kwargs: dict[str, int]) -> None:
    with pytest.raises(TypeError):
        NGramGeometry(**kwargs)


def test_row_ids_match_official_examples() -> None:
    geometry = NGramGeometry.qwen38_flash_next()

    ids = geometry.row_ids([[10, 20, 30]])

    assert ids.shape == (1, 3, 16)
    assert ids[0, -1].tolist() == [
        9_878_115,
        26_555_603,
        54_895_210,
        62_571_545,
        80_580_723,
        119_917_398,
        128_922_427,
        147_596_134,
        168_936_175,
        195_223_391,
        219_226_064,
        233_524_685,
        246_670_267,
        279_816_194,
        297_531_600,
        306_108_296,
    ]


def test_eos_aware_shift_does_not_cross_segment() -> None:
    geometry = NGramGeometry.qwen38_flash_next()

    ids = geometry.row_ids([[10, 248_044, 20, 30]])

    assert ids[0, -1, 8:].tolist() == [
        161_934_820,
        194_063_368,
        213_365_097,
        236_446_602,
        256_287_798,
        276_488_700,
        283_302_268,
        317_969_618,
    ]


def test_incremental_chunks_equal_whole_sequence_for_batch_and_eos() -> None:
    geometry = NGramGeometry.qwen38_flash_next()
    tokens = [
        [10, 20, 248_044, 30, 40, 50, 60],
        [248_044, 3, 4, 5, 248_044, 6, 7],
    ]
    whole = geometry.row_ids(tokens)

    first, context = geometry.plan_incremental([row[:1] for row in tokens])
    second, context = geometry.plan_incremental(
        [row[1:4] for row in tokens], prior_context=context
    )
    third, context = geometry.plan_incremental(
        [row[4:] for row in tokens], prior_context=context
    )
    tokenwise = []
    token_context = None
    for column in range(len(tokens[0])):
        planned, token_context = geometry.plan_incremental(
            [[row[column]] for row in tokens], prior_context=token_context
        )
        tokenwise.append(planned)

    import numpy as np

    assert np.array_equal(np.concatenate((first, second, third), axis=1), whole)
    assert np.array_equal(np.concatenate(tokenwise, axis=1), whole)
    assert context == ((50, 60), (6, 7))


@pytest.mark.parametrize(
    ("tokens", "context"),
    [
        ([[248_320]], None),
        ([[-1]], None),
        ([[1], [2, 3]], None),
        ([[1]], ((1, 2, 3),)),
        ([[1]], ((1, 248_320),)),
        ([[True]], None),
    ],
)
def test_planning_rejects_invalid_inputs(tokens: object, context: object) -> None:
    geometry = NGramGeometry.qwen38_flash_next()

    with pytest.raises((TypeError, ValueError)):
        geometry.plan_incremental(tokens, prior_context=context)  # type: ignore[arg-type]


def _shard(
    root: Path,
    *,
    name: str,
    start_row: int,
    rows: list[bytes],
    prefix: bytes = b"HEAD",
) -> NGramShard:
    payload = b"".join(rows)
    contents = prefix + payload
    (root / name).write_bytes(contents)
    return NGramShard(
        name=name,
        tensor=f"tensor-{start_row}",
        start_row=start_row,
        row_count=len(rows),
        data_offset=len(prefix),
        data_bytes=len(payload),
        file_size=len(contents),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _tiny_manifest(root: Path) -> NGramManifest:
    first = _shard(root, name="part-1.bin", start_row=0, rows=[b"aa", b"bb"])
    second = _shard(root, name="part-2.bin", start_row=2, rows=[b"cc"])
    return NGramManifest(
        source_repo="example/repo",
        source_revision="a" * 40,
        storage="bf16",
        row_width=1,
        row_bytes=2,
        padded_rows=3,
        shards=(first, second),
    ).with_digest()


def test_manifest_roundtrip_verify_and_locate_exact_offsets(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    path = tmp_path / "ngram-manifest.json"

    saved = save_ngram_manifest(manifest, path)
    loaded = load_ngram_manifest(path)
    report = verify_ngram_manifest(tmp_path, loaded)

    assert loaded == saved == manifest
    assert report == {"shards": 2, "rows": 3, "bytes": 6}
    shard, offset = loaded.locate_row(0)
    assert (shard.name, offset) == ("part-1.bin", 4)
    shard, offset = loaded.locate_row(2)
    assert (shard.name, offset) == ("part-2.bin", 4)
    with pytest.raises(IndexError):
        loaded.locate_row(3)


def test_qwen38_manifest_constructor_pins_provenance_and_layout() -> None:
    shard = NGramShard(
        name="all.bin",
        tensor="model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        start_row=0,
        row_count=320_001_536,
        data_offset=0,
        data_bytes=32_000_153_600,
        file_size=32_000_153_600,
        sha256="0" * 64,
    )

    manifest = qwen38_ngram_manifest("affine-q4-g32", (shard,))

    assert manifest.source_repo == "Qwen/Qwen3.8-Flash-Next"
    assert manifest.source_revision == QWEN38_FLASH_NEXT_REVISION
    assert manifest.row_width == 160
    assert manifest.row_bytes == 100
    assert manifest.padded_rows == 320_001_536
    assert manifest.digest is not None


@pytest.mark.parametrize(
    "change",
    [
        {"storage": "q8"},
        {"row_width": 2},
        {"row_bytes": 3},
        {"shards": ()},
    ],
)
def test_manifest_rejects_bad_storage_geometry_and_empty_coverage(
    tmp_path: Path, change: dict[str, object]
) -> None:
    manifest = _tiny_manifest(tmp_path)

    with pytest.raises(NGramManifestError):
        replace(manifest, **change).validate_structure()


@pytest.mark.parametrize("kind", ["gap", "overlap", "duplicate"])
def test_manifest_rejects_noncontiguous_shards(tmp_path: Path, kind: str) -> None:
    manifest = _tiny_manifest(tmp_path)
    first, second = manifest.shards
    if kind == "gap":
        second = replace(second, start_row=3)
    elif kind == "overlap":
        second = replace(second, start_row=1)
    else:
        second = replace(second, name=first.name)

    with pytest.raises(NGramManifestError):
        replace(manifest, shards=(first, second)).validate_structure()


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "part.bin/",
        "./part.bin",
        "part/../part.bin",
        "part//alias.bin",
        "a/b.bin",
        "a\\b.bin",
        "/tmp/escape.bin",
        "\x00part.bin",
        "part\x00.bin",
        "ngram-e\u0301.bin",
    ],
)
def test_manifest_rejects_unsafe_single_component_names(
    tmp_path: Path, name: str
) -> None:
    manifest = _tiny_manifest(tmp_path)

    with pytest.raises(NGramManifestError):
        replace(manifest.shards[0], name=name)


def test_manifest_accepts_plain_unicode_single_component_name(tmp_path: Path) -> None:
    shard = _shard(
        tmp_path,
        name="ngram-数据.bin",
        start_row=0,
        rows=[b"aa", b"bb", b"cc"],
    )
    manifest = NGramManifest(
        source_repo="example/repo",
        source_revision="a" * 40,
        storage="bf16",
        row_width=1,
        row_bytes=2,
        padded_rows=3,
        shards=(shard,),
    ).with_digest()

    assert verify_ngram_manifest(tmp_path, manifest)["rows"] == 3


@pytest.mark.parametrize("name", ["part.bin/", "part\x00.bin", "ngram-e\u0301.bin"])
def test_load_rejects_unsafe_shard_name_as_manifest_error(
    tmp_path: Path, name: str
) -> None:
    value = _tiny_manifest(tmp_path).to_dict()
    value["shards"][0]["name"] = name
    value.pop("digest")
    path = tmp_path / "unsafe-manifest.json"
    path.write_text(json.dumps(value))

    with pytest.raises(NGramManifestError, match="shard name"):
        load_ngram_manifest(path, verify_digest=False)


def test_verify_rejects_symlink(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    target = tmp_path / "target.bin"
    target.write_bytes((tmp_path / "part-1.bin").read_bytes())
    (tmp_path / "part-1.bin").unlink()
    os.symlink(target, tmp_path / "part-1.bin")

    with pytest.raises(NGramManifestError):
        verify_ngram_manifest(tmp_path, manifest)


@pytest.mark.parametrize("corruption", ["short", "payload"])
def test_verify_rejects_short_or_corrupt_payload(
    tmp_path: Path, corruption: str
) -> None:
    manifest = _tiny_manifest(tmp_path)
    path = tmp_path / "part-1.bin"
    if corruption == "short":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.write_bytes(b"HEADaXbb")

    with pytest.raises(NGramManifestError):
        verify_ngram_manifest(tmp_path, manifest)


def test_load_rejects_stale_digest_malformed_types_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    manifest = _tiny_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    save_ngram_manifest(manifest, path)
    value = json.loads(path.read_text())
    value["source_revision"] = "b" * 40
    path.write_text(json.dumps(value))
    with pytest.raises(NGramManifestError, match="digest"):
        load_ngram_manifest(path)

    value["padded_rows"] = "3"
    value["digest"] = None
    path.write_text(json.dumps(value))
    with pytest.raises(NGramManifestError):
        load_ngram_manifest(path, verify_digest=False)

    path.write_text('{"format":"one","format":"two"}')
    with pytest.raises(NGramManifestError, match="duplicate"):
        load_ngram_manifest(path)
