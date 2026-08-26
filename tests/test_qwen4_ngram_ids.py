from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import resource
import tempfile
import threading
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from mtplx import qwen4_ngram
from mtplx.qwen4_ngram import (
    QWEN38_FLASH_NEXT_REVISION,
    NGramComponent,
    NGramGeometry,
    NGramManifest,
    NGramManifestError,
    NGramShard,
    load_ngram_manifest,
    qwen38_ngram_manifest,
    save_ngram_manifest,
    segmented_ngram_shard,
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

    assert NGramGeometry() == geometry
    assert NGramGeometry.qwen38_flash_next() == geometry
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


def test_geometry_rejects_frozen_dataclass_subclass_drift() -> None:
    def define_drifted_geometry() -> None:
        @dataclass(frozen=True)
        class DriftedGeometry(NGramGeometry):
            vocab_size: int = 10
            eos_token_id: int = 1
            ngram_size: int = 2
            heads_per_ngram: int = 1
            seed: int = 0

        assert DriftedGeometry().seed == 0
        assert DriftedGeometry.qwen38().vocab_size == 10

    with pytest.raises(TypeError, match="subclass"):
        define_drifted_geometry()


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


@pytest.mark.parametrize(
    "tokens",
    [
        [[10**100]],
        np.asarray([[10**100]], dtype=object),
        np.asarray([[(1 << 64) - 1]], dtype=np.uint64),
    ],
)
def test_planning_translates_oversized_tokens_to_range_error(tokens: object) -> None:
    with pytest.raises(ValueError, match="range|vocabulary"):
        NGramGeometry.qwen38().row_ids(tokens)


@pytest.mark.parametrize(
    "context",
    [
        ((10**100, 1),),
        np.asarray([[10**100, 1]], dtype=object),
        np.asarray([[(1 << 64) - 1, 1]], dtype=np.uint64),
    ],
)
def test_planning_translates_oversized_context_to_range_error(
    context: object,
) -> None:
    with pytest.raises(ValueError, match="range|vocabulary"):
        NGramGeometry.qwen38().plan_incremental(
            [[1]], prior_context=context  # type: ignore[arg-type]
        )


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
    path = root / name
    path.write_bytes(contents)
    path.chmod(0o444)
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
    with verify_ngram_manifest(tmp_path, loaded) as artifact:
        report = artifact.report
        retained_root_fd = artifact.root_fileno()
        retained_fd = artifact.shards[0].fileno()
        assert artifact.shards[0].pread(4, 2) == b"aa"
        assert os.fstat(retained_fd).st_size == 8
        assert artifact.shards[0].identity.mtime_ns > 0
        assert artifact.shards[0].identity.ctime_ns > 0

    assert artifact.closed
    assert all(shard.closed for shard in artifact.shards)
    with pytest.raises(OSError):
        os.fstat(retained_fd)
    with pytest.raises(OSError):
        os.fstat(retained_root_fd)
    artifact.close()
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

    assert manifest.source_repo == "Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP"
    assert manifest.source_revision == QWEN38_FLASH_NEXT_REVISION
    assert manifest.row_width == 160
    assert manifest.row_bytes == 100
    assert manifest.padded_rows == 320_001_536
    assert manifest.digest is not None


def test_segmented_affine_manifest_roundtrips_source_safetensors_layout() -> None:
    file_digest = "1" * 64
    components = (
        NGramComponent(
            component="weight",
            name="model.safetensors",
            tensor="ngram.shard_0.weight",
            data_offset=0,
            row_bytes=16,
            data_bytes=32,
            file_size=40,
            file_sha256=file_digest,
            dtype="U32",
            shape=(2, 4),
        ),
        NGramComponent(
            component="scales",
            name="model.safetensors",
            tensor="ngram.shard_0.scales",
            data_offset=32,
            row_bytes=2,
            data_bytes=4,
            file_size=40,
            file_sha256=file_digest,
            dtype="BF16",
            shape=(2, 1),
        ),
        NGramComponent(
            component="biases",
            name="model.safetensors",
            tensor="ngram.shard_0.biases",
            data_offset=36,
            row_bytes=2,
            data_bytes=4,
            file_size=40,
            file_sha256=file_digest,
            dtype="BF16",
            shape=(2, 1),
        ),
    )
    shard = segmented_ngram_shard(
        name="ngram-shard-000",
        tensor="ngram.shard_0",
        start_row=0,
        row_count=2,
        components=components,
    )
    manifest = NGramManifest(
        source_repo="Vontra/example",
        source_revision="a" * 40,
        storage="affine-q4-g32",
        row_width=32,
        row_bytes=20,
        padded_rows=2,
        shards=(shard,),
    ).with_digest()

    restored = NGramManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert restored.shards[0].components == components
    assert restored.shards[0].data_bytes == 40


def test_segmented_affine_manifest_rejects_component_layout_drift() -> None:
    component = NGramComponent(
        component="weight",
        name="model.safetensors",
        tensor="ngram.shard_0.weight",
        data_offset=0,
        row_bytes=16,
        data_bytes=32,
        file_size=40,
        file_sha256="1" * 64,
        dtype="U32",
        shape=(2, 4),
    )

    with pytest.raises(NGramManifestError, match="weight, scales, biases"):
        segmented_ngram_shard(
            name="ngram-shard-000",
            tensor="ngram.shard_0",
            start_row=0,
            row_count=2,
            components=(component,),
        )


def test_segmented_affine_artifact_reads_exact_rows_from_source_components(
    tmp_path: Path,
) -> None:
    weights = b"A" * 16 + b"B" * 16
    scales = b"c0" + b"c1"
    biases = b"d0" + b"d1"
    payload = weights + scales + biases
    source = tmp_path / "model.safetensors"
    source.write_bytes(payload)
    source.chmod(0o444)
    digest = hashlib.sha256(payload).hexdigest()
    components = (
        NGramComponent(
            "weight",
            source.name,
            "ngram.shard_0.weight",
            0,
            16,
            32,
            len(payload),
            digest,
            "U32",
            (2, 4),
        ),
        NGramComponent(
            "scales",
            source.name,
            "ngram.shard_0.scales",
            32,
            2,
            4,
            len(payload),
            digest,
            "BF16",
            (2, 1),
        ),
        NGramComponent(
            "biases",
            source.name,
            "ngram.shard_0.biases",
            36,
            2,
            4,
            len(payload),
            digest,
            "BF16",
            (2, 1),
        ),
    )
    shard = segmented_ngram_shard(
        name="ngram-shard-000",
        tensor="ngram.shard_0",
        start_row=0,
        row_count=2,
        components=components,
    )
    manifest = NGramManifest(
        source_repo="Vontra/example",
        source_revision="a" * 40,
        storage="affine-q4-g32",
        row_width=32,
        row_bytes=20,
        padded_rows=2,
        shards=(shard,),
    ).with_digest()

    with verify_ngram_manifest(tmp_path, manifest) as artifact:
        assert artifact.report == {"shards": 1, "rows": 2, "bytes": 40}
        assert artifact.read_row(0) == b"A" * 16 + b"c0d0"
        assert artifact.read_row(1) == b"B" * 16 + b"c1d1"


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

    with verify_ngram_manifest(tmp_path, manifest) as artifact:
        assert artifact.report["rows"] == 3


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


def test_verify_rejects_symlink_artifact_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    manifest = _tiny_manifest(real_root)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(NGramManifestError):
        verify_ngram_manifest(linked_root, manifest)


def test_verify_is_anchored_when_root_path_is_replaced_by_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _tiny_manifest(root)
    detached = tmp_path / "detached"
    original_open = os.open
    replaced = False

    def racing_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal replaced
        if path == "part-1.bin" and dir_fd is not None and not replaced:
            root.rename(detached)
            root.symlink_to(detached, target_is_directory=True)
            replaced = True
        if dir_fd is None:
            return original_open(path, flags, mode)  # type: ignore[arg-type]
        return original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(qwen4_ngram.os, "open", racing_open)
    monkeypatch.setattr(
        qwen4_ngram.os,
        "supports_dir_fd",
        frozenset((*qwen4_ngram.os.supports_dir_fd, racing_open)),
    )
    try:
        with verify_ngram_manifest(root, manifest) as artifact:
            assert replaced
            assert artifact.shards[0].pread(4, 4) == b"aabb"
    finally:
        if root.is_symlink():
            root.unlink()
        if detached.exists():
            detached.rename(root)


def test_verify_is_anchored_when_ancestor_is_replaced_by_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = tmp_path / "owner"
    root = ancestor / "artifact"
    root.mkdir(parents=True)
    manifest = _tiny_manifest(root)
    detached = tmp_path / "detached-owner"
    original_open = os.open
    replaced = False

    def racing_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal replaced
        if path == "part-1.bin" and dir_fd is not None and not replaced:
            ancestor.rename(detached)
            ancestor.symlink_to(detached, target_is_directory=True)
            replaced = True
        if dir_fd is None:
            return original_open(path, flags, mode)  # type: ignore[arg-type]
        return original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(qwen4_ngram.os, "open", racing_open)
    monkeypatch.setattr(
        qwen4_ngram.os,
        "supports_dir_fd",
        frozenset((*qwen4_ngram.os.supports_dir_fd, racing_open)),
    )
    try:
        with verify_ngram_manifest(root, manifest) as artifact:
            assert replaced
            assert artifact.read_row(0) == b"aa"
    finally:
        if ancestor.is_symlink():
            ancestor.unlink()
        if detached.exists():
            detached.rename(ancestor)


def test_verify_rejects_hardlinked_shard(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    os.link(tmp_path / "part-1.bin", tmp_path / "part-1-hardlink.bin")

    with pytest.raises(NGramManifestError, match="link"):
        verify_ngram_manifest(tmp_path, manifest)


def test_verify_rejects_writable_shard(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    (tmp_path / manifest.shards[0].name).chmod(0o644)
    artifact = None
    try:
        with pytest.raises(NGramManifestError, match="read-only|write permission"):
            artifact = verify_ngram_manifest(tmp_path, manifest)
    finally:
        if artifact is not None:
            artifact.close()


def test_verify_detects_equal_length_same_inode_mutation_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(tmp_path)
    path = tmp_path / "part-1.bin"
    path.chmod(0o444)
    (tmp_path / "part-2.bin").chmod(0o444)
    original_pread = os.pread
    mutated = False

    def mutating_pread(descriptor: int, length: int, offset: int) -> bytes:
        nonlocal mutated
        data = original_pread(descriptor, length, offset)
        if not mutated and offset == manifest.shards[0].data_offset:
            mutated = True
            path.chmod(0o644)
            with path.open("r+b") as handle:
                handle.seek(manifest.shards[0].data_offset)
                handle.write(b"zzzz")
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o444)
        return data

    monkeypatch.setattr(qwen4_ngram.os, "pread", mutating_pread)
    artifact = None
    try:
        with pytest.raises(NGramManifestError, match="identity changed"):
            artifact = verify_ngram_manifest(tmp_path, manifest)
    finally:
        if artifact is not None:
            artifact.close()


def test_verified_artifact_refuses_competing_exclusive_lock(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    for shard in manifest.shards:
        (tmp_path / shard.name).chmod(0o444)

    competitor = os.open(tmp_path / manifest.shards[0].name, os.O_RDONLY)
    try:
        with verify_ngram_manifest(tmp_path, manifest) as artifact:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    competitor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            assert artifact.read_row(0) == b"aa"
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(competitor, fcntl.LOCK_UN)
    finally:
        os.close(competitor)


def test_verify_rejects_shard_with_active_exclusive_writer_lock(
    tmp_path: Path,
) -> None:
    manifest = _tiny_manifest(tmp_path)
    writer = os.open(tmp_path / manifest.shards[0].name, os.O_RDONLY)
    try:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(NGramManifestError, match="shared lock"):
            verify_ngram_manifest(tmp_path, manifest)
    finally:
        fcntl.flock(writer, fcntl.LOCK_UN)
        os.close(writer)


def test_verified_descriptor_owners_reject_copy_and_deepcopy(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    for shard in manifest.shards:
        (tmp_path / shard.name).chmod(0o444)

    with verify_ngram_manifest(tmp_path, manifest) as artifact:
        for owner in (artifact, artifact.shards[0]):
            with pytest.raises(TypeError, match="copy"):
                copy.copy(owner)
            with pytest.raises(TypeError, match="copy"):
                copy.deepcopy(owner)


def test_verified_reads_survive_post_verification_path_replacement(
    tmp_path: Path,
) -> None:
    manifest = _tiny_manifest(tmp_path)
    artifact = verify_ngram_manifest(tmp_path, manifest)
    original = tmp_path / "part-1.bin"
    displaced = tmp_path / "displaced.bin"
    original.rename(displaced)
    original.write_bytes(b"HEADXXXX")
    try:
        assert artifact.shards[0].pread(4, 4) == b"aabb"
        identity = artifact.shards[0].identity
        assert (identity.device, identity.inode, identity.size) == (
            os.stat(displaced).st_dev,
            os.stat(displaced).st_ino,
            8,
        )
    finally:
        artifact.close()


def test_close_waits_for_concurrent_pread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _tiny_manifest(tmp_path)
    for shard in manifest.shards:
        (tmp_path / shard.name).chmod(0o444)
    artifact = verify_ngram_manifest(tmp_path, manifest)
    original_pread = os.pread
    entered = threading.Event()
    release = threading.Event()
    result: list[bytes] = []
    errors: list[Exception] = []

    def blocking_pread(descriptor: int, length: int, offset: int) -> bytes:
        entered.set()
        assert release.wait(timeout=5)
        return original_pread(descriptor, length, offset)

    def read() -> None:
        try:
            result.append(artifact.read_row(0))
        except (AssertionError, NGramManifestError) as exc:
            errors.append(exc)

    monkeypatch.setattr(qwen4_ngram.os, "pread", blocking_pread)
    reader = threading.Thread(target=read)
    closer = threading.Thread(target=artifact.close)
    reader.start()
    assert entered.wait(timeout=5)
    closer.start()
    closer.join(timeout=0.05)
    close_waited = closer.is_alive()
    release.set()
    reader.join(timeout=5)
    closer.join(timeout=5)

    assert close_waited, "close must wait for the active descriptor read"
    assert not errors
    assert result == [b"aa"]
    assert artifact.closed


def test_fd_limit_failure_is_domain_error_and_closes_partial_construction() -> None:
    with tempfile.TemporaryDirectory(
        prefix="qwen4-ngram-fd-", dir="/private/tmp"
    ) as raw:
        root = Path(raw)
        manifest = _tiny_manifest(root)
        for shard in manifest.shards:
            (root / shard.name).chmod(0o444)
        before = len(os.listdir("/dev/fd"))
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limited_soft = min(old_soft, before + 2)
        if limited_soft <= before:
            pytest.skip("process FD limit has no safe test headroom")
        limited_artifact = None
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (limited_soft, old_hard))
            with pytest.raises(NGramManifestError):
                limited_artifact = verify_ngram_manifest(root, manifest)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (old_soft, old_hard))
            if limited_artifact is not None:
                limited_artifact.close()

        with verify_ngram_manifest(root, manifest) as artifact:
            assert artifact.read_row(2) == b"cc"


def test_verify_closes_every_fd_after_mid_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(tmp_path)
    failing_path = tmp_path / "part-2.bin"
    failing_path.chmod(0o644)
    failing_path.write_bytes(b"HEADxx")
    failing_path.chmod(0o444)
    original_open = os.open
    opened: list[int] = []

    def tracking_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if dir_fd is None:
            fd = original_open(path, flags, mode)  # type: ignore[arg-type]
        else:
            fd = original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        opened.append(fd)
        return fd

    monkeypatch.setattr(qwen4_ngram.os, "open", tracking_open)
    monkeypatch.setattr(
        qwen4_ngram.os,
        "supports_dir_fd",
        frozenset((*qwen4_ngram.os.supports_dir_fd, tracking_open)),
    )
    with pytest.raises(NGramManifestError):
        verify_ngram_manifest(tmp_path, manifest)

    assert opened
    for descriptor in set(opened):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_verify_missing_root_is_domain_error(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)

    with pytest.raises(NGramManifestError):
        verify_ngram_manifest(tmp_path / "missing", manifest)


@pytest.mark.parametrize(
    "missing",
    [
        "fcntl",
        "flock",
        "lock_sh",
        "lock_ex",
        "lock_nb",
        "lock_un",
        "o_nofollow",
        "o_directory",
        "o_cloexec",
        "dir_fd_open",
        "pread",
        "close",
    ],
)
def test_verify_fails_before_path_open_when_capability_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    manifest = _tiny_manifest(tmp_path)
    if missing == "fcntl":
        monkeypatch.setattr(qwen4_ngram, "fcntl", None)
    elif missing == "flock":
        monkeypatch.delattr(qwen4_ngram.fcntl, "flock")
    elif missing == "lock_sh":
        monkeypatch.delattr(qwen4_ngram.fcntl, "LOCK_SH")
    elif missing == "lock_ex":
        monkeypatch.delattr(qwen4_ngram.fcntl, "LOCK_EX")
    elif missing == "lock_nb":
        monkeypatch.delattr(qwen4_ngram.fcntl, "LOCK_NB")
    elif missing == "lock_un":
        monkeypatch.delattr(qwen4_ngram.fcntl, "LOCK_UN")
    elif missing == "o_nofollow":
        monkeypatch.delattr(qwen4_ngram.os, "O_NOFOLLOW")
    elif missing == "o_directory":
        monkeypatch.delattr(qwen4_ngram.os, "O_DIRECTORY")
    elif missing == "o_cloexec":
        monkeypatch.delattr(qwen4_ngram.os, "O_CLOEXEC")
    elif missing == "dir_fd_open":
        monkeypatch.setattr(
            qwen4_ngram.os,
            "supports_dir_fd",
            frozenset(
                function
                for function in qwen4_ngram.os.supports_dir_fd
                if function is not qwen4_ngram.os.open
            ),
        )
    elif missing == "pread":
        monkeypatch.setattr(qwen4_ngram.os, "pread", None)
    else:
        monkeypatch.setattr(qwen4_ngram.os, "close", None)

    def unexpected_root_open(root: object) -> int:
        raise AssertionError(f"artifact root was opened: {root}")

    monkeypatch.setattr(qwen4_ngram, "_open_root_nofollow", unexpected_root_open)
    with pytest.raises(NGramManifestError, match="capabilit|required primitive"):
        verify_ngram_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LOCK_SH", 0),
        ("LOCK_NB", False),
        ("LOCK_UN", "8"),
        ("LOCK_EX", fcntl.LOCK_SH),
        ("LOCK_NB", fcntl.LOCK_SH | fcntl.LOCK_NB),
    ],
)
def test_verify_rejects_invalid_or_overlapping_lock_capabilities_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
) -> None:
    manifest = _tiny_manifest(tmp_path)
    monkeypatch.setattr(qwen4_ngram.fcntl, name, value)

    def unexpected_root_open(root: object) -> int:
        raise AssertionError(f"artifact root was opened: {root}")

    monkeypatch.setattr(qwen4_ngram, "_open_root_nofollow", unexpected_root_open)
    with pytest.raises(NGramManifestError, match="capabilit|required primitive"):
        verify_ngram_manifest(tmp_path, manifest)


@pytest.mark.parametrize("corruption", ["short", "payload"])
def test_verify_rejects_short_or_corrupt_payload(
    tmp_path: Path, corruption: str
) -> None:
    manifest = _tiny_manifest(tmp_path)
    path = tmp_path / "part-1.bin"
    path.chmod(0o644)
    if corruption == "short":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.write_bytes(b"HEADaXbb")
    path.chmod(0o444)

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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"padded_rows":' + b"9" * 5_000 + b"}", "integer"),
        (b"[" * 200 + b"0" + b"]" * 200, "depth"),
        (b'{"source_repo":"\\ud800"}', "Unicode"),
        (json.dumps([[] for _ in range(5_000)]).encode(), "collection"),
        (json.dumps({"source_repo": "x" * 5_000}).encode(), "string"),
    ],
)
def test_load_rejects_adversarial_json_with_domain_error(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "adversarial.json"
    path.write_bytes(payload)

    with pytest.raises(NGramManifestError, match=message):
        load_ngram_manifest(path, verify_digest=False)


def test_load_rejects_too_many_shards_before_materialization(tmp_path: Path) -> None:
    value = _tiny_manifest(tmp_path).to_dict()
    value.pop("digest")
    value["shards"] = [value["shards"][0] for _ in range(129)]
    path = tmp_path / "too-many-shards.json"
    path.write_text(json.dumps(value))

    with pytest.raises(NGramManifestError, match="shard|collection"):
        load_ngram_manifest(path, verify_digest=False)


def test_load_accepts_bounded_string_written_with_unicode_escapes(
    tmp_path: Path,
) -> None:
    manifest = replace(_tiny_manifest(tmp_path), source_repo="a" * 2_000).with_digest()
    value = manifest.to_dict()
    payload = json.dumps(value, separators=(",", ":"))
    literal = '"source_repo":"' + "a" * 2_000 + '"'
    escaped = '"source_repo":"' + "\\u0061" * 2_000 + '"'
    assert literal in payload
    path = tmp_path / "escaped.json"
    path.write_text(payload.replace(literal, escaped))

    assert load_ngram_manifest(path) == manifest


def test_load_missing_manifest_is_domain_error(tmp_path: Path) -> None:
    with pytest.raises(NGramManifestError):
        load_ngram_manifest(tmp_path / "missing.json")


def test_manifest_canonicalization_rejects_lone_surrogate(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)

    with pytest.raises(NGramManifestError):
        replace(manifest, source_repo="bad\ud800").with_digest()


def test_save_rejects_oversize_before_creating_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(tmp_path)
    monkeypatch.setattr(qwen4_ngram, "MAX_MANIFEST_BYTES", 1)

    def unexpected_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise AssertionError("temporary file must not be created")

    monkeypatch.setattr(qwen4_ngram.tempfile, "mkstemp", unexpected_mkstemp)
    with pytest.raises(NGramManifestError, match="byte limit"):
        save_ngram_manifest(manifest, tmp_path / "manifest.json")
