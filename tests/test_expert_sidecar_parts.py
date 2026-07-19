"""Multi-part expert banks (sharded sidecar).

An expert bank may span several files so it can be hosted where a single 89 GB
or 226 GB file cannot go.  ``pread`` never cared how many files there were;
what these tests pin down is that splitting the bank changes only *which file*
a record is read from, and that every manifest written before parts existed
still parses, re-serializes, and reads byte-for-byte as it did.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    SidecarInfo,
    SidecarPart,
    build_expert_manifest,
    build_expert_sidecar,
    load_expert_manifest,
    make_sidecar_authoritative,
    read_expert_record,
    save_expert_manifest,
    verify_expert_manifest,
)
from mtplx.expert_io import ExpertIOError, PositionalExpertReader

from test_expert_manifest import _make_checkpoint


PINNED_REAL_MANIFESTS = (
    Path("/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q2"),
    Path("/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q4"),
    Path("/Users/davidtai/.cache/huggingface/glm52-expert-only-mlx-q2"),
    Path("/Users/davidtai/.cache/huggingface/glm52-expert-only-mlx-q1t"),
)


class _ComponentDestination:
    def __init__(self, lengths: tuple[int, ...]) -> None:
        self.buffers = tuple(bytearray(length) for length in lengths)

    def record_views(self, _record: object) -> tuple[memoryview, ...]:
        return tuple(memoryview(buffer) for buffer in self.buffers)

    def payload(self) -> bytes:
        return b"".join(self.buffers)


class _RecordingReader(PositionalExpertReader):
    """Records every (file, offset) the reader actually issues."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.reads: list[tuple[str, int]] = []

    def _readv_range_into(self, relative_name, offset, views, **kwargs):  # type: ignore[no-untyped-def]
        self.reads.append((relative_name, offset))
        return super()._readv_range_into(relative_name, offset, views, **kwargs)

    def _read_range_into(self, relative_name, offset, view, **kwargs):  # type: ignore[no-untyped-def]
        self.reads.append((relative_name, offset))
        return super()._read_range_into(relative_name, offset, view, **kwargs)


def _single_part_checkpoint(root: Path) -> tuple[object, ExpertManifest]:
    spec, _expected = _make_checkpoint(root, bits=4, separate_resident=True)
    manifest = build_expert_manifest(root, spec, hash_records=True, hash_shards=True)
    return spec, build_expert_sidecar(manifest, root, root / "experts.bin")


def _split_into_parts(
    manifest: ExpertManifest,
    root: Path,
    *,
    header_bytes: int = 0,
) -> ExpertManifest:
    """Rewrite a one-file bank as one file per record, records at offset 0.

    ``header_bytes`` prepends filler to each part and declares it as
    ``data_start``, which is how a safetensors-framed part carries its header
    without moving any record's manifest offset.
    """

    assert manifest.sidecar is not None
    blob = (root / manifest.sidecar.file).read_bytes()
    parts: list[SidecarPart] = []
    records = []
    for index, record in enumerate(manifest.records):
        assert record.sidecar_offset is not None
        assert record.sidecar_length is not None
        payload = blob[
            record.sidecar_offset : record.sidecar_offset + record.sidecar_length
        ]
        name = f"experts-{index + 1:05d}-of-{len(manifest.records):05d}.bin"
        content = bytes(header_bytes) + payload
        (root / name).write_bytes(content)
        parts.append(
            SidecarPart(
                file=name,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                data_start=header_bytes,
            )
        )
        records.append(replace(record, sidecar_offset=0, part=index))
    (root / manifest.sidecar.file).unlink()
    return replace(
        manifest,
        records=tuple(records),
        sidecar=SidecarInfo(
            alignment=manifest.sidecar.alignment,
            parts=tuple(parts),
        ),
        manifest_sha256=None,
    ).with_digest()


# --------------------------------------------------------------------------
# 1. single-part manifests are untouched
# --------------------------------------------------------------------------


def test_single_part_sidecar_serializes_exactly_as_before(tmp_path: Path) -> None:
    """The scalar spelling survives a full save/load round trip unchanged."""

    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    assert manifest.sidecar is not None

    payload = manifest.sidecar.to_dict()
    assert payload == {
        "file": "experts.bin",
        "alignment": manifest.sidecar.alignment,
        "size": manifest.sidecar.size,
        "sha256": manifest.sidecar.sha256,
    }
    # No record carries a "part" key, so the record JSON is unchanged too.
    assert all("part" not in record.to_dict() for record in manifest.records)
    assert all(record.part == 0 for record in manifest.records)

    path = root / "expert-manifest.json"
    save_expert_manifest(manifest, path)
    reloaded = load_expert_manifest(path)
    assert reloaded.to_dict() == manifest.to_dict()
    assert reloaded.manifest_sha256 == manifest.manifest_sha256


def test_scalar_sidecar_accessors_and_replace_still_work(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    assert manifest.sidecar is not None

    assert manifest.sidecar.file == "experts.bin"
    assert manifest.sidecar.size == (root / "experts.bin").stat().st_size
    assert len(manifest.sidecar.parts) == 1

    # dataclasses.replace with a scalar override is the pre-parts idiom.
    tampered = replace(manifest.sidecar, sha256="f" * 64)
    assert tampered.sha256 == "f" * 64
    assert tampered.file == manifest.sidecar.file
    assert tampered.size == manifest.sidecar.size


@pytest.mark.parametrize("artifact_root", PINNED_REAL_MANIFESTS, ids=lambda p: p.name)
def test_real_manifests_parse_and_reserialize_byte_identically(
    artifact_root: Path,
) -> None:
    """The regression that matters: the shipped 48 MB manifests are unmoved."""

    path = artifact_root / "expert-manifest.json"
    if not path.is_file():
        pytest.skip(f"pinned artifact not present: {artifact_root}")

    manifest = load_expert_manifest(path, verify_digest=True)
    on_disk = json.loads(path.read_text(encoding="utf-8"))

    assert manifest.to_dict() == on_disk
    assert manifest.manifest_sha256 == manifest.with_digest().manifest_sha256
    assert manifest.sidecar is not None
    assert len(manifest.sidecar.parts) == 1
    assert manifest.sidecar.parts[0].data_start == 0
    assert all(record.part == 0 for record in manifest.records)


# --------------------------------------------------------------------------
# 2. multi-part round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_bytes", [0, 16 * 1024])
def test_multi_part_manifest_round_trips_through_disk(
    tmp_path: Path, header_bytes: int
) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root, header_bytes=header_bytes)

    assert sharded.sidecar is not None
    assert len(sharded.sidecar.parts) == len(sharded.records) == 2
    payload = sharded.sidecar.to_dict()
    assert set(payload) == {"alignment", "parts"}
    assert payload["parts"][0]["file"] == "experts-00001-of-00002.bin"
    if header_bytes:
        assert payload["parts"][0]["data_start"] == header_bytes
    else:
        assert "data_start" not in payload["parts"][0]

    path = root / "expert-manifest.json"
    save_expert_manifest(sharded, path)
    reloaded = load_expert_manifest(path)
    assert reloaded.to_dict() == sharded.to_dict()
    assert reloaded.sidecar is not None
    assert reloaded.sidecar.parts == sharded.sidecar.parts
    assert [record.part for record in reloaded.records] == [0, 1]

    verify_expert_manifest(reloaded, root, verify_sidecar_hash=True)


def test_multi_part_sidecar_refuses_the_scalar_spelling(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)
    assert sharded.sidecar is not None

    # A whole-bank file/size/sha256 has no meaning once the bank is split;
    # answering with part 0's values would be a quiet lie.
    for attribute in ("file", "size", "sha256"):
        with pytest.raises(ExpertManifestError, match="multiple parts"):
            getattr(sharded.sidecar, attribute)
    with pytest.raises(ExpertManifestError, match="cannot"):
        replace(sharded.sidecar, sha256="f" * 64)


# --------------------------------------------------------------------------
# 3. a record resolves out of a non-zero part
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_bytes", [0, 16 * 1024])
def test_record_reads_from_its_own_part(tmp_path: Path, header_bytes: int) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    expected = {
        (record.layer, record.expert): read_expert_record(
            manifest, root, record.layer, record.expert
        )
        for record in manifest.records
    }
    sharded = _split_into_parts(manifest, root, header_bytes=header_bytes)

    for record in sharded.records:
        payload = read_expert_record(sharded, root, record.layer, record.expert)
        assert payload == expected[(record.layer, record.expert)]
        assert hashlib.sha256(payload).hexdigest() == record.sha256

    # The second record genuinely lives in the second file.
    second = sharded.records[1]
    assert second.part == 1
    assert sharded.sidecar is not None
    assert sharded.sidecar.part_for(second).file == "experts-00002-of-00002.bin"
    assert sharded.sidecar.absolute_offset(second) == header_bytes


@pytest.mark.parametrize("header_bytes", [0, 16 * 1024])
def test_positional_reader_selects_the_part_fd(
    tmp_path: Path, header_bytes: int
) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    expected = {
        (record.layer, record.expert): read_expert_record(
            manifest, root, record.layer, record.expert
        )
        for record in manifest.records
    }
    sharded = _split_into_parts(manifest, root, header_bytes=header_bytes)

    with _RecordingReader(root, use_native=False) as reader:
        for record in sharded.records:
            destination = bytearray(record.logical_bytes)
            digest = reader.read_record_into(sharded, record, destination)
            assert bytes(destination) == expected[(record.layer, record.expert)]
            assert digest == record.sha256
        assert reader.reads == [
            ("experts-00001-of-00002.bin", header_bytes),
            ("experts-00002-of-00002.bin", header_bytes),
        ]


def test_record_naming_a_missing_part_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)

    stray = replace(sharded.records[0], part=7)
    with pytest.raises(ExpertManifestError, match="does not exist"):
        replace(sharded, records=(stray, sharded.records[1])).validate_structure()

    with _RecordingReader(root, use_native=False) as reader:
        with pytest.raises(ExpertIOError, match="no part 7"):
            reader.read_record_into(sharded, stray, bytearray(stray.logical_bytes))


# --------------------------------------------------------------------------
# 4. a record may never straddle a part
# --------------------------------------------------------------------------


def test_record_running_past_its_part_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)
    assert sharded.sidecar is not None

    # Shrink part 1 by one byte: its record now ends past the file, which is
    # exactly the shape a straddling record would have.
    first, second = sharded.sidecar.parts
    truncated = replace(second, size=second.size - 1)
    with pytest.raises(ExpertManifestError, match="exceeds file size"):
        replace(
            sharded,
            sidecar=SidecarInfo(
                alignment=sharded.sidecar.alignment, parts=(first, truncated)
            ),
        ).validate_structure()


def test_data_start_counts_against_the_part_size(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)
    assert sharded.sidecar is not None

    first, second = sharded.sidecar.parts
    # A header pushes the record region forward.  The file grew by the header
    # minus one byte, so the record now ends one byte past the end.
    shifted = replace(second, data_start=16 * 1024, size=16 * 1024 + second.size - 1)
    with pytest.raises(ExpertManifestError, match="exceeds file size"):
        replace(
            sharded,
            sidecar=SidecarInfo(
                alignment=sharded.sidecar.alignment, parts=(first, shifted)
            ),
        ).validate_structure()


def test_records_within_one_part_must_not_overlap(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    assert manifest.sidecar is not None

    # Both records in part 0 at offset 0 -- overlapping in the same file.
    collided = tuple(
        replace(record, sidecar_offset=0, part=0) for record in manifest.records
    )
    with pytest.raises(ExpertManifestError, match="overlap or are unsorted"):
        replace(manifest, records=collided).validate_structure()


def test_same_offset_in_different_parts_is_not_an_overlap(tmp_path: Path) -> None:
    """Non-overlap is per-part; two parts are two address spaces."""

    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)

    assert [record.sidecar_offset for record in sharded.records] == [0, 0]
    assert [record.part for record in sharded.records] == [0, 1]
    sharded.validate_structure()


def test_unaligned_record_offset_is_still_rejected_per_part(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)
    assert sharded.sidecar is not None

    nudged = replace(sharded.records[1], sidecar_offset=1)
    with pytest.raises(ExpertManifestError, match="not aligned"):
        replace(sharded, records=(sharded.records[0], nudged)).validate_structure()


def test_unaligned_part_data_start_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)
    assert sharded.sidecar is not None

    first, second = sharded.sidecar.parts
    with pytest.raises(ExpertManifestError, match="data_start is not aligned"):
        replace(
            sharded,
            sidecar=SidecarInfo(
                alignment=sharded.sidecar.alignment,
                parts=(replace(first, data_start=1), second),
            ),
        ).validate_structure()


# --------------------------------------------------------------------------
# 5. per-part digests
# --------------------------------------------------------------------------


def test_per_part_digest_mismatch_is_caught_and_names_the_part(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)
    assert sharded.sidecar is not None

    verify_expert_manifest(sharded, root, verify_sidecar_hash=True)

    first, second = sharded.sidecar.parts
    wrong = replace(second, sha256="f" * 64)
    broken = replace(
        sharded,
        sidecar=SidecarInfo(alignment=sharded.sidecar.alignment, parts=(first, wrong)),
        manifest_sha256=None,
    ).with_digest()
    with pytest.raises(ExpertManifestError, match="experts-00002-of-00002.bin"):
        verify_expert_manifest(broken, root, verify_sidecar_hash=True)


def test_per_part_size_mismatch_is_caught_on_disk(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    sharded = _split_into_parts(manifest, root)

    target = root / "experts-00002-of-00002.bin"
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(ExpertManifestError, match="sidecar size mismatch"):
        verify_expert_manifest(sharded, root)


def test_per_record_digest_survives_the_move_between_parts(tmp_path: Path) -> None:
    """Record hashes are position-independent, so sharding does not touch them."""

    root = tmp_path / "model"
    _spec, manifest = _single_part_checkpoint(root)
    before = {
        (record.layer, record.expert): record.sha256 for record in manifest.records
    }
    sharded = _split_into_parts(manifest, root)

    after = {(record.layer, record.expert): record.sha256 for record in sharded.records}
    assert after == before
    verify_expert_manifest(sharded, root, verify_records=True)


def test_authoritative_manifest_takes_one_sidecar_shard_per_part(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(
        root, bits=2, key="hy3-expert-q2", separate_resident=True
    )
    manifest = build_expert_manifest(root, spec, hash_records=True, hash_shards=True)
    with_sidecar = build_expert_sidecar(manifest, root, root / "experts.bin")
    sharded = _split_into_parts(with_sidecar, root)

    authoritative = make_sidecar_authoritative(sharded, spec)
    sidecar_shards = [
        shard for shard in authoritative.shards if shard.kind == "sidecar"
    ]
    assert [shard.name for shard in sidecar_shards] == [
        "experts-00001-of-00002.bin",
        "experts-00002-of-00002.bin",
    ]
    # Each record's components point at its own part.
    assert authoritative.sidecar is not None
    for record in authoritative.records:
        part = authoritative.sidecar.part_for(record)
        assert {segment.shard for segment in record.segments} == {part.file}
    authoritative.validate_structure()

    # A part with no sidecar shard behind it breaks the one-per-part rule.
    orphan = SidecarPart(file="experts-00003.bin", size=4096, sha256="a" * 64)
    with pytest.raises(ExpertManifestError, match="one sidecar shard per"):
        replace(
            authoritative,
            sidecar=SidecarInfo(
                alignment=authoritative.sidecar.alignment,
                parts=(*authoritative.sidecar.parts, orphan),
            ),
        ).validate_structure()


# --------------------------------------------------------------------------
# 6. batching never coalesces across a part boundary
# --------------------------------------------------------------------------


def _batch_record(
    *, expert: int, part: int, offset: int, lengths: tuple[int, ...]
) -> SimpleNamespace:
    return SimpleNamespace(
        layer=1,
        expert=expert,
        logical_bytes=sum(lengths),
        segments=tuple(SimpleNamespace(length=length) for length in lengths),
        sidecar_offset=offset,
        sidecar_length=sum(lengths),
        sha256=None,
        part=part,
    )


def _two_part_bank(tmp_path: Path) -> SimpleNamespace:
    (tmp_path / "part-a.bin").write_bytes(b"AAAABBBB")
    (tmp_path / "part-b.bin").write_bytes(b"CCCCDDDD")
    return SimpleNamespace(
        sidecar=SimpleNamespace(
            alignment=1,
            parts=(
                SimpleNamespace(file="part-a.bin", data_start=0),
                SimpleNamespace(file="part-b.bin", data_start=0),
            ),
        )
    )


def test_batch_does_not_coalesce_across_a_part_boundary(tmp_path: Path) -> None:
    """Offset adjacency means nothing across files: one preadv per part."""

    manifest = _two_part_bank(tmp_path)
    # Offsets 0..4 and 4..8 look perfectly adjacent -- but they are in
    # different files, so coalescing them would read part-a twice.
    first = _batch_record(expert=1, part=0, offset=0, lengths=(2, 2))
    second = _batch_record(expert=2, part=1, offset=4, lengths=(2, 2))
    first_destination = _ComponentDestination((2, 2))
    second_destination = _ComponentDestination((2, 2))

    with _RecordingReader(tmp_path, use_native=False) as reader:
        reader.read_component_records_into(
            manifest,
            ((first, first_destination), (second, second_destination)),
            verify_hash=False,
        )

    assert first_destination.payload() == b"AAAA"
    assert second_destination.payload() == b"DDDD"
    assert reader.reads == [("part-a.bin", 0), ("part-b.bin", 4)]


def test_batch_still_coalesces_adjacent_records_inside_one_part(
    tmp_path: Path,
) -> None:
    manifest = _two_part_bank(tmp_path)
    first = _batch_record(expert=1, part=0, offset=0, lengths=(2, 2))
    second = _batch_record(expert=2, part=0, offset=4, lengths=(2, 2))
    first_destination = _ComponentDestination((2, 2))
    second_destination = _ComponentDestination((2, 2))

    with _RecordingReader(tmp_path, use_native=False) as reader:
        reader.read_component_records_into(
            manifest,
            ((first, first_destination), (second, second_destination)),
            verify_hash=False,
        )

    assert first_destination.payload() == b"AAAA"
    assert second_destination.payload() == b"BBBB"
    assert reader.reads == [("part-a.bin", 0)]


def test_batch_orders_by_part_then_offset(tmp_path: Path) -> None:
    """A low offset in a later part must not sort ahead of an earlier part."""

    manifest = _two_part_bank(tmp_path)
    late = _batch_record(expert=2, part=1, offset=0, lengths=(4,))
    early = _batch_record(expert=1, part=0, offset=4, lengths=(4,))
    late_destination = _ComponentDestination((4,))
    early_destination = _ComponentDestination((4,))

    with _RecordingReader(tmp_path, use_native=False) as reader:
        digests = reader.read_component_records_into(
            manifest,
            ((late, late_destination), (early, early_destination)),
            verify_hash=False,
        )

    assert digests == ("unverified", "unverified")
    assert early_destination.payload() == b"BBBB"
    assert late_destination.payload() == b"CCCC"
    assert reader.reads == [("part-a.bin", 4), ("part-b.bin", 0)]


def test_batch_honours_part_data_start(tmp_path: Path) -> None:
    (tmp_path / "framed.bin").write_bytes(b"HEADER!!" + b"PAYLOAD.")
    manifest = SimpleNamespace(
        sidecar=SimpleNamespace(
            alignment=1,
            parts=(SimpleNamespace(file="framed.bin", data_start=8),),
        )
    )
    record = _batch_record(expert=1, part=0, offset=0, lengths=(4, 4))
    destination = _ComponentDestination((4, 4))

    with _RecordingReader(tmp_path, use_native=False) as reader:
        reader.read_component_records_into(
            manifest, ((record, destination),), verify_hash=False
        )

    assert destination.payload() == b"PAYLOAD."
    assert reader.reads == [("framed.bin", 8)]


def test_single_file_sidecar_without_parts_still_reads(tmp_path: Path) -> None:
    """The duck-typed single-file sidecar keeps working untouched."""

    (tmp_path / "legacy.bin").write_bytes(b"abcdefgh")
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="legacy.bin"))
    record = _batch_record(expert=1, part=0, offset=0, lengths=(4, 4))
    destination = _ComponentDestination((4, 4))

    with _RecordingReader(tmp_path, use_native=False) as reader:
        reader.read_component_records_into(
            manifest, ((record, destination),), verify_hash=False
        )

    assert destination.payload() == b"abcdefgh"
    assert reader.reads == [("legacy.bin", 0)]


def test_nonzero_part_against_a_single_file_sidecar_fails_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "legacy.bin").write_bytes(b"abcdefgh")
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="legacy.bin"))
    record = _batch_record(expert=1, part=3, offset=0, lengths=(4, 4))

    with _RecordingReader(tmp_path, use_native=False) as reader:
        with pytest.raises(ExpertIOError, match="single-file"):
            reader.read_component_records_into(
                manifest,
                ((record, _ComponentDestination((4, 4))),),
                verify_hash=False,
            )
