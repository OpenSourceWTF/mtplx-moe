"""Round-trip tests for the expert-bank sharding converter.

The converter is only worth anything if the parts-aware reader can read what
it writes, so these build a real bank with the repo's own checkpoint helpers,
shard it, and read every record back through the real reader.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from mtplx.expert_manifest import (
    DEFAULT_ALIGNMENT,
    build_expert_manifest,
    build_expert_sidecar,
    load_expert_manifest,
)
from mtplx.expert_io import PositionalExpertReader
from scripts.shard_expert_bank import (
    ShardConversionError,
    build_part_header,
    convert,
    part_filename,
    plan_parts,
)
from test_expert_manifest import _make_checkpoint


def _bank(root: Path):
    """A real single-file bank, built the way the repo builds them."""

    spec, _expected = _make_checkpoint(root, bits=4, separate_resident=True)
    manifest = build_expert_manifest(root, spec, hash_records=True, hash_shards=True)
    return build_expert_sidecar(manifest, root, root / "experts.bin")


class _Sized:
    def __init__(self, n: int) -> None:
        self.sidecar_length = n


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def test_records_are_never_split_across_parts() -> None:
    groups = plan_parts([_Sized(100) for _ in range(10)], part_bytes=250)
    assert [len(g) for g in groups] == [2, 2, 2, 2, 2]


def test_part_too_small_for_a_record_is_an_error() -> None:
    with pytest.raises(ShardConversionError, match="cannot hold the largest record"):
        plan_parts([_Sized(4096)], part_bytes=1024)


def test_part_filename_is_hf_conventional() -> None:
    assert part_filename("experts", 0, 6) == "experts-00001-of-00006.safetensors"
    assert part_filename("experts", 5, 6) == "experts-00006-of-00006.safetensors"


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def test_header_padding_makes_every_record_page_aligned(tmp_path) -> None:
    """The whole alignment story: pad the header, get aligned records free."""

    manifest = _bank(tmp_path / "src")
    header, placements = build_part_header(manifest.records, DEFAULT_ALIGNMENT)
    assert len(header) % DEFAULT_ALIGNMENT == 0
    for _record, offset in placements:
        assert offset % DEFAULT_ALIGNMENT == 0


def test_header_is_parseable_safetensors(tmp_path) -> None:
    manifest = _bank(tmp_path / "src")
    header, _ = build_part_header(manifest.records, DEFAULT_ALIGNMENT)
    length = struct.unpack("<Q", header[:8])[0]
    parsed = json.loads(header[8 : 8 + length])
    assert parsed
    for entry in parsed.values():
        assert entry["data_offsets"][1] > entry["data_offsets"][0]
        assert entry["dtype"] and entry["shape"]


# --------------------------------------------------------------------------
# round trip — the test that matters
# --------------------------------------------------------------------------


def test_sharded_bank_round_trips_through_the_real_reader(tmp_path) -> None:
    root = tmp_path / "src"
    manifest = _bank(root)
    out = tmp_path / "sharded"

    # one record per part, so every record is read from a distinct part
    smallest = min(r.sidecar_length for r in manifest.records)
    sharded = convert(manifest, root, out, part_bytes=smallest, progress=False)
    (out / "expert-manifest.json").write_text(json.dumps(sharded.to_dict()))

    assert len(sharded.sidecar.parts) == len(manifest.records)
    assert {r.part for r in sharded.records} == set(range(len(manifest.records)))

    reloaded = load_expert_manifest(out / "expert-manifest.json")
    reader = PositionalExpertReader(out)
    try:
        for original in manifest.records:
            moved = reloaded.record(original.layer, original.expert)
            buffer = bytearray(moved.sidecar_length)
            # verify_hash=True makes the reader itself check the per-record
            # digest, so a wrong part or offset fails here rather than silently
            # returning the wrong expert.
            reader.read_record_into(reloaded, moved, buffer, verify_hash=True)
            assert hashlib.sha256(bytes(buffer)).hexdigest() == original.sha256
    finally:
        reader.close()


def test_per_record_digests_survive_the_move(tmp_path) -> None:
    """Per-record sha256 is position-independent — why sharding is safe at all."""

    root = tmp_path / "src"
    manifest = _bank(root)
    sharded = convert(
        manifest, root, tmp_path / "s", part_bytes=1 << 30, progress=False
    )
    assert {(r.layer, r.expert): r.sha256 for r in manifest.records} == {
        (r.layer, r.expert): r.sha256 for r in sharded.records
    }


def test_parts_declare_safetensors_kind_not_sidecar(tmp_path) -> None:
    """A framed part is a safetensors shard; only a raw .bin is a sidecar."""

    root = tmp_path / "src"
    manifest = _bank(root)
    sharded = convert(
        manifest, root, tmp_path / "s", part_bytes=1 << 30, progress=False
    )
    assert {s.kind for s in sharded.shards} == {"safetensors"}
    assert all(s.header_bytes > 0 for s in sharded.shards)


def test_each_part_carries_its_own_size_and_digest(tmp_path) -> None:
    root = tmp_path / "src"
    manifest = _bank(root)
    out = tmp_path / "s"
    sharded = convert(manifest, root, out, part_bytes=1 << 30, progress=False)
    for part in sharded.sidecar.parts:
        blob = (out / part.file).read_bytes()
        assert len(blob) == part.size
        assert hashlib.sha256(blob).hexdigest() == part.sha256


def test_refuses_to_shard_an_already_sharded_bank(tmp_path) -> None:
    root = tmp_path / "src"
    manifest = _bank(root)
    out = tmp_path / "s"
    smallest = min(r.sidecar_length for r in manifest.records)
    sharded = convert(manifest, root, out, part_bytes=smallest, progress=False)
    assert len(sharded.sidecar.parts) > 1
    with pytest.raises(ShardConversionError, match="already sharded"):
        convert(sharded, out, tmp_path / "s2", progress=False)


def test_source_artifact_is_never_modified(tmp_path) -> None:
    root = tmp_path / "src"
    manifest = _bank(root)
    before = hashlib.sha256((root / "experts.bin").read_bytes()).hexdigest()
    convert(manifest, root, tmp_path / "s", part_bytes=1 << 30, progress=False)
    after = hashlib.sha256((root / "experts.bin").read_bytes()).hexdigest()
    assert before == after
