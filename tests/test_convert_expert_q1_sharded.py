from __future__ import annotations

import hashlib
from pathlib import Path

from mtplx.expert_q1 import Q1Manifest, Q1Record, Q1Segment, load_q1_manifest
from scripts.convert_expert_q1_sharded import contiguous_ranges, merge_q1_shards


def _shard(root: Path, name: str, *, layer: int, payload: bytes) -> Path:
    root.mkdir(parents=True)
    bin_path = root / "experts-q1-t158.bin"
    bin_path.write_bytes(payload)
    segment = Q1Segment(
        component="gate_proj.packed",
        dtype="U8",
        shape=(len(payload),),
        offset=0,
        length=len(payload),
    )
    manifest = Q1Manifest(
        format="mtplx-expert-q1-v1",
        model_key="source-q1t158",
        codec="t158",
        group_size=64,
        file=bin_path.name,
        source_model_key="source",
        source_manifest_sha256="a" * 64,
        records=(
            Q1Record(
                layer=layer,
                expert=0,
                offset=0,
                length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                segments=(segment,),
            ),
        ),
        path=root / name,
    )
    manifest.path.write_text(__import__("json").dumps(manifest.to_json()))
    return manifest.path


def test_contiguous_ranges_cover_layers_once() -> None:
    ranges = contiguous_ranges((3, 4, 5, 7, 9), workers=3)
    assert ranges == ((3, 4), (5,), (7, 9))
    assert tuple(layer for chunk in ranges for layer in chunk) == (3, 4, 5, 7, 9)


def test_merge_q1_shards_concatenates_and_rebases_offsets(tmp_path: Path) -> None:
    first = _shard(tmp_path / "s0", "manifest.json", layer=3, payload=b"abc")
    second = _shard(tmp_path / "s1", "manifest.json", layer=4, payload=b"defgh")
    output_bin = tmp_path / "artifact" / "experts-q1-t158.bin"
    output_manifest = tmp_path / "artifact" / "expert-manifest-q1-t158.json"

    merged = merge_q1_shards(
        (first, second),
        output_bin=output_bin,
        output_manifest=output_manifest,
    )

    assert output_bin.read_bytes() == b"abcdefgh"
    assert [
        (record.layer, record.offset, record.length) for record in merged.records
    ] == [
        (3, 0, 3),
        (4, 3, 5),
    ]
    assert load_q1_manifest(output_manifest) == merged
