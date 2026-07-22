from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mtplx.expert_q1 import Q1Manifest, Q1Record, Q1Segment
from mtplx.expert_rans import decode_bank_reference, deserialize_component


_COMPONENTS = (
    "gate_proj.packed",
    "gate_proj.scales",
    "up_proj.packed",
    "up_proj.scales",
    "down_proj.packed",
    "down_proj.scales",
)


def _q1_artifact(
    tmp_path: Path,
) -> tuple[Q1Manifest, dict[tuple[int, str], np.ndarray]]:
    rng = np.random.default_rng(54001)
    expert_count = 2
    out_dim = 64
    record_payloads: list[bytes] = []
    records: list[Q1Record] = []
    expected: dict[tuple[int, str], np.ndarray] = {}
    record_cursor = 0
    for expert in range(expert_count):
        payload = bytearray()
        segments: list[Q1Segment] = []
        component_cursor = 0
        for component in _COMPONENTS:
            if component.endswith(".packed"):
                value = rng.integers(
                    0,
                    243,
                    size=(out_dim, 13),
                    dtype=np.uint8,
                )
                dtype = "U8"
            else:
                value = rng.integers(
                    1,
                    np.iinfo(np.uint16).max,
                    size=(out_dim, 1),
                    dtype=np.uint16,
                )
                dtype = "U16"
            raw = value.tobytes()
            segments.append(
                Q1Segment(
                    component=component,
                    dtype=dtype,
                    shape=value.shape,
                    offset=component_cursor,
                    length=len(raw),
                )
            )
            payload.extend(raw)
            component_cursor += len(raw)
            expected[(expert, component)] = value
        blob = bytes(payload)
        records.append(
            Q1Record(
                layer=3,
                expert=expert,
                offset=record_cursor,
                length=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
                segments=tuple(segments),
            )
        )
        record_payloads.append(blob)
        record_cursor += len(blob)
    source_bin = tmp_path / "existing-q1t.bin"
    source_bin.write_bytes(b"".join(record_payloads))
    source_manifest_path = tmp_path / "existing-q1t.json"
    source_manifest_path.write_text("{}")
    return (
        Q1Manifest(
            format="mtplx-expert-q1-v1",
            model_key="glm52-expert-q1t",
            codec="t158",
            group_size=64,
            file=source_bin.name,
            source_model_key="glm52-expert-q2",
            source_manifest_sha256="a" * 64,
            records=tuple(records),
            path=source_manifest_path,
        ),
        expected,
    )


def test_write_fused_artifact_uses_six_separate_aligned_containers(
    tmp_path: Path,
) -> None:
    from mtplx.glm52_q1t_rans_artifact import (
        COMPONENT_ALIGNMENT,
        FusedRansArtifactError,
        load_glm52_q1t_fused_rans_manifest,
        write_glm52_q1t_fused_rans_artifact,
    )

    source, expected = _q1_artifact(tmp_path)
    output_bin = tmp_path / "fused" / "glm52-q1t-fused-rans.bin"
    output_manifest = tmp_path / "fused" / "glm52-q1t-fused-rans.json"
    manifest = write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=output_bin,
        output_manifest=output_manifest,
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
    )

    assert source.bin_path().read_bytes() == (tmp_path / "existing-q1t.bin").read_bytes()
    assert manifest.format == "mtplx-glm52-q1t-fused-rans-v1"
    assert manifest.model_key == "glm52-expert-q1t"
    assert manifest.codec == "rans32x-v1"
    assert manifest.source_codec == "t158"
    assert manifest.source_manifest_sha256 == "b" * 64
    assert manifest.source_q1_parent_manifest_sha256 == "a" * 64
    assert manifest.output_tile == 32
    assert manifest.routed_layers == (3,)
    assert tuple(component.component for component in manifest.layers[0].components) == (
        _COMPONENTS
    )
    for component in manifest.layers[0].components:
        assert component.offset % COMPONENT_ALIGNMENT == 0
        assert component.mapped_length % COMPONENT_ALIGNMENT == 0
        assert component.length <= component.mapped_length
        assert component.payload_offset > component.directory_offset
        assert component.payload_length > 0
        assert component.guard_bytes == 8
        blob = output_bin.read_bytes()[
            component.offset : component.offset + component.length
        ]
        assert hashlib.sha256(blob).hexdigest() == component.sha256
        container = deserialize_component(blob)
        decoded = decode_bank_reference(
            type(
                "Streams",
                (),
                {
                    "payload": container.payload,
                    "directory": container.directory.reshape(
                        container.expert_count, container.lanes
                    ),
                    "seg_len": container.seg_len,
                    "per_lane": container.per_lane,
                    "expert_count": container.expert_count,
                    "lanes": container.lanes,
                },
            )(),
            container.table,
        )
        rows = np.stack(
            [expected[(expert, component.component)] for expert in range(2)]
        )
        row_bytes = rows.dtype.itemsize * rows.shape[-1]
        expected_segments = rows.view(np.uint8).reshape(
            2 * (rows.shape[1] // 32),
            32 * row_bytes,
        )
        assert np.array_equal(decoded, expected_segments)

    assert load_glm52_q1t_fused_rans_manifest(output_manifest) == manifest
    with pytest.raises(FusedRansArtifactError, match="already exists"):
        write_glm52_q1t_fused_rans_artifact(
            source,
            output_bin=output_bin,
            output_manifest=output_manifest,
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="b" * 64,
        )


def test_uniform_packed_artifact_keeps_scales_compressed_and_roundtrips(
    tmp_path: Path,
) -> None:
    from mtplx.expert_rans import deserialize_component
    from mtplx.glm52_q1t_rans_artifact import (
        FUSED_RANS_UNIFORM_PACKED_CODEC,
        write_glm52_q1t_fused_rans_artifact,
    )

    source, expected = _q1_artifact(tmp_path)
    manifest = write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=tmp_path / "uniform" / "fused.bin",
        output_manifest=tmp_path / "uniform" / "fused.json",
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
        uniform_packed=True,
    )

    assert manifest.codec == FUSED_RANS_UNIFORM_PACKED_CODEC
    binary = manifest.bin_path().read_bytes()
    for component in manifest.layers[0].components:
        container = deserialize_component(
            binary[component.offset : component.offset + component.length]
        )
        decoded = decode_bank_reference(
            type(
                "Streams",
                (),
                {
                    "payload": container.payload,
                    "directory": container.directory.reshape(
                        container.expert_count, container.lanes
                    ),
                    "seg_len": container.seg_len,
                    "per_lane": container.per_lane,
                    "expert_count": container.expert_count,
                    "lanes": container.lanes,
                },
            )(),
            container.table,
        )
        if component.component.endswith(".packed"):
            assert np.array_equal(
                container.table.freq,
                np.full(256, 16, dtype=np.uint32),
            )
            assert container.payload.size == (
                component.record_count * component.lanes * (component.per_lane + 4)
            )
        else:
            assert not np.array_equal(
                container.table.freq,
                np.full(256, 16, dtype=np.uint32),
            )
        rows = np.stack(
            [expected[(expert, component.component)] for expert in range(2)]
        )
        row_bytes = rows.dtype.itemsize * rows.shape[-1]
        expected_segments = rows.view(np.uint8).reshape(
            2 * (rows.shape[1] // 32),
            32 * row_bytes,
        )
        assert np.array_equal(decoded, expected_segments)


def test_uniform_packed_encoder_is_byte_identical_to_generic_rans() -> None:
    from mtplx.expert_rans import encode_bank, table_from_freq
    from mtplx.glm52_q1t_rans_artifact import _encode_uniform_bank

    rng = np.random.default_rng(51912)
    segments = rng.integers(0, 256, size=(7, 32 * 41), dtype=np.uint8)
    table = table_from_freq(np.full(256, 16, dtype=np.uint32))

    expected = encode_bank(segments, table)
    actual = _encode_uniform_bank(segments)

    assert np.array_equal(actual.payload, expected.payload)
    assert np.array_equal(actual.directory, expected.directory)
    assert actual.seg_len == expected.seg_len
    assert actual.per_lane == expected.per_lane
    assert actual.expert_count == expected.expert_count
    assert actual.lanes == expected.lanes


def test_fused_writer_discards_completed_source_layer_pages() -> None:
    import mmap

    from mtplx.glm52_q1t_rans_artifact import _discard_source_pages

    advice = []
    source_map = SimpleNamespace(
        _mmap=SimpleNamespace(madvise=advice.append),
    )

    _discard_source_pages(source_map)

    assert advice == [mmap.MADV_DONTNEED]


def test_fused_writer_reopens_source_mapping_between_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mtplx.glm52_q1t_rans_artifact as artifact

    source, _expected = _q1_artifact(tmp_path)
    source_bytes = source.bin_path().read_bytes()
    source.bin_path().write_bytes(source_bytes + source_bytes)
    source = replace(
        source,
        records=source.records
        + tuple(
            replace(record, layer=4, offset=record.offset + len(source_bytes))
            for record in source.records
        ),
    )
    original_memmap = artifact.np.memmap
    opened: list[np.memmap] = []

    def tracked_memmap(*args, **kwargs):
        source_map = original_memmap(*args, **kwargs)
        opened.append(source_map)
        return source_map

    monkeypatch.setattr(artifact.np, "memmap", tracked_memmap)
    artifact.write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=tmp_path / "fused.bin",
        output_manifest=tmp_path / "fused.json",
        layers=(3, 4),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
    )

    assert len(opened) == 2
    assert all(source_map._mmap.closed for source_map in opened)

def test_fused_artifact_rejects_non_glm_q1t_source(tmp_path: Path) -> None:
    from mtplx.glm52_q1t_rans_artifact import (
        FusedRansArtifactError,
        write_glm52_q1t_fused_rans_artifact,
    )

    source, _expected = _q1_artifact(tmp_path)
    wrong = Q1Manifest(
        **{
            **source.__dict__,
            "model_key": "hy3-expert-q1t158",
        }
    )
    with pytest.raises(FusedRansArtifactError, match="glm52-expert-q1t"):
        write_glm52_q1t_fused_rans_artifact(
            wrong,
            output_bin=tmp_path / "wrong.bin",
            output_manifest=tmp_path / "wrong.json",
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="b" * 64,
        )


def test_fused_artifact_component_chunking_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.glm52_q1t_rans_artifact as artifact

    source, _expected = _q1_artifact(tmp_path)
    monkeypatch.setattr(artifact, "_ENCODE_RECORD_CHUNK", 1)
    chunked = artifact.write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=tmp_path / "chunked" / "fused.bin",
        output_manifest=tmp_path / "chunked" / "fused.json",
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(artifact, "_ENCODE_RECORD_CHUNK", 2048)
    single_chunk = artifact.write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=tmp_path / "single" / "fused.bin",
        output_manifest=tmp_path / "single" / "fused.json",
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
    )

    assert chunked.file_sha256 == single_chunk.file_sha256
    assert chunked.bin_path().read_bytes() == single_chunk.bin_path().read_bytes()


def test_fused_artifact_resume_keeps_completed_component_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mtplx.glm52_q1t_rans_artifact as artifact

    source, _expected = _q1_artifact(tmp_path)
    output_bin = tmp_path / "fused" / "glm52-q1t-fused-rans.bin"
    output_manifest = tmp_path / "fused" / "glm52-q1t-fused-rans.json"
    original = artifact._encode_component
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact, "_encode_component", fail_after_first)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        artifact.write_glm52_q1t_fused_rans_artifact(
            source,
            output_bin=output_bin,
            output_manifest=output_manifest,
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="b" * 64,
            resume=True,
        )

    partial = output_bin.with_name(output_bin.name + ".partial")
    progress = output_manifest.with_name(output_manifest.name + ".progress")
    assert partial.stat().st_size > 0
    first_size = partial.stat().st_size
    assert progress.exists()

    monkeypatch.setattr(artifact, "_encode_component", original)
    manifest = artifact.write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=output_bin,
        output_manifest=output_manifest,
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
        resume=True,
    )

    assert manifest.file_bytes > first_size
    assert output_bin.exists()
    assert output_manifest.exists()
    assert not partial.exists()
    assert not progress.exists()


def test_fused_artifact_resume_rehashes_completed_component_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mtplx.glm52_q1t_rans_artifact as artifact

    source, _expected = _q1_artifact(tmp_path)
    output_bin = tmp_path / "fused" / "glm52-q1t-fused-rans.bin"
    output_manifest = tmp_path / "fused" / "glm52-q1t-fused-rans.json"
    original = artifact._encode_component
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact, "_encode_component", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        artifact.write_glm52_q1t_fused_rans_artifact(
            source,
            output_bin=output_bin,
            output_manifest=output_manifest,
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="b" * 64,
            resume=True,
        )

    partial = output_bin.with_name(output_bin.name + ".partial")
    with partial.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))

    monkeypatch.setattr(artifact, "_encode_component", original)
    with pytest.raises(artifact.FusedRansArtifactError, match="hash mismatch"):
        artifact.write_glm52_q1t_fused_rans_artifact(
            source,
            output_bin=output_bin,
            output_manifest=output_manifest,
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="b" * 64,
            resume=True,
        )


def test_fused_artifact_resume_finalizes_published_binary_without_reencoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mtplx.glm52_q1t_rans_artifact as artifact

    source, _expected = _q1_artifact(tmp_path)
    output_bin = tmp_path / "fused" / "glm52-q1t-fused-rans.bin"
    output_manifest = tmp_path / "fused" / "glm52-q1t-fused-rans.json"
    real_replace = artifact.os.replace

    def fail_manifest_publish(source_path, destination_path):
        if Path(destination_path) == output_manifest:
            raise RuntimeError("synthetic manifest publish interruption")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(artifact.os, "replace", fail_manifest_publish)
    with pytest.raises(RuntimeError, match="manifest publish interruption"):
        artifact.write_glm52_q1t_fused_rans_artifact(
            source,
            output_bin=output_bin,
            output_manifest=output_manifest,
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="b" * 64,
            resume=True,
        )

    assert output_bin.exists()
    assert not output_manifest.exists()
    assert not output_bin.with_name(output_bin.name + ".partial").exists()

    monkeypatch.setattr(artifact.os, "replace", real_replace)
    monkeypatch.setattr(
        artifact,
        "_encode_component",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed artifact must not be re-encoded")
        ),
    )
    manifest = artifact.write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=output_bin,
        output_manifest=output_manifest,
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
        resume=True,
    )

    assert output_manifest.exists()
    assert manifest.file_sha256 == hashlib.sha256(output_bin.read_bytes()).hexdigest()


def test_fused_artifact_requires_authoritative_q1_manifest_digest(
    tmp_path: Path,
) -> None:
    from mtplx.glm52_q1t_rans_artifact import (
        FusedRansArtifactError,
        write_glm52_q1t_fused_rans_artifact,
    )

    source, _expected = _q1_artifact(tmp_path)
    with pytest.raises(FusedRansArtifactError, match="authoritative"):
        write_glm52_q1t_fused_rans_artifact(
            source,
            output_bin=tmp_path / "fused.bin",
            output_manifest=tmp_path / "fused.json",
            layers=(3,),
            expected_expert_count=2,
            source_expert_manifest_sha256="not-a-digest",
        )


def test_construction_integrity_rejects_corrupt_compressed_component(
    tmp_path: Path,
) -> None:
    from mtplx.glm52_q1t_rans_artifact import (
        write_glm52_q1t_fused_rans_artifact,
    )
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansConstructionError,
        verify_glm52_q1t_fused_rans_artifact,
    )

    source, _expected = _q1_artifact(tmp_path)
    output_bin = tmp_path / "fused" / "glm52-q1t-fused-rans.bin"
    manifest = write_glm52_q1t_fused_rans_artifact(
        source,
        output_bin=output_bin,
        output_manifest=tmp_path / "fused" / "glm52-q1t-fused-rans.json",
        layers=(3,),
        expected_expert_count=2,
        source_expert_manifest_sha256="b" * 64,
    )
    assert verify_glm52_q1t_fused_rans_artifact(manifest) >= 0

    component = manifest.layers[0].components[0]
    with output_bin.open("r+b") as handle:
        handle.seek(component.offset + component.payload_offset)
        value = handle.read(1)
        handle.seek(component.offset + component.payload_offset)
        handle.write(bytes([value[0] ^ 1]))

    with pytest.raises(Glm52Q1TFusedRansConstructionError, match="hash mismatch"):
        verify_glm52_q1t_fused_rans_artifact(manifest)
