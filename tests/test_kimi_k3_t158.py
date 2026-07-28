from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

import mtplx.kimi_k3_t158 as t158_module
from mtplx.expert_shadow import decode_t158, encode_t158
from mtplx.kimi_k3_gguf import (
    GGML_TYPE_BF16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    GGUFFile,
    GGUFFileIdentity,
    GGUFTensor,
    KimiK3Inventory,
)
from mtplx.kimi_k3_t158 import (
    KIMI_K3_T158_LAYER_BYTES,
    KIMI_K3_T158_RECORD_BYTES,
    KimiK3Layout,
    copy_resident_safetensors,
    convert_layer,
    encode_expert_record,
)


def test_production_layout_has_exact_measured_part_geometry() -> None:
    layout = KimiK3Layout()

    assert layout.record_bytes == KIMI_K3_T158_RECORD_BYTES == 7_741_440
    assert layout.layer_bytes == KIMI_K3_T158_LAYER_BYTES == 6_936_330_240


def _source(
    path: Path,
    tensors: tuple[GGUFTensor, ...],
    payload: bytes,
) -> GGUFFile:
    path.write_bytes(payload)
    status = path.stat()
    return GGUFFile(
        path=path,
        version=3,
        alignment=1,
        data_offset=0,
        file_size=len(payload),
        metadata={},
        tensors=tensors,
        identity=GGUFFileIdentity(
            device=status.st_dev,
            inode=status.st_ino,
            size=status.st_size,
            mtime_ns=status.st_mtime_ns,
            ctime_ns=status.st_ctime_ns,
        ),
    )


def _q2_block(*, quant: int = 0, d: float = 1.0, dmin: float = 0.0) -> bytes:
    if not 0 <= quant <= 3:
        raise ValueError("quant must be a two-bit value")
    scales = bytes([0x01] * 16)
    packed = quant | (quant << 2) | (quant << 4) | (quant << 6)
    return scales + bytes([packed] * 64) + struct.pack("<ee", d, dmin)


def _tiny_layer_source(
    root: Path,
    *,
    revision: str = "tiny-revision",
) -> tuple[GGUFFile, KimiK3Inventory, KimiK3Layout]:
    expert_count = 2
    # Reversed GGUF dims give (expert=2, out=1, in=256): one distinct Q2_K
    # block per expert and six blocks across the three merged tensors.
    dims = (256, 1, expert_count)
    projection_payloads = (
        _q2_block(quant=0, d=1.0) + _q2_block(quant=1, d=1.0),
        _q2_block(quant=2, d=1.0) + _q2_block(quant=3, d=1.0),
        _q2_block(quant=1, d=2.0) + _q2_block(quant=2, d=3.0),
    )
    tensor_bytes = len(projection_payloads[0])
    payload = b"".join(projection_payloads)
    tensors = (
        GGUFTensor("blk.1.ffn_gate_exps.weight", dims, GGML_TYPE_Q2_K, 0),
        GGUFTensor(
            "blk.1.ffn_up_exps.weight",
            dims,
            GGML_TYPE_Q2_K,
            tensor_bytes,
        ),
        GGUFTensor(
            "blk.1.ffn_down_exps.weight",
            dims,
            GGML_TYPE_Q2_K,
            tensor_bytes * 2,
        ),
    )
    source = _source(root / "source.gguf", tensors, payload)
    inventory = KimiK3Inventory(
        revision=revision,
        files=(source,),
        expert_tensors=tensors,
        resident_tensors=(),
        layers=(1,),
    )
    return (
        source,
        inventory,
        KimiK3Layout(
            expert_count=expert_count,
            layer_count=1,
            gate_shape=(1, 256),
            up_shape=(1, 256),
            down_shape=(1, 256),
        ),
    )


def _read_safetensors(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    header_length = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 : 8 + header_length]), raw[8 + header_length :]


def _layer_paths(output: Path, layout: KimiK3Layout) -> tuple[Path, Path, Path]:
    final = output / (f"experts-t158-layer-001-of-{layout.layer_count:03d}.bin")
    return (
        final,
        final.with_name(final.name + ".partial"),
        final.with_name(final.name + ".journal.jsonl"),
    )


def _rewrite_journal(path: Path, mutate) -> None:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(lines)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_record_has_six_ordered_t158_segments() -> None:
    rng = np.random.default_rng(7)
    projections = {
        "gate_proj": rng.normal(size=(4, 64)).astype(np.float32),
        "up_proj": rng.normal(size=(6, 64)).astype(np.float32),
        "down_proj": rng.normal(size=(5, 64)).astype(np.float32),
    }

    record = encode_expert_record(
        projections,
        layer=1,
        expert=3,
        shard="experts.bin",
        record_offset=128,
    )

    assert [segment.component for segment in record.segments] == [
        "gate_proj.packed",
        "gate_proj.scales",
        "up_proj.packed",
        "up_proj.scales",
        "down_proj.packed",
        "down_proj.scales",
    ]
    gate_packed_expected, gate_scales_expected = encode_t158(projections["gate_proj"])
    gate_packed = np.frombuffer(
        record.payload[: gate_packed_expected.nbytes], dtype=np.uint8
    ).reshape(gate_packed_expected.shape)
    gate_scales = np.frombuffer(
        record.payload[
            gate_packed_expected.nbytes : gate_packed_expected.nbytes
            + gate_scales_expected.nbytes
        ],
        dtype="<u2",
    ).reshape(gate_scales_expected.shape)
    np.testing.assert_array_equal(
        decode_t158(gate_packed, gate_scales, projections["gate_proj"].shape[1]),
        decode_t158(
            gate_packed_expected,
            gate_scales_expected,
            projections["gate_proj"].shape[1],
        ),
    )
    assert record.record_offset == 128
    assert record.logical_bytes == len(record.payload)
    assert record.sha256 == hashlib.sha256(record.payload).hexdigest()


def test_layer_resume_adopts_completed_journal(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    first = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )
    before = first.path.read_bytes()

    second = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )

    assert second == first
    assert second.path.read_bytes() == before
    assert len(second.records) == 2
    assert all(record.payload is None for record in first.records)
    assert all(record.payload is None for record in second.records)
    completion = json.loads(second.journal_path.read_text().splitlines()[-1])
    assert completion == {
        "completion": {
            "record_count": 2,
            "size": len(before),
            "sha256": hashlib.sha256(before).hexdigest(),
        }
    }


def test_layer_recovers_valid_final_plus_partial_hard_link_crash(
    tmp_path: Path,
) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )
    final, partial, _journal = _layer_paths(output, layout)
    partial.hardlink_to(final)
    assert final.stat().st_ino == partial.stat().st_ino

    recovered = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )

    assert recovered == converted
    assert final.is_file()
    assert not partial.exists()


def test_layer_recovers_lone_empty_initial_partial(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    final, partial, journal = _layer_paths(output, layout)
    partial.write_bytes(b"")

    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )

    assert converted.path == final
    assert journal.is_file()
    assert not partial.exists()


def test_layer_refuses_lone_nonempty_partial(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    _final, partial, _journal = _layer_paths(output, layout)
    partial.write_bytes(b"not-an-initial-partial")

    with pytest.raises(ValueError, match="lone|journal|initial"):
        convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)


def test_layer_resume_truncates_torn_tail(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )
    final = converted.path
    partial = final.with_name(final.name + ".partial")
    final.replace(partial)
    with partial.open("ab") as handle:
        handle.write(b"torn")
    journal = final.with_name(final.name + ".journal.jsonl")
    with journal.open("ab") as handle:
        handle.write(b'{"ordinal":')

    resumed = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )

    assert resumed.path == final
    assert final.stat().st_size == converted.logical_bytes
    assert journal.read_bytes().endswith(b"\n")


def test_layer_resume_refuses_wrong_source_revision(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )
    converted.path.replace(converted.path.with_name(converted.path.name + ".partial"))
    changed = KimiK3Inventory(
        revision="different",
        files=inventory.files,
        expert_tensors=inventory.expert_tensors,
        resident_tensors=(),
        layers=(1,),
    )

    with pytest.raises(ValueError, match="revision|source identity"):
        convert_layer(source, output, changed, layer=1, resume=True, layout=layout)


def test_layer_resume_refuses_changed_tensor_identity(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )
    converted.path.replace(converted.path.with_name(converted.path.name + ".partial"))
    altered_tensor = GGUFTensor(
        inventory.expert_tensors[0].name,
        inventory.expert_tensors[0].dims,
        inventory.expert_tensors[0].ggml_type,
        inventory.expert_tensors[0].offset + 84,
    )
    changed = KimiK3Inventory(
        revision=inventory.revision,
        files=inventory.files,
        expert_tensors=(altered_tensor,) + inventory.expert_tensors[1:],
        resident_tensors=(),
        layers=(1,),
    )

    with pytest.raises(ValueError, match="identity"):
        convert_layer(source, output, changed, layer=1, resume=True, layout=layout)


@pytest.mark.parametrize("mutation", ("modify", "replace"))
def test_layer_reopen_refuses_source_changed_after_inspection(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)
    if mutation == "modify":
        with source.path.open("r+b") as handle:
            handle.seek(20)
            original = handle.read(1)
            handle.seek(20)
            handle.write(bytes([original[0] ^ 1]))
            handle.flush()
            os.fsync(handle.fileno())
    else:
        original = source.path.read_bytes()
        source.path.unlink()
        source.path.write_bytes(original)

    with pytest.raises(ValueError, match="changed since inspection"):
        convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)


def test_layer_resume_rebuilds_altered_completed_record(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )
    expected = converted.path.read_bytes()
    partial = converted.path.with_name(converted.path.name + ".partial")
    converted.path.replace(partial)
    with partial.open("r+b") as handle:
        handle.seek(5)
        handle.write(b"\xff")

    resumed = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )

    assert resumed.path.read_bytes() == expected


@pytest.mark.parametrize(
    "field,value",
    (
        ("tensor", "wrong.tensor"),
        ("component", "up_proj.packed"),
        ("dtype", "BF16"),
        ("shape", [1, 1]),
        ("offset", 1),
        ("length", 1),
        ("sha256", "0" * 64),
    ),
)
def test_layer_final_adoption_rejects_tampered_segment_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    converted = convert_layer(
        source, output, inventory, layer=1, resume=True, layout=layout
    )

    def mutate(lines: list[dict]) -> None:
        lines[1]["output"]["segments"][0][field] = value

    _rewrite_journal(converted.journal_path, mutate)

    with pytest.raises(ValueError, match="segment|component|geometry|hash"):
        convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)


def test_layer_refuses_non_finite_decoded_projection_before_record_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"

    def non_finite(_blob, *, value_count: int) -> np.ndarray:
        return np.full(value_count, np.inf, dtype=np.float32)

    monkeypatch.setattr(t158_module, "dequantize_q2_k", non_finite)

    with pytest.raises(ValueError, match="non-finite"):
        convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)

    _final, partial, journal = _layer_paths(output, layout)
    assert partial.stat().st_size == 0
    assert len(journal.read_text().splitlines()) == 1


def test_layer_refuses_non_finite_encoded_scale_before_record_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    original = t158_module.encode_t158

    def non_finite_scale(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        packed, scales = original(weights)
        scales.fill(np.uint16(0x7F80))
        return packed, scales

    monkeypatch.setattr(t158_module, "encode_t158", non_finite_scale)

    with pytest.raises(ValueError, match="scale.*non-finite"):
        convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)

    _final, partial, journal = _layer_paths(output, layout)
    assert partial.stat().st_size == 0
    assert len(journal.read_text().splitlines()) == 1


def test_layer_readback_hash_must_match_before_final_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)
    output = tmp_path / "output"
    original = t158_module._hash_file

    def wrong_partial_hash(path: Path) -> str:
        if path.name.endswith(".partial"):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(t158_module, "_hash_file", wrong_partial_hash)

    with pytest.raises(ValueError, match="readback|durable|completion|hash"):
        convert_layer(source, output, inventory, layer=1, resume=True, layout=layout)

    final, partial, _journal = _layer_paths(output, layout)
    assert not final.exists()
    assert partial.is_file()


def test_layer_refuses_source_output_directory_equality(tmp_path: Path) -> None:
    source, inventory, layout = _tiny_layer_source(tmp_path)

    with pytest.raises(ValueError, match="separate|source"):
        convert_layer(
            source, source.path.parent, inventory, layer=1, resume=True, layout=layout
        )


def test_resident_copy_is_raw_and_excludes_q2(tmp_path: Path) -> None:
    f32 = np.array([1.25, -2.5, 3.5, 4.5, -5.5, 6.5], dtype="<f4").tobytes()
    bf16 = bytes.fromhex("803f00c040400041a040c040")
    q2 = _q2_block()
    tensors = (
        GGUFTensor("resident.f32", (3, 2), GGML_TYPE_F32, 0),
        GGUFTensor("resident.bf16", (2, 1, 3), GGML_TYPE_BF16, len(f32)),
        GGUFTensor(
            "blk.1.ffn_gate_exps.weight",
            (256,),
            GGML_TYPE_Q2_K,
            len(f32) + len(bf16),
        ),
    )
    source = _source(tmp_path / "source.gguf", tensors, f32 + bf16 + q2)
    output = tmp_path / "artifact" / "resident-001.safetensors"

    residents = copy_resident_safetensors(source, output)

    header, payload = _read_safetensors(output)
    assert set(header) == {"resident.f32", "resident.bf16"}
    assert header["resident.f32"]["dtype"] == "F32"
    assert header["resident.f32"]["shape"] == [2, 3]
    assert header["resident.bf16"]["dtype"] == "BF16"
    assert header["resident.bf16"]["shape"] == [3, 1, 2]
    f32_start, f32_end = header["resident.f32"]["data_offsets"]
    bf16_start, bf16_end = header["resident.bf16"]["data_offsets"]
    assert payload[f32_start:f32_end] == f32
    assert payload[bf16_start:bf16_end] == bf16
    assert [resident.tensor for resident in residents] == [
        "resident.bf16",
        "resident.f32",
    ]


def test_resident_valid_completed_shard_is_adopted(tmp_path: Path) -> None:
    raw = np.arange(8, dtype="<f4").tobytes()
    source = _source(
        tmp_path / "source.gguf",
        (GGUFTensor("resident", (8,), GGML_TYPE_F32, 0),),
        raw,
    )
    output = tmp_path / "artifact" / "resident.safetensors"
    first = copy_resident_safetensors(source, output)
    before = output.stat()

    second = copy_resident_safetensors(source, output)

    after = output.stat()
    assert second == first
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_resident_recovers_valid_final_plus_partial_hard_link(
    tmp_path: Path,
) -> None:
    raw = np.arange(8, dtype="<f4").tobytes()
    source = _source(
        tmp_path / "source.gguf",
        (GGUFTensor("resident", (8,), GGML_TYPE_F32, 0),),
        raw,
    )
    output = tmp_path / "artifact" / "resident.safetensors"
    copy_resident_safetensors(source, output)
    partial = output.with_name(output.name + ".partial")
    partial.hardlink_to(output)

    copy_resident_safetensors(source, output)

    assert output.is_file()
    assert not partial.exists()


def test_resident_refuses_final_plus_unrelated_partial(tmp_path: Path) -> None:
    raw = np.arange(8, dtype="<f4").tobytes()
    source = _source(
        tmp_path / "source.gguf",
        (GGUFTensor("resident", (8,), GGML_TYPE_F32, 0),),
        raw,
    )
    output = tmp_path / "artifact" / "resident.safetensors"
    copy_resident_safetensors(source, output)
    partial = output.with_name(output.name + ".partial")
    partial.write_bytes(b"unrelated")

    with pytest.raises(ValueError, match="partial|inode|hard.link|identity"):
        copy_resident_safetensors(source, output)

    assert output.is_file()
    assert partial.read_bytes() == b"unrelated"


def test_resident_partial_is_rebuilt(tmp_path: Path) -> None:
    raw = np.arange(8, dtype="<f4").tobytes()
    source = _source(
        tmp_path / "source.gguf",
        (GGUFTensor("resident", (8,), GGML_TYPE_F32, 0),),
        raw,
    )
    output = tmp_path / "artifact" / "resident.safetensors"
    output.parent.mkdir()
    output.with_name(output.name + ".partial").write_bytes(b"stale")

    copy_resident_safetensors(source, output)

    assert output.is_file()
    assert not output.with_name(output.name + ".partial").exists()


def test_resident_refuses_unrelated_existing_final(tmp_path: Path) -> None:
    raw = np.arange(8, dtype="<f4").tobytes()
    source = _source(
        tmp_path / "source.gguf",
        (GGUFTensor("resident", (8,), GGML_TYPE_F32, 0),),
        raw,
    )
    output = tmp_path / "artifact" / "resident.safetensors"
    output.parent.mkdir()
    output.write_bytes(b"unrelated")

    with pytest.raises(ValueError, match="refus|receipt"):
        copy_resident_safetensors(source, output)

    assert output.read_bytes() == b"unrelated"


def test_resident_refuses_final_when_receipt_hash_is_wrong(tmp_path: Path) -> None:
    raw = np.arange(8, dtype="<f4").tobytes()
    source = _source(
        tmp_path / "source.gguf",
        (GGUFTensor("resident", (8,), GGML_TYPE_F32, 0),),
        raw,
    )
    output = tmp_path / "artifact" / "resident.safetensors"
    copy_resident_safetensors(source, output)
    receipt = output.with_name(output.name + ".receipt.json")
    parsed = json.loads(receipt.read_text())
    parsed["output"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(parsed))

    with pytest.raises(ValueError, match="hash|receipt"):
        copy_resident_safetensors(source, output)
