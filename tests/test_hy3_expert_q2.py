from __future__ import annotations

import math
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mtplx.hy3_expert_q2 as q2_module
from mtplx import expert_manifest as expert_manifest_module
from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
)
from mtplx.hy3_expert_q2 import (
    ProjectionDiagnostics,
    ResidentReuse,
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL_KEY,
    TARGET_MODEL_KEY,
    requantize_expert_record_q4_to_q2,
    requantize_projection_q4_to_q2,
    stage_exact_residents,
)


_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "scales", "biases")
_DTYPES = ("U32", "BF16", "BF16")
_ANCILLARY_PAYLOADS = {
    "config.json": b'{"model_type":"hy_v3","quantization":{"bits":4}}\n',
    "generation_config.json": b'{"temperature":0.9}\n',
    "tokenizer.json": b'{"version":"1.0"}\n',
    "tokenizer_config.json": b'{"chat_template":"source"}\n',
    "special_tokens_map.json": b'{"eos_token":"<eos>"}\n',
    "chat_template.jinja": b"{{ messages }}\n",
}


def _write_resident_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> tuple[ShardInfo, tuple[ResidentTensor, ...]]:
    header: dict[str, object] = {}
    payload = bytearray()
    ranges: dict[str, tuple[int, int]] = {}
    for name in sorted(tensors):
        dtype, shape, tensor_payload = tensors[name]
        start = len(payload)
        payload.extend(tensor_payload)
        ranges[name] = (start, len(payload))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": list(ranges[name]),
        }
    header_raw = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    header_raw += b" " * (-len(header_raw) % 8)
    length_raw = len(header_raw).to_bytes(8, "little")
    contents = length_raw + header_raw + payload
    path.write_bytes(contents)
    data_start = len(length_raw) + len(header_raw)
    residents = tuple(
        ResidentTensor(
            tensor=name,
            shard=path.name,
            offset=data_start + ranges[name][0],
            length=len(tensor_payload),
            dtype=dtype,
            shape=shape,
        )
        for name, (dtype, shape, tensor_payload) in sorted(tensors.items())
    )
    return (
        ShardInfo(
            name=path.name,
            size=len(contents),
            header_bytes=data_start,
            header_sha256=hashlib.sha256(length_raw + header_raw).hexdigest(),
            sha256=hashlib.sha256(contents).hexdigest(),
        ),
        residents,
    )


def _resident_source(
    root: Path,
    *,
    resident_names: tuple[str, str] = (
        "model.embed_tokens.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ),
) -> tuple[ExpertManifest, dict[str, bytes]]:
    root.mkdir()
    resident_specs = (
        (resident_names[0], "BF16", (4,), bytes(range(8))),
        (resident_names[1], "F32", (2,), bytes(range(8, 16))),
    )
    shards = []
    residents = []
    copied_payloads: dict[str, bytes] = {}
    weight_map: dict[str, str] = {}
    for index, (name, dtype, shape, payload) in enumerate(resident_specs, start=1):
        shard_name = f"model-{index:05d}-of-00003.safetensors"
        shard, shard_residents = _write_resident_safetensors(
            root / shard_name,
            {name: (dtype, shape, payload)},
        )
        shards.append(shard)
        residents.extend(shard_residents)
        weight_map[name] = shard_name
        copied_payloads[shard_name] = (root / shard_name).read_bytes()

    _unreferenced, _routed = _write_resident_safetensors(
        root / "model-00003-of-00003.safetensors",
        {
            "model.layers.1.mlp.switch_mlp.gate_proj.weight": (
                "U32",
                (1,),
                b"q4!!",
            )
        },
    )
    index_payload = (
        json.dumps(
            {"metadata": {"total_size": 16}, "weight_map": weight_map},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (root / "model.safetensors.index.json").write_bytes(index_payload)
    copied_payloads["model.safetensors.index.json"] = index_payload
    for name, payload in _ANCILLARY_PAYLOADS.items():
        (root / name).write_bytes(payload)
        copied_payloads[name] = payload
    mtp_root = root / "mtp"
    mtp_root.mkdir()
    (mtp_root / "model.safetensors").write_bytes(b"do-not-copy-mtp")
    manifest = ExpertManifest(
        model_key="hy3-expert-only-q4",
        source_repo="local/hy3-expert-only-mlx-q4",
        source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        artifact_tensor_bytes=16,
        resident_tensor_bytes=16,
        routed_expert_bytes=0,
        shards=tuple(shards),
        resident_tensors=tuple(sorted(residents, key=lambda item: item.tensor)),
        records=(),
    ).with_digest()
    return manifest, copied_payloads


def _bf16_matrix(rows: int, columns: int, *, offset: float) -> mx.array:
    values = np.linspace(
        -1.75 + offset,
        1.75 + offset,
        num=rows * columns,
        dtype=np.float32,
    ).reshape(rows, columns)
    return mx.array(values).astype(mx.bfloat16)


def _array_bytes(value: mx.array, dtype: str) -> bytes:
    mx.eval(value)
    if dtype == "U32":
        return np.array(value, copy=True).astype("<u4", copy=False).tobytes()
    return (
        np.array(value.view(mx.uint16), copy=True).astype("<u2", copy=False).tobytes()
    )


def _q4_projection(
    *,
    input_size: int,
    output_size: int,
    offset: float,
) -> tuple[tuple[bytes, bytes, bytes], tuple[mx.array, mx.array, mx.array]]:
    dense = _bf16_matrix(output_size, input_size, offset=offset)
    quantized = mx.quantize(
        dense,
        bits=4,
        group_size=64,
        mode="affine",
    )
    mx.eval(quantized)
    return (
        tuple(
            _array_bytes(value, dtype)
            for value, dtype in zip(quantized, _DTYPES, strict=True)
        ),
        quantized,
    )


def _decode_q2(
    payloads: tuple[bytes, bytes, bytes],
    *,
    input_size: int,
    output_size: int,
) -> tuple[mx.array, mx.array, mx.array]:
    weight = mx.array(
        np.frombuffer(payloads[0], dtype="<u4")
        .copy()
        .reshape(output_size, input_size * 2 // 32),
        dtype=mx.uint32,
    )

    def bf16(payload: bytes) -> mx.array:
        words = mx.array(
            np.frombuffer(payload, dtype="<u2")
            .copy()
            .reshape(output_size, input_size // 64),
            dtype=mx.uint16,
        )
        return words.view(mx.bfloat16)

    return weight, bf16(payloads[1]), bf16(payloads[2])


def _q4_record() -> tuple[ExpertRecord, bytes]:
    payload = bytearray()
    segments = []
    dimensions = {
        "gate_proj": (64, 128),
        "up_proj": (64, 128),
        "down_proj": (128, 64),
    }
    for projection_index, projection in enumerate(_PROJECTIONS):
        input_size, output_size = dimensions[projection]
        components, _arrays = _q4_projection(
            input_size=input_size,
            output_size=output_size,
            offset=projection_index / 8,
        )
        shapes = (
            (output_size, input_size * 4 // 32),
            (output_size, input_size // 64),
            (output_size, input_size // 64),
        )
        for leaf, dtype, shape, component_payload in zip(
            _LEAVES,
            _DTYPES,
            shapes,
            components,
            strict=True,
        ):
            component = f"{projection}.{leaf}"
            segments.append(
                TensorSegment(
                    component=component,
                    tensor=f"model.layers.1.mlp.switch_mlp.{component}",
                    shard="experts.bin",
                    offset=len(payload),
                    length=len(component_payload),
                    dtype=dtype,
                    shape=shape,
                )
            )
            payload.extend(component_payload)
    return (
        ExpertRecord(
            layer=1,
            expert=2,
            logical_bytes=len(payload),
            segments=tuple(segments),
        ),
        bytes(payload),
    )


def test_projection_module_import_is_lazy_about_mlx() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mtplx.hy3_expert_q2; "
                "assert not any(name == 'mlx' or name.startswith('mlx.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_projection_source_and_target_identity_are_pinned() -> None:
    assert SOURCE_MODEL_KEY == "hy3-expert-only-q4"
    assert TARGET_MODEL_KEY == "hy3-expert-q2"
    assert SOURCE_MANIFEST_SHA256 == (
        "507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43"
    )


def test_projection_requantizes_genuine_q4_to_canonical_q2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _arrays = _q4_projection(
        input_size=64,
        output_size=128,
        offset=0.0,
    )
    original_dequantize = mx.dequantize
    original_quantize = mx.quantize
    calls: list[tuple[str, int, int, str]] = []

    def spy_dequantize(*args, **kwargs):
        calls.append(
            (
                "dequantize",
                kwargs["bits"],
                kwargs["group_size"],
                kwargs["mode"],
            )
        )
        return original_dequantize(*args, **kwargs)

    def spy_quantize(*args, **kwargs):
        calls.append(("quantize", kwargs["bits"], kwargs["group_size"], kwargs["mode"]))
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr(mx, "dequantize", spy_dequantize)
    monkeypatch.setattr(mx, "quantize", spy_quantize)

    output, diagnostics = requantize_projection_q4_to_q2(
        *source,
        projection="gate_proj",
        input_size=64,
        output_size=128,
    )

    assert calls == [
        ("dequantize", 4, 64, "affine"),
        ("quantize", 2, 64, "affine"),
        ("dequantize", 2, 64, "affine"),
    ]
    assert isinstance(diagnostics, ProjectionDiagnostics)
    assert diagnostics.component == "gate_proj"
    assert diagnostics.finite is True
    assert math.isfinite(diagnostics.cosine_q4_q2)
    assert math.isfinite(diagnostics.normalized_error_q4_q2)
    assert tuple(len(item) for item in output) == (2_048, 256, 256)

    weight, scales, biases = _decode_q2(
        output,
        input_size=64,
        output_size=128,
    )
    assert weight.dtype == mx.uint32
    assert scales.dtype == mx.bfloat16
    assert biases.dtype == mx.bfloat16
    assert tuple(weight.shape) == (128, 4)
    assert tuple(scales.shape) == (128, 1)
    assert tuple(biases.shape) == (128, 1)
    roundtrip = mx.dequantize(
        weight,
        scales,
        biases,
        bits=2,
        group_size=64,
        mode="affine",
    )
    mx.eval(roundtrip)
    assert tuple(roundtrip.shape) == (128, 64)
    assert mx.all(mx.isfinite(roundtrip)).item()


@pytest.mark.parametrize("group_size", [32, 64.0, True])
def test_projection_rejects_non_group64_geometry(group_size: object) -> None:
    source, _arrays = _q4_projection(
        input_size=64,
        output_size=64,
        offset=0.0,
    )

    with pytest.raises(ValueError, match="group_size must be 64"):
        requantize_projection_q4_to_q2(
            *source,
            projection="gate_proj",
            input_size=64,
            output_size=64,
            group_size=group_size,
        )


@pytest.mark.parametrize(
    "fault",
    ["weight_dtype", "weight_shape", "scales_dtype", "scales_shape"],
)
def test_projection_rejects_noncanonical_q2_output_metadata(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source, _arrays = _q4_projection(
        input_size=64,
        output_size=64,
        offset=0.0,
    )
    original_quantize = mx.quantize

    def invalid_quantize(*args, **kwargs):
        weight, scales, biases = original_quantize(*args, **kwargs)
        if fault == "weight_dtype":
            weight = weight.astype(mx.uint16)
        elif fault == "weight_shape":
            weight = weight.reshape(32, 8)
        elif fault == "scales_dtype":
            scales = scales.astype(mx.float32)
        else:
            scales = scales.reshape(32, 2)
        return weight, scales, biases

    monkeypatch.setattr(mx, "quantize", invalid_quantize)

    with pytest.raises(ValueError, match="invalid Q2"):
        requantize_projection_q4_to_q2(
            *source,
            projection="gate_proj",
            input_size=64,
            output_size=64,
        )


def test_canonical_record_converts_one_projection_at_a_time_then_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, source = _q4_record()
    events: list[str] = []
    outputs: dict[str, bytes] = {}
    original_clear_cache = mx.clear_cache

    def clear_cache() -> None:
        events.append("clear")
        original_clear_cache()

    def read_component(segment: TensorSegment) -> bytes:
        return source[segment.offset : segment.offset + segment.length]

    def write_component(component: str, payload: bytes) -> None:
        events.append(component)
        outputs[component] = payload

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    diagnostics = requantize_expert_record_q4_to_q2(
        record,
        read_component,
        write_component,
        hidden_size=64,
        expert_hidden_size=128,
    )

    canonical = tuple(
        f"{projection}.{leaf}" for projection in _PROJECTIONS for leaf in _LEAVES
    )
    assert tuple(outputs) == canonical
    assert events == [
        "clear",
        "clear",
        "clear",
        *canonical,
    ]
    assert tuple(item.component for item in diagnostics) == _PROJECTIONS
    assert all(item.finite for item in diagnostics)
    assert tuple(len(outputs[name]) for name in canonical) == (
        2_048,
        256,
        256,
        2_048,
        256,
        256,
        2_048,
        256,
        256,
    )


@pytest.mark.parametrize("fault", ["order", "shape", "dtype", "length"])
def test_canonical_record_rejects_wrong_metadata_before_read_or_write(
    fault: str,
) -> None:
    record, source = _q4_record()
    segments = list(record.segments)
    if fault == "order":
        segments[0], segments[1] = segments[1], segments[0]
    elif fault == "shape":
        segments[0] = replace(segments[0], shape=(128, 7))
    elif fault == "dtype":
        segments[0] = replace(segments[0], dtype="BF16")
    else:
        segments[0] = replace(segments[0], length=segments[0].length - 1)
    changed = replace(record, segments=tuple(segments))
    read: list[str] = []
    written: list[str] = []

    def read_component(segment: TensorSegment) -> bytes:
        read.append(segment.component)
        return source[segment.offset : segment.offset + segment.length]

    with pytest.raises(ValueError, match="canonical|shape|dtype|length"):
        requantize_expert_record_q4_to_q2(
            changed,
            read_component,
            lambda component, _payload: written.append(component),
            hidden_size=64,
            expert_hidden_size=128,
        )

    assert read == []
    assert written == []


def test_canonical_record_rejects_short_source_read_before_output_acceptance() -> None:
    record, source = _q4_record()
    written: list[str] = []

    def short_read(segment: TensorSegment) -> bytes:
        payload = source[segment.offset : segment.offset + segment.length]
        return payload[:-1] if segment is record.segments[0] else payload

    with pytest.raises(ValueError, match="short read"):
        requantize_expert_record_q4_to_q2(
            record,
            short_read,
            lambda component, _payload: written.append(component),
            hidden_size=64,
            expert_hidden_size=128,
        )

    assert written == []


@pytest.mark.parametrize("projection_index", [1, 2], ids=["up", "down"])
@pytest.mark.parametrize("fault", ["short", "oversized", "nonfinite"])
def test_canonical_record_late_source_fault_never_emits_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    projection_index: int,
    fault: str,
) -> None:
    record, source = _q4_record()
    first_segment = record.segments[projection_index * 3]
    scales_segment = record.segments[projection_index * 3 + 1]
    written: list[str] = []
    conversions: list[str] = []
    original_requantize = q2_module.requantize_projection_q4_to_q2

    def tracking_requantize(*args, **kwargs):
        conversions.append(kwargs["projection"])
        return original_requantize(*args, **kwargs)

    monkeypatch.setattr(
        q2_module,
        "requantize_projection_q4_to_q2",
        tracking_requantize,
    )

    def faulty_read(segment: TensorSegment) -> bytes:
        payload = source[segment.offset : segment.offset + segment.length]
        if fault == "short" and segment is first_segment:
            return payload[:-1]
        if fault == "oversized" and segment is first_segment:
            return payload + b"\0"
        if fault == "nonfinite" and segment is scales_segment:
            return np.full(segment.shape, 0x7F80, dtype="<u2").tobytes()
        return payload

    with pytest.raises(ValueError, match="short|oversized|non-finite"):
        requantize_expert_record_q4_to_q2(
            record,
            faulty_read,
            lambda component, _payload: written.append(component),
            hidden_size=64,
            expert_hidden_size=128,
        )

    assert written == []
    if fault in {"short", "oversized"}:
        assert conversions == []
    else:
        assert conversions == list(_PROJECTIONS[: projection_index + 1])


@pytest.mark.parametrize("nonfinite_stage", ["source", "target"])
def test_nonfinite_projection_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    nonfinite_stage: str,
) -> None:
    source, _arrays = _q4_projection(
        input_size=64,
        output_size=64,
        offset=0.25,
    )
    original_dequantize = mx.dequantize
    call_count = 0

    def nonfinite_dequantize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = original_dequantize(*args, **kwargs)
        selected = (nonfinite_stage == "source" and call_count == 1) or (
            nonfinite_stage == "target" and call_count == 2
        )
        if selected:
            return mx.full(result.shape, float("inf"), dtype=result.dtype)
        return result

    monkeypatch.setattr(mx, "dequantize", nonfinite_dequantize)

    with pytest.raises(ValueError, match="non-finite"):
        requantize_projection_q4_to_q2(
            *source,
            projection="down_proj",
            input_size=64,
            output_size=64,
        )


@pytest.mark.parametrize("leaf_index", [1, 2])
def test_nonfinite_projection_q4_parameters_are_rejected(leaf_index: int) -> None:
    source, _arrays = _q4_projection(
        input_size=64,
        output_size=64,
        offset=0.25,
    )
    changed = list(source)
    changed[leaf_index] = np.full((64, 1), 0x7F80, dtype="<u2").tobytes()

    with pytest.raises(ValueError, match="non-finite Q4"):
        requantize_projection_q4_to_q2(
            *changed,
            projection="down_proj",
            input_size=64,
            output_size=64,
        )


def test_resident_reuse_copies_exact_index_allowlist_and_excludes_q4_and_mtp(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, expected_payloads = _resident_source(source_root)
    work_root.mkdir()

    result = stage_exact_residents(
        source_root,
        manifest,
        work_root,
        copy_chunk_bytes=3,
    )

    assert isinstance(result, ResidentReuse)
    assert result.shards == manifest.shards
    assert result.tensors == manifest.resident_tensors
    assert set(result.copied_files) == set(expected_payloads)
    assert {path.name for path in work_root.iterdir()} == set(expected_payloads)
    for name, expected in expected_payloads.items():
        source_path = source_root / name
        target_path = work_root / name
        assert target_path.read_bytes() == expected
        assert result.copied_files[name] == hashlib.sha256(expected).hexdigest()
        source_status = source_path.stat()
        target_status = target_path.stat()
        assert (target_status.st_dev, target_status.st_ino) != (
            source_status.st_dev,
            source_status.st_ino,
        )
        assert target_status.st_nlink == 1
    assert (work_root / "config.json").read_bytes() == _ANCILLARY_PAYLOADS[
        "config.json"
    ]
    assert not (work_root / "model-00003-of-00003.safetensors").exists()
    assert not (work_root / "mtp").exists()


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_resident_index_must_equal_manifest_allowlist(
    tmp_path: Path,
    fault: str,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    index_path = source_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if fault == "missing":
        removed = sorted(index["weight_map"])[0]
        del index["weight_map"][removed]
        index["metadata"]["total_size"] = 8
    else:
        index["weight_map"]["model.layers.1.mlp.switch_mlp.gate_proj.weight"] = (
            "model-00003-of-00003.safetensors"
        )
        index["metadata"]["total_size"] = 20
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="resident index|routed expert"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize(
    "resident_name",
    [
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.layers.80.self_attn.q_proj.weight",
    ],
    ids=["routed", "mtp"],
)
def test_resident_staging_rejects_routed_or_mtp_tensor_contamination(
    tmp_path: Path,
    resident_name: str,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(
        source_root,
        resident_names=(resident_name, "model.norm.weight"),
    )
    work_root.mkdir()

    with pytest.raises(ValueError, match="routed expert|MTP"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_resident_metadata_must_match_index_headers(
    tmp_path: Path,
    fault: str,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    residents = list(manifest.resident_tensors)
    if fault == "dtype":
        residents[0] = replace(residents[0], dtype="I16")
    else:
        residents[0] = replace(residents[0], shape=(2, 2))
    changed = replace(manifest, resident_tensors=tuple(residents)).with_digest()

    with pytest.raises(ValueError, match="resident metadata"):
        stage_exact_residents(source_root, changed, work_root)

    assert list(work_root.iterdir()) == []


def test_resident_source_full_hash_must_match_manifest_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    shard_path = source_root / manifest.shards[0].name
    contents = bytearray(shard_path.read_bytes())
    contents[-1] ^= 0xFF
    shard_path.write_bytes(contents)

    with pytest.raises(ValueError, match="hash|provenance"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


def test_ancillary_allowlist_is_required_and_copied_without_rewriting(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    (source_root / "special_tokens_map.json").unlink()

    with pytest.raises(ValueError, match="ancillary|special_tokens_map"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


def test_resident_index_path_escape_is_rejected_before_target_mutation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    index_path = source_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"][sorted(index["weight_map"])[0]] = "../outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe|escape|path"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize("target_kind", ["root_symlink", "member_symlink"])
def test_resident_staging_refuses_target_symlinks(
    tmp_path: Path,
    target_kind: str,
) -> None:
    source_root = tmp_path / "source"
    manifest, _payloads = _resident_source(source_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"unchanged")
    work_root = tmp_path / "work"
    if target_kind == "root_symlink":
        work_root.symlink_to(outside, target_is_directory=True)
    else:
        work_root.mkdir()
        (work_root / "config.json").symlink_to(sentinel)

    with pytest.raises(ValueError, match="target|work root|empty|symlink"):
        stage_exact_residents(source_root, manifest, work_root)

    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("mutation", ["config_bytes", "extra_mtp", "config_symlink"])
def test_resident_final_state_rejects_post_copy_mutation_and_cleans_created_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, expected_payloads = _resident_source(source_root)
    work_root.mkdir()
    original_copy = q2_module._copy_independent_file

    def mutate_after_last_copy(*args, **kwargs):
        digest = original_copy(*args, **kwargs)
        if args[2] == "chat_template.jinja":
            target_config = work_root / "config.json"
            if mutation == "config_bytes":
                target_config.write_bytes(b"mutated-after-copy")
            elif mutation == "extra_mtp":
                (work_root / "mtp").mkdir()
            else:
                target_config.unlink()
                target_config.symlink_to(source_root / "config.json")
        return digest

    monkeypatch.setattr(q2_module, "_copy_independent_file", mutate_after_last_copy)

    with pytest.raises(ValueError, match="final|inventory|hash|regular|symlink|swap"):
        stage_exact_residents(source_root, manifest, work_root)

    remaining = {path.name for path in work_root.iterdir()}
    if mutation == "extra_mtp":
        assert remaining == {"mtp"}
    elif mutation == "config_symlink":
        assert remaining == {"config.json"}
        assert (work_root / "config.json").is_symlink()
    else:
        assert remaining == set()
    assert (source_root / "config.json").read_bytes() == expected_payloads[
        "config.json"
    ]


def test_resident_final_identity_detects_swap_during_descriptor_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, expected_payloads = _resident_source(source_root)
    work_root.mkdir()
    original_copy = q2_module._copy_independent_file
    original_hash_fd = q2_module._hash_fd
    target_config_inode: int | None = None
    copies_finished = False
    swapped = False

    def observe_last_copy(*args, **kwargs):
        nonlocal target_config_inode, copies_finished
        digest = original_copy(*args, **kwargs)
        if args[2] == "config.json":
            target_config_inode = (work_root / "config.json").stat().st_ino
        if args[2] == "chat_template.jinja":
            copies_finished = True
        return digest

    def swap_after_final_hash(fd: int, *, length: int, chunk_bytes: int) -> str:
        nonlocal swapped
        digest = original_hash_fd(fd, length=length, chunk_bytes=chunk_bytes)
        if (
            copies_finished
            and not swapped
            and target_config_inode is not None
            and os.fstat(fd).st_ino == target_config_inode
        ):
            swapped = True
            target = work_root / "config.json"
            target.unlink()
            target.symlink_to(source_root / "config.json")
        return digest

    monkeypatch.setattr(q2_module, "_copy_independent_file", observe_last_copy)
    monkeypatch.setattr(q2_module, "_hash_fd", swap_after_final_hash)

    with pytest.raises(ValueError, match="swap|identity|symlink|regular"):
        stage_exact_residents(source_root, manifest, work_root)

    assert swapped is True
    assert {path.name for path in work_root.iterdir()} == {"config.json"}
    assert (work_root / "config.json").is_symlink()
    assert (source_root / "config.json").read_bytes() == expected_payloads[
        "config.json"
    ]


def test_resident_final_index_and_headers_are_parsed_from_held_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    original_inventory = expert_manifest_module._checkpoint_inventory

    def reject_path_following_target_inventory(root: Path, *, hash_shards: bool):
        if Path(root).resolve() == work_root.resolve():
            raise AssertionError("final inventory followed target paths")
        return original_inventory(root, hash_shards=hash_shards)

    monkeypatch.setattr(
        expert_manifest_module,
        "_checkpoint_inventory",
        reject_path_following_target_inventory,
    )

    result = stage_exact_residents(source_root, manifest, work_root)

    assert result.tensors == manifest.resident_tensors


@pytest.mark.parametrize("symlinked_root", ["source", "work"])
def test_resident_staging_rejects_symlinked_directory_ancestors(
    tmp_path: Path,
    symlinked_root: str,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    source_real = real_parent / "source"
    manifest, _payloads = _resident_source(source_real)
    work_real = real_parent / "work"
    work_real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    source_root = alias / "source" if symlinked_root == "source" else source_real
    work_root = alias / "work" if symlinked_root == "work" else work_real

    with pytest.raises(ValueError, match="ancestor|symlink"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_real.iterdir()) == []


def test_resident_final_directory_identity_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    moved_root = tmp_path / "work-moved"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    original_copy = q2_module._copy_independent_file

    def swap_work_path_after_last_copy(*args, **kwargs):
        receipt = original_copy(*args, **kwargs)
        if args[2] == "chat_template.jinja":
            work_root.rename(moved_root)
            work_root.symlink_to(moved_root, target_is_directory=True)
        return receipt

    monkeypatch.setattr(
        q2_module,
        "_copy_independent_file",
        swap_work_path_after_last_copy,
    )

    with pytest.raises(ValueError, match="identity|symlink|path|swap"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(moved_root.iterdir()) == []


def test_resident_final_source_identity_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    moved_source = tmp_path / "source-moved"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    original_copy = q2_module._copy_independent_file

    def swap_source_path_after_last_copy(*args, **kwargs):
        receipt = original_copy(*args, **kwargs)
        if args[2] == "chat_template.jinja":
            source_root.rename(moved_source)
            source_root.symlink_to(moved_source, target_is_directory=True)
        return receipt

    monkeypatch.setattr(
        q2_module,
        "_copy_independent_file",
        swap_source_path_after_last_copy,
    )

    with pytest.raises(ValueError, match="identity|symlink|path|swap"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


def test_resident_final_directory_fsync_failure_cleans_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    original_fsync = q2_module.os.fsync
    directory_fsyncs = 0
    injected = False

    def fail_final_directory_fsync(fd: int) -> None:
        nonlocal directory_fsyncs, injected
        metadata = os.fstat(fd)
        if stat.S_ISDIR(metadata.st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 10 and not injected:
                injected = True
                raise OSError("injected final directory fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(q2_module.os, "fsync", fail_final_directory_fsync)

    with pytest.raises(OSError, match="injected final directory fsync failure"):
        stage_exact_residents(source_root, manifest, work_root)

    assert injected is True
    assert list(work_root.iterdir()) == []
    result = stage_exact_residents(source_root, manifest, work_root)
    assert result.tensors == manifest.resident_tensors


@pytest.mark.parametrize("chunk_bytes", [0, 64 * 1024**2 + 1])
def test_resident_copy_chunk_bound_is_strict(
    tmp_path: Path,
    chunk_bytes: int,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()

    with pytest.raises(ValueError, match="copy_chunk_bytes"):
        stage_exact_residents(
            source_root,
            manifest,
            work_root,
            copy_chunk_bytes=chunk_bytes,
        )

    assert list(work_root.iterdir()) == []
