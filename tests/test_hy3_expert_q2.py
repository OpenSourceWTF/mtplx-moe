from __future__ import annotations

import ast
import math
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.hy3_expert_q2 as q2_module
from mtplx import expert_manifest as expert_manifest_module
from mtplx.expert_manifest import (
    EMPTY_SHA256,
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    SidecarInfo,
    ShardInfo,
    TensorSegment,
)
from mtplx.expert_streaming_models import ExpertStreamingModelSpec
from mtplx.hy3_expert_q2 import (
    ConversionConfig,
    ProjectionDiagnostics,
    ResidentReuse,
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL_KEY,
    TARGET_MODEL_KEY,
    convert_expert_records,
    finalize_hy3_expert_q2,
    pilot_hy3_expert_q2,
    preflight_hy3_expert_q2,
    requantize_expert_record_q4_to_q2,
    requantize_projection_q4_to_q2,
    stage_exact_residents,
    stage_hy3_expert_q2,
    verify_hy3_expert_q2,
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


def _resident_hf_source(
    root: Path,
) -> tuple[Path, Path, ExpertManifest, dict[str, bytes]]:
    repository = root / "repository"
    source_root = repository / "snapshots" / ("a" * 40)
    source_root.parent.mkdir(parents=True)
    manifest, expected_payloads = _resident_source(source_root)
    blobs = repository / "blobs"
    blobs.mkdir()
    for name, payload in expected_payloads.items():
        blob_name = hashlib.sha256(payload).hexdigest()
        source = source_root / name
        source.replace(blobs / blob_name)
        source.symlink_to(Path("..") / ".." / "blobs" / blob_name)
    return source_root, blobs, manifest, expected_payloads


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


def test_resident_source_and_final_inventory_are_parsed_from_held_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()

    def reject_path_following_inventory(root: Path, *, hash_shards: bool):
        raise AssertionError(f"inventory followed paths under {root}")

    monkeypatch.setattr(
        expert_manifest_module,
        "_checkpoint_inventory",
        reject_path_following_inventory,
    )

    result = stage_exact_residents(source_root, manifest, work_root)

    assert result.tensors == manifest.resident_tensors


def test_resident_staging_accepts_only_pinned_hf_blob_symlinks(
    tmp_path: Path,
) -> None:
    source_root, _blobs, manifest, expected_payloads = _resident_hf_source(tmp_path)
    work_root = tmp_path / "work"
    work_root.mkdir()

    result = stage_exact_residents(source_root, manifest, work_root)

    assert result.tensors == manifest.resident_tensors
    for name, payload in expected_payloads.items():
        assert (work_root / name).read_bytes() == payload


def test_resident_staging_preserves_regular_source_inside_snapshots_directory(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "repository" / "snapshots" / ("a" * 40)
    source_root.parent.mkdir(parents=True)
    manifest, expected = _resident_source(source_root)
    work_root = tmp_path / "work"
    work_root.mkdir()

    result = stage_exact_residents(source_root, manifest, work_root)

    assert result.tensors == manifest.resident_tensors
    for name, payload in expected.items():
        assert (work_root / name).read_bytes() == payload


def test_resident_staging_rejects_hf_looking_link_outside_snapshot_layout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    work_root = tmp_path / "work"
    manifest, _payloads = _resident_source(source_root)
    work_root.mkdir()
    config = source_root / "config.json"
    config.unlink()
    config.symlink_to(Path("..") / ".." / "blobs" / ("b" * 64))

    with pytest.raises(ValueError, match="HF snapshot|symlink"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


def test_resident_staging_rejects_persistent_hf_blob_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, blobs, manifest, _payloads = _resident_hf_source(tmp_path)
    moved_blobs = blobs.with_name("blobs-moved")
    work_root = tmp_path / "work"
    work_root.mkdir()
    original_copy = q2_module._copy_independent_file

    def swap_blobs_after_last_copy(*args, **kwargs):
        receipt = original_copy(*args, **kwargs)
        if args[2] == "chat_template.jinja":
            blobs.rename(moved_blobs)
            blobs.mkdir()
        return receipt

    monkeypatch.setattr(
        q2_module,
        "_copy_independent_file",
        swap_blobs_after_last_copy,
    )

    with pytest.raises(ValueError, match="HF|blob|identity"):
        stage_exact_residents(source_root, manifest, work_root)

    assert list(work_root.iterdir()) == []


def test_resident_staging_never_discovers_hf_parents_from_reparented_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, genuine_blobs, manifest, expected = _resident_hf_source(tmp_path)
    attacker_repository = tmp_path / "attacker-repository"
    attacker_snapshots = attacker_repository / "snapshots"
    attacker_blobs = attacker_repository / "blobs"
    attacker_snapshots.mkdir(parents=True)
    attacker_blobs.mkdir()
    for blob in genuine_blobs.iterdir():
        (attacker_blobs / blob.name).write_bytes(blob.read_bytes())
    config_blob = hashlib.sha256(expected["config.json"]).hexdigest()
    (attacker_blobs / config_blob).write_bytes(b"attacker-config")
    attacker_revision = attacker_snapshots / source_root.name
    work_root = tmp_path / "work"
    work_root.mkdir()
    original_open_source = q2_module._open_source_artifact
    original_copy = q2_module._copy_independent_file
    original_recheck = q2_module._recheck_source_artifacts
    reparented = False

    def reparent_before_first_member(*args, **kwargs):
        nonlocal reparented
        if args[2] == "model.safetensors.index.json" and not reparented:
            source_root.rename(attacker_revision)
            reparented = True
        return original_open_source(*args, **kwargs)

    def restore_after_last_copy(*args, **kwargs):
        receipt = original_copy(*args, **kwargs)
        if args[2] == "chat_template.jinja":
            attacker_revision.rename(source_root)
        return receipt

    def satisfy_separated_parent_recheck(*args, **kwargs):
        source_root.rename(attacker_revision)
        try:
            return original_recheck(*args, **kwargs)
        finally:
            attacker_revision.rename(source_root)

    monkeypatch.setattr(
        q2_module,
        "_open_source_artifact",
        reparent_before_first_member,
    )
    monkeypatch.setattr(q2_module, "_copy_independent_file", restore_after_last_copy)
    monkeypatch.setattr(
        q2_module,
        "_recheck_source_artifacts",
        satisfy_separated_parent_recheck,
    )

    try:
        stage_exact_residents(source_root, manifest, work_root)
    except ValueError:
        assert list(work_root.iterdir()) == []
    else:
        assert (work_root / "config.json").read_bytes() == expected["config.json"]


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


def test_resident_transient_source_root_swap_cannot_supply_attacker_ancillary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    moved_source = tmp_path / "source-moved"
    work_root = tmp_path / "work"
    manifest, expected_payloads = _resident_source(source_root)
    work_root.mkdir()
    original_copy = q2_module._copy_independent_file

    def swap_source_path_while_copying_config(*args, **kwargs):
        if args[2] != "config.json":
            return original_copy(*args, **kwargs)
        source_root.rename(moved_source)
        source_root.mkdir()
        (source_root / "config.json").write_bytes(b"attacker-config")
        try:
            return original_copy(*args, **kwargs)
        finally:
            (source_root / "config.json").unlink()
            source_root.rmdir()
            moved_source.rename(source_root)

    monkeypatch.setattr(
        q2_module,
        "_copy_independent_file",
        swap_source_path_while_copying_config,
    )

    result = stage_exact_residents(source_root, manifest, work_root)

    assert result.tensors == manifest.resident_tensors
    assert (work_root / "config.json").read_bytes() == expected_payloads["config.json"]


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


def _test_component_metadata(
    spec: ExpertStreamingModelSpec,
) -> tuple[tuple[str, str, tuple[int, int], int], ...]:
    result = []
    for projection in _PROJECTIONS:
        input_size = (
            spec.hidden_size if projection != "down_proj" else spec.expert_hidden_size
        )
        output_size = (
            spec.expert_hidden_size if projection != "down_proj" else spec.hidden_size
        )
        shapes = (
            (output_size, input_size * spec.quant_bits // 32),
            (output_size, input_size // spec.quant_group_size),
            (output_size, input_size // spec.quant_group_size),
        )
        for leaf, dtype, shape in zip(_LEAVES, _DTYPES, shapes, strict=True):
            item_size = 4 if dtype == "U32" else 2
            result.append(
                (
                    f"{projection}.{leaf}",
                    dtype,
                    shape,
                    shape[0] * shape[1] * item_size,
                )
            )
    return tuple(result)


def _test_conversion_spec(
    *, bits: int, resident_bytes: int
) -> ExpertStreamingModelSpec:
    hidden_size = 64
    expert_hidden_size = 128
    component_bytes = sum(
        item[3]
        for item in _test_component_metadata(
            SimpleNamespace(
                hidden_size=hidden_size,
                expert_hidden_size=expert_hidden_size,
                quant_bits=bits,
                quant_group_size=64,
            )
        )
    )
    key = "hy3-expert-only-q4" if bits == 4 else "hy3-expert-q2"
    return ExpertStreamingModelSpec(
        key=key,
        display_name=f"Test Hy3 Q{bits}",
        source_model="test/tencent-hy3",
        source_revision="resident-revision",
        quant_model="test/hy3-q4",
        quant_revision="oracle-revision",
        total_tensor_bytes=resident_bytes + 3 * component_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=3,
        top_k=1,
        hidden_size=hidden_size,
        expert_hidden_size=expert_hidden_size,
        quant_bits=bits,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="source bfloat16",
        router_matmul_dtype="float32",
        router_bytes=0,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )


def _write_test_manifest(path: Path, manifest: ExpertManifest) -> tuple[str, str]:
    finalized = manifest.with_digest()
    payload = (json.dumps(finalized.to_dict(), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(payload)
    assert finalized.manifest_sha256 is not None
    return hashlib.sha256(payload).hexdigest(), finalized.manifest_sha256


def _conversion_test_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    source_root = tmp_path / "hy3-expert-only-mlx-q4"
    resident_manifest, _resident_payloads = _resident_source(source_root)
    source_spec = _test_conversion_spec(bits=4, resident_bytes=16)
    target_spec = _test_conversion_spec(bits=2, resident_bytes=16)
    source_components = _test_component_metadata(source_spec)
    source_record_bytes = sum(item[3] for item in source_components)
    target_record_bytes = target_spec.expert_record_bytes
    one_record = b"".join(
        bytes([index + 1]) * length
        for index, (_component, _dtype, _shape, length) in enumerate(source_components)
    )
    records = []
    for expert in range(source_spec.expert_count):
        record_offset = expert * source_record_bytes
        cursor = record_offset
        segments = []
        for component, dtype, shape, length in source_components:
            segments.append(
                TensorSegment(
                    component=component,
                    tensor=(
                        f"model.layers.1.mlp.switch_mlp.experts.{expert}.{component}"
                    ),
                    shard="experts.bin",
                    offset=cursor,
                    length=length,
                    dtype=dtype,
                    shape=shape,
                )
            )
            cursor += length
        records.append(
            ExpertRecord(
                layer=1,
                expert=expert,
                logical_bytes=source_record_bytes,
                segments=tuple(segments),
                sha256=hashlib.sha256(one_record).hexdigest(),
                sidecar_offset=record_offset,
                sidecar_length=source_record_bytes,
            )
        )
    source_sidecar = one_record * source_spec.expert_count
    (source_root / "experts.bin").write_bytes(source_sidecar)
    sidecar_sha256 = hashlib.sha256(source_sidecar).hexdigest()
    sidecar = SidecarInfo(
        file="experts.bin",
        alignment=256,
        size=len(source_sidecar),
        sha256=sidecar_sha256,
    )
    sidecar_shard = ShardInfo(
        name="experts.bin",
        size=len(source_sidecar),
        header_bytes=0,
        header_sha256=EMPTY_SHA256,
        sha256=sidecar_sha256,
        kind="sidecar",
    )
    manifest = replace(
        resident_manifest,
        model_key=source_spec.key,
        source_repo=source_spec.quant_model,
        source_revision=source_spec.quant_revision,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        artifact_tensor_bytes=16 + len(source_sidecar),
        resident_tensor_bytes=16,
        routed_expert_bytes=len(source_sidecar),
        shards=(*resident_manifest.shards, sidecar_shard),
        records=tuple(records),
        sidecar=sidecar,
        manifest_sha256=None,
    )
    manifest_path = source_root / "expert-manifest.json"
    manifest_file_sha256, manifest_sha256 = _write_test_manifest(
        manifest_path,
        manifest,
    )
    provenance = {
        "format": "mtplx-hy3-expert-only-q4-provenance-v1",
        "source": {
            "repo": source_spec.source_model,
            "revision": source_spec.source_revision,
        },
        "oracle": {
            "repo": source_spec.quant_model,
            "revision": source_spec.quant_revision,
        },
    }
    provenance_path = source_root / "conversion-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    expectations = q2_module._ConversionExpectations(
        source_root=source_root,
        manifest_file_sha256=manifest_file_sha256,
        manifest_sha256=manifest_sha256,
        provenance_sha256=file_sha256(provenance_path),
        index_sha256=file_sha256(source_root / "model.safetensors.index.json"),
        config_sha256=file_sha256(source_root / "config.json"),
        sidecar_sha256=sidecar_sha256,
        source_sidecar_bytes=len(source_sidecar),
        record_count=source_spec.expert_count,
        source_record_bytes=source_record_bytes,
        target_record_bytes=target_record_bytes,
        target_sidecar_bytes=target_record_bytes * target_spec.expert_count,
        resident_tensor_bytes=16,
        target_tensor_bytes=16 + target_record_bytes * target_spec.expert_count,
        resident_shard_count=len(resident_manifest.shards),
        alignment=256,
        resident_source_repo=source_spec.source_model,
        resident_source_revision=source_spec.source_revision,
        oracle_repo=source_spec.quant_model,
        oracle_revision=source_spec.quant_revision,
    )
    expectation_box = [expectations]
    producer_box = [{"git_commit": "a" * 40, "dirty": False}]
    mlx_box = ["0.31.test"]
    target_state_box: list[dict[str, object] | None] = [None]
    free_box = [10**15]
    monkeypatch.setattr(
        q2_module,
        "_conversion_expectations",
        lambda: expectation_box[0],
    )
    monkeypatch.setattr(q2_module, "_source_descriptor", lambda: source_spec)
    monkeypatch.setattr(q2_module, "_target_descriptor", lambda: target_spec)
    monkeypatch.setattr(q2_module, "_producer_state", lambda: producer_box[0])
    monkeypatch.setattr(q2_module, "_mlx_version", lambda: mlx_box[0])
    original_target_state = q2_module._target_descriptor_state
    monkeypatch.setattr(
        q2_module,
        "_target_descriptor_state",
        lambda spec: target_state_box[0] or original_target_state(spec),
    )
    monkeypatch.setattr(
        q2_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10**15, used=0, free=free_box[0]),
    )
    conversions: list[tuple[int, int]] = []

    def fake_requantize(
        record: ExpertRecord,
        read_component,
        write_component,
        **_kwargs,
    ):
        conversions.append((record.layer, record.expert))
        for source_segment, target in zip(
            record.segments,
            _test_component_metadata(target_spec),
            strict=True,
        ):
            component, _dtype, _shape, length = target
            seed = hashlib.sha256(
                read_component(source_segment) + component.encode("utf-8")
            ).digest()
            payload = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
            write_component(component, payload)
        return tuple(
            ProjectionDiagnostics(
                component=projection,
                cosine_q4_q2=0.99,
                normalized_error_q4_q2=0.01,
                finite=True,
            )
            for projection in _PROJECTIONS
        )

    monkeypatch.setattr(
        q2_module,
        "requantize_expert_record_q4_to_q2",
        fake_requantize,
    )
    output_root = tmp_path / "hy3-expert-only-mlx-q2"
    config = ConversionConfig(
        source_root=source_root,
        source_manifest=manifest_path,
        source_provenance=provenance_path,
        output_root=output_root,
        alignment=256,
    )
    return {
        "config": config,
        "source_root": source_root,
        "output_root": output_root,
        "work_root": output_root.with_name(f".{output_root.name}.incomplete"),
        "manifest_path": manifest_path,
        "provenance_path": provenance_path,
        "expectation_box": expectation_box,
        "producer_box": producer_box,
        "mlx_box": mlx_box,
        "target_state_box": target_state_box,
        "free_box": free_box,
        "conversions": conversions,
        "target_record_bytes": target_record_bytes,
        "source_record_bytes": source_record_bytes,
        "manifest": manifest.with_digest(),
        "source_spec": source_spec,
        "target_spec": target_spec,
    }


def _rewrite_conversion_test_manifest(
    env: dict[str, object],
    manifest: ExpertManifest,
) -> ExpertManifest:
    finalized = manifest.with_digest()
    file_sha256, manifest_sha256 = _write_test_manifest(
        env["manifest_path"],
        finalized,
    )
    env["expectation_box"][0] = replace(
        env["expectation_box"][0],
        manifest_file_sha256=file_sha256,
        manifest_sha256=manifest_sha256,
    )
    env["manifest"] = finalized
    return finalized


def _complete_staged_conversion(env: dict[str, object]) -> ConversionConfig:
    pilot = pilot_hy3_expert_q2(env["config"], ((1, 0),))
    pilot_report = env["source_root"].parent / "pilot-report.json"
    pilot_report.write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    env["conversions"].clear()
    stage_hy3_expert_q2(env["config"])
    convert_expert_records(env["config"], resume=True)
    return replace(env["config"], pilot_report=pilot_report)


def test_provenance_contract_uses_exact_hy3_counts_bytes_and_derivation() -> None:
    expectations = q2_module._conversion_expectations()
    manifest = q2_module._minimum_conversion_manifest(expectations)

    assert expectations.resident_shard_count == 18
    assert manifest["schema"] == "mtplx-hy3-expert-q2-conversion-v1"
    assert manifest["derivation"] == {
        "kind": "q4_to_q2",
        "source_bits": 4,
        "target_bits": 2,
        "group_size": 64,
        "mode": "affine",
        "external_q2_artifact_used": False,
    }
    assert manifest["target"] == {
        "model_key": "hy3-expert-q2",
        "record_count": 15_168,
        "record_bytes": 5_898_240,
        "sidecar_bytes": 89_464_504_320,
        "resident_tensor_bytes": 17_494_289_664,
        "tensor_bytes": 106_958_793_984,
        "mtp_included": False,
    }


def test_preflight_checks_exact_source_and_space_before_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = env["config"]

    report = preflight_hy3_expert_q2(config, deep_source_hash=True)

    assert not env["work_root"].exists()
    assert report["source"]["model_key"] == "hy3-expert-only-q4"
    assert report["source"]["record_count"] == 3
    assert report["target"]["record_bytes"] == env["target_record_bytes"]
    assert report["space"]["required_bytes"] == (
        (report["space"]["base_bytes"] * 105 + 99) // 100 + 64 * 1024**2
    )
    assert report["space"]["free_bytes"] == 10**15


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_file",
        "index",
        "config",
        "provenance",
        "sidecar_size",
        "sidecar_hash",
        "resident_size",
        "dirty",
        "free_space",
    ],
)
def test_preflight_rejects_source_or_producer_fault_before_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    source_root = env["source_root"]
    if fault == "manifest_file":
        env["manifest_path"].write_bytes(env["manifest_path"].read_bytes() + b" ")
    elif fault in {"index", "config", "provenance"}:
        name = {
            "index": "model.safetensors.index.json",
            "config": "config.json",
            "provenance": "conversion-provenance.json",
        }[fault]
        (source_root / name).write_bytes((source_root / name).read_bytes() + b" ")
    elif fault == "sidecar_size":
        sidecar = source_root / "experts.bin"
        sidecar.write_bytes(sidecar.read_bytes()[:-1])
    elif fault == "sidecar_hash":
        sidecar = source_root / "experts.bin"
        payload = bytearray(sidecar.read_bytes())
        payload[-1] ^= 0xFF
        sidecar.write_bytes(payload)
    elif fault == "resident_size":
        shard = source_root / "model-00001-of-00003.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])
    elif fault == "dirty":
        env["producer_box"][0] = {"git_commit": "a" * 40, "dirty": True}
    else:
        env["free_box"][0] = 0

    with pytest.raises(ValueError, match="hash|size|dirty|space|provenance"):
        preflight_hy3_expert_q2(env["config"], deep_source_hash=True)

    assert not env["work_root"].exists()


@pytest.mark.parametrize(
    "fault",
    ["model_key", "q4_metadata", "record_product", "resident_bytes", "upstream"],
)
def test_preflight_rejects_structural_source_fault_before_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    manifest = env["manifest"]
    if fault == "model_key":
        _rewrite_conversion_test_manifest(
            env,
            replace(manifest, model_key="wrong-source-key", manifest_sha256=None),
        )
    elif fault == "q4_metadata":
        _rewrite_conversion_test_manifest(
            env,
            replace(manifest, quant_bits=2, manifest_sha256=None),
        )
    elif fault == "record_product":
        records = manifest.records[:-1]
        routed_bytes = sum(record.logical_bytes for record in records)
        _rewrite_conversion_test_manifest(
            env,
            replace(
                manifest,
                records=records,
                routed_expert_bytes=routed_bytes,
                artifact_tensor_bytes=manifest.resident_tensor_bytes + routed_bytes,
                manifest_sha256=None,
            ),
        )
    elif fault == "resident_bytes":
        _rewrite_conversion_test_manifest(
            env,
            replace(
                manifest,
                resident_tensor_bytes=manifest.resident_tensor_bytes + 1,
                artifact_tensor_bytes=manifest.artifact_tensor_bytes + 1,
                manifest_sha256=None,
            ),
        )
    else:
        provenance_path = env["provenance_path"]
        provenance = json.loads(provenance_path.read_text())
        provenance["source"]["repo"] = "attacker/repo"
        provenance_path.write_text(json.dumps(provenance) + "\n")
        env["expectation_box"][0] = replace(
            env["expectation_box"][0],
            provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        )

    with pytest.raises(ValueError, match="model|manifest|record|resident|provenance"):
        preflight_hy3_expert_q2(env["config"], deep_source_hash=False)

    assert not env["work_root"].exists()


def test_convert_checks_free_space_before_creating_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    env["free_box"][0] = 0

    with pytest.raises(ValueError, match="free space"):
        convert_expert_records(env["config"], resume=True)

    assert not env["work_root"].exists()


def test_convert_deep_hashes_entire_sidecar_before_creating_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    sidecar = env["source_root"] / "experts.bin"
    payload = bytearray(sidecar.read_bytes())
    payload[-1] ^= 0xFF
    sidecar.write_bytes(payload)

    with pytest.raises(ValueError, match="sidecar hash"):
        convert_expert_records(env["config"], resume=True)

    assert env["conversions"] == []
    assert not env["work_root"].exists()


def test_resume_deep_hashes_resident_shards_before_touching_durable_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    output = env["work_root"] / "experts.bin"
    journal = env["work_root"] / "conversion-journal.jsonl"
    durable_state = (output.read_bytes(), journal.read_bytes())
    resident = env["source_root"] / env["manifest"].resident_tensors[0].shard
    payload = bytearray(resident.read_bytes())
    payload[-1] ^= 0xFF
    resident.write_bytes(payload)
    env["conversions"].clear()

    with pytest.raises(ValueError, match="resident shard hash"):
        convert_expert_records(env["config"], resume=True)

    assert env["conversions"] == []
    assert (output.read_bytes(), journal.read_bytes()) == durable_state


def test_convert_rejects_source_root_replacement_after_deep_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    source_root = env["source_root"]
    moved_source = source_root.with_name(f"{source_root.name}-moved")
    original_disk_usage = q2_module.shutil.disk_usage
    replaced = False

    def replace_source_after_validation(path):
        nonlocal replaced
        usage = original_disk_usage(path)
        source_root.rename(moved_source)
        source_root.mkdir()
        (source_root / "experts.bin").write_bytes(
            (moved_source / "experts.bin").read_bytes()
        )
        replaced = True
        return usage

    monkeypatch.setattr(q2_module.shutil, "disk_usage", replace_source_after_validation)

    with pytest.raises(ValueError, match="source root.*identity|identity.*source root"):
        convert_expert_records(env["config"], resume=True)

    assert replaced is True
    assert env["conversions"] == []
    assert not env["work_root"].exists()


def test_convert_rejects_output_parent_replacement_before_workdir_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    moved_parent = tmp_path.with_name(f"{tmp_path.name}-moved")
    replacement_work = env["work_root"]
    held_work = moved_parent / replacement_work.name
    original_disk_usage = q2_module.shutil.disk_usage
    replaced = False

    def replace_output_parent_after_validation(path):
        nonlocal replaced
        usage = original_disk_usage(path)
        tmp_path.rename(moved_parent)
        tmp_path.mkdir()
        replaced = True
        return usage

    monkeypatch.setattr(
        q2_module.shutil,
        "disk_usage",
        replace_output_parent_after_validation,
    )

    with pytest.raises(
        ValueError, match="output parent.*identity|identity.*output parent"
    ):
        convert_expert_records(env["config"], resume=True)

    assert replaced is True
    assert not replacement_work.exists()
    assert not held_work.exists()


def test_conversion_config_rejects_unpinned_paths_and_alignment(tmp_path: Path) -> None:
    source_root = tmp_path / "wrong-source"
    output_root = tmp_path / "hy3-expert-only-mlx-q2"

    with pytest.raises(ValueError, match="source_root"):
        ConversionConfig(
            source_root=source_root,
            source_manifest=source_root / "expert-manifest.json",
            source_provenance=source_root / "conversion-provenance.json",
            output_root=output_root,
        )
    source_root = tmp_path / "hy3-expert-only-mlx-q4"
    with pytest.raises(ValueError, match="alignment"):
        ConversionConfig(
            source_root=source_root,
            source_manifest=source_root / "expert-manifest.json",
            source_provenance=source_root / "conversion-provenance.json",
            output_root=output_root,
            alignment=3,
        )


def test_journal_fsyncs_output_before_record_and_binds_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    original_fsync = q2_module.os.fsync
    fsync_inodes: list[int] = []

    def record_fsync(fd: int) -> None:
        fsync_inodes.append(os.fstat(fd).st_ino)
        original_fsync(fd)

    monkeypatch.setattr(q2_module.os, "fsync", record_fsync)
    records = convert_expert_records(env["config"], resume=True)
    output_path = env["work_root"] / "experts.bin"
    journal_path = env["work_root"] / "conversion-journal.jsonl"
    output_inode = output_path.stat().st_ino
    journal_inode = journal_path.stat().st_ino
    relevant = [
        inode for inode in fsync_inodes if inode in {output_inode, journal_inode}
    ]

    assert len(records) == 3
    assert relevant == [
        journal_inode,
        output_inode,
        journal_inode,
        output_inode,
        journal_inode,
        output_inode,
        journal_inode,
    ]
    lines = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert lines[0]["source"]["fingerprint_sha256"]
    assert lines[0]["producer"] == {"git_commit": "a" * 40, "dirty": False}
    assert lines[0]["mlx_version"] == "0.31.test"
    assert lines[0]["derivation"]["external_q2_artifact_used"] is False
    assert all(len(line["source"]["components"]) == 9 for line in lines[1:])
    assert all(len(line["output"]["components"]) == 9 for line in lines[1:])


def test_journal_never_records_a_record_when_output_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    original_fsync = q2_module.os.fsync
    failed = False

    def fail_first_output_fsync(fd: int) -> None:
        nonlocal failed
        output = env["work_root"] / "experts.bin"
        if (
            not failed
            and output.exists()
            and os.fstat(fd).st_ino == output.stat().st_ino
            and os.fstat(fd).st_size > 0
        ):
            failed = True
            raise OSError("injected output fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(q2_module.os, "fsync", fail_first_output_fsync)

    with pytest.raises(OSError, match="injected output fsync failure"):
        convert_expert_records(env["config"], resume=True)

    journal = env["work_root"] / "conversion-journal.jsonl"
    assert failed is True
    assert len(journal.read_text().splitlines()) == 1


def test_resume_refuses_noncontiguous_journal_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    journal = env["work_root"] / "conversion-journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b"".join((lines[0], lines[1], lines[3])))

    with pytest.raises(ValueError, match="contiguous|journal|chain"):
        convert_expert_records(env["config"], resume=True)


def test_resume_accepts_only_durable_contiguous_prefix_and_discards_extra_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    journal = env["work_root"] / "conversion-journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b"".join(lines[:-1]))
    env["conversions"].clear()

    records = convert_expert_records(env["config"], resume=True)

    assert len(records) == 3
    assert env["conversions"] == [(1, 2)]
    assert (env["work_root"] / "experts.bin").stat().st_size == (
        3 * env["target_record_bytes"]
    )


@pytest.mark.parametrize(
    "change",
    ["commit", "mlx", "source_fingerprint", "target_descriptor"],
)
def test_resume_refuses_changed_build_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    if change == "commit":
        env["producer_box"][0] = {"git_commit": "b" * 40, "dirty": False}
    elif change == "mlx":
        env["mlx_box"][0] = "0.32.test"
    elif change == "source_fingerprint":
        tokenizer = env["source_root"] / "tokenizer.json"
        tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    else:
        env["target_state_box"][0] = {"key": "changed-target"}

    with pytest.raises(ValueError, match="fingerprint|header|resume"):
        convert_expert_records(env["config"], resume=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_name", "changed display"),
        ("top_k", 2),
        ("router_storage", "changed storage"),
        ("router_matmul_dtype", "changed dtype"),
        ("router_bytes", 1),
        ("kv_bytes_per_token", 1),
        ("full_indexer_layers", (0,)),
    ],
)
def test_resume_binds_every_target_descriptor_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    changed = replace(env["target_spec"], **{field: value})
    env["target_state_box"][0] = q2_module._target_descriptor_state(changed)

    with pytest.raises(ValueError, match="header|resume"):
        convert_expert_records(env["config"], resume=True)


def test_resume_source_mismatch_is_fatal_without_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    output = env["work_root"] / "experts.bin"
    journal = env["work_root"] / "conversion-journal.jsonl"
    original_sizes = (output.stat().st_size, journal.stat().st_size)
    source = env["source_root"] / "experts.bin"
    payload = bytearray(source.read_bytes())
    payload[env["source_record_bytes"] + 1] ^= 0xFF
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="source.*hash|source.*mismatch"):
        convert_expert_records(env["config"], resume=True)

    assert (output.stat().st_size, journal.stat().st_size) == original_sizes


def test_resume_output_corruption_truncates_to_prefix_and_recomputes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    convert_expert_records(env["config"], resume=True)
    output = env["work_root"] / "experts.bin"
    expected_output = output.read_bytes()
    env["conversions"].clear()
    payload = bytearray(expected_output)
    payload[env["target_record_bytes"] + 1] ^= 0xFF
    output.write_bytes(payload)

    records = convert_expert_records(env["config"], resume=True)

    assert len(records) == 3
    assert env["conversions"] == [(1, 1), (1, 2)]
    assert output.read_bytes() == expected_output


def test_pilot_is_read_only_and_reports_requested_real_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)

    report = pilot_hy3_expert_q2(env["config"], ((1, 0), (1, 2)))

    assert report["passed"] is True
    assert [(item["layer"], item["expert"]) for item in report["records"]] == [
        (1, 0),
        (1, 2),
    ]
    assert all(len(item["diagnostics"]) == 3 for item in report["records"])
    assert report["producer"] == {"git_commit": "a" * 40, "dirty": False}
    assert report["mlx_version"] == "0.31.test"
    assert not env["work_root"].exists()
    assert not env["output_root"].exists()


def test_stage_refuses_existing_target_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    env["output_root"].mkdir()
    sentinel = env["output_root"] / "sentinel"
    sentinel.write_bytes(b"unchanged")

    with pytest.raises(ValueError, match="final output.*exists"):
        stage_hy3_expert_q2(env["config"])

    assert sentinel.read_bytes() == b"unchanged"
    assert not env["work_root"].exists()


def test_stage_interruption_leaves_only_sibling_work_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("injected stage interruption")

    monkeypatch.setattr(q2_module, "stage_exact_residents", interrupt)

    with pytest.raises(KeyboardInterrupt, match="injected stage interruption"):
        stage_hy3_expert_q2(env["config"])

    assert env["work_root"].is_dir()
    assert list(env["work_root"].iterdir()) == []
    assert not env["output_root"].exists()


def test_finalize_publishes_authoritative_output_atomically_and_records_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    parent_inode = tmp_path.stat().st_ino
    original_rename = q2_module._exclusive_directory_rename
    original_fsync = q2_module.os.fsync
    events: list[str] = []

    def record_rename(source_directory_fd, source, target_directory_fd, target):
        result = original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )
        if target == env["output_root"].name:
            events.append("directory_replace")
        return result

    def record_fsync(fd: int) -> None:
        original_fsync(fd)
        if events and events[-1] == "directory_replace":
            metadata = os.fstat(fd)
            if stat.S_ISDIR(metadata.st_mode) and metadata.st_ino == parent_inode:
                events.append("parent_fsync")

    monkeypatch.setattr(q2_module, "_exclusive_directory_rename", record_rename)
    monkeypatch.setattr(q2_module.os, "fsync", record_fsync)

    published = finalize_hy3_expert_q2(config)

    assert published == env["output_root"]
    assert events[-2:] == ["directory_replace", "parent_fsync"]
    assert not env["work_root"].exists()
    assert not (published / "conversion-journal.jsonl").exists()
    assert not (published / "mtp").exists()
    manifest = ExpertManifest.from_dict(
        json.loads((published / "expert-manifest.json").read_text())
    )
    resident_shards = [
        shard for shard in manifest.shards if shard.kind == "safetensors"
    ]
    sidecar_shards = [shard for shard in manifest.shards if shard.kind == "sidecar"]
    assert len(resident_shards) == env["expectation_box"][0].resident_shard_count
    assert len(sidecar_shards) == 1
    assert sidecar_shards[0].name == "experts.bin"
    assert {
        segment.shard for record in manifest.records for segment in record.segments
    } == {"experts.bin"}
    conversion = json.loads((published / "conversion-manifest.json").read_text())
    assert conversion["journal"]["record_count"] == 3
    assert len(conversion["journal"]["sha256"]) == 64
    receipt_directory = tmp_path / q2_module._RETAINED_JOURNAL_DIRECTORY
    retained_journals = list(receipt_directory.iterdir())
    assert len(retained_journals) == 1
    assert stat.S_IMODE(receipt_directory.stat().st_mode) == 0o700
    assert retained_journals[0].name == f"{conversion['journal']['sha256']}.jsonl"
    assert (
        hashlib.sha256(retained_journals[0].read_bytes()).hexdigest()
        == conversion["journal"]["sha256"]
    )
    assert conversion["target"]["expert_manifest_sha256"] == manifest.manifest_sha256
    assert verify_hy3_expert_q2(published, deep=True)["passed"] is True


def test_failed_deep_verify_prevents_atomic_publish_and_preserves_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)

    def reject(*_args, **_kwargs):
        raise ValueError("injected deep verification failure")

    monkeypatch.setattr(q2_module, "_verify_hy3_fd", reject)

    with pytest.raises(ValueError, match="deep verification failure"):
        finalize_hy3_expert_q2(config)

    assert not env["output_root"].exists()
    assert env["work_root"].is_dir()
    assert (env["work_root"] / "conversion-journal.jsonl").is_file()


def test_finalize_interruption_after_journal_removal_is_safely_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    original_rename = q2_module._exclusive_directory_rename
    interrupted = False

    def interrupt_first_directory_publish(
        source_directory_fd,
        source,
        target_directory_fd,
        target,
    ):
        nonlocal interrupted
        if source == env["work_root"].name and not interrupted:
            interrupted = True
            raise OSError("injected publish interruption")
        return original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )

    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        interrupt_first_directory_publish,
    )

    with pytest.raises(OSError, match="publish interruption"):
        finalize_hy3_expert_q2(config)

    assert interrupted is True
    assert env["work_root"].is_dir()
    assert not (env["work_root"] / "conversion-journal.jsonl").exists()
    assert not env["output_root"].exists()
    receipt_directory = tmp_path / q2_module._RETAINED_JOURNAL_DIRECTORY
    retained_before_retry = list(receipt_directory.iterdir())
    assert len(retained_before_retry) == 1
    retained_inode = retained_before_retry[0].stat().st_ino

    assert finalize_hy3_expert_q2(config) == env["output_root"]
    retained_after_retry = list(receipt_directory.iterdir())
    assert len(retained_after_retry) == 1
    assert retained_after_retry[0].stat().st_ino == retained_inode
    assert verify_hy3_expert_q2(env["output_root"], deep=True)["passed"] is True


def test_finalize_verifies_the_held_directory_that_is_later_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    work_root = env["work_root"]
    moved_work = work_root.with_name(f"{work_root.name}-held-original")
    original_read_json = q2_module._read_json_member
    substituted = False

    def substitute_valid_clone_after_manifest_reads(*args, **kwargs):
        nonlocal substituted
        result = original_read_json(*args, **kwargs)
        if args[1] == "conversion-manifest.json" and not substituted:
            work_root.rename(moved_work)
            shutil.copytree(moved_work, work_root)
            sidecar = moved_work / "experts.bin"
            payload = bytearray(sidecar.read_bytes())
            payload[-1] ^= 0xFF
            sidecar.write_bytes(payload)
            substituted = True
        return result

    monkeypatch.setattr(
        q2_module,
        "_read_json_member",
        substitute_valid_clone_after_manifest_reads,
    )

    try:
        with pytest.raises(ValueError, match="sidecar|record|hash|changed|identity"):
            finalize_hy3_expert_q2(config)
    finally:
        if moved_work.exists():
            if work_root.exists():
                shutil.rmtree(work_root)
            moved_work.rename(work_root)

    assert substituted is True
    assert not env["output_root"].exists()
    assert work_root.is_dir()


def test_atomic_publish_never_overwrites_target_created_after_absence_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    original_stat = q2_module.os.stat
    raced_target_inode: int | None = None

    def create_target_after_absence_check(path, *args, **kwargs):
        nonlocal raced_target_inode
        try:
            return original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            directory_fd = kwargs.get("dir_fd")
            if (
                path == env["output_root"].name
                and directory_fd is not None
                and raced_target_inode is None
            ):
                work_fd: int | None = None
                try:
                    work_fd = os.open(
                        env["work_root"].name,
                        q2_module._directory_flags(),
                        dir_fd=directory_fd,
                    )
                    original_stat(
                        "conversion-manifest.json",
                        dir_fd=work_fd,
                        follow_symlinks=False,
                    )
                    try:
                        original_stat(
                            "conversion-journal.jsonl",
                            dir_fd=work_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        os.mkdir(path, dir_fd=directory_fd)
                        raced_target_inode = original_stat(
                            path,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        ).st_ino
                except (FileNotFoundError, NotADirectoryError):
                    pass
                finally:
                    if work_fd is not None:
                        os.close(work_fd)
            raise

    monkeypatch.setattr(q2_module.os, "stat", create_target_after_absence_check)

    with pytest.raises(OSError, match="exist|exclusive|publish"):
        finalize_hy3_expert_q2(config)

    assert raced_target_inode is not None
    assert env["output_root"].is_dir()
    assert env["output_root"].stat().st_ino == raced_target_inode
    assert env["work_root"].is_dir()


def test_finalize_rejects_pilot_output_not_bound_to_journal_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    pilot = json.loads(config.pilot_report.read_text())
    pilot["records"][0]["output_sha256"] = "0" * 64
    config.pilot_report.write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pilot.*journal|pilot.*output|record receipt"):
        finalize_hy3_expert_q2(config)

    assert not env["output_root"].exists()
    assert (env["work_root"] / "conversion-journal.jsonl").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("producer", {"git_commit": "b" * 40, "dirty": False}),
        ("mlx_version", "0.31.other"),
    ],
)
def test_finalize_binds_pilot_to_producer_and_mlx_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    pilot = json.loads(config.pilot_report.read_text())
    pilot[field] = value
    config.pilot_report.write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pilot.*provenance"):
        finalize_hy3_expert_q2(config)

    assert not env["output_root"].exists()
    assert (env["work_root"] / "conversion-journal.jsonl").is_file()


def test_publish_rolls_back_source_name_substituted_at_rename_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    held_work = env["work_root"].with_name(f"{env['work_root'].name}-held")
    verified_inode = env["work_root"].stat().st_ino
    original_rename = q2_module._exclusive_directory_rename
    substituted = False

    def substitute_source_then_rename(
        source_directory_fd,
        source,
        target_directory_fd,
        target,
    ):
        nonlocal substituted
        if source == env["work_root"].name and ".publish-" in target:
            env["work_root"].rename(held_work)
            env["work_root"].mkdir()
            (env["work_root"] / "attacker-owned").write_bytes(b"preserve me")
            substituted = True
        return original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )

    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        substitute_source_then_rename,
    )

    with pytest.raises(ValueError, match="private publication source.*identity"):
        finalize_hy3_expert_q2(config)

    assert substituted is True
    assert not env["output_root"].exists()
    assert env["work_root"].stat().st_ino == verified_inode
    assert not held_work.exists()
    assert any(
        path.is_dir() and (path / "attacker-owned").read_bytes() == b"preserve me"
        for path in tmp_path.iterdir()
        if (path / "attacker-owned").is_file()
    )


def test_private_publish_recovers_canonical_work_after_swap_and_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    verified_inode = env["work_root"].stat().st_ino
    hidden_verified = env["work_root"].with_name(
        f"{env['work_root'].name}-attacker-hidden"
    )
    original_rename = q2_module._exclusive_directory_rename
    interrupted = False

    def swap_then_rename_and_interrupt(
        source_directory_fd,
        source,
        target_directory_fd,
        target,
    ):
        nonlocal interrupted
        if source == env["work_root"].name and not interrupted:
            env["work_root"].rename(hidden_verified)
            env["work_root"].mkdir()
            (env["work_root"] / "attacker-owned").write_bytes(b"preserve me")
            original_rename(
                source_directory_fd,
                source,
                target_directory_fd,
                target,
            )
            interrupted = True
            raise KeyboardInterrupt("injected swapped-source interruption")
        return original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )

    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        swap_then_rename_and_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="swapped-source interruption"):
        finalize_hy3_expert_q2(config)

    assert interrupted is True
    assert not env["output_root"].exists()
    assert env["work_root"].is_dir()
    assert env["work_root"].stat().st_ino == verified_inode
    assert any(
        path.is_dir() and (path / "attacker-owned").read_bytes() == b"preserve me"
        for path in tmp_path.iterdir()
        if (path / "attacker-owned").is_file()
    )

    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        original_rename,
    )
    assert finalize_hy3_expert_q2(config) == env["output_root"]


def test_finalize_does_not_unlink_a_substituted_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    journal = env["work_root"] / "conversion-journal.jsonl"
    held_journal = env["work_root"] / "conversion-journal.held"
    retained_name = f"{hashlib.sha256(journal.read_bytes()).hexdigest()}.jsonl"
    original_rename = q2_module._exclusive_name_rename
    substituted = False

    def substitute_at_delete_rename(
        source_directory_fd,
        source,
        target_directory_fd,
        target,
    ):
        nonlocal substituted
        if source == journal.name and target == retained_name:
            journal.rename(held_journal)
            journal.write_bytes(b"attacker-owned replacement\n")
            substituted = True
        return original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )

    monkeypatch.setattr(
        q2_module,
        "_exclusive_name_rename",
        substitute_at_delete_rename,
    )

    with pytest.raises(ValueError, match="retained conversion journal.*identity"):
        finalize_hy3_expert_q2(config)

    assert substituted is True
    assert held_journal.is_file()
    receipt_directory = tmp_path / q2_module._RETAINED_JOURNAL_DIRECTORY
    retained = list(receipt_directory.iterdir())
    assert len(retained) == 1
    assert retained[0].read_bytes() == b"attacker-owned replacement\n"
    assert not env["output_root"].exists()


def test_finalize_does_not_unlink_tombstone_substituted_after_lease_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    receipt_directory = tmp_path / q2_module._RETAINED_JOURNAL_DIRECTORY
    held_journal = receipt_directory / ".conversion-journal.held"
    original_remove = q2_module._VerifiedDirectory.remove
    replacement: Path | None = None

    def substitute_after_lease_drop(verified, name):
        nonlocal replacement
        original_remove(verified, name)
        if name == "conversion-journal.jsonl":
            tombstones = [
                path for path in receipt_directory.iterdir() if path.suffix == ".jsonl"
            ]
            assert len(tombstones) == 1
            replacement = tombstones[0]
            replacement.rename(held_journal)
            replacement.write_bytes(b"attacker-owned replacement\n")

    monkeypatch.setattr(
        q2_module._VerifiedDirectory,
        "remove",
        substitute_after_lease_drop,
    )

    with pytest.raises(ValueError, match="retained conversion journal.*identity"):
        finalize_hy3_expert_q2(config)

    assert replacement is not None
    conversion = json.loads((env["work_root"] / "conversion-manifest.json").read_text())
    assert (
        hashlib.sha256(replacement.read_bytes()).hexdigest()
        == conversion["journal"]["sha256"]
    )
    assert not held_journal.exists()
    assert any(
        path.read_bytes() == b"attacker-owned replacement\n"
        for path in receipt_directory.iterdir()
        if path.is_file() and path != replacement
    )
    assert not env["output_root"].exists()


def test_finalize_never_deletes_a_finally_substituted_journal_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    receipt_directory = tmp_path / q2_module._RETAINED_JOURNAL_DIRECTORY
    held_journal = receipt_directory / ".conversion-journal.final-held"
    original_assert = q2_module._assert_named_file_identity
    replacement: Path | None = None
    retained_checks = 0

    def substitute_after_final_retained_check(
        directory_fd,
        name,
        member_fd,
        *,
        label,
    ):
        nonlocal replacement, retained_checks
        result = original_assert(
            directory_fd,
            name,
            member_fd,
            label=label,
        )
        if label == "retained conversion journal":
            retained_checks += 1
            if retained_checks == 5:
                replacement = receipt_directory / name
                os.rename(
                    name,
                    held_journal.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                replacement.write_bytes(b"attacker-owned replacement\n")
        return result

    monkeypatch.setattr(
        q2_module,
        "_assert_named_file_identity",
        substitute_after_final_retained_check,
    )

    published = finalize_hy3_expert_q2(config)

    assert retained_checks == 5
    assert replacement is not None
    assert replacement.read_bytes() == b"attacker-owned replacement\n"
    assert held_journal.is_file()
    assert verify_hy3_expert_q2(published, deep=True)["passed"] is True


def test_retained_journal_recovery_candidate_scan_is_bounded(tmp_path: Path) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir(mode=0o700)
    held = receipt_directory / "held"
    held.write_bytes(b"held receipt")
    for ordinal in range(3):
        (receipt_directory / f"candidate-{ordinal}").write_bytes(b"candidate")
    directory_fd = os.open(receipt_directory, q2_module._directory_flags())
    held_fd = os.open(held, q2_module._read_flags())
    try:
        with pytest.raises(ValueError, match="candidate scan is bounded"):
            q2_module._scan_directory_for_file_identity(
                directory_fd,
                held_fd,
                max_entries=3,
            )
    finally:
        os.close(held_fd)
        os.close(directory_fd)


def test_retained_journal_swap_and_publish_failure_remain_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    receipt_directory = tmp_path / q2_module._RETAINED_JOURNAL_DIRECTORY
    held_journal = receipt_directory / ".conversion-journal.non-prefix-held"
    original_assert = q2_module._assert_named_file_identity
    original_rename = q2_module._exclusive_directory_rename
    retained_checks = 0
    publish_failed = False

    def swap_after_final_retained_check(
        directory_fd,
        name,
        member_fd,
        *,
        label,
    ):
        nonlocal retained_checks
        result = original_assert(
            directory_fd,
            name,
            member_fd,
            label=label,
        )
        if label == "retained conversion journal":
            retained_checks += 1
            if retained_checks == 4:
                retained = receipt_directory / name
                retained.rename(held_journal)
                retained.write_bytes(b"attacker-owned replacement\n")
        return result

    def fail_publication_once(
        source_directory_fd,
        source,
        target_directory_fd,
        target,
    ):
        nonlocal publish_failed
        if target == env["output_root"].name and not publish_failed:
            publish_failed = True
            raise OSError("injected publication failure after retained swap")
        return original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )

    monkeypatch.setattr(
        q2_module,
        "_assert_named_file_identity",
        swap_after_final_retained_check,
    )
    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        fail_publication_once,
    )

    with pytest.raises(OSError, match="publication failure after retained swap"):
        finalize_hy3_expert_q2(config)

    assert retained_checks == 5
    assert publish_failed is True
    assert not env["output_root"].exists()
    assert env["work_root"].is_dir()
    assert not held_journal.exists()
    conversion = json.loads((env["work_root"] / "conversion-manifest.json").read_text())
    deterministic = receipt_directory / f"{conversion['journal']['sha256']}.jsonl"
    assert (
        hashlib.sha256(deterministic.read_bytes()).hexdigest()
        == conversion["journal"]["sha256"]
    )
    assert any(
        path.read_bytes() == b"attacker-owned replacement\n"
        for path in receipt_directory.iterdir()
        if path.is_file() and path != deterministic
    )

    monkeypatch.setattr(
        q2_module,
        "_assert_named_file_identity",
        original_assert,
    )
    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        original_rename,
    )
    assert finalize_hy3_expert_q2(config) == env["output_root"]


def test_finalize_rolls_back_when_parent_fsync_fails_after_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    original_fsync = q2_module.os.fsync
    parent_inode = tmp_path.stat().st_ino
    interrupted = False

    def fail_once_after_directory_rename(fd: int) -> None:
        nonlocal interrupted
        descriptor = os.fstat(fd)
        if (
            not interrupted
            and stat.S_ISDIR(descriptor.st_mode)
            and descriptor.st_ino == parent_inode
            and env["output_root"].is_dir()
            and not env["work_root"].exists()
        ):
            interrupted = True
            raise OSError("injected parent fsync interruption")
        original_fsync(fd)

    monkeypatch.setattr(q2_module.os, "fsync", fail_once_after_directory_rename)

    with pytest.raises(OSError, match="parent fsync interruption"):
        finalize_hy3_expert_q2(config)

    assert interrupted is True
    assert not env["output_root"].exists()
    assert env["work_root"].is_dir()
    assert finalize_hy3_expert_q2(config) == env["output_root"]
    assert verify_hy3_expert_q2(env["output_root"], deep=True)["passed"] is True


def test_finalize_rolls_back_when_publish_rename_returns_an_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    config = _complete_staged_conversion(env)
    original_rename = q2_module._exclusive_directory_rename
    interrupted = False

    def rename_then_interrupt(
        source_directory_fd,
        source,
        target_directory_fd,
        target,
    ):
        nonlocal interrupted
        result = original_rename(
            source_directory_fd,
            source,
            target_directory_fd,
            target,
        )
        if target == env["output_root"].name:
            interrupted = True
            raise KeyboardInterrupt("injected post-rename interruption")
        return result

    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        rename_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="post-rename interruption"):
        finalize_hy3_expert_q2(config)

    assert interrupted is True
    assert not env["output_root"].exists()
    assert env["work_root"].is_dir()

    monkeypatch.setattr(
        q2_module,
        "_exclusive_directory_rename",
        original_rename,
    )
    assert finalize_hy3_expert_q2(config) == env["output_root"]


@pytest.mark.parametrize(
    "corruption",
    ["sidecar", "resident", "ancillary", "provenance", "mtp", "extra_q4"],
)
def test_deep_verify_rejects_every_authoritative_output_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    env = _conversion_test_environment(tmp_path, monkeypatch)
    published = finalize_hy3_expert_q2(_complete_staged_conversion(env))
    manifest = ExpertManifest.from_dict(
        json.loads((published / "expert-manifest.json").read_text())
    )
    if corruption == "sidecar":
        target = published / "experts.bin"
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0xFF
        target.write_bytes(payload)
    elif corruption == "resident":
        target = published / manifest.resident_tensors[0].shard
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0xFF
        target.write_bytes(payload)
    elif corruption == "ancillary":
        (published / "config.json").write_bytes(b"corrupt")
    elif corruption == "provenance":
        target = published / "conversion-manifest.json"
        value = json.loads(target.read_text())
        value["derivation"]["external_q2_artifact_used"] = True
        target.write_text(json.dumps(value), encoding="utf-8")
    elif corruption == "mtp":
        (published / "mtp").mkdir()
    else:
        (published / "model-99999-of-99999.safetensors").write_bytes(b"q4")

    with pytest.raises(
        ValueError,
        match="sidecar|record|shard|hash|ancillary|provenance|inventory|MTP|unexpected",
    ):
        verify_hy3_expert_q2(published, deep=True)


def test_cpu_only_cli_phases_have_isolated_lazy_import_paths(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/build_hy3_expert_q2.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mtplx")
        for node in tree.body
    )
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        """
import builtins
_original_import = builtins.__import__
def _guard(name, *args, **kwargs):
    if name == 'mlx' or name.startswith('mlx.'):
        raise RuntimeError('forbidden MLX import in CPU phase')
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _guard
""".strip()
        + "\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing"
    common = [
        "--source-manifest",
        os.fspath(missing / "expert-manifest.json"),
        "--source-provenance",
        os.fspath(missing / "conversion-provenance.json"),
    ]
    commands = [
        ["preflight", os.fspath(missing), *common],
        ["stage", os.fspath(missing), os.fspath(tmp_path / "output"), *common],
        [
            "finalize",
            os.fspath(missing),
            os.fspath(tmp_path / "output"),
            *common,
            "--pilot-report",
            os.fspath(missing / "pilot.json"),
        ],
        ["verify", os.fspath(missing), "--deep"],
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (os.fspath(blocker), os.fspath(Path(__file__).resolve().parents[1]))
    )
    for command in commands:
        completed = subprocess.run(
            [sys.executable, os.fspath(script), *command],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        assert completed.returncode != 0
        assert "forbidden MLX import" not in completed.stderr
