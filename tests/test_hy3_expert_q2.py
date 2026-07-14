from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.expert_manifest import ExpertRecord, TensorSegment
from mtplx.hy3_expert_q2 import (
    ProjectionDiagnostics,
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL_KEY,
    TARGET_MODEL_KEY,
    requantize_expert_record_q4_to_q2,
    requantize_projection_q4_to_q2,
)


_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "scales", "biases")
_DTYPES = ("U32", "BF16", "BF16")


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


def test_canonical_record_converts_and_writes_one_projection_at_a_time(
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
        *canonical[:3],
        "clear",
        *canonical[3:6],
        "clear",
        *canonical[6:],
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
