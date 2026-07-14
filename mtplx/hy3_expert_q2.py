"""Bounded conversion primitives for the explicit Hy3 expert-Q2 lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .expert_manifest import ExpertRecord, TensorSegment


SOURCE_MODEL_KEY = "hy3-expert-only-q4"
TARGET_MODEL_KEY = "hy3-expert-q2"
SOURCE_MANIFEST_SHA256 = (
    "507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43"
)

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "scales", "biases")
_DTYPES = ("U32", "BF16", "BF16")
_GROUP_SIZE = 64


@dataclass(frozen=True)
class ProjectionDiagnostics:
    component: str
    cosine_q4_q2: float
    normalized_error_q4_q2: float
    finite: bool


def _byte_view(
    payload: bytes | memoryview,
    *,
    component: str,
    expected_bytes: int,
) -> memoryview:
    try:
        view = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{component} must be a contiguous byte buffer") from exc
    if view.nbytes != expected_bytes:
        qualifier = "short read" if view.nbytes < expected_bytes else "oversized read"
        raise ValueError(
            f"{qualifier} for {component}: got {view.nbytes} bytes; "
            f"expected {expected_bytes}"
        )
    return view


def _projection_shapes(
    *,
    input_size: int,
    output_size: int,
    bits: int,
    group_size: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if (
        isinstance(input_size, bool)
        or not isinstance(input_size, int)
        or input_size <= 0
    ):
        raise ValueError("input_size must be a positive integer")
    if (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size <= 0
    ):
        raise ValueError("output_size must be a positive integer")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size != _GROUP_SIZE
    ):
        raise ValueError(f"group_size must be {_GROUP_SIZE}")
    if input_size % group_size:
        raise ValueError("input_size must be divisible by group_size")
    if input_size * bits % 32:
        raise ValueError("input_size is not representable by packed uint32 weights")
    return (
        (output_size, input_size * bits // 32),
        (output_size, input_size // group_size),
        (output_size, input_size // group_size),
    )


def _shape_bytes(shape: tuple[int, int], dtype: str) -> int:
    item_size = 4 if dtype == "U32" else 2
    return shape[0] * shape[1] * item_size


def requantize_projection_q4_to_q2(
    weight_bytes: bytes | memoryview,
    scales_bytes: bytes | memoryview,
    biases_bytes: bytes | memoryview,
    *,
    projection: str,
    input_size: int,
    output_size: int,
    group_size: int = 64,
) -> tuple[tuple[bytes, bytes, bytes], ProjectionDiagnostics]:
    """Convert one affine-Q4 projection to canonical affine-Q2 bytes.

    MLX is imported only inside this numerical boundary. All MLX arrays are
    released and its cache is cleared before returning so records can be
    processed one projection at a time without retaining dense expert weights.
    """

    if projection not in _PROJECTIONS:
        raise ValueError(f"unsupported expert projection {projection!r}")
    source_shapes = _projection_shapes(
        input_size=input_size,
        output_size=output_size,
        bits=4,
        group_size=group_size,
    )
    target_shapes = _projection_shapes(
        input_size=input_size,
        output_size=output_size,
        bits=2,
        group_size=group_size,
    )
    source_views = tuple(
        _byte_view(
            payload,
            component=f"{projection}.{leaf}",
            expected_bytes=_shape_bytes(shape, dtype),
        )
        for payload, leaf, shape, dtype in zip(
            (weight_bytes, scales_bytes, biases_bytes),
            _LEAVES,
            source_shapes,
            _DTYPES,
            strict=True,
        )
    )

    import mlx.core as mx

    q4_weight = None
    q4_scales = None
    q4_biases = None
    dense = None
    q2_weight = None
    q2_scales = None
    q2_biases = None
    q2_dense = None
    source_fp32 = None
    target_fp32 = None
    try:
        q4_weight = mx.array(
            np.frombuffer(source_views[0], dtype="<u4")
            .copy()
            .reshape(source_shapes[0]),
            dtype=mx.uint32,
        )

        def decode_bf16(view: memoryview, shape: tuple[int, int]):
            words = mx.array(
                np.frombuffer(view, dtype="<u2").copy().reshape(shape),
                dtype=mx.uint16,
            )
            return words.view(mx.bfloat16)

        q4_scales = decode_bf16(source_views[1], source_shapes[1])
        q4_biases = decode_bf16(source_views[2], source_shapes[2])
        mx.eval(q4_weight, q4_scales, q4_biases)
        if not bool(
            mx.all(mx.isfinite(q4_scales)).item()
            and mx.all(mx.isfinite(q4_biases)).item()
        ):
            raise ValueError(f"projection {projection} has non-finite Q4 values")

        dense = mx.dequantize(
            q4_weight,
            q4_scales,
            q4_biases,
            bits=4,
            group_size=group_size,
            mode="affine",
        )
        mx.eval(dense)
        if not bool(mx.all(mx.isfinite(dense)).item()):
            raise ValueError(f"projection {projection} has non-finite Q4 values")

        q2_weight, q2_scales, q2_biases = mx.quantize(
            dense,
            bits=2,
            group_size=group_size,
            mode="affine",
        )
        mx.eval(q2_weight, q2_scales, q2_biases)
        if q2_weight.dtype != mx.uint32 or tuple(q2_weight.shape) != target_shapes[0]:
            raise ValueError(
                f"projection {projection} produced invalid Q2 weight metadata"
            )
        for label, value, shape in (
            ("scales", q2_scales, target_shapes[1]),
            ("biases", q2_biases, target_shapes[2]),
        ):
            if value.dtype != mx.bfloat16 or tuple(value.shape) != shape:
                raise ValueError(
                    f"projection {projection} produced invalid Q2 {label} metadata"
                )

        q2_dense = mx.dequantize(
            q2_weight,
            q2_scales,
            q2_biases,
            bits=2,
            group_size=group_size,
            mode="affine",
        )
        mx.eval(q2_dense)
        finite = bool(
            mx.all(mx.isfinite(q2_scales)).item()
            and mx.all(mx.isfinite(q2_biases)).item()
            and mx.all(mx.isfinite(q2_dense)).item()
        )
        if not finite:
            raise ValueError(f"projection {projection} produced non-finite Q2 values")

        source_fp32 = dense.astype(mx.float32).reshape(-1)
        target_fp32 = q2_dense.astype(mx.float32).reshape(-1)
        source_norm = float(mx.linalg.norm(source_fp32).item())
        target_norm = float(mx.linalg.norm(target_fp32).item())
        dot = float(mx.sum(source_fp32 * target_fp32).item())
        error_norm = float(mx.linalg.norm(source_fp32 - target_fp32).item())
        if source_norm == 0.0:
            cosine = 1.0 if target_norm == 0.0 else 0.0
            normalized_error = error_norm
        else:
            cosine = dot / (source_norm * target_norm) if target_norm else 0.0
            normalized_error = error_norm / source_norm
        if not all(math.isfinite(value) for value in (cosine, normalized_error)):
            raise ValueError(f"projection {projection} produced non-finite diagnostics")

        output = (
            np.array(q2_weight, copy=True).astype("<u4", copy=False).tobytes(),
            np.array(q2_scales.view(mx.uint16), copy=True)
            .astype("<u2", copy=False)
            .tobytes(),
            np.array(q2_biases.view(mx.uint16), copy=True)
            .astype("<u2", copy=False)
            .tobytes(),
        )
        expected_output_bytes = tuple(
            _shape_bytes(shape, dtype)
            for shape, dtype in zip(target_shapes, _DTYPES, strict=True)
        )
        if tuple(len(item) for item in output) != expected_output_bytes:
            raise ValueError(
                f"projection {projection} Q2 serialization has the wrong byte counts"
            )
        diagnostics = ProjectionDiagnostics(
            component=projection,
            cosine_q4_q2=cosine,
            normalized_error_q4_q2=normalized_error,
            finite=True,
        )
        return output, diagnostics
    finally:
        q4_weight = q4_scales = q4_biases = None
        dense = q2_weight = q2_scales = q2_biases = q2_dense = None
        source_fp32 = target_fp32 = None
        value = None
        mx.clear_cache()


def _canonical_q4_metadata(
    *,
    hidden_size: int,
    expert_hidden_size: int,
    group_size: int,
) -> tuple[tuple[str, str, tuple[int, int], int], ...]:
    dimensions = {
        "gate_proj": (hidden_size, expert_hidden_size),
        "up_proj": (hidden_size, expert_hidden_size),
        "down_proj": (expert_hidden_size, hidden_size),
    }
    expected = []
    for projection in _PROJECTIONS:
        input_size, output_size = dimensions[projection]
        shapes = _projection_shapes(
            input_size=input_size,
            output_size=output_size,
            bits=4,
            group_size=group_size,
        )
        expected.extend(
            (
                f"{projection}.{leaf}",
                dtype,
                shape,
                _shape_bytes(shape, dtype),
            )
            for leaf, dtype, shape in zip(_LEAVES, _DTYPES, shapes, strict=True)
        )
    return tuple(expected)


def requantize_expert_record_q4_to_q2(
    record: ExpertRecord,
    read_component: Callable[[TensorSegment], bytes | memoryview],
    write_component: Callable[[str, bytes], None],
    *,
    hidden_size: int,
    expert_hidden_size: int,
    group_size: int = 64,
) -> tuple[ProjectionDiagnostics, ...]:
    """Validate and convert a record in gate/up/down projection order."""

    expected = _canonical_q4_metadata(
        hidden_size=hidden_size,
        expert_hidden_size=expert_hidden_size,
        group_size=group_size,
    )
    if len(record.segments) != len(expected):
        raise ValueError("expert record must contain nine canonical Q4 components")
    for segment, (component, dtype, shape, length) in zip(
        record.segments,
        expected,
        strict=True,
    ):
        if segment.component != component:
            raise ValueError(
                "expert record component order is not canonical: "
                f"expected {component}; found {segment.component}"
            )
        if segment.dtype != dtype:
            raise ValueError(
                f"expert record component {component} dtype {segment.dtype!r} "
                f"does not match {dtype!r}"
            )
        if tuple(segment.shape) != shape:
            raise ValueError(
                f"expert record component {component} shape {segment.shape!r} "
                f"does not match {shape!r}"
            )
        if segment.length != length:
            raise ValueError(
                f"expert record component {component} length {segment.length} "
                f"does not match {length}"
            )
    expected_record_bytes = sum(item[3] for item in expected)
    if record.logical_bytes != expected_record_bytes:
        raise ValueError(
            f"expert record logical_bytes {record.logical_bytes} does not match "
            f"canonical Q4 length {expected_record_bytes}"
        )

    dimensions = {
        "gate_proj": (hidden_size, expert_hidden_size),
        "up_proj": (hidden_size, expert_hidden_size),
        "down_proj": (expert_hidden_size, hidden_size),
    }
    diagnostics = []
    for projection_index, projection in enumerate(_PROJECTIONS):
        projection_segments = record.segments[
            projection_index * 3 : projection_index * 3 + 3
        ]
        source_payloads = tuple(
            bytes(
                _byte_view(
                    read_component(segment),
                    component=segment.component,
                    expected_bytes=segment.length,
                )
            )
            for segment in projection_segments
        )
        input_size, output_size = dimensions[projection]
        converted, projection_diagnostics = requantize_projection_q4_to_q2(
            *source_payloads,
            projection=projection,
            input_size=input_size,
            output_size=output_size,
            group_size=group_size,
        )
        for segment, payload in zip(projection_segments, converted, strict=True):
            write_component(segment.component, payload)
        diagnostics.append(projection_diagnostics)
        del source_payloads, converted, projection_diagnostics
    return tuple(diagnostics)
