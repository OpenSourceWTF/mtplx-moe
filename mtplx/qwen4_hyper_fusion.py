"""Construction-bound fixed M=2 Qwen4 hyper-connection route."""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from .kernels import qwen4_hyper_fusion as kernels


class Qwen4HyperFusionConfigError(RuntimeError):
    """The exact Qwen4 M=2 hyper route cannot be installed."""


HYPER_FUSION_ENV = "MTPLX_QWEN4_HYPER_M2"


def qwen4_hyper_fusion_enabled() -> bool:
    return os.environ.get(HYPER_FUSION_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _bind(module: Any) -> Callable[[Any, Any], tuple[Any, Any, Any]]:
    kernel_call = kernels.bind_m2(module)

    def call(hidden: Any, normed: Any):
        mixed, inject = kernel_call(normed)
        return mixed, hidden, inject

    return call


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


def _validate_projection(
    projection: Any,
    *,
    label: str,
    weight_shape: tuple[int, int],
    metadata_shape: tuple[int, int],
) -> None:
    if (
        int(getattr(projection, "bits", 0) or 0) != 4
        or int(getattr(projection, "group_size", 0) or 0) != 32
        or str(getattr(projection, "mode", "")) != "affine"
        or getattr(projection, "weight", None) is None
        or getattr(projection, "scales", None) is None
        or getattr(projection, "biases", None) is None
        or _shape(projection.weight) != weight_shape
        or _shape(projection.scales) != metadata_shape
        or _shape(projection.biases) != metadata_shape
        or projection.weight.dtype != mx.uint32
        or projection.scales.dtype != mx.bfloat16
        or projection.biases.dtype != mx.bfloat16
        or getattr(projection, "bias", None) is not None
    ):
        raise Qwen4HyperFusionConfigError(
            f"{label} does not match affine q4/g32 storage"
        )


def _validate_hyper(module: Any, index: int) -> None:
    if int(getattr(module, "hc", 0) or 0) != 4 or int(
        getattr(module, "d", 0) or 0
    ) != 2560:
        raise Qwen4HyperFusionConfigError(
            f"Qwen4 block {index} hyper geometry is not 4x2560"
        )
    _validate_projection(
        getattr(module, "input_mix_weight_down", None),
        label=f"Qwen4 block {index} hyper down",
        weight_shape=(320, 1280),
        metadata_shape=(320, 320),
    )
    _validate_projection(
        getattr(module, "input_mix_weight_up", None),
        label=f"Qwen4 block {index} hyper up",
        weight_shape=(10240, 40),
        metadata_shape=(10240, 10),
    )
    _validate_projection(
        getattr(module, "block_inject_weight", None),
        label=f"Qwen4 block {index} hyper inject",
        weight_shape=(4, 1280),
        metadata_shape=(4, 320),
    )


def _stock_hyper(module: Any, hidden: Any, normed: Any):
    weight = mx.sigmoid(
        module.input_mix_weight_up(
            nn.silu(module.input_mix_weight_down(normed) / 4)
        )
    )
    weight = weight.reshape(1, 2, 4, 2560)
    mixed = (weight * normed.reshape(1, 2, 4, 2560)).mean(axis=-2)
    inject = 2 * mx.sigmoid(module.block_inject_weight(normed) / 4)
    return mixed, hidden, inject


def _selfcheck(module: Any, binding: Callable[[Any, Any], Any]) -> float:
    indices = mx.arange(2 * 10240, dtype=mx.float32)
    normed = mx.sin(indices / 97.0).reshape(1, 2, 10240).astype(mx.bfloat16)
    hidden = mx.cos(indices / 89.0).reshape(1, 2, 10240).astype(mx.bfloat16)
    expected = _stock_hyper(module, hidden, normed)
    actual = binding(hidden, normed)
    mx.eval(*expected, *actual)
    if not bool(mx.array_equal(actual[1], expected[1])):
        raise Qwen4HyperFusionConfigError(
            "Qwen4 M=2 hyper route changed residual ownership"
        )
    deltas = {
        label: float(
            mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))
        )
        for label, got, want in (
            ("mixed", actual[0], expected[0]),
            ("inject", actual[2], expected[2]),
        )
    }
    for label, delta in deltas.items():
        if not math.isfinite(delta):
            raise Qwen4HyperFusionConfigError(
                f"Qwen4 M=2 hyper {label} self-check produced a non-finite delta"
            )
        if delta > 0.03125:
            raise Qwen4HyperFusionConfigError(
                f"Qwen4 M=2 hyper {label} self-check exceeded BF16 tolerance: "
                f"{delta}"
            )
    return max(deltas.values())


def configure_qwen4_hyper_fusion(
    model: Any,
    *,
    validate_storage: bool = True,
    run_selfcheck: bool = True,
) -> dict[str, Any]:
    """Validate once, then install the direct fixed M=2 route on all blocks."""

    inner = getattr(getattr(model, "language_model", None), "model", None)
    layers = tuple(getattr(inner, "layers", ()))
    if len(layers) != 48:
        raise Qwen4HyperFusionConfigError("Qwen4 hyper fusion requires 48 blocks")

    if validate_storage:
        for index, layer in enumerate(layers):
            _validate_hyper(layer.mlp_hyper_connection, index)
    bindings = tuple(_bind(layer.mlp_hyper_connection) for layer in layers)
    selfcheck_dmax = None
    if run_selfcheck:
        selfcheck_dmax = max(
            _selfcheck(layer.mlp_hyper_connection, binding)
            for layer, binding in zip(layers, bindings)
        )
    for layer, binding in zip(layers, bindings):
        layer.mlp_hyper_connection._mtplx_m2_hyper_call = binding

    return {
        "installed": True,
        "installed_blocks": len(bindings),
        "rows": 2,
        "selfcheck_dmax": selfcheck_dmax,
    }


__all__ = [
    "HYPER_FUSION_ENV",
    "Qwen4HyperFusionConfigError",
    "configure_qwen4_hyper_fusion",
    "qwen4_hyper_fusion_enabled",
]
