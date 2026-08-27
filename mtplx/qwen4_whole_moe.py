"""Construction-installed exact Qwen4 two-row whole-MoE route."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import mlx.core as mx

from .kernels import qwen4_whole_moe as kernels


WHOLE_MOE_ENV = "MTPLX_QWEN4_WHOLE_MOE_M2"


class Qwen4WholeMoeConfigError(RuntimeError):
    """The exact Qwen4 whole-MoE route cannot be installed."""


@dataclass(frozen=True)
class _Binding:
    router: Any
    routed: Any
    shared: Any
    shared_gate: Any


@dataclass(frozen=True)
class _Route:
    accepted_call: Callable[[Any, Any], Any]
    m2_call: Callable[[Any], Any]


_STATS: dict[str, Any] = {
    "enabled": False,
    "installed": False,
    "installed_blocks": 0,
    "selfcheck_dmax": None,
    "geometry": None,
}


def qwen4_whole_moe_enabled() -> bool:
    return os.environ.get(WHOLE_MOE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


def _require_quantized(
    module: Any,
    *,
    label: str,
    bits: int,
    group_size: int,
    weight_shape: tuple[int, ...],
    metadata_shape: tuple[int, ...],
) -> None:
    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    if (
        int(getattr(module, "bits", 0) or 0) != bits
        or int(getattr(module, "group_size", 0) or 0) != group_size
        or str(getattr(module, "mode", "")) != "affine"
        or weight is None
        or scales is None
        or biases is None
        or _shape(weight) != weight_shape
        or _shape(scales) != metadata_shape
        or _shape(biases) != metadata_shape
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or getattr(module, "bias", None) is not None
    ):
        raise Qwen4WholeMoeConfigError(
            f"{label} does not match affine q{bits}/g{group_size} storage"
        )


def _validate_config(config: dict[str, Any]) -> None:
    text = config.get("text_config")
    if (
        config.get("model_type") != "qwen4_exp"
        or not isinstance(text, dict)
        or text.get("model_type") != "qwen4_exp_text"
        or int(text.get("hidden_size", -1)) != 2560
        or int(text.get("num_hidden_layers", -1)) != 48
        or int(text.get("num_experts", -1)) != 512
        or int(text.get("num_experts_per_tok", -1)) != 10
        or int(text.get("moe_intermediate_size", -1)) != 640
        or int(text.get("shared_expert_intermediate_size", -1)) != 640
    ):
        raise Qwen4WholeMoeConfigError(
            "whole-MoE M=2 requires Qwen4 2560/640/512/top-10"
        )


def _validate_block(block: Any, index: int) -> None:
    gate = getattr(block, "gate", None)
    gate_weight = getattr(gate, "weight", None)
    if (
        int(getattr(block, "top_k", 0) or 0) != 10
        or gate_weight is None
        or _shape(gate_weight) != (512, 2560)
        or gate_weight.dtype != mx.bfloat16
        or getattr(gate, "bias", None) is not None
    ):
        raise Qwen4WholeMoeConfigError(
            f"target block {index} router is not dense BF16 [512,2560]"
        )
    routed = getattr(block, "switch_mlp", None)
    shared = getattr(block, "shared_expert", None)
    if routed is None or shared is None:
        raise Qwen4WholeMoeConfigError(f"target block {index} is incomplete")
    for name in ("gate_proj", "up_proj"):
        _require_quantized(
            getattr(routed, name, None),
            label=f"target block {index} routed {name}",
            bits=4,
            group_size=32,
            weight_shape=(512, 640, 320),
            metadata_shape=(512, 640, 80),
        )
        _require_quantized(
            getattr(shared, name, None),
            label=f"target block {index} shared {name}",
            bits=8,
            group_size=128,
            weight_shape=(640, 640),
            metadata_shape=(640, 20),
        )
    _require_quantized(
        getattr(routed, "down_proj", None),
        label=f"target block {index} routed down",
        bits=4,
        group_size=32,
        weight_shape=(512, 2560, 80),
        metadata_shape=(512, 2560, 20),
    )
    _require_quantized(
        getattr(shared, "down_proj", None),
        label=f"target block {index} shared down",
        bits=8,
        group_size=128,
        weight_shape=(2560, 160),
        metadata_shape=(2560, 5),
    )
    _require_quantized(
        getattr(block, "shared_expert_gate", None),
        label=f"target block {index} shared scalar gate",
        bits=8,
        group_size=64,
        weight_shape=(1, 640),
        metadata_shape=(1, 40),
    )


def _m2_call(block: Any, binding: _Binding, value: Any) -> Any:
    logits, shared_gate = kernels.stage1(value, binding)
    expert_ids = mx.argpartition(-logits, block.top_k - 1, axis=-1)[
        ..., : block.top_k
    ]
    route_scores = mx.softmax(
        mx.take_along_axis(logits, expert_ids, axis=-1),
        axis=-1,
        precise=True,
    )
    activations = kernels.stage2(value, expert_ids, binding)
    output = kernels.stage3(
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
    )
    return output.reshape(value.shape)


def _installed_call(self: Any, value: Any) -> Any:
    rows = 1
    for dimension in value.shape[:-1]:
        rows *= int(dimension)
    route = type(self)._mtplx_qwen4_whole_moe_route
    if rows == 2:
        return route.m2_call(value)
    return route.accepted_call(self, value)


def _bind(block: Any) -> _Binding:
    return _Binding(
        router=block.gate,
        routed=block.switch_mlp,
        shared=block.shared_expert,
        shared_gate=block.shared_expert_gate,
    )


def _selfcheck(block: Any, accepted_call: Callable[[Any, Any], Any]) -> float:
    values = mx.sin(mx.arange(2 * 2560, dtype=mx.float32) / 97.0)
    values = values.reshape(1, 2, 2560).astype(mx.bfloat16)
    expected = accepted_call(block, values)
    actual = _m2_call(block, _bind(block), values)
    mx.eval(expected, actual)
    dmax = float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))))
    if dmax > 0.5:
        raise Qwen4WholeMoeConfigError(
            f"whole-MoE M=2 self-check exceeded 0.5 BF16 tolerance: {dmax}"
        )
    return dmax


def configure_qwen4_whole_moe(
    model: Any,
    *,
    config: dict[str, Any],
    validate_storage: bool = True,
    run_selfcheck: bool = True,
) -> dict[str, Any]:
    """Validate once, self-check once, then install the direct M=2 route."""

    enabled = qwen4_whole_moe_enabled()
    _STATS.update(
        {
            "enabled": enabled,
            "installed": False,
            "installed_blocks": 0,
            "selfcheck_dmax": None,
            "geometry": None,
        }
    )
    if not enabled:
        return dict(_STATS)
    _validate_config(config)
    inner = getattr(getattr(model, "language_model", None), "model", None)
    layers = tuple(getattr(inner, "layers", ()) or ())
    if len(layers) != 48:
        raise Qwen4WholeMoeConfigError("whole-MoE M=2 requires 48 target layers")
    blocks = tuple(getattr(layer, "mlp", None) for layer in layers)
    if any(block is None for block in blocks):
        raise Qwen4WholeMoeConfigError("whole-MoE M=2 target block is missing")
    if validate_storage:
        for index, block in enumerate(blocks):
            _validate_block(block, index)

    accepted_call = type(blocks[0]).__call__
    dmax = _selfcheck(blocks[0], accepted_call) if run_selfcheck else None
    for index, block in enumerate(blocks):
        accepted_call = type(block).__call__
        binding = _bind(block)

        def m2_call(value: Any, *, _block=block, _binding=binding):
            return _m2_call(_block, _binding, value)

        route = _Route(accepted_call=accepted_call, m2_call=m2_call)
        installed_type = type(
            f"Qwen4WholeM2MoE_{index}_{type(block).__name__}",
            (type(block),),
            {
                "__call__": _installed_call,
                "_mtplx_qwen4_whole_moe_route": route,
            },
        )
        block.__class__ = installed_type

    _STATS.update(
        {
            "installed": True,
            "installed_blocks": 48,
            "selfcheck_dmax": dmax,
            "geometry": {
                "rows": 2,
                "hidden": 2560,
                "intermediate": 640,
                "experts": 512,
                "top_k": 10,
                "routed": "q4/g32",
                "shared": "q8/g128",
                "shared_gate": "q8/g64",
                "router": "dense_bf16",
            },
        }
    )
    return dict(_STATS)


__all__ = [
    "Qwen4WholeMoeConfigError",
    "WHOLE_MOE_ENV",
    "configure_qwen4_whole_moe",
    "qwen4_whole_moe_enabled",
]
