"""Exact Qwen3.6-35B-A3B whole-MoE construction and installation.

Checkpoint and model facts are validated once while constructing the
experimental route. Installed execution is added separately after the fixed
Metal stages pass their exact self-checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

import mlx.core as mx


A3BWholeMoeVariant = Literal[
    "target_q8g64_q4g64",
    "mtp_dense_q4g32_dense",
]
_A3B_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class A3BWholeMoeConfigError(RuntimeError):
    """The external model contract cannot install the whole-MoE route."""


@dataclass(frozen=True)
class ProjectionStorage:
    """Model-owned fixed projection arrays bound at construction."""

    weight: Any
    scales: Any | None = None
    biases: Any | None = None


@dataclass(frozen=True)
class A3BWholeMoeBinding:
    """One model-owned sparse block and its fixed storage variant."""

    block: Any
    variant: A3BWholeMoeVariant
    router: ProjectionStorage
    routed_gate: ProjectionStorage
    routed_up: ProjectionStorage
    routed_down: ProjectionStorage
    shared_gate: ProjectionStorage
    shared_up: ProjectionStorage
    shared_down: ProjectionStorage
    shared_scalar_gate: ProjectionStorage


@dataclass(frozen=True)
class A3BWholeMoeInstallPlan:
    """All exact sparse-block ownership awaiting kernel self-checks."""

    target_bindings: tuple[A3BWholeMoeBinding, ...]
    mtp_bindings: tuple[A3BWholeMoeBinding, ...]


def a3b_whole_moe_enabled() -> bool:
    """Read the experimental switch at the construction boundary only."""

    return os.environ.get("MTPLX_A3B_WHOLE_MOE_FUSION", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
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
) -> ProjectionStorage:
    if (
        int(getattr(module, "bits", 0) or 0) != bits
        or int(getattr(module, "group_size", 0) or 0) != group_size
        or str(getattr(module, "mode", "")) != "affine"
    ):
        raise A3BWholeMoeConfigError(
            f"{label} must be affine q{bits}/group{group_size}"
        )
    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    if (
        weight is None
        or scales is None
        or biases is None
        or _shape(weight) != weight_shape
        or _shape(scales) != metadata_shape
        or _shape(biases) != metadata_shape
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise A3BWholeMoeConfigError(
            f"{label} storage does not match the exact packed tensor contract"
        )
    return ProjectionStorage(weight=weight, scales=scales, biases=biases)


def _require_dense(
    module: Any,
    *,
    label: str,
    weight_shape: tuple[int, ...],
) -> ProjectionStorage:
    weight = getattr(module, "weight", None)
    if (
        weight is None
        or _shape(weight) != weight_shape
        or weight.dtype != mx.bfloat16
        or int(getattr(module, "bits", 0) or 0) != 0
        or getattr(module, "scales", None) is not None
        or getattr(module, "biases", None) is not None
    ):
        raise A3BWholeMoeConfigError(
            f"{label} must be dense BF16 with shape {weight_shape}"
        )
    return ProjectionStorage(weight=weight)


def _require_common_block(block: Any, *, label: str, norm_weight: Any) -> None:
    for attribute in (
        "gate",
        "switch_mlp",
        "shared_expert",
        "shared_expert_gate",
        "num_experts",
        "top_k",
        "norm_topk_prob",
        "sharding_group",
    ):
        if not hasattr(block, attribute):
            raise A3BWholeMoeConfigError(f"{label} is missing {attribute}")
    if int(block.num_experts) != 256:
        raise A3BWholeMoeConfigError(f"{label} requires 256 experts")
    if int(block.top_k) != 8:
        raise A3BWholeMoeConfigError(f"{label} requires exact top-k 8")
    if not bool(block.norm_topk_prob):
        raise A3BWholeMoeConfigError(f"{label} requires score normalization")
    if block.sharding_group is not None:
        raise A3BWholeMoeConfigError(f"{label} does not support sharding")
    if (
        norm_weight is None
        or _shape(norm_weight) != (2048,)
        or norm_weight.dtype != mx.bfloat16
    ):
        raise A3BWholeMoeConfigError(
            f"{label} requires BF16 hidden width 2048 ownership"
        )


def _target_binding(block: Any, *, norm_weight: Any) -> A3BWholeMoeBinding:
    label = "target whole-MoE"
    _require_common_block(block, label=label, norm_weight=norm_weight)
    switch = block.switch_mlp
    shared = block.shared_expert
    return A3BWholeMoeBinding(
        block=block,
        variant="target_q8g64_q4g64",
        router=_require_quantized(
            block.gate,
            label="target router",
            bits=8,
            group_size=64,
            weight_shape=(256, 512),
            metadata_shape=(256, 32),
        ),
        routed_gate=_require_quantized(
            switch.gate_proj,
            label="target routed gate",
            bits=4,
            group_size=64,
            weight_shape=(256, 512, 256),
            metadata_shape=(256, 512, 32),
        ),
        routed_up=_require_quantized(
            switch.up_proj,
            label="target routed up",
            bits=4,
            group_size=64,
            weight_shape=(256, 512, 256),
            metadata_shape=(256, 512, 32),
        ),
        routed_down=_require_quantized(
            switch.down_proj,
            label="target routed down",
            bits=4,
            group_size=64,
            weight_shape=(256, 2048, 64),
            metadata_shape=(256, 2048, 8),
        ),
        shared_gate=_require_quantized(
            shared.gate_proj,
            label="target shared gate",
            bits=4,
            group_size=64,
            weight_shape=(512, 256),
            metadata_shape=(512, 32),
        ),
        shared_up=_require_quantized(
            shared.up_proj,
            label="target shared up",
            bits=4,
            group_size=64,
            weight_shape=(512, 256),
            metadata_shape=(512, 32),
        ),
        shared_down=_require_quantized(
            shared.down_proj,
            label="target shared down",
            bits=4,
            group_size=64,
            weight_shape=(2048, 64),
            metadata_shape=(2048, 8),
        ),
        shared_scalar_gate=_require_quantized(
            block.shared_expert_gate,
            label="target shared scalar gate",
            bits=8,
            group_size=64,
            weight_shape=(1, 512),
            metadata_shape=(1, 32),
        ),
    )


def _mtp_binding(block: Any, *, norm_weight: Any) -> A3BWholeMoeBinding:
    label = "MTP whole-MoE"
    _require_common_block(block, label=label, norm_weight=norm_weight)
    switch = block.switch_mlp
    shared = block.shared_expert
    return A3BWholeMoeBinding(
        block=block,
        variant="mtp_dense_q4g32_dense",
        router=_require_dense(
            block.gate,
            label="MTP router",
            weight_shape=(256, 2048),
        ),
        routed_gate=_require_quantized(
            switch.gate_proj,
            label="MTP routed gate",
            bits=4,
            group_size=32,
            weight_shape=(256, 512, 256),
            metadata_shape=(256, 512, 64),
        ),
        routed_up=_require_quantized(
            switch.up_proj,
            label="MTP routed up",
            bits=4,
            group_size=32,
            weight_shape=(256, 512, 256),
            metadata_shape=(256, 512, 64),
        ),
        routed_down=_require_quantized(
            switch.down_proj,
            label="MTP routed down",
            bits=4,
            group_size=32,
            weight_shape=(256, 2048, 64),
            metadata_shape=(256, 2048, 16),
        ),
        shared_gate=_require_dense(
            shared.gate_proj,
            label="MTP shared gate",
            weight_shape=(512, 2048),
        ),
        shared_up=_require_dense(
            shared.up_proj,
            label="MTP shared up",
            weight_shape=(512, 2048),
        ),
        shared_down=_require_dense(
            shared.down_proj,
            label="MTP shared down",
            weight_shape=(2048, 512),
        ),
        shared_scalar_gate=_require_dense(
            block.shared_expert_gate,
            label="MTP shared scalar gate",
            weight_shape=(1, 2048),
        ),
    )


def _require_exact_config(config: dict[str, Any]) -> None:
    text = config.get("text_config")
    if (
        config.get("model_type") != "qwen3_5_moe"
        or config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or not isinstance(text, dict)
        or text.get("model_type") != "qwen3_5_moe_text"
    ):
        raise A3BWholeMoeConfigError(
            "whole-MoE requires the exact Qwen3.6-35B-A3B model architecture"
        )
    norm_topk_prob = bool(text.get("norm_topk_prob", True))
    if (
        int(text.get("hidden_size", -1)) != 2048
        or int(text.get("num_hidden_layers", -1)) != 40
        or tuple(text.get("layer_types", ())) != _A3B_LAYER_TYPES
        or int(text.get("num_experts", -1)) != 256
        or int(text.get("num_experts_per_tok", -1)) != 8
        or not norm_topk_prob
        or int(text.get("moe_intermediate_size", -1)) != 512
        or int(text.get("shared_expert_intermediate_size", -1)) != 512
        or int(text.get("mtp_num_hidden_layers", -1)) != 1
    ):
        raise A3BWholeMoeConfigError(
            "whole-MoE config does not match the exact A3B topology"
        )


def prepare_a3b_whole_moe(
    model: Any,
    *,
    config: dict[str, Any],
) -> A3BWholeMoeInstallPlan | None:
    """Collect exact target/MTP ownership without mutating the model."""

    if not a3b_whole_moe_enabled():
        return None

    _require_exact_config(config)
    text_model = getattr(model, "language_model", None)
    inner = getattr(text_model, "model", None)
    target_layers = tuple(getattr(inner, "layers", ()) or ())
    mtp = getattr(model, "mtp", None)
    mtp_layers = tuple(getattr(mtp, "layers", ()) or ())
    if len(target_layers) != 40:
        raise A3BWholeMoeConfigError(
            "whole-MoE requires exactly 40 target sparse blocks"
        )
    if len(mtp_layers) != 1:
        raise A3BWholeMoeConfigError(
            "whole-MoE requires exactly one MTP sparse block"
        )
    actual_types = tuple(
        "linear_attention"
        if bool(getattr(layer, "is_linear", hasattr(layer, "linear_attn")))
        else "full_attention"
        for layer in target_layers
    )
    if actual_types != _A3B_LAYER_TYPES or not hasattr(mtp_layers[0], "self_attn"):
        raise A3BWholeMoeConfigError(
            "whole-MoE target/MTP layer ownership does not match A3B topology"
        )
    return A3BWholeMoeInstallPlan(
        target_bindings=tuple(
            _target_binding(
                layer.mlp,
                norm_weight=layer.post_attention_layernorm.weight,
            )
            for layer in target_layers
        ),
        mtp_bindings=tuple(
            _mtp_binding(
                layer.mlp,
                norm_weight=layer.post_attention_layernorm.weight,
            )
            for layer in mtp_layers
        ),
    )
