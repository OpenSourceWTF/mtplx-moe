"""Exact Qwen3.6-35B-A3B whole-MoE construction and installation.

Checkpoint and model facts are validated once while constructing the
experimental route. Installed execution is added separately after the fixed
Metal stages pass their exact self-checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Callable, Literal

import mlx.core as mx
from mlx_lm.models.qwen3_next import swiglu

from .attention_context import current_attention_phase
from .kernels import a3b_whole_moe as kernel_module
from .moe_packed_projections import (
    PackedGateUpMLP,
    PackedSwitchGLU,
    _PackedDenseProjection,
    _PackedQuantizedProjection,
)
from .qwen_row_owned_router import _qwen_row_owned_route_unchecked


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


class A3BWholeMoeRouteError(RuntimeError):
    """A request reached a phase or row geometry outside its installed route."""


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
    stock_call: Callable[[Any, Any], Any]
    variant: A3BWholeMoeVariant
    router: ProjectionStorage
    routed_gate_up: ProjectionStorage
    routed_down: ProjectionStorage
    shared_gate_up: ProjectionStorage
    shared_down: ProjectionStorage
    shared_scalar_gate: ProjectionStorage


@dataclass(frozen=True)
class A3BWholeMoeInstallPlan:
    """All exact sparse-block ownership awaiting kernel self-checks."""

    target_bindings: tuple[A3BWholeMoeBinding, ...]
    mtp_bindings: tuple[A3BWholeMoeBinding, ...]


@dataclass(frozen=True)
class _TargetA3BWholeMoeRoute:
    """Prebound accepted execution plus the exact target M3/M2/M1 overrides.

    ``m3_call`` is the 3-row (k=2) verify override; it is bound and
    dispatched on 3-row decode forwards but reached only once the depth-2
    request/route/generation path selects a 3-row verify geometry.  The
    shipped K1 paths (m2_call at 2 rows, m1_call at 1 row) are unchanged.
    """

    accepted_call: Callable[[Any, Any], Any]
    m2_call: Callable[[Any], Any]
    m1_call: Callable[[Any], Any]
    m3_call: Callable[[Any], Any] | None = None


_SELFCHECK_LIMITS = {
    "expert_ids": 0.0,
    "route_scores": 0.0078125,
    "shared_gate": 0.0625,
    "activations": 0.125,
    "stage3_output": 0.5,
    "output": 0.5,
}
_STATS: dict[str, Any] = {
    "enabled": False,
    "installed": False,
    "installation_status": "disabled",
    "installation_error": None,
    "target_blocks": 0,
    "mtp_blocks": 0,
    "accepted_mtp_blocks": 0,
    "validated_contract": None,
    "selfcheck_lanes": {},
    "selfcheck_dmax": {},
    "selfcheck_components": {},
}


def _validated_contract() -> dict[str, Any]:
    return {
        "model": "Qwen3.6-35B-A3B",
        "hidden_size": 2048,
        "target_blocks": 40,
        "mtp_blocks": 1,
        "experts": 256,
        "top_k": 8,
        "normalized_scores": True,
        "intermediate_size": 512,
        "target": {
            "router": "affine_q8_group64_[256,512]",
            "routed_gate_up": "affine_q4_group64_[256,1024,256]",
            "routed_down": "affine_q4_group64_[256,2048,64]",
            "shared_gate_up": "affine_q4_group64_[1024,256]",
            "shared_down": "affine_q4_group64_[2048,64]",
            "shared_scalar_gate": "affine_q8_group64_[1,512]",
            "routes": {
                "M1": "m2_row_arithmetic_at_rows1_stage23_row_parity",
                "M2": "fixed_tiled_stage1_stage23_one_read_row_paired",
            },
        },
        "mtp": {
            "router": "dense_bf16_[256,2048]",
            "routed_gate_up": "affine_q4_group32_[256,1024,256]",
            "routed_down": "affine_q4_group32_[256,2048,64]",
            "shared_gate_up": "dense_bf16_[1024,2048]",
            "shared_down": "dense_bf16_[2048,512]",
            "shared_scalar_gate": "dense_bf16_[1,2048]",
            "routes": {"M1": "accepted_row_owned_router_combine"},
        },
        "materialized_activation": "bf16_[M,9,512]",
        "eliminated": ("bf16_[M,8,2048]", "bf16_[M,2048]_shared"),
        "prefill": "packed_stock",
    }


def a3b_whole_moe_enabled() -> bool:
    """Read the experimental switch at the construction boundary only."""

    return os.environ.get("MTPLX_A3B_WHOLE_MOE_FUSION", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def a3b_whole_moe_m3_selfcheck_enabled() -> bool:
    """Whether the opt-in k=2 (m3) install-time byte-exactness gate is armed.

    When on, the whole-MoE selfcheck computes the 3-row parity lanes and the
    install requires them at limit 0.0 (fail-closed) -- the k=2 driving path
    arms this so a byte-inexact m3 kernel can never serve.
    """

    return os.environ.get(
        "MTPLX_A3B_WHOLE_MOE_M3_SELFCHECK", ""
    ).strip().lower() in {"1", "true", "on", "yes"}


def validate_a3b_whole_moe_load_options(
    *,
    mtp_adapter: Any | None,
    merge_mtp_adapter: bool,
) -> None:
    """Reject load options that would mutate storage after it is bound."""

    if a3b_whole_moe_enabled() and (
        mtp_adapter is not None or bool(merge_mtp_adapter)
    ):
        raise A3BWholeMoeConfigError(
            "whole-MoE does not support MTP adapters; use the exact checkpoint"
        )


def validate_a3b_whole_moe_request(
    *,
    verify_strategy: str,
    requested_speculative_depth: int,
    speculative_depth: int,
    verify_core: str,
    draft_core: str,
    compiled_target_prefix: bool,
    session_bank_present: bool,
    vision_splice_present: bool,
    prefill_layout: str,
) -> None:
    """Validate request-owned facts once before prefill or measured decode."""

    if verify_strategy != "target_prefix":
        raise A3BWholeMoeConfigError(
            "whole-MoE requires the exact target-prefix request route"
        )
    # K1 (depth 1, 2-row verify) is the shipped path; depth 2 (k=2, 3-row verify
    # via the m3 kernels) is the opt-in experiment lane.  The m3 fused route is
    # byte-exact per row to the M1 route (install parity lane), so a greedy k=2
    # stream stays byte-comparable to greedy generate_ar.
    if (
        requested_speculative_depth not in (1, 2)
        or speculative_depth not in (1, 2)
        or requested_speculative_depth != speculative_depth
    ):
        raise A3BWholeMoeConfigError(
            "whole-MoE requires exact K1 (verify len 2) or k=2 (verify len 3) "
            "ownership"
        )
    if verify_core != "stock":
        raise A3BWholeMoeConfigError(
            "whole-MoE requires the stock capture arithmetic contract"
        )
    if draft_core != "stock":
        raise A3BWholeMoeConfigError(
            "whole-MoE requires the stock draft arithmetic contract"
        )
    if not compiled_target_prefix:
        raise A3BWholeMoeConfigError(
            "whole-MoE requires the exact compiled target-prefix factory"
        )
    if session_bank_present:
        raise A3BWholeMoeConfigError(
            "whole-MoE requires cold prompt ownership for exact cache geometry"
        )
    if vision_splice_present:
        raise A3BWholeMoeConfigError(
            "whole-MoE does not install a vision request geometry"
        )
    if prefill_layout != "contiguous_dense_decode":
        raise A3BWholeMoeConfigError(
            "whole-MoE requires the contiguous dense request cache layout"
        )


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


def _require_no_additive_bias(module: Any, *, label: str) -> None:
    if getattr(module, "bias", None) is not None:
        raise A3BWholeMoeConfigError(f"{label} must not carry an additive bias")


def _require_quantized(
    module: Any,
    *,
    label: str,
    bits: int,
    group_size: int,
    weight_shape: tuple[int, ...],
    metadata_shape: tuple[int, ...],
) -> ProjectionStorage:
    _require_no_additive_bias(module, label=label)
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
    _require_no_additive_bias(module, label=label)
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


def _require_exact_packed_owners(
    block: Any,
    *,
    label: str,
    variant: A3BWholeMoeVariant,
) -> None:
    switch = block.switch_mlp
    shared = block.shared_expert
    if type(switch) is not PackedSwitchGLU:
        raise A3BWholeMoeConfigError(f"{label} requires exact PackedSwitchGLU ownership")
    if type(shared) is not PackedGateUpMLP:
        raise A3BWholeMoeConfigError(f"{label} requires exact PackedGateUpMLP ownership")
    if int(switch._split_at) != 512 or int(shared._split_at) != 512:
        raise A3BWholeMoeConfigError(f"{label} requires exact gate-before-up split 512")
    if type(switch.gate_up_proj) is not _PackedQuantizedProjection:
        raise A3BWholeMoeConfigError(
            f"{label} requires exact packed routed quantized projection ownership"
        )
    expected_shared_type = (
        _PackedQuantizedProjection
        if variant == "target_q8g64_q4g64"
        else _PackedDenseProjection
    )
    if type(shared.gate_up_proj) is not expected_shared_type:
        raise A3BWholeMoeConfigError(
            f"{label} requires exact packed shared projection ownership"
        )


def _require_common_block(block: Any, *, label: str, norm_weight: Any) -> None:
    block_type = type(block)
    if hasattr(block_type, "_mtplx_a3b_whole_moe_route") or hasattr(
        block_type, "_mtplx_a3b_router_route"
    ):
        raise A3BWholeMoeConfigError(f"{label} has an installed route conflict")
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
    _require_exact_packed_owners(
        block,
        label=label,
        variant="target_q8g64_q4g64",
    )
    switch = block.switch_mlp
    shared = block.shared_expert
    return A3BWholeMoeBinding(
        block=block,
        stock_call=type(block).__call__,
        variant="target_q8g64_q4g64",
        router=_require_quantized(
            block.gate,
            label="target router",
            bits=8,
            group_size=64,
            weight_shape=(256, 512),
            metadata_shape=(256, 32),
        ),
        routed_gate_up=_require_quantized(
            switch.gate_up_proj,
            label="target routed gate/up",
            bits=4,
            group_size=64,
            weight_shape=(256, 1024, 256),
            metadata_shape=(256, 1024, 32),
        ),
        routed_down=_require_quantized(
            switch.down_proj,
            label="target routed down",
            bits=4,
            group_size=64,
            weight_shape=(256, 2048, 64),
            metadata_shape=(256, 2048, 8),
        ),
        shared_gate_up=_require_quantized(
            shared.gate_up_proj,
            label="target shared gate/up",
            bits=4,
            group_size=64,
            weight_shape=(1024, 256),
            metadata_shape=(1024, 32),
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
    _require_exact_packed_owners(
        block,
        label=label,
        variant="mtp_dense_q4g32_dense",
    )
    switch = block.switch_mlp
    shared = block.shared_expert
    return A3BWholeMoeBinding(
        block=block,
        stock_call=type(block).__call__,
        variant="mtp_dense_q4g32_dense",
        router=_require_dense(
            block.gate,
            label="MTP router",
            weight_shape=(256, 2048),
        ),
        routed_gate_up=_require_quantized(
            switch.gate_up_proj,
            label="MTP routed gate/up",
            bits=4,
            group_size=32,
            weight_shape=(256, 1024, 256),
            metadata_shape=(256, 1024, 64),
        ),
        routed_down=_require_quantized(
            switch.down_proj,
            label="MTP routed down",
            bits=4,
            group_size=32,
            weight_shape=(256, 2048, 64),
            metadata_shape=(256, 2048, 16),
        ),
        shared_gate_up=_require_dense(
            shared.gate_up_proj,
            label="MTP shared gate/up",
            weight_shape=(1024, 2048),
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
    plan = A3BWholeMoeInstallPlan(
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
    _STATS.update(
        {
            "enabled": True,
            "installed": False,
            "installation_status": "awaiting_selfcheck",
            "installation_error": None,
            "target_blocks": len(plan.target_bindings),
            "mtp_blocks": 0,
            "accepted_mtp_blocks": len(plan.mtp_bindings),
            "validated_contract": _validated_contract(),
            "selfcheck_lanes": {},
            "selfcheck_dmax": {},
            "selfcheck_components": {},
        }
    )
    return plan


def _row_owned_stage1_unchecked(
    value: Any,
    binding: A3BWholeMoeBinding,
    *,
    rows: int,
) -> tuple[Any, Any, Any]:
    """Run exact row-owned routing and the independent shared scalar gate."""

    probabilities = mx.softmax(binding.block.gate(value), axis=-1, precise=True)
    expert_ids, route_scores = _qwen_row_owned_route_unchecked(
        probabilities,
        rows=rows,
    )
    shared_gate = binding.block.shared_expert_gate(value)
    return (
        expert_ids.reshape(rows, 8),
        route_scores.reshape(rows, 8),
        shared_gate.reshape(rows, 1),
    )


def _packed_stage2_unchecked(
    value: Any,
    expert_ids: Any,
    binding: A3BWholeMoeBinding,
    *,
    rows: int,
) -> Any:
    """Run exact packed selected/shared gate-up and BF16 SwiGLU."""

    packed_routed = binding.block.switch_mlp.gate_up_proj.gather(
        mx.expand_dims(value, (-2, -3)),
        expert_ids.reshape(1, rows, 8),
        False,
    )
    routed_gate, routed_up = mx.split(packed_routed, [512], axis=-1)
    routed_activations = swiglu(routed_gate, routed_up).reshape(rows, 8, 512)
    packed_shared = binding.block.shared_expert.gate_up_proj(value)
    shared_activation_gate, shared_activation_up = mx.split(
        packed_shared,
        [512],
        axis=-1,
    )
    shared_activations = swiglu(
        shared_activation_gate,
        shared_activation_up,
    ).reshape(rows, 1, 512)
    return mx.concatenate(
        [routed_activations, shared_activations],
        axis=1,
    )


def _packed_stage12_unchecked(
    value: Any,
    binding: A3BWholeMoeBinding,
    *,
    rows: int,
) -> tuple[Any, Any, Any, Any]:
    """Compose the two proven MLX boundaries before the fused-down kernel."""

    expert_ids, route_scores, shared_gate = _row_owned_stage1_unchecked(
        value,
        binding,
        rows=rows,
    )
    activations = _packed_stage2_unchecked(
        value,
        expert_ids,
        binding,
        rows=rows,
    )
    return expert_ids, route_scores, shared_gate, activations


def _target_m1_route(binding: A3BWholeMoeBinding) -> Callable[[Any], Any]:
    """Bind the M1 decode route with M2-arithmetic stage2/stage3 at ROWS=1.

    Per-row bit-parity with the M2 verify route is the AR-exactness
    contract: a greedy K1 request must byte-match greedy generate_ar of the
    same serving configuration, so the single-row decode forwards must
    compute the exact per-row arithmetic of the M2 verify kernels.  Stage1
    (precise-softmax row-owned routing + shared scalar gate) is bit-equal
    to the fused M2 stage1 and its ids are exact integers; stage2/stage3
    run the M2 kernel sources compiled at ROWS=1 (the routed loop and the
    row-0 chains are textually identical to the M2 kernels).  The install
    selfcheck enforces row parity at limit 0.0.
    """

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _row_owned_stage1_unchecked(
            value,
            binding,
            rows=1,
        )
        activations = kernel_module.target_m2r1_stage2(
            value, expert_ids, binding
        )
        return kernel_module.target_m2r1_stage3(
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
        ).reshape(1, 1, 2048)

    return call


def _target_m2_route(binding: A3BWholeMoeBinding) -> Callable[[Any], Any]:
    """Bind the three exact one-read target-M2 stages at construction."""

    return kernel_module.bind_target_m2(binding)


def _target_m3_route(binding: A3BWholeMoeBinding) -> Callable[[Any], Any]:
    """Bind the four row-tripled target-M3 stages (k=2, 3-row verify).

    Rows 0/1 of the M3 stage3 are byte-identical to the shipped M2 kernel and
    every row is byte-identical to the M1 route (parity lane, limit 0.0), so a
    greedy k=2 stream stays byte-comparable to greedy generate_ar of the same
    configuration -- the same AR-exactness contract the K1 stack proves at 2
    rows, extended to the 3-row ``[primary, d1, d2]`` verify geometry.
    """

    return kernel_module.bind_target_m3(binding)


def _mtp_m1_route(binding: A3BWholeMoeBinding) -> Callable[[Any], Any]:
    """Bind exact MTP M1 row routing, packed gate/up, and fused down."""

    stage3 = kernel_module.bind_mtp_m1_stage3(binding)

    def call(value: Any):
        expert_ids, route_scores, shared_gate, activations = (
            _packed_stage12_unchecked(value, binding, rows=1)
        )
        return stage3(
            activations,
            expert_ids,
            route_scores,
            shared_gate,
        ).reshape(1, 1, 2048)

    return call


def _check_whole_moe_m1_m2_row_parity(binding: A3BWholeMoeBinding) -> float:
    """Max abs diff between M2 rows and the M1 route run per row (limit 0.0)."""

    fixture = mx.arange(2 * 2048, dtype=mx.float32).reshape(1, 2, 2048)
    value = (
        mx.sin(fixture * 0.013) * 0.25
        + mx.cos(fixture * 0.007) * 0.0625
    ).astype(mx.bfloat16)
    m2_out = _target_m2_route(binding)(value)
    m1_route = _target_m1_route(binding)
    row0 = m1_route(value[:, 0:1, :])
    row1 = m1_route(value[:, 1:2, :])
    mx.eval(m2_out, row0, row1)
    return max(
        _max_abs_diff(m2_out[:, 0:1, :], row0),
        _max_abs_diff(m2_out[:, 1:2, :], row1),
    )


def _check_whole_moe_m1_m3_row_parity(binding: A3BWholeMoeBinding) -> float:
    """Max abs diff between the 3-row M3 verify and the M1 route (limit 0.0).

    The k=2 AR-exactness contract: each row of the 3-row ``[primary, d1, d2]``
    verify must reproduce the single-row M1 decode route EXACTLY, or a greedy
    k=2 request cannot byte-match greedy generate_ar of the same config.
    """

    fixture = mx.arange(3 * 2048, dtype=mx.float32).reshape(1, 3, 2048)
    value = (
        mx.sin(fixture * 0.013) * 0.25
        + mx.cos(fixture * 0.007) * 0.0625
    ).astype(mx.bfloat16)
    m3_out = _target_m3_route(binding)(value)
    m1_route = _target_m1_route(binding)
    row0 = m1_route(value[:, 0:1, :])
    row1 = m1_route(value[:, 1:2, :])
    row2 = m1_route(value[:, 2:3, :])
    mx.eval(m3_out, row0, row1, row2)
    return max(
        _max_abs_diff(m3_out[:, 0:1, :], row0),
        _max_abs_diff(m3_out[:, 1:2, :], row1),
        _max_abs_diff(m3_out[:, 2:3, :], row2),
    )


def _check_whole_moe_m2_m3_prefix_parity(binding: A3BWholeMoeBinding) -> float:
    """Rows 0/1 of the 3-row M3 verify must match the shipped 2-row M2 (limit 0.0).

    Guards the "rows 0/1 stay identical to the shipped paired stage-3" claim:
    the row-tripled kernel must not perturb the first two verify rows.
    """

    fixture = mx.arange(3 * 2048, dtype=mx.float32).reshape(1, 3, 2048)
    value = (
        mx.sin(fixture * 0.013) * 0.25
        + mx.cos(fixture * 0.007) * 0.0625
    ).astype(mx.bfloat16)
    m3_out = _target_m3_route(binding)(value)
    m2_out = _target_m2_route(binding)(value[:, 0:2, :])
    mx.eval(m3_out, m2_out)
    return max(
        _max_abs_diff(m3_out[:, 0:1, :], m2_out[:, 0:1, :]),
        _max_abs_diff(m3_out[:, 1:2, :], m2_out[:, 1:2, :]),
    )


def check_a3b_whole_moe_m3_row_parity(plan: A3BWholeMoeInstallPlan) -> dict[str, float]:
    """Aggregate the k=2 kernel parity lanes across every target binding.

    Exposed for the k=2 build / GPU smoke; not part of the K1 install gate, so
    the shipped depth-1 path never pays the extra M3 compiles.
    """

    m1_m3 = 0.0
    m2_m3 = 0.0
    for binding in plan.target_bindings:
        try:
            m1_m3 = max(m1_m3, _check_whole_moe_m1_m3_row_parity(binding))
            m2_m3 = max(m2_m3, _check_whole_moe_m2_m3_prefix_parity(binding))
        except Exception:
            m1_m3 = float("inf")
            m2_m3 = float("inf")
            break
    return {
        "a3b_whole_moe_target_m1_m3_row_parity": m1_m3,
        "a3b_whole_moe_target_m2_m3_prefix_parity": m2_m3,
    }


def _max_abs_diff(candidate: Any, reference: Any) -> float:
    if tuple(candidate.shape) != tuple(reference.shape):
        return float("inf")
    difference = mx.abs(
        candidate.astype(mx.float32) - reference.astype(mx.float32)
    )
    value = float(difference.max())
    return value if math.isfinite(value) else float("inf")


def _check_whole_moe_lane(
    binding: A3BWholeMoeBinding,
    *,
    rows: int,
) -> dict[str, float]:
    """Check all three stages and the compiled route on deterministic input."""

    fixture = mx.arange(rows * 2048, dtype=mx.float32).reshape(1, rows, 2048)
    value = (
        mx.sin(fixture * 0.013) * 0.25
        + mx.cos(fixture * 0.007) * 0.0625
    ).astype(mx.bfloat16)

    if binding.variant == "target_q8g64_q4g64":
        stage3 = (
            kernel_module.target_m1_stage3
            if rows == 1
            else kernel_module.target_m2_stage3
        )
        route = _target_m1_route(binding) if rows == 1 else _target_m2_route(binding)
    else:
        stage3 = kernel_module.mtp_m1_stage3
        route = _mtp_m1_route(binding)

    if binding.variant == "target_q8g64_q4g64" and rows == 2:
        expert_ids, route_scores, shared_gate = (
            kernel_module.target_m2_stage1(value, binding)
        )
        activations = kernel_module.target_m2_stage2(
            value,
            expert_ids,
            binding,
        )
    else:
        expert_ids, route_scores, shared_gate, activations = (
            _packed_stage12_unchecked(value, binding, rows=rows)
        )
    probabilities = mx.softmax(binding.block.gate(value), axis=-1, precise=True)
    reference_ids = mx.argpartition(probabilities, kth=-8, axis=-1)[..., -8:]
    reference_scores = mx.take_along_axis(
        probabilities,
        reference_ids,
        axis=-1,
    )
    reference_scores = reference_scores / reference_scores.sum(
        axis=-1,
        keepdims=True,
    )
    reference_ids = reference_ids.reshape(rows, 8)
    reference_scores = reference_scores.reshape(rows, 8)
    reference_shared_gate = binding.block.shared_expert_gate(value).reshape(rows, 1)

    packed_routed = binding.block.switch_mlp.gate_up_proj.gather(
        mx.expand_dims(value, (-2, -3)),
        reference_ids.reshape(1, rows, 8),
        False,
    )
    routed_gate, routed_up = mx.split(packed_routed, [512], axis=-1)
    packed_shared = binding.block.shared_expert.gate_up_proj(value)
    shared_activation_gate, shared_activation_up = mx.split(
        packed_shared,
        [512],
        axis=-1,
    )
    from mlx_lm.models.qwen3_next import swiglu

    routed_activations = swiglu(routed_gate, routed_up).reshape(rows, 8, 512)
    shared_activations = swiglu(
        shared_activation_gate,
        shared_activation_up,
    ).reshape(rows, 1, 512)
    reference_activations = mx.concatenate(
        [routed_activations, shared_activations],
        axis=1,
    )
    stage3_candidate = stage3(
        reference_activations,
        reference_ids,
        reference_scores,
        reference_shared_gate,
        binding,
    ).reshape(*value.shape)

    compiled_candidate = mx.compile(route)(value)
    compiled_reference = mx.compile(lambda current: binding.block(current))(value)
    mx.eval(
        expert_ids,
        route_scores,
        shared_gate,
        reference_ids,
        reference_scores,
        reference_shared_gate,
        activations,
        reference_activations,
        stage3_candidate,
        compiled_candidate,
        compiled_reference,
    )
    if not bool(mx.array_equal(expert_ids, reference_ids).item()):
        return {component: float("inf") for component in _SELFCHECK_LIMITS}
    return {
        "expert_ids": 0.0,
        "route_scores": _max_abs_diff(route_scores, reference_scores),
        "shared_gate": _max_abs_diff(shared_gate, reference_shared_gate),
        "activations": _max_abs_diff(activations, reference_activations),
        "stage3_output": _max_abs_diff(stage3_candidate, compiled_reference),
        "output": _max_abs_diff(compiled_candidate, compiled_reference),
    }


def run_a3b_whole_moe_selfcheck(
    plan: A3BWholeMoeInstallPlan,
    base_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add exact model-bound lane verdicts before installation."""

    report = dict(base_report or {})
    lanes = dict(report.get("lanes") or {})
    dmax = dict(report.get("dmax") or {})
    component_report: dict[str, dict[str, float]] = {}
    checks = (
        (
            "a3b_whole_moe_target_m2",
            tuple((binding, 2) for binding in plan.target_bindings),
        ),
    )
    for lane, lane_checks in checks:
        aggregate = {component: 0.0 for component in _SELFCHECK_LIMITS}
        for binding, rows in lane_checks:
            try:
                differences = _check_whole_moe_lane(binding, rows=rows)
            except Exception:
                differences = {
                    component: float("inf") for component in _SELFCHECK_LIMITS
                }
            for component in aggregate:
                aggregate[component] = max(
                    aggregate[component],
                    float(differences[component]),
                )
        component_report[lane] = aggregate
        dmax[lane] = max(aggregate.values())
        lanes[lane] = (
            "ok"
            if all(
                aggregate[component] <= limit
                for component, limit in _SELFCHECK_LIMITS.items()
            )
            else "failed"
        )
    # M1/M2 per-row bit-parity: the AR-exactness contract.  The M1 decode
    # route must reproduce each M2 verify row EXACTLY (limit 0.0) or a greedy
    # K1 request cannot byte-match greedy generate_ar of the same config.
    parity_lane = "a3b_whole_moe_target_m1_m2_row_parity"
    parity_dmax = 0.0
    for binding in plan.target_bindings:
        try:
            parity_dmax = max(
                parity_dmax, _check_whole_moe_m1_m2_row_parity(binding)
            )
        except Exception:
            parity_dmax = float("inf")
            break
    component_report[parity_lane] = {"row_parity": parity_dmax}
    dmax[parity_lane] = parity_dmax
    lanes[parity_lane] = "ok" if parity_dmax == 0.0 else "failed"
    # k=2 (m3) kernel parity is opt-in: the shipped depth-1 install never pays
    # the extra M3 compiles.  Enabled by the k=2 build / a GPU smoke to prove
    # each 3-row verify row bit-matches the M1 route (limit 0.0).
    if a3b_whole_moe_m3_selfcheck_enabled():
        m3_parity = check_a3b_whole_moe_m3_row_parity(plan)
        for m3_lane, m3_value in m3_parity.items():
            component_report[m3_lane] = {"row_parity": m3_value}
            dmax[m3_lane] = m3_value
            lanes[m3_lane] = "ok" if m3_value == 0.0 else "failed"
    report["lanes"] = lanes
    report["dmax"] = dmax
    report["a3b_whole_moe_components"] = component_report
    return report


def _target_a3b_whole_moe_call(self: Any, value: Any) -> Any:
    """Override exact target decode forwards after construction.

    Verify rows (2) run the fused M2 route; single decode rows (the route's
    repair forward and pure-AR decode) run the M1 route, whose per-row
    arithmetic bit-matches M2 -- the whole decode-time model function is one
    consistent arithmetic, which is what makes the greedy K1 stream
    byte-comparable to greedy generate_ar of the same configuration.
    Prefill keeps the stock path (identical for every entrypoint).
    """

    route = type(self)._mtplx_a3b_whole_moe_route
    phase = current_attention_phase()
    if phase in ("decode_verify", "ar_decode"):
        rows = math.prod(int(dimension) for dimension in value.shape[:-1])
        if rows == 2:
            return route.m2_call(value)
        if rows == 1:
            return route.m1_call(value)
        if rows == 3 and route.m3_call is not None:
            # k=2 verify [primary, d1, d2]; dormant until the depth-2
            # request/route/generation path emits a 3-row decode forward.
            return route.m3_call(value)
    return route.accepted_call(self, value)


def _installed_class(
    base_class: type,
    route: _TargetA3BWholeMoeRoute,
    *,
    index: int,
) -> type:
    return type(
        f"A3BWholeMoeTargetM2_{index}_{base_class.__name__}",
        (base_class,),
        {
            "__module__": __name__,
            "__call__": _target_a3b_whole_moe_call,
            "_mtplx_a3b_whole_moe_route": route,
        },
    )


def _route_for_binding(
    binding: A3BWholeMoeBinding,
) -> _TargetA3BWholeMoeRoute:
    return _TargetA3BWholeMoeRoute(
        accepted_call=type(binding.block).__call__,
        m2_call=_target_m2_route(binding),
        m1_call=_target_m1_route(binding),
        m3_call=_target_m3_route(binding),
    )


def install_a3b_whole_moe(
    plan: A3BWholeMoeInstallPlan,
    selfcheck_report: dict[str, Any] | None,
    *,
    compiled_preflight: Callable[[], dict[str, str]],
) -> dict[str, Any]:
    """Commit the 40 target-M2 overrides after exact compiled preflight."""

    lanes = {} if selfcheck_report is None else selfcheck_report.get("lanes", {})
    required_lanes = (
        "a3b_whole_moe_target_m2",
        "a3b_whole_moe_target_m1_m2_row_parity",
    )
    if a3b_whole_moe_m3_selfcheck_enabled():
        # k=2 driving path armed: the 3-row kernels must be byte-exact (each
        # verify row == M1 route, rows 0/1 == shipped M2) or install fails.
        required_lanes = required_lanes + (
            "a3b_whole_moe_target_m1_m3_row_parity",
            "a3b_whole_moe_target_m2_m3_prefix_parity",
        )
    failed = tuple(lane for lane in required_lanes if lanes.get(lane) != "ok")
    if failed:
        _STATS.update(
            {
                "enabled": True,
                "installed": False,
                "installation_status": "configuration_error",
                "installation_error": (
                    "whole-MoE self-check failed for " + ", ".join(failed)
                ),
            }
        )
        raise A3BWholeMoeConfigError(_STATS["installation_error"])

    full_graph_lanes = (
        "a3b_whole_moe_target_prefix_full_graph_m1",
        "a3b_whole_moe_target_prefix_full_graph_m2",
    )

    prepared = tuple(
        (
            binding.block,
            type(binding.block),
            _installed_class(
                type(binding.block),
                _route_for_binding(binding),
                index=index,
            ),
        )
        for index, binding in enumerate(plan.target_bindings)
    )
    changed: list[tuple[Any, type]] = []
    try:
        for block, original_class, installed_class in prepared:
            block.__class__ = installed_class
            changed.append((block, original_class))
        full_graph_report = dict(compiled_preflight())
        failed_full_graph = tuple(
            lane for lane in full_graph_lanes if full_graph_report.get(lane) != "ok"
        )
        if failed_full_graph:
            raise A3BWholeMoeConfigError(
                "whole-MoE full compiled target-prefix preflight failed for "
                + ", ".join(failed_full_graph)
            )
    except Exception as exc:
        for block, original_class in reversed(changed):
            block.__class__ = original_class
        message = "whole-MoE full compiled target-prefix preflight failed"
        _STATS.update(
            {
                "enabled": True,
                "installed": False,
                "installation_status": "configuration_error",
                "installation_error": f"{message}: {exc}",
            }
        )
        raise A3BWholeMoeConfigError(_STATS["installation_error"]) from exc

    # Surface any opt-in k=2 (m3) parity lanes the selfcheck computed, so a
    # build / GPU smoke can read the on-device byte-exactness result.  Never
    # gating (not in required_lanes) -- the shipped depth-1 install is unchanged.
    report_lanes = (selfcheck_report or {}).get("lanes", {}) or {}
    m3_lanes = {
        lane: status
        for lane, status in report_lanes.items()
        if "_m3_" in lane
    }
    installed_lanes = {
        **{lane: lanes[lane] for lane in required_lanes},
        **{lane: full_graph_report[lane] for lane in full_graph_lanes},
        **m3_lanes,
    }
    _STATS.update(
        {
            "enabled": True,
            "installed": True,
            "installation_status": "installed",
            "installation_error": None,
            "target_blocks": len(plan.target_bindings),
            "mtp_blocks": 0,
            "accepted_mtp_blocks": len(plan.mtp_bindings),
            "validated_contract": _validated_contract(),
            "selfcheck_lanes": installed_lanes,
            "selfcheck_dmax": {
                **{
                    lane: (selfcheck_report or {}).get("dmax", {}).get(lane)
                    for lane in required_lanes
                },
                **{lane: 0.0 for lane in full_graph_lanes},
                **{
                    lane: (selfcheck_report or {}).get("dmax", {}).get(lane)
                    for lane in m3_lanes
                },
            },
            "selfcheck_components": dict(
                (selfcheck_report or {}).get("a3b_whole_moe_components") or {}
            ),
        }
    )
    return a3b_whole_moe_stats()


def a3b_whole_moe_stats() -> dict[str, Any]:
    """Return construction status without execution-path counters."""

    return dict(_STATS)


def _reset_a3b_whole_moe_for_tests() -> None:
    _STATS.update(
        {
            "enabled": False,
            "installed": False,
            "installation_status": "disabled",
            "installation_error": None,
            "target_blocks": 0,
            "mtp_blocks": 0,
            "accepted_mtp_blocks": 0,
            "validated_contract": None,
            "selfcheck_lanes": {},
            "selfcheck_dmax": {},
            "selfcheck_components": {},
        }
    )
