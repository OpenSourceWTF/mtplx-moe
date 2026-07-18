"""Experimental row-owned routing for the exact Qwen A3B model.

The model checkpoint owns the gate arithmetic and MoE topology.  Construction
validates those external facts once, an exact self-check validates the Metal
finalizer, and installation replaces all 40 target blocks plus the single MTP
block atomically.  Installed execution routes only on phase and logical rows;
it does not re-prove model facts or silently recover from a broken custom lane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import mlx.core as mx

from .attention_context import current_attention_phase


_EXPERTS = 256
_TOP_K = 8
_EXACT_ROWS = tuple(range(1, 17))
_SIMD_GROUPS = 8
_THREADS = _SIMD_GROUPS * 32
_COMBINE_HIDDEN = 2048
_COMBINE_TOP_K = 8
_KERNEL = None
_COMBINE_KERNEL = None
_A3B_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)
_STATS: dict[str, Any] = {
    "enabled": False,
    "installed": False,
    "installation_status": "disabled",
    "installation_error": None,
    "target_routers": 0,
    "mtp_routers": 0,
    "validated_contract": None,
}
_INSTALLED_ROUTERS: list[tuple[Any, type]] = []
_INSTALLED_CLASSES: dict[tuple[type, bool], type] = {}


class QwenRowOwnedRouterIneligible(ValueError):
    """An external model fact does not match the exact A3B router contract."""


class QwenRowOwnedRouterConfigError(RuntimeError):
    """The exact A3B row-owned router lane could not be installed."""


@dataclass(frozen=True)
class QwenRowOwnedRouterInstallPlan:
    """Validated A3B router ownership awaiting the exact Metal self-check."""

    target_blocks: tuple[Any, ...]
    mtp_blocks: tuple[Any, ...]
    combine_tail: bool


@dataclass(frozen=True)
class _InstalledA3BRouterRoute:
    stock_call: Any


def _a3b_router_contract(*, combine_tail: bool = False) -> dict[str, Any]:
    contract = {
        "model": "Qwen3.6-35B-A3B",
        "target_routers": 40,
        "mtp_routers": 1,
        "hidden_size": 2048,
        "experts": 256,
        "top_k": 8,
        "normalization": True,
        "input_dtype": "bfloat16",
        "target_gate": "affine_q8_group64_[256,2048]",
        "mtp_gate": "dense_bf16_[256,2048]",
        "routes": {
            "prefill": "stock",
            "decode_verify": list(_EXACT_ROWS),
            "ar_decode": list(_EXACT_ROWS),
            "other": "stock",
        },
    }
    if combine_tail:
        contract["combine_tail"] = {
            "decode_verify": [1, 2],
            "ar_decode": [1, 2],
            "other_rows": "stock_weighted_reduction",
        }
    return contract


def qwen_row_owned_router_enabled() -> bool:
    """Read the experimental switch at the construction boundary only."""
    return os.environ.get("MTPLX_QWEN_ROW_OWNED_ROUTER", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def qwen_combine_tail_enabled() -> bool:
    """Read the fixed M1/M2 combine switch at construction only."""
    return os.environ.get("MTPLX_QWEN_COMBINE_TAIL", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def qwen_row_owned_router_eligible(
    *,
    rows: int,
    experts: int,
    input_dtype: mx.Dtype,
    top_k: int,
    norm_topk_prob: bool,
    available: bool | None = None,
) -> bool:
    """Checked public helper; installed execution does not call this predicate."""
    supported = True if available is None else bool(available)
    return (
        supported
        and int(rows) in _EXACT_ROWS
        and int(experts) == _EXPERTS
        and input_dtype == mx.bfloat16
        and int(top_k) == _TOP_K
        and bool(norm_topk_prob)
    )


def qwen_row_owned_router_source() -> str:
    """Emit one threadgroup per probability row with no global protocol."""
    return f"""
        using namespace metal;

        constexpr int N = {_EXPERTS};
        constexpr int TOPK = {_TOP_K};
        constexpr int SIMD_GROUPS = {_SIMD_GROUPS};
        constexpr int LOCAL_CANDIDATES = SIMD_GROUPS * TOPK;

        uint row = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        threadgroup float local_probabilities[LOCAL_CANDIDATES];
        threadgroup int local_indices[LOCAL_CANDIDATES];
        threadgroup float merged_probabilities[TOPK];
        threadgroup int merged_indices[TOPK];

        int expert = int(simd_gid) * 32 + int(lane);
        float candidate_probability = float(probabilities[row * N + expert]);
        int candidate_index = expert;

        _Pragma("unroll")
        for (int rank = 0; rank < TOPK; ++rank) {{
            float winner_probability = simd_max(candidate_probability);
            float winner_index_value = simd_max(
                candidate_probability == winner_probability
                    ? float(candidate_index)
                    : -1.0f);
            int winner_index = int(winner_index_value);
            if (lane == 0) {{
                int destination = int(simd_gid) * TOPK + rank;
                local_probabilities[destination] = winner_probability;
                local_indices[destination] = winner_index;
            }}
            if (candidate_index == winner_index) {{
                candidate_probability = -INFINITY;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {{
            int slot0 = int(lane);
            int slot1 = int(lane) + 32;
            float probability0 = local_probabilities[slot0];
            float probability1 = local_probabilities[slot1];
            int index0 = local_indices[slot0];
            int index1 = local_indices[slot1];

            _Pragma("unroll")
            for (int rank = 0; rank < TOPK; ++rank) {{
                bool take1 = probability1 > probability0
                    || (probability1 == probability0 && index1 > index0);
                float lane_probability = take1 ? probability1 : probability0;
                int lane_index = take1 ? index1 : index0;
                float winner_probability = simd_max(lane_probability);
                float winner_index_value = simd_max(
                    lane_probability == winner_probability
                        ? float(lane_index)
                        : -1.0f);
                int winner_index = int(winner_index_value);
                if (lane == 0) {{
                    merged_probabilities[rank] = winner_probability;
                    merged_indices[rank] = winner_index;
                }}
                if (lane_index == winner_index) {{
                    if (take1) {{
                        probability1 = -INFINITY;
                    }} else {{
                        probability0 = -INFINITY;
                    }}
                }}
            }}

            if (lane == 0) {{
                // MLX's top partition is ascending. Preserve that order and
                // BF16 denominator accumulation exactly.
                bfloat rounded_denominator = bfloat(0.0f);
                _Pragma("unroll")
                for (int output = 0; output < TOPK; ++output) {{
                    rounded_denominator = bfloat(
                        float(rounded_denominator)
                        + merged_probabilities[TOPK - 1 - output]);
                }}
                _Pragma("unroll")
                for (int output = 0; output < TOPK; ++output) {{
                    int source = TOPK - 1 - output;
                    int destination = int(row) * TOPK + output;
                    expert_ids[destination] = uint(merged_indices[source]);
                    route_scores[destination] = bfloat(
                        merged_probabilities[source]
                        / float(rounded_denominator));
                }}
            }}
        }}
    """


def qwen_combine_tail_source() -> str:
    """Preserve the A3B BF16 multiply/add order with one output owner."""
    return f"""
        using namespace metal;

        constexpr int TOPK = {_COMBINE_TOP_K};
        constexpr int HIDDEN = {_COMBINE_HIDDEN};

        uint output_index = thread_position_in_grid.x;
        uint row = output_index / HIDDEN;
        uint column = output_index - row * HIDDEN;

        bfloat accumulator = bfloat(0.0f);
        _Pragma("unroll")
        for (int expert = 0; expert < TOPK; ++expert) {{
            uint routed_index = (row * TOPK + uint(expert)) * HIDDEN + column;
            uint score_index = row * TOPK + uint(expert);
            bfloat product = bfloat(
                float(routed[routed_index]) * float(scores[score_index]));
            accumulator = bfloat(float(accumulator) + float(product));
        }}
        combined[output_index] = accumulator;
    """


def _build_qwen_row_owned_router_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen_a3b_row_owned_top8_bf16_exact",
            input_names=["probabilities"],
            output_names=["expert_ids", "route_scores"],
            source=qwen_row_owned_router_source(),
            ensure_row_contiguous=True,
        )
    return _KERNEL


def _build_qwen_combine_tail_kernel():
    global _COMBINE_KERNEL
    if _COMBINE_KERNEL is None:
        _COMBINE_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen_a3b_combine_bf16_exact",
            input_names=["routed", "scores"],
            output_names=["combined"],
            source=qwen_combine_tail_source(),
            ensure_row_contiguous=True,
        )
    return _COMBINE_KERNEL


def qwen_combine_tail_m1(
    routed: mx.array,
    scores: mx.array,
) -> mx.array:
    """Launch the installed BF16 [1,1,8,2048] combine geometry."""
    kernel = _build_qwen_combine_tail_kernel()
    (combined,) = kernel(
        inputs=[routed, scores],
        grid=(2048, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 2048)],
        output_dtypes=[mx.bfloat16],
    )
    return combined.reshape(1, 1, 2048)


def qwen_combine_tail_m2(
    routed: mx.array,
    scores: mx.array,
) -> mx.array:
    """Launch the installed BF16 [1,2,8,2048] combine geometry."""
    kernel = _build_qwen_combine_tail_kernel()
    (combined,) = kernel(
        inputs=[routed, scores],
        grid=(4096, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(2, 2048)],
        output_dtypes=[mx.bfloat16],
    )
    return combined.reshape(1, 2, 2048)


def _qwen_row_owned_route_unchecked(
    probabilities: mx.array,
    *,
    rows: int,
) -> tuple[mx.array, mx.array]:
    """Launch the construction-owned BF16 width-256 top-8 finalizer."""
    kernel = _build_qwen_row_owned_router_kernel()
    expert_ids, route_scores = kernel(
        inputs=[probabilities.reshape(rows, 256)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 8), (rows, 8)],
        output_dtypes=[mx.uint32, mx.bfloat16],
    )
    output_shape = (*probabilities.shape[:-1], 8)
    return expert_ids.reshape(output_shape), route_scores.reshape(output_shape)


def qwen_row_owned_route(
    probabilities: mx.array,
    *,
    top_k: int = _TOP_K,
    norm_topk_prob: bool = True,
    available: bool | None = None,
) -> tuple[mx.array, mx.array]:
    """Checked helper for tests and load-time self-checks."""
    if probabilities.ndim < 2 or int(probabilities.shape[-1]) != _EXPERTS:
        raise QwenRowOwnedRouterIneligible(
            "Qwen row-owned finalizer requires probability rows of width 256"
        )
    rows = math.prod(int(dimension) for dimension in probabilities.shape[:-1])
    if not qwen_row_owned_router_eligible(
        rows=rows,
        experts=int(probabilities.shape[-1]),
        input_dtype=probabilities.dtype,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        available=available,
    ):
        raise QwenRowOwnedRouterIneligible(
            "Qwen row-owned finalizer call is outside the exact M1-M16 lane"
        )
    return _qwen_row_owned_route_unchecked(probabilities, rows=rows)


def _installed_a3b_router_call(self: Any, value: mx.array) -> mx.array:
    """Route only on phase and row count after exact installation."""
    route = type(self)._mtplx_a3b_router_route
    phase = current_attention_phase()
    if phase == "prefill":
        return route.stock_call(self, value)
    rows = math.prod(int(dimension) for dimension in value.shape[:-1])
    if phase not in {"decode_verify", "ar_decode"} or rows not in _EXACT_ROWS:
        return route.stock_call(self, value)
    probabilities = mx.softmax(self.gate(value), axis=-1, precise=True)
    indices, scores = _qwen_row_owned_route_unchecked(probabilities, rows=rows)
    routed = self.switch_mlp(value, indices)
    routed = (routed * scores[..., None]).sum(axis=-2)
    shared = self.shared_expert(value)
    shared = mx.sigmoid(self.shared_expert_gate(value)) * shared
    return routed + shared


def _installed_a3b_router_combine_call(self: Any, value: mx.array) -> mx.array:
    """Use the installed K1 M1/M2 combine route without revalidating it."""
    route = type(self)._mtplx_a3b_router_route
    phase = current_attention_phase()
    if phase == "prefill":
        return route.stock_call(self, value)
    rows = math.prod(int(dimension) for dimension in value.shape[:-1])
    if phase not in {"decode_verify", "ar_decode"} or rows not in _EXACT_ROWS:
        return route.stock_call(self, value)
    probabilities = mx.softmax(self.gate(value), axis=-1, precise=True)
    indices, scores = _qwen_row_owned_route_unchecked(probabilities, rows=rows)
    routed = self.switch_mlp(value, indices)
    if rows == 1:
        routed = qwen_combine_tail_m1(routed, scores)
    elif rows == 2:
        routed = qwen_combine_tail_m2(routed, scores)
    else:
        routed = (routed * scores[..., None]).sum(axis=-2)
    shared = self.shared_expert(value)
    shared = mx.sigmoid(self.shared_expert_gate(value)) * shared
    return routed + shared


def _installed_class(base_class: type, *, combine_tail: bool) -> type:
    key = (base_class, combine_tail)
    installed = _INSTALLED_CLASSES.get(key)
    if installed is None:
        installed_call = (
            _installed_a3b_router_combine_call
            if combine_tail
            else _installed_a3b_router_call
        )
        installed = type(
            f"A3BInstalledRowOwned{'Combine' if combine_tail else ''}{base_class.__name__}",
            (base_class,),
            {
                "__module__": __name__,
                "__call__": installed_call,
                "_mtplx_a3b_router_route": _InstalledA3BRouterRoute(
                    stock_call=base_class.__call__
                ),
            },
        )
        _INSTALLED_CLASSES[key] = installed
    return installed


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


def _validate_target_gate(gate: Any) -> None:
    if (
        int(getattr(gate, "bits", 0) or 0) != 8
        or int(getattr(gate, "group_size", 0) or 0) != 64
        or str(getattr(gate, "mode", "")) != "affine"
    ):
        raise QwenRowOwnedRouterIneligible(
            "target gate must be affine q8/group64"
        )
    weight = getattr(gate, "weight", None)
    scales = getattr(gate, "scales", None)
    biases = getattr(gate, "biases", None)
    if (
        weight is None
        or scales is None
        or biases is None
        or _shape(weight) != (256, 512)
        or _shape(scales) != (256, 32)
        or _shape(biases) != (256, 32)
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise QwenRowOwnedRouterIneligible(
            "target gate storage must be q8/group64 [256,2048] with BF16 metadata"
        )


def _validate_mtp_gate(gate: Any) -> None:
    weight = getattr(gate, "weight", None)
    if weight is None or _shape(weight) != (256, 2048) or weight.dtype != mx.bfloat16:
        raise QwenRowOwnedRouterIneligible(
            "MTP gate must be dense BF16 [256,2048]"
        )
    if int(getattr(gate, "bits", 0) or 0) != 0:
        raise QwenRowOwnedRouterIneligible("MTP gate must remain dense")


def _validate_block(block: Any, *, target: bool, norm_weight: Any) -> None:
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
            raise QwenRowOwnedRouterIneligible(
                f"A3B sparse block is missing {attribute}"
            )
    if int(block.num_experts) != 256:
        raise QwenRowOwnedRouterIneligible("A3B router requires 256 experts")
    if int(block.top_k) != 8:
        raise QwenRowOwnedRouterIneligible("A3B router requires exact top-k 8")
    if not bool(block.norm_topk_prob):
        raise QwenRowOwnedRouterIneligible(
            "A3B router requires top-k probability normalization"
        )
    if block.sharding_group is not None:
        raise QwenRowOwnedRouterIneligible(
            "A3B row-owned router does not support model sharding"
        )
    if (
        norm_weight is None
        or _shape(norm_weight) != (2048,)
        or norm_weight.dtype != mx.bfloat16
    ):
        raise QwenRowOwnedRouterIneligible(
            "A3B router input ownership requires BF16 hidden width 2048"
        )
    if target:
        _validate_target_gate(block.gate)
    else:
        _validate_mtp_gate(block.gate)


def _clear_installed_routers() -> None:
    for block, original_class in reversed(_INSTALLED_ROUTERS):
        block.__class__ = original_class
    _INSTALLED_ROUTERS.clear()
    _INSTALLED_CLASSES.clear()


def _fail_router_configuration(message: str) -> None:
    _clear_installed_routers()
    _STATS["installed"] = False
    _STATS["installation_status"] = "configuration_error"
    _STATS["installation_error"] = str(message)
    raise QwenRowOwnedRouterConfigError(message)


def prepare_qwen_row_owned_routers(
    model: Any,
    *,
    config: dict[str, Any],
) -> QwenRowOwnedRouterInstallPlan | None:
    """Validate checkpoint-owned A3B facts once without installing execution."""
    _reset_qwen_row_owned_router_for_tests()
    router_enabled = qwen_row_owned_router_enabled()
    combine_tail = qwen_combine_tail_enabled()
    if combine_tail and not router_enabled:
        _fail_router_configuration(
            "A3B combine tail requires the row-owned router installation"
        )
    if not router_enabled:
        return None
    _STATS["enabled"] = True
    text_config = config.get("text_config")
    norm_config = (
        True
        if isinstance(text_config, dict) and "norm_topk_prob" not in text_config
        else bool(text_config.get("norm_topk_prob"))
        if isinstance(text_config, dict)
        else False
    )
    if (
        config.get("model_type") != "qwen3_5_moe"
        or config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or not isinstance(text_config, dict)
        or text_config.get("model_type") != "qwen3_5_moe_text"
        or int(text_config.get("hidden_size", -1)) != 2048
        or int(text_config.get("num_hidden_layers", -1)) != 40
        or tuple(text_config.get("layer_types", ())) != _A3B_LAYER_TYPES
        or int(text_config.get("num_experts", -1)) != 256
        or int(text_config.get("num_experts_per_tok", -1)) != 8
        or not norm_config
        or int(text_config.get("moe_intermediate_size", -1)) != 512
        or int(text_config.get("shared_expert_intermediate_size", -1)) != 512
        or int(text_config.get("mtp_num_hidden_layers", -1)) != 1
    ):
        _fail_router_configuration(
            "A3B row-owned router config requires the exact A3B model topology"
        )

    text_model = getattr(model, "language_model", None)
    inner = getattr(text_model, "model", None)
    target_layers = list(getattr(inner, "layers", None) or [])
    mtp = getattr(model, "mtp", None)
    mtp_layers = list(getattr(mtp, "layers", None) or [])
    target_blocks = [getattr(layer, "mlp", None) for layer in target_layers]
    mtp_blocks = [getattr(layer, "mlp", None) for layer in mtp_layers]
    actual_types = tuple(
        "linear_attention"
        if bool(getattr(layer, "is_linear", hasattr(layer, "linear_attn")))
        else "full_attention"
        for layer in target_layers
    )
    if (
        len(target_layers) != 40
        or len(mtp_layers) != 1
        or len(target_blocks) != 40
        or len(mtp_blocks) != 1
        or any(block is None for block in target_blocks + mtp_blocks)
        or actual_types != _A3B_LAYER_TYPES
        or not hasattr(mtp_layers[0], "self_attn")
    ):
        _fail_router_configuration(
            "A3B row-owned router requires exactly 40 target and one MTP sparse router"
        )
    try:
        for layer, block in zip(target_layers, target_blocks):
            norm = getattr(getattr(layer, "post_attention_layernorm", None), "weight", None)
            _validate_block(block, target=True, norm_weight=norm)
        mtp_norm = getattr(
            getattr(mtp_layers[0], "post_attention_layernorm", None), "weight", None
        )
        _validate_block(mtp_blocks[0], target=False, norm_weight=mtp_norm)
    except Exception as exc:
        _fail_router_configuration(f"A3B row-owned router validation failed: {exc}")

    _STATS.update(
        {
            "installation_status": "awaiting_selfcheck",
            "installation_error": None,
            "target_routers": 40,
            "mtp_routers": 1,
            "validated_contract": _a3b_router_contract(combine_tail=combine_tail),
        }
    )
    return QwenRowOwnedRouterInstallPlan(
        target_blocks=tuple(target_blocks),
        mtp_blocks=tuple(mtp_blocks),
        combine_tail=combine_tail,
    )


def install_qwen_row_owned_routers(
    plan: QwenRowOwnedRouterInstallPlan,
    selfcheck_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Atomically install all validated blocks after the exact self-check."""
    lanes = {} if selfcheck_report is None else selfcheck_report.get("lanes", {})
    if lanes.get("qwen_row_owned_router") != "ok":
        _fail_router_configuration(
            "A3B row-owned router selfcheck did not validate the exact M1-M16 route"
        )
    if plan.combine_tail and lanes.get("qwen_combine_tail_m1_m2") != "ok":
        _fail_router_configuration(
            "A3B combine tail selfcheck did not validate fixed M1/M2 arithmetic"
        )
    installed: list[tuple[Any, type]] = []
    try:
        for block in (*plan.target_blocks, *plan.mtp_blocks):
            original_class = type(block)
            block.__class__ = _installed_class(
                original_class, combine_tail=plan.combine_tail
            )
            installed.append((block, original_class))
    except Exception:
        for block, original_class in reversed(installed):
            block.__class__ = original_class
        raise
    _INSTALLED_ROUTERS.extend(installed)
    _STATS["installed"] = True
    _STATS["installation_status"] = "installed"
    return qwen_row_owned_router_stats()


def qwen_row_owned_router_stats() -> dict[str, Any]:
    report = dict(_STATS)
    contract = report.get("validated_contract")
    report["validated_contract"] = dict(contract) if isinstance(contract, dict) else None
    return report


def _reset_qwen_row_owned_router_for_tests() -> None:
    global _COMBINE_KERNEL, _KERNEL
    _clear_installed_routers()
    _KERNEL = None
    _COMBINE_KERNEL = None
    _STATS.update(
        {
            "enabled": False,
            "installed": False,
            "installation_status": "disabled",
            "installation_error": None,
            "target_routers": 0,
            "mtp_routers": 0,
            "validated_contract": None,
        }
    )
