"""MLX implementation of Tencent Hy3 with parameter-free routed experts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from mtplx.attention_context import current_attention_phase
from mtplx.hy3_router_last_arrival import hy3_router_last_arrival_route
from mtplx.hy3_router_row_owned import (
    hy3_router_row_owned_route,
    prepare_hy3_router_row_owned_weight,
)
from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    hy3_router_fp32_available,
    hy3_router_fp32_exact_route,
    hy3_router_fp32_exact_splitk_route,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_exact_splitk_weight,
    prepare_hy3_router_fp32_exact_weight,
    prepare_hy3_router_fp32_weight,
)

from mtplx.compile_state import compile_trace_active
from mtplx import roofline_profile as _rp
from .expert_mlx import UnboundExpertSwitch, run_switch_with_shared_overlap


def _kv_nbytes(cache) -> int:
    """Real bytes of the KV a decode attention step reads (0 if unavailable)."""
    try:
        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        if keys is not None and values is not None:
            return int(keys.nbytes) + int(values.nbytes)
        state = getattr(cache, "state", None)
        if isinstance(state, (list, tuple)):
            return sum(int(a.nbytes) for a in state if hasattr(a, "nbytes"))
    except Exception:
        pass
    return 0


def _moe_read_nbytes(switch_mlp) -> int:
    """Real bytes a decode token reads from the expert bank: top_k of E experts,
    not the whole resident bank."""
    try:
        bank = getattr(switch_mlp, "_bank", None)
        top_k = int(switch_mlp.runtime.spec.top_k)
        record_bytes = getattr(bank, "record_bytes", None)
        if record_bytes is not None:
            return int(record_bytes) * top_k
        arrays = getattr(bank, "arrays", None)
        if arrays:
            total = sum(int(a.nbytes) for a in arrays.values())
            experts = int(switch_mlp.runtime.spec.expert_count)
            return total * top_k // max(experts, 1)
    except Exception:
        pass
    return 0


FUSE_SHARED_GATE_UP_ENV = "MTPLX_FUSE_HY3_SHARED_GATE_UP_PROJECTIONS"
FUSE_QKV_ENV = "MTPLX_FUSE_HY3_QKV_PROJECTIONS"
SUBMIT_CADENCE_ENV = "MTPLX_HY3_SUBMIT_CADENCE"


def _decode_submit_cadence() -> int:
    """Layers between decode-lane GPU submission checkpoints (0 = off)."""

    raw = os.environ.get(SUBMIT_CADENCE_ENV, "").strip()
    if not raw:
        return 0
    try:
        cadence = int(raw)
    except ValueError:
        return 0
    return cadence if cadence > 0 else 0
_PACKABLE_LINEAR_SUFFIXES = ("weight", "scales", "biases", "bias")


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _projection_pack_plan(
    weights: dict[str, mx.array],
    *,
    target: str,
    sources: tuple[str, ...],
    equal_output_widths: bool = False,
) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """Validate one construction-time linear pack without allocating arrays."""

    target_suffixes = {
        suffix
        for suffix in _PACKABLE_LINEAR_SUFFIXES
        if f"{target}.{suffix}" in weights
    }
    source_suffixes = [
        {
            suffix
            for suffix in _PACKABLE_LINEAR_SUFFIXES
            if f"{source}.{suffix}" in weights
        }
        for source in sources
    ]
    has_sources = any(source_suffixes)
    if target_suffixes and has_sources:
        raise ValueError(f"packed projection target conflicts with sources: {target}")
    if target_suffixes:
        if "weight" not in target_suffixes:
            raise ValueError(f"packed projection target has no weight: {target}")
        return None
    if not has_sources:
        return None
    if any(suffixes != source_suffixes[0] for suffixes in source_suffixes[1:]):
        raise ValueError(f"incomplete packed projection source for {target}")
    suffixes = source_suffixes[0]
    if "weight" not in suffixes:
        raise ValueError(f"incomplete packed projection source for {target}")
    if ("scales" in suffixes) != ("biases" in suffixes):
        raise ValueError(f"incomplete affine quantization source for {target}")
    for suffix in suffixes:
        arrays = [weights[f"{source}.{suffix}"] for source in sources]
        tail_shapes = {tuple(array.shape[1:]) for array in arrays}
        if len(tail_shapes) != 1:
            raise ValueError(f"incompatible {suffix} shapes for {target}")
        if equal_output_widths and len({int(array.shape[0]) for array in arrays}) != 1:
            raise ValueError(f"incompatible output widths for {target}")
    ordered_suffixes = tuple(
        suffix for suffix in _PACKABLE_LINEAR_SUFFIXES if suffix in suffixes
    )
    return target, sources, ordered_suffixes


def _pack_linear_projection(
    weights: dict[str, mx.array],
    plan: tuple[str, tuple[str, ...], tuple[str, ...]],
) -> None:
    """Materialize one packed linear and release each set of source arrays."""

    target, sources, suffixes = plan
    for suffix in suffixes:
        arrays = [weights.pop(f"{source}.{suffix}") for source in sources]
        packed = mx.concatenate(arrays, axis=0)
        mx.eval(packed)
        weights[f"{target}.{suffix}"] = packed
        del arrays


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    first_k_dense_replace: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    head_dim: int = 128
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    qk_norm: bool = True
    route_norm: bool = True
    router_scaling_factor: float = 1.0
    moe_router_enable_expert_bias: bool = True
    enable_moe_fp32_combine: bool = False
    enable_attention_fp32_softmax: bool = False
    enable_lm_head_fp32: bool = False
    tie_word_embeddings: bool = False
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    mlp_layer_types: Optional[List[str]] = None
    num_nextn_predict_layers: int = 0

    def __post_init__(self) -> None:
        if self.rope_parameters:
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if index < self.first_k_dense_replace else "sparse"
                for index in range(self.num_hidden_layers)
            ]
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types must cover every Hy3 layer")
        if self.hidden_act != "silu":
            raise ValueError("Hy3 MLX currently supports only silu/SwiGLU")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        # QKV fusion (issue #51 T1a, flag-gated). One packed projection replaces
        # three: BIT-EXACT because g64 quant groups run along the input axis, so
        # concatenating output rows keeps each row's own scale/bias. Folds the two
        # low-occupancy 1024-wide k/v matvecs into the 10240-wide grid -> fewer
        # launches and more rows in flight at batch=1 (the measured attention tax).
        self._fuse_qkv = _env_enabled(FUSE_QKV_ENV)
        if self._fuse_qkv:
            self._qkv_splits = [q_dim, q_dim + kv_dim]
            self.qkv_proj = nn.Linear(
                args.hidden_size, q_dim + 2 * kv_dim, bias=args.attention_bias
            )
        else:
            self.q_proj = nn.Linear(args.hidden_size, q_dim, bias=args.attention_bias)
            self.k_proj = nn.Linear(args.hidden_size, kv_dim, bias=args.attention_bias)
            self.v_proj = nn.Linear(args.hidden_size, kv_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(q_dim, args.hidden_size, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_parameters,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        if self._fuse_qkv:
            q, k, v = mx.split(self.qkv_proj(x), self._qkv_splits, axis=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = q.reshape(
            batch, length, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = k.reshape(
            batch, length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = v.reshape(
            batch, length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, *, intermediate_size: int | None = None):
        super().__init__()
        width = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, width, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, width, bias=args.mlp_bias)
        self.down_proj = nn.Linear(width, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class FusedSharedMLP(nn.Module):
    """Sparse-layer shared MLP with one packed gate/up projection."""

    def __init__(self, args: ModelArgs, *, intermediate_size: int):
        super().__init__()
        self._split_at = intermediate_size
        self.gate_up_proj = nn.Linear(
            args.hidden_size,
            2 * intermediate_size,
            bias=args.mlp_bias,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            args.hidden_size,
            bias=args.mlp_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), [self._split_at], axis=-1)
        return self.down_proj(swiglu(gate, up))


def _router_storage_module(module: nn.Module) -> nn.Module:
    """Find the stored linear beneath instrumentation or adapter wrappers."""

    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        if isinstance(current, (nn.Linear, nn.QuantizedLinear)):
            return current
        seen.add(id(current))
        base = getattr(current, "base", None)
        if not isinstance(base, nn.Module):
            break
        current = base
    return current


@dataclass(slots=True)
class _RouterKernelState:
    """Private non-parameter state for one load-time router selection."""

    selector: str = "stock"
    prepared_weight: mx.array | None = None
    splitk_m1: bool = False
    sigmoid_mode: str = "precise"


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.route_norm = args.route_norm
        self.router_scaling_factor = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)
        # Leading underscore keeps an optional alternate layout out of MLX's
        # parameter tree. It is a load-time runtime artifact, not a checkpoint
        # parameter, and is accounted for explicitly by configure_kernel().
        self._mtplx_router_kernel_state = _RouterKernelState()

    def configure_kernel(
        self,
        selector: str,
        *,
        available: bool | None = None,
        splitk_m1: bool = False,
        sigmoid_mode: str = "precise",
    ) -> dict[str, int | bool | str]:
        """Select and prepare one router implementation at model load time."""

        if not isinstance(splitk_m1, bool):
            raise TypeError("splitk_m1 must be bool")
        if sigmoid_mode not in ("precise", "fast"):
            raise ValueError("router sigmoid_mode must be 'precise' or 'fast'")
        if sigmoid_mode == "fast" and selector != "mpp-row-owned-fused":
            raise ValueError("fast sigmoid is selectable only on the row-owned router")

        if selector not in {
            "stock",
            "steel-r1-fused-r2",
            "mpp-r1-fused-r2",
            "mpp-fp32-splitk-r1-fused-r2",
            "mpp-r1-last-arrival-fused-r2",
            "mpp-row-owned-fused",
        }:
            raise ValueError(
                "Hy3 router kernel must be 'stock', 'steel-r1-fused-r2', "
                "'mpp-r1-fused-r2', 'mpp-fp32-splitk-r1-fused-r2', "
                "'mpp-r1-last-arrival-fused-r2', or 'mpp-row-owned-fused'"
            )
        if splitk_m1 and selector != "mpp-fp32-splitk-r1-fused-r2":
            raise ValueError("splitk_m1 requires the FP32 split-K router selector")

        storage_gate = _router_storage_module(self.gate)
        source_weight = getattr(storage_gate, "weight", None)
        source_bytes = int(getattr(source_weight, "nbytes", 0))
        if selector == "stock":
            self._mtplx_router_kernel_state = _RouterKernelState()
            return {
                "selector": selector,
                "enabled": False,
                "source_weight_bytes": source_bytes,
                "prepared_weight_bytes": 0,
                "incremental_bytes": 0,
            }

        if self.gate is not storage_gate or not isinstance(storage_gate, nn.Linear):
            raise Hy3RouterFP32Ineligible(
                "optimized Hy3 router requires an unwrapped nn.Linear gate"
            )
        if (
            source_weight is None
            or source_weight.ndim != 2
            or tuple(int(dimension) for dimension in source_weight.shape) != (192, 4096)
            or source_weight.dtype != mx.bfloat16
        ):
            raise Hy3RouterFP32Ineligible(
                "optimized Hy3 router requires a BF16 [192, 4096] gate"
            )
        if self.top_k != 8 or self.num_experts != 192 or not self.route_norm:
            raise Hy3RouterFP32Ineligible(
                "optimized Hy3 router requires the exact top-8 normalized contract"
            )
        supported = (
            hy3_router_fp32_available() if available is None else bool(available)
        )
        if not supported:
            raise Hy3RouterFP32Ineligible(
                "optimized Hy3 router is unavailable on this Metal device"
            )

        if selector == "steel-r1-fused-r2":
            prepared_weight = prepare_hy3_router_fp32_exact_weight(source_weight)
            # Replace the BF16 checkpoint array. This preserves the stock
            # row-major dispatch while avoiding a per-call BF16->FP32 promote.
            storage_gate.weight = prepared_weight
            state_weight = None
        elif selector == "mpp-fp32-splitk-r1-fused-r2":
            prepared_weight = prepare_hy3_router_fp32_exact_splitk_weight(source_weight)
            # Retain the source row-major BF16 gate for target AR at M1.
            # MTP-head M1 and all M2..M8 calls consume this K-major FP32
            # split-K layout so K>=1 draft and verification use one R1 order.
            state_weight = prepared_weight
        elif selector == "mpp-row-owned-fused":
            prepared_weight = prepare_hy3_router_row_owned_weight(source_weight)
            # Tile-major layout: each SIMD group streams one contiguous
            # 16-expert block; stock M1 keeps the source row-major gate.
            state_weight = prepared_weight
        else:
            prepared_weight = prepare_hy3_router_fp32_weight(source_weight)
            # MPP consumes a K-major BF16 layout while stock large-M fallback
            # continues to use the source row-major gate.
            state_weight = prepared_weight

        prepared_bytes = int(prepared_weight.nbytes)
        self._mtplx_router_kernel_state = _RouterKernelState(
            selector=selector,
            prepared_weight=state_weight,
            splitk_m1=splitk_m1,
            sigmoid_mode=sigmoid_mode,
        )
        report: dict[str, int | bool | str] = {
            "selector": selector,
            "enabled": True,
            "source_weight_bytes": source_bytes,
            "prepared_weight_bytes": prepared_bytes,
            "incremental_bytes": (
                prepared_bytes - source_bytes
                if selector == "steel-r1-fused-r2"
                else prepared_bytes
            ),
        }
        if selector == "mpp-fp32-splitk-r1-fused-r2":
            report["m1_policy"] = "splitk" if splitk_m1 else "stock"
            report["m4_grid_k_parts"] = 32
            report["other_grid_k_parts"] = 16
        elif selector == "mpp-r1-last-arrival-fused-r2":
            report["supported_rows"] = 4
            report["dispatch_count"] = 1
            report["sigmoid_mode"] = "precise"
            report["topology"] = "n16-p16-sg4-in-kernel-pad"
            report["threadgroups"] = 48
            report["attention_phase"] = "decode_verify"
        elif selector == "mpp-row-owned-fused":
            report["supported_rows"] = "1-8"
            report["mpp_descriptor_rows"] = 8
            report["dispatch_count"] = 1
            report["sigmoid_mode"] = "precise"
            report["topology"] = "row-owned-g12-p16-tiled-g6"
            report["weight_layout"] = "tile-major-12x4096x16"
            report["threadgroups_per_dispatch"] = "one-per-row"
            report["device_synchronization"] = "none"
            report["sigmoid_mode"] = sigmoid_mode
            report["authority_phases"] = "all"
        return report

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        state = self._mtplx_router_kernel_state
        if state.selector == "mpp-row-owned-fused":
            rows = math.prod(int(dimension) for dimension in x.shape[:-1])
            # M1 stays on the stock path: one row-owned threadgroup cannot
            # saturate DRAM alone, and M1 serves every MTP draft call.
            if 2 <= rows <= 8:
                assert state.prepared_weight is not None
                expert_ids, route_weights = hy3_router_row_owned_route(
                    x.reshape(1, rows, 4096).astype(mx.float32),
                    state.prepared_weight,
                    self.expert_bias,
                    top_k=self.top_k,
                    route_norm=self.route_norm,
                    scaling_factor=self.router_scaling_factor,
                    sigmoid_mode=state.sigmoid_mode,
                )
                output_shape = (*x.shape[:-1], 8)
                return (
                    expert_ids.reshape(output_shape),
                    route_weights.reshape(output_shape),
                )

        storage_gate = _router_storage_module(self.gate)
        rows = math.prod(int(dimension) for dimension in x.shape[:-1])
        last_arrival_eligible = state.selector != ("mpp-r1-last-arrival-fused-r2") or (
            x.ndim == 3
            and tuple(int(dimension) for dimension in x.shape) == (1, 4, 4096)
            and current_attention_phase() == "decode_verify"
        )
        if (
            state.selector != "stock"
            and state.selector != "mpp-row-owned-fused"
            and 1 <= rows <= 8
            and last_arrival_eligible
            and not (
                state.selector == "mpp-fp32-splitk-r1-fused-r2"
                and rows == 1
                and not state.splitk_m1
            )
            and self.gate is storage_gate
            and isinstance(storage_gate, nn.Linear)
        ):
            value = x.astype(mx.float32)
            if state.selector == "steel-r1-fused-r2":
                return hy3_router_fp32_exact_route(
                    value,
                    storage_gate.weight,
                    self.expert_bias,
                    top_k=self.top_k,
                    route_norm=self.route_norm,
                    scaling_factor=self.router_scaling_factor,
                    finalizer_mode="simd",
                )
            assert state.prepared_weight is not None
            if state.selector == "mpp-fp32-splitk-r1-fused-r2":
                return hy3_router_fp32_exact_splitk_route(
                    value,
                    state.prepared_weight,
                    self.expert_bias,
                    n_tile=32,
                    grid_k_parts=32 if rows == 4 else 16,
                    operand_mode="direct",
                    top_k=self.top_k,
                    route_norm=self.route_norm,
                    scaling_factor=self.router_scaling_factor,
                    finalizer_mode="simd",
                    sigmoid_mode="precise",
                )
            if state.selector == "mpp-r1-last-arrival-fused-r2":
                output = hy3_router_last_arrival_route(
                    value,
                    state.prepared_weight,
                    self.expert_bias,
                    top_k=self.top_k,
                    route_norm=self.route_norm,
                    scaling_factor=self.router_scaling_factor,
                    sigmoid_mode="precise",
                )
                return output.expert_ids, output.route_weights
            return hy3_router_fp32_route(
                value,
                state.prepared_weight,
                self.expert_bias,
                n_tile=16,
                grid_k_parts=8,
                operand_mode="direct",
                top_k=self.top_k,
                route_norm=self.route_norm,
                scaling_factor=self.router_scaling_factor,
                finalizer_mode="simd",
                sigmoid_mode="precise",
            )
        if isinstance(storage_gate, nn.QuantizedLinear):
            # Preserve the pinned community-Q4 affine-Q8 execution contract.
            # Its packed approximation is a separate model from Tencent's
            # source-BF16 router and was validated in the activation dtype.
            gate_input = x
        else:
            # Official Hy3 stores router weights in BF16 but promotes both
            # operands before the reduction so its 4,096-term logits and
            # discrete top-k choice are computed in FP32.
            gate_input = x.astype(mx.float32)
        # Always dispatch through the installed module so activation-stat and
        # LoRA wrappers retain their behavior.
        logits = self.gate(gate_input).astype(mx.float32)
        scores = mx.sigmoid(logits)
        selection_scores = scores + self.expert_bias.astype(mx.float32)
        top_k = self.top_k
        indices = mx.argpartition(selection_scores, kth=-top_k, axis=-1)[..., -top_k:]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.route_norm:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return indices, weights * self.router_scaling_factor


def configure_hy3_router_kernels(
    root: nn.Module,
    selector: str,
    *,
    available: bool | None = None,
    sigmoid_mode: str = "precise",
) -> dict[str, int | str]:
    """Configure every Hy3 router and return explicit model memory accounting."""

    reports = []
    for name, module in root.named_modules():
        if not isinstance(module, Router):
            continue
        splitk_m1 = selector == "mpp-fp32-splitk-r1-fused-r2" and "mtp" in name.split(
            "."
        )
        reports.append(
            module.configure_kernel(
                selector,
                available=available,
                splitk_m1=splitk_m1,
                sigmoid_mode=sigmoid_mode,
            )
        )
    summary: dict[str, int | str] = {
        "selector": selector,
        "router_count": len(reports),
        "enabled_count": sum(bool(report["enabled"]) for report in reports),
        "source_weight_bytes": sum(
            int(report["source_weight_bytes"]) for report in reports
        ),
        "prepared_weight_bytes": sum(
            int(report["prepared_weight_bytes"]) for report in reports
        ),
        "incremental_bytes": sum(
            int(report["incremental_bytes"]) for report in reports
        ),
    }
    if selector == "mpp-fp32-splitk-r1-fused-r2":
        summary["m1_splitk_count"] = sum(
            report.get("m1_policy") == "splitk" for report in reports
        )
        summary["m4_grid_k_parts"] = 32
        summary["other_grid_k_parts"] = 16
    elif selector == "mpp-r1-last-arrival-fused-r2":
        summary["supported_rows"] = 4
        summary["dispatch_count"] = 1
        summary["sigmoid_mode"] = "precise"
        summary["topology"] = "n16-p16-sg4-in-kernel-pad"
        summary["threadgroups"] = 48
        summary["attention_phase"] = "decode_verify"
    elif selector == "mpp-row-owned-fused":
        summary["sigmoid_mode"] = sigmoid_mode
        summary["weight_layout"] = "tile-major-12x4096x16"
        summary["supported_rows"] = "1-8"
        summary["dispatch_count"] = 1
        summary["sigmoid_mode"] = "precise"
        summary["topology"] = "row-owned-g12-p16-precise-g6"
        summary["device_synchronization"] = "none"
        summary["authority_phases"] = "all"
    return summary


def estimate_hy3_router_kernel_incremental_bytes(
    config: dict[str, Any],
    selector: str,
    *,
    include_mtp: bool,
) -> int:
    """Estimate prepared-layout bytes before expert-cache admission."""

    if selector not in {
        "stock",
        "steel-r1-fused-r2",
        "mpp-r1-fused-r2",
        "mpp-fp32-splitk-r1-fused-r2",
        "mpp-r1-last-arrival-fused-r2",
        "mpp-row-owned-fused",
    }:
        raise ValueError(
            "Hy3 router kernel must be 'stock', 'steel-r1-fused-r2', "
            "'mpp-r1-fused-r2', 'mpp-fp32-splitk-r1-fused-r2', "
            "'mpp-r1-last-arrival-fused-r2', or 'mpp-row-owned-fused'"
        )
    if selector == "stock":
        return 0
    if str(config.get("model_type") or "") != "hy_v3":
        raise Hy3RouterFP32Ineligible(
            "optimized Hy3 router requires model_type='hy_v3'"
        )
    try:
        layer_count = int(config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Hy3RouterFP32Ineligible(
            "optimized Hy3 router requires num_hidden_layers"
        ) from exc
    explicit_types = config.get("mlp_layer_types")
    if explicit_types is None:
        try:
            dense_prefix = int(config.get("first_k_dense_replace", 0))
        except (TypeError, ValueError) as exc:
            raise Hy3RouterFP32Ineligible(
                "optimized Hy3 router requires a valid dense-layer prefix"
            ) from exc
        sparse_count = max(0, layer_count - dense_prefix)
    else:
        if (
            not isinstance(explicit_types, (list, tuple))
            or len(explicit_types) != layer_count
        ):
            raise Hy3RouterFP32Ineligible(
                "optimized Hy3 router requires one MLP type per target layer"
            )
        sparse_count = sum(str(layer_type) == "sparse" for layer_type in explicit_types)
    router_count = sparse_count + (1 if include_mtp else 0)
    prepared_element_bytes = 4 if selector == "mpp-fp32-splitk-r1-fused-r2" else 2
    return router_count * 192 * 4096 * prepared_element_bytes


_LOOKAHEAD_ROWS: list[tuple[int, Any, Any]] | None = None


def _lookahead_capture(layer_index: int, x: mx.array, indices: mx.array) -> None:
    """Diagnostic lane: capture decode hiddens + routed ids per sparse layer.

    MTPLX_LOOKAHEAD_CAPTURE_PATH enables it; MTPLX_LOOKAHEAD_CAPTURE_TOKENS
    bounds the row count (default 300 tokens x layers). Rows persist to an
    .npz at exit. Decode rows only (single-token calls).
    """

    global _LOOKAHEAD_ROWS
    path = os.environ.get("MTPLX_LOOKAHEAD_CAPTURE_PATH")
    if not path or int(x.shape[-2]) != 1:
        return
    limit = int(os.environ.get("MTPLX_LOOKAHEAD_CAPTURE_TOKENS", "300"))
    if _LOOKAHEAD_ROWS is None:
        _LOOKAHEAD_ROWS = []

        def _save() -> None:
            import numpy as _np

            rows = _LOOKAHEAD_ROWS or []
            if not rows:
                return
            mx.eval(*[r[1] for r in rows], *[r[2] for r in rows])
            _np.savez_compressed(
                path,
                layers=_np.array([r[0] for r in rows], dtype=_np.int16),
                hiddens=_np.stack(
                    [_np.array(r[1], copy=False) for r in rows]
                ).astype(_np.float16),
                expert_ids=_np.stack(
                    [_np.array(r[2], copy=False) for r in rows]
                ).astype(_np.int16),
            )

        import atexit

        atexit.register(_save)
    # 79 sparse layers per token; bound total rows by tokens * layers.
    if len(_LOOKAHEAD_ROWS) >= limit * 79:
        return
    _LOOKAHEAD_ROWS.append(
        (
            int(layer_index),
            x.reshape(-1).astype(mx.float32),
            indices.reshape(-1).astype(mx.int32),
        )
    )


class _LookaheadRouters:
    """Plain holder for sibling router references; deliberately NOT a Module.

    ``mlx.nn.Module.__setattr__`` registers array/dict/list/tuple values as
    children, so assigning the sibling routers directly would re-register
    their parameters under a second path in the tree. A plain object rides
    outside the parameter tree entirely.
    """

    __slots__ = ("entries",)

    def __init__(self, entries: tuple) -> None:
        # tuple of (layer_index, Router) for the next up-to-3 sparse layers.
        self.entries = entries


class SparseMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_index: int,
        *,
        fuse_shared_gate_up: bool = False,
    ):
        super().__init__()
        self.router = Router(args)
        self.layer_index = int(layer_index)
        self.switch_mlp = UnboundExpertSwitch(layer_index)
        shared_width = args.moe_intermediate_size * args.num_shared_experts
        shared_cls = FusedSharedMLP if fuse_shared_gate_up else MLP
        self.shared_mlp = shared_cls(
            args,
            intermediate_size=shared_width,
        )
        self.enable_moe_fp32_combine = args.enable_moe_fp32_combine

    def _maybe_prefetch_lookahead(self, x: mx.array, indices: mx.array) -> None:
        """Residual-stream lookahead prefetch (issue #51 C6).

        Applies the next up-to-3 sparse layers' routers to this layer's
        hidden state (measured 74.3%/66%/61% top-k overlap at L=1/2/3
        against a 4.2% identity baseline) and hands the predicted expert
        ids to the runtime's speculative ring. Decode single-token calls
        only; free unless an attached runtime enables prefetch.
        """

        lookahead = getattr(self, "_mtplx_next_routers", None)
        if lookahead is None or not lookahead.entries:
            return
        runtime = getattr(self.switch_mlp, "runtime", None)
        if runtime is None:
            return
        config = getattr(runtime, "config", None)
        if config is None or getattr(config, "prefetch_slots", 0) <= 0:
            return
        prefetch = getattr(runtime, "prefetch_experts", None)
        if prefetch is None or int(x.shape[-2]) != 1:
            return
        # Fire only from STREAMED source layers, for STREAMED targets.
        # Island layers have no per-layer host sync of their own, so
        # firing there added a brand-new sync per island per token —
        # measured d0 12.5 -> 9.2 at the 90 GiB config. The selection is
        # config-dependent; cache it per layer as plain ints (a tuple of
        # modules would re-register parameters under Module.__setattr__).
        selection = getattr(self, "_mtplx_lookahead_selection", None)
        if selection is None:
            islands = getattr(runtime, "island_layer_set", frozenset())
            if self.layer_index in islands:
                selection = ()
            else:
                selection = tuple(
                    position
                    for position, (next_layer, _router) in enumerate(
                        lookahead.entries
                    )
                    if next_layer not in islands
                )[:3]
            self._mtplx_lookahead_selection = selection
        if not selection:
            return
        # Admission pressure: skip the router compute entirely rather
        # than predict into a saturated lane.
        if getattr(runtime, "speculation_saturated", False):
            return
        predictions = [
            (lookahead.entries[position][0], lookahead.entries[position][1](x)[0])
            for position in selection
        ]
        # The predictions are not ancestors of this layer's output, so they
        # would not ride along with the switch's own eval; materialize them
        # together with the layer's own indices in one host sync (the
        # streamed switch would sync on these indices anyway).
        mx.eval(indices, *(predicted for _layer, predicted in predictions))
        # Nearest lookahead first: measured route overlap decays with
        # depth (74.3% at L=1 vs 61% at L=3), so when the lane saturates
        # mid-set the farther, lower-value layers are the ones dropped.
        for next_layer, predicted in predictions:
            prefetch(
                next_layer,
                [int(value) for value in predicted.reshape(-1).tolist()],
            )
            if getattr(runtime, "speculation_saturated", False):
                break

    def __call__(self, x: mx.array) -> mx.array:
        if _rp.enabled() and int(x.shape[-2]) == 1:
            return self._profiled_call(x)
        indices, scores = self.router(x)
        _lookahead_capture(self.layer_index, x, indices)
        self._maybe_prefetch_lookahead(x, indices)
        # Match the pinned Hy3 MLX reference: when FP32 MoE combining is
        # disabled, both the routing multiply and its reduction happen in the
        # activation dtype.  Keeping ``scores`` in FP32 here subtly changes
        # target logits even though the selected experts are identical.
        if not self.enable_moe_fp32_combine:
            scores = scores.astype(x.dtype)
            # Issue #65: dense-bank switches expose a fused expert wave that
            # owns the routing multiply and reduction (bitwise-identical BF16
            # combine). It declines ineligible shapes by returning None, and
            # the classic path below remains the only other execution.
            wave_call = getattr(self.switch_mlp, "wave_call", None)
            if wave_call is not None:
                combined = wave_call(x, indices, scores)
                if combined is not None:
                    return combined.astype(x.dtype) + self.shared_mlp(x)
        routed, shared = run_switch_with_shared_overlap(
            self.switch_mlp,
            x,
            indices,
            lambda: self.shared_mlp(x),
        )
        routed = (routed * scores[..., None]).sum(axis=-2)
        if self.enable_moe_fp32_combine:
            return (routed.astype(mx.float32) + shared.astype(mx.float32)).astype(
                x.dtype
            )
        return routed.astype(x.dtype) + shared

    def _profiled_call(self, x: mx.array) -> mx.array:
        """Same math as __call__, but captures real (module, input) samples for
        router / routed experts / shared expert so the profiler can bench each on
        the queued lane at exit (MTPLX_ROOFLINE_PROFILE). Bypasses the wave and the
        shared-overlap wrapper so the three are separable; for a single decode
        token the wave declines anyway, so the arithmetic matches __call__."""
        _rp.capture_dense("router", self.router, x, _rp.module_nbytes(self.router))
        indices, scores = self.router(x)
        _rp.capture_moe(
            self.switch_mlp, x, indices, _moe_read_nbytes(self.switch_mlp)
        )
        # T0a: 32-assignment wave reads 4x the experts (32 distinct)
        _rp.capture_moe_wave(
            self.switch_mlp, x, _moe_read_nbytes(self.switch_mlp) * 4
        )
        _rp.capture_dense(
            "shared_expert", self.shared_mlp, x, _rp.module_nbytes(self.shared_mlp)
        )
        routed = self.switch_mlp(x, indices)
        shared = self.shared_mlp(x)
        if not self.enable_moe_fp32_combine:
            scores = scores.astype(x.dtype)
        routed = (routed * scores[..., None]).sum(axis=-2)
        if self.enable_moe_fp32_combine:
            return (routed.astype(mx.float32) + shared.astype(mx.float32)).astype(
                x.dtype
            )
        return routed.astype(x.dtype) + shared


class DecoderLayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_index: int,
        *,
        mlp_type: str | None = None,
        fuse_shared_gate_up: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(args)
        resolved_mlp_type = mlp_type or args.mlp_layer_types[layer_index]
        self.mlp = (
            SparseMLP(
                args,
                layer_index,
                fuse_shared_gate_up=fuse_shared_gate_up,
            )
            if resolved_mlp_type == "sparse"
            else MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if _rp.enabled() and int(x.shape[-2]) == 1:
            normed = self.input_layernorm(x)
            # T0b: prove the norm weights are bf16 (fp32 -> fp32 keys -> KV trap)
            for _nm in ("q_norm", "k_norm"):
                _w = getattr(getattr(self.self_attn, _nm, None), "weight", None)
                if _w is not None:
                    _rp.note_dtype(f"self_attn.{_nm}.weight", _w)
            _rp.note_dtype("kv_cache.keys", getattr(cache, "keys", None))
            _rp.capture_attention(
                self.self_attn, normed, mask, cache,
                _rp.module_nbytes(self.self_attn) + _kv_nbytes(cache),
            )
            hidden = x + self.self_attn(normed, mask, cache)
            return hidden + self.mlp(self.post_attention_layernorm(hidden))
        hidden = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Hy3MTPLayer(nn.Module):
    """Hy3 layer-80 NextN head with a fully resident routed-expert bank.

    Standard NextN structure: the shifted token's embedding and the trunk's
    hidden state are normalized (enorm/hnorm), concatenated, and projected by
    ``eh_proj`` into one transformer block whose MoE routes over the head's
    own experts.  Unlike trunk layers 1-79, those experts are ordinary
    resident weights (a stacked ``SwitchGLU``), not streamed slots: every
    draft touches the whole layer-80 bank (~7.5 GB in the default BF16
    build, ~1.94 GiB in Q4 — callers budget accordingly), so streaming
    would only add SSD traffic to the speculative path.  The embedding table
    and ``lm_head`` are shared with the trunk and passed in at call time; the
    head applies its own checkpoint ``final_layernorm`` before the shared
    head.
    """

    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.mtp_block = DecoderLayer(args, layer_index, mlp_type="sparse")
        self.mtp_block.mlp.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.mlp_bias,
        )
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._lm_head_fp32 = bool(args.enable_lm_head_fp32)

    def __call__(
        self,
        input_ids: mx.array,
        previous_hidden_states: mx.array,
        *,
        embed_tokens: Any,
        lm_head: Any,
        cache: Optional[Any] = None,
    ) -> tuple[mx.array, mx.array]:
        inputs_embeds = embed_tokens(input_ids)
        mixed = self.eh_proj(
            mx.concatenate(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                axis=-1,
            )
        )
        mask = create_attention_mask(mixed, cache, return_array=True)
        hidden = self.mtp_block(mixed, mask, cache)
        recurrent_hidden = self.final_layernorm(hidden)
        # Cast logits, not the head input: fp32 x BF16 matmul materializes
        # an fp32 weight copy per call (see Hy3ForCausalLM.__call__).
        logits = lm_head(recurrent_hidden)
        if self._lm_head_fp32:
            logits = logits.astype(mx.float32)
        return logits, recurrent_hidden


class Hy3MTP(nn.Module):
    """Container for Hy3 NextN heads, mirroring other mtplx MTP modules."""

    def __init__(self, args: ModelArgs, num_mtp_layers: int = 1):
        super().__init__()
        if num_mtp_layers < 1:
            raise ValueError("Hy3 MTP requires at least one NextN layer")
        start_layer = args.num_hidden_layers
        self.layers = [
            Hy3MTPLayer(args, start_layer + index) for index in range(num_mtp_layers)
        ]
        self.start_layer = start_layer
        self.num_mtp_layers = num_mtp_layers


class Hy3Model(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        *,
        fuse_shared_gate_up: bool = False,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args,
                layer_index,
                fuse_shared_gate_up=fuse_shared_gate_up,
            )
            for layer_index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Lookahead prefetch wiring: each sparse layer holds plain
        # references to the next up-to-3 sparse layers' routers so decode
        # can predict their expert routes from the current hidden state.
        sparse_mlps = [
            (layer_index, layer.mlp)
            for layer_index, layer in enumerate(self.layers)
            if isinstance(layer.mlp, SparseMLP)
        ]
        # A wide window (8) lets a streamed layer reach past intervening
        # island layers to the next few STREAMED targets; the hook filters
        # and caps at fire time.
        for position, (_layer_index, mlp) in enumerate(sparse_mlps):
            mlp._mtplx_next_routers = _LookaheadRouters(
                tuple(
                    (next_index, next_mlp.router)
                    for next_index, next_mlp in sparse_mlps[
                        position + 1 : position + 9
                    ]
                )
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        return_pre_norm: bool = False,
    ) -> Any:
        hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(hidden, cache[0])
        # Decode-lane submission cadence: without checkpoints the island
        # segments accumulate as unsubmitted lazy graph while Python walks
        # the layers, and the GPU idles until the next streamed seam's eval
        # drains the whole backlog at once. async_eval every N layers keeps
        # the device fed during host graph-build. Scheduling only — kernel
        # math and ordering are unchanged, so outputs are bit-identical.
        cadence = _decode_submit_cadence()
        if compile_trace_active():
            # Inside a compiled forward the per-layer async_eval is both illegal
            # (graph transformation) and moot — the traced graph is a single
            # submission, so there is no host graph-build window left to fill.
            cadence = 0
        async_eval = getattr(mx, "async_eval", None) if cadence else None
        if async_eval is not None and int(hidden.shape[-2]) > 8:
            async_eval = None
        pending_layers = 0
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer(hidden, mask, layer_cache)
            if async_eval is not None:
                pending_layers += 1
                if pending_layers >= cadence:
                    async_eval(hidden)
                    pending_layers = 0
        if return_pre_norm:
            # NextN heads normalize the trunk hidden themselves (hnorm); hand
            # them the raw last-layer output, not the lm_head's normed view.
            return self.norm(hidden), hidden
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self._fuse_shared_gate_up = _env_enabled(FUSE_SHARED_GATE_UP_ENV)
        self._fuse_qkv = _env_enabled(FUSE_QKV_ENV)
        self.model = Hy3Model(
            args,
            fuse_shared_gate_up=self._fuse_shared_gate_up,
        )
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def _logits_head(self):
        """T2a (flag-gated): trunk lm_head quantized to MTPLX_HY3_LM_HEAD_QUANT_BITS
        (q8/q4). The head is bf16 990MB/token — the one saturated resident read left
        unquantized. Quantizing it FREES memory below the plan budget (safe), and the
        q-weight dequants to bf16 -> fp32 accumulate -> fp32 logit cast, so the
        enable_lm_head_fp32 softmax precision is preserved (David's accumulate rule).
        CHANGES the output distribution -> quality-gated, off by default. Built once."""
        bits = int(os.environ.get("MTPLX_HY3_LM_HEAD_QUANT_BITS", "0") or "0")
        if bits <= 0:
            return self.lm_head
        if not int(getattr(self, "_mtplx_lm_head_quant_bits", 0) or 0):
            if isinstance(self.lm_head, nn.QuantizedLinear):
                # Already quantized upstream (e.g. a q4 checkpoint) — nothing to do.
                self._mtplx_lm_head_quant_bits = int(self.lm_head.bits)
                return self.lm_head
            head = nn.QuantizedLinear.from_linear(
                self.lm_head, group_size=64, bits=bits
            )
            mx.eval(head.parameters())
            # REPLACE the module; do NOT cache the quantized head alongside it.
            # Keeping self.lm_head bound holds the 990MB bf16 weight resident, which
            # inverts this lever's premise: instead of freeing ~712MB it ADDS ~278MB.
            # That matters because the 79-island config peaks at 98 of the 100 GiB
            # wired limit, where a retained copy is enough to trip the GPU
            # command-buffer watchdog (kIOGPUCommandBufferCallbackErrorTimeout).
            self.lm_head = head
            self._mtplx_lm_head_quant_bits = bits
            clear = getattr(mx, "clear_cache", None)
            if clear is not None:
                clear()
        return self.lm_head

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        hidden = self.model(inputs, cache)
        # fp32 activations against the head weight would force MLX to materialize
        # an fp32 copy of the full [vocab, hidden] matrix per call (~9.4 ms). The
        # GEMM already accumulates in fp32, so casting the LOGITS keeps the flag's
        # fp32-softmax semantics at the head's output rounding without that copy.
        head = self._logits_head()
        if _rp.enabled() and int(inputs.shape[-1]) == 1:
            _rp.capture_dense("lm_head", head, hidden, _rp.module_nbytes(head))
        logits = head(hidden)
        if self.args.enable_lm_head_fp32:
            logits = logits.astype(mx.float32)
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        result: dict[str, mx.array] = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key:
                continue
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    if int(parts[2]) >= self.args.num_hidden_layers:
                        continue
                except ValueError:
                    pass
            result[key] = value
        plans: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        if self._fuse_shared_gate_up:
            for layer_index, mlp_type in enumerate(self.args.mlp_layer_types):
                if mlp_type != "sparse":
                    continue
                base = f"model.layers.{layer_index}.mlp.shared_mlp"
                plan = _projection_pack_plan(
                    result,
                    target=f"{base}.gate_up_proj",
                    sources=(f"{base}.gate_proj", f"{base}.up_proj"),
                    equal_output_widths=True,
                )
                if plan is not None:
                    plans.append(plan)
        if self._fuse_qkv:
            for layer_index in range(self.args.num_hidden_layers):
                base = f"model.layers.{layer_index}.self_attn"
                plan = _projection_pack_plan(
                    result,
                    target=f"{base}.qkv_proj",
                    sources=(
                        f"{base}.q_proj",
                        f"{base}.k_proj",
                        f"{base}.v_proj",
                    ),
                    equal_output_widths=False,
                )
                if plan is not None:
                    plans.append(plan)
        if plans:
            # The caller replaces its source mapping with this return value.
            # Clearing it first lets each evaluated packed layer release its
            # original arrays instead of retaining a second model-sized copy.
            weights.clear()
            for plan in plans:
                _pack_linear_projection(result, plan)
        return result

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Set by the resident loader when the runtime requests quantized
        # trunk KV. The MTP draft cache stays a stock KVCache — the
        # compiled draft's tensor-offset adapter reads dense KV state, and
        # one layer of BF16 KV is within the plan's runtime reserve.
        mode = getattr(self, "_mtplx_kv_quant", None)
        if mode:
            from mlx_lm.models.cache import QuantizedKVCache

            bits = {"q8": 8, "q4": 4}[mode]
            return [
                QuantizedKVCache(group_size=64, bits=bits)
                for _layer in self.layers
            ]
        return [KVCache() for _layer in self.layers]
