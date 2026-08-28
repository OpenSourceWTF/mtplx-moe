"""Construction-time small-row projection routes for Qwen4 Flash-Next.

Qwen4's 36 Gated DeltaNet blocks issue four affine-Q4 projections from the
same activation, and its 12 full-attention blocks issue three.  At M <= 4 the
MLX qmv arithmetic is row-independent, so one row-concatenated projection is
element-identical and removes 132 projection dispatches per target pass.

The installer validates and packs the exact Qwen4 modules once.  Installed
modules make one runtime decision on the genuinely variable matrix row count:
small-row decode uses the fused payload; wider prefill uses quantized row views
of the same payload and preserves the stock arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .models.qwen4_omlx import Attention, GatedDeltaNet
from .proj_fusion import _make_quantized_linear


_GDN_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_ATTN_NAMES = ("q_proj", "k_proj", "v_proj")


def _validated_quantized_members(owner: Any, names: tuple[str, ...]):
    members = tuple(getattr(owner, name, None) for name in names)
    if any(not isinstance(member, nn.QuantizedLinear) for member in members):
        raise TypeError("Qwen4 projection group is not fully affine-quantized")
    first = members[0]
    signature = (
        int(first.group_size),
        int(first.bits),
        str(getattr(first, "mode", "affine")),
    )
    if signature[2] != "affine":
        raise TypeError("Qwen4 fused projections require affine quantization")
    input_width = int(first.weight.shape[1])
    scale_width = int(first.scales.shape[1])
    has_biases = first.get("biases") is not None
    for member in members:
        current = (
            int(member.group_size),
            int(member.bits),
            str(getattr(member, "mode", "affine")),
        )
        if current != signature:
            raise TypeError("Qwen4 projection quantization layouts differ")
        if "bias" in member:
            raise TypeError("Qwen4 projection unexpectedly has an additive bias")
        if member.weight.ndim != 2 or int(member.weight.shape[1]) != input_width:
            raise TypeError("Qwen4 projection weight layouts differ")
        if int(member.scales.shape[1]) != scale_width:
            raise TypeError("Qwen4 projection scale layouts differ")
        if (member.get("biases") is not None) != has_biases:
            raise TypeError("Qwen4 projection affine-bias layouts differ")
    return members, signature, has_biases


def _pack_group(owner: Any, names: tuple[str, ...], fused_attr: str) -> list[int]:
    members, (group_size, bits, mode), has_biases = _validated_quantized_members(
        owner, names
    )
    weight = mx.concatenate([member.weight for member in members], axis=0)
    scales = mx.concatenate([member.scales for member in members], axis=0)
    biases = (
        mx.concatenate([member.biases for member in members], axis=0)
        if has_biases
        else None
    )
    mx.eval(weight, scales, *(() if biases is None else (biases,)))

    fused = _make_quantized_linear(
        weight,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    setattr(owner, fused_attr, fused)

    split_points: list[int] = []
    weight_at = 0
    scale_at = 0
    for index, (name, member) in enumerate(zip(names, members)):
        weight_rows = int(member.weight.shape[0])
        scale_rows = int(member.scales.shape[0])
        replacement = _make_quantized_linear(
            weight[weight_at : weight_at + weight_rows],
            scales[scale_at : scale_at + scale_rows],
            None if biases is None else biases[scale_at : scale_at + scale_rows],
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        setattr(owner, name, replacement)
        weight_at += weight_rows
        scale_at += scale_rows
        if index + 1 < len(members):
            split_points.append(weight_at)
    return split_points


def _split_contiguous(out: mx.array, split_points: list[int]):
    return tuple(mx.contiguous(part) for part in mx.split(out, split_points, axis=-1))


class FusedProjectionGatedDeltaNet(GatedDeltaNet):
    def _project_inputs(self, x):
        if math.prod(x.shape[:-1]) > self._mtplx_fused_max_rows:
            return GatedDeltaNet._project_inputs(self, x)
        return _split_contiguous(
            self._mtplx_fused_in_proj(x), self._mtplx_fused_in_proj_splits
        )


class FusedProjectionAttention(Attention):
    def _project_qkv(self, x):
        if math.prod(x.shape[:-1]) > self._mtplx_fused_max_rows:
            return Attention._project_qkv(self, x)
        return _split_contiguous(
            self._mtplx_fused_qkv_proj(x), self._mtplx_fused_qkv_splits
        )


def install_qwen4_fused_projection_routes(
    model: Any,
    *,
    groups: set[str],
    max_rows: int = 4,
) -> dict[str, int]:
    """Install exact Qwen4 projection routes or fail before generation."""

    report = {"gdn": 0, "attn": 0, "skipped": 0}
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    for module in modules:
        if "gdn" in groups and type(module) is GatedDeltaNet:
            splits = _pack_group(module, _GDN_NAMES, "_mtplx_fused_in_proj")
            module._mtplx_fused_in_proj_splits = splits
            module._mtplx_fused_max_rows = int(max_rows)
            module.__class__ = FusedProjectionGatedDeltaNet
            report["gdn"] += 1
        elif "attn" in groups and type(module) is Attention:
            splits = _pack_group(module, _ATTN_NAMES, "_mtplx_fused_qkv_proj")
            module._mtplx_fused_qkv_splits = splits
            module._mtplx_fused_max_rows = int(max_rows)
            module.__class__ = FusedProjectionAttention
            report["attn"] += 1
    if report["gdn"] or report["attn"]:
        mx.clear_cache()
    return report
