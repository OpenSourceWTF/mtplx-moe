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

from .models.qwen4_omlx import (
    Attention,
    GatedDeltaNet,
    _rope_partial,
    scaled_dot_product_attention,
)
from .proj_fusion import _make_quantized_linear


_GDN_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_ATTN_NAMES = ("q_proj", "k_proj", "v_proj")


def _validated_quantized_members(members: tuple[Any, ...]):
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


def _pack_bindings(
    fused_owner: Any,
    bindings: tuple[tuple[Any, str], ...],
    fused_attr: str,
) -> list[int]:
    members, (group_size, bits, mode), has_biases = _validated_quantized_members(
        tuple(getattr(owner, name, None) for owner, name in bindings)
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
    setattr(fused_owner, fused_attr, fused)

    split_points: list[int] = []
    weight_at = 0
    scale_at = 0
    for index, ((owner, name), member) in enumerate(zip(bindings, members)):
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


def _pack_group(owner: Any, names: tuple[str, ...], fused_attr: str) -> list[int]:
    return _pack_bindings(
        owner,
        tuple((owner, name) for name in names),
        fused_attr,
    )


def _split_contiguous(out: mx.array, split_points: list[int]):
    return tuple(mx.contiguous(part) for part in mx.split(out, split_points, axis=-1))


def _split_views(out: mx.array, split_points: list[int]):
    return tuple(mx.split(out, split_points, axis=-1))


class FusedProjectionGatedDeltaNet(GatedDeltaNet):
    def _project_inputs(self, x):
        if math.prod(x.shape[:-1]) > self._mtplx_fused_max_rows:
            return GatedDeltaNet._project_inputs(self, x)
        return _split_views(
            self._mtplx_fused_in_proj(x), self._mtplx_fused_in_proj_splits
        )


class FusedProjectionAttention(Attention):
    def _project_qkv(self, x):
        if math.prod(x.shape[:-1]) > self._mtplx_fused_max_rows:
            return Attention._project_qkv(self, x)
        return self._project_indexer_qkv(x)[1:]

    def _project_indexer_qkv(self, x):
        if math.prod(x.shape[:-1]) > self._mtplx_fused_max_rows:
            return (self.indexer.index_qk_proj(x), *Attention._project_qkv(self, x))
        return _split_contiguous(
            self._mtplx_fused_indexer_qkv_proj(x),
            self._mtplx_fused_indexer_qkv_splits,
        )

    def __call__(self, x, rope, mask, cache, idx_cache):
        if math.prod(x.shape[:-1]) > self._mtplx_fused_max_rows:
            return Attention.__call__(self, x, rope, mask, cache, idx_cache)

        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0
        index_qk, q_proj, k_proj, v_proj = self._project_indexer_qkv(x)
        sparse = self.indexer.select_projected(index_qk, rope, idx_cache, offset)

        q, gate = mx.split(q_proj.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k_proj.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = v_proj.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = rope(mx.arange(offset, offset + S)[None])
        cos, sin = cos[:, None], sin[:, None]
        q, k = _rope_partial(q, cos, sin), _rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if sparse is not None:
            neg = mx.finfo(q.dtype).min if hasattr(mx, "finfo") else -1e9
            add = mx.where(sparse, mx.array(0, q.dtype), mx.array(neg, q.dtype))
            mask = (
                add
                if mask is None
                else (mask + add if not isinstance(mask, str) else add)
            )

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


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
            bindings = (
                (module.indexer, "index_qk_proj"),
                *((module, name) for name in _ATTN_NAMES),
            )
            splits = _pack_bindings(
                module,
                bindings,
                "_mtplx_fused_indexer_qkv_proj",
            )
            module._mtplx_fused_indexer_qkv_splits = splits
            module._mtplx_fused_max_rows = int(max_rows)
            module.__class__ = FusedProjectionAttention
            report["attn"] += 1
    if report["gdn"] or report["attn"]:
        mx.clear_cache()
    return report
