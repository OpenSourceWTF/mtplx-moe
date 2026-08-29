# SPDX-License-Identifier: MIT
# Vendored from ml-explore/mlx-lm PR #1788; native MTP ownership follows
# jundot/omlx PRs #3161 and #3163. MTPLX adapts only the n-gram lookup seam.
# MLX port of Qwen3.8-Flash-Next (HF model_type: qwen4_exp)
# New compared to qwen3_next: QSA sparse attention, gated residual
# (hyper-connections), sharded n-gram / PLE embedding, split deltanet projections.

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..attention_context import current_attention_phase
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, _BaseCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # n-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 1234
    eos_token_id: Any = 248044
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.text.partial_rotary_factor)
        )
        if not self.text.layer_types:
            n, k = self.text.num_hidden_layers, self.text.full_attention_interval
            self.text.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(n)
            ]


# --------------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """RMSNorm, normalized per group when group_size is given.

    Hyper-connections normalize each of the hc_count streams separately, hence the
    reshape: one weight of size hc_count*hidden, but one statistic per stream.
    """

    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} non divisible par group_size {group_size}")

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * self.weight


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        g = act(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


# ------------------------------------------------------------------- rope / helpers


def _rope_partial(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rope to the first `rotary_dim` dimensions only."""
    d = cos.shape[-1]
    # cos/sin are computed in float32: without this cast they promote x and the
    # whole attention falls back to float32.
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    xr, xp = x[..., :d], x[..., d:]
    half = d // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


def _positions_from_offset(offset: int | mx.array, rows: int) -> mx.array:
    """Build positions without converting an array offset to a host scalar."""

    return mx.arange(int(rows), dtype=mx.int32) + offset


class RotaryEmbedding:
    def __init__(self, dim: int, base: float):
        self.dim = dim
        self.inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)

    def __call__(self, positions: mx.array):
        # positions: (B, T) -> cos/sin (B, T, dim)
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


# ------------------------------------------------------------------------ QSA


QSA_COMPACT_MAX_ROWS = 4
_QSA_COMPACT_PHASES = frozenset(("decode_verify", "ar_decode"))


def _qsa_compact_runtime_enabled(x: mx.array, cache: Any | None) -> bool:
    """Enable compact rows only for the installed incremental phases."""

    return (
        cache is not None
        and current_attention_phase() in _QSA_COMPACT_PHASES
        and int(x.shape[-2]) <= QSA_COMPACT_MAX_ROWS
    )


@dataclass(frozen=True)
class QSACompactSelection:
    """Fixed-width per-query token rows selected by the QSA indexer."""

    indices: mx.array
    valid: mx.array


@dataclass(frozen=True)
class _QSASelectedBlocks:
    kv_len: int
    n_blocks: int
    top: mx.array
    top_visible: mx.array
    q_pos: mx.array


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks.

    The reference PyTorch implementation loops over (batch, query); here everything
    is vectorized: pooled keys do not depend on the query, so they are computed once
    and followed by a per-row top-k.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x, rope, cache, offset: int) -> Optional[mx.array]:
        qk = self.index_qk_proj(x)
        return self.select_projected(qk, rope, cache, offset)

    def _select_blocks(
        self, qk, rope, cache, offset: int
    ) -> Optional[_QSASelectedBlocks]:
        B, S, _ = qk.shape
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            raw_k = cache.update(raw_k)
            if getattr(cache, "fixed_capacity", False):
                return self._select_blocks_fixed_capacity(
                    q, raw_k, rope, cache, offset
                )
        kv_len = raw_k.shape[1]

        # No sparsification possible: every visible token fits in the budget, so the
        # top-k would keep them all. The usual causal mask is enough.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        block_starts = mx.arange(n_blocks) * self.compress_ratio
        if cache is None:
            pooled = raw_k[:, : n_blocks * self.compress_ratio].reshape(
                B, n_blocks, self.compress_ratio, self.head_dim
            )
            pooled = self.k_layernorm(
                pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
            )
            cos_k, sin_k = rope(block_starts[None, :])
            pooled = _rope_partial(pooled, cos_k, sin_k)
        else:
            pooled_at = cache.pooled_offset
            if pooled_at < n_blocks:
                suffix = raw_k[
                    :,
                    pooled_at * self.compress_ratio : n_blocks * self.compress_ratio,
                ].reshape(B, n_blocks - pooled_at, self.compress_ratio, self.head_dim)
                suffix = self.k_layernorm(
                    suffix.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
                )
                suffix_starts = mx.arange(pooled_at, n_blocks) * self.compress_ratio
                cos_k, sin_k = rope(suffix_starts[None, :])
                suffix = _rope_partial(suffix, cos_k, sin_k)
                pooled = cache.append_pooled(suffix)
            else:
                pooled = cache.pooled_keys

        q_pos = _positions_from_offset(offset, S)
        cos_q, sin_q = rope(q_pos[None, :])
        q = self.q_layernorm(q)
        q = _rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])

        # scores: sum over heads of relu(q.k), per block
        scores = mx.einsum(
            "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        # a block is only a candidate if it lies entirely in the query's past
        block_end = block_starts + self.compress_ratio - 1
        visible = block_end[None, None, :] <= q_pos[None, :, None]
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]  # (B, S, k)
        top_visible = mx.take_along_axis(visible, top, axis=-1)
        return _QSASelectedBlocks(
            kv_len=kv_len,
            n_blocks=n_blocks,
            top=top,
            top_visible=top_visible,
            q_pos=q_pos,
        )

    def _select_blocks_fixed_capacity(self, q, raw_k, rope, cache, offset):
        """Select from fixed banks while logical lengths remain tensor state."""

        B, S = int(q.shape[0]), int(q.shape[1])
        ratio = self.compress_ratio
        logical_length = cache.offset
        pooled_at = cache.pooled_offset
        completed_blocks = logical_length // ratio

        # Exact M=2 advances across at most one four-token pool boundary. The
        # candidate block is always computed, but becomes visible and mutates
        # the derived bank only when the tensor logical length completed it.
        raw_block = mx.slice(
            raw_k,
            pooled_at * ratio,
            axes=(1,),
            slice_size=(B, ratio, self.head_dim),
        )
        pooled_new = self.k_layernorm(
            raw_block.astype(mx.float32).mean(axis=1).astype(raw_k.dtype)
        )[:, None, :]
        block_start = (pooled_at * ratio).reshape(1, 1)
        cos_k, sin_k = rope(block_start)
        pooled_new = _rope_partial(pooled_new, cos_k, sin_k)
        pooled_candidate = mx.slice_update(
            cache.pooled_keys, pooled_new, pooled_at, axes=(1,)
        )
        completed_new_block = completed_blocks > pooled_at
        cache.cache[2] = mx.where(
            completed_new_block, pooled_candidate, cache.pooled_keys
        )
        cache.cache[3] = mx.where(
            completed_new_block, pooled_at + 1, pooled_at
        )
        pooled = cache.pooled_keys

        n_blocks = int(pooled.shape[1])
        block_starts = mx.arange(n_blocks) * ratio
        q_pos = offset + mx.arange(S)
        cos_q, sin_q = rope(q_pos[None, :])
        q = self.q_layernorm(q)
        q = _rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])
        scores = mx.einsum(
            "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)
        block_end = block_starts + ratio - 1
        visible = (block_end[None, None, :] <= q_pos[None, :, None]) & (
            block_starts[None, None, :] < logical_length
        )
        scores = mx.where(visible, scores, -mx.inf)
        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
        top_visible = mx.take_along_axis(visible, top, axis=-1)
        return _QSASelectedBlocks(
            kv_len=logical_length,
            n_blocks=n_blocks,
            top=top,
            top_visible=top_visible,
            q_pos=q_pos,
        )

    def select_projected(self, qk, rope, cache, offset: int) -> Optional[mx.array]:
        B, S, _ = qk.shape
        selected = self._select_blocks(qk, rope, cache, offset)
        if selected is None:
            return None

        keep_block = mx.zeros((B, S, selected.n_blocks + 1), dtype=mx.bool_)
        top = mx.where(
            selected.top_visible, selected.top, selected.n_blocks
        )
        keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[
            ..., : selected.n_blocks
        ]

        # Remap selected complete blocks to tokens. For each query, retain its
        # own incomplete visible block exactly; a block that is complete only
        # relative to a later query must not disappear from an earlier query.
        keep = mx.repeat(keep_block, self.compress_ratio, axis=-1)
        if getattr(cache, "fixed_capacity", False):
            mask_capacity = int(cache.attention_capacity)
            keep = keep[..., :mask_capacity]
        else:
            mask_capacity = selected.kv_len
            tail = mask_capacity - selected.n_blocks * self.compress_ratio
            if tail:
                keep = mx.concatenate(
                    [keep, mx.zeros((B, S, tail), dtype=mx.bool_)], axis=-1
                )
        token_positions = mx.arange(mask_capacity)[None, None, :]
        visible_length = selected.q_pos[None, :, None] + 1
        partial_start = (
            visible_length // self.compress_ratio
        ) * self.compress_ratio
        partial = (token_positions >= partial_start) & (
            token_positions < visible_length
        )
        keep = keep | partial
        return keep[:, None]  # (B, 1, S, kv_len)

    def select_projected_compact(
        self, qk, rope, cache, offset: int
    ) -> Optional[QSACompactSelection]:
        """Return compact token rows without materializing a token mask."""

        B, S, _ = qk.shape
        selected = self._select_blocks(qk, rope, cache, offset)
        if selected is None:
            return None

        ratio = self.compress_ratio
        full_width = selected.top.shape[-1] * ratio
        block_offsets = mx.arange(ratio)[None, None, None, :]
        full_positions = (
            selected.top[..., None] * ratio + block_offsets
        ).reshape(B, S, full_width)
        full_valid = mx.broadcast_to(
            selected.top_visible[..., None], (B, S, selected.top.shape[-1], ratio)
        ).reshape(B, S, full_width)

        partial_width = max(ratio - 1, 0)
        if partial_width:
            visible_length = selected.q_pos[None, :, None] + 1
            partial_start = (visible_length // ratio) * ratio
            partial_offsets = mx.arange(partial_width)[None, None, :]
            partial_positions = mx.broadcast_to(
                partial_start + partial_offsets, (B, S, partial_width)
            )
            partial_valid = (partial_positions < visible_length) & (
                partial_positions < selected.kv_len
            )
        else:
            partial_positions = mx.zeros((B, S, 0), dtype=mx.int32)
            partial_valid = mx.zeros((B, S, 0), dtype=mx.bool_)

        positions = mx.concatenate(
            [full_positions, partial_positions], axis=-1
        )
        valid = mx.concatenate([full_valid, partial_valid], axis=-1)
        indices = mx.where(valid, positions, mx.array(0, positions.dtype))
        return QSACompactSelection(indices=indices, valid=valid)


def _qsa_compact_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    selection: QSACompactSelection,
    *,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """Run SDPA independently for each query over its selected K/V rows."""

    B, n_heads, S, head_dim = q.shape
    n_kv_heads, kv_len = k.shape[1], k.shape[2]
    width = selection.indices.shape[-1]
    rows = B * S

    q_rows = q.transpose(0, 2, 1, 3).reshape(rows, n_heads, 1, head_dim)
    indices = selection.indices.reshape(rows, width)
    # Gather directly into the row-major layout consumed by SDPA.  Placing the
    # query-row axis before the KV-head axis avoids a transpose followed by a
    # multi-megabyte contiguous copy for every K and V bank.
    indices_batched = selection.indices[:, :, None, :, None]
    k_rows = mx.take_along_axis(
        k[:, None, :, :, :], indices_batched, axis=3
    ).reshape(rows, n_kv_heads, width, head_dim)
    v_rows = mx.take_along_axis(
        v[:, None, :, :, :], indices_batched, axis=3
    ).reshape(rows, n_kv_heads, width, v.shape[-1])

    neg = mx.finfo(q.dtype).min if hasattr(mx, "finfo") else -1e9
    valid = selection.valid.reshape(rows, width)
    compact_mask = mx.where(
        valid, mx.array(0, q.dtype), mx.array(neg, q.dtype)
    ).reshape(rows, 1, 1, width)

    # QSA already applies causal visibility. Numeric masks (for example padding)
    # still need to be gathered alongside the selected K/V rows; string masks are
    # the standard causal marker and are therefore redundant here.
    if mask is not None and not isinstance(mask, str):
        if mask.ndim == 2:
            mask_rows = mx.broadcast_to(mask[None, :, :], (B, S, kv_len))
            mask_rows = mask_rows.reshape(rows, kv_len)
            mask_rows = mx.take_along_axis(mask_rows, indices, axis=-1)
            mask_rows = mask_rows.reshape(rows, 1, 1, width)
        elif mask.ndim == 3:
            mask_rows = mx.broadcast_to(mask, (B, S, kv_len))
            mask_rows = mask_rows.reshape(rows, kv_len)
            mask_rows = mx.take_along_axis(mask_rows, indices, axis=-1)
            mask_rows = mask_rows.reshape(rows, 1, 1, width)
        else:
            n_mask_heads = mask.shape[1]
            mask_rows = mx.broadcast_to(mask, (B, n_mask_heads, S, kv_len))
            mask_rows = mx.take_along_axis(
                mask_rows, selection.indices[:, None, :, :], axis=-1
            )
            mask_rows = mask_rows.transpose(0, 2, 1, 3).reshape(
                rows, n_mask_heads, width
            )
            mask_rows = mask_rows[:, :, None, :]
        if mask.dtype == mx.bool_:
            compact_mask = mx.where(
                valid[:, None, None, :] & mask_rows,
                mx.array(0, q.dtype),
                mx.array(neg, q.dtype),
            )
        else:
            compact_mask = compact_mask + mask_rows.astype(q.dtype)

    out = scaled_dot_product_attention(
        q_rows,
        k_rows,
        v_rows,
        cache=None,
        scale=scale,
        mask=compact_mask,
    )
    return out.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj also carries the output gate: n_heads * head_dim * 2
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)

    def _project_qkv(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def __call__(self, x, rope, mask, cache, idx_cache) -> mx.array:
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0

        sparse = None
        compact = None
        if _qsa_compact_runtime_enabled(x, cache):
            index_qk = self.indexer.index_qk_proj(x)
            compact = self.indexer.select_projected_compact(
                index_qk, rope, idx_cache, offset
            )
        else:
            sparse = self.indexer(x, rope, idx_cache, offset)

        q_proj, k_proj, v_proj = self._project_qkv(x)
        q, gate = mx.split(q_proj.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k_proj.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = v_proj.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = rope(_positions_from_offset(offset, S)[None])
        cos, sin = cos[:, None], sin[:, None]
        q, k = _rope_partial(q, cos, sin), _rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if compact is not None:
            out = _qsa_compact_attention(
                q, k, v, compact, scale=self.scale, mask=mask
            )
        else:
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


# ------------------------------------------------------------------- gated deltanet


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        # unlike qwen3-next, the projections are split
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def _project_inputs(self, x):
        return (
            self.in_proj_qkv(x),
            self.in_proj_z(x),
            self.in_proj_b(x),
            self.in_proj_a(x),
        )

    def __call__(self, x, mask, cache) -> mx.array:
        B, S, _ = x.shape
        mixed_qkv, z, b, a = self._project_inputs(x)
        z = z.reshape(B, S, self.n_v, self.dv)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


# ------------------------------------------------------------------------- MoE


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        weights = mx.softmax(
            mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True
        )
        routed = (self.switch_mlp(x, idx) * weights[..., None]).sum(axis=-2)
        routed = routed.astype(x.dtype)
        return routed + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ------------------------------------------------------ hyper-connections (residual)


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None
        )

    def __call__(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject


# -------------------------------------------------------------- n-gram / PLE


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class UnboundNGramRows(nn.Module):
    """Parameter-free seam replaced by the exact MTPLX row-cache provider."""

    def __init__(self, layer_index: int, row_width: int):
        super().__init__()
        self.layer_index = int(layer_index)
        self.row_width = int(row_width)

    def __call__(self, _row_ids: mx.array) -> mx.array:
        raise RuntimeError(
            f"streamed n-gram layer {self.layer_index} has no bound row cache"
        )


class NGramEmbedding(nn.Module):
    """N-gram hash table, sharded into `split_ngram_parts` pieces.

    ~51B parameters: a dense lookup is never performed. Indices are sorted by shard
    on the host side, as in the llama.cpp implementation.
    """

    def __init__(
        self,
        args: TextArgs,
        embed_dim: int,
        ple_layer_index: int = 0,
        layer_index: int = 0,
    ):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes

        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        self.n_shards = args.split_ngram_parts
        self.rows_per_shard = math.ceil(padded / self.n_shards)
        self.ngram_embedding = UnboundNGramRows(layer_index, head_dim)

        # buffers taken as-is from the checkpoint
        mults = []
        max_long = (1 << 63) - 1
        half = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + _PRIME_1 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(
                2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
            )
        # Public attributes: only there to absorb the checkpoint tensors. They live
        # in parameters(), so an astype(float16) would destroy them; the values
        # actually used live in the `_`-prefixed copies, outside parameters() and
        # rebuilt identically from the config.
        self.layer_multipliers = mx.array(mults, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self._mults = mx.array(mults, dtype=mx.int64)
        self._sizes = mx.array(sizes, dtype=mx.int64)
        self._offsets = mx.array(offsets, dtype=mx.int64)

    def _shift_right(self, ids: mx.array, shift: int) -> mx.array:
        """Shift right by `shift`, without crossing an EOS boundary."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = mx.arange(T)
        eos_pos = mx.where(ids == self.eos_token_id, pos, -1)
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=prev_incl.dtype), prev_incl[:, :-1]], axis=1
        )
        in_segment = pos[None] - (prev + 1)
        src = pos - shift
        gathered = mx.take_along_axis(
            ids, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1
        )
        ok = (in_segment >= shift) & (src[None] >= 0)
        return mx.where(ok, gathered, self.eos_token_id)

    def __call__(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        n_new = ids.shape[1]
        history = mx.concatenate([prev_context, ids], axis=1).astype(mx.int64)
        shifted = [self._shift_right(history, s) for s in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self._mults[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self._mults[p])
            gid = mixed[..., None] % self._sizes[lo:hi].reshape(1, 1, -1)
            blocks.append(gid + self._offsets[lo:hi].reshape(1, 1, -1))

        gid = mx.concatenate(blocks, axis=-1)[:, -n_new:]
        return self.ngram_embedding(gid).reshape(*gid.shape[:2], -1)


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, ple_layer_index: int, layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(
            args,
            args.ple_embed_dim,
            ple_layer_index,
            layer_index,
        )
        k = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=self.dilation,
            bias=False,
        )

    @staticmethod
    def _apply_mask(hidden: mx.array, mask: mx.array | None) -> mx.array:
        if mask is None:
            return hidden
        if mask.ndim > 2:
            mask = mask.reshape(mask.shape[0], -1)[:, -hidden.shape[1] :]
        return mx.where(mask[..., None], hidden, 0)

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        S = x.shape[1]
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(full[:, -n:, :])
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def __call__(
        self,
        hidden: mx.array,
        ids: mx.array,
        prev_ctx: mx.array,
        cache,
        conv_mask: mx.array | None = None,
        precomputed_embedding: mx.array | None = None,
    ) -> mx.array:
        emb = (
            self.ple_embedding(ids, prev_ctx)
            if precomputed_embedding is None
            else precomputed_embedding
        ).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        normalized = self.norm_conv(gated)
        gated = self._apply_mask(gated, conv_mask)
        normalized = self._apply_mask(normalized, conv_mask)
        return gated + self._short_conv(normalized, cache)


# ------------------------------------------------------------------- decoder / model


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_idx = (
            args.ple_layer_ids.index(layer_idx + 1)
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.ple = PLELayer(args, ple_idx, layer_idx) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if self.ple is not None:
            h = h + self.ple(h, ids, prev_ctx, cache, conv_mask=conv_mask)

        x, hyper, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, rope, mask, cache, idx_cache)
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)

        x, hyper, inject = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # no final `norm` in this model: this mixer carries it
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, args.rope_theta)
        self.ple_layers = [
            i for i in range(args.num_hidden_layers) if (i + 1) in args.ple_layer_ids
        ]
        self.ssm_idx = next(i for i, layer in enumerate(self.layers) if layer.is_linear)
        self.fa_idx = next(i for i, layer in enumerate(self.layers) if not layer.is_linear)

    def __call__(
        self,
        ids: mx.array,
        cache=None,
        input_embeddings=None,
        *,
        return_hidden: bool = False,
    ):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.ssm_idx])

        prev_ctx = None
        if self.ple_layers:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            pc = cache[self.ple_layers[0]]
            prev = pc[3] if pc is not None else None
            prev_ctx = (
                prev
                if prev is not None
                else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            )
            if pc is not None:
                tail = mx.concatenate([prev_ctx, ids], axis=1)[:, -ctx_len:]
                pc[3] = tail

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache):
            idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
            h = layer(h, self.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        output = self.hyper_connection_mixer(h)
        return (output, h) if return_hidden else output


class _IndexerCache(_BaseCache):
    """Own QSA raw keys and the derived completed-block key bank."""

    raw_step = 2048

    def __init__(self, compress_ratio: int = 4):
        self.keys = None
        self.offset = 0
        self.compress_ratio = int(compress_ratio)
        self._pooled_keys = None
        self.pooled_offset = 0

    def update(self, k: mx.array) -> mx.array:
        previous = self.offset
        needed = previous + int(k.shape[1])
        if self.keys is None or needed > int(self.keys.shape[1]):
            current = 0 if self.keys is None else int(self.keys.shape[1])
            capacity = (
                ((needed + self.raw_step - 1) // self.raw_step) + 1
            ) * self.raw_step
            extension = mx.zeros(
                (int(k.shape[0]), capacity - current, int(k.shape[2])),
                dtype=k.dtype,
            )
            self.keys = (
                extension
                if self.keys is None
                else mx.concatenate([self.keys, extension], axis=1)
            )
        self.offset = needed
        self.keys[:, previous:needed] = k
        return self.keys[:, :needed]

    @property
    def pooled_keys(self):
        if self._pooled_keys is None:
            return None
        return self._pooled_keys[:, : self.pooled_offset]

    def append_pooled(self, pooled: mx.array) -> mx.array:
        previous = self.pooled_offset
        needed = previous + int(pooled.shape[1])
        step = self.raw_step // self.compress_ratio
        if self._pooled_keys is None or needed > int(self._pooled_keys.shape[1]):
            current = 0 if self._pooled_keys is None else int(self._pooled_keys.shape[1])
            capacity = ((needed + step - 1) // step + 1) * step
            extension = mx.zeros(
                (int(pooled.shape[0]), capacity - current, int(pooled.shape[2])),
                dtype=pooled.dtype,
            )
            self._pooled_keys = (
                extension
                if self._pooled_keys is None
                else mx.concatenate([self._pooled_keys, extension], axis=1)
            )
        self.pooled_offset = needed
        self._pooled_keys[:, previous:needed] = pooled
        return self.pooled_keys

    def is_trimmable(self):
        return True

    def trim(self, count: int) -> int:
        trimmed = min(self.offset, count)
        self.offset -= trimmed
        self.pooled_offset = min(
            self.pooled_offset, self.offset // self.compress_ratio
        )
        return trimmed

    @property
    def state(self):
        return None if self.keys is None else self.keys[:, : self.offset]

    @state.setter
    def state(self, v):
        self.keys = v
        self.offset = 0 if v is None else int(v.shape[1])
        self._pooled_keys = None
        self.pooled_offset = 0

    @property
    def nbytes(self):
        raw = 0 if self.keys is None else self.keys.nbytes
        pooled = 0 if self._pooled_keys is None else self._pooled_keys.nbytes
        return raw + pooled


class QSAKVCache(KVCache):
    """QSA attention cache whose raw indexer keys share rollback ownership."""

    def __init__(self, compress_ratio: int = 4):
        super().__init__()
        self.indexer = _IndexerCache(compress_ratio)

    def update_indexer(self, raw_keys: mx.array) -> mx.array:
        return self.indexer.update(raw_keys)

    @property
    def indexer_offset(self) -> int:
        return self.indexer.offset

    @indexer_offset.setter
    def indexer_offset(self, value: int) -> None:
        self.indexer.offset = int(value)
        self.indexer.pooled_offset = min(
            self.indexer.pooled_offset,
            self.indexer.offset // self.indexer.compress_ratio,
        )

    def trim(self, count: int) -> int:
        trimmed = super().trim(count)
        self.indexer.trim(trimmed)
        return trimmed

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.indexer.state
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
            self.indexer.state,
        )

    @state.setter
    def state(self, value) -> None:
        if len(value) == 2:
            self.keys, self.values = value
            raw_keys = None
        else:
            self.keys, self.values, raw_keys = value
        self.offset = 0 if self.keys is None else int(self.keys.shape[2])
        compress_ratio = getattr(getattr(self, "indexer", None), "compress_ratio", 4)
        self.indexer = _IndexerCache(compress_ratio)
        self.indexer.state = raw_keys

    @property
    def meta_state(self):
        return str(self.indexer.compress_ratio)

    @meta_state.setter
    def meta_state(self, value) -> None:
        raw_keys = self.indexer.state
        self.indexer = _IndexerCache(int(value))
        self.indexer.state = raw_keys

    @property
    def nbytes(self):
        return super().nbytes + self.indexer.nbytes


def _register_qsa_cache_type() -> None:
    """Make MLX prompt-cache name-based restoration resolve this cache."""

    import mlx_lm.models.cache as cache_module

    registered = getattr(cache_module, "QSAKVCache", None)
    if registered is not None and registered is not QSAKVCache:
        raise RuntimeError("mlx_lm already registered a different QSAKVCache type")
    cache_module.QSAKVCache = QSAKVCache


_register_qsa_cache_type()


class GemmaRMSNorm(nn.Module):
    """MTP pre-fusion RMSNorm whose checkpoint weight is offset from one."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        value = x.astype(mx.float32)
        value = value * mx.rsqrt(
            mx.mean(value * value, axis=-1, keepdims=True) + self.eps
        )
        return (value * (1.0 + self.weight.astype(mx.float32))).astype(dtype)


class Qwen4ExpMTPModule(nn.Module):
    """Native one-layer Qwen4 MTP head from the OMLX implementation."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_width = self.hc_count * self.hidden_size
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            hc_width,
            eps=args.rms_norm_eps,
        )
        self.fc_embedding = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.fc_hidden = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        mtp_args = copy(args)
        mtp_args.num_hidden_layers = 1
        mtp_args.layer_types = ["full_attention"]
        mtp_args.full_attention_interval = 1
        mtp_args.ple_layer_ids = []
        self.layers = [DecoderLayer(mtp_args, 0)]
        self.hyper_connection_mixer = GatedResidual(
            mtp_args,
            use_combine=False,
        )
        rotary_dim = int(mtp_args.head_dim * mtp_args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, mtp_args.rope_theta)

    def fuse_inputs(
        self,
        input_embeddings: mx.array,
        hidden_states: mx.array,
    ) -> mx.array:
        embeddings = self.fc_embedding(self.pre_fc_norm_embedding(input_embeddings))
        hidden = self.pre_fc_norm_hidden(hidden_states).reshape(
            *hidden_states.shape[:-1],
            self.hc_count,
            self.hidden_size,
        )
        hidden = self.fc_hidden(hidden)
        return (embeddings[..., None, :] + hidden).reshape(*hidden_states.shape)

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        embeddings = (
            embed_tokens(next_token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        hidden_states = self.fuse_inputs(embeddings, hidden_states)
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = create_attention_mask(
            hidden_states,
            [cache[0]] if cache and cache[0] is not None else None,
        )
        for layer, layer_cache in zip(self.layers, cache):
            indexer = (
                layer_cache.indexer
                if layer_cache is not None and hasattr(layer_cache, "indexer")
                else None
            )
            hidden_states = layer(
                hidden_states,
                self.rope,
                attention_mask,
                None,
                layer_cache,
                indexer,
                next_token_ids,
                None,
            )
        return self.hyper_connection_mixer(hidden_states), hidden_states


class LanguageModel(nn.Module):
    """Qwen4 text and native-MTP surfaces consumed by the MTPLX runner."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        self.mtp = Qwen4ExpMTPModule(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def _logits(self, hidden: mx.array, logits_keep: int | None = None):
        if logits_keep and hidden.shape[1] > logits_keep:
            hidden = hidden[:, -logits_keep:]
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
        input_embeddings=None,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        **_kwargs,
    ):
        del hidden_variant
        output, hc_hidden = self.model(
            inputs,
            cache,
            input_embeddings,
            return_hidden=True,
        )
        logits = self._logits(output, logits_keep) if emit_logits else None
        return (logits, hc_hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        logits_keep: int = 0,
    ):
        del concat_order, mtp_hidden_variant, position_offset
        output, hc_hidden = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        logits = self._logits(output, logits_keep)
        return (logits, hc_hidden) if return_hidden else logits

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order=None,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        input_embeddings=None,
    ):
        del concat_order, mtp_hidden_variant, position_offset
        _output, hc_hidden = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
            input_embeddings=input_embeddings,
        )
        return hc_hidden

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for t in self.args.text.layer_types:
            if t == "full_attention":
                caches.append(QSAKVCache(self.args.text.indexer_compress_ratio))
            else:
                # 0: deltanet conv, 1: ssm state, 2: PLE conv, 3: n-gram context
                caches.append(ArraysCache(4))
        return caches

    def make_mtp_cache(self):
        return [
            QSAKVCache(self.args.text.indexer_compress_ratio)
            for _ in self.mtp.layers
        ]


class Model(nn.Module):
    speculative_cache_mode = "snapshot_rollback"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args)

    def __call__(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_forward(self, *args, **kwargs):
        return self.language_model.mtp_forward(*args, **kwargs)

    def mtp_update_cache(self, *args, **kwargs):
        return self.language_model.mtp_update_cache(*args, **kwargs)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("vision_tower.") or k.startswith("model.visual."):
                continue
            # The only externalized weights are the published n-gram table
            # shards. Target and MTP MoE tensors remain in the normal resident
            # model and must pass through this loader unchanged.
            if ".ngram_embedding." in k:
                continue
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                if v.shape[1] == 1:
                    v = v.transpose(0, 2, 1)
            out[k] = v
        return out

    @property
    def quant_predicate(self):
        def fn(path, module, _):
            # only the MoE router stays in full precision (norms and conv1d are
            # never quantized anyway)
            return not path.endswith("mlp.gate")

        return fn
