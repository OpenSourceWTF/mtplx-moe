# Copyright © 2026 MTPLX.
#
# Qwen4-Exp (Qwen3.8-Flash-Next) — MTPLX-owned MLX backend.
#
# The pinned mlx-lm has no implementation for model_type "qwen4_exp"
# (Qwen4ExpForConditionalGeneration). This module implements the text trunk
# natively, reusing the pinned mlx-lm building blocks where the architecture
# genuinely overlaps (GatedDeltaNet, the qwen3_next MoE block) and adding the
# four genuinely new pieces:
#
#   * Gated Residual ("hyper-connections"): hc_count widened residual streams
#     with a learned low-rank read mix and per-stream scalar write gates.
#     There are NO input/post-attention layernorms and no final model.norm in
#     this family — the per-block hc_norm and the final hyper_connection_mixer
#     play those roles.
#   * QSA (Qwen Sparse Attention): standard gated GQA whose causal mask is
#     intersected with a per-query token selection produced by a
#     DeepSeek-V3.2-class indexer (relu-scored mean-pooled key blocks,
#     top-(budget/ratio) blocks + the incomplete tail block).
#   * PLE (Per-Layer Embedding): a hashed n-gram lookup memory (~51B params,
#     320M rows x 160) injected on one early linear-attention layer through a
#     per-stream sigmoid gate and a dilated depthwise convolution. The table
#     is deliberately NEVER materialized: it stays an SSD-resident sidecar
#     (ngram-table.safetensors) gathered row-wise through numpy memmaps, so
#     the OS page cache is the hot-row cache.
#   * mrope carried by the family config; for text-only serving with equal
#     t/h/w positions the interleaved mrope is numerically identical to the
#     standard partial rotary embedding, which is what this module applies
#     (same treatment the pinned mlx-lm gives qwen3_5).
#
# Reference: transformers' modular_qwen4_exp.py (read 2026-08-26, T+9h after
# the weight drop). Norm convention: the Qwen4ExpTextRMSNorm family is stored
# zero-centered ((1+w) convention) in HF checkpoints and shifted by +1.0 in
# sanitize; the GDN gated norm is stored one-centered and is NOT shifted.

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet as _Qwen3_5GatedDeltaNet
from mlx_lm.models.qwen3_next import (
    Qwen3NextSparseMoeBlock as _Qwen3NextSparseMoeBlock,
)


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4

    # GatedDeltaNet (names shared with qwen3_5 so the mlx-lm module reads them)
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"

    # MoE (names shared with qwen3_next's SparseMoeBlock)
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1

    # Gated Residual / hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320

    # QSA indexer
    indexer_n_heads: Optional[int] = 4
    indexer_kv_heads: Optional[int] = 1
    indexer_head_dim: Optional[int] = 128
    indexer_budget: Optional[int] = 2048
    indexer_compress_ratio: Optional[int] = 4

    # PLE / n-gram embedding
    ple_layer_ids: Optional[List[int]] = None  # ONE-indexed, per the HF config
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    eos_token_id: Union[int, List[int], None] = None
    # True in MTPLX packs: the table ships as ngram-table.safetensors and is
    # gathered lazily from SSD — no weight parameter is ever constructed.
    ngram_sidecar: bool = False

    # Rope
    rope_parameters: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0

    def __post_init__(self):
        if self.rope_parameters:
            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # The shipped config says "full_attention"; those layers carry the
        # indexer whenever the QSA fields are set.
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def eos_id(self) -> int:
        eos = self.eos_token_id
        if isinstance(eos, list):
            return int(eos[0])
        return int(eos if eos is not None else 0)


def _rope_cos_sin(positions: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    """Non-interleaved (rotate-half) rope tables for arbitrary integer positions."""
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_partial_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the first `2 * inv_freq.size` features of the last axis of
    x[..., S, H, D] with per-position tables cos/sin of shape [S, rot]."""
    rot = cos.shape[-1]
    x_rope = x[..., :rot]
    x_pass = x[..., rot:]
    half = rot // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    x_rope = (x_rope.astype(mx.float32) * cos + rotated.astype(mx.float32) * sin).astype(
        x.dtype
    )
    return mx.concatenate([x_rope, x_pass], axis=-1)


class GroupedRMSNorm(nn.Module):
    """RMSNorm normalized per contiguous group of `group_size` features, with a
    full-width weight. Used by every hc_norm and the PLE norms (weight arrives
    +1-shifted from sanitize)."""

    def __init__(self, dims: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        if dims % group_size:
            raise ValueError(f"dims ({dims}) not divisible by group_size ({group_size})")
        self.weight = mx.ones((dims,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        grouped = x.reshape(*shape[:-1], -1, self.group_size)
        normed = mx.fast.rms_norm(grouped, None, self.eps)
        return normed.reshape(shape) * self.weight


class SigmoidRMSNormGated(nn.Module):
    """GDN output norm with a sigmoid (not silu) gate — output_gate_type of
    this family. Stored one-centered; NOT +1-shifted in sanitize."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None):
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return x.astype(hidden_states.dtype)
        g = mx.sigmoid(gate.astype(mx.float32))
        return (g * x.astype(mx.float32)).astype(hidden_states.dtype)


class GatedDeltaNet(_Qwen3_5GatedDeltaNet):
    """qwen3_5's GDN with the family's output gate activation (sigmoid) and
    the reference q/k normalization.

    mlx-lm folds the attention scale through mx.fast.rms_norm, whose eps sits
    on mean(x²) — an effective d²·1e-6 on Σx² versus the reference FLA
    l2norm's d·1e-6 (transformers qwen3_5 l2norm: x·rsqrt(Σx²+1e-6)). At
    d=128 that skew is a measured, systematic ~1e-4-class divergence per
    layer (pinned by CPU-exact stage bisection, 2026-08-26), so this forward
    is mlx-lm's verbatim except the two q/k lines reproduce l2norm exactly.
    """

    def __init__(self, args: TextArgs):
        super().__init__(args)
        if getattr(args, "output_gate_type", "sigmoid") == "sigmoid":
            self.norm = SigmoidRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        from mlx_lm.models.gated_delta import gated_delta_update

        B, S, _ = inputs.shape

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5

        def _l2norm(x: mx.array) -> mx.array:
            xf = x.astype(mx.float32)
            return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + 1e-6)).astype(
                x.dtype
            )

        q = inv_scale * _l2norm(q)
        k = _l2norm(k)

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

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class GatedResidual(nn.Module):
    """The Gated Residual read/write mixer (hyper-connections)."""

    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)

    def __call__(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        grouped = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed_input = mx.mean(mix * grouped, axis=-2)
        if "block_inject_weight" not in self:
            return mixed_input
        inject = 2.0 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed_input, hyper_input, inject


class SparseMoeBlock(_Qwen3NextSparseMoeBlock):
    pass


class QSACache:
    """Cache for one QSA layer: the attention KV plus the indexer's raw key
    stream and the incrementally maintained pooled (mean->norm->rope) block
    keys. Append-only, single-sequence."""

    def __init__(self):
        self.kv = KVCache()
        self.raw_keys: Optional[mx.array] = None  # [1, T, index_head_dim]
        self.pooled: Optional[mx.array] = None  # [1, nb, index_head_dim]

    @property
    def offset(self) -> int:
        return self.kv.offset

    def append_raw(self, keys: mx.array) -> mx.array:
        if self.raw_keys is None:
            self.raw_keys = keys
        else:
            self.raw_keys = mx.concatenate([self.raw_keys, keys], axis=1)
        return self.raw_keys

    @property
    def state(self):
        return self.kv.state

    @state.setter
    def state(self, v):
        self.kv.state = v


class QSAIndexer(nn.Module):
    """Vectorized exact port of the reference indexer for the single-sequence
    causal case (B=1, no padding): every query selects its top
    (budget/compress_ratio) complete key blocks by relu-scored pooled keys,
    plus the visible incomplete tail."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.budget = args.indexer_budget
        self.ratio = args.indexer_compress_ratio
        self.block_topk = self.budget // self.ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        rot = args.rotary_dim
        self._inv_freq = args.rope_theta ** (
            -mx.arange(0, rot, 2, dtype=mx.float32) / rot
        )

    def _extend_pooled(self, cache: QSACache) -> Optional[mx.array]:
        raw = cache.raw_keys
        total = raw.shape[1]
        nb_total = total // self.ratio
        nb_old = 0 if cache.pooled is None else cache.pooled.shape[1]
        if nb_total > nb_old:
            fresh = raw[:, nb_old * self.ratio : nb_total * self.ratio, :]
            fresh = fresh.reshape(1, nb_total - nb_old, self.ratio, self.head_dim)
            pooled = mx.mean(fresh.astype(mx.float32), axis=2).astype(raw.dtype)
            pooled = self.k_layernorm(pooled)
            starts = mx.arange(nb_old, nb_total, dtype=mx.int32) * self.ratio
            cos, sin = _rope_cos_sin(starts, self._inv_freq)
            pooled = _apply_partial_rope(pooled[:, :, None, :], cos, sin)[:, :, 0, :]
            cache.pooled = (
                pooled if cache.pooled is None else mx.concatenate([cache.pooled, pooled], axis=1)
            )
        return cache.pooled

    def __call__(self, hidden: mx.array, pos_start: int, cache: QSACache) -> Optional[mx.array]:
        B, S, _ = hidden.shape
        if B != 1:
            raise NotImplementedError("qwen4_exp QSA serves single sequences (B=1)")
        qk = self.index_qk_proj(hidden)
        q, k = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        k = k.reshape(B, S, self.head_dim)
        q = self.q_layernorm(q)
        positions = mx.arange(pos_start, pos_start + S, dtype=mx.int32)
        cos, sin = _rope_cos_sin(positions, self._inv_freq)
        q = _apply_partial_rope(q, cos, sin)

        cache.append_raw(k)
        pooled = self._extend_pooled(cache)
        T = cache.raw_keys.shape[1]
        nb_total = 0 if pooled is None else pooled.shape[1]

        # Per-query complete-block counts. If every visible prefix fits inside
        # the budget the selection is the full causal mask — skip the work.
        last_nb = (pos_start + S) // self.ratio
        if last_nb <= self.block_topk:
            return None  # dense == sparse in this regime

        pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]  # [1,1,D,nb]
        scores = mx.matmul(q.astype(mx.float32), pooled_t)  # [1,S,H,nb]
        scores = mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(self.head_dim)
        scores = scores[0]  # [S, nb]

        qpos = mx.arange(pos_start, pos_start + S, dtype=mx.int32)  # abs position
        nb_q = (qpos + 1) // self.ratio  # complete blocks visible per query [S]
        blk = mx.arange(nb_total, dtype=mx.int32)
        valid = blk[None, :] < nb_q[:, None]  # [S, nb]
        neg = mx.array(-mx.inf, dtype=mx.float32)
        masked_scores = mx.where(valid, scores, neg)
        # torch.topk tie-break (lowest index wins). Exact ties are common:
        # a block whose every head-dot is negative relu-scores exactly 0.0.
        masked_scores = masked_scores - blk.astype(mx.float32)[None, :] * 1e-12

        k_eff = min(self.block_topk, nb_total)
        top_idx = mx.argpartition(masked_scores, kth=nb_total - k_eff, axis=-1)[
            :, nb_total - k_eff :
        ]
        selected = mx.zeros((S, nb_total), dtype=mx.bool_)
        selected = mx.put_along_axis(
            selected, top_idx.astype(mx.int64), mx.array(True), axis=-1
        )
        selected = selected & valid  # -inf padding rows never select

        # Blocks -> tokens, plus the visible tail, intersected with causal.
        tok_sel = mx.repeat(selected, self.ratio, axis=1)  # [S, nb*ratio]
        if nb_total * self.ratio < T:
            pad = mx.zeros((S, T - nb_total * self.ratio), dtype=mx.bool_)
            tok_sel = mx.concatenate([tok_sel, pad], axis=1)
        tpos = mx.arange(T, dtype=mx.int32)
        tail = tpos[None, :] >= (nb_q[:, None] * self.ratio)
        causal = tpos[None, :] <= qpos[:, None]
        mask = (tok_sel | tail) & causal  # [S, T]
        return mask[None, None]  # [1, 1, S, T]


class Attention(nn.Module):
    """Gated GQA (qwen3_5 style: double-width q_proj, sigmoid output gate,
    per-head q/k RMSNorm, partial rotary) masked by the QSA indexer."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim * 2, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args) if args.indexer_n_heads else None
        rot = args.rotary_dim
        self._inv_freq = args.rope_theta ** (
            -mx.arange(0, rot, 2, dtype=mx.float32) / rot
        )

    def __call__(self, x: mx.array, cache: QSACache) -> mx.array:
        B, S, _ = x.shape
        pos_start = cache.offset

        sel_mask = None
        if self.indexer is not None:
            sel_mask = self.indexer(x, pos_start, cache)

        q = self.q_proj(x)
        q, gate = mx.split(q.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        positions = mx.arange(pos_start, pos_start + S, dtype=mx.int32)
        cos, sin = _rope_cos_sin(positions, self._inv_freq)
        q = _apply_partial_rope(q, cos, sin)
        k = _apply_partial_rope(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        k, v = cache.kv.update_and_fetch(k, v)
        T = k.shape[2]

        if sel_mask is not None:
            mask = sel_mask
        elif S > 1:
            qpos = mx.arange(pos_start, pos_start + S, dtype=mx.int32)
            tpos = mx.arange(T, dtype=mx.int32)
            mask = (tpos[None, :] <= qpos[:, None])[None, None]
        else:
            mask = None

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(vocab: int, ngram_size: int, ple_index: int, seed: int):
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(vocab, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_index
    out = []
    for i in range(ngram_size):
        v = (base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64
        out.append(2 * (_splitmix64(v) % half_bound) + 1)
    return out


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def _head_vocab_layout(base: int, heads: int, ple_index: int):
    sizes, offsets, total = [], [], 0
    prime = base - 1
    # global head index runs across PLE layers; sizes are consecutive primes
    for h in range(ple_index * heads + heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        if h >= ple_index * heads:
            sizes.append(prime)
            offsets.append(total)
            total += prime
    return sizes, offsets, total


class NGramTable(nn.Module):
    """The hashed n-gram embedding. Two modes:

    * materialized (tiny/test configs): a QuantizedEmbedding-shaped or plain
      `weight` parameter, gathered with mx.take.
    * sidecar (the real 51B table): `attach_sidecar()` points row gathers at
      numpy memmaps over ngram-table.safetensors. Rows are dequantized after
      the gather; only touched pages ever become resident. No parameter is
      registered, so the table never counts as loadable weight.
    """

    def __init__(self, rows: int, dim: int, sidecar: bool = False):
        super().__init__()
        self.rows = rows
        self.dim = dim
        self._sidecar_mode = sidecar
        if not sidecar:
            self.weight = mx.zeros((max(rows, 1), dim))
        self._sidecar = None

    def attach_sidecar(self, path: Path):
        self.pop("weight", None)
        header, data_start = _read_safetensors_header(path)
        meta = header.get("__metadata__", {})
        entries = {}
        names = ("weight",) if int(meta.get("ngram_bits", 4)) == 0 else ("weight", "scales", "biases")
        for name in names:
            info = header[f"ngram.{name}"]
            entries[name] = (info, data_start)
        self._sidecar = _SidecarGather(
            path,
            entries,
            bits=int(meta.get("ngram_bits", 4)),
            group_size=int(meta.get("ngram_group_size", 32)),
        )

    def __call__(self, ids: mx.array) -> mx.array:
        if self._sidecar is not None:
            return self._sidecar(ids, self.dim)
        if self._sidecar_mode:
            raise RuntimeError(
                "qwen4_exp n-gram table sidecar was never attached — "
                "ngram-table.safetensors is missing from the model directory"
            )
        return self.weight[ids]


class _SidecarGather:
    def __init__(self, path: Path, entries, bits: int, group_size: int):
        import numpy as np

        self.bits = bits
        self.group_size = group_size
        self._maps = {}
        for name, (info, data_start) in entries.items():
            dtype = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16}[info["dtype"]]
            shape = tuple(info["shape"])
            offset = data_start + info["data_offsets"][0]
            self._maps[name] = (
                np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape),
                info["dtype"],
            )

    def __call__(self, ids: mx.array, dim: int) -> mx.array:
        import numpy as np

        flat = np.asarray(ids.reshape(-1), dtype=np.int64)
        if self.bits == 0:  # raw bf16 rows, no dequantize
            mm, dt = self._maps["weight"]
            rows = mx.array(np.ascontiguousarray(mm[flat]))
            rows = rows.view(mx.bfloat16 if dt == "BF16" else mx.float16)
            return rows.reshape(*ids.shape, dim)
        parts = []
        for name in ("weight", "scales", "biases"):
            mm, dt = self._maps[name]
            rows = mx.array(np.ascontiguousarray(mm[flat]))
            if dt == "BF16":
                rows = rows.view(mx.bfloat16)
            elif dt == "F16":
                rows = rows.view(mx.float16)
            parts.append(rows)
        w, s, b = parts
        out = mx.dequantize(w, s, b, group_size=self.group_size, bits=self.bits)
        return out.reshape(*ids.shape, dim)


def _read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


class NGramEmbedding(nn.Module):
    def __init__(self, args: TextArgs, ple_index: int):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.eos_id = args.eos_id
        head_dim = args.ple_embed_dim // self.ngram_heads

        sizes, offsets, total = _head_vocab_layout(
            args.ngram_vocab_size_base, self.ngram_heads, ple_index
        )
        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        # Checkpoint buffers overwrite these derived values on load.
        self.layer_multipliers = mx.array(
            _build_layer_multipliers(args.vocab_size, args.ngram_size, ple_index, args.seed),
            dtype=mx.int64,
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = NGramTable(
            padded, head_dim, sidecar=getattr(args, "ngram_sidecar", False)
        )

    def _shift_ignore_eos(self, ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return ids
        B, L = ids.shape
        pos = mx.arange(L, dtype=mx.int64)[None, :]
        eos_pos = mx.where(ids == self.eos_id, pos, mx.array(-1, dtype=mx.int64))
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=mx.int64), prev_incl[:, :-1]], axis=1
        )
        seg_start = prev + 1
        pos_in_seg = pos - seg_start
        src = pos - shift
        gather = mx.maximum(src, 0)
        shifted = mx.take_along_axis(ids, gather, axis=1)
        valid = (pos_in_seg >= shift) & (src >= 0)
        return mx.where(valid, shifted, mx.array(self.eos_id, dtype=mx.int64))

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache], state_idx: int):
        ids = input_ids.astype(mx.int64)
        B, S = ids.shape
        if cache is not None and cache[state_idx] is not None:
            prev = cache[state_idx]
        else:
            prev = mx.full((B, self.context_len), self.eos_id, dtype=mx.int64)
        history = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[state_idx] = history[:, -self.context_len :]

        shifted = [self._shift_ignore_eos(history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            head_ids = mx.remainder(mixed[..., None], sizes.reshape(1, 1, -1))
            blocks.append(head_ids + offsets.reshape(1, 1, -1))
        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -S:]
        emb = self.ngram_embedding(ngram_ids)
        return emb.reshape(B, S, -1)


class PLELayer(nn.Module):
    """Per-Layer Embedding injection (runs on one linear-attention layer,
    before its hyper-connections). Cache slots: state_idx 2 = conv state,
    state_idx 3 = n-gram context ids."""

    CONV_IDX = 2
    NGRAM_IDX = 3

    def __init__(self, args: TextArgs, ple_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.ple_embedding = NGramEmbedding(args, ple_index)
        self.conv_kernel_size = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        self.norm_query = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        self.norm_conv = GroupedRMSNorm(hc_hidden, args.hidden_size, eps=args.rms_norm_eps)
        # Depthwise dilated conv, stored [channels, kernel, 1] (mlx layout).
        self.conv_weight = mx.zeros((hc_hidden, self.conv_kernel_size, 1))

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache]) -> mx.array:
        B, S, C = x.shape
        if cache is not None and cache[self.CONV_IDX] is not None:
            state = cache[self.CONV_IDX]
        else:
            state = mx.zeros((B, self.conv_state_len, C), dtype=x.dtype)
        window = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[self.CONV_IDX] = window[:, -self.conv_state_len :, :]
        out = mx.conv1d(
            window,
            self.conv_weight,
            stride=1,
            padding=0,
            dilation=self.conv_dilation,
            groups=C,
        )
        return nn.silu(out[:, -S:, :])

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache) -> mx.array:
        emb = self.ple_embedding(input_ids, cache, self.NGRAM_IDX)
        emb = emb.astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc_count, self.hidden_size)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc_count, self.hidden_size)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*hidden.shape)
        return gated + self._short_conv(self.norm_conv(gated), cache)


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
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)
        if (layer_idx + 1) in args.ple_layer_ids:
            self.ple = PLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
        self._hc = args.hc_count

    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        if "ple" in self:
            hidden = hidden + self.ple(hidden, input_ids, cache)

        mixed, hyper, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block_out = self.linear_attn(mixed, ssm_mask, cache)
        else:
            block_out = self.self_attn(mixed, cache)
        hidden = hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
            *hyper.shape
        )

        mixed, hyper, inject = self.mlp_hyper_connection(hidden)
        block_out = self.mlp(mixed)
        hidden = hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
            *hyper.shape
        )
        return hidden


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = (
            args.layer_types.index("linear_attention")
            if "linear_attention" in args.layer_types
            else 0
        )
        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t != "linear_attention"),
            self.ssm_idx,
        )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        h = mx.tile(h, (1, 1, self.args.hc_count))
        for layer, c in zip(self.layers, cache):
            h = layer(h, input_ids=inputs, ssm_mask=ssm_mask, cache=c)
        return self.hyper_connection_mixer(h)


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self):
        caches = []
        for i, layer in enumerate(self.model.layers):
            if not layer.is_linear:
                caches.append(QSACache())
            elif "ple" in layer:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(ArraysCache(size=2))
        return caches


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params.get("model_type", "qwen4_exp"), text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    """House-shaped wrapper: language_model.{model,lm_head}. Vision tensors
    ship in the pack (model-vision.safetensors) but are not constructed here —
    text serving is the phase-1 contract; the honest capability report for
    image inputs is "not yet supported for this family"."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_config = dict(args.text_config)
        # eos flows from the top-level config when text_config omits it
        self.language_model = TextModel(TextArgs.from_dict(text_config))

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.language_model(inputs, cache, input_embeddings)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def post_weight_load(self, model_path) -> None:
        """Attach the SSD-resident n-gram sidecar after weights load."""
        path = Path(model_path) / "ngram-table.safetensors"
        if not path.exists():
            return
        for layer in self.layers:
            if "ple" in layer:
                layer.ple.ple_embedding.ngram_embedding.attach_sidecar(path)

    # -- weight plumbing ---------------------------------------------------

    _HF_NORM_SHIFT_SUFFIXES = (
        ".q_norm.weight",
        ".k_norm.weight",
        ".q_layernorm.weight",
        ".k_layernorm.weight",
        ".hc_norm.weight",
        ".norm_key.weight",
        ".norm_query.weight",
        ".norm_conv.weight",
    )

    def sanitize(self, weights):
        # Raw-HF discriminator: unconverted checkpoints carry HF conv layout
        # ([ch, 1, k]) and the model.language_model prefix.
        raw = any(
            k.endswith("conv1d.weight") and v.shape[-1] != 1 for k, v in weights.items()
        ) or any(k.startswith("model.language_model.") for k in weights)

        out = {}
        stacked: dict[str, dict[int, mx.array]] = {}
        for k, v in weights.items():
            if k.startswith("model.visual.") or k.startswith("vision_tower."):
                continue
            if k.startswith("mtp."):
                continue
            if k.startswith("model.language_model."):
                k = k.replace("model.language_model.", "language_model.model.", 1)
            elif k == "lm_head.weight":
                k = "language_model.lm_head.weight"
            elif not k.startswith("language_model."):
                k = "language_model." + k

            if raw:
                # numbered per-expert tensors -> stacked switch_mlp
                if ".mlp.experts." in k and ".weight" in k and "scale_inv" not in k:
                    prefix, rest = k.split(".mlp.experts.", 1)
                    idx_s, proj_rest = rest.split(".", 1)
                    proj = proj_rest.rsplit(".weight", 1)[0]
                    dest = f"{prefix}.mlp.switch_mlp.{proj}.weight"
                    stacked.setdefault(dest, {})[int(idx_s)] = v
                    continue
                if ".mlp.experts.gate_up_proj" in k or ".mlp.experts.down_proj" in k:
                    # Two packed layouts exist. transformers save_pretrained
                    # writes the runtime bmm orientation (gate_up [E, hidden,
                    # 2*inter], down [E, inter, hidden]); the hub bf16 repo
                    # ships Linear [out, in] halves (gate_up [E, 2*inter,
                    # hidden], down [E, hidden, inter]). Keyed on which axis
                    # equals hidden; square (test-config) tensors resolve to
                    # the transformers branch, which parity validated.
                    prefix = k.split(".mlp.experts.", 1)[0]
                    hid = self.language_model.args.hidden_size
                    if k.endswith("gate_up_proj"):
                        if v.shape[1] == hid:
                            gate, up = mx.split(v, 2, axis=-1)
                            gate = gate.swapaxes(1, 2)
                            up = up.swapaxes(1, 2)
                        else:
                            gate, up = mx.split(v, 2, axis=1)
                        out[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                        out[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
                    else:
                        if v.shape[2] == hid:
                            v = v.swapaxes(1, 2)
                        out[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = v
                    continue
                if k.endswith("ple.conv1d.weight"):
                    out[k.replace("ple.conv1d.weight", "ple.conv_weight")] = v.moveaxis(2, 1)
                    continue
                if k.endswith("linear_attn.conv1d.weight") and v.shape[-1] != 1:
                    v = v.moveaxis(2, 1)
                if v.ndim == 1 and any(k.endswith(s) for s in self._HF_NORM_SHIFT_SUFFIXES):
                    v = v + 1.0
            else:
                if k.endswith("ple.conv1d.weight"):
                    k = k.replace("ple.conv1d.weight", "ple.conv_weight")

            out[k] = v

        for dest, parts in stacked.items():
            out[dest] = mx.stack([parts[i] for i in range(len(parts))])

        # The 51B table never loads as a parameter; sidecar rows are gathered
        # lazily. Materialized shard concat is only accepted for tiny test
        # configs (parity harnesses) where the table actually fits.
        table_keys = [k for k in out if ".ngram_embedding.shard_" in k]
        if table_keys:
            shards = {}
            for k in table_keys:
                idx = int(k.rsplit("shard_", 1)[1].split(".")[0])
                shards[idx] = out.pop(k)
            table = mx.concatenate([shards[i] for i in range(len(shards))], axis=0)
            key = next(
                (
                    k
                    for k in _tree_keys(self)
                    if k.endswith("ple.ple_embedding.ngram_embedding.weight")
                ),
                None,
            )
            if key is not None:
                want = _tree_get(self, key).shape[0]
                if table.shape[0] < want:
                    pad = mx.zeros(
                        (want - table.shape[0], table.shape[1]), dtype=table.dtype
                    )
                    table = mx.concatenate([table, pad], axis=0)
                out[key] = table

        return out

    @property
    def cast_predicate(self):
        def predicate(path: str) -> bool:
            if path.endswith("A_log"):
                return False
            if "ngram" in path or path.endswith("layer_multipliers"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        """Convert-time recipe (Optimized Speed): 4-bit/g32 base; 8-bit/g64
        embeddings, lm_head, GDN out_proj, router gates, shared expert and the
        QSA indexer projection; the structural small stuff stays bf16."""

        def predicate(path: str, module) -> Union[bool, dict]:
            if not hasattr(module, "to_quantized"):
                return False
            eight = (
                "embed_tokens",
                "lm_head",
                "linear_attn.out_proj",
                "mlp.gate",
                "shared_expert_gate",
                "shared_expert.gate_proj",
                "shared_expert.up_proj",
                "shared_expert.down_proj",
                "indexer.index_qk_proj",
            )
            keep = (
                "input_mix_weight_down",
                "input_mix_weight_up",
                "block_inject_weight",
                "ple.key_proj",
                "ple.value_proj",
                "ngram_embedding",
            )
            if any(path.endswith(s) or s in path for s in keep):
                return False
            if any(path.endswith(s) for s in eight):
                return {"bits": 8, "group_size": 64}
            return True

        return predicate


def _tree_keys(module: nn.Module):
    from mlx.utils import tree_flatten

    return [k for k, _ in tree_flatten(module.parameters())]


def _tree_get(module: nn.Module, dotted: str):
    from mlx.utils import tree_flatten

    for k, v in tree_flatten(module.parameters()):
        if k == dotted:
            return v
    raise KeyError(dotted)
