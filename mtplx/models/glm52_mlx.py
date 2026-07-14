"""GLM-5.2 MLX overlay with IndexShare and FP32 MoE routing.

IndexShare structure is adapted from mlx-lm PR #1410 (MIT), the branch used to
produce the pinned community Q4 checkpoint.  Router projection precision is
aligned with the official Transformers GLM-5.2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32MLP,
    Model as DeepseekV32CausalModel,
    MoEGate,
    group_expert_select,
)
from mlx_lm.models.switch_layers import SwitchGLU

from .expert_mlx import UnboundExpertSwitch, run_switch_with_shared_overlap


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict[str, Any]
    attention_bias: bool
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None
    indexer_rope_interleave: bool = True
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False
    moe_router_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.rope_scaling = self.rope_parameters
        self.rope_theta = float(self.rope_parameters["rope_theta"])
        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [
                        {"F": "full", "S": "shared"}[character] for character in pattern
                    ]
                else:
                    self.indexer_types = list(pattern)
            else:
                frequency = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full"
                    if (max(index - offset + 1, 0) % frequency) == 0
                    else "shared"
                    for index in range(self.num_hidden_layers)
                ]
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError("indexer_types must cover every GLM layer")
        if any(value not in {"full", "shared"} for value in self.indexer_types):
            raise ValueError("indexer_types values must be 'full' or 'shared'")


class FP32MoEGate(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        for name in (
            "top_k",
            "norm_topk_prob",
            "n_routed_experts",
            "routed_scaling_factor",
            "n_group",
            "topk_group",
        ):
            setattr(self, name, getattr(original, name))
        self.weight = original.weight
        self.e_score_correction_bias = original.e_score_correction_bias

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_expert_select(
            logits,
            self.e_score_correction_bias.astype(mx.float32),
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(
        self,
        config: ModelArgs,
        layer_index: int,
        *,
        indexer_type: Literal["full", "shared"] | None = None,
    ):
        super().__init__(config)
        resolved_indexer_type = (
            config.indexer_types[layer_index] if indexer_type is None else indexer_type
        )
        if resolved_indexer_type not in {"full", "shared"}:
            raise ValueError("indexer_type must be 'full' or 'shared'")
        self.skip_topk = resolved_indexer_type == "shared"
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
        *,
        compute_topk: bool | None = None,
        kv_read_boundary: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        batch, length, _ = x.shape
        qr = self.q_a_layernorm(self.q_a_proj(x))
        queries = self.q_b_proj(qr)
        queries = queries.reshape(
            batch,
            length,
            self.num_heads,
            self.q_head_dim,
        ).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(queries, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(
            batch,
            length,
            1,
            self.qk_rope_head_dim,
        ).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2
        should_compute_topk = self.indexer is not None and compute_topk is not False
        if should_compute_topk:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices
        if kv_read_boundary is not None:
            read_boundary = min(
                max(int(kv_read_boundary), 0),
                int(kv_latent.shape[2]),
            )
            kv_latent = kv_latent[..., :read_boundary, :]
            k_pe = k_pe[..., :read_boundary, :]
            if mask is not None:
                mask = mask[..., :read_boundary]
        if topk_indices is not None:
            if length == 1:
                index = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(index, index.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(index, index.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask,
                    topk_indices,
                    mx.array(True),
                    axis=-1,
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask
        if should_compute_topk and cache[0] is not None:
            cache[0].keys = mx.depends(
                cache[0].keys,
                (cache[1].keys, cache[1].values),
            )
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )
        if length == 1:
            q_nope = self.embed_q(q_nope)
            keys = values = kv_latent
        else:
            keys = self.embed_q(kv_latent, transpose=False)
            values = self.unembed_out(kv_latent)
        output = scaled_dot_product_attention(
            q_nope,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
        )
        if length == 1:
            output = self.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output), topk_indices


class StreamedMoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_index: int):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = UnboundExpertSwitch(layer_index)
        self.gate = FP32MoEGate(MoEGate(config))
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV32MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        indices, scores = self.gate(x)
        if self.config.n_shared_experts is None:
            output = self.switch_mlp(x, indices)
            shared = None
        else:
            output, shared = run_switch_with_shared_overlap(
                self.switch_mlp,
                x,
                indices,
                lambda: self.shared_experts(x),
            )
        output = (output * scores[..., None]).sum(axis=-2).astype(output.dtype)
        if shared is not None:
            output = output + shared
        return output


class GlmMoeDsaResidentMoE(nn.Module):
    """GLM routed MoE with one resident, stacked BF16 expert bank."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.switch_mlp.set_dtype(mx.bfloat16)
        self.gate = FP32MoEGate(MoEGate(config))
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV32MLP(
                config=config,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
            )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        indices, scores = self.gate(x)
        output = self.switch_mlp(x, indices)
        output = (output * scores[..., None]).sum(axis=-2).astype(output.dtype)
        if self.config.n_shared_experts is not None:
            output = output + self.shared_experts(x)
        return output


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        layer_index: int,
        *,
        expert_mode: Literal["streamed", "resident"] = "streamed",
        indexer_type: Literal["full", "shared"] | None = None,
    ):
        super().__init__()
        if expert_mode not in {"streamed", "resident"}:
            raise ValueError("expert_mode must be 'streamed' or 'resident'")
        self.self_attn = GlmMoeDsaAttention(
            config,
            layer_index,
            indexer_type=indexer_type,
        )
        is_sparse = (
            config.n_routed_experts is not None
            and layer_index >= config.first_k_dense_replace
            and layer_index % config.moe_layer_freq == 0
        )
        if is_sparse and expert_mode == "resident":
            self.mlp = GlmMoeDsaResidentMoE(config)
        elif is_sparse:
            self.mlp = StreamedMoE(config, layer_index)
        else:
            self.mlp = DeepseekV32MLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
        *,
        compute_topk: bool | None = None,
        kv_read_boundary: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        residual, topk_indices = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            prev_topk_indices,
            compute_topk=compute_topk,
            kv_read_boundary=kv_read_boundary,
        )
        hidden = x + residual
        residual = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + residual, topk_indices


class Glm52MTPLayer(nn.Module):
    """GLM-5.2 NextN layer with resident experts and call-time shared heads."""

    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.mtp_block = GlmMoeDsaDecoderLayer(
            args,
            layer_index,
            expert_mode="resident",
            indexer_type="full",
        )
        self.shared_head_norm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        input_ids: mx.array,
        previous_hidden_states: mx.array,
        *,
        embed_tokens: Any,
        lm_head: Any,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
        compute_topk: bool | None = None,
        kv_read_boundary: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        inputs_embeds = embed_tokens(input_ids)
        mixed = self.eh_proj(
            mx.concatenate(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                axis=-1,
            )
        )
        main_cache = cache[0] if cache is not None else None
        mask = create_attention_mask(mixed, main_cache, return_array=True)
        hidden, topk_indices = self.mtp_block(
            mixed,
            mask,
            cache,
            prev_topk_indices,
            compute_topk=compute_topk,
            kv_read_boundary=kv_read_boundary,
        )
        recycle_hidden = self.shared_head_norm(hidden)
        logits = lm_head(recycle_hidden)
        return logits, recycle_hidden, topk_indices


class Glm52MTP(nn.Module):
    """Container for GLM-5.2 NextN layers."""

    def __init__(self, args: ModelArgs, num_mtp_layers: int = 1):
        super().__init__()
        if num_mtp_layers < 1:
            raise ValueError("GLM-5.2 MTP requires at least one NextN layer")
        self.start_layer = args.num_hidden_layers
        self.layers = [
            Glm52MTPLayer(args, self.start_layer + index)
            for index in range(num_mtp_layers)
        ]
        self.num_mtp_layers = num_mtp_layers


class GlmMoeDsaModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            hidden,
            cache[0][0] if cache[0] else None,
            return_array=True,
        )
        previous_topk = None
        for index in range(self.num_layers):
            hidden, previous_topk = self.layers[self.start_idx + index](
                hidden,
                mask,
                cache[index],
                previous_topk,
            )
        return self.norm(hidden)


class Model(DeepseekV32CausalModel):
    def __init__(self, config: ModelArgs):
        nn.Module.__init__(self)
        self.args = config
        self.model_type = config.model_type
        self.model = GlmMoeDsaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
