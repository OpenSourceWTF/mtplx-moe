"""MLX implementation of Tencent Hy3 with parameter-free routed experts."""

from __future__ import annotations

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

from .expert_mlx import UnboundExpertSwitch, run_switch_with_shared_overlap


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
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
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
        queries = (
            self.q_proj(x)
            .reshape(batch, length, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
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


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.route_norm = args.route_norm
        self.router_scaling_factor = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        storage_gate = _router_storage_module(self.gate)
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


class SparseMLP(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.router = Router(args)
        self.switch_mlp = UnboundExpertSwitch(layer_index)
        self.shared_mlp = MLP(
            args,
            intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
        )
        self.enable_moe_fp32_combine = args.enable_moe_fp32_combine

    def __call__(self, x: mx.array) -> mx.array:
        indices, scores = self.router(x)
        # Match the pinned Hy3 MLX reference: when FP32 MoE combining is
        # disabled, both the routing multiply and its reduction happen in the
        # activation dtype.  Keeping ``scores`` in FP32 here subtly changes
        # target logits even though the selected experts are identical.
        if not self.enable_moe_fp32_combine:
            scores = scores.astype(x.dtype)
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


class DecoderLayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_index: int,
        *,
        mlp_type: str | None = None,
    ):
        super().__init__()
        self.self_attn = Attention(args)
        resolved_mlp_type = mlp_type or args.mlp_layer_types[layer_index]
        self.mlp = (
            SparseMLP(args, layer_index) if resolved_mlp_type == "sparse" else MLP(args)
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
        head_hidden = recurrent_hidden
        if self._lm_head_fp32:
            head_hidden = head_hidden.astype(mx.float32)
        return lm_head(head_hidden), recurrent_hidden


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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_index)
            for layer_index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer(hidden, mask, layer_cache)
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
        self.model = Hy3Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        hidden = self.model(inputs, cache)
        if self.args.enable_lm_head_fp32:
            hidden = hidden.astype(mx.float32)
        return self.lm_head(hidden)

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
        return result

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _layer in self.layers]
