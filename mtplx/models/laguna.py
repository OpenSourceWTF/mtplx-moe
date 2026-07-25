# Copyright © 2026 PipeNetwork
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from pipenetwork/Laguna-S-2.1-MLX-4bit at revision
# 5544297f819d50330bc3616dd15cbc7edb598b2f (the architecture source is
# unchanged; only the admitted artifact moved to mlx-community/Laguna-S-2.1-oQ4e).
# Adaptations are absolute mlx-lm imports, formatting, construction-time geometry
# validation, and the oQ4e weight-name layout adapter in Model.sanitize().

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    decoder_sparse_step: int
    norm_topk_prob: bool
    moe_routed_scaling_factor: float = 1.0
    moe_router_logit_softcapping: float = 0.0
    mlp_only_layers: List[int] = field(default_factory=lambda: [0])
    num_attention_heads_per_layer: Optional[List[int]] = None
    gating: Any = "per-head"
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    swa_rope_parameters: Optional[Dict[str, Any]] = None
    partial_rotary_factor: Optional[float] = None
    max_position_embeddings: int = 1_048_576
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        unsupported_layer_types = set(self.layer_types) - {
            "full_attention",
            "sliding_attention",
        }
        if unsupported_layer_types:
            raise ValueError(
                f"unsupported layer_types: {sorted(unsupported_layer_types)}"
            )
        if (
            self.num_attention_heads_per_layer is not None
            and len(self.num_attention_heads_per_layer) != self.num_hidden_layers
        ):
            raise ValueError(
                "num_attention_heads_per_layer must match num_hidden_layers"
            )
        head_counts = (
            self.num_attention_heads_per_layer
            or [self.num_attention_heads] * self.num_hidden_layers
        )
        if self.num_key_value_heads <= 0 or any(
            count <= 0 or count % self.num_key_value_heads != 0 for count in head_counts
        ):
            raise ValueError(
                "attention head counts must be divisible by num_key_value_heads"
            )
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError(
                "sliding_window must be positive when sliding attention is installed"
            )
        rope_parameters = self.rope_parameters
        if self.swa_rope_parameters is None and isinstance(rope_parameters, dict):
            self.swa_rope_parameters = rope_parameters.get("sliding_attention")


def _rope_offset(offset: int, batch: int, memo: dict | None):
    """Return a rope offset that survives a batched single-token step.

    MLX 0.31.2's ``mx.fast.rope`` takes a "single" fast path whenever the input
    is row-contiguous, the sequence length is 1, and the offset holds one value.
    That path dispatches a TWO-dimensional grid — ``(dims/2, heads)`` — and
    indexes with ``pos.x + pos.y * stride``.  There is no batch term anywhere in
    it, so only batch element 0 is ever written; rows 1..B-1 come back as
    whatever the freshly allocated output buffer happened to hold.

    Every batched decode step has exactly that shape: ``[B, heads, 1, head_dim]``
    is row-contiguous because transposing a length-1 sequence dimension leaves
    the strides untouched.  Prefill (T > 1) is unaffected, which is why the
    corruption only appears once generation starts.

    Handing rope a length-B offset vector fails the ``offset.size() == 1``
    condition and routes the call to the general kernel, which computes a real
    ``batch_idx`` and reads a per-row offset.  The vector form is also what a
    ragged batch would need, so this is the API's intended shape, not a dodge.
    """

    if batch <= 1:
        return offset
    if memo is None:
        return mx.full((batch,), offset, dtype=mx.int32)
    cached = memo.get(offset)
    if cached is None:
        cached = mx.full((batch,), offset, dtype=mx.int32)
        memo[offset] = cached
    return cached


def _rope_for(config: dict, head_dim: int, max_position_embeddings: int) -> nn.Module:
    config = dict(config or {})
    theta = float(config.get("rope_theta", 10_000.0))
    partial = float(config.get("partial_rotary_factor", 1.0))
    dims = int(head_dim * partial)
    rope_type = config.get("rope_type", "default")
    if rope_type in ("default", "linear"):
        return initialize_rope(dims, base=theta, traditional=False)
    scaling_config = {
        "rope_type": rope_type,
        "factor": float(config.get("factor", 1.0)),
    }
    for key in (
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
    ):
        if config.get(key) is not None:
            scaling_config[key] = config[key]
    return initialize_rope(
        dims,
        base=theta,
        traditional=False,
        scaling_config=scaling_config,
        max_position_embeddings=max_position_embeddings,
    )


def _stock_per_head_gate(
    output: mx.array, gate_logits: mx.array, n_heads: int, head_dim: int
) -> mx.array:
    """Softplus the per-head gate logits and scale each head's slice by it."""

    batch, length, _ = output.shape
    gate = mx.logaddexp(
        gate_logits.astype(mx.float32), mx.array(0.0)
    ).astype(output.dtype)
    return (
        output.reshape(batch, length, n_heads, head_dim) * gate[..., None]
    ).reshape(batch, length, -1)


# Swapped by `laguna_fused.install_compiled_attention_gate`; the stock callable
# is the shipped behaviour and stays the default.
PER_HEAD_GATE_IMPL = _stock_per_head_gate


def _stock_moe_combine(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> mx.array:
    """Weighted sum of the routed expert outputs plus the shared expert."""

    combined = (
        expert_out * weights.astype(expert_out.dtype)[..., None]
    ).sum(axis=-2)
    return combined + shared


# Swapped by `laguna_fused.install_kernel_moe_combine`; the stock callable is
# the shipped behaviour and stays the default.
MOE_COMBINE_IMPL = _stock_moe_combine


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        per_layer = args.num_attention_heads_per_layer
        self.n_heads = (
            per_layer[layer_idx] if per_layer is not None else args.num_attention_heads
        )
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.gating = bool(args.gating)
        self.gate_per_head = args.gating == "per-head"
        if self.gating:
            gate_output = (
                self.n_heads if self.gate_per_head else self.n_heads * head_dim
            )
            self.g_proj = nn.Linear(dim, gate_output, bias=False)

        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = args.sliding_window if self.is_sliding else None
        rope_config = (
            args.swa_rope_parameters
            if self.is_sliding
            else (args.rope_parameters or {}).get(
                "full_attention", args.rope_parameters
            )
        )
        self.rope = _rope_for(
            rope_config,
            head_dim,
            args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask=None, cache=None, rope_memo=None) -> mx.array:
        batch, length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(batch, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(batch, length, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        offset = _rope_offset(
            cache.offset if cache is not None else 0, batch, rope_memo
        )
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
        output = output.transpose(0, 2, 1, 3).reshape(
            batch,
            length,
            self.n_heads * self.head_dim,
        )

        if self.gating:
            gate_logits = self.g_proj(x)
            if self.gate_per_head:
                output = PER_HEAD_GATE_IMPL(
                    output, gate_logits, self.n_heads, self.head_dim
                )
            else:
                gate = mx.logaddexp(
                    gate_logits.astype(mx.float32), mx.array(0.0)
                ).astype(output.dtype)
                output = output * gate

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.softcap = args.moe_router_logit_softcapping

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((self.num_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            self.num_experts,
        )
        self.shared_expert = MLP(
            args.hidden_size,
            args.shared_expert_intermediate_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, hidden = x.shape
        flattened = x.reshape(-1, hidden)

        logits = self.gate(flattened).astype(mx.float32)
        if self.softcap and self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        scores = mx.sigmoid(logits)
        scores_for_choice = scores + self.e_score_correction_bias.astype(mx.float32)

        indices = mx.argpartition(
            -scores_for_choice,
            kth=self.top_k - 1,
            axis=-1,
        )[..., : self.top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)

        output = self.switch_mlp(flattened, indices)
        output = MOE_COMBINE_IMPL(
            output, weights, self.shared_expert(flattened)
        )
        return output.reshape(batch, length, hidden)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )
        is_moe = (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        if is_moe:
            self.mlp = LagunaSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x: mx.array, mask=None, cache=None, rope_memo=None) -> mx.array:
        hidden = x + self.self_attn(
            self.input_layernorm(x), mask, cache, rope_memo
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        layer_types = args.layer_types
        self._first_full = next(
            (
                index
                for index, layer_type in enumerate(layer_types)
                if layer_type == "full_attention"
            ),
            0,
        )
        self._first_swa = next(
            (
                index
                for index, layer_type in enumerate(layer_types)
                if layer_type == "sliding_attention"
            ),
            0,
        )
        self._has_swa = "sliding_attention" in layer_types

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(hidden, cache[self._first_full])
        if self._has_swa:
            sliding_mask = create_attention_mask(
                hidden,
                cache[self._first_swa],
                window_size=self.args.sliding_window,
            )
        else:
            sliding_mask = full_mask

        # One offset vector per distinct cache offset per forward, shared by
        # every layer, so the batched-rope fix costs one tiny array and not one
        # per attention block.  See `_rope_offset` for why the vector is needed.
        rope_memo: dict[int, mx.array] = {}
        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.self_attn.is_sliding else full_mask
            hidden = layer(hidden, mask, layer_cache, rope_memo)

        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size,
                args.vocab_size,
                bias=False,
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings=None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ) -> mx.array | None:
        output = self.model(inputs, cache, input_embeddings)
        if not emit_logits:
            return None
        if logits_keep is not None:
            output = output[:, -max(1, int(logits_keep)) :, :]
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(output)
        return self.lm_head(output)

    def sanitize(self, weights):
        """Map the oQ4e export layout onto the vendored module tree.

        The pinned checkpoint wraps every tensor under ``language_model.`` and
        keeps the (unquantized) MoE router inside a ``gate.proj`` submodule with
        the load-balancing bias parked one level up under ``gate``. Strip the
        wrapper and fold both back onto the module-native names. The transform
        is total: every source key maps to exactly one target key, and the
        resulting set matches the module tree. Weights already in the native
        layout (no ``language_model.`` prefix, e.g. a directly constructed test
        model) pass through untouched.
        """

        if not any(name.startswith("language_model.") for name in weights):
            return weights
        remapped = {}
        for name, value in weights.items():
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            if name.endswith(".mlp.gate.proj.weight"):
                name = name[: -len(".proj.weight")] + ".weight"
            elif name.endswith(".mlp.gate.e_score_correction_bias"):
                name = (
                    name[: -len(".gate.e_score_correction_bias")]
                    + ".e_score_correction_bias"
                )
            remapped[name] = value
        return remapped

    def make_cache(self):
        caches = []
        for layer_type in self.args.layer_types:
            if layer_type == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    # No ``quant_predicate``: mlx-lm's load path quantizes from the checkpoint's
    # own config['quantization'] dict (per-path overrides, gs128 base, BF16
    # routers). The runtime strips the export prefix from that dict before the
    # load so it matches this module tree.

    @property
    def layers(self):
        return self.model.layers
