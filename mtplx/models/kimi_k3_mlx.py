"""Exact MLX text components for the pinned Moonshot Kimi K3 checkpoint.

This is a dedicated MTPLX overlay.  It deliberately does not patch
``mlx_lm.models.kimi_linear`` because that implementation has different
q-projection, MoE, KDA-gate, and attention-residual arithmetic.
"""

from __future__ import annotations

import inspect
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs

from .expert_mlx import UnboundExpertSwitch


SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
KDA_GATE_LOWER_BOUND = -5.0
K3_CHECKPOINT_A_LOG_HEADS = 128
K3_RUNTIME_KDA_HEADS = 96
_TEST_ONLY_GEOMETRY_CAPABILITY = object()


def situ(
    gate: mx.array,
    up: mx.array,
) -> mx.array:
    """K3's SITU activation, including its required FP32 work dtype."""

    output_dtype = gate.dtype
    gate_fp32 = gate.astype(mx.float32)
    up_fp32 = up.astype(mx.float32)
    situ_gate = SITU_BETA * mx.tanh(gate_fp32 / SITU_BETA) * mx.sigmoid(gate_fp32)
    bounded_up = SITU_LINEAR_BETA * mx.tanh(up_fp32 / SITU_LINEAR_BETA)
    return (situ_gate * bounded_up).astype(output_dtype)


class KimiRMSNorm(nn.Module):
    """The checkpoint RMSNorm: FP32 variance, then cast before weighting."""

    def __init__(self, dimensions: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        if dimensions <= 0:
            raise ValueError("RMSNorm dimensions must be positive")
        if eps != 1e-5:
            raise ValueError("Kimi K3 RMSNorm epsilon must be exactly 1e-5")
        self.eps = float(eps)
        self.weight = mx.ones((dimensions,), dtype=mx.float32)

    def __call__(self, value: mx.array) -> mx.array:
        dtype = value.dtype
        work = value.astype(mx.float32)
        normalized = work * mx.rsqrt(
            mx.mean(work * work, axis=-1, keepdims=True) + self.eps
        )
        return normalized.astype(dtype) * self.weight


class KimiSITUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.down_proj(situ(self.gate_proj(value), self.up_proj(value)))


class KimiMoERouter(nn.Module):
    """K3's one-group FP32 sigmoid router.

    The correction bias affects selection only.  Returned mixture weights are
    gathered from the unbiased scores and renormalized before applying the
    routed scaling factor.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        num_expert_group: int = 1,
        topk_group: int = 1,
    ) -> None:
        super().__init__()
        if not 1 < top_k <= num_experts:
            raise ValueError("K3 router top_k must be in [2, num_experts]")
        if num_expert_group != 1 or topk_group != 1:
            raise ValueError("Kimi K3 routing has exactly one expert group")
        self.top_k = int(top_k)
        self.num_experts = int(num_experts)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.weight = mx.zeros((num_experts, hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)
        self._route = self._route_topk_renormalized

    def _route_topk_renormalized(
        self, hidden_states: mx.array
    ) -> tuple[mx.array, mx.array]:
        logits = hidden_states.astype(mx.float32) @ self.weight.astype(
            mx.float32
        ).swapaxes(-1, -2)
        unbiased_scores = mx.sigmoid(logits)
        selection_scores = unbiased_scores + self.e_score_correction_bias.astype(
            mx.float32
        )
        indices = mx.argpartition(-selection_scores, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        weights = mx.take_along_axis(unbiased_scores, indices, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return indices, weights * self.routed_scaling_factor

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        return self._route(hidden_states)


def _weighted_routed_sum(selected: mx.array, weights: mx.array) -> mx.array:
    """Apply FP32 router weights, then restore the expert output dtype."""

    return mx.sum(
        selected.astype(mx.float32) * weights.astype(mx.float32)[..., None],
        axis=-2,
    ).astype(selected.dtype)


class KimiLatentMoE(nn.Module):
    """K3 latent MoE with the MTPLX streamed-switch ownership seam."""

    def __init__(
        self,
        *,
        hidden_size: int,
        routed_hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int,
        routed_scaling_factor: float,
        rms_norm_eps: float,
        layer_index: int = 0,
    ) -> None:
        super().__init__()
        if num_shared_experts != 2:
            raise ValueError("Kimi K3 requires exactly two shared experts")
        if rms_norm_eps != 1e-5:
            raise ValueError("Kimi K3 latent MoE RMSNorm epsilon must be 1e-5")
        self.gate = KimiMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.routed_expert_down_proj = nn.Linear(
            hidden_size, routed_hidden_size, bias=False
        )
        self.switch_mlp: nn.Module = UnboundExpertSwitch(layer_index)
        self.routed_expert_norm = KimiRMSNorm(routed_hidden_size, eps=rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(
            routed_hidden_size, hidden_size, bias=False
        )
        self.shared_experts = KimiSITUMLP(
            hidden_size, intermediate_size * num_shared_experts
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        identity = hidden_states
        indices, weights = self.gate(identity)
        latent = self.routed_expert_down_proj(identity)
        selected = self.switch_mlp(latent, indices)
        routed = _weighted_routed_sum(selected, weights)
        routed = self.routed_expert_norm(routed)
        routed = self.routed_expert_up_proj(routed)
        return routed + self.shared_experts(identity)


class MLALatentCache:
    """One-head compressed MLA cache.

    K3 stores one latent head whose last dimension is
    ``kv_lora_rank + qk_rope_head_dim`` (512 + 64 in production).
    """

    step = 256

    def __init__(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        *,
        step: int | None = None,
    ) -> None:
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.step = type(self).step if step is None else int(step)
        if self.kv_lora_rank <= 0 or self.qk_rope_head_dim <= 0:
            raise ValueError("MLA cache dimensions must be positive")
        if self.step <= 0:
            raise ValueError("MLA cache step must be positive")
        self.buffer: mx.array | None = None
        self.offset = 0

    @property
    def capacity(self) -> int:
        return 0 if self.buffer is None else int(self.buffer.shape[-2])

    @property
    def state(self) -> mx.array | None:
        if self.buffer is None:
            return None
        if self.offset == self.capacity:
            return self.buffer
        return self.buffer[..., : self.offset, :]

    @state.setter
    def state(self, value: mx.array | None) -> None:
        if value is None:
            self.buffer = None
            self.offset = 0
            return
        expected_width = self.kv_lora_rank + self.qk_rope_head_dim
        if value.ndim != 4 or value.shape[-1] != expected_width:
            raise ValueError(
                f"MLA cache state must have shape (batch, 1, length, {expected_width})"
            )
        if value.shape[1] != 1:
            raise ValueError("MLA cache state must contain exactly one latent head")
        self.buffer = value
        self.offset = int(value.shape[-2])

    def replace_state(self, value: mx.array | None) -> None:
        self.state = value

    def update_and_fetch(
        self, latent: mx.array, k_rot: mx.array
    ) -> tuple[mx.array, mx.array]:
        combined = mx.concatenate([latent, k_rot], axis=-1)
        previous = self.offset
        length = int(combined.shape[-2])
        required = previous + length
        buffer = self.buffer
        capacity = 0 if buffer is None else int(buffer.shape[-2])
        if buffer is None or required > capacity:
            batch, heads, _, width = combined.shape
            capacity = max(self.step, capacity)
            while capacity < required:
                capacity *= 2
            grown = mx.zeros((batch, heads, capacity, width), dtype=combined.dtype)
            if buffer is not None:
                grown[..., :previous, :] = buffer[..., :previous, :]
            buffer = grown
            self.buffer = buffer

        self.offset = required
        buffer[..., previous:required, :] = combined
        logical = buffer[..., :required, :]
        return (
            logical[..., : self.kv_lora_rank],
            logical[..., self.kv_lora_rank :],
        )

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self) -> bool:
        return self.buffer is None

    @property
    def nbytes(self) -> int:
        return 0 if self.buffer is None else int(self.buffer.nbytes)


class KimiMLAAttention(nn.Module):
    """K3 q-LoRA MLA with the checkpoint's published NoPE behavior."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        if (
            min(
                hidden_size,
                num_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
            )
            <= 0
        ):
            raise ValueError("all K3 MLA dimensions must be positive")
        if rms_norm_eps != 1e-5:
            raise ValueError("K3 MLA RMSNorm epsilon must be exactly 1e-5")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.q_lora_rank = int(q_lora_rank)
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = int(v_head_dim)
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = KimiRMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = KimiRMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.g_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None = None,
        cache: MLALatentCache | None = None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        previous_length = 0 if cache is None else cache.offset

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape(batch, length, self.num_heads, self.q_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_rot = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_rot = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        latent = self.kv_a_layernorm(latent)
        latent = latent[:, None, :, :]
        k_rot = k_rot[:, None, :, :]
        if cache is not None:
            latent, k_rot = cache.update_and_fetch(latent, k_rot)

        kv_weight = self.kv_b_proj.weight.reshape(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        key_weight = kv_weight[:, : self.qk_nope_head_dim, :]
        value_weight = kv_weight[:, self.qk_nope_head_dim :, :]
        q_latent = mx.einsum("bhln,hnr->bhlr", q_nope, key_weight)
        scores = q_latent @ latent.swapaxes(-1, -2)
        scores = scores + q_rot @ k_rot.swapaxes(-1, -2)
        scores = scores * self.scale

        key_length = int(latent.shape[-2])
        causal = (
            mx.arange(previous_length, previous_length + length)[:, None]
            >= mx.arange(key_length)[None, :]
        )
        scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
        if mask is not None and not isinstance(mask, str):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        probs = mx.softmax(scores.astype(mx.float32), axis=-1, precise=True).astype(
            q.dtype
        )
        latent_output = probs @ latent
        output = mx.einsum("bhlr,hvr->bhlv", latent_output, value_weight)
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, length, self.num_heads * self.v_head_dim
        )
        output = output * mx.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(output)


class KDACache:
    _META_VERSION = "mtplx-kimi-k3-kda-cache-v1"

    def __init__(self) -> None:
        self.q_conv: mx.array | None = None
        self.k_conv: mx.array | None = None
        self.v_conv: mx.array | None = None
        self.recurrent: mx.array | None = None
        self.offset = 0

    @property
    def state(
        self,
    ) -> tuple[
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
    ]:
        return self.q_conv, self.k_conv, self.v_conv, self.recurrent

    @state.setter
    def state(self, value: Any) -> None:
        if value is None:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None
            self.recurrent = None
            self.offset = 0
            return
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise ValueError("KDA cache state must contain Q/K/V conv and recurrent")
        self.q_conv, self.k_conv, self.v_conv, self.recurrent = value

    def replace_state(self, value: Any) -> None:
        self.state = value

    @property
    def meta_state(self) -> tuple[str, str]:
        return self._META_VERSION, str(self.offset)

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or value[0] != self._META_VERSION
        ):
            raise ValueError(f"unsupported KDA cache meta state: {value!r}")
        self.offset = int(value[1])

    def is_trimmable(self) -> bool:
        return False

    def size(self) -> int:
        return int(self.offset)

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.state if value is not None)


class ShortConv1d(nn.Module):
    """Causal depthwise convolution with the checkpoint's MLX weight layout."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 1:
            raise ValueError("short convolution requires channels>0 and kernel>1")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.weight = mx.zeros((channels, kernel_size, 1), dtype=mx.float32)

    def __call__(
        self,
        value: mx.array,
        state: mx.array | None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if mask is not None:
            value = mx.where(mask[..., None], value, 0)
        if state is None:
            state = mx.zeros(
                (value.shape[0], self.kernel_size - 1, self.channels),
                dtype=value.dtype,
            )
        padded = mx.concatenate([state, value], axis=1)
        channel_weight = self.weight[..., 0].swapaxes(0, 1)
        outputs = [
            mx.sum(
                padded[:, index : index + self.kernel_size, :] * channel_weight,
                axis=1,
            )
            for index in range(value.shape[1])
        ]
        output = nn.silu(mx.stack(outputs, axis=1))
        return output, mx.contiguous(padded[:, -(self.kernel_size - 1) :, :])


def map_kda_a_log(value: mx.array, *, num_heads: int) -> mx.array:
    """Apply the pinned vLLM K3 checkpoint mapping, failing closed.

    The published checkpoint carries 128 values while K3 configures 96 KDA
    heads.  vLLM's construction loader narrows that exact source to its first
    96 values.  No alternate short or surplus layout is accepted here.
    """

    if num_heads != K3_RUNTIME_KDA_HEADS:
        raise ValueError("K3 A_log mapping requires exactly 96 configured heads")
    if tuple(value.shape) != (K3_CHECKPOINT_A_LOG_HEADS,):
        raise ValueError("K3 checkpoint A_log must have exact shape [128]")
    return mx.contiguous(value[:num_heads].astype(mx.float32))


def _normalize_kda_qk(q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
    """FP32 unscaled L2 normalization, then projection-dtype restore."""

    q_dtype = q.dtype
    k_dtype = k.dtype
    q_fp32 = q.astype(mx.float32)
    k_fp32 = k.astype(mx.float32)
    q_normalized = q_fp32 * mx.rsqrt(
        mx.sum(q_fp32 * q_fp32, axis=-1, keepdims=True) + 1e-6
    )
    k_normalized = k_fp32 * mx.rsqrt(
        mx.sum(k_fp32 * k_fp32, axis=-1, keepdims=True) + 1e-6
    )
    return q_normalized.astype(q_dtype), k_normalized.astype(k_dtype)


class KimiDeltaAttention(nn.Module):
    """K3 KDA with safe vector forget gate and FP32 recurrent state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
        gate_lower_bound: float,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_heads, head_dim) <= 0:
            raise ValueError("all KDA dimensions must be positive")
        if gate_lower_bound != KDA_GATE_LOWER_BOUND:
            raise ValueError("Kimi K3 safe KDA lower bound must be exactly -5")
        if rms_norm_eps != 1e-5:
            raise ValueError("Kimi K3 KDA RMSNorm epsilon must be exactly 1e-5")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = int(conv_kernel_size)
        self.gate_lower_bound = float(gate_lower_bound)
        self.q_scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.q_conv1d = ShortConv1d(self.projection_size, conv_kernel_size)
        self.k_conv1d = ShortConv1d(self.projection_size, conv_kernel_size)
        self.v_conv1d = ShortConv1d(self.projection_size, conv_kernel_size)
        self.A_log = mx.zeros((num_heads,), dtype=mx.float32)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.dt_bias = mx.zeros((self.projection_size,), dtype=mx.float32)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.o_norm = KimiRMSNorm(head_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)
        self._prefill_chunk_size = 128
        self._direct_executor = self._call_direct
        self._long_prefill_executor = self._call_chunked

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        cache: KDACache | None = None,
    ) -> mx.array:
        if hidden_states.shape[1] <= self._prefill_chunk_size:
            return self._direct_executor(hidden_states, mask, cache)
        return self._long_prefill_executor(hidden_states, mask, cache)

    def _call_chunked(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        cache: KDACache | None = None,
    ) -> mx.array:
        active_cache = KDACache() if cache is None else cache
        outputs = []
        for start in range(0, hidden_states.shape[1], self._prefill_chunk_size):
            stop = min(start + self._prefill_chunk_size, hidden_states.shape[1])
            chunk_mask = None if mask is None else mask[:, start:stop]
            output = self._direct_executor(
                hidden_states[:, start:stop, :],
                chunk_mask,
                active_cache,
            )
            mx.eval(
                output,
                *(value for value in active_cache.state if value is not None),
            )
            outputs.append(output)
        return mx.concatenate(outputs, axis=1)

    def _call_direct(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        cache: KDACache | None = None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        q_state = None if cache is None else cache.q_conv
        k_state = None if cache is None else cache.k_conv
        v_state = None if cache is None else cache.v_conv

        q, q_state = self.q_conv1d(self.q_proj(hidden_states), q_state, mask)
        k, k_state = self.k_conv1d(self.k_proj(hidden_states), k_state, mask)
        v, v_state = self.v_conv1d(self.v_proj(hidden_states), v_state, mask)
        if cache is not None:
            cache.q_conv = q_state
            cache.k_conv = k_state
            cache.v_conv = v_state

        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)
        q, k = _normalize_kda_qk(q, k)

        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        beta = mx.sigmoid(self.b_proj(hidden_states).astype(mx.float32))
        a = mx.exp(self.A_log.astype(mx.float32))[None, None, :, None]
        dt_bias = self.dt_bias.astype(mx.float32).reshape(self.num_heads, self.head_dim)
        log_decay = self.gate_lower_bound * mx.sigmoid(
            a * (raw_gate.astype(mx.float32) + dt_bias)
        )
        decay = mx.exp(log_decay)

        state = None if cache is None else cache.recurrent
        if state is None:
            state = mx.zeros(
                (
                    batch,
                    self.num_heads,
                    self.head_dim,
                    self.head_dim,
                ),
                dtype=mx.float32,
            )
        outputs = []
        for index in range(length):
            old_state = state
            state = state * decay[:, index, :, None, :]
            memory = mx.sum(state * k[:, index, :, None, :].astype(mx.float32), axis=-1)
            delta = (v[:, index].astype(mx.float32) - memory) * beta[:, index, :, None]
            state = (
                state + k[:, index, :, None, :].astype(mx.float32) * delta[..., None]
            )
            recurrent_q = q[:, index, :, None, :].astype(mx.float32) * self.q_scale
            output = mx.sum(state * recurrent_q, axis=-1)
            if mask is not None:
                active = mask[:, index, None, None]
                state = mx.where(active, state, old_state)
                output = mx.where(active, output, 0)
            outputs.append(output.astype(hidden_states.dtype))
        output = mx.stack(outputs, axis=1)
        if cache is not None:
            cache.recurrent = state
            cache.offset += length

        output = self.o_norm(output)
        output_gate = self.g_proj(hidden_states).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        output = output * mx.sigmoid(output_gate)
        return self.o_proj(output.reshape(batch, length, self.projection_size))


def apply_attn_res(
    prefix_sum: mx.array,
    block_residual: mx.array,
    *,
    proj: nn.Linear,
    norm: nn.Module,
) -> mx.array:
    """K3 attention-residual weighted reduction in FP32."""

    values = mx.concatenate([block_residual, prefix_sum[:, None, :]], axis=1)
    values_fp32 = values.astype(mx.float32)
    eps = float(getattr(norm, "eps"))
    normalized = values_fp32 * mx.rsqrt(
        mx.mean(values_fp32 * values_fp32, axis=-1, keepdims=True) + eps
    )
    score_weight = norm.weight.astype(mx.float32) * proj.weight.squeeze(axis=0).astype(
        mx.float32
    )
    scores = mx.sum(normalized * score_weight, axis=-1)
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    output = mx.sum(probabilities[..., None] * values_fp32, axis=1)
    return output.astype(values.dtype)


def _default_linear_attn_config() -> Dict[str, Any]:
    full = list(range(4, 94, 4)) + [93]
    kda = [layer for layer in range(1, 94) if layer not in full]
    return {
        "full_attn_layers": full,
        "kda_layers": kda,
        "num_heads": 96,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    intermediate_size: int = 33792
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: int = 3584
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    linear_attn_config: Dict[str, Any] = field(
        default_factory=_default_linear_attn_config
    )
    rms_norm_eps: float = 1e-5
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    hidden_act: str = "situ"
    latent_moe_use_norm: bool = True
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    routed_scaling_factor: float = 1.0
    moe_renormalize: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    attn_res_block_size: int = 12
    tie_word_embeddings: bool = False
    _test_geometry_capability: InitVar[object | None] = None

    def __post_init__(self, _test_geometry_capability: object | None) -> None:
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.moe_intermediate_size,
            self.routed_expert_hidden_size,
            self.num_experts,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all Kimi K3 model dimensions must be positive")
        if not 1 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("num_experts_per_token is outside expert geometry")
        if self.num_shared_experts != 2:
            raise ValueError("Kimi K3 requires exactly two shared experts")
        if (
            self.hidden_act != "situ"
            or self.activation_situ_beta != 4.0
            or self.activation_situ_linear_beta != 25.0
        ):
            raise ValueError("Kimi K3 requires exact SITU(4,25) activation")
        if self.rms_norm_eps != 1e-5:
            raise ValueError("Kimi K3 RMSNorm epsilon must be exactly 1e-5")
        if not self.latent_moe_use_norm:
            raise ValueError("Kimi K3 requires latent MoE normalization")
        if not self.mla_use_nope or not self.mla_use_output_gate:
            raise ValueError("Kimi K3 requires NoPE MLA with its output gate")
        if self.num_expert_group != 1 or self.topk_group != 1:
            raise ValueError("Kimi K3 requires one expert group")
        if not self.moe_renormalize:
            raise ValueError("Kimi K3 requires renormalized routed weights")
        if (
            self.routed_scaling_factor != 1.0
            or self.first_k_dense_replace != 1
            or self.moe_layer_freq != 1
            or self.attn_res_block_size != 12
            or self.tie_word_embeddings
        ):
            raise ValueError(
                "Kimi K3 requires scale=1, dense layer 0, block=12, and untied head"
            )
        config = self.linear_attn_config
        if (
            not config.get("use_full_rank_gate")
            or config.get("gate_lower_bound") != -5.0
        ):
            raise ValueError("Kimi K3 requires full-rank safe KDA gate -5")
        if int(config["num_heads"]) != self.num_attention_heads:
            raise ValueError("KDA and MLA head counts must match")
        covered = set(config.get("kda_layers", ())) | set(
            config.get("full_attn_layers", ())
        )
        if covered != set(range(1, self.num_hidden_layers + 1)):
            raise ValueError("KDA/full-attention routes must cover every layer once")
        if set(config.get("kda_layers", ())) & set(config.get("full_attn_layers", ())):
            raise ValueError("KDA/full-attention layer routes overlap")
        if _test_geometry_capability is _TEST_ONLY_GEOMETRY_CAPABILITY:
            return

        pinned_scalars = {
            "model_type": "kimi_linear",
            "vocab_size": 163840,
            "hidden_size": 7168,
            "num_hidden_layers": 93,
            "num_attention_heads": 96,
            "num_key_value_heads": 96,
            "intermediate_size": 33792,
            "moe_intermediate_size": 3072,
            "routed_expert_hidden_size": 3584,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rms_norm_eps": 1e-5,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "hidden_act": "situ",
            "latent_moe_use_norm": True,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "routed_scaling_factor": 1.0,
            "moe_renormalize": True,
            "num_expert_group": 1,
            "topk_group": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "attn_res_block_size": 12,
            "tie_word_embeddings": False,
        }
        if any(
            getattr(self, name) != expected for name, expected in pinned_scalars.items()
        ):
            raise ValueError("production ModelArgs must match pinned Kimi K3 geometry")
        if self.linear_attn_config != _default_linear_attn_config():
            raise ValueError("production ModelArgs must match pinned Kimi K3 topology")

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        """Checkpoint construction cannot opt into the test-only geometry."""

        allowed = inspect.signature(cls).parameters
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in allowed and key != "_test_geometry_capability"
            }
        )

    @classmethod
    def tiny(cls, **values: Any) -> "ModelArgs":
        """Explicit test-only geometry constructor with production invariants."""

        return cls(
            **values,
            _test_geometry_capability=_TEST_ONLY_GEOMETRY_CAPABILITY,
        )


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.attn_res_block_size = int(args.attn_res_block_size)
        self.is_linear_attn = (layer_index + 1) in set(
            args.linear_attn_config["kda_layers"]
        )
        if self.is_linear_attn:
            self.self_attn: nn.Module = KimiDeltaAttention(
                hidden_size=args.hidden_size,
                num_heads=args.linear_attn_config["num_heads"],
                head_dim=args.linear_attn_config["head_dim"],
                conv_kernel_size=args.linear_attn_config["short_conv_kernel_size"],
                rms_norm_eps=args.rms_norm_eps,
                gate_lower_bound=args.linear_attn_config["gate_lower_bound"],
            )
        else:
            self.self_attn = KimiMLAAttention(
                hidden_size=args.hidden_size,
                num_heads=args.num_attention_heads,
                q_lora_rank=args.q_lora_rank,
                kv_lora_rank=args.kv_lora_rank,
                qk_nope_head_dim=args.qk_nope_head_dim,
                qk_rope_head_dim=args.qk_rope_head_dim,
                v_head_dim=args.v_head_dim,
                rms_norm_eps=args.rms_norm_eps,
            )
        if (
            layer_index >= args.first_k_dense_replace
            and layer_index % args.moe_layer_freq == 0
        ):
            self.mlp: nn.Module = KimiLatentMoE(
                hidden_size=args.hidden_size,
                routed_hidden_size=args.routed_expert_hidden_size,
                intermediate_size=args.moe_intermediate_size,
                num_experts=args.num_experts,
                top_k=args.num_experts_per_token,
                num_shared_experts=args.num_shared_experts,
                routed_scaling_factor=args.routed_scaling_factor,
                rms_norm_eps=args.rms_norm_eps,
                layer_index=layer_index,
            )
        else:
            self.mlp = KimiSITUMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = KimiRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.self_attention_res_norm = KimiRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp_res_norm = KimiRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
        if layer_index == 0:
            self._prepare_attn_res = self._prepare_first_block
            self._combine_attention = self._replace_prefix
        elif layer_index % self.attn_res_block_size == 0:
            self._prepare_attn_res = self._prepare_new_block
            self._combine_attention = self._replace_prefix
        else:
            self._prepare_attn_res = self._prepare_existing_block
            self._combine_attention = self._add_prefix

    def _prepare_first_block(
        self, hidden_states: mx.array, block_residual: mx.array
    ) -> tuple[mx.array, None, mx.array]:
        hidden_size = hidden_states.shape[-1]
        block_residual = mx.concatenate(
            [
                block_residual,
                hidden_states.reshape(-1, hidden_size)[:, None, :],
            ],
            axis=1,
        )
        return hidden_states, None, block_residual

    def _prepare_new_block(
        self, hidden_states: mx.array, block_residual: mx.array
    ) -> tuple[mx.array, None, mx.array]:
        hidden_size = hidden_states.shape[-1]
        prefix_sum = hidden_states
        hidden_states = apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            proj=self.self_attention_res_proj,
            norm=self.self_attention_res_norm,
        ).reshape(hidden_states.shape)
        block_residual = mx.concatenate(
            [
                block_residual,
                prefix_sum.reshape(-1, hidden_size)[:, None, :],
            ],
            axis=1,
        )
        return hidden_states, None, block_residual

    def _prepare_existing_block(
        self, hidden_states: mx.array, block_residual: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        prefix_sum = hidden_states
        hidden_states = apply_attn_res(
            prefix_sum.reshape(-1, hidden_states.shape[-1]),
            block_residual,
            proj=self.self_attention_res_proj,
            norm=self.self_attention_res_norm,
        ).reshape(hidden_states.shape)
        return hidden_states, prefix_sum, block_residual

    @staticmethod
    def _replace_prefix(_prefix_sum: None, attention_output: mx.array) -> mx.array:
        return attention_output

    @staticmethod
    def _add_prefix(prefix_sum: mx.array, attention_output: mx.array) -> mx.array:
        return prefix_sum + attention_output

    def __call__(
        self,
        hidden_states: mx.array,
        block_residual: mx.array,
        cache: Any | None,
    ) -> tuple[mx.array, mx.array]:
        batch, length, hidden_size = hidden_states.shape
        hidden_states, prefix_sum, block_residual = self._prepare_attn_res(
            hidden_states, block_residual
        )
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states), cache=cache
        )
        prefix_sum = self._combine_attention(prefix_sum, attention_output)
        mlp_input = apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            proj=self.mlp_res_proj,
            norm=self.mlp_res_norm,
        ).reshape(batch, length, hidden_size)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_input))
        return prefix_sum + mlp_output, block_residual


class KimiLinearModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiDecoderLayer(args, index) for index in range(args.num_hidden_layers)
        ]
        self.output_attn_res_norm = KimiRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
        self.norm = KimiRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, tokens: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        hidden_states = self.embed_tokens(tokens)
        batch, length, hidden_size = hidden_states.shape
        if cache is None:
            cache = [None] * len(self.layers)
        block_residual = mx.zeros(
            (batch * length, 0, hidden_size), dtype=hidden_states.dtype
        )
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states, block_residual = layer(
                hidden_states, block_residual, layer_cache
            )
        hidden_states = apply_attn_res(
            hidden_states.reshape(-1, hidden_size),
            block_residual,
            proj=self.output_attn_res_proj,
            norm=self.output_attn_res_norm,
        ).reshape(batch, length, hidden_size)
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiLinearModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self) -> List[KimiDecoderLayer]:
        return self.model.layers

    def __call__(self, tokens: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        hidden_states = self.model(tokens, cache)
        return self.lm_head(hidden_states)

    def make_cache(self) -> List[KDACache | MLALatentCache]:
        caches: List[KDACache | MLALatentCache] = []
        for layer in self.layers:
            if layer.is_linear_attn:
                caches.append(KDACache())
            else:
                caches.append(
                    MLALatentCache(self.args.kv_lora_rank, self.args.qk_rope_head_dim)
                )
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map official K3 text names and exact construction-only layouts."""

        mapped: Dict[str, mx.array] = {}
        for original_name, original_value in weights.items():
            name = (
                original_name[len("language_model.") :]
                if original_name.startswith("language_model.")
                else original_name
            )
            name = name.replace(".block_sparse_moe.", ".mlp.")
            value = original_value
            if name.endswith(".self_attn.A_log"):
                value = map_kda_a_log(
                    value, num_heads=self.args.linear_attn_config["num_heads"]
                )
            if name.endswith(
                (
                    ".self_attn.q_conv1d.weight",
                    ".self_attn.k_conv1d.weight",
                    ".self_attn.v_conv1d.weight",
                )
            ):
                if value.ndim != 3 or value.shape[1] != 1:
                    raise ValueError(
                        f"K3 convolution tensor {name} must have [C,1,K] layout"
                    )
                value = mx.contiguous(value.swapaxes(1, 2))
            if name.endswith(
                (
                    ".self_attn.q_conv1d.weight",
                    ".self_attn.k_conv1d.weight",
                    ".self_attn.v_conv1d.weight",
                    ".self_attn.o_norm.weight",
                )
            ):
                value = value.astype(mx.bfloat16)
            if name in mapped:
                raise ValueError(f"duplicate K3 text tensor after mapping: {name}")
            mapped[name] = value
        return mapped

    @property
    def cast_predicate(self):
        def predicate(path: str) -> bool:
            return not (
                path.endswith("A_log")
                or path.endswith("dt_bias")
                or "e_score_correction_bias" in path
            )

        return predicate
