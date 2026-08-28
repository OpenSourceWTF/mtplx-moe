"""Construction-installed prefix-state capture for the exact Qwen4 M=2 route."""

from __future__ import annotations

from types import MethodType
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .gdn_capture import (
    _linear_gated_delta_from_conv_headquarter_kernel,
    _stock_conv1d_capture,
)
from .models.qwen4_omlx import create_attention_mask, create_ssm_mask


class Qwen4CaptureConfigError(RuntimeError):
    """The exact two-row Qwen4 capture route cannot be installed."""


def _make_qwen4_norm_gate_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="mtplx_qwen4_m2_sigmoid_norm_gate_bf16",
        input_names=["x", "gate", "weight", "eps"],
        output_names=["y"],
        source=r"""
            constexpr uint AXIS = 128;
            constexpr uint READS = 4;
            uint row = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            uint base = row * AXIS;
            float values[READS];
            float sumsq = 0.0f;
            for (uint i = 0; i < READS; ++i) {
                uint column = lane * READS + i;
                values[i] = float(x[base + column]);
                sumsq += values[i] * values[i];
            }
            sumsq = simd_sum(sumsq);
            float inv_rms = metal::precise::rsqrt(sumsq / float(AXIS) + eps);
            for (uint i = 0; i < READS; ++i) {
                uint column = lane * READS + i;
                InT normed_t = weight[column]
                    * static_cast<InT>(values[i] * inv_rms);
                float gate_f = float(gate[base + column]);
                float sigmoid_y = 1.0f / (1.0f + metal::exp(metal::abs(gate_f)));
                float sigmoid_value = gate_f < 0.0f ? sigmoid_y : 1.0f - sigmoid_y;
                y[base + column] = static_cast<InT>(sigmoid_value * float(normed_t));
            }
        """,
    )


_QWEN4_NORM_GATE_KERNEL = _make_qwen4_norm_gate_kernel()


def _make_qwen4_combine_norm_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="mtplx_qwen4_m2_hyper_combine_norm_bf16",
        input_names=["residual", "value", "inject", "weight", "eps"],
        output_names=["hidden", "normed"],
        source=r"""
            constexpr uint D = 2560;
            constexpr uint HC = 4;
            constexpr uint READS = 10;
            uint unit = thread_position_in_grid.z;
            uint token = unit / HC;
            uint stream = unit - token * HC;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint base = token * HC * D + stream * D;
            InT values[READS];
            float sumsq = 0.0f;
            threadgroup float partial[32];
            threadgroup float inv_rms;
            for (uint i = 0; i < READS; ++i) {
                uint column = tid + i * 256;
                InT product = static_cast<InT>(
                    value[token * D + column] * inject[token * HC + stream]);
                values[i] = static_cast<InT>(residual[base + column] + product);
                sumsq += float(values[i]) * float(values[i]);
                hidden[base + column] = values[i];
            }
            sumsq = simd_sum(sumsq);
            if (simd == 0) partial[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) partial[simd] = sumsq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                float total = simd_sum(partial[lane]);
                if (lane == 0)
                    inv_rms = metal::precise::rsqrt(total / float(D) + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < READS; ++i) {
                uint column = tid + i * 256;
                InT normalized = static_cast<InT>(float(values[i]) * inv_rms);
                normed[base + column] = static_cast<InT>(
                    normalized * weight[stream * D + column]);
            }
        """,
    )


_QWEN4_COMBINE_NORM_KERNEL = _make_qwen4_combine_norm_kernel()


def _qwen4_norm_gate(x: mx.array, gate: mx.array, weight: mx.array, eps: float):
    rows = int(x.shape[0]) * int(x.shape[1]) * 48
    (out,) = _QWEN4_NORM_GATE_KERNEL(
        inputs=[x, mx.contiguous(gate), weight, float(eps)],
        template=[("InT", x.dtype)],
        grid=(32 * rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[tuple(x.shape)],
        output_dtypes=[x.dtype],
    )
    return out


def _qwen4_combine_norm(residual, value, inject, norm):
    batch, rows, _ = residual.shape
    hidden, normed = _QWEN4_COMBINE_NORM_KERNEL(
        inputs=[residual, value, inject, norm.weight, float(norm.eps)],
        template=[("InT", residual.dtype)],
        grid=(256, 1, batch * rows * 4),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(residual.shape), tuple(residual.shape)],
        output_dtypes=[residual.dtype, residual.dtype],
    )
    return hidden, normed


def _qwen4_hyper_from_normed(module, hidden, normed):
    weight = mx.sigmoid(
        module.input_mix_weight_up(nn.silu(module.input_mix_weight_down(normed) / 4))
    )
    weight = weight.reshape(*weight.shape[:-1], 4, 2560)
    mixed = (weight * normed.reshape(*normed.shape[:-1], 4, 2560)).mean(axis=-2)
    inject = 2 * mx.sigmoid(module.block_inject_weight(normed) / 4)
    return mixed, hidden, inject


def is_exact_qwen4_capture_config(config: dict[str, Any]) -> bool:
    """Return whether *config* matches the measured fixed-shape capture lane."""

    text = config.get("text_config")
    return bool(
        config.get("model_type") == "qwen4_exp"
        and isinstance(text, dict)
        and text.get("model_type") == "qwen4_exp_text"
        and int(text.get("hidden_size", -1)) == 2560
        and int(text.get("hc_count", -1)) == 4
        and int(text.get("hc_lowrank", -1)) == 320
        and int(text.get("num_hidden_layers", -1)) == 48
        and int(text.get("linear_num_key_heads", -1)) == 16
        and int(text.get("linear_num_value_heads", -1)) == 48
        and int(text.get("linear_key_head_dim", -1)) == 128
        and int(text.get("linear_value_head_dim", -1)) == 128
        and text.get("output_gate_type") == "sigmoid"
        and float(text.get("rms_norm_eps", -1.0)) == 1e-6
    )


def _gdn_with_capture(gdn: Any, inputs: mx.array, mask: Any, cache: Any):
    batch, rows, _ = inputs.shape
    mixed_qkv, z, b, a = gdn._project_inputs(inputs)
    z = z.reshape(batch, rows, 48, 128)
    conv_state = (
        cache[0]
        if cache is not None and cache[0] is not None
        else mx.zeros((batch, 3, 10240), dtype=inputs.dtype)
    )
    if mask is not None:
        mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
    conv_out, conv_states = _stock_conv1d_capture(mixed_qkv, conv_state, gdn)
    state = (
        cache[1]
        if cache is not None and cache[1] is not None
        else mx.zeros((batch, 48, 128, 128), dtype=mx.float32)
    )
    out, states = _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, gdn.A_log, gdn.dt_bias, state, rows],
        template=[
            ("InT", inputs.dtype),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 48),
            ("KeyDim", 2048),
            ("ConvDim", 10240),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, batch * 48),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, rows, 48, 128), (batch, rows, 48, 128, 128)],
        output_dtypes=[inputs.dtype, mx.float32],
    )
    if cache is not None:
        cache[0] = mx.contiguous(conv_states[:, -1, :, :])
        cache[1] = states[:, -1, :, :, :]
        cache.advance(rows)
    out = _qwen4_norm_gate(out, z, gdn.norm.weight, gdn.norm.eps).reshape(
        batch, rows, -1
    )
    return gdn.out_proj(out), {"conv_states": conv_states, "states": states}


def _qwen4_forward_with_capture(
    model: Any,
    inputs: mx.array,
    *,
    cache: list[Any],
    return_hidden: bool,
):
    text_model = model.language_model
    inner = text_model.model
    hidden = inner.embed_tokens(inputs)
    mask = create_attention_mask(hidden, cache[inner.fa_idx])
    conv_mask = create_ssm_mask(hidden, cache[inner.ssm_idx])

    prev_ctx = None
    if inner.ple_layers:
        context_len = inner.args.ngram_size - 1
        eos = inner.args.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        ple_cache = cache[inner.ple_layers[0]]
        previous = ple_cache[3] if ple_cache is not None else None
        prev_ctx = (
            previous
            if previous is not None
            else mx.full((inputs.shape[0], context_len), eos, inputs.dtype)
        )
        if ple_cache is not None:
            ple_cache[3] = mx.concatenate([prev_ctx, inputs], axis=1)[:, -context_len:]

    hidden = mx.tile(hidden, (1, 1, inner.hc))
    captures: dict[int, dict[str, mx.array]] = {}
    for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        indexer_cache = (
            layer_cache.indexer
            if layer_cache is not None and hasattr(layer_cache, "indexer")
            else None
        )
        if layer.ple is not None:
            hidden = hidden + layer.ple(
                hidden,
                inputs,
                prev_ctx,
                layer_cache,
                conv_mask=conv_mask,
            )

        value, residual, inject = layer.attn_hyper_connection(hidden)
        if layer.is_linear:
            value, capture = _gdn_with_capture(
                layer.linear_attn, value, conv_mask, layer_cache
            )
            captures[layer_index] = capture
        else:
            value = layer.self_attn(
                value,
                inner.rope,
                mask,
                layer_cache,
                indexer_cache,
            )
        hidden, normed = _qwen4_combine_norm(
            residual, value, inject, layer.mlp_hyper_connection.hc_norm
        )
        value, residual, inject = _qwen4_hyper_from_normed(
            layer.mlp_hyper_connection, hidden, normed
        )
        value = layer.mlp(value)
        hidden = residual + (value[..., None, :] * inject[..., None]).reshape(
            *value.shape[:-1], -1
        )

    output = inner.hyper_connection_mixer(hidden)
    logits = text_model._logits(output)
    return (logits, hidden, captures) if return_hidden else (logits, captures)


def _forward_ar_capture(
    self: Any,
    input_ids: mx.array,
    cache=None,
    return_hidden: bool = False,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
):
    del hidden_variant, capture_backend
    if cache is None:
        cache = self.model.make_cache()
    return _qwen4_forward_with_capture(
        self.model,
        input_ids,
        cache=cache,
        return_hidden=return_hidden,
    )


def install_qwen4_capture_route(runtime: Any, *, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed Qwen4 graph once and bind its direct capture method."""

    if not is_exact_qwen4_capture_config(config):
        raise Qwen4CaptureConfigError("capture route requires exact Qwen4 config")
    inner = runtime.model.language_model.model
    layers = tuple(inner.layers)
    linear = tuple(layer for layer in layers if layer.is_linear)
    if len(layers) != 48 or len(linear) != 36:
        raise Qwen4CaptureConfigError("capture route requires 36/48 linear layers")
    expected = (16, 48, 128, 128, 2048, 10240)
    for layer in linear:
        gdn = layer.linear_attn
        observed = (gdn.n_k, gdn.n_v, gdn.dk, gdn.dv, gdn.key_dim, gdn.conv_dim)
        if observed != expected:
            raise Qwen4CaptureConfigError("capture recurrent geometry is invalid")
        norm = gdn.norm
        if (
            tuple(norm.weight.shape) != (128,)
            or float(norm.eps) != 1e-6
            or norm.activation != "sigmoid"
        ):
            raise Qwen4CaptureConfigError("capture norm geometry is invalid")
    for layer in layers:
        hyper = layer.mlp_hyper_connection
        if (
            int(hyper.hc) != 4
            or int(hyper.d) != 2560
            or tuple(hyper.hc_norm.weight.shape) != (10240,)
            or float(hyper.hc_norm.eps) != 1e-6
            or hyper.block_inject_weight is None
        ):
            raise Qwen4CaptureConfigError("capture hyper geometry is invalid")
    if (
        _linear_gated_delta_from_conv_headquarter_kernel is None
        or _QWEN4_NORM_GATE_KERNEL is None
        or _QWEN4_COMBINE_NORM_KERNEL is None
    ):
        raise Qwen4CaptureConfigError("capture Metal kernels are unavailable")
    runtime.forward_ar_capture = MethodType(_forward_ar_capture, runtime)
    return {"installed": True, "linear_layers": 36, "rows": 2}


__all__ = [
    "Qwen4CaptureConfigError",
    "is_exact_qwen4_capture_config",
    "install_qwen4_capture_route",
]
