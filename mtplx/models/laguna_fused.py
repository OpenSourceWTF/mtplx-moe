"""Fused execution paths for Laguna-S-2.1, each independently env-gated.

Laguna has no MTP head, so batch is the only throughput lever and per-step cost
is the whole game.  A decode step is ~48 attention blocks and 47 MoE blocks,
each of which submits a handful of tiny elementwise kernels around two or three
real matmuls; at batch 1 those tiny kernels are pure launch overhead.

Three paths live here, ordered by how certain they are to help:

``MTPLX_LAGUNA_FUSED_GATE_UP``
    Concatenate each expert's gate and up projections along the output
    dimension at load time and issue ONE ``gather_qmm`` instead of two.  Weight
    bytes are unchanged, so this is not a bandwidth play — it halves the launch
    count for the widest op in the block and doubles the output rows per
    launch, which is what occupancy responds to.  Bit-exact by construction:
    quantization groups run along the input dimension, so concatenating output
    rows carries each row's own scales and biases untouched.

``MTPLX_LAGUNA_COMPILED_ROUTER``
    Put the router's elementwise chain (sigmoid, bias add, gather, normalize,
    scale) under ``mx.compile``.  Thirteen ops per MoE block times 47 blocks is
    a lot of dispatch for arithmetic on 256 floats.

``MTPLX_LAGUNA_COMPILED_ATTN_GATE``
    Same treatment for the per-head attention gate: a softplus and a broadcast
    multiply that currently cost four kernels per attention block.

Most of these change no weights and no arithmetic order that MLX would not
itself change, and the A/B bench verifies bit-exactness rather than assuming
it.  ``MTPLX_LAGUNA_KERNEL_ROUTER_GEMV`` is the deliberate exception: folding
the routing matmul into the router kernel reassociates its fp32 dot, so it
belongs to the inexact configuration set and says so at its own installer.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

# Imported at module scope, unlike every other kernel here: this one is called
# once per dense MLP per decode step (47 shared experts plus the layer-0 block),
# and a function-local import would spend a meaningful slice of what it saves on
# `sys.modules` lookups.  `mtplx.kernels` imports nothing from `mtplx.models`,
# so there is no cycle to avoid.
from ..kernels.laguna_decode import fused_glu
from .laguna import LagunaSparseMoeBlock as _LagunaSparseMoeBlock

ENV_FUSED_GATE_UP = "MTPLX_LAGUNA_FUSED_GATE_UP"
ENV_COMPILED_ROUTER = "MTPLX_LAGUNA_COMPILED_ROUTER"
ENV_COMPILED_ATTN_GATE = "MTPLX_LAGUNA_COMPILED_ATTN_GATE"
ENV_KERNEL_ROUTER = "MTPLX_LAGUNA_KERNEL_ROUTER"
ENV_KERNEL_ROUTER_GEMV = "MTPLX_LAGUNA_KERNEL_ROUTER_GEMV"
ENV_KERNEL_ATTN_GATE = "MTPLX_LAGUNA_KERNEL_ATTN_GATE"
ENV_FUSED_RESIDUAL_NORM = "MTPLX_LAGUNA_FUSED_RESIDUAL_NORM"
ENV_KERNEL_QK_ROPE = "MTPLX_LAGUNA_KERNEL_QK_ROPE"
ENV_KERNEL_COMBINE = "MTPLX_LAGUNA_KERNEL_COMBINE"
ENV_FUSED_SHARED_GATE_UP = "MTPLX_LAGUNA_FUSED_SHARED_GATE_UP"
ENV_CACHED_LHS = "MTPLX_LAGUNA_CACHED_LHS"
ENV_FUSED_QKVG = "MTPLX_LAGUNA_FUSED_QKVG"
ENV_FIXED_M2_ROUTER = "MTPLX_LAGUNA_FIXED_M2_ROUTER"


class LagunaFixedM2ConfigError(RuntimeError):
    """The requested fixed Laguna router route failed construction checks."""


# The pristine Attention.__call__, captured on first install so the fused
# variant can delegate any shape it does not cover and benches can restore it.
_STOCK_ATTENTION_CALL = None
# Unlike the older optional router patches, the fixed-M2 route owns an explicit
# stock prefill route.  Capture the shipped implementation at module import so
# installer order cannot accidentally bind another experimental forward.
_STOCK_LAGUNA_MOE_CALL = _LagunaSparseMoeBlock.__call__


def _enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


# Pinned by the bench so a batch sweep stays on one kernel path.  ``None`` keeps
# mlx-lm's stock heuristic, which turns gather-sorting on at indices.size >= 64
# — at top-10 that flips between B=6 and B=7, mid-sweep.
SORT_DECISION: bool | None = None


def should_sort(indices: mx.array) -> bool:
    if SORT_DECISION is None:
        return bool(indices.size >= 64)
    return bool(SORT_DECISION)


# ---------------------------------------------------------------------------
# gate/up fusion
# ---------------------------------------------------------------------------
class FusedGateUpSwitchGLU(nn.Module):
    """SwitchGLU with gate and up issued as a single gather over 2H rows.

    Holds the concatenated tensors directly rather than wrapping them in a
    ``QuantizedSwitchLinear``: that class quantizes a fresh random tensor in its
    constructor, which would cost a 2048x3072x256 allocation per layer for
    values we immediately overwrite.
    """

    def __init__(self, switch_glu: Any) -> None:
        super().__init__()
        gate_proj = switch_glu.gate_proj
        up_proj = switch_glu.up_proj

        self.quantized = "scales" in gate_proj
        self.hidden_dims = int(gate_proj.scales.shape[1]) if self.quantized else int(
            gate_proj.weight.shape[1]
        )
        self.gate_up_weight = mx.concatenate(
            [gate_proj.weight, up_proj.weight], axis=1
        )
        if self.quantized:
            self.gate_up_scales = mx.concatenate(
                [gate_proj.scales, up_proj.scales], axis=1
            )
            gate_biases = gate_proj.get("biases")
            if gate_biases is not None:
                self.gate_up_biases = mx.concatenate(
                    [gate_biases, up_proj["biases"]], axis=1
                )
            self.group_size = int(gate_proj.group_size)
            self.bits = int(gate_proj.bits)
            self.mode = gate_proj.mode

        self.down_proj = switch_glu.down_proj
        self.activation = switch_glu.activation

    def _gate_up(self, x: mx.array, indices: mx.array, sorted_indices: bool):
        lhs_indices = (
            cached_lhs_indices(tuple(x.shape[:-2]))
            if _CACHED_LHS_ACTIVE
            else None
        )
        if self.quantized:
            return mx.gather_qmm(
                x,
                self["gate_up_weight"],
                self["gate_up_scales"],
                self.get("gate_up_biases"),
                lhs_indices=lhs_indices,
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
        return mx.gather_mm(
            x,
            self["gate_up_weight"].swapaxes(-1, -2),
            lhs_indices=lhs_indices,
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        from mlx_lm.models import switch_layers as sl

        x = mx.expand_dims(x, (-2, -3))
        do_sort = should_sort(indices)
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = sl._gather_sort(x, indices)

        fused = self._gate_up(x, idx, do_sort)
        hidden = self.hidden_dims
        x_gate = fused[..., :hidden]
        x_up = fused[..., hidden:]

        x = self.down_proj(
            self.activation(x_up, x_gate), idx, sorted_indices=do_sort
        )
        if do_sort:
            x = sl._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


# ---------------------------------------------------------------------------
# cached gather lhs_indices
# ---------------------------------------------------------------------------
#
# When gather_qmm/gather_mm get no lhs_indices, MLX builds
# `arange(prod(x.shape[:-2])).reshape(x.shape[:-2])` as a GRAPH OP — a real
# kernel dispatch — on every call.  Three gathers per MoE layer put 141 arange
# launches in every decode step.  The default is a pure function of x's shape,
# so passing a cached copy is value-identical and drops the dispatches.

_LHS_CACHE: dict[tuple[int, ...], mx.array] = {}


def cached_lhs_indices(leading_shape: tuple[int, ...]) -> mx.array:
    cached = _LHS_CACHE.get(leading_shape)
    if cached is None:
        total = 1
        for dim in leading_shape:
            total *= int(dim)
        cached = mx.arange(total, dtype=mx.uint32).reshape(leading_shape)
        mx.eval(cached)
        _LHS_CACHE[leading_shape] = cached
    return cached


def _patched_quantized_switch_call(self, x, indices, sorted_indices=False):
    x = mx.gather_qmm(
        x,
        self["weight"],
        self["scales"],
        self.get("biases"),
        lhs_indices=cached_lhs_indices(tuple(x.shape[:-2])),
        rhs_indices=indices,
        transpose=True,
        group_size=self.group_size,
        bits=self.bits,
        mode=self.mode,
        sorted_indices=sorted_indices,
    )
    if "bias" in self:
        x = x + mx.expand_dims(self["bias"][indices], -2)
    return x


def _patched_switch_call(self, x, indices, sorted_indices=False):
    x = mx.gather_mm(
        x,
        self["weight"].swapaxes(-1, -2),
        lhs_indices=cached_lhs_indices(tuple(x.shape[:-2])),
        rhs_indices=indices,
        sorted_indices=sorted_indices,
    )
    if "bias" in self:
        x = x + mx.expand_dims(self["bias"][indices], -2)
    return x


_STOCK_QUANTIZED_SWITCH_CALL = None
_STOCK_SWITCH_CALL = None
_CACHED_LHS_ACTIVE = False


def install_cached_gather_indices(model: Any) -> dict[str, Any]:
    """Feed every expert gather a cached lhs_indices instead of a fresh arange.

    Patches the mlx-lm SwitchLinear call sites process-wide (they are the only
    reachable expert-gather paths) with bodies identical to stock except for
    the explicit, value-identical lhs_indices argument.
    """

    from mlx_lm.models import switch_layers as sl

    global _STOCK_QUANTIZED_SWITCH_CALL, _STOCK_SWITCH_CALL, _CACHED_LHS_ACTIVE
    if _STOCK_QUANTIZED_SWITCH_CALL is None:
        _STOCK_QUANTIZED_SWITCH_CALL = sl.QuantizedSwitchLinear.__call__
        _STOCK_SWITCH_CALL = sl.SwitchLinear.__call__
    sl.QuantizedSwitchLinear.__call__ = _patched_quantized_switch_call
    sl.SwitchLinear.__call__ = _patched_switch_call
    _CACHED_LHS_ACTIVE = True

    inner = getattr(model, "model", model)
    count = sum(
        1
        for layer in inner.layers
        if getattr(layer.mlp, "switch_mlp", None) is not None
    )
    return {"path": "cached_lhs_indices", "moe_layers": count}


def reset_cached_gather_indices() -> None:
    from mlx_lm.models import switch_layers as sl

    global _CACHED_LHS_ACTIVE
    _CACHED_LHS_ACTIVE = False
    if _STOCK_QUANTIZED_SWITCH_CALL is not None:
        sl.QuantizedSwitchLinear.__call__ = _STOCK_QUANTIZED_SWITCH_CALL
        sl.SwitchLinear.__call__ = _STOCK_SWITCH_CALL


class _ArrayBox:
    """Holds an mx.array where nn.Module attribute traversal cannot see it."""

    __slots__ = ("value",)

    def __init__(self, value: mx.array) -> None:
        self.value = value


class FusedGateUpMLP(nn.Module):
    """A dense MLP with gate and up issued as ONE (quantized) matmul.

    Same argument as the expert-bank fusion: quantization groups run along the
    input dimension, so concatenating output rows carries each row's own scales
    and biases untouched — the per-row arithmetic is identical and the fusion
    is bit-exact by construction.  Applies to the 47 shared experts and the
    lone dense layer-0 MLP, each of which currently pays two launches for one
    weight read's worth of work.

    The activation is the stock ``nn.silu(gate) * up``, evaluated by
    :func:`~mtplx.kernels.laguna_decode.fused_glu`, which reads the two halves
    at their strides and writes ONE contiguous activation.  That removes the
    slice materialization and the contiguity copy ``down_proj`` would otherwise
    need, and it is bit-exact: the kernel mirrors MLX's own Metal ``Sigmoid``
    and ``Multiply`` structs, including the bfloat16 rounding of the
    exponential and of ``silu(gate)``.  On any shape or device it does not
    cover — the CPU device, most obviously — ``fused_glu`` evaluates the stock
    expression verbatim instead.  It has no environment variable of its own: it
    is the internals of the path ``MTPLX_LAGUNA_FUSED_SHARED_GATE_UP`` already
    installs, so a run that did not ask for the gate/up concatenation never
    reaches it.
    """

    def __init__(self, mlp: Any) -> None:
        super().__init__()
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj

        self.quantized = "scales" in gate_proj
        if self.quantized:
            if (
                int(gate_proj.group_size) != int(up_proj.group_size)
                or int(gate_proj.bits) != int(up_proj.bits)
                or gate_proj.mode != up_proj.mode
            ):
                raise ValueError(
                    "gate/up quantization differs; concatenation would change "
                    "the arithmetic"
                )
            self.hidden_dims = int(gate_proj.scales.shape[0])
            self.gate_up_scales = mx.concatenate(
                [gate_proj.scales, up_proj.scales], axis=0
            )
            gate_biases = gate_proj.get("biases")
            if gate_biases is not None:
                self.gate_up_biases = mx.concatenate(
                    [gate_biases, up_proj["biases"]], axis=0
                )
            self.group_size = int(gate_proj.group_size)
            self.bits = int(gate_proj.bits)
            self.mode = gate_proj.mode
        else:
            self.hidden_dims = int(gate_proj.weight.shape[0])
        self.gate_up_weight = mx.concatenate(
            [gate_proj.weight, up_proj.weight], axis=0
        )
        self.down_proj = mlp.down_proj

    def __call__(self, x: mx.array) -> mx.array:
        if self.quantized:
            fused = mx.quantized_matmul(
                x,
                self["gate_up_weight"],
                self["gate_up_scales"],
                self.get("gate_up_biases"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            fused = x @ self["gate_up_weight"].swapaxes(-1, -2)
        return self.down_proj(fused_glu(fused, self.hidden_dims))


def install_fused_shared_gate_up(model: Any) -> dict[str, Any]:
    """Fuse gate/up for every dense MLP: shared experts and the layer-0 block.

    Destructive in the same sense as the expert-bank fusion — the originals
    are dropped as each layer converts — but the transient cost is a few MB
    per layer, not 38 GB.
    """

    from .laguna import MLP

    inner = getattr(model, "model", model)
    converted = 0
    for layer in inner.layers:
        mlp = layer.mlp
        if isinstance(mlp, MLP):
            layer.mlp = FusedGateUpMLP(mlp)
            converted += 1
            continue
        shared = getattr(mlp, "shared_expert", None)
        if shared is not None and isinstance(shared, MLP):
            mlp.shared_expert = FusedGateUpMLP(shared)
            converted += 1
    mx.eval(inner.parameters())
    gc.collect()
    mx.clear_cache()
    return {"path": "fused_shared_gate_up", "layers_converted": converted}


def install_fused_gate_up(model: Any) -> dict[str, Any]:
    """Replace every MoE block's SwitchGLU with the fused-gate/up variant.

    Done one layer at a time with the originals dropped immediately, so the
    transient cost is one layer's concatenation rather than a second copy of
    every expert weight in the model.
    """

    inner = getattr(model, "model", model)
    converted = 0
    for layer in inner.layers:
        mlp = layer.mlp
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is None or isinstance(switch_mlp, FusedGateUpSwitchGLU):
            continue
        fused = FusedGateUpSwitchGLU(switch_mlp)
        mx.eval(fused.parameters())
        mlp.switch_mlp = fused
        del switch_mlp
        gc.collect()
        converted += 1
    mx.clear_cache()
    return {"path": "fused_gate_up", "layers_converted": converted}


# ---------------------------------------------------------------------------
# compiled elementwise chains
# ---------------------------------------------------------------------------
@mx.compile
def _router_weights(
    logits: mx.array, correction_bias: mx.array
) -> tuple[mx.array, mx.array]:
    scores = mx.sigmoid(logits)
    return scores, scores + correction_bias


@mx.compile
def _router_normalize(gathered: mx.array, scale: mx.array) -> mx.array:
    return (gathered / gathered.sum(axis=-1, keepdims=True)) * scale


def _fused_moe_call(self, x: mx.array) -> mx.array:
    batch, length, hidden = x.shape
    flattened = x.reshape(-1, hidden)

    logits = self.gate(flattened).astype(mx.float32)
    if self.softcap and self.softcap > 0.0:
        logits = mx.tanh(logits / self.softcap) * self.softcap

    scores, scores_for_choice = _router_weights(
        logits, self.e_score_correction_bias.astype(mx.float32)
    )
    indices = mx.argpartition(
        -scores_for_choice, kth=self.top_k - 1, axis=-1
    )[..., : self.top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if self.norm_topk_prob:
        weights = _router_normalize(
            weights, mx.array(self.routed_scaling_factor, dtype=mx.float32)
        ).astype(x.dtype)
    else:
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)

    from . import laguna

    output = self.switch_mlp(flattened, indices)
    output = laguna.MOE_COMBINE_IMPL(
        output, weights, self.shared_expert(flattened)
    )
    return output.reshape(batch, length, hidden)


def install_compiled_router(model: Any) -> dict[str, Any]:
    from .laguna import LagunaSparseMoeBlock

    LagunaSparseMoeBlock.__call__ = _fused_moe_call
    inner = getattr(model, "model", model)
    count = sum(
        1
        for layer in inner.layers
        if isinstance(layer.mlp, LagunaSparseMoeBlock)
    )
    return {"path": "compiled_router", "layers_affected": count}


@mx.compile
def _per_head_gate(output: mx.array, gate_logits: mx.array) -> mx.array:
    """The shipped expression verbatim, so compilation is the only variable.

    Written out rather than reshaped by the caller: mx.compile traces the whole
    chain, and any restructuring done outside the traced function is a second
    change riding along with the one being measured.
    """

    gate = mx.logaddexp(
        gate_logits.astype(mx.float32), mx.array(0.0)
    ).astype(output.dtype)
    return output * gate[..., None]


def apply_per_head_gate(
    output: mx.array, gate_logits: mx.array, n_heads: int, head_dim: int
) -> mx.array:
    batch, length, _ = output.shape
    gated = _per_head_gate(
        output.reshape(batch, length, n_heads, head_dim), gate_logits
    )
    return gated.reshape(batch, length, -1)


def install_compiled_attention_gate(model: Any) -> dict[str, Any]:
    from . import laguna

    laguna.PER_HEAD_GATE_IMPL = apply_per_head_gate
    inner = getattr(model, "model", model)
    count = sum(1 for layer in inner.layers if layer.self_attn.gating)
    return {"path": "compiled_attn_gate", "layers_affected": count}


# ---------------------------------------------------------------------------
# fused residual + RMSNorm across the layer boundary
# ---------------------------------------------------------------------------
def _fused_residual_forward(self, inputs, cache=None, input_embeddings=None):
    """Run the decoder stack as a residual stream with fused add+RMSNorm.

    The shipped layer does ``h = x + attn(norm(x))`` then ``h + mlp(norm(h))``,
    which is four separate kernels per layer: two adds and two norms, 192
    dispatches across 48 layers for arithmetic that moves 3072 floats.

    Rewritten as a residual stream, every add pairs with the norm that consumes
    its result — including ACROSS the layer boundary, where the mlp residual add
    pairs with the next layer's input norm.  All 96 pairs become 96 fused
    kernels instead of 192 ops.

    The fused kernel keeps MLX's order exactly (the residual add is rounded back
    to the input dtype before the RMS sum), so this is a dispatch change, not a
    numerics change.
    """

    from ..kernels.fused_norm import fused_add_rmsnorm
    from mlx_lm.models.base import create_attention_mask

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
            hidden, cache[self._first_swa], window_size=self.args.sliding_window
        )
    else:
        sliding_mask = full_mask

    rope_memo: dict[int, mx.array] = {}
    layers = self.layers
    first = layers[0]
    normed = mx.fast.rms_norm(
        hidden, first.input_layernorm.weight, first.input_layernorm.eps
    )

    for index, (layer, layer_cache) in enumerate(zip(layers, cache)):
        mask = sliding_mask if layer.self_attn.is_sliding else full_mask
        attention_out = layer.self_attn(normed, mask, layer_cache, rope_memo)
        hidden, normed = fused_add_rmsnorm(
            attention_out,
            hidden,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
        )
        mlp_out = layer.mlp(normed)
        if index + 1 < len(layers):
            following = layers[index + 1]
            hidden, normed = fused_add_rmsnorm(
                mlp_out,
                hidden,
                following.input_layernorm.weight,
                following.input_layernorm.eps,
            )
        else:
            hidden, normed = fused_add_rmsnorm(
                mlp_out, hidden, self.norm.weight, self.norm.eps
            )

    return normed


def install_fused_residual_norm(model: Any) -> dict[str, Any]:
    from ..kernels.fused_norm import is_fused_add_rmsnorm_eligible
    from .laguna import LagunaModel

    inner = getattr(model, "model", model)

    # `fused_add_rmsnorm` silently falls back to stock ops on any shape it does
    # not cover.  Without this probe an ineligible dtype would show up as "the
    # fusion bought nothing" instead of "the fusion never ran".
    first = inner.layers[0]
    weight = first.post_attention_layernorm.weight
    probe = mx.zeros((1, 1, int(weight.shape[0])), dtype=weight.dtype)
    engaged = bool(is_fused_add_rmsnorm_eligible(probe, probe, weight))

    LagunaModel.__call__ = _fused_residual_forward
    return {
        "path": "fused_residual_norm",
        "layers_affected": len(inner.layers),
        "kernel_engaged": engaged,
        "norm_dtype": str(weight.dtype),
    }


# ---------------------------------------------------------------------------
# hand-written Metal kernels
# ---------------------------------------------------------------------------
def _kernel_moe_call(self, x: mx.array) -> mx.array:
    from . import laguna
    from ..kernels.laguna_decode import (
        fused_router_gemv_topk,
        fused_router_topk,
        is_router_gemv_eligible,
    )

    batch, length, hidden = x.shape
    flattened = x.reshape(-1, hidden)

    # The correction bias is a constant; the boxed float32 copy from install
    # time saves one cast dispatch per MoE layer per step.
    bias_box = getattr(self, "_router_bias_f32", None)
    bias_f32 = (
        bias_box.value
        if bias_box is not None
        else self.e_score_correction_bias.astype(mx.float32)
    )
    softcapped = bool(self.softcap and self.softcap > 0.0)

    # `install_kernel_router_gemv` leaves a `_router_gemv_pack` holding this
    # block's router weight; its presence is the switch, and the eligibility
    # check refuses anything the kernel does not cover exactly (CPU runs, a
    # quantized router, a row count past the gate).  When it engages, the
    # `self.gate` matmul is taken over by a kernel that writes float32 directly
    # and the `.astype(mx.float32)` below disappears outright — three dispatches
    # per MoE layer per step become two.  A softcapped
    # block is excluded here as well as at install: the kernel has no softcap
    # and must never be handed a config that needs one.
    pack = getattr(self, "_router_gemv_pack", None)
    if (
        pack is not None
        and not softcapped
        and is_router_gemv_eligible(flattened, pack.weight, bias_f32, self.top_k)
    ):
        indices, weights = fused_router_gemv_topk(
            flattened,
            pack.weight,
            bias_f32,
            self.top_k,
            normalize=bool(self.norm_topk_prob),
            scale=float(self.routed_scaling_factor),
        )
        output = self.switch_mlp(flattened, indices)
        output = laguna.MOE_COMBINE_IMPL(
            output, weights, self.shared_expert(flattened)
        )
        return output.reshape(batch, length, hidden)

    logits = self.gate(flattened).astype(mx.float32)
    if softcapped:
        logits = mx.tanh(logits / self.softcap) * self.softcap

    indices, weights = fused_router_topk(
        logits,
        bias_f32,
        self.top_k,
        normalize=bool(self.norm_topk_prob),
        scale=float(self.routed_scaling_factor),
    )

    output = self.switch_mlp(flattened, indices)
    output = laguna.MOE_COMBINE_IMPL(
        output, weights, self.shared_expert(flattened)
    )
    return output.reshape(batch, length, hidden)


def install_kernel_router(model: Any) -> dict[str, Any]:
    from .laguna import LagunaSparseMoeBlock

    LagunaSparseMoeBlock.__call__ = _kernel_moe_call
    inner = getattr(model, "model", model)
    count = 0
    for layer in inner.layers:
        block = layer.mlp
        if not isinstance(block, LagunaSparseMoeBlock):
            continue
        bias_f32 = block.e_score_correction_bias.astype(mx.float32)
        mx.eval(bias_f32)
        block._router_bias_f32 = _ArrayBox(bias_f32)
        count += 1
    return {"path": "kernel_router", "layers_affected": count}


class _RouterGemvPack:
    """The router's own weight, held where module traversal cannot see it.

    Same trick as :class:`_ArrayBox`: a plain slotted object under an
    underscore-prefixed attribute, so ``Module.valid_parameter_filter`` walks
    straight past it and the weight is not counted twice in ``parameters()``.
    It is a REFERENCE to the router's existing weight, not a copy — nothing is
    duplicated and ``mlp.gate`` keeps working for any layer that falls back.
    """

    __slots__ = ("weight",)

    def __init__(self, weight: mx.array) -> None:
        self.weight = weight


def install_kernel_router_gemv(model: Any) -> dict[str, Any]:
    """Fold the routing matmul into the router kernel.  INEXACT by design.

    Implies the kernel-router path and installs it if it is not already in
    place: the select/normalize/scale epilogue and the boxed float32 correction
    bias both live there, and ``_kernel_moe_call`` is the only forward that
    knows how to read a ``_router_gemv_pack``.

    Per layer this turns three dispatches on the serial spine into two, 47 times
    per decode step: the ``[rows, 3072] x [3072, 256]`` bfloat16 gemv is taken
    over by a kernel that writes float32 logits, so the bf16 -> f32 cast of its
    output stops existing.  What it costs is the GUARANTEE of
    exactness: the in-kernel dot accumulates in a different order than MLX's
    gemv, and a near-tie between two experts can therefore resolve the other
    way.  That is why it is env-gated and belongs to the inexact configuration
    set.  The measured divergence is much smaller than the classification
    suggests — the kernel rounds its dot back to bfloat16 exactly as the stock
    gemv's own output is rounded, and at the real shape nothing has moved yet
    (see the note in ``kernels/laguna_decode.py``) — but "not observed" is not a
    guarantee, and the set a path lives in is decided by the guarantee.

    A layer whose router is quantized, carries a bias, or is not an ``nn.Linear``
    at all cannot be packed and is left on the two-step path and counted; so is
    a softcapped block, because the kernel has no softcap.  The shipped oQ4e
    checkpoint has neither — its routers carry no quantization entry (BF16) and
    ``moe_router_logit_softcapping`` is pinned to 0.0.
    """

    from ..kernels.laguna_decode import is_router_gemv_eligible
    from .laguna import LagunaSparseMoeBlock

    # The epilogue contract lives in the kernel-router forward, so this path
    # requires it.  Installing it again would only rebuild bias boxes that are
    # already correct, so it is skipped when it is already the forward.
    router_report: dict[str, Any] | None = None
    if LagunaSparseMoeBlock.__call__ is not _kernel_moe_call:
        router_report = install_kernel_router(model)

    inner = getattr(model, "model", model)
    packed = 0
    skipped = 0
    reasons: list[str] = []
    weight_dtypes: set[str] = set()
    probe_weight: mx.array | None = None
    for index, layer in enumerate(inner.layers):
        block = layer.mlp
        if not isinstance(block, LagunaSparseMoeBlock):
            continue
        gate = block.gate
        reason = None
        if not isinstance(gate, nn.Linear):
            reason = f"router is {type(gate).__name__}, not an unquantized nn.Linear"
        elif "scales" in gate:
            reason = "router is quantized; the kernel reads dense weights only"
        elif "bias" in gate:
            reason = "router carries a bias; the kernel has no bias term"
        elif block.softcap and block.softcap > 0.0:
            reason = f"router softcap {float(block.softcap)} != 0; the kernel has none"
        if reason is not None:
            block._router_gemv_pack = None
            skipped += 1
            reasons.append(f"layer {index}: {reason}")
            continue
        weight = gate.weight
        weight_dtypes.add(str(weight.dtype))
        if probe_weight is None:
            probe_weight = weight
        block._router_gemv_pack = _RouterGemvPack(weight)
        packed += 1

    # A silent fallback has to be visible, not read as "the fusion bought
    # nothing": probe the real router shape at one row and report whether the
    # kernel would actually engage.
    engaged = False
    if probe_weight is not None:
        dims = int(probe_weight.shape[1])
        probe_x = mx.zeros((1, dims), dtype=probe_weight.dtype)
        probe_bias = mx.zeros((int(probe_weight.shape[0]),), dtype=mx.float32)
        first = next(
            layer.mlp
            for layer in inner.layers
            if isinstance(layer.mlp, LagunaSparseMoeBlock)
        )
        engaged = bool(
            is_router_gemv_eligible(probe_x, probe_weight, probe_bias, first.top_k)
        )

    report: dict[str, Any] = {
        "path": "kernel_router_gemv",
        "layers_packed": packed,
        "layers_skipped": skipped,
        "skip_reasons": reasons,
        "weight_dtypes": sorted(weight_dtypes),
        "kernel_engaged": engaged,
    }
    if router_report is not None:
        report["installed_kernel_router"] = router_report
    return report


# ---------------------------------------------------------------------------
# validated direct M1/M2 router route
# ---------------------------------------------------------------------------
_FIXED_M2_LAYERS = 47
_FIXED_M2_HIDDEN = 3072
_FIXED_M2_EXPERTS = 256
_FIXED_M2_TOP_K = 10
_FIXED_M2_SCALE = 2.5
_FIXED_M2_INTERMEDIATE = 1024

# Exact safetensors header contract at the pinned oQ4e revision
# 8e3f5cad513746264940c1c4195de48d7ea345a5.  Quantized MLX weights are packed
# U32; affine scales and zero-point ``biases`` are BF16.  These are storage
# shapes, not inferred logical shapes.
_FIXED_M2_ROUTED_LAYOUT = {
    "gate_proj": (
        (256, 1024, 384),
        (256, 1024, 24),
    ),
    "up_proj": (
        (256, 1024, 384),
        (256, 1024, 24),
    ),
    "down_proj": (
        (256, 3072, 128),
        (256, 3072, 8),
    ),
}
_FIXED_M2_SHARED_LAYOUT = {
    "gate_proj": ((1024, 768), (1024, 24)),
    "up_proj": ((1024, 768), (1024, 24)),
    "down_proj": ((3072, 256), (3072, 8)),
}


def _fixed_router_topk_direct(
    logits: mx.array,
    correction_bias: mx.array,
    *,
    rows: int,
    _kernel: Any,
) -> tuple[mx.array, mx.array]:
    """Launch the proven Laguna top-k kernel with no runtime eligibility gate."""

    indices, weights = _kernel(
        inputs=[logits, correction_bias, _FIXED_M2_SCALE, True],
        grid=(_FIXED_M2_EXPERTS * rows, 1, 1),
        threadgroup=(_FIXED_M2_EXPERTS, 1, 1),
        output_shapes=[
            (rows, _FIXED_M2_TOP_K),
            (rows, _FIXED_M2_TOP_K),
        ],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, weights


def _fixed_m2_route_direct(
    x: mx.array,
    *,
    project_logits: Callable[[mx.array], mx.array],
    select_topk: Callable[[mx.array], tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array]:
    return select_topk(project_logits(x))


def _fixed_m2_apply_routed(
    x: mx.array,
    routing: tuple[mx.array, mx.array],
    *,
    switch_mlp: Callable[[mx.array, mx.array], mx.array],
    shared_expert: Callable[[mx.array], mx.array],
    combine: Callable[[mx.array, mx.array, mx.array], mx.array],
) -> mx.array:
    batch, length, hidden = x.shape
    flattened = x.reshape(-1, hidden)
    indices, weights = routing
    output = switch_mlp(flattened, indices)
    output = combine(
        output,
        weights,
        shared_expert(flattened),
    )
    return output.reshape(batch, length, hidden)


@dataclass(frozen=True, slots=True)
class _FixedM2RouterPack:
    """Construction-validated arrays and exact bound route entrypoints."""

    weight: mx.array
    correction_bias: mx.array
    route_m1: Callable[[mx.array], tuple[mx.array, mx.array]]
    route_m2: Callable[[mx.array], tuple[mx.array, mx.array]]
    prefill_stock: Callable[[mx.array], mx.array]
    apply_routed_and_shared_experts: Callable[
        [mx.array, tuple[mx.array, mx.array]],
        mx.array,
    ]


def _fixed_m2_moe_call(self, x: mx.array) -> mx.array:
    """Direct installed route; logical M is its only execution-time decision."""

    pack = self._fixed_m2_router_pack
    x2d = x.reshape(-1, int(x.shape[-1]))
    logical_m = int(x2d.shape[0])
    if logical_m == 1:
        routing = pack.route_m1(x2d)
    elif logical_m == 2:
        routing = pack.route_m2(x2d)
    else:
        return pack.prefill_stock(x)
    return pack.apply_routed_and_shared_experts(x, routing)


def _new_fixed_m2_installed_block_type(block_type: type) -> type:
    """Create the layout-compatible direct-call type for one model install."""

    return type(
        f"_FixedM2Installed{block_type.__name__}",
        (block_type,),
        {
            "__module__": __name__,
            "__slots__": (),
            "__call__": _fixed_m2_moe_call,
        },
    )


def _install_fixed_m2_block_route(
    block: Any,
    pack: _FixedM2RouterPack,
    installed_type: type,
) -> None:
    """Mutate one fully validated block; the caller owns transaction rollback."""

    block._fixed_m2_router_pack = pack
    block.__class__ = installed_type


def _fixed_m2_contract_error(
    layer_number: int | None,
    property_name: str,
    actual: Any,
    expected: Any,
) -> LagunaFixedM2ConfigError:
    prefix = "" if layer_number is None else f"layer {layer_number} "
    return LagunaFixedM2ConfigError(
        f"{prefix}{property_name} is {actual!r}; expected {expected!r}"
    )


def _fixed_m2_tensor_shape(value: Any) -> Any:
    if value is None:
        return None
    try:
        shape = value.shape
    except Exception:
        return None
    if shape is None:
        return None
    try:
        return tuple(shape)
    except Exception:
        return shape


def _fixed_m2_property_matches(actual: Any, expected: Any) -> bool:
    try:
        return bool(actual == expected)
    except Exception:
        return False


def _fixed_m2_required_component(
    block: Any,
    component_name: str,
    layer_number: int,
) -> Any:
    component = getattr(block, component_name, None)
    if component is None:
        raise _fixed_m2_contract_error(
            layer_number,
            component_name,
            None,
            "present component",
        )
    return component


def _validate_fixed_tensor(
    owner: Any,
    tensor_name: str,
    *,
    layer_number: int,
    label: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    value = getattr(owner, tensor_name, None)
    property_name = (
        tensor_name.removeprefix("gate_up_")
        if label.endswith("gate_up")
        else tensor_name
    )
    if value is None:
        raise _fixed_m2_contract_error(
            layer_number,
            f"{label} {property_name}",
            None,
            f"non-None {dtype} tensor {shape}",
        )
    actual_shape = _fixed_m2_tensor_shape(value)
    if actual_shape != shape:
        raise _fixed_m2_contract_error(
            layer_number,
            f"{label} {property_name} shape",
            actual_shape,
            shape,
        )
    actual_dtype = getattr(value, "dtype", None)
    if actual_dtype != dtype:
        raise _fixed_m2_contract_error(
            layer_number,
            f"{label} {property_name} dtype",
            actual_dtype,
            dtype,
        )


def _validate_fixed_quantized_projection(
    projection: Any,
    *,
    layer_number: int,
    label: str,
    bits: int,
    weight_shape: tuple[int, ...],
    metadata_shape: tuple[int, ...],
) -> None:
    for property_name, expected in (
        ("bits", bits),
        ("group_size", 128),
        ("mode", "affine"),
    ):
        actual = getattr(projection, property_name, None)
        if actual != expected:
            raise _fixed_m2_contract_error(
                layer_number,
                f"{label} layout {property_name}",
                actual,
                expected,
            )
    if hasattr(projection, "bias"):
        raise _fixed_m2_contract_error(
            layer_number,
            f"{label} output bias",
            "present",
            "absent",
        )
    _validate_fixed_tensor(
        projection,
        "weight",
        layer_number=layer_number,
        label=label,
        shape=weight_shape,
        dtype=mx.uint32,
    )
    for tensor_name in ("scales", "biases"):
        _validate_fixed_tensor(
            projection,
            tensor_name,
            layer_number=layer_number,
            label=label,
            shape=metadata_shape,
            dtype=mx.bfloat16,
        )


def _validate_fixed_fused_projection(
    projection: Any,
    *,
    layer_number: int,
    label: str,
    bits: int,
    weight_shape: tuple[int, ...],
    metadata_shape: tuple[int, ...],
) -> None:
    for property_name, expected in (
        ("quantized", True),
        ("hidden_dims", _FIXED_M2_INTERMEDIATE),
        ("bits", bits),
        ("group_size", 128),
        ("mode", "affine"),
    ):
        actual = getattr(projection, property_name, None)
        if actual != expected:
            raise _fixed_m2_contract_error(
                layer_number,
                f"{label} layout {property_name}",
                actual,
                expected,
            )
    if hasattr(projection, "bias"):
        raise _fixed_m2_contract_error(
            layer_number,
            f"{label} output bias",
            "present",
            "absent",
        )
    _validate_fixed_tensor(
        projection,
        "gate_up_weight",
        layer_number=layer_number,
        label=label,
        shape=weight_shape,
        dtype=mx.uint32,
    )
    for tensor_name in ("gate_up_scales", "gate_up_biases"):
        _validate_fixed_tensor(
            projection,
            tensor_name,
            layer_number=layer_number,
            label=label,
            shape=metadata_shape,
            dtype=mx.bfloat16,
        )


def _validate_fixed_routed_layout(
    routed: Any,
    shared: Any,
    layer_number: int,
) -> None:
    if isinstance(routed, FusedGateUpSwitchGLU):
        _validate_fixed_fused_projection(
            routed,
            layer_number=layer_number,
            label="routed expert gate_up",
            bits=4,
            weight_shape=(256, 2048, 384),
            metadata_shape=(256, 2048, 24),
        )
        _validate_fixed_quantized_projection(
            routed.down_proj,
            layer_number=layer_number,
            label="routed expert down_proj",
            bits=4,
            weight_shape=_FIXED_M2_ROUTED_LAYOUT["down_proj"][0],
            metadata_shape=_FIXED_M2_ROUTED_LAYOUT["down_proj"][1],
        )
    else:
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight_shape, metadata_shape = _FIXED_M2_ROUTED_LAYOUT[name]
            _validate_fixed_quantized_projection(
                getattr(routed, name, None),
                layer_number=layer_number,
                label=f"routed expert {name}",
                bits=4,
                weight_shape=weight_shape,
                metadata_shape=metadata_shape,
            )

    if isinstance(shared, FusedGateUpMLP):
        _validate_fixed_fused_projection(
            shared,
            layer_number=layer_number,
            label="shared expert gate_up",
            bits=8,
            weight_shape=(2048, 768),
            metadata_shape=(2048, 24),
        )
        _validate_fixed_quantized_projection(
            shared.down_proj,
            layer_number=layer_number,
            label="shared expert down_proj",
            bits=8,
            weight_shape=_FIXED_M2_SHARED_LAYOUT["down_proj"][0],
            metadata_shape=_FIXED_M2_SHARED_LAYOUT["down_proj"][1],
        )
    else:
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight_shape, metadata_shape = _FIXED_M2_SHARED_LAYOUT[name]
            _validate_fixed_quantized_projection(
                getattr(shared, name, None),
                layer_number=layer_number,
                label=f"shared expert {name}",
                bits=8,
                weight_shape=weight_shape,
                metadata_shape=metadata_shape,
            )


def _fixed_m2_selfcheck_inputs() -> mx.array:
    """Two deterministic BF16 rows shared by all 47 construction checks."""

    values = mx.arange(2 * _FIXED_M2_HIDDEN, dtype=mx.float32)
    values = ((values % 251) - 125.0) / 128.0
    inputs = values.reshape(2, _FIXED_M2_HIDDEN).astype(mx.bfloat16)
    mx.eval(inputs)
    return inputs


def _fixed_m2_bitwise_array_equal(left: mx.array, right: mx.array) -> bool:
    """Compare tensor representation, including signed zero and NaN payloads."""

    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    return bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)))


def _selfcheck_fixed_m2_block(
    *,
    layer_number: int,
    inputs: mx.array,
    project_m1: Callable[[mx.array], mx.array],
    project_m2: Callable[[mx.array], mx.array],
    select_m1: Callable[[mx.array], tuple[mx.array, mx.array]],
    select_m2: Callable[[mx.array], tuple[mx.array, mx.array]],
) -> None:
    try:
        m1_logits = mx.concatenate(
            [
                project_m1(inputs[0:1]),
                project_m1(inputs[1:2]),
            ],
            axis=0,
        )
        m2_logits = project_m2(inputs)
        mx.eval(m1_logits, m2_logits)
    except Exception as error:
        raise LagunaFixedM2ConfigError(
            f"layer {layer_number} fixed-M2 logits self-check failed: {error}"
        ) from error
    if not _fixed_m2_bitwise_array_equal(m1_logits, m2_logits):
        raise LagunaFixedM2ConfigError(
            f"layer {layer_number} fixed-M2 logits self-check mismatch"
        )

    try:
        m1_indices_0, m1_weights_0 = select_m1(m1_logits[0:1])
        m1_indices_1, m1_weights_1 = select_m1(m1_logits[1:2])
        m1_indices = mx.concatenate([m1_indices_0, m1_indices_1], axis=0)
        m1_weights = mx.concatenate([m1_weights_0, m1_weights_1], axis=0)
        m2_indices, m2_weights = select_m2(m2_logits)
        mx.eval(m1_indices, m1_weights, m2_indices, m2_weights)
    except Exception as error:
        raise LagunaFixedM2ConfigError(
            f"layer {layer_number} fixed-M2 route self-check failed: {error}"
        ) from error
    if not bool(mx.array_equal(m1_indices, m2_indices)):
        raise LagunaFixedM2ConfigError(
            f"layer {layer_number} fixed-M2 expert IDs self-check mismatch"
        )
    if not _fixed_m2_bitwise_array_equal(m1_weights, m2_weights):
        raise LagunaFixedM2ConfigError(
            f"layer {layer_number} fixed-M2 route weights self-check mismatch"
        )


def install_fixed_m2_router(model: Any) -> dict[str, Any]:
    """Validate and atomically install the direct Laguna M1/M2 router routes."""

    from ..kernels import laguna_decode as kernels
    from . import laguna

    LagunaSparseMoeBlock = laguna.LagunaSparseMoeBlock

    inner = getattr(model, "model", model)
    blocks = [
        (index, layer.mlp)
        for index, layer in enumerate(inner.layers)
        if isinstance(layer.mlp, LagunaSparseMoeBlock)
    ]
    if len(blocks) != _FIXED_M2_LAYERS:
        raise _fixed_m2_contract_error(
            None,
            "MoE block count",
            len(blocks),
            _FIXED_M2_LAYERS,
        )

    args = getattr(inner, "args", getattr(model, "args", None))
    for property_name, attribute, expected in (
        ("hidden size", "hidden_size", _FIXED_M2_HIDDEN),
        ("expert count", "num_experts", _FIXED_M2_EXPERTS),
        ("top-k", "num_experts_per_tok", _FIXED_M2_TOP_K),
        (
            "routed intermediate size",
            "moe_intermediate_size",
            _FIXED_M2_INTERMEDIATE,
        ),
        (
            "shared intermediate size",
            "shared_expert_intermediate_size",
            _FIXED_M2_INTERMEDIATE,
        ),
        ("normalization", "norm_topk_prob", True),
        ("scaling", "moe_routed_scaling_factor", _FIXED_M2_SCALE),
        ("softcap", "moe_router_logit_softcapping", 0.0),
    ):
        actual = getattr(args, attribute, None)
        if actual != expected:
            raise _fixed_m2_contract_error(
                None,
                property_name,
                actual,
                expected,
            )

    validated: list[
        tuple[int, Any, mx.array, mx.array, Callable, Callable]
    ] = []
    router_weight_ids: set[int] = set()
    for layer_index, block in blocks:
        layer_number = layer_index
        for property_name, attribute, expected in (
            ("expert count", "num_experts", _FIXED_M2_EXPERTS),
            ("top-k", "top_k", _FIXED_M2_TOP_K),
            ("normalization", "norm_topk_prob", True),
            ("scaling", "routed_scaling_factor", _FIXED_M2_SCALE),
            ("softcap", "softcap", 0.0),
        ):
            actual = getattr(block, attribute, None)
            if not _fixed_m2_property_matches(actual, expected):
                raise _fixed_m2_contract_error(
                    layer_number,
                    property_name,
                    actual,
                    expected,
                )

        gate = _fixed_m2_required_component(block, "gate", layer_number)
        switch_mlp = _fixed_m2_required_component(
            block,
            "switch_mlp",
            layer_number,
        )
        shared_expert = _fixed_m2_required_component(
            block,
            "shared_expert",
            layer_number,
        )
        if hasattr(gate, "scales"):
            raise _fixed_m2_contract_error(
                layer_number,
                "router layout",
                "quantized",
                "dense BF16",
            )
        if hasattr(gate, "bias"):
            raise _fixed_m2_contract_error(
                layer_number,
                "router bias",
                "present",
                "absent",
            )
        weight = getattr(gate, "weight", None)
        weight_shape = _fixed_m2_tensor_shape(weight)
        if weight_shape != (_FIXED_M2_EXPERTS, _FIXED_M2_HIDDEN):
            raise _fixed_m2_contract_error(
                layer_number,
                "router weight shape",
                weight_shape,
                (_FIXED_M2_EXPERTS, _FIXED_M2_HIDDEN),
            )
        weight_dtype = getattr(weight, "dtype", None)
        if weight_dtype != mx.bfloat16:
            raise _fixed_m2_contract_error(
                layer_number,
                "router weight dtype",
                weight_dtype,
                mx.bfloat16,
            )
        if id(weight) in router_weight_ids:
            raise LagunaFixedM2ConfigError(
                f"layer {layer_number} violates 47 distinct router weights"
            )
        router_weight_ids.add(id(weight))

        bias = getattr(block, "e_score_correction_bias", None)
        bias_shape = _fixed_m2_tensor_shape(bias)
        if bias_shape != (_FIXED_M2_EXPERTS,):
            raise _fixed_m2_contract_error(
                layer_number,
                "correction bias shape",
                bias_shape,
                (_FIXED_M2_EXPERTS,),
            )
        bias_dtype = getattr(bias, "dtype", None)
        if bias_dtype != mx.bfloat16:
            raise _fixed_m2_contract_error(
                layer_number,
                "correction bias dtype",
                bias_dtype,
                mx.bfloat16,
            )
        if not isinstance(bias, mx.array):
            raise _fixed_m2_contract_error(
                layer_number,
                "correction bias type",
                type(bias).__name__,
                "mlx.core.array",
            )
        _validate_fixed_routed_layout(
            switch_mlp,
            shared_expert,
            layer_number,
        )
        bias_f32 = bias.astype(mx.float32)
        mx.eval(bias_f32)
        validated.append(
            (
                layer_number,
                block,
                weight,
                bias_f32,
                switch_mlp,
                shared_expert,
            )
        )

    if not kernels._on_metal_device():
        raise LagunaFixedM2ConfigError(
            "Metal/custom kernel availability is false; expected active Metal device"
        )

    try:
        m1_kernel = kernels._router_gemv_logits_kernel(
            _FIXED_M2_EXPERTS,
            _FIXED_M2_HIDDEN,
        )
    except Exception as error:
        raise LagunaFixedM2ConfigError(
            f"Metal/custom M1 projection kernel construction failed: {error}"
        ) from error
    try:
        m2_kernel = kernels._router_gemv_logits_m2_kernel()
    except Exception as error:
        raise LagunaFixedM2ConfigError(
            f"Metal/custom M2 projection kernel construction failed: {error}"
        ) from error
    try:
        topk_kernel = kernels._router_kernel(
            _FIXED_M2_EXPERTS,
            _FIXED_M2_TOP_K,
        )
    except Exception as error:
        raise LagunaFixedM2ConfigError(
            f"Metal/custom top-k kernel construction failed: {error}"
        ) from error

    inputs = _fixed_m2_selfcheck_inputs()
    staged: list[
        tuple[
            int,
            Any,
            mx.array,
            mx.array,
            Callable,
            Callable,
            Callable[[mx.array], tuple[mx.array, mx.array]],
            Callable[[mx.array], tuple[mx.array, mx.array]],
        ]
    ] = []
    for (
        layer_number,
        _block,
        weight,
        bias_f32,
        switch_mlp,
        shared_expert,
    ) in validated:
        project_m1 = partial(
            kernels._router_gemv_logits_m1_direct,
            gate_weight=weight,
            _kernel=m1_kernel,
        )
        project_m2 = partial(
            kernels._router_gemv_logits_m2_prebound,
            gate_weight=weight,
            _kernel=m2_kernel,
        )
        select_m1 = partial(
            _fixed_router_topk_direct,
            correction_bias=bias_f32,
            rows=1,
            _kernel=topk_kernel,
        )
        select_m2 = partial(
            _fixed_router_topk_direct,
            correction_bias=bias_f32,
            rows=2,
            _kernel=topk_kernel,
        )
        _selfcheck_fixed_m2_block(
            layer_number=layer_number,
            inputs=inputs,
            project_m1=project_m1,
            project_m2=project_m2,
            select_m1=select_m1,
            select_m2=select_m2,
        )
        route_m1 = partial(
            _fixed_m2_route_direct,
            project_logits=project_m1,
            select_topk=select_m1,
        )
        route_m2 = partial(
            _fixed_m2_route_direct,
            project_logits=project_m2,
            select_topk=select_m2,
        )
        staged.append(
            (
                layer_number,
                _block,
                weight,
                bias_f32,
                switch_mlp,
                shared_expert,
                route_m1,
                route_m2,
            )
        )

    packs: list[tuple[int, Any, _FixedM2RouterPack]] = []
    for (
        layer_number,
        block,
        weight,
        bias_f32,
        switch_mlp,
        shared_expert,
        route_m1,
        route_m2,
    ) in staged:
        packs.append(
            (
                layer_number,
                block,
                _FixedM2RouterPack(
                    weight=weight,
                    correction_bias=bias_f32,
                    route_m1=route_m1,
                    route_m2=route_m2,
                    prefill_stock=partial(_STOCK_LAGUNA_MOE_CALL, block),
                    apply_routed_and_shared_experts=partial(
                        _fixed_m2_apply_routed,
                        switch_mlp=switch_mlp,
                        shared_expert=shared_expert,
                        combine=laguna.MOE_COMBINE_IMPL,
                    ),
                ),
            )
        )

    block_types = {
        type(block) for _layer_number, block, _pack in packs
    }
    installed_types = {
        block_type: _new_fixed_m2_installed_block_type(block_type)
        for block_type in block_types
    }
    missing_pack = object()
    changed: list[tuple[Any, type, Any]] = []
    for layer_number, block, pack in packs:
        previous_pack = getattr(
            block,
            "_fixed_m2_router_pack",
            missing_pack,
        )
        changed.append((block, type(block), previous_pack))
        try:
            _install_fixed_m2_block_route(
                block,
                pack,
                installed_types[type(block)],
            )
        except Exception as error:
            rollback_errors = []
            for changed_block, previous_type, old_pack in reversed(changed):
                try:
                    changed_block.__class__ = previous_type
                except Exception as rollback_error:
                    rollback_errors.append(str(rollback_error))
                try:
                    if old_pack is missing_pack:
                        if hasattr(changed_block, "_fixed_m2_router_pack"):
                            delattr(changed_block, "_fixed_m2_router_pack")
                    else:
                        changed_block._fixed_m2_router_pack = old_pack
                except Exception as rollback_error:
                    rollback_errors.append(str(rollback_error))
            detail = ""
            if rollback_errors:
                detail = f"; rollback failed: {rollback_errors!r}"
            raise LagunaFixedM2ConfigError(
                f"layer {layer_number} fixed-M2 route installation failed: "
                f"{error}{detail}"
            ) from error
    return {
        "path": "fixed_m2_router",
        "layers_validated": _FIXED_M2_LAYERS,
        "layers_selfchecked": _FIXED_M2_LAYERS,
    }


def install_kernel_attention_gate(model: Any) -> dict[str, Any]:
    from . import laguna
    from ..kernels.laguna_decode import fused_per_head_gate

    laguna.PER_HEAD_GATE_IMPL = fused_per_head_gate
    inner = getattr(model, "model", model)
    count = sum(1 for layer in inner.layers if layer.self_attn.gating)
    return {"path": "kernel_attn_gate", "layers_affected": count}


# ---------------------------------------------------------------------------
# fused q/k/v/gate projection
# ---------------------------------------------------------------------------
class FusedQkvgProj(nn.Module):
    """q, k, v and the attention gate issued as ONE (quantized) matmul.

    All four projections read the SAME input — the post-``input_layernorm``
    hidden state — so their weights stack along the output dimension and apply
    in a single pass.  The transform is bit-exact by the same argument the
    gate/up fusions rest on: quantization groups run along the INPUT dimension,
    so concatenating output ROWS carries each row's own scales and biases
    untouched, and every output element is still the same products summed in
    the same order.  Nothing about a row's arithmetic can see which other rows
    it was issued alongside.

    The motivation is the serial link rather than bandwidth.  At B=1 these are
    four separate dispatches at the head of every attention block — 192 across
    48 layers, each one encoded, submitted and drained in turn — and each pays a
    launch floor that is now most of what a decode step costs.  Bytes read are
    unchanged; the launch count drops to one and the rows per launch quadruple,
    which is what occupancy responds to.

    The four results come back as SLICES of the fused output rather than as
    separately allocated buffers.  At the B=1/T=1 decode shape those slices are
    row-contiguous in MLX's own sense (its contiguity check excuses dimensions
    of size 1), so they are views and nothing is copied.  At B > 1 they are
    genuinely strided and MLX materializes them wherever a consumer needs them
    contiguous — the regime this path is not built for, and part of why it is
    env-gated rather than default.
    """

    _NAMES = ("q_proj", "k_proj", "v_proj", "g_proj")

    def __init__(self, attention: Any) -> None:
        super().__init__()

        self.gating = bool(getattr(attention, "gating", False))
        names = self._NAMES if self.gating else self._NAMES[:3]
        parts = [getattr(attention, name) for name in names]

        for name, part in zip(names, parts):
            if "bias" in part:
                # A per-output bias would concatenate cleanly too, but the
                # admitted checkpoint sets attention_bias=False everywhere, so
                # an unexpected bias means this is not the module tree the
                # fusion was reasoned about.
                raise ValueError(f"{name} carries a bias; refusing to concatenate")

        quantized = [("scales" in part) for part in parts]
        if any(quantized) and not all(quantized):
            raise ValueError(
                "q/k/v/g are not uniformly quantized; concatenation would need "
                "two different matmuls"
            )
        self.quantized = bool(quantized[0])

        reference = parts[0]
        if self.quantized:
            for name, part in zip(names[1:], parts[1:]):
                if (
                    int(part.group_size) != int(reference.group_size)
                    or int(part.bits) != int(reference.bits)
                    or part.mode != reference.mode
                ):
                    raise ValueError(
                        f"{name} is {int(part.bits)}-bit/gs{int(part.group_size)}"
                        f"/{part.mode} but q_proj is {int(reference.bits)}-bit"
                        f"/gs{int(reference.group_size)}/{reference.mode}; "
                        "concatenating would change the arithmetic"
                    )
            has_biases = [(part.get("biases") is not None) for part in parts]
            if any(has_biases) and not all(has_biases):
                raise ValueError(
                    "q/k/v/g disagree on whether the quantization carries "
                    "biases; concatenation would be ill-defined"
                )
            self.group_size = int(reference.group_size)
            self.bits = int(reference.bits)
            self.mode = reference.mode

        widths = {int(part.weight.shape[1]) for part in parts}
        if len(widths) != 1:
            raise ValueError(
                f"q/k/v/g do not share an input width ({sorted(widths)}); they "
                "cannot be reading the same hidden state"
            )

        # Split points, as the cumulative row counts of the concatenation.  Held
        # as plain ints so `__call__` slices with no per-step arithmetic, and so
        # module traversal never sees them as parameters.
        rows = [
            int(part.scales.shape[0]) if self.quantized else int(part.weight.shape[0])
            for part in parts
        ]
        self.q_end = rows[0]
        self.k_end = self.q_end + rows[1]
        self.v_end = self.k_end + rows[2]

        self.qkvg_weight = mx.concatenate([part.weight for part in parts], axis=0)
        if self.quantized:
            self.qkvg_scales = mx.concatenate([part.scales for part in parts], axis=0)
            if reference.get("biases") is not None:
                self.qkvg_biases = mx.concatenate(
                    [part["biases"] for part in parts], axis=0
                )

    def __call__(self, x: mx.array):
        """Return ``(queries, keys, values, gate_logits)`` from one matmul.

        ``gate_logits`` is None for a non-gating layer, which is what lets the
        callers keep the shipped ``if self.gating`` shape verbatim.
        """

        if self.quantized:
            fused = mx.quantized_matmul(
                x,
                self["qkvg_weight"],
                self["qkvg_scales"],
                self.get("qkvg_biases"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            fused = x @ self["qkvg_weight"].swapaxes(-1, -2)

        queries = fused[..., : self.q_end]
        keys = fused[..., self.q_end : self.k_end]
        values = fused[..., self.k_end : self.v_end]
        gate_logits = fused[..., self.v_end :] if self.gating else None
        return queries, keys, values, gate_logits


def _capture_stock_attention_call():
    """Pin the pristine ``Attention.__call__`` before anything patches it.

    Both attention installers call this before their own swap and it only ever
    fires once, so whichever runs first records the shipped implementation and
    the second cannot record the first's patch.  That is what makes the two
    installs compose in either order.
    """

    from .laguna import Attention

    global _STOCK_ATTENTION_CALL
    if _STOCK_ATTENTION_CALL is None:
        _STOCK_ATTENTION_CALL = Attention.__call__
    return _STOCK_ATTENTION_CALL


def _fused_qkvg_attention_call(self, x, mask=None, cache=None, rope_memo=None):
    """``Attention.__call__`` verbatim, reading the fused projection.

    Only the first line differs from the shipped forward: the four projections
    arrive as slices of one matmul instead of four separate calls, and the gate
    logits are already in hand by the time the gating branch needs them.  Every
    other op — the norms, the transposes, the rope offset vector, the gate
    expression — is the module code unchanged, so this path is bit-exact
    against stock and stays that way if the shipped forward is edited only in
    the ways that also have to be mirrored here.
    """

    from . import laguna

    batch, length, _ = x.shape
    queries, keys, values, gate_logits = self._qkvg(x)

    queries = self.q_norm(
        queries.reshape(batch, length, self.n_heads, -1)
    ).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(batch, length, self.n_kv_heads, -1)).transpose(
        0, 2, 1, 3
    )
    values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    offset = laguna._rope_offset(
        cache.offset if cache is not None else 0, batch, rope_memo
    )
    queries = self.rope(queries, offset=offset)
    keys = self.rope(keys, offset=offset)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    from mlx_lm.models.base import scaled_dot_product_attention

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
        if self.gate_per_head:
            output = laguna.PER_HEAD_GATE_IMPL(
                output, gate_logits, self.n_heads, self.head_dim
            )
        else:
            gate = mx.logaddexp(
                gate_logits.astype(mx.float32), mx.array(0.0)
            ).astype(output.dtype)
            output = output * gate

    return self.o_proj(output)


def _qkvg_attention_dispatch(self, x, mask=None, cache=None, rope_memo=None):
    """Send each layer to the forward its own modules can still support.

    The install is per layer and is allowed to skip one (mixed quantization),
    so the dispatch is per layer too: a converted layer no longer HAS q_proj
    and must take the fused path, while a skipped layer kept its four modules
    and runs the shipped code untouched.
    """

    if getattr(self, "_qkvg", None) is not None:
        return _fused_qkvg_attention_call(self, x, mask, cache, rope_memo)
    return _STOCK_ATTENTION_CALL(self, x, mask, cache, rope_memo)


def install_fused_qkvg(model: Any) -> dict[str, Any]:
    """Fuse q/k/v/g into one projection per attention block.  DESTRUCTIVE.

    Each converted layer's ``q_proj``, ``k_proj``, ``v_proj`` and ``g_proj``
    are DROPPED once their rows are concatenated, so the layer holds one copy
    of the weights rather than two.  After this install the only forwards that
    still work on a converted layer are the ones that know about ``_qkvg``:

    * ``_qkvg_attention_dispatch`` / ``_fused_qkvg_attention_call`` (swapped in
      here, unless the qk-rope kernel path is already installed);
    * ``_kernel_attention_call``, which reads ``_qkvg`` when it is present;
    * ``mtplx.laguna_compiled_step.build_step``, likewise.

    Anything else that reaches for ``attention.q_proj`` — the pristine
    ``Attention.__call__``, weight export, a self-check that walks the module
    tree — will raise on a converted layer.  Like the gate/up concatenations
    this is a one-way conversion: undoing it means reloading the model, so a
    bench must run this arm LAST.

    Composition with ``install_kernel_qk_rope`` is explicit and order-free.
    Both installers capture the pristine forward through
    :func:`_capture_stock_attention_call`, which fires once; the qk-rope path
    supersedes the plain dispatcher (it handles ``_qkvg`` itself and falls back
    THROUGH the dispatcher for shapes its kernel does not cover), so whichever
    order they are installed in, both installed ends at
    ``_kernel_attention_call`` and qkvg-only ends at the dispatcher.

    A layer whose q/k/v/g do not share one (bits, group_size, mode) cannot be
    concatenated without changing the arithmetic; it is left entirely stock and
    counted in the report.  The shipped oQ4e checkpoint has no such layer — its
    per-layer table gives q, k, v and g the same width on all 48 layers (layer
    33 differs only in ``o_proj``, which is not part of this concatenation).
    """

    from .laguna import Attention

    _capture_stock_attention_call()

    inner = getattr(model, "model", model)
    converted = 0
    skipped = 0
    reasons: list[str] = []
    for index, layer in enumerate(inner.layers):
        attention = layer.self_attn
        if getattr(attention, "_qkvg", None) is not None:
            continue
        try:
            fused = FusedQkvgProj(attention)
        except ValueError as error:
            skipped += 1
            reasons.append(f"layer {index}: {error}")
            continue
        # Evaluated HERE rather than left to a later `mx.eval(model.parameters())`:
        # `Module.valid_parameter_filter` drops keys that start with an
        # underscore, so `_qkvg` is in the module tree but out of
        # `parameters()`.  Materializing now is also what frees the originals —
        # an unevaluated concatenation still holds them alive.
        mx.eval(fused.parameters())
        attention._qkvg = fused
        for name in FusedQkvgProj._NAMES:
            if name in attention:
                del attention[name]
        gc.collect()
        converted += 1

    # Never downgrade the kernel path: it already handles `_qkvg` and covers
    # strictly more than the dispatcher does.
    if Attention.__call__ is not _kernel_attention_call:
        Attention.__call__ = _qkvg_attention_dispatch
    mx.clear_cache()
    return {
        "path": "fused_qkvg",
        "layers_converted": converted,
        "layers_skipped": skipped,
        "skip_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# fused q/k norm + rope
# ---------------------------------------------------------------------------
def _kernel_attention_call(self, x, mask=None, cache=None, rope_memo=None):
    """Attention forward that fuses norm+transpose+rope at the decode step.

    Any shape the kernel does not cover — prefill, CPU runs, a rope module the
    installer did not recognize — delegates wholesale to
    :func:`_qkvg_attention_dispatch`, which is the pristine implementation
    captured at install time unless the destructive q/k/v/g fusion has taken
    this layer's projections away, in which case it is that fusion's mirror of
    it.  Delegating THROUGH the dispatcher rather than straight to the pristine
    call is what lets the two installs compose in either order.
    """

    from ..kernels.laguna_decode import (
        fused_qk_norm_rope,
        is_qk_norm_rope_eligible,
    )
    from mlx_lm.models.base import scaled_dot_product_attention

    spec = getattr(self, "_qk_rope_spec", None)
    batch, length, _ = x.shape
    if spec is None or length != 1:
        return _qkvg_attention_dispatch(self, x, mask, cache, rope_memo)
    offset = cache.offset if cache is not None else 0
    if isinstance(offset, mx.array) and offset.size != 1:
        # Ragged serving owns one logical offset per batch row.  The fixed
        # qk+rope kernel accepts one scalar position, while the installed qkvg
        # dispatcher preserves the vector and uses MLX's general per-row rope
        # path.  Route before projection so this explicit runtime phase choice
        # never computes qkvg twice.
        return _qkvg_attention_dispatch(self, x, mask, cache, rope_memo)

    # One matmul for all four projections when the fusion is installed; the
    # gate logits then come for free instead of costing a fifth dispatch below.
    qkvg = getattr(self, "_qkvg", None)
    if qkvg is not None:
        queries, keys, values, gate_logits = qkvg(x)
    else:
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        gate_logits = None
    if not is_qk_norm_rope_eligible(
        queries, keys, self.q_norm.weight, self.k_norm.weight, spec
    ):
        return _qkvg_attention_dispatch(self, x, mask, cache, rope_memo)

    queries, keys = fused_qk_norm_rope(
        queries,
        keys,
        self.q_norm.weight,
        self.k_norm.weight,
        float(self.q_norm.eps),
        int(offset),
        spec,
    )
    values = values.reshape(batch, 1, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        from . import laguna

        if gate_logits is None:
            gate_logits = self.g_proj(x)
        if self.gate_per_head:
            output = laguna.PER_HEAD_GATE_IMPL(
                output, gate_logits, self.n_heads, self.head_dim
            )
        else:
            gate = mx.logaddexp(
                gate_logits.astype(mx.float32), mx.array(0.0)
            ).astype(output.dtype)
            output = output * gate

    return self.o_proj(output)


def _qk_rope_spec_for(attention: Any) -> "Any | None":
    """Build the fused-kernel spec from a layer's own rope module.

    Returns None for any rope this install does not recognize EXACTLY; those
    layers keep the stock path via the per-call eligibility check.
    """

    import mlx.nn as mlx_nn
    from mlx_lm.models.rope_utils import YarnRoPE

    from ..kernels.laguna_decode import QkRopeSpec

    rope = attention.rope
    head_dim = int(attention.head_dim)
    if isinstance(rope, YarnRoPE):
        if rope.traditional:
            return None
        freqs = rope._freqs
        if freqs.dtype != mx.float32:
            return None
        return QkRopeSpec(
            n_q_heads=int(attention.n_heads),
            n_kv_heads=int(attention.n_kv_heads),
            head_dim=head_dim,
            rot_dims=int(rope.dims),
            freqs=freqs,
            base_log2=None,
            mscale=float(rope.mscale) if rope.mscale != 1.0 else None,
        )
    if isinstance(rope, mlx_nn.RoPE):
        # nn.RoPE computes inv_freq from the base; the host passes log2(base).
        if rope.traditional or float(rope.scale) != 1.0:
            return None
        import math

        return QkRopeSpec(
            n_q_heads=int(attention.n_heads),
            n_kv_heads=int(attention.n_kv_heads),
            head_dim=head_dim,
            rot_dims=int(rope.dims),
            freqs=None,
            base_log2=float(math.log2(float(rope.base))),
            mscale=None,
        )
    return None


def install_kernel_qk_rope(model: Any) -> dict[str, Any]:
    """Swap in the fused norm+rope forward.  Supersedes the q/k/v/g dispatcher.

    ``_kernel_attention_call`` reads ``_qkvg`` itself and falls back through
    ``_qkvg_attention_dispatch``, so it covers everything the dispatcher does
    and installing it second is an upgrade, not a clobber.
    """

    from .laguna import Attention

    _capture_stock_attention_call()

    inner = getattr(model, "model", model)
    covered = 0
    skipped = 0
    for layer in inner.layers:
        attention = layer.self_attn
        spec = _qk_rope_spec_for(attention)
        attention._qk_rope_spec = spec
        if spec is not None and spec.head_dim == 128 and spec.rot_dims in (64, 128):
            covered += 1
        else:
            skipped += 1

    Attention.__call__ = _kernel_attention_call
    return {
        "path": "kernel_qk_rope",
        "layers_covered": covered,
        "layers_skipped": skipped,
    }


def install_kernel_moe_combine(model: Any) -> dict[str, Any]:
    from . import laguna
    from ..kernels.laguna_decode import fused_moe_combine

    laguna.MOE_COMBINE_IMPL = fused_moe_combine
    inner = getattr(model, "model", model)
    count = sum(
        1 for layer in inner.layers if hasattr(layer.mlp, "switch_mlp")
    )
    return {"path": "kernel_moe_combine", "layers_affected": count}


# ---------------------------------------------------------------------------
def install_from_env(model: Any) -> list[dict[str, Any]]:
    """Install whichever fused paths the environment asks for."""

    report: list[dict[str, Any]] = []
    fixed_m2_requested = _enabled(ENV_FIXED_M2_ROUTER)
    if _enabled(ENV_FUSED_GATE_UP):
        report.append(install_fused_gate_up(model))
    if _enabled(ENV_COMPILED_ROUTER):
        report.append(install_compiled_router(model))
    if _enabled(ENV_COMPILED_ATTN_GATE):
        report.append(install_compiled_attention_gate(model))
    # The hand kernels win over the compiled variants where both are asked for.
    if _enabled(ENV_KERNEL_ROUTER):
        report.append(install_kernel_router(model))
    # Implies the kernel router and installs it itself, so it composes whether
    # or not ENV_KERNEL_ROUTER was also asked for.
    if _enabled(ENV_KERNEL_ROUTER_GEMV):
        report.append(install_kernel_router_gemv(model))
    if _enabled(ENV_KERNEL_ATTN_GATE):
        report.append(install_kernel_attention_gate(model))
    if _enabled(ENV_FUSED_RESIDUAL_NORM):
        report.append(install_fused_residual_norm(model))
    if _enabled(ENV_KERNEL_QK_ROPE):
        report.append(install_kernel_qk_rope(model))
    if _enabled(ENV_KERNEL_COMBINE):
        report.append(install_kernel_moe_combine(model))
    if _enabled(ENV_FUSED_SHARED_GATE_UP):
        report.append(install_fused_shared_gate_up(model))
    if _enabled(ENV_CACHED_LHS):
        report.append(install_cached_gather_indices(model))
    # Last, and after ENV_KERNEL_QK_ROPE by construction: the q/k/v/g fusion
    # drops the projections it concatenates, so every path that still wants to
    # read them has to have run first.  `install_fused_qkvg` will not downgrade
    # the qk-rope forward if that one is already in place.
    if _enabled(ENV_FUSED_QKVG):
        report.append(install_fused_qkvg(model))
    # Last: this route validates the final installed expert layout and must own
    # the MoE call after any older router experiment asked for in the same env.
    # A requested failure raises; it never becomes a skip report.
    if fixed_m2_requested:
        report.append(install_fixed_m2_router(model))
    return report
