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

Nothing here changes weights or arithmetic order that MLX would not itself
change; each path is expected to be bit-exact against stock, and the A/B bench
verifies that rather than assuming it.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

ENV_FUSED_GATE_UP = "MTPLX_LAGUNA_FUSED_GATE_UP"
ENV_COMPILED_ROUTER = "MTPLX_LAGUNA_COMPILED_ROUTER"
ENV_COMPILED_ATTN_GATE = "MTPLX_LAGUNA_COMPILED_ATTN_GATE"
ENV_KERNEL_ROUTER = "MTPLX_LAGUNA_KERNEL_ROUTER"
ENV_KERNEL_ATTN_GATE = "MTPLX_LAGUNA_KERNEL_ATTN_GATE"


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
        if self.quantized:
            return mx.gather_qmm(
                x,
                self["gate_up_weight"],
                self["gate_up_scales"],
                self.get("gate_up_biases"),
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

    output = self.switch_mlp(flattened, indices)
    output = (output * weights[..., None]).sum(axis=-2)
    output = output + self.shared_expert(flattened)
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
# hand-written Metal kernels
# ---------------------------------------------------------------------------
def _kernel_moe_call(self, x: mx.array) -> mx.array:
    from ..kernels.laguna_decode import fused_router_topk

    batch, length, hidden = x.shape
    flattened = x.reshape(-1, hidden)

    logits = self.gate(flattened).astype(mx.float32)
    if self.softcap and self.softcap > 0.0:
        logits = mx.tanh(logits / self.softcap) * self.softcap

    indices, weights = fused_router_topk(
        logits,
        self.e_score_correction_bias.astype(mx.float32),
        self.top_k,
        normalize=bool(self.norm_topk_prob),
        scale=float(self.routed_scaling_factor),
    )

    output = self.switch_mlp(flattened, indices)
    output = (output * weights.astype(x.dtype)[..., None]).sum(axis=-2)
    output = output + self.shared_expert(flattened)
    return output.reshape(batch, length, hidden)


def install_kernel_router(model: Any) -> dict[str, Any]:
    from .laguna import LagunaSparseMoeBlock

    LagunaSparseMoeBlock.__call__ = _kernel_moe_call
    inner = getattr(model, "model", model)
    count = sum(
        1 for layer in inner.layers if isinstance(layer.mlp, LagunaSparseMoeBlock)
    )
    return {"path": "kernel_router", "layers_affected": count}


def install_kernel_attention_gate(model: Any) -> dict[str, Any]:
    from . import laguna
    from ..kernels.laguna_decode import fused_per_head_gate

    laguna.PER_HEAD_GATE_IMPL = fused_per_head_gate
    inner = getattr(model, "model", model)
    count = sum(1 for layer in inner.layers if layer.self_attn.gating)
    return {"path": "kernel_attn_gate", "layers_affected": count}


# ---------------------------------------------------------------------------
def install_from_env(model: Any) -> list[dict[str, Any]]:
    """Install whichever fused paths the environment asks for."""

    report: list[dict[str, Any]] = []
    if _enabled(ENV_FUSED_GATE_UP):
        report.append(install_fused_gate_up(model))
    if _enabled(ENV_COMPILED_ROUTER):
        report.append(install_compiled_router(model))
    if _enabled(ENV_COMPILED_ATTN_GATE):
        report.append(install_compiled_attention_gate(model))
    # The hand kernels win over the compiled variants where both are asked for.
    if _enabled(ENV_KERNEL_ROUTER):
        report.append(install_kernel_router(model))
    if _enabled(ENV_KERNEL_ATTN_GATE):
        report.append(install_kernel_attention_gate(model))
    return report
