"""Construction-time gate/up projection packing for Qwen MoE blocks.

Qwen's sparse MoE block runs the gate and up projections as two separate
matmuls in both halves of the block::

    routed experts (SwitchGLU): gather_qmm(gate), gather_qmm(up), gather_qmm(down)
    shared expert  (MLP):       qmm(gate),        qmm(up),        qmm(down)

Each pair reads the same activation and differs only in which output rows of
a weight matrix it consumes, so the two matrices can be concatenated along
their output-feature axis once at load time and evaluated as a single matmul
whose result is split in two.

Affine quantization groups run along the *input* axis, so concatenating
output rows leaves every group intact: no requantization happens, and each
output element is still produced by exactly the dot product that produced it
before.  ``tests/test_moe_packed_projections.py`` asserts that bitwise on the
CPU backend for both the plain and the gathered (routed-expert) form.

The win here is dispatch count, not arithmetic.  A 35B-A3B decode forward
touches roughly 1.4 GB of active weights but issues several hundred kernel
launches to do it, which leaves it far below the memory-bandwidth roofline
and makes per-launch and Python graph-construction cost the dominant term.
Packing removes two matmul dispatches -- and the graph nodes that build them
-- from every MoE layer, on both the trunk and the MTP draft block.

Nothing in this module is a custom Metal kernel.  The packed forward is
ordinary MLX ops, so unlike a ``mx.fast.metal_kernel`` lane it still composes
with ``mx.compile`` and the graphbank verify paths.

Interaction with the other optional lanes, since packing changes the *type*
of the two projections it replaces:

- the packed shared expert is no longer a ``Qwen3NextMLP``, so the
  ``MTPLX_MLP_CALL_VARIANT`` lane in :mod:`mtplx.native_mlp` no longer sees
  it (that lane targets the dense-model MLP and is inactive at M=1 anyway);
- the packed projections call ``mx.quantized_matmul``/``mx.gather_qmm``
  directly rather than through ``nn.QuantizedLinear``, so the NAX verify
  patch in :mod:`mtplx.nax_verify` does not route them.

Neither lane is on by default, and the routed-expert ``down_proj`` and the
router itself are untouched either way.

Default off.  Enable with ``MTPLX_QWEN_MOE_PACK_GATE_UP=1``.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn


PACK_GATE_UP_ENV = "MTPLX_QWEN_MOE_PACK_GATE_UP"

_STATS: dict[str, Any] = {
    "enabled": False,
    "packed_switch_mlp": 0,
    "packed_shared_expert": 0,
    "skipped_blocks": 0,
    "skip_reasons": [],
}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def moe_pack_gate_up_enabled() -> bool:
    """Whether construction-time MoE gate/up packing is requested."""

    return _env_enabled(PACK_GATE_UP_ENV)


def moe_packed_projection_stats() -> dict[str, Any]:
    """Snapshot of what the last :func:`configure_moe_packed_projections` did."""

    stats = dict(_STATS)
    stats["skip_reasons"] = list(_STATS["skip_reasons"])
    return stats


class _PackedQuantizedProjection(nn.Module):
    """One affine-quantized projection holding two stacked source matrices."""

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array | None,
        *,
        group_size: int,
        bits: int,
        mode: str,
    ):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = str(mode)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def gather(self, x: mx.array, indices: mx.array, sorted_indices: bool) -> mx.array:
        return mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class _PackedDenseProjection(nn.Module):
    """One unquantized projection holding two stacked source matrices."""

    def __init__(self, weight: mx.array):
        super().__init__()
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self["weight"].swapaxes(-1, -2)

    def gather(self, x: mx.array, indices: mx.array, sorted_indices: bool) -> mx.array:
        return mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )


class PackedSwitchGLU(nn.Module):
    """SwitchGLU with gate/up packed into a single gathered projection.

    Reproduces :class:`mlx_lm.models.switch_layers.SwitchGLU` call for call,
    including the token-sorting path, with the two expert matmuls replaced by
    one over the packed weight.
    """

    def __init__(self, gate_up_proj: nn.Module, down_proj: Any, activation: Any, split_at: int):
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.activation = activation
        self._split_at = int(split_at)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        packed = self.gate_up_proj.gather(x, idx, do_sort)
        x_gate, x_up = mx.split(packed, [self._split_at], axis=-1)

        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class PackedGateUpMLP(nn.Module):
    """Shared-expert MLP with gate/up packed into a single projection.

    Reproduces :class:`mlx_lm.models.qwen3_next.Qwen3NextMLP` with the two
    projection matmuls replaced by one over the packed weight.
    """

    def __init__(self, gate_up_proj: nn.Module, down_proj: Any, split_at: int):
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self._split_at = int(split_at)

    def __call__(self, x: mx.array) -> mx.array:
        from mlx_lm.models.qwen3_next import swiglu

        packed = self.gate_up_proj(x)
        gate, up = mx.split(packed, [self._split_at], axis=-1)
        return self.down_proj(swiglu(gate, up))


def _quant_metadata(module: Any) -> tuple[int, int, str] | None:
    """Return (group_size, bits, mode) when the module is quantized."""

    if "scales" not in module:
        return None
    return (
        int(getattr(module, "group_size", 64)),
        int(getattr(module, "bits", 4)),
        str(getattr(module, "mode", "affine")),
    )


def _pack_pair(gate: Any, up: Any, axis: int) -> tuple[nn.Module, int] | str:
    """Concatenate a gate/up pair along ``axis``, or return a skip reason.

    ``axis`` is the output-feature axis: 0 for a plain ``[out, in]`` linear,
    1 for a switch ``[experts, out, in]`` linear.
    """

    if "bias" in gate or "bias" in up:
        return "projection carries an additive bias"

    gate_quant = _quant_metadata(gate)
    up_quant = _quant_metadata(up)
    if gate_quant != up_quant:
        return "gate and up quantization differ"

    gate_weight = gate["weight"]
    up_weight = up["weight"]
    if gate_weight.ndim != up_weight.ndim or gate_weight.ndim != axis + 2:
        return "unexpected weight rank"
    if gate_weight.shape[axis] != up_weight.shape[axis]:
        return "gate and up output widths differ"
    if gate_weight.shape[:axis] != up_weight.shape[:axis]:
        return "gate and up expert counts differ"
    if gate_weight.shape[axis + 1 :] != up_weight.shape[axis + 1 :]:
        return "gate and up input widths differ"
    if gate_weight.dtype != up_weight.dtype:
        return "gate and up weight dtypes differ"

    split_at = int(gate_weight.shape[axis])

    if gate_quant is None:
        packed = _PackedDenseProjection(mx.concatenate([gate_weight, up_weight], axis=axis))
        mx.eval(packed["weight"])
        return packed, split_at

    group_size, bits, mode = gate_quant
    if gate["scales"].shape != up["scales"].shape:
        return "gate and up scale shapes differ"
    has_gate_bias = gate.get("biases") is not None
    has_up_bias = up.get("biases") is not None
    if has_gate_bias != has_up_bias:
        return "only one of gate/up carries quantization biases"

    weight = mx.concatenate([gate_weight, up_weight], axis=axis)
    scales = mx.concatenate([gate["scales"], up["scales"]], axis=axis)
    biases = None
    if has_gate_bias:
        biases = mx.concatenate([gate["biases"], up["biases"]], axis=axis)

    packed = _PackedQuantizedProjection(
        weight,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    mx.eval(packed["weight"], packed["scales"])
    if biases is not None:
        mx.eval(biases)
    return packed, split_at


def _pack_block(block: Any) -> tuple[int, int, list[str]]:
    """Pack one sparse MoE block in place. Returns (switch, shared, reasons)."""

    switch_packed = 0
    shared_packed = 0
    reasons: list[str] = []

    switch_mlp = getattr(block, "switch_mlp", None)
    if switch_mlp is not None and hasattr(switch_mlp, "gate_proj"):
        result = _pack_pair(switch_mlp.gate_proj, switch_mlp.up_proj, axis=1)
        if isinstance(result, str):
            reasons.append(f"switch_mlp: {result}")
        else:
            packed, split_at = result
            block.switch_mlp = PackedSwitchGLU(
                packed,
                switch_mlp.down_proj,
                switch_mlp.activation,
                split_at,
            )
            switch_packed = 1

    shared = getattr(block, "shared_expert", None)
    if shared is not None and hasattr(shared, "gate_proj"):
        result = _pack_pair(shared.gate_proj, shared.up_proj, axis=0)
        if isinstance(result, str):
            reasons.append(f"shared_expert: {result}")
        else:
            packed, split_at = result
            block.shared_expert = PackedGateUpMLP(packed, shared.down_proj, split_at)
            shared_packed = 1

    return switch_packed, shared_packed, reasons


def configure_moe_packed_projections(model: Any | None = None) -> dict[str, Any]:
    """Pack gate/up projections in every Qwen sparse MoE block of ``model``.

    No-op unless ``MTPLX_QWEN_MOE_PACK_GATE_UP`` is set.  Safe to call on a
    model without MoE blocks, and idempotent: an already-packed block exposes
    no ``gate_proj`` to pack a second time.
    """

    _STATS["enabled"] = moe_pack_gate_up_enabled()
    _STATS["packed_switch_mlp"] = 0
    _STATS["packed_shared_expert"] = 0
    _STATS["skipped_blocks"] = 0
    _STATS["skip_reasons"] = []

    if not _STATS["enabled"] or model is None:
        return moe_packed_projection_stats()

    try:
        from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    except ImportError:
        _STATS["skip_reasons"] = ["mlx_lm Qwen MoE block unavailable"]
        return moe_packed_projection_stats()

    for _, module in model.named_modules():
        if not isinstance(module, Qwen3NextSparseMoeBlock):
            continue
        switch_packed, shared_packed, reasons = _pack_block(module)
        _STATS["packed_switch_mlp"] += switch_packed
        _STATS["packed_shared_expert"] += shared_packed
        if reasons:
            _STATS["skipped_blocks"] += 1
            for reason in reasons:
                if reason not in _STATS["skip_reasons"]:
                    _STATS["skip_reasons"].append(reason)

    return moe_packed_projection_stats()
