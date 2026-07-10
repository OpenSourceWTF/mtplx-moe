"""MLX execution adapters for slot-backed affine-Q4 routed experts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.activations import swiglu

from mtplx.expert_runtime import ExpertStreamingRuntime
from mtplx.expert_slots import ExpertSlotBinding
from mtplx.expert_streaming import RoutingPhase
from mtplx.expert_streaming_models import ExpertMemoryPlan, ExpertStreamingModelSpec


_ROUTING_PHASE: ContextVar[RoutingPhase | None] = ContextVar(
    "mtplx_expert_routing_phase",
    default=None,
)


@contextmanager
def expert_routing_phase(phase: RoutingPhase | str) -> Iterator[None]:
    token = _ROUTING_PHASE.set(RoutingPhase(phase))
    try:
        yield
    finally:
        _ROUTING_PHASE.reset(token)


def current_expert_routing_phase(*, token_count: int) -> RoutingPhase:
    explicit = _ROUTING_PHASE.get()
    if explicit is not None:
        return explicit
    return RoutingPhase.PREFILL if token_count > 1 else RoutingPhase.DECODE


class UnboundExpertSwitch(nn.Module):
    """Parameter-free placeholder installed before resident-only loading."""

    def __init__(self, layer_index: int):
        super().__init__()
        self.layer_index = int(layer_index)

    def __call__(self, _x: mx.array, _indices: mx.array) -> mx.array:
        raise RuntimeError(
            f"streamed expert layer {self.layer_index} has no bound runtime"
        )


def _component_array(binding: ExpertSlotBinding, component: str) -> mx.array:
    segment = None
    offset = 0
    for candidate in binding.record.segments:
        if candidate.component == component:
            segment = candidate
            break
        offset += candidate.length
    if segment is None:
        raise KeyError(component)
    if isinstance(binding.buffer, mx.array):
        raw = binding.buffer[offset : offset + segment.length]
        if segment.dtype == "U32":
            return raw.view(mx.uint32).reshape(segment.shape)
        if segment.dtype == "BF16":
            return raw.view(mx.bfloat16).reshape(segment.shape)
        raise TypeError(f"unsupported streamed component dtype {segment.dtype}")
    view = binding.component_view(component)
    if segment.dtype == "U32":
        host = np.frombuffer(view, dtype=np.dtype("<u4")).reshape(segment.shape)
        value = mx.array(host)
    elif segment.dtype == "BF16":
        host = np.frombuffer(view, dtype=np.dtype("<u2")).reshape(segment.shape)
        value = mx.array(host).view(mx.bfloat16)
    else:
        raise TypeError(f"unsupported streamed component dtype {segment.dtype}")
    return value


def mlx_slot_buffer_allocator(size: int, _label: str) -> mx.array:
    """Allocate one stable writable MLX/Metal byte buffer for direct ``pread``."""

    value = mx.zeros((int(size),), dtype=mx.uint8)
    mx.eval(value)
    view = memoryview(value)
    if view.readonly or not view.c_contiguous or view.nbytes != int(size):
        raise RuntimeError("MLX slot buffer is not writable contiguous shared memory")
    view.release()
    return value


def make_mlx_slot_buffer_allocator(
    plan: ExpertMemoryPlan,
    spec: ExpertStreamingModelSpec,
) -> Callable[[int, str], mx.array]:
    """Create banked stable slot views backed by a small number of MTLBuffers."""

    banks: dict[str, mx.array] = {}
    backend = "mlx-metal-slot-bank"

    def make_bank(count: int, size: int) -> mx.array:
        bank = mx.zeros((count, size), dtype=mx.uint8)
        mx.eval(bank)
        return bank

    def allocate(size: int, label: str) -> mx.array:
        if size != spec.expert_record_bytes:
            raise ValueError("slot allocator size differs from the model descriptor")
        parts = label.split("-")
        if label.startswith("layer-") and "-persistent-" in label:
            layer = int(parts[1])
            slot = int(parts[-1])
            key = f"layer-{layer}"
            count = plan.slots_per_layer
        elif label.startswith("global-transient-"):
            slot = int(parts[-1])
            key = "transient"
            count = plan.transient_slots
        else:
            raise ValueError(f"unknown expert slot label {label!r}")
        if count <= 0:
            raise ValueError(f"slot bank {key} has no planned capacity")
        bank = banks.get(key)
        if bank is None:
            bank = make_bank(count, size)
            banks[key] = bank
        value = bank[slot]
        mx.eval(value)
        return value

    setattr(allocate, "backend", backend)
    setattr(allocate, "banks", banks)
    return allocate


def _run_q4_expert(
    x: mx.array, binding: ExpertSlotBinding, *, group_size: int
) -> mx.array:
    gate_weight = _component_array(binding, "gate_proj.weight")
    gate_scales = _component_array(binding, "gate_proj.scales")
    gate_biases = _component_array(binding, "gate_proj.biases")
    up_weight = _component_array(binding, "up_proj.weight")
    up_scales = _component_array(binding, "up_proj.scales")
    up_biases = _component_array(binding, "up_proj.biases")
    down_weight = _component_array(binding, "down_proj.weight")
    down_scales = _component_array(binding, "down_proj.scales")
    down_biases = _component_array(binding, "down_proj.biases")

    gate = mx.quantized_matmul(
        x,
        gate_weight,
        scales=gate_scales,
        biases=gate_biases,
        group_size=group_size,
        bits=4,
        mode="affine",
    )
    up = mx.quantized_matmul(
        x,
        up_weight,
        scales=up_scales,
        biases=up_biases,
        group_size=group_size,
        bits=4,
        mode="affine",
    )
    hidden = swiglu(gate, up)
    return mx.quantized_matmul(
        hidden,
        down_weight,
        scales=down_scales,
        biases=down_biases,
        group_size=group_size,
        bits=4,
        mode="affine",
    )


class HotExpertSwitchGLU(nn.Module):
    """Correctness-first slot-backed replacement for ``SwitchGLU``.

    This portable path reconstructs MLX arrays from each fixed host slot and
    evaluates a bounded wave before releasing it.  The native extension can
    replace the component binding without changing router or cache semantics.
    """

    def __init__(self, runtime: ExpertStreamingRuntime, layer_index: int):
        super().__init__()
        self.runtime = runtime
        self.layer_index = int(layer_index)
        self.group_size = runtime.spec.quant_group_size

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if indices.ndim < 1:
            raise ValueError("expert indices must include a top-k dimension")
        if int(indices.shape[-1]) != self.runtime.spec.top_k:
            raise ValueError(
                f"router selected {indices.shape[-1]} experts; expected "
                f"{self.runtime.spec.top_k}"
            )
        hidden_size = int(x.shape[-1])
        if hidden_size != self.runtime.spec.hidden_size:
            raise ValueError(
                f"expert input width {hidden_size} does not match "
                f"{self.runtime.spec.hidden_size}"
            )
        tokens = x.reshape(-1, hidden_size)
        top_k = int(indices.shape[-1])
        mx.eval(indices)
        expert_ids = tuple(int(value) for value in indices.reshape(-1).tolist())
        phase = current_expert_routing_phase(token_count=int(tokens.shape[0]))

        outputs: list[mx.array] = []
        output_positions: list[int] = []
        for wave in self.runtime.route_waves(expert_ids):
            ready = self.runtime.ensure_route(
                self.layer_index,
                wave.experts,
                phase=phase,
            )
            try:
                by_expert: dict[int, list[int]] = {}
                binding_by_expert: dict[int, ExpertSlotBinding] = {}
                for global_position, binding in zip(
                    wave.positions, ready.bindings, strict=True
                ):
                    by_expert.setdefault(binding.expert, []).append(global_position)
                    binding_by_expert.setdefault(binding.expert, binding)
                wave_outputs: list[mx.array] = []
                wave_positions: list[int] = []
                for expert, positions in by_expert.items():
                    token_positions = mx.array(
                        [position // top_k for position in positions],
                        dtype=mx.int32,
                    )
                    selected = mx.take(tokens, token_positions, axis=0)
                    result = _run_q4_expert(
                        selected,
                        binding_by_expert[expert],
                        group_size=self.group_size,
                    )
                    wave_outputs.append(result)
                    wave_positions.extend(positions)
                mx.eval(wave_outputs)
                outputs.extend(wave_outputs)
                output_positions.extend(wave_positions)
            finally:
                # All Q4 component copies and qmm outputs above are evaluated,
                # so the source byte slots can be reused without a stale read.
                ready.release(synchronize=True)

        if not outputs:
            raise ValueError("router produced no expert assignments")
        joined = mx.concatenate(outputs, axis=0)
        order = mx.argsort(mx.array(output_positions, dtype=mx.int32))
        joined = mx.take(joined, order, axis=0)
        return joined.reshape((*indices.shape, hidden_size))


def bind_streamed_switches(model: Any, runtime: ExpertStreamingRuntime) -> int:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model does not expose transformer layers")
    bound = 0
    for layer_index in runtime.spec.routed_layer_indices:
        layer = layers[layer_index]
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            raise TypeError(f"layer {layer_index} has no switch_mlp seam")
        mlp.switch_mlp = HotExpertSwitchGLU(runtime, layer_index)
        bound += 1
    return bound
