"""MLX execution adapters for slot-backed affine-quantized routed experts."""

from __future__ import annotations

import gc
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.activations import swiglu

from mtplx import expert_route_probe as _route_probe

from mtplx.expert_runtime import ExpertStreamingRuntime
from mtplx.expert_manifest import ExpertManifest, ExpertRecord
from mtplx.expert_slots import ExpertSlotBinding, ReadyRoute
from mtplx.expert_streaming import RoutingPhase
from mtplx.expert_streaming_models import ExpertMemoryPlan, ExpertStreamingModelSpec
from mtplx.mmap_mlx import mmap_u32




_ROUTING_PHASE: ContextVar[RoutingPhase | None] = ContextVar(
    "mtplx_expert_routing_phase",
    default=None,
)
_SLOT_INDEX_PATTERN = r"(?:0|[1-9][0-9]*)"
_LAYER_PERSISTENT_LABEL = re.compile(
    rf"layer-({_SLOT_INDEX_PATTERN})-persistent-({_SLOT_INDEX_PATTERN})"
)
_GLOBAL_PERSISTENT_LABEL = re.compile(rf"global-persistent-({_SLOT_INDEX_PATTERN})")
_GLOBAL_TRANSIENT_LABEL = re.compile(rf"global-transient-({_SLOT_INDEX_PATTERN})")


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


def _mark_pipeline_incomplete(ledger: Any, phase: RoutingPhase) -> None:
    try:
        ledger.mark_incomplete(phase=phase)
    except Exception:
        pass


def _begin_pipeline_work(
    ledger: Any,
    method: str,
    *args: Any,
    phase: RoutingPhase,
) -> Any | None:
    """Open optional diagnostics without changing model execution outcomes."""

    try:
        return getattr(ledger, method)(*args, phase=phase)
    except Exception:
        _mark_pipeline_incomplete(ledger, phase)
        return None


def _pipeline_work_call(
    ledger: Any,
    target: Any,
    method: str,
    *args: Any,
    phase: RoutingPhase,
) -> None:
    """Publish one optional work transition while preserving data-path errors."""

    try:
        getattr(target, method)(*args)
    except Exception:
        _mark_pipeline_incomplete(ledger, phase)


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
    """Create stable direct MLX/Metal buffers without materialized bank slices.

    MLX integer indexing does not expose a writable view: evaluating
    ``bank[slot]`` allocates a second buffer.  Keeping both the bank and all
    evaluated slices therefore doubled the expert-cache allocation.  Direct
    fixed slots preserve positional-I/O and generation semantics while making
    physical allocation match the memory plan.
    """

    slots: dict[str, mx.array] = {}
    backend = "mlx-metal-direct-slots"

    def allocate(size: int, label: str) -> mx.array:
        if size != spec.expert_record_bytes:
            raise ValueError("slot allocator size differs from the model descriptor")
        parts = label.split("-")
        if label.startswith("layer-") and "-persistent-" in label:
            layer = int(parts[1])
            slot = int(parts[-1])
            count = plan.slots_per_layer
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"persistent slot layer {layer} is not routed")
        elif label.startswith("global-persistent-"):
            slot = int(parts[-1])
            count = plan.persistent_slots
        elif label.startswith("global-transient-"):
            slot = int(parts[-1])
            count = plan.transient_slots
        else:
            raise ValueError(f"unknown expert slot label {label!r}")
        if count <= 0:
            raise ValueError(f"slot {label} has no planned capacity")
        if not 0 <= slot < count:
            raise ValueError(f"slot {label} is outside planned capacity {count}")
        if label in slots:
            raise ValueError(f"slot {label} was allocated twice")
        value = mlx_slot_buffer_allocator(size, label)
        slots[label] = value
        return value

    setattr(allocate, "backend", backend)
    setattr(allocate, "slots", slots)
    return allocate


class MlxComponentBank:
    """Component-major writable MLX storage for a fixed expert-slot tier."""

    def __init__(
        self,
        *,
        capacity: int,
        record: ExpertRecord,
        label: str,
    ) -> None:
        self.capacity = int(capacity)
        self.label = str(label)
        self.record_bytes = int(record.logical_bytes)
        self.arrays: dict[str, mx.array] = {}
        self._views: dict[str, memoryview] = {}
        self._segment_bytes: dict[str, int] = {}
        if self.capacity <= 0:
            raise ValueError("component bank capacity must be positive")
        try:
            for segment in record.segments:
                if segment.component in self.arrays:
                    raise ValueError(
                        f"duplicate component {segment.component!r} in expert record"
                    )
                dtype = {
                    "U32": mx.uint32,
                    "BF16": mx.bfloat16,
                }.get(segment.dtype)
                if dtype is None:
                    raise TypeError(f"unsupported component-bank dtype {segment.dtype}")
                value = mx.zeros((self.capacity, *segment.shape), dtype=dtype)
                mx.eval(value)
                view = memoryview(value)
                if view.readonly or not view.c_contiguous:
                    raise RuntimeError(
                        f"component bank {label}/{segment.component} is not writable"
                    )
                raw = view.cast("B")
                expected = self.capacity * segment.length
                if raw.nbytes != expected:
                    raise RuntimeError(
                        f"component bank {label}/{segment.component} has "
                        f"{raw.nbytes} bytes; expected {expected}"
                    )
                self.arrays[segment.component] = value
                self._views[segment.component] = raw
                self._segment_bytes[segment.component] = segment.length
        except Exception:
            for view in self._views.values():
                view.release()
            self._views.clear()
            self.arrays.clear()
            raise

    def component_view(self, slot: int, component: str) -> memoryview:
        if not 0 <= int(slot) < self.capacity:
            raise IndexError("component-bank slot is outside capacity")
        length = self._segment_bytes[component]
        start = int(slot) * length
        return self._views[component][start : start + length]

    def close(self) -> None:
        for view in self._views.values():
            try:
                view.release()
            except Exception:
                pass
        self._views.clear()
        self.arrays.clear()
        self._segment_bytes.clear()


class MlxComponentSlot:
    """One logical slot backed by a row in nine component-major MLX arrays."""

    def __init__(
        self,
        bank: MlxComponentBank,
        bank_index: int,
        *,
        label: str,
    ) -> None:
        self.bank = bank
        self.bank_index = int(bank_index)
        self.label = str(label)
        self.nbytes = bank.record_bytes

    def record_views(self, record: ExpertRecord) -> tuple[memoryview, ...]:
        if int(record.logical_bytes) != self.nbytes:
            raise ValueError("record size differs from component-bank slot")
        return tuple(
            self.bank.component_view(self.bank_index, segment.component)
            for segment in record.segments
        )

    def component_view(self, component: str) -> memoryview:
        return self.bank.component_view(self.bank_index, component)


class MappedExpertRecord:
    """One expert record backed directly by its sidecar file pages."""

    def __init__(self, record: ExpertRecord, base: mx.array) -> None:
        self.record = record
        self.base = base
        self._arrays: dict[str, mx.array] | None = None

    @property
    def arrays(self) -> dict[str, mx.array]:
        arrays = self._arrays
        if arrays is not None:
            return arrays
        arrays = {}
        cursor = 0
        for segment in self.record.segments:
            if segment.dtype == "U32":
                typed = self.base
                item_size = 4
            elif segment.dtype == "BF16":
                typed = mx.view(self.base, mx.bfloat16)
                item_size = 2
            else:
                raise TypeError(f"unsupported mapped component dtype {segment.dtype}")
            if cursor % item_size:
                raise ValueError(f"component {segment.component} is not dtype-aligned")
            arrays[segment.component] = mx.as_strided(
                typed,
                shape=segment.shape,
                offset=cursor // item_size,
            )
            cursor += segment.length
        if cursor != self.record.logical_bytes:
            raise ValueError("mapped component layout does not cover the record")
        self._arrays = arrays
        return arrays


class MappedExpertStore:
    """Virtual-map every sidecar record without adding it to MLX residency.

    The MTLBuffers remain addressable for the life of the model, but their
    pages are not in MLX's process-wide wired residency set. Metal binds only
    the routed record buffers for a QMM command; macOS can retain or evict the
    corresponding file pages through its normal page cache.
    """

    def __init__(
        self,
        root: Path | str,
        manifest: ExpertManifest,
        *,
        workers: int = 96,
    ) -> None:
        if manifest.sidecar is None:
            raise ValueError("metal-mmap execution requires a sidecar manifest")
        self.root = Path(root).resolve()
        self.path = self.root / manifest.sidecar.file
        self.records = tuple(manifest.records)
        self.workers = max(1, min(int(workers), 256))
        self._mapped: dict[tuple[int, int], MappedExpertRecord] = {}
        self._lock = threading.Lock()
        self._mapping_seconds = 0.0
        self._qmm_experts = 0
        self._closed = False
        expected = {(record.layer, record.expert) for record in self.records}
        if len(expected) != len(self.records):
            raise ValueError("sidecar contains duplicate layer/expert records")
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        for record in self.records:
            if record.sidecar_offset is None or record.sidecar_length is None:
                raise ValueError("metal-mmap record has no sidecar range")
            if record.sidecar_length != record.logical_bytes:
                raise ValueError("metal-mmap sidecar length differs from record")
            if record.sidecar_offset % page_size or record.sidecar_length % page_size:
                raise ValueError("metal-mmap sidecar records must be page aligned")

    def prepare(self) -> None:
        if self._closed:
            raise RuntimeError("mapped expert store is closed")
        if len(self._mapped) == len(self.records):
            return
        started = time.perf_counter()

        def map_record(
            record: ExpertRecord,
        ) -> tuple[tuple[int, int], MappedExpertRecord]:
            assert record.sidecar_offset is not None
            assert record.sidecar_length is not None
            base = mmap_u32(
                self.path,
                record.sidecar_offset,
                record.sidecar_length,
                wired=False,
            )
            return (record.layer, record.expert), MappedExpertRecord(record, base)

        with ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="mtplx-mmap",
        ) as executor:
            mapped = dict(executor.map(map_record, self.records))
        with self._lock:
            self._mapped = mapped
            self._mapping_seconds += time.perf_counter() - started

    def get(self, layer: int, expert: int) -> MappedExpertRecord:
        try:
            return self._mapped[(int(layer), int(expert))]
        except KeyError as exc:
            raise KeyError(f"mapped expert ({layer}, {expert}) is unavailable") from exc

    def observe_qmm(self, count: int) -> None:
        with self._lock:
            self._qmm_experts += int(count)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "metal-mmap-unwired-records",
                "record_count": len(self.records),
                "mapped_records": len(self._mapped),
                "virtual_bytes": sum(
                    int(record.logical_bytes) for record in self.records
                ),
                "mapping_seconds": self._mapping_seconds,
                "workers": self.workers,
                "qmm_experts": self._qmm_experts,
                "globally_wired_bytes": 0,
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mapped.clear()
        # Evaluated MLX arrays can retain graph-input cycles until cyclic GC.
        # Collect now so every external MTLBuffer releases before its mmap.
        gc.collect()


def make_mlx_component_bank_allocator(
    plan: ExpertMemoryPlan,
    spec: ExpertStreamingModelSpec,
    manifest: ExpertManifest,
) -> Callable[[int, str], MlxComponentSlot]:
    """Allocate slot bytes as component-major banks usable by ``gather_qmm``.

    Unlike a record-major byte bank, these arrays are both directly writable
    through unified-memory views and directly consumable by MLX grouped QMM
    kernels. No persistent slice or stacked weight copy is materialized.
    """

    record_by_key: dict[tuple[int, int], ExpertRecord] = {}
    duplicate_keys: set[tuple[int, int]] = set()
    record_by_layer: dict[int, ExpertRecord] = {}
    for record in manifest.records:
        key = (record.layer, record.expert)
        if key in record_by_key:
            duplicate_keys.add(key)
        else:
            record_by_key[key] = record
        record_by_layer.setdefault(record.layer, record)
    missing = set(spec.routed_layer_indices) - set(record_by_layer)
    if missing:
        raise ValueError(
            f"manifest has no exemplar records for layers {sorted(missing)}"
        )
    if plan.cache_scope == "global":
        expected_keys = {
            (layer, expert)
            for layer in spec.routed_layer_indices
            for expert in range(spec.expert_count)
        }
        actual_keys = set(record_by_key)
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        if missing_keys or extra_keys or duplicate_keys:
            raise ValueError(
                "manifest routed expert keys differ from model descriptor: "
                f"missing={missing_keys}, extra={extra_keys}, "
                f"duplicates={sorted(duplicate_keys)}"
            )

    def component_signature(record: ExpertRecord) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                segment.component,
                segment.dtype,
                tuple(segment.shape),
                int(segment.length),
            )
            for segment in record.segments
        )

    expected_signature: list[tuple[str, str, tuple[int, ...], int]] = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        output_size = (
            spec.expert_hidden_size
            if projection in {"gate_proj", "up_proj"}
            else spec.hidden_size
        )
        input_size = (
            spec.hidden_size
            if projection in {"gate_proj", "up_proj"}
            else spec.expert_hidden_size
        )
        weight_shape = (output_size, input_size * spec.quant_bits // 32)
        parameter_shape = (output_size, input_size // spec.quant_group_size)
        expected_signature.extend(
            (
                (
                    f"{projection}.weight",
                    "U32",
                    weight_shape,
                    output_size * input_size * spec.quant_bits // 8,
                ),
                (
                    f"{projection}.scales",
                    "BF16",
                    parameter_shape,
                    output_size
                    * (input_size // spec.quant_group_size)
                    * spec.quant_parameter_bytes,
                ),
                (
                    f"{projection}.biases",
                    "BF16",
                    parameter_shape,
                    output_size
                    * (input_size // spec.quant_group_size)
                    * spec.quant_parameter_bytes,
                ),
            )
        )

    exemplar_layer = spec.routed_layer_indices[0]
    exemplar_signature = component_signature(record_by_layer[exemplar_layer])
    if exemplar_signature != tuple(expected_signature):
        raise ValueError(
            "manifest component geometry does not match the model descriptor"
        )
    routed_layers = set(spec.routed_layer_indices)
    for record in manifest.records:
        if (
            record.layer in routed_layers
            and component_signature(record) != exemplar_signature
        ):
            raise ValueError(
                "routed-layer component geometry differs for "
                f"expert ({record.layer}, {record.expert}) from canonical "
                f"layer {exemplar_layer}"
            )

    banks: dict[tuple[str, int], MlxComponentBank] = {}
    slots: dict[str, MlxComponentSlot] = {}
    backend = "mlx-metal-component-banks"

    def bank_for(kind: str, layer: int) -> MlxComponentBank:
        key = (kind, layer if kind == "persistent" else -1)
        bank = banks.get(key)
        if bank is not None:
            return bank
        if kind == "persistent":
            capacity = plan.slots_per_layer
            record = record_by_layer[layer]
            label = f"layer-{layer}-persistent-bank"
        elif kind == "global-persistent":
            capacity = plan.persistent_slots
            record = record_by_layer[exemplar_layer]
            label = "global-persistent-bank"
        else:
            capacity = plan.transient_slots
            record = record_by_layer[exemplar_layer]
            label = "global-transient-bank"
        bank = MlxComponentBank(capacity=capacity, record=record, label=label)
        banks[key] = bank
        return bank

    def allocate(size: int, label: str) -> MlxComponentSlot:
        if int(size) != spec.expert_record_bytes:
            raise ValueError("slot allocator size differs from model descriptor")
        layer_persistent = _LAYER_PERSISTENT_LABEL.fullmatch(label)
        global_persistent = _GLOBAL_PERSISTENT_LABEL.fullmatch(label)
        global_transient = _GLOBAL_TRANSIENT_LABEL.fullmatch(label)
        if layer_persistent is not None:
            if plan.cache_scope != "layer":
                raise ValueError(
                    "layer-persistent slot label conflicts with global cache scope"
                )
            layer = int(layer_persistent.group(1))
            slot_index = int(layer_persistent.group(2))
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"persistent slot layer {layer} is not routed")
            if not 0 <= slot_index < plan.slots_per_layer:
                raise ValueError("persistent slot is outside planned capacity")
            bank = bank_for("persistent", layer)
        elif label.startswith("layer-") and "-persistent-" in label:
            raise ValueError(f"unknown expert slot label {label!r}")
        elif global_persistent is not None:
            if plan.cache_scope != "global":
                raise ValueError(
                    "global-persistent slot label conflicts with layer cache scope"
                )
            slot_index = int(global_persistent.group(1))
            if not 0 <= slot_index < plan.persistent_slots:
                raise ValueError("global persistent slot is outside planned capacity")
            bank = bank_for("global-persistent", -1)
        elif label.startswith("global-persistent-"):
            raise ValueError(f"unknown expert slot label {label!r}")
        elif global_transient is not None:
            slot_index = int(global_transient.group(1))
            if not 0 <= slot_index < plan.transient_slots:
                raise ValueError("transient slot is outside planned capacity")
            bank = bank_for("transient", -1)
        elif label.startswith("global-transient-"):
            raise ValueError(f"unknown expert slot label {label!r}")
        else:
            raise ValueError(f"unknown expert slot label {label!r}")
        if label in slots:
            raise ValueError(f"slot {label} was allocated twice")
        slot = MlxComponentSlot(bank, slot_index, label=label)
        slots[label] = slot
        return slot

    def close_banks() -> None:
        for bank in tuple(banks.values()):
            bank.close()
        banks.clear()
        slots.clear()
        _release_mlx_cache()

    setattr(allocate, "backend", backend)
    setattr(allocate, "slots", slots)
    setattr(allocate, "banks", banks)
    setattr(allocate, "close", close_banks)
    return allocate


def _release_mlx_cache() -> None:
    try:
        mx.clear_cache()
    except Exception:  # pragma: no cover - compatibility with older MLX
        pass


def _run_q4_expert(
    x: mx.array,
    binding: ExpertSlotBinding,
    *,
    group_size: int,
    bits: int = 4,
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
        bits=bits,
        mode="affine",
    )
    up = mx.quantized_matmul(
        x,
        up_weight,
        scales=up_scales,
        biases=up_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    hidden = swiglu(gate, up)
    return mx.quantized_matmul(
        hidden,
        down_weight,
        scales=down_scales,
        biases=down_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


def _run_component_bank_q4(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    group_size: int,
    bits: int = 4,
) -> mx.array:
    """Execute assignment-aligned rows from one component-major slot bank."""

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")
    selected = x.reshape((len(bindings), 1, 1, int(x.shape[-1])))
    slot_indices = mx.array(
        [int(binding.buffer.bank_index) for binding in bindings],
        dtype=mx.int32,
    ).reshape((-1, 1))

    def qmm(values: mx.array, projection: str) -> mx.array:
        return mx.gather_qmm(
            values,
            bank.arrays[f"{projection}.weight"],
            bank.arrays[f"{projection}.scales"],
            bank.arrays[f"{projection}.biases"],
            rhs_indices=slot_indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )

    gate = qmm(selected, "gate_proj")
    up = qmm(selected, "up_proj")
    output = qmm(swiglu(gate, up), "down_proj")
    return output.reshape((len(bindings), int(output.shape[-1])))


def _run_mapped_q4(
    x: mx.array,
    mapped: MappedExpertRecord,
    *,
    group_size: int,
    bits: int = 4,
) -> mx.array:
    arrays = mapped.arrays

    def qmm(values: mx.array, projection: str) -> mx.array:
        return mx.quantized_matmul(
            values,
            arrays[f"{projection}.weight"],
            scales=arrays[f"{projection}.scales"],
            biases=arrays[f"{projection}.biases"],
            group_size=group_size,
            bits=bits,
            mode="affine",
        )

    return qmm(swiglu(qmm(x, "gate_proj"), qmm(x, "up_proj")), "down_proj")


class MappedExpertSwitchGLU(nn.Module):
    """Execute routed quantized experts from record-sized file-backed MTLBuffers."""

    def __init__(
        self,
        runtime: ExpertStreamingRuntime,
        store: MappedExpertStore,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.store = store
        self.layer_index = int(layer_index)
        self.group_size = runtime.spec.quant_group_size
        self.bits = runtime.spec.quant_bits

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        hidden_size = int(x.shape[-1])
        top_k = int(indices.shape[-1])
        if top_k != self.runtime.spec.top_k:
            raise ValueError("mapped router top-k differs from the model descriptor")
        tokens = x.reshape((-1, hidden_size))
        mx.eval(indices)
        expert_ids = tuple(int(value) for value in indices.reshape(-1).tolist())
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        self.runtime.observe_route(
            self.layer_index,
            phase,
            expert_ids,
            token_count=int(tokens.shape[0]),
        )
        by_expert: dict[int, list[int]] = {}
        for position, expert in enumerate(expert_ids):
            by_expert.setdefault(expert, []).append(position)

        outputs: list[mx.array] = []
        output_positions: list[int] = []
        for expert, positions in by_expert.items():
            token_positions = mx.array(
                [position // top_k for position in positions],
                dtype=mx.int32,
            )
            selected = mx.take(tokens, token_positions, axis=0)
            outputs.append(
                _run_mapped_q4(
                    selected,
                    self.store.get(self.layer_index, expert),
                    group_size=self.group_size,
                    bits=self.bits,
                )
            )
            output_positions.extend(positions)
        mx.eval(outputs)
        self.store.observe_qmm(len(by_expert))
        joined = mx.concatenate(outputs, axis=0)
        order = mx.argsort(mx.array(output_positions, dtype=mx.int32))
        return mx.take(joined, order, axis=0).reshape((*indices.shape, hidden_size))


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
        self.bits = runtime.spec.quant_bits

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        output, _overlap_result = self._run(
            x,
            indices,
            shared_work=None,
        )
        return output

    def run_with_shared_overlap(
        self,
        x: mx.array,
        indices: mx.array,
        shared_work: Callable[[], mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Evaluate resident shared work while decode misses stream from SSD."""

        output, shared = self._run(
            x,
            indices,
            shared_work=shared_work,
        )
        assert shared is not None
        return output, shared

    def _run(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        shared_work: Callable[[], mx.array] | None,
    ) -> tuple[mx.array, mx.array | None]:
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
        with _route_probe.bracket("hot.eval_indices"):
            mx.eval(indices)
        # This eval materialized every earlier layer's wave output, so any
        # deferred pin releases are now covered without their own fence.
        flush_deferred = getattr(self.runtime, "flush_deferred_slot_releases", None)
        if flush_deferred is not None:
            flush_deferred()
        with _route_probe.bracket("hot.route_host"):
            expert_ids = tuple(int(value) for value in indices.reshape(-1).tolist())
        # Batch size is not a generation phase. A batched decode has shape
        # ``[B, 1, H]`` and must still train/use the persistent decode hot set;
        # only the sequence length distinguishes prefill from decode here.
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        self.runtime.observe_route(
            self.layer_index,
            phase,
            expert_ids,
            token_count=int(tokens.shape[0]),
        )
        if phase is RoutingPhase.PREFILL:
            self.runtime.prepare_prefill_seed(self.layer_index, expert_ids)

        outputs: list[mx.array] = []
        output_positions: list[int] = []
        shared: mx.array | None = None

        def update_fence_metrics(ready: ReadyRoute, **values: int) -> None:
            metrics = getattr(getattr(ready, "pool", None), "metrics", None)
            update = getattr(metrics, "update", None)
            if callable(update):
                update(**values)

        def synchronous_fence(ready: ReadyRoute, values: Any) -> None:
            if self.runtime.config.resource_telemetry:
                update_fence_metrics(
                    ready,
                    synchronous_fences=1,
                    synchronous_fence_slots=len(ready.bindings),
                )
            try:
                mx.eval(values)
            except BaseException as exc:
                update_fence_metrics(ready, completion_fence_failures=1)
                record = getattr(
                    getattr(ready, "pool", None),
                    "_record_completion_error",
                    None,
                )
                if callable(record):
                    record(exc)
                raise

        def fence_bindings(
            ready: ReadyRoute,
            bindings: tuple[ExpertSlotBinding, ...],
            wave_outputs: list[mx.array],
            *,
            force_sync: bool = False,
        ) -> None:
            raw = os.environ.get("MTPLX_EXPERT_SLOT_FENCES", "1")
            enabled = raw.strip().lower() not in {"0", "false", "no", "off"}
            async_eval = getattr(mx, "async_eval", None)
            if not enabled:
                synchronous_fence(ready, wave_outputs)
                return
            if not callable(async_eval):
                update_fence_metrics(ready, completion_fence_fallbacks=1)
                synchronous_fence(ready, wave_outputs)
                return
            if force_sync:
                update_fence_metrics(
                    ready,
                    completion_fences=1,
                    completion_fence_slots=len(bindings),
                )
                synchronous_fence(ready, wave_outputs)
                return
            try:
                async_eval(wave_outputs)
            except Exception:
                # Older/stripped MLX builds may expose the name without a
                # usable asynchronous evaluator. Preserve the generation
                # fence with the original synchronous barrier.
                update_fence_metrics(ready, completion_fence_fallbacks=1)
                synchronous_fence(ready, wave_outputs)
                return
            roots = tuple(wave_outputs)
            defer = getattr(ready, "defer_bindings_until", None)
            if not callable(defer):
                synchronous_fence(ready, roots)
                return
            try:
                defer(
                    bindings,
                    lambda: mx.eval(roots),
                )
            except Exception:
                # If the completion lane rejects or cannot represent this
                # binding set, do not release the route on an async promise.
                update_fence_metrics(ready, completion_fence_fallbacks=1)
                synchronous_fence(ready, roots)
                raise_completion_error = getattr(
                    getattr(ready, "pool", None),
                    "_raise_completion_error",
                    None,
                )
                if callable(raise_completion_error):
                    raise_completion_error()

        def evaluate_component_bindings(
            positions: tuple[int, ...] | list[int],
            bindings: tuple[ExpertSlotBinding, ...],
            ready: ReadyRoute,
            *,
            force_sync: bool = False,
        ) -> None:
            if not positions:
                return
            by_bank: dict[int, list[tuple[int, ExpertSlotBinding]]] = {}
            for global_position, binding in zip(positions, bindings, strict=True):
                by_bank.setdefault(id(binding.buffer.bank), []).append(
                    (global_position, binding)
                )
            wave_outputs: list[mx.array] = []
            wave_positions: list[int] = []
            for assignments in by_bank.values():
                grouped_positions = [position for position, _binding in assignments]
                grouped_bindings = tuple(binding for _position, binding in assignments)
                token_positions = mx.array(
                    [position // top_k for position in grouped_positions],
                    dtype=mx.int32,
                )
                selected = mx.take(tokens, token_positions, axis=0)
                wave_outputs.append(
                    _run_component_bank_q4(
                        selected,
                        grouped_bindings,
                        group_size=self.group_size,
                        bits=self.bits,
                    )
                )
                wave_positions.extend(grouped_positions)
            fence_bindings(
                ready,
                bindings,
                wave_outputs,
                force_sync=force_sync,
            )
            outputs.extend(wave_outputs)
            output_positions.extend(wave_positions)

        def evaluate_direct_bindings(
            positions: tuple[int, ...] | list[int],
            bindings: tuple[ExpertSlotBinding, ...],
            ready: ReadyRoute,
            *,
            force_sync: bool = False,
        ) -> None:
            if not positions:
                return
            by_expert: dict[int, list[int]] = {}
            binding_by_expert: dict[int, ExpertSlotBinding] = {}
            for global_position, binding in zip(positions, bindings, strict=True):
                by_expert.setdefault(binding.expert, []).append(global_position)
                binding_by_expert.setdefault(binding.expert, binding)
            wave_outputs: list[mx.array] = []
            wave_positions: list[int] = []
            for expert, expert_positions in by_expert.items():
                token_positions = mx.array(
                    [position // top_k for position in expert_positions],
                    dtype=mx.int32,
                )
                selected = mx.take(tokens, token_positions, axis=0)
                wave_outputs.append(
                    _run_q4_expert(
                        selected,
                        binding_by_expert[expert],
                        group_size=self.group_size,
                        bits=self.bits,
                    )
                )
                wave_positions.extend(expert_positions)
            fence_bindings(
                ready,
                bindings,
                wave_outputs,
                force_sync=force_sync,
            )
            outputs.extend(wave_outputs)
            output_positions.extend(wave_positions)

        pipeline_ledger = getattr(self.runtime, "_pipeline_ledger", None)
        shared_pipeline_work = None
        if (
            pipeline_ledger is not None
            and shared_work is not None
            and phase is RoutingPhase.DECODE
        ):
            shared_pipeline_work = _begin_pipeline_work(
                pipeline_ledger,
                "begin_shared_work",
                phase=phase,
            )

        try:
            for wave in self.runtime.route_waves(
                expert_ids,
                sort_unique=(
                    phase is RoutingPhase.PREFILL
                    and self.runtime.manifest.sidecar is not None
                ),
            ):
                # Keep the all-hit optimization inside the authoritative bounded
                # route-wave loop.  A successful probe avoids split-route futures
                # and per-expert grouping while retaining the normal policy epoch,
                # counters, and assignment order for this wave.
                if (
                    phase is RoutingPhase.DECODE
                    and self.runtime.config.slot_layout == "component-banks"
                ):
                    with _route_probe.bracket("hot.try_all_hit"):
                        ready = self.runtime.try_all_hit_route(
                            self.layer_index,
                            wave.experts,
                            phase=phase,
                        )
                    _route_probe.count(
                        "hot.all_hit" if ready is not None else "hot.split_route"
                    )
                    if ready is not None:
                        hit_pipeline_work = None
                        if pipeline_ledger is not None:
                            hit_pipeline_work = _begin_pipeline_work(
                                pipeline_ledger,
                                "begin_hit_work",
                                ready.plan.hits,
                                phase=phase,
                            )
                        try:
                            if wave.positions == tuple(range(len(expert_ids))):
                                assignment_inputs = mx.broadcast_to(
                                    tokens[:, None, :],
                                    (int(tokens.shape[0]), top_k, hidden_size),
                                ).reshape((-1, hidden_size))
                            else:
                                token_positions = mx.array(
                                    [position // top_k for position in wave.positions],
                                    dtype=mx.int32,
                                )
                                assignment_inputs = mx.take(
                                    tokens,
                                    token_positions,
                                    axis=0,
                                )
                            if (
                                pipeline_ledger is not None
                                and hit_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    hit_pipeline_work,
                                    "claim",
                                    phase=phase,
                                )
                            with _route_probe.bracket("hot.allhit_dispatch_build"):
                                wave_output = _run_component_bank_q4(
                                    assignment_inputs,
                                    ready.bindings,
                                    group_size=self.group_size,
                                    bits=self.bits,
                                )
                            # Slot pins may be released only after the lazy graph
                            # has consumed the currently bound bank generations.
                            # Deferred mode: the next generation-thread eval is
                            # that consumption proof; no per-layer fence runs.
                            deferred_release = False
                            # Promoted default via ExpertStreamingConfig after
                            # the C3 matrix; fakes without the field keep the
                            # fence path.
                            if getattr(
                                self.runtime.config,
                                "deferred_pin_release",
                                False,
                            ):
                                self.runtime.defer_slot_release(ready, wave_output)
                                deferred_release = True
                                _route_probe.count("hot.allhit_defer")
                            else:
                                with _route_probe.bracket("hot.allhit_fence_eval"):
                                    synchronous_fence(ready, wave_output)
                            outputs.append(wave_output)
                            output_positions.extend(wave.positions)
                        finally:
                            if (
                                pipeline_ledger is not None
                                and hit_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    hit_pipeline_work,
                                    "close",
                                    phase=phase,
                                )
                            if not deferred_release:
                                ready.release(synchronize=False)
                        continue

                # Both layouts pin hits and start miss reads first, then run the
                # resident experts on the GPU while the misses stream from SSD.
                evaluate_bindings = (
                    evaluate_component_bindings
                    if self.runtime.config.slot_layout == "component-banks"
                    else evaluate_direct_bindings
                )
                with _route_probe.bracket("hot.begin_split_route"):
                    pending = self.runtime.begin_split_route(
                        self.layer_index,
                        wave.experts,
                        phase=phase,
                    )
                try:
                    hit_pipeline_work = None
                    if pipeline_ledger is not None and pending.hit_ready is not None:
                        hit_pipeline_work = _begin_pipeline_work(
                            pipeline_ledger,
                            "begin_hit_work",
                            pending.plan.hits,
                            phase=phase,
                        )
                    try:
                        hit_set = set(pending.plan.hits)
                        hit_positions = tuple(
                            position
                            for position, expert in zip(
                                wave.positions, wave.experts, strict=True
                            )
                            if expert in hit_set
                        )
                        if pending.hit_ready is not None:
                            # Split parts feed one shared lazy graph. Keep every MLX
                            # eval on the generation thread; evaluating a fence on the
                            # completion lane can race the next part's graph traversal.
                            if (
                                pipeline_ledger is not None
                                and hit_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    hit_pipeline_work,
                                    "claim",
                                    phase=phase,
                                )
                            evaluate_bindings(
                                hit_positions,
                                pending.hit_ready.bindings,
                                pending.hit_ready,
                                force_sync=True,
                            )
                    finally:
                        if (
                            pipeline_ledger is not None
                            and hit_pipeline_work is not None
                        ):
                            _pipeline_work_call(
                                pipeline_ledger,
                                hit_pipeline_work,
                                "close",
                                phase=phase,
                            )
                    if pending.hit_ready is not None:
                        pending.release_hits()
                    # The resident shared branch depends only on ``x``.  Force it
                    # on Metal while the native readers own miss futures, so
                    # all-miss layers have useful GPU work instead of an empty
                    # device.  Keep prefill unchanged: at 128K, eagerly retaining
                    # the full shared output across routed waves would violate the
                    # bounded-memory execution contract.
                    if (
                        shared_work is not None
                        and shared is None
                        and phase is RoutingPhase.DECODE
                        and pending.misses_pending
                    ):
                        if (
                            pipeline_ledger is not None
                            and shared_pipeline_work is not None
                        ):
                            _pipeline_work_call(
                                pipeline_ledger,
                                shared_pipeline_work,
                                "claim",
                                phase=phase,
                            )
                        try:
                            shared = shared_work()
                            mx.eval(shared)
                        finally:
                            if (
                                pipeline_ledger is not None
                                and shared_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    shared_pipeline_work,
                                    "close",
                                    phase=phase,
                                )
                                shared_pipeline_work = None
                    for miss_ready in pending.iter_ready_misses():
                        part_error: BaseException | None = None
                        try:
                            ready_experts = set(miss_ready.plan.experts)
                            miss_positions = tuple(
                                position
                                for position, expert in zip(
                                    wave.positions, wave.experts, strict=True
                                )
                                if expert not in hit_set and expert in ready_experts
                            )
                            if pipeline_ledger is not None:
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    pending,
                                    "claim_misses",
                                    miss_ready,
                                    phase=phase,
                                )
                            evaluate_bindings(
                                miss_positions,
                                miss_ready.bindings,
                                miss_ready,
                                force_sync=True,
                            )
                        except BaseException as exc:
                            part_error = exc
                            raise
                        finally:
                            try:
                                pending.release_miss(miss_ready)
                            except BaseException:
                                if part_error is None:
                                    raise
                except BaseException as exc:
                    pending.abort(exc)
                    raise
                finally:
                    pending.close()

            if not outputs:
                raise ValueError("router produced no expert assignments")
            if len(outputs) == 1 and output_positions == list(range(len(expert_ids))):
                joined = outputs[0]
            else:
                joined = mx.concatenate(outputs, axis=0)
                order = mx.argsort(mx.array(output_positions, dtype=mx.int32))
                joined = mx.take(joined, order, axis=0)
            output = joined.reshape((*indices.shape, hidden_size))
            if shared_work is not None and shared is None:
                # No physical wait remained to hide, or this was a bounded-memory
                # prefill. Preserve the original routed-then-shared ordering.
                if pipeline_ledger is not None and shared_pipeline_work is not None:
                    _pipeline_work_call(
                        pipeline_ledger,
                        shared_pipeline_work,
                        "claim",
                        phase=phase,
                    )
                try:
                    shared = shared_work()
                finally:
                    if pipeline_ledger is not None and shared_pipeline_work is not None:
                        _pipeline_work_call(
                            pipeline_ledger,
                            shared_pipeline_work,
                            "close",
                            phase=phase,
                        )
                        shared_pipeline_work = None
            return output, shared
        finally:
            if pipeline_ledger is not None and shared_pipeline_work is not None:
                _pipeline_work_call(
                    pipeline_ledger,
                    shared_pipeline_work,
                    "close",
                    phase=phase,
                )


def run_switch_with_shared_overlap(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    shared_work: Callable[[], mx.array],
) -> tuple[mx.array, mx.array]:
    """Use streamed miss overlap when supported, otherwise preserve ordering."""

    overlap = getattr(switch_mlp, "run_with_shared_overlap", None)
    if callable(overlap):
        return overlap(x, indices, shared_work)
    return switch_mlp(x, indices), shared_work()


def bind_streamed_switches(model: Any, runtime: ExpertStreamingRuntime) -> int:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model does not expose transformer layers")
    bound = 0
    mapped_store = None
    if runtime.config.slot_layout == "metal-mmap":
        workers_text = os.environ.get("MTPLX_MMAP_WORKERS", "96")
        try:
            workers = int(workers_text)
        except ValueError as exc:
            raise ValueError("MTPLX_MMAP_WORKERS must be an integer") from exc
        mapped_store = MappedExpertStore(
            runtime.root,
            runtime.manifest,
            workers=workers,
        )
        mapped_store.prepare()
        runtime._mapped_expert_store = mapped_store
    for layer_index in runtime.spec.routed_layer_indices:
        layer = layers[layer_index]
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            raise TypeError(f"layer {layer_index} has no switch_mlp seam")
        if mapped_store is None:
            mlp.switch_mlp = HotExpertSwitchGLU(runtime, layer_index)
        else:
            mlp.switch_mlp = MappedExpertSwitchGLU(
                runtime,
                mapped_store,
                layer_index,
            )
        bound += 1
    return bound
