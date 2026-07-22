"""MLX execution adapters for slot-backed affine-quantized routed experts."""

from __future__ import annotations

import gc
import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.activations import swiglu

from mtplx import expert_route_probe as _route_probe

from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_runtime import ExpertStreamingRuntime
from mtplx.expert_manifest import ExpertManifest, ExpertRecord
from mtplx.expert_slots import ExpertSlotBinding, ReadyRoute
from mtplx.expert_streaming import RoutingPhase
from mtplx.expert_streaming_models import (
    MIXED_OFFICIAL_CODEC,
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
)

# Mixed-official (issue #51, M2) per-tier affine bit widths: gate/up is 2-bit
# affine when the layer's tier is "affine2"; down is always 3-bit affine.
_MIXED_AFFINE_BITS = {"affine2": 2, "affine3": 3}
_MIXED_DOWN_BITS = 3
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
# Mixed-official (issue #51 M2b): one transient service bank per gate/up tier so
# partial-residency misses on either tier class land in a matching geometry.
_MIXED_TRANSIENT_LABEL = re.compile(
    rf"mixed-transient-([a-z0-9]+)-({_SLOT_INDEX_PATTERN})"
)
_LAYER_PREFETCH_LABEL = re.compile(
    rf"layer-({_SLOT_INDEX_PATTERN})-prefetch-({_SLOT_INDEX_PATTERN})"
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


class _DeferredSplitClose:
    """Replay a split route's release/close sequence at deferred-flush time.

    The split-route lease bookkeeping (consumer leases, finalize, the layer
    lock) must run through release_miss/close — releasing raw routes would
    corrupt the state machine. This adapter duck-types the ``release``
    surface ``defer_slot_release`` expects and performs today's exact call
    order, one covering eval later.
    """

    def __init__(self, pending: Any, parts: tuple[Any, ...]) -> None:
        self._pending = pending
        self._parts = parts

    def release(self, *, synchronize: bool = False) -> None:
        del synchronize
        first_error: BaseException | None = None
        for part in self._parts:
            try:
                self._pending.release_miss(part)
            except BaseException as exc:  # noqa: BLE001 - drain all parts
                if first_error is None:
                    first_error = exc
        try:
            self._pending.close()
        except BaseException as exc:  # noqa: BLE001 - propagate after close
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error


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
        if segment.dtype == "U16":
            return raw.view(mx.uint16).reshape(segment.shape)
        if segment.dtype == "U8":
            return raw.reshape(segment.shape)
        raise TypeError(f"unsupported streamed component dtype {segment.dtype}")
    view = binding.component_view(component)
    if segment.dtype == "U32":
        host = np.frombuffer(view, dtype=np.dtype("<u4")).reshape(segment.shape)
        value = mx.array(host)
    elif segment.dtype == "BF16":
        host = np.frombuffer(view, dtype=np.dtype("<u2")).reshape(segment.shape)
        value = mx.array(host).view(mx.bfloat16)
    elif segment.dtype == "U16":
        host = np.frombuffer(view, dtype=np.dtype("<u2")).reshape(segment.shape)
        value = mx.array(host)
    elif segment.dtype == "U8":
        host = np.frombuffer(view, dtype=np.dtype("u1")).reshape(segment.shape)
        value = mx.array(host)
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
        elif label.startswith("layer-") and "-prefetch-" in label:
            layer = int(parts[1])
            slot = int(parts[-1])
            count = plan.prefetch_slots_per_layer
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"prefetch slot layer {layer} is not routed")
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
                    "U16": mx.uint16,
                    "U8": mx.uint8,
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
        self.sidecar = manifest.sidecar
        # One mapped path per bank part; ``path`` stays meaningful for the
        # single-part banks every existing artifact uses.
        self.paths = tuple(self.root / part.file for part in manifest.sidecar.parts)
        self.path = self.paths[0]
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
            if not 0 <= record.part < len(self.paths):
                raise ValueError("metal-mmap record names a missing sidecar part")
            part = manifest.sidecar.part(record.part)
            if part.data_start % page_size:
                raise ValueError("metal-mmap sidecar parts must start on a page")
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
            part = self.sidecar.part(record.part)
            base = mmap_u32(
                self.paths[record.part],
                part.data_start + record.sidecar_offset,
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


class DenseIslandStore:
    """Capacity-guaranteed dense per-layer expert banks (issue #63, C5).

    An island layer holds all of its experts resident for the model's
    lifetime in one component-major bank whose row index IS the expert id.
    Router indices therefore address ``gather_qmm`` directly: no
    expert-to-slot translation, no residency probe, no pins, no fences.
    A miss is impossible by construction, so the streamed route machinery
    never runs for these layers.
    """

    def __init__(
        self,
        manifest: ExpertManifest,
        layers: Iterable[int],
        *,
        expert_count: int,
    ) -> None:
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        self.expert_count = int(expert_count)
        if not self.layers:
            raise ValueError("dense island store requires at least one layer")
        records = {
            (record.layer, record.expert): record for record in manifest.records
        }
        self._records: dict[int, tuple[ExpertRecord, ...]] = {}
        self._banks: dict[int, MlxComponentBank] = {}
        self._fill_seconds = 0.0
        self._filled_layers: set[int] = set()
        self._closed = False
        try:
            for layer in self.layers:
                layer_records = []
                for expert in range(self.expert_count):
                    record = records.get((layer, expert))
                    if record is None:
                        raise ValueError(
                            f"manifest has no record for island layer {layer} "
                            f"expert {expert}"
                        )
                    layer_records.append(record)
                self._records[layer] = tuple(layer_records)
                self._banks[layer] = MlxComponentBank(
                    capacity=self.expert_count,
                    record=layer_records[0],
                    label=f"island-layer-{layer}",
                )
        except Exception:
            self.close()
            raise

    @property
    def island_bytes(self) -> int:
        return sum(
            bank.capacity * bank.record_bytes for bank in self._banks.values()
        )

    def fill(
        self,
        manifest: ExpertManifest,
        reader: PositionalExpertReader,
        *,
        verify_hash: bool = True,
    ) -> None:
        """Bulk-read every island expert into its bank row (one-time cost)."""

        if self._closed:
            raise RuntimeError("dense island store is closed")
        started = time.perf_counter()
        for layer in self.layers:
            if layer in self._filled_layers:
                continue
            bank = self._banks[layer]
            items = tuple(
                (
                    record,
                    MlxComponentSlot(
                        bank,
                        expert,
                        label=f"island-layer-{layer}-expert-{expert}",
                    ),
                )
                for expert, record in enumerate(self._records[layer])
            )
            reader.read_component_records_into(
                manifest,
                items,
                verify_hash=verify_hash,
            )
            self._filled_layers.add(layer)
        self._fill_seconds += time.perf_counter() - started

    def bank_for_layer(self, layer: int) -> MlxComponentBank:
        if self._closed:
            raise RuntimeError("dense island store is closed")
        if layer not in self._filled_layers:
            raise RuntimeError(f"island layer {layer} has not been filled")
        return self._banks[layer]

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "dense-island-banks",
            "layers": list(self.layers),
            "expert_count": self.expert_count,
            "island_bytes": self.island_bytes,
            "filled_layers": len(self._filled_layers),
            "fill_seconds": self._fill_seconds,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for bank in self._banks.values():
            bank.close()
        self._banks.clear()
        self._records.clear()
        self._filled_layers.clear()


class DenseIslandSwitchGLU(nn.Module):
    """Expert dispatch for a dense island layer: raw indices, zero host asks.

    The layer's bank row index equals the expert id, so the router's device
    indices are the ``gather_qmm`` rhs_indices as-is. No host sync, no route
    planning, no pin lifecycle: the wave stays inside the lazy graph and
    materializes with the next streamed layer's blocking sync (or the
    row-end fence).
    """

    def __init__(
        self,
        runtime: ExpertStreamingRuntime,
        store: DenseIslandStore,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.layer_index = int(layer_index)
        self.group_size = runtime.spec.quant_group_size
        self.bits = runtime.spec.quant_bits
        self._bank = store.bank_for_layer(self.layer_index)
        # Lazily-built compiled expert gather (issue #51, 70 tps full-residency
        # goal). Full residency rebuilds the 79-layer graph in Python every
        # decode step; compiling the per-layer gather traces it once and
        # replays. The island bank is fully resident and never mutated, so it
        # is safe to capture in the compiled closure. A plain callable is not
        # registered as an nn.Module child.
        # NUMERICS: NOT bitwise vs eager. fp32 matmul compiles exactly, but
        # mx.compile selects a different fused kernel for the quantized
        # gather_qmm that diverges ~0.1-0.6% — non-associative FP, the same
        # class as the vk_k split-K divergence (#171), tagged as a documented
        # FP issue per David's 2026-07-18 ruling. Whether it holds token-sha
        # end-to-end is a guarded-A/B question; if it flips tokens it is a
        # divergent secondary line, not a bug. Default OFF.
        self._compiled_gather = None

    def wave_call(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array,
    ) -> mx.array | None:
        """Fused K3 expert wave over the island bank (issue #65).

        Owns the routing multiply and reduction in the wave kernel's BF16
        combine mode, which bitwise-matches the block's external combine.
        Returns None for any shape the fixed-M4 wave does not cover; the
        caller then runs the classic dispatch unchanged.
        """

        from mtplx.hy3_expert_wave_m4 import (
            HY3_M4_BATCH,
            HY3_M4_ROWS,
            HY3_M4_TOP_K,
            Hy3M4ExpertWaveIneligible,
            hy3_q2_m4_expert_wave,
        )

        spec = self.runtime.spec
        shape = tuple(int(dim) for dim in x.shape)
        if shape != (HY3_M4_BATCH, HY3_M4_ROWS, spec.hidden_size):
            return None
        if tuple(int(dim) for dim in indices.shape) != (
            HY3_M4_BATCH,
            HY3_M4_ROWS,
            HY3_M4_TOP_K,
        ):
            return None
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        try:
            output = hy3_q2_m4_expert_wave(
                x,
                indices.astype(mx.int32),
                scores,
                self._bank.arrays,
                validated_slot_bounds=(0, spec.expert_count - 1),
                combine_mode="bf16",
                hidden_size=spec.hidden_size,
                intermediate_size=spec.expert_hidden_size,
                group_size=self.group_size,
                bits=self.bits,
            )
        except Hy3M4ExpertWaveIneligible:
            return None
        _route_probe.count(
            f"hot.island.wave.{phase.name.lower()}.layer{self.layer_index:02d}"
        )
        return output.hidden_rows

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
        rows = int(tokens.shape[0])
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        _route_probe.count(
            f"hot.island.{phase.name.lower()}.layer{self.layer_index:02d}"
        )
        assignment_inputs = mx.broadcast_to(
            tokens[:, None, :],
            (rows, top_k, hidden_size),
        ).reshape(-1, hidden_size)
        slot_indices = indices.reshape((-1, 1)).astype(mx.int32)
        if os.environ.get("MTPLX_HY3_COMPILE_ISLAND") == "1":
            output = self._compiled_island_gather()(assignment_inputs, slot_indices)
        else:
            output = _gather_component_bank(
                assignment_inputs,
                self._bank,
                slot_indices,
                group_size=self.group_size,
                bits=self.bits,
            )
        return output.reshape((*indices.shape, hidden_size))

    def _compiled_island_gather(self):
        """Build-once compiled expert gather closing over this layer's bank."""
        if self._compiled_gather is None:
            bank = self._bank
            group_size = self.group_size
            bits = self.bits

            def gather(assignment_inputs: mx.array, slot_indices: mx.array) -> mx.array:
                return _gather_component_bank(
                    assignment_inputs,
                    bank,
                    slot_indices,
                    group_size=group_size,
                    bits=bits,
                )

            self._compiled_gather = mx.compile(gather)
        return self._compiled_gather


class BankedMmapBank:
    """Duck-typed component bank whose mapped row index is the expert id."""

    __slots__ = ("arrays",)

    def __init__(self, arrays: dict[str, mx.array]) -> None:
        self.arrays = arrays


class BankedMmapIslandStore:
    """Dense island banks served from a mapped banked sidecar (issue #51, C6).

    Same dispatch contract as ``DenseIslandStore`` — bank row == expert id,
    raw router indices feed ``gather_qmm`` — but physical residency belongs
    to the macOS pager: the banked file regions are mapped into Metal
    without copies, pages arrive on first touch, and the page cache keeps
    or evicts them under normal kernel policy. No slot pool, no reads, no
    misses exist for these layers. ``prepare`` verifies each region hash by
    reading the file once, which doubles as the page-cache warmup.
    """

    def __init__(
        self,
        banked_manifest: Path | str,
        layers: Iterable[int],
        *,
        expert_count: int,
        verify_hash: bool = True,
        wired: bool = True,
    ) -> None:
        from mtplx.expert_banked import BankedManifestError, load_banked_manifest

        self.wired = bool(wired)
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        if not self.layers:
            raise ValueError("banked island store requires at least one layer")
        self._banked = load_banked_manifest(banked_manifest)
        # ``none`` maps raw bank bytes into Metal zero-copy; ``rans32x-v1``
        # reads the compressed region and rebuilds each bank with the in-kernel
        # rANS decoder in ``prepare`` (issue #51, C7). Any other codec has no
        # decode path here.
        if self._banked.codec not in ("none", "rans32x-v1"):
            raise BankedManifestError(
                f"banked codec {self._banked.codec!r} has no island decode path"
            )
        self._codec = self._banked.codec
        if self._banked.expert_count != int(expert_count):
            raise BankedManifestError(
                f"banked manifest holds {self._banked.expert_count} experts "
                f"per layer; the model routes {expert_count}"
            )
        missing = [
            layer for layer in self.layers if layer not in self._banked.layer_set
        ]
        if missing:
            raise BankedManifestError(
                f"banked manifest does not cover layers {missing}"
            )
        self.expert_count = int(expert_count)
        self._verify_hash = bool(verify_hash)
        self._banks: dict[int, BankedMmapBank] = {}
        self._bases: list[mx.array] = []
        self._verify_seconds = 0.0
        self._closed = False

    def prepare(self) -> None:
        from mtplx.expert_banked import BankedManifestError

        if self._closed:
            raise RuntimeError("banked island store is closed")
        if self._banks:
            return
        path = self._banked.bin_path()
        alignment = self._banked.alignment
        file_size = path.stat().st_size
        for layer in self.layers:
            entry = self._banked.layer_entry(layer)
            if self._verify_hash:
                started = time.perf_counter()
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    handle.seek(entry.offset)
                    remaining = entry.length
                    while remaining:
                        chunk = handle.read(min(remaining, 8 << 20))
                        if not chunk:
                            raise BankedManifestError(
                                f"banked layer {layer} region is truncated"
                            )
                        digest.update(chunk)
                        remaining -= len(chunk)
                if digest.hexdigest() != entry.sha256:
                    raise BankedManifestError(
                        f"banked layer {layer} region hash mismatch"
                    )
                self._verify_seconds += time.perf_counter() - started
            if self._codec != "none":
                self._banks[layer] = BankedMmapBank(
                    self._decode_layer_banks(path, entry)
                )
                continue
            mapped_length = -(-entry.length // alignment) * alignment
            if entry.offset + mapped_length > file_size:
                raise BankedManifestError(
                    f"banked layer {layer} extent exceeds {path.name}"
                )
            # wired=True registers the mapping in MLX's process-wide residency
            # set: Metal keeps it permanently resident like ordinary weights.
            # Untracked (unwired) buffers make Metal rebuild residency around
            # every submission — measured as the ENTIRE MLX residency set
            # wiring/unwiring (~85 GiB swings) per wave, ~4x decode slowdown.
            # Unwired remains selectable for bands larger than RAM (Q4/GLM).
            base = mmap_u32(path, entry.offset, mapped_length, wired=self.wired)
            arrays: dict[str, mx.array] = {}
            for component in entry.components:
                if component.dtype == "U32":
                    typed = base
                    item_size = 4
                elif component.dtype == "BF16":
                    typed = mx.view(base, mx.bfloat16)
                    item_size = 2
                else:
                    raise TypeError(
                        f"unsupported banked component dtype {component.dtype}"
                    )
                if component.offset % item_size:
                    raise BankedManifestError(
                        f"banked component {component.component} is not "
                        "dtype-aligned"
                    )
                arrays[component.component] = mx.as_strided(
                    typed,
                    shape=(self.expert_count, *component.shape),
                    offset=component.offset // item_size,
                )
            self._banks[layer] = BankedMmapBank(arrays)
            self._bases.append(base)

    def _decode_layer_banks(self, path, entry) -> dict[str, mx.array]:
        """Rebuild one layer's component banks from its compressed region.

        Reads the entropy-coded region, runs the in-kernel rANS decoder per
        component, and materializes each bank as ``[expert_count, *shape]`` in
        its native dtype — row index == expert id, identical to the mmap path.
        Decode happens once at prepare; the banks are then plain resident
        arrays consumed by ``gather_qmm`` with no per-call host work.
        """

        from mtplx.expert_banked import BankedManifestError
        from mtplx.expert_rans_metal import decode_container

        with path.open("rb") as handle:
            handle.seek(entry.offset)
            region = handle.read(entry.length)
        if len(region) != entry.length:
            raise BankedManifestError(
                f"banked layer {entry.layer} region is truncated"
            )
        arrays: dict[str, mx.array] = {}
        for component in entry.components:
            blob = region[component.offset : component.offset + component.length]
            decoded = decode_container(blob)  # uint8[expert_count * seg_len]
            if component.dtype == "U32":
                typed = mx.view(decoded, mx.uint32)
            elif component.dtype == "BF16":
                typed = mx.view(decoded, mx.bfloat16)
            else:
                raise TypeError(
                    f"unsupported banked component dtype {component.dtype}"
                )
            bank = typed.reshape(self.expert_count, *component.shape)
            mx.eval(bank)  # materialize; releases the compressed payload graph
            arrays[component.component] = bank
        return arrays

    def prefetch_all(self) -> int:
        """Issue one MADV_WILLNEED batch across every mapped layer region."""

        from mtplx.mmap_mlx import prefetch

        if not self._bases:
            return 0
        return prefetch(list(self._bases))

    def bank_for_layer(self, layer: int) -> BankedMmapBank:
        if self._closed:
            raise RuntimeError("banked island store is closed")
        bank = self._banks.get(layer)
        if bank is None:
            raise RuntimeError(f"banked island layer {layer} is not prepared")
        return bank

    def snapshot(self) -> dict[str, Any]:
        stored_bytes = sum(
            self._banked.layer_entry(layer).length for layer in self.layers
        )
        resident_bytes = sum(
            component.raw_bytes
            for layer in self.layers
            for component in self._banked.layer_entry(layer).components
        )
        return {
            "backend": (
                "banked-mmap-island-banks"
                if self._codec == "none"
                else "banked-rans-island-banks"
            ),
            "codec": self._codec,
            "wired": self.wired,
            "layers": list(self.layers),
            "expert_count": self.expert_count,
            "mapped_bytes": stored_bytes,
            "resident_bytes": resident_bytes,
            "verify_seconds": self._verify_seconds,
            "prepared_layers": len(self._banks),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._banks.clear()
        self._bases.clear()
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

    expert_codec = getattr(spec, "expert_codec", "affine")
    is_mixed = expert_codec == MIXED_OFFICIAL_CODEC
    routed_layers = set(spec.routed_layer_indices)
    exemplar_layer = spec.routed_layer_indices[0]
    record_by_gate_up_tier: dict[str, ExpertRecord] = {}

    if is_mixed:
        # Per-layer bank geometry (D3): every routed layer has its OWN expected
        # signature, derived from that layer's tier via the single source of
        # truth (plan_record_segments), and enforced within the layer. There is
        # no cross-layer exemplar because t158 and affine2 layers differ.
        from mtplx.expert_mixed_official import plan_record_segments

        mixed_dims = {
            "gate_proj": (spec.expert_hidden_size, spec.hidden_size),
            "up_proj": (spec.expert_hidden_size, spec.hidden_size),
            "down_proj": (spec.hidden_size, spec.expert_hidden_size),
        }
        # One exemplar record per gate/up tier, for the per-tier transient banks.
        layer_signatures: dict[int, tuple[tuple[Any, ...], ...]] = {}
        for layer in spec.routed_layer_indices:
            gate_up_tier, down_tier = manifest.mixed_tier_for_layer(layer)
            record_by_gate_up_tier.setdefault(gate_up_tier, record_by_layer[layer])
            planned = plan_record_segments(
                {"gate_up": gate_up_tier, "down": down_tier}, mixed_dims
            )
            layer_signatures[layer] = tuple(
                (seg.component, seg.dtype, tuple(seg.shape), seg.length)
                for seg in planned
            )
            if component_signature(record_by_layer[layer]) != layer_signatures[layer]:
                raise ValueError(
                    f"manifest component geometry for layer {layer} does not "
                    "match the mixed-official tier layout"
                )
        for record in manifest.records:
            if (
                record.layer in routed_layers
                and component_signature(record) != layer_signatures[record.layer]
            ):
                raise ValueError(
                    "routed-layer component geometry differs for expert "
                    f"({record.layer}, {record.expert}) from its layer's tier"
                )
    else:
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
            if expert_codec != "affine":
                # Shadow-codec (q1) records: packed sign/trit words plus one
                # bf16-bit scale per g64 group, no bias leaf (gate 3 of
                # research/streamed-q1-codec-gap-analysis.md).
                from mtplx.expert_shadow import (
                    SHADOW_GROUP,
                    _B1_WORDS_PER_GROUP,
                    _T158_BYTES_PER_GROUP,
                )

                groups = input_size // SHADOW_GROUP
                if expert_codec == "b1":
                    packed_dtype = "U32"
                    packed_shape = (output_size, groups * _B1_WORDS_PER_GROUP)
                    packed_length = output_size * groups * _B1_WORDS_PER_GROUP * 4
                else:
                    packed_dtype = "U8"
                    packed_shape = (output_size, groups * _T158_BYTES_PER_GROUP)
                    packed_length = output_size * groups * _T158_BYTES_PER_GROUP
                expected_signature.extend(
                    (
                        (
                            f"{projection}.packed",
                            packed_dtype,
                            packed_shape,
                            packed_length,
                        ),
                        (
                            f"{projection}.scales",
                            "U16",
                            (output_size, groups),
                            output_size * groups * 2,
                        ),
                    )
                )
                continue
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

        exemplar_signature = component_signature(record_by_layer[exemplar_layer])
        if exemplar_signature != tuple(expected_signature):
            raise ValueError(
                "manifest component geometry does not match the model descriptor"
            )
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

    banks: dict[tuple[str, object], MlxComponentBank] = {}
    slots: dict[str, MlxComponentSlot] = {}
    backend = "mlx-metal-component-banks"

    def bank_for(kind: str, discriminator: object = -1) -> MlxComponentBank:
        keyed = {"persistent", "prefetch", "mixed-transient"}
        key = (kind, discriminator if kind in keyed else -1)
        bank = banks.get(key)
        if bank is not None:
            return bank
        if kind == "persistent":
            capacity = plan.slots_per_layer
            record = record_by_layer[discriminator]
            label = f"layer-{discriminator}-persistent-bank"
        elif kind == "prefetch":
            capacity = plan.prefetch_slots_per_layer
            record = record_by_layer[discriminator]
            label = f"layer-{discriminator}-prefetch-bank"
        elif kind == "mixed-transient":
            # One transient bank per gate/up tier (issue #51 M2b): a miss on a
            # differing-tier layer lands in a bank of ITS geometry.
            capacity = plan.transient_slots
            record = record_by_gate_up_tier[discriminator]
            label = f"mixed-transient-bank-{discriminator}"
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
        # The per-record byte size is validated against the resolved bank below:
        # for mixed banks it is the layer's (or transient exemplar's) record
        # size, not a single uniform ``spec.expert_record_bytes`` (which raises).
        layer_persistent = _LAYER_PERSISTENT_LABEL.fullmatch(label)
        global_persistent = _GLOBAL_PERSISTENT_LABEL.fullmatch(label)
        global_transient = _GLOBAL_TRANSIENT_LABEL.fullmatch(label)
        mixed_transient = _MIXED_TRANSIENT_LABEL.fullmatch(label)
        layer_prefetch = _LAYER_PREFETCH_LABEL.fullmatch(label)
        if mixed_transient is not None:
            tier = mixed_transient.group(1)
            slot_index = int(mixed_transient.group(2))
            if tier not in record_by_gate_up_tier:
                raise ValueError(f"mixed transient tier {tier!r} is not routed")
            if not 0 <= slot_index < plan.transient_slots:
                raise ValueError("transient slot is outside planned capacity")
            bank = bank_for("mixed-transient", tier)
        elif layer_prefetch is not None:
            if plan.cache_scope != "layer":
                raise ValueError(
                    "prefetch slot label conflicts with global cache scope"
                )
            layer = int(layer_prefetch.group(1))
            slot_index = int(layer_prefetch.group(2))
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"prefetch slot layer {layer} is not routed")
            if not 0 <= slot_index < plan.prefetch_slots_per_layer:
                raise ValueError("prefetch slot is outside planned capacity")
            bank = bank_for("prefetch", layer)
        elif layer_persistent is not None:
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
        if int(size) != bank.record_bytes:
            raise ValueError("slot allocator size differs from the bank record")
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
    slot_indices = mx.array(
        [int(binding.buffer.bank_index) for binding in bindings],
        dtype=mx.int32,
    ).reshape((-1, 1))
    return _gather_component_bank(
        x,
        bank,
        slot_indices,
        group_size=group_size,
        bits=bits,
    )


def _gather_component_bank(
    x: mx.array,
    bank: MlxComponentBank,
    slot_indices: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    """Row-gathered three-matrix expert MLP against one component bank."""

    rows = int(x.shape[0])
    selected = x.reshape((rows, 1, 1, int(x.shape[-1])))

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
    return output.reshape((rows, int(output.shape[-1])))


def _run_component_bank_shadow(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    codec: str,
) -> mx.array:
    """Execute assignment-aligned rows of a shadow-codec (q1) slot bank.

    The q2 lane's equal (gate 4 of the gap analysis): records were read
    into component-bank slots by the ordinary slot machinery, and the rows
    execute through ``shadow_gather_mm`` — the bank row fed to the kernel
    is the slot index, not the expert id.  Eager-only, like the miss-shadow
    lane: the shadow kernel is not traceable under ``mx.compile``.
    """

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")
    slot_rows = mx.array(
        [int(binding.buffer.bank_index) for binding in bindings],
        dtype=mx.int32,
    )
    return _shadow_gather_component_bank(x, bank, slot_rows, codec=codec)


def _shadow_gather_component_bank(
    x: mx.array,
    bank: MlxComponentBank,
    slot_rows: mx.array,
    *,
    codec: str,
) -> mx.array:
    """Row-gathered three-projection shadow MLP against one component bank."""

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    def projection(values: mx.array, name: str) -> mx.array:
        return shadow_gather_mm(
            values,
            slot_rows,
            bank.arrays[f"{name}.packed"],
            bank.arrays[f"{name}.scales"],
            codec=codec,
        )

    gate = projection(x, "gate_proj")
    up = projection(x, "up_proj")
    return projection(swiglu(gate, up), "down_proj")


def _run_component_bank_mixed(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    gate_up_tier: str,
    group_size: int,
) -> mx.array:
    """Execute a mixed-official (issue #51, M2) component-bank wave.

    Per-projection-group dispatch (D4): the layer's gate/up pair runs through
    ``shadow_gather_mm`` (t158-tier) or ``gather_qmm(bits=2)`` (affine2-tier),
    and ``down`` always runs through ``gather_qmm(bits=3)`` — all in one pass,
    against one shared per-layer bank. Eager-only, like the shadow lane.
    """

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")
    slot_rows = [int(binding.buffer.bank_index) for binding in bindings]
    slot_rows_1d = mx.array(slot_rows, dtype=mx.int32)
    slot_indices_2d = mx.array(slot_rows, dtype=mx.int32).reshape((-1, 1))
    return _gather_component_bank_mixed(
        x,
        bank,
        slot_indices_2d,
        slot_rows_1d,
        gate_up_tier=gate_up_tier,
        group_size=group_size,
    )


def _gather_component_bank_mixed(
    x: mx.array,
    bank: MlxComponentBank,
    slot_indices_2d: mx.array,
    slot_rows_1d: mx.array,
    *,
    gate_up_tier: str,
    group_size: int,
) -> mx.array:
    """Row-gathered mixed-tier expert MLP against one shared per-layer bank."""

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    rows = int(x.shape[0])
    in_dim = int(x.shape[-1])

    def affine(values: mx.array, projection: str, bits: int) -> mx.array:
        selected = values.reshape((rows, 1, 1, int(values.shape[-1])))
        out = mx.gather_qmm(
            selected,
            bank.arrays[f"{projection}.weight"],
            bank.arrays[f"{projection}.scales"],
            bank.arrays[f"{projection}.biases"],
            rhs_indices=slot_indices_2d,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        return out.reshape((rows, int(out.shape[-1])))

    def shadow(values: mx.array, projection: str) -> mx.array:
        return shadow_gather_mm(
            values,
            slot_rows_1d,
            bank.arrays[f"{projection}.packed"],
            bank.arrays[f"{projection}.scales"],
            codec="t158",
        )

    if gate_up_tier == "t158":
        gate = shadow(x.reshape((rows, in_dim)), "gate_proj")
        up = shadow(x.reshape((rows, in_dim)), "up_proj")
    elif gate_up_tier == "affine2":
        bits = _MIXED_AFFINE_BITS["affine2"]
        gate = affine(x, "gate_proj", bits)
        up = affine(x, "up_proj", bits)
    else:  # pragma: no cover - guarded by the manifest tier contract
        raise ValueError(f"unsupported mixed gate/up tier {gate_up_tier!r}")
    hidden = swiglu(gate, up)
    return affine(hidden, "down_proj", _MIXED_DOWN_BITS)


def _run_shadow_bank(
    x: mx.array,
    expert_rows: mx.array,
    bank: Any,
) -> mx.array:
    """Three-projection expert MLP against a low-precision shadow bank.

    ``expert_rows`` are raw expert ids — the shadow bank row index is the
    expert id (island-bank convention). Output is a quality tier, close to
    the exact quantized path but not bitwise. Eager-only: the shadow
    kernel is not traceable under ``mx.compile``.
    """

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    def projection(values: mx.array, name: str) -> mx.array:
        return shadow_gather_mm(
            values,
            expert_rows,
            bank.arrays[f"{name}.packed"],
            bank.arrays[f"{name}.scales"],
            codec=bank.codec,
        )

    gate = projection(x, "gate_proj")
    up = projection(x, "up_proj")
    return projection(swiglu(gate, up), "down_proj")


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
        # Streamed record codec (issue #51 q1 lane): "affine" executes
        # slot-resident records through gather_qmm; a shadow codec ("b1",
        # "t158") executes them through shadow_gather_mm instead.  The read
        # path is identical either way — only the component-bank dispatch
        # differs (gate 4 of the gap analysis).
        self.codec = getattr(runtime.spec, "expert_codec", "affine")
        # Mixed-official (issue #51, M2): resolve this layer's (gate_up, down)
        # tier from the loaded manifest's layer_tier_map — never the spec (D1).
        # Fail closed if the layer has no tier entry.
        self._gate_up_tier: str | None = None
        self._down_tier: str | None = None
        if self.codec == MIXED_OFFICIAL_CODEC:
            self._gate_up_tier, self._down_tier = (
                runtime.manifest.mixed_tier_for_layer(self.layer_index)
            )
        # Shadow miss fallback (issue #51): when the runtime opened with
        # miss_shadow, decode misses on this streamed layer are served from
        # a resident low-precision bank instead of waiting on SSD.
        shadow_lookup = getattr(runtime, "shadow_bank_for_layer", None)
        self._shadow_bank = (
            shadow_lookup(self.layer_index) if callable(shadow_lookup) else None
        )

    def _dispatch_component_bank(
        self,
        selected: mx.array,
        bindings: tuple[ExpertSlotBinding, ...],
    ) -> mx.array:
        """Execute one component-bank wave under this layer's record codec."""

        if self.codec == "affine":
            return _run_component_bank_q4(
                selected,
                bindings,
                group_size=self.group_size,
                bits=self.bits,
            )
        if self.codec == MIXED_OFFICIAL_CODEC:
            assert self._gate_up_tier is not None
            return _run_component_bank_mixed(
                selected,
                bindings,
                gate_up_tier=self._gate_up_tier,
                group_size=self.group_size,
            )
        return _run_component_bank_shadow(selected, bindings, codec=self.codec)

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
        # Shared-branch hoist (issue #51, flag-gated, exact-quality). Normally
        # the resident shared MLP is submitted AFTER begin_split_route so it
        # overlaps the miss reads (~0.44 ms). But the router host-sync below
        # (mx.eval(indices), the per-streamed-layer barrier, p50 ~1 ms) is the
        # bigger exposed cost, and the shared branch depends only on x — not on
        # the routed indices — so it can be dispatched HERE to fill the GPU-idle
        # window of the sync round-trip instead. Pure execution reorder: same
        # shared_mlp(x), same combine, bitwise-identical output. Decode single
        # token only; requires async_eval so the dispatch does not itself block.
        _hoisted_shared: mx.array | None = None
        if (
            shared_work is not None
            and int(x.shape[-2]) == 1
            and os.environ.get("MTPLX_HY3_SHARED_HOIST") == "1"
        ):
            _async_eval = getattr(mx, "async_eval", None)
            if callable(_async_eval):
                _hoisted_shared = shared_work()
                _async_eval(_hoisted_shared)
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
        # Seed from the hoist (above): when set, the shared branch is already
        # computed and submitted, so the late dispatch sites skip recomputing it.
        shared: mx.array | None = _hoisted_shared

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
            defer: bool = False,
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
                    self._dispatch_component_bank(selected, grouped_bindings)
                )
                wave_positions.extend(grouped_positions)
            if defer:
                # No release obligation here (the lease replay is queued),
                # but the GPU still needs the part submitted now — without
                # it the device idles through every miss wait.
                async_eval = getattr(mx, "async_eval", None)
                if callable(async_eval):
                    async_eval(wave_outputs)
                else:
                    mx.eval(wave_outputs)
            else:
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
            defer: bool = False,
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
            if defer:
                # No release obligation here (the lease replay is queued),
                # but the GPU still needs the part submitted now — without
                # it the device idles through every miss wait.
                async_eval = getattr(mx, "async_eval", None)
                if callable(async_eval):
                    async_eval(wave_outputs)
                else:
                    mx.eval(wave_outputs)
            else:
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
            waves = tuple(
                self.runtime.route_waves(
                    expert_ids,
                    sort_unique=(
                        phase is RoutingPhase.PREFILL
                        and self.runtime.manifest.sidecar is not None
                    ),
                )
            )
            # Deferring hands this layer's lock and slot pins to the next
            # generation-thread flush, so it is only legal on the wave that
            # ends the layer's routing.  A wide verify batch splits into
            # several waves, and the layer lock is not reentrant: a deferred
            # non-final wave would park the generation thread on its own lock
            # when the next wave re-entered the same layer (issue #120,
            # GLM depth>=4 at 6-row verify / 32 transient slots).
            final_wave = len(waves) - 1
            for wave_index, wave in enumerate(waves):
                # Shadow miss fallback (issue #51): with a shadow bank bound,
                # a decode wave never waits on SSD.  Resident experts run
                # exactly from cache; every other assignment is served from
                # the resident low-precision shadow bank, and the exact tier
                # warms asynchronously through the speculative-prefetch lane
                # so future routes turn these serves into exact hits.
                # Prefill is untouched (exact, transient-served).
                shadow_bank = self._shadow_bank
                resident: frozenset[int] = frozenset()
                if shadow_bank is not None and phase is RoutingPhase.DECODE:
                    resident = self.runtime.peek_resident_experts(
                        self.layer_index, wave.experts
                    )
                if (
                    shadow_bank is not None
                    and phase is RoutingPhase.DECODE
                    and not all(expert in resident for expert in wave.experts)
                ):
                    exact_assignments = tuple(
                        (position, expert)
                        for position, expert in zip(
                            wave.positions, wave.experts, strict=True
                        )
                        if expert in resident
                    )
                    shadow_positions = [
                        position
                        for position, expert in zip(
                            wave.positions, wave.experts, strict=True
                        )
                        if expert not in resident
                    ]
                    shadow_experts = [
                        expert for expert in wave.experts if expert not in resident
                    ]
                    if exact_assignments:
                        with _route_probe.bracket("hot.begin_shadow_route"):
                            pending = self.runtime.begin_split_route(
                                self.layer_index,
                                tuple(
                                    expert for _position, expert in exact_assignments
                                ),
                                phase=phase,
                            )
                        try:
                            hit_set = set(pending.plan.hits)
                            hit_positions = tuple(
                                position
                                for position, expert in exact_assignments
                                if expert in hit_set
                            )
                            if pending.hit_ready is not None:
                                evaluate_component_bindings(
                                    hit_positions,
                                    pending.hit_ready.bindings,
                                    pending.hit_ready,
                                    force_sync=True,
                                )
                                pending.release_hits()
                            # The residency peek is a snapshot, not a
                            # reservation: an expert evicted between peek and
                            # pin becomes a planned miss here and is serviced
                            # exactly (slower on this route, never wrong).
                            for miss_ready in pending.iter_ready_misses():
                                try:
                                    ready_experts = set(miss_ready.plan.experts)
                                    miss_positions = tuple(
                                        position
                                        for position, expert in exact_assignments
                                        if expert not in hit_set
                                        and expert in ready_experts
                                    )
                                    evaluate_component_bindings(
                                        miss_positions,
                                        miss_ready.bindings,
                                        miss_ready,
                                        force_sync=True,
                                    )
                                finally:
                                    pending.release_miss(miss_ready)
                        except BaseException as exc:
                            pending.abort(exc)
                            raise
                        finally:
                            pending.close()
                    with _route_probe.bracket("hot.shadow_serve"):
                        token_positions = mx.array(
                            [position // top_k for position in shadow_positions],
                            dtype=mx.int32,
                        )
                        selected = mx.take(tokens, token_positions, axis=0)
                        outputs.append(
                            _run_shadow_bank(
                                selected,
                                mx.array(shadow_experts, dtype=mx.int32),
                                shadow_bank,
                            )
                        )
                        output_positions.extend(shadow_positions)
                    unique_shadowed = tuple(dict.fromkeys(shadow_experts))
                    _route_probe.count("hot.shadow_route")
                    self.runtime.note_shadow_serve(
                        self.layer_index,
                        assignments=len(shadow_positions),
                        experts=len(unique_shadowed),
                    )
                    self.runtime.prefetch_experts(
                        self.layer_index, unique_shadowed
                    )
                    continue
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
                    outcome = "all_hit" if ready is not None else "split_route"
                    _route_probe.count(f"hot.{outcome}")
                    _route_probe.count(
                        f"hot.{outcome}.{phase.name.lower()}"
                        f".layer{self.layer_index:02d}"
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
                                wave_output = self._dispatch_component_bank(
                                    assignment_inputs,
                                    ready.bindings,
                                )
                            # Slot pins may be released only after the lazy graph
                            # has consumed the currently bound bank generations.
                            # Deferred mode: the next generation-thread eval is
                            # that consumption proof; no per-layer fence runs.
                            deferred_release = False
                            # Promoted default via ExpertStreamingConfig after
                            # the C3 matrix; fakes without the field keep the
                            # fence path.
                            if wave_index == final_wave and getattr(
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
                # Deferred split release: no per-part fence; lease
                # bookkeeping replays at the next generation-thread eval
                # (the same coverage proof deferred pins already rely on).
                # A deferred split also keeps the layer lock until that
                # flush, so only the final wave may take this path.
                deferred_split = (
                    wave_index == final_wave
                    and phase is RoutingPhase.DECODE
                    and getattr(
                        self.runtime.config, "split_route_release", "fenced"
                    )
                    == "deferred"
                    and getattr(self.runtime.config, "deferred_pin_release", False)
                )
                deferred_parts: list[ReadyRoute] = []
                split_completed = False
                wave_output_start = len(outputs)
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
                                defer=deferred_split,
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
                    if pending.hit_ready is not None and not deferred_split:
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
                            # Submit the shared branch so the GPU has work
                            # during miss I/O. Deferred-pin mode already
                            # trusts the next generation-thread eval for
                            # coverage, so the blocking barrier serves no
                            # release obligation there; strict mode keeps it.
                            async_eval = getattr(mx, "async_eval", None)
                            if (
                                getattr(
                                    self.runtime.config,
                                    "deferred_pin_release",
                                    False,
                                )
                                and callable(async_eval)
                            ):
                                async_eval(shared)
                            else:
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
                                defer=deferred_split,
                            )
                        except BaseException as exc:
                            part_error = exc
                            raise
                        finally:
                            if deferred_split and part_error is None:
                                deferred_parts.append(miss_ready)
                            else:
                                try:
                                    pending.release_miss(miss_ready)
                                except BaseException:
                                    if part_error is None:
                                        raise
                    split_completed = True
                except BaseException as exc:
                    pending.abort(exc)
                    raise
                finally:
                    if deferred_split and split_completed:
                        self.runtime.defer_slot_release(
                            _DeferredSplitClose(
                                pending, tuple(deferred_parts)
                            ),
                            tuple(outputs[wave_output_start:]),
                        )
                    else:
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
                    if shared is None:
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
    island_layers = frozenset(runtime.config.island_layers)
    island_store = None
    if island_layers:
        island_store = DenseIslandStore(
            runtime.manifest,
            island_layers,
            expert_count=runtime.spec.expert_count,
        )
        island_store.fill(
            runtime.manifest,
            runtime.reader,
            verify_hash=(
                runtime.config.verify_record_hashes
                and not runtime.config.verify_sidecar_hash_at_open
            ),
        )
        runtime._island_store = island_store
    banked_layers = frozenset(runtime.config.mmap_island_layers)
    banked_store = None
    if banked_layers:
        banked_store = BankedMmapIslandStore(
            Path(runtime.config.banked_manifest),
            banked_layers,
            expert_count=runtime.spec.expert_count,
            wired=runtime.config.mmap_island_wired,
        )
        banked_store.prepare()
        banked_store.prefetch_all()
        runtime._banked_island_store = banked_store
    for layer_index in runtime.spec.routed_layer_indices:
        layer = layers[layer_index]
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            raise TypeError(f"layer {layer_index} has no switch_mlp seam")
        if island_store is not None and layer_index in island_layers:
            mlp.switch_mlp = DenseIslandSwitchGLU(
                runtime,
                island_store,
                layer_index,
            )
        elif banked_store is not None and layer_index in banked_layers:
            mlp.switch_mlp = DenseIslandSwitchGLU(
                runtime,
                banked_store,
                layer_index,
            )
        elif mapped_store is None:
            mlp.switch_mlp = HotExpertSwitchGLU(runtime, layer_index)
        else:
            mlp.switch_mlp = MappedExpertSwitchGLU(
                runtime,
                mapped_store,
                layer_index,
            )
        bound += 1
    return bound
