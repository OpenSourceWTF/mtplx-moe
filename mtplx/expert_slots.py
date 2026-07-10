"""Fixed expert slot buffers with generation-safe load and pin lifetimes."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from .expert_io import ExpertIOError, PositionalExpertReader
from .expert_manifest import ExpertManifest, ExpertManifestError, ExpertRecord
from .expert_streaming import RoutePlan, SlotLoad
from .expert_streaming_models import (
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
)


class ExpertSlotError(RuntimeError):
    pass


class ExpertSlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class ExpertSlotMetrics:
    ensure_calls: int = 0
    load_requests: int = 0
    owned_loads: int = 0
    deduplicated_loads: int = 0
    ready_hits: int = 0
    load_failures: int = 0
    generation_replacements: int = 0
    pin_waits: int = 0
    load_waits: int = 0
    active_routes: int = 0
    active_routes_peak: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **values: int) -> None:
        with self._lock:
            for name, value in values.items():
                setattr(self, name, int(getattr(self, name)) + int(value))
            self.active_routes_peak = max(self.active_routes_peak, self.active_routes)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                name: int(getattr(self, name))
                for name in (
                    "ensure_calls",
                    "load_requests",
                    "owned_loads",
                    "deduplicated_loads",
                    "ready_hits",
                    "load_failures",
                    "generation_replacements",
                    "pin_waits",
                    "load_waits",
                    "active_routes",
                    "active_routes_peak",
                )
            }


@dataclass
class _PhysicalSlot:
    label: str
    buffer: Any
    state: ExpertSlotState = ExpertSlotState.EMPTY
    layer: int | None = None
    expert: int | None = None
    generation: int = 0
    pins: int = 0
    digest: str | None = None
    error: BaseException | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)


@dataclass(frozen=True)
class ExpertSlotBinding:
    layer: int
    expert: int
    logical_slot: int
    generation: int
    record: ExpertRecord
    buffer: Any

    def component_view(self, component: str) -> memoryview:
        view = memoryview(self.buffer)
        if view.readonly or not view.c_contiguous:
            raise ExpertSlotError("slot buffer is not a writable contiguous buffer")
        raw = view.cast("B")
        cursor = 0
        for segment in self.record.segments:
            end = cursor + segment.length
            if segment.component == component:
                return raw[cursor:end]
            cursor = end
        raise KeyError(component)


class ReadyRoute:
    """Pinned slot bindings that remain valid until explicit release."""

    def __init__(
        self,
        pool: ExpertSlotPool,
        plan: RoutePlan,
        bindings: tuple[ExpertSlotBinding, ...],
        pinned: tuple[_PhysicalSlot, ...],
    ) -> None:
        self.pool = pool
        self.plan = plan
        self.bindings = bindings
        self._pinned = pinned
        self._released = False

    @property
    def slots(self) -> tuple[int, ...]:
        return self.plan.slots

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    def validate(self) -> None:
        if self._released:
            raise ExpertSlotError("ready route has already been released")
        for binding, slot in zip(self.bindings, self._binding_slots(), strict=True):
            with slot.condition:
                if (
                    slot.state is not ExpertSlotState.READY
                    or slot.layer != binding.layer
                    or slot.expert != binding.expert
                    or slot.generation != binding.generation
                ):
                    raise ExpertSlotError(
                        "ready route references a stale slot generation"
                    )

    def _binding_slots(self) -> tuple[_PhysicalSlot, ...]:
        return tuple(
            self.pool._physical(binding.layer, binding.logical_slot)
            for binding in self.bindings
        )

    def release(self, *, synchronize: bool = True) -> None:
        if self._released:
            return
        if synchronize and self.pool.device_synchronize is not None:
            self.pool.device_synchronize()
        for slot in self._pinned:
            with slot.condition:
                if slot.pins <= 0:
                    raise ExpertSlotError("slot pin accounting underflow")
                slot.pins -= 1
                slot.condition.notify_all()
        self._released = True
        self.pool._route_released()

    def __enter__(self) -> ReadyRoute:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _CombinedCancel:
    def __init__(
        self, external: threading.Event | None, internal: threading.Event
    ) -> None:
        self.external = external
        self.internal = internal

    def is_set(self) -> bool:
        return self.internal.is_set() or (
            self.external is not None and self.external.is_set()
        )


class ExpertSlotPool:
    """Own all persistent and transient record buffers for one model."""

    def __init__(
        self,
        spec: ExpertStreamingModelSpec,
        plan: ExpertMemoryPlan,
        manifest: ExpertManifest,
        reader: PositionalExpertReader,
        *,
        buffer_allocator: Callable[[int, str], Any] | None = None,
        max_inflight_io_bytes: int | None = None,
        prefer_sidecar: bool = True,
        verify_hashes: bool = True,
        device_synchronize: Callable[[], None] | None = None,
    ) -> None:
        if plan.model_key != spec.key or manifest.model_key != spec.key:
            raise ValueError("spec, memory plan, and manifest model keys must match")
        if (
            manifest.quant_bits != spec.quant_bits
            or manifest.quant_group_size != spec.quant_group_size
        ):
            raise ValueError("manifest quantization does not match model descriptor")
        if manifest.routed_expert_bytes != spec.routed_expert_bytes:
            raise ValueError("manifest routed bytes do not match model descriptor")
        if plan.transient_slots < spec.top_k:
            raise ValueError("memory plan transient slots do not cover model top-k")
        if max_inflight_io_bytes is None:
            max_inflight_io_bytes = max(plan.transient_bytes, spec.expert_record_bytes)
        if (
            isinstance(max_inflight_io_bytes, bool)
            or not isinstance(max_inflight_io_bytes, int)
            or max_inflight_io_bytes < spec.expert_record_bytes
        ):
            raise ValueError(
                "max_inflight_io_bytes must cover at least one expert record"
            )
        self.spec = spec
        self.plan = plan
        self.manifest = manifest
        self.reader = reader
        self.prefer_sidecar = prefer_sidecar
        self.verify_hashes = verify_hashes
        self.device_synchronize = device_synchronize
        self.metrics = ExpertSlotMetrics()
        self._allocator = buffer_allocator or (lambda size, _label: bytearray(size))
        self.buffer_backend = str(
            getattr(self._allocator, "backend", "python-bytearray")
        )
        self._persistent: dict[tuple[int, int], _PhysicalSlot] = {}
        self._transient: tuple[_PhysicalSlot, ...]
        self._record_map = {
            (record.layer, record.expert): record for record in manifest.records
        }
        self._ensure_locks = {
            layer: threading.Lock() for layer in spec.routed_layer_indices
        }
        self._lifecycle = threading.Condition()
        self._closed = False

        allocated = 0
        try:
            for layer in spec.routed_layer_indices:
                for slot_index in range(plan.slots_per_layer):
                    label = f"layer-{layer}-persistent-{slot_index}"
                    buffer = self._allocate_buffer(label)
                    self._persistent[(layer, slot_index)] = _PhysicalSlot(label, buffer)
                    allocated += spec.expert_record_bytes
            transient: list[_PhysicalSlot] = []
            for slot_index in range(plan.transient_slots):
                label = f"global-transient-{slot_index}"
                buffer = self._allocate_buffer(label)
                transient.append(_PhysicalSlot(label, buffer))
                allocated += spec.expert_record_bytes
            self._transient = tuple(transient)
        except Exception:
            self._persistent.clear()
            self._transient = ()
            raise
        expected = plan.persistent_cache_bytes + plan.transient_bytes
        if allocated != expected:
            raise ExpertSlotError(
                f"allocated slot bytes {allocated} do not match memory plan {expected}"
            )
        self.allocated_bytes = allocated
        workers = max(1, max_inflight_io_bytes // spec.expert_record_bytes)
        workers = min(workers, plan.transient_slots)
        self.max_inflight_io_bytes = workers * spec.expert_record_bytes
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mtplx-expert-io",
        )

    def _allocate_buffer(self, label: str) -> Any:
        buffer = self._allocator(self.spec.expert_record_bytes, label)
        try:
            view = memoryview(buffer)
        except TypeError as exc:
            raise ExpertSlotError(
                f"allocator returned a non-buffer for {label}"
            ) from exc
        if (
            view.readonly
            or not view.c_contiguous
            or view.nbytes != self.spec.expert_record_bytes
        ):
            raise ExpertSlotError(
                f"allocator returned invalid buffer for {label}: "
                f"readonly={view.readonly}, contiguous={view.c_contiguous}, bytes={view.nbytes}"
            )
        return buffer

    def _physical(self, layer: int, logical_slot: int) -> _PhysicalSlot:
        if layer not in self.spec.routed_layer_indices:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer")
        if logical_slot < self.plan.slots_per_layer:
            try:
                return self._persistent[(layer, logical_slot)]
            except KeyError as exc:
                raise ExpertSlotError(
                    "persistent slot is outside the memory plan"
                ) from exc
        transient_index = logical_slot - self.plan.slots_per_layer
        if not 0 <= transient_index < len(self._transient):
            raise ExpertSlotError("transient slot is outside the memory plan")
        return self._transient[transient_index]

    @staticmethod
    def _remaining(deadline_ns: int | None) -> float | None:
        if deadline_ns is None:
            return None
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            raise TimeoutError("expert slot deadline exceeded")
        return remaining_ns / 1e9

    def _prepare_load(
        self,
        layer: int,
        load: SlotLoad,
        *,
        deadline_ns: int | None,
    ) -> tuple[_PhysicalSlot, int, bool]:
        slot = self._physical(layer, load.slot)
        with slot.condition:
            while True:
                if self._closed or slot.state is ExpertSlotState.CLOSED:
                    raise ExpertSlotError("expert slot pool is closed")
                if (
                    slot.layer == layer
                    and slot.expert == load.expert
                    and slot.state in {ExpertSlotState.LOADING, ExpertSlotState.READY}
                ):
                    if slot.state is ExpertSlotState.LOADING:
                        self.metrics.update(deduplicated_loads=1)
                    else:
                        self.metrics.update(ready_hits=1)
                    return slot, slot.generation, False
                if slot.state is ExpertSlotState.LOADING:
                    self.metrics.update(load_waits=1)
                    slot.condition.wait(self._remaining(deadline_ns))
                    continue
                if slot.pins:
                    self.metrics.update(pin_waits=1)
                    slot.condition.wait(self._remaining(deadline_ns))
                    continue
                replacing = slot.state is ExpertSlotState.READY
                slot.generation += 1
                slot.layer = layer
                slot.expert = load.expert
                slot.state = ExpertSlotState.LOADING
                slot.digest = None
                slot.error = None
                if replacing:
                    self.metrics.update(generation_replacements=1)
                self.metrics.update(owned_loads=1)
                return slot, slot.generation, True

    def _fill(
        self,
        slot: _PhysicalSlot,
        generation: int,
        record: ExpertRecord,
        *,
        cancel_event: Any,
        deadline_ns: int | None,
    ) -> None:
        try:
            digest = self.reader.read_record_into(
                self.manifest,
                record,
                slot.buffer,
                prefer_sidecar=self.prefer_sidecar,
                verify_hash=self.verify_hashes,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
            )
        except BaseException as exc:
            with slot.condition:
                if (
                    slot.generation == generation
                    and slot.state is ExpertSlotState.LOADING
                ):
                    slot.state = ExpertSlotState.FAILED
                    slot.error = exc
                    slot.condition.notify_all()
            self.metrics.update(load_failures=1)
            raise
        with slot.condition:
            if (
                slot.generation != generation
                or slot.state is not ExpertSlotState.LOADING
            ):
                raise ExpertSlotError("slot generation changed during an expert read")
            slot.digest = digest
            slot.state = ExpertSlotState.READY
            slot.error = None
            slot.condition.notify_all()

    def _wait_ready(
        self,
        slot: _PhysicalSlot,
        *,
        layer: int,
        expert: int,
        generation: int,
        deadline_ns: int | None,
    ) -> None:
        with slot.condition:
            while slot.state is ExpertSlotState.LOADING:
                self.metrics.update(load_waits=1)
                slot.condition.wait(self._remaining(deadline_ns))
            if (
                slot.state is not ExpertSlotState.READY
                or slot.layer != layer
                or slot.expert != expert
                or slot.generation != generation
            ):
                if slot.error is not None:
                    raise ExpertSlotError(
                        f"expert load failed for ({layer}, {expert}): {slot.error}"
                    ) from slot.error
                raise ExpertSlotError(
                    "expert slot did not reach the requested generation"
                )

    def ensure_route(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute:
        """Load all misses, validate mappings, and pin the route's slots."""

        try:
            layer_lock = self._ensure_locks[layer]
        except KeyError as exc:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer") from exc
        with layer_lock:
            return self._ensure_route_locked(
                layer,
                plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
            )

    def _ensure_route_locked(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
    ) -> ReadyRoute:

        if len(plan.experts) != len(plan.slots):
            raise ExpertSlotError("route experts and slots differ in length")
        with self._lifecycle:
            if self._closed:
                raise ExpertSlotError("expert slot pool is closed")
            self.metrics.update(active_routes=1)
        self.metrics.update(ensure_calls=1, load_requests=len(plan.loads))
        internal_cancel = threading.Event()
        combined_cancel = _CombinedCancel(cancel_event, internal_cancel)
        prepared: dict[tuple[int, int], tuple[_PhysicalSlot, int]] = {}
        futures: list[Future[None]] = []
        try:
            for load in plan.loads:
                try:
                    record = self._record_map[(layer, load.expert)]
                except KeyError as exc:
                    raise ExpertSlotError(
                        f"manifest has no expert record ({layer}, {load.expert})"
                    ) from exc
                slot, generation, owner = self._prepare_load(
                    layer,
                    load,
                    deadline_ns=deadline_ns,
                )
                prepared[(load.expert, load.slot)] = (slot, generation)
                if owner:
                    futures.append(
                        self._executor.submit(
                            self._fill,
                            slot,
                            generation,
                            record,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                        )
                    )
            future_error: BaseException | None = None
            for future in futures:
                try:
                    future.result(timeout=self._remaining(deadline_ns))
                except BaseException as exc:
                    internal_cancel.set()
                    if future_error is None:
                        future_error = exc
            if future_error is not None:
                for future in futures:
                    if not future.done():
                        try:
                            future.result()
                        except BaseException:
                            pass
                raise future_error

            bindings: list[ExpertSlotBinding] = []
            unique_pins: dict[int, _PhysicalSlot] = {}
            try:
                for expert, logical_slot in zip(plan.experts, plan.slots, strict=True):
                    slot = self._physical(layer, logical_slot)
                    with slot.condition:
                        current_generation = slot.generation
                    generation = prepared.get(
                        (expert, logical_slot), (slot, current_generation)
                    )[1]
                    self._wait_ready(
                        slot,
                        layer=layer,
                        expert=expert,
                        generation=generation,
                        deadline_ns=deadline_ns,
                    )
                    try:
                        record = self._record_map[(layer, expert)]
                    except KeyError as exc:
                        raise ExpertSlotError(
                            f"manifest has no expert record ({layer}, {expert})"
                        ) from exc
                    with slot.condition:
                        if id(slot) not in unique_pins:
                            slot.pins += 1
                            unique_pins[id(slot)] = slot
                        bindings.append(
                            ExpertSlotBinding(
                                layer=layer,
                                expert=expert,
                                logical_slot=logical_slot,
                                generation=generation,
                                record=record,
                                buffer=slot.buffer,
                            )
                        )
            except BaseException:
                for slot in unique_pins.values():
                    with slot.condition:
                        slot.pins -= 1
                        slot.condition.notify_all()
                raise
        except (ExpertManifestError, ExpertIOError) as exc:
            internal_cancel.set()
            self._route_released()
            raise ExpertSlotError(str(exc)) from exc
        except BaseException:
            internal_cancel.set()
            self._route_released()
            raise

        return ReadyRoute(
            self,
            plan,
            tuple(bindings),
            tuple(unique_pins.values()),
        )

    def _route_released(self) -> None:
        with self._lifecycle:
            self.metrics.update(active_routes=-1)
            self._lifecycle.notify_all()

    def invalidate(self, layer: int, logical_slot: int) -> None:
        slot = self._physical(layer, logical_slot)
        with slot.condition:
            if slot.state is ExpertSlotState.LOADING or slot.pins:
                raise ExpertSlotError("cannot invalidate an active expert slot")
            slot.state = ExpertSlotState.EMPTY
            slot.layer = None
            slot.expert = None
            slot.digest = None
            slot.error = None
            slot.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        states: dict[str, int] = {state.value: 0 for state in ExpertSlotState}
        pins = 0
        for slot in (*self._persistent.values(), *self._transient):
            with slot.condition:
                states[slot.state.value] += 1
                pins += slot.pins
        return {
            "buffer_backend": self.buffer_backend,
            "allocated_bytes": self.allocated_bytes,
            "max_inflight_io_bytes": self.max_inflight_io_bytes,
            "persistent_slot_count": len(self._persistent),
            "transient_slot_count": len(self._transient),
            "states": states,
            "pins": pins,
            "metrics": self.metrics.as_dict(),
            "io": self.reader.metrics.as_dict(),
        }

    def reset(self) -> None:
        with self._lifecycle:
            if self.metrics.as_dict()["active_routes"]:
                raise ExpertSlotError("cannot reset while expert routes are active")
        for slot in (*self._persistent.values(), *self._transient):
            with slot.condition:
                if slot.state is ExpertSlotState.LOADING or slot.pins:
                    raise ExpertSlotError("cannot reset an active expert slot")
                slot.state = ExpertSlotState.EMPTY
                slot.layer = None
                slot.expert = None
                slot.digest = None
                slot.error = None
                slot.condition.notify_all()

    def close(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lifecycle:
            self._closed = True
            while self.metrics.as_dict()["active_routes"]:
                if deadline is None:
                    self._lifecycle.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "active expert routes did not release before close"
                        )
                    self._lifecycle.wait(remaining)
        self._executor.shutdown(wait=True, cancel_futures=True)
        for slot in (*self._persistent.values(), *self._transient):
            with slot.condition:
                slot.state = ExpertSlotState.CLOSED
                slot.condition.notify_all()
        self.reader.close()

    def __enter__(self) -> ExpertSlotPool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def release_routes(routes: Iterable[ReadyRoute]) -> None:
    for route in routes:
        route.release()
