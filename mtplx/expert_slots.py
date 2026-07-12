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


class ExpertCompletionFenceError(ExpertSlotError):
    """Sticky completion failure annotated with policy rollback safety."""

    def __init__(self, message: str, *, policy_rollback_safe: bool) -> None:
        super().__init__(message)
        self.policy_rollback_safe = bool(policy_rollback_safe)


@dataclass
class RouteIOAdmission:
    """Per-route record of executor work accepted past the rollback boundary."""

    accepted_submissions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _parent: RouteIOAdmission | None = field(default=None, repr=False)

    @property
    def any_accepted(self) -> bool:
        with self._lock:
            return self.accepted_submissions > 0

    def mark_accepted(self) -> None:
        with self._lock:
            self.accepted_submissions += 1
        if self._parent is not None:
            self._parent.mark_accepted()

    def child(self) -> RouteIOAdmission:
        """Create a part-local admission that also updates this route."""

        return RouteIOAdmission(_parent=self)


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
    completion_fences: int = 0
    completion_fence_slots: int = 0
    completion_fence_fallbacks: int = 0
    completion_fence_failures: int = 0
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
                    "completion_fences",
                    "completion_fence_slots",
                    "completion_fence_fallbacks",
                    "completion_fence_failures",
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
class _PreparedSlotState:
    slot: _PhysicalSlot
    owned_generation: int
    state: ExpertSlotState
    layer: int | None
    expert: int | None
    generation: int
    digest: str | None
    error: BaseException | None


@dataclass(frozen=True)
class ExpertSlotBinding:
    layer: int
    expert: int
    logical_slot: int
    generation: int
    record: ExpertRecord
    buffer: Any

    def component_view(self, component: str) -> memoryview:
        direct = getattr(self.buffer, "component_view", None)
        if callable(direct):
            return direct(component)
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
        self._route_finished = False
        self._release_lock = threading.Lock()
        self._release_condition = threading.Condition(self._release_lock)
        self._pending_slots = {id(slot): slot for slot in pinned}
        self._scheduled_slots: set[int] = set()
        self._completion_futures: list[Future[None]] = []
        self._registrations_in_progress = 0

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

    def defer_bindings_until(
        self,
        bindings: Iterable[ExpertSlotBinding],
        completion_waiter: Callable[[], None],
    ) -> bool:
        """Hold selected slot generations until their Metal work completes.

        The waiter runs on the pool's bounded completion lane. This lets the
        generation thread proceed to miss I/O without making a slot reusable
        before the asynchronous kernels that consume it have completed.
        """

        if not callable(completion_waiter):
            raise TypeError("completion_waiter must be callable")
        selected: dict[int, _PhysicalSlot] = {}
        for binding in bindings:
            slot = self.pool._physical(binding.layer, binding.logical_slot)
            selected[id(slot)] = slot
        if not selected:
            return False
        with self._release_condition:
            if self._released:
                raise ExpertSlotError("cannot fence a released ready route")
            unknown = set(selected) - set(self._pending_slots)
            if unknown:
                raise ExpertSlotError("completion fence references an unpinned slot")
            duplicate = set(selected) & self._scheduled_slots
            if duplicate:
                raise ExpertSlotError("slot already has a completion fence")
            self._scheduled_slots.update(selected)
            self._registrations_in_progress += 1
        try:
            future = self.pool._submit_completion_fence(
                completion_waiter,
                lambda: self._finish_slots(tuple(selected.values())),
                slot_count=len(selected),
            )
        except BaseException:
            with self._release_condition:
                self._scheduled_slots.difference_update(selected)
                self._registrations_in_progress -= 1
                self._release_condition.notify_all()
            raise
        with self._release_condition:
            if future is not None:
                self._completion_futures.append(future)
            self._registrations_in_progress -= 1
            self._release_condition.notify_all()
        return True

    def _binding_slots(self) -> tuple[_PhysicalSlot, ...]:
        return tuple(
            self.pool._physical(binding.layer, binding.logical_slot)
            for binding in self.bindings
        )

    def _finish_slots(self, slots: tuple[_PhysicalSlot, ...]) -> None:
        for slot in slots:
            with slot.condition:
                if slot.pins <= 0:
                    raise ExpertSlotError("slot pin accounting underflow")
                slot.pins -= 1
                slot.condition.notify_all()
        finish_route = False
        with self._release_lock:
            for slot in slots:
                slot_id = id(slot)
                self._pending_slots.pop(slot_id, None)
                self._scheduled_slots.discard(slot_id)
            if self._released and not self._pending_slots and not self._route_finished:
                self._route_finished = True
                finish_route = True
        if finish_route:
            self.pool._route_released()

    def release(self, *, synchronize: bool = True) -> None:
        with self._release_condition:
            first_release = not self._released
            self._released = True
            while self._registrations_in_progress:
                self._release_condition.wait()
            immediate = (
                tuple(
                    slot
                    for slot_id, slot in self._pending_slots.items()
                    if slot_id not in self._scheduled_slots
                )
                if first_release
                else ()
            )
            futures = tuple(self._completion_futures)
            finish_empty = (
                first_release and not self._pending_slots and not self._route_finished
            )
            if finish_empty:
                self._route_finished = True
        synchronize_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        future_error: BaseException | None = None
        if first_release and synchronize and self.pool.device_synchronize is not None:
            try:
                self.pool.device_synchronize()
            except BaseException as exc:
                self.pool.metrics.update(completion_fence_failures=1)
                self.pool._record_completion_error(exc)
                synchronize_error = exc
        try:
            if immediate:
                self._finish_slots(immediate)
            elif finish_empty:
                self.pool._route_released()
        except BaseException as exc:
            cleanup_error = exc
        if synchronize:
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    if future_error is None:
                        future_error = exc
        if synchronize_error is not None:
            raise synchronize_error
        if cleanup_error is not None:
            raise cleanup_error
        if future_error is not None:
            raise ExpertSlotError("expert completion fence failed") from future_error
        if synchronize:
            self.pool._raise_completion_error()

    def __enter__(self) -> ReadyRoute:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _RouteLifecycle:
    """One extra active-route hold spanning a multi-part transaction."""

    def __init__(self, pool: ExpertSlotPool) -> None:
        self.pool = pool
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self.pool._route_released()


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
        cache_scope: str = "layer",
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
        if cache_scope not in {"layer", "global"}:
            raise ValueError("cache_scope must be 'layer' or 'global'")
        self.cache_scope = cache_scope
        self.global_persistent_slots = (
            plan.persistent_slots if cache_scope == "global" else 0
        )
        self._persistent_route_capacity = (
            self.global_persistent_slots
            if cache_scope == "global"
            else plan.slots_per_layer
        )
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
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False

        allocated = 0
        try:
            persistent_layout = (
                (
                    (None, slot_index)
                    for slot_index in range(self.global_persistent_slots)
                )
                if self.cache_scope == "global"
                else (
                    (layer, slot_index)
                    for layer in spec.routed_layer_indices
                    for slot_index in range(plan.slots_per_layer)
                )
            )
            for layer, slot_index in persistent_layout:
                label = (
                    f"global-persistent-{slot_index}"
                    if layer is None
                    else f"layer-{layer}-persistent-{slot_index}"
                )
                buffer = self._allocate_buffer(label)
                key_layer = -1 if layer is None else layer
                self._persistent[(key_layer, slot_index)] = _PhysicalSlot(label, buffer)
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
        self._completion_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mtplx-slot-fence",
        )
        self._completion_error_lock = threading.Lock()
        self._completion_error: BaseException | None = None

    def _submit_completion_fence(
        self,
        completion_waiter: Callable[[], None],
        on_complete: Callable[[], None],
        *,
        slot_count: int,
    ) -> Future[None] | None:
        self.metrics.update(
            completion_fences=1,
            completion_fence_slots=slot_count,
        )

        def wait_and_release() -> None:
            try:
                completion_waiter()
            except BaseException as exc:
                self.metrics.update(completion_fence_failures=1)
                self._record_completion_error(exc)
                raise
            finally:
                on_complete()

        try:
            return self._completion_executor.submit(wait_and_release)
        except RuntimeError:
            # Shutdown races or stripped-down runtimes fail closed: wait on the
            # caller before allowing any generation to be overwritten.
            self.metrics.update(completion_fence_fallbacks=1)
            wait_and_release()
            return None

    def _record_completion_error(self, error: BaseException) -> None:
        with self._completion_error_lock:
            if self._completion_error is None:
                self._completion_error = error

    def _raise_completion_error_locked(
        self,
        *,
        policy_rollback_safe: bool,
    ) -> None:
        error = self._completion_error
        if error is not None:
            raise ExpertCompletionFenceError(
                "an asynchronous expert completion fence failed",
                policy_rollback_safe=policy_rollback_safe,
            ) from error

    def _raise_completion_error(
        self,
        *,
        policy_rollback_safe: bool = True,
    ) -> None:
        with self._completion_error_lock:
            self._raise_completion_error_locked(
                policy_rollback_safe=policy_rollback_safe
            )

    def raise_if_unhealthy(self) -> None:
        """Reject new policy or slot work after a terminal fence failure."""

        self._raise_completion_error()

    def _drain_completion_fences(self) -> None:
        """Wait for completion tasks queued before this diagnostic snapshot."""

        try:
            barrier = self._completion_executor.submit(lambda: None)
        except RuntimeError:
            return
        barrier.result()

    def _allocate_buffer(self, label: str) -> Any:
        buffer = self._allocator(self.spec.expert_record_bytes, label)
        direct_nbytes = getattr(buffer, "nbytes", None)
        record_views = getattr(buffer, "record_views", None)
        if callable(record_views):
            if int(direct_nbytes or -1) != self.spec.expert_record_bytes:
                raise ExpertSlotError(
                    f"allocator returned invalid composite buffer for {label}: "
                    f"bytes={direct_nbytes}"
                )
            return buffer
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
        if logical_slot < self._persistent_route_capacity:
            try:
                key_layer = -1 if self.cache_scope == "global" else layer
                return self._persistent[(key_layer, logical_slot)]
            except KeyError as exc:
                raise ExpertSlotError(
                    "persistent slot is outside the memory plan"
                ) from exc
        transient_index = logical_slot - self._persistent_route_capacity
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
    ) -> tuple[_PhysicalSlot, int, bool, _PreparedSlotState | None]:
        slot = self._physical(layer, load.slot)
        with slot.condition:
            while True:
                if self._closed or slot.state is ExpertSlotState.CLOSED:
                    raise ExpertSlotError("expert slot pool is closed")
                if self._closing:
                    raise ExpertSlotError("expert slot pool is closing")
                self._raise_completion_error()
                if (
                    slot.layer == layer
                    and slot.expert == load.expert
                    and slot.state in {ExpertSlotState.LOADING, ExpertSlotState.READY}
                    and (load.generation is None or slot.generation == load.generation)
                ):
                    if slot.state is ExpertSlotState.LOADING:
                        self.metrics.update(deduplicated_loads=1)
                    else:
                        self.metrics.update(ready_hits=1)
                    return slot, slot.generation, False, None
                if slot.state is ExpertSlotState.LOADING:
                    self.metrics.update(load_waits=1)
                    slot.condition.wait(self._remaining(deadline_ns))
                    continue
                if slot.pins:
                    self.metrics.update(pin_waits=1)
                    slot.condition.wait(self._remaining(deadline_ns))
                    self._raise_completion_error()
                    continue
                self._raise_completion_error()
                previous_state = slot.state
                previous_layer = slot.layer
                previous_expert = slot.expert
                previous_generation = slot.generation
                previous_digest = slot.digest
                previous_error = slot.error
                replacing = slot.state is ExpertSlotState.READY
                if load.generation is None:
                    slot.generation += 1
                else:
                    if load.generation <= slot.generation:
                        raise ExpertSlotError(
                            "stale global cache generation requested for slot"
                        )
                    slot.generation = load.generation
                previous = _PreparedSlotState(
                    slot=slot,
                    owned_generation=slot.generation,
                    state=previous_state,
                    layer=previous_layer,
                    expert=previous_expert,
                    generation=previous_generation,
                    digest=previous_digest,
                    error=previous_error,
                )
                slot.layer = layer
                slot.expert = load.expert
                slot.state = ExpertSlotState.LOADING
                slot.digest = None
                slot.error = None
                if replacing:
                    self.metrics.update(generation_replacements=1)
                self.metrics.update(owned_loads=1)
                return slot, slot.generation, True, previous

    @staticmethod
    def _restore_prepared_slots(prepared: Iterable[_PreparedSlotState]) -> None:
        for previous in reversed(tuple(prepared)):
            slot = previous.slot
            with slot.condition:
                if (
                    slot.state is not ExpertSlotState.LOADING
                    or slot.generation != previous.owned_generation
                ):
                    continue
                slot.state = previous.state
                slot.layer = previous.layer
                slot.expert = previous.expert
                slot.generation = previous.generation
                slot.digest = previous.digest
                slot.error = previous.error
                slot.condition.notify_all()

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

    def _fill_batch(
        self,
        owned: tuple[tuple[_PhysicalSlot, int, ExpertRecord], ...],
        *,
        cancel_event: Any,
        deadline_ns: int | None,
    ) -> None:
        try:
            digests = self.reader.read_component_records_into(
                self.manifest,
                tuple((record, slot.buffer) for slot, _generation, record in owned),
                verify_hash=self.verify_hashes,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
            )
        except BaseException as exc:
            for slot, generation, _record in owned:
                with slot.condition:
                    if (
                        slot.generation == generation
                        and slot.state is ExpertSlotState.LOADING
                    ):
                        slot.state = ExpertSlotState.FAILED
                        slot.error = exc
                        slot.condition.notify_all()
            self.metrics.update(load_failures=len(owned))
            raise
        for (slot, generation, _record), digest in zip(owned, digests, strict=True):
            with slot.condition:
                if (
                    slot.generation != generation
                    or slot.state is not ExpertSlotState.LOADING
                ):
                    raise ExpertSlotError(
                        "slot generation changed during a batched expert read"
                    )
                slot.digest = digest
                slot.state = ExpertSlotState.READY
                slot.error = None
                slot.condition.notify_all()

    def _can_batch_component_sidecar(
        self,
        plan: RoutePlan,
        owned: list[tuple[_PhysicalSlot, int, ExpertRecord]],
    ) -> bool:
        if (
            plan.phase.value != "prefill"
            or not self.prefer_sidecar
            or self.manifest.sidecar is None
            or len(owned) < 2
            or not all(
                callable(getattr(slot.buffer, "record_views", None))
                for slot, _generation, _record in owned
            )
        ):
            return False
        ordered = sorted(
            (record for _slot, _generation, record in owned),
            key=lambda record: int(record.sidecar_offset or 0),
        )
        return all(
            int(left.sidecar_offset or 0) + int(left.sidecar_length or 0)
            == int(right.sidecar_offset or 0)
            for left, right in zip(ordered, ordered[1:], strict=False)
        )

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
        io_admission: RouteIOAdmission | None = None,
    ) -> ReadyRoute:
        """Load all misses, validate mappings, and pin the route's slots."""

        self._raise_completion_error()
        try:
            layer_lock = self._ensure_locks[layer]
        except KeyError as exc:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer") from exc
        with layer_lock:
            self._raise_completion_error()
            if io_admission is None:
                return self._ensure_route_locked(
                    layer,
                    plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
            return self._ensure_route_locked(
                layer,
                plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                io_admission=io_admission,
            )

    def retain_split_lifecycle(self) -> _RouteLifecycle:
        """Keep close from finalizing slots during multi-part commit/rollback."""

        self._raise_completion_error()
        with self._lifecycle:
            if self._closed:
                raise ExpertSlotError("expert slot pool is closed")
            if self._closing:
                raise ExpertSlotError("expert slot pool is closing")
            self.metrics.update(active_routes=1)
        return _RouteLifecycle(self)

    def ensure_route_part(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
        io_admission: RouteIOAdmission | None = None,
    ) -> ReadyRoute:
        """Load one disjoint part of a runtime-locked route transaction.

        ``ExpertStreamingRuntime`` holds its layer transaction lock while all
        parts are active. Each part owns distinct logical slots, so individual
        slot conditions provide the remaining synchronization without forcing
        completed records to wait behind the slowest read in the layer.
        """

        self._raise_completion_error()
        if layer not in self._ensure_locks:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer")
        return self._ensure_route_locked(
            layer,
            plan,
            cancel_event=cancel_event,
            deadline_ns=deadline_ns,
            io_admission=io_admission,
        )

    def _ensure_route_locked(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
        io_admission: RouteIOAdmission | None = None,
    ) -> ReadyRoute:

        self._raise_completion_error()
        if len(plan.experts) != len(plan.slots):
            raise ExpertSlotError("route experts and slots differ in length")
        with self._lifecycle:
            if self._closed:
                raise ExpertSlotError("expert slot pool is closed")
            if self._closing:
                raise ExpertSlotError("expert slot pool is closing")
            self.metrics.update(active_routes=1)
        self.metrics.update(ensure_calls=1, load_requests=len(plan.loads))
        internal_cancel = threading.Event()
        combined_cancel = _CombinedCancel(cancel_event, internal_cancel)
        prepared: dict[tuple[int, int], tuple[_PhysicalSlot, int]] = {}
        prepared_states: list[_PreparedSlotState] = []
        futures: list[Future[None]] = []
        owned_loads: list[tuple[_PhysicalSlot, int, ExpertRecord]] = []
        submitted_loads: set[tuple[int, int]] = set()
        admission = io_admission if io_admission is not None else RouteIOAdmission()
        try:
            try:
                for load in plan.loads:
                    try:
                        record = self._record_map[(layer, load.expert)]
                    except KeyError as exc:
                        raise ExpertSlotError(
                            f"manifest has no expert record ({layer}, {load.expert})"
                        ) from exc
                    slot, generation, owner, previous = self._prepare_load(
                        layer,
                        load,
                        deadline_ns=deadline_ns,
                    )
                    prepared[(load.expert, load.slot)] = (slot, generation)
                    if owner:
                        owned_loads.append((slot, generation, record))
                        assert previous is not None
                        prepared_states.append(previous)
                use_batch = self._can_batch_component_sidecar(plan, owned_loads)
                # Completion failure recording uses this same lock.  Keeping
                # it through every submission makes the rollback boundary
                # exact: either no read was accepted, or policy restoration is
                # conservatively unsafe because at least one read may replace
                # a victim generation.
                with self._completion_error_lock:
                    self._raise_completion_error_locked(policy_rollback_safe=True)
                    if use_batch:
                        future = self._executor.submit(
                            self._fill_batch,
                            tuple(owned_loads),
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                        )
                        admission.mark_accepted()
                        futures.append(future)
                        submitted_loads.update(
                            (id(slot), generation)
                            for slot, generation, _record in owned_loads
                        )
                    else:
                        for slot, generation, record in owned_loads:
                            future = self._executor.submit(
                                self._fill,
                                slot,
                                generation,
                                record,
                                cancel_event=combined_cancel,
                                deadline_ns=deadline_ns,
                            )
                            admission.mark_accepted()
                            futures.append(future)
                            submitted_loads.add((id(slot), generation))
            except BaseException:
                self._restore_prepared_slots(
                    previous
                    for previous in prepared_states
                    if (id(previous.slot), previous.owned_generation)
                    not in submitted_loads
                )
                if futures:
                    internal_cancel.set()
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
                raise
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
                plan_generations = (
                    plan.generations
                    if plan.generations
                    else (None,) * len(plan.experts)
                )
                if len(plan_generations) != len(plan.experts):
                    raise ExpertSlotError(
                        "route experts and generations differ in length"
                    )
                for expert, logical_slot, expected_generation in zip(
                    plan.experts,
                    plan.slots,
                    plan_generations,
                    strict=True,
                ):
                    slot = self._physical(layer, logical_slot)
                    with slot.condition:
                        current_generation = slot.generation
                    generation = prepared.get(
                        (expert, logical_slot), (slot, current_generation)
                    )[1]
                    if expected_generation is not None:
                        generation = expected_generation
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
                        if admission.any_accepted:
                            self._raise_completion_error(policy_rollback_safe=False)
                        else:
                            self._raise_completion_error()
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
                if admission.any_accepted:
                    self._raise_completion_error(policy_rollback_safe=False)
                else:
                    self._raise_completion_error()
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

    def invalidate(
        self,
        layer: int,
        logical_slot: int,
        *,
        expert: int | None = None,
        generation: int | None = None,
    ) -> bool:
        slot = self._physical(layer, logical_slot)
        with slot.condition:
            if slot.layer != layer:
                return False
            if expert is not None and slot.expert != expert:
                return False
            if generation is not None and slot.generation != generation:
                return False
            if slot.state is ExpertSlotState.LOADING or slot.pins:
                raise ExpertSlotError("cannot invalidate an active expert slot")
            slot.state = ExpertSlotState.EMPTY
            slot.layer = None
            slot.expert = None
            slot.digest = None
            slot.error = None
            slot.condition.notify_all()
            return True

    def snapshot(self) -> dict[str, Any]:
        self._drain_completion_fences()
        self._raise_completion_error()
        states: dict[str, int] = {state.value: 0 for state in ExpertSlotState}
        pins = 0
        for slot in (*self._persistent.values(), *self._transient):
            with slot.condition:
                states[slot.state.value] += 1
                pins += slot.pins
        return {
            "buffer_backend": self.buffer_backend,
            "io_cache_mode": self.reader.cache_mode,
            "allocated_bytes": self.allocated_bytes,
            "max_inflight_io_bytes": self.max_inflight_io_bytes,
            "persistent_slot_count": len(self._persistent),
            "cache_scope": self.cache_scope,
            "persistent_route_capacity": self._persistent_route_capacity,
            "transient_slot_count": len(self._transient),
            "states": states,
            "pins": pins,
            "metrics": self.metrics.as_dict(),
            "io": self.reader.metrics.as_dict(),
        }

    def reset(self) -> None:
        self._raise_completion_error()
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
        if deadline is None:
            self._close_lock.acquire()
        else:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._close_lock.acquire(timeout=remaining):
                raise TimeoutError("expert slot close already in progress at deadline")
        try:
            with self._lifecycle:
                if not self._closed:
                    self._closing = True
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
                finalized = self._closed
            if not finalized:
                self._completion_executor.shutdown(wait=True, cancel_futures=False)
                self._executor.shutdown(wait=True, cancel_futures=True)
                for slot in (*self._persistent.values(), *self._transient):
                    with slot.condition:
                        slot.state = ExpertSlotState.CLOSED
                        slot.condition.notify_all()
                self.reader.close()
                allocator_close = getattr(self._allocator, "close", None)
                if callable(allocator_close):
                    allocator_close()
                with self._lifecycle:
                    self._closed = True
                    self._closing = False
                    self._lifecycle.notify_all()
            self._raise_completion_error()
        finally:
            self._close_lock.release()

    def __enter__(self) -> ExpertSlotPool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def release_routes(routes: Iterable[ReadyRoute]) -> None:
    for route in routes:
        route.release()
