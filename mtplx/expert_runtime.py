"""Model-independent orchestration for bounded SSD expert streaming."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .expert_io import PositionalExpertReader
from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    load_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from .expert_slots import (
    ExpertCompletionFenceError,
    ExpertSlotError,
    ExpertSlotPool,
    ReadyRoute,
    RouteIOAdmission,
)
from .expert_streaming import (
    CacheCounters,
    GlobalExpertSlotBank,
    LayerExpertSlotBank,
    RoutePlan,
    RoutePolicyTxn,
    RoutingPhase,
)
from .expert_streaming_models import (
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
    get_model_spec,
    plan_expert_memory,
    resident_quant_covers,
    resident_quant_kept_bytes,
)
from .resource_metrics import ExpertPipelineLedger, ExpertPipelineRoute


_MEMORY_RE = re.compile(r"^([0-9]+)([kmgt]i?b?|b)?$", re.IGNORECASE)


class ExpertStreamingConfigurationError(ValueError):
    pass


def _pipeline_call(
    ledger: ExpertPipelineLedger | None,
    route: ExpertPipelineRoute,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Publish optional diagnostics without changing runtime outcomes."""

    try:
        getattr(route, method)(*args, **kwargs)
    except Exception:
        if ledger is not None:
            try:
                ledger.mark_incomplete(phase=route.phase)
            except Exception:
                pass


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def parse_memory_bytes(value: str | int) -> int:
    if isinstance(value, bool):
        raise ExpertStreamingConfigurationError("memory size must not be bool")
    if isinstance(value, int):
        if value <= 0:
            raise ExpertStreamingConfigurationError("memory size must be positive")
        return value
    if not isinstance(value, str):
        raise ExpertStreamingConfigurationError(
            "memory size must be bytes or a suffixed string"
        )
    normalized = value.strip().lower()
    match = _MEMORY_RE.fullmatch(normalized)
    if match is None:
        raise ExpertStreamingConfigurationError(f"invalid memory size {value!r}")
    number = int(match.group(1))
    suffix = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    result = number * multipliers[suffix]
    if result <= 0:
        raise ExpertStreamingConfigurationError("memory size must be positive")
    return result


@dataclass(frozen=True)
class ExpertStreamingConfig:
    model_key: str
    memory_limit_bytes: int
    max_live_kv_tokens: int
    runtime_reserve_bytes: int = 16 * 1024**3
    expert_cache_limit_bytes: int | None = None
    transient_slots: int | None = None
    io_staging_bytes: int = 0
    execution_workspace_bytes: int = 0
    max_inflight_io_bytes: int | None = None
    max_open_files: int = 16
    max_read_chunk_bytes: int = 8 * 1024 * 1024
    frequency_decay: float = 0.995
    prefer_sidecar: bool = True
    verify_record_hashes: bool = True
    verify_artifact_headers: bool = True
    verify_sidecar_hash_at_open: bool = False
    prefill_admission: bool = False
    slot_layout: str = "direct-slots"
    trace_routes: bool = False
    cache_policy: str = "frequency"
    cache_scope: str = "layer"
    bypass_page_cache: bool = False
    resource_telemetry: bool = False
    q2_expert_kernel: str = "stock"
    hy3_router_kernel: str = "mpp-r1-fused-r2"
    hy3_router_sigmoid: str = "precise"
    hy3_mtp_shared_kernel: str = "stock"
    hy3_mtp_shared_kernel_depth: int = 3
    deferred_pin_release: bool = False
    island_layers: tuple[int, ...] = ()
    island_layer_count: int | None = None
    mmap_island_layers: tuple[int, ...] = ()
    banked_manifest: str | None = None
    banked_codec: str = "none"
    mmap_island_wired: bool = True
    resident_quant: str | None = None
    kv_quant: str | None = None
    split_route_release: str = "fenced"
    prefetch_slots: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise TypeError("model_key must be a non-empty string")
        for name, minimum in (
            ("memory_limit_bytes", 1),
            ("max_live_kv_tokens", 0),
            ("runtime_reserve_bytes", 0),
            ("io_staging_bytes", 0),
            ("execution_workspace_bytes", 0),
            ("max_open_files", 1),
            ("max_read_chunk_bytes", 1),
        ):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=minimum)
            )
        for name in (
            "expert_cache_limit_bytes",
            "transient_slots",
            "max_inflight_io_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(name, value, minimum=0))
        if self.max_inflight_io_bytes == 0:
            raise ValueError("max_inflight_io_bytes must be positive when supplied")
        if isinstance(self.frequency_decay, bool):
            raise TypeError("frequency_decay must be numeric")
        decay = float(self.frequency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")
        object.__setattr__(self, "frequency_decay", decay)
        if self.cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        if self.cache_scope not in {"layer", "global"}:
            raise ValueError("cache_scope must be 'layer' or 'global'")
        if self.q2_expert_kernel not in {
            "stock",
            "nax",
            "fused",
            "fused-nax",
        }:
            raise ValueError(
                "q2_expert_kernel must be 'stock', 'nax', 'fused', or 'fused-nax'"
            )
        if self.hy3_router_kernel not in {
            "stock",
            "steel-r1-fused-r2",
            "mpp-r1-fused-r2",
            "mpp-fp32-splitk-r1-fused-r2",
            "mpp-r1-last-arrival-fused-r2",
            "mpp-row-owned-fused",
        }:
            raise ValueError(
                "hy3_router_kernel must be 'stock', 'steel-r1-fused-r2', "
                "'mpp-r1-fused-r2', 'mpp-fp32-splitk-r1-fused-r2', "
                "'mpp-r1-last-arrival-fused-r2', or 'mpp-row-owned-fused'"
            )
        if not isinstance(self.deferred_pin_release, bool):
            raise ValueError("deferred_pin_release must be bool")
        if self.resident_quant not in {None, "q8", "q4"}:
            raise ValueError("resident_quant must be None, 'q8', or 'q4'")
        if self.kv_quant not in {None, "q8", "q4"}:
            raise ValueError("kv_quant must be None, 'q8', or 'q4'")
        if self.split_route_release not in {"fenced", "deferred"}:
            raise ValueError(
                "split_route_release must be 'fenced' or 'deferred'"
            )
        object.__setattr__(
            self,
            "prefetch_slots",
            _integer("prefetch_slots", self.prefetch_slots, minimum=0),
        )
        if self.prefetch_slots and self.cache_scope != "layer":
            raise ValueError("prefetch_slots require cache_scope 'layer'")
        if self.island_layer_count is not None:
            if self.island_layers:
                raise ValueError(
                    "island_layers and island_layer_count are mutually "
                    "exclusive; give the count OR the explicit list"
                )
            count = _integer(
                "island_layer_count", self.island_layer_count, minimum=1
            )
            spec = get_model_spec(self.model_key)
            if not spec.island_pin_order:
                raise ValueError(
                    f"{self.model_key} has no measured island pin order; "
                    "count-based selection requires explicit island_layers"
                )
            if count > len(spec.island_pin_order):
                raise ValueError(
                    f"island_layer_count {count} exceeds the {self.model_key} "
                    f"pin order ({len(spec.island_pin_order)} layers)"
                )
            object.__setattr__(
                self,
                "island_layers",
                tuple(sorted(spec.island_pin_order[:count])),
            )
            object.__setattr__(self, "island_layer_count", count)
        island_layers = self.island_layers
        if isinstance(island_layers, (str, bytes)) or not isinstance(
            island_layers, (tuple, list)
        ):
            raise TypeError("island_layers must be a tuple of layer indices")
        normalized_islands = tuple(
            sorted(
                {
                    _integer("island_layers entry", layer, minimum=0)
                    for layer in island_layers
                }
            )
        )
        object.__setattr__(self, "island_layers", normalized_islands)
        if normalized_islands:
            if self.cache_scope != "layer":
                raise ValueError("island_layers require cache_scope 'layer'")
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "island_layers require the component-banks slot layout"
                )
            if self.trace_routes:
                raise ValueError(
                    "island_layers execute without host route observation; "
                    "trace_routes must be disabled"
                )
        if not isinstance(self.mmap_island_wired, bool):
            raise TypeError("mmap_island_wired must be bool")
        mmap_islands = self.mmap_island_layers
        if isinstance(mmap_islands, (str, bytes)) or not isinstance(
            mmap_islands, (tuple, list)
        ):
            raise TypeError("mmap_island_layers must be a tuple of layer indices")
        normalized_mmap = tuple(
            sorted(
                {
                    _integer("mmap_island_layers entry", layer, minimum=0)
                    for layer in mmap_islands
                }
            )
        )
        object.__setattr__(self, "mmap_island_layers", normalized_mmap)
        if self.banked_codec not in {"none", "rans32x-v1"}:
            raise ValueError(
                "banked_codec must be 'none' or 'rans32x-v1'"
            )
        if self.banked_codec != "none":
            raise ValueError(
                "banked_codec 'rans32x-v1' requires the in-kernel decoder "
                "(issue #51, C7); use 'none' until it ships"
            )
        if normalized_mmap:
            if self.banked_manifest is None:
                raise ValueError(
                    "mmap_island_layers require a banked_manifest path"
                )
            if self.cache_scope != "layer":
                raise ValueError(
                    "mmap_island_layers require cache_scope 'layer'"
                )
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "mmap_island_layers require the component-banks slot layout"
                )
            if self.trace_routes:
                raise ValueError(
                    "mmap_island_layers execute without host route observation; "
                    "trace_routes must be disabled"
                )
            overlap = set(normalized_mmap) & set(normalized_islands)
            if overlap:
                raise ValueError(
                    "island_layers and mmap_island_layers must be disjoint; "
                    f"both claim {sorted(overlap)}"
                )
        if self.hy3_router_sigmoid not in {"precise", "fast"}:
            raise ValueError("hy3_router_sigmoid must be 'precise' or 'fast'")
        if (
            self.hy3_router_sigmoid == "fast"
            and self.hy3_router_kernel != "mpp-row-owned-fused"
        ):
            raise ValueError(
                "fast router sigmoid is selectable only with mpp-row-owned-fused"
            )
        if self.hy3_mtp_shared_kernel not in {"stock", "metal-exact"}:
            raise ValueError(
                "hy3_mtp_shared_kernel must be 'stock' or 'metal-exact'"
            )
        object.__setattr__(
            self,
            "hy3_mtp_shared_kernel_depth",
            _integer(
                "hy3_mtp_shared_kernel_depth",
                self.hy3_mtp_shared_kernel_depth,
                minimum=1,
            ),
        )
        if self.slot_layout not in {
            "direct-slots",
            "component-banks",
            "metal-mmap",
        }:
            raise ValueError(
                "slot_layout must be 'direct-slots', 'component-banks', or 'metal-mmap'"
            )
        for name in (
            "prefer_sidecar",
            "verify_record_hashes",
            "verify_artifact_headers",
            "verify_sidecar_hash_at_open",
            "prefill_admission",
            "trace_routes",
            "bypass_page_cache",
            "resource_telemetry",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.verify_sidecar_hash_at_open and not self.prefer_sidecar:
            raise ValueError(
                "verify_sidecar_hash_at_open requires prefer_sidecar: source-shard "
                "reads are not covered by the sidecar digest"
            )
        if self.slot_layout == "metal-mmap" and not self.verify_sidecar_hash_at_open:
            raise ValueError(
                "metal-mmap executes mapped weights without per-record hashing; "
                "it requires verify_sidecar_hash_at_open"
            )
        if self.cache_scope == "global" and self.slot_layout == "metal-mmap":
            raise ValueError(
                "global expert caching requires direct-slots or component-banks"
            )
        if self.prefill_admission:
            raise ValueError(
                "prefill admission is not implemented; prefill must use transient slots"
            )

    def memory_plan(
        self,
        spec: ExpertStreamingModelSpec,
        *,
        additional_resident_bytes: int = 0,
        resident_discount_bytes: int = 0,
    ) -> ExpertMemoryPlan:
        # File-backed Metal records use the OS page cache as their physical
        # tier and never consume fixed MLX expert slots. Retain only a tiny
        # unreachable transient pool so the generic runtime invariants and
        # diagnostics remain valid while the mapped switch owns execution.
        expert_cache_limit_bytes = self.expert_cache_limit_bytes
        transient_slots = self.transient_slots
        if self.slot_layout == "metal-mmap":
            expert_cache_limit_bytes = 0
            transient_slots = spec.top_k
        return plan_expert_memory(
            spec,
            total_limit_bytes=self.memory_limit_bytes,
            context_tokens=self.max_live_kv_tokens,
            runtime_reserve_bytes=self.runtime_reserve_bytes,
            expert_cache_limit_bytes=expert_cache_limit_bytes,
            transient_slots=transient_slots,
            io_staging_bytes=self.io_staging_bytes,
            execution_workspace_bytes=self.execution_workspace_bytes,
            additional_resident_bytes=additional_resident_bytes,
            resident_discount_bytes=resident_discount_bytes,
            kv_quant=self.kv_quant,
            cache_scope=self.cache_scope,
            island_layer_count=len(self.island_layers),
            mmap_island_layer_count=len(self.mmap_island_layers),
            mmap_islands_wired=self.mmap_island_wired,
            prefetch_slots_per_layer=self.prefetch_slots,
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _pipeline_ledger_for_config(
    config: ExpertStreamingConfig,
) -> ExpertPipelineLedger | None:
    """Enable pipeline attribution only on the instrumented slot-backed path."""

    if not config.resource_telemetry or config.slot_layout == "metal-mmap":
        return None
    return ExpertPipelineLedger(strict=False)


@dataclass(frozen=True)
class RouteWave:
    positions: tuple[int, ...]
    experts: tuple[int, ...]


def partition_route_waves(
    expert_ids: Iterable[int],
    *,
    max_unique_experts: int,
    sort_unique: bool = False,
) -> tuple[RouteWave, ...]:
    """Greedily partition flattened assignments into bounded expert unions."""

    capacity = _integer("max_unique_experts", max_unique_experts, minimum=1)
    experts = tuple(expert_ids)
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for expert in experts:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError("expert ids must be exact integers")
        if expert not in seen:
            seen.add(expert)
            ordered_unique.append(expert)
    if sort_unique:
        ordered_unique.sort()
    waves: list[RouteWave] = []
    for start in range(0, len(ordered_unique), capacity):
        selected = set(ordered_unique[start : start + capacity])
        positions = tuple(
            position for position, expert in enumerate(experts) if expert in selected
        )
        waves.append(
            RouteWave(
                positions=positions,
                experts=tuple(experts[position] for position in positions),
            )
        )
    return tuple(waves)


@dataclass
class KVAdmission:
    runtime: ExpertStreamingRuntime
    tokens: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.runtime.release_kv_tokens(self.tokens)
        self.released = True

    def __enter__(self) -> KVAdmission:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _ReadyRouteGroup:
    """Several independently completed route parts in original route order."""

    def __init__(self, plan: RoutePlan, parts: tuple[ReadyRoute, ...]) -> None:
        if not parts:
            raise ExpertSlotError("incremental miss group must contain a route part")
        pool = parts[0].pool
        if any(part.pool is not pool for part in parts[1:]):
            raise ExpertSlotError("incremental miss parts belong to different pools")
        self.plan = plan
        self.parts = parts
        self.pool = pool
        bindings_by_slot: dict[tuple[int, int], list[Any]] = {}
        for part in parts:
            for binding in part.bindings:
                bindings_by_slot.setdefault(
                    (binding.expert, binding.logical_slot),
                    [],
                ).append(binding)
        try:
            self.bindings = tuple(
                bindings_by_slot[(expert, slot)].pop(0)
                for expert, slot in zip(plan.experts, plan.slots, strict=True)
            )
        except (KeyError, IndexError) as exc:
            raise ExpertSlotError(
                "incremental miss parts do not cover the original route"
            ) from exc
        if any(bindings_by_slot.values()):
            raise ExpertSlotError(
                "incremental miss parts exceed the original route coverage"
            )

    @property
    def slots(self) -> tuple[int, ...]:
        return self.plan.slots

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    def validate(self) -> None:
        for part in self.parts:
            part.validate()


class _RouteCancel:
    """One route-local cancellation source composed with its caller."""

    def __init__(
        self,
        caller: threading.Event | None,
        internal: threading.Event,
    ) -> None:
        self.caller = caller
        self.internal = internal

    def is_set(self) -> bool:
        return self.internal.is_set() or (
            self.caller is not None and self.caller.is_set()
        )


class PendingSplitRoute:
    """One layer transaction with pinned hits and asynchronously loading misses."""

    def __init__(
        self,
        runtime: "ExpertStreamingRuntime",
        layer: int,
        plan: RoutePlan,
        layer_lock: threading.Lock,
        hit_ready: ReadyRoute | None,
        miss_futures: dict[Future[ReadyRoute], RoutePlan],
        policy_txn: RoutePolicyTxn | None = None,
        io_admission: RouteIOAdmission | None = None,
        miss_cancel_event: threading.Event | None = None,
        lifecycle_release: Callable[[], None] | None = None,
        miss_parts: tuple[RoutePlan, ...] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> None:
        self.runtime = runtime
        self.layer = layer
        self.plan = plan
        self._policy_txn = policy_txn or RoutePolicyTxn(rollback=lambda: None)
        self._io_admission = io_admission
        self._policy_observed = False
        self.hit_ready = hit_ready
        self._miss_futures = dict(miss_futures)
        self._miss_ordinals = {
            future: ordinal for ordinal, future in enumerate(self._miss_futures)
        }
        self._all_miss_ordinals = set(self._miss_ordinals.values())
        initial_parts = tuple(self._miss_futures.values())
        self._all_miss_parts = {
            ordinal: part
            for ordinal, part in enumerate(
                initial_parts if miss_parts is None else miss_parts
            )
        }
        self._submitted_miss_ordinals = set(self._miss_ordinals.values())
        self._miss_admissions: dict[int, RouteIOAdmission] = {}
        self._completed_miss_ordinals: set[int] = set()
        self._miss_ready_parts: dict[int, ReadyRoute] = {}
        self._claimed_miss_futures: dict[Future[ReadyRoute], int] = {}
        self._consumer_leases: set[int] = set()
        self._releasing_consumer_leases: set[int] = set()
        self._miss_ready: _ReadyRouteGroup | None = None
        self._aggregate_lease: _ReadyRouteGroup | None = None
        self._aggregate_lease_ordinals: set[int] = set()
        self._miss_cancel_event = miss_cancel_event or threading.Event()
        self._lifecycle_release = lifecycle_release
        self._pipeline_route = pipeline_route
        self._layer_lock = layer_lock
        self._state_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._failure_callbacks = 0
        self._failure_finalizing = False
        self._failure_finalized = False
        self._cleanup_error: BaseException | None = None
        self._ready_cleanup_complete = False
        self._ready_cleanup_finalizing = False
        self._hits_released = hit_ready is None
        self._close_requested = False
        self._finalized = False
        self._closed = False

    def release_hits(self) -> None:
        ready = self.hit_ready
        if ready is None:
            return
        self.hit_ready = None
        self._hits_released = True
        try:
            ready.release(synchronize=False)
        except BaseException as exc:
            self._record_cleanup_error(exc)
            raise

    @property
    def misses_pending(self) -> bool:
        """Whether miss I/O still offers useful work-overlap headroom."""

        with self._state_lock:
            return any(not future.done() for future in self._miss_futures)

    def claim_misses(self, ready: ReadyRoute | _ReadyRouteGroup) -> None:
        """Record the point at which runnable miss bindings are consumed."""

        route = self._pipeline_route
        if route is None:
            return
        experts = tuple(dict.fromkeys(load.expert for load in ready.plan.loads))
        if experts:
            _pipeline_call(
                self.runtime._pipeline_ledger,
                route,
                "claim_misses",
                experts,
            )

    def _attach_miss_future(
        self,
        future: Future[ReadyRoute],
        plan: RoutePlan,
        *,
        ordinal: int,
        io_admission: RouteIOAdmission | None = None,
    ) -> None:
        with self._state_lock:
            self._miss_futures[future] = plan
            self._miss_ordinals[future] = ordinal
            self._all_miss_ordinals.add(ordinal)
            self._all_miss_parts[ordinal] = plan
            self._submitted_miss_ordinals.add(ordinal)
            if io_admission is not None:
                self._miss_admissions[ordinal] = io_admission

    def _retain_lifecycle_after_admission(self) -> None:
        with self._state_lock:
            if (
                self._lifecycle_release is not None
                or self._failure_finalized
                or self._policy_observed
            ):
                return
            lifecycle = self.runtime.slots.retain_admitted_split_lifecycle()
            self._lifecycle_release = lifecycle.release

    def _record_cleanup_error(self, error: BaseException) -> None:
        promote = False
        with self._state_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error
                promote = True
        if promote:
            # Never nest the runtime health lock under Pending state.
            recorder = getattr(self.runtime, "_record_cleanup_error", None)
            if callable(recorder):
                recorder(error)

    def _release_lifecycle(self) -> None:
        with self._state_lock:
            release = self._lifecycle_release
            self._lifecycle_release = None
        if release is None:
            return
        try:
            release()
        except BaseException as exc:
            with self._state_lock:
                if self._lifecycle_release is None:
                    self._lifecycle_release = release
            self._record_cleanup_error(exc)

    @staticmethod
    def _release_routes(routes: Iterable[ReadyRoute]) -> BaseException | None:
        first_error: BaseException | None = None
        for ready in routes:
            try:
                ready.release(synchronize=False)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    def _store_completed_future(
        self,
        future: Future[ReadyRoute],
        ordinal: int,
    ) -> None:
        try:
            ready = future.result()
        except BaseException:
            return
        with self._state_lock:
            self._miss_ready_parts[ordinal] = ready

    def _failure_callback(
        self,
        future: Future[ReadyRoute],
        ordinal: int,
    ) -> None:
        self._store_completed_future(future, ordinal)
        with self._state_lock:
            self._failure_callbacks -= 1
        self._finish_failure_if_ready()

    def abort(self, error: BaseException) -> None:
        """Cancel this transaction without waiting for running miss workers."""

        # Cancellation is the first observable failure action. Every miss part
        # sees this same route-local event, composed with the caller event.
        self._miss_cancel_event.set()
        with self._state_lock:
            if self._failure is not None:
                return
            self._failure = error
            pending = tuple(
                sorted(
                    self._miss_futures,
                    key=lambda future: self._miss_ordinals[future],
                )
            )
            ordinals = {future: self._miss_ordinals[future] for future in pending}
            self._miss_futures.clear()
            self._miss_ordinals.clear()
            self._miss_ready = None
            # Claim every future before failure becomes observable without the
            # state lock. Completed paths and callbacks each consume one claim.
            self._failure_callbacks = len(pending)

        for future in pending:
            future.cancel()
        completed: list[Future[ReadyRoute]] = []
        running: list[Future[ReadyRoute]] = []
        for future in pending:
            (completed if future.done() else running).append(future)
        for future in completed:
            self._failure_callback(future, ordinals[future])
        for future in running:
            future.add_done_callback(
                lambda done, ordinal=ordinals[future]: self._failure_callback(
                    done,
                    ordinal,
                )
            )
        self._finish_failure_if_ready()

    def _finish_failure_if_ready(self) -> None:
        with self._state_lock:
            if (
                self._failure is None
                or self._failure_callbacks
                or self._claimed_miss_futures
                or self._consumer_leases
                or self._releasing_consumer_leases
                or self._aggregate_lease is not None
                or self._ready_cleanup_finalizing
                or self._failure_finalizing
                or self._failure_finalized
            ):
                return
            self._failure_finalizing = True
            failure = self._failure
            policy_observed = self._policy_observed

        try:
            with self._state_lock:
                routes = tuple(
                    self._miss_ready_parts[ordinal]
                    for ordinal in sorted(self._miss_ready_parts)
                )
                self._miss_ready_parts.clear()
                self._miss_ready = None
            release_error = self._release_routes(routes)
            if release_error is not None:
                self._record_cleanup_error(release_error)
            if not policy_observed:
                try:
                    accepted_ordinals = {
                        ordinal
                        for ordinal, admission in self._miss_admissions.items()
                        if admission.any_accepted
                    }
                    if not self._miss_admissions and (
                        self._io_admission is not None
                        and self._io_admission.any_accepted
                    ):
                        accepted_ordinals = set(self._submitted_miss_ordinals)
                    accepted_parts = tuple(
                        self._all_miss_parts[ordinal]
                        for ordinal in sorted(accepted_ordinals)
                    )
                    self.runtime._handle_split_route_failure(
                        self.layer,
                        self.plan,
                        self._policy_txn,
                        failure,
                        accepted_parts=accepted_parts,
                        io_admission=self._io_admission,
                    )
                except BaseException as exc:
                    self._record_cleanup_error(exc)
        finally:
            self._release_lifecycle()
            with self._state_lock:
                self._ready_cleanup_complete = True
                self._failure_finalized = True
                self._failure_finalizing = False
            self._finalize_if_ready()

    def _iter_pipeline_miss_completions(
        self,
        snapshot: tuple[Future[ReadyRoute], ...],
    ) -> Iterable[Future[ReadyRoute]]:
        """Bound a completion step that may block on the existing iterator.

        The readiness scan and ``next(as_completed(...))`` cannot be atomic through
        the public Future API. The measured span is therefore an upper bound that
        can include completion races, iterator work, and telemetry-lock delay.
        """

        route = self._pipeline_route
        assert route is not None
        completions = iter(as_completed(snapshot))
        remaining = set(snapshot)
        while remaining:
            try:
                may_block_for_next = not any(future.done() for future in remaining)
            except Exception:
                may_block_for_next = False
                ledger = self.runtime._pipeline_ledger
                if ledger is not None:
                    try:
                        ledger.mark_incomplete(phase=route.phase)
                    except Exception:
                        pass
            if may_block_for_next:
                _pipeline_call(
                    self.runtime._pipeline_ledger,
                    route,
                    "begin_potentially_blocking_next_miss_step",
                )
            try:
                future = next(completions)
            finally:
                if may_block_for_next:
                    _pipeline_call(
                        self.runtime._pipeline_ledger,
                        route,
                        "end_potentially_blocking_next_miss_step",
                    )
            remaining.discard(future)
            yield future

    def iter_ready_misses(self) -> Iterable[ReadyRoute]:
        """Yield authoritative miss bindings in physical completion order."""

        with self._state_lock:
            snapshot = tuple(self._miss_futures)
        completion_order = (
            as_completed(snapshot)
            if self._pipeline_route is None
            else self._iter_pipeline_miss_completions(snapshot)
        )
        for future in completion_order:
            with self._state_lock:
                if future not in self._miss_futures:
                    continue
                self._miss_futures.pop(future)
                ordinal = self._miss_ordinals.pop(future)
                self._claimed_miss_futures[future] = ordinal
            try:
                ready = future.result()
            except BaseException as exc:
                with self._state_lock:
                    self._claimed_miss_futures.pop(future, None)
                    failure = self._failure
                if failure is not None:
                    self._finish_failure_if_ready()
                    raise failure
                self.abort(exc)
                raise
            with self._state_lock:
                self._claimed_miss_futures.pop(future, None)
                self._miss_ready_parts[ordinal] = ready
                self._completed_miss_ordinals.add(ordinal)
                failure = self._failure
            if failure is not None:
                self._finish_failure_if_ready()
                raise failure
            try:
                self.runtime.slots.raise_if_unhealthy()
            except BaseException as exc:
                self.abort(exc)
                raise
            with self._state_lock:
                is_final_part = (
                    not self._miss_futures and not self._claimed_miss_futures
                )
            if is_final_part:
                try:
                    self.runtime.slots.commit_if_healthy(
                        lambda ordinal=ordinal: self._validate_and_commit_policy(
                            lease_ordinal=ordinal
                        )
                    )
                except BaseException as exc:
                    self.abort(exc)
                    raise
            else:
                with self._state_lock:
                    if self._failure is None and not self._close_requested:
                        self._consumer_leases.add(ordinal)
            with self._state_lock:
                leased = ordinal in self._consumer_leases
                failure = self._failure
                if failure is not None and leased:
                    self._consumer_leases.remove(ordinal)
                    leased = False
            if failure is not None:
                self._finish_failure_if_ready()
                raise failure
            if not leased:
                failure = ExpertSlotError("split route closed before miss yield")
                self.abort(failure)
                self._finish_failure_if_ready()
                raise failure
            yield ready
        if not self._policy_observed:
            try:
                self.runtime.slots.raise_if_unhealthy()
                self.runtime.slots.commit_if_healthy(self._validate_and_commit_policy)
            except BaseException as exc:
                self.abort(exc)
                raise

    def _validate_completed_misses(self) -> None:
        with self._state_lock:
            if self._completed_miss_ordinals != self._all_miss_ordinals:
                raise ExpertSlotError(
                    "incremental miss completion does not cover every route part"
                )

    def _validate_and_commit_policy(self, *, lease_ordinal: int | None = None) -> None:
        """Validate and publish only host policy state and counters."""

        self._validate_completed_misses()
        self._commit_policy(lease_ordinal=lease_ordinal)

    def _prepare_ready_group(self) -> _ReadyRouteGroup | None:
        miss_plan = self.runtime._subset_route_plan(self.plan, hits=False)
        if miss_plan is None:
            return None
        with self._state_lock:
            if self._miss_ready is not None:
                return self._miss_ready
            if self._failure is not None:
                raise self._failure
            if self._close_requested:
                raise ExpertSlotError("split route closed before miss aggregation")
            if self._completed_miss_ordinals != self._all_miss_ordinals:
                raise ExpertSlotError(
                    "incremental miss completion does not cover every route part"
                )
            expected_ordinals = set(self._all_miss_ordinals)
            if set(self._miss_ready_parts) != expected_ordinals:
                raise ExpertSlotError(
                    "incremental miss parts do not cover the original route"
                )
            if self._consumer_leases != expected_ordinals:
                raise ExpertSlotError(
                    "incremental miss aggregation does not own every route part"
                )
            parts = tuple(
                self._miss_ready_parts[ordinal] for ordinal in sorted(expected_ordinals)
            )
            group = _ReadyRouteGroup(miss_plan, parts)
            self._consumer_leases.clear()
            self._aggregate_lease = group
            self._aggregate_lease_ordinals = expected_ordinals
            self._miss_ready = group
            return group

    def finish_misses(self) -> _ReadyRouteGroup | None:
        with self._state_lock:
            if self._miss_ready is not None:
                return self._miss_ready
            empty = not self._miss_futures and not self._miss_ready_parts
        if empty:
            return None
        try:
            for _ready in self.iter_ready_misses():
                pass
            return self._prepare_ready_group()
        except BaseException as exc:
            self.abort(exc)
            with self._state_lock:
                leased_readies = tuple(
                    self._miss_ready_parts[ordinal]
                    for ordinal in sorted(self._consumer_leases)
                )
            for ready in leased_readies:
                try:
                    self.release_miss(ready)
                except BaseException:
                    pass
            self._finish_failure_if_ready()
            raise

    def _commit_policy(self, *, lease_ordinal: int | None = None) -> None:
        committed = False
        with self._state_lock:
            if self._failure is not None:
                return
            if not self._policy_observed:
                incremental_parts = len(self._completed_miss_ordinals)
                self.runtime._publish_route_transaction(
                    self.layer,
                    self.plan,
                    self._policy_txn,
                    incremental_parts=incremental_parts,
                )
                self._policy_observed = True
                committed = True
            if lease_ordinal is not None:
                self._consumer_leases.add(lease_ordinal)
        if committed:
            self._release_lifecycle()

    def release_miss(self, ready: ReadyRoute) -> None:
        """Release one streamed part while retaining sole route ownership."""

        with self._state_lock:
            matches = tuple(
                ordinal
                for ordinal, candidate in self._miss_ready_parts.items()
                if candidate is ready
            )
            if len(matches) != 1:
                raise ExpertSlotError("miss part is not owned by this split route")
            ordinal = matches[0]
            if ordinal not in self._consumer_leases:
                raise ExpertSlotError("miss part has no active consumer lease")
            self._consumer_leases.remove(ordinal)
            self._releasing_consumer_leases.add(ordinal)
            self._miss_ready_parts.pop(ordinal)
            self._miss_ready = None
        release_error: BaseException | None = None
        try:
            ready.release(synchronize=False)
        except BaseException as exc:
            release_error = exc
            self._record_cleanup_error(exc)
        with self._state_lock:
            self._releasing_consumer_leases.remove(ordinal)
        success_error = self._finish_success_close_if_ready()
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if release_error is not None:
            raise release_error
        if success_error is not None:
            raise success_error

    def release_misses(self, ready: _ReadyRouteGroup) -> None:
        """Return one aggregate consumer lease to Pending ownership."""

        with self._state_lock:
            if self._aggregate_lease is not ready:
                raise ExpertSlotError("miss aggregate is not owned by this split route")
            ordinals = tuple(sorted(self._aggregate_lease_ordinals))
            routes = tuple(self._miss_ready_parts[ordinal] for ordinal in ordinals)
            self._aggregate_lease = None
            self._aggregate_lease_ordinals.clear()
            self._miss_ready = None
            self._releasing_consumer_leases.update(ordinals)
            for ordinal in ordinals:
                self._miss_ready_parts.pop(ordinal)
        release_error = self._release_routes(routes)
        if release_error is not None:
            self._record_cleanup_error(release_error)
        with self._state_lock:
            self._releasing_consumer_leases.difference_update(ordinals)
        success_error = self._finish_success_close_if_ready()
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if release_error is not None:
            raise release_error
        if success_error is not None:
            raise success_error

    def _finish_success_close_if_ready(self) -> BaseException | None:
        with self._state_lock:
            if (
                self._failure is not None
                or not self._close_requested
                or self._claimed_miss_futures
                or self._consumer_leases
                or self._releasing_consumer_leases
                or self._aggregate_lease is not None
                or self._ready_cleanup_finalizing
                or self._ready_cleanup_complete
            ):
                return None
            self._ready_cleanup_finalizing = True
            routes = tuple(
                self._miss_ready_parts[ordinal]
                for ordinal in sorted(self._miss_ready_parts)
            )
            self._miss_ready_parts.clear()
            self._miss_ready = None
        release_error = self._release_routes(routes)
        if release_error is not None:
            self._record_cleanup_error(release_error)
        with self._state_lock:
            self._ready_cleanup_complete = True
            self._ready_cleanup_finalizing = False
        self._finalize_if_ready()
        return release_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._state_lock:
            needs_abort = self._failure is None and not self._policy_observed
        if needs_abort:
            self.abort(ExpertSlotError("split route closed before commit"))
        first_error: BaseException | None = None
        try:
            self.release_hits()
        except BaseException as exc:
            first_error = exc
        with self._state_lock:
            failure = self._failure
            self._close_requested = True
        release_error = self._finish_success_close_if_ready()
        if first_error is None:
            first_error = release_error
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if failure is None and first_error is not None:
            raise first_error

    def _finalize_if_ready(self) -> None:
        with self._state_lock:
            if (
                self._finalized
                or not self._close_requested
                or not self._hits_released
                or not self._ready_cleanup_complete
            ):
                return
            self._finalized = True
            pipeline_route = self._pipeline_route
            self._pipeline_route = None
        try:
            if pipeline_route is not None:
                _pipeline_call(
                    self.runtime._pipeline_ledger,
                    pipeline_route,
                    "close",
                )
        finally:
            self._layer_lock.release()

    def __enter__(self) -> "PendingSplitRoute":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        if exc is not None:
            self.abort(exc)
        self.close()


def resident_quant_plan_discount(manifest: Any, resident_quant: str | None) -> int:
    """Bytes the load-time resident quantization removes from the fixed side.

    Computed per manifest tensor with the same scope predicate the loader
    applies, so the plan prices the post-quantization footprint instead of
    the stored BF16 one (kept bytes round up — the discount understates).
    """

    if not resident_quant:
        return 0
    discount = 0
    for tensor in manifest.resident_tensors:
        if tensor.dtype.upper() not in {"BF16", "BFLOAT16"}:
            continue
        name = tensor.tensor
        if name.endswith(".weight"):
            name = name[: -len(".weight")]
        else:
            continue
        if resident_quant_covers(name):
            discount += tensor.length - resident_quant_kept_bytes(
                tensor.length, resident_quant
            )
    return discount


def reconcile_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Resolve the MLX-owned portion and reject a conflicting env cap."""

    # A wired band is registered in MLX's residency set and counted by MLX
    # memory accounting (it already sits on the plan's fixed side); only a
    # paged band lives in the page cache outside the cap.
    paged_band_bytes = (
        0
        if getattr(plan, "mmap_islands_wired", True)
        else getattr(plan, "mmap_island_bytes", 0)
    )
    mlx_limit = (
        plan.total_limit_bytes
        - plan.runtime_reserve_bytes
        - plan.io_staging_bytes
        - paged_band_bytes
    )
    if mlx_limit <= 0:
        raise ExpertStreamingConfigurationError(
            "memory plan leaves no MLX allocation budget"
        )
    # A paged band's pages are the pager's first eviction victims. If MLX's
    # buffer cache may balloon into them, every decode wave re-faults from
    # SSD — fail fast instead. (Measured: 10.85 -> 2.77 tok/s under exactly
    # this squeeze.)
    wired_need = (
        plan.fixed_bytes
        - plan.runtime_reserve_bytes
        - plan.io_staging_bytes
        + plan.persistent_cache_bytes
    )
    if paged_band_bytes and mlx_limit < wired_need:
        raise ExpertStreamingConfigurationError(
            "MLX budget cannot hold the wired footprint next to the paged "
            f"mmap island band: cap {mlx_limit} < wired need {wired_need}; "
            "raise memory_limit_bytes or shrink the band"
        )
    source = os.environ if env is None else env
    existing = source.get("MTPLX_MEMORY_LIMIT_BYTES")
    if existing:
        parsed = parse_memory_bytes(existing)
        if parsed != mlx_limit:
            raise ExpertStreamingConfigurationError(
                "MTPLX_MEMORY_LIMIT_BYTES conflicts with expert streaming plan: "
                f"env={parsed}, planned={mlx_limit}"
            )
    return mlx_limit


def apply_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    mx_module: Any | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the reconciled cap before resident or expert-slot allocation."""

    target_env = os.environ if env is None else env
    limit = reconcile_mlx_memory_cap(plan, env=target_env)
    target_env["MTPLX_MEMORY_LIMIT_BYTES"] = str(limit)
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {
                "applied": False,
                "reason": "mlx_unavailable",
                "error": repr(exc),
                "limit": limit,
            }
    else:
        mx = mx_module
    setter = getattr(mx, "set_memory_limit", None)
    if not callable(setter):
        metal = getattr(mx, "metal", None)
        setter = getattr(metal, "set_memory_limit", None)
    if not callable(setter):
        raise ExpertStreamingConfigurationError("MLX memory limit API is unavailable")
    setter(limit)
    return {"applied": True, "limit": limit}


def mlx_memory_telemetry(mx_module: Any | None = None) -> dict[str, int | str]:
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {"error": repr(exc)}
    else:
        mx = mx_module
    report: dict[str, int | str] = {}
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        getter = getattr(mx, name, None)
        if not callable(getter):
            getter = getattr(getattr(mx, "metal", None), name, None)
        if callable(getter):
            try:
                report[name.removeprefix("get_") + "_bytes"] = int(getter())
            except Exception as exc:
                report[name + "_error"] = repr(exc)
    return report


class ExpertStreamingRuntime:
    """Connect cache policy, checked I/O, fixed slots, and KV admission."""

    def __init__(
        self,
        root: Path,
        spec: ExpertStreamingModelSpec,
        config: ExpertStreamingConfig,
        manifest: ExpertManifest,
        plan: ExpertMemoryPlan,
        reader: PositionalExpertReader,
        slots: ExpertSlotPool,
        *,
        memory_cap_report: dict[str, Any] | None = None,
        integrity_report: dict[str, Any] | None = None,
        pipeline_ledger: ExpertPipelineLedger | None = None,
    ) -> None:
        self.root = root
        self.spec = spec
        self.config = config
        self.manifest = manifest
        self.plan = plan
        self.reader = reader
        self.slots = slots
        self.memory_cap_report = memory_cap_report
        self.integrity_report = integrity_report
        self._pipeline_ledger = pipeline_ledger
        self.counters = CacheCounters()
        self._layer_counters = {
            layer: CacheCounters() for layer in spec.routed_layer_indices
        }
        self._phase_counters = {phase: CacheCounters() for phase in RoutingPhase}
        self._counter_lock = threading.Lock()
        self._global_bank = (
            GlobalExpertSlotBank(
                layer_indices=spec.routed_layer_indices,
                expert_count=spec.expert_count,
                persistent_slots=plan.persistent_slots,
                transient_slots=plan.transient_slots,
                prefill_slots_per_layer=plan.slots_per_layer,
                frequency_decay=config.frequency_decay,
                cache_policy=config.cache_policy,
            )
            if config.cache_scope == "global"
            else None
        )
        # Wired and mmap-banked islands share the routing contract: streamed
        # route entry points reject both, and neither owns slot-pool state.
        self.island_layer_set = frozenset(config.island_layers) | frozenset(
            config.mmap_island_layers
        )
        self._banks = (
            {}
            if self._global_bank is not None
            else {
                layer: LayerExpertSlotBank(
                    expert_count=spec.expert_count,
                    persistent_slots=plan.slots_per_layer,
                    transient_slots=plan.transient_slots,
                    frequency_decay=config.frequency_decay,
                    cache_policy=config.cache_policy,
                    prefetch_slots=plan.prefetch_slots_per_layer,
                )
                for layer in spec.routed_layer_indices
                if layer not in self.island_layer_set
            }
        )
        if self._global_bank is not None:
            # A route holds this lock through hit execution and miss loading.
            # Physical pinning remains the final overwrite fence, while this
            # lock prevents another layer from selecting the same global
            # victim before the current transaction publishes its mapping.
            global_lock = threading.Lock()
            self._layer_locks = {
                layer: global_lock for layer in spec.routed_layer_indices
            }
        else:
            self._layer_locks = {
                layer: threading.Lock() for layer in spec.routed_layer_indices
            }
        self._kv_lock = threading.Lock()
        self._live_kv_tokens = 0
        self._live_kv_peak = 0
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._cleanup_error_lock = threading.Lock()
        self._cleanup_error: BaseException | None = None
        self._mapped_expert_store: Any | None = None
        self._island_store: Any | None = None
        self._banked_island_store: Any | None = None
        self._route_trace_lock = threading.Lock()
        self._route_trace: list[dict[str, Any]] = []
        self._route_trace_epoch = 0
        self._route_trace_decode_step = 0
        self._route_trace_decode_layers_seen: set[int] = set()
        self._incremental_miss_routes = 0
        self._incremental_miss_parts = 0
        self._split_executor = ThreadPoolExecutor(
            max_workers=max(1, plan.transient_slots),
            thread_name_prefix="mtplx-route-miss",
        )
        # Speculative ring loads run off the route-miss executor so a
        # prediction burst can never starve a real miss read.
        self._prefetch_lock = threading.Lock()
        self._prefetch_futures: set[Future] = set()
        self._prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=max(1, plan.prefetch_slots_per_layer),
                thread_name_prefix="mtplx-prefetch",
            )
            if config.prefetch_slots > 0
            else None
        )

    @classmethod
    def open(
        cls,
        root: Path | str,
        manifest_path: Path | str,
        config: ExpertStreamingConfig,
        *,
        spec: ExpertStreamingModelSpec | None = None,
        buffer_allocator: Callable[[int, str], Any] | None = None,
        device_synchronize: Callable[[], None] | None = None,
        apply_memory_cap: bool = True,
        mx_module: Any | None = None,
        env: dict[str, str] | None = None,
        additional_resident_bytes: int = 0,
    ) -> ExpertStreamingRuntime:
        artifact_root = Path(root).resolve()
        model_spec = get_model_spec(config.model_key) if spec is None else spec
        if model_spec.key != config.model_key:
            raise ExpertStreamingConfigurationError("config and spec model keys differ")
        manifest = load_expert_manifest(manifest_path)
        cls._validate_manifest_identity(manifest, model_spec)
        if config.island_layers and not set(config.island_layers) <= set(
            model_spec.routed_layer_indices
        ):
            raise ExpertStreamingConfigurationError(
                "island_layers must be routed layers of "
                f"{model_spec.key}: {sorted(model_spec.routed_layer_indices)[:3]}"
                f"..{sorted(model_spec.routed_layer_indices)[-1]}"
            )
        if config.mmap_island_layers:
            if not set(config.mmap_island_layers) <= set(
                model_spec.routed_layer_indices
            ):
                raise ExpertStreamingConfigurationError(
                    "mmap_island_layers must be routed layers of "
                    f"{model_spec.key}"
                )
            from mtplx.expert_banked import (
                BankedManifestError,
                load_banked_manifest,
            )

            try:
                banked = load_banked_manifest(Path(config.banked_manifest))
            except BankedManifestError as exc:
                raise ExpertStreamingConfigurationError(
                    f"banked manifest is unusable: {exc}"
                ) from exc
            uncovered = sorted(
                set(config.mmap_island_layers) - banked.layer_set
            )
            if uncovered:
                raise ExpertStreamingConfigurationError(
                    f"banked manifest does not cover mmap island layers "
                    f"{uncovered}"
                )
            if banked.expert_count != model_spec.expert_count:
                raise ExpertStreamingConfigurationError(
                    f"banked manifest holds {banked.expert_count} experts per "
                    f"layer; {model_spec.key} routes {model_spec.expert_count}"
                )
        integrity_report = None
        if config.verify_artifact_headers or config.verify_sidecar_hash_at_open:
            if config.verify_sidecar_hash_at_open and manifest.sidecar is None:
                raise ExpertStreamingConfigurationError(
                    "verified-sidecar mode requires a sidecar manifest"
                )
            integrity_report = verify_expert_manifest(
                manifest,
                artifact_root,
                verify_sidecar_hash=config.verify_sidecar_hash_at_open,
            )
        plan = config.memory_plan(
            model_spec,
            additional_resident_bytes=additional_resident_bytes,
            resident_discount_bytes=resident_quant_plan_discount(
                manifest, config.resident_quant
            ),
        )
        if not plan.fits_fixed:
            raise ExpertStreamingConfigurationError(
                f"fixed expert-streaming footprint exceeds limit by {-plan.unallocated_bytes} bytes"
            )
        cap_report = (
            apply_mlx_memory_cap(plan, mx_module=mx_module, env=env)
            if apply_memory_cap
            else None
        )
        pipeline_ledger = _pipeline_ledger_for_config(config)
        pipeline_kwargs = (
            {} if pipeline_ledger is None else {"pipeline_ledger": pipeline_ledger}
        )
        reader = PositionalExpertReader(
            artifact_root,
            max_open_files=config.max_open_files,
            max_read_chunk_bytes=config.max_read_chunk_bytes,
            bypass_page_cache=config.bypass_page_cache,
            **pipeline_kwargs,
        )
        try:
            slots = ExpertSlotPool(
                model_spec,
                plan,
                manifest,
                reader,
                buffer_allocator=buffer_allocator,
                max_inflight_io_bytes=config.max_inflight_io_bytes,
                prefer_sidecar=config.prefer_sidecar,
                verify_hashes=(
                    config.verify_record_hashes
                    and not config.verify_sidecar_hash_at_open
                ),
                device_synchronize=device_synchronize,
                cache_scope=config.cache_scope,
                resource_telemetry=config.resource_telemetry,
                island_layers=(
                    config.island_layers + config.mmap_island_layers
                ),
                **pipeline_kwargs,
            )
        except Exception:
            reader.close()
            raise
        return cls(
            artifact_root,
            model_spec,
            config,
            manifest,
            plan,
            reader,
            slots,
            memory_cap_report=cap_report,
            integrity_report=integrity_report,
            **pipeline_kwargs,
        )

    @staticmethod
    def _validate_manifest_identity(
        manifest: ExpertManifest,
        spec: ExpertStreamingModelSpec,
    ) -> None:
        try:
            validate_expert_manifest_spec(manifest, spec)
        except ExpertManifestError as exc:
            raise ExpertStreamingConfigurationError(
                "manifest does not match pinned model descriptor: " + str(exc)
            ) from exc

    def _record_cleanup_error(self, error: BaseException) -> None:
        with self._cleanup_error_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error

    def _raise_cleanup_error(self) -> None:
        with self._cleanup_error_lock:
            error = self._cleanup_error
        if error is not None:
            raise ExpertSlotError("expert streaming runtime cleanup failed") from error

    def _raise_if_unhealthy(self) -> None:
        self.slots.raise_if_unhealthy()
        self._raise_cleanup_error()

    def _reject_island_layer(self, layer: int) -> None:
        # Island layers execute dense against their own full-resident bank;
        # reaching the streamed route machinery for one is a wiring bug, not
        # a runtime condition to fall back from.
        if layer in self.island_layer_set:
            raise ExpertStreamingConfigurationError(
                f"layer {layer} is a dense island layer; streamed routing is "
                "not available for it"
            )

    def admit_kv_tokens(self, tokens: int) -> KVAdmission:
        # Lifecycle order is close -> slot/runtime health -> KV accounting.
        # Release takes only the KV lock so an existing lease can always drain.
        with self._close_lock:
            if self._closed:
                raise ExpertSlotError("expert streaming runtime is closed")
            if self._closing:
                raise ExpertSlotError("expert streaming runtime is closing")
            self._raise_if_unhealthy()
            count = _integer("tokens", tokens, minimum=1)
            with self._kv_lock:
                requested = self._live_kv_tokens + count
                if requested > self.config.max_live_kv_tokens:
                    raise ExpertStreamingConfigurationError(
                        f"live KV admission {requested} exceeds planned "
                        f"{self.config.max_live_kv_tokens} tokens"
                    )
                self._live_kv_tokens = requested
                self._live_kv_peak = max(self._live_kv_peak, requested)
            return KVAdmission(self, count)

    def release_kv_tokens(self, tokens: int) -> None:
        count = _integer("tokens", tokens, minimum=1)
        with self._kv_lock:
            if count > self._live_kv_tokens:
                raise RuntimeError("KV admission accounting underflow")
            self._live_kv_tokens -= count

    def ensure_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute:
        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            route_plan, policy_txn = self._plan_route_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
            ready: ReadyRoute | None = None
            io_admission = RouteIOAdmission()
            try:
                ready = self.slots.ensure_route(
                    layer,
                    route_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    io_admission=io_admission,
                )
                policy_txn.commit()
            except BaseException as exc:
                if ready is not None:
                    ready.release(synchronize=False)
                self._handle_route_failure(
                    layer,
                    route_plan,
                    policy_txn,
                    exc,
                    io_admission=io_admission,
                )
                raise
            self._observe_plan(layer, route_plan)
            assert ready is not None
            return ready

    def defer_slot_release(self, ready, wave_output) -> None:
        """Queue a pinned route for release after the next generation eval.

        The wave output is an ancestor of the next layer's router indices, so
        the next ``mx.eval`` on the generation thread materializes it; pins
        release then without a per-layer blocking fence. The output reference
        is retained so the lazy graph cannot drop it before that eval.
        """

        pending = getattr(self, "_deferred_slot_releases", None)
        if pending is None:
            pending = []
            self._deferred_slot_releases = pending
        pending.append((ready, wave_output))

    def flush_deferred_slot_releases(self, *, evaluate: bool = False) -> None:
        """Release queued routes; the caller just completed a covering eval.

        ``evaluate=True`` fences the pending wave outputs first and is safe at
        any generation-thread boundary (row end, reset, close) where no later
        eval is guaranteed to cover them.
        """

        pending = getattr(self, "_deferred_slot_releases", None)
        if not pending:
            return
        if evaluate:
            # Local import: this module stays MLX-free at import time so the
            # I/O layer never touches MLX from worker contexts.
            import mlx.core as mx

            mx.eval(*[wave_output for _ready, wave_output in pending])
        first_error: BaseException | None = None
        while pending:
            ready, _wave_output = pending.pop(0)
            try:
                ready.release(synchronize=False)
            except BaseException as error:  # noqa: BLE001 - propagate after drain
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def try_all_hit_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute | None:
        """Pin one fully resident route without wave or split execution.

        The policy probe is side-effect free when any assignment misses,
        allowing the caller to use the regular split route unchanged.
        """

        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        self._reject_island_layer(layer)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            planned = (
                self._global_bank.try_plan_all_hits_transaction(
                    layer,
                    expert_ids,
                    phase=phase,
                )
                if self._global_bank is not None
                else self._banks[layer].try_plan_all_hits_transaction(
                    expert_ids,
                    phase=phase,
                )
            )
            if planned is None:
                return None
            route_plan, policy_txn = planned
            ready: ReadyRoute | None = None

            def publish_route() -> None:
                self._publish_route_transaction(layer, route_plan, policy_txn)

            try:
                ready = self.slots.ensure_route(
                    layer,
                    route_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                self.slots.commit_if_healthy(publish_route)
            except BaseException:
                # A successful all-hit probe has no loads and therefore can
                # never cross the destructive I/O boundary.  Any pin-path
                # failure must restore its decode history and epoch exactly.
                if ready is not None:
                    try:
                        ready.release(synchronize=False)
                    except BaseException as cleanup_error:
                        self._record_cleanup_error(cleanup_error)
                        try:
                            ready.release(synchronize=False)
                        except BaseException as retry_error:
                            self._record_cleanup_error(retry_error)
                try:
                    policy_txn.rollback_publication()
                except BaseException as rollback_error:
                    self._record_cleanup_error(rollback_error)
                raise
            assert ready is not None
            return ready

    def _observe_plan(self, layer: int, plan: RoutePlan) -> None:
        with self._counter_lock:
            self._observe_plan_unlocked(layer, plan)

    def _observe_plan_unlocked(self, layer: int, plan: RoutePlan) -> None:
        self.counters.observe(
            plan,
            expert_record_bytes=self.spec.expert_record_bytes,
        )
        self._layer_counters[layer].observe(
            plan,
            expert_record_bytes=self.spec.expert_record_bytes,
        )
        self._phase_counters[plan.phase].observe(
            plan,
            expert_record_bytes=self.spec.expert_record_bytes,
        )

    def _observe_incremental_unlocked(self, *, routes: int, parts: int) -> None:
        self._incremental_miss_routes += routes
        self._incremental_miss_parts += parts

    def _publish_route_transaction(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        *,
        incremental_parts: int = 0,
    ) -> None:
        counters = (
            self.counters,
            self._layer_counters[layer],
            self._phase_counters[plan.phase],
        )
        with self._counter_lock:
            counter_snapshots = tuple(counter.__dict__.copy() for counter in counters)
            incremental_snapshot = (
                self._incremental_miss_routes,
                self._incremental_miss_parts,
            )
            try:
                self._observe_plan_unlocked(layer, plan)
                if plan.phase is RoutingPhase.DECODE and plan.misses:
                    self._observe_incremental_unlocked(
                        routes=1,
                        parts=incremental_parts,
                    )
                # Publish policy last.  Counter observation is reversible and
                # cannot expose a partial snapshot while this lock is held.
                # Deferring the commit also means an all-hit global route does
                # not need to copy the entire LRU merely to undo a later
                # counter failure.
                policy_txn.commit()
            except BaseException:
                for counter, snapshot in zip(counters, counter_snapshots, strict=True):
                    counter.__dict__.clear()
                    counter.__dict__.update(snapshot)
                (
                    self._incremental_miss_routes,
                    self._incremental_miss_parts,
                ) = incremental_snapshot
                try:
                    policy_txn.rollback_publication()
                except BaseException as rollback_error:
                    self._record_cleanup_error(rollback_error)
                raise

    def _plan_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        if self._global_bank is not None:
            return self._global_bank.plan(layer, expert_ids, phase=phase)
        return self._banks[layer].plan(expert_ids, phase=phase)

    def _plan_route_transaction(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn]:
        if self._global_bank is not None:
            return self._global_bank.plan_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
        return self._banks[layer].plan_transaction(expert_ids, phase=phase)

    def _invalidate_policy_expert(self, layer: int, expert: int) -> int | None:
        if self._global_bank is not None:
            return self._global_bank.invalidate_expert(layer, expert)
        return self._banks[layer].invalidate_expert(expert)

    @staticmethod
    def _subset_route_plan(
        plan: RoutePlan,
        *,
        hits: bool,
    ) -> RoutePlan | None:
        hit_set = set(plan.hits)
        selected_indices = [
            index
            for index, expert in enumerate(plan.experts)
            if (expert in hit_set) is hits
        ]
        selected = tuple(
            (plan.experts[index], plan.slots[index]) for index in selected_indices
        )
        if not selected:
            return None
        return RoutePlan(
            phase=plan.phase,
            experts=tuple(expert for expert, _slot in selected),
            slots=tuple(slot for _expert, slot in selected),
            hits=plan.hits if hits else (),
            misses=() if hits else plan.misses,
            loads=() if hits else plan.loads,
            evictions=() if hits else plan.evictions,
            generations=(
                tuple(plan.generations[index] for index in selected_indices)
                if plan.generations
                else ()
            ),
        )

    @staticmethod
    def _miss_route_parts(plan: RoutePlan) -> tuple[RoutePlan, ...]:
        """Split a miss plan by expert while preserving assignment duplicates."""

        unique_experts = tuple(dict.fromkeys(plan.experts))
        load_experts = tuple(load.expert for load in plan.loads)
        if len(set(load_experts)) != len(load_experts) or set(load_experts) != set(
            unique_experts
        ):
            raise ExpertSlotError(
                "incremental miss experts and slot loads must match one-to-one"
            )
        if len({load.slot for load in plan.loads}) != len(plan.loads):
            raise ExpertSlotError("incremental miss parts must own disjoint slots")
        parts: list[RoutePlan] = []
        for expert in unique_experts:
            positions = tuple(
                index
                for index, candidate in enumerate(plan.experts)
                if candidate == expert
            )
            loads = tuple(load for load in plan.loads if load.expert == expert)
            if len(loads) != 1:
                raise ExpertSlotError(
                    "each incremental miss expert must own exactly one slot load"
                )
            parts.append(
                RoutePlan(
                    phase=plan.phase,
                    experts=tuple(plan.experts[index] for index in positions),
                    slots=tuple(plan.slots[index] for index in positions),
                    hits=(),
                    misses=(expert,),
                    loads=loads,
                    evictions=tuple(
                        eviction
                        for eviction in plan.evictions
                        if eviction.next_expert == expert
                    ),
                    generations=(
                        tuple(plan.generations[index] for index in positions)
                        if plan.generations
                        else ()
                    ),
                )
            )
        return tuple(parts)

    def _rollback_route_loads(self, layer: int, plan: RoutePlan) -> None:
        for load in plan.loads:
            if load.persistent:
                self._invalidate_policy_expert(layer, load.expert)
            try:
                self.slots.invalidate(
                    layer,
                    load.slot,
                    expert=load.expert,
                    generation=load.generation,
                )
            except ExpertSlotError:
                pass

    def _handle_route_failure(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        error: BaseException,
        *,
        io_admission: RouteIOAdmission | None = None,
    ) -> None:
        rollback_safe = (
            not io_admission.any_accepted
            if io_admission is not None
            else (
                isinstance(error, ExpertCompletionFenceError)
                and error.policy_rollback_safe
            )
        )
        if rollback_safe:
            policy_txn.rollback_completion()
            return
        self._rollback_route_loads(layer, plan)

    def _handle_split_route_failure(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        error: BaseException,
        *,
        accepted_parts: tuple[RoutePlan, ...],
        io_admission: RouteIOAdmission | None,
    ) -> None:
        """Restore untouched victims while quarantining accepted split loads."""

        if io_admission is None or not io_admission.any_accepted:
            policy_txn.rollback_completion()
            return
        if not accepted_parts:
            self._handle_route_failure(
                layer,
                plan,
                policy_txn,
                error,
                io_admission=io_admission,
            )
            return

        # Every submitted future has settled before this runs. Remove only the
        # physical records that crossed their part-local admission boundary,
        # then restore the full policy snapshot. Accepted evictions cannot be
        # restored physically, so quarantine those victims again afterward.
        for part in accepted_parts:
            self._rollback_route_loads(layer, part)
        policy_txn.rollback_completion()
        for part in accepted_parts:
            for eviction in part.evictions:
                previous_layer = (
                    layer
                    if eviction.previous_layer is None
                    else eviction.previous_layer
                )
                self._invalidate_policy_expert(
                    previous_layer,
                    eviction.previous_expert,
                )
        if self._global_bank is not None:
            for part in accepted_parts:
                for load in part.loads:
                    if load.persistent and load.generation is not None:
                        self._global_bank.reconcile_slot_generation(
                            load.slot,
                            load.generation,
                        )

    def begin_split_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> PendingSplitRoute:
        """Pin hits now and load misses while the caller evaluates hit work."""

        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        self._reject_island_layer(layer)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        lock.acquire()
        plan: RoutePlan | None = None
        policy_txn: RoutePolicyTxn | None = None
        hit_ready: ReadyRoute | None = None
        pending: PendingSplitRoute | None = None
        pipeline_route: ExpertPipelineRoute | None = None
        io_admission = RouteIOAdmission()
        miss_cancel_event = threading.Event()
        combined_cancel = _RouteCancel(cancel_event, miss_cancel_event)
        try:
            self._raise_if_unhealthy()
            plan, policy_txn = self._plan_route_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
            hit_plan = self._subset_route_plan(plan, hits=True)
            miss_plan = self._subset_route_plan(plan, hits=False)
            miss_parts = (
                self._miss_route_parts(miss_plan)
                if miss_plan is not None and plan.phase is RoutingPhase.DECODE
                else ((miss_plan,) if miss_plan is not None else ())
            )
            pipeline_ledger = self._pipeline_ledger
            if pipeline_ledger is not None:
                try:
                    load_experts = tuple(
                        dict.fromkeys(
                            load.expert
                            for load in (() if miss_plan is None else miss_plan.loads)
                        )
                    )
                    pipeline_route = pipeline_ledger.begin_route(
                        layer=layer,
                        phase=plan.phase,
                        load_experts=load_experts,
                        load_logical_bytes=tuple(
                            self.manifest.record(layer, expert).logical_bytes
                            for expert in load_experts
                        ),
                    )
                except Exception:
                    pipeline_route = None
                    try:
                        pipeline_ledger.mark_incomplete(phase=plan.phase)
                    except Exception:
                        pass
            hit_ready = (
                self.slots.ensure_route(
                    layer,
                    hit_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                if hit_plan is not None
                else None
            )
            pending = PendingSplitRoute(
                runtime=self,
                layer=layer,
                plan=plan,
                layer_lock=lock,
                hit_ready=hit_ready,
                miss_futures={},
                policy_txn=policy_txn,
                io_admission=io_admission,
                miss_cancel_event=miss_cancel_event,
                miss_parts=miss_parts,
                pipeline_route=pipeline_route,
            )
            if miss_plan is not None:
                ensure = (
                    self.slots.ensure_route_part
                    if plan.phase is RoutingPhase.DECODE
                    else self.slots.ensure_route
                )
                for ordinal, miss_part in enumerate(miss_parts):
                    part_admission = io_admission.child()
                    if pipeline_route is None:
                        future = self._split_executor.submit(
                            ensure,
                            layer,
                            miss_part,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                            io_admission=part_admission,
                            route_admitted=pending._retain_lifecycle_after_admission,
                        )
                    else:
                        future = self._split_executor.submit(
                            ensure,
                            layer,
                            miss_part,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                            io_admission=part_admission,
                            route_admitted=pending._retain_lifecycle_after_admission,
                            pipeline_route=pipeline_route,
                        )
                    pending._attach_miss_future(
                        future,
                        miss_part,
                        ordinal=ordinal,
                        io_admission=part_admission,
                    )
            else:
                pending._commit_policy()
            return pending
        except BaseException as setup_error:
            # Mirror the sync-path rollback: without it, a failed hit pin or
            # submit leaves the bank mapping experts to never-loaded slots,
            # wedging every later route on this layer until reset().
            miss_cancel_event.set()
            if pending is not None:
                pending.abort(setup_error)
                pending.close()
            else:
                if pipeline_route is not None:
                    _pipeline_call(
                        self._pipeline_ledger,
                        pipeline_route,
                        "close",
                    )
                if hit_ready is not None:
                    try:
                        hit_ready.release(synchronize=False)
                    except BaseException:
                        pass
                if policy_txn is not None:
                    try:
                        self._handle_route_failure(
                            layer,
                            plan,
                            policy_txn,
                            setup_error,
                            io_admission=io_admission,
                        )
                    except BaseException:
                        pass
                lock.release()
            raise

    def route_waves(
        self,
        expert_ids: Iterable[int],
        *,
        sort_unique: bool = False,
    ) -> tuple[RouteWave, ...]:
        return partition_route_waves(
            expert_ids,
            max_unique_experts=self.plan.transient_slots,
            sort_unique=sort_unique,
        )

    def observe_route(
        self,
        layer: int,
        phase: RoutingPhase | str,
        expert_ids: Iterable[int],
        *,
        token_count: int,
    ) -> None:
        if not self.config.trace_routes:
            return
        normalized_phase = RoutingPhase(phase)
        routed_experts = [int(expert) for expert in expert_ids]
        with self._route_trace_lock:
            entry = {
                "layer": int(layer),
                "phase": normalized_phase.value,
                "trace_epoch": self._route_trace_epoch,
                "token_count": int(token_count),
                "expert_ids": routed_experts,
            }
            if normalized_phase is RoutingPhase.DECODE:
                routed_layers = set(self.spec.routed_layer_indices)
                if layer in self._route_trace_decode_layers_seen:
                    # Preserve evidence rather than silently assigning a
                    # duplicate layer to the current step. The analyzer will
                    # reject the resulting incomplete step sequences.
                    self._route_trace_decode_step += 1
                    self._route_trace_decode_layers_seen.clear()
                entry["decode_step"] = self._route_trace_decode_step
                self._route_trace_decode_layers_seen.add(int(layer))
                if self._route_trace_decode_layers_seen == routed_layers:
                    self._route_trace_decode_step += 1
                    self._route_trace_decode_layers_seen.clear()
            self._route_trace.append(entry)

    def route_trace(self) -> list[dict[str, Any]]:
        with self._route_trace_lock:
            return [dict(entry) for entry in self._route_trace]

    def prepare_prefill_seed(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[int, ...]:
        self._reject_island_layer(layer)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            if self._global_bank is not None:
                return self._global_bank.prepare_prefill_seed(layer, expert_ids)
            return self._banks[layer].prepare_prefill_seed(expert_ids)

    def prefetch_experts(self, layer: int, expert_ids: Iterable[int]) -> int:
        """Speculatively load predicted experts into the layer's ring tier.

        Returns the number of asynchronous loads issued. Zero when prefetch
        is disabled, the layer is an island or unrouted, or every prediction
        is already resident, published, or inflight. A completed load
        publishes through ``commit_prefetch`` and resolves as an ordinary
        ``plan()`` hit; a failed load only invalidates its ring assignment —
        speculation never marks the runtime unhealthy.
        """

        if self.config.prefetch_slots <= 0:
            return 0
        if layer in self.island_layer_set:
            return 0
        bank = self._banks.get(layer)
        if bank is None:
            return 0
        if self._closed or self._closing:
            return 0
        executor = self._prefetch_executor
        if executor is None:
            return 0
        lock = self._layer_locks[layer]
        # Never block the generation thread on a layer transaction: under
        # deferred split-route release the previous token's pending split
        # holds this lock until the next covering-eval flush, and that
        # flush runs on the very thread calling here. Speculation is
        # best-effort — skip the step instead of waiting.
        if not lock.acquire(blocking=False):
            return 0
        try:
            loads = bank.plan_prefetch(expert_ids)
        finally:
            lock.release()
        issued = 0
        for load in loads:
            try:
                future = executor.submit(
                    self.slots.load_speculative,
                    layer,
                    load,
                )
            except RuntimeError:
                # The executor shut down mid-flight; forget the assignment
                # so a later plan_prefetch can reuse the ring slot.
                with lock:
                    bank.invalidate_prefetch(load.expert)
                continue
            with self._prefetch_lock:
                self._prefetch_futures.add(future)
            future.add_done_callback(
                lambda completed, layer=layer, expert=load.expert: (
                    self._finish_prefetch_load(layer, expert, completed)
                )
            )
            issued += 1
        if issued:
            with self._counter_lock:
                self.counters.prefetch_issued += issued
                self._layer_counters[layer].prefetch_issued += issued
        return issued

    def _finish_prefetch_load(
        self,
        layer: int,
        expert: int,
        future: Future,
    ) -> None:
        with self._prefetch_lock:
            self._prefetch_futures.discard(future)
        try:
            future.result()
        except BaseException:
            succeeded = False
        else:
            succeeded = True
        bank = self._banks.get(layer)
        lock = self._layer_locks.get(layer)
        if bank is None or lock is None:
            return
        committed = False
        with lock:
            if succeeded:
                committed = bank.commit_prefetch(expert)
            else:
                bank.invalidate_prefetch(expert)
        if committed:
            with self._counter_lock:
                self.counters.prefetch_committed += 1
                self._layer_counters[layer].prefetch_committed += 1

    def _drain_prefetch_loads(self) -> None:
        """Wait out in-flight speculative loads (bounded single-record reads).

        Reset and close call this before touching pool or bank state. The
        layer locks must NOT be held here: a pending commit callback takes
        its layer lock, and waiting on its future while holding that lock
        would deadlock.
        """

        with self._prefetch_lock:
            pending = tuple(self._prefetch_futures)
        for future in pending:
            try:
                future.result()
            except BaseException:
                pass

    def reset(self) -> None:
        # Deferred pin releases must flush (with a covering fence) before the
        # pool resets, or reset waits forever on the final routes' pins.
        self.flush_deferred_slot_releases(evaluate=True)
        # Straggler speculative loads hold pool lifecycle claims; without
        # this drain the pool reset below would reject them as active
        # routes. The bank reset then retires any not-yet-committed
        # assignment, so a late commit callback publishes nothing.
        self._drain_prefetch_loads()
        locks = tuple(dict.fromkeys(self._layer_locks.values()))
        for lock in locks:
            lock.acquire()
        try:
            self._raise_if_unhealthy()
            self.slots.reset()
            if self._global_bank is not None:
                self._global_bank.reset()
            else:
                for bank in self._banks.values():
                    bank.reset()
            with self._counter_lock:
                self.counters = CacheCounters()
                self._layer_counters = {
                    layer: CacheCounters() for layer in self.spec.routed_layer_indices
                }
                self._phase_counters = {
                    phase: CacheCounters() for phase in RoutingPhase
                }
                self._incremental_miss_routes = 0
                self._incremental_miss_parts = 0
            if self.config.trace_routes:
                with self._route_trace_lock:
                    previous_epoch = self._route_trace_epoch
                    self._route_trace_epoch += 1
                    self._route_trace_decode_step = 0
                    self._route_trace_decode_layers_seen.clear()
                    self._route_trace.append(
                        {
                            "phase": "reset",
                            "previous_trace_epoch": previous_epoch,
                            "trace_epoch": self._route_trace_epoch,
                        }
                    )
        finally:
            for lock in reversed(locks):
                lock.release()

    def snapshot(self, *, mx_module: Any | None = None) -> dict[str, Any]:
        with self._kv_lock:
            live_kv = self._live_kv_tokens
            peak_kv = self._live_kv_peak
        with self._counter_lock:
            cache = self.counters.as_dict()
            cache_by_layer = {
                str(layer): counters.as_dict()
                for layer, counters in self._layer_counters.items()
            }
            cache_by_phase = {
                phase.value: counters.as_dict()
                for phase, counters in self._phase_counters.items()
            }
            incremental_misses = {
                "routes": self._incremental_miss_routes,
                "parts": self._incremental_miss_parts,
            }
        # Never hold the counter lock across slot health/fence inspection.
        slots = self.slots.snapshot()
        snapshot = {
            "model_key": self.spec.key,
            "manifest_sha256": self.manifest.manifest_sha256,
            "memory_plan": {
                "total_limit_bytes": self.plan.total_limit_bytes,
                "fixed_bytes": self.plan.fixed_bytes,
                "persistent_cache_bytes": self.plan.persistent_cache_bytes,
                "slots_per_layer": self.plan.slots_per_layer,
                "cache_scope": self.config.cache_scope,
                "global_persistent_slots": (
                    self.plan.persistent_slots
                    if self.config.cache_scope == "global"
                    else None
                ),
                "transient_slots": self.plan.transient_slots,
                "allocated_bytes": self.plan.allocated_bytes,
                "unallocated_bytes": self.plan.unallocated_bytes,
            },
            "memory_cap": self.memory_cap_report,
            "integrity": self.integrity_report,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "live_kv_tokens": live_kv,
            "live_kv_tokens_peak": peak_kv,
            "cache": cache,
            "cache_by_layer": cache_by_layer,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            "slots": slots,
        }
        if self._global_bank is not None:
            snapshot["global_cache"] = {
                **self._global_bank.snapshot(),
                "resident_experts_by_layer": {
                    str(layer): list(experts)
                    for layer, experts in self._global_bank.resident_experts_by_layer.items()
                },
            }
        if self._mapped_expert_store is not None:
            snapshot["mapped_experts"] = self._mapped_expert_store.snapshot()
        if self._island_store is not None:
            snapshot["island_experts"] = self._island_store.snapshot()
        if self._banked_island_store is not None:
            snapshot["banked_island_experts"] = (
                self._banked_island_store.snapshot()
            )
        if self._pipeline_ledger is not None:
            snapshot["expert_pipeline"] = self._pipeline_ledger.snapshot()
        self._raise_if_unhealthy()
        return snapshot

    def resource_telemetry_snapshot(
        self,
        *,
        mx_module: Any | None = None,
    ) -> dict[str, Any]:
        """Return cheap cumulative counters for the benchmark sampler."""

        with self._counter_lock:
            cache = self.counters.as_dict()
            cache_by_layer = {
                str(layer): counters.as_dict()
                for layer, counters in self._layer_counters.items()
            }
            cache_by_phase = {
                phase.value: counters.as_dict()
                for phase, counters in self._phase_counters.items()
            }
            incremental_misses = {
                "routes": self._incremental_miss_routes,
                "parts": self._incremental_miss_parts,
            }
        # Pool occupancy has independent locks; do not hold the counter lock
        # across that snapshot.
        slots = self.slots.resource_telemetry_snapshot()
        snapshot = {
            "model_key": self.spec.key,
            "quant_bits": self.spec.quant_bits,
            "expert_record_bytes": self.spec.expert_record_bytes,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "cache": cache,
            "cache_by_layer": cache_by_layer,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            **slots,
        }
        if self._pipeline_ledger is not None:
            snapshot["expert_pipeline"] = self._pipeline_ledger.snapshot()
        return snapshot

    def close(self, *, timeout: float | None = None) -> None:
        self.flush_deferred_slot_releases(evaluate=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        if deadline is None:
            self._close_lock.acquire()
        else:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._close_lock.acquire(timeout=remaining):
                raise TimeoutError(
                    "expert streaming runtime close already in progress at deadline"
                )
        try:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if self._closed:
                slots_error: BaseException | None = None
                try:
                    self.slots.close(timeout=remaining)
                except BaseException as exc:
                    slots_error = exc
                if slots_error is not None:
                    raise slots_error
                self._raise_cleanup_error()
                return
            self._closing = True
            # In-flight speculative loads hold pool lifecycle claims; drain
            # them (and stop accepting more) before the pool close waits on
            # active routes.
            self._drain_prefetch_loads()
            if self._prefetch_executor is not None:
                self._prefetch_executor.shutdown(
                    wait=deadline is None,
                    cancel_futures=True,
                )
            slots_error: BaseException | None = None
            try:
                self.slots.close(timeout=remaining)
            except BaseException as exc:
                if not self.slots._closed:
                    raise
                slots_error = exc
            self._split_executor.shutdown(
                wait=deadline is None,
                cancel_futures=True,
            )
            if self._mapped_expert_store is not None:
                self._mapped_expert_store.close()
                self._mapped_expert_store = None
            if self._island_store is not None:
                self._island_store.close()
                self._island_store = None
            if self._banked_island_store is not None:
                self._banked_island_store.close()
                self._banked_island_store = None
            self._closed = True
            self._closing = False
            if slots_error is not None:
                raise slots_error
            self._raise_cleanup_error()
        finally:
            self._close_lock.release()

    def __enter__(self) -> ExpertStreamingRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_configured_expert_runtime(
    root: Path | str,
    manifest_path: Path | str,
    config: ExpertStreamingConfig,
    **kwargs: Any,
) -> ExpertStreamingRuntime:
    try:
        return ExpertStreamingRuntime.open(root, manifest_path, config, **kwargs)
    except ExpertManifestError as exc:
        raise ExpertStreamingConfigurationError(str(exc)) from exc
