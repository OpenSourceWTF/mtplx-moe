"""Low-contention cumulative occupancy counters for runtime work pools."""

from __future__ import annotations

import threading
import time
from bisect import bisect_left
from operator import index
from typing import Callable


def _nonnegative_units(units: object) -> int:
    if isinstance(units, bool):
        raise ValueError("units must be a nonnegative integer")
    try:
        value = index(units)
    except TypeError as exc:
        raise ValueError("units must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError("units must be a nonnegative integer")
    return int(value)


class PoolOccupancy:
    """Integrate queue and active occupancy at state-change boundaries.

    The cumulative nanosecond counters let a sampler recover exact mean
    occupancy over an interval without polling every executor transition.
    ``units`` are bytes for reader work and pinned slot generations for
    completion-fence work.
    """

    def __init__(
        self,
        *,
        worker_capacity: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if isinstance(worker_capacity, bool) or not isinstance(worker_capacity, int):
            raise ValueError("worker_capacity must be a positive integer")
        if worker_capacity <= 0:
            raise ValueError("worker_capacity must be a positive integer")
        self.worker_capacity = worker_capacity
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._started_ns = int(clock_ns())
        self._last_ns = self._started_ns
        self._queued_work = 0
        self._active_work = 0
        self._queued_units = 0
        self._active_units = 0
        self._queued_work_peak = 0
        self._active_work_peak = 0
        self._queued_units_peak = 0
        self._active_units_peak = 0
        self._queued_work_ns = 0
        self._active_work_ns = 0
        self._queued_unit_ns = 0
        self._active_unit_ns = 0
        self._accepted_submissions = 0
        self._started = 0
        self._completed = 0
        self._rejected_submissions = 0

    def _accrue(self, now_ns: int) -> None:
        span = max(0, now_ns - self._last_ns)
        self._queued_work_ns += self._queued_work * span
        self._active_work_ns += self._active_work * span
        self._queued_unit_ns += self._queued_units * span
        self._active_unit_ns += self._active_units * span
        self._last_ns = max(self._last_ns, now_ns)

    def submitted(self, units: object) -> None:
        value = _nonnegative_units(units)
        with self._lock:
            self._accrue(int(self._clock_ns()))
            self._accepted_submissions += 1
            self._queued_work += 1
            self._queued_units += value
            self._queued_work_peak = max(self._queued_work_peak, self._queued_work)
            self._queued_units_peak = max(self._queued_units_peak, self._queued_units)

    def rejected(self, units: object) -> None:
        value = _nonnegative_units(units)
        with self._lock:
            self._accrue(int(self._clock_ns()))
            if (
                self._accepted_submissions < 1
                or self._queued_work < 1
                or self._queued_units < value
            ):
                raise RuntimeError("pool telemetry queue underflow")
            self._accepted_submissions -= 1
            self._rejected_submissions += 1
            self._queued_work -= 1
            self._queued_units -= value

    def started(self, units: object) -> None:
        value = _nonnegative_units(units)
        with self._lock:
            self._accrue(int(self._clock_ns()))
            if self._queued_work < 1 or self._queued_units < value:
                raise RuntimeError("pool telemetry queue underflow")
            self._started += 1
            self._queued_work -= 1
            self._queued_units -= value
            self._active_work += 1
            self._active_units += value
            self._active_work_peak = max(self._active_work_peak, self._active_work)
            self._active_units_peak = max(self._active_units_peak, self._active_units)

    def completed(self, units: object) -> None:
        value = _nonnegative_units(units)
        with self._lock:
            self._accrue(int(self._clock_ns()))
            if self._active_work < 1 or self._active_units < value:
                raise RuntimeError("pool telemetry active underflow")
            self._completed += 1
            self._active_work -= 1
            self._active_units -= value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            now_ns = int(self._clock_ns())
            self._accrue(now_ns)
            return {
                "worker_capacity": self.worker_capacity,
                "observation_ns": max(0, now_ns - self._started_ns),
                "accepted_submissions": self._accepted_submissions,
                "started": self._started,
                "completed": self._completed,
                "rejected_submissions": self._rejected_submissions,
                "queued_work": self._queued_work,
                "active_work": self._active_work,
                "queued_units": self._queued_units,
                "active_units": self._active_units,
                "queued_work_peak": self._queued_work_peak,
                "active_work_peak": self._active_work_peak,
                "queued_units_peak": self._queued_units_peak,
                "active_units_peak": self._active_units_peak,
                "queued_work_ns": self._queued_work_ns,
                "active_work_ns": self._active_work_ns,
                "queued_unit_ns": self._queued_unit_ns,
                "active_unit_ns": self._active_unit_ns,
            }


_PIPELINE_PHASES = ("decode", "prefill", "unscoped")
_PIPELINE_BLOCK_REASONS = (
    "operation_credit",
    "byte_credit",
    "authoritative_reserve",
    "slot_unavailable",
    "pin_held",
    "slot_loading",
)
_PIPELINE_MEASURED_BLOCK_REASONS = ("pin_held", "slot_loading")
_PIPELINE_LATENCY_BOUNDS_NS = (
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
)
_PIPELINE_COUNTER_NAMES = (
    "logical_record_jobs",
    "logical_record_bytes",
    "eligible_record_jobs",
    "eligible_record_bytes",
    "submission_attempts",
    "submission_attempted_record_jobs",
    "submission_attempted_record_bytes",
    "provisional_submission_record_jobs",
    "provisional_submission_record_bytes",
    "accepted_submissions",
    "accepted_record_jobs",
    "accepted_record_bytes",
    "submission_rejections",
    "rejected_record_jobs",
    "rejected_record_bytes",
    "queued_reader_tasks",
    "queued_reader_record_jobs",
    "queued_reader_record_bytes",
    "started_reader_tasks",
    "active_reader_record_jobs",
    "active_reader_record_bytes",
    "completed_reader_tasks",
    "failed_reader_tasks",
    "verified_record_jobs",
    "verified_record_bytes",
    "satisfied_without_submit_record_jobs",
    "satisfied_without_submit_record_bytes",
    "runnable_record_jobs",
    "runnable_record_bytes",
    "claimed_record_jobs",
    "claimed_record_bytes",
    "abandoned_record_jobs",
    "abandoned_record_bytes",
    "failed_record_jobs",
    "failed_record_bytes",
    "claimed_hit_work",
    "claimed_shared_work",
    "abandoned_hit_work",
    "abandoned_shared_work",
    "generation_expert_input_wait_events",
    "started_logical_ranges",
    "started_logical_range_bytes",
    "completed_logical_ranges",
    "completed_logical_range_bytes",
    "reader_thread_cpu_ns",
    "diagnostic_hook_failures",
)
_PIPELINE_GAUGE_NAMES = (
    "open_routes",
    "eligible_unsubmitted_records",
    "eligible_unsubmitted_record_bytes",
    "provisional_submission_records",
    "provisional_submission_record_bytes",
    "accepted_unstarted_records",
    "accepted_unstarted_record_bytes",
    "reader_active_records",
    "reader_active_record_bytes",
    "verified_not_runnable_records",
    "verified_not_runnable_record_bytes",
    "satisfied_not_runnable_records",
    "satisfied_not_runnable_record_bytes",
    "runnable_miss_records",
    "runnable_miss_record_bytes",
    "provisional_reader_tasks",
    "provisional_reader_task_bytes",
    "queued_reader_tasks",
    "queued_reader_task_bytes",
    "active_reader_tasks",
    "active_reader_task_bytes",
    "open_hit_work_spans",
    "open_shared_work_spans",
    "runnable_hit_work",
    "runnable_shared_work",
    "generation_wait_active",
    "active_logical_ranges",
    "active_logical_range_bytes",
)
_PIPELINE_GAUGE_INTEGRALS = {
    "eligible_unsubmitted_records": "eligible_unsubmitted_record_ns",
    "eligible_unsubmitted_record_bytes": "eligible_unsubmitted_record_byte_ns",
    "provisional_submission_records": "provisional_submission_record_ns",
    "provisional_submission_record_bytes": "provisional_submission_record_byte_ns",
    "accepted_unstarted_records": "accepted_unstarted_record_ns",
    "accepted_unstarted_record_bytes": "accepted_unstarted_record_byte_ns",
    "reader_active_records": "reader_active_record_ns",
    "reader_active_record_bytes": "reader_active_record_byte_ns",
    "verified_not_runnable_records": "verified_not_runnable_record_ns",
    "verified_not_runnable_record_bytes": "verified_not_runnable_record_byte_ns",
    "satisfied_not_runnable_records": "satisfied_not_runnable_record_ns",
    "satisfied_not_runnable_record_bytes": "satisfied_not_runnable_record_byte_ns",
    "runnable_miss_records": "runnable_miss_unclaimed_record_ns",
    "runnable_miss_record_bytes": "runnable_miss_unclaimed_record_byte_ns",
    "provisional_reader_tasks": "provisional_reader_task_ns",
    "provisional_reader_task_bytes": "provisional_reader_task_byte_ns",
    "queued_reader_tasks": "queued_reader_task_ns",
    "queued_reader_task_bytes": "queued_reader_task_byte_ns",
    "active_reader_tasks": "active_reader_task_ns",
    "active_reader_task_bytes": "active_reader_task_byte_ns",
    "runnable_hit_work": "runnable_hit_work_ns",
    "runnable_shared_work": "runnable_shared_work_ns",
    "generation_wait_active": "generation_expert_input_wait_ns",
    "active_logical_ranges": "active_logical_range_ns",
    "active_logical_range_bytes": "active_logical_range_byte_ns",
}
_PIPELINE_PRIMARY_NAMES = (
    "generation_thread_expert_input_wait",
    "logical_range_active",
    "reader_completion_active",
    "submitted_queued",
    "eligible_unsubmitted",
    "host_runnable_work",
    "route_publication_pending",
    "no_known_useful_work",
)


def _positive_units(name: str, value: object) -> int:
    result = _nonnegative_units(value)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _expert_tuple(experts: object) -> tuple[int, ...]:
    if not isinstance(experts, tuple):
        raise TypeError("experts must be a tuple of nonnegative integers")
    result: list[int] = []
    for expert in experts:
        if isinstance(expert, bool):
            raise TypeError("expert ids must be nonnegative integers")
        try:
            value = index(expert)
        except TypeError as exc:
            raise TypeError("expert ids must be nonnegative integers") from exc
        if value < 0:
            raise ValueError("expert ids must be nonnegative integers")
        result.append(int(value))
    return tuple(result)


def _logical_byte_tuple(values: object) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("load_logical_bytes must be a tuple of positive integers")
    return tuple(_positive_units("record logical bytes", value) for value in values)


def _phase_name(phase: object) -> str:
    value = getattr(phase, "value", phase)
    if value in {"decode", "prefill"}:
        return str(value)
    return "unscoped"


def _zeroes(names: tuple[str, ...]) -> dict[str, int]:
    return dict.fromkeys(names, 0)


class _FixedHistogram:
    def __init__(self) -> None:
        self.bucket_counts = [0] * (len(_PIPELINE_LATENCY_BOUNDS_NS) + 1)
        self.sample_count = 0

    def observe(self, duration_ns: int) -> None:
        bucket = bisect_left(_PIPELINE_LATENCY_BOUNDS_NS, max(0, duration_ns))
        self.bucket_counts[bucket] += 1
        self.sample_count += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "bounds_ns": _PIPELINE_LATENCY_BOUNDS_NS,
            "bucket_counts": tuple(self.bucket_counts),
            "sample_count": self.sample_count,
            "overflow_count": self.bucket_counts[-1],
        }


class _ReaderTask:
    def __init__(self, experts: tuple[int, ...], logical_bytes: int) -> None:
        self.experts = experts
        self.logical_bytes = logical_bytes
        self.status = "provisional"
        self.accepted = False


class _ExpertPipelineWorkSpan:
    def __init__(
        self,
        ledger: ExpertPipelineLedger,
        *,
        kind: str,
        units: int,
        phase: str,
    ) -> None:
        self._ledger = ledger
        self._kind = kind
        self._units = units
        self._phase = phase
        self._disabled = units == 0
        self._state = "active"

    def claim(self) -> None:
        ledger = self._ledger
        with ledger._lock:
            if self._disabled:
                return
            ledger._accrue_locked(ledger._now_locked())
            if self._state != "active":
                ledger._violation_locked(
                    f"{self._kind} work was already claimed", self._phase
                )
                return
            if not ledger._change_gauge_locked(
                f"runnable_{self._kind}_work", -self._units, self._phase
            ):
                return
            ledger._add_counter_locked(
                f"claimed_{self._kind}_work", self._units, self._phase
            )
            self._state = "claimed"

    def close(self) -> None:
        ledger = self._ledger
        with ledger._lock:
            if self._disabled:
                return
            if self._state == "closed":
                return
            ledger._accrue_locked(ledger._now_locked())
            if self._state == "active":
                if not ledger._change_gauges_locked(
                    {
                        f"runnable_{self._kind}_work": -self._units,
                        f"open_{self._kind}_work_spans": -1,
                    },
                    self._phase,
                ):
                    return
                ledger._add_counter_locked(
                    f"abandoned_{self._kind}_work", self._units, self._phase
                )
            elif self._state == "claimed" and not ledger._change_gauge_locked(
                f"open_{self._kind}_work_spans", -1, self._phase
            ):
                return
            self._state = "closed"


class ExpertPipelineLedger:
    """Bounded diagnostic ledger with one local lock and no raw event log."""

    def __init__(
        self,
        *,
        strict: bool = False,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.strict = strict
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._started_ns = int(clock_ns())
        self._last_ns = self._started_ns
        self._next_route_id = 1
        self._next_range_id = 1
        self._routes: dict[int, ExpertPipelineRoute] = {}
        self._active_ranges: dict[int, tuple[int, int, str]] = {}
        self._generation_wait_route: int | None = None
        self._invariant_failures = 0
        self._phase_invariant_failures = dict.fromkeys(_PIPELINE_PHASES, 0)
        self._explicitly_incomplete = False
        self._phase_explicitly_incomplete = dict.fromkeys(_PIPELINE_PHASES, False)
        self._counters = _zeroes(_PIPELINE_COUNTER_NAMES)
        self._gauges = _zeroes(_PIPELINE_GAUGE_NAMES)
        integral_names = tuple(_PIPELINE_GAUGE_INTEGRALS.values()) + (
            "generation_wait_storage_active_ns",
            "generation_wait_reader_task_active_ns",
            "generation_wait_submitted_queued_ns",
            "generation_wait_eligible_unsubmitted_ns",
            "generation_wait_runnable_ns",
        )
        self._integrals_ns = _zeroes(integral_names)
        self._primary_integrals_ns = _zeroes(_PIPELINE_PRIMARY_NAMES)
        self._phase_counters = {
            phase: _zeroes(_PIPELINE_COUNTER_NAMES) for phase in _PIPELINE_PHASES
        }
        self._phase_gauges = {
            phase: _zeroes(_PIPELINE_GAUGE_NAMES) for phase in _PIPELINE_PHASES
        }
        self._phase_integrals = {
            phase: _zeroes(integral_names) for phase in _PIPELINE_PHASES
        }
        self._phase_primary = {
            phase: _zeroes(_PIPELINE_PRIMARY_NAMES) for phase in _PIPELINE_PHASES
        }
        self._phase_observation_ns = dict.fromkeys(_PIPELINE_PHASES, 0)
        self._block_counts = dict.fromkeys(_PIPELINE_MEASURED_BLOCK_REASONS, 0)
        self._block_ns = dict.fromkeys(_PIPELINE_MEASURED_BLOCK_REASONS, 0)
        self._phase_block_counts = {
            phase: dict.fromkeys(_PIPELINE_MEASURED_BLOCK_REASONS, 0)
            for phase in _PIPELINE_PHASES
        }
        self._phase_block_ns = {
            phase: dict.fromkeys(_PIPELINE_MEASURED_BLOCK_REASONS, 0)
            for phase in _PIPELINE_PHASES
        }
        self._range_latency = _FixedHistogram()
        self._record_latency = _FixedHistogram()
        self._phase_range_latency = {
            phase: _FixedHistogram() for phase in _PIPELINE_PHASES
        }
        self._phase_record_latency = {
            phase: _FixedHistogram() for phase in _PIPELINE_PHASES
        }

    def _now_locked(self) -> int:
        return int(self._clock_ns())

    def _violation_locked(self, message: str, phase: str = "unscoped") -> bool:
        self._invariant_failures += 1
        self._phase_invariant_failures[phase] += 1
        if self.strict:
            raise RuntimeError(message)
        return False

    def _add_counter_locked(self, name: str, amount: int, phase: str) -> None:
        self._counters[name] += amount
        self._phase_counters[phase][name] += amount

    def _change_gauge_locked(self, name: str, delta: int, phase: str) -> bool:
        if self._gauges[name] + delta < 0:
            return self._violation_locked(
                f"expert pipeline gauge underflow: {name}", phase
            )
        if self._phase_gauges[phase][name] + delta < 0:
            return self._violation_locked(
                f"expert pipeline phase gauge underflow: {phase}.{name}", phase
            )
        self._gauges[name] += delta
        self._phase_gauges[phase][name] += delta
        return True

    def _change_gauges_locked(self, deltas: dict[str, int], phase: str) -> bool:
        for name, delta in deltas.items():
            if self._gauges[name] + delta < 0:
                return self._violation_locked(
                    f"expert pipeline gauge underflow: {name}", phase
                )
            if self._phase_gauges[phase][name] + delta < 0:
                return self._violation_locked(
                    f"expert pipeline phase gauge underflow: {phase}.{name}", phase
                )
        for name, delta in deltas.items():
            self._gauges[name] += delta
            self._phase_gauges[phase][name] += delta
        return True

    @staticmethod
    def _primary_state(gauges: dict[str, int]) -> str:
        if gauges["generation_wait_active"]:
            return "generation_thread_expert_input_wait"
        if gauges["active_logical_ranges"]:
            return "logical_range_active"
        if gauges["active_reader_tasks"] or gauges["reader_active_records"]:
            return "reader_completion_active"
        if gauges["provisional_reader_tasks"] or gauges["queued_reader_tasks"]:
            return "submitted_queued"
        if gauges["eligible_unsubmitted_records"]:
            return "eligible_unsubmitted"
        if (
            gauges["runnable_miss_records"]
            or gauges["runnable_hit_work"]
            or gauges["runnable_shared_work"]
        ):
            return "host_runnable_work"
        if (
            gauges["verified_not_runnable_records"]
            or gauges["satisfied_not_runnable_records"]
        ):
            return "route_publication_pending"
        return "no_known_useful_work"

    @staticmethod
    def _phase_observation_active(gauges: dict[str, int]) -> bool:
        return bool(
            gauges["open_routes"]
            or gauges["active_logical_ranges"]
            or gauges["open_hit_work_spans"]
            or gauges["open_shared_work_spans"]
            or gauges["generation_wait_active"]
        )

    def _accrue_metric_set_locked(
        self,
        gauges: dict[str, int],
        integrals: dict[str, int],
        primary: dict[str, int],
        span: int,
        *,
        accrue_primary: bool,
    ) -> None:
        for gauge, integral in _PIPELINE_GAUGE_INTEGRALS.items():
            integrals[integral] += gauges[gauge] * span
        if accrue_primary:
            primary[self._primary_state(gauges)] += span

    def _accrue_locked(self, now_ns: int) -> None:
        if now_ns < self._last_ns:
            self._violation_locked("expert pipeline clock moved backwards")
            now_ns = self._last_ns
        span = now_ns - self._last_ns
        if not span:
            return
        self._accrue_metric_set_locked(
            self._gauges,
            self._integrals_ns,
            self._primary_integrals_ns,
            span,
            accrue_primary=True,
        )
        for phase in _PIPELINE_PHASES:
            phase_active = self._phase_observation_active(self._phase_gauges[phase])
            self._accrue_metric_set_locked(
                self._phase_gauges[phase],
                self._phase_integrals[phase],
                self._phase_primary[phase],
                span,
                accrue_primary=phase_active,
            )
            if phase_active:
                self._phase_observation_ns[phase] += span
            phase_gauges = self._phase_gauges[phase]
            phase_integrals = self._phase_integrals[phase]
            phase_waiting = phase_gauges["generation_wait_active"] > 0
            if phase_waiting and phase_gauges["active_logical_ranges"] > 0:
                phase_integrals["generation_wait_storage_active_ns"] += span
            if phase_waiting and phase_gauges["active_reader_tasks"] > 0:
                phase_integrals["generation_wait_reader_task_active_ns"] += span
            if phase_waiting and (
                phase_gauges["provisional_reader_tasks"] > 0
                or phase_gauges["queued_reader_tasks"] > 0
            ):
                phase_integrals["generation_wait_submitted_queued_ns"] += span
            if phase_waiting and phase_gauges["eligible_unsubmitted_records"] > 0:
                phase_integrals["generation_wait_eligible_unsubmitted_ns"] += span
            phase_runnable = (
                phase_gauges["runnable_miss_records"]
                + phase_gauges["runnable_hit_work"]
                + phase_gauges["runnable_shared_work"]
            ) > 0
            if phase_waiting and phase_runnable:
                phase_integrals["generation_wait_runnable_ns"] += span
        waiting = self._gauges["generation_wait_active"] > 0
        storage = self._gauges["active_logical_ranges"] > 0
        reader_active = self._gauges["active_reader_tasks"] > 0
        runnable = (
            self._gauges["runnable_miss_records"]
            + self._gauges["runnable_hit_work"]
            + self._gauges["runnable_shared_work"]
        ) > 0
        if waiting and storage:
            self._integrals_ns["generation_wait_storage_active_ns"] += span
        if waiting and reader_active:
            self._integrals_ns["generation_wait_reader_task_active_ns"] += span
        if waiting and (
            self._gauges["provisional_reader_tasks"] > 0
            or self._gauges["queued_reader_tasks"] > 0
        ):
            self._integrals_ns["generation_wait_submitted_queued_ns"] += span
        if waiting and self._gauges["eligible_unsubmitted_records"] > 0:
            self._integrals_ns["generation_wait_eligible_unsubmitted_ns"] += span
        if waiting and runnable:
            self._integrals_ns["generation_wait_runnable_ns"] += span
        self._last_ns = now_ns

    def begin_route(
        self,
        *,
        layer: int,
        phase: object,
        load_experts: tuple[int, ...],
        load_logical_bytes: tuple[int, ...],
    ) -> ExpertPipelineRoute:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        phase_name = _phase_name(phase)
        experts = _expert_tuple(load_experts)
        byte_values = _logical_byte_tuple(load_logical_bytes)
        if len(experts) != len(byte_values):
            raise ValueError("load experts and logical bytes must have equal length")
        with self._lock:
            self._accrue_locked(self._now_locked())
            if len(set(experts)) != len(experts):
                self._violation_locked(
                    "duplicate load expert in pipeline route", phase_name
                )
                return ExpertPipelineRoute.disabled(self)
            route_id = self._next_route_id
            self._next_route_id += 1
            route = ExpertPipelineRoute(
                self,
                route_id=route_id,
                layer=layer,
                phase=phase_name,
                records=dict(zip(experts, byte_values, strict=True)),
            )
            self._routes[route_id] = route
            records = len(experts)
            logical_bytes = sum(byte_values)
            for name, amount in (
                ("logical_record_jobs", records),
                ("logical_record_bytes", logical_bytes),
                ("eligible_record_jobs", records),
                ("eligible_record_bytes", logical_bytes),
            ):
                self._add_counter_locked(name, amount, phase_name)
            self._change_gauge_locked("open_routes", 1, phase_name)
            self._change_gauge_locked(
                "eligible_unsubmitted_records", records, phase_name
            )
            self._change_gauge_locked(
                "eligible_unsubmitted_record_bytes", logical_bytes, phase_name
            )
            return route

    def begin_hit_work(
        self,
        experts: tuple[int, ...],
        *,
        phase: object = "unscoped",
    ) -> _ExpertPipelineWorkSpan:
        values = _expert_tuple(experts)
        if not values:
            raise ValueError("hit work must contain at least one expert")
        phase_name = _phase_name(phase)
        with self._lock:
            self._accrue_locked(self._now_locked())
            if len(set(values)) != len(values):
                self._violation_locked("duplicate expert in hit work", phase_name)
                return _ExpertPipelineWorkSpan(
                    self, kind="hit", units=0, phase=phase_name
                )
            self._change_gauges_locked(
                {
                    "open_hit_work_spans": 1,
                    "runnable_hit_work": len(values),
                },
                phase_name,
            )
            return _ExpertPipelineWorkSpan(
                self, kind="hit", units=len(values), phase=phase_name
            )

    def begin_shared_work(
        self,
        *,
        phase: object = "unscoped",
    ) -> _ExpertPipelineWorkSpan:
        phase_name = _phase_name(phase)
        with self._lock:
            self._accrue_locked(self._now_locked())
            self._change_gauges_locked(
                {
                    "open_shared_work_spans": 1,
                    "runnable_shared_work": 1,
                },
                phase_name,
            )
            return _ExpertPipelineWorkSpan(
                self, kind="shared", units=1, phase=phase_name
            )

    def range_started(
        self,
        logical_bytes: object,
        *,
        phase: object = "unscoped",
    ) -> int:
        byte_count = _positive_units("logical_bytes", logical_bytes)
        phase_name = _phase_name(phase)
        with self._lock:
            now_ns = self._now_locked()
            self._accrue_locked(now_ns)
            token = self._next_range_id
            self._next_range_id += 1
            self._active_ranges[token] = (byte_count, now_ns, phase_name)
            self._add_counter_locked("started_logical_ranges", 1, phase_name)
            self._add_counter_locked(
                "started_logical_range_bytes", byte_count, phase_name
            )
            self._change_gauge_locked("active_logical_ranges", 1, phase_name)
            self._change_gauge_locked(
                "active_logical_range_bytes", byte_count, phase_name
            )
            return token

    def range_completed(self, token: object) -> None:
        with self._lock:
            now_ns = self._now_locked()
            self._accrue_locked(now_ns)
            if isinstance(token, bool) or not isinstance(token, int):
                self._violation_locked("range token must be an integer")
                return
            active = self._active_ranges.get(token)
            if active is None:
                self._violation_locked(f"unknown active range token {token}")
                return
            byte_count, started_ns, phase_name = active
            if not self._change_gauges_locked(
                {
                    "active_logical_ranges": -1,
                    "active_logical_range_bytes": -byte_count,
                },
                phase_name,
            ):
                return
            self._active_ranges.pop(token)
            self._add_counter_locked("completed_logical_ranges", 1, phase_name)
            self._add_counter_locked(
                "completed_logical_range_bytes", byte_count, phase_name
            )
            self._range_latency.observe(now_ns - started_ns)
            self._phase_range_latency[phase_name].observe(now_ns - started_ns)

    def mark_incomplete(self, *, phase: object = "unscoped") -> None:
        """Fail open while retaining an explicit incomplete-coverage signal."""
        phase_name = _phase_name(phase)
        with self._lock:
            self._explicitly_incomplete = True
            self._phase_explicitly_incomplete[phase_name] = True
            self._add_counter_locked("diagnostic_hook_failures", 1, phase_name)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            now_ns = self._now_locked()
            self._accrue_locked(now_ns)
            range_histogram = self._range_latency.snapshot()
            record_histogram = self._record_latency.snapshot()
            incomplete = (
                self._invariant_failures > 0
                or self._counters["submission_rejections"] > 0
                or self._explicitly_incomplete
            )
            block_coverage = {
                reason: (
                    "measured"
                    if reason in _PIPELINE_MEASURED_BLOCK_REASONS
                    else "unavailable"
                )
                for reason in _PIPELINE_BLOCK_REASONS
            }
            return {
                "schema": "mtplx-expert-pipeline-attribution-v1",
                "strict": self.strict,
                "observation_ns": max(0, now_ns - self._started_ns),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "integrals_ns": dict(self._integrals_ns),
                "primary_integrals_ns": dict(self._primary_integrals_ns),
                "by_phase": {
                    phase: {
                        "observation_ns": self._phase_observation_ns[phase],
                        "counters": dict(self._phase_counters[phase]),
                        "gauges": dict(self._phase_gauges[phase]),
                        "integrals_ns": dict(self._phase_integrals[phase]),
                        "primary_integrals_ns": dict(self._phase_primary[phase]),
                        "block_counts": dict(self._phase_block_counts[phase]),
                        "block_ns": dict(self._phase_block_ns[phase]),
                        "block_coverage": dict(block_coverage),
                        "histograms": {
                            "logical_range_latency_ns": self._phase_range_latency[
                                phase
                            ].snapshot(),
                            "complete_record_latency_ns": self._phase_record_latency[
                                phase
                            ].snapshot(),
                        },
                        "invariant_failures": self._phase_invariant_failures[phase],
                        "coverage": {
                            "attribution": (
                                "incomplete"
                                if (
                                    self._phase_invariant_failures[phase] > 0
                                    or self._phase_counters[phase][
                                        "submission_rejections"
                                    ]
                                    > 0
                                    or self._phase_explicitly_incomplete[phase]
                                )
                                else "measured"
                            )
                        },
                    }
                    for phase in _PIPELINE_PHASES
                },
                "block_counts": dict(self._block_counts),
                "block_ns": dict(self._block_ns),
                "block_coverage": block_coverage,
                "histograms": {
                    "logical_range_latency_ns": range_histogram,
                    "complete_record_latency_ns": record_histogram,
                },
                "invariant_failures": self._invariant_failures,
                "coverage": {
                    "attribution": "incomplete" if incomplete else "measured",
                    "logical_record_lifecycle": "measured",
                    "reader_task_accounting": "measured",
                    "logical_range_accounting": "measured",
                    "host_runnable_work": "measured",
                    "generation_expert_input_wait": "measured",
                    "outer_split_executor_queue": "unavailable",
                    "eligible_unsubmitted_cause": "unattributed",
                    "admitted_read_ranges": "unavailable",
                    "scheduled_read_ranges": "unavailable",
                    "physical_device_operations": "unavailable",
                    "physical_device_bytes": "unavailable",
                    "physical_device_queue_depth": "unavailable",
                    "gpu_expert_wait": "unavailable",
                },
            }


class ExpertPipelineRoute:
    """Unique authoritative miss-record lifecycle owned by one ledger."""

    _STATE_GAUGES = {
        "eligible": (
            "eligible_unsubmitted_records",
            "eligible_unsubmitted_record_bytes",
        ),
        "provisional": (
            "provisional_submission_records",
            "provisional_submission_record_bytes",
        ),
        "accepted": (
            "accepted_unstarted_records",
            "accepted_unstarted_record_bytes",
        ),
        "reader-active": ("reader_active_records", "reader_active_record_bytes"),
        "verified": (
            "verified_not_runnable_records",
            "verified_not_runnable_record_bytes",
        ),
        "satisfied": (
            "satisfied_not_runnable_records",
            "satisfied_not_runnable_record_bytes",
        ),
        "runnable": ("runnable_miss_records", "runnable_miss_record_bytes"),
    }

    def __init__(
        self,
        ledger: ExpertPipelineLedger,
        *,
        route_id: int,
        layer: int,
        phase: str,
        records: dict[int, int],
        disabled: bool = False,
    ) -> None:
        self._ledger = ledger
        self._route_id = route_id
        self.layer = layer
        self.phase = phase
        self.load_experts = tuple(records)
        self._record_bytes = records
        self._states = dict.fromkeys(records, "eligible")
        self._tasks: dict[tuple[int, ...], _ReaderTask] = {}
        self._record_started_ns: dict[int, int] = {}
        self._disabled = disabled
        self._closed = False

    @classmethod
    def disabled(cls, ledger: ExpertPipelineLedger) -> ExpertPipelineRoute:
        return cls(
            ledger,
            route_id=0,
            layer=0,
            phase="unscoped",
            records={},
            disabled=True,
        )

    def _ensure_open_locked(self) -> bool:
        if self._disabled:
            return False
        if self._closed:
            return self._violation_locked("expert pipeline route is closed")
        return True

    def _violation_locked(self, message: str) -> bool:
        return self._ledger._violation_locked(message, self.phase)

    def _select_locked(
        self,
        experts: object,
        *,
        expected: str | set[str],
    ) -> tuple[int, ...] | None:
        values = _expert_tuple(experts)
        if not values:
            self._violation_locked("record transition requires at least one expert")
            return None
        if len(set(values)) != len(values):
            self._violation_locked("duplicate expert in record transition")
            return None
        allowed = {expected} if isinstance(expected, str) else expected
        for expert in values:
            state = self._states.get(expert)
            if state is None:
                self._violation_locked(
                    f"expert {expert} is not owned by this pipeline route"
                )
                return None
            if state not in allowed:
                expected_text = "/".join(sorted(allowed))
                self._violation_locked(
                    f"record {expert} is {state}; expected {expected_text}"
                )
                return None
        return values

    def _bytes(self, experts: tuple[int, ...]) -> int:
        return sum(self._record_bytes[expert] for expert in experts)

    def _move_records_locked(
        self,
        experts: tuple[int, ...],
        new_state: str,
        *,
        extra_deltas: dict[str, int] | None = None,
    ) -> bool:
        deltas = dict(extra_deltas or {})
        for expert in experts:
            old_state = self._states[expert]
            byte_count = self._record_bytes[expert]
            old_gauges = self._STATE_GAUGES.get(old_state)
            new_gauges = self._STATE_GAUGES.get(new_state)
            if old_gauges is not None:
                deltas[old_gauges[0]] = deltas.get(old_gauges[0], 0) - 1
                deltas[old_gauges[1]] = deltas.get(old_gauges[1], 0) - byte_count
            if new_gauges is not None:
                deltas[new_gauges[0]] = deltas.get(new_gauges[0], 0) + 1
                deltas[new_gauges[1]] = deltas.get(new_gauges[1], 0) + byte_count
        if not self._ledger._change_gauges_locked(deltas, self.phase):
            return False
        for expert in experts:
            self._states[expert] = new_state
        return True

    def observe_block(self, expert: int, reason: str, *, elapsed_ns: object) -> None:
        elapsed = _nonnegative_units(elapsed_ns)
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            if self._select_locked((expert,), expected="eligible") is None:
                return
            if reason not in ledger._block_counts:
                self._violation_locked(
                    f"pipeline block reason {reason!r} is unavailable"
                )
                return
            ledger._block_counts[reason] += 1
            ledger._block_ns[reason] += elapsed
            ledger._phase_block_counts[self.phase][reason] += 1
            ledger._phase_block_ns[self.phase][reason] += elapsed

    def satisfied_without_submit(self, experts: tuple[int, ...]) -> None:
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            values = self._select_locked(experts, expected="eligible")
            if values is None:
                return
            if not self._move_records_locked(values, "satisfied"):
                return
            jobs = len(values)
            byte_count = self._bytes(values)
            ledger._add_counter_locked(
                "satisfied_without_submit_record_jobs", jobs, self.phase
            )
            ledger._add_counter_locked(
                "satisfied_without_submit_record_bytes", byte_count, self.phase
            )

    def submission_attempted(self, experts: tuple[int, ...]) -> None:
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            values = self._select_locked(experts, expected="eligible")
            if values is None:
                return
            if values in self._tasks:
                self._violation_locked("reader task was attempted twice")
                return
            byte_count = self._bytes(values)
            if not self._move_records_locked(
                values,
                "provisional",
                extra_deltas={
                    "provisional_reader_tasks": 1,
                    "provisional_reader_task_bytes": byte_count,
                },
            ):
                return
            self._tasks[values] = _ReaderTask(values, byte_count)
            for name, amount in (
                ("submission_attempts", 1),
                ("submission_attempted_record_jobs", len(values)),
                ("submission_attempted_record_bytes", byte_count),
                ("provisional_submission_record_jobs", len(values)),
                ("provisional_submission_record_bytes", byte_count),
            ):
                ledger._add_counter_locked(name, amount, self.phase)

    def submission_accepted(self, experts: tuple[int, ...]) -> None:
        values = _expert_tuple(experts)
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            task = self._tasks.get(values)
            if task is None:
                self._violation_locked("submission acceptance has no attempt")
                return
            if task.accepted:
                self._violation_locked("submission was already accepted")
                return
            byte_count = task.logical_bytes
            if task.status == "provisional":
                if self._select_locked(values, expected="provisional") is None:
                    return
                if not self._move_records_locked(
                    values,
                    "accepted",
                    extra_deltas={
                        "provisional_reader_tasks": -1,
                        "provisional_reader_task_bytes": -byte_count,
                        "queued_reader_tasks": 1,
                        "queued_reader_task_bytes": byte_count,
                    },
                ):
                    return
                task.status = "queued"
                for name, amount in (
                    ("queued_reader_tasks", 1),
                    ("queued_reader_record_jobs", len(values)),
                    ("queued_reader_record_bytes", byte_count),
                ):
                    ledger._add_counter_locked(name, amount, self.phase)
            elif task.status not in {"active", "completed", "failed"}:
                self._violation_locked(
                    f"cannot accept reader task in state {task.status}"
                )
                return
            task.accepted = True
            for name, amount in (
                ("accepted_submissions", 1),
                ("accepted_record_jobs", len(values)),
                ("accepted_record_bytes", byte_count),
            ):
                ledger._add_counter_locked(name, amount, self.phase)
            if task.status in {"completed", "failed"}:
                self._tasks.pop(values)

    def submission_rejected(self, experts: tuple[int, ...]) -> None:
        values = _expert_tuple(experts)
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            task = self._tasks.get(values)
            if task is None or task.status != "provisional" or task.accepted:
                self._violation_locked(
                    "submission rejection has no provisional attempt"
                )
                return
            if self._select_locked(values, expected="provisional") is None:
                return
            byte_count = task.logical_bytes
            if not self._move_records_locked(
                values,
                "eligible",
                extra_deltas={
                    "provisional_reader_tasks": -1,
                    "provisional_reader_task_bytes": -byte_count,
                },
            ):
                return
            self._tasks.pop(values)
            for name, amount in (
                ("submission_rejections", 1),
                ("rejected_record_jobs", len(values)),
                ("rejected_record_bytes", byte_count),
            ):
                ledger._add_counter_locked(name, amount, self.phase)

    def reader_started(self, experts: tuple[int, ...]) -> None:
        values = _expert_tuple(experts)
        ledger = self._ledger
        with ledger._lock:
            now_ns = ledger._now_locked()
            ledger._accrue_locked(now_ns)
            if not self._ensure_open_locked():
                return
            task = self._tasks.get(values)
            if task is None or task.status not in {"provisional", "queued"}:
                self._violation_locked("reader start has no pending task")
                return
            expected = "provisional" if task.status == "provisional" else "accepted"
            if self._select_locked(values, expected=expected) is None:
                return
            byte_count = task.logical_bytes
            old_task = task.status
            deltas = {
                "active_reader_tasks": 1,
                "active_reader_task_bytes": byte_count,
            }
            if old_task == "provisional":
                deltas["provisional_reader_tasks"] = -1
                deltas["provisional_reader_task_bytes"] = -byte_count
            else:
                deltas["queued_reader_tasks"] = -1
                deltas["queued_reader_task_bytes"] = -byte_count
            if not self._move_records_locked(
                values, "reader-active", extra_deltas=deltas
            ):
                return
            task.status = "active"
            for expert in values:
                self._record_started_ns[expert] = now_ns
            for name, amount in (
                ("started_reader_tasks", 1),
                ("active_reader_record_jobs", len(values)),
                ("active_reader_record_bytes", byte_count),
            ):
                ledger._add_counter_locked(name, amount, self.phase)

    def record_verified(self, expert: int) -> None:
        ledger = self._ledger
        with ledger._lock:
            now_ns = ledger._now_locked()
            ledger._accrue_locked(now_ns)
            if not self._ensure_open_locked():
                return
            values = self._select_locked((expert,), expected="reader-active")
            if values is None:
                return
            started_ns = self._record_started_ns.get(expert)
            if started_ns is None:
                self._violation_locked(f"record {expert} has no reader start time")
                return
            if not self._move_records_locked(values, "verified"):
                return
            byte_count = self._record_bytes[expert]
            ledger._add_counter_locked("verified_record_jobs", 1, self.phase)
            ledger._add_counter_locked("verified_record_bytes", byte_count, self.phase)
            ledger._record_latency.observe(now_ns - started_ns)
            ledger._phase_record_latency[self.phase].observe(now_ns - started_ns)
            self._record_started_ns.pop(expert)

    def reader_completed(
        self,
        experts: tuple[int, ...],
        *,
        thread_cpu_ns: object,
    ) -> None:
        cpu_ns = _nonnegative_units(thread_cpu_ns)
        values = _expert_tuple(experts)
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            task = self._tasks.get(values)
            if task is None or task.status != "active":
                self._violation_locked("reader completion has no active task")
                return
            if self._select_locked(values, expected="verified") is None:
                return
            if not ledger._change_gauges_locked(
                {
                    "active_reader_tasks": -1,
                    "active_reader_task_bytes": -task.logical_bytes,
                },
                self.phase,
            ):
                return
            ledger._add_counter_locked("completed_reader_tasks", 1, self.phase)
            ledger._add_counter_locked("reader_thread_cpu_ns", cpu_ns, self.phase)
            if task.accepted:
                self._tasks.pop(values)
            else:
                task.status = "completed"

    def reader_failed(
        self,
        experts: tuple[int, ...],
        *,
        thread_cpu_ns: object = 0,
    ) -> None:
        cpu_ns = _nonnegative_units(thread_cpu_ns)
        values = _expert_tuple(experts)
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            task = self._tasks.get(values)
            if task is None or task.status != "active":
                self._violation_locked("reader failure has no active task")
                return
            if (
                self._select_locked(values, expected={"reader-active", "verified"})
                is None
            ):
                return
            if not self._move_records_locked(
                values,
                "abandoned",
                extra_deltas={
                    "active_reader_tasks": -1,
                    "active_reader_task_bytes": -task.logical_bytes,
                },
            ):
                return
            jobs = len(values)
            byte_count = task.logical_bytes
            for name, amount in (
                ("failed_reader_tasks", 1),
                ("failed_record_jobs", jobs),
                ("failed_record_bytes", byte_count),
                ("abandoned_record_jobs", jobs),
                ("abandoned_record_bytes", byte_count),
                ("reader_thread_cpu_ns", cpu_ns),
            ):
                ledger._add_counter_locked(name, amount, self.phase)
            for expert in values:
                self._record_started_ns.pop(expert, None)
            if task.accepted:
                self._tasks.pop(values)
            else:
                task.status = "failed"

    def record_runnable(self, expert: int) -> None:
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            values = self._select_locked((expert,), expected={"verified", "satisfied"})
            if values is None:
                return
            if self._states[expert] == "verified" and any(
                task.status == "active" and expert in task.experts
                for task in self._tasks.values()
            ):
                self._violation_locked(f"record {expert} reader task is still active")
                return
            if not self._move_records_locked(values, "runnable"):
                return
            byte_count = self._record_bytes[expert]
            ledger._add_counter_locked("runnable_record_jobs", 1, self.phase)
            ledger._add_counter_locked("runnable_record_bytes", byte_count, self.phase)

    def claim_misses(self, experts: tuple[int, ...]) -> None:
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            values = self._select_locked(experts, expected="runnable")
            if values is None or not self._move_records_locked(values, "claimed"):
                return
            byte_count = self._bytes(values)
            ledger._add_counter_locked("claimed_record_jobs", len(values), self.phase)
            ledger._add_counter_locked("claimed_record_bytes", byte_count, self.phase)

    def begin_generation_wait(self) -> None:
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            if ledger._generation_wait_route is not None:
                self._violation_locked("generation expert-input wait is already active")
                return
            if not ledger._change_gauge_locked("generation_wait_active", 1, self.phase):
                return
            ledger._generation_wait_route = self._route_id
            ledger._add_counter_locked(
                "generation_expert_input_wait_events", 1, self.phase
            )

    def end_generation_wait(self) -> None:
        ledger = self._ledger
        with ledger._lock:
            ledger._accrue_locked(ledger._now_locked())
            if not self._ensure_open_locked():
                return
            if ledger._generation_wait_route != self._route_id:
                self._violation_locked("generation expert-input wait is not active")
                return
            if not ledger._change_gauge_locked(
                "generation_wait_active", -1, self.phase
            ):
                return
            ledger._generation_wait_route = None

    def close(self) -> None:
        ledger = self._ledger
        with ledger._lock:
            if self._disabled or self._closed:
                self._closed = True
                return
            ledger._accrue_locked(ledger._now_locked())
            deltas: dict[str, int] = {"open_routes": -1}
            if ledger._generation_wait_route == self._route_id:
                deltas["generation_wait_active"] = -1
            for task in self._tasks.values():
                task_gauges = {
                    "provisional": (
                        "provisional_reader_tasks",
                        "provisional_reader_task_bytes",
                    ),
                    "queued": ("queued_reader_tasks", "queued_reader_task_bytes"),
                    "active": ("active_reader_tasks", "active_reader_task_bytes"),
                }.get(task.status)
                if task_gauges is not None:
                    deltas[task_gauges[0]] = deltas.get(task_gauges[0], 0) - 1
                    deltas[task_gauges[1]] = (
                        deltas.get(task_gauges[1], 0) - task.logical_bytes
                    )
            abandoned = [
                expert
                for expert, state in self._states.items()
                if state not in {"claimed", "abandoned"}
            ]
            for expert in abandoned:
                state = self._states[expert]
                gauges = self._STATE_GAUGES.get(state)
                if gauges is None:
                    self._violation_locked(
                        f"unknown record state during route close: {state}"
                    )
                    return
                deltas[gauges[0]] = deltas.get(gauges[0], 0) - 1
                deltas[gauges[1]] = (
                    deltas.get(gauges[1], 0) - self._record_bytes[expert]
                )
            if not ledger._change_gauges_locked(deltas, self.phase):
                return
            jobs = len(abandoned)
            byte_count = self._bytes(tuple(abandoned))
            ledger._add_counter_locked("abandoned_record_jobs", jobs, self.phase)
            ledger._add_counter_locked("abandoned_record_bytes", byte_count, self.phase)
            for expert in abandoned:
                self._states[expert] = "abandoned"
                self._record_started_ns.pop(expert, None)
            if ledger._generation_wait_route == self._route_id:
                ledger._generation_wait_route = None
            self._tasks.clear()
            ledger._routes.pop(self._route_id, None)
            self._closed = True
