"""Correlated resource evidence for streamed-generation diagnostics."""

from __future__ import annotations

import plistlib
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any


_POOL_INTEGRALS = (
    "queued_work_ns",
    "active_work_ns",
    "queued_unit_ns",
    "active_unit_ns",
)

_PIPELINE_SOURCE_SCHEMA = "mtplx-expert-pipeline-attribution-v1"
_PIPELINE_SUMMARY_SCHEMA = "mtplx-expert-pipeline-summary-v1"
_PIPELINE_PRIMARY_STATES = (
    "generation_thread_expert_input_wait",
    "logical_range_active",
    "reader_completion_active",
    "submitted_queued",
    "eligible_unsubmitted",
    "host_runnable_work",
    "route_publication_pending",
    "no_known_useful_work",
)
_PIPELINE_OVERLAPS = {
    "generation_wait_storage_active": "generation_wait_storage_active_ns",
    "generation_wait_reader_task_active": ("generation_wait_reader_task_active_ns"),
    "generation_wait_submitted_queued": ("generation_wait_submitted_queued_ns"),
    "generation_wait_eligible_unsubmitted": ("generation_wait_eligible_unsubmitted_ns"),
    "generation_wait_runnable": "generation_wait_runnable_ns",
}
_PIPELINE_BLOCK_REASONS = (
    "operation_credit",
    "byte_credit",
    "authoritative_reserve",
    "slot_unavailable",
    "pin_held",
    "slot_loading",
)
_PIPELINE_BACKEND_COUNTERS = {
    "logical_range_reader_invocations": "read_operations",
    "python_preadv_invocations": "python_preadv_invocations",
    "preadv_bytes_returned": "preadv_bytes_returned",
    "native_positional_calls": "native_positional_calls",
    "native_bytes_returned": "native_bytes_returned",
}
_PIPELINE_HISTOGRAMS = (
    "logical_range_latency_ns",
    "complete_record_latency_ns",
)
_PIPELINE_COVERAGE_LIMITATIONS = {
    "operation_credit": "unavailable",
    "byte_credit": "unavailable",
    "authoritative_reserve": "unavailable",
    "slot_capacity_admission": "unavailable",
    "outer_split_executor_queue": "unavailable",
    "eligible_unsubmitted_cause": "unattributed",
    "admitted_read_ranges": "unavailable",
    "scheduled_read_ranges": "unavailable",
    "physical_device_operations": "unavailable",
    "physical_device_bytes": "unavailable",
    "physical_device_queue_depth": "unavailable",
    "gpu_expert_wait": "unavailable",
    "gpu_idle_time": "unavailable",
    "future_layer_eligibility": "unavailable",
    "speculative_record_accounting": "unavailable",
    "python_preadv_when_native_reader": "unavailable",
}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _counter_delta(before: object, after: object) -> int:
    return max(0, _integer(after) - _integer(before))


def _monotonic_delta(before: object, after: object) -> tuple[int, bool]:
    left = _integer(before)
    right = _integer(after)
    if right < left:
        return 0, True
    return right - left, False


def _mapping_deltas(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[dict[str, int], bool]:
    result: dict[str, int] = {}
    reset = False
    for name in sorted(set(before) | set(after)):
        result[name], changed = _monotonic_delta(before.get(name), after.get(name))
        reset = reset or changed
    return result, reset


def _histogram_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    before_bounds = tuple(_integer(value) for value in before.get("bounds_ns", ()))
    after_bounds = tuple(_integer(value) for value in after.get("bounds_ns", ()))
    before_counts = tuple(_integer(value) for value in before.get("bucket_counts", ()))
    after_counts = tuple(_integer(value) for value in after.get("bucket_counts", ()))
    valid_shape = (
        bool(after_bounds)
        and before_bounds == after_bounds
        and len(before_counts) == len(after_counts) == len(after_bounds) + 1
    )
    if not valid_shape:
        return {
            "bounds_ns": after_bounds,
            "bucket_counts": tuple(0 for _value in after_counts),
            "sample_count": 0,
            "overflow_count": 0,
        }, True

    valid_totals = (
        _integer(before.get("sample_count")) == sum(before_counts)
        and _integer(after.get("sample_count")) == sum(after_counts)
        and _integer(before.get("overflow_count")) == before_counts[-1]
        and _integer(after.get("overflow_count")) == after_counts[-1]
    )
    bucket_counts: list[int] = []
    reset = not valid_totals
    for left, right in zip(before_counts, after_counts, strict=True):
        delta, changed = _monotonic_delta(left, right)
        bucket_counts.append(delta)
        reset = reset or changed
    sample_count, sample_reset = _monotonic_delta(
        before.get("sample_count"), after.get("sample_count")
    )
    overflow_count, overflow_reset = _monotonic_delta(
        before.get("overflow_count"), after.get("overflow_count")
    )
    reset = reset or sample_reset or overflow_reset
    if sample_count != sum(bucket_counts) or overflow_count != bucket_counts[-1]:
        reset = True
    return {
        "bounds_ns": after_bounds,
        "bucket_counts": tuple(bucket_counts),
        "sample_count": sample_count,
        "overflow_count": overflow_count,
    }, reset


def _pipeline_interval(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    span_ns: int,
) -> dict[str, Any]:
    """Difference one decode-phase pipeline snapshot on the sampler clock."""
    before_pipeline = _mapping(before.get("expert_pipeline"))
    after_pipeline = _mapping(after.get("expert_pipeline"))
    source_schema = str(after_pipeline.get("schema") or "unavailable")
    schema_mismatch = (
        before_pipeline.get("schema") != _PIPELINE_SOURCE_SCHEMA
        or after_pipeline.get("schema") != _PIPELINE_SOURCE_SCHEMA
    )
    before_decode = _mapping(_mapping(before_pipeline.get("by_phase")).get("decode"))
    after_decode = _mapping(_mapping(after_pipeline.get("by_phase")).get("decode"))
    required_decode_keys = (
        "observation_ns",
        "counters",
        "integrals_ns",
        "primary_integrals_ns",
        "block_counts",
        "block_ns",
        "block_coverage",
        "histograms",
        "coverage",
    )
    source_components_present = all(
        name in snapshot
        for snapshot in (before_decode, after_decode)
        for name in required_decode_keys
    )
    primary_components_present = all(
        name in _mapping(snapshot.get("primary_integrals_ns"))
        for snapshot in (before_decode, after_decode)
        for name in _PIPELINE_PRIMARY_STATES
    )

    observation_ns, observation_reset = _monotonic_delta(
        before_decode.get("observation_ns"), after_decode.get("observation_ns")
    )
    counters, counters_reset = _mapping_deltas(
        _mapping(before_decode.get("counters")),
        _mapping(after_decode.get("counters")),
    )
    integrals_ns, integrals_reset = _mapping_deltas(
        _mapping(before_decode.get("integrals_ns")),
        _mapping(after_decode.get("integrals_ns")),
    )
    primary_integrals_ns, primary_reset = _mapping_deltas(
        _mapping(before_decode.get("primary_integrals_ns")),
        _mapping(after_decode.get("primary_integrals_ns")),
    )
    block_counts, block_count_reset = _mapping_deltas(
        _mapping(before_decode.get("block_counts")),
        _mapping(after_decode.get("block_counts")),
    )
    block_ns, block_ns_reset = _mapping_deltas(
        _mapping(before_decode.get("block_ns")),
        _mapping(after_decode.get("block_ns")),
    )

    before_histograms = _mapping(before_decode.get("histograms"))
    after_histograms = _mapping(after_decode.get("histograms"))
    required_histograms_present = all(
        name in histograms
        for histograms in (before_histograms, after_histograms)
        for name in _PIPELINE_HISTOGRAMS
    )
    histograms: dict[str, dict[str, Any]] = {}
    histogram_reset = False
    for name in sorted(set(before_histograms) | set(after_histograms)):
        histogram, changed = _histogram_delta(
            _mapping(before_histograms.get(name)),
            _mapping(after_histograms.get(name)),
        )
        histograms[name] = histogram
        histogram_reset = histogram_reset or changed

    before_io = _mapping(before.get("io"))
    after_io = _mapping(after.get("io"))
    backend_available = all(
        source_name in snapshot
        for snapshot in (before_io, after_io)
        for source_name in _PIPELINE_BACKEND_COUNTERS.values()
    )
    backend: dict[str, int] = {}
    backend_reset = False
    for report_name, source_name in _PIPELINE_BACKEND_COUNTERS.items():
        backend[report_name], changed = _monotonic_delta(
            before_io.get(source_name), after_io.get(source_name)
        )
        backend_reset = backend_reset or changed

    source_incomplete = (
        _mapping(after_decode.get("coverage")).get("attribution") != "measured"
    )
    decode_reset = any(
        (
            observation_reset,
            counters_reset,
            integrals_reset,
            primary_reset,
            block_count_reset,
            block_ns_reset,
            histogram_reset,
        )
    )
    primary_identity_valid = sum(primary_integrals_ns.values()) == observation_ns
    decode_incomplete = (
        schema_mismatch
        or source_incomplete
        or not source_components_present
        or not primary_components_present
        or not required_histograms_present
        or not primary_identity_valid
        or decode_reset
    )
    if not backend_available:
        backend_coverage = "unavailable"
    elif backend_reset:
        backend_coverage = "incomplete_reset"
    else:
        backend_coverage = "measured_all_phases"
    reset_detected = decode_reset or backend_reset
    incomplete = decode_incomplete or backend_coverage != "measured_all_phases"
    return {
        "source_schema": source_schema,
        "scope": "decode",
        "sampler_span_ns": max(0, int(span_ns)),
        "decode_observation_ns": observation_ns,
        "counters": counters,
        "integrals_ns": integrals_ns,
        "primary_integrals_ns": primary_integrals_ns,
        "block_counts": block_counts,
        "block_ns": block_ns,
        "block_coverage": dict(_mapping(after_decode.get("block_coverage"))),
        "histograms": histograms,
        "sampler_window_backend": backend,
        "coverage": {
            "attribution": "incomplete" if incomplete else "measured",
            "decode_phase": "incomplete" if decode_incomplete else "measured",
            "sampler_window_backend": backend_coverage,
            "reset_detected": reset_detected,
        },
    }


def _pool_interval(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    span_ns: int,
) -> dict[str, float | int]:
    safe_span = max(1, span_ns)
    deltas = {
        name: _counter_delta(before.get(name), after.get(name))
        for name in _POOL_INTEGRALS
    }
    return {
        "worker_capacity": _integer(after.get("worker_capacity"), 1),
        "mean_queued_work": deltas["queued_work_ns"] / safe_span,
        "mean_active_work": deltas["active_work_ns"] / safe_span,
        "mean_queued_units": deltas["queued_unit_ns"] / safe_span,
        "mean_active_units": deltas["active_unit_ns"] / safe_span,
        "accepted_submissions": _counter_delta(
            before.get("accepted_submissions"), after.get("accepted_submissions")
        ),
        "started": _counter_delta(before.get("started"), after.get("started")),
        "completed": _counter_delta(before.get("completed"), after.get("completed")),
        "rejected_submissions": _counter_delta(
            before.get("rejected_submissions"),
            after.get("rejected_submissions"),
        ),
        "queued_work_end": _integer(after.get("queued_work")),
        "active_work_end": _integer(after.get("active_work")),
        "queued_units_end": _integer(after.get("queued_units")),
        "active_units_end": _integer(after.get("active_units")),
        "lifetime_queued_work_peak": _integer(after.get("queued_work_peak")),
        "lifetime_active_work_peak": _integer(after.get("active_work_peak")),
        "lifetime_queued_units_peak": _integer(after.get("queued_units_peak")),
        "lifetime_active_units_peak": _integer(after.get("active_units_peak")),
    }


@dataclass(frozen=True)
class ResourceTick:
    monotonic_ns: int
    completion_tokens: int
    snapshot: dict[str, Any]


def _interval(previous: ResourceTick, current: ResourceTick) -> dict[str, Any] | None:
    span_ns = current.monotonic_ns - previous.monotonic_ns
    if span_ns <= 0:
        return None
    span_s = span_ns / 1e9
    before = previous.snapshot
    after = current.snapshot
    before_io = _mapping(before.get("io"))
    after_io = _mapping(after.get("io"))
    before_cache = _mapping(before.get("cache"))
    after_cache = _mapping(after.get("cache"))
    before_metrics = _mapping(before.get("metrics"))
    after_metrics = _mapping(after.get("metrics"))
    reader = _pool_interval(
        _mapping(before.get("reader_pool")),
        _mapping(after.get("reader_pool")),
        span_ns,
    )
    fences = _pool_interval(
        _mapping(before.get("completion_fences")),
        _mapping(after.get("completion_fences")),
        span_ns,
    )
    read_bytes = _counter_delta(before_io.get("read_bytes"), after_io.get("read_bytes"))
    read_operations = _counter_delta(
        before_io.get("read_operations"), after_io.get("read_operations")
    )
    expert_requests = _counter_delta(
        before_cache.get("expert_requests"), after_cache.get("expert_requests")
    )
    expert_hits = _counter_delta(
        before_cache.get("expert_hits"), after_cache.get("expert_hits")
    )
    expert_misses = _counter_delta(
        before_cache.get("expert_misses"), after_cache.get("expert_misses")
    )
    completion_tokens = max(0, current.completion_tokens - previous.completion_tokens)
    io_active = reader["mean_active_work"] > 0.0 or read_bytes > 0
    fence_pending = fences["mean_active_work"] > 0.0
    result: dict[str, Any] = {
        "interval_seconds": span_s,
        "io_cache_mode": str(after.get("io_cache_mode") or "unknown"),
        "reader_read_bytes": read_bytes,
        "reader_read_operations": read_operations,
        "reader_gib_per_second": read_bytes / 1024**3 / span_s,
        "reader_iops": read_operations / span_s,
        "bytes_per_read_operation": (
            read_bytes / read_operations if read_operations else 0.0
        ),
        "expert_requests": expert_requests,
        "expert_requests_per_second": expert_requests / span_s,
        "expert_hits": expert_hits,
        "expert_misses": expert_misses,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / span_s,
        "reader_worker_capacity": reader["worker_capacity"],
        "mean_queued_reads": reader["mean_queued_work"],
        "mean_active_readers": reader["mean_active_work"],
        "mean_queued_bytes": reader["mean_queued_units"],
        "mean_active_bytes": reader["mean_active_units"],
        "reader_queue_nonempty": reader["mean_queued_work"] > 0.0,
        "reader_pool": reader,
        "completion_fences": fences,
        "io_active": io_active,
        "completion_fence_pending": fence_pending,
        "io_and_completion_fence_seen_in_interval": io_active and fence_pending,
        "completion_fence_registrations": _counter_delta(
            before_metrics.get("completion_fences"),
            after_metrics.get("completion_fences"),
        ),
        "completion_fence_slots": _counter_delta(
            before_metrics.get("completion_fence_slots"),
            after_metrics.get("completion_fence_slots"),
        ),
        "completion_fence_fallbacks": _counter_delta(
            before_metrics.get("completion_fence_fallbacks"),
            after_metrics.get("completion_fence_fallbacks"),
        ),
        "completion_fence_failures": _counter_delta(
            before_metrics.get("completion_fence_failures"),
            after_metrics.get("completion_fence_failures"),
        ),
        "synchronous_fences": _counter_delta(
            before_metrics.get("synchronous_fences"),
            after_metrics.get("synchronous_fences"),
        ),
        "synchronous_fence_slots": _counter_delta(
            before_metrics.get("synchronous_fence_slots"),
            after_metrics.get("synchronous_fence_slots"),
        ),
    }
    if (
        before.get("expert_pipeline") is not None
        or after.get("expert_pipeline") is not None
    ):
        result["expert_pipeline"] = _pipeline_interval(before, after, span_ns)
    if _integer(after.get("quant_bits")) == 4:
        result["q4_assignments_per_second"] = expert_requests / span_s
    return result


def _layer_deltas(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    keys = set(before) | set(after)
    for layer in sorted(keys, key=lambda item: int(item)):
        left = _mapping(before.get(layer))
        right = _mapping(after.get(layer))
        row: dict[str, int | float] = {}
        for name in (
            "route_calls",
            "expert_requests",
            "expert_hits",
            "expert_misses",
            "persistent_loads",
            "transient_loads",
            "evictions",
            "bytes_read",
        ):
            row[name] = _counter_delta(left.get(name), right.get(name))
        total = int(row["expert_hits"]) + int(row["expert_misses"])
        row["hit_rate"] = int(row["expert_hits"]) / total if total else 0.0
        if any(value for name, value in row.items() if name != "hit_rate"):
            result[str(layer)] = row
    return result


class ResourceTelemetrySampler:
    """Capture bounded same-clock samples from the runtime's cheap snapshot."""

    def __init__(
        self,
        snapshot: Callable[[], dict[str, Any] | None],
        *,
        token_count: Callable[[], int],
        interval_s: float = 0.25,
        max_samples: int = 4096,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("resource telemetry interval must be positive")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise ValueError("resource telemetry max samples must be an integer")
        if max_samples < 2:
            raise ValueError("resource telemetry max samples must be at least 2")
        self._snapshot = snapshot
        self._token_count = token_count
        self._interval_s = float(interval_s)
        self._clock_ns = clock_ns
        self._first_tick: ResourceTick | None = None
        self._recent_ticks: deque[ResourceTick] = deque(maxlen=max_samples - 1)
        self._capture_count = 0
        self._sampling_failure: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def capture(self) -> ResourceTick | None:
        snapshot = self._snapshot()
        if snapshot is None:
            return None
        tick = ResourceTick(
            monotonic_ns=int(self._clock_ns()),
            completion_tokens=int(self._token_count()),
            snapshot=dict(snapshot),
        )
        with self._lock:
            if self._first_tick is None:
                self._first_tick = tick
            else:
                self._recent_ticks.append(tick)
            self._capture_count += 1
        return tick

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.capture()
            except BaseException as exc:
                self._record_sampling_failure(exc)
                self._stop.set()
                return

    def _record_sampling_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._sampling_failure is None:
                self._sampling_failure = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> ResourceTelemetrySampler:
        self.capture()
        self._thread = threading.Thread(
            target=self._loop,
            name="mtplx-resource-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_s * 4))
        try:
            self.capture()
        except BaseException as exc:
            self._record_sampling_failure(exc)

    @property
    def ticks(self) -> tuple[ResourceTick, ...]:
        with self._lock:
            if self._first_tick is None:
                return ()
            return (self._first_tick, *self._recent_ticks)

    def report(
        self,
        *,
        ssd_ceiling_gib_s: float | None = None,
        powermetrics: dict[str, Any] | None = None,
        generation_thread_cpu_ns: int | None = None,
        generation_elapsed_ns: int | None = None,
        final_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        ticks = self.ticks
        intervals = [
            item
            for item in (
                _interval(left, right)
                for left, right in zip(ticks, ticks[1:], strict=False)
            )
            if item is not None
        ]
        thread_fraction = None
        if (
            generation_thread_cpu_ns is not None
            and generation_elapsed_ns is not None
            and generation_elapsed_ns > 0
        ):
            thread_fraction = max(0.0, generation_thread_cpu_ns / generation_elapsed_ns)
        summary = summarize_intervals(
            intervals,
            ssd_ceiling_gib_s=ssd_ceiling_gib_s,
            powermetrics=powermetrics,
            generation_thread_core_fraction=thread_fraction,
        )
        samples_dropped = max(0, self._capture_count - len(ticks))
        with self._lock:
            sampling_failure = self._sampling_failure
        if samples_dropped:
            summary["coverage"]["timeline"] = "retained_start_and_recent_tail"
            candidates = list(summary["attribution"]["candidates"])
            if "increase_resource_max_samples" not in candidates:
                candidates.append("increase_resource_max_samples")
            summary["attribution"] = {
                "status": "incomplete",
                "candidates": candidates,
            }
        else:
            summary["coverage"]["timeline"] = "complete"
        if sampling_failure is not None:
            summary["coverage"]["timeline"] = "incomplete_sampler_failure"
            candidates = list(summary["attribution"]["candidates"])
            if "resource_sampler_failure" not in candidates:
                candidates.append("resource_sampler_failure")
            summary["attribution"] = {
                "status": "incomplete",
                "candidates": candidates,
            }
        payload: dict[str, Any] = {
            "schema": "mtplx-resource-telemetry-v2",
            "sample_interval_seconds": self._interval_s,
            "sample_count": len(ticks),
            "samples_dropped": samples_dropped,
            "sampling_failures": int(sampling_failure is not None),
            "sampling_failure": sampling_failure,
            **summary,
            "timeline": intervals,
        }
        if ticks:
            payload["cache_by_layer"] = _layer_deltas(
                _mapping(ticks[0].snapshot.get("cache_by_layer")),
                _mapping(ticks[-1].snapshot.get("cache_by_layer")),
            )
        if final_completion_tokens is not None:
            payload["throughput"]["final_completion_tokens"] = int(
                final_completion_tokens
            )
        return payload


def _weighted_mean(
    intervals: list[dict[str, Any]],
    name: str,
    elapsed: float,
) -> float:
    if elapsed <= 0:
        return 0.0
    return (
        sum(
            _number(item.get(name)) * _number(item.get("interval_seconds"))
            for item in intervals
        )
        / elapsed
    )


def _weighted_nested_mean(
    intervals: list[dict[str, Any]],
    group: str,
    name: str,
    elapsed: float,
) -> float:
    if elapsed <= 0:
        return 0.0
    return (
        sum(
            _number(_mapping(item.get(group)).get(name))
            * _number(item.get("interval_seconds"))
            for item in intervals
        )
        / elapsed
    )


def _nested_max(
    intervals: list[dict[str, Any]],
    group: str,
    name: str,
) -> int:
    return max(
        (_integer(_mapping(item.get(group)).get(name)) for item in intervals),
        default=0,
    )


def _sum_nested_mappings(
    rows: list[Mapping[str, Any]],
    name: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        for key, value in _mapping(row.get(name)).items():
            result[str(key)] = result.get(str(key), 0) + _integer(value)
    return result


def _merge_pipeline_histogram(
    rows: list[Mapping[str, Any]],
    name: str,
) -> tuple[dict[str, Any], bool]:
    bounds: tuple[int, ...] | None = None
    bucket_counts: list[int] | None = None
    incomplete = False
    for row in rows:
        histogram = _mapping(_mapping(row.get("histograms")).get(name))
        current_bounds = tuple(
            _integer(value) for value in histogram.get("bounds_ns", ())
        )
        current_counts = tuple(
            _integer(value) for value in histogram.get("bucket_counts", ())
        )
        if not current_bounds or len(current_counts) != len(current_bounds) + 1:
            incomplete = True
            continue
        if bounds is None:
            bounds = current_bounds
            bucket_counts = [0] * len(current_counts)
        if current_bounds != bounds or bucket_counts is None:
            incomplete = True
            continue
        for index, value in enumerate(current_counts):
            bucket_counts[index] += value

    if bounds is None or bucket_counts is None:
        return {
            "bounds_ns": (),
            "bucket_counts": (),
            "sample_count": 0,
            "overflow_count": 0,
        }, True
    sample_count = sum(bucket_counts)
    return {
        "bounds_ns": bounds,
        "bucket_counts": tuple(bucket_counts),
        "sample_count": sample_count,
        "overflow_count": bucket_counts[-1],
    }, incomplete


def _histogram_percentile(
    histogram: Mapping[str, Any],
    percentile: int,
) -> tuple[int | None, str]:
    counts = tuple(_integer(value) for value in histogram.get("bucket_counts", ()))
    bounds = tuple(_integer(value) for value in histogram.get("bounds_ns", ()))
    sample_count = _integer(histogram.get("sample_count"))
    if sample_count <= 0 or len(counts) != len(bounds) + 1:
        return None, "unavailable"
    rank = (sample_count * percentile + 99) // 100
    cumulative = 0
    for bucket, count in enumerate(counts):
        cumulative += count
        if cumulative >= rank:
            if bucket == len(bounds):
                return None, "censored_overflow"
            return bounds[bucket], "bounded"
    return None, "unavailable"


def _histogram_report(histogram: Mapping[str, Any]) -> dict[str, Any]:
    p50, p50_status = _histogram_percentile(histogram, 50)
    p95, p95_status = _histogram_percentile(histogram, 95)
    return {
        "bounds_ns": tuple(histogram.get("bounds_ns", ())),
        "bucket_counts": tuple(histogram.get("bucket_counts", ())),
        "sample_count": _integer(histogram.get("sample_count")),
        "overflow_count": _integer(histogram.get("overflow_count")),
        "p50_upper_bound_ns": p50,
        "p50_status": p50_status,
        "p95_upper_bound_ns": p95,
        "p95_status": p95_status,
    }


def _unavailable_pipeline_summary() -> dict[str, Any]:
    return {
        "schema": _PIPELINE_SUMMARY_SCHEMA,
        "source_schema": "unavailable",
        "scope": "decode",
        "decode_observation_ns": 0,
        "generation_expert_input_wait_fraction": None,
        "coverage": {
            "attribution": "unavailable",
            "decode_phase": "unavailable",
            "sampler_window_backend": "unavailable",
            **_PIPELINE_COVERAGE_LIMITATIONS,
        },
        "physical_device_queue_depth": {"status": "unavailable"},
        "gpu_expert_wait": {"status": "unavailable"},
    }


def _summarize_pipeline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline_rows = [
        _mapping(row.get("expert_pipeline"))
        for row in rows
        if isinstance(row.get("expert_pipeline"), Mapping)
    ]
    if not pipeline_rows:
        return _unavailable_pipeline_summary()

    counters = _sum_nested_mappings(pipeline_rows, "counters")
    integrals_ns = _sum_nested_mappings(pipeline_rows, "integrals_ns")
    primary_state_ns = _sum_nested_mappings(pipeline_rows, "primary_integrals_ns")
    block_counts = _sum_nested_mappings(pipeline_rows, "block_counts")
    block_ns = _sum_nested_mappings(pipeline_rows, "block_ns")
    backend = _sum_nested_mappings(pipeline_rows, "sampler_window_backend")
    decode_observation_ns = sum(
        _integer(row.get("decode_observation_ns")) for row in pipeline_rows
    )
    source_schemas = {str(row.get("source_schema")) for row in pipeline_rows}
    interval_incomplete = any(
        _mapping(row.get("coverage")).get("attribution") != "measured"
        for row in pipeline_rows
    )
    decode_statuses = {
        str(_mapping(row.get("coverage")).get("decode_phase") or "incomplete")
        for row in pipeline_rows
    }
    decode_incomplete = decode_statuses != {"measured"} or source_schemas != {
        _PIPELINE_SOURCE_SCHEMA
    }
    backend_statuses = {
        str(
            _mapping(row.get("coverage")).get("sampler_window_backend") or "unavailable"
        )
        for row in pipeline_rows
    }
    if backend_statuses == {"measured_all_phases"}:
        backend_coverage = "measured_all_phases"
    elif "incomplete_reset" in backend_statuses:
        backend_coverage = "incomplete_reset"
    elif backend_statuses == {"unavailable"}:
        backend_coverage = "unavailable"
    else:
        backend_coverage = "incomplete"

    histogram_names = set()
    for row in pipeline_rows:
        histogram_names.update(_mapping(row.get("histograms")))
    histogram_reports: dict[str, Any] = {}
    for name in sorted(histogram_names):
        histogram, histogram_incomplete = _merge_pipeline_histogram(
            pipeline_rows, str(name)
        )
        decode_incomplete = decode_incomplete or histogram_incomplete
        histogram_reports[str(name)] = _histogram_report(histogram)

    primary_state_ns = {
        name: primary_state_ns.get(name, 0) for name in _PIPELINE_PRIMARY_STATES
    }
    decode_incomplete = (
        decode_incomplete or sum(primary_state_ns.values()) != decode_observation_ns
    )
    incomplete = (
        interval_incomplete
        or decode_incomplete
        or backend_coverage != "measured_all_phases"
    )
    primary_state_fraction = {
        name: (value / decode_observation_ns if decode_observation_ns > 0 else None)
        for name, value in primary_state_ns.items()
    }
    overlap_ns = {
        report_name: integrals_ns.get(source_name, 0)
        for report_name, source_name in _PIPELINE_OVERLAPS.items()
    }
    overlap_fraction = {
        name: (value / decode_observation_ns if decode_observation_ns > 0 else None)
        for name, value in overlap_ns.items()
    }

    block_coverage: dict[str, str] = {}
    for reason in _PIPELINE_BLOCK_REASONS:
        statuses = {
            str(_mapping(row.get("block_coverage")).get(reason) or "unavailable")
            for row in pipeline_rows
        }
        block_coverage[reason] = (
            "measured" if statuses == {"measured"} else "unavailable"
        )
    block_reasons = {
        reason: (
            {
                "status": "measured",
                "count": block_counts.get(reason, 0),
                "duration_ns": block_ns.get(reason, 0),
            }
            if block_coverage[reason] == "measured"
            else {"status": "unavailable"}
        )
        for reason in _PIPELINE_BLOCK_REASONS
    }

    return {
        "schema": _PIPELINE_SUMMARY_SCHEMA,
        "source_schema": (
            _PIPELINE_SOURCE_SCHEMA
            if source_schemas == {_PIPELINE_SOURCE_SCHEMA}
            else "mixed_or_unavailable"
        ),
        "scope": "decode",
        "logical_record_jobs": counters.get("logical_record_jobs", 0),
        "logical_record_bytes": counters.get("logical_record_bytes", 0),
        "accepted_executor_submissions": counters.get("accepted_submissions", 0),
        "submission_accepted_record_jobs": counters.get("accepted_record_jobs", 0),
        "submission_accepted_record_bytes": counters.get("accepted_record_bytes", 0),
        "reader_tasks_started": counters.get("started_reader_tasks", 0),
        "reader_tasks_completed": counters.get("completed_reader_tasks", 0),
        "reader_tasks_failed": counters.get("failed_reader_tasks", 0),
        "decode_logical_ranges_started": counters.get("started_logical_ranges", 0),
        "decode_observation_ns": decode_observation_ns,
        "decode_counters": counters,
        "decode_integrals_ns": integrals_ns,
        "primary_state_ns": primary_state_ns,
        "generation_expert_input_wait_fraction": (
            integrals_ns.get("generation_expert_input_wait_ns", 0)
            / decode_observation_ns
            if decode_observation_ns > 0
            else None
        ),
        "orthogonal_overlap": {
            "denominator": "decode_observation_ns",
            "duration_ns": overlap_ns,
            "fraction_of_decode_observation": overlap_fraction,
        },
        "primary_state_fraction": primary_state_fraction,
        "block_reasons": block_reasons,
        "latency_histograms": histogram_reports,
        "sampler_window_backend": {
            "scope": "sampler_window_all_phases",
            **{
                name: (
                    None if backend_coverage == "unavailable" else backend.get(name, 0)
                )
                for name in _PIPELINE_BACKEND_COUNTERS
            },
        },
        "physical_device_queue_depth": {"status": "unavailable"},
        "gpu_expert_wait": {"status": "unavailable"},
        "coverage": {
            "attribution": "incomplete" if incomplete else "measured",
            "decode_phase": "incomplete" if decode_incomplete else "measured",
            "sampler_window_backend": backend_coverage,
            **_PIPELINE_COVERAGE_LIMITATIONS,
        },
    }


def summarize_intervals(
    intervals: Iterable[dict[str, Any]],
    *,
    ssd_ceiling_gib_s: float | None,
    powermetrics: dict[str, Any] | None,
    generation_thread_core_fraction: float | None = None,
) -> dict[str, Any]:
    rows = list(intervals)
    elapsed = sum(_number(item.get("interval_seconds")) for item in rows)
    count = len(rows)
    read_bytes = sum(_integer(item.get("reader_read_bytes")) for item in rows)
    read_operations = sum(_integer(item.get("reader_read_operations")) for item in rows)
    io_cache_modes = sorted(
        {str(item.get("io_cache_mode") or "unknown") for item in rows}
    )
    uncached_reader = bool(rows) and io_cache_modes == ["f-nocache"]
    requests = sum(_integer(item.get("expert_requests")) for item in rows)
    misses = sum(_integer(item.get("expert_misses")) for item in rows)
    completion_tokens = sum(_integer(item.get("completion_tokens")) for item in rows)
    completion_fence_registrations = sum(
        _integer(item.get("completion_fence_registrations")) for item in rows
    )
    completion_fence_slots = sum(
        _integer(item.get("completion_fence_slots")) for item in rows
    )
    completion_fence_fallbacks = sum(
        _integer(item.get("completion_fence_fallbacks")) for item in rows
    )
    completion_fence_failures = sum(
        _integer(item.get("completion_fence_failures")) for item in rows
    )
    synchronous_fences = sum(_integer(item.get("synchronous_fences")) for item in rows)
    synchronous_fence_slots = sum(
        _integer(item.get("synchronous_fence_slots")) for item in rows
    )
    ssd_gib_s = read_bytes / 1024**3 / elapsed if elapsed > 0 else 0.0
    queue_fraction = (
        sum(bool(item.get("reader_queue_nonempty")) for item in rows) / count
        if count
        else 0.0
    )
    io_fraction = (
        sum(bool(item.get("io_active")) for item in rows) / count if count else 0.0
    )
    fence_fraction = (
        sum(bool(item.get("completion_fence_pending")) for item in rows) / count
        if count
        else 0.0
    )
    same_interval_activity_fraction = (
        sum(
            bool(item.get("io_active")) and bool(item.get("completion_fence_pending"))
            for item in rows
        )
        / count
        if count
        else 0.0
    )
    neither_activity_interval_fraction = (
        sum(
            not bool(item.get("io_active"))
            and not bool(item.get("completion_fence_pending"))
            for item in rows
        )
        / count
        if count
        else 0.0
    )
    active_readers = _weighted_mean(rows, "mean_active_readers", elapsed)
    queued_reads = _weighted_mean(rows, "mean_queued_reads", elapsed)
    queued_bytes = _weighted_mean(rows, "mean_queued_bytes", elapsed)
    active_bytes = _weighted_mean(rows, "mean_active_bytes", elapsed)
    worker_capacity = max(
        (_integer(item.get("reader_worker_capacity"), 1) for item in rows),
        default=1,
    )
    active_capacity_fraction = active_readers / max(1, worker_capacity)
    backpressure = active_capacity_fraction >= 0.75 and queue_fraction >= 0.50
    queued_fences = _weighted_nested_mean(
        rows, "completion_fences", "mean_queued_work", elapsed
    )
    active_fences = _weighted_nested_mean(
        rows, "completion_fences", "mean_active_work", elapsed
    )
    queued_fence_slots = _weighted_nested_mean(
        rows, "completion_fences", "mean_queued_units", elapsed
    )
    active_fence_slots = _weighted_nested_mean(
        rows, "completion_fences", "mean_active_units", elapsed
    )
    pipeline_summary = _summarize_pipeline(rows)

    if not uncached_reader:
        ssd_utilization = None
        ssd_status = "unavailable"
    elif ssd_ceiling_gib_s is None or ssd_ceiling_gib_s <= 0:
        ssd_utilization = None
        ssd_status = "unavailable"
    else:
        ssd_utilization = ssd_gib_s / ssd_ceiling_gib_s
        ssd_status = (
            "supported" if ssd_utilization >= 0.75 and backpressure else "not_supported"
        )

    power = powermetrics or {"available": False, "reason": "not collected"}
    gpu_busy = power.get("process_gpu_busy_fraction")
    if power.get("available") and isinstance(gpu_busy, (int, float)):
        gpu_coverage = "measured_process_time"
        gpu_status = "supported" if float(gpu_busy) >= 0.75 else "not_supported"
    elif power.get("available") and isinstance(
        power.get("system_gpu_active_fraction"), (int, float)
    ):
        gpu_coverage = "system_only"
        gpu_status = "system_only"
    else:
        gpu_coverage = "unavailable"
        gpu_status = "unavailable"

    candidates: list[str] = []
    if backpressure and ssd_status != "supported":
        candidates.append("reader_pool_or_read_shape")
    if (
        misses > 0
        and queue_fraction < 0.10
        and ssd_utilization is not None
        and ssd_utilization < 0.40
    ):
        candidates.append("submission_or_dependency_starvation")
    if (
        generation_thread_core_fraction is not None
        and generation_thread_core_fraction >= 0.85
        and ssd_utilization is not None
        and ssd_utilization < 0.40
        and gpu_status in {"not_supported", "unavailable"}
    ):
        candidates.append("host_orchestration")
    if (
        io_fraction >= 0.25
        and fence_fraction >= 0.25
        and same_interval_activity_fraction < 0.10
    ):
        candidates.append("coarse_io_fence_separation")
    if synchronous_fences > 0 and fence_fraction < 0.10:
        candidates.append("synchronous_fence_or_evaluation")

    if ssd_status == "supported":
        candidates.append("storage_throughput")
    if gpu_status == "supported" and queue_fraction < 0.10:
        candidates.append("gpu_compute")
    attribution = {
        "status": "incomplete",
        "candidates": list(dict.fromkeys(candidates)) or ["unidentified"],
    }

    coverage: dict[str, Any] = {
        "runtime_occupancy": "measured",
        "storage_reads": (
            "uncached_reader_bytes" if uncached_reader else "logical_reader_bytes"
        ),
        "ssd_ceiling": "supplied" if ssd_utilization is not None else "unavailable",
        "gpu": gpu_coverage,
        "dram_bandwidth": "unavailable",
        "generation_thread_cpu": (
            "measured" if generation_thread_core_fraction is not None else "unavailable"
        ),
    }
    if gpu_coverage == "unavailable":
        coverage["gpu_reason"] = str(power.get("reason") or "not measured")

    return {
        "interval_count": count,
        "elapsed_seconds": elapsed,
        "throughput": {
            "completion_tokens": completion_tokens,
            "completion_tokens_per_second": (
                completion_tokens / elapsed if elapsed > 0 else 0.0
            ),
            "expert_requests": requests,
            "expert_requests_per_second": requests / elapsed if elapsed > 0 else 0.0,
            "q4_assignments_per_second": requests / elapsed if elapsed > 0 else 0.0,
        },
        "storage": {
            "io_cache_modes": io_cache_modes,
            "reader_read_bytes": read_bytes,
            "reader_read_operations": read_operations,
            "mean_gib_per_second": ssd_gib_s,
            "iops": read_operations / elapsed if elapsed > 0 else 0.0,
            "bytes_per_read_operation": (
                read_bytes / read_operations if read_operations else 0.0
            ),
            "ceiling_gib_per_second": ssd_ceiling_gib_s,
            "utilization_of_ceiling": ssd_utilization,
        },
        "reader_pool": {
            "worker_capacity": worker_capacity,
            "mean_active_readers": active_readers,
            "mean_queued_reads": queued_reads,
            "mean_active_bytes": active_bytes,
            "mean_queued_bytes": queued_bytes,
            "active_capacity_fraction": active_capacity_fraction,
            "queue_nonempty_fraction": queue_fraction,
            "lifetime_queue_depth_peak": _nested_max(
                rows, "reader_pool", "lifetime_queued_work_peak"
            ),
            "lifetime_active_readers_peak": _nested_max(
                rows, "reader_pool", "lifetime_active_work_peak"
            ),
            "lifetime_queued_bytes_peak": _nested_max(
                rows, "reader_pool", "lifetime_queued_units_peak"
            ),
            "lifetime_active_bytes_peak": _nested_max(
                rows, "reader_pool", "lifetime_active_units_peak"
            ),
        },
        "completion_fences": {
            "registrations": completion_fence_registrations,
            "registered_slots": completion_fence_slots,
            "fallbacks": completion_fence_fallbacks,
            "failures": completion_fence_failures,
            "synchronous_fences": synchronous_fences,
            "synchronous_fence_slots": synchronous_fence_slots,
            "synchronous_fences_per_second": (
                synchronous_fences / elapsed if elapsed > 0 else 0.0
            ),
            "synchronous_fences_per_token": (
                synchronous_fences / completion_tokens if completion_tokens else None
            ),
            "mean_queued_fences": queued_fences,
            "mean_active_fences": active_fences,
            "mean_queued_slots": queued_fence_slots,
            "mean_active_slots": active_fence_slots,
            "lifetime_queued_fences_peak": _nested_max(
                rows, "completion_fences", "lifetime_queued_work_peak"
            ),
            "lifetime_active_fences_peak": _nested_max(
                rows, "completion_fences", "lifetime_active_work_peak"
            ),
            "lifetime_queued_slots_peak": _nested_max(
                rows, "completion_fences", "lifetime_queued_units_peak"
            ),
            "lifetime_active_slots_peak": _nested_max(
                rows, "completion_fences", "lifetime_active_units_peak"
            ),
        },
        "overlap": {
            "measurement": "same_interval_coactivity",
            "simultaneous_overlap_measured": False,
            "io_activity_interval_fraction": io_fraction,
            "fence_activity_interval_fraction": fence_fraction,
            "same_interval_activity_fraction": same_interval_activity_fraction,
            "neither_activity_interval_fraction": (neither_activity_interval_fraction),
        },
        "host": {
            "generation_thread_core_fraction": generation_thread_core_fraction,
        },
        "expert_pipeline": pipeline_summary,
        "powermetrics": power,
        "coverage": coverage,
        "evidence": {
            "ssd_saturation": {
                "status": ssd_status,
                "utilization_of_ceiling": ssd_utilization,
                "requires_reader_pressure": True,
            },
            "reader_backpressure": {
                "status": "present" if backpressure else "absent",
                "active_capacity_fraction": active_capacity_fraction,
                "queue_nonempty_fraction": queue_fraction,
            },
            "gpu_activity": {
                "status": gpu_status,
                "process_gpu_busy_fraction": gpu_busy,
            },
            "dram_bandwidth": {
                "status": "unavailable",
                "reason": "no direct DRAM traffic counter is collected",
            },
        },
        "attribution": attribution,
    }


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _walk_mappings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _value_for(mapping: Mapping[str, Any], *aliases: str) -> object | None:
    wanted = {_normalized_key(alias) for alias in aliases}
    for key, value in mapping.items():
        if _normalized_key(key) in wanted:
            return value
    return None


def _timestamp_ns(document: Mapping[str, Any]) -> int | None:
    value = _value_for(document, "timestamp_ns", "sample_timestamp_ns")
    if isinstance(value, datetime):
        return int(value.timestamp() * 1e9)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    value = _value_for(document, "timestamp", "sample_timestamp")
    if isinstance(value, datetime):
        return int(value.timestamp() * 1e9)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return int(number if number > 1e14 else number * 1e9)
    return None


def _process_record(document: Mapping[str, Any], pid: int) -> Mapping[str, Any] | None:
    for mapping in _walk_mappings(document):
        value = _value_for(mapping, "pid", "process_id", "processid")
        if _integer(value, -1) == pid:
            return mapping
    return None


def parse_powermetrics_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    pid: int,
) -> list[dict[str, int | float]]:
    samples: list[dict[str, int | float]] = []
    for document in documents:
        process = _process_record(document, pid)
        if process is None:
            continue
        sample: dict[str, int | float] = {}
        timestamp = _timestamp_ns(document)
        if timestamp is not None:
            sample["timestamp_ns"] = timestamp
        gpu_time = _value_for(
            process,
            "gpu_time_ns",
            "gpu_runtime_ns",
            "gpu_ns",
            "gputime_ns",
        )
        if isinstance(gpu_time, (int, float)) and not isinstance(gpu_time, bool):
            sample["process_gpu_time_ns"] = int(gpu_time)
        gpu_rate = _value_for(
            process,
            "gpu_time_ms_per_s",
            "gputime_ms_per_s",
        )
        if isinstance(gpu_rate, (int, float)) and not isinstance(gpu_rate, bool):
            sample["process_gpu_ms_per_s"] = float(gpu_rate)
        cpu = _value_for(
            process,
            "cpu_ms_per_s",
            "cpu_time_ms_per_s",
            "sample_normalized_cpu_ms_per_s",
            "cputime_sample_ms_per_s",
        )
        if isinstance(cpu, (int, float)) and not isinstance(cpu, bool):
            sample["process_cpu_ms_per_s"] = float(cpu)
        disk = _value_for(
            process,
            "disk_read_bytes",
            "read_bytes",
            "io_read_bytes",
            "diskio_bytesread",
        )
        if isinstance(disk, (int, float)) and not isinstance(disk, bool):
            sample["process_disk_read_bytes"] = int(disk)
        wait_time = _value_for(
            process,
            "sfi_wait_time_ns",
            "wait_time_ns",
            "process_wait_time_ns",
            "sfi_ns",
        )
        if isinstance(wait_time, (int, float)) and not isinstance(wait_time, bool):
            sample["process_wait_time_ns"] = int(wait_time)
        samples.append(sample)
    return samples


def _powermetrics_documents(blob: bytes) -> list[Mapping[str, Any]]:
    documents: list[Mapping[str, Any]] = []
    for chunk in blob.split(b"\0"):
        if not chunk.strip():
            continue
        try:
            document = plistlib.loads(chunk)
        except Exception:
            continue
        if isinstance(document, Mapping):
            documents.append(document)
    return documents


def _summarize_powermetrics(
    samples: list[dict[str, int | float]],
) -> dict[str, Any]:
    if not samples:
        return {"available": False, "reason": "no benchmark process samples"}
    report: dict[str, Any] = {
        "available": True,
        "scope": "benchmark_process",
        "sample_count": len(samples),
    }
    cpu = [
        float(item["process_cpu_ms_per_s"])
        for item in samples
        if "process_cpu_ms_per_s" in item
    ]
    if cpu:
        report["process_cpu_ms_per_s_mean"] = sum(cpu) / len(cpu)
    gpu = [
        item
        for item in samples
        if "process_gpu_time_ns" in item and "timestamp_ns" in item
    ]
    gpu_rates = [
        float(item["process_gpu_ms_per_s"])
        for item in samples
        if "process_gpu_ms_per_s" in item
    ]
    if gpu_rates:
        report["process_gpu_ms_per_s_mean"] = sum(gpu_rates) / len(gpu_rates)
        report["process_gpu_busy_fraction"] = min(
            1.0,
            report["process_gpu_ms_per_s_mean"] / 1000.0,
        )
    if len(gpu) >= 2:
        elapsed = int(gpu[-1]["timestamp_ns"]) - int(gpu[0]["timestamp_ns"])
        gpu_delta = int(gpu[-1]["process_gpu_time_ns"]) - int(
            gpu[0]["process_gpu_time_ns"]
        )
        if elapsed > 0 and gpu_delta >= 0:
            report.setdefault(
                "process_gpu_busy_fraction",
                min(1.0, gpu_delta / elapsed),
            )
            report["process_gpu_time_ns"] = gpu_delta
    elif gpu:
        report["process_gpu_time_ns"] = int(gpu[0]["process_gpu_time_ns"])
    waits = [
        int(item["process_wait_time_ns"])
        for item in samples
        if "process_wait_time_ns" in item
    ]
    if waits:
        report["process_wait_time_ns"] = (
            max(waits) - min(waits) if len(waits) > 1 else waits[0]
        )
    reads = [
        int(item["process_disk_read_bytes"])
        for item in samples
        if "process_disk_read_bytes" in item
    ]
    if reads:
        report["process_disk_read_bytes"] = (
            max(reads) - min(reads) if len(reads) > 1 else reads[0]
        )
    return report


class PowermetricsCollector:
    """Optional non-interactive process/GPU collector for macOS."""

    def __init__(self, *, enabled: bool, pid: int, interval_ms: int = 250) -> None:
        self.enabled = bool(enabled)
        self.pid = int(pid)
        self.interval_ms = max(100, int(interval_ms))
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: Any | None = None
        self._stderr: Any | None = None
        self._report: dict[str, Any] = {
            "available": False,
            "reason": "disabled",
        }

    def start(self) -> None:
        if not self.enabled:
            return
        if platform.system() != "Darwin":
            self._report = {"available": False, "reason": "macOS required"}
            return
        sudo = shutil.which("sudo")
        powermetrics = shutil.which("powermetrics")
        if sudo is None or powermetrics is None:
            self._report = {
                "available": False,
                "reason": "sudo or powermetrics not found",
            }
            return
        try:
            authorization = subprocess.run(
                [sudo, "-n", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=2.0,
            )
        except Exception as exc:
            self._report = {"available": False, "reason": repr(exc)}
            return
        if authorization.returncode != 0:
            detail = authorization.stderr.decode(errors="replace").strip()
            self._report = {
                "available": False,
                "reason": detail or "sudo authorization unavailable",
            }
            return
        self._stdout = tempfile.TemporaryFile()
        self._stderr = tempfile.TemporaryFile()
        command = [
            sudo,
            "-n",
            powermetrics,
            "--format",
            "plist",
            "--sample-rate",
            str(self.interval_ms),
            "--samplers",
            "tasks,disk,cpu_power,gpu_power",
            "--show-process-gpu",
            "--show-process-samp-norm",
            "--show-process-wait-times",
            "--show-process-io",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._stdout,
                stderr=self._stderr,
            )
        except Exception as exc:
            self._report = {"available": False, "reason": repr(exc)}
            self._close_files()

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        assert self._stdout is not None
        assert self._stderr is not None
        self._stdout.seek(0)
        self._stderr.seek(0)
        blob = self._stdout.read()
        error = self._stderr.read().decode(errors="replace").strip()
        documents = _powermetrics_documents(blob)
        samples = parse_powermetrics_documents(documents, pid=self.pid)
        self._report = _summarize_powermetrics(samples)
        if not self._report.get("available") and error:
            self._report["reason"] = error
        self._process = None
        self._close_files()

    def _close_files(self) -> None:
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()
        self._stdout = None
        self._stderr = None

    def __enter__(self) -> PowermetricsCollector:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def report(self) -> dict[str, Any]:
        return dict(self._report)


@dataclass
class ResourceRun:
    sampler: ResourceTelemetrySampler
    powermetrics: PowermetricsCollector

    def report(self, **conditions: Any) -> dict[str, Any]:
        return self.sampler.report(
            powermetrics=self.powermetrics.report(),
            **conditions,
        )
