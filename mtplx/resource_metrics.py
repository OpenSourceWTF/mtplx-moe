"""Low-contention cumulative occupancy counters for runtime work pools."""

from __future__ import annotations

import threading
import time
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
