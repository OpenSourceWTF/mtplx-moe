"""Opt-in per-layer stage attribution for the streamed expert hot path.

Enabled by MTPLX_ROUTE_STAGE_PROBE=1. When disabled (default), the only cost
is one module-level boolean check per bracket. When enabled, brackets
accumulate nanosecond sums/counts per (phase, stage) and an atexit hook dumps
the aggregate to MTPLX_ROUTE_STAGE_PROBE_PATH (default
/tmp/mtplx-route-stage-probe.json). Diagnostic lane only: throughput measured
with the probe enabled is not promotion evidence.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

ENABLED = os.environ.get("MTPLX_ROUTE_STAGE_PROBE", "").strip() == "1"
_PATH = os.environ.get(
    "MTPLX_ROUTE_STAGE_PROBE_PATH", "/tmp/mtplx-route-stage-probe.json"
)

_SUMS: dict[str, int] = defaultdict(int)
_COUNTS: dict[str, int] = defaultdict(int)


@contextmanager
def bracket(stage: str):
    if not ENABLED:
        yield
        return
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        _SUMS[stage] += time.perf_counter_ns() - started
        _COUNTS[stage] += 1


def count(event: str, amount: int = 1) -> None:
    if ENABLED:
        _COUNTS[event] += amount


def snapshot() -> dict:
    stages = sorted(set(_SUMS) | set(_COUNTS))
    return {
        "enabled": ENABLED,
        "stages": {
            name: {
                "total_ms": _SUMS.get(name, 0) / 1e6,
                "count": _COUNTS.get(name, 0),
                "mean_us": (
                    (_SUMS.get(name, 0) / _COUNTS[name] / 1e3)
                    if _COUNTS.get(name)
                    else None
                ),
            }
            for name in stages
        },
    }


def _dump() -> None:
    if ENABLED and (_SUMS or _COUNTS):
        with open(_PATH, "w") as handle:
            json.dump(snapshot(), handle, indent=2, sort_keys=True)


atexit.register(_dump)
