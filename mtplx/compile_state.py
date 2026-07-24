"""Shared 'inside a compiled forward trace' flag (issue #51, 70 tps goal).

A dependency-free home for one bit of state so `compiled_forward` (which sets it)
and the model forwards (which read it) can agree without an import cycle.

The model decode loop keeps the GPU fed during Python graph-build by calling
`mx.async_eval` every N layers (the submit cadence). Inside an `mx.compile`
trace that call is (a) illegal — "[async_eval] Not allowed inside a graph
transformation" — and (b) pointless, because the whole reason to compile is to
replace the per-layer Python walk with a single traced submission. So while a
compiled forward is tracing/replaying, the forward checks `compile_trace_active()`
and suppresses those scheduling-only host-syncs. Kernel math and ordering are
unchanged; this only removes an eval whose sole job was to paper over the
graph-build stall that compilation eliminates outright.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

_COMPILE_TRACE_ACTIVE = False


def compile_trace_active() -> bool:
    """True while a compiled AR forward is executing (and while it traces)."""
    return _COMPILE_TRACE_ACTIVE


@contextlib.contextmanager
def compile_trace() -> Iterator[None]:
    """Mark the enclosed block as running inside a compiled forward."""
    global _COMPILE_TRACE_ACTIVE
    previous = _COMPILE_TRACE_ACTIVE
    _COMPILE_TRACE_ACTIVE = True
    try:
        yield
    finally:
        _COMPILE_TRACE_ACTIVE = previous
