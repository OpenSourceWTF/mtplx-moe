"""Shared-branch hoist (issue #51, MTPLX_HY3_SHARED_HOIST).

The resident shared MLP depends only on x, not on the routed indices, so it can
be dispatched BEFORE the router host-sync (mx.eval(indices), the per-streamed-
layer barrier) to fill the GPU-idle window of the sync round-trip — instead of
after begin_split_route where it only overlaps the cheaper miss reads. This is a
pure execution-order reorder: same shared_mlp(x), same combine, so the routed
output and the returned shared must be UNCHANGED. These tests lock that:

  1. flag ON  -> shared_work is invoked once, BEFORE the indices eval;
  2. flag OFF -> shared_work is invoked once, AFTER the indices eval (unchanged);
  3. either way the shared branch is computed exactly once (no double-compute)
     and the routed output is identical.
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.models.expert_mlx import HotExpertSwitchGLU
from tests.test_streamed_models import (
    _bank_overlap_inputs,
    _BankOverlapPending,
    _BankOverlapRuntime,
)


def _drive(monkeypatch, flag: str | None):
    """Run one decode call, returning (call order log, shared_call_count, routed, shared)."""
    if flag is None:
        monkeypatch.delenv("MTPLX_HY3_SHARED_HOIST", raising=False)
    else:
        monkeypatch.setenv("MTPLX_HY3_SHARED_HOIST", flag)

    events: list[str] = []
    order: list[str] = []
    runtime = _BankOverlapRuntime(events, _BankOverlapPending(events))
    runtime.config.deferred_pin_release = True  # async_eval path

    from mtplx.models import expert_mlx

    real_eval = expert_mlx.mx.eval
    real_async = getattr(expert_mlx.mx, "async_eval", None)

    def logged_eval(*a, **k):
        order.append("eval")
        return real_eval(*a, **k)

    def logged_async(*a, **k):
        order.append("async_eval")
        return real_async(*a, **k) if real_async else None

    monkeypatch.setattr(expert_mlx.mx, "eval", logged_eval)
    if real_async is not None:
        monkeypatch.setattr(expert_mlx.mx, "async_eval", logged_async)
    monkeypatch.setattr(
        expert_mlx, "_run_component_bank_q4", lambda selected, *a, **k: selected
    )

    shared_calls = {"n": 0}

    def shared_work() -> mx.array:
        shared_calls["n"] += 1
        order.append("shared")
        return mx.ones((1, 1, 2), dtype=mx.bfloat16)

    x, indices = _bank_overlap_inputs()
    switch = HotExpertSwitchGLU(runtime, 1)
    routed, shared = switch.run_with_shared_overlap(x, indices, shared_work)
    mx.eval(routed, shared)
    return order, shared_calls["n"], routed, shared


def test_hoist_on_computes_shared_before_the_indices_sync(monkeypatch) -> None:
    order, n, _routed, shared = _drive(monkeypatch, "1")
    assert n == 1, "shared must be computed exactly once (no double-compute)"
    # 'shared' and its async_eval must precede the first indices eval.
    assert "shared" in order and "eval" in order
    assert order.index("shared") < order.index("eval"), (
        f"hoist must dispatch shared before the indices sync; got {order}"
    )
    assert shared is not None


def test_hoist_off_keeps_shared_after_the_indices_sync(monkeypatch) -> None:
    order, n, _routed, _shared = _drive(monkeypatch, None)
    assert n == 1
    # Unchanged behavior: the indices eval happens before shared is computed.
    assert order.index("eval") < order.index("shared"), (
        f"without the flag the indices sync precedes shared; got {order}"
    )


def test_hoist_preserves_routed_output_bitwise(monkeypatch) -> None:
    _order_off, _n_off, routed_off, shared_off = _drive(monkeypatch, None)
    _order_on, _n_on, routed_on, shared_on = _drive(monkeypatch, "1")
    # Same math, only dispatch order differs -> identical arrays.
    assert mx.array_equal(routed_off, routed_on), "routed output changed under the hoist"
    assert mx.array_equal(shared_off, shared_on), "shared output changed under the hoist"
