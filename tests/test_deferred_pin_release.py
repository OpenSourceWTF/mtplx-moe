"""Deferred pin release: the next generation-thread eval covers the fence.

The all-hit expert wave's output is an ancestor of the next layer's router
indices, so `mx.eval(indices)` at layer L+1 materializes layer L's wave. Pin
release for L can therefore run after that eval on the generation thread —
no per-layer blocking fence and no completion-lane eval.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.expert_runtime import ExpertStreamingRuntime


class _FakeReady:
    def __init__(self, log: list, name: str) -> None:
        self._log = log
        self._name = name
        self.released: bool = False

    def release(self, *, synchronize: bool = True) -> None:
        assert not self.released, "double release"
        self.released = True
        self._log.append((self._name, synchronize))


def _runtime_stub() -> ExpertStreamingRuntime:
    runtime = ExpertStreamingRuntime.__new__(ExpertStreamingRuntime)
    runtime._deferred_slot_releases = []
    return runtime


def test_defer_then_flush_releases_in_order_without_synchronize() -> None:
    runtime = _runtime_stub()
    log: list = []
    first = _FakeReady(log, "first")
    second = _FakeReady(log, "second")
    runtime.defer_slot_release(first, mx.zeros((1,)))
    runtime.defer_slot_release(second, mx.zeros((1,)))
    assert log == []
    runtime.flush_deferred_slot_releases()
    assert log == [("first", False), ("second", False)]
    assert not runtime._deferred_slot_releases


def test_flush_is_idempotent_and_safe_when_empty() -> None:
    runtime = _runtime_stub()
    runtime.flush_deferred_slot_releases()
    log: list = []
    ready = _FakeReady(log, "only")
    runtime.defer_slot_release(ready, mx.zeros((1,)))
    runtime.flush_deferred_slot_releases()
    runtime.flush_deferred_slot_releases()
    assert log == [("only", False)]


def test_flush_releases_remaining_entries_when_one_raises() -> None:
    runtime = _runtime_stub()
    log: list = []

    class _Exploding(_FakeReady):
        def release(self, *, synchronize: bool = True) -> None:
            super().release(synchronize=synchronize)
            raise RuntimeError("completion error")

    first = _Exploding(log, "boom")
    second = _FakeReady(log, "after")
    runtime.defer_slot_release(first, mx.zeros((1,)))
    runtime.defer_slot_release(second, mx.zeros((1,)))
    with pytest.raises(RuntimeError, match="completion error"):
        runtime.flush_deferred_slot_releases()
    assert ("after", False) in log
    assert not runtime._deferred_slot_releases


def test_all_hit_switch_defers_release_when_enabled(monkeypatch) -> None:
    from types import SimpleNamespace

    from mtplx.models import expert_mlx
    from mtplx.models.expert_mlx import HotExpertSwitchGLU
    from tests.test_streamed_models import (
        _bank_overlap_inputs,
        _BankOverlapPending,
        _BankOverlapRuntime,
    )

    events: list[str] = []
    runtime = _BankOverlapRuntime(events, _BankOverlapPending(events))
    runtime.spec.quant_bits = 2
    released: list[bool] = []
    deferred: list[tuple] = []
    bank = object()
    bindings = tuple(
        SimpleNamespace(expert=expert, buffer=SimpleNamespace(bank=bank))
        for expert in (0, 1, 2)
    )
    ready = SimpleNamespace(
        plan=SimpleNamespace(hits=(0, 1, 2)),
        bindings=bindings,
        release=lambda **kwargs: released.append(kwargs.get("synchronize", True)),
    )
    runtime.try_all_hit_route = lambda *_args, **_kwargs: ready
    runtime.defer_slot_release = lambda r, out: deferred.append((r, out))
    monkeypatch.setattr(
        expert_mlx, "_run_component_bank_q4", lambda selected, *a, **k: selected
    )
    fences: list[object] = []
    monkeypatch.setattr(expert_mlx.mx, "eval", lambda *values: fences.append(values))

    monkeypatch.setattr(expert_mlx, "_DEFERRED_PIN_RELEASE", True)
    HotExpertSwitchGLU(runtime, 1)(*_bank_overlap_inputs())
    assert len(deferred) == 1 and deferred[0][0] is ready
    assert released == []

    deferred.clear()
    monkeypatch.setattr(expert_mlx, "_DEFERRED_PIN_RELEASE", False)
    HotExpertSwitchGLU(runtime, 1)(*_bank_overlap_inputs())
    assert deferred == []
    assert released == [False]
