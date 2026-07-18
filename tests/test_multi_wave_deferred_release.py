"""A non-final route wave must never defer its release (issue #120).

Deferring hands the layer lock and the slot pins to the next
generation-thread flush.  A verify batch wider than the transient-slot
capacity splits into several route waves inside a single forward, and the
per-layer lock is not reentrant: deferring a non-final wave leaves the
generation thread holding a lock that the very next wave re-acquires, so the
thread waits on itself forever.  That is what hung GLM-5.2 at depth >= 4 (a
6-row verify batch at top-8 needs 40 assignments against 32 transient slots,
so depth 3 fits one wave and depth 4 does not) with zero CPU burned and every
MLX worker idle.

The fakes here mirror production lock ownership exactly: ``begin_split_route``
acquires the layer lock and hands it to the pending route (released only by
``close``), and ``try_all_hit_route`` re-takes the same lock with ``with``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.expert_runtime import RouteWave
from mtplx.expert_streaming import RoutingPhase
from mtplx.models import expert_mlx
from mtplx.models.expert_mlx import HotExpertSwitchGLU, expert_routing_phase

_JOIN_TIMEOUT_S = 20.0


class _LayerLockPending:
    """Split route that owns the layer lock until it closes."""

    def __init__(
        self,
        events: list[str],
        lock: threading.Lock,
        bank: object,
        experts: tuple[int, ...],
    ) -> None:
        self.events = events
        self._lock = lock
        self.plan = SimpleNamespace(hits=experts, misses=())
        self.misses_pending = False
        self.hit_ready = SimpleNamespace(
            bindings=tuple(
                SimpleNamespace(expert=expert, buffer=SimpleNamespace(bank=bank))
                for expert in experts
            )
        )
        self.closed = False

    def iter_ready_misses(self):
        return iter(())

    def release_hits(self) -> None:
        self.events.append("release-hits")

    def release_miss(self, _part) -> None:  # pragma: no cover - no misses here
        raise AssertionError("this fake plans no misses")

    def abort(self, error: BaseException) -> None:
        self.events.append(f"abort:{type(error).__name__}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.events.append("close")
        self._lock.release()


class _MultiWaveRuntime:
    """Runtime double whose layer lock behaves like the real one."""

    def __init__(self, events: list[str], *, all_hit: bool = False) -> None:
        self.events = events
        self.all_hit = all_hit
        self.spec = SimpleNamespace(
            top_k=1,
            hidden_size=2,
            quant_group_size=64,
            quant_bits=4,
        )
        self.manifest = SimpleNamespace(sidecar=None)
        self.config = SimpleNamespace(
            slot_layout="component-banks",
            resource_telemetry=False,
            deferred_pin_release=True,
            split_route_release="deferred",
        )
        self._pipeline_ledger = None
        self._bank = object()
        self.layer_lock = threading.Lock()
        self.deferred: list[object] = []
        self.released: list[bool] = []

    def observe_route(self, *_args, **_kwargs) -> None:
        return None

    def prepare_prefill_seed(self, *_args, **_kwargs) -> tuple[int, ...]:
        return ()

    def route_waves(self, expert_ids, **_kwargs):
        """One assignment per wave: a batch too wide for the slot capacity."""

        experts = tuple(expert_ids)
        return tuple(
            RouteWave(positions=(index,), experts=(expert,))
            for index, expert in enumerate(experts)
        )

    def try_all_hit_route(self, _layer, _experts, **_kwargs):
        # Production probes under the layer lock, so a lock retained by an
        # earlier wave parks the generation thread right here.
        with self.layer_lock:
            self.events.append("try-all-hit")
            if not self.all_hit:
                return None
            return SimpleNamespace(
                plan=SimpleNamespace(hits=(0,)),
                bindings=(
                    SimpleNamespace(
                        expert=0,
                        buffer=SimpleNamespace(bank=self._bank),
                    ),
                ),
                release=lambda **kwargs: self.released.append(
                    kwargs.get("synchronize", True)
                ),
            )

    def begin_split_route(self, _layer, experts, **_kwargs):
        self.layer_lock.acquire()
        self.events.append("begin-split")
        return _LayerLockPending(
            self.events,
            self.layer_lock,
            self._bank,
            tuple(experts),
        )

    def defer_slot_release(self, ready, _outputs) -> None:
        self.deferred.append(ready)
        self.events.append("defer")


def _inputs(tokens: int) -> tuple[mx.array, mx.array]:
    return (
        mx.zeros((tokens, 1, 2), dtype=mx.bfloat16),
        mx.array([[[expert]] for expert in range(tokens)], dtype=mx.int32),
    )


def _run_switch(runtime: _MultiWaveRuntime, tokens: int) -> threading.Thread:
    """Drive one forward on a worker so a deadlock fails instead of hanging."""

    switch = HotExpertSwitchGLU(runtime, 1)
    box: dict[str, BaseException] = {}

    def target() -> None:
        try:
            with expert_routing_phase(RoutingPhase.DECODE):
                switch(*_inputs(tokens))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name="generation")
    thread.start()
    thread.join(_JOIN_TIMEOUT_S)
    if "error" in box:
        raise box["error"]
    return thread


@pytest.fixture(autouse=True)
def _stub_component_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        expert_mlx,
        "_run_component_bank_q4",
        lambda selected, *_args, **_kwargs: selected,
    )


def test_multi_wave_decode_does_not_deadlock_on_the_layer_lock() -> None:
    events: list[str] = []
    runtime = _MultiWaveRuntime(events)

    thread = _run_switch(runtime, tokens=2)

    assert not thread.is_alive(), (
        "multi-wave decode deadlocked: the generation thread is waiting on a "
        "layer lock it already holds from the previous wave's deferred release"
    )
    assert events.count("begin-split") == 2
    # Exactly one deferral, and only after the earlier wave gave the lock back.
    assert events.count("defer") == 1
    assert events.count("close") == 1
    assert events.index("close") < events.index("defer")
    # The final wave's release is still outstanding by design: the lock stays
    # held until the next generation-thread flush.
    assert runtime.layer_lock.locked()
    assert len(runtime.deferred) == 1


def test_single_wave_decode_still_defers() -> None:
    """The measured fast path is untouched when one wave covers the batch."""

    events: list[str] = []
    runtime = _MultiWaveRuntime(events)

    thread = _run_switch(runtime, tokens=1)

    assert not thread.is_alive()
    assert events.count("begin-split") == 1
    assert events.count("defer") == 1
    assert "close" not in events
    assert runtime.layer_lock.locked()


def test_multi_wave_all_hit_fences_every_wave_but_the_last() -> None:
    events: list[str] = []
    runtime = _MultiWaveRuntime(events, all_hit=True)

    thread = _run_switch(runtime, tokens=3)

    assert not thread.is_alive()
    assert events.count("try-all-hit") == 3
    # Two fenced releases, then one deferral on the wave that ends the layer.
    assert runtime.released == [False, False]
    assert len(runtime.deferred) == 1
    assert not runtime.layer_lock.locked()
