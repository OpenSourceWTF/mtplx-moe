"""Stream stall watchdog (#86 containment).

The probe compares successive readings of the model-owner progress heartbeat:
time alone must never breach (long prefills and model loads are healthy while
the heartbeat ticks) — only a heartbeat frozen for the full deadline does.
"""

from __future__ import annotations

from mtplx import progress_heartbeat
from mtplx.server.openai import _OwnerStallProbe


def test_probe_never_breaches_while_progress_advances():
    ticks = iter(range(10_000))
    clocks = iter(float(i * 100.0) for i in range(10_000))
    probe = _OwnerStallProbe(
        deadline_s=300.0,
        progress=lambda: next(ticks),
        clock=lambda: next(clocks),
    )
    # 100 simulated seconds between polls, far past the deadline in wall time,
    # but the owner ticks between every poll: never a breach.
    for _ in range(50):
        assert probe.observe() is None


def test_probe_breaches_only_after_full_deadline_with_frozen_progress():
    now = {"t": 0.0}
    probe = _OwnerStallProbe(
        deadline_s=300.0, progress=lambda: 7, clock=lambda: now["t"]
    )
    now["t"] = 299.9
    assert probe.observe() is None
    now["t"] = 300.0
    assert probe.observe() == 300.0


def test_probe_resets_when_progress_ticks_mid_wait():
    state = {"progress": 0, "t": 0.0}
    probe = _OwnerStallProbe(
        deadline_s=300.0,
        progress=lambda: state["progress"],
        clock=lambda: state["t"],
    )
    state["t"] = 200.0
    assert probe.observe() is None
    state["progress"] = 1  # the owner made progress: the window restarts
    state["t"] = 400.0
    assert probe.observe() is None
    state["t"] = 699.9
    assert probe.observe() is None
    state["t"] = 700.0
    assert probe.observe() == 300.0


def test_probe_disabled_with_zero_deadline():
    probe = _OwnerStallProbe(deadline_s=0.0, progress=lambda: 3, clock=lambda: 0.0)
    assert probe.observe(10_000.0) is None


def test_heartbeat_ticks_are_monotone():
    before = progress_heartbeat.value()
    progress_heartbeat.tick()
    progress_heartbeat.tick()
    assert progress_heartbeat.value() == before + 2
