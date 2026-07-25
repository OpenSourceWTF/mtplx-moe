"""Issue #144: shed cache weight under system memory pressure.

The bank held its full budget while a 64 GB Mac swapped 60 GB. The guard
loop shrinks the bank to half budget on WARNING and empties it on
CRITICAL; `shrink_to_bytes` is the bank-side primitive.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace


import mtplx.server.openai as srv


class FakeBank:
    def __init__(self, total, max_bytes):
        self.total_nbytes = total
        self.max_bytes = max_bytes
        self.calls = []

    def shrink_to_bytes(self, target, *, reason):
        self.calls.append((target, reason))
        evicted = 1 if self.total_nbytes > target else 0
        self.total_nbytes = min(self.total_nbytes, target)
        return evicted


def make_state(bank):
    return SimpleNamespace(
        sessions=SimpleNamespace(bank=bank),
        dashboard=SimpleNamespace(last_memory_pressure_level=0),
    )


def run_one_tick(state, level, monkeypatch):
    monkeypatch.setattr(srv, "_memory_pressure_level", lambda: level)

    async def one_tick():
        task = asyncio.ensure_future(
            srv._memory_pressure_loop(state, interval_s=3600)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_tick())


def test_warning_shrinks_bank_to_half_budget(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=2, monkeypatch=monkeypatch)
    assert bank.calls == [(4 << 30, "memory_pressure_warning")]
    assert state.dashboard.last_memory_pressure_level == 2


def test_critical_empties_bank(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=4, monkeypatch=monkeypatch)
    assert bank.calls == [(0, "memory_pressure_critical")]


def test_normal_level_touches_nothing(monkeypatch):
    bank = FakeBank(total=8 << 30, max_bytes=8 << 30)
    state = make_state(bank)
    run_one_tick(state, level=1, monkeypatch=monkeypatch)
    assert bank.calls == []
    assert state.dashboard.last_memory_pressure_level == 1


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_MEMORY_PRESSURE_GUARD", "0")
    assert not srv._memory_pressure_guard_enabled()
    monkeypatch.delenv("MTPLX_MEMORY_PRESSURE_GUARD")
    assert srv._memory_pressure_guard_enabled()


def test_guard_core_sustained_warning_rearms_after_interval():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=1000.0, busy=False) is True  # rising edge acts
    assert g.decide(2, now=1010.0, busy=False) is False  # no per-tick re-trim
    assert g.decide(2, now=1100.0, busy=False) is False  # inside 120s window
    assert g.decide(2, now=1125.0, busy=False) is True  # re-armed


def test_guard_core_warning_defers_while_busy_then_acts():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=True) is False
    assert g.decide(2, now=30.0, busy=True) is False
    assert g.decide(2, now=61.0, busy=True) is True  # defer window expired


def test_guard_core_warning_acts_when_engine_goes_idle():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=True) is False
    assert g.decide(2, now=10.0, busy=False) is True


def test_guard_core_critical_never_defers():
    g = srv._MemoryPressureGuard()
    assert g.decide(4, now=0.0, busy=True) is True


def test_guard_core_flapping_cannot_retrigger_fast():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=False) is True
    assert g.decide(1, now=10.0, busy=False) is False
    assert g.decide(2, now=20.0, busy=False) is False  # edge inside 30s spacing
    assert g.decide(1, now=30.0, busy=False) is False
    assert g.decide(2, now=40.0, busy=False) is True  # edge after spacing


def test_guard_core_escalation_to_critical_acts_immediately():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=False) is True
    assert g.decide(4, now=10.0, busy=True) is True  # 2->4 edge, no spacing gate


def test_guard_core_recovery_clears_pending_action():
    g = srv._MemoryPressureGuard()
    assert g.decide(2, now=0.0, busy=True) is False  # owed, deferred
    assert g.decide(1, now=10.0, busy=False) is False  # recovered: owed cleared
    assert g.decide(1, now=200.0, busy=False) is False


def test_bank_shrink_to_bytes_evicts_lru_first():
    from mtplx.session_bank import SessionBank

    bank = SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    # Fabricate three entries with staggered ages via the internal table:
    # shrink must drop the oldest-accessed first.
    from mtplx.session_bank import SessionBankEntry, CacheSnapshot

    def entry(name, last_access, nbytes):
        return SessionBankEntry(
            token_ids=(hash(name) % 1000, 2, 3),
            token_hash=name,
            model_path="/m",
            mtp_enabled=False,
            hidden_variant=None,
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            cache_ref=None,
            nbytes=nbytes,
            session_id=name,
            last_access_s=last_access,
        )

    # The bank's invariant: entries are keyed by their token_ids tuple.
    # (This test originally used string keys, which made _evict_entry's
    # pop-by-token_ids miss forever — shrink_to_bytes spun allocating
    # eviction-log records until the machine ran out of RAM.)
    fabricated = [
        entry("old", last_access=1.0, nbytes=400),
        entry("mid", last_access=2.0, nbytes=400),
        entry("new", last_access=3.0, nbytes=400),
    ]
    bank._entries = {e.token_ids: e for e in fabricated}
    evicted = bank.shrink_to_bytes(500)
    assert evicted == 2
    assert [e.session_id for e in bank._entries.values()] == ["new"]
