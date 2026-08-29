"""Idle-persistence forward progress + shutdown flush (issue #290).

The scheduler's persistence band was reachable only while the
idle_postcommit deque was COMPLETELY empty, so any self-chaining idle
occupant (the 2.8.2 background warm ladder chained rungs for minutes
after the last request) starved SSD session-cache writes forever: the
entry sat in the RAM bank, every cold-tier counter stayed zero, and a
restart lost the session. SIGTERM made it unconditional — lifespan
shutdown cancelled the queued futures before anything could write.

These tests drive the real ``ModelWorkScheduler`` /
``SessionBank`` / ``SessionBankColdTier`` contract, wired exactly the way
``mtplx/server/openai.py`` wires them — no fabricated internal state.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import mlx.core as mx

from mtplx.cache_bank import SessionBankColdTier
from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.session_bank import SessionBank


class FakeRuntime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []


class TrimmableKV:
    """Minimal real-contract cache leaf: state + meta_state + trimmable."""

    def __init__(self, rows: int = 64, width: int = 8) -> None:
        self.state = (
            mx.arange(rows * width, dtype=mx.float32)
            .reshape(1, 1, rows, width)
            .astype(mx.float16)
        )
        self.meta_state = ("kv", str(rows))

    def is_trimmable(self) -> bool:
        return True


LOOKUP_IDENTITY = {
    "model_path": "models/example",
    "mtp_enabled": True,
    "template_hash": "template-a",
    "policy_fingerprint": "policy-a",
}


def make_cold(tmp_path, **kwargs) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
        **kwargs,
    )


def put_small_entry(bank: SessionBank, tokens=(1, 2, 3, 4, 5)):
    return bank.put(
        runtime=FakeRuntime(),
        token_ids=list(tokens),
        cache=[TrimmableKV()],
        logits=mx.array([[0.5, 1.5]], dtype=mx.float16),
        hidden=mx.array([[[2.0, 3.0]]], dtype=mx.float16),
        session_id="s1",
        template_hash="template-a",
        policy_fingerprint="policy-a",
        snapshot_epoch=7,
    )


def wire_like_server(bank: SessionBank, cold: SessionBankColdTier,
                     scheduler: ModelWorkScheduler) -> None:
    """The exact openai.py wiring: idle-persistence dispatch + foreground
    yield signal."""
    bank.cold_enqueue_dispatch = lambda job: scheduler.submit_idle_persistence(
        job,
        batch_key="ssd.cold_enqueue",
        coalesce_key=getattr(job, "coalesce_key", None),
    )
    cold.foreground_busy = scheduler.foreground_busy


def start_idle_chain(scheduler: ModelWorkScheduler, stop: Event) -> None:
    """A self-chaining idle_postcommit occupant: each step enqueues its
    successor before completing — the background-warm-ladder shape that
    keeps the idle deque non-empty at every scheduler decision point."""

    def step(index: int = 0) -> None:
        if not stop.is_set():
            scheduler.submit_idle_postcommit(
                lambda: step(index + 1), batch_key=f"warmup.background:{index}"
            )

    scheduler.submit_idle_postcommit(step, batch_key="warmup.background:0")


def test_idle_chain_starves_persistence_without_pump():
    """Pin the #290 mechanism: with the idle band occupied by a chain, a
    queued persistence item never runs — the band is only reachable when
    the idle deque is empty and every completion re-arms the quiet
    anchor."""
    scheduler = ModelWorkScheduler(name="starve-pin")
    stop = Event()
    ran: list[float] = []
    try:
        start_idle_chain(scheduler, stop)
        time.sleep(0.1)
        scheduler.submit_idle_persistence(
            lambda: ran.append(time.monotonic()), coalesce_key="ssd_cold:s1"
        )
        time.sleep(1.5)  # several chain generations
        assert not ran
        assert scheduler.stats()["persistence_pending"] == 1
    finally:
        stop.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_pump_drains_persistence_through_busy_idle_band_to_disk(tmp_path):
    """(a) Entries enqueued while the idle band is busy reach DISK once
    the server-side pump opens slots — without any further put() calls."""
    cold = make_cold(tmp_path)
    scheduler = ModelWorkScheduler(name="pump-drain")
    stop = Event()
    try:
        bank = SessionBank(cold_tier=cold)
        wire_like_server(bank, cold, scheduler)
        start_idle_chain(scheduler, stop)
        time.sleep(0.1)
        entry = scheduler.submit_foreground(
            lambda: put_small_entry(bank), batch_key="postcommit.stream.final:s1"
        ).result(timeout=10)
        assert entry is not None and not entry.live_ref_only
        # The chain occupies the idle band: the encode stays queued.
        time.sleep(1.2)
        assert cold.stats().get("writes_enqueued", 0) == 0
        assert scheduler.stats()["persistence_pending"] == 1
        # The idle pump (what _idle_persistence_pump_loop does per tick).
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            scheduler.pump_persistence(budget=1)
            if cold.stats().get("writes_enqueued", 0) > 0:
                break
            time.sleep(0.1)
        assert cold.stats()["writes_enqueued"] == 1
        assert scheduler.stats()["persistence_pumped"] >= 1
        assert cold.flush(timeout_s=10.0)
        assert cold.stats()["writes_completed"] == 1
        # The payload's blobs really exist on disk.
        blob_files = list((tmp_path / "session-bank" / "blobs").rglob("*.bin"))
        assert blob_files
        assert cold.lookup([1, 2, 3, 4, 5], **LOOKUP_IDENTITY) is not None
    finally:
        stop.set()
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()


def test_foreground_still_defers_and_disarms_the_pump():
    """(c) Foreground-busy still defers: an armed pump never lets the
    persistence head run ahead of foreground work, and a foreground
    submission zeroes any armed budget (the tail-gap fence)."""
    scheduler = ModelWorkScheduler(name="pump-fence")
    order: list[str] = []
    started = Event()
    release = Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=5)
        order.append("foreground-1")

    try:
        first = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        scheduler.submit_idle_persistence(lambda: order.append("persistence"))
        assert scheduler.pump_persistence(budget=4)
        # Arming while a foreground runs must not admit the encode.
        time.sleep(0.3)
        assert "persistence" not in order
        # A foreground submission disarms the remaining budget entirely.
        second = scheduler.submit_foreground(lambda: order.append("foreground-2"))
        assert scheduler.stats()["persistence_pump_budget"] == 0
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)
        # With no idle occupancy the item then drains through the normal
        # quiet window — foreground strictly first.
        deadline = time.monotonic() + 5.0
        while "persistence" not in order and time.monotonic() < deadline:
            time.sleep(0.05)
        assert order == ["foreground-1", "foreground-2", "persistence"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_pump_is_inert_without_pending_persistence():
    scheduler = ModelWorkScheduler(name="pump-inert")
    try:
        assert scheduler.pump_persistence(budget=2) is False
        assert scheduler.stats()["persistence_pump_budget"] == 0
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


class _FlushState:
    """The minimal ServerState surface _shutdown_flush_cold_writes reads."""

    def __init__(self, scheduler, cold, bank) -> None:
        self.model_scheduler = scheduler
        self.session_bank_cold_tier = cold

        class _Sessions:
            def __init__(self, inner_bank) -> None:
                self.bank = inner_bank

            def flush_cold_tier(self, *, timeout_s: float = 30.0) -> bool:
                flush = getattr(self.bank, "flush_cold_tier", None)
                if callable(flush):
                    return bool(flush(timeout_s=timeout_s))
                return True

        self.sessions = _Sessions(bank)


def test_shutdown_flush_writes_pending_entries_within_bound(tmp_path, capsys):
    """(b) SIGTERM path: entries still queued behind a busy idle band are
    flushed to disk by the bounded shutdown drain, with an honest console
    line — instead of being cancelled and lost."""
    from mtplx.server.openai import _shutdown_flush_cold_writes

    cold = make_cold(tmp_path)
    scheduler = ModelWorkScheduler(name="shutdown-flush")
    stop = Event()
    try:
        bank = SessionBank(cold_tier=cold)
        wire_like_server(bank, cold, scheduler)
        start_idle_chain(scheduler, stop)
        time.sleep(0.1)
        scheduler.submit_foreground(
            lambda: put_small_entry(bank), batch_key="postcommit.stream.final:s1"
        ).result(timeout=10)
        time.sleep(1.0)
        assert cold.stats().get("writes_enqueued", 0) == 0  # starved, pending
        outcome = _shutdown_flush_cold_writes(
            _FlushState(scheduler, cold, bank)
        )
        assert outcome["pending_before"] >= 1
        assert outcome["flushed"] >= 1
        assert outcome["remained"] == 0
        assert cold.stats()["writes_completed"] == 1
        assert cold.lookup([1, 2, 3, 4, 5], **LOOKUP_IDENTITY) is not None
        console = capsys.readouterr().out
        assert "shutdown: flushing" in console
        assert "flushed 1/1" in console or "flushed" in console
    finally:
        stop.set()
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()


def test_shutdown_flush_sees_the_in_flight_writer_write(
    tmp_path, capsys, monkeypatch
):
    """Regression pin (live-caught): a staged write the writer thread has
    already popped shows writer_queue_depth == 0 while the file is still
    being written — the flush must count it as outstanding (enqueued -
    completed accounting), wait for it, and report honestly when the bound
    is hit rather than silently skipping and letting the daemon writer die
    mid-file."""
    from mtplx.server.openai import _shutdown_flush_cold_writes

    monkeypatch.setenv("MTPLX_SHUTDOWN_SSD_FLUSH_S", "1")
    # Split the two halves of the foreground-yield contract: the ENCODE
    # must complete (so the write stages), while the WRITER pauses on the
    # same signal (so the staged write is popped but not yet durable).
    monkeypatch.setenv("MTPLX_SSD_ENCODE_FOREGROUND_YIELD", "0")
    cold = make_cold(tmp_path)
    scheduler = ModelWorkScheduler(name="shutdown-inflight")
    try:
        # Bank without the tier wired: this test drives the tier directly
        # so exactly ONE staged write exists.
        bank = SessionBank()
        entry = put_small_entry(bank)
        assert entry is not None
        # Pin the writer in its foreground pause: the write is popped from
        # the queue (depth 0) but NOT durable (completed 0). One callable,
        # mutable value: the writer captures the callable at pause entry.
        busy = {"value": True}
        cold.foreground_busy = lambda: busy["value"]
        assert cold.put_entry(entry, capabilities=["ar_insert"])
        deadline = time.monotonic() + 5.0
        while (
            cold.stats().get("writer_queue_depth", 0) != 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        stats = cold.stats()
        assert stats["writes_enqueued"] == 1
        assert stats["writes_completed"] == 0
        assert stats["writer_queue_depth"] == 0  # invisible to depth alone
        outcome = _shutdown_flush_cold_writes(_FlushState(scheduler, cold, bank))
        assert outcome["pending_before"] == 1
        assert outcome["remained"] == 1  # bound hit while the writer is held
        console = capsys.readouterr().out
        assert "shutdown: flushing 1 pending" in console
        assert "1 remained (bound hit)" in console
        # Release the writer: the held write completes durably.
        busy["value"] = False
        assert cold.flush(timeout_s=10.0)
        assert cold.stats()["writes_completed"] == 1
    finally:
        try:
            busy["value"] = False
        except NameError:
            pass
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()


def test_shutdown_flush_is_silent_with_nothing_pending(tmp_path, capsys):
    from mtplx.server.openai import _shutdown_flush_cold_writes

    cold = make_cold(tmp_path)
    scheduler = ModelWorkScheduler(name="shutdown-noop")
    try:
        bank = SessionBank(cold_tier=cold)
        wire_like_server(bank, cold, scheduler)
        outcome = _shutdown_flush_cold_writes(_FlushState(scheduler, cold, bank))
        assert outcome == {
            "pending_before": 0,
            "flushed": 0,
            "remained": 0,
            "elapsed_s": 0.0,
        }
        assert "shutdown: flushing" not in capsys.readouterr().out
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()


def test_idle_pump_loop_gates_on_request_idleness(tmp_path):
    """The server loop arms the pump only after real request-level quiet:
    a recent request (or in-flight foreground) keeps it inert."""
    import asyncio

    from mtplx.server.openai import (
        _IDLE_PUMP_AFTER_S,
        _idle_persistence_pump_loop,
        _server_request_idle_s,
    )

    cold = make_cold(tmp_path)
    scheduler = ModelWorkScheduler(name="pump-loop")
    stop = Event()

    class _State:
        def __init__(self) -> None:
            self.model_scheduler = scheduler
            self.last_request_started_at = 0.0
            self.last_request_at = 0.0

        def has_foreground(self) -> bool:
            return False

    state = _State()
    try:
        bank = SessionBank(cold_tier=cold)
        wire_like_server(bank, cold, scheduler)
        start_idle_chain(scheduler, stop)
        time.sleep(0.1)
        scheduler.submit_foreground(
            lambda: put_small_entry(bank), batch_key="postcommit.stream.final:s1"
        ).result(timeout=10)

        # A request that just finished keeps the pump inert.
        state.last_request_at = time.time()
        assert _server_request_idle_s(state) < _IDLE_PUMP_AFTER_S

        async def drive(seconds: float) -> None:
            task = asyncio.get_running_loop().create_task(
                _idle_persistence_pump_loop(state, interval_s=0.05)
            )
            try:
                await asyncio.sleep(seconds)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(drive(0.6))
        assert cold.stats().get("writes_enqueued", 0) == 0  # still starved

        # The server has now been idle past the arm threshold: drains.
        state.last_request_at = time.time() - (_IDLE_PUMP_AFTER_S + 1.0)
        assert _server_request_idle_s(state) >= _IDLE_PUMP_AFTER_S
        asyncio.run(drive(3.0))
        assert cold.stats().get("writes_enqueued", 0) == 1
        assert cold.flush(timeout_s=10.0)
        assert cold.stats()["writes_completed"] == 1
    finally:
        stop.set()
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()
