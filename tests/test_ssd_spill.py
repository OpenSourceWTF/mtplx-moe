"""Streaming SSD spill — durability for the sessions the caps excluded.

Issues #305/#323: three stacked gates guaranteed that exactly the
sessions whose re-prefill costs minutes could never persist —

* the per-session RAM cap turned >8 GiB snapshots into live-ref leases,
* ``_enqueue_cold_entry`` hard-skipped live-ref entries (#323),
* the writer queue's 4 GiB backlog budget stages the fully encoded
  payload in RAM, so >4 GiB entries could not even be enqueued.

``spill_entry`` streams tensor-by-tensor to the same on-disk format, and
the bank schedules idle-lane spills for live-ref sessions. These tests
drive the real ``SessionBank``/``SessionBankColdTier`` contract — no
fabricated internal state (mistakes-ledger rule: the fabricated-state
shrink test spun an eviction loop to 122 GB once).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from mtplx.cache_bank import SessionBankColdTier
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


def make_cold(tmp_path, **kwargs) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
        **kwargs,
    )


LOOKUP_IDENTITY = {
    "model_path": "models/example",
    "mtp_enabled": True,
    "template_hash": "template-a",
    "policy_fingerprint": "policy-a",
}


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


def test_spill_entry_round_trips_via_lookup(tmp_path):
    cold = make_cold(tmp_path)
    try:
        bank = SessionBank()
        entry = put_small_entry(bank)
        assert entry is not None and not entry.live_ref_only

        assert cold.spill_entry(
            entry, capabilities=["ar_insert", "mtp_full"]
        )
        stats = cold.stats()
        assert stats["spill_writes_completed"] == 1
        assert stats["writes_completed"] == 1
        # Nothing rode the staged queue.
        assert stats.get("writes_enqueued", 0) == 0

        record = cold.lookup([1, 2, 3, 4, 5], **LOOKUP_IDENTITY)
        assert record is not None
        # The payload's blobs really exist on disk (streamed writes).
        blob_files = list((tmp_path / "session-bank" / "blobs").rglob("*.bin"))
        assert blob_files
    finally:
        cold.close()


def test_spill_dedupes_against_existing_blobs(tmp_path):
    cold = make_cold(tmp_path)
    try:
        bank = SessionBank()
        entry = put_small_entry(bank)
        assert cold.spill_entry(entry, capabilities=["ar_insert"])
        # Same identity, same tensors: the second spill dedupes at the
        # entry level (manifest hit), not by rewriting blobs.
        assert cold.spill_entry(entry, capabilities=["ar_insert"])
        assert cold.stats()["spill_writes_completed"] == 1
    finally:
        cold.close()


def test_spill_succeeds_where_the_staged_queue_cannot(tmp_path, monkeypatch):
    # A backlog budget of one byte makes put_entry's staged path refuse
    # everything — the pre-spill world for >4 GiB sessions, shrunk to
    # test scale. The streaming path must not care.
    monkeypatch.setenv("MTPLX_SSD_WRITER_BACKLOG_BYTES", "1")
    cold = make_cold(tmp_path)
    try:
        bank = SessionBank()
        entry = put_small_entry(bank)
        assert not cold.put_entry(entry, capabilities=["ar_insert"])
        assert cold.spill_entry(entry, capabilities=["ar_insert"])
        assert cold.lookup([1, 2, 3, 4, 5], **LOOKUP_IDENTITY) is not None
    finally:
        cold.close()


def test_cold_enqueue_routes_backlog_sized_entries_to_spill(tmp_path):
    cold = make_cold(tmp_path)
    try:
        bank = SessionBank(cold_tier=cold)
        # Every entry counts as backlog-sized: the enqueue job must pick
        # the streaming path on its own.
        cold._backlog_budget_bytes = 1
        entry = put_small_entry(bank)
        assert entry is not None
        cold.flush(timeout_s=5.0)
        stats = cold.stats()
        assert stats["spill_writes_completed"] == 1
        assert stats.get("writes_enqueued", 0) == 0
    finally:
        cold.close()


def test_live_ref_put_schedules_spill_and_persists(tmp_path):
    cold = make_cold(tmp_path)
    try:
        jobs = []
        bank = SessionBank(per_session_max_bytes=1, cold_tier=cold)
        bank.cold_enqueue_dispatch = jobs.append
        entry = bank.put(
            runtime=FakeRuntime(),
            token_ids=[1, 2, 3, 4, 5],
            cache=[TrimmableKV()],
            logits=mx.array([[0.5, 1.5]], dtype=mx.float16),
            hidden=mx.array([[[2.0, 3.0]]], dtype=mx.float16),
            keep_live_ref=True,
            session_id="s1",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=7,
        )
        assert entry is not None and entry.live_ref_only
        assert bank.last_put_skipped_oversized_snapshot
        # #323 regression pin: the skip now schedules a durable spill.
        assert len(jobs) == 1
        assert jobs[0].coalesce_key == "ssd_cold:s1"

        assert jobs[0]() is True
        assert entry.cold_encode_completed_at is not None
        assert cold.stats()["spill_writes_completed"] == 1
        assert cold.lookup([1, 2, 3, 4, 5], **LOOKUP_IDENTITY) is not None
    finally:
        cold.close()


def test_live_ref_spill_epoch_guard_skips_superseded_commits(tmp_path):
    cold = make_cold(tmp_path)
    try:
        bank = SessionBank(per_session_max_bytes=1, cold_tier=cold)
        bank.cold_enqueue_dispatch = lambda job: None  # hold jobs forever
        entry = bank.put(
            runtime=FakeRuntime(),
            token_ids=[1, 2, 3, 4, 5],
            cache=[TrimmableKV()],
            logits=None,
            hidden=None,
            keep_live_ref=True,
            session_id="s1",
            snapshot_epoch=7,
        )
        assert entry is not None and entry.live_ref_only
        # A job captured at epoch 6 must refuse to persist epoch-7 state.
        assert bank.run_live_ref_spill((1, 2, 3, 4, 5), 6) is False
        assert cold.stats().get("spill_writes_completed", 0) == 0
        # The matching epoch runs.
        assert bank.run_live_ref_spill((1, 2, 3, 4, 5), 7) is True
    finally:
        cold.close()


def test_spill_respects_the_hourly_write_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_WRITE_BUDGET_PER_HOUR", "1")
    cold = make_cold(tmp_path)
    try:
        bank = SessionBank()
        entry = put_small_entry(bank)
        assert cold.spill_entry(entry, capabilities=["ar_insert"]) is False
        assert cold.stats()["skipped_write_budget"] >= 1
    finally:
        cold.close()


def test_interrupted_live_ref_spill_redispatches(tmp_path):
    cold = make_cold(tmp_path)
    try:
        jobs = []
        bank = SessionBank(per_session_max_bytes=1, cold_tier=cold)
        bank.cold_enqueue_dispatch = jobs.append
        entry = bank.put(
            runtime=FakeRuntime(),
            token_ids=[1, 2, 3, 4, 5],
            cache=[TrimmableKV()],
            logits=None,
            hidden=None,
            keep_live_ref=True,
            session_id="s1",
            snapshot_epoch=7,
        )
        assert entry is not None and len(jobs) == 1
        # A foreground request is active for the whole encode window: the
        # per-tensor abort fires and the job re-queues itself for the
        # next quiet window (coalesced by session).
        cold.foreground_busy = lambda: True
        assert jobs[0]() is False
        assert len(jobs) == 2
        assert jobs[1].coalesce_key == "ssd_cold:s1"
        assert cold.stats().get("spill_writes_completed", 0) == 0
        # Quiet window arrives: the retry persists.
        cold.foreground_busy = lambda: False
        assert jobs[1]() is True
        assert cold.stats()["spill_writes_completed"] == 1
    finally:
        cold.close()


def test_size_capped_spill_warns_once_per_session(tmp_path, caplog):
    """A spill refused for size must say so — once — instead of dying as a
    silent counter (2026-08-27 48G-sim finding: a near-full disk capped the
    lane at free/4 and every 100k+ session skipped with zero trace; the
    operator-visible symptom was an unexplained full re-prefill after
    restart, #278's silence class in a new lane)."""
    import logging

    cold = make_cold(tmp_path, max_bytes=1)  # effective cap ~1 byte
    try:
        bank = SessionBank()
        entry = put_small_entry(bank)
        assert entry is not None
        with caplog.at_level(logging.WARNING, logger="mtplx.cache_bank.cold_tier"):
            assert not cold.spill_entry(entry, capabilities=["ar_insert"])
            assert not cold.spill_entry(entry, capabilities=["ar_insert"])
        assert cold.stats()["skipped_size_cap"] == 2
        warnings = [
            record
            for record in caplog.records
            if "spill skipped" in record.getMessage()
        ]
        assert len(warnings) == 1, "one line per session, repeats stay counters"
        message = warnings[0].getMessage()
        assert "effective cap" in message and "re-prefill" in message
    finally:
        cold.close()
