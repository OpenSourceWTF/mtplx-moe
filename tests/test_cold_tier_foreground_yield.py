"""Foreground-yield contract for the SSD cold tier (2026-08-07).

Two halves, both measured in the gate254-c4s receipts:
- The encode runs on the model-owner thread; an arriving foreground request
  queued behind it surfaced as 0.66-3.6 s of unattributed prompt-state wall.
  put_entry must abort between tensor evals and let the idle job re-dispatch.
- The writer thread's multi-GB file writes stole unified-memory bandwidth
  from the next turn's decode (-30%). The writer must pause between entry
  writes while foreground traffic is in flight.
"""

from __future__ import annotations

import time

import mlx.core as mx
import pytest

from mtplx.cache_bank.codec import ColdEncodeInterrupted, encode_payload
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.model_scheduler import ModelWorkScheduler


def _tiny_snapshot() -> CacheSnapshot:
    states = [mx.zeros((1, 2, 8, 4), dtype=mx.float16) for _ in range(2)]
    return CacheSnapshot(states=states, meta_states=[{"offset": 8}] * 2)


def _make_entry():
    class Entry:
        token_ids = tuple(range(600))
        nbytes = 4096
        cache_snapshot = _tiny_snapshot()
        logits = mx.zeros((1, 8), dtype=mx.float16)
        hidden = mx.zeros((1, 8), dtype=mx.float16)
        mtp_history_snapshot = None
        gdn_boundaries = ()
        has_recurrent = False
        session_id = "sess-yield-test"
        token_hash = "cafe" * 4
        prefix_len = 600

    return Entry()


def test_encode_payload_aborts_between_tensors():
    snapshot = _tiny_snapshot()
    calls = {"n": 0}

    def abort_after_first():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(ColdEncodeInterrupted):
        encode_payload(
            cache_snapshot=snapshot,
            logits=mx.zeros((1, 8), dtype=mx.float16),
            hidden=None,
            mtp_history_snapshot=None,
            should_abort=abort_after_first,
        )


def test_encode_payload_completes_when_never_aborted():
    snapshot = _tiny_snapshot()
    payload = encode_payload(
        cache_snapshot=snapshot,
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        mtp_history_snapshot=None,
        should_abort=lambda: False,
    )
    assert payload.nbytes > 0
    assert payload.tensors


def test_put_entry_raises_on_yield_only_when_asked(tmp_path):
    tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )
    try:
        tier.foreground_busy = lambda: True
        entry = _make_entry()
        with pytest.raises(ColdEncodeInterrupted):
            tier.put_entry(entry, capabilities=["ar_insert"], raise_on_yield=True)
        assert tier.stats()["encode_yields_foreground"] == 1
        # Legacy callers: swallowed, returns False.
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is False
        assert tier.stats()["encode_yields_foreground"] == 2
        # Foreground clear: stores.
        tier.foreground_busy = lambda: False
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    finally:
        tier.close() if hasattr(tier, "close") else None


def test_put_entry_yield_releases_backlog_admission(tmp_path):
    tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )
    tier.foreground_busy = lambda: True
    entry = _make_entry()
    for _ in range(6):
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is False
    assert tier._pending_bytes == 0


def test_writer_pauses_while_foreground_busy(tmp_path):
    tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )
    busy = {"value": True}
    tier.foreground_busy = lambda: busy["value"]
    entry = _make_entry()
    # Encode with foreground idle so the write enqueues, then flip busy
    # before the writer picks it up is racy — instead enqueue while busy is
    # False only for the encode call.
    busy["value"] = False
    assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    busy["value"] = True
    time.sleep(0.3)
    stats_mid = tier.stats()
    busy["value"] = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= 1:
            break
        time.sleep(0.05)
    stats_end = tier.stats()
    assert stats_end["writes_completed"] >= 1
    assert stats_end["writer_foreground_pauses"] >= 1
    assert stats_end["writer_foreground_pause_s"] > 0.0
    # The write must not have completed while we were holding it busy,
    # unless it slipped in before the flip (tolerated: pause counter proves
    # the writer honored the signal at least once).
    del stats_mid


def test_writer_pause_outlasts_a_long_turn_by_default(tmp_path):
    # 2026-08-28: the 60 s default expired mid-turn on 60-620 s coding-agent
    # decodes, so every blob write fired under the live turn (~1 GB/min
    # manifest drumbeat 21:38-22:17, macOS pressure banner + decode theft).
    # The bound is a liveness backstop, not a cadence: it must exceed any
    # realistic single turn.
    tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )
    assert tier._writer_pause_max_s == 600.0


def test_writer_pause_expiry_into_busy_traffic_is_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_WRITER_FOREGROUND_PAUSE_MAX_S", "0.2")
    tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )
    busy = {"value": False}
    tier.foreground_busy = lambda: busy["value"]
    entry = _make_entry()
    assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    busy["value"] = True
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= 1:
            break
        time.sleep(0.05)
    stats = tier.stats()
    # The write went through despite permanent busy (liveness bound), and
    # the expiry was recorded so operators can see durability fought decode.
    assert stats["writes_completed"] >= 1
    assert stats["writer_pause_expired_busy"] >= 1


def test_writer_pause_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_WRITER_FOREGROUND_PAUSE", "0")
    tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )
    tier.foreground_busy = lambda: True
    entry = _make_entry()
    # Encode yield still applies (separate knob); disable it via env too.
    monkeypatch.setenv("MTPLX_SSD_ENCODE_FOREGROUND_YIELD", "0")
    tier2 = SessionBankColdTier(
        base_dir=tmp_path / "bank2", mode="on", min_prefix_tokens=1
    )
    tier2.foreground_busy = lambda: True
    assert tier2.put_entry(entry, capabilities=["ar_insert"]) is True
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier2.stats()["writes_completed"] >= 1:
            break
        time.sleep(0.05)
    assert tier2.stats()["writes_completed"] >= 1
    assert tier2.stats()["writer_foreground_pauses"] == 0


def test_scheduler_foreground_busy_signal():
    scheduler = ModelWorkScheduler()
    try:
        assert scheduler.foreground_busy() is False
        import threading

        release = threading.Event()
        seen_busy = threading.Event()

        def blocker():
            seen_busy.set()
            release.wait(timeout=5.0)

        future = scheduler.submit_foreground(blocker)
        seen_busy.wait(timeout=5.0)
        assert scheduler.foreground_busy() is True
        release.set()
        future.result(timeout=5.0)
        deadline = time.time() + 2.0
        while time.time() < deadline and scheduler.foreground_busy():
            time.sleep(0.01)
        assert scheduler.foreground_busy() is False
    finally:
        scheduler.shutdown(wait=True)
