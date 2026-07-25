"""Issues #144/#145: the SSD cold tier under distinct-prefix churn.

Measured live 2026-07-09 (soak, Speed q4): the count-bounded writer queue
pinned ~30 GB of live KV payloads (active memory climbed 35 -> 66 GB while
the bank ledger stayed flat) and wrote 58 GB to disk in 45 minutes with
restore_hits=0. These tests pin the byte-bounded backlog and the rolling
hourly write budget that bound both.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


from mtplx.cache_bank.cold_tier import SessionBankColdTier


def make_tier(tmp_path, monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return SessionBankColdTier(base_dir=tmp_path / "ssd", mode="on")


def fake_entry(nbytes, tokens=2048):
    return SimpleNamespace(
        token_ids=tuple(range(tokens)),
        nbytes=nbytes,
        cache_snapshot=None,
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        gdn_boundaries=[],
        has_recurrent=False,
        session_id="s",
        token_hash="h",
        model_path="/m",
        mtp_enabled=False,
        hidden_variant=None,
        template_hash=None,
        mtp_history_policy=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=2048,
        mtp_snapshot_epoch=None,
    )


class TestBacklogByteCap:
    def test_admission_rejects_beyond_backlog_budget(self, tmp_path, monkeypatch):
        tier = make_tier(
            tmp_path, monkeypatch, MTPLX_SSD_WRITER_BACKLOG_BYTES="3G"
        )
        try:
            assert tier._admit_write(2 << 30)
            assert not tier._admit_write(2 << 30), (
                "second 2G write must exceed the 3G backlog budget"
            )
            assert tier.stats()["skipped_backlog_bytes"] == 1
            tier._release_pending(2 << 30)
            assert tier._admit_write(2 << 30), "released bytes free the budget"
        finally:
            tier.close()

    def test_backlog_bytes_exposed_in_stats(self, tmp_path, monkeypatch):
        tier = make_tier(
            tmp_path, monkeypatch, MTPLX_SSD_WRITER_BACKLOG_BYTES="8G"
        )
        try:
            assert tier._admit_write(1 << 30)
            stats = tier.stats()
            assert stats["writer_backlog_bytes"] == 1 << 30
            assert stats["writer_backlog_budget_bytes"] == 8 << 30
        finally:
            tier.close()


class TestHourlyWriteBudget:
    def test_budget_rejects_after_window_fills(self, tmp_path, monkeypatch):
        tier = make_tier(
            tmp_path,
            monkeypatch,
            MTPLX_SSD_WRITE_BUDGET_PER_HOUR="4G",
            MTPLX_SSD_WRITER_BACKLOG_BYTES="100G",
        )
        try:
            now = time.time()
            with tier._stats_lock:
                tier._written_window.append((now - 60, 3 << 30))
            assert not tier._admit_write(2 << 30), (
                "3G written this hour + 2G request must exceed the 4G budget"
            )
            assert tier.stats()["skipped_write_budget"] == 1
            # Old traffic outside the window no longer counts.
            with tier._stats_lock:
                tier._written_window.clear()
                tier._written_window.append((now - 3700, 3 << 30))
            assert tier._admit_write(2 << 30)
        finally:
            tier.close()

    def test_written_last_hour_exposed(self, tmp_path, monkeypatch):
        tier = make_tier(tmp_path, monkeypatch)
        try:
            with tier._stats_lock:
                tier._written_window.append((time.time(), 5 << 30))
            assert tier.stats()["written_bytes_last_hour"] == 5 << 30
        finally:
            tier.close()


class TestPutEntryIntegration:
    def test_oversized_backlog_skips_before_serializing(self, tmp_path, monkeypatch):
        tier = make_tier(
            tmp_path, monkeypatch, MTPLX_SSD_WRITER_BACKLOG_BYTES="1G"
        )
        try:
            assert tier.put_entry(fake_entry(nbytes=2 << 30)) is False
            assert tier.stats()["skipped_backlog_bytes"] == 1
            assert tier.stats()["writer_backlog_bytes"] == 0
        finally:
            tier.close()
