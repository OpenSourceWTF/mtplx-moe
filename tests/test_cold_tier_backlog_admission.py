"""SSD writer admission: the backlog budget bounds queued bytes, not entries.

Issue #384 (sapiens77, 2026-08-28): a single snapshot larger than
``MTPLX_SSD_WRITER_BACKLOG_BYTES`` (default 4 GiB) could never satisfy
``pending + entry <= budget`` even with an empty queue, so at ~84 KB/token
of 27B KV the SSD tier was silently off past ~50k tokens — exactly the
sessions it exists to serve — with only a ``skipped_backlog_bytes`` counter
as the trace. Contract pinned here: an empty queue is not backlog pressure;
a lone oversized entry is admitted and counted by name, while a genuinely
backlogged writer still rejects.
"""

from __future__ import annotations

from mtplx.cache_bank import SessionBankColdTier


def _tier(tmp_path, budget_bytes: int) -> SessionBankColdTier:
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=64 * 1024**3,
        min_prefix_tokens=2,
    )
    cold._backlog_budget_bytes = int(budget_bytes)
    return cold


def test_lone_entry_over_budget_is_admitted_with_named_stat(tmp_path):
    cold = _tier(tmp_path, budget_bytes=4 * 1024**3)
    try:
        assert cold._pending_bytes == 0
        assert cold._admit_write(9 * 1024**3) is True
        assert int(cold._stats.get("admitted_oversized_alone", 0)) == 1
        assert int(cold._stats.get("skipped_backlog_bytes", 0)) == 0
        # The admitted bytes are still accounted as pending, so a SECOND
        # oversized entry behind it is real backlog and must be rejected.
        assert cold._admit_write(9 * 1024**3) is False
        assert int(cold._stats.get("skipped_backlog_bytes", 0)) == 1
    finally:
        cold.close()


def test_backlogged_writer_still_rejects_within_budget_entries(tmp_path):
    cold = _tier(tmp_path, budget_bytes=4 * 1024**3)
    try:
        assert cold._admit_write(3 * 1024**3) is True
        # 3 GiB pending + 2 GiB entry > 4 GiB budget: classic backlog.
        assert cold._admit_write(2 * 1024**3) is False
        assert int(cold._stats.get("skipped_backlog_bytes", 0)) == 1
        assert int(cold._stats.get("admitted_oversized_alone", 0)) == 0
        # Queue drains: the same entry is admitted.
        cold._release_pending(3 * 1024**3)
        assert cold._admit_write(2 * 1024**3) is True
    finally:
        cold.close()


def test_within_budget_admission_unchanged(tmp_path):
    cold = _tier(tmp_path, budget_bytes=4 * 1024**3)
    try:
        assert cold._admit_write(1 * 1024**3) is True
        assert cold._admit_write(2 * 1024**3) is True
        assert int(cold._stats.get("admitted_oversized_alone", 0)) == 0
        assert int(cold._stats.get("skipped_backlog_bytes", 0)) == 0
    finally:
        cold.close()
