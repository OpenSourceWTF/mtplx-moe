"""Unit tests for SessionBank cap env-var overrides wired in engine_session.

Covers the entry-count override (MTPLX_SESSION_BANK_MAX_ENTRIES) added on top
of the existing byte-cap envs (MTPLX_SESSION_BANK_MAX_BYTES,
MTPLX_SESSION_BANK_PER_SESSION_BYTES). The byte-cap helper is exercised in
detail in tests/test_engine_session_env.py; the regression cases here just
confirm all three envs compose correctly when wired through
EngineSessionManager.__init__.
"""
from __future__ import annotations

import importlib
import logging

import pytest


def _reload_engine_session():
    import mtplx.engine_session
    return importlib.reload(mtplx.engine_session)


# --- _bank_entries_from_env helper ----------------------------------------


def test_bank_entries_from_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_BANK_ENTRIES", raising=False)
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 8


def test_bank_entries_from_env_default_when_empty(monkeypatch):
    monkeypatch.setenv("TEST_BANK_ENTRIES", "")
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 8


def test_bank_entries_from_env_valid_integer(monkeypatch):
    monkeypatch.setenv("TEST_BANK_ENTRIES", "24")
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 24


def test_bank_entries_from_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("TEST_BANK_ENTRIES", "  16  ")
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 16


@pytest.mark.parametrize("raw", ["abc", "12.5", "1e2", "0x10", ""])
def test_bank_entries_from_env_invalid_falls_back(monkeypatch, caplog, raw):
    monkeypatch.setenv("TEST_BANK_ENTRIES", raw)
    es = _reload_engine_session()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mtplx.engine_session"):
        result = es._bank_entries_from_env("TEST_BANK_ENTRIES", 8)
    assert result == 8
    if raw == "":
        # Empty string is the unset-equivalent path; no warning expected.
        assert not any(
            "TEST_BANK_ENTRIES" in rec.getMessage() for rec in caplog.records
        )
    else:
        assert any(
            "TEST_BANK_ENTRIES" in rec.getMessage()
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), f"expected warning for raw={raw!r}, got {caplog.records!r}"


@pytest.mark.parametrize("raw", ["0", "-1", "-3"])
def test_bank_entries_from_env_below_one_falls_back(monkeypatch, caplog, raw):
    monkeypatch.setenv("TEST_BANK_ENTRIES", raw)
    es = _reload_engine_session()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mtplx.engine_session"):
        result = es._bank_entries_from_env("TEST_BANK_ENTRIES", 8)
    assert result == 8
    assert any(
        "TEST_BANK_ENTRIES" in rec.getMessage()
        and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"expected warning for raw={raw!r}, got {caplog.records!r}"


# --- EngineSessionManager wiring ------------------------------------------


def test_manager_uses_env_max_entries(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", "24")
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 24


def test_manager_default_max_entries_when_unset(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    monkeypatch.setattr(es.sys, "platform", "linux")
    mgr = es.EngineSessionManager()
    # Mirrors SessionBank's public default.
    assert mgr.bank.max_entries == 8


@pytest.mark.parametrize("raw", ["abc", "0", "-3"])
def test_manager_invalid_max_entries_falls_back_to_default(
    monkeypatch, caplog, raw
):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raw)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    monkeypatch.setattr(es.sys, "platform", "linux")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mtplx.engine_session"):
        mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 8
    assert any(
        "MTPLX_SESSION_BANK_MAX_ENTRIES" in rec.getMessage()
        and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_manager_byte_caps_still_work_alongside_entries(monkeypatch):
    """Regression: setting all three env vars composes correctly."""
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", "20")
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "32G")
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "16G")
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 20
    assert mgr.bank.max_bytes == 32 * 1024**3
    assert mgr.bank.per_session_max_bytes == 16 * 1024**3


def test_manager_byte_caps_alone_still_work(monkeypatch):
    """Regression: byte-cap envs without the new entries env still work
    and leave max_entries at the default."""
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "16G")
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "8G")
    es = _reload_engine_session()
    monkeypatch.setattr(es.sys, "platform", "linux")
    mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 8
    assert mgr.bank.max_bytes == 16 * 1024**3
    assert mgr.bank.per_session_max_bytes == 8 * 1024**3


def test_manager_quiesce_aborts_pending_postcommits_and_flushes_cold_tier(
    monkeypatch,
):
    """`/admin/cache/clear` quiesce: pending idle postcommits are aborted and
    the SSD deferred-encode queue is drained before state is dropped."""

    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()

    class FakeFuture:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True
            return True

    session = mgr.get_or_create("quiesce-test")
    session.set_pending_postcommit(FakeFuture(), reason="test", token_count=4)
    record = session._pending_postcommit
    assert record is not None

    flushes: list[float] = []
    monkeypatch.setattr(
        mgr, "flush_cold_tier", lambda *, timeout_s: flushes.append(timeout_s) or True
    )

    class FakeColdTier:
        def __init__(self):
            self.cancelled = 0

        def cancel_pending(self):
            self.cancelled += 3
            return 3

    fake_tier = FakeColdTier()
    monkeypatch.setattr(mgr.bank, "cold_tier", fake_tier, raising=False)

    outcome = mgr.quiesce(reason="admin_cache_clear")

    assert outcome["postcommits_aborted"] == 1
    assert outcome["ssd_writes_cancelled"] == 3
    assert outcome["cold_tier_flushed"] is True
    assert record.abort_event.is_set()
    assert record.last_abort_reason == "admin_cache_clear"
    assert record.future.cancelled is True
    assert flushes == [10.0]
    assert fake_tier.cancelled == 3


def test_manager_uses_24g_per_session_default_on_high_memory_darwin(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    monkeypatch.setattr(es.sys, "platform", "darwin")
    monkeypatch.setattr(
        es.subprocess,
        "check_output",
        lambda *args, **kwargs: str(128 * 1024**3),
    )

    mgr = es.EngineSessionManager()

    assert mgr.bank.max_entries == 16
    assert mgr.bank.max_bytes == 24 * 1024**3
    assert mgr.bank.per_session_max_bytes == 24 * 1024**3


def test_manager_keeps_8g_per_session_default_below_high_memory_threshold(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    monkeypatch.setattr(es.sys, "platform", "darwin")
    monkeypatch.setattr(
        es.subprocess,
        "check_output",
        lambda *args, **kwargs: str(64 * 1024**3),
    )

    mgr = es.EngineSessionManager()

    assert mgr.bank.max_entries == 8
    assert mgr.bank.per_session_max_bytes == 8 * 1024**3


def test_manager_per_session_env_override_wins_on_high_memory_darwin(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "12G")
    es = _reload_engine_session()
    monkeypatch.setattr(es.sys, "platform", "darwin")
    monkeypatch.setattr(
        es.subprocess,
        "check_output",
        lambda *args, **kwargs: str(128 * 1024**3),
    )

    mgr = es.EngineSessionManager()

    assert mgr.bank.per_session_max_bytes == 12 * 1024**3
