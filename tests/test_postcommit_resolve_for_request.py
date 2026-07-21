"""Tests for the abort-don't-wait foreground policy.

POSTCOMMIT_STALL_DESIGN step 2 (2026-07-17): a live user request never
queues behind cache maintenance. EngineSession.resolve_pending_postcommit_
for_request() replaces the unconditional bounded wait in the server request
path: a pending postcommit is aborted immediately (reason
"foreground_preempted_postcommit", same telemetry vocabulary as the old
timeout path) and the request admits without dead air. Setting
MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S explicitly restores the old bounded wait.

Invariants exercised here:
  - NO-PENDING : short-circuits to "no_pending" without blocking
  - ABORT      : a still-running pending job is aborted, the reference is
                 cleared, and the call returns in well under the old bound
  - LANDED     : an already-resolved future reports "completed" and clears
  - ENV        : an explicit MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S falls back to
                 the legacy bounded wait (including its timeout outcome)
"""

from __future__ import annotations

import time
from concurrent.futures import Future

from mtplx.engine_session import EngineSession


def _new_session(sid: str = "sess-resolve") -> EngineSession:
    return EngineSession(sid)


def test_resolve_no_pending_short_circuits(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    session = _new_session()
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    assert time.monotonic() - t0 < 0.5
    assert outcome["outcome"] == "no_pending"
    assert outcome["waited"] is False
    assert session.last_postcommit_wait is outcome


def test_resolve_aborts_running_pending_without_blocking(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    session = _new_session()
    future: Future = Future()  # never resolves: simulates an in-flight job
    future.set_running_or_notify_cancel()  # started: cancel() must fail
    record = session.set_pending_postcommit(
        future, reason="tool_call_history_rewrite", token_count=20_000
    )
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"abort path must not block (took {elapsed:.3f}s)"
    assert outcome["outcome"] == "aborted_for_foreground"
    assert outcome["waited"] is False
    assert outcome["abort_requested"] is True
    assert outcome["abort_reason"] == "foreground_preempted_postcommit"
    # The running job observes the abort through its abort_event; the
    # pending reference is cleared so the next turn starts clean.
    assert record.abort_event.is_set()
    assert session.pending_postcommit is None
    assert session.has_pending_postcommit() is False


def test_resolve_cancels_not_yet_started_pending(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    session = _new_session()
    future: Future = Future()  # queued but never started: cancellable
    session.set_pending_postcommit(future, reason="postcommit")
    outcome = session.resolve_pending_postcommit_for_request()
    assert outcome["outcome"] == "aborted_for_foreground"
    assert outcome["future_cancelled"] is True
    assert future.cancelled()
    assert session.pending_postcommit is None


def test_resolve_landed_pending_reports_completed(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    session = _new_session()
    future: Future = Future()
    future.set_result({"stored": True})
    session.set_pending_postcommit(future, reason="postcommit")
    outcome = session.resolve_pending_postcommit_for_request()
    assert outcome["outcome"] == "completed"
    assert outcome["waited"] is False
    assert session.pending_postcommit is None


def test_resolve_env_override_restores_bounded_wait(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", "0.2")
    session = _new_session()
    future: Future = Future()
    future.set_running_or_notify_cancel()
    session.set_pending_postcommit(future, reason="postcommit")
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    elapsed = time.monotonic() - t0
    assert outcome["waited"] is True
    assert outcome["outcome"] == "timeout"
    assert 0.15 <= elapsed < 2.0, f"expected the legacy bounded wait, got {elapsed:.3f}s"
