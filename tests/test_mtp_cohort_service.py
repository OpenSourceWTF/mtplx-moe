from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mtplx.mtp_k2_stepper import (
    MTPK2AcceptanceContext,
    MTPK2ContextCopyResult,
    MTPK2ContextCopyTicket,
    MTPK2PrefillResult,
    MTPK2PrefillTicket,
    MTPK2VerifyResult,
    MTPK2VerifyTicket,
)
from mtplx.server.mtp_cohort import (
    MTPK2CohortCancelled,
    MTPK2CohortGenerationService,
    MTPK2CohortJob,
)


class _Scheduler:
    def __init__(self) -> None:
        self.queued: list[tuple[Any, str | None, Any]] = []
        self.current_batch_key: str | None = None

    def submit_foreground(self, fn, *, batch_key=None):
        future = SimpleNamespace()
        self.queued.append((fn, batch_key, future))
        return future

    def run_next(self) -> None:
        fn, batch_key, _future = self.queued.pop(0)
        self.current_batch_key = batch_key
        try:
            fn()
        finally:
            self.current_batch_key = None


class _CancelEvent:
    def __init__(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value

    def set(self) -> None:
        self.value = True


@dataclass
class _FinalState:
    final_trunk_cache: list[Any]
    safe_to_commit: bool = True


@dataclass
class _Output:
    request_id: str
    final_state: _FinalState


class _RequestState:
    def __init__(
        self,
        request_id: str,
        tickets: list[Any],
        *,
        cancel_event: _CancelEvent | None = None,
    ) -> None:
        self.request_id = request_id
        self._tickets = list(tickets)
        self._index = -1
        self.pending_ticket = None
        self.target_cache: list[Any] = []
        self.status = "created"
        self.output = None
        self.error = None
        self.cancel_event = cancel_event

    def _cancel_requested(self) -> bool:
        return bool(self.cancel_event and self.cancel_event.is_set())

    def _publish(self):
        self._index += 1
        if self._index >= len(self._tickets):
            self.pending_ticket = None
            self.status = "finished"
            self.output = _Output(
                self.request_id,
                _FinalState([f"final-{self.request_id}"]),
            )
            return None
        self.pending_ticket = self._tickets[self._index]
        self.status = "ready"
        return self.pending_ticket

    def start(self):
        assert self.status == "created"
        if self._cancel_requested():
            self.close(status="cancelled")
            return None
        return self._publish()

    def require_ticket(self):
        assert self.pending_ticket is not None
        return self.pending_ticket

    def execute_pending(self, executor):
        if self._cancel_requested():
            self.close(status="cancelled")
            raise MTPK2CohortCancelled(self.request_id)
        return executor(self.require_ticket())

    def resume(self, result):
        if self._cancel_requested():
            self.close(status="cancelled")
            return None
        self.target_cache = result.request_cache
        return self._publish()

    def fail(self, exc):
        self.error = exc
        self.pending_ticket = None
        self.status = "failed"
        raise exc

    def close(self, *, status="closed"):
        self.pending_ticket = None
        self.status = status


def _prefill(request_id: str, start: int, stop: int) -> MTPK2PrefillTicket:
    return MTPK2PrefillTicket(
        request_id=request_id,
        input_ids=np.zeros((1, stop - start), dtype=np.int32),
        request_cache=[f"cache-{request_id}"],
        prompt_start=start,
        prompt_stop=stop,
    )


def _verify(request_id: str, cache: list[Any] | None = None) -> MTPK2VerifyTicket:
    return MTPK2VerifyTicket(
        request_id=request_id,
        input_ids=np.zeros((1, 3), dtype=np.int32),
        request_cache=list(cache or [f"cache-{request_id}"]),
        draft_distributions=(object(), object()),
        acceptance_context=MTPK2AcceptanceContext(
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
            hidden_variant="post_norm",
        ),
    )


def _context_copy(
    request_id: str,
    cache: list[Any] | None = None,
) -> MTPK2ContextCopyTicket:
    return MTPK2ContextCopyTicket(
        request_id=request_id,
        input_ids=np.zeros((1, 5), dtype=np.int32),
        request_cache=list(cache or [f"cache-{request_id}"]),
        hidden_variant="post_norm",
        capture_backend="linear_gdn_from_conv_tape",
    )


class _Runner:
    def __init__(self) -> None:
        self.widths: list[int] = []
        self.on_step = None
        self.after_step = None
        self.failure: BaseException | None = None
        self.events: list[str] | None = None

    def step(self, states):
        self.widths.append(len(states))
        if self.events is not None:
            self.events.append(f"verify-{len(states)}")
        if self.on_step is not None:
            self.on_step(states)
        if self.failure is not None:
            raise self.failure
        results = []
        for state in states:
            if state._cancel_requested():
                state.close(status="cancelled")
                continue
            ticket = state.require_ticket()
            results.append(
                MTPK2VerifyResult(
                    logits=f"logits-{state.request_id}",
                    hidden=f"hidden-{state.request_id}",
                    captures={"request_id": state.request_id},
                    request_cache=[f"owned-{state.request_id}"],
                    commit_prefix=lambda _steps, cache=state.request_id: [cache],
                    forward_elapsed_s=0.25,
                )
            )
            assert isinstance(ticket, MTPK2VerifyTicket)
        if self.after_step is not None:
            self.after_step(states)
        return tuple(results)


@dataclass
class _Harness:
    service: MTPK2CohortGenerationService
    scheduler: _Scheduler
    runner: _Runner
    states: dict[str, _RequestState]
    make_calls: list[tuple[str, list[int], dict[str, Any]]]
    prefill_order: list[str]
    context_copy_order: list[str]
    normalize_calls: list[str]
    commit_checks: list[list[Any]]
    events: list[str]


def _harness(
    ticket_map: dict[str, list[Any]],
    *,
    cancel_events: dict[str, _CancelEvent] | None = None,
) -> _Harness:
    scheduler = _Scheduler()
    runner = _Runner()
    states: dict[str, _RequestState] = {}
    make_calls: list[tuple[str, list[int], dict[str, Any]]] = []
    prefill_order: list[str] = []
    context_copy_order: list[str] = []
    normalize_calls: list[str] = []
    commit_checks: list[list[Any]] = []
    events: list[str] = []
    runner.events = events

    def make_state(_runtime, prompt_ids, *, request_id, **kwargs):
        make_calls.append((request_id, list(prompt_ids), dict(kwargs)))
        state = _RequestState(
            request_id,
            ticket_map[request_id],
            cancel_event=(cancel_events or {}).get(request_id),
        )
        states[request_id] = state
        return state

    def execute_prefill(ticket):
        prefill_order.append(ticket.request_id)
        events.append(f"prefill-{ticket.request_id}")
        return MTPK2PrefillResult(
            logits=f"logits-{ticket.request_id}",
            hidden=f"hidden-{ticket.request_id}",
            request_cache=ticket.request_cache,
        )

    def execute_context_copy(ticket):
        context_copy_order.append(ticket.request_id)
        events.append(f"context-copy-{ticket.request_id}")
        return MTPK2ContextCopyResult(
            logits=f"copy-logits-{ticket.request_id}",
            hidden=f"copy-hidden-{ticket.request_id}",
            captures={"request_id": ticket.request_id},
            request_cache=ticket.request_cache,
        )

    def normalize(_lane, cache):
        request_id = str(cache[0]).removeprefix("cache-")
        normalize_calls.append(request_id)
        return [f"normalized-{request_id}"]

    def assert_local(_lane, cache):
        commit_checks.append(cache)

    state = SimpleNamespace(
        runtime=object(),
        model_scheduler=scheduler,
    )
    dependencies = SimpleNamespace(
        runner=runner,
        make_state=make_state,
        execute_context_copy=execute_context_copy,
        execute_prefill=execute_prefill,
        normalize_cache=normalize,
        assert_request_cache=assert_local,
    )
    service = MTPK2CohortGenerationService(
        state,
        lane=object(),
        dependencies=dependencies,
    )
    return _Harness(
        service=service,
        scheduler=scheduler,
        runner=runner,
        states=states,
        make_calls=make_calls,
        prefill_order=prefill_order,
        context_copy_order=context_copy_order,
        normalize_calls=normalize_calls,
        commit_checks=commit_checks,
        events=events,
    )


def _job(request_id: str, **state_kwargs: Any) -> MTPK2CohortJob:
    return MTPK2CohortJob(
        request_id=request_id,
        prompt_ids=[1, 2, 3],
        state_kwargs=state_kwargs,
    )


def test_submit_schedules_one_foreground_pump_and_preserves_state_inputs() -> None:
    harness = _harness(
        {
            "first": [
                _prefill("first", 0, 3),
                _verify("first"),
                _verify("first"),
            ],
            "second": [
                _prefill("second", 0, 3),
                _verify("second"),
                _verify("second"),
            ],
        }
    )
    first = _job(
        "first",
        constraint="constraint-a",
        session_id="session-a",
        stop_token_ids={7},
        seed=11,
        sampler="penalty-sampler-a",
        token_callback="callback-a",
    )
    second = _job("second", seed=22, sampler="penalty-sampler-b")

    first_future = harness.service.submit(first)
    second_future = harness.service.submit(second)

    assert len(harness.scheduler.queued) == 1
    assert harness.scheduler.queued[0][1] == "mtp_cohort.pump"
    harness.scheduler.run_next()
    assert first_future.result().request_id == "first"
    assert second_future.result().request_id == "second"
    assert harness.runner.widths == [1, 2, 1]
    assert harness.events == [
        "prefill-first",
        "verify-1",
        "prefill-second",
        "verify-2",
        "verify-1",
    ]
    assert harness.make_calls[0][2]["constraint"] == "constraint-a"
    assert harness.make_calls[0][2]["session_id"] == "session-a"
    assert harness.make_calls[0][2]["stop_token_ids"] == {7}
    assert harness.make_calls[0][2]["seed"] == 11
    assert harness.make_calls[0][2]["sampler"] == "penalty-sampler-a"
    assert harness.make_calls[0][2]["token_callback"] == "callback-a"


def test_job_freezes_environment_before_owner_admission() -> None:
    harness = _harness(
        {"frozen": [_prefill("frozen", 0, 3), _verify("frozen")]}
    )
    environment = {
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_PREFILL_CHUNK_SIZE": "1024",
    }
    job = MTPK2CohortJob(
        request_id="frozen",
        prompt_ids=[1, 2, 3],
        state_kwargs={"seed": 7},
        environment=environment,
    )
    environment["MTPLX_SUSTAINED_PREFILL"] = "0"

    with pytest.raises(TypeError):
        job.environment["MTPLX_SUSTAINED_PREFILL"] = "0"

    harness.service.submit(job)
    harness.scheduler.run_next()

    assert job.future.result().request_id == "frozen"
    assert harness.make_calls[0][2]["environment"] == {
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_PREFILL_CHUNK_SIZE": "1024",
    }


def test_owner_finalize_runs_before_future_publication_and_maps_result() -> None:
    harness = _harness(
        {"finalized": [_prefill("finalized", 0, 3), _verify("finalized")]}
    )
    events: list[str] = []
    job: MTPK2CohortJob

    def owner_finalize(output):
        assert not job.future.done()
        assert output.request_id == "finalized"
        assert harness.scheduler.current_batch_key == "mtp_cohort.pump"
        events.append("owner-finalize")
        return {"public": output.request_id}

    job = MTPK2CohortJob(
        request_id="finalized",
        prompt_ids=[1, 2, 3],
        state_kwargs={},
        owner_finalize=owner_finalize,
    )
    job.future.add_done_callback(lambda _future: events.append("future-done"))

    harness.service.submit(job)
    harness.scheduler.run_next()

    assert job.future.result() == {"public": "finalized"}
    assert events == ["owner-finalize", "future-done"]
    assert harness.commit_checks == [["final-finalized"]]


def test_owner_finalize_commits_request_local_cache_before_surviving_peer() -> None:
    harness = _harness(
        {
            "short": [_prefill("short", 0, 3), _verify("short")],
            "long": [
                _prefill("long", 0, 3),
                _verify("long"),
                _verify("long"),
            ],
        }
    )
    events = harness.events

    def finalize_short(output):
        events.append(f"commit-{output.final_state.final_trunk_cache[0]}")
        return output

    short = MTPK2CohortJob(
        request_id="short",
        prompt_ids=[1, 2, 3],
        state_kwargs={},
        owner_finalize=finalize_short,
    )
    long = _job("long")

    harness.service.submit(short)
    harness.service.submit(long)
    harness.scheduler.run_next()

    assert short.future.result().request_id == "short"
    assert long.future.result().request_id == "long"
    assert events == [
        "prefill-short",
        "verify-1",
        "commit-final-short",
        "prefill-long",
        "verify-1",
        "verify-1",
    ]


def test_owner_finalize_failure_is_isolated_from_surviving_peer() -> None:
    harness = _harness(
        {
            "failed": [_prefill("failed", 0, 3), _verify("failed")],
            "survivor": [_prefill("survivor", 0, 3), _verify("survivor")],
        }
    )

    def fail_finalize(_output):
        raise RuntimeError("owner finalization failed")

    job = MTPK2CohortJob(
        request_id="failed",
        prompt_ids=[1, 2, 3],
        state_kwargs={},
        owner_finalize=fail_finalize,
    )
    survivor = _job("survivor")

    harness.service.submit(job)
    harness.service.submit(survivor)
    harness.scheduler.run_next()

    with pytest.raises(RuntimeError, match="owner finalization failed"):
        job.future.result()
    assert harness.states["failed"].status == "failed"
    assert survivor.future.result().request_id == "survivor"


def test_lone_job_executes_width1_without_cohort_cache_normalization() -> None:
    harness = _harness(
        {"solo": [_prefill("solo", 0, 3), _verify("solo")]}
    )

    future = harness.service.submit(_job("solo"))
    harness.scheduler.run_next()

    assert future.result().request_id == "solo"
    assert harness.runner.widths == [1]
    assert harness.normalize_calls == []
    assert harness.commit_checks == [["final-solo"]]


def test_context_copy_phase_executes_stock_width1_before_resuming_k2_verify() -> None:
    harness = _harness(
        {
            "copy": [
                _prefill("copy", 0, 3),
                _context_copy("copy"),
                _verify("copy"),
            ],
            "peer": [
                _prefill("peer", 0, 3),
                _verify("peer"),
                _verify("peer"),
            ],
        }
    )

    copy = harness.service.submit(_job("copy"))
    peer = harness.service.submit(_job("peer"))
    harness.scheduler.run_next()

    assert copy.result().request_id == "copy"
    assert peer.result().request_id == "peer"
    assert harness.context_copy_order == ["copy"]
    assert "context-copy-copy" in harness.events
    assert harness.runner.widths[-1] == 1


def test_newly_ready_decode_runs_before_peer_prefill_then_peer_advances() -> None:
    harness = _harness(
        {
            "short": [
                _prefill("short", 0, 8),
                _verify("short"),
                _verify("short"),
            ],
            "long": [
                _prefill("long", 0, 1024),
                _prefill("long", 1024, 2048),
                _verify("long"),
            ],
        }
    )

    short = _job("short")
    long = _job("long")
    harness.service.submit(short)
    harness.service.submit(long)
    harness.scheduler.run_next()

    assert short.future.result().request_id == "short"
    assert long.future.result().request_id == "long"
    assert harness.events == [
        "prefill-short",
        "verify-1",
        "prefill-long",
        "verify-1",
        "prefill-long",
        "verify-1",
    ]


def test_prefill_is_chunk_bounded_fair_and_decode_precedes_next_chunk() -> None:
    harness = _harness(
        {
            "long": [
                _prefill("long", 0, 1024),
                _prefill("long", 1024, 2048),
                _verify("long"),
                _verify("long"),
            ],
            "short": [
                _prefill("short", 0, 8),
                _verify("short"),
            ],
        }
    )

    long_future = harness.service.submit(_job("long"))
    short_future = harness.service.submit(_job("short"))
    harness.scheduler.run_next()

    assert long_future.result().request_id == "long"
    assert short_future.result().request_id == "short"
    assert harness.prefill_order[:3] == ["long", "short", "long"]
    assert harness.runner.widths == [1, 1, 1]
    assert harness.events == [
        "prefill-long",
        "prefill-short",
        "verify-1",
        "prefill-long",
        "verify-1",
        "verify-1",
    ]
    assert max(
        ticket.prompt_stop - ticket.prompt_start
        for tickets in (
            [_prefill("long", 0, 1024), _prefill("long", 1024, 2048)],
        )
        for ticket in tickets
    ) == 1024


def test_pending_job_joins_only_after_the_current_target_cycle() -> None:
    harness = _harness(
        {
            "first": [_prefill("first", 0, 3), _verify("first"), _verify("first")],
            "second": [
                _prefill("second", 0, 3),
                _verify("second"),
                _verify("second"),
            ],
        }
    )
    second = _job("second")
    submitted = [False]

    def submit_during_first_cycle(_states):
        if not submitted[0]:
            submitted[0] = True
            harness.service.submit(second)

    harness.runner.on_step = submit_during_first_cycle
    first_future = harness.service.submit(_job("first"))
    harness.scheduler.run_next()

    assert first_future.result().request_id == "first"
    assert second.future.result().request_id == "second"
    assert harness.runner.widths == [1, 2, 1]
    assert harness.events == [
        "prefill-first",
        "verify-1",
        "prefill-second",
        "verify-2",
        "verify-1",
    ]
    assert harness.normalize_calls.count("first") == 1
    assert harness.normalize_calls.count("second") == 1
    assert len(harness.scheduler.queued) == 0


def test_prefill_over_frozen_chunk_budget_fails_before_executor() -> None:
    harness = _harness(
        {"oversized": [_prefill("oversized", 0, 1025)]}
    )
    job = _job("oversized")

    harness.service.submit(job)
    harness.scheduler.run_next()

    with pytest.raises(RuntimeError, match="exceeds frozen 1024-token budget"):
        job.future.result()
    assert harness.prefill_order == []


def test_cancelled_pending_job_never_reaches_owner_finalizer() -> None:
    cancel = _CancelEvent()
    cancel.set()
    harness = _harness(
        {"cancelled": [_prefill("cancelled", 0, 3)]},
        cancel_events={"cancelled": cancel},
    )
    job = MTPK2CohortJob(
        request_id="cancelled",
        prompt_ids=[1, 2, 3],
        state_kwargs={},
        owner_finalize=lambda _output: pytest.fail(
            "cancelled pending job reached owner finalization"
        ),
    )

    harness.service.submit(job)
    harness.scheduler.run_next()

    with pytest.raises(MTPK2CohortCancelled):
        job.future.result()
    assert harness.states["cancelled"].status == "cancelled"


@pytest.mark.parametrize("submission_mode", ["failed", "cancelled"])
def test_pump_submission_failure_completes_pending_job(submission_mode) -> None:
    pump_future = Future()
    if submission_mode == "failed":
        pump_future.set_exception(RuntimeError("scheduler rejected pump"))
    else:
        pump_future.cancel()

    class Scheduler:
        def submit_foreground(self, _fn, *, batch_key=None):
            assert batch_key == "mtp_cohort.pump"
            return pump_future

    state = SimpleNamespace(runtime=object(), model_scheduler=Scheduler())
    dependencies = SimpleNamespace(
        runner=pytest.fail,
        make_state=pytest.fail,
        execute_context_copy=pytest.fail,
        execute_prefill=pytest.fail,
        normalize_cache=pytest.fail,
        assert_request_cache=pytest.fail,
    )
    service = MTPK2CohortGenerationService(
        state,
        lane=object(),
        dependencies=dependencies,
    )
    job = _job("pending")

    service.submit(job)

    expected = (
        "scheduler rejected pump"
        if submission_mode == "failed"
        else "cancelled before execution"
    )
    with pytest.raises(RuntimeError, match=expected):
        job.future.result()
    assert service.snapshot()["pending"] == 0
    assert service.snapshot()["pump_scheduled"] is False


def test_cancellation_during_shared_forward_discards_only_cancelled_row() -> None:
    first_cancel = _CancelEvent()
    harness = _harness(
        {
            "first": [
                _prefill("first", 0, 3),
                _verify("first"),
                _verify("first"),
            ],
            "second": [_prefill("second", 0, 3), _verify("second")],
        },
        cancel_events={"first": first_cancel},
    )

    def cancel_first(states):
        if len(states) == 2:
            first_cancel.set()

    harness.runner.on_step = cancel_first
    first = _job("first")
    second = _job("second")
    harness.service.submit(first)
    harness.service.submit(second)
    harness.scheduler.run_next()

    with pytest.raises(MTPK2CohortCancelled):
        first.future.result()
    assert second.future.result().request_id == "second"
    assert harness.states["first"].status == "cancelled"


def test_cancellation_after_runner_filter_keeps_result_alignment() -> None:
    first_cancel = _CancelEvent()
    harness = _harness(
        {
            "first": [
                _prefill("first", 0, 3),
                _verify("first"),
                _verify("first"),
            ],
            "second": [_prefill("second", 0, 3), _verify("second")],
        },
        cancel_events={"first": first_cancel},
    )

    def cancel_after_filter(states):
        if len(states) == 2:
            first_cancel.set()

    harness.runner.after_step = cancel_after_filter
    first = _job("first")
    second = _job("second")
    harness.service.submit(first)
    harness.service.submit(second)
    harness.scheduler.run_next()

    with pytest.raises(MTPK2CohortCancelled):
        first.future.result()
    assert second.future.result().request_id == "second"
    assert harness.states["first"].status == "cancelled"
    assert harness.states["second"].status == "finished"
    assert harness.runner.widths == [1, 2]


def test_shared_target_exception_fails_both_without_width1_retry() -> None:
    harness = _harness(
        {
            "first": [
                _prefill("first", 0, 3),
                _verify("first"),
                _verify("first"),
            ],
            "second": [_prefill("second", 0, 3), _verify("second")],
        }
    )

    def fail_width2(states):
        if len(states) == 2:
            harness.runner.failure = RuntimeError("shared target failed")

    harness.runner.on_step = fail_width2
    first = _job("first")
    second = _job("second")
    harness.service.submit(first)
    harness.service.submit(second)
    harness.scheduler.run_next()

    for future in (first.future, second.future):
        with pytest.raises(RuntimeError, match="shared target failed"):
            future.result()
    assert harness.runner.widths == [1, 2]


def test_snapshot_is_bounded_state_not_dispatch_telemetry() -> None:
    harness = _harness(
        {"solo": [_prefill("solo", 0, 3), _verify("solo")]}
    )
    observed = []

    def capture_snapshot(_states):
        observed.append(harness.service.snapshot())

    harness.runner.on_step = capture_snapshot
    harness.service.submit(_job("solo"))
    before = harness.service.snapshot()
    harness.scheduler.run_next()
    after = harness.service.snapshot()

    assert before["pending_request_ids"] == ["solo"]
    assert observed[0]["active_request_ids"] == ["solo"]
    assert observed[0]["active_width"] == 1
    assert after == {
        "pending": 0,
        "active": 0,
        "pending_request_ids": [],
        "active_request_ids": [],
        "active_width": 0,
        "pump_scheduled": False,
        "last_error": None,
    }
    assert not any("cycle" in key or "dispatch" in key for key in after)


def test_session_bank_proxy_rejects_cohort_cache_before_put() -> None:
    scheduler = _Scheduler()
    put_calls: list[list[Any]] = []

    class Bank:
        def put(self, *, cache, **_kwargs):
            put_calls.append(cache)

    def make_state(_runtime, _prompt_ids, *, session_bank, **_kwargs):
        session_bank.put(cache=["cohort-cache"])
        raise AssertionError("guard allowed cohort cache into session bank")

    def assert_request_cache(_lane, cache):
        if cache == ["cohort-cache"]:
            raise TypeError("session commit received cohort cache")

    service = MTPK2CohortGenerationService(
        SimpleNamespace(runtime=object(), model_scheduler=scheduler),
        lane=object(),
        dependencies=SimpleNamespace(
            runner=_Runner(),
            make_state=make_state,
            execute_context_copy=lambda _ticket: None,
            execute_prefill=lambda _ticket: None,
            normalize_cache=lambda _lane, cache: cache,
            assert_request_cache=assert_request_cache,
        ),
    )
    job = _job("guarded", session_bank=Bank())

    service.submit(job)
    scheduler.run_next()

    with pytest.raises(TypeError, match="session commit received cohort cache"):
        job.future.result()
    assert put_calls == []
