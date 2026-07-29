"""Single-owner scheduler for one or two resumable depth-two MTP requests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from threading import Condition
from types import MappingProxyType, SimpleNamespace
from typing import Any

from mtplx.mtp_k2_stepper import (
    MTPK2ContextCopyTicket,
    MTPK2PrefillTicket,
    MTPK2VerifyTicket,
)


class MTPK2CohortCancelled(RuntimeError):
    """A cohort request was cancelled before its next request-local resume."""


@dataclass
class MTPK2CohortJob:
    request_id: str
    prompt_ids: list[int]
    state_kwargs: Mapping[str, Any]
    environment: Mapping[str, str] = field(default_factory=dict)
    owner_finalize: Callable[[Any], Any] | None = None
    future: Future = field(default_factory=Future)

    def __post_init__(self) -> None:
        self.request_id = str(self.request_id)
        self.prompt_ids = [int(token) for token in self.prompt_ids]
        self.state_kwargs = MappingProxyType(dict(self.state_kwargs))
        self.environment = MappingProxyType(
            {
                str(name): str(value)
                for name, value in self.environment.items()
            }
        )
        if self.owner_finalize is not None and not callable(self.owner_finalize):
            raise TypeError("owner_finalize must be callable or None")


@dataclass
class _ActiveJob:
    job: MTPK2CohortJob
    state: Any
    decode_cache_normalized: bool = False


class _RequestLocalSessionBank:
    """Transparent bank proxy that guards every request cache publication."""

    def __init__(self, bank: Any, guard: Any) -> None:
        self._bank = bank
        self._guard = guard

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bank, name)

    def put(self, *args: Any, **kwargs: Any) -> Any:
        self._guard(kwargs["cache"])
        return self._bank.put(*args, **kwargs)


def _service_dependencies(server_state: Any, lane: Any) -> SimpleNamespace:
    from mtplx.generation import (
        execute_solo_mtpk2_context_copy_ticket,
        execute_solo_mtpk2_prefill_ticket,
        make_mtpk2_request_state_from_environment,
    )
    from mtplx.qwen27b_mtp_cohort import (
        MTPK2CohortRunner,
        assert_request_local_target_cache,
        normalize_target_cache,
    )

    return SimpleNamespace(
        runner=MTPK2CohortRunner(lane),
        make_state=make_mtpk2_request_state_from_environment,
        execute_context_copy=lambda ticket: (
            execute_solo_mtpk2_context_copy_ticket(
                server_state.runtime,
                ticket,
            )
        ),
        execute_prefill=lambda ticket: execute_solo_mtpk2_prefill_ticket(
            server_state.runtime,
            ticket,
        ),
        normalize_cache=normalize_target_cache,
        assert_request_cache=assert_request_local_target_cache,
    )


class MTPK2CohortGenerationService:
    """Own request machines and multiplex their fixed K2 target cycles."""

    def __init__(
        self,
        server_state: Any,
        lane: Any,
        *,
        dependencies: Any | None = None,
    ) -> None:
        deps = (
            _service_dependencies(server_state, lane)
            if dependencies is None
            else dependencies
        )
        self._server_state = server_state
        self._lane = lane
        self._runner = deps.runner
        self._make_state = deps.make_state
        self._execute_context_copy = deps.execute_context_copy
        self._execute_prefill = deps.execute_prefill
        self._normalize_cache = deps.normalize_cache
        self._assert_request_cache = deps.assert_request_cache
        self._condition = Condition()
        self._pending: deque[MTPK2CohortJob] = deque()
        self._active: list[_ActiveJob] = []
        self._pump_scheduled = False
        self._active_width = 0
        self._last_error: str | None = None
        self._prefill_cursor = 0

    def submit(self, job: MTPK2CohortJob) -> Future:
        schedule_pump = False
        with self._condition:
            self._pending.append(job)
            if not self._pump_scheduled:
                self._pump_scheduled = True
                schedule_pump = True
        if schedule_pump:
            self._schedule_pump()
        return job.future

    def _schedule_pump(self) -> None:
        scheduler = self._server_state.model_scheduler
        try:
            future = scheduler.submit_foreground(
                self._pump,
                batch_key="mtp_cohort.pump",
            )
        except BaseException as exc:
            self._fail_pump_submission(exc)
            return
        add_done_callback = getattr(future, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(self._pump_submission_done)

    def _pump_submission_done(self, future: Future) -> None:
        if future.cancelled():
            self._fail_pump_submission(
                RuntimeError("MTPK2 cohort pump cancelled before execution")
            )
            return
        try:
            error = future.exception()
        except BaseException as exc:
            error = exc
        if error is not None:
            self._fail_pump_submission(error)

    def _fail_pump_submission(self, exc: BaseException) -> None:
        with self._condition:
            if not self._pump_scheduled:
                return
            self._last_error = f"{type(exc).__name__}: {exc}"
            pending = tuple(self._pending)
            self._pending.clear()
            self._pump_scheduled = False
        for job in pending:
            if not job.future.done():
                job.future.set_exception(exc)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending": len(self._pending),
                "active": len(self._active),
                "pending_request_ids": [
                    job.request_id for job in self._pending
                ],
                "active_request_ids": [
                    active.job.request_id for active in self._active
                ],
                "active_width": int(self._active_width),
                "pump_scheduled": bool(self._pump_scheduled),
                "last_error": self._last_error,
            }

    def _remove_active(self, active: _ActiveJob) -> None:
        with self._condition:
            if active in self._active:
                self._active.remove(active)

    def _finish(self, active: _ActiveJob) -> None:
        output = active.state.output
        final_state = getattr(output, "final_state", None)
        final_cache = getattr(final_state, "final_trunk_cache", None)
        if final_cache is not None:
            self._assert_request_cache(self._lane, final_cache)
        try:
            result = (
                output
                if active.job.owner_finalize is None
                else active.job.owner_finalize(output)
            )
        except BaseException as exc:
            active.state.error = exc
            active.state.status = "failed"
            self._remove_active(active)
            if not active.job.future.done():
                active.job.future.set_exception(exc)
            return
        self._remove_active(active)
        if not active.job.future.done():
            active.job.future.set_result(result)

    def _cancel(self, active: _ActiveJob) -> None:
        active.state.close(status="cancelled")
        self._remove_active(active)
        if not active.job.future.done():
            active.job.future.set_exception(
                MTPK2CohortCancelled(
                    f"MTPK2 cohort request {active.job.request_id!r} cancelled"
                )
            )

    def _fail(self, active: _ActiveJob, exc: BaseException) -> None:
        try:
            active.state.fail(exc)
        except BaseException:
            pass
        self._remove_active(active)
        if not active.job.future.done():
            active.job.future.set_exception(exc)

    def _publish_ticket(self, active: _ActiveJob, ticket: Any | None) -> None:
        if ticket is None:
            if active.state.status == "cancelled":
                self._cancel(active)
            elif active.state.status == "finished":
                self._finish(active)
            else:
                self._fail(
                    active,
                    RuntimeError(
                        "MTPK2 request published no ticket from nonterminal "
                        f"status {active.state.status!r}"
                    ),
                )
            return
        if isinstance(ticket, MTPK2ContextCopyTicket | MTPK2VerifyTicket):
            return
        if not isinstance(ticket, MTPK2PrefillTicket):
            raise TypeError(
                f"unknown MTPK2 ticket {type(ticket).__name__} for "
                f"{active.job.request_id!r}"
            )

    def _admit_one(self) -> bool:
        with self._condition:
            if len(self._active) >= 2 or not self._pending:
                return False
            job = self._pending.popleft()
        try:
            state_kwargs = dict(job.state_kwargs)
            session_bank = state_kwargs.get("session_bank")
            if session_bank is not None:
                state_kwargs["session_bank"] = _RequestLocalSessionBank(
                    session_bank,
                    lambda cache: self._assert_request_cache(
                        self._lane,
                        cache,
                    ),
                )
            state = self._make_state(
                self._server_state.runtime,
                job.prompt_ids,
                request_id=job.request_id,
                environment=job.environment,
                **state_kwargs,
            )
            active = _ActiveJob(job=job, state=state)
            with self._condition:
                self._active.append(active)
            self._publish_ticket(active, state.start())
        except BaseException as exc:
            with self._condition:
                self._last_error = f"{type(exc).__name__}: {exc}"
            if not job.future.done():
                job.future.set_exception(exc)
        return True

    def _ready(self, ticket_type: type) -> list[_ActiveJob]:
        with self._condition:
            active = tuple(self._active)
        return [
            item
            for item in active
            if item.state.status == "ready"
            and isinstance(item.state.require_ticket(), ticket_type)
        ]

    def _discard_cancelled(self) -> None:
        with self._condition:
            active = tuple(self._active)
        for item in active:
            if item.state._cancel_requested():
                self._cancel(item)

    def _execute_verify(self, ready: list[_ActiveJob]) -> None:
        selected = ready[:2]
        if len(selected) == 2:
            for active in selected:
                if active.decode_cache_normalized:
                    continue
                ticket = active.state.require_ticket()
                normalized = self._normalize_cache(
                    self._lane,
                    ticket.request_cache,
                )
                active.state.pending_ticket = replace(
                    ticket,
                    request_cache=normalized,
                )
                active.state.target_cache = normalized
                active.decode_cache_normalized = True
        with self._condition:
            self._active_width = len(selected)
        try:
            results = self._runner.step(
                tuple(active.state for active in selected)
            )
        except BaseException as exc:
            with self._condition:
                self._last_error = f"{type(exc).__name__}: {exc}"
            for active in selected:
                self._fail(active, exc)
            return
        finally:
            with self._condition:
                self._active_width = 0

        survivors: list[_ActiveJob] = []
        for active in selected:
            if active.state.status == "ready":
                survivors.append(active)
            else:
                self._cancel(active)
        if len(results) != len(survivors):
            mismatch = RuntimeError(
                "MTPK2 cohort runner result count does not match live rows"
            )
            for active in survivors:
                self._fail(active, mismatch)
            raise mismatch
        for active, result in zip(survivors, results, strict=True):
            try:
                self._publish_ticket(active, active.state.resume(result))
            except BaseException as exc:
                self._fail(active, exc)

    def _execute_context_copy_ticket(self, active: _ActiveJob) -> None:
        with self._condition:
            self._active_width = 1
        try:
            result = active.state.execute_pending(self._execute_context_copy)
            self._publish_ticket(active, active.state.resume(result))
        finally:
            with self._condition:
                self._active_width = 0

    def _next_prefill(self, ready: list[_ActiveJob]) -> _ActiveJob:
        index = self._prefill_cursor % len(ready)
        self._prefill_cursor += 1
        return ready[index]

    def _execute_prefill_ticket(self, active: _ActiveJob) -> None:
        ticket = active.state.require_ticket()
        chunk_tokens = int(ticket.prompt_stop) - int(ticket.prompt_start)
        if chunk_tokens > 1024:
            raise RuntimeError(
                f"MTPK2 prefill chunk exceeds frozen 1024-token budget: "
                f"{chunk_tokens}"
            )
        result = active.state.execute_pending(self._execute_prefill)
        self._publish_ticket(active, active.state.resume(result))

    def _pump(self) -> None:
        prefilled_since_target: set[str] = set()
        prefill_between_target_cycles = False
        try:
            while True:
                self._discard_cancelled()
                with self._condition:
                    active_ids = {
                        active.job.request_id for active in self._active
                    }
                prefilled_since_target.intersection_update(active_ids)
                verify_ready = self._ready(MTPK2VerifyTicket)
                context_copy_ready = self._ready(MTPK2ContextCopyTicket)
                prefill_ready = self._ready(MTPK2PrefillTicket)
                prefill_eligible = [
                    active
                    for active in prefill_ready
                    if active.job.request_id not in prefilled_since_target
                ]
                if (
                    verify_ready
                    and prefill_between_target_cycles
                    and prefill_eligible
                ):
                    active = self._next_prefill(prefill_eligible)
                    try:
                        self._execute_prefill_ticket(active)
                        prefilled_since_target.add(active.job.request_id)
                    except BaseException as exc:
                        if active.state.status == "cancelled":
                            self._cancel(active)
                        else:
                            self._fail(active, exc)
                    prefill_between_target_cycles = False
                    continue
                if context_copy_ready:
                    active = context_copy_ready[0]
                    try:
                        self._execute_context_copy_ticket(active)
                    except BaseException as exc:
                        if active.state.status == "cancelled":
                            self._cancel(active)
                        else:
                            self._fail(active, exc)
                    prefilled_since_target.clear()
                    prefill_between_target_cycles = True
                    self._admit_one()
                    continue
                if verify_ready:
                    self._execute_verify(verify_ready)
                    prefilled_since_target.clear()
                    prefill_between_target_cycles = True
                    self._admit_one()
                    continue

                prefill_between_target_cycles = False
                if self._admit_one():
                    continue

                prefill_ready = self._ready(MTPK2PrefillTicket)
                prefill_eligible = [
                    active
                    for active in prefill_ready
                    if active.job.request_id not in prefilled_since_target
                ]
                if prefill_eligible:
                    active = self._next_prefill(prefill_eligible)
                    try:
                        self._execute_prefill_ticket(active)
                        prefilled_since_target.add(active.job.request_id)
                    except BaseException as exc:
                        if active.state.status == "cancelled":
                            self._cancel(active)
                        else:
                            self._fail(active, exc)
                    continue
                if prefill_ready:
                    prefilled_since_target.clear()
                    continue

                verify_ready = self._ready(MTPK2VerifyTicket)
                if verify_ready:
                    continue
                context_copy_ready = self._ready(MTPK2ContextCopyTicket)
                if context_copy_ready:
                    continue

                with self._condition:
                    if self._pending or self._active:
                        continue
                    self._pump_scheduled = False
                    return
        except BaseException as exc:
            with self._condition:
                self._last_error = f"{type(exc).__name__}: {exc}"
                active = tuple(self._active)
                pending = tuple(self._pending)
                self._pending.clear()
            for item in active:
                self._fail(item, exc)
            for job in pending:
                if not job.future.done():
                    job.future.set_exception(exc)
            with self._condition:
                self._pump_scheduled = False
