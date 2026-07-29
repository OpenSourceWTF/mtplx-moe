"""Request-local coroutine boundary for native depth-two MTP generation.

The module deliberately has no MLX imports.  A model owner can therefore keep
request state and scheduling policy separate from the prebound target routes
that execute prefill and verification tickets.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MTPK2AcceptanceContext:
    """Fixed target-verification choices plus request-local acceptance state."""

    verify_strategy: str
    verify_core: str
    hidden_variant: str
    purpose: str = "verify"
    before_verify: Any | None = None
    event: Any | None = None
    execution: Any | None = None


@dataclass(frozen=True)
class MTPK2VerifyTicket:
    request_id: str
    input_ids: Any
    request_cache: list[Any]
    draft_distributions: tuple[Any, Any]
    acceptance_context: MTPK2AcceptanceContext


@dataclass(frozen=True)
class MTPK2PrefillTicket:
    request_id: str
    input_ids: Any
    request_cache: list[Any]
    prompt_start: int
    prompt_stop: int
    execution: Any | None = None


@dataclass(frozen=True)
class MTPK2PrefillResult:
    logits: Any | None
    hidden: Any | None
    request_cache: list[Any]
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class MTPK2ContextCopyTicket:
    """One variable-length prompt-lookup verify owned by a single request."""

    request_id: str
    input_ids: Any
    request_cache: list[Any]
    hidden_variant: str
    capture_backend: str


@dataclass(frozen=True)
class MTPK2ContextCopyResult:
    logits: Any
    hidden: Any
    captures: dict[int, dict[str, Any]] | Any
    request_cache: list[Any]
    forward_elapsed_s: float | None = None


@dataclass(frozen=True)
class MTPK2VerifyResult:
    logits: Any
    hidden: Any
    captures: dict[int, dict[str, Any]] | Any
    request_cache: list[Any]
    commit_prefix: Callable[[int], list[Any]]
    forward_elapsed_s: float | None = None


MTPK2Ticket = (
    MTPK2PrefillTicket
    | MTPK2ContextCopyTicket
    | MTPK2VerifyTicket
)
MTPK2TicketResult = (
    MTPK2PrefillResult
    | MTPK2ContextCopyResult
    | MTPK2VerifyResult
)


class MTPK2RequestCancelled(RuntimeError):
    """Raised when a ready request is cancelled before ticket dispatch."""


@dataclass(frozen=True)
class MTPK2RequestConfig:
    """Construction-bound flags used while one request machine advances."""

    prompt_tokens: int
    prefill_chunk_tokens: int
    environment: Mapping[str, str]
    compiled_verify_mode: str
    compiled_verify_max_len: int
    loop_guard_config: Any
    target_prefill_cache_factory: Callable[[], list[Any]]
    target_prefill_repage_route: Callable[[list[Any]], Any]
    mtp_cache_factory: Callable[[], Any]
    sustained_prefill_enabled: bool
    state_rebase_every: int
    sustained_prefill_layout: str
    defer_verify_hidden_eval: bool
    clear_cache_every: int
    skip_verify_snapshot: bool
    omit_speculative_bonus: bool
    batch_target_arrays: bool
    batch_target_distributions: bool
    compiled_verify_stats: bool


_CURRENT_MTPK2_REQUEST_CONFIG: ContextVar[MTPK2RequestConfig | None] = ContextVar(
    "mtplx_current_k2_request_config",
    default=None,
)


def current_mtpk2_request_config() -> MTPK2RequestConfig | None:
    return _CURRENT_MTPK2_REQUEST_CONFIG.get()


@dataclass
class MTPK2RequestState:
    request_id: str
    _machine: Generator[MTPK2Ticket, MTPK2TicketResult, Any] | None = field(
        repr=False
    )
    config: MTPK2RequestConfig
    lane: Any
    width1_commit_route: Callable[..., list[Any]]
    target_cache: list[Any]
    mtp_cache: Any
    tokens: list[int]
    rng: np.random.Generator
    sampler: Any
    draft_sampler: Any
    constraint: Any
    stop_token_ids: set[int]
    token_callback: Callable[[list[int]], None] | None
    prefill_callback: Callable[..., None] | None
    cancel_event: Any | None
    session_id: str | None
    stats: Any
    prefill_chunk_tokens: int = 1024
    context_copy_enabled: bool = False
    lazy_bonus_verify_requested: bool = False
    lazy_bonus_verify_min_depth: int = 2
    pending_ticket: MTPK2Ticket | None = None
    status: str = "created"
    output: Any | None = None
    error: BaseException | None = field(default=None, repr=False)

    def _require_machine(
        self,
    ) -> Generator[MTPK2Ticket, MTPK2TicketResult, Any]:
        if self._machine is None:
            raise RuntimeError("MTPK2RequestState machine is not installed")
        return self._machine

    def _cancel_requested(self) -> bool:
        is_set = getattr(self.cancel_event, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        return bool(self.cancel_event()) if callable(self.cancel_event) else False

    def _publish_or_finish(
        self,
        advance: Callable[
            [Generator[MTPK2Ticket, MTPK2TicketResult, Any]],
            MTPK2Ticket,
        ],
    ) -> MTPK2Ticket | None:
        if self._cancel_requested():
            self.close(status="cancelled")
            return None
        machine = self._require_machine()
        token = _CURRENT_MTPK2_REQUEST_CONFIG.set(self.config)
        try:
            ticket = advance(machine)
        except StopIteration as finished:
            self.pending_ticket = None
            self.output = finished.value
            self.status = "finished"
            return None
        except BaseException as exc:
            self.pending_ticket = None
            self.error = exc
            self.status = "failed"
            machine.close()
            raise
        finally:
            _CURRENT_MTPK2_REQUEST_CONFIG.reset(token)
        self.pending_ticket = ticket
        self.status = "ready"
        return ticket

    def start(self) -> MTPK2Ticket | None:
        if self.status != "created":
            raise RuntimeError(
                f"MTPK2RequestState.start requires created status, got {self.status}"
            )
        return self._publish_or_finish(next)

    def resume(self, result: MTPK2TicketResult) -> MTPK2Ticket | None:
        if self.status != "ready" or self.pending_ticket is None:
            raise RuntimeError("MTPK2RequestState.resume requires a pending ticket")
        if self._cancel_requested():
            self.close(status="cancelled")
            return None
        self.pending_ticket = None
        self.target_cache = result.request_cache
        return self._publish_or_finish(lambda machine: machine.send(result))

    def fail(self, exc: BaseException) -> None:
        if self.status != "ready" or self.pending_ticket is None:
            raise RuntimeError("MTPK2RequestState.fail requires a pending ticket")
        machine = self._require_machine()
        self.pending_ticket = None
        self.error = exc
        self.status = "failed"
        token = _CURRENT_MTPK2_REQUEST_CONFIG.set(self.config)
        try:
            try:
                machine.throw(exc)
            except StopIteration as swallowed:
                raise AssertionError(
                    "request machine swallowed ticket failure"
                ) from swallowed
        finally:
            _CURRENT_MTPK2_REQUEST_CONFIG.reset(token)
            machine.close()
        raise AssertionError("request machine yielded after ticket failure")

    def close(self, *, status: str = "closed") -> None:
        terminal = {"closed", "cancelled", "failed", "finished"}
        if self.status in terminal:
            return
        machine = self._machine
        self.pending_ticket = None
        if machine is not None:
            token = _CURRENT_MTPK2_REQUEST_CONFIG.set(self.config)
            try:
                machine.close()
            finally:
                _CURRENT_MTPK2_REQUEST_CONFIG.reset(token)
        self.status = status

    def require_ticket(self) -> MTPK2Ticket:
        if self.pending_ticket is None:
            raise RuntimeError("MTPK2RequestState has no pending ticket")
        return self.pending_ticket

    def execute_pending(
        self,
        executor: Callable[[MTPK2Ticket], MTPK2TicketResult],
    ) -> MTPK2TicketResult:
        ticket = self.require_ticket()
        if self._cancel_requested():
            self.close(status="cancelled")
            raise MTPK2RequestCancelled(
                f"MTPK2 request {self.request_id!r} cancelled before dispatch"
            )
        token = _CURRENT_MTPK2_REQUEST_CONFIG.set(self.config)
        try:
            return executor(ticket)
        finally:
            _CURRENT_MTPK2_REQUEST_CONFIG.reset(token)


def drive_solo_mtpk2(
    state: MTPK2RequestState,
    *,
    execute_prefill: Callable[[MTPK2PrefillTicket], MTPK2PrefillResult],
    execute_verify: Callable[[MTPK2VerifyTicket], MTPK2VerifyResult],
    execute_context_copy: Callable[
        [MTPK2ContextCopyTicket],
        MTPK2ContextCopyResult,
    ]
    | None = None,
) -> Any:
    """Consume one request's tickets immediately, preserving serial behavior."""

    ticket = state.start()
    while ticket is not None:
        try:
            if isinstance(ticket, MTPK2PrefillTicket):
                result = state.execute_pending(execute_prefill)
            elif isinstance(ticket, MTPK2ContextCopyTicket):
                if execute_context_copy is None:
                    raise RuntimeError(
                        "context-copy ticket requires an explicit executor"
                    )
                result = state.execute_pending(execute_context_copy)
            else:
                result = state.execute_pending(execute_verify)
        except MTPK2RequestCancelled:
            return state.output
        except BaseException as exc:
            state.fail(exc)
            raise AssertionError("unreachable after request failure") from exc
        ticket = state.resume(result)
    return state.output
