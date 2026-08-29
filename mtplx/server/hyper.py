"""Hyper scheduler chassis (stage H0): admission gate + width seam.

``--scheduler-mode hyper`` is the mode that will spend batch width on
SELF-GENERATED speculative rows for a single request (HYPER-PLAN Phase 3),
never on multiple external requests. Its two contracts are fixed here, at
stage H0, before any width machinery exists:

1. **External admission = 1.** Exactly one request is in flight; simultaneous
   requests queue FIFO behind the active one — the same semantics serial mode
   gives today. This gate deliberately adds NO lock, semaphore, or wait window
   of its own: the cap is enforced by the single model-owner thread's FIFO
   foreground band (``mtplx.model_scheduler.ModelWorkScheduler``), the exact
   mechanism serial mode uses. The gate only *accounts* (queue depth, waits,
   totals) so the admission surface is observable and so H1/H2 width receipts
   have a stable home.

2. **Singleton passthrough.** At width 1 the request rides the UNTOUCHED
   serial B1 path — solo MTP oracle, paged KV cache, custom Metal fast paths,
   CompiledVerifyBank, all intact. No padded batch object is ever built at
   width 1 and no gather/wait window is ever held (the oMLX lesson: a width-1
   ride through a batched driver eats the ~380-vs-137 ms eager-verify tax and
   loses the parity gate). Width-1 must therefore never route through any
   batched driver — dense batched-MTP drivers force-disable paged KV + custom
   kernels for the cohort's duration and are structurally wrong for hyper's
   width-1 floor.

The **width seam** (see :meth:`HyperAdmissionGate.install_width_executor` and
:class:`HyperWidthExecutor`) is where H1 (dummy-row geometry probe) and H2
(exact B2 tree) attach without touching the serial path again. The seam
contract is documented on the protocol below; the H0 gate ships with no
executor installed, plans every request as ``width=1``, and fails loudly if a
wider plan ever appears without an executor.

Parity gate for this stage (controller-run, GPU): hyper at width 1 must hold
>= 0.99x serial tok/s AND a byte-identical trajectory sha on the
nprompt-16k/71k cells. Hyper must never ship slower at width 1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, Callable, Protocol

from mtplx.model_scheduler import _sample_summary

# External concurrency is definitional for hyper, not a tunable: width is spent
# on self-generated rows for the one admitted request. Validation rejects any
# launch flag that contradicts this (see openai._validate_hyper_settings).
HYPER_ADMISSION_CAP = 1

# Chassis stage marker surfaced in stats; H1/H2 bump it when they install.
HYPER_STAGE_H0 = "h0"

# Receipt string for the width-1 route: the serial B1 solo-oracle path.
SINGLETON_PASSTHROUGH_ROUTE = "serial_b1"


@dataclass(frozen=True)
class HyperRequestMeta:
    """What the width planner may see about a request. Deliberately small:
    the planner must never need model tensors to decide a width."""

    request_id: str | None
    prompt_tokens: int
    generation_mode: str


@dataclass(frozen=True)
class HyperCohortPlan:
    """A width decision for one admitted request.

    ``width`` counts rows in the verify forward: 1 = singleton passthrough
    (serial B1 path, no batch object); N>1 = one real row plus N-1
    self-generated speculative rows for the SAME request. ``reason`` is the
    telemetry label for why this width was chosen (H2: fork-trigger receipts).
    """

    width: int
    reason: str

    def __post_init__(self) -> None:
        if int(self.width) < 1:
            raise ValueError(f"HyperCohortPlan.width must be >= 1, got {self.width}")


SINGLETON_PLAN = HyperCohortPlan(width=1, reason="h0_singleton_passthrough")


class HyperWidthExecutor(Protocol):
    """The H1/H2 attach point. Install via
    :meth:`HyperAdmissionGate.install_width_executor`.

    Seam contract (binding on every implementation):

    - ``plan`` runs per request on the model-owner thread, before any model
      work for that request. Returning ``width=1`` means singleton
      passthrough: the gate calls the serial closure DIRECTLY — the executor
      is bypassed entirely, no batch object is built, no wait window is held.
      An executor must plan ``width=1`` until it actually holds a second
      self-generated row worth verifying (never pad, never wait for one).
    - ``run_cohort`` owns width>1 execution end to end and must return the
      same result mapping the serial closure returns (the server tail cannot
      tell them apart). It receives the untouched serial closure as
      ``singleton`` and MUST fall back to calling it — not to a padded
      width-1 cohort — whenever its cohort cannot form.
    - Width is per-request internal machinery. External admission stays 1;
      the executor never sees a second request.
    - Never route through a batched driver that disables the paged KV cache,
      custom kernels, or the CompiledVerifyBank for width-1 traffic.
    - Cache identity: H0's singleton adds no policy-fingerprint part because
      it IS the serial path (byte-identical trajectories, interchangeable
      session state). An executor whose cohorts change execution numerics
      (the H-fork "named lane" outcome) must stamp its identity into the
      server's ``_policy_fingerprint`` before serving, exactly as the
      mtp_batch lane does.
    """

    def plan(self, request: HyperRequestMeta) -> HyperCohortPlan: ...

    def run_cohort(
        self,
        plan: HyperCohortPlan,
        singleton: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass
class HyperTicket:
    """Admission accounting handle for one request (state guarded by the gate
    lock; tickets themselves are single-owner and never shared)."""

    request_id: str | None
    prompt_tokens: int
    admitted_at_s: float
    started_at_s: float | None = None
    finished: bool = False


class HyperAdmissionGate:
    """Admission accounting + width seam for hyper mode.

    Threading model: ``admit``/``release`` run on request-handler threads;
    the bound closure (``bind``) runs on the single model-owner thread, so at
    most one ticket is ever *started* at a time — that invariant is inherited
    from the owner thread, not re-implemented here. All counter mutation
    happens under one small lock; nothing here touches MLX or the model.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._width_executor: HyperWidthExecutor | None = None
        self._queued = 0
        self._active = 0
        self._admitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._abandoned_before_start_total = 0
        self._queue_wait_samples_s: deque[float] = deque(maxlen=256)
        self._run_samples_s: deque[float] = deque(maxlen=256)
        # Width receipts: how many requests committed at each width. H0 only
        # ever records width 1; H1/H2 executors report wider cohorts through
        # record_cohort_width so their receipts land in the same histogram.
        self._requests_by_width: dict[int, int] = {}
        # Live width of the in-flight request. Definitionally 1 at H0; a
        # width executor stamps wider values via set_live_width while its
        # cohort runs and restores 1 at join.
        self._live_width = 1
        # Reserved H1/H2 counters (structurally present so the stats surface
        # is stable before the width machinery exists; only a width executor
        # increments them).
        self._self_spec_rows_launched = 0
        self._self_spec_rows_committed = 0
        self._fork_attempts = 0
        self._fork_wins = 0

    # ------------------------------------------------------------------
    # Width seam
    # ------------------------------------------------------------------

    def install_width_executor(self, executor: HyperWidthExecutor) -> None:
        """Attach the H1/H2 width machinery. H0 never calls this; it exists
        so the later stages plug in without another routing change."""

        if not hasattr(executor, "plan") or not hasattr(executor, "run_cohort"):
            raise TypeError(
                "hyper width executor must provide plan() and run_cohort() "
                "(see HyperWidthExecutor)"
            )
        with self._lock:
            self._width_executor = executor

    def clear_width_executor(self) -> None:
        with self._lock:
            self._width_executor = None

    @property
    def width_executor_installed(self) -> bool:
        with self._lock:
            return self._width_executor is not None

    def plan_cohort(self, request: HyperRequestMeta) -> HyperCohortPlan:
        """Width decision for one admitted request.

        Without an installed executor this is constantly ``width=1``: the
        H0 chassis has no width machinery, and the singleton plan is the
        documented safe floor, not a fallback.
        """

        with self._lock:
            executor = self._width_executor
        if executor is None:
            return SINGLETON_PLAN
        plan = executor.plan(request)
        if not isinstance(plan, HyperCohortPlan):
            raise TypeError(
                "hyper width executor plan() must return HyperCohortPlan, "
                f"got {type(plan).__name__}"
            )
        return plan

    def record_cohort_width(self, width: int) -> None:
        """Width-histogram hook for H1/H2 executors (per committed request)."""

        with self._lock:
            key = max(1, int(width))
            self._requests_by_width[key] = self._requests_by_width.get(key, 0) + 1

    def set_live_width(self, width: int) -> None:
        """Live-width receipt for the in-flight request. Only a width
        executor calls this (stamp N while the cohort runs, restore 1 at
        join); H0 traffic never moves it off 1."""

        with self._lock:
            self._live_width = max(1, int(width))

    # ------------------------------------------------------------------
    # Admission accounting
    # ------------------------------------------------------------------

    def admit(
        self,
        *,
        request_id: str | None,
        prompt_tokens: int,
    ) -> HyperTicket:
        """Account one request entering the FIFO. Never blocks and never
        rejects: the cap-at-1 semantics come from the model-owner thread's
        FIFO band, exactly as in serial mode."""

        ticket = HyperTicket(
            request_id=request_id,
            prompt_tokens=int(prompt_tokens),
            admitted_at_s=time.monotonic(),
        )
        with self._lock:
            self._queued += 1
            self._admitted_total += 1
        return ticket

    def bind(
        self,
        ticket: HyperTicket,
        singleton: Callable[[], dict[str, Any]],
        *,
        generation_mode: str = "mtp",
    ) -> Callable[[], dict[str, Any]]:
        """Wrap the serial closure with admission accounting + the width seam.

        The returned callable is what the server submits to the model-owner
        FIFO in place of ``singleton``. On execution it plans a width; width 1
        calls ``singleton`` directly (the serial code object, untouched);
        width>1 requires an installed executor and hands it the plan plus the
        same closure. Accounting is failure-safe: completion or exception both
        settle the ticket.
        """

        meta = HyperRequestMeta(
            request_id=ticket.request_id,
            prompt_tokens=ticket.prompt_tokens,
            generation_mode=str(generation_mode),
        )

        def bound() -> dict[str, Any]:
            self._mark_started(ticket)
            try:
                plan = self.plan_cohort(meta)
                if plan.width == 1:
                    result = singleton()
                else:
                    with self._lock:
                        executor = self._width_executor
                    if executor is None:
                        raise RuntimeError(
                            "hyper cohort plan requested width "
                            f"{plan.width} but no width executor is "
                            "installed (H0 chassis is singleton-only)"
                        )
                    result = executor.run_cohort(plan, singleton)
                self._mark_finished(ticket, plan_width=plan.width, failed=False)
                return result
            except BaseException:
                self._mark_finished(ticket, plan_width=1, failed=True)
                raise

        return bound

    def release(self, ticket: HyperTicket) -> None:
        """Settle a ticket whose bound closure never ran (cancelled or failed
        before the owner thread picked it up). Idempotent; a no-op for
        tickets that started."""

        with self._lock:
            if ticket.finished or ticket.started_at_s is not None:
                return
            ticket.finished = True
            self._queued = max(0, self._queued - 1)
            self._abandoned_before_start_total += 1

    def _mark_started(self, ticket: HyperTicket) -> None:
        now = time.monotonic()
        with self._lock:
            ticket.started_at_s = now
            self._queued = max(0, self._queued - 1)
            self._active += 1
            self._queue_wait_samples_s.append(max(0.0, now - ticket.admitted_at_s))

    def _mark_finished(
        self, ticket: HyperTicket, *, plan_width: int, failed: bool
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if ticket.finished:
                return
            ticket.finished = True
            self._active = max(0, self._active - 1)
            if ticket.started_at_s is not None:
                self._run_samples_s.append(max(0.0, now - ticket.started_at_s))
            if failed:
                self._failed_total += 1
            else:
                self._completed_total += 1
                key = max(1, int(plan_width))
                self._requests_by_width[key] = (
                    self._requests_by_width.get(key, 0) + 1
                )

    # ------------------------------------------------------------------
    # Stats surface
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Stats block served under ``scheduler.hyper`` in /health and the
        dashboard. Field stability matters: H1/H2 receipts append here."""

        with self._lock:
            widths = dict(self._requests_by_width)
            return {
                "stage": HYPER_STAGE_H0,
                "admission_cap": HYPER_ADMISSION_CAP,
                "width": self._live_width,
                "passthrough": SINGLETON_PASSTHROUGH_ROUTE,
                "width_executor_installed": self._width_executor is not None,
                "active": self._active,
                "queue_depth": self._queued,
                "admitted_total": self._admitted_total,
                "completed_total": self._completed_total,
                "failed_total": self._failed_total,
                "abandoned_before_start_total": (
                    self._abandoned_before_start_total
                ),
                "queue_wait_s": _sample_summary(self._queue_wait_samples_s),
                "run_s": _sample_summary(self._run_samples_s),
                # Width receipts. requests_by_width is the histogram home for
                # H1/H2 cells; the row/fork counters are reserved names that
                # only a width executor increments (H2 fork-trigger EV cells
                # land here).
                "width_stats": {
                    "requests_by_width": {
                        str(width): count
                        for width, count in sorted(widths.items())
                    },
                    "self_spec_rows_launched": self._self_spec_rows_launched,
                    "self_spec_rows_committed": self._self_spec_rows_committed,
                    "fork_attempts": self._fork_attempts,
                    "fork_wins": self._fork_wins,
                },
            }
