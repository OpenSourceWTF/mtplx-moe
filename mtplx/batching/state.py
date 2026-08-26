"""Request-state model for MTPLX cooperative batching.

These dataclasses are deliberately lightweight: they hold scheduling metadata
and per-request ownership, not model tensors. That keeps the future native-MTP
cohort path honest: speculative state remains request-local until a commit is
known to be exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from threading import Event
import time
from typing import Any, Callable


class SchedulerMode(StrEnum):
    """Runtime scheduler modes exposed to CLI, dashboard, and native app."""

    SERIAL = "serial"
    COOPERATIVE = "cooperative"
    AR_BATCH = "ar_batch"
    MTP_BATCH = "mtp_batch"
    MTP_COHORT_EXPERIMENTAL = "mtp_cohort_experimental"
    # Hyper: batch machinery spent on ONE request (external admission = 1,
    # simultaneous requests queue FIFO exactly like serial). Width, when it
    # exists (H1/H2), is self-generated speculative rows for that request —
    # never a second client. Stage H0 is the singleton chassis: width 1 rides
    # the untouched serial B1 path. See mtplx/server/hyper.py.
    HYPER = "hyper"


class SchedulerPreset(StrEnum):
    """User-facing batching presets."""

    SOLO = "solo"
    LATENCY = "latency"
    AGENT = "agent"
    THROUGHPUT = "throughput"


class RequestPhase(StrEnum):
    QUEUED = "queued"
    RESTORING = "restoring"
    PREFILLING = "prefilling"
    DECODE_READY = "decode_ready"
    DECODING = "decoding"
    POSTCOMMIT = "postcommit"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RequestPriority(IntEnum):
    BACKGROUND = 0
    NORMAL = 10
    FOREGROUND = 20
    TOOL_TURN = 30


@dataclass(frozen=True)
class PresetConfig:
    name: SchedulerPreset
    max_active_requests: int
    decode_batch_max: int
    batch_wait_ms: float
    prefill_chunk_tokens: int
    soft_memory_fraction: float = 0.85
    hard_memory_fraction: float = 0.92


_PRESETS: dict[SchedulerPreset, PresetConfig] = {
    SchedulerPreset.SOLO: PresetConfig(
        name=SchedulerPreset.SOLO,
        max_active_requests=1,
        decode_batch_max=1,
        batch_wait_ms=0.0,
        prefill_chunk_tokens=1024,
    ),
    SchedulerPreset.LATENCY: PresetConfig(
        name=SchedulerPreset.LATENCY,
        max_active_requests=1,
        decode_batch_max=1,
        batch_wait_ms=0.0,
        prefill_chunk_tokens=1024,
    ),
    SchedulerPreset.AGENT: PresetConfig(
        name=SchedulerPreset.AGENT,
        max_active_requests=4,
        decode_batch_max=4,
        batch_wait_ms=50.0,
        prefill_chunk_tokens=2048,
    ),
    SchedulerPreset.THROUGHPUT: PresetConfig(
        name=SchedulerPreset.THROUGHPUT,
        max_active_requests=8,
        decode_batch_max=8,
        batch_wait_ms=20.0,
        prefill_chunk_tokens=2048,
    ),
}


def preset_config(name: str | SchedulerPreset) -> PresetConfig:
    try:
        preset = name if isinstance(name, SchedulerPreset) else SchedulerPreset(str(name))
    except ValueError:
        preset = SchedulerPreset.LATENCY
    return _PRESETS[preset]


@dataclass
class RequestState:
    """Cooperative scheduler state owned by one request."""

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    max_tokens: int = 0
    phase: RequestPhase = RequestPhase.QUEUED
    priority: RequestPriority = RequestPriority.NORMAL
    client_kind: str = "openai"
    session_id: str | None = None
    batch_key: str | None = None
    sampler: Any | None = None
    draft_sampler: Any | None = None
    speculative_depth: int = 0
    stop_token_ids: set[int] = field(default_factory=set)
    deadline_s: float | None = None
    stream_callback: Callable[[list[int]], None] | None = None
    cancel_event: Event = field(default_factory=Event)
    created_s: float = field(default_factory=time.monotonic)
    admitted_s: float | None = None
    phase_started_s: float = field(default_factory=time.monotonic)
    last_step_s: float | None = None
    tokens_generated: int = 0
    prompt_tokens_done: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_phase(self, phase: RequestPhase, *, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else float(now_s)
        self.phase = phase
        self.phase_started_s = now
        self.last_step_s = now
        if phase != RequestPhase.QUEUED and self.admitted_s is None:
            self.admitted_s = now

    def cancel(self) -> None:
        self.cancel_event.set()
        self.mark_phase(RequestPhase.CANCELLED)

    def fail(self, exc: BaseException | str) -> None:
        self.error = str(exc)
        self.mark_phase(RequestPhase.FAILED)

    def finish(self) -> None:
        self.mark_phase(RequestPhase.FINISHED)

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            RequestPhase.FINISHED,
            RequestPhase.CANCELLED,
            RequestPhase.FAILED,
        }

    @property
    def queue_wait_s(self) -> float:
        end = self.admitted_s if self.admitted_s is not None else time.monotonic()
        return max(0.0, end - self.created_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "phase": self.phase.value,
            "priority": int(self.priority),
            "client_kind": self.client_kind,
            "session_id": self.session_id,
            "batch_key": self.batch_key,
            "speculative_depth": self.speculative_depth,
            "prompt_tokens": len(self.prompt_ids),
            "prompt_tokens_done": self.prompt_tokens_done,
            "tokens_generated": self.tokens_generated,
            "max_tokens": self.max_tokens,
            "queue_wait_s": self.queue_wait_s,
            "cancelled": self.cancel_event.is_set(),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class DecodeState:
    """Per-request decode ownership for future stepable AR/MTP execution."""

    request: RequestState
    generated_tokens: list[int] = field(default_factory=list)
    rng_seed: int | None = None
    accepted_by_depth: list[int] = field(default_factory=list)
    drafted_by_depth: list[int] = field(default_factory=list)
    mtp_disabled_reason: str | None = None
    cache_owner_id: str | None = None
    safe_to_commit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "generated_tokens": len(self.generated_tokens),
            "rng_seed": self.rng_seed,
            "accepted_by_depth": list(self.accepted_by_depth),
            "drafted_by_depth": list(self.drafted_by_depth),
            "mtp_disabled_reason": self.mtp_disabled_reason,
            "cache_owner_id": self.cache_owner_id,
            "safe_to_commit": self.safe_to_commit,
        }
