"""Thinking Guard: a surfaced per-request budget for reasoning segments.

Why this exists (2026-07-20, live agent-session forensics): on the
OpenCode agent lane a single tool-loop turn produced a 66,396-char
``<think>`` segment — 21,222 tokens, 954 s, zero user-visible output — a
semantic self-doubt loop (re-deriving en-passant rules) that never repeats
verbatim, so the Loop Guard's DRY detector is blind to it by design and the
repetition stop never fires. Across that session ~75% of ALL generated
tokens were thinking; wall-clock per project turn was dominated not by
decode speed but by unbounded reasoning. Full forensics:
the 2026-07-20 investigation record.

This is a BUDGET, not steering (project policy, 2026-07-08: no synthetic
steering touches sampling by default — DRY remains opt-in). The guard:

1. Counts committed reasoning tokens (cumulative across ``<think>`` blocks,
   template-preopened blocks included).
2. Below the budget it emits nothing and the decode path stays bit-exact.
3. At the budget it force-closes the reasoning segment by emitting a fixed
   bridge + ``</think>`` token sequence through the standard target-side
   overlay slot (`penalty_overlay`, negative value = raw-logit boost applied
   BEFORE top-k, so the forced token always enters the support). Draft
   mismatches are rejected and corrected by the ordinary residual math, so
   speculative acceptance stays exact.
4. After the close it bans ``<think>`` re-entry for a short window
   (tool-call payload spans exempt — literal "<think>" text inside written
   files must never be corrupted, per the 2026-07-09 Loop Guard span-mask
   lesson), then goes fully dormant so the remainder of the request runs the
   untouched fast paths. A later re-open past the ban window is closed again
   on sight (budget is cumulative).

An optional, DEFAULT-OFF novelty close (``novelty_close``) additionally
force-closes a reasoning segment whose trailing window has collapsed into
shingle recurrence (the Loop Guard arming test, restricted to reasoning
tokens) — the "clearly circling far below budget" case. It is a sampler
intervention beyond a plain cap, so it ships opt-in by default.


Interface mirrors mtplx/loop_guard.py: one instance per generation call,
``observe(tokens)`` once per step on committed tokens,
``overlay_for(working)`` per sampling position while ``steering_active``,
``summary()`` for telemetry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "ThinkingGuard",
    "ThinkingGuardConfig",
    "think_marker_ids",
    "thinking_guard_config_from_env",
]

# Raw-logit magnitudes for the overlay slot (subtracted; negative = boost).
# 1e4 dwarfs any real logit while staying float32-safe through the
# temperature divide and logsumexp.
_FORCE_BOOST = -1.0e4
_REENTRY_BAN = 1.0e4


@dataclass(frozen=True)
class ThinkingGuardConfig:
    enabled: bool = False
    think_open_token: int | None = None
    think_close_token: int | None = None
    # Cumulative reasoning-token budget for the request. At/over this count
    # the forced close engages. <=0 disables the budget lane.
    budget_tokens: int = 3072
    # Bridge emitted inside the reasoning segment before the close marker so
    # the visible answer never starts mid-sentence. Encoded by the caller
    # (ids, not text) so tokenization is fixed at config-build time.
    forced_close_ids: tuple[int, ...] = ()
    # Whether generation starts inside an already-open reasoning segment
    # (Qwen templates pre-open ``<think>`` in the generation prompt).
    starts_in_think: bool = False
    # <think> re-entry ban window (committed tokens after the forced close).
    reentry_ban_tokens: int = 64
    # Tool-call span markers: the ban never applies inside tool payloads.
    mask_open_token: int | None = None
    mask_close_token: int | None = None
    # Optional novelty close (default OFF; opt-in).
    novelty_close: bool = False
    novelty_ngram: int = 24
    novelty_occurrences: int = 3
    novelty_window: int = 4096
    novelty_min_tokens: int = 1024
    novelty_scan_interval: int = 64


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def think_marker_ids(tokenizer: Any) -> tuple[int, int] | None:
    """Resolve single-token ``<think>``/``</think>`` ids, else None."""
    if tokenizer is None:
        return None
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None

    def _single_id(text: str) -> int | None:
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
        except Exception:
            return None
        try:
            ids = [int(token) for token in ids]
        except (TypeError, ValueError):
            return None
        return ids[0] if len(ids) == 1 else None

    open_id = _single_id("<think>")
    close_id = _single_id("</think>")
    if open_id is None or close_id is None or open_id == close_id:
        return None
    return open_id, close_id


_DISABLED_VALUES = {"0", "false", "off", "no"}


def thinking_guard_config_from_env(
    enabled: bool,
    *,
    budget_tokens: int,
    tokenizer: Any = None,
    starts_in_think: bool = False,
    bridge_text: str | None = None,
    novelty_close: bool = False,
) -> ThinkingGuardConfig:
    """Build the guard config; ``enabled``/``budget_tokens`` are the caller's
    resolved product policy (lane + effort mapping live server-side).

    ``MTPLX_THINKING_BUDGET`` overrides the budget ("off"/"0" disables,
    an int replaces); ``MTPLX_THINKING_NOVELTY_CLOSE=1`` opts the novelty
    lane in. A tokenizer without single-token think markers disables the
    guard outright (no approximate matching).
    """
    raw = os.environ.get("MTPLX_THINKING_BUDGET")
    if raw is not None and raw.strip() != "":
        lowered = raw.strip().lower()
        if lowered in _DISABLED_VALUES:
            enabled = False
        else:
            try:
                budget_tokens = int(lowered)
                enabled = budget_tokens > 0
            except ValueError:
                pass
    nov_raw = os.environ.get("MTPLX_THINKING_NOVELTY_CLOSE", "").strip().lower()
    if nov_raw != "":
        novelty_close = nov_raw not in _DISABLED_VALUES
    if not enabled or budget_tokens <= 0:
        return ThinkingGuardConfig(enabled=False)
    markers = think_marker_ids(tokenizer)
    if markers is None:
        return ThinkingGuardConfig(enabled=False)
    open_id, close_id = markers
    bridge = (
        bridge_text
        if bridge_text is not None
        else os.environ.get(
            "MTPLX_THINKING_BRIDGE",
            "\n\nI have enough analysis — deciding now and acting.\n",
        )
    )
    forced: list[int] = []
    encode = getattr(tokenizer, "encode", None)
    if bridge and encode is not None:
        try:
            forced = [
                int(token)
                for token in encode(bridge, add_special_tokens=False)
            ]
        except TypeError:
            forced = [int(token) for token in encode(bridge)]
        except Exception:
            forced = []
    forced.append(close_id)
    from .loop_guard import tool_call_marker_ids

    tool_markers = tool_call_marker_ids(tokenizer)
    return ThinkingGuardConfig(
        enabled=True,
        think_open_token=open_id,
        think_close_token=close_id,
        budget_tokens=int(budget_tokens),
        forced_close_ids=tuple(forced),
        starts_in_think=bool(starts_in_think),
        reentry_ban_tokens=max(0, _env_int("MTPLX_THINKING_REENTRY_BAN", 64)),
        mask_open_token=tool_markers[0] if tool_markers else None,
        mask_close_token=tool_markers[1] if tool_markers else None,
        novelty_close=bool(novelty_close),
        novelty_ngram=max(8, _env_int("MTPLX_THINKING_NOVELTY_NGRAM", 24)),
        novelty_occurrences=max(
            2, _env_int("MTPLX_THINKING_NOVELTY_OCCURRENCES", 3)
        ),
        novelty_window=max(512, _env_int("MTPLX_THINKING_NOVELTY_WINDOW", 4096)),
        novelty_min_tokens=max(
            256, _env_int("MTPLX_THINKING_NOVELTY_MIN_TOKENS", 1024)
        ),
        novelty_scan_interval=max(
            16, _env_int("MTPLX_THINKING_NOVELTY_SCAN_INTERVAL", 64)
        ),
    )


@dataclass
class _State:
    """Derived, replayable view of the committed completion tokens."""

    count: int = 0
    in_think: bool = False
    in_tool_span: bool = False
    think_tokens: int = 0
    think_block_tokens: list[int] = field(default_factory=list)
    natural_closes: int = 0
    reopens: int = 0


class ThinkingGuard:
    """Per-generation reasoning-budget state.

    Not thread safe; one instance per generation call. ``observe(tokens)``
    per step on the committed completion; ``overlay_for(working)`` per
    sampling position while ``steering_active``; ``summary()`` for stats.
    """

    def __init__(self, config: ThinkingGuardConfig) -> None:
        self.config = config
        self._state = _State(in_think=bool(config.starts_in_think))
        # engaged_at: committed length at which the forced-close sequence
        # begins (queue index = position - engaged_at).
        self.engaged: str | None = None
        self.engaged_at: int | None = None
        self.forced_done = False
        self.closed_at: int | None = None
        self.dormant = False
        self.forced_emitted = 0
        self.reentry_banned_positions = 0
        self._last_novelty_scan = -1

    # ------------------------------------------------------------------
    # Committed-stream tracking

    def _replay(self, tokens: Sequence[int], upto: int) -> _State:
        state = _State(in_think=bool(self.config.starts_in_think))
        self._consume(state, tokens, 0, upto)
        return state

    def _consume(
        self, state: _State, tokens: Sequence[int], start: int, end: int
    ) -> None:
        open_id = self.config.think_open_token
        close_id = self.config.think_close_token
        mask_open = self.config.mask_open_token
        mask_close = self.config.mask_close_token
        for index in range(start, end):
            token = int(tokens[index])
            if token == open_id and not state.in_tool_span:
                if state.count and not state.in_think:
                    state.reopens += 1
                state.in_think = True
            elif token == close_id and not state.in_tool_span:
                if state.in_think:
                    state.natural_closes += 1
                state.in_think = False
            elif state.in_think:
                state.think_tokens += 1
                state.think_block_tokens.append(token)
            if mask_open is not None and token == mask_open:
                state.in_tool_span = True
            elif mask_close is not None and token == mask_close:
                state.in_tool_span = False
            state.count = index + 1

    def observe(self, tokens: Sequence[int]) -> str | None:
        """Advance committed state; returns a transition marker or None.

        Markers: "budget_close_engaged", "novelty_close_engaged",
        "closed", "dormant".
        """
        config = self.config
        if not config.enabled or self.dormant:
            return None
        state = self._state
        count = len(tokens)
        if count < state.count:
            # Upstream trim (repetition-stop): rebuild the derived view.
            self._state = state = self._replay(tokens, count)
        elif count > state.count:
            self._consume(state, tokens, state.count, count)

        if self.engaged is not None and not self.forced_done:
            # Queue progress is measured directly on the committed stream.
            done = count - int(self.engaged_at or 0)
            self.forced_emitted = max(0, min(done, len(config.forced_close_ids)))
            if self.forced_emitted >= len(config.forced_close_ids):
                self.forced_done = True
                self.closed_at = count
                return "closed"
            return None
        if self.forced_done:
            if (
                self.closed_at is not None
                and count - self.closed_at >= config.reentry_ban_tokens
                and not state.in_think
            ):
                self.dormant = True
                return "dormant"
            if state.in_think:
                # Re-opened past the ban window with the budget already
                # spent: close again on sight.
                self.engaged = "budget"
                self.engaged_at = count
                self.forced_done = False
                self.forced_emitted = 0
                return "budget_close_engaged"
            return None
        if not state.in_think:
            return None
        if state.think_tokens >= config.budget_tokens > 0:
            self.engaged = "budget"
            self.engaged_at = count
            return "budget_close_engaged"
        if (
            config.novelty_close
            and state.think_tokens >= config.novelty_min_tokens
            and (
                self._last_novelty_scan < 0
                or state.think_tokens - self._last_novelty_scan
                >= config.novelty_scan_interval
            )
        ):
            self._last_novelty_scan = state.think_tokens
            if self._novelty_recurrence(state.think_block_tokens):
                self.engaged = "novelty"
                self.engaged_at = count
                return "novelty_close_engaged"
        return None

    def _novelty_recurrence(self, think_tokens: Sequence[int]) -> bool:
        config = self.config
        arr = np.asarray(think_tokens[-config.novelty_window :], dtype=np.int64)
        if arr.shape[0] < config.novelty_ngram * config.novelty_occurrences:
            return False
        shingles = np.lib.stride_tricks.sliding_window_view(
            arr, config.novelty_ngram
        )
        _, counts = np.unique(shingles, axis=0, return_counts=True)
        return int(counts.max()) >= config.novelty_occurrences

    # ------------------------------------------------------------------
    # Sampling-side interface

    @property
    def steering_active(self) -> bool:
        """True while any overlay may be emitted (routing gate)."""
        if not self.config.enabled or self.dormant:
            return False
        if self.engaged is not None and not self.forced_done:
            return True
        if self.forced_done:
            return True  # re-entry ban window
        return False

    def overlay_for(self, working: Sequence[int]) -> dict[int, float] | None:
        """Sparse raw-logit overlay for the next position after ``working``.

        Positive values are subtracted (ban), negative boost (force) —
        `apply_penalties_mlx` semantics. Returns None when inactive.
        """
        config = self.config
        if not config.enabled or self.dormant:
            return None
        if self.engaged is not None and not self.forced_done:
            index = len(working) - int(self.engaged_at or 0)
            if 0 <= index < len(config.forced_close_ids):
                return {int(config.forced_close_ids[index]): _FORCE_BOOST}
            if index >= len(config.forced_close_ids):
                return self._ban_overlay(working)
            return None
        if self.forced_done:
            return self._ban_overlay(working)
        return None

    def _ban_overlay(self, working: Sequence[int]) -> dict[int, float] | None:
        config = self.config
        open_id = config.think_open_token
        if open_id is None or config.reentry_ban_tokens <= 0:
            return None
        # Never ban inside a tool-call payload: literal "<think>" text in a
        # written file is content, not a reasoning marker.
        mask_open = config.mask_open_token
        mask_close = config.mask_close_token
        if mask_open is not None and mask_close is not None:
            in_span = False
            state = self._state
            # Committed prefix state is tracked; walk only the uncommitted
            # tail of ``working``.
            in_span = state.in_tool_span
            for token in working[state.count :]:
                token = int(token)
                if token == mask_open:
                    in_span = True
                elif token == mask_close:
                    in_span = False
            if in_span:
                return None
        self.reentry_banned_positions += 1
        return {int(open_id): _REENTRY_BAN}

    def summary(self) -> dict[str, object]:
        state = self._state
        return {
            "enabled": bool(self.config.enabled),
            "budget_tokens": int(self.config.budget_tokens),
            "think_tokens": int(state.think_tokens),
            "engaged": self.engaged,
            "forced_emitted": int(self.forced_emitted),
            "closed_at": self.closed_at,
            "dormant": bool(self.dormant),
            "natural_closes": int(state.natural_closes),
            "reopens": int(state.reopens),
            "novelty_close": bool(self.config.novelty_close),
            "reentry_banned_positions": int(self.reentry_banned_positions),
        }
