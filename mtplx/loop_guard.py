"""Loop Guard: loop-armed dynamic anti-repetition steering for live decode.

Why this exists (2026-07-08, chess execute-plan marathons): Qwen3.6-class
hybrid models can collapse into verbatim sentence/paragraph cycling during
long reasoning segments ("OK, I'm going to start creating files..." repeated
dozens of times). Qwen's own model card recommends presence_penalty 0-2
against "endless repetitions", but static presence penalties degrade coding
quality (they tax every reused identifier), so MTPLX refuses to ship one.

Loop Guard is the surgical alternative, DRY-style (see p-e-w's DRY sampler,
oobabooga/text-generation-webui#5677) but ARMED ONLY when a real loop is
detected, so normal decoding is bit-exact untouched:

1. ARMING DETECTOR (cheap, runs every ``scan_interval`` committed tokens):
   if any ``ngram``-token shingle occurs >= ``arm_occurrences`` times within
   the trailing ``window`` tokens of the completion, the guard arms.
2. STEERING (only while armed): for each sampling position, find earlier
   occurrences of the current suffix in the window. If the suffix extends a
   previous occurrence by >= ``allowed_length`` tokens verbatim, the token
   that CONTINUED that earlier occurrence gets a raw-logit penalty of
   ``penalty * growth ** (match_len - allowed_length)`` (capped at
   ``penalty_cap``) — small nudges at the threshold, a hard wall for deep
   cycles. Everything else in the distribution is untouched.
3. DISARM: after ``disarm_after`` tokens without a penalized position the
   guard disarms and decoding returns to the exact unpenalized path.

Exactness contract: while DISARMED the guard emits no penalties and callers
must keep their fast paths — output distributions are bit-identical to a
guardless run. While ARMED the penalty applies to the TARGET distribution
only (the MTP draft proposal q stays untouched: proposal mismatch only costs
acceptance, never correctness, per the Leviathan-Chen residual math).

TOOL-CALL SPAN MASKING (2026-07-09, chess "truncated" write corruption):
tool-call payloads are the one place verbatim repetition is *legitimate at
loop density* — CSS/code file contents, long absolute paths, repeated XML
parameter scaffolding. The founder's first execute-plan turn on the guarded
bundle armed the guard inside a 5.5 KB `write style.css` call and DRY
steering subtracted 3-16 raw logits from the model's own correct
continuations (`height: 64px;` …), corrupting the tool call: OpenCode saw
SchemaError(Missing key ["filePath"]) and the model spiralled into
"truncated" retries. Qwen's `<tool_call>`/`</tool_call>` markers are single
vocab tokens, so the guard tracks those spans exactly on the committed token
stream: shingles inside a span never arm the detector, positions inside a
span are never steered, and the marker tokens themselves are never
penalized (a legitimate re-attempt must always be able to open a tool
call). Reasoning/prose marathons — the pathology the guard exists for —
live outside these spans and stay fully guarded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .runtime_options import env_bool

__all__ = [
    "LoopGuard",
    "LoopGuardConfig",
    "loop_guard_config_from_env",
    "tool_call_marker_ids",
]


@dataclass(frozen=True)
class LoopGuardConfig:
    enabled: bool = False
    # Arming detector. ngram=12 / occurrences=4 calibrated 2026-07-08 against
    # nine captured loop transcripts (chess execute-plan marathons, q4+q8,
    # MTP+AR): every loop corpus arms 1-6k tokens into the marathon, healthy
    # tool-calling runs stay disarmed except stretches that ARE verbatim
    # repetition (a "recovered" run whose reasoning repeated one sentence 8x
    # arms too — correct: that is the pathology forming). Arming is benign by
    # design — steering only fires on >=allowed_length verbatim suffix
    # extensions — so the detector favors recall over precision.
    scan_interval: int = 16
    window: int = 2048
    ngram: int = 12
    arm_occurrences: int = 4
    min_tokens: int = 256
    # DRY-style steering. allowed_length=12 calibrated 2026-07-08: real chess
    # marathons repeat 8-18-token connective sentences ("OK, let me start
    # building now.") with fresh code between cycles, so a 20-token threshold
    # never fired. min_distinct guards structured runs (markdown dividers,
    # dash rows) from being steered — a qualifying match must contain at
    # least this many distinct token values.
    allowed_length: int = 12
    min_distinct: int = 4
    penalty: float = 3.0
    growth: float = 1.3
    penalty_cap: float = 16.0
    max_candidates: int = 32
    # Hysteresis.
    disarm_after: int = 256
    # Structured-span masking: single-token markers that open/close a span in
    # which verbatim repetition is legitimate (tool-call payloads). ``None``
    # disables masking; the guard is then byte-identical to the pre-masking
    # behavior. Resolved from the tokenizer by loop_guard_config_from_env.
    mask_open_token: int | None = None
    mask_close_token: int | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def tool_call_marker_ids(tokenizer: Any) -> tuple[int, int] | None:
    """Resolve single-token ``<tool_call>``/``</tool_call>`` ids, else None.

    Qwen-family tokenizers carry both markers as dedicated vocab entries; a
    tokenizer where either marker splits into multiple tokens (or that has no
    such markers at all) gets no masking rather than approximate masking.
    """
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

    open_id = _single_id("<tool_call>")
    close_id = _single_id("</tool_call>")
    if open_id is None or close_id is None or open_id == close_id:
        return None
    return open_id, close_id


def loop_guard_enabled_from_env(*, default: bool) -> bool:
    """The single parse of ``MTPLX_LOOP_GUARD``.

    ``default`` is the caller's product default (OFF on the serve path);
    the env var overrides in either direction. The server used to answer
    this question with a *lease-token* vocabulary that counted "none" and
    "unlimited" as off, so ``MTPLX_LOOP_GUARD=unlimited`` made the server
    report the guard disabled while this module built it enabled.
    """

    return env_bool("MTPLX_LOOP_GUARD", default=default)


def loop_guard_config_from_env(
    enabled: bool,
    *,
    tokenizer: Any = None,
) -> LoopGuardConfig:
    """Build the guard config; ``enabled`` is the caller's product default.

    ``MTPLX_LOOP_GUARD`` overrides in either direction ("0" kills the guard,
    "1" forces it on); the remaining knobs tune the detector/steering.
    ``tokenizer`` (when provided) resolves the tool-call marker tokens for
    span masking; ``MTPLX_LOOP_GUARD_MASK_TOOL_CALLS=0`` turns masking off.
    """
    enabled = loop_guard_enabled_from_env(default=enabled)
    if not enabled:
        return LoopGuardConfig(enabled=False)
    markers: tuple[int, int] | None = None
    mask_raw = os.environ.get("MTPLX_LOOP_GUARD_MASK_TOOL_CALLS", "").strip().lower()
    if mask_raw not in {"0", "false", "off", "no"}:
        markers = tool_call_marker_ids(tokenizer)
    return LoopGuardConfig(
        enabled=True,
        scan_interval=max(1, _env_int("MTPLX_LOOP_GUARD_SCAN_INTERVAL", 16)),
        window=max(64, _env_int("MTPLX_LOOP_GUARD_WINDOW", 2048)),
        ngram=max(4, _env_int("MTPLX_LOOP_GUARD_NGRAM", 12)),
        arm_occurrences=max(2, _env_int("MTPLX_LOOP_GUARD_ARM_OCCURRENCES", 4)),
        min_tokens=max(0, _env_int("MTPLX_LOOP_GUARD_MIN_TOKENS", 256)),
        allowed_length=max(4, _env_int("MTPLX_LOOP_GUARD_ALLOWED_LENGTH", 12)),
        min_distinct=max(1, _env_int("MTPLX_LOOP_GUARD_MIN_DISTINCT", 4)),
        penalty=max(0.0, _env_float("MTPLX_LOOP_GUARD_PENALTY", 3.0)),
        growth=max(1.0, _env_float("MTPLX_LOOP_GUARD_GROWTH", 1.3)),
        penalty_cap=max(0.0, _env_float("MTPLX_LOOP_GUARD_PENALTY_CAP", 16.0)),
        max_candidates=max(1, _env_int("MTPLX_LOOP_GUARD_MAX_CANDIDATES", 32)),
        disarm_after=max(16, _env_int("MTPLX_LOOP_GUARD_DISARM_AFTER", 256)),
        mask_open_token=markers[0] if markers else None,
        mask_close_token=markers[1] if markers else None,
    )


class LoopGuard:
    """Per-generation loop detector + DRY-style steering state.

    Not thread safe; one instance per generation call, driven from the decode
    loop: ``observe(tokens)`` once per step, ``penalties_for(working)`` per
    sampling position while ``armed``.
    """

    def __init__(self, config: LoopGuardConfig) -> None:
        self.config = config
        self.armed = False
        self.arm_events = 0
        self.disarm_events = 0
        self.penalized_positions = 0
        self.max_match_len = 0
        self.span_suppressed_positions = 0
        self.last_arm_token_count = 0
        self._last_scan_at = -1
        self._last_fire_at = 0
        # Tool-call span mask over the committed tokens. _mask[i] is True when
        # token i lies inside a masked span (markers included);
        # _in_span_after[i] is the span state after consuming token i, kept so
        # truncations (repetition-stop trims) can resync exactly.
        self._masking = (
            config.mask_open_token is not None
            and config.mask_close_token is not None
        )
        self._mask: list[bool] = []
        self._in_span_after: list[bool] = []

    def _advance_mask(self, tokens: Sequence[int]) -> None:
        """Extend (or resync after truncation) the committed span mask."""
        if not self._masking:
            return
        count = len(tokens)
        if count < len(self._mask):
            del self._mask[count:]
            del self._in_span_after[count:]
        open_id = self.config.mask_open_token
        close_id = self.config.mask_close_token
        in_span = self._in_span_after[-1] if self._in_span_after else False
        for index in range(len(self._mask), count):
            token = int(tokens[index])
            if token == open_id:
                in_span = True
                self._mask.append(True)
            elif token == close_id:
                self._mask.append(True)
                in_span = False
            else:
                self._mask.append(in_span)
            self._in_span_after.append(in_span)

    def _working_mask(self, working: Sequence[int], n: int) -> np.ndarray | None:
        """Span mask aligned with ``working[-n:]``.

        The committed prefix reuses the incrementally-maintained mask; any
        tail beyond it (this step's primary + in-block draft prefix) is
        walked locally without touching persistent state.
        """
        if not self._masking:
            return None
        total = len(working)
        known = min(len(self._mask), total)
        mask = self._mask[:known]
        if known < total:
            open_id = self.config.mask_open_token
            close_id = self.config.mask_close_token
            in_span = self._in_span_after[known - 1] if known else False
            tail: list[bool] = []
            for index in range(known, total):
                token = int(working[index])
                if token == open_id:
                    in_span = True
                    tail.append(True)
                elif token == close_id:
                    tail.append(True)
                    in_span = False
                else:
                    tail.append(in_span)
            mask = mask + tail
        return np.asarray(mask[-n:], dtype=bool)

    def observe(self, tokens: Sequence[int]) -> str | None:
        """Update armed state from the committed completion tokens.

        Returns "armed"/"disarmed" on a state change (for event logging),
        else None. Cheap: the shingle scan runs every ``scan_interval``
        tokens; between scans this is two integer comparisons plus an
        incremental span-mask extension over the newly committed tokens.
        """
        config = self.config
        if not config.enabled:
            return None
        self._advance_mask(tokens)
        count = len(tokens)
        if self.armed:
            anchor = max(self._last_fire_at, self.last_arm_token_count)
            if count - anchor >= config.disarm_after:
                self.armed = False
                self.disarm_events += 1
                return "disarmed"
            return None
        if count < config.min_tokens:
            return None
        if self._last_scan_at >= 0 and count - self._last_scan_at < config.scan_interval:
            return None
        self._last_scan_at = count
        if self._shingle_recurrence(tokens):
            self.armed = True
            self.arm_events += 1
            self.last_arm_token_count = count
            self._last_fire_at = count
            return "armed"
        return None

    def _shingle_recurrence(self, tokens: Sequence[int]) -> bool:
        config = self.config
        arr = np.asarray(tokens[-config.window :], dtype=np.int64)
        if arr.shape[0] < config.ngram * config.arm_occurrences:
            return False
        shingles = np.lib.stride_tricks.sliding_window_view(arr, config.ngram)
        if self._masking:
            # Shingles touching a tool-call span are legitimate structure
            # (file contents, paths, XML scaffolding) — they never arm.
            mask_arr = np.asarray(self._mask[-arr.shape[0] :], dtype=bool)
            unmasked = ~np.lib.stride_tricks.sliding_window_view(
                mask_arr, config.ngram
            ).any(axis=1)
            shingles = shingles[unmasked]
            if shingles.shape[0] < config.arm_occurrences:
                return False
        _, counts = np.unique(shingles, axis=0, return_counts=True)
        return int(counts.max()) >= config.arm_occurrences

    def penalties_for(self, working: Sequence[int]) -> dict[int, float] | None:
        """DRY penalties for the next position given ``working`` history.

        ``working`` is the full sampled-so-far sequence for this position
        (committed tokens plus any in-block draft prefix). Returns a mapping
        of token_id -> positive raw-logit subtraction, or None when nothing
        qualifies. Only call while ``armed``.
        """
        config = self.config
        if not self.armed or not config.enabled:
            return None
        arr = np.asarray(working[-config.window :], dtype=np.int64)
        n = int(arr.shape[0])
        if n < config.allowed_length + 1:
            return None
        mask_arr = self._working_mask(working, n)
        if mask_arr is not None and bool(mask_arr[-1]):
            # Inside a tool-call span: repetition here is payload (code,
            # paths, scaffolding), and steering it corrupts the tool call.
            self.span_suppressed_positions += 1
            return None
        last = int(arr[-1])
        # Earlier occurrences of the current last token that have a
        # continuation inside the window (position i, continuation i+1).
        candidates = np.nonzero(arr[: n - 2] == last)[0]
        if candidates.size == 0:
            return None
        if mask_arr is not None and candidates.size:
            # Matches anchored inside a masked span steer prose off legit
            # tool-call content (e.g. re-stating a written file's path);
            # drop candidates whose occurrence or continuation is masked.
            keep = ~(mask_arr[candidates] | mask_arr[candidates + 1])
            candidates = candidates[keep]
        if candidates.size == 0:
            return None
        if candidates.size > config.max_candidates:
            candidates = candidates[-config.max_candidates :]
        penalties: dict[int, float] = {}
        for i in candidates:
            i = int(i)
            span = min(i, n - 1)
            if span + 1 < config.allowed_length:
                continue
            a = arr[i - span : i][::-1]
            b = arr[n - 1 - span : n - 1][::-1]
            mismatch = np.nonzero(a != b)[0]
            match_len = (int(mismatch[0]) if mismatch.size else span) + 1
            if match_len < config.allowed_length:
                continue
            # Structured-run protection: dash rows, table borders, whitespace
            # runs and similar low-entropy spans repeat legitimately; a
            # qualifying loop match must carry real content.
            if (
                np.unique(arr[i - match_len + 1 : i + 1]).size
                < config.min_distinct
            ):
                continue
            continuation = int(arr[i + 1])
            value = min(
                config.penalty_cap,
                config.penalty
                * config.growth ** float(match_len - config.allowed_length),
            )
            if value <= 0.0:
                continue
            if value > penalties.get(continuation, 0.0):
                penalties[continuation] = value
            if match_len > self.max_match_len:
                self.max_match_len = match_len
        if self._masking:
            # A re-attempt must always be able to open (or close) a tool
            # call; the markers themselves are never steering targets.
            penalties.pop(int(self.config.mask_open_token or -1), None)
            penalties.pop(int(self.config.mask_close_token or -1), None)
        if not penalties:
            return None
        self.penalized_positions += 1
        # Track fire position in true-sequence coordinates (not the
        # window-clamped ``n``) so the disarm hysteresis in observe() works.
        self._last_fire_at = len(working)
        return penalties

    def summary(self) -> dict[str, object]:
        return {
            "enabled": bool(self.config.enabled),
            "armed": bool(self.armed),
            "arm_events": int(self.arm_events),
            "disarm_events": int(self.disarm_events),
            "penalized_positions": int(self.penalized_positions),
            "max_match_len": int(self.max_match_len),
            "tool_call_mask": bool(self._masking),
            "span_suppressed_positions": int(self.span_suppressed_positions),
        }
