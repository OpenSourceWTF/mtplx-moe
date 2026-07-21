"""Unit tests for the Thinking Guard (mtplx/thinking_guard.py).

Pure-python state-machine tests — no MLX, no model. The guard's contract:
below budget it emits nothing (bit-exact decode); at budget it forces the
bridge + </think> sequence via sparse overlays; afterwards it bans <think>
re-entry for a window (tool-call spans exempt) and then goes dormant.
"""

from __future__ import annotations

import pytest

from mtplx.thinking_guard import (
    ThinkingGuard,
    ThinkingGuardConfig,
    think_marker_ids,
    thinking_guard_config_from_env,
)

THINK_OPEN = 100
THINK_CLOSE = 101
TOOL_OPEN = 102
TOOL_CLOSE = 103
BRIDGE = (7, 8, 9)


class _FakeTokenizer:
    """encode() that mirrors Qwen: markers are single dedicated ids."""

    _VOCAB = {
        "<think>": [THINK_OPEN],
        "</think>": [THINK_CLOSE],
        "<tool_call>": [TOOL_OPEN],
        "</tool_call>": [TOOL_CLOSE],
    }

    def encode(self, text, add_special_tokens=False):
        if text in self._VOCAB:
            return list(self._VOCAB[text])
        return [7, 8, 9]  # any prose encodes to the bridge stand-in


def _config(**overrides) -> ThinkingGuardConfig:
    base = dict(
        enabled=True,
        think_open_token=THINK_OPEN,
        think_close_token=THINK_CLOSE,
        budget_tokens=8,
        forced_close_ids=(*BRIDGE, THINK_CLOSE),
        starts_in_think=False,
        reentry_ban_tokens=4,
        mask_open_token=TOOL_OPEN,
        mask_close_token=TOOL_CLOSE,
    )
    base.update(overrides)
    return ThinkingGuardConfig(**base)


def _drive(guard: ThinkingGuard, tokens: list[int]) -> list[str]:
    """Feed tokens one commit at a time, honoring active forcing.

    While forcing is active, the committed token is whatever the overlay
    dictates (the loop's rejection/correction machinery guarantees this in
    the real decode path).
    """
    committed: list[int] = []
    transitions: list[str] = []
    queue = list(tokens)
    while queue:
        overlay = guard.overlay_for(committed)
        forced_token = None
        if overlay:
            boosts = [token for token, value in overlay.items() if value < 0]
            if boosts:
                forced_token = boosts[0]
        committed.append(forced_token if forced_token is not None else queue.pop(0))
        marker = guard.observe(committed)
        if marker is not None:
            transitions.append(marker)
    return transitions


def test_disabled_guard_never_steers():
    guard = ThinkingGuard(_config(enabled=False))
    assert guard.observe([THINK_OPEN, 1, 2, 3]) is None
    assert guard.overlay_for([THINK_OPEN, 1, 2, 3]) is None
    assert guard.steering_active is False


def test_below_budget_is_inert():
    guard = ThinkingGuard(_config(budget_tokens=100))
    tokens = [THINK_OPEN, *range(1, 50)]
    for i in range(1, len(tokens) + 1):
        assert guard.observe(tokens[:i]) is None
        assert guard.overlay_for(tokens[:i]) is None
    assert guard.steering_active is False
    assert guard.summary()["think_tokens"] == 49


def test_budget_close_forces_bridge_then_close_then_dormant():
    guard = ThinkingGuard(_config(budget_tokens=4, reentry_ban_tokens=3))
    # 1 open + 4 think tokens crosses the budget on observe.
    committed = [THINK_OPEN, 1, 2, 3, 4]
    assert guard.observe(committed) == "budget_close_engaged"
    assert guard.steering_active is True
    # Forced sequence: BRIDGE then close, one overlay per position.
    for expected in (*BRIDGE, THINK_CLOSE):
        overlay = guard.overlay_for(committed)
        assert overlay == {expected: pytest.approx(-1.0e4)}
        committed.append(expected)
        guard.observe(committed)
    assert guard.forced_done is True
    assert guard.summary()["closed_at"] == len(committed)
    # Ban window: <think> is banned (positive subtraction) for 3 tokens.
    overlay = guard.overlay_for(committed)
    assert overlay == {THINK_OPEN: pytest.approx(1.0e4)}
    committed += [11, 12, 13]
    marker = guard.observe(committed)
    assert marker == "dormant"
    assert guard.steering_active is False
    assert guard.overlay_for(committed) is None


def test_starts_in_think_counts_without_open_marker():
    guard = ThinkingGuard(_config(budget_tokens=3, starts_in_think=True))
    assert guard.observe([1, 2]) is None
    assert guard.observe([1, 2, 3]) == "budget_close_engaged"


def test_natural_close_before_budget_never_fires():
    guard = ThinkingGuard(_config(budget_tokens=5))
    tokens = [THINK_OPEN, 1, 2, THINK_CLOSE, *range(10, 40)]
    transitions = _drive(guard, tokens)
    assert transitions == []
    assert guard.summary()["natural_closes"] == 1
    assert guard.steering_active is False


def test_reopen_after_dormant_is_closed_again():
    guard = ThinkingGuard(_config(budget_tokens=2, reentry_ban_tokens=2))
    committed = [THINK_OPEN, 1, 2]
    assert guard.observe(committed) == "budget_close_engaged"
    for expected in (*BRIDGE, THINK_CLOSE):
        committed.append(expected)
        guard.observe(committed)
    committed += [21, 22]
    assert guard.observe(committed) == "dormant"
    # The model reopens past the ban window: dormant guards stay dormant
    # (the request-level budget already closed one segment; a fresh segment
    # after dormancy is intentionally out of scope for v1).
    committed += [THINK_OPEN, 31, 32, 33]
    assert guard.observe(committed) is None


def test_reopen_during_ban_window_is_reengaged():
    guard = ThinkingGuard(_config(budget_tokens=2, reentry_ban_tokens=50))
    committed = [THINK_OPEN, 1, 2]
    assert guard.observe(committed) == "budget_close_engaged"
    for expected in (*BRIDGE, THINK_CLOSE):
        committed.append(expected)
        guard.observe(committed)
    # Inside the ban window the overlay bans <think>; if the model still
    # lands one (e.g. a draft slipped through before the ban row), the guard
    # re-engages the forced close on sight.
    committed.append(THINK_OPEN)
    assert guard.observe(committed) == "budget_close_engaged"


def test_ban_suppressed_inside_tool_call_span():
    guard = ThinkingGuard(_config(budget_tokens=2, reentry_ban_tokens=100))
    committed = [THINK_OPEN, 1, 2]
    guard.observe(committed)
    for expected in (*BRIDGE, THINK_CLOSE):
        committed.append(expected)
        guard.observe(committed)
    # Outside a span: ban active.
    assert guard.overlay_for(committed) == {THINK_OPEN: pytest.approx(1.0e4)}
    # Inside a tool-call payload: literal "<think>" text is content.
    committed.append(TOOL_OPEN)
    guard.observe(committed)
    assert guard.overlay_for(committed) is None
    committed.append(TOOL_CLOSE)
    guard.observe(committed)
    assert guard.overlay_for(committed) == {THINK_OPEN: pytest.approx(1.0e4)}


def test_repetition_stop_trim_resyncs():
    guard = ThinkingGuard(_config(budget_tokens=50))
    tokens = [THINK_OPEN, *range(1, 20)]
    guard.observe(tokens)
    assert guard.summary()["think_tokens"] == 19
    trimmed = tokens[:8]
    guard.observe(trimmed)
    assert guard.summary()["think_tokens"] == 7


def test_novelty_close_fires_on_shingle_recurrence():
    config = _config(
        budget_tokens=100_000,
        novelty_close=True,
        novelty_ngram=8,
        novelty_occurrences=3,
        novelty_window=4096,
        novelty_min_tokens=64,
        novelty_scan_interval=16,
    )
    guard = ThinkingGuard(config)
    cycle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    committed = [THINK_OPEN]
    fired = None
    for _ in range(30):
        committed.extend(cycle)
        fired = guard.observe(committed)
        if fired is not None:
            break
    assert fired == "novelty_close_engaged"


def test_novelty_close_stays_quiet_on_fresh_content():
    config = _config(
        budget_tokens=100_000,
        novelty_close=True,
        novelty_ngram=8,
        novelty_occurrences=3,
        novelty_min_tokens=64,
        novelty_scan_interval=16,
    )
    guard = ThinkingGuard(config)
    committed = [THINK_OPEN, *range(1000, 1600)]
    for i in range(2, len(committed) + 1):
        assert guard.observe(committed[:i]) is None


def test_think_marker_ids_resolution():
    assert think_marker_ids(_FakeTokenizer()) == (THINK_OPEN, THINK_CLOSE)
    assert think_marker_ids(None) is None


def test_config_from_env_budget_override(monkeypatch):
    monkeypatch.setenv("MTPLX_THINKING_BUDGET", "512")
    config = thinking_guard_config_from_env(
        False, budget_tokens=3072, tokenizer=_FakeTokenizer()
    )
    assert config.enabled is True
    assert config.budget_tokens == 512
    monkeypatch.setenv("MTPLX_THINKING_BUDGET", "off")
    config = thinking_guard_config_from_env(
        True, budget_tokens=3072, tokenizer=_FakeTokenizer()
    )
    assert config.enabled is False


def test_config_from_env_builds_forced_sequence():
    config = thinking_guard_config_from_env(
        True,
        budget_tokens=1024,
        tokenizer=_FakeTokenizer(),
        starts_in_think=True,
    )
    assert config.enabled is True
    assert config.forced_close_ids[-1] == THINK_CLOSE
    assert len(config.forced_close_ids) > 1  # bridge + close
    assert config.starts_in_think is True
    assert config.mask_open_token == TOOL_OPEN


def test_config_without_markers_disables():
    class _NoMarkers:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]  # every marker splits

    config = thinking_guard_config_from_env(
        True, budget_tokens=1024, tokenizer=_NoMarkers()
    )
    assert config.enabled is False
