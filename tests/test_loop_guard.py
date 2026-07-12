"""Loop Guard: detector, DRY steering, exactness-when-disarmed, env plumbing."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.fast_sampling import (
    apply_penalties_mlx,
    sparse_distribution_from_mlx_logits,
)
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.loop_guard import LoopGuard, LoopGuardConfig, loop_guard_config_from_env
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig, apply_penalties, distribution_from_logits


def _config(**overrides) -> LoopGuardConfig:
    base = dict(
        enabled=True,
        scan_interval=4,
        window=512,
        ngram=8,
        arm_occurrences=3,
        min_tokens=16,
        allowed_length=6,
        min_distinct=4,
        penalty=2.0,
        growth=1.2,
        penalty_cap=16.0,
        max_candidates=32,
        disarm_after=32,
    )
    base.update(overrides)
    return LoopGuardConfig(**base)


def _looping_tokens(block: list[int], repeats: int, prefix: list[int] | None = None) -> list[int]:
    return list(prefix or []) + block * repeats


# --- arming detector ---


def test_guard_stays_quiet_on_fresh_text():
    guard = LoopGuard(_config())
    fresh = list(range(400))  # no shingle ever repeats
    assert guard.observe(fresh) is None
    assert not guard.armed


def test_guard_arms_on_repeated_shingle():
    guard = LoopGuard(_config())
    block = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    loop = _looping_tokens(block, repeats=6, prefix=list(range(100)))
    transition = guard.observe(loop)
    assert transition == "armed"
    assert guard.armed
    assert guard.arm_events == 1


def test_guard_respects_min_tokens_and_scan_interval():
    guard = LoopGuard(_config(min_tokens=1000))
    block = [1, 2, 3, 4, 5, 6, 7, 8]
    assert guard.observe(_looping_tokens(block, repeats=10)) is None
    assert not guard.armed


def test_guard_disarms_after_quiet_period():
    guard = LoopGuard(_config(disarm_after=16))
    block = [7, 3, 9, 4, 11, 5, 13, 6]
    loop = _looping_tokens(block, repeats=8)
    assert guard.observe(loop) == "armed"
    # No penalties fire; the completion grows past the hysteresis window.
    grown = loop + list(range(1000, 1000 + 20))
    assert guard.observe(grown) == "disarmed"
    assert not guard.armed
    assert guard.disarm_events == 1


def test_disabled_guard_never_arms():
    guard = LoopGuard(LoopGuardConfig(enabled=False))
    block = [1, 2, 3, 4, 5, 6, 7, 8]
    assert guard.observe(_looping_tokens(block, repeats=20)) is None
    assert not guard.armed
    assert guard.penalties_for(_looping_tokens(block, repeats=20)) is None


# --- DRY steering ---


def test_penalties_target_the_loop_continuation_token():
    guard = LoopGuard(_config())
    block = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    loop = _looping_tokens(block, repeats=6)
    assert guard.observe(loop) == "armed"

    penalties = guard.penalties_for(loop)
    assert penalties is not None
    # The sequence ends with the full block; the loop continuation is the
    # block's first token.
    continuation = block[0]
    assert continuation in penalties
    assert penalties[continuation] > 0.0
    # Deep verbatim match saturates at the cap.
    assert penalties[continuation] == 16.0
    assert guard.penalized_positions == 1


def test_penalty_grows_with_match_length_and_respects_allowed_length():
    guard = LoopGuard(_config(allowed_length=6, penalty=2.0, growth=1.5, penalty_cap=1e9))
    prefix = list(range(100, 160))
    block = [1, 2, 3, 4, 5, 6, 7, 8]
    # Exactly two occurrences: history ...block...block-minus-last-token.
    working = prefix + block + [42, 43] + block[:-1]
    guard.armed = True  # steering unit test: bypass the arming detector
    penalties = guard.penalties_for(working)
    assert penalties is not None
    # Suffix (...,1..7) matches the earlier block occurrence for 7 tokens; the
    # continuation there was block[-1] == 8.
    assert set(penalties) == {8}
    assert penalties[8] == 2.0 * 1.5 ** (7 - 6)

    # Below allowed_length: no penalty at all.
    short = prefix + block + [42, 43] + block[:3]
    assert guard.penalties_for(short) is None


def test_no_penalty_without_earlier_occurrence():
    guard = LoopGuard(_config())
    guard.armed = True
    assert guard.penalties_for(list(range(300))) is None


def test_low_entropy_structured_runs_are_never_penalized():
    # A dash-row / divider pattern: one token repeated forever. It matches
    # verbatim at any length but carries < min_distinct distinct tokens.
    guard = LoopGuard(_config())
    guard.armed = True
    divider = [5] * 200
    assert guard.penalties_for(divider) is None
    # Two-token alternation (e.g. "- " cells) is protected too.
    table = [5, 6] * 100
    assert guard.penalties_for(table) is None


# --- tool-call span masking (2026-07-09 chess write-corruption fix) ---


OPEN, CLOSE = 9001, 9002


def _masked_config(**overrides) -> LoopGuardConfig:
    return _config(mask_open_token=OPEN, mask_close_token=CLOSE, **overrides)


def _tool_call(payload: list[int]) -> list[int]:
    return [OPEN, *payload, CLOSE]


def test_repetitive_tool_call_payload_never_arms():
    # The founder's failing turn: CSS-like repetition INSIDE a write call.
    guard = LoopGuard(_masked_config())
    block = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    completion = list(range(100, 140)) + _tool_call(block * 8)
    assert guard.observe(completion) is None
    assert not guard.armed


def test_same_repetition_outside_tool_call_still_arms():
    # Control: identical payload as prose must keep arming (the chess
    # marathon pathology stays guarded).
    guard = LoopGuard(_masked_config())
    block = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    completion = list(range(100, 140)) + block * 8
    assert guard.observe(completion) == "armed"


def test_prose_loop_straddling_tool_calls_still_arms():
    # A retry marathon whose repeated connective sentence lives BETWEEN tool
    # calls arms even though tool spans sit inside the window.
    guard = LoopGuard(_masked_config())
    sentence = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    completion: list[int] = list(range(100, 130))
    for _ in range(4):
        completion += sentence + _tool_call([41, 42, 43, 44, 45])
    assert guard.observe(completion) == "armed"


def test_no_steering_inside_tool_call_span():
    guard = LoopGuard(_masked_config())
    guard.armed = True
    block = [1, 2, 3, 4, 5, 6, 7, 8]
    # Working sequence is currently INSIDE an (unclosed) tool call whose
    # payload verbatim-extends an earlier payload occurrence.
    working = list(range(100, 140)) + [OPEN] + block + [9, 10] + block[:-1]
    assert guard.penalties_for(working) is None
    assert guard.span_suppressed_positions == 1
    # Identical shape without the marker steers (control).
    control = LoopGuard(_masked_config())
    control.armed = True
    working_prose = list(range(100, 140)) + block + [9, 10] + block[:-1]
    assert control.penalties_for(working_prose) is not None


def test_span_closes_and_steering_resumes_after_tool_call():
    guard = LoopGuard(_masked_config())
    guard.armed = True
    block = [1, 2, 3, 4, 5, 6, 7, 8]
    # The repeated block sits in prose AFTER a closed tool call.
    working = (
        list(range(100, 120))
        + _tool_call([61, 62, 63])
        + block
        + [9, 10]
        + block[:-1]
    )
    penalties = guard.penalties_for(working)
    assert penalties is not None
    assert set(penalties) == {8}


def test_matches_anchored_in_masked_span_do_not_steer_prose():
    guard = LoopGuard(_masked_config())
    guard.armed = True
    block = [1, 2, 3, 4, 5, 6, 7, 8]
    # Earlier occurrence lives INSIDE a tool call; the model then re-states
    # the same run in prose (e.g. quoting the file path it just wrote).
    working = list(range(100, 120)) + _tool_call(block) + [9, 10] + block[:-1]
    assert guard.penalties_for(working) is None


def test_marker_tokens_are_never_penalized():
    guard = LoopGuard(_masked_config())
    guard.armed = True
    # The loop's continuation token IS the open marker: a repeated
    # "prose sentence + <tool_call>" retry shape. The guard must not
    # suppress the model's ability to open the next tool call.
    sentence = [1, 2, 3, 4, 5, 6, 7]
    working = (
        list(range(100, 120))
        + sentence
        + [OPEN, 61, CLOSE]
        + [9, 10]
        + sentence
    )
    penalties = guard.penalties_for(working)
    assert penalties is None or OPEN not in penalties


def test_mask_resyncs_after_truncation():
    # repetition_stop trims committed tokens; the mask must follow exactly.
    # observe() runs every decode step, so the guard always sees the
    # truncated list before any regrowth.
    guard = LoopGuard(_masked_config(min_tokens=8, scan_interval=1))
    prefix = list(range(100, 130)) + [OPEN, 1, 2, 3]
    guard.observe(prefix)  # unclosed span at the tip
    trimmed = prefix[:-4]  # trim removes the OPEN marker too
    guard.observe(trimmed)
    assert guard._in_span_after[-1] is False  # span state resynced
    block = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    grown = trimmed + block * 8
    assert guard.observe(grown) == "armed"  # prose repetition arms again


def test_masking_disabled_config_is_byte_identical_to_legacy():
    legacy = LoopGuard(_config())
    unmasked = LoopGuard(_config(mask_open_token=None, mask_close_token=None))
    block = [7, 3, 9, 4, 11, 5, 13, 6, 17, 2]
    loop = _looping_tokens(block, repeats=6, prefix=list(range(100)))
    assert legacy.observe(loop) == unmasked.observe(loop) == "armed"
    assert legacy.penalties_for(loop) == unmasked.penalties_for(loop)


def test_marker_resolution_from_tokenizer(monkeypatch):
    from mtplx.loop_guard import tool_call_marker_ids

    class _QwenLike:
        def encode(self, text, add_special_tokens=True):
            table = {"<tool_call>": [248058], "</tool_call>": [248059]}
            return table.get(text, [1, 2, 3])

    assert tool_call_marker_ids(_QwenLike()) == (248058, 248059)

    class _NoMarkers:
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    assert tool_call_marker_ids(_NoMarkers()) is None
    assert tool_call_marker_ids(None) is None

    monkeypatch.setenv("MTPLX_LOOP_GUARD", "1")
    config = loop_guard_config_from_env(True, tokenizer=_QwenLike())
    assert config.mask_open_token == 248058
    assert config.mask_close_token == 248059

    monkeypatch.setenv("MTPLX_LOOP_GUARD_MASK_TOOL_CALLS", "0")
    config = loop_guard_config_from_env(True, tokenizer=_QwenLike())
    assert config.mask_open_token is None
    assert config.mask_close_token is None


# --- exactness plumbing ---


def test_apply_penalties_numpy_overlay_and_noop_identity():
    logits = np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float64)
    # No counts, no overlay: same object back (bit-exact no-op path).
    assert apply_penalties(logits, None) is logits
    assert apply_penalties(logits, None, penalty_overlay=None) is logits

    out = apply_penalties(logits, None, penalty_overlay={0: 10.0})
    assert out is not logits
    assert out[0] == -5.0
    assert np.array_equal(out[1:], logits[1:])


def test_apply_penalties_mlx_overlay_and_noop_identity():
    logits = mx.array([5.0, 4.0, 3.0, 2.0], dtype=mx.float32)
    assert apply_penalties_mlx(logits, None) is logits
    out = apply_penalties_mlx(logits, None, penalty_overlay={1: 3.0})
    mx.eval(out)
    assert float(out[1].item()) == 1.0
    assert float(out[0].item()) == 5.0


def test_sparse_distribution_overlay_removes_loop_token_from_support():
    # Token 0 dominates; a strong overlay must evict it from the sampled support.
    logits = mx.array([10.0, 2.0, 1.5, 1.0], dtype=mx.float32)
    config = SamplerConfig(temperature=0.6, top_p=0.95, top_k=2)
    base = sparse_distribution_from_mlx_logits(logits, config)
    assert base is not None and 0 in base.token_ids.tolist()

    steered = sparse_distribution_from_mlx_logits(
        logits, config, penalty_overlay={0: 16.0}
    )
    assert steered is not None
    assert 0 not in steered.token_ids.tolist()

    # Overlay=None keeps the distribution bit-identical to the base call.
    again = sparse_distribution_from_mlx_logits(logits, config, penalty_overlay=None)
    assert again is not None
    assert np.array_equal(again.token_ids, base.token_ids)
    assert np.array_equal(again.probs, base.probs)


def test_dense_distribution_overlay_matches_manual_subtraction():
    logits = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    config = SamplerConfig(temperature=0.6, top_p=1.0, top_k=0)
    steered = distribution_from_logits(logits, config, penalty_overlay={0: 2.5})
    manual = logits.copy()
    manual[0] -= 2.5
    expected = distribution_from_logits(manual, config)
    assert np.allclose(steered, expected)


# --- env plumbing ---


def test_env_kill_switch_and_defaults(monkeypatch):
    monkeypatch.delenv("MTPLX_LOOP_GUARD", raising=False)
    assert loop_guard_config_from_env(True).enabled
    assert not loop_guard_config_from_env(False).enabled

    monkeypatch.setenv("MTPLX_LOOP_GUARD", "0")
    assert not loop_guard_config_from_env(True).enabled

    monkeypatch.setenv("MTPLX_LOOP_GUARD", "1")
    config = loop_guard_config_from_env(False)
    assert config.enabled
    # Calibrated 2026-07-08 against nine captured loop transcripts: 12-token
    # x4 arming shingles (16/x3 missed long-period plan marathons), 12-token
    # steering threshold (chess loops cycle 8-18-token connective sentences).
    assert config.ngram == 12
    assert config.arm_occurrences == 4
    assert config.allowed_length == 12
    assert config.penalty == 3.0
    assert config.min_distinct == 4


def test_env_knobs_override(monkeypatch):
    monkeypatch.setenv("MTPLX_LOOP_GUARD", "1")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_ALLOWED_LENGTH", "32")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_PENALTY", "3.5")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_WINDOW", "4096")
    config = loop_guard_config_from_env(True)
    assert config.allowed_length == 32
    assert config.penalty == 3.5
    assert config.window == 4096


def test_summary_shape_is_json_primitive_only():
    guard = LoopGuard(_config())
    summary = guard.summary()
    assert set(summary) == {
        "enabled",
        "armed",
        "arm_events",
        "disarm_events",
        "penalized_positions",
        "max_match_len",
        "tool_call_mask",
        "span_suppressed_positions",
    }
    for value in summary.values():
        assert isinstance(value, (bool, int))


# --- generation integration (cyclic-automaton stub models) ---


VOCAB = 8
# Raw-logit margin for the scripted next token. At temp 0.6 the scripted
# token carries p ~ 0.9999999 — a hard verbatim loop absent intervention —
# while a saturated guard penalty (24 raw) decisively evicts it.
MARGIN = 10.0


class _CyclicTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class _CyclicModel:
    """Deterministic loop machine: after token t the model wants (t+1) % VOCAB."""

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits_for(self, last_tokens: list[int]) -> mx.array:
        rows = []
        for token in last_tokens:
            row = [0.0] * VOCAB
            row[(int(token) + 1) % VOCAB] = MARGIN
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        keep = len(tokens) if logits_keep is None else min(len(tokens), max(1, int(logits_keep)))
        logits = self._logits_for(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        if return_hidden:
            return logits, hidden
        return logits


class _CyclicMTPModel(_CyclicModel):
    """MTP sibling whose draft head follows the same cyclic script."""

    def __init__(self):
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits_for(tokens)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def _cyclic_runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_CyclicTokenizer(),
        model_path=Path("tiny-cyclic"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _pure_cycle(start: int, length: int) -> list[int]:
    return [(start + 1 + index) % VOCAB for index in range(length)]


def _set_guard_env(monkeypatch, enabled: bool) -> None:
    monkeypatch.setenv("MTPLX_LOOP_GUARD", "1" if enabled else "0")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_SCAN_INTERVAL", "8")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_NGRAM", "8")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_ARM_OCCURRENCES", "3")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_MIN_TOKENS", "32")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_ALLOWED_LENGTH", "6")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_PENALTY", "16.0")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_GROWTH", "1.5")
    monkeypatch.setenv("MTPLX_LOOP_GUARD_PENALTY_CAP", "24.0")


def test_generate_ar_without_guard_loops_forever(monkeypatch):
    _set_guard_env(monkeypatch, enabled=False)
    out = generate_ar(
        _cyclic_runtime(_CyclicModel()),
        [0],
        max_tokens=120,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=4),
        seed=7,
        stop_token_ids=set(),
        loop_guard=True,  # env kill-switch must win
    )
    assert list(out.tokens) == _pure_cycle(0, len(out.tokens))
    assert out.stats.loop_guard == {}


def test_generate_ar_guard_breaks_the_cycle(monkeypatch):
    _set_guard_env(monkeypatch, enabled=True)
    out = generate_ar(
        _cyclic_runtime(_CyclicModel()),
        [0],
        max_tokens=120,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=4),
        seed=7,
        stop_token_ids=set(),
        loop_guard=True,
    )
    summary = out.stats.loop_guard
    assert summary["enabled"] and summary["arm_events"] >= 1
    assert summary["penalized_positions"] > 0
    assert list(out.tokens) != _pure_cycle(0, len(out.tokens))
    # Guard armed only after min_tokens: the head of the run is untouched.
    assert list(out.tokens[:16]) == _pure_cycle(0, 16)


def test_generate_mtpk_guard_breaks_the_cycle(monkeypatch):
    _set_guard_env(monkeypatch, enabled=True)
    out = generate_mtpk(
        _cyclic_runtime(_CyclicMTPModel()),
        [0],
        max_tokens=120,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=4),
        speculative_depth=2,
        seed=7,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        loop_guard=True,
    )
    summary = out.stats.loop_guard
    assert summary["enabled"] and summary["arm_events"] >= 1
    assert summary["penalized_positions"] > 0
    assert list(out.tokens) != _pure_cycle(0, len(out.tokens))
    assert list(out.tokens[:16]) == _pure_cycle(0, 16)


def test_generate_mtpk_without_guard_stays_on_cycle_and_stats_empty(monkeypatch):
    _set_guard_env(monkeypatch, enabled=False)
    out = generate_mtpk(
        _cyclic_runtime(_CyclicMTPModel()),
        [0],
        max_tokens=120,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=4),
        speculative_depth=2,
        seed=7,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        loop_guard=True,
    )
    assert list(out.tokens) == _pure_cycle(0, len(out.tokens))
    assert out.stats.loop_guard == {}
