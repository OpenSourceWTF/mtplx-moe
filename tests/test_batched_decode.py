"""CPU tests for the Phase-1 multi-stream batched greedy decoder.

These exercise the DRIVER's batching bookkeeping (prefill batching, per-stream
argmax, [B,2] verify, uniform full-B repair + rollback, per-stream termination)
against a tiny FAKE runtime — deterministic per-row logits over a 64-token vocab,
tiny MLX tensors, no model.  The fake is ROW-ISOLATED by construction, so any
per-stream sha divergence between a batched run and the same stream run alone is
a driver bug (cross-stream contamination), which is exactly the Phase-1
correctness contract.  The real-model per-stream sha gate is fable-main's GPU
window (``a3b_174_batched_decode_bench.py``); it validates batch-numerical
invariance, which a CPU fake cannot.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.batched_decode import (
    BATCHED_DECODE_ENV,
    batched_decode_enabled,
    diff_streams,
    generate_greedy_batched,
    left_pad_prompts,
    streams_all_match,
    token_sha,
)

VOCAB = 64
HID = 4
STOP_ID = 63


class _FakeTrunkEntry:
    """One non-trimmable cache entry holding per-row token histories.

    Non-trimmable so ``snapshot_untrimmable_cache`` clones ``state`` and
    ``rollback_after_verify`` restores it — faithfully modelling the GDN
    recurrent snapshot/restore the driver relies on for its full-B repair.
    """

    def __init__(self) -> None:
        self.histories: list[list[int]] | None = None
        self.prompt_len: list[int] | None = None

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self) -> list[list[int]] | None:
        return self.histories

    @state.setter
    def state(self, value: list[list[int]] | None) -> None:
        self.histories = value


class _FakeRuntime:
    """Deterministic, row-isolated stand-in for MTPLXRuntime.

    ``forward_ar`` returns one-hot logits whose per-row/per-position argmax is a
    deterministic ``next_token`` of that row's cumulative history — so greedy
    decode is a fixed pseudo-sequence per row.  ``draft_mtp`` returns the exact
    correct next token for rows whose identity is NOT in ``broken_rids`` (forcing
    accept) and a wrong token otherwise (forcing the repair path).  Because the
    output is greedy, it is draft-independent: batched and single-stream agree
    regardless of which rows draft badly — that invariance is under test.
    """

    def __init__(
        self,
        *,
        broken_rids: set[int] | None = None,
        stop_at: dict[int, int] | None = None,
        seed: int = 7,
    ) -> None:
        self.mtp_enabled = True
        self.broken_rids = set(broken_rids or set())
        self.stop_at = dict(stop_at or {})
        self.seed = int(seed)
        self._trunk_entry: _FakeTrunkEntry | None = None

    # -- cache factories ---------------------------------------------------
    def make_cache(self) -> list[_FakeTrunkEntry]:
        entry = _FakeTrunkEntry()
        self._trunk_entry = entry
        return [entry]

    def make_mtp_cache(self) -> object:
        return object()

    # -- deterministic token model ----------------------------------------
    def _next_token(self, hist: list[int], prompt_len: int) -> int:
        rid = hist[0]
        generated = len(hist) - int(prompt_len)
        limit = self.stop_at.get(rid)
        if limit is not None and generated >= int(limit):
            return STOP_ID
        pseudo = (
            rid * 1000003 + sum(hist) * 7 + len(hist) * 13 + self.seed
        ) % (VOCAB - 2)
        return pseudo + 1  # in [1, VOCAB-2]; never STOP_ID or 0

    @staticmethod
    def _onehot(token: int) -> list[float]:
        row = [0.0] * VOCAB
        row[int(token)] = 10.0
        return row

    # -- forwards ----------------------------------------------------------
    def forward_ar(self, input_ids, cache, return_hidden: bool = False, **_kw):
        rows = input_ids.tolist()
        batch = len(rows)
        length = len(rows[0])
        entry = cache[0]
        if entry.histories is None:
            entry.histories = [[] for _ in range(batch)]
            entry.prompt_len = [length for _ in range(batch)]  # prefill = prompt
        logits: list[list[list[float]]] = []
        hidden: list[list[list[float]]] = []
        for b in range(batch):
            hist = list(entry.histories[b])
            row_logits: list[list[float]] = []
            row_hidden: list[list[float]] = []
            for i in range(length):
                hist.append(int(rows[b][i]))
                nxt = self._next_token(hist, entry.prompt_len[b])
                row_logits.append(self._onehot(nxt))
                row_hidden.append([float(len(hist))] * HID)
            entry.histories[b] = hist
            logits.append(row_logits)
            hidden.append(row_hidden)
        log = mx.array(logits)
        if return_hidden:
            return log, mx.array(hidden)
        return log

    def draft_mtp(self, hidden, next_token_ids, mtp_cache=None, **_kw):
        assert self._trunk_entry is not None and self._trunk_entry.histories is not None
        x0_rows = next_token_ids.tolist()
        batch = len(x0_rows)
        out: list[list[list[float]]] = []
        for b in range(batch):
            hist = list(self._trunk_entry.histories[b])
            x0 = int(x0_rows[b][-1])
            correct = self._next_token(hist + [x0], self._trunk_entry.prompt_len[b])
            rid = hist[0]
            if rid in self.broken_rids:
                wrong = correct + 1 if correct + 1 != STOP_ID else correct + 2
                token = wrong % VOCAB
            else:
                token = correct
            out.append([self._onehot(token)])
        return mx.array(out)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _distinct_prompts(batch: int, length: int = 3) -> list[list[int]]:
    # prompt[0] is a UNIQUE row identity (rid); rest fill the shared length.
    return [[10 + b] + [1 + ((b + j) % 5) for j in range(length - 1)] for b in range(batch)]


def _reference_single_stream(
    prompts: list[list[int]], *, max_new_tokens: int, stop_token_ids=None, **rt_kwargs
) -> list[list[int]]:
    """Run each prompt ALONE (B=1) through the same driver — the sha reference."""
    out: list[list[int]] = []
    for prompt in prompts:
        rt = _FakeRuntime(**rt_kwargs)
        res = generate_greedy_batched(
            rt, [prompt], max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids
        )
        out.append(res.streams[0].tokens)
    return out


# --------------------------------------------------------------------------- #
# Correctness: batched per-stream sha == single-stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [2, 4, 8])
def test_all_accept_matches_single_stream(batch: int) -> None:
    prompts = _distinct_prompts(batch)
    rt = _FakeRuntime()  # no broken rows -> every cycle all-accepts
    res = generate_greedy_batched(rt, prompts, max_new_tokens=16)
    ref = _reference_single_stream(prompts, max_new_tokens=16)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.repair_cycles == 0  # perfect drafts -> speculative fast path only
    assert res.all_accept_cycles > 0
    assert res.batch_size == batch
    # 2 tokens per cycle, all-accept => 1 forward per cycle.
    assert res.forwards == res.cycles


@pytest.mark.parametrize("batch", [2, 4])
def test_all_broken_matches_single_stream(batch: int) -> None:
    prompts = _distinct_prompts(batch)
    broken = {p[0] for p in prompts}
    rt = _FakeRuntime(broken_rids=broken)  # every row drafts wrong -> repair every cycle
    res = generate_greedy_batched(rt, prompts, max_new_tokens=16)
    ref = _reference_single_stream(prompts, max_new_tokens=16, broken_rids=broken)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.all_accept_cycles == 0
    assert res.repair_cycles == res.cycles
    assert res.forwards == 2 * res.cycles  # verify + repair each cycle


def test_mixed_accept_reject_isolation() -> None:
    """The load-bearing test: an accepting stream batched next to a rejecting
    one (which forces the full-B repair) must stay byte-identical to running it
    alone (where it would take the all-accept fast path).  Proves the repair
    re-forward never perturbs an accepting neighbour."""
    prompts = _distinct_prompts(6)
    broken = {prompts[b][0] for b in (1, 3, 5)}  # half draft badly
    rt = _FakeRuntime(broken_rids=broken)
    res = generate_greedy_batched(rt, prompts, max_new_tokens=24)
    # Reference: each row alone keeps its OWN broken-ness (keyed by rid), so a
    # "perfect" row runs all-accept alone yet must equal its batched (repaired)
    # self.  Output is greedy => draft-independent => must match either way.
    ref = _reference_single_stream(prompts, max_new_tokens=24, broken_rids=broken)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    assert res.repair_cycles > 0  # at least one stream rejected each cycle


def test_per_stream_stop_termination() -> None:
    prompts = _distinct_prompts(4)
    # Each row stops after a DIFFERENT number of generated tokens.
    stop_at = {prompts[b][0]: 3 + 4 * b for b in range(4)}  # 3, 7, 11, 15
    rt = _FakeRuntime(stop_at=stop_at)
    res = generate_greedy_batched(
        rt, prompts, max_new_tokens=64, stop_token_ids={STOP_ID}
    )
    ref = _reference_single_stream(
        prompts, max_new_tokens=64, stop_token_ids={STOP_ID}, stop_at=stop_at
    )
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    # Every stream stopped (not length/cap), at its own scheduled point.
    for b in range(4):
        stream = res.streams[b]
        assert stream.finish_reason == "stop", stream
        assert stream.tokens[-1] == STOP_ID
        # length == scheduled generated count (+1 for the committed stop token).
        assert len(stream.tokens) == stop_at[prompts[b][0]] + 1
    # Streams have genuinely different lengths (ragged termination).
    assert len({len(s.tokens) for s in res.streams}) == 4


def test_odd_max_new_tokens_final_single_commit() -> None:
    prompts = _distinct_prompts(3)
    rt = _FakeRuntime()
    res = generate_greedy_batched(rt, prompts, max_new_tokens=7)  # odd
    ref = _reference_single_stream(prompts, max_new_tokens=7)
    records = diff_streams([s.tokens for s in res.streams], ref)
    assert streams_all_match(records), records
    for stream in res.streams:
        assert len(stream.tokens) == 7
        assert stream.finish_reason == "length"


def test_output_is_draft_independent() -> None:
    """use_mtp_draft False (self-draft, mostly rejects) yields the SAME greedy
    sequence as the real MTP draft — confirms the sequence is a pure function of
    the prompt, not of speculation."""
    prompts = _distinct_prompts(4)
    with_draft = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=16, use_mtp_draft=True
    )
    without_draft = generate_greedy_batched(
        _FakeRuntime(), prompts, max_new_tokens=16, use_mtp_draft=False
    )
    records = diff_streams(
        [s.tokens for s in with_draft.streams],
        [s.tokens for s in without_draft.streams],
    )
    assert streams_all_match(records), records


# --------------------------------------------------------------------------- #
# Guards / validation
# --------------------------------------------------------------------------- #
def test_ragged_prompts_rejected() -> None:
    rt = _FakeRuntime()
    with pytest.raises(ValueError, match="share a length"):
        generate_greedy_batched(rt, [[1, 2, 3], [4, 5]], max_new_tokens=4)


def test_requires_mtp_runtime() -> None:
    rt = _FakeRuntime()
    rt.mtp_enabled = False
    with pytest.raises(RuntimeError, match="MTP-enabled"):
        generate_greedy_batched(rt, [[1, 2]], max_new_tokens=4)


def test_bad_max_tokens_rejected() -> None:
    rt = _FakeRuntime()
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_greedy_batched(rt, [[1, 2]], max_new_tokens=0)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_token_sha_stable_and_sensitive() -> None:
    assert token_sha([1, 2, 3]) == token_sha([1, 2, 3])
    assert token_sha([1, 2, 3]) != token_sha([1, 2, 4])
    assert len(token_sha([1, 2, 3])) == 16


def test_left_pad_prompts() -> None:
    padded, lengths = left_pad_prompts([[5, 6, 7], [8], [9, 10]], pad_id=0)
    assert lengths == [3, 1, 2]
    assert padded == [[5, 6, 7], [0, 0, 8], [0, 9, 10]]
    assert len({len(p) for p in padded}) == 1


def test_diff_streams_localizes_divergence() -> None:
    records = diff_streams([[1, 2, 3, 4]], [[1, 2, 9, 4]])
    assert records[0]["match"] is False
    assert records[0]["first_divergence"] == 2
    assert records[0]["batched_window"] == [1, 2, 3, 4]
    assert not streams_all_match(records)
    good = diff_streams([[1, 2]], [[1, 2]])
    assert streams_all_match(good)


def test_diff_streams_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="stream count"):
        diff_streams([[1]], [[1], [2]])


def test_batched_decode_enabled_failclosed() -> None:
    assert batched_decode_enabled({}) is False
    assert batched_decode_enabled({BATCHED_DECODE_ENV: "0"}) is False
    assert batched_decode_enabled({BATCHED_DECODE_ENV: "nope"}) is False
    for truthy in ("1", "true", "YES", "On"):
        assert batched_decode_enabled({BATCHED_DECODE_ENV: truthy}) is True
