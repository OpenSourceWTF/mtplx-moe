"""Scoring-semantics tests for the WikiText c512 independent-chunk perplexity
mode, replicating ggml-org/llama.cpp perplexity.cpp perplexity() @ 56142c5f8.

Uses a deterministic fake runtime (no model) so the off-by-one / window / metric
/ BOS semantics are locked against regression.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_streamed_quality as C  # noqa: E402

VOCAB = 50


class _FakeRuntime:
    """forward_ar favours the TRUE next token input[p+1] with a large margin on
    the scored positions [first, n_ctx-2] and a small margin elsewhere, so a
    wrong window or misalignment shifts mean_nll detectably."""

    def __init__(self, n_ctx: int, tokenizer=None):
        self.n_ctx = n_ctx
        self.first = n_ctx // 2
        self.tokenizer = tokenizer
        self.seen_first_tokens: list[int] = []

    def make_cache(self):
        return object()

    def quality_input_array(self, ids):
        self.seen_first_tokens.append(int(ids[0]))
        return np.asarray([list(ids)], dtype=np.int64)

    def forward_ar(self, arr, *, cache=None):
        toks = list(np.asarray(arr)[0])
        n = len(toks)
        logits = np.zeros((1, n, VOCAB), dtype=np.float32)
        for p in range(n - 1):
            margin = 10.0 if (self.first <= p <= n - 2) else 1.0
            logits[0, p, toks[p + 1]] = margin
        return logits


def _nll_for_margin(margin: float) -> float:
    return math.log(math.exp(margin) + (VOCAB - 1)) - margin


def test_scored_count_alignment_and_metric():
    n_ctx, n_chunks = 8, 2
    tokens = [(i * 7 + 3) % VOCAB for i in range(n_ctx * n_chunks)]
    res = C.wikitext_c512_loss(
        _FakeRuntime(n_ctx), tokens, n_ctx=n_ctx, n_chunks=n_chunks,
        add_bos=False, bos_id=None,
    )
    scored_per_chunk = n_ctx - n_ctx // 2 - 1  # 3
    assert res["scored_token_count"] == n_chunks * scored_per_chunk == 6
    assert res["nominal_token_count"] == n_chunks * n_ctx == 16
    assert res["n_chunks_evaluated"] == n_chunks
    assert res["finite"] is True
    # every scored position has margin 10 -> mean_nll == _nll_for_margin(10)
    assert res["mean_nll"] == pytest.approx(_nll_for_margin(10.0), abs=1e-5)
    # metric definition: PPL == exp(mean_nll)
    assert res["perplexity"] == pytest.approx(math.exp(res["mean_nll"]), abs=1e-9)
    # window correctness: had the margin-1 (unscored) positions been included,
    # mean_nll would be pulled toward _nll_for_margin(1) ~ 2.95.
    assert res["mean_nll"] < 0.01
    assert len(res["per_chunk_perplexity"]) == n_chunks


def test_bos_overwrites_chunk_position_zero_when_add_bos():
    n_ctx, n_chunks, bos = 8, 2, 999
    tokens = [(i * 3 + 1) % VOCAB for i in range(n_ctx * n_chunks)]
    rt = _FakeRuntime(n_ctx)
    C.wikitext_c512_loss(
        rt, tokens, n_ctx=n_ctx, n_chunks=n_chunks, add_bos=True, bos_id=bos,
    )
    # each chunk's fed position-0 token must be BOS
    assert rt.seen_first_tokens == [bos, bos]


def test_no_bos_keeps_original_first_token():
    n_ctx, n_chunks = 8, 2
    tokens = [(i * 3 + 1) % VOCAB for i in range(n_ctx * n_chunks)]
    rt = _FakeRuntime(n_ctx)
    C.wikitext_c512_loss(
        rt, tokens, n_ctx=n_ctx, n_chunks=n_chunks, add_bos=False, bos_id=None,
    )
    assert rt.seen_first_tokens == [tokens[0], tokens[n_ctx]]


def test_n_chunk_capped_by_available_tokens():
    n_ctx = 8
    tokens = [(i * 3 + 1) % VOCAB for i in range(n_ctx * 3)]  # only 3 chunks
    res = C.wikitext_c512_loss(
        _FakeRuntime(n_ctx), tokens, n_ctx=n_ctx, n_chunks=128,
        add_bos=False, bos_id=None,
    )
    assert res["n_chunks_evaluated"] == 3


def test_insufficient_tokens_raises():
    with pytest.raises(ValueError):
        C.wikitext_c512_loss(
            _FakeRuntime(8), [1, 2, 3], n_ctx=8, n_chunks=1,
            add_bos=False, bos_id=None,
        )


def test_add_bos_requires_bos_id():
    with pytest.raises(ValueError):
        C.wikitext_c512_loss(
            _FakeRuntime(8), list(range(16)), n_ctx=8, n_chunks=2,
            add_bos=True, bos_id=None,
        )
