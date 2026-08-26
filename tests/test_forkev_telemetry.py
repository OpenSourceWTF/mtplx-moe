"""CPU unit tests for the fork-EV shadow telemetry accounting.

Feeds synthetic accept/reject round streams into ForkEVRecorder and asserts
the EV math end to end: margin extraction, hit detection, pending resolution
(saved_lo / saved_hi), the same-depth window cap, the first/at_rejection
policy split, exclusions, and finalize semantics. No model, no MLX evals.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mtplx.forkev_telemetry import (
    DEFAULT_THRESHOLDS,
    ForkEVRecorder,
    _decile,
)
from mtplx.sampling import SparseDistribution


def sparse(pairs: list[tuple[int, float]], vocab: int = 1000) -> SparseDistribution:
    ids = np.array([p[0] for p in pairs], dtype=np.int64)
    probs = np.array([p[1] for p in pairs], dtype=np.float64)
    return SparseDistribution(token_ids=ids, probs=probs, vocab_size=vocab)


def dist_with_margin(margin: float, top1: int = 1, top2: int = 2) -> SparseDistribution:
    """Two-token support whose normalized top-2 gap is exactly `margin`."""
    p1 = (1.0 + margin) / 2.0
    return sparse([(top1, p1), (top2, 1.0 - p1)])


# --------------------------------------------------------------- margin math


def test_margin_two_token_support() -> None:
    margins, top2 = ForkEVRecorder.margins_and_top2(
        [dist_with_margin(0.2, top1=7, top2=9)]
    )
    assert margins[0] == pytest.approx(0.2)
    assert top2[0] == 9


def test_margin_orders_by_probability_not_input_order() -> None:
    # top-1 listed second: argsort must still find the true #2 candidate.
    margins, top2 = ForkEVRecorder.margins_and_top2(
        [sparse([(5, 0.3), (6, 0.7)])]
    )
    assert margins[0] == pytest.approx(0.4)
    assert top2[0] == 5


def test_margin_single_support_is_fully_confident() -> None:
    margins, top2 = ForkEVRecorder.margins_and_top2([sparse([(3, 1.0)])])
    assert margins[0] == 1.0
    assert top2[0] is None


def test_margin_unavailable_for_none_and_dense() -> None:
    margins, top2 = ForkEVRecorder.margins_and_top2(
        [None, np.array([0.5, 0.5])]
    )
    assert margins == [None, None]
    assert top2 == [None, None]


def test_decile_edges() -> None:
    assert _decile(0.0) == 0
    assert _decile(0.09) == 0
    assert _decile(0.1) == 1
    assert _decile(0.999) == 9
    assert _decile(1.0) == 9


# ------------------------------------------------------- hit + resolution EV


def hit_round(
    rec: ForkEVRecorder,
    *,
    attempted: int,
    rejection_index: int,
    margin: float,
    accepted: int | None = None,
) -> None:
    """One round whose rejection at `rejection_index` hits the #2 candidate.

    All other positions carry margin 0.9 (never triggers any threshold).
    """
    probs = []
    for index in range(attempted):
        if index == rejection_index:
            probs.append(dist_with_margin(margin, top1=100 + index, top2=200 + index))
        else:
            probs.append(dist_with_margin(0.9, top1=100 + index, top2=200 + index))
    rec.observe_round(
        draft_probs=probs,
        attempted=attempted,
        accepted=rejection_index if accepted is None else accepted,
        rejection_index=rejection_index,
        correction=200 + rejection_index,  # == the #2 candidate -> hit
        clamped=False,
    )


def quiet_round(rec: ForkEVRecorder, *, attempted: int = 4, accepted: int = 4) -> None:
    probs = [dist_with_margin(0.9) for _ in range(attempted)]
    rec.observe_round(
        draft_probs=probs,
        attempted=attempted,
        accepted=accepted,
        rejection_index=None,
        correction=None,
        clamped=False,
    )


def test_hit_resolves_with_next_round_run_capped_at_window() -> None:
    rec = ForkEVRecorder()
    # K=4, rejection at depth 2 (index 1), margin 0.15 -> h = 4 - 2 = 2.
    hit_round(rec, attempted=4, rejection_index=1, margin=0.15)
    # Next round accepts a'=3 -> saved_lo = 1 + min(3, h-1) = 2,
    #                            saved_hi = 1 + min(3, h)   = 3.
    quiet_round(rec, attempted=4, accepted=3)
    snap = rec.snapshot()
    assert snap["rounds"] == 2
    assert snap["rejections"] == 1
    assert snap["hits"] == 1
    # 0.15 fires thresholds 0.2 / 0.3 / 0.5 and is the FIRST low-margin
    # position, so both policy variants credit it.
    by_t = {row["threshold"]: row for row in snap["policy"]}
    for threshold in (0.2, 0.3, 0.5):
        row = by_t[threshold]
        assert row["fired_rounds"] == 1
        assert row["first_fork_rejections"] == 1
        assert row["first_hits"] == 1
        assert row["first_saved_tokens_lo"] == 2
        assert row["first_saved_tokens_hi"] == 3
        assert row["first_ev_tokens_per_round_lo"] == pytest.approx(1.0)
        assert row["at_rejection_saved_tokens_lo"] == 2
        assert row["at_rejection_ev_tokens_per_round_hi"] == pytest.approx(1.5)
    for threshold in (0.05, 0.1):
        row = by_t[threshold]
        assert row["fired_rounds"] == 0
        assert row["first_fork_rejections"] == 0
        assert row["at_rejection_saved_tokens_lo"] == 0
    # Bins: rejection/hit/saved land in (depth 2, decile of 0.15 = 1).
    assert snap["bins"]["rejections"]["d2"][1] == 1
    assert snap["bins"]["hits"]["d2"][1] == 1
    assert snap["bins"]["saved_lo"]["d2"][1] == 2
    assert snap["bins"]["saved_hi"]["d2"][1] == 3
    # Extended-window estimator: 1 + a' = 4, uncapped by h.
    assert snap["bins"]["saved_ext"]["d2"][1] == 4
    assert by_t[0.2]["first_saved_tokens_ext"] == 4
    assert by_t[0.2]["at_rejection_ev_tokens_per_round_ext"] == pytest.approx(2.0)


def test_short_next_run_bounds_saved() -> None:
    rec = ForkEVRecorder()
    # K=5, rejection at depth 1 (index 0) -> h = 4.
    hit_round(rec, attempted=5, rejection_index=0, margin=0.05 - 1e-9)
    # a' = 1 -> saved_lo = 1 + min(1, 3) = 2, saved_hi = 1 + min(1, 4) = 2.
    quiet_round(rec, attempted=5, accepted=1)
    snap = rec.snapshot()
    by_t = {row["threshold"]: row for row in snap["policy"]}
    assert by_t[0.05]["first_saved_tokens_lo"] == 2
    assert by_t[0.05]["first_saved_tokens_hi"] == 2


def test_zero_next_run_still_earns_structural_token() -> None:
    rec = ForkEVRecorder()
    hit_round(rec, attempted=3, rejection_index=0, margin=0.15)  # h = 2
    quiet_round(rec, attempted=3, accepted=0)  # a' = 0
    snap = rec.snapshot()
    by_t = {row["threshold"]: row for row in snap["policy"]}
    # The row after the hit token is inside the verify forward: one committed
    # token needs no draft to be right. saved_lo = saved_hi = 1.
    assert by_t[0.2]["first_saved_tokens_lo"] == 1
    assert by_t[0.2]["first_saved_tokens_hi"] == 1


def test_deepest_position_hit_saves_nothing() -> None:
    rec = ForkEVRecorder()
    # K=3, rejection at depth 3 (index 2) -> h = 0: a same-depth tree has no
    # continuation slots there, so even a hit saves zero.
    hit_round(rec, attempted=3, rejection_index=2, margin=0.15)
    quiet_round(rec, attempted=3, accepted=3)
    snap = rec.snapshot()
    assert snap["hits"] == 1
    by_t = {row["threshold"]: row for row in snap["policy"]}
    assert by_t[0.2]["first_hits"] == 1
    assert by_t[0.2]["first_saved_tokens_lo"] == 0
    assert by_t[0.2]["first_saved_tokens_hi"] == 0
    assert snap["bins"]["saved_lo"]["d3"][1] == 0
    # The extended-window tree still prices the deepest fork: 1 + a' = 4.
    assert by_t[0.2]["first_saved_tokens_ext"] == 4
    assert snap["bins"]["saved_ext"]["d3"][1] == 4


def test_zero_draft_round_resolves_pending_with_zero_run() -> None:
    rec = ForkEVRecorder()
    hit_round(rec, attempted=3, rejection_index=0, margin=0.15)  # h = 2
    rec.observe_round(
        draft_probs=[],
        attempted=0,
        accepted=0,
        rejection_index=None,
        correction=None,
    )
    snap = rec.snapshot()
    assert snap["rounds"] == 2
    by_t = {row["threshold"]: row for row in snap["policy"]}
    assert by_t[0.2]["first_saved_tokens_lo"] == 1  # structural token only


def test_miss_rejection_never_creates_pending() -> None:
    rec = ForkEVRecorder()
    probs = [dist_with_margin(0.15, top1=1, top2=2)]
    rec.observe_round(
        draft_probs=probs,
        attempted=1,
        accepted=0,
        rejection_index=0,
        correction=777,  # neither candidate -> miss
    )
    quiet_round(rec, attempted=4, accepted=4)
    snap = rec.snapshot()
    assert snap["rejections"] == 1
    assert snap["hits"] == 0
    by_t = {row["threshold"]: row for row in snap["policy"]}
    assert by_t[0.2]["at_rejection_forks"] == 1  # fork spent, no benefit
    assert by_t[0.2]["at_rejection_hits"] == 0
    assert by_t[0.2]["at_rejection_saved_tokens_lo"] == 0


# ------------------------------------------------- policy variant semantics


def test_first_variant_requires_rejection_at_first_trigger() -> None:
    rec = ForkEVRecorder()
    # Positions: margins [0.15, 0.9, 0.15, 0.9]; rejection at index 2 (hit).
    probs = [
        dist_with_margin(0.15, top1=10, top2=20),
        dist_with_margin(0.9, top1=11, top2=21),
        dist_with_margin(0.15, top1=12, top2=22),
        dist_with_margin(0.9, top1=13, top2=23),
    ]
    rec.observe_round(
        draft_probs=probs,
        attempted=4,
        accepted=2,
        rejection_index=2,
        correction=22,  # hit at index 2
    )
    quiet_round(rec, attempted=4, accepted=4)  # a' = 4; h = 4 - 3 = 1
    snap = rec.snapshot()
    by_t = {row["threshold"]: row for row in snap["policy"]}
    row = by_t[0.2]
    # Single-fork-budget policy forked at index 0 (first trigger) — wasted.
    assert row["fired_rounds"] == 1
    assert row["first_fork_rejections"] == 0
    assert row["first_saved_tokens_lo"] == 0
    # Fork-at-every-trigger policy catches it: margin at the rejection < T.
    assert row["at_rejection_forks"] == 1
    assert row["at_rejection_hits"] == 1
    assert row["at_rejection_saved_tokens_lo"] == 1  # 1 + min(4, h-1=0)
    assert row["at_rejection_saved_tokens_hi"] == 2  # 1 + min(4, h=1)


def test_fired_counts_on_all_accept_rounds() -> None:
    rec = ForkEVRecorder()
    probs = [dist_with_margin(0.15), dist_with_margin(0.9)]
    rec.observe_round(
        draft_probs=probs,
        attempted=2,
        accepted=2,
        rejection_index=None,
        correction=None,
    )
    snap = rec.snapshot()
    by_t = {row["threshold"]: row for row in snap["policy"]}
    # The trigger cannot know verify outcomes at draft time: cost is real.
    assert by_t[0.2]["fired_rounds"] == 1
    assert by_t[0.2]["first_fork_rejections"] == 0
    assert snap["rejections"] == 0


# ------------------------------------------------------------- exclusions


def test_clamped_rejection_is_excluded() -> None:
    rec = ForkEVRecorder()
    probs = [dist_with_margin(0.15, top1=1, top2=2)]
    rec.observe_round(
        draft_probs=probs,
        attempted=1,
        accepted=0,
        rejection_index=0,
        correction=2,
        clamped=True,
    )
    snap = rec.snapshot()
    assert snap["rejections"] == 1
    assert snap["clamped_rejections"] == 1
    assert snap["hits"] == 0
    assert snap["bins"]["rejections"] == {}
    by_t = {row["threshold"]: row for row in snap["policy"]}
    assert by_t[0.2]["at_rejection_forks"] == 0


def test_margin_unavailable_rejection_is_counted_not_binned() -> None:
    rec = ForkEVRecorder()
    rec.observe_round(
        draft_probs=[None, None],
        attempted=2,
        accepted=0,
        rejection_index=0,
        correction=5,
    )
    snap = rec.snapshot()
    assert snap["rejections"] == 1
    assert snap["margin_unavailable_rejections"] == 1
    assert snap["bins"]["rejections"] == {}


def test_stop_break_round_is_not_a_rejection() -> None:
    rec = ForkEVRecorder()
    # accepted < attempted but no rejection (accepted stop token broke the
    # loop): rejection_index None must not fabricate a rejection.
    probs = [dist_with_margin(0.9) for _ in range(3)]
    rec.observe_round(
        draft_probs=probs,
        attempted=3,
        accepted=1,
        rejection_index=None,
        correction=None,
    )
    snap = rec.snapshot()
    assert snap["rounds"] == 1
    assert snap["rejections"] == 0


# --------------------------------------------------------------- finalize


def test_finalize_resolves_dangling_pending_to_zero() -> None:
    rec = ForkEVRecorder()
    hit_round(rec, attempted=4, rejection_index=1, margin=0.15)  # h = 2
    rec.finalize()
    snap = rec.snapshot()
    assert snap["hits"] == 1
    assert snap["pending_unresolved"] == 1
    by_t = {row["threshold"]: row for row in snap["policy"]}
    # The stream ended: nothing followed to save. Hit stays counted.
    assert by_t[0.2]["first_hits"] == 1
    assert by_t[0.2]["first_saved_tokens_lo"] == 0
    assert by_t[0.2]["first_saved_tokens_ext"] == 0
    assert snap["bins"]["saved_lo"]["d2"][1] == 0
    assert snap["bins"]["saved_ext"]["d2"][1] == 0


def test_unconditioned_anchor_reproduces_delta_baseline_shape() -> None:
    rec = ForkEVRecorder()
    # Two margined rejections: depth-2 hit (h=2, a'=3) and depth-1 miss.
    hit_round(rec, attempted=4, rejection_index=1, margin=0.15)
    quiet_round(rec, attempted=4, accepted=3)  # resolves lo=2 hi=3 ext=4
    probs = [dist_with_margin(0.9, top1=1, top2=2)]
    rec.observe_round(
        draft_probs=probs,
        attempted=1,
        accepted=0,
        rejection_index=0,
        correction=999,  # miss
    )
    snap = rec.snapshot()
    anchor = snap["unconditioned"]
    assert anchor["rejections_with_margin"] == 2
    assert anchor["hits"] == 1
    assert anchor["hit_rate"] == pytest.approx(0.5)
    assert anchor["hit_rate_by_depth"] == {
        "d1": pytest.approx(0.0),
        "d2": pytest.approx(1.0),
    }
    assert anchor["saved_tokens_lo"] == 2
    assert anchor["saved_tokens_ext"] == 4
    assert anchor["ev_tokens_per_round_lo"] == pytest.approx(2 / 3)


def test_finalize_is_idempotent() -> None:
    rec = ForkEVRecorder()
    hit_round(rec, attempted=4, rejection_index=1, margin=0.15)
    rec.finalize()
    rec.finalize()
    assert rec.snapshot()["pending_unresolved"] == 1


# ------------------------------------------------------------- aggregates


def test_ev_denominator_is_all_observed_rounds() -> None:
    rec = ForkEVRecorder()
    hit_round(rec, attempted=4, rejection_index=1, margin=0.15)  # h = 2
    quiet_round(rec, attempted=4, accepted=3)  # resolves: lo=2, hi=3
    quiet_round(rec)
    quiet_round(rec)
    snap = rec.snapshot()
    assert snap["rounds"] == 4
    by_t = {row["threshold"]: row for row in snap["policy"]}
    assert by_t[0.2]["first_ev_tokens_per_round_lo"] == pytest.approx(2 / 4)
    assert by_t[0.2]["first_ev_tokens_per_round_hi"] == pytest.approx(3 / 4)


def test_position_census_bins_every_margined_position() -> None:
    rec = ForkEVRecorder()
    quiet_round(rec, attempted=3, accepted=3)
    snap = rec.snapshot()
    # margin 0.9 -> decile 9 at depths 1..3.
    for depth in (1, 2, 3):
        assert snap["bins"]["positions"][f"d{depth}"][9] == 1


def test_snapshot_is_json_serializable() -> None:
    rec = ForkEVRecorder()
    hit_round(rec, attempted=4, rejection_index=1, margin=0.15)
    quiet_round(rec, attempted=4, accepted=2)
    rec.finalize()
    payload = json.dumps(rec.snapshot())
    assert "at_rejection_ev_tokens_per_round_lo" in payload


def test_stderr_summary_mentions_every_threshold() -> None:
    rec = ForkEVRecorder()
    quiet_round(rec)
    line = rec.stderr_summary()
    assert line.startswith("[forkev-telemetry] rounds=1")
    for threshold in DEFAULT_THRESHOLDS:
        assert f"T={threshold:g}:" in line


# ------------------------------------------------------------------- env


def test_from_env_gate_and_threshold_override(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_FORKEV_TELEMETRY", raising=False)
    assert ForkEVRecorder.from_env() is None
    monkeypatch.setenv("MTPLX_FORKEV_TELEMETRY", "1")
    rec = ForkEVRecorder.from_env()
    assert rec is not None
    assert rec.thresholds == DEFAULT_THRESHOLDS
    monkeypatch.setenv("MTPLX_FORKEV_THRESHOLDS", "0.4,0.1")
    rec = ForkEVRecorder.from_env()
    assert rec is not None
    assert rec.thresholds == (0.1, 0.4)
    monkeypatch.setenv("MTPLX_FORKEV_THRESHOLDS", "garbage")
    rec = ForkEVRecorder.from_env()
    assert rec is not None
    assert rec.thresholds == DEFAULT_THRESHOLDS


def test_generation_stats_carries_forkev_field() -> None:
    from mtplx.generation import GenerationStats

    stats = GenerationStats(
        mode="mtpk", generated_tokens=0, elapsed_s=0.0, tok_s=0.0
    )
    assert stats.to_dict()["forkev"] == {}
