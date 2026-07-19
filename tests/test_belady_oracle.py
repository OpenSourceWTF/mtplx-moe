"""Runtime Belady oracle: exact clairvoyant floor, island exclusion, zero-cost-off."""

from __future__ import annotations

import os

from mtplx.belady_oracle import BeladyOracle, belady_fetches


def test_belady_fetches_matches_hand_computed_small_case() -> None:
    # 3 slots, a sequence where clairvoyance beats naive: keep the soon-reused.
    seq = [(1, 2, 3), (1, 2, 4), (1, 2, 3), (1, 2, 4)]
    # 3 slots hold {1,2}+one of {3,4}. Belady evicts the one whose next use is
    # farther. First step: fetch 1,2,3 (3). Step2 needs 4, evict 3 (next use t2)
    # vs... only 3 evictable (1,2 needed) -> fetch 4 (1). Step3 needs 3, evict 4
    # -> fetch 3 (1). Step4 needs 4 -> fetch 4 (1). Total 6.
    assert belady_fetches(seq, 3) == 6


def test_belady_never_exceeds_naive_and_zero_slack_is_all_uniques() -> None:
    seq = [(1, 2), (3, 4), (1, 2), (5, 6)]
    # capacity >= working set (6) -> only compulsory (first-touch) fetches = 6
    assert belady_fetches(seq, 6) == 6
    # capacity 0 -> every step's uniques fetched
    assert belady_fetches(seq, 0) == 8


def test_oracle_excludes_island_layers() -> None:
    oracle = BeladyOracle(island_layers=frozenset({5}))
    oracle.observe(5, [1, 2, 3])  # island -> ignored
    oracle.observe(7, [1, 2, 3])
    oracle.observe(7, [1, 2, 4])
    report = oracle.report(slots_by_layer=3)
    assert "5" not in report["per_layer"], "island layer must not carry an eviction ceiling"
    assert "7" in report["per_layer"]
    assert report["streamed_layers"] == 1


def test_oracle_report_shape_and_per_token_normalization() -> None:
    oracle = BeladyOracle()
    for step in [(1, 2, 3), (1, 2, 4), (1, 2, 3)]:
        oracle.observe(39, step)
    report = oracle.report(slots_by_layer={39: 3})
    layer = report["per_layer"]["39"]
    assert layer["steps"] == 3
    assert layer["unique_experts"] == 4
    assert layer["slots"] == 3
    # fetches: 3 (fetch 1,2,3) + 1 (fetch 4, evict 3) + 1 (fetch 3, evict 4) = 5
    assert layer["belady_fetches_per_token"] == round(5 / 3, 4)
    assert report["schema"] == "belady-oracle-v1"


def test_within_step_dedup() -> None:
    oracle = BeladyOracle()
    oracle.observe(1, [7, 7, 8])  # duplicate 7 in one step counts once
    report = oracle.report(slots_by_layer=4)
    assert report["per_layer"]["1"]["unique_experts"] == 2


def test_enabled_reads_env(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_BELADY_ORACLE", raising=False)
    assert BeladyOracle.enabled() is False
    monkeypatch.setenv("MTPLX_BELADY_ORACLE", "1")
    assert BeladyOracle.enabled() is True
