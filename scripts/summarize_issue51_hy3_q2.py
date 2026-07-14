#!/usr/bin/env python3
"""Validate and summarize checkpointed Issue #51 Hy3-Q2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.issue51 import (  # noqa: E402
    A1_CANDIDATES,
    CampaignCell,
    ScheduledRun,
    build_abba_schedule,
    decide_performance,
    pair_abba_rows,
    paired_decode_statistics,
    validate_a1_child,
)
from scripts.run_issue51_hy3_q2 import A1_SCHEMA  # noqa: E402


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be an array")
    return value


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_bound_child(
    entry: Mapping[str, Any],
    *,
    base_dir: Path,
    seen_paths: set[Path],
    expected_arm: str,
) -> tuple[Path, dict[CampaignCell, dict[str, float]]]:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact path must be a nonempty relative string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path escapes the campaign root: {raw_path}")
    path = (base_dir / relative).resolve()
    if not path.is_relative_to(base_dir) or not path.is_file() or path.is_symlink():
        raise ValueError(
            f"artifact is not a regular file under campaign root: {raw_path}"
        )
    if path in seen_paths:
        raise ValueError(f"duplicate artifact path: {raw_path}")
    seen_paths.add(path)
    if entry.get("arm") != expected_arm:
        raise ValueError(f"artifact candidate declaration disagrees for {raw_path}")
    declared_digest = entry.get("sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise ValueError(f"artifact digest is invalid for {raw_path}")
    observed_digest = _file_sha256(path)
    if observed_digest != declared_digest:
        raise ValueError(f"artifact digest disagrees for {raw_path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact payload is not an object: {raw_path}")
    return path, validate_a1_child(payload, arm=expected_arm)


def _scheduled_run(value: Mapping[str, Any], *, context: str) -> ScheduledRun:
    try:
        return ScheduledRun(
            index=value["index"],
            block=value["block"],
            arm=value["arm"],
            pair_slot=value["pair_slot"],
        )
    except KeyError as exc:
        raise ValueError(f"{context} is missing {exc.args[0]}") from exc


def summarize_a1_index(index: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Validate a full A1 index, all raw digests, and every paired decision."""

    root = base_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"campaign root is not a directory: {root}")
    campaign = _mapping(index, context="campaign index")
    if campaign.get("schema") != A1_SCHEMA or campaign.get("stage") != "a1":
        raise ValueError("campaign index has the wrong A1 schema or stage")
    if campaign.get("status") != "passed":
        raise ValueError("campaign index must have passed status")
    configuration = _mapping(campaign.get("configuration"), context="configuration")
    if configuration.get("contexts") != [1024, 2048]:
        raise ValueError("campaign contexts must be exactly [1024, 2048]")
    if configuration.get("depths") != [1, 2]:
        raise ValueError("campaign depths must be exactly [1, 2]")
    if configuration.get("output_tokens") != 128:
        raise ValueError("campaign output token count must be exactly 128")
    retained_pairs = configuration.get("retained_pairs")
    if isinstance(retained_pairs, bool) or not isinstance(retained_pairs, int):
        raise ValueError("campaign retained_pairs must be an integer")

    seen_paths: set[Path] = set()
    qualifications = _sequence(campaign.get("qualifications"), context="qualifications")
    if len(qualifications) != len(A1_CANDIDATES):
        raise ValueError("A1 qualification must contain all four candidates once")
    qualified = []
    for arm, raw_entry in zip(A1_CANDIDATES, qualifications, strict=True):
        entry = _mapping(raw_entry, context=f"qualification {arm}")
        _load_bound_child(
            entry,
            base_dir=root,
            seen_paths=seen_paths,
            expected_arm=arm,
        )
        qualified.append(arm)

    comparisons_summary = []
    comparisons = _sequence(campaign.get("comparisons"), context="comparisons")
    if not comparisons:
        raise ValueError("A1 campaign has no performance comparisons")
    names: set[str] = set()
    for comparison_index, raw_comparison in enumerate(comparisons):
        comparison = _mapping(raw_comparison, context=f"comparison {comparison_index}")
        name = comparison.get("name")
        control = comparison.get("control")
        candidate = comparison.get("candidate")
        if not all(
            isinstance(item, str) and item for item in (name, control, candidate)
        ):
            raise ValueError(
                "comparison name/control/candidate must be nonempty strings"
            )
        if name in names:
            raise ValueError(f"duplicate comparison name: {name}")
        names.add(name)
        expected_schedule = build_abba_schedule(
            control=control,
            candidate=candidate,
            retained_pairs=retained_pairs,
        )
        raw_schedule = _sequence(comparison.get("schedule"), context=f"{name} schedule")
        schedule = tuple(
            _scheduled_run(
                _mapping(row, context=f"{name} schedule row {index}"),
                context=f"{name} schedule row {index}",
            )
            for index, row in enumerate(raw_schedule)
        )
        if schedule != expected_schedule:
            raise ValueError(f"{name} schedule disagrees with fixed ABBA schedule")
        pairs = pair_abba_rows(schedule)
        raw_artifacts = _sequence(
            comparison.get("artifacts"), context=f"{name} artifacts"
        )
        if len(raw_artifacts) != len(schedule):
            raise ValueError(f"{name} artifacts do not complete the ABBA schedule")
        metrics_by_index: dict[int, dict[CampaignCell, dict[str, float]]] = {}
        for row, raw_entry in zip(schedule, raw_artifacts, strict=True):
            entry = _mapping(raw_entry, context=f"{name} artifact {row.index}")
            expected_metadata = {
                "schedule_index": row.index,
                "block": row.block,
                "pair_slot": row.pair_slot,
            }
            if any(entry.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError(f"{name} artifact schedule metadata disagrees")
            _path, metrics = _load_bound_child(
                entry,
                base_dir=root,
                seen_paths=seen_paths,
                expected_arm=row.arm,
            )
            metrics_by_index[row.index] = metrics

        cell_summaries = []
        cells = sorted(next(iter(metrics_by_index.values())))
        for cell in cells:
            paired_rows = []
            for control_row, candidate_row in pairs:
                control_metrics = metrics_by_index[control_row.index][cell]
                candidate_metrics = metrics_by_index[candidate_row.index][cell]
                paired_rows.append(
                    {
                        "control_decode_tok_s": control_metrics["decode_tok_s"],
                        "candidate_decode_tok_s": candidate_metrics["decode_tok_s"],
                        "control_end_to_end_tok_s": control_metrics["end_to_end_tok_s"],
                        "candidate_end_to_end_tok_s": candidate_metrics[
                            "end_to_end_tok_s"
                        ],
                    }
                )
            statistics = paired_decode_statistics(paired_rows)
            cell_summaries.append(
                {
                    "cell": asdict(cell),
                    "statistics": statistics,
                    "decision": decide_performance(statistics),
                }
            )
        comparison_promotes = all(
            cell["decision"]["promote"] for cell in cell_summaries
        )
        comparisons_summary.append(
            {
                "name": name,
                "control": control,
                "candidate": candidate,
                "retained_pairs": retained_pairs,
                "cells": cell_summaries,
                "decision": {
                    "promote": comparison_promotes,
                    "reason": (
                        "all fixed cells passed"
                        if comparison_promotes
                        else "one or more fixed cells failed"
                    ),
                },
            }
        )
    return {
        "schema": "mtplx-issue51-a1-summary-v1",
        "qualification": {"passed": True, "candidates": qualified},
        "comparisons": comparisons_summary,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "## A1 correctness",
        "",
        "Passed: all four process-isolated candidates qualified on 1024/2048 x D1/D2 with 128 output tokens.",
        "",
        "Candidates: " + ", ".join(summary["qualification"]["candidates"]),
        "",
        "## A1 performance",
        "",
    ]
    for comparison in summary["comparisons"]:
        decision = "GO" if comparison["decision"]["promote"] else "NO-GO"
        lines.append(
            f"- {comparison['control']} vs {comparison['candidate']}: {decision}"
        )
        for cell in comparison["cells"]:
            identity = cell["cell"]
            statistics = cell["statistics"]
            interval = statistics["bootstrap_95_interval"]
            end_to_end = statistics["end_to_end_bootstrap_95_interval"]
            lines.append(
                "  - "
                f"{identity['context_tokens']} / D{identity['depth']}: "
                f"decode mean {statistics['mean_fractional_decode_gain'] * 100:.2f}% "
                f"(95% CI {interval[0] * 100:.2f}% to {interval[1] * 100:.2f}%); "
                f"end-to-end CI {end_to_end[0] * 100:.2f}% to {end_to_end[1] * 100:.2f}%"
            )
    return "\n".join(lines) + "\n"


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    index = _load_json_object(input_path)
    summary = summarize_a1_index(index, base_dir=input_path.parent)
    _write_json_exclusive(args.output_json, summary)
    print(render_markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
