#!/usr/bin/env python3
"""Retired component-bank logical-M census for the rejected GLM Q1T route.

This file remains only to preserve the earlier experimental work. It observes
the reconstructed component-bank dispatcher and therefore cannot qualify the
fused rANS weight representation. Its CLI fails closed instead of restarting
the invalid benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from functools import wraps
import json
import os
from pathlib import Path
import sys
from typing import Any


SCHEMA = "mtplx-glm52-q1t-component-shape-census-v1"


class ComponentShapeCensus:
    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str, int, tuple[int, ...]]] = Counter()

    def observe(
        self,
        *,
        attention_phase: str,
        bank_label: str,
        slot_rows: tuple[int, ...],
    ) -> None:
        identities: dict[int, int] = {}
        normalized: list[int] = []
        for slot_row in slot_rows:
            normalized.append(identities.setdefault(int(slot_row), len(identities)))
        pattern = tuple(normalized)
        self._counts[
            (str(attention_phase), str(bank_label), len(pattern), pattern)
        ] += 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "records": [
                {
                    "attention_phase": attention_phase,
                    "bank_label": bank_label,
                    "logical_m": logical_m,
                    "slot_pattern": list(slot_pattern),
                    "dispatches": dispatches,
                }
                for (
                    attention_phase,
                    bank_label,
                    logical_m,
                    slot_pattern,
                ), dispatches in sorted(self._counts.items())
            ],
        }


def instrument_component_dispatch(
    original: Callable[..., Any],
    census: ComponentShapeCensus,
    *,
    phase_reader: Callable[[], str],
) -> Callable[..., Any]:
    """Wrap the diagnostic process's dispatcher without altering its result."""

    @wraps(original)
    def observed(instance, selected, bindings):
        bank = bindings[0].buffer.bank
        census.observe(
            attention_phase=phase_reader(),
            bank_label=str(getattr(bank, "label", type(bank).__name__)),
            slot_rows=tuple(int(binding.buffer.bank_index) for binding in bindings),
        )
        return original(instance, selected, bindings)

    return observed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run_census(
    *,
    benchmark_main: Callable[[list[str]], int],
    dispatcher_owner: type,
    phase_reader: Callable[[], str],
    benchmark_argv: list[str],
    output_path: Path,
) -> int:
    census = ComponentShapeCensus()
    original = dispatcher_owner._dispatch_component_bank
    dispatcher_owner._dispatch_component_bank = instrument_component_dispatch(
        original,
        census,
        phase_reader=phase_reader,
    )
    try:
        return int(benchmark_main(list(benchmark_argv)))
    finally:
        dispatcher_owner._dispatch_component_bank = original
        _write_json_atomic(output_path, census.to_json_dict())


def main(
    argv: list[str] | None = None,
    *,
    benchmark_main: Callable[[list[str]], int] | None = None,
    dispatcher_owner: type | None = None,
    phase_reader: Callable[[], str] | None = None,
) -> int:
    if benchmark_main is None and dispatcher_owner is None and phase_reader is None:
        raise RuntimeError(
            "retired GLM Q1T component-bank census: use the fused-rANS "
            "real-record microbenchmark"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-json", type=Path, required=True)
    parser.add_argument("benchmark_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    benchmark_argv = list(args.benchmark_argv)
    if benchmark_argv[:1] == ["--"]:
        benchmark_argv.pop(0)
    if not benchmark_argv:
        parser.error("benchmark arguments are required after --")

    if benchmark_main is None or dispatcher_owner is None or phase_reader is None:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from mtplx.attention_context import current_attention_phase
        from mtplx.models.expert_mlx import HotExpertSwitchGLU
        from scripts import benchmark_q2_mtp_depth_matrix

        benchmark_main = benchmark_q2_mtp_depth_matrix.main
        dispatcher_owner = HotExpertSwitchGLU
        phase_reader = current_attention_phase

    return run_census(
        benchmark_main=benchmark_main,
        dispatcher_owner=dispatcher_owner,
        phase_reader=phase_reader,
        benchmark_argv=benchmark_argv,
        output_path=args.census_json,
    )


__all__ = [
    "ComponentShapeCensus",
    "SCHEMA",
    "instrument_component_dispatch",
    "main",
    "run_census",
]


if __name__ == "__main__":
    raise SystemExit(main())
