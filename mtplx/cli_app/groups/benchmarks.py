"""Benchmark CLI command ownership and registration boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


COMMANDS = (
    "bench-preflight",
    "inspect-model",
    "bench",
    "qa",
    "runtime-smoke",
    "probe-contract",
    "verify-ratio",
    "verify-profile",
    "verify-qmm-probe",
    "multi-qmv-probe",
    "batch-equivalence",
    "capture-commit-equivalence",
    "mtp1-greedy-gate",
    "mtp1-sampler-smoke",
    "mtp-depth-sweep",
    "mtp-chain-probe",
    "mtp-tree-probe",
    "mtp-depth-grid",
    "mtp-adaptive",
    "dflash-mlx-baseline",
    "ddtree-mlx-baseline",
    "truth-report",
    "session-bank",
)


@dataclass(frozen=True)
class BenchmarkGroupContext:
    default_model: str


def register_benchmark_commands(
    sub: argparse._SubParsersAction, context: BenchmarkGroupContext
) -> None:
    _ = context.default_model
    missing = tuple(command for command in COMMANDS if command not in sub.choices)
    if missing:
        raise RuntimeError(
            f"benchmark parser group did not register: {', '.join(missing)}"
        )
