"""Operations CLI command ownership and registration boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


COMMANDS = (
    "env",
    "doctor",
    "report",
    "profile",
    "thermal",
    "max",
    "debug",
    "metrics",
    "dashboard",
    "integrate",
)


@dataclass(frozen=True)
class OperationsGroupContext:
    default_model: str


def register_operations_commands(
    sub: argparse._SubParsersAction, context: OperationsGroupContext
) -> None:
    _ = context.default_model
    missing = tuple(command for command in COMMANDS if command not in sub.choices)
    if missing:
        raise RuntimeError(
            f"operations parser group did not register: {', '.join(missing)}"
        )
