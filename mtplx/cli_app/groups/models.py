"""Model CLI command ownership and registration boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


COMMANDS = (
    "inspect",
    "forge",
    "init",
    "profiles",
    "pull",
    "list",
    "remove",
    "model",
    "config",
)


@dataclass(frozen=True)
class ModelGroupContext:
    default_model: str


def register_model_commands(
    sub: argparse._SubParsersAction, context: ModelGroupContext
) -> None:
    """Assert the compatibility registrations owned by this group are complete."""

    _ = context.default_model
    missing = tuple(command for command in COMMANDS if command not in sub.choices)
    if missing:
        raise RuntimeError(f"model parser group did not register: {', '.join(missing)}")
