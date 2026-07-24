"""Product CLI command ownership and registration boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


COMMANDS = (
    "hardware",
    "start",
    "setup",
    "status",
    "stop",
    "settings",
    "ask",
    "quickstart",
    "connect",
    "openwebui",
    "models",
)


@dataclass(frozen=True)
class ProductGroupContext:
    default_model: str


def register_product_commands(
    sub: argparse._SubParsersAction, context: ProductGroupContext
) -> None:
    """Assert the compatibility registrations owned by this group are complete."""

    _ = context.default_model
    missing = tuple(command for command in COMMANDS if command not in sub.choices)
    if missing:
        raise RuntimeError(f"product parser group did not register: {', '.join(missing)}")
