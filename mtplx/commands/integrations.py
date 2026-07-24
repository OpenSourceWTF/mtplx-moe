"""Compatibility boundary for client integration command handlers."""

from __future__ import annotations

from .public import (  # noqa: F401
    cmd_dashboard_public,
    cmd_integrate_public,
    cmd_openwebui_public,
)

__all__ = (
    "cmd_dashboard_public",
    "cmd_integrate_public",
    "cmd_openwebui_public",
)
