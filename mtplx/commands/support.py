"""Compatibility boundary for support and diagnostics command handlers."""

from __future__ import annotations

# These aliases preserve the exact function objects while the tightly coupled
# support helper cluster is peeled out of the legacy warehouse incrementally.
from .public import (  # noqa: F401
    _hotpath_boundary_report,
    _redact_secret_value,
    cmd_debug_public,
    cmd_doctor,
    cmd_stop_public,
)

__all__ = ("cmd_debug_public", "cmd_doctor", "cmd_stop_public")
