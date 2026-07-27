from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "diagnostics"
    / "glm52_t158_geometry_sweep.py"
)
_SPEC = importlib.util.spec_from_file_location("glm52_t158_geometry_sweep", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_geometry_sweep_ranks_against_the_unchanged_default() -> None:
    results = [
        {"threads": 128, "stage": False, "median_ms": 1.0},
        {"threads": 64, "stage": False, "median_ms": 0.8},
        {"threads": 256, "stage": True, "median_ms": 1.1},
    ]

    ranked = _MODULE._rank_results(results)

    assert [row["threads"] for row in ranked] == [64, 128, 256]
    assert ranked[0]["over_default"] == 0.8
    assert ranked[1]["over_default"] == 1.0
    assert ranked[2]["over_default"] == 1.1
