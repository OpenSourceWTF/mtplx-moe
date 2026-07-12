from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "midform_gate.py"


def _load_midform_gate():
    spec = importlib.util.spec_from_file_location("midform_gate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_by_depth_handles_missing_draft_rows():
    gate = _load_midform_gate()
    rates = gate._acceptance_by_depth(
        {"accepted_by_depth": [9, 4, 1], "drafted_by_depth": [10, 8]}
    )
    assert rates == [0.9, 0.5, None]


def test_window_rate_uses_requested_side():
    gate = _load_midform_gate()
    times = [0.0, 1.0, 2.0, 4.0, 8.0]
    assert gate._rate(times, first=True, window=3) == 1.0
    assert gate._rate(times, first=False, window=3) == 2.0 / 6.0


def test_parse_env_requires_key_value_pairs():
    gate = _load_midform_gate()
    assert gate._parse_env(["A=1", "B=two=parts"]) == {"A": "1", "B": "two=parts"}
