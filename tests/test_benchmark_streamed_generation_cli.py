"""Pin the benchmark harness defaults that guarantee run comparability."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "benchmark_streamed_generation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_streamed_generation", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unflagged_runs_are_reproducible_and_bounded() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(["/model", "/manifest", "--model-key", "hy3-q4",
                              "--memory-limit", "112GiB",
                              "--max-live-kv-tokens", "2048"])
    # A run with no sampling/length flags must be deterministic and bounded:
    # silent default drift here is what makes old and new results
    # incomparable (review finding 9).
    assert args.generation_profile == "deterministic"
    assert args.max_tokens == 256
    assert args.window_telemetry is True
    assert args.window_tokens == 32
    assert args.seed == 0


def test_window_telemetry_can_be_disabled() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(["/model", "/manifest", "--model-key", "hy3-q4",
                              "--memory-limit", "112GiB",
                              "--max-live-kv-tokens", "2048",
                              "--no-window-telemetry"])
    assert args.window_telemetry is False
