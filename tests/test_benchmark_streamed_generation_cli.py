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


_BASE_ARGS = ["/model", "/manifest", "--memory-limit", "112GiB",
              "--max-live-kv-tokens", "2048"]


def test_mtp_defaults_off_so_ar_runs_are_unchanged() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4"])
    assert args.enable_mtp is False
    assert args.mtp_artifacts is None


def test_enable_mtp_parses_with_artifacts_for_hy3() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--enable-mtp", "--mtp-artifacts", "/artifacts"])
    module.validate_mtp_flags(parser, args)
    assert args.enable_mtp is True
    assert str(args.mtp_artifacts) == "/artifacts"


def test_enable_mtp_requires_artifacts_and_hy3(capsys) -> None:
    import pytest

    module = _load_module()
    parser = module.build_parser()

    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--enable-mtp"])
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "--mtp-artifacts" in capsys.readouterr().err

    args = parser.parse_args([*_BASE_ARGS, "--model-key", "glm52-q4",
                              "--enable-mtp", "--mtp-artifacts", "/artifacts"])
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "hy3-q4" in capsys.readouterr().err

    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--mtp-artifacts", "/artifacts"])
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "--enable-mtp" in capsys.readouterr().err


def test_mtp_precision_defaults_bf16_and_requires_enable_mtp(capsys) -> None:
    import pytest

    module = _load_module()
    parser = module.build_parser()

    # Default resolves to bf16 (Forge contract section 6: quantized MTP heads
    # collapse acceptance).
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--enable-mtp", "--mtp-artifacts", "/artifacts"])
    module.validate_mtp_flags(parser, args)
    assert args.mtp_precision == "bf16"

    # q4 stays selectable.
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--enable-mtp", "--mtp-artifacts", "/artifacts",
                              "--mtp-precision", "q4"])
    module.validate_mtp_flags(parser, args)
    assert args.mtp_precision == "q4"

    # Unknown precisions are rejected at parse time.
    with pytest.raises(SystemExit):
        parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                           "--enable-mtp", "--mtp-artifacts", "/artifacts",
                           "--mtp-precision", "fp8"])
    capsys.readouterr()

    # The flag is meaningless without MTP.
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--mtp-precision", "q4"])
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "--enable-mtp" in capsys.readouterr().err


def test_mtp_rejects_concurrency(capsys) -> None:
    import pytest

    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4",
                              "--enable-mtp", "--mtp-artifacts", "/artifacts",
                              "--concurrency", "4"])
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "single-stream" in capsys.readouterr().err
