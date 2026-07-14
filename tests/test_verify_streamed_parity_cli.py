"""Pin explicit streamed-artifact selection for the parity CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "verify_streamed_parity.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_streamed_parity", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model_key", ["hy3-expert-only-q4", "hy3-expert-q2"])
def test_parity_cli_accepts_explicit_hy3_expert_q2_and_expert_only_q4(
    model_key: str,
) -> None:
    args = (
        _load_module()
        .build_parser()
        .parse_args(
            [
                "/model",
                "/manifest",
                "/probes",
                "--model-key",
                model_key,
                "--memory-limit",
                "112GiB",
                "--max-live-kv-tokens",
                "2048",
            ]
        )
    )

    assert args.model_key == model_key
