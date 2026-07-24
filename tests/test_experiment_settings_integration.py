from __future__ import annotations

import pytest

from mtplx.cli import build_parser
from mtplx.settings.argparse import resolve_args_settings


def test_lab_uri_uses_same_bundle_precedence_as_file_bundle(tmp_path):
    args = build_parser().parse_args(
        [
            "serve",
            "--settings",
            "lab:compiled-verify-control",
            "--set",
            "verify.compiled.mode=on",
        ]
    )
    resolved = resolve_args_settings(
        args, environ={}, user_path=tmp_path / "missing.toml"
    )
    assert resolved["verify.compiled.mode"] == "on"
    assert resolved.bundle_provenance[0].id == "compiled-verify-control"


def test_lab_uri_refuses_incompatible_model_family(tmp_path):
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "gemma4/example",
            "--settings",
            "lab:compiled-verify-control",
        ]
    )
    with pytest.raises(ValueError, match="supports qwen3-next"):
        resolve_args_settings(
            args,
            environ={},
            user_path=tmp_path / "missing.toml",
            model_family="gemma4",
        )
