from __future__ import annotations

import pytest

from mtplx.cli import build_parser, main
from mtplx.settings.argparse import resolve_args_settings


@pytest.mark.parametrize(
    ("command", "required"),
    [
        ("start", []),
        ("ask", []),
        ("run", []),
        ("chat", ["--prompt", "hello"]),
        ("serve", []),
        ("quickstart", []),
    ],
)
def test_runtime_command_accepts_generic_set(command, required):
    args = build_parser().parse_args(
        [
            command,
            *required,
            "--set",
            "runtime.profile=turbo",
            "--set",
            "generation.temperature=0.7",
        ]
    )
    resolved = resolve_args_settings(args, environ={})
    assert resolved["runtime.profile"] == "turbo"
    assert resolved["generation.temperature"] == 0.7
    assert args.profile == "turbo"
    assert args.temperature == 0.7


def test_explicit_generic_set_beats_legacy_flag():
    args = build_parser().parse_args(
        [
            "serve",
            "--profile",
            "sustained",
            "--set",
            "runtime.profile=turbo",
        ]
    )
    resolved = resolve_args_settings(args, environ={})
    assert resolved["runtime.profile"] == "turbo"
    assert resolved.provenance["runtime.profile"].source.name == "CLI_SET"


def test_settings_bundle_applies_without_mutating_user_config(tmp_path):
    bundle = tmp_path / "run.toml"
    bundle.write_text(
        '[settings]\n"runtime.mtp.depth" = 2\n', encoding="utf-8"
    )
    args = build_parser().parse_args(["run", "--settings", str(bundle)])
    resolve_args_settings(args, environ={}, user_path=tmp_path / "missing.toml")
    assert args.depth == 2
    assert not (tmp_path / "missing.toml").exists()


def test_generic_mtp_enabled_uses_inverted_legacy_namespace_destination(tmp_path):
    args = build_parser().parse_args(
        ["serve", "--set", "runtime.mtp.enabled=false"]
    )
    resolve_args_settings(args, environ={}, user_path=tmp_path / "missing.toml")
    assert args.no_mtp is True


def test_environment_alias_beats_user_setting(tmp_path):
    user = tmp_path / "config.toml"
    user.write_text(
        '[settings]\n"runtime.scheduler.mode" = "serial"\n',
        encoding="utf-8",
    )
    args = build_parser().parse_args(["serve"])
    resolved = resolve_args_settings(
        args,
        environ={"MTPLX_SCHEDULER_MODE": "cooperative"},
        user_path=user,
    )
    assert args.scheduler_mode == "cooperative"
    assert resolved.provenance["runtime.scheduler.mode"].source.name == "ENV"


def test_main_rejects_unknown_generic_setting_before_handler(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing.toml"))
    assert main(["serve", "--set", "runtime.proflie=turbo"]) == 2
    assert "runtime.profile" in capsys.readouterr().out
