from __future__ import annotations

from mtplx.cli import build_parser
from mtplx.config import apply_user_config


def test_legacy_profile_flag_beats_user_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('profile = "exact"\n', encoding="utf-8")
    args = build_parser().parse_args(["serve", "--profile", "sustained"])
    apply_user_config(args, config_path=path)
    assert args.profile == "sustained"


def test_legacy_user_config_applies_without_explicit_flag(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('profile = "exact"\n', encoding="utf-8")
    args = build_parser().parse_args(["serve"])
    apply_user_config(args, config_path=path)
    assert args.profile == "exact"


def test_historical_settings_set_shape_remains_live_daemon_compatible():
    args = build_parser().parse_args(["settings", "set", "depth=2"])
    assert args.settings_action == "set"
    assert args.pairs == ["depth=2"]
    assert args.func.__name__ == "cmd_settings_public"
