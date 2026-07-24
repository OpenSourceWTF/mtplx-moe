from __future__ import annotations

import json

from mtplx.cli import build_parser, main


def test_settings_parser_has_unambiguous_scopes():
    parser = build_parser()

    effective = parser.parse_args(["settings", "show"])
    assert effective.settings_operation == "show"
    assert effective.settings_scope == "effective"

    user = parser.parse_args(
        ["settings", "user", "set", "runtime.profile=sustained"]
    )
    assert user.settings_scope == "user"
    assert user.settings_operation == "set"
    assert user.pairs == ["runtime.profile=sustained"]

    live = parser.parse_args(["settings", "live", "show"])
    assert live.settings_scope == "live"
    assert live.settings_operation == "show"


def test_settings_explain_is_no_mlx_and_reports_source(
    capsys, tmp_path, monkeypatch
):
    path = tmp_path / "config.toml"
    path.write_text(
        '[settings]\n"runtime.profile" = "turbo"\n', encoding="utf-8"
    )
    monkeypatch.setenv("MTPLX_CONFIG", str(path))

    assert main(["settings", "explain", "runtime.profile", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "runtime.profile"
    assert payload["value"] == "turbo"
    assert payload["source"] == "USER"


def test_settings_user_rejects_unknown_key_without_writing(capsys, tmp_path):
    path = tmp_path / "config.toml"

    assert (
        main(
            [
                "settings",
                "user",
                "set",
                "runtime.proflie=turbo",
                "--config",
                str(path),
            ]
        )
        == 2
    )

    assert not path.exists()
    assert "runtime.profile" in capsys.readouterr().out


def test_settings_user_set_and_unset_round_trip(capsys, tmp_path):
    path = tmp_path / "config.toml"

    assert (
        main(
            [
                "settings",
                "user",
                "set",
                "runtime.profile=sustained",
                "--config",
                str(path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"] == {"runtime.profile": "sustained"}

    assert (
        main(
            [
                "settings",
                "user",
                "unset",
                "runtime.profile",
                "--config",
                str(path),
            ]
        )
        == 0
    )
    assert "removed: runtime.profile" in capsys.readouterr().out


def test_settings_user_output_redacts_secret_values(capsys, tmp_path):
    path = tmp_path / "config.toml"
    secret_path = "/private/secret-api-key-location"

    assert (
        main(
            [
                "settings",
                "user",
                "set",
                f"server.api_key_file={secret_path}",
                "--config",
                str(path),
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert secret_path not in output
    assert json.loads(output)["settings"]["server.api_key_file"] == "[redacted]"

    assert (
        main(
            [
                "settings",
                "user",
                "show",
                "--config",
                str(path),
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert secret_path not in output
    assert json.loads(output)["settings"]["server.api_key_file"] == "[redacted]"


def test_bare_legacy_settings_pair_still_means_live_set(monkeypatch):
    calls = []

    def fake_handler(args):
        calls.append(args)
        return 0

    monkeypatch.setattr("mtplx.cli.cmd_settings_public", fake_handler)
    assert main(["settings", "depth=2"]) == 0
    assert calls[0].settings_action == "set"
    assert calls[0].pairs == ["depth=2"]
