from __future__ import annotations

import os

import pytest

from mtplx.settings.bundles import load_settings_bundle
from mtplx.settings.storage import load_user_settings, update_user_setting


def test_bundle_loads_only_settings_table(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[settings]\n"runtime.profile" = "turbo"\n', encoding="utf-8")
    assert load_settings_bundle(path) == {"runtime.profile": "turbo"}


def test_bundle_rejects_unknown_top_level_tables(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[shell]\ncommand = "rm -rf /"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"only \[settings\]"):
        load_settings_bundle(path)


def test_user_update_is_atomic_private_and_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    update_user_setting(path, "runtime.profile", "sustained")
    update_user_setting(path, "generation.temperature", 0.7)
    assert load_user_settings(path) == {
        "runtime.profile": "sustained",
        "generation.temperature": 0.7,
    }
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_api_key_file_reference_can_be_persisted_but_raw_key_cannot(tmp_path):
    path = tmp_path / "config.toml"
    update_user_setting(path, "server.api_key_file", "~/.mtplx/api-key")
    assert load_user_settings(path)["server.api_key_file"] == "~/.mtplx/api-key"
    with pytest.raises(ValueError, match="unknown setting: server.api_key"):
        update_user_setting(path, "server.api_key", "top-secret")
