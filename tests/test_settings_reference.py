from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_settings_reference import render_reference


def test_reference_contains_every_public_setting_and_redacts_secrets():
    markdown, payload = render_reference()
    names = {item["name"] for item in payload["settings"]}
    assert "runtime.profile" in names
    assert "generation.temperature" in names
    assert "server.api_key_file" in names
    assert "/private" not in markdown
    assert all(
        item["visibility"] in {"public", "advanced", "experimental"}
        for item in payload["settings"]
    )


def test_checked_in_reference_matches_renderer():
    root = Path(__file__).resolve().parents[1]
    markdown, payload = render_reference()
    assert (root / "docs/reference/settings.md").read_text(
        encoding="utf-8"
    ) == markdown
    assert json.loads(
        (root / "docs/reference/settings.json").read_text(encoding="utf-8")
    ) == payload
