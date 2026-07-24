from __future__ import annotations

import json

from mtplx.cli import main


VALID_RECIPE = '''
[experiment]
id = "compiled-verify-control"
title = "Compiled verify disabled control"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Isolate compiled verify."
[settings]
"verify.compiled.mode" = "off"
'''


def test_lab_list_is_json_and_no_mlx(capsys):
    assert main(["lab", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(item["status"] == "active" for item in payload["experiments"])


def test_lab_show_includes_hash_and_settings(capsys):
    assert main(["lab", "show", "compiled-verify-control", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "compiled-verify-control"
    assert len(payload["sha256"]) == 64
    assert payload["settings"]["verify.compiled.mode"] == "off"


def test_lab_validate_rejects_unknown_setting(capsys, tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        VALID_RECIPE.replace("verify.compiled.mode", "verify.compiled.typo"),
        encoding="utf-8",
    )
    assert main(["lab", "validate", str(path), "--json"]) == 2
    assert "unknown setting" in capsys.readouterr().out
