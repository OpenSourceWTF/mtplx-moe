"""Issue #147: forge probe says qwen3_5_mtp is forgeable, but the load path
hands the raw `*_mtp` model_type to mlx_lm's class table and fails.

`_mtp_alias_load_path` builds a symlink wrapper with the stripped base
model_type when (and only when) mlx_lm lacks the full name but has the base.
"""

from __future__ import annotations

import json


from mtplx.runtime import _mtp_alias_load_path


def make_model_dir(tmp_path, model_type: str):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {"model_type": model_type, "hidden_size": 64}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"fake-shard")
    (model_dir / "tokenizer.json").write_text("{}")
    return model_dir, config


def test_non_mtp_type_untouched(tmp_path):
    model_dir, config = make_model_dir(tmp_path, "qwen3_6")
    assert _mtp_alias_load_path(model_dir, config) == model_dir


def test_known_alias_builds_patched_wrapper(tmp_path, monkeypatch):
    # qwen3_5 exists in mlx_lm; qwen3_5_mtp does not.
    model_dir, config = make_model_dir(tmp_path, "qwen3_5_mtp")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    wrapper = _mtp_alias_load_path(model_dir, config)
    assert wrapper != model_dir, "alias type must load through a wrapper"
    patched = json.loads((wrapper / "config.json").read_text())
    assert patched["model_type"] == "qwen3_5"
    assert patched["hidden_size"] == 64
    # Weights and tokenizer ride along as symlinks to the original files.
    assert (wrapper / "model.safetensors").resolve() == (
        model_dir / "model.safetensors"
    ).resolve()
    assert (wrapper / "tokenizer.json").resolve() == (
        model_dir / "tokenizer.json"
    ).resolve()
    # Idempotent: second call reuses the same wrapper.
    assert _mtp_alias_load_path(model_dir, config) == wrapper


def test_unknown_base_returns_original_path(tmp_path, monkeypatch):
    model_dir, config = make_model_dir(tmp_path, "totally_unknown_arch_mtp")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _mtp_alias_load_path(model_dir, config) == model_dir


def test_native_mtp_module_wins_over_wrapper(tmp_path, monkeypatch):
    """If a future mlx_lm ships the *_mtp module natively, load it raw."""

    model_dir, config = make_model_dir(tmp_path, "qwen3_5_mtp")

    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "mlx_lm.models.qwen3_5_mtp":
            return real_find_spec("mlx_lm.models.qwen3_5")
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert _mtp_alias_load_path(model_dir, config) == model_dir
