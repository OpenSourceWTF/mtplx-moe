"""Unit tests for engine_session bank-cap env-var overrides."""

import importlib

import pytest


def _reload_module():
    import mtplx.engine_session
    return importlib.reload(mtplx.engine_session)


def test_bank_bytes_from_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_BANK_BYTES", raising=False)
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 1234) == 1234


def test_bank_bytes_from_env_plain_integer(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "987654321")
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 0) == 987654321


@pytest.mark.parametrize("raw,expected", [
    ("16G", 16 * 1024**3),
    ("16g", 16 * 1024**3),
    ("32G", 32 * 1024**3),
    ("8G",   8 * 1024**3),
    ("512M", 512 * 1024**2),
    ("4K",   4 * 1024),
    ("1T",   1 * 1024**4),
    ("0.5G", int(0.5 * 1024**3)),
])
def test_bank_bytes_from_env_with_suffix(monkeypatch, raw, expected):
    monkeypatch.setenv("TEST_BANK_BYTES", raw)
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 0) == expected


def test_bank_bytes_from_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "not-a-number")
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 5555) == 5555


def test_bank_bytes_from_env_empty_string_uses_default(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "")
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 7777) == 7777


@pytest.mark.parametrize("raw", ["0", "-1", "0G", "-2G"])
def test_bank_bytes_from_env_nonpositive_uses_default(monkeypatch, raw):
    monkeypatch.setenv("TEST_BANK_BYTES", raw)
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 8888) == 8888


def test_short_no_history_api_request_is_foreground_by_default():
    es = _reload_module()
    messages = [
        {"role": "system", "content": "Return only the final answer."},
        {"role": "user", "content": "Compute 17 + 29 + 101."},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={},
            metadata={},
            main_system_hash=None,
        )
        is False
    )


def test_openwebui_task_header_still_marks_background():
    es = _reload_module()
    messages = [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Conversation text"},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={"x-openwebui-task": "title"},
            metadata={},
            main_system_hash=None,
        )
        is True
    )


def test_system_prompt_mismatch_still_marks_background():
    es = _reload_module()
    main_hash = es.hash_text("main chat system")
    messages = [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Conversation text"},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={},
            metadata={},
            main_system_hash=main_hash,
        )
        is True
    )


# --- model-aware auto budget (v2, founder memory ruling 2026-07-05) ----------

GIB = 1024**3


def _es_with_ram(monkeypatch, total_ram_bytes):
    es = _reload_module()
    monkeypatch.setattr(
        es, "_detect_total_ram_bytes_for_session_bank", lambda: total_ram_bytes
    )
    return es


def test_auto_budget_is_half_the_post_model_surplus(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 64 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (int(45 * GIB * 0.5), True)


def test_auto_budget_capped_on_big_machines(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "auto")
    es = _es_with_ram(monkeypatch, 128 * GIB)
    # 0.5 * (128 - 19) = 54.5G -> capped at 48G
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (48 * GIB, True)


def test_auto_budget_floors_when_model_fills_ram(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 16 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (1 * GIB, True)


def test_auto_budget_small_machine_gets_small_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 36 * GIB)
    # 0.5 * (36 - 19) = 8.5G — NOT the old flat 24G default.
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (int(17 * GIB * 0.5), True)


def test_auto_budget_unknown_model_falls_back_to_legacy_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 128 * GIB)
    from mtplx.session_bank import DEFAULT_MAX_BYTES

    assert es.resolve_session_bank_max_bytes(None) == (DEFAULT_MAX_BYTES, False)


def test_auto_budget_unknown_ram_falls_back_to_legacy_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, None)
    from mtplx.session_bank import DEFAULT_MAX_BYTES

    assert es.resolve_session_bank_max_bytes(19 * GIB) == (DEFAULT_MAX_BYTES, False)


def test_explicit_max_bytes_env_overrides_auto(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "16G")
    es = _es_with_ram(monkeypatch, 128 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (16 * GIB, False)


def test_per_session_auto_is_two_thirds_of_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_module()
    assert es.resolve_session_bank_per_session_bytes(30 * GIB) == 20 * GIB


def test_per_session_explicit_env_clamped_to_budget(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "24G")
    es = _reload_module()
    assert es.resolve_session_bank_per_session_bytes(8 * GIB) == 8 * GIB


def test_model_weights_bytes_sums_safetensors(tmp_path):
    es = _reload_module()
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1024)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"y" * 2048)
    (tmp_path / "mtp.safetensors").write_bytes(b"z" * 512)
    (tmp_path / "config.json").write_text("{}")
    assert es.model_weights_bytes(tmp_path) == 1024 + 2048 + 512
    assert es.model_weights_bytes(tmp_path / "missing") is None


def test_manager_uses_auto_budget_for_bank(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 64 * GIB)
    manager = es.EngineSessionManager(model_weights_bytes=19 * GIB)
    expected = int(45 * GIB * 0.5)
    assert manager.bank.max_bytes == expected
    assert manager.bank.per_session_max_bytes == max(GIB, expected * 2 // 3)
