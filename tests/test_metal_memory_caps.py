from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mtplx.server import openai


GiB = 1024**3


def _fake_mx(*, top_level: bool = True):
    calls: list[tuple[str, int]] = []

    metal = SimpleNamespace(
        is_available=lambda: True,
        set_memory_limit=lambda value: calls.append(("metal_memory", int(value))),
        set_wired_limit=lambda value: calls.append(("metal_wired", int(value))),
    )
    mx = SimpleNamespace(metal=metal)
    if top_level:
        mx.set_memory_limit = lambda value: calls.append(("memory", int(value)))
        mx.set_wired_limit = lambda value: calls.append(("wired", int(value)))
    return mx, calls


def test_parse_metal_memory_size_bytes_accepts_suffixes_and_fallbacks():
    assert openai._parse_metal_memory_size_bytes("64G", 1) == 64 * GiB
    assert openai._parse_metal_memory_size_bytes("1.5T", 1) == int(1.5 * 1024**4)
    assert openai._parse_metal_memory_size_bytes("512M", 1) == 512 * 1024**2
    assert openai._parse_metal_memory_size_bytes("bad", 123) == 123
    assert openai._parse_metal_memory_size_bytes("", 456) == 456


def test_detect_total_ram_uses_psutil_when_available(monkeypatch):
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=128 * GiB)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    total, source = openai._detect_total_ram_bytes_for_metal_caps()

    assert total == 128 * GiB
    assert source == "psutil"


def test_detect_total_ram_falls_back_to_sysctl_on_macos(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(openai.sys, "platform", "darwin")
    monkeypatch.setattr(
        openai.subprocess,
        "check_output",
        lambda *_args, **_kwargs: str(64 * GiB),
    )

    total, source = openai._detect_total_ram_bytes_for_metal_caps()

    assert total == 64 * GiB
    assert source == "sysctl_hw_memsize"


def test_apply_metal_memory_caps_uses_top_level_mlx_apis(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "64G")
    monkeypatch.setenv("MTPLX_WIRED_LIMIT_BYTES", "48G")

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 64 * GiB
    assert result["wired_limit_bytes"] == 48 * GiB
    assert result["memory_limit_api"] == "mx.set_memory_limit"
    assert result["wired_limit_api"] == "mx.set_wired_limit"
    assert calls == [("memory", 64 * GiB), ("wired", 48 * GiB)]


def test_apply_metal_memory_caps_caps_large_unified_memory_defaults(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=512 * GiB,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 192 * GiB
    assert result["wired_limit_bytes"] == 160 * GiB
    assert calls == [("memory", 192 * GiB), ("wired", 160 * GiB)]


def test_apply_metal_memory_caps_preserves_128g_defaults(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 96 * GiB
    assert result["wired_limit_bytes"] == int(128 * GiB * 0.60)
    assert calls == [("memory", 96 * GiB), ("wired", int(128 * GiB * 0.60))]


def test_apply_metal_memory_caps_raises_default_wired_floor_for_laguna(
    monkeypatch,
):
    from mtplx.models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=96 * GiB,
        minimum_resident_bytes=LAGUNA_S_2_1_MIN_RESIDENT_BYTES,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 72 * GiB
    assert result["wired_limit_bytes"] == LAGUNA_S_2_1_MIN_RESIDENT_BYTES
    assert calls == [
        ("memory", 72 * GiB),
        ("wired", LAGUNA_S_2_1_MIN_RESIDENT_BYTES),
    ]


def test_apply_metal_memory_caps_rejects_insufficient_ram_for_laguna(
    monkeypatch,
):
    from mtplx.models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=64 * GiB,
        minimum_resident_bytes=LAGUNA_S_2_1_MIN_RESIDENT_BYTES,
    )

    assert result["applied"] is False
    assert result["reason"] == "insufficient_ram"
    assert result["minimum_resident_bytes"] == LAGUNA_S_2_1_MIN_RESIDENT_BYTES
    assert calls == []


def test_laguna_explicit_context_must_fit_active_metal_cap(monkeypatch):
    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR
    from mtplx.models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    mx, _calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    caps = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
        minimum_resident_bytes=LAGUNA_S_2_1_MIN_RESIDENT_BYTES,
    )

    openai._validate_backend_context_memory_budget(
        LAGUNA_AR_DESCRIPTOR,
        caps,
        None,
    )
    with pytest.raises(RuntimeError, match="context window 1,048,576"):
        openai._validate_backend_context_memory_budget(
            LAGUNA_AR_DESCRIPTOR,
            caps,
            1_048_576,
        )


def test_laguna_server_uses_safe_default_but_preserves_explicit_context():
    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR

    assert (
        openai._select_backend_context_window(
            LAGUNA_AR_DESCRIPTOR,
            model_max=1_048_576,
            requested=None,
        )
        == 32_768
    )
    assert (
        openai._select_backend_context_window(
            LAGUNA_AR_DESCRIPTOR,
            model_max=1_048_576,
            requested=65_536,
        )
        == 65_536
    )


def test_apply_metal_memory_caps_falls_back_to_deprecated_metal_apis(monkeypatch):
    mx, calls = _fake_mx(top_level=False)
    monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "32G")
    monkeypatch.setenv("MTPLX_WIRED_LIMIT_BYTES", "64G")

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
    )

    assert result["applied"] is True
    assert result["wired_limit_clamped_to_memory_limit"] is True
    assert result["memory_limit_bytes"] == 32 * GiB
    assert result["wired_limit_bytes"] == 32 * GiB
    assert result["memory_limit_api"] == "mx.metal.set_memory_limit"
    assert result["wired_limit_api"] == "mx.metal.set_wired_limit"
    assert calls == [("metal_memory", 32 * GiB), ("metal_wired", 32 * GiB)]


def test_apply_metal_memory_caps_skips_when_ram_unknown_without_overrides(
    monkeypatch,
):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=0)

    assert result == {"applied": False, "reason": "ram_unknown"}
    assert calls == []
