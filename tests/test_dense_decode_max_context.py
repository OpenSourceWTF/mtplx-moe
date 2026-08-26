"""The dense-decode context ceiling (the 147.4k decode-cliff constant).

Receipts: MEASUREMENTS 2026-08-26 07:58 (root cause) and 08:24 (fix
verified: 12.0 -> 16.3/18.4 tok/s at 147.4k once decode stays dense).
"""

import os

import pytest

from mtplx.generation import (
    _dense_decode_max_context,
    _sustained_prefill_layout,
)

CEILING = "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        CEILING,
        "MTPLX_DENSE_KV_BYTES_PER_TOKEN",
        "MTPLX_DENSE_DECODE_RAM_PERCENT",
        "MTPLX_CONTEXT_WINDOW_TOKENS",
        "MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        "MTPLX_KV_QUANT",
        "MTPLX_PAGED_KV_QUANT",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def _fake_sysconf(total_bytes):
    def sysconf(name):
        if name == "SC_PAGE_SIZE":
            return 4096
        if name == "SC_PHYS_PAGES":
            return total_bytes // 4096
        raise ValueError(name)

    return sysconf


def test_default_is_the_shipped_literal():
    assert _dense_decode_max_context() == 131072


def test_numeric_env_wins(monkeypatch):
    monkeypatch.setenv(CEILING, "262144")
    assert _dense_decode_max_context() == 262144


def test_garbage_env_falls_back(monkeypatch):
    monkeypatch.setenv(CEILING, "lots")
    assert _dense_decode_max_context() == 131072


def test_auto_budgets_from_ram(monkeypatch):
    # 96 GiB machine, Qwen3.8 geometry, 15% budget:
    # 96 GiB * 0.15 / 65536 B = 235929 tokens.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 235929


def test_auto_clamps_to_context_window(monkeypatch):
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_CONTEXT_WINDOW_TOKENS", "200000")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 200000


def test_auto_never_regresses_below_shipped_literal(monkeypatch):
    # 36 GiB machine at 15% = 88473 tokens < 131072: auto must not make
    # the product SLOWER than today's default anywhere.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(36 * 1024**3))
    assert _dense_decode_max_context() == 131072


def test_auto_survives_missing_sysconf(monkeypatch):
    monkeypatch.setenv(CEILING, "auto")

    def broken(_name):
        raise ValueError("no sysconf here")

    monkeypatch.setattr(os, "sysconf", broken)
    assert _dense_decode_max_context() == 131072


def test_auto_honours_geometry_env(monkeypatch):
    # A model with half the KV bytes per token affords twice the tokens.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN", "32768")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 471859


def test_layout_stays_dense_below_ceiling(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv(CEILING, "262144")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "147434")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"


def test_layout_repages_above_ceiling(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv(CEILING, "131072")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "147434")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
