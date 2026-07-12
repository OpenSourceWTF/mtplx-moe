"""Load-time turbo kernel self-validation: pass, fallback, and force-fallback."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import kernel_selfcheck, nax_verify
from mtplx.kernel_selfcheck import (
    lane_disabled,
    report_for_health,
    run_kernel_selfcheck,
    selfcheck_enabled,
)


@pytest.fixture(autouse=True)
def _clean_selfcheck_state():
    kernel_selfcheck._reset_for_tests()
    yield
    kernel_selfcheck._reset_for_tests()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bits", [4, 8])
def test_selfcheck_passes_on_this_machine(monkeypatch, dtype, bits) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    report = run_kernel_selfcheck(dtype, bits, 64)
    lanes = report["lanes"]
    checked = {lane: s for lane, s in lanes.items() if s != "skipped"}
    assert checked, "no lanes engaged — the selfcheck validated nothing"
    bad = {lane: s for lane, s in checked.items() if s != "ok"}
    assert not bad, f"selfcheck lanes failed on this machine: {bad} dmax={report['dmax']}"
    assert not any(lane_disabled(lane) for lane in lanes)
    # The qmm lanes for the model's bits and the packed-GQA lane must be
    # among the validated set.
    assert lanes["qmm_m4"] == "ok"
    assert lanes["qmm_m6"] == "ok"
    assert lanes["gqa_packed_sdpa"] == "ok"


def test_selfcheck_mismatch_disables_lane_and_surfaces_in_health(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")

    original = nax_verify.nax_qmm_m4

    def corrupted(x2, w_q, scales, biases, *, group_size=64):
        return original(x2, w_q, scales, biases, group_size=group_size) + 1000.0

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", corrupted)
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["qmm_m4"] == "fallback"
    assert lane_disabled("qmm_m4")
    # The sibling lanes stay engaged: fallback is per-lane, not global.
    assert report["lanes"]["qmm_m6"] == "ok"
    assert not lane_disabled("qmm_m6")

    health = report_for_health()
    assert health["ran"] is True
    assert health["qmm_m4"] == "fallback"
    assert health["qmm_m6"] == "ok"
    json.dumps(health)  # JSON primitives only — the watchdog Codable lesson


def test_disabled_lane_routes_stock_through_the_qlinear_patch(monkeypatch) -> None:
    from mtplx.attention_context import attention_phase

    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")

    original = nax_verify.nax_qmm_m4

    def corrupted(x2, w_q, scales, biases, *, group_size=64):
        return original(x2, w_q, scales, biases, group_size=group_size) + 1000.0

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", corrupted)
    run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert lane_disabled("qmm_m4")

    calls = {"m4": 0}

    def counting(x2, w_q, scales, biases, *, group_size=64):
        calls["m4"] += 1
        return original(x2, w_q, scales, biases, group_size=group_size)

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", counting)
    report = nax_verify.install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            y = layer(x)
            mx.eval(y)
        assert y.shape == (4, 256)
        assert calls["m4"] == 0, "disabled qmm_m4 lane still routed the custom kernel"
    finally:
        nax_verify.uninstall_nax_qlinear_patch()


def test_selfcheck_kernel_exception_falls_back_instead_of_raising(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(nax_verify, "nax_qmm_m6", broken)
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["qmm_m6"] == "fallback"
    assert lane_disabled("qmm_m6")
    assert report["lanes"]["qmm_m4"] == "ok"


def test_force_gpu_family_fallback_disables_nax_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    nax_verify.nax_available.cache_clear()
    try:
        assert nax_verify.nax_available() is False
        assert not nax_verify.m16_nax_eligible(8, 5120, 17408, 4, 64, mx.bfloat16)
        report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
        assert report["lanes"]["qmm_m16_nax"] == "skipped"
        # The plain-SIMD lanes carry the win and must still validate.
        assert report["lanes"]["qmm_m4"] == "ok"
        assert report["lanes"]["qmm_m6"] == "ok"
    finally:
        nax_verify.nax_available.cache_clear()


def test_selfcheck_enabled_gating(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_KERNEL_SELFCHECK", raising=False)
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    assert selfcheck_enabled() is False
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    assert selfcheck_enabled() is True
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    assert selfcheck_enabled() is False
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    assert selfcheck_enabled() is True


def test_health_payload_before_any_run_is_safe() -> None:
    payload = report_for_health()
    assert payload == {"ran": False}
    json.dumps(payload)
