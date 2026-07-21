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


# --- Routed expert bank gather lane (expert-streaming specs) ------------------
#
# These use only stock mx.gather_qmm / mx.quantized_matmul / mx.quantize, so
# they run on both the GPU and the CPU (unlike the nax/gqa Metal lanes above,
# which are GPU-only). They deliberately leave MTPLX_NAX_VERIFY /
# MTPLX_GQA_PACKED_SDPA unset so only the expert_gather lane engages.


@pytest.mark.parametrize(
    "spec_attr,expected_bits,expected_group_size",
    [
        ("HY3_EXPERT_OQ2E", 2, 128),
        ("HY3_EXPERT_Q2", 2, 64),
        ("HY3_EXPERT_ONLY_Q4", 4, 64),
    ],
)
def test_expert_signature_derived_from_spec(
    spec_attr, expected_bits, expected_group_size
) -> None:
    import mtplx.expert_streaming_models as esm

    spec = getattr(esm, spec_attr)
    sig = kernel_selfcheck._expert_quant_signature(spec)
    assert sig == (mx.bfloat16, expected_bits, expected_group_size)


def test_expert_signature_none_without_spec() -> None:
    assert kernel_selfcheck._expert_quant_signature(None) is None


def test_expert_signature_none_for_shadow_codec() -> None:
    # Shadow-codec (q1 lane) banks do not run the affine gather_qmm path, so
    # they must yield no expert signature and add no gather lane.
    from mtplx.expert_streaming_models import GLM52_EXPERT_Q1T

    assert GLM52_EXPERT_Q1T.expert_codec != "affine"
    assert kernel_selfcheck._expert_quant_signature(GLM52_EXPERT_Q1T) is None


@pytest.mark.parametrize("group_size", [64, 128], ids=["gs64", "gs128"])
def test_expert_gather_lane_passes_on_healthy_bank(group_size) -> None:
    report = run_kernel_selfcheck(
        mx.bfloat16, 4, 64, expert_signature=(mx.bfloat16, 2, group_size)
    )
    assert report["lanes"]["expert_gather"] == "ok"
    assert report["dmax"]["expert_gather"] <= kernel_selfcheck._QMM_TOLERANCE
    assert not lane_disabled("expert_gather")


def test_expert_gather_lane_fails_closed_on_corrupt_kernel(monkeypatch) -> None:
    original = mx.gather_qmm

    def corrupted(*args, **kwargs):
        return original(*args, **kwargs) + 1000.0

    monkeypatch.setattr(mx, "gather_qmm", corrupted)
    report = run_kernel_selfcheck(
        mx.bfloat16, 4, 64, expert_signature=(mx.bfloat16, 2, 128)
    )
    assert report["lanes"]["expert_gather"] == "fallback"
    assert lane_disabled("expert_gather")

    health = report_for_health()
    assert health["ran"] is True
    assert health["expert_gather"] == "fallback"
    json.dumps(health)  # JSON primitives only


def test_expert_gather_lane_fails_closed_on_wrong_group_size() -> None:
    # A group_size that does not divide the synthetic bank's K raises inside
    # the probe; the recorder catches it as a hard fallback (dmax == inf),
    # exactly as a broken resident lane surfaces.
    report = run_kernel_selfcheck(
        mx.bfloat16, 4, 64, expert_signature=(mx.bfloat16, 2, 96)
    )
    assert report["lanes"]["expert_gather"] == "fallback"
    assert report["dmax"]["expert_gather"] == float("inf")
    assert lane_disabled("expert_gather")


def test_expert_gather_check_mismatched_group_size_raises() -> None:
    # The literal "wrong group_size fed to the check": bank quantized at gs=64,
    # gather run at gs=128. gather_qmm's weight/scales contract fails closed.
    with pytest.raises(Exception):
        y = kernel_selfcheck._check_expert_gather(
            mx, mx.bfloat16, 2, 128, bank_group_size=64
        )
        mx.eval(y)


def test_no_expert_lane_without_signature() -> None:
    # Non-streaming path: no expert_signature -> the report carries no expert
    # lane at all (not even "skipped"), keeping dense loads byte-identical.
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert "expert_gather" not in report["lanes"]
    assert "expert_gather" not in report["dmax"]
    assert not lane_disabled("expert_gather")


def test_maybe_run_dense_model_adds_no_expert_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    report = kernel_selfcheck.maybe_run_model_selfcheck(object())
    assert report is not None
    assert "expert_gather" not in report["lanes"]
    assert not lane_disabled("expert_gather")


def test_maybe_run_streaming_spec_adds_expert_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    from mtplx.expert_streaming_models import HY3_EXPERT_OQ2E

    report = kernel_selfcheck.maybe_run_model_selfcheck(
        object(), expert_spec=HY3_EXPERT_OQ2E
    )
    assert report is not None
    assert report["lanes"]["expert_gather"] == "ok"
    assert not lane_disabled("expert_gather")
    assert report_for_health()["expert_gather"] == "ok"


def test_maybe_run_disabled_skips_expert_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    from mtplx.expert_streaming_models import HY3_EXPERT_OQ2E

    assert (
        kernel_selfcheck.maybe_run_model_selfcheck(
            object(), expert_spec=HY3_EXPERT_OQ2E
        )
        is None
    )
