"""Env-gated selection between the inline_g and headquarter post-conv routes.

Pure-CPU contract tests: no A3B model load and no real GPU kernel launch. The
install-path tests drive ``install_a3b_gdn_postconv`` with a minimal plan and a
mocked self-check report; the self-check-gating tests exercise
``run_kernel_selfcheck`` with the only active probe monkeypatched (every other
lane is skipped when just ``MTPLX_FUSE_GDN_POST_CONV`` is set), so no Metal
kernel is dispatched.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import gdn_capture
from mtplx import kernel_selfcheck


def _plan(n: int = 3) -> gdn_capture.A3BGDNPostconvInstallPlan:
    gdns = tuple(
        SimpleNamespace(A_log=object(), dt_bias=object()) for _ in range(n)
    )
    return gdn_capture.A3BGDNPostconvInstallPlan(gdns=gdns)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raising=False)
    monkeypatch.delenv("MTPLX_FUSE_GDN_POST_CONV", raising=False)
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    monkeypatch.delenv("MTPLX_QWEN_ROW_OWNED_ROUTER", raising=False)
    monkeypatch.delenv("MTPLX_QWEN_COMBINE_TAIL", raising=False)
    monkeypatch.delenv("MTPLX_FUSE_POST_NORM_RESIDUAL", raising=False)
    monkeypatch.delenv("MTPLX_FUSE_GDN_NORM_GATE", raising=False)
    gdn_capture._reset_gdn_postconv_stats_for_tests()
    kernel_selfcheck._reset_for_tests()
    yield
    gdn_capture._reset_gdn_postconv_stats_for_tests()
    kernel_selfcheck._reset_for_tests()


# ---------------------------------------------------------------------------
# (a) default env -> inline_g selected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", (None, "", "  ", "inline_g", "  Inline_G "))
def test_default_or_inline_g_env_selects_inline_g(monkeypatch, raw) -> None:
    if raw is None:
        monkeypatch.delenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raising=False)
    else:
        monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raw)

    factory = gdn_capture.install_a3b_gdn_postconv(
        _plan(), {"lanes": {"gdn_postconv_inline_g": "ok"}}
    )

    assert len(factory.m1_implementations) == 3
    assert all(
        impl.func is gdn_capture._apply_enabled_a3b_gdn_postconv_m1_tgy4
        for impl in factory.m1_implementations
    )
    assert all(
        impl.func is gdn_capture._apply_enabled_a3b_gdn_postconv_m2_tgy4
        for impl in factory.m2_implementations
    )
    report = gdn_capture.gdn_postconv_stats()
    assert report["installed"] is True
    assert report["implementation"] == "inline_g"


def test_inline_g_requires_its_own_lane_ok(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raising=False)
    # A green headquarter lane must not satisfy the default inline_g route.
    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError, match="selfcheck"):
        gdn_capture.install_a3b_gdn_postconv(
            _plan(), {"lanes": {"gdn_postconv_headquarter": "ok"}}
        )
    assert gdn_capture.gdn_postconv_stats()["installed"] is False


# ---------------------------------------------------------------------------
# (b) headquarter -> headquarter selected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ("headquarter", "  HeadQuarter "))
def test_headquarter_env_selects_headquarter(monkeypatch, raw) -> None:
    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raw)

    factory = gdn_capture.install_a3b_gdn_postconv(
        _plan(), {"lanes": {"gdn_postconv_headquarter": "ok"}}
    )

    assert len(factory.m2_implementations) == 3
    assert all(
        impl.func is gdn_capture._apply_enabled_a3b_gdn_postconv_m1_headquarter
        for impl in factory.m1_implementations
    )
    assert all(
        impl.func is gdn_capture._apply_enabled_a3b_gdn_postconv_m2_headquarter
        for impl in factory.m2_implementations
    )
    report = gdn_capture.gdn_postconv_stats()
    assert report["installed"] is True
    assert report["implementation"] == "headquarter"


# ---------------------------------------------------------------------------
# (c) unknown value -> hard fail (fail-closed).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ("turbocharged", "inlineg", "hq", "1"))
def test_unknown_impl_value_hard_fails(monkeypatch, raw) -> None:
    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raw)
    # Both lanes green: the failure is proven to be the unknown value, not a lane.
    with pytest.raises(
        gdn_capture.A3BGDNPostconvConfigError,
        match="MTPLX_A3B_GDN_POSTCONV_IMPL",
    ):
        gdn_capture.install_a3b_gdn_postconv(
            _plan(),
            {
                "lanes": {
                    "gdn_postconv_inline_g": "ok",
                    "gdn_postconv_headquarter": "ok",
                }
            },
        )
    report = gdn_capture.gdn_postconv_stats()
    assert report["installed"] is False
    assert report["installation_status"] == "configuration_error"


# ---------------------------------------------------------------------------
# (d) headquarter requested but lane not ok -> hard fail, no fallback.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lanes",
    (
        {"gdn_postconv_headquarter": "fallback"},
        {"gdn_postconv_headquarter": "skipped"},
        {},  # lane absent from the report
        {"gdn_postconv_inline_g": "ok"},  # only the OTHER route validated
    ),
)
def test_headquarter_lane_not_ok_fails_closed(monkeypatch, lanes) -> None:
    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "headquarter")
    with pytest.raises(
        gdn_capture.A3BGDNPostconvConfigError, match="headquarter"
    ):
        gdn_capture.install_a3b_gdn_postconv(_plan(), {"lanes": lanes})
    assert gdn_capture.gdn_postconv_stats()["installed"] is False


def test_none_report_fails_closed_for_headquarter(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "headquarter")
    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError):
        gdn_capture.install_a3b_gdn_postconv(_plan(), None)
    assert gdn_capture.gdn_postconv_stats()["installed"] is False


# ---------------------------------------------------------------------------
# Selection helpers (pure, non-raising vs fail-closed).
# ---------------------------------------------------------------------------
def test_impl_selection_helper_contract(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raising=False)
    assert gdn_capture._a3b_gdn_postconv_impl_selection() == "inline_g"
    assert gdn_capture._a3b_gdn_postconv_headquarter_requested() is False

    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "")
    assert gdn_capture._a3b_gdn_postconv_impl_selection() == "inline_g"

    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "  HeadQuarter ")
    assert gdn_capture._a3b_gdn_postconv_impl_selection() == "headquarter"
    assert gdn_capture._a3b_gdn_postconv_headquarter_requested() is True

    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "nonsense")
    assert gdn_capture._a3b_gdn_postconv_headquarter_requested() is False
    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError):
        gdn_capture._a3b_gdn_postconv_impl_selection()


# ---------------------------------------------------------------------------
# Self-check lane gating (GPU-free: the only active probe is monkeypatched and
# every other lane is skipped when just MTPLX_FUSE_GDN_POST_CONV is set).
# ---------------------------------------------------------------------------
def test_selfcheck_skips_headquarter_by_default(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.delenv("MTPLX_A3B_GDN_POSTCONV_IMPL", raising=False)
    monkeypatch.setattr(
        kernel_selfcheck,
        "_check_gdn_postconv_inline_g",
        lambda mx_module, dtype: 0.0,
        raising=False,
    )
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["gdn_postconv_inline_g"] == "ok"
    assert report["lanes"]["gdn_postconv_headquarter"] == "skipped"


def test_selfcheck_runs_headquarter_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "headquarter")
    monkeypatch.setattr(
        kernel_selfcheck,
        "_check_gdn_postconv_headquarter",
        lambda mx_module, dtype: 0.0,
        raising=False,
    )
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["gdn_postconv_headquarter"] == "ok"
    assert report["lanes"]["gdn_postconv_inline_g"] == "skipped"


def test_selfcheck_headquarter_fallback_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FUSE_GDN_POST_CONV", "1")
    monkeypatch.setenv("MTPLX_A3B_GDN_POSTCONV_IMPL", "headquarter")
    monkeypatch.setattr(
        kernel_selfcheck,
        "_check_gdn_postconv_headquarter",
        lambda mx_module, dtype: 1.0,  # above the 0.03125 tolerance
        raising=False,
    )
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["gdn_postconv_headquarter"] == "fallback"
    # A fallen-back headquarter lane must not install (fail-closed at install).
    with pytest.raises(gdn_capture.A3BGDNPostconvConfigError):
        gdn_capture.install_a3b_gdn_postconv(_plan(), report)
