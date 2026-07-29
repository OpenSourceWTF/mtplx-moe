"""Tests for the m4/NAX verify kernel module."""

from __future__ import annotations

from importlib.metadata import version
from threading import Event, Thread

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.nax_verify import (
    FixedQMMExecution,
    fixed_qmm_execution_scope,
    install_nax_qlinear_patch,
    m4_ksplit_eligible,
    m16_nax_eligible,
    nax_available,
    nax_qmm_m4,
    nax_qmm_m16,
    qlinear_patch_snapshot,
    uninstall_nax_qlinear_patch,
)


def test_eligibility_shape_policy() -> None:
    dt = mx.bfloat16
    # m4: exact 4 rows only, no NAX hardware requirement
    assert m4_ksplit_eligible(4, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(5, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(4, 5120, 17408, 8, 64, dt)
    # m16: K % 256, N % 32, 4-bit, M in 1..16 (and NAX hardware)
    expect = nax_available()
    assert m16_nax_eligible(5, 5120, 17408, 4, 64, dt) == expect
    assert m16_nax_eligible(16, 17408, 5120, 4, 64, dt) == expect
    assert not m16_nax_eligible(17, 5120, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120 + 64, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120, 17408 + 8, 4, 64, dt)


def _quantized_fixture(K: int, N: int):
    mx.random.seed(3)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _stock(x, w_q, scales, biases):
    return mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
    )


def test_m4_kernel_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    x = (mx.random.normal((4, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    y = nax_qmm_m4(x, w_q, scales, biases, group_size=64)
    ref = _stock(x, w_q, scales, biases)
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert y.shape == (4, N)
    assert diff < 0.25, f"m4 kernel drift too large: {diff}"


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_m16_nax_kernel_pads_and_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 16):
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m16(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"nax16 kernel drift too large at M={m}: {diff}"


def test_qlinear_patch_routes_only_verify_shapes() -> None:
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        for m in (1, 3, 4, 8, 17, 64):
            x = (mx.random.normal((m, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
            y = layer(x)
            mx.eval(y)
            assert y.shape == (m, 256)
    finally:
        uninstall_nax_qlinear_patch()


def test_qlinear_patch_snapshot_captures_stock_without_mutation() -> None:
    before = qlinear_patch_snapshot()

    assert before.installed is False
    assert before.stock_call is nn.QuantizedLinear.__call__
    report = install_nax_qlinear_patch()
    try:
        assert report["installed"] is True
        after = qlinear_patch_snapshot()
        assert after.installed is True
        assert after.stock_call is before.stock_call
    finally:
        uninstall_nax_qlinear_patch()


def test_fixed_patch_lease_preserves_initial_stock_and_blocks_uninstall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.nax_verify import prepare_fixed_qlinear_patch_lease

    original = nn.QuantizedLinear.__call__
    lease = prepare_fixed_qlinear_patch_lease()
    assert lease.stock_call is original
    assert lease.initially_dynamic is False
    lease.acquire()
    layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    route_calls: list[int] = []
    sentinel = object()

    class Route:
        def execute(self, _x, *, width):
            route_calls.append(width)
            return sentinel

    execution = FixedQMMExecution(routes={id(layer): Route()}, width=2)
    x = mx.zeros((4, 512), dtype=mx.bfloat16)
    try:
        outside = layer(x)
        mx.eval(outside)
        assert outside.shape == (4, 256)
        assert route_calls == []
        with fixed_qmm_execution_scope(execution):
            assert layer(object()) is sentinel
        assert route_calls == [2]
        with pytest.raises(RuntimeError, match="fixed QMM owner"):
            uninstall_nax_qlinear_patch()
        with fixed_qmm_execution_scope(execution):
            assert layer(object()) is sentinel
    finally:
        lease.release()
    assert nn.QuantizedLinear.__call__ is original


def test_fixed_patch_lease_preserves_initial_dynamic_outside_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx import nax_verify
    from mtplx.attention_context import attention_phase
    from mtplx.nax_verify import prepare_fixed_qlinear_patch_lease

    original = nn.QuantizedLinear.__call__
    install_nax_qlinear_patch()
    dynamic = nn.QuantizedLinear.__call__
    lease = prepare_fixed_qlinear_patch_lease()
    assert lease.stock_call is original
    assert lease.initially_dynamic is True
    lease.acquire()
    layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    dynamic_calls: list[tuple[int, int]] = []

    def fake_m4(x, weight, scales, biases, *, group_size):
        del scales, biases, group_size
        dynamic_calls.append((int(x.shape[0]), int(weight.shape[0])))
        return mx.zeros((int(x.shape[0]), int(weight.shape[0])), dtype=x.dtype)

    monkeypatch.setattr(nax_verify, "nax_qmm_m4", fake_m4)
    x = mx.zeros((4, 512), dtype=mx.bfloat16)
    try:
        with attention_phase("decode_verify"):
            outside = layer(x)
            mx.eval(outside)
        assert outside.shape == (4, 256)
        assert dynamic_calls == [(4, 256)]
        with pytest.raises(RuntimeError, match="fixed QMM owner"):
            uninstall_nax_qlinear_patch()
        assert nn.QuantizedLinear.__call__ is dynamic
    finally:
        lease.release()
        uninstall_nax_qlinear_patch()
    assert nn.QuantizedLinear.__call__ is original


def test_acquire_and_uninstall_share_one_serialized_state_transition() -> None:
    from mtplx import nax_verify
    from mtplx.nax_verify import prepare_fixed_qlinear_patch_lease

    original = nn.QuantizedLinear.__call__
    install_nax_qlinear_patch()
    dynamic = nn.QuantizedLinear.__call__
    lease = prepare_fixed_qlinear_patch_lease()
    acquire_started = Event()
    acquire_done = Event()
    uninstall_started = Event()
    uninstall_done = Event()
    acquire_errors: list[Exception] = []
    uninstall_errors: list[Exception] = []

    def acquire() -> None:
        acquire_started.set()
        try:
            lease.acquire()
        except Exception as exc:
            acquire_errors.append(exc)
        finally:
            acquire_done.set()

    def uninstall() -> None:
        uninstall_started.set()
        try:
            uninstall_nax_qlinear_patch()
        except Exception as exc:
            uninstall_errors.append(exc)
        finally:
            uninstall_done.set()

    nax_verify._QLINEAR_PATCH_LOCK.acquire()
    acquire_thread = Thread(target=acquire)
    uninstall_thread = Thread(target=uninstall)
    acquire_was_blocked = False
    uninstall_was_blocked = False
    try:
        acquire_thread.start()
        assert acquire_started.wait(timeout=1.0)
        uninstall_thread.start()
        assert uninstall_started.wait(timeout=1.0)
        acquire_was_blocked = not acquire_done.wait(timeout=0.05)
        uninstall_was_blocked = not uninstall_done.wait(timeout=0.05)
    finally:
        nax_verify._QLINEAR_PATCH_LOCK.release()
    acquire_thread.join(timeout=1.0)
    uninstall_thread.join(timeout=1.0)

    try:
        assert acquire_was_blocked
        assert uninstall_was_blocked
        assert acquire_done.is_set()
        assert uninstall_done.is_set()
        assert not (lease.active and nn.QuantizedLinear.__call__ is original)
        if lease.active:
            assert nn.QuantizedLinear.__call__ is dynamic
            assert not acquire_errors
            assert len(uninstall_errors) == 1
        else:
            assert nn.QuantizedLinear.__call__ is original
            assert len(acquire_errors) == 1
            assert not uninstall_errors
    finally:
        if lease.active:
            lease.release()
        if nax_verify._QLINEAR_PATCH["installed"]:
            uninstall_nax_qlinear_patch()
    assert nn.QuantizedLinear.__call__ is original


def test_concurrent_dynamic_install_cannot_capture_fixed_wrapper_as_stock() -> None:
    from mtplx import nax_verify
    from mtplx.nax_verify import prepare_fixed_qlinear_patch_lease

    original = nn.QuantizedLinear.__call__
    lease = prepare_fixed_qlinear_patch_lease()
    acquire_started = Event()
    acquire_done = Event()
    install_started = Event()
    install_done = Event()
    acquire_errors: list[Exception] = []
    install_errors: list[Exception] = []

    def acquire() -> None:
        acquire_started.set()
        try:
            lease.acquire()
        except Exception as exc:
            acquire_errors.append(exc)
        finally:
            acquire_done.set()

    def install() -> None:
        install_started.set()
        try:
            install_nax_qlinear_patch()
        except Exception as exc:
            install_errors.append(exc)
        finally:
            install_done.set()

    nax_verify._QLINEAR_PATCH_LOCK.acquire()
    acquire_thread = Thread(target=acquire)
    install_thread = Thread(target=install)
    acquire_was_blocked = False
    install_was_blocked = False
    try:
        acquire_thread.start()
        assert acquire_started.wait(timeout=1.0)
        install_thread.start()
        assert install_started.wait(timeout=1.0)
        acquire_was_blocked = not acquire_done.wait(timeout=0.05)
        install_was_blocked = not install_done.wait(timeout=0.05)
    finally:
        nax_verify._QLINEAR_PATCH_LOCK.release()
    acquire_thread.join(timeout=1.0)
    install_thread.join(timeout=1.0)

    try:
        assert acquire_was_blocked
        assert install_was_blocked
        assert acquire_done.is_set()
        assert install_done.is_set()
        assert not install_errors
        assert nax_verify.qlinear_patch_snapshot().stock_call is original
        if nax_verify._QLINEAR_PATCH["installed"]:
            assert nax_verify._QLINEAR_PATCH["original"] is original
        if lease.active:
            assert not acquire_errors
        else:
            assert len(acquire_errors) == 1
    finally:
        if lease.active:
            lease.release()
        if nax_verify._QLINEAR_PATCH["installed"]:
            uninstall_nax_qlinear_patch()
    assert nn.QuantizedLinear.__call__ is original


def test_qlinear_patch_fixed_scope_executes_prebuilt_route_before_dynamic_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx import nax_verify
    from mtplx import kernel_selfcheck

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    sentinel = object()
    calls: list[tuple[object, int]] = []

    class Route:
        def execute(self, x, *, width):
            calls.append((x, width))
            return sentinel

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dynamic gate reached from fixed qlinear scope")

    monkeypatch.setattr(nax_verify, "m6_ksplit_eligible", forbidden)
    monkeypatch.setattr(kernel_selfcheck, "lane_disabled", forbidden)
    monkeypatch.setattr(nax_verify.os.environ, "get", forbidden)
    x = object()
    execution = FixedQMMExecution(routes={id(layer): Route()}, width=2)
    try:
        with fixed_qmm_execution_scope(execution):
            assert layer(x) is sentinel
        assert calls == [(x, 2)]
    finally:
        uninstall_nax_qlinear_patch()


def test_turbo_profile_carries_nax_env() -> None:
    from mtplx.profiles import PROFILES, PROFILE_CHOICES, apply_profile_env, restore_profile_env
    import os

    assert "turbo" in PROFILE_CHOICES
    profile = PROFILES["turbo"]
    assert profile.env_dict().get("MTPLX_NAX_VERIFY") == "1"
    assert profile.product_claim_eligible is False
    # Sustained env must be a subset (turbo = sustained + kernels).
    sustained = PROFILES["sustained"].env_dict()
    turbo = profile.env_dict()
    missing = {k: v for k, v in sustained.items() if turbo.get(k) != v}
    assert not missing, f"turbo drops sustained env keys: {missing}"
    previous = apply_profile_env("turbo")
    try:
        assert os.environ.get("MTPLX_NAX_VERIFY") == "1"
    finally:
        restore_profile_env(previous)
        assert os.environ.get("MTPLX_NAX_VERIFY") != "1"


def test_qlinear_patch_never_routes_in_prefill_phase() -> None:
    """Regression guard: prefill must stay on stock kernels byte-for-byte."""
    import mlx.core as mx
    from mtplx.attention_context import attention_phase
    from mtplx import nax_verify

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0, "m16": 0}
    orig_m4, orig_m16 = nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16

    def count_m4(*a, **k):
        calls["m4"] += 1
        return orig_m4(*a, **k)

    def count_m16(*a, **k):
        calls["m16"] += 1
        return orig_m16(*a, **k)

    nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = count_m4, count_m16
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("prefill"):
            mx.eval(layer(x))
        assert calls == {"m4": 0, "m16": 0}, f"kernels routed during prefill: {calls}"
        with attention_phase("decode_verify"):
            mx.eval(layer(x))
        assert calls["m4"] == 1, f"m4 kernel did not engage outside prefill: {calls}"
    finally:
        nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = orig_m4, orig_m16
        uninstall_nax_qlinear_patch()


def test_m6_kernel_matches_stock_within_tolerance() -> None:
    from mtplx.nax_verify import m6_ksplit_eligible, nax_qmm_m6

    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 6):
        assert m6_ksplit_eligible(m, K, N, 4, 64, mx.bfloat16)
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m6(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"m6 kernel drift too large at M={m}: {diff}"
    assert not m6_ksplit_eligible(4, K, N, 4, 64, mx.bfloat16)
    assert not m6_ksplit_eligible(7, K, N, 4, 64, mx.bfloat16)


def test_m6_kp1_bn2_kernel_matches_stock_within_tolerance() -> None:
    from mtplx.nax_verify import nax_qmm_m6_kp1_bn2

    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    x = (mx.random.normal((6, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)

    y = nax_qmm_m6_kp1_bn2(x, w_q, scales, biases, group_size=64)
    ref = _stock(x, w_q, scales, biases)
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())

    assert y.shape == (6, N)
    assert diff < 0.25, f"m6 Kp1/BN2 kernel drift too large: {diff}"


@pytest.mark.skipif(
    tuple(int(part) for part in version("mlx").split(".")[:2]) < (0, 32),
    reason="MLX before 0.32 routes M=6 through qmv_fast instead of qmv_wide",
)
@pytest.mark.parametrize(("K", "N"), [(5120, 48), (6144, 1024)])
def test_m6_qmv_wide_vec6_kernel_is_bit_exact_to_stock(K: int, N: int) -> None:
    from mtplx.nax_verify import nax_qmm_m6_qmv_wide_vec6

    w_q, scales, biases = _quantized_fixture(K, N)
    x = (mx.random.normal((6, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)

    y = nax_qmm_m6_qmv_wide_vec6(
        x,
        w_q,
        scales,
        biases,
        group_size=64,
    )
    ref = _stock(x, w_q, scales, biases)
    mx.eval(y, ref)

    assert y.shape == (6, N)
    assert mx.array_equal(y, ref).item()


def test_m6_qmv_wide_vec6_uses_sustained_two_simdgroup_layer_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.nax_verify as nax_verify

    observed: dict[str, object] = {}

    def fake_kernel(**kwargs):
        observed.update(kwargs)
        return (mx.zeros((6, 48), dtype=mx.bfloat16),)

    monkeypatch.setattr(
        nax_verify,
        "_build_kernel_m6_qmv_wide_vec6",
        lambda _group_size, _dtype, _k, _n: fake_kernel,
    )
    nax_verify.nax_qmm_m6_qmv_wide_vec6(
        mx.zeros((6, 5120), dtype=mx.bfloat16),
        mx.zeros((48, 640), dtype=mx.uint32),
        mx.zeros((48, 80), dtype=mx.bfloat16),
        mx.zeros((48, 80), dtype=mx.bfloat16),
    )

    assert observed["grid"] == (32, 12, 1)
    assert observed["threadgroup"] == (32, 2, 1)


def test_m6_qmv_wide_vec6_compiles_the_committed_shape_without_tail_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.nax_verify as nax_verify

    observed: dict[str, object] = {}

    def fake_metal_kernel(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(nax_verify.mx.fast, "metal_kernel", fake_metal_kernel)
    nax_verify._VERIFY_KERNEL_CACHE.pop(
        ("m6_qmv_wide_vec6_sg2", 64, mx.bfloat16, 5120, 48),
        None,
    )

    nax_verify._build_kernel_m6_qmv_wide_vec6(
        64,
        mx.bfloat16,
        5120,
        48,
    )

    source = str(observed["source"])
    assert "constexpr int K = 5120;" in source
    assert "constexpr int N = 48;" in source
    assert "min(out_row, N - 1)" not in source
    assert "out_row < N" not in source
    assert observed["input_names"] == ["x", "w_q", "scales", "biases"]


def test_m6_qmv_wide_vec6_uses_sg4_only_for_the_unique_lm_head_shape(
) -> None:
    import mtplx.nax_verify as nax_verify

    assert nax_verify._m6_qmv_wide_simdgroups(5120, 248320) == 4
    assert nax_verify._m6_qmv_wide_simdgroups(5120, 17408) == 2
    assert nax_verify._m6_qmv_wide_simdgroups(17408, 5120) == 2


def test_vk_6bit_hexpack_ksplit_matches_stock() -> None:
    """The 9B-tier 6-bit lane (2026-07-07): MLX packs 6-bit values
    bit-contiguously little-endian; the hexpack kernels must agree with
    stock quantized_matmul within the accumulation-order ULP band."""
    from mtplx.verify_kernels import (
        vk_eligible_ksplit,
        vk_qmm_m4_ksplit,
        vk_qmm_m6_ksplit,
    )

    K, N = 4096, 1024
    for dtype in (mx.bfloat16, mx.float16):
        for gs in (32, 64, 128):
            mx.random.seed(5)
            w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(dtype)
            w_q, scales, biases = mx.quantize(w, group_size=gs, bits=6)
            mx.eval(w_q, scales, biases)
            for m, fn in ((4, vk_qmm_m4_ksplit), (5, vk_qmm_m6_ksplit), (6, vk_qmm_m6_ksplit)):
                assert vk_eligible_ksplit(m, K, N, 6, gs, dtype)
                x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(dtype)
                y = fn(x, w_q, scales, biases, bits=6, group_size=gs)
                ref = mx.quantized_matmul(
                    x, w_q, scales=scales, biases=biases,
                    transpose=True, group_size=gs, bits=6,
                )
                diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
                assert y.shape == (m, N)
                assert diff < 0.05, f"6-bit drift {dtype} gs={gs} M={m}: {diff}"


def test_qlinear_patch_routes_6bit_verify_shapes() -> None:
    """The patch routes 6-bit verify shapes (N >= 2048 floor) through the
    hexpack kernels and leaves small-N projections on stock."""
    from mtplx import verify_kernels

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0}
    orig = verify_kernels.vk_qmm_m4_ksplit

    def counting(*a, **k):
        calls["m4"] += 1
        return orig(*a, **k)

    from mtplx.attention_context import attention_phase

    import mtplx.nax_verify  # noqa: F401  (patch reads through the module)

    verify_kernels.vk_qmm_m4_ksplit = counting
    try:
        big = nn.QuantizedLinear(512, 2048, bias=False, group_size=64, bits=6)
        small = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=6)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "6-bit verify shape did not route the hexpack kernel"
            mx.eval(small(x))
            assert calls["m4"] == 1, "small-N 6-bit projection must stay stock"
        with attention_phase("prefill"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "prefill must stay stock"
    finally:
        verify_kernels.vk_qmm_m4_ksplit = orig
        uninstall_nax_qlinear_patch()
