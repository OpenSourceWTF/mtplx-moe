"""Verify-width fused GDN conv+silu+l2norm parity (GPU: Metal).

The S<=6 rows kernel must reproduce the eager verify-block chain (conv-state
concat -> depthwise conv1d -> silu -> split -> per-head l2norm -> q scale)
through a decode step FOLLOWED BY a verify block, so the sliding in-block
conv window and the rolled state are both exercised. It must also leave the
capture-commit stash rows equal to the eager chain's (tolerance =
rounding-order class, same as the shipped S=1 kernel). An anti-vacuous
counter asserts the kernel actually ran on the fused arm.
"""

import mlx.core as mx
import pytest

import mtplx.models.qwen4_exp as q4
from mtplx.models.qwen4_exp import GatedDeltaNet, TextArgs, verify_capture_scope


class _StubCache:
    lengths = None

    def __init__(self):
        self._s = [None, None]

    def __getitem__(self, i):
        return self._s[i]

    def __setitem__(self, i, v):
        self._s[i] = v

    def advance(self, S):
        pass


@pytest.fixture()
def gdn():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(41)
    layer = GatedDeltaNet(TextArgs())  # family geometry
    layer.eval()  # serve-mode: the gates refuse training modules
    layer.conv1d.weight = mx.random.normal(layer.conv1d.weight.shape) * 0.3
    mx.eval(layer.parameters())
    return layer


def _decode_then_verify(layer, x1, xs, capture=False):
    cache = _StubCache()
    o1 = layer(x1, cache=cache)
    if capture:
        with verify_capture_scope():
            o2 = layer(xs, cache=cache)
    else:
        o2 = layer(xs, cache=cache)
    rows = getattr(cache, "_mtplx_verify_rows", None)
    mx.eval(o1, o2, cache[0])
    if rows is not None:
        mx.eval(*rows)
    return o1, o2, cache[0], rows


def _rel(a, b):
    scale = mx.abs(a.astype(mx.float32)).max().item() + 1e-6
    return (
        mx.abs(b.astype(mx.float32) - a.astype(mx.float32)) / scale
    ).max().item()


@pytest.mark.parametrize("s_rows", [2, 4, 6])
def test_verify_block_parity_and_capture_rows(gdn, monkeypatch, s_rows):
    mx.random.seed(7 + s_rows)
    x1 = (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16)
    xs = (mx.random.normal((1, s_rows, 2560)) * 0.5).astype(mx.bfloat16)

    monkeypatch.setenv("MTPLX_FUSED_GDN_CONVNORM", "0")
    monkeypatch.setenv("MTPLX_FUSED_GDN_STEP", "0")
    monkeypatch.setenv("MTPLX_FUSED_CONVNORM_VERIFY", "0")
    r1, r2, rstate, rrows = _decode_then_verify(gdn, x1, xs, capture=True)
    assert rrows is not None

    calls = {"n": 0}
    real = q4.fused_gdn_conv_norm_rows if hasattr(q4, "fused_gdn_conv_norm_rows") else None
    from mtplx.kernels import gdn_conv_norm as gcn

    orig = gcn.fused_gdn_conv_norm_rows

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(gcn, "fused_gdn_conv_norm_rows", counting)
    monkeypatch.setenv("MTPLX_FUSED_CONVNORM_VERIFY", "1")
    f1, f2, fstate, frows = _decode_then_verify(gdn, x1, xs, capture=True)
    assert calls["n"] == 1, "rows kernel did not run — vacuous parity"

    assert _rel(r2, f2) < 2e-2, f"verify block rel err {_rel(r2, f2)}"
    serr = (
        mx.abs(fstate.astype(mx.float32) - rstate.astype(mx.float32))
    ).max().item()
    assert serr < 1e-5, f"conv state err {serr}"

    # capture stash rows: (qkv, q, k, v, a, b) — q/k/v in tolerance class
    for name, rr, ff, tol in (
        ("qkv", rrows[0], frows[0], 0.0),
        ("q", rrows[1], frows[1], 2e-2),
        ("k", rrows[2], frows[2], 2e-2),
        ("v", rrows[3], frows[3], 2e-2),
    ):
        if tol == 0.0:
            assert (rr == ff).all().item(), f"capture {name} must be identical"
        else:
            assert _rel(rr, ff) < tol, f"capture {name} rel err {_rel(rr, ff)}"


def test_rows_gate_refusals(gdn, monkeypatch):
    monkeypatch.setenv("MTPLX_FUSED_CONVNORM_VERIFY", "1")
    cache = _StubCache()
    assert gdn._fused_conv_norm_rows_applies(1, 2, None, cache)
    assert gdn._fused_conv_norm_rows_applies(1, 6, None, cache)
    assert not gdn._fused_conv_norm_rows_applies(1, 1, None, cache)
    assert not gdn._fused_conv_norm_rows_applies(1, 7, None, cache)
    assert not gdn._fused_conv_norm_rows_applies(2, 4, None, cache)
    assert not gdn._fused_conv_norm_rows_applies(1, 4, mx.ones((1, 4)), cache)
    assert not gdn._fused_conv_norm_rows_applies(1, 4, None, None)
    ragged = _StubCache()
    ragged.lengths = mx.array([1])
    assert not gdn._fused_conv_norm_rows_applies(1, 4, None, ragged)
    monkeypatch.setenv("MTPLX_FUSED_CONVNORM_VERIFY", "0")
    assert not gdn._fused_conv_norm_rows_applies(1, 4, None, cache)
