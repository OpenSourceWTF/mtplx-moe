"""Fused GDN conv+silu+l2norm parity (GPU: Metal).

The between-GEMVs kernel must reproduce the eager chain (conv-state concat
-> depthwise conv1d -> silu -> split -> per-head l2norm -> q scale) through
TWO decode steps, so the rolled conv state is exercised, not just the first
window. Tolerance is rounding-order class: the kernel norms the fp32 conv
values where eager norms the bf16-rounded ones."""

import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import GatedDeltaNet, TextArgs


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
    layer.eval()  # serve-mode: the gate refuses training modules
    layer.conv1d.weight = mx.random.normal(layer.conv1d.weight.shape) * 0.3
    mx.eval(layer.parameters())
    return layer


def _steps(layer, xs):
    cache = _StubCache()
    outs = [layer(x, cache=cache) for x in xs]
    mx.eval(*outs, cache[0])
    return outs, cache[0]


def test_two_step_decode_parity(gdn, monkeypatch):
    mx.random.seed(5)
    xs = [
        (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16)
        for _ in range(2)
    ]
    monkeypatch.setenv("MTPLX_FUSED_GDN_CONVNORM", "0")
    ref, ref_state = _steps(gdn, xs)
    monkeypatch.setenv("MTPLX_FUSED_GDN_CONVNORM", "1")
    fused, fused_state = _steps(gdn, xs)
    for r, f in zip(ref, fused):
        scale = mx.abs(r.astype(mx.float32)).max().item() + 1e-6
        err = (
            mx.abs(f.astype(mx.float32) - r.astype(mx.float32)) / scale
        ).max().item()
        assert err < 2e-2, f"step rel err {err}"
    serr = (
        mx.abs(fused_state.astype(mx.float32) - ref_state.astype(mx.float32))
    ).max().item()
    assert serr < 1e-5, f"conv state err {serr}"


def test_gate_refuses_multirow_and_ragged(gdn, monkeypatch):
    monkeypatch.setenv("MTPLX_FUSED_GDN_CONVNORM", "1")
    cache = _StubCache()
    assert gdn._fused_conv_norm_applies(1, 1, None, cache)
    assert not gdn._fused_conv_norm_applies(1, 3, None, cache)
    assert not gdn._fused_conv_norm_applies(1, 1, mx.ones((1, 1)), cache)
    ragged = _StubCache()
    ragged.lengths = mx.array([1])
    assert not gdn._fused_conv_norm_applies(1, 1, None, ragged)
