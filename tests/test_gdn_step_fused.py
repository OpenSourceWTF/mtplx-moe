"""One-dispatch GDN step parity (GPU: Metal).

Three-step decode protocol: step 1 runs the staged chain in BOTH arms (the
gate requires a live delta state, so a fresh cache self-arms on step 2) and
steps 2-3 compare the fused kernel against (a) the shipping convnorm+delta
chain and (b) the fully eager chain. The armed-call counter makes vacuous
parity impossible — a gate that silently refuses fails the test, it does not
pass it (the 08-27 vacuous-parity trap).
"""

import mlx.core as mx
import pytest

import mtplx.kernels.gdn_step_fused as gsf
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
    layer.norm.weight = mx.random.normal(layer.norm.weight.shape) * 0.2 + 1.0
    layer.A_log = mx.random.normal(layer.A_log.shape) * 0.5
    layer.dt_bias = mx.random.normal(layer.dt_bias.shape) * 0.5
    mx.eval(layer.parameters())
    return layer


def _steps(layer, xs):
    cache = _StubCache()
    outs = [layer(x, cache=cache) for x in xs]
    mx.eval(*outs, cache[0], cache[1])
    return outs, cache


def _max_rel(a, b):
    scale = mx.abs(a.astype(mx.float32)).max().item() + 1e-6
    return (
        mx.abs(b.astype(mx.float32) - a.astype(mx.float32)) / scale
    ).max().item()


def test_three_step_decode_parity(gdn, monkeypatch):
    mx.random.seed(5)
    xs = [
        (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16)
        for _ in range(3)
    ]

    monkeypatch.setenv("MTPLX_FUSED_GDN_STEP", "0")
    monkeypatch.setenv("MTPLX_FUSED_GDN_CONVNORM", "0")
    eager, eager_cache = _steps(gdn, xs)
    monkeypatch.setenv("MTPLX_FUSED_GDN_CONVNORM", "1")
    ship, ship_cache = _steps(gdn, xs)

    calls = {"n": 0}
    real = gsf.fused_gdn_step

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gsf, "fused_gdn_step", counting)
    monkeypatch.setenv("MTPLX_FUSED_GDN_STEP", "1")
    fused, fused_cache = _steps(gdn, xs)

    # Anti-vacuous: fresh cache self-arms on step 2 -> exactly steps 2 and 3.
    assert calls["n"] == 2, f"fused step armed {calls['n']} times, expected 2"

    for i, (s, f) in enumerate(zip(ship, fused)):
        err = _max_rel(s, f)
        assert err < 5e-3, f"step {i} vs shipping chain rel err {err}"
    for i, (e, f) in enumerate(zip(eager, fused)):
        err = _max_rel(e, f)
        assert err < 2e-2, f"step {i} vs eager rel err {err}"

    cerr = (
        mx.abs(
            fused_cache[0].astype(mx.float32) - ship_cache[0].astype(mx.float32)
        )
    ).max().item()
    assert cerr < 1e-5, f"conv state err {cerr}"
    derr = (
        mx.abs(
            fused_cache[1].astype(mx.float32) - ship_cache[1].astype(mx.float32)
        )
    ).max().item()
    assert derr < 1e-3, f"delta state err {derr}"
    assert fused_cache[1].dtype == mx.float32


def test_gate_refuses_multirow_ragged_and_cold_state(gdn, monkeypatch):
    monkeypatch.setenv("MTPLX_FUSED_GDN_STEP", "1")
    cache = _StubCache()
    cache[1] = mx.zeros((1, 48, 128, 128), dtype=mx.float32)
    assert gdn._fused_step_applies(1, 1, None, cache)
    assert not gdn._fused_step_applies(1, 3, None, cache)
    assert not gdn._fused_step_applies(1, 1, mx.ones((1, 1)), cache)
    cold = _StubCache()
    assert not gdn._fused_step_applies(1, 1, None, cold)  # no delta state yet
    ragged = _StubCache()
    ragged.lengths = mx.array([1])
    ragged[1] = cache[1]
    assert not gdn._fused_step_applies(1, 1, None, ragged)
    bf16_state = _StubCache()
    bf16_state[1] = cache[1].astype(mx.bfloat16)
    assert not gdn._fused_step_applies(1, 1, None, bf16_state)


def test_gate_refuses_training_module(monkeypatch):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    monkeypatch.setenv("MTPLX_FUSED_GDN_STEP", "1")
    layer = GatedDeltaNet(TextArgs())  # fresh module: training=True
    cache = _StubCache()
    cache[1] = mx.zeros((1, 48, 128, 128), dtype=mx.float32)
    assert not layer._fused_step_applies(1, 1, None, cache)
