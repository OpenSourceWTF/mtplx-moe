"""Bit-exactness contract for the headquarter tape-capture kernel.

The headquarter kernel is an execution-layout change only: for every input it
must produce BIT-EQUAL y / final_state / tape versus the incumbent TGY tape
kernel. Runs at the Qwen3.6-27B GDN geometry; skipped without Metal.
"""

from types import SimpleNamespace

import pytest

import mlx.core as mx

from mtplx.gdn_capture import (
    _configured_qmv8_matmul,
    _linear_gated_delta_from_conv_tape_capture,
    _linear_gated_delta_from_conv_tape_capture_configured,
    _linear_gated_delta_from_conv_tape_replay,
    _linear_gated_delta_from_conv_tape_replay_configured,
    bind_qwen_tape_replay,
)
from mtplx.kernels.gdn_tape_headquarter import headquarter_tape_capture

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal"
)


def _gdn():
    gdn = SimpleNamespace(
        conv_dim=10240,
        head_k_dim=128,
        head_v_dim=128,
        num_k_heads=16,
        num_v_heads=48,
        key_dim=2048,
    )
    gdn.A_log = mx.log(mx.random.uniform(low=0.5, high=8.0, shape=(gdn.num_v_heads,)))
    gdn.dt_bias = mx.ones(gdn.num_v_heads) * 0.5
    mx.eval(gdn.A_log, gdn.dt_bias)
    return gdn


def test_tape_replay_reads_geometry_attributes_from_dict_backed_module():
    class DictBackedGDN(dict):
        pass

    gdn = DictBackedGDN()
    gdn.conv_dim = 64
    gdn.head_k_dim = 32
    gdn.head_v_dim = 16
    gdn.num_k_heads = 2
    gdn.num_v_heads = 2
    gdn.key_dim = 64

    replay = bind_qwen_tape_replay(
        gdn_layers=(gdn,),
        target_width=2,
        verified_tokens=3,
        tgy=4,
    )

    assert callable(replay)


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_headquarter_matches_incumbent_bitwise(T, seed, monkeypatch):
    from mlx_lm.models.gated_delta import compute_g

    # The reference arm routes through the env-gated wrapper: a stray
    # MTPLX_LINEAR_GDN_TAPE_IMPL=headquarter in the invoking shell would turn
    # this into headquarter-vs-headquarter and pass vacuously.
    monkeypatch.delenv("MTPLX_LINEAR_GDN_TAPE_IMPL", raising=False)
    mx.random.seed(0)
    gdn = _gdn()
    key = mx.random.key(1000 * T + seed)
    ks = mx.random.split(key, 4)
    conv_out = mx.random.normal((1, T, gdn.conv_dim), key=ks[0]).astype(mx.bfloat16)
    a = (mx.random.normal((1, T, gdn.num_v_heads), key=ks[1]) * 0.5).astype(mx.bfloat16)
    b = (mx.random.normal((1, T, gdn.num_v_heads), key=ks[2]) * 0.5).astype(mx.bfloat16)
    state = (
        mx.random.normal((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), key=ks[3])
        * 0.5
    ).astype(mx.float32)
    beta = mx.sigmoid(b)
    g = compute_g(gdn.A_log, a, gdn.dt_bias)

    ref = _linear_gated_delta_from_conv_tape_capture(conv_out, g, beta, state, gdn)
    cand = headquarter_tape_capture(conv_out, g, beta, state, gdn)
    assert ref is not None and cand is not None
    for name, r, c in zip(("y", "final_state", "tape"), ref, cand):
        mx.eval(r, c)
        assert bool(mx.array_equal(r, c).item()), f"{name} diverged at T={T} seed={seed}"


def test_headquarter_fail_closed_on_bad_geometry():
    gdn = _gdn()
    gdn.head_k_dim = 100  # not divisible by 32 -> wrapper must decline
    conv_out = mx.zeros((1, 1, gdn.conv_dim), dtype=mx.bfloat16)
    g = mx.zeros((1, 1, gdn.num_v_heads))
    beta = mx.zeros((1, 1, gdn.num_v_heads), dtype=mx.bfloat16)
    state = mx.zeros((1, gdn.num_v_heads, gdn.head_v_dim, 100), dtype=mx.float32)
    assert headquarter_tape_capture(conv_out, g, beta, state, gdn) is None


def test_configured_tape_capture_uses_prebound_backend_without_environment(
    monkeypatch,
):
    conv_out = mx.random.normal((1, 2, 64), dtype=mx.float32)
    g = mx.random.normal((1, 2, 2), dtype=mx.float32)
    beta = mx.sigmoid(mx.random.normal((1, 2, 2), dtype=mx.float32))
    state = mx.zeros((1, 2, 16, 32), dtype=mx.float32)
    gdn = SimpleNamespace(
        conv_dim=64,
        head_k_dim=32,
        head_v_dim=16,
        num_k_heads=2,
        num_v_heads=2,
        key_dim=64,
    )
    calls = []

    monkeypatch.setattr(
        "mtplx.kernels.gdn_tape_headquarter.headquarter_tape_capture",
        lambda *args: calls.append(args) or ("out", "state", "tape"),
    )
    monkeypatch.setattr(
        "mtplx.gdn_capture.os.environ.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("configured capture read the environment")
        ),
    )

    result = _linear_gated_delta_from_conv_tape_capture_configured(
        conv_out,
        g,
        beta,
        state,
        gdn,
        implementation="headquarter",
        tgy=8,
    )

    assert result == ("out", "state", "tape")
    assert len(calls) == 1


@pytest.mark.parametrize("batch_width", [1, 2])
def test_configured_rejection_replay_matches_generic_for_fixed_width(
    batch_width,
):
    gdn = SimpleNamespace(
        conv_dim=160,
        head_k_dim=32,
        head_v_dim=16,
        num_k_heads=2,
        num_v_heads=2,
        key_dim=64,
    )
    conv_out = (
        mx.arange(batch_width * 3 * gdn.conv_dim)
        .reshape(batch_width, 3, gdn.conv_dim)
        .astype(mx.bfloat16)
        / 128
    )
    g = (
        mx.arange(batch_width * 3 * gdn.num_v_heads)
        .reshape(batch_width, 3, gdn.num_v_heads)
        .astype(mx.float32)
        / 32
    )
    beta = mx.full(
        (batch_width, 3, gdn.num_v_heads),
        0.5,
        dtype=mx.bfloat16,
    )
    state = mx.zeros(
        (batch_width, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
        dtype=mx.float32,
    )
    captured = _linear_gated_delta_from_conv_tape_capture(
        conv_out, g, beta, state, gdn
    )
    assert captured is not None
    _out, _final, tape = captured
    capture = {
        "tape": tape,
        "conv_out": conv_out,
        "g": g,
        "state_in": state,
    }
    configured = bind_qwen_tape_replay(
        gdn_layers=(gdn,),
        target_width=batch_width,
        verified_tokens=3,
        tgy=8,
    )

    expected = _linear_gated_delta_from_conv_tape_replay(
        tape,
        conv_out,
        g,
        state,
        gdn,
        steps=1,
        tgy=8,
    )
    actual = configured(capture, steps=1)
    mx.eval(expected, actual)

    assert mx.array_equal(expected, actual).item()


def test_configured_replay_source_has_no_dynamic_gate_or_false_fallback() -> None:
    import inspect

    source = inspect.getsource(
        _linear_gated_delta_from_conv_tape_replay_configured
    )

    assert "os.environ" not in source
    assert "return None" not in source
    assert "return False" not in source
    assert "if " not in source


def test_configured_qmv8_flattens_every_leading_dimension() -> None:
    import mlx.nn as nn

    linear = nn.Linear(256, 16, bias=False)
    linear.weight = (
        mx.arange(16 * 256).reshape(16, 256).astype(mx.bfloat16) / 4096
    )
    module = nn.QuantizedLinear.from_linear(
        linear,
        group_size=64,
        bits=8,
    )
    x = (
        mx.arange(2 * 3 * 256).reshape(2, 3, 256).astype(mx.bfloat16)
        / 1024
    )

    expected = module(x)
    actual = _configured_qmv8_matmul(x, module)
    mx.eval(expected, actual)

    assert actual.shape == (2, 3, 16)
    assert mx.array_equal(expected, actual).item()
