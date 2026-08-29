"""Single-matrix qmv4 parity (GPU: Metal).

The stripped-epilogue port of the verify-width qmv4 family must reproduce
stock mx.quantized_matmul across decode and verify widths (M=1..6), both
shipped group sizes, and a ragged-N tile edge. Ineligible shapes must route
to the stock fallback and stay byte-identical to it.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.kernels.qmv4_matmul import qmv4_eligible, qmv4_matmul


def _qlinear(n, k, group_size, seed):
    mx.random.seed(seed)
    lin = nn.Linear(k, n, bias=False)
    lin.weight = mx.random.normal((n, k)) * 0.08
    q = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=4)
    q.scales = q.scales.astype(mx.bfloat16)
    q.biases = q.biases.astype(mx.bfloat16)
    mx.eval(q.parameters())
    return q


@pytest.fixture(autouse=True)
def _gpu():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")


@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("m", [1, 2, 4, 6])
def test_parity_family_shapes(group_size, m):
    # GDN in_proj-class [K=2560 -> N ragged] and out_proj-class [6144 -> 2560]
    for n, k, seed in ((1032, 2560, 3), (2560, 6144, 7)):
        q = _qlinear(n, k, group_size, seed)
        mx.random.seed(11 + m)
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        assert qmv4_eligible(x, q), f"eligibility refused N={n} K={k} gs={group_size}"
        stock = mx.quantized_matmul(
            x, q.weight, q.scales, q.biases,
            transpose=True, group_size=group_size, bits=4,
        )
        got = qmv4_matmul(x, q)
        # fp32 oracle from the dequantized weights: the honest yardstick.
        # Measured 2026-08-27: M=1 is BYTE-IDENTICAL to stock (same qmv
        # accumulation); M>=2 stock switches to a tiled qmm kernel with a
        # different summation tree, so both paths carry bf16-ulp-class noise
        # against the oracle — gate ours to stock's own noise class instead
        # of to stock's bytes.
        wf = mx.dequantize(
            q.weight, q.scales, q.biases, group_size=group_size, bits=4
        ).astype(mx.float32)
        oracle = x.astype(mx.float32) @ wf.T
        mx.eval(stock, got, oracle)
        scale = mx.abs(oracle).max().item() + 1e-6
        e_stock = (mx.abs(stock.astype(mx.float32) - oracle) / scale).max().item()
        e_ours = (mx.abs(got.astype(mx.float32) - oracle) / scale).max().item()
        if m == 1:
            assert (got == stock).all().item(), (
                f"M=1 must be byte-identical to stock (gs={group_size} N={n})"
            )
        else:
            assert e_ours <= e_stock * 3.5 + 1e-3, (
                f"N={n} K={k} gs={group_size} M={m}: ours->oracle {e_ours:.5f} "
                f"outside stock's noise class ({e_stock:.5f})"
            )


def test_leading_batch_dims_roundtrip():
    q = _qlinear(1032, 2560, 64, 3)
    x = (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16)
    got = qmv4_matmul(x, q)
    assert got.shape == (1, 1, 1032)


def test_fallback_on_ineligible_shapes():
    q = _qlinear(1032, 2560, 64, 3)
    # M > 6 falls back to stock and matches it exactly
    x7 = (mx.random.normal((7, 2560)) * 0.5).astype(mx.bfloat16)
    assert not qmv4_eligible(x7, q)
    ref = mx.quantized_matmul(
        x7, q.weight, q.scales, q.biases, transpose=True, group_size=64, bits=4
    )
    got = qmv4_matmul(x7, q)
    mx.eval(ref, got)
    assert (got == ref).all().item()
    # K not a multiple of 512 refuses eligibility
    q2 = _qlinear(512, 2496, 64, 9)
    x2 = (mx.random.normal((1, 2496)) * 0.5).astype(mx.bfloat16)
    assert not qmv4_eligible(x2, q2)
    # fp32 scales (dtype mismatch with bf16 x) refuses eligibility
    q3 = _qlinear(512, 2560, 64, 13)
    q3.scales = q3.scales.astype(mx.float32)
    q3.biases = q3.biases.astype(mx.float32)
    x3 = (mx.random.normal((1, 2560)) * 0.5).astype(mx.bfloat16)
    assert not qmv4_eligible(x3, q3)
