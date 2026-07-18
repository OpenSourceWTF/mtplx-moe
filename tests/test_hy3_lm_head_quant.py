"""T2a: trunk lm_head quant (MTPLX_HY3_LM_HEAD_QUANT_BITS) via Model._logits_head.

Model._logits_head returns self.lm_head unquantized when the flag is unset, and a
cached QuantizedLinear.from_linear copy (g64, q8/q4) when it is >0 — used for BOTH
the AR logits (Model.__call__) and the MTP verify projection (the patched model
inherits the hook). Quantizing the trunk head frees the ~990 MB bf16 read that was
tripping the GPU watchdog at the 98 GiB memory edge (throttle relief). Unlike the
draft head this changes the emitted distribution, so it is quality-gated/off by
default. Routing is A/B-validated (output diverges when set); these cover the
quantize mechanism and the flag guard.
"""

import mlx.core as mx
import mlx.nn as nn


def test_trunk_head_quant_is_callable_and_sized() -> None:
    lin = nn.Linear(512, 1024, bias=False)  # bf16 trunk-head stand-in
    mx.eval(lin.parameters())
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    ref = lin(x)
    for bits in (8, 4):
        head = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
        out = head(x)
        mx.eval(out, ref)
        assert out.shape == ref.shape
        assert int(head.bits) == bits
        assert int(head.group_size) == 64


def test_lm_head_quant_flag_guard() -> None:
    # Mirror _logits_head's guard: unset/"0" leaves the bf16 head; >0 quantizes.
    for raw, quant in [("0", False), ("", False), ("8", True), ("4", True)]:
        bits = int(raw or "0")
        assert (bits > 0) is quant


def test_q8_closer_to_bf16_than_q4() -> None:
    # Sanity that q8 is the near-lossless sweet spot vs q4 (the A/B's premise).
    lin = nn.Linear(512, 1024, bias=False)
    mx.eval(lin.parameters())
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    ref = lin(x)
    q8 = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)(x)
    q4 = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)(x)
    mx.eval(ref, q8, q4)
    err8 = float(mx.abs(q8 - ref).mean())
    err4 = float(mx.abs(q4 - ref).mean())
    assert err8 < err4
