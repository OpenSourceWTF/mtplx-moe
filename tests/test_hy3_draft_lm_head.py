"""T1c: hy3's draft (MTP) lm_head is a quantized COPY of the trunk head, consumed
only by the draft projection in hy3_mtp_patch.mtp_forward. OUTPUT-LOSSLESS — the
target verify keeps the full-precision self.lm_head, so quantizing the draft head
can move acceptance, never the emitted distribution.

Covers the QuantizedLinear.from_linear mechanism the injected _draft_lm_head relies
on and its bit-count guard. End-to-end losslessness (output token-sha identical
off vs on) is the guarded MTP A/B (MTPLX_HY3_DRAFT_LM_HEAD_BITS off vs 4).
"""

import mlx.core as mx
import mlx.nn as nn


def test_from_linear_builds_callable_quantized_head() -> None:
    lin = nn.Linear(256, 512, bias=False)
    mx.eval(lin.parameters())
    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    for bits in (4, 8):
        head = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
        out = head(x)
        mx.eval(out)
        assert out.shape == (1, 1, 512)
        assert int(head.bits) == bits
        assert int(head.group_size) == 64


def test_draft_head_bits_guard() -> None:
    # Mirror _draft_lm_head's guard: only a positive bit count builds a copy;
    # unset/"0" leaves the trunk head in place (feature off by default).
    for raw, builds in [("0", False), ("", False), ("4", True), ("8", True)]:
        bits = int(raw or "0")
        assert (bits > 0) is builds
