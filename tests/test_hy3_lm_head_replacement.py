"""The trunk lm_head quant must REPLACE the bf16 head, not cache a copy beside it.

Regression test for a real defect. The first implementation of Model._logits_head
built `nn.QuantizedLinear.from_linear(self.lm_head, ...)` and stored it in a NEW
attribute (`_mtplx_quant_lm_head`) while leaving `self.lm_head` bound to the
original bf16 module. Because both stayed reachable, enabling the flag did not
free the 990MB bf16 weight — it ADDED a 278MB quantized copy on top of it. With
the draft-head flag also on, a third copy appeared: 990 + 278 + 278 = 1546MB,
i.e. +556MB against a 990MB baseline.

That inverts the entire premise of the lever. The 79-island full-residency config
peaks at 98.0 of the 100 GiB wired limit, so a lever sold as "frees ~712MB" that
actually adds ~556MB is enough to push it into the GPU command-buffer watchdog
(kIOGPUCommandBufferCallbackErrorTimeout) — which is what both arms of the K0-K3
run did, aborting in 22-29s with zero data.

These tests bind the REAL Model._logits_head to a stub rather than re-implementing
its logic, so they fail if the replacement semantics regress.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map

from mtplx.models.hy3_mlx import Model

FLAG = "MTPLX_HY3_LM_HEAD_QUANT_BITS"


def _stub(out: int = 1024, inp: int = 512):
    """A stand-in carrying only what _logits_head touches: .lm_head."""
    lin = nn.Linear(inp, out, bias=False)
    mx.eval(lin.parameters())
    return SimpleNamespace(lm_head=lin)


@pytest.fixture(autouse=True)
def _clean_flag():
    prior = os.environ.get(FLAG)
    os.environ.pop(FLAG, None)
    yield
    if prior is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = prior


def test_flag_off_leaves_the_bf16_head_untouched() -> None:
    stub = _stub()
    original = stub.lm_head
    assert Model._logits_head(stub) is original
    assert stub.lm_head is original


def test_quantizing_REPLACES_the_head_and_drops_the_bf16_module() -> None:
    stub = _stub()
    original = stub.lm_head
    os.environ[FLAG] = "4"

    head = Model._logits_head(stub)

    assert isinstance(head, nn.QuantizedLinear)
    # The returned head IS the model's head — not a copy stored beside it.
    assert stub.lm_head is head
    assert stub.lm_head is not original
    # No attribute may still reference the original bf16 module, or the weight
    # stays resident and the lever adds memory instead of freeing it.
    leaked = [
        name for name, value in vars(stub).items() if value is original
    ]
    assert leaked == [], f"bf16 head still referenced by {leaked}"


def test_quantization_is_idempotent_across_calls() -> None:
    stub = _stub()
    os.environ[FLAG] = "4"
    first = Model._logits_head(stub)
    second = Model._logits_head(stub)
    # Re-quantizing an already-quantized head would raise; sharing proves the
    # guard holds and that repeated calls allocate nothing further.
    assert first is second
    assert stub.lm_head is first


def test_already_quantized_head_is_accepted_not_requantized() -> None:
    lin = nn.Linear(512, 1024, bias=False)
    mx.eval(lin.parameters())
    pre = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    mx.eval(pre.parameters())
    stub = SimpleNamespace(lm_head=pre)
    os.environ[FLAG] = "4"

    assert Model._logits_head(stub) is pre
    assert stub.lm_head is pre


def test_quantized_head_moves_far_fewer_bytes() -> None:
    """The point of the lever: the q4 head is ~3.6x smaller than bf16.

    The real trunk head is BF16 (990MB -> 278MB q4 = 3.56x), so the stub must be
    bf16 too. An fp32 stub reports 6.4x and would overstate the lever by ~1.8x.
    """
    lin = nn.Linear(512, 1024, bias=False)
    lin.update(tree_map(lambda p: p.astype(mx.bfloat16), lin.parameters()))
    mx.eval(lin.parameters())
    q4 = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    mx.eval(q4.parameters())

    def nbytes(module) -> int:
        return sum(
            v.nbytes
            for v in module.parameters().values()
            if isinstance(v, mx.array)
        )

    dense, quant = nbytes(lin), nbytes(q4)
    assert quant < dense
    # q4 + per-group scales/biases at group_size 64 lands near a 3.5x ratio.
    assert 3.0 < dense / quant < 4.5, f"unexpected ratio {dense / quant:.2f}"
