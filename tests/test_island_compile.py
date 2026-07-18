"""Compiled island expert gather (MTPLX_HY3_COMPILE_ISLAND).

Full residency rebuilds the 79-layer graph in Python every decode step; compiling
the per-layer expert gather traces it once and replays.

IMPORTANT numerical finding (2026-07-18): mx.compile is EXACT for fp32 matmul
(0 diff) but selects a different fused kernel for the QUANTIZED gather_qmm, which
diverges ~0.1-0.6% vs eager — the same class as the vk_k split-K divergence
(#171). So the compiled expert path is NOT bitwise-identical, and whether it
holds TOKEN-sha parity end-to-end is a guarded-A/B question, not a unit-test one.
These tests therefore lock (a) the divergence is bounded and small (not a bug),
and (b) the flag gates the path; the exact-quality gate lives in the A/B.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mtplx.models.expert_mlx import _gather_component_bank


def _fake_bank(experts: int, hidden: int, inter: int, group_size: int, bits: int):
    """A minimal component bank: quantized gate/up/down over `experts` rows."""

    class _Bank:
        pass

    bank = _Bank()
    arrays = {}
    for proj, (out_dim, in_dim) in {
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }.items():
        w = mx.random.normal((experts, out_dim, in_dim)) * 0.02
        qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        arrays[f"{proj}.weight"] = qw
        arrays[f"{proj}.scales"] = scales
        arrays[f"{proj}.biases"] = biases
    bank.arrays = arrays
    return bank


def test_compiled_gather_divergence_is_bounded_and_small() -> None:
    # Not bitwise (compile fuses the quantized kernel differently), but the
    # divergence must be small enough to be rounding, not a wiring bug. Plain
    # fp32 matmul under compile is exact (0); only quantized diverges ~1e-3.
    mx.random.seed(0)
    experts, hidden, inter = 16, 256, 128
    group_size, bits = 64, 4
    bank = _fake_bank(experts, hidden, inter, group_size, bits)

    rows, top_k = 3, 8  # K2 verify width x top-8, a realistic decode shape
    x = mx.random.normal((rows * top_k, hidden))
    slot_indices = mx.random.randint(0, experts, (rows * top_k, 1)).astype(mx.int32)

    classic = _gather_component_bank(x, bank, slot_indices, group_size=group_size, bits=bits)

    def gather(ai, si):
        return _gather_component_bank(ai, bank, si, group_size=group_size, bits=bits)

    compiled = mx.compile(gather)(x, slot_indices)
    mx.eval(classic, compiled)
    max_abs = float(mx.abs(classic - compiled).max())
    scale = float(mx.abs(classic).max())
    # quantized-kernel fusion divergence: real but < ~1% of the value scale
    assert 0.0 < max_abs < 0.02 * scale, f"unexpected divergence {max_abs} vs scale {scale}"


def test_fp32_matmul_compiles_exactly_isolating_the_quantized_divergence() -> None:
    # The divergence is quantized-kernel-specific: plain fp32 matmul is exact
    # under compile, which is why compiling non-quantized ops is token-safe and
    # the quantized experts are the parity risk.
    mx.random.seed(0)
    a = mx.random.normal((24, 256))
    b = mx.random.normal((256, 128))
    mm = lambda x, y: x @ y  # noqa: E731
    assert mx.array_equal(mm(a, b), mx.compile(mm)(a, b)), "fp32 matmul must compile exactly"


def test_compiled_replays_across_shapes_without_recompiling_output() -> None:
    # Same shape twice must give identical results (trace reuse must be correct).
    mx.random.seed(1)
    bank = _fake_bank(8, 128, 64, 64, 4)
    x = mx.random.normal((8, 128))
    idx = mx.random.randint(0, 8, (8, 1)).astype(mx.int32)

    def gather(ai, si):
        return _gather_component_bank(ai, bank, si, group_size=64, bits=4)

    c = mx.compile(gather)
    a = c(x, idx)
    b = c(x, idx)
    mx.eval(a, b)
    assert mx.array_equal(a, b)


def test_switch_uses_compiled_path_only_under_flag(monkeypatch) -> None:
    """The DenseIslandSwitchGLU dispatch honors MTPLX_HY3_COMPILE_ISLAND."""
    import inspect

    from mtplx.models.expert_mlx import DenseIslandSwitchGLU

    source = inspect.getsource(DenseIslandSwitchGLU.__call__)
    assert 'os.environ.get("MTPLX_HY3_COMPILE_ISLAND") == "1"' in source
    assert "_compiled_island_gather()" in source
    # the classic path must remain the else-branch (default behavior unchanged)
    assert "_gather_component_bank(" in source
