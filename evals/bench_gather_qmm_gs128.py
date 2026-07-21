#!/usr/bin/env python3
"""Queued-lane microbench: affine-2bit gather_qmm at gs64 vs gs128, M4 wave shapes.

Decides whether the gs128 island fast-path port is constants-only (kernels
equal) or a real MLX-kernel tiling problem (gs128 slower). Timing follows the
queued-lane law: N dispatches queued per timed region, ONE mx.eval sync.
Shapes follow the M4 wave contract exactly — x is [assignments, 1, K] and the
output is asserted [assignments, 1, N] (the [rows,K] calling-convention trap
does 8x the work and fakes wins).

Run INSIDE a guarded GPU window only.
"""

from __future__ import annotations

import json
import time

import mlx.core as mx
import numpy as np

ASSIGNMENTS = 32  # M4 rows x top-8
HIDDEN = 4096
INTER = 1536
EXPERTS = 192
BITS = 2
ITERS = 200
WARMUP = 20


def bench_group_size(group_size: int, bits: int = BITS) -> dict:
    rng = np.random.default_rng(7)
    results = {}
    # One bank per projection, all 192 experts resident, like an island layer.
    for name, out_dim, in_dim in (
        ("gate", INTER, HIDDEN),
        ("up", INTER, HIDDEN),
        ("down", HIDDEN, INTER),
    ):
        src = mx.array(rng.standard_normal((EXPERTS, out_dim, in_dim)).astype("float32")).astype(mx.bfloat16)
        w, s, b = mx.quantize(src, bits=bits, group_size=group_size, mode="affine")
        mx.eval(w, s, b)
        x = mx.array(rng.standard_normal((ASSIGNMENTS, 1, in_dim)).astype("float32")).astype(mx.bfloat16)
        lhs = mx.array(rng.integers(0, EXPERTS, size=(ASSIGNMENTS,)).astype("uint32"))
        mx.eval(x, lhs)

        def call():
            return mx.gather_qmm(
                x, w, s, b,
                rhs_indices=lhs,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

        out = call()
        mx.eval(out)
        assert tuple(out.shape) == (ASSIGNMENTS, 1, out_dim), out.shape

        for _ in range(WARMUP):
            out = call()
        mx.eval(out)

        t0 = time.perf_counter_ns()
        outs = None
        for _ in range(ITERS):
            outs = call()
        mx.eval(outs)
        dt_us = (time.perf_counter_ns() - t0) / 1000 / ITERS
        results[name] = round(dt_us, 2)
    results["sum_us"] = round(sum(v for v in results.values()), 2)
    return results


def main() -> int:
    out = {"iters": ITERS, "assignments": ASSIGNMENTS, "experts": EXPERTS}
    for bits, gs in ((2, 64), (2, 128), (4, 64)):
        out[f"b{bits}gs{gs}"] = bench_group_size(gs, bits)
    out["gs64"] = out["b2gs64"]; out["gs128"] = out["b2gs128"]
    g64, g128 = out["gs64"]["sum_us"], out["gs128"]["sum_us"]
    out["gs128_over_gs64"] = round(g128 / g64, 4)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
