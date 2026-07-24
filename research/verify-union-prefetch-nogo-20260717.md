# Verify-union prefetch — offline coverage gate: NO-GO (issue #106)

2026-07-17. Branch `feat/verify-union-prefetch` off
`experiment/issue51-stack-69`. Offline analysis only; no runtime change
shipped.

## What was gated

Issue #106 lever 2: at K2/K3 the verify forward is the only full-trunk
pass per step, and the draft tokens are known before verify launches, so
the verify batch's per-layer expert routes are "partially predictable"
ahead of time. The proposed v1 (from the task design) is a verify-phase
*boost* of the existing residual-stream lookahead: at verify launch,
re-issue the decode-lookahead's per-layer route predictions for the last
committed token under a raised speculative budget, so the prefetch ring
pre-warms exactly what the verify rows will miss on.

Before building anything, the gate asks: **does the previous token's
L=1..3 lookahead prediction actually cover the next token's per-layer
routed experts?** If coverage `< ~50%`, the feature dies here.

The prediction mechanism is the shipped lookahead
(`SparseMLP._maybe_prefetch_lookahead` in `mtplx/models/hy3_mlx.py`):
for target sparse layer `M`, predict its route from the up-to-3 preceding
sparse layers' router inputs,
`pred_M(t) = U_{S in {M-1,M-2,M-3}} top8( router_M(h_S(t)) )`.
The verify-union bet is that `pred_M(t)` — loads issuable while token `t`
is being processed — covers `route_M(t+1)`, the route the *next* verify
forward will demand.

## Data + method

- Trace: `/tmp/issue51-lookahead.npz` — 400 decode tokens x 79 streamed
  layers, per-(token,layer) router input `h` (fp16) and actual top-8
  route (issue #51 C6 capture, 90g config). Token-major, layers 1..79.
- Routers reconstructed exactly as `Router.__call__` for the affine-Q8
  gate: `quantized_matmul` (activation dtype), sigmoid, `+expert_bias`
  (fp32), top-8. Gate weights read lazily from the `pipenetwork/Hy3-4bit`
  shards (`mx.load` mmap; only the 192x4096 router gates fault in, never
  the expert weights).
- Scripts (committed alongside this doc):
  `research/verify_union_gate_trace_only.py` (persistence ceiling, no
  router weights) and `research/verify_union_gate_real_lookahead.py`
  (real router predictions).

**Pipeline validation (all pass):**

| check | expected | measured |
|---|---|---|
| `router_M(h_M)` vs trace `route_M` | ~100% | **99.25%** |
| `router_M(h_{M-1})` vs same-token `route_M` | ~74.3% | **74.4%** |
| `router_M(h_{M-2})` vs same-token `route_M` | ~66% | **66.1%** |
| `router_M(h_{M-3})` vs same-token `route_M` | ~61% | **60.8%** |

The router reconstruction is exact and the within-token overlaps
reproduce the lookahead docstring's 74.3/66/61% to a tenth of a point —
so the coverage numbers below are trustworthy, not an artifact of a
mismodeled router.

## Result: coverage is 29.4%, far under the 50% gate

Verify-union coverage = mean over streamed layers and token pairs of
`|pred_M(t) ∩ route_M(t+1)| / 8`:

| predictor | coverage | candidates/layer |
|---|---|---|
| union L=1 (nearest only) | 24.5% | 8.0 |
| union L=1..2 | 27.3% | 9.9 |
| **union L=1..3 (the shipped lookahead)** | **29.4%** | 11.4 |

Cross-token persistence (the ceiling — if the lookahead perfectly
reproduced the previous token's route), from actual routes only:

| predictor | coverage |
|---|---|
| prev token, same layer (K=1) | 25.5% |
| union of prev 2 tokens (~K=2) | 34.5% |
| union of prev 3 tokens (~K=3) | 41.3% |
| union of prev 4 tokens (~K=4) | 45.9% |
| identity / random-8 baseline | 4.2% |

Deeper verify rows are worse, not better: union-L=1..3 from token `t`
covers `route(t+1)` 29.4%, `route(t+2)` 24.0%, `route(t+3)` 23.1%.

The only way to reach the gate is to spend the whole budget widening the
net — union the L=1..3 predictions of the last `W` committed tokens:

| W (committed tokens unioned) | coverage | candidates/layer | speculative load cost |
|---|---|---|---|
| 1 | 29.4% | 11.4 | ~1.4x an 8-expert route |
| 2 | 39.1% | 19.6 | ~2.5x |
| 3 | 46.2% | 26.6 | ~3.3x |
| 4 | **50.9%** | 32.7 | **~4.1x** |

Crossing 50% costs ~4.1x the per-layer speculative I/O of a single
route, and only ~half of those loads are ever used. The prefetch lane
caps speculation at `speculative_io_fraction = 0.25` of inflight-miss
budget precisely to stay off the demand-miss critical path; a 4x-wide
speculative fan competes head-on with the demand misses that #106 itself
names as verify's dominant cost ("verify's dominant cost is streamed-
layer misses on the row union"). Spending 4x SSD bandwidth to prewarm a
set that is half wrong is a net loss against that critical path.

## Root cause

The lookahead is a strong *within-token* route predictor — union L=1..3
covers the **same** token's route 80.8% — but expert routes decorrelate
sharply across adjacent tokens. Same-layer route persistence from token
`t` to `t+1` is only 25.5%. The verify rows are next-token positions, so
the lookahead's within-token strength (its entire reason to exist on the
decode lane) does not transfer to the verify-union target. There is no
cheap per-layer signal in token `t` that predicts token `t+1`'s route
above ~30% at the natural 1x budget.

Secondary finding (relevant to #106, not required for the gate): in K2/K3
speculative decoding the trunk decode lookahead never fires anyway —
`_maybe_prefetch_lookahead` gates on `x.shape[-2] == 1`, and the only
full-trunk pass is the multi-row verify forward (`rows = K+1 >= 2`). So
the "boost the predictions already issued" framing has nothing to boost
during spec decode; a verify-union would have to generate the predictions
fresh, at the 29.4%-useful hit rate measured above.

## Decision

**No-go.** Coverage (29.4%) is well below the ~50% gate at any budget the
speculative lane can afford, and the root cause is structural (cross-token
route decorrelation), not a tuning gap. Building the verify-phase boost
would add router compute and 1.4x-4x speculative SSD traffic to warm a set
that is <=30% relevant, directly taxing the demand-miss path that is the
real verify cost. The miss-economy levers #106 already lists (prefetch
issue-path fix, islands, cache) remain the verify speedup; this predictor
is not a viable feeder for them.

Reproduce: `python research/verify_union_gate_real_lookahead.py`
(needs `/tmp/issue51-lookahead.npz` and the `pipenetwork/Hy3-4bit`
snapshot in the HF cache).
