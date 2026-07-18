# Hy3 Q2 router kernel A/B — issue #51

**Date:** 2026-07-18 · **Machine:** M5 Max (128 GB) · **Data:** `issue51-hy3-router-kernel-ab-20260718.json`

Six router kernels, 3 reps, **arms interleaved within each rep**, one exclusive
MLX lock hold with qwen guarded down. The router kernel was the only variable.

## Result

| Kernel | n | mean tok/s | reps | vs default |
| --- | --- | --- | --- | --- |
| `mpp-fp32-splitk-r1-fused-r2` | 2 | **40.59** | 40.58, 40.60 | **+1.45%** |
| `mpp-r1-fused-r2` *(shipped default)* | 2 | 40.01 | 40.08, 39.94 | — |
| `mpp-row-owned-fused` | 2 | 38.27 | 38.39, 38.15 | −4.35% |
| `steel-r1-fused-r2` | 2 | 38.25 | 38.33, 38.16 | −4.40% |
| `mpp-r1-last-arrival-fused-r2` | 3 | 37.90 | 37.97, 37.94, 37.78 | −5.27% |
| `stock` | 2 | 37.77 | 37.90, 37.64 | −5.60% |

`mpp-fp32-splitk-r1-fused-r2` applies split-K to the **MTP routers only** —
`splitk_m1 = selector == "mpp-fp32-splitk-r1-fused-r2" and is_mtp_router` in
`configure_hy3_router_kernels` (`mtplx/models/hy3_mlx.py`) — which is why it
separates from plain `mpp-r1-fused-r2`.

## Caveats

- **n=2 for five of six arms.** The intended 3 reps did not complete for most
  arms; aborts cluster at window start (see
  `issue51-hy3-island-abort-rate-20260718.json`). Burn a warm-up cell before
  measuring.
- **Bit-exactness is not uniform across arms.** Four distinct token SHAs appear
  across the six kernels, so "exact variant" naming does not imply bit-exact
  output. Hash the tokens before crediting a delta; a kernel that diverges is
  running a different workload, not just a faster one.
- The +1.45% over the shipped default is small relative to the run-to-run
  spread of the slower arms. The gap to `stock` (+7.5%) is the solid signal.

## Configuration

`--island-layer-count 79` (nothing streams; `--expert-cache-limit` is
vestigial), `--proj-quant q4`, depth 2, 2048 ctx / 1024 out. Full command in
`docs/HY3_SSD_EXPERT_STREAMING.md` → "Current levers" → "Reference
configuration".

## Status

**The default was not changed.** `mpp-r1-fused-r2` remains the default in both
`mtplx/expert_runtime.py` and the benchmark CLI, so the fastest kernel is
opt-in. Flipping it is a pending decision.
