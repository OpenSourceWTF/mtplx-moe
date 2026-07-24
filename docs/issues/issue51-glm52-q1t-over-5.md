# GLM-5.2 t158 exceeds 5 tok/s by spending its actual cache headroom

The 54 GiB t158 configuration was close but did not reliably clear the goal:
two 1024/1024 replicates averaged 4.477 tok/s AR and 4.986 tok/s at depth 3.
The limiting mistake was carrying forward Q2 cache geometry after the routed
record size changed. The installed t158 records are smaller, so a 54 GiB cache
left 19.6 GiB of the 96 GiB envelope unused.

The fit-for-purpose t158 plan uses a 72 GiB expert cache. It installs 116 fixed
slots per streamed layer instead of 87, retains 1.65 GiB of planned headroom,
and executes the installed route directly. No eligibility check, fallback, or
engagement counter was added to generation.

## Repeated reversed-order result

All rows used the same realistic 1024-token prompt, 1024 generated tokens,
seed 0, Q4 layer-78 MTP head, 96 GiB memory limit, 12 GiB reserve, frequency
cache with layer scope, component banks, 48 transient slots, 8 MiB reads,
stock expert kernel, deferred split release, no prefetch, and no streamed
codec. Each arm ran under the exclusive MLX lock with Qwen stopped; the second
pair reversed candidate/control order.

| cache | AR r1 / r2 | AR mean | d3 r1 / r2 | d3 mean | hard peak |
|---|---:|---:|---:|---:|---:|
| 54 GiB, 87 slots/layer | 4.497 / 4.457 | 4.477 | 4.931 / 5.040 | 4.986 | 71.05 GiB |
| 72 GiB, 116 slots/layer | **5.388 / 5.262** | **5.325** | **5.907 / 5.962** | **5.934** | 84.12 GiB |

The candidate improves mean AR by 18.95% and depth 3 by 19.03%. AR decode hit
rate rises from 0.6998 to 0.7752 and misses fall from 184,240 to 138,003. The
hard peak remains 11.88 GiB below the 96 GiB limit. Both candidate replicates
clear 5 tok/s in AR and depth 3, and all benchmark gates pass.

## Quality boundary

The cache change is bit-identical: cache54/cache72 AR and depth-3 rows share
token SHA-256 `4b0c1741123cad8d59f8a01dfb2d3e9f30c28eef9f97573aa14def0f75a92f31`.
That parity applies only within the t158 artifact. t158 is lossy relative to
the Q2 weights; the earlier synthetic-hidden probe measured 0.92235 mean
combined-output cosine. This result therefore proves the throughput goal for
the t158 lane, not an exact-Q2-quality >5 tok/s result.

The machine-readable receipt is
`research/glm52-q1t-cache72-benchmark-20260718.json`.
