# GLM-5.2 Q1-t158 profiler candidate screen

## Decision

Do not move GLM tensor work from Metal to the CPU.

For the exact 1,024-input / 64-output Q1-t158 + Q4-MTP lane, use speculative
depth 4 rather than depth 3 when reproducing the measured throughput result.
Two fresh-process depth-4 rows reached 7.022466 and 7.015873 token/s. Their
7.019169 token/s median is 2.218% above the fresh same-harness depth-3 control
at 6.866858 token/s. This is a workload-scoped result, not evidence for changing
the default depth of every GLM request.

## Verified 7 token/s result

All three matched rows used the same current coding-agent prompt, greedy
sampler, Q4 MTP head, capture-commit verification, 96 GiB total plan, 12 GiB
runtime reserve, 72 GiB expert-cache ceiling, 109 persistent slots per layer,
48 transient slots, frequency admission, component banks, `F_NOCACHE`,
deferred split release, and context-copy disabled.

| Row | Decode token/s | Verify calls | Drafted / accepted | Verify time | Decode expert bytes | Peak MLX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Depth-3 fresh control | 6.866858 | 17 | 49 / 47 | 9.071044 s | 74,910,597,120 | 90,470,328,522 |
| Depth-4 fresh candidate r1 | 7.022466 | 14 | 54 / 50 | 8.861392 s | 74,627,481,600 | 90,470,328,522 |
| Depth-4 fresh candidate r2 | 7.015873 | 14 | 54 / 50 | 8.864767 s | 74,627,481,600 | 90,470,328,522 |

Every row emitted the same 64-token SHA-256
`e2abfa810932ce1e59c8ff0796e14b2983ac777ed5c397acfe9ec3ac1807fc3d`,
finished by length, and passed exact target/MTP final-cache-offset gates.
Depth 4 wins by amortizing target verification over more accepted tokens:
it removes three target verify calls even though it drafts five more tokens.
It does not win by increasing the cache or changing the t158 kernel. Expert
requests rise from 39,600 to 40,800, while the realized route sequence happens
to read 270 MiB fewer expert bytes.

Evidence:

- `profiler/glm52-other-candidates-20260726/depth3-control-fresh-r3.json`
- `profiler/glm52-other-candidates-20260726/depth4-candidate-fresh-r1.json`
- `profiler/glm52-other-candidates-20260726/depth4-candidate-fresh-r2.json`

## Additional candidate outcomes

The projection-front read split is rejected. An ownership-safe isolated gate
used exact 8.4375 MiB GLM t158 records and bit-exact M=1/2/3/4 miss-service
shapes. Gate/up-first loses 3.946% at M=1 in both arm-order strata. M=1 is
76.941% of observed decode expert occurrences; trace-weighting all shapes
projects a 1.342% service-time regression. The M=3/4 gains do not justify the
old partial-route lifecycle risk or a low-coverage hot route.

The existing `overlap_miss_reads` construction arm is also rejected for this
workload. It engaged on 1,273 decode routes and 8,467 records with identical
tokens, cache behavior, and 74,910,597,120 read bytes, but produced 6.379674
token/s, 5.390% below the original 6.743119 token/s two-control median.
Telemetry measured 0.649 s of resident dispatch overlap and 5.761 s of exposed
miss wait: batching removed useful per-expert read concurrency.

Using more persistent cache proves a hit-rate improvement but not a throughput
improvement. An 11 GiB runtime reserve installs 111 slots/layer, cuts reads
2.126%, and stays at a 91.797 GB peak, yet produces only 6.561473 token/s.
An 8 GiB reserve installs 116 slots/layer, raises hit rate from 75.919% to
77.992%, and cuts reads 8.256%, but its 95.115 GB peak crosses the observed
memory-pressure knee and collapses throughput to 1.937786 token/s.

Additional evidence:

- `profiler/glm52-other-candidates-20260726/t158-projection-overlap-isolated-r1.json`
- `profiler/glm52-other-candidates-20260726/moe-overlap-candidate-fresh-r1.json`
- `profiler/glm52-other-candidates-20260726/cache-reserve11-candidate-fresh-r1.json`
- `profiler/glm52-other-candidates-20260726/cache-reserve8-candidate-fresh-r1.json`

The profiled lane is the raw-t158 `component-banks` route, not the optional
`fused-rans` layout. Apple Silicon gives the CPU and GPU one unified-memory
fabric. A CPU projection or CPU t158 decode would therefore compete for the
same DRAM bandwidth, add a device/host execution dependency, and leave the
GPU waiting for a slower producer. CPU decoding would additionally expand the
compressed weights and require the GPU to read the expanded result again.

The useful CPU work is storage and control work that does not reread the
expert tensors. The current component-bank route already starts SSD miss
reads, dispatches resident-hit expert work on Metal, and runs the shared
expert while misses are pending.

## Profiler bounds

The retained GLM-5.2 Q1-t158 D3 decode produced 16 tokens in 2.959133 seconds
at 5.406989 token/s. Its MLX dispatch census measured:

- 1,580.717 ms of summed GPU work;
- 1,259.215 ms of GPU-busy union time;
- 216.451 ms of host-exposed decode-window time;
- 8,985 raw-t158 projection kernels;
- 289.664 ms in t158-only command buffers as a strict lower bound;
- 818.386 ms in command buffers containing t158 work as an upper bound.

The 216.451 ms host-exposed interval is only 7.31% of decode wall time.
Erasing all of it, which is not achievable, would cap throughput at about
5.83 token/s, a 7.89% improvement. It is the relevant ceiling for additional
CPU/I/O overlap; the summed GPU-work number is not CPU-offload headroom.

Decode expert streaming read 21,852,979,200 bytes for 16 output tokens:
2,862 misses among 14,400 expert requests, an 80.125% hit rate. This makes
miss-I/O scheduling worth an exact A/B, but it does not make CPU tensor math
attractive.

## Measured candidates

| Candidate | Exact result | Decision |
| --- | ---: | --- |
| K3 weight-owner fusion | 5.47x slower GPU projection mix | Reject |
| Capture-commit verification | 1.328x decode throughput | Already the public server default |
| Construction-time prebound t158 callable | 0.071% slower projection mix | Reject as wash |
| Real-shape t158 threadgroup sweep | 0.649% installed three-projection speedup | Adopted for GLM q1t only at David's direction |

The K3 result is from a zero-drop MLX profiler trace with one target operation
per command buffer. Gate/up regressed from 0.340791 ms to 1.972167 ms, and
down regressed from 0.313896 ms to 1.499458 ms.

Capture-commit was measured in one loaded runtime with alternating
`batched / capture_commit / capture_commit / batched` order. All four
16-token outputs were identical and all final target/MTP cache-offset gates
passed. Median decode throughput rose from 9.953303 to 13.219897 token/s, but
`mtplx serve` already defaults to `capture_commit`, so this is an explanation
of the modern product path rather than a new change.

The repeated geometry screen used the exact 28-row, 17-slot GLM route with 20
warmups and 400 alternating samples per arm. Gate/up improved by 2.30% at 32
threads, while down improved by only 0.25% at 64 threads. The independently
best projection mix fell from 1.555166 ms to 1.529562 ms. Applying that
improvement only to the profiler's t158 lower/upper time bounds yields a
0.16% to 0.46% full-decode estimate.

After installation, the complete gate/up/SwiGLU/down path was remeasured in
one guarded window with 20 warmups and 400 alternating samples. The unchanged
128/128/128 default took 1.140563 ms median; the GLM-specific 32/32/64 route
took 1.133208 ms, a 0.649% speedup, with bit-exact output. Model identity and
codec are resolved once when the switch is constructed. Hy3 and other t158
models retain the generic 128-thread default.

## Speculation screen

The retained profiler run cannot score MTP route predictors because its
configuration explicitly has `trace_routes=false`. It does show why the next
capture is worth doing: K3 produced an 80.125% expert-cache hit rate, but the
frequency cache was trained by all verify rows before acceptance was known.
Rejected suffix rows therefore influence persistent residency.

An older 1,023-step GLM autoregressive route trace was replayed chronologically
with the current 109 slots/layer and 8.4375 MiB t158 records. On the held-out
last 512 steps:

- frequency admission hit 76.13% and read 73,343 records;
- LRU hit 74.68% and read 77,776 records;
- a cold-start Belady lower bound read 40,301 records.

The frequency policy remains better than LRU, but the clairvoyant floor shows
that useful retention headroom exists. This does not prove that a deployable
predictor can realize it.

Cheap GLM history predictors failed the offline gate:

- previous-token route: 17.46% recall at 8 candidates/layer;
- union of the last four routes: 34.17% recall at 26.21 candidates/layer;
- trained layer-specific transition table: 14.61% recall at 8 candidates, or
  35.40% at 32 candidates.

The ranked follow-ups are therefore:

1. run an exact fixed-depth-5 A/B against depth 4, then qualify the winning
   depth across multiple prompts and 256-1,024 output tokens;
2. capture draft-route, target-verify-row, acceptance-depth, and cache-miss
   alignment in one diagnostic run, then score draft-route prefetch offline;
3. screen a raw-t158 gate/up/SwiGLU kernel on the exact observed shapes before
   considering full-model integration.

None should be installed before the trace/replay gate. Every wrong full-record
hint on the current q1t artifact costs 8.4375 MiB.

## Remaining candidates

Depth 5 is the highest-priority next experiment. Depth 4 accepted 50 of 54
drafts and reduced target verify calls from 17 to 14. A depth-5 verify wave uses
six rows times top-k 8, exactly matching the current 48 transient slots. That
makes it a measurable boundary rather than a reason to change the global
default: it must beat the matched depth-4 median in fresh processes without
increasing memory pressure or changing tokens.

The I/O candidates tested in this campaign are closed. `overlap_miss_reads`
regressed throughput 5.390%; commitment-only cache credit increased reads; and
projection-front loading projects a 1.342% service-time regression. A new
draft-route prefetcher is only credible if offline replay first proves useful
recall with bounded byte amplification and cache pollution. The authoritative
target router must continue to own admission and eviction.

The next GPU candidate, if a sub-percent campaign is worthwhile, is a
raw-t158 gate/up/SwiGLU screen that preserves the two BF16 projection
rounding boundaries. It can remove one dispatch and intermediate activation
traffic, but it cannot remove either projection's weight reads, so its likely
ceiling is small and it must be rejected before full-model work if the exact
real-shape microbenchmark washes.

## Evidence

- `benchmarks/results/profiler/glm52-t158-owner-fusion-20260726/profiler-report.json`
- `benchmarks/results/profiler/glm52-other-candidates-20260726/capture-commit-current-prompt-pair.json`
- `benchmarks/results/profiler/glm52-other-candidates-20260726/t158-prebound-paired.json`
- `benchmarks/results/profiler/glm52-other-candidates-20260726/t158-geometry-sweep-r2.json`
- `benchmarks/results/profiler/glm52-other-candidates-20260726/t158-installed-path-ab.json`
