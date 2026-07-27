# GLM-5.2 Q1-t158 profiler candidate screen

## Decision

Do not move GLM tensor work from Metal to the CPU.

For the exact 1,024-input / 64-output Q1-t158 + Q4-MTP lane, fixed speculative
depth 5 is the best qualified fixed depth. Its fresh-process ABBA median is
7.114647 token/s, 0.776% above the matched depth-4 median at 7.059836 token/s.

A rejection-aware K2-for-three-cycles then K5 schedule is the highest-value
new candidate. Its fresh-process ABBA median is 7.389868 token/s, 3.725% above
the matched fixed-K5 median at 7.124458 token/s. It removes both cold
rejections, reduces verify time 6.178%, and preserves exact output and final
cache offsets. Do not install it yet: the retained proof is one prompt and one
64-token phase alignment, so it still needs multi-prompt and 256-1,024-output
qualification. The result is evidence for rejection-aware depth selection,
not for changing every GLM request to the same token threshold.

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

## Depth 5 and rejection-aware depth

The fixed-depth ABBA order was K4 / K5 / K5 / K4 with a fresh model process
per row:

| Fixed depth | Decode token/s rows | Median | Verify calls | Drafted / accepted | Rejects |
| --- | ---: | ---: | ---: | ---: | ---: |
| K4 | 7.041807, 7.077866 | 7.059836 | 14 | 54 / 50 | 2 |
| K5 | 7.124527, 7.104767 | 7.114647 | 12 | 58 / 52 | 2 |

K5 wins by 0.776%. It removes two more target verify calls than K4 while
retaining the same token hash, exact target offset 1,088, exact MTP offset
1,087, and the same 90,470,328,522-byte MLX peak. It is a clean but small
workload-local win: reads rise by 203,489,280 bytes and evictions rise from
3,201 to 3,253.

The discovery-only K5 route trace proves where the remaining waste occurs.
The first two verify calls each route all 75 target MoE layers at six input
rows. Both accept drafts one and two, reject draft three, and never evaluate
drafts four and five. The later nine complete K5 calls accept all five drafts.
Capture-commit repair takes about 43 microseconds on each miss; the useful
target is therefore avoiding the wasted target rows, not moving host repair
work.

The existing late-depth construction parameters were then screened with a
generated-token threshold of nine. Because an accepted bonus is already
emitted as the next cycle primary, the observed route was K2 for three cycles
then K5, not two K2 cycles. The ABBA order was fixed K5 / K2-then-K5 /
K2-then-K5 / fixed K5:

| Route | Decode token/s rows | Median | Verify time median | Drafted / accepted | Rejects | Read bytes | Evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed K5 | 7.120053, 7.128863 | 7.124458 | 8.724922 s | 58 / 52 | 2 | 240,064,266,240 | 3,253 |
| K2 x3 then K5 | 7.403696, 7.376040 | 7.389868 | 8.185870 s | 51 / 51 | 0 | 236,436,848,640 | 2,967 |

The candidate improves decode throughput 3.725%, reduces verify time 6.178%,
reads 1.511% fewer bytes, and causes 8.792% fewer evictions. All four rows pass
the same exactness gates and emit the same SHA-256. This is the first candidate
in the campaign whose full-lane gain comes primarily from rejection
performance rather than deeper speculation or a sub-percent kernel change.

The BF16 MTP artifact does not improve prediction accuracy enough to justify
its size. Against the same target output, Q4 and BF16 both accept 52 of 58
drafts and reject twice. BF16 moves the rejection depths from 3/3 to 4/2:
it fixes one Q4 draft but creates different misses. Its 19,905,942,064-byte
artifact forces the target cache from 109 to 88 slots/layer under the same
96 GiB cap, compared with the 6,014,594,720-byte Q4 artifact. Reject BF16 for
this lane.

Pre-verification confidence also fails the cross-prompt gate. On the original
prompt, top-8 entropy appeared to separate the two rejected drafts cleanly:
accepted drafts had entropy at most 1.367242 while rejected drafts were at
least 1.480905. A 1.4 threshold would have caught both without a false stop.
That separation disappears on an alternate Python JSONL task. K5 accepts only
34 drafts and rejects 29; accepted entropy reaches 1.865469 while rejected
entropy falls to 0.701155. The same 1.4 threshold falsely stops six accepted
drafts, catches 16 rejections, and misses 13. Reject per-draft confidence
gating rather than adding its top-k/host-synchronization cost to the hot path.

The alternate prompt also exposed a final-state correctness bug at the
performance boundary. When `max_tokens` was reached by a freshly sampled
primary, `generate_mtpk` emitted the token and exited before forwarding it
into the target and committed-MTP caches. The final state then reported
`safe_to_commit=true` with both offsets one token short. The terminal branch
now marks that fresh primary pending and reuses the existing final commit path.
The real GLM rerun preserves the exact 64-token SHA-256 while correcting target
offset 1,087 to 1,088 and MTP offset 1,086 to 1,087. The missing terminal
commit measured 0.185064 seconds on this rejection-heavy row, so the old
3.973250 token/s diagnostic was also slightly overstated; the corrected row is
3.934064 token/s.

Compact machine-readable proof:

- `profiler/glm52-other-candidates-20260726/depth5-rejection-screen-summary.json`

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

1. replace the fixed cold-prefix threshold with a rejection-history controller
   and qualify it across prompts with both high and collapsed K5 acceptance;
2. qualify 256-1,024 output lengths against unchanged fixed K5 after every
   terminal final-state gate passes;
3. screen a raw-t158 gate/up/SwiGLU kernel on the exact observed shapes before
   considering full-model integration.

None should be installed before the trace/replay gate. Every wrong full-record
hint on the current q1t artifact costs 8.4375 MiB.

## Remaining candidates

Depth 5 has cleared its fresh-process fixed-depth gate, and K2-for-three-cycles
then K5 has cleared the exact current-prompt rejection gate. The remaining
promotion boundary is generalization: a token-count threshold can benefit from
64-token phase alignment without being the best policy for longer generation.
Qualification must preserve the target path's output, cache offsets, 96 GiB
plan, and unchanged fixed-K5 control on every prompt/output stratum.

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
- `benchmarks/results/profiler/glm52-other-candidates-20260726/depth5-rejection-screen-summary.json`
