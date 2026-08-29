# Qwen4 PR #368 optimization ledger

This is the durable candidate and evidence ledger for the Qwen3.8 Flash-Next
Qwen4 depth-one MTP work in PR #368. It records ideas even when they are not
retained so later work does not repeat rejected experiments or transplant an
optimization by name alone.

The upstream audit snapshots are `mlx-serve` at
`bbe27908df6034b3cfc7071889d8cd5577f44496` and the named oMLX commits below.
"Candidate" never means safe to copy: every item still has to
preserve the target lane's arithmetic, state ownership, fixed M=2 shapes, and
compilation behavior, then beat the unchanged production control.

## Production benchmark contract

- 16,384 input tokens and exactly 1,024 generated tokens.
- Python-programming fixture; thinking enabled with low reasoning effort.
- Target sampler: temperature 1.0, top-p 0.95, top-k 20, min-p 0.0,
  presence penalty 0.0, repetition penalty 1.0.
- Native MTP depth 1, physical target verifier M=2, capture-prefix commit,
  one warmup, 1 GiB n-gram payload ceiling.
- Primary score: decode TPS. Every retained scheduling candidate must also
  report decode-window GPU busy time, idle time, and utilization.
- Current repeatable frontier: 69.8365 and 69.6411 TPS; mean 69.7388 TPS.

The current profiler receipt measured 12.6535 seconds GPU-active and 2.3191
seconds GPU-idle over the 14.9725-second decode window, or 84.51% utilization.
The most expensive active command-buffer families are the 22-op linear blocks
(5.2841 seconds), full-attention blocks (1.8790 seconds), and separate MoE
stage-3 residual/down kernels (1.3935 seconds).

## Retained MTPLX work

| Commit | Mechanism | Exact-shape fit / evidence | Status |
|---|---|---|---|
| `6c5ca95d`, `7d14bc47` | Fuse the exact two-row MoE stages and row-paired routing | First construction-bound Qwen4 M=2 MoE route | retained foundation |
| `8ceead08` | Fuse short-row hyper projections | Qwen4 short-row verifier geometry | retained foundation |
| `7894c62e`, `32673fa3` | Fuse small-row input projections and preserve packed GDN projection views | Removes projection launches without repacking weights in the hot path | retained |
| `2b051654` | Commit the already-computed accepted verify prefix | Production receipt has zero repair forwards | retained |
| `c4b23d6b`, `01fa94b5` | Fuse captured sigmoid/norm/gate and the verifier hyper boundary | Fixed Qwen4 M=2 capture path | retained |
| `b1d19667` | Right-size M=2 router occupancy | Geometry derived for the physical two-row verifier | retained |
| `0ca7672c` | Own M=2 n-gram row copies | Fixed two-row PLE input ownership | retained |
| `2deb1dae`, `508452ee`, `24367143` | Own top-10 routing and stage 3, then prebind the M=2 MoE stages | Removes generic routing and invariant binding work from the installed hot lane | retained |
| `80505ff0` | Fuse PLE key/value projections | Qwen4 PLE path; retained in the production stack | retained |
| `8c6237bf` | Fuse the Qwen4 MoE residual tail | M=2 Qwen4 verifier; stage 3 remains a separate command buffer to preserve its cross-threadgroup reduction and BF16 rounding boundary | retained |
| `3c3212c3` | Fuse Qwen4 sparse-attention projections | Qwen4 sparse-attention path | retained |
| `6f9206f1` | Fuse Qwen4 M=2 hyper D/U | Fixed M=2 target verifier | retained |
| `c8e73cff` | Fuse Qwen4 M=2 attention hyper D/U | Fixed M=2 full-attention layers | retained |
| `d631025d` | Incrementally cache QSA pooled keys | Avoids rebuilding the full pooled history | retained |
| `1cf938ff` | Compact QSA attention rows | Exact compact Qwen4 QSA layout | retained |
| `a4260605` | Right-shape compact QSA gathers | Exact compact gather geometry | retained |
| `4cd8afa1` | Reuse Qwen4 MoE stage-1 activation tiles | Reuses each activation tile across four routed experts plus the shared gate | retained |
| `8ec9a0fc` | Compile the fixed-M2 verifier | Qwen4 depth-one M=2 capture route | retained |
| `a5e9872d` | Batch both depth-one target-distribution rows | One target-distribution materialization boundary per M=2 window | retained |
| `73597066` | Pin compiled verification in the benchmark contract | Fresh-shell repeats: 69.8365 and 69.6411 TPS; +11.89% over eager verification | retained |

Alternate M=3 work (`ca018d05`) remains useful as a geometry reference, but it
is not part of the depth-one physical-M=2 production contract. The QSA rotary
graph experiment (`21ef81a4`) was explicitly reverted by `ac27542c` and must
not be treated as retained work.

## Measured MTPLX candidates and rejections

| Candidate | Result | Decision |
|---|---|---|
| Existing cache-mutating compiled `draft_core=device` at depth 1 | Warmup failed with `Attempting to eval an array without a primitive` after poisoning the compiled MTP cache; its generic fallback would violate the enabled-lane contract | rejected; do not retry without a new state-owning design |
| Parallel n-gram shard acquisition | Did not beat the unchanged production control | reverted |
| Larger monolithic stage-2/stage-3 MoE fusion | Stage 3 owns a routed/shared reduction and BF16 accumulator rounding; direct fusion needs a global barrier or recomputation and can delay concurrent dispatch | investigate only with an exact arithmetic design and isolated kernel evidence |
| Dedicated MLX/Metal stream | Upstream `mlx-serve` attempt crashed under thread-local stream behavior and was reverted | rejected; use earlier dependency-chained submission on the default stream |
| M=2 split-K verifier QMM | Upstream's own depth-one/T=1 table reports 42.0 ms deferred baseline versus 44.6 ms with its verify kernels, despite wins at larger M | isolated exact-shape benchmark only; do not install from topology alone |

## `mlx-serve` candidates

| Upstream commit | Mechanism | Qwen4 M=2 assessment | Status |
|---|---|---|---|
| `f2601ba` | Build and async-dispatch the next MTP draft at the end of the current round | Directly targets the verifier-to-next-draft gap, but Qwen4 QSA, pooled/indexer, PLE, and MTP-cache state must all represent the committed prefix | high-priority audit |
| `3d437f`, `eafcc0` | Keep draft sampling lazy and feed the device token directly into verify | Targets the draft-to-target gap; MTPLX's host-backed n-gram provider still needs the draft ID for exact row acquisition | high-priority, partial fit |
| `d032c1f`, `49610d7` | Device top-k/top-p/categorical sampling with row-axis-correct top-k | Fits temperature 1/top-k 20; preserve the target/draft RNG contract and exact p/q distributions | candidate building block |
| `1e32a74` | Batch stochastic acceptance, correction samples, and recurrent-state capture into one async submission | Strong fit for the 0.598/0.325/0.174-second decision gaps after Qwen4 state ownership is proven | high-priority |
| `1c5c58f`, `f9648b9` | Bundle cache materialization, lazy token, and next forward in one async submission | Scheduling principle applies; Qwen4 PLE and auxiliary cache semantics differ | audit |
| `441f718`, `49610d7` | Split-K affine quantized matmul for verifier M=2-7, including a wide-message route for N at least 100,000 | Physical M=2, K=2560, Q4/G32, and most N values fit; small N stays stock. It changes FP32 reduction order and upstream's aggregate T=1 result regressed, so only isolated per-shape evidence can promote it | highest-priority compute experiment, not yet retained |
| `2d54af8` | QSA fused mask/split-row kernels | Reported gates favor qL at least 16 or qL*gqa above the M=2 Qwen4 geometry | likely no fit at depth 1 |
| `a1abe17`, `02322db`, `b195610` | Qwen4 HC, GDN, and decode fusion | Compare field by field with MTPLX's retained hyper/GDN routes before considering any code | verifier audit |
| `b243069` | Grouped-expert NAX experiment | Upstream measured about a 9% regression and did not ship it; MTPLX already has a fixed Qwen4 whole-MoE lane | rejected upstream |
| `1e3d1c5` / revert `e1feb98` | Dedicated device stream | Reverted upstream after crashes | rejected |

## oMLX candidates

| Upstream commit | Mechanism | Qwen4 M=2 assessment | Status |
|---|---|---|---|
| `8a9b1972` (PR #2113) | Lightning MTP: one backbone verify, device stochastic decision, one batched host read, and immediate async next draft | Best control-flow reference; transplant the scheduling principle while preserving Qwen4's distinct target/draft sampler and full auxiliary state | high-priority |
| `7395ca52` | Device-side filtered p/q acceptance and residual sampling with `mx.where` | Direct fit for the small decision fusion; validate sparse top-k/top-p arithmetic and RNG law | high-priority |
| `1e99d353` | Run multiple decode scheduler steps per executor handoff | May remove server/executor handoff cost but does not by itself remove in-loop PLE or sampling boundaries | scheduler audit |
| `5a39ba3a` (PR #2238) | Concatenate routed gate/up projection and reduce launches | MTPLX Qwen4 whole-MoE stage 2 already fuses gate/up per routed slot | already represented |
| `7e6f4224` | Extend the packed gate/up route to Qwen4 | Relevant arithmetic reference, but MTPLX's fixed whole-MoE M=2 lane already contains the reduction | already represented |
| `293d697c` | Split verify attention | Qwen3.5-specific attention geometry; compare only after exact Qwen4 M=2 route audit | likely no direct fit |
| `ba635b30` | Lazy graph/runtime parameters to avoid retracing | Use only if trace evidence shows retracing; fixed-M2 shared compilation already exists | low priority |
| `18324a34` | Route tiny verify rows away from prefill kernels | Audit current M=2 construction route; do not add a per-call eligibility fallback | route audit |
| `71d538a` | Truncate to the accepted prefix | MTPLX capture-prefix commit already retains the accepted prefix with zero repair forwards in the production receipt | already represented |
| `0c5229c` | Qwen4 Lightning tiny-shape correctness fixes | Useful state/shape reference, not standalone throughput evidence | correctness reference |
| `a9de32ca` | Fused router | Reduction ordering differs and upstream uses fallback-style routing; only reconsider with construction-time binding and accepted tie policy | low priority |
| `4be471c5`, `6ae51ed3` | Disk-backed PLE rows | The implementation blocks on device-to-host IDs before mmap reads; MTPLX's overlapped primary-row prefetch is the better scheduling reference | do not transplant |
| `93accf79` | Remove a Qwen3.5 M=3-6 verify QMM and loop exact M=1 rows | Does not fit physical Qwen4 M=2 and would increase dispatches | no fit |
| `5820985f` | Async-evaluate mutable cache leaves | Useful graph-lifetime principle; apply only through Qwen4's fixed cache installation | principle already represented |

## MTPLX candidates awaiting an exact production gate

| Candidate | Intended reduction | Required gate |
|---|---|---|
| Construction-bound M=2 split-K Q4/G32 QMM | Replace stock M=2 quantized matmuls for the large Qwen4 projections; use a separate huge-N geometry for the LM head | Isolated ABBA on each actual `(K,N)` shape first, parity at the agreed ULP tolerance, then repeated exact 16K/1K TPS and utilization |
| Fuse top-10 route normalization into the row-owned MoE router | Remove the separate FP32 `block_softmax` launch after each routed layer without building a monolithic MoE kernel | Preserve precise-softmax semantics closely enough for end-to-end parity, measure launch savings, and verify no dispatch-overlap regression |
| Construction-bound device stochastic decision graph | Collapse p/q acceptance, residual correction, accepted-prefix selection, and the next lazy token submission | Preserve temperature-1 top-k/top-p probability and RNG laws; prove Qwen4 cache/QSA/GDN/PLE ownership at the committed prefix |
| Draft-dependent n-gram row pipeline | Hide synchronous row acquisition behind already-enqueued draft work | Prove exact row identity and storage layout; report row-hit/miss timing outside the measured hot path and exact production TPS |

## Current starvation attribution

The decode trace contains 600 depth-one draft cycles, 391 accepted drafts, and
213 rejection corrections. Its 2.3191 seconds of active-command-buffer gaps
are dominated by transitions whose counts align with those control events:

| Transition family | Count | Idle seconds | Interpretation |
|---|---:|---:|---|
| elementwise sampling tail to gather | 601 | 0.5985 | per-cycle draft distribution/token handoff before target verification |
| PLE dequantization to gather | 391 | 0.6162 | accepted-cycle target decision/next-work boundary |
| copy to gather | 219 | 0.3252 | rejected-cycle target decision/next-work boundary |
| elementwise tail to residual subtract | 213 | 0.1745 | rejection residual/correction path |
| gather to PLE dequantization | 1,205 | 0.1286 | host-backed n-gram row materialization boundary |

The profiler's `cap_wait` and `sched_backpressure` buckets are inclusive views
of the same pressure and must not be added to the 2.3191-second GPU-idle total.

## Next measured gates

1. Isolate the M=2 split-K QMM on every actual Qwen4 projection shape; discard
   it unless it beats stock MLX before any production wiring.
2. Prototype a construction-bound M=2 device p/q acceptance plus residual
   decision graph; no enabled hot-path eligibility checks or fallback.
3. Re-profile exact 16K/1K and retain only if repeated decode TPS improves and
   GPU idle/utilization does not regress.
4. Evaluate split-K verifier QMM only after proving which current Qwen4 M=2
   projections it replaces and preserving the target path's arithmetic contract.
