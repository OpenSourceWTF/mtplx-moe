# Issue 51 Hy3-Q2 MTP look-ahead acceleration

## Status

Approved in conversation on 2026-07-14 with these constraints:

- target the existing Hy3 expert-only affine-Q2 artifact and its BF16 MTP
  layer;
- verify 1,024- and 2,048-token inputs with exactly 128 generated tokens;
- test D1 first and test D2 only if D1 improves both speed and utilization;
- treat K=0 AR as the one-row Next-K control and exclude K=3 or deeper;
- evaluate MTPLX's Qwen-style compiled whole-window verifier;
- evaluate predictive expert loading only after the verifier work has its own
  decision;
- evaluate NAX as a separate, measurement-gated Q2 candidate at the lowest
  priority of the three tests.

The work remains experimental and off by default. Each candidate must pass on
its own before combinations are measured.

## Premise

The performance problem exists. The corrected Hy3 v3 baseline is
correctness-qualified but speculative decoding is slower than AR:

| Context | AR decode tok/s | D1 | D2 |
|---:|---:|---:|---:|
| 1,024 | 6.063 | 5.826 | 5.664 |
| 2,048 | 4.941 | 4.047 | 3.653 |

`generate_mtpk` already implements MTP look-ahead: it recurrently produces
draft tokens and verifies `[primary, *drafts]` with one batched target forward.
The existing Q2 benchmark used the default `verify_strategy="batched"`; it did
not test Qwen's compiled capture/commit verifier.

NAX is not synonymous with look-ahead. In Qwen Turbo it is one optional inner
operator beneath the same batched verifier. The existing NAX patch accelerates
4/6/8-bit `nn.QuantizedLinear` calls and therefore does not touch Hy3's BF16
trunk or its Q2 `mx.gather_qmm` streamed-expert path. A Q2 NAX candidate is new
kernel work and must earn its existence from the actual Q2 profile.

The proportional order is therefore:

1. reuse and test the existing compiled whole-window verifier;
2. run predictive loading offline, then pilot it only if its own gate passes;
3. profile the Q2 expert operator and G1/G2/G3 route-reuse geometry;
4. prototype a Q2 NAX operator only if the profile shows addressable headroom.

Q2 NAX and multi-row grouping must not delay either higher-priority test.
K=0 is its single-row control; K=1 and K=2 measure the additional value of
multi-row reuse.

Not doing the work leaves D1 3.9% below AR at 1,024 and 18.1% below AR at
2,048. Building a new Q2 Metal kernel without a positive operator premise would
repeat a closed Q4 experiment whose stock `gather_qmm` reached 95.6% of the
single-dispatch memory roofline.

## Isolation and base

Use:

- worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/51-q2-mtp-lookahead`;
- branch: `experiment/issue51-q2-mtp-lookahead`;
- starting commit: `bc49e27`.

Before implementation, integrate the committed #49/#50 v3 harness and greedy
correction fix. The current dirty `codex/q2-bf16-mtp-bench` worktree is not a
source from which to copy uncommitted files. Record the integrated base commit
in every artifact.

Keep raw traces under ignored `benchmarks/raw/`. Track only compact curated
tables, compatibility maps, and go/no-go decisions.

## Fixed model and benchmark contract

The target is `hy3-expert-q2`:

- routed experts: affine Q2, group size 64, BF16 scales and biases;
- target trunk, router, attention, shared experts, and LM head: the existing
  resident precision/layout, with no re-quantization for this issue;
- MTP layer 80: resident BF16;
- expert slot layout: component banks;
- sampler: greedy, temperature 0, top-k 1, top-p 1, seed 0;
- MTP cache: persistent;
- MTP history: committed;
- repetition stop and loop guard: disabled;
- output length: exactly 128 tokens with an empty stop-token set.

The gated sequence is:

- contexts: 1,024 and 2,048 input tokens;
- K=0 AR control: one target row;
- D1/K=1: two-row target verification, measured first;
- D2/K=2: three-row target verification, measured only after D1 has positive
  paired speed and active-reader intervals at both contexts;
- K=3 and deeper: non-goal because the streamed Q2 expert path is already
  bandwidth-sensitive.

Headline speed rows run without resource telemetry. A separate diagnostic lane
measures decode mean-active readers out of the 32-worker pool from the
prefill-complete boundary to generation completion. Instrumented TPS is not a
headline value.

All arms use identical prompt IDs, artifacts, runtime capacity, expert-cache
policy, slot state, sampler, and generated-token limit.

## Subproject A1: compiled whole-window verification

### Objective

Transfer Qwen's compiled capture/commit execution structure to Hy3 without
changing MTP proposals or authoritative target decisions.

### Candidate ladder

Measure one mechanism at a time:

1. `batched-stock`: `verify_strategy="batched"`, compiled verify off.
2. `capture-eager`: `verify_strategy="capture_commit"`, compiled verify off.
3. `capture-compiled-parity`: capture/commit with `CompiledVerifyBank` in
   parity mode; this is correctness evidence, not a performance row.
4. `capture-compiled`: compiled capture/commit authoritative after parity
   passes.

Use `verify_core="stock"`. Hy3 has no Qwen GDN/conv tape, so
`linear-gdn-from-conv-tape` is not an eligible transfer.

### Compiled boundary

The compiled function may contain only pure MLX computation and explicit cache
state leaves. Authoritative router decisions, expert-cache admission, slot
generation changes, SSD I/O, pins, evictions, counters, and completion handling
must retain their normal semantics.

First attempt the existing `CompiledVerifyBank` unchanged in parity mode. If
tracing streamed expert Python side effects freezes routes, binds stale slot
generations, or falls back on every call, do not weaken the gates. Narrow the
compiled region around pure resident computation or reject A1 if that boundary
cannot be isolated without excessive complexity.

### A1 correctness gate

For every authorized D1 or D2 cell:

- every speculative event reconstructs the emitted tokens;
- accepted drafts, corrections, bonuses, verify calls, and per-depth counters
  match recomputation;
- target cache offsets end at prompt plus output;
- committed MTP history ends at prompt plus output minus one;
- router expert IDs and order match the stock same-shape row;
- expert request identities and integrity counters match;
- output tokens and finish reason match stock;
- same-shape repeated execution is deterministic;
- compiled cache leaves, logits, hidden rows, and captures satisfy the parity
  contract;
- no fallback, stale generation, reserve, pin, or final-slot-health violation
  is hidden from the artifact.

### A1 performance gate

The complete verifier block must have a positive paired speed interval, and
sustained decode TPS must improve without material regression in prefill,
memory, p95, expert-cache hit rate, or I/O. Reduced dispatch count or compiled
micro-time alone is not sufficient.

D1 additionally requires a positive paired decode active-reader interval at
both contexts before D2 is authorized. D2 is the maximum tested depth.

## Subproject A2: Q2 NAX evaluation (execution priority 3)

Despite the historical A2 label, execute this only after Subproject B has its
own offline/runtime decision. Q2 NAX and grouping are the lowest-priority test.

### Compatibility map

The existing NAX `QuantizedLinear` patch is not the candidate:

- the Hy3 trunk is not a 4/6/8-bit `nn.QuantizedLinear` trunk;
- streamed Q2 experts execute assignment-aligned `mx.gather_qmm` with bits 2;
- the current NAX kernels assume one shared RHS matrix across their row tile,
  while a gathered expert assignment may select a different RHS per row.

D1/D2 nevertheless expand 2/3 target rows through top-k 8 routing into 16/24
expert assignments per routed layer. Repeated expert IDs across speculative
positions may permit grouping by expert and reusing one Q2 matrix across a
small row batch. That is the only initial NAX hypothesis.

### Measure-first premise gate

On retained D1/D2 route traces, report per layer:

- assignment count and unique expert count;
- rows per repeated expert;
- repeated-expert fraction;
- component-bank grouping and scatter/reassembly cost;
- stock Q2 `gather_qmm` time for gate, up, and down;
- zero-arithmetic Q2 stream floor and effective bandwidth;
- host graph-construction and synchronization cost;
- projected end-to-end decode value if the entire operator gap vanished.

Compare stock gathered Q2 with direct per-expert Q2 matmul for the observed
group sizes. Do not write a production kernel unless the lower confidence
bound on addressable operator time is positive and the full measured operator
gap projects to at least a 5% decode-TPS gain after host dispatch, grouping,
and model-wide dilution. A smaller bound is retained as a no-go result.

### Prototype, only after the premise gate

The minimal prototype may unpack affine Q2 group-64 weights into an MPP/NAX
tile and evaluate rows grouped by one expert. It must not alter the artifact or
slot layout. It must be off by default and fail closed unless all eligibility
conditions match:

- Apple G17-class NAX hardware and supported macOS;
- bits 2, affine mode, group size 64;
- BF16 activation/scales/biases;
- exact Hy3 gate/up/down dimensions;
- component-bank slot generation is pinned for the full lazy graph;
- grouped rows and scatter positions are complete and non-overlapping.

Ineligible shapes, singleton groups that do not benefit, route misses, and any
unsupported device use unchanged stock `gather_qmm`.

### A2 correctness and performance gate

Compare the Q2 NAX candidate against stock batched verification with compiled
verification off. Require:

- bounded operator error and deterministic same-shape output;
- identical router choices at every downstream layer;
- identical verifier acceptance/correction decisions and output tokens;
- identical expert identities, slot generations, cache counters, and final
  health;
- a positive operator interval after grouping/scatter/host cost;
- a positive paired end-to-end decode-TPS interval with no material prefill,
  memory, p95, or cache-hit regression.

Reject the candidate if it only wins a synthetic shared-RHS tile, reduces
arithmetic while remaining at the same memory floor, or loses after the real
gather/group/scatter path is charged.

## Subproject A3: combination

Do not combine compiled verification and Q2 NAX until A1 and A2 independently
pass. If both pass, measure `capture-compiled + q2-nax` against each winning
individual arm, not only against stock. Retain the combination only if it adds
a positive paired interval and preserves every individual correctness gate.

## Subproject B: hint-only predictive expert loading (execution priority 2)

### Offline gate

With stock verification, derive next-layer expert candidates from retained MTP
draft hidden states for D1 and D2. Evaluate N=1,2,4,8,12 separately at both
contexts. Compare against:

- zero hint;
- prior-token route;
- a simple per-layer transition baseline;
- an oracle upper bound.

Report miss-specific recall, candidate union size, useful lead time,
ready-before-demand rate, late/cancelled/unused work, physical bytes, byte
amplification, and projected expert wait removed. Replay the real cache
capacity and preserve the authoritative reserve of at least 8 operations and
84,934,656 bytes.

Stop before runtime code unless the lower confidence bound on useful wait saved
remains positive after predictor compute, byte cost, and interference.

### Runtime pilot, only after the offline gate

Predicted bytes remain in a separate bounded staging area. The trunk router is
authoritative. Hints may not publish into, evict, pin, delay, or consume the
reserve of the authoritative cache. Demand may atomically adopt an exact ready
match `(layer, expert, component, generation)` or ignore it.

Measure zero-hint, oracle-hint, predictor-hint, and demand-only controls. Require
exact router/token/integrity results, zero reserve violations, bounded byte
amplification, lower measured expert wait, and a positive paired decode-TPS
interval.

## Benchmark method

### Qualification

For each candidate, run one correctness-qualified observation for every
context/depth cell after a discarded depth-specific warmup. Parity modes and
instrumented traces are qualification evidence and are excluded from headline
TPS.

### Paired performance

After qualification, collect at least eight retained pairs per
context/depth/candidate comparison in a balanced ABBA schedule. Each process
has one fixed candidate configuration; implementations change only between
complete runs, never during an in-flight Metal dispatch.

Record:

- ingestion TPS and target-prefill seconds/TPS;
- decode TPS and end-to-end TPS;
- draft, verify-forward, verify-eval, repair, commit, and rollback time;
- verify calls, accepted yield, and acceptance by depth;
- expert-cache hit rate, bytes, reads, evictions, waits, and reserve pressure;
- candidate dispatch/fallback counts;
- peak process and Metal memory;
- every correctness and final-state gate.

Report each paired delta, mean, median, p95, 95% interval, and arm order. A
candidate is a measured win only when the interval is positive and the result
survives the complete end-to-end lane. Default promotion additionally requires
at least 5% repeated target-lane decode improvement; a smaller positive result
remains experimental evidence rather than default enablement.

## Hardware and service safety

Keep Qwen loaded during preparation. Before any MLX/Metal measurement:

1. wait for `/tmp/mtplx-gpu-exclusive` to be free;
2. capture the exact Qwen model/service state;
3. acquire the exclusive lane;
4. unload Qwen only for the measurement window;
5. run one fixed candidate at a time;
6. restore and verify the captured Qwen state;
7. release the lane even after failure.

Service lifecycle and lane ownership remain external orchestration, never
repository test behavior.

## Error handling

- Checkpoint every completed cell and retain full failure evidence.
- On parity, integrity, offset, reserve, or slot-health failure, stop that
  candidate immediately and keep later candidates disabled.
- On a kernel eligibility miss, record the reason and execute stock rather
  than partially engaging NAX.
- On compiled-verifier exceptions, record exact fallback reasons; repeated
  fallback is a no-go, not a performance result.
- On interruption, restore Qwen from the captured state before publishing or
  releasing the lane.

## Failure-mode review

1. **Critical: compiled tracing freezes streamed expert side effects or stale
   slot bindings.** Mitigation: parity first, explicit expert/slot evidence,
   and a pure-MLX compiled boundary only. Reject A1 if isolation is not safe.
2. **Critical: a Q2 NAX microbenchmark wins by assuming a shared RHS that the
   real gathered route does not have.** Mitigation: derive group sizes from
   retained routes and charge grouping, scatter, host dispatch, and singleton
   fallback in both micro and end-to-end gates.
3. **Critical: predictive reads interfere with authoritative demand.**
   Mitigation: offline cost gate, isolated staging, strict demand priority,
   immutable reserve, and zero-tolerance reserve telemetry.
4. **Minor: A1 and A2 wins do not compose.** This is acceptable; keep the
   independently passing arm and reject the combination.
5. **Minor: benefits appear only at one context.** Do not authorize D2 unless
   D1 improves both fixed contexts; retain the scoped D1 evidence as a no-go.

## Deliverables

- Qwen-to-Hy3 compiled-verifier compatibility map;
- D1 stock, capture-eager, and compiled speed/utilization results at
  1,024/2,048 input and 128 output tokens, followed by D2 only if authorized;
- Q2 NAX premise profile and, only if gated, prototype and end-to-end results;
- predictive-loading oracle and cheap-baseline comparison;
- runtime predictive-loading pilot only if the offline gate passes;
- raw artifact identities and compact curated tables;
- separate go/no-go decisions for A1, A2, A3, B-offline, and B-runtime;
- an evidence update on GitHub Issue #51.

No serving default changes, PR publication, or branch integration occur until
the corresponding candidate clears its full gate.
