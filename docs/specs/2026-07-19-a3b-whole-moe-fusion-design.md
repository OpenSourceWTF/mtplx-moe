# A3B Whole-MoE Small-Row Fusion Design

**Date:** 2026-07-19  
**Status:** Approved architecture; local implementation in progress; benchmark not run  
**Branch:** `perf/a3b-174-moe-whole-fusion`  
**Base:** `8db96d65907eea9201f7dfcf2f42fe8c4c7b298a`  
**Benchmark control:** unchanged accepted stack at the same commit

## Objective

Replace the fragmented Qwen3.6-35B-A3B sparse-MoE path used by target M1,
target M2, and MTP M1 with one construction-installed route composed of the
minimum architecturally sound number of Metal dispatches. The route must reduce
launch and intermediate-materialization overhead without changing the declared
router, quantization, SwiGLU, BF16-rounding, routed-reduction, shared-gate, or
final-add arithmetic.

The accepted gate/up packing remains the storage owner and runs before exact
whole-MoE binding. The whole route binds the resulting 40 target packed blocks
and one packed MTP block to fixed three-stage entrypoints while retaining the
accepted packed owners unchanged.

The performance target for the larger A3B program remains 200 generated
tokens/s. This experiment is promoted only if an isolated natural K1/TGY4
C/X/C/X measurement beats candidate/control spread and lowers cycle time.

## Scope and non-goals

In scope:

- all 40 target MoE blocks and the one injected MTP MoE block;
- fixed small-row target M1 correction/draft and M2 K1 verification;
- fixed MTP M1 draft;
- precise router softmax, exact top-8 selection and normalization;
- target q8/group64 router and scalar-gate projections;
- target q4/group64 routed and shared projections;
- MTP dense router, q4/group32 routed projections, and dense shared paths;
- explicit packed-stock prefill routing;
- accepted packed gate/up ownership for all 40 target blocks and the MTP block;
- a load-time full-graph compatibility probe and a separate request-specific
  exact-geometry certificate;
- cold, text-only requests that retain the contiguous-dense cache layout from
  prefill through decode;
- compatibility with compiled target-prefix graphs and the accepted A3B stack.

Not in scope:

- replacing the prefill MoE path;
- fixed 1024/1024 benchmarks or TGY8;
- sampler or RNG changes;
- hot-path validation, eligibility checks, fallback, engagement counters, or
  replacement instrumentation;
- MTP adapters, session-bank restores, vision-spliced prompts, or repaging the
  prompt cache into a paged/owned layout;
- merging or promoting the experiment without explicit approval;
- changing the accepted-stack worktree.

## Evidence and current call graph

The dependency's stock sparse block computes router projection, precise
softmax, top-k, optional score normalization, routed experts, route-weighted
reduction, shared expert, shared scalar gate, and the final addition in that
order. See the installed `mlx_lm` source
`mlx_lm/models/qwen3_next.py:308-354`. The stock routed expert performs separate
gate, up, and down projections around SwiGLU; see
`mlx_lm/models/switch_layers.py:160-199`.

The accepted stack's packed projection classes combine gate and up storage but
still perform a packed gate/up projection, split, SwiGLU, and a separate down
projection; see `mtplx/moe_packed_projections.py:152-213`. The accepted
row-owned route replaces the top-k portion but still materializes the routed
and shared paths separately; see `mtplx/qwen_row_owned_router.py:366-404`.

Construction rejects incompatible adapter options first, injects MTP, and then
runs accepted gate/up packing before the exact whole-MoE plan is prepared. The
plan therefore validates and retains the 40 target plus one MTP packed storage
owners. Only when no whole-MoE plan exists is the subordinate row-owned-router
plan prepared and installed. Model self-checks, compiled-target factory
construction, runtime construction, and an atomic load-time full-graph probe
then precede the final 41-block ownership commitment; see `mtplx/runtime.py`.

The expected source-level device boundaries per accepted-stack MoE block are:

| # | Boundary | Result |
|---:|---|---|
| 1 | router q8 QMM or dense matmul | BF16 `[M, 256]` logits |
| 2 | precise softmax | BF16 `[M, 256]` probabilities |
| 3 | row-owned top-8 and normalization | indices and BF16 `[M, 8]` scores |
| 4 | packed routed gate/up gather-QMM | BF16 `[M, 8, 1024]` |
| 5 | routed SwiGLU | BF16 `[M, 8, 512]` |
| 6 | routed down gather-QMM | BF16 `[M, 8, 2048]` |
| 7 | score multiply and routed reduction | BF16 `[M, 2048]` |
| 8 | packed shared gate/up QMM | BF16 `[M, 1024]` |
| 9 | shared SwiGLU | BF16 `[M, 512]` |
| 10 | shared down QMM | BF16 `[M, 2048]` |
| 11 | shared scalar-gate projection and sigmoid | BF16 `[M, 1]` |
| 12 | shared scaling and routed/shared addition | BF16 `[M, 2048]` |

This is a source-level boundary count, not a claim about undocumented internal
dispatch splitting in MLX. Any launch-count confirmation must come from an
out-of-band trace and must not add work to the measured path.

## Proven checkpoint contract

The model configuration declares default q4/group64 affine target weights,
q8/group64 target router and shared scalar gates, and a separately prequantized
q4/group32 MTP expert policy while keeping plain MTP tensors BF16. See the
loaded model's `config.json:2339-2349` and `config.json:2357-2375`.

All execution inputs and outputs are BF16. The shared topology is hidden size
2048, 256 routed experts, top-k 8 with normalized scores, routed intermediate
512, shared intermediate 512, 40 target MoE blocks, and one MTP block.

### Storage and active work

`U32` weights contain packed affine quantized values. Metadata includes both
BF16 scales and BF16 biases.

| Path | Projection | Physical storage | Active bytes, M1 | QMM FLOPs, M1 |
|---|---|---|---:|---:|
| target | router | q8/g64: U32 `[256,512]`, metadata `[256,32]` | 557,056 | 1,048,576 |
| target | routed gate+up, top 8 | each q4/g64: U32 `[256,512,256]`, metadata `[256,512,32]`; packed GU `[256,1024,256]` | 9,437,184 | 33,554,432 |
| target | routed down, top 8 | q4/g64: U32 `[256,2048,64]`, metadata `[256,2048,8]` | 4,718,592 | 16,777,216 |
| target | shared gate+up | q4/g64: each U32 `[512,256]`, metadata `[512,32]`; packed GU `[1024,256]` | 1,179,648 | 4,194,304 |
| target | shared down | q4/g64: U32 `[2048,64]`, metadata `[2048,8]` | 589,824 | 2,097,152 |
| target | shared scalar gate | q8/g64: U32 `[1,512]`, metadata `[1,32]` | 2,176 | 4,096 |
| **target total** | | | **16,484,480 (15.721 MiB)** | **57,675,776** |
| MTP | router | dense BF16 `[256,2048]` | 1,048,576 | 1,048,576 |
| MTP | routed gate+up, top 8 | each q4/g32: U32 `[256,512,256]`, metadata `[256,512,64]`; packed GU `[256,1024,256]` | 10,485,760 | 33,554,432 |
| MTP | routed down, top 8 | q4/g32: U32 `[256,2048,64]`, metadata `[256,2048,16]` | 5,242,880 | 16,777,216 |
| MTP | shared gate+up | dense BF16, each `[512,2048]`; packed `[1024,2048]` | 4,194,304 | 4,194,304 |
| MTP | shared down | dense BF16 `[2048,512]` | 2,097,152 | 2,097,152 |
| MTP | shared scalar gate | dense BF16 `[1,2048]` | 4,096 | 4,096 |
| **MTP total** | | | **23,072,768 (22.004 MiB)** | **57,675,776** |

Target M2 doubles QMM FLOPs to 115,351,552 per block. Storage traffic depends
on selected-expert overlap: 15.721 MiB if both rows select the same eight
experts and 29.221 MiB if their selected sets are disjoint. The implementation
may exploit natural cache reuse but may not assume expert overlap.

### Logical and physical rows

| Consumer | Logical M | Input | Router result | Selected activation | Output |
|---|---:|---|---|---|---|
| target correction/draft | 1 | BF16 `[1,1,2048]` | `[1,1,256]` | `[1,1,9,512]` | `[1,1,2048]` |
| target K1 verification | 2 | BF16 `[1,2,2048]` | `[1,2,256]` | `[1,2,9,512]` | `[1,2,2048]` |
| MTP draft | 1 | BF16 `[1,1,2048]` | `[1,1,256]` | `[1,1,9,512]` | `[1,1,2048]` |

The nine activation slots are the eight selected routed experts in current
top-k order plus one shared expert. Prefill is not represented in this table
because it is deliberately bound to stock at construction.

## Rejected partitions

### A. One-dispatch monolith

One row-owning threadgroup would perform router, top-k, nine expert paths, and
the final 2048-wide reduction. It provides only one M1 or two M2 threadgroups.
On a 40-core GPU that exposes at most 2.5% or 5% of one-threadgroup-per-core
occupancy before resource limits. It also serializes 57.68 or 115.35 MFLOPs and
must retain or repeatedly reconstruct selected activations and output
accumulators. The minimal launch count therefore destroys the available
parallelism and is rejected without implementation.

### B. Two-stage route/expert design

Stage 1 could produce indices, scores, and the scalar shared gate, while Stage
2 computes every expert and final output. A parallel output-owned Stage 2 must
recompute the gate/up projections for every output tile because threadgroups
cannot share the 512-wide activation. With 16 output columns per threadgroup it
repeats gate/up work 128 times: approximately 1.27 GiB and 4.83 GFLOPs of
gate/up traffic/work per target block. Even a 128-column tile repeats it 16
times, approximately 162 MiB and 604 MFLOPs. Making Stage 2 row-owned avoids
recomputation only by recreating the occupancy failure of the monolith. This
partition is rejected.

### Partition resource comparison

The estimates below use target M1 unless stated otherwise. They are design
screens to choose an architecture, not direct-call performance verdicts.

| Partition | Weight bytes / QMM FLOPs | Threadgroups | Threadgroup memory | Register-pressure estimate | Occupancy consequence |
|---|---|---:|---:|---|---|
| one-dispatch monolith | 15.721 MiB / 57.676 MFLOPs per M1 row; twice the FLOPs at M2 | 1 M1, 2 M2 | about 20 KiB if M1 router state, nine activations, and output accumulators are retained; about 40 KiB at M2 | at least 8 M1 or 16 M2 distributed output accumulators per lane, plus current expert dots and selection state | hard-capped at 2.5%/5% of 40 one-threadgroup-per-core slots before resource limits |
| two-stage, 16-column output-owned expert stage | about 1.27 GiB and 4.83 GFLOPs from repeating the 10.125 MiB / 37.749 MFLOP gate/up stage 128 times, plus down work | 1/2 route TGs, then 128 output TGs | approximately zero with SIMD-local reductions | roughly 8 M1 or 16 M2 output/current-dot accumulators per lane | enough TG count, but made overwhelmingly redundant-weight/compute bound |
| two-stage, 128-column output-owned expert stage | about 162 MiB and 604 MFLOPs from 16 gate/up repeats, plus down work | 1/2 route TGs, then 16 output TGs | approximately zero if activations are recomputed | at least 64 M1 or 128 M2 output accumulators per lane-equivalent tile before current dots | marginal TG count and severe spill/occupancy risk |
| approved Stage 1 | target router plus scalar gate: 0.533 MiB / 1.053 MFLOPs | 1 M1, 2 M2 | under 3 KiB per row | distributed logits plus top-8 state, targeted below 16 scalar values per lane | intentionally low occupancy for the small global dependency |
| approved Stage 2 | selected routed plus shared gate/up: 10.125 MiB / 37.749 MFLOPs per row | 288 at M1 and row-paired M2 | zero | 8 M1, at most 16 M2 gate/up dot accumulators per lane | ample independent expert/column tiles |
| approved Stage 3 | selected routed plus shared down: 5.063 MiB / 18.874 MFLOPs per row | 128 at M1 and row-paired M2 | zero | targeted at 8 M1, at most 16 M2 current-dot plus routed accumulators per lane | ample independent output-column tiles |

## Approved three-stage partition

The installed route owns the complete MoE operation, but uses three fixed
dispatch stages because router selection is a global row dependency and the
expert down projection needs output-column parallelism.

### Stage 1: route and shared scalar gate

One row-owned 256-thread threadgroup per logical row performs:

1. target q8/group64 or MTP dense router projection;
2. the declared BF16 router-output rounding;
3. precise FP32 softmax internals;
4. exact current top-8 ordering and tie behavior;
5. BF16 score extraction and current sequential BF16 normalization;
6. target q8/group64 or MTP dense shared scalar-gate projection.

Outputs are compact indices `[M,8]`, BF16 normalized scores `[M,8]`, and the
BF16 raw shared-gate value `[M,1]`. The threadgroup needs about 1 KiB for 256
FP32 logits and less than 3 KiB total scratch after selection state. This stage
has deliberately low occupancy, but owns only about 0.53 MiB and 1.05 MFLOPs
for the target M1 router plus a 2048-element scalar projection. Splitting it
would introduce another global boundary without meaningful parallel work.

### Stage 2: selected gate/up and SwiGLU

Fixed storage-specific entrypoints compute the eight selected routed gate/up
projections and the one shared gate/up projection, then apply the exact SwiGLU
operation and store only BF16 `[M,9,512]`.

The starting tile is four output columns per SIMDgroup and four SIMDgroups per
128-thread threadgroup, or 16 activation columns per threadgroup. Thus each
expert slot uses 32 threadgroups and each row route exposes 288 threadgroups.
M2 is row-paired so it keeps the same 288 tiles while doubling row work; its
gate/up accumulators are capped at 16 live scalar accumulators per lane. No
threadgroup memory or barrier is required; reduction is SIMD-local.

There are distinct fixed entrypoints/descriptors for:

- target q4/group64 routed plus q4/group64 shared projections at M1;
- target q4/group64 routed plus q4/group64 shared projections at M2;
- MTP q4/group32 routed plus dense BF16 shared projections at M1.

### Stage 3: down projections, reductions, and final add

Output-column-owned threadgroups read `[M,9,512]`, compute all eight routed down
projections and the shared down projection, apply normalized route scores,
apply the sigmoid shared scalar gate, and store the final BF16 `[M,2048]`
directly.

The starting tile is again four output columns per SIMDgroup and four
SIMDgroups per 128-thread threadgroup: 16 output columns per threadgroup and
128 threadgroups for the hidden dimension. M2 is row-paired. Experts are
processed sequentially inside a tile so only the current down-dot and routed
accumulators remain live; the M2 design target is at most 16 live scalar
accumulators per lane. No full routed or shared output crosses a dispatch
boundary.

There are distinct fixed target q4/group64 M1, target q4/group64 M2, and MTP
q4/group32-routed/dense-shared M1 entrypoints.

### Eliminated boundaries and intermediates

The source-level route contracts from 12 expected device boundaries to 3,
eliminating nine boundaries per block. Only compact `[M,9,512]` activations
cross from Stage 2 to Stage 3: 9 KiB at M1 and 18 KiB at M2.

The fused route does not materialize:

- routed BF16 `[M,8,2048]` outputs (32 KiB M1, 64 KiB M2);
- a separate shared BF16 `[M,2048]` output (4 KiB M1, 8 KiB M2);
- a standalone combined routed BF16 `[M,2048]` output (4 KiB M1, 8 KiB M2);
- separate routed/shared elementwise launch outputs after Stage 3.

The accepted packed gate/up `[M,8,1024]` and `[M,1024]` intermediates also do
not cross a dispatch boundary because Stage 2 consumes the gate/up values
before its BF16 activation store.

## Synchronization and fit-for-purpose geometry

Top-k requires all 256 logits for a row. Metal threadgroups have no global
barrier, so a multi-threadgroup fused router requires another dispatch or an
unsafe cross-threadgroup protocol. Stage 1 therefore owns one row. Stages 2 and
3 start only after route selection and recover occupancy from expert/activation
and output-column tiling.

This geometry is derived from exact A3B M1/M2 shapes:

- the row count is too small to supply occupancy by row tiling;
- the activation width 512 yields 32 independent 16-column tiles per expert;
- nine expert slots yield 288 Stage-2 tiles;
- hidden width 2048 yields 128 independent Stage-3 tiles;
- M2 row pairing reuses selected weight cache lines without assuming equal
  expert selections and avoids multiplying launch geometry by row count;
- four columns per SIMDgroup bounds M2 live accumulators before Metal compiler
  inspection.

The 16-column choice is a starting geometry, not a topology transplant. Exact
Metal compilation, register allocation, spills, and occupancy must be checked
before performance benchmarking. A geometry that spills or reduces active
threadgroups materially is rejected or revised from these resource facts, not
through an unbounded tuning sweep.

## Arithmetic contract

Target M1, target M2, and MTP M1 use the same explicit arithmetic contract:

1. Dequantize affine q4/q8 weights using their exact packed orientation,
   group size, scale, and bias metadata; dense MTP tensors use BF16 directly.
2. Accumulate each projection in the same order declared by its fixed
   entrypoint and round its public result to BF16 at the current projection
   boundary.
3. Router softmax uses precise FP32 internal arithmetic and emits the current
   BF16 probabilities.
4. Top-8 ordering, ties, and selected index order match the accepted row-owned
   implementation.
5. Score normalization follows the current sequential BF16 denominator and
   division behavior.
6. Stage 2 performs the current `silu(gate) * up` order and stores BF16
   activations.
7. Stage 3 rounds each routed down projection to BF16, performs the BF16
   route-score product, and accumulates routed experts sequentially in the
   current top-8 order with the current BF16 boundaries.
8. The shared down result is rounded to BF16. The shared scalar gate follows
   the current BF16 projection/sigmoid/multiply boundaries.
9. Routed and shared results are added with the current BF16 final-add
   boundary.

No sampler or RNG compensation is permitted. If an intentional reassociation
is later proposed, it is a distinct experiment and must be applied consistently
to verifier, correction, and MTP graphs with a documented rounding boundary.

## Correct-by-construction installation

The environment flag is read once during runtime construction. Flag off makes
no model mutation and leaves all 40 target blocks and the MTP block on the
accepted stock/installed route.

Flag on performs one prepare/self-check/probe/install transaction after model
load, MTP injection, checkpoint-coverage validation, and accepted gate/up
packing, and before any request can generate:

1. reject `mtp_adapter` and `merge_mtp_adapter` load options before they can
   mutate storage that the experimental route would bind;
2. require the exact packed owners for all 40 target blocks and the one MTP
   block, including the gate-before-up split and packed projection classes;
3. validate exact Qwen3.6-35B-A3B model type, 40+1 topology,
   hidden/expert/top-k/intermediate/normalization facts, and BF16 boundaries;
4. validate target q8/group64 and q4/group64 packed storage and orientation;
5. validate MTP q4/group32 packed routed storage plus its dense BF16
   router/shared storage;
6. reject sharding, native-tail, or incompatible installed-route conflicts;
7. run per-component route-score, shared-gate, activation, and final-output
   self-checks across every one of the 40 target bindings at M1 and M2, plus
   the MTP binding at M1;
8. build the exact compiled target-prefix factory and construct the runtime;
9. prepare all 41 route classes with construction-prebound target M1/M2 or MTP
   M1 Stage 1/2/3 kernels over direct model-owned packed arrays, then
   tentatively swap them as one transaction;
10. execute direct compiled M2 and M1 full-graph calls over disposable state;
    retain all 41 swaps only when both load-time compatibility lanes pass, and
    restore every original class on any probe or mutation failure.

The accepted packing step always precedes whole-MoE binding and remains the
storage owner. The row-owned router alone is construction-superseded, and only
when a valid whole-MoE plan exists; otherwise the normal row-router plan and
install path remain unchanged.

Each installed descriptor closes over fixed model-owned arrays and prebound
target-M1, target-M2, or MTP-M1 kernel callables plus the packed-stock prefill
method. Execution may branch only on genuine dynamic routing facts: attention
phase and logical M. Prefill calls the explicitly captured packed-stock route.
Target M1/M2 and MTP M1 call the prebound three-stage route directly.

The installed small-row path contains no environment read, dtype/shape/bits/
group/topology/storage validation, marker/status/self-check lookup, eligibility
predicate, exception fallback, stock fallback, lane-disabled check, fallback
accounting, or engagement counter. It also never tries one kernel and falls
back to another. Invalid external facts or failed self-checks prevent
installation before generation. Intermediate tensors produced by the installed
route are not revalidated.

## Errors and atomicity

Preparation returns an immutable installation plan but does not mutate blocks.
Any mismatch raises one configuration error naming the external invariant and
the observed checkpoint/model fact. Self-check failure raises before class
mutation. The load-time full-graph probe temporarily uses the prepared classes
inside the installation transaction; any compile, execution, output-ownership,
or mutation failure restores all prior classes. Installation is all-or-nothing
across 40 target blocks plus MTP; partial installation is forbidden.

The load probe uses a one-token synthetic prefill with `prompt_tokens=1` and a
small reserve only to prove that the installed 41-block graph is compatible
with direct compiled M2 and M1 execution. It does **not** prove that MLX has
naturally specialized the later benchmark request's prompt length, cache
capacity, hidden variant, or ordered state-leaf topology.

## Exact request geometry and certificate

Every installed whole-MoE request is validated once before prompt work. It must
be exact K1 `target_prefix` with stock capture/draft arithmetic, the exact
compiled-target factory, no session bank, no vision splice, and
`contiguous_dense_decode`. Session-bank restores, vision requests, and
`contiguous_then_repage` or any other repaged/owned cache layout are rejected
for this experimental lane rather than routed through a fallback.

Before the real prompt prefill, a request-specific synthetic preflight:

1. invokes the same contiguous-dense target-cache factory used by real prefill;
2. evaluates one synthetic token, then assigns the exact prompt offset to each
   full-attention cache position without replaying or mirror-committing the
   prompt;
3. constructs the fixed target-prefix state with `max_tokens + 2` reserve and
   verifies the resulting physical capacity for
   `prompt_tokens + max_tokens + 2`, rounded only by the cache's 256-token
   allocation step;
4. invokes the compiled M2 function and then the compiled M1 function directly,
   with no `verify_m2`/`repair_m1` wrapper or mirror-commit path;
5. evaluates every output, checks the fixed output counts and hidden/cache
   ownership, and records the ordered input, state, primary, and final leaf
   shape/dtype signatures;
6. hashes the hidden variant, fixed ordered state specification, fixed M2/M1
   input signatures, and canonical ordered M2 state-leaf shape/dtype sequence
   into the request certificate key.

The certificate is memoized only for the exact logical capacity, hidden
variant, and contiguous-dense layout. After the real prompt prefill constructs
the actual target-prefix route, that route independently computes its canonical
ordered leaf shape/dtype hash and must find the same certificate before the
first decode iteration. A missing or different key is a configuration failure,
not a compile-on-demand success or a stock fallback.

The benchmark gate proves engagement from construction status and existing
request statistics. For K1:

- target M2 work equals `verify_calls`;
- target M1 work equals `compiled_calls - verify_calls`, equivalently drafted
  minus accepted corrections under the validated target-prefix contract;
- MTP M1 follows existing draft statistics;
- compiled-verifier fallback and growth demotion must both be zero.

No new hot counter is added for these equations.

## Test design

TDD begins with failing tests for:

1. flag-off identity for all 40 target blocks and MTP;
2. load-option rejection for MTP adapters before packing or binding;
3. accepted packing before exact binding and exact packed ownership for all
   40 target blocks plus MTP;
4. exact contract installing 41 prebound routes;
5. construction failure for invalid topology, packed storage, dtype,
   quantization, layer/expert/top-k/normalization, or compilation facts;
6. component-specific Stage 1/2/3 checks across all 40 target M1 and all 40
   target M2 bindings, plus MTP M1;
7. self-check failure preventing mutation and full-graph-probe failure restoring
   every tentatively swapped class;
8. explicit packed-stock prefill route and row-router installation only when no
   whole-MoE plan exists;
9. no environment/status/eligibility/validation/fallback/stock branch inside
   installed M1/M2 execution;
10. no runtime validators in fixed Metal entrypoints;
11. exact Stage-1 logits/top-k/scores parity;
12. exact Stage-2 routed/shared activation and BF16-boundary parity;
13. exact Stage-3 down/reduction/shared/final-add parity;
14. shared arithmetic across target correction, target verification, and MTP;
15. direct compiled M2/M1 load compatibility without claiming natural-request
    specialization;
16. request rejection for session-bank, vision, and repaged layouts;
17. exact request preflight from the contiguous-dense cache factory, including
    prompt offset, `max_tokens + 2` reserve, canonical ordered leaf shape/dtype
    hash, and actual-route certificate match before decode;
18. absence of routed `[M,8,2048]` and separate shared `[M,2048]`
    materialization;
19. existing compiled-verifier reserve and zero-fallback/growth behavior;
20. engagement derived only from existing request statistics.

Verification covers focused fused-MoE, row-owned-router, packed-projection,
row-major QMM, kernel-self-check, compiled-target-prefix,
compiled-verifier, generation, and complete A3B harness-integrity tests, plus
Ruff and `git diff --check`.

## Performance gate and predicted value

The accepted-stack reference recorded in the workspace
`bench/a3b/OPTIMIZATION_LEDGER.md` is approximately 158.827 tok/s at 10.63065
ms/cycle and 1.68828 accepted tokens/cycle. Holding accepted tokens/cycle
constant, 200 tok/s requires about 8.4414 ms/cycle: a 2.1893 ms/cycle or 20.59%
reduction.

Contracting 12 expected boundaries to 3 removes nine launches per MoE block.
Across 40 target blocks, one MTP block, and the observed correction frequency,
the working estimate is about 435 eliminated launches per K1 cycle. At an
out-of-band 3-5 microseconds per eliminated launch, the hypothesis predicts
about 1.3-2.2 ms/cycle. That range reaches most or all of the remaining gap, but
is not a result. The experiment is worth a natural benchmark only if tests,
Metal compilation, and resource inspection show no correctness failure or
occupancy/register disaster. A measured saving below roughly 0.4 ms/cycle is
unlikely to clear ordinary candidate/control spread or justify the complexity.

After all correctness gates, benchmark only:

- upstream `long_code_uncapped`, natural stop;
- K1 and TGY4;
- process-isolated C/X/C/X;
- repeat-2 verdict cells;
- exclusive GPU lane;
- unchanged accepted-stack control;
- exact installation/self-check gates;
- a 64-token unreported warmup followed by the exact full token-budget request
  preflight before configuration emission or any timed repeat;
- a measured post-prefill route whose certificate key matches that exact
  request preflight;
- zero compiled-verifier fallback and zero growth demotion.

If K1 is positive beyond spread and lowers cycle time, extend the same proven
route through exact M1-M6 geometries for K0-K5, then run the complete natural
K0-K5 matrix. Prefill at 128, 512, 2048, 8192, and 16384 prompt tokens remains
the explicit packed-stock route and must retain contiguous-dense decode for
this experimental lane. Record the independent commit, flags, commands, hashes,
repeat-2 results, cycle metrics, parity disclosure, installation proof, and
verdict in `bench/a3b/OPTIMIZATION_LEDGER.md`.

Do not add a wash or regression to the beneficial stack, and do not merge or
promote without explicit approval.

## Failure criteria

Stop or revise this partition before natural benchmarking if any of the
following occurs:

- Stage 1 cannot reproduce exact precise-softmax/top-k behavior;
- Stage 2 or Stage 3 spills materially at the fixed M2 row-paired geometry;
- MTP group32 routed or dense shared arithmetic cannot use explicit fixed
  entrypoints;
- compiled target-prefix capture rejects a fixed entrypoint;
- the implementation needs hot validation, fallback, or instrumentation to be
  safe;
- the claimed eliminated intermediate is still present in the constructed
  graph.

If the three-stage implementation is correct but loses, inspect the measured
reason and distinguish an implementation verdict from a concept verdict. Do
not silently substitute the rejected monolith or two-stage design, and keep any
materially different partition as its own commit and benchmark unit.
