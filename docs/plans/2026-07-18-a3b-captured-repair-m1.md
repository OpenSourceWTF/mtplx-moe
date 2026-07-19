# A3B Captured-Primary M1 Repair Implementation Plan

**Base:** `perf/a3b-174-beneficial-stack` at `960849b`

**Goal:** Replace the K1 target-prefix rejection path's redundant compiled M2
`[primary, correction]` re-forward with a fixed M1 `[correction]` continuation
from state already produced after `primary`.

The implementation follows one rule throughout: external model, checkpoint,
self-check, and request facts are validated at their owning boundary; every
object built from those facts is then trusted and called directly. The exact
path must not validate our factories, caches, shadows, state leaves, or route
selection after construction.

## Fixed architecture

Model loading constructs one immutable `A3BGDNPostconvFactory` after validating
the exact A3B model and running the combined M1/M2 parity self-check. The
factory owns two order-stable tuples of 30 prebound callables:

- M1: BF16 `[1,1,8192]` recurrence, FP32 `[1,32,128,128]` state, TGY4;
- M2: BF16 `[1,2,8192]` recurrence, FP32 `[1,32,128,128]` state, TGY4.

`A3BCompiledTargetPrefixFactory` owns that post-conv factory directly. The
runtime owns the compiled-target factory directly. Nothing is written onto or
rediscovered from the model or individual GDN modules.

The exact trace chain is separate from the generic capture stack:

```text
fixed M1/M2 compiled trace
  -> MTPLXRuntime._forward_ar_capture_a3b_postconv
  -> forward_with_a3b_gdn_postconv_capture
  -> _a3b_gdn_forward_with_fixed_postconv
  -> prebound fixed M1 or M2 Metal callable
```

It must never call `CompiledVerifyBank`, `forward_with_gdn_capture`,
`gdn_forward_with_capture`, backend resolution, eligibility predicates, lane
status, environment-selected arithmetic, or stock fallback.

Request construction checks only externally selected K1 target-prefix settings,
the stochastic top-k sampler required by this exact schedule, and the installed
context ceiling. It directly transforms the ten known
full-attention cache positions into tensor-offset caches with request-aware
capacity and constructs the exact 40-position shadow. It does not scan or
validate those constructed caches afterward.

## Task 1: Fixed M1/M2 post-conv factory

Files:

- `mtplx/gdn_capture.py`
- `mtplx/kernel_selfcheck.py`
- `tests/test_gdn_postconv_fusion.py`
- `tests/test_kernel_selfcheck.py`

Status: implemented and GPU-verified.

- [x] Add fixed M1 and M2/TGY4 Metal entrypoints with encoded geometry.
- [x] Add deterministic nonzero-state stock parity for both entrypoints.
- [x] Return a frozen factory containing separate M1/M2 callable tuples.
- [x] Reject the orphaned post-conv-on/compiled-target-prefix-off flag
  combination before self-check or installation.
- [x] Remove module markers, dynamic logical-M routing, and generic hot-path
  engagement.
- [x] Add an unchecked exact GDN forward with direct projections, stock Conv1d
  capture, the prebound post-conv callable, direct final cache leaves, stock
  norm, and direct output projection.
- [x] Add a separate exact 40-layer forward whose linear positions consume the
  callable tuple and whose full-attention positions never advance it.
- [x] Run focused MLX parity tests in the exclusive GPU lane.

## Task 2: Fixed captured-primary state route

Files:

- `mtplx/a3b_compiled_target_prefix.py`
- `mtplx/runtime.py`
- `mtplx/generation.py` for direct factory ownership only
- `tests/test_a3b_compiled_target_prefix.py`
- `tests/test_graphbank_compiled_verify.py`

Status: implemented and independently reviewed; full-model benchmark parity pending.

- [x] Keep the post-conv and compiled-target factories on the runtime, not the
  model.
- [x] Validate only compiled-target-specific external facts after trusting the
  post-conv factory's existing A3B proof.
- [x] Build separate compiled M2 verification and M1 continuation callables.
- [x] Define the exact 90-leaf state layout: 30 GDN pairs plus ten attention
  triples.
- [x] Return M2 primary state before final M2 state.
- [x] Use GDN row-zero captures for primary state.
- [x] Reuse full-attention M2 K/V buffers with tensor offset minus one.
- [x] Remove generic bank construction, cache promotion scans, cache
  validation, bucket probes, eager checks, and failure dictionaries.
- [x] Derive M2/M1 engagement from existing request statistics.
- [ ] Run full state-continuation parity in the exclusive GPU lane.

## Task 3: Generation rejection schedule

Files:

- `mtplx/generation.py`
- generation and compiled-route tests

Status: implemented, independently reviewed, and schedule-verified.

Use TDD and preserve sampling order exactly.

- [x] Save the M2 primary-state tuple returned by `verify_m2`.
- [x] Skip snapshot construction for the installed exact route without reading
  an environment flag in the cycle.
- [x] On acceptance, retain M2 final state and existing live logits/hidden.
- [x] On rejection, call `repair_m1([[correction]], primary_state)` directly.
- [x] Install only M1 final state; do not trim, roll back, capture-commit,
  re-forward M2, or use `pending_primary`.
- [x] Pair the correction token with `verify_hidden[:,0:1,:]` in committed MTP
  history.
- [x] Use M1 logits/hidden as live target outputs while leaving the next normal
  primary sample in its original position.
- [x] Add accept-only, reject-only, and mixed spy-route schedule tests.
- [x] Reject greedy or non-top-k sampler requests before prompt construction;
  the installed rejection path can therefore consume its correction directly.
- [x] Validate exact and generic target-prefix sampler compatibility once before
  prompt construction; every valid request calls the raw sampler directly.
- [x] Commit the exact rejection correction directly before M1; keep the
  optional correction branch only after the exact path has continued.
- [x] Prove the exact target forward feeds the one unchanged common target
  sampler call before acceptance logic; do not add an RNG experiment.
- [x] Run those schedule tests in the exclusive GPU lane and prove token and
  schedule parity.
- [x] Prove every non-exact strategy retains its existing behavior.

## Verification

Do not start MLX work while another process owns
`/tmp/mtplx-gpu-exclusive.lock`.

After Task 3 is complete, run:

```bash
python -m pytest -q \
  tests/test_gdn_postconv_fusion.py \
  tests/test_kernel_selfcheck.py \
  tests/test_a3b_compiled_target_prefix.py \
  tests/test_graphbank_compiled_verify.py

python -m pytest -q tests
python -m ruff check mtplx tests
git diff --check
```

Review the exact transitive call graph and search the installed path for
environment reads, validation, eligibility, fallback, generic capture,
snapshot, rollback, trim, and `pending_primary` before benchmarking.

## Benchmark

Commit the complete isolated candidate, then use only
`bench/a3b/run_a3b_174_captured_repair_m1.py` with:

- unchanged `960849b` control;
- upstream `long_code_uncapped` natural stop;
- K1 and TGY4;
- process-isolated C/X/C/X;
- repeat 2 verdict cells;
- exact factory, M1/M2 graph-key, call-count, and zero-growth gates;
- output SHA, acceptance, and tokens/cycle parity.

Do not use fixed 1024/1024, TGY8, RNG experiments, direct-call microbenchmark
verdicts, barriers, or replacement instrumentation. Record the isolated result
in `bench/a3b/OPTIMIZATION_LEDGER.md`; do not stack a wash or regression.
