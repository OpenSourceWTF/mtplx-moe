# Qwen4 Row-Serial Cycle Fold Implementation Plan

> **Outcome: rejected on the exact production gate.** The row-serial arithmetic
> and output trajectory were exact, but proactive `mx.async_eval` moved work
> into the following verifier and reduced decode throughput. The full-root
> candidate measured 63.3944 tok/s (16.1529 s) and the output-owned isolate
> measured 61.8677 tok/s (16.5515 s), versus the unchanged 69.6408 tok/s
> (14.7040 s) control. Generation-loop and harness integration were removed.
> The next investigation is compiled graph reuse without early async submission.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit the next exact Qwen4 depth-one MTP draft before CPU telemetry and streaming work, while retaining the existing sequential M1 arithmetic and complete QSA cache ownership.

**Architecture:** The retained target verifier still commits only the accepted prefix. After that commit establishes the next primary, a construction-installed Qwen4 route builds the ordinary M1 draft against the request-owned MTP cache and submits its logits, hidden state, and every QSA state leaf with one `mx.async_eval`. On an all-accept depth-one cycle this draft depends on the existing lazy accepted-draft history M1, producing the intended exact M1+M1 row-serial chain. On a rejection, the correction is deliberately removed from the history append, so the ticket contains only the ordinary correction draft M1; inventing a correction history row would change arithmetic. The following iteration consumes either ticket instead of rebuilding the draft; no physical M2 call, dedicated stream, Python worker, hot-path eligibility fallback, or changed sampler is introduced.

**Tech Stack:** Python 3.12, MLX 0.32.2, pytest, existing Qwen4 capture/runtime and `generate_mtpk` infrastructure.

**Assumptions:**

- Assumes the exact `qwen4_exp` one-layer MTP artifact and its single native
  `QSAKVCache` entry — will NOT install for another family, paged MTP cache,
  adapter-backed head, or different topology.
- Assumes depth one, persistent committed-history cache, `capture_commit`, no grammar constraint, and the stock stochastic draft loop — will NOT run for adaptive depth, context-copy-owned cycles, device draft cores, target-prefix verification, or request shapes other than the proven lane.
- Assumes an already-built lazy history append is safe to chain into the next ordinary M1 draft on all-accept cycles — rejection tickets remain one physical M1 because the accepted-prefix cache owns no correction history row.

---

## File structure

- Create `mtplx/qwen4_cycle_fold.py`: construction validator, immutable ticket, exact M1 issue method, complete QSA root extraction, and prebound installation.
- Modify `mtplx/qwen4_capture.py`: install the cycle-fold callable only after the exact capture topology has passed construction checks.
- Modify `mtplx/runtime.py`: typed optional cycle-fold callable on `MTPLXRuntime`.
- Modify `mtplx/generation.py`: prebind request eligibility, avoid tickets for a context-copy cycle, issue after accepted-prefix ownership is settled, and consume before ordinary draft construction.
- Create `tests/test_qwen4_cycle_fold.py`: construction and state-root ownership tests.
- Modify `tests/test_generation_sustained.py`: accepted/rejected ticket sequencing, context-copy exclusion, stop/max-token exclusion, and output identity tests.
- Modify `scripts/qwen38_flash_next_oq4_harness.py`: construction-time A/B switch and receipt lane declaration.
- Modify `tests/test_qwen38_resident_harness.py`: harness lane contract tests.
- Modify `docs/qwen4-pr368-optimization-ledger.md`: exact parity, benchmark, and profiler result.

### Task 1: Construction-installed exact M1 ticket

**Files:**
- Create: `mtplx/qwen4_cycle_fold.py`
- Modify: `mtplx/qwen4_capture.py`
- Modify: `mtplx/runtime.py`
- Test: `tests/test_qwen4_cycle_fold.py`
- Test: `tests/test_qwen4_capture.py`

**Security flag:** none

**Does NOT cover:** Generation-loop routing, context-copy interaction, benchmarking, or any non-Qwen4 cache topology.

- [ ] **Step 1: Write failing construction and ownership tests**

```python
def test_installer_binds_only_the_exact_qwen4_qsa_topology():
    runtime = exact_qwen4_runtime()
    report = install_qwen4_cycle_fold(runtime, config=exact_qwen4_config())
    assert report == {"installed": True, "ticket_rows": 1, "qsa_layers": 1}
    assert callable(runtime.qwen4_cycle_fold_issue)


def test_ticket_submits_logits_hidden_and_all_qsa_state_roots(monkeypatch):
    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda *roots: submitted.extend(roots))
    runtime, cache = exact_fake_runtime_and_cache()
    install_qwen4_cycle_fold(runtime, config=exact_qwen4_config())
    ticket = runtime.qwen4_cycle_fold_issue(
        hidden=mx.zeros((1, 1, 2560), dtype=mx.bfloat16),
        primary=17,
        mtp_cache=cache,
        mtp_hidden_variant="post_norm",
        compiled_aux_prefetch="owned-prefetch",
    )
    assert ticket.primary == 17
    assert ticket.compiled_aux_prefetch == "owned-prefetch"
    assert submitted[:2] == [ticket.logits, ticket.hidden]
    assert tuple(submitted[2:]) == tuple(
        leaf for entry in cache for leaf in entry.state_leaves
    )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_qwen4_cycle_fold.py tests/test_qwen4_capture.py::test_installer_binds_exact_qwen4_capture_route`

Expected: FAIL because `qwen4_cycle_fold` and `runtime.qwen4_cycle_fold_issue` do not exist.

- [ ] **Step 3: Add the ticket and construction-bound issuer**

Implement this public shape in `mtplx/qwen4_cycle_fold.py`:

```python
@dataclass(frozen=True)
class Qwen4CycleFoldTicket:
    primary: int
    logits: Any
    hidden: Any
    compiled_aux_prefetch: Any | None


def _issue_qwen4_cycle_fold(
    self,
    *,
    hidden: Any,
    primary: int,
    mtp_cache: Any,
    mtp_hidden_variant: str,
    compiled_aux_prefetch: Any | None,
) -> Qwen4CycleFoldTicket:
    with attention_phase(None):
        logits, next_hidden = self.draft_mtp(
            hidden,
            mx.array([[int(primary)]]),
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=1,
            position_offset=mtp_position_offset_for_cache(mtp_cache),
        )
    roots = tuple(
        leaf
        for entry in mtp_cache
        for leaf in entry.state_leaves
    )
    mx.async_eval(logits, next_hidden, *roots)
    return Qwen4CycleFoldTicket(
        primary=int(primary),
        logits=logits,
        hidden=next_hidden,
        compiled_aux_prefetch=compiled_aux_prefetch,
    )
```

`install_qwen4_cycle_fold` must validate the exact config and the actual runtime
cache allocation route's single QSA entry once, then bind
`_issue_qwen4_cycle_fold` with `MethodType`. Add
`qwen4_cycle_fold_issue: Callable[..., Any] | None = field(default=None,
init=False, repr=False)` to `MTPLXRuntime`. Install only after the exact
depth-one n-gram/runtime resources have been bound.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/test_qwen4_cycle_fold.py tests/test_qwen4_capture.py`

Expected: PASS; invalid topology fails once during installation and the enabled issuer contains no eligibility branch.

- [ ] **Step 5: Commit**

```bash
git add mtplx/qwen4_cycle_fold.py mtplx/qwen4_capture.py mtplx/runtime.py tests/test_qwen4_cycle_fold.py tests/test_qwen4_capture.py
git commit -m "perf(qwen4): install exact row-serial cycle ticket"
```

### Task 2: Holistic generation-loop ticket ownership

**Files:**
- Modify: `mtplx/generation.py`
- Modify: `tests/test_generation_sustained.py`

**Security flag:** none

**Does NOT cover:** Depth greater than one, target-prefix lanes, grammar-constrained decoding, context-copy-owned rounds, adapter ensembles, device draft cores, or tickets issued before stop/max/repetition decisions are known.

- [ ] **Step 1: Write failing sequencing tests**

Add a fake issuer that records `("issue", primary)`, returns known logits/hidden, and increments the fake cache exactly once. Cover these assertions:

```python
assert candidate.tokens == control.tokens
assert candidate.stats.accepted_by_depth == control.stats.accepted_by_depth
assert candidate.stats.correction_tokens == control.stats.correction_tokens
assert candidate_runtime.events.index(("issue", next_primary)) < candidate_runtime.events.index(("emit", next_primary))
assert candidate_runtime.direct_draft_calls == control_runtime.direct_draft_calls - candidate_runtime.ticket_consumptions
```

Add separate all-accept and reject-correction cases. Assert the all-accept ticket is dependency-chained after exactly one accepted-draft history append, while the reject ticket is issued after zero history rows and contains only its ordinary correction draft M1. Add cases proving zero issues when the next loop would take a context-copy block, when the pending primary is a stop token, or when generation has reached `max_tokens`.

- [ ] **Step 2: Run the sequencing tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_generation_sustained.py -k 'cycle_fold_ticket'`

Expected: FAIL because `generate_mtpk` never issues or consumes a cycle ticket.

- [ ] **Step 3: Prebind the request route and integrate one-owner issue/consume**

At generation setup, bind one callable or `None` from request-invariant conditions:

```python
cycle_fold_issue = (
    rt.qwen4_cycle_fold_issue
    if rt.qwen4_cycle_fold_issue is not None
    and speculative_depth == 1
    and verify_strategy == "capture_commit"
    and mtp_cache_policy == "persistent"
    and _mtp_history_uses_committed_cache(mtp_history_policy)
    and constraint is None
    and draft_core == "stock"
    and adaptive_policy is None
    and adaptive_width_policy is None
    and mtp_corrector is None
    and mtp_topk_reranker is None
    and not adapter_ensemble_q
    else None
)
cycle_fold_ticket = None
```

Before the ordinary draft call, consume only a ticket whose `primary` equals the already committed `primary`; use its logits/hidden and transfer its `compiled_aux_prefetch` to the verifier. The enabled route must not catch and fall back.

After accepted-prefix target and MTP-cache ownership is complete, issue the next ticket before `append_event`, `emit_new_tokens`, or `emit_trace`. For an all-accept cycle, issue after the accepted-draft history M1 and bonus sampling, so the ticket's M1 is chained behind that exact append. For rejection, issue after the correction becomes `pending_primary` and rollback completes, but do not append the correction to history: this ticket is draft-only M1. Do not issue if stop, max-token, repetition-stop, or this pure lookahead says context copy owns the next cycle:

```python
def context_copy_would_own_next_cycle(next_primary: int) -> bool:
    if not (ccopy_active and _ccopy_capture_lane):
        return False
    if len(tokens) < ccopy_suspend_until:
        return False
    pos, ext = ccopy_index.find(prompt_ids + tokens, max_pos=len(prompt_ids))
    return pos is not None and ext >= ccopy_min_ext
```

Start `compiled_aux_prefetcher(primary=next_primary, prior_context=...)` before issuing and store ownership on the ticket. Do not duplicate that prefetch when consuming the ticket.

- [ ] **Step 4: Run generation and context-copy tests**

Run: `.venv/bin/python -m pytest -q tests/test_generation_sustained.py tests/test_context_copy_stats.py tests/test_ccopy_bank_route.py`

Expected: PASS with identical token/acceptance outcomes and no ticket mutation on excluded cycles.

- [ ] **Step 5: Commit**

```bash
git add mtplx/generation.py tests/test_generation_sustained.py
git commit -m "perf(qwen4): submit next draft before host bookkeeping"
```

### Task 3: Exact harness lane and parity receipt

**Files:**
- Modify: `scripts/qwen38_flash_next_oq4_harness.py`
- Modify: `tests/test_qwen38_resident_harness.py`
- Modify: `tests/test_qwen4_cycle_fold.py`

**Security flag:** none

**Does NOT cover:** Promotion TPS; this task proves exact real-state arithmetic and cache ownership before performance measurement.

- [ ] **Step 1: Write failing harness contract tests**

```python
def test_cycle_fold_candidate_is_construction_time_and_receipted(monkeypatch):
    args = parse_harness_args(["--cycle-fold"])
    lanes = install_benchmark_lanes(args)
    assert lanes["MTPLX_QWEN4_CYCLE_FOLD"] == "1"


def test_production_receipt_rejects_cycle_fold_digest_or_trajectory_drift():
    assert_production_parity(
        control=canonical_control,
        candidate=cycle_fold_candidate,
        fields=(
            "prompt_token_sha256",
            "output_token_sha256",
            "accepted_drafts",
            "drafted_tokens",
            "verify_calls",
            "correction_tokens",
        ),
    )
```

- [ ] **Step 2: Run harness tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/test_qwen38_resident_harness.py -k 'cycle_fold'`

Expected: FAIL because the harness has no construction-time cycle-fold lane.

- [ ] **Step 3: Add the guarded A/B switch and real 16K parity probe**

Add `--cycle-fold/--no-cycle-fold`, defaulting off until the performance gate. Export it before `load()` and include it in `optimization_lanes`. Extend the guarded parity probe to compare the candidate with the unchanged production control at the frozen prompt digest, including output digest, 391/600 trajectory, 605 verifier calls, zero repair, and complete final QSA state/offset parity.

- [ ] **Step 4: Run focused tests, then the guarded real parity pair**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_qwen38_resident_harness.py tests/test_qwen4_cycle_fold.py
.venv/bin/python scripts/qwen38_flash_next_oq4_harness.py --depth 1 --verify-strategy capture_commit --warmup-runs 1 --no-cycle-fold --output /Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/pr368/cycle-fold-control-parity-r1.json
.venv/bin/python scripts/qwen38_flash_next_oq4_harness.py --depth 1 --verify-strategy capture_commit --warmup-runs 1 --cycle-fold --output /Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/pr368/cycle-fold-candidate-parity-r1.json
```

Expected: Both guarded runs pass under `/tmp/mtplx-gpu-exclusive.lock`; candidate matches the frozen prompt/output digest, trajectory, final state, and zero-repair contract.

- [ ] **Step 5: Commit**

```bash
git add scripts/qwen38_flash_next_oq4_harness.py tests/test_qwen38_resident_harness.py tests/test_qwen4_cycle_fold.py
git commit -m "test(qwen4): gate row-serial fold on exact production state"
```

### Task 4: Counterbalanced production and profiler gate

**Measured disposition:** rejected before ABBA because both exact candidate
isolates were materially below the existing control while preserving the same
prompt digest, output digest, 391/600 acceptance trajectory, 605 verifier calls,
five context-copy rounds, and zero repair.

**Files:**
- Modify: `docs/qwen4-pr368-optimization-ledger.md`
- Modify: `scripts/qwen38_flash_next_oq4_harness.py` only if the candidate is retained and its default flips on.
- Modify: `tests/test_qwen38_resident_harness.py` only if the retained default changes.

**Security flag:** none

**Does NOT cover:** Temperature-zero vanity results, shorter prompts, MTP depths 2/3, or a physical M2 fold.

- [ ] **Step 1: Run an ABBA exact production sequence**

Run control/candidate/candidate/control with the Task 3 command, unique receipt names, one warmup each, and no other lane changes. Require the candidate mean decode TPS to beat the matched control mean and preserve every parity field.

- [ ] **Step 2: Re-profile the exact winner**

Use the existing MLX 0.32.2 production profiler path. Report decode-window GPU busy, idle, utilization, and the five named transition families from the optimization ledger. Require no utilization regression and specifically compare the 391 accepted-cycle PLE-to-gather gaps and 213 rejection decision gaps.

- [ ] **Step 3: Retain or revert from evidence**

If the repeated candidate wins, make `--cycle-fold` the exact harness default and retain the construction route. If parity fails or TPS does not improve, revert Tasks 1-3 completely while preserving only the measured rejection row in the ledger.

- [ ] **Step 4: Run the merged verification suite**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_qwen4_cycle_fold.py tests/test_qwen4_capture.py tests/test_generation_sustained.py tests/test_context_copy_stats.py tests/test_ccopy_bank_route.py tests/test_graphbank_compiled_verify.py tests/test_qwen38_resident_harness.py tests/test_qwen4_exp_capture_commit.py
.venv/bin/python -m py_compile mtplx/qwen4_cycle_fold.py mtplx/qwen4_capture.py mtplx/runtime.py mtplx/generation.py scripts/qwen38_flash_next_oq4_harness.py
git diff --check
```

Expected: PASS, no conflict markers or whitespace errors, service restored healthy, and no orphan guarded worker.

- [ ] **Step 5: Commit, push, and update the PR**

```bash
git add docs/qwen4-pr368-optimization-ledger.md scripts/qwen38_flash_next_oq4_harness.py tests/test_qwen38_resident_harness.py
git commit -m "docs(qwen4): record row-serial production gate"
git push moe HEAD:port/qwen38-flash-next-resident-q4
```

Update PR #368's benchmark-history and accepted/rejected optimization tables with receipt names, current commit, TPS, acceptance, verify calls, repair time, GPU busy/idle/utilization, and the maintainer-closed status if it has not been reopened.

## Self-review

- Spec coverage: construction invariants, exact sequential M1 arithmetic, complete QSA roots, accepted/rejected ownership, context-copy exclusion, stop/final-state safety, production parity, ABBA performance, profiling, commit/push, and PR reporting are each assigned.
- Placeholder scan: no TBD/TODO or unspecified error-handling step remains.
- Type consistency: `Qwen4CycleFoldTicket`, `qwen4_cycle_fold_issue`, and `compiled_aux_prefetch` use the same names across construction, generation, tests, and harness.
- Scope reduction scan: the plan preserves the approved depth-one production contract and explicitly excludes physical M2 arithmetic rather than silently weakening it.
