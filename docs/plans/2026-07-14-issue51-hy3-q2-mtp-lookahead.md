# Issue 51 Hy3-Q2 MTP Look-ahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:executing-plans` to implement this plan task-by-task.
> Use `superpowers-optimized:test-driven-development` for every behavior
> change, `superpowers-optimized:performance-investigation` for every timing
> decision, and `superpowers-optimized:verification-before-completion` before
> any completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate three related but independently attributable tracks:
Next-K speculative execution, the persistent MTP cache/commit lifecycle, and
Q2 inner-product/NAX execution for Hy3 at 1,024/2,048 input tokens and exactly
128 output tokens.

**Architecture:** Extend the corrected Q2 depth-matrix runner with an explicit,
fail-closed candidate contract while preserving its current defaults. Run K=1
against K=0 first in separate headline-speed and diagnostic-utilization lanes;
K=2 requires a passing K=1 summary, and K>=3 is rejected. Every row verifies
the persistent MTP cache, committed history, capture/rollback/commit, terminal
offsets, and safe resume state. Reuse retained traces for predictive-cache and
Q2 operator evidence; NAX remains third priority and receives no production
kernel unless measured headroom clears the gate.

**Tech Stack:** Python 3.12, MLX 0.31.x, mlx-lm, pytest, Ruff, JSON/JSONL,
existing `CompiledVerifyBank`, existing streamed Hy3 component-bank experts,
existing Qwen exclusive-window guard, and authenticated `gh` for Issue #51.

**Assumptions:**

- Work stays in
  `/Users/davidtai/projects/OpenSourceWTF/.worktrees/51-q2-mtp-lookahead` on
  `experiment/issue51-q2-mtp-lookahead` — it will NOT use or overwrite the
  dirty files in `codex/q2-bf16-mtp-bench`.
- Commit `7a3bbb42a5077164b43b5ec023424d0d4da4e645` is the exact committed v3
  qualification/greedy-correction prerequisite — it will NOT pull the later
  Issue #31 merge or unrelated cold-tier change.
- The target remains `hy3-expert-q2` with affine Q2 group-size 64 streamed
  experts and a resident BF16 layer-80 MTP head — this plan will NOT
  re-quantize the target trunk or MTP head.
- The fixed contexts are 1,024/2,048 input and exactly 128 output with greedy
  sampling. K=0 is the one-row control, K=1 is measured first, K=2 is
  conditional, and K>=3 is out of scope.
- Qwen remains loaded during code/test preparation and is unloaded only by
  `scripts/run_with_qwen_stopped.py` around an exclusive hardware campaign —
  unit tests will NOT import or allocate the real model.
- A negative gate is a completed experimental result. Predictive runtime
  staging and a Q2 Metal/NAX kernel are forbidden unless their preceding
  measurement gates pass.

---

## Priority and dependency order

| Priority | Test | Entry gate | Terminal outcomes |
|---:|---|---|---|
| 1 | K=1 compiled Next-K plus MTP cache lifecycle | v3 prerequisite integrated | authorize K=2, or retain no-go evidence |
| 2 | MTP cache/predictive loading | K=1 has its own decision | offline no-go, or guarded runtime pilot |
| 3 | Q2 inner-product/NAX and grouping | cache track has its own decision | premise no-go, or guarded Q2 kernel prototype |

The next priority may begin after the prior test has a recorded decision; it
does not require the prior candidate to win.

## File structure

### Create

- `mtplx/benchmarks/issue51.py`: pure candidate validation, ABBA scheduling,
  paired statistics, predictor alignment/scoring, grouping histograms, and
  promotion decisions.
- `scripts/run_issue51_hy3_q2.py`: checkpointed stage runner for A1
  qualification/performance, B offline evidence, and A2 premise evidence.
- `scripts/summarize_issue51_hy3_q2.py`: pure-data validator and Markdown/JSON
  report renderer.
- `tests/test_issue51.py`: hardware-independent candidate, schedule,
  predictor, grouping, checkpoint, and summary tests.
- `benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.md`: curated
  evidence created only after the retained campaigns finish.
- `benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.json`: matching
  machine-readable decisions and provenance.

### Modify

- `scripts/benchmark_q2_mtp_depth_matrix.py`: explicit verify strategy,
  compiled mode, trace, and observer inputs with unchanged v3 defaults.
- `tests/test_benchmark_q2_mtp_depth_matrix.py`: RED/GREEN coverage for the
  candidate contract and evidence gates.
- `mtplx/generation.py`: optional diagnostic draft observer, inactive by
  default.
- `tests/test_generation_sustained.py`: observer order, depth, hidden-state,
  and failure propagation tests.
- `mtplx/expert_runtime.py`: monotonic route-observation timestamps only when
  route tracing is enabled.
- `tests/test_expert_slots_runtime.py`: trace timestamp and reset-order tests.
- `docs/specs/2026-07-14-issue51-hy3-q2-mtp-lookahead-design.md`: retain the
  user-corrected priority order.

### Conditional production files

These files are created only when the corresponding measured premise passes:

- `mtplx/expert_hints.py` and `tests/test_expert_hints.py`: isolated,
  demand-subordinate hint staging for a positive B-offline result.
- `mtplx/q2_nax.py` and `tests/test_q2_nax.py`: fail-closed Q2 affine NAX
  operator for a positive A2 premise result.

---

### Task 1: Integrate and verify the corrected v3 prerequisite

**Files:**

- Integrate commit: `7a3bbb42a5077164b43b5ec023424d0d4da4e645`
- Existing tests: `tests/test_generation_sustained.py`
- Existing tests: `tests/test_benchmark_q2_mtp_depth_matrix.py`
- Existing tests: `tests/test_expert_manifest.py`

**Security flag:** `none`

**Does NOT cover:** The later Issue #31 merge, cold-tier persistence work, or
the untracked target-divergence probe files in the source worktree.

- [ ] **Step 1: prove the isolated branch and source commit are cleanly scoped**

```bash
git status --short --branch
git show --stat --oneline 7a3bbb42a5077164b43b5ec023424d0d4da4e645
git diff --name-only bc49e27 7a3bbb42a5077164b43b5ec023424d0d4da4e645
```

Expected: the branch contains only the Issue #51 docs; the prerequisite
touches only the manifest, generation, Q2 matrix runner, and their tests.

- [ ] **Step 2: integrate only the prerequisite**

```bash
git cherry-pick 7a3bbb42a5077164b43b5ec023424d0d4da4e645
```

- [ ] **Step 3: verify the v3 and greedy-correction contracts**

```bash
.venv/bin/python -m pytest -q \
  tests/test_generation_sustained.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py \
  tests/test_expert_manifest.py
git diff --check
```

Expected: all selected tests pass, including
`test_generate_mtpk_greedy_rejection_commits_batched_verifier_correction` and
the v3 final-state gates.

---

### Task 2: Add a fail-closed Issue #51 candidate contract to the Q2 runner

**Files:**

- Modify: `scripts/benchmark_q2_mtp_depth_matrix.py`
- Test: `tests/test_benchmark_q2_mtp_depth_matrix.py`

**Security flag:** `none`

**Does NOT cover:** Changing the default `batched` lane, changing verifier
semantics, changing model weights, or switching candidate implementation
inside a generation call.

- [ ] **Step 1: write failing candidate validation tests**

Add tests with these exact assertions:

```python
args = module.build_parser().parse_args([])
assert args.verify_strategy == "batched"
assert args.compiled_verify_mode == "off"
assert args.trace_routes is False

with pytest.raises(module.BenchmarkConfigurationError, match="capture_commit"):
    module.run_depth_matrix(
        [_requests(tmp_path)[0]],
        contexts=(1024,),
        verify_strategy="batched",
        compiled_verify_mode="parity",
        apis=_fake_apis(module)[0],
    )
```

Add a fake D1 observation whose `stats.graphbank` is empty and require
`compiled_verify_mode="on"` to fail with `compiled verifier emitted no
evidence`. Add a fake compiled observation with `calls=4`, `compiled_calls=4`,
`fallback_calls=0`, and require it to pass.

- [ ] **Step 2: run RED**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py \
  -k 'verify_strategy or compiled_verify or trace_routes'
```

Expected: FAIL because the runner has no explicit candidate contract.

- [ ] **Step 3: add the explicit candidate inputs and unchanged defaults**

Add these public constants and parser choices:

```python
VERIFY_STRATEGIES = ("batched", "capture_commit")
COMPILED_VERIFY_MODES = ("off", "parity", "on")
```

Extend `run_depth_matrix`, `_run_depth_matrix_impl`, and `_run_observation`
with keyword-only `verify_strategy: str = "batched"`,
`compiled_verify_mode: str = "off"`, `trace_routes: bool = False`, and
`draft_observer: Callable[[Mapping[str, Any]], None] | None = None`.

Validate before model loading:

```python
if verify_strategy not in VERIFY_STRATEGIES:
    raise BenchmarkConfigurationError("unsupported verify strategy")
if compiled_verify_mode not in COMPILED_VERIFY_MODES:
    raise BenchmarkConfigurationError("unsupported compiled verify mode")
if compiled_verify_mode != "off" and verify_strategy != "capture_commit":
    raise BenchmarkConfigurationError(
        "compiled verify requires capture_commit verify strategy"
    )
```

Require the process environment to match the declared mode instead of
mutating it in the runner:

```python
observed_mode = (os.environ.get("MTPLX_COMPILED_VERIFY") or "off").strip().lower()
observed_mode = "on" if observed_mode in {"1", "true", "yes", "on"} else observed_mode
if observed_mode != compiled_verify_mode:
    raise BenchmarkConfigurationError(
        "declared compiled verify mode differs from process environment"
    )
```

Pass `verify_strategy`, `draft_observer`, and the existing fixed generation
arguments to `generate_mtpk`. Set `trace_routes` in `ExpertStreamingConfig`.
Record all three values under `configuration.candidate`.

- [ ] **Step 4: gate compiled evidence per retained row**

For `parity` and `on`, require
`stats.graphbank.compiled_verify.calls > 0`, `fallback_calls == 0`, and
`compiled_calls == calls`. For `off`, require no compiled calls. Preserve the
full stats dictionary in the row on both pass and failure.

- [ ] **Step 5: run GREEN and the unchanged-default regression**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py
.venv/bin/ruff check scripts/benchmark_q2_mtp_depth_matrix.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py
.venv/bin/ruff format --check scripts/benchmark_q2_mtp_depth_matrix.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py
git diff --check
```

- [ ] **Step 6: commit**

```bash
git add scripts/benchmark_q2_mtp_depth_matrix.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py
git commit -m "feat(bench): qualify issue 51 verifier candidates"
```

---

### Task 3: Build the checkpointed A1 ABBA campaign and summarizer

**Files:**

- Create: `mtplx/benchmarks/issue51.py`
- Create: `scripts/run_issue51_hy3_q2.py`
- Create: `scripts/summarize_issue51_hy3_q2.py`
- Create: `tests/test_issue51.py`

**Security flag:** `none`

**Does NOT cover:** B predictor scoring, A2 operator changes, Qwen lifecycle
inside unit tests, or combining candidates.

- [ ] **Step 1: write failing schedule, pairing, and checkpoint tests**

Define and test this fixed candidate order:

```python
ISSUE51_PRIORITY = (
    "compiled_whole_window",
    "mtp_hint_only_prediction",
    "q2_nax_grouping",
)
A1_CANDIDATES = (
    "batched-stock",
    "capture-eager",
    "capture-compiled-parity",
    "capture-compiled",
)

schedule = build_abba_schedule(
    control="capture-eager",
    candidate="capture-compiled",
    retained_pairs=8,
)
assert [row.arm for row in schedule] == [
    "capture-eager", "capture-compiled", "capture-compiled", "capture-eager"
] * 4
assert len(pair_abba_rows(schedule)) == 8
```

Tests must also reject duplicate artifact paths, a non-passed child payload,
wrong contexts/depths/output count, missing final-state gates, fallback counts,
non-finite metrics, incomplete ABBA blocks, and replacement of an existing raw
artifact.

- [ ] **Step 2: run RED**

```bash
.venv/bin/python -m pytest -q tests/test_issue51.py
```

Expected: FAIL because the campaign module and scripts do not exist.

- [ ] **Step 3: implement the pure campaign contract**

Use frozen dataclasses `CampaignCell(context_tokens, depth)` and
`ScheduledRun(index, block, arm, pair_slot)`. Implement the schedule exactly:

```python
def build_abba_schedule(
    *, control: str, candidate: str, retained_pairs: int = 8
) -> tuple[ScheduledRun, ...]:
    if retained_pairs <= 0 or retained_pairs % 2:
        raise ValueError("retained_pairs must be a positive even integer")
    rows = []
    for block in range(retained_pairs // 2):
        for pair_slot, arm in enumerate((control, candidate, candidate, control)):
            rows.append(
                ScheduledRun(
                    index=len(rows),
                    block=block,
                    arm=arm,
                    pair_slot=pair_slot,
                )
            )
    return tuple(rows)
```

Expose `pair_abba_rows(schedule)`, `validate_a1_child(payload, *, arm)`,
`paired_decode_statistics(rows)`, and
`decide_performance(stats, *, default_threshold=0.05)`. The validator requires
`status == "passed"`, `passed is True`, the fixed four cells, and the compiled
evidence contract from Task 2. `paired_decode_statistics` reports every paired
delta, mean, median, p95, and a deterministic 10,000-resample bootstrap 95%
interval using `random.Random(0)`. `decide_performance` requires a positive
lower interval bound; default promotion additionally requires a mean
fractional decode gain of at least 0.05.

- [ ] **Step 4: implement one-candidate-per-child execution**

`scripts/run_issue51_hy3_q2.py a1` must invoke
`scripts/benchmark_q2_mtp_depth_matrix.py` as a subprocess for each schedule
row. Each child gets exactly one of these fixed environments/arguments:

```python
A1_PROCESS_CONFIG = {
    "batched-stock": ("off", "batched"),
    "capture-eager": ("off", "capture_commit"),
    "capture-compiled-parity": ("parity", "capture_commit"),
    "capture-compiled": ("on", "capture_commit"),
}
```

Qualification runs each of the four candidates once. Performance excludes
parity and compares stock versus eager first, then the winning eager control
versus compiled in four ABBA blocks. Checkpoint after every complete child and
never overwrite an existing child artifact.

- [ ] **Step 5: implement pure-data summary verification**

The summarizer must refuse raw input whose file digest, candidate declaration,
matrix, gates, or schedule disagrees with the campaign index. Render separate
A1 correctness and performance decisions; dispatch count without a positive
end-to-end interval is a no-go.

- [ ] **Step 6: run GREEN and commit**

```bash
.venv/bin/python -m pytest -q tests/test_issue51.py
.venv/bin/ruff check mtplx/benchmarks/issue51.py \
  scripts/run_issue51_hy3_q2.py scripts/summarize_issue51_hy3_q2.py \
  tests/test_issue51.py
.venv/bin/ruff format --check mtplx/benchmarks/issue51.py \
  scripts/run_issue51_hy3_q2.py scripts/summarize_issue51_hy3_q2.py \
  tests/test_issue51.py
git diff --check
git add mtplx/benchmarks/issue51.py scripts/run_issue51_hy3_q2.py \
  scripts/summarize_issue51_hy3_q2.py tests/test_issue51.py
git commit -m "feat(bench): orchestrate issue 51 verifier campaign"
```

---

### Task 4: Run K=1 qualification, speed, and utilization on the exclusive lane

**Files:**

- Raw: `benchmarks/raw/issue51/<run-id>/a1/`
- Update after validation: Issue #51 evidence comment

**Security flag:** `none`

**Does NOT cover:** B or A2 execution, a combined candidate, or changing a
candidate during an in-flight Metal dispatch.

- [ ] **Step 1: verify tests, model artifacts, branch identity, and Qwen state**

```bash
git status --short --branch
.venv/bin/python -m pytest -q tests/test_issue51.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py \
  tests/test_generation_sustained.py
curl -fsS http://127.0.0.1:8080/v1/models
```

- [ ] **Step 2: run the guarded campaign**

```bash
.venv/bin/python scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 1800 \
  -- \
  .venv/bin/python scripts/run_issue51_hy3_q2.py a1 \
  --contexts 1024,2048 \
  --depths 1 \
  --output-tokens 128 \
  --retained-pairs 8 \
  --diagnostic-repeats 4 \
  --output-dir benchmarks/raw/issue51/2026-07-14-k1
```

Expected: the wrapper restores the exact captured Qwen model on every exit;
the campaign either completes all cells or retains the exact failed cell and
stops that candidate.

- [ ] **Step 3: verify restoration and summarize**

```bash
curl -fsS http://127.0.0.1:8080/v1/models
.venv/bin/python scripts/summarize_issue51_hy3_q2.py \
  --input benchmarks/raw/issue51/2026-07-14-k1/index.json \
  --output-json benchmarks/raw/issue51/2026-07-14-k1/summary.json
```

- [ ] **Step 4: record the K=1 decision**

Post the exact candidate, commit, matrix, fallback/parity result, paired decode
interval, active-reader interval, and go/no-go to Issue #51. K=1 is complete
even when rejected.

- [ ] **Step 5: run K=2 only when the K=1 summary authorizes it**

```bash
.venv/bin/python scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 1800 \
  -- \
  .venv/bin/python scripts/run_issue51_hy3_q2.py a1 \
  --contexts 1024,2048 --depths 2 --output-tokens 128 \
  --retained-pairs 8 --diagnostic-repeats 4 \
  --k1-summary benchmarks/raw/issue51/2026-07-14-k1/summary.json \
  --output-dir benchmarks/raw/issue51/2026-07-14-k2
```

Expected: the runner refuses K=2 when the K=1 summary is missing, malformed,
or reports a non-positive speed/utilization gate. No K=3 command exists.

---

### Task 5: Add diagnostic draft-hidden and target-route evidence

**Files:**

- Modify: `mtplx/generation.py`
- Modify: `mtplx/expert_runtime.py`
- Test: `tests/test_generation_sustained.py`
- Test: `tests/test_expert_slots_runtime.py`

**Security flag:** `none`

**Does NOT cover:** Runtime prefetch, cache admission, eviction, demand
priority, or any callback cost in normal generation.

- [ ] **Step 1: write failing observer tests**

Add a D2 tiny-model test:

```python
observed = []
out = generate_mtpk(
    _runtime(TinyMTPModel(), mtp_enabled=True),
    [0],
    max_tokens=4,
    speculative_depth=2,
    sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
    stop_token_ids=set(),
    draft_observer=observed.append,
)
assert out.stats.generated_tokens == 4
assert observed
assert {row["depth"] for row in observed} == {1, 2}
assert all(row["cycle"] >= 0 for row in observed)
assert all(row["token"] >= 0 for row in observed)
assert all(hasattr(row["hidden"], "shape") for row in observed)
```

Add a second test proving an observer exception propagates before target
verification, and an observer-omitted test proving no new stats/event field is
emitted.

For route tracing, monkeypatch `mtplx.expert_runtime.time.monotonic_ns` to
return `123456789` and assert every non-reset trace entry includes
`observed_ns == 123456789`, while reset entries remain ordering markers.

- [ ] **Step 2: run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_generation_sustained.py \
  tests/test_expert_slots_runtime.py \
  -k 'draft_observer or route_trace_timestamp'
```

- [ ] **Step 3: implement the diagnostic-only observer**

Add keyword-only `draft_observer: Callable[[Mapping[str, Any]], None] | None =
None` to `generate_mtpk`. Immediately after sampling each successful draft and
before the next depth, call it with:

```python
{
    "cycle": int(len(events)),
    "depth": int(depth_index + 1),
    "token": int(draft_token),
    "hidden": draft_hidden_next,
    "available_ns": time.monotonic_ns(),
}
```

Do not evaluate, copy, serialize, or retain the hidden state when the callback
is absent. In `ExpertStreamingRuntime.observe_route`, add `observed_ns` only
inside the existing `trace_routes` branch.

- [ ] **Step 4: run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/test_generation_sustained.py tests/test_expert_slots_runtime.py
.venv/bin/ruff check mtplx/generation.py mtplx/expert_runtime.py \
  tests/test_generation_sustained.py tests/test_expert_slots_runtime.py
git diff --check
git add mtplx/generation.py mtplx/expert_runtime.py \
  tests/test_generation_sustained.py tests/test_expert_slots_runtime.py
git commit -m "feat(bench): expose issue 51 draft route evidence"
```

---

### Task 6: Implement and run the priority-2 offline predictor gate

**Files:**

- Modify: `mtplx/benchmarks/issue51.py`
- Modify: `scripts/run_issue51_hy3_q2.py`
- Modify: `scripts/summarize_issue51_hy3_q2.py`
- Modify: `tests/test_issue51.py`
- Conditional create after a positive gate: `mtplx/expert_hints.py`
- Conditional test after a positive gate: `tests/test_expert_hints.py`

**Security flag:** `none`

**Does NOT cover:** Letting predictions alter router IDs, evict demand entries,
consume the authoritative reserve, or publish incomplete speculative records.

- [ ] **Step 1: write failing alignment and scoring tests**

Use synthetic D2 evidence with two draft predictions and a 3-row target route.
Require `align_predictions_to_verify_routes` to compare depth 1 with row 1 and
depth 2 with row 2, never row 0. Test N=1/2/4/8/12, zero hint, prior-token,
transition baseline, and oracle. Reject missing layers, duplicate candidates,
widths above 12, non-monotonic timestamps, incomplete verifier windows, and
reserve values below 8 operations or 84,934,656 bytes.

- [ ] **Step 2: run RED**

```bash
.venv/bin/python -m pytest -q tests/test_issue51.py \
  -k 'predictor or hint or oracle or lead_time'
```

- [ ] **Step 3: implement the offline predictor**

Implement `MtpRouterProbe` in the runner. For each draft hidden row, evaluate
every target routed layer's existing BF16 router in FP32, retain the ordered
top 12 expert IDs, materialize those small ID arrays, and record
`available_ns`. Do not alter the model or cache.

Expose `align_predictions_to_verify_routes(predictions, routes, *, depth,
top_k=8)` and `score_hint_policy(aligned, *, candidate_width, record_bytes,
measured_bytes_per_second)`. Alignment uses this exact row rule:

```python
row_start = prediction_depth * top_k
row_stop = row_start + top_k
actual_experts = tuple(verify_route["expert_ids"][row_start:row_stop])
if len(actual_experts) != top_k:
    raise ValueError("verify route does not contain the predicted draft row")
```

Report miss-specific recall, candidate union, lead time, estimated
ready-before-demand, late/cancelled/unused records, physical bytes, byte
amplification, and wait removable. Keep predictor timing and trace runs out of
headline TPS.

- [ ] **Step 4: run GREEN and commit the offline tooling**

```bash
.venv/bin/python -m pytest -q tests/test_issue51.py
.venv/bin/ruff check mtplx/benchmarks/issue51.py \
  scripts/run_issue51_hy3_q2.py scripts/summarize_issue51_hy3_q2.py \
  tests/test_issue51.py
git diff --check
git add mtplx/benchmarks/issue51.py scripts/run_issue51_hy3_q2.py \
  scripts/summarize_issue51_hy3_q2.py tests/test_issue51.py
git commit -m "feat(bench): score issue 51 MTP expert hints"
```

- [ ] **Step 5: run the guarded B-offline campaign**

```bash
.venv/bin/python scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 1800 \
  -- \
  .venv/bin/python scripts/run_issue51_hy3_q2.py predictor \
  --contexts 1024,2048 --depths 1,2 --output-tokens 128 \
  --candidate-widths 1,2,4,8,12 \
  --output-dir benchmarks/raw/issue51/2026-07-14-predictor
```

- [ ] **Step 6: apply the runtime-pilot branch exactly**

If the interference-adjusted lower confidence bound on useful wait saved is
non-positive, record B-runtime as `not_started_offline_no_go` and create no
runtime files. If positive, create `ExpertHintStagingPool` in
`mtplx/expert_hints.py` with exact-key
`(layer, expert, component, generation)` adoption, a separate byte/operation
budget, strict demand priority, and immutable authoritative reserve. Write
RED/GREEN tests proving hints cannot evict, pin, delay, publish, or consume
demand reserve; then run zero/oracle/predictor/demand-only ABBA using the same
matrix and five-percent promotion rule.

- [ ] **Step 7: record the B decision before starting priority 3**

Post the offline metrics and, when eligible, runtime-pilot result to Issue #51.

---

### Task 7: Implement and run the priority-3 Q2 NAX/grouping premise

**Files:**

- Modify: `mtplx/benchmarks/issue51.py`
- Modify: `scripts/run_issue51_hy3_q2.py`
- Modify: `scripts/summarize_issue51_hy3_q2.py`
- Modify: `tests/test_issue51.py`
- Conditional create after a positive gate: `mtplx/q2_nax.py`
- Conditional test after a positive gate: `tests/test_q2_nax.py`

**Security flag:** `none`

**Does NOT cover:** Treating different expert RHS matrices as one NAX tile,
using synthetic G4/G8 as promotion evidence, enabling NAX by default, or
delaying A1/B work.

- [ ] **Step 1: write failing G1/G2/G3 and premise tests**

For row-major top-k IDs, require:

```python
assert expert_group_histogram([1, 2, 3, 4, 1, 5, 6, 7], rows=2, top_k=4) == {
    1: 6,
    2: 1,
    3: 0,
}
```

Add K=0/1/2 cases, mixed repeated experts, all-unique routes, all-three-row
repeats, invalid widths, duplicate expert IDs within one row, and a premise
case where a synthetic kernel win is rejected after grouping/scatter/model
dilution falls below five percent.

- [ ] **Step 2: run RED**

```bash
.venv/bin/python -m pytest -q tests/test_issue51.py \
  -k 'group_histogram or q2_operator or nax_premise'
```

- [ ] **Step 3: implement route-reuse and operator-headroom evidence**

Report G1/G2/G3 frequency per layer and context/depth, repeated-expert
fraction, stock gathered Q2 gate/up/down time, direct per-expert Q2 time at
observed group sizes, grouping/scatter/host time, zero-arithmetic stream floor,
effective bandwidth, and the model-wide upper bound. K=0 is the G1 control;
K=1/K=2 measure additional reuse. G4/G8 may be diagnostic only.

- [ ] **Step 4: run GREEN and commit the premise tooling**

```bash
.venv/bin/python -m pytest -q tests/test_issue51.py
.venv/bin/ruff check mtplx/benchmarks/issue51.py \
  scripts/run_issue51_hy3_q2.py scripts/summarize_issue51_hy3_q2.py \
  tests/test_issue51.py
git diff --check
git add mtplx/benchmarks/issue51.py scripts/run_issue51_hy3_q2.py \
  scripts/summarize_issue51_hy3_q2.py tests/test_issue51.py
git commit -m "feat(bench): gate issue 51 Q2 NAX premise"
```

- [ ] **Step 5: run the guarded A2 premise campaign**

```bash
.venv/bin/python scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --lock-timeout-seconds 1800 \
  -- \
  .venv/bin/python scripts/run_issue51_hy3_q2.py q2-nax-premise \
  --contexts 1024,2048 --depths 0,1,2 --output-tokens 128 \
  --group-sizes 1,2,3 \
  --output-dir benchmarks/raw/issue51/2026-07-14-q2-nax-premise
```

- [ ] **Step 6: apply the kernel branch exactly**

If the lower confidence bound on addressable operator time is non-positive or
the full measured gap projects below five-percent decode TPS after all costs,
record A2 as `premise_no_go` and create no kernel. If both gates pass, add a
fail-closed `mtplx/q2_nax.py` path restricted to G17/M5, supported macOS,
affine bits=2/group-size=64, BF16 activation/scales/biases, exact Hy3
gate/up/down shapes, pinned component-bank generations, and complete
non-overlapping scatter positions. Write RED/GREEN parity and eligibility tests
before kernel code, then compare stock versus Q2 NAX at K=0/1/2 and the fixed
1,024/2,048 by 128-output matrix. Keep it off by default.

- [ ] **Step 7: test the combination only after two individual wins**

Run compiled-plus-NAX only if both A1 and A2 independently pass. Compare it
against each winner, not just stock, and reject it unless its paired lower
interval is positive with all correctness gates intact.

---

### Task 8: Curate evidence, run full verification, and update Issue #51

**Files:**

- Create: `benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.md`
- Create: `benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.json`
- Update: GitHub Issue #51

**Security flag:** `none`

**Does NOT cover:** Publishing a PR, changing serving defaults, merging the
branch, or representing a gated-out conditional implementation as completed.

- [ ] **Step 1: render and verify the curated artifacts**

```bash
.venv/bin/python scripts/summarize_issue51_hy3_q2.py \
  --a1 benchmarks/raw/issue51/2026-07-14-a1/index.json \
  --predictor benchmarks/raw/issue51/2026-07-14-predictor/index.json \
  --q2-nax benchmarks/raw/issue51/2026-07-14-q2-nax-premise/index.json \
  --output-json benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.json \
  --output-md benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.md
```

- [ ] **Step 2: run the focused and broad verification suites**

```bash
.venv/bin/python -m pytest -q \
  tests/test_issue51.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py \
  tests/test_generation_sustained.py \
  tests/test_expert_slots_runtime.py \
  tests/test_graphbank_compiled_verify.py
.venv/bin/ruff check mtplx scripts tests
.venv/bin/ruff format --check mtplx scripts tests
git diff --check
git status --short --branch
```

- [ ] **Step 3: commit only curated evidence and implementation**

```bash
git add mtplx scripts tests docs/specs docs/plans \
  benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.json \
  benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.md
git commit -m "bench: conclude issue 51 Hy3 Q2 MTP look-ahead"
```

- [ ] **Step 4: update Issue #51**

Post the exact branch/commit, artifact identities, 1,024/2,048 by 128-output
tables, A1/B/A2 decisions, K=0/1/2 NAX control, Qwen restoration receipt, and
all rejected conditional branches. Do not open a PR or change defaults without
a separate user instruction.

---

## Plan self-review checklist

- [ ] The corrected priority is compiled verifier, predictor, then Q2 NAX.
- [ ] K=1 is measured against K=0 at 1,024/2,048 by 128 output before K=2.
- [ ] K=2 requires a passing K=1 speed and utilization summary; K>=3 is rejected.
- [ ] Persistent MTP cache/history/rollback/commit and terminal offsets gate every row.
- [ ] Every behavior change begins with a failing test and observed RED.
- [ ] Every optimization is measured independently before combinations.
- [ ] Qwen unload/restore and exclusive-lane ownership wrap hardware only.
- [ ] Raw evidence remains ignored; curated summaries carry provenance.
- [ ] Negative gates terminate their conditional production work cleanly.
- [ ] No default, PR, merge, or publication action is implied.
