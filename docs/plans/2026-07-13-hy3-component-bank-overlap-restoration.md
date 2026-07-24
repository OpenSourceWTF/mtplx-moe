# Hy3 Component-Bank Overlap Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unsupported component-bank-wide miss barrier, lock the original per-slot overlap behavior with a regression test, and correct PR #35 so its no-go conclusion rests only on the measured performance gate.

**Architecture:** Keep the existing slot-generation, pin, completion-fence, and per-miss-future ownership model. Component-bank hit rows execute while disjoint miss rows load, and each miss row executes only after its own future completes. The serialized `finish_misses()` control remains historical benchmark evidence but is not a production requirement or a basis for invalidating the original measurements.

**Tech Stack:** Python 3.12, MLX 0.31, pytest, Ruff, JSON, GitHub CLI.

**Assumptions:**

- Assumes the existing per-slot pins, generation validation, and completion fences correctly prevent same-slot overwrite while Metal consumes that slot; this change does not weaken those mechanisms.
- Assumes no evidence currently demonstrates corruption from CPU writes and Metal reads targeting disjoint component-bank rows; this plan will not claim that absence of observed failure proves a universal Metal guarantee.
- Assumes PR #35 remains a no-go because the original repeated B1 gain was 4.858%, below the declared 5% gate; this plan does not promote global caching.
- Assumes the checked-out `experiment/hy3-cache-scheduling` worktree is the PR #35 head and is clean; this plan does not modify the production `direct-slots` default.

---

## File Structure

- `tests/test_streamed_models.py`: regression coverage for component-bank hit/shared/Miss overlap and per-part release.
- `mtplx/models/expert_mlx.py`: remove only the component-bank-wide aggregate miss barrier and restore the common incremental route path.
- `benchmarks/results/hy3-cache-scheduling-issue29-20260713.md`: restore the original benchmark payload as promotion evidence and describe the serialized run as a rejected premise control.
- `benchmarks/results/hy3-cache-scheduling-issue29-20260713.json`: make the same evidence-status correction in machine-readable form.
- GitHub PR #35 body/comment: retract the unsupported safety claim and retain the performance-only no-go.

### Task 1: Lock the Overlap Contract and Remove the Barrier

**Files:**
- Modify: `tests/test_streamed_models.py`
- Modify: `mtplx/models/expert_mlx.py`

**Security flag:** none

**Does NOT cover:** This restores overlap only for disjoint logical slots that remain protected by the existing generation, pin, and completion-fence lifecycle. It does not permit a slot to be overwritten while its own Metal work is in flight.

- [x] **Step 1: Write a failing component-bank overlap test**

Replace the barrier-specific fixture/tests with a fixture whose `finish_misses()` raises if called and a regression test with this required event order:

```python
class _BankOverlapPending:
    plan = SimpleNamespace(hits=(0,), misses=(1, 2))
    misses_pending = True

    def __init__(self, events: list[str]) -> None:
        self.events = events
        bank = object()

        def binding(expert: int) -> SimpleNamespace:
            return SimpleNamespace(expert=expert, buffer=SimpleNamespace(bank=bank))

        self.hit_ready = SimpleNamespace(bindings=(binding(0),))
        self.miss_parts = (
            SimpleNamespace(bindings=(binding(1),), plan=SimpleNamespace(experts=(1,))),
            SimpleNamespace(bindings=(binding(2),), plan=SimpleNamespace(experts=(2,))),
        )

    def finish_misses(self):
        raise AssertionError("component-bank overlap must not aggregate all misses")

    def iter_ready_misses(self):
        for part in self.miss_parts:
            self.events.append(f"ready:{part.plan.experts}")
            yield part

    def release_hits(self) -> None:
        self.events.append("release-hits")

    def release_miss(self, part) -> None:
        self.events.append(f"release-miss:{part.plan.experts}")

    def abort(self, error: BaseException) -> None:
        self.events.append(f"abort:{type(error).__name__}")

    def close(self) -> None:
        self.events.append("close")
```

```python
def test_component_bank_overlaps_hit_and_shared_work_with_incremental_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    pending = _BankOverlapPending(events)
    runtime = _BankOverlapRuntime(events, pending)
    switch = HotExpertSwitchGLU(runtime, 1)

    def fake_q4(selected, bindings, *, group_size):
        assert group_size == 64
        events.append(f"q4:{tuple(item.expert for item in bindings)}")
        return selected

    monkeypatch.setattr(expert_mlx, "_run_component_bank_q4", fake_q4)

    def shared_work():
        events.append("shared")
        return mx.ones((3, 1, 2), dtype=mx.bfloat16)

    routed, shared = switch.run_with_shared_overlap(
        mx.zeros((3, 1, 2), dtype=mx.bfloat16),
        mx.array([[[0]], [[1]], [[2]]], dtype=mx.int32),
        shared_work,
    )
    mx.eval(routed, shared)

    assert events == [
        "begin-misses",
        "q4:(0,)",
        "release-hits",
        "shared",
        "ready:(1,)",
        "q4:(1,)",
        "release-miss:(1,)",
        "ready:(2,)",
        "q4:(2,)",
        "release-miss:(2,)",
        "close",
    ]
```

- [x] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_streamed_models.py::test_component_bank_overlaps_hit_and_shared_work_with_incremental_misses
```

Expected: FAIL because the current runtime calls `pending.finish_misses()` before hit Q4.

- [x] **Step 3: Restore the incremental route path**

In `HotExpertSwitchGLU.run_with_shared_overlap`, remove `component_bank_layout`, `aggregate_misses`, and the component-bank-only `finish_misses()`/aggregate-release branch. Restore the common control flow:

```python
pending = self.runtime.begin_split_route(
    self.layer_index,
    wave.experts,
    phase=phase,
)
try:
    hit_set = set(pending.plan.hits)
    # Existing hit evaluation and shared-work overlap remain unchanged.
    for miss_ready in pending.iter_ready_misses():
        part_error: BaseException | None = None
        try:
            ready_experts = set(miss_ready.plan.experts)
            miss_positions = tuple(
                position
                for position, expert in zip(
                    wave.positions, wave.experts, strict=True
                )
                if expert not in hit_set and expert in ready_experts
            )
            evaluate_bindings(
                miss_positions,
                miss_ready.bindings,
                miss_ready,
                force_sync=True,
            )
        except BaseException as exc:
            part_error = exc
            raise
        finally:
            try:
                pending.release_miss(miss_ready)
            except BaseException:
                if part_error is None:
                    raise
except BaseException as exc:
    pending.abort(exc)
    raise
finally:
    pending.close()
```

- [x] **Step 4: Verify GREEN and the surrounding streamed-model tests**

Run:

```bash
uv run --frozen --extra dev --extra server pytest -q tests/test_streamed_models.py
```

Expected: PASS, including the new overlap ordering test and existing pin/fence lifecycle tests.

### Task 2: Correct the Curated Evidence

**Files:**
- Modify: `benchmarks/results/hy3-cache-scheduling-issue29-20260713.md`
- Modify: `benchmarks/results/hy3-cache-scheduling-issue29-20260713.json`

**Security flag:** none

- [x] **Step 1: Rewrite the Markdown decision and serialized-control section**

The report must state all of the following explicitly:

```markdown
**No-go for promoting the global component-bank cache on performance.** The repeated B1 mean gain is 4.858%, below the issue's 5% gate; the second paired run reaches only 4.674%. Static B2/B4/B8 gains are 2.02%, 3.06%, and 2.86%; the mixed prefill/decode B4 gain is 1.26%.
```

```markdown
## Rejected serialized-control premise

Commit `058b40b` tested a conservative schedule that drained every component-bank miss before any Q4 dispatch. It was motivated by a hypothesis that CPU writes to one logical row required exclusion from Metal reads of disjoint sibling rows in the same MLX allocation.

The experiment did not reproduce corruption, stale reads, an integrity failure, or an attributable crash, and the existing per-slot path already prevents same-slot replacement until its Metal consumer completes. The resource-wide exclusion premise therefore remains unproven. The barrier is removed rather than promoted, and the original exact-parity benchmark payload remains the promotion evidence for this cache-policy experiment.

The serialized control is retained only to document its cost: 2.9545 tok/s mean, 51.587% below the original layer mean, despite exact output parity. It does not establish that the original schedule was unsafe.
```

Remove or replace every statement that calls the original payload `unsafe`, `diagnostic only`, or `superseded`.

- [x] **Step 2: Make the JSON semantics match**

Rename top-level `safety_commit` to `serialized_control_commit`. Replace `safety_correction` with:

```json
"serialized_control": {
  "premise": "CPU writes to one component-bank row require exclusion from Metal reads of disjoint sibling rows in the same MLX allocation.",
  "premise_status": "unproven; no reproduced corruption, stale read, integrity failure, or attributable crash",
  "schedule": "Finish every component-bank miss before any Q4 dispatch from the bank.",
  "original_payload_status": "primary promotion evidence with exact deterministic output parity and no observed integrity failure",
  "control": {
    "scope": "layer",
    "run": "issue29-fenced-layer-B1-p01-20260713T073735Z-058b40b28de9",
    "mean_tokens_per_second": 2.9544528863828594,
    "change_from_original_layer_mean_percent": -51.587469360692495
  },
  "decision": "reject the serialized barrier; it measures cost but does not prove the original schedule unsafe"
}
```

Preserve the full existing repeat details under `control`; do not discard measured data. Update `decision.reason` so it cites only the missed 5% performance gate. Update verification wording from “safety commit” to “serialized-control commit.”

- [x] **Step 3: Validate curated evidence**

Run:

```bash
jq empty benchmarks/results/hy3-cache-scheduling-issue29-20260713.json
rg -n -i 'unsafe|diagnostic only|diagnostic-only|supersed' \
  benchmarks/results/hy3-cache-scheduling-issue29-20260713.md \
  benchmarks/results/hy3-cache-scheduling-issue29-20260713.json
```

Expected: `jq` exits 0; `rg` finds no statement invalidating the original throughput payload. The held-out quota analysis may continue to call its own result diagnostic because that is unrelated to the removed safety claim.

### Task 3: Verify the Branch

**Files:**
- Verify all files changed relative to `origin/experiment/moe-pr13-pr14-stack`.

**Security flag:** none

- [x] **Step 1: Run focused and full tests**

```bash
uv run --frozen --extra dev --extra server pytest -q tests/test_streamed_models.py
uv run --frozen --extra dev --extra server pytest -q
```

Expected: both commands exit 0 with only existing skips/deprecation warnings.

- [x] **Step 2: Run changed-file lint, format, JSON, and diff checks**

```bash
files=($(git diff --name-only origin/experiment/moe-pr13-pr14-stack...HEAD -- '*.py'))
uv run --frozen --extra dev --extra server ruff check $files
uv run --frozen --extra dev --extra server ruff format --check $files
jq empty benchmarks/results/hy3-cache-scheduling-issue29-20260713.json
git diff --check
```

Expected: every command exits 0.

- [x] **Step 3: Confirm scope**

```bash
git diff --stat
git diff -- mtplx/models/expert_mlx.py tests/test_streamed_models.py \
  benchmarks/results/hy3-cache-scheduling-issue29-20260713.md \
  benchmarks/results/hy3-cache-scheduling-issue29-20260713.json
```

Expected: runtime changes only remove the aggregate barrier; tests preserve per-slot overlap; evidence retains the performance no-go without an unsupported safety claim.

### Task 4: Publish and Correct PR #35

**Files:**
- Commit the verified branch.
- Update: `https://github.com/davidtai/MTPLX/pull/35`

**Security flag:** none

- [ ] **Step 1: Commit and push**

```bash
git add \
  mtplx/models/expert_mlx.py \
  tests/test_streamed_models.py \
  benchmarks/results/hy3-cache-scheduling-issue29-20260713.md \
  benchmarks/results/hy3-cache-scheduling-issue29-20260713.json \
  docs/plans/2026-07-13-hy3-component-bank-overlap-restoration.md
git commit -m "fix(hy3): restore component-bank overlap evidence"
git push origin experiment/hy3-cache-scheduling
```

Expected: push succeeds and PR #35 head advances to the new commit.

- [ ] **Step 2: Replace the PR body**

The updated body must retain the measured cache economics and verification commands, remove every claim that the original path was unsafe or diagnostic-only, state that `058b40b` was a rejected serialized premise control, and keep the decision as performance-only no-go.

Verify after the write:

```bash
gh pr view 35 -R davidtai/MTPLX --json body,headRefOid,url
```

Expected: body contains `no-go on performance`, `resource-wide exclusion premise remains unproven`, and no unsupported safety claim.

- [ ] **Step 3: Add a transparent correction comment**

Post this correction, substituting the actual new short SHA from `git rev-parse --short HEAD`:

```markdown
Correction: commit `058b40b` was a conservative serialized control based on an unproven resource-wide exclusion premise. We did not reproduce corruption, a stale read, an integrity failure, or an attributable crash, and the production default remains `direct-slots`. The barrier has been removed and the original exact-parity measurements are restored as the promotion evidence. The global cache remains a no-go solely because its repeated B1 gain was 4.858%, below the declared 5% gate. The correction is published in the current PR head.
```

Verify after the write:

```bash
gh pr view 35 -R davidtai/MTPLX --json comments --jq '.comments[-1].body'
```

Expected: the latest comment is the correction above.
