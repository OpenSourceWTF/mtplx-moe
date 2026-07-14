# Benchmark Partial Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every completed Q2/BF16 benchmark row and the active failure context in an atomic partial result.

**Architecture:** `run_depth_matrix` owns a mutable campaign snapshot and optionally emits deep-copied checkpoints at stable boundaries. `main` binds checkpoints to the existing atomic JSON writer when `--output-json` is supplied, persists a final failed snapshot before returning 1, and preserves stdout-only behavior when no path is supplied.

**Tech Stack:** Python 3.11+, pytest, JSON, `os.replace` atomic files.

**Assumptions:** The payload is small enough to rewrite after each retained row; this does not add JSONL replay or per-cell fragments. Caught Python exceptions can persist failure metadata; SIGKILL and power loss preserve only the latest `status: "running"` checkpoint.

---

## File structure

- `scripts/benchmark_q2_mtp_depth_matrix.py`: campaign state, active-cell tracking, callback, atomic persistence, and failure payload.
- `tests/test_benchmark_q2_mtp_depth_matrix.py`: fake-runtime callback and CLI persistence regressions.
- `docs/specs/2026-07-14-benchmark-partial-checkpoints-design.md`: approved behavior contract.

### Task 1: Add incremental campaign checkpoints

**Files:**
- Modify: `scripts/benchmark_q2_mtp_depth_matrix.py`
- Test: `tests/test_benchmark_q2_mtp_depth_matrix.py`

**Security flag:** none

**Does NOT cover:** Warm-up rows remain discarded; a row failing validation is not appended; callers omitting the callback remain in-memory only.

- [x] **Step 1: Write the failing callback test**

Pass `checkpoint=snapshots.append` for one Hy3 context and assert the first snapshot is running and failed-closed, the first one-row snapshot contains only AR, the final payload is passed with no active cell, and mutating one received snapshot cannot affect a later snapshot.

```python
assert snapshots[0]["status"] == "running"
assert snapshots[0]["passed"] is False
one_row = next(s for s in snapshots if len(s["models"][0]["observations"]) == 1)
assert one_row["models"][0]["observations"][0]["cell"] == "ar"
assert payload["status"] == "passed"
assert payload["active_cell"] is None
```

- [x] **Step 2: Run the test and observe the expected failure**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py -k checkpoint_callback
```

Expected: FAIL because `run_depth_matrix` lacks `checkpoint` and status fields.

- [x] **Step 3: Implement stable checkpoint boundaries**

Add a `Checkpoint = Callable[[Mapping[str, Any]], None]` alias and an emitter that passes `copy.deepcopy(payload)`. Construct the top-level payload before allocation with `status="running"`, `passed=False`, `active_cell`, `failure=None`, configuration, and models. Create an in-progress model payload before load; update `active_cell` before load, prompt, warm-up, and retained generation; append and emit only after a retained row validates. Mark models passed only after all cells finish. Final success sets `status="passed"`, `passed=True`, and clears `active_cell`.

- [x] **Step 4: Run all focused runner tests**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py
```

Expected: PASS with existing payload fields and fake call counts unchanged.

### Task 2: Persist partial failure payloads atomically

**Files:**
- Modify: `scripts/benchmark_q2_mtp_depth_matrix.py`
- Test: `tests/test_benchmark_q2_mtp_depth_matrix.py`

**Security flag:** none

**Does NOT cover:** Automatic resume, signal handlers, partial stdout streaming, or recovery from a checkpoint write failure.

- [x] **Step 1: Write failing CLI persistence tests**

Run `main` with `--output-json`, a fake Hy3 runtime, and `mismatch_depth=1`. Assert nonzero exit, failed status, the completed AR row, a D1 retained active cell, structured error metadata, and stdout equality with the saved JSON. Add a loader-failure output-path case with zero observations, `active_cell.phase == "load"`, and no temporary file left behind.

```python
assert exit_code == 1
assert saved["status"] == "failed"
assert [row["cell"] for row in saved["models"][0]["observations"]] == ["ar"]
assert saved["failure"]["active_cell"]["cell"] == "d1"
assert saved["failure"]["active_cell"]["phase"] == "retained"
```

- [x] **Step 2: Run the tests and observe the expected failure**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py -k 'partial_failure or loader_failure_checkpoint'
```

Expected: FAIL because the exception path prints only a four-field object and does not write the output path.

- [x] **Step 3: Bind callback persistence and the failed terminal snapshot**

In `main`, retain the latest deep-copied checkpoint and atomically write each
checkpoint when an output path exists. The runner owns the terminal exception
snapshot: it sets `status="failed"`, keeps `passed=False`, records the live
active cell, emits, and re-raises. `main` atomically persists and prints that
same payload, then returns 1. A checkpoint serialization/write error aborts
instead of letting the benchmark continue.

- [x] **Step 4: Verify behavior and code quality**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py
.venv/bin/ruff check scripts/benchmark_q2_mtp_depth_matrix.py tests/test_benchmark_q2_mtp_depth_matrix.py
.venv/bin/ruff format --check scripts/benchmark_q2_mtp_depth_matrix.py tests/test_benchmark_q2_mtp_depth_matrix.py
git diff --check
```

Expected: all commands exit 0; failed fake runs retain AR and identify D1.

### Task 3: Verify integration without using the hardware lane

**Files:**
- Verify: `scripts/benchmark_q2_mtp_depth_matrix.py`
- Verify: `tests/test_benchmark_q2_mtp_depth_matrix.py`

**Security flag:** none

- [x] **Step 1: Run the benchmark-focused suite and compile check**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_q2_mtp_depth_matrix.py tests/test_streamed_models.py
.venv/bin/python -m py_compile scripts/benchmark_q2_mtp_depth_matrix.py
```

Expected: all tests pass and compilation exits 0.

- [x] **Step 2: Inspect final scope**

```bash
git status --short
git diff -- scripts/benchmark_q2_mtp_depth_matrix.py tests/test_benchmark_q2_mtp_depth_matrix.py docs/specs/2026-07-14-benchmark-partial-checkpoints-design.md docs/plans/2026-07-14-benchmark-partial-checkpoints.md
git diff --check
```

Expected: only the runner/tests, spec, plan, and already-present MTP-disabled baseline edits are changed; no model files or benchmark artifacts are added.
