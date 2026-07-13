# Hy3 Cache Economics and Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing cross-layer global cache usable with component-bank Q4 execution, then measure physical-byte and B=1/2/4/8 behavior against the retained stack base.
**Architecture:** Keep `GlobalExpertSlotBank` as the only dynamic allocator and teach `MlxComponentBank` to represent one global persistent tier. The runtime continues to own policy, generation, pin, and I/O transactions; the allocator only maps logical slots to component rows. Reuse the existing held-out route analyzer rather than adding a second oracle.
**Tech Stack:** Python 3.12, MLX 0.31, pytest, Ruff, Git worktrees, GitHub draft PRs.
**Assumptions:** Assumes every routed layer in a model spec has identical expert component geometry; this will NOT work for a model whose routed layers change hidden, intermediate, quantization, or component shape. Assumes a global transaction lock is acceptable for the first measured arm; this does NOT claim concurrent cross-layer mutation.

## Pre-change baseline

- [x] Rebase onto `experiment/moe-pr13-pr14-stack@f3a812e` before implementation.
- [x] Run the existing 2,048-token deterministic layer/component gate twice on the exclusive M5 Max lane.
- [x] Confirm deterministic output and less than 5% run-to-run variance before editing production code.

Observed baseline: both runs stopped naturally after the same 1,905 generated tokens with identical token hashes, cache counters, physical reads, and peak MLX memory. End-to-end throughput was 5.992 and 6.048 tok/s (0.93% variance); the retained raw payloads are under `benchmarks/raw/moe-runtime/hy3-cache-scheduling-base-r0*`.

---

## Files

- Modify `mtplx/expert_runtime.py`: admit global component banks and expose global all-hit transactions.
- Modify `mtplx/models/expert_mlx.py`: allocate one global persistent component bank.
- Modify `mtplx/expert_streaming.py`: add a side-effect-safe global all-hit transaction.
- Modify `scripts/benchmark_streamed_generation.py`: label global-component and saturation lanes explicitly.
- Modify `scripts/analyze_expert_route_trace.py`: emit a reusable held-out capacity recommendation with byte totals.
- Test `tests/test_expert_streaming.py`, `tests/test_expert_slots_runtime.py`, `tests/test_streamed_models.py`, `tests/test_analyze_expert_route_trace.py`, and benchmark CLI tests.
- Add benchmark results under `benchmarks/results/` only after an exclusive-machine run.

### Task 1: Admit and allocate global component banks

**Security flag:** none
**Does NOT cover:** Metal execution or all-hit bypass; it only makes the existing global slot transaction representable by component-major storage.

- [x] **Step 1: Write failing configuration and allocator tests**

Add tests equivalent to:

```python
def test_global_component_bank_config_is_valid(tiny_config):
    config = replace(tiny_config, cache_scope="global", slot_layout="component-banks")
    assert config.cache_scope == "global"

def test_global_component_allocator_uses_one_persistent_bank(plan, spec, manifest):
    allocator = make_mlx_component_bank_allocator(plan, spec, manifest)
    left = allocator(spec.expert_record_bytes, "global-persistent-0")
    right = allocator(spec.expert_record_bytes, "global-persistent-1")
    assert left.bank is right.bank
    assert left.bank.capacity == plan.persistent_slots
```

- [x] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_slots_runtime.py tests/test_streamed_models.py \
  -k 'global and component'
```

Expected: configuration rejects the combination or the allocator rejects `global-persistent-*`.

- [x] **Step 3: Implement the global persistent bank**

Remove only the `global + component-banks` rejection from `ExpertStreamingConfig`. In `make_mlx_component_bank_allocator`, accept `global-persistent-N`, create a bank keyed as `("global-persistent", -1)`, use `plan.persistent_slots`, and use an exemplar record only after proving every routed layer has the same `(component, dtype, shape, length)` signature.

- [x] **Step 4: Verify GREEN and memory accounting**

Run the target tests plus:

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_streaming_models.py tests/test_expert_slots_runtime.py \
  tests/test_streamed_models.py
```

Expected: pass; allocated bytes equal `plan.persistent_cache_bytes + plan.transient_bytes`.

- [x] **Step 5: Commit**

```bash
git add mtplx/expert_runtime.py mtplx/models/expert_mlx.py \
  tests/test_expert_slots_runtime.py tests/test_streamed_models.py
git commit -m "perf(hy3): enable global component-bank cache"
```

### Task 2: Add a safe global all-hit transaction

**Security flag:** none
**Does NOT cover:** miss handling or concurrent global route mutation; misses retain the existing split-route transaction.

- [x] **Step 1: Write failing policy and runtime tests**

Test that a fully resident global route returns bindings without loads, preserves duplicate assignment order, updates LRU history only on commit, and restores state if pinning fails.

```python
planned = bank.try_plan_all_hits_transaction(2, [3, 1, 3], phase="decode")
assert planned is not None
plan, txn = planned
assert plan.experts == (3, 1, 3)
assert plan.loads == ()
txn.rollback_completion()
assert bank.snapshot() == before
```

- [x] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_streaming.py tests/test_expert_slots_runtime.py \
  -k 'global_all_hit'
```

Expected: `GlobalExpertSlotBank` has no safe all-hit transaction.

- [x] **Step 3: Implement the transaction**

Add `GlobalExpertSlotBank.try_plan_all_hits_transaction(layer, expert_ids, phase)` using the same snapshot/commit/rollback contract as `LayerExpertSlotBank`. Update `ExpertStreamingRuntime.try_all_hit_route` to select the global or layer-local transaction instead of returning `None` for global cache scope.

- [x] **Step 4: Verify GREEN and fault paths**

Run target tests and the slot-generation/fence tests in `tests/test_expert_slots_runtime.py`.

- [x] **Step 5: Commit**

```bash
git add mtplx/expert_streaming.py mtplx/expert_runtime.py \
  tests/test_expert_streaming.py tests/test_expert_slots_runtime.py
git commit -m "perf(hy3): fast-path global all-hit routes"
```

### Task 3: Export held-out capacity and saturation evidence

**Security flag:** none
**Does NOT cover:** automatically changing a production cache allocation from the same trace being evaluated.

- [ ] **Step 1: Write failing analyzer and CLI tests**

Require the analyzer payload to include `recommended_capacity` with per-layer quotas, total slots, total bytes, training gain, and held-out delta. Require benchmark configuration labels to encode cache scope, slot layout, and concurrency.

- [ ] **Step 2: Verify RED**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_analyze_expert_route_trace.py \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_benchmark_streamed_generation_concurrency_cli.py
```

- [ ] **Step 3: Implement export and labels**

Promote the already computed train-only quota data into a stable JSON object. Do not recompute it from evaluation rows. Add explicit `cache_scope`, `slot_layout`, and `concurrency` fields to benchmark labels and summaries.

- [ ] **Step 4: Verify GREEN**

Run the same tests and deterministic JSON snapshot assertions.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_expert_route_trace.py scripts/benchmark_streamed_generation.py \
  tests/test_analyze_expert_route_trace.py \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_benchmark_streamed_generation_concurrency_cli.py
git commit -m "bench(hy3): export held-out cache and batch evidence"
```

### Task 4: Verify and benchmark PR 1

**Security flag:** none
**Does NOT cover:** kernel, serialization, MTP, KV-precision, or lower-bit claims.

- [ ] Run full pytest: `uv run --frozen --extra dev --extra server pytest -q`.
- [ ] Run Ruff on changed Python files only.
- [ ] Stop the Qwen MLX server only for the benchmark window and record its exact restart command.
- [ ] Run two or more exclusive long AR baselines for layer/component and global/component arms with identical cache bytes.
- [ ] Run B=1/2/4/8 using `scripts/benchmark_streamed_generation.py --concurrency N`.
- [ ] Reject global component banks if repeated results miss the 5% target or materially regress single-stream latency.
- [ ] Save raw JSON, a concise Markdown comparison, and the exact commands.
- [ ] Restart Qwen and verify `/v1/models` returns the expected model.
- [ ] Push `experiment/hy3-cache-scheduling` and open a draft PR against `experiment/moe-pr13-pr14-stack`, linking #29.
