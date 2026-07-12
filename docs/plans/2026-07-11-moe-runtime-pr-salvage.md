# MoE Runtime PR Salvage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair and sequentially gate the useful work from PRs #15, #12, #17, #14, #18, #16, and #11 on top of retained PR #13, advancing draft PR #20 only with passing candidates.

**Architecture:** The integration branch is a monotonic chain: each source commit is transplanted onto the latest retained tip, repaired with RED-GREEN tests, verified, and compared against that immediately preceding tip. The existing cache planner, bounded route waves, slot/pin ownership, PR #13 shared-MLP overlap, and deterministic Hy3 runner remain authoritative. Benchmark-only instrumentation is applied in a throwaway overlay and never becomes an unreviewed runtime dependency.

**Tech Stack:** Python 3.12, MLX, pytest, Ruff, Git worktrees/cherry-pick, `scripts/benchmark_streamed_generation.py`, macOS launchd, the repository's SSD/memory runner hooks.

**Assumptions:**

- Assumes the pinned Hy3 snapshot and sidecar remain at revision `160619d3f96c8470350b6dac0ef033a8381551e3` — results are not comparable if model bytes change.
- Assumes PR #13 at `939fe57` remains the retained runtime baseline — this plan does not re-evaluate or remove PR #13.
- Assumes Qwen can be stopped with `com.tea.qwen.plist` while `com.tea.qwen-gateway` remains loaded — GPU results are invalid if the Qwen model process is resident.
- Assumes each source SHA names the reviewed candidate — this plan never imports `bbe0b0d`, `05f2f13`, or a precomposed local stack.
- Assumes every later task starts from the latest retained tip — a skipped candidate contributes no production code to later candidates.

---

## File structure

- `mtplx/expert_streaming.py`: cache-policy validation, route planning, epochs, hotsets.
- `mtplx/expert_runtime.py`: runtime orchestration, route transactions, cancellation, close/snapshot state.
- `mtplx/expert_slots.py`: slot, pin, fence, projection, and storage lifetime ownership.
- `mtplx/models/expert_mlx.py`: component-bank execution and shared-work overlap.
- `mtplx/expert_streaming_models.py`: physical memory planning.
- `mtplx/runtime.py`, `mtplx/generation.py`: cache attestation and request-boundary ownership.
- `mtplx/expert_io.py`: staged component reads used by PR #18.
- `scripts/benchmark_streamed_generation.py`: deterministic hardware lane and serialized runner-hook fields.
- `tests/test_expert_streaming.py`: policy and route-wave regressions.
- `tests/test_expert_slots_runtime.py`: lifecycle, cancellation, pin, fence, and close regressions.
- `tests/test_streamed_models.py`: MLX execution, ordering, parity, and error cleanup.
- `tests/test_expert_streaming_models.py`, `tests/test_cache_state.py`, `tests/test_generation_sustained.py`, `tests/test_hy3_streamed_mtp.py`: memory and request-boundary contracts.
- `docs/MOE_RUNTIME_PR_BENCHMARKS.md`, `benchmarks/results/moe-runtime-gate-matrix.md`: decisions and exact evidence.
- `benchmarks/results/`: committed timing and instrumented raw artifacts.

## Shared gate commands

Use this exact timing lane for every ordinary Hy3 pair; substitute only `LABEL`, `OUTPUT_JSON`, and candidate-specific flags:

```bash
MODEL_ROOT=/Users/davidtai/.cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots/160619d3f96c8470350b6dac0ef033a8381551e3
MANIFEST="$MODEL_ROOT/expert-manifest-sidecar.json"
uv run python scripts/benchmark_streamed_generation.py \
  "$MODEL_ROOT" "$MANIFEST" \
  --model-key hy3-q4 \
  --memory-limit 112GiB \
  --max-live-kv-tokens 4096 \
  --runtime-reserve 8GiB \
  --expert-cache-limit 78GiB \
  --cache-policy lru \
  --cache-scope layer \
  --prompt-file benchmarks/prompts/moe_streaming_realistic.md \
  --chat --no-enable-thinking \
  --generation-profile deterministic \
  --max-tokens 2048 --no-window-telemetry \
  --repeats 1 --seed 0 --concurrency 1 --max-prefills-per-step 1 \
  --transient-slots 32 --read-chunk 64MiB \
  --f-nocache --slot-layout component-banks --trust-sidecar \
  --output-dir benchmarks/results \
  --run-label "$LABEL" --output-json "$OUTPUT_JSON"
```

Before the first GPU run:

```bash
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tea.qwen.plist"
! pgrep -f 'mtplx.server.openai.*Qwen3.6'
launchctl list | grep -q 'com.tea.qwen-gateway'
! pgrep -f 'benchmark_streamed_generation|probe_mtp|probe_paged' 
```

For each retained candidate, create one throwaway instrumentation worktree from its final SHA, transplant `74e8091` and `e963654`, and run the same lane with:

```bash
--instrument --sample-interval 0.5 \
--ssd-ceiling-gib-s 12.469 --memory-ceiling-gb-s 614
```

Do not use the instrumented replay for headline timing. Run at least three alternating process-isolated base/candidate pairs for a sub-5% effect. Every pair must have identical completion token IDs, token hash, natural stop condition, cache configuration, and runner version.

### Task 1: Repair and gate PR #15 all-hit decode

**Files:**
- Modify: `mtplx/expert_streaming.py`
- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/models/expert_mlx.py`
- Test: `tests/test_expert_streaming.py`
- Test: `tests/test_streamed_models.py`

**Security flag:** `none`

**Does NOT cover:** global-cache layouts, mapped experts, or miss-route acceleration; they continue through the PR #13 path.

- [x] **Step 1: Transplant only the reviewed source**

```bash
git cherry-pick -x 7652fa2
```

Resolve the `_run(..., shared_work=...)` conflict by retaining PR #13's tuple-returning method and inserting the source fast-path logic without importing later merge/fix commits.

- [x] **Step 2: Write failing policy and execution tests**

```python
def test_all_hit_probe_matches_normal_wave_policy_and_next_lru_victims():
    normal = LayerExpertSlotBank(
        expert_count=8, persistent_slots=4, transient_slots=2, cache_policy="lru"
    )
    fast = LayerExpertSlotBank(
        expert_count=8, persistent_slots=4, transient_slots=2, cache_policy="lru"
    )
    for expert in (0, 1, 2, 3):
        normal.plan([expert], phase="decode")
        fast.plan([expert], phase="decode")
    for wave in ((2, 3), (0, 1)):
        expected = normal.plan(wave, phase="decode")
        actual = fast.try_plan_all_hits(wave, phase="decode")
        assert actual == expected
    assert fast.plan([4], phase="decode").evictions == normal.plan([4], phase="decode").evictions
    assert fast.plan([5], phase="decode").evictions == normal.plan([5], phase="decode").evictions

def test_failed_all_hit_probe_is_side_effect_free():
    control = LayerExpertSlotBank(
        expert_count=8, persistent_slots=1, transient_slots=1
    )
    probe = LayerExpertSlotBank(
        expert_count=8, persistent_slots=1, transient_slots=1
    )
    control.plan([0], phase="decode")
    probe.plan([0], phase="decode")
    before = probe.snapshot()
    assert probe.try_plan_all_hits([0, 1], phase="decode") is None
    assert probe.snapshot() == before
    assert probe.plan([1], phase="decode") == control.plan([1], phase="decode")
```

Add `test_component_bank_all_hit_decode_preserves_route_waves_counters_and_shared_order` and `test_component_bank_all_hit_decode_releases_pins_on_q4_error` in `tests/test_streamed_models.py`. The warmed three-unique/two-capacity route must still produce two waves/two route calls, all fast Q4 events must precede `shared`, and both success and injected Q4 failure must end with `pins == 0`.

- [x] **Step 3: Prove RED**

```bash
uv run pytest -q tests/test_expert_streaming.py tests/test_streamed_models.py
```

Expected: the raw source collapses the warmed route into one policy epoch/route call and mishandles PR #13's shared-work return contract.

- [x] **Step 4: Implement the repair**

Keep `try_plan_all_hits()` side-effect-free until all unique IDs are resident. Invoke it inside the existing `for wave in runtime.route_waves(...)` loop, never on the flattened route. On a hit wave, run assignment-aligned component-bank Q4 in `wave.positions` order, `mx.eval()` before releasing the route, append output/positions, and continue. On `None`, execute the unchanged PR #13 split path. Do not early-return from `_run`; call `shared_work()` exactly once after routed work when no miss overlap exists. Skip concatenate/argsort/take only when one output already spans `range(total_assignments)`.

- [x] **Step 5: Verify and commit repair**

```bash
uv run pytest -q tests/test_expert_streaming.py tests/test_streamed_models.py
uv run ruff check mtplx/expert_streaming.py mtplx/expert_runtime.py mtplx/models/expert_mlx.py tests/test_expert_streaming.py tests/test_streamed_models.py
uv run ruff format --check mtplx/expert_streaming.py mtplx/expert_runtime.py mtplx/models/expert_mlx.py tests/test_expert_streaming.py tests/test_streamed_models.py
uv run pytest -q
git diff --check
git add mtplx tests
git commit -m 'fix: preserve route semantics in all-hit decode'
```

- [x] **Step 6: Run the matched gate and record the decision**

Run at least three alternating pairs against the pre-#15 tip plus one instrumented candidate replay. Retain only if token/stop parity holds, route counters and expert bytes/operations are invariant, pins end at zero, and both matched median and mean decode tok/s are positive without systematic reversals. Expand the run if the historical `+0.78%` overlaps noise.

### Task 2: Repair and gate PR #12 Q8 KV accounting

**Files:**
- Modify: `mtplx/expert_cli.py`, `mtplx/expert_runtime.py`, `mtplx/expert_streaming_models.py`, `mtplx/runtime.py`
- Modify: `scripts/plan_expert_memory.py`, `scripts/benchmark_streamed_generation.py`
- Test: `tests/test_cache_state.py`, `tests/test_expert_cli_runtime.py`, `tests/test_expert_slots_runtime.py`, `tests/test_expert_streaming_models.py`, `tests/test_generation_sustained.py`, `tests/test_hy3_streamed_mtp.py`, `tests/test_benchmark_streamed_generation_cli.py`

**Security flag:** `none`

**Does NOT cover:** TurboQuant memory savings or GLM Q8; both must fail closed rather than receive Hy3 Q8 credit.

- [x] **Step 1: Transplant source and add RED tests**

```bash
git cherry-pick -x a8ef882
```

```python
def test_hy3_q8_128k_plan_charges_rounded_capacity_margin():
    gib = 1024**3
    spec = get_model_spec("hy3-q4")
    plan = plan_expert_memory(spec, total_limit_bytes=112 * gib,
                              context_tokens=131_072, runtime_reserve_bytes=8 * gib,
                              transient_slots=32, paged_kv_quantization="q8")
    assert plan.context_tokens == 131_072
    assert plan.kv_capacity_tokens == 131_200
    assert plan.kv_bytes == 21_831_680_000
    assert plan.kv_bytes - 131_072 * 166_400 == 21_299_200

def test_q8_capacity_rounding_can_fail_the_fixed_footprint():
    gib = 1024**3
    spec = get_model_spec("hy3-q4")
    required = spec.resident_bytes + spec.transient_scratch_bytes + 8 * gib + 21_831_680_000
    assert not plan_expert_memory(spec, total_limit_bytes=required - 1,
                                  context_tokens=131_072, runtime_reserve_bytes=8 * gib,
                                  transient_slots=32, paged_kv_quantization="q8").fits_fixed
```

Add tests that reject truthy `MTPLX_VLLM_METAL_PAGED_TURBOQUANT`, skipped/non-Q8 cache installs, wrong converted-layer counts, and a one-byte memory deficit before manifest or allocator access. Extend the benchmark CLI test so `--paged-kv-quantization q8` reaches both `ExpertStreamingConfig` and serialized results.

- [x] **Step 2: Prove RED and repair**

```bash
uv run pytest -q tests/test_expert_streaming_models.py tests/test_expert_cli_runtime.py tests/test_expert_slots_runtime.py tests/test_cache_state.py tests/test_generation_sustained.py tests/test_hy3_streamed_mtp.py tests/test_benchmark_streamed_generation_cli.py
```

Compute `kv_capacity_tokens = 0 if logical == 0 else ceil((logical + 128) / 16) * 16`, charge physical capacity, and serialize the capacity, block size, and margin. Reject TurboQuant precedence. Fail before manifest/model allocation when `fits_fixed` is false. Add `attest_expert_streaming_kv_cache()` and require q8 mode, quantized caches, 80 entries, zero skips, and zero TurboQuant entries after cache construction; persist attestation in runner-visible diagnostics.

- [x] **Step 3: Verify, commit, and gate**

Run focused tests, full pytest, Ruff, and `git diff --check`. Compare a conservative 75-slot BF16-accounted base with repaired attested-Q8/100-slot candidate using the same cap and runner hooks. Require token parity, physical planned/realized byte equality, positive median+mean reduction in expert misses and physical expert bytes/token, and no material throughput/tail-latency/memory regression.

**Result:** correctness and attestation passed at repaired tip `124f4ce`, but the
100-slot candidate failed the measured gate: token-identical decode fell 78.97%,
p95 latency rose 351.5%, and peak MLX rose 20.97 GB. Runner hooks showed lower
SSD traffic but a serialized route/read/compute critical path. PR #12 was not
promoted; the passed-only integration tip remains `fb4c1d5`.

### Task 3: Repair and gate PR #17 completion fences

**Files:**
- Modify: `mtplx/expert_slots.py`, `mtplx/expert_runtime.py`, `mtplx/models/expert_mlx.py`
- Test: `tests/test_expert_slots_runtime.py`, `tests/test_streamed_models.py`

**Security flag:** `none`

**Does NOT cover:** cancellation of sibling read futures; Task 4 owns that behavior.

- [ ] **Step 1: Transplant source and add RED tests**

```bash
git cherry-pick -x d484a94
```

Add controlled-event tests named `test_completion_fence_failure_stops_replacement_already_waiting_on_pin`, `test_completion_fence_failure_is_visible_to_snapshot_and_close`, and `test_runtime_close_timeout_is_retryable_after_route_release`. Assert failed fences never change expert/generation/read bytes, terminal cause remains observable by both snapshot and close, and a timed-out close succeeds after the held route is released.

- [ ] **Step 2: Prove RED and repair**

```bash
uv run pytest -q tests/test_expert_slots_runtime.py -k 'completion_fence or close_timeout'
uv run pytest -q tests/test_streamed_models.py -k 'slot_fence or streamed_decode_evaluates_shared_work or 128k_prefill_preserves'
```

Make the completion error sticky and check it inside `_ensure_route_locked` immediately before reuse after any pin wait. Snapshot and close drain and re-raise without consuming it. Separate closing from finalized-closed state so timeout is retryable. Fence bindings in both hit and miss evaluators while retaining PR #13 shared-before-miss ordering.

- [ ] **Step 3: Verify, commit, and gate**

Run full pytest/Ruff/diff checks. Gate with `MTPLX_EXPERT_SLOT_FENCES=1`; require token parity, `completion_fences > 0`, zero fence fallbacks/failures, intact overlap order, and no material throughput/tail/bytes/memory regression.

### Task 4: Repair and gate PR #14 ready-miss streaming

**Files:**
- Modify: `mtplx/expert_runtime.py`, `mtplx/expert_slots.py`, `mtplx/models/expert_mlx.py`
- Test: `tests/test_expert_slots_runtime.py`, `tests/test_streamed_models.py`

**Security flag:** `none`

**Does NOT cover:** staged gate/up/down projection loading; Task 5 composes that behavior onto this per-expert future model.

- [ ] **Step 1: Transplant source and add RED tests**

```bash
git cherry-pick -x 87ea0b7
```

Add `test_incremental_miss_failure_cancels_running_sibling_without_blocking_primary`, `test_incremental_submit_failure_releases_pinned_hit`, `test_streamed_miss_parts_have_one_owner`, `test_streamed_miss_compute_failure_releases_current_part`, and `test_incremental_part_observes_sticky_completion_failure`. Use events/fake futures, bounded joins, and release counters; never rely on sleep-only timing.

- [ ] **Step 2: Prove RED and repair**

```bash
uv run pytest -q tests/test_expert_slots_runtime.py -k 'split_route or incremental or pending_split or completion_fence'
uv run pytest -q tests/test_streamed_models.py -k 'streamed_decode_evaluates_shared_work or 128k_prefill_preserves or streamed_miss'
```

Give all parts one internal cancellation event composed with caller cancellation/deadline. On failure, signal cancellation first, consume only completed futures, attach cleanup callbacks to running siblings, and preserve the primary error. `PendingSplitRoute` owns and releases every hit/miss route exactly once; model code never releases a part directly. Keep PR #17 sticky checks/fences and PR #13 shared overlap.

- [ ] **Step 3: Verify, commit, and gate**

Run full pytest/Ruff/diff checks. Require exercised incremental-route/part counters, token parity, positive mean+median decode gain without systematic reversal, and no material tail latency, bytes/op, SSD, or peak-memory regression.

### Task 5: Repair and gate PR #18 projection/read pipelining

**Files:**
- Modify: `mtplx/expert_io.py`, `mtplx/expert_runtime.py`, `mtplx/expert_slots.py`, `mtplx/models/expert_mlx.py`
- Test: `tests/test_expert_slots_runtime.py`, `tests/test_streamed_models.py`

**Security flag:** `none`

**Does NOT cover:** unverified/trusted-only sidecars or prefill projection pipelining; the candidate gate uses verified sidecars and bounded full-record prefill.

- [ ] **Step 1: Transplant source and add RED tests**

```bash
git cherry-pick -x f37be96
```

Add `test_close_waits_for_projection_pin_after_suffix_failure`, `test_suffix_failure_is_not_masked_by_projection_release_sync_failure`, `test_projection_wait_honors_cancel_and_deadline`, and `test_incremental_projection_parts_preserve_order_and_single_ownership`. Preserve existing logit-parity, sidecar-verification, shared-before-prefix, and 128K-prefill-disabled tests.

- [ ] **Step 2: Prove RED and repair**

```bash
uv run pytest -q tests/test_expert_slots_runtime.py -k 'projection or incremental or completion_fence or close'
uv run pytest -q tests/test_streamed_models.py -k 'projection or slot_fence or streamed_decode_evaluates_shared_work or 128k'
```

Count projection routes as active leases until exactly-once release. Forward combined cancellation/deadline through prefix and suffix waits. Adapt projection readiness to Task 4's per-expert futures rather than restoring one `_miss_future`. After `stage_component_gate_up()`, explicitly `mx.eval(hidden)` and release that projection lease with `synchronize=False`; down consumes the staged tensor. Preserve a suffix exception as primary and record any cleanup error for snapshot/close. Fence final down outputs through Task 3.

- [ ] **Step 3: Verify, commit, and gate**

Run full pytest/Ruff/diff checks. Use `--verified-sidecar` for both arms. Require `progressive_loads > 0`, `projection_ready_routes > 0`, token/stop parity, positive mean+median decode gain, bounded peak memory, and no material tail/physical-SSD/co-tenant regression.

### Task 6: Repair and gate PR #16 Metal-resident routing

**Files:**
- Modify: `mtplx/models/expert_mlx.py`, `mtplx/expert_runtime.py`, `mtplx/expert_slots.py`, `mtplx/expert_streaming.py`
- Test: `tests/test_streamed_models.py`, `tests/test_expert_streaming.py`, `tests/test_expert_slots_runtime.py`

**Security flag:** `none`

**Does NOT cover:** making this experiment the default; disabled behavior must remain identical and enabled results must pass a separate gate.

- [ ] **Step 1: Transplant source and add RED tests**

```bash
git cherry-pick -x 99f0c2b
```

Add `test_disabled_metal_route_requires_no_experimental_runtime_fields`, parameterized `test_experimental_metal_route_rejects_invalid_ids_before_device_work` for `-1` and `expert_count`, `test_experimental_metal_route_preserves_waves_counters_and_lru_victims`, and `test_metal_probe_failure_fences_candidate_before_releasing_lease`. The last test requires `lease-enter -> candidate-launch -> probe-failure -> candidate-fence -> lease-exit`.

- [ ] **Step 2: Prove RED and repair**

```bash
uv run --extra dev --extra server pytest -q tests/test_expert_streaming.py tests/test_expert_slots_runtime.py tests/test_streamed_models.py
```

When disabled, never access experimental runtime/spec fields. When enabled, materialize and range-check host IDs before `mx.take` or speculative Q4. Probe and commit inside each existing route wave, preserving per-wave epochs/counters/victims and one shared-work execution. Fence every launched candidate in `finally` before its mapping/route lifetime exits; preserve the primary error and fail closed on fence failure.

- [ ] **Step 3: Verify, commit, and gate**

First prove disabled candidate matches the prior tip. Then run at least three alternating pairs with `MTPLX_EXPERIMENTAL_METAL_ROUTE_RESOLUTION=1`. Require token/stop parity, positive median+mean, `all_hits > 0`, zero leaked pins/routes, and no material memory/SSD regression.

### Task 7: Repair and gate PR #11 prompt-wide hotsets

**Files:**
- Modify: `mtplx/expert_streaming.py`, `mtplx/expert_runtime.py`, `mtplx/models/expert_mlx.py`, `mtplx/runtime.py`, `mtplx/generation.py`
- Test: `tests/test_expert_streaming.py`, `tests/test_expert_slots_runtime.py`, `tests/test_streamed_models.py`, `tests/test_generation_sustained.py`

**Security flag:** `none`

**Does NOT cover:** changing the decode admission policy; it only seeds the existing policy from a request-scoped prompt window.

- [ ] **Step 1: Transplant the clean source and add RED tests**

```bash
git cherry-pick -x 3d7c158
```

Add layer/global times LRU/frequency final-wave tests where expert 0 appears three times early and once in the final wave while expert 1 appears five times in that final wave; expert 1 must be persistent for first decode. Add `test_prefill_request_window_clears_state_after_abort` plus success, `PostcommitAbort`, and arbitrary-failure generation cases. Add `test_global_prefill_seed_accepts_chunk_wider_than_route_capacity` with transient capacity 1 and seed `(0, 1)`.

- [ ] **Step 2: Prove RED and repair**

```bash
uv run --extra dev --extra server pytest -q tests/test_expert_streaming.py tests/test_expert_slots_runtime.py tests/test_streamed_models.py tests/test_generation_sustained.py
```

Defer selected hotset candidates to candidate-only final waves so temporary pins are gone before persistent assignment. Add serialized `ExpertStreamingRuntime.prefill_request()` ownership that begins seed state at entry and finalizes it in `finally`; delegate through `MTPLXRuntime` and wrap the whole restore/prefill operation. Add global seed validation that checks type/range without applying per-wave cardinality until after partitioning.

- [ ] **Step 3: Verify, commit, and gate**

Run full pytest/Ruff/diff checks. Use a 128K prompt with 2,048-token prefill chunks followed by the deterministic complete-generation lane. Require token/stop parity, expected final resident hotset, cleared state after injected abort, no global-cardinality rejection, repeatable positive first-decode/complete-generation effect, and no material peak-memory or SSD regression.

### Task 8: Final evidence, review, and publication

**Files:**
- Modify: `docs/MOE_RUNTIME_PR_BENCHMARKS.md`
- Modify: `benchmarks/results/moe-runtime-gate-matrix.md`
- Add: candidate raw JSON and response Markdown under `benchmarks/results/`

**Security flag:** `none`

**Does NOT cover:** merging PR #20 into its base or closing source PRs; PR #20 remains a draft until separately authorized.

- [ ] **Step 1: Update evidence after every gate**

Record source SHA, repaired tip, focused/full/static commands, all raw artifact paths, token hashes, runner-hook fields, matched statistics, and retain/skip reason. Never record an unrun performance gate as zero.

- [ ] **Step 2: Run final fresh verification**

```bash
uv run pytest -q
uv run ruff check mtplx tests scripts
uv run ruff format --check mtplx tests scripts
git diff --check
git status --short
```

Expected: pytest exit 0 with only documented skips; Ruff and diff checks exit 0; status contains only intentional evidence/doc changes before the final commit.

- [ ] **Step 3: Restore Qwen and verify both services**

```bash
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tea.qwen.plist"
launchctl list | grep -q 'com.tea.qwen'
launchctl list | grep -q 'com.tea.qwen-gateway'
curl --fail --silent http://127.0.0.1:8080/v1/models
```

- [ ] **Step 4: Commit, push, and update draft PR #20**

```bash
git add docs benchmarks/results
git commit -m 'bench: record repaired MoE runtime gates'
git push origin codex/moe-runtime-gated-integration
gh pr checks 20 --repo davidtai/MTPLX
```

Push only a fully verified retained tip. Keep PR #20 draft and link issue #21 as non-blocking follow-up storage research.
