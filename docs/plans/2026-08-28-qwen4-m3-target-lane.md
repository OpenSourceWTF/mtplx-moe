# Qwen4 M=3 Whole-MoE Target Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact construction-bound M=3 Qwen4 whole-MoE target lane and promote it only if it repeatably improves the unchanged depth-2 production workload without changing correctness.

**Architecture:** Generate distinct M=2 and M=3 Metal programs from the existing arithmetic by making rows a construction-time source constant and cache key. Bind stage 1, row-owned top-10 routing, stage 2, and stage 3 into immutable per-row callables. Install direct M=2 and M=3 routes only after both construction-time self-checks pass; other widths retain the explicitly captured stock route.

**Tech Stack:** Python 3.12, MLX 0.32.2, Metal kernels, pytest, the guarded Qwen3.8 oQ4 production harness, MLX profiler, git, and GitHub CLI.

**Assumptions:**

- Assumes the existing M=2 arithmetic is the exact target contract — this plan will not transplant arithmetic, ownership, tiling, data layout, or tie-breaking from another model.
- Assumes compile-time `ROWS=3` is accepted by Metal without register pressure erasing the dispatch savings — the lane will not be promoted if matched production measurements fail.
- Assumes deterministic production output and current acceptance receipts are the correctness gate — a throughput result with a changed digest or trajectory will be rejected.
- Assumes the guarded harness exclusively acquires `/tmp/mtplx-gpu-exclusive.lock` and restores `mtplx-qwen38-27b-dflash2` — no unguarded Metal or model benchmark is permitted.

---

## File structure

- Modify `mtplx/kernels/qwen4_whole_moe.py`: generate, cache, launch, and bind exact row-specialized M=2 and M=3 Metal programs.
- Modify `mtplx/qwen4_whole_moe.py`: bind both row variants at construction, self-check both, and install direct M=2/M=3 routes.
- Modify `tests/test_qwen4_whole_moe.py`: lock source geometry, row-owned routing, construction self-check, and installed route behavior.
- Update `docs/specs/2026-08-28-qwen4-m3-target-lane-design.md`: append measured outcome only if the candidate passes.
- Update PR #368 through `/tmp/pr368-body.md`: add successful production and discussion-only benchmark receipts after a pushed commit exists.

### Task 1: Generate and bind exact row-specialized kernels

**Files:**
- Modify: `mtplx/kernels/qwen4_whole_moe.py`
- Modify: `tests/test_qwen4_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** Row specialization supports only rows 2 and 3; it does not add a dynamic-row Metal kernel or an M=4 variant.

- [ ] **Step 1: Write failing source, geometry, and three-row routing tests**

```python
def test_exact_sources_encode_qwen4_storage_and_right_shapes():
    for rows in (2, 3):
        sources = kernels.sources(rows)
        assert f"constexpr uint ROWS = {rows}" in sources["stage1"]
        assert f"constexpr uint ROWS = {rows}" in sources["stage3"]
        assert "constexpr uint EXPERTS = 512" in sources["stage1"]
        assert "constexpr uint HIDDEN = 2560" in sources["stage2"]
        assert "constexpr uint TOP_K = 10" in sources["stage2"]
        assert "constexpr uint Q4_GROUP = 32" in sources["stage2"]
        assert "constexpr uint Q8_GROUP = 128" in sources["stage3"]
        assert kernels.launch_geometry(rows) == {
            "stage1": ((128 * 32, 1, 1), (32, 1, 1)),
            "stage2": ((440 * 128, 1, 1), (128, 1, 1)),
            "stage3": ((rows * 160 * 128, 1, 1), (128, 1, 1)),
        }
        assert "uint row = group / OUTPUT_TILES" in sources["stage3"]


def test_row_owned_top10_matches_qwen4_argpartition_order_for_three_rows():
    logits = mx.sin(mx.arange(3 * 512, dtype=mx.float32) * 1.337).reshape(3, 512)
    expected_ids = mx.argpartition(-logits, 9, axis=-1)[..., :10]
    expected_logits = mx.take_along_axis(logits, expected_ids, axis=-1)
    actual_ids, actual_logits = kernels.route_top10(logits, rows=3)
    mx.eval(expected_ids, expected_logits, actual_ids, actual_logits)
    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(np.asarray(actual_logits), np.asarray(expected_logits))
```

- [ ] **Step 2: Run the no-Metal source test and verify the red state**

```bash
.venv/bin/python -m pytest -q tests/test_qwen4_whole_moe.py -k exact_sources
```

Expected: FAIL because `sources()` and `launch_geometry()` do not accept a row count.

- [ ] **Step 3: Parameterize source generation, cache identity, geometry, and binding**

Use this interface and cache contract throughout `mtplx/kernels/qwen4_whole_moe.py`:

```python
SUPPORTED_ROWS = (2, 3)
_KERNELS: dict[tuple[int, str], Any] = {}


def _require_rows(rows: int) -> int:
    rows = int(rows)
    if rows not in SUPPORTED_ROWS:
        raise ValueError(f"Qwen4 whole-MoE rows must be one of {SUPPORTED_ROWS}: {rows}")
    return rows


def _preamble(rows: int) -> str:
    return f"""
        using namespace metal;
        constexpr uint HIDDEN = {HIDDEN};
        constexpr uint EXPERTS = {EXPERTS};
        constexpr uint TOP_K = {TOP_K};
        constexpr uint INTERMEDIATE = {INTERMEDIATE};
        constexpr uint ACTIVATION_SLOTS = {ACTIVATION_SLOTS};
        constexpr uint ROWS = {rows};
    """


def sources(rows: int = 2) -> dict[str, str]:
    rows = _require_rows(rows)
    return {
        "stage1": _stage1_source(rows),
        "route": _route_source(rows),
        "stage2": _stage2_source(rows),
        "stage3": _stage3_source(rows),
    }


def launch_geometry(rows: int = 2):
    rows = _require_rows(rows)
    return {
        "stage1": ((STAGE1_GROUPS * STAGE1_THREADS, 1, 1), (STAGE1_THREADS, 1, 1)),
        "stage2": ((STAGE2_GROUPS * THREADS, 1, 1), (THREADS, 1, 1)),
        "stage3": ((rows * (HIDDEN // 16) * THREADS, 1, 1), (THREADS, 1, 1)),
    }
```

Change every `_stage*_source` plus `_route_source` to accept `rows` and call `_preamble(rows)`. Make `_kernel`, `_stage1_kernel`, and the route kernel use cache keys `(rows, key)`, unique Metal names ending in `_m{rows}`, and `sources(rows)[key]`.

Change `bind_stages` to `bind_stages(*, rows: int, router, routed, shared, shared_gate)`; validate rows before closures are constructed and return `(stage1, route_top10, stage2, stage3)`. Each closure captures the validated integer and directly uses row-derived output shapes and grids. The bound route is:

```python
def route_top10(router_logits: Any):
    expert_ids, selected_logits = route_kernel(
        inputs=[router_logits],
        grid=(rows * ROUTE_THREADS, 1, 1),
        threadgroup=(ROUTE_THREADS, 1, 1),
        output_shapes=[(rows, TOP_K), (rows, TOP_K)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return expert_ids, selected_logits
```

Keep `route_top10(router_logits, *, rows=2)` for focused parity tests, backed by the same row-specialized route kernel. Do not change Metal arithmetic or loop ownership beyond substituting the source constant.

- [ ] **Step 4: Run source verification**

```bash
.venv/bin/python -m pytest -q tests/test_qwen4_whole_moe.py -k exact_sources
git diff --check
```

Expected: the source test passes and the diff check emits no output.

- [ ] **Step 5: Commit the approved documents and kernel slice**

```bash
git add docs/specs/2026-08-28-qwen4-m3-target-lane-design.md \
  docs/plans/2026-08-28-qwen4-m3-target-lane.md \
  mtplx/kernels/qwen4_whole_moe.py tests/test_qwen4_whole_moe.py
git commit -m "Parameterize Qwen4 whole-MoE row kernels"
```

### Task 2: Install direct M=2 and M=3 construction routes

**Files:**
- Modify: `mtplx/qwen4_whole_moe.py`
- Modify: `tests/test_qwen4_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** Other logical widths remain on the captured stock implementation; runtime failures in an installed lane do not fall back to stock.

- [ ] **Step 1: Write failing construction-route and self-check tests**

```python
def test_exact_m2_m3_routes_are_installed_once_and_other_rows_stay_stock(monkeypatch):
    model = fake_model_with_48_blocks()
    monkeypatch.setenv(whole_moe.WHOLE_MOE_ENV, "1")
    monkeypatch.setattr(
        whole_moe, "_bind",
        lambda block, rows: SimpleNamespace(block=block, rows=rows),
    )
    monkeypatch.setattr(
        whole_moe, "_whole_call",
        lambda block, binding, value: ("whole", binding.rows, value.shape[1]),
    )
    report = whole_moe.configure_qwen4_whole_moe(
        model, config=QWEN4_CONFIG, validate_storage=False, run_selfcheck=False
    )
    block = model.language_model.model.layers[0].mlp
    assert report["geometry"]["rows"] == (2, 3)
    assert block(_Value(2)) == ("whole", 2, 2)
    assert block(_Value(3)) == ("whole", 3, 3)
    assert block(_Value(1)) == ("stock", 1)
    assert block(_Value(4)) == ("stock", 4)


def test_construction_selfchecks_both_rows_before_install(monkeypatch):
    seen = []
    monkeypatch.setattr(
        whole_moe, "_selfcheck",
        lambda block, accepted_call, rows: seen.append(rows) or rows / 10,
    )
    report = configure_fake_model(monkeypatch, run_selfcheck=True)
    assert seen == [2, 3]
    assert report["selfcheck_dmax"] == {"m2": 0.2, "m3": 0.3}
```

Define `QWEN4_CONFIG`, `fake_model_with_48_blocks`, and `configure_fake_model` in the test file from the existing 48-layer fixture and exact config literal.

- [ ] **Step 2: Run route tests and verify the red state**

```bash
.venv/bin/python -m pytest -q tests/test_qwen4_whole_moe.py \
  -k 'm2_m3_routes or construction_selfchecks'
```

Expected: FAIL because `_Route` has no M=3 callable, `_bind` has no row argument, and self-check is M=2-only.

- [ ] **Step 3: Bind both variants and install the direct route**

Use these structures and direct call path:

```python
@dataclass(frozen=True)
class _Binding:
    stage1: Callable[[Any], tuple[Any, Any]]
    route_top10: Callable[[Any], tuple[Any, Any]]
    stage2: Callable[[Any, Any], Any]
    stage3: Callable[[Any, Any, Any, Any], Any]


@dataclass(frozen=True)
class _Route:
    accepted_call: Callable[[Any, Any], Any]
    m2_call: Callable[[Any], Any]
    m3_call: Callable[[Any], Any]


def _whole_call(block: Any, binding: _Binding, value: Any) -> Any:
    logits, shared_gate = binding.stage1(value)
    expert_ids, selected_logits = binding.route_top10(logits)
    route_scores = mx.softmax(selected_logits, axis=-1, precise=True)
    activations = binding.stage2(value, expert_ids)
    output = binding.stage3(activations, expert_ids, route_scores, shared_gate)
    return output.reshape(value.shape)


def _installed_call(self: Any, value: Any) -> Any:
    rows = 1
    for dimension in value.shape[:-1]:
        rows *= int(dimension)
    route = type(self)._mtplx_qwen4_whole_moe_route
    if rows == 2:
        return route.m2_call(value)
    if rows == 3:
        return route.m3_call(value)
    return route.accepted_call(self, value)
```

Implement `_bind(block, rows)` through `kernels.bind_stages(rows=rows, ...)`. Implement `_selfcheck(block, accepted_call, rows)` using `rows * 2560` deterministic BF16 inputs, the accepted call, and `_whole_call`; preserve the 0.5 BF16 error threshold and include `M={rows}` in errors.

Before changing a block class, self-check rows 2 and 3 on the first block. For each block, bind both variants once and close them into `m2_call` and `m3_call`; install `_Route` without enabled-lane exception handling. Report `selfcheck_dmax={"m2": dmax2, "m3": dmax3}` and `geometry["rows"]=(2, 3)`.

- [ ] **Step 4: Run all no-model focused tests**

```bash
.venv/bin/python -m pytest -q tests/test_qwen4_whole_moe.py -k 'not row_owned_top10'
git diff --check
```

Expected: selected tests pass and the diff check emits no output.

- [ ] **Step 5: Commit the construction-installed M=3 lane**

```bash
git add mtplx/qwen4_whole_moe.py tests/test_qwen4_whole_moe.py
git commit -m "Install exact Qwen4 M3 whole-MoE route"
```

### Task 3: Prove Metal parity and production performance

**Files:**
- Modify only on success: `docs/specs/2026-08-28-qwen4-m3-target-lane-design.md`

**Security flag:** `none`

**Does NOT cover:** A short greedy palindrome result cannot promote the candidate, and a production regression cannot be hidden by acceptance or profiler diagnostics.

- [ ] **Step 1: Run focused Metal tests under the canonical guard**

```bash
.venv/bin/python /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-path /tmp/mtplx-gpu-exclusive.lock \
  --lock-timeout-seconds 21600 --child-timeout-seconds 7200 -- \
  .venv/bin/python -m pytest -q tests/test_qwen4_whole_moe.py
```

Expected: all tests pass and the guard restores the service.

- [ ] **Step 2: Capture two unchanged depth-2 production controls**

From the clean diagnostic worktree at `ce915106`, run this command twice, changing `r1` to `r2`:

```bash
env MTPLX_FUSE_PROJ=gdn,attn,hyper MTPLX_QWEN4_WHOLE_MOE_M2=1 \
  .venv/bin/python scripts/qwen38_flash_next_oq4_harness.py \
  --depth 2 --verify-strategy capture_commit --warmup-runs 1 \
  --profile sustained --output /tmp/pr368-m3-control-production-r1.json
```

Expected: exact 16,384-input/1,024-output production sampler receipts. Record TPS, digest, verifier calls, acceptance, repair, target evaluation time, and target-forward time.

- [ ] **Step 3: Capture two candidate depth-2 production receipts**

From this worktree, run twice, changing `r1` to `r2`:

```bash
env MTPLX_FUSE_PROJ=gdn,attn,hyper MTPLX_QWEN4_WHOLE_MOE_M2=1 \
  .venv/bin/python scripts/qwen38_flash_next_oq4_harness.py \
  --depth 2 --verify-strategy capture_commit --warmup-runs 1 \
  --profile sustained --output /tmp/pr368-m3-candidate-production-r1.json
```

Expected: both candidate receipts retain the control digest, correctness result, and deterministic acceptance trajectory while repeatably exceeding the control mean.

- [ ] **Step 4: Reject or record the measured candidate**

If digest, correctness, acceptance trajectory, or throughput fails, retain raw receipts, use non-destructive `git revert` for the unpromoted code commits, and do not publish a success claim.

If all gates pass, append `## Measured outcome` to the design spec with both control and candidate TPS values, means, digest, verifier calls, acceptance by depth, repair count/cost, target evaluation time, and target-forward time. Commit it:

```bash
git add docs/specs/2026-08-28-qwen4-m3-target-lane-design.md
git commit -m "Record Qwen4 M3 production benchmark"
```

### Task 4: Publish the successful result and discussion number

**Files:**
- Modify: `/tmp/pr368-body.md`

**Security flag:** `none`

**Does NOT cover:** The palindrome number is not production throughput and does not replace the 16K/1K temperature-1 benchmark.

- [ ] **Step 1: Run palindrome only after production promotion passes**

Run twice, changing `r1` to `r2`:

```bash
env MTPLX_FUSE_PROJ=gdn,attn,hyper MTPLX_QWEN4_WHOLE_MOE_M2=1 \
  .venv/bin/python scripts/qwen38_flash_next_oq4_harness.py \
  --headline --depth 2 --verify-strategy capture_commit --warmup-runs 1 \
  --profile sustained --output /tmp/pr368-m3-palindrome-r1.json
```

Expected: 100-token prompt, greedy sampler, and valid output. The open goal remains unmet unless the repeatable mean is at least 90 tok/s.

- [ ] **Step 2: Push commits and update PR #368**

Add production control/candidate and clearly labeled discussion-only rows to `/tmp/pr368-body.md`. Include the M=3 summary, exact sampler, means, digest/parity, acceptance, repair, and commit SHA. Then:

```bash
git push origin perf/qwen4-projection-fusion
gh pr edit 368 --repo youssofal/MTPLX --body-file /tmp/pr368-body.md
```

Expected: the remote branch contains every successful improvement and the PR body reports only measured evidence.

- [ ] **Step 3: Verify remote state, checks, service, and cleanliness**

```bash
gh pr view 368 --repo youssofal/MTPLX \
  --json headRefOid,title,body,statusCheckRollup,url
git status --short
curl --fail --silent http://127.0.0.1:8080/health
```

Expected: remote head matches local `HEAD`, checks pass or run, the worktree is clean, and the restored service returns HTTP 200.
