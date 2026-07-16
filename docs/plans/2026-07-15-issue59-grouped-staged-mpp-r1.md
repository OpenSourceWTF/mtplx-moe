# Issue 59 Grouped Staged-MPP R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaust and qualify grouped-direct and grouped-staged MPP R1 schedules for the authoritative Hy3 router without changing its FP32 arithmetic contract.

**Architecture:** Preserve the current `[P,8,192]` FP32 partial layout and precise G6 R2. Group only N-tile tasks sharing one K partition. The staged arm loads the shared `8 x K-span` FP32 activation slice once per threadgroup, while each SIMDgroup consumes a distinct BF16 device-weight tile.

**Tech Stack:** Python 3.12, MLX 0.31.2, Metal 4 MPP tensor operations, pytest, Ruff.

**Assumptions:** Requires Apple G17 MPP support and the exact Hy3 `M1...M8 x K4096 x N192` router — it will not run on other shapes or devices. Staged P8/P16/P32 requires 16/8/4 KiB threadgroup memory — P1/P2 are excluded because their 128/64 KiB activation slices exceed the intended budget.

---

## File structure

- `mtplx/hy3_router_fp32.py`: tiling contract, grouped Metal builders, dispatch, traffic accounting.
- `tests/test_hy3_router_fp32.py`: validation, source mapping, dispatch geometry, and arithmetic-contract tests.
- `benchmarks/hy3_router_tiling.py`: direct P2/P32 screen plus grouped direct/staged frontier, failure capture, paired evidence.
- `tests/test_hy3_router_tiling_benchmark.py`: candidate coverage and authoritative decision rules.
- `benchmarks/results/issue59-hy3-router-grouped-r1-20260715.md`: curated locked evidence.

### Task 1: Extend the tiling contract and analytical accounting

**Files:**
- Modify: `mtplx/hy3_router_fp32.py`
- Test: `tests/test_hy3_router_fp32.py`

**Security flag:** `none`

**Does NOT cover:** It does not make grouped/staged schedules selectable at runtime; it only defines and validates legal geometry.

- [ ] **Step 1: Write failing tests**

```python
def test_grouped_tiling_requires_one_k_part_and_divisible_n_tiles() -> None:
    tiling = Hy3RouterFP32Tiling(
        16, 8, "grouped-staged", simd_groups_per_threadgroup=4
    )
    assert tiling.stage1_threadgroups == 24
    assert tiling.total_simdgroups == 96
    assert tiling.staged_threadgroup_bytes == 16 * 1024
    assert tiling.modeled_activation_bytes == 384 * 1024

@pytest.mark.parametrize("groups", (5, 7, 8))
def test_grouped_tiling_rejects_cross_part_groups(groups: int) -> None:
    with pytest.raises(ValueError, match="divide N tiles"):
        Hy3RouterFP32Tiling(
            16, 8, "grouped-staged", simd_groups_per_threadgroup=groups
        )
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py -k grouped_tiling`

Expected: FAIL because `simd_groups_per_threadgroup` and grouped accounting do not exist.

- [ ] **Step 3: Implement the contract**

```python
@dataclass(frozen=True, slots=True)
class Hy3RouterFP32Tiling:
    n_tile: Literal[16, 32, 64]
    grid_k_parts: Literal[1, 2, 4, 8, 16, 32]
    operand_mode: Literal[
        "direct", "staged", "grouped-direct", "grouped-staged"
    ] = "direct"
    k_tile: Literal[16, 32] | None = None
    simd_groups_per_threadgroup: int = 1

    def __post_init__(self) -> None:
        n_tiles = 192 // self.n_tile
        groups = int(self.simd_groups_per_threadgroup)
        if groups < 1 or groups > 8 or n_tiles % groups:
            raise ValueError("router SIMDgroups must divide N tiles within one K part")
        if self.operand_mode == "direct" and groups != 1:
            raise ValueError("direct router mode requires one SIMDgroup per threadgroup")
        if self.operand_mode in {"direct", "grouped-direct", "grouped-staged"} and self.k_tile is not None:
            raise ValueError("direct and grouped router modes do not accept k_tile")
        if self.operand_mode == "staged" and self.k_tile not in (16, 32):
            raise ValueError("legacy staged router mode requires K tile 16 or 32")
        if self.operand_mode == "grouped-staged" and self.grid_k_parts < 8:
            raise ValueError("grouped-staged router mode requires P8, P16, or P32")
```

Add `total_simdgroups`, grouped `stage1_threadgroups`, staged bytes, modeled activation bytes, weight bytes, and partial bytes properties using the exact padded-M8 geometry.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py -k grouped_tiling`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtplx/hy3_router_fp32.py tests/test_hy3_router_fp32.py
git commit -m "perf(router): model grouped MPP R1 schedules"
```

### Task 2: Implement grouped-direct dispatch

**Files:**
- Modify: `mtplx/hy3_router_fp32.py`
- Test: `tests/test_hy3_router_fp32.py`

**Security flag:** `none`

**Does NOT cover:** It does not use threadgroup memory; this arm attributes scheduling independently from activation reuse.

- [ ] **Step 1: Write failing source and dispatch tests**

```python
def test_grouped_direct_source_maps_each_simdgroup_to_one_n_tile() -> None:
    source = _grouped_partial_source(
        Hy3RouterFP32Tiling(
            16, 8, "grouped-direct", simd_groups_per_threadgroup=4
        )
    )
    assert "part = int(tg) / GROUPS_PER_PART" in source
    assert "n_tile_index = group_in_part * SGPTG + int(sg_id)" in source
    assert "threadgroup float A_tile" not in source

def test_grouped_direct_dispatch_uses_four_simdgroups_per_threadgroup(fake_metal) -> None:
    hy3_router_fp32_project(value, weight, n_tile=16, grid_k_parts=8,
        operand_mode="grouped-direct", simd_groups_per_threadgroup=4)
    assert fake_metal.threadgroup == (128, 1, 1)
    assert fake_metal.grid == (24 * 128, 1, 1)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py -k grouped_direct`

Expected: FAIL because grouped source and dispatch do not exist.

- [ ] **Step 3: Implement grouped-direct Metal**

Generate one task per SIMDgroup with:

```metal
uint tg = threadgroup_position_in_grid.x;
uint sg_id = simdgroup_index_in_threadgroup;
int part = int(tg) / GROUPS_PER_PART;
int group_in_part = int(tg) - part * GROUPS_PER_PART;
int n_tile_index = group_in_part * SGPTG + int(sg_id);
int n0 = n_tile_index * BN;
int k0 = part * KS;
```

Run the existing direct-device MPP descriptor and store each SIMDgroup's tile to the unchanged partial offset. Dispatch `stage1_threadgroups * SGPTG * 32` threads with `SGPTG * 32` threads/threadgroup.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py -k grouped_direct`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtplx/hy3_router_fp32.py tests/test_hy3_router_fp32.py
git commit -m "perf(router): group direct MPP N tiles"
```

### Task 3: Implement staged-activation dispatch

**Files:**
- Modify: `mtplx/hy3_router_fp32.py`
- Test: `tests/test_hy3_router_fp32.py`

**Security flag:** `none`

**Does NOT cover:** Weights remain device-resident BF16 and R2 remains a separate precise launch.

- [ ] **Step 1: Write failing source tests**

```python
def test_staged_source_loads_one_shared_activation_slice() -> None:
    source = _grouped_partial_source(
        Hy3RouterFP32Tiling(
            16, 8, "grouped-staged", simd_groups_per_threadgroup=4
        )
    )
    assert "threadgroup float A_tile[BM * KS]" in source
    assert "offset += SGPTG * 32" in source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in source
    assert "tensor<threadgroup float" in source
    assert "tensor<device bfloat" in source
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py -k staged_source`

Expected: FAIL because staged MPP operands are rejected.

- [ ] **Step 3: Implement staged activation**

```metal
threadgroup float A_tile[BM * KS];
for (int offset = int(tid); offset < BM * KS; offset += SGPTG * 32) {
    int row = offset / KS;
    int column = offset - row * KS;
    A_tile[offset] = x[row * K + k0 + column];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

Construct the left MPP tensor over `A_tile` with extents `{KS, BM}` and strides `{1, KS}`. Keep the BF16 right tensor and FP32 destination in device memory. Preserve the descriptor's `multiply` mode and partial indexing.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py -k staged_source`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtplx/hy3_router_fp32.py tests/test_hy3_router_fp32.py
git commit -m "perf(router): stage shared MPP activations"
```

### Task 4: Build a fail-soft architecture frontier benchmark

**Files:**
- Modify: `benchmarks/hy3_router_tiling.py`
- Test: `tests/test_hy3_router_tiling_benchmark.py`

**Security flag:** `none`

**Does NOT cover:** Screen timings cannot promote a runtime schedule; they only select finalists.

- [ ] **Step 1: Write failing frontier tests**

```python
def test_grouped_frontier_covers_direct_and_staged_controls() -> None:
    candidates = grouped_router_tiling_candidates()
    assert "n16_p8_sg4_grouped_direct" in candidates
    assert "n16_p8_sg4_staged" in candidates
    assert "n32_p16_sg6_staged" in candidates
    assert "n64_p32_sg3_staged" in candidates

def test_candidate_failure_is_recorded_without_aborting_other_arms() -> None:
    result = evaluate_candidate_arms({"bad": raises_resource, "good": returns_route})
    assert result["bad"]["failure_phase"] == "resource"
    assert result["good"]["correctness"]["candidate_topk_exact"] is True
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_hy3_router_tiling_benchmark.py -k 'grouped_frontier or failure_is_recorded'`

Expected: FAIL because the grouped frontier and per-arm failure capture are absent.

- [ ] **Step 3: Implement the screen**

Generate every legal grouped-direct/staged combination from the spec. Record compile/resource/dispatch/correctness failures per arm, modeled traffic, and direct paired comparisons against N16/P8. Keep stock-route differences diagnostic; call `authoritative_candidate_contract()` with candidate-relative R2, normalized finite weights, and repeat determinism.

- [ ] **Step 4: Verify GREEN and broader unit suite**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py tests/test_hy3_router_tiling_benchmark.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/hy3_router_tiling.py tests/test_hy3_router_tiling_benchmark.py
git commit -m "bench(router): exhaust grouped staged R1 frontier"
```

### Task 5: Locked screen, refinement, runtime policy, and publication

**Files:**
- Modify: `mtplx/models/hy3_mlx.py` only if a schedule wins refinement
- Modify: `benchmarks/results/issue59-hy3-router-grouped-r1-20260715.md`
- Test: `tests/test_hy3_router_kernel_selection.py`

**Security flag:** `none`

**Does NOT cover:** A router-only winner is not an end-to-end promotion; K0-K7 qualification remains mandatory.

- [ ] **Step 1: Run the locked M1-M8 screen**

Run:

```bash
.venv/bin/python benchmarks/hy3_router_tiling.py \
  --warmups 8 --repeats 80 --bootstrap-resamples 10000 \
  --lock-timeout-seconds 21600 \
  --output-json /tmp/issue59-grouped-r1-screen.json
```

Expected: immutable artifact with per-arm failures/correctness/timing and exact Qwen restoration receipt.

- [ ] **Step 2: Refine the best exact K3 arm**

Run the winning arm and N16/P8 control with 2,000 paired interleaved repeats and 20,000 bootstrap resamples. Expected: candidate-relative correctness passes and a direct interval that either lies wholly above parity or honestly rejects that topology.

- [ ] **Step 3: Add a failing schedule-policy test only for a winner**

```python
def test_authoritative_mpp_selector_dispatches_refined_k3_schedule(monkeypatch) -> None:
    router = _exact_router()
    router.configure_kernel("mpp-r1-fused-r2", available=True)
    router(mx.zeros((1, 4, 4096), dtype=mx.bfloat16))
    assert calls[0]["operand_mode"] == WINNING_MODE
    assert calls[0]["simd_groups_per_threadgroup"] == WINNING_GROUPS
```

Run: `.venv/bin/pytest -q tests/test_hy3_router_kernel_selection.py -k refined_k3_schedule`

Expected: FAIL until the measured schedule is wired.

- [ ] **Step 4: Wire only the measured winner and verify**

Run: `.venv/bin/pytest -q tests/test_hy3_router_fp32.py tests/test_hy3_router_kernel_selection.py tests/test_hy3_router_tiling_benchmark.py`

Expected: PASS on the qualified G17 lane; non-hardware tests pass everywhere.

- [ ] **Step 5: Publish evidence and queue full qualification**

Post the curated table and artifact SHA256 to Issue #59, update parent #51, and run matched isolated Hy3-Q2 1,024/1,024 K0-K7 control/candidate matrices plus the reader/cache telemetry lane. Do not label the mode promoted until those gates pass.

- [ ] **Step 6: Commit**

```bash
git add mtplx/models/hy3_mlx.py tests/test_hy3_router_kernel_selection.py benchmarks/results/issue59-hy3-router-grouped-r1-20260715.md
git commit -m "perf(router): select measured grouped MPP R1 schedule"
```
