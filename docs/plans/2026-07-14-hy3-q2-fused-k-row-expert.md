# Hy3 Q2 Fused K-Row Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:executing-plans` to implement this plan task-by-task.
> Use `superpowers-optimized:test-driven-development` for behavior changes,
> `superpowers-optimized:performance-investigation` for timing decisions, and
> `superpowers-optimized:verification-before-completion` before completion
> claims. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently measurable Q2 NAX, fused K-row, and fused-plus-NAX
expert execution arms for streamed Hy3 without changing router, cache, or slot
ownership semantics.

**Architecture:** Keep the stock component-bank path as the default and add one
fail-closed runtime selector. Q2 NAX groups rows that share a pinned expert;
the fused arm keeps assignment-indexed mixed-expert rows and replaces the
gate/up/SwiGLU sequence with one Metal kernel plus a Q2 down kernel. The
combined arm reuses NAX grouping while fusing gate/up/SwiGLU.

**Tech Stack:** Python 3.12, MLX 0.31.x, `mx.fast.metal_kernel`, Metal 4
MetalPerformancePrimitives tensor operations, pytest, Ruff, and the existing
Hy3 component-bank runtime.

**Assumptions:**

- The sustained K=0...6 matrix selects useful depths before end-to-end kernel
  benchmarking — it will NOT infer the winning K from 128-output diagnostics.
- The 2,048-token serial-versus-batched parity drift is classified first — a
  custom kernel will NOT be promoted against a moving correctness reference.
- The host exposes Apple G17 NAX for NAX arms — unsupported hardware will run
  stock and report ineligible, never a plain-SIMD result labeled NAX.
- The artifact remains affine Q2 group-64 with BF16 metadata and Hy3 dimensions
  4096 -> 1536 -> 4096 — other models and quantizations remain stock.

---

## File structure

### Create

- `mtplx/q2_nax.py`: Q2 shared-RHS NAX kernels, eligibility, padding, and
  counters.
- `mtplx/kernels/q2_fused_expert.py`: assignment-indexed Q2
  gate+up+SwiGLU and down kernels.
- `scripts/benchmark_q2_expert_kernels.py`: real-record operator benchmark for
  stock, NAX, fused, and fused-NAX arms.
- `tests/test_q2_nax.py`: Q2 NAX eligibility, unpacking, exactness, and fallback.
- `tests/test_q2_fused_expert.py`: mixed-slot fused-kernel exactness and shape
  tests.
- `tests/test_benchmark_q2_expert_kernels.py`: pure CLI/schema/gate tests.

### Modify

- `mtplx/expert_runtime.py`: validated `q2_expert_kernel` selector in the
  immutable streaming config.
- `mtplx/models/expert_mlx.py`: dispatch the selected arm inside the existing
  pinned component-bank wave.
- `scripts/benchmark_q2_mtp_depth_matrix.py`: expose and serialize the selector.
- `tests/test_expert_slots_runtime.py`: config validation and serialization.
- `tests/test_streamed_models.py`: assignment order, fallback, pin/fence, and
  arm-dispatch integration.
- `tests/test_benchmark_q2_mtp_depth_matrix.py`: CLI/runtime plumbing.

## Task 1: Resolve the stock batched parity reference

**Files:**

- Modify only after diagnosis: `mtplx/generation.py` or the proven cache/shape
  owner
- Test: `tests/test_generation_sustained.py`
- Test: `tests/test_benchmark_q2_mtp_depth_matrix.py`

**Security flag:** `none`

**Does NOT cover:** Custom expert arithmetic or tolerance-based acceptance.

- [ ] **Step 1: Capture the first divergent layer and tensor**

Add a diagnostic observer that hashes target inputs, logits, hidden rows, cache
offsets, and router IDs for serial B=1 and batched B=K+1 calls at the existing
2,048-token token-5 repro. Keep it inactive unless explicitly supplied.

- [ ] **Step 2: Run the exact repro**

```bash
MTPLX_SUSTAINED_PREFILL=1 .venv/bin/python scripts/benchmark_q2_mtp_depth_matrix.py \
  --model hy3-q2 --contexts 2048 --output-tokens 128 --hy3-depths 1 \
  --transient-slots 16 --verify-strategy capture_commit \
  --compiled-verify-mode off --no-resource-telemetry
```

Expected: reproduce the token-5 divergence and identify the earliest differing
tensor without changing execution.

- [ ] **Step 3: Write a failing regression at the proven boundary**

The test must assert exact same-shape output/cache behavior and fail on the
specific shape, mask, cache, or rounding defect shown by Step 2.

- [ ] **Step 4: Implement the smallest proven correction**

Preserve the target causal mask, committed-prefix cache offset, and activation
dtype. Do not serialize K+1 rows into K+1 target calls.

- [ ] **Step 5: Verify**

```bash
.venv/bin/python -m pytest -q tests/test_generation_sustained.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py
```

Expected: unit regression passes and the 2,048 hardware row has exact AR token
parity.

## Task 2: Add a fail-closed expert-kernel selector

**Files:**

- Modify: `mtplx/expert_runtime.py`
- Modify: `scripts/benchmark_q2_mtp_depth_matrix.py`
- Test: `tests/test_expert_slots_runtime.py`
- Test: `tests/test_benchmark_q2_mtp_depth_matrix.py`

**Security flag:** `none`

**Does NOT cover:** Non-component-bank layouts, prefill, non-Q2 artifacts, or
runtime arm switching during a request.

- [ ] **Step 1: Write failing config and CLI tests**

```python
assert ExpertStreamingConfig(...).q2_expert_kernel == "stock"
assert ExpertStreamingConfig(..., q2_expert_kernel="nax").to_dict()[
    "q2_expert_kernel"
] == "nax"
with pytest.raises(ValueError, match="q2_expert_kernel"):
    ExpertStreamingConfig(..., q2_expert_kernel="unknown")

args = module.build_parser().parse_args(["--q2-expert-kernel", "fused"])
assert args.q2_expert_kernel == "fused"
```

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_expert_slots_runtime.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py -k q2_expert_kernel
```

Expected: FAIL because the selector does not exist.

- [ ] **Step 3: Add the immutable selector**

Add `q2_expert_kernel: str = "stock"` and validate exactly
`{"stock", "nax", "fused", "fused-nax"}`. Add the matching CLI choice and
pass it into `ExpertStreamingConfig`. Serialize it in every matrix artifact.

- [ ] **Step 4: Run GREEN**

Run the Step 2 command. Expected: PASS.

## Task 3: Implement and gate the independent Q2 NAX primitive

**Files:**

- Create: `mtplx/q2_nax.py`
- Create: `tests/test_q2_nax.py`

**Security flag:** `none`

**Does NOT cover:** Gathered mixed-RHS rows, fusion, FP32 metadata, or devices
without G17 NAX.

- [ ] **Step 1: Write failing eligibility tests**

```python
assert q2_nax_eligible(1, 4096, 1536, 2, 64, mx.bfloat16, available=True)
assert q2_nax_eligible(7, 1536, 4096, 2, 64, mx.bfloat16, available=True)
assert not q2_nax_eligible(17, 4096, 1536, 2, 64, mx.bfloat16, available=True)
assert not q2_nax_eligible(1, 4096, 1536, 4, 64, mx.bfloat16, available=True)
assert not q2_nax_eligible(1, 4096, 1536, 2, 64, mx.bfloat16, available=False)
```

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_q2_nax.py
```

Expected: FAIL because `mtplx.q2_nax` does not exist.

- [ ] **Step 3: Implement Q2 unpacking and the NAX tile**

Port the existing m16 MPP structure from `mtplx/nax_verify.py`, changing the B
tile load so one `uint32_t` supplies sixteen 2-bit values:

```metal
uint32_t packed = w_q[n_global * (K / 16) + (k0 >> 4)];
for (int ki = 0; ki < 16; ++ki) {
  uint q = (packed >> (ki * 2)) & 0x3u;
  B_tile[sg_id][ki * BN + lane] = T(float(q) * scale + bias);
}
```

Pad M to 16, slice the requested rows on return, and expose explicit eligible,
executed, and padded-row counters.

- [ ] **Step 4: Add stock-parity tests**

For M=1...7 and both 4096x1536 and 1536x4096 shapes, quantize deterministic
weights to Q2, compare with `mx.quantized_matmul`, require finite deterministic
output, and record maximum error.

- [ ] **Step 5: Run GREEN and lint**

```bash
.venv/bin/python -m pytest -q tests/test_q2_nax.py
.venv/bin/ruff check mtplx/q2_nax.py tests/test_q2_nax.py
```

Expected: PASS on G17; hardware-specific execution tests skip elsewhere while
pure eligibility tests still pass.

## Task 4: Wire and microbenchmark NAX alone

**Files:**

- Modify: `mtplx/models/expert_mlx.py`
- Modify: `tests/test_streamed_models.py`
- Create: `scripts/benchmark_q2_expert_kernels.py`
- Create: `tests/test_benchmark_q2_expert_kernels.py`

**Security flag:** `none`

**Does NOT cover:** Gate/up fusion or assignment-indexed fused kernels.

- [ ] **Step 1: Write failing integration tests**

Construct repeated and unique component-bank bindings. With `kernel="nax"`,
assert groups are keyed by `(id(bank), bank_index, generation)`, three NAX
calls occur per group, original assignment order is restored, and outputs stay
roots of the existing completion fence. Assert ineligible K=0 falls back before
any partial NAX dispatch and increments one fallback counter.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_streamed_models.py -k 'q2_nax or expert_kernel'
```

Expected: FAIL because NAX dispatch is not wired.

- [ ] **Step 3: Implement grouped NAX dispatch**

Add a focused `_run_component_bank_q2_nax` helper. It must gather activation
rows by group, invoke Q2 NAX gate/up/down with the existing standalone SwiGLU,
and return group positions for the unchanged deterministic reassembly path.

- [ ] **Step 4: Add the operator benchmark**

The benchmark must alternate stock/NAX, use real pinned Q2 component-bank
records, cover observed M-group histograms for K=0...6, synchronize every timed
sample, and emit per-stage and total timings plus exactness and counters.

- [ ] **Step 5: Verify and run the NAX-only hardware arm**

```bash
.venv/bin/python -m pytest -q tests/test_streamed_models.py \
  tests/test_benchmark_q2_expert_kernels.py
.venv/bin/python scripts/benchmark_q2_expert_kernels.py \
  --arms stock,nax --depths 0,1,2,3,4,5,6 --output-json /tmp/issue51-q2-nax.json
```

Expected: complete K=0...6 NAX table before fused-kernel timing begins.

## Task 5: Implement assignment-indexed fused K-row Q2 kernels

**Files:**

- Create: `mtplx/kernels/q2_fused_expert.py`
- Create: `tests/test_q2_fused_expert.py`

**Security flag:** `none`

**Does NOT cover:** NAX tensor operations, weighted router reduction, prefill,
or cross-bank execution.

- [ ] **Step 1: Write failing eligibility and parity tests**

Cover M=1,8,16,24,32,40,48,56; unique/repeated slot indices; Q2 group-64
BF16; and both projection shapes. Require rejection for mismatched bank shapes,
indices, bits, dtype, or M>56.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_q2_fused_expert.py
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement gather-aware gate+up+SwiGLU**

Adapt the exact Q4 structure in `mtplx/kernels/verify_mlp_fused.py` to Q2 and
add `slot_indices[row]` to every component-bank base address. One packed word
contains sixteen Q2 values. Round gate/up to T before the existing
`sigmoid_mlx_exact` and SwiGLU sequence.

- [ ] **Step 4: Implement gather-aware down**

Use the same row-to-slot mapping and Q2 affine unpacking for 1536 -> 4096.
Return `[M,4096]` in input assignment order so no new scatter is required.

- [ ] **Step 5: Run GREEN and lint**

```bash
.venv/bin/python -m pytest -q tests/test_q2_fused_expert.py
.venv/bin/ruff check mtplx/kernels/q2_fused_expert.py \
  tests/test_q2_fused_expert.py
```

Expected: PASS with deterministic bounded-error parity against three stock
`mx.gather_qmm` operations.

## Task 6: Wire and benchmark fused K-row execution alone

**Files:**

- Modify: `mtplx/models/expert_mlx.py`
- Modify: `tests/test_streamed_models.py`
- Modify: `scripts/benchmark_q2_expert_kernels.py`
- Modify: `tests/test_benchmark_q2_expert_kernels.py`

**Security flag:** `none`

**Does NOT cover:** NAX or changes to Hy3 score multiplication/reduction.

- [ ] **Step 1: Write failing arm-dispatch tests**

With `kernel="fused"`, assert exactly one fused gate/up/SwiGLU call and one
fused down call per component-bank subwave, unchanged assignment order, stock
fallback for prefill/ineligible shapes, and completion fences rooted in the
down output.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_streamed_models.py -k q2_fused
```

Expected: FAIL because fused dispatch is not wired.

- [ ] **Step 3: Implement dispatch and counters**

Pass the immutable selector from `HotExpertSwitchGLU` into
`_run_component_bank_q4`. Select the custom path only when the whole subwave is
eligible; otherwise execute the unchanged stock path before any custom launch.

- [ ] **Step 4: Extend and run the operator benchmark**

```bash
.venv/bin/python scripts/benchmark_q2_expert_kernels.py \
  --arms stock,fused --depths 0,1,2,3,4,5,6 \
  --output-json /tmp/issue51-q2-fused.json
```

Expected: per-K stock/fused operator table with grouping absent from fused
timings and exactness gates passed.

## Task 7: Implement and benchmark fused plus NAX

**Files:**

- Modify: `mtplx/q2_nax.py`
- Modify: `mtplx/models/expert_mlx.py`
- Modify: `tests/test_q2_nax.py`
- Modify: `tests/test_streamed_models.py`
- Modify: `scripts/benchmark_q2_expert_kernels.py`

**Security flag:** `none`

**Does NOT cover:** Running the combined arm if either individual arm fails
correctness, or reporting a win only against stock.

- [ ] **Step 1: Write failing fused-NAX tests**

Require the first shared-RHS NAX launch to compute gate+up+exact-SwiGLU, the
second to compute down, two launches per group, unchanged grouping/reassembly,
and exact fallback before partial dispatch.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_q2_nax.py tests/test_streamed_models.py \
  -k fused_nax
```

Expected: FAIL because the fused NAX primitive does not exist.

- [ ] **Step 3: Implement fused NAX gate/up/SwiGLU**

Reuse one activation tile and run gate/up MPP operations before writing the
rounded SwiGLU intermediate. Keep the existing NAX down primitive as launch 2.

- [ ] **Step 4: Run the combined operator comparison**

```bash
.venv/bin/python scripts/benchmark_q2_expert_kernels.py \
  --arms stock,nax,fused,fused-nax --depths 0,1,2,3,4,5,6 \
  --output-json /tmp/issue51-q2-all-kernels.json
```

Expected: fused-NAX is compared directly with both individual arms and all
correctness fields pass.

## Task 8: Run end-to-end promotion gates and publish Issue 51

**Files:**

- Modify: `benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.md`
- Modify: `benchmarks/results/issue51-hy3-q2-mtp-lookahead-20260714.json`

**Security flag:** `none`

**Does NOT cover:** Enabling a losing arm by default or replacing the 128-token
qualification contract with the sustained experiment.

- [ ] **Step 1: Select end-to-end cells from the sustained matrix**

Retain K=0 plus every K whose 1,028-output decode result is within 5% of the
best at any context. Record this mechanical selection in the artifact.

- [ ] **Step 2: Run matched exclusive campaigns**

For each eligible arm, run 1,024/2,048/4,096 inputs with 1,028 outputs, fixed 56
reader slots, max-live-KV 8192, one resident model load, interactive QoS, and
the exact Qwen unload/restore guard.

- [ ] **Step 3: Run the 128-output qualification lane**

Only an arm with positive sustained paired decode evidence advances. Require
all Issue 51 event/cache/final-state gates and exact token parity.

- [ ] **Step 4: Verify the full affected suite**

```bash
.venv/bin/python -m pytest -q tests/test_q2_nax.py \
  tests/test_q2_fused_expert.py tests/test_streamed_models.py \
  tests/test_expert_slots_runtime.py \
  tests/test_benchmark_q2_expert_kernels.py \
  tests/test_benchmark_q2_mtp_depth_matrix.py
.venv/bin/ruff check mtplx/q2_nax.py mtplx/kernels/q2_fused_expert.py \
  mtplx/models/expert_mlx.py mtplx/expert_runtime.py \
  scripts/benchmark_q2_expert_kernels.py \
  scripts/benchmark_q2_mtp_depth_matrix.py tests
git diff --check
```

Expected: all tests and lint pass.

- [ ] **Step 5: Publish and restore**

Post operator and end-to-end tables to Issue 51 as context blocks complete.
Restore `com.tea.qwen`, wait for `/v1/models`, and require exact model
`mtplx-qwen36-27b-optimized-speed` before releasing the exclusive lane.

