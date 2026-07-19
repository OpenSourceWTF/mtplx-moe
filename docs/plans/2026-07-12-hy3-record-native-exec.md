# Hy3 Record-Native Sparse Execution Implementation Plan

**Status: NOT IMPLEMENTED (verified 2026-07-18).** `mtplx/kernels/moe_record_q4.py`, `scripts/benchmark_hy3_record_q4.py`, and `tests/test_moe_record_q4.py` do not exist. Kept as a design record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and, only if it wins, integrate a two-stage Hy3 affine-Q4 primitive that consumes the existing v1 expert record and reduces weighted top-8 down outputs directly.
**Architecture:** Build on PR 1. Keep cache policy and I/O on the host, preserve the gate/up-to-down dependency as two device stages, and make each kernel arm independently selectable. Reuse only the verified affine-Q4 primitives from the kernel spike; do not inherit its conclusions or wire its monolithic API.
**Tech Stack:** Python, MLX `mx.fast.metal_kernel`, optional native MLX extension after proof, pytest, checked-in micro/end-to-end benchmarks.
**Assumptions:** Assumes Hy3 Q4/group-64 and the manifest v1 nine-component order; this will NOT accept arbitrary quantization. Assumes a prototype can use JIT Metal for proof; promotion will NOT keep per-call JIT construction if its host cost erases the gain.

---

## Files

- Create `mtplx/kernels/moe_record_q4.py`: v1 layout validation and two-stage kernel API.
- Modify `mtplx/kernels/__init__.py`: explicit experimental export.
- Modify `mtplx/models/expert_mlx.py`: opt-in execution arm and weighted reduction boundary.
- Modify `mtplx/models/hy3_mlx.py`: pass scores into a fused-combine-capable switch without changing the fallback.
- Create `scripts/benchmark_hy3_record_q4.py` and `tests/test_moe_record_q4.py`.
- Extend streamed model tests for cache, pin, repeated-expert, and fallback behavior.

### Task 1: Lock the v1 record and combine contracts

- [ ] Write failing tests for exact component offsets, shapes, dtypes, total 10,616,832-byte coverage, Q4/group-64 rejection, top-8 shape, activation-dtype scores, and deterministic fallback.
- [ ] Verify RED: `pytest -q tests/test_moe_record_q4.py -k contract` fails because the module does not exist.
- [ ] Implement immutable `RecordQ4Layout` and `validate_hy3_record_q4(record, spec)`; no Metal code yet.
- [ ] Verify GREEN and commit `test(hy3): lock record-native Q4 contract`.

### Task 2: Implement independent Stage A and Stage B kernels

- [ ] Write failing parity tests against `mx.gather_qmm` for BF16/FP16, permuted slot indices, repeated experts, and B=1/2/4/8.
- [ ] Verify RED for missing `record_gate_up_swiglu_q4` and `record_down_weighted_reduce_q4`.
- [ ] Implement Stage A as specialized gate/up QMV + SwiGLU producing `[assignments,1536]`.
- [ ] Implement Stage B with output tiles over 4096, looping over the token's eight assignments, applying activation-dtype router scores, and reducing in the accepted order to `[tokens,4096]`.
- [ ] Keep a stock oracle in the test module; production fallback calls the current component-bank path.
- [ ] Verify bit-exact or declared zero-tolerance parity before any timing; mutation-test score order and one packed nibble.
- [ ] Commit `feat(hy3): add record-native weighted MoE kernels`.

### Task 3: Integrate an opt-in execution arm

- [ ] Write failing integration tests proving the arm is off by default, preserves router authority, releases pins after errors, rejects unsupported records, and falls back without changing logits.
- [ ] Add an explicit execution selector such as `expert_q4_backend="component"|"record-native"`; do not overload `slot_layout`.
- [ ] Extend the switch interface with `run_weighted(x, indices, scores)` only for the supported Hy3 path. Keep `SparseMLP`'s existing multiply/sum fallback unchanged.
- [ ] Preserve the shared expert as a separate first arm; evaluate shared final-add integration only after routed parity.
- [ ] Verify target tests, full streamed tests, and commit `perf(hy3): wire opt-in record-native MoE execution`.

### Task 4: Attribute and gate the mechanism

- [ ] Benchmark stock component banks, specialized Stage A only, Stage B weighted reduction, and the complete arm independently.
- [ ] Use a true dependency chain and end-to-end generation; do not price independent fake layers as pipeline speed.
- [ ] Run all-hit and controlled-miss B=1/2/4/8 traces.
- [ ] Retain the arm only after repeated >=5% end-to-end improvement with parity and equal bytes.
- [ ] If it loses, leave the tested primitive unwired/experimental and document the negative result.
- [ ] Push `experiment/hy3-record-native-exec` and open a draft PR against `experiment/hy3-cache-scheduling`, linking #30.
