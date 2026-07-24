# GLM-5.2 Q1T Fused rANS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans and superpowers-optimized:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install `rans32x-v1` as the GLM-5.2 Q1T MoE weight representation and exceed 10 token/s without any decoded weight bank or expert cache.

**Architecture:** A GLM-only artifact stores separate page-aligned rANS containers for gate/up/down packed bytes and scales. Each container is tiled by 32 output rows; rANS lane `l` owns the unchanged t158 byte stream for output row `tile * 32 + l`. A fused Metal projection dispatch maps routed expert IDs directly to those lane streams, decodes packed/scales in registers, performs the unchanged t158 LUT and FP32 accumulation, and returns only the final projection output. Construction verifies the complete artifact, installs fixed GLM geometry and logical-M routes, runs exact output self-checks, and binds every routed layer directly.

**Tech Stack:** Python 3.13, pytest, NumPy artifact tooling, MLX 0.31.2, Metal kernels.

**Assumptions:**

- Assumes model key `glm52-expert-q1t`, codec `t158`, 75 routed layers 3-77, 256 experts, top-8, hidden 6144, expert hidden 2048; every other model key, codec, or geometry fails before model execution.
- Assumes 32-output tiles because Apple SIMDgroups and `rans32x-v1` both use 32 lanes; the real-shape microbenchmark must reject this geometry if it loses to an unchanged `shadow_gather_mm` control.
- Assumes the new artifact is written to new paths and leaves existing raw, streamed-rANS, and banked artifacts unchanged.
- Assumes the fixed benchmark contract uses logical M in `{1, 2, 3, 4, 1024}`; no unsupported M silently falls back.

**Consolidated-baseline audit (2026-07-19):** The implementation branch is
based at `57352c1`; the current consolidated GLM line is
`scratch/merge-to-default` at `339c9ea` (30 commits ahead). New fused files,
`resident_loader.py`, `runtime.py`, and the benchmark script apply there
cleanly. `expert_runtime.py` requires one explicit manual port because
`0b0c6ab` added derived expert-cache planning; the fused port must keep that
policy unchanged for stock layouts, exclude `fused-rans` from it, and return a
zero-transient/zero-cache plan. `76b9a60` Q1 resume hardening and `fff6a23`
parts-aware manifests remain authoritative. No wholesale merge or older
execution-lane flag is part of this work.

---

## File structure

- `tests/test_glm52_q1t_fused_rans.py`: fused projection and three-projection parity, output-only, no-decode-call, repeated/arbitrary expert routing, and real-shape M tests.
- `tests/test_glm52_q1t_fused_rans_construction.py`: GLM-only installation, rejection, self-check, zero-cache, and no-`MlxComponentBank` gates.
- `mtplx/glm52_q1t_rans_artifact.py`: GLM-only manifest, tiled component encoder, strict loader, integrity verification, and non-overwriting conversion boundary.
- `mtplx/kernels/glm52_q1t_fused_rans.py`: fixed-geometry projection binder and direct rANS->t158->FP32 matmul Metal dispatch.
- `mtplx/models/glm52_q1t_fused_rans.py`: mapped component ownership, construction self-check, fused GLU switch, GLM runtime owner, and layer binder.
- `scripts/convert_glm52_q1t_fused_rans.py`: resumable conversion from the existing Q1T artifact to a distinct GLM fused-rANS artifact.
- `scripts/benchmark_glm52_q1t_fused_rans.py`: focused real-record geometry and materialization audit.
- `mtplx/expert_runtime.py`, `mtplx/resident_loader.py`, `mtplx/runtime.py`, `scripts/benchmark_q2_mtp_depth_matrix.py`: exact-lane construction routing only; stock and other family routes remain unchanged.

### Task 1: First RED fused projection parity test

**Files:**

- Create: `tests/test_glm52_q1t_fused_rans.py`

**Security flag:** none

- [x] Encode deterministic t158 packed/scales arrays for three experts and dimensions divisible by 64.
- [x] Tile each component into `[expert, output_tile, byte, lane]` rANS32x streams so every lane owns one output row.
- [x] Route expert IDs `[2, 0, 2, 1]`, call the not-yet-implemented fused projection, and require `mx.array_equal` against unchanged `shadow_gather_mm`.
- [x] Assert the fused result shape is only `(assignments, out_dim)`.
- [x] Run `uv run --frozen pytest -q tests/test_glm52_q1t_fused_rans.py -k projection_parity` and observe failure because `mtplx.kernels.glm52_q1t_fused_rans` does not exist.

### Task 2: Fused rANS t158 projection kernel

**Files:**

- Create: `mtplx/kernels/glm52_q1t_fused_rans.py`
- Modify: `tests/test_glm52_q1t_fused_rans.py`

**Security flag:** none

- [x] Add a construction binder accepting mapped packed/scales payloads, directories, derived tables, fixed `in_dim`, `out_dim`, `expert_count=256`, and `output_tile=32`.
- [x] Validate every invariant in the binder and return a direct callable with no codec/shape/eligibility checks in `__call__`.
- [x] In one dispatch, let each thread own one output row, initialize packed/scales rANS states, decode scale bits and 13 t158 bytes per 64 inputs, use the existing 1215-entry t158 LUT, and preserve `gd = fma(..., gd)` followed by `acc += scale * gd`.
- [x] Return only `(assignments, out_dim)` in the input dtype.
- [x] Run the focused test and require bitwise equality.
- [x] Add M1/M2/M3/M4 repeated/arbitrary expert parity cases and prove monkeypatching `expert_rans_metal.decode_component` and `decode_container` to raise does not affect the fused call.
- [x] Run `uv run --frozen pytest -q tests/test_glm52_q1t_fused_rans.py` and require all cases to pass under the GPU guard.

### Task 3: GLM-specific page-aligned fused artifact

**Files:**

- Create: `mtplx/glm52_q1t_rans_artifact.py`
- Create: `scripts/convert_glm52_q1t_fused_rans.py`
- Modify: `tests/test_glm52_q1t_fused_rans.py`

**Security flag:** none

- [x] Add failing schema/conversion tests for six components per layer, separate page-aligned extents, `rans32x-v1`, t158, 32 lanes, complete layer coverage, exact source identity, component hashes, and refusal to overwrite any output.
- [x] Run the schema tests and observe the missing API failure.
- [x] Implement a manifest whose component metadata binds container, frequency, directory, payload, guard, tile, row-stride, and projection geometry offsets without decoding weights.
- [x] Convert Q1T records into lane-owned 32-row tiles while preserving every output row's original packed/scales byte order.
- [x] Make resumption accept only verified complete component extents and write the final manifest atomically after all 75 layers pass structural verification.
- [x] Run the focused artifact tests and require exact source-byte round trips in test-only CPU reference decoding.

### Task 4: GLM-only construction route

**Files:**

- Create: `mtplx/models/glm52_q1t_fused_rans.py`
- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/resident_loader.py`
- Modify: `mtplx/runtime.py`
- Modify: `scripts/benchmark_q2_mtp_depth_matrix.py`
- Create: `tests/test_glm52_q1t_fused_rans_construction.py`

**Security flag:** none

**Does NOT cover:** Hy3, GLM Q2/Q4, B1, generic component-bank execution, or stock execution lanes.

- [ ] Add failing tests that install only for `glm52-expert-q1t` and reject Hy3, GLM Q2/Q4, B1, wrong model/config identity, wrong codec, missing layers/components, incompatible shapes, wrong tile geometry, corrupt hashes, and failed exact self-checks.
- [ ] Change the GLM-Q1T fused configuration contract to require the fused banked manifest, `banked_codec=rans32x-v1`, zero cache bytes, zero transient slots, no streamed codec, no islands, and no component-bank slot allocation.
- [ ] Map each page-aligned component directly with `mmap_u32`, derive only the small rANS lookup tables at construction, and retain mapped payload/directory/table ownership per layer.
- [ ] Run stored final-output self-check vectors through the fused kernel; compare exact output bytes and fail before returning the model.
- [ ] Bind a `Glm52Q1TFusedRansSwitchGLU` into all 75 routed layers and install direct logical-M routes for `{1,2,3,4,1024}`.
- [ ] Keep `construct_resident_model` stock behavior unchanged by passing the GLM binder only from the exact execution lane.
- [ ] Run the focused construction tests and unchanged GLM/Hy3 model-family tests.

### Task 5: Zero decoded-cache and hot-path proof

**Files:**

- Modify: `tests/test_glm52_q1t_fused_rans_construction.py`
- Modify: `tests/test_glm52_q1t_fused_rans.py`

**Security flag:** none

- [ ] Add a failing memory-plan test requiring `decoded_expert_cache_bytes == 0`, no `MlxComponentBank`, no `mx.zeros` weight allocation, and no `PositionalExpertReader` construction.
- [ ] Add source/behavior tests proving the installed switch directly invokes its fixed projection routes and contains no environment read, counter, eligibility branch, fallback, retry, standalone decoder, NumPy decoded array, or reconstructed component view.
- [ ] Make the runtime snapshot report mapped compressed bytes, table bytes, decoded cache bytes zero, and construction/self-check durations without per-token counters.
- [ ] Run `python3 -m pytest -q tests/test_glm52_q1t_fused_rans_construction.py tests/test_glm52_q1t_fused_rans.py`.

### Task 6: Real-shape geometry and real-record gate

**Files:**

- Create: `scripts/benchmark_glm52_q1t_fused_rans.py`
- Modify: `tests/test_glm52_q1t_fused_rans.py`

**Security flag:** none

- [ ] Measure actual packed/scales row strides and component extents from the fused artifact rather than copying shadow decoder geometry.
- [ ] Benchmark candidate output-tile/threadgroup ownership at gate/up `(6144 -> 2048)` and down `(2048 -> 6144)` for assignment counts 8, 16, 24, and 32.
- [ ] Compare every arm bitwise to unchanged `shadow_gather_mm`, record median/p95 latency, bytes touched, grid/threadgroup geometry, and reject any non-bitwise arm.
- [ ] Audit MLX allocations and kernel outputs to prove no decoded weight record or bank is emitted.
- [ ] Select geometry only from the measured real-record result and save the immutable JSON receipt.

### Task 7: Isolated end-to-end acceptance benchmark

**Files:**

- Modify: `docs/issues/issue51-glm52-q1t-over-5.md`
- Modify: `docs/plans/2026-07-19-glm52-q1t-over-10.md`

**Security flag:** none

- [ ] Run all unit, parity, construction, memory, model-family, Ruff, and compile gates before taking the GPU lane.
- [ ] Acquire `/tmp/mtplx-gpu-exclusive.lock` with the repository's `fcntl` guard, capture the exact Qwen model state, stop Qwen only inside the guarded window, and restore/verify it before releasing the lock.
- [ ] Run the real-record fused-kernel microbenchmark and materialization audit.
- [ ] Run unchanged control and fused-rANS GLM-5.2 Q1T at exactly 1024 input, 1024 requested output, depth 3, with matched prompt/sampler/MTP artifacts.
- [ ] Require exact emitted-token hash/parity, two fused rows above 10 token/s, peak memory below 96 GiB, decoded cache bytes zero, and no fallback/validation error.
- [ ] Record construction time, peak memory, token SHA-256, acceptance statistics, throughput, commands, artifact hashes, and Qwen restoration evidence.

## Measured experiment ledger

- Direct device-table, one-state baseline: exact across 72 real-shape arms; selected 64 threads for gate/up/down; AR8 total 1.434584 ms/layer and 9.294 token/s 75-layer compute-only ceiling. This remains the restored code baseline and is not qualified for the binding >10 token/s target.
- Packed slot-transition table: exact, but AR8 total 1.460521 ms/layer and 9.129 token/s ceiling. Rejected and fully reverted.
- Combined gate/up decode plus SwiGLU: exact, but AR8 total 1.502793 ms/layer and 8.872 token/s ceiling, with larger verify-count regressions. Rejected and fully removed.
- Two alternating rANS states per output: exact across 72 arms, zero decoded outputs, artifact SHA-256 `fa08dc078af75a79453ae0a4f306b1e1788f6ebb82a3f58c81e950f511d5a425`; best AR8 geometry still totaled 1.451771 ms/layer and 9.184 token/s. Rejected and fully reverted. Immutable report SHA-256: `ebd03d42ef6b166091500e4c46bf4bdc5e9cd9c24e7df6db1cf6b8cd1db6ce7b`.
- GLM-only bounded artifact encoding is byte-identical across record chunk sizes. The real layer-3 conversion fell from 55,270,162,432 to 18,826,625,024 bytes maximum RSS while retaining full source-record hashing.
- GLM-specific 9-bit rANS scale: exact across the real layer-3 matrix with zero decoded outputs, but selected AR8 totaled 1.5590205 ms/layer (8.55 token/s ceiling) and the best AR8 arm totaled 1.428646 ms/layer (9.333 token/s). The 2,054,750,208-byte probe artifact has SHA-256 `c9ad91338e651f871a61df74a99858cc37cfadfa2414c63a57eb59a3a49648f0`; immutable report file SHA-256 is `594d480bef5f7839b3a849a0e3afdc10e7f5decaef390741d73244a97917dabb`. Rejected and fully reverted from code, construction, CLI, and tests; the probe artifact remains separate and immutable.
- Four-byte chunk-swizzled packed rANS streams: exact across all 72 real layer-3 arms with zero decoded outputs, but selected AR8 totaled 1.491688 ms/layer (8.938 token/s ceiling) and the best AR8 arm totaled 1.4673335 ms/layer (9.087 token/s). The 2,063,876,096-byte separate probe artifact has SHA-256 `51e1c7389942475d1cac077d8996044e9c01a08f1c482751bc1385e81e0a3ff4`; immutable report file SHA-256 is `42407b2617c9ba9a01abea31a6ed047d0acafa87beb31db704dcc9f0eb9966dc`. Rejected and fully reverted from code, construction, CLI, and tests; the probe artifact remains separate and immutable.
- Scale-high decode scheduled after packed byte 0: exact across all 72 real layer-3 arms with zero decoded outputs, but selected AR8 totaled 1.4414375 ms/layer (9.250 token/s ceiling) and the best AR8 arm totaled 1.429125 ms/layer (9.330 token/s). Immutable report file SHA-256 is `2937d4b2281b25d45577f0d38af6e4d4743587a404b07526f318107aaa3fa531`. Rejected and fully reverted from kernel, construction, benchmark CLI, and tests.

## Self-review

- The plan covers fused decode/multiply, separate projection containers, register/threadgroup-only decoded lifetime, exact arithmetic, GLM-only construction, fail-closed validation, zero cache, no generic fallback, real geometry measurement, microbenchmark, and guarded end-to-end evidence.
- The only `v1` reference is the user-required `rans32x-v1` codec and the new artifact schema name; it is not a scope reduction.
- Existing dirty work is preserved. No commit step is included because the worktree already contains user-owned changes and the user did not request a commit.
