# Qwen4 M=2 Attention Hyper D/U Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task.

**Goal:** Remove the measured attention-side hyper D/U projection chain from exact batch-1/M=2 verification while preserving the stock grouped norm and all non-M2 routes.

**Architecture:** Construction binds the existing q4/g32 D/U callable to both attention and MLP hyper modules. The capture route preselects an attention wrapper that runs the unchanged stock norm and invokes the fixed D/U callable only for `(1, 2, 10240)` arrays; every other logical shape calls the stock module.

**Tech Stack:** Python, MLX 0.32.2, Metal kernels already in `mtplx/kernels/qwen4_hyper_fusion.py`, pytest, ruff.

**Assumptions:** The attention modules have the same validated q4/g32 geometry as the accepted MLP modules. This will not work for different hidden width, HC count, quantization, B>1, M other than 2, or partial 48-layer installation; those states remain stock or fail before installation.

---

## Files

- Modify `mtplx/qwen4_hyper_fusion.py`: validate, self-check, and bind all 48 attention modules plus all 48 MLP modules.
- Modify `mtplx/qwen4_capture.py`: construction-select and invoke the exact-shape attention wrapper.
- Modify `tests/test_qwen4_hyper_fusion.py`: lock complete 96-module construction behavior.
- Modify `tests/test_qwen4_capture.py`: lock exact M=2 routing and B>1 stock behavior.

### Task 1: Bind complete attention and MLP module sets

**Security flag:** none

**Does NOT cover:** It does not enable different geometry, partial model installation, M=3, or B>1.

- [x] Add attention modules to `_model()` and change the construction test to require 96 bindings, 48 attention installations, and 48 MLP installations.
- [x] Change the pre-install test to require validation, binding, and finite self-checks for both module sets before either set receives `_mtplx_m2_hyper_call`.
- [x] Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_qwen4_hyper_fusion.py::test_exact_m2_hyper_route_is_bound_once_at_construction \
  tests/test_qwen4_hyper_fusion.py::test_storage_and_exact_selfcheck_run_before_install
```

  Expected before implementation: failure because only MLP modules are bound.
- [x] Update `configure_qwen4_hyper_fusion()` to construct ordered attention/MLP module tuples, validate and bind all 96 modules, run all self-checks before installation, and report both counts.
- [x] Re-run the two tests; expected: pass.

### Task 2: Route exact M=2 attention hyper traffic

**Security flag:** none

**Does NOT cover:** B>1, M other than 2, prefill, or any shape other than `(1, 2, 10240)`; they call the stock module directly.

- [x] Add a failing wrapper test equivalent to:

```python
module = SimpleNamespace(
    hc_norm=lambda hidden: "normed",
    _mtplx_m2_hyper_call=lambda hidden, normed: ("fused", hidden, normed),
)
hidden = SimpleNamespace(shape=(1, 2, 10240))
assert capture._qwen4_m2_attn_hyper(module, hidden) == (
    "fused", hidden, "normed"
)
```

- [x] Add a failing B>1 test that monkeypatches the stock attention helper and requires `(2, 2, 10240)` to return the stock sentinel without invoking the fixed callable.
- [x] Add a failing installer test that installs all attention and MLP bindings and requires `_mtplx_capture_attn_hyper.__name__ == "_qwen4_m2_attn_hyper"`; add partial-install rejection coverage.
- [x] Run the new focused tests; expected before implementation: missing wrapper/route failures.
- [x] Implement `_qwen4_stock_attn_hyper`, `_qwen4_m2_attn_hyper`, exact complete-set construction selection, and one prebound `attn_hyper` local used by the capture loop.
- [x] Re-run focused tests; expected: pass.

### Task 3: Verify and promote only a production win

**Security flag:** none

- [x] Run focused tests and lint:

```bash
.venv/bin/python -m pytest -q \
  tests/test_qwen4_hyper_fusion.py \
  tests/test_qwen4_capture.py \
  tests/test_qwen38_resident_harness.py
.venv/bin/python -m ruff check \
  mtplx/qwen4_hyper_fusion.py mtplx/qwen4_capture.py \
  tests/test_qwen4_hyper_fusion.py tests/test_qwen4_capture.py
```

- [x] Under `/tmp/mtplx-gpu-exclusive.lock`, run the unchanged exact production harness:

```bash
.venv/bin/python scripts/qwen38_flash_next_oq4_harness.py \
  --depth 1 --verify-strategy capture_commit --warmup-runs 1 \
  --profile sustained --output /tmp/pr368-attn-hyper-production.json
```

- [x] Compare wall time and decode TPS to the four-run 56.66 tok/s boundary; require 1,024 generated tokens, temperature 1.0, stable nondegenerate output, valid acceptance, and zero repair.
- [ ] If it regresses, revert Tasks 1-2 and record the rejection. If it improves repeatably, commit the code and design/plan, cherry-pick to the PR worktree, push `moe/port/qwen38-flash-next-resident-q4`, update the PR benchmark history, and require all GitHub checks green.
