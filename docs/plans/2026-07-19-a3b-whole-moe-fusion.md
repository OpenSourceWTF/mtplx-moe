# A3B Whole-MoE Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a three-dispatch, correct-by-construction whole-MoE route for
exact Qwen3.6-35B-A3B target M1/M2 and MTP M1, then determine its isolated
natural K1/TGY4 value against the unchanged accepted stack.

**Architecture:** Stage 1 fuses router projection, precise softmax, exact top-8
normalization, and the shared scalar gate. Stage 2 computes eight selected
routed plus one shared gate/up projection and stores only BF16 `[M,9,512]`.
Stage 3 performs all down projections, routed score reduction, shared sigmoid
scaling, and the final BF16 add directly into `[M,2048]`. External model facts
are validated once after the accepted packing step has established 40 target
plus one MTP packed storage owners. Construction runs all-binding component
self-checks, prebinds fixed kernels, and executes an atomic load-time full-graph
probe before retaining all 41 routes. Each request separately certifies its
exact contiguous-dense compiled state before prompt work, and the post-prefill
route must match that certificate before decode. Installed execution routes
only on attention phase and logical M.

**Tech stack:** Python 3.12, MLX, `mx.fast.metal_kernel`, Metal C++, pytest,
Ruff, workspace A3B benchmark harness.

**Assumptions:**

- Assumes the accepted-stack base remains
  `8db96d65907eea9201f7dfcf2f42fe8c4c7b298a` — do not continue if the worktree
  has moved or acquired unrelated edits.
- Assumes target projection storage is q8/group64 router and scalar gate plus
  q4/group64 routed/shared projections — it will not install on another
  checkpoint layout.
- Assumes MTP storage is dense BF16 router/shared/scalar plus q4/group32 routed
  experts — it will not select a generic runtime variant.
- Assumes accepted packed-MoE construction has already replaced gate/up for all
  40 target blocks and the MTP block — exact whole-MoE binding consumes those
  owners while leaving the packing step and storage ownership intact.
- Assumes K1 construction guarantees target M1/M2 and MTP M1 during measured
  decode — unsupported measured shapes are configuration errors, not stock
  fallbacks.
- Assumes prefill is explicitly bound to the accepted packed-stock call and the
  request remains on `contiguous_dense_decode` — the fused kernels do not
  support prompt rows, session-bank restores, vision splices, or repaged cache
  layouts.
- Assumes MTP adapters are absent; adapter or adapter-merge options fail at the
  load boundary before packed storage is bound.
- Assumes Metal correctness and benchmark runs can acquire the exclusive GPU
  lock — never overlap them with the GDN candidate or unrelated GLM/Qwen work.

---

## File structure

- Create `mtplx/a3b_whole_moe.py`: flag read, adapter/request guards, exact
  checkpoint and packed-owner contract, immutable 40-target/one-MTP binding
  descriptors, prebound kernel routes, prepare/self-check/install transaction,
  phase/M route table, construction report, and checked reference helpers.
- Create `mtplx/kernels/a3b_whole_moe.py`: fixed Stage 1/2/3 Metal source,
  cached kernel builders, and direct target-M1/target-M2/MTP-M1 entrypoints.
- Modify `mtplx/runtime.py`: reject adapters, perform accepted packing after MTP
  injection, prepare exact whole-MoE bindings from those packed owners, prepare
  the row router only when no whole route exists, construct the compiled
  factory/runtime, and install atomically through a load-time full-graph probe.
- Extend the whole-MoE construction self-check: direct three-stage target M1/M2
  checks over every target binding plus MTP M1, with component-specific limits
  consumed only at construction.
- Modify `mtplx/a3b_compiled_target_prefix.py`: provide the minimum atomic
  load-compatibility probe plus a request-specific exact-geometry certificate;
  invoke direct compiled target M2/M1 without mirror commits or per-dispatch
  instrumentation.
- Create `tests/test_a3b_whole_moe.py`: construction, route, source, launch,
  arithmetic, materialization, packing-before-binding, row-router conditional
  ownership, all-40 component checks, atomic rollback, request rejections,
  exact synthetic geometry, and certificate matching.
- Modify `tests/test_a3b_compiled_target_prefix.py`: compiled graph acceptance.
- Create workspace-only
  `bench/a3b/run_a3b_174_whole_moe_fusion.py`: isolated natural K1 C/X/C/X
  runner copied from the accepted-stack runner with only candidate flag,
  commit, and installation gates changed.
- Create workspace-only `bench/a3b/test_whole_moe_fusion_runner.py` and modify
  `bench/a3b/test_harness_integrity.py`: runner, engagement, exact request
  certificate, and warmup/preflight/measurement-order gates.
- Modify workspace-only `bench/a3b/OPTIMIZATION_LEDGER.md` only after a real
  result exists.

## Task 1: Exact construction contract and immutable bindings

**Files:**

- Create: `mtplx/a3b_whole_moe.py`
- Create: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** Metal execution, installation, prefill routing, compiled
capture, or benchmarks.

- [ ] **Step 1: Write flag-off and exact-plan tests**

```python
def test_flag_off_returns_no_plan_and_preserves_all_block_classes(monkeypatch):
    model = make_exact_a3b_model()
    original = tuple(type(block) for block in all_sparse_blocks(model))
    monkeypatch.delenv("MTPLX_A3B_WHOLE_MOE_FUSION", raising=False)
    assert prepare_a3b_whole_moe(model, config=exact_a3b_config()) is None
    assert tuple(type(block) for block in all_sparse_blocks(model)) == original


def test_exact_checkpoint_builds_40_target_and_one_mtp_binding(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model = make_exact_packed_a3b_model()
    plan = prepare_a3b_whole_moe(model, config=exact_a3b_config())
    assert len(plan.target_bindings) == 40
    assert len(plan.mtp_bindings) == 1
    assert {binding.variant for binding in plan.target_bindings} == {
        "target_q8g64_q4g64"
    }
    assert plan.mtp_bindings[0].variant == "mtp_dense_q4g32_dense"
    assert all(binding.routed_gate_up is not None for binding in plan.target_bindings)
    assert plan.mtp_bindings[0].routed_gate_up is not None
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k 'flag_off or exact_checkpoint'
```

Expected: collection fails because `mtplx.a3b_whole_moe` does not exist.

- [ ] **Step 3: Add immutable plan/binding types and flag-off preparation**

```python
@dataclass(frozen=True)
class A3BWholeMoeWeights:
    variant: Literal["target_q8g64_q4g64", "mtp_dense_q4g32_dense"]
    router: ProjectionStorage
    routed_gate_up: ProjectionStorage
    routed_down: ProjectionStorage
    shared_gate_up: ProjectionStorage
    shared_down: ProjectionStorage
    shared_scalar_gate: ProjectionStorage


@dataclass(frozen=True)
class A3BWholeMoeInstallPlan:
    target_bindings: tuple[A3BWholeMoeBinding, ...]
    mtp_bindings: tuple[A3BWholeMoeBinding, ...]


def a3b_whole_moe_enabled() -> bool:
    return os.environ.get("MTPLX_A3B_WHOLE_MOE_FUSION", "").strip().lower() in {
        "1", "true", "on", "yes"
    }
```

`ProjectionStorage` holds direct model-owned arrays and fixed metadata from the
already accepted packed owner; it does not copy, repack, or validate any
intermediate created by the route.

- [ ] **Step 4: Add parameterized construction-failure tests**

```python
@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda c, m: c["text_config"].update(hidden_size=4096), "hidden size"),
        (lambda c, m: c["text_config"].update(num_hidden_layers=39), "40 target"),
        (lambda c, m: c["text_config"].update(num_experts=128), "256 experts"),
        (lambda c, m: c["text_config"].update(num_experts_per_tok=4), "top-k 8"),
        (lambda c, m: c["text_config"].update(norm_topk_prob=False), "normalization"),
        (lambda c, m: setattr(target_blocks(m)[0].gate, "group_size", 32), "q8/group64"),
        (lambda c, m: setattr(mtp_block(m).switch_mlp.gate_up_proj, "group_size", 64), "q4/group32"),
        (lambda c, m: unpack_gate_up(target_blocks(m)[0]), "packed"),
        (lambda c, m: setattr(target_blocks(m)[0], "sharding_group", object()), "sharding"),
    ],
)
def test_external_contract_mismatch_fails_before_install(monkeypatch, mutation, expected):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    config, model = exact_a3b_config(), make_exact_a3b_model()
    mutation(config, model)
    with pytest.raises(A3BWholeMoeConfigError, match=expected):
        prepare_a3b_whole_moe(model, config=config)
    assert not any(is_installed(block) for block in all_sparse_blocks(model))
```

- [ ] **Step 5: Implement complete external/model storage validation**

Validate exactly the shapes and dtypes from the design spec, including target
and MTP packed gate/up/down orientations, exact packed owner classes and split,
both affine metadata arrays, dense MTP weights, 40-layer attention topology,
one MTP block, and lack of route conflicts. Return fixed
`A3BWholeMoeWeights` bindings only after all 40 target blocks and the MTP block
pass. Packing is a required predecessor and is never performed or bypassed by
this preparation function.

- [ ] **Step 6: Run construction tests GREEN and commit**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k 'flag_off or checkpoint or contract'
git add mtplx/a3b_whole_moe.py tests/test_a3b_whole_moe.py
git commit -m "perf(a3b): define whole-MoE construction contract"
```

## Task 2: Fixed kernel sources and launch geometry

**Files:**

- Create: `mtplx/kernels/a3b_whole_moe.py`
- Modify: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** arithmetic parity, model installation, or runtime routing.

- [ ] **Step 1: Write source and fake-launch RED tests**

```python
@pytest.mark.parametrize("source", all_whole_moe_sources())
def test_fixed_sources_have_no_runtime_validation_or_fallback(source):
    forbidden = (
        "getenv", "dtype", "shape", "eligible", "fallback", "lane_disabled",
        "record_", "counter", "threadgroup_barrier",
    )
    assert not any(word in source for word in forbidden)


def test_target_m2_stage2_is_row_paired_and_fixed_to_288_threadgroups(monkeypatch):
    launch = capture_metal_launch(monkeypatch)
    target_m2_stage2(exact_target_stage2_inputs())
    assert launch.grid == (288 * 128, 1, 1)
    assert launch.threadgroup == (128, 1, 1)
    assert launch.output_shapes == [(2, 9, 512)]
    assert launch.output_dtypes == [mx.bfloat16]
```

- [ ] **Step 2: Run source/launch tests RED**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k 'fixed_sources or launch or row_paired'
```

Expected: fixed entrypoints are not importable.

- [ ] **Step 3: Create fixed constants, source functions, and builders**

```python
HIDDEN = 2048
EXPERTS = 256
TOP_K = 8
INTERMEDIATE = 512
ACTIVATION_SLOTS = 9
THREADS = 128
SIMDGROUPS = 4
COLUMNS_PER_SIMDGROUP = 4
STAGE2_THREADGROUPS = 9 * (512 // 16)
STAGE3_THREADGROUPS = 2048 // 16
```

Provide distinct cached kernels and direct functions:

```python
target_m1_stage1(...)
target_m2_stage1(...)
mtp_m1_stage1(...)
target_m1_stage2(...)
target_m2_stage2(...)
mtp_m1_stage2(...)
target_m1_stage3(...)
target_m2_stage3(...)
mtp_m1_stage3(...)
```

Every entrypoint reshapes with fixed constants and launches directly. No public
checked helper is used by installed execution.

- [ ] **Step 4: Assert exact launch table**

```python
EXPECTED = {
    "target_m1_stage1": ((256, 1, 1), (256, 1, 1)),
    "target_m2_stage1": ((512, 1, 1), (256, 1, 1)),
    "mtp_m1_stage1": ((256, 1, 1), (256, 1, 1)),
    "target_m1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
    "target_m2_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
    "mtp_m1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
    "target_m1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
    "target_m2_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
    "mtp_m1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
}
```

- [ ] **Step 5: Run source/launch tests GREEN and commit**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k 'source or launch or row_paired'
git add mtplx/kernels/a3b_whole_moe.py tests/test_a3b_whole_moe.py
git commit -m "perf(a3b): add fixed whole-MoE kernel geometry"
```

## Task 3: Stage 1 exact router/top-k arithmetic

**Files:**

- Modify: `mtplx/kernels/a3b_whole_moe.py`
- Modify: `mtplx/a3b_whole_moe.py`
- Modify: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** expert gate/up, down projections, installation, or
benchmarking.

- [ ] **Step 1: Write target M1/M2 and MTP M1 parity tests**

```python
@pytest.mark.parametrize("variant, rows", [
    ("target_q8g64_q4g64", 1),
    ("target_q8g64_q4g64", 2),
    ("mtp_dense_q4g32_dense", 1),
])
def test_stage1_matches_precise_stock_router_and_shared_gate(variant, rows):
    fixture = exact_stage1_fixture(variant=variant, rows=rows)
    candidate = run_stage1(fixture)
    expected = stock_stage1_reference(fixture)
    mx.eval(*candidate, *expected)
    assert mx.array_equal(candidate.expert_ids, expected.expert_ids).item()
    assert mx.array_equal(candidate.route_scores, expected.route_scores).item()
    assert mx.array_equal(candidate.shared_gate, expected.shared_gate).item()
```

- [ ] **Step 2: Run parity tests RED on an exclusive Metal lane**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k stage1_matches
```

Expected: Stage 1 returns incorrect/unimplemented outputs, not an infrastructure
or no-Metal failure. If the shared GPU lock is held, do not queue this command;
continue with static work and return when the lane is free.

- [ ] **Step 3: Implement target q8/group64 and MTP dense dot products**

Use one 256-thread row-owning group. Each lane computes one router logit with
the fixed storage variant, rounds it to BF16, and stores it in threadgroup
scratch. The same group computes the scalar shared-gate dot. Source variants
encode q8/group64 or dense BF16 directly; they do not branch on storage.

- [ ] **Step 4: Implement precise softmax and accepted top-8 ordering**

Reuse the accepted row-owned router's tie rule and reverse output order, but
compute precise softmax from the Stage-1 BF16 logits in FP32 before selection.
Normalize with the accepted sequential BF16 denominator and emit BF16 scores.

- [ ] **Step 5: Run Stage-1 parity GREEN and commit**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k stage1
git add mtplx/kernels/a3b_whole_moe.py mtplx/a3b_whole_moe.py tests/test_a3b_whole_moe.py
git commit -m "perf(a3b): fuse exact A3B router stage"
```

## Task 4: Stage 2 selected gate/up plus SwiGLU

**Files:**

- Modify: `mtplx/kernels/a3b_whole_moe.py`
- Modify: `mtplx/a3b_whole_moe.py`
- Modify: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** down projections, route reduction, shared scaling, or final
addition.

- [ ] **Step 1: Write exact BF16 activation parity tests**

```python
@pytest.mark.parametrize("variant, rows", [
    ("target_q8g64_q4g64", 1),
    ("target_q8g64_q4g64", 2),
    ("mtp_dense_q4g32_dense", 1),
])
def test_stage2_matches_nine_stock_swiglu_activations(variant, rows):
    fixture = exact_stage2_fixture(variant=variant, rows=rows)
    actual = run_stage2(fixture)
    expected = stock_selected_activation_reference(fixture)
    mx.eval(actual, expected)
    assert actual.shape == (rows, 9, 512)
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, expected).item()
```

- [ ] **Step 2: Run Stage-2 tests RED**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k stage2
```

- [ ] **Step 3: Implement target q4/group64 selected routed and shared tiles**

Each 128-thread group owns 16 activation columns for one expert slot. Four
SIMDgroups compute four columns each. Gate and up dot products use the exact
affine metadata and BF16 projection rounding, then `silu(gate) * up` in the
accepted order with one BF16 store. M2 computes both rows in the same tile with
at most 16 scalar dot accumulators per lane.

- [ ] **Step 4: Implement MTP q4/group32 routed plus dense shared tiles**

Use a distinct fixed source/entrypoint. Expert slots 0-7 address construction-
stacked q4/group32 routed arrays; slot 8 addresses prebound dense BF16 shared
arrays. The slot ownership is a fixed kernel design fact, not a storage
eligibility branch.

- [ ] **Step 5: Assert no large gate/up intermediate**

```python
def test_stage2_exposes_only_compact_activation_output(monkeypatch):
    launch = capture_metal_launch(monkeypatch)
    target_m2_stage2(exact_target_stage2_inputs())
    assert launch.output_shapes == [(2, 9, 512)]
    assert (2, 8, 1024) not in launch.output_shapes
    assert (2, 1024) not in launch.output_shapes
```

- [ ] **Step 6: Run Stage-2 parity GREEN and commit**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k stage2
git add mtplx/kernels/a3b_whole_moe.py mtplx/a3b_whole_moe.py tests/test_a3b_whole_moe.py
git commit -m "perf(a3b): fuse selected MoE activation stage"
```

## Task 5: Stage 3 down/reduce/shared/final add

**Files:**

- Modify: `mtplx/kernels/a3b_whole_moe.py`
- Modify: `mtplx/a3b_whole_moe.py`
- Modify: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** model class installation or compiled capture.

- [ ] **Step 1: Write final-output parity and materialization RED tests**

```python
@pytest.mark.parametrize("variant, rows", [
    ("target_q8g64_q4g64", 1),
    ("target_q8g64_q4g64", 2),
    ("mtp_dense_q4g32_dense", 1),
])
def test_stage3_matches_stock_down_reduce_shared_and_final_add(variant, rows):
    fixture = exact_stage3_fixture(variant=variant, rows=rows)
    actual = run_stage3(fixture)
    expected = stock_stage3_reference(fixture)
    mx.eval(actual, expected)
    assert actual.shape == (rows, 2048)
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, expected).item()


def test_stage3_has_only_final_output(monkeypatch):
    launch = capture_metal_launch(monkeypatch)
    target_m2_stage3(exact_target_stage3_inputs())
    assert launch.output_shapes == [(2, 2048)]
    assert (2, 8, 2048) not in launch.output_shapes
    assert launch.output_shapes.count((2, 2048)) == 1
```

- [ ] **Step 2: Run Stage-3 tests RED**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k stage3
```

- [ ] **Step 3: Implement output-column-owned target and MTP sources**

For each of 128 tiles, compute fixed 16 output columns. For routed slots 0-7,
round each down projection to BF16, round the route-score product to BF16, and
perform sequential BF16 additions in current selected order. Compute the shared
down result, BF16 sigmoid/multiply, and final BF16 addition before the only
output store. Process expert slots sequentially to cap M2 live accumulators at
16 per lane.

- [ ] **Step 4: Run Stage-3 parity GREEN and commit**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k stage3
git add mtplx/kernels/a3b_whole_moe.py mtplx/a3b_whole_moe.py tests/test_a3b_whole_moe.py
git commit -m "perf(a3b): fuse A3B MoE down and reduction stage"
```

## Task 6: Atomic installed route and runtime construction

**Files:**

- Modify: `mtplx/a3b_whole_moe.py`
- Modify: `mtplx/runtime.py`
- Modify: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** unsupported models, unsupported decode widths, or prefill
fusion. Those remain uninstalled or explicit stock construction routes.

- [ ] **Step 1: Write installation/route RED tests**

```python
def test_successful_install_atomically_replaces_all_41_blocks(monkeypatch):
    plan = exact_plan(monkeypatch)
    report = install_a3b_whole_moe(
        plan,
        exact_selfcheck_report(),
        compiled_preflight=passing_full_graph_preflight,
    )
    assert report["installation_status"] == "installed"
    assert sum(is_installed(block) for block in all_sparse_blocks(plan.model)) == 41


def test_selfcheck_failure_installs_nothing(monkeypatch):
    plan = exact_plan(monkeypatch)
    with pytest.raises(A3BWholeMoeConfigError, match="self-check"):
        install_a3b_whole_moe(
            plan,
            {"lanes": {"a3b_whole_moe_target_m1": "failed"}},
            compiled_preflight=unexpected_preflight,
        )
    assert not any(is_installed(block) for block in all_sparse_blocks(plan.model))


def test_full_graph_probe_failure_restores_all_41_original_classes(monkeypatch):
    plan = exact_plan(monkeypatch)
    original = tuple(type(block) for block in all_sparse_blocks(plan.model))
    with pytest.raises(A3BWholeMoeConfigError, match="full compiled"):
        install_a3b_whole_moe(
            plan,
            exact_selfcheck_report(),
            compiled_preflight=failing_full_graph_preflight,
        )
    assert tuple(type(block) for block in all_sparse_blocks(plan.model)) == original


def test_prefill_calls_prebound_packed_stock_while_small_rows_are_direct(monkeypatch):
    target, mtp = installed_blocks(monkeypatch)
    with attention_phase("prefill"):
        assert target.packed_stock_sentinel(INPUT) == target(INPUT)
    with attention_phase("ar_decode"):
        assert target.m1_sentinel(INPUT) == target(INPUT)
        assert mtp.m1_sentinel(INPUT) == mtp(INPUT)
    with attention_phase("decode_verify"):
        assert target.m2_sentinel(INPUT_M2) == target(INPUT_M2)
```

- [ ] **Step 2: Write hot-source prohibition test**

```python
def test_installed_call_has_only_phase_and_logical_m_routing():
    source = inspect.getsource(_installed_a3b_whole_moe_call)
    forbidden = (
        "os.environ", "dtype", "bits", "group_size", "eligible", "selfcheck",
        "installed", "fallback", "lane_disabled", "try:", "except",
        "mx.softmax", "switch_mlp", "shared_expert",
    )
    assert not any(word in source for word in forbidden)
```

- [ ] **Step 3: Run installation tests RED**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k 'install or prefill or hot_source or runtime_constructs'
```

- [ ] **Step 4: Implement prebound route descriptors and atomic class swap**

```python
@dataclass(frozen=True)
class InstalledA3BWholeMoeRoute:
    stock_call: Callable[..., mx.array]  # captured after accepted packing
    m1_call: Callable[[mx.array], mx.array]
    m2_call: Callable[[mx.array], mx.array] | None


def _installed_a3b_whole_moe_call(self, value):
    route = type(self)._mtplx_a3b_whole_moe_route
    phase = current_attention_phase()
    if phase == "prefill":
        return route.stock_call(self, value)
    rows = math.prod(value.shape[:-1])
    if rows == 1:
        return route.m1_call(value)
    if rows == 2 and route.m2_call is not None:
        return route.m2_call(value)
    raise A3BWholeMoeRouteError(
        f"installed A3B whole-MoE route has no {phase} M{rows} entrypoint"
    )
```

The M1/M2 callables close over fixed model-owned packed arrays and call Stage 1,
Stage 2, and Stage 3 directly. The prefill callable is captured after accepted
packing, so it is the packed-stock implementation. Installation prepares all
classes/callables first, tentatively swaps every block, runs direct compiled M2
and M1 full-graph calls, and restores all original classes on any mutation,
compile, execution, or output-ownership error.

- [ ] **Step 5: Wire construction order in `runtime.py`**

Reject MTP adapter/merge options before any storage mutation. After MTP
injection and checkpoint coverage, run accepted packed-projection construction
for all target and MTP blocks. Prepare the exact whole-MoE plan from those
packed owners. Prepare/install the row-owned router only when that whole plan is
absent; report the router as construction-superseded only in the whole-plan
case. Run shared and all-binding whole-MoE self-checks, prepare the GDN and
compiled target-prefix factories, construct the runtime, then atomically
install all 41 whole routes through the load-time full-graph compatibility
probe. Packing remains the required predecessor and storage owner.

- [ ] **Step 6: Run installation/runtime tests GREEN and commit**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py -k 'whole_moe or install or prefill or runtime_constructs'
git add mtplx/a3b_whole_moe.py mtplx/runtime.py tests/test_a3b_whole_moe.py
git commit -m "perf(a3b): install whole-MoE route by construction"
```

## Task 7: All-binding self-check, load probe, and request certificate

**Files:**

- Modify: `mtplx/a3b_whole_moe.py`
- Modify: `mtplx/a3b_compiled_target_prefix.py`
- Modify: `mtplx/generation.py`
- Modify: `tests/test_a3b_compiled_target_prefix.py`
- Modify: `tests/test_a3b_whole_moe.py`

**Security flag:** `none`

**Does NOT cover:** benchmark instrumentation or engagement counters.

- [ ] **Step 1: Write all-binding, load-probe, and certificate RED tests**

```python
def test_whole_moe_selfcheck_covers_all_40_target_bindings_and_mtp(monkeypatch):
    report, calls = run_recording_selfcheck_with_exact_whole_moe(monkeypatch)
    assert report["lanes"]["a3b_whole_moe_target_m1"] == "ok"
    assert report["lanes"]["a3b_whole_moe_target_m2"] == "ok"
    assert report["lanes"]["a3b_whole_moe_mtp_m1"] == "ok"
    assert calls.count("target_m1") == 40
    assert calls.count("target_m2") == 40
    assert calls.count("mtp_m1") == 1
    assert set(report["a3b_whole_moe_components"]["a3b_whole_moe_target_m2"]) == {
        "route_scores", "shared_gate", "activations", "output"
    }


def test_request_preflight_uses_exact_contiguous_dense_geometry(monkeypatch):
    proof = preflight_exact_request(prompt_tokens=181, max_tokens=4096)
    assert proof["prefill_layout"] == "contiguous_dense_decode"
    assert proof["growth_reserve_tokens"] == 4098
    assert proof["full_attention_key_shape"] == [1, 2, 4352, 256]
    assert len(proof["canonical_key"]) == 64


def test_actual_postprefill_route_requires_matching_certificate_before_decode():
    assert actual_route_matches_preflight_certificate()
```

- [ ] **Step 2: Run self-check/compiled tests RED**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py tests/test_a3b_compiled_target_prefix.py \
  -k 'selfcheck or preflight or certificate'
```

- [ ] **Step 3: Register exact construction-only self-check lanes**

Use deterministic BF16 fixtures and direct fixed entrypoints. Evaluate every
one of the 40 target bindings at both M1 and M2, then the MTP binding at M1.
Aggregate component maxima only after each binding has executed; a single
representative target block is insufficient. Require exact indices and the
declared component-specific limits for route scores, shared scalar gate,
Stage-2 activation, and final output. Do not alter the sampler.

- [ ] **Step 4: Add the atomic load-time full-graph compatibility probe**

After all component checks pass, prepare all 41 prebound route classes. Swap
them as one rollback-capable transaction and run a one-token/minimal-cache
probe through the compiled target M2 function followed by the compiled target
M1 function. Evaluate fixed output ownership and both lanes. On any failure,
restore all 41 original classes. This `prompt_tokens=1` probe establishes only
load compatibility; it must never be described as proof of the natural
request's compile specialization.

- [ ] **Step 5: Add the exact request preflight and post-prefill certificate gate**

Before prompt work, reject non-K1 target-prefix ownership, non-stock capture or
draft arithmetic, session-bank state, vision splices, adapters, and any layout
other than `contiguous_dense_decode`. Invoke the same contiguous-dense cache
factory used by real prefill, evaluate one token, then set the full-attention
offset to the exact prompt length. Install disposable fixed state with
`max_tokens + 2` reserve; verify the physical capacity for
`prompt_tokens + max_tokens + 2` with the 256-token allocation step.

Call compiled M2 and compiled M1 directly without `verify_m2`, `repair_m1`,
snapshot, capture-commit, or mirror-commit work. Record fixed output ownership
and ordered leaf shape/dtype signatures. Compute the canonical SHA-256 from the
hidden variant, fixed ordered state spec, M2/M1 input signatures, and the
ordered M2 state-leaf shape/dtype sequence. After actual prefill, construct the
real route, independently compute the same key, and fail before the decode loop
unless the request certificate exists and matches.

- [ ] **Step 6: Run focused and adjacent suites GREEN and commit**

```bash
uv run pytest -q \
  tests/test_a3b_whole_moe.py \
  tests/test_a3b_compiled_target_prefix.py \
  tests/test_qwen_row_owned_router.py \
  tests/test_moe_packed_projections.py \
  tests/test_graphbank_compiled_verify.py
git add mtplx/a3b_whole_moe.py mtplx/a3b_compiled_target_prefix.py \
  mtplx/generation.py tests/test_a3b_compiled_target_prefix.py \
  tests/test_a3b_whole_moe.py
git commit -m "test(a3b): prove whole-MoE compiled route"
```

## Task 8: Full verification and candidate commit

**Files:** all branch files changed above plus design/plan documents.

**Security flag:** `none`

**Does NOT cover:** a performance claim; direct tests are correctness gates.

- [ ] **Step 1: Inspect Metal compiler resource output out of band**

Confirm Stage 2 and Stage 3 M2 have no threadgroup barrier, no threadgroup
memory, no material register spill, and retain the planned 288/128
threadgroups. If a spill is proven, change only the derived 4-column/4-SIMDgroup
tile in a separate commit and repeat Stage-2/3 parity.

- [ ] **Step 2: Run required test suites**

```bash
uv run pytest -q tests/test_a3b_whole_moe.py
uv run pytest -q tests/test_qwen_row_owned_router.py tests/test_moe_packed_projections.py tests/test_qmm_kernels.py
uv run pytest -q tests/test_kernel_selfcheck.py tests/test_a3b_compiled_target_prefix.py
uv run pytest -q tests/test_graphbank_compiled_verify.py tests/test_generation.py
uv run pytest -q /Users/davidtai/projects/OpenSourceWTF/bench/a3b/test_harness_integrity.py
uv run ruff check mtplx/a3b_whole_moe.py mtplx/kernels/a3b_whole_moe.py \
  mtplx/runtime.py mtplx/a3b_compiled_target_prefix.py mtplx/generation.py \
  mtplx/server/openai.py tests/test_a3b_whole_moe.py \
  tests/test_a3b_compiled_target_prefix.py
git diff --check
```

Use the actual existing QMM/generation filenames returned by `rg --files
tests` if either named file differs; record the exact executed path rather than
silently omitting the suite.

- [ ] **Step 3: Audit the installed hot path**

```bash
rg -n 'getenv|environ|eligible|fallback|lane_disabled|selfcheck|record_|counter' \
  mtplx/a3b_whole_moe.py mtplx/kernels/a3b_whole_moe.py
```

Every match must be construction-only, reporting-only, or a checked public
test helper. No match may be reachable from the installed M1/M2 route or fixed
entrypoints.

- [ ] **Step 4: Commit the verified candidate**

```bash
git add docs/specs/2026-07-19-a3b-whole-moe-fusion-design.md \
  docs/plans/2026-07-19-a3b-whole-moe-fusion.md \
  mtplx/a3b_whole_moe.py mtplx/kernels/a3b_whole_moe.py \
  mtplx/runtime.py mtplx/a3b_compiled_target_prefix.py mtplx/generation.py \
  mtplx/server/openai.py tests/test_a3b_whole_moe.py \
  tests/test_a3b_compiled_target_prefix.py
git commit -m "perf(a3b): consolidate exact small-row MoE"
```

If Tasks 1-7 already produced commits, this final commit contains only docs or
verification corrections. Do not squash before the benchmark; the exact tested
HEAD is the benchmark identity.

## Task 9: Isolated natural K1 benchmark and verdict

**Files:**

- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/a3b/run_a3b_174_whole_moe_fusion.py`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/a3b/test_whole_moe_fusion_runner.py`
- Modify: `/Users/davidtai/projects/OpenSourceWTF/bench/a3b/test_harness_integrity.py`
- Modify after result: `/Users/davidtai/projects/OpenSourceWTF/bench/a3b/OPTIMIZATION_LEDGER.md`

**Security flag:** `none`

**Does NOT cover:** K0-K5 or prefill-size matrices unless K1 first passes the
promotion gate.

- [ ] **Step 1: Write runner gate tests RED**

```python
def test_whole_moe_runner_is_natural_k1_tgy4_cxcx():
    source = inspect.getsource(runner)
    assert "long_code_uncapped" in source
    assert '"C", "X", "C", "X"' in source
    assert "repeat" in source and "2" in source
    assert "TGY4" in source
    assert "1024/1024" not in source
    assert "TGY8" not in source


def test_candidate_gate_derives_work_without_hot_counters():
    cell = exact_candidate_cell()
    assert runner.derived_engagement(cell) == {
        "target_m1": cell["drafted"] - cell["accepted"],
        "target_m2": cell["verify_calls"],
        "mtp_m1": cell["drafted"],
    }
    source = inspect.getsource(runner.derived_engagement)
    assert "m1_calls" in source and "m2_calls" in source
    assert "drafted" in source
    assert "whole_moe_counter" not in source


def test_harness_runs_exact_token_budget_preflight_after_warmup_before_timing():
    source = A3B_ONESHOT.read_text()
    warmup = source.index("run(max_tokens_override=min(64, token_budget)")
    preflight = source.index(
        "whole_moe_request_preflight = ensure_a3b_whole_moe_request_preflight("
    )
    timed = source.index("for repeat in range(1, args.repeats + 1):")
    assert warmup < preflight < timed


def test_candidate_gate_requires_actual_route_certificate_match():
    cell, config = exact_candidate_cell_and_config()
    failures = runner._gate_cell(
        label="candidate", cell=cell, config=config, candidate=True
    )
    assert failures == []
    assert cell["compiled_verify"]["request_preflight_key"] == (
        cell["whole_moe_request_preflight"]["canonical_key"]
    )
```

- [ ] **Step 2: Implement runner from unchanged accepted-stack control**

Copy the accepted-stack natural K1 runner. Candidate adds only
`MTPLX_A3B_WHOLE_MOE_FUSION=1` and the exact construction/self-check gate.
Both arms retain TGY4 GDN post-conv, compiled reserve, captured repair, and all
accepted flags. Derive target M2 from `verify_calls`, target M1 from
`compiled_calls - verify_calls` / `drafted - accepted`, and MTP M1 from normal
draft statistics. Require zero fallback reasons, permanent-eager demotion, and
growth demotion.

For the natural workload, the one-shot harness first runs an unreported
`min(64, token_budget)` warmup. It then invokes the request preflight again with
the exact prompt and full natural token budget, before configuration emission
and before the timed repeat loop. Gate the reported proof's prompt length,
token budget, `max_tokens + 2` reserve, physical cache capacity, contiguous-
dense layout, output signatures, and canonical key. Each measured candidate
cell must report `request_preflight_status == "matched"` and the actual post-
prefill route key must equal that exact preflight key.

- [ ] **Step 3: Run harness tests GREEN and validate exact inputs**

```bash
uv run pytest -q \
  /Users/davidtai/projects/OpenSourceWTF/bench/a3b/test_whole_moe_fusion_runner.py \
  /Users/davidtai/projects/OpenSourceWTF/bench/a3b/test_harness_integrity.py
uv run python /Users/davidtai/projects/OpenSourceWTF/bench/a3b/run_a3b_174_whole_moe_fusion.py \
  --expected-candidate-commit "$(git rev-parse HEAD)" --validate-inputs
```

- [ ] **Step 4: Reconcile and acquire the exclusive GPU lane**

Check all A3B/GLM/Qwen processes, both lock sentinels, partial artifacts, and
Qwen service state. Do not queue behind an unrelated owner and do not run in
parallel with the GDN Metal/benchmark lane.

- [ ] **Step 5: Run process-isolated natural K1 C/X/C/X**

Use only the runner's full natural-stop command, K1, TGY4, repeat-2 cells, and
unchanged control. The exact full-token-budget certificate must already exist
after warmup and before either timed repeat; the actual route must match it
before decode in every candidate cell. Save JSONL and summary with
candidate/control commits, exact flags, request certificate, and match status.

- [ ] **Step 6: Apply the promotion verdict**

Promote only if TPS improves beyond arm spread, cycle ms decreases, every
correctness/configuration gate passes, and no stock execution is hidden. Record
the exact result and verdict in `OPTIMIZATION_LEDGER.md`. A wash/regression stays
isolated and is not added to the beneficial stack.

- [ ] **Step 7: Continue to K0-K5 only after a positive K1**

Extend fixed routes to the exact M1-M6 geometries required by K0-K5 using the
same TDD/arithmetic process. Then run natural K0-K5 and confirm explicit stock
prefill at 128, 512, 2048, 8192, and 16384. Keep each material geometry change
as an attributable commit and do not merge without approval.
