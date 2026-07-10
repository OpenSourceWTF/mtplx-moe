> **Repository scope:** This is a repository-wide `davidtai/MTPLX` issue.
> **Target branch:** `codex/moe-ssd-hy3-glm52`.
> **Implementation status:** only the pure cache-policy scaffold exists today;
> the runtime and model execution path described below are proposed work.

## Objective

Define the model-independent contract for resident-router-first MoE execution
and convert a user-defined total-memory threshold into fixed, transient, and
persistent allocations with no hidden overcommit.

This is a layer-local state machine. We cannot route every layer up front,
because layer `n`'s router input does not exist until layer `n - 1` completes.

```text
hidden[n]
  -> resident router[n]
  -> (top-k ids, unbiased route weights)
  -> LayerExpertSlotBank.plan(ids, phase)
  -> await loads for missing ids
  -> expert compute using resolved slot ids
  -> hidden[n + 1]
```

## Proposed files and API

- `mtplx/expert_streaming_models.py`
  - `ExpertStreamingModelSpec`: exact layer ranges, expert dimensions, Q4
    layout, router requirements, fixed bytes, KV bytes/token, and artifact pins.
  - `ExpertMemoryPlan`: immutable resolved allocation and safety margin.
  - `plan_expert_memory(...)`: pure integer sizing function.
  - built-in `hy3-q4` and `glm52-q4` descriptors.
- `mtplx/expert_streaming.py`
  - retain `LayerExpertSlotBank`, `RoutePlan`, `SlotLoad`, and `SlotEviction` as
    the policy boundary; add no disk or MLX dependency here.
- `scripts/plan_expert_memory.py`
  - print a machine-readable JSON plan before the full model is opened.
- `tests/test_expert_streaming_models.py`
  - exact layout arithmetic, boundary behavior, and invalid-plan tests.

Proposed planner signature:

```python
def plan_expert_memory(
    spec: ExpertStreamingModelSpec,
    *,
    total_limit_bytes: int,
    context_tokens: int,
    runtime_reserve_bytes: int = 0,
    expert_cache_limit_bytes: int | None = None,
    transient_slots: int | None = None,
    io_staging_bytes: int = 0,
    execution_workspace_bytes: int = 0,
) -> ExpertMemoryPlan: ...
```

Use this accounting, with integer bytes throughout:

```text
fixed = resident_model_bytes
      + context_tokens * kv_bytes_per_token
      + runtime_reserve_bytes
      + transient_slots * expert_record_bytes
      + io_staging_bytes
      + execution_workspace_bytes

persistent_budget = min(
    max(total_limit_bytes - fixed, 0),
    expert_cache_limit_bytes if set else infinity,
)

slots_per_layer = floor(
    persistent_budget / (routed_layer_count * expert_record_bytes)
)
slots_per_layer = min(slots_per_layer, expert_count)
```

The initial allocator should use the same persistent slot count for every
sparse layer. Uneven per-layer allocation can be added later from traces, but
must not complicate the first correctness path. One transient top-k scratch
bank is reused across sequential layers, so it is not multiplied by layer
count. Batched execution may require a larger explicit transient slot count or
microbatching; it may not allocate an unplanned overflow buffer.

## Exact descriptor invariants

### Hy3 Q4

- 79 routed layers, 192 experts/layer, top-8.
- Hidden size 4096; routed expert hidden size 1536.
- One affine Q4/group-64 expert record: 10.125 MiB.
- Routed corpus: 149.98 GiB; cold top-8 reads: 6.249 GiB/token.
- Fixed non-routed checkpoint bytes: 4,952,354,048 (4.612 GiB).
- Exact BF16 KV cache: 327,680 bytes/token.
- Reusable top-8 transient bank: 81 MiB.

### GLM-5.2 Q4

- 78 target layers; layers 0-2 dense and layers 3-77 routed (75 layers).
- 256 experts/layer, top-8; hidden size 6144; expert hidden size 2048.
- One affine Q4/group-64 expert record: 20.25 MiB.
- Routed corpus: 379.6875 GiB; cold top-8 reads: 11.865234 GiB/token.
- Fixed non-routed checkpoint bytes: 10,634,546,688 (9.904193 GiB).
- MLA+DSA cache: 95,232 bytes/token.
- Reusable top-8 transient bank: 162 MiB.

## Configuration semantics

- `memory_limit_bytes` is a hard ceiling for fixed model buffers, planned
  context state, runtime reserve, transient expert buffers, and persistent
  expert buffers owned by this runtime. It is not by itself a hard cap on
  macOS file cache or unrelated processes.
- `context_tokens` is required and means aggregate live KV tokens across all
  admitted sequences. Zero is an explicit load-only plan.
- Known staging/in-flight I/O and execution workspaces are explicit fixed
  costs; do not conceal them inside an optimistic zero-byte reserve.
- `expert_cache_limit_bytes` is an optional hard sub-cap, never a request to
  consume more than the total limit permits.
- Runtime reserve covers temporary compute allocations and external MLX use;
  expose a documented default, but always report it explicitly.
- Persistent slots may resolve to zero. This is slow but valid: misses use the
  transient bank and correctness is unchanged.
- A request for longer context than planned must be rejected or trigger an
  explicit re-plan/reload; it must not silently consume the cache margin.
- Reconcile this setting with `MTPLX_MEMORY_LIMIT_BYTES`; reject conflicting
  limits, call `mx.set_memory_limit` before allocation, and compare observed
  `mx.get_active_memory()` high-water telemetry against the plan.

## Failure handling

- Raise a typed configuration error if fixed bytes exceed the total limit,
  any byte/count input is negative, top-k exceeds transient slots, or a model
  descriptor is internally inconsistent.
- Refuse fractional/float byte accounting and guard integer multiplication.
- If actual allocations exceed the resolved plan, stop the load and report
  planned versus observed bytes; do not shrink buffers behind active requests.
- If a route references an expert or layer outside the descriptor range, fail
  before issuing I/O.

## Acceptance criteria

- [ ] Unit tests reproduce every exact record, corpus, cold-read, fixed, KV,
      and transient value listed above for both descriptors.
- [ ] Planner tests cover caps below fixed footprint, exactly fixed footprint,
      zero persistent slots, one slot/layer, full expert residency, explicit
      cache sub-caps, context growth, and leftover bytes.
- [ ] The JSON CLI output is deterministic and includes all inputs, derived
      components, slots/layer, persistent bytes, and safety margin.
- [ ] A runtime integration test proves the router executes before any expert
      read at every sparse layer and that the next layer is not routed early.
- [ ] A mocked batched route whose unique expert count exceeds transient slots
      is microbatched or rejected without allocating extra memory.
- [ ] No-MLX unit tests can import and exercise descriptors and policy on Linux.

## Dependencies

- No native-I/O dependency for the pure planner and policy tests.
- Model adapters consume this descriptor/state-machine contract.
- The native slot loader consumes `SlotLoad` plans and fixed slot shapes.
- Runtime cache-policy integration owns phase selection and telemetry.
