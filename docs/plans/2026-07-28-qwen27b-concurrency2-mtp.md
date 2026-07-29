# Qwen 3.6 27B Concurrency-2 MTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:subagent-driven-development` to execute this plan task
> by task, with requirements and quality review after each task.

**Goal:** Add an opt-in Qwen 3.6 27B depth-2 MTP execution lane that runs one
request with target shape `[1, 3]` and two requests with one shared target shape
`[2, 3]`, while retaining request-local acceptance, cache ownership, streaming,
cancellation, and session commits.

**Architecture:** Refactor the existing depth-2 `generate_mtpk` loop into a
resumable request machine that yields at configured prefill-chunk boundaries
and at its target-verify boundary. A single model-owner service drives up to two
machines, runs decode-ready K2 tickets before starting the next 1024-token
prefill chunk, merges target caches, executes one prebound width-1 or width-2
target route, extracts each row, and resumes each machine independently.
Construction installs and self-checks the Qwen-only q4/group-size-64 route
once; the enabled execution path selects only cohort width and current request
phase and never validates, retries, or silently falls back.

**Tech Stack:** Python 3.13, MLX 0.32, mlx-lm cache containers, MTPLX native K2
generation, FastAPI/OpenAI serving, pytest, guarded Metal benchmark windows,
`vmmap -summary`/`footprint`.

**Assumptions:**

- Work remains isolated in
  `/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen27b-concurrency2-mtp`
  on `perf/qwen27b-concurrency2-mtp`.
- The target checkpoint is
  `/Users/davidtai/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed`.
- The Metal runtime is
  `/opt/homebrew/var/mtplx/venv-2.3.0-src/bin/python`.
- The existing Qwen service is controlled by
  `/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist` and may only be
  stopped and restored by the serialized guard.
- The unchanged control is commit `3e4bde2`; the design commit is `82a86a6`.
- No launcher, live service, remote branch, PR, or non-Qwen model path is
  changed by this plan.
- Preserve `prefill_chunk_tokens=1024`, the measured Laguna throughput boundary.
  Do not transplant Laguna's AR topology or globally shrink the chunk: derive
  Qwen chunk execution from the actual Qwen cache and prefill path.

**Execution status:**

- [x] Task 1: Freeze the unchanged control and prove the actual model contract.
- [x] Task 2: Install immutable width-1 and width-2 Qwen target routes.
- [x] Task 3: Make configured GDN capture and target caches cohort-safe.
- [x] Task 4: Refactor K2 generation into a resumable request machine.
- [x] Task 5: Execute one or two verify tickets as one fixed cohort.
- [x] Task 6: Integrate the cohort owner service with OpenAI serving.
- [x] Task 7: Run actual-model Metal construction, parity, and isolation gates.
- [ ] Task 8: End-to-end verification and paired performance gate.

---

## Task 1: Freeze the unchanged control and prove the actual model contract

**Files:**

- Create: `scripts/qwen27b_mtp_cohort_receipt.py`
- Create: `tests/test_qwen27b_mtp_cohort_receipt.py`
- Create at runtime: `bench/qwen27b/concurrency2-control-<timestamp>.json`

### Step 1: Write the failing receipt tests

The receipt helper must compute paired cells without importing MLX at module
import time and must reject incomplete data.

```python
from scripts.qwen27b_mtp_cohort_receipt import summarize_cell


def test_summarize_cell_reports_aggregate_and_per_request_rates() -> None:
    cell = summarize_cell(
        [
            {
                "request_id": "a",
                "completion_tokens": 256,
                "elapsed_s": 10.0,
                "ttft_s": 0.2,
                "decode_tok_s": 26.0,
            },
            {
                "request_id": "b",
                "completion_tokens": 256,
                "elapsed_s": 10.0,
                "ttft_s": 0.3,
                "decode_tok_s": 25.0,
            },
        ]
    )
    assert cell["aggregate_output_tok_s"] == 51.2
    assert cell["per_request_decode_tok_s"] == [26.0, 25.0]
    assert cell["max_ttft_s"] == 0.3


def test_summarize_cell_rejects_missing_completion_tokens() -> None:
    with pytest.raises(ValueError, match="completion_tokens"):
        summarize_cell([{"request_id": "a", "elapsed_s": 1.0}])
```

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_receipt.py -q
```

Expected: FAIL because the receipt helper does not exist.

### Step 2: Implement the benchmark/contract receipt helper

The script must:

- assert `mtplx.__file__` resolves under the selected worktree;
- assert MLX is `0.32.x`;
- load the exact local Qwen model offline;
- record backend ID, model path, native MTP depth, q4 affine bits, group size,
  activation dtype, target layer types, quantized-linear `(K, N)` geometries,
  and target-cache types after a real prefill;
- prove real K2 verify inputs are `[1, 3]` and merged inputs are `[2, 3]`;
- issue streaming HTTP cells for concurrency 1 and serialized concurrency 2;
- run three repeats with alternating prompt order;
- save tokens, current generation statistics, TTFT, decode throughput,
  end-to-end throughput, and process memory;
- fail if the server exits, a response is incomplete, or the live service is
  not restored by the outer guard.

Keep aggregation pure:

```python
def summarize_cell(rows: list[dict[str, object]]) -> dict[str, object]:
    required = {
        "request_id",
        "completion_tokens",
        "elapsed_s",
        "ttft_s",
        "decode_tok_s",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"receipt row missing {sorted(missing)}")
    wall_s = max(float(row["elapsed_s"]) for row in rows)
    total_tokens = sum(int(row["completion_tokens"]) for row in rows)
    return {
        "aggregate_output_tok_s": total_tokens / wall_s,
        "per_request_decode_tok_s": [
            float(row["decode_tok_s"]) for row in rows
        ],
        "max_ttft_s": max(float(row["ttft_s"]) for row in rows),
    }
```

The child script launches the unchanged worktree server on `127.0.0.1:18081`
with:

```text
--generation-mode mtp
--depth 2
--verify-strategy capture_commit
--verify-core linear-gdn-from-conv-tape
--profile turbo
--scheduler-mode serial
--max-active-requests 2
```

### Step 3: Run the receipt unit tests

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_receipt.py \
  tests/test_no_mlx_imports.py -q
```

Expected: PASS.

### Step 4: Capture the fresh unchanged control in an exclusive Metal window

Run before any behavior code is changed:

```bash
/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --child-timeout-seconds 5400 \
  -- \
  /opt/homebrew/var/mtplx/venv-2.3.0-src/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen27b-concurrency2-mtp/scripts/qwen27b_mtp_cohort_receipt.py \
  --worktree /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen27b-concurrency2-mtp \
  --server-port 18081 \
  --mode control \
  --repeats 3 \
  --output /Users/davidtai/projects/OpenSourceWTF/bench/qwen27b/concurrency2-control-$(date +%Y%m%d-%H%M%S).json
```

Expected:

- exact Qwen/q4/group-size-64/depth-2 receipt;
- K2 verify shape `[1, 3]`;
- three complete control repeats;
- a `vmmap -summary` or `footprint` result;
- `com.tea.qwen` restored healthy on `:8080`.

If the actual cache types or target geometry differ from the approved design,
stop and amend the design before implementation.

### Step 5: Commit the benchmark harness and control schema

```bash
git add scripts/qwen27b_mtp_cohort_receipt.py \
  tests/test_qwen27b_mtp_cohort_receipt.py
git commit -m "bench: capture Qwen 27B cohort control"
```

Do not commit generated benchmark receipts.

## Task 2: Install immutable width-1 and width-2 Qwen target routes

**Files:**

- Create: `mtplx/qwen27b_mtp_cohort.py`
- Create: `tests/test_qwen27b_mtp_cohort_contract.py`
- Modify: `mtplx/nax_verify.py`
- Modify: `mtplx/runtime.py`
- Modify: `tests/test_nax_verify.py`

### Step 1: Write failing construction-boundary tests

Cover:

- exact backend, depth, bits, group size, verify strategy, and verify core;
- all routed `(K, N, dtype)` shapes present at width 1 and width 2;
- width 1 calling the captured stock `QuantizedLinear.__call__`;
- width 2 calling a prebound M6 callable;
- explicit construction failure rather than partial installation;
- no `os.environ.get`, `m6_ksplit_eligible`, or `lane_disabled` call after
  construction while the fixed route is active.

The public internal contract is:

```python
@dataclass(frozen=True)
class Qwen27BK2DualLane:
    backend_id: str
    depth: int
    bits: int
    group_size: int
    activation_dtype: object
    hidden_variant: str
    verify_strategy: str
    verify_core: str
    max_width: int
    width1_target: Callable[..., TargetForwardResult]
    width2_target: Callable[..., TargetForwardResult]
    cache_routes: tuple[LayerCacheRoute, ...]
    qlinear_routes: Mapping[int, FixedQLinearRoute]
    construction_receipt: Mapping[str, object]

    def target_for_width(self, width: int) -> Callable[..., TargetForwardResult]:
        if width == 1:
            return self.width1_target
        if width == 2:
            return self.width2_target
        raise ValueError(f"Qwen27BK2DualLane width must be 1 or 2, got {width}")
```

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_contract.py \
  tests/test_nax_verify.py -q
```

Expected: FAIL on missing contract and route installation.

### Step 2: Add the construction-only route builder

Implement:

```python
def install_qwen27b_k2_dual_lane(
    runtime: MTPLXRuntime,
    *,
    backend_id: str,
    depth: int,
    verify_strategy: str,
    verify_core: str,
) -> Qwen27BK2DualLane:
    """Validate, self-check, and atomically return the fixed Qwen lane."""
```

The builder walks actual target `QuantizedLinear` modules once and creates a
`FixedQLinearRoute` per module. Each width-2 callable binds:

- the module's exact weight, scale, bias, bits, and group size;
- its exact input `K`, output `N`, activation dtype, and output reshape;
- `nax_qmm_m6` only after construction eligibility and numeric checks pass.

Add a `ContextVar` route scope at the top of the existing qlinear patch:

```python
_FIXED_QMM_ROUTE: ContextVar[FixedQMMExecution | None] = ContextVar(
    "mtplx_fixed_qmm_route",
    default=None,
)


def patched(self, x):
    fixed = _FIXED_QMM_ROUTE.get()
    if fixed is not None:
        return fixed.routes[id(self)].execute(x, width=fixed.width)
    return existing_dynamic_dispatch(self, x)
```

`FixedQLinearRoute.execute` may branch only on `width`. Width 1 directly invokes
the captured stock callable. Width 2 directly invokes its prebound M6 callable.
It must contain no eligibility check, environment read, self-check lookup,
exception fallback, or engagement counter.

Keep the existing dynamic qlinear behavior unchanged outside the fixed scope.

### Step 3: Add configured target-forward entrypoints

Add a runtime method that accepts the installed route and does not introspect
model capabilities:

```python
def forward_qwen27b_k2_target(
    self,
    lane: Qwen27BK2DualLane,
    width: int,
    input_ids: Any,
    cache: list[Any],
) -> TargetForwardResult:
    target = lane.target_for_width(width)
    return target(input_ids=input_ids, cache=cache)
```

The two target callables bind the fixed hidden variant, capture backend, and
qlinear execution scope at construction. Do not add counters to this method.

### Step 4: Run the focused tests

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_contract.py \
  tests/test_nax_verify.py tests/test_runtime_kpis.py \
  tests/test_no_mlx_imports.py -q
```

Expected: PASS.

### Step 5: Commit the construction contract

```bash
git add mtplx/qwen27b_mtp_cohort.py mtplx/nax_verify.py mtplx/runtime.py \
  tests/test_qwen27b_mtp_cohort_contract.py tests/test_nax_verify.py
git commit -m "feat: install fixed Qwen 27B K2 target routes"
```

## Task 3: Make configured GDN capture and target caches cohort-safe

**Files:**

- Modify: `mtplx/gdn_capture.py`
- Modify: `mtplx/cache_state.py`
- Modify: `mtplx/qwen27b_mtp_cohort.py`
- Create: `tests/test_qwen27b_mtp_cohort_cache.py`
- Modify: `tests/test_gdn_tape_headquarter.py`
- Modify: `tests/test_cache_state.py`

### Step 1: Write failing cache ownership tests

Use real MLX arrays where available and small fake cache objects for import-only
tests. Cover:

- two unequal-length `KVCache` entries merge to `BatchKVCache` with correct
  left padding and extract back to the original rows;
- two `OwnedRecurrentStateCache` entries merge and extract without aliasing;
- captured `conv_states`, `state_in`, `tape`, and replayed recurrent state
  split on batch row before request-local commit;
- committing row 0 cannot alter row 1;
- a paged, rotating, quantized, or unknown target cache is either normalized at
  admission by a prebound conversion or rejected before lane installation;
- cohort containers cannot be passed to the session-bank commit adapter.

The prebound layer contract is:

```python
@dataclass(frozen=True)
class LayerCacheRoute:
    layer_index: int
    request_type: type
    cohort_type: type
    normalize_request: Callable[[Any], Any]
    merge: Callable[[tuple[Any, ...]], Any]
    extract: Callable[[Any, int], Any]
```

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_cache.py \
  tests/test_cache_state.py tests/test_gdn_tape_headquarter.py -q
```

Expected: FAIL on missing merge, extraction, and configured capture paths.

### Step 2: Bind capture configuration once

Create an immutable `QwenGDNVerifyConfig` from the already-resolved turbo
profile and add:

```python
def forward_with_gdn_capture_configured(
    model: Any,
    inputs: mx.array,
    cache: list[Any],
    *,
    config: QwenGDNVerifyConfig,
) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
    return _forward_with_gdn_capture_impl(
        model,
        inputs,
        cache,
        config=config,
    )
```

The configured function receives fixed choices for:

- capture backend;
- linear-conv capture;
- native or stock GDN tail;
- fused norm/residual choices;
- layer-evaluation cadence and thresholds;
- hidden variant.

It must not read environment variables or call `lane_disabled`. Existing
general paths keep their current behavior.

### Step 3: Implement installed cache normalization, merge, and extraction

At request admission, convert supported full-attention caches into plain
request-local `KVCache` entries and recurrent layers into
`OwnedRecurrentStateCache`. This is outside measured target cycles.

At target execution:

```python
def merge_target_caches(
    lane: Qwen27BK2DualLane,
    request_caches: tuple[list[Any], ...],
) -> list[Any]:
    return [
        route.merge(tuple(cache[route.layer_index] for cache in request_caches))
        for route in lane.cache_routes
    ]


def extract_target_cache(
    lane: Qwen27BK2DualLane,
    cohort_cache: list[Any],
    row: int,
) -> list[Any]:
    return [
        route.extract(cohort_cache[route.layer_index], row)
        for route in lane.cache_routes
    ]
```

Add `extract_captured_row(captures, row)` that slices every batch-owned capture
leaf to `[row:row+1]` before `commit_captured_prefix` can see it. Preserve
non-array metadata unchanged.

No merged cache may become authoritative request or session state.

### Step 4: Run focused cache and capture tests

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_cache.py \
  tests/test_cache_state.py tests/test_gdn_tape_headquarter.py \
  tests/test_gdn_boundary_retention.py -q
```

Expected: PASS.

### Step 5: Commit cache ownership support

```bash
git add mtplx/gdn_capture.py mtplx/cache_state.py \
  mtplx/qwen27b_mtp_cohort.py tests/test_qwen27b_mtp_cohort_cache.py \
  tests/test_gdn_tape_headquarter.py tests/test_cache_state.py
git commit -m "feat: isolate Qwen MTP cohort cache rows"
```

## Task 4: Refactor K2 generation into a resumable request machine

**Files:**

- Create: `mtplx/mtp_k2_stepper.py`
- Create: `tests/test_mtp_k2_stepper.py`
- Modify: `mtplx/generation.py`
- Modify: `tests/test_generation_sustained.py`
- Modify: `tests/test_constrained.py`

### Step 1: Lock current K2 behavior before refactoring

Add deterministic fake-runtime tests that record:

- target inputs per cycle;
- sampled, drafted, accepted, correction, bonus, and emitted tokens;
- target and MTP cache offsets;
- recurrent commit prefix;
- stop and constraint decisions;
- callback ordering;
- `GenerationStats` fields.

For each fixture, record the output from the unchanged `generate_mtpk` path and
assert the later request-machine adapter produces the exact same trace.

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_mtp_k2_stepper.py -q
```

Expected: FAIL because the request machine does not exist.

### Step 2: Define the request-machine boundary

Create:

```python
@dataclass(frozen=True)
class MTPK2VerifyTicket:
    request_id: str
    input_ids: Any
    request_cache: list[Any]
    draft_distributions: tuple[Any, Any]
    acceptance_context: MTPK2AcceptanceContext


@dataclass(frozen=True)
class MTPK2PrefillTicket:
    request_id: str
    input_ids: Any
    request_cache: list[Any]
    prompt_start: int
    prompt_stop: int


@dataclass(frozen=True)
class MTPK2VerifyResult:
    logits: Any
    hidden: Any
    captures: dict[int, dict[str, Any]]
    request_cache: list[Any]


@dataclass
class MTPK2RequestState:
    request_id: str
    machine: Generator[
        MTPK2PrefillTicket | MTPK2VerifyTicket,
        MTPK2PrefillResult | MTPK2VerifyResult,
        GenerationOutput,
    ]
    target_cache: list[Any]
    mtp_cache: Any
    tokens: list[int]
    rng: np.random.Generator
    sampler: SamplerConfig
    draft_sampler: SamplerConfig
    constraint: Any
    stop_token_ids: set[int]
    token_callback: Callable[[list[int]], None] | None
    prefill_callback: Callable[..., None] | None
    cancel_event: Event | None
    session_id: str | None
    stats: GenerationStats
    pending_ticket: MTPK2VerifyTicket | None = None
```

The machine owns every mutable per-request value. A ticket is immutable and
does not mutate the authoritative target cache.

Add behavior-locking tests proving that each prefill ticket contains at most
1024 tokens and that resuming all chunks produces the same logits, hidden
state, target cache, MTP history, output tokens, callbacks, and statistics as
the unchanged prefill path.

### Step 3: Extract the existing K2 loop mechanically

Rename the current implementation body to `_generate_mtpk_machine`. Extract the
existing `_prefill` loop so it yields one `MTPK2PrefillTicket` per configured
1024-token slice, then replace only the main depth-2 target
`forward_ar_capture` call with:

```python
verify_result = yield MTPK2VerifyTicket(
    request_id=request_state.request_id,
    input_ids=verify_input,
    request_cache=cache,
    draft_distributions=tuple(draft_distributions),
    acceptance_context=acceptance_context,
)
verify_logits = verify_result.logits
verify_hidden = verify_result.hidden
captures = verify_result.captures
cache = verify_result.request_cache
request_state.target_cache = cache
```

Keep the existing acceptance, rejection residual, constraint, stop, bonus,
repair, capture commit, MTP-history, session, callback, and metrics code in the
same order.

Implement the existing public `generate_mtpk` as a solo adapter:

```python
def generate_mtpk(
    runtime: MTPLXRuntime,
    input_ids: list[int],
    *args: Any,
    **kwargs: Any,
) -> GenerationOutput:
    state = make_mtpk2_request_state(runtime, input_ids, *args, **kwargs)
    ticket = next(state.machine)
    while True:
        result = execute_solo_ticket(runtime, ticket)
        try:
            ticket = state.machine.send(result)
        except StopIteration as finished:
            return finished.value
```

Only depth 2 uses the request machine. Other depths stay on the unchanged
implementation. The adapter's target execution must reproduce the prior B1
forward exactly. The solo adapter consumes prefill tickets consecutively, so
the public serial path remains behaviorally unchanged.

### Step 4: Prove behavior parity

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_mtp_k2_stepper.py \
  tests/test_generation_sustained.py tests/test_constrained.py \
  tests/test_sampling.py tests/test_penalties.py -q
```

Expected: PASS with exact token, trace, callback, cache, and stats parity.

### Step 5: Commit the resumable K2 engine

```bash
git add mtplx/mtp_k2_stepper.py mtplx/generation.py \
  tests/test_mtp_k2_stepper.py tests/test_generation_sustained.py \
  tests/test_constrained.py
git commit -m "refactor: make Qwen K2 verification resumable"
```

## Task 5: Execute one or two verify tickets as one fixed cohort

**Files:**

- Modify: `mtplx/qwen27b_mtp_cohort.py`
- Create: `tests/test_qwen27b_mtp_cohort_runner.py`

### Step 1: Write failing runner state-machine tests

Cover:

- width 1 executes immediately without a wait hook;
- width 2 preserves input and output request order;
- exactly one target call for two tickets;
- inputs stack to `[2, 3]`;
- one row rejects while the other fully accepts;
- `1 -> 2 -> 1` across join and departure;
- cancellation before route selection;
- cancellation during a shared forward discards only that row after
  extraction;
- a width-2 exception fails both affected requests and never invokes width 1;
- stepping with zero or more than two requests raises before target execution;
- monkeypatched environment and eligibility functions raise if touched by
  `step`.

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_runner.py -q
```

Expected: FAIL because the runner does not exist.

### Step 2: Implement the fixed cohort executor

```python
class MTPK2CohortRunner:
    def __init__(self, lane: Qwen27BK2DualLane) -> None:
        self._lane = lane

    def step(
        self,
        requests: tuple[MTPK2RequestState, ...],
    ) -> tuple[MTPK2VerifyResult, ...]:
        live = tuple(
            request
            for request in requests
            if not request.cancel_event or not request.cancel_event.is_set()
        )
        width = len(live)
        if width not in (1, 2):
            raise ValueError(f"cohort step requires one or two live requests, got {width}")
        tickets = tuple(require_ticket(request) for request in live)
        if width == 1:
            return (execute_width1(self._lane, tickets[0]),)
        return execute_width2(self._lane, tickets)
```

`execute_width2` must:

1. stack token rows in request order;
2. merge request target caches using the installed layer routes;
3. call `lane.width2_target` once;
4. evaluate logits, hidden, captures, and cache roots once;
5. extract cache and capture row 0 and row 1;
6. return independent `MTPK2VerifyResult` values.

It must not catch target exceptions. The owner service handles the cohort
failure without retry.

### Step 3: Run focused runner tests

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_runner.py \
  tests/test_qwen27b_mtp_cohort_cache.py \
  tests/test_qwen27b_mtp_cohort_contract.py -q
```

Expected: PASS.

### Step 4: Commit the runner

```bash
git add mtplx/qwen27b_mtp_cohort.py \
  tests/test_qwen27b_mtp_cohort_runner.py
git commit -m "feat: execute paired Qwen K2 target cycles"
```

## Task 6: Integrate the cohort owner service with OpenAI serving

**Files:**

- Create: `mtplx/server/mtp_cohort.py`
- Create: `tests/test_mtp_cohort_service.py`
- Modify: `mtplx/server/openai.py`
- Modify: `tests/test_server_openai.py`
- Modify: `tests/test_dashboard_endpoints.py`
- Modify: `tests/test_public_cli.py`

### Step 1: Write failing service and routing tests

Cover:

- startup installs the lane only for
  `mtp_cohort_experimental + --experimental-mtp-cohorts`;
- requested mode fails startup if max active/decode width are not exactly 2,
  wait is not zero, model contract is wrong, or lane self-check fails;
- cohort mode never calls `_BatchedARGenerationService`;
- constraints, sessions, stops, explicit seeds, penalties, callbacks, and
  cancellation are carried into distinct request states;
- a lone job starts immediately;
- one prefill ticket processes no more than the configured 1024-token budget;
- a decode-ready request runs before the next chunk of another request's
  prefill;
- a second long-prefill request advances fairly without batching prompt rows;
- a second pending job joins only after the current target cycle;
- a completed or cancelled job leaves the survivor on width 1;
- each future completes with its own output or exception;
- session-bank commits receive only extracted request caches;
- snapshot reports pending, active request IDs, and current width without
  per-cycle or per-dispatch counters;
- dashboard reports `mtp_cohort_width_1`/`mtp_cohort_width_2` and no
  `batch_size_gt_1` MTP-disabled reason.

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_mtp_cohort_service.py \
  tests/test_server_openai.py tests/test_dashboard_endpoints.py \
  tests/test_public_cli.py -q
```

Expected: FAIL on missing service and current AR fallback.

### Step 2: Implement the single-owner cohort pump

Create `_MTPK2CohortJob` with the same request-local inputs currently passed to
`generate_mtpk`. Implement `_MTPK2CohortGenerationService`:

```python
class MTPK2CohortGenerationService:
    def submit(self, job: MTPK2CohortJob) -> Future:
        with self._condition:
            self._pending.append(job)
            if not self._pump_scheduled:
                self._pump_scheduled = True
                _submit_foreground_model_work(
                    self._state,
                    self._pump,
                    batch_key="mtp_cohort.pump",
                )
        return job.future
```

The pump:

- admits at most two jobs;
- advances each admitted machine to either a prefill or verify ticket;
- executes at most one 1024-token prefill ticket before re-evaluating the
  decode-ready queue;
- prioritizes ready K2 verify tickets over starting another prefill chunk;
- if one ticket is ready, executes it immediately;
- drains one pending job only between cycles;
- if two tickets are ready, calls `runner.step((first, second))`;
- resumes each machine with only its row result;
- emits callbacks from the request machine in request order;
- finishes, cancels, or fails futures independently;
- if the shared target forward raises, fails that cohort and does not retry;
- schedules no second model worker.

### Step 3: Install the service before warmup

In `ServerState.__init__`, resolve scheduler configuration after runtime load
and before `_run_startup_warmup`:

```python
if config.mode == SchedulerMode.MTP_COHORT_EXPERIMENTAL:
    if not config.experimental_mtp_cohorts:
        raise RuntimeError(
            "mtp_cohort_experimental requires --experimental-mtp-cohorts"
        )
    self.mtp_cohort_lane = install_qwen27b_k2_dual_lane(
        self.runtime,
        backend_id=self.backend_descriptor.backend_id,
        depth=int(self.args.depth),
        verify_strategy=str(self.args.verify_strategy),
        verify_core=str(self.args.verify_core),
    )
    self.mtp_cohort_service = MTPK2CohortGenerationService(
        self,
        self.mtp_cohort_lane,
    )
else:
    self.mtp_cohort_lane = None
    self.mtp_cohort_service = None
```

This is an explicit startup contract. Do not catch the installation error and
continue with another lane.

### Step 4: Route experimental cohort requests before AR batching

In `_run_generation_dispatched`, route the explicit cohort mode directly to
`mtp_cohort_service`. Preserve the AR path for `ar_batch`; remove
`MTP_COHORT_EXPERIMENTAL` from `_use_live_ar_batch` and
`_ar_batch_mtp_fallback_reason`.

The cohort result continues through the existing native-MTP metrics/session
finalization, not `_finalize_batched_ar_generation`.

### Step 5: Update dashboard truth

Use the service snapshot and installed construction receipt:

```python
if config.mode == SchedulerMode.MTP_COHORT_EXPERIMENTAL:
    snapshot = state.mtp_cohort_service.snapshot()
    active_width = int(snapshot["active_width"])
    active_lane = (
        f"mtp_cohort_width_{active_width}"
        if active_width in (1, 2)
        else "mtp_cohort_idle"
    )
    mtp_disabled_reason = None
```

Do not add hot-path engagement counters.

### Step 6: Run serving tests

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_mtp_cohort_service.py \
  tests/test_server_openai.py tests/test_dashboard_endpoints.py \
  tests/test_public_cli.py tests/test_engine_session_concurrency.py \
  tests/test_ar_batch_penalties.py -q
```

Expected: PASS.

### Step 7: Commit service integration

```bash
git add mtplx/server/mtp_cohort.py mtplx/server/openai.py \
  tests/test_mtp_cohort_service.py tests/test_server_openai.py \
  tests/test_dashboard_endpoints.py tests/test_public_cli.py
git commit -m "feat: serve Qwen K2 cohorts at concurrency two"
```

## Task 7: Run actual-model Metal construction, parity, and isolation gates

**Files:**

- Create: `scripts/qwen27b_mtp_cohort_selfcheck.py`
- Create: `tests/test_qwen27b_mtp_cohort_selfcheck.py`
- Modify: `mtplx/qwen27b_mtp_cohort.py`

### Step 1: Write failing self-check report tests

The report must reject:

- a missing qlinear shape;
- any q4 M6 kernel `dmax` over the existing turbo qmm tolerance;
- logits/hidden/capture shape mismatch;
- attention offset mismatch;
- recurrent-state mismatch;
- extracted-row aliasing;
- token or acceptance mismatch against the stock B2 cohort.

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_selfcheck.py -q
```

Expected: FAIL because the report checker does not exist.

### Step 2: Implement atomic actual-model self-checks

At lane installation:

- enumerate every actual qlinear route;
- generate deterministic activation rows in the installed dtype for its exact
  `K`;
- compare width-2 M6 custom output to
  `mx.quantized_matmul(activation, route.weight, scales=route.scales,
  biases=route.biases, transpose=True, bits=4, group_size=64)`;
- build two deterministic prompt-prefilled request caches;
- clone those caches into custom and stock references;
- run one `[2, 3]` custom target cycle and one unchanged `[2, 3]` stock cycle;
- evaluate and compare logits, hidden, attention state, recurrent captures,
  extracted request caches, acceptance, and generated tokens;
- mutate/commit each extracted custom row in turn and prove the sibling row is
  unchanged.

The report is accumulated locally and the immutable lane object is returned
only after all checks pass. Any mismatch raises with model path, layer index,
shape, observed delta, and accepted tolerance.

### Step 3: Run CPU report tests

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_selfcheck.py \
  tests/test_qwen27b_mtp_cohort_contract.py \
  tests/test_qwen27b_mtp_cohort_cache.py -q
```

Expected: PASS.

### Step 4: Run actual-model self-check in a guarded Metal window

```bash
/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --child-timeout-seconds 3600 \
  -- \
  /opt/homebrew/var/mtplx/venv-2.3.0-src/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen27b-concurrency2-mtp/scripts/qwen27b_mtp_cohort_selfcheck.py \
  --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed \
  --output /Users/davidtai/projects/OpenSourceWTF/bench/qwen27b/concurrency2-selfcheck-$(date +%Y%m%d-%H%M%S).json
```

Expected: every real Qwen qlinear shape passes M6 comparison; full target-cycle
and cache isolation checks pass; live Qwen service is restored.

### Step 5: Commit the actual-model gate

```bash
git add scripts/qwen27b_mtp_cohort_selfcheck.py \
  tests/test_qwen27b_mtp_cohort_selfcheck.py mtplx/qwen27b_mtp_cohort.py
git commit -m "test: gate Qwen K2 cohort construction"
```

## Task 8: End-to-end verification and paired performance gate

**Files:**

- Modify: `scripts/qwen27b_mtp_cohort_receipt.py`
- Modify: `tests/test_qwen27b_mtp_cohort_receipt.py`
- Create at runtime: `bench/qwen27b/concurrency2-candidate-<timestamp>.json`
- Create at runtime: `bench/qwen27b/concurrency2-comparison-<timestamp>.json`

### Step 1: Add failing comparison-gate tests

Implement and test:

```python
def evaluate_promotion(
    control: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    solo_ratio = candidate["c1"]["aggregate_output_tok_s"] / control["c1"][
        "aggregate_output_tok_s"
    ]
    pair_ratio = candidate["c2"]["aggregate_output_tok_s"] / control["c2"][
        "aggregate_output_tok_s"
    ]
    return {
        "solo_ratio": solo_ratio,
        "pair_ratio": pair_ratio,
        "solo_pass": solo_ratio >= 0.99,
        "pair_pass": pair_ratio >= 1.35,
    }
```

Tests must also fail promotion for token/parity failures, missing repeats,
material 4K/production TTFT regression, cache isolation failure, nonzero
fallback reason, or wrong scheduler lane.

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest tests/test_qwen27b_mtp_cohort_receipt.py -q
```

Expected: FAIL until candidate comparison is implemented.

### Step 2: Run the complete local test gate

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest \
  tests/test_no_mlx_imports.py \
  tests/test_public_cli.py \
  tests/test_runtime_kpis.py \
  tests/test_batching_foundation.py \
  tests/test_qwen27b_mtp_cohort_contract.py \
  tests/test_qwen27b_mtp_cohort_cache.py \
  tests/test_mtp_k2_stepper.py \
  tests/test_qwen27b_mtp_cohort_runner.py \
  tests/test_mtp_cohort_service.py \
  tests/test_qwen27b_mtp_cohort_selfcheck.py \
  tests/test_server_openai.py \
  tests/test_dashboard_endpoints.py \
  tests/test_engine_session_concurrency.py \
  tests/test_constrained.py \
  tests/test_penalties.py -q
```

Expected: PASS.

Then run the full suite:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q
```

Expected: PASS, excluding only tests already documented as requiring a
separate external dependency or service.

### Step 3: Run candidate HTTP cells in an exclusive Metal window

The candidate server uses:

```text
--scheduler-mode mtp_cohort_experimental
--experimental-mtp-cohorts
--max-active-requests 2
--decode-batch-max 2
--batch-wait-ms 0
--prefill-chunk-tokens 1024
```

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --child-timeout-seconds 7200 \
  -- \
  /opt/homebrew/var/mtplx/venv-2.3.0-src/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen27b-concurrency2-mtp/scripts/qwen27b_mtp_cohort_receipt.py \
  --worktree /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen27b-concurrency2-mtp \
  --server-port 18081 \
  --mode candidate \
  --repeats 3 \
  --control-glob '/Users/davidtai/projects/OpenSourceWTF/bench/qwen27b/concurrency2-control-*.json' \
  --output /Users/davidtai/projects/OpenSourceWTF/bench/qwen27b/concurrency2-candidate-$(date +%Y%m%d-%H%M%S).json
```

Cells:

- concurrency 1, short prompt, 256 generated tokens;
- concurrency 2, distinct short prompts, 256 tokens each;
- concurrency 2, distinct approximately 4K prompts, 256 tokens each;
- concurrency 2 with uneven completion limits to exercise `2 -> 1`;
- production-shaped long prompts;
- one approximately 8K-token prefill started before a short request, proving
  the short request is admitted between 1024-token chunks;
- session restore/final commit;
- one constrained/tool request paired with an unconstrained request;
- cancellation during a shared target forward.

Expected:

- all cells complete with native MTP enabled;
- width-2 cells report `mtp_cohort_width_2`;
- departure cells finish on `mtp_cohort_width_1`;
- no `mtp_disabled_reason`, AR fallback, or retry;
- independent tokens, streams, sessions, and cancellation;
- resident-memory receipts from `vmmap -summary` or `footprint`;
- live Qwen service restored healthy afterward.

### Step 4: Evaluate the promotion gates

The comparison must require:

- three paired repeats per performance cell;
- concurrency-1 candidate/control ratio at least `0.99`;
- concurrency-2 aggregate candidate/control ratio at least `1.35`;
- no material TTFT regression on 4K or production-shaped cells;
- long-prefill overlap admits ready work between chunks and keeps long-prompt
  prefill throughput within 5% of the unchanged 1024-token control;
- token, acceptance, cache, session, streaming, constraint, and cancellation
  gates green;
- no enabled-path environment read, invariant validation, fallback, retry, or
  new engagement counter.

If any gate fails, retain the branch locally and report the failed receipt.
Do not modify the live launcher.

### Step 5: Review the final diff and repository state

Run:

```bash
git diff 3e4bde2...HEAD --check
git status --short --branch
git log --oneline --decorate 3e4bde2..HEAD
```

Review specifically for:

- changes outside Qwen cohort, shared cache primitives, K2 refactor, server
  routing, tests, and benchmark harnesses;
- any `eligible-or-stock`, `try-custom-then-fallback`, or hot environment read;
- any counter added to per-token, per-layer, per-cycle, or per-dispatch code;
- any accidental launcher, Hy3, GLM, Laguna, A3B, or remote change.

### Step 6: Run verification-before-completion

Use `superpowers-optimized:verification-before-completion` with fresh output
from the focused suite, full suite, actual-model self-check, paired benchmark,
diff check, and git status.

Present:

- commits and changed file groups;
- exact control and candidate receipt paths;
- correctness/isolation results;
- solo and pair throughput ratios;
- TTFT and resident-memory comparison;
- remaining risks;
- confirmation that the branch is local, the feature is off by default, and
  the live Qwen launcher was not changed.

Do not push, open a PR, merge, or enable the service without explicit approval.
