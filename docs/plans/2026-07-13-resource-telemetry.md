# Resource Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, low-contention benchmark telemetry that distinguishes storage saturation, reader backpressure, submission starvation, host pressure, and coarse same-interval I/O/Metal coactivity without inventing unavailable GPU or RAM measurements.

**Architecture:** Runtime hot paths publish cumulative queue, worker, byte, and completion-fence occupancy through state-change counters. A benchmark-only sampler reads a cheap snapshot, differences cumulative counters on one clock, and emits raw intervals plus an evidence matrix. An optional non-interactive `powermetrics` collector adds process/GPU evidence when privilege and hardware support exist; absent measurements remain explicitly unavailable. The benchmark harness marks telemetry-enabled runs as diagnostic and keeps the default timing lane unchanged.

**Tech Stack:** Python 3.12+, `threading`, `time.monotonic_ns`, `plistlib`, `subprocess`, pytest, the existing streamed-generation JSON harness.

**Assumptions:**
- Assumes positional reads and completion fences pass through `ExpertSlotPool` — it will not observe unrelated filesystem work.
- Assumes the measured SSD ceiling is supplied for saturation evidence — without it, SSD saturation remains unavailable.
- Assumes passwordless `sudo -n powermetrics` or an already-authorized process — the harness will never prompt or claim GPU coverage when collection fails.
- Assumes telemetry-enabled runs are diagnostic — their token rate is not a promotion headline until separately reproduced with telemetry disabled.
- The approved schema name is `mtplx-resource-telemetry-v1`; this version label was explicitly accepted with the design.

---

## File Structure

- `mtplx/resource_metrics.py` — lock-protected state-change occupancy counters shared by reader and completion pools.
- `mtplx/expert_slots.py` — wraps accepted executor work and exposes a no-drain, no-slot-walk resource snapshot.
- `mtplx/expert_runtime.py` — publishes cheap cache, per-layer demand, I/O, reader, completion, and MLX-memory counters.
- `mtplx/runtime.py` — exposes the cheap snapshot through the top-level runtime.
- `mtplx/benchmarks/resource_telemetry.py` — interval differencing, evidence generation, bounded timeline, synchronous-fence operation counts, and optional `powermetrics` collection.
- `scripts/benchmark_streamed_generation.py` — opt-in CLI and per-repeat telemetry integration for single-stream and concurrent lanes.
- `tests/test_resource_metrics.py` — deterministic occupancy accounting tests with a fake monotonic clock.
- `tests/test_resource_telemetry.py` — synthetic interval, evidence, coverage, and plist-parser tests.
- `tests/test_benchmark_streamed_generation_cli.py` — default-off and CLI contract tests.
- `tests/test_expert_slots_runtime.py` — executor tracking and cheap-snapshot regression tests.
- `docs/RESOURCE_TELEMETRY.md` — agent-facing interpretation guide and decision matrix.
- `project-map.md` — points future agents to the interpretation guide and records the no-guessing constraint.

### Task 1: Publish exact runtime occupancy counters

**Files:**
- Create: `mtplx/resource_metrics.py`
- Modify: `mtplx/expert_slots.py`
- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/runtime.py`
- Test: `tests/test_resource_metrics.py`
- Test: `tests/test_expert_slots_runtime.py`

**Security flag:** `none`

**Does NOT cover:** It does not sample OS CPU, GPU, disk, or DRAM counters; it reports only MTPLX-owned work and physical read counters.

- [ ] **Step 1: Write failing occupancy and cheap-snapshot tests**

```python
class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, nanoseconds: int) -> None:
        self.now_ns += int(nanoseconds)


def test_pool_occupancy_integrates_queue_workers_and_units():
    clock = FakeClock()
    metrics = PoolOccupancy(worker_capacity=2, clock_ns=clock)
    metrics.submitted(100)
    clock.advance(10)
    metrics.started(100)
    clock.advance(20)
    metrics.completed(100)
    snapshot = metrics.snapshot()
    assert snapshot["accepted_submissions"] == 1
    assert snapshot["completed"] == 1
    assert snapshot["queued_work_ns"] == 10
    assert snapshot["active_work_ns"] == 20
    assert snapshot["queued_unit_ns"] == 1_000
    assert snapshot["active_unit_ns"] == 2_000
    assert snapshot["queued_work"] == 0
    assert snapshot["active_work"] == 0


def test_rejected_submission_restores_queue_accounting():
    metrics = PoolOccupancy(worker_capacity=1)
    metrics.submitted(64)
    metrics.rejected(64)
    snapshot = metrics.snapshot()
    assert snapshot["accepted_submissions"] == 0
    assert snapshot["rejected_submissions"] == 1
    assert snapshot["queued_work"] == 0
    assert snapshot["queued_units"] == 0


def open_tiny_runtime(tmp_path):
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    plan = _plan(spec)
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=plan.total_limit_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
    )
    return ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )


def test_resource_snapshot_does_not_call_the_full_slot_snapshot(
    tmp_path, monkeypatch
):
    runtime = open_tiny_runtime(tmp_path)
    monkeypatch.setattr(
        runtime.slots,
        "snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("must not full-snapshot")),
    )
    try:
        snapshot = runtime.resource_telemetry_snapshot(mx_module=object())
        assert snapshot["reader_pool"]["worker_capacity"] >= 1
        assert "io" in snapshot
        assert "cache_by_layer" in snapshot
    finally:
        runtime.close()
```

- [ ] **Step 2: Run tests and verify the missing API fails**

Run: `uv run --no-project --with pytest --with numpy --with mlx --with safetensors python -m pytest -q tests/test_resource_metrics.py tests/test_expert_slots_runtime.py`

Expected: FAIL because `PoolOccupancy` and `resource_telemetry_snapshot()` do not exist.

- [ ] **Step 3: Implement state-change accounting and tracked submissions**

```python
class PoolOccupancy:
    def __init__(self, *, worker_capacity: int, clock_ns=time.monotonic_ns):
        self.worker_capacity = int(worker_capacity)
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._last_ns = int(clock_ns())
        self._queued_work = self._active_work = 0
        self._queued_units = self._active_units = 0
        self._queued_work_peak = self._active_work_peak = 0
        self._queued_units_peak = self._active_units_peak = 0
        self._queued_work_ns = self._active_work_ns = 0
        self._queued_unit_ns = self._active_unit_ns = 0
        self._accepted = self._started = self._completed = self._rejected = 0

    def _accrue(self, now_ns: int) -> None:
        span = max(0, now_ns - self._last_ns)
        self._queued_work_ns += self._queued_work * span
        self._active_work_ns += self._active_work * span
        self._queued_unit_ns += self._queued_units * span
        self._active_unit_ns += self._active_units * span
        self._last_ns = now_ns

    def submitted(self, units: int) -> None:
        with self._lock:
            self._accrue(int(self._clock_ns()))
            self._accepted += 1
            self._queued_work += 1
            self._queued_units += int(units)
            self._queued_work_peak = max(self._queued_work_peak, self._queued_work)
            self._queued_units_peak = max(self._queued_units_peak, self._queued_units)

    def rejected(self, units: int) -> None:
        with self._lock:
            self._accrue(int(self._clock_ns()))
            if self._queued_work < 1 or self._queued_units < int(units):
                raise RuntimeError("pool telemetry queue underflow")
            self._accepted -= 1
            self._rejected += 1
            self._queued_work -= 1
            self._queued_units -= int(units)

    def started(self, units: int) -> None:
        with self._lock:
            self._accrue(int(self._clock_ns()))
            self._started += 1
            self._queued_work -= 1
            self._queued_units -= int(units)
            self._active_work += 1
            self._active_units += int(units)
            self._active_work_peak = max(self._active_work_peak, self._active_work)
            self._active_units_peak = max(self._active_units_peak, self._active_units)

    def completed(self, units: int) -> None:
        with self._lock:
            self._accrue(int(self._clock_ns()))
            self._completed += 1
            self._active_work -= 1
            self._active_units -= int(units)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._accrue(int(self._clock_ns()))
            return {
                "worker_capacity": self.worker_capacity,
                "accepted_submissions": self._accepted,
                "started": self._started,
                "completed": self._completed,
                "rejected_submissions": self._rejected,
                "queued_work": self._queued_work,
                "active_work": self._active_work,
                "queued_units": self._queued_units,
                "active_units": self._active_units,
                "queued_work_peak": self._queued_work_peak,
                "active_work_peak": self._active_work_peak,
                "queued_units_peak": self._queued_units_peak,
                "active_units_peak": self._active_units_peak,
                "queued_work_ns": self._queued_work_ns,
                "active_work_ns": self._active_work_ns,
                "queued_unit_ns": self._queued_unit_ns,
                "active_unit_ns": self._active_unit_ns,
            }
```

Route both `_fill` and `_fill_batch` through one `_submit_io()` wrapper that reserves queue accounting before `ThreadPoolExecutor.submit`, rolls it back on rejection, marks start in the worker, and completes in `finally`. Route only real completion fences through the same pattern with slot count as the unit; the diagnostic drain barrier remains untracked.

Track current and peak queue depth, active workers, queued units, and active
units alongside the occupancy integrals shown above. Reject any transition
that would underflow current work or units so instrumentation cannot silently
publish impossible states.

Add `ExpertSlotPool.resource_telemetry_snapshot()` returning only `metrics`, `io`, `reader_pool`, and `completion_fences`. Add `ExpertStreamingRuntime.resource_telemetry_snapshot()` returning that payload plus global and per-layer cache counters and MLX memory. Add `MTPLXRuntime.expert_resource_telemetry_snapshot()` as the public harness entrypoint.

- [ ] **Step 4: Run focused and existing slot tests**

Run: `uv run --no-project --with pytest --with numpy --with mlx --with safetensors python -m pytest -q tests/test_resource_metrics.py tests/test_expert_slots_runtime.py`

Expected: PASS with no occupancy underflow and no changed slot-safety behavior.

- [ ] **Step 5: Commit**

```bash
git add mtplx/resource_metrics.py mtplx/expert_slots.py mtplx/expert_runtime.py mtplx/runtime.py tests/test_resource_metrics.py tests/test_expert_slots_runtime.py
git commit -m "feat(bench): expose causal runtime occupancy"
```

### Task 2: Build bounded correlated telemetry and honest evidence

**Files:**
- Create: `mtplx/benchmarks/resource_telemetry.py`
- Test: `tests/test_resource_telemetry.py`

**Security flag:** `none`

**Does NOT cover:** `powermetrics` unavailability, missing per-process GPU support, and absent DRAM counters remain explicit coverage gaps; system GPU activity is never relabeled as per-process utilization.

- [ ] **Step 1: Write failing synthetic evidence tests**

```python
def synthetic_intervals(
    *,
    ssd_gib_s: float,
    queued_fraction: float = 0.0,
    active_workers: float = 0.0,
    worker_capacity: int = 4,
    expert_misses: int = 8,
) -> list[dict[str, float | int | bool]]:
    interval_count = 10
    queued_count = round(interval_count * queued_fraction)
    return [
        {
            "interval_seconds": 1.0,
            "reader_read_bytes": int(ssd_gib_s * 1024**3),
            "reader_read_operations": 1024,
            "ssd_gib_per_second": ssd_gib_s,
            "expert_misses": expert_misses,
            "reader_queue_nonempty": index < queued_count,
            "mean_active_readers": active_workers,
            "reader_worker_capacity": worker_capacity,
            "io_active": active_workers > 0,
            "completion_fence_pending": False,
        }
        for index in range(interval_count)
    ]


def test_backed_up_readers_below_ssd_ceiling_are_not_called_storage_bound():
    report = summarize_intervals(
        synthetic_intervals(
            ssd_gib_s=5.0,
            queued_fraction=0.9,
            active_workers=4.0,
            worker_capacity=4,
        ),
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )
    assert report["evidence"]["ssd_saturation"]["status"] == "not_supported"
    assert report["evidence"]["reader_backpressure"]["status"] == "present"
    assert report["attribution"]["status"] == "incomplete"
    assert "reader_pool_or_read_shape" in report["attribution"]["candidates"]
    assert "bound_by" not in report


def test_storage_pressure_screen_routes_a_candidate_without_claiming_causality():
    report = summarize_intervals(
        synthetic_intervals(
            ssd_gib_s=11.8,
            queued_fraction=0.8,
            active_workers=3.8,
            worker_capacity=4,
        ),
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )
    assert report["evidence"]["ssd_saturation"]["status"] == "supported"
    assert report["attribution"] == {
        "status": "incomplete",
        "candidates": ["storage_throughput"],
    }


def test_missing_powermetrics_is_coverage_not_zero_gpu_usage():
    report = summarize_intervals(
        synthetic_intervals(ssd_gib_s=2.0),
        ssd_ceiling_gib_s=12.5,
        powermetrics={"available": False, "reason": "sudo requires a password"},
    )
    assert report["coverage"]["gpu"] == "unavailable"
    assert report["evidence"]["gpu_activity"]["status"] == "unavailable"
    assert report["coverage"]["dram_bandwidth"] == "unavailable"


def test_powermetrics_plist_extracts_only_the_benchmark_pid():
    documents = [
        {
            "timestamp_ns": 1_000,
            "tasks": [
                {"pid": 41, "gpu_time_ns": 10, "cpu_ms_per_s": 20},
                {
                    "pid": 42,
                    "gpu_time_ns": 600,
                    "cpu_ms_per_s": 850,
                    "disk_read_bytes": 4096,
                    "sfi_wait_time_ns": 50,
                },
            ],
        }
    ]
    samples = parse_powermetrics_documents(documents, pid=42)
    assert samples == [{
        "timestamp_ns": 1_000,
        "process_gpu_time_ns": 600,
        "process_cpu_ms_per_s": 850.0,
        "process_disk_read_bytes": 4096,
        "process_wait_time_ns": 50,
    }]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_resource_telemetry.py`

Expected: FAIL because the sampler, summarizer, and plist parser do not exist.

- [ ] **Step 3: Implement the sampler and evidence matrix**

```python
@dataclass(frozen=True)
class ResourceTick:
    monotonic_ns: int
    completion_tokens: int
    snapshot: dict[str, Any]


class ResourceTelemetrySampler:
    def __init__(self, snapshot, *, token_count, interval_s=0.25, max_samples=4096):
        if interval_s <= 0:
            raise ValueError("resource telemetry interval must be positive")
        if max_samples < 2:
            raise ValueError("resource telemetry max samples must be at least 2")
        self._snapshot = snapshot
        self._token_count = token_count
        self._interval_s = float(interval_s)
        self._first_tick = None
        self._recent_ticks = deque(maxlen=int(max_samples) - 1)

    def _tick(self) -> ResourceTick:
        return ResourceTick(
            monotonic_ns=time.monotonic_ns(),
            completion_tokens=int(self._token_count()),
            snapshot=self._snapshot(),
        )


@dataclass
class ResourceRun:
    sampler: ResourceTelemetrySampler
    powermetrics: PowermetricsCollector

    def report(self, **conditions: Any) -> dict[str, Any]:
        return build_resource_report(
            self.sampler.ticks,
            powermetrics=self.powermetrics.report(),
            **conditions,
        )
```

Difference `read_bytes`, `read_operations`, cache requests/misses, and every occupancy integral over the same monotonic interval. Emit wall-rate GiB/s, IOPS, bytes/read, expert requests/s, completion tokens/s, mean queue depth, mean active readers, mean queued/active bytes, completion-fence occupancy, and interval booleans for I/O active, fence pending, both, and neither. Keep the first tick plus a bounded recent tail so cumulative summaries cover the whole run. If samples are dropped, mark interval attribution incomplete until the run is repeated with a larger bound.

Expose `q4_assignments_per_second` as the physical Q4 routed-assignment rate,
derived from `cache.expert_requests` only for the Q4 model keys supported by
this harness. Keep `cache_by_layer` deltas in the run summary so layer skew is
not hidden by the aggregate.

Evidence rules are deterministic and named in output:

- SSD saturation is `supported` only when `F_NOCACHE` is active, supplied-ceiling utilization is at least 0.75, reader active-capacity fraction is at least 0.75, and queued-read interval fraction is at least 0.50.
- Reader backpressure is `present` when reader active-capacity fraction is at least 0.75 and queued-read interval fraction is at least 0.50.
- Submission/dependency starvation is only a candidate when uncached reader throughput is below 0.40 of the SSD ceiling, queued-read interval fraction is below 0.10, and cache misses occurred.
- GPU evidence is `unavailable` unless the collector returns measured process GPU time or an explicitly labeled system GPU residency sample.
- RAM bandwidth remains `unavailable`; routed expert bytes are reported as demand, not utilization.
- Fixed thresholds are screening heuristics, not promotion gates or causal cutoffs. Attribution remains `incomplete` with evidence-backed candidates until a matched intervention produces a repeatable throughput response with uncertainty.

Implement `PowermetricsCollector` with `sudo -n`, `--format plist`, `--samplers tasks,disk,cpu_power,gpu_power`, `--show-process-gpu`, `--show-process-samp-norm`, `--show-process-wait-times`, and `--show-process-io`. It writes to a temporary binary stream, stops cleanly, splits NUL-separated plist documents, extracts only the requested PID, and returns `{available: false, reason: ...}` on privilege, parser, or hardware failure. It never prompts.

- [ ] **Step 4: Run telemetry unit tests**

Run: `uv run --no-project --with pytest python -m pytest -q tests/test_resource_telemetry.py`

Expected: PASS, including the absence of `bound_by` and explicit unavailable coverage.

- [ ] **Step 5: Commit**

```bash
git add mtplx/benchmarks/resource_telemetry.py tests/test_resource_telemetry.py
git commit -m "feat(bench): correlate resource throughput evidence"
```

### Task 3: Integrate diagnostic telemetry into both generation lanes

**Files:**
- Modify: `scripts/benchmark_streamed_generation.py`
- Modify: `tests/test_benchmark_streamed_generation_cli.py`

**Security flag:** `none`

**Does NOT cover:** Telemetry-disabled runs intentionally contain no resource fields; telemetry-enabled token rates are labeled diagnostic and do not replace the existing promotion lane. MTP runs have final token throughput but no per-interval token rate because `generate_mtp1` has no token callback.

- [ ] **Step 1: Write failing CLI and row-contract tests**

```python
def test_resource_telemetry_is_opt_in_and_bounded():
    parser = _load_module().build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4"])
    assert args.resource_telemetry is False
    assert args.resource_sample_interval == 0.25
    assert args.resource_max_samples == 4096
    assert args.powermetrics is False


def test_resource_telemetry_flags_parse_without_enabling_window_walks():
    parser = _load_module().build_parser()
    args = parser.parse_args([
        *_BASE_ARGS,
        "--model-key", "hy3-q4",
        "--resource-telemetry",
        "--resource-sample-interval", "0.5",
        "--resource-max-samples", "1024",
        "--ssd-ceiling-gib-s", "12.5",
        "--powermetrics",
        "--no-window-telemetry",
    ])
    assert args.resource_telemetry is True
    assert args.window_telemetry is False
    assert args.ssd_ceiling_gib_s == 12.5
    assert args.powermetrics is True
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `uv run --no-project --with pytest --with mlx python -m pytest -q tests/test_benchmark_streamed_generation_cli.py tests/test_benchmark_streamed_generation_concurrency_cli.py`

Expected: FAIL because the resource flags and integration context do not exist.

- [ ] **Step 3: Add opt-in contexts and row fields**

```python
@contextmanager
def _resource_telemetry(args, runtime, token_count):
    if not args.resource_telemetry:
        yield None
        return
    sampler = ResourceTelemetrySampler(
        runtime.expert_resource_telemetry_snapshot,
        token_count=token_count,
        interval_s=args.resource_sample_interval,
        max_samples=args.resource_max_samples,
    )
    power = PowermetricsCollector(
        enabled=args.powermetrics,
        pid=os.getpid(),
        interval_ms=max(100, int(args.resource_sample_interval * 1000)),
    )
    with sampler, power:
        yield ResourceRun(sampler=sampler, powermetrics=power)
```

Add `--resource-telemetry/--no-resource-telemetry` defaulting false, positive interval and sample-count validators, `--ssd-ceiling-gib-s`, and `--powermetrics/--no-powermetrics` defaulting false. Wrap only the generation window in both `_run_concurrent_repeats()` and the single-stream loop. Record caller-thread CPU time with `time.thread_time_ns()` around the same window. Add to each run:

```python
if run is not None:
    row["diagnostic_run"] = True
    row["resource_telemetry"] = run.report(
        ssd_ceiling_gib_s=args.ssd_ceiling_gib_s,
        generation_thread_cpu_ns=thread_cpu_finished - thread_cpu_started,
        generation_elapsed_ns=int(elapsed * 1e9),
        final_completion_tokens=row_completion_tokens,
    )
```

For a telemetry-enabled concurrent lane, pass an `on_step` callback to
`StreamedBatchRunner` that publishes the token count in already-finalized
results plus the generated tokens of its current live streams. This accounting
is computed by the diagnostic callback rather than the runner's production
finalization path, so a finished stream transfers its final count without a
temporal drop or disabled-lane work. Do not allocate the counter, install the
diagnostic callback, or sample caller-thread CPU in the telemetry-disabled
static lane; mixed-join keeps only its required join-submission callback. The
report schema is additive when enabled; disabled rows contain no
`diagnostic_run` or `resource_telemetry` key. Existing defaults, deterministic
token output, and timing fields remain unchanged. Reject `--powermetrics`
unless `--resource-telemetry` is enabled.

- [ ] **Step 4: Run harness tests**

Run: `uv run --no-project --with pytest --with numpy --with mlx --with safetensors python -m pytest -q tests/test_benchmark_streamed_generation_cli.py tests/test_benchmark_streamed_generation_concurrency_cli.py tests/test_resource_telemetry.py`

Expected: PASS for default-off behavior and both lane contracts.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_streamed_generation.py tests/test_benchmark_streamed_generation_cli.py
git commit -m "feat(bench): add opt-in resource diagnostics"
```

### Task 4: Preserve interpretation rules and verify the complete change

**Files:**
- Create: `docs/RESOURCE_TELEMETRY.md`
- Modify: `project-map.md`
- Modify: `docs/plans/2026-07-13-resource-telemetry.md`

**Security flag:** `none`

**Does NOT cover:** The guide does not claim current hardware is storage-, GPU-, CPU-, or RAM-bound; it defines what future evidence can and cannot prove.

- [ ] **Step 1: Write the agent-facing interpretation guide**

Document:

```markdown
## Reading a resource report

Treat tokens/s as the outcome, not the bottleneck diagnosis. Start with
`coverage`; an unavailable counter is unknown, never zero. Then read physical
throughput, queue/worker occupancy, and coarse coactivity on the same intervals.
The sampled I/O/fence signal does not establish simultaneous overlap within an
interval.

| Observed evidence | Defensible conclusion |
| --- | --- |
| Queue nonempty, readers full, SSD near supplied ceiling | Storage throughput |
| Queue nonempty, readers full, SSD below ceiling | Reader pool, request shape, syscall, or completion path |
| Misses occur, queue empty, SSD low | Submission, routing, prefetch, or dependency starvation |
| Caller thread near one core, SSD/GPU low | Host orchestration candidate |
| Measured GPU activity high, I/O queue empty | GPU compute candidate |
| I/O and completion-fence work rarely appear in the same sampling interval | Coarse I/O/fence separation; add a narrower probe before claiming serialization |

`completion_fences.active_work` means Metal consumer work is outstanding; it is
not GPU utilization. `cache.expert_requests * expert_record_bytes` is routed
weight demand; it is not measured DRAM traffic. `read_mib_per_second` uses
summed reader service time; use telemetry wall-rate SSD GiB/s for device demand.
Do not publish `bound_by` from incomplete evidence.
```

Include exact diagnostic and headline commands. The diagnostic command uses `--resource-telemetry --ssd-ceiling-gib-s 12.5 --powermetrics --no-window-telemetry`; the matched headline command removes `--resource-telemetry` and `--powermetrics` while retaining every model, prompt, cache, seed, and generation flag.

- [ ] **Step 2: Update project context**

Add `docs/RESOURCE_TELEMETRY.md` and `mtplx/benchmarks/resource_telemetry.py` to Key Files. Add a Critical Constraint stating that benchmark attribution must use same-clock throughput plus occupancy evidence, must report missing GPU/DRAM coverage as unavailable, and must never infer serialization from elapsed time or low SSD utilization alone. Refresh the map timestamp and staleness hash to the branch base used for the map update.

- [ ] **Step 3: Run formatting, focused tests, broader tests, and stub scan**

Run:

```bash
uv run --no-project --with ruff ruff check mtplx/resource_metrics.py mtplx/benchmarks/resource_telemetry.py mtplx/expert_slots.py mtplx/expert_runtime.py mtplx/runtime.py scripts/benchmark_streamed_generation.py tests/test_resource_metrics.py tests/test_resource_telemetry.py tests/test_benchmark_streamed_generation_cli.py tests/test_expert_slots_runtime.py
uv run --no-project --with pytest --with numpy --with mlx --with safetensors python -m pytest -q tests/test_resource_metrics.py tests/test_resource_telemetry.py tests/test_benchmark_streamed_generation_cli.py tests/test_benchmark_streamed_generation_concurrency_cli.py tests/test_expert_slots_runtime.py tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
rg -n "TODO|FIXME|placeholder|NotImplementedError|raise NotImplementedError" mtplx/resource_metrics.py mtplx/benchmarks/resource_telemetry.py scripts/benchmark_streamed_generation.py
```

Expected: Ruff exits zero, all selected tests pass, and the stub scan finds no implementation stubs.

- [ ] **Step 4: Measure instrumentation overhead contract**

Run a deterministic synthetic sampler loop 10,000 times with telemetry disabled and enabled. Verify the disabled branch allocates no sampler and adds no telemetry row fields; record enabled collector cost in the PR. On the hardware lane, compare three matched telemetry-disabled repeats with three telemetry-enabled repeats and report median token-rate delta. If enabled overhead exceeds 2%, retain the diagnostic label and investigate counter contention before merge.

- [ ] **Step 5: Commit documentation and completed plan**

```bash
git add docs/RESOURCE_TELEMETRY.md project-map.md docs/plans/2026-07-13-resource-telemetry.md
git commit -m "docs(bench): explain resource evidence"
```

- [ ] **Step 6: Publish and merge**

Push `codex/resource-telemetry`, open a PR against `experiment/moe-pr13-pr14-stack`, include the exact verification commands and the diagnostic-versus-headline distinction, wait for required checks, merge without force-pushing, fetch the remote default branch, and verify the PR merge commit is an ancestor of `origin/experiment/moe-pr13-pr14-stack`.
