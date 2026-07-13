# Project Map
_Generated: 2026-07-13 12:30 CDT | Git: c3aca92_

## Directory Structure
.github/ — CI, build, hygiene, release, ownership, and issue/PR automation.
apps/MTPLXApp/ — Native Swift macOS application and app-facing orchestration.
dashboard/ — Vite/TypeScript source for the web dashboard embedded in the Python package.
mtplx/ — Python runtime, CLI, model backends, serving, generation, caching, and benchmarking library.
mtplx/backends/ — Architecture-specific target and MTP backend implementations plus registry metadata.
mtplx/benchmarks/ — Reusable benchmark schemas, runners, validators, and packaged prompt fixtures.
mtplx/models/ — MLX model overlays, including streamed Hy3/GLM routed-expert execution.
mtplx/server/ — OpenAI/Anthropic-compatible service and live dashboard state.
native_extensions/ — Optional nanobind/C++ expert-I/O and verification extensions.
vllm_metal/ — Experimental Metal paged-attention bridge and kernels.
scripts/ — Benchmark, artifact, conversion, parity, training, and operational entrypoints.
tests/ — Unit, integration, lifecycle, parity, CLI, server, and hardware-independent regression tests.
docs/ — Architecture, runtime contracts, benchmark reports, issue specs, plans, and release documentation.
benchmarks/ — Curated benchmark prompts/results; ignored raw run payloads live under benchmarks/raw/.
templates/ — Model/chat templates shipped with the repository.
examples/ — CLI, HTTP, SDK, and benchmark examples.
tools/ — Focused development and benchmark support utilities.
bin/ — Repository-local command wrappers.

## Key Files
pyproject.toml — Package metadata, Python/MLX dependency bounds, optional extras, entrypoints, and test defaults.
mtplx/cli.py — Primary command-line parser and dispatch surface.
mtplx/generation.py — Core autoregressive and multi-token generation loops and generation telemetry.
mtplx/runtime.py — Runtime construction and model execution coordination.
mtplx/engine_session.py — Session lifecycle, commits, concurrency, and state ownership.
mtplx/server/openai.py — OpenAI/Anthropic API server, request scheduling, streaming, and metrics exposure.
mtplx/resource_metrics.py — Telemetry-only, phase-scoped expert-pipeline ledger, state integrals, primary precedence, overlaps, and bounded histograms.
mtplx/expert_runtime.py — Streamed-MoE planning plus authoritative miss lifecycle, potentially blocking next-miss upper-bound timing, route publication, and pipeline snapshots.
mtplx/expert_slots.py — Fixed slot ownership plus reader-task/record lifecycle, pin/loading wait observations, and asynchronous cleanup.
mtplx/expert_io.py — Positional reads with separate logical-range, Python `preadv`, native-call, and returned-byte counters.
mtplx/expert_streaming.py — Hardware-independent expert-cache planning and route telemetry.
mtplx/models/expert_mlx.py — MLX Q4 expert dispatch with telemetry claims immediately before unchanged hit/shared/miss evaluation.
mtplx/models/hy3_mlx.py — Resident Hy3 model overlay and streamed sparse-layer integration.
mtplx/streamed_batch.py — Continuously admitted streamed-AR batch execution.
scripts/benchmark_streamed_generation.py — Canonical streamed-generation benchmark harness and result schema.
scripts/run_issue30_starvation_attribution.py — Safe, provenance-complete eight-run telemetry off/on campaign with exclusive-lane and exact Qwen restoration.
mtplx/benchmarks/resource_telemetry.py — Resource schema v2 differencing and duration-weighted decode primary/orthogonal attribution.
scripts/benchmark_expert_io.py — Expert read-path throughput and concurrency benchmark.
scripts/build_expert_manifest.py — Validated expert-layout manifest builder.
scripts/verify_streamed_parity.py — Deterministic streamed-versus-resident parity gate.
docs/MOE_SSD_STREAMING_OPTIMIZATION_ROADMAP.md — Measured bottlenecks, staged optimization contract, and stop/go rules.
docs/MOE_RUNTIME_PR_BENCHMARKS.md — Curated per-PR runtime correctness and performance evidence.
docs/specs/2026-07-13-hy3-starvation-attribution-design.md — Issue #30 Phase 1 ontology, hook boundaries, coverage limits, and evidence gate.
docs/RESOURCE_TELEMETRY.md — Interpretation contract for resource schema v2 and expert-pipeline attribution.
CONTRIBUTING.md — Required verification, benchmark provenance, and workspace-level worktree rules.

## Critical Constraints
- The GitHub default branch is `experiment/moe-pr13-pr14-stack`, not `main`; verify `origin/HEAD` before branching or pushing.
- Compute paths require Apple Silicon/macOS and MLX 0.31.x, while non-compute CLI imports must remain usable without MLX.
- `transformers` stays below 5.13 until mlx-lm supports its tokenizer registration change.
- SSD-streamed Hy3/GLM execution is opt-in; the resident router remains authoritative and selected experts must be present before dispatch.
- The pinned Hy3/GLM community Q4 artifacts omit declared MTP weights; AR and MTP claims require separate pinned artifacts and benchmark lanes.
- Memory planning is fail-closed: reserve resident weights, KV, runtime headroom, persistent slots, and transient slots under one explicit limit.
- A slot cannot be overwritten until its generation's last Metal consumer is complete; preserve pins, generations, and completion-fence ownership.
- Performance changes require deterministic parity, matched repeated evidence against the immediate predecessor, exact commands, and resource counters.
- Expert-pipeline attribution is telemetry-only on slot-backed layouts; its local lock is never held across runtime work, telemetry-off enters no ledger instrumentation, and the uninstrumented `metal-mmap` path reports it unavailable.
- Records, reader tasks, logical ranges, Python/native calls, and device operations are distinct identities; physical device QD, GPU expert-wait, and GPU idle time remain unavailable.
- Decode primary fractions use covered `decode_observation_ns`; the potentially blocking next-miss step is an upper bound because its readiness scan is non-atomic and its end hook can wait for the ledger lock. Exact generation-thread expert-input blocked time remains unavailable, orthogonal overlaps remain independent evidence, and neither form is itself a bottleneck claim.
- Bottleneck attribution requires same-clock throughput plus occupancy evidence and matched telemetry-off reproduction; no fixed worker target or state percentage is causal.
- Raw benchmark output belongs only under ignored `benchmarks/raw/<benchmark>/<run-id>/`; commit curated summaries with full provenance.
- Auxiliary worktrees are flat direct children of the workspace-level `.worktrees/` directory and must not be created inside the clone.
- Do not silently fall back for unsupported models, layouts, quantization, missing artifacts, integrity failures, or memory-plan breaches.
- Treat cache policy, prefetch, MTP, lower precision, and continuous batching as separate claims with independent gates.

## Hot Files
tests/test_expert_slots_runtime.py, mtplx/expert_runtime.py, mtplx/models/expert_mlx.py, tests/test_streamed_models.py, mtplx/expert_slots.py, mtplx/expert_streaming.py, scripts/benchmark_streamed_generation.py, mtplx/models/hy3_mlx.py
