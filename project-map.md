# Project Map
_Generated: 2026-07-13 06:06 CDT | Git: fb9268a_

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
mtplx/expert_runtime.py — Streamed-MoE memory planning, route admission, cache policy, and runtime snapshots.
mtplx/expert_slots.py — Fixed slot-bank ownership, generations, pins, completion fences, and I/O lifecycle.
mtplx/expert_io.py — Positional expert reads, descriptor caching, integrity checks, and I/O counters.
mtplx/expert_streaming.py — Hardware-independent expert-cache planning and route telemetry.
mtplx/models/expert_mlx.py — MLX Q4 expert dispatch, hit/miss overlap, evaluation, and slot-release ordering.
mtplx/models/hy3_mlx.py — Resident Hy3 model overlay and streamed sparse-layer integration.
mtplx/streamed_batch.py — Continuously admitted streamed-AR batch execution.
scripts/benchmark_streamed_generation.py — Canonical streamed-generation benchmark harness and result schema.
scripts/benchmark_expert_io.py — Expert read-path throughput and concurrency benchmark.
scripts/build_expert_manifest.py — Validated expert-layout manifest builder.
scripts/verify_streamed_parity.py — Deterministic streamed-versus-resident parity gate.
docs/MOE_SSD_STREAMING_OPTIMIZATION_ROADMAP.md — Measured bottlenecks, staged optimization contract, and stop/go rules.
docs/MOE_RUNTIME_PR_BENCHMARKS.md — Curated per-PR runtime correctness and performance evidence.
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
- Raw benchmark output belongs only under ignored `benchmarks/raw/<benchmark>/<run-id>/`; commit curated summaries with full provenance.
- Auxiliary worktrees are flat direct children of the workspace-level `.worktrees/` directory and must not be created inside the clone.
- Do not silently fall back for unsupported models, layouts, quantization, missing artifacts, integrity failures, or memory-plan breaches.
- Treat cache policy, prefetch, MTP, lower precision, and continuous batching as separate claims with independent gates.

## Hot Files
tests/test_expert_slots_runtime.py, mtplx/expert_runtime.py, mtplx/models/expert_mlx.py, tests/test_streamed_models.py, mtplx/expert_slots.py, mtplx/expert_streaming.py, scripts/benchmark_streamed_generation.py, mtplx/models/hy3_mlx.py
