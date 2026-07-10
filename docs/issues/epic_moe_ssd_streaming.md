> **Repository scope:** Issues are repository-wide in `davidtai/MTPLX`.
> **Target implementation branch:** `codex/moe-ssd-hy3-glm52`.
> **Current state:** the branch has a pure-Python expert-cache policy scaffold and
> simulator. It does **not** yet have a native SSD-to-MLX loader or a working
> streamed model adapter.

## Goal

Run Hy3 Q4 and GLM-5.2 Q4 on Apple Silicon without materializing every routed
expert in unified memory. Keep attention, embeddings, normalization, shared
experts, and every MoE router resident. At each MoE layer, run that layer's
router first, resolve its top-k experts against a bounded slot bank, load only
the missing expert records, and then execute the routed MLP.

Routing cannot be done for all layers at the beginning of a token. The router
at layer `n` consumes the hidden state produced by layer `n - 1`; therefore the
correct execution order is sequential and layer-local:

1. Execute the resident attention/residual path for layer `n`.
2. Evaluate the resident router for layer `n`.
3. Produce top-k expert IDs and unbiased routing weights.
4. Map IDs to persistent hot slots or shared transient slots.
5. Load missing expert records from SSD and wait for their slot generations.
6. Execute the expert GLU and resident shared expert, then continue to `n + 1`.

The user controls a total-memory threshold and may also set a narrower expert
cache cap. The planner must subtract the fixed resident model, context-dependent
KV/DSA state, runtime reserve, and one reusable transient top-k bank before it
allocates persistent expert slots. No code path may exceed that plan or silently
fall back to loading every checkpoint shard.

## Pinned model targets

| Target | Sparse layers | Experts/layer | Top-k | Q4 expert record | Cold routed bytes/token | Fixed non-routed artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hy3 Q4 | 79 | 192 | 8 | 10.125 MiB | 6.249 GiB | 4.612 GiB |
| GLM-5.2 Q4 | 75 (layers 3-77) | 256 | 8 | 20.25 MiB | 11.865 GiB | 9.904 GiB |

For GLM-5.2, the 75 resident routers plus correction biases occupy about
0.2198 GiB and are included in the fixed non-routed footprint. The Q4 artifact
is about 389.59 GiB in total. GLM-5.2's DSA token indexer is distinct from the
MoE expert router; both selectors stay resident, but only the MoE selection
causes expert records to be loaded.

## Proposed work breakdown

- **Router-first execution and memory budgeting:** define model descriptors,
  exact slot sizing, public configuration, and the layer-local state machine.
- **Validated manifest and optional aligned sidecar:** turn safetensors tensor
  slices into deterministic expert records without trusting guessed offsets.
- **Native slot loader:** copy records into preallocated MLX/Metal buffers with
  explicit slot generations, fences, cancellation, and bounded I/O.
- **GLM-5.2 adapter:** preserve official FP32 routing and IndexShare semantics
  while loading only the resident subset at startup.
- **Runtime cache policy:** connect the existing policy scaffold to prefill,
  decode, batching, admission, telemetry, and server/CLI configuration.
- **Parity, benchmarks, and MTP gates:** prove routing/output correctness before
  reporting throughput; add MTP only from explicit, validated weights.

## Tracked implementation issues

- [ ] #4 — Router-first execution and user memory budgeting.
- [ ] #1 — Versioned expert manifest and aligned sidecar generator.
- [ ] #3 — Native SSD-to-MLX slot loader with generations and fences.
- [ ] #2 — GLM-5.2 Q4 adapter with FP32 router and IndexShare.
- [ ] #6 — Runtime cache policy, batching, configuration, and telemetry.
- [ ] #5 — AR parity, benchmarks, and explicit MTP artifacts.

## Proposed top-level API

```python
runtime = load_runtime(
    model_path,
    expert_streaming=True,
    memory_limit_bytes=96 * 2**30,
    expert_cache_limit_bytes=None,
    runtime_reserve_bytes=16 * 2**30,
    max_live_kv_tokens=32768,
    io_staging_bytes=512 * 2**20,
)
```

Suggested CLI equivalents:

```text
--expert-streaming
--memory-limit-gib 96
--expert-cache-limit-gib 64       # optional hard sub-cap
--expert-runtime-reserve-gib 16
--expert-max-live-kv-tokens 32768
--expert-io-staging-gib 0.5
--expert-manifest /path/to/expert-manifest.json
--expert-sidecar /path/to/experts.bin  # optional
```

The resolved plan must be printed before allocation and exposed through runtime
metadata: fixed bytes, KV/DSA bytes at the requested context, transient bytes,
persistent slots per layer, known I/O/workspace bytes, persistent bytes, and
unallocated safety margin. Runtime integration must reconcile this ceiling
with `MTPLX_MEMORY_LIMIT_BYTES` and apply the MLX cap before allocation.

## Failure handling

- Reject an unknown model revision, manifest version, quantization layout, file
  size, tensor shape/dtype, or record checksum before allocating GPU buffers.
- Fail before model load when the total threshold cannot fit the fixed resident
  subset, requested context state, runtime reserve, and transient bank.
- Treat short reads, checksum failures, slot-generation mismatches, and Metal
  command failures as fatal for the affected request; never execute a partially
  populated or stale slot.
- Never use the current GLM MTP fallback that scans/loads all shards when the
  expected MTP layer is absent. Autoregressive mode must work without MTP.
- Do not silently change router dtype, top-k, routing weights, IndexShare reuse,
  or quantization parameters to make a checkpoint load.

## Epic acceptance criteria

- [ ] Hy3 Q4 autoregressive generation runs through the streamed path with 79
      resident routers, top-8 expert selection, and bounded expert buffers.
- [ ] GLM-5.2 Q4 autoregressive generation runs through the streamed path with
      layers 3-77 routed, official FP32 router semantics, and correct IndexShare.
- [ ] Measured peak allocated model/cache memory remains at or below the user's
      total-memory threshold in cold, warm, prefill, decode, cancellation, and
      concurrent-request tests.
- [ ] A zero-persistent-slot plan remains correct by serving every miss from the
      reusable transient bank; an undersized fixed-footprint plan fails early.
- [ ] Route IDs, route weights, logits, and generated tokens pass the parity
      gates defined in the validation issue before performance claims are made.
- [ ] Corrupt/truncated manifests, sidecars, and checkpoint shards fail closed;
      tests prove there is no whole-checkpoint fallback.
- [ ] Benchmarks report time-to-first-token, decode tokens/s, cache hit rate,
      bytes read/token, I/O wait, eviction count, and peak unified memory for
      cold and warmed workloads.
- [ ] Documentation includes a reproducible artifact revision, manifest build
      command, memory plan, launch command, and current limitations.

## Dependency order

1. Router/memory-budget contract and manifest format can proceed in parallel.
2. Native loading depends on the manifest format and slot-buffer contract.
3. Hy3 and GLM adapters depend on the router contract; end-to-end streaming
   additionally depends on the native loader.
4. Runtime policy integration depends on the model adapter and slot loader.
5. Parity/benchmark gates consume all prior work. MTP remains a separate final
   gate and is not required for the first autoregressive milestone.
