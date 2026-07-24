> **Repository scope:** This is a repository-wide `davidtai/MTPLX` issue.
> **Target branch:** `codex/moe-ssd-hy3-glm52`.
> **Implementation status:** the resident-only GLM-5.2 overlay, FP32 router,
> IndexShare schedule, streamed Q4 switch, strict loader, and tiny end-to-end
> forward are implemented. Full-checkpoint parity and memory benchmarks remain.

## Objective

Add a GLM-5.2 Q4 adapter that loads only resident tensors at startup, preserves
official router and DSA/IndexShare behavior, and replaces routed expert storage
with fixed streamed slots for layers 3-77.

The pinned target is `mlx-community/GLM-5.2-4bit` revision
`6b347a6472d46bf55de65ee34032136a3929d778` (91 shards,
418,320,895,488 tensor bytes). The behavioral oracle is the official GLM-5.2
Transformers implementation, not a guessed interpretation of the older Q4
config.

## Required architecture semantics

- 78 target layers indexed 0-77; layers 0-2 are dense and layers 3-77 are MoE.
- 256 routed experts/layer, exact top-8, plus one resident shared expert.
- Hidden size 6144; routed expert intermediate size 2048.
- The resident router projection must execute in FP32. Apply sigmoid, add the
  FP32 correction bias for **selection only**, gather the original unbiased
  sigmoid scores, normalize selected scores, and multiply by routing scale 2.5.
- The Q4 router weight tensors are stored as BF16 and correction biases as F32;
  explicitly cast the router matmul rather than inheriting a BF16 MLX default.
- DSA token selection and MoE expert routing are independent resident selectors.
  DSA selection must not trigger expert loads or share cache-policy state.
- IndexShare has 21 full indexer layers: `0, 1, 2, 6, 10, ..., 74`. The other
  57 layers reuse the previous full layer's selected token indices.
- The 4-bit artifact omits MTP layer 78 despite config metadata. Autoregressive
  loading must accept that omission and must not search/load every shard.

## Proposed files and integration seam

- `mtplx/models/glm52_mlx.py`
  - resident GLM-5.2 module wrapper, FP32 router, IndexShare schedule, resident
    shared expert, and streamed `HotExpertSwitchGLU`.
- `mtplx/models/hot_expert_switch_glu.py`
  - model-neutral Q4 slot-backed gate/up/down execution consuming resolved slot
    IDs while preserving route weights and token aggregation.
- `mtplx/resident_loader.py`
  - allowlisted resident-tensor load that never invokes generic full-model
    `mlx_lm.load` before expert tensors are intercepted.
- `mtplx/runtime.py`
  - branch to the streamed loader before the current generic load seam.
- `tests/test_glm52_streamed_mtp.py`
  - tensor allowlist, router math, IndexShare, shape, and end-to-end parity.

Base IndexShare support on the relevant upstream `mlx-lm` GLM-5.2 work (PR
#1410 and its test follow-up), but pin/vendor the minimum reviewed behavior until
an official released `mlx-lm` dependency contains it. Do not depend on current
main behavior that instantiates an indexer on every layer.

## Resident-load contract

The startup loader must enumerate an exact allowlist and validate all expected
keys. It should load approximately 10,634,546,688 bytes of non-routed tensors,
including about 236,006,400 bytes of routers/correction biases, attention/DSA,
shared experts, dense MLP layers 0-2, embeddings, norms, and output head. Routed
expert records remain represented only by the manifest plus fixed slot buffers.

Generic `mlx_lm.load` currently materializes the whole model and is therefore
not an acceptable precursor to swapping in streamed modules.

## Failure handling

- Fail on missing/extra resident keys, unexpected tensor shapes/dtypes, wrong
  layer schedule, artifact revision, quantization group, or router metadata.
- Fail if route IDs/weights differ from the official FP32 oracle; never silently
  use BF16 router matmul for speed.
- Fail if a shared IndexShare layer has no valid full-layer indices in the same
  forward pass. Do not instantiate random/missing indexer parameters.
- Reject attempts to enter MTP mode when explicit validated layer-78 weights are
  unavailable. Do not create a random separate shared output head.
- Missing expert records propagate a typed streaming error; no whole-shard or
  whole-checkpoint fallback is allowed.

## Acceptance criteria

- [ ] A load-spy test proves startup opens/loads the exact resident allowlist and
      does not read routed expert payloads before the first route miss.
- [ ] Resident tensor bytes and router bytes match 10,634,546,688 and
      236,006,400 respectively for the pinned artifact.
- [ ] For fixed hidden-state fixtures, all 75 sparse layers match official
      top-8 IDs and normalized/scaled route weights, including near-tie cases
      where BF16 and FP32 selection would differ.
- [x] Tests prove correction bias changes selection only and does not contaminate
      the gathered routing weight.
- [ ] IndexShare tests assert exactly 21 full indexer executions and 57 reuses;
      sequences longer than 2,048 and 4,096 tokens exercise reuse behavior.
- [ ] Full-resident test slots and streamed slots produce equivalent layer
      outputs within a documented Q4 numerical tolerance.
- [x] Autoregressive model execution works without layer 78 and requested MTP mode
      fails early with a clear missing-weight error.
- [ ] Peak memory is governed by the resolved plan and not the 389.59 GiB total
      Q4 artifact size.

## Dependencies

- Depends on router/memory descriptors and GLM-5.2 manifest support.
- End-to-end streamed execution depends on the native slot loader.
- Parity/benchmark work supplies the official oracle fixtures and release gate.
- MTP support is intentionally deferred to the separate validation/MTP gate.
