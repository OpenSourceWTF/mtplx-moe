# GLM-5.2 Q4 target with shared BF16 MTP

## Status

Approved in conversation on 2026-07-14. The requested behavior is the same
external BF16 layer-78 MTP attachment already used by `glm52-expert-q2`, with
the target changed to the pinned `glm52-q4` streamed model.

## Premise

The work should exist, but it should remain small. The GLM-specific MTP loader,
cache semantics, generation path, and authenticated BF16 artifact already
work for the Q2 target. The runtime currently authorizes that backend only for
the `glm52-expert-q2` model key, so the otherwise compatible Q4 target cannot
request the same head through the supported fail-closed path.

The proportional fix is to authorize and verify the existing Q4 descriptor.
It is not proportional to copy hundreds of gigabytes, rewrite the community
checkpoint, or create a second MTP implementation.

Without this work, the local Q4 target remains AR-only in MTPLX even though the
verified official BF16 MTP head is available locally.

## Precision contract

The combined runtime pair has this exact precision split:

- target routed experts: 19,200 MLX affine-Q4 records, group size 64, with BF16
  scales and biases;
- target router weights: 75 BF16 matrices;
- target router correction biases: 75 F32 vectors;
- MTP layer 78: 790 BF16 tensors and one F32 router correction bias;
- MTP routed experts: resident BF16 weights, not Q4 and not Q2;
- target embeddings and LM head: shared by the MTP path, not duplicated.

The Q4 target remains byte-for-byte unchanged. The MTP artifact remains
byte-for-byte unchanged. "Combined" means one authenticated runtime pairing,
not a rewritten Hugging Face checkpoint.

## Considered approaches

### 1. Reuse the external-MTP runtime pairing (selected)

Add `glm52-q4` to the existing GLM streamed-MTP backend registry. Reuse the
same artifact verification, strict layer construction, memory accounting,
cache semantics, and generation path as `glm52-expert-q2`.

This has the smallest blast radius, preserves both source packages, and makes
one BF16 MTP head usable by both Q2 and Q4 targets.

### 2. Create a physical combined directory (rejected)

Hard-link or copy the 769 GB Q4 target and 19.9 GB MTP artifact under a new
root. This does not improve runtime correctness and introduces another package
whose links, provenance, and lifecycle must be maintained.

### 3. Insert layer 78 into the community MLX checkpoint (rejected)

Rewrite its index and shards as a conventional checkpoint. Stock `mlx-lm`
explicitly sanitizes layers at or beyond `num_hidden_layers`, so this is both
expensive and incompatible with the loader behavior the external adapter was
created to handle.

## Scope

- Authorize `glm52-q4` to use the existing GLM BF16 streamed-MTP backend.
- Preserve existing Q2 behavior and all AR-only defaults.
- Add regression tests proving Q2 and Q4 select the same GLM MTP backend while
  unsupported MTP precision still fails before target allocation.
- Prove the real Q4 manifest and router precision contract from local files.
- Deep-verify the real BF16 MTP artifact before a model allocation.
- Run a guarded real-model AR/D1 smoke comparison with deterministic sampling,
  exact token equivalence, and full cleanup of the exclusive MLX lane.

## Non-goals

- Do not quantize any MTP tensor.
- Do not change, requantize, or repack the Q4 target.
- Do not download additional GLM shards.
- Do not create a new model key or duplicate model package.
- Do not change default serving behavior; MTP remains explicit and off by
  default.
- Do not claim a performance win or promote an MTP depth from a smoke test.

## Architecture and data flow

The caller supplies the existing Q4 target root, its expert manifest,
`model_key="glm52-q4"`, `mtp=True`, `mtp_precision="bf16"`, and the external
MTP artifact directory.

Before target allocation, the runtime:

1. resolves `glm52-q4` to the GLM MTP backend and rejects any non-BF16 MTP
   request;
2. validates the GLM MTP contract and deep-verifies the artifact revision,
   inventory, and payload hashes;
3. charges the exact 19,905,841,664-byte MTP payload to the fixed memory plan;
4. constructs and strictly loads the layer-78 module;
5. opens the unchanged Q4 streamed target and injects the prebuilt MTP module;
6. exposes the existing AR and recurrent MTP generation interfaces.

The verified artifact context remains open until target construction and
injection finish, matching the Q2 lifecycle.

## Interfaces and contracts

No public function signature changes. The existing `mtplx.runtime.load()`
arguments remain authoritative:

```python
load(
    q4_root,
    mtp=True,
    expert_streaming_config=config_for("glm52-q4"),
    expert_manifest=q4_root / "expert-manifest.json",
    mtp_artifacts=glm52_mtp_root,
    mtp_precision="bf16",
)
```

The Q4 target revision remains
`mlx-community/GLM-5.2-4bit@6b347a6472d46bf55de65ee34032136a3929d778`.
The MTP source revision remains
`zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`.

## Error handling

Fail before target allocation when:

- the model key does not support streamed MTP;
- MTP precision is not BF16;
- the MTP directory, manifest, payload, revision, inventory, shape, or dtype is
  invalid;
- the target/MTP memory plan does not fit;
- the GLM MTP contract is incompatible;
- strict MTP construction or injection fails.

Every failure must close an opened artifact context and any partially opened
runtime. AR loading without `mtp=True` remains unchanged.

## Verification strategy

Use test-driven development:

1. Add failing tests showing `glm52-q4` does not currently select the GLM MTP
   backend.
2. Parameterize dispatch/lifecycle tests over `glm52-expert-q2` and
   `glm52-q4`.
3. Retain negative tests for Q4 MTP precision and pre-allocation provenance,
   contract, and memory failures.
4. Run the focused GLM artifact, runtime, model-descriptor, and generation
   suites, then the full repository suite and Ruff.
5. Re-audit the real Q4 manifest/header inventory and deep-verify the real MTP
   artifact.
6. Under the shared hardware lock, capture Qwen state, stop it only for the
   exclusive window, run deterministic paired AR and D1 generation, require
   identical output token IDs, then restore and verify the exact prior Qwen
   state before releasing the lane.

## Failure-mode assessment

- **Critical: wrong target or MTP revision.** Existing fail-closed provenance
  checks remain mandatory and run before allocation.
- **Critical: Q4 target plus the 19.9 GB head exceeds the configured fixed
  memory plan.** Preserve exact additional-resident-byte accounting and reject
  before allocating either model component.
- **Minor: D1 is correct but slower or has poor acceptance.** Record the smoke
  result without promotion; a separate repeated campaign is required for any
  performance decision.

## Rollout

The behavior stays opt-in. Successful tests and one exact real-model smoke run
establish that the same BF16 MTP works with both Q2 and Q4 targets. They do not
change defaults, delete the Q2 target, or select an MTP depth automatically.
