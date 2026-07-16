# Experimental SSD-streamed MoE

The `codex/moe-ssd-hy3-glm52` branch contains an opt-in autoregressive path for the pinned `pipenetwork/Hy3-4bit` and `mlx-community/GLM-5.2-4bit` artifacts. It keeps attention, shared experts, and each router resident; the router runs first at every sparse layer, and only its selected Q4 experts are read into a user-bounded hot slot bank. Fixed MLX/Metal byte banks are filled in place with positional reads, so routed expert tensors are never instantiated as a full parameter tree.

This is not enabled by default and has not yet earned a full-checkpoint performance claim. Tiny end-to-end model fixtures, failure injection, exact artifact-layout audits, GLM FP32 near-tie routing, and IndexShare tests pass. The 166 GB Hy3 and 418 GB GLM Q4 checkpoints still require hardware parity and sustained benchmark runs using the included gates.

```bash
# Optional native GIL-free positional reader
DEBUG=0 uv pip install -e native_extensions/expert_io

# Build and verify an exact manifest; an aligned sidecar is optional
uv run python scripts/build_expert_manifest.py /path/to/model --model hy3-q4
uv run python scripts/verify_expert_manifest.py /path/to/model /path/to/model/expert-manifest.json

# The total threshold includes resident weights, KV reservation, slots, and reserve.
mtplx serve --model /path/to/model \
  --expert-streaming \
  --expert-manifest /path/to/model/expert-manifest.json \
  --expert-memory-limit 96GiB \
  --expert-max-live-kv-tokens 32768
```

For a 128 GB Mac, this is the conservative starting profile for the pinned
GLM-5.2 affine-Q4 checkpoint. The 104 GiB ceiling leaves system headroom while
reserving 16 GiB inside the runtime plan; MTP is disabled automatically because
the checkpoint does not contain its declared MTP layer.

```bash
MODEL=/path/to/mlx-community/GLM-5.2-4bit

uv run python scripts/build_expert_manifest.py "$MODEL" --model glm52-q4
uv run python scripts/verify_expert_manifest.py \
  "$MODEL" "$MODEL/expert-manifest.json" --records

uv run mtplx serve --model "$MODEL" \
  --expert-streaming \
  --expert-model-key glm52-q4 \
  --expert-manifest "$MODEL/expert-manifest.json" \
  --expert-memory-limit 104GiB \
  --expert-runtime-reserve 16GiB \
  --expert-max-live-kv-tokens 32768
```

These two community Q4 artifacts omit their declared MTP layers, so streamed loading intentionally selects target-only AR and rejects MTP adapters. See [the implementation and validation guide](../MOE_SSD_STREAMING_PLAN.md) for pinned revisions, GLM IndexShare details, sidecar packaging, memory planning, parity, and benchmark commands.
