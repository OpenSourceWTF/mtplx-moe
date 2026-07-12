## Problem

The current SSD-streaming development layout keeps both:

- the complete MLX checkpoint (~166 GB for Hy3 Q4), including ~161 GB of routed expert tensors; and
- a contiguous expert sidecar (~161 GB) containing those same expert tensors.

The original checkpoint is still used for the always-resident tensors (router/gates, attention, embeddings, norms, and LM head), so the working installation consumes ~327 GB even though only ~166 GB of unique model data is required.

## Proposed design

Build a compact, self-contained streamed artifact:

1. Copy/link tokenizer, config, generation config, chat-template, and model metadata.
2. Repack every non-routed/resident tensor into one or more resident-only safetensors shards.
3. Keep routed expert tensors only in the contiguous `experts.bin` sidecar.
4. Emit a new safetensors index and streaming manifest that jointly cover the complete logical model without duplicate expert storage.
5. Record immutable hashes, source revision, quantization parameters, tensor shapes/dtypes, and sidecar offsets.
6. Teach `mtplx.load()` to load the resident artifact while resolving expert records exclusively through the sidecar.
7. Provide a verifier that proves the compact artifact is tensor-for-tensor equivalent to the source checkpoint before allowing source deletion.

## Safety requirements

- Never delete source shards automatically.
- A separate explicit cleanup command may remove them only after full verification succeeds.
- Write output atomically through a temporary directory and rename on success.
- Refuse output paths inside the source snapshot.
- Detect symlink/path traversal and reject mutable or mismatched source files.
- Preserve provenance sufficient to reproduce and audit the conversion.

## Acceptance criteria

- Hy3 Q4 compact artifact is approximately 5 GB resident + 161 GB sidecar, not ~327 GB total.
- GLM-5.2 Q4 supports the same layout.
- Both models produce matching logits/tokens against the current full-checkpoint + sidecar path on deterministic test prompts.
- Full source/sidecar hash verification and manifest coverage tests pass.
- The existing 1024-in/1024-out benchmark runs from the compact artifact with no material throughput regression.
- Documentation includes build, verify, load, archive, and explicit cleanup commands.

## Non-goals

This issue is storage/layout deduplication. It should not change expert routing, cache policy, numerical quantization, or the active performance experiment except where required to load the compact artifact.
