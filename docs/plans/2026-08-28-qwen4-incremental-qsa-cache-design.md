# Qwen4 incremental QSA pooled-key cache

## Objective

Remove the repeated full-context QSA key-pooling work from Qwen4 target
verification while preserving the existing QSA scoring, visibility, top-k, and
attention arithmetic exactly.  The acceptance workload is the production
16,384-token Python prompt followed by 1,024 generated tokens at native-MTP
depth 1 and the production thinking sampler.

## Measured problem

The post-projection-fusion MLX trace shows that each full-attention layer starts
verification by revisiting the entire raw indexer-key history.  Across the 12
full-attention layers, that start command buffer costs about 1.87 ms per target
verification.  `QSAIndexer.select_projected` currently concatenates the new raw
keys to the complete history, then recomputes mean pooling, key normalization,
and key RoPE for all complete ratio-4 blocks.

At 16K context that is 4,096 completed blocks.  A depth-1 verify appends only two
rows, so it completes zero or one new blocks.  Rebuilding all 4,096 is work that
does not depend on the current query.

## Design

`_IndexerCache` remains the sole owner of indexer-key rollback.  It gains two
capacity-backed stores:

- raw projected keys, indexed by token offset;
- normalized and RoPE-applied pooled keys, indexed by completed ratio-4 block.

Appending raw keys writes only the new rows into fixed 2,048-token capacity
steps and returns only the logical visible prefix, never the physical capacity
buffer.  The first allocation reserves one extra step.  A 16,384-token prefill
therefore owns capacity through 18,432 tokens and does not copy the complete raw
bank inside the measured 1,024-token decode.  `QSAIndexer` derives the first
missing completed block from the cache's pooled offset, computes only the
missing suffix with the existing float32 mean, existing `RMSNorm`, and existing
partial RoPE, then appends that suffix to the pooled store.  Scoring, visibility,
top-k selection, partial-block retention, and the attention mask remain
unchanged.

The only hot decisions depend on runtime-varying sequence length: whether raw
or pooled capacity must grow and whether this append completed a new block.
Model topology, ratio, dtype, and arithmetic are not revalidated per call.

## Rollback ownership

`QSAKVCache.trim` already owns target-KV and indexer rollback together.  A trim
sets the raw visible offset to the accepted token prefix and sets the pooled
visible offset to `floor(raw_offset / compress_ratio)`.  Capacity beyond either
offset may remain allocated but is never visible; the next append overwrites
it.

This preserves every pooled block wholly contained in the accepted prefix.  If
a trim crosses a completed-block boundary, only the now-incomplete final block
is dropped and recomputed when later completed.  It avoids invalidating the
entire pooled bank after normal speculative rejection.

`QSAKVCache.state` becomes a composite of target K/V and the logical raw
indexer-key prefix so prompt/session snapshots no longer silently omit QSA
state.  State restoration reconstructs both visible offsets and invalidates the
derived pooled store.  A derived cache is never serialized as authoritative
model state.  Generic snapshot/restore rollback is therefore correct but must
rebuild the derived pool; the measured `capture_commit` route uses direct
accepted-prefix trimming and preserves all wholly accepted pooled blocks.
The custom cache type is registered once at model-module import so MLX's
name-based prompt-cache loader can reconstruct the composite state.

The compiled graphbank cache promotion currently cannot carry nested QSA state.
This experiment is limited to the exact eager/capture-commit M=2 route and does
not claim graphbank compatibility.  Graphbank must remain uninstalled for this
lane until its state spec explicitly owns the raw indexer leaves.

## Correctness gates

- Incremental masks must equal a from-scratch full-history QSA mask exactly.
- Trimming inside and across a compression block must reproduce a clean cache
  rebuilt from the accepted prefix exactly.
- Raw and pooled visible offsets must track accepted tokens and completed
  blocks across append/trim/append sequences.
- Composite state snapshot/restore must reproduce a clean cache and invalidate
  only the derived pooled bank.
- Generic `rollback_after_verify` must remain output-correct; the production
  capture-commit trim must preserve the accepted pooled prefix.
- A 16,384-token initial append must reserve enough raw capacity to avoid growth
  through the 1,024-token measured generation window.
- Existing Qwen4 projection-fusion and resident-model tests must remain green.
- The production digest must match the accepted control receipt.

## Performance gate

Run the exact production harness with one warmup and the sustained 16K/1K
profile.  Compare the candidate against an unchanged control under the same GPU
lock and service conditions.  Promote only a repeatable decode-throughput or
wall-time improvement with the stable output digest.  Record the commit,
settings, TPS, wall time, acceptance counts, and delta in PR #368's benchmark
history.

Memory accounting includes both allocated raw and pooled backing stores even
though only raw logical state is serialized.
