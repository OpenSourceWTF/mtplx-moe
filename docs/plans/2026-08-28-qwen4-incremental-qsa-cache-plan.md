# Qwen4 incremental QSA cache implementation plan

1. Add focused cache tests that compare chunked incremental QSA selection with
   a from-scratch oracle, exercise rollback within/across ratio-4 blocks, and
   cover composite snapshot restoration plus the 16K capacity boundary.
2. Convert `_IndexerCache` raw storage to capacity-backed append semantics and
   add a capacity-backed derived pooled-key store.
3. Route `QSAIndexer.select_projected` through the cached completed-block prefix
   while leaving scoring and mask construction unchanged.
4. Confirm the exact capture-commit benchmark lane does not install graphbank,
   then run focused QSA, projection-fusion, cache-state, and resident-model tests
   plus formatting.
5. Run the exact 16K/1K production candidate under
   `/tmp/mtplx-gpu-exclusive.lock`, verify the stable digest, and bracket it with
   the unchanged control if the first result is promising.
6. Commit and push only a verified improvement, then update PR #368's benchmark
   history and optimization summary.
