# SSD-streamed MoE optimization roadmap

This roadmap targets the measured Hy3 and GLM-5.2 affine-Q4 streamed paths on
the 128 GB M5 Max. It keeps Python as the control plane, the router resident,
the expert cache user-bounded, and the model artifacts bit-identical. Each
stage must earn its place with a matched benchmark before the next stage.

## Current baseline

- Hy3 Q4 source shards, realistic 313-token prompt:
  - 8.14 prompt tok/s cold, 9.42 warm.
  - 1.17 decode tok/s cold, 1.36 warm.
  - 83.5 GB peak MLX memory.
- Hy3 Q4 verified sidecar, complete 1,391-token response:
  - 12.62 prompt tok/s.
  - 1.39 decode tok/s; 1.57 best 32-token window.
  - 83.69 GB peak MLX memory.
- GLM-5.2 Q4 verified sidecar, short warm trace:
  - about 0.49 decode tok/s.
  - about 90.4 GB active MLX memory.
- Live Hy3 sidecar sample:
  - 80–88% host CPU idle; Python process about 1.2 cores.
  - 0.74–1.34 GB/s disk throughput.
  - cool package, indicating scheduler/I/O/Metal bubbles rather than thermal
    throttling.
- A Python grouped-weight experiment changed the same first-256 window from
  1.3225 to 1.3261 tok/s (+0.3%). It was removed because copying slot tensors
  cancels the launch savings.

The immediate bottleneck is serialized expert fault handling and low SSD queue
depth. The predicted next bottleneck is small-batch Metal occupancy and
router-to-host synchronization. At long context, KV/attention bandwidth will
eventually replace expert I/O as the limiting resource.

## Benchmark contract

Run every optimization in these lanes:

1. **Matched Qwen AR lane**
   - exact 512- and 1,024-token MTPLX prompt builds;
   - 128 output tokens;
   - thinking off, temperature 0.6, top-p 0.95, top-k 20;
   - AR only, because the pinned Hy3/GLM Q4 artifacts omit MTP weights;
   - compare with the recorded Qwen3.6 27B AR result (about 24.6 tok/s).
2. **Realistic single-stream lane**
   - the checked-in 313-token engineering prompt;
   - natural EOS with the model-specific 65,536-token default ceiling;
   - complete generated response saved alongside raw telemetry.
3. **Context lane**
   - 4K, 16K, 64K, and at most 256K admitted context;
   - GLM uses the 124 GiB/256K/50-slots-per-layer plan;
   - Hy3 uses the 115 GiB/69,632-token/93-slots-per-layer default plan and a
     separately labeled reduced-expert-cache profile for literal 256K.
4. **Saturation lane**
   - batch/concurrency 1, 2, 4, and 8;
   - report both per-stream latency and aggregate output tok/s.

Record prompt tok/s, decode tok/s, end-to-end tok/s, TTFT, 32-token rolling
decode, expert hit rate by phase, bytes/token, read operations, disk MB/s, CPU
idle, MLX active/peak memory, swap, output hash, finish reason, and all I/O or
integrity errors.

An optimization is retained only if it gives at least 5% on its target lane in
repeated runs, does not materially regress another lane, preserves generated
token parity in deterministic mode, and stays inside the memory plan.

## Stage 1 — saturate the sidecar read path

1. Raise transient/I/O slots from 8 to 32. This adds about 243 MiB for Hy3 and
   486 MiB for GLM while retaining the 93- and 50-slot persistent banks.
2. Raise the positional-read chunk from 8 MiB to 64 MiB so a 10.125 MiB Hy3 or
   20.25 MiB GLM sidecar record is filled by one native request.
3. During prefill, sort missing records by sidecar offset.
4. Coalesce adjacent records with a native scatter positional read into their
   already allocated destination slots. The sidecar records are aligned and
   contiguous, so a high-fanout prompt can become a mostly sequential scan.
5. Benchmark normal cached I/O, read-ahead, and `F_NOCACHE` modes. Keep the mode
   that maximizes sustained reads without duplicating the 80 GB application
   cache in the macOS file cache.

**Expected next bottleneck:** Metal execution and host synchronization once the
SSD reaches several GB/s. Do not claim a target in advance; use the disk trace
to determine whether the internal SSD or expert execution becomes limiting.

## Stage 2 — stop rereading and rehashing avoidable work

1. Let a cold prefill initialize only empty persistent slots with the most
   frequently selected experts for that layer. Prefill must never evict an
   existing decode-hot expert.
2. Preserve decode-only TinyLFU aging after the initial fill.
3. Add a dynamic per-layer capacity experiment. Lend slots from low-entropy
   layers to high-miss layers while keeping the global byte ceiling exact.
4. Add integrity tiers:
   - `record`: current per-load SHA-256, always safe and the default;
   - `verified-sidecar`: full sidecar SHA-256 at process start plus immutable
     file identity checks, then skip repeated hashes of the same bytes;
   - no unchecked mode in the server path.
5. Persist only frequency metadata, never live MLX slot references, across
   sessions. Prewarm from the profile under the same byte admission gate.

**Expected next bottleneck:** Q4 compute/dispatch on high-hit decode. Measure
bytes/token after this stage; a cache-policy win must reduce physical reads, not
only improve a logical counter.

## Stage 3 — overlap resident hits with misses

1. Split each route plan into immediately pinnable persistent hits and pending
   misses.
2. Submit miss reads to the existing GIL-free native I/O pool.
3. Execute hit experts on Metal while misses are in flight.
4. Evaluate miss experts when their slots become ready, then scatter both
   result sets back to router order.
5. Replace device-wide synchronization with per-wave completion events. Slot
   reuse remains fenced by the last consumer of that slot.
6. Add cancellation, deadline, generation-number, and exception tests for every
   split-route transition.

**Expected next bottleneck:** small Q4 kernels and the per-layer router-ID host
round trip. This stage should raise SSD and Metal duty cycle without requiring
a driver change.

## Stage 4 — maximize MLX work per dispatch

1. Add an all-hit fixed-shape graph for batch sizes 1, 2, 4, and 8.
2. Pass an indirect expert-to-slot table to a custom MLX/Metal operation so
   selected weights remain in their existing slot buffers. Do not stack or copy
   whole expert records in Python.
3. Fuse gate Q4, up Q4, SwiGLU, down Q4, router weighting, and scatter for the
   selected assignments where MLX/Metal supports it.
4. Keep the Python route/cache policy initially; move only the proven hot data
   path native if profiling still attributes meaningful time to Python.
5. Re-run deterministic token parity after every fusion boundary.

**Expected next bottleneck:** unified-memory bandwidth and attention/KV at long
context. At batch 1, transformer layer dependencies limit available parallel
work even after launch overhead is removed.

## Stage 5 — continuous batching for aggregate throughput

1. Add a streamed-AR batch runner using the already supported `[B, 1, H]`
   decode shape.
2. At each sparse layer, take the union of selected experts across live
   sequences, load each missing record once, and run assignments grouped by
   expert.
3. Separate prefill and decode queues so a large prefill cannot stall active
   decoders or evict their hot experts.
4. Tune batches 1, 2, 4, and 8 against latency SLOs. Report aggregate tok/s and
   per-stream tok/s separately.

This is the most direct way to use more of the M5 Max without pretending that a
single autoregressive sequence contains unlimited parallelism.

## Stage 6 — long-context KV and attention

1. Validate paged attention for both streamed overlays.
2. Add Q8 KV first, then evaluate Q4 KV only with long-context quality gates.
3. Reinvest saved KV bytes into expert slots or runtime headroom.
4. Tune prompt chunks only after measuring expert rereads. Whole-prefill
   grouping can be better for expert reuse; smaller chunks can be better for
   memory and responsiveness.
5. Cap this project’s benchmark matrix at 256K even though the pinned GLM
   config advertises a 1M architectural context.

**Expected next bottleneck:** attention and KV memory bandwidth. Expert-side
optimizations should no longer be credited for time spent in attention.

## Optional later stages

- Add a native event-driven route/cache scheduler only after profiling proves
  Python control overhead is material once I/O and Metal are overlapped.
- Add MTP or another speculative decoder only when a separately pinned and
  validated draft artifact exists. MTP is a separate acceleration claim and
  must not be mixed into the AR comparison lane.
- Evaluate lower-bit experts only as a separate quality/performance artifact;
  never relabel a different quantization as the current affine-Q4 model.

## Stop/go criteria

Stop and fix correctness before performance if any run shows a hash mismatch,
short read, stale slot generation, memory-plan breach, unexplained token drift,
or output truncation presented as a complete answer. Promote an optimization
only with saved raw results, the exact command/profile, and a reproducible
comparison against its immediate predecessor.
