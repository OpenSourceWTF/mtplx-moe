> **Repository scope:** This is a repository-wide `davidtai/MTPLX` issue.
> **Target branch:** `codex/moe-ssd-hy3-glm52`.
> **Implementation status:** the current policy produces planned `SlotLoad`
> operations only; no native SSD-to-MLX/Metal transfer path exists yet.

## Objective

Implement a bounded native loader that fills preallocated quantized expert
slots from manifest-described records. The loader must preserve buffer identity
used by compiled MLX graphs, prevent stale-slot execution, and expose explicit
completion rather than relying on incidental Python or MLX synchronization.

At each sparse layer, the resident router runs first. Its top-k IDs are passed
to `LayerExpertSlotBank.plan(...)`; only the returned misses issue reads. All
loads for that layer must reach the requested generation before the expert GLU
dispatch. Routers for later layers cannot run early because their hidden states
do not exist yet.

## Proposed files and API

- `mtplx/expert_io.py`
  - Python ownership, manifest lookup, queue selection, cancellation, metrics,
    and a portable/mock backend for tests.
- `native_extensions/expert_io/`
  - nanobind C++/Objective-C++ extension using `pread`/`preadv` into bounded
    staging memory and explicit Metal/MLX buffer copies.
- `mtplx/expert_slots.py`
  - fixed per-layer slot tensors, slot-generation table, state transitions,
    and `ensure_route(...)` orchestration.
- `tests/test_expert_io.py` and `tests/test_expert_slots.py`
  - short reads, cancellation, races, reuse, eviction, and stale generation.

Proposed boundary:

```python
class ExpertSlotLoader:
    def ensure_route(
        self,
        *,
        layer: int,
        plan: RoutePlan,
        deadline_ns: int | None = None,
    ) -> ReadyRoute: ...

class ReadyRoute:
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    completion: CompletionEvent
```

Each physical slot has a state machine:

```text
EMPTY -> LOADING(expert, generation) -> READY(expert, generation)
                |                           |
                +-> FAILED ----------------+
READY -> LOADING(next_expert, generation + 1)
```

Expert compute receives both slot IDs and generations. Dispatch is allowed only
after the completion event and only if the slot table still maps every pair to
the routed expert. A transient slot is reusable after that layer's expert
dispatch completes; a persistent slot remains ready until policy eviction.

## Buffer and I/O contract

- Allocate all slot tensors once from the resolved memory plan. Their addresses
  and shapes must remain stable across record replacement.
- Preserve the source Q4 affine layout (packed weights plus BF16 scales/biases)
  and never dequantize whole experts into resident buffers.
- Start with a correctness path that performs aligned bounded reads and explicit
  copies/fences. Add direct-storage optimizations only after proving MLX buffer
  lifetime and coherence on supported macOS/Apple Silicon versions.
- Coalesce segments for a record when the manifest/sidecar permits it. Do not
  open or page unrelated checkpoint shards.
- Bound file descriptors, in-flight bytes, staging buffers, and queue depth.
  These bytes must be included in the runtime reserve or an explicit I/O budget.
- Deduplicate concurrent requests for the same `(layer, expert, generation)`.

## Failure handling

- Treat `EINTR` as retryable; treat EOF/short read, checksum mismatch, closed
  file, invalid range, deadline expiry, or Metal copy failure as a failed slot.
- A failed or cancelled read never transitions to `READY` and never overwrites
  a newer generation. Wake all waiters with the same typed error.
- Keep the prior resident record valid until replacement is fully loaded when
  buffer topology permits; otherwise mark the slot unavailable before writing.
- On device loss or unrecoverable extension error, fail affected requests and
  require runtime reconstruction. Do not continue with stale data.
- Never substitute zeros, a different expert, or a broad checkpoint load.

## Acceptance criteria

- [ ] Mock-backend tests exercise every state transition, concurrent dedupe,
      cancellation, timeout, eviction during load, and generation ABA hazard.
- [ ] Native tests fill a slot from known bytes, run an MLX operation against
      it, replace the record in the same slot, and observe the new result while
      preserving planned allocation size.
- [ ] Fault-injection tests for short reads and payload corruption fail before
      expert dispatch and leave no slot falsely marked ready.
- [ ] Repeated cold/warm routes never allocate beyond fixed slot buffers plus
      configured bounded staging/in-flight I/O memory.
- [ ] Hy3 can load one 10.125 MiB record and GLM-5.2 one 20.25 MiB record with
      all packed weights/scales/biases matching source bytes.
- [ ] One top-8 transient bank is safely reused across sequential layers: 81
      MiB for Hy3 and 162 MiB for GLM-5.2.
- [ ] Metrics expose requested/read bytes, read latency, queue latency, deduped
      loads, failures, cancellations, and wait time per layer.
- [ ] A no-fallback test verifies a missing expert record does not cause all
      safetensors shards to be opened.

## Dependencies

- Blocked by the validated manifest/sidecar schema.
- Consumes `RoutePlan` and fixed slot counts from router/memory planning.
- Blocks end-to-end model adapters and runtime cache-policy integration.
