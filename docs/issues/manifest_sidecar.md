> **Repository scope:** This is a repository-wide `davidtai/MTPLX` issue.
> **Target branch:** `codex/moe-ssd-hy3-glm52`.
> **Implementation status:** this proposes a new artifact contract; no trusted
> expert manifest or aligned sidecar generator is implemented yet.

## Objective

Create a versioned, validated mapping from each `(layer, expert)` to the exact
Q4 tensor segments required by the streamed GLU. Support direct reads from the
original safetensors shards and an optional locally generated, contiguous,
aligned sidecar optimized for `pread`.

The loader must not infer offsets from filenames, assume every model has the
same packing order, or scan/materialize all checkpoint shards when a tensor is
missing. Artifact identity and layout are part of the correctness boundary.

## Proposed files and commands

- `mtplx/expert_manifest.py`
  - frozen schema types, strict JSON parsing, artifact verification, and
    `(layer, expert) -> ExpertRecord` lookup.
- `scripts/build_expert_manifest.py`
  - read safetensors headers only, validate the selected model descriptor, and
    emit a deterministic manifest.
- `scripts/build_expert_sidecar.py`
  - copy validated segments to one or more local sidecar files with 16 KiB
    record alignment; never publish or upload artifacts automatically.
- `docs/expert-manifest-v1.schema.json`
  - machine-readable schema with `additionalProperties: false` at signed
    boundaries.
- `tests/test_expert_manifest.py`
  - fixtures for cross-shard records, corruption, truncation, duplicate keys,
    integer overflow, symlink/path escape, and revision mismatch.

Example logical schema:

```json
{
  "format": "mtplx-expert-manifest-v1",
  "model_key": "glm52-q4",
  "source_repo": "mlx-community/GLM-5.2-4bit",
  "source_revision": "6b347a6472d46bf55de65ee34032136a3929d778",
  "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
  "artifact": {"tensor_bytes": 418320895488, "shard_count": 91},
  "records": [
    {
      "layer": 3,
      "expert": 0,
      "logical_bytes": 21233664,
      "segments": [
        {
          "tensor": "...gate_proj.weight",
          "shard": "model-00001-of-00091.safetensors",
          "offset": 0,
          "length": 0,
          "dtype": "U32",
          "shape": [256, 2048, 768],
          "expert_axis": 0,
          "expert_index": 0
        }
      ]
    }
  ]
}
```

The example's offsets are placeholders; generation must derive and validate
real absolute payload offsets from safetensors headers. Each logical record
contains the selected expert's gate/up/down packed weights plus their scales
and biases. A sidecar form replaces the segment list with file, aligned offset,
length, and checksum while retaining source provenance.

## Validation rules

- Pin source repository and immutable revision; record shard relative paths,
  exact file sizes, safetensors header hashes, and a whole-manifest digest.
- Reject absolute paths, `..`, symlink escapes from the artifact root, overlap,
  out-of-file ranges, duplicate `(layer, expert)` keys, unsupported endianness,
  unexpected dtype/shape, and non-integer offsets.
- Validate per-projection quantized weights, scales, and biases rather than
  checking only aggregate bytes.
- Require a complete rectangular routed set:
  - Hy3: `79 * 192` records of 10.125 MiB.
  - GLM-5.2: `75 * 256` records of 20.25 MiB.
- Verify aggregate routed bytes exactly:
  - Hy3: 161,036,107,776 bytes.
  - GLM-5.2: 407,686,348,800 bytes.
- Keep resident routers, shared experts, and dense layers out of the expert
  records and verify their expected keys separately.
- Sidecar generation must use bounded buffers and streaming copies; it may not
  mmap or materialize the entire 150-380 GiB routed corpus in RAM.

## Failure handling

- Fail closed before model allocation on any provenance, schema, file-size,
  layout, or checksum mismatch.
- A short read or changed source file invalidates the record and sidecar build;
  never pad with zeros or continue to another shard.
- Write sidecars to a temporary filename, `fsync`, validate all record hashes,
  then atomically rename. Clean incomplete temporaries on the next invocation.
- Do not redistribute third-party weights. Sidecars are generated locally and
  retain source license/provenance metadata.
- Never fall back to broad shard loading if a requested key or MTP layer is
  absent; return a typed missing-record error.

## Acceptance criteria

- [ ] Manifest generation from each pinned Q4 artifact is deterministic and a
      second run is byte-for-byte identical.
- [ ] The record count, logical record size, aggregate routed bytes, layer
      range, expert range, and quantization tensor shapes match the invariants
      above for Hy3 and GLM-5.2.
- [ ] Randomly sampled records reconstruct the same nine Q4 tensor slices as the
      source safetensors reader, byte-for-byte.
- [ ] Every sidecar record begins at a 16 KiB-aligned offset and a full audit
      verifies hashes and source provenance after generation.
- [ ] Tests corrupt a header, payload byte, size, path, offset, shape, revision,
      and checksum and assert an early typed failure.
- [ ] A memory-usage test proves manifest/sidecar generation remains bounded by
      the configured copy buffer rather than artifact size.
- [ ] The native loader can query a record without opening unrelated shards.

## Dependencies

- Consumes exact tensor/layout values from `ExpertStreamingModelSpec`.
- Blocks the native slot-loader end-to-end path.
- Can be developed and unit-tested in parallel with router/memory planning.
