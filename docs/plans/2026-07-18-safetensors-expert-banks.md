# Sharded safetensors expert banks

**Status: reader IMPLEMENTED 2026-07-18 (commit fff6a23). Converter not written.**
Written 2026-07-18. Every number below was
verified from disk; the "verified" markers say how.

## The two questions

> I don't think the big sequential file needs to be a single file for pread
> does it

It does not. `pread` is per-fd at an arbitrary offset; nothing about it wants
one file. The single-file constraint is **entirely artificial** and lives in
two places:

- `SidecarInfo.file` is a scalar `str` (`mtplx/expert_manifest.py`)
- `validate_structure` hard-rejects more than one sidecar shard:
  `"an authoritative manifest requires exactly one sidecar shard"`

The reader is *already* multi-file on one of its two paths: the source-segment
path issues one `_read_range_into(segment.shard, ...)` per segment, and each
`TensorSegment` carries its own `shard` name — so a record's 9 components may
legally live in 9 different files **today**. Only the sidecar path
(`experts.bin`) is single-file.

> can you make the system fully compatible with safetensors?

Yes, and the current layout is already byte-compatible with it. The custom
`.bin` is not buying anything that safetensors doesn't give for free.

---

## Why it works — three verified facts

**1. A record is exactly its 9 components, contiguous, with no gaps.**
Verified on `hy3-expert-only-mlx-q2` record (layer 1, expert 0): segments at
offsets 0 / 1,572,864 / 1,769,472 / … / 5,701,632, each starting exactly where
the previous ended. Span = 5,898,240 = `logical_bytes` exactly.

So an expert record is *already* just "9 tensors written back to back". That is
precisely what a safetensors data section is.

**2. safetensors packs tensors with zero padding.**
Verified against `model-00001-of-00018.safetensors`: 0 gaps across all tensors;
each `data_offsets` pair begins where the previous ended. Layout is
`[8-byte header length][JSON header][contiguous data]`, with tensor offsets
relative to the start of the data section.

Reading a tensor is therefore `pread(fd, data_start + offset, length)` — which
is *exactly the operation the expert reader already performs.*

**3. Record sizes are exact multiples of the page size.**

| Model | record bytes | ÷ 16384 |
|---|---|---|
| hy3 q2 | 5,898,240 | **360.0000** |
| glm52 q2 | 11,796,480 | **720.0000** |

macOS page size here is 16384, and the manifest's `DEFAULT_ALIGNMENT` is also
16384. So **if the data section starts at a 16 KiB-aligned offset and records
are written adjacently, every record boundary is automatically both 16
KiB-aligned and page-aligned — with no inter-record padding at all.**

That third fact is what makes this clean rather than a compromise. The two
alignment regimes that matter (`DEFAULT_ALIGNMENT` for the manifest contract,
`mmap_u32`'s page requirement for the `metal-mmap` slot layout) collapse into
the same constraint, and it is satisfied for free.

### The one thing needing care

`data_start` is `8 + header_len`, and in the existing shards that is **3,504** —
not aligned. Fix: pad the JSON header with trailing whitespace until
`8 + header_len ≡ 0 (mod 16384)`. This is spec-legal (the header is JSON;
trailing whitespace is insignificant) and costs at most 16 KiB per shard. HF's
own writer already pads to 8; we pad further.

Without that padding the `pread` path still works perfectly — only the
`metal-mmap` slot layout would be unable to map records directly.

---

## Target layout

```
hy3-expert-q2/
  config.json  tokenizer.json  tokenizer_config.json  ...
  model-00001-of-00018.safetensors        # residents, unchanged
  ...
  model.safetensors.index.json            # residents, unchanged
  experts-00001-of-00006.safetensors      # ~15 GiB each, HF-native
  ...
  expert-manifest.json                    # (layer,expert) -> (shard, range)
```

89.46 GB of hy3 q2 experts → 6 shards at ~15 GB, comfortably inside HF's 50 GB
hard per-file limit and near the ~20 GB practical recommendation. GLM's 226 GB
→ ~15 shards.

Tensor names inside an expert shard keep the existing convention
(`layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`), so
the shards are loadable by anything that reads safetensors — including plain
`mlx.core.load` — without MTPLX at all. **That is what "fully compatible"
buys: the artifact stops being ours.**

---

## Manifest changes

`SidecarInfo` gains parts; the scalar stays as the single-part spelling so
existing manifests keep parsing:

```jsonc
"sidecar": {
  "parts": [
    {"file": "experts-00001-of-00006.safetensors",
     "size": 16106127360, "sha256": "...", "data_start": 16384},
    ...
  ]
}
```

`ExpertRecord` gains a `part` index; `sidecar_offset` becomes relative to that
part's `data_start` rather than absolute into one giant file.

Rules:
- **A record never straddles a part.** Records are 5.6–11.3 MiB against ~15 GiB
  parts, so whole-record packing wastes at most one record per part. This is
  what keeps the change to file selection instead of the range machinery.
- Per-shard `sha256` + `size` replace the single whole-sidecar digest. Per-record
  `sha256` is unchanged and remains the hot-path authority — it is already
  position-independent, which is why sharding doesn't disturb it. Per-shard
  digests also give HF-side download verification per file.

---

## Code changes

| Site | Change |
|---|---|
| `expert_manifest.py` `SidecarInfo` | parts list; keep scalar as 1-part |
| `expert_manifest.py` `validate_structure` | drop the single-sidecar rule; `segment.shard` becomes set-membership; bounds/non-overlap become per-part |
| `expert_manifest.py` `ExpertRecord` | `part: int = 0` |
| `expert_io.py` sidecar read | select the part fd (the LRU fd cache already handles N files) |
| `expert_io.py` batch coalescing | must not coalesce across a part boundary |
| `expert_io.py` codec path | same part selection |
| `models/expert_mlx.py` | single `self.path` → per-part |
| `build_expert_sidecar.py` | emit safetensors shards + aligned header padding, roll to a new part on size limit |
| `expert_banked.py` | **fix first, unrelated:** `BankedManifest.file` is read as a bare `str()` with no `_safe_relative_name` — the lone manifest that skips path validation. Becomes a real vector once artifacts are downloaded from the Hub. |

`resolve_artifact_member` already permits subdirectory components and rejects
absolute paths, `..`, and backslashes — so shard names need no new validation.
Caveat: `expert_manifest.py`'s safetensors cross-check uses a **non-recursive**
`glob("*.safetensors")` compared by basename, so expert shards must sit at the
artifact root, not in a subdirectory, unless that glob is fixed too.

---

## Migration

Non-destructive and reversible. The existing `experts.bin` is a *concatenation*
of records that are already contiguous, so conversion is a copy with a header
written per part — no re-quantization, no re-encoding, and per-record sha256
values carry over unchanged and re-verify after the move.

Ship the parts-aware reader first (it reads 1-part manifests identically), then
convert artifacts one at a time.

---

## What this does not solve

- The rANS/banked variants keep their own manifests and their own offset
  conventions (banked offsets are region-relative — a second, incompatible
  scheme). They are experimental arms, not publishable deliverables, and are out
  of scope.
- **The `rans-shards` artifact is unloadable, and that is the code working.**
  `expert-streamed-codec-rans32x.json` carries `size` 12,784,969,412 and
  shard00's `sha256` while describing a 194,261,709,700-byte file — both
  copied verbatim from `rans-shards/shard00.json` instead of being recomputed
  for the concatenation. Verified by direct comparison.

  But it never gets that far: `load_streamed_codec_manifest` **rejects it
  outright** with `streamed codec record (8, 0) is not aligned`, so the
  rebased offsets are also not on the 16 KiB boundary the format requires.
  It fails closed at load, unconditionally — not only under
  `verify_sidecar_hash_at_open`, which is what I first assumed.

  So there is **no code fix owed here.** The validation caught a hand-merged
  artifact, which is exactly its job. The artifact is an experimental arm and
  is not a publishable deliverable; the producing script is not in version
  control. If that lane is ever revived, the merge step needs to recompute
  `size`/`sha256` and re-align offsets — which the parts-aware design above
  would make unnecessary, because per-shard digests replace the concatenated
  whole-file digest entirely.
- Publishing itself. `forge publish` is one `upload_folder` call with no
  chunking, resume, retry, or filtering; a failed 90 GB upload leaves
  `bytes_uploaded: 0` and no way to resume. Sharding makes that *survivable*
  (a failed part is 15 GB, not 90) but does not make it good.
- Metadata scrubbing before publish: `hy3-q4-mlx-mtp/mtplx_runtime.json`
  embeds five absolute `/Users/davidtai/...` paths and an `intended_hf_repo`;
  `conversion-manifest.json` embeds another. No secrets were found — token
  handling is stdin-only and test-enforced — but the paths leak build-machine
  detail and should be stripped by the publish path.


---

## Implementation notes 2026-07-18 — where this plan was wrong

The reader landed as specified. Three corrections found while building it:

**1. The single-sidecar rule was not the only blocker — and the converter will
hit the other one.** Sidecar-kind shards are additionally required to have
`header_bytes == 0` and `header_sha256 == EMPTY_SHA256`. That is correct for a
raw `.bin`, and it directly contradicts a safetensors-framed part, which has a
real JSON header. The reader sidesteps it by carrying the framing in
`SidecarPart.data_start` and leaving the shard-kind rule alone. **A converter
that writes real safetensors parts must reconcile this**, and the change table
above does not mention it. This is the single most important omission here.

**2. `replace(sidecar, sha256=...)` is load-bearing.** An existing test
(`test_make_sidecar_authoritative_rejects_stale_input_digest`) relies on it, so
the scalar single-part spelling has to work as an *override on an existing
bank*, not merely as a construction form. A clean "parts XOR scalars"
constructor breaks it.

**3. The io tests duck-type.** `test_expert_io_metrics.py` passes
`SimpleNamespace` records and sidecars, so reading `record.part` /
`sidecar.parts` directly raises `AttributeError`. The read path needs `getattr`
tolerance.

Backward compatibility was verified against the real artifacts, not fixtures:
hy3 q2/q4 (15,168 records) and glm52 q2 (19,200 records, 62 MB) each parse,
re-serialize to a dict identical to what is on disk, and keep their
`manifest_sha256`.

Left out deliberately: the converter; `expert_banked.py` (its
`BankedManifest.file` gap is real but unrelated — banked/rANS arms stay
single-part and raise clearly on a multi-part bank); the codec sidecar
(`StreamedCodecManifest` is a different type with its own offset convention);
and the non-recursive `glob("*.safetensors")`, so parts must live at the
artifact root — a converter constraint, not a reader bug.

Unrelated pre-existing defect noticed in passing:
`hy3-q4-mlx-mtp/conversion-manifest.json` fails to parse with
`unknown keys: ['auxiliary_files']`.
