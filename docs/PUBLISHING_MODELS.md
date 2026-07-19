# Publishing MTPLX models to Hugging Face

Everything below is implemented and tested. The one thing missing is a
Hugging Face **write token** — there is none on this machine (no
`~/.cache/huggingface/token`, nothing in the environment), so the upload step
cannot be run unattended.

## Why this needed work at all

`experts.bin` is one file: **89.46 GB** for hy3 q2, **226 GB** for glm52 q2.
Hugging Face's hard per-file limit is **50 GB** (~20 GB is the practical
recommendation), so these artifacts could not be uploaded at all. The
single-file layout was never a technical requirement — `pread` does not care
how many files there are — it was `SidecarInfo.file` being a scalar plus a
`validate_structure` rule rejecting more than one sidecar shard.

Both are lifted. A bank can now span parts, and a one-part manifest still
parses byte-identically, so unconverted artifacts keep loading unchanged.

## Step 1 — plan the split (read-only, seconds)

```bash
python3 scripts/shard_expert_bank.py ~/.cache/huggingface/hy3-expert-only-mlx-q2 \
  --out /tmp/plan --dry-run
```

Real output today:

```
15168 records, 83.32 GiB -> 6 parts
  experts-00001-of-00006.safetensors: 2730 records, 15.00 GiB
  ...
  experts-00006-of-00006.safetensors: 1518 records,  8.34 GiB
```

## Step 2 — convert (a ~90 GB copy)

```bash
python3 scripts/shard_expert_bank.py ~/.cache/huggingface/hy3-expert-only-mlx-q2 \
  --out ~/publish/hy3-expert-q2
```

- The source artifact is opened **read-only** and never modified (there is a
  test asserting exactly that).
- It is a copy, not a re-quantization: an expert record is already its 9
  components written back to back, which is what a safetensors data section
  is. No weight is touched.
- Per-record sha256 is re-verified during the copy; the converter aborts if a
  digest changes.
- Needs ~90 GB free for hy3 (~226 GB for glm52). Do not run it while a
  benchmark holds the machine — it is I/O heavy and will distort timings.

## Step 3 — copy the rest of the artifact

The expert bank is not the whole model. Also required for a third party to
load it:

```
config.json  tokenizer.json  tokenizer_config.json  special_tokens_map.json
generation_config.json  model-*.safetensors  model.safetensors.index.json
```

`hy3-mtp-layer80` has **no tokenizer at all**, so it is not loadable as a
standalone repo — publish it alongside a trunk or not at all.

## Step 4 — publish

```bash
mtplx forge publish ~/publish/hy3-expert-q2 --repo <org>/<name> --token stdin
```

Token is read from stdin, never argv, and never logged — there is a test
asserting it does not land in `publish.json` or `mtplx_runtime.json`.

Publishing scrubs machine-identifying provenance automatically: absolute
`/Users/...` paths in `forge_provenance.forge_inputs` and `intended_hf_repo`
are replaced before upload, via a staged copy plus
`ignore_patterns=["mtplx_runtime.json"]`, so the raw paths never enter the
published repo's git history and your local provenance survives intact.

**Known gap:** `upload_folder` is a single blocking call with no chunking,
resume, or retry. Sharding makes a failure survivable — a failed part is
15 GB, not 90 — but a mid-upload failure still restarts that part.

## Step 5 — verify a fresh consumer

```bash
mtplx pull <org>/<name>          # add --no-expert-banks for a config-only pull
```

Completeness is expert-aware: a model whose expert manifest or bank is missing
or truncated is reported incomplete rather than silently "ready". A deliberate
`--no-expert-banks` pull is recorded and stays distinguishable from a corrupt
one.

## What is guaranteed

- **Records round-trip bit-exact.** Every record is read back through the real
  `PositionalExpertReader` with `verify_hash=True`, so a wrong part or offset
  fails loudly instead of returning the wrong expert.
- **Parts are ordinary safetensors.** Verified with plain `mlx.core.load` —
  no MTPLX involved. That is the point: the artifact stops being ours.
- **Alignment is preserved.** Record starts are aligned exactly as
  `build_expert_sidecar` does, so the `metal-mmap` slot layout keeps working.
- **Per-part digests.** Each part carries its own `size` + `sha256`, which
  also gives HF-side per-file download verification. Per-record sha256 is
  unchanged and remains the hot-path authority — it is position-independent,
  which is why sharding is safe at all.

## Still open

- **No HF token on this machine.** Blocks steps 4 and 5.
- **Version skew.** PyPI ships `mtplx` **2.1.0**; this branch is **2.0.2**.
  Someone who `pip install mtplx` today gets code this branch does not
  contain: bounded MLX allocator cache, re-clamped session admission under
  96 GB, a q4 kv-quant crash fix on the split-SDPA path, and `--memory-budget`.

  Measured, so the size of the decision is clear: the 2.1.0 line
  (`a391973`, authored by Youssof) is **8 commits / 95 files / +8,791 lines**
  ahead of the merge base, and merging it into the default branch produces
  **12 conflicting regions**. That is a real integration of someone else's
  release, not a version bump — and bumping the version string without the
  code would be worse than the skew, since it would claim fixes that are not
  present.
