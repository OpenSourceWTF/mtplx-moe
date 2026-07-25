# OpenSourceWTF MTPLX-MOE experts.bin CLI fork design

Date: 2026-07-24
Status: approved for implementation
Target branch: `release/230-experts-cli`
Base: `ship/upstream-230-hy3-mtp` at `e560cb2`

## Goal

Make the source-installed OpenSourceWTF fork support a complete first-run path:

```bash
python3 -m pip install "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

The command must download the resident weights, `expert-manifest.json`, and
literal `experts.bin`; validate the artifact before model load; select a
measured memory envelope; start the local OpenAI-compatible server; and accept
OpenAI and LiteLLM Chat Completions without expert-specific flags.

## Scope

- Integrate the exact SSD-streaming implementation from `davidtai/MTPLX`
  into the OpenSourceWTF fork based on upstream 2.3.0 without changing its expert
  arithmetic, ownership, tiling, record layout, or compiled execution paths.
- Reuse the completed `feat/moe-streaming-oob` work for manifest-authoritative
  model identity, alias resolution, automatic streaming activation, artifact
  inspection, and LiteLLM documentation.
- Package only benchmark-promoted Hy3 oQ2e serve profiles and expose them
  through one public `--expert-profile` option.
- Admit downloaded artifacts once, before model construction. The enabled
  streaming hot path must not perform per-record hashes, eligibility checks,
  environment reads, fallback accounting, or engagement instrumentation.
- Keep this first compatibility release on the fork's already-validated AR
  serve route. Report that route explicitly instead of silently advertising
  MTP.
- Preserve the existing OpenAI Chat Completions, Completions, Models, and
  Anthropic Messages endpoints and the non-loopback API-key requirement.

## Non-goals

- Adding `/v1/responses` or claiming full OpenAI API compatibility.
- Publishing to PyPI, Homebrew, or Hugging Face before the release gates pass.
- Automatically enabling candidate sweep profiles that have only an admission
  estimate and no promoted runtime benchmark.
- Porting GLM 5.2 or experimental q1/rANS/mixed-official configurations into
  the first release slice.
- Enabling MTP for streamed serving before each production profile has a
  measured correctness and throughput gate at its supported context lengths.
- Adding runtime fallbacks from the installed streaming lane to stock MLX.

## Integration strategy

The expert fork and the upstream 2.3.0 line share merge base `510ac8c`. The
expert fork is hundreds of commits ahead and the upstream line contains newer
public CLI and server work. Copying expert modules by filename would risk
separating the runtime from the exact model patches and ownership rules it
depends on.

Implementation will therefore merge the fork history into the release branch
and resolve shared-file conflicts semantically:

- Upstream 2.3 remains authoritative for package versioning, generic CLI
  behavior, public server behavior, settings, and non-expert model paths.
- The fork remains authoritative for streamed-expert formats, readers, memory
  planning, Hy3 model execution, MTP integration, and their tests.
- Conflicts in `runtime.py`, `commands/public.py`, `server/openai.py`,
  `hf_loader.py`, and `cli.py` are resolved by retaining upstream behavior and
  adding the expert route at construction or request-admission boundaries.
- The `feat/moe-streaming-oob` commits are applied after the merge so their
  zero-flag behavior is evaluated against the final 2.3 surfaces.

The source-only fork build is identified as `2.3.0+opensourcewtf.moe`. It is
not an official MTPLX release and is not published to PyPI or Homebrew.

## Artifact admission

`expert-manifest.json` is authoritative for streamed model identity. Its
`model_key` must name an installed model specification and must agree with the
manifest geometry. `config.json.model_type` is only a fallback for legacy
artifacts without a manifest key.

`mtplx pull` downloads expert banks by default. At completion it validates:

1. Repository revision and complete file inventory.
2. Indexed resident shards and tokenizer/config files.
3. Manifest structure, model key, record bounds, shapes, dtypes, and headers.
4. The `experts.bin` size and SHA-256 declared by the manifest and Hugging Face
   LFS metadata.

Successful admission writes an atomic receipt outside the immutable artifact,
under the MTPLX cache, keyed by repository revision, manifest digest, bank
digest, and file identity. Interrupted or mismatched downloads never receive a
receipt and cannot be served.

A local path produced by `hf download` is also supported. If no matching
receipt exists, the CLI performs the same one-time bank verification before
construction and stores a receipt in the MTPLX receipt cache. This may make the
first launch slower, but subsequent launches do not rehash the bank.

After a valid receipt is installed, construction sets
`verify_record_hashes=False`. There is no enabled hot-path branch back to
per-read validation. Per-record hashing remains available only in an explicit
diagnostic configuration.

## Installed serve profiles

Promoted profiles are package data, not repository-relative benchmark files:

| Public name | Process ceiling | Weight envelope | Promoted geometry | Route |
|---|---:|---:|---|---|
| `hy3-oq2e-64` | 71 GiB | 64 GiB | zero islands and a 49.9921875 GiB frequency cache | AR |
| `hy3-oq2e-88` | 95 GiB | 88 GiB | 74 islands and five streamed layers | AR |
| `hy3-oq2e-96` | 103 GiB | 96 GiB | all 79 routed layers pinned | AR |

Each profile installs an immutable construction contract containing the exact
model key, memory limit, reserve, KV admission, cache scope and policy, slot
layout, island placement count, router kernel, resident requantization choice,
integrity mode, and generation mode. The 88 and 96 GiB rows reproduce their
benchmark configurations directly. The 64 GiB product row retains the measured
zero-island/cache-heavy execution geometry but uses its 4K KV control ceiling;
its 71 GiB memory limit is the byte-exact result of the admission sweep, while
the performance A/B used the same execution geometry at 16K KV and a
corresponding 74.75 GiB ceiling.

The 64 GiB profile is the measured cache-heavy winner from the three-repetition
64 GiB island-versus-cache A/B. The 88 GiB profile is the measured five-streamed
layer configuration. The 96 GiB profile is the measured fully pinned control.
Their immutable provenance is:

| Profiles | Evidence commit | Receipts |
|---|---|---|
| 88 and 96 GiB | `191ed9aa362e645f48f1a105a6ec024ea4fd5cf4` | `evals/tier2/NOTES.md` and the oQ2e campaign artifacts referenced there |
| 64 GiB cache-heavy | `14c8b57fff358bee3da2d10968a855b955b86847` | `evals/tier2/t3_64x16k_arm{A,B_frequency,B_lru}.json`, per-repetition receipts, and `research/envelope-admission-sweep-2026-07-22.json` |

The 64 GiB evidence commit currently lives on `eval/hy3-q2-2p6bit`; only its
promoted immutable configuration and provenance are consumed here. Its
experimental code and unrelated evaluation artifacts are not transplanted.
The 48, 56, 72, and 80 GiB sweep entries remain benchmark candidates and are
not production-selectable in this release.

The CLI exposes:

```text
--expert-profile {auto,hy3-oq2e-64,hy3-oq2e-88,hy3-oq2e-96}
```

`auto` chooses the largest promoted profile that fits installed RAM and current
preflight availability. An explicit profile still runs the same preflight. If
it cannot fit, startup fails once with the required and available bytes; it
does not fall through to a smaller or stock route.

Advanced `--expert-streaming-config` remains supported. Explicit individual
expert flags override a named profile, but the completed configuration must
pass the same construction checks before the lane is installed.

## Generation routing

Each first-release profile constructs the existing AR callable directly. The
server reports `generation_mode=ar` during startup and in health metadata.
There is no `generation_mode=auto` branch, MTP eligibility check, or
try-MTP-then-AR fallback in token generation. An explicit MTP request against
these profiles is rejected before generation with a message that no
streamed-serving MTP profile is installed.

This deliberately preserves the proven `feat/moe-streaming-oob` behavior.
Shipping MTP later requires a separately promoted profile whose exact
streaming geometry, context length, correctness, and unchanged-AR control have
all passed the benchmark gate.

## User and client flow

The primary flow is:

```bash
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

An explicit envelope is:

```bash
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --expert-profile hy3-oq2e-64
```

OpenAI clients use `http://127.0.0.1:8000/v1`, any non-empty local API key, and
the model returned by `/v1/models`. LiteLLM uses provider model
`openai/<served-model-id>` with the same base URL. Documentation calls this
OpenAI-compatible Chat Completions rather than the full OpenAI API.

## Error handling

- Missing or truncated bank: fail before construction and recommend rerunning
  `mtplx pull`.
- Manifest key or geometry mismatch: fail before construction; never infer a
  different expert specification from the model family.
- No promoted profile fits: print required versus available memory and list
  the promoted profiles; do not silently choose an unmeasured configuration.
- Integrity receipt mismatch: invalidate the receipt and reverify before load.
- Explicit MTP requested: fail before generation because the first-release
  streamed profiles are AR-only; do not substitute AR under an MTP request.
- Non-loopback bind without an API key: retain the existing startup rejection.

## Verification

### Unit and construction tests

- Manifest key wins over `config.json.model_type`.
- Only complete streamed artifacts auto-enable streaming.
- Named profile packaging works from an installed wheel with no repository
  checkout.
- Profile selection and rejection use exact byte plans.
- Pull receipts are atomic, revision-bound, digest-bound, and invalidated by a
  changed bank, manifest, or file identity.
- Construction installs no per-record hash path after a valid receipt.
- All installed streamed profiles construct AR directly, and an MTP request is
  rejected before generation.
- Existing non-expert models and generic profiles are unchanged.

### Release smoke

Run from an empty cache and a clean `2.3.0+opensourcewtf.moe` fork wheel:

1. Pull the pinned public Hugging Face revision including `experts.bin`.
2. Confirm the bank size/digest and installed receipt.
3. Start `mtplx serve` with no expert flags on a supported 128 GiB Mac.
4. Confirm `/health` reports the exact artifact revision, expert profile,
   active backend, and request route.
5. Send non-streaming and streaming Chat Completions through the official
   OpenAI client and LiteLLM.
6. Exercise short and long AR requests and verify the route reported at
   request admission.
7. Restart without rehashing the bank and repeat the API smoke.

The full focused suite, wheel build/check, Ruff, and `git diff --check` must
also pass. Existing standard-model serving tests are regression gates.

## Failure-mode review

### Critical: the merge overwrites newer 2.3 behavior

Resolved by treating upstream 2.3 as authoritative for shared public surfaces,
reviewing every conflict explicitly, and rerunning the 501-test clean baseline
plus the complete expert suite. A merge that changes an unrelated standard
model route is rejected.

### Critical: automatic detection installs the wrong expert geometry

Resolved by making the manifest key authoritative, validating it against the
installed specification and record geometry, and failing before construction.
There is no family-name inference when a manifest key exists.

### Critical: automatic profile selection overcommits a busy machine

Resolved by requiring both installed-RAM eligibility and a launch-time
availability preflight. Failure is explicit; there is no automatic downgrade
or stock fallback.

### Critical: integrity work returns to the measured path

Resolved by requiring an admission receipt before lane installation and
constructing the slot reader with per-record verification disabled. Diagnostic
hashing is a separate explicit construction mode and cannot be engaged by the
production profile.

### Minor: first launch from an arbitrary local path can be slow

Accepted. Without a trusted download receipt, computing the 75 GiB bank digest
once is preferable to silently trusting an unverified local file. The supported
`mtplx pull` path verifies while downloading and avoids a second full pass.
