# OpenSourceWTF MTPLX-MOE experts.bin CLI Fork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers-optimized:subagent-driven-development (recommended) or
> superpowers-optimized:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a source-installed OpenSourceWTF fork, based on upstream 2.3.0,
whose normal CLI can download, admit, configure, and serve the public Hy3 oQ2e
`experts.bin` model to OpenAI and LiteLLM clients without expert-specific
flags.

**Architecture:** Preserve the upstream 2.3 CLI, server, generation, and
standard-model implementation, merge the fork's exact expert runtime modules,
and reconnect them only at construction boundaries. A receipt-backed artifact
admission module verifies the immutable expert bank once. Package-owned serve
profiles install measured geometry before runtime construction, and the first
release constructs AR directly.

**Tech Stack:** Python 3.11+, argparse, FastAPI/Uvicorn, Hugging Face Hub,
MLX/MLX-LM, pytest, setuptools package data, OpenAI Python client, LiteLLM.

**Assumptions:**

- Assumes the public model remains
  `OpensourceWTF/Hy3-oQ2e-MTPLX-streaming` at revision
  `d33ce31c0605fc571c374cdf0aa0f085ec50ff88` — the release smoke will NOT be
  accepted against a moving branch or a different bank.
- Assumes the published bank remains 80,518,053,888 bytes with SHA-256
  `c72fb8c0a66020439f4a78591ab9a79d8da3d38412635a531d604ffbf0d2e7d4` —
  admission will fail closed if either value changes.
- Assumes a 128 GiB Apple Silicon Mac and at least 110 GiB of free disk are
  available for the real release smoke — unit and wheel tests do NOT replace
  that gate.
- Assumes `origin/main@e7089885` remains the exact fork integration source and
  `feat/moe-streaming-oob@43348c3` remains the reviewed zero-flag follow-up —
  this plan will NOT silently substitute a later branch tip.
- Assumes the promoted 64 GiB profile evidence at `14c8b57` remains valid —
  unrelated code and large evaluation artifacts from that branch will NOT be
  merged.

---

## File structure

### Merge-owned existing files

- `mtplx/cli.py`: keep the upstream 2.3 parser and add the fork's expert argument
  group to `run`, `chat`, and `serve`.
- `mtplx/runtime.py`: keep upstream runtime behavior and install the exact expert
  runtime only when construction kwargs contain an admitted expert config.
- `mtplx/server/openai.py`: keep the upstream 2.3 API implementation; add expert
  construction arguments, health metadata, and direct AR rejection rules.
- `mtplx/commands/public.py`: keep the clean merge of the 2.3 serve path and the
  fork's expert child-process forwarding, then layer zero-flag resolution and
  named profiles on it.
- `mtplx/hf_loader.py`: keep upstream interrupted-shard safeguards and add expert
  bank completeness, streamed hashing, resolved revision, and admission calls.
- `mtplx/generation.py`: keep upstream generation arithmetic; bind the one
  profile-enabled prefill environment decision at process construction.
- `mtplx/models/hy3_mlx.py`: bind Hy3 submit cadence at module construction,
  outside layer/token execution.
- `mtplx/optimization_profiles.py`: keep the newer upstream 2.3 registry;
  production expert serve profiles live in their own module.

### New focused files

- `mtplx/expert_admission.py`: validate manifest/spec/bank provenance and manage
  atomic external receipts.
- `mtplx/expert_profiles.py`: load package-owned production serve profiles,
  perform memory preflight, and build immutable `ExpertStreamingConfig` values.
- `mtplx/data/expert_profiles.json`: exact promoted 64/88/96 GiB configuration
  and benchmark provenance.
- `tests/test_expert_admission.py`: receipt, mutation, digest, revision, and
  atomicity tests.
- `tests/test_expert_profiles.py`: package loading, exact configuration, and
  memory selection tests.
- `tests/test_expert_profile_packaging.py`: wheel-resource test.
- `scripts/release_smoke_expert_api.py`: OpenAI and LiteLLM client smoke against
  a real running release-candidate server.

## Task 1: Merge the exact fork runtime without replacing upstream 2.3

**Files:**

- Merge source: `origin/main@e7089885ef5442daa53798b32627843beee04c40`
- Modify: `docs/quickstart.md`
- Modify: `mtplx/cli.py`
- Modify: `mtplx/compiled_forward.py`
- Modify: `mtplx/generation.py`
- Modify: `mtplx/hf_loader.py`
- Modify: `mtplx/optimization_profiles.py`
- Modify: `mtplx/runtime.py`
- Modify: `mtplx/server/openai.py`
- Modify: `tests/test_optimization_profiles.py`
- Modify: `tests/test_proj_quant.py`
- Test: `tests/test_expert_cli_runtime.py`
- Test: `tests/test_expert_streaming.py`
- Test: `tests/test_streamed_models.py`
- Test: `tests/test_public_cli.py`
- Test: `tests/test_server_openai.py`

**Security flag:** `none`

**Does NOT cover:** Zero-flag alias resolution, artifact receipts, named serve
profiles, or release documentation.

- [x] **Step 1: Reconfirm the clean upstream baseline**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_public_cli.py \
  tests/test_server_openai.py \
  tests/test_runtime_kpis.py \
  tests/test_hy_v3_mtp_backend.py \
  tests/test_hy_v3_mtp_graft.py
```

Expected: `501 passed, 2 skipped`.

- [x] **Step 2: Start the history-preserving merge**

```bash
git merge --no-ff --no-commit e7089885ef5442daa53798b32627843beee04c40
```

Expected: exactly these ten conflicts:

```text
docs/quickstart.md
mtplx/cli.py
mtplx/compiled_forward.py
mtplx/generation.py
mtplx/hf_loader.py
mtplx/optimization_profiles.py
mtplx/runtime.py
mtplx/server/openai.py
tests/test_optimization_profiles.py
tests/test_proj_quant.py
```

- [x] **Step 3: Restore upstream 2.3 as the base for every conflicted file**

```bash
git checkout --ours \
  docs/quickstart.md \
  mtplx/cli.py \
  mtplx/compiled_forward.py \
  mtplx/generation.py \
  mtplx/hf_loader.py \
  mtplx/optimization_profiles.py \
  mtplx/runtime.py \
  mtplx/server/openai.py \
  tests/test_optimization_profiles.py \
  tests/test_proj_quant.py
git add \
  docs/quickstart.md \
  mtplx/compiled_forward.py \
  mtplx/generation.py \
  mtplx/hf_loader.py \
  mtplx/optimization_profiles.py \
  tests/test_optimization_profiles.py \
  tests/test_proj_quant.py
```

Expected: only `mtplx/cli.py`, `mtplx/runtime.py`, and
`mtplx/server/openai.py` remain intentionally unstaged while their narrow
construction seams are restored.

- [x] **Step 4: Add the expert parser seam to the upstream CLI**

Add this helper beside the other argument-group helpers:

```python
def _add_expert_streaming_args(parser: argparse.ArgumentParser) -> None:
    from .expert_cli import add_expert_streaming_args

    add_expert_streaming_args(parser)
```

Call `_add_expert_streaming_args(...)` exactly once for each of the `run`,
`chat`, and `serve` parsers. Do not replace `build_parser`, public help, or
dispatch logic.

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_expert_cli_runtime.py \
  tests/test_public_cli.py
```

Expected: parser/forwarding tests pass; any remaining failures must name the
runtime or server construction seams, not missing CLI flags.

- [x] **Step 5: Restore the exact resident-only streamed construction branch**

Port from `e7089885:mtplx/runtime.py` the complete streamed construction path at
lines 481-924, not a post-load wrapper. A normal `mlx_lm.load` would materialize
the removed expert tensors and is not an acceptable substitute.

Split the current upstream `load` body into the source branch's `_load_impl`
plus ownership-transferring `load` wrapper. Preserve the upstream arguments
`contract`, `mtp_adapter`, `merge_mtp_adapter`, `gemma4_draft_block_size`,
`gemma4_target_distribution_mode`, `proj_quant`, and `proj_requant`, and add
the source arguments `expert_streaming_config`, `expert_manifest`,
`mtp_artifacts`, and `mtp_precision`. `_load_impl` additionally receives the
required `_expert_runtime_owner: list[Any]`; `load` creates that list, closes
its runtime on construction failure, and otherwise returns the completed
`MTPLXRuntime`.

Keep the source branch's exact sequence:

1. Require config and manifest together.
2. Load and validate the streaming spec and manifest.
3. Compute the fixed-memory plan before model allocation.
4. Open `ExpertStreamingRuntime` and transfer ownership to the temporary owner.
5. Call `construct_resident_model(...)`, which loads only resident tensors.
6. Apply the exact Hy3 model patches, router kernels, resident requantization,
   and component-bank allocator from the source block.
7. Attach the expert runtime to the constructed model.
8. Close the expert runtime on every construction failure.

The upstream non-streaming branch remains byte-for-byte the 2.3 implementation.
Port the fork's `MTPLXRuntime.close`, `memory_plan`,
`expert_streaming_snapshot`, KV admission, and ownership methods. Preserve the
source's outer `expert_runtime_owner` try/except so ownership transfers only
after successful construction.

- [x] **Step 6: Restore the server construction seam**

In the upstream `ServerState.__init__`, retain every upstream cache, vision,
batching, and API initialization, then add:

```python
from mtplx.expert_cli import expert_streaming_load_kwargs

self.expert_streaming_load_kwargs = expert_streaming_load_kwargs(
    args, Path(args.model)
)
if self.expert_streaming_load_kwargs:
    args.generation_mode = "ar"
```

Merge the mapping into the existing runtime load call:

```python
streaming_kwargs = dict(self.expert_streaming_load_kwargs)
self.runtime = load(
    args.model,
    mtp=args.load_mtp and args.generation_mode == "mtp",
    **existing_upstream_load_kwargs,
    **streaming_kwargs,
)
```

Call `add_expert_streaming_args(parser)` in `parse_args`, and expose the
existing fork snapshot only under `/health["expert_streaming"]`. Do not replace
the upstream endpoint implementations.

Port the fork's request-boundary memory ownership seams as well:

1. `_runtime_kv_admission(runtime, tokens)` acquires
   `runtime.admit_kv_tokens(len(prompt_ids) + response_max)` before both batched
   and directly dispatched generation, and releases the lease in each existing
   `finally` block.
2. `_expert_streaming_allows_session_live_refs(runtime)` forces streamed
   requests to `snapshot_only`; an unmetered live session KV reference must not
   outlive the aggregate admission lease.
3. Server shutdown calls `runtime.close()` after request schedulers stop.
4. The existing upstream non-streaming paths keep `nullcontext()` admission and
   their current live-reference policy.

These are ownership boundaries, not eligibility fallbacks. Do not port
request-loop counters or add a per-token proof-of-engagement check.

- [x] **Step 7: Verify merged source and both product paths**

```bash
git diff --check
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_expert_manifest.py \
  tests/test_expert_streaming.py \
  tests/test_expert_streaming_models.py \
  tests/test_expert_slots_runtime.py \
  tests/test_expert_cli_runtime.py \
  tests/test_streamed_models.py \
  tests/test_public_cli.py \
  tests/test_server_openai.py
```

Expected: all selected tests pass and `git diff --check` is silent.

- [x] **Step 8: Commit the semantic merge**

```bash
git add mtplx/cli.py mtplx/runtime.py mtplx/server/openai.py
git commit -m "merge: integrate exact expert runtime with MTPLX 2.3"
```

## Task 2: Apply and verify the reviewed zero-flag flow

**Files:**

- Apply commits: `26d34b2` through `43348c3`
- Modify on conflict: `mtplx/commands/public.py`
- Modify on conflict: `README.md`
- Test: `tests/test_artifacts.py`
- Test: `tests/test_default_models.py`
- Test: `tests/test_expert_cli_runtime.py`
- Test: `tests/test_serve_streaming_autodetect.py`

**Security flag:** `none`

**Does NOT cover:** Cryptographic admission receipts, production profile
selection, or MTP enablement.

- [x] **Step 1: Apply the reviewed commits in original order**

```bash
git cherry-pick \
  26d34b25c84755b4a6bdb0ba2e83e9f652c746f3 \
  d6760819304e71881e3048d75fecee76b3fcdd15 \
  0a3477604a5c545230ec9230b3de25e89102094d \
  8b4b0f9e6cd56bc9d3ff03f772608d07cee6a632 \
  00cc5a14a5416b20f5aa2b6136eed47ed2936116 \
  72a7f0fa6681931f33ecbf0346c8b8b5b6b2af24 \
  348fc281983e21cc898de080e8b2b4144636c5a0 \
  43348c344c9418ee5d3a8736ff5b94b0ef669977
```

If `mtplx/commands/public.py` conflicts, retain the upstream 2.3 serve body and
port only `_maybe_enable_expert_streaming`,
`_maybe_rewrite_streaming_model_ref`, their calls after model resolution, and
`append_expert_streaming_child_args(cmd, args)`.

- [x] **Step 2: Run the source branch's behavior locks**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_artifacts.py \
  tests/test_default_models.py \
  tests/test_expert_cli_runtime.py \
  tests/test_serve_streaming_autodetect.py \
  tests/test_public_cli.py \
  tests/test_server_openai.py
```

Expected: all selected tests pass.

- [x] **Step 3: Confirm the existing ordinary-model behavior lock**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_serve_streaming_autodetect.py::test_plain_model_dir_stays_normal_load
```

Expected: pass.

## Task 3: Add receipt-backed expert artifact admission

**Files:**

- Create: `mtplx/expert_admission.py`
- Create: `tests/test_expert_admission.py`
- Modify: `mtplx/hf_loader.py`
- Modify: `tests/test_hf_loader_expert_artifacts.py`

**Security flag:** `security`

**Does NOT cover:** Runtime geometry selection or client API behavior. A local
artifact without a valid receipt is deliberately rehashed once.

- [x] **Step 1: Write failing receipt and mutation tests**

Create tests around this public interface:

```python
from pathlib import Path

import pytest

from mtplx.expert_manifest import save_expert_manifest
from mtplx.expert_admission import (
    admit_expert_artifact,
    ensure_expert_admitted,
    load_valid_admission_receipt,
)
from tests.test_expert_manifest import _make_authoritative_checkpoint


@pytest.fixture
def authoritative_expert_artifact(tmp_path, monkeypatch):
    root = tmp_path / "model"
    spec, manifest = _make_authoritative_checkpoint(root)
    saved = save_expert_manifest(manifest, root / "expert-manifest.json")
    monkeypatch.setattr(
        "mtplx.expert_admission.get_model_spec",
        lambda model_key: spec if model_key == spec.key else None,
    )
    return root, saved, root / saved.sidecar.file


def test_admission_writes_revision_bound_external_receipt(
    authoritative_expert_artifact, tmp_path
):
    root, manifest, bank = authoritative_expert_artifact
    receipt = admit_expert_artifact(
        root,
        repo_id="owner/model",
        revision="a" * 40,
        receipt_root=tmp_path / "receipts",
    )

    assert receipt["revision"] == "a" * 40
    assert receipt["manifest_sha256"] == manifest.manifest_sha256
    assert receipt["banks"][0]["sha256"] == manifest.sidecar.sha256
    assert Path(receipt["receipt_path"]).parent == tmp_path / "receipts"
    assert not (root / ".mtplx_admission.json").exists()


def test_same_size_bank_mutation_invalidates_receipt(
    authoritative_expert_artifact, tmp_path
):
    root, _manifest, bank = authoritative_expert_artifact
    receipt_root = tmp_path / "receipts"
    admit_expert_artifact(
        root, repo_id="owner/model", revision="b" * 40, receipt_root=receipt_root
    )
    original = bank.read_bytes()
    bank.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))

    assert load_valid_admission_receipt(root, receipt_root=receipt_root) is None
```

Also cover truncated bank, wrong manifest digest, wrong revision, unsafe sidecar
path, missing resident shard, interrupted atomic write, and a valid receipt
that returns without invoking the bank hasher.

- [x] **Step 2: Run the tests and confirm the module is missing**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q tests/test_expert_admission.py
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'mtplx.expert_admission'`.

- [x] **Step 3: Implement the admission API**

Use immutable dataclasses and these signatures:

```python
RECEIPT_SCHEMA = 1
DEFAULT_RECEIPT_ROOT = Path("~/.mtplx/receipts").expanduser()


@dataclass(frozen=True)
class TrustedFileDigest:
    sha256: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int


```

Implement these four public call signatures:

- `admission_receipt_path(artifact_root: Path | str, *, receipt_root: Path | str
  | None = None) -> Path`
- `load_valid_admission_receipt(artifact_root: Path | str, *, revision: str |
  None = None, receipt_root: Path | str | None = None) -> dict[str, Any] | None`
- `admit_expert_artifact(artifact_root: Path | str, *, repo_id: str | None =
  None, revision: str | None = None, receipt_root: Path | str | None = None,
  trusted_bank_digests: Mapping[str, TrustedFileDigest] | None = None) ->
  dict[str, Any]`
- `ensure_expert_admitted(artifact_root: Path | str, *, repo_id: str | None =
  None, revision: str | None = None, receipt_root: Path | str | None = None) ->
  dict[str, Any]`

When `receipt_root` is omitted, resolve `MTPLX_RECEIPT_DIR` once at this
admission boundary and otherwise use `DEFAULT_RECEIPT_ROOT`. This environment
override exists for isolated installs and release smoke; it is never read
during generation.

Implementation order inside `admit_expert_artifact`:

```python
manifest_path = root / "expert-manifest.json"
manifest = load_expert_manifest(manifest_path)
spec = get_model_spec(manifest.model_key)
validate_expert_manifest_spec(manifest, spec)
verify_expert_manifest(
    manifest,
    root,
    verify_records=False,
    verify_shard_hashes=False,
    verify_sidecar_hash=False,
)
```

For every sidecar part, compare exact size and SHA-256, using a downloader-
provided `TrustedFileDigest` only when every stored identity field matches the
final installed file.
Store `st_dev`, `st_ino`, `st_size`, and `st_mtime_ns`. Write JSON to a
same-directory temporary file, `fsync`, `chmod(0o600)`, then `os.replace`.
`ensure_expert_admitted` logs either `expert admission receipt reused; bank
SHA-256 skipped` or `expert admission receipt missing or stale; verifying
bank`, so a release run can prove which boundary path executed.

- [x] **Step 4: Extend Hub metadata and streaming download hashes**

Extend `RepoFile`:

```python
@dataclass(frozen=True)
class RepoFile:
    path: str
    size_bytes: int | None
    sha256: str | None = None
```

Populate `sha256` from `sibling.lfs.sha256`. Make
`_download_repo_file(...)` return the digest accumulated while writing. On a
resumed `.incomplete` file, hash the existing prefix once before appending.
For a fresh expert repository, force the structured downloader even when no
progress callback was supplied so `experts.bin` is not reread after download.

Resolve and return the immutable Hub commit SHA from `model_info.sha`; never
write a receipt bound only to `"main"`.

- [x] **Step 5: Admit at the end of `pull_model`**

After completeness checks and before the `"complete"` progress event:

```python
expert_status = expert_artifact_status(resolved)
admission = None
if expert_status.get("streamed_experts"):
    admission = admit_expert_artifact(
        resolved,
        repo_id=repo_id,
        revision=resolved_revision,
        trusted_bank_digests=downloaded_digests,
    )
```

Return `resolved_revision` and `expert_admission` in the pull result. Reused
caches call `ensure_expert_admitted`; a valid receipt must not hash the bank.

- [x] **Step 6: Run security and downloader tests**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_expert_admission.py \
  tests/test_hf_loader.py \
  tests/test_hf_loader_expert_artifacts.py
```

Expected: all pass.

- [x] **Step 7: Commit**

```bash
git add \
  mtplx/expert_admission.py \
  mtplx/hf_loader.py \
  tests/test_expert_admission.py \
  tests/test_hf_loader_expert_artifacts.py
git commit -m "feat(experts): admit streamed artifacts once with receipts"
```

## Task 4: Package exact promoted serve profiles

**Files:**

- Create: `mtplx/data/expert_profiles.json`
- Create: `mtplx/expert_profiles.py`
- Create: `tests/test_expert_profiles.py`
- Modify: `pyproject.toml`

**Security flag:** `none`

**Does NOT cover:** The candidate 48, 56, 72, or 80 GiB sweep entries, GLM,
q1, rANS, mixed-official, or automatic MTP routing.

- [x] **Step 1: Write failing profile resource and selector tests**

```python
from mtplx.expert_profiles import (
    build_expert_streaming_config,
    load_expert_profiles,
    select_expert_profile,
)


def test_only_promoted_oq2e_profiles_are_installed():
    assert set(load_expert_profiles()) == {
        "hy3-oq2e-64",
        "hy3-oq2e-88",
        "hy3-oq2e-96",
    }


def test_auto_selects_largest_profile_that_passes_both_memory_gates():
    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=128 * 1024**3,
        available_bytes=97 * 1024**3,
    )
    assert selected.name == "hy3-oq2e-88"


def test_64_profile_installs_measured_cache_heavy_geometry():
    profile = load_expert_profiles()["hy3-oq2e-64"]
    config = build_expert_streaming_config(profile)
    assert config.memory_limit_bytes == 71 * 1024**3
    assert config.runtime_reserve_bytes == 7 * 1024**3
    assert config.expert_cache_limit_bytes == 53_678_702_592
    assert config.island_layers == ()
    assert config.island_layer_count is None
    assert config.proj_requant == "q4"
    assert config.verify_record_hashes is False
```

Also test exact 88/96 island counts, model-key mismatch, explicit overcommit,
auto with no fit, and error text containing required and available bytes.

- [x] **Step 2: Run and confirm the resource/module are missing**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q tests/test_expert_profiles.py
```

Expected: collection fails on missing `mtplx.expert_profiles`.

- [x] **Step 3: Add the immutable JSON resource**

The resource must contain these exact public rows:

```json
{
  "schema": 1,
  "profiles": [
    {
      "name": "hy3-oq2e-64",
      "model_key": "hy3-expert-oq2e",
      "process_ceiling_bytes": 76235669504,
      "weight_envelope_bytes": 68719476736,
      "generation_mode": "ar",
      "evidence_commit": "14c8b57fff358bee3da2d10968a855b955b86847",
      "evidence_receipts": [
        "evals/tier2/t3_64x16k_armB_frequency.json",
        "research/envelope-admission-sweep-2026-07-22.json"
      ],
      "config": {
        "memory_limit_bytes": 76235669504,
        "runtime_reserve_bytes": 7516192768,
        "expert_cache_limit_bytes": 53678702592,
        "max_live_kv_tokens": 4096,
        "cache_policy": "frequency",
        "cache_scope": "layer",
        "slot_layout": "component-banks",
        "hy3_router_kernel": "mpp-fp32-splitk-r1-fused-r2",
        "proj_requant": "q4",
        "split_route_release": "deferred",
        "deferred_pin_release": true,
        "verify_record_hashes": false,
        "verify_sidecar_hash_at_open": false
      },
      "child_env": {
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_HY3_SUBMIT_CADENCE": "8"
      }
    },
    {
      "name": "hy3-oq2e-88",
      "model_key": "hy3-expert-oq2e",
      "process_ceiling_bytes": 102005473280,
      "weight_envelope_bytes": 94489280512,
      "generation_mode": "ar",
      "evidence_commit": "191ed9aa362e645f48f1a105a6ec024ea4fd5cf4",
      "evidence_receipts": ["evals/tier2/NOTES.md"],
      "config": {
        "memory_limit_bytes": 102005473280,
        "runtime_reserve_bytes": 7516192768,
        "expert_cache_limit_bytes": 2147483648,
        "max_live_kv_tokens": 4096,
        "cache_policy": "frequency",
        "cache_scope": "layer",
        "slot_layout": "component-banks",
        "island_layer_count": 74,
        "hy3_router_kernel": "mpp-fp32-splitk-r1-fused-r2",
        "split_route_release": "deferred",
        "deferred_pin_release": true,
        "verify_record_hashes": false,
        "verify_sidecar_hash_at_open": false
      },
      "child_env": {
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_HY3_SUBMIT_CADENCE": "8"
      }
    },
    {
      "name": "hy3-oq2e-96",
      "model_key": "hy3-expert-oq2e",
      "process_ceiling_bytes": 110595407872,
      "weight_envelope_bytes": 103079215104,
      "generation_mode": "ar",
      "evidence_commit": "191ed9aa362e645f48f1a105a6ec024ea4fd5cf4",
      "evidence_receipts": ["evals/tier2/NOTES.md"],
      "config": {
        "memory_limit_bytes": 110595407872,
        "runtime_reserve_bytes": 7516192768,
        "expert_cache_limit_bytes": 2147483648,
        "max_live_kv_tokens": 4096,
        "cache_policy": "frequency",
        "cache_scope": "layer",
        "slot_layout": "component-banks",
        "island_layer_count": 79,
        "hy3_router_kernel": "mpp-fp32-splitk-r1-fused-r2",
        "verify_record_hashes": false,
        "verify_sidecar_hash_at_open": false
      },
      "child_env": {
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_HY3_SUBMIT_CADENCE": "8"
      }
    }
  ]
}
```

- [x] **Step 4: Implement the profile loader and preflight**

Define:

```python
@dataclass(frozen=True)
class ExpertServeProfile:
    name: str
    model_key: str
    process_ceiling_bytes: int
    weight_envelope_bytes: int
    generation_mode: str
    evidence_commit: str
    evidence_receipts: tuple[str, ...]
    config: Mapping[str, Any]
    child_env: Mapping[str, str]


```

Implement these public call signatures:

- `load_expert_profiles() -> Mapping[str, ExpertServeProfile]`
- `available_memory_bytes() -> int`
- `select_expert_profile(requested: str, *, model_key: str,
  installed_ram_bytes: int | None = None, available_bytes: int | None = None)
  -> ExpertServeProfile`
- `build_expert_streaming_config(profile: ExpertServeProfile, *, overrides:
  Mapping[str, Any] | None = None) -> ExpertStreamingConfig`

Use `importlib.resources.files("mtplx").joinpath("data/expert_profiles.json")`.
Parse `/usr/bin/vm_stat` outside runtime construction and treat
free + inactive + speculative + purgeable pages as preflight availability.
`auto` chooses the largest profile passing installed and available gates.
Explicit selection fails rather than downgrading.

`build_expert_streaming_config` constructs its keyword mapping as
`{"model_key": profile.model_key, **profile.config,
**normalized_overrides}`. It calls `resolve_island_placement` only when
`island_layer_count` is not `None`; the measured 64 GiB profile must retain
`island_layers == ()` and must not acquire inferred islands.

The 64 GiB row intentionally combines two receipts from the same evidence
commit: `t3_64x16k_armB_frequency.json` proves the zero-island, 49.9921875 GiB
frequency-cache execution geometry, while
`research/envelope-admission-sweep-2026-07-22.json` proves that lowering the
KV ceiling from 16K to the product's 4K control removes exactly 3.75 GiB and
admits under the 71 GiB limit. Do not describe 71 GiB/4K as the literal 16K
benchmark configuration.

- [x] **Step 5: Package and test the resource**

Add:

```toml
[tool.setuptools.package-data]
mtplx = ["templates/**/*.jinja", "data/*.json"]
```

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q tests/test_expert_profiles.py
```

Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add \
  mtplx/data/expert_profiles.json \
  mtplx/expert_profiles.py \
  pyproject.toml \
  tests/test_expert_profiles.py
git commit -m "feat(experts): package promoted Hy3 serve profiles"
```

## Task 5: Bind profile environment decisions outside hot paths

**Files:**

- Modify: `mtplx/generation.py`
- Modify: `mtplx/models/hy3_mlx.py`
- Create: `tests/test_expert_hot_path_invariants.py`

**Security flag:** `none`

**Does NOT cover:** Other diagnostic environment variables that production
expert profiles do not set.

- [x] **Step 1: Write failing one-time binding tests**

```python
def test_sustained_prefill_is_bound_before_generation(monkeypatch):
    policy = generation.bind_generation_feature_policy(
        {"MTPLX_SUSTAINED_PREFILL": "1"}
    )
    monkeypatch.setattr(generation, "_GENERATION_FEATURE_POLICY", policy)
    monkeypatch.setattr(
        generation.os.environ,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot-path environment read")
        ),
    )
    assert generation._sustained_prefill_enabled() is True


def test_hy3_submit_cadence_is_bound_before_forward(monkeypatch):
    policy = hy3_mlx.bind_hy3_execution_policy(
        {"MTPLX_HY3_SUBMIT_CADENCE": "8"}
    )
    monkeypatch.setattr(hy3_mlx, "_HY3_EXECUTION_POLICY", policy)
    monkeypatch.setattr(
        hy3_mlx.os.environ,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot-path environment read")
        ),
    )
    assert hy3_mlx._decode_submit_cadence() == 8
```

- [x] **Step 2: Run and verify the binding APIs are absent**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q tests/test_expert_hot_path_invariants.py
```

Expected: fail because both `bind_*_policy` functions are missing.

- [x] **Step 3: Implement immutable process-construction policies**

In `generation.py`:

```python
@dataclass(frozen=True)
class GenerationFeaturePolicy:
    sustained_prefill: bool


def bind_generation_feature_policy(
    environ: Mapping[str, str],
) -> GenerationFeaturePolicy:
    return GenerationFeaturePolicy(
        sustained_prefill=str(
            environ.get("MTPLX_SUSTAINED_PREFILL", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
    )


_GENERATION_FEATURE_POLICY = bind_generation_feature_policy(os.environ)


def _sustained_prefill_enabled() -> bool:
    return _GENERATION_FEATURE_POLICY.sustained_prefill
```

In `models/hy3_mlx.py`:

```python
@dataclass(frozen=True)
class Hy3ExecutionPolicy:
    submit_cadence: int


def bind_hy3_execution_policy(environ: Mapping[str, str]) -> Hy3ExecutionPolicy:
    raw = str(environ.get(SUBMIT_CADENCE_ENV, "")).strip()
    try:
        cadence = int(raw) if raw else 0
    except ValueError:
        cadence = 0
    return Hy3ExecutionPolicy(submit_cadence=max(0, cadence))


_HY3_EXECUTION_POLICY = bind_hy3_execution_policy(os.environ)


def _decode_submit_cadence() -> int:
    return _HY3_EXECUTION_POLICY.submit_cadence
```

- [x] **Step 4: Run invariant and generation regression tests**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_expert_hot_path_invariants.py \
  tests/test_generation_sustained.py \
  tests/test_hy3_expert_q2.py
```

Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add \
  mtplx/generation.py \
  mtplx/models/hy3_mlx.py \
  tests/test_expert_hot_path_invariants.py
git commit -m "perf(experts): bind serve profile policy before generation"
```

## Task 6: Connect receipts and profiles to the public CLI and server

**Files:**

- Modify: `mtplx/expert_cli.py`
- Modify: `mtplx/commands/public.py`
- Modify: `mtplx/server/openai.py`
- Modify: `tests/test_expert_cli_runtime.py`
- Modify: `tests/test_serve_streaming_autodetect.py`
- Modify: `tests/test_server_openai.py`

**Security flag:** `security`

**Does NOT cover:** `/v1/responses`; streamed serve profiles remain AR-only.

- [x] **Step 1: Write failing public-profile and explicit-MTP tests**

```python
from types import SimpleNamespace

from mtplx.commands import public


def test_expert_profile_is_public_and_defaults_to_auto():
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "OpensourceWTF/Hy3-oQ2e-MTPLX-streaming",
            "--download",
        ]
    )
    assert args.expert_profile == "auto"


def test_explicit_mtp_is_rejected_for_streamed_profile():
    args = SimpleNamespace(
        generation_mode="mtp",
        _cli_flags={"generation-mode"},
    )
    assert public._streamed_generation_mode_error(args) == (
        "promoted streamed profiles are AR-only in the OpenSourceWTF "
        "MTPLX-MOE fork"
    )
```

Add a server health test asserting:

```python
assert health["generation_mode"] == "ar"
assert health["expert_profile"]["name"] == "hy3-oq2e-64"
assert health["expert_profile"]["evidence_commit"] == (
    "14c8b57fff358bee3da2d10968a855b955b86847"
)
assert health["expert_admission"]["bank_sha256"] == "a" * 64
```

- [x] **Step 2: Run and observe missing profile behavior**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_expert_cli_runtime.py \
  tests/test_serve_streaming_autodetect.py \
  tests/test_server_openai.py
```

Expected: fail because `--expert-profile`, receipt construction, and profile
health metadata are not connected.

- [x] **Step 3: Add and forward `--expert-profile`**

In `add_expert_streaming_args`:

```python
group.add_argument(
    "--expert-profile",
    choices=["auto", "hy3-oq2e-64", "hy3-oq2e-88", "hy3-oq2e-96"],
    default="auto",
    help="Promoted SSD expert memory profile (default: auto).",
)
```

Forward it to the server child. Keep advanced expert flags and JSON config
support; explicit individual flags override the selected profile.

- [x] **Step 4: Resolve admission and profile before installing the lane**

Inside `expert_streaming_load_kwargs`:

```python
receipt = ensure_expert_admitted(root)
model_key = _resolve_model_key(root, manifest)
profile = resolve_expert_profile_for_args(
    args,
    root,
    model_key=model_key,
)
config = build_expert_streaming_config(profile, overrides=explicit_overrides)
if getattr(args, "expert_verify_record_hashes", None) is True:
    config = dataclasses.replace(config, verify_record_hashes=True)
else:
    config = dataclasses.replace(
        config,
        verify_record_hashes=False,
        verify_sidecar_hash_at_open=False,
    )
setattr(args, "_resolved_expert_profile", profile)
setattr(args, "_expert_admission_receipt", receipt)
return {
    "mtp": False,
    "expert_streaming_config": config,
    "expert_manifest": manifest,
}
```

An explicit JSON config without a named profile still requires admission. It
may enable record hashes only when the user explicitly passed the diagnostic
flag.

- [x] **Step 5: Set child environment and reject explicit MTP**

Add one reusable policy helper:

```python
def _streamed_generation_mode_error(args: Any) -> str | None:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if (
        "generation-mode" in cli_flags
        and _generation_mode_from_args(args) == GENERATION_MODE_MTP
    ):
        return (
            "promoted streamed profiles are AR-only in the OpenSourceWTF "
            "MTPLX-MOE fork"
        )
    return None
```

In `cmd_serve_public`, after model resolution and streaming detection but before
the child command is built:

```python
if streaming_requested:
    generation_error = _streamed_generation_mode_error(args)
    if generation_error is not None:
        _print_serve_start_line(f"error: {generation_error}")
        return 2
    args.generation_mode = GENERATION_MODE_AR
```

Resolve the profile once for child environment. The helper reads the admitted
manifest when `model_key` was not already supplied:

```python
def resolve_expert_profile_for_args(
    args: Any,
    model_path: Path | str,
    *,
    model_key: str | None = None,
) -> ExpertServeProfile:
    resolved_model_key = model_key
    if resolved_model_key is None:
        manifest = load_expert_manifest(Path(model_path) / "expert-manifest.json")
        resolved_model_key = manifest.model_key
    return select_expert_profile(
        getattr(args, "expert_profile", "auto"),
        model_key=resolved_model_key,
    )


resolved_profile = resolve_expert_profile_for_args(args, runtime_model)
child_env_base.update(resolved_profile.child_env)
```

The server child repeats admission/profile validation before constructing the
runtime; it never falls back to stock or a smaller profile.

- [x] **Step 6: Add health metadata**

Add boundary-safe serializers:

```python
def expert_profile_health_payload(
    profile: ExpertServeProfile | None,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "name": profile.name,
        "model_key": profile.model_key,
        "evidence_commit": profile.evidence_commit,
        "backend": profile.config.get("io_backend", "pread"),
        "generation_mode": profile.generation_mode,
    }


def expert_admission_health_payload(
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if receipt is None:
        return None
    banks = receipt.get("banks") or []
    return {
        "revision": receipt.get("revision"),
        "manifest_sha256": receipt.get("manifest_sha256"),
        "bank_sha256": banks[0].get("sha256") if banks else None,
    }
```

Return:

```python
"expert_profile": (
    expert_profile_health_payload(
        getattr(state.args, "_resolved_expert_profile", None)
    )
),
"expert_admission": (
    expert_admission_health_payload(
        getattr(state.args, "_expert_admission_receipt", None)
    )
),
```

Do not expose filesystem inode/device values or cache paths; health includes
revision, manifest digest, bank digest, profile name, evidence commit, backend,
and generation mode.

- [x] **Step 7: Verify public behavior**

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_expert_cli_runtime.py \
  tests/test_serve_streaming_autodetect.py \
  tests/test_server_openai.py \
  tests/test_public_cli.py
```

Expected: all pass.

- [x] **Step 8: Commit**

```bash
git add \
  mtplx/expert_cli.py \
  mtplx/commands/public.py \
  mtplx/server/openai.py \
  tests/test_expert_cli_runtime.py \
  tests/test_serve_streaming_autodetect.py \
  tests/test_server_openai.py
git commit -m "feat(serve): make admitted Hy3 experts zero-flag"
```

## Task 7: Version, document, and prove wheel packaging

**Files:**

- Modify: `pyproject.toml`
- Modify: `mtplx/version.py`
- Modify: `README.md`
- Modify: `docs/advanced/ssd-streamed-moe.md`
- Modify: `docs/server.md`
- Create: `tests/test_expert_profile_packaging.py`
- Create: `scripts/release_smoke_expert_api.py`

**Security flag:** `none`

**Does NOT cover:** Publishing to PyPI/Homebrew/Hugging Face or claiming
`/v1/responses`.

- [x] **Step 1: Write the wheel-resource test**

```python
import json
import subprocess
import sys
import zipfile


def test_built_wheel_contains_expert_profiles(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        check=True,
    )
    wheel = next(tmp_path.glob("mtplx-2.3.0+opensourcewtf.moe-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        payload = json.loads(
            archive.read("mtplx/data/expert_profiles.json")
        )
    assert [row["name"] for row in payload["profiles"]] == [
        "hy3-oq2e-64",
        "hy3-oq2e-88",
        "hy3-oq2e-96",
    ]
```

- [x] **Step 2: Set the source-fork version**

Set both package sources to:

```toml
version = "2.3.0+opensourcewtf.moe"
```

```python
__version__ = "2.3.0+opensourcewtf.moe"
DISPLAY_VERSION = "2.3.0+opensourcewtf.moe"
```

- [x] **Step 3: Document the exact user flow and configurations**

Document:

```bash
python3 -m pip install "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

Document explicit `--expert-profile hy3-oq2e-{64,88,96}`, that profile numbers
are weight envelopes rather than machine RAM sizes, exact process ceilings,
AR-only status, `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and
the absence of `/v1/responses`.

Include LiteLLM:

```yaml
model_list:
  - model_name: hy3-local
    litellm_params:
      model: openai/OpensourceWTF/Hy3-oQ2e-MTPLX-streaming
      api_base: http://127.0.0.1:8000/v1
      api_key: mtplx-local
```

- [x] **Step 4: Add the API smoke script**

The script accepts `--base-url`, `--model`, and `--api-key`; it must:

```python
openai_client = OpenAI(base_url=args.base_url, api_key=args.api_key)
models = openai_client.models.list()
nonstream = openai_client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": "Reply with exactly OK"}],
    temperature=0,
)
stream = openai_client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": "Reply with exactly OK"}],
    temperature=0,
    stream=True,
)
litellm_response = litellm.completion(
    model=f"openai/{args.model}",
    api_base=args.base_url,
    api_key=args.api_key,
    messages=[{"role": "user", "content": "Reply with exactly OK"}],
    temperature=0,
)
long_response = openai_client.chat.completions.create(
    model=args.model,
    messages=[
        {
            "role": "user",
            "content": "Summarize in one sentence: "
            + ("MTPLX streams experts from SSD. " * 512),
        }
    ],
    max_tokens=32,
    temperature=0,
)
```

Assert nonempty model inventory and nonempty text from all four requests.
After the short and long calls, fetch `/health` by removing the trailing `/v1`
from `args.base_url`; assert `generation_mode == "ar"` and that
`expert_streaming` reports the same active backend and route installed at
request admission.

- [x] **Step 5: Build and inspect the wheel**

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q tests/test_expert_profile_packaging.py
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m build
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m twine check dist/*
```

Expected: packaging test passes, build succeeds, and `twine check` passes.

- [x] **Step 6: Commit**

```bash
git add \
  pyproject.toml \
  mtplx/version.py \
  README.md \
  docs/advanced/ssd-streamed-moe.md \
  docs/server.md \
  docs/specs/2026-07-24-experts-bin-cli-release-design.md \
  docs/plans/2026-07-24-experts-bin-cli-release.md \
  tests/test_expert_profile_packaging.py \
  scripts/release_smoke_expert_api.py
git commit -m "docs: identify OpenSourceWTF expert-serving fork"
```

## Task 8: Run full regression and real empty-cache release smoke

**Files:**

- Verify: entire worktree
- Runtime cache: `/Users/davidtai/projects/OpenSourceWTF/.release-smoke/mtplx-moe-fork`

**Security flag:** `security`

**Does NOT cover:** PyPI/Homebrew publication. A failed real smoke blocks release
readiness even when unit tests pass.

- [ ] **Step 1: Run static and complete automated gates**

```bash
git diff --check
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m ruff check mtplx tests scripts
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -o addopts='' -q tests
```

Expected: no diff errors, Ruff passes, full pytest passes. A no-Metal sandbox
failure must be rerun on the Metal-capable host before classification.

- [ ] **Step 2: Install only the built wheel into a clean environment**

```bash
SMOKE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.release-smoke/mtplx-moe-fork
test ! -e "$SMOKE_ROOT" || {
  echo "release smoke root already exists; archive it before this empty-cache gate"
  exit 1
}
mkdir -p "$SMOKE_ROOT"
export MTPLX_RECEIPT_DIR="$SMOKE_ROOT/receipts"
python3 -m venv "$SMOKE_ROOT/venv"
"$SMOKE_ROOT/venv/bin/python" -m pip install --upgrade pip
"$SMOKE_ROOT/venv/bin/python" -m pip install \
  dist/mtplx-2.3.0+opensourcewtf.moe-*.whl \
  openai \
  litellm
"$SMOKE_ROOT/venv/bin/mtplx" --version
```

Expected: `mtplx 2.3.0+opensourcewtf.moe (2.3.0+opensourcewtf.moe)`.

- [ ] **Step 3: Confirm disk and download the pinned public revision**

```bash
SMOKE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.release-smoke/mtplx-moe-fork
export MTPLX_RECEIPT_DIR="$SMOKE_ROOT/receipts"
df -Pk "$SMOKE_ROOT"
"$SMOKE_ROOT/venv/bin/mtplx" pull \
  OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --revision d33ce31c0605fc571c374cdf0aa0f085ec50ff88 \
  --cache-dir "$SMOKE_ROOT/models" \
  --progress-json | tee "$SMOKE_ROOT/pull.jsonl"
```

Expected: at least 110 GiB free before pull; completion payload contains the
pinned revision, an admitted receipt, bank size `80518053888`, and bank digest
`c72fb8c0a66020439f4a78591ab9a79d8da3d38412635a531d604ffbf0d2e7d4`.

- [ ] **Step 4: Start the zero-flag server from the installed wheel**

```bash
SMOKE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.release-smoke/mtplx-moe-fork
export MTPLX_RECEIPT_DIR="$SMOKE_ROOT/receipts"
"$SMOKE_ROOT/venv/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --cache-dir "$SMOKE_ROOT/models" \
  --host 127.0.0.1 \
  --port 18081 \
  --api-key mtplx-local \
  >"$SMOKE_ROOT/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$SMOKE_ROOT/server.pid"
for attempt in {1..180}; do
  curl -fsS -H 'Authorization: Bearer mtplx-local' \
    http://127.0.0.1:18081/health >"$SMOKE_ROOT/health.json" &&
    break
  sleep 2
done
curl -fsS -H 'Authorization: Bearer mtplx-local' \
  http://127.0.0.1:18081/health
```

Expected after model load: `/health` reports `generation_mode=ar`, exact
revision/digests, selected profile, active expert I/O backend, and no MTP
availability claim for the streamed profile.

- [ ] **Step 5: Run OpenAI and LiteLLM client smoke**

```bash
SMOKE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.release-smoke/mtplx-moe-fork
MODEL_ID=$(
  curl -fsS -H 'Authorization: Bearer mtplx-local' \
    http://127.0.0.1:18081/v1/models |
  "$SMOKE_ROOT/venv/bin/python" -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
)
"$SMOKE_ROOT/venv/bin/python" scripts/release_smoke_expert_api.py \
  --base-url http://127.0.0.1:18081/v1 \
  --api-key mtplx-local \
  --model "$MODEL_ID"
```

Expected: script exits zero after OpenAI short non-streaming, OpenAI streaming,
LiteLLM, and long-context AR responses, and confirms the health route remains
AR.

- [ ] **Step 6: Restart and prove the bank is not rehashed**

Record the receipt modification time, stop the server by its recorded PID,
restart the identical installed-wheel command, and compare:

```bash
SMOKE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.release-smoke/mtplx-moe-fork
export MTPLX_RECEIPT_DIR="$SMOKE_ROOT/receipts"
find "$SMOKE_ROOT/receipts" -type f -name '*.json' \
  -exec stat -f '%m %N' {} \; | sort \
  >"$SMOKE_ROOT/receipt-before.txt"
kill "$(tr -d '\n' <"$SMOKE_ROOT/server.pid")"
wait "$(tr -d '\n' <"$SMOKE_ROOT/server.pid")" || true
"$SMOKE_ROOT/venv/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --cache-dir "$SMOKE_ROOT/models" \
  --host 127.0.0.1 \
  --port 18081 \
  --api-key mtplx-local \
  >"$SMOKE_ROOT/server-restart.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$SMOKE_ROOT/server.pid"
for attempt in {1..180}; do
  curl -fsS -H 'Authorization: Bearer mtplx-local' \
    http://127.0.0.1:18081/health >"$SMOKE_ROOT/health-restart.json" &&
    break
  sleep 2
done
MODEL_ID=$(
  curl -fsS -H 'Authorization: Bearer mtplx-local' \
    http://127.0.0.1:18081/v1/models |
  "$SMOKE_ROOT/venv/bin/python" -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
)
```

After restart:

```bash
find "$SMOKE_ROOT/receipts" -type f -name '*.json' \
  -exec stat -f '%m %N' {} \; | sort \
  >"$SMOKE_ROOT/receipt-after.txt"
diff -u "$SMOKE_ROOT/receipt-before.txt" "$SMOKE_ROOT/receipt-after.txt"
rg -q 'expert admission receipt reused; bank SHA-256 skipped' \
  "$SMOKE_ROOT/server-restart.log"
"$SMOKE_ROOT/venv/bin/python" scripts/release_smoke_expert_api.py \
  --base-url http://127.0.0.1:18081/v1 \
  --api-key mtplx-local \
  --model "$MODEL_ID"
kill "$(tr -d '\n' <"$SMOKE_ROOT/server.pid")"
wait "$(tr -d '\n' <"$SMOKE_ROOT/server.pid")" || true
```

Expected: no receipt rewrite, server log reports receipt reuse, and the client
smoke passes again.

- [ ] **Step 7: Verify final branch state**

```bash
git status --short
git log --oneline --decorate -12
git diff --check HEAD^
```

Expected: the worktree is clean; the approved design, implementation plan,
implementation, tests, docs, and version changes are committed.
