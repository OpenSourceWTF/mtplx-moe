# SSD-streamed MoE

MTPLX recognizes two prepacked SSD-streamed models on Hugging Face:

| Model | Routed-expert bank | Model key | OpenSourceWTF fork status |
|---|---|---|---|
| `OpensourceWTF/Hy3-oQ2e-MTPLX-streaming` | `experts.bin` | `hy3-expert-oq2e` | Promoted profiles |
| `OpensourceWTF/GLM-5.2-t158-MTPLX-streaming` | `experts-q1-t158.bin` | `glm52-expert-q1t` | Manual, experimental configuration |

MTPLX keeps resident tensors and a bounded expert bank in unified memory and
reads the remaining routed experts from the serialized Hugging Face bank. Users
do not need the original expert safetensors or a local repack step.

This fork's streamed path is autoregressive-only. The serve path installs the AR
callable directly, reports `generation_mode=ar`, and rejects an explicit MTP
request before generation. There is no try-MTP-then-AR fallback. Only the Hy3
profiles are promoted; the GLM artifact has no promoted fork profile or
quality receipt.

## Install and serve

This fork is not published to PyPI. It uses the same `mtplx` Python
distribution, import package, and CLI name as upstream MTPLX, so install it in
its own virtual environment:

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
python3 -m venv "$MTPLX_MOE_VENV"
"$MTPLX_MOE_VENV/bin/python" -m pip install --upgrade pip
"$MTPLX_MOE_VENV/bin/python" -m pip install \
  "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
"$MTPLX_MOE_VENV/bin/python" -m pip check
"$MTPLX_MOE_VENV/bin/mtplx" --version

"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

The version must contain `+opensourcewtf.moe`. If upstream MTPLX is also
installed, use the venv's absolute command as shown; see
[INSTALL.md](../../INSTALL.md) for package, PATH, port, and unified-memory
collision handling.

`--download` fetches the resident shards, tokenizer/config files,
`expert-manifest.json`, and the expert bank named by that manifest: literal
`experts.bin` for Hy3 or `experts-q1-t158.bin` for GLM. Before model
construction, the streamed path validates its manifest and expert bank and
writes a revision- and digest-bound admission receipt outside the immutable
model snapshot. A subsequent launch reuses a matching receipt rather than
hashing the bank again.

To download before occupying the server port, run:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" pull \
  OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --progress-json
```

The pull is resumable. The tested Hy3 snapshot contains 97,608,670,094 logical
bytes, including the 80,518,053,888-byte `experts.bin`.

The promoted Hy3 release is pinned to revision
`d33ce31c0605fc571c374cdf0aa0f085ec50ff88`. Its `experts.bin` is
80,518,053,888 bytes with SHA-256
`c72fb8c0a66020439f4a78591ab9a79d8da3d38412635a531d604ffbf0d2e7d4`.
A mismatch fails before construction.

The generic `pull --json` validator also checks upstream MTP release metadata.
For this AR-only streamed artifact it can report `validation.ok: false` with
missing `mtplx_runtime.json` or MTP-sidecar entries even when
`expert_admission` succeeds. That generic result does not invalidate the
streamed expert bank. For this path, require all of the following:

- `pull` exits successfully;
- `expert_admission` records the pinned revision and bank/manifest digests;
- `serve` prints `Runtime contract verified` and reaches `MTPLX is ready`; and
- `/health` reports the same admission evidence and AR-only route.

The related “family-compatible model without a recorded
`mtplx_runtime.json` exactness baseline” warning means generic MTP statistics
are unverified. It is not a missing-`experts.bin` warning.

## Promoted profiles

`--expert-profile auto` is the default. It chooses the largest promoted profile
whose process ceiling fits both installed RAM and launch-time available memory.
An explicit profile runs the same preflight and fails once with required and
available bytes if it does not fit; it never silently downgrades or routes to a
stock loader.

| Profile | Weight envelope | Process ceiling | Promoted geometry | Trunk | Route |
|---|---:|---:|---|---|---|
| `hy3-oq2e-64` | 64 GiB | 71 GiB | zero islands; 49.9921875 GiB frequency cache | q4 requant | AR |
| `hy3-oq2e-88` | 88 GiB | 95 GiB | 74 pinned island layers; five streamed layers | q8 | AR |
| `hy3-oq2e-96` | 96 GiB | 103 GiB | all 79 routed layers pinned | q8 | AR |

The number in a profile name is its weight envelope, not a machine-RAM
requirement. The process ceiling adds the exact 7 GiB runtime reserve:
76,235,669,504 bytes, 102,005,473,280 bytes, and 110,595,407,872 bytes,
respectively. The real release smoke uses a 128 GiB Apple Silicon Mac and
requires at least 110 GiB of free disk before download.

The process ceiling is a budget, not a requirement: each profile's own cache cap
leaves part of it unallocated. Admission compares the footprint the plan
actually reaches -- 66.992 GiB, 89.216 GiB, and 91.992 GiB respectively -- and
applies any cache-sizing override first, so a machine smaller than the ceiling
can still run the profile. See the cache-tier section for the arithmetic.

Choose an envelope explicitly when reproducibility matters:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --expert-profile hy3-oq2e-64
```

All three profiles reserve at most 4,096 aggregate live KV tokens. The 64 GiB
row is the promoted zero-island, cache-heavy execution geometry at its 4K
product KV ceiling; its separate 16K performance A/B used a 74.75 GiB process
ceiling and is not the installed product configuration.

## Hy3 q4 resident-trunk requantization

`proj_requant` is separate from the expert-bank quantization. The Hugging Face
artifact keeps its published oQ2e `experts.bin`; at model construction,
`"proj_requant": "q4"` converts only supported resident q8 trunk `*_proj`
matrices to q4/gs64. Expert banks, router gates, embeddings, the LM head, norms,
and biases retain their loaded precision. The `hy3-oq2e-64` profile already
enables this setting.

To use q4 resident projections with the full-resident 96 GiB weight envelope,
save this as `hy3-rq4.json`:

```json
{
  "proj_requant": "q4"
}
```

Then overlay it on the 79-island profile:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --expert-profile hy3-oq2e-96 \
  --expert-streaming-config hy3-rq4.json
```

This keeps 79 pinned islands, BF16 KV, and the 4,096-token live-KV ceiling from
the selected profile while changing the trunk to q4. Because the effective
configuration differs from the promoted `hy3-oq2e-96` row, `/health` reports
`"customized": true` and does not attach that row's q8 evidence commit.

The flagship q4 receipt is a three-run M5 Max mean at a 1,024-token real-code
context and 256 generated tokens: **48.04 tok/s** at MTP depth 1, versus a
41.36 tok/s AR control, with all 79 islands pinned, BF16 KV, and
`proj_requant=q4`. The paired full-suite HumanEvalPlus result was 142/164
(86.6%) for q4 versus 143/164 (87.2%) for q8; McNemar's exact two-sided test
was p=1.0. The
[compact receipt](../../evals/tier2/hy3_oq2e_rq4_flagship_summary.json)
retains the exact per-run values, settings, source commit, and raw-receipt
hashes. The current streamed `mtplx serve` route is AR-only, so the command
above applies the same requantization to serving but does not enable the MTP
depth-1 benchmark lane.

## Islands versus paged streaming

An island keeps every routed expert for one transformer layer resident. A
streamed layer keeps only a fixed number of expert records in slots and reads a
missing record from the prepacked Hugging Face bank. The configured memory plan
is fixed before model construction:

| Configuration | Dense islands | Streamed layers | Fixed expert slots |
|---|---:|---:|---:|
| Hy3 `hy3-oq2e-64` | 0 | 79 | 49.9921875 GiB total cache |
| Hy3 `hy3-oq2e-88` | 74 | 5 | 2 GiB total cache |
| Hy3 `hy3-oq2e-96` | 79 | 0 | none required |
| GLM t158 measured 96 GiB config | 0 | 75 | 72 GiB cap; 116 slots/layer |

Here “streamed” or “paged” means checked record reads from `experts.bin` or
`experts-q1-t158.bin` into bounded component-bank slots. It does **not** mean
`mmap_island_layers`. That separate zero-copy mode requires a component-major
`experts-banked-manifest.json`, which neither published Hugging Face repository
ships. Do not set `mmap_island_layers` or `banked_manifest` for these downloads.

Hy3's affine expert records support dense islands. The named profiles are the
recommended way to select them. An advanced JSON overlay may set
`island_layer_count`; MTPLX resolves the count through the admitted model's
measured placement and validates the resulting memory plan before installing
the route. For example, this makes four measured Hy3 layers resident while
retaining the 64 GiB profile's other settings:

```json
{
  "island_layer_count": 4
}
```

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --expert-profile hy3-oq2e-64 \
  --expert-streaming-config hy3-four-islands.json
```

Custom geometry clears the named profile's promoted benchmark evidence and
appears as `"customized": true` in `/health`. `island_layer_count` and an
explicit `island_layers` list are mutually exclusive. Island layers must use
`cache_scope: "layer"`, `slot_layout: "component-banks"`, and
`trace_routes: false`; invalid combinations fail during construction.

The published GLM-5.2 bank uses the t158 codec rather than affine expert
records. The affine dense-island and mmap-island kernels cannot interpret that
layout, so `island_layers`, `island_layer_count`, and `mmap_island_layers` are
unsupported for this artifact and fail before loading. Configure its bounded
streaming slots instead.

This GLM configuration reproduces the measured 96 GiB memory plan: a 12 GiB
runtime reserve, 72 GiB expert-cache cap, 48 transient slots, and 116 persistent
slots per each of the 75 streamed layers. It also retains the measured
component-bank, frequency-cache, `F_NOCACHE`, deferred-release, no-prefetch,
and headers-only integrity settings.

```json
{
  "model_key": "glm52-expert-q1t",
  "memory_limit_bytes": "96GiB",
  "max_live_kv_tokens": 4096,
  "runtime_reserve_bytes": "12GiB",
  "expert_cache_limit_bytes": "72GiB",
  "transient_slots": 48,
  "cache_policy": "frequency",
  "cache_scope": "layer",
  "slot_layout": "component-banks",
  "max_read_chunk_bytes": "8MiB",
  "bypass_page_cache": true,
  "q2_expert_kernel": "stock",
  "split_route_release": "deferred",
  "deferred_pin_release": true,
  "prefetch_slots": 0,
  "streamed_codec": "none",
  "verify_record_hashes": false,
  "verify_artifact_headers": true,
  "verify_sidecar_hash_at_open": false,
  "trace_routes": false
}
```

```bash
MTPLX_SUSTAINED_PREFILL=1 "$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/GLM-5.2-t158-MTPLX-streaming \
  --download \
  --expert-streaming-config glm52-t158-96g.json
```

The 96 GiB value is a process ceiling, not the model's download size. The GLM
repository is about 187 GiB on disk. Its t158 weights are lossy and have
construction-time numeric evidence but no task-quality validation, so this is
an experimental serving configuration rather than a quality recommendation.
Changing the memory limit, reserve, cache cap, context budget, or workload
creates a new unmeasured configuration; startup will still validate that its
fixed footprint fits.

## Cache tiers

`--expert-cache-limit` sets a construction-time ceiling on the persistent
expert cache. It does not cap the whole MTPLX process: resident weights, KV
cache, transient service slots, and the runtime reserve are budgeted separately
by the base plan you select. A tier label such as “32 GiB” names that cache
size, not a machine's total RAM, and the same optimized execution stack applies
at that size or any smaller one.

### How a cache size becomes a memory footprint

The footprint follows one rule:

```
realized total  =  fixed floor  +  cache cap rounded down to whole slots
```

The fixed floor is everything the cache does not cover:

| Base plan | Resident | KV @ 4,096 tok | Transient | Reserve | Fixed floor |
|---|---:|---:|---:|---:|---:|
| `hy3-oq2e-64` | 8.710 GiB | 1.250 GiB | 0.040 GiB | 7 GiB | 17.000 GiB |
| GLM t158 96 GiB plan | 9.904 GiB | 0.363 GiB | 0.396 GiB | 12 GiB | 22.663 GiB |

A layer-scoped cache allocates whole expert slots per streamed layer, so the
realized cache rounds down from the requested cap by up to one slot: about
0.39 GiB for Hy3, 0.62 GiB for GLM.

Admission compares that realized total against two machine measurements, and
resolves `--expert-cache-limit` before comparing, so a smaller cache lowers the
requirement:

- **Installed RAM**, from `hw.memsize`.
- **Available memory**, counting free, inactive, speculative, and purgeable
  pages. This is usually the binding gate, and it sits well below installed RAM
  on a machine that has been running for a while.

Both checks run before anything is downloaded or constructed. A base plan's
declared process ceiling (71 GiB for `hy3-oq2e-64`, 96 GiB for the GLM plan) is
a budget rather than a requirement: whatever the cache cap leaves unspent stays
unallocated, and admission does not ask for it.

### Choosing a size for your machine

1. Read this machine's admission inputs:

   ```bash
   "$MTPLX_MOE_VENV/bin/python" -c "from mtplx.expert_profiles import \
     available_memory_bytes, _installed_ram_bytes; g=1024**3; \
     print(f'installed {_installed_ram_bytes()/g:.1f} GiB, \
     available {available_memory_bytes()/g:.1f} GiB')"
   ```

2. Subtract the fixed floor from the smaller of the two, keep a margin for
   everything else you run, and use the remainder as the cap. With 40 GiB
   available on Hy3: `40 - 17.0 = 23`, so pass `--expert-cache-limit 22GiB`.

3. Start the server. If admission refuses, the error reports the exact bytes
   required and detected; lower the cap by the shortfall and retry.

A larger cache holds more experts resident and streams less from SSD. A smaller
one always runs, down to a single slot per layer. Any size other than a base
plan's own default is an unmeasured geometry: startup validates it, but it
carries no throughput or quality receipt.

### Hy3 oQ2e

Use `hy3-oq2e-64` as the base. The named profile carries the q4 resident-trunk
requantization, measured FP32 split-K router kernel, frequency policy,
component banks, deferred split and pin release, trusted admission-integrity
settings, sustained prefill, and the measured submit cadence. Do not
reconstruct those settings manually in a cache-only override.

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --expert-profile hy3-oq2e-64 \
  --expert-cache-limit 40GiB
```

Replace `40GiB` with any size up to the profile's own 49.9921875 GiB allowance.
Every size keeps the 7 GiB runtime reserve and the 4,096-token aggregate
live-KV ceiling:

| Cache cap | Slots/layer | Realized cache | Unallocated | Realized total |
|---:|---:|---:|---:|---:|
| 49.9921875 GiB (default) | 128 | 49.992 GiB | 4.008 GiB | 66.992 GiB |
| 47 GiB | 120 | 46.868 GiB | 7.132 GiB | 63.868 GiB |
| 40 GiB | 102 | 39.838 GiB | 14.162 GiB | 56.838 GiB |
| 32 GiB | 81 | 31.636 GiB | 22.364 GiB | 48.636 GiB |

These totals are upper bounds: `proj_requant=q4` removes a further
manifest-dependent amount from the resident side. A cache larger than the
49.9921875 GiB allowance needs a new total-memory plan and benchmark, not a
raised sub-cap.

### GLM-5.2 t158

Save the complete optimized GLM configuration above as
`glm52-t158-96g.json`. Keep its 96 GiB process ceiling, 12 GiB reserve, 48
transient slots, frequency policy, component banks, `F_NOCACHE`, deferred
release, no-prefetch control, and headers-only integrity settings. The explicit
cache flag below overrides only `expert_cache_limit_bytes` in that JSON.

```bash
MTPLX_SUSTAINED_PREFILL=1 "$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/GLM-5.2-t158-MTPLX-streaming \
  --download \
  --expert-streaming-config glm52-t158-96g.json \
  --expert-cache-limit 32GiB
```

Replace `32GiB` with any size up to the plan's 72 GiB allowance:

| Cache cap | Slots/layer | Realized cache | Realized total |
|---:|---:|---:|---:|
| 48 GiB | 77 | 47.585 GiB | 70.248 GiB |
| 32 GiB | 51 | 31.517 GiB | 54.180 GiB |

### What has been verified

Smoke-tested on a 128 GiB M5 Max with a clean
`mtplx 2.3.0+opensourcewtf.moe` install: Hy3 at 32 GiB, GLM at 32 GiB, and GLM
at 48 GiB. Each constructed its real model, reached `MTPLX is ready`, reported
the slot count above through `/health`, listed its model through `/v1/models`,
and returned a nonempty AR chat completion. That shows those configurations
execute. It is not a throughput receipt, and it is not a GLM t158 task-quality
validation.

The Hy3 40 GiB and 47 GiB rows come from the same memory plan the runtime
builds at construction and are covered by `tests/test_expert_cache_tiers.py`,
but neither has been constructed on hardware.

One sizing caveat worth stating directly: a 47 GiB Hy3 cache clears a 64 GiB
machine's installed-RAM check by only 132 MiB, and will still be refused by the
available-memory gate there. On a 64 GiB machine, 32 GiB is the size that
admits with margin.

### Verify the running server

Wait for `MTPLX is ready`, then verify the route rather than relying on the
early startup card:

```bash
curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

`/health` must report `generation_mode: "ar"`,
`available_generation_modes: ["ar"]`, the expected expert model key, and a
memory plan whose persistent allocation is no larger than the requested cache
sub-cap. A cache-only override customizes the promoted Hy3 profile, so
`expert_profile.customized` must be `true` and `evidence_commit` must be
`null`.

Copy the ID returned by `/v1/models` into this request:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID_FROM_V1_MODELS","messages":[{"role":"user","content":"Reply with exactly OK"}],"temperature":0,"max_tokens":8}' \
  | python3 -m json.tool
```

Stop the foreground server with `Ctrl-C`. To test both tiers, restart it with
the other cache value; do not run two large unified-memory servers at once.

## Configuration precedence

For either complete admitted artifact, the normal serve command automatically
enables streaming. Configuration is resolved once, before the runtime is
constructed:

1. For Hy3, `auto` selects the largest fitting promoted profile. A concrete
   `--expert-profile hy3-oq2e-64`, `hy3-oq2e-88`, or `hy3-oq2e-96` selects that
   profile and still runs preflight. GLM has no promoted profile, so `auto`
   leaves its configuration to the JSON/default construction path.
2. `--expert-streaming-config FILE.json` overlays that profile with one JSON
   object. For GLM it supplies the manual configuration directly.
3. Individual expert flags, such as `--expert-cache-policy` or
   `--expert-max-live-kv-tokens`, override the corresponding JSON values.
4. The completed immutable configuration is validated before the lane is
   installed. Invalid geometry, an unavailable memory ceiling, or a manifest
   model-key mismatch stops startup; there is no enabled-path fallback.

For example, this keeps the selected profile identity but changes its effective
configuration:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --expert-profile hy3-oq2e-64 \
  --expert-streaming-config expert-overrides.json \
  --expert-cache-policy frequency
```

When any effective profile field differs from the promoted row, `/health`
reports `"customized": true`, clears `evidence_commit` to `null`, and includes a
whitelisted `effective` configuration. It does not claim the promoted
benchmark evidence for custom geometry. An unchanged named profile reports its
immutable evidence commit.

Diagnostic integrity flags are explicit construction controls. A valid
admission receipt disables per-record and open-time bank hashing in the
production profile; only an explicitly enabled diagnostic flag turns that work
back on.

An explicit `--no-expert-streaming` prevents automatic activation. Combining
that opt-out with a positive expert selector, including `--expert-streaming`, a
named profile, an expert config, a manifest, or an individual expert flag, is a
configuration error. Likewise, `--generation-mode mtp`, `--mtp`, or
`--load-mtp` is rejected for a streamed profile in this fork.

## OpenAI and LiteLLM clients

The OpenAI-compatible surface for this fork is:

- `GET /v1/models`
- `POST /v1/chat/completions`, including streaming
- `POST /v1/completions`

`/v1/responses` is not implemented. “OpenAI-compatible” refers to the
documented endpoints above, not the full OpenAI API. The separate
Anthropic-compatible endpoint remains `POST /v1/messages`.

The Hugging Face repository ID selects the download. It is not necessarily the
API model ID. Discover the loaded server's ID after startup:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

For the tested Hy3 command, `/v1/models` returns
`hy3-oq2e-mtplx-streaming`. Use that value in clients. The loaded model
determines the response model ID; a request's `model` field does not load or
select a different model.

Keep client and proxy dependencies outside the fork's server venv. In
particular, LiteLLM 1.93.0 requires `rich<14`, while this fork requires
`rich>=14`.

```bash
MTPLX_CLIENT_VENV="$HOME/.venvs/mtplx-clients"
python3 -m venv "$MTPLX_CLIENT_VENV"
"$MTPLX_CLIENT_VENV/bin/python" -m pip install --upgrade pip
"$MTPLX_CLIENT_VENV/bin/python" -m pip install \
  "openai>=1" "litellm[proxy]"
"$MTPLX_CLIENT_VENV/bin/python" -m pip check
```

This split was verified with OpenAI 2.48.0 and LiteLLM 1.93.0. Do not install
`mtplx` into the client venv or `litellm[proxy]` into the server venv.

OpenAI Python:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="mtplx-local",
)
response = client.chat.completions.create(
    model="hy3-oq2e-mtplx-streaming",
    messages=[{"role": "user", "content": "Reply with exactly OK"}],
)
print(response.choices[0].message.content)
```

Run that script with `$MTPLX_CLIENT_VENV/bin/python`.

LiteLLM proxy configuration (`litellm.yaml`):

```yaml
model_list:
  - model_name: hy3-local
    litellm_params:
      model: openai/hy3-oq2e-mtplx-streaming
      api_base: http://127.0.0.1:8000/v1
      api_key: mtplx-local
```

Start the proxy in the client environment:

```bash
"$MTPLX_CLIENT_VENV/bin/litellm" \
  --config litellm.yaml \
  --port 4000
```

OpenAI-compatible clients of the proxy use `http://127.0.0.1:4000/v1` and
model `hy3-local`.
LiteLLM forwards that alias to the one model already loaded by MTPLX; it does
not load another copy.

Loopback serving does not require server-side authentication, but the OpenAI
client requires a nonempty key value. If MTPLX was started with
`--api-key mtplx-local`, the client value must match. Binding to a non-loopback
host always requires an API key.

The fork and upstream MTPLX both default to port 8000, while LiteLLM commonly
uses 4000. Check for listeners before starting either process:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:4000 -sTCP:LISTEN
```

If another MTPLX server owns 8000, use a different fork port such as 18080 and
change LiteLLM's `api_base` and the direct OpenAI `base_url` to match. Separate
venvs do not isolate TCP ports or unified memory.

## Health evidence

After construction, `/health` reports:

- `generation_mode: "ar"` and `available_generation_modes: ["ar"]`;
- `expert_admission.revision`, `manifest_sha256`, and `bank_sha256`;
- the selected `expert_profile.name`, model key, actual runtime I/O backend,
  route, and promoted evidence state; and
- the active `expert_streaming` model key, manifest digest, memory plan, and
  route counters.

The profile backend is read from the constructed expert runtime, not copied from
the profile file. Filesystem paths, inode/device values, and receipt paths are
not exposed. The release smoke checks that admission, profile, customization,
evidence, actual backend, and AR route remain consistent after short,
streaming, LiteLLM, and long requests.

The early generic startup card can currently label the selected sustained
profile as “Sustained MTP” before the streamed route is constructed. That text
is not route evidence for these artifacts. After `MTPLX is ready`, the
authoritative checks are `/health` and `expert_profile.generation_mode`; both
must report `ar`, and `available_generation_modes` must contain only `ar`.

For the lower-level artifact layout and parity gates, see the
[SSD-streamed MoE plan](../MOE_SSD_STREAMING_PLAN.md).
