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

This fork is not published to PyPI. Install it directly from GitHub:

```bash
python3 -m pip install "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

`--download` fetches the resident shards, tokenizer/config files,
`expert-manifest.json`, and the expert bank named by that manifest: literal
`experts.bin` for Hy3 or `experts-q1-t158.bin` for GLM. Before model
construction, MTPLX validates the complete artifact and writes a revision- and
digest-bound admission receipt outside the immutable model snapshot. A
subsequent launch reuses a matching receipt rather than hashing the bank again.

The promoted Hy3 release is pinned to revision
`d33ce31c0605fc571c374cdf0aa0f085ec50ff88`. Its `experts.bin` is
80,518,053,888 bytes with SHA-256
`c72fb8c0a66020439f4a78591ab9a79d8da3d38412635a531d604ffbf0d2e7d4`.
A mismatch fails before construction.

## Promoted profiles

`--expert-profile auto` is the default. It chooses the largest promoted profile
whose process ceiling fits both installed RAM and launch-time available memory.
An explicit profile runs the same preflight and fails once with required and
available bytes if it does not fit; it never silently downgrades or routes to a
stock loader.

| Profile | Weight envelope | Process ceiling | Promoted geometry | Route |
|---|---:|---:|---|---|
| `hy3-oq2e-64` | 64 GiB | 71 GiB | zero islands; 49.9921875 GiB frequency cache | AR |
| `hy3-oq2e-88` | 88 GiB | 95 GiB | 74 pinned island layers; five streamed layers | AR |
| `hy3-oq2e-96` | 96 GiB | 103 GiB | all 79 routed layers pinned | AR |

The number in a profile name is its weight envelope, not a machine-RAM
requirement. The process ceiling adds the exact 7 GiB runtime reserve:
76,235,669,504 bytes, 102,005,473,280 bytes, and 110,595,407,872 bytes,
respectively. The real release smoke uses a 128 GiB Apple Silicon Mac and
requires at least 110 GiB of free disk before download.

Choose an envelope explicitly when reproducibility matters:

```bash
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --expert-profile hy3-oq2e-64
```

All three profiles reserve at most 4,096 aggregate live KV tokens. The 64 GiB
row is the promoted zero-island, cache-heavy execution geometry at its 4K
product KV ceiling; its separate 16K performance A/B used a 74.75 GiB process
ceiling and is not the installed product configuration.

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
mtplx serve \
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
slots per each of the 75 streamed layers.

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
  "prefetch_slots": 0,
  "streamed_codec": "none",
  "trace_routes": false
}
```

```bash
mtplx serve \
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
mtplx serve \
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

Use the exact served model ID returned by `/v1/models`. For the primary command
it is `OpensourceWTF/Hy3-oQ2e-MTPLX-streaming`. The loaded model determines the
response model ID; a request's `model` field does not select a different loaded
model.

OpenAI Python:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="mtplx-local",
)
response = client.chat.completions.create(
    model="OpensourceWTF/Hy3-oQ2e-MTPLX-streaming",
    messages=[{"role": "user", "content": "Reply with exactly OK"}],
)
```

LiteLLM proxy configuration:

```yaml
model_list:
  - model_name: hy3-local
    litellm_params:
      model: openai/OpensourceWTF/Hy3-oQ2e-MTPLX-streaming
      api_base: http://127.0.0.1:8000/v1
      api_key: mtplx-local
```

Loopback serving does not require server-side authentication, but the OpenAI
client requires a nonempty key value. If MTPLX was started with
`--api-key mtplx-local`, the client value must match. Binding to a non-loopback
host always requires an API key.

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

For the lower-level artifact layout and parity gates, see the
[SSD-streamed MoE plan](../MOE_SSD_STREAMING_PLAN.md).
