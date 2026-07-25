# SSD-streamed MoE

MTPLX 2.3.1rc1 promotes one SSD-streamed model:
`OpensourceWTF/Hy3-oQ2e-MTPLX-streaming`. Its routed expert weights live in
`experts.bin`; MTPLX keeps the resident tensors and a bounded expert bank in
unified memory and reads the remaining routed experts from SSD.

This first release is autoregressive-only. The promoted profiles install the AR
callable directly, report `generation_mode=ar`, and reject an explicit MTP
request before generation. There is no try-MTP-then-AR fallback. This release
does not promote a GLM streamed profile.

## Install and serve

```bash
python3 -m pip install mtplx==2.3.1rc1
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

`--download` fetches the resident shards, tokenizer/config files,
`expert-manifest.json`, and literal `experts.bin`. Before model construction,
MTPLX validates the complete artifact and writes a revision- and digest-bound
admission receipt outside the immutable model snapshot. A subsequent launch
reuses a matching receipt rather than hashing the 75 GiB bank again.

The release is pinned to revision
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

## Configuration precedence

For a complete admitted Hy3 artifact, the normal serve command automatically
enables streaming. Configuration is resolved once, before the runtime is
constructed:

1. `auto` selects the largest fitting promoted profile. A concrete
   `--expert-profile hy3-oq2e-64`, `hy3-oq2e-88`, or `hy3-oq2e-96` selects that
   profile and still runs preflight.
2. `--expert-streaming-config FILE.json` overlays that profile with one JSON
   object.
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
`--load-mtp` is rejected for a streamed profile in 2.3.1rc1.

## OpenAI and LiteLLM clients

The OpenAI-compatible surface for this release is:

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
