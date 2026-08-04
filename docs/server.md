# Server

The server provides OpenAI-compatible local serving and Anthropic Messages compatibility for coding
clients and harnesses.

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --host 127.0.0.1 \
  --port 8000 \
  --no-stats-footer
```

Both upstream MTPLX and this fork default to port 8000. If they are installed
side by side, first verify which command and process you are using:

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
type -a mtplx
"$MTPLX_MOE_VENV/bin/mtplx" --version
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

The OpenSourceWTF fork version contains `+opensourcewtf.moe`. If another server
owns 8000, stop it normally or choose a different port and update every client
base URL. Separate Python environments do not isolate ports or unified memory.

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`
- `GET /admin/sessions`
- `POST /admin/cache/clear`

`/v1/responses` is not implemented. OpenAI compatibility means the Models,
Chat Completions, and Completions endpoints listed above; it is not a claim of
full OpenAI API compatibility.

Binding to a non-localhost host requires an API key:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$MTPLX_API_KEY"
```

For Open WebUI, set the OpenAI-compatible base URL to:

```text
http://127.0.0.1:8000/v1
```

For Dockerized Open WebUI, the container must use the host gateway URL, not the host's loopback URL:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" openwebui docker-command
```

That helper disables Open WebUI's Ollama probe and background task generations
so MTPLX only serves visible chat turns by default.

## Hy3 experts.bin fork

This fork is not published to PyPI and collides with upstream MTPLX's Python
distribution/import/CLI names. Follow [INSTALL.md](../INSTALL.md) to create a
dedicated fork environment, then start the package-owned, AR-only Hy3
expert-serving path:

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

The default `--expert-profile auto` selects the largest fitting promoted
weight envelope. The explicit choices are `hy3-oq2e-64`,
`hy3-oq2e-88`, and `hy3-oq2e-96`. Those numbers are weight envelopes, not
machine RAM sizes; the exact process ceilings are 71 GiB, 95 GiB, and 103 GiB.
Every explicit choice still runs installed- and available-memory preflight.

The server constructs AR directly for these profiles. `/health` reports
`generation_mode: "ar"`, only `["ar"]` in `available_generation_modes`, the
admitted artifact revision and digests, selected profile, actual expert I/O
backend, and effective evidence state. A customized profile reports
`customized: true`, clears `evidence_commit`, and exposes its safe effective
configuration instead of claiming promoted evidence.

Use the exact ID returned by `GET /v1/models`. For the tested primary flow it
is `hy3-oq2e-mtplx-streaming`, not the Hugging Face repository ID. Responses
carry the loaded server model ID even if a client submits a stale request
model name.

LiteLLM:

```yaml
model_list:
  - model_name: hy3-local
    litellm_params:
      model: openai/hy3-oq2e-mtplx-streaming
      api_base: http://127.0.0.1:8000/v1
      api_key: mtplx-local
```

Install OpenAI and LiteLLM in a separate client venv, not the fork server venv:
LiteLLM 1.93.0 requires `rich<14`, while the fork requires `rich>=14`. See the
[complete client setup](advanced/ssd-streamed-moe.md#openai-and-litellm-clients).

On loopback, `mtplx-local` satisfies clients that require a nonempty API key
even when server-side authentication is disabled. If the server was started
with `--api-key`, the client key must match it. See the
[SSD-streamed MoE guide](advanced/ssd-streamed-moe.md) for profile geometry,
configuration precedence, contradictions, and admission behavior. This release
does not promote streamed MTP or a GLM expert profile.

For Anthropic Messages-compatible clients, point the client base URL at the
bare server root — no `/v1` suffix:

```text
http://127.0.0.1:8000
```

The Anthropic SDK appends `/v1/messages` itself; a `/v1` base would request
`/v1/v1/messages`, which is not a registered route.

## Android Studio

Android Studio's external model provider should use the OpenAI-compatible URL
schema and the MTPLX `/v1` base URL:

```text
URL: http://127.0.0.1:8008/v1
URL schema: OpenAI-compatible
API key: leave blank for localhost unless MTPLX was started with --api-key
```

Refresh the model list after the server starts. MTPLX supports the OpenAI chat,
streaming, and tool-call request shape used by local coding clients; Gemini-only
proprietary behavior is outside that compatibility contract. To verify a local
setup, run:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" doctor android-studio --port 8008
```

Use `--no-stats-footer` for Open WebUI, Claude Code, OpenCode, and other
clients that treat assistant content as the only user-visible answer. Metrics
remain available at `/metrics`.
