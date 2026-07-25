# Server

The server provides OpenAI-compatible local serving and Anthropic Messages compatibility for coding
clients and harnesses.

```bash
mtplx serve --host 127.0.0.1 --port 8000 --no-stats-footer
```

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
mtplx serve --host 0.0.0.0 --port 8000 --api-key "$MTPLX_API_KEY"
```

For Open WebUI, set the OpenAI-compatible base URL to:

```text
http://127.0.0.1:8000/v1
```

For Dockerized Open WebUI, the container must use the host gateway URL, not the host's loopback URL:

```bash
mtplx openwebui docker-command
```

That helper disables Open WebUI's Ollama probe and background task generations
so MTPLX only serves visible chat turns by default.

## Hy3 experts.bin release candidate

Install and start the package-owned, AR-only Hy3 expert-serving path:

```bash
python3 -m pip install mtplx==2.3.1rc1
mtplx serve \
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

Use the exact ID returned by `GET /v1/models`. For the primary flow it is
`OpensourceWTF/Hy3-oQ2e-MTPLX-streaming`; responses carry the loaded server
model ID even if a client submits a stale request model name.

LiteLLM:

```yaml
model_list:
  - model_name: hy3-local
    litellm_params:
      model: openai/OpensourceWTF/Hy3-oQ2e-MTPLX-streaming
      api_base: http://127.0.0.1:8000/v1
      api_key: mtplx-local
```

On loopback, `mtplx-local` satisfies clients that require a nonempty API key
even when server-side authentication is disabled. If the server was started
with `--api-key`, the client key must match it. See the
[SSD-streamed MoE guide](advanced/ssd-streamed-moe.md) for profile geometry,
configuration precedence, contradictions, and admission behavior. This release
does not promote streamed MTP or a GLM expert profile.

For Anthropic Messages-compatible clients, point the client base URL at the
same local server root:

```text
http://127.0.0.1:8000/v1
```

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
mtplx doctor android-studio --port 8008
```

Use `--no-stats-footer` for Open WebUI, Claude Code, OpenCode, and other
clients that treat assistant content as the only user-visible answer. Metrics
remain available at `/metrics`.
