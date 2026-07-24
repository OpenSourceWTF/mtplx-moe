# SSD-streamed MoE

Some mixture-of-experts models are far larger than any Mac's memory, but at each
token only a few experts per layer actually run. MTPLX serves these models by
keeping attention, the shared experts, and every router resident in memory, and
streaming only the experts a token routes to from SSD into a bounded hot bank.
The routed expert tensors are never built into a full parameter tree, so a model
whose weights dwarf your RAM still runs.

This is a target-only autoregressive path: the streamed checkpoints omit their
MTP layer, so multi-token prediction is off here and decoding is ordinary
autoregressive sampling. Output stays exact at your sampling settings.

## Two ready-to-run models

Both are published in MTPLX's streaming layout — resident shards, an
`experts.bin` bank, and an authoritative `expert-manifest.json` — so there is
nothing to convert or build. Pull one and serve it.

| Model | Repo | On disk | Resident in RAM |
|---|---|---|---|
| Hy3 oQ2e | `OpensourceWTF/Hy3-oQ2e-MTPLX-streaming` | ~98 GB | ~9 GB + hot bank + KV |
| GLM-5.2 t158 | `OpensourceWTF/GLM-5.2-t158-MTPLX-streaming` | ~187 GB | ~10 GB + hot bank + KV |

The experts live on SSD and stream in as needed, so the memory you need is the
resident weights plus the hot-expert bank plus KV — not the full download. You
do need disk space for the whole download, and a fast SSD helps: expert reads
are on the decode path.

## Quickstart

```bash
# Download the streaming model (resident shards + experts.bin + manifest).
mtplx pull OpensourceWTF/Hy3-oQ2e-MTPLX-streaming

# Serve it. No manifest to build, no memory math, no model-key flag.
mtplx serve --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming
```

MTPLX detects the streaming layout from the downloaded manifest and turns on
expert streaming for you; you do not pass `--expert-streaming`. The short alias
`hy3-oq2e` works for `serve` too.

It serves the same OpenAI- and Anthropic-compatible API as any other MTPLX model
on `http://127.0.0.1:8000`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Point any OpenAI or Anthropic client at that URL — Cline, Continue, Open WebUI,
the `openai`/`anthropic` Python clients, curl. In Cline, add an
OpenAI-compatible provider with base URL `http://127.0.0.1:8000/v1` and any
model name.

GLM-5.2 is the same flow, just larger:

```bash
mtplx pull OpensourceWTF/GLM-5.2-t158-MTPLX-streaming
mtplx serve --model OpensourceWTF/GLM-5.2-t158-MTPLX-streaming
```

## What the defaults do

Served with no tuning flags, MTPLX:

- reads the served model from the manifest, so the right expert bank loads
  without a `--expert-model-key`;
- sets the process memory ceiling to about 75% of installed RAM;
- sizes KV admission to 32K tokens and gives the rest of the envelope to the
  hot-expert bank, because on this path the number of resident expert slots is
  what governs decode speed.

If the machine cannot hold the resident weights and a minimal expert bank,
serving stops with a clear error instead of thrashing.

You can override any of it. More context, at the cost of decode speed:

```bash
mtplx serve --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --expert-max-live-kv-tokens 131072
```

`--expert-memory-limit`, `--expert-runtime-reserve`, and `--expert-cache-policy`
are there when you want them; `mtplx serve --help` lists the full set under
"SSD expert streaming".

## Faster expert reads (optional)

Expert reads use a portable `preadv` path that works everywhere. For a GIL-free
native reader:

```bash
uv pip install -e native_extensions/expert_io
```

Nothing requires it; it only speeds up the SSD read path.

## Requirements and honesty

- Apple Silicon, macOS 14+, and enough RAM for the resident weights plus a
  usable expert bank. These models want a large-memory Mac; the runtime tells
  you when yours is too small rather than failing obscurely.
- Free disk for the full download (~98 GB / ~187 GB), on a fast SSD.
- This path is opt-in and has no full-checkpoint performance claim.

The Hy3 and GLM-5.2 weights are governed by their upstream licenses (Tencent Hy3
and Zhipu GLM); the streaming builds repackage those weights for this runtime and
do not change their terms.

For the implementation and validation details — resident-selector boundary,
memory-plan contract, artifact layout, parity gates, and benchmark commands —
see the [SSD-streamed MoE plan](../MOE_SSD_STREAMING_PLAN.md).
