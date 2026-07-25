<div align="center">

<img src="docs/assets/readme/hero.svg" alt="MTPLX" width="100%" />

# Run local LLMs on Apple Silicon, around twice as fast.

[![PyPI](https://img.shields.io/pypi/v/mtplx?label=PyPI)](https://pypi.org/project/mtplx/)
[![CI](https://github.com/OpenSourceWTF/mtplx-moe/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSourceWTF/mtplx-moe/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

</div>

MTPLX is a native Mac app and command line for running local language models
with multi-token prediction. Modern Qwen 3.5/3.6 models ship with built-in MTP
heads; MTPLX uses them to draft ahead, verify in one batched forward pass, and
keep only tokens that pass exact rejection sampling. Same model, same output
distribution, measured 1.6x faster on a 16 GB M4 Mac mini and 2.24x on an M5
Max.

There is no second draft model eating RAM and no greedy shortcut that changes
what the model would have said at real sampling settings. The acceptance math
is the Leviathan and Chen rejection sampling theorem with residual correction,
so `temperature=0.6, top_p=0.95` behaves like normal decoding, just faster.

## Get it

**The Mac app** is the easiest route. Download the DMG at
[mtplx.com](https://mtplx.com/download), drag it to Applications, and the app
checks hardware, recommends and downloads a model that fits, installs its own
Python engine, configures fan control, puts `mtplx` on `PATH`, and tunes draft
depth on the Mac.

**The CLI**:

```bash
brew install youssofal/mtplx/mtplx
```

Pip also works: `python3 -m pip install mtplx`. Releases are listed at
[mtplx.com/releases](https://mtplx.com/releases/).

The 2.3.1 release candidate adds the package-owned Hy3 `experts.bin` serving
path:

```bash
python3 -m pip install mtplx==2.3.1rc1
mtplx serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

Requirements: Apple Silicon (M1 or newer), macOS 14+. 16 GB runs the 4B and 9B
models comfortably; 27B wants 32 GB and up. The app checks before recommending
anything.

## Start in 60 seconds

```bash
mtplx start
```

Onboarding chooses the model, runtime mode, and chat surface. See the
[quickstart](docs/quickstart.md) for server and client setup.

## App

<img src="docs/assets/readme/app-dashboard.jpg" alt="MTPLX dashboard with live decode gauge" width="100%" />

The dashboard shows live tokens per second, acceptance by draft depth, verify
waterfall, cache state, and system pressure while the model runs.

<img src="docs/assets/readme/app-chat.jpg" alt="Chat streaming with live speed badge" width="100%" />

Chat is native, streams with thinking cards, accepts file attachments, and can
search the web. One click launches OpenCode, Pi, Hermes, Open WebUI, or another
OpenAI/Anthropic-compatible client against the local server. The app also has
an AIME runner with disclosed, coaching-free prompts.

## Connect clients and APIs

Start the API server and print client configuration:

```bash
mtplx quickstart --port 8000
mtplx connect openwebui
mtplx start opencode --port 18083
```

The server exposes OpenAI-compatible `/v1/chat/completions`,
`/v1/completions`, and `/v1/models`, plus Anthropic-compatible `/v1/messages`,
streaming, tool calls, `/health`, and `/metrics`. The app and CLI share one
server, so attaching a client does not load a second model. `/v1/responses` is
not implemented; OpenAI compatibility here names the supported endpoints, not
the full OpenAI API.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Warm-prefix session state keeps multi-turn chats fast, and an optional SSD
cache restores sessions across restarts.

## Tune and benchmark

Draft depth depends on chip, memory bandwidth, and thermals. MTPLX measures the
real model at each depth with autoregressive decoding as the baseline, saves a
depth only when it wins, and says so when none does.

```bash
mtplx tune --retune
mtplx bench aime --quick
```

On a 16 GB M4 Mac mini, tuning the 9B model lands on depth 1: 14.4 tok/s
baseline becomes 23.0 tok/s.

## Modes

| Mode | What it does | When |
|---|---|---|
| **Sustained** | Default long-context MTP with chunked prefill and request-sized KV | Everyday use, big files, 16K-200K prompts |
| **Sustained Max** | Sustained with fans pinned at 100% | Long work where maximum cooling matters |
| **Burst** | Legacy short-context benchmark lane, loud | Short prompts and benchmarks only |

Fan-backed modes restore automatic fan control even after `kill -9` or closing
the terminal; a detached watchdog handles it.

## Forge

<img src="docs/assets/readme/app-forge.jpg" alt="Forge verifying a freshly built MTP model" width="100%" />

Forge converts a Hugging Face repository to MLX, trains an MTP adapter, verifies
that it is exact and actually faster, and can publish it back to the Hub. It
reports measured verdicts rather than assuming training helped (for example,
"Depth 1 is fastest: 227.1 to 296.1, 1.30x"). Use the app or `mtplx forge`.

The official [Youssofal catalog](https://huggingface.co/Youssofal) includes
Qwen 3.5 (4B, 9B), Qwen 3.6 (27B, 35B MoE) speed/balance/quality builds, and
Gemma 4. The app recommends among them for the detected hardware.

## Advanced and compatibility

`mtplx inspect` classifies models before loading: verified,
architecture-compatible but unverified, incompatible architecture, or no MTP
heads. There are no silent fallbacks. Existing individual flags, flat config
keys, and reviewed environment variables remain compatibility controls.

MTPLX can also serve mixture-of-experts models larger than a selected memory
envelope by streaming routed experts from prepacked Hugging Face banks. The
promoted 2.3.1rc1 profiles are specifically for
`OpensourceWTF/Hy3-oQ2e-MTPLX-streaming`; the published
`OpensourceWTF/GLM-5.2-t158-MTPLX-streaming` artifact uses a manual,
experimental paging configuration. This release does not promote GLM profiles
or streamed MTP. The primary command above admits the artifact once, selects
`hy3-oq2e-64`, `hy3-oq2e-88`, or `hy3-oq2e-96`, and constructs the AR route
directly.

The numbers in those names are weight envelopes, not required machine RAM.
Their exact process ceilings, including the 7 GiB runtime reserve, are 71 GiB,
95 GiB, and 103 GiB. `auto` chooses the largest promoted profile whose process
ceiling fits both installed RAM and launch-time available memory. See the
[SSD-streamed MoE guide](docs/advanced/ssd-streamed-moe.md) for explicit
Hy3 island profiles, the measured GLM t158 paging configuration, advanced
override precedence, health evidence, and LiteLLM setup.

Use `mtplx help advanced` for QA, profiling, support bundles, and kernel tools.
See the [documentation index](docs/README.md).

### Laguna S-2.1

[Laguna-S-2.1 oQ4e](https://huggingface.co/mlx-community/Laguna-S-2.1-oQ4e) is supported through its exact MLX architecture in target-only AR mode:

```bash
mtplx start cli \
  --model mlx-community/Laguna-S-2.1-oQ4e \
  --download \
  --no-mtp
```

MTPLX pins that model to revision
`8e3f5cad513746264940c1c4195de48d7ea345a5` and verifies the 13-shard layout,
tokenizer, generation config, special tokens map, and Poolside chat template
before admitting it. The checkpoint has no native MTP head, so an MTP launch is
rejected before weights load instead of falling back during execution. The
weights occupy 59.72 GiB (64.13 GB); use a Mac with at least 96 GiB unified
memory (128 GiB is recommended). MTPLX defaults Laguna to a 32,768-token context
and response cap, and checks larger explicit server contexts against the active
Metal memory cap.

## What MTPLX is not

- Not an external-drafter system. The drafter is the target model's own MTP heads.
- Not a greedy-argmax trick. Acceptance is exact rejection sampling at any temperature.
- Not a CUDA project. MTPLX is MLX-native and Apple Silicon first. For Linux, use vLLM.

## License and credit

Apache-2.0: use it, modify it, and ship it commercially. Keep the license and
[NOTICE](NOTICE) attribution when redistributing. MTPLX builds on
[MLX](https://github.com/ml-explore/mlx), Qwen, and Gemma; speculative sampling
follows Leviathan and Chen (2023). Fan control uses
[ThermalForge](https://github.com/ProducerGuy/ThermalForge). Model weights keep
their upstream licenses.

If MTPLX powers a public project, benchmark, or paper, please credit it:

> Powered by MTPLX by Youssof Altoukhi
> https://github.com/youssofal/MTPLX

Built by [Youssof Altoukhi](https://github.com/youssofal). Bug reports and
benchmark replications are welcome via
[Issues](https://github.com/youssofal/MTPLX/issues).
