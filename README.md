<div align="center">

<img src="docs/assets/readme/hero.svg" alt="MTPLX" width="100%" />

# Run local LLMs on Apple Silicon, around twice as fast.

[![PyPI](https://img.shields.io/pypi/v/mtplx?label=PyPI)](https://pypi.org/project/mtplx/)
[![CI](https://github.com/youssofal/MTPLX/actions/workflows/ci.yml/badge.svg)](https://github.com/youssofal/MTPLX/actions/workflows/ci.yml)
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

Requirements: Apple Silicon (M1 or newer), macOS 14+. 16 GB runs the 4B and 9B
models comfortably; 27B wants 32 GB and up. The app checks before recommending
anything.

## Start in 60 seconds

```bash
mtplx start
```

Onboarding chooses the model, runtime mode, and chat surface. To make the
normal long-context profile persistent and inspect why it won:

```bash
mtplx settings user set runtime.profile=sustained
mtplx settings explain runtime.profile
```

For one run only:

```bash
mtplx start --set generation.temperature=0.7
```

See [Getting started](docs/getting-started.md) for server and client setup.

## Configure with settings

Reusable configuration has canonical dotted names. Persist it with `settings
user set`, override one run with repeatable `--set`, or layer data-only TOML and
active experiment recipes with repeatable `--settings`.

```bash
mtplx settings show
mtplx settings list --group generation
mtplx serve --set runtime.mtp.depth=2
mtplx serve --settings ./run.toml
```

Resolution records the winning source and any hard model constraint. User TOML
writes are atomic and private; raw API keys cannot be persisted, only an API
key file reference. Read [Settings](docs/settings.md), the generated [settings
reference](docs/reference/settings.md), and the [migration
guide](docs/migration-settings.md).

For agents and integrations, these are the configuration sources of truth:

| Need | Use |
|---|---|
| Discover the installed canonical catalog | `mtplx settings list --json` |
| Read every canonical name, type, default, lifecycle, visibility, and alias | Generated [settings reference](docs/reference/settings.md) |
| Translate an older flag or reviewed environment variable | Generated [migration tables](docs/migration-settings.md) |
| Find command operands and mechanical flags | `mtplx help flags` or the command's `--help` |
| Classify internal experiment switches | Generated [experiment inventory](docs/experiments/inventory.md) |

Do not infer a public setting from an arbitrary `MTPLX_*` name in source. Environment variables not
listed in the generated migration tables are internal, experimental, diagnostic, or command-owned
until explicitly cataloged.

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
server, so attaching a client does not load a second model.

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
mtplx bench run --settings lab:compiled-verify-control --dry-run
```

On a 16 GB M4 Mac mini, tuning the 9B model lands on depth 1: 14.4 tok/s
baseline becomes 23.0 tok/s. Experiment bundles are strict data-only controls
with lifecycle and normalized hashes; read [Experiments](docs/experiments.md).

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
keys, reviewed environment variables, and `mtplx settings get/set` remain
compatibility aliases; new automation should use canonical settings and
explicit `settings live` scope.

MTPLX also has an opt-in target-only AR path for two pinned community MoE Q4
artifacts that streams routed experts into a bounded hot bank. It is not on by
default and has no full-checkpoint performance claim. See the [advanced
SSD-streamed MoE guide](docs/advanced/ssd-streamed-moe.md).

Use `mtplx help advanced` for QA, profiling, support bundles, and kernel tools.
See the [CLI guide](docs/cli.md), [compatibility mappings](docs/migration-settings.md),
and [documentation index](docs/README.md).

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
