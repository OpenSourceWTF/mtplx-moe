<div align="center">

<img src="docs/assets/readme/hero.svg" alt="MTPLX" width="100%" />

# Run local LLMs on Apple Silicon, around twice as fast.

[![CI](https://github.com/OpenSourceWTF/mtplx-moe/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSourceWTF/mtplx-moe/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

</div>

> [!IMPORTANT]
> This repository is the OpenSourceWTF `mtplx-moe` fork of
> [youssofal/MTPLX](https://github.com/youssofal/MTPLX). It is not an official
> MTPLX release and is not published to PyPI or the upstream Homebrew tap.

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

## Install this fork

The fork and upstream MTPLX intentionally have the same Python distribution
name (`mtplx`), import package (`mtplx`), and CLI command (`mtplx`). They cannot
coexist in one Python environment. Installing either one into an environment
that already contains the other replaces that environment's MTPLX install.

Use a dedicated virtual environment for this fork:

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
python3 -m venv "$MTPLX_MOE_VENV"
"$MTPLX_MOE_VENV/bin/python" -m pip install --upgrade pip
"$MTPLX_MOE_VENV/bin/python" -m pip install \
  "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
"$MTPLX_MOE_VENV/bin/python" -m pip check
"$MTPLX_MOE_VENV/bin/mtplx" --version
```

The version must contain `+opensourcewtf.moe`. Use the venv's absolute
`mtplx` path when the upstream CLI, Homebrew formula, or Mac app is also
installed; `type -a mtplx` shows every PATH-visible command.

The upstream Mac app, PyPI package, Homebrew tap, and official release numbers
belong to [youssofal/MTPLX](https://github.com/youssofal/MTPLX). They do not
install this fork. In particular, `pip install mtplx` installs the upstream
PyPI package, not this repository. See [INSTALL.md](INSTALL.md) for coexistence,
upgrade, port-collision, and separate LiteLLM environment guidance.

This fork adds package-owned SSD-streamed MoE serving for Hy3, GLM-5.2, and
Kimi K3. Hy3 is the zero-configuration entrypoint:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download
```

Requirements: Apple Silicon (M1 or newer), macOS 14+. 16 GB runs the 4B and 9B
models comfortably; 27B wants 32 GB and up. The app checks before recommending
anything.

## Supported models

This is the complete list of preconfigured or revision-pinned Hugging Face
repositories recognized by this fork. Other compatible or locally forged
models may pass `mtplx inspect`, but they are not part of this verified roster.

The catalog's peak-memory figures are planning estimates before the CLI applies
its safety margin and request-specific KV budget. “M1–M2” and “M3–M5” identify
the catalog's preferred hardware lane, not a hard architecture restriction.

### MTP catalog

| Exact model ID | Runtime | Approx. peak RAM | Preferred lane |
|---|---|---:|---|
| [`Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed) | MTP or AR | 2.86 GiB | M3–M5 |
| [`Youssofal/Qwen3.5-4B-MTPLX-Optimized-Quality`](https://huggingface.co/Youssofal/Qwen3.5-4B-MTPLX-Optimized-Quality) | MTP or AR | 4.75 GiB | M3–M5 |
| [`Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed) | MTP or AR | 10.0 GiB | M3–M5 |
| [`Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16`](https://huggingface.co/Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16) | MTP or AR | 10.5 GiB | M1–M2 |
| [`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed) | MTP or AR | 17.0 GiB | M3–M5 |
| [`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16`](https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16) | MTP or AR | 17.5 GiB | M1–M2 |
| [`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality`](https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality) | MTP or AR | 27.62 GiB | M3–M5 |
| [`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16`](https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16) | MTP or AR | 28.12 GiB | M1–M2 |
| [`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed) | MTP or AR | 28.0 GiB | M3–M5 |
| [`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16) | MTP or AR | 28.5 GiB | M1–M2 |
| [`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance) | MTP or AR | 32.0 GiB | M3–M5 |
| [`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16) | MTP or AR | 32.5 GiB | M1–M2 |
| [`Youssofal/Gemma4-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Gemma4-MTPLX-Optimized-Speed) | MTP or AR | 18.0 GiB | M3–M5 |

Launch any catalog model by exact ID:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" start cli \
  --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --download
```

### Fork-specific models

Quality and speed appear only where this fork retains a receipt for the exact
artifact and quantization. “Not measured” means there is no comparable retained
run, not that the result is zero.

| Exact model ID | Runtime | Capacity requirement | HumanEvalPlus pass@1 | Retained decode results | Support status |
|---|---|---|---|---|---|
| [`OpensourceWTF/Hy3-oQ2e-MTPLX-streaming`](https://huggingface.co/OpensourceWTF/Hy3-oQ2e-MTPLX-streaming) | SSD-streamed AR; retained MTP benchmark | 71, 95, or 103 GiB process ceiling; about 97.6 GB download | `q4` requant: 86.6% (142/164); `q8`: 87.2% (143/164) | Flagship: **48.04 tok/s** with q4 requant at MTP depth 1; 41.36 tok/s AR control. Promoted AR profiles: `-64`: 9.31; `-88`: 22.35; `-96`: 30.17 tok/s | Promoted `hy3-oq2e-64`, `-88`, and `-96` profiles |
| [`OpensourceWTF/GLM-5.2-t158-MTPLX-streaming`](https://huggingface.co/OpensourceWTF/GLM-5.2-t158-MTPLX-streaming) | SSD-streamed AR | Measured 96 GiB configuration; about 187 GB download | Not measured | Not measured | Manual and experimental; no task-quality receipt |
| [`OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX-streaming`](https://huggingface.co/OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX-streaming) | SSD-streamed AR | Measured 96 or 110 GiB configuration; about 752 GB download | Not measured | 96 GiB: **1.18 tok/s**; 110 GiB: **1.11 tok/s**, both at 1,024/1,024 | Manual and experimental; text-only, no MTP |
| [`mlx-community/Laguna-S-2.1-oQ4e`](https://huggingface.co/mlx-community/Laguna-S-2.1-oQ4e) | Native AR only | At least 96 GiB unified memory; 128 GiB recommended; 64.13 GB download | Not measured | Not measured | Exact revision pinned and artifact-verified |

The HumanEvalPlus v0.1.10 receipts use all 164 tasks, one sample per task,
greedy decoding, temperature 0, seed 42, and the chat endpoint. They were
collected through the MTP depth-2 evaluation lane. Requantizing the supported
resident trunk projections from q8 to q4 changed the result by one task:
86.6% (142/164) versus 87.2% (143/164), with no directional quality signal
(paired McNemar exact two-sided p=1.0).

The flagship Hy3 receipt rounds to **48 tok/s**. It is a three-repetition mean
on an M5 Max with 128 GB unified memory: q4/gs64 resident-trunk requantization,
the published oQ2e expert bank, all 79 routed layers pinned as islands, BF16
KV, MTP depth 1, a 1,024-token real-code prompt, and 256 generated tokens. Its
matching AR control averaged 41.36 tok/s. The q4 pass covers trunk
`*_proj` matrices only; it does not requantize `experts.bin`, routers,
embeddings, the LM head, or norms. The per-run values, exact settings, source
commit, and receipt hashes are in the
[flagship summary](evals/tier2/hy3_oq2e_rq4_flagship_summary.json).

The current `mtplx serve` route is AR-only, so 48.04 tok/s is the retained
MTP benchmark result rather than API-server throughput. The promoted-profile
figures are separate single-stream AR receipts: `-64` is the three-repetition
mean for the zero-island cache-heavy geometry at its 16K-KV performance
envelope, while `-88` and `-96` are retained 4K-ceiling measurements. The
installed `-64` profile uses q4 trunk requantization and a 4K KV ceiling;
`-88` and `-96` retain q8 trunks unless explicitly overlaid. Different
hardware, context, cache warmth, or concurrent work will change absolute
throughput. The [SSD-streamed MoE guide](docs/advanced/ssd-streamed-moe.md#hy3-q4-resident-trunk-requantization)
shows the requant setting.

Kimi K3 requantizes only its 92 routed-MoE layers from the pinned Q2_K source
to t158. Eligible resident linear and embedding weights are dynamically
quantized to q8 during construction; the checkpoint is not rewritten, and
routers, residual score projections, norms, convolutions, recurrent vectors,
and biases retain source precision. The retained M5 Max run used exactly 1,024
prompt and 1,024 output tokens while a four-worker Hugging Face upload remained
active. The 96 GiB plan measured 7.39 tok/s prefill, 1.18 tok/s decode, and
83.8 GiB peak physical footprint; the 110 GiB plan measured 6.43 tok/s,
1.11 tok/s, and 97.9 GiB. Outputs were identical with zero I/O or integrity
errors. The larger cache reduced reads but did not improve throughput in that
concurrent-load run, so this is a memory-safety receipt rather than a speed
claim.

Hy3 starts with the zero-config `mtplx serve` command above. GLM and Kimi
require their manual paging configurations in the
[SSD-streamed MoE guide](docs/advanced/ssd-streamed-moe.md); Kimi includes
separate 96 GiB and 110 GiB launch examples. Laguna launches with the same
`mtplx start cli --model MODEL_ID --download` form as the catalog models and
rejects MTP before loading.

## Start in 60 seconds

```bash
"$MTPLX_MOE_VENV/bin/mtplx" start
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
"$MTPLX_MOE_VENV/bin/mtplx" quickstart --port 8000
"$MTPLX_MOE_VENV/bin/mtplx" connect openwebui
"$MTPLX_MOE_VENV/bin/mtplx" start opencode --port 18083
```

The server exposes OpenAI-compatible `/v1/chat/completions`,
`/v1/completions`, and `/v1/models`, plus Anthropic-compatible `/v1/messages`,
streaming, tool calls, `/health`, and `/metrics`. The app and CLI share one
server, so attaching a client does not load a second model. `/v1/responses` is
not implemented; OpenAI compatibility here names the supported endpoints, not
the full OpenAI API.

The fork and upstream servers also default to port 8000. A virtual environment
isolates Python packages, not TCP ports or unified memory. If another MTPLX
server is already running, stop it or run this fork on a different port, such
as `--port 18080`, and use the matching client base URL. For the streamed Hy3
quickstart, `/v1/models` returns `hy3-oq2e-mtplx-streaming`; the Hugging Face
repository ID used for download is not the served API model ID. The
[SSD-streamed MoE guide](docs/advanced/ssd-streamed-moe.md#openai-and-litellm-clients)
shows isolated OpenAI and LiteLLM client setup.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3-oq2e-mtplx-streaming","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Warm-prefix session state keeps multi-turn chats fast, and an optional SSD
cache restores sessions across restarts.

## Tune and benchmark

Draft depth depends on chip, memory bandwidth, and thermals. MTPLX measures the
real model at each depth with autoregressive decoding as the baseline, saves a
depth only when it wins, and says so when none does.

```bash
"$MTPLX_MOE_VENV/bin/mtplx" tune --retune
"$MTPLX_MOE_VENV/bin/mtplx" bench aime --quick
```

On a 16 GB M4 Mac mini, tuning the 9B model lands on depth 1: 14.4 tok/s
baseline becomes 23.0 tok/s.

On a 128 GB M4 Max (Mac16,5, macOS 26.5.1, fans on auto), the same
[`Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed)
catalog model tunes to depth 3 under `performance-cold`: 56.4 tok/s AR becomes
114.5 tok/s (2.03×). A follow-up `bench aime --quick` on that depth scored
5/5. Receipt:
[community M4 Max 9B tune + AIME](benchmarks/results/community-qwen35-9b-tune-aime-m4max-128gb-20260725.json).

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

The app recommends among the preconfigured models in the
[supported-model table](#supported-models) for the detected hardware.

## Advanced and compatibility

`mtplx inspect` classifies models before loading: verified,
architecture-compatible but unverified, incompatible architecture, or no MTP
heads. There are no silent fallbacks. Existing individual flags, flat config
keys, and reviewed environment variables remain compatibility controls.

MTPLX can also serve mixture-of-experts models larger than a selected memory
envelope by streaming routed experts from prepacked Hugging Face banks. The
promoted OpenSourceWTF fork profiles are specifically for
`OpensourceWTF/Hy3-oQ2e-MTPLX-streaming`; the published
`OpensourceWTF/GLM-5.2-t158-MTPLX-streaming` and
`OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX-streaming` artifacts use manual,
experimental paging configurations. This fork does not promote GLM or Kimi
profiles or streamed MTP. The primary command above admits the Hy3 artifact
once, selects `hy3-oq2e-64`, `hy3-oq2e-88`, or `hy3-oq2e-96`, and constructs
the AR route directly.

The numbers in those names are weight envelopes, not required machine RAM.
Their exact process ceilings, including the 7 GiB runtime reserve, are 71 GiB,
95 GiB, and 103 GiB. `auto` chooses the largest promoted profile whose process
ceiling fits both installed RAM and launch-time available memory. See the
[SSD-streamed MoE guide](docs/advanced/ssd-streamed-moe.md) for explicit
Hy3 island profiles, the measured GLM t158 paging configuration, both measured
Kimi K3 memory plans, advanced override precedence, health evidence, and
LiteLLM setup.

Use `mtplx help advanced` for QA, profiling, support bundles, and kernel tools.
See the [documentation index](docs/README.md).

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

Upstream MTPLX is built by
[Youssof Altoukhi](https://github.com/youssofal). This fork is maintained by
[OpenSourceWTF](https://github.com/OpenSourceWTF); report fork-specific bugs
and benchmark replications in
[OpenSourceWTF/mtplx-moe Issues](https://github.com/OpenSourceWTF/mtplx-moe/issues).
