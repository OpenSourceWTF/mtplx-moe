# Install the OpenSourceWTF MTPLX-MOE fork

This repository is a source-installed fork of
[youssofal/MTPLX](https://github.com/youssofal/MTPLX). It is not an official
MTPLX release and is not published to PyPI or the upstream Homebrew tap.

## Requirements

- Apple Silicon Mac
- Python 3.11+
- macOS with MLX support
- Enough disk for the selected model

## Install

### Why this fork needs its own environment

This fork and upstream MTPLX use all three of the same identifiers:

- Python distribution: `mtplx`
- Python import: `mtplx`
- CLI executable: `mtplx`

| Collision surface | What happens | Safe handling |
|---|---|---|
| Same Python environment | Installing one `mtplx` distribution replaces the other | Use a dedicated fork venv |
| Shell PATH | `mtplx` may resolve to Homebrew, upstream Python, or this fork | Use `type -a mtplx` and the fork venv's absolute executable |
| TCP port 8000 | The second server cannot bind, or a client reaches the wrong server | Stop the existing server or give the fork a different port |
| Unified memory and GPU | Separate processes still compete for one Apple-Silicon memory/GPU pool | Do not load two large models unless the machine can hold both |
| `~/.mtplx` runtime data | Both installs see the default config, model cache, receipts, and session cache | Treat the directory as shared; do not delete it merely to switch CLIs |
| LiteLLM dependencies | LiteLLM 1.93.0 and this fork require incompatible `rich` ranges | Put OpenAI/LiteLLM in a client-only venv |

Python cannot install both distributions into one environment. Installing this
fork over upstream MTPLX replaces upstream files in that environment, and
installing upstream `mtplx` later replaces the fork. A Homebrew or Mac-app
install can remain on the machine, but PATH may select a different `mtplx`
command than the one you intended.

Use a dedicated server environment:

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
python3 -m venv "$MTPLX_MOE_VENV"
"$MTPLX_MOE_VENV/bin/python" -m pip install --upgrade pip
"$MTPLX_MOE_VENV/bin/python" -m pip install \
  "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
"$MTPLX_MOE_VENV/bin/python" -m pip check
"$MTPLX_MOE_VENV/bin/mtplx" --version
"$MTPLX_MOE_VENV/bin/mtplx" help
```

The version output must contain `+opensourcewtf.moe`. If it does not, that
command is not this fork.

The upstream installer, PyPI package, Homebrew formula, Mac app, and official
version numbers belong to the upstream project. They do not install this fork.
Do not use `pip install mtplx` for this repository; that name resolves to the
upstream PyPI package.

### Coexistence checks

Use these commands when upstream MTPLX is already installed:

```bash
type -a mtplx
"$MTPLX_MOE_VENV/bin/python" -c \
  'import mtplx; print(mtplx.__file__)'
"$MTPLX_MOE_VENV/bin/mtplx" --version
```

The import path must be inside `$MTPLX_MOE_VENV`, and the version must identify
the OpenSourceWTF fork. Activating the venv is optional; using its absolute
paths is unambiguous and leaves the upstream installation untouched.

Virtual environments do not isolate listening ports, model memory, or the
shared `~/.mtplx` runtime directory. The Hugging Face repository ID gives the
streamed model its own model-cache directory, but both installations still see
the default config, receipts, and session cache. Only one process can listen on
a given host and port. Check the default port before serving:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

If another MTPLX server owns port 8000, either stop it normally or use a
different port:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" serve \
  --model OpensourceWTF/Hy3-oQ2e-MTPLX-streaming \
  --download \
  --port 18080
```

Clients must then use `http://127.0.0.1:18080/v1`. Do not start two large MLX
models merely because their Python environments are separate; both consume the
same unified memory and GPU.

### OpenAI and LiteLLM dependencies

The OpenAI SDK and LiteLLM are clients of the MTPLX server; they do not need to
share its Python environment. Keep them in a separate client environment:

```bash
MTPLX_CLIENT_VENV="$HOME/.venvs/mtplx-clients"
python3 -m venv "$MTPLX_CLIENT_VENV"
"$MTPLX_CLIENT_VENV/bin/python" -m pip install --upgrade pip
"$MTPLX_CLIENT_VENV/bin/python" -m pip install \
  "openai>=1" "litellm[proxy]"
"$MTPLX_CLIENT_VENV/bin/python" -m pip check
```

Do not install `litellm[proxy]` into the fork server environment. LiteLLM
1.93.0 requires `rich<14`, while this fork requires `rich>=14`; installing both
distributions together leaves that environment with a broken declared
dependency set. The separate client environment above was verified with
OpenAI 2.48.0 and LiteLLM 1.93.0 and contains no `mtplx` package.

See the
[SSD-streamed MoE client examples](docs/advanced/ssd-streamed-moe.md#openai-and-litellm-clients)
for served-model discovery and LiteLLM configuration.

### Upgrade this fork

Upgrade only the dedicated fork environment:

```bash
"$MTPLX_MOE_VENV/bin/python" -m pip install --upgrade \
  "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
"$MTPLX_MOE_VENV/bin/python" -m pip check
"$MTPLX_MOE_VENV/bin/mtplx" --version
```

For local development:

```bash
python -m pip install -e ".[dev,server]"
```

## Runtime dependencies

`mtplx --help`, `mtplx doctor`, `mtplx inspect`, `mtplx settings`, and `mtplx init` are designed to
work without loading MLX. Generation and serving require the Apple-Silicon `mlx`/`mlx-lm` dependencies
declared by the installed MTPLX release and a compatible model. Use `mtplx doctor --summary` instead
of installing an old dependency recipe from a copied guide.

## Optional Thermal Tools

`--max` is opt-in. It is for users who need sustained throughput and accept fan noise. It is never part of the default quick start and is never used for no-fan product claims.

Check the local thermal-control state:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" max --status
```

If ThermalForge or TG Pro is not present, MTPLX prints install instructions and continues without fan control for `run`, `chat`, and `serve --max`. It must not silently enable spin-loop or clock-anchor modes.

Supported public commands:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" max --on       # Performance profile
"$MTPLX_MOE_VENV/bin/mtplx" max --max      # Max profile
"$MTPLX_MOE_VENV/bin/mtplx" max --off      # Silent profile
"$MTPLX_MOE_VENV/bin/mtplx" max --status   # tool/status report
```

`MTPLX_GPU_CLOCK_ANCHOR=1` is an explicit experimental diagnostic only. Do not use it for README, release, or product benchmark claims.
