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

Install this fork directly from GitHub:

```bash
python3 -m pip install "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
mtplx help
```

The upstream installer, PyPI package, Homebrew formula, Mac app, and official
version numbers belong to the upstream project. They do not install this fork.

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
mtplx max --status
```

If ThermalForge or TG Pro is not present, MTPLX prints install instructions and continues without fan control for `run`, `chat`, and `serve --max`. It must not silently enable spin-loop or clock-anchor modes.

Supported public commands:

```bash
mtplx max --on       # Performance profile
mtplx max --max      # Max profile
mtplx max --off      # Silent profile
mtplx max --status   # tool/status report
```

`MTPLX_GPU_CLOCK_ANCHOR=1` is an explicit experimental diagnostic only. Do not use it for README, release, or product benchmark claims.
