# Quickstart

Install the OpenSourceWTF fork directly from GitHub in its own virtual
environment. It is not published to PyPI or the upstream Homebrew tap, and it
uses the same `mtplx` distribution/import/command names as upstream MTPLX.

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
python3 -m venv "$MTPLX_MOE_VENV"
"$MTPLX_MOE_VENV/bin/python" -m pip install --upgrade pip
"$MTPLX_MOE_VENV/bin/python" -m pip install \
  "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
"$MTPLX_MOE_VENV/bin/python" -m pip check
"$MTPLX_MOE_VENV/bin/mtplx" --version

"$MTPLX_MOE_VENV/bin/mtplx" help
"$MTPLX_MOE_VENV/bin/mtplx" doctor --summary
"$MTPLX_MOE_VENV/bin/mtplx" pull \
  Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
"$MTPLX_MOE_VENV/bin/mtplx" inspect \
  Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --json
```

The upstream Homebrew formula, PyPI package, Mac app, and official release
artifacts do not install this fork. The version above must contain
`+opensourcewtf.moe`; otherwise PATH selected upstream MTPLX. See
[the installation guide](../INSTALL.md) for coexistence, upgrades, separate
LiteLLM dependencies, and port collisions.

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
"$MTPLX_MOE_VENV/bin/mtplx" start
"$MTPLX_MOE_VENV/bin/mtplx" start cli
"$MTPLX_MOE_VENV/bin/mtplx" start cli --no-mtp
"$MTPLX_MOE_VENV/bin/mtplx" quickstart --port 8000 --no-stats-footer
```

Both upstream MTPLX and this fork default to port 8000. If
`lsof -nP -iTCP:8000 -sTCP:LISTEN` reports another server, stop it normally or
choose a different port and use that port in every client base URL. Separate
venvs do not isolate unified memory, so avoid loading two large MLX models at
the same time.

`--no-mtp` switches generation to target-only AR. For MTP-equipped models the
MTP runtime stays loaded, so terminal chat can use `/mtp off`, `/mtp on`, and
`/mtp status` without reloading. Native AR-only models such as
`mlx-community/Laguna-S-2.1-oQ4e` instead install an unloaded AR route at
construction because there is no MTP head to retain.

The Laguna download is pinned automatically. It needs about 64.13 GB of disk
space and at least 96 GiB unified memory; 128 GiB is recommended. Its default
context and maximum response are 32,768 tokens. A larger explicit server
context is accepted only when it fits the active Metal resident-memory cap.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
