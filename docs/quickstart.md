# Quickstart

```bash
brew install youssofal/mtplx/mtplx

mtplx help
mtplx doctor --summary
mtplx pull Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
mtplx inspect Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --json
```

Homebrew is the recommended macOS path. Python-only installs can use PyPI:

```bash
python3 -m pip install -U mtplx
```

Pin the PyPI version when a reproducible install is required:

```bash
VERSION=X.Y.Z  # replace with the release to reproduce
python3 -m pip install "mtplx==$VERSION"
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
mtplx start
mtplx start cli
mtplx start cli --set runtime.mtp.enabled=false
mtplx quickstart --set runtime.profile=sustained --port 8000 --no-stats-footer
```

`runtime.mtp.enabled=false` switches generation to target-only AR while keeping the same runtime load
path. In terminal chat, use `/mtp off`, `/mtp on`, and `/mtp status` to switch the next turn without
reloading the model. See [Settings](settings.md) for persistent and per-run scopes.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
