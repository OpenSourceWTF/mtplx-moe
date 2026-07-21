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

The GitHub release wheel remains available for reproducible installs:

```bash
gh release download v0.3.0 --repo youssofal/mtplx --pattern 'mtplx-0.3.0-py3-none-any.whl'
python3 -m pip install ./mtplx-0.3.0-py3-none-any.whl
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
mtplx start
mtplx start cli
mtplx start cli --no-mtp
mtplx quickstart --port 8000 --no-stats-footer
```

`--no-mtp` switches generation to target-only AR while keeping the same runtime load path. In terminal chat, use `/mtp off`, `/mtp on`, and `/mtp status` to switch the next turn without reloading the model.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
