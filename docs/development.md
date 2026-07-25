# Development

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

Keep generated artifacts, model weights, and local credentials out of Git.

## Fork artifacts

The OpenSourceWTF fork is not published to PyPI or the upstream Homebrew tap.
Build a local wheel and source archive with:

```bash
python -m build
python -m twine check dist/*
```

Install the built fork wheel into a clean environment for smoke testing:

```bash
python3 -m venv /tmp/mtplx-moe-fork-verify
/tmp/mtplx-moe-fork-verify/bin/python -m pip install dist/mtplx-*.whl
/tmp/mtplx-moe-fork-verify/bin/mtplx --version
```

The `build fork artifacts` workflow performs the same build and smoke checks
for tags or a manually selected ref. It cannot publish to PyPI.
