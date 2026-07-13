# Contributing

MTPLX is preview software. Good contributions are small, measurable, and honest about evidence.

Before opening a PR:

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

## Benchmark artifacts

Store raw machine-generated artifacts under
`benchmarks/raw/<benchmark>/<run-id>/`. This tree is ignored by Git; never
force-add files from it. Store only curated, human-readable summaries under
`benchmarks/results/`.

Run IDs must use this grammar:

```text
<benchmark>-<variant>-<YYYYMMDDTHHMMSSZ>-<short-sha>
```

Use lowercase ASCII letters, digits, and hyphens for `<benchmark>` and
`<variant>`. The timestamp must be UTC, and `<short-sha>` must contain 7-12
lowercase hexadecimal characters. Name repeated raw artifacts
`base-r01.json`, `candidate-r01.json`, and `response-r01.md`, incrementing the
two-digit repeat number together for later repeats.

Benchmark summaries must include the complete reproducibility metadata listed
in `benchmarks/results/README.md`. Do not use fan-controlled runs for product
headline claims.

## Local worktrees

Keep auxiliary worktrees as direct children of the workspace-level
`.worktrees/` directory beside your main clone. Do not create worktree
directories inside the clone or scatter them across the workspace root. Never
move a worktree while another process or agent owns it; active worktrees stay in
place until their owner releases them.
