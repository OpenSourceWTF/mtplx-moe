# Experiment Settings Bundles

Experiment recipes are data-only TOML controls. They contain required ownership,
tracking, model-family, review-date, lifecycle, purpose, and scalar settings
metadata. They cannot execute shell or Python code, expand environment
variables, or write files.

## Inspect and validate

```bash
mtplx lab list
mtplx lab list --all
mtplx lab show compiled-verify-control --json
mtplx lab validate ./candidate.toml --json
```

Validation checks the strict recipe schema and every canonical setting name,
type, and visibility without importing MLX.

## Apply an active control

```bash
mtplx serve --settings lab:compiled-verify-control
mtplx bench run --settings lab:packed-gqa-control --dry-run
```

The catalog validates a known model family before runtime loading. A repeated
`--set` remains stronger than the recipe, making the override explicit in
settings provenance.

## Lifecycle

- `active`: executable until its future review date.
- `retained`: archived evidence that remains useful but is not executable.
- `rejected`: a measured or reviewed dead end; never executable.
- `superseded`: replaced by another recipe or implementation.
- `expired`: passed its review window without renewal.

Archived recipes may record a result and replacement. They stay inspectable
with `lab list --all` but `lab:` resolution refuses them.

## Hashes and provenance

Each recipe has a normalized SHA-256 derived from canonical metadata and
settings, so whitespace-only TOML edits do not change identity. Product
benchmark envelopes record the ordered bundle id, SHA-256, source URI,
redacted effective settings, and winning sources without changing benchmark
scoring.

The generated [experiment inventory](experiments/inventory.md) groups active
controls and compatibility switches, including ownership gaps recommended for
retention or investigation. A valid recipe does not authorize a hardware run;
normal isolation and benchmark approval still apply.
