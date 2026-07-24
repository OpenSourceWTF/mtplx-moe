# CLI Guide

## Everyday commands

- `mtplx start` runs onboarding and opens the selected chat surface.
- `mtplx ask` and `mtplx run` execute a one-shot prompt.
- `mtplx quickstart` and `mtplx serve` start the local compatible API server.
- `mtplx settings` inspects persistent, per-run, and live configuration.
- `mtplx models`, `pull`, `inspect`, and `forge` manage model artifacts.
- `mtplx status`, `doctor`, and `stop` operate the local installation.

Run `mtplx help`, `mtplx help commands`, or `mtplx help flags` for generated
parser help. Specialized commands remain under `mtplx help advanced`.

For configuration lookup, use the machine-readable catalog and generated mappings instead of copying
parser output into another tool:

```bash
mtplx settings list --json
mtplx help flags
```

The [canonical settings reference](reference/settings.md) covers reusable settings. The generated
[migration tables](migration-settings.md) cover supported legacy flags and reviewed environment
aliases. An unlisted `MTPLX_*` variable found in source is not automatically a public interface.

## Settings versus command inputs

A setting is a reusable value that can reasonably be saved across runs, such
as the runtime profile, sampler temperature, context window, or session-cache
capacity. Use `settings user set`, `--set`, or `--settings` for those values.

An operand or mechanical option identifies the work for this invocation: a
prompt, an output file, a model to inspect, a recipe path to validate, a server
port to contact, or a benchmark suite. Those remain ordinary command arguments.

```bash
mtplx run "Explain this diff" --set generation.temperature=0.2
mtplx inspect ./models/example
mtplx lab validate ./control.toml
```

## Lab commands

Experiment recipes are inspectable without MLX:

```bash
mtplx lab list
mtplx lab show compiled-verify-control
mtplx lab validate ./control.toml
```

Only active recipes can be applied with `--settings lab:ID`. See
[Experiments](experiments.md) for lifecycle and provenance rules.

## Compatibility

Existing individual flags, flat config keys, environment variables, and
`commands.public` handler imports remain compatibility contracts. New scripts
should prefer canonical settings and domain handler modules. See the
[migration guide](migration-settings.md) for exact mappings.
