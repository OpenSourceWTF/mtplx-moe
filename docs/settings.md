# Settings

MTPLX uses canonical dotted setting names for reusable configuration. Command
inputs such as prompts, model files to forge, output paths, and benchmark suites
remain ordinary operands.

The generated [settings reference](reference/settings.md) lists every supported
name, type, default, lifecycle, visibility, apply behavior, and compatibility
alias.

## Inspect effective settings

These commands do not import MLX or load a model:

```bash
mtplx settings show
mtplx settings list --group runtime
mtplx settings explain runtime.profile
```

`show` includes the winning source. `explain` adds requested values, shadowed
sources, constraints, aliases, and whether a setting is live-mutable.

## Persistent user settings

Persist reusable choices explicitly in the user scope:

```bash
mtplx settings user set runtime.profile=sustained
mtplx settings user set generation.temperature=0.7 generation.top_p=0.9
mtplx settings user show
mtplx settings user unset generation.temperature
```

Writes update `~/.mtplx/config.toml` atomically with mode `0600`. The complete
candidate configuration is validated before the destination is replaced.
Existing flat config keys remain readable as compatibility aliases.

## Per-run settings

Runtime and product benchmark commands accept repeatable `--set` values:

```bash
mtplx start --set generation.temperature=0.7
mtplx serve --set runtime.mtp.depth=2 --set generation.reasoning=off
mtplx bench run --set generation.temperature=0.6 --dry-run
```

Values are typed by the catalog. Unknown names fail with close-name
suggestions before the runtime handler starts.

## Data-only bundles

Use `--settings` for a repeatable TOML bundle that does not modify user state:

```toml
[settings]
"runtime.profile" = "sustained"
"generation.temperature" = 0.7
```

```bash
mtplx serve --settings ./run.toml
mtplx serve --settings ./base.toml --settings ./machine.toml
```

Later bundles override earlier bundles. Bundles accept only the `[settings]`
table and scalar values; shell commands, substitutions, nested directives, and
environment expansion are rejected. Active experiment controls use the same
path through a `lab:` URI; see [Experiments](experiments.md).

## Live settings

Live operations are deliberately scoped to a running daemon:

```bash
mtplx settings live show
mtplx settings live set depth=2 reasoning=off
```

The daemon remains authoritative about which keys are mutable. A restart-only
key returns a structured error. Historical `mtplx settings get/set` forms are
still accepted as compatibility aliases.

## Precedence

From strongest to weakest, resolution is:

1. Hard safety and model-compatibility constraints.
2. Repeated `--set NAME=VALUE` entries.
3. Explicit legacy CLI setting flags.
4. Repeated `--settings PATH` bundles, with the last bundle winning.
5. Reviewed `MTPLX_*` environment compatibility aliases.
6. Persistent user settings.
7. Model-specific defaults when available.
8. The selected product profile.
9. Built-in defaults.

Constraints retain both requested and effective values with a reason. Use
`settings explain` instead of inferring precedence from command output.

## Secrets and API key file handling

Persist a reference to an API key file, never the raw key:

```bash
mtplx settings user set server.api_key_file=~/.mtplx/api-key
```

The API key file setting is redacted in settings output and benchmark
provenance. `server.api_key` is not a persistable setting.

## Common recipes

```bash
# Stable reusable profile
mtplx settings user set runtime.profile=sustained

# One lower-temperature run
mtplx ask "Summarize this patch" --set generation.temperature=0.2

# Limit context for one server launch
mtplx serve --set model.context_window=32768

# Adjust persistent SSD session-cache capacity
mtplx settings user set cache.session.ssd.max_size=50GB
```

For raw legacy mappings, see [Migrating flags and environment
variables](migration-settings.md). Advanced streamed-expert options remain a
specialized compatibility surface documented in the advanced guide.
