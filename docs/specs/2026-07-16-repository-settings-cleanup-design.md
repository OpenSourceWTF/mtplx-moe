# Repository Settings Cleanup Design

Date: 2026-07-16
Status: Design complete; awaiting user review
Base commit: `f3e08cb0693fcd4938ab5834b2b9422ececbb6e3`

## Problem

MTPLX has accumulated product controls, benchmark inputs, experimental toggles,
compatibility aliases, and internal process state behind one undifferentiated
command-line and environment-variable surface.

The audited base contains:

- 367 distinct `MTPLX_*` names in production Python.
- 341 distinct long-form flags declared by the root CLI parser.
- 16 `MTPLX_*` names represented in ordinary Markdown outside historical
  results and plans.
- A 3,862-line root parser, a 13,738-line public command module, and several
  runtime modules above 2,000 lines.

The public/advanced help split and product profiles are useful foundations, but
they do not provide one schema, one resolution path, or a lifecycle for
experiments.

## Goal

Replace individual runtime-configuration flags with a hierarchical settings
system while preserving current behavior for existing scripts during a bounded
compatibility period.

Normal use should look like this:

```bash
mtplx settings user set runtime.profile=sustained
mtplx settings user set generation.temperature=0.6
mtplx start

# One-run overrides do not mutate persistent settings.
mtplx start --set generation.temperature=0.7

# Reproducible bundles work for deployments and experiments.
mtplx bench prefill-ladder --settings ./runs/long-context.toml
mtplx bench prefill-ladder --settings lab:compiled-verify-control
```

Individual flags remain only when they describe command mechanics or operands,
not reusable runtime configuration. Examples include `--help`, `--json`,
`--dry-run`, output paths, benchmark input data, and destructive-action
confirmation.

## Scope

This effort is split into four independently reviewable projects:

1. [Hierarchical settings system](2026-07-16-settings-system-design.md)
2. [Experiment settings bundles](2026-07-16-experiment-settings-design.md)
3. [CLI and command modularization](2026-07-16-cli-modularization-design.md)
4. [Documentation information architecture](2026-07-16-documentation-design.md)

The implementation order is the same. The settings schema must exist before
experiments or legacy flags can target it, and behavior locks must exist before
the parser and handlers move.

## Non-goals

- Change inference math, performance policy, model compatibility, or defaults.
- Promote an experimental optimization into a product profile.
- Remove a legacy flag or environment alias in the first release of the new
  settings system.
- Rewrite model, Metal, MLX, server, or benchmark internals merely to reduce
  file length.
- Bundle bug fixes or performance changes into structural cleanup commits.
- Run hardware benchmarks solely to validate a configuration refactor.

## Architecture

```text
                           hard safety constraints
                                     |
command operands --> settings resolver --> immutable resolved settings
                         ^     ^    ^                  |
                         |     |    |                  +--> runtime consumers
                    user TOML  |  model defaults      +--> effective-config view
                               |
                  --set / --settings bundles
                               ^
                               |
               legacy CLI and MTPLX_* compatibility adapters
```

The schema defines names, types, visibility, lifecycle, aliases, validation,
redaction, and ownership. The resolver produces an immutable settings snapshot
plus provenance for every value. Runtime code consumes typed settings through
domain accessors. Direct environment reads are migrated by domain and prevented
for new settings by an automated source audit.

## Settings Versus Command Inputs

A value is a setting when it can reasonably be saved and reused across runs and
changes runtime policy or behavior. A value remains a command input when it
identifies the action's subject or result.

| Value | Classification | Example |
|---|---|---|
| Runtime profile | Setting | `runtime.profile=sustained` |
| Sampling temperature | Setting | `generation.temperature=0.6` |
| Session-cache budget | Setting | `cache.session.max_size=32GiB` |
| Benchmark prompt file | Command input | `--prompts prompts.jsonl` |
| Report output path | Command input | `--output result.json` |
| Machine-readable output | Command mechanic | `--json` |
| Confirmation | Command mechanic | `--yes` |

This boundary avoids replacing hundreds of flags with an untyped key-value
junk drawer.

## Compatibility Strategy

The first settings-native release will:

- Accept existing individual flags and `MTPLX_*` variables.
- Translate them into the same typed resolver used by native settings.
- Keep their current defaults and observable behavior.
- Exclude legacy configuration flags from primary help and examples while
  listing them in generated compatibility documentation.
- Diagnose mixed legacy/native conflicts with the winning source in
  `mtplx settings explain`.
- Preserve `mtplx settings get/set` as deprecated aliases for today's live
  daemon behavior; the unambiguous replacements are `settings live show/set`.

Removal of compatibility aliases requires a separate decision and release note.

## Failure-mode Check

### Registry becomes a second stale truth

Severity: critical.

Mitigation: every production `MTPLX_*` key must be registered or explicitly
identified as internal process state. New runtime configuration may not read the
environment directly. A source audit fails tests for unregistered keys and new
direct reads outside the shrinking compatibility boundary.

### Generic settings become less discoverable than flags

Severity: critical.

Mitigation: the schema drives `settings list`, `settings explain`, validation,
shell completion metadata, examples, and generated reference docs. Unknown keys
produce suggestions and a non-zero exit instead of being ignored.

### Settings silently change existing precedence or defaults

Severity: critical.

Mitigation: characterize current parser namespaces, profile application,
environment override behavior, config precedence, help, and errors before
introducing adapters. Legacy syntax retains those results. Settings-native
syntax has one documented precedence order and reports every winning source.

### Persistent and live mutation are confused

Severity: critical.

Mitigation: mutation always names its scope: `settings user set` writes the
user TOML, `settings live set` calls the running daemon, and `--set` is
process-local. The historical ambiguous syntax remains only as a deprecated
live-daemon alias.

### Experiment bundles become permanent folklore

Severity: minor if lifecycle checks are enforced; critical otherwise.

Mitigation: every built-in lab bundle records an owner, status, issue, creation
date, review date, supported models, and expected evidence. Rejected recipes are
archived as evidence rather than kept executable.

## Testing Strategy

Before structural changes:

- Keep the full-suite baseline green. The isolated worktree completed the suite
  at 100% with exit code 0 and four existing skips.
- Snapshot representative parsed namespaces for public, advanced, benchmark,
  streamed-expert, and compatibility commands.
- Characterize config/profile/environment precedence and live settings behavior.
- Capture compact, verbose, advanced, command, and flags help output.

During implementation:

- Add resolver tests before each production behavior slice.
- Move one parser or handler domain at a time and run its focused behavior lock.
- Run the full suite after each module-boundary move.
- Check generated settings documentation for drift.
- Scan separately for unregistered names, direct environment reads, old import
  paths, dynamic imports, string aliases, tests, and documentation references.

## Rollout

1. Land the schema, resolver, provenance model, and read-only settings commands.
2. Add persistent user settings, `--set`, and `--settings` without removing
   legacy syntax.
3. Convert product runtime controls to settings-native call sites by domain.
4. Add lab bundles and migrate active experiment launch commands.
5. Extract parser and handler domains behind compatibility imports.
6. Rewrite the README and publish generated settings and migration references.
7. After a release window, audit remaining legacy use and separately approve
   any removals.

## Acceptance Criteria

- Everyday README workflows require no individual runtime-configuration flags.
- Every active runtime setting has one schema entry and one typed resolution
  path.
- `settings show` and `settings explain` work without importing MLX.
- Persistent, per-run, bundled, live, legacy CLI, and legacy environment sources
  are distinguishable in provenance.
- Existing supported commands and benchmark scripts continue to parse during
  the compatibility period.
- Active experiments are reproducible settings bundles rather than shell lines
  containing many independent environment assignments.
- Generated reference docs match the schema.
- The full Python suite and targeted CLI/config tests pass.
- The primary checkout and its untracked benchmark reports remain unchanged.
