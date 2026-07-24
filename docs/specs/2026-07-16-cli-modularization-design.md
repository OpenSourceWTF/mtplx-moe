# CLI and Command Modularization Design

Date: 2026-07-16
Parent: [Repository Settings Cleanup](2026-07-16-repository-settings-cleanup-design.md)

## Purpose

Make the CLI and command handlers understandable in bounded modules without
changing existing command behavior. New settings-native behavior is developed
with tests first; file movement is a separate behavior-preserving sequence.

## Stable Contracts

- Console entry points remain `mtplx.cli:main` and `mtplx.cli:main_tune`.
- `mtplx.cli.build_parser` remains importable during the compatibility period.
- Existing command names, exit codes, parse results, defaults, aliases, and
  machine-readable output remain unchanged unless the settings design explicitly
  introduces a new path.
- Existing imports from `mtplx.commands.public` continue through re-exports
  until callers migrate.
- Lazy runtime imports remain lazy so help and settings inspection need no MLX.

## Target Structure

`mtplx/cli.py` becomes a thin compatibility entry point. New code lives under a
non-conflicting package name:

```text
mtplx/cli_app/
  main.py
  help.py
  parsing.py
  settings_args.py
  groups/
    product.py
    server.py
    models.py
    integrations.py
    support.py
    benchmarks.py
    lab.py

mtplx/commands/
  runtime.py
  server.py
  models.py
  integrations.py
  support.py
  benchmarks.py
  lab.py
  public.py        # temporary compatibility re-exports
  forge.py
```

Argparse registration stays imperative and grouped by domain. The project does
not introduce a general command framework or encode handler logic in the
settings catalog.

## Flag Consolidation

Runtime parser groups share only two settings-native options:

```text
--set KEY=VALUE
--settings PATH_OR_LAB_URI
```

Legacy configuration options are registered by a compatibility module so they
can be hidden from primary help, tested as a single boundary, and removed later
without editing every command group.

Command operands and mechanics remain explicit. Parser-group review classifies
each current flag as:

- canonical setting alias;
- command operand;
- output/control mechanic;
- compatibility alias;
- retired no-op candidate.

No flag is removed merely because its name looks experimental.

## Extraction Sequence

1. Capture parser, help, dispatch, and config-precedence behavior locks.
2. Extract help formatting with byte-for-byte output tests.
3. Extract shared parsing and settings arguments.
4. Extract one parser domain while retaining `build_parser` composition.
5. Run focused and full tests before the next parser domain.
6. Split one handler domain from `commands/public.py`, add compatibility
   re-exports, and audit imports.
7. Repeat handler extraction by domain.
8. Remove a compatibility re-export only after a separate reference audit.

An extraction commit performs one structural move. New settings behavior and
module movement do not share a commit.

## Behavior Locks

Tests cover:

- Compact, verbose, advanced, command-specific, and all-flags help.
- Representative namespaces for product, server, benchmark, QA, Forge,
  streamed-expert, and diagnostic commands.
- Unknown commands, invalid choices, malformed values, and exit codes.
- `_cli_flags` explicit-input recording.
- User config and profile precedence.
- The old `settings get/set` live-daemon path.
- No-MLX import behavior.
- Public handler import paths used by tests and external scripts.

Snapshot tests normalize only inherently variable content such as version and
temporary paths. They do not normalize option names, defaults, ordering, or
help text.

## Dependency Rules

- Parser modules may import schema/type helpers and lightweight constants.
- Parser modules may not import MLX, model implementations, the FastAPI server,
  or benchmark runners.
- Handler modules may depend on runtime domains; runtime domains never import
  parser modules.
- Cross-domain handler calls use a small explicit service function rather than
  importing another parser group.
- Compatibility modules may depend on new modules, never the reverse.

## Error Handling

Structural moves preserve existing errors. New settings parse errors use the
settings system's exit-2 contract. Import-cycle or optional-dependency failures
must be caught by no-MLX help/settings smoke tests before a move lands.

## Verification

After each extraction:

- Run the focused CLI/handler tests.
- Run Ruff on changed files.
- Run the full Python suite.
- Search separately for old imports, string import paths, re-exports, mocks,
  documentation links, and handler-name references.
- Check that the primary help imports without MLX.

## Acceptance Criteria

- Root parser and public handler files become compatibility/composition modules,
  not implementation warehouses.
- Each domain can be understood and tested independently.
- Settings-native runtime commands expose two generic configuration options.
- Existing scripts keep parsing during the compatibility period.
- The full suite remains green after every structural move.
