# Documentation Information Architecture Design

Date: 2026-07-16
Parent: [Repository Settings Cleanup](2026-07-16-repository-settings-cleanup-design.md)

## Purpose

Make the root README answer how to install and use MTPLX, while detailed
settings, lab, operations, and compatibility material lives in focused,
testable references.

## Root README

The README becomes an outcome-first product entry point:

1. What MTPLX does and which Macs/models it supports.
2. Install the app or CLI.
3. Sixty-second start: `mtplx start`.
4. Configure normal behavior through `mtplx settings`.
5. Connect OpenAI/Anthropic clients.
6. Tune and benchmark.
7. Supported runtime modes.
8. Forge overview.
9. Advanced and lab pointers.
10. Compatibility, troubleshooting, contribution, license, and credit.

The detailed SSD-streamed MoE launch instructions move out of the product
README into a dedicated advanced guide. The README keeps a short capability
summary and link.

README examples use canonical settings-native syntax. Legacy individual
configuration flags do not appear in the normal path.

## Documentation Tree

```text
docs/
  README.md
  getting-started.md
  settings.md
  cli.md
  experiments.md
  migration-settings.md
  advanced/
    ssd-streamed-moe.md
  reference/
    settings.md          # generated and checked in
    settings.json        # generated machine-readable catalog
```

Existing specialized guides remain where they are unless the new index links
or redirects them. This project does not reorganize unrelated historical plans
and benchmark evidence.

## Content Contracts

### `docs/settings.md`

- Persistent user settings.
- One-run `--set` overrides.
- Settings bundles.
- Effective-value and provenance inspection.
- Persistent versus live scopes.
- Precedence and constraints.
- Secret-file handling.
- Common recipes by user goal.

### `docs/cli.md`

- Public, advanced, and lab command maps.
- Definition of settings versus command inputs.
- Minimal command-mechanics reference.
- Links to generated settings instead of duplicating setting defaults.

### `docs/experiments.md`

- Difference between product profiles and lab bundles.
- Discovering, inspecting, validating, and applying a bundle.
- Lifecycle and evidence rules.
- Reproducibility and safety expectations.

### `docs/migration-settings.md`

- Legacy flag and environment alias to canonical setting mappings.
- Scope changes for the existing live `settings get/set` command.
- Compatibility window and removal policy.
- Conflict and precedence examples.

## Generated Reference

The settings catalog generates Markdown and JSON references. Generated content
includes canonical name, type, default description, domain, visibility,
lifecycle, live/restart behavior, aliases, and deprecation replacement. Secret
values and internal-only operational details are never rendered.

Generation is deterministic. A `--check` mode exits non-zero when checked-in
artifacts differ from the catalog. Human guides link to generated data instead
of copying tables that can drift.

## Verification

- Execute or dry-run every root README command that does not require a model.
- Parse every shell block and reject obsolete individual runtime flags in the
  normal-use sections.
- Validate local links and generated anchors.
- Run the settings reference generator in check mode.
- Assert every public setting is documented and every compatibility alias is in
  the migration reference.
- Keep benchmark claims and hardware numbers unchanged unless separately
  re-measured.

## Error and Drift Policy

Documentation generation failures block completion. If a command cannot be
verified without hardware or a model, mark the prerequisite and validate its
parser path; do not claim a live run. Historical benchmark claims are preserved
or removed, never silently updated from unrelated measurements.

## Acceptance Criteria

- A new user can install, start, configure, and connect MTPLX from the README.
- Runtime configuration examples use hierarchical settings.
- Advanced streamed-MoE and experiment details no longer interrupt the normal
  product path.
- Public settings and compatibility aliases have complete generated coverage.
- All documented no-model commands are freshly verified.
