# Experiment Settings Bundles Design

Date: 2026-07-16
Parent: [Repository Settings Cleanup](2026-07-16-repository-settings-cleanup-design.md)

## Purpose

Replace long experiment launch lines made of independent environment variables
and flags with validated, attributable settings bundles.

Product profiles remain product policy. Lab bundles are opt-in hypotheses and
must never silently alter the default product path.

## User Experience

```bash
mtplx lab list
mtplx lab show compiled-verify-control
mtplx lab validate ./experiments/my-candidate.toml

mtplx bench prefill-ladder \
  --settings lab:compiled-verify-control \
  --prompts prompts.jsonl \
  --output result.json
```

The `lab:` URI is explicit opt-in to a packaged experimental bundle. A normal
path loads a user-provided settings bundle. Both use the same settings resolver.

## Bundle Contract

Built-in lab bundles are package data and use TOML:

```toml
[experiment]
id = "compiled-verify-control"
title = "Compiled verify disabled control"
status = "active"
owner = "runtime"
tracking = "docs/experiments/compiled-verify-control.md"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Isolate compiled verify from the sustained baseline."

[settings]
"verify.compiled.mode" = "off"
```

Required metadata is `id`, `title`, `status`, `owner`, `tracking`, `created`,
`review_after`, `models`, and `purpose`. Active built-ins may use only settings
whose schema visibility is `experimental` or `advanced`, plus an explicitly
listed product baseline profile.

## Lifecycle

| Status | Executable | Meaning |
|---|---:|---|
| `active` | Yes | Currently gathering evidence |
| `retained` | No | Graduated into a product profile or supported setting |
| `rejected` | No | Evidence failed the gate; retained as documentation only |
| `superseded` | No | Replaced by a named newer bundle |
| `expired` | No | Review date passed without renewal |

`lab list` shows executable bundles by default and can show archived metadata
with `--all`. Loading a non-executable bundle fails and points to its result or
replacement.

Promotion moves the proven behavior into a product profile and archives the
bundle. Rejection removes the executable settings but keeps the hypothesis,
result link, and conclusion. No rejected experiment remains as a convenient
runtime toggle.

## Reproducibility

Every run using a bundle records:

- Bundle id, normalized content hash, and source path or package version.
- Fully resolved settings with secrets redacted.
- Value provenance and any compatibility aliases involved.
- Git commit, model identity, profile, command inputs, and output artifact.
- Existing benchmark contamination and hardware-lane metadata where relevant.

Bundles cannot contain shell commands, Python expressions, environment
expansion, file writes, or arbitrary imports. They are data only.

## Organization

```text
mtplx/experiments/
  catalog.py
  schema.py
  recipes/
    compiled-verify-control.toml
docs/experiments/
  README.md
  archive/
    compiled-verify-control.md
```

Issue-specific raw benchmark artifacts stay in the existing ignored results
locations. The package contains only small active recipes and metadata.

## Compatibility Migration

Existing benchmark scripts continue to work. Migration replaces clusters of
environment assignments with one settings bundle at a time. The old launch
form and the bundle form must resolve to identical effective settings before a
script switches.

An experiment key used by no active bundle, product profile, compatibility
test, or documented external contract becomes a retirement candidate. Removal
is a separate behavior change with its own tests and approval.

## Error Handling

- Missing required metadata: validation error before runtime imports.
- Unknown or non-experimental key: validation error naming the setting tier.
- Expired/rejected/superseded built-in: refuse execution and show status.
- Model mismatch: fail before load with supported model families.
- Bundle conflict: use standard resolver precedence and show both sources.
- Hash mismatch for a recorded built-in: fail provenance validation.

## Tests

- TOML schema and lifecycle validation.
- Data-only security boundary.
- Bundle-to-resolved-settings equivalence with migrated legacy commands.
- Expiry, model compatibility, and archived-status refusal.
- Stable normalized hashes.
- Provenance inclusion in benchmark envelopes.
- Package-data installation smoke test.

## Acceptance Criteria

- Active experiments launch through named bundles instead of copied clusters of
  raw environment variables.
- Product profiles contain no unreviewed experiment defaults.
- Every executable built-in has ownership and an evidence lifecycle.
- Rejected and superseded work remains discoverable but cannot accidentally run.
- Benchmark output identifies the exact bundle and resolved settings.
