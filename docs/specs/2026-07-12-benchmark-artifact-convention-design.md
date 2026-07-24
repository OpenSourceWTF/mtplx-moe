# Benchmark Artifact Convention

## Scope

Standardize repository-local benchmark output so machine-generated artifacts are
kept out of Git while reviewed summaries remain easy to find and compare.

This change covers benchmark paths, run identifiers, repeat filenames, and
repository guidance. It does not delete historical benchmark summaries, add an
artifact service, or restrict tools from writing to explicitly requested paths.

## Layout

- Write machine-generated benchmark artifacts beneath
  `benchmarks/raw/<benchmark>/<run-id>/`.
- Ignore the complete `benchmarks/raw/` tree with one root `.gitignore` rule.
- Keep reviewed comparison tables and human-written summaries in
  `benchmarks/results/`.
- Treat the existing tracked JSON files in `benchmarks/results/` as legacy
  curated summaries. Do not relocate or delete them in this change.

## Naming Contract

Use this run identifier:

```text
<benchmark>-<variant>-<YYYYMMDDTHHMMSSZ>-<short-sha>
```

Names use lowercase ASCII letters, digits, and hyphens. The timestamp is UTC,
and `short-sha` is the tested commit's 7-12 character hexadecimal revision.
The variant identifies the comparison arm or material configuration difference,
such as `base`, `candidate`, or `candidate-cbank99`.

Within a run directory, use stable role and repeat names such as:

```text
base-r01.json
candidate-r01.json
response-r01.md
decode-trace.jsonl
```

Repeat numbers are one-based and zero-padded to two digits. A curated report
records the run identifier, tested revisions, exact command/configuration, and
required evidence fields rather than linking to ignored local files.

## Repository Guidance

The root `CONTRIBUTING.md` is authoritative for contributors and automation.
`benchmarks/results/README.md` mirrors the raw-versus-curated distinction and
provides a concrete path example. Current MoE benchmark instructions are updated
to write new raw outputs beneath `benchmarks/raw/`.

## Error Handling

The convention is enforced by Git ignore behavior and review guidance, not by
rejecting arbitrary command-line output paths. This preserves scratch, CI, and
external-storage workflows. Contributors must not force-add files from
`benchmarks/raw/`; selected measurements are promoted by summarizing them in a
reviewed result document.

## Verification

- `git check-ignore` must identify representative JSON, JSONL, and Markdown
  files beneath `benchmarks/raw/`.
- A representative curated Markdown file beneath `benchmarks/results/` must not
  be ignored.
- Repository documentation must use the same path and naming grammar.
- Existing tracked benchmark summaries must remain tracked.
- The normal Python test, Ruff, compile, and staged-diff checks remain required
  before publishing the integration branch.

## Failure Modes and Boundaries

- Existing tracked files are unaffected by `.gitignore`; they are deliberately
  grandfathered as legacy summaries.
- Raw output written elsewhere can still be accidentally staged; contributor
  guidance makes `benchmarks/raw/` the required repository-local destination.
- Ignored raw files cannot serve as durable report links; curated reports record
  reproducible metadata and may reference an external attachment when raw data
  must be retained.
