# Archived: 2026-05 prefill-ladder results

94 result files from the 2026-05 prefill-ladder era, moved here 2026-07-18 to
get `benchmarks/results/` down to records that can actually be interpreted.

**These are real measurements whose invocations are unrecoverable.** They all
carry `"kind": "prefill_ladder"` and were produced by
`mtplx/prefill_bench.py` via `mtplx bench prefill-ladder` — but the payload
schema of that era has no `command` or `argv` field, so the mapping from
filename to the flags that produced it is lost. The filename variety
(`-chunk4096`, `-cleanup2`, `-history16k`, `-no-defer`, `-profile-default`,
…) encodes CLI-flag variety that is no longer resolvable to specific
invocations.

Kept rather than deleted because the numbers are genuine and cannot be
reproduced. Do not cite them for a performance claim: without the invocation
you cannot establish what was being compared.

Two files from this era stayed in `benchmarks/results/` because docs still
reference them:

- `v0.2.0-release-m5max-32k-64k-128k.json` — cited by `RELEASE_NOTES_v0.2.0.md`
- `prefill-fixed-m5max-local-16k-32k-coherent-tail.json`

## Going forward

`CONTRIBUTING.md`, `project-map.md`, and `benchmarks/results/README.md` all
already define the intended layout, which nothing had adopted:

- raw output → `benchmarks/raw/<benchmark>/<run-id>/` (gitignored)
- curated summaries → `benchmarks/results/`
- run-id grammar → `<benchmark>-<variant>-<YYYYMMDDTHHMMSSZ>-<short-sha>`

The `issue69-hy3-mtp-*-r<N>.json` family is the best-provenanced set in the
repo — it embeds the full `command` array. Copy that pattern.
