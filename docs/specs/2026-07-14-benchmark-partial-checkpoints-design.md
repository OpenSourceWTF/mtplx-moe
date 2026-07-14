# Benchmark partial checkpoints

## Status

Approved in conversation on 2026-07-14. A failed benchmark must retain every
completed measurement instead of discarding the entire in-memory payload.

## Scope

Change only the Q2/BF16 MTP depth-matrix harness and its focused tests. The
harness will atomically checkpoint completed retained rows and will preserve a
machine-readable failure record. It will continue to fail closed: partial
evidence is never a passing campaign and the CLI still returns nonzero.

This change does not weaken token-parity, artifact, memory, or metric gates. It
does not resume an interrupted model automatically, change benchmark ordering,
or alter model/runtime behavior.

## Alternatives

1. **Atomic snapshot after each retained row (selected).** Rewrite the small
   campaign JSON through a temporary file and `os.replace`. This preserves the
   existing single-JSON consumer contract and guarantees that readers see the
   previous or new complete checkpoint.
2. **Append-only JSONL event journal.** This is naturally incremental, but it
   introduces a second schema and requires a compaction/replay step before
   existing analysis can consume a campaign.
3. **One file per cell plus a manifest.** This isolates writes well, but adds
   fragment cleanup, discovery, and assembly complexity that is unnecessary
   for the current matrix size.

## Data contract

The output path is initialized before model allocation with:

- the existing schema identifier and immutable campaign configuration;
- `status: "running"`;
- `passed: false`;
- an empty `models` list;
- `active_cell`, initially the setup phase;
- `failure: null`.

After every completed retained observation, the harness atomically rewrites the
snapshot. Warm-up observations remain discarded and are not checkpointed as
measurements. A model currently in progress is represented with
`passed: false`; only a fully completed model may set its own `passed: true`.

On a caught exception, the harness atomically writes:

- `status: "failed"` and `passed: false`;
- every previously completed retained observation;
- the last `active_cell`, including model, context, depth/cell, phase, and
  whether the failure occurred during warm-up or retained measurement;
- `failure.error`, `failure.error_type`, and `failure.active_cell`.

If runtime cleanup also fails while another exception is active, the original
benchmark exception remains authoritative and the close failure is appended to
`cleanup_errors`. A close failure with no earlier error is attributed to the
model's `close` phase.

The CLI then prints the same failure payload and returns 1. On complete success,
the final atomic write sets `status: "passed"`, `passed: true`, clears
`active_cell`, and keeps the existing final result fields.

If no output path is supplied, library behavior remains in-memory and the CLI
still emits its final success or failure JSON to stdout.

## Architecture

`run_depth_matrix` gains an optional checkpoint callback. It owns the mutable
campaign state and calls the callback only at stable boundaries: initial setup,
after a retained row, after a completed model, and after final success. The CLI
binds that callback to the existing atomic JSON writer when `--output-json` is
present.

The runner updates `active_cell` immediately before model load, prompt build,
warm-up, or retained generation. Its exception path adds failure metadata to
the latest state and invokes the same callback before re-raising. This keeps
the reusable runner independent from filesystem policy while ensuring the CLI
can persist partial evidence.

## Error handling

- Checkpoint serialization or write failure is fatal; the benchmark must not
  continue while claiming recoverable evidence.
- Only fully validated retained rows enter `observations`. A row that throws or
  fails a gate is represented by `active_cell` and `failure`, never as a valid
  observation.
- Atomic replacement keeps the last complete checkpoint if the process dies
  during a rewrite. A SIGKILL or power loss may leave `status: "running"`; that
  explicitly means interrupted, not passed.
- Failure while loading or building a prompt still produces a failure artifact
  with zero observations and the relevant setup phase.

## Testing strategy

Write failing tests before production changes for:

1. AR completes, D1 parity fails, and the output retains the AR row plus the D1
   active-cell failure while returning 1.
2. A model-load failure writes a failed artifact with zero observations.
3. A successful run checkpoints rows and ends with the unchanged success data
   plus `status: "passed"`.
4. Warm-up rows never appear as retained observations.
5. A checkpoint callback failure aborts the run.
6. Every filesystem checkpoint parses as complete JSON; no temporary file is
   left after success or a caught failure.

## Failure-mode check

### Critical: partial rows are mistaken for a valid campaign

Mitigation: `passed` remains false throughout execution and failure, model
payloads remain false until complete, and only the terminal success checkpoint
may set campaign `passed: true`.

### Critical: the failing row is accidentally retained as valid evidence

Mitigation: append a row only after all row gates pass. Record failures only in
`active_cell`/`failure`.

### Critical: checkpointing corrupts the sole result file

Mitigation: serialize with `allow_nan=False`, fsync the temporary file, and
atomically replace the target. A write failure aborts the campaign.

### Minor: checkpoint I/O perturbs throughput

Checkpointing occurs after synchronization and timing extraction, outside the
measured observation interval. The payload is small and rewritten only for
retained rows, not warm-ups.

## Rollout

Land the harness behavior with focused fake-runtime tests before another MLX
window. The next Hy3 diagnostic/campaign must use an output path so its AR row
and any later failure survive. Existing successful-result consumers may ignore
the additive `status`, `active_cell`, and `failure` fields.
