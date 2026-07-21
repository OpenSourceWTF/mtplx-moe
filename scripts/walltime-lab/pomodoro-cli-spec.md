# Pomodoro CLI — Design Document

Build a zero-dependency Node.js pomodoro timer CLI. Follow this spec exactly.
Create every file listed. When all files exist, run the tests with
`node --test tests/` and fix failures until they pass.

## Files

1. `js/config.js` — Defaults and validation.
   - `DEFAULTS = { work: 25, shortBreak: 5, longBreak: 15, cyclesPerLong: 4 }`
     (minutes, integers).
   - `loadConfig(argvOverrides)` merges CLI overrides onto defaults; throws
     `RangeError` for non-positive or non-integer values.
2. `js/state.js` — Pure session state machine.
   - `createSession(config)` returns `{ phase: 'work', cycle: 1, completed: 0,
     remainingSec, config }`.
   - `advance(state)` transitions work→shortBreak, work→longBreak every
     `cyclesPerLong`-th cycle, break→work (incrementing `cycle` after a
     break), and increments `completed` after each finished work phase.
   - `tick(state, seconds)` decrements `remainingSec`, clamping at 0; when it
     hits 0 the caller advances. Pure functions only — no timers, no I/O.
3. `js/format.js` — Rendering helpers.
   - `formatClock(seconds)` → `MM:SS` (zero-padded).
   - `formatStatus(state)` → e.g. `[work 2/4] 24:59 (3 done)`.
   - `progressBar(state, width=20)` → `[#####-------------]` proportional to
     phase elapsed time.
4. `js/stats.js` — Persistence.
   - `recordCompletion(path, isoDate)` appends a JSON line
     `{ date, completedAt }` to the file (creates it if missing).
   - `readStats(path)` returns `{ totalSessions, byDate }` aggregated from the
     file; missing file → zeros.
5. `js/cli.js` — Argument parsing + main loop wiring.
   - Parse `--work N --short N --long N --cycles N --stats-file PATH --once`.
   - `--once` runs a single simulated work phase at 60x speed (1 real second
     per simulated minute) then exits — used for manual smoke runs.
   - Exports `parseArgs(argv)` (pure) and `main()`; `main` prints status lines
     with `process.stdout.write('\r' + …)`.
6. `bin/pomodoro.js` — `#!/usr/bin/env node`, imports `main` from
   `../js/cli.js` and runs it.
7. `tests/state.test.mjs` — `node:test` coverage for `createSession`,
   `advance` through two full long-break cycles, and `tick` clamping.
8. `tests/format-stats.test.mjs` — clock/status/bar formatting cases and
   stats round-trip through a temp file (`fs.mkdtempSync`).

## Conventions

- ES modules throughout (`"type": "module"` in a minimal `package.json`).
- No third-party packages. Node 20+.
- Every exported function gets a one-line JSDoc comment.

## Acceptance

- `node --test tests/` exits 0.
- `node bin/pomodoro.js --once --work 1` finishes in ~1s of simulated phase
  and prints a final `done` line.
