# Local Worktree Layout

## Scope

`$PRIMARY` denotes the retained main clone, and `$WORKSPACE` denotes its parent
directory. Keep `$PRIMARY` in place and relocate auxiliary worktrees for its Git
common directory to direct children of `$WORKSPACE/.worktrees/`.

The current inventory contains 38 registered worktrees: one retained primary
checkout, one actively owned exception, and 36 auxiliary worktrees to relocate.
No branch, commit, tracked change, or untracked artifact may be deleted by this
migration.

## Canonical Layout

The canonical root is one level above the main clone:

```text
$WORKSPACE/.worktrees/
```

Each worktree uses its existing basename directly beneath the canonical root.
Examples:

```text
mtplx-exec-opt
02-q8
29-cache-scheduling
mtplx-runtime-gate-pr17
```

The actively owned `$WORKSPACE/.worktrees/29-cache-scheduling` worktree already
uses the canonical layout. Another agent is using it, so this migration must not
inspect its contents recursively or change its path. No relocation begins if
any other registered worktree changes again before execution.

An initial pass placed 36 worktrees under `$PRIMARY/.worktrees/`. The user
clarified that the canonical root is one level up, like `29-cache-scheduling`.
The corrective pass moves those 36 worktrees to flat direct children of
`$WORKSPACE/.worktrees/`.

## Repository Rule

The default branch keeps a defensive `/.worktrees/` ignore rule to prevent
accidental in-clone containers. `CONTRIBUTING.md` defines the actual canonical
location as the workspace-level sibling directory `$WORKSPACE/.worktrees/`.

## Migration Contract

Before moving anything:

1. Acquire an atomic local migration lock.
2. Require no active `git worktree move`, benchmark, or repository process whose
   command line references an auxiliary worktree.
3. Snapshot each worktree's old path, destination, HEAD, branch or detached
   state, porcelain status hash including untracked files, and disk footprint.
4. Require all destinations to be absent and all worktrees to have no initialized
   submodules, locks, or prunable metadata.

Move one worktree at a time with `git worktree move`. Dirty worktrees are allowed
and must retain an identical porcelain status hash. After each move, verify its
HEAD, branch/detached state, status hash, and registration before continuing.
Stop on the first mismatch. Do not use `--force`.

After all moves, remove only empty directories beneath `$PRIMARY/.worktrees/`,
then remove that empty in-clone root. Preserve `$WORKSPACE/.worktrees/` and its
active `29-cache-scheduling` worktree. Do not remove a directory containing any
unregistered file.

## Verification

- Exactly 38 worktrees remain registered.
- The primary path and active `29-cache-scheduling` path and branch are
  unchanged; its owner may advance its HEAD. All 37 auxiliary worktrees are
  direct children of `$WORKSPACE/.worktrees/`.
- Every relocated worktree's pre-migration HEAD, branch/detached state, and
  status hash matches.
- The default work-off branch, GitHub default branch, and `origin/HEAD` remain
  `experiment/moe-pr13-pr14-stack` at the same commit.
- `$PRIMARY/.worktrees/` is absent after the corrective move.
- No former top-level registered worktree path remains.
- Qwen and the GPU lock are not modified by this filesystem-only migration.

## Rollback and Failure Handling

The preflight snapshot is the rollback map. If a move succeeds but its immediate
verification fails, move that worktree back to its recorded old path when the
old path is still absent, then stop. If an unrelated process changes the
worktree registry or occupies either path, stop without forcing or deleting
anything and report the partial state.

Moved per-worktree virtual environments may contain console scripts with old
absolute shebangs. Repository state remains valid, and `python -m ...` continues
to work when the interpreter itself is usable; recreate a moved worktree's
virtual environment before relying on its entry-point scripts. Rebuilding every
historical environment is outside this migration.
