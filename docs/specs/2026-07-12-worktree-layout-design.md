# Local Worktree Layout

## Scope

`$PRIMARY` denotes the retained main clone, and `$WORKSPACE` denotes its parent
directory. Keep `$PRIMARY` in place and relocate inactive auxiliary worktrees
for its Git common directory beneath the main clone's ignored `.worktrees/`
directory.

The current inventory contains 38 registered worktrees: one retained primary
checkout, one actively owned exception, and 36 auxiliary worktrees to relocate.
No branch, commit, tracked change, or untracked artifact may be deleted by this
migration.

## Canonical Layout

The canonical root is:

```text
$PRIMARY/.worktrees/
```

Paths currently rooted directly beneath `$WORKSPACE` retain their relative
grouping beneath the canonical root. Examples:

```text
mtplx-exec-opt
mtplx-opt-prs/02-q8
mtplx-hy3-stack/29-cache-scheduling
mtplx-runtime-gate-pr17
```

Legacy worktrees under `$PRIMARY/.claude/worktrees/` move to:

```text
.worktrees/claude/<existing-name>
```

The actively owned
`$WORKSPACE/.worktrees/29-cache-scheduling` worktree
is an explicit temporary exception. Another agent is using it, so this migration
must not inspect its contents recursively, change its path, or remove its parent
directory. It moves to the canonical root only in a later operation after its
owner releases it. No other relocation begins if any registered worktree changes
again before execution.

## Repository Rule

The default branch adds `/.worktrees/` to `.gitignore` and documents in
`CONTRIBUTING.md` that auxiliary MTPLX worktrees belong under the primary
checkout's `.worktrees/` directory. The local shared Git exclude file also adds
`/.worktrees/` so the retained primary checkout ignores the directory before it
eventually receives the committed rule.

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

After all moves, remove only empty former grouping directories such as
`mtplx-opt-prs` and `mtplx-hy3-stack`. Preserve the workspace-root `.worktrees/`
directory and its active `29-cache-scheduling` worktree. Do not remove a
directory containing any unregistered file.

## Verification

- Exactly 38 worktrees remain registered.
- The primary path and active `29-cache-scheduling` exception path and branch
  are unchanged; its owner may advance its HEAD. The other 36 paths are beneath
  the primary checkout's `.worktrees/` directory.
- Every relocated worktree's pre-migration HEAD, branch/detached state, and
  status hash matches.
- The default work-off branch, GitHub default branch, and `origin/HEAD` remain
  `experiment/moe-pr13-pr14-stack` at the same commit.
- The primary checkout no longer reports `.worktrees/` as untracked.
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
