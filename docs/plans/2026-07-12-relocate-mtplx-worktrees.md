# Relocate MTPLX Worktrees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the primary clone in place and place all auxiliary MTPLX worktrees as flat direct children of the workspace-level `.worktrees/` directory without changing any branch, commit, dirty state, untracked artifact, Qwen process, or GPU lock.

**Architecture:** The primary checkout remains `$PRIMARY`; the canonical auxiliary root is `$WORKSPACE/.worktrees/`, one level above it. A one-shot locked migration snapshots every inactive worktree, moves each with `git worktree move`, and verifies its identity and porcelain-status hash immediately. The actively owned `$WORKSPACE/.worktrees/29-cache-scheduling` worktree is already canonical and excluded.

**Tech Stack:** Git worktrees, zsh, GitHub CLI, Markdown.

**Assumptions:** Assumes the registry still contains 38 worktrees — abort if it changes. Assumes `29-cache-scheduling` remains actively owned — this plan will not move, inspect recursively, or remove it. Assumes no initialized submodules, locked worktrees, or prunable entries among the other 36 — abort rather than force if any appear. Assumes moved virtual-environment entry-point scripts may retain old shebangs — this plan does not rebuild historical environments.

**Path variables:** Resolve the Git common directory from any registered
worktree, set `$PRIMARY` to the main clone containing that common `.git`
directory, and set `$WORKSPACE` to its parent. No committed document depends on
a user-specific absolute path.

---

## File Structure

- `.gitignore`: permanently ignores `/.worktrees/` on branches containing the rule.
- `CONTRIBUTING.md`: defines the repository-level auxiliary-worktree placement and ownership rule.
- `.git/info/exclude`: applies `/.worktrees/` immediately to the retained primary checkout without changing its older branch.
- `docs/specs/2026-07-12-worktree-layout-design.md`: approved migration contract.
- `docs/plans/2026-07-12-relocate-mtplx-worktrees.md`: executable migration and verification record.
- `/tmp/mtplx-relocate-worktrees.zsh`: one-shot local migration driver, not committed.
- `/tmp/mtplx-worktree-relocation-<timestamp>.tsv`: generated rollback/evidence snapshot, not committed.

## Task 1: Add the durable and local placement rules

**Files:**
- Modify: `.gitignore`
- Modify: `CONTRIBUTING.md`
- Modify: `.git/info/exclude`
- Stage: `docs/specs/2026-07-12-worktree-layout-design.md`
- Stage: `docs/plans/2026-07-12-relocate-mtplx-worktrees.md`

**Security flag:** `none`

- [x] **Step 1: Verify the rule is absent**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
! rg -n '^/\.worktrees/$' .gitignore
! rg -n '^/\.worktrees/$' $PRIMARY/.git/info/exclude
```

Expected: neither repository nor local shared rule exists before the change.

- [x] **Step 2: Add the repository ignore rule**

Add this block beneath generated artifacts in `.gitignore`:

```gitignore
# Local auxiliary Git worktrees
/.worktrees/
```

- [x] **Step 3: Add the contributor rule**

Append this section to `CONTRIBUTING.md`:

```markdown
## Local worktrees

Keep auxiliary worktrees as direct children of the workspace-level
`.worktrees/` directory beside your main clone. Do not create worktree
directories inside the clone or scatter them across the workspace root. Never
move a worktree while another process or agent owns it; active worktrees stay in
place until their owner releases them.
```

- [x] **Step 4: Add the shared local exclude**

Use `apply_patch` to add this exact line to
`$PRIMARY/.git/info/exclude`:

```gitignore
/.worktrees/
```

- [x] **Step 5: Verify, commit, and push the rule before moving paths**

```bash
git check-ignore -v .worktrees/example/file
git diff --check
git add .gitignore CONTRIBUTING.md \
  docs/specs/2026-07-12-worktree-layout-design.md \
  docs/plans/2026-07-12-relocate-mtplx-worktrees.md
git commit -m "chore: standardize local worktree layout"
git push
```

Expected: the committed rule and approved documents are pushed on `experiment/moe-pr13-pr14-stack`; the common exclude remains local and uncommitted.

## Task 2: Preflight and snapshot all inactive worktrees

**Files:**
- Create: `/tmp/mtplx-relocate-worktrees.zsh`
- Generate: `/tmp/mtplx-worktree-relocation-<timestamp>.tsv`

**Security flag:** `none`

**Does NOT cover:** The active `29-cache-scheduling` exception and the retained primary checkout are recorded but never moved.

- [x] **Step 1: Require a stable registry and no competing migration**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
WORKSPACE="$(dirname "$PRIMARY")"
ACTIVE=$WORKSPACE/.worktrees/29-cache-scheduling
test "$(git -C "$PRIMARY" worktree list --porcelain | rg -c '^worktree ')" = 38
test -d "$ACTIVE"
test "$(git -C "$ACTIVE" rev-parse --show-toplevel)" = "$ACTIVE"
! git -C "$PRIMARY" worktree list --porcelain | rg '^(locked|prunable)'
! ps -axo command= | rg 'git .*worktree (move|remove|add)'
test ! -e /tmp/mtplx-worktree-relocation.lock
```

Expected: 38 stable registrations, active exception present, no locks/prunable entries or competing worktree mutation, and the migration lock available. The active owner may advance its HEAD without invalidating this migration.

- [x] **Step 2: Create the one-shot driver with `apply_patch`**

Create `/tmp/mtplx-relocate-worktrees.zsh` with `apply_patch` using this complete
content:

```zsh
#!/bin/zsh
set -euo pipefail

MODE="${1:---dry-run}"
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
WORKSPACE="${PRIMARY:h}"
ACTIVE=$WORKSPACE/.worktrees/29-cache-scheduling
DEST_ROOT=$WORKSPACE/.worktrees
INTEGRATION_OLD=$PRIMARY/.worktrees/mtplx-experimental-pr13-pr14-main
LOCK=/tmp/mtplx-worktree-relocation.lock
SNAPSHOT=/tmp/mtplx-worktree-relocation-$(date -u +%Y%m%dT%H%M%SZ).tsv

if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  print -u2 "usage: $0 --dry-run|--execute"
  exit 2
fi

if ! mkdir "$LOCK" 2>/dev/null; then
  print -u2 "migration lock is held: $LOCK"
  exit 1
fi
cleanup() {
  rmdir "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

status_hash() {
  git -C "$1" status --porcelain=v1 -z --untracked-files=all |
    shasum -a 256 | awk '{print $1}'
}

branch_state() {
  git -C "$1" symbolic-ref --quiet --short HEAD 2>/dev/null || print DETACHED
}

destination_for() {
  local old="$1" rel
  case "$old" in
    "$PRIMARY/.worktrees/"*)
      rel="$(basename "$old")"
      ;;
    *)
      print -u2 "unmapped worktree: $old"
      return 1
      ;;
  esac
  print "$DEST_ROOT/$rel"
}

typeset -a registry old_paths destinations heads branches hashes sizes order
typeset -A seen_destinations
registry=("${(@f)$(git -C "$PRIMARY" worktree list --porcelain |
  awk '/^worktree /{print substr($0,10)}')}")

if (( ${#registry[@]} != 38 )); then
  print -u2 "registry changed: expected 38, found ${#registry[@]}"
  exit 1
fi
if [[ "${registry[(Ie)$PRIMARY]}" -eq 0 || "${registry[(Ie)$ACTIVE]}" -eq 0 ]]; then
  print -u2 "primary or active exception is not registered"
  exit 1
fi
if git -C "$PRIMARY" worktree list --porcelain | grep -Eq '^(locked|prunable)'; then
  print -u2 "locked or prunable worktree found"
  exit 1
fi

printf 'path\thead\tbranch_state\n%s\t%s\t%s\n' \
  "$ACTIVE" "$(git -C "$ACTIVE" rev-parse HEAD)" "$(branch_state "$ACTIVE")" \
  > "$SNAPSHOT.active"

process_snapshot="$(ps -axo pid=,command=)"
for old in "${registry[@]}"; do
  [[ "$old" == "$PRIMARY" || "$old" == "$ACTIVE" ]] && continue
  if print -r -- "$process_snapshot" | grep -F -- "$old" >/dev/null; then
    print -u2 "active process references worktree: $old"
    exit 1
  fi
  if ! submodule_state="$(git -C "$old" submodule status 2>&1)"; then
    print -u2 "could not inspect submodules for $old: $submodule_state"
    exit 1
  fi
  if [[ -n "$submodule_state" ]]; then
    print -u2 "initialized submodule found: $old"
    exit 1
  fi
  destination="$(destination_for "$old")"
  if [[ -e "$destination" || -n "${seen_destinations[$destination]-}" ]]; then
    print -u2 "occupied or duplicate destination: $destination"
    exit 1
  fi
  seen_destinations[$destination]=1
  old_paths+=("$old")
  destinations+=("$destination")
  heads+=("$(git -C "$old" rev-parse HEAD)")
  branches+=("$(branch_state "$old")")
  hashes+=("$(status_hash "$old")")
  sizes+=("$(du -sk "$old" | awk '{print $1}')")
done

if (( ${#old_paths[@]} != 36 || ${#seen_destinations[@]} != 36 )); then
  print -u2 "move-set mismatch: paths=${#old_paths[@]} destinations=${#seen_destinations[@]}"
  exit 1
fi

printf 'old_path\tdestination\thead\tbranch_state\tstatus_sha256\tdisk_kib\n' > "$SNAPSHOT"
for (( i=1; i<=${#old_paths[@]}; i++ )); do
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${old_paths[$i]}" "${destinations[$i]}" "${heads[$i]}" \
    "${branches[$i]}" "${hashes[$i]}" "${sizes[$i]}" >> "$SNAPSHOT"
  printf '%s -> %s\n' "${old_paths[$i]}" "${destinations[$i]}"
done
print "snapshot=$SNAPSHOT"

if [[ "$MODE" == "--dry-run" ]]; then
  print "dry-run verified 36 moves; no paths changed"
  exit 0
fi

for (( i=1; i<=${#old_paths[@]}; i++ )); do
  [[ "${old_paths[$i]}" == "$INTEGRATION_OLD" ]] || order+=("$i")
done
for (( i=1; i<=${#old_paths[@]}; i++ )); do
  [[ "${old_paths[$i]}" == "$INTEGRATION_OLD" ]] && order+=("$i")
done

for i in "${order[@]}"; do
  old="${old_paths[$i]}"
  destination="${destinations[$i]}"
  mkdir -p "$(dirname "$destination")"
  git -C "$PRIMARY" worktree move "$old" "$destination"
  if [[ -e "$old" ]] ||
     [[ "$(git -C "$destination" rev-parse HEAD)" != "${heads[$i]}" ]] ||
     [[ "$(branch_state "$destination")" != "${branches[$i]}" ]] ||
     [[ "$(status_hash "$destination")" != "${hashes[$i]}" ]]; then
    print -u2 "verification failed after move: $old -> $destination"
    if [[ ! -e "$old" ]]; then
      mkdir -p "$(dirname "$old")"
      git -C "$PRIMARY" worktree move "$destination" "$old" || true
    fi
    exit 1
  fi
  printf 'verified\t%s\t%s\n' "$old" "$destination" >> "$SNAPSHOT.moved"
done

print "execute verified 36 moves"
```

- [x] **Step 3: Validate the driver contract**

```bash
zsh -n /tmp/mtplx-relocate-worktrees.zsh
rg -n '!= 38|!= 36|old_paths|status_hash|submodule status|worktree move|INTEGRATION_OLD' \
  /tmp/mtplx-relocate-worktrees.zsh
```

Expected: syntax passes and the driver contains the registry, count, status,
submodule, move, and move-integration-last guards.

- [x] **Step 4: Dry-run the mapping**

```bash
zsh /tmp/mtplx-relocate-worktrees.zsh --dry-run
```

Expected: exit zero, 36 collision-free old-to-new mappings printed, snapshot written, no paths moved, and the registry unchanged. The active exception sidecar is observational; its owner may advance HEAD later.

## Task 3: Move and immediately verify 36 worktrees

**Files:**
- Move: 36 registered worktree directories

**Security flag:** `none`

**Does NOT cover:** The primary checkout and active `29-cache-scheduling` exception remain at their original paths.

- [x] **Step 1: Revalidate the dry-run snapshot immediately before mutation**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
test "$(git -C "$PRIMARY" \
  worktree list --porcelain | rg -c '^worktree ')" = 38
! ps -axo command= | rg 'git .*worktree (move|remove|add)'
```

Expected: registry and process admission still match the dry run.

- [x] **Step 2: Execute each move from the retained primary checkout**

For each snapshot row, the driver must run:

```bash
mkdir -p "$(dirname "$destination")"
git -C "$PRIMARY" worktree move "$old_path" "$destination"
```

Move `mtplx-experimental-pr13-pr14-main` last. Never use `--force`.

- [x] **Step 3: Verify immediately after every move**

Require the destination's HEAD, branch/detached state, and status hash to equal
the snapshot and require the old path to be absent. If any check fails and the
old path is free, move that single worktree back and stop. Do not continue after
a mismatch.

- [x] **Step 4: Execute the migration**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
cd "$PRIMARY"
zsh /tmp/mtplx-relocate-worktrees.zsh --execute
```

Expected: 36 moves and 36 immediate verification passes; primary and active exception paths untouched. The active owner may continue committing.

- [x] **Step 5: Remove the empty in-clone worktree root**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
find "$PRIMARY/.worktrees" -depth -type d -empty -exec rmdir {} \;
test ! -e "$PRIMARY/.worktrees"
```

Expected: the now-empty in-clone container disappears. Do not remove `$WORKSPACE/.worktrees/`.

## Task 4: Verify final local and remote state

**Files:**
- Test: Git worktree registry, dirty-state snapshot, default branch, CI, Qwen/lock state

**Security flag:** `none`

- [x] **Step 1: Verify registry placement and identity**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
WORKSPACE="$(dirname "$PRIMARY")"
ACTIVE=$WORKSPACE/.worktrees/29-cache-scheduling
test "$(git -C "$PRIMARY" worktree list --porcelain | rg -c '^worktree ')" = 38
test "$(git -C "$PRIMARY" worktree list --porcelain |
  rg -c "^worktree $WORKSPACE/\.worktrees/[^/]+$")" = 37
test "$(git -C "$PRIMARY" worktree list --porcelain |
  rg -c "^worktree $PRIMARY/\.worktrees/")" = 0
```

Expected: one primary and 37 flat canonical auxiliary worktrees, including the active worktree.

- [x] **Step 2: Verify all snapshot identities and dirty-state hashes**

```zsh
set -euo pipefail
SNAPSHOT="$(ls -t /tmp/mtplx-worktree-relocation-*.tsv | head -1)"
status_hash() {
  git -C "$1" status --porcelain=v1 -z --untracked-files=all |
    shasum -a 256 | awk '{print $1}'
}
branch_state() {
  git -C "$1" symbolic-ref --quiet --short HEAD 2>/dev/null || print DETACHED
}

verified=0
while IFS=$'\t' read -r old destination head branch hash disk_kib; do
  [[ "$old" == old_path ]] && continue
  [[ ! -e "$old" ]]
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$head" ]]
  [[ "$(branch_state "$destination")" == "$branch" ]]
  [[ "$(status_hash "$destination")" == "$hash" ]]
  (( ++verified ))
done < "$SNAPSHOT"
(( verified == 36 ))

IFS=$'\t' read -r active_path active_head_before active_branch \
  < <(tail -n 1 "$SNAPSHOT.active")
[[ "$(branch_state "$active_path")" == "$active_branch" ]]
active_head_after="$(git -C "$active_path" rev-parse HEAD)"
printf 'active exception HEAD before=%s after=%s (owner may advance)\n' \
  "$active_head_before" "$active_head_after"
```

Expected: all 36 destination identities and status hashes match, every old path
is absent, and the active exception's registered path and branch are unchanged
without traversing its contents. Its HEAD may advance under its owner.

- [x] **Step 3: Verify repository/default-branch state**

```bash
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
WORKSPACE="$(dirname "$PRIMARY")"
NEW_ACTIVE=$WORKSPACE/.worktrees/mtplx-experimental-pr13-pr14-main
test -z "$(git -C "$NEW_ACTIVE" status --porcelain)"
test "$(gh repo view davidtai/MTPLX --json defaultBranchRef --jq '.defaultBranchRef.name')" = \
  experiment/moe-pr13-pr14-stack
test "$(git -C "$NEW_ACTIVE" symbolic-ref --short refs/remotes/origin/HEAD)" = \
  origin/experiment/moe-pr13-pr14-stack
test "$(git -C "$NEW_ACTIVE" rev-parse HEAD)" = \
  "$(git -C "$NEW_ACTIVE" rev-parse origin/experiment/moe-pr13-pr14-stack)"
```

Expected: clean default work-off branch at the same pushed commit and unchanged GitHub/local default branch.

- [x] **Step 4: Verify ignore, CI, and service invariants**

```bash
set -e
COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY="${COMMON_DIR%/.git}"
git -C "$PRIMARY" check-ignore -v .worktrees/example/file
gh pr checks 32 --repo davidtai/MTPLX --json state > /tmp/mtplx-pr32-checks.json
test -z "$(jq -r '.[] | select(.state != "SUCCESS") | .name' \
  /tmp/mtplx-pr32-checks.json)"
test ! -d /tmp/mtplx-gpu-exclusive
curl -fsS --max-time 3 http://127.0.0.1:8080/v1/models |
  jq -e '.data[] | select(.id == "mtplx-qwen36-27b-optimized-speed")'
```

Expected: local ignore active, PR checks green, GPU lock absent, and Qwen unchanged/ready.

- [x] **Step 5: Mark the plan complete and publish the evidence-only update**

From `$NEW_ACTIVE`, mark all plan checkboxes complete and stage the corrected
`.gitignore`, `CONTRIBUTING.md`, layout spec, and plan. Run `git diff --check`,
commit those four files as `docs: correct workspace-level worktree layout`,
push, and wait for all PR #32 checks to pass again.
