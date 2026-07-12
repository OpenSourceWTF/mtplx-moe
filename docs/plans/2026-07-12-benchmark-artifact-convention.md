# Benchmark Artifact Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep machine-generated benchmark artifacts out of Git under one standardized path, then publish the verified main-based MoE integration as a draft PR.

**Architecture:** `benchmarks/raw/` is the ignored repository-local artifact boundary, while `benchmarks/results/` contains reviewed summaries. Contributor and benchmark documentation define one run-ID grammar and convert current MoE commands and references to that boundary. The resolved integration tree is then tested in the declared environments, committed as a two-parent merge, pushed, and opened as a draft PR.

**Tech Stack:** Git ignore rules, Markdown, Git worktrees, Python/pytest, Ruff, MLX, Swift Package Manager, GitHub CLI.

**Assumptions:** Assumes the 99 tracked JSON files in `benchmarks/results/` are legacy curated summaries — this plan will not delete or relocate them. Assumes `origin/main` remains at `43c8f96dc6f4209114bcc9e6d91351ccb78f0fbc` through publication — the merge must be re-audited if it moves. Assumes the shared `mtplx-hy3-ssd/.venv` still matches the lockfile and contains the declared `dev,server` extras — use a fresh project environment if dependency versions diverge.

---

## File Structure

- `.gitignore`: owns the single ignored raw-artifact boundary.
- `CONTRIBUTING.md`: authoritative repository-wide benchmark naming and evidence rule.
- `benchmarks/results/README.md`: explains raw versus curated storage with a path example.
- `mtplx/benchmarks/README.md`: corrects the stale curated-report location.
- `docs/MOE_RUNTIME_PR_BENCHMARKS.md`: preserves conclusions without links to ignored artifacts.
- `docs/plans/2026-07-11-moe-runtime-pr-salvage.md`: routes benchmark commands to `benchmarks/raw/`.
- `docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md`: aligns its retention rule.
- `docs/specs/2026-07-12-benchmark-artifact-convention-design.md`: approved design.
- `docs/plans/2026-07-12-benchmark-artifact-convention.md`: execution plan.

### Task 1: Establish the raw-artifact boundary and naming contract

**Files:**
- Modify: `.gitignore`
- Modify: `CONTRIBUTING.md`
- Modify: `benchmarks/results/README.md`
- Modify: `mtplx/benchmarks/README.md`

**Security flag:** `none`

- [x] **Step 1: Verify the raw path is not yet ignored**

```bash
test "$(git check-ignore -q benchmarks/raw/moe-runtime/example/run.json; echo $?)" = 1
```

Expected: PASS because the representative raw JSON is not ignored before the change.

- [x] **Step 2: Add the ignored path**

Add beneath the generated-artifacts section in `.gitignore`:

```gitignore
# Repository-local raw benchmark artifacts
/benchmarks/raw/
```

- [x] **Step 3: Add the authoritative contributor rule**

Append this contract to `CONTRIBUTING.md`:

```markdown
## Benchmark artifacts

Write repository-local machine-generated artifacts beneath
`benchmarks/raw/<benchmark>/<run-id>/`. Git ignores that entire tree. Keep only
reviewed, human-readable comparison summaries in `benchmarks/results/`; do not
force-add raw JSON, JSONL, traces, or generated responses.

Use `<benchmark>-<variant>-<YYYYMMDDTHHMMSSZ>-<short-sha>` for `run-id`.
Names are lowercase ASCII letters, digits, and hyphens; timestamps are UTC;
`short-sha` is the tested 7-12 character commit. Inside a run directory, use
one-based, two-digit repeat names such as `base-r01.json`,
`candidate-r01.json`, and `response-r01.md`.
```

- [x] **Step 4: Mirror the boundary in benchmark documentation**

Update `benchmarks/results/README.md` to explain raw versus curated storage and
that curated reports record reproducibility metadata instead of linking to
ignored files. Correct `mtplx/benchmarks/README.md` to point to
`benchmarks/results/` instead of the nonexistent `mtplx/benchmarks/reports/`.

- [x] **Step 5: Verify ignore behavior and legacy preservation**

```bash
git check-ignore -v \
  benchmarks/raw/moe-runtime/run/base-r01.json \
  benchmarks/raw/moe-runtime/run/decode-trace.jsonl \
  benchmarks/raw/moe-runtime/run/response-r01.md
test "$(git check-ignore -q benchmarks/results/moe-runtime-gate-matrix.md; echo $?)" = 1
test "$(git ls-files 'benchmarks/results/*.json' | wc -l | tr -d ' ')" = 99
```

Expected: all raw files match `/benchmarks/raw/`, curated Markdown is not ignored, and 99 legacy summaries remain tracked.

### Task 2: Align the active MoE benchmark documentation

**Files:**
- Modify: `docs/MOE_RUNTIME_PR_BENCHMARKS.md`
- Modify: `docs/plans/2026-07-11-moe-runtime-pr-salvage.md`
- Modify: `docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md`

**Security flag:** `none`

- [x] **Step 1: Capture stale raw-result references**

```bash
rg -n '\.\./benchmarks/results/.*(json|repeat-.*\.md)|raw artifacts under `benchmarks/results/`|output-dir benchmarks/results' \
  docs/MOE_RUNTIME_PR_BENCHMARKS.md \
  docs/plans/2026-07-11-moe-runtime-pr-salvage.md \
  docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md
```

Expected: prints the legacy links and paths, proving the desired final condition does not yet hold.

- [x] **Step 2: Convert commands and retention language**

Use `benchmarks/raw/moe-runtime/<run-id>/` for generated JSON and responses.
Replace links to absent raw files with inline artifact identifiers while
retaining raw decode values and conclusions. State that consolidated reports
are tracked and raw artifacts are ignored or externally attached.

- [x] **Step 3: Verify documentation consistency**

```bash
! rg -n '\.\./benchmarks/results/.*(json|repeat-.*\.md)|raw artifacts under `benchmarks/results/`|output-dir benchmarks/results' \
  docs/MOE_RUNTIME_PR_BENCHMARKS.md \
  docs/plans/2026-07-11-moe-runtime-pr-salvage.md \
  docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md
rg -n 'benchmarks/raw/|benchmarks/results/' \
  CONTRIBUTING.md benchmarks/results/README.md \
  docs/MOE_RUNTIME_PR_BENCHMARKS.md \
  docs/plans/2026-07-11-moe-runtime-pr-salvage.md \
  docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md
```

Expected: no stale raw-result links or output paths and consistent standardized locations.

### Task 3: Verify the complete resolved integration tree

**Files:**
- Test: all staged Python and Swift changes

**Security flag:** `none`

- [x] **Step 1: Revalidate merge and critical-file integrity**

```bash
test "$(git rev-parse HEAD)" = 43c8f96dc6f4209114bcc9e6d91351ccb78f0fbc
test "$(git rev-parse MERGE_HEAD)" = cc659f93eea2c0f4ed5e5b0eff3aca126e4ecf28
test -z "$(git diff --name-only --diff-filter=U)"
for file in mtplx/server/openai.py mtplx/profiles.py mtplx/expert_runtime.py \
  mtplx/expert_slots.py mtplx/expert_streaming.py mtplx/models/expert_mlx.py; do
  git diff --quiet bench/pr13-pr14-stack-retest -- "$file"
done
! rg -n 'aggregate_misses|finish_misses\(\)' mtplx/models/expert_mlx.py
```

Expected: intended parents, no unresolved paths, critical files equal `cc659f9`, and the excluded barrier is absent.

- [x] **Step 2: Acquire the exclusive lane and unload Qwen**

```bash
mkdir /tmp/mtplx-gpu-exclusive
launchctl disable "gui/$(id -u)/com.tea.qwen"
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tea.qwen.plist" || true
! pgrep -f 'mtplx.server.openai.*Qwen3.6'
! pgrep -f 'benchmark_streamed_generation|probe_mtp|probe_paged'
```

Expected: lock acquired and no competing model or benchmark runner exists.

- [x] **Step 3: Run the full Python suite**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q
```

Expected: full suite passes with exact counts and duration reported.

- [x] **Step 4: Run Python static checks**

```bash
git diff --cached --name-only --diff-filter=ACMR -z -- '*.py' |
  xargs -0 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check --
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m compileall -q mtplx scripts tests
git diff --cached --check
```

Expected: changed files pass Ruff lint, compile successfully, and contain no whitespace errors. Ruff formatting is not applied in this integration: the candidate inherits 77 changed-file and 228 repository-wide formatting deviations, and no repository workflow enforces `ruff format`; formatting them would create an unrelated broad rewrite.

- [x] **Step 5: Attempt Swift tests and release build**

```bash
swift test --package-path apps/MTPLXApp
swift build --package-path apps/MTPLXApp -c release --product MTPLXApp
```

Expected: both commands exit zero when full Xcode supplies `SwiftDataMacros`.

Local result: both commands stop before tests because this Mac has only Command Line Tools 26.6, no Xcode installation, and no `SwiftDataMacros` plugin. An untouched `origin/main` archive fails at the identical `ChatModels.swift:27` macro expansion, proving an environmental limitation rather than a merge regression. Installing full Xcode is disproportionate to publishing this experimental draft branch; retain this as an explicit outstanding release gate.

- [x] **Step 6: Restore Qwen even if a check fails**

```bash
rmdir /tmp/mtplx-gpu-exclusive
launchctl enable "gui/$(id -u)/com.tea.qwen"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tea.qwen.plist"
curl -fsS http://127.0.0.1:8080/v1/models |
  jq -e '.data[] | select(.id == "mtplx-qwen36-27b-optimized-speed")'
```

Expected: lock absent and Qwen serving the expected model.

### Task 4: Commit, publish, set the default branch, and finish worktree cleanup

**Files:**
- Stage: all resolved merge files and benchmark convention documentation

**Security flag:** `none`

- [ ] **Step 1: Stage and inspect final scope**

```bash
git add .gitignore CONTRIBUTING.md benchmarks/results/README.md \
  mtplx/benchmarks/README.md docs/MOE_RUNTIME_PR_BENCHMARKS.md \
  docs/plans/2026-07-11-moe-runtime-pr-salvage.md \
  docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md \
  docs/specs/2026-07-12-benchmark-artifact-convention-design.md \
  docs/plans/2026-07-12-benchmark-artifact-convention.md
git status --short --branch
git diff --cached --stat
git diff --cached --check
```

Expected: no raw artifacts, unresolved files, or unintended convention changes.

- [ ] **Step 2: Create and verify the merge commit**

```bash
git commit -m "merge: establish stacked MoE runtime experiment"
git show -s --format='%H%n%P%n%s' HEAD
test "$(git rev-parse HEAD^1)" = 43c8f96dc6f4209114bcc9e6d91351ccb78f0fbc
test "$(git rev-parse HEAD^2)" = cc659f93eea2c0f4ed5e5b0eff3aca126e4ecf28
```

Expected: one merge commit with `origin/main` first and tested candidate second.

- [ ] **Step 3: Push and open a draft PR**

```bash
git push -u origin experiment/moe-pr13-pr14-stack
gh pr create --repo davidtai/MTPLX --base main \
  --head experiment/moe-pr13-pr14-stack --draft \
  --title "experiment: stack retained MoE runtime optimizations on main" \
  --body-file /tmp/mtplx-experimental-pr-body.md
```

The PR body must name both merge parents, the candidate-wins policy for 75
conflicting paths, final validation, raw decode canary
`6.26989944912 -> 6.628692480060882 tok/s (+5.7225%)`, six-pair result
`6.2868045458 -> 6.3647565480 tok/s (+1.23993%, 5/6 positive)`, and exclusions
`0bde6ac`, `e1fca36`, global lock changes, and raw artifacts.

- [ ] **Step 4: Verify remote state**

```bash
git ls-remote origin refs/heads/experiment/moe-pr13-pr14-stack
gh pr view --repo davidtai/MTPLX experiment/moe-pr13-pr14-stack \
  --json number,title,state,isDraft,headRefName,baseRefName,url,headRefOid
```

Expected: remote branch points to the merge commit and an open draft PR targets `main`.

- [ ] **Step 5: Set and verify the repository default branch**

```bash
gh repo edit davidtai/MTPLX --default-branch experiment/moe-pr13-pr14-stack
git remote set-head origin -a
test "$(gh repo view davidtai/MTPLX --json defaultBranchRef --jq '.defaultBranchRef.name')" = \
  experiment/moe-pr13-pr14-stack
test "$(git symbolic-ref --short refs/remotes/origin/HEAD)" = \
  origin/experiment/moe-pr13-pr14-stack
```

Expected: GitHub and this checkout's `origin/HEAD` both identify the experimental integration branch as the default.

- [ ] **Step 6: Remove the final superseded source worktree**

```bash
git -C /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-q4-native \
  worktree remove /Users/davidtai/projects/OpenSourceWTF/mtplx-runtime-pr14-current
test ! -e /Users/davidtai/projects/OpenSourceWTF/mtplx-runtime-pr14-current
git -C /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-q4-native \
  show-ref --verify refs/heads/eval/repaired-pr14
```

Expected: clean worktree removed and its branch retained.
