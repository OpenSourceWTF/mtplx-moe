# Publishing & reparenting MTPLX (staged — nothing here has been executed)

This is a decision + runbook doc for taking the MoE-SSD-streaming build of
`davidtai/MTPLX` public and moving it to the `OpenSourceWTF` org. **No step in
this file has been run.** Each outward-facing / irreversible action waits for an
explicit go, per the repo's hard rules (never open/transfer/repoint unasked).

## Current state (read-only checks, 2026-07-24)

```
origin      https://github.com/davidtai/MTPLX.git      # isFork:false, isPrivate:true, parent:null
upstream    https://github.com/youssofal/MTPLX.git     # isFork:false, isPrivate:false
pr167-head  https://github.com/davidtai/MTPLX-1.git
```

`OpenSourceWTF` org exists (10 public + 7 private repos) and has **no**
`mtplx`/`MTPLX` repo yet.

## Step 1 — Make `davidtai/MTPLX` public  *(self-service, low blast radius)*

Repo → Settings → Danger Zone → Change visibility → Public. Exposes history +
issues. Before flipping: run the secret scan (hard rule — scan/exclude secrets
before anything goes public), confirm `keys/` and any `.env` are gitignored and
absent from history.

## Step 2 — "Reparent to upstream `youssofal/MTPLX`"  *(NOT self-service — decision needed)*

The intent is a visible "forked from youssofal/MTPLX" relationship. **This is
the hard part:** `davidtai/MTPLX` was created standalone (`isFork:false`), so
GitHub has no fork-network edge to redirect. There is no Settings toggle for it.
Two realistic paths — pick one:

- **A. Re-fork + push (gets a real fork badge, changes repo identity).** Fork
  `youssofal/MTPLX` via GitHub's UI/API to create a genuine fork, push every
  branch/tag we care about into it (`feat/moe-streaming-oob`, plus the named
  branches still in play — `perf/a3b-decode-stack`, `feat/ccopy-target-prefix`,
  etc.), then repoint local remotes. Open question: what happens to open work
  against today's `origin` (e.g. PR #174) — it does not migrate automatically.
- **B. GitHub Support relink (metadata-only, uncertain).** Ask Support to attach
  the existing repo to the upstream fork network. Historically Support only
  relinks a repo that is *already* a fork somewhere in the target network;
  attaching a standalone repo is not documented/guaranteed. Low effort to ask,
  may be declined.

Recommendation: if the fork badge matters, **A**; otherwise credit upstream
prominently in README/NOTICE (already done) and skip the badge. Decision is
David's.

## Step 3 — Move to `OpenSourceWTF/mtplx`  *(self-service transfer if admin on both sides)*

Repo → Settings → Danger Zone → Transfer ownership → `OpenSourceWTF/mtplx`.
GitHub preserves stars/issues/PR history and redirects the old URL. Transfer
does **not** by itself change fork-network parentage — sequence it relative to
Step 2 deliberately.

Post-transfer checklist (things that do NOT carry over automatically):
- CI/Actions secrets + variables scoped to `davidtai/MTPLX` → re-add under the org.
- Webhooks / deploy keys → re-create.
- Hardcoded repo URLs: README badges, `pyproject.toml`, `CITATION.cff`, `NOTICE`,
  and the two HF model cards (which link back to the source repo).
- Collaborator/team access → re-grant under org permissions.
- Branch protection rules → re-apply.

## Suggested sequencing

1. Secret scan → **Step 1** (public).
2. Decide Step 2 path (A vs B) — this determines whether Step 3 is a transfer of
   the current repo or a push into a fresh fork.
3. **Step 3** (org move) + run the post-transfer checklist.
4. Update HF model-card "source" links to the final canonical URL.

Nothing above runs without David's explicit, per-step go.
