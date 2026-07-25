# OpenSourceWTF fork topology

The MoE and SSD-streaming distribution lives at
[`OpenSourceWTF/mtplx-moe`](https://github.com/OpenSourceWTF/mtplx-moe).
It is a public GitHub fork of
[`youssofal/MTPLX`](https://github.com/youssofal/MTPLX), so upstream ancestry
and attribution remain visible in the repository network.

## Repository state (2026-07-25)

```text
moe         https://github.com/OpenSourceWTF/mtplx-moe.git
upstream    https://github.com/youssofal/MTPLX.git
origin      https://github.com/davidtai/MTPLX.git
```

The legacy private `davidtai/MTPLX` repository was not made public, transferred,
renamed, or deleted. Existing worktrees retain `origin` so active private work
is not disrupted; release and integration work targets the explicit `moe`
remote.

The fork keeps Youssof Altoukhi's upstream authorship and citation metadata.
Fork-specific project, documentation, issue, and CI links point at
`OpenSourceWTF/mtplx-moe`.

## Syncing upstream

Fetch upstream and merge only reviewed commits or pull-request heads:

```bash
git fetch upstream main
git merge --no-ff upstream/main
git push moe HEAD:main
```

Feature pull requests that cannot be fast-forwarded should be merged on an
integration branch, verified, and then pushed to `moe/main` without rewriting
published history.
