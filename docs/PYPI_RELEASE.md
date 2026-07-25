# Fork distribution policy

The OpenSourceWTF `mtplx-moe` fork is not published to PyPI or the upstream
Homebrew tap. It shares upstream MTPLX's `mtplx` distribution, import, and CLI
names, so install it directly from GitHub in the dedicated environment
described in [INSTALL.md](../INSTALL.md):

```bash
MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"
"$MTPLX_MOE_VENV/bin/python" -m pip install \
  "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
```

The `build fork artifacts` GitHub Actions workflow builds and checks wheels and
source archives. It has no PyPI credentials, OIDC publishing permission, or
publish job.

Official MTPLX distribution belongs to
[youssofal/MTPLX](https://github.com/youssofal/MTPLX). Its release process and
version numbers do not apply to this fork.
