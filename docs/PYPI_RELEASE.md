# Fork distribution policy

The OpenSourceWTF `mtplx-moe` fork is not published to PyPI or the upstream
Homebrew tap. Install it directly from GitHub:

```bash
python3 -m pip install "mtplx @ git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"
```

The `build fork artifacts` GitHub Actions workflow builds and checks wheels and
source archives. It has no PyPI credentials, OIDC publishing permission, or
publish job.

Official MTPLX distribution belongs to
[youssofal/MTPLX](https://github.com/youssofal/MTPLX). Its release process and
version numbers do not apply to this fork.
