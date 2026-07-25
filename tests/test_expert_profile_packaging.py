import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def _build_wheel(outdir: Path, *, epoch: int) -> Path:
    outdir.mkdir()
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(outdir),
        ],
        check=True,
        env=env,
    )
    return next(outdir.glob("mtplx-2.3.1rc1-*.whl"))


def test_built_wheel_is_reproducible_and_contains_expert_profiles(tmp_path):
    first = _build_wheel(tmp_path / "first", epoch=1_700_000_000)
    second = _build_wheel(tmp_path / "second", epoch=1_700_000_000)

    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()

    wheel = first
    with zipfile.ZipFile(wheel) as archive:
        payload = json.loads(
            archive.read("mtplx/data/expert_profiles.json")
        )
    assert [row["name"] for row in payload["profiles"]] == [
        "hy3-oq2e-64",
        "hy3-oq2e-88",
        "hy3-oq2e-96",
    ]


def test_release_workflow_sets_commit_epoch_before_build():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "release.yml"
    ).read_text(encoding="utf-8")

    epoch = 'SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD)'
    assert epoch in workflow
    assert workflow.index(epoch) < workflow.index("python -m build")
