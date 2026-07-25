import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


def _clean_source_tree(destination: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    destination.mkdir()
    included_paths = [
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "mtplx",
        "scripts",
        "tests",
        "vllm_metal",
    ]
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", "HEAD", "--", *included_paths],
        cwd=repo,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    assert process.wait() == 0
    for name in ("pyproject.toml", "MANIFEST.in", "setup.py"):
        source = repo / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return destination


def _build_release(source: Path, outdir: Path, *, epoch: int) -> tuple[Path, Path]:
    outdir.mkdir()
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--outdir",
            str(outdir),
        ],
        check=True,
        cwd=source,
        env=env,
    )
    wheel = next(outdir.glob("mtplx-2.3.1rc1-*.whl"))
    sdist = next(outdir.glob("mtplx-2.3.1rc1.tar.gz"))
    return wheel, sdist


def test_built_release_is_reproducible_and_contains_expert_profiles(tmp_path):
    epoch = 1_700_000_000
    first_source = _clean_source_tree(tmp_path / "source-first")
    second_source = _clean_source_tree(tmp_path / "source-second")
    first_wheel, first_sdist = _build_release(
        first_source,
        tmp_path / "dist-first",
        epoch=epoch,
    )
    second_wheel, second_sdist = _build_release(
        second_source,
        tmp_path / "dist-second",
        epoch=epoch,
    )

    for first, second in (
        (first_wheel, second_wheel),
        (first_sdist, second_sdist),
    ):
        assert first.read_bytes() == second.read_bytes()
        assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
            second.read_bytes()
        ).digest()

    assert int.from_bytes(first_sdist.read_bytes()[4:8], "little") == epoch
    with tarfile.open(first_sdist) as archive:
        members = archive.getmembers()
    assert {member.mtime for member in members} == {epoch}
    assert "mtplx-2.3.1rc1/setup.py" in {
        member.name for member in members
    }

    with zipfile.ZipFile(first_wheel) as archive:
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
