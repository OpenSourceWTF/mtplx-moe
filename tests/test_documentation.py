from __future__ import annotations

from pathlib import Path
import tomllib

from scripts.check_documentation import check_documentation


ROOT = Path(__file__).resolve().parents[1]


def test_streamed_moe_release_flow_lives_in_advanced_guide():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_expert_manifest.py" not in root_readme
    assert "scripts/build_expert_manifest.py" not in advanced
    source_install = (
        'python3 -m pip install "mtplx @ '
        'git+https://github.com/OpenSourceWTF/mtplx-moe.git@main"'
    )
    install_guides = [
        root_readme,
        advanced,
        (ROOT / "INSTALL.md").read_text(encoding="utf-8"),
        (ROOT / "docs/quickstart.md").read_text(encoding="utf-8"),
    ]
    assert all(source_install in guide for guide in install_guides)
    assert "--expert-profile hy3-oq2e-64" in advanced


def test_mtplx_moe_identifies_as_a_source_installed_fork():
    user_guides = [
        ROOT / "README.md",
        ROOT / "INSTALL.md",
        ROOT / "docs/quickstart.md",
        ROOT / "docs/server.md",
        ROOT / "docs/advanced/ssd-streamed-moe.md",
    ]
    guide_text = "\n".join(
        path.read_text(encoding="utf-8") for path in user_guides
    )
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version_source = (ROOT / "mtplx/version.py").read_text(encoding="utf-8")

    assert project["project"]["version"] == "2.3.0+opensourcewtf.moe"
    assert '__version__ = "2.3.0+opensourcewtf.moe"' in version_source
    assert "2.3.1rc1" not in guide_text
    assert "release candidate" not in guide_text.lower()
    assert "not published to PyPI" in guide_text


def test_fork_release_workflow_cannot_publish_the_upstream_pypi_package():
    workflow = (ROOT / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    distribution_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "docs/PYPI_RELEASE.md",
            ROOT / "docs/development.md",
        ]
    )

    assert "publish_to_pypi" not in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "id-token: write" not in workflow
    assert "publish_to_pypi" not in distribution_docs
    assert "not published to PyPI" in distribution_docs


def test_documentation_checker_skips_plan_pseudocode_but_checks_user_guides(
    tmp_path,
):
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "internal.md").write_text(
        '# Internal\n\n```bash\necho "pseudocode\n```\n',
        encoding="utf-8",
    )
    guide = tmp_path / "docs" / "guide.md"
    guide.write_text(
        '# Guide\n\n```bash\necho "broken\n```\n',
        encoding="utf-8",
    )

    report = check_documentation(tmp_path)

    assert not [item for item in report.invalid_shell_blocks if "plans/" in item]
    assert len(report.invalid_shell_blocks) == 1
    assert report.invalid_shell_blocks[0].startswith("docs/guide.md:")


def test_documentation_checker_accepts_repository_docs():
    report = check_documentation(ROOT)
    assert report.missing_links == ()
    assert report.invalid_shell_blocks == ()
    assert report.unknown_commands == ()
    assert report.legacy_normal_path_flags == ()
