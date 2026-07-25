from __future__ import annotations

from pathlib import Path

from scripts.check_documentation import check_documentation


ROOT = Path(__file__).resolve().parents[1]


def test_streamed_moe_release_flow_lives_in_advanced_guide():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_expert_manifest.py" not in root_readme
    assert "scripts/build_expert_manifest.py" not in advanced
    assert "python3 -m pip install mtplx==2.3.1rc1" in advanced
    assert "--expert-profile hy3-oq2e-64" in advanced


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
