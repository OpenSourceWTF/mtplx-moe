from __future__ import annotations

from pathlib import Path

from scripts.check_documentation import check_documentation


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_settings_guide_covers_all_scopes_and_precedence():
    text = _text("settings.md")
    for phrase in (
        "settings user set",
        "settings live set",
        "--set",
        "--settings",
        "Precedence",
        "settings explain",
        "API key file",
    ):
        assert phrase in text


def test_cli_guide_explains_settings_versus_operands():
    text = _text("cli.md")
    assert "Settings versus command inputs" in text
    assert "mtplx help advanced" in text
    assert "mtplx lab list" in text


def test_experiment_guide_covers_lifecycle_and_provenance():
    text = _text("experiments.md")
    for phrase in (
        "active",
        "retained",
        "rejected",
        "superseded",
        "expired",
        "SHA-256",
    ):
        assert phrase in text


def test_migration_guide_maps_every_compatibility_alias():
    text = _text("migration-settings.md")
    from mtplx.settings.builtins import BUILTIN_SETTINGS

    for spec in BUILTIN_SETTINGS:
        for alias in spec.aliases:
            if alias.source in {"cli", "env"}:
                assert alias.name in text


def test_docs_index_links_required_user_journeys():
    text = _text("README.md")
    for target in (
        "getting-started.md",
        "settings.md",
        "cli.md",
        "experiments.md",
        "migration-settings.md",
        "advanced/ssd-streamed-moe.md",
    ):
        assert f"]({target})" in text


def test_streamed_moe_commands_live_in_advanced_guide_not_root_readme():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_expert_manifest.py" not in root_readme
    assert "scripts/build_expert_manifest.py" in advanced
    assert "--expert-memory-limit 104GiB" in advanced


def test_root_readme_has_settings_native_normal_path():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "mtplx start",
        "mtplx settings user set runtime.profile=sustained",
        "mtplx start --set generation.temperature=0.7",
        "mtplx settings explain runtime.profile",
        "docs/settings.md",
        "docs/experiments.md",
    )
    for phrase in required:
        assert phrase in text


def test_configuration_sources_of_truth_are_easy_to_find():
    root = (ROOT / "README.md").read_text(encoding="utf-8")
    index = _text("README.md")
    settings = _text("settings.md")
    migration = _text("migration-settings.md")

    for text in (root, index, settings):
        assert "mtplx settings list --json" in text
        assert "reference/settings.md" in text
        assert "migration-settings.md" in text
    assert "generated" in migration.lower()
    assert "do not hand-edit" in migration.lower()


def test_root_readme_normal_sections_do_not_teach_legacy_runtime_flags():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normal = text.split("## Advanced and compatibility", 1)[0]
    forbidden = (
        "--profile sustained",
        "--default-temperature",
        "--default-top-p",
        "--adaptive-policy",
        "MTPLX_COMPILED_VERIFY=",
        "MTPLX_NAX_VERIFY=",
    )
    assert not [item for item in forbidden if item in normal]


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
