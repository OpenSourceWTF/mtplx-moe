from __future__ import annotations

import json
from pathlib import Path
import re
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
    install_guides = [
        root_readme,
        advanced,
        (ROOT / "INSTALL.md").read_text(encoding="utf-8"),
        (ROOT / "docs/quickstart.md").read_text(encoding="utf-8"),
    ]
    assert all(
        'MTPLX_MOE_VENV="$HOME/.venvs/mtplx-moe"' in guide
        for guide in install_guides
    )
    assert all(
        '"$MTPLX_MOE_VENV/bin/python" -m pip install' in guide
        for guide in install_guides
    )
    assert all(
        "git+https://github.com/OpenSourceWTF/mtplx-moe.git@main" in guide
        for guide in install_guides
    )
    assert "--expert-profile hy3-oq2e-64" in advanced
    assert "OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX-streaming" in advanced
    assert "configs/kimi-k3-t158-110g-host.json" in advanced
    assert "configs/kimi-k3-t158-128g-host.json" in advanced
    assert "Kimi K3 t158" in advanced


def test_fork_docs_cover_upstream_and_litellm_collisions():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (ROOT / "INSTALL.md").read_text(encoding="utf-8")
    server = (ROOT / "docs/server.md").read_text(encoding="utf-8")
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(
        encoding="utf-8"
    )

    assert "same Python distribution" in readme
    assert "type -a mtplx" in install
    assert "port 8000" in install
    assert "shared `~/.mtplx`" in install
    assert 'MTPLX_CLIENT_VENV="$HOME/.venvs/mtplx-clients"' in install
    assert "1.93.0 requires `rich<14`" in install
    assert "hy3-oq2e-mtplx-streaming" in server
    assert "hy3-oq2e-mtplx-streaming" in advanced
    assert "openai/hy3-oq2e-mtplx-streaming" in advanced
    assert "`validation.ok: false`" in advanced
    assert "“Sustained MTP”" in advanced
    assert "http://127.0.0.1:4000/v1" in advanced
    assert "For the primary command\nit is `OpensourceWTF/" not in advanced


def test_docs_index_surfaces_fork_guides_after_introduction():
    index = (ROOT / "docs/README.md").read_text(encoding="utf-8")

    introduction = index.index("This index separates")
    fork_section = index.index("## OpenSourceWTF fork")
    start_section = index.index("## Start")
    assert introduction < fork_section < start_section
    assert "(../INSTALL.md)" in index
    assert "(advanced/ssd-streamed-moe.md)" in index
    assert "(server.md#hy3-expertsbin-fork)" in index
    assert "(PUBLISH_AND_REPARENT.md)" in index
    assert "(PYPI_RELEASE.md)" in index


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


def test_readme_lists_every_preconfigured_model_in_one_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    catalog_source = (ROOT / "mtplx/model_catalog.py").read_text(
        encoding="utf-8"
    )
    streaming_source = (ROOT / "mtplx/default_models.py").read_text(
        encoding="utf-8"
    )
    laguna_source = (ROOT / "mtplx/models/laguna_config.py").read_text(
        encoding="utf-8"
    )
    kimi_streaming_id = "OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX-streaming"

    assert "## Supported models" in readme
    supported_section = readme.split("## Supported models", 1)[1].split(
        "\n## ", 1
    )[0]
    expected_ids = set(
        re.findall(
            r'hf_model_id="([^"]+)"',
            catalog_source + streaming_source,
        )
    )
    laguna_match = re.search(
        r'LAGUNA_S_2_1_REPO_ID = "([^"]+)"',
        laguna_source,
    )
    assert laguna_match is not None
    expected_ids.add(laguna_match.group(1))
    expected_ids.add(kimi_streaming_id)

    documented_ids = set(
        re.findall(
            r"https://huggingface\.co/([^)\s]+)",
            supported_section,
        )
    )
    assert documented_ids == expected_ids
    assert len(documented_ids) == 17


def test_readme_publishes_kimi_streaming_contract_and_receipt():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    supported_section = readme.split("## Supported models", 1)[1].split(
        "\n## ", 1
    )[0]

    assert "OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX-streaming" in supported_section
    assert "96 GiB: **1.18 tok/s**" in supported_section
    assert "110 GiB: **1.11 tok/s**" in supported_section
    assert "Eligible resident linear and embedding weights are dynamically" in (
        supported_section
    )
    assert "four-worker Hugging Face upload" in supported_section
    assert "memory-safety receipt" in supported_section
    assert "rather than a speed" in supported_section
    assert "separate 96 GiB and 110 GiB launch examples" in supported_section


def test_readme_publishes_retained_hy3_quality_and_speed_receipts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    supported_section = readme.split("## Supported models", 1)[1].split(
        "\n## ", 1
    )[0]
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(
        encoding="utf-8"
    )
    flagship = json.loads(
        (
            ROOT
            / "evals/tier2/hy3_oq2e_rq4_flagship_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert "HumanEvalPlus pass@1" in supported_section
    assert "`q4` requant: 86.6% (142/164)" in supported_section
    assert "`q8`: 87.2% (143/164)" in supported_section
    assert "**48.04 tok/s**" in supported_section
    assert "41.36 tok/s AR control" in supported_section
    assert "MTP depth 1" in supported_section
    assert "`-64`: 9.31" in supported_section
    assert "`-88`: 22.35" in supported_section
    assert "`-96`: 30.17 tok/s" in supported_section
    assert "M5 Max with 128 GB unified memory" in supported_section
    assert "current `mtplx serve` route is AR-only" in supported_section
    assert '"proj_requant": "q4"' in advanced
    assert "79 pinned islands" in advanced
    assert "BF16 KV" in advanced
    assert flagship["configuration"]["proj_requant"] == "q4"
    assert flagship["quality_receipts"]["q4_requant_passed"] == 142
    assert flagship["quality_receipts"]["q8_control_passed"] == 143
    assert flagship["means"]["mtp_depth_1_decode_tok_s"] == 48.044216448165095
    assert flagship["means"]["ar_decode_tok_s"] == 41.36380811105886
    assert all(row["token_parity"] for row in flagship["repetitions"])
    assert "Not measured" in supported_section


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
