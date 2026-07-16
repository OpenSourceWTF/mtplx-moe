# Documentation and README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the root README a concise, verified path from installation to normal settings-based use, with generated settings references and focused advanced/lab/migration guides.

**Architecture:** Human guides explain workflows and link to a deterministic reference generated from the settings catalog. A documentation checker validates generated artifacts, local links, shell parsing, and the absence of legacy runtime flags in normal-use README sections. Advanced streamed-MoE details move intact to a focused guide.

**Tech Stack:** Markdown; Python stdlib `json`, `pathlib`, `re`, `shlex`; existing CLI; pytest; Ruff.

**Assumptions:**

- Assumes settings, experiment, and CLI plans have passed — will NOT document interfaces that do not exist.
- Assumes benchmark claims remain unchanged — will NOT refresh throughput or hardware numbers without a separate measured run.
- Assumes current specialized documents remain valid unless directly linked/moved — will NOT reorganize historical plans or result archives.
- Assumes docs checks run without a model — will NOT claim hardware execution for parser-only smoke tests.

---

## File Structure

- `docs/README.md` — documentation index by user goal.
- `docs/getting-started.md` — CLI install, first start, and client connection.
- `docs/settings.md` — persistent/per-run/bundle/live scopes, precedence, and common recipes.
- `docs/cli.md` — public/advanced/lab command map and settings-versus-operands boundary.
- `docs/experiments.md` — lab lifecycle, validation, application, and provenance.
- `docs/migration-settings.md` — legacy flag/environment aliases and compatibility window.
- `docs/advanced/ssd-streamed-moe.md` — moved advanced Hy3/GLM instructions.
- `docs/reference/settings.md`, `docs/reference/settings.json` — generated catalog.
- `scripts/generate_settings_reference.py` — deterministic Markdown/JSON generation.
- `scripts/check_documentation.py` — links, code blocks, README policy, and command parser checks.
- `tests/test_settings_reference.py`, `tests/test_documentation.py` — generated and human-doc contracts.

### Task 1: Generate Settings Reference from the Catalog

**Files:**
- Create: `scripts/generate_settings_reference.py`
- Create: `docs/reference/settings.md`
- Create: `docs/reference/settings.json`
- Create: `tests/test_settings_reference.py`

**Security flag:** `security`

**Does NOT cover:** Internal-only settings, secret values, or migration prose.

- [x] **Step 1: Write failing generator tests**

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_settings_reference import render_reference


def test_reference_contains_every_public_setting_and_redacts_secrets():
    markdown, payload = render_reference()
    names = {item["name"] for item in payload["settings"]}
    assert "runtime.profile" in names
    assert "generation.temperature" in names
    assert "server.api_key_file" in names
    assert "/private" not in markdown
    assert all(item["visibility"] in {"public", "advanced", "experimental"} for item in payload["settings"])


def test_checked_in_reference_matches_renderer():
    root = Path(__file__).resolve().parents[1]
    markdown, payload = render_reference()
    assert (root / "docs/reference/settings.md").read_text(encoding="utf-8") == markdown
    assert json.loads((root / "docs/reference/settings.json").read_text(encoding="utf-8")) == payload
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_reference.py`

Expected: FAIL because generator/reference files do not exist.

- [x] **Step 3: Implement deterministic renderer**

Render sorted domains and canonical names. Markdown columns are setting, type,
default description, visibility, lifecycle, live/restart behavior, and legacy
aliases. JSON contains the same fields plus descriptions. Exclude internal
settings and secret values. Support `--write` and `--check`; JSON uses indent 2,
sorted keys, UTF-8, and a trailing newline.

- [x] **Step 4: Generate files and verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_settings_reference.py --write && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_reference.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_settings_reference.py --check`

Expected: PASS.

- [x] **Step 5: Commit and update issue**

```bash
git add scripts/generate_settings_reference.py docs/reference tests/test_settings_reference.py
git commit -m "docs: generate canonical settings reference"
gh issue comment 90 --repo davidtai/MTPLX --body "Documentation task 1 complete: deterministic Markdown/JSON settings references and drift checks landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 2: Write Focused Settings, CLI, Experiment, and Migration Guides

**Files:**
- Create: `docs/settings.md`
- Create: `docs/cli.md`
- Create: `docs/experiments.md`
- Create: `docs/migration-settings.md`
- Create: `tests/test_documentation.py`

**Security flag:** `security`

**Does NOT cover:** Root README, installation troubleshooting, or advanced streamed-MoE details.

- [x] **Step 1: Add failing content-contract tests**

```python
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_settings_guide_covers_all_scopes_and_precedence():
    text = _text("settings.md")
    for phrase in (
        "settings user set", "settings live set", "--set", "--settings",
        "Precedence", "settings explain", "API key file",
    ):
        assert phrase in text


def test_cli_guide_explains_settings_versus_operands():
    text = _text("cli.md")
    assert "Settings versus command inputs" in text
    assert "mtplx help advanced" in text
    assert "mtplx lab list" in text


def test_experiment_guide_covers_lifecycle_and_provenance():
    text = _text("experiments.md")
    for phrase in ("active", "retained", "rejected", "superseded", "expired", "SHA-256"):
        assert phrase in text


def test_migration_guide_maps_every_compatibility_alias():
    text = _text("migration-settings.md")
    from mtplx.settings.builtins import BUILTIN_SETTINGS
    for spec in BUILTIN_SETTINGS:
        for alias in spec.aliases:
            if alias.source in {"cli", "env"}:
                assert alias.name in text
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py`

Expected: FAIL because the four guides do not exist.

- [x] **Step 3: Write `docs/settings.md`**

Document user, per-run, bundle, live, environment compatibility, model/default,
profile, and constraint sources; the exact precedence; atomic TOML behavior;
secret-file rules; `show/list/explain`; and goal-oriented examples for profile,
sampling, context, cache, server, and streamed experts.

- [x] **Step 4: Write `docs/cli.md` and `docs/experiments.md`**

The CLI guide lists public/advanced/lab command maps and explains settings
versus operands/mechanics. The experiment guide documents data-only recipes,
`lab list/show/validate`, `lab:` application, lifecycle, model constraints,
hashes, evidence, and archive behavior.

- [x] **Step 5: Write generated alias migration sections**

`docs/migration-settings.md` explains the compatibility window and explicit
live/user scope change. Generate its CLI/environment mapping tables from the
catalog inside `generate_settings_reference.py` so every alias test is stable;
keep surrounding migration prose hand-authored.

- [x] **Step 6: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_settings_reference.py --write && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py tests/test_settings_reference.py`

Expected: PASS.

- [x] **Step 7: Commit and update issue**

```bash
git add docs/settings.md docs/cli.md docs/experiments.md docs/migration-settings.md docs/reference scripts/generate_settings_reference.py tests/test_documentation.py
git commit -m "docs: explain settings CLI experiments and migration"
gh issue comment 90 --repo davidtai/MTPLX --body "Documentation task 2 complete: focused settings, CLI, experiment, and compatibility guides landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 3: Move Streamed-MoE Detail and Add Documentation Index

**Files:**
- Create: `docs/advanced/ssd-streamed-moe.md`
- Create: `docs/getting-started.md`
- Create: `docs/README.md`
- Modify: `README.md`
- Modify: `tests/test_documentation.py`

**Security flag:** `none`

**Does NOT cover:** Changing streamed-MoE commands, memory recommendations, benchmark claims, or artifact contracts.

- [x] **Step 1: Add failing index and advanced-guide tests**

```python
def test_docs_index_links_required_user_journeys():
    text = _text("README.md")
    for target in (
        "getting-started.md", "settings.md", "cli.md", "experiments.md",
        "migration-settings.md", "advanced/ssd-streamed-moe.md",
    ):
        assert f"]({target})" in text


def test_streamed_moe_commands_live_in_advanced_guide_not_root_readme():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(encoding="utf-8")
    assert "scripts/build_expert_manifest.py" not in root_readme
    assert "scripts/build_expert_manifest.py" in advanced
    assert "--expert-memory-limit 104GiB" in advanced
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py`

Expected: FAIL because the advanced guide/index do not exist and README still
contains the detailed commands.

- [x] **Step 3: Move the streamed-MoE section without factual edits**

Move the full `Experimental SSD-streamed MoE` content, both command blocks,
memory guidance, model restrictions, and guide link into
`docs/advanced/ssd-streamed-moe.md`. Update its relative links. Replace the root
section with a three-to-five sentence capability/status summary and one link.

- [x] **Step 4: Write getting-started and docs index**

`getting-started.md` covers app and CLI installation, `mtplx start`, one
persistent setting, one per-run setting, server start, and client connection.
`docs/README.md` groups links under Start, Configure, Operate, Experiment, and
Develop/Reference.

- [x] **Step 5: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add README.md docs/README.md docs/getting-started.md docs/advanced/ssd-streamed-moe.md tests/test_documentation.py
git commit -m "docs: separate normal and advanced MTPLX workflows"
gh issue comment 90 --repo davidtai/MTPLX --body "Documentation task 3 complete: streamed-MoE detail moved to an advanced guide and the documentation index/getting-started path landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 4: Rewrite Root README Around Settings-native Use

**Files:**
- Modify: `README.md`
- Modify: `tests/test_documentation.py`

**Security flag:** `security`

**Does NOT cover:** New product claims, installation methods, model support tiers, or measured performance numbers.

- [x] **Step 1: Add failing README contract tests**

```python
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


def test_root_readme_normal_sections_do_not_teach_legacy_runtime_flags():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normal = text.split("## Advanced and compatibility", 1)[0]
    forbidden = (
        "--profile sustained", "--default-temperature", "--default-top-p",
        "--adaptive-policy", "MTPLX_COMPILED_VERIFY=", "MTPLX_NAX_VERIFY=",
    )
    assert not [item for item in forbidden if item in normal]
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py`

Expected: FAIL because the README does not contain the settings-native journey.

- [x] **Step 3: Rewrite and preserve verified facts**

Use this section order: product outcome, Get it, Start in 60 seconds, Configure
with settings, App, Connect clients/API, Tune and benchmark, Modes, Forge,
Advanced and compatibility, What MTPLX is not, License and credit. Keep current
measured performance numbers and citations verbatim; do not invent updated
claims. Use settings-native commands in normal sections and link full details.

- [x] **Step 4: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py tests/test_public_cli.py`

Expected: PASS.

- [x] **Step 5: Commit and update issue**

```bash
git add README.md tests/test_documentation.py
git commit -m "docs: make settings the primary MTPLX workflow"
gh issue comment 90 --repo davidtai/MTPLX --body "Documentation task 4 complete: root README now teaches install/start/settings/connect/tune first and routes advanced/lab detail to focused guides. Commit: $(git rev-parse --short HEAD)."
```

### Task 5: Add Documentation Verification and Run Final Gates

**Files:**
- Create: `scripts/check_documentation.py`
- Modify: `tests/test_documentation.py`
- Modify: `docs/plans/2026-07-16-documentation-refresh.md` (checkboxes only)

**Security flag:** `security`

**Does NOT cover:** Commands requiring a model or privileged hardware; those receive parser-path validation only.

- [x] **Step 1: Add failing checker tests**

```python
from scripts.check_documentation import check_documentation


def test_documentation_checker_accepts_repository_docs():
    report = check_documentation(ROOT)
    assert report.missing_links == ()
    assert report.invalid_shell_blocks == ()
    assert report.unknown_commands == ()
    assert report.legacy_normal_path_flags == ()
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py`

Expected: FAIL because the checker does not exist.

- [x] **Step 3: Implement checker**

Parse local Markdown links and verify targets; extract fenced `bash` blocks;
parse lines with `shlex` after removing comments and variable assignments;
invoke `build_parser().parse_known_args` for no-model `mtplx` commands; exempt
documented placeholders such as filesystem/model paths from existence checks;
and enforce the README legacy-flag policy before the Advanced section. Return a
frozen report and expose `--json`/exit status CLI.

- [x] **Step 4: Verify focused docs and generated references**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_settings_reference.py --check && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_experiment_inventory.py --check && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/check_documentation.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_documentation.py tests/test_settings_reference.py`

Expected: PASS with zero missing links, invalid shell blocks, unknown no-model
commands, or legacy normal-path flags.

- [x] **Step 5: Run Ruff, stub scan, and full suite**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check scripts/generate_settings_reference.py scripts/generate_experiment_inventory.py scripts/check_documentation.py tests/test_documentation.py tests/test_settings_reference.py && ! rg -n 'TODO|FIXME|placeholder|NotImplementedError' scripts/generate_settings_reference.py scripts/generate_experiment_inventory.py scripts/check_documentation.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q`

Expected: exit 0 with no new skips or failures.

- [x] **Step 6: Commit and record final documentation checkpoint**

```bash
git add scripts/check_documentation.py tests/test_documentation.py docs/plans/2026-07-16-documentation-refresh.md
git commit -m "test: enforce documentation and settings-reference drift"
gh issue comment 90 --repo davidtai/MTPLX --body "Documentation phase verified: generated references, experiment inventory, links, shell blocks, no-model parser paths, README policy, Ruff, and full suite pass on $(git rev-parse --short HEAD)."
```
