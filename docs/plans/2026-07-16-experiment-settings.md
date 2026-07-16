# Experiment Settings Bundles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace active experiment flag clusters with validated, attributable settings bundles and an explicit lab lifecycle.

**Architecture:** A data-only experiment schema loads TOML metadata and settings through the settings catalog. A built-in catalog exposes active recipes by `lab:` URI and refuses archived, expired, or model-incompatible recipes. Lab commands inspect and validate recipes without MLX, while benchmark envelopes record bundle hashes and resolved settings.

**Tech Stack:** Python stdlib `dataclasses`, `datetime`, `hashlib`, `json`, `tomllib`, `pathlib`; pytest; existing benchmark envelope APIs.

**Assumptions:**

- Assumes the hierarchical settings plan has passed — will NOT duplicate setting parsing or precedence in the experiment package.
- Assumes built-in recipes are data only — will NOT execute shell, Python, substitutions, or file-write directives.
- Assumes active recipes are explicit controls supported by the audited branch — will NOT resurrect rejected GitHub experiments as executable toggles.
- Assumes hardware runs remain separately authorized and isolated — will NOT execute a benchmark merely because a recipe validates.

---

## File Structure

- `mtplx/experiments/schema.py` — immutable metadata and recipe validation.
- `mtplx/experiments/catalog.py` — built-in discovery, lifecycle, normalized hash, and `lab:` resolution.
- `mtplx/experiments/recipes/*.toml` — active data-only recipes.
- `mtplx/commands/lab.py` — list/show/validate commands.
- `mtplx/settings/bundles.py` — delegates `lab:` URIs to the experiment catalog.
- `mtplx/kpi/runtime_kpis.py` — adds recipe provenance to envelopes.
- `docs/experiments/inventory.md` — generated grouped experiment inventory.
- `tests/test_experiment_schema.py`, `tests/test_experiment_catalog.py`, `tests/test_lab_cli.py`, `tests/test_experiment_provenance.py` — lifecycle and integration tests.

### Task 1: Add Data-only Experiment Schema

**Files:**
- Create: `mtplx/experiments/__init__.py`
- Create: `mtplx/experiments/schema.py`
- Create: `tests/test_experiment_schema.py`

**Security flag:** `security`

**Does NOT cover:** Built-in discovery, CLI, settings resolution, or benchmark execution.

- [x] **Step 1: Write failing schema tests**

```python
from __future__ import annotations

from datetime import date

import pytest

from mtplx.experiments.schema import ExperimentStatus, load_experiment


def _recipe(tmp_path, text: str):
    path = tmp_path / "recipe.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_complete_active_recipe(tmp_path):
    recipe = load_experiment(_recipe(tmp_path, '''
[experiment]
id = "compiled-verify-control"
title = "Compiled verify disabled control"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Isolate compiled verify."
[settings]
"verify.compiled.mode" = "off"
'''))
    assert recipe.status is ExperimentStatus.ACTIVE
    assert recipe.settings == {"verify.compiled.mode": "off"}


def test_rejects_executable_content(tmp_path):
    path = _recipe(tmp_path, '''
[experiment]
id = "unsafe"
title = "Unsafe"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Unsafe fixture."
[settings]
"runtime.profile" = "sustained"
[shell]
command = "echo unsafe"
''')
    with pytest.raises(ValueError, match="unsupported top-level table: shell"):
        load_experiment(path)


def test_active_recipe_must_have_future_review_date(tmp_path):
    path = _recipe(tmp_path, '''
[experiment]
id = "expired"
title = "Expired"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-01-01"
review_after = "2026-01-02"
models = ["qwen3-next"]
purpose = "Expired fixture."
[settings]
"runtime.profile" = "sustained"
''')
    with pytest.raises(ValueError, match="review date has passed"):
        load_experiment(path, today=date(2026, 7, 16))
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_schema.py`

Expected: FAIL because `mtplx.experiments` does not exist.

- [x] **Step 3: Implement schema and strict loader**

Define `ExperimentStatus` with `active`, `retained`, `rejected`, `superseded`,
and `expired`; an immutable `ExperimentRecipe`; required metadata validation;
ISO date parsing; scalar settings only; exact allowed top-level tables
`experiment` and `settings`; and active review-date enforcement. Archived
statuses may include `replacement` and `result` but are never executable.

- [x] **Step 4: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_schema.py`

Expected: PASS.

- [x] **Step 5: Commit and update issue**

```bash
git add mtplx/experiments tests/test_experiment_schema.py
git commit -m "feat: define experiment settings schema"
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment task 1 complete: strict data-only recipe schema and lifecycle validation landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 2: Add Built-in Catalog, Lifecycle Refusal, and Stable Hashes

**Files:**
- Create: `mtplx/experiments/catalog.py`
- Create: `tests/test_experiment_catalog.py`
- Modify: `mtplx/experiments/__init__.py`
- Modify: `pyproject.toml`

**Security flag:** `security`

**Does NOT cover:** Applying recipes to runtime commands or model loading.

- [x] **Step 1: Write failing catalog tests**

```python
from __future__ import annotations

import pytest

from mtplx.experiments.catalog import ExperimentCatalog


ACTIVE_RECIPE = '''
[experiment]
id = "compiled-verify-control"
title = "Compiled verify disabled control"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Isolate compiled verify."
[settings]
"verify.compiled.mode" = "off"
'''


def _catalog_with(root, text):
    recipes = root / "recipes"
    recipes.mkdir(parents=True)
    (recipes / "control.toml").write_text(text, encoding="utf-8")
    return ExperimentCatalog(recipes)


def test_catalog_resolves_active_lab_uri(tmp_path):
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    path = recipes / "control.toml"
    path.write_text(ACTIVE_RECIPE, encoding="utf-8")
    catalog = ExperimentCatalog(recipes)
    resolved = catalog.resolve("lab:compiled-verify-control")
    assert resolved.recipe.id == "compiled-verify-control"
    assert len(resolved.sha256) == 64


def test_catalog_hash_is_stable_across_toml_whitespace(tmp_path):
    first = _catalog_with(tmp_path / "a", ACTIVE_RECIPE).resolve("lab:compiled-verify-control")
    second = _catalog_with(tmp_path / "b", ACTIVE_RECIPE.replace("title =", "title    =")).resolve("lab:compiled-verify-control")
    assert first.sha256 == second.sha256


def test_catalog_refuses_archived_recipe(tmp_path):
    catalog = _catalog_with(tmp_path, ACTIVE_RECIPE.replace('status = "active"', 'status = "rejected"'))
    with pytest.raises(ValueError, match="rejected.*not executable"):
        catalog.resolve("lab:compiled-verify-control")
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_catalog.py`

Expected: FAIL because catalog does not exist.

- [x] **Step 3: Implement catalog and normalized hashing**

Discover sorted `*.toml`, reject duplicate ids, list by lifecycle, resolve only
named `lab:` URIs such as `lab:compiled-verify-control`, and hash canonical JSON consisting of sorted metadata and settings
with compact separators. `resolve` returns an immutable object containing
recipe, hash, and source path.

- [x] **Step 4: Package recipes**

Add `"mtplx.experiments" = ["recipes/*.toml"]` under
`[tool.setuptools.package-data]` and add an installation test that uses
`importlib.resources.files("mtplx.experiments").joinpath("recipes")`.

- [x] **Step 5: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_catalog.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/experiments pyproject.toml tests/test_experiment_catalog.py
git commit -m "feat: catalog executable experiment bundles"
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment task 2 complete: built-in catalog, lifecycle refusal, package data, and stable recipe hashes landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 3: Add No-MLX Lab Commands

**Files:**
- Create: `mtplx/commands/lab.py`
- Create: `tests/test_lab_cli.py`
- Modify: `mtplx/cli.py`

**Security flag:** `security`

**Does NOT cover:** Executing a benchmark, mutating settings, or enabling archived recipes.

- [x] **Step 1: Write failing CLI tests**

```python
from __future__ import annotations

import json

from mtplx.cli import main


VALID_RECIPE = '''
[experiment]
id = "compiled-verify-control"
title = "Compiled verify disabled control"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Isolate compiled verify."
[settings]
"verify.compiled.mode" = "off"
'''


def test_lab_list_is_json_and_no_mlx(capsys):
    assert main(["lab", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(item["status"] == "active" for item in payload["experiments"])


def test_lab_show_includes_hash_and_settings(capsys):
    assert main(["lab", "show", "compiled-verify-control", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "compiled-verify-control"
    assert len(payload["sha256"]) == 64
    assert payload["settings"]["verify.compiled.mode"] == "off"


def test_lab_validate_rejects_unknown_setting(capsys, tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(VALID_RECIPE.replace("verify.compiled.mode", "verify.compiled.typo"), encoding="utf-8")
    assert main(["lab", "validate", str(path), "--json"]) == 2
    assert "unknown setting" in capsys.readouterr().out
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_lab_cli.py`

Expected: FAIL because the lab command is unknown.

- [x] **Step 3: Implement parser and handlers**

Register `lab list [--all] [--json]`, `lab show ID [--json]`, and
`lab validate PATH [--json]`. Handlers load schema/catalog/settings catalog,
validate every setting name/type/tier, produce human or JSON output, and import
no runtime/model modules.

- [x] **Step 4: Verify GREEN and no-MLX import**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_lab_cli.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -c 'import sys; from mtplx.cli import main; assert main(["lab", "list", "--json"]) == 0; assert "mlx" not in sys.modules'`

Expected: PASS and no `mlx` module in `sys.modules`.

- [x] **Step 5: Commit and update issue**

```bash
git add mtplx/cli.py mtplx/commands/lab.py tests/test_lab_cli.py
git commit -m "feat: add experiment lab inspection commands"
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment task 3 complete: no-MLX lab list/show/validate commands landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 4: Resolve `lab:` URIs Through Generic Settings Bundles

**Files:**
- Modify: `mtplx/settings/bundles.py`
- Modify: `mtplx/settings/argparse.py`
- Create: `tests/test_experiment_settings_integration.py`

**Security flag:** `security`

**Does NOT cover:** Automatic model loading or bypassing recipe model constraints.

- [x] **Step 1: Write failing integration tests**

```python
from __future__ import annotations

from mtplx.cli import build_parser
from mtplx.settings.argparse import resolve_args_settings


def test_lab_uri_uses_same_bundle_precedence_as_file_bundle():
    args = build_parser().parse_args([
        "serve",
        "--settings", "lab:compiled-verify-control",
        "--set", "verify.compiled.mode=on",
    ])
    resolved = resolve_args_settings(args, environ={})
    assert resolved["verify.compiled.mode"] == "on"
    assert resolved.bundle_provenance[0].id == "compiled-verify-control"


def test_lab_uri_refuses_incompatible_model_family():
    args = build_parser().parse_args([
        "serve", "--model", "gemma4/example", "--settings", "lab:compiled-verify-control"
    ])
    try:
        resolve_args_settings(args, environ={}, model_family="gemma4")
    except ValueError as exc:
        assert "supports qwen3-next" in str(exc)
    else:
        raise AssertionError("model-incompatible recipe should fail")
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_settings_integration.py`

Expected: FAIL because `lab:` is treated as a missing filesystem path and
bundle provenance is absent.

- [x] **Step 3: Implement lab URI delegation and provenance**

`load_settings_bundle` returns a `LoadedSettingsBundle` with settings, source,
optional id/hash, and experiment metadata. For `lab:` sources it delegates to
`ExperimentCatalog.resolve`, validates model family when known, and preserves
metadata in `ResolvedSettings.bundle_provenance`. File bundles keep source path
and a normalized settings hash.

- [x] **Step 4: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_settings_integration.py tests/test_settings_storage.py tests/test_runtime_settings_args.py`

Expected: PASS.

- [x] **Step 5: Commit and update issue**

```bash
git add mtplx/settings tests/test_experiment_settings_integration.py
git commit -m "feat: resolve lab recipes as settings bundles"
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment task 4 complete: lab URIs use the canonical settings resolver with hashes, model constraints, and source provenance. Commit: $(git rev-parse --short HEAD)."
```

### Task 5: Add Active Control Recipes and Inventory

**Files:**
- Create: `mtplx/experiments/recipes/compiled-verify-control.toml`
- Create: `mtplx/experiments/recipes/nax-verify-control.toml`
- Create: `mtplx/experiments/recipes/packed-gqa-control.toml`
- Create: `scripts/generate_experiment_inventory.py`
- Create: `docs/experiments/inventory.md`
- Create: `tests/test_builtin_experiment_recipes.py`

**Security flag:** `none`

**Does NOT cover:** Promoting candidates, assigning performance claims, or creating executable recipes for rejected experiments.

- [x] **Step 1: Write failing built-in recipe tests**

```python
from __future__ import annotations

from mtplx.experiments.catalog import default_experiment_catalog


def test_builtin_controls_are_active_typed_and_owned():
    catalog = default_experiment_catalog()
    expected = {
        "compiled-verify-control": {"verify.compiled.mode": "off"},
        "nax-verify-control": {"verify.nax.enabled": False},
        "packed-gqa-control": {"attention.gqa_packed_sdpa.enabled": False},
    }
    assert {item.id for item in catalog.list(active_only=True)} == set(expected)
    for experiment_id, settings in expected.items():
        recipe = catalog.resolve(f"lab:{experiment_id}").recipe
        assert recipe.settings == settings
        assert recipe.owner
        assert recipe.tracking.endswith("/90")
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_builtin_experiment_recipes.py`

Expected: FAIL because built-in recipes do not exist.

- [x] **Step 3: Add the three control recipes**

Each recipe uses issue #90 as tracking, creation date 2026-07-16, review date
2026-08-16, model family `qwen3-next`, status `active`, owner `runtime`, and a
purpose that explicitly describes it as a control. Register canonical settings
and current environment aliases:

```text
verify.compiled.mode               <- MTPLX_COMPILED_VERIFY
verify.nax.enabled                 <- MTPLX_NAX_VERIFY
attention.gqa_packed_sdpa.enabled <- MTPLX_GQA_PACKED_SDPA
```

- [x] **Step 4: Generate the complete grouped inventory**

`generate_experiment_inventory.py` reads the settings catalog and built-in lab
catalog, groups experimental/compatibility environment aliases by domain and
lifecycle, identifies whether each is used by an active bundle, and writes
deterministic Markdown. It has `--write` and `--check` modes.

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_experiment_inventory.py --write`

Expected: creates `docs/experiments/inventory.md` with every experimental or
compatibility setting classified; no setting is silently omitted.

- [x] **Step 5: Verify GREEN and generated drift**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_builtin_experiment_recipes.py tests/test_experiment_catalog.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_experiment_inventory.py --check`

Expected: PASS.

- [x] **Step 6: Comment newly identified cleanup candidates**

For each inventory row with no active bundle/profile/test and lifecycle
`compatibility`, add one consolidated issue #90 comment listing the setting,
source locations, apparent last purpose, and recommendation: archive, retain as
internal, or investigate. Do not remove it in this task.

- [x] **Step 7: Commit and update issue**

```bash
git add mtplx/experiments/recipes mtplx/settings/builtins.py scripts/generate_experiment_inventory.py docs/experiments/inventory.md tests/test_builtin_experiment_recipes.py
git commit -m "feat: organize active experiments as settings bundles"
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment task 5 complete: active controls are named lab bundles and the complete experiment/compatibility inventory is generated and grouped. Commit: $(git rev-parse --short HEAD)."
```

### Task 6: Record Bundle Provenance in Benchmark Envelopes

**Files:**
- Modify: `mtplx/kpi/runtime_kpis.py`
- Modify: `mtplx/commands/public.py`
- Create: `tests/test_experiment_provenance.py`

**Security flag:** `security`

**Does NOT cover:** Changing benchmark scoring, hardware isolation, model loading, or benchmark claims.

- [x] **Step 1: Write failing provenance test**

```python
from __future__ import annotations

from mtplx.kpi.runtime_kpis import build_benchmark_envelope


def test_benchmark_envelope_records_redacted_bundle_provenance():
    envelope = build_benchmark_envelope(
        result={"rows": [], "summary": {}},
        model_inspection={"model": "example"},
        run_id="run-1",
        suite="unit",
        exactness_smoke=None,
        fan_controlled=False,
        strict=False,
        strict_cold=False,
        runtime_profile="sustained",
        settings={"generation.temperature": 0.6},
        settings_provenance={"generation.temperature": {"source": "BUNDLE"}},
        settings_bundles=[{
            "id": "compiled-verify-control",
            "sha256": "a" * 64,
            "source": "lab:compiled-verify-control",
        }],
    )
    assert envelope["settings"]["bundles"][0]["id"] == "compiled-verify-control"
    assert envelope["settings"]["effective"]["generation.temperature"] == 0.6
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_provenance.py`

Expected: FAIL because the envelope has no settings provenance fields.

- [x] **Step 3: Extend envelope schema and benchmark call sites**

Add optional `settings`, `settings_provenance`, and `settings_bundles` keyword
arguments. Emit a sorted `settings` object only when provided, redact schema-
marked secrets before the call, and pass `args.mtplx_settings` from product
benchmark actions. Preserve the byte shape of envelopes from callers that do
not supply settings.

- [x] **Step 4: Verify GREEN and existing KPI tests**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_provenance.py tests/test_runtime_kpis.py tests/test_public_cli.py`

Expected: PASS and unchanged payloads for no-settings callers.

- [x] **Step 5: Commit and update issue**

```bash
git add mtplx/kpi/runtime_kpis.py mtplx/commands/public.py tests/test_experiment_provenance.py
git commit -m "feat: record settings bundles in benchmark provenance"
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment task 6 complete: benchmark envelopes record resolved settings and exact bundle ids/hashes without changing scoring. Commit: $(git rev-parse --short HEAD)."
```

### Task 7: Verify the Experiment Settings Phase

**Files:**
- Modify: `docs/plans/2026-07-16-experiment-settings.md` (checkboxes only)
- Test: experiment/settings/benchmark tests and full suite

**Security flag:** `security`

- [x] **Step 1: Run focused verification**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_experiment_schema.py tests/test_experiment_catalog.py tests/test_lab_cli.py tests/test_experiment_settings_integration.py tests/test_builtin_experiment_recipes.py tests/test_experiment_provenance.py tests/test_runtime_settings_args.py`

Expected: PASS.

- [x] **Step 2: Run generated checks and Ruff**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/audit_settings_catalog.py --check && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/generate_experiment_inventory.py --check && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check mtplx/experiments mtplx/commands/lab.py tests/test_experiment_*.py tests/test_lab_cli.py scripts/generate_experiment_inventory.py`

Expected: PASS.

- [x] **Step 3: Run full suite and stub scan**

Run: `! rg -n 'TODO|FIXME|placeholder|NotImplementedError' mtplx/experiments mtplx/commands/lab.py scripts/generate_experiment_inventory.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q`

Expected: exit 0 with no new skips or failures.

- [x] **Step 4: Record checkpoint**

```bash
gh issue comment 90 --repo davidtai/MTPLX --body "Experiment settings phase verified: schema/catalog/lab/bundle/provenance tests, generated inventory checks, Ruff, and full repository suite pass on $(git rev-parse --short HEAD)."
```
