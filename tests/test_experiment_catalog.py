from __future__ import annotations

from importlib.resources import files

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
    first = _catalog_with(tmp_path / "a", ACTIVE_RECIPE).resolve(
        "lab:compiled-verify-control"
    )
    second = _catalog_with(
        tmp_path / "b", ACTIVE_RECIPE.replace("title =", "title    =")
    ).resolve("lab:compiled-verify-control")
    assert first.sha256 == second.sha256


def test_catalog_refuses_archived_recipe(tmp_path):
    catalog = _catalog_with(
        tmp_path,
        ACTIVE_RECIPE.replace('status = "active"', 'status = "rejected"'),
    )
    with pytest.raises(ValueError, match="rejected.*not executable"):
        catalog.resolve("lab:compiled-verify-control")


def test_catalog_rejects_duplicate_recipe_ids(tmp_path):
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "a.toml").write_text(ACTIVE_RECIPE, encoding="utf-8")
    (recipes / "b.toml").write_text(ACTIVE_RECIPE, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate experiment id"):
        ExperimentCatalog(recipes)


def test_recipe_package_resource_is_discoverable():
    recipes = files("mtplx.experiments").joinpath("recipes")
    assert recipes.is_dir()
