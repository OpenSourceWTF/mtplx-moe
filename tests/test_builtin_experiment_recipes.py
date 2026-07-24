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
        assert dict(recipe.settings) == settings
        assert recipe.owner
        assert recipe.tracking.endswith("/90")
