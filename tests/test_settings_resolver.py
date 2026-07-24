from __future__ import annotations

import pytest

from mtplx.config import CONFIG_VALUE_KEYS
from mtplx.settings.builtins import BUILTIN_SETTINGS
from mtplx.settings.catalog import SettingCatalog
from mtplx.settings.resolver import SettingSource, SettingsResolver
from mtplx.settings.schema import SettingSpec, SettingType


def test_catalog_rejects_duplicate_canonical_names_and_aliases():
    spec = SettingSpec("runtime.profile", SettingType.STRING, "sustained")
    with pytest.raises(ValueError, match="duplicate setting"):
        SettingCatalog((spec, spec))


def test_catalog_suggests_close_canonical_name():
    catalog = SettingCatalog(BUILTIN_SETTINGS)
    assert catalog.suggest("generation.temprature")[0] == "generation.temperature"


def test_product_catalog_covers_every_legacy_config_key_once():
    config_aliases = {
        alias.name
        for spec in BUILTIN_SETTINGS
        for alias in spec.aliases
        if alias.source == "config"
    }
    assert len(BUILTIN_SETTINGS) == 33
    assert config_aliases == set(CONFIG_VALUE_KEYS)


def test_resolver_uses_documented_source_order_and_keeps_provenance():
    catalog = SettingCatalog(BUILTIN_SETTINGS)
    resolved = SettingsResolver(catalog).resolve(
        {
            SettingSource.PROFILE: {"generation.temperature": 0.5},
            SettingSource.USER: {"generation.temperature": 0.6},
            SettingSource.BUNDLE: {"generation.temperature": 0.7},
            SettingSource.CLI_SET: {"generation.temperature": 0.8},
        }
    )
    assert resolved["generation.temperature"] == 0.8
    record = resolved.provenance["generation.temperature"]
    assert record.source is SettingSource.CLI_SET
    assert [item.source for item in record.shadowed] == [
        SettingSource.BUNDLE,
        SettingSource.USER,
        SettingSource.PROFILE,
        SettingSource.BUILTIN,
    ]


def test_resolver_redacts_secret_provenance():
    catalog = SettingCatalog(
        (SettingSpec("server.api_key", SettingType.STRING, "", secret=True),)
    )
    resolved = SettingsResolver(catalog).resolve(
        {
            SettingSource.USER: {"server.api_key": "old-secret"},
            SettingSource.CLI_SET: {"server.api_key": "top-secret"},
        }
    )
    record = resolved.provenance["server.api_key"]
    assert resolved["server.api_key"] == "top-secret"
    assert record.display_value == "[redacted]"
    assert [item.value for item in record.shadowed] == [
        "[redacted]",
        "",
    ]

    constrained = resolved.with_constraints(
        {"server.api_key": ("policy-secret", "test policy")}
    )
    constrained_record = constrained.provenance["server.api_key"]
    assert constrained["server.api_key"] == "policy-secret"
    assert constrained_record.display_value == "[redacted]"
    assert constrained_record.requested_value == "[redacted]"
    assert [item.value for item in constrained_record.shadowed] == [
        "[redacted]",
        "[redacted]",
        "",
    ]
