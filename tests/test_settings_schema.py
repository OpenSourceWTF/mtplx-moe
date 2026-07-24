from __future__ import annotations

import pytest

from mtplx.settings.schema import (
    Lifecycle,
    SettingAlias,
    SettingSpec,
    SettingType,
    Visibility,
)


def test_setting_spec_parses_bool_int_float_and_choice():
    enabled = SettingSpec("runtime.mtp.enabled", SettingType.BOOL, False)
    depth = SettingSpec("runtime.mtp.depth", SettingType.INT, 3, minimum=1)
    temperature = SettingSpec(
        "generation.temperature", SettingType.FLOAT, 0.6, minimum=0.0
    )
    profile = SettingSpec(
        "runtime.profile",
        SettingType.STRING,
        "sustained",
        choices=("stable", "sustained", "turbo"),
    )
    assert enabled.parse("yes") is True
    assert depth.parse("4") == 4
    assert temperature.parse("0.7") == 0.7
    assert profile.parse("turbo") == "turbo"


def test_setting_spec_rejects_invalid_values_with_canonical_name():
    spec = SettingSpec("runtime.mtp.depth", SettingType.INT, 3, minimum=1)
    with pytest.raises(ValueError, match=r"runtime\.mtp\.depth.*>= 1"):
        spec.parse("0")


def test_alias_metadata_is_orthogonal_to_visibility_and_lifecycle():
    spec = SettingSpec(
        "runtime.profile",
        SettingType.STRING,
        "sustained",
        visibility=Visibility.PUBLIC,
        lifecycle=Lifecycle.ACTIVE,
        aliases=(
            SettingAlias("profile", "cli"),
            SettingAlias("MTPLX_PROFILE", "env"),
        ),
    )
    assert spec.aliases[0].source == "cli"
    assert spec.visibility is Visibility.PUBLIC
