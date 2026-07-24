from __future__ import annotations

from types import SimpleNamespace

from mtplx.commands import public
from mtplx.settings.argparse import resolve_args_settings
from mtplx.settings.builtins import default_setting_catalog
from mtplx.settings.resolver import SettingSource, SettingsResolver


def test_constraint_keeps_requested_and_effective_values():
    resolved = SettingsResolver(default_setting_catalog()).resolve(
        {SettingSource.CLI_SET: {"runtime.mtp.enabled": True}}
    )
    constrained = resolved.with_constraints(
        {"runtime.mtp.enabled": (False, "model has no compatible MTP heads")}
    )
    assert constrained["runtime.mtp.enabled"] is False
    record = constrained.provenance["runtime.mtp.enabled"]
    assert record.source is SettingSource.CONSTRAINT
    assert record.requested_value is True
    assert record.reason == "model has no compatible MTP heads"


def test_compatible_value_is_not_relabelled_as_constraint():
    resolved = SettingsResolver(default_setting_catalog()).resolve(
        {SettingSource.CLI_SET: {"runtime.mtp.depth": 2}}
    )
    constrained = resolved.with_constraints(
        {"runtime.mtp.depth": (2, "model maximum depth is 2")}
    )
    assert (
        constrained.provenance["runtime.mtp.depth"].source
        is SettingSource.CLI_SET
    )


def test_existing_model_depth_default_is_recorded_as_constraint(tmp_path):
    args = SimpleNamespace(
        command="serve",
        profile="sustained",
        depth=3,
        no_mtp=False,
        setting_overrides=[],
        settings_bundles=[],
        _cli_flags=set(),
    )
    resolve_args_settings(args, environ={}, user_path=tmp_path / "missing.toml")

    public._apply_model_contract_depth_default(
        args,
        {
            "compatibility": {
                "runtime_contract": {"mtp_depth_max": 2},
            }
        },
        SimpleNamespace(name="sustained"),
    )

    assert args.depth == 2
    record = args.mtplx_settings.provenance["runtime.mtp.depth"]
    assert record.source is SettingSource.CONSTRAINT
    assert record.requested_value == 3
    assert record.reason == "model maximum MTP depth is 2"


def test_existing_failed_model_gate_records_disabled_mtp(tmp_path):
    args = SimpleNamespace(
        command="serve",
        profile="sustained",
        depth=3,
        no_mtp=False,
        setting_overrides=["runtime.mtp.enabled=true"],
        settings_bundles=[],
        _cli_flags={"set"},
    )
    resolve_args_settings(args, environ={}, user_path=tmp_path / "missing.toml")

    public._record_model_gate_constraint(
        args,
        {
            "compatibility": {
                "can_run": False,
                "message": "model has no compatible MTP heads",
            }
        },
    )

    record = args.mtplx_settings.provenance["runtime.mtp.enabled"]
    assert record.source is SettingSource.CONSTRAINT
    assert record.requested_value is True
    assert record.display_value is False
    assert record.reason == "model has no compatible MTP heads"
