from __future__ import annotations

from pathlib import Path

from scripts.audit_settings_catalog import audit_source_settings


def test_every_production_mtplx_name_is_classified():
    root = Path(__file__).resolve().parents[1]
    report = audit_source_settings(root / "mtplx")
    assert report.unclassified == ()
    assert report.duplicate_aliases == ()


def test_new_direct_setting_reads_are_confined_to_compatibility_boundary():
    root = Path(__file__).resolve().parents[1]
    report = audit_source_settings(root / "mtplx")
    assert report.unauthorized_direct_reads == ()
