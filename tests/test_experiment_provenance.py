from __future__ import annotations

from mtplx.cli import build_parser
from mtplx.commands import public
from mtplx.kpi.runtime_kpis import build_benchmark_envelope
from mtplx.settings.argparse import resolve_args_settings


def _envelope(**kwargs):
    return build_benchmark_envelope(
        result={"rows": [], "summary": {}},
        model_inspection={"model": "example"},
        run_id="run-1",
        suite="unit",
        exactness_smoke=None,
        fan_controlled=False,
        strict=False,
        strict_cold=False,
        runtime_profile="sustained",
        **kwargs,
    )


def test_benchmark_envelope_records_redacted_bundle_provenance():
    envelope = _envelope(
        settings={"generation.temperature": 0.6},
        settings_provenance={
            "generation.temperature": {"source": "BUNDLE"}
        },
        settings_bundles=[
            {
                "id": "compiled-verify-control",
                "sha256": "a" * 64,
                "source": "lab:compiled-verify-control",
            }
        ],
    )
    assert envelope["settings"]["bundles"][0]["id"] == "compiled-verify-control"
    assert envelope["settings"]["effective"]["generation.temperature"] == 0.6


def test_benchmark_envelope_omits_settings_for_legacy_callers():
    assert "settings" not in _envelope()


def test_product_benchmark_serializes_resolved_lab_bundle(tmp_path):
    args = build_parser().parse_args(
        ["bench", "run", "--settings", "lab:compiled-verify-control"]
    )
    resolve_args_settings(args, environ={}, user_path=tmp_path / "missing.toml")
    kwargs = public._benchmark_settings_kwargs(args)
    assert kwargs["settings_bundles"][0]["id"] == "compiled-verify-control"
    assert kwargs["settings"]["verify.compiled.mode"] == "off"
