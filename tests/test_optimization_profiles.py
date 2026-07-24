"""Per-model optimization-profile registry."""
import pytest

from mtplx.optimization_profiles import (
    KNOB_NAMES,
    KnobEntry,
    OptimizationProfile,
    canonical_model_key,
    get_profile,
    not_applicable_violations,
    profile_conflict_warnings,
    resolve_profile_defaults,
)


def test_schema_rejects_bad_state_and_empty_provenance():
    with pytest.raises(ValueError, match="state must be one of"):
        KnobEntry(state="on", value=True, provenance="x")
    with pytest.raises(ValueError, match="provenance"):
        KnobEntry(state="default_on", value=True, provenance="  ")
    with pytest.raises(ValueError, match="unknown knob names"):
        OptimizationProfile(
            model_key="m",
            knobs={"bogus": KnobEntry("default_on", 1, "measured somewhere")},
        )


def test_registry_entries_are_schema_valid():
    profile = get_profile("hy3-oq2e")
    assert profile is not None
    for name, entry in profile.knobs.items():
        assert name in KNOB_NAMES
        assert entry.provenance.strip()


def test_aliases_resolve_to_canonical_key():
    assert canonical_model_key("mlx-community/Hy3-oQ2e") == "hy3-oq2e"
    assert canonical_model_key("hy3-oq2e-r4") == "hy3-oq2e"
    assert resolve_profile_defaults("hy3-oq2e-stock-mtp")


def test_unregistered_model_is_silent():
    assert resolve_profile_defaults("some-other-model") == {}
    assert profile_conflict_warnings("some-other-model", {"draft_core": "x"}) == []
    assert not_applicable_violations("some-other-model", {"kv_quant": "q4"}) == []


def test_conflict_warnings_are_advisory_and_specific():
    warnings = profile_conflict_warnings(
        "hy3-oq2e", {"draft_core": "stock", "speculative_depth": 1}
    )
    assert len(warnings) == 1
    assert "draft_core" in warnings[0] and "device" in warnings[0]
    # matching values and unset knobs warn nothing
    assert profile_conflict_warnings("hy3-oq2e", {"draft_core": "device"}) == []
    assert profile_conflict_warnings("hy3-oq2e", {"draft_core": None}) == []


def test_not_applicable_violation_names_the_measurement():
    violations = not_applicable_violations("hy3-oq2e", {"kv_quant": "q4"})
    assert len(violations) == 1
    assert "kv_quant" in violations[0] and "not_applicable" in violations[0]
    assert not_applicable_violations("hy3-oq2e", {"kv_quant": "off"}) == []
