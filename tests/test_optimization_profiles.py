"""Tests for the per-model optimization-profile registry (issue #99, v1).

Covers the four required surfaces:
- registry validation (states, knob coverage, entry types),
- seed-profile content sanity (every entry has provenance; measured winners),
- the read-only benchmark consumer warning,
- the not_applicable enforcement path (pure function + runtime raise).
"""

from __future__ import annotations

import pytest

from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.optimization_profiles import (
    ENFORCEABLE_KNOBS,
    KNOB_NAMES,
    PROFILES,
    VALID_STATES,
    KnobEntry,
    OptimizationProfile,
    get_profile,
    not_applicable_violations,
    profile_conflict_warnings,
    resolve_profile_defaults,
)


# --------------------------------------------------------------------------
# Registry validation
# --------------------------------------------------------------------------


def test_registry_keys_match_profile_model_keys() -> None:
    assert set(PROFILES) == {
        "hy3-expert-q2",
        "hy3-expert-oq2e",
        "glm52-expert-q2",
        "qwen36-27b",
    }
    for key, profile in PROFILES.items():
        assert profile.model_key == key


def test_every_profile_covers_exactly_the_known_knobs() -> None:
    for profile in PROFILES.values():
        assert set(profile.knobs) == set(KNOB_NAMES), profile.model_key


def test_every_entry_has_a_valid_state() -> None:
    for profile in PROFILES.values():
        for name, entry in profile.knobs.items():
            assert entry.state in VALID_STATES, (profile.model_key, name)


def test_knob_entry_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        KnobEntry(state="maybe", value="q4", provenance="x" * 20)


def test_knob_entry_rejects_empty_provenance() -> None:
    with pytest.raises(ValueError):
        KnobEntry(state="default_on", value="q4", provenance="   ")


def test_profile_rejects_missing_knob() -> None:
    partial = {name: KnobEntry("unvalidated", None, "seed") for name in KNOB_NAMES[:-1]}
    with pytest.raises(ValueError, match="missing knob entries"):
        OptimizationProfile(model_key="x", knobs=partial)


def test_profile_rejects_unknown_knob() -> None:
    knobs = {name: KnobEntry("unvalidated", None, "seed") for name in KNOB_NAMES}
    knobs["not_a_knob"] = KnobEntry("unvalidated", None, "seed")
    with pytest.raises(ValueError, match="unknown knobs"):
        OptimizationProfile(model_key="x", knobs=knobs)


def test_profile_knobs_are_read_only() -> None:
    profile = get_profile("hy3-expert-q2")
    assert profile is not None
    with pytest.raises(TypeError):
        profile.knobs["proj_quant"] = KnobEntry("default_off", None, "tamper-attempt")


def test_enforceable_knobs_are_config_fields_and_known_knobs() -> None:
    fields = set(ExpertStreamingConfig.__dataclass_fields__)
    for knob in ENFORCEABLE_KNOBS:
        assert knob in KNOB_NAMES
        assert knob in fields, knob


def test_proj_requant_is_a_known_but_non_enforceable_knob() -> None:
    # The requant experiment knob is registered so every profile must carry a
    # decision for it, but it is deliberately NOT enforced: q8 -> q4 double
    # quantization is a sanctioned experiment, never a structural error.
    assert "proj_requant" in KNOB_NAMES
    assert "proj_requant" not in ENFORCEABLE_KNOBS
    assert "proj_requant" in ExpertStreamingConfig.__dataclass_fields__
    for profile in PROFILES.values():
        assert "proj_requant" in profile.knobs, profile.model_key


def test_oq2e_proj_requant_is_the_pending_experiment_arm() -> None:
    entry = PROFILES["hy3-expert-oq2e"].knobs["proj_requant"]
    assert entry.state == "unvalidated"
    assert entry.value == "q4"
    assert "q8->q4" in entry.provenance
    assert "6.443" in entry.provenance and "0.95" in entry.provenance


# --------------------------------------------------------------------------
# Seed-profile content sanity
# --------------------------------------------------------------------------


def test_every_seed_entry_names_a_measurement_in_provenance() -> None:
    for profile in PROFILES.values():
        for name, entry in profile.knobs.items():
            assert entry.provenance.strip(), (profile.model_key, name)
            # Provenance must be substantive, not a placeholder.
            assert len(entry.provenance.strip()) >= 12, (profile.model_key, name)


def test_not_applicable_entries_carry_no_value() -> None:
    for profile in PROFILES.values():
        for name, entry in profile.knobs.items():
            if entry.state == "not_applicable":
                assert entry.value is None, (profile.model_key, name)


def test_hy3_measured_winners() -> None:
    knobs = PROFILES["hy3-expert-q2"].knobs
    assert knobs["proj_quant"].state == "default_on"
    assert knobs["proj_quant"].value == "q4"
    assert knobs["expert_integrity"].state == "default_on"
    assert knobs["expert_integrity"].value == "headers-only"
    assert knobs["split_route_release"].state == "default_on"
    assert knobs["split_route_release"].value == "deferred"
    assert knobs["island_layer_count"].state == "default_on"
    assert knobs["island_layer_count"].value == 60
    # Depth is config-dependent K2/K3; K3 must be the recorded default value.
    assert knobs["mtp_depth"].value == 3
    # Prefetch mechanism is validated but no end-to-end delta -> unvalidated.
    assert knobs["prefetch_slots"].state == "unvalidated"


def test_glm52_not_applicable_facts() -> None:
    knobs = PROFILES["glm52-expert-q2"].knobs
    # resident/kv-quant NOT applicable, with the why in provenance.
    assert knobs["proj_quant"].state == "not_applicable"
    assert "already ships quantized" in knobs["proj_quant"].provenance
    assert knobs["kv_quant"].state == "not_applicable"
    assert "make_cache ignores" in knobs["kv_quant"].provenance
    # headers-only 1.43x measured; deferred 1.05x measured.
    assert knobs["expert_integrity"].state == "default_on"
    assert "1.43x" in knobs["expert_integrity"].provenance
    assert "1.05x" in knobs["split_route_release"].provenance
    # islands net-negative at 96 GiB -> not applicable (advisory, not enforced).
    assert knobs["island_layer_count"].state == "not_applicable"
    assert "net NEGATIVE" in knobs["island_layer_count"].provenance
    # d3+ depth default.
    assert knobs["mtp_depth"].value == 3


def test_qwen36_mostly_not_applicable_streaming_knobs() -> None:
    knobs = PROFILES["qwen36-27b"].knobs
    streaming_knobs = (
        "expert_integrity",
        "split_route_release",
        "prefetch_slots",
        "miss_shadow",
        "island_source",
        "island_layer_count",
    )
    for knob in streaming_knobs:
        assert knobs[knob].state == "not_applicable", knob
    # kv_quant is a known regression -> default_off, not merely N/A.
    assert knobs["kv_quant"].state == "default_off"
    # K3 depth measured optimal (isolated 58-61 tok/s).
    assert knobs["mtp_depth"].state == "default_on"
    assert knobs["mtp_depth"].value == 3
    assert "58-61" in knobs["mtp_depth"].provenance


# --------------------------------------------------------------------------
# Consumer warning (read-only, warning-only)
# --------------------------------------------------------------------------


def test_resolve_profile_defaults_empty_for_unregistered_model() -> None:
    assert resolve_profile_defaults("no-such-model") == {}
    assert profile_conflict_warnings("no-such-model", {"proj_quant": "q4"}) == []


def test_default_on_conflict_emits_advisory() -> None:
    # hy3 proj_quant default is q4; running with None conflicts.
    lines = profile_conflict_warnings("hy3-expert-q2", {"proj_quant": None})
    assert len(lines) == 1
    assert "proj_quant" in lines[0]
    assert "advisory only" in lines[0]
    # provenance is carried into the advisory.
    assert "d7a695a" in lines[0]


def test_default_on_agreement_is_silent() -> None:
    lines = profile_conflict_warnings("hy3-expert-q2", {"proj_quant": "q4"})
    assert lines == []


def test_sweep_value_satisfies_default_on() -> None:
    # A depth sweep that includes the measured default (3) does not warn.
    assert profile_conflict_warnings("hy3-expert-q2", {"mtp_depth": (1, 2, 3)}) == []
    # A sweep that misses it does warn.
    lines = profile_conflict_warnings("hy3-expert-q2", {"mtp_depth": (1, 4, 5)})
    assert len(lines) == 1
    assert "mtp_depth" in lines[0]


def test_unvalidated_entry_never_warns() -> None:
    # hy3 kv_quant is unvalidated -> no advisory regardless of the value.
    assert profile_conflict_warnings("hy3-expert-q2", {"kv_quant": None}) == []
    assert profile_conflict_warnings("hy3-expert-q2", {"kv_quant": "q4"}) == []


def test_not_applicable_forced_warns_but_unforced_is_silent() -> None:
    # Forcing a not_applicable knob warns...
    lines = profile_conflict_warnings("glm52-expert-q2", {"kv_quant": "q4"})
    assert len(lines) == 1
    assert "not_applicable" in lines[0]
    # ...but leaving it off does not.
    assert profile_conflict_warnings("glm52-expert-q2", {"kv_quant": None}) == []


def test_default_off_warns_only_when_forced_to_a_different_value() -> None:
    # qwen kv_quant is default_off; forcing q4 warns, leaving off is silent.
    assert profile_conflict_warnings("qwen36-27b", {"kv_quant": "q4"})
    assert profile_conflict_warnings("qwen36-27b", {"kv_quant": None}) == []


def test_absent_knob_is_skipped() -> None:
    # Only knobs present in the observed mapping are compared.
    assert profile_conflict_warnings("hy3-expert-q2", {}) == []


# --------------------------------------------------------------------------
# not_applicable enforcement path
# --------------------------------------------------------------------------


def test_not_applicable_violations_reports_forced_enforceable_knob() -> None:
    violations = not_applicable_violations("glm52-expert-q2", {"kv_quant": "q4"})
    assert len(violations) == 1
    assert "kv_quant" in violations[0]
    assert "glm52-expert-q2" in violations[0]


def test_not_applicable_violations_ignores_unforced_and_non_enforceable() -> None:
    # Unforced enforceable knob -> no violation.
    assert not_applicable_violations("glm52-expert-q2", {"kv_quant": None}) == []
    # island_layer_count is not_applicable for glm but NOT enforceable
    # (envelope-conditional): forcing it is advisory, never a hard error.
    assert (
        not_applicable_violations("glm52-expert-q2", {"island_layer_count": 44}) == []
    )


def test_not_applicable_violations_empty_for_default_on_knob() -> None:
    # hy3 proj_quant is default_on, not not_applicable -> never a violation.
    assert not_applicable_violations("hy3-expert-q2", {"proj_quant": "q4"}) == []


def test_not_applicable_violations_empty_for_unregistered_model() -> None:
    assert not_applicable_violations("no-such-model", {"kv_quant": "q4"}) == []


def _minimal_config(**overrides) -> ExpertStreamingConfig:
    base = dict(
        model_key="glm52-expert-q2",
        memory_limit_bytes=64 * 1024**3,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
    )
    base.update(overrides)
    return ExpertStreamingConfig(**base)


def test_runtime_open_raises_on_forced_not_applicable_kv_quant() -> None:
    config = _minimal_config(kv_quant="q4")
    # The profile enforcement runs before any artifact IO, so nonexistent
    # paths still surface the profile error (not a file error).
    with pytest.raises(ExpertStreamingConfigurationError, match="not_applicable"):
        ExpertStreamingRuntime.open(
            "/nonexistent/root", "/nonexistent/manifest.json", config
        )


def test_runtime_open_raises_on_forced_not_applicable_proj_quant() -> None:
    config = _minimal_config(proj_quant="q4")
    with pytest.raises(
        ExpertStreamingConfigurationError, match="proj_quant.*not_applicable"
    ):
        ExpertStreamingRuntime.open(
            "/nonexistent/root", "/nonexistent/manifest.json", config
        )


def test_runtime_open_does_not_raise_profile_error_when_knobs_off() -> None:
    # With the not_applicable knobs left off, the profile check passes and the
    # failure (if any) comes from the missing artifacts, not the profile.
    config = _minimal_config()
    with pytest.raises(Exception) as excinfo:
        ExpertStreamingRuntime.open(
            "/nonexistent/root", "/nonexistent/manifest.json", config
        )
    assert "not_applicable" not in str(excinfo.value)


def test_proj_requant_never_appears_in_not_applicable_violations() -> None:
    # proj_requant is not enforceable, so even where a profile marks it (it is
    # unvalidated everywhere) forcing it can never produce a hard violation.
    assert not_applicable_violations("hy3-expert-oq2e", {"proj_requant": "q4"}) == []
    assert not_applicable_violations("hy3-expert-q2", {"proj_requant": "q4"}) == []


def test_oq2e_proj_quant_still_enforced_while_proj_requant_passes_clean() -> None:
    # The oq2e directive is: never proj_quant (matches zero BF16 Linears), but
    # proj_requant q8 -> q4 is the sanctioned experiment. Forcing proj_quant
    # must still raise; forcing proj_requant must not.
    forced_proj_quant = _minimal_config(model_key="hy3-expert-oq2e", proj_quant="q4")
    with pytest.raises(
        ExpertStreamingConfigurationError, match="proj_quant.*not_applicable"
    ):
        ExpertStreamingRuntime.open(
            "/nonexistent/root", "/nonexistent/manifest.json", forced_proj_quant
        )

    clean_requant = _minimal_config(model_key="hy3-expert-oq2e", proj_requant="q4")
    with pytest.raises(Exception) as excinfo:
        ExpertStreamingRuntime.open(
            "/nonexistent/root", "/nonexistent/manifest.json", clean_requant
        )
    # Fails on missing artifacts, NOT on a profile/not_applicable violation.
    assert "not_applicable" not in str(excinfo.value)
