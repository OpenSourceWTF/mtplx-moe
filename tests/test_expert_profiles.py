import json
import subprocess
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import mtplx.expert_profiles as expert_profiles
from mtplx.expert_profiles import (
    build_expert_streaming_config,
    load_expert_profiles,
    profile_required_bytes,
    select_expert_profile,
)


GiB = 1024**3


def _profile_resource_document():
    resource = expert_profiles.files("mtplx").joinpath(
        "data/expert_profiles.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _parse_profile_document(document):
    return expert_profiles._parse_expert_profiles_resource(
        json.dumps(document)
    )


def test_only_promoted_oq2e_profiles_are_installed():
    assert set(load_expert_profiles()) == {
        "hy3-oq2e-64",
        "hy3-oq2e-88",
        "hy3-oq2e-96",
    }


def test_loaded_profiles_are_immutable():
    profiles = load_expert_profiles()
    profile = profiles["hy3-oq2e-64"]

    with pytest.raises(TypeError):
        profiles["candidate"] = profile
    with pytest.raises(TypeError):
        profile.config["memory_limit_bytes"] = 1
    with pytest.raises(TypeError):
        profile.child_env["MTPLX_SUSTAINED_PREFILL"] = "0"
    with pytest.raises(FrozenInstanceError):
        profile.name = "candidate"


def test_auto_selects_largest_profile_that_passes_both_memory_gates():
    # 97 GiB is under hy3-oq2e-96's 103 GiB declared ceiling but over the
    # 91.99 GiB its plan reaches, and admission compares the latter.
    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=128 * GiB,
        available_bytes=97 * GiB,
    )
    assert selected.name == "hy3-oq2e-96"


def test_auto_falls_to_the_next_profile_below_the_largest_realized_footprint():
    largest = load_expert_profiles()["hy3-oq2e-96"]
    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=128 * GiB,
        available_bytes=profile_required_bytes(largest) - GiB,
    )
    assert selected.name == "hy3-oq2e-88"


def test_auto_selection_observes_the_installed_ram_gate():
    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=87 * GiB,
        available_bytes=128 * GiB,
    )
    assert selected.name == "hy3-oq2e-64"


def test_auto_selection_measures_memory_once_at_construction(monkeypatch):
    calls = {"installed": 0, "available": 0}

    def installed_ram_bytes():
        calls["installed"] += 1
        return 128 * GiB

    def available_memory_bytes():
        calls["available"] += 1
        return 97 * GiB

    monkeypatch.setattr(
        expert_profiles, "_installed_ram_bytes", installed_ram_bytes
    )
    monkeypatch.setattr(
        expert_profiles, "available_memory_bytes", available_memory_bytes
    )

    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
    )

    assert selected.name == "hy3-oq2e-96"
    assert calls == {"installed": 1, "available": 1}


def test_64_profile_installs_measured_cache_heavy_geometry(monkeypatch):
    profile = load_expert_profiles()["hy3-oq2e-64"]
    monkeypatch.setattr(
        expert_profiles,
        "resolve_island_placement",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-island profile must not infer islands"
        ),
    )

    config = build_expert_streaming_config(profile)

    assert profile.process_ceiling_bytes == 71 * GiB
    assert profile.weight_envelope_bytes == 64 * GiB
    assert profile.generation_mode == "ar"
    assert profile.evidence_receipts == (
        "evals/tier2/t3_64x16k_armB_frequency.json",
        "research/envelope-admission-sweep-2026-07-22.json",
    )
    assert config.memory_limit_bytes == 71 * GiB
    assert config.runtime_reserve_bytes == 7 * GiB
    assert config.expert_cache_limit_bytes == 53_678_702_592
    assert config.max_live_kv_tokens == 4096
    assert config.island_layers == ()
    assert config.island_layer_count is None
    assert config.proj_requant == "q4"
    assert config.verify_record_hashes is False
    assert config.verify_sidecar_hash_at_open is False


def test_88_and_96_profiles_install_exact_island_counts():
    profiles = load_expert_profiles()

    config_88 = build_expert_streaming_config(profiles["hy3-oq2e-88"])
    assert profiles["hy3-oq2e-88"].process_ceiling_bytes == 95 * GiB
    assert profiles["hy3-oq2e-88"].weight_envelope_bytes == 88 * GiB
    assert config_88.island_layer_count == 74
    assert len(config_88.island_layers) == 74
    assert config_88.expert_cache_limit_bytes == 2 * GiB
    assert config_88.split_route_release == "deferred"
    assert config_88.deferred_pin_release is True

    config_96 = build_expert_streaming_config(profiles["hy3-oq2e-96"])
    assert profiles["hy3-oq2e-96"].process_ceiling_bytes == 103 * GiB
    assert profiles["hy3-oq2e-96"].weight_envelope_bytes == 96 * GiB
    assert config_96.island_layer_count == 79
    assert len(config_96.island_layers) == 79
    assert config_96.expert_cache_limit_bytes == 2 * GiB
    assert config_96.split_route_release == "fenced"
    assert config_96.deferred_pin_release is False


def test_profile_overrides_normalize_memory_values():
    profile = load_expert_profiles()["hy3-oq2e-64"]

    config = build_expert_streaming_config(
        profile,
        overrides={
            "memory_limit_bytes": "70GiB",
            "expert_cache_limit_bytes": "50GiB",
        },
    )

    assert config.memory_limit_bytes == 70 * GiB
    assert config.expert_cache_limit_bytes == 50 * GiB


def test_profile_override_cannot_exceed_admitted_process_ceiling():
    profile = load_expert_profiles()["hy3-oq2e-64"]

    with pytest.raises(ValueError) as excinfo:
        build_expert_streaming_config(
            profile,
            overrides={"memory_limit_bytes": "192GiB"},
        )

    message = str(excinfo.value)
    assert "memory_limit_bytes" in message
    assert str(192 * GiB) in message
    assert str(profile.process_ceiling_bytes) in message


def test_profile_override_cannot_replace_model_identity():
    profile = load_expert_profiles()["hy3-oq2e-64"]

    with pytest.raises(ValueError, match="model_key"):
        build_expert_streaming_config(
            profile,
            overrides={"model_key": "hy3-expert-q2"},
        )


def test_profile_resource_rejects_duplicate_json_keys():
    resource = """\
{"schema": 1, "schema": 1, "profiles": []}
"""

    with pytest.raises(ValueError, match="duplicate JSON key 'schema'"):
        expert_profiles._parse_expert_profiles_resource(resource)


def test_profile_resource_rejects_config_ceiling_mismatch():
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["memory_limit_bytes"] -= 1

    with pytest.raises(ValueError, match="config.memory_limit_bytes"):
        _parse_profile_document(document)


def test_profile_resource_rejects_weight_and_reserve_mismatch():
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["runtime_reserve_bytes"] -= 1

    with pytest.raises(ValueError, match="weight_envelope_bytes"):
        _parse_profile_document(document)


def test_profile_resource_rejects_config_identity_shadowing():
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["model_key"] = "hy3-expert-q2"

    with pytest.raises(ValueError, match="config.*model_key"):
        _parse_profile_document(document)


def test_model_key_mismatch_fails_before_selection():
    with pytest.raises(ValueError) as excinfo:
        select_expert_profile(
            "hy3-oq2e-64",
            model_key="hy3-expert-q2",
            installed_ram_bytes=128 * GiB,
            available_bytes=128 * GiB,
        )

    message = str(excinfo.value)
    assert "hy3-oq2e-64" in message
    assert "hy3-expert-q2" in message
    assert "hy3-expert-oq2e" in message


def test_explicit_profile_fails_instead_of_downgrading():
    available = 70 * GiB
    profile = load_expert_profiles()["hy3-oq2e-88"]
    required = profile_required_bytes(profile)

    with pytest.raises(ValueError) as excinfo:
        select_expert_profile(
            "hy3-oq2e-88",
            model_key="hy3-expert-oq2e",
            installed_ram_bytes=128 * GiB,
            available_bytes=available,
        )

    message = str(excinfo.value)
    assert str(required) in message
    assert str(available) in message
    assert "required" in message
    assert "available" in message
    # The reported requirement is what the plan reaches, not the declared
    # ceiling, which the profile's own cache cap leaves partly unallocated.
    assert required < profile.process_ceiling_bytes


def test_auto_fails_when_no_promoted_profile_fits():
    smallest = load_expert_profiles()["hy3-oq2e-64"]
    required = profile_required_bytes(smallest)
    available = required - GiB

    with pytest.raises(ValueError) as excinfo:
        select_expert_profile(
            "auto",
            model_key="hy3-expert-oq2e",
            installed_ram_bytes=128 * GiB,
            available_bytes=available,
        )

    message = str(excinfo.value)
    assert str(required) in message
    assert str(available) in message
    assert "hy3-oq2e-64" in message
    assert "hy3-oq2e-88" in message
    assert "hy3-oq2e-96" in message


def test_auto_admits_between_the_realized_footprint_and_the_ceiling():
    """A machine too small for the declared ceiling still runs the profile."""

    profile = load_expert_profiles()["hy3-oq2e-64"]
    required = profile_required_bytes(profile)
    assert required < profile.process_ceiling_bytes

    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=profile.process_ceiling_bytes - GiB,
        available_bytes=required,
    )
    assert selected.name == "hy3-oq2e-64"


def test_cache_cap_override_lowers_the_admission_requirement():
    """`--expert-cache-limit` reaches admission, not just construction."""

    profile = load_expert_profiles()["hy3-oq2e-64"]
    capped = profile_required_bytes(
        profile, overrides={"expert_cache_limit_bytes": "32GiB"}
    )
    assert capped < profile_required_bytes(profile)

    selected = select_expert_profile(
        "hy3-oq2e-64",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=64 * GiB,
        available_bytes=capped,
        overrides={"expert_cache_limit_bytes": "32GiB"},
    )
    assert selected.name == "hy3-oq2e-64"

    with pytest.raises(ValueError, match="required"):
        select_expert_profile(
            "hy3-oq2e-64",
            model_key="hy3-expert-oq2e",
            installed_ram_bytes=64 * GiB,
            available_bytes=64 * GiB,
        )


def test_available_memory_counts_reclaimable_vm_stat_pages(monkeypatch):
    output = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  10.
Pages active:                                20.
Pages inactive:                              30.
Pages speculative:                           40.
Pages throttled:                              0.
Pages wired down:                            60.
Pages purgeable:                             50.
"""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(expert_profiles.subprocess, "run", run)

    assert expert_profiles.available_memory_bytes() == 130 * 16384
    assert calls == [
        (
            ["/usr/bin/vm_stat"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 2.0,
            },
        )
    ]


def test_available_memory_rejects_missing_page_size(monkeypatch):
    output = """\
Mach Virtual Memory Statistics:
Pages free: 10.
Pages inactive: 30.
Pages speculative: 40.
Pages purgeable: 50.
"""
    monkeypatch.setattr(
        expert_profiles.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    with pytest.raises(RuntimeError, match="page size"):
        expert_profiles.available_memory_bytes()


def test_available_memory_rejects_zero_page_size(monkeypatch):
    output = """\
Mach Virtual Memory Statistics: (page size of 0 bytes)
Pages free: 10.
Pages inactive: 30.
Pages speculative: 40.
Pages purgeable: 50.
"""
    monkeypatch.setattr(
        expert_profiles.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    with pytest.raises(RuntimeError, match="page size.*positive"):
        expert_profiles.available_memory_bytes()


def test_available_memory_rejects_missing_required_counter(monkeypatch):
    output = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 10.
Pages inactive: 30.
Pages speculative: 40.
"""
    monkeypatch.setattr(
        expert_profiles.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    with pytest.raises(RuntimeError, match="Pages purgeable"):
        expert_profiles.available_memory_bytes()


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["/usr/bin/vm_stat"], timeout=2.0),
        subprocess.CalledProcessError(1, ["/usr/bin/vm_stat"]),
    ],
    ids=["timeout", "nonzero-exit"],
)
def test_available_memory_reports_subprocess_failure(monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(expert_profiles.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="vm_stat preflight failed"):
        expert_profiles.available_memory_bytes()
