"""Publish-time scrubbing of machine-identifying artifact metadata.

The fixtures below mirror the shapes found in real local artifacts:
``mtplx_runtime.json``'s ``forge_provenance.forge_inputs`` and a conversion
manifest's ``source``/``target`` paths.
"""

from __future__ import annotations

import copy

from mtplx.metadata_scrub import (
    REDACTED_PATH,
    runtime_metadata_leaks,
    scrub_path_value,
    scrub_runtime_metadata,
)


def _runtime_fixture() -> dict:
    """Mirrors ~/.cache/huggingface/hy3-q4-mlx-mtp/mtplx_runtime.json."""

    return {
        "arch_id": "hy_v3",
        "artifact_role": "forge-local",
        "base_trunk": "tencent/Hy3",
        "mtp_depth_max": 3,
        "mtp_sidecar": "mtp.safetensors",
        "mtp_sidecar_file": "mtp.safetensors",
        "mtplx_version": "2.0.2",
        "forge_provenance": {
            "forge_inputs": {
                "bf16_head_source_path": "/Users/davidtai/.cache/huggingface/hy3-mtp-layer80/layer80-bf16.safetensors",
                "checkpoint_path": "/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp/conversion-checkpoint.jsonl",
                "layout_oracle_path": "/Users/davidtai/.cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots/160619d3",
                "output_path": "/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp",
                "source_path": "/Users/davidtai/.cache/huggingface/hy3-mtp-layer80",
            },
            "forge_recipe": {"mtp_policy": "keep_bf16", "bits": 4},
            "forged_at": "2026-07-11T15:23:33Z",
            "forged_locally": True,
            "mtplx_version": "2.0.2",
            "intended_hf_repo": "davidtai/hy3-q4-mlx-mtp",
            "published_to_hf": None,
            "source_format": "bf16_native",
            "source_repo": "tencent/Hy3",
            "source_sha": "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
            "tool": "mtplx.hy3_native_quantizer v1",
        },
    }


def _conversion_manifest_fixture() -> dict:
    """Mirrors hy3-expert-only-mlx-q2/conversion-manifest.json."""

    return {
        "alignment": 16384,
        "producer": "mtplx.hy3_expert_q2",
        "source": {
            "path": "/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q4",
            "revision": "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
        },
        "journal": [
            {"step": "quantize", "note": "read /Users/davidtai/models/in.safetensors ok"},
        ],
        "target_descriptor": {"bits": 2, "group_size": 32},
    }


def test_forge_inputs_paths_are_redacted():
    scrubbed = scrub_runtime_metadata(_runtime_fixture())
    inputs = scrubbed["forge_provenance"]["forge_inputs"]

    assert inputs["source_path"] == f"{REDACTED_PATH}/hy3-mtp-layer80"
    assert inputs["output_path"] == f"{REDACTED_PATH}/hy3-q4-mlx-mtp"
    assert inputs["bf16_head_source_path"] == f"{REDACTED_PATH}/layer80-bf16.safetensors"
    for value in inputs.values():
        assert "/Users/" not in value


def test_intended_hf_repo_is_dropped():
    scrubbed = scrub_runtime_metadata(_runtime_fixture())

    assert "intended_hf_repo" not in scrubbed["forge_provenance"]


def test_useful_provenance_survives():
    scrubbed = scrub_runtime_metadata(_runtime_fixture())
    provenance = scrubbed["forge_provenance"]

    assert provenance["source_repo"] == "tencent/Hy3"
    assert provenance["source_sha"] == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert provenance["forge_recipe"] == {"mtp_policy": "keep_bf16", "bits": 4}
    assert provenance["forged_at"] == "2026-07-11T15:23:33Z"
    assert provenance["mtplx_version"] == "2.0.2"
    assert scrubbed["arch_id"] == "hy_v3"
    assert scrubbed["mtp_sidecar_file"] == "mtp.safetensors"
    assert scrubbed["mtp_depth_max"] == 3


def test_input_is_not_mutated():
    original = _runtime_fixture()
    snapshot = copy.deepcopy(original)

    scrub_runtime_metadata(original)

    assert original == snapshot


def test_conversion_manifest_paths_are_redacted():
    scrubbed = scrub_runtime_metadata(_conversion_manifest_fixture())

    assert scrubbed["source"]["path"] == f"{REDACTED_PATH}/hy3-expert-only-mlx-q4"
    assert scrubbed["source"]["revision"] == "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
    assert scrubbed["alignment"] == 16384


def test_paths_embedded_in_free_text_are_redacted():
    scrubbed = scrub_runtime_metadata(_conversion_manifest_fixture())

    note = scrubbed["journal"][0]["note"]
    assert "/Users/davidtai" not in note
    assert note.startswith("read ") and note.endswith(" ok")


def test_leak_detector_finds_and_then_clears():
    fixture = _runtime_fixture()

    assert runtime_metadata_leaks(fixture)
    assert runtime_metadata_leaks(scrub_runtime_metadata(fixture)) == []


def test_scrub_covers_linux_and_temp_dirs():
    assert scrub_path_value("/home/alice/models/x") == f"{REDACTED_PATH}/x"
    assert scrub_path_value("/var/folders/ab/T/run") == f"{REDACTED_PATH}/run"
    assert scrub_path_value("~/models/y") == f"{REDACTED_PATH}/y"


def test_non_path_values_are_left_alone():
    payload = {"repo": "owner/name", "license": "apache-2.0", "n": 3, "ok": True}

    assert scrub_runtime_metadata(payload) == payload


def test_lists_of_paths_are_redacted():
    scrubbed = scrub_runtime_metadata(
        {"inputs": ["/Users/davidtai/a.safetensors", "relative/b.safetensors"]}
    )

    assert scrubbed["inputs"] == [
        f"{REDACTED_PATH}/a.safetensors",
        "relative/b.safetensors",
    ]
