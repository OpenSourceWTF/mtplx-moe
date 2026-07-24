"""q1 expert artifact conversion lane (issue #51): smoke coverage."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from test_expert_shadow import _shadow_hy3_artifact

from mtplx.expert_manifest import load_expert_manifest, read_expert_record
from mtplx.expert_q1 import (
    Q1ManifestError,
    convert_expert_q1,
    decode_q1_record,
    load_q1_manifest,
    read_q1_record,
    verify_q1_against_source,
)
from mtplx.expert_shadow import (
    decode_shadow,
    dequantize_record_projections,
    encode_shadow,
    shadow_record_bytes,
)
from mtplx.expert_streaming_models import (
    GLM52_EXPERT_Q2,
    get_model_spec,
    plan_expert_memory,
)


@pytest.mark.parametrize("codec", ("b1", "t158"))
def test_convert_round_trips_records_bitwise(tmp_path: Path, codec: str) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    source = load_expert_manifest(manifest_path)
    output = tmp_path / "q1"
    written = convert_expert_q1(
        source, root, output, codec=codec, spec=spec, layers=(1,)
    )
    assert written.codec == codec
    assert written.source_model_key == source.model_key
    assert len(written.records) == spec.expert_count
    expected_bytes = shadow_record_bytes(codec, spec.expert_source_parameters)
    assert all(record.length == expected_bytes for record in written.records)

    loaded = load_q1_manifest(output / f"expert-manifest-q1-{codec}.json")
    assert loaded == written

    # A written record must decode to exactly the shadow decode of a fresh
    # encode from the dequantized source (bitwise payload verified below).
    record = loaded.record(1, 0)
    blob = read_q1_record(loaded, record)
    decoded = decode_q1_record(loaded, record, blob)
    source_blob = read_expert_record(source, root, 1, 0, verify_hash=False)
    dense = dequantize_record_projections(
        mx,
        source.record(1, 0),
        source_blob,
        bits=spec.quant_bits,
        group_size=spec.quant_group_size,
    )
    for projection, weights in dense.items():
        packed, scales = encode_shadow(codec, weights)
        reference = decode_shadow(codec, packed, scales, weights.shape[1])
        np.testing.assert_array_equal(decoded[projection], reference)
        # Quality tier: correlated with the source, not equal to it.
        cosine = float(
            (weights * reference).sum()
            / (np.linalg.norm(weights) * np.linalg.norm(reference))
        )
        assert cosine > 0.5

    verified = list(verify_q1_against_source(loaded, source, root, sample=2))
    assert len(verified) == 2


def test_convert_selection_bounds_and_corruption_detection(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    source = load_expert_manifest(manifest_path)
    output = tmp_path / "q1"
    written = convert_expert_q1(
        source, root, output, codec="t158", spec=spec, layers=(1,), limit=1
    )
    assert len(written.records) == 1

    with pytest.raises(Q1ManifestError, match="selected no records"):
        convert_expert_q1(
            source, root, tmp_path / "empty", codec="t158", spec=spec, layers=(0,)
        )

    # Hash gate: flip one payload byte and the read must fail loudly.
    bin_path = written.bin_path()
    payload = bytearray(bin_path.read_bytes())
    payload[7] ^= 0xFF
    bin_path.write_bytes(bytes(payload))
    with pytest.raises(Q1ManifestError, match="hash mismatch"):
        read_q1_record(written, written.records[0])


# ---------------------------------------------------------------------------
# q1 registry entries


@pytest.mark.parametrize(
    ("key", "codec"),
    (("glm52-expert-q1t", "t158"), ("glm52-expert-q1b1", "b1")),
)
def test_q1_spec_entries_price_shadow_records(key: str, codec: str) -> None:
    spec = get_model_spec(key)
    assert spec.expert_codec == codec
    assert spec.expert_record_bytes == shadow_record_bytes(
        codec, spec.expert_source_parameters
    )
    # Same checkpoint: only the routed record bytes differ from Q2.
    assert spec.resident_bytes == GLM52_EXPERT_Q2.resident_bytes
    assert spec.expert_record_bytes < GLM52_EXPERT_Q2.expert_record_bytes
    assert spec.island_pin_order == GLM52_EXPERT_Q2.island_pin_order
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=96 * 1024**3,
        context_tokens=4096,
        runtime_reserve_bytes=12 * 1024**3,
    )
    baseline = plan_expert_memory(
        GLM52_EXPERT_Q2,
        total_limit_bytes=96 * 1024**3,
        context_tokens=4096,
        runtime_reserve_bytes=12 * 1024**3,
    )
    # Smaller records => the same knob holds strictly more experts.
    assert plan.fits_fixed
    assert plan.slots_per_layer > baseline.slots_per_layer


def test_expert_codec_field_is_validated() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="expert_codec"):
        replace(GLM52_EXPERT_Q2, expert_codec="q9")


def test_resume_continues_interrupted_burn_bitwise(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    source = load_expert_manifest(manifest_path)

    fresh = convert_expert_q1(
        source, root, tmp_path / "full", codec="t158", spec=spec, layers=(1,)
    )
    reference = fresh.bin_path().read_bytes()

    # Interrupted run: one complete record plus a torn tail.
    partial_dir = tmp_path / "resume"
    convert_expert_q1(
        source, root, partial_dir, codec="t158", spec=spec, layers=(1,), limit=1
    )
    bin_path = partial_dir / "experts-q1-t158.bin"
    (partial_dir / "expert-manifest-q1-t158.json").unlink()
    with bin_path.open("ab") as handle:
        handle.write(b"\xde\xad\xbe\xef")

    resumed = convert_expert_q1(
        source,
        root,
        partial_dir,
        codec="t158",
        spec=spec,
        layers=(1,),
        resume=True,
    )
    assert bin_path.read_bytes() == reference
    assert [
        (record.layer, record.expert, record.offset, record.sha256)
        for record in resumed.records
    ] == [
        (record.layer, record.expert, record.offset, record.sha256)
        for record in fresh.records
    ]
    assert resumed.records[0].segments == fresh.records[0].segments
    list(verify_q1_against_source(resumed, source, root, sample=2))


def test_resume_refuses_a_changed_selection(tmp_path: Path) -> None:
    """A resume whose selection moved must not relabel the old bytes.

    The record identity axis is ``(layer, expert)``; this fixture has one
    routed layer, so the selection is moved along the expert axis. The
    guard compares the full identity, so a changed ``layers=`` set trips
    the same check.
    """

    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path, expert_count=4)
    source = load_expert_manifest(manifest_path)
    output = tmp_path / "q1"

    convert_expert_q1(
        source, root, output, codec="t158", spec=spec, layers=(1,), experts=(0, 1)
    )
    (output / "expert-manifest-q1-t158.json").unlink()

    with pytest.raises(Q1ManifestError, match="identity"):
        convert_expert_q1(
            source,
            root,
            output,
            codec="t158",
            spec=spec,
            layers=(1,),
            experts=(2, 3),
            resume=True,
        )


def test_resume_refuses_without_a_progress_journal(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    source = load_expert_manifest(manifest_path)
    output = tmp_path / "q1"

    convert_expert_q1(
        source, root, output, codec="t158", spec=spec, layers=(1,), limit=1
    )
    (output / "expert-manifest-q1-t158.json").unlink()
    (output / "experts-q1-t158.progress.jsonl").unlink()

    with pytest.raises(Q1ManifestError, match="progress journal"):
        convert_expert_q1(
            source, root, output, codec="t158", spec=spec, layers=(1,), resume=True
        )


def test_resume_rejects_bytes_that_do_not_match_the_journaled_hash(
    tmp_path: Path,
) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    source = load_expert_manifest(manifest_path)
    output = tmp_path / "q1"

    convert_expert_q1(
        source, root, output, codec="t158", spec=spec, layers=(1,), limit=1
    )
    (output / "expert-manifest-q1-t158.json").unlink()
    bin_path = output / "experts-q1-t158.bin"
    payload = bytearray(bin_path.read_bytes())
    payload[3] ^= 0xFF
    bin_path.write_bytes(bytes(payload))

    with pytest.raises(Q1ManifestError, match="hash mismatch"):
        convert_expert_q1(
            source, root, output, codec="t158", spec=spec, layers=(1,), resume=True
        )


def test_resume_rejects_a_different_source_artifact(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    source = load_expert_manifest(manifest_path)
    output = tmp_path / "q1"

    convert_expert_q1(
        source, root, output, codec="t158", spec=spec, layers=(1,), limit=1
    )
    (output / "expert-manifest-q1-t158.json").unlink()

    from dataclasses import replace

    other = replace(source, model_key="some-other-checkpoint")
    with pytest.raises(Q1ManifestError, match="different source"):
        convert_expert_q1(
            other, root, output, codec="t158", spec=spec, layers=(1,), resume=True
        )


def test_cli_emits_authoritative_manifest_for_assembled_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.convert_expert_q1 import main
    from mtplx import expert_streaming_models

    source_root, source_spec, source_manifest_path = _shadow_hy3_artifact(tmp_path)
    output_root = tmp_path / "q1-artifact"
    output_root.mkdir()
    shutil.copy(source_root / "config.json", output_root / "config.json")

    all_weights = mx.load(str(source_root / "model.safetensors"))
    resident = {
        name: value for name, value in all_weights.items() if "switch_mlp" not in name
    }
    resident_path = output_root / "model.safetensors"
    mx.save_safetensors(str(resident_path), resident)
    resident_bytes = sum(int(value.nbytes) for value in resident.values())
    (output_root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": resident_bytes},
                "weight_map": {name: resident_path.name for name in resident},
            }
        )
    )

    q1_spec = replace(
        source_spec,
        key="tiny-hy3-q1t158",
        display_name="Tiny Hy3 q1 t158",
        expert_codec="t158",
        quant_bits=2,
        total_tensor_bytes=(
            resident_bytes
            + source_spec.expert_count
            * shadow_record_bytes("t158", source_spec.expert_source_parameters)
        ),
    )
    real_get_model_spec = expert_streaming_models.get_model_spec

    def get_model_spec(key: str):
        if key == q1_spec.key:
            return q1_spec
        if key == source_spec.key:
            return source_spec
        return real_get_model_spec(key)

    monkeypatch.setattr(expert_streaming_models, "get_model_spec", get_model_spec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_expert_q1.py",
            "--source-root",
            str(source_root),
            "--manifest",
            str(source_manifest_path),
            "--output-dir",
            str(output_root),
            "--codec",
            "t158",
            "--layers",
            "1",
            "--streamed-model-key",
            q1_spec.key,
            "--verify-sample",
            "0",
        ],
    )

    assert main() == 0
    authoritative = load_expert_manifest(output_root / "expert-manifest.json")
    assert authoritative.model_key == q1_spec.key
    assert authoritative.quant_mode == "t158"
    assert len(authoritative.records) == source_spec.expert_count
