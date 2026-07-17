"""q1 expert artifact conversion lane (issue #51): smoke coverage."""

from __future__ import annotations

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

    verified = list(
        verify_q1_against_source(loaded, source, root, sample=2)
    )
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
