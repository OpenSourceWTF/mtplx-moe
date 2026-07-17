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
