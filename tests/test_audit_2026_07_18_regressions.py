"""Regression tests for the 2026-07-18 audit findings.

One test per fixed finding. The rANS zero-frequency case is guarded by a
subprocess timeout: a regression there HANGS rather than fails, so an
in-process assertion would wedge CI instead of reporting.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import mtplx.expert_rans as R

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Finding 4: rANS infinite loop on a zero-frequency byte -----------------


def _table_missing_symbol() -> R.RansTable:
    """A table that assigns zero frequency to symbol 7."""

    freq = np.zeros(256, dtype=np.int64)
    freq[0] = R.M - 1
    freq[1] = 1
    return R.table_from_freq(freq)


def test_zero_frequency_symbol_raises_instead_of_hanging_scalar() -> None:
    table = _table_missing_symbol()
    data = np.full(32, 7, dtype=np.uint8)  # symbol 7 has freq 0
    with pytest.raises(R.RansError, match="zero frequency"):
        R.encode_segment(data, table)


def test_zero_frequency_symbol_raises_instead_of_hanging_vectorized() -> None:
    table = _table_missing_symbol()
    segments = np.full((2, 32), 7, dtype=np.uint8)
    with pytest.raises(R.RansError, match="zero frequency"):
        R.encode_bank(segments, table)


@pytest.mark.parametrize("count", [1, 2, 31, 32, 64])
def test_zero_frequency_detected_for_odd_and_even_counts(count: int) -> None:
    """The guard must not depend on the parity of the symbol count.

    A bitwise AND against raw counts (`counts & (freq == 0)`) passes for any
    even count -- 32 & 1 == 0 -- and falls straight through to the hang.
    """

    table = _table_missing_symbol()
    data = np.full(count * R.LANES, 7, dtype=np.uint8)
    with pytest.raises(R.RansError, match="zero frequency"):
        R.encode_segment(data, table)
    with pytest.raises(R.RansError, match="zero frequency"):
        R.encode_bank(data.reshape(1, -1), table)


def test_zero_frequency_encode_terminates_under_timeout() -> None:
    """Hard guard: before the fix both encoders spun forever.

    Run in a subprocess so a regression fails on the timeout instead of
    hanging the suite.
    """

    script = textwrap.dedent(
        """
        import numpy as np
        import mtplx.expert_rans as R

        freq = np.zeros(256, dtype=np.int64)
        freq[0] = R.M - 1
        freq[1] = 1
        table = R.table_from_freq(freq)
        data = np.full(32, 7, dtype=np.uint8)
        for call in (
            lambda: R.encode_segment(data, table),
            lambda: R.encode_bank(data.reshape(1, 32), table),
        ):
            try:
                call()
            except R.RansError:
                pass
            else:
                raise SystemExit("expected RansError")
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_zero_frequency_is_still_legal_for_unused_symbols() -> None:
    """A real table is mostly zeros -- the guard must not reject that."""

    data = np.random.RandomState(0).randint(0, 4, size=4096).astype(np.uint8)
    table = R.build_table(R.histogram(data))
    assert int((table.freq == 0).sum()) > 200
    assert len(R.encode_segment(data, table)) == R.LANES


# --- Finding 5: rANS silent 4 GiB directory wrap ---------------------------


def test_directory_offset_at_uint32_boundary_is_accepted() -> None:
    R._require_directory_fits((1 << 32) - 1)


def test_directory_offset_past_uint32_raises_instead_of_wrapping() -> None:
    with pytest.raises(R.RansError, match="exceeds the uint32 range"):
        R._require_directory_fits(1 << 32)


def test_vectorized_encoder_still_matches_scalar_reference() -> None:
    """The two encoders are documented as bit-identical; keep it that way."""

    rng = np.random.RandomState(7)
    segments = rng.randint(0, 256, size=(3, 128)).astype(np.uint8)
    table = R.build_table(R.histogram(segments.ravel()))
    scalar = R.encode_bank_scalar(segments, table)
    vector = R.encode_bank(segments, table)
    np.testing.assert_array_equal(scalar.payload, vector.payload)
    np.testing.assert_array_equal(scalar.directory, vector.directory)


# --- Finding 3: MTP payload completeness guards ----------------------------


def test_deepseek_payload_guard_rejects_stray_keys() -> None:
    from mtplx.deepseek_mtp_patch import _has_complete_deepseek_mtp_payload

    complete = {
        "layers.0.enorm.weight": 1,
        "layers.0.hnorm.weight": 1,
        "layers.0.eh_proj.weight": 1,
        "layers.0.mtp_block.self_attn.q_proj.weight": 1,
    }
    assert _has_complete_deepseek_mtp_payload(complete, num_mtp_layers=1)
    # Non-empty, but no real MTP tensors -- the old `if not mapped:` passed.
    assert not _has_complete_deepseek_mtp_payload(
        {"layers.0.something_else": 1}, num_mtp_layers=1
    )
    # Projections present but the draft block missing.
    missing_block = {k: v for k, v in complete.items() if "mtp_block" not in k}
    assert not _has_complete_deepseek_mtp_payload(missing_block, num_mtp_layers=1)
    # Declared two layers, only one supplied.
    assert not _has_complete_deepseek_mtp_payload(complete, num_mtp_layers=2)


def test_mimo_payload_guard_rejects_stray_keys() -> None:
    from mtplx.mimo_mtp_patch import _has_complete_mimo_mtp_payload

    complete = {
        "layers.0.token_layernorm.weight": 1,
        "layers.0.hidden_layernorm.weight": 1,
        "layers.0.input_proj.weight": 1,
        "layers.0.final_layernorm.weight": 1,
        "layers.0.mtp_block.self_attn.q_proj.weight": 1,
    }
    assert _has_complete_mimo_mtp_payload(complete, num_mtp_layers=1)
    assert not _has_complete_mimo_mtp_payload(
        {"lm_head.weight": 1}, num_mtp_layers=1
    )
    missing_block = {k: v for k, v in complete.items() if "mtp_block" not in k}
    assert not _has_complete_mimo_mtp_payload(missing_block, num_mtp_layers=1)


def test_nemotron_h_payload_guard_rejects_stray_keys() -> None:
    from mtplx.nemotron_h_mtp_patch import _has_complete_nemotron_h_mtp_payload

    complete = {
        "layers.0.norm.weight": 1,
        "layers.0.mixer.in_proj.weight": 1,
    }
    assert _has_complete_nemotron_h_mtp_payload(complete, physical_layers=1)
    assert not _has_complete_nemotron_h_mtp_payload(
        {"layers.0.block_type": 1}, physical_layers=1
    )
    assert not _has_complete_nemotron_h_mtp_payload(
        {"layers.0.norm.weight": 1}, physical_layers=1
    )
    assert not _has_complete_nemotron_h_mtp_payload(complete, physical_layers=2)


# --- Finding 6: StreamedBatchRunner lm_head fp32-cast trap ------------------


def _run_decode_step(*, enable_lm_head_fp32: bool, with_logits_head: bool):
    """Drive StreamedBatchRunner._decode_step over a minimal fake model.

    Returns (head_input_dtype, logits_dtype, logits_head_used).
    """

    import mlx.core as mx

    from mtplx.streamed_batch import StreamedBatchRunner

    seen: dict[str, object] = {"logits_head_used": False}

    class _Head:
        def __call__(self, hidden):
            seen["head_input_dtype"] = hidden.dtype
            # A real head emits bf16; fp32 must come from casting the OUTPUT.
            return mx.zeros((hidden.shape[0], hidden.shape[1], 8), dtype=mx.bfloat16)

    head = _Head()

    class _Inner:
        layers: list = []

        def embed_tokens(self, batch):
            return mx.zeros((batch.shape[0], 1, 4), dtype=mx.bfloat16)

        def norm(self, hidden):
            return hidden

    class _Args:
        pass

    args = _Args()
    args.enable_lm_head_fp32 = enable_lm_head_fp32

    class _Model:
        model_type = "hy3"

        def __init__(self) -> None:
            self.model = _Inner()
            self.args = args
            self.lm_head = head

        if with_logits_head:

            def _logits_head(self):
                seen["logits_head_used"] = True
                return head

    class _Stream:
        tokens = [1]
        cache: list = []

        def __init__(self) -> None:
            self.decode_steps = 0

        def sample(self, row):
            seen["logits_dtype"] = row.dtype

    class _Runtime:
        pass

    runtime = _Runtime()
    runtime.model = _Model()

    class _FakeRunner:
        # Real walk helper; with no layers it passes hidden straight through,
        # which keeps this test focused on the head.
        _split_attention_layer_walk = (
            StreamedBatchRunner.__dict__["_split_attention_layer_walk"]
        )

    runner = _FakeRunner()
    runner._live = [_Stream()]
    runner._rt = runtime

    StreamedBatchRunner._decode_step(runner)
    return (
        seen.get("head_input_dtype"),
        seen.get("logits_dtype"),
        seen["logits_head_used"],
    )


def test_lm_head_fp32_casts_logits_not_the_head_input() -> None:
    """The audit's untested branch: enable_lm_head_fp32=True.

    Casting the INPUT forces MLX to materialize an fp32 copy of the whole
    [vocab, hidden] head per step. The head input must stay bf16 while the
    logits come back fp32.
    """

    import mlx.core as mx

    head_input, logits, used = _run_decode_step(
        enable_lm_head_fp32=True, with_logits_head=True
    )
    assert head_input == mx.bfloat16, f"head input was cast to {head_input}"
    assert logits == mx.float32, f"logits were not cast to fp32 (got {logits})"
    assert used, "decode did not route through _logits_head()"


def test_lm_head_routes_through_logits_head_when_flag_is_off() -> None:
    import mlx.core as mx

    head_input, logits, used = _run_decode_step(
        enable_lm_head_fp32=False, with_logits_head=True
    )
    assert head_input == mx.bfloat16
    assert logits == mx.bfloat16  # no cast when the flag is off
    assert used


def test_lm_head_falls_back_when_model_has_no_logits_head() -> None:
    """Non-hy3 models expose only .lm_head; the fallback must still work."""

    import mlx.core as mx

    head_input, logits, used = _run_decode_step(
        enable_lm_head_fp32=True, with_logits_head=False
    )
    assert head_input == mx.bfloat16
    assert logits == mx.float32
    assert not used


# --- Finding 8b: shallow verify reports what it actually checked -----------


def test_shallow_verify_reports_payload_hash_not_verified() -> None:
    """`deep=False` cannot hash shard payloads; the result must say so."""

    import inspect

    import mtplx.hy3_expert_q2 as q2

    source = inspect.getsource(q2._verify_held_expert_payloads)
    # The self-comparison is now gated on `deep` rather than always evaluated.
    assert "deep and parsed_shard.sha256 != expected_shard.sha256" in source
    assert '"shard_payload_verified": deep' in source


# --- Finding 7: Q4->Q2 fidelity floor --------------------------------------


def test_q2_fidelity_floor_is_defined_and_below_the_q4_floor() -> None:
    from mtplx.glm52_mtp_artifact import Q4_MIN_ROUNDTRIP_COSINE
    from mtplx.hy3_expert_q2 import Q2_MIN_ROUNDTRIP_COSINE

    assert 0.0 < Q2_MIN_ROUNDTRIP_COSINE < Q4_MIN_ROUNDTRIP_COSINE


def test_diagnostics_json_rejects_a_noise_projection() -> None:
    from mtplx.hy3_expert_q2 import (
        ProjectionDiagnostics,
        Q2_MIN_ROUNDTRIP_COSINE,
        _diagnostics_json,
        _PROJECTIONS,
    )

    def build(cosine_for_first: float) -> tuple[ProjectionDiagnostics, ...]:
        return tuple(
            ProjectionDiagnostics(
                component=component,
                cosine_q4_q2=cosine_for_first if index == 0 else 0.95,
                normalized_error_q4_q2=0.1,
                finite=True,
            )
            for index, component in enumerate(_PROJECTIONS)
        )

    # A healthy set still passes.
    assert len(_diagnostics_json(build(0.95))) == len(_PROJECTIONS)
    # The audit's cosine-0.02 case no longer reports "passed".
    with pytest.raises(ValueError, match="below the pinned floor"):
        _diagnostics_json(build(0.02))
    # Exactly at the floor is accepted; just below is not.
    assert len(_diagnostics_json(build(Q2_MIN_ROUNDTRIP_COSINE))) == len(_PROJECTIONS)
    with pytest.raises(ValueError, match="below the pinned floor"):
        _diagnostics_json(build(Q2_MIN_ROUNDTRIP_COSINE - 0.01))
