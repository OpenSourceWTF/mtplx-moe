from __future__ import annotations

import json
import struct

import pytest

from mtplx.qwen4_ngram import NGramManifest, NGramShard


def _write_safetensors(path, tensors):
    cursor = 0
    header = {}
    payload = bytearray()
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [cursor, cursor + size],
        }
        payload.extend(b"\0" * size)
        cursor += size
    encoded = json.dumps(header).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _manifest(rows: int = 1_000_000) -> NGramManifest:
    data_bytes = rows * 100
    return NGramManifest(
        source_repo="repo",
        source_revision="revision",
        storage="affine-q4-g32",
        row_width=160,
        row_bytes=100,
        padded_rows=rows,
        shards=(
            NGramShard(
                name="ngram.bin",
                tensor="ngram",
                start_row=0,
                row_count=rows,
                data_offset=0,
                data_bytes=data_bytes,
                file_size=data_bytes,
                sha256="0" * 64,
            ),
        ),
    )


def test_header_scan_excludes_only_ngram_payload(tmp_path) -> None:
    from mtplx.qwen4_preflight import scan_qwen4_weight_bytes

    _write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors",
        [
            ("language_model.model.layers.0.mlp.switch_mlp.weight", 100),
            (
                "language_model.model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight",
                20,
            ),
            ("language_model.mtp.layers.0.mlp.switch_mlp.weight", 30),
        ],
    )

    inventory = scan_qwen4_weight_bytes(tmp_path)

    assert inventory.total_bytes == 150
    assert inventory.resident_bytes == 130
    assert inventory.ngram_bytes == 20
    assert inventory.resident_moe_bytes == 130
    assert inventory.mtp_bytes == 30


def test_preflight_rejects_before_mlx_when_resident_weights_exceed_target() -> None:
    from mtplx.qwen4_preflight import plan_qwen4_resident_preflight

    with pytest.raises(ValueError, match="before MLX load"):
        plan_qwen4_resident_preflight(
            resident_weight_bytes=76 * 1024**3,
            manifest=_manifest(),
            context_tokens=17_408,
            payload_ceiling_bytes=10 * 1024**3,
            target_residency_bytes=75 * 1024**3,
        )


def test_preflight_accounts_configured_prefill_transient_bytes() -> None:
    from mtplx.qwen4_preflight import plan_qwen4_resident_preflight

    common = {
        "resident_weight_bytes": 1 * 1024**3,
        "manifest": _manifest(),
        "context_tokens": 17_408,
        "payload_ceiling_bytes": 2 * 1024**3,
        "target_residency_bytes": 8 * 1024**3,
    }
    small = plan_qwen4_resident_preflight(
        **common,
        prefill_chunk_tokens=2_048,
    )
    large = plan_qwen4_resident_preflight(
        **common,
        prefill_chunk_tokens=4_096,
    )

    assert large.cache_overhead_bytes - small.cache_overhead_bytes == 3_309_568


def test_preflight_rejects_non_oq4_artifact_contract() -> None:
    from mtplx.qwen4_preflight import validate_qwen4_oq4_contract

    with pytest.raises(ValueError, match="published oQ4"):
        validate_qwen4_oq4_contract(
            {"model_type": "qwen4_exp", "quantization": {"bits": 8}},
            _manifest(),
        )


def test_preflight_rejects_self_asserted_qwen4_manifest_provenance() -> None:
    from mtplx.qwen4_preflight import validate_qwen4_oq4_contract

    with pytest.raises(ValueError, match="published oQ4"):
        validate_qwen4_oq4_contract(
            {
                "model_type": "qwen4_exp",
                "quantization": {
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                },
            },
            _manifest(),
        )


def test_darwin_available_memory_parser_counts_reclaimable_pages_once() -> None:
    from mtplx.qwen4_preflight import parse_darwin_available_memory_bytes

    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10.
Pages active:                             40.
Pages inactive:                           20.
Pages speculative:                         3.
Pages purgeable:                            7.
"""

    # Purgeable pages are not added: vm_stat may report them as a subset of
    # another category, and the safety gate must never double-count memory.
    assert parse_darwin_available_memory_bytes(output) == 33 * 16_384


@pytest.mark.parametrize(
    "output",
    [
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n",
        "Pages free: 1.\nPages inactive: 2.\nPages speculative: 3.\n",
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: -1.\nPages inactive: 2.\nPages speculative: 3.\n",
    ],
)
def test_darwin_available_memory_parser_rejects_untrusted_output(output) -> None:
    from mtplx.qwen4_preflight import parse_darwin_available_memory_bytes

    with pytest.raises(ValueError, match="vm_stat"):
        parse_darwin_available_memory_bytes(output)
