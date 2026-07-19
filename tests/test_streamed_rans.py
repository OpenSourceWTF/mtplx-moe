"""Streamed rANS miss-read integration (issue #113).

Covers the converter (per-record container round-trip), the streamed codec
manifest (validation + pricing + base binding), and the reader decode-on-miss
path proven against a live forward: output bitwise-identical to the
uncompressed run, read-bytes dropped, and the memory plan untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.expert_manifest import (
    build_expert_sidecar,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_rans_metal import decode_container
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_streamed_codec import (
    STREAMED_CODEC_FORMAT,
    StreamedCodecError,
    StreamedCodecManifest,
    decode_record_reference,
    encode_record_payload,
    load_streamed_codec_manifest,
    validate_against_base,
    write_streamed_rans_sidecar,
)
from mtplx.models.expert_mlx import make_mlx_component_bank_allocator
from mtplx.resident_loader import construct_resident_model

from test_streamed_models import _integrated_hy3_artifact


# ------------------------------------------------------------------ codec core


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00" * 6912,  # all zeros (weights) -- maximally compressible
        bytes(bytearray(range(256)) * 27),  # 32-aligned mixed bytes
        np.random.default_rng(0).integers(0, 4, size=1000).astype(np.uint8).tobytes(),
        np.random.default_rng(1).integers(0, 256, size=4096).astype(np.uint8).tobytes(),
        b"\x81\x3f" * 500 + b"\x00" * 33,  # 1033 bytes -> exercises 32-lane padding
    ],
)
def test_encode_decode_record_roundtrip_reference(payload: bytes) -> None:
    blob = encode_record_payload(payload)
    assert decode_record_reference(blob, len(payload)) == payload


@pytest.mark.parametrize(
    "size",
    [6912, 1000, 1033, 4096],
)
def test_metal_decode_matches_source(size: int) -> None:
    payload = np.random.default_rng(size).integers(0, 6, size=size).astype(np.uint8).tobytes()
    blob = encode_record_payload(payload)
    decoded = decode_container(blob)
    host = bytes(np.array(decoded, dtype=np.uint8).reshape(-1)[:size])
    assert host == payload


def test_encode_rejects_unknown_codec() -> None:
    with pytest.raises(StreamedCodecError):
        encode_record_payload(b"\x00" * 32, codec="zstd-19")


# ------------------------------------------------------------- manifest schema


def _codec_record(**kwargs):
    from mtplx.expert_streamed_codec import StreamedCodecRecord

    base = dict(layer=1, expert=0, offset=0, length=64, raw_length=100, sha256="ab")
    base.update(kwargs)
    return StreamedCodecRecord(**base)


def _manifest(records, *, alignment=16 * 1024, size=1 << 20):
    return StreamedCodecManifest(
        format=STREAMED_CODEC_FORMAT,
        model_key="tiny-hy3-q4",
        codec="rans32x-v1",
        file="experts-rans32x.bin",
        alignment=alignment,
        expert_count=2,
        base_manifest_sha256="deadbeef",
        size=size,
        sha256="feedface",
        records=tuple(records),
        path=None,
    )


def test_manifest_validate_rejects_misaligned_record() -> None:
    manifest = _manifest([_codec_record(offset=7)])
    with pytest.raises(StreamedCodecError, match="not aligned"):
        manifest.validate()


def test_manifest_validate_rejects_overlap() -> None:
    # The first record spills past the 16K stride into the second's extent.
    manifest = _manifest(
        [
            _codec_record(expert=0, offset=0, length=20 * 1024),
            _codec_record(expert=1, offset=16 * 1024, length=1024),
        ]
    )
    with pytest.raises(StreamedCodecError, match="overlap"):
        manifest.validate()


def test_manifest_validate_rejects_record_past_end() -> None:
    manifest = _manifest([_codec_record(offset=0, length=100)], size=50)
    with pytest.raises(StreamedCodecError, match="exceeds the sidecar size"):
        manifest.validate()


def test_manifest_from_json_roundtrips_and_rejects_bad_codec() -> None:
    manifest = _manifest([_codec_record()])
    obj = manifest.to_json()
    restored = StreamedCodecManifest.from_json(obj)
    assert restored.records == manifest.records
    obj["codec"] = "zstd-19"
    with pytest.raises(StreamedCodecError):
        StreamedCodecManifest.from_json(obj)


# ------------------------------------------------------------------ converter


def _base_with_sidecar(tmp_path: Path):
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    base = load_expert_manifest(manifest_path)
    updated = build_expert_sidecar(base, root, root / "experts.bin")
    save_expert_manifest(updated, manifest_path)
    return root, config, spec, manifest_path, load_expert_manifest(manifest_path)


def test_converter_roundtrips_every_record_bitwise(tmp_path: Path) -> None:
    root, _config, _spec, manifest_path, base = _base_with_sidecar(tmp_path)
    codec_manifest_path = root / "expert-streamed-codec-rans32x.json"
    manifest = write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x.bin",
        output_manifest=codec_manifest_path,
    )
    manifest.validate()
    assert manifest.stored_bytes < manifest.raw_bytes
    assert manifest.compression_ratio() > 1.0
    # Every container decodes back to the exact base record payload.
    from mtplx.expert_manifest import read_expert_record

    blob_bytes = manifest.bin_path().read_bytes()
    for record in manifest.records:
        source = read_expert_record(
            base, root, record.layer, record.expert, verify_hash=True
        )
        blob = blob_bytes[record.offset : record.offset + record.length]
        assert decode_record_reference(blob, record.raw_length) == source
    # The on-disk manifest reloads and binds to the base manifest.
    reloaded = load_streamed_codec_manifest(codec_manifest_path)
    validate_against_base(reloaded, base)
    assert reloaded.base_manifest_sha256 == base.manifest_sha256


def test_converter_smoke_slice_limits_records(tmp_path: Path) -> None:
    root, _config, _spec, _manifest_path, base = _base_with_sidecar(tmp_path)
    manifest = write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x-slice.bin",
        output_manifest=root / "codec-slice.json",
        layers=[1],
        experts=[0],
        limit=1,
    )
    assert len(manifest.records) == 1
    assert manifest.records[0].expert == 0


def test_validate_against_base_rejects_wrong_raw_length(tmp_path: Path) -> None:
    root, _config, _spec, _manifest_path, base = _base_with_sidecar(tmp_path)
    manifest = write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x.bin",
        output_manifest=root / "codec.json",
    )
    from dataclasses import replace

    broken = replace(
        manifest,
        records=(replace(manifest.records[0], raw_length=manifest.records[0].raw_length + 1),)
        + manifest.records[1:],
    )
    with pytest.raises(StreamedCodecError, match="raw_length"):
        validate_against_base(broken, base)


# --------------------------------------------------------- reader decode-on-miss


def _open_runtime(root, manifest_path, spec, config_dict, *, streamed_codec, codec_manifest, codec_verify=True):
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    cfg = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        slot_layout="component-banks",
        streamed_codec=streamed_codec,
        streamed_codec_manifest=(str(codec_manifest) if streamed_codec != "none" else None),
        streamed_codec_verify=codec_verify,
    )
    plan = cfg.memory_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        cfg,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan, spec, load_expert_manifest(manifest_path)
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    return runtime, plan


def test_reader_decode_on_miss_is_bitwise_identical_and_saves_read_bytes(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path, base = _base_with_sidecar(tmp_path)
    codec_manifest_path = root / "expert-streamed-codec-rans32x.json"
    codec_manifest = write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x.bin",
        output_manifest=codec_manifest_path,
    )

    def forward(streamed_codec):
        runtime, plan = _open_runtime(
            root,
            manifest_path,
            spec,
            config,
            streamed_codec=streamed_codec,
            codec_manifest=codec_manifest_path,
        )
        try:
            resident = construct_resident_model(root, runtime, config=config)
            logits = resident.model(mx.array([[1, 2]], dtype=mx.int32))
            mx.eval(logits)
            io = runtime.reader.metrics.as_dict()
            snap_io = runtime.snapshot(mx_module=mx)["slots"]["io"]
            return mx.array(logits), io, snap_io, plan
        finally:
            runtime.close()

    logits_none, io_none, _snap_none, plan_none = forward("none")
    logits_rans, io_rans, snap_rans, plan_rans = forward("rans32x-v1")

    # 1) Bitwise-identical output -- lossless, exact-quality ledger.
    assert bool(mx.array_equal(logits_none, logits_rans).item())

    # 2) Fewer bytes pulled off SSD; the win is a runtime read counter.
    assert io_rans["read_bytes"] < io_none["read_bytes"]
    assert io_rans["decoded_records"] >= 1
    assert io_rans["bytes_read_saved"] == io_none["read_bytes"] - io_rans["read_bytes"]
    assert io_rans["decoded_raw_bytes"] == io_none["read_bytes"]
    # Telemetry surfaces through the runtime snapshot io counters.
    assert snap_rans["bytes_read_saved"] == io_rans["bytes_read_saved"]
    assert snap_rans["integrity_errors"] == 0

    # 3) Slots do NOT shrink: the memory plan is byte-identical either way.
    assert plan_none.slots_per_layer == plan_rans.slots_per_layer
    assert plan_none.persistent_cache_bytes == plan_rans.persistent_cache_bytes
    assert plan_none.fixed_bytes == plan_rans.fixed_bytes
    assert plan_none.allocated_bytes == plan_rans.allocated_bytes


def test_codec_read_bytes_reflect_compressed_container_size(tmp_path: Path) -> None:
    root, config, spec, manifest_path, base = _base_with_sidecar(tmp_path)
    codec_manifest_path = root / "expert-streamed-codec-rans32x.json"
    codec_manifest = write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x.bin",
        output_manifest=codec_manifest_path,
    )
    runtime, _plan = _open_runtime(
        root,
        manifest_path,
        spec,
        config,
        streamed_codec="rans32x-v1",
        codec_manifest=codec_manifest_path,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        # A single token routes exactly one expert -> one decoded miss.
        mx.eval(resident.model(mx.array([[1]], dtype=mx.int32)))
        io = runtime.reader.metrics.as_dict()
        compressed_lengths = {record.length for record in codec_manifest.records}
        assert io["decoded_records"] == 1
        # read_bytes is the compressed container actually pulled off SSD.
        assert io["read_bytes"] in compressed_lengths
        # Accounting closes: bytes pulled + bytes saved == the raw record size.
        assert io["read_bytes"] + io["bytes_read_saved"] == io["decoded_raw_bytes"]
        assert io["decoded_raw_bytes"] == spec.expert_record_bytes
    finally:
        runtime.close()


# ----------------------------------------------------------- config validation


def test_config_requires_manifest_for_codec() -> None:
    with pytest.raises(ValueError, match="requires a streamed_codec_manifest"):
        ExpertStreamingConfig(
            model_key="tiny-hy3-q4",
            memory_limit_bytes=1 << 30,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            streamed_codec="rans32x-v1",
        )


def test_config_rejects_manifest_without_codec() -> None:
    with pytest.raises(ValueError, match="requires streamed_codec"):
        ExpertStreamingConfig(
            model_key="tiny-hy3-q4",
            memory_limit_bytes=1 << 30,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            streamed_codec="none",
            streamed_codec_manifest="x.json",
        )


def test_config_rejects_codec_with_metal_mmap() -> None:
    with pytest.raises(ValueError, match="metal-mmap"):
        ExpertStreamingConfig(
            model_key="tiny-hy3-q4",
            memory_limit_bytes=1 << 30,
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            slot_layout="metal-mmap",
            verify_sidecar_hash_at_open=True,
            streamed_codec="rans32x-v1",
            streamed_codec_manifest="x.json",
        )


def test_open_rejects_mismatched_codec_manifest(tmp_path: Path) -> None:
    root, _config, spec, manifest_path, base = _base_with_sidecar(tmp_path)
    codec_manifest_path = root / "codec.json"
    write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x.bin",
        output_manifest=codec_manifest_path,
    )
    # Corrupt the base_manifest_sha256 binding so open() rejects it.
    obj = json.loads(codec_manifest_path.read_text())
    obj["base_manifest_sha256"] = "0" * 64
    codec_manifest_path.write_text(json.dumps(obj))
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    cfg = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        slot_layout="component-banks",
        streamed_codec="rans32x-v1",
        streamed_codec_manifest=str(codec_manifest_path),
    )
    plan = cfg.memory_plan(spec)
    with pytest.raises(ExpertStreamingConfigurationError, match="streamed codec"):
        ExpertStreamingRuntime.open(
            root,
            manifest_path,
            cfg,
            spec=spec,
            buffer_allocator=make_mlx_component_bank_allocator(
                plan, spec, load_expert_manifest(manifest_path)
            ),
            device_synchronize=mx.synchronize,
            apply_memory_cap=False,
        )


def test_streamed_codec_verify_off_skips_hash_but_stays_bitwise(tmp_path: Path) -> None:
    """verify=False must skip the post-decode sha256 (David: optional, default
    flips off after the 16k validation) while output stays bitwise-identical —
    the container's structural guards, not the hash, carry correctness."""

    root, config, spec, manifest_path, base = _base_with_sidecar(tmp_path)
    codec_manifest_path = root / "expert-streamed-codec-rans32x.json"
    write_streamed_rans_sidecar(
        base,
        root,
        output_bin=root / "experts-rans32x.bin",
        output_manifest=codec_manifest_path,
    )

    def forward(codec, verify=True):
        runtime, _plan = _open_runtime(
            root, manifest_path, spec, config,
            streamed_codec=codec, codec_manifest=codec_manifest_path,
            codec_verify=verify,
        )
        try:
            assert runtime.reader.codec_verify is verify
            resident = construct_resident_model(root, runtime, config=config)
            logits = resident.model(mx.array([[1, 2]], dtype=mx.int32))
            mx.eval(logits)
            return mx.array(logits), runtime.reader.metrics.as_dict()
        finally:
            runtime.close()

    logits_ref, _ = forward("none")
    logits_off, io_off = forward("rans32x-v1", verify=False)
    assert bool(mx.array_equal(logits_ref, logits_off).item())
    assert io_off["decoded_records"] >= 1
    assert io_off["integrity_errors"] == 0

    # verify=True catches decode output that no structural guard can see:
    # deterministically corrupt by wrapping the decoder to flip one byte of
    # its (structurally valid) output -- byte-position gambling on the real
    # container is flaky because random fixture layouts move the payload.
    def forward_corrupted(codec_verify):
        runtime, _plan = _open_runtime(
            root, manifest_path, spec, config,
            streamed_codec="rans32x-v1", codec_manifest=codec_manifest_path,
            codec_verify=codec_verify,
        )
        try:
            real = runtime.reader._decode_container_fn()

            def corrupting(payload):
                import numpy as _np

                out = _np.array(real(payload), dtype=_np.uint8).reshape(-1).copy()
                out[out.size // 2] ^= 0xFF
                return out

            runtime.reader._decode_container = corrupting
            resident = construct_resident_model(root, runtime, config=config)
            mx.eval(resident.model(mx.array([[1, 2]], dtype=mx.int32)))
            return runtime.reader.metrics.as_dict()
        finally:
            runtime.close()

    with pytest.raises(Exception, match="hash mismatch|integrity"):
        forward_corrupted(True)
    io_corrupt_off = forward_corrupted(False)
    assert io_corrupt_off["integrity_errors"] == 0  # hash never consulted

def test_streamed_codec_verify_validation() -> None:
    with pytest.raises(TypeError, match="streamed_codec_verify"):
        ExpertStreamingConfig(
            model_key="hy3-expert-q2",
            memory_limit_bytes=96 * 1024**3,
            max_live_kv_tokens=4096,
            streamed_codec_verify="yes",
        )
