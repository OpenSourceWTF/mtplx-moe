"""q1 shadow-codec artifacts stream and execute (issue #51, gates 1/3/4).

The converter already writes q1 sidecars + manifests (test_expert_q1). These
tests close the loop: a q1 artifact is unified into the streamed
``ExpertManifest`` schema, opened by the runtime, and its routed records are
read into component-bank slots and executed through ``shadow_gather_mm`` — the
q2 lane's equal (read records into slots, execute from slots) but through the
shadow kernel instead of ``gather_qmm``. Output is compared against the same
tiny expert MLP executed with dequantized-t158 weights (close, not bitwise
vs affine — the codec differs by construction).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.activations import swiglu

from test_expert_shadow import _shadow_hy3_artifact

from mtplx import models as _models  # noqa: F401 - ensure package import path
from mtplx.expert_manifest import (
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_q1 import (
    build_streamed_expert_manifest,
    convert_expert_q1,
    decode_q1_record,
    read_q1_record,
)
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_shadow import shadow_record_bytes
from mtplx.models import expert_mlx
from mtplx.models.expert_mlx import (
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)
from mtplx.resident_loader import construct_resident_model

_NOMINAL_BITS = {"t158": 2, "b1": 1}


def _assemble_q1_artifact(tmp_path: Path, *, codec: str, expert_count: int, top_k: int):
    """Build a servable q1 artifact from a tiny affine source via the converter.

    Returns ``(q1_root, config_json, q1_spec, streamed_manifest_path,
    q1_manifest)`` — a resident-only checkpoint plus the shadow-codec record
    bin, unified into an authoritative streamed manifest.
    """

    src_parent = tmp_path / "src"
    src_parent.mkdir()
    src_root, src_spec, src_manifest_path = _shadow_hy3_artifact(
        src_parent, expert_count=expert_count, top_k=top_k
    )
    source = load_expert_manifest(src_manifest_path)

    # The converter produces {experts-q1-<codec>.bin, expert-manifest-q1-...}.
    q1_out = tmp_path / "q1out"
    q1_manifest = convert_expert_q1(
        source, src_root, q1_out, codec=codec, spec=src_spec, layers=(1,)
    )

    # Assemble the servable artifact: resident-only safetensors + the record
    # bin next to them, exactly the shape a real q1 burn ships.
    q1_root = tmp_path / "q1"
    q1_root.mkdir()
    shutil.copy(src_root / "config.json", q1_root / "config.json")
    shutil.copy(q1_out / q1_manifest.file, q1_root / q1_manifest.file)
    all_weights = mx.load(str(src_root / "model.safetensors"))
    resident = {
        name: value for name, value in all_weights.items() if "switch_mlp" not in name
    }
    mx.save_safetensors(str(q1_root / "model.safetensors"), resident)
    resident_bytes = sum(int(value.nbytes) for value in resident.values())
    index = {
        "metadata": {"total_size": resident_bytes},
        "weight_map": {name: "model.safetensors" for name in resident},
    }
    (q1_root / "model.safetensors.index.json").write_text(json.dumps(index))

    routed_bytes = expert_count * shadow_record_bytes(
        codec, src_spec.expert_source_parameters
    )
    q1_spec = replace(
        src_spec,
        key=f"tiny-hy3-q1{codec}",
        display_name=f"Tiny Hy3 q1 {codec}",
        expert_codec=codec,
        quant_bits=_NOMINAL_BITS[codec],
        total_tensor_bytes=resident_bytes + routed_bytes,
    )
    assert q1_spec.routed_expert_bytes == routed_bytes
    assert q1_spec.resident_bytes == resident_bytes

    streamed = build_streamed_expert_manifest(q1_root, q1_manifest, q1_spec)
    streamed_manifest_path = q1_root / "expert-manifest.json"
    save_expert_manifest(streamed, streamed_manifest_path)

    config_json = json.loads((q1_root / "config.json").read_text())
    return q1_root, config_json, q1_spec, streamed_manifest_path, q1_manifest


def _q1_config(q1_spec, *, miss_shadow=None, slot_layout="component-banks"):
    return ExpertStreamingConfig(
        model_key=q1_spec.key,
        memory_limit_bytes=(
            q1_spec.resident_bytes
            + q1_spec.transient_scratch_bytes
            + q1_spec.persistent_cache_bytes(q1_spec.expert_count)
            + 8 * 1024 * 1024
        ),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="layer",
        slot_layout=slot_layout,
        miss_shadow=miss_shadow,
    )


def _open_q1_runtime(q1_root, q1_spec, manifest_path):
    config = _q1_config(q1_spec)
    return ExpertStreamingRuntime.open(
        q1_root,
        manifest_path,
        config,
        spec=q1_spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            config.memory_plan(q1_spec),
            q1_spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )


def test_streamed_manifest_unifies_q1_into_authoritative_schema(tmp_path: Path) -> None:
    q1_root, _config, q1_spec, manifest_path, q1_manifest = _assemble_q1_artifact(
        tmp_path, codec="t158", expert_count=2, top_k=1
    )
    manifest = load_expert_manifest(manifest_path)
    # Gate 1: the streamed manifest carries the shadow codec as its quant mode
    # and the six ordered shadow components per record, all addressing the bin.
    assert manifest.quant_mode == "t158"
    assert manifest.quant_bits == 2
    assert manifest.sidecar is not None
    assert manifest.sidecar.file == q1_manifest.file
    assert len(manifest.records) == q1_spec.expert_count
    record = manifest.record(1, 0)
    assert tuple(segment.component for segment in record.segments) == (
        "gate_proj.packed",
        "gate_proj.scales",
        "up_proj.packed",
        "up_proj.scales",
        "down_proj.packed",
        "down_proj.scales",
    )
    assert record.logical_bytes == q1_spec.expert_record_bytes
    assert all(segment.shard == q1_manifest.file for segment in record.segments)


def test_q1_forward_executes_through_shadow_kernel(tmp_path: Path) -> None:
    q1_root, config_json, q1_spec, manifest_path, _q1 = _assemble_q1_artifact(
        tmp_path, codec="t158", expert_count=2, top_k=1
    )
    runtime = _open_q1_runtime(q1_root, q1_spec, manifest_path)

    # Spy on the shadow component-bank dispatch without replacing its math:
    # this is the load-bearing evidence that routes execute through
    # shadow_gather_mm (gate 4), not gather_qmm.
    seen_codecs: list[str] = []
    original = expert_mlx._run_component_bank_shadow

    def spy(selected, bindings, *, codec):
        seen_codecs.append(codec)
        return original(selected, bindings, codec=codec)

    expert_mlx._run_component_bank_shadow = spy
    try:
        resident = construct_resident_model(q1_root, runtime, config=config_json)
        # The affine gather kernel must never run for a q1 layer.
        gather_calls = 0
        original_gather = expert_mlx.mx.gather_qmm

        def count_gather(*args, **kwargs):
            nonlocal gather_calls
            gather_calls += 1
            return original_gather(*args, **kwargs)

        expert_mlx.mx.gather_qmm = count_gather
        try:
            logits = resident.model(mx.array([[1]], dtype=mx.int32))
            mx.eval(logits)
        finally:
            expert_mlx.mx.gather_qmm = original_gather

        assert logits.shape == (1, 1, config_json["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        assert seen_codecs and all(codec == "t158" for codec in seen_codecs)
        assert gather_calls == 0

        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["expert_codec"] == "t158"
        # The record was streamed into a slot and executed from it (the q2
        # lane's equal), not served from any RAM shadow bank.
        assert snapshot["cache"]["expert_requests"] >= 1
        assert "shadow_experts" not in snapshot
    finally:
        expert_mlx._run_component_bank_shadow = original
        runtime.close()


@pytest.mark.parametrize("codec", ("t158", "b1"))
def test_q1_switch_output_matches_dequantized_reference(
    tmp_path: Path, codec: str
) -> None:
    q1_root, _config, q1_spec, manifest_path, q1_manifest = _assemble_q1_artifact(
        tmp_path, codec=codec, expert_count=2, top_k=1
    )
    runtime = _open_q1_runtime(q1_root, q1_spec, manifest_path)
    try:
        switch = HotExpertSwitchGLU(runtime, 1)
        assert switch.codec == codec
        rng = np.random.default_rng(7)
        x = mx.array(rng.standard_normal((1, 1, 64)).astype(np.float32))
        expert = 1
        output = switch(x, mx.array([[[expert]]], dtype=mx.uint32))
        mx.eval(output)

        # Reference: the SAME expert MLP with dequantized-<codec> weights,
        # decoded straight from the streamed record's bytes.
        record = q1_manifest.record(1, expert)
        dense = decode_q1_record(
            q1_manifest, record, read_q1_record(q1_manifest, record)
        )
        xr = mx.array(np.asarray(x).reshape(1, 64))
        gate = mx.matmul(xr, mx.array(dense["gate_proj"]).T)
        up = mx.matmul(xr, mx.array(dense["up_proj"]).T)
        reference = mx.matmul(swiglu(gate, up), mx.array(dense["down_proj"]).T)
        mx.eval(reference)

        got = np.asarray(output).reshape(-1)
        ref = np.asarray(reference).reshape(-1)
        np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)
    finally:
        runtime.close()


def test_miss_shadow_rejected_on_q1_primary_spec(tmp_path: Path) -> None:
    q1_root, _config, q1_spec, manifest_path, _q1 = _assemble_q1_artifact(
        tmp_path, codec="t158", expert_count=2, top_k=1
    )
    # miss_shadow shadows an EXACT affine artifact with a low-precision bank;
    # a q1 spec is already a shadow codec, so shadowing it is nonsense.
    config = _q1_config(q1_spec, miss_shadow="t158")
    with pytest.raises(ExpertStreamingConfigurationError, match="shadow"):
        ExpertStreamingRuntime.open(
            q1_root,
            manifest_path,
            config,
            spec=q1_spec,
            device_synchronize=mx.synchronize,
            apply_memory_cap=False,
        )


def test_q1_requires_component_banks_slot_layout(tmp_path: Path) -> None:
    q1_root, _config, q1_spec, manifest_path, _q1 = _assemble_q1_artifact(
        tmp_path, codec="t158", expert_count=2, top_k=1
    )
    config = _q1_config(q1_spec, slot_layout="direct-slots")
    with pytest.raises(ExpertStreamingConfigurationError, match="component-banks"):
        ExpertStreamingRuntime.open(
            q1_root,
            manifest_path,
            config,
            spec=q1_spec,
            device_synchronize=mx.synchronize,
            apply_memory_cap=False,
        )


def test_kv_and_resident_quant_stay_orthogonal_to_q1(tmp_path: Path) -> None:
    # kv_quant / resident_quant are untouched by the q1 lane: a q1 config
    # accepts them and the codec still routes through the shadow kernel.
    q1_root, config_json, q1_spec, manifest_path, _q1 = _assemble_q1_artifact(
        tmp_path, codec="t158", expert_count=2, top_k=1
    )
    config = replace(
        _q1_config(q1_spec),
        resident_quant="q4",
    )
    runtime = ExpertStreamingRuntime.open(
        q1_root,
        manifest_path,
        config,
        spec=q1_spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            config.memory_plan(q1_spec),
            q1_spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(q1_root, runtime, config=config_json)
        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert mx.all(mx.isfinite(logits)).item()
        assert runtime.snapshot(mx_module=mx)["expert_codec"] == "t158"
    finally:
        runtime.close()
