"""Mixed-official banks stream and execute (issue #51, M2).

These tests close the consumer loop for the mixed-precision "official recipe"
bank: a tiny two-layer bank (one t158-tier layer, one affine2-tier layer) is
built through the REAL M1 converter (``convert_mixed_official``), unified into
the streamed ``ExpertManifest`` schema, opened by the runtime, and served
through the component-banks lane. The load-bearing evidence is per-projection-
group dispatch (D4): a t158 layer runs gate/up through ``shadow_gather_mm`` and
down through ``gather_qmm(bits=3)`` in one pass; an affine2 layer runs gate/up
through ``gather_qmm(bits=2)`` and down through ``gather_qmm(bits=3)``. Output
is compared against a ``decode_projection`` dense reference (close, not bitwise
for the shadow tier — the codec differs by construction).

CPU-only: no GPU/Metal beyond the tiny gather kernels the q1 serve test also
runs, no real bank, no network.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu

from test_convert_expert_mixed_official import _build_fixture

from mtplx import models as _models  # noqa: F401 - ensure package import path
from mtplx.expert_manifest import (
    ExpertManifestError,
    build_mixed_official_manifest,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_mixed_official import decode_projection
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_streaming_models import (
    MIXED_OFFICIAL_CODEC,
    ExpertStreamingModelSpec,
    plan_expert_memory,
)
from mtplx.models import expert_mlx
from mtplx.models.expert_mlx import (
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.resident_loader import construct_resident_model

HIDDEN = 64
EXPERT_HIDDEN = 64
ROUTED_LAYERS = (1, 2)  # layer 1 -> t158 (IQ1_M), layer 2 -> affine2 (IQ2_XXS)
_MIXED_DIMS = {
    "gate_proj": (EXPERT_HIDDEN, HIDDEN),
    "up_proj": (EXPERT_HIDDEN, HIDDEN),
    "down_proj": (HIDDEN, EXPERT_HIDDEN),
}
_LAYER_GGUF = {1: "IQ1_M", 2: "IQ2_XXS"}


# ---------------------------------------------------------------------------
# fixture assembly
# ---------------------------------------------------------------------------
def _hy3_args(*, expert_count: int, top_k: int) -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=HIDDEN,
        num_hidden_layers=3,
        intermediate_size=128,
        moe_intermediate_size=EXPERT_HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=expert_count,
        num_experts_per_tok=top_k,
        num_shared_experts=1,
        first_k_dense_replace=1,  # layer 0 dense; layers 1, 2 routed
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _assemble_mixed_artifact(tmp_path: Path, *, expert_count: int = 2, top_k: int = 1):
    """Build a servable mixed-official artifact through the REAL converter.

    Returns ``(root, config_json, spec, manifest_path, mixed_manifest)`` — a
    resident-only checkpoint plus the mixed record bin, unified into an
    authoritative streamed manifest.
    """

    # 1. tiny hy3 checkpoint: config.json + resident tensors (routed switch_mlp
    #    is streamed, so it is stripped from the resident safetensors).
    args = _hy3_args(expert_count=expert_count, top_k=top_k)
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    resident = {name: value for name, value in weights.items() if "switch_mlp" not in name}
    mx.eval(list(resident.values()))

    root = tmp_path / "mix"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), resident)
    resident_bytes = sum(int(value.nbytes) for value in resident.values())
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": resident_bytes},
                "weight_map": {name: "model.safetensors" for name in resident},
            }
        )
    )

    # 2. bf16 expert source + coverage + recipe through the M1 fixture builder,
    #    then the REAL converter (unaligned tiny dims are explicitly waived).
    fixture = _build_fixture(
        tmp_path / "src",
        layers=_LAYER_GGUF,
        experts=expert_count,
        dims=_MIXED_DIMS,
    )
    from scripts.convert_expert_mixed_official import convert_mixed_official

    # convert_mixed_official pins the default device to CPU for the encode; the
    # serving path (shadow_gather_mm is a GPU metal kernel, exactly as the q1
    # serve test runs it) needs it restored afterwards.
    prior_device = mx.default_device()
    mixed_manifest = convert_mixed_official(
        source_manifest=fixture["coverage"],
        recipe_map=fixture["recipe"],
        bf16_root=fixture["bf16_root"],
        output_root=tmp_path / "bank",
        allow_unaligned_segments=True,
        verify_sample=1000,
    )
    mx.set_default_device(prior_device)
    shutil.copy(tmp_path / "bank" / mixed_manifest["sidecar"]["file"], root / mixed_manifest["sidecar"]["file"])

    # 3. pinned mixed spec: resident model + mixed routed bytes (no scalar bits).
    routed_bytes = mixed_manifest["artifact"]["routed_expert_bytes"]
    spec = ExpertStreamingModelSpec(
        key="tiny-hy3-mixofficial",
        display_name="Tiny Hy3 mixed-official",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="test/tiny-hy3-mixofficial",
        quant_revision="quant",
        total_tensor_bytes=resident_bytes + routed_bytes,
        total_layers=3,
        routed_layer_start=1,
        routed_layer_count=2,
        expert_count=expert_count,
        top_k=top_k,
        hidden_size=HIDDEN,
        expert_hidden_size=EXPERT_HIDDEN,
        quant_bits=2,  # inert placeholder; mixed uses per-layer tiers
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=expert_count * HIDDEN * 4 + expert_count * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=3,
        mtp_included=False,
        expert_codec=MIXED_OFFICIAL_CODEC,
    )

    # 4. unify into the streamed schema (fresh runtime digest) and persist.
    manifest = build_mixed_official_manifest(root, mixed_manifest, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)

    config_json = json.loads((root / "config.json").read_text())
    return root, config_json, spec, manifest_path, mixed_manifest


def _mixed_transient_bytes(spec, layer_bytes) -> int:
    """One transient service bank per distinct tier geometry."""

    return spec.top_k * sum(set(layer_bytes.values()))


def _mixed_memory_limit(spec, manifest) -> int:
    """A ceiling that admits full per-layer residency (slots_per_layer == E)."""

    layer_bytes = manifest.record_bytes_by_layer()
    routed = spec.expert_count * sum(layer_bytes.values())
    resident = spec.total_tensor_bytes - routed
    return resident + routed + _mixed_transient_bytes(spec, layer_bytes) + 16 * 1024 * 1024


def _mixed_partial_limit(spec, manifest) -> int:
    """A ceiling that admits exactly one persistent slot per layer.

    With ``slots_per_layer == 1 < expert_count`` a prefill route that touches
    more than one expert per layer must spill to the transient tier — the path
    that a differing-tier miss used to crash.
    """

    layer_bytes = manifest.record_bytes_by_layer()
    streamed_sum = sum(layer_bytes.values())  # one slot per layer, all layers
    resident = spec.total_tensor_bytes - spec.expert_count * streamed_sum
    fixed = resident + _mixed_transient_bytes(spec, layer_bytes)
    # available in [streamed_sum, 2*streamed_sum) -> slots_per_layer == 1.
    return fixed + streamed_sum + streamed_sum // 2


def _mixed_config(spec, manifest, **overrides) -> ExpertStreamingConfig:
    base = dict(
        model_key=spec.key,
        memory_limit_bytes=_mixed_memory_limit(spec, manifest),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="layer",
        slot_layout="component-banks",
    )
    base.update(overrides)
    return ExpertStreamingConfig(**base)


def _open_mixed(root, spec, manifest_path, config=None, *, memory_limit_bytes=None):
    manifest = load_expert_manifest(manifest_path)
    if config is None:
        overrides = (
            {"memory_limit_bytes": memory_limit_bytes}
            if memory_limit_bytes is not None
            else {}
        )
        config = _mixed_config(spec, manifest, **overrides)
    plan = config.memory_plan(spec, layer_record_bytes=manifest.record_bytes_by_layer())
    return ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(plan, spec, manifest),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )


def _assert_served_matches(got: np.ndarray, ref: np.ndarray) -> None:
    """Parity that proves "the served weights ARE the decoded record".

    The served path (Metal shadow/gather kernels) and the numpy
    decode_projection reference round the codec slightly differently, and the
    fixture's full-scale N(0,1) weights push a rare near-cancellation output
    element to ~15% relative error. So assert the bulk is near-exact (median)
    and the whole vector matches in relative L2 norm (robust to lone outliers),
    instead of an element-wise max bound.
    """

    got = np.asarray(got).reshape(-1)
    ref = np.asarray(ref).reshape(-1)
    relative = np.abs(got - ref) / (np.abs(ref) + 1e-6)
    assert np.median(relative) < 1e-3
    assert np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-6) < 5e-3


def _decode_reference(manifest, root, layer: int, expert: int) -> dict[str, np.ndarray]:
    """Dense (out, in) fp32 weights straight from the stored record bytes."""

    record = manifest.record(layer, expert)
    bin_path = Path(root) / manifest.sidecar.file
    with bin_path.open("rb") as handle:
        handle.seek(record.sidecar_offset)
        blob = handle.read(record.sidecar_length)
    dtypes = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "BF16": np.uint16}
    leaves: dict[str, dict[str, np.ndarray]] = {}
    cursor = 0
    for segment in record.segments:
        raw = blob[cursor : cursor + segment.length]
        cursor += segment.length
        projection, leaf = segment.component.split(".", 1)
        leaves.setdefault(projection, {})[leaf] = np.frombuffer(
            raw, dtype=dtypes[segment.dtype]
        ).reshape(segment.shape)
    gate_up_tier, down_tier = manifest.mixed_tier_for_layer(layer)
    tiers = {"gate_proj": gate_up_tier, "up_proj": gate_up_tier, "down_proj": down_tier}
    return {
        projection: decode_projection(tiers[projection], leaves[projection])
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


# ---------------------------------------------------------------------------
# manifest unification
# ---------------------------------------------------------------------------
def test_unified_manifest_carries_the_tier_schedule(tmp_path: Path) -> None:
    _root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    assert manifest.quant_mode == MIXED_OFFICIAL_CODEC
    assert manifest.quant_bits is None
    assert manifest.mixed_tier_for_layer(1) == ("t158", "affine3")
    assert manifest.mixed_tier_for_layer(2) == ("affine2", "affine3")
    # The two layers have different record sizes and component orders.
    layer_bytes = manifest.record_bytes_by_layer()
    assert layer_bytes[1] != layer_bytes[2]
    t158_record = manifest.record(1, 0)
    affine2_record = manifest.record(2, 0)
    assert tuple(seg.component for seg in t158_record.segments) == (
        "gate_proj.packed",
        "gate_proj.scales",
        "up_proj.packed",
        "up_proj.scales",
        "down_proj.weight",
        "down_proj.scales",
        "down_proj.biases",
    )
    assert len(affine2_record.segments) == 9
    # quant_tier is preserved verbatim on every segment (D5).
    assert t158_record.segments[0].quant_tier == "t158"
    assert t158_record.segments[-1].quant_tier == "affine3"
    assert affine2_record.segments[0].quant_tier == "affine2"


# ---------------------------------------------------------------------------
# per-projection-group dispatch (D4)
# ---------------------------------------------------------------------------
class _KernelSpy:
    def __init__(self) -> None:
        self.shadow_projections = 0
        self.gather_bits: list[int] = []

    def install(self, monkeypatch) -> None:
        import mtplx.kernels.shadow_gather as shadow_module

        original_shadow = shadow_module.shadow_gather_mm

        def shadow(*args, **kwargs):
            self.shadow_projections += 1
            return original_shadow(*args, **kwargs)

        monkeypatch.setattr(shadow_module, "shadow_gather_mm", shadow)

        original_gather = expert_mlx.mx.gather_qmm

        def gather(*args, **kwargs):
            self.gather_bits.append(int(kwargs.get("bits")))
            return original_gather(*args, **kwargs)

        monkeypatch.setattr(expert_mlx.mx, "gather_qmm", gather)


def test_t158_layer_dispatches_shadow_gate_up_and_affine3_down(
    tmp_path: Path, monkeypatch
) -> None:
    root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    runtime = _open_mixed(root, spec, manifest_path)
    try:
        switch = HotExpertSwitchGLU(runtime, 1)
        assert switch.codec == MIXED_OFFICIAL_CODEC
        assert switch._gate_up_tier == "t158"
        spy = _KernelSpy()
        spy.install(monkeypatch)
        rng = np.random.default_rng(3)
        x = mx.array(rng.standard_normal((1, 1, HIDDEN)).astype(np.float32))
        output = switch(x, mx.array([[[1]]], dtype=mx.uint32))
        mx.eval(output)
        # gate + up shadow; down affine3 (bits=3), no bits=2.
        assert spy.shadow_projections == 2
        assert spy.gather_bits == [3]
    finally:
        runtime.close()


def test_affine2_layer_dispatches_affine2_gate_up_and_affine3_down(
    tmp_path: Path, monkeypatch
) -> None:
    root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    runtime = _open_mixed(root, spec, manifest_path)
    try:
        switch = HotExpertSwitchGLU(runtime, 2)
        assert switch._gate_up_tier == "affine2"
        spy = _KernelSpy()
        spy.install(monkeypatch)
        rng = np.random.default_rng(5)
        x = mx.array(rng.standard_normal((1, 1, HIDDEN)).astype(np.float32))
        output = switch(x, mx.array([[[0]]], dtype=mx.uint32))
        mx.eval(output)
        # gate + up affine2 (bits=2); down affine3 (bits=3); no shadow.
        assert spy.shadow_projections == 0
        assert sorted(spy.gather_bits) == [2, 2, 3]
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# numeric parity vs a decode_projection dense reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layer", ROUTED_LAYERS)
def test_switch_output_matches_dense_reference(tmp_path: Path, layer: int) -> None:
    root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    runtime = _open_mixed(root, spec, manifest_path)
    manifest = load_expert_manifest(manifest_path)
    try:
        switch = HotExpertSwitchGLU(runtime, layer)
        rng = np.random.default_rng(layer * 17)
        x = mx.array(rng.standard_normal((1, 1, HIDDEN)).astype(np.float32))
        expert = 1
        output = switch(x, mx.array([[[expert]]], dtype=mx.uint32))
        mx.eval(output)

        dense = _decode_reference(manifest, root, layer, expert)
        xr = mx.array(np.asarray(x).reshape(1, HIDDEN))
        gate = mx.matmul(xr, mx.array(dense["gate_proj"]).T)
        up = mx.matmul(xr, mx.array(dense["up_proj"]).T)
        reference = mx.matmul(swiglu(gate, up), mx.array(dense["down_proj"]).T)
        mx.eval(reference)

        _assert_served_matches(np.asarray(output), np.asarray(reference))
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# full model forward through open + construct_resident_model + bind
# ---------------------------------------------------------------------------
def test_full_forward_exercises_both_tiers(tmp_path: Path, monkeypatch) -> None:
    root, config_json, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    runtime = _open_mixed(root, spec, manifest_path)
    try:
        spy = _KernelSpy()
        spy.install(monkeypatch)
        resident = construct_resident_model(root, runtime, config=config_json)
        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config_json["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        # The t158 layer fired the shadow kernel; both layers' down (and the
        # affine2 layer's gate/up) fired gather_qmm — both codecs in one pass.
        assert spy.shadow_projections >= 2
        assert 3 in spy.gather_bits and 2 in spy.gather_bits
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["expert_codec"] == MIXED_OFFICIAL_CODEC
        assert snapshot["cache"]["expert_requests"] >= 1
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# partial residency: per-tier transient service (issue #51, M2b)
# ---------------------------------------------------------------------------
def test_partial_residency_serves_both_tiers_through_transient(
    tmp_path: Path, monkeypatch
) -> None:
    """One run, one persistent slot per layer: a second expert on BOTH a
    t158-tier and an affine2-tier layer spills to the per-tier transient bank,
    is filled + dispatched + numerically correct, and never crashes."""

    root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    limit = _mixed_partial_limit(spec, manifest)
    runtime = _open_mixed(root, spec, manifest_path, memory_limit_bytes=limit)
    try:
        # The plan admits exactly one persistent slot per layer.
        assert runtime.plan.slots_per_layer == 1
        assert runtime.plan.slots_per_layer < spec.expert_count

        for layer, expected_tier in ((1, "t158"), (2, "affine2")):
            switch = HotExpertSwitchGLU(runtime, layer)
            assert switch._gate_up_tier == expected_tier
            spy = _KernelSpy()
            spy.install(monkeypatch)
            rng = np.random.default_rng(100 + layer)
            # Two tokens, top_k=1, routing experts 0 then 1: expert 0 is seeded
            # persistent, expert 1 must be served from the transient tier.
            x = mx.array(rng.standard_normal((1, 2, HIDDEN)).astype(np.float32))
            indices = mx.array([[[0], [1]]], dtype=mx.uint32)
            output = switch(x, indices)  # PREFILL (seq len 2)
            mx.eval(output)

            # Dispatch fired for this layer's tier (proves execution ran, not a
            # crash on the differing-tier transient bank).
            if expected_tier == "t158":
                assert spy.shadow_projections >= 2  # gate + up
                assert 3 in spy.gather_bits  # affine3 down
            else:
                assert spy.shadow_projections == 0
                assert 2 in spy.gather_bits and 3 in spy.gather_bits

            # Numeric parity for BOTH assignments, including the transient one.
            got = np.asarray(output).reshape(2, HIDDEN)
            for token, expert in ((0, 0), (1, 1)):
                dense = _decode_reference(manifest, root, layer, expert)
                xr = mx.array(np.asarray(x)[0, token].reshape(1, HIDDEN))
                gate = mx.matmul(xr, mx.array(dense["gate_proj"]).T)
                up = mx.matmul(xr, mx.array(dense["up_proj"]).T)
                ref = np.asarray(
                    mx.matmul(swiglu(gate, up), mx.array(dense["down_proj"]).T)
                ).reshape(-1)
                _assert_served_matches(got[token], ref)

        # The transient tier actually serviced misses (fill happened).
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["cache"]["transient_loads"] >= 2
    finally:
        runtime.close()


def test_partial_residency_full_forward_is_finite(tmp_path: Path) -> None:
    root, config_json, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    limit = _mixed_partial_limit(spec, manifest)
    runtime = _open_mixed(root, spec, manifest_path, memory_limit_bytes=limit)
    try:
        assert runtime.plan.slots_per_layer == 1
        resident = construct_resident_model(root, runtime, config=config_json)
        # Multi-token prefill through the whole model at partial residency.
        logits = resident.model(mx.array([[1, 5, 9, 3]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 4, config_json["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# memory-plan arithmetic with two record sizes (D2)
# ---------------------------------------------------------------------------
def test_memory_plan_prices_two_record_sizes(tmp_path: Path) -> None:
    _root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    layer_bytes = manifest.record_bytes_by_layer()
    t158_bytes = layer_bytes[1]
    affine2_bytes = layer_bytes[2]
    per_layer_sum = t158_bytes + affine2_bytes
    routed_total = spec.expert_count * per_layer_sum
    expected_resident = spec.total_tensor_bytes - routed_total

    # Full residency: one persistent slot per expert in each layer.
    full = plan_expert_memory(
        spec,
        total_limit_bytes=expected_resident + routed_total + 64 * 1024 * 1024,
        context_tokens=0,
        layer_record_bytes=layer_bytes,
    )
    assert full.resident_bytes == expected_resident
    assert full.slots_per_layer == spec.expert_count
    assert full.persistent_cache_bytes == spec.expert_count * per_layer_sum
    # One transient service bank per distinct tier geometry (t158 + affine2).
    assert full.transient_bytes == spec.top_k * (t158_bytes + affine2_bytes)

    # A budget for exactly one persistent slot per layer (transient is now one
    # bank per distinct tier geometry).
    one_slot = plan_expert_memory(
        spec,
        total_limit_bytes=expected_resident
        + spec.top_k * (t158_bytes + affine2_bytes)
        + per_layer_sum,
        context_tokens=0,
        layer_record_bytes=layer_bytes,
    )
    assert one_slot.slots_per_layer == 1
    assert one_slot.persistent_cache_bytes == per_layer_sum
    assert one_slot.fits_fixed


def test_expert_record_bytes_raises_for_mixed_spec(tmp_path: Path) -> None:
    _root, _cfg, spec, _manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    with pytest.raises(ValueError, match="mixed-official"):
        _ = spec.expert_record_bytes  # noqa: B018 - property access is the assertion


# ---------------------------------------------------------------------------
# fail-closed serving lane (D6) and tier coverage (D1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"slot_layout": "direct-slots"}, "component-banks"),
        (
            {
                "slot_layout": "metal-mmap",
                "verify_sidecar_hash_at_open": True,
                "prefer_sidecar": True,
            },
            "component-banks",
        ),
        ({"miss_shadow": "t158"}, "shadow"),
        ({"island_layers": (1,)}, "island"),
        ({"prefetch_slots": 1}, "prefetch"),
        ({"cache_scope": "global"}, "cache_scope"),
    ],
)
def test_forbidden_lanes_are_rejected(tmp_path: Path, overrides, match) -> None:
    root, _cfg, spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    config = _mixed_config(spec, manifest, **overrides)
    with pytest.raises(ExpertStreamingConfigurationError, match=match):
        ExpertStreamingRuntime.open(
            root,
            manifest_path,
            config,
            spec=spec,
            device_synchronize=mx.synchronize,
            apply_memory_cap=False,
        )


def test_missing_tier_entry_fails_closed(tmp_path: Path) -> None:
    _root, _cfg, _spec, manifest_path, _mm = _assemble_mixed_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    # A routed layer with no tier entry is a bug, not a default (D1).
    with pytest.raises(ExpertManifestError, match="no tier entry"):
        manifest.mixed_tier_for_layer(99)
    # A manifest whose layer_tier_map drops a routed layer fails structure.
    tampered = replace(manifest, layer_tier_map=(manifest.layer_tier_map[0],))
    with pytest.raises(ExpertManifestError):
        tampered.validate_structure()
