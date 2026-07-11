"""Streamed Hy3 MTP wiring: opt-in flag, exact fallback, decode routing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import test_hy3_mtp as mtp_fixtures
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_streaming import RoutingPhase
from mtplx.expert_streaming_models import ExpertStreamingModelSpec
from mtplx.generation import generate_ar, generate_mtp1
from mtplx.hy3_mtp_patch import Hy3MTPLoadError, inject_hy3_streamed_mtp_support
from mtplx.expert_manifest import build_expert_manifest, save_expert_manifest
from mtplx.models.expert_mlx import (
    current_expert_routing_phase,
    make_mlx_slot_buffer_allocator,
)
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.mtp_patch import MTPContract, validate_mtp_support
from mtplx.resident_loader import construct_resident_model
from mtplx.runtime import MTPLXRuntime, load
from mtplx.sampling import SamplerConfig

TEST_REVISION = mtp_fixtures.TEST_REVISION


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)

    def encode(self, text):
        return [1, 2, 3]


def _streamed_args() -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
        num_nextn_predict_layers=1,
    )


def _integrated_streamed_hy3(tmp_path: Path):
    """Tiny streamed Hy3 artifact, mirroring tests/test_streamed_models.py."""

    mx.random.seed(3)
    args = _streamed_args()
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    expert_shapes = {
        "gate_proj.weight": (2, 64, 8),
        "gate_proj.scales": (2, 64, 1),
        "gate_proj.biases": (2, 64, 1),
        "up_proj.weight": (2, 64, 8),
        "up_proj.scales": (2, 64, 1),
        "up_proj.biases": (2, 64, 1),
        "down_proj.weight": (2, 64, 8),
        "down_proj.scales": (2, 64, 1),
        "down_proj.biases": (2, 64, 1),
    }
    for component, shape in expert_shapes.items():
        dtype = mx.uint32 if component.endswith("weight") else mx.bfloat16
        value = mx.zeros(shape, dtype=dtype)
        if component.endswith("scales"):
            value = mx.ones(shape, dtype=dtype)
        weights[f"model.layers.1.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = tmp_path / "hy3"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-hy3-q4",
        display_name="Tiny Hy3 Q4",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="test/tiny-hy3-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=2 * 64 * 4 + 2 * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, config, spec, manifest_path


def _open_streamed_runtime(root, spec, manifest_path):
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    return ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )


def _mtplx_runtime(model, streaming, *, mtp_enabled: bool) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
        expert_streaming=streaming,
    )


def _phase_for(rt: MTPLXRuntime, tokens: int) -> RoutingPhase:
    with rt._expert_routing_context(mx.zeros((1, tokens), dtype=mx.int32)):
        # An explicit context always wins over the token-count heuristic.
        return current_expert_routing_phase(token_count=1)


def test_flag_off_leaves_streamed_model_and_routing_unchanged(tmp_path: Path) -> None:
    root, config, spec, manifest_path = _integrated_streamed_hy3(tmp_path)
    runtime = _open_streamed_runtime(root, spec, manifest_path)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        model = resident.model

        # Without the flag the model is the plain streamed Hy3 class: no MTP
        # module, no patched call surface, no speculative hooks.
        assert type(model) is Hy3Model
        assert getattr(model, "mtp", None) is None
        assert not hasattr(model, "mtp_forward")
        assert not hasattr(model, "mtp_verify_width")

        rt = _mtplx_runtime(model, runtime, mtp_enabled=False)
        assert _phase_for(rt, 1) is RoutingPhase.DECODE
        assert _phase_for(rt, 2) is RoutingPhase.PREFILL
        assert _phase_for(rt, 3) is RoutingPhase.PREFILL
    finally:
        runtime.close()


def test_injection_preserves_ar_forward_exactly(tmp_path: Path) -> None:
    root, config, spec, manifest_path = _integrated_streamed_hy3(tmp_path)
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    mtp_fixtures._write_tiny_artifacts(mtp_dir)
    runtime = _open_streamed_runtime(root, spec, manifest_path)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        model = resident.model
        prompt = mx.array([[1, 2, 4]], dtype=mx.int32)
        before = model(prompt)
        mx.eval(before)

        injected = inject_hy3_streamed_mtp_support(
            model, mtp_dir, config, MTPContract(), expected_revision=TEST_REVISION
        )
        assert injected is True
        assert validate_mtp_support(model)
        assert isinstance(model, Hy3Model)
        assert type(model) is not Hy3Model
        assert model.mtp_verify_width == 2

        after = model(prompt)
        logits_rh, hidden = model(prompt, return_hidden=True)
        mx.eval(after, logits_rh, hidden)
        assert mx.array_equal(before, after).item()
        assert mx.array_equal(before, logits_rh).item()
        assert hidden.shape == (1, 3, spec.hidden_size)
    finally:
        runtime.close()


def test_mtp_enabled_runtime_classifies_verify_batches_as_decode(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_streamed_hy3(tmp_path)
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    mtp_fixtures._write_tiny_artifacts(mtp_dir)
    runtime = _open_streamed_runtime(root, spec, manifest_path)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        inject_hy3_streamed_mtp_support(
            resident.model,
            mtp_dir,
            config,
            MTPContract(),
            expected_revision=TEST_REVISION,
        )
        rt = _mtplx_runtime(resident.model, runtime, mtp_enabled=True)
        assert _phase_for(rt, 1) is RoutingPhase.DECODE
        # A primary+draft verify batch trains the decode hot set.
        assert _phase_for(rt, 2) is RoutingPhase.DECODE
        assert _phase_for(rt, 3) is RoutingPhase.PREFILL
    finally:
        runtime.close()


def _injected_runtime_pair(tmp_path: Path):
    root, config, spec, manifest_path = _integrated_streamed_hy3(tmp_path)
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    mtp_fixtures._write_tiny_artifacts(mtp_dir)
    runtime = _open_streamed_runtime(root, spec, manifest_path)
    resident = construct_resident_model(root, runtime, config=config)
    inject_hy3_streamed_mtp_support(
        resident.model, mtp_dir, config, MTPContract(), expected_revision=TEST_REVISION
    )
    rt_ar = _mtplx_runtime(resident.model, runtime, mtp_enabled=False)
    rt_mtp = _mtplx_runtime(resident.model, runtime, mtp_enabled=True)
    return runtime, rt_ar, rt_mtp


def test_generate_mtp1_drafts_and_matches_greedy_ar_exactly(tmp_path: Path) -> None:
    runtime, rt_ar, rt_mtp = _injected_runtime_pair(tmp_path)
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=1)
    prompt = [1, 2, 3]
    try:
        ar = generate_ar(rt_ar, prompt, max_tokens=8, sampler=sampler, seed=0)
        out = generate_mtp1(rt_mtp, prompt, max_tokens=8, sampler=sampler, seed=0)

        # Exact rejection sampling: speculation may only change speed, never
        # the greedy token stream.
        assert out.tokens == ar.tokens
        assert out.stats.drafted_tokens > 0
        assert (
            out.stats.accepted_drafts + out.stats.rejected_drafts
            == out.stats.drafted_tokens
        )
        assert rt_mtp.diagnostic_counters.get("draft_mtp_calls", 0) > 0
        assert rt_mtp.diagnostic_counters.get("make_mtp_cache_calls", 0) > 0
    finally:
        runtime.close()


def test_rejected_drafts_fall_back_to_target_tokens_exactly(tmp_path: Path) -> None:
    runtime, rt_ar, rt_mtp = _injected_runtime_pair(tmp_path)
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=1)
    prompt = [1, 2, 3]
    try:
        ar = generate_ar(rt_ar, prompt, max_tokens=8, sampler=sampler, seed=0)
        model = rt_mtp.model
        vocab = 128
        wrong = next(
            token for token in range(vocab) if token not in set(ar.tokens)
        )
        one_hot = (mx.arange(vocab) == wrong).astype(mx.float32)
        original_forward = model.mtp_forward

        def rejecting_mtp_forward(hidden_states, next_token_ids, **kwargs):
            result = original_forward(hidden_states, next_token_ids, **kwargs)
            if isinstance(result, tuple):
                logits, hidden = result
                forced = mx.broadcast_to(one_hot, logits.shape).astype(logits.dtype)
                return forced, hidden
            forced = mx.broadcast_to(one_hot, result.shape).astype(result.dtype)
            return forced

        model.mtp_forward = rejecting_mtp_forward
        out = generate_mtp1(rt_mtp, prompt, max_tokens=8, sampler=sampler, seed=0)

        # Every draft is wrong; every step must repair to the AR token.
        assert out.tokens == ar.tokens
        assert out.stats.drafted_tokens > 0
        assert out.stats.accepted_drafts == 0
        assert out.stats.rejected_drafts == out.stats.drafted_tokens
    finally:
        runtime.close()


def test_injection_requires_declared_nextn_layer(tmp_path: Path) -> None:
    root, config, spec, manifest_path = _integrated_streamed_hy3(tmp_path)
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    mtp_fixtures._write_tiny_artifacts(mtp_dir)
    runtime = _open_streamed_runtime(root, spec, manifest_path)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        resident.model.args.num_nextn_predict_layers = 0
        with pytest.raises(Hy3MTPLoadError, match="NextN"):
            inject_hy3_streamed_mtp_support(
                resident.model,
                mtp_dir,
                config,
                MTPContract(),
                expected_revision=TEST_REVISION,
            )
    finally:
        runtime.close()


def _guard_config_dir(tmp_path: Path) -> Path:
    root = tmp_path / "guard-model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "hy_v3"}), encoding="utf-8"
    )
    return root


def test_load_requires_artifacts_for_streamed_mtp(tmp_path: Path) -> None:
    root = _guard_config_dir(tmp_path)
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    with pytest.raises(RuntimeError, match="mtp_artifacts"):
        load(
            root,
            mtp=True,
            expert_streaming_config=config,
            expert_manifest=root / "expert-manifest.json",
        )


def test_load_rejects_streamed_mtp_for_non_hy3_models(tmp_path: Path) -> None:
    root = _guard_config_dir(tmp_path)
    config = ExpertStreamingConfig(
        model_key="glm52-q4",
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    with pytest.raises(RuntimeError, match="hy3-q4 only"):
        load(
            root,
            mtp=True,
            expert_streaming_config=config,
            expert_manifest=root / "expert-manifest.json",
            mtp_artifacts=tmp_path,
        )


def test_load_rejects_mtp_artifacts_without_streaming(tmp_path: Path) -> None:
    root = _guard_config_dir(tmp_path)
    with pytest.raises(ValueError, match="streamed checkpoints only"):
        load(root, mtp=True, mtp_artifacts=tmp_path)


def test_return_hidden_defaults_to_pre_norm_for_the_nextn_head(tmp_path: Path) -> None:
    """Regression for the 20.8%-acceptance gate: the head's hnorm must consume
    the trunk's RAW last-layer hidden, not the lm_head's final-norm output."""
    root, config, spec, manifest_path = _integrated_streamed_hy3(tmp_path)
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    mtp_fixtures._write_tiny_artifacts(mtp_dir)
    runtime = _open_streamed_runtime(root, spec, manifest_path)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        model = resident.model
        injected = inject_hy3_streamed_mtp_support(
            model, mtp_dir, config, MTPContract(), expected_revision=TEST_REVISION
        )
        assert injected is True
        prompt = mx.array([[1, 2, 4]], dtype=mx.int32)

        post_ref, pre_ref = model.model(prompt, return_pre_norm=True)
        _logits, hidden_default = model(prompt, return_hidden=True)
        _logits2, hidden_pre = model(prompt, return_hidden=True, hidden_variant="pre_norm")
        _logits3, hidden_post = model(prompt, return_hidden=True, hidden_variant="post_norm")
        mx.eval(post_ref, pre_ref, hidden_default, hidden_pre, hidden_post)

        assert mx.array_equal(hidden_default, pre_ref).item()
        assert mx.array_equal(hidden_pre, pre_ref).item()
        assert mx.array_equal(hidden_post, post_ref).item()
        assert not mx.array_equal(pre_ref, post_ref).item()
    finally:
        runtime.close()
