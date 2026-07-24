from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.expert_cli as expert_cli
from mtplx.expert_cli import (
    add_expert_streaming_args,
    append_expert_streaming_child_args,
    expert_streaming_load_kwargs,
)
from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
    save_expert_manifest,
)
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_streaming_models import (
    ExpertStreamingModelSpec,
    get_model_spec,
    plan_expert_memory,
)
from mtplx.attention_context import attention_phase
from mtplx.expert_streaming import RoutingPhase
from mtplx.models.expert_mlx import current_expert_routing_phase
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_expert_streaming_args(parser)
    return parser


def _model_root(tmp_path: Path, model_type: str = "hy_v3") -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    (root / "expert-manifest.json").write_text("{}", encoding="utf-8")
    return root


def _streaming_root(
    tmp_path: Path,
    *,
    model_type: str,
    manifest_model_key: str | None = None,
    write_manifest: bool = True,
    name: str = "model",
) -> Path:
    """Minimal streaming artifact: a config.json plus a tiny manifest.

    ``manifest_model_key`` writes a top-level ``model_key`` into the manifest
    (mirroring the published streaming repos). ``None`` writes a manifest with
    no model_key so the config.json fallback is exercised.
    """

    root = tmp_path / name
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    if write_manifest:
        manifest: dict[str, str] = {}
        if manifest_model_key is not None:
            manifest["model_key"] = manifest_model_key
        (root / "expert-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return root


def test_expert_cli_builds_explicit_bounded_config(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
            "--expert-cache-limit",
            "24GiB",
            "--expert-runtime-reserve",
            "8GiB",
            "--no-expert-prefer-sidecar",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    config = kwargs["expert_streaming_config"]

    assert kwargs["mtp"] is False
    assert kwargs["expert_manifest"] == root / "expert-manifest.json"
    assert config.model_key == "hy3-q4"
    assert config.memory_limit_bytes == 96 * 1024**3
    assert config.max_live_kv_tokens == 8192
    assert config.expert_cache_limit_bytes == 24 * 1024**3
    assert config.runtime_reserve_bytes == 8 * 1024**3
    assert config.prefer_sidecar is False


@pytest.mark.parametrize(
    ("model_type", "expected_model_key"),
    [("hy_v3", "hy3-q4"), ("glm_moe_dsa", "glm52-q4")],
)
def test_hy3_default_and_glm_default_stay_on_production_q4(
    tmp_path: Path,
    model_type: str,
    expected_model_key: str,
) -> None:
    root = _model_root(tmp_path, model_type)
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)

    assert kwargs["expert_streaming_config"].model_key == expected_model_key


@pytest.mark.parametrize(
    "model_key",
    ["hy3-expert-only-q4", "hy3-expert-q2", "glm52-expert-q2"],
)
def test_expert_cli_accepts_explicit_expert_only_model_keys(
    model_key: str,
) -> None:
    args = _parser().parse_args(["--expert-model-key", model_key])

    assert args.expert_model_key == model_key


def _tiny_affine_spec(*, bits: int) -> ExpertStreamingModelSpec:
    key = "hy3-expert-q2" if bits == 2 else "hy3-expert-only-q4"
    weight_bytes = 64 * 64 * bits // 8
    parameter_bytes = 64 * 2
    record_bytes = 3 * (weight_bytes + 2 * parameter_bytes)
    return ExpertStreamingModelSpec(
        key=key,
        display_name=f"Tiny affine Q{bits}",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="local/tiny-hy3",
        quant_revision="quant",
        total_tensor_bytes=record_bytes + 1,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=1,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=bits,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="bfloat16",
        router_matmul_dtype="float32",
        router_bytes=0,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )


def _write_tiny_affine_manifest(
    root: Path,
    spec: ExpertStreamingModelSpec,
) -> Path:
    root.mkdir()
    cursor = 1
    segments = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        output_size = 64
        input_size = 64
        for leaf, dtype, shape, length in (
            (
                "weight",
                "U32",
                (output_size, input_size * spec.quant_bits // 32),
                output_size * input_size * spec.quant_bits // 8,
            ),
            ("scales", "BF16", (output_size, 1), output_size * 2),
            ("biases", "BF16", (output_size, 1), output_size * 2),
        ):
            component = f"{projection}.{leaf}"
            segments.append(
                TensorSegment(
                    component=component,
                    tensor=f"model.layers.1.mlp.switch_mlp.{component}",
                    shard="source.safetensors",
                    offset=cursor,
                    length=length,
                    dtype=dtype,
                    shape=shape,
                )
            )
            cursor += length
    record = ExpertRecord(
        layer=1,
        expert=0,
        logical_bytes=spec.expert_record_bytes,
        segments=tuple(segments),
    )
    shard = root / "source.safetensors"
    shard.write_bytes(b"\0" * (cursor + 1))
    manifest = ExpertManifest(
        model_key=spec.key,
        source_repo=spec.quant_model,
        source_revision=spec.quant_revision,
        quant_bits=spec.quant_bits,
        quant_group_size=spec.quant_group_size,
        quant_mode="affine",
        artifact_tensor_bytes=spec.total_tensor_bytes,
        resident_tensor_bytes=1,
        routed_expert_bytes=spec.routed_expert_bytes,
        shards=(
            ShardInfo(
                name=shard.name,
                size=cursor + 1,
                header_bytes=1,
                header_sha256=hashlib.sha256(b"\0").hexdigest(),
            ),
        ),
        resident_tensors=(
            ResidentTensor(
                tensor="model.norm.flag",
                shard=shard.name,
                offset=cursor,
                length=1,
                dtype="U8",
                shape=(1,),
            ),
        ),
        records=(record,),
    ).with_digest()
    path = root / "expert-manifest.json"
    save_expert_manifest(manifest, path)
    return path


@pytest.mark.parametrize(("manifest_bits", "descriptor_bits"), [(2, 4), (4, 2)])
def test_expert_q2_and_expert_only_q4_manifest_descriptor_mismatch_is_exact(
    tmp_path: Path,
    manifest_bits: int,
    descriptor_bits: int,
) -> None:
    manifest_spec = _tiny_affine_spec(bits=manifest_bits)
    descriptor = replace(
        _tiny_affine_spec(bits=descriptor_bits),
        quant_model=manifest_spec.quant_model,
        quant_revision=manifest_spec.quant_revision,
    )
    root = tmp_path / f"q{manifest_bits}"
    manifest_path = _write_tiny_affine_manifest(root, manifest_spec)
    config = ExpertStreamingConfig(
        model_key=descriptor.key,
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
    )

    with pytest.raises(ExpertStreamingConfigurationError) as failed:
        ExpertStreamingRuntime.open(
            root,
            manifest_path,
            config,
            spec=descriptor,
            apply_memory_cap=False,
        )

    assert str(failed.value) == (
        "manifest does not match pinned model descriptor: "
        f"manifest bits {manifest_bits} do not match descriptor bits {descriptor_bits}"
    )


def test_expert_cli_json_and_flags_are_strict_and_forwarded(tmp_path: Path) -> None:
    root = _model_root(tmp_path, "glm_moe_dsa")
    config_path = tmp_path / "stream.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "glm52-q4",
                "memory_limit_bytes": "256GiB",
                "max_live_kv_tokens": 4096,
                "runtime_reserve_bytes": "12GiB",
            }
        ),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--expert-streaming-config",
            str(config_path),
            "--expert-memory-limit",
            "320GiB",
            "--no-expert-verify-record-hashes",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    assert kwargs["expert_streaming_config"].memory_limit_bytes == 320 * 1024**3
    assert kwargs["expert_streaming_config"].verify_record_hashes is False

    command = ["python", "-m", "mtplx.server.openai"]
    append_expert_streaming_child_args(command, args)
    assert "--expert-streaming" in command
    assert command[command.index("--expert-memory-limit") + 1] == "320GiB"
    assert "--no-expert-verify-record-hashes" in command


@pytest.mark.parametrize(
    ("model_type", "manifest_model_key"),
    [
        ("hy_v3", "hy3-expert-oq2e"),
        ("glm_moe_dsa", "glm52-expert-q1t"),
    ],
)
def test_manifest_model_key_wins_over_shared_config_type(
    tmp_path: Path,
    model_type: str,
    manifest_model_key: str,
) -> None:
    # oQ2e / t158 share a config.json model_type with the production q4 bank,
    # so the manifest's declared spec key must be authoritative when the user
    # passes no --expert-model-key.
    root = _streaming_root(
        tmp_path, model_type=model_type, manifest_model_key=manifest_model_key
    )
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)

    assert kwargs["expert_streaming_config"].model_key == manifest_model_key


def test_manifest_without_model_key_falls_back_to_config_q4(tmp_path: Path) -> None:
    # A manifest that declares no model_key must not upgrade the bank; the
    # coarse config.json model_type map still picks the safe production q4 key.
    root = _streaming_root(tmp_path, model_type="hy_v3", manifest_model_key=None)
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)

    assert kwargs["expert_streaming_config"].model_key == "hy3-q4"


def test_explicit_model_key_overrides_manifest_model_key(tmp_path: Path) -> None:
    root = _streaming_root(
        tmp_path, model_type="hy_v3", manifest_model_key="hy3-expert-oq2e"
    )
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-model-key",
            "hy3-expert-q2",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)

    assert kwargs["expert_streaming_config"].model_key == "hy3-expert-q2"


def test_memory_and_kv_limits_auto_derived_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path, model_type="hy_v3", manifest_model_key="hy3-expert-oq2e"
    )
    monkeypatch.setattr(
        expert_cli, "_installed_ram_bytes", lambda: 128 * 1024**3
    )
    args = _parser().parse_args(["--expert-streaming"])

    config = expert_streaming_load_kwargs(args, root)["expert_streaming_config"]

    # 75% of installed RAM, floored at 8GiB and capped at 192GiB.
    assert config.memory_limit_bytes == 96 * 1024**3
    # The derived KV ceiling prioritizes the expert slot bank over KV: the
    # backward plan search is clamped to _DEFAULT_KV_TOKENS, and this fixture's
    # config.json declares no context window (so _declared_context_window is the
    # larger _DEFAULT_CONTEXT_CEILING). 32768 tokens still fit at a 96 GiB
    # envelope for hy3-expert-oq2e, so the search returns _DEFAULT_KV_TOKENS.
    assert config.max_live_kv_tokens == expert_cli._DEFAULT_KV_TOKENS
    assert config.max_live_kv_tokens < expert_cli._DEFAULT_CONTEXT_CEILING
    assert config.max_live_kv_tokens > 0

    # Regression guard: the derived default must never starve the expert cache
    # to a single slot per layer (the "maximize KV" failure mode). Rebuild the
    # plan at the derived KV ceiling exactly as _derive_max_live_kv_tokens does
    # and assert a healthy slot bank survives.
    plan = plan_expert_memory(
        get_model_spec(config.model_key),
        total_limit_bytes=config.memory_limit_bytes,
        context_tokens=config.max_live_kv_tokens,
        runtime_reserve_bytes=config.runtime_reserve_bytes,
        transient_slots=config.transient_slots,
        io_staging_bytes=config.io_staging_bytes,
        execution_workspace_bytes=config.execution_workspace_bytes,
        kv_quant=config.kv_quant,
        cache_scope=config.cache_scope,
    )
    assert plan.fits_fixed
    assert plan.slots_per_layer > 1
    # The README-tested streaming profile keeps well over 100 slots/layer.
    assert plan.slots_per_layer >= 128

    # Explicit flags always win over the derived defaults.
    explicit = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "64GiB",
            "--expert-max-live-kv-tokens",
            "4096",
        ]
    )
    explicit_config = expert_streaming_load_kwargs(explicit, root)[
        "expert_streaming_config"
    ]
    assert explicit_config.memory_limit_bytes == 64 * 1024**3
    assert explicit_config.max_live_kv_tokens == 4096


def test_derived_memory_limit_too_small_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path, model_type="hy_v3", manifest_model_key="hy3-expert-oq2e"
    )
    # 6 GiB cannot hold the ~9 GiB resident footprint, let alone slots.
    monkeypatch.setattr(expert_cli, "_installed_ram_bytes", lambda: 6 * 1024**3)
    args = _parser().parse_args(["--expert-streaming"])

    with pytest.raises(ValueError, match="too small"):
        expert_streaming_load_kwargs(args, root)


class _PhaseModel:
    def __init__(self) -> None:
        self.phases: list[RoutingPhase] = []

    def __call__(self, input_ids, cache=None):
        del cache
        self.phases.append(current_expert_routing_phase(token_count=999))
        return input_ids


class _StreamingStub:
    def __init__(self) -> None:
        self.closed = False

    def close(self, *, timeout=None) -> None:
        del timeout
        self.closed = True

    def snapshot(self):
        return {"ok": True}

    def admit_kv_tokens(self, tokens):
        return SimpleNamespace(tokens=tokens)


def test_streamed_plain_decode_uses_prebound_eager_route_without_engagement_counter(
    monkeypatch,
) -> None:
    model = _PhaseModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=_StreamingStub(),
    )

    def reject_compiled_eligibility(_cache):
        raise AssertionError("streamed decode re-entered compiled eligibility")

    monkeypatch.setattr(runtime, "_compiled_ar_forward", reject_compiled_eligibility)
    token = SimpleNamespace(shape=(1, 1))

    assert runtime.forward_ar(token, cache=[object()]) is token
    assert "compiled_forward_calls" not in runtime.diagnostic_counters


def test_mtplx_runtime_marks_prefill_decode_and_closes_streaming_runtime() -> None:
    model = _PhaseModel()
    streaming = _StreamingStub()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )

    runtime.forward_ar(SimpleNamespace(shape=(1, 3)))
    runtime.forward_ar(SimpleNamespace(shape=(1, 1)))

    assert model.phases == [RoutingPhase.PREFILL, RoutingPhase.DECODE]
    assert runtime.expert_streaming_snapshot() == {"ok": True}
    runtime.close()
    assert streaming.closed is True


def test_attention_phase_context_overrides_routing_shape_heuristic() -> None:
    model = _PhaseModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=_StreamingStub(),
    )

    # A one-token prefill tail chunk is still prefill traffic.
    with attention_phase("prefill"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 1)))
    # MTP verify batches are decode traffic despite their width.
    with attention_phase("decode_verify"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 2)))
    with attention_phase("ar_decode"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 1)))
    with attention_phase("postcommit"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 2)))
    # Unrecognized phases normalize to unknown and keep the heuristic.
    with attention_phase("ar_batch_shared_prefill"):
        runtime.forward_ar(SimpleNamespace(shape=(1, 3)))
        runtime.forward_ar(SimpleNamespace(shape=(1, 1)))

    assert model.phases == [
        RoutingPhase.PREFILL,
        RoutingPhase.DECODE,
        RoutingPhase.DECODE,
        RoutingPhase.DECODE,
        RoutingPhase.PREFILL,
        RoutingPhase.DECODE,
    ]
    runtime.close()
