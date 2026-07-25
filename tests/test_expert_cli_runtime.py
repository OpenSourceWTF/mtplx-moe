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
    expert_streaming_requested,
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


@pytest.fixture(autouse=True)
def _admitted_artifact_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        expert_cli,
        "ensure_expert_admitted",
        lambda _root: {
            "manifest_sha256": "a" * 64,
            "banks": [{"sha256": "b" * 64}],
        },
    )
    profile = expert_cli.load_expert_profiles()["hy3-oq2e-96"]
    monkeypatch.setattr(
        expert_cli,
        "select_expert_profile",
        lambda requested, *, model_key: (
            profile
            if model_key == profile.model_key
            else (_ for _ in ()).throw(
                ValueError(f"no promoted expert profiles match model key {model_key!r}")
            )
        ),
    )


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


def test_expert_profile_is_public_and_defaults_to_auto() -> None:
    args = _parser().parse_args([])

    assert args.expert_profile == "auto"
    assert expert_streaming_requested(args) is False


def test_concrete_expert_profile_implies_streaming_and_is_forwarded() -> None:
    args = _parser().parse_args(["--expert-profile", "hy3-oq2e-64"])
    args._cli_flags = {"expert-profile"}
    command = ["python", "-m", "mtplx.server.openai"]

    assert expert_streaming_requested(args) is True
    append_expert_streaming_child_args(command, args)

    assert command[command.index("--expert-profile") + 1] == "hy3-oq2e-64"


def test_concrete_profile_rejects_explicit_streaming_opt_out() -> None:
    args = _parser().parse_args(
        ["--expert-profile", "hy3-oq2e-64", "--no-expert-streaming"]
    )
    args._cli_flags = {"expert-profile", "no-expert-streaming"}

    with pytest.raises(
        ValueError,
        match="--expert-profile cannot be combined with --no-expert-streaming",
    ):
        expert_streaming_requested(args)


@pytest.mark.parametrize(
    ("argv", "cli_flags", "selector"),
    [
        (
            ["--expert-streaming", "--no-expert-streaming"],
            {"expert-streaming", "no-expert-streaming"},
            "--expert-streaming",
        ),
        (
            ["--expert-profile", "hy3-oq2e-64", "--no-expert-streaming"],
            {"expert-profile", "no-expert-streaming"},
            "--expert-profile",
        ),
        (
            ["--expert-streaming-config", "stream.json", "--no-expert-streaming"],
            {"expert-streaming-config", "no-expert-streaming"},
            "--expert-streaming-config",
        ),
        (
            ["--expert-manifest", "manifest.json", "--no-expert-streaming"],
            {"expert-manifest", "no-expert-streaming"},
            "--expert-manifest",
        ),
        (
            ["--expert-cache-policy", "lru", "--no-expert-streaming"],
            {"expert-cache-policy", "no-expert-streaming"},
            "--expert-cache-policy",
        ),
    ],
)
def test_explicit_streaming_opt_out_rejects_every_positive_selector(
    argv: list[str],
    cli_flags: set[str],
    selector: str,
) -> None:
    args = _parser().parse_args(argv)
    args._cli_flags = cli_flags

    with pytest.raises(ValueError, match=selector):
        expert_streaming_requested(args)


def test_promoted_profile_admits_then_freezes_auto_and_attaches_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
    )
    events: list[str] = []
    receipt = {
        "manifest_sha256": "a" * 64,
        "banks": [{"sha256": "b" * 64}],
    }
    monkeypatch.setattr(
        expert_cli,
        "ensure_expert_admitted",
        lambda admitted_root: events.append(f"admit:{admitted_root}") or receipt,
    )
    profile = expert_cli.load_expert_profiles()["hy3-oq2e-64"]
    monkeypatch.setattr(
        expert_cli,
        "select_expert_profile",
        lambda requested, *, model_key: (
            events.append(f"profile:{requested}:{model_key}") or profile
        ),
    )
    args = _parser().parse_args(["--expert-streaming"])

    kwargs = expert_streaming_load_kwargs(args, root)

    assert events == [
        f"admit:{root.resolve()}",
        "profile:auto:hy3-expert-oq2e",
    ]
    assert args.expert_profile == "hy3-oq2e-64"
    assert args._resolved_expert_profile is profile
    assert args._expert_admission_receipt is receipt
    assert kwargs["expert_admission_receipt"] is receipt
    assert kwargs["expert_streaming_config"].model_key == "hy3-expert-oq2e"
    assert kwargs["mtp"] is False


def test_promoted_profile_geometry_override_is_customized_in_parent_and_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
    )
    profile = expert_cli.load_expert_profiles()["hy3-oq2e-96"]
    monkeypatch.setattr(
        expert_cli,
        "select_expert_profile",
        lambda requested, *, model_key: profile,
    )
    args = _parser().parse_args(
        [
            "--expert-profile",
            profile.name,
            "--expert-cache-policy",
            "lru",
        ]
    )
    args._cli_flags = {"expert-profile", "expert-cache-policy"}

    kwargs = expert_streaming_load_kwargs(args, root)

    assert kwargs["expert_streaming_config"].cache_policy == "lru"
    assert args._resolved_expert_profile is profile
    assert args._resolved_expert_profile_customized is True
    assert "cache_policy" in args._resolved_expert_profile_customized_fields
    assert args._resolved_expert_effective_config["cache_policy"] == "lru"

    child_command: list[str] = []
    append_expert_streaming_child_args(child_command, args)
    child = _parser().parse_args(child_command)
    child._cli_flags = {
        token[2:] for token in child_command if token.startswith("--")
    }

    child_kwargs = expert_streaming_load_kwargs(child, root)

    assert child_kwargs["expert_streaming_config"].cache_policy == "lru"
    assert child._resolved_expert_profile_customized is True
    assert child._resolved_expert_profile_customized_fields == (
        "cache_policy",
    )
    assert child._resolved_expert_effective_config["cache_policy"] == "lru"


def test_promoted_profile_unknown_override_is_a_controlled_value_error(
    tmp_path: Path,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
    )
    config_path = tmp_path / "unknown-key.json"
    config_path.write_text(
        json.dumps({"not_a_profile_field": 1}),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--expert-profile",
            "hy3-oq2e-96",
            "--expert-streaming-config",
            str(config_path),
        ]
    )

    with pytest.raises(
        ValueError,
        match="invalid expert profile overrides.*not_a_profile_field",
    ):
        expert_streaming_load_kwargs(args, root)


def test_explicit_load_mtp_is_rejected_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
    )
    args = _parser().parse_args(["--expert-profile", "hy3-oq2e-96"])
    args.load_mtp = True
    args._cli_flags = {"expert-profile", "load-mtp"}
    monkeypatch.setattr(
        expert_cli,
        "ensure_expert_admitted",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("MTP contradiction must fail before admission")
        ),
    )

    with pytest.raises(ValueError, match="AR-only"):
        expert_streaming_load_kwargs(args, root)


def test_legacy_config_for_unpromoted_glm_is_admitted_without_hy3_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="glm_moe_dsa",
        manifest_model_key="glm52-expert-q2",
    )
    config_path = tmp_path / "legacy.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "glm52-expert-q2",
                "memory_limit_bytes": "256GiB",
                "max_live_kv_tokens": 4096,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        expert_cli,
        "select_expert_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported legacy model must not select a Hy3 profile")
        ),
    )
    args = _parser().parse_args(
        ["--expert-streaming-config", str(config_path)]
    )

    kwargs = expert_streaming_load_kwargs(args, root)

    assert kwargs["expert_streaming_config"].model_key == "glm52-expert-q2"
    assert args._resolved_expert_profile is None


def test_admitted_manifest_model_key_cannot_be_replaced_by_cli(
    tmp_path: Path,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
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

    with pytest.raises(ValueError, match="does not match admitted manifest"):
        expert_streaming_load_kwargs(args, root)


def test_record_hash_diagnostic_requires_cli_flag_provenance(
    tmp_path: Path,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="glm_moe_dsa",
        manifest_model_key="glm52-expert-q2",
    )
    config_path = tmp_path / "legacy.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "glm52-expert-q2",
                "memory_limit_bytes": "256GiB",
                "max_live_kv_tokens": 4096,
                "verify_record_hashes": True,
            }
        ),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        ["--expert-streaming-config", str(config_path)]
    )
    args._cli_flags = {"expert-streaming-config"}

    config = expert_streaming_load_kwargs(args, root)[
        "expert_streaming_config"
    ]
    assert config.verify_record_hashes is False

    explicit = _parser().parse_args(
        [
            "--expert-streaming-config",
            str(config_path),
            "--expert-verify-record-hashes",
        ]
    )
    explicit._cli_flags = {
        "expert-streaming-config",
        "expert-verify-record-hashes",
    }
    explicit_config = expert_streaming_load_kwargs(explicit, root)[
        "expert_streaming_config"
    ]
    assert explicit_config.verify_record_hashes is True


def test_external_manifest_must_match_admitted_root_manifest(
    tmp_path: Path,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
    )
    outside = tmp_path / "other-manifest.json"
    outside.write_text(
        json.dumps({"model_key": "hy3-expert-q2"}),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-manifest",
            str(outside),
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
        ]
    )

    with pytest.raises(ValueError, match="admitted root manifest"):
        expert_streaming_load_kwargs(args, root)


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


def test_runtime_load_accepts_admission_receipt_through_real_reader_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.expert_runtime as expert_runtime_module
    import mtplx.models.expert_mlx as expert_mlx_module
    import mtplx.resident_loader as resident_loader_module
    import mtplx.runtime as runtime_module
    from mtplx.expert_admission import admit_expert_artifact
    from mtplx.expert_streaming_models import MODEL_SPECS
    from test_expert_manifest import _make_authoritative_checkpoint

    root = tmp_path / "admitted"
    spec, manifest = _make_authoritative_checkpoint(root)
    save_expert_manifest(
        manifest,
        root / "expert-manifest.json",
    )
    manifest_path = root / "expert-manifest.json"
    (root / "config.json").write_text(
        json.dumps({"model_type": "hy_v3"}),
        encoding="utf-8",
    )
    monkeypatch.setitem(MODEL_SPECS, spec.key, spec)
    receipt = admit_expert_artifact(
        root,
        revision="a" * 40,
        receipt_root=tmp_path / "receipts",
    )
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=1 << 20,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        expert_cache_limit_bytes=spec.expert_record_bytes,
        transient_slots=1,
        verify_artifact_headers=False,
        hy3_router_kernel="stock",
    )
    reached: dict[str, object] = {}

    class LaterResidentBoundary(RuntimeError):
        pass

    def stop_after_reader_open(model_path, expert_runtime, *, config):
        reached["model_path"] = model_path
        reached["backend"] = expert_runtime.reader.backend
        reached["pinned_banks"] = tuple(
            expert_runtime.reader._pinned_entries
        )
        raise LaterResidentBoundary("reader construction reached")

    monkeypatch.setattr(
        expert_runtime_module,
        "apply_mlx_memory_cap",
        lambda *_args, **_kwargs: {"applied": False},
    )
    monkeypatch.setattr(
        expert_mlx_module,
        "make_mlx_slot_buffer_allocator",
        lambda _plan, _spec: lambda size, _label: bytearray(size),
    )
    monkeypatch.setattr(
        resident_loader_module,
        "construct_resident_model",
        stop_after_reader_open,
    )

    with pytest.raises(LaterResidentBoundary, match="reader construction reached"):
        runtime_module.load(
            root,
            mtp=False,
            expert_streaming_config=config,
            expert_manifest=manifest_path,
            expert_admission_receipt=receipt,
        )

    assert reached["model_path"] == root
    assert reached["backend"] in {"native", "python-preadv"}
    assert reached["pinned_banks"] == tuple(
        bank["file"] for bank in receipt["banks"]
    )


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


def test_manifest_model_key_supports_repository_local_hf_blob_symlink(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "models--owner--streamed"
        / "snapshots"
        / "revision"
    )
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "hy_v3"}),
        encoding="utf-8",
    )
    blob_root = root.parents[1] / "blobs"
    blob_root.mkdir()
    blob = blob_root / "manifest-digest"
    blob.write_text(
        json.dumps({"model_key": "hy3-expert-oq2e"}),
        encoding="utf-8",
    )
    manifest_path = root / "expert-manifest.json"
    manifest_path.symlink_to(Path("..") / ".." / "blobs" / blob.name)

    assert expert_cli._read_manifest_model_key(manifest_path) == "hy3-expert-oq2e"


def test_default_manifest_rejects_external_model_root_symlink(
    tmp_path: Path,
) -> None:
    root = _streaming_root(
        tmp_path,
        model_type="hy_v3",
        manifest_model_key="hy3-expert-oq2e",
    )
    manifest_path = root / "expert-manifest.json"
    outside = tmp_path / "external-manifest.json"
    manifest_path.replace(outside)
    manifest_path.symlink_to(outside)
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
        ]
    )

    with pytest.raises(ValueError, match="escapes root"):
        expert_streaming_load_kwargs(args, root)


def test_default_manifest_accepts_repository_local_hf_blob_symlink(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "models--owner--streamed"
        / "snapshots"
        / "revision"
    )
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "hy_v3"}),
        encoding="utf-8",
    )
    blob_root = root.parents[1] / "blobs"
    blob_root.mkdir()
    blob = blob_root / "manifest-digest"
    blob.write_text(
        json.dumps({"model_key": "hy3-expert-oq2e"}),
        encoding="utf-8",
    )
    manifest_path = root / "expert-manifest.json"
    manifest_path.symlink_to(Path("..") / ".." / "blobs" / blob.name)
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

    assert kwargs["expert_manifest"] == blob.resolve()
    assert kwargs["expert_streaming_config"].model_key == "hy3-expert-oq2e"


@pytest.mark.parametrize("oversized", [False, True])
def test_manifest_model_key_rejects_unsafe_symlink_targets(
    tmp_path: Path,
    oversized: bool,
) -> None:
    root = (
        tmp_path
        / "models--owner--streamed"
        / "snapshots"
        / "revision"
    )
    root.mkdir(parents=True)
    if oversized:
        from mtplx.expert_manifest import MAX_MANIFEST_BYTES

        blob_root = root.parents[1] / "blobs"
        blob_root.mkdir()
        target = blob_root / "oversized-manifest"
        with target.open("wb") as handle:
            handle.write(b'{"model_key":"hy3-expert-oq2e","padding":"')
            handle.truncate(MAX_MANIFEST_BYTES + 1)
    else:
        target = tmp_path / "external-manifest.json"
        target.write_text(
            json.dumps({"model_key": "hy3-expert-oq2e"}),
            encoding="utf-8",
        )
    manifest_path = root / "expert-manifest.json"
    manifest_path.symlink_to(target)

    assert expert_cli._read_manifest_model_key(manifest_path) is None


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


def test_explicit_model_key_cannot_override_manifest_model_key(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="does not match admitted manifest"):
        expert_streaming_load_kwargs(args, root)


def test_memory_and_kv_limits_auto_derived_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _streaming_root(
        tmp_path, model_type="hy_v3", manifest_model_key="hy3-expert-q2"
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
    assert plan.slots_per_layer >= 100

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
        tmp_path, model_type="hy_v3", manifest_model_key="hy3-expert-q2"
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
