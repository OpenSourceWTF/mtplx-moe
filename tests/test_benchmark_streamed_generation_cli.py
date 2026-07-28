"""Pin the benchmark harness defaults that guarantee run comparability."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "benchmark_streamed_generation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_streamed_generation", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unflagged_runs_are_reproducible_and_bounded() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(
        [
            "/model",
            "/manifest",
            "--model-key",
            "hy3-q4",
            "--memory-limit",
            "112GiB",
            "--max-live-kv-tokens",
            "2048",
        ]
    )
    # A run with no sampling/length flags must be deterministic and bounded:
    # silent default drift here is what makes old and new results
    # incomparable (review finding 9).
    assert args.generation_profile == "deterministic"
    assert args.max_tokens == 256
    assert args.window_telemetry is True
    assert args.window_tokens == 32
    assert args.seed == 0


@pytest.mark.parametrize("model_key", ["hy3-expert-only-q4", "hy3-expert-q2"])
def test_benchmark_accepts_explicit_hy3_expert_lanes_with_hy3_defaults(
    model_key: str,
) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "/model",
            "/manifest",
            "--model-key",
            model_key,
            "--memory-limit",
            "112GiB",
            "--max-live-kv-tokens",
            "2048",
        ]
    )

    assert args.model_key == model_key
    assert module.model_defaults_for_key(model_key) == (
        module.model_defaults_for_key("hy3-q4")
    )
    assert module.model_defaults_for_key(model_key) == {
        "max_tokens": 65_536,
        "max_output_tokens": 262_144,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "reasoning_effort": None,
    }


def test_glm52_expert_q2_benchmark_uses_glm52_q4_defaults() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "/model",
            "/manifest",
            "--model-key",
            "glm52-expert-q2",
            "--memory-limit",
            "112GiB",
            "--max-live-kv-tokens",
            "2048",
        ]
    )

    assert args.model_key == "glm52-expert-q2"
    assert module.model_defaults_for_key("glm52-expert-q2") == (
        module.model_defaults_for_key("glm52-q4")
    )
    assert module.model_defaults_for_key("glm52-expert-q2") == {
        "max_tokens": 65_536,
        "max_output_tokens": 131_072,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "enable_thinking": True,
        "reasoning_effort": "max",
    }


def test_window_telemetry_can_be_disabled() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(
        [
            "/model",
            "/manifest",
            "--model-key",
            "hy3-q4",
            "--memory-limit",
            "112GiB",
            "--max-live-kv-tokens",
            "2048",
            "--no-window-telemetry",
        ]
    )
    assert args.window_telemetry is False


def test_configuration_summary_exports_explicit_cache_and_batch_identity() -> None:
    module = _load_module()
    settings = {
        "cache_policy": "frequency",
        "expert_cache_limit_bytes": None,
        "max_prefills_per_step": 2,
        "memory_limit_bytes": 112 * 1024**3,
        "max_live_kv_tokens": 2048,
    }

    summary = module.build_configuration_summary(
        "fixture",
        cache_scope="global",
        slot_layout="component-banks",
        concurrency=4,
        execution_lane="continuous-batch-ar",
        performance_settings=settings,
    )

    assert summary["run_label"] == "fixture"
    assert summary["configuration_label"].startswith(
        "cache-global-layout-component-banks-B4-lane-continuous-batch-ar-cfg-"
    )
    assert len(summary["configuration_fingerprint"]) == 16
    assert summary["cache_scope"] == "global"
    assert summary["slot_layout"] == "component-banks"
    assert summary["concurrency"] == 4
    assert summary["requested_concurrency"] == 4
    assert summary["execution_lane"] == "continuous-batch-ar"
    assert summary["performance_settings"] == settings


def test_configuration_identity_is_deterministic_and_covers_performance_arms() -> None:
    module = _load_module()
    base = module.build_stable_performance_settings(
        runtime_config={
            "cache_policy": "frequency",
            "memory_limit_bytes": 100,
            "max_live_kv_tokens": 2048,
            "runtime_reserve_bytes": 10,
            "expert_cache_limit_bytes": 20,
            "transient_slots": 8,
            "max_read_chunk_bytes": 1024,
            "bypass_page_cache": False,
        },
        sampler={"temperature": 0.0, "top_p": 1.0, "top_k": 1},
        seed=0,
        prompt_identity={"content_sha256": "prompt-a", "token_sha256": "tokens-a"},
        prompt_options={
            "chat": False,
            "system_prompt": "system",
            "prompt_style": "coding-agent",
            "enable_thinking": False,
            "reasoning_effort": None,
        },
        generation={
            "max_tokens": 256,
            "window_tokens": 32,
            "window_telemetry": True,
            "context_tokens": None,
        },
        scheduler={"max_prefills_per_step": 1},
        mtp={"enabled": False, "precision": "bf16", "artifact_identity": None},
        model_artifact={"manifest_sha256": "manifest-a", "source_revision": "rev-a"},
    )

    def identity(settings: dict) -> str:
        return module.build_configuration_summary(
            "fixture",
            cache_scope="global",
            slot_layout="component-banks",
            concurrency=4,
            execution_lane="continuous-batch-ar",
            performance_settings=settings,
        )["configuration_fingerprint"]

    assert identity(dict(base)) == identity(dict(reversed(list(base.items()))))
    for section, key, value in (
        ("runtime_config", "cache_policy", "lru"),
        ("runtime_config", "memory_limit_bytes", 200),
        ("runtime_config", "max_live_kv_tokens", 4096),
        ("runtime_config", "runtime_reserve_bytes", 11),
        ("runtime_config", "expert_cache_limit_bytes", 21),
        ("runtime_config", "transient_slots", 9),
        ("runtime_config", "max_read_chunk_bytes", 2048),
        ("runtime_config", "bypass_page_cache", True),
        ("sampler", "temperature", 0.5),
        ("sampler", "top_p", 0.9),
        ("sampler", "top_k", 8),
        (None, "seed", 1),
        ("prompt_identity", "content_sha256", "prompt-b"),
        ("prompt_identity", "token_sha256", "tokens-b"),
        ("prompt_options", "chat", True),
        ("prompt_options", "system_prompt", "different"),
        ("prompt_options", "prompt_style", "legacy-repeat"),
        ("prompt_options", "enable_thinking", True),
        ("prompt_options", "reasoning_effort", "max"),
        ("generation", "max_tokens", 128),
        ("generation", "window_tokens", 16),
        ("generation", "window_telemetry", False),
        ("generation", "context_tokens", 1024),
        ("scheduler", "max_prefills_per_step", 2),
        ("mtp", "enabled", True),
        ("mtp", "precision", "q4"),
        ("mtp", "artifact_identity", {"files": ["different"]}),
        ("model_artifact", "manifest_sha256", "manifest-b"),
        ("model_artifact", "harness_source", {"source_sha256": "source-b"}),
    ):
        changed = json.loads(json.dumps(base))
        if section is None:
            changed[key] = value
        else:
            changed[section][key] = value
        assert identity(changed) != identity(base)

    for cache_scope, slot_layout, concurrency in (
        ("layer", "component-banks", 4),
        ("global", "direct-slots", 4),
        ("global", "component-banks", 2),
    ):
        changed_identity = module.build_configuration_summary(
            "fixture",
            cache_scope=cache_scope,
            slot_layout=slot_layout,
            concurrency=concurrency,
            execution_lane="continuous-batch-ar",
            performance_settings=base,
        )["configuration_fingerprint"]
        assert changed_identity != identity(base)


def test_content_identities_are_path_independent_and_byte_sensitive(
    tmp_path: Path,
) -> None:
    module = _load_module()
    first = tmp_path / "first" / "expert-manifest.json"
    second = tmp_path / "second" / "expert-manifest.json"
    first.parent.mkdir()
    second.parent.mkdir()
    payload = {
        "model_key": "hy3-q4",
        "source_revision": "revision-a",
        "manifest_sha256": "declared-digest",
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    first.write_bytes(encoded)
    second.write_bytes(encoded)

    assert module.build_expert_manifest_identity(first) == (
        module.build_expert_manifest_identity(second)
    )
    original = module.build_expert_manifest_identity(first)
    payload["source_revision"] = "revision-b"
    first.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    assert module.build_expert_manifest_identity(first) != original

    assert module.build_prompt_identity("same", [1, 2, 3]) == (
        module.build_prompt_identity("same", [1, 2, 3])
    )
    assert module.build_prompt_identity("changed", [1, 2, 3]) != (
        module.build_prompt_identity("same", [1, 2, 3])
    )
    assert module.build_prompt_identity("same", [1, 2, 4]) != (
        module.build_prompt_identity("same", [1, 2, 3])
    )


def test_mtp_identity_hashes_small_headers_not_machine_paths(tmp_path: Path) -> None:
    module = _load_module()
    header = json.dumps({"__metadata__": {"source_revision": "rev"}}).encode()
    artifact = len(header).to_bytes(8, "little") + header + b"payload"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "layer80-bf16.safetensors").write_bytes(artifact)

    assert module.build_mtp_artifact_identity(first, precision="bf16") == (
        module.build_mtp_artifact_identity(second, precision="bf16")
    )
    original = module.build_mtp_artifact_identity(first, precision="bf16")
    changed_header = json.dumps(
        {"__metadata__": {"source_revision": "different"}}
    ).encode()
    (first / "layer80-bf16.safetensors").write_bytes(
        len(changed_header).to_bytes(8, "little") + changed_header + b"payload"
    )
    assert module.build_mtp_artifact_identity(first, precision="bf16") != original

    body_only = bytearray(artifact)
    body_only[-1] ^= 0xFF
    (second / "layer80-bf16.safetensors").write_bytes(body_only)
    assert module.build_mtp_artifact_identity(second, precision="bf16") != original


def test_mtp_identity_rejects_authenticated_declaration_that_differs_from_bytes(
    tmp_path: Path,
) -> None:
    module = _load_module()
    header = b"{}"
    artifact = len(header).to_bytes(8, "little") + header + b"payload"
    filename = "layer80-bf16.safetensors"
    (tmp_path / filename).write_bytes(artifact)
    unsigned = {
        "format": "mtplx-artifact-digests-v1",
        "files": {filename: {"size": len(artifact), "sha256": "0" * 64}},
    }
    payload = {
        **unsigned,
        "manifest_sha256": hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (tmp_path / "artifact-digests.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="digest differs"):
        module.build_mtp_artifact_identity(tmp_path, precision="bf16")


def test_nocache_receipt_reuse_and_stat_invalidation(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"first")
    receipt = tmp_path / "receipt.json"
    calls = []
    monkeypatch.setattr(module.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(module.fcntl, "fcntl", lambda *args: calls.append(args) or 0)

    first = module._verified_file_digest(
        artifact, require_nocache=True, receipt_path=receipt
    )
    second = module._verified_file_digest(
        artifact, require_nocache=True, receipt_path=receipt
    )
    artifact.write_bytes(b"second-and-different")
    third = module._verified_file_digest(
        artifact, require_nocache=True, receipt_path=receipt
    )

    assert first["receipt_reused"] is False
    assert second["receipt_reused"] is True
    assert third["receipt_reused"] is False
    assert first["sha256"] != third["sha256"]
    assert len(calls) == 2


def test_required_nocache_fails_closed_and_settle_is_mockable(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bytes")
    monkeypatch.delattr(module.fcntl, "F_NOCACHE", raising=False)
    with pytest.raises(RuntimeError, match="non-caching verification is unavailable"):
        module._verified_file_digest(artifact, require_nocache=True)

    sleeps = []
    assert module.settle_after_artifact_verification(
        {"verification_method": "streamed_full_file_sha256"},
        3.5,
        sleeper=sleeps.append,
    )
    assert sleeps == [3.5]
    assert not module.settle_after_artifact_verification(
        {"verification_method": "versioned_prior_nocache_receipt"},
        3.5,
        sleeper=sleeps.append,
    )
    assert sleeps == [3.5]


def test_model_identity_covers_small_resident_bytes_and_reuses_sidecar_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    sidecar_bytes = b"verified expert bytes"
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()

    def validated_manifest(_path, *, verify_digest):
        assert verify_digest is True
        return SimpleNamespace(
            sidecar=SimpleNamespace(
                file="experts.bin", size=len(sidecar_bytes), sha256=sidecar_sha256
            ),
            resident_tensors=(),
        )

    monkeypatch.setattr(module, "load_expert_manifest", validated_manifest)
    identities = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
        (root / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
        manifest = root / "expert-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "model_key": "hy3-q4",
                    "source_revision": "rev",
                    "manifest_sha256": "manifest",
                    "sidecar": {
                        "file": "experts.bin",
                        "size": len(sidecar_bytes),
                        "sha256": sidecar_sha256,
                    },
                    "resident_tensors": [],
                    "shards": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "experts.bin").write_bytes(sidecar_bytes)
        identities.append(module.build_model_artifact_identity(root, manifest))

    assert identities[0] == identities[1]
    assert identities[0]["expert_payload"]["method"] == "verified_sidecar_sha256"
    assert identities[0]["expert_payload"]["sha256"] == sidecar_sha256
    (tmp_path / "first" / "config.json").write_text(
        '{"hidden_size": 16}', encoding="utf-8"
    )
    assert (
        module.build_model_artifact_identity(
            tmp_path / "first", tmp_path / "first" / "expert-manifest.json"
        )
        != identities[0]
    )
    (tmp_path / "first" / "experts.bin").write_bytes(b"mutated expert bytes")
    with pytest.raises(Exception, match="sidecar (size|digest) differs"):
        module.build_model_artifact_identity(
            tmp_path / "first", tmp_path / "first" / "expert-manifest.json"
        )


def test_model_identity_hashes_every_multipart_sidecar_without_scalar_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    part_payloads = (b"first expert part", b"second expert part")
    parts = tuple(
        SimpleNamespace(
            file=f"experts-{index}.bin",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for index, payload in enumerate(part_payloads)
    )

    def validated_manifest(_path, *, verify_digest):
        assert verify_digest is True
        return SimpleNamespace(
            sidecar=SimpleNamespace(parts=parts),
            resident_tensors=(),
        )

    monkeypatch.setattr(module, "load_expert_manifest", validated_manifest)
    root = tmp_path / "model"
    root.mkdir()
    manifest = root / "expert-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_key": "kimi-k3-q1t",
                "source_revision": "rev",
                "manifest_sha256": "manifest",
            }
        ),
        encoding="utf-8",
    )
    for part, payload in zip(parts, part_payloads, strict=True):
        (root / part.file).write_bytes(payload)

    identity = module.build_model_artifact_identity(root, manifest)

    assert identity["expert_payload"]["method"] == ("verified_multipart_sidecar_sha256")
    assert [part["file"] for part in identity["expert_payload"]["parts"]] == [
        "experts-0.bin",
        "experts-1.bin",
    ]
    assert identity["expert_payload"]["size"] == sum(map(len, part_payloads))


def test_benchmark_accepts_service_streaming_config_for_kimi_k3(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config_path = tmp_path / "kimi-streaming.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "kimi-k3-q1t",
                "memory_limit_bytes": 128 * 1024**3,
                "max_live_kv_tokens": 1024,
                "runtime_reserve_bytes": 0,
                "slot_layout": "component-banks",
            }
        ),
        encoding="utf-8",
    )
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "/model",
            "/manifest",
            "--expert-streaming-config",
            str(config_path),
        ]
    )

    config = module.resolve_streaming_config(parser, args)

    assert config.model_key == "kimi-k3-q1t"
    assert config.memory_limit_bytes == 128 * 1024**3
    assert config.slot_layout == "component-banks"
    assert args.model_key == "kimi-k3-q1t"
    assert args.max_live_kv_tokens == 1024


def test_harness_source_identity_detects_uncommitted_source_change(
    tmp_path: Path,
) -> None:
    module = _load_module()
    (tmp_path / "mtplx").mkdir()
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "mtplx" / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "benchmark_streamed_generation.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (tmp_path / "scripts" / "analyze_expert_route_trace.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text("project(x)\n")

    before = module.build_harness_source_identity(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = module.build_harness_source_identity(tmp_path)

    assert before["source_sha256"] != after["source_sha256"]
    assert before.get("git_head") == after.get("git_head")
    before_cmake = after
    cmake.write_text("project(y)\n")
    assert (
        module.build_harness_source_identity(tmp_path)["source_sha256"]
        != (before_cmake["source_sha256"])
    )


def test_harness_source_identity_hashes_imported_native_extension_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    (tmp_path / "mtplx").mkdir()
    (tmp_path / "mtplx" / "runtime.py").write_text("VALUE = 1\n")
    binary = tmp_path / "_ext.so"
    binary.write_bytes(b"native-one")

    def find_spec(name):
        if name == "mtplx_native_expert_io._ext":
            return SimpleNamespace(origin=str(binary))
        return None

    monkeypatch.setattr(module.importlib.util, "find_spec", find_spec)
    before = module.build_harness_source_identity(tmp_path)
    binary.write_bytes(b"native-two")
    after = module.build_harness_source_identity(tmp_path)

    assert before["source_sha256"] != after["source_sha256"]
    assert before["native_binaries"][0]["module"] == ("mtplx_native_expert_io._ext")


def test_response_filename_preserves_legacy_names_and_bounds_long_labels() -> None:
    module = _load_module()

    assert module.build_response_filename("hy3-q4", "fixture", 0) == (
        "hy3-q4-fixture-repeat-0.md"
    )
    assert (
        module.build_response_filename("hy3-q4", "fixture", 0, request_id="stream-00")
        == "hy3-q4-fixture-repeat-0-stream-00.md"
    )
    long_name = module.build_response_filename(
        "hy3-q4", "x" * 230, 0, request_id="stream-00"
    )
    assert len(long_name.encode("utf-8")) <= 255
    assert long_name == module.build_response_filename(
        "hy3-q4", "x" * 230, 0, request_id="stream-00"
    )


def test_response_writer_never_overwrites_different_configuration_arms(
    tmp_path: Path,
) -> None:
    module = _load_module()

    first = module.write_response_file(
        tmp_path,
        "first\n",
        model_key="hy3-q4",
        run_label="fixture",
        repeat=0,
        configuration_fingerprint="a" * 16,
    )
    second = module.write_response_file(
        tmp_path,
        "second\n",
        model_key="hy3-q4",
        run_label="fixture",
        repeat=0,
        configuration_fingerprint="b" * 16,
    )

    assert first.name == "hy3-q4-fixture-repeat-0.md"
    assert second != first
    assert first.read_text() == "first\n"
    assert second.read_text() == "second\n"
    with pytest.raises(FileExistsError, match="already exists"):
        module.write_response_file(
            tmp_path,
            "third\n",
            model_key="hy3-q4",
            run_label="fixture",
            repeat=0,
            configuration_fingerprint="b" * 16,
        )


def test_response_writer_never_leaves_partial_final_on_publish_failure(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publish failed")),
    )

    with pytest.raises(OSError, match="publish failed"):
        module.write_response_file(
            tmp_path,
            "partial evidence must not escape\n",
            model_key="hy3-q4",
            run_label="fixture",
            repeat=0,
            configuration_fingerprint="a" * 16,
        )
    assert list(tmp_path.iterdir()) == []


def test_json_evidence_targets_are_reserved_exclusively_and_disjoint(
    tmp_path: Path,
) -> None:
    module = _load_module()
    output = tmp_path / "result.json"
    trace = tmp_path / "trace.json"

    reservation = module.reserve_json_evidence_targets(output, trace)
    try:
        assert not output.exists()
        assert not trace.exists()
        with pytest.raises(FileExistsError, match="already exists"):
            module.reserve_json_evidence_targets(output, None)
    finally:
        reservation.cleanup()
    assert not output.exists()
    assert not trace.exists()

    with pytest.raises(ValueError, match="must be different"):
        module.reserve_json_evidence_targets(output, output)


def test_json_evidence_reservation_is_race_safe_across_threads(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "result.json"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(module.reserve_json_evidence_targets, target, None)
            for _ in range(2)
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError:
                outcomes.append(None)

    winners = [outcome for outcome in outcomes if outcome is not None]
    assert len(winners) == 1
    winners[0].cleanup()


def test_json_evidence_commit_fails_if_owner_lock_is_replaced(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "result.json"
    reservation = module.reserve_json_evidence_targets(target, None)
    lock = tmp_path / ".result.json.lock"
    lock.unlink()
    lock.write_text('{"owner_token":"thief"}', encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="ownership changed"):
            reservation.commit(target, "{}\n")
        assert not target.exists()
    finally:
        reservation.cleanup()
        lock.unlink(missing_ok=True)


def test_cli_rejects_same_json_targets_before_model_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    target = tmp_path / "same.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_streamed_generation.py",
            str(tmp_path / "model"),
            str(tmp_path / "manifest.json"),
            "--model-key",
            "hy3-q4",
            "--memory-limit",
            "112GiB",
            "--max-live-kv-tokens",
            "2048",
            "--output-json",
            str(target),
            "--route-trace-json",
            str(target),
        ],
    )
    monkeypatch.setattr(module, "load", lambda *_args, **_kwargs: pytest.fail("loaded"))

    with pytest.raises(ValueError, match="must be different"):
        module.main()


def test_cli_cleans_reserved_json_placeholders_on_preload_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    output = tmp_path / "output.json"
    trace = tmp_path / "trace.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_streamed_generation.py",
            str(tmp_path / "model"),
            str(tmp_path / "missing-manifest.json"),
            "--model-key",
            "hy3-q4",
            "--memory-limit",
            "112GiB",
            "--max-live-kv-tokens",
            "2048",
            "--output-json",
            str(output),
            "--route-trace-json",
            str(trace),
        ],
    )

    with pytest.raises(FileNotFoundError):
        module.main()
    assert not output.exists()
    assert not trace.exists()


def test_default_and_explicit_run_labels_preserve_legacy_values() -> None:
    module = _load_module()
    parser = module.build_parser()

    default_args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4"])
    explicit_args = parser.parse_args(
        [*_BASE_ARGS, "--model-key", "hy3-q4", "--run-label", "chosen"]
    )

    assert module.resolve_run_label(default_args) == "manifest"
    assert module.resolve_run_label(explicit_args) == "chosen"
    assert (
        module.build_response_filename(
            "hy3-q4", module.resolve_run_label(default_args), 0
        )
        == "hy3-q4-manifest-repeat-0.md"
    )
    assert (
        module.build_response_filename(
            "hy3-q4", module.resolve_run_label(explicit_args), 0
        )
        == "hy3-q4-chosen-repeat-0.md"
    )


_BASE_ARGS = [
    "/model",
    "/manifest",
    "--memory-limit",
    "112GiB",
    "--max-live-kv-tokens",
    "2048",
]


def test_resource_telemetry_is_opt_in_and_bounded() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4"])

    assert args.resource_telemetry is False
    assert args.resource_sample_interval == 0.25
    assert args.resource_max_samples == 4096
    assert args.powermetrics is False


def test_resource_telemetry_help_does_not_claim_simultaneous_overlap() -> None:
    help_text = _load_module().build_parser().format_help()

    assert "same-interval I/O/Metal coactivity" in help_text
    assert "I/O/Metal overlap" not in help_text


def test_resource_telemetry_flags_parse_without_enabling_window_walks() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "hy3-q4",
            "--resource-telemetry",
            "--resource-sample-interval",
            "0.5",
            "--resource-max-samples",
            "1024",
            "--ssd-ceiling-gib-s",
            "12.5",
            "--f-nocache",
            "--powermetrics",
            "--no-window-telemetry",
        ]
    )

    assert args.resource_telemetry is True
    assert args.window_telemetry is False
    assert args.resource_sample_interval == 0.5
    assert args.resource_max_samples == 1024
    assert args.ssd_ceiling_gib_s == 12.5
    assert args.powermetrics is True


def test_powermetrics_requires_resource_telemetry(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4", "--powermetrics"])

    with pytest.raises(SystemExit):
        module.validate_resource_flags(parser, args)

    assert "--resource-telemetry" in capsys.readouterr().err


def test_ssd_ceiling_requires_uncached_reader_lane(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "hy3-q4",
            "--resource-telemetry",
            "--ssd-ceiling-gib-s",
            "12.5",
        ]
    )

    with pytest.raises(SystemExit):
        module.validate_resource_flags(parser, args)

    assert "--f-nocache" in capsys.readouterr().err


def test_resource_report_fields_are_absent_when_disabled() -> None:
    module = _load_module()
    row = {"completion_tokens": 4}

    module._attach_resource_report(
        row,
        None,
        ssd_ceiling_gib_s=None,
        generation_thread_cpu_ns=1,
        generation_elapsed_ns=2,
        final_completion_tokens=4,
    )

    assert "diagnostic_run" not in row
    assert "resource_telemetry" not in row


def test_telemetry_disabled_reference_generation_skips_thread_cpu_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    clock = iter((10.0, 20.0))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        module.time,
        "thread_time_ns",
        lambda: pytest.fail("disabled telemetry sampled thread CPU time"),
    )
    runtime = SimpleNamespace(admit_kv_tokens=lambda _tokens: nullcontext())

    result, started, finished, cpu_started, cpu_finished = (
        module._run_reference_generation(
            SimpleNamespace(enable_mtp=False, seed=0),
            runtime,
            prompt_ids=[1, 2],
            max_tokens=3,
            sampler=object(),
            token_callback=lambda _tokens: None,
            resource_run=None,
            generate_ar_fn=(
                lambda *_args, **_kwargs: SimpleNamespace(tokens=(1, 2, 3))
            ),
            generate_mtp1_fn=lambda *_args, **_kwargs: pytest.fail(
                "AR reference called MTP generation"
            ),
        )
    )

    assert result.tokens == (1, 2, 3)
    assert (started, finished) == (10.0, 20.0)
    assert (cpu_started, cpu_finished) == (0, 0)


def test_mtp_defaults_off_so_ar_runs_are_unchanged() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4"])
    assert args.enable_mtp is False
    assert args.mtp_artifacts is None


def test_enable_mtp_parses_with_artifacts_for_hy3() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "hy3-q4",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
        ]
    )
    module.validate_mtp_flags(parser, args)
    assert args.enable_mtp is True
    assert str(args.mtp_artifacts) == "/artifacts"


def test_enable_mtp_parses_with_bf16_artifacts_for_glm52_q4() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "glm52-q4",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
            "--mtp-precision",
            "bf16",
        ]
    )

    module.validate_mtp_flags(parser, args)

    assert args.enable_mtp is True
    assert args.mtp_precision == "bf16"
    assert str(args.mtp_artifacts) == "/artifacts"


def test_glm52_q4_rejects_q4_mtp_precision_before_load(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "glm52-q4",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
            "--mtp-precision",
            "q4",
        ]
    )

    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)

    assert "BF16" in capsys.readouterr().err


def test_enable_mtp_requires_artifacts(capsys) -> None:
    import pytest

    module = _load_module()
    parser = module.build_parser()

    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4", "--enable-mtp"])
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "--mtp-artifacts" in capsys.readouterr().err

    args = parser.parse_args(
        [*_BASE_ARGS, "--model-key", "hy3-q4", "--mtp-artifacts", "/artifacts"]
    )
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "--enable-mtp" in capsys.readouterr().err


@pytest.mark.parametrize("model_key", ["hy3-expert-only-q4", "hy3-expert-q2"])
def test_benchmark_rejects_mtp_for_explicit_hy3_expert_lanes_before_load(
    model_key: str,
    capsys,
) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            model_key,
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
        ]
    )

    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)

    assert "hy3-q4 or glm52-q4" in capsys.readouterr().err


def test_sidecar_trust_requires_validated_sidecar_and_preserves_source_hashes(
    capsys,
) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--model-key", "hy3-q4", "--trust-sidecar"])
    no_sidecar = SimpleNamespace(sidecar=None)

    with pytest.raises(SystemExit):
        module.validate_sidecar_flags(parser, args, no_sidecar)
    assert "requires a validated manifest sidecar" in capsys.readouterr().err
    assert module.should_verify_source_records(args, no_sidecar) is True

    actual_sidecar = SimpleNamespace(sidecar=SimpleNamespace(file="experts.bin"))
    module.validate_sidecar_flags(parser, args, actual_sidecar)
    assert module.should_verify_source_records(args, actual_sidecar) is False


def test_mtp_precision_defaults_bf16_and_requires_enable_mtp(capsys) -> None:
    import pytest

    module = _load_module()
    parser = module.build_parser()

    # Default resolves to bf16 (Forge contract section 6: quantized MTP heads
    # collapse acceptance).
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "hy3-q4",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
        ]
    )
    module.validate_mtp_flags(parser, args)
    assert args.mtp_precision == "bf16"

    # q4 stays selectable.
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "hy3-q4",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
            "--mtp-precision",
            "q4",
        ]
    )
    module.validate_mtp_flags(parser, args)
    assert args.mtp_precision == "q4"

    # Unknown precisions are rejected at parse time.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *_BASE_ARGS,
                "--model-key",
                "hy3-q4",
                "--enable-mtp",
                "--mtp-artifacts",
                "/artifacts",
                "--mtp-precision",
                "fp8",
            ]
        )
    capsys.readouterr()

    # The flag is meaningless without MTP.
    args = parser.parse_args(
        [*_BASE_ARGS, "--model-key", "hy3-q4", "--mtp-precision", "q4"]
    )
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "--enable-mtp" in capsys.readouterr().err


def test_mtp_rejects_concurrency(capsys) -> None:
    import pytest

    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--model-key",
            "hy3-q4",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
            "--concurrency",
            "4",
        ]
    )
    with pytest.raises(SystemExit):
        module.validate_mtp_flags(parser, args)
    assert "single-stream" in capsys.readouterr().err
