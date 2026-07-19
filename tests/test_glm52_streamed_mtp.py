"""Streamed GLM-5.2 MTP runtime dispatch and lifecycle integration."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mtplx.expert_runtime import ExpertStreamingConfig
from mtplx.glm52_mtp_patch import inject_glm52_streamed_mtp_support
from mtplx.mtp_patch import MTPContract, validate_mtp_support
from mtplx.runtime import MTPLXRuntime, _streamed_mtp_backend, load


SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
MTP_HEAD_BYTES = 19_905_841_664
MTP_Q4_HEAD_BYTES = 6_014_306_816
HY3_MTP_HEAD_BYTES = 7_505_224_960


class _FakeExpertRuntime:
    def __init__(self) -> None:
        self.closed = False

    def close(self, *, timeout: float | None = None) -> None:
        del timeout
        self.closed = True


class _PlainTarget:
    pass


def _streaming_config(
    model_key: str = "glm52-expert-q2",
    **overrides: Any,
) -> ExpertStreamingConfig:
    return ExpertStreamingConfig(
        model_key=model_key,
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        **overrides,
    )


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "glm52-q2"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "num_nextn_predict_layers": 1,
                "index_share_for_mtp_iteration": True,
            }
        ),
        encoding="utf-8",
    )
    return root


def _patch_streamed_load_without_allocation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    events: list[Any],
    reject_allocation: bool = False,
    model_key: str = "glm52-expert-q2",
    source_revision: str = SOURCE_REVISION,
    plan_fits: bool = True,
) -> _FakeExpertRuntime:
    """Replace target allocation while retaining runtime dispatch behavior."""

    from mtplx import expert_runtime as expert_runtime_module

    fake_runtime = _FakeExpertRuntime()
    spec = SimpleNamespace(source_revision=source_revision)
    plan = SimpleNamespace(
        fits_fixed=plan_fits,
        unallocated_bytes=0 if plan_fits else -123,
    )

    def memory_plan(_self, _spec, *, additional_resident_bytes=0):
        if additional_resident_bytes:
            events.append({"planned_mtp_resident_bytes": additional_resident_bytes})
        return plan

    monkeypatch.setattr(ExpertStreamingConfig, "memory_plan", memory_plan)
    monkeypatch.setattr(
        "mtplx.expert_streaming_models.get_model_spec",
        lambda requested_model_key: (
            spec
            if requested_model_key == model_key
            else pytest.fail(f"unexpected model key: {requested_model_key}")
        ),
    )

    def allocate(*_args, **_kwargs):
        events.append("allocate")
        if reject_allocation:
            raise AssertionError("target allocation happened before MTP validation")
        return object()

    monkeypatch.setattr(
        "mtplx.models.expert_mlx.make_mlx_slot_buffer_allocator",
        allocate,
    )
    monkeypatch.setattr(
        "mtplx.models.expert_mlx.make_mlx_component_bank_allocator",
        allocate,
    )

    def open_runtime(*_args, **kwargs):
        additional_resident_bytes = kwargs.get("additional_resident_bytes", 0)
        if additional_resident_bytes:
            events.append({"opened_mtp_resident_bytes": additional_resident_bytes})
        events.append("open")
        if reject_allocation:
            raise AssertionError("target runtime opened before MTP validation")
        return fake_runtime

    monkeypatch.setattr(
        expert_runtime_module.ExpertStreamingRuntime,
        "open",
        staticmethod(open_runtime),
    )
    monkeypatch.setattr(
        "mtplx.resident_loader.construct_resident_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=model,
            report=SimpleNamespace(as_dict=lambda: {"fake": True}),
        ),
    )
    monkeypatch.setattr(
        "mtplx.runtime._load_tokenizer_resilient",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "mtplx.attention_split.configure_split_full_attention",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mtplx.native_mlp.configure_native_mlp",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("mtplx.nax_verify.nax_env_enabled", lambda: False)
    monkeypatch.setattr(
        "mtplx.kernel_selfcheck.maybe_run_model_selfcheck",
        lambda *_args, **_kwargs: None,
    )
    return fake_runtime


def _accept_glm_artifact(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: Path,
    events: list[Any] | None = None,
):
    receipt = {
        "source": {"revision": SOURCE_REVISION},
        "inventory": {"payload_bytes": MTP_HEAD_BYTES},
    }
    verified = SimpleNamespace(
        root=artifacts.resolve(),
        manifest=receipt,
        file=io.BytesIO(b"verified-artifact"),
    )

    @contextlib.contextmanager
    def open_verified(root: Path, *, deep: bool = True):
        assert Path(root).resolve() == artifacts.resolve()
        assert deep is True
        if events is not None:
            events.append("artifact-enter")
        try:
            yield verified
        finally:
            if events is not None:
                events.append("artifact-exit")

    monkeypatch.setattr(
        "mtplx.glm52_mtp_artifact.open_verified_glm52_mtp_layer78",
        open_verified,
    )
    return verified


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_streamed_dispatches_to_glm_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    root = _model_root(tmp_path)
    artifacts = tmp_path / "mtp"
    artifacts.mkdir()
    events: list[Any] = []
    model = _PlainTarget()
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=model,
        events=events,
        model_key=model_key,
    )
    verified = _accept_glm_artifact(monkeypatch, artifacts, events)
    glm_args = SimpleNamespace()
    prebuilt_mtp = SimpleNamespace(layers=[object()])
    monkeypatch.setattr(
        "mtplx.models.glm52_mlx.ModelArgs.from_dict",
        lambda config: glm_args,
    )
    monkeypatch.setattr(
        "mtplx.expert_runtime.apply_mlx_memory_cap",
        lambda plan, **_kwargs: events.append(("memory-cap", plan)),
    )

    def build_mtp(
        artifact_dir,
        args,
        *,
        expected_revision,
        precision,
        verified_artifact,
    ):
        events.append(
            (
                "build-mtp",
                Path(artifact_dir),
                args,
                expected_revision,
                precision,
                verified_artifact,
            )
        )
        return prebuilt_mtp

    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.build_glm52_mtp_module",
        build_mtp,
    )

    def inject_glm(
        target,
        artifact_dir,
        config,
        contract,
        *,
        expected_revision,
        verified_artifact,
        mtp_module,
    ):
        events.append(
            (
                "glm-inject",
                target,
                Path(artifact_dir),
                config["model_type"],
                contract,
                expected_revision,
                verified_artifact,
                mtp_module,
            )
        )
        return True

    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.inject_glm52_streamed_mtp_support",
        inject_glm,
    )
    monkeypatch.setattr(
        "mtplx.hy3_mtp_patch.inject_hy3_streamed_mtp_support",
        lambda *_args, **_kwargs: pytest.fail("GLM load dispatched to Hy3"),
    )
    monkeypatch.setattr("mtplx.runtime.validate_mtp_support", lambda _model: True)

    runtime = load(
        root,
        mtp=True,
        expert_streaming_config=_streaming_config(model_key),
        expert_manifest=root / "expert-manifest.json",
        mtp_artifacts=artifacts,
        mtp_precision="bf16",
    )

    assert runtime.model is model
    assert runtime.mtp_enabled is True
    assert {tuple(event.items())[0] for event in events if isinstance(event, dict)} == {
        ("planned_mtp_resident_bytes", MTP_HEAD_BYTES),
        ("opened_mtp_resident_bytes", MTP_HEAD_BYTES),
    }
    tuple_events = [event for event in events if isinstance(event, tuple)]
    assert [event[0] for event in tuple_events] == [
        "memory-cap",
        "build-mtp",
        "glm-inject",
    ]
    injection = tuple_events[-1]
    assert injection[1:] == (
        model,
        artifacts,
        "glm_moe_dsa",
        runtime.contract,
        SOURCE_REVISION,
        verified,
        prebuilt_mtp,
    )
    assert events.index("artifact-enter") < events.index("allocate")
    assert events.index(tuple_events[0]) < events.index(tuple_events[1])
    assert events.index(tuple_events[1]) < events.index("allocate")
    assert events.index("open") < events.index("artifact-exit")
    assert events.index(injection) < events.index("artifact-exit")


def _accept_glm_q4_artifact(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: Path,
    events: list[Any] | None = None,
):
    receipt = {
        "schema": "mtplx-glm52-mtp-layer78-q4-v1",
        "source": {"revision": SOURCE_REVISION},
        "inventory": {"payload_bytes": MTP_Q4_HEAD_BYTES},
    }
    verified = SimpleNamespace(
        root=artifacts.resolve(),
        manifest=receipt,
        file=io.BytesIO(b"verified-q4-artifact"),
    )

    @contextlib.contextmanager
    def open_verified_q4(root: Path, *, deep: bool = True):
        assert Path(root).resolve() == artifacts.resolve()
        assert deep is True
        if events is not None:
            events.append("artifact-enter")
        try:
            yield verified
        finally:
            if events is not None:
                events.append("artifact-exit")

    monkeypatch.setattr(
        "mtplx.glm52_mtp_artifact.open_verified_glm52_mtp_layer78_q4",
        open_verified_q4,
    )
    monkeypatch.setattr(
        "mtplx.glm52_mtp_artifact.open_verified_glm52_mtp_layer78",
        lambda *_args, **_kwargs: pytest.fail("q4 load opened the BF16 artifact"),
    )
    return verified


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_streamed_q4_prices_and_builds_from_the_q4_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    """precision=q4 opens the Q4 sibling and prices its (smaller) payload."""

    root = _model_root(tmp_path)
    artifacts = tmp_path / "mtp"
    artifacts.mkdir()
    events: list[Any] = []
    model = _PlainTarget()
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=model,
        events=events,
        model_key=model_key,
    )
    verified = _accept_glm_q4_artifact(monkeypatch, artifacts, events)
    glm_args = SimpleNamespace()
    prebuilt_mtp = SimpleNamespace(layers=[object()])
    monkeypatch.setattr(
        "mtplx.models.glm52_mlx.ModelArgs.from_dict",
        lambda config: glm_args,
    )
    monkeypatch.setattr(
        "mtplx.expert_runtime.apply_mlx_memory_cap",
        lambda plan, **_kwargs: events.append(("memory-cap", plan)),
    )
    observed: dict[str, Any] = {}

    def build_mtp(
        artifact_dir,
        args,
        *,
        expected_revision,
        precision,
        verified_artifact,
    ):
        observed["precision"] = precision
        observed["verified_artifact"] = verified_artifact
        return prebuilt_mtp

    monkeypatch.setattr("mtplx.glm52_mtp_patch.build_glm52_mtp_module", build_mtp)
    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.inject_glm52_streamed_mtp_support",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr("mtplx.runtime.validate_mtp_support", lambda _model: True)

    runtime = load(
        root,
        mtp=True,
        expert_streaming_config=_streaming_config(model_key),
        expert_manifest=root / "expert-manifest.json",
        mtp_artifacts=artifacts,
        mtp_precision="q4",
    )

    assert runtime.mtp_enabled is True
    assert observed["precision"] == "q4"
    assert observed["verified_artifact"] is verified
    # The plan is priced from the Q4 artifact size (~5.6 GiB), not the 18.5
    # GiB BF16 head: pricing follows the actual selected precision.
    assert MTP_Q4_HEAD_BYTES < MTP_HEAD_BYTES
    priced = {tuple(event.items())[0] for event in events if isinstance(event, dict)}
    assert priced == {
        ("planned_mtp_resident_bytes", MTP_Q4_HEAD_BYTES),
        ("opened_mtp_resident_bytes", MTP_Q4_HEAD_BYTES),
    }


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_incompatible_contract_fails_before_artifact_or_target_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    root = _model_root(tmp_path)
    artifacts = tmp_path / "mtp"
    artifacts.mkdir()
    events: list[Any] = []
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=_PlainTarget(),
        events=events,
        reject_allocation=True,
        model_key=model_key,
    )
    _accept_glm_artifact(monkeypatch, artifacts, events)

    with pytest.raises(RuntimeError, match="contract|position|incompatible"):
        load(
            root,
            mtp=True,
            contract=MTPContract(mtp_position_mode="absolute"),
            expert_streaming_config=_streaming_config(model_key),
            expert_manifest=root / "expert-manifest.json",
            mtp_artifacts=artifacts,
            mtp_precision="bf16",
        )

    assert events == []


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_nonfitting_memory_plan_rejects_before_head_or_target_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    root = _model_root(tmp_path)
    artifacts = tmp_path / "mtp"
    artifacts.mkdir()
    events: list[Any] = []
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=_PlainTarget(),
        events=events,
        reject_allocation=True,
        model_key=model_key,
        plan_fits=False,
    )
    _accept_glm_artifact(monkeypatch, artifacts, events)
    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.build_glm52_mtp_module",
        lambda *_args, **_kwargs: pytest.fail("built head for non-fitting plan"),
    )

    with pytest.raises(ValueError, match="fixed.*footprint|exceeds limit"):
        load(
            root,
            mtp=True,
            expert_streaming_config=_streaming_config(model_key),
            expert_manifest=root / "expert-manifest.json",
            mtp_artifacts=artifacts,
            mtp_precision="bf16",
        )

    assert events == [
        "artifact-enter",
        {"planned_mtp_resident_bytes": MTP_HEAD_BYTES},
        "artifact-exit",
    ]


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_streamed_runtime_closes_when_post_injection_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    root = _model_root(tmp_path)
    artifacts = tmp_path / "mtp"
    artifacts.mkdir()
    events: list[Any] = []
    fake_runtime = _patch_streamed_load_without_allocation(
        monkeypatch,
        model=_PlainTarget(),
        events=events,
        model_key=model_key,
    )
    _accept_glm_artifact(monkeypatch, artifacts, events)
    monkeypatch.setattr(
        "mtplx.models.glm52_mlx.ModelArgs.from_dict",
        lambda _config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mtplx.expert_runtime.apply_mlx_memory_cap",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.build_glm52_mtp_module",
        lambda *_args, **_kwargs: SimpleNamespace(layers=[object()]),
    )
    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.inject_glm52_streamed_mtp_support",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr("mtplx.runtime.validate_mtp_support", lambda _model: True)
    monkeypatch.setattr(
        "mtplx.attention_split.configure_split_full_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic post-injection failure")
        ),
    )

    with pytest.raises(RuntimeError, match="post-injection failure"):
        load(
            root,
            mtp=True,
            expert_streaming_config=_streaming_config(model_key),
            expert_manifest=root / "expert-manifest.json",
            mtp_artifacts=artifacts,
            mtp_precision="bf16",
        )

    assert fake_runtime.closed is True
    assert events[-1] == "artifact-exit"


def test_hy3_expert_q2_charges_bf16_mtp_before_target_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "hy3-q2"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hy_v3",
                "num_hidden_layers": 80,
                "first_k_dense_replace": 1,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "mtp"
    artifacts.mkdir()
    events: list[Any] = []
    model = _PlainTarget()
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=model,
        events=events,
        model_key="hy3-expert-q2",
        source_revision="hy3-source-revision",
    )
    hy3_args = SimpleNamespace()
    prebuilt_mtp = SimpleNamespace(layers=[object()])
    monkeypatch.setattr(
        "mtplx.models.hy3_mlx.ModelArgs.from_dict",
        lambda config: hy3_args,
    )
    monkeypatch.setattr(
        "mtplx.expert_runtime.apply_mlx_memory_cap",
        lambda plan, **_kwargs: events.append(("memory-cap", plan)),
    )
    verified = SimpleNamespace(
        root=artifacts.resolve(),
        precision="bf16",
        payload_bytes=HY3_MTP_HEAD_BYTES,
        source_revision="hy3-source-revision",
        files={},
    )

    @contextlib.contextmanager
    def open_verified(root, *, precision, expected_revision):
        assert Path(root).resolve() == artifacts.resolve()
        assert precision == "bf16"
        assert expected_revision == "hy3-source-revision"
        events.append("artifact-enter")
        try:
            yield verified
        finally:
            events.append("artifact-exit")

    monkeypatch.setattr(
        "mtplx.hy3_mtp_patch.open_verified_hy3_mtp_artifacts",
        open_verified,
    )

    def build_hy3(
        artifact_dir,
        args,
        *,
        expected_revision,
        precision,
        shared_kernel,
        shared_kernel_depth,
        verified_artifacts,
    ):
        events.append(
            (
                "build-mtp",
                Path(artifact_dir),
                args,
                expected_revision,
                precision,
                shared_kernel,
                shared_kernel_depth,
                verified_artifacts,
            )
        )
        return prebuilt_mtp

    monkeypatch.setattr("mtplx.hy3_mtp_patch.build_hy3_mtp_module", build_hy3)

    def inject_hy3(
        target,
        artifact_dir,
        config,
        contract,
        *,
        expected_revision,
        mtp_precision,
        shared_kernel,
        shared_kernel_depth,
        mtp_module,
    ):
        events.append(
            (
                "hy3-inject",
                target,
                Path(artifact_dir),
                config["model_type"],
                contract,
                expected_revision,
                mtp_precision,
                shared_kernel,
                shared_kernel_depth,
                mtp_module,
            )
        )
        return True

    monkeypatch.setattr(
        "mtplx.hy3_mtp_patch.inject_hy3_streamed_mtp_support",
        inject_hy3,
    )
    monkeypatch.setattr("mtplx.runtime.validate_mtp_support", lambda _model: True)
    router_bytes = 80 * 192 * 4096 * 2

    def configure_router(target, selector, *, sigmoid_mode):
        events.append(("configure-router", target, selector, sigmoid_mode))
        return {
            "selector": selector,
            "router_count": 80,
            "enabled_count": 80,
            "incremental_bytes": router_bytes,
        }

    monkeypatch.setattr(
        "mtplx.models.hy3_mlx.configure_hy3_router_kernels",
        configure_router,
    )

    runtime = load(
        root,
        mtp=True,
        expert_streaming_config=_streaming_config(
            "hy3-expert-q2",
            hy3_router_kernel="mpp-r1-fused-r2",
        ),
        expert_manifest=root / "expert-manifest.json",
        mtp_artifacts=artifacts,
        mtp_precision="bf16",
    )

    assert runtime.model is model
    assert runtime.mtp_enabled is True
    assert {tuple(event.items())[0] for event in events if isinstance(event, dict)} == {
        ("planned_mtp_resident_bytes", HY3_MTP_HEAD_BYTES + router_bytes),
        ("opened_mtp_resident_bytes", HY3_MTP_HEAD_BYTES + router_bytes),
    }
    tuple_events = [event for event in events if isinstance(event, tuple)]
    assert [event[0] for event in tuple_events] == [
        "memory-cap",
        "build-mtp",
        "hy3-inject",
        "configure-router",
    ]
    injection = tuple_events[2]
    assert injection[0] == "hy3-inject"
    assert injection[1:] == (
        model,
        artifacts,
        "hy_v3",
        runtime.contract,
        "hy3-source-revision",
        "bf16",
        "stock",
        3,
        prebuilt_mtp,
    )
    assert tuple_events[1][-1] is verified
    assert tuple_events[-1] == (
        "configure-router",
        model,
        "mpp-r1-fused-r2",
        "precise",
    )
    assert events.index("artifact-enter") < events.index(tuple_events[1])
    assert events.index(tuple_events[1]) < events.index("allocate")
    assert events[-1] == "artifact-exit"


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_streamed_dispatches_bf16_and_q4_to_glm_backend(
    model_key: str,
) -> None:
    # Both GLM lanes resolve to the glm52 backend for either head precision
    # (issue #100): the Q4 sibling head is selectable, BF16 remains default.
    assert _streamed_mtp_backend(model_key, "bf16") == "glm52"
    assert _streamed_mtp_backend(model_key, "q4") == "glm52"

    with pytest.raises(RuntimeError, match="not supported"):
        _streamed_mtp_backend(model_key, "fp8")


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_streamed_missing_artifact_fails_before_target_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    root = _model_root(tmp_path)
    events: list[Any] = []
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=_PlainTarget(),
        events=events,
        reject_allocation=True,
        model_key=model_key,
    )

    with pytest.raises(RuntimeError, match="mtp_artifacts"):
        load(
            root,
            mtp=True,
            expert_streaming_config=_streaming_config(model_key),
            expert_manifest=root / "expert-manifest.json",
        )

    assert events == []


@pytest.mark.parametrize("model_key", ["glm52-expert-q2", "glm52-q4"])
def test_glm52_streamed_invalid_provenance_fails_before_target_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
) -> None:
    root = _model_root(tmp_path)
    artifacts = tmp_path / "invalid-mtp"
    artifacts.mkdir()
    (artifacts / "mtp-artifact-manifest.json").write_text("{}", encoding="utf-8")
    events: list[Any] = []
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=_PlainTarget(),
        events=events,
        reject_allocation=True,
        model_key=model_key,
    )

    with pytest.raises(RuntimeError, match="artifact|manifest|provenance"):
        load(
            root,
            mtp=True,
            expert_streaming_config=_streaming_config(model_key),
            expert_manifest=root / "expert-manifest.json",
            mtp_artifacts=artifacts,
            mtp_precision="bf16",
        )

    assert events == []


class _TargetBody:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, inputs, cache=None):
        self.calls.append((inputs, cache))
        return ("target-hidden", inputs)

    def embed_tokens(self, token_ids):
        return ("embedding", token_ids)


class _TargetHead:
    def __call__(self, hidden):
        return ("target-logits", hidden)


class _DraftLayer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("AR forward invoked the GLM MTP layer")


class _InjectableGLMTarget:
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            num_nextn_predict_layers=1,
            index_share_for_mtp_iteration=True,
        )
        self.model = _TargetBody()
        self.lm_head = _TargetHead()
        self.mtp = None


def _runtime(model: Any, *, mtp_enabled: bool) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=object(),
        model_path=Path("synthetic-glm52"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
    )


def test_glm52_injected_ar_preservation_does_not_call_mtp_and_sets_verify_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _InjectableGLMTarget()
    target_class = type(target)
    draft_layer = _DraftLayer()
    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.build_glm52_mtp_module",
        lambda *_args, **_kwargs: SimpleNamespace(layers=[draft_layer]),
    )

    assert inject_glm52_streamed_mtp_support(
        target,
        Path("synthetic-mtp"),
        {
            "model_type": "glm_moe_dsa",
            "num_nextn_predict_layers": 1,
            "index_share_for_mtp_iteration": True,
        },
        MTPContract(),
        expected_revision=SOURCE_REVISION,
    )
    assert isinstance(target, target_class)
    assert validate_mtp_support(target)
    assert target.mtp_verify_width == 6

    cache = object()
    output = _runtime(target, mtp_enabled=True).forward_ar([1, 2, 3], cache=cache)

    assert output == ("target-logits", ("target-hidden", [1, 2, 3]))
    assert target.model.calls == [([1, 2, 3], cache)]
    assert draft_layer.calls == 0


def test_glm52_injected_prefill_honors_logit_suppression_and_tail_keep() -> None:
    class Body:
        def __call__(self, _inputs, _cache=None):
            return mx.arange(6, dtype=mx.float32).reshape(1, 3, 2)

        def embed_tokens(self, token_ids):
            return token_ids

    class Head:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def __call__(self, hidden):
            self.shapes.append(tuple(int(dim) for dim in hidden.shape))
            return hidden

    target = _InjectableGLMTarget()
    target.model = Body()
    target.lm_head = Head()
    assert inject_glm52_streamed_mtp_support(
        target,
        Path("synthetic-mtp"),
        {"model_type": "glm_moe_dsa"},
        MTPContract(),
        expected_revision=SOURCE_REVISION,
        mtp_module=SimpleNamespace(layers=[object()]),
    )

    logits, hidden = target(
        [1, 2, 3],
        return_hidden=True,
        emit_logits=False,
    )
    assert logits is None
    assert hidden.shape == (1, 3, 2)
    assert target.lm_head.shapes == []

    logits, hidden = target(
        [1, 2, 3],
        return_hidden=True,
        emit_logits=True,
        logits_keep=1,
    )
    assert logits.shape == (1, 1, 2)
    assert hidden.shape == (1, 3, 2)
    assert target.lm_head.shapes == [(1, 1, 2)]


def test_glm52_expert_q2_no_mtp_load_preserves_plain_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _model_root(tmp_path)
    events: list[Any] = []
    model = _PlainTarget()
    original_class = type(model)
    _patch_streamed_load_without_allocation(
        monkeypatch,
        model=model,
        events=events,
    )
    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.inject_glm52_streamed_mtp_support",
        lambda *_args, **_kwargs: pytest.fail("mtp=False invoked the GLM injector"),
    )

    runtime = load(
        root,
        mtp=False,
        expert_streaming_config=_streaming_config(),
        expert_manifest=root / "expert-manifest.json",
    )

    assert runtime.model is model
    assert type(runtime.model) is original_class
    assert runtime.mtp_enabled is False
    assert not hasattr(runtime.model, "mtp_forward")
    assert events == ["allocate", "open"]


def test_runtime_finish_mtp_cycle_delegates_when_supported() -> None:
    cache = object()
    seen: list[Any] = []
    model = SimpleNamespace(finish_mtp_cycle=seen.append)

    _runtime(model, mtp_enabled=True).finish_mtp_cycle(cache)

    assert seen == [cache]


def test_runtime_finish_mtp_cycle_is_compatible_with_other_backends() -> None:
    _runtime(SimpleNamespace(), mtp_enabled=True).finish_mtp_cycle(object())
