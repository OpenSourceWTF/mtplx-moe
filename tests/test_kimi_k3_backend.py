from types import SimpleNamespace

from mtplx.backends.descriptors import (
    descriptor_for_backend_id,
    descriptor_from_runtime,
)


def test_kimi_k3_backend_is_registered_as_target_only_ar() -> None:
    descriptor = descriptor_for_backend_id("kimi_k3_ar")

    assert descriptor.backend_id == "kimi_k3_ar"
    assert descriptor.architecture_id == "kimi-k3-streaming-ar"
    assert descriptor.model_family == "kimi_k3"
    assert descriptor.runtime_capabilities == (
        "target_logits",
        "target_only_ar",
        "expert_streaming",
    )
    assert descriptor.uses_external_assistant is False
    assert descriptor.uses_draft_lm_head is False
    assert descriptor.mtp_history_policy == "none"
    assert descriptor.reasoning_codec.supported is False


def test_kimi_k3_backend_survives_runtime_descriptor_resolution() -> None:
    runtime = SimpleNamespace(
        backend_id=None,
        gemma4_external_assistant=False,
    )
    args = SimpleNamespace(backend_id="kimi_k3_ar")

    assert descriptor_from_runtime(runtime, args).backend_id == "kimi_k3_ar"
