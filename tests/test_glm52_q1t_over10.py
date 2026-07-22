from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.expert_runtime import ExpertStreamingConfig
from mtplx.glm52_q1t_over10 import validate_glm52_q1t_fused_rans_config


# The fused-rANS route was closed 2026-07-20 (kernel loses to the stock control,
# roofline below goal). These tests construct a live fused-rANS ExpertStreamingConfig,
# which the stock config gate no longer accepts.
pytestmark = pytest.mark.skip(reason="fused-rANS route closed 2026-07-20")


GIB = 1024**3


def _valid_config() -> ExpertStreamingConfig:
    return ExpertStreamingConfig(
        model_key="glm52-expert-q1t",
        memory_limit_bytes=96 * GIB,
        max_live_kv_tokens=4096,
        runtime_reserve_bytes=12 * GIB,
        expert_cache_limit_bytes=72 * GIB,
        transient_slots=48,
        slot_layout="fused-rans",
        banked_manifest="/artifact/expert-manifest-glm52-q1t-fused-rans.json",
        banked_codec="rans32x-uniform-packed-v1",
        streamed_codec="none",
        streamed_codec_manifest=None,
        streamed_codec_verify=False,
        verify_record_hashes=False,
        verify_sidecar_hash_at_open=False,
        cache_policy="frequency",
        cache_scope="layer",
        max_read_chunk_bytes=8 * 1024**2,
        deferred_pin_release=True,
        split_route_release="deferred",
        prefetch_slots=0,
        route_census=False,
        resource_telemetry=False,
        trace_routes=False,
        q2_expert_kernel="stock",
    )


def test_fused_rans_contract_requires_the_exact_cache72_control_envelope() -> None:
    config = _valid_config()

    assert validate_glm52_q1t_fused_rans_config(config) is None
    assert config.memory_limit_bytes == 96 * GIB
    assert config.runtime_reserve_bytes == 12 * GIB
    assert config.expert_cache_limit_bytes == 72 * GIB
    assert config.max_live_kv_tokens == 4096


def test_fused_rans_contract_rejects_total_memory_above_96_gib() -> None:
    candidate = SimpleNamespace(**_valid_config().to_dict())
    candidate.memory_limit_bytes = 96 * GIB + 1

    with pytest.raises(ValueError, match="memory_limit_bytes"):
        validate_glm52_q1t_fused_rans_config(candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_key", "hy3-expert-q2"),
        ("model_key", "glm52-expert-q2"),
        ("model_key", "glm52-q4"),
        ("slot_layout", "component-banks"),
        ("expert_cache_limit_bytes", 64 * GIB),
        ("transient_slots", 32),
        ("banked_manifest", None),
        ("banked_codec", "none"),
        ("streamed_codec", "rans32x-v1"),
        ("streamed_codec_manifest", "/artifact/old-record-rans.json"),
        ("streamed_codec_verify", True),
        ("verify_record_hashes", True),
        ("verify_sidecar_hash_at_open", True),
        ("island_layers", (3,)),
        ("mmap_island_layers", (3,)),
        ("miss_shadow", "t158"),
        ("prefetch_slots", 1),
        ("route_census", True),
        ("resource_telemetry", True),
        ("trace_routes", True),
        ("q2_expert_kernel", "nax"),
    ),
)
def test_fused_rans_contract_rejects_cross_lane_state(
    field: str, value: object
) -> None:
    candidate = SimpleNamespace(**_valid_config().to_dict())
    setattr(candidate, field, value)

    with pytest.raises(ValueError, match=field):
        validate_glm52_q1t_fused_rans_config(candidate)
