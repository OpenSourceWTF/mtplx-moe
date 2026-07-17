"""Speculative expert-prefetch wiring: config knob, runtime API, lookahead."""

from __future__ import annotations

import pytest

from mtplx.expert_runtime import ExpertStreamingConfig
from mtplx.expert_streaming_models import get_model_spec


def _config_kwargs(**overrides):
    kwargs = {
        "model_key": "hy3-expert-q2",
        "memory_limit_bytes": 96 * 1024**3,
        "max_live_kv_tokens": 4096,
    }
    kwargs.update(overrides)
    return kwargs


def test_prefetch_slots_default_zero_and_validation() -> None:
    config = ExpertStreamingConfig(**_config_kwargs())
    assert config.prefetch_slots == 0

    config = ExpertStreamingConfig(**_config_kwargs(prefetch_slots=8))
    assert config.prefetch_slots == 8

    with pytest.raises(ValueError, match="prefetch_slots"):
        ExpertStreamingConfig(**_config_kwargs(prefetch_slots=-1))
    with pytest.raises(TypeError, match="prefetch_slots"):
        ExpertStreamingConfig(**_config_kwargs(prefetch_slots=True))
    with pytest.raises(TypeError, match="prefetch_slots"):
        ExpertStreamingConfig(**_config_kwargs(prefetch_slots="8"))


def test_prefetch_slots_require_layer_cache_scope() -> None:
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        ExpertStreamingConfig(
            **_config_kwargs(prefetch_slots=8, cache_scope="global")
        )
    # Zero slots never constrain the scope.
    config = ExpertStreamingConfig(
        **_config_kwargs(prefetch_slots=0, cache_scope="global")
    )
    assert config.prefetch_slots == 0


def test_memory_plan_carries_prefetch_slots_per_layer() -> None:
    spec = get_model_spec("hy3-expert-q2")
    config = ExpertStreamingConfig(**_config_kwargs(prefetch_slots=8))
    plan = config.memory_plan(spec)
    assert plan.prefetch_slots_per_layer == 8
    assert plan.prefetch_bytes == (
        spec.routed_layer_count * 8 * spec.expert_record_bytes
    )

    baseline = ExpertStreamingConfig(**_config_kwargs()).memory_plan(spec)
    assert baseline.prefetch_slots_per_layer == 0
    assert baseline.prefetch_bytes == 0
