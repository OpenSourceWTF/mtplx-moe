from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.models import expert_mlx
from mtplx.models.expert_mlx import HotExpertSwitchGLU


def _runtime(model_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(
            key=model_key,
            quant_group_size=64,
            quant_bits=2,
            expert_codec="t158",
        ),
        shadow_bank_for_layer=lambda _layer: None,
    )


def test_glm52_q1t_uses_measured_projection_threadgroups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, bool]] = []

    def fake_shadow_gather(
        values,
        _slot_rows,
        packed,
        _scales,
        *,
        codec,
        threads_per_tg,
        stage,
    ):
        assert codec == "t158"
        calls.append((packed, threads_per_tg, stage))
        rows = int(values.shape[0])
        width = 6144 if packed == "down" else 2048
        return mx.zeros((rows, width), dtype=values.dtype)

    monkeypatch.setattr(
        "mtplx.kernels.shadow_gather.shadow_gather_mm",
        fake_shadow_gather,
    )
    bank = SimpleNamespace(
        arrays={
            "gate_proj.packed": "gate",
            "gate_proj.scales": object(),
            "up_proj.packed": "up",
            "up_proj.scales": object(),
            "down_proj.packed": "down",
            "down_proj.scales": object(),
        }
    )

    output = expert_mlx._glm52_q1t_t158_component_bank(
        mx.zeros((3, 6144), dtype=mx.bfloat16),
        bank,
        mx.array([0, 1, 2], dtype=mx.int32),
    )

    assert output.shape == (3, 6144)
    assert calls == [
        ("gate", 32, False),
        ("up", 32, False),
        ("down", 64, False),
    ]


@pytest.mark.parametrize(
    ("model_key", "expected"),
    (
        ("glm52-expert-q1t", "glm"),
        ("tiny-hy3-q1t158", "stock"),
    ),
)
def test_t158_component_bank_route_is_fixed_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
    expected: str,
) -> None:
    calls: list[str] = []

    def glm_route(selected, bindings):
        calls.append("glm")
        return selected

    def stock_route(selected, bindings, *, codec):
        assert codec == "t158"
        calls.append("stock")
        return selected

    monkeypatch.setattr(
        expert_mlx,
        "_run_glm52_q1t_t158_component_bank",
        glm_route,
    )
    monkeypatch.setattr(expert_mlx, "_run_component_bank_shadow", stock_route)
    switch = HotExpertSwitchGLU(_runtime(model_key), 1)
    selected = mx.zeros((1, 6144), dtype=mx.bfloat16)

    output = switch._dispatch_component_bank(selected, (object(),))

    assert output is selected
    assert calls == [expected]
