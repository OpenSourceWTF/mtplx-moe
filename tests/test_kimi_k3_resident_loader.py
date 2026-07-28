from __future__ import annotations

from mtplx.models.kimi_k3_mlx import Model as KimiK3Model
from mtplx.models.kimi_k3_mlx import ModelArgs as KimiK3ModelArgs
from mtplx.resident_loader import get_streaming_model_classes


def test_kimi_linear_uses_the_dedicated_k3_streaming_overlay() -> None:
    model_class, args_class = get_streaming_model_classes(
        {
            "model_type": "kimi_linear",
            "hidden_act": "situ",
            "num_hidden_layers": 93,
            "num_experts": 896,
        }
    )

    assert model_class is KimiK3Model
    assert args_class is KimiK3ModelArgs
