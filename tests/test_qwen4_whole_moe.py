from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.kernels import qwen4_whole_moe as kernels
import mtplx.qwen4_whole_moe as whole_moe


class _Value:
    def __init__(self, rows: int):
        self.shape = (1, rows, 2560)


class _Block:
    top_k = 10

    def __call__(self, value):
        return ("stock", value.shape[1])


def test_exact_sources_encode_qwen4_storage_and_right_shapes():
    sources = kernels.sources()
    assert "constexpr uint EXPERTS = 512" in sources["stage1"]
    assert "constexpr uint HIDDEN = 2560" in sources["stage2"]
    assert "constexpr uint TOP_K = 10" in sources["stage2"]
    assert "constexpr uint Q4_GROUP = 32" in sources["stage2"]
    assert "constexpr uint Q8_GROUP = 128" in sources["stage3"]
    assert kernels.launch_geometry() == {
        "stage1": ((128 * 32, 1, 1), (32, 1, 1)),
        "stage2": ((440 * 128, 1, 1), (128, 1, 1)),
        "stage3": ((320 * 128, 1, 1), (128, 1, 1)),
    }
    assert "uint row = group / OUTPUT_TILES" in sources["stage3"]


def test_row_owned_top10_matches_qwen4_argpartition_order():
    logits = mx.sin(mx.arange(2 * 512, dtype=mx.float32) * 1.337).reshape(2, 512)
    expected_ids = mx.argpartition(-logits, 9, axis=-1)[..., :10]
    expected_logits = mx.take_along_axis(logits, expected_ids, axis=-1)

    actual_ids, actual_logits = kernels.route_top10(logits)
    mx.eval(expected_ids, expected_logits, actual_ids, actual_logits)

    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(
        np.asarray(actual_logits), np.asarray(expected_logits)
    )


def test_exact_m2_route_is_installed_once_and_other_rows_stay_stock(monkeypatch):
    monkeypatch.setenv(whole_moe.WHOLE_MOE_ENV, "1")
    layers = [SimpleNamespace(mlp=_Block()) for _ in range(48)]
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    monkeypatch.setattr(
        whole_moe,
        "_bind",
        lambda block: SimpleNamespace(block=block),
    )
    monkeypatch.setattr(
        whole_moe,
        "_m2_call",
        lambda block, binding, value: ("whole", value.shape[1]),
    )

    report = whole_moe.configure_qwen4_whole_moe(
        model,
        config={
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
            },
        },
        validate_storage=False,
        run_selfcheck=False,
    )

    assert report["installed_blocks"] == 48
    assert model.language_model.model.layers[0].mlp(_Value(2)) == ("whole", 2)
    assert model.language_model.model.layers[0].mlp(_Value(1)) == ("stock", 1)
    assert model.language_model.model.layers[0].mlp(_Value(3)) == ("stock", 3)
