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


QWEN4_CONFIG = {
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
}


def _fake_model():
    layers = [SimpleNamespace(mlp=_Block()) for _ in range(48)]
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )


def test_exact_sources_encode_qwen4_storage_and_right_shapes():
    for rows in (2, 3):
        sources = kernels.sources(rows)
        assert f"constexpr uint ROWS = {rows}" in sources["stage1"]
        assert f"constexpr uint ROWS = {rows}" in sources["stage3"]
        assert "constexpr uint EXPERTS = 512" in sources["stage1"]
        assert "constexpr uint HIDDEN = 2560" in sources["stage2"]
        assert "constexpr uint TOP_K = 10" in sources["stage2"]
        assert "constexpr uint Q4_GROUP = 32" in sources["stage2"]
        assert "constexpr uint Q8_GROUP = 128" in sources["stage3"]
        assert kernels.launch_geometry(rows) == {
            "stage1": ((128 * 32, 1, 1), (32, 1, 1)),
            "stage2": ((440 * 128, 1, 1), (128, 1, 1)),
            "stage3": ((rows * 160 * 128, 1, 1), (128, 1, 1)),
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


def test_row_owned_top10_matches_qwen4_argpartition_order_for_three_rows():
    logits = mx.sin(mx.arange(3 * 512, dtype=mx.float32) * 1.337).reshape(3, 512)
    expected_ids = mx.argpartition(-logits, 9, axis=-1)[..., :10]
    expected_logits = mx.take_along_axis(logits, expected_ids, axis=-1)

    actual_ids, actual_logits = kernels.route_top10(logits, rows=3)
    mx.eval(expected_ids, expected_logits, actual_ids, actual_logits)

    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(
        np.asarray(actual_logits), np.asarray(expected_logits)
    )


def test_exact_m2_m3_routes_are_installed_once_and_other_rows_stay_stock(
    monkeypatch,
):
    monkeypatch.setenv(whole_moe.WHOLE_MOE_ENV, "1")
    model = _fake_model()
    monkeypatch.setattr(
        whole_moe,
        "_bind",
        lambda block, rows: SimpleNamespace(block=block, rows=rows),
    )
    monkeypatch.setattr(
        whole_moe,
        "_whole_call",
        lambda block, binding, value: ("whole", binding.rows, value.shape[1]),
    )

    report = whole_moe.configure_qwen4_whole_moe(
        model,
        config=QWEN4_CONFIG,
        validate_storage=False,
        run_selfcheck=False,
    )

    assert report["installed_blocks"] == 48
    assert report["geometry"]["rows"] == (2, 3)
    assert model.language_model.model.layers[0].mlp(_Value(2)) == (
        "whole",
        2,
        2,
    )
    assert model.language_model.model.layers[0].mlp(_Value(3)) == (
        "whole",
        3,
        3,
    )
    assert model.language_model.model.layers[0].mlp(_Value(1)) == ("stock", 1)
    assert model.language_model.model.layers[0].mlp(_Value(4)) == ("stock", 4)


def test_construction_selfchecks_both_rows_before_install(monkeypatch):
    monkeypatch.setenv(whole_moe.WHOLE_MOE_ENV, "1")
    model = _fake_model()
    seen = []
    monkeypatch.setattr(
        whole_moe,
        "_bind",
        lambda block, rows: SimpleNamespace(block=block, rows=rows),
    )
    monkeypatch.setattr(
        whole_moe,
        "_selfcheck",
        lambda block, accepted_call, rows: seen.append(rows) or rows / 10,
    )

    report = whole_moe.configure_qwen4_whole_moe(
        model,
        config=QWEN4_CONFIG,
        validate_storage=False,
        run_selfcheck=True,
    )

    assert seen == [2, 3]
    assert report["selfcheck_dmax"] == {"m2": 0.2, "m3": 0.3}
