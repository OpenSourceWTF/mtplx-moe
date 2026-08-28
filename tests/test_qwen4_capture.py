from __future__ import annotations

from types import SimpleNamespace

import pytest


class _TailMlp:
    def __call__(self, value):
        return ("stock-mlp", value.shape[-2])

    def _mtplx_residual_call(self, value, residual, inject):
        return ("fused", value.shape[-2], residual, inject)


def _config():
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_hidden_layers": 48,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-6,
        },
    }


def _runtime():
    def hyper():
        return SimpleNamespace(
            hc=4,
            d=2560,
            hc_norm=SimpleNamespace(weight=SimpleNamespace(shape=(10240,)), eps=1e-6),
            block_inject_weight=object(),
        )

    layers = [
        SimpleNamespace(
            is_linear=True,
            attn_hyper_connection=hyper(),
            mlp_hyper_connection=hyper(),
            linear_attn=SimpleNamespace(
                n_k=16,
                n_v=48,
                dk=128,
                dv=128,
                key_dim=2048,
                conv_dim=10240,
                norm=SimpleNamespace(
                    weight=SimpleNamespace(shape=(128,)),
                    eps=1e-6,
                    activation="sigmoid",
                ),
            ),
        )
        for _ in range(36)
    ] + [
        SimpleNamespace(
            is_linear=False,
            attn_hyper_connection=hyper(),
            mlp_hyper_connection=hyper(),
        )
        for _ in range(12)
    ]
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    return SimpleNamespace(model=model)


def test_installer_binds_exact_qwen4_capture_route():
    from mtplx.qwen4_capture import install_qwen4_capture_route

    runtime = _runtime()

    report = install_qwen4_capture_route(runtime, config=_config())

    assert report == {"installed": True, "linear_layers": 36, "rows": 2}
    assert runtime.forward_ar_capture.__func__.__name__ == "_forward_ar_capture"


def test_installer_selects_constructed_m2_hyper_route():
    from mtplx.qwen4_capture import install_qwen4_capture_route

    runtime = _runtime()
    inner = runtime.model.language_model.model
    for layer in inner.layers:
        layer.attn_hyper_connection._mtplx_m2_hyper_call = lambda hidden, normed: (
            hidden,
            normed,
            hidden,
        )
        layer.mlp_hyper_connection._mtplx_m2_hyper_call = lambda hidden, normed: (
            hidden,
            normed,
            hidden,
        )

    install_qwen4_capture_route(runtime, config=_config())

    assert (
        inner._mtplx_capture_hyper_from_normed.__name__
        == "_qwen4_m2_hyper_from_normed"
    )
    assert inner._mtplx_capture_attn_hyper.__name__ == "_qwen4_m2_attn_hyper"


def test_m2_attention_hyper_uses_stock_norm_and_bound_kernel():
    import mtplx.qwen4_capture as capture

    hidden = SimpleNamespace(shape=(1, 2, 10240))
    module = SimpleNamespace(
        hc_norm=lambda value: ("normed", value),
        _mtplx_m2_hyper_call=lambda value, normed: ("fused", value, normed),
    )

    assert capture._qwen4_m2_attn_hyper(module, hidden) == (
        "fused",
        hidden,
        ("normed", hidden),
    )


def test_m2_attention_hyper_rejects_batched_two_row_input(monkeypatch):
    import mtplx.qwen4_capture as capture

    hidden = SimpleNamespace(shape=(2, 2, 10240))
    module = SimpleNamespace(
        _mtplx_m2_hyper_call=lambda value, normed: "fused",
    )
    monkeypatch.setattr(
        capture,
        "_qwen4_stock_attn_hyper",
        lambda candidate, residual: "stock",
    )

    assert capture._qwen4_m2_attn_hyper(module, hidden) == "stock"


def test_installer_rejects_partial_attention_hyper_route():
    from mtplx.qwen4_capture import Qwen4CaptureConfigError, install_qwen4_capture_route

    runtime = _runtime()
    inner = runtime.model.language_model.model
    inner.layers[0].attn_hyper_connection._mtplx_m2_hyper_call = (
        lambda hidden, normed: None
    )

    with pytest.raises(Qwen4CaptureConfigError, match="attention hyper M=2 route"):
        install_qwen4_capture_route(runtime, config=_config())


def test_m2_hyper_route_rejects_batched_two_row_input(monkeypatch):
    import mtplx.qwen4_capture as capture

    module = SimpleNamespace(
        _mtplx_m2_hyper_call=lambda hidden, normed: "fused"
    )
    hidden = SimpleNamespace(shape=(2, 2, 10240))
    normed = SimpleNamespace(shape=(2, 2, 10240))
    monkeypatch.setattr(
        capture,
        "_qwen4_stock_hyper_from_normed",
        lambda candidate, residual, normalized: "stock",
    )

    assert capture._qwen4_m2_hyper_from_normed(module, hidden, normed) == "stock"


def test_installer_rejects_wrong_recurrent_geometry():
    from mtplx.qwen4_capture import Qwen4CaptureConfigError, install_qwen4_capture_route

    runtime = _runtime()
    runtime.model.language_model.model.layers[0].linear_attn.n_v = 32

    with pytest.raises(Qwen4CaptureConfigError, match="recurrent geometry"):
        install_qwen4_capture_route(runtime, config=_config())


def test_capture_route_support_is_limited_to_the_measured_config():
    from mtplx.qwen4_capture import is_exact_qwen4_capture_config

    config = _config()
    assert is_exact_qwen4_capture_config(config)

    config["text_config"]["hidden_size"] = 2048
    assert not is_exact_qwen4_capture_config(config)


def test_capture_route_support_rejects_a_different_output_gate():
    from mtplx.qwen4_capture import is_exact_qwen4_capture_config

    config = _config()
    config["text_config"]["output_gate_type"] = "silu"

    assert not is_exact_qwen4_capture_config(config)


def test_residual_mlp_tail_routes_only_fixed_decode_rows(monkeypatch):
    import mtplx.qwen4_capture as capture

    layer = SimpleNamespace(mlp=_TailMlp())
    monkeypatch.setattr(
        capture,
        "_qwen4_stock_mlp_tail",
        lambda _layer, value, residual, inject: (
            "stock",
            value.shape[-2],
            residual,
            inject,
        ),
    )

    for rows in (2, 3):
        value = SimpleNamespace(shape=(1, rows, 2560))
        assert capture._qwen4_residual_mlp_tail(layer, value, "r", "i") == (
            "fused",
            rows,
            "r",
            "i",
        )
    value = SimpleNamespace(shape=(1, 9, 2560))
    assert capture._qwen4_residual_mlp_tail(layer, value, "r", "i") == (
        "stock",
        9,
        "r",
        "i",
    )
