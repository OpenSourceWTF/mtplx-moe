from __future__ import annotations

from types import SimpleNamespace

import pytest


def _config():
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
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
    linear = SimpleNamespace(
        is_linear=True,
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
    attention = SimpleNamespace(is_linear=False)
    layers = [linear for _ in range(36)] + [attention for _ in range(12)]
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
