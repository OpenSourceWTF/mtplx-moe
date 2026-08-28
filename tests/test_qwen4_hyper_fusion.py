from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


def _model():
    layers = [
        SimpleNamespace(mlp_hyper_connection=SimpleNamespace()) for _ in range(48)
    ]
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )


def test_exact_m2_hyper_route_is_bound_once_at_construction(monkeypatch):
    import mtplx.qwen4_hyper_fusion as fusion

    model = _model()
    bound = []
    monkeypatch.setattr(
        fusion,
        "_bind",
        lambda module: bound.append(module) or (lambda hidden, normed: None),
    )

    report = fusion.configure_qwen4_hyper_fusion(
        model,
        validate_storage=False,
        run_selfcheck=False,
    )

    assert report == {
        "installed": True,
        "installed_blocks": 48,
        "rows": 2,
        "selfcheck_dmax": None,
    }
    assert len(bound) == 48
    assert all(
        callable(layer.mlp_hyper_connection._mtplx_m2_hyper_call)
        for layer in model.language_model.model.layers
    )


def test_storage_and_exact_selfcheck_run_before_install(monkeypatch):
    import mtplx.qwen4_hyper_fusion as fusion

    model = _model()
    validated = []
    checked = []
    monkeypatch.setattr(
        fusion,
        "_validate_hyper",
        lambda module, index: validated.append((module, index)),
    )
    monkeypatch.setattr(fusion, "_bind", lambda module: object())
    monkeypatch.setattr(
        fusion,
        "_selfcheck",
        lambda module, binding: checked.append((module, binding)) or 0.0,
    )

    report = fusion.configure_qwen4_hyper_fusion(model)

    assert [index for _, index in validated] == list(range(48))
    assert len(checked) == 48
    assert report["selfcheck_dmax"] == 0.0


def test_bound_kernel_preserves_hidden_ownership(monkeypatch):
    import mtplx.qwen4_hyper_fusion as fusion

    module = SimpleNamespace()
    monkeypatch.setattr(
        fusion,
        "kernels",
        SimpleNamespace(bind_m2=lambda candidate: lambda normed: ("mixed", "inject")),
        raising=False,
    )

    call = fusion._bind(module)

    assert call("hidden", "normed") == ("mixed", "hidden", "inject")


def test_selfcheck_requires_exact_hidden_and_bounded_projection_delta(monkeypatch):
    import mtplx.qwen4_hyper_fusion as fusion

    expected_mixed = mx.zeros((1, 2, 2560), dtype=mx.bfloat16)
    expected_hidden = mx.ones((1, 2, 10240), dtype=mx.bfloat16)
    expected_inject = mx.zeros((1, 2, 4), dtype=mx.bfloat16)
    monkeypatch.setattr(
        fusion,
        "_stock_hyper",
        lambda module, hidden, normed: (
            expected_mixed,
            expected_hidden,
            expected_inject,
        ),
    )

    dmax = fusion._selfcheck(
        SimpleNamespace(),
        lambda hidden, normed: (
            expected_mixed,
            expected_hidden,
            expected_inject,
        ),
    )

    assert dmax == 0.0


def test_selfcheck_rejects_nonfinite_projection_output(monkeypatch):
    import mtplx.qwen4_hyper_fusion as fusion

    expected_mixed = mx.zeros((1, 2, 2560), dtype=mx.bfloat16)
    expected_hidden = mx.ones((1, 2, 10240), dtype=mx.bfloat16)
    expected_inject = mx.zeros((1, 2, 4), dtype=mx.bfloat16)
    monkeypatch.setattr(
        fusion,
        "_stock_hyper",
        lambda module, hidden, normed: (
            expected_mixed,
            expected_hidden,
            expected_inject,
        ),
    )

    with pytest.raises(fusion.Qwen4HyperFusionConfigError, match="non-finite"):
        fusion._selfcheck(
            SimpleNamespace(),
            lambda hidden, normed: (
                mx.full((1, 2, 2560), float("nan"), dtype=mx.bfloat16),
                expected_hidden,
                expected_inject,
            ),
        )


def test_hyper_route_is_explicitly_enabled_at_construction(monkeypatch):
    import mtplx.qwen4_hyper_fusion as fusion

    monkeypatch.delenv("MTPLX_QWEN4_HYPER_M2", raising=False)
    assert not fusion.qwen4_hyper_fusion_enabled()

    monkeypatch.setenv("MTPLX_QWEN4_HYPER_M2", "1")
    assert fusion.qwen4_hyper_fusion_enabled()
