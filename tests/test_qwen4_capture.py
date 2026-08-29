from __future__ import annotations

import gc
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache


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
    layers[1].ple = SimpleNamespace()
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
    assert (
        runtime.prepare_compiled_verify_aux.__func__.__name__
        == "_prepare_compiled_verify_aux"
    )
    assert runtime.context_copy_probation_k == 24


def test_installer_can_construct_the_corrected_prefix_route_without_prefetch(
    monkeypatch,
):
    from mtplx.qwen4_capture import install_qwen4_capture_route

    monkeypatch.setenv("MTPLX_QWEN4_PLE_PREFETCH", "0")
    runtime = _runtime()

    install_qwen4_capture_route(runtime, config=_config())

    assert not hasattr(runtime, "prefetch_compiled_verify_aux")
    assert runtime._mtplx_capture_extra_layout == (
        (1, ("ple_conv_full", "ple_context_full")),
    )


def test_compiled_ple_aux_is_prepared_without_mutating_context():
    from mtplx.qwen4_capture import _prepare_compiled_verify_aux

    calls = []

    def embedding(ids, previous):
        calls.append((ids, previous))
        return mx.ones((1, 2, 8), dtype=mx.bfloat16)

    inner = SimpleNamespace(
        ple_layers=[1],
        args=SimpleNamespace(ngram_size=3, eos_token_id=0),
        layers=[None, SimpleNamespace(ple=SimpleNamespace(ple_embedding=embedding))],
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(model=inner))
    )
    previous = mx.array([[7, 8]], dtype=mx.int32)
    cache = [None, [None, None, None, previous]]
    ids = mx.array([[9, 10]], dtype=mx.int32)

    result = _prepare_compiled_verify_aux(runtime, ids, cache)

    assert calls == [(ids, previous)]
    assert result.shape == (1, 2, 8)
    assert cache[1][3] is previous


def test_compiled_ple_prefetches_primary_row_before_draft_row_materializes():
    from mtplx.qwen4_capture import (
        _prefetch_compiled_verify_aux,
        _prepare_compiled_verify_aux,
    )

    calls: list[tuple] = []
    class Marker:
        def cancel(self):
            calls.append(("cancel",))
            return True

    marker = Marker()

    class Geometry:
        eos_token_id = 0

        def plan_incremental(self, ids, prior_context=None):
            host = np.asarray(ids, dtype=np.int64)
            calls.append(("plan", host.tolist(), prior_context))
            token = int(host[0, 0])
            return (
                np.array([[[10 * token, 10 * token + 1]]], dtype=np.int64),
                ((int(prior_context[0][-1]), token),),
            )

    class Rows:
        def prefetch_prevalidated_rows(self, rows):
            calls.append(("prefetch", rows))
            return marker

        def materialize_prevalidated_rows(
            self, rows, *, logical_shape, prefetched
        ):
            calls.append(("materialize", rows, logical_shape, prefetched))
            return mx.ones((*logical_shape, 8), dtype=mx.bfloat16)

    rows = Rows()
    embedding = SimpleNamespace(ngram_embedding=rows)
    inner = SimpleNamespace(
        ple_layers=[1],
        args=SimpleNamespace(ngram_size=3, eos_token_id=0),
        layers=[None, SimpleNamespace(ple=SimpleNamespace(ple_embedding=embedding))],
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(model=inner)),
        _mtplx_qwen4_ngram_geometry=Geometry(),
    )
    previous = mx.array([[7, 8]], dtype=mx.int32)
    cache = [None, [None, None, None, previous]]

    prefetched = _prefetch_compiled_verify_aux(
        runtime, primary=9, prior_context=(7, 8)
    )
    result = _prepare_compiled_verify_aux(
        runtime,
        mx.array([[9, 10]], dtype=mx.int32),
        cache,
        prefetched=prefetched,
    )

    assert calls == [
        ("plan", [[9]], ((7, 8),)),
        ("prefetch", (90, 91)),
        ("plan", [[10]], ((8, 9),)),
        ("materialize", (90, 91, 100, 101), (1, 2, 2), marker),
    ]
    assert result.shape == (1, 2, 16)
    assert cache[1][3] is previous


def test_unconsumed_ple_prefetch_cancels_when_cycle_owner_unwinds():
    from mtplx.qwen4_capture import _CompiledAuxPrefetch

    cancellations: list[bool] = []

    class Future:
        def cancel(self):
            cancellations.append(True)
            return True

    prefetched = _CompiledAuxPrefetch(
        rows=(1, 2),
        next_context=((7, 8),),
        future=Future(),
    )

    del prefetched
    gc.collect()

    assert cancellations == [True]


def test_capture_commit_restores_ple_state_to_the_accepted_prefix():
    from mtplx.gdn_capture import commit_captured_prefix

    entry = ArraysCache(4)
    entry[0] = mx.zeros((1, 3, 1), dtype=mx.float32)
    entry[1] = mx.zeros((1, 1, 1, 1), dtype=mx.float32)
    entry[2] = mx.full((1, 2, 1), 99, dtype=mx.float32)
    entry[3] = mx.array([[99, 99]], dtype=mx.int32)
    capture = {
        "conv_states": mx.array(
            [[[[1.0]], [[2.0]], [[3.0]]], [[[4.0]], [[5.0]], [[6.0]]]]
        ).reshape(1, 2, 3, 1),
        "states": mx.array([1.0, 2.0]).reshape(1, 2, 1, 1, 1),
        "ple_conv_full": mx.array([10.0, 11.0, 20.0, 21.0]).reshape(1, 4, 1),
        "ple_context_full": mx.array([[7, 8, 9, 10]], dtype=mx.int32),
    }

    committed = commit_captured_prefix(
        [entry], {0: capture}, keep_tokens=1, verified_tokens=2
    )
    mx.eval(*entry.cache)

    assert committed is True
    np.testing.assert_array_equal(np.asarray(entry[2]).reshape(-1), [11.0, 20.0])
    np.testing.assert_array_equal(np.asarray(entry[3]).reshape(-1), [8, 9])


def test_ple_capture_helper_matches_stock_arithmetic_and_state():
    from mtplx.models.qwen4_omlx import PLELayer
    from mtplx.qwen4_capture import _qwen4_ple_with_capture

    args = SimpleNamespace(
        hidden_size=4,
        hc_count=2,
        ple_embed_dim=4,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=0,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=1,
        vocab_size=32,
        seed=7,
        rms_norm_eps=1e-6,
    )
    ple = PLELayer(args, ple_layer_index=0, layer_index=1)
    hidden = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) / 16
    embedding = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4) / 8
    ids = mx.array([[9, 10]], dtype=mx.int32)
    previous = mx.array([[7, 8]], dtype=mx.int32)
    mask = mx.array([[True, False]])
    initial_state = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 24
    stock_cache = [None, None, initial_state, previous]
    capture_cache = [None, None, initial_state, previous]

    stock = ple(
        hidden,
        ids,
        previous,
        stock_cache,
        conv_mask=mask,
        precomputed_embedding=embedding,
    )
    captured, full = _qwen4_ple_with_capture(
        ple,
        hidden,
        ids,
        previous,
        capture_cache,
        conv_mask=mask,
        precomputed_embedding=embedding,
    )
    mx.eval(stock, captured, full, stock_cache[2], capture_cache[2])

    np.testing.assert_allclose(np.asarray(captured), np.asarray(stock), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(capture_cache[2]), np.asarray(stock_cache[2]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(full[:, -ple.short_conv_state_len :, :]),
        np.asarray(stock_cache[2]),
        rtol=0,
        atol=0,
    )


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
