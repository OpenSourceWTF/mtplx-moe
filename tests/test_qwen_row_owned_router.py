"""Construction-owned row routing for the exact Qwen A3B model."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import kernel_selfcheck, runtime as runtime_module
import mtplx.qwen_row_owned_router as router_module
from mtplx.qwen_row_owned_router import (
    install_qwen_row_owned_routers,
    prepare_qwen_row_owned_routers,
    qwen_combine_tail_enabled,
    qwen_combine_tail_m1,
    qwen_combine_tail_m2,
    qwen_row_owned_route,
    qwen_row_owned_router_eligible,
    qwen_row_owned_router_enabled,
    qwen_row_owned_router_source,
)


_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class _ArraySpec:
    def __init__(self, shape, dtype=mx.bfloat16) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype


class _FakeGate:
    def __init__(self, *, target: bool) -> None:
        if target:
            self.bits = 8
            self.group_size = 64
            self.mode = "affine"
            self.weight = _ArraySpec((256, 512), mx.uint32)
            self.scales = _ArraySpec((256, 32))
            self.biases = _ArraySpec((256, 32))
        else:
            self.weight = _ArraySpec((256, 2048))

    def __call__(self, value: mx.array) -> mx.array:
        return mx.zeros((*value.shape[:-1], 256), dtype=value.dtype)

    def __contains__(self, name: str) -> bool:
        return hasattr(self, name)

    def __getitem__(self, name: str):
        return getattr(self, name)


class _FakeSparseBlock:
    def __init__(self, *, target: bool) -> None:
        self.gate = _FakeGate(target=target)
        self.num_experts = 256
        self.top_k = 8
        self.norm_topk_prob = True
        self.sharding_group = None
        self.switch_mlp = lambda x, inds: mx.zeros(
            (*inds.shape, int(x.shape[-1])), dtype=x.dtype
        )
        self.shared_expert = lambda x: mx.zeros_like(x)
        self.shared_expert_gate = lambda x: mx.zeros(
            (*x.shape[:-1], 1), dtype=x.dtype
        )
        self.stock_calls: list[mx.array] = []

    def __call__(self, value: mx.array) -> mx.array:
        self.stock_calls.append(value)
        return value + 1


def _fake_a3b_config():
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(_LAYER_TYPES),
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "norm_topk_prob": True,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
        },
    }


def _fake_a3b_model(*, target_count: int = 40, mtp_count: int = 1):
    targets = [_FakeSparseBlock(target=True) for _ in range(target_count)]
    target_layers = []
    for index, block in enumerate(targets):
        kind = _LAYER_TYPES[index]
        layer = SimpleNamespace(
            is_linear=kind == "linear_attention",
            mlp=block,
            post_attention_layernorm=SimpleNamespace(
                weight=_ArraySpec((2048,), mx.bfloat16)
            ),
        )
        if kind == "linear_attention":
            layer.linear_attn = object()
        else:
            layer.self_attn = object()
        target_layers.append(layer)
    mtp_blocks = [_FakeSparseBlock(target=False) for _ in range(mtp_count)]
    mtp_layers = [
        SimpleNamespace(
            mlp=block,
            post_attention_layernorm=SimpleNamespace(
                weight=_ArraySpec((2048,), mx.bfloat16)
            ),
            self_attn=object(),
        )
        for block in mtp_blocks
    ]
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=target_layers)
        ),
        mtp=SimpleNamespace(layers=mtp_layers),
    )
    return model, targets, mtp_blocks


def _stock_route(probabilities: mx.array) -> tuple[mx.array, mx.array]:
    indices = mx.argpartition(probabilities, kth=-8, axis=-1)[..., -8:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


def _stock_combine(routed: mx.array, scores: mx.array) -> mx.array:
    return (routed * scores[..., None]).sum(axis=-2)


@pytest.fixture(autouse=True)
def _clean_installation_and_selfcheck():
    router_module._reset_qwen_row_owned_router_for_tests()
    kernel_selfcheck._reset_for_tests()
    yield
    router_module._reset_qwen_row_owned_router_for_tests()
    kernel_selfcheck._reset_for_tests()


def test_row_owned_router_is_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_QWEN_ROW_OWNED_ROUTER", raising=False)
    assert not qwen_row_owned_router_enabled()
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    assert qwen_row_owned_router_enabled()


def test_combine_tail_is_read_only_at_construction(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_QWEN_COMBINE_TAIL", raising=False)
    assert not qwen_combine_tail_enabled()
    monkeypatch.setenv("MTPLX_QWEN_COMBINE_TAIL", "1")
    assert qwen_combine_tail_enabled()


def test_fixed_m1_m2_combine_entrypoints_are_bitwise_stock() -> None:
    mx.random.seed(174)
    for rows, entrypoint in ((1, qwen_combine_tail_m1), (2, qwen_combine_tail_m2)):
        routed = mx.random.normal(
            (1, rows, 8, 2048), dtype=mx.float32
        ).astype(mx.bfloat16)
        scores = mx.softmax(
            mx.random.normal((1, rows, 8), dtype=mx.float32), axis=-1
        ).astype(mx.bfloat16)
        stock = _stock_combine(routed, scores)
        candidate = entrypoint(routed, scores)
        mx.eval(stock, candidate)
        assert candidate.shape == (1, rows, 2048)
        assert mx.array_equal(candidate, stock).item()


def test_fixed_combine_entrypoints_contain_no_runtime_validation() -> None:
    for entrypoint in (qwen_combine_tail_m1, qwen_combine_tail_m2):
        source = inspect.getsource(entrypoint)
        for forbidden in (
            "os.environ",
            "lane_disabled",
            "selfcheck",
            ".dtype",
            "eligible",
            "fallback",
            "try:",
            "raise ",
            "_STATS",
        ):
            assert forbidden not in source


def test_checked_public_helper_has_exact_m1_to_m16_contract() -> None:
    for rows in range(1, 17):
        assert qwen_row_owned_router_eligible(
            rows=rows,
            experts=256,
            input_dtype=mx.bfloat16,
            top_k=8,
            norm_topk_prob=True,
            available=True,
        )
    assert not qwen_row_owned_router_eligible(
        rows=17,
        experts=256,
        input_dtype=mx.bfloat16,
        top_k=8,
        norm_topk_prob=True,
        available=True,
    )


def test_kernel_source_is_one_threadgroup_per_row_without_global_protocol() -> None:
    source = qwen_row_owned_router_source()
    assert "uint row = threadgroup_position_in_grid.x" in source
    assert "constexpr int SIMD_GROUPS = 8" in source
    assert "threadgroup_barrier" in source
    assert "matmul2d_descriptor" not in source
    for forbidden in ("atomic_", "device_barrier", "epoch", "scratch"):
        assert forbidden not in source


def test_m2_rows_equal_independent_m1_routes_bitwise() -> None:
    mx.random.seed(358)
    logits = (mx.random.normal((2, 256), dtype=mx.float32) * 0.5).astype(
        mx.bfloat16
    )
    probabilities = mx.softmax(logits, axis=-1, precise=True)
    batch_ids, batch_scores = qwen_row_owned_route(probabilities)
    row0_ids, row0_scores = qwen_row_owned_route(probabilities[:1])
    row1_ids, row1_scores = qwen_row_owned_route(probabilities[1:])
    expected_ids = mx.concatenate([row0_ids, row1_ids])
    expected_scores = mx.concatenate([row0_scores, row1_scores])
    mx.eval(batch_ids, batch_scores, expected_ids, expected_scores)
    assert mx.array_equal(batch_ids, expected_ids).item()
    assert mx.array_equal(batch_scores, expected_scores).item()


def test_finalizer_is_bitwise_stock_for_every_supported_row() -> None:
    mx.random.seed(174)
    logits = (mx.random.normal((16, 256), dtype=mx.float32) * 0.5).astype(
        mx.bfloat16
    )
    probabilities = mx.softmax(logits, axis=-1, precise=True)
    for rows in range(1, 17):
        stock_ids, stock_scores = _stock_route(probabilities[:rows])
        candidate_ids, candidate_scores = qwen_row_owned_route(probabilities[:rows])
        mx.eval(stock_ids, stock_scores, candidate_ids, candidate_scores)
        assert mx.array_equal(candidate_ids, stock_ids).item()
        assert mx.array_equal(candidate_scores, stock_scores).item()


def test_configuration_validates_then_installs_all_41_after_selfcheck(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    model, targets, mtp = _fake_a3b_model()
    original_classes = [type(block) for block in targets + mtp]

    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    assert plan is not None
    assert [type(block) for block in targets + mtp] == original_classes
    report = install_qwen_row_owned_routers(
        plan, {"lanes": {"qwen_row_owned_router": "ok"}}
    )

    assert report == {
        "enabled": True,
        "installed": True,
        "installation_status": "installed",
        "installation_error": None,
        "target_routers": 40,
        "mtp_routers": 1,
        "validated_contract": router_module._a3b_router_contract(),
    }
    assert all(type(block).__call__ is router_module._installed_a3b_router_call for block in targets + mtp)


def test_combine_flag_installs_the_fixed_m1_m2_class_after_both_selfchecks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    monkeypatch.setenv("MTPLX_QWEN_COMBINE_TAIL", "1")
    model, targets, mtp = _fake_a3b_model()

    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    assert plan is not None
    assert plan.combine_tail
    report = install_qwen_row_owned_routers(
        plan,
        {
            "lanes": {
                "qwen_row_owned_router": "ok",
                "qwen_combine_tail_m1_m2": "ok",
            }
        },
    )

    assert report["validated_contract"]["combine_tail"] == {
        "decode_verify": [1, 2],
        "ar_decode": [1, 2],
        "other_rows": "stock_weighted_reduction",
    }
    assert all(
        type(block).__call__ is router_module._installed_a3b_router_combine_call
        for block in targets + mtp
    )


def test_combine_requires_row_owned_router_installation(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_QWEN_ROW_OWNED_ROUTER", raising=False)
    monkeypatch.setenv("MTPLX_QWEN_COMBINE_TAIL", "1")
    model, targets, mtp = _fake_a3b_model()

    with pytest.raises(router_module.QwenRowOwnedRouterConfigError, match="requires"):
        prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    assert all(type(block) is _FakeSparseBlock for block in targets + mtp)


def test_combine_selfcheck_failure_prevents_installation(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    monkeypatch.setenv("MTPLX_QWEN_COMBINE_TAIL", "1")
    model, targets, mtp = _fake_a3b_model()
    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())

    with pytest.raises(router_module.QwenRowOwnedRouterConfigError, match="combine"):
        install_qwen_row_owned_routers(
            plan,
            {
                "lanes": {
                    "qwen_row_owned_router": "ok",
                    "qwen_combine_tail_m1_m2": "fallback",
                }
            },
        )
    assert all(type(block) is _FakeSparseBlock for block in targets + mtp)


def test_flag_off_leaves_every_stock_block_unchanged() -> None:
    model, targets, mtp = _fake_a3b_model()
    original_classes = [type(block) for block in targets + mtp]

    assert prepare_qwen_row_owned_routers(model, config=_fake_a3b_config()) is None
    assert [type(block) for block in targets + mtp] == original_classes
    value = mx.zeros((1, 1, 2048), dtype=mx.bfloat16)
    assert mx.array_equal(targets[0](value), value + 1).item()
    assert targets[0].stock_calls == [value]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda config, model: config.update(model_type="qwen3_5"), "config"),
        (lambda config, model: config["text_config"].update(num_hidden_layers=39), "config"),
        (lambda config, model: setattr(model.language_model.model.layers[0].mlp, "top_k", 4), "top-k"),
        (lambda config, model: setattr(model.language_model.model.layers[0].mlp, "norm_topk_prob", False), "normalization"),
        (lambda config, model: setattr(model.language_model.model.layers[0].mlp, "sharding_group", object()), "sharding"),
        (lambda config, model: setattr(model.language_model.model.layers[0].mlp, "num_experts", 128), "expert"),
        (lambda config, model: setattr(model.language_model.model.layers[0].mlp.gate, "bits", 4), "target gate"),
        (lambda config, model: setattr(model.mtp.layers[0].mlp.gate.weight, "dtype", mx.float16), "MTP gate"),
        (lambda config, model: setattr(model.language_model.model.layers[0].post_attention_layernorm.weight, "dtype", mx.float16), "BF16"),
    ],
)
def test_invalid_external_model_fact_prevents_installation(monkeypatch, mutate, match) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    config = _fake_a3b_config()
    model, targets, mtp = _fake_a3b_model()
    original_classes = [type(block) for block in targets + mtp]
    mutate(config, model)

    with pytest.raises(router_module.QwenRowOwnedRouterConfigError, match=match):
        prepare_qwen_row_owned_routers(model, config=config)
    assert [type(block) for block in targets + mtp] == original_classes


def test_incomplete_router_ownership_prevents_installation(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    model, targets, _mtp = _fake_a3b_model(mtp_count=0)

    with pytest.raises(router_module.QwenRowOwnedRouterConfigError, match="40 target and one MTP"):
        prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    assert all(type(block) is _FakeSparseBlock for block in targets)


def test_selfcheck_failure_prevents_installation(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    model, targets, mtp = _fake_a3b_model()
    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())

    with pytest.raises(router_module.QwenRowOwnedRouterConfigError, match="selfcheck"):
        install_qwen_row_owned_routers(
            plan, {"lanes": {"qwen_row_owned_router": "fallback"}}
        )
    assert all(type(block) is _FakeSparseBlock for block in targets + mtp)


def test_installed_m2_decode_routes_directly(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    model, targets, _mtp = _fake_a3b_model()
    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    install_qwen_row_owned_routers(
        plan, {"lanes": {"qwen_row_owned_router": "ok"}}
    )
    observed = []

    def fake_route(probabilities, *, rows):
        observed.append((probabilities.shape, rows))
        shape = (*probabilities.shape[:-1], 8)
        return (
            mx.zeros(shape, dtype=mx.uint32),
            mx.full(shape, 1.0 / 8.0, dtype=mx.bfloat16),
        )

    monkeypatch.setattr(router_module, "current_attention_phase", lambda: "decode_verify")
    monkeypatch.setattr(router_module, "_qwen_row_owned_route_unchecked", fake_route)
    value = mx.zeros((1, 2, 2048), dtype=mx.bfloat16)
    output = targets[0](value)
    mx.eval(output)
    assert output.shape == value.shape
    assert observed == [((1, 2, 256), 2)]
    assert targets[0].stock_calls == []


def test_installed_combine_routes_m1_m2_directly_and_m3_explicitly_stock(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    monkeypatch.setenv("MTPLX_QWEN_COMBINE_TAIL", "1")
    model, targets, _mtp = _fake_a3b_model()
    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    install_qwen_row_owned_routers(
        plan,
        {
            "lanes": {
                "qwen_row_owned_router": "ok",
                "qwen_combine_tail_m1_m2": "ok",
            }
        },
    )
    observed: list[int] = []

    def fake_route(probabilities, *, rows):
        shape = (*probabilities.shape[:-1], 8)
        return (
            mx.zeros(shape, dtype=mx.uint32),
            mx.full(shape, 1.0 / 8.0, dtype=mx.bfloat16),
        )

    def fake_m1(routed, scores):
        observed.append(1)
        return _stock_combine(routed, scores)

    def fake_m2(routed, scores):
        observed.append(2)
        return _stock_combine(routed, scores)

    monkeypatch.setattr(router_module, "current_attention_phase", lambda: "decode_verify")
    monkeypatch.setattr(router_module, "_qwen_row_owned_route_unchecked", fake_route)
    monkeypatch.setattr(router_module, "qwen_combine_tail_m1", fake_m1)
    monkeypatch.setattr(router_module, "qwen_combine_tail_m2", fake_m2)
    for rows in (1, 2, 3):
        output = targets[0](mx.zeros((1, rows, 2048), dtype=mx.bfloat16))
        mx.eval(output)
        assert output.shape == (1, rows, 2048)
    assert observed == [1, 2]
    assert targets[0].stock_calls == []


def test_prefill_and_unsupported_phase_are_explicit_stock_routes(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    model, targets, _mtp = _fake_a3b_model()
    plan = prepare_qwen_row_owned_routers(model, config=_fake_a3b_config())
    install_qwen_row_owned_routers(
        plan, {"lanes": {"qwen_row_owned_router": "ok"}}
    )
    block = targets[0]
    value = mx.zeros((1, 2, 2048), dtype=mx.bfloat16)

    monkeypatch.setattr(router_module, "current_attention_phase", lambda: "prefill")
    assert mx.array_equal(block(value), value + 1).item()
    monkeypatch.setattr(router_module, "current_attention_phase", lambda: "postcommit")
    assert mx.array_equal(block(value), value + 1).item()
    assert block.stock_calls == [value, value]


def test_installed_execution_checks_only_dynamic_phase_and_rows() -> None:
    hot_source = inspect.getsource(router_module._installed_a3b_router_call)
    unchecked_source = inspect.getsource(router_module._qwen_row_owned_route_unchecked)
    runtime_source = inspect.getsource(runtime_module.load)

    assert "current_attention_phase" in hot_source
    assert "value.shape[:-1]" in hot_source
    for forbidden in (
        "os.environ",
        "lane_disabled",
        "selfcheck",
        "sharding_group",
        ".dtype",
        "top_k",
        "norm_topk_prob",
        "num_experts",
        "eligible",
        "fallback",
        "_STATS",
        "try:",
    ):
        assert forbidden not in hot_source
    for forbidden in ("raise ", ".dtype", "eligible", "_STATS", "fallback"):
        assert forbidden not in unchecked_source
    prepare = runtime_source.index("prepare_qwen_row_owned_routers(")
    selfcheck = runtime_source.index("maybe_run_model_selfcheck(model)")
    install = runtime_source.index("install_qwen_row_owned_routers(")
    assert prepare < selfcheck < install


def test_installed_combine_execution_has_no_validation_or_fallback_accounting() -> None:
    hot_source = inspect.getsource(router_module._installed_a3b_router_combine_call)
    assert "current_attention_phase" in hot_source
    assert "value.shape[:-1]" in hot_source
    for forbidden in (
        "os.environ",
        "lane_disabled",
        "selfcheck",
        "sharding_group",
        ".dtype",
        "top_k",
        "norm_topk_prob",
        "num_experts",
        "eligible",
        "fallback",
        "_STATS",
        "try:",
    ):
        assert forbidden not in hot_source


def test_kernel_selfcheck_validates_the_complete_m1_m16_route(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["qwen_row_owned_router"] == "ok"
    assert report["dmax"]["qwen_row_owned_router"] == 0.0


def test_kernel_selfcheck_validates_exact_combine_m1_m2(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_OWNED_ROUTER", "1")
    monkeypatch.setenv("MTPLX_QWEN_COMBINE_TAIL", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    report = kernel_selfcheck.run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert report["lanes"]["qwen_combine_tail_m1_m2"] == "ok"
    assert report["dmax"]["qwen_combine_tail_m1_m2"] == 0.0


def test_combine_selfcheck_fixture_does_not_mutate_global_rng() -> None:
    combine_source = inspect.getsource(
        kernel_selfcheck._check_qwen_combine_tail_m1_m2
    )

    assert "mx.random" not in combine_source


def test_router_selfcheck_fixture_does_not_mutate_global_rng() -> None:
    source = inspect.getsource(kernel_selfcheck._check_qwen_row_owned_router)

    assert "mx.random" not in source
