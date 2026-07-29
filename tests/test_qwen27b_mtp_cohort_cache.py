from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import inspect
from types import MappingProxyType, SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from mtplx.cache_state import (
    OwnedRecurrentStateCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
)
from mtplx.gdn_capture import (
    QwenGDNVerifyConfig,
    _bind_configured_layer_routes,
    _forward_with_gdn_capture_impl,
    _kernel_tape_capture_configured,
    bind_qwen_capture_commit_route,
    bind_qwen_authoritative_state_operation,
    bind_qwen_cache_context_length,
    extract_captured_row,
    extract_captured_row_lazy,
    forward_with_gdn_capture_configured,
    gdn_forward_with_capture_configured,
)
from mtplx.graphbank import TensorOffsetKVCache
from mtplx.qwen27b_mtp_cohort import (
    LayerCacheRoute,
    Qwen27BCompiledWidth2Target,
    QwenTensorOffsetBatchKVCache,
    Qwen27BK2DualLane,
    _demote_selfcheck_request_cache,
    assert_request_local_target_cache,
    build_qwen27b_cache_routes,
    extract_target_cache,
    merge_target_caches,
    normalize_target_cache,
)


def _kv(values: list[float]) -> KVCache:
    cache = KVCache()
    data = mx.array(values, dtype=mx.float32).reshape(1, 1, -1, 1)
    cache.state = (data, data + 100)
    mx.eval(*cache.state)
    return cache


def _owned(value: float) -> OwnedRecurrentStateCache:
    cache = OwnedRecurrentStateCache(
        initial=[
            mx.full((1, 2, 3), value),
            mx.full((1, 2, 3, 4), value + 10),
        ]
    )
    mx.eval(*cache.state)
    return cache


def _lane(routes: tuple[LayerCacheRoute, ...]) -> Qwen27BK2DualLane:
    return Qwen27BK2DualLane(
        backend_id="qwen3_next",
        depth=2,
        bits=4,
        group_size=64,
        activation_dtype=mx.bfloat16,
        hidden_variant="post_norm",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        max_width=2,
        width1_target=lambda **_kwargs: None,
        width2_target=lambda **_kwargs: None,
        cache_routes=routes,
        qlinear_routes=MappingProxyType({}),
        construction_receipt=MappingProxyType({}),
    )


def test_unequal_kv_merge_keeps_row_local_offsets_and_extracts_rows() -> None:
    routes = build_qwen27b_cache_routes(
        (SimpleNamespace(is_linear=False),),
        strict_topology=False,
    )
    lane = _lane(routes)
    first = _kv([1, 2, 3])
    second = _kv([7])

    merged = merge_target_caches(
        lane,
        (
            normalize_target_cache(lane, [first]),
            normalize_target_cache(lane, [second]),
        ),
    )

    assert isinstance(merged[0], QwenTensorOffsetBatchKVCache)
    assert merged[0].offset.tolist() == [3, 1]
    assert mx.array_equal(
        merged[0].keys[:, 0, :3, 0],
        mx.array([[1, 2, 3], [7, 0, 0]], dtype=mx.float32),
    ).item()
    extracted_first = extract_target_cache(lane, merged, 0)[0]
    extracted_second = extract_target_cache(lane, merged, 1)[0]
    assert type(extracted_first) is TensorOffsetKVCache
    assert type(extracted_second) is TensorOffsetKVCache
    assert extracted_first.keys[:, :, :3].tolist() == [
        [[[1.0], [2.0], [3.0]]]
    ]
    assert extracted_second.keys[:, :, :1].tolist() == [
        [[[7.0]]]
    ]


def test_unequal_kv_update_then_extract_preserves_each_logical_tail() -> None:
    routes = build_qwen27b_cache_routes(
        (SimpleNamespace(is_linear=False),),
        strict_topology=False,
    )
    lane = _lane(routes)
    merged = merge_target_caches(
        lane,
        (
            normalize_target_cache(lane, [_kv([1, 2, 3])]),
            normalize_target_cache(lane, [_kv([7])]),
        ),
    )
    keys = mx.array([11, 17], dtype=mx.float32).reshape(2, 1, 1, 1)
    values = keys + 100

    merged[0].update_and_fetch(keys, values)
    first = extract_target_cache(lane, merged, 0)[0]
    second = extract_target_cache(lane, merged, 1)[0]
    mx.eval(*first.state, *second.state)

    assert first.size() == 4
    assert second.size() == 2
    assert first.keys[..., :4, :].reshape(-1).tolist() == [
        1.0,
        2.0,
        3.0,
        11.0,
    ]
    assert second.keys[..., :2, :].reshape(-1).tolist() == [
        7.0,
        17.0,
    ]


def test_pair_attention_cache_keeps_fixed_capacity_and_capacity_mask() -> None:
    routes = build_qwen27b_cache_routes(
        (SimpleNamespace(is_linear=False),),
        strict_topology=False,
    )
    lane = _lane(routes)
    merged = merge_target_caches(
        lane,
        (
            normalize_target_cache(lane, [_kv([1, 2, 3])]),
            normalize_target_cache(lane, [_kv([7])]),
        ),
    )
    capacity = int(merged[0].keys.shape[2])
    keys = mx.array(
        [11, 12, 13, 17, 18, 19],
        dtype=mx.float32,
    ).reshape(2, 1, 3, 1)

    mask = merged[0].make_mask(3)
    fetched_keys, _ = merged[0].update_and_fetch(keys, keys + 100)
    mx.eval(fetched_keys, mask)

    assert int(fetched_keys.shape[2]) == capacity
    assert int(merged[0].keys.shape[2]) == capacity
    assert merged[0].offset.tolist() == [6, 4]
    assert list(mask.shape) == [2, 1, 3, capacity]
    assert mask[0, 0, -1, :7].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert mask[1, 0, -1, :5].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]


def test_compiled_width2_target_uses_explicit_state_inputs_and_outputs() -> None:
    routes = build_qwen27b_cache_routes(
        (
            SimpleNamespace(is_linear=False),
            SimpleNamespace(is_linear=True),
        ),
        strict_topology=False,
    )
    attention = QwenTensorOffsetBatchKVCache(
        mx.zeros((2, 1, 8, 1)),
        mx.zeros((2, 1, 8, 1)),
        mx.array([2, 3], dtype=mx.int32),
    )
    recurrent = OwnedRecurrentStateCache(
        initial=[
            mx.zeros((2, 1, 2)),
            mx.zeros((2, 1, 1, 1)),
        ]
    )
    compile_calls = []
    async_calls = []
    scope_calls = []

    def capture_forward(input_ids, cache):
        cache[0].cache[0] = cache[0].cache[0] + 1
        cache[0].cache[1] = cache[0].cache[1] + 2
        cache[0].cache[2] = cache[0].cache[2] + 3
        cache[1].cache[0] = cache[1].cache[0] + 4
        cache[1].cache[1] = cache[1].cache[1] + 5
        captures = {
            1: {
                name: mx.full((2, 3, 1), index)
                for index, name in enumerate(
                    ("conv_states", "conv_out", "g", "state_in", "tape"),
                    start=1,
                )
            }
        }
        return input_ids + 10, input_ids + 20, captures

    def compile_fn(fn):
        compile_calls.append(fn)
        return fn

    def async_eval(*values):
        async_calls.append(values)
        mx.eval(*values)

    @contextmanager
    def fixed_scope(execution):
        scope_calls.append(execution)
        yield

    execution = object()
    target = Qwen27BCompiledWidth2Target(
        execution=execution,
        capture_forward=capture_forward,
        fixed_scope=fixed_scope,
        cache_routes=routes,
        compile_fn=compile_fn,
        async_eval=async_eval,
    )
    result = target(
        input_ids=mx.array([[1, 2, 3], [4, 5, 6]]),
        cache=[attention, recurrent],
    )
    mx.eval(result.logits, result.hidden, *attention.state, *recurrent.state)

    assert len(compile_calls) == 1
    assert scope_calls == [execution]
    assert len(async_calls) == 2
    assert result.cache == [attention, recurrent]
    assert result.logits.tolist() == [[11, 12, 13], [14, 15, 16]]
    assert result.hidden.tolist() == [[21, 22, 23], [24, 25, 26]]
    assert attention.offset.tolist() == [5, 6]
    assert float(mx.max(attention.keys).item()) == 1.0
    assert float(mx.max(attention.values).item()) == 2.0
    assert float(mx.max(recurrent.state[0]).item()) == 4.0
    assert float(mx.max(recurrent.state[1]).item()) == 5.0
    assert tuple(result.captures[1]) == (
        "conv_states",
        "conv_out",
        "g",
        "state_in",
        "tape",
    )


def test_compiled_width2_target_releases_construction_state_without_touching_requests() -> None:
    routes = build_qwen27b_cache_routes(
        (
            SimpleNamespace(is_linear=False),
            SimpleNamespace(is_linear=True),
        ),
        strict_topology=False,
    )
    attention = QwenTensorOffsetBatchKVCache(
        mx.zeros((2, 1, 8, 1)),
        mx.zeros((2, 1, 8, 1)),
        mx.array([2, 3], dtype=mx.int32),
    )
    recurrent = OwnedRecurrentStateCache(
        initial=[
            mx.zeros((2, 1, 2)),
            mx.zeros((2, 1, 1, 1)),
        ]
    )

    def capture_forward(input_ids, cache):
        captures = {
            1: {
                name: mx.zeros((2, 3, 1))
                for name in (
                    "conv_states",
                    "conv_out",
                    "g",
                    "state_in",
                    "tape",
                )
            }
        }
        return input_ids, input_ids, captures

    @contextmanager
    def fixed_scope(_execution):
        yield

    target = Qwen27BCompiledWidth2Target(
        execution=object(),
        capture_forward=capture_forward,
        fixed_scope=fixed_scope,
        cache_routes=routes,
        compile_fn=lambda fn: fn,
        async_eval=mx.eval,
    )
    target(
        input_ids=mx.array([[1, 2, 3], [4, 5, 6]]),
        cache=[attention, recurrent],
    )
    request_leaves = tuple(attention.state) + tuple(recurrent.state)

    target.release_construction_state()

    assert all(value is None for entry in target._shadow for value in entry.state)
    assert tuple(attention.state) + tuple(recurrent.state) == request_leaves


def test_compiled_width2_enabled_call_has_no_fallback_or_dynamic_controls() -> None:
    source = inspect.getsource(Qwen27BCompiledWidth2Target.__call__)

    assert "fallback" not in source
    assert "except" not in source
    assert "os.environ" not in source
    assert "getattr(" not in source
    assert "isinstance(" not in source
    assert ".item(" not in source
    assert "mx.eval" not in source


@pytest.mark.parametrize(
    ("target_width", "row", "steps"),
    [(1, 0, 1), (2, 0, 2), (2, 1, 1)],
)
def test_fixed_capture_commit_route_extracts_trims_and_replays(
    monkeypatch: pytest.MonkeyPatch,
    target_width: int,
    row: int,
    steps: int,
) -> None:
    layers = (
        SimpleNamespace(is_linear=True, linear_attn=object()),
        SimpleNamespace(is_linear=False),
    )
    routes = build_qwen27b_cache_routes(layers, strict_topology=False)
    lane = _lane(routes)
    request_caches = (
        [_owned(1), _kv([10, 11, 12, 13])],
        [_owned(2), _kv([20, 21, 22, 23])],
    )
    if target_width == 1:
        cache = request_caches[0]
    else:
        cohort_cache = merge_target_caches(
            lane,
            tuple(
                normalize_target_cache(lane, request_cache)
                for request_cache in request_caches
            ),
        )
        cache = extract_target_cache(lane, cohort_cache, row)
    captures = {
        0: {
            "conv_states": mx.arange(target_width * 3)
            .reshape(target_width, 3, 1, 1)
            .astype(mx.float32),
            "replayed": (
                mx.arange(target_width).reshape(target_width, 1, 1, 1)
                + 100
            ).astype(mx.float32),
        }
    }
    monkeypatch.setattr(
        "mtplx.gdn_capture.bind_qwen_tape_replay",
        lambda **_kwargs: (
            lambda capture, *, steps: capture["replayed"] + steps
        ),
    )
    config = QwenGDNVerifyConfig.stock(
        capture_backend="linear_gdn_from_conv_tape",
        hidden_variant="post_norm",
    )
    route = bind_qwen_capture_commit_route(
        config=config,
        cache_routes=routes,
        layers=layers,
        target_width=target_width,
        row=row,
        verified_tokens=3,
    )

    commit_kwargs = (
        {}
        if target_width == 1
        else {"replay_memo": {}}
    )
    committed = route(cache, captures, steps=steps, **commit_kwargs)
    mx.eval(*committed[0].state, *committed[1].state)

    assert isinstance(committed[0], OwnedRecurrentStateCache)
    expected_attention_type = (
        KVCache if target_width == 1 else TensorOffsetKVCache
    )
    assert type(committed[1]) is expected_attention_type
    committed_attention_size = (
        committed[1].offset
        if target_width == 1
        else committed[1].size()
    )
    assert committed_attention_size == 4 - (3 - steps)
    assert committed[0].state[0].reshape(-1).tolist() == [
        float(row * 3 + steps - 1)
    ]
    assert committed[0].state[1].reshape(-1).tolist() == [
        float(100 + row + steps)
    ]


def test_width2_commit_rows_share_equal_prefix_tape_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = (SimpleNamespace(is_linear=True, linear_attn=object()),)
    routes = build_qwen27b_cache_routes(layers, strict_topology=False)
    lane = _lane(routes)
    cohort_cache = merge_target_caches(
        lane,
        (
            normalize_target_cache(lane, [_owned(1)]),
            normalize_target_cache(lane, [_owned(2)]),
        ),
    )
    request_caches = tuple(
        extract_target_cache(lane, cohort_cache, row)
        for row in range(2)
    )
    captures = {
        0: {
            "conv_states": mx.arange(6)
            .reshape(2, 3, 1, 1)
            .astype(mx.float32),
            "replayed": mx.arange(2)
            .reshape(2, 1, 1, 1)
            .astype(mx.float32),
        }
    }
    replay_calls = []

    def bind_replay(**_kwargs):
        def replay(capture, *, steps):
            replay_calls.append(steps)
            return capture["replayed"] + steps

        return replay

    monkeypatch.setattr(
        "mtplx.gdn_capture.bind_qwen_tape_replay",
        bind_replay,
    )
    config = QwenGDNVerifyConfig.stock(
        capture_backend="linear_gdn_from_conv_tape",
        hidden_variant="post_norm",
    )
    commit_routes = tuple(
        bind_qwen_capture_commit_route(
            config=config,
            cache_routes=routes,
            layers=layers,
            target_width=2,
            row=row,
            verified_tokens=3,
        )
        for row in range(2)
    )
    replay_memo = {}
    original_eval = mx.eval

    def forbidden_eval(*_values):
        raise AssertionError("width-2 commit materialized recurrent state")

    monkeypatch.setattr(mx, "eval", forbidden_eval)
    try:
        for row in range(2):
            commit_routes[row](
                request_caches[row],
                captures,
                steps=2,
                replay_memo=replay_memo,
            )
    finally:
        monkeypatch.setattr(mx, "eval", original_eval)

    original_eval(*(leaf for cache in request_caches for leaf in cache[0].state))
    assert replay_calls == [2]
    assert request_caches[0][0].state[1].reshape(-1).tolist() == [2.0]
    assert request_caches[1][0].state[1].reshape(-1).tolist() == [3.0]


def test_width1_capture_commit_route_preserves_stock_arrays_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = (SimpleNamespace(is_linear=True, linear_attn=object()),)
    routes = build_qwen27b_cache_routes(layers, strict_topology=False)
    cache = ArraysCache(size=2)
    cache.state = [
        mx.zeros((1, 1, 1), dtype=mx.float32),
        mx.zeros((1, 1, 1, 1), dtype=mx.float32),
    ]
    captures = {
        0: {
            "conv_states": mx.arange(3).reshape(1, 3, 1, 1).astype(mx.float32),
            "replayed": mx.full((1, 1, 1, 1), 100, dtype=mx.float32),
        }
    }
    monkeypatch.setattr(
        "mtplx.gdn_capture.bind_qwen_tape_replay",
        lambda **_kwargs: (
            lambda capture, *, steps: capture["replayed"] + steps
        ),
    )
    route = bind_qwen_capture_commit_route(
        config=QwenGDNVerifyConfig.stock(
            capture_backend="linear_gdn_from_conv_tape",
            hidden_variant="post_norm",
        ),
        cache_routes=routes,
        layers=layers,
        target_width=1,
        row=0,
        verified_tokens=3,
    )

    committed = route([cache], captures, steps=2)
    mx.eval(*committed[0].state)

    assert committed[0] is cache
    assert type(committed[0]) is ArraysCache
    assert committed[0].state[0].reshape(-1).tolist() == [1.0]
    assert committed[0].state[1].reshape(-1).tolist() == [102.0]


def test_width_specific_context_lookups_bind_plain_and_batch_kv_size() -> None:
    width1_type, width1_context = bind_qwen_cache_context_length(
        layer_index=0,
        target_width=1,
    )
    width2_type, width2_context = bind_qwen_cache_context_length(
        layer_index=0,
        target_width=2,
    )
    request = _kv([1, 2, 3])
    cohort = BatchKVCache.merge((request, _kv([7])))

    assert width1_type == "KVCache"
    assert width1_context([request]) == 3
    assert width2_type == "BatchKVCache"
    assert width2_context([cohort]) == 3


def test_owned_recurrent_merge_and_extract_do_not_alias_rows() -> None:
    routes = build_qwen27b_cache_routes(
        (SimpleNamespace(is_linear=True),),
        strict_topology=False,
    )
    lane = _lane(routes)
    first = _owned(1)
    second = _owned(2)

    merged = merge_target_caches(lane, ([first], [second]))
    row0 = extract_target_cache(lane, merged, 0)[0]
    row1 = extract_target_cache(lane, merged, 1)[0]
    row0.state[0][:] = 99
    mx.eval(row0.state[0], row1.state[0], merged[0].state[0])

    assert isinstance(merged[0], OwnedRecurrentStateCache)
    assert isinstance(row0, OwnedRecurrentStateCache)
    assert row1.state[0].tolist() == [[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
    assert merged[0].state[0][1:2].tolist() == row1.state[0].tolist()


def test_capture_extract_slices_every_batch_array_leaf_and_preserves_metadata() -> None:
    captures = {
        0: {
            "conv_states": mx.arange(24).reshape(2, 2, 2, 3),
            "state_in": mx.arange(48).reshape(2, 2, 2, 3, 2),
            "tape": mx.arange(16).reshape(2, 2, 2, 2),
            "replayed_state": mx.arange(48).reshape(2, 2, 2, 3, 2),
            "gdn_meta": {"head_k_dim": 2, "label": "unchanged"},
            "capture_start": 0,
        },
        "__final_only__": False,
    }

    row = extract_captured_row(captures, 1)

    for key in ("conv_states", "state_in", "tape", "replayed_state"):
        assert row[0][key].shape[0] == 1
        assert row[0][key].tolist() == captures[0][key][1:2].tolist()
    assert row[0]["gdn_meta"] is captures[0]["gdn_meta"]
    assert row[0]["capture_start"] == 0
    assert row["__final_only__"] is False


def test_capture_row_zero_commit_data_cannot_alter_row_one() -> None:
    captures = {0: {"conv_states": mx.arange(24).reshape(2, 2, 2, 3)}}
    row0 = extract_captured_row(captures, 0)
    row1 = extract_captured_row(captures, 1)

    row0[0]["conv_states"][:] = -1
    mx.eval(row0[0]["conv_states"], row1[0]["conv_states"])

    assert row1[0]["conv_states"].tolist() == captures[0]["conv_states"][1:2].tolist()


def test_cohort_merge_extract_and_capture_routes_insert_no_eval_or_scalar_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = build_qwen27b_cache_routes(
        (
            SimpleNamespace(is_linear=True),
            SimpleNamespace(is_linear=False),
        ),
        strict_topology=False,
    )
    lane = _lane(routes)
    first = [_owned(1), _kv([1, 2, 3])]
    second = [_owned(2), _kv([7])]
    first = normalize_target_cache(lane, first)
    second = normalize_target_cache(lane, second)
    captures = {
        0: {
            "conv_states": mx.arange(24).reshape(2, 2, 2, 3),
            "gdn_meta": {"head_k_dim": 2},
        }
    }

    def forbidden_eval(*_args, **_kwargs):
        raise AssertionError("cohort graph construction inserted mx.eval")

    monkeypatch.setattr(mx, "eval", forbidden_eval)
    merged = merge_target_caches(lane, (first, second))
    row0 = extract_target_cache(lane, merged, 0)
    row1 = extract_target_cache(lane, merged, 1)
    capture0 = extract_captured_row_lazy(captures, 0)
    capture1 = extract_captured_row_lazy(captures, 1)

    assert isinstance(merged[0], OwnedRecurrentStateCache)
    assert isinstance(merged[1], QwenTensorOffsetBatchKVCache)
    assert isinstance(row0[0], OwnedRecurrentStateCache)
    assert type(row0[1]) is TensorOffsetKVCache
    assert type(row1[1]) is TensorOffsetKVCache
    assert capture0[0]["conv_states"].shape[0] == 1
    assert capture1[0]["conv_states"].shape[0] == 1
    source = (
        inspect.getsource(OwnedRecurrentStateCache.merge_lazy)
        + inspect.getsource(OwnedRecurrentStateCache.extract_lazy)
        + inspect.getsource(routes[1].merge)
        + inspect.getsource(routes[1].extract)
        + inspect.getsource(extract_captured_row_lazy)
    )
    assert "mx.eval" not in source
    assert ".item(" not in source


def test_admission_normalizes_real_post_prefill_cache_types() -> None:
    layers = tuple(
        SimpleNamespace(is_linear=(index % 4 != 3))
        for index in range(64)
    )
    lane = _lane(build_qwen27b_cache_routes(layers))
    request_cache = []
    for index in range(64):
        if index % 4 == 3:
            paged = VllmMetalPagedKVCache(block_size=4, num_blocks=2)
            paged.state = _kv([index, index + 1]).state
            request_cache.append(paged)
        else:
            arrays = ArraysCache(size=2)
            arrays.state = _owned(float(index)).state
            request_cache.append(arrays)

    normalized = normalize_target_cache(lane, request_cache)

    assert [type(entry) for entry in normalized].count(
        OwnedRecurrentStateCache
    ) == 48
    assert [type(entry) for entry in normalized].count(
        TensorOffsetKVCache
    ) == 16
    assert all(
        type(normalized[index]) is TensorOffsetKVCache
        for index in range(3, 64, 4)
    )
    assert normalized is not request_cache


@pytest.mark.parametrize("preserve_paged", [False, True])
def test_pair_transition_normalizes_compiled_verify_tensor_offsets(
    preserve_paged: bool,
) -> None:
    lane = _lane(
        build_qwen27b_cache_routes(
            (SimpleNamespace(is_linear=False),),
            strict_topology=False,
        )
    )
    source = _kv([1, 2, 3])
    if preserve_paged:
        paged = VllmMetalPagedKVCache(block_size=4, num_blocks=2)
        paged.state = source.state
        compiled = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    else:
        compiled = TensorOffsetKVCache.from_kv_cache(
            source,
            reserve_tokens=3,
        )

    normalized = normalize_target_cache(lane, [compiled])[0]
    mx.eval(*normalized.state)

    assert type(normalized) is TensorOffsetKVCache
    assert normalized.size() == 3
    assert normalized.keys[..., :3, :].reshape(-1).tolist() == [1.0, 2.0, 3.0]
    assert normalized.values[..., :3, :].reshape(-1).tolist() == [
        101.0,
        102.0,
        103.0,
    ]


def test_selfcheck_demotion_compares_only_logical_attention_state() -> None:
    routes = build_qwen27b_cache_routes(
        (SimpleNamespace(is_linear=False),),
        strict_topology=False,
    )
    lane = _lane(routes)
    normalized = normalize_target_cache(lane, [_kv([1, 2, 3])])

    demoted = _demote_selfcheck_request_cache(normalized)
    keys, values = demoted[0].state

    assert type(demoted[0]) is KVCache
    assert demoted[0].offset == 3
    assert list(keys.shape) == [1, 1, 3, 1]
    assert list(values.shape) == [1, 1, 3, 1]


@pytest.mark.parametrize("missing_index", [0, 1])
def test_recurrent_admission_rejects_incomplete_source_state(
    missing_index: int,
) -> None:
    lane = _lane(
        build_qwen27b_cache_routes(
            (SimpleNamespace(is_linear=True),),
            strict_topology=False,
        )
    )
    source = ArraysCache(size=2)
    source.state = [
        mx.ones((1, 2, 3)),
        mx.ones((1, 2, 3, 4)),
    ]
    source.state[missing_index] = None

    with pytest.raises(ValueError, match="layer 0.*both recurrent state leaves"):
        normalize_target_cache(lane, [source])


def test_recurrent_admission_rejects_non_request_batch_and_owns_batch1() -> None:
    lane = _lane(
        build_qwen27b_cache_routes(
            (SimpleNamespace(is_linear=True),),
            strict_topology=False,
        )
    )
    batch_source = ArraysCache(size=2)
    batch_source.state = [
        mx.ones((2, 2, 3)),
        mx.ones((2, 2, 3, 4)),
    ]
    with pytest.raises(ValueError, match="layer 0.*batch size 1"):
        normalize_target_cache(lane, [batch_source])

    source = ArraysCache(size=2)
    source.state = [
        mx.ones((1, 2, 3)),
        mx.ones((1, 2, 3, 4)),
    ]
    normalized = normalize_target_cache(lane, [source])[0]
    source.state[0][:] = 9
    mx.eval(source.state[0], normalized.state[0])

    assert isinstance(normalized, OwnedRecurrentStateCache)
    assert normalized.batch_size == 1
    assert normalized.state[0].tolist() == [
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    ]


@pytest.mark.parametrize(
    "entry",
    [
        RotatingKVCache(max_size=16),
        QuantizedKVCache(group_size=64, bits=4),
        object(),
    ],
)
def test_attention_normalization_rejects_noninstalled_cache_types(entry: object) -> None:
    lane = _lane(
        build_qwen27b_cache_routes(
            (SimpleNamespace(is_linear=False),),
            strict_topology=False,
        )
    )

    with pytest.raises(TypeError, match="layer 0"):
        normalize_target_cache(lane, [entry])


def test_cohort_cache_is_rejected_by_request_session_commit_guard() -> None:
    lane = _lane(
        build_qwen27b_cache_routes(
            (SimpleNamespace(is_linear=False),),
            strict_topology=False,
        )
    )
    cohort_cache = merge_target_caches(
        lane,
        (
            normalize_target_cache(lane, [_kv([1])]),
            normalize_target_cache(lane, [_kv([2])]),
        ),
    )

    with pytest.raises(TypeError, match="cohort cache"):
        assert_request_local_target_cache(lane, cohort_cache)


def test_solo_post_prefill_cache_types_pass_request_session_commit_guard() -> None:
    lane = _lane(
        build_qwen27b_cache_routes(
            (
                SimpleNamespace(is_linear=True),
                SimpleNamespace(is_linear=False),
            ),
            strict_topology=False,
        )
    )
    recurrent = ArraysCache(size=2)
    recurrent.state = _owned(3).state
    attention = VllmMetalPagedKVCache(block_size=4, num_blocks=2)
    attention.state = _kv([7, 8]).state

    assert_request_local_target_cache(lane, [recurrent, attention])


def test_fixed_configured_capture_entrypoint_has_no_dynamic_controls() -> None:
    source = (
        inspect.getsource(forward_with_gdn_capture_configured)
        + inspect.getsource(_forward_with_gdn_capture_impl)
    )

    assert "os.environ" not in source
    assert "lane_disabled" not in source
    assert "resolve_gdn_capture_backend" not in source
    assert ".is_linear" not in source
    assert "getattr(" not in source
    assert ".model" not in source
    assert ".fa_idx" not in source
    assert ".ssm_idx" not in source
    assert "tie_word_embeddings" not in source
    assert "except" not in source
    recurrent_source = inspect.getsource(gdn_forward_with_capture_configured)
    assert "_maybe_contiguous_authoritative_gdn_leaf" not in recurrent_source
    assert "cache is not None" not in recurrent_source
    assert "cache[0] is not None" not in recurrent_source
    assert "cache[1] is not None" not in recurrent_source


def test_qwen_verify_config_is_immutable() -> None:
    config = QwenGDNVerifyConfig.stock(
        capture_backend="linear_gdn_from_conv_tape",
        hidden_variant="post_norm",
    )

    with pytest.raises(AttributeError):
        config.hidden_variant = "pre_norm"
    assert replace(config, layer_eval_every=8).layer_eval_every == 8
    assert config.capture_delta.func is _kernel_tape_capture_configured


def test_authoritative_state_operation_is_resolved_once_from_profile(
    monkeypatch,
) -> None:
    value = mx.arange(4).reshape(1, 1, 4)[:, :, ::2]
    monkeypatch.delenv("MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE", raising=False)
    path, operation = bind_qwen_authoritative_state_operation()
    assert path == "identity"
    assert operation(value) is value

    monkeypatch.setenv("MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE", "1")
    path, operation = bind_qwen_authoritative_state_operation()
    owned = operation(value)
    mx.eval(owned)
    assert path == "contiguous"
    assert owned.tolist() == value.tolist()


def test_configured_forward_does_not_reinspect_model_or_layer_topology() -> None:
    config = QwenGDNVerifyConfig.stock(
        capture_backend="linear_gdn_from_conv_tape",
        hidden_variant="post_norm",
    )

    class PoisonModel:
        @property
        def language_model(self):
            raise AssertionError("configured forward reinspected the model")

    logits, hidden, captures = forward_with_gdn_capture_configured(
        PoisonModel(),
        mx.ones((1, 1, 2)),
        [],
        config=config,
    )
    assert logits.tolist() == [[[1.0, 1.0]]]
    assert hidden.tolist() == logits.tolist()
    assert captures == {}

    class AttentionLayer:
        def __init__(self):
            self.topology_open = True
            self.input_layernorm = lambda value: value
            self.self_attn = (
                lambda value, *, mask, cache: mx.zeros_like(value)
            )
            self.post_attention_layernorm = lambda value: value
            self.mlp = lambda value: mx.zeros_like(value)

        @property
        def is_linear(self):
            if not self.topology_open:
                raise AssertionError("installed layer route read is_linear")
            return False

    layer = AttentionLayer()
    routes = _bind_configured_layer_routes(config, (layer,))
    layer.topology_open = False
    output, capture = routes[0](mx.ones((1, 1, 2)), None, None, None)
    assert output.tolist() == [[[1.0, 1.0]]]
    assert capture is None


def test_configured_recurrent_route_uses_only_prebound_complete_state(
    monkeypatch,
) -> None:
    config = replace(
        QwenGDNVerifyConfig.stock(
            capture_backend="linear_gdn_from_conv_tape",
            hidden_variant="post_norm",
        ),
        project_inputs=lambda _gdn, values: (
            values,
            values[..., :1],
            mx.ones((1, 1, 1)),
            mx.ones((1, 1, 1)),
        ),
        capture_conv=lambda values, _conv_state, _gdn: (
            values,
            values[:, :, None, :],
        ),
        capture_delta=lambda values, _g, _beta, state, _gdn: (
            values[:, :, None, :],
            state + 1,
            mx.ones((1, 1, 1, 1)),
        ),
        apply_gdn_tail=lambda _gdn, values, _gate: values,
        compute_g=lambda _a_log, values, _dt_bias: values,
        authoritative_state_path="identity",
        own_authoritative_state=lambda value: value,
    )

    class Cache(list):
        def advance(self, amount):
            self.advanced = amount

    cache = Cache(
        [
            mx.ones((1, 1, 2)),
            mx.ones((1, 1, 1, 1)),
        ]
    )
    gdn = SimpleNamespace(
        A_log=mx.ones((1,)),
        dt_bias=mx.ones((1,)),
        conv_dim=2,
        head_k_dim=1,
        head_v_dim=1,
        num_k_heads=1,
        num_v_heads=1,
        key_dim=1,
    )
    monkeypatch.setattr(
        "mtplx.gdn_capture.os.environ.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed recurrent route read the environment")
        ),
    )
    monkeypatch.setattr(
        "mtplx.kernel_selfcheck.lane_disabled",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed recurrent route called lane_disabled")
        ),
    )
    monkeypatch.setattr(
        mx,
        "zeros",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed recurrent route synthesized missing state")
        ),
    )

    output, capture = gdn_forward_with_capture_configured(
        gdn,
        mx.ones((1, 1, 2)),
        None,
        cache,
        config=config,
    )

    assert output.shape == (1, 1, 1, 2)
    assert cache[0].shape == (1, 1, 2)
    assert cache[1].tolist() == [[[[2.0]]]]
    assert cache.advanced == 1
    assert capture["state_in"].tolist() == [[[[1.0]]]]
