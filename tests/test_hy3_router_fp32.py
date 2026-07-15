from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

import mtplx.hy3_router_fp32 as hy3_router_module  # noqa: E402

from mtplx.hy3_router_fp32 import (  # noqa: E402
    Hy3RouterFP32Ineligible,
    Hy3RouterFP32Tiling,
    _balanced_splitk_reduction_source,
    hy3_router_fp32_available,
    hy3_router_fp32_eligible,
    hy3_router_fp32_exact_project,
    hy3_router_fp32_exact_route,
    hy3_router_fp32_exact_splitk_project,
    hy3_router_fp32_exact_splitk_route,
    hy3_router_fp32_finalize,
    hy3_router_fp32_project,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_exact_splitk_weight,
    prepare_hy3_router_fp32_exact_weight,
    prepare_hy3_router_fp32_weight,
)


def test_hy3_router_fp32_p32_reduction_is_a_complete_balanced_tree() -> None:
    source = _balanced_splitk_reduction_source(32)

    assert source.count("partials[") == 32
    for part in range(32):
        assert f"partials[{part} * STRIDE + index]" in source
    assert source.count("float reduce_l0_") == 16
    assert source.count("float reduce_l1_") == 8
    assert source.count("float reduce_l2_") == 4
    assert source.count("float reduce_l3_") == 2
    assert "float total = reduce_l3_0 + reduce_l3_1;" in source


@pytest.mark.parametrize("simd_groups", tuple(range(1, 9)))
def test_hy3_router_supported_simd_topologies_reach_availability_gate(
    simd_groups: int,
) -> None:
    logits = mx.zeros((1, 192), dtype=mx.float32)
    expert_bias = mx.zeros((192,), dtype=mx.float32)

    with pytest.raises(Hy3RouterFP32Ineligible, match="outside"):
        hy3_router_fp32_finalize(
            logits,
            expert_bias,
            available=False,
            finalizer_mode="simd",
            simd_groups=simd_groups,
        )


def test_hy3_router_rejects_unsupported_simd_topology_before_dispatch() -> None:
    logits = mx.zeros((1, 192), dtype=mx.float32)
    expert_bias = mx.zeros((192,), dtype=mx.float32)

    with pytest.raises(Hy3RouterFP32Ineligible, match="SIMDgroup count"):
        hy3_router_fp32_finalize(
            logits,
            expert_bias,
            available=False,
            finalizer_mode="simd",
            simd_groups=9,
        )


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
@pytest.mark.parametrize("sigmoid_mode", ("precise", "fast-exp"))
@pytest.mark.parametrize("simd_groups", (1, 2, 3, 4, 5, 7, 8))
def test_hy3_router_candidate_simd_finalize_matches_six_simd_on_g17(
    rows: int,
    sigmoid_mode: str,
    simd_groups: int,
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(519_000 + rows)
    logits = (mx.random.normal((rows, 192)) * 3.0).astype(mx.float32)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    control_ids, control_weights = hy3_router_fp32_finalize(
        logits,
        expert_bias,
        finalizer_mode="simd",
        simd_groups=6,
        sigmoid_mode=sigmoid_mode,
    )
    candidate_ids, candidate_weights = hy3_router_fp32_finalize(
        logits,
        expert_bias,
        finalizer_mode="simd",
        simd_groups=simd_groups,
        sigmoid_mode=sigmoid_mode,
    )
    mx.eval(control_ids, control_weights, candidate_ids, candidate_weights)

    assert bool(mx.array_equal(candidate_ids, control_ids).item())
    tolerance = 1e-7 if sigmoid_mode == "precise" else 1e-5
    assert (
        float(mx.max(mx.abs(candidate_weights - control_weights)).item()) <= tolerance
    )


@pytest.mark.parametrize("rows", (1, 8))
@pytest.mark.parametrize("simd_groups", (1, 2, 3, 4, 5, 7, 8))
def test_hy3_router_candidate_simd_mpp_route_matches_six_simd_on_g17(
    rows: int,
    simd_groups: int,
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(520_000 + rows)
    value = mx.random.normal((rows, 4096)).astype(mx.float32)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    prepared_weight = prepare_hy3_router_fp32_weight(source_weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    control_ids, control_weights = hy3_router_fp32_route(
        value,
        prepared_weight,
        expert_bias,
        n_tile=16,
        grid_k_parts=8,
        finalizer_mode="simd",
        simd_groups=6,
    )
    candidate_ids, candidate_weights = hy3_router_fp32_route(
        value,
        prepared_weight,
        expert_bias,
        n_tile=16,
        grid_k_parts=8,
        finalizer_mode="simd",
        simd_groups=simd_groups,
    )
    mx.eval(control_ids, control_weights, candidate_ids, candidate_weights)

    assert bool(mx.array_equal(candidate_ids, control_ids).item())
    assert float(mx.max(mx.abs(candidate_weights - control_weights)).item()) <= 1e-7


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_hy3_router_fp32_finalize_matches_stock_logits_on_g17(rows: int) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(515_000 + rows)
    logits = mx.random.normal((rows, 192)).astype(mx.float32)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    scores = mx.sigmoid(logits)
    selection_scores = scores + expert_bias
    reference_ids = mx.argpartition(selection_scores, kth=-8, axis=-1)[..., -8:]
    reference_weights = mx.take_along_axis(scores, reference_ids, axis=-1)
    reference_weights = reference_weights / (
        reference_weights.sum(axis=-1, keepdims=True) + 1e-20
    )
    reference_weights = reference_weights * 2.826

    observed_ids, observed_weights = hy3_router_fp32_finalize(
        logits,
        expert_bias,
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode="simd",
    )
    mx.eval(reference_ids, reference_weights, observed_ids, observed_weights)

    assert bool(mx.array_equal(observed_ids, reference_ids).item())
    assert float(mx.max(mx.abs(observed_weights - reference_weights)).item()) <= 1e-7


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_hy3_router_fp32_fast_exp_sigmoid_stays_close_on_g17(rows: int) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(516_000 + rows)
    logits = (mx.random.normal((rows, 192)) * 3.0).astype(mx.float32)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    exact_ids, exact_weights = hy3_router_fp32_finalize(
        logits,
        expert_bias,
        finalizer_mode="simd",
        sigmoid_mode="precise",
    )
    fast_ids, fast_weights = hy3_router_fp32_finalize(
        logits,
        expert_bias,
        finalizer_mode="simd",
        sigmoid_mode="fast-exp",
    )
    mx.eval(exact_ids, exact_weights, fast_ids, fast_weights)

    assert bool(mx.array_equal(fast_ids, exact_ids).item())
    assert float(mx.max(mx.abs(fast_weights - exact_weights)).item()) <= 1e-5


def test_hy3_router_fp32_tiling_accounts_for_grid_work() -> None:
    tiling = Hy3RouterFP32Tiling(
        n_tile=16,
        grid_k_parts=4,
        operand_mode="direct",
    )

    assert tiling.stage1_threadgroups == 48
    assert tiling.k_span == 1024
    assert tiling.partial_bytes == 24 * 1024
    assert tiling.staged_threadgroup_bytes == 0

    reuse_frontier = Hy3RouterFP32Tiling(
        n_tile=64,
        grid_k_parts=16,
        operand_mode="direct",
    )
    assert reuse_frontier.stage1_threadgroups == 48
    assert reuse_frontier.k_span == 256
    assert reuse_frontier.partial_bytes == 96 * 1024

    narrow_frontier = Hy3RouterFP32Tiling(
        n_tile=16,
        grid_k_parts=2,
        operand_mode="direct",
    )
    assert narrow_frontier.stage1_threadgroups == 24
    assert narrow_frontier.k_span == 2048
    assert narrow_frontier.partial_bytes == 12 * 1024

    k3_p32_frontier = Hy3RouterFP32Tiling(
        n_tile=32,
        grid_k_parts=32,
        operand_mode="direct",
    )
    assert k3_p32_frontier.stage1_threadgroups == 192
    assert k3_p32_frontier.k_span == 128
    assert k3_p32_frontier.partial_bytes == 192 * 1024

    adjacent_p32_frontier = Hy3RouterFP32Tiling(
        n_tile=64,
        grid_k_parts=32,
        operand_mode="direct",
    )
    assert adjacent_p32_frontier.stage1_threadgroups == 96
    assert adjacent_p32_frontier.k_span == 128


def test_grouped_staged_tiling_reports_shared_activation_reuse() -> None:
    tiling = Hy3RouterFP32Tiling(
        n_tile=16,
        grid_k_parts=8,
        operand_mode="grouped-staged",
        simd_groups_per_threadgroup=4,
    )

    assert tiling.total_simdgroups == 96
    assert tiling.stage1_threadgroups == 24
    assert tiling.staged_threadgroup_bytes == 16 * 1024
    assert tiling.modeled_activation_bytes == 384 * 1024
    assert tiling.modeled_weight_bytes == 192 * 4096 * 2
    assert tiling.partial_bytes == 48 * 1024


def test_grouped_direct_tiling_attributes_scheduling_without_reuse() -> None:
    tiling = Hy3RouterFP32Tiling(
        n_tile=16,
        grid_k_parts=8,
        operand_mode="grouped-direct",
        simd_groups_per_threadgroup=4,
    )

    assert tiling.total_simdgroups == 96
    assert tiling.stage1_threadgroups == 24
    assert tiling.staged_threadgroup_bytes == 0
    assert tiling.modeled_activation_bytes == 1536 * 1024
    assert tiling.modeled_weight_bytes == 192 * 4096 * 2


def test_grouped_direct_source_maps_each_simdgroup_to_one_n_tile() -> None:
    source = hy3_router_module._grouped_partial_source(
        Hy3RouterFP32Tiling(
            n_tile=16,
            grid_k_parts=8,
            operand_mode="grouped-direct",
            simd_groups_per_threadgroup=4,
        )
    )

    assert "part = int(tg) / GROUPS_PER_PART" in source
    assert "n_tile_index = group_in_part * SGPTG + int(sg_id)" in source
    assert "threadgroup float A_tile" not in source


def test_grouped_staged_source_loads_one_shared_activation_slice() -> None:
    source = hy3_router_module._grouped_partial_source(
        Hy3RouterFP32Tiling(
            n_tile=16,
            grid_k_parts=8,
            operand_mode="grouped-staged",
            simd_groups_per_threadgroup=4,
        )
    )

    assert "threadgroup float A_tile[BM * KS]" in source
    assert "offset += SGPTG * 32" in source
    assert source.count("A_tile[offset] =") == 1
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in source
    assert "tensor<threadgroup float" in source
    assert "tensor<device bfloat" in source


def test_grouped_direct_dispatch_uses_four_simdgroups_per_threadgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[int, int, int]] = {}

    def fake_partial_builder(*args: object, **kwargs: object):
        def fake_partial_kernel(**dispatch: object):
            calls["grid"] = dispatch["grid"]  # type: ignore[assignment]
            calls["threadgroup"] = dispatch["threadgroup"]  # type: ignore[assignment]
            output_shape = dispatch["output_shapes"][0]  # type: ignore[index]
            return (mx.zeros(output_shape, dtype=mx.float32),)

        return fake_partial_kernel

    def fake_reduce_builder(*args: object, **kwargs: object):
        def fake_reduce_kernel(**dispatch: object):
            return (mx.zeros((8, 192), dtype=mx.float32),)

        return fake_reduce_kernel

    monkeypatch.setattr(
        hy3_router_module,
        "_build_hy3_router_fp32_grouped_partial_kernel",
        fake_partial_builder,
        raising=False,
    )
    monkeypatch.setattr(
        hy3_router_module,
        "_build_hy3_router_fp32_reduce_kernel",
        fake_reduce_builder,
    )
    value = mx.zeros((4, 4096), dtype=mx.float32)
    weight = mx.zeros((4096, 192), dtype=mx.bfloat16)

    hy3_router_fp32_project(
        value,
        weight,
        available=True,
        n_tile=16,
        grid_k_parts=8,
        operand_mode="grouped-direct",
        simd_groups_per_threadgroup=4,
    )

    assert calls["threadgroup"] == (128, 1, 1)
    assert calls["grid"] == (24 * 128, 1, 1)


@pytest.mark.parametrize("groups", (5, 7, 8))
def test_grouped_tiling_requires_groups_to_divide_n_tiles(groups: int) -> None:
    with pytest.raises(ValueError, match="divide N tiles"):
        Hy3RouterFP32Tiling(
            n_tile=16,
            grid_k_parts=8,
            operand_mode="grouped-direct",
            simd_groups_per_threadgroup=groups,
        )


@pytest.mark.parametrize("grid_k_parts", (1, 2, 4))
def test_grouped_staged_tiling_requires_short_k_spans(grid_k_parts: int) -> None:
    with pytest.raises(ValueError, match="P8, P16, or P32"):
        Hy3RouterFP32Tiling(
            n_tile=16,
            grid_k_parts=grid_k_parts,
            operand_mode="grouped-staged",
            simd_groups_per_threadgroup=4,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"n_tile": 8, "grid_k_parts": 4, "operand_mode": "direct"},
        {"n_tile": 16, "grid_k_parts": 3, "operand_mode": "direct"},
        {"n_tile": 32, "grid_k_parts": 64, "operand_mode": "direct"},
        {"n_tile": 16, "grid_k_parts": 4, "operand_mode": "staged"},
        {
            "n_tile": 16,
            "grid_k_parts": 4,
            "operand_mode": "direct",
            "k_tile": 32,
        },
    ),
)
def test_hy3_router_fp32_tiling_rejects_invalid_contracts(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        Hy3RouterFP32Tiling(**kwargs)


def test_hy3_router_fp32_eligibility_matches_next_k_shapes() -> None:
    assert hy3_router_fp32_eligible(
        rows=1,
        input_width=4096,
        experts=192,
        input_dtype=mx.float32,
        weight_dtype=mx.bfloat16,
        available=True,
    )
    assert hy3_router_fp32_eligible(
        rows=8,
        input_width=4096,
        experts=192,
        input_dtype=mx.float32,
        weight_dtype=mx.bfloat16,
        available=True,
    )
    assert not hy3_router_fp32_eligible(
        rows=9,
        input_width=4096,
        experts=192,
        input_dtype=mx.float32,
        weight_dtype=mx.bfloat16,
        available=True,
    )
    assert not hy3_router_fp32_eligible(
        rows=1,
        input_width=4096,
        experts=192,
        input_dtype=mx.bfloat16,
        weight_dtype=mx.bfloat16,
        available=True,
    )
    assert not hy3_router_fp32_eligible(
        rows=1,
        input_width=4096,
        experts=192,
        input_dtype=mx.float32,
        weight_dtype=mx.float32,
        available=True,
    )
    assert not hy3_router_fp32_eligible(
        rows=1,
        input_width=4096,
        experts=192,
        input_dtype=mx.float32,
        weight_dtype=mx.bfloat16,
        available=False,
    )


def test_prepare_hy3_router_fp32_exact_weight_matches_stock_promotion() -> None:
    source = mx.zeros((192, 4096), dtype=mx.bfloat16)

    prepared = prepare_hy3_router_fp32_exact_weight(source)

    assert tuple(prepared.shape) == (192, 4096)
    assert prepared.dtype == mx.float32


def test_prepare_hy3_router_fp32_exact_splitk_weight_is_k_major_fp32() -> None:
    source = mx.zeros((192, 4096), dtype=mx.bfloat16)

    prepared = prepare_hy3_router_fp32_exact_splitk_weight(source)

    assert tuple(prepared.shape) == (4096, 192)
    assert prepared.dtype == mx.float32


def test_hy3_router_fp32_exact_splitk_reaches_availability_gate() -> None:
    value = mx.zeros((4, 4096), dtype=mx.float32)
    weight = mx.zeros((4096, 192), dtype=mx.float32)
    expert_bias = mx.zeros((192,), dtype=mx.float32)

    with pytest.raises(Hy3RouterFP32Ineligible, match="outside"):
        hy3_router_fp32_exact_splitk_route(
            value,
            weight,
            expert_bias,
            available=False,
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="outside"):
        hy3_router_fp32_exact_splitk_project(
            value,
            weight,
            available=False,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"top_k": 4},
        {"route_norm": False},
        {"scaling_factor": 0.0},
        {"scaling_factor": float("inf")},
        {"finalizer_mode": "unsupported"},
    ),
)
def test_hy3_router_fp32_route_rejects_non_exact_finalizer_contract(
    overrides: dict[str, object],
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    value = mx.zeros((1, 4096), dtype=mx.float32)
    weight = mx.zeros((4096, 192), dtype=mx.bfloat16)
    expert_bias = mx.zeros((192,), dtype=mx.float32)

    with pytest.raises(ValueError):
        hy3_router_fp32_route(
            value,
            weight,
            expert_bias,
            available=True,
            **overrides,
        )


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
@pytest.mark.parametrize(
    ("n_tile", "grid_k_parts"),
    (
        (16, 1),
        (32, 4),
        (16, 4),
        (32, 8),
        (16, 8),
        (64, 8),
        (64, 16),
        (32, 16),
    ),
)
@pytest.mark.parametrize("finalizer_mode", ("serial", "simd"))
def test_hy3_router_fp32_project_matches_stock_ids_on_g17(
    rows: int,
    n_tile: int,
    grid_k_parts: int,
    finalizer_mode: str,
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(5100 + rows)
    value = (mx.random.normal((rows, 4096)) * 0.5).astype(mx.float32)
    weight = (mx.random.normal((192, 4096)) * 0.02).astype(mx.bfloat16)
    prepared_weight = prepare_hy3_router_fp32_weight(weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)

    observed = hy3_router_fp32_project(
        value,
        prepared_weight,
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
        operand_mode="direct",
    )
    observed_indices, observed_weights = hy3_router_fp32_route(
        value,
        prepared_weight,
        expert_bias,
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
        operand_mode="direct",
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode=finalizer_mode,
    )
    reference = value @ weight.astype(mx.float32).T
    reference_scores = mx.sigmoid(reference)
    reference_selection = reference_scores + expert_bias
    reference_indices = mx.argpartition(
        reference_selection,
        kth=-8,
        axis=-1,
    )[..., -8:]
    reference_weights = mx.take_along_axis(
        reference_scores,
        reference_indices,
        axis=-1,
    )
    reference_weights = reference_weights / (
        reference_weights.sum(axis=-1, keepdims=True) + 1e-20
    )
    reference_weights = reference_weights * 2.826
    mx.eval(
        observed,
        reference,
        observed_indices,
        observed_weights,
        reference_indices,
        reference_weights,
    )
    error = mx.abs(observed - reference)
    route_weight_error = mx.abs(observed_weights - reference_weights)
    mx.eval(error, route_weight_error)

    assert tuple(observed.shape) == (rows, 192)
    assert observed.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(observed)).item())
    assert float(mx.max(error).item()) <= 0.01
    assert bool(mx.array_equal(observed_indices, reference_indices).item())
    assert tuple(observed_indices.shape) == (rows, 8)
    assert tuple(observed_weights.shape) == (rows, 8)
    assert observed_weights.dtype == mx.float32
    assert float(mx.max(route_weight_error).item()) <= 5e-4


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_hy3_router_fp32_exact_path_matches_stock_on_g17(rows: int) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(515_100 + rows)
    value = mx.random.normal((rows, 4096)).astype(mx.float32)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    prepared_weight = prepare_hy3_router_fp32_exact_weight(source_weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)

    observed_logits = hy3_router_fp32_exact_project(
        value,
        prepared_weight,
    )
    observed_ids, observed_weights = hy3_router_fp32_exact_route(
        value,
        prepared_weight,
        expert_bias,
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode="simd",
    )
    reference_logits = value @ source_weight.astype(mx.float32).T
    reference_scores = mx.sigmoid(reference_logits)
    selection_scores = reference_scores + expert_bias
    reference_ids = mx.argpartition(selection_scores, kth=-8, axis=-1)[..., -8:]
    reference_weights = mx.take_along_axis(
        reference_scores,
        reference_ids,
        axis=-1,
    )
    reference_weights = reference_weights / (
        reference_weights.sum(axis=-1, keepdims=True) + 1e-20
    )
    reference_weights = reference_weights * 2.826
    mx.eval(
        observed_logits,
        observed_ids,
        observed_weights,
        reference_logits,
        reference_ids,
        reference_weights,
    )

    assert bool(mx.array_equal(observed_logits, reference_logits).item())
    assert bool(mx.array_equal(observed_ids, reference_ids).item())
    assert float(mx.max(mx.abs(observed_weights - reference_weights)).item()) <= 1e-7


@pytest.mark.parametrize("rows", (1, 4, 8))
@pytest.mark.parametrize(
    ("n_tile", "grid_k_parts"),
    ((16, 8), (32, 32), (64, 32)),
)
def test_hy3_router_fp32_exact_splitk_matches_exact_routes_on_g17(
    rows: int,
    n_tile: int,
    grid_k_parts: int,
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    mx.random.seed(515_200 + rows)
    value = mx.random.normal((rows, 4096)).astype(mx.float32)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    exact_weight = prepare_hy3_router_fp32_exact_weight(source_weight)
    splitk_weight = prepare_hy3_router_fp32_exact_splitk_weight(source_weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)

    reference_logits = hy3_router_fp32_exact_project(value, exact_weight)
    candidate_logits = hy3_router_fp32_exact_splitk_project(
        value,
        splitk_weight,
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
    )
    reference_ids, reference_weights = hy3_router_fp32_exact_route(
        value,
        exact_weight,
        expert_bias,
    )
    candidate_ids, candidate_weights = hy3_router_fp32_exact_splitk_route(
        value,
        splitk_weight,
        expert_bias,
        n_tile=n_tile,
        grid_k_parts=grid_k_parts,
    )
    mx.eval(
        reference_logits,
        candidate_logits,
        reference_ids,
        reference_weights,
        candidate_ids,
        candidate_weights,
    )

    logit_difference = candidate_logits - reference_logits
    logit_nrmse = mx.sqrt(mx.mean(mx.square(logit_difference))) / mx.sqrt(
        mx.mean(mx.square(reference_logits))
    )
    assert float(mx.max(mx.abs(logit_difference)).item()) <= 0.125
    assert float(logit_nrmse.item()) <= 5e-4
    assert bool(mx.array_equal(candidate_ids, reference_ids).item())
    assert float(mx.max(mx.abs(candidate_weights - reference_weights)).item()) <= 5e-4


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
@pytest.mark.parametrize("finalizer_mode", ("serial", "simd"))
def test_hy3_router_fp32_route_matches_adversarial_top8_on_g17(
    rows: int,
    finalizer_mode: str,
) -> None:
    if not hy3_router_fp32_available():
        pytest.skip("router tensor-op execution requires Apple G17 and macOS 26.2+")

    value = mx.zeros((rows, 4096), dtype=mx.float32)
    weight = mx.zeros((192, 4096), dtype=mx.bfloat16)
    prepared_weight = prepare_hy3_router_fp32_weight(weight)
    selected = {3, 17, 29, 61, 97, 133, 171, 190}
    correction_bias = mx.array(
        [0.25 if index in selected else 0.0 for index in range(192)],
        dtype=mx.float32,
    )
    bias_fixtures = (
        mx.zeros((192,), dtype=mx.float32),
        mx.arange(192, dtype=mx.float32) * 1e-7,
        correction_bias,
    )

    reference_logits = value @ weight.astype(mx.float32).T
    reference_scores = mx.sigmoid(reference_logits)
    for expert_bias in bias_fixtures:
        observed_indices, observed_weights = hy3_router_fp32_route(
            value,
            prepared_weight,
            expert_bias,
            n_tile=16,
            grid_k_parts=4,
            operand_mode="direct",
            top_k=8,
            route_norm=True,
            scaling_factor=2.826,
            finalizer_mode=finalizer_mode,
        )
        reference_selection = reference_scores + expert_bias
        reference_indices = mx.argpartition(
            reference_selection,
            kth=-8,
            axis=-1,
        )[..., -8:]
        reference_weights = mx.take_along_axis(
            reference_scores,
            reference_indices,
            axis=-1,
        )
        reference_weights = reference_weights / (
            reference_weights.sum(axis=-1, keepdims=True) + 1e-20
        )
        reference_weights = reference_weights * 2.826
        mx.eval(
            observed_indices,
            observed_weights,
            reference_indices,
            reference_weights,
        )

        assert bool(mx.array_equal(observed_indices, reference_indices).item())
        assert float(mx.max(mx.abs(observed_weights - reference_weights)).item()) <= (
            5e-4
        )
