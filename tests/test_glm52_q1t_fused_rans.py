from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pytest

from mtplx import expert_rans
from mtplx.expert_shadow import encode_t158


_OUTPUT_TILE = expert_rans.LANES


@dataclass(frozen=True)
class _EncodedComponent:
    payload: np.ndarray
    directory: np.ndarray
    cum2sym: np.ndarray
    freq: np.ndarray
    cum: np.ndarray
    row_bytes: int


def _encode_lane_owned_component(
    values: np.ndarray,
    *,
    verify_reference: bool = True,
    uniform_model: bool = False,
) -> _EncodedComponent:
    """Encode [expert, output, byte] with one output row per rANS lane."""

    experts, outputs, row_bytes = values.shape
    assert outputs % _OUTPUT_TILE == 0
    tiles = outputs // _OUTPUT_TILE
    segments = values.reshape(
        experts * tiles,
        _OUTPUT_TILE * row_bytes,
    )
    from mtplx.glm52_q1t_rans_artifact import _encode_bank_bounded

    table = (
        expert_rans.table_from_freq(np.full(256, 16, dtype=np.uint32))
        if uniform_model
        else expert_rans.build_table(expert_rans.histogram(segments.reshape(-1)))
    )
    streams = _encode_bank_bounded(segments, table)
    assert streams.lanes == _OUTPUT_TILE
    assert streams.per_lane == row_bytes
    if verify_reference:
        decoded = expert_rans.decode_bank_reference(streams, table)
        assert np.array_equal(decoded.reshape(values.shape), values)
    return _EncodedComponent(
        payload=np.ascontiguousarray(streams.payload),
        directory=np.ascontiguousarray(streams.directory.reshape(-1)),
        cum2sym=np.ascontiguousarray(table.cum2sym),
        freq=np.ascontiguousarray(table.freq, dtype=np.uint32),
        cum=np.ascontiguousarray(table.cum[:256], dtype=np.uint32),
        row_bytes=row_bytes,
    )


def _t158_bank(
    *, experts: int, out_dim: int, in_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(51032)
    weights = rng.standard_normal((experts, out_dim, in_dim), dtype=np.float32)
    packed_rows = []
    scale_rows = []
    for expert in range(experts):
        packed, scales = encode_t158(weights[expert])
        packed_rows.append(packed)
        scale_rows.append(scales)
    return np.stack(packed_rows), np.stack(scale_rows)


def _random_t158_bank(
    *,
    experts: int,
    out_dim: int,
    in_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = in_dim // 64
    packed = rng.integers(
        0,
        243,
        size=(experts, out_dim, groups * 13),
        dtype=np.uint8,
    )
    scale_values = rng.uniform(
        0.01,
        0.5,
        size=(experts, out_dim, groups),
    ).astype(np.float32)
    scales = (scale_values.view(np.uint32) >> 16).astype(np.uint16)
    return packed, scales


def _bind_projection(
    mx,
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    in_dim: int,
    dtype,
    threads_per_tg: int = 32,
    verify_reference: bool = True,
    uniform_packed_rans: bool = False,
    input_repeat: int = 1,
):
    from mtplx.kernels.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_projection,
    )

    experts, out_dim, _packed_bytes = packed.shape
    packed_component = _encode_lane_owned_component(
        packed,
        verify_reference=verify_reference,
        uniform_model=uniform_packed_rans,
    )
    scales_component = _encode_lane_owned_component(
        scales.view(np.uint8).reshape(experts, out_dim, -1),
        verify_reference=verify_reference,
    )
    return bind_glm52_q1t_fused_rans_projection(
        packed_payload=mx.array(packed_component.payload),
        packed_directory=mx.array(packed_component.directory),
        packed_cum2sym=mx.array(packed_component.cum2sym),
        packed_freq=mx.array(packed_component.freq),
        packed_cum=mx.array(packed_component.cum),
        scales_payload=mx.array(scales_component.payload),
        scales_directory=mx.array(scales_component.directory),
        scales_cum2sym=mx.array(scales_component.cum2sym),
        scales_freq=mx.array(scales_component.freq),
        scales_cum=mx.array(scales_component.cum),
        expert_count=experts,
        in_dim=in_dim,
        out_dim=out_dim,
        output_tile=_OUTPUT_TILE,
        threads_per_tg=threads_per_tg,
        dtype=dtype,
        uniform_packed_rans=uniform_packed_rans,
        input_repeat=input_repeat,
    )


def _slice_encoded_expert(component: _EncodedComponent, expert: int, outputs: int):
    directory = component.directory.reshape(-1, outputs)
    start = int(directory[expert, 0])
    if expert + 1 < directory.shape[0]:
        stop = int(directory[expert + 1, 0])
    else:
        stop = int(component.payload.size)
    return (
        np.ascontiguousarray(component.payload[start:stop]),
        np.ascontiguousarray(directory[expert] - np.uint32(start)),
    )


def _cached_expert_image(components: tuple[_EncodedComponent, ...]) -> np.ndarray:
    """Pack six compressed directory/payload pairs into one cache slot."""

    header = np.zeros(18, dtype=np.uint32)
    parts: list[np.ndarray] = [header.view(np.uint8)]
    cursor = int(header.nbytes)
    for index, component in enumerate(components):
        directory = np.ascontiguousarray(component.directory, dtype=np.uint32)
        payload = np.ascontiguousarray(component.payload, dtype=np.uint8)
        aligned = (cursor + 3) & ~3
        if aligned != cursor:
            parts.append(np.zeros(aligned - cursor, dtype=np.uint8))
            cursor = aligned
        header[index * 3] = cursor
        parts.append(directory.view(np.uint8))
        cursor += int(directory.nbytes)
        header[index * 3 + 1] = cursor
        header[index * 3 + 2] = 0
        parts.append(payload)
        cursor += int(payload.nbytes)
    return np.concatenate(parts)


def _cached_expert_images(
    components: tuple[_EncodedComponent, ...],
    *,
    experts: int,
    outputs: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Build fixed-slot images while preserving each component's shared table."""

    images = []
    for expert in range(experts):
        sliced = []
        for component, output_count in zip(components, outputs, strict=True):
            payload, directory = _slice_encoded_expert(
                component,
                expert,
                output_count,
            )
            sliced.append(
                _EncodedComponent(
                    payload=payload,
                    directory=directory,
                    cum2sym=component.cum2sym,
                    freq=component.freq,
                    cum=component.cum,
                    row_bytes=component.row_bytes,
                )
            )
        images.append(_cached_expert_image(tuple(sliced)))
    return tuple(images)


def _scale_transition(mx, component: _EncodedComponent):
    from mtplx.models.glm52_q1t_fused_rans import _pack_rans_transition_table

    return mx.array(
        _pack_rans_transition_table(
            component.cum2sym,
            component.freq,
            component.cum,
        ),
        dtype=mx.uint32,
    )


def _bind_expert_projection(
    mx,
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    expert: int,
    in_dim: int,
    dtype,
    input_repeat: int = 1,
):
    from mtplx.kernels.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_projection,
    )

    _experts, out_dim, _packed_bytes = packed.shape
    packed_component = _encode_lane_owned_component(
        packed,
        uniform_model=True,
    )
    scales_component = _encode_lane_owned_component(
        scales.view(np.uint8).reshape(packed.shape[0], out_dim, -1),
    )
    packed_payload, packed_directory = _slice_encoded_expert(
        packed_component,
        expert,
        out_dim,
    )
    scales_payload, scales_directory = _slice_encoded_expert(
        scales_component,
        expert,
        out_dim,
    )
    return bind_glm52_q1t_fused_rans_projection(
        packed_payload=mx.array(packed_payload),
        packed_directory=mx.array(packed_directory),
        packed_cum2sym=mx.array(packed_component.cum2sym),
        packed_freq=mx.array(packed_component.freq),
        packed_cum=mx.array(packed_component.cum),
        scales_payload=mx.array(scales_payload),
        scales_directory=mx.array(scales_directory),
        scales_cum2sym=mx.array(scales_component.cum2sym),
        scales_freq=mx.array(scales_component.freq),
        scales_cum=mx.array(scales_component.cum),
        expert_count=1,
        expert_base=expert,
        in_dim=in_dim,
        out_dim=out_dim,
        output_tile=_OUTPUT_TILE,
        threads_per_tg=32,
        dtype=dtype,
        uniform_packed_rans=True,
        input_repeat=input_repeat,
    )


def test_fused_launch_geometry_is_explicit_and_not_inherited_from_rans_lanes() -> None:
    from mtplx.kernels import glm52_q1t_fused_rans as fused

    parameter = inspect.signature(
        fused.bind_glm52_q1t_fused_rans_projection
    ).parameters["threads_per_tg"]
    assert parameter.default is inspect.Parameter.empty
    uniform_parameter = inspect.signature(
        fused.bind_glm52_q1t_fused_rans_projection
    ).parameters["uniform_packed_rans"]
    assert uniform_parameter.default is False
    call_source = inspect.getsource(fused.BoundGlm52Q1TFusedRansProjection.__call__)
    assert "threadgroup=(self.threads_per_tg, 1, 1)" in call_source
    for threads_per_tg in (32, 64, 128, 256, 512, 1024):
        source = fused._source(128, 256, threads_per_tg)
        assert "uint lane = output % 32u" in source


def test_fused_decoder_uses_immutable_device_cached_tables() -> None:
    from mtplx.kernels import glm52_q1t_fused_rans as fused

    source = fused._source(128, 256, 256)
    assert "packed_cum2sym_tg" not in source
    assert "threadgroup_barrier" not in source
    assert "packed_payload, packed_cum2sym, packed_freq, packed_cum" in source


def test_uniform_packed_source_has_one_fixed_refill_per_symbol() -> None:
    from mtplx.kernels import glm52_q1t_fused_rans as fused

    source = fused._source(128, 256, 64, uniform_packed_rans=True)

    assert "rans_next_uniform_packed" in source
    assert "packed_cum2sym" not in source
    assert "while (state <" not in source.split("rans_next_uniform_packed", 1)[1]


def test_uniform_packed_rans_projection_is_bitwise() -> None:
    import mlx.core as mx

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    packed, scales = _random_t158_bank(
        experts=3,
        out_dim=128,
        in_dim=256,
        seed=51908,
    )
    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=256,
        dtype=mx.bfloat16,
        uniform_packed_rans=True,
    )
    x = mx.arange(6 * 256, dtype=mx.float32).reshape(6, 256).astype(mx.bfloat16)
    expert_ids = mx.array([2, 0, 2, 1, 0, 2], dtype=mx.int32)

    result = fused(x, expert_ids)
    reference = shadow_gather_mm(
        x,
        expert_ids,
        mx.array(packed),
        mx.array(scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert mx.array_equal(result, reference).item()


@pytest.mark.parametrize("threads_per_tg", (32, 64, 128, 256, 512, 1024))
def test_fused_rans_projection_candidate_threadgroups_are_bitwise(
    threads_per_tg: int,
) -> None:
    import mlx.core as mx

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    packed, scales = _random_t158_bank(
        experts=2,
        out_dim=1024,
        in_dim=128,
        seed=51100 + threads_per_tg,
    )
    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=128,
        dtype=mx.bfloat16,
        threads_per_tg=threads_per_tg,
    )
    rng = np.random.default_rng(51200 + threads_per_tg)
    x = mx.array(rng.standard_normal((4, 128), dtype=np.float32)).astype(mx.bfloat16)
    ids = mx.array([1, 0, 1, 0], dtype=mx.int32)

    result = fused(x, ids)
    reference = shadow_gather_mm(
        x,
        ids,
        mx.array(packed),
        mx.array(scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert mx.array_equal(result, reference).item()


def test_fused_rans_projection_parity_repeated_arbitrary_experts() -> None:
    import mlx.core as mx

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    experts = 3
    in_dim = 128
    out_dim = 32
    packed, scales = _t158_bank(
        experts=experts,
        out_dim=out_dim,
        in_dim=in_dim,
    )
    rng = np.random.default_rng(51033)
    x = mx.array(rng.standard_normal((4, in_dim), dtype=np.float32))
    expert_ids = mx.array([2, 0, 2, 1], dtype=mx.int32)

    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=in_dim,
        dtype=mx.float32,
    )
    result = fused(x, expert_ids)
    reference = shadow_gather_mm(
        x,
        expert_ids,
        mx.array(packed),
        mx.array(scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert tuple(result.shape) == (4, out_dim)
    assert mx.array_equal(result, reference).item()


def test_fused_rans_projection_accepts_router_uint32_ids_without_cast() -> None:
    import mlx.core as mx

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    packed, scales = _t158_bank(experts=3, out_dim=32, in_dim=128)
    rng = np.random.default_rng(51034)
    x = mx.array(rng.standard_normal((4, 128), dtype=np.float32))
    expert_ids = mx.array([2, 0, 2, 1], dtype=mx.uint32)
    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=128,
        dtype=mx.float32,
    )

    result = fused(x, expert_ids)
    reference = shadow_gather_mm(
        x,
        expert_ids,
        mx.array(packed),
        mx.array(scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert mx.array_equal(result, reference).item()


@pytest.mark.parametrize("assignments", (8, 16, 24, 32))
def test_fused_rans_projection_bitwise_for_glm_assignment_counts(
    assignments: int,
) -> None:
    import mlx.core as mx

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    in_dim = 128
    out_dim = 32
    packed, scales = _random_t158_bank(
        experts=3,
        out_dim=out_dim,
        in_dim=in_dim,
        seed=52000 + assignments,
    )
    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=in_dim,
        dtype=mx.bfloat16,
    )
    rng = np.random.default_rng(52100 + assignments)
    x = mx.array(rng.standard_normal((assignments, in_dim), dtype=np.float32)).astype(
        mx.bfloat16
    )
    ids_np = np.resize(np.array([2, 0, 2, 1], dtype=np.int32), assignments)
    expert_ids = mx.array(ids_np)

    result = fused(x, expert_ids)
    reference = shadow_gather_mm(
        x,
        expert_ids,
        mx.array(packed),
        mx.array(scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert mx.array_equal(result, reference).item()


@pytest.mark.parametrize(
    ("in_dim", "out_dim", "assignments"),
    (
        (6144, 2048, 8),
        (2048, 6144, 32),
    ),
)
def test_fused_rans_projection_bitwise_at_real_glm_geometry(
    in_dim: int,
    out_dim: int,
    assignments: int,
) -> None:
    import mlx.core as mx

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    packed, scales = _random_t158_bank(
        experts=2,
        out_dim=out_dim,
        in_dim=in_dim,
        seed=in_dim + out_dim,
    )
    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=in_dim,
        dtype=mx.bfloat16,
        verify_reference=False,
    )
    rng = np.random.default_rng(in_dim * 10 + out_dim)
    x = mx.array(rng.standard_normal((assignments, in_dim), dtype=np.float32)).astype(
        mx.bfloat16
    )
    expert_ids = mx.array(np.resize(np.array([1, 0, 1], dtype=np.int32), assignments))

    result = fused(x, expert_ids)
    reference = shadow_gather_mm(
        x,
        expert_ids,
        mx.array(packed),
        mx.array(scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert mx.array_equal(result, reference).item()


def test_fused_projection_never_calls_standalone_rans_decoder(monkeypatch) -> None:
    import mlx.core as mx

    from mtplx import expert_rans_metal

    def forbidden(*_args, **_kwargs):
        raise AssertionError("standalone rANS decode must not be called")

    monkeypatch.setattr(expert_rans_metal, "decode_component", forbidden)
    monkeypatch.setattr(expert_rans_metal, "decode_container", forbidden)
    packed, scales = _random_t158_bank(
        experts=2,
        out_dim=32,
        in_dim=128,
        seed=53000,
    )
    fused = _bind_projection(
        mx,
        packed,
        scales,
        in_dim=128,
        dtype=mx.float32,
    )
    result = fused(
        mx.ones((2, 128), dtype=mx.float32),
        mx.array([1, 0], dtype=mx.int32),
    )
    mx.eval(result)

    assert tuple(result.shape) == (2, 32)


def test_cached_rans_bank_dispatch_routes_repeated_expert_ids_bitwise(
    monkeypatch,
) -> None:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import swiglu

    from mtplx import expert_rans_metal
    from mtplx.kernels.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_cached_bank,
    )
    from mtplx.kernels.shadow_gather import shadow_gather_mm

    for name in ("decode_component", "decode_container"):
        monkeypatch.setattr(
            expert_rans_metal,
            name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cached bank dispatch must decode inside matmul")
            ),
        )

    experts = 3
    hidden_size = 128
    expert_hidden_size = 128
    gate_packed, gate_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56401,
    )
    up_packed, up_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56402,
    )
    down_packed, down_scales = _random_t158_bank(
        experts=experts,
        out_dim=hidden_size,
        in_dim=expert_hidden_size,
        seed=56403,
    )
    components = (
        _encode_lane_owned_component(gate_packed, uniform_model=True),
        _encode_lane_owned_component(
            gate_scales.view(np.uint8).reshape(experts, expert_hidden_size, -1)
        ),
        _encode_lane_owned_component(up_packed, uniform_model=True),
        _encode_lane_owned_component(
            up_scales.view(np.uint8).reshape(experts, expert_hidden_size, -1)
        ),
        _encode_lane_owned_component(down_packed, uniform_model=True),
        _encode_lane_owned_component(
            down_scales.view(np.uint8).reshape(experts, hidden_size, -1)
        ),
    )
    images = _cached_expert_images(
        components,
        experts=experts,
        outputs=(
            expert_hidden_size,
            expert_hidden_size,
            expert_hidden_size,
            expert_hidden_size,
            hidden_size,
            hidden_size,
        ),
    )
    route_table_bytes = 256 * 4
    slot_bytes = (max(image.size for image in images) + route_table_bytes + 3) & ~3

    def padded(image):
        return np.pad(image, (0, slot_bytes - image.size))

    # Experts 2 and 0 occupy persistent slots 0/1; expert 1 occupies
    # transient slot 0. The kernel must perform this lookup from routed IDs.
    persistent_host = np.concatenate((padded(images[2]), padded(images[0])))
    persistent_host[-route_table_bytes:].view(np.uint32)[:3] = (1, 2, 0)
    persistent = mx.array(persistent_host)
    transient = mx.array(padded(images[1]))
    gate_transition = _scale_transition(mx, components[1])
    up_transition = _scale_transition(mx, components[3])
    down_transition = _scale_transition(mx, components[5])
    route = bind_glm52_q1t_fused_rans_cached_bank(
        persistent_bytes=persistent,
        transient_bytes=transient,
        slot_bytes=slot_bytes,
        persistent_slots=2,
        gate_scales_transition=gate_transition,
        up_scales_transition=up_transition,
        down_scales_transition=down_transition,
        hidden_size=hidden_size,
        expert_hidden_size=expert_hidden_size,
        gate_up_threads_per_tg=64,
        down_threads_per_tg=64,
        dtype=mx.bfloat16,
    )
    x = mx.arange(4 * hidden_size, dtype=mx.float32).reshape(4, hidden_size)
    x = x.astype(mx.bfloat16)
    ids = mx.array([2, 0, 2, 1], dtype=mx.uint32)

    result = route(x, ids)
    reference_ids = ids.astype(mx.int32)
    gate = shadow_gather_mm(
        x,
        reference_ids,
        mx.array(gate_packed),
        mx.array(gate_scales),
        codec="t158",
    )
    up = shadow_gather_mm(
        x,
        reference_ids,
        mx.array(up_packed),
        mx.array(up_scales),
        codec="t158",
    )
    reference = shadow_gather_mm(
        swiglu(gate, up),
        reference_ids,
        mx.array(down_packed),
        mx.array(down_scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert tuple(result.shape) == (4, hidden_size)
    assert mx.array_equal(result, reference).item()
    assert route.output_count == 1
    from mtplx.kernels import glm52_q1t_fused_rans as fused

    gate_source = fused._cached_gate_up_source(
        hidden_size,
        expert_hidden_size,
        bank_slot_bytes=slot_bytes,
        persistent_slots=2,
    )
    down_source = fused._cached_down_source(
        hidden_size,
        expert_hidden_size,
        bank_slot_bytes=slot_bytes,
        persistent_slots=2,
    )
    for source in (gate_source, down_source):
        assert "expert_ids[assignment]" in source
        assert "expert_slots[" in source
        assert "persistent_bytes" in source
        assert f"- {route_table_bytes}u" in source
        assert "transient_bytes" in source


def test_cached_uniform_packed_decode_is_specialized_by_t158_group() -> None:
    from mtplx.kernels import glm52_q1t_fused_rans as fused

    sources = (
        fused._cached_gate_up_source(
            128,
            128,
            bank_slot_bytes=4096,
            persistent_slots=2,
        ),
        fused._cached_down_source(
            128,
            128,
            bank_slot_bytes=4096,
            persistent_slots=2,
        ),
    )
    for source in sources:
        assert "rans_next_uniform_packed(" not in source
        assert "packed_rans_byte_12" in source
        assert "packed_pos += 13u" in source
        assert "packed_symbol" in source
        assert "packed_carry" in source


def test_uniform_packed_group_specialization_matches_the_rans_transition() -> None:
    rng = np.random.default_rng(56404)
    mask = 4095
    for _ in range(256):
        initial_state = int(rng.integers(1 << 16, 1 << 31, dtype=np.uint32))
        payload = rng.integers(0, 256, size=13, dtype=np.uint8)

        state = initial_state
        expected = []
        for byte in payload:
            expected.append((state & mask) >> 4)
            reduced = ((state >> 8) & ~15) | (state & 15)
            state = (reduced << 8) | int(byte)

        symbol = (initial_state & mask) >> 4
        carry = initial_state & 15
        actual = [symbol]
        actual.append((carry << 4) | (int(payload[0]) >> 4))
        actual.extend(
            ((int(payload[index - 2]) & 15) << 4) | (int(payload[index - 1]) >> 4)
            for index in range(2, 13)
        )

        assert actual == expected
        assert (((int(payload[11]) & 15) << 4) | (int(payload[12]) >> 4)) == (
            (state & mask) >> 4
        )
        assert int(payload[12]) & 15 == state & 15


def test_packed_scale_transition_table_matches_the_generic_rans_table() -> None:
    from mtplx.expert_rans import M, table_from_freq
    from mtplx.models.glm52_q1t_fused_rans import _pack_rans_transition_table

    frequency = np.ones(256, dtype=np.uint32)
    frequency[0] += M - int(frequency.sum())
    table = table_from_freq(frequency)
    packed = _pack_rans_transition_table(
        table.cum2sym,
        table.freq,
        table.cum[:-1],
    )

    assert packed.dtype == np.uint32
    assert packed.shape == (M,)
    for slot, entry in enumerate(packed):
        value = int(entry)
        symbol = value & 255
        decoded_frequency = ((value >> 8) & 4095) + 1
        residue = value >> 20
        expected_symbol = int(table.cum2sym[slot])
        assert symbol == expected_symbol
        assert decoded_frequency == int(table.freq[expected_symbol])
        assert residue == slot - int(table.cum[expected_symbol])


def test_cached_rans_expert_dispatch_is_bitwise_and_returns_only_final_output(
    monkeypatch,
) -> None:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import swiglu

    from mtplx import expert_rans_metal
    from mtplx.kernels.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_cached_expert,
    )
    from mtplx.kernels.shadow_gather import shadow_gather_mm

    for name in ("decode_component", "decode_container"):
        monkeypatch.setattr(
            expert_rans_metal,
            name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cached fused dispatch must decode inside matmul")
            ),
        )

    hidden_size = 128
    expert_hidden_size = 128
    gate_packed, gate_scales = _random_t158_bank(
        experts=1,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56301,
    )
    up_packed, up_scales = _random_t158_bank(
        experts=1,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56302,
    )
    down_packed, down_scales = _random_t158_bank(
        experts=1,
        out_dim=hidden_size,
        in_dim=expert_hidden_size,
        seed=56303,
    )
    components = (
        _encode_lane_owned_component(gate_packed, uniform_model=True),
        _encode_lane_owned_component(
            gate_scales.view(np.uint8).reshape(1, expert_hidden_size, -1)
        ),
        _encode_lane_owned_component(up_packed, uniform_model=True),
        _encode_lane_owned_component(
            up_scales.view(np.uint8).reshape(1, expert_hidden_size, -1)
        ),
        _encode_lane_owned_component(down_packed, uniform_model=True),
        _encode_lane_owned_component(
            down_scales.view(np.uint8).reshape(1, hidden_size, -1)
        ),
    )
    route = bind_glm52_q1t_fused_rans_cached_expert(
        slot_bytes=mx.array(_cached_expert_image(components)),
        gate_scales_transition=_scale_transition(mx, components[1]),
        up_scales_transition=_scale_transition(mx, components[3]),
        down_scales_transition=_scale_transition(mx, components[5]),
        hidden_size=hidden_size,
        expert_hidden_size=expert_hidden_size,
        gate_up_threads_per_tg=64,
        down_threads_per_tg=64,
        dtype=mx.bfloat16,
    )
    x = mx.arange(8 * hidden_size, dtype=mx.float32).reshape(8, hidden_size)
    x = x.astype(mx.bfloat16)

    result = route(x)
    ids = mx.zeros((8,), dtype=mx.int32)
    gate = shadow_gather_mm(
        x,
        ids,
        mx.array(gate_packed),
        mx.array(gate_scales),
        codec="t158",
    )
    up = shadow_gather_mm(
        x,
        ids,
        mx.array(up_packed),
        mx.array(up_scales),
        codec="t158",
    )
    reference = shadow_gather_mm(
        swiglu(gate, up),
        ids,
        mx.array(down_packed),
        mx.array(down_scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert route.output_count == 1
    assert tuple(result.shape) == (8, hidden_size)
    assert mx.array_equal(result, reference).item()
    source = inspect.getsource(route.__call__)
    for forbidden in ("decode_component", "decode_container", "mx.zeros", "if "):
        assert forbidden not in source


def test_fused_rans_switch_glu_bitwise_matches_three_shadow_projections() -> None:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import swiglu

    from mtplx.kernels.shadow_gather import shadow_gather_mm
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansBandEndSwitchGLU,
        Glm52Q1TFusedRansMlpRoute,
        Glm52Q1TFusedRansSwitchGLU,
    )

    experts = 3
    hidden_size = 128
    expert_hidden_size = 128
    gate_packed, gate_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=55001,
    )
    up_packed, up_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=55002,
    )
    down_packed, down_scales = _random_t158_bank(
        experts=experts,
        out_dim=hidden_size,
        in_dim=expert_hidden_size,
        seed=55003,
    )
    gate = _bind_projection(
        mx,
        gate_packed,
        gate_scales,
        in_dim=hidden_size,
        dtype=mx.bfloat16,
        input_repeat=2,
    )
    up = _bind_projection(
        mx,
        up_packed,
        up_scales,
        in_dim=hidden_size,
        dtype=mx.bfloat16,
        input_repeat=2,
    )
    down = _bind_projection(
        mx,
        down_packed,
        down_scales,
        in_dim=expert_hidden_size,
        dtype=mx.bfloat16,
    )
    route = Glm52Q1TFusedRansMlpRoute(gate=gate, up=up, down=down)
    switch = Glm52Q1TFusedRansSwitchGLU(
        route=route,
        hidden_size=hidden_size,
        top_k=2,
    )
    fenced_switch = Glm52Q1TFusedRansBandEndSwitchGLU(
        route=route,
        hidden_size=hidden_size,
        top_k=2,
    )
    x = mx.arange(hidden_size, dtype=mx.float32).reshape(1, 1, -1).astype(mx.bfloat16)
    indices = mx.array([[[2, 0]]], dtype=mx.int32)

    result = switch(x, indices)
    fenced_result = fenced_switch(x, indices)
    assignments = mx.broadcast_to(x.reshape(1, 1, hidden_size), (1, 2, hidden_size))
    assignments = assignments.reshape(2, hidden_size)
    expert_ids = indices.reshape(-1)
    gate_reference = shadow_gather_mm(
        assignments,
        expert_ids,
        mx.array(gate_packed),
        mx.array(gate_scales),
        codec="t158",
    )
    up_reference = shadow_gather_mm(
        assignments,
        expert_ids,
        mx.array(up_packed),
        mx.array(up_scales),
        codec="t158",
    )
    reference = shadow_gather_mm(
        swiglu(gate_reference, up_reference),
        expert_ids,
        mx.array(down_packed),
        mx.array(down_scales),
        codec="t158",
    ).reshape(1, 1, 2, hidden_size)
    mx.eval(result, fenced_result, reference)

    assert mx.array_equal(result, reference).item()
    assert mx.array_equal(fenced_result, reference).item()


def test_expert_banked_fused_route_is_bitwise_for_repeated_arbitrary_experts() -> None:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import swiglu

    from mtplx.kernels.shadow_gather import shadow_gather_mm
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansExpertBankRoute,
        Glm52Q1TFusedRansMlpRoute,
    )

    experts = 3
    hidden_size = 128
    expert_hidden_size = 128
    gate_packed, gate_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56101,
    )
    up_packed, up_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56102,
    )
    down_packed, down_scales = _random_t158_bank(
        experts=experts,
        out_dim=hidden_size,
        in_dim=expert_hidden_size,
        seed=56103,
    )
    routes = tuple(
        Glm52Q1TFusedRansMlpRoute(
            gate=_bind_expert_projection(
                mx,
                gate_packed,
                gate_scales,
                expert=expert,
                in_dim=hidden_size,
                dtype=mx.bfloat16,
            ),
            up=_bind_expert_projection(
                mx,
                up_packed,
                up_scales,
                expert=expert,
                in_dim=hidden_size,
                dtype=mx.bfloat16,
            ),
            down=_bind_expert_projection(
                mx,
                down_packed,
                down_scales,
                expert=expert,
                in_dim=expert_hidden_size,
                dtype=mx.bfloat16,
            ),
        )
        for expert in range(experts)
    )
    route = Glm52Q1TFusedRansExpertBankRoute(
        experts=routes,
        hidden_size=hidden_size,
        top_k=2,
    )
    x = (
        mx.arange(2 * hidden_size, dtype=mx.float32)
        .reshape(2, hidden_size)
        .astype(mx.bfloat16)
    )
    ids = mx.array([2, 0, 2, 1], dtype=mx.int32)

    result = route(x, ids)
    assignments = mx.broadcast_to(x.reshape(2, 1, hidden_size), (2, 2, hidden_size))
    assignments = assignments.reshape(4, hidden_size)
    gate_reference = shadow_gather_mm(
        assignments,
        ids,
        mx.array(gate_packed),
        mx.array(gate_scales),
        codec="t158",
    )
    up_reference = shadow_gather_mm(
        assignments,
        ids,
        mx.array(up_packed),
        mx.array(up_scales),
        codec="t158",
    )
    reference = shadow_gather_mm(
        swiglu(gate_reference, up_reference),
        ids,
        mx.array(down_packed),
        mx.array(down_scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert tuple(result.shape) == (4, hidden_size)
    assert mx.array_equal(result, reference).item()
    source = inspect.getsource(Glm52Q1TFusedRansExpertBankRoute.__call__)
    for forbidden in (
        "eligible",
        "fallback",
        "retry",
        "getenv",
        "environ",
        "counter",
        "validate",
    ):
        assert forbidden not in source


def test_expert_banked_single_dispatch_mlp_is_bitwise_and_final_output_only() -> None:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import swiglu

    from mtplx.kernels.glm52_q1t_fused_rans import (
        bind_glm52_q1t_fused_rans_expert_gate_up,
        bind_glm52_q1t_fused_rans_expert_mlp,
    )
    from mtplx.kernels.shadow_gather import shadow_gather_mm
    from mtplx.models.glm52_q1t_fused_rans import (
        Glm52Q1TFusedRansExpertBankRoute,
        Glm52Q1TFusedRansGateUpDownRoute,
    )

    experts = 3
    hidden_size = 128
    expert_hidden_size = 128
    gate_packed, gate_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56201,
    )
    up_packed, up_scales = _random_t158_bank(
        experts=experts,
        out_dim=expert_hidden_size,
        in_dim=hidden_size,
        seed=56202,
    )
    down_packed, down_scales = _random_t158_bank(
        experts=experts,
        out_dim=hidden_size,
        in_dim=expert_hidden_size,
        seed=56203,
    )
    expert_routes = []
    expert_projections = []
    for expert in range(experts):
        gate = _bind_expert_projection(
            mx,
            gate_packed,
            gate_scales,
            expert=expert,
            in_dim=hidden_size,
            dtype=mx.bfloat16,
        )
        up = _bind_expert_projection(
            mx,
            up_packed,
            up_scales,
            expert=expert,
            in_dim=hidden_size,
            dtype=mx.bfloat16,
        )
        down = _bind_expert_projection(
            mx,
            down_packed,
            down_scales,
            expert=expert,
            in_dim=expert_hidden_size,
            dtype=mx.bfloat16,
        )
        expert_routes.append(
            bind_glm52_q1t_fused_rans_expert_mlp(
                gate=gate,
                up=up,
                down=down,
                threads_per_tg=64,
            )
        )
        expert_projections.append((gate, up, down))
    route = Glm52Q1TFusedRansExpertBankRoute(
        experts=tuple(expert_routes),
        hidden_size=hidden_size,
        top_k=2,
    )
    x = (
        mx.arange(2 * hidden_size, dtype=mx.float32)
        .reshape(2, hidden_size)
        .astype(mx.bfloat16)
    )
    ids = mx.array([2, 0, 2, 1], dtype=mx.int32)

    result = route(x, ids)
    assignments = mx.broadcast_to(x.reshape(2, 1, hidden_size), (2, 2, hidden_size))
    assignments = assignments.reshape(4, hidden_size)
    gate_reference = shadow_gather_mm(
        assignments,
        ids,
        mx.array(gate_packed),
        mx.array(gate_scales),
        codec="t158",
    )
    up_reference = shadow_gather_mm(
        assignments,
        ids,
        mx.array(up_packed),
        mx.array(up_scales),
        codec="t158",
    )
    reference = shadow_gather_mm(
        swiglu(gate_reference, up_reference),
        ids,
        mx.array(down_packed),
        mx.array(down_scales),
        codec="t158",
    )
    mx.eval(result, reference)

    assert tuple(result.shape) == (4, hidden_size)
    assert mx.array_equal(result, reference).item()
    assert expert_routes[0].output_count == 1

    two_dispatch_route = Glm52Q1TFusedRansExpertBankRoute(
        experts=tuple(
            Glm52Q1TFusedRansGateUpDownRoute(
                gate_up=bind_glm52_q1t_fused_rans_expert_gate_up(
                    gate=gate,
                    up=up,
                    threads_per_tg=64,
                ),
                down=down,
            )
            for gate, up, down in expert_projections
        ),
        hidden_size=hidden_size,
        top_k=2,
    )
    two_dispatch_result = two_dispatch_route(x, ids)
    mx.eval(two_dispatch_result)

    assert mx.array_equal(two_dispatch_result, reference).item()
