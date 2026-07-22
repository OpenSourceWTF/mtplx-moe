from __future__ import annotations

import gc
import sys


def test_shared_readonly_region_is_exact_metal_input(tmp_path) -> None:
    import mlx.core as mx
    import numpy as np

    from mtplx.mmap_mlx import activate_region, metal_u32, plan_region

    values = np.arange(4096, dtype=np.uint32)
    path = tmp_path / "shared-readonly.bin"
    path.write_bytes(values.tobytes())
    plan = plan_region(path, 0, 16384, shared_readonly=True)

    observed = []
    for _ in range(2):
        region = activate_region(plan)
        mapped = metal_u32(region)
        result = mx.sum(mapped)
        mx.eval(result)
        observed.append(int(result.item()))
        del result, mapped, region
        gc.collect()

    assert observed == [int(values.sum()), int(values.sum())]


def test_shared_readonly_region_exposes_page_aligned_metal_subbuffer(tmp_path) -> None:
    import mlx.core as mx
    import numpy as np

    from mtplx.mmap_mlx import (
        activate_region,
        metal_u32_slice,
        plan_region,
    )

    values = np.arange(3 * 4096, dtype=np.uint32)
    path = tmp_path / "shared-readonly-slice.bin"
    path.write_bytes(values.tobytes())
    plan = plan_region(path, 0, 3 * 16384, shared_readonly=True)
    region = activate_region(plan)

    mapped = metal_u32_slice(region, 16384, 16384)
    result = mx.sum(mapped)
    mx.eval(result)

    assert int(result.item()) == int(values[4096:8192].sum())


def test_region_plan_activation_and_transient_metal_view_are_separate_types(
    monkeypatch,
) -> None:
    import mtplx.mmap_mlx as mmap_mlx

    calls: list[tuple[object, ...]] = []

    class RegionPlan:
        def __init__(self, path, offset, length, shared_readonly):
            calls.append(("plan", path, offset, length, shared_readonly))

    class Region:
        def __init__(self, plan):
            calls.append(("activate", plan))

    class Extension:
        MappedRegionPlan = RegionPlan
        MappedRegion = Region

        @staticmethod
        def metal_u32(region):
            calls.append(("metal", region))
            return "transient-metal-array"

        @staticmethod
        def metal_u32_slice(region, offset, length):
            calls.append(("metal-slice", region, offset, length))
            return "transient-metal-slice"

    monkeypatch.setattr(
        mmap_mlx,
        "load_mmap_extension",
        Extension,
    )

    plan = mmap_mlx.plan_region(
        "artifact.bin", 16384, 32768, shared_readonly=True
    )
    region = mmap_mlx.activate_region(plan)
    result = mmap_mlx.metal_u32(region)
    sliced = mmap_mlx.metal_u32_slice(region, 16384, 16384)

    assert result == "transient-metal-array"
    assert sliced == "transient-metal-slice"
    assert calls == [
        ("plan", "artifact.bin", 16384, 32768, True),
        ("activate", plan),
        ("metal", region),
        ("metal-slice", region, 16384, 16384),
    ]


def test_mutable_metal_cache_buffer_uses_native_allocation_not_mlx_zeros(
    monkeypatch,
) -> None:
    import mtplx.mmap_mlx as mmap_mlx

    calls: list[int] = []

    class Extension:
        @staticmethod
        def allocate_metal_u8(length):
            calls.append(length)
            return "mutable-metal-u8"

    monkeypatch.setattr(mmap_mlx, "load_mmap_extension", Extension)

    assert mmap_mlx.allocate_metal_u8(8_847_360) == "mutable-metal-u8"
    assert calls == [8_847_360]


def test_native_mutable_metal_cache_buffer_is_writable_shared_memory() -> None:
    import mlx.core as mx

    import mtplx.mmap_mlx as mmap_mlx

    value = mmap_mlx.allocate_metal_u8(4096)
    view = memoryview(value).cast("B")
    try:
        assert value.dtype == mx.uint8
        assert tuple(value.shape) == (4096,)
        assert not view.readonly
        assert view.c_contiguous
        view[:8] = b"rans-t15"
        assert bytes(view[:8]) == b"rans-t15"
    finally:
        view.release()


def test_extension_loader_reuses_the_registered_native_module(monkeypatch) -> None:
    import mtplx.mmap_mlx as mmap_mlx

    loaded = object()
    monkeypatch.setitem(sys.modules, "mtplx._mmap_mlx", loaded)

    assert mmap_mlx.load_mmap_extension() is loaded
