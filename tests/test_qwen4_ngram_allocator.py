from __future__ import annotations

from types import SimpleNamespace


def test_ngram_allocator_exposes_only_fixed_2d_metal_arena(monkeypatch) -> None:
    import mtplx.mmap_mlx as allocator

    calls = []
    extension = SimpleNamespace(
        allocate_metal_u8_2d=lambda rows, columns: (
            calls.append((rows, columns)) or object()
        )
    )
    monkeypatch.setattr(allocator, "load_mmap_extension", lambda: extension)

    result = allocator.allocate_metal_u8_2d(37, 100)

    assert result is not None
    assert calls == [(37, 100)]
    assert not hasattr(allocator, "mmap_u32")
    assert not hasattr(allocator, "plan_region")
