from __future__ import annotations

from scripts.convert_streamed_rans_sharded import contiguous_ranges


def test_contiguous_ranges_cover_every_layer_once() -> None:
    ranges = contiguous_ranges((3, 4, 5, 7, 9), workers=3)
    assert ranges == ((3, 4), (5,), (7, 9))
    assert tuple(layer for chunk in ranges for layer in chunk) == (3, 4, 5, 7, 9)
