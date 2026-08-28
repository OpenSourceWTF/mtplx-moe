"""Construction-bound MLX view over the exact Qwen4 streamed n-gram cache."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .qwen4_omlx import UnboundNGramRows


class AffineQ4NGramRows(nn.Module):
    """Gather and dequantize exact published affine-Q4/group-32 cache rows."""

    def __init__(self, cache: Any, *, row_width: int) -> None:
        super().__init__()
        if type(row_width) is not int or row_width <= 0 or row_width % 32:
            raise ValueError("row_width must be a positive multiple of 32")
        manifest = cache.manifest
        if manifest.storage != "affine-q4-g32":
            raise ValueError("Qwen4 n-gram cache must use affine-q4-g32 storage")
        if int(manifest.row_width) != row_width:
            raise ValueError("Qwen4 n-gram row width does not match the cache manifest")
        group_count = row_width // 32
        weight_bytes = row_width // 2
        parameter_bytes = group_count * 2
        row_bytes = weight_bytes + 2 * parameter_bytes
        if int(manifest.row_bytes) != row_bytes:
            raise ValueError("Qwen4 n-gram packed row geometry is invalid")
        arena = cache.arena_object
        if int(arena.size) % row_bytes:
            raise ValueError("Qwen4 n-gram arena contains a partial packed row")

        self._cache = cache
        self._acquire_rows = cache.acquire_prevalidated_rows
        self._arena_rows = arena.reshape((-1, row_bytes))
        self._arena_host_rows = np.frombuffer(
            memoryview(arena), dtype=np.uint8
        ).reshape((-1, row_bytes))
        self._row_width = row_width
        self._weight_bytes = weight_bytes
        self._parameter_bytes = parameter_bytes

    def __call__(self, row_ids: mx.array) -> mx.array:
        logical_shape = tuple(int(dimension) for dimension in row_ids.shape)
        host_rows = np.asarray(row_ids).reshape(-1)
        requested = tuple(int(row_id) for row_id in host_rows)
        lease = self._acquire_rows(requested)
        try:
            if len(logical_shape) == 3 and logical_shape[1] == 2:
                # The fixed verifier M=2 route owns a tiny packed-row copy.
                # Once copied, cache slots can be unpinned while dequantize
                # remains lazy and composes with the rest of the verifier.
                packed_host = np.array(
                    self._arena_host_rows[
                        np.asarray(lease.slot_ids, dtype=np.intp)
                    ],
                    copy=True,
                    order="C",
                )
                packed = mx.array(packed_host)
                lease.release()
                lease = None
            else:
                packed = mx.take(
                    self._arena_rows,
                    mx.array(lease.slot_ids, dtype=mx.int64),
                    axis=0,
                )
            weight_end = self._weight_bytes
            scale_end = weight_end + self._parameter_bytes
            weights = packed[:, :weight_end].view(mx.uint32)
            scales = packed[:, weight_end:scale_end].view(mx.bfloat16)
            biases = packed[:, scale_end:].view(mx.bfloat16)
            output = mx.dequantize(
                weights,
                scales,
                biases,
                group_size=32,
                bits=4,
                mode="affine",
            ).reshape((*logical_shape, self._row_width))
            if lease is not None:
                mx.eval(output)
            return output
        finally:
            if lease is not None:
                lease.release()

    def detach(self, cache: Any) -> None:
        """Drop every arena/cache reference after construction-owned unbinding."""

        if self._cache is not cache:
            raise ValueError("Qwen4 n-gram provider is bound to a different cache")
        self._cache = None
        self._acquire_rows = None
        self._arena_rows = None
        self._arena_host_rows = None


def bind_streamed_ngram_rows(model: Any, cache: Any) -> int:
    """Replace the target PLE placeholder with one prevalidated cache provider."""

    language_model = getattr(model, "language_model", model)
    text_model = getattr(language_model, "model", None)
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, (list, tuple)):
        raise TypeError("Qwen4 model does not expose target transformer layers")
    row_width = int(cache.manifest.row_width)
    bound = 0
    for layer_index, layer in enumerate(layers):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        embedding = getattr(ple, "ple_embedding", None)
        seam = getattr(embedding, "ngram_embedding", None)
        if not isinstance(seam, UnboundNGramRows):
            raise TypeError(f"Qwen4 PLE layer {layer_index} is already bound")
        if seam.layer_index != layer_index:
            raise ValueError("Qwen4 PLE layer index does not match its cache seam")
        embedding.ngram_embedding = AffineQ4NGramRows(
            cache,
            row_width=row_width,
        )
        bound += 1
    if bound == 0:
        raise ValueError("Qwen4 model exposes no target n-gram cache seam")
    return bound


def unbind_streamed_ngram_rows(model: Any, cache: Any) -> int:
    """Remove cache providers before releasing their fixed payload arena."""

    language_model = getattr(model, "language_model", model)
    text_model = getattr(language_model, "model", None)
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, (list, tuple)):
        raise TypeError("Qwen4 model does not expose target transformer layers")
    unbound = 0
    for layer_index, layer in enumerate(layers):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        embedding = getattr(ple, "ple_embedding", None)
        seam = getattr(embedding, "ngram_embedding", None)
        if isinstance(seam, UnboundNGramRows):
            continue
        if not isinstance(seam, AffineQ4NGramRows):
            raise TypeError(f"Qwen4 PLE layer {layer_index} has an unknown cache seam")
        row_width = seam._row_width
        seam.detach(cache)
        embedding.ngram_embedding = UnboundNGramRows(layer_index, row_width)
        unbound += 1
    return unbound


__all__ = [
    "AffineQ4NGramRows",
    "bind_streamed_ngram_rows",
    "unbind_streamed_ngram_rows",
]
