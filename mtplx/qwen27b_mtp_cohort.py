"""Construction-only fixed routes for the Qwen 3.6 27B K2 target.

This module intentionally has no MLX imports at normal import time.  The
installed lane binds MLX objects only after the exact model contract has been
validated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from .mtp_k2_stepper import (
    MTPK2RequestState,
    MTPK2VerifyResult,
    MTPK2VerifyTicket,
)


EXPECTED_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
EXPECTED_MODEL_DIRECTORY = "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
EXPECTED_BACKEND_ID = "qwen3_next"
EXPECTED_DEPTH = 2
EXPECTED_BITS = 4
EXPECTED_GROUP_SIZE = 64
EXPECTED_LAYER_COUNT = 64
EXPECTED_QLINEAR_COUNT = 497
EXPECTED_VERIFY_STRATEGY = "capture_commit"
EXPECTED_VERIFY_CORE = "linear-gdn-from-conv-tape"
EXPECTED_HIDDEN_VARIANT = "post_norm"
EXPECTED_POST_PREFILL_CACHE_TYPES = (
    "ArraysCache",
    "VllmMetalPagedKVCache",
)
# Frozen from bench/qwen27b/concurrency2-control-20260728-120938.json.
EXPECTED_QLINEAR_GEOMETRY_HISTOGRAM = (
    (5120, 48, 96),
    (5120, 1024, 32),
    (5120, 6144, 48),
    (5120, 10240, 48),
    (5120, 12288, 16),
    (5120, 17408, 128),
    (5120, 248320, 1),
    (6144, 5120, 64),
    (17408, 5120, 64),
)
EXPECTED_LAYER_STRUCTURE_SHA256 = (
    "3ff2c2c8ac7cc348801dfd0341fe8afa8985750b34f303f1b610dc7cdbddfdfc"
)
EXPECTED_QLINEAR_STRUCTURE_SHA256 = (
    "7c83a60bf2afaf71f4894bfa98a2fd22ab561c69d3d17b9fdc67fad258a5908e"
)
EXPECTED_ARCHITECTURE_ID = "qwen3-next-mtp"
EXPECTED_NATIVE_MTP_DEPTH_MAX = 3
_QMM_TOLERANCE = 0.0
_TARGET_LOGITS_TOLERANCE = 1.0
_TARGET_HIDDEN_TOLERANCE = 1.0
_TARGET_CAPTURE_TOLERANCE = 1.0
_TARGET_CACHE_TOLERANCE = 1.0


def _selfcheck_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"self-check {path} must be an object")
    return value


def _selfcheck_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"self-check {path} must be a list")
    return list(value)


def _selfcheck_shape_histogram(value: Any, path: str) -> Counter[tuple[int, int]]:
    histogram: Counter[tuple[int, int]] = Counter()
    for raw in _selfcheck_list(value, path):
        item = _selfcheck_mapping(raw, path)
        key = (int(item.get("k", -1)), int(item.get("n", -1)))
        count = int(item.get("module_count", 0))
        if min(*key, count) <= 0:
            raise ValueError(f"self-check {path} contains an invalid qlinear shape")
        histogram[key] += count
    return histogram


def _validate_selfcheck_comparison(
    value: Any,
    *,
    context: str,
    expected_tolerance: float,
) -> None:
    comparison = _selfcheck_mapping(value, context)
    path = str(comparison.get("path") or context)
    candidate_shape = comparison.get("candidate_shape")
    reference_shape = comparison.get("reference_shape")
    dmax = float(comparison.get("dmax", float("inf")))
    tolerance = float(comparison.get("tolerance", -1.0))
    diagnostic = (
        f"candidate_shape={candidate_shape}, "
        f"reference_shape={reference_shape}, "
        f"dmax={dmax}, tolerance={tolerance}"
    )
    if candidate_shape != reference_shape:
        raise ValueError(
            f"self-check {path} shape mismatch: {diagnostic}"
        )
    candidate_dtype = comparison.get("candidate_dtype")
    reference_dtype = comparison.get("reference_dtype")
    if (
        candidate_dtype is not None
        and reference_dtype is not None
        and candidate_dtype != reference_dtype
    ):
        raise ValueError(
            f"self-check {path} dtype mismatch: "
            f"candidate={candidate_dtype}, reference={reference_dtype}, "
            f"{diagnostic}"
        )
    if tolerance != expected_tolerance:
        raise ValueError(
            f"self-check {path} tolerance must be {expected_tolerance}: "
            f"{diagnostic}"
        )
    if dmax != dmax or dmax > expected_tolerance:
        raise ValueError(
            f"self-check {path} dmax exceeded tolerance: {diagnostic}"
        )


def _validate_selfcheck_cache_layers(
    value: Any,
    *,
    context: str,
    expected_tolerance: float,
) -> None:
    layers = _selfcheck_list(value, context)
    expected_attention = set(range(3, EXPECTED_LAYER_COUNT, 4))
    expected_pairs = {
        (row, layer_index)
        for row in range(2)
        for layer_index in range(EXPECTED_LAYER_COUNT)
    }
    observed_pairs: set[tuple[int, int]] = set()
    for raw in layers:
        layer = _selfcheck_mapping(raw, f"{context} layer")
        row = int(layer.get("row", -1))
        layer_index = int(layer.get("layer_index", -1))
        pair = (row, layer_index)
        if pair in observed_pairs:
            raise ValueError(
                f"self-check {context} duplicates row {row} layer {layer_index}"
            )
        observed_pairs.add(pair)
        expected_kind = (
            "attention"
            if layer_index in expected_attention
            else "recurrent"
        )
        kind = str(layer.get("kind") or "")
        if kind != expected_kind:
            raise ValueError(
                f"self-check {context} row {row} layer {layer_index} "
                f"kind must be {expected_kind!r}, got {kind!r}"
            )
        if kind == "attention":
            candidate_offset = int(layer.get("candidate_offset", -1))
            reference_offset = int(layer.get("reference_offset", -1))
            if candidate_offset != reference_offset:
                raise ValueError(
                    f"self-check {context} attention row {row} "
                    f"layer {layer_index} offset mismatch: "
                    f"candidate={candidate_offset}, "
                    f"reference={reference_offset}"
                )
        state_comparisons = _selfcheck_list(
            layer.get("state_comparisons"),
            f"{context} row {row} layer {layer_index} state comparisons",
        )
        if not state_comparisons:
            raise ValueError(
                f"self-check {context} row {row} layer {layer_index} "
                "has no state comparisons"
            )
        for item in state_comparisons:
            _validate_selfcheck_comparison(
                item,
                context=f"{context} row {row} layer {layer_index}",
                expected_tolerance=expected_tolerance,
            )
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            f"self-check {context} layer coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )


def validate_qwen27b_mtp_cohort_selfcheck_report(
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reject an incomplete or divergent actual-model construction receipt."""

    root = _selfcheck_mapping(report, "report")
    if root.get("schema") != "qwen27b-mtp-cohort-selfcheck-v1":
        raise ValueError("self-check report schema is not supported")
    if root.get("status") != "pass":
        raise ValueError(f"self-check report status is {root.get('status')!r}")
    if int(root.get("prefill_chunk_tokens", 0)) != 1024:
        raise ValueError(
            "self-check prefill_chunk_tokens must remain 1024"
        )
    prefill_spans = _selfcheck_mapping(
        root.get("prefill_spans"),
        "prefill_spans",
    )
    prefill_prompt_tokens = _selfcheck_mapping(
        root.get("prefill_prompt_tokens"),
        "prefill_prompt_tokens",
    )
    if (
        set(prefill_spans) != {"0", "1"}
        or set(prefill_prompt_tokens) != {"0", "1"}
    ):
        raise ValueError("self-check prefill spans require rows 0 and 1")
    for row, raw_spans in prefill_spans.items():
        spans = _selfcheck_list(raw_spans, f"prefill_spans.{row}")
        prompt_tokens = int(prefill_prompt_tokens[row])
        if prompt_tokens <= 1024 or len(spans) < 2:
            raise ValueError(
                f"self-check prefill row {row} must execute a full 1024-token "
                "chunk and a second chunk"
            )
        expected_start = 0
        for raw_span in spans:
            span = _selfcheck_list(raw_span, f"prefill_spans.{row}.span")
            if len(span) != 2:
                raise ValueError(
                    f"self-check prefill row {row} has invalid span {span}"
                )
            start, stop = (int(value) for value in span)
            if start != expected_start:
                raise ValueError(
                    f"self-check prefill row {row} must be contiguous: "
                    f"expected start {expected_start}, got {span}"
                )
            if start < 0 or stop <= start or stop - start > 1024:
                raise ValueError(
                    f"self-check prefill row {row} exceeded 1024: {span}"
                )
            expected_start = stop
        if list(spans[0]) != [0, 1024]:
            raise ValueError(
                f"self-check prefill row {row} must execute a full 1024-token "
                f"first chunk, got {spans[0]}"
            )
        if expected_start != prompt_tokens:
            raise ValueError(
                f"self-check prefill row {row} coverage ended at "
                f"{expected_start}, expected {prompt_tokens}"
            )
    if (
        root.get("target_cache_reference")
        != "two_owned_clones_per_prefilled_row"
    ):
        raise ValueError(
            "self-check requires two owned target-cache clones per "
            "prefilled request row"
        )

    qlinear = _selfcheck_mapping(root.get("qlinear"), "qlinear")
    if qlinear.get("reference") != "mx.quantized_matmul_transpose_q4_group64":
        raise ValueError(
            "self-check qlinear reference must be stock "
            "mx.quantized_matmul q4 group-size 64"
        )
    expected_count = int(qlinear.get("expected_module_count", 0))
    tested_count = int(qlinear.get("tested_module_count", 0))
    routes = _selfcheck_list(qlinear.get("routes"), "qlinear.routes")
    if expected_count <= 0 or tested_count != expected_count or len(routes) != expected_count:
        raise ValueError(
            "self-check qlinear route coverage mismatch: "
            f"expected={expected_count}, tested={tested_count}, routes={len(routes)}"
        )
    expected_shapes = _selfcheck_shape_histogram(
        qlinear.get("expected_shapes"),
        "qlinear.expected_shapes",
    )
    tested_shapes = _selfcheck_shape_histogram(
        qlinear.get("tested_shapes"),
        "qlinear.tested_shapes",
    )
    if tested_shapes != expected_shapes:
        raise ValueError(
            "self-check qlinear shape coverage mismatch: "
            f"expected={dict(expected_shapes)}, tested={dict(tested_shapes)}"
        )
    route_shapes: Counter[tuple[int, int]] = Counter()
    for index, raw in enumerate(routes):
        route = _selfcheck_mapping(raw, f"qlinear.routes[{index}]")
        module_path = str(route.get("module_path") or f"route[{index}]")
        k = int(route.get("k", -1))
        n = int(route.get("n", -1))
        route_shapes[(k, n)] += 1
        expected_input = [2, 3, k]
        expected_output = [2, 3, n]
        if route.get("input_shape") != expected_input:
            raise ValueError(
                f"self-check qlinear {module_path} input shape mismatch: "
                f"got={route.get('input_shape')}, wanted={expected_input}"
            )
        if route.get("output_shape") != expected_output:
            raise ValueError(
                f"self-check qlinear {module_path} output shape mismatch: "
                f"got={route.get('output_shape')}, wanted={expected_output}"
            )
        dmax = float(route.get("dmax", float("inf")))
        if dmax != dmax or dmax > _QMM_TOLERANCE:
            raise ValueError(
                f"self-check qlinear {module_path} dmax {dmax} "
                f"exceeded {_QMM_TOLERANCE}"
            )
    if route_shapes != expected_shapes:
        raise ValueError(
            "self-check qlinear route shape coverage mismatch: "
            f"expected={dict(expected_shapes)}, routes={dict(route_shapes)}"
        )

    target = _selfcheck_mapping(root.get("target_cycle"), "target_cycle")
    if target.get("input_shape") != [2, 3]:
        raise ValueError(
            "self-check target cycle input shape must be [2, 3], "
            f"got {target.get('input_shape')}"
        )
    output_comparisons = _selfcheck_list(
        target.get("output_comparisons"),
        "target_cycle.output_comparisons",
    )
    comparison_paths: set[str] = set()
    capture_layers = {0: set(), 1: set()}
    for item in output_comparisons:
        comparison = _selfcheck_mapping(item, "target output")
        path = str(comparison.get("path") or "")
        if path in comparison_paths:
            raise ValueError(
                f"self-check target output duplicates comparison path {path!r}"
            )
        comparison_paths.add(path)
        if path == "logits":
            expected_tolerance = _TARGET_LOGITS_TOLERANCE
        elif path == "hidden":
            expected_tolerance = _TARGET_HIDDEN_TOLERANCE
        elif path.startswith("captures."):
            pieces = path.split(".")
            if (
                len(pieces) < 4
                or pieces[1] not in {"row0", "row1"}
            ):
                raise ValueError(
                    f"self-check capture path is invalid: {path!r}"
                )
            row = int(pieces[1].removeprefix("row"))
            try:
                layer_index = int(pieces[2])
            except ValueError:
                raise ValueError(
                    f"self-check capture path is invalid: {path!r}"
                ) from None
            capture_layers[row].add(layer_index)
            expected_tolerance = _TARGET_CAPTURE_TOLERANCE
        else:
            raise ValueError(
                f"self-check target output path is unknown: {path!r}"
            )
        _validate_selfcheck_comparison(
            comparison,
            context="target output",
            expected_tolerance=expected_tolerance,
        )
    if not {"logits", "hidden"}.issubset(comparison_paths):
        raise ValueError(
            "self-check target output comparisons require logits and hidden"
        )
    expected_capture_layers = set(range(EXPECTED_LAYER_COUNT)) - set(
        range(3, EXPECTED_LAYER_COUNT, 4)
    )
    for row, observed_layers in capture_layers.items():
        if observed_layers != expected_capture_layers:
            missing = sorted(expected_capture_layers - observed_layers)
            extra = sorted(observed_layers - expected_capture_layers)
            raise ValueError(
                f"self-check capture layer coverage mismatch for row {row}: "
                f"missing={missing}, extra={extra}"
            )

    _validate_selfcheck_cache_layers(
        target.get("starting_cache_layers"),
        context="starting cache",
        expected_tolerance=0.0,
    )
    starting_aliasing = _selfcheck_list(
        target.get("starting_cache_aliasing"),
        "target_cycle.starting_cache_aliasing",
    )
    if [
        int(_selfcheck_mapping(item, "starting cache aliasing").get("row", -1))
        for item in starting_aliasing
    ] != [0, 1]:
        raise ValueError("self-check starting cache aliasing requires rows 0 and 1")
    for raw in starting_aliasing:
        item = _selfcheck_mapping(raw, "starting cache aliasing")
        row = int(item["row"])
        alias_fields = (
            "candidate_reference_aliasing",
            "candidate_sibling_aliasing",
            "reference_sibling_aliasing",
        )
        aliases = [field for field in alias_fields if item.get(field) is not False]
        if aliases:
            raise ValueError(
                f"self-check starting cache row {row} aliases another owner: "
                + ", ".join(aliases)
            )

    _validate_selfcheck_cache_layers(
        target.get("cache_layers"),
        context="target cache",
        expected_tolerance=_TARGET_CACHE_TOLERANCE,
    )
    _validate_selfcheck_cache_layers(
        target.get("commit_order_layers"),
        context="commit order",
        expected_tolerance=0.0,
    )

    rows = _selfcheck_list(target.get("rows"), "target_cycle.rows")
    if [int(_selfcheck_mapping(row, "target row").get("row", -1)) for row in rows] != [
        0,
        1,
    ]:
        raise ValueError("self-check target cycle requires rows 0 and 1")
    if (
        target.get("acceptance_source")
        != "GenerationOutput.stats.accepted_drafts"
    ):
        raise ValueError(
            "self-check acceptance must come from "
            "GenerationOutput.stats.accepted_drafts"
        )
    for raw in rows:
        row = _selfcheck_mapping(raw, "target row")
        index = int(row["row"])
        if row.get("candidate_tokens") != row.get("reference_tokens"):
            raise ValueError(f"self-check row {index} token parity mismatch")
        if int(row.get("candidate_accepted_drafts", -1)) != int(
            row.get("reference_accepted_drafts", -1)
        ):
            raise ValueError(f"self-check row {index} acceptance parity mismatch")

    isolation = _selfcheck_list(
        target.get("isolation"),
        "target_cycle.isolation",
    )
    isolation_pairs = [
        (
            int(_selfcheck_mapping(item, "target isolation").get("row", -1)),
            int(
                _selfcheck_mapping(item, "target isolation").get(
                    "sibling_row",
                    -1,
                )
            ),
        )
        for item in isolation
    ]
    if isolation_pairs != [(0, 1), (1, 0)]:
        raise ValueError(
            "self-check target isolation requires exact row pairs "
            "(0, 1) and (1, 0)"
        )
    for raw in isolation:
        item = _selfcheck_mapping(raw, "target isolation")
        row = int(item.get("row", -1))
        sibling = int(item.get("sibling_row", -1))
        if bool(item.get("extracted_aliasing", True)):
            raise ValueError(
                f"self-check extracted row {row} aliases sibling row {sibling}"
            )
        for isolation_field in (
            "sibling_unchanged_after_mutation",
            "sibling_unchanged_after_commit",
        ):
            if item.get(isolation_field) is not True:
                raise ValueError(
                    f"self-check row {row} changed sibling row {sibling} "
                    "during "
                    f"{isolation_field.removeprefix('sibling_unchanged_after_')}"
                )
    return report


@dataclass(frozen=True)
class TargetForwardResult:
    logits: Any
    hidden: Any
    captures: Mapping[int, Mapping[str, Any]] | Mapping[str, Any]
    cache: list[Any]


@dataclass(frozen=True)
class LayerCacheRoute:
    """Installed cache ownership operations for one exact target layer."""

    layer_index: int
    request_type: type
    cohort_type: type
    normalize_request: Callable[[Any], Any]
    merge: Callable[[tuple[Any, ...]], Any]
    extract: Callable[[Any, int], Any]
    own_request: Callable[[Any], Any]
    request_types: tuple[type, ...] = ()


PAIR_ATTENTION_GROWTH_RESERVE_TOKENS = 512


def _lazy_owned_array(mx: Any, value: Any) -> Any:
    owned = mx.zeros(value.shape, dtype=value.dtype)
    owned[tuple(slice(None) for _ in value.shape)] = value
    return owned


class QwenTensorOffsetBatchKVCache:
    """Ephemeral width-two cache with fixed capacity and row-local offsets."""

    def __init__(
        self,
        keys: Any,
        values: Any,
        offsets: Any,
        *,
        step: int = 256,
    ) -> None:
        self.cache = [keys, values, offsets]
        self.step = int(step)
        self._idx = 0 if keys is None else int(keys.shape[2])

    @property
    def keys(self) -> Any:
        return self.cache[0]

    @keys.setter
    def keys(self, value: Any) -> None:
        self.cache[0] = value

    @property
    def values(self) -> Any:
        return self.cache[1]

    @values.setter
    def values(self, value: Any) -> None:
        self.cache[1] = value

    @property
    def offset(self) -> Any:
        return self.cache[2]

    @offset.setter
    def offset(self, value: Any) -> None:
        self.cache[2] = value

    @property
    def state(self) -> list[Any]:
        return self.cache

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @staticmethod
    def _pad_capacity(mx: Any, value: Any, capacity: int) -> Any:
        extra = int(capacity) - int(value.shape[2])
        if extra <= 0:
            return value
        shape = (*value.shape[:2], extra, value.shape[3])
        return mx.concatenate(
            (value, mx.zeros(shape, dtype=value.dtype)),
            axis=2,
        )

    @classmethod
    def merge_lazy(
        cls,
        caches: tuple[Any, ...],
    ) -> "QwenTensorOffsetBatchKVCache":
        if len(caches) != 2:
            raise ValueError(
                f"Qwen tensor-offset cohort requires two rows, got {len(caches)}"
            )
        import mlx.core as mx

        capacity = max(int(cache.keys.shape[2]) for cache in caches)
        keys = mx.concatenate(
            tuple(cls._pad_capacity(mx, cache.keys, capacity) for cache in caches),
            axis=0,
        )
        values = mx.concatenate(
            tuple(
                cls._pad_capacity(mx, cache.values, capacity)
                for cache in caches
            ),
            axis=0,
        )
        offsets = mx.concatenate(
            tuple(cache.offset.reshape(1) for cache in caches),
            axis=0,
        )
        return cls(
            keys,
            values,
            offsets,
            step=max(int(getattr(cache, "step", 256)) for cache in caches),
        )

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        key_rows = []
        value_rows = []
        for row in range(self.batch_size):
            key_rows.append(
                mx.slice_update(
                    self.keys[row : row + 1],
                    keys[row : row + 1],
                    self.offset[row],
                    axes=(2,),
                )
            )
            value_rows.append(
                mx.slice_update(
                    self.values[row : row + 1],
                    values[row : row + 1],
                    self.offset[row],
                    axes=(2,),
                )
            )
        self.keys = mx.concatenate(tuple(key_rows), axis=0)
        self.values = mx.concatenate(tuple(value_rows), axis=0)
        self.offset = self.offset + int(keys.shape[2])
        return self.keys, self.values

    def make_mask(
        self,
        tokens: int,
        window_size: int | None = None,
        return_array: bool = False,
    ) -> Any:
        del return_array
        import mlx.core as mx

        capacity = int(self.keys.shape[2])
        key_positions = mx.arange(capacity)
        query_positions = self.offset[:, None] + mx.arange(int(tokens))[None, :]
        mask = query_positions[:, None, :, None] >= key_positions[None, None, None, :]
        if window_size is not None:
            mask = mask & (
                query_positions[:, None, :, None]
                < key_positions[None, None, None, :] + int(window_size)
            )
        return mask

    def extract_lazy(self, idx: int) -> Any:
        import mlx.core as mx

        from .graphbank import TensorOffsetKVCache

        request = TensorOffsetKVCache(
            _lazy_owned_array(mx, self.keys[idx : idx + 1]),
            _lazy_owned_array(mx, self.values[idx : idx + 1]),
            self.offset[idx],
            step=self.step,
        )
        request._granted = True
        return request

    def size(self) -> int:
        return self._idx

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes + self.offset.nbytes)


QWEN_TAPE_CAPTURE_ARRAY_KEYS = (
    "conv_states",
    "conv_out",
    "g",
    "state_in",
    "tape",
)


class Qwen27BCompiledWidth2Target:
    """Direct compiled B2 target over explicit fixed-layout cache leaves."""

    _ATTENTION = "attention"
    _RECURRENT = "recurrent"

    def __init__(
        self,
        *,
        execution: Any,
        capture_forward: Callable[[Any, list[Any]], tuple[Any, Any, Any]],
        fixed_scope: Callable[..., Any],
        cache_routes: tuple[LayerCacheRoute, ...],
        compile_fn: Callable[[Callable[..., Any]], Callable[..., Any]],
        async_eval: Callable[..., None],
    ) -> None:
        from .cache_state import OwnedRecurrentStateCache

        spec: list[tuple[int, str, int]] = []
        shadow: list[Any] = [None] * len(cache_routes)
        capture_layers: list[int] = []
        for route in cache_routes:
            if route.cohort_type is QwenTensorOffsetBatchKVCache:
                spec.append((route.layer_index, self._ATTENTION, 3))
                shadow[route.layer_index] = QwenTensorOffsetBatchKVCache(
                    None,
                    None,
                    None,
                )
            elif route.cohort_type is OwnedRecurrentStateCache:
                spec.append((route.layer_index, self._RECURRENT, 2))
                shadow[route.layer_index] = OwnedRecurrentStateCache(size=2)
                capture_layers.append(route.layer_index)
            else:
                raise TypeError(
                    "compiled Qwen width-2 target requires only installed "
                    "tensor-offset attention and owned recurrent cohort routes"
                )
        self._execution = execution
        self._capture_forward = capture_forward
        self._fixed_scope = fixed_scope
        self._spec = tuple(spec)
        self._shadow = shadow
        self._capture_layers = tuple(capture_layers)
        self._capture_leaf_count = (
            len(self._capture_layers) * len(QWEN_TAPE_CAPTURE_ARRAY_KEYS)
        )
        self._state_leaf_count = sum(item[2] for item in self._spec)
        self._async_eval = async_eval
        self._compiled = compile_fn(self._make_compiled_step())

    def _make_compiled_step(self) -> Callable[..., tuple[Any, ...]]:
        spec = self._spec
        shadow = self._shadow
        capture_layers = self._capture_layers
        capture_forward = self._capture_forward
        fixed_scope = self._fixed_scope
        execution = self._execution

        def compiled_step(input_ids: Any, *state_in: Any) -> tuple[Any, ...]:
            position = 0
            for layer_index, kind, leaf_count in spec:
                entry = shadow[layer_index]
                entry.cache[0] = state_in[position]
                entry.cache[1] = state_in[position + 1]
                if kind == self._ATTENTION:
                    entry.cache[2] = state_in[position + 2]
                    entry._idx = int(state_in[position].shape[2])
                position += leaf_count
            with fixed_scope(execution):
                logits, hidden, captures = capture_forward(input_ids, shadow)
            capture_out = tuple(
                captures[layer_index][key]
                for layer_index in capture_layers
                for key in QWEN_TAPE_CAPTURE_ARRAY_KEYS
            )
            state_out = tuple(
                entry.cache[slot]
                for layer_index, _kind, leaf_count in spec
                for entry in (shadow[layer_index],)
                for slot in range(leaf_count)
            )
            return (logits, hidden, *capture_out, *state_out)

        return compiled_step

    def _read_state(self, cache: list[Any]) -> tuple[Any, ...]:
        return tuple(
            entry.cache[slot]
            for layer_index, _kind, leaf_count in self._spec
            for entry in (cache[layer_index],)
            for slot in range(leaf_count)
        )

    def _mirror_state(self, cache: list[Any], state_out: tuple[Any, ...]) -> None:
        position = 0
        for layer_index, _kind, leaf_count in self._spec:
            entry = cache[layer_index]
            for slot in range(leaf_count):
                entry.cache[slot] = state_out[position + slot]
            position += leaf_count

    def _rebuild_captures(self, capture_out: tuple[Any, ...]) -> dict[int, Any]:
        return {
            layer_index: {
                key: capture_out[
                    layer_position * len(QWEN_TAPE_CAPTURE_ARRAY_KEYS)
                    + key_position
                ]
                for key_position, key in enumerate(QWEN_TAPE_CAPTURE_ARRAY_KEYS)
            }
            for layer_position, layer_index in enumerate(self._capture_layers)
        }

    def __call__(self, *, input_ids: Any, cache: list[Any]) -> TargetForwardResult:
        state_in = self._read_state(cache)
        self._async_eval(*state_in)
        outputs = self._compiled(input_ids, *state_in)
        self._async_eval(*outputs)
        capture_stop = 2 + self._capture_leaf_count
        capture_out = outputs[2:capture_stop]
        state_out = outputs[capture_stop:]
        self._mirror_state(cache, state_out)
        return TargetForwardResult(
            logits=outputs[0],
            hidden=outputs[1],
            captures=self._rebuild_captures(capture_out),
            cache=cache,
        )

    def release_construction_state(self) -> None:
        """Drop actual-model self-check leaves retained by the compiled shadow."""
        for layer_index, kind, leaf_count in self._spec:
            entry = self._shadow[layer_index]
            for slot in range(leaf_count):
                entry.cache[slot] = None
            if kind == self._RECURRENT:
                entry._owned_buffers = [None] * leaf_count
            else:
                entry._idx = 0


def _own_tensor_offset_request_lazy(cache: Any) -> Any:
    import mlx.core as mx

    from .graphbank import TensorOffsetKVCache

    request = TensorOffsetKVCache(
        _lazy_owned_array(mx, cache.keys),
        _lazy_owned_array(mx, cache.values),
        _lazy_owned_array(mx, cache.offset),
        step=int(getattr(cache, "step", 256)),
    )
    request._granted = True
    return request


def _normalize_recurrent_request(
    entry: Any,
    *,
    layer_index: int,
    arrays_type: type,
    owned_type: type,
) -> Any:
    if type(entry) not in {arrays_type, owned_type}:
        raise TypeError(
            f"Qwen target cache layer {layer_index} requires "
            f"{arrays_type.__name__} or {owned_type.__name__}, "
            f"got {type(entry).__name__}"
        )
    import mlx.core as mx

    state = entry.state
    if (
        not isinstance(state, list | tuple)
        or len(state) != 2
        or any(leaf is None for leaf in state)
    ):
        raise ValueError(
            f"Qwen target cache layer {layer_index} requires both recurrent "
            "state leaves before decode admission"
        )
    if any(
        not isinstance(leaf, mx.array)
        or len(leaf.shape) == 0
        or int(leaf.shape[0]) != 1
        for leaf in state
    ):
        raise ValueError(
            f"Qwen target cache layer {layer_index} requires request batch size 1"
        )

    def own_metadata(value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, mx.array) or int(value.size) != 1:
            raise ValueError(
                f"Qwen target cache layer {layer_index} requires request batch size 1"
            )
        owned = mx.contiguous(value)
        mx.eval(owned)
        return owned

    return owned_type(
        size=2,
        mode=getattr(entry, "mode", "persistent_eval"),
        initial=list(state),
        left_padding=own_metadata(getattr(entry, "left_padding", None)),
        lengths=own_metadata(getattr(entry, "lengths", None)),
    )


def _normalize_attention_request(
    entry: Any,
    *,
    layer_index: int,
    kv_type: type,
    paged_type: type,
    tensor_offset_type: type,
    tensor_offset_paged_type: type,
    reserve_tokens: int,
) -> Any:
    if type(entry) is tensor_offset_type:
        entry.ensure_capacity(entry.size() + int(reserve_tokens))
        return entry
    elif type(entry) is tensor_offset_paged_type:
        entry = entry.demote()
    if type(entry) is not kv_type and type(entry) is not paged_type:
        raise TypeError(
            f"Qwen target cache layer {layer_index} requires "
            f"{paged_type.__name__}, {tensor_offset_type.__name__}, "
            f"{tensor_offset_paged_type.__name__}, or plain "
            f"{kv_type.__name__}, "
            f"got {type(entry).__name__}"
        )

    import mlx.core as mx

    if type(entry) is paged_type:
        request = kv_type()
        keys, values = entry.state
        if keys is None or values is None:
            return request
        owned_keys = mx.contiguous(keys)
        owned_values = mx.contiguous(values)
        mx.eval(owned_keys, owned_values)
        request.state = (owned_keys, owned_values)
        entry = request
    return tensor_offset_type.from_kv_cache(
        entry,
        reserve_tokens=int(reserve_tokens),
    )


def build_qwen27b_cache_routes(
    layers: tuple[Any, ...],
    *,
    strict_topology: bool = True,
) -> tuple[LayerCacheRoute, ...]:
    """Bind exact request/cohort cache operations before lane publication."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    from .cache_state import (
        OwnedRecurrentStateCache,
        TensorOffsetVllmMetalPagedKVCache,
        VllmMetalPagedKVCache,
    )
    from .graphbank import TensorOffsetKVCache

    routes: list[LayerCacheRoute] = []
    recurrent_indices: list[int] = []
    attention_indices: list[int] = []
    for layer_index, layer in enumerate(layers):
        if bool(getattr(layer, "is_linear", False)):
            recurrent_indices.append(layer_index)
            routes.append(
                LayerCacheRoute(
                    layer_index=layer_index,
                    request_type=OwnedRecurrentStateCache,
                    cohort_type=OwnedRecurrentStateCache,
                    normalize_request=partial(
                        _normalize_recurrent_request,
                        layer_index=layer_index,
                        arrays_type=ArraysCache,
                        owned_type=OwnedRecurrentStateCache,
                    ),
                    merge=OwnedRecurrentStateCache.merge_lazy,
                    extract=OwnedRecurrentStateCache.extract_lazy,
                    own_request=partial(
                        OwnedRecurrentStateCache.extract_lazy,
                        idx=0,
                    ),
                    request_types=(ArraysCache, OwnedRecurrentStateCache),
                )
            )
        else:
            attention_indices.append(layer_index)
            routes.append(
                LayerCacheRoute(
                    layer_index=layer_index,
                    request_type=KVCache,
                    cohort_type=QwenTensorOffsetBatchKVCache,
                    normalize_request=partial(
                        _normalize_attention_request,
                        layer_index=layer_index,
                        kv_type=KVCache,
                        paged_type=VllmMetalPagedKVCache,
                        tensor_offset_type=TensorOffsetKVCache,
                        tensor_offset_paged_type=(
                            TensorOffsetVllmMetalPagedKVCache
                        ),
                        reserve_tokens=PAIR_ATTENTION_GROWTH_RESERVE_TOKENS,
                    ),
                    merge=QwenTensorOffsetBatchKVCache.merge_lazy,
                    extract=QwenTensorOffsetBatchKVCache.extract_lazy,
                    own_request=_own_tensor_offset_request_lazy,
                    request_types=(KVCache, VllmMetalPagedKVCache),
                )
            )
    if strict_topology:
        expected_attention = tuple(range(3, EXPECTED_LAYER_COUNT, 4))
        if len(routes) != EXPECTED_LAYER_COUNT:
            raise ValueError(
                f"Qwen cache routes require {EXPECTED_LAYER_COUNT} layers, "
                f"got {len(routes)}"
            )
        if tuple(attention_indices) != expected_attention:
            raise ValueError(
                "Qwen cache routes require full-attention layers at "
                f"{expected_attention}, got {tuple(attention_indices)}"
            )
        if len(recurrent_indices) != 48:
            raise ValueError(
                f"Qwen cache routes require 48 recurrent layers, "
                f"got {len(recurrent_indices)}"
            )
    return tuple(routes)


def normalize_target_cache(
    lane: "Qwen27BK2DualLane",
    request_cache: list[Any],
) -> list[Any]:
    """Normalize one request at admission, outside the measured target cycle."""
    if len(request_cache) != len(lane.cache_routes):
        raise ValueError(
            f"Qwen target cache has {len(request_cache)} layers; "
            f"installed route table has {len(lane.cache_routes)}"
        )
    return [
        route.normalize_request(request_cache[route.layer_index])
        for route in lane.cache_routes
    ]


def merge_target_caches(
    lane: "Qwen27BK2DualLane",
    request_caches: tuple[list[Any], ...],
) -> list[Any]:
    """Build an ephemeral target cohort cache through prebound route calls."""
    return [
        route.merge(tuple(cache[route.layer_index] for cache in request_caches))
        for route in lane.cache_routes
    ]


def extract_target_cache(
    lane: "Qwen27BK2DualLane",
    cohort_cache: list[Any],
    row: int,
) -> list[Any]:
    """Materialize one request-owned cache row from an ephemeral cohort."""
    return [
        route.extract(cohort_cache[route.layer_index], row)
        for route in lane.cache_routes
    ]


def own_request_target_cache(
    lane: "Qwen27BK2DualLane",
    request_cache: list[Any],
) -> list[Any]:
    """Build lazy owned request-cache roots after a width-one target call."""
    return [
        route.own_request(request_cache[route.layer_index])
        for route in lane.cache_routes
    ]


def _demote_selfcheck_request_cache(request_cache: list[Any]) -> list[Any]:
    """Mirror the generation finalization boundary for self-check references."""
    from .graphbank import TensorOffsetKVCache

    return [
        entry.demote() if type(entry) is TensorOffsetKVCache else entry
        for entry in request_cache
    ]


def assert_request_local_target_cache(
    lane: "Qwen27BK2DualLane",
    cache: list[Any],
) -> None:
    """Commit-boundary guard that rejects every installed cohort container."""
    if len(cache) != len(lane.cache_routes):
        raise TypeError("request/session commit received an incomplete target cache")
    for route in lane.cache_routes:
        entry = cache[route.layer_index]
        if route.cohort_type is not route.request_type and type(entry) is route.cohort_type:
            raise TypeError(
                f"request/session commit received cohort cache at layer "
                f"{route.layer_index}"
            )
        request_types = route.request_types or (route.request_type,)
        if type(entry) not in request_types:
            raise TypeError(
                f"request/session commit received non-request cache at layer "
                f"{route.layer_index}: {type(entry).__name__}"
            )
        batch_size = getattr(entry, "batch_size", None)
        if batch_size is None:
            keys = getattr(entry, "keys", None)
            batch_size = 1 if keys is None else int(keys.shape[0])
        batch_size = int(batch_size)
        if batch_size != 1:
            raise TypeError(
                f"request/session commit received cohort cache at layer "
                f"{route.layer_index}: batch_size={batch_size}"
            )


@dataclass(frozen=True)
class FixedQLinearRoute:
    module_path: str
    module_id: int
    k: int
    n: int
    bits: int
    group_size: int
    activation_dtype: object
    layout: str
    stock_call: Callable[[Any], Any]
    width2_call: Callable[[Any], Any]
    weight: Any = field(default=None, repr=False, compare=False)
    scales: Any = field(default=None, repr=False, compare=False)
    biases: Any = field(default=None, repr=False, compare=False)
    output_bias: Any = field(default=None, repr=False, compare=False)

    def execute(self, x: Any, *, width: int) -> Any:
        if width == 1:
            return self.stock_call(x)
        if width == 2:
            return self.width2_call(x)
        raise ValueError(f"fixed Qwen qlinear width must be 1 or 2, got {width}")


@dataclass(frozen=True)
class Qwen27BK2DualLane:
    backend_id: str
    depth: int
    bits: int
    group_size: int
    activation_dtype: object
    hidden_variant: str
    verify_strategy: str
    verify_core: str
    max_width: int
    width1_target: Callable[..., TargetForwardResult]
    width2_target: Callable[..., TargetForwardResult]
    cache_routes: tuple[LayerCacheRoute, ...]
    qlinear_routes: Mapping[int, FixedQLinearRoute]
    construction_receipt: Mapping[str, object]
    width1_execute_ticket: Callable[
        [MTPK2VerifyTicket],
        MTPK2VerifyResult,
    ] | None = field(default=None, repr=False, compare=False)
    capture_commit_routes: Mapping[tuple[int, int], Callable[..., list[Any]]] = (
        field(default_factory=lambda: MappingProxyType({}))
    )
    stock_width2_target: Callable[..., TargetForwardResult] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    qlinear_patch_lease: Any = field(default=None, repr=False, compare=False)

    def target_for_width(self, width: int) -> Callable[..., TargetForwardResult]:
        if width == 1:
            return self.width1_target
        if width == 2:
            return self.width2_target
        raise ValueError(f"Qwen27BK2DualLane width must be 1 or 2, got {width}")

    def capture_commit_for(
        self,
        width: int,
        row: int,
    ) -> Callable[..., list[Any]]:
        try:
            return self.capture_commit_routes[(int(width), int(row))]
        except KeyError:
            raise ValueError(
                "Qwen27BK2DualLane capture commit requires "
                f"(width, row) in ((1, 0), (2, 0), (2, 1)); "
                f"got ({width}, {row})"
            ) from None


def _materialize_cohort_trees(mx: Any, *values: Any) -> None:
    roots: list[Any] = []
    seen: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, mx.array):
            identity = id(value)
            if identity not in seen:
                seen.add(identity)
                roots.append(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, list | tuple):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    if roots:
        mx.eval(*roots)


def _cohort_dependencies() -> SimpleNamespace:
    import mlx.core as mx
    from time import perf_counter

    from .gdn_capture import extract_captured_row_lazy

    return SimpleNamespace(
        stack_rows=partial(mx.concatenate, axis=0),
        materialize=partial(_materialize_cohort_trees, mx),
        extract_captures=extract_captured_row_lazy,
        clock=perf_counter,
    )


def _commit_prefix(
    route: Callable[..., list[Any]],
    cache: list[Any],
    captures: Any,
    replay_memo: dict[tuple[int, int], Any],
    steps: int,
) -> list[Any]:
    return route(
        cache,
        captures,
        steps=steps,
        replay_memo=replay_memo,
    )


class MTPK2CohortRunner:
    """Execute one fixed-width Qwen K2 target cycle without dynamic fallback."""

    def __init__(
        self,
        lane: Qwen27BK2DualLane,
        *,
        dependencies: Any | None = None,
        execute_width1: Callable[[MTPK2VerifyTicket], MTPK2VerifyResult]
        | None = None,
    ) -> None:
        deps = _cohort_dependencies() if dependencies is None else dependencies
        installed_width1 = (
            execute_width1
            if execute_width1 is not None
            else getattr(deps, "execute_width1", None)
        )
        if installed_width1 is None:
            installed_width1 = lane.width1_execute_ticket
        if not callable(installed_width1):
            raise TypeError(
                "MTPK2CohortRunner requires a prebound width-one ticket executor"
            )
        self._lane = lane
        self._execute_width1 = installed_width1
        self._stack_rows = deps.stack_rows
        self._materialize = deps.materialize
        self._extract_captures = deps.extract_captures
        self._clock = deps.clock
        self._targets = (
            lane.target_for_width(1),
            lane.target_for_width(2),
        )
        self._commit_routes = (
            (lane.capture_commit_for(1, 0),),
            (
                lane.capture_commit_for(2, 0),
                lane.capture_commit_for(2, 1),
            ),
        )

    def step(
        self,
        requests: tuple[MTPK2RequestState, ...],
    ) -> tuple[MTPK2VerifyResult, ...]:
        live: list[MTPK2RequestState] = []
        for request in requests:
            if request._cancel_requested():
                request.close(status="cancelled")
            else:
                live.append(request)

        width = len(live)
        if width not in (1, 2):
            raise ValueError(
                f"cohort step requires one or two live requests, got {width}"
            )

        tickets: list[MTPK2VerifyTicket] = []
        for request in live:
            ticket = request.require_ticket()
            if not isinstance(ticket, MTPK2VerifyTicket):
                raise TypeError(
                    "MTPK2CohortRunner requires one pending verify ticket "
                    f"per request, got {type(ticket).__name__} for "
                    f"{request.request_id!r}"
                )
            tickets.append(ticket)

        if width == 1:
            result = self._execute_width1(tickets[0])
            request = live[0]
            if request._cancel_requested():
                request.close(status="cancelled")
                return ()
            return (result,)

        input_ids = self._stack_rows(
            tuple(ticket.input_ids for ticket in tickets)
        )
        cohort_cache = merge_target_caches(
            self._lane,
            tuple(ticket.request_cache for ticket in tickets),
        )
        target_cache = cohort_cache
        target = self._targets[width - 1]
        started = self._clock()
        forward = target(input_ids=input_ids, cache=target_cache)

        request_caches = tuple(
            extract_target_cache(self._lane, forward.cache, row)
            for row in range(width)
        )
        request_captures = tuple(
            self._extract_captures(forward.captures, row)
            for row in range(width)
        )
        request_logits = tuple(
            forward.logits[row : row + 1] for row in range(width)
        )
        request_hidden = tuple(
            forward.hidden[row : row + 1] for row in range(width)
        )
        self._materialize(
            forward.logits,
            forward.hidden,
            forward.captures,
            tuple(entry.state for entry in forward.cache),
            request_logits,
            request_hidden,
            request_captures,
        )
        forward_elapsed_s = self._clock() - started
        commit_routes = self._commit_routes[width - 1]
        replay_memo: dict[tuple[int, int], Any] = {}
        results = tuple(
            MTPK2VerifyResult(
                logits=request_logits[row],
                hidden=request_hidden[row],
                captures=request_captures[row],
                request_cache=request_caches[row],
                commit_prefix=partial(
                    _commit_prefix,
                    commit_routes[row],
                    request_caches[row],
                    forward.captures,
                    replay_memo,
                ),
                forward_elapsed_s=forward_elapsed_s,
            )
            for row in range(width)
        )

        survivors: list[MTPK2VerifyResult] = []
        for request, result in zip(live, results, strict=True):
            if request._cancel_requested():
                request.close(status="cancelled")
            else:
                survivors.append(result)
        return tuple(survivors)


def _construction_dependencies() -> SimpleNamespace:
    import mlx.core as mx
    import mlx.nn as nn

    from .nax_verify import (
        FixedQMMExecution,
        fixed_qmm_execution_scope,
        m6_ksplit_eligible,
        nax_qmm_m6_qmv_wide_vec6,
        prepare_fixed_qlinear_patch_lease,
    )
    from .artifacts import inspect_model
    from .generation import execute_solo_mtpk2_verify_ticket
    from .gdn_capture import (
        bind_qwen_capture_commit_route,
        forward_with_gdn_capture_configured,
        resolve_qwen_gdn_verify_config,
    )

    return SimpleNamespace(
        quantized_linear_type=nn.QuantizedLinear,
        bfloat16=mx.bfloat16,
        float16=mx.float16,
        prepare_patch_lease=prepare_fixed_qlinear_patch_lease,
        inspect_model_contract=partial(_inspect_model_contract, inspect_model),
        m6_eligible=m6_ksplit_eligible,
        nax_qmm_m6_qmv_wide_vec6=nax_qmm_m6_qmv_wide_vec6,
        numeric_self_check=partial(_numeric_self_check, mx),
        fixed_execution=FixedQMMExecution,
        fixed_scope=fixed_qmm_execution_scope,
        configured_capture=forward_with_gdn_capture_configured,
        resolve_capture_config=resolve_qwen_gdn_verify_config,
        bind_capture_commit_route=bind_qwen_capture_commit_route,
        build_compiled_width2_target=partial(
            Qwen27BCompiledWidth2Target,
            compile_fn=mx.compile,
            async_eval=mx.async_eval,
        ),
        actual_model_self_check=_actual_model_target_self_check,
        execute_width1_ticket=execute_solo_mtpk2_verify_ticket,
        expected_qlinear_count=EXPECTED_QLINEAR_COUNT,
        expected_geometry_histogram=EXPECTED_QLINEAR_GEOMETRY_HISTOGRAM,
        expected_layer_structure_sha256=EXPECTED_LAYER_STRUCTURE_SHA256,
        expected_qlinear_structure_sha256=EXPECTED_QLINEAR_STRUCTURE_SHA256,
    )


def _construction_error(
    model_path: Path,
    invariant: str,
    *,
    module_path: str | None = None,
) -> RuntimeError:
    owner = f", module/layer {module_path}" if module_path is not None else ""
    return RuntimeError(
        f"Qwen27BK2DualLane construction failed for model {model_path}{owner}: "
        f"{invariant}"
    )


def _load_json(path: Path, *, invariant: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _construction_error(path.parent, f"{invariant}: {exc}") from exc
    if not isinstance(value, dict):
        raise _construction_error(path.parent, f"{invariant}: expected JSON object")
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _layer_structure_sha256(layers: tuple[Any, ...]) -> str:
    records = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        records.append(
            {
                "attention_type": (
                    None if attention is None else _qualified_type(attention)
                ),
                "is_linear": bool(getattr(layer, "is_linear", False)),
                "layer_index": index,
                "layer_type": _qualified_type(layer),
            }
        )
    return _sha256_json(records)


def _qlinear_structure_sha256(
    candidates: list[tuple[str, Any, int, int, Any, Any, Any]],
) -> str:
    records = []
    for module_path, module, k, n, weight, _scales, _biases in candidates:
        records.append(
            {
                "bits": int(module.bits),
                "group_size": int(module.group_size),
                "k": k,
                "mode": str(module.mode),
                "n": n,
                "path": module_path,
                "type": _qualified_type(module),
                "weight_dtype": str(weight.dtype),
                "weight_shape": [int(value) for value in weight.shape],
            }
        )
    records.sort(key=lambda item: item["path"])
    return _sha256_json(records)


def _inspect_model_contract(
    inspect_model: Callable[[Path], Any],
    runtime: Any,
    model_path: Path,
) -> dict[str, object]:
    inspection = inspect_model(model_path)
    compatibility = getattr(inspection, "compatibility", None)
    compatibility = compatibility if isinstance(compatibility, dict) else {}
    runtime_contract = compatibility.get("runtime_contract")
    runtime_contract = (
        runtime_contract if isinstance(runtime_contract, dict) else {}
    )
    mtp = getattr(inspection, "mtp", None)
    return {
        "backend_id": compatibility.get("recommended_backend"),
        "architecture_id": compatibility.get("arch_id"),
        "native_mtp_enabled": bool(
            getattr(mtp, "exists", False)
            and compatibility.get("mtp_supported") == "yes"
            and getattr(runtime, "mtp_enabled", False)
        ),
        "native_mtp_model_depth_max": runtime_contract.get("mtp_depth_max"),
    }


def _stock_call(original: Callable[..., Any], module: Any, x: Any) -> Any:
    return original(module, x)


def _execute_m6_without_output_bias(
    x: Any,
    *,
    kernel: Callable[..., Any],
    weight: Any,
    scales: Any,
    biases: Any,
    bits: int,
    group_size: int,
    k: int,
    n: int,
    activation_dtype: object,
) -> Any:
    del bits, activation_dtype
    y = kernel(
        x.reshape(6, k),
        weight,
        scales,
        biases,
        group_size=group_size,
    )
    return y.reshape(*x.shape[:-1], n)


def _execute_m6_with_output_bias(
    x: Any,
    *,
    kernel: Callable[..., Any],
    weight: Any,
    scales: Any,
    biases: Any,
    output_bias: Any,
    bits: int,
    group_size: int,
    k: int,
    n: int,
    activation_dtype: object,
) -> Any:
    del bits, activation_dtype
    y = kernel(
        x.reshape(6, k),
        weight,
        scales,
        biases,
        group_size=group_size,
    )
    return y.reshape(*x.shape[:-1], n) + output_bias


def _prebound_m6_call(
    *,
    kernel: Callable[..., Any],
    module: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    bits: int,
    group_size: int,
    k: int,
    n: int,
    activation_dtype: object,
) -> Callable[[Any], Any]:
    bound = {
        "kernel": kernel,
        "weight": weight,
        "scales": scales,
        "biases": biases,
        "bits": bits,
        "group_size": group_size,
        "k": k,
        "n": n,
        "activation_dtype": activation_dtype,
    }
    if "bias" in module:
        return partial(
            _execute_m6_with_output_bias,
            output_bias=module["bias"],
            **bound,
        )
    return partial(_execute_m6_without_output_bias, **bound)


def _numeric_self_check(
    mx: Any,
    *,
    module: Any,
    route: FixedQLinearRoute,
) -> float:
    mx.random.seed(27)
    x = (
        mx.random.normal((2, 3, route.k), dtype=mx.float32) * 0.25
    ).astype(route.activation_dtype)
    if route.weight is None or route.scales is None or route.biases is None:
        stock = route.stock_call(x)
    else:
        stock = mx.quantized_matmul(
            x,
            route.weight,
            scales=route.scales,
            biases=route.biases,
            transpose=True,
            bits=route.bits,
            group_size=route.group_size,
        )
        if route.output_bias is not None:
            stock = stock + route.output_bias
    candidate = route.width2_call(x)
    mx.eval(stock, candidate)
    if tuple(stock.shape) != tuple(candidate.shape):
        return float("inf")
    diff = mx.max(
        mx.abs(
            candidate.astype(mx.float32)
            - stock.astype(mx.float32)
        )
    )
    mx.eval(diff)
    del module
    value = float(diff.item())
    return value if value == value else float("inf")


def _target_callable(
    *,
    execution: Any,
    capture_forward: Callable[[Any, list[Any]], tuple[Any, Any, Any]],
    fixed_scope: Callable[..., Any],
) -> Callable[..., TargetForwardResult]:
    def target(*, input_ids: Any, cache: list[Any]) -> TargetForwardResult:
        with fixed_scope(execution):
            logits, hidden, captures = capture_forward(input_ids, cache)
        return TargetForwardResult(
            logits=logits,
            hidden=hidden,
            captures=captures,
            cache=cache,
        )

    return target


def _configured_capture_callable(
    runtime: Any,
    *,
    configured_capture: Callable[..., tuple[Any, Any, Any]],
    config: Any,
) -> Callable[[Any, list[Any]], tuple[Any, Any, Any]]:
    return partial(
        configured_capture,
        runtime.model,
        config=config,
    )


def _capture_config_receipt(config: Any) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "target_width": config.target_width,
            "attention_cache_type": config.attention_cache_type,
            "capture_backend": config.capture_backend,
            "projection_path": config.projection_path,
            "linear_conv_path": config.linear_conv_path,
            "authoritative_state_path": config.authoritative_state_path,
            "gdn_tail_path": config.gdn_tail_path,
            "residual_path": config.residual_path,
            "hidden_variant": config.hidden_variant,
            "layer_eval_every": config.layer_eval_every,
            "layer_eval_schedule": config.layer_eval_schedule,
            "layer_eval_context_threshold": (
                config.layer_eval_context_threshold
            ),
            "layer_eval_max_q": config.layer_eval_max_q,
            "tape_replay_tgy": config.tape_replay_tgy,
        }
    )


def _selfcheck_array_leaves(
    mx: Any,
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    leaves: dict[str, Any] = {}

    def visit(item: Any, current: str) -> None:
        if isinstance(item, mx.array):
            leaves[current] = item
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda value: str(value)):
                visit(item[key], f"{current}.{key}")
            return
        if isinstance(item, list | tuple):
            for index, child in enumerate(item):
                visit(child, f"{current}.{index}")
            return
        if hasattr(item, "state"):
            visit(getattr(item, "state"), f"{current}.state")

    visit(value, path)
    return leaves


def _selfcheck_array_comparison(
    mx: Any,
    *,
    path: str,
    candidate: Any | None,
    reference: Any | None,
    tolerance: float,
) -> dict[str, object]:
    candidate_shape = (
        None
        if candidate is None
        else [int(value) for value in candidate.shape]
    )
    reference_shape = (
        None
        if reference is None
        else [int(value) for value in reference.shape]
    )
    candidate_dtype = None if candidate is None else str(candidate.dtype)
    reference_dtype = None if reference is None else str(reference.dtype)
    dmax = float("inf")
    if (
        candidate is not None
        and reference is not None
        and candidate_shape == reference_shape
        and candidate_dtype == reference_dtype
    ):
        if int(candidate.size) == 0:
            dmax = 0.0
        else:
            delta = mx.max(
                mx.abs(
                    candidate.astype(mx.float32)
                    - reference.astype(mx.float32)
                )
            )
            mx.eval(delta)
            dmax = float(delta.item())
            if dmax != dmax:
                dmax = float("inf")
    return {
        "path": path,
        "candidate_shape": candidate_shape,
        "reference_shape": reference_shape,
        "candidate_dtype": candidate_dtype,
        "reference_dtype": reference_dtype,
        "dmax": dmax,
        "tolerance": float(tolerance),
    }


def _selfcheck_tree_comparisons(
    mx: Any,
    *,
    path: str,
    candidate: Any,
    reference: Any,
    tolerance: float,
) -> list[dict[str, object]]:
    candidate_leaves = _selfcheck_array_leaves(mx, candidate, path=path)
    reference_leaves = _selfcheck_array_leaves(mx, reference, path=path)
    paths = sorted(set(candidate_leaves) | set(reference_leaves))
    return [
        _selfcheck_array_comparison(
            mx,
            path=leaf_path,
            candidate=candidate_leaves.get(leaf_path),
            reference=reference_leaves.get(leaf_path),
            tolerance=tolerance,
        )
        for leaf_path in paths
    ]


def _selfcheck_snapshot_cache(mx: Any, cache: list[Any]) -> dict[str, Any]:
    leaves = _selfcheck_array_leaves(mx, cache, path="cache")
    snapshots = {
        path: value + mx.zeros((), dtype=value.dtype)
        for path, value in leaves.items()
    }
    if snapshots:
        mx.eval(*snapshots.values())
    return snapshots


def _selfcheck_cache_matches_snapshot(
    mx: Any,
    cache: list[Any],
    snapshot: Mapping[str, Any],
) -> bool:
    leaves = _selfcheck_array_leaves(mx, cache, path="cache")
    if set(leaves) != set(snapshot):
        return False
    comparisons = (
        _selfcheck_array_comparison(
            mx,
            path=path,
            candidate=leaves[path],
            reference=snapshot[path],
            tolerance=0.0,
        )
        for path in sorted(leaves)
    )
    return all(
        item["candidate_shape"] == item["reference_shape"]
        and float(item["dmax"]) == 0.0
        for item in comparisons
    )


def _selfcheck_mutate_cache(mx: Any, cache: list[Any]) -> None:
    leaves = tuple(
        value
        for value in _selfcheck_array_leaves(
            mx,
            cache,
            path="cache",
        ).values()
        if int(value.size) > 0
    )
    if not leaves:
        raise RuntimeError("extracted request cache has no mutable array leaves")
    for leaf in leaves:
        index = tuple(0 for _ in leaf.shape)
        leaf[index] = leaf[index] + mx.array(1, dtype=leaf.dtype)
    mx.eval(*leaves)


def _selfcheck_ticket_tokens(ticket: MTPK2VerifyTicket) -> list[int]:
    values = ticket.input_ids.reshape(-1).tolist()
    if isinstance(values, list):
        return [int(value) for value in values]
    return [int(values)]


def _selfcheck_prompt_ids(
    runtime: Any,
    *,
    row: int,
    prompt_tokens: int,
) -> list[int]:
    seed = [
        int(token)
        for token in runtime.tokenizer.encode(
            (
                "Laguna fixed prefill chunking validates the exact Qwen "
                f"concurrency-two request lane for row {row}. "
            ),
            add_special_tokens=False,
        )
    ]
    if not seed:
        raise RuntimeError(f"self-check tokenizer returned no tokens for row {row}")
    repeats = (prompt_tokens + len(seed) - 1) // len(seed)
    return (seed * repeats)[:prompt_tokens]


def _selfcheck_prefill_ticket_key(ticket: Any) -> tuple[object, ...]:
    execution = ticket.execution
    tokens = ticket.input_ids.reshape(-1).tolist()
    return (
        int(ticket.prompt_start),
        int(ticket.prompt_stop),
        str(getattr(execution, "route", "")),
        tuple(int(token) for token in tokens),
    )


def _prepare_actual_selfcheck_state_pair(
    runtime: Any,
    lane: Qwen27BK2DualLane,
    *,
    prompt_ids: list[int],
    row: int,
    seed: int,
    environment: Mapping[str, str],
) -> tuple[
    MTPK2RequestState,
    MTPK2RequestState,
    tuple[tuple[int, int], ...],
]:
    from .generation import (
        execute_solo_mtpk2_prefill_ticket,
        make_mtpk2_request_state_from_environment,
    )
    from .mtp_k2_stepper import MTPK2PrefillTicket
    from .sampling import SamplerConfig

    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=4)

    def construct(role: str) -> MTPK2RequestState:
        state = make_mtpk2_request_state_from_environment(
            runtime,
            prompt_ids,
            request_id=f"selfcheck-{role}-{row}",
            environment=environment,
            _construction_lane=lane,
            max_tokens=2,
            sampler=sampler,
            draft_sampler=sampler,
            speculative_depth=2,
            seed=seed,
            stop_token_ids=set(),
            mtp_hidden_variant=EXPECTED_HIDDEN_VARIANT,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy=EXPECTED_VERIFY_STRATEGY,
            verify_core=EXPECTED_VERIFY_CORE,
            adaptive_policy=None,
            draft_margin_threshold=None,
            capture_final_state=False,
        )
        if (
            int(state.prefill_chunk_tokens) != 1024
            or int(state.config.prefill_chunk_tokens) != 1024
        ):
            raise RuntimeError(
                "self-check installed request state did not freeze Laguna "
                "prefill_chunk_tokens=1024"
            )
        return state

    candidate = construct("candidate")
    reference: MTPK2RequestState | None = None
    try:
        ticket = candidate.start()
        spans: list[tuple[int, int]] = []
        prefill_receipts: list[tuple[Any, Any]] = []
        while isinstance(ticket, MTPK2PrefillTicket):
            span = (int(ticket.prompt_start), int(ticket.prompt_stop))
            if span[1] - span[0] > 1024:
                raise RuntimeError(
                    f"self-check prefill chunk exceeded 1024 tokens: {span}"
                )
            spans.append(span)
            result = candidate.execute_pending(
                lambda pending: execute_solo_mtpk2_prefill_ticket(
                    runtime,
                    pending,
                )
            )
            prefill_receipts.append((ticket, result))
            ticket = candidate.resume(result)
        if not isinstance(ticket, MTPK2VerifyTicket):
            raise RuntimeError(
                f"self-check candidate row {row} did not reach a K2 verify ticket"
            )

        normalized = normalize_target_cache(lane, ticket.request_cache)
        candidate_cache = own_request_target_cache(lane, normalized)
        reference_cache = own_request_target_cache(lane, normalized)
        candidate_ticket = replace(ticket, request_cache=candidate_cache)
        candidate.pending_ticket = candidate_ticket
        candidate.target_cache = candidate_cache

        reference = construct("reference")
        reference_ticket = reference.start()
        for candidate_prefill, result in prefill_receipts:
            if not isinstance(reference_ticket, MTPK2PrefillTicket):
                raise RuntimeError(
                    f"self-check reference row {row} ended before prefill replay"
                )
            candidate_key = _selfcheck_prefill_ticket_key(candidate_prefill)
            reference_key = _selfcheck_prefill_ticket_key(reference_ticket)
            if candidate_key != reference_key:
                raise RuntimeError(
                    f"self-check prefill replay diverged for row {row}: "
                    f"candidate={candidate_key[:3]}, "
                    f"reference={reference_key[:3]}"
                )
            reference_ticket = reference.resume(
                replace(result, request_cache=reference_cache)
            )
        if not isinstance(reference_ticket, MTPK2VerifyTicket):
            raise RuntimeError(
                f"self-check reference row {row} did not reach a K2 verify ticket"
            )
        reference_ticket = replace(
            reference_ticket,
            request_cache=reference_cache,
        )
        reference.pending_ticket = reference_ticket
        reference.target_cache = reference_cache
        return candidate, reference, tuple(spans)
    except BaseException:
        candidate.close()
        if reference is not None:
            reference.close()
        raise


def _selfcheck_cache_layer_receipts(
    mx: Any,
    lane: Qwen27BK2DualLane,
    *,
    path: str,
    candidates: tuple[list[Any], list[Any]],
    references: tuple[list[Any], list[Any]],
    tolerance: float,
) -> list[dict[str, object]]:
    layers: list[dict[str, object]] = []
    for row, (candidate_cache, reference_cache) in enumerate(
        zip(candidates, references, strict=True)
    ):
        for route in lane.cache_routes:
            candidate_entry = candidate_cache[route.layer_index]
            reference_entry = reference_cache[route.layer_index]
            kind = (
                "attention"
                if route.request_type.__name__ == "KVCache"
                else "recurrent"
            )
            item: dict[str, object] = {
                "row": row,
                "layer_index": route.layer_index,
                "kind": kind,
                "state_comparisons": _selfcheck_tree_comparisons(
                    mx,
                    path=(
                        f"{path}.row{row}.layer{route.layer_index}.state"
                    ),
                    candidate=candidate_entry,
                    reference=reference_entry,
                    tolerance=tolerance,
                ),
            }
            if kind == "attention":
                item.update(
                    {
                        "candidate_offset": int(
                            getattr(candidate_entry, "offset", -1)
                        ),
                        "reference_offset": int(
                            getattr(reference_entry, "offset", -1)
                        ),
                    }
                )
            layers.append(item)
    return layers


def _selfcheck_cache_array_ids(mx: Any, cache: list[Any]) -> set[int]:
    return {
        id(value)
        for value in _selfcheck_array_leaves(
            mx,
            cache,
            path="cache",
        ).values()
    }


def _selfcheck_cache_layer_receipts_pass(
    layers: list[dict[str, object]],
) -> bool:
    for layer in layers:
        if (
            layer["kind"] == "attention"
            and layer["candidate_offset"] != layer["reference_offset"]
        ):
            return False
        comparisons = layer["state_comparisons"]
        if not isinstance(comparisons, list) or not comparisons:
            return False
        for raw in comparisons:
            comparison = _selfcheck_mapping(raw, "cache comparison")
            if (
                comparison.get("candidate_shape")
                != comparison.get("reference_shape")
                or float(comparison.get("dmax", float("inf"))) != 0.0
            ):
                return False
    return True


def _selfcheck_request_for_ticket(
    ticket: MTPK2VerifyTicket,
    *,
    request_id: str,
) -> Any:
    return SimpleNamespace(
        request_id=request_id,
        _cancel_requested=lambda: False,
        close=lambda **_kwargs: None,
        require_ticket=lambda: ticket,
    )


def _actual_model_target_self_check(
    runtime: Any,
    lane: Qwen27BK2DualLane,
    *,
    qlinear_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    import os

    import mlx.core as mx

    from .cache_state import restore_cache, snapshot_cache

    if lane.stock_width2_target is None:
        raise RuntimeError("stock width-2 reference target was not installed")
    environment_values = {
        str(name): str(value)
        for name, value in os.environ.items()
    }
    environment_values.update(
        {
            "MTPLX_CONTEXT_COPY": "0",
            "MTPLX_STATE_REBASE_EVERY": "0",
            "MTPLX_SUSTAINED_PREFILL": "1",
        }
    )
    environment = MappingProxyType(environment_values)
    prompt_ids_by_row = (
        _selfcheck_prompt_ids(runtime, row=0, prompt_tokens=1100),
        _selfcheck_prompt_ids(runtime, row=1, prompt_tokens=1101),
    )
    custom_states: list[MTPK2RequestState] = []
    stock_states: list[MTPK2RequestState] = []
    prefill_spans: dict[str, list[list[int]]] = {}
    try:
        for row, prompt_ids in enumerate(prompt_ids_by_row):
            custom, stock, spans = _prepare_actual_selfcheck_state_pair(
                runtime,
                lane,
                prompt_ids=prompt_ids,
                row=row,
                seed=2700 + row,
                environment=environment,
            )
            prefill_spans[str(row)] = [list(span) for span in spans]
            custom_states.append(custom)
            stock_states.append(stock)

        custom_tickets = tuple(
            state.require_ticket()
            for state in custom_states
        )
        stock_tickets = tuple(
            state.require_ticket()
            for state in stock_states
        )
        for row, (candidate, reference) in enumerate(
            zip(custom_tickets, stock_tickets, strict=True)
        ):
            if (
                not isinstance(candidate, MTPK2VerifyTicket)
                or not isinstance(reference, MTPK2VerifyTicket)
            ):
                raise RuntimeError(
                    f"self-check row {row} did not produce verify tickets"
                )
            candidate_tokens = _selfcheck_ticket_tokens(candidate)
            reference_tokens = _selfcheck_ticket_tokens(reference)
            if candidate_tokens != reference_tokens:
                raise RuntimeError(
                    f"verify input token mismatch at row {row}: "
                    f"candidate={candidate_tokens}, reference={reference_tokens}"
                )

        starting_cache_layers = _selfcheck_cache_layer_receipts(
            mx,
            lane,
            path="starting_cache",
            candidates=tuple(
                ticket.request_cache for ticket in custom_tickets
            ),
            references=tuple(
                ticket.request_cache for ticket in stock_tickets
            ),
            tolerance=0.0,
        )
        custom_start_ids = tuple(
            _selfcheck_cache_array_ids(mx, ticket.request_cache)
            for ticket in custom_tickets
        )
        stock_start_ids = tuple(
            _selfcheck_cache_array_ids(mx, ticket.request_cache)
            for ticket in stock_tickets
        )
        starting_cache_aliasing = [
            {
                "row": row,
                "candidate_reference_aliasing": bool(
                    custom_start_ids[row] & stock_start_ids[row]
                ),
                "candidate_sibling_aliasing": bool(
                    custom_start_ids[row] & custom_start_ids[1 - row]
                ),
                "reference_sibling_aliasing": bool(
                    stock_start_ids[row] & stock_start_ids[1 - row]
                ),
            }
            for row in range(2)
        ]

        stacked_input = mx.concatenate(
            tuple(ticket.input_ids for ticket in custom_tickets),
            axis=0,
        )
        mx.eval(stacked_input)
        input_shape = [int(value) for value in stacked_input.shape]

        custom_runner = MTPK2CohortRunner(lane)
        stock_lane = replace(
            lane,
            width2_target=lane.stock_width2_target,
        )
        stock_runner = MTPK2CohortRunner(stock_lane)
        custom_results = custom_runner.step(tuple(custom_states))
        stock_results = stock_runner.step(tuple(stock_states))
        if len(custom_results) != 2 or len(stock_results) != 2:
            raise RuntimeError(
                "actual-model B2 self-check did not return two request rows"
            )

        isolation_tickets = tuple(
            replace(
                ticket,
                request_cache=own_request_target_cache(
                    lane,
                    ticket.request_cache,
                ),
            )
            for ticket in custom_tickets
        )
        isolation_results = custom_runner.step(
            tuple(
                _selfcheck_request_for_ticket(
                    ticket,
                    request_id=f"selfcheck-order-{row}",
                )
                for row, ticket in enumerate(isolation_tickets)
            )
        )
        if len(isolation_results) != 2:
            raise RuntimeError(
                "actual-model commit-order self-check did not return two rows"
            )

        custom_logits = mx.concatenate(
            tuple(result.logits for result in custom_results),
            axis=0,
        )
        stock_logits = mx.concatenate(
            tuple(result.logits for result in stock_results),
            axis=0,
        )
        custom_hidden = mx.concatenate(
            tuple(result.hidden for result in custom_results),
            axis=0,
        )
        stock_hidden = mx.concatenate(
            tuple(result.hidden for result in stock_results),
            axis=0,
        )
        output_comparisons = [
            _selfcheck_array_comparison(
                mx,
                path="logits",
                candidate=custom_logits,
                reference=stock_logits,
                tolerance=_TARGET_LOGITS_TOLERANCE,
            ),
            _selfcheck_array_comparison(
                mx,
                path="hidden",
                candidate=custom_hidden,
                reference=stock_hidden,
                tolerance=_TARGET_HIDDEN_TOLERANCE,
            ),
        ]
        for row, (candidate, reference) in enumerate(
            zip(custom_results, stock_results, strict=True)
        ):
            output_comparisons.extend(
                _selfcheck_tree_comparisons(
                    mx,
                    path=f"captures.row{row}",
                    candidate=candidate.captures,
                    reference=reference.captures,
                    tolerance=_TARGET_CAPTURE_TOLERANCE,
                )
            )

        cache_layers = _selfcheck_cache_layer_receipts(
            mx,
            lane,
            path="cache",
            candidates=tuple(
                result.request_cache for result in custom_results
            ),
            references=tuple(
                result.request_cache for result in stock_results
            ),
            tolerance=_TARGET_CACHE_TOLERANCE,
        )

        extracted_aliasing = []
        for row in range(2):
            sibling = 1 - row
            row_ids = _selfcheck_cache_array_ids(
                mx,
                custom_results[row].request_cache,
            )
            sibling_ids = _selfcheck_cache_array_ids(
                mx,
                custom_results[sibling].request_cache,
            )
            extracted_aliasing.append(bool(row_ids & sibling_ids))

        mutation_isolation = [False, False]
        for row in range(2):
            sibling = 1 - row
            row_snapshot = snapshot_cache(
                custom_results[row].request_cache
            )
            row_arrays_before = _selfcheck_snapshot_cache(
                mx,
                custom_results[row].request_cache,
            )
            sibling_snapshot = _selfcheck_snapshot_cache(
                mx,
                custom_results[sibling].request_cache,
            )
            _selfcheck_mutate_cache(mx, custom_results[row].request_cache)
            mutation_isolation[row] = _selfcheck_cache_matches_snapshot(
                mx,
                custom_results[sibling].request_cache,
                sibling_snapshot,
            )
            restore_cache(
                custom_results[row].request_cache,
                row_snapshot,
            )
            if not _selfcheck_cache_matches_snapshot(
                mx,
                custom_results[row].request_cache,
                row_arrays_before,
            ):
                raise RuntimeError(
                    f"self-check could not restore extracted row {row} "
                    "after the isolation mutation"
                )

        for row in range(2):
            next_ticket = custom_states[row].resume(custom_results[row])
            if next_ticket is not None or custom_states[row].status != "finished":
                raise RuntimeError(
                    f"self-check candidate row {row} did not finish one cycle"
                )

        for row in (1, 0):
            next_ticket = stock_states[row].resume(stock_results[row])
            if next_ticket is not None or stock_states[row].status != "finished":
                raise RuntimeError(
                    f"self-check reference row {row} did not finish one cycle"
                )

        candidate_row_receipts = [
            {
                "tokens": [
                    int(token)
                    for token in custom_states[row].output.tokens
                ],
                "accepted_drafts": int(
                    custom_states[row].output.stats.accepted_drafts
                ),
            }
            for row in range(2)
        ]
        reference_row_receipts = [
            {
                "tokens": [
                    int(token)
                    for token in stock_states[row].output.tokens
                ],
                "accepted_drafts": int(
                    stock_states[row].output.stats.accepted_drafts
                ),
            }
            for row in range(2)
        ]
        for row, receipt in enumerate(candidate_row_receipts):
            accepted_drafts = int(receipt["accepted_drafts"])
            if accepted_drafts not in (0, 1):
                raise RuntimeError(
                    f"self-check row {row} accepted {accepted_drafts} drafts "
                    "with a one-draft logical output cap"
                )

        isolation_committed: list[list[Any] | None] = [None, None]
        for row in (1, 0):
            isolation_committed[row] = _demote_selfcheck_request_cache(
                isolation_results[row].commit_prefix(
                    1 + int(candidate_row_receipts[row]["accepted_drafts"])
                )
            )
        if any(cache is None for cache in isolation_committed):
            raise RuntimeError("self-check commit-order cache is incomplete")
        commit_order_layers = _selfcheck_cache_layer_receipts(
            mx,
            lane,
            path="commit_order",
            candidates=tuple(
                state.target_cache for state in custom_states
            ),
            references=(
                isolation_committed[0],
                isolation_committed[1],
            ),
            tolerance=0.0,
        )
        commit_isolation = [
            _selfcheck_cache_layer_receipts_pass(
                [
                    layer
                    for layer in commit_order_layers
                    if int(layer["row"]) == row
                ]
            )
            for row in range(2)
        ]

        report: dict[str, Any] = {
            "schema": "qwen27b-mtp-cohort-selfcheck-v1",
            "status": "pass",
            "prefill_chunk_tokens": int(
                custom_states[0].config.prefill_chunk_tokens
            ),
            "prefill_prompt_tokens": {
                str(row): len(prompt_ids)
                for row, prompt_ids in enumerate(prompt_ids_by_row)
            },
            "prefill_spans": prefill_spans,
            "target_cache_reference": "two_owned_clones_per_prefilled_row",
            "qlinear": dict(qlinear_report),
            "target_cycle": {
                "input_shape": input_shape,
                "acceptance_source": (
                    "GenerationOutput.stats.accepted_drafts"
                ),
                "output_comparisons": output_comparisons,
                "starting_cache_layers": starting_cache_layers,
                "starting_cache_aliasing": starting_cache_aliasing,
                "cache_layers": cache_layers,
                "commit_order_layers": commit_order_layers,
                "rows": [
                    {
                        "row": row,
                        "candidate_tokens": candidate_row_receipts[row]["tokens"],
                        "reference_tokens": reference_row_receipts[row]["tokens"],
                        "candidate_accepted_drafts": candidate_row_receipts[row][
                            "accepted_drafts"
                        ],
                        "reference_accepted_drafts": reference_row_receipts[row][
                            "accepted_drafts"
                        ],
                    }
                    for row in range(2)
                ],
                "isolation": [
                    {
                        "row": row,
                        "sibling_row": 1 - row,
                        "extracted_aliasing": extracted_aliasing[row],
                        "sibling_unchanged_after_mutation": mutation_isolation[
                            row
                        ],
                        "sibling_unchanged_after_commit": commit_isolation[row],
                    }
                    for row in range(2)
                ],
            },
        }
        return validate_qwen27b_mtp_cohort_selfcheck_report(report)
    finally:
        for state in (*custom_states, *stock_states):
            state.close()


def _validate_exact_model_identity(runtime: Any, model_path: Path) -> dict[str, Any]:
    if model_path.name != EXPECTED_MODEL_DIRECTORY:
        raise _construction_error(
            model_path,
            f"model path identity must end in {EXPECTED_MODEL_DIRECTORY!r}",
        )
    manifest = _load_json(
        model_path / "MTPLX_PUBLISH_MANIFEST.json",
        invariant="publish manifest is unavailable",
    )
    if manifest.get("repo_id") != EXPECTED_MODEL_ID:
        raise _construction_error(
            model_path,
            f"publish manifest repo_id must be {EXPECTED_MODEL_ID!r}, "
            f"got {manifest.get('repo_id')!r}",
        )
    if not bool(getattr(runtime, "mtp_enabled", False)):
        raise _construction_error(model_path, "native MTP must be enabled")
    return _load_json(
        model_path / "config.json",
        invariant="model config is unavailable",
    )


def _validate_actual_model_inspection(
    runtime: Any,
    model_path: Path,
    deps: SimpleNamespace,
) -> Mapping[str, object]:
    observed = deps.inspect_model_contract(runtime, model_path)
    expected = {
        "backend_id": EXPECTED_BACKEND_ID,
        "architecture_id": EXPECTED_ARCHITECTURE_ID,
        "native_mtp_enabled": True,
        "native_mtp_model_depth_max": EXPECTED_NATIVE_MTP_DEPTH_MAX,
    }
    if observed != expected:
        raise _construction_error(
            model_path,
            f"actual model inspection must be {expected}, got {observed}",
        )
    return MappingProxyType(dict(observed))


def _validate_config(
    runtime: Any,
    model_path: Path,
    config: Mapping[str, Any],
    deps: SimpleNamespace,
) -> object:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise _construction_error(model_path, "text_config must be present")
    exact = {
        "model_type": (config.get("model_type"), "qwen3_5"),
        "text_config.model_type": (text_config.get("model_type"), "qwen3_5_text"),
        "text_config.hidden_size": (text_config.get("hidden_size"), 5120),
        "text_config.num_hidden_layers": (
            text_config.get("num_hidden_layers"),
            EXPECTED_LAYER_COUNT,
        ),
        "text_config.mtp_num_hidden_layers": (
            text_config.get("mtp_num_hidden_layers"),
            1,
        ),
    }
    for name, (observed, expected) in exact.items():
        if observed != expected:
            raise _construction_error(
                model_path,
                f"{name} must be {expected!r}, got {observed!r}",
            )
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise _construction_error(model_path, "target quantization must be present")
    expected_quantization = {
        "bits": EXPECTED_BITS,
        "group_size": EXPECTED_GROUP_SIZE,
        "mode": "affine",
    }
    observed_quantization = {
        "bits": int(quantization.get("bits") or 0),
        "group_size": int(quantization.get("group_size") or 0),
        "mode": str(quantization.get("mode") or ""),
    }
    if observed_quantization != expected_quantization:
        raise _construction_error(
            model_path,
            "target quantization must be q4 affine group_size 64, "
            f"got {observed_quantization}",
        )
    dtype_name = str(text_config.get("dtype") or "").strip().lower()
    if dtype_name not in {"bfloat16", "bf16"}:
        raise _construction_error(
            model_path,
            "activation dtype must match the clean BF16 control receipt, "
            f"got {dtype_name!r}",
        )
    hidden_variant = str(getattr(runtime.contract, "hidden_variant", ""))
    if hidden_variant != EXPECTED_HIDDEN_VARIANT:
        raise _construction_error(
            model_path,
            f"runtime hidden_variant must be {EXPECTED_HIDDEN_VARIANT!r}, "
            f"got {hidden_variant!r}",
        )
    return deps.bfloat16


def _validate_qlinear_layout(
    model_path: Path,
    module_path: str,
    module: Any,
) -> tuple[int, int, Any, Any, Any]:
    bits = int(getattr(module, "bits", 0) or 0)
    group_size = int(getattr(module, "group_size", 0) or 0)
    mode = str(getattr(module, "mode", ""))
    for name, observed, expected in (
        ("bits", bits, EXPECTED_BITS),
        ("group_size", group_size, EXPECTED_GROUP_SIZE),
        ("mode", mode, "affine"),
    ):
        if observed != expected:
            raise _construction_error(
                model_path,
                f"{name} must be {expected!r}, got {observed!r}",
                module_path=module_path,
            )
    try:
        weight = module["weight"]
        scales = module["scales"]
        biases = module["biases"]
        weight_shape = tuple(int(value) for value in weight.shape)
        scales_shape = tuple(int(value) for value in scales.shape)
        biases_shape = tuple(int(value) for value in biases.shape)
    except Exception as exc:
        raise _construction_error(
            model_path,
            f"q4 affine tensors are incomplete: {exc}",
            module_path=module_path,
        ) from exc
    if len(weight_shape) != 2:
        raise _construction_error(
            model_path,
            f"packed weight layout must be rank 2, got {weight_shape}",
            module_path=module_path,
        )
    n = weight_shape[0]
    packed_k = weight_shape[1]
    if packed_k * 32 % bits:
        raise _construction_error(
            model_path,
            f"packed K layout is not divisible by bits={bits}: {weight_shape}",
            module_path=module_path,
        )
    k = packed_k * (32 // bits)
    expected_aux_shape = (n, k // group_size)
    if k % group_size or scales_shape != expected_aux_shape:
        raise _construction_error(
            model_path,
            f"scales layout must be {expected_aux_shape}, got {scales_shape}",
            module_path=module_path,
        )
    if biases_shape != expected_aux_shape:
        raise _construction_error(
            model_path,
            f"biases layout must be {expected_aux_shape}, got {biases_shape}",
            module_path=module_path,
        )
    if str(weight.dtype).lower() not in {"uint32", "mlx.core.uint32"}:
        raise _construction_error(
            model_path,
            f"packed weight dtype must be uint32, got {weight.dtype}",
            module_path=module_path,
        )
    if "bias" in module:
        output_bias_shape = tuple(int(value) for value in module["bias"].shape)
        if output_bias_shape != (n,):
            raise _construction_error(
                model_path,
                f"output bias layout must be {(n,)}, got {output_bias_shape}",
                module_path=module_path,
            )
    return k, n, weight, scales, biases


def install_qwen27b_k2_dual_lane(
    runtime: Any,
    *,
    backend_id: str,
    depth: int,
    verify_strategy: str,
    verify_core: str,
) -> Qwen27BK2DualLane:
    """Validate, self-check, and atomically install the fixed Qwen lane."""

    model_path = Path(runtime.model_path).resolve()
    exact_inputs = (
        ("backend_id", str(backend_id), EXPECTED_BACKEND_ID),
        ("depth", int(depth), EXPECTED_DEPTH),
        ("verify_strategy", str(verify_strategy), EXPECTED_VERIFY_STRATEGY),
        ("verify_core", str(verify_core), EXPECTED_VERIFY_CORE),
    )
    for name, observed, expected in exact_inputs:
        if observed != expected:
            raise _construction_error(
                model_path,
                f"{name} must be {expected!r}, got {observed!r}",
            )

    deps = _construction_dependencies()
    actual_self_check = getattr(deps, "actual_model_self_check", None)
    if not callable(actual_self_check):
        raise _construction_error(
            model_path,
            "actual-model B2 self-check dependency is required before "
            "lane publication",
            module_path="language_model",
        )
    required_capture_dependencies = (
        "resolve_capture_config",
        "configured_capture",
        "bind_capture_commit_route",
        "build_compiled_width2_target",
    )
    missing_capture_dependencies = tuple(
        name
        for name in required_capture_dependencies
        if not callable(getattr(deps, name, None))
    )
    if missing_capture_dependencies:
        raise _construction_error(
            model_path,
            "configured GDN capture dependencies are required: "
            + ", ".join(missing_capture_dependencies),
        )
    config = _validate_exact_model_identity(runtime, model_path)
    actual_inspection = _validate_actual_model_inspection(
        runtime,
        model_path,
        deps,
    )
    activation_dtype = _validate_config(runtime, model_path, config, deps)
    text_model = getattr(runtime.model, "language_model", runtime.model)
    inner = getattr(text_model, "model", text_model)
    layers = tuple(getattr(inner, "layers", ()))
    if len(layers) != EXPECTED_LAYER_COUNT:
        raise _construction_error(
            model_path,
            f"target layer depth must be {EXPECTED_LAYER_COUNT}, got {len(layers)}",
            module_path="model.layers",
        )
    layer_structure_sha256 = _layer_structure_sha256(layers)
    expected_layer_structure_sha256 = getattr(
        deps,
        "expected_layer_structure_sha256",
        None,
    )
    if (
        expected_layer_structure_sha256 is not None
        and layer_structure_sha256 != expected_layer_structure_sha256
    ):
        raise _construction_error(
            model_path,
            "layer structural fingerprint does not match the clean control "
            f"receipt: got {layer_structure_sha256}, "
            f"wanted {expected_layer_structure_sha256}",
            module_path="model.layers",
        )

    try:
        cache_routes = build_qwen27b_cache_routes(
            layers,
            strict_topology=expected_layer_structure_sha256 is not None,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _construction_error(
            model_path,
            f"cache route installation failed: {exc}",
            module_path="model.layers",
        ) from exc
    capture_backend = EXPECTED_VERIFY_CORE.replace("-", "_")
    resolve_capture_config = deps.resolve_capture_config
    configured_capture = deps.configured_capture
    capture_configs = {}
    for target_width in (1, 2):
        try:
            capture_configs[target_width] = resolve_capture_config(
                capture_backend=capture_backend,
                hidden_variant=EXPECTED_HIDDEN_VARIANT,
                model=runtime.model,
                target_width=target_width,
                layers=layers,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _construction_error(
                model_path,
                "configured GDN capture installation failed for "
                f"width {target_width}: {exc}",
                module_path="model.layers",
            ) from exc
    width1_capture_forward = _configured_capture_callable(
        runtime,
        configured_capture=configured_capture,
        config=capture_configs[1],
    )
    width2_capture_forward = _configured_capture_callable(
        runtime,
        configured_capture=configured_capture,
        config=capture_configs[2],
    )
    capture_receipts: Mapping[int, Mapping[str, object]] = MappingProxyType(
        {
            width: _capture_config_receipt(config)
            for width, config in capture_configs.items()
        }
    )
    try:
        capture_commit_routes = MappingProxyType(
            {
                (target_width, row): deps.bind_capture_commit_route(
                    config=capture_configs[target_width],
                    cache_routes=cache_routes,
                    layers=layers,
                    target_width=target_width,
                    row=row,
                    verified_tokens=EXPECTED_DEPTH + 1,
                )
                for target_width, row in ((1, 0), (2, 0), (2, 1))
            }
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _construction_error(
            model_path,
            f"configured GDN capture commit installation failed: {exc}",
            module_path="model.layers",
        ) from exc

    candidates: list[tuple[str, Any, int, int, Any, Any, Any]] = []
    seen_modules: set[int] = set()
    for raw_path, module in text_model.named_modules():
        module_path = str(raw_path)
        module_components = module_path.split(".")
        if (
            id(module) in seen_modules
            or "mtp" in module_components
            or "_mtplx_draft_lm_head" in module_components
        ):
            continue
        if not isinstance(module, deps.quantized_linear_type):
            continue
        seen_modules.add(id(module))
        k, n, weight, scales, biases = _validate_qlinear_layout(
            model_path,
            module_path,
            module,
        )
        if not deps.m6_eligible(
            6,
            k,
            n,
            EXPECTED_BITS,
            EXPECTED_GROUP_SIZE,
            activation_dtype,
        ):
            raise _construction_error(
                model_path,
                "width-2 M6 geometry is ineligible for "
                f"K={k}, N={n}, dtype={activation_dtype}",
                module_path=module_path,
            )
        candidates.append((module_path, module, k, n, weight, scales, biases))

    expected_qlinear_count = int(
        getattr(deps, "expected_qlinear_count", EXPECTED_QLINEAR_COUNT)
    )
    if len(candidates) != expected_qlinear_count:
        raise _construction_error(
            model_path,
            "complete target QuantizedLinear traversal must contain "
            f"{expected_qlinear_count} unique modules, got {len(candidates)}",
            module_path="language_model",
        )

    expected_geometry_histogram = getattr(
        deps,
        "expected_geometry_histogram",
        None,
    )
    observed_geometry_counts = Counter(
        (k, n) for _path, _module, k, n, _weight, _scales, _biases in candidates
    )
    observed_geometry_histogram = tuple(
        (k, n, count)
        for (k, n), count in sorted(observed_geometry_counts.items())
    )
    if (
        expected_geometry_histogram is not None
        and observed_geometry_histogram != tuple(expected_geometry_histogram)
    ):
        raise _construction_error(
            model_path,
            "target QuantizedLinear geometry histogram does not match the "
            f"clean control receipt: got {observed_geometry_histogram}, "
            f"wanted {tuple(expected_geometry_histogram)}",
            module_path="language_model",
        )

    qlinear_structure_sha256 = _qlinear_structure_sha256(candidates)
    expected_qlinear_structure_sha256 = getattr(
        deps,
        "expected_qlinear_structure_sha256",
        None,
    )
    if (
        expected_qlinear_structure_sha256 is not None
        and qlinear_structure_sha256 != expected_qlinear_structure_sha256
    ):
        raise _construction_error(
            model_path,
            "qlinear structural fingerprint does not match the clean control "
            f"receipt: got {qlinear_structure_sha256}, "
            f"wanted {expected_qlinear_structure_sha256}",
            module_path="language_model",
        )

    patch_lease = deps.prepare_patch_lease()
    original = patch_lease.stock_call
    local_routes: dict[int, FixedQLinearRoute] = {}
    geometry_probes: dict[tuple[int, int, object, str], float] = {}
    route_probes: list[dict[str, object]] = []
    for module_path, module, k, n, weight, scales, biases in candidates:
        output_bias = module["bias"] if "bias" in module else None
        qmv_layout = (
            "q4_affine_output_major_packed_u32_m6_qmv_wide_vec6_sg4_lm_head"
            if module_path == "lm_head"
            else "q4_affine_output_major_packed_u32_m6_qmv_wide_vec6_sg2_layer"
        )
        route = FixedQLinearRoute(
            module_path=module_path,
            module_id=id(module),
            k=k,
            n=n,
            bits=EXPECTED_BITS,
            group_size=EXPECTED_GROUP_SIZE,
            activation_dtype=activation_dtype,
            layout=qmv_layout,
            stock_call=partial(_stock_call, original, module),
            width2_call=_prebound_m6_call(
                kernel=deps.nax_qmm_m6_qmv_wide_vec6,
                module=module,
                weight=weight,
                scales=scales,
                biases=biases,
                bits=EXPECTED_BITS,
                group_size=EXPECTED_GROUP_SIZE,
                k=k,
                n=n,
                activation_dtype=activation_dtype,
            ),
            weight=weight,
            scales=scales,
            biases=biases,
            output_bias=output_bias,
        )
        key = (k, n, activation_dtype, route.layout)
        try:
            dmax = float(
                deps.numeric_self_check(module=module, route=route)
            )
        except Exception as exc:
            raise _construction_error(
                model_path,
                f"width-2 M6 numeric self-check raised "
                f"{type(exc).__name__}: {exc}",
                module_path=module_path,
            ) from exc
        if dmax > _QMM_TOLERANCE:
            raise _construction_error(
                model_path,
                f"width-2 M6 numeric self-check exceeded {_QMM_TOLERANCE}: "
                f"dmax={dmax}, K={k}, N={n}, dtype={activation_dtype}",
                module_path=module_path,
            )
        geometry_probes[key] = max(geometry_probes.get(key, 0.0), dmax)
        route_probes.append(
            {
                "module_path": module_path,
                "k": k,
                "n": n,
                "input_shape": [2, 3, k],
                "output_shape": [2, 3, n],
                "dmax": dmax,
            }
        )
        local_routes[id(module)] = route

    if len(local_routes) != len(candidates):
        raise _construction_error(
            model_path,
            "route table lost a QuantizedLinear owner before installation",
            module_path="language_model",
        )

    routes = MappingProxyType(local_routes)
    width1_execution = deps.fixed_execution(routes=routes, width=1)
    width2_execution = deps.fixed_execution(routes=routes, width=2)
    width1_target = _target_callable(
        execution=width1_execution,
        capture_forward=width1_capture_forward,
        fixed_scope=deps.fixed_scope,
    )
    width2_target = deps.build_compiled_width2_target(
        execution=width2_execution,
        capture_forward=width2_capture_forward,
        fixed_scope=deps.fixed_scope,
        cache_routes=cache_routes,
    )
    stock_width2_target = _target_callable(
        execution=width1_execution,
        capture_forward=width2_capture_forward,
        fixed_scope=deps.fixed_scope,
    )
    expected_geometry_receipt = tuple(
        expected_geometry_histogram or observed_geometry_histogram
    )
    qlinear_report: Mapping[str, object] = MappingProxyType(
        {
            "reference": "mx.quantized_matmul_transpose_q4_group64",
            "expected_module_count": expected_qlinear_count,
            "tested_module_count": len(route_probes),
            "expected_shapes": tuple(
                {
                    "k": k,
                    "n": n,
                    "module_count": count,
                }
                for k, n, count in expected_geometry_receipt
            ),
            "tested_shapes": tuple(
                {
                    "k": k,
                    "n": n,
                    "module_count": count,
                }
                for k, n, count in observed_geometry_histogram
            ),
            "routes": tuple(route_probes),
        }
    )
    receipt_values: dict[str, object] = {
            "model_id": EXPECTED_MODEL_ID,
            "model_path": str(model_path),
            "backend_id": EXPECTED_BACKEND_ID,
            "target_layer_count": EXPECTED_LAYER_COUNT,
            "qlinear_module_count": len(routes),
            "actual_model_qlinear_module_count": EXPECTED_QLINEAR_COUNT,
            "actual_model_inspection": actual_inspection,
            "layer_structure_sha256": layer_structure_sha256,
            "qlinear_structure_sha256": qlinear_structure_sha256,
            "control_qlinear_geometry_histogram": (
                EXPECTED_QLINEAR_GEOMETRY_HISTOGRAM
            ),
            "activation_dtype_from_config": "bfloat16",
            "numeric_probe_dtype": str(activation_dtype),
            "qlinear_patch_was_dynamic": bool(
                patch_lease.initially_dynamic
            ),
            "qlinear_geometries": tuple(
                {
                    "k": k,
                    "n": n,
                    "dtype": str(dtype),
                    "layout": layout,
                    "numeric_dmax": dmax,
                }
                for (k, n, dtype, layout), dmax in geometry_probes.items()
            ),
            "post_prefill_cache_types": EXPECTED_POST_PREFILL_CACHE_TYPES,
            "cache_route_status": "installed",
            "cache_route_count": len(cache_routes),
            "recurrent_cache_route_count": sum(
                route.request_type.__name__ == "OwnedRecurrentStateCache"
                for route in cache_routes
            ),
            "attention_cache_route_count": sum(
                route.request_type.__name__ == "KVCache"
                for route in cache_routes
            ),
            "gdn_verify_config": capture_receipts[2],
            "gdn_verify_config_by_width": capture_receipts,
            "width2_execution": "compiled_explicit_cache_state",
            "qlinear_selfcheck": qlinear_report,
    }
    receipt = MappingProxyType(receipt_values)
    lane = Qwen27BK2DualLane(
        backend_id=EXPECTED_BACKEND_ID,
        depth=EXPECTED_DEPTH,
        bits=EXPECTED_BITS,
        group_size=EXPECTED_GROUP_SIZE,
        activation_dtype=activation_dtype,
        hidden_variant=EXPECTED_HIDDEN_VARIANT,
        verify_strategy=EXPECTED_VERIFY_STRATEGY,
        verify_core=EXPECTED_VERIFY_CORE,
        max_width=2,
        width1_target=width1_target,
        width2_target=width2_target,
        cache_routes=cache_routes,
        qlinear_routes=routes,
        construction_receipt=receipt,
        width1_execute_ticket=partial(
            deps.execute_width1_ticket,
            runtime,
        ),
        capture_commit_routes=capture_commit_routes,
        stock_width2_target=stock_width2_target,
        qlinear_patch_lease=patch_lease,
    )
    patch_lease.acquire()
    try:
        selfcheck_report = actual_self_check(
            runtime,
            lane,
            qlinear_report=qlinear_report,
        )
        reported_qlinear = _selfcheck_mapping(
            _selfcheck_mapping(selfcheck_report, "report").get("qlinear"),
            "qlinear",
        )
        if dict(reported_qlinear) != dict(qlinear_report):
            raise ValueError(
                "self-check qlinear receipt differs from the construction "
                "route table"
            )
        selfcheck_report = validate_qwen27b_mtp_cohort_selfcheck_report(
            selfcheck_report
        )
        width2_target.release_construction_state()
    except BaseException as exc:
        patch_lease.release()
        raise _construction_error(
            model_path,
            "actual-model B2 self-check failed: "
            f"{type(exc).__name__}: {exc}",
            module_path="language_model",
        ) from exc
    final_receipt = MappingProxyType(
        {
            **receipt_values,
            "actual_model_selfcheck": selfcheck_report,
        }
    )
    final_lane = replace(lane, construction_receipt=final_receipt)
    runtime.qwen27b_k2_dual_lane = final_lane
    return final_lane
