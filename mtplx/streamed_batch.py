"""Continuous batching for the streamed AR decode path (roadmap Stage 5).

This runner drives several independent AR streams through one streamed-expert
model.  Each decode step evaluates every live stream as a single ``[B, 1, H]``
forward, so at every sparse layer the router presents the union of the
selected experts across live sequences to the existing wave/split-route
machinery: a record selected by several streams in the same step is planned,
loaded, and hashed once (``LayerExpertSlotBank.plan`` deduplicates unique
experts; ``partition_route_waves`` partitions unique experts across waves).

Sequence isolation is structural rather than masked: every stream owns its own
per-layer KV cache list, and only the position-wise MLP/expert work is
evaluated batched.  Attention for each stream runs on its own cache with its
own offsets, which keeps prompt positions exact without left-padding and
releases a stream's whole KV allocation the moment it finishes.

Determinism note: with one live stream the runner performs the same MLX
operations as :func:`mtplx.generation.generate_ar` and produces byte-identical
tokens.  With two or more live streams the batched kernels see different
shapes than a single-stream run of the same prompt, so ``B > 1`` outputs are
NOT guaranteed token-identical to ``B = 1`` runs; batch size is part of the
run configuration label and results must only be compared at equal batch
sizes.

Scope: AR only (the pinned streamed artifacts omit MTP weights) and the
Hy3-style pre-norm decoder layout.  The constructor fails closed on anything
else.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from mlx_lm.models.base import create_attention_mask

from .attention_context import attention_phase
from .expert_runtime import ExpertStreamingConfigurationError
from .expert_streaming import RoutingPhase
from .generation import (
    _decode,
    _default_stop_tokens,
    _eval,
    _finish_reason_from_tokens,
    _is_stop,
    _prefill,
    _sample_from_logits,
    _strip_terminal_stop,
)
from .models.expert_mlx import expert_routing_phase
from .sampling import SamplerConfig


class StreamedBatchError(RuntimeError):
    """A batching invariant (admission, structure, progress) was violated."""


@dataclass(frozen=True)
class StreamedBatchRequest:
    """One AR generation request for the streamed batch runner."""

    request_id: str
    prompt_ids: tuple[int, ...]
    max_tokens: int
    sampler: SamplerConfig
    seed: int = 0
    stop_token_ids: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise TypeError("request_id must be a non-empty string")
        prompt = tuple(int(token) for token in self.prompt_ids)
        if not prompt:
            raise ValueError("prompt_ids must not be empty")
        object.__setattr__(self, "prompt_ids", prompt)
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an exact integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not isinstance(self.sampler, SamplerConfig):
            raise TypeError("sampler must be a SamplerConfig")
        if self.stop_token_ids is not None:
            object.__setattr__(
                self,
                "stop_token_ids",
                frozenset(int(token) for token in self.stop_token_ids),
            )

    @property
    def kv_tokens(self) -> int:
        """Tokens reserved for the stream's whole lifetime, prompt included."""

        return len(self.prompt_ids) + self.max_tokens


@dataclass(frozen=True)
class StreamedBatchResult:
    """Per-stream outcome plus the timing needed for per-stream tok/s."""

    request_id: str
    tokens: tuple[int, ...]
    text: str
    finish_reason: str
    prompt_tokens: int
    admitted_step: int
    finished_step: int
    decode_steps: int
    prefill_seconds: float
    admitted_s: float
    first_token_s: float
    last_token_s: float
    token_times_s: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tokens": list(self.tokens),
            "text": self.text,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "admitted_step": self.admitted_step,
            "finished_step": self.finished_step,
            "decode_steps": self.decode_steps,
            "prefill_seconds": self.prefill_seconds,
            "admitted_s": self.admitted_s,
            "first_token_s": self.first_token_s,
            "last_token_s": self.last_token_s,
            "token_times_s": list(self.token_times_s),
        }


@dataclass(frozen=True)
class StreamedBatchStreamView:
    """Read-only view of one live stream at a step boundary."""

    request_id: str
    generated_tokens: int
    cache_offset: int
    reserved_kv_tokens: int


@dataclass(frozen=True)
class StreamedBatchStepState:
    """Step-boundary snapshot handed to the ``on_step`` hook."""

    step: int
    live: tuple[StreamedBatchStreamView, ...]
    pending: tuple[str, ...]
    reserved_kv_tokens: int


class _LiveStream:
    __slots__ = (
        "request",
        "rng",
        "stop_token_ids",
        "cache",
        "admission",
        "tokens",
        "token_times_s",
        "finish_reason",
        "admitted_step",
        "decode_steps",
        "prefill_seconds",
        "admitted_s",
    )

    def __init__(
        self,
        request: StreamedBatchRequest,
        *,
        stop_token_ids: set[int],
        admission: Any | None,
        admitted_step: int,
        admitted_s: float,
    ) -> None:
        self.request = request
        self.rng = np.random.default_rng(request.seed)
        self.stop_token_ids = stop_token_ids
        self.cache: Any | None = None
        self.admission = admission
        self.tokens: list[int] = []
        self.token_times_s: list[float] = []
        self.finish_reason: str | None = None
        self.admitted_step = admitted_step
        self.decode_steps = 0
        self.prefill_seconds = 0.0
        self.admitted_s = admitted_s

    def sample(self, logits_row: mx.array) -> int:
        """Mirror ``generate_ar``'s per-step sampling exactly."""

        sampler = self.request.sampler
        token, _ = _sample_from_logits(
            logits_row,
            sampler,
            self.rng,
            token_counts=(
                Counter(self.tokens)
                if (sampler.presence_penalty or sampler.frequency_penalty)
                else None
            ),
        )
        self.tokens.append(int(token))
        self.token_times_s.append(time.perf_counter())
        if _is_stop(int(token), self.stop_token_ids):
            self.finish_reason = "stop"
        elif len(self.tokens) >= self.request.max_tokens:
            self.finish_reason = "length"
        return int(token)

    def release(self) -> None:
        if self.admission is not None:
            self.admission.release()
            self.admission = None
        self.cache = None

    def cache_offset(self) -> int:
        if self.cache is None:
            return 0
        first = self.cache[0]
        return int(getattr(first, "offset", 0))

    def reserved_kv_tokens(self) -> int:
        if self.admission is None:
            return 0
        return int(self.admission.tokens)


def _validate_decoder_layout(model: Any) -> None:
    """Fail closed unless the model exposes the Hy3-style pre-norm layout."""

    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if (
        layers is None
        or getattr(inner, "embed_tokens", None) is None
        or getattr(inner, "norm", None) is None
        or getattr(model, "lm_head", None) is None
    ):
        raise TypeError(
            "streamed batch decode requires model.model.{embed_tokens,layers,norm} "
            "and model.lm_head"
        )
    for index, layer in enumerate(layers):
        for attribute in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "mlp",
        ):
            if getattr(layer, attribute, None) is None:
                raise TypeError(
                    f"streamed batch decode requires layer {index} to expose "
                    f"{attribute}; only the Hy3-style pre-norm decoder layout "
                    "is supported"
                )


class StreamedBatchRunner:
    """Admit, prefill, and step several streamed AR sequences together.

    Scheduling model:

    - Requests are admitted strictly in submission order at decode step
      boundaries only.  Admission reserves the stream's full KV budget
      (``prompt + max_tokens``) through ``admit_kv_tokens`` before any
      forward runs and fails closed if a request can never fit the plan.
    - While decoders are live, at most ``max_prefills_per_step`` joining
      prefills run per boundary, so active streams are stalled by at most
      that bounded prefill work at a single step boundary.  When nothing is
      decoding, pending requests fill the batch in one boundary (static
      batch start).
    - A joining prefill reuses the single-stream ``_prefill`` path, so its
      sparse routes carry the prefill phase: the slot banks serve prefill
      misses from transient slots and never evict a decode-hot persistent
      expert (see ``LayerExpertSlotBank.plan``).
    - A stream that finishes releases its KV admission and drops its cache
      at that same step boundary; the remaining streams continue.
    """

    def __init__(
        self,
        rt: Any,
        *,
        max_concurrency: int,
        max_prefills_per_step: int = 1,
        on_step: Callable[[StreamedBatchStepState], None] | None = None,
    ) -> None:
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise TypeError("max_concurrency must be an exact integer")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if isinstance(max_prefills_per_step, bool) or not isinstance(
            max_prefills_per_step, int
        ):
            raise TypeError("max_prefills_per_step must be an exact integer")
        if max_prefills_per_step < 1:
            raise ValueError("max_prefills_per_step must be at least 1")
        if getattr(rt, "mtp_enabled", False):
            raise StreamedBatchError(
                "the streamed batch runner is AR-only; load with mtp=False"
            )
        _validate_decoder_layout(rt.model)
        self._rt = rt
        self.max_concurrency = max_concurrency
        self.max_prefills_per_step = max_prefills_per_step
        self._on_step = on_step
        self._pending: deque[StreamedBatchRequest] = deque()
        self._live: list[_LiveStream] = []
        self._results: dict[str, StreamedBatchResult] = {}
        self._order: list[str] = []
        self._step = 0
        self._running = False
        self._step_live_counts: list[int] = []

    def submit(self, request: StreamedBatchRequest) -> None:
        """Queue a request; joins at the next decode step boundary."""

        if not isinstance(request, StreamedBatchRequest):
            raise TypeError("request must be a StreamedBatchRequest")
        known = set(self._order)
        if request.request_id in known:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        self._order.append(request.request_id)
        self._pending.append(request)

    def run(self) -> list[StreamedBatchResult]:
        """Drive all submitted (and newly submitted) requests to completion."""

        if self._running:
            raise StreamedBatchError("runner is already running")
        if not self._pending:
            raise StreamedBatchError("no requests were submitted")
        self._running = True
        try:
            while self._pending or self._live:
                self._admit_pending()
                self._finalize_finished()
                if self._on_step is not None:
                    self._on_step(self._step_state())
                if not self._live:
                    if self._pending:
                        # Nothing is decoding and nothing could be admitted:
                        # no future step boundary can free KV, so waiting
                        # would hang. Fail closed instead.
                        raise StreamedBatchError(
                            "no live streams and the next pending request "
                            "cannot be admitted under the KV plan"
                        )
                    break
                self._step_live_counts.append(len(self._live))
                self._decode_step()
                self._step += 1
                self._finalize_finished()
        finally:
            self._running = False
            for stream in self._live:
                stream.release()
            self._live = []
        return [self._results[request_id] for request_id in self._order]

    def stats(self) -> dict[str, Any]:
        steps = len(self._step_live_counts)
        return {
            "decode_steps": steps,
            "live_stream_counts": list(self._step_live_counts),
            "mean_live_streams": (
                sum(self._step_live_counts) / steps if steps else 0.0
            ),
        }

    # -- scheduling -----------------------------------------------------

    def _step_state(self) -> StreamedBatchStepState:
        live = tuple(
            StreamedBatchStreamView(
                request_id=stream.request.request_id,
                generated_tokens=len(stream.tokens),
                cache_offset=stream.cache_offset(),
                reserved_kv_tokens=stream.reserved_kv_tokens(),
            )
            for stream in self._live
        )
        return StreamedBatchStepState(
            step=self._step,
            live=live,
            pending=tuple(request.request_id for request in self._pending),
            reserved_kv_tokens=sum(view.reserved_kv_tokens for view in live),
        )

    def _admit_pending(self) -> None:
        # Whether decoders were active is decided once per boundary: an empty
        # batch fills to capacity in one boundary (static-batch start), while
        # joining prefills next to live decoders are bounded per boundary.
        decoders_active = bool(self._live)
        admitted = 0
        while self._pending and len(self._live) < self.max_concurrency:
            if decoders_active and admitted >= self.max_prefills_per_step:
                break
            request = self._pending[0]
            admission = self._try_admit_kv(request)
            if admission is None and self._rt.expert_streaming is not None:
                # Head-of-line request must wait for a stream to release KV.
                break
            self._pending.popleft()
            stream = _LiveStream(
                request,
                stop_token_ids=(
                    set(request.stop_token_ids)
                    if request.stop_token_ids is not None
                    else _default_stop_tokens(self._rt.tokenizer)
                ),
                admission=admission,
                admitted_step=self._step,
                admitted_s=time.perf_counter(),
            )
            try:
                self._prefill_stream(stream)
            except BaseException:
                stream.release()
                raise
            self._live.append(stream)
            admitted += 1

    def _try_admit_kv(self, request: StreamedBatchRequest) -> Any | None:
        streaming = self._rt.expert_streaming
        if streaming is None:
            return None
        if request.kv_tokens > streaming.config.max_live_kv_tokens:
            raise StreamedBatchError(
                f"request {request.request_id!r} needs {request.kv_tokens} KV "
                f"tokens but the plan admits at most "
                f"{streaming.config.max_live_kv_tokens}"
            )
        try:
            return streaming.admit_kv_tokens(request.kv_tokens)
        except ExpertStreamingConfigurationError as exc:
            if not self._live:
                raise StreamedBatchError(
                    f"request {request.request_id!r} cannot be admitted and no "
                    "live stream can release KV tokens"
                ) from exc
            return None

    def _prefill_stream(self, stream: _LiveStream) -> None:
        # The single-stream prefill path keeps a joining stream's routing in
        # the prefill phase (transient-slot service, no decode-hot eviction)
        # and its cache/logits byte-identical to generate_ar's own prefill.
        cache, last_logits, _hidden, prefill_seconds = _prefill(
            self._rt,
            list(stream.request.prompt_ids),
            return_hidden=False,
        )
        stream.cache = cache
        stream.prefill_seconds = prefill_seconds
        stream.sample(last_logits[0])

    # -- batched decode ---------------------------------------------------

    def _decode_step(self) -> None:
        streams = self._live
        model = self._rt.model
        inner = model.model
        batch = mx.array([[stream.tokens[-1]] for stream in streams])
        with attention_phase("ar_decode"), expert_routing_phase(RoutingPhase.DECODE):
            hidden = inner.embed_tokens(batch)
            if getattr(model, "model_type", None) == "glm_moe_dsa":
                hidden = self._glm_layer_walk(inner, hidden, streams)
            else:
                hidden = self._split_attention_layer_walk(inner, hidden, streams)
            hidden = inner.norm(hidden)
            if getattr(getattr(model, "args", None), "enable_lm_head_fp32", False):
                hidden = hidden.astype(mx.float32)
            logits = model.lm_head(hidden)
        _eval(logits)
        for position, stream in enumerate(streams):
            stream.decode_steps += 1
            stream.sample(logits[position, -1, :])

    @staticmethod
    def _split_attention_layer_walk(
        inner: Any,
        hidden: mx.array,
        streams: list["_LiveStream"],
    ) -> mx.array:
        """Hy3-shaped layers: per-stream attention, batched MLP.

        Attention runs per row (each stream owns its cache and mask) while
        the MLP sees all rows at once so the expert runtime loads each
        unique routed expert a single time per layer.
        """

        for layer_index, layer in enumerate(inner.layers):
            normed = layer.input_layernorm(hidden)
            attn_rows = []
            for position, stream in enumerate(streams):
                row = normed[position : position + 1]
                layer_cache = stream.cache[layer_index]
                mask = create_attention_mask(row, layer_cache)
                attn_rows.append(layer.self_attn(row, mask, layer_cache))
            attended = (
                attn_rows[0]
                if len(attn_rows) == 1
                else mx.concatenate(attn_rows, axis=0)
            )
            hidden = hidden + attended
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden

    @staticmethod
    def _glm_layer_walk(
        inner: Any,
        hidden: mx.array,
        streams: list["_LiveStream"],
    ) -> mx.array:
        """GLM MoE-DSA layers: run each stream through the layer forward.

        GLM attention returns ``(hidden, topk_indices)`` and threads the
        DSA indexer's top-k across layers, its per-layer cache is a
        ``CacheList`` (mask reads entry 0), and the layer forward owns the
        residual/per-row arithmetic — hand-rolling the hy3 split here is
        what produced 'array + tuple' failures. Per-stream full-layer
        calls keep the contract intact; grouping routed experts across
        streams within a layer is a future batching optimization.
        """

        rows = [hidden[position : position + 1] for position in range(len(streams))]
        stream_topk: list[Any] = [None] * len(streams)
        for layer_index, layer in enumerate(inner.layers):
            for position, stream in enumerate(streams):
                layer_cache = stream.cache[layer_index]
                mask = create_attention_mask(
                    rows[position],
                    layer_cache[0] if layer_cache else None,
                    return_array=True,
                )
                rows[position], stream_topk[position] = layer(
                    rows[position],
                    mask,
                    layer_cache,
                    stream_topk[position],
                )
        return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)

    def _finalize_finished(self) -> None:
        remaining: list[_LiveStream] = []
        for stream in self._live:
            if stream.finish_reason is None:
                remaining.append(stream)
                continue
            stream.release()
            request = stream.request
            tokens = tuple(stream.tokens)
            self._results[request.request_id] = StreamedBatchResult(
                request_id=request.request_id,
                tokens=tokens,
                text=_decode(
                    self._rt.tokenizer,
                    _strip_terminal_stop(list(tokens), stream.stop_token_ids),
                ),
                finish_reason=_finish_reason_from_tokens(
                    list(tokens),
                    stop_token_ids=stream.stop_token_ids,
                    max_tokens=request.max_tokens,
                ),
                prompt_tokens=len(request.prompt_ids),
                admitted_step=stream.admitted_step,
                finished_step=self._step,
                decode_steps=stream.decode_steps,
                prefill_seconds=stream.prefill_seconds,
                admitted_s=stream.admitted_s,
                first_token_s=stream.token_times_s[0],
                last_token_s=stream.token_times_s[-1],
                token_times_s=tuple(stream.token_times_s),
            )
        self._live = remaining
