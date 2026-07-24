"""High-level MTPLX runtime loading primitives."""

from __future__ import annotations

import hashlib
import inspect as py_inspect
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import inspect_model, load_config
from .mtp_adapters import (
    install_saved_mtp_lora_adapter,
    merge_installed_mtp_lora_adapters,
    mtp_adapter_depth,
)
from .mtp_patch import MTPContract, inject_mtp_support, validate_mtp_support

logger = logging.getLogger(__name__)


def _streamed_mtp_backend(model_key: str, precision: str) -> str:
    """Resolve the strict external MTP adapter before model allocation."""

    support = {
        "hy3-q4": ("hy3", {"bf16", "q4"}),
        "hy3-expert-only-q4": ("hy3", {"bf16"}),
        "hy3-expert-q2": ("hy3", {"bf16"}),
        "hy3-expert-oq2e": ("hy3", {"bf16"}),
        # The Q4 head sibling (issue #100) is lane-agnostic: it is selectable
        # for every GLM streamed lane: q2 and q1t have the same resident trunk,
        # router, and layer-78 contract; only routed record storage differs.
        "glm52-expert-q2": ("glm52", {"bf16", "q4"}),
        "glm52-expert-q1t": ("glm52", {"bf16", "q4"}),
        "glm52-q4": ("glm52", {"bf16", "q4"}),
    }
    selected = support.get(str(model_key))
    if selected is None:
        raise RuntimeError(f"streamed MTP is not supported for model key {model_key!r}")
    backend, precisions = selected
    if precision not in precisions:
        if precisions == {"bf16"}:
            raise RuntimeError(
                f"streamed MTP for {model_key!r} requires the validated BF16 head"
            )
        raise RuntimeError(
            f"streamed MTP precision {precision!r} is not supported for {model_key!r}"
        )
    return backend


@dataclass
class MTPLXRuntime:
    model: Any
    tokenizer: Any
    model_path: Path
    mtp_enabled: bool
    contract: MTPContract
    mtp_adapter_path: Path | None = None
    mtp_adapter_metadata: dict[str, Any] | None = None
    mtp_adapter_merge_report: dict[str, Any] | None = None
    expert_streaming: Any | None = None
    resident_load_report: dict[str, Any] | None = None
    diagnostic_counters: dict[str, int] = field(default_factory=dict)
    _forward_ar_supports_emit_logits: bool | None = field(
        default=None, init=False, repr=False
    )
    _forward_ar_supports_logits_keep: bool | None = field(
        default=None, init=False, repr=False
    )
    _expert_compiled_ar_enabled: bool = field(default=False, init=False, repr=False)

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = int(self.diagnostic_counters.get(key, 0)) + int(
            amount
        )

    @staticmethod
    def _sequence_len(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", ())
        if len(shape) >= 2:
            return int(shape[1])
        if shape:
            return int(shape[0])
        return 1

    def _forward_ar_capabilities(self) -> tuple[bool, bool]:
        if (
            self._forward_ar_supports_emit_logits is None
            or self._forward_ar_supports_logits_keep is None
        ):
            try:
                params = py_inspect.signature(self.model.__call__).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            patched_kwargs = bool(self.mtp_enabled and accepts_kwargs)
            self._forward_ar_supports_emit_logits = (
                "emit_logits" in params or patched_kwargs
            )
            self._forward_ar_supports_logits_keep = (
                "logits_keep" in params or patched_kwargs
            )
        return (
            bool(self._forward_ar_supports_emit_logits),
            bool(self._forward_ar_supports_logits_keep),
        )

    def embed_tokens(self, input_ids):
        """Embed token ids with the text model's embedding table."""

        text_model = getattr(self.model, "language_model", self.model)
        return text_model.model.embed_tokens(input_ids)

    def _expert_routing_context(self, input_ids: Any):
        if self.expert_streaming is None:
            return nullcontext()
        from .attention_context import current_attention_phase
        from .expert_streaming import RoutingPhase
        from .models.expert_mlx import expert_routing_phase

        attention = current_attention_phase()
        if attention == "prefill":
            # A one-token prefill tail chunk is still prefill traffic: the
            # width heuristic below would classify it as decode and pollute
            # the persistent decode hot set.
            return expert_routing_phase(RoutingPhase.PREFILL)
        if attention in {"ar_decode", "decode_verify", "postcommit"}:
            # MTP verify batches are decode traffic regardless of width.
            return expert_routing_phase(RoutingPhase.DECODE)

        decode_width = 1
        if self.mtp_enabled:
            # MTP verify batches are decode traffic: routing them as prefill
            # would stop the persistent decode hot set from ever training
            # once speculation is on.  With MTP off this stays exactly the
            # historical single-token decode classification.
            decode_width = max(
                decode_width,
                int(getattr(self.model, "mtp_verify_width", 1)),
            )
        phase = (
            RoutingPhase.PREFILL
            if self._sequence_len(input_ids) > decode_width
            else RoutingPhase.DECODE
        )
        return expert_routing_phase(phase)

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        self._count(
            "forward_ar_hidden_calls" if return_hidden else "forward_ar_plain_calls"
        )
        if not self.mtp_enabled and return_hidden:
            raise RuntimeError("return_hidden requires an MTP-patched runtime")
        if input_embeddings is not None and not self.mtp_enabled:
            raise RuntimeError("vision splice requires the MTP-patched runtime")
        kwargs = {}
        if hidden_variant is not None:
            kwargs["hidden_variant"] = hidden_variant
        if input_embeddings is not None:
            # Vision splice path: the patched text model takes the rows
            # directly; ids still travel for mask construction.
            kwargs["input_embeddings"] = input_embeddings
        supports_emit_logits, supports_logits_keep = self._forward_ar_capabilities()
        if supports_emit_logits:
            kwargs["emit_logits"] = bool(emit_logits)
        elif not emit_logits:
            self._count("forward_ar_emit_logits_unsupported")
        if logits_keep is not None and supports_logits_keep:
            kwargs["logits_keep"] = int(logits_keep)
        elif logits_keep is not None:
            self._count("forward_ar_logits_keep_unsupported")
        sequence_len = self._sequence_len(input_ids)
        if bool(emit_logits) or not supports_emit_logits:
            if logits_keep is not None and supports_logits_keep:
                emitted = min(sequence_len, max(1, int(logits_keep)))
            else:
                emitted = sequence_len
            self._count("logits_tokens_emitted", emitted)
            if emitted == 1:
                self._count("final_logits_tokens_emitted", 1)
            else:
                self._count("full_logits_tokens_emitted", emitted)
        with self._expert_routing_context(input_ids):
            # kwargs == {"emit_logits": True} is semantically the plain call —
            # MTP-patched wrappers advertise emit_logits via **kwargs, so on MTP
            # runtimes the bare-kwargs case never occurs and the compiled hook
            # must accept the default-emit form too.
            plain_call = not kwargs or (
                set(kwargs) == {"emit_logits"} and kwargs["emit_logits"] is True
            )
            if not return_hidden and hidden_variant is None and plain_call:
                # Decode-only (seq_len == 1). Prefill is multi-token over an
                # unprimed cache: seeding the compiled graph from its None KV
                # leaves throws, and its shape differs from a single-token
                # decode step, forcing a retrace. Prefill stays eager.
                compiled = (
                    self._compiled_ar_forward(cache) if sequence_len == 1 else None
                )
                if compiled is not None:
                    # Engagement proof: arm A (flag off) must report 0 here,
                    # arm B (on) > 0 — the A/B credits nothing without it.
                    self._count("compiled_forward_calls")
                    return compiled(input_ids, cache)
                if not kwargs:
                    return self.model(input_ids, cache=cache)
            return self.model(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                **kwargs,
            )

    def _compiled_ar_forward(self, cache):
        """Compiled target forward (MTPLX_COMPILE_AR_FORWARD).

        Kills the per-token Python graph rebuild by tracing the full trunk
        forward once (CompiledARForward, KV state threaded). Applies to
        fully-resident loads with a standard per-layer KV cache; a host-sync
        buried in the model forward surfaces as an error on the first traced
        call rather than silently degrading. Rebuilds per cache identity so a
        new generation gets fresh threaded state. Returns None (the eager
        path) otherwise.
        """
        from .compiled_forward import CompiledARForward, compile_forward_enabled

        if self.expert_streaming is None:
            if not compile_forward_enabled():
                return None
        elif not self._expert_compiled_ar_enabled:
            return None
        if not cache:
            return None
        # An unprimed cache (empty context / first token) has None KV leaves
        # that would crash the compiled graph. Only compile once the cache
        # holds real keys, and only for the plain growable KVCache shape the
        # fixed-buffer conversion understands.
        first = cache[0]
        if getattr(first, "keys", None) is None:
            return None
        if any(
            not hasattr(entry, "keys")
            or not hasattr(entry, "values")
            or not hasattr(entry, "offset")
            for entry in cache
        ):
            return None
        cache_key = id(first)
        if (
            getattr(self, "_compiled_ar", None) is None
            or getattr(self, "_compiled_ar_key", None) != cache_key
        ):
            if self.expert_streaming is not None:
                reserve = int(self.expert_streaming.config.max_live_kv_tokens)
            else:
                import os as _os

                reserve = int(
                    _os.environ.get("MTPLX_COMPILE_AR_RESERVE_TOKENS", "4096")
                )
            self._compiled_ar = CompiledARForward(self.model, reserve_tokens=reserve)
            self._compiled_ar_key = cache_key
        return self._compiled_ar

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        text_model = getattr(self.model, "language_model", self.model)
        inner = getattr(text_model, "model", None)
        if not (hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx")):
            # Uniform full-attention model (e.g. hy_v3): every layer is plain
            # causal attention, so there is no GDN/recurrent state to capture
            # and forward_with_gdn_capture's hybrid layout (fa_idx/ssm_idx,
            # layer.is_linear) does not exist. The verify forward is just the
            # plain AR forward; commit_captured_prefix with empty captures
            # commits by trimming the standard (trimmable) KV caches, which is
            # the correct prefix commit for pure-attention layers.
            self._count("forward_ar_capture_plain_attention_calls")
            if return_hidden:
                logits, hidden = self.forward_ar(
                    input_ids,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant=hidden_variant,
                )
                return logits, hidden, {}
            logits = self.forward_ar(input_ids, cache=cache)
            return logits, {}

        from .gdn_capture import forward_with_gdn_capture

        with self._expert_routing_context(input_ids):
            return forward_with_gdn_capture(
                self.model,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                capture_backend=capture_backend,
            )

    def draft_mtp(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        mtp_depth: int | None = None,
        position_offset: int | None = None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("draft_mtp_calls")
        resolved_hidden_variant = (
            self.contract.hidden_variant
            if mtp_hidden_variant in {None, "auto", "contract"}
            else str(mtp_hidden_variant)
        )
        resolved_concat_order = (
            self.contract.concat_order
            if concat_order in {None, "auto", "contract"}
            else concat_order
        )
        with mtp_adapter_depth(self.model, mtp_depth):
            kwargs = {
                "mtp_cache": mtp_cache,
                "concat_order": resolved_concat_order,
                "return_hidden": return_hidden,
                "mtp_hidden_variant": resolved_hidden_variant,
                "position_offset": position_offset,
            }
            try:
                params = py_inspect.signature(self.model.mtp_forward).parameters
            except Exception:
                params = {}
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = mtp_depth
            return self.model.mtp_forward(hidden_states, next_token_ids, **kwargs)

    def configure_mtp_execution_depth(self, depth: int | None) -> Any | None:
        """Select any depth-gated MTP operators before prefill/decode starts."""

        if depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int)
        ):
            raise TypeError("MTP execution depth must be an integer or None")
        if depth is not None and depth < 1:
            raise ValueError("MTP execution depth must be positive")
        configure = getattr(self.model, "configure_mtp_execution_depth", None)
        if not callable(configure):
            return None
        self._count("mtp_execution_depth_configurations")
        return configure(depth)

    def update_mtp_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        input_embeddings=None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("update_mtp_cache_calls")
        resolved_hidden_variant = (
            self.contract.hidden_variant
            if mtp_hidden_variant in {None, "auto", "contract"}
            else str(mtp_hidden_variant)
        )
        resolved_concat_order = (
            self.contract.concat_order
            if concat_order in {None, "auto", "contract"}
            else concat_order
        )
        update = getattr(self.model, "mtp_update_cache", None)
        if update is not None:
            try:
                params = py_inspect.signature(update).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            candidates = {
                "mtp_cache": mtp_cache,
                "concat_order": resolved_concat_order,
                "mtp_hidden_variant": resolved_hidden_variant,
                "position_offset": position_offset,
                "input_embeddings": input_embeddings,
            }
            kwargs = {
                key: value
                for key, value in candidates.items()
                if accepts_kwargs or key in params
            }
            if input_embeddings is not None and "input_embeddings" not in kwargs:
                # Silently dropping the spliced vision rows would rebuild the
                # exact draft-history corruption this parameter fixes (#103).
                raise RuntimeError(
                    "this MTP backend does not accept input_embeddings; "
                    "vision history append is unsupported for it"
                )
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = None
            return update(hidden_states, next_token_ids, **kwargs)
        if input_embeddings is not None:
            raise RuntimeError(
                "mtp_forward fallback does not accept input_embeddings; "
                "vision history append is unsupported for it"
            )
        _logits, hidden = self.model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache=mtp_cache,
            concat_order=resolved_concat_order,
            return_hidden=True,
            mtp_hidden_variant=resolved_hidden_variant,
            position_offset=position_offset,
        )
        return hidden

    def make_cache(self):
        inner = getattr(self.model, "language_model", self.model)
        cache = inner.make_cache()
        from .cache_state import (
            configure_owned_recurrent_state_cache,
            configure_tail_owned_attention_kv_cache,
        )

        configure_owned_recurrent_state_cache(cache)
        configure_tail_owned_attention_kv_cache(cache)
        return cache

    def make_mtp_cache(self):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("make_mtp_cache_calls")
        cache = self.model.make_mtp_cache()
        from .cache_state import configure_mtp_attention_kv_cache

        configure_mtp_attention_kv_cache(cache)
        return cache

    def finish_mtp_cycle(self, mtp_cache) -> None:
        """Discard backend-owned speculative cache state, if any."""

        finish = getattr(self.model, "finish_mtp_cycle", None)
        if callable(finish):
            finish(mtp_cache)

    def admit_kv_tokens(self, tokens: int):
        """Reserve request KV capacity under the streamed memory plan."""

        if self.expert_streaming is None:
            return nullcontext()
        return self.expert_streaming.admit_kv_tokens(tokens)

    def expert_streaming_snapshot(self) -> dict[str, Any] | None:
        if self.expert_streaming is None:
            return None
        return self.expert_streaming.snapshot()

    def expert_resource_telemetry_snapshot(self) -> dict[str, Any] | None:
        if self.expert_streaming is None:
            return None
        return self.expert_streaming.resource_telemetry_snapshot()

    def close(self, *, timeout: float | None = None) -> None:
        if self.expert_streaming is not None:
            self.expert_streaming.close(timeout=timeout)


def _load_impl(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    gemma4_draft_block_size: int | None = None,
    gemma4_target_distribution_mode: str | None = None,
    proj_quant: str | None = None,
    proj_requant: str | None = None,
    expert_streaming_config: Any | None = None,
    expert_manifest: Path | str | None = None,
    mtp_artifacts: Path | str | None = None,
    mtp_precision: str = "bf16",
    _expert_runtime_owner: list[Any],
) -> MTPLXRuntime:
    """Load an MLX model and optionally inject native MTP support.

    ``proj_quant`` / ``proj_requant`` (or the ``MTPLX_PROJ_QUANT`` /
    ``MTPLX_PROJ_REQUANT`` environment variables) quantize a normal resident
    trunk before MTP injection. Streamed loads encode resident projection
    precision in ``expert_streaming_config`` so the memory plan and installed
    layout agree before allocation.

    ``mtp_precision`` selects the streamed external draft head. The Hy3-expert-Q2
    lane requires its exact BF16 head; the Hy3-Q4 and both GLM-5.2 lanes also
    accept a Q4 head (BF16 default, Q4 selectable and priced from its own
    artifact) pending each lane's acceptance-rate validation.
    """
    from .hy3_mtp_patch import HY3_MTP_PRECISIONS

    path = Path(model_path)
    if mtp_precision not in HY3_MTP_PRECISIONS:
        raise ValueError(
            f"mtp_precision must be one of {HY3_MTP_PRECISIONS}; got {mtp_precision!r}"
        )
    streaming_requested = (
        expert_streaming_config is not None or expert_manifest is not None
    )
    if (expert_streaming_config is None) != (expert_manifest is None):
        raise ValueError(
            "expert_streaming_config and expert_manifest must be supplied together"
        )
    if streaming_requested and (proj_quant is not None or proj_requant is not None):
        raise ValueError(
            "streamed loads configure proj_quant/proj_requant through "
            "expert_streaming_config so the construction-time memory plan "
            "matches the installed resident layout"
        )
    if mtp_artifacts is not None and not streaming_requested:
        raise ValueError(
            "mtp_artifacts applies to streamed checkpoints only; non-streamed "
            "models carry their own MTP weights"
        )
    from .gemma4_pair import resolve_gemma4_pair_paths

    gemma4_pair = resolve_gemma4_pair_paths(path)
    if gemma4_pair is not None:
        if streaming_requested:
            raise ValueError("expert streaming does not support Gemma assistant pairs")
        if mtp:
            from .backends.gemma4_assistant import (
                DEFAULT_DRAFT_BLOCK_SIZE,
                Gemma4AssistantRuntimeConfig,
                load_gemma4_assistant_pair,
            )

            metadata = gemma4_pair["metadata"]
            benchmark = metadata.get("benchmark") if isinstance(metadata, dict) else {}
            draft_block_size = DEFAULT_DRAFT_BLOCK_SIZE
            if isinstance(benchmark, dict):
                try:
                    draft_block_size = int(
                        benchmark.get("best_block_size") or draft_block_size
                    )
                except (TypeError, ValueError):
                    draft_block_size = DEFAULT_DRAFT_BLOCK_SIZE
            if gemma4_draft_block_size is not None:
                draft_block_size = int(gemma4_draft_block_size)
            runtime = load_gemma4_assistant_pair(
                Gemma4AssistantRuntimeConfig.from_paths(
                    target_model_path=gemma4_pair["target_model"],
                    assistant_model_path=gemma4_pair["assistant_model"],
                    draft_block_size=draft_block_size,
                    target_distribution_mode=gemma4_target_distribution_mode,
                )
            )
            runtime.model_path = path
            runtime.path = path
            runtime.bundle_path = path
            return runtime
        path = Path(gemma4_pair["target_model"])
    config = load_config(path)
    runtime_metadata = _load_runtime_metadata(path)
    contract = (
        (contract or MTPContract())
        .with_runtime_metadata(runtime_metadata, preserve_explicit=True)
        .with_config_defaults(config)
    )
    from .step3p5_mtp_patch import is_step3p5_mtp_config
    from .qwen3_5_mtp_patch import (
        install_qwen3_5_mtp_trunk_shim,
        is_qwen3_5_mtp_config,
    )

    # Qwen3.5-MoE MTP exports carry model_type ``qwen3_5_mtp`` (no mlx-lm module);
    # the trunk is a vanilla ``qwen3_5_moe``. Alias it so the trunk loads.
    if is_qwen3_5_mtp_config(config):
        install_qwen3_5_mtp_trunk_shim()

    expert_runtime = None
    resident_load_report = None
    streamed_mtp_backend = None
    streamed_mtp_resident_bytes = 0
    hy3_router_incremental_bytes = 0
    mtp_enabled = False
    if streaming_requested:
        from .expert_runtime import (
            ExpertStreamingConfig,
            ExpertStreamingConfigurationError,
            ExpertStreamingRuntime,
            apply_mlx_memory_cap,
        )
        from .expert_streaming_models import get_model_spec
        from .models.expert_mlx import (
            make_mlx_component_bank_allocator,
            make_mlx_slot_buffer_allocator,
        )
        from .resident_loader import construct_resident_model

        import mlx.core as mx

        if not isinstance(expert_streaming_config, ExpertStreamingConfig):
            raise TypeError("expert_streaming_config must be an ExpertStreamingConfig")
        streaming_spec = get_model_spec(expert_streaming_config.model_key)
        verified_artifact_context = nullcontext(None)
        if mtp:
            streamed_mtp_backend = _streamed_mtp_backend(
                expert_streaming_config.model_key,
                mtp_precision,
            )
            if mtp_artifacts is None:
                raise RuntimeError(
                    "this streamed checkpoint omits its trained MTP layer; pass "
                    "mtp_artifacts=<validated external artifact directory> or "
                    "load with mtp=False"
                )
            if streamed_mtp_backend == "glm52":
                from .glm52_mtp_patch import _validate_glm52_mtp_contract

                _validate_glm52_mtp_contract(contract)
                if mtp_precision == "q4":
                    from .glm52_mtp_artifact import (
                        open_verified_glm52_mtp_layer78_q4,
                    )

                    verified_artifact_context = open_verified_glm52_mtp_layer78_q4(
                        Path(mtp_artifacts), deep=True
                    )
                else:
                    from .glm52_mtp_artifact import (
                        open_verified_glm52_mtp_layer78,
                    )

                    verified_artifact_context = open_verified_glm52_mtp_layer78(
                        Path(mtp_artifacts), deep=True
                    )
            elif streamed_mtp_backend == "hy3":
                from .hy3_mtp_patch import open_verified_hy3_mtp_artifacts

                verified_artifact_context = open_verified_hy3_mtp_artifacts(
                    Path(mtp_artifacts),
                    precision=mtp_precision,
                    expected_revision=streaming_spec.source_revision,
                )
        with verified_artifact_context as verified_streamed_artifact:
            if streamed_mtp_backend == "glm52":
                receipt = verified_streamed_artifact.manifest
                inventory = receipt.get("inventory")
                if not isinstance(inventory, dict):
                    raise RuntimeError("GLM-5.2 MTP manifest inventory is missing")
                payload_bytes = inventory.get("payload_bytes")
                if (
                    isinstance(payload_bytes, bool)
                    or not isinstance(payload_bytes, int)
                    or payload_bytes <= 0
                ):
                    raise RuntimeError(
                        "GLM-5.2 MTP manifest payload byte count is invalid"
                    )
                streamed_mtp_resident_bytes = payload_bytes
            elif streamed_mtp_backend == "hy3":
                streamed_mtp_resident_bytes = verified_streamed_artifact.payload_bytes
                if (
                    isinstance(streamed_mtp_resident_bytes, bool)
                    or not isinstance(streamed_mtp_resident_bytes, int)
                    or streamed_mtp_resident_bytes <= 0
                ):
                    raise RuntimeError("Hy3 MTP artifact payload byte count is invalid")

            if (
                str(config.get("model_type") or "") == "hy_v3"
                and expert_streaming_config.hy3_router_kernel != "stock"
            ):
                from .models.hy3_mlx import (
                    estimate_hy3_router_kernel_incremental_bytes,
                )

                hy3_router_incremental_bytes = (
                    estimate_hy3_router_kernel_incremental_bytes(
                        config,
                        expert_streaming_config.hy3_router_kernel,
                        include_mtp=streamed_mtp_backend == "hy3" and bool(mtp),
                    )
                )
            additional_resident_bytes = (
                streamed_mtp_resident_bytes + hy3_router_incremental_bytes
            )
            plan_kwargs = (
                {"additional_resident_bytes": additional_resident_bytes}
                if additional_resident_bytes
                else {}
            )
            # ExpertStreamingRuntime.open computes the same discount from the
            # manifest itself, so plan_kwargs stays free of it.
            # Resolve a pending island_layer_count BEFORE the pre-flight plan
            # (census-first precedence; open() re-resolves idempotently) —
            # census-only specs previously hit the unresolved-count guard here.
            if expert_streaming_config.island_layer_count is not None:
                from .expert_runtime import resolve_island_placement

                expert_streaming_config = resolve_island_placement(
                    expert_streaming_config, Path(expert_manifest).parent
                )
            preflight_plan_kwargs = dict(plan_kwargs)
            if expert_streaming_config.proj_quant or getattr(
                expert_streaming_config, "proj_requant", None
            ):
                from .expert_manifest import load_expert_manifest
                from .expert_runtime import (
                    proj_quant_plan_discount,
                    proj_requant_plan_discount,
                )

                _preflight_manifest = load_expert_manifest(expert_manifest)
                preflight_plan_kwargs["resident_discount_bytes"] = (
                    proj_quant_plan_discount(
                        _preflight_manifest,
                        expert_streaming_config.proj_quant,
                    )
                    + proj_requant_plan_discount(
                        _preflight_manifest,
                        getattr(expert_streaming_config, "proj_requant", None),
                    )
                )
            if bool(getattr(streaming_spec, "is_mixed_official", False)):
                # Mixed-official has no uniform record size; the preflight gate
                # must see the same manifest-derived per-layer sizes as open()
                # (issue #51 M2, D2).
                from .expert_manifest import load_expert_manifest

                preflight_plan_kwargs["layer_record_bytes"] = load_expert_manifest(
                    expert_manifest
                ).record_bytes_by_layer()
            streaming_plan = expert_streaming_config.memory_plan(
                streaming_spec,
                **preflight_plan_kwargs,
            )
            if not streaming_plan.fits_fixed:
                raise ExpertStreamingConfigurationError(
                    "fixed expert-streaming footprint exceeds limit by "
                    f"{-streaming_plan.unallocated_bytes} bytes"
                )
            prebuilt_glm_mtp = None
            prebuilt_hy3_mtp = None
            if mtp:
                # Materialize external MTP heads before allocating expert-cache
                # banks. Stacking their routed experts has a large transient
                # footprint that can breach an otherwise-valid steady-state plan.
                apply_mlx_memory_cap(streaming_plan, mx_module=mx)
            if streamed_mtp_backend == "glm52":
                from .glm52_mtp_patch import build_glm52_mtp_module
                from .models.glm52_mlx import ModelArgs as Glm52ModelArgs

                prebuilt_glm_mtp = build_glm52_mtp_module(
                    mtp_artifacts,
                    Glm52ModelArgs.from_dict(config),
                    expected_revision=streaming_spec.source_revision,
                    precision=mtp_precision,
                    verified_artifact=verified_streamed_artifact,
                )
            elif streamed_mtp_backend == "hy3":
                from .hy3_mtp_patch import build_hy3_mtp_module
                from .models.hy3_mlx import ModelArgs as Hy3ModelArgs

                prebuilt_hy3_mtp = build_hy3_mtp_module(
                    mtp_artifacts,
                    Hy3ModelArgs.from_dict(config),
                    expected_revision=streaming_spec.source_revision,
                    precision=mtp_precision,
                    shared_kernel=expert_streaming_config.hy3_mtp_shared_kernel,
                    shared_kernel_depth=(
                        expert_streaming_config.hy3_mtp_shared_kernel_depth
                    ),
                    verified_artifacts=verified_streamed_artifact,
                )
            if expert_streaming_config.slot_layout == "component-banks":
                from .expert_manifest import load_expert_manifest

                streaming_manifest = load_expert_manifest(expert_manifest)
                slot_allocator = make_mlx_component_bank_allocator(
                    streaming_plan,
                    streaming_spec,
                    streaming_manifest,
                )
            else:
                slot_allocator = make_mlx_slot_buffer_allocator(
                    streaming_plan, streaming_spec
                )

            if mtp_adapter is not None or merge_mtp_adapter:
                raise RuntimeError("MTP adapters are unavailable for streamed loading")
            expert_runtime = ExpertStreamingRuntime.open(
                path,
                expert_manifest,
                expert_streaming_config,
                spec=streaming_spec,
                buffer_allocator=slot_allocator,
                device_synchronize=mx.synchronize,
                apply_memory_cap=True,
                mx_module=mx,
                **plan_kwargs,
            )
            _expert_runtime_owner[:] = [expert_runtime]
            try:
                resident = construct_resident_model(path, expert_runtime, config=config)
                model = resident.model
                resident_load_report = resident.report.as_dict()
                tokenizer = _load_tokenizer_resilient(path, config)
                if mtp:
                    if streamed_mtp_backend == "hy3":
                        from .hy3_mtp_patch import inject_hy3_streamed_mtp_support

                        mtp_enabled = inject_hy3_streamed_mtp_support(
                            model,
                            mtp_artifacts,
                            config,
                            contract,
                            expected_revision=streaming_spec.source_revision,
                            mtp_precision=mtp_precision,
                            shared_kernel=(
                                expert_streaming_config.hy3_mtp_shared_kernel
                            ),
                            shared_kernel_depth=(
                                expert_streaming_config.hy3_mtp_shared_kernel_depth
                            ),
                            mtp_module=prebuilt_hy3_mtp,
                        )
                    elif streamed_mtp_backend == "glm52":
                        from .glm52_mtp_patch import (
                            inject_glm52_streamed_mtp_support,
                        )

                        mtp_enabled = inject_glm52_streamed_mtp_support(
                            model,
                            mtp_artifacts,
                            config,
                            contract,
                            expected_revision=streaming_spec.source_revision,
                            verified_artifact=verified_streamed_artifact,
                            mtp_module=prebuilt_glm_mtp,
                        )
                    else:
                        raise RuntimeError(
                            f"unresolved streamed MTP backend {streamed_mtp_backend!r}"
                        )
                    if not mtp_enabled or not validate_mtp_support(model):
                        raise RuntimeError(f"streamed MTP injection failed for {path}")
                if (
                    str(config.get("model_type") or "") == "hy_v3"
                    and expert_streaming_config.hy3_router_kernel != "stock"
                ):
                    from .models.hy3_mlx import configure_hy3_router_kernels

                    import os as _os

                    # EXPERIMENT gate (2026-07-21): "all" extends the split-K
                    # kernel to trunk routers at rows==1 (AR decode), which
                    # historically fell back to the stock host path. Default
                    # "mtp" preserves the measured ladder behavior exactly.
                    router_kernel_report = configure_hy3_router_kernels(
                        model,
                        expert_streaming_config.hy3_router_kernel,
                        sigmoid_mode=expert_streaming_config.hy3_router_sigmoid,
                        splitk_m1_scope=_os.environ.get(
                            "MTPLX_HY3_ROUTER_SPLITK_M1", "mtp"
                        ),
                    )
                    actual_incremental = int(
                        router_kernel_report.get("incremental_bytes", -1)
                    )
                    if actual_incremental != hy3_router_incremental_bytes:
                        raise RuntimeError(
                            "Hy3 router prepared-layout bytes do not match "
                            f"admission plan: {actual_incremental} != "
                            f"{hy3_router_incremental_bytes}"
                        )
                    setattr(
                        model,
                        "_mtplx_hy3_router_kernel_report",
                        router_kernel_report,
                    )
                    if isinstance(resident_load_report, dict):
                        resident_load_report["hy3_router_kernel"] = router_kernel_report
            except BaseException:
                expert_runtime.close()
                _expert_runtime_owner.clear()
                raise
    elif is_step3p5_mtp_config(config):
        from mlx_lm.utils import load_model

        tokenizer = _load_tokenizer_resilient(path, config)
        model, _loaded_config = load_model(path)
    else:
        from mlx_lm.utils import load as mlx_lm_load

        model, tokenizer = mlx_lm_load(str(_mtp_alias_load_path(path, config)))
    if expert_runtime is None:
        import os as _os

        proj_quant = proj_quant or _os.environ.get("MTPLX_PROJ_QUANT") or None
        proj_requant = proj_requant or _os.environ.get("MTPLX_PROJ_REQUANT") or None
    if expert_runtime is None and (proj_quant or proj_requant):
        from .proj_quant import quantize_projections, requantize_projections

        if proj_quant:
            touched = quantize_projections(model, proj_quant)
            logger.info(
                "[proj-quant] quantized %d trunk *_proj modules to %s",
                len(touched), proj_quant,
            )
        if proj_requant:
            touched = requantize_projections(model, proj_requant)
            logger.info(
                "[proj-quant] requantized %d trunk *_proj modules to %s",
                len(touched), proj_requant,
            )
    if mtp and expert_runtime is None:
        from .deepseek_mtp_patch import (
            inject_deepseek_mtp_support,
            is_deepseek_mtp_config,
        )
        from .glm_mtp_patch import inject_glm_mtp_support, is_glm_mtp_config
        from .mimo_mtp_patch import inject_mimo_mtp_support, is_mimo_mtp_config
        from .nemotron_h_mtp_patch import (
            inject_nemotron_h_mtp_support,
            is_nemotron_h_mtp_config,
        )
        from .step3p5_mtp_patch import inject_step3p5_mtp_support
        from .hy_v3_mtp_patch import inject_hy_v3_mtp_support, is_hy_v3_mtp_config
        from .qwen3_5_mtp_patch import inject_qwen3_5_mtp_support

        if is_nemotron_h_mtp_config(config):
            mtp_enabled = inject_nemotron_h_mtp_support(model, path, config, contract)
        elif is_mimo_mtp_config(config):
            mtp_enabled = inject_mimo_mtp_support(model, path, config, contract)
        elif is_glm_mtp_config(config):
            mtp_enabled = inject_glm_mtp_support(model, path, config, contract)
        elif is_step3p5_mtp_config(config):
            mtp_enabled = inject_step3p5_mtp_support(model, path, config, contract)
        elif is_hy_v3_mtp_config(config):
            mtp_enabled = inject_hy_v3_mtp_support(model, path, config, contract)
        elif is_qwen3_5_mtp_config(config):
            mtp_enabled = inject_qwen3_5_mtp_support(model, path, config, contract)
        elif is_deepseek_mtp_config(config):
            mtp_enabled = inject_deepseek_mtp_support(model, path, config, contract)
        else:
            mtp_enabled = inject_mtp_support(model, path, config, contract)
        if not mtp_enabled or not validate_mtp_support(model):
            raise RuntimeError(f"MTP injection failed for {path}")
    from .attention_split import configure_split_full_attention
    from .native_mlp import configure_native_mlp

    configure_split_full_attention(model)
    configure_native_mlp(model)
    from .nax_verify import install_nax_qlinear_patch, nax_env_enabled

    if nax_env_enabled():
        nax_report = install_nax_qlinear_patch()
        logger.info("[nax-verify] %s", nax_report)
    from .kernel_selfcheck import maybe_run_model_selfcheck

    # Turbo lanes validate themselves once per load on the model's actual
    # dtype/quant format; a mismatching lane disables itself and serving
    # continues on the stock path (surfaced in /health kernel_selfcheck).
    # Expert-streaming loads also pass their spec so the routed expert bank's
    # gather_qmm lane is validated at its own (possibly different) quant format.
    maybe_run_model_selfcheck(
        model,
        expert_spec=getattr(expert_runtime, "spec", None),
    )
    adapter_path = Path(mtp_adapter) if mtp_adapter is not None else None
    adapter_metadata = None
    adapter_merge_report = None
    if adapter_path is not None:
        if not mtp_enabled:
            raise RuntimeError("MTP adapter requires mtp=True")
        adapter_metadata = install_saved_mtp_lora_adapter(model, adapter_path)
        if merge_mtp_adapter:
            adapter_merge_report = merge_installed_mtp_lora_adapters(model)
    elif merge_mtp_adapter:
        raise RuntimeError("merge_mtp_adapter requires mtp_adapter")
    runtime = MTPLXRuntime(
        model,
        tokenizer,
        path,
        mtp_enabled,
        contract,
        mtp_adapter_path=adapter_path,
        mtp_adapter_metadata=adapter_metadata,
        mtp_adapter_merge_report=adapter_merge_report,
        expert_streaming=expert_runtime,
        resident_load_report=resident_load_report,
    )
    if expert_runtime is not None:
        from .compiled_forward import compile_forward_enabled

        routed = set(getattr(streaming_spec, "routed_layer_indices", ()))
        islands = getattr(expert_runtime, "island_layer_set", frozenset())
        runtime._expert_compiled_ar_enabled = bool(
            compile_forward_enabled() and routed and routed <= islands
        )
    return runtime


def load(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    gemma4_draft_block_size: int | None = None,
    gemma4_target_distribution_mode: str | None = None,
    proj_quant: str | None = None,
    proj_requant: str | None = None,
    expert_streaming_config: Any | None = None,
    expert_manifest: Path | str | None = None,
    mtp_artifacts: Path | str | None = None,
    mtp_precision: str = "bf16",
) -> MTPLXRuntime:
    """Load a model and transfer streamed-runtime ownership only on success."""

    expert_runtime_owner: list[Any] = []
    try:
        runtime = _load_impl(
            model_path,
            mtp=mtp,
            contract=contract,
            mtp_adapter=mtp_adapter,
            merge_mtp_adapter=merge_mtp_adapter,
            gemma4_draft_block_size=gemma4_draft_block_size,
            gemma4_target_distribution_mode=gemma4_target_distribution_mode,
            proj_quant=proj_quant,
            proj_requant=proj_requant,
            expert_streaming_config=expert_streaming_config,
            expert_manifest=expert_manifest,
            mtp_artifacts=mtp_artifacts,
            mtp_precision=mtp_precision,
            _expert_runtime_owner=expert_runtime_owner,
        )
    except BaseException:
        if expert_runtime_owner:
            expert_runtime_owner[0].close()
        raise
    expert_runtime_owner.clear()
    return runtime


def inspect(path: Path | str):
    return inspect_model(path)


def _load_tokenizer_resilient(model_path: Path, config: dict[str, Any]) -> Any:
    from mlx_lm.utils import load_tokenizer

    try:
        return load_tokenizer(model_path)
    except Exception as exc:  # noqa: BLE001 - transformers raises several strict-config errors
        logger.warning(
            "[tokenizer] AutoTokenizer parse failed (%s); using tokenizer.json fallback",
            exc,
        )

    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import PreTrainedTokenizerFast

    tcfg_path = model_path / "tokenizer_config.json"
    tcfg = (
        json.loads(tcfg_path.read_text(encoding="utf-8")) if tcfg_path.exists() else {}
    )
    passthrough = {
        key: tcfg[key]
        for key in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "additional_special_tokens",
        )
        if key in tcfg
    }
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        **passthrough,
    )
    chat_template = tcfg.get("chat_template")
    if not chat_template:
        jinja = model_path / "chat_template.jinja"
        if jinja.exists():
            chat_template = jinja.read_text(encoding="utf-8")
    if chat_template:
        hf_tokenizer.chat_template = chat_template
    eos = config.get("eos_token_id")
    if eos is None:
        eos = (config.get("text_config") or {}).get("eos_token_id")
    if isinstance(eos, int):
        eos_ids = [eos]
    elif isinstance(eos, (list, tuple)):
        eos_ids = list(eos)
    else:
        eos_ids = None
    return TokenizerWrapper(
        hf_tokenizer,
        eos_token_ids=eos_ids,
        chat_template=None,
    )


def _mtp_alias_load_path(path: Path, config: dict[str, Any] | None) -> Path:
    """Loadable path for `*_mtp`-typed checkpoints (issue #147).

    vLLM-convention MTP checkpoints ship config.json with model_type like
    ``qwen3_5_mtp``: the trunk is the plain base architecture plus an
    embedded MTP head. mlx_lm's class table has no ``*_mtp`` modules, so
    handing it the raw dir fails with "Model type ... not supported" even
    though the forge probe correctly reports the family as supported. When
    the stripped base module exists in mlx_lm and the full name does not,
    build a symlink wrapper with a patched config.json (model_type=base)
    and load through it; MTP injection later picks the head up from the
    original weights. Everything else returns the path untouched.
    """

    model_type = str((config or {}).get("model_type") or "")
    if not model_type.endswith("_mtp"):
        return path
    base_type = model_type[: -len("_mtp")]
    import importlib.util

    def _mlx_lm_has(model_type_name: str) -> bool:
        return (
            importlib.util.find_spec(f"mlx_lm.models.{model_type_name}")
            is not None
        )

    if _mlx_lm_has(model_type) or not _mlx_lm_has(base_type):
        return path
    try:
        wrapper_root = Path.home() / ".mtplx" / "build-cache" / "mtp-alias-load"
        digest = hashlib.sha256(
            f"{path.resolve()}::{base_type}".encode("utf-8")
        ).hexdigest()[:16]
        wrapper = wrapper_root / f"{path.name}-{base_type}-{digest}"
        patched_config = dict(config or {})
        patched_config["model_type"] = base_type
        marker = wrapper / ".mtplx-alias-source"
        if not marker.exists() or marker.read_text(encoding="utf-8") != str(
            path.resolve()
        ):
            wrapper.mkdir(parents=True, exist_ok=True)
            for item in path.iterdir():
                if item.name in {"config.json", ".mtplx-alias-source"}:
                    continue
                link = wrapper / item.name
                if link.is_symlink() or link.exists():
                    continue
                link.symlink_to(item)
            marker.write_text(str(path.resolve()), encoding="utf-8")
        # Rewrite the config every time: the source config may have changed.
        (wrapper / "config.json").write_text(
            json.dumps(patched_config, indent=2), encoding="utf-8"
        )
        return wrapper
    except Exception:
        # Wrapper construction is best-effort; the raw path preserves the
        # original (informative) mlx_lm error.
        return path


def _load_runtime_metadata(path: Path) -> dict[str, Any] | None:
    runtime_path = path / "mtplx_runtime.json"
    if not runtime_path.exists():
        return None
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
