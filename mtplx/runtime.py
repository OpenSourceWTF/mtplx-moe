"""High-level MTPLX runtime loading primitives."""

from __future__ import annotations

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
            if not return_hidden and hidden_variant is None and not kwargs:
                return self.model(input_ids, cache=cache)
            return self.model(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                **kwargs,
            )

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
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


def load(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    gemma4_draft_block_size: int | None = None,
    gemma4_target_distribution_mode: str | None = None,
    expert_streaming_config: Any | None = None,
    expert_manifest: Path | str | None = None,
    mtp_artifacts: Path | str | None = None,
    mtp_precision: str = "bf16",
) -> MTPLXRuntime:
    """Load an MLX model and optionally inject native MTP support.

    ``mtp_precision`` selects the streamed Hy3 layer-80 head build: ``"bf16"``
    (default, per docs/FORGE_BACKEND_CONTRACT.md section 6 — quantized MTP
    heads collapse acceptance) loads the bit-exact BF16 artifact (~7.5 GB
    resident; budget it against the expert cache), ``"q4"`` loads the pinned
    quantized artifacts (~1.94 GiB expert bank).
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
    from .step3p5_mtp_patch import is_step3p5_mtp_config

    expert_runtime = None
    resident_load_report = None
    if streaming_requested:
        from .expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
        from .expert_streaming_models import get_model_spec
        from .models.expert_mlx import (
            make_mlx_component_bank_allocator,
            make_mlx_slot_buffer_allocator,
        )
        from .resident_loader import construct_resident_model

        import mlx.core as mx

        streaming_spec = get_model_spec(expert_streaming_config.model_key)
        streaming_plan = expert_streaming_config.memory_plan(streaming_spec)
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

        if not isinstance(expert_streaming_config, ExpertStreamingConfig):
            raise TypeError("expert_streaming_config must be an ExpertStreamingConfig")
        if mtp and mtp_artifacts is None:
            raise RuntimeError(
                "the pinned Hy3-4bit and GLM-5.2-4bit artifacts omit MTP weights; "
                "pass mtp_artifacts=<layer-80 artifact directory> to enable "
                "streamed Hy3 MTP or load with mtp=False"
            )
        if mtp and expert_streaming_config.model_key != "hy3-q4":
            raise RuntimeError(
                "streamed MTP artifacts are packaged for hy3-q4 only; "
                f"got {expert_streaming_config.model_key!r}"
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
        )
        try:
            resident = construct_resident_model(path, expert_runtime, config=config)
            model = resident.model
            resident_load_report = resident.report.as_dict()
            tokenizer = _load_tokenizer_resilient(path, config)
        except BaseException:
            expert_runtime.close()
            raise
    elif is_step3p5_mtp_config(config):
        from mlx_lm.utils import load_model

        tokenizer = _load_tokenizer_resilient(path, config)
        model, _loaded_config = load_model(path)
    else:
        from mlx_lm.utils import load as mlx_lm_load

        model, tokenizer = mlx_lm_load(str(path))
    runtime_metadata = _load_runtime_metadata(path)
    contract = (
        (contract or MTPContract())
        .with_runtime_metadata(runtime_metadata, preserve_explicit=True)
        .with_config_defaults(config)
    )
    mtp_enabled = False
    if mtp and expert_runtime is not None:
        from .hy3_mtp_patch import inject_hy3_streamed_mtp_support

        try:
            mtp_enabled = inject_hy3_streamed_mtp_support(
                model, mtp_artifacts, config, contract, mtp_precision=mtp_precision
            )
            if not mtp_enabled or not validate_mtp_support(model):
                raise RuntimeError(f"streamed Hy3 MTP injection failed for {path}")
        except BaseException:
            expert_runtime.close()
            raise
    elif mtp:
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

        if is_nemotron_h_mtp_config(config):
            mtp_enabled = inject_nemotron_h_mtp_support(model, path, config, contract)
        elif is_mimo_mtp_config(config):
            mtp_enabled = inject_mimo_mtp_support(model, path, config, contract)
        elif is_glm_mtp_config(config):
            mtp_enabled = inject_glm_mtp_support(model, path, config, contract)
        elif is_step3p5_mtp_config(config):
            mtp_enabled = inject_step3p5_mtp_support(model, path, config, contract)
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
    maybe_run_model_selfcheck(model)
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
    return MTPLXRuntime(
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


def _load_runtime_metadata(path: Path) -> dict[str, Any] | None:
    runtime_path = path / "mtplx_runtime.json"
    if not runtime_path.exists():
        return None
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
