"""High-level MTPLX runtime loading primitives."""

from __future__ import annotations

import hashlib
import inspect as py_inspect
import json
import logging
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
    diagnostic_counters: dict[str, int] = field(default_factory=dict)
    _forward_ar_supports_emit_logits: bool | None = field(default=None, init=False, repr=False)
    _forward_ar_supports_logits_keep: bool | None = field(default=None, init=False, repr=False)

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = int(self.diagnostic_counters.get(key, 0)) + int(amount)

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
        self._count("forward_ar_hidden_calls" if return_hidden else "forward_ar_plain_calls")
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
            # leaves throws, and its shape differs from a single-token decode
            # step, forcing a retrace. Prefill stays eager.
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

        if not compile_forward_enabled() or not cache:
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
            import os as _os

            reserve = int(_os.environ.get("MTPLX_COMPILE_AR_RESERVE_TOKENS", "4096"))
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
            self.contract.concat_order if concat_order in {None, "auto", "contract"} else concat_order
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
            self.contract.concat_order if concat_order in {None, "auto", "contract"} else concat_order
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
) -> MTPLXRuntime:
    """Load an MLX model and optionally inject native MTP support.

    ``proj_quant`` / ``proj_requant`` (or the ``MTPLX_PROJ_QUANT`` /
    ``MTPLX_PROJ_REQUANT`` environment variables) quantize the trunk
    ``*_proj`` Linears at load time — see :mod:`mtplx.proj_quant`. Applied
    to the trunk only, before MTP injection, so a draft head's precision is
    never reduced.
    """
    path = Path(model_path)
    from .gemma4_pair import resolve_gemma4_pair_paths

    gemma4_pair = resolve_gemma4_pair_paths(path)
    if gemma4_pair is not None:
        if mtp:
            from .backends.gemma4_assistant import (
                DEFAULT_DRAFT_BLOCK_SIZE,
                Gemma4AssistantRuntimeConfig,
                load_gemma4_assistant_pair,
            )

            metadata = gemma4_pair["metadata"]
            benchmark = (
                metadata.get("benchmark") if isinstance(metadata, dict) else {}
            )
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
    from .qwen3_5_mtp_patch import (
        install_qwen3_5_mtp_trunk_shim,
        is_qwen3_5_mtp_config,
    )

    # Qwen3.5-MoE MTP exports carry model_type ``qwen3_5_mtp`` (no mlx-lm module);
    # the trunk is a vanilla ``qwen3_5_moe``. Alias it so the trunk loads.
    if is_qwen3_5_mtp_config(config):
        install_qwen3_5_mtp_trunk_shim()

    if is_step3p5_mtp_config(config):
        from mlx_lm.utils import load_model

        tokenizer = _load_tokenizer_resilient(path, config)
        model, _loaded_config = load_model(path)
    else:
        from mlx_lm.utils import load as mlx_lm_load

        model, tokenizer = mlx_lm_load(str(_mtp_alias_load_path(path, config)))
    import os as _os

    proj_quant = proj_quant or _os.environ.get("MTPLX_PROJ_QUANT") or None
    proj_requant = proj_requant or _os.environ.get("MTPLX_PROJ_REQUANT") or None
    if proj_quant or proj_requant:
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
    runtime_metadata = _load_runtime_metadata(path)
    contract = (
        (contract or MTPContract())
        .with_runtime_metadata(runtime_metadata, preserve_explicit=True)
        .with_config_defaults(config)
    )
    mtp_enabled = False
    if mtp:
        from .deepseek_mtp_patch import inject_deepseek_mtp_support, is_deepseek_mtp_config
        from .glm_mtp_patch import inject_glm_mtp_support, is_glm_mtp_config
        from .mimo_mtp_patch import inject_mimo_mtp_support, is_mimo_mtp_config
        from .nemotron_h_mtp_patch import inject_nemotron_h_mtp_support, is_nemotron_h_mtp_config
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
    tcfg = json.loads(tcfg_path.read_text(encoding="utf-8")) if tcfg_path.exists() else {}
    passthrough = {
        key: tcfg[key]
        for key in ("bos_token", "eos_token", "pad_token", "unk_token", "additional_special_tokens")
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
