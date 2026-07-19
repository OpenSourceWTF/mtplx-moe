"""Runtime MTP injection for Nemotron-H MLX models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, text_config

logger = logging.getLogger(__name__)

NEMOTRON_H_MODEL_TYPES = {"nemotron_h", "nemotron_h_puzzle"}


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("num_nextn_predict_layers")
        or tcfg.get("mtp_num_hidden_layers")
        or config.get("num_nextn_predict_layers")
        or config.get("mtp_num_hidden_layers")
        or 0
    )


def _model_type(config: dict[str, Any]) -> str:
    tcfg = text_config(config)
    return str(tcfg.get("model_type") or config.get("model_type") or "").lower()


def _mtp_pattern(config: dict[str, Any]) -> str:
    tcfg = text_config(config)
    raw = (
        tcfg.get("mtp_hybrid_override_pattern")
        or config.get("mtp_hybrid_override_pattern")
        or tcfg.get("hybrid_override_pattern")
        or config.get("hybrid_override_pattern")
    )
    mapping = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
    if isinstance(raw, str):
        if "," in raw:
            parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
            return "".join(mapping.get(part, part) for part in parts)
        return raw
    if isinstance(raw, list):
        return "".join(mapping.get(str(part).lower(), str(part)) for part in raw)
    return ""


def is_nemotron_h_mtp_config(config: dict[str, Any]) -> bool:
    pattern = _mtp_pattern(config)
    return (
        _model_type(config) in NEMOTRON_H_MODEL_TYPES
        and _num_mtp_layers(config) == 1
        and bool(pattern)
        and set(pattern).issubset({"*", "E"})
    )


def _load_weight_file(path: Path) -> dict[str, Any]:
    import mlx.core as mx

    if path.suffix == ".json":
        return {}
    return dict(mx.load(str(path)))


def _candidate_weight_files(model_path: Path, config: dict[str, Any]) -> list[Path]:
    mtp_file = expected_mtp_file(model_path, config)
    if mtp_file.exists():
        return [mtp_file]

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        start = int(text_config(config).get("num_hidden_layers") or config.get("num_hidden_layers") or 0)
        physical_layers = max(len(_mtp_pattern(config)), 1)
        wanted_prefixes = tuple(
            [f"model.layers.{start + i}." for i in range(physical_layers)]
            + [f"backbone.layers.{start + i}." for i in range(physical_layers)]
            + [f"mtp.layers.{i}." for i in range(physical_layers)]
        )
        selected = {
            model_path / rel
            for key, rel in weight_map.items()
            if str(key).startswith(wanted_prefixes)
            or str(key).startswith(("mtp.", "lm_head.", "backbone.embeddings."))
        }
        if selected:
            return sorted(selected)

    return sorted(model_path.glob("model*.safetensors"))


def _num_routed_experts(config: dict[str, Any], args: Any) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("mtp_n_routed_experts")
        or config.get("mtp_n_routed_experts")
        or getattr(args, "n_routed_experts", 0)
        or 0
    )


def _stack_moe_experts(weights: dict[str, Any], prefix: str, n_routed_experts: int) -> None:
    import mlx.core as mx

    if n_routed_experts <= 0:
        return
    mapping = {"up_proj": "fc1", "down_proj": "fc2"}
    for source, target in mapping.items():
        for leaf in ("weight", "scales", "biases"):
            first = f"{prefix}.mixer.experts.0.{source}.{leaf}"
            if first not in weights:
                continue
            values = [
                weights.pop(f"{prefix}.mixer.experts.{idx}.{source}.{leaf}")
                for idx in range(n_routed_experts)
            ]
            weights[f"{prefix}.mixer.switch_mlp.{target}.{leaf}"] = mx.stack(values)


def _has_complete_nemotron_h_mtp_payload(
    weights: dict[str, Any],
    *,
    physical_layers: int,
) -> bool:
    """Return true only when every Nemotron-H MTP layer has real weights.

    A prefix-mismatched checkpoint still yields a non-empty ``mapped`` from
    stray keys, and ``strict=False`` then loads nothing -- injection would
    report success while the draft head stays randomly initialized. Only
    ``norm`` and ``mixer`` are unconditional per block; enorm/hnorm/eh_proj
    and final_layernorm depend on the block's position flags.
    """

    for local_idx in range(physical_layers):
        prefix = f"layers.{local_idx}."
        if f"{prefix}norm.weight" not in weights:
            return False
        if not any(key.startswith(f"{prefix}mixer.") for key in weights):
            return False
    return True


def _rewrite_nemotron_h_mtp_weights(
    raw: dict[str, Any],
    *,
    args: Any,
    config: dict[str, Any],
    start_layer: int,
    physical_layers: int,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        if "rotary_emb.inv_freq" in key:
            continue
        if key.startswith(("lm_head.", "backbone.embeddings.", "model.embed_tokens.", "embed_tokens.")):
            continue
        if key.startswith("layers."):
            mapped[key] = value
            continue
        if key.startswith("mtp.layers."):
            mapped[key.removeprefix("mtp.")] = value
            continue
        for local_idx in range(physical_layers):
            raw_prefixes = (
                f"model.layers.{start_layer + local_idx}.",
                f"backbone.layers.{start_layer + local_idx}.",
                f"model.mtp.layers.{local_idx}.",
            )
            matched_prefix = next((prefix for prefix in raw_prefixes if key.startswith(prefix)), None)
            if matched_prefix is None:
                continue
            mapped[f"layers.{local_idx}.{key.removeprefix(matched_prefix)}"] = value
            break

    n_routed = _num_routed_experts(config, args)
    for local_idx in range(physical_layers):
        _stack_moe_experts(mapped, f"layers.{local_idx}", n_routed)
    return mapped


def _quantize_for_loaded_weights(mtp: Any, config: dict[str, Any], weights: dict[str, Any]) -> None:
    import mlx.nn as nn

    quantization = config.get("quantization") or config.get("quantization_config") or {}
    if not quantization:
        return
    if "group_size" not in quantization or "bits" not in quantization:
        return

    def class_predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" in weights:
            return {
                "group_size": int(quantization["group_size"]),
                "bits": int(quantization["bits"]),
                "mode": quantization.get("mode", "affine"),
            }
        return False

    nn.quantize(
        mtp,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _make_nemotron_h_mtp_module(config: dict[str, Any], args: Any):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import nemotron_h
    from mlx_lm.models.base import create_attention_mask

    pattern = _mtp_pattern(config)

    class _NemotronHMTPBlock(nn.Module):
        def __init__(self, block_type: str, *, has_start_projections: bool, has_end_norm: bool):
            super().__init__()
            self.block_type = block_type
            self.has_start_projections = has_start_projections
            self.has_end_norm = has_end_norm
            if has_start_projections:
                self.enorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
                self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
                self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
            if block_type == "*":
                self.mixer = nemotron_h.NemotronHAttention(args)
            elif block_type == "E":
                self.mixer = nemotron_h.NemotronHMoE(args)
            else:
                raise ValueError(f"Unsupported Nemotron-H MTP pattern char: {block_type!r}")
            if has_end_norm:
                self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        def __call__(self, inputs_embeds, hidden_states, *, cache=None):
            if self.has_start_projections:
                hidden_states = self.eh_proj(
                    mx.concatenate(
                        [self.enorm(inputs_embeds), self.hnorm(hidden_states)],
                        axis=-1,
                    )
                )
            residual = hidden_states
            normed = self.norm(hidden_states)
            if self.block_type == "*":
                mask = create_attention_mask(normed, cache)
                hidden_states = residual + self.mixer(normed, mask=mask, cache=cache)
            else:
                hidden_states = residual + self.mixer(normed)
            if self.has_end_norm:
                hidden_states = self.final_layernorm(hidden_states)
            return hidden_states

    class _NemotronHMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                _NemotronHMTPBlock(
                    block_type,
                    has_start_projections=idx == 0,
                    has_end_norm=idx == len(pattern) - 1,
                )
                for idx, block_type in enumerate(pattern)
            ]
            self.pattern = pattern
            self.start_layer = int(getattr(args, "num_hidden_layers"))
            self.num_mtp_layers = 1

    return _NemotronHMTP()


def inject_nemotron_h_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
) -> bool:
    """Attach Nemotron-H native MTP support to a loaded mlx-lm model."""
    import mlx.core as mx
    from mlx_lm.models import nemotron_h
    from mlx_lm.models.cache import KVCache

    if not is_nemotron_h_mtp_config(config):
        return False

    model_path = Path(model_path)
    tcfg = text_config(config)
    args = getattr(model, "args", None)
    if args is None:
        args = nemotron_h.ModelArgs.from_dict(tcfg)

    mtp = _make_nemotron_h_mtp_module(config, args)
    raw_weights: dict[str, Any] = {}
    for file in _candidate_weight_files(model_path, config):
        raw_weights.update(_load_weight_file(file))

    mapped = _rewrite_nemotron_h_mtp_weights(
        raw_weights,
        args=args,
        config=config,
        start_layer=int(getattr(args, "num_hidden_layers")),
        physical_layers=len(mtp.layers),
    )
    if not mapped or not _has_complete_nemotron_h_mtp_payload(
        mapped, physical_layers=len(mtp.layers)
    ):
        logger.warning("[Nemotron-H MTP inject] No Nemotron-H MTP weights found in %s", model_path)
        return False

    _quantize_for_loaded_weights(mtp, config, mapped)
    mtp.load_weights(list(mapped.items()), strict=False)
    mx.eval(mtp.parameters())

    original_outer_class = model.__class__

    class _MTPLXNemotronHModel(original_outer_class):
        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            **kwargs,
        ):
            if input_embeddings is not None:
                raise ValueError("Nemotron-H MTP backend does not support input_embeddings")
            hidden = self.backbone(inputs, cache=cache)
            logits = self.lm_head(hidden)
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "post_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError("Nemotron-H MTP backend supports embedding_hidden concat order only")
            if mtp_depth is not None and int(mtp_depth) > 1:
                raise ValueError("Nemotron-H MTP backend currently supports mtp_depth=1 only")
            inputs_embeds = self.backbone.embeddings(next_token_ids)
            hidden = hidden_states
            attention_cache_idx = 0
            for layer in self.mtp.layers:
                layer_cache = None
                if layer.block_type == "*" and mtp_cache is not None:
                    layer_cache = mtp_cache[attention_cache_idx]
                    attention_cache_idx += 1
                hidden = layer(inputs_embeds, hidden, cache=layer_cache)
            logits = self.lm_head(hidden)
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                mtp_depth=mtp_depth,
            )
            return hidden

        def make_mtp_cache(self):
            return [KVCache() for layer in self.mtp.layers if layer.block_type == "*"]

        def make_cache(self):
            make_cache = getattr(super(), "make_cache", None)
            if callable(make_cache):
                return make_cache()
            return [KVCache() for layer in self.layers if getattr(layer, "block_type", None) == "*"]

    model.mtp = mtp
    model.__class__ = _MTPLXNemotronHModel
    logger.info("[Nemotron-H MTP inject] Loaded %d tensors from %s", len(mapped), model_path)
    return True
