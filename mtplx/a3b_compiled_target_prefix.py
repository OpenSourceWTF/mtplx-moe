"""Exact compiled target-prefix route for the A3B K1 decode contract."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .attention_context import attention_phase
from .graphbank import (
    CompiledVerifyBank,
    TensorOffsetKVCache,
    VERIFY_SPEC_KIND_FULL_ATTN,
    VERIFY_SPEC_KIND_GDN,
    _compiled_verify_boundary,
    _compiled_verify_donation_enabled,
    _owned_state_env_active,
    build_verify_state_spec,
    cache_has_python_offsets,
    promote_kv_cache_offsets,
)


_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)
_STATE_SPEC = tuple(
    (
        index,
        VERIFY_SPEC_KIND_GDN if kind == "linear_attention" else VERIFY_SPEC_KIND_FULL_ATTN,
        2 if kind == "linear_attention" else 3,
    )
    for index, kind in enumerate(_LAYER_TYPES)
)
_CAPTURE_LEAVES = 30 * 2
_STATE_START = 2 + _CAPTURE_LEAVES
_MAX_REQUEST_CONTEXT = 12_288
_FACTORY_ATTRIBUTE = "_mtplx_a3b_compiled_target_prefix_factory"
_SHARED_M2_STEPS: dict[
    tuple[int, str],
    tuple[Callable[..., Any], dict[str, Any], weakref.ReferenceType[Any]],
] = {}


class A3BCompiledTargetPrefixConfigError(RuntimeError):
    """The exact A3B K1 compiled target-prefix lane cannot be installed."""


@dataclass(frozen=True)
class A3BCompiledTargetPrefixFactory:
    """Model-load proof that the exact A3B target graph owns the route."""

    layer_types: tuple[str, ...]
    gdn_layers: int
    full_attention_layers: int
    hidden_size: int
    quantization: str


def _enabled() -> bool:
    return os.environ.get("MTPLX_COMPILED_TARGET_PREFIX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fail(message: str) -> None:
    raise A3BCompiledTargetPrefixConfigError(message)


def prepare_a3b_compiled_target_prefix(
    model: Any,
    *,
    config: dict[str, Any],
) -> A3BCompiledTargetPrefixFactory | None:
    """Validate checkpoint-owned facts once, while the model is constructed."""
    if not _enabled():
        if hasattr(model, _FACTORY_ATTRIBUTE):
            delattr(model, _FACTORY_ATTRIBUTE)
        return None

    text = config.get("text_config")
    quant = config.get("quantization") or config.get("quantization_config")
    if (
        config.get("model_type") != "qwen3_5_moe"
        or config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or not isinstance(text, dict)
        or text.get("model_type") != "qwen3_5_moe_text"
        or text.get("dtype") != "bfloat16"
        or int(text.get("hidden_size", -1)) != 2048
        or int(text.get("num_hidden_layers", -1)) != 40
        or tuple(text.get("layer_types", ())) != _LAYER_TYPES
        or int(text.get("linear_num_value_heads", -1)) != 32
        or int(text.get("linear_num_key_heads", -1)) != 16
        or int(text.get("linear_value_head_dim", -1)) != 128
        or int(text.get("linear_key_head_dim", -1)) != 128
        or int(text.get("linear_conv_kernel_dim", -1)) != 4
        or int(text.get("num_attention_heads", -1)) != 16
        or int(text.get("num_key_value_heads", -1)) != 2
        or int(text.get("head_dim", -1)) != 256
        or int(text.get("mtp_num_hidden_layers", -1)) != 1
        or not isinstance(quant, dict)
        or int(quant.get("bits", -1)) != 4
        or int(quant.get("group_size", -1)) != 64
        or str(quant.get("mode", "")) != "affine"
    ):
        _fail("compiled A3B target-prefix requires the exact q4/group64 A3B config")

    text_model = getattr(model, "language_model", None)
    inner = getattr(text_model, "model", None)
    layers = list(getattr(inner, "layers", ()) or ())
    mtp_layers = list(getattr(getattr(model, "mtp", None), "layers", ()) or ())
    actual_types = tuple(
        "linear_attention"
        if bool(getattr(layer, "is_linear", hasattr(layer, "linear_attn")))
        else "full_attention"
        for layer in layers
    )
    if len(layers) != 40 or len(mtp_layers) != 1 or actual_types != _LAYER_TYPES:
        _fail("compiled A3B target-prefix requires 30 GDN and 10 attention layers")

    for index, (layer, kind) in enumerate(zip(layers, _LAYER_TYPES)):
        if kind == "linear_attention":
            gdn = getattr(layer, "linear_attn", None)
            if (
                gdn is None
                or getattr(gdn, "sharding_group", None) is not None
                or int(getattr(gdn, "num_v_heads", -1)) != 32
                or int(getattr(gdn, "num_k_heads", -1)) != 16
                or int(getattr(gdn, "head_v_dim", -1)) != 128
                or int(getattr(gdn, "head_k_dim", -1)) != 128
                or int(getattr(gdn, "conv_kernel_size", -1)) != 4
                or int(getattr(gdn, "conv_dim", -1)) != 8192
            ):
                _fail(f"compiled A3B target-prefix GDN geometry mismatch at layer {index}")
        elif not hasattr(layer, "self_attn"):
            _fail(f"compiled A3B target-prefix attention ownership missing at layer {index}")

    factory = A3BCompiledTargetPrefixFactory(
        layer_types=_LAYER_TYPES,
        gdn_layers=30,
        full_attention_layers=10,
        hidden_size=2048,
        quantization="affine_q4_group64",
    )
    setattr(model, _FACTORY_ATTRIBUTE, factory)
    return factory


def a3b_compiled_target_prefix_factory(
    model: Any,
) -> A3BCompiledTargetPrefixFactory | None:
    factory = getattr(model, _FACTORY_ATTRIBUTE, None)
    return factory if isinstance(factory, A3BCompiledTargetPrefixFactory) else None


def _make_a3b_k1_target_prefix_m2_step(
    *,
    host: dict[str, Any],
) -> Callable[..., Any]:
    """Build the fixed M2 trace body; Python executes only while tracing."""
    spec = _STATE_SPEC

    def step(input_ids, *state_in):
        bank = host["bank"]
        shadow = bank._shadow
        position = 0
        for index, kind, leaves in spec:
            entry = shadow[index]
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                entry.cache[0] = state_in[position]
                entry.cache[1] = state_in[position + 1]
                entry.cache[2] = state_in[position + 2]
                entry.rollback_state[0] = None
                entry.rollback_state[1] = None
                entry.rollback_state[2] = None
            else:
                entry.cache[0] = state_in[position]
                entry.cache[1] = state_in[position + 1]
            position += leaves
        with attention_phase("decode_verify"):
            logits, hidden, captures = bank._runtime_forward(
                input_ids,
                cache=shadow,
                return_hidden=True,
                hidden_variant=host["hidden_variant"],
            )
        flattened_captures: list[Any] = []
        state_out: list[Any] = []
        for index, kind, _leaves in spec:
            entry = shadow[index]
            if kind == VERIFY_SPEC_KIND_GDN:
                layer_capture = captures[index]
                flattened_captures.extend(
                    (layer_capture["conv_states"], layer_capture["states"])
                )
                state_out.extend((entry.cache[0], entry.cache[1]))
            else:
                state_out.extend((entry.cache[0], entry.cache[1], entry.cache[2]))
        return (logits, hidden, *flattened_captures, *state_out)

    return step


def _shared_m2_step(
    runtime: Any,
    bank: CompiledVerifyBank,
    hidden_variant: str | None,
) -> Callable[..., Any]:
    key = (id(runtime), str(hidden_variant or ""))
    entry = _SHARED_M2_STEPS.get(key)
    if entry is not None:
        compiled, host, runtime_ref = entry
        if runtime_ref() is runtime:
            host["bank"] = bank
            return compiled
        _SHARED_M2_STEPS.pop(key, None)
    host = {"bank": bank, "hidden_variant": hidden_variant}
    compiled = mx.compile(_make_a3b_k1_target_prefix_m2_step(host=host))
    _SHARED_M2_STEPS[key] = (compiled, host, weakref.ref(runtime))
    return compiled


@dataclass
class A3BK1TargetPrefixRoute:
    """Request-owned direct M2 route installed after prompt cache creation."""

    bank: CompiledVerifyBank
    cache: list[Any]
    compiled_m2: Callable[..., Any]
    state_slots: tuple[tuple[list[Any], int], ...]
    rollback_slots: tuple[list[Any], ...]
    request_max_tokens: int
    growth_reserve_tokens: int
    prompt_tokens: int

    def verify_m2(self, input_ids):
        return self._forward_m2(input_ids)

    def repair_m2(self, input_ids):
        return self._forward_m2(input_ids)

    def _forward_m2(self, input_ids):
        state_in = [container[slot] for container, slot in self.state_slots]
        outputs = self.compiled_m2(input_ids, *state_in)
        for (container, slot), value in zip(
            self.state_slots,
            outputs[_STATE_START:],
        ):
            container[slot] = value
        for rollback in self.rollback_slots:
            rollback[0] = None
            rollback[1] = None
            rollback[2] = None
        mx.async_eval(*outputs)
        return outputs[0], outputs[1], None

    def demote(self) -> int:
        return self.bank.demote(self.cache)

    def final_report(self, *, verify_calls: int, repair_calls: int) -> dict[str, Any]:
        m2_calls = int(verify_calls) + int(repair_calls)
        return {
            "mode": "a3b_k1_target_prefix",
            "installed": True,
            "installation_status": "installed",
            "calls": m2_calls,
            "compiled_calls": m2_calls,
            "m2_calls": m2_calls,
            "buckets": {"0": m2_calls},
            "fallback_calls": 0,
            "fallback_reasons": {},
            "growth_demotions": 0,
            "request_max_tokens": self.request_max_tokens,
            "max_verify_len": 2,
            "speculative_headroom": 2,
            "growth_reserve_tokens": self.growth_reserve_tokens,
            "prompt_tokens": self.prompt_tokens,
            "max_request_context": _MAX_REQUEST_CONTEXT,
            "capture_backend": "stock",
            "compiled_entry_count": 1,
            "compiled_keys": ["m2:default:b0"],
            "permanent_eager": False,
        }


def _validate_request_cache(cache: list[Any], *, required_capacity: int) -> None:
    if len(cache) != 40:
        _fail("compiled A3B target-prefix requires exactly 40 target cache entries")
    spec, reason = build_verify_state_spec(cache)
    if spec is None or tuple(spec) != _STATE_SPEC:
        _fail(f"compiled A3B target-prefix cache ownership mismatch: {reason or spec}")
    if cache_has_python_offsets(cache):
        _fail("compiled A3B target-prefix requires tensor-owned KV offsets")

    for index, kind, _leaves in _STATE_SPEC:
        entry = cache[index]
        if kind == VERIFY_SPEC_KIND_GDN:
            if not isinstance(entry, ArraysCache):
                _fail(f"compiled A3B target-prefix requires ArraysCache at layer {index}")
            conv_state, recurrent_state = entry.cache
            if (
                tuple(conv_state.shape) != (1, 3, 8192)
                or conv_state.dtype != mx.bfloat16
                or tuple(recurrent_state.shape) != (1, 32, 128, 128)
                or recurrent_state.dtype != mx.float32
            ):
                _fail(f"compiled A3B target-prefix GDN cache mismatch at layer {index}")
        else:
            if not isinstance(entry, TensorOffsetKVCache):
                _fail(f"compiled A3B target-prefix requires dense KV at layer {index}")
            keys, values, _offset = entry.cache
            if (
                tuple(keys.shape[:2]) != (1, 2)
                or int(keys.shape[-1]) != 256
                or tuple(values.shape) != tuple(keys.shape)
                or keys.dtype != mx.bfloat16
                or values.dtype != mx.bfloat16
                or int(keys.shape[2]) < required_capacity
            ):
                _fail(f"compiled A3B target-prefix dense KV mismatch at layer {index}")


def install_a3b_k1_target_prefix_route(
    runtime: Any,
    cache: list[Any],
    *,
    max_tokens: int,
    prompt_tokens: int,
    verify_strategy: str,
    speculative_depth: int,
    requested_speculative_depth: int,
    verify_core: str,
    hidden_variant: str | None,
    state_rebase_every: int,
) -> A3BK1TargetPrefixRoute:
    """Validate request/cache facts once and install the fixed bucket-0 route."""
    factory = getattr(getattr(runtime, "model", None), _FACTORY_ATTRIBUTE, None)
    if not isinstance(factory, A3BCompiledTargetPrefixFactory):
        _fail("compiled A3B target-prefix model contract was not installed at load")
    if (
        verify_strategy != "target_prefix"
        or int(speculative_depth) != 1
        or int(requested_speculative_depth) != 1
        or str(verify_core) != "stock"
        or int(state_rebase_every) != 0
        or int(max_tokens) <= 0
        or int(prompt_tokens) <= 0
    ):
        _fail("compiled A3B target-prefix requires exact K1 stock request ownership")
    if _owned_state_env_active("MTPLX_OWNED_ATTN_KV") or _owned_state_env_active(
        "MTPLX_OWNED_RECURRENT_STATE"
    ):
        _fail("compiled A3B target-prefix conflicts with owned-state wrappers")
    if int(prompt_tokens) + int(max_tokens) > _MAX_REQUEST_CONTEXT:
        _fail("compiled A3B target-prefix request exceeds its installed context ceiling")
    if _compiled_verify_boundary() != "both" or not _compiled_verify_donation_enabled():
        _fail("compiled A3B target-prefix requires the measured donation boundary")

    reserve = int(max_tokens) + 2
    _promoted, failures = promote_kv_cache_offsets(
        cache,
        reserve_tokens=2,
        preserve_paged=True,
        initial_reserve_tokens=reserve,
    )
    if failures:
        _fail("compiled A3B target-prefix cache promotion failed: " + ",".join(failures))
    full_attention_entries = [
        cache[index]
        for index, kind, _leaves in _STATE_SPEC
        if kind == VERIFY_SPEC_KIND_FULL_ATTN
    ]
    required_capacity = max(int(entry.size()) for entry in full_attention_entries) + reserve
    _validate_request_cache(cache, required_capacity=required_capacity)

    bank = CompiledVerifyBank(
        runtime,
        max_verify_len=2,
        request_max_tokens=int(max_tokens),
        capture_backend="stock",
    )
    if bank.permanent_eager:
        _fail("compiled A3B target-prefix requires affine q4/group64 target weights")
    bank._spec = list(_STATE_SPEC)
    bank._ensure_shadow(cache)
    if bank._resolve_bucket(cache, 2) != 0:
        _fail("compiled A3B target-prefix requires the fixed dense bucket-0 cache")
    bank._clear_shadow_leaf_refs()

    state_slots: list[tuple[list[Any], int]] = []
    rollback_slots: list[list[Any]] = []
    for index, kind, leaves in _STATE_SPEC:
        entry = cache[index]
        state_slots.extend((entry.cache, slot) for slot in range(leaves))
        if kind == VERIFY_SPEC_KIND_FULL_ATTN:
            rollback_slots.append(entry.rollback_state)

    return A3BK1TargetPrefixRoute(
        bank=bank,
        cache=cache,
        compiled_m2=_shared_m2_step(runtime, bank, hidden_variant),
        state_slots=tuple(state_slots),
        rollback_slots=tuple(rollback_slots),
        request_max_tokens=int(max_tokens),
        growth_reserve_tokens=reserve,
        prompt_tokens=int(prompt_tokens),
    )
