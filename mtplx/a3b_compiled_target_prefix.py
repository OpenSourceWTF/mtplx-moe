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
    TensorOffsetKVCache,
    VERIFY_SPEC_KIND_FULL_ATTN,
    VERIFY_SPEC_KIND_GDN,
    _compiled_verify_boundary,
    _compiled_verify_donation_enabled,
    _owned_state_env_active,
)
from .gdn_capture import A3BGDNPostconvFactory


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
_STATE_LEAVES = sum(leaves for _index, _kind, leaves in _STATE_SPEC)
_FULL_ATTENTION_INDICES = tuple(
    index
    for index, kind, _leaves in _STATE_SPEC
    if kind == VERIFY_SPEC_KIND_FULL_ATTN
)
_PRIMARY_STATE_START = 2
_FINAL_STATE_START = _PRIMARY_STATE_START + _STATE_LEAVES
_M1_FINAL_STATE_START = 2
_MAX_REQUEST_CONTEXT = 12_288
_SHARED_M2_STEPS: dict[
    tuple[int, str],
    tuple[Callable[..., Any], dict[str, Any], weakref.ReferenceType[Any]],
] = {}
_SHARED_M1_STEPS: dict[
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
    gdn_postconv: A3BGDNPostconvFactory


def _enabled() -> bool:
    return os.environ.get("MTPLX_COMPILED_TARGET_PREFIX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fail(message: str) -> None:
    raise A3BCompiledTargetPrefixConfigError(message)


def validate_a3b_k1_target_prefix_sampler(sampler: Any) -> None:
    """Prove the external sampler contract once before prompt construction."""
    if float(sampler.temperature) <= 0.0 or int(sampler.top_k or 0) <= 0:
        _fail("compiled A3B target-prefix requires a stochastic top-k sampler")


def validate_a3b_k1_device_draft_request(
    draft_sampler: Any,
    *,
    draft_margin_threshold: float | None,
    adaptive_policy: Any | None,
    draft_core: str,
    online_correction_cache: bool,
    prompt_correction_cache: bool,
    adapter_ensemble_q: bool,
    mtp_topk_reranker: Any | None,
    loop_guard: bool,
    presence_penalty: float,
    frequency_penalty: float,
) -> None:
    """Prove once that the installed K1 lane can keep its draft on-device."""
    unsupported_sampler = (
        float(draft_sampler.temperature) > 0.0
        and int(draft_sampler.top_k or 0) <= 0
        and 0.0 < float(draft_sampler.top_p) < 1.0
    )
    host_only_modifier = any(
        (
            draft_margin_threshold is not None,
            adaptive_policy is not None,
            str(draft_core) != "stock",
            bool(online_correction_cache),
            bool(prompt_correction_cache),
            bool(adapter_ensemble_q),
            mtp_topk_reranker is not None,
            bool(loop_guard),
            bool(presence_penalty),
            bool(frequency_penalty),
        )
    )
    if unsupported_sampler or host_only_modifier:
        _fail(
            "compiled A3B device draft requires the fixed stock K1 sampler contract"
        )


def prepare_a3b_compiled_target_prefix(
    model: Any,
    *,
    config: dict[str, Any],
    gdn_postconv_factory: A3BGDNPostconvFactory | None = None,
) -> A3BCompiledTargetPrefixFactory | None:
    """Validate checkpoint-owned facts once, while the model is constructed."""
    if not _enabled():
        return None
    if gdn_postconv_factory is None:
        _fail("compiled A3B target-prefix requires the constructed GDN postconv factory")

    text = config["text_config"]
    quant = config.get("quantization") or config.get("quantization_config")
    if (
        int(text.get("num_attention_heads", -1)) != 16
        or int(text.get("num_key_value_heads", -1)) != 2
        or int(text.get("head_dim", -1)) != 256
        or int(text.get("mtp_num_hidden_layers", -1)) != 1
        or not isinstance(quant, dict)
        or int(quant.get("bits", -1)) != 4
        or int(quant.get("group_size", -1)) != 64
        or str(quant.get("mode", "")) != "affine"
    ):
        _fail("compiled A3B target-prefix requires the exact q4/group64 A3B config")

    if len(model.mtp.layers) != 1:
        _fail("compiled A3B target-prefix requires one constructed MTP layer")
    layers = model.language_model.model.layers
    for index in _FULL_ATTENTION_INDICES:
        attention = getattr(layers[index], "self_attn", None)
        if (
            attention is None
            or getattr(attention, "sharding_group", None) is not None
            or int(getattr(attention, "num_attention_heads", -1)) != 16
            or int(getattr(attention, "num_key_value_heads", -1)) != 2
            or int(getattr(attention, "head_dim", -1)) != 256
        ):
            _fail(f"compiled A3B target-prefix attention ownership missing at layer {index}")

    factory = A3BCompiledTargetPrefixFactory(
        layer_types=_LAYER_TYPES,
        gdn_layers=30,
        full_attention_layers=10,
        hidden_size=2048,
        quantization="affine_q4_group64",
        gdn_postconv=gdn_postconv_factory,
    )
    return factory


def _make_a3b_k1_target_prefix_m2_step(
    *,
    host: dict[str, Any],
) -> Callable[..., Any]:
    """Build the fixed M2 trace body; Python executes only while tracing."""
    spec = _STATE_SPEC

    def step(input_ids, *state_in):
        shadow = host["shadow"]
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
            logits, hidden, captures = host[
                "runtime"
            ]._forward_ar_capture_a3b_postconv(
                input_ids,
                cache=shadow,
                hidden_variant=host["hidden_variant"],
                postconv_implementations=host["postconv_implementations"],
            )
        primary_state: list[Any] = []
        final_state: list[Any] = []
        for index, kind, _leaves in spec:
            entry = shadow[index]
            if kind == VERIFY_SPEC_KIND_GDN:
                layer_capture = captures[index]
                primary_state.extend(
                    (
                        layer_capture["conv_states"][:, 0, :, :],
                        layer_capture["states"][:, 0, :, :, :],
                    )
                )
            else:
                primary_state.extend(
                    (entry.cache[0], entry.cache[1], entry.cache[2] - 1)
                )
            final_state.extend(entry.cache)
        return (logits, hidden, *primary_state, *final_state)

    return step


def _make_a3b_k1_target_prefix_m1_step(
    *,
    host: dict[str, Any],
) -> Callable[..., Any]:
    """Build the fixed M1 continuation trace; Python runs only while tracing."""
    spec = _STATE_SPEC

    def step(input_ids, *state_in):
        shadow = host["shadow"]
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
            logits, hidden, _captures = host[
                "runtime"
            ]._forward_ar_capture_a3b_postconv(
                input_ids,
                cache=shadow,
                hidden_variant=host["hidden_variant"],
                postconv_implementations=host["postconv_implementations"],
            )
        final_state: list[Any] = []
        for index, _kind, _leaves in spec:
            final_state.extend(shadow[index].cache)
        return (logits, hidden, *final_state)

    return step


def _shared_m2_step(
    runtime: Any,
    shadow: list[Any],
    hidden_variant: str | None,
    postconv_implementations: tuple[Callable[..., Any], ...],
) -> Callable[..., Any]:
    key = (id(runtime), str(hidden_variant or ""))
    entry = _SHARED_M2_STEPS.get(key)
    if entry is not None:
        compiled, host, runtime_ref = entry
        if runtime_ref() is runtime:
            host["shadow"] = shadow
            return compiled
        _SHARED_M2_STEPS.pop(key, None)
    host = {
        "shadow": shadow,
        "runtime": runtime,
        "hidden_variant": hidden_variant,
        "postconv_implementations": postconv_implementations,
    }
    compiled = mx.compile(_make_a3b_k1_target_prefix_m2_step(host=host))
    _SHARED_M2_STEPS[key] = (compiled, host, weakref.ref(runtime))
    return compiled


def _shared_m1_step(
    runtime: Any,
    shadow: list[Any],
    hidden_variant: str | None,
    postconv_implementations: tuple[Callable[..., Any], ...],
) -> Callable[..., Any]:
    key = (id(runtime), str(hidden_variant or ""))
    entry = _SHARED_M1_STEPS.get(key)
    if entry is not None:
        compiled, host, runtime_ref = entry
        if runtime_ref() is runtime:
            host["shadow"] = shadow
            return compiled
        _SHARED_M1_STEPS.pop(key, None)
    host = {
        "shadow": shadow,
        "runtime": runtime,
        "hidden_variant": hidden_variant,
        "postconv_implementations": postconv_implementations,
    }
    compiled = mx.compile(_make_a3b_k1_target_prefix_m1_step(host=host))
    _SHARED_M1_STEPS[key] = (compiled, host, weakref.ref(runtime))
    return compiled


@dataclass
class A3BK1TargetPrefixRoute:
    """Request-owned fixed M2 verifier and captured-primary M1 continuation."""

    cache: list[Any]
    compiled_m2: Callable[..., Any]
    compiled_m1: Callable[..., Any]
    state_slots: tuple[tuple[list[Any], int], ...]
    rollback_slots: tuple[list[Any], ...]
    request_max_tokens: int
    growth_reserve_tokens: int
    prompt_tokens: int

    def verify_m2(self, input_ids):
        return self._forward_m2(input_ids)

    def repair_m1(self, input_ids, primary_state):
        return self._forward_m1(input_ids, primary_state)

    def _forward_m2(self, input_ids):
        state_in = [container[slot] for container, slot in self.state_slots]
        outputs = self.compiled_m2(input_ids, *state_in)
        for (container, slot), value in zip(
            self.state_slots,
            outputs[_FINAL_STATE_START:],
        ):
            container[slot] = value
        for rollback in self.rollback_slots:
            rollback[0] = None
            rollback[1] = None
            rollback[2] = None
        mx.async_eval(*outputs)
        primary_state = tuple(outputs[_PRIMARY_STATE_START:_FINAL_STATE_START])
        return outputs[0], outputs[1], primary_state

    def _forward_m1(self, input_ids, primary_state):
        outputs = self.compiled_m1(input_ids, *primary_state)
        for (container, slot), value in zip(
            self.state_slots,
            outputs[_M1_FINAL_STATE_START:],
        ):
            container[slot] = value
        for rollback in self.rollback_slots:
            rollback[0] = None
            rollback[1] = None
            rollback[2] = None
        mx.async_eval(*outputs)
        return outputs[0], outputs[1], None

    def demote(self) -> int:
        for index in _FULL_ATTENTION_INDICES:
            self.cache[index] = self.cache[index].demote()
        return 10

    def final_report(self, *, verify_calls: int, repair_calls: int) -> dict[str, Any]:
        m2_calls = int(verify_calls)
        m1_calls = int(repair_calls)
        compiled_calls = m2_calls + m1_calls
        return {
            "mode": "a3b_k1_target_prefix",
            "installed": True,
            "installation_status": "installed",
            "calls": compiled_calls,
            "compiled_calls": compiled_calls,
            "m2_calls": m2_calls,
            "m1_calls": m1_calls,
            "m2_verify_calls": m2_calls,
            "m1_repair_calls": m1_calls,
            "buckets": {"m2_verify:0": m2_calls, "m1_repair:0": m1_calls},
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
            "compiled_entry_count": 2,
            "compiled_keys": ["m2:verify:b0", "m1:repair:b0"],
            "permanent_eager": False,
        }


def _construct_a3b_target_shadow(cache: list[Any]) -> list[Any]:
    """Construct the fixed shadow topology from trusted promoted ownership."""
    shadow: list[Any] = [None] * 40
    for index, kind, _leaves in _STATE_SPEC:
        if kind == VERIFY_SPEC_KIND_GDN:
            shadow[index] = ArraysCache(2)
        else:
            source = cache[index]
            entry = TensorOffsetKVCache(
                source.cache[0],
                source.cache[1],
                source.cache[2],
                step=source.step,
            )
            entry.cache = [None, None, None]
            shadow[index] = entry
    return shadow


def _construct_a3b_target_cache(
    cache: list[Any],
    *,
    reserve_tokens: int,
) -> list[Any]:
    """Promote the ten proven attention positions and construct their shadow."""
    for index in _FULL_ATTENTION_INDICES:
        cache[index] = TensorOffsetKVCache.from_kv_cache(
            cache[index],
            reserve_tokens=reserve_tokens,
        )
    return _construct_a3b_target_shadow(cache)


def install_a3b_k1_target_prefix_route(
    runtime: Any,
    cache: list[Any],
    *,
    factory: A3BCompiledTargetPrefixFactory,
    max_tokens: int,
    prompt_tokens: int,
    verify_strategy: str,
    speculative_depth: int,
    requested_speculative_depth: int,
    verify_core: str,
    hidden_variant: str | None,
    state_rebase_every: int,
) -> A3BK1TargetPrefixRoute:
    """Validate external request facts and construct the fixed route."""
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
    shadow = _construct_a3b_target_cache(
        cache,
        reserve_tokens=reserve,
    )

    state_slots: list[tuple[list[Any], int]] = []
    rollback_slots: list[list[Any]] = []
    for index, kind, leaves in _STATE_SPEC:
        entry = cache[index]
        state_slots.extend((entry.cache, slot) for slot in range(leaves))
        if kind == VERIFY_SPEC_KIND_FULL_ATTN:
            rollback_slots.append(entry.rollback_state)

    return A3BK1TargetPrefixRoute(
        cache=cache,
        compiled_m2=_shared_m2_step(
            runtime,
            shadow,
            hidden_variant,
            factory.gdn_postconv.m2_implementations,
        ),
        compiled_m1=_shared_m1_step(
            runtime,
            shadow,
            hidden_variant,
            factory.gdn_postconv.m1_implementations,
        ),
        state_slots=tuple(state_slots),
        rollback_slots=tuple(rollback_slots),
        request_max_tokens=int(max_tokens),
        growth_reserve_tokens=reserve,
        prompt_tokens=int(prompt_tokens),
    )
