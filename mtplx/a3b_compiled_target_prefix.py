"""Exact compiled target-prefix route for the A3B K1 decode contract."""

from __future__ import annotations

import hashlib
import json
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
_FULL_ATTENTION_CACHE_STEP = 256
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
        runtime = host["runtime_ref"]()
        if runtime is None:
            _fail("compiled A3B target-prefix runtime was released")
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
            logits, hidden, captures = runtime._forward_ar_capture_a3b_postconv(
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
        runtime = host["runtime_ref"]()
        if runtime is None:
            _fail("compiled A3B target-prefix runtime was released")
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
            logits, hidden, _captures = runtime._forward_ar_capture_a3b_postconv(
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
        "runtime_ref": weakref.ref(runtime),
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
        "runtime_ref": weakref.ref(runtime),
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
    request_preflight_key: str | None = None
    request_preflight_status: str = "not_required"

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
            "request_preflight_key": self.request_preflight_key,
            "request_preflight_status": self.request_preflight_status,
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


def _array_signature(value: Any) -> tuple[tuple[int, ...], str]:
    return (
        tuple(int(dimension) for dimension in value.shape),
        str(value.dtype),
    )


def _route_state_signature(
    route: A3BK1TargetPrefixRoute,
) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        _array_signature(container[slot]) for container, slot in route.state_slots
    )


def _route_compile_specialization_key(
    route: A3BK1TargetPrefixRoute,
    *,
    hidden_variant: str | None,
) -> str:
    payload = {
        "hidden_variant": str(hidden_variant or ""),
        "m2_input": ((1, 2), "int32"),
        "m1_input": ((1, 1), "int32"),
        "state_spec": _STATE_SPEC,
        "m2_state": _route_state_signature(route),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    require_request_preflight: bool = False,
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

    route = A3BK1TargetPrefixRoute(
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
    if require_request_preflight:
        specialization_key = _route_compile_specialization_key(
            route,
            hidden_variant=hidden_variant,
        )
        certificates = runtime._a3b_whole_moe_request_preflights
        if specialization_key not in certificates:
            _fail(
                "compiled A3B target-prefix request geometry was not preflighted "
                "before generation"
            )
        route.request_preflight_key = specialization_key
        route.request_preflight_status = "matched"
    return route


def _request_cache_capacity(*, prompt_tokens: int, max_tokens: int) -> int:
    needed = int(prompt_tokens) + int(max_tokens) + 2
    if int(prompt_tokens) <= 0 or int(max_tokens) <= 0:
        _fail("compiled A3B target-prefix preflight requires positive request geometry")
    if needed - 2 > _MAX_REQUEST_CONTEXT:
        _fail("compiled A3B target-prefix request exceeds its installed context ceiling")
    return (
        (needed + _FULL_ATTENTION_CACHE_STEP - 1) // _FULL_ATTENTION_CACHE_STEP
    ) * _FULL_ATTENTION_CACHE_STEP


def preflight_a3b_k1_target_prefix_full_graph(
    runtime: Any,
    factory: A3BCompiledTargetPrefixFactory,
    *,
    cache: list[Any],
    prompt_tokens: int,
    max_tokens: int,
    hidden_variant: str | None,
) -> dict[str, Any]:
    """Compile exact request-shaped target M2/M1 graphs over disposable state."""

    route = install_a3b_k1_target_prefix_route(
        runtime,
        cache,
        factory=factory,
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        verify_strategy="target_prefix",
        speculative_depth=1,
        requested_speculative_depth=1,
        verify_core="stock",
        hidden_variant=hidden_variant,
        state_rebase_every=0,
        require_request_preflight=False,
    )
    m2_input = mx.array([[0, 1]])
    m2_state = tuple(
        container[slot] for container, slot in route.state_slots
    )
    m2_outputs = tuple(route.compiled_m2(m2_input, *m2_state))
    if len(m2_outputs) != 182:
        _fail("compiled A3B target-prefix M2 preflight returned invalid output ownership")
    mx.eval(*m2_outputs)
    primary_state = tuple(m2_outputs[_PRIMARY_STATE_START:_FINAL_STATE_START])
    m1_input = mx.array([[0]])
    m1_outputs = tuple(route.compiled_m1(m1_input, *primary_state))
    if len(m1_outputs) != 92:
        _fail("compiled A3B target-prefix M1 preflight returned invalid output ownership")
    mx.eval(*m1_outputs)
    m2_logits, m2_hidden = m2_outputs[:2]
    m1_logits, m1_hidden = m1_outputs[:2]

    if (
        tuple(m2_hidden.shape) != (1, 2, 2048)
        or tuple(m1_hidden.shape) != (1, 1, 2048)
        or m2_hidden.dtype != mx.bfloat16
        or m1_hidden.dtype != mx.bfloat16
    ):
        _fail("compiled A3B target-prefix full-graph preflight returned invalid hidden ownership")

    key_shapes = {
        tuple(int(dimension) for dimension in route.cache[index].cache[0].shape)
        for index in _FULL_ATTENTION_INDICES
    }
    value_shapes = {
        tuple(int(dimension) for dimension in route.cache[index].cache[1].shape)
        for index in _FULL_ATTENTION_INDICES
    }
    expected_capacity = _request_cache_capacity(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
    )
    expected_shape = (1, 2, expected_capacity, 256)
    if key_shapes != {expected_shape} or value_shapes != {expected_shape}:
        _fail("compiled A3B target-prefix preflight returned invalid cache geometry")
    specialization_key = _route_compile_specialization_key(
        route,
        hidden_variant=hidden_variant,
    )
    return {
        "canonical_key": specialization_key,
        "full_attention_key_shape": list(expected_shape),
        "full_attention_value_shape": list(expected_shape),
        "hidden_variant": str(hidden_variant or ""),
        "m2_input_signature": _array_signature(m2_input),
        "m1_input_signature": _array_signature(m1_input),
        "m2_state_signature": _route_state_signature(route),
        "m1_primary_signature": tuple(
            _array_signature(value) for value in primary_state
        ),
        "m2_logits_signature": _array_signature(m2_logits),
        "m2_hidden_signature": _array_signature(m2_hidden),
        "m1_logits_signature": _array_signature(m1_logits),
        "m1_hidden_signature": _array_signature(m1_hidden),
        "m2_final_state_signature": tuple(
            _array_signature(value) for value in m2_outputs[_FINAL_STATE_START:]
        ),
        "m1_final_state_signature": tuple(
            _array_signature(value) for value in m1_outputs[_M1_FINAL_STATE_START:]
        ),
        "m2_output_count": len(m2_outputs),
        "m1_output_count": len(m1_outputs),
        "lanes": {
            "a3b_whole_moe_request_full_graph_m1": "ok",
            "a3b_whole_moe_request_full_graph_m2": "ok",
        },
    }


def _preflight_a3b_k1_target_prefix_request_geometry(
    runtime: Any,
    factory: A3BCompiledTargetPrefixFactory,
    *,
    prompt_tokens: int,
    max_tokens: int,
    hidden_variant: str | None,
    cache_factory: Callable[[], list[Any]],
    prefill_layout: str,
) -> dict[str, Any]:
    """Build fixed-shape state without redundantly evaluating the full prompt."""

    cache = cache_factory()
    with attention_phase("prefill"):
        prefill_logits, prefill_hidden = runtime.forward_ar(
            mx.array([[0]]),
            cache=cache,
            return_hidden=True,
            hidden_variant=hidden_variant,
        )
    mx.eval(prefill_logits, prefill_hidden)
    for index in _FULL_ATTENTION_INDICES:
        entry = cache[index]
        entry.offset = int(prompt_tokens)
    proof = preflight_a3b_k1_target_prefix_full_graph(
        runtime,
        factory,
        cache=cache,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        hidden_variant=hidden_variant,
    )
    proof["prefill_layout"] = str(prefill_layout)
    return proof


def ensure_a3b_whole_moe_request_preflight(
    runtime: Any,
    factory: A3BCompiledTargetPrefixFactory,
    *,
    prompt_tokens: int,
    max_tokens: int,
    hidden_variant: str | None,
    cache_factory: Callable[[], list[Any]],
    prefill_layout: str,
) -> dict[str, Any]:
    """Prime one compiled graph per exact cache shape before generation."""

    capacity = _request_cache_capacity(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
    )
    logical_key = (capacity, str(hidden_variant or ""), str(prefill_layout))
    proofs = runtime._a3b_whole_moe_request_preflights
    geometry_keys = runtime._a3b_whole_moe_request_geometry_keys
    canonical_key = geometry_keys.get(logical_key)
    proof = None if canonical_key is None else proofs.get(canonical_key)
    if proof is None:
        proof = _preflight_a3b_k1_target_prefix_request_geometry(
            runtime,
            factory,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            hidden_variant=hidden_variant,
            cache_factory=cache_factory,
            prefill_layout=prefill_layout,
        )
        canonical_key = str(proof["canonical_key"])
        proofs[canonical_key] = proof
        geometry_keys[logical_key] = canonical_key
    return {
        **proof,
        "status": "ok",
        "prompt_tokens": int(prompt_tokens),
        "max_tokens": int(max_tokens),
        "growth_reserve_tokens": int(max_tokens) + 2,
        "full_attention_layers": len(_FULL_ATTENTION_INDICES),
        "m1_rows": 1,
        "m2_rows": 2,
    }


def preflight_a3b_k1_target_prefix_load_graph(
    runtime: Any,
    factory: A3BCompiledTargetPrefixFactory,
) -> dict[str, str]:
    """Prove minimum full-graph compatibility before committing installation."""

    proof = _preflight_a3b_k1_target_prefix_request_geometry(
        runtime,
        factory,
        prompt_tokens=1,
        max_tokens=2,
        hidden_variant=None,
        cache_factory=runtime.make_cache,
        prefill_layout="load_probe",
    )
    lanes = proof["lanes"]
    return {
        "a3b_whole_moe_target_prefix_full_graph_m1": lanes[
            "a3b_whole_moe_request_full_graph_m1"
        ],
        "a3b_whole_moe_target_prefix_full_graph_m2": lanes[
            "a3b_whole_moe_request_full_graph_m2"
        ],
    }
