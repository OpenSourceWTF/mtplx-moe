"""Reference AR and native-MTP generation loops.

These loops intentionally favor correctness and observability over speed. The
optimized runtime can tighten the same contracts after the MTP-1 gates pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal

import mlx.core as mx
import numpy as np

from .a3b_compiled_target_prefix import (
    ensure_a3b_whole_moe_request_preflight as _ensure_a3b_whole_moe_request_preflight,
    install_a3b_k1_target_prefix_route,
    validate_a3b_k1_device_draft_request,
    validate_a3b_k1_target_prefix_sampler,
)
from .a3b_whole_moe import validate_a3b_whole_moe_request
from .adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy
from .attention_context import attention_phase, model_forward_kind
from .deepseek_v4_adaptive_width import (
    validate_installed_deepseek_v4_adaptive_width_policy,
)
from .progress_heartbeat import tick as _owner_progress_tick
from .cache_state import (
    detach_array_leaf,
    detach_cache_state,
    owned_recurrent_state_stats,
    restore_cache,
    rollback_after_verify,
    trim_verified_window_without_snapshot,
    snapshot_cache,
    snapshot_untrimmable_cache,
    tail_owned_attention_kv_stats,
    trim_verified_window_to_prefix,
)
from .fast_sampling import (
    MAX_DEVICE_TOP_K_ORDER,
    BatchedSparseDistributions,
    apply_penalties_mlx,
    batched_sparse_distributions_from_mlx_logits,
    sample_token_ids_from_mlx_logits,
    sparse_distribution_from_mlx_logits,
    sparse_distributions_from_mlx_logits,
)
from .gdn_capture import resolve_gdn_capture_backend
from .graphbank import (
    CompiledVerifyBank,
    SpecDecodeGraphBank,
    cache_array_tree,
    compiled_verify_mode,
    paged_offsets_context_ok as _paged_offsets_context_ok,
    promote_kv_cache_offsets,
    set_paged_offsets_context_ok,
)
from .native_mlp import set_native_mlp_context
from .loop_guard import LoopGuard, loop_guard_config_from_env
from .thinking_guard import ThinkingGuard, ThinkingGuardConfig
from .profiles import resolve_long_context_mtp_depth
from .runtime import MTPLXRuntime
from .sampling import (
    SamplerConfig,
    SparseDistribution,
    acceptance_probability as compute_acceptance_probability,
    distribution_from_logits as dense_distribution_from_logits,
    residual_distribution,
    sample_from_distribution,
)
from .session_bank import _boundary_true_restore_enabled
from .runtime_options import block_prefix_restore_enabled, env_bool

Mode = Literal["ar", "mtp1", "mtpk", "mtpa"]
VerifyStrategy = Literal[
    "batched",
    "sequential",
    "capture",
    "capture_commit",
    "graphbank",
    "graphbank_capture_commit",
    "target_prefix",
    "trim_commit",
]

_PREFILL_CHUNK_SIZE_OVERRIDE: ContextVar[int | None] = ContextVar(
    "mtplx_prefill_chunk_size_override",
    default=None,
)


def reject_non_k1_a3b_whole_moe_request(rt: MTPLXRuntime, *, entrypoint: str) -> None:
    """Reject unsupported generation modes once, before they construct a prompt.

    generate_ar is supported: every one of its decode forwards is a single
    row, which the installed M1 route serves with per-row arithmetic that
    bit-matches the M2 verify route (enforced at install by the
    a3b_whole_moe_target_m1_m2_row_parity selfcheck lane).  Pure AR under
    whole-MoE is the ground-truth arm of the K1 AR-exactness gate.
    """

    if entrypoint == "generate_ar":
        return
    if bool(getattr(rt, "a3b_whole_moe_installed", False)):
        raise RuntimeError(
            f"installed A3B whole-MoE is owned by exact K1 generate_mtpk, not {entrypoint}"
        )


def ensure_a3b_whole_moe_request_preflight(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    base_hidden_variant: str,
    prefill_layout: str | None = None,
) -> dict[str, Any]:
    """Prime the installed exact request geometry before prompt generation."""

    if not bool(getattr(rt, "a3b_whole_moe_installed", False)):
        return {"status": "disabled"}
    os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] = str(len(prompt_ids))
    layout = _sustained_prefill_layout() if prefill_layout is None else prefill_layout
    return _ensure_a3b_whole_moe_request_preflight(
        rt,
        rt.a3b_compiled_target_prefix_factory,
        prompt_tokens=len(prompt_ids),
        max_tokens=max_tokens,
        hidden_variant=base_hidden_variant,
        cache_factory=lambda: _make_target_prefill_cache(rt),
        prefill_layout=layout,
    )


def _resolve_runtime_mtp_hidden_variant(
    rt: MTPLXRuntime,
    requested: str | None,
) -> str:
    if requested in {None, "auto", "contract"}:
        return str(getattr(rt.contract, "hidden_variant", "post_norm") or "post_norm")
    return str(requested)


def _resolve_runtime_base_hidden_variant(
    rt: MTPLXRuntime,
    requested: str | None,
) -> str:
    if requested in {None, "auto", "contract"}:
        return str(getattr(rt.contract, "base_hidden_variant", "post_norm") or "post_norm")
    return str(requested)


def _resolve_runtime_mtp_position_mode(rt: MTPLXRuntime) -> str:
    raw = os.environ.get("MTPLX_MTP_POSITION_MODE")
    if raw is None:
        raw = getattr(rt.contract, "mtp_position_mode", "cache")
    normalized = str(raw or "cache").strip().lower().replace("-", "_")
    if normalized in {"", "0", "off", "false", "default", "cache", "local"}:
        return "cache"
    return normalized


def _eval_value_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": "array",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": [str(key) for key in value.keys()],
            "items": {
                str(key): _eval_value_summary(item) for key, item in value.items()
            },
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [_eval_value_summary(item) for item in value],
        }
    return {"type": type(value).__name__}


def _eval(*values: Any, _caller_depth: int = 1) -> None:
    audit_path = os.environ.get("MTPLX_EVAL_AUDIT")
    if not audit_path:
        mx.eval(*values)
        # Every settled engine forward (prefill chunk, verify, AR step) proves
        # the model owner is alive; the stream stall watchdog compares readings.
        _owner_progress_tick()
        return

    try:
        caller = sys._getframe(_caller_depth)
    except ValueError:
        caller = None
    started = time.perf_counter()
    mx.eval(*values)
    _owner_progress_tick()
    elapsed_s = time.perf_counter() - started
    entry = {
        "elapsed_s": elapsed_s,
        "function": caller.f_code.co_name if caller is not None else None,
        "line": caller.f_lineno if caller is not None else None,
        "values": [_eval_value_summary(value) for value in values],
    }
    out = Path(audit_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    if os.environ.get("MTPLX_EVAL_AUDIT_STDERR"):
        print(json.dumps(entry, sort_keys=True), file=sys.stderr)


def _env_enabled_default_on(name: str) -> bool:
    """Opt-out env read: unset resolves ON, "0"/"false"/"no"/"off" disables.

    The greedy-trio knobs (#313/#315c1/#318) moved to this resolution on the
    night-20260822 round-4 ruling (n=4 counterbalanced ABBA blend +2.7% mean,
    byte-identity held on greedy and sampled-seed lanes). Same falsy set as
    graphbank._batch_paged_offsets_enabled so the trio reads stay symmetric.
    """
    return str(os.environ.get(name, "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _trio_max_context() -> int:
    """Prompt-token fence for the greedy-trio defaults (0 = no fence).

    Night-20260822 receipts: the trio stack blends +2.5..+9.8% on the
    0.5k-8k rungs but measured −2.9%/−2.7% at 16k/32k in the dedicated
    order-symmetric quad — so the defaults route by context, the same
    pattern as MTPLX_COMPILED_VERIFY_MAX_CONTEXT. Decided once per request
    from the prompt length (a request that grows past the fence mid-decode
    keeps its entry decision).
    """
    raw = os.environ.get("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "12288").strip().lower()
    if raw in ("0", "off", "none", "unlimited"):
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 12288


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_falsey(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _skip_verify_snapshot() -> bool:
    """The single parse of ``MTPLX_SKIP_VERIFY_SNAPSHOT`` (default OFF).

    The serve fast path force-sets this to "1"; whether that is safe is
    decided by the verify strategy, and the server now answers that from an
    explicit list of strategies known to survive without the snapshot
    rather than from a two-element list of the ones that need it.
    """

    return env_bool("MTPLX_SKIP_VERIFY_SNAPSHOT", default=False)


def _runtime_skip_verify_snapshot(rt: Any) -> bool:
    """Resolve snapshot ownership once from the installed model contract."""

    if getattr(rt.model, "speculative_cache_mode", None) == "snapshot_rollback":
        return False
    return _skip_verify_snapshot()


def _draft_confidence_trace() -> bool:
    """Head-cal diagnostic (default OFF): record the draft head's softmax
    p(drafted token) per depth and attribute it to accept/reject at verify.
    Greedy lane only — under temperature the drafted token is not the argmax
    and its shaped distribution is not a raw softmax."""

    return env_bool("MTPLX_DRAFT_CONFIDENCE_TRACE", default=False)


def _draft_confidence_width_threshold() -> float | None:
    """Head-cal leg 2b (default OFF): stop drafting the cycle once the draft
    head's p(drafted) falls below this threshold. The triggering draft is
    KEPT (native gated-stop semantics); only deeper drafts are skipped, so
    committed output tokens are invariant — the knob trades speculation
    width against doomed-draft verify work. Greedy stock loop only."""

    raw = os.environ.get("MTPLX_DRAFT_CONFIDENCE_WIDTH_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 < value < 1.0 else None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _generation_rate_fields(
    *,
    generated_tokens: int,
    elapsed_s: float,
    prompt_eval_time_s: float,
    cache_restore_time_s: float = 0.0,
) -> dict[str, float]:
    end_to_end_tok_s = generated_tokens / elapsed_s if elapsed_s > 0.0 else 0.0
    non_decode_elapsed_s = min(
        max(0.0, prompt_eval_time_s) + max(0.0, cache_restore_time_s),
        max(0.0, elapsed_s),
    )
    decode_elapsed_s = max(0.0, elapsed_s - non_decode_elapsed_s)
    decode_tok_s = (
        generated_tokens / decode_elapsed_s if decode_elapsed_s > 0.0 else 0.0
    )
    return {
        "tok_s": decode_tok_s,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": end_to_end_tok_s,
    }


def _normalize_mtp_history_policy(policy: str | None) -> str:
    normalized = (policy or "cycle").strip().lower().replace("-", "_")
    aliases = {
        "full": "committed",
        "lastwindow": "last_window",
        "window": "last_window",
        "none": "cycle",
        "off": "cycle",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "cycle", "committed", "last_window"}:
        raise ValueError(
            "mtp_history_policy must be 'auto', 'cycle', 'committed', "
            "'full', 'last_window', or 'none'"
        )
    return normalized


def _mtp_history_uses_committed_cache(policy: str) -> bool:
    return _normalize_mtp_history_policy(policy) in {"committed", "last_window"}


def _mtp_history_last_window_tokens() -> int:
    return max(1, _env_int("MTPLX_MTP_HISTORY_LAST_WINDOW", 8192))


def _resolve_mtp_history_policy(requested_policy: str, prompt_tokens: int) -> str:
    requested = _normalize_mtp_history_policy(requested_policy)
    env_policy = os.environ.get("MTPLX_MTP_HISTORY_POLICY")
    # Honor the env-var override whenever the caller requested either the
    # product default "committed" or the auto-resolution path. This keeps
    # diagnostic history-policy overrides reachable from the server hot path.
    if env_policy and requested in ("committed", "auto"):
        requested = _normalize_mtp_history_policy(env_policy)
    if requested != "auto":
        return requested
    threshold = max(
        1,
        _env_int("MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD", 16384),
    )
    return "last_window" if int(prompt_tokens) >= threshold else "committed"


def _runtime_count(rt: MTPLXRuntime, key: str, amount: int = 1) -> None:
    counters = getattr(rt, "diagnostic_counters", None)
    if counters is None:
        return
    counters[key] = int(counters.get(key, 0)) + int(amount)


def _runtime_counter_snapshot(rt: MTPLXRuntime) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in getattr(rt, "diagnostic_counters", {}).items()
    }


def _runtime_counter_delta(
    rt: MTPLXRuntime,
    before: dict[str, int],
) -> dict[str, int]:
    current = getattr(rt, "diagnostic_counters", {})
    keys = set(before) | set(current)
    return {
        str(key): int(current.get(key, 0)) - int(before.get(key, 0)) for key in keys
    }


def _attach_runtime_diagnostics(
    stats: "GenerationStats",
    rt: MTPLXRuntime,
    before: dict[str, int],
    *,
    ar_return_hidden: bool | None = None,
) -> None:
    counters = _runtime_counter_delta(rt, before)
    stats.runtime_mtp_enabled = bool(getattr(rt, "mtp_enabled", False))
    if ar_return_hidden is not None:
        stats.ar_return_hidden = bool(ar_return_hidden)
    stats.forward_ar_hidden_calls = int(counters.get("forward_ar_hidden_calls", 0))
    stats.forward_ar_plain_calls = int(counters.get("forward_ar_plain_calls", 0))
    stats.mtp_forward_calls = int(counters.get("draft_mtp_calls", 0))
    stats.make_mtp_cache_calls = int(counters.get("make_mtp_cache_calls", 0))
    stats.update_mtp_cache_calls = int(counters.get("update_mtp_cache_calls", 0))
    stats.mtp_history_append_calls = int(counters.get("mtp_history_append_calls", 0))
    stats.full_logits_tokens_emitted = int(
        counters.get("full_logits_tokens_emitted", 0)
    )
    stats.final_logits_tokens_emitted = int(
        counters.get("final_logits_tokens_emitted", 0)
    )
    stats.logits_tokens_emitted = int(counters.get("logits_tokens_emitted", 0))
    stats.prefill_chunks = int(counters.get("prefill_chunks", 0))
    stats.prefill_chunk_size = _prefill_chunk_size()
    stats.prefill_chunk_cache_cleanup_enabled = _prefill_chunk_cache_cleanup_enabled()
    stats.prefill_chunk_cache_cleanup_every = _prefill_chunk_cache_cleanup_every()
    stats.prefill_chunk_cache_cleanup_events = int(
        counters.get("prefill_chunk_cache_cleanup_events", 0)
    )
    stats.prefill_stock_cache_only_enabled = _prefill_stock_cache_only_enabled()
    stats.prefill_stock_cache_only_calls = int(
        counters.get("prefill_stock_cache_only_calls", 0)
    )
    stats.prefill_omlx_external_enabled = _prefill_omlx_external_enabled()
    stats.prefill_omlx_external_calls = int(
        counters.get("prefill_omlx_external_calls", 0)
    )
    stats.prefill_external_emit_logits_enabled = _prefill_external_emit_logits_enabled()
    stats.prefill_external_cache_only_calls = int(
        counters.get("prefill_external_cache_only_calls", 0)
    )
    owned_attn = stats.owned_attn_kv if isinstance(stats.owned_attn_kv, dict) else {}
    stats.paged_kv_capacity_tokens = int(owned_attn.get("capacity") or 0)
    stats.paged_kv_num_blocks = int(owned_attn.get("num_blocks") or 0)
    stats.paged_active_array_calls = int(owned_attn.get("active_array_calls") or 0)
    stats.paged_active_array_time_s = float(
        owned_attn.get("active_array_time_s") or 0.0
    )
    stats.paged_turboquant = bool(owned_attn.get("turboquant") or False)
    stats.paged_turboquant_k_quant = str(owned_attn.get("turboquant_k_quant") or "")
    stats.paged_turboquant_v_quant = str(owned_attn.get("turboquant_v_quant") or "")
    stats.paged_turboquant_attention_calls = int(
        owned_attn.get("turboquant_attention_calls") or 0
    )
    stats.paged_kv_quant = bool(owned_attn.get("kv_quant") or False)
    stats.paged_kv_quant_mode = str(owned_attn.get("kv_quant_mode") or "")
    stats.paged_kv_quant_attention_calls = int(
        owned_attn.get("kv_quant_attention_calls") or 0
    )
    stats.paged_kv_quant_dequant_calls = int(
        owned_attn.get("kv_quant_dequant_calls") or 0
    )
    stats.paged_kv_quant_dequant_time_s = float(
        owned_attn.get("kv_quant_dequant_time_s") or 0.0
    )
    stats.paged_kv_quant_dequant_tokens = int(
        owned_attn.get("kv_quant_dequant_tokens") or 0
    )
    stats.paged_kv_quant_dequant_memo_hits = int(
        owned_attn.get("kv_quant_dequant_memo_hits") or 0
    )
    stats.paged_kv_quant_dequant_memo_rebuilds = int(
        owned_attn.get("kv_quant_dequant_memo_rebuilds") or 0
    )
    stats.paged_kv_quant_kernel_calls = int(
        owned_attn.get("kv_quant_kernel_calls") or 0
    )
    stats.paged_gqa_sdpa_calls = int(owned_attn.get("gqa_sdpa_calls") or 0)
    gqa_by_route = owned_attn.get("gqa_sdpa_calls_by_route") or {}
    stats.paged_gqa_sdpa_calls_by_route = (
        dict(gqa_by_route) if isinstance(gqa_by_route, dict) else {}
    )
    gqa_by_phase = owned_attn.get("gqa_sdpa_calls_by_phase") or {}
    stats.paged_gqa_sdpa_calls_by_phase = (
        dict(gqa_by_phase) if isinstance(gqa_by_phase, dict) else {}
    )
    gqa_misses = owned_attn.get("gqa_sdpa_route_misses_by_phase_reason") or {}
    stats.paged_gqa_sdpa_route_misses_by_phase_reason = (
        dict(gqa_misses) if isinstance(gqa_misses, dict) else {}
    )
    gqa_misses_by_q = owned_attn.get("gqa_sdpa_route_misses_by_q_len") or {}
    stats.paged_gqa_sdpa_route_misses_by_q_len = (
        dict(gqa_misses_by_q) if isinstance(gqa_misses_by_q, dict) else {}
    )
    gqa_last_miss = owned_attn.get("gqa_sdpa_last_route_miss") or {}
    stats.paged_gqa_sdpa_last_route_miss = (
        dict(gqa_last_miss) if isinstance(gqa_last_miss, dict) else {}
    )
    stats.attention_dense_fallback_calls = int(
        owned_attn.get("dense_fallback_calls") or 0
    )
    stats.prefill_dense_fallback_calls = int(
        owned_attn.get("prefill_dense_fallback_calls") or 0
    )
    stats.decode_dense_fallback_calls = int(
        owned_attn.get("decode_dense_fallback_calls") or 0
    )
    stats.ar_dense_fallback_calls = int(owned_attn.get("ar_dense_fallback_calls") or 0)
    stats.postcommit_dense_fallback_calls = int(
        owned_attn.get("postcommit_dense_fallback_calls") or 0
    )
    bailouts = owned_attn.get("paged_attention_bailouts_by_phase_reason") or {}
    stats.paged_attention_bailouts_by_phase_reason = (
        dict(bailouts) if isinstance(bailouts, dict) else {}
    )
    stats.paged_attention_large_q_path = str(
        owned_attn.get("paged_attention_large_q_path") or ""
    )
    stats.prefill_route = (
        _sustained_prefill_layout()
        if _contiguous_prefill_cache_layout_enabled()
        else stats.paged_attention_large_q_path
    )
    stats.large_q_split_sdpa_fallback_calls = int(
        owned_attn.get("large_q_split_sdpa_fallback_calls") or 0
    )
    large_q_by_phase = (
        owned_attn.get("large_q_split_sdpa_fallback_calls_by_phase") or {}
    )
    stats.large_q_split_sdpa_fallback_calls_by_phase = (
        dict(large_q_by_phase) if isinstance(large_q_by_phase, dict) else {}
    )
    stats.prefill_large_q_split_sdpa_fallback_calls = int(
        owned_attn.get("prefill_large_q_split_sdpa_fallback_calls") or 0
    )
    stats.decode_large_q_split_sdpa_fallback_calls = int(
        owned_attn.get("decode_large_q_split_sdpa_fallback_calls") or 0
    )
    stats.partitioned_paged_calls = int(owned_attn.get("partitioned_paged_calls") or 0)
    partitioned_by_phase = owned_attn.get("partitioned_paged_calls_by_phase") or {}
    stats.partitioned_paged_calls_by_phase = (
        dict(partitioned_by_phase) if isinstance(partitioned_by_phase, dict) else {}
    )
    stats.prefill_partitioned_paged_calls = int(
        owned_attn.get("prefill_partitioned_paged_calls") or 0
    )
    stats.decode_partitioned_paged_calls = int(
        owned_attn.get("decode_partitioned_paged_calls") or 0
    )


def _sustained_prefill_enabled() -> bool:
    return _env_truthy("MTPLX_SUSTAINED_PREFILL")


def _final_logits_prefill_enabled() -> bool:
    return _sustained_prefill_enabled() or _env_falsey(
        "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"
    )


def _prefill_chunk_cache_cleanup_enabled() -> bool:
    return _env_truthy("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP")


def _prefill_chunk_cache_cleanup_every() -> int:
    raw = os.environ.get("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY")
    if raw is None or not str(raw).strip():
        return 1
    raw_text = str(raw).strip().lower()
    if raw_text == "auto":
        # Dense layout: cleanup every 4 chunks. The per-chunk
        # synchronize+clear_cache was costing 5-21% prefill throughput with
        # zero memory benefit (A/B 2026-07-05, fresh daemon per arm, max
        # fans, 2048-token chunks: 16k 565->682 pp, 32k 521->547, 64k
        # 423->464, 128k 294->315 tok/s; peak memory byte-identical at
        # 20.7/25.4/30.6/41.5 GB). The repage layout keeps its measured
        # every-2 cadence: its chunk intermediates feed the repage copy and
        # accumulate differently.
        return 2 if _sustained_prefill_layout() == "contiguous_then_repage" else 4
    try:
        return max(1, int(raw_text))
    except ValueError:
        return 1


def _prefill_chunk_cache_cleanup(rt: MTPLXRuntime) -> float:
    if not _prefill_chunk_cache_cleanup_enabled():
        return 0.0
    every = _prefill_chunk_cache_cleanup_every()
    pending = (
        int(rt.diagnostic_counters.get("_prefill_chunks_since_cache_cleanup", 0)) + 1
    )
    rt.diagnostic_counters["_prefill_chunks_since_cache_cleanup"] = pending
    if pending < every:
        return 0.0
    rt.diagnostic_counters["_prefill_chunks_since_cache_cleanup"] = 0
    started = time.perf_counter()
    try:
        mx.synchronize()
    except RuntimeError:
        pass
    mx.clear_cache()
    _runtime_count(rt, "prefill_chunk_cache_cleanup_events")
    return time.perf_counter() - started


def _prefill_stock_cache_only_enabled() -> bool:
    return _env_truthy("MTPLX_PREFILL_STOCK_CACHE_ONLY") and _env_truthy(
        "MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY"
    )


def _unsafe_long_context_prefill_guard_tokens() -> int:
    raw = os.environ.get("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS")
    if raw is None or not str(raw).strip():
        return 16384
    try:
        return max(0, int(str(raw).strip()))
    except ValueError:
        return 16384


def _unsafe_long_context_prefill_allowed() -> bool:
    return _env_truthy("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL")


def _assert_safe_long_context_prefill(prompt_tokens: int) -> None:
    if _sustained_prefill_enabled() or _unsafe_long_context_prefill_allowed():
        return
    threshold = _unsafe_long_context_prefill_guard_tokens()
    if threshold <= 0 or int(prompt_tokens) < threshold:
        return
    raise RuntimeError(
        "Blocked unsafe long-context MTP prefill path: "
        f"{int(prompt_tokens)} prompt tokens would use the non-Sustained full "
        "hidden/logits prefill route. Start MTPLX with `--profile sustained` "
        "or run `mtplx config set profile sustained`. To intentionally run "
        "this diagnostic path, set MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL=1."
    )


def _prefill_omlx_external_enabled() -> bool:
    return _env_truthy("MTPLX_PREFILL_OMLX_EXTERNAL")


def _prefill_external_cache_only_enabled() -> bool:
    return _prefill_omlx_external_enabled() or _prefill_stock_cache_only_enabled()


def _prefill_external_emit_logits_enabled() -> bool:
    return not _env_falsey("MTPLX_PREFILL_EXTERNAL_EMIT_LOGITS")


def _batched_token_array(token_ids: Any) -> mx.array:
    if hasattr(token_ids, "shape") and hasattr(token_ids, "dtype"):
        if len(token_ids.shape) == 1:
            return token_ids[None]
        return token_ids
    return mx.array([token_ids])


def _prefill_cache_only_forward(
    rt: MTPLXRuntime,
    token_ids: Any,
    cache: Any,
    input_embeddings: Any | None = None,
) -> Any:
    token_array = _batched_token_array(token_ids)
    if not _prefill_external_cache_only_enabled():
        return rt.forward_ar(
            token_array,
            cache=cache,
            return_hidden=False,
            emit_logits=not _final_logits_prefill_enabled(),
            input_embeddings=input_embeddings,
        )
    _runtime_count(rt, "prefill_external_cache_only_calls")
    if _prefill_stock_cache_only_enabled():
        _runtime_count(rt, "prefill_stock_cache_only_calls")
    if _prefill_omlx_external_enabled():
        _runtime_count(rt, "prefill_omlx_external_calls")
    if not _prefill_external_emit_logits_enabled():
        return rt.forward_ar(
            token_array,
            cache=cache,
            return_hidden=False,
            emit_logits=False,
            input_embeddings=input_embeddings,
        )
    if input_embeddings is not None:
        unused_logits = rt.model(
            token_array, cache=cache, input_embeddings=input_embeddings
        )
    else:
        unused_logits = rt.model(token_array, cache=cache)
    del unused_logits
    return None


def _forward_ar_optional_hidden(
    rt: MTPLXRuntime,
    token_array: Any,
    *,
    cache: Any,
    hidden_variant: str | None,
    emit_logits: bool = True,
    logits_keep: int | None = None,
    input_embeddings: Any | None = None,
) -> tuple[Any, Any]:
    """`forward_ar` as (logits, hidden), with hidden None on target-only runtimes.

    Only request hidden states from a runtime that can produce them. Target-only
    AR runtimes (laguna_ar) have no draft head: their forward_ar returns logits
    alone, so an ungated ``return_hidden=True`` unpacks a lone logits array as
    ``(logits, hidden)`` and raises "not enough values to unpack (expected 2,
    got 1)" — the live serving crash in the warm session-restore suffix prefill.
    `hidden_variant` travels only on the hidden branch for the same reason: the
    generic runtime forwards it to the model as a kwarg a stock target does not
    accept. This mirrors the cold prefill path and generate_ar, which both gate
    return_hidden on rt.mtp_enabled. Callers must treat hidden as optional.
    """

    if not rt.mtp_enabled:
        logits = rt.forward_ar(
            token_array,
            cache=cache,
            return_hidden=False,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
            input_embeddings=input_embeddings,
        )
        return logits, None
    return rt.forward_ar(
        token_array,
        cache=cache,
        return_hidden=True,
        hidden_variant=hidden_variant,
        emit_logits=emit_logits,
        logits_keep=logits_keep,
        input_embeddings=input_embeddings,
    )


def _prefill_chunk_size() -> int:
    override = _PREFILL_CHUNK_SIZE_OVERRIDE.get()
    if override is not None:
        return max(1, int(override))
    raw = (os.environ.get("MTPLX_PREFILL_CHUNK_SIZE") or "2048").strip().lower()
    if raw == "auto":
        layout = _sustained_prefill_layout()
        if layout == "contiguous_dense_decode":
            return max(1, _env_int("MTPLX_PREFILL_CHUNK_SIZE_DENSE", 2048))
        return max(1, _env_int("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", 2048))
    try:
        return max(1, int(raw))
    except ValueError:
        return 2048


@contextmanager
def prefill_chunk_size_override(chunk_size: int | None):
    """Apply a request-local prefill chunk override.

    The legacy env knob remains supported for profiles and CLI diagnostics, but
    the native app needs a live next-request setting. A ContextVar keeps that
    override off process-global environment state.
    """

    token = _PREFILL_CHUNK_SIZE_OVERRIDE.set(
        None if chunk_size is None else max(1, int(chunk_size))
    )
    try:
        yield
    finally:
        _PREFILL_CHUNK_SIZE_OVERRIDE.reset(token)


def _iter_prefill_chunks(token_ids: list[int]) -> list[list[int]]:
    if not token_ids:
        return []
    if not _sustained_prefill_enabled():
        return [token_ids]
    chunk_size = _prefill_chunk_size()
    return [
        token_ids[start : start + chunk_size]
        for start in range(0, len(token_ids), chunk_size)
    ]


def _split_spans_at(
    spans: list[tuple[int, int]], edges: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Split contiguous spans so every in-range edge is an exact span end.

    Used to align a prefill chunk boundary with a stable prompt-prefix
    position (the pre-injection boundary of the transient trailing tool
    hint), so the existing gdn-boundary capture records recurrent state
    exactly there. Chunked prefill is mathematically split-invariant; only
    the chunk layout changes. Edges outside (0, total) or already on a
    span end are no-ops.
    """
    if not spans or not edges:
        return spans
    out = spans
    for edge in sorted(set(int(e) for e in edges)):
        split: list[tuple[int, int]] = []
        for start, end in out:
            if start < edge < end:
                split.append((start, edge))
                split.append((edge, end))
            else:
                split.append((start, end))
        out = split
    return out


def _iter_prefill_chunk_spans(
    token_count: int,
    *,
    mandatory_edges: tuple[int, ...] = (),
    chunk_size: int | None = None,
) -> list[tuple[int, int]]:
    if token_count <= 0:
        return []
    if chunk_size is None and not _sustained_prefill_enabled():
        return _split_spans_at([(0, token_count)], mandatory_edges)
    resolved_chunk_size = (
        _prefill_chunk_size() if chunk_size is None else max(1, int(chunk_size))
    )
    return _split_spans_at(
        [
            (start, min(token_count, start + resolved_chunk_size))
            for start in range(0, token_count, resolved_chunk_size)
        ],
        mandatory_edges,
    )


def _sustained_prefill_layout() -> str:
    layout = (
        os.environ.get("MTPLX_SUSTAINED_PREFILL_LAYOUT", "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if layout != "auto":
        return layout
    # Canonicalize through the one parser: a raw membership test here missed
    # documented spellings ("8", "8bit", "uint8") that the rest of the stack
    # honours as q8, and silently picked the dense-decode layout for a
    # quantized cache.
    from .kv_quant import paged_kv_quant_mode_from_env

    if paged_kv_quant_mode_from_env() != "off":
        return "contiguous_then_repage"
    context_tokens = _env_int("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", 0)
    dense_max = _env_int("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", 131072)
    if context_tokens > 0 and context_tokens <= dense_max:
        return "contiguous_dense_decode"
    return "contiguous_then_repage"


def _defer_verify_hidden_eval_enabled() -> bool:
    raw = (os.environ.get("MTPLX_DEFER_VERIFY_HIDDEN_EVAL") or "").strip().lower()
    if raw == "auto":
        context_tokens = _env_int("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", 0)
        dense_max = _env_int("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", 131072)
        return context_tokens > 0 and context_tokens <= dense_max
    return _env_truthy("MTPLX_DEFER_VERIFY_HIDDEN_EVAL")


def _verify_hidden_mode() -> str:
    raw = (
        (os.environ.get("MTPLX_VERIFY_HIDDEN_MODE") or "default")
        .strip()
        .lower()
        .replace("-", "_")
    )
    return raw or "default"


def _clear_cache_every() -> int:
    raw = (os.environ.get("MTPLX_CLEAR_CACHE_EVERY") or "auto").strip().lower()
    if raw == "auto":
        context_tokens = _env_int("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", 0)
        # Lowered default 98304 -> 16384 so clear_cache fires for the typical
        # opencode subagent context regime (16-40K) where wired-memory pressure
        # has been observed in practice. The previous threshold only kicked in
        # past 96K, well above the crash zone.
        threshold = _env_int("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", 16384)
        if context_tokens >= threshold and _contiguous_dense_decode_prefill_enabled():
            # Default 16 tokens was per-step aggressive (sync barrier every
            # tick). 256 amortized it; 1024 (2026-07-16) removes the remaining
            # -3.8% decode tax on 512-token generations at 33k ctx while
            # marathon responses still get periodic allocator bounding.
            return max(0, _env_int("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", 1024))
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _contiguous_then_repage_prefill_enabled() -> bool:
    return _sustained_prefill_layout() == "contiguous_then_repage"


def _contiguous_dense_decode_prefill_enabled() -> bool:
    return _sustained_prefill_layout() == "contiguous_dense_decode"


def _contiguous_prefill_cache_layout_enabled() -> bool:
    return (
        _contiguous_then_repage_prefill_enabled()
        or _contiguous_dense_decode_prefill_enabled()
    )


@contextmanager
def _target_prefill_cache_layout_scope():
    if not _contiguous_prefill_cache_layout_enabled():
        yield
        return
    keys = (
        "MTPLX_VLLM_METAL_PAGED_ATTN",
        "MTPLX_OWNED_ATTN_KV",
        "MTPLX_BLOCK_OWNED_ATTN_KV",
    )
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] = "0"
    os.environ["MTPLX_OWNED_ATTN_KV"] = "0"
    os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] = "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_target_prefill_cache(rt: MTPLXRuntime):
    with _target_prefill_cache_layout_scope():
        return rt.make_cache()


def _maybe_repage_target_prefill_cache(rt: MTPLXRuntime, cache: Any) -> float:
    if not _contiguous_then_repage_prefill_enabled():
        return 0.0

    started = time.perf_counter()
    if not rt.repage_target_prefill_cache(cache):
        return 0.0
    _eval_cache_roots(cache)
    return time.perf_counter() - started


def _session_restore_cache_factory(rt: MTPLXRuntime) -> Callable[[], Any] | None:
    if not _contiguous_prefill_cache_layout_enabled():
        return None
    return lambda: _make_target_prefill_cache(rt)


def _session_live_frontier_reference_restore_enabled() -> bool:
    name = "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE"
    if name not in os.environ:
        name = "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER"
    if name not in os.environ:
        return False
    return _env_truthy(name)


def _eval_cache_roots(cache: Any) -> None:
    arrays = _tree_mx_arrays(cache)
    if not arrays:
        return
    deduped: list[mx.array] = []
    seen: set[int] = set()
    for array in arrays:
        ident = id(array)
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(array)
    if deduped:
        _eval(*deduped, _caller_depth=2)


def _eval_verify_outputs(
    verify_logits: mx.array, verify_hidden: mx.array, captures: Any | None = None
) -> dict[str, float]:
    # Keep capture tensors lazy; commit_captured_prefix materializes only the selected prefix slice.
    timings = {
        "verify_logits_eval_time_s": 0.0,
        "verify_hidden_eval_time_s": 0.0,
        "verify_joint_eval_time_s": 0.0,
    }
    if _env_truthy("MTPLX_LAZY_VERIFY_LOGITS"):
        started = time.perf_counter()
        _eval(verify_hidden, _caller_depth=2)
        timings["verify_hidden_eval_time_s"] += time.perf_counter() - started
        return timings
    if _env_truthy("MTPLX_SPLIT_VERIFY_EVAL"):
        started = time.perf_counter()
        _eval(verify_logits, _caller_depth=2)
        timings["verify_logits_eval_time_s"] += time.perf_counter() - started
        started = time.perf_counter()
        _eval(verify_hidden, _caller_depth=2)
        timings["verify_hidden_eval_time_s"] += time.perf_counter() - started
        return timings
    started = time.perf_counter()
    _eval(verify_logits, verify_hidden, _caller_depth=2)
    timings["verify_joint_eval_time_s"] += time.perf_counter() - started
    return timings


def _tree_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """Best-effort recursive byte count for MLX/NumPy array trees."""
    if value is None:
        return 0
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if isinstance(value, mx.array):
        return int(value.nbytes)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return 0
    if isinstance(value, dict):
        return sum(_tree_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_tree_nbytes(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tree_nbytes(getattr(value, item.name), seen) for item in fields(value)
        )
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return sum(_tree_nbytes(item, seen) for item in attrs.values())
    return 0


def _tree_mx_arrays(value: Any, seen: set[int] | None = None) -> list[mx.array]:
    if value is None:
        return []
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, np.ndarray):
        return []
    if isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return []
    if isinstance(value, dict):
        arrays: list[mx.array] = []
        for item in value.values():
            arrays.extend(_tree_mx_arrays(item, seen))
        return arrays
    if isinstance(value, (list, tuple, set)):
        arrays = []
        for item in value:
            arrays.extend(_tree_mx_arrays(item, seen))
        return arrays
    if is_dataclass(value) and not isinstance(value, type):
        arrays = []
        for item in fields(value):
            arrays.extend(_tree_mx_arrays(getattr(value, item.name), seen))
        return arrays
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        arrays = []
        for item in attrs.values():
            arrays.extend(_tree_mx_arrays(item, seen))
        return arrays
    return []


def _mlx_memory_stats() -> dict[str, int]:
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }


# Live decode telemetry slot: the server installs a per-request publisher
# (flight recorder) before dispatching a generation on the model-owner thread
# and clears it after. _DecodeTrace captures the slot at construction and
# publishes by-depth acceptance totals at most once per second, riding the
# interval machinery it already has — accepted-by-depth is otherwise invisible
# until the final receipt. Single writer (owner thread), tear-tolerant readers,
# no lock: the progress_heartbeat precedent.
_LIVE_DECODE_SINK: Callable[[dict[str, Any]], None] | None = None


def set_live_decode_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    global _LIVE_DECODE_SINK
    _LIVE_DECODE_SINK = sink


class _DecodeTrace:
    def __init__(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        speculative_depth: int,
        sampler: SamplerConfig,
        verify_strategy: str,
        verify_core: str,
        mtp_history_policy: str,
        mtp_cache_policy: str,
        trace_label: str | None,
        trace_metadata: dict[str, Any] | None,
    ) -> None:
        trace_path = os.environ.get("MTPLX_DECODE_TRACE_JSONL")
        self.enabled = bool(trace_path)
        self.path = Path(trace_path).expanduser() if trace_path else None
        self.interval_s = max(
            0.1,
            float(os.environ.get("MTPLX_DECODE_TRACE_INTERVAL_S") or 1.0),
        )
        self.run_id = f"{int(time.time() * 1000)}-{os.getpid()}-{id(self):x}"
        self.label = trace_label or os.environ.get("MTPLX_DECODE_TRACE_LABEL") or None
        self.metadata = dict(trace_metadata or {})
        self.prompt_tokens = int(prompt_tokens)
        self.max_tokens = int(max_tokens)
        self.speculative_depth = int(speculative_depth)
        self.sampler = sampler
        self.verify_strategy = verify_strategy
        self.verify_core = verify_core
        self.mtp_history_policy = mtp_history_policy
        self.mtp_cache_policy = mtp_cache_policy
        self.started_s = time.perf_counter()
        self.last_emit_s = self.started_s
        self.live_sink = _LIVE_DECODE_SINK
        self._last_live_s = 0.0
        self.bucket_index = 0
        self.last_totals: dict[str, Any] = {
            "generated_tokens": 0,
            "accepted_drafts": 0,
            "rejected_drafts": 0,
            "drafted_tokens": 0,
            "verify_calls": 0,
            "correction_tokens": 0,
            "bonus_tokens": 0,
            "verify_time_s": 0.0,
            "verify_forward_time_s": 0.0,
            "verify_eval_time_s": 0.0,
            "verify_logits_eval_time_s": 0.0,
            "verify_hidden_eval_time_s": 0.0,
            "verify_joint_eval_time_s": 0.0,
            "verify_target_distribution_time_s": 0.0,
            "target_distribution_materialized_rows": 0,
            "target_distribution_materialized_windows": 0,
            "lazy_bonus_verify_calls": 0,
            "lazy_bonus_commit_time_s": 0.0,
            "verify_eval_unattributed_time_s": 0.0,
            "draft_time_s": 0.0,
            "accept_time_s": 0.0,
            "repair_time_s": 0.0,
            "commit_time_s": 0.0,
            "capture_commit_time_s": 0.0,
            "snapshot_time_s": 0.0,
            "bonus_time_s": 0.0,
            "verify_output_nbytes": 0,
            "draft_output_nbytes": 0,
            "mtp_history_append_nbytes": 0,
            "clear_cache_events": 0,
            "clear_cache_time_s": 0.0,
            "trunk_cache_materialize_events": 0,
            "trunk_cache_materialize_time_s": 0.0,
            "dirty_detach_events": 0,
            "dirty_detach_time_s": 0.0,
            "dirty_detach_arrays": 0,
            "dirty_detach_bytes": 0,
            "live_output_detach_events": 0,
            "live_output_detach_time_s": 0.0,
            "live_output_detach_arrays": 0,
            "live_output_detach_bytes": 0,
            "state_rebase_events": 0,
            "state_rebase_time_s": 0.0,
            "state_root_eval_events": 0,
            "state_root_eval_time_s": 0.0,
            "state_root_eval_arrays": 0,
            "trace_accounting_time_s": 0.0,
            "accepted_by_depth": [0 for _ in range(speculative_depth)],
            "drafted_by_depth": [0 for _ in range(speculative_depth)],
            "accept_probability_sum_by_depth": [0.0 for _ in range(speculative_depth)],
            "draft_confidence_width_stops": 0,
            "draft_confidence_sum_by_depth": [0.0 for _ in range(speculative_depth)],
            "draft_confidence_count_by_depth": [0 for _ in range(speculative_depth)],
            "draft_confidence_accepted_sum_by_depth": [
                0.0 for _ in range(speculative_depth)
            ],
            "draft_confidence_accepted_count_by_depth": [
                0 for _ in range(speculative_depth)
            ],
            "draft_confidence_rejected_sum_by_depth": [
                0.0 for _ in range(speculative_depth)
            ],
            "draft_confidence_rejected_count_by_depth": [
                0 for _ in range(speculative_depth)
            ],
            "draft_confidence_accepted_hist_flat": [
                0 for _ in range(speculative_depth * 10)
            ],
            "draft_confidence_rejected_hist_flat": [
                0 for _ in range(speculative_depth * 10)
            ],
        }
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _delta(self, totals: dict[str, Any], key: str) -> Any:
        # Lanes maintain different counter sets (AR omits MTP-only keys);
        # a counter absent on either side is a zero delta, not an error.
        value = totals.get(key)
        previous = self.last_totals.get(key)
        if value is None:
            value = previous if previous is not None else 0
        if previous is None:
            previous = [0.0] * len(value) if isinstance(value, list) else 0
        if isinstance(value, list):
            return [(float(item) - float(prev)) for item, prev in zip(value, previous)]
        return value - previous

    def maybe_emit(
        self,
        *,
        force: bool,
        final: bool,
        totals: dict[str, Any],
        cache: Any,
        mtp_cache: Any,
        mtp_history_materialize_every: int,
        mtp_history_materialize_events: int,
    ) -> None:
        sink = self.live_sink
        if sink is not None:
            now_live = time.perf_counter()
            if force or final or now_live - self._last_live_s >= 1.0:
                self._last_live_s = now_live
                try:
                    sink(
                        {
                            "generated_tokens": totals.get("generated_tokens"),
                            "accepted_by_depth": list(
                                totals.get("accepted_by_depth") or []
                            ),
                            "drafted_by_depth": list(
                                totals.get("drafted_by_depth") or []
                            ),
                            "verify_calls": totals.get("verify_calls"),
                            "verify_time_s": totals.get("verify_time_s"),
                            "draft_time_s": totals.get("draft_time_s"),
                        }
                    )
                except Exception:
                    # A broken sink must never touch decode again this request.
                    self.live_sink = None
        if not self.enabled or self.path is None:
            return
        now = time.perf_counter()
        if not force and now - self.last_emit_s < self.interval_s:
            return
        elapsed_s = max(0.0, now - self.last_emit_s)
        generated_delta = int(self._delta(totals, "generated_tokens"))
        drafted_by_depth_delta = [
            int(item) for item in self._delta(totals, "drafted_by_depth")
        ]
        accepted_by_depth_delta = [
            int(item) for item in self._delta(totals, "accepted_by_depth")
        ]
        accept_probability_sum_delta = [
            float(item)
            for item in self._delta(totals, "accept_probability_sum_by_depth")
        ]
        acceptance_rate_by_depth_delta = [
            (float(accepted) / int(drafted) if drafted else None)
            for accepted, drafted in zip(
                accepted_by_depth_delta, drafted_by_depth_delta
            )
        ]
        mean_accept_probability_by_depth_delta = [
            (float(total) / int(drafted) if drafted else None)
            for total, drafted in zip(
                accept_probability_sum_delta, drafted_by_depth_delta
            )
        ]

        def _conf_pair(kind: str) -> tuple[list[float], list[int], list[float | None]]:
            # A lane that never carried these keys (AR after last_totals was
            # re-snapshotted from its own totals) gets scalar-zero deltas
            # from _delta; the tolerant shape for a by-depth counter is [].
            raw_sums = self._delta(totals, f"draft_confidence_{kind}sum_by_depth")
            raw_counts = self._delta(
                totals, f"draft_confidence_{kind}count_by_depth"
            )
            sums = [
                float(item)
                for item in (raw_sums if isinstance(raw_sums, list) else [])
            ]
            counts = [
                int(item)
                for item in (raw_counts if isinstance(raw_counts, list) else [])
            ]
            means = [
                (s / c if c else None) for s, c in zip(sums, counts)
            ]
            return sums, counts, means

        (
            _conf_sum_unused,
            draft_confidence_count_delta,
            draft_confidence_mean_delta,
        ) = _conf_pair("")
        (
            _conf_accepted_sum_unused,
            draft_confidence_accepted_count_delta,
            draft_confidence_accepted_mean_delta,
        ) = _conf_pair("accepted_")
        (
            _conf_rejected_sum_unused,
            draft_confidence_rejected_count_delta,
            draft_confidence_rejected_mean_delta,
        ) = _conf_pair("rejected_")
        draft_confidence_width_stops_delta = int(
            self._delta(totals, "draft_confidence_width_stops")
        )

        def _hist_delta(key: str) -> list[int]:
            raw = self._delta(totals, key)
            return [int(item) for item in (raw if isinstance(raw, list) else [])]

        draft_confidence_accepted_hist_delta = _hist_delta(
            "draft_confidence_accepted_hist_flat"
        )
        draft_confidence_rejected_hist_delta = _hist_delta(
            "draft_confidence_rejected_hist_flat"
        )
        verify_calls_delta = int(self._delta(totals, "verify_calls"))
        accepted_drafts_delta = int(self._delta(totals, "accepted_drafts"))
        drafted_tokens_delta = int(self._delta(totals, "drafted_tokens"))
        verify_time_delta = float(self._delta(totals, "verify_time_s"))
        verify_forward_time_delta = float(self._delta(totals, "verify_forward_time_s"))
        verify_eval_time_delta = float(self._delta(totals, "verify_eval_time_s"))
        verify_logits_eval_time_delta = float(
            self._delta(totals, "verify_logits_eval_time_s")
        )
        verify_hidden_eval_time_delta = float(
            self._delta(totals, "verify_hidden_eval_time_s")
        )
        verify_joint_eval_time_delta = float(
            self._delta(totals, "verify_joint_eval_time_s")
        )
        verify_target_distribution_time_delta = float(
            self._delta(totals, "verify_target_distribution_time_s")
        )
        target_distribution_rows_delta = int(
            self._delta(totals, "target_distribution_materialized_rows")
        )
        target_distribution_windows_delta = int(
            self._delta(totals, "target_distribution_materialized_windows")
        )
        lazy_bonus_verify_calls_delta = int(
            self._delta(totals, "lazy_bonus_verify_calls")
        )
        lazy_bonus_commit_time_delta = float(
            self._delta(totals, "lazy_bonus_commit_time_s")
        )
        verify_eval_unattributed_time_delta = float(
            self._delta(totals, "verify_eval_unattributed_time_s")
        )
        draft_time_delta = float(self._delta(totals, "draft_time_s"))
        clear_cache_events_delta = int(self._delta(totals, "clear_cache_events"))
        clear_cache_time_delta = float(self._delta(totals, "clear_cache_time_s"))
        trunk_cache_materialize_events_delta = int(
            self._delta(totals, "trunk_cache_materialize_events")
        )
        trunk_cache_materialize_time_delta = float(
            self._delta(totals, "trunk_cache_materialize_time_s")
        )
        dirty_detach_events_delta = int(self._delta(totals, "dirty_detach_events"))
        dirty_detach_time_delta = float(self._delta(totals, "dirty_detach_time_s"))
        dirty_detach_arrays_delta = int(self._delta(totals, "dirty_detach_arrays"))
        dirty_detach_bytes_delta = int(self._delta(totals, "dirty_detach_bytes"))
        live_output_detach_events_delta = int(
            self._delta(totals, "live_output_detach_events")
        )
        live_output_detach_time_delta = float(
            self._delta(totals, "live_output_detach_time_s")
        )
        live_output_detach_arrays_delta = int(
            self._delta(totals, "live_output_detach_arrays")
        )
        live_output_detach_bytes_delta = int(
            self._delta(totals, "live_output_detach_bytes")
        )
        state_rebase_events_delta = int(self._delta(totals, "state_rebase_events"))
        state_rebase_time_delta = float(self._delta(totals, "state_rebase_time_s"))
        state_root_eval_events_delta = int(
            self._delta(totals, "state_root_eval_events")
        )
        state_root_eval_time_delta = float(
            self._delta(totals, "state_root_eval_time_s")
        )
        state_root_eval_arrays_delta = int(
            self._delta(totals, "state_root_eval_arrays")
        )
        trace_accounting_time_delta = float(
            self._delta(totals, "trace_accounting_time_s")
        )
        bytes_delta = {
            "verify_output_nbytes_delta": int(
                self._delta(totals, "verify_output_nbytes")
            ),
            "draft_output_nbytes_delta": int(
                self._delta(totals, "draft_output_nbytes")
            ),
            "mtp_history_append_nbytes_delta": int(
                self._delta(totals, "mtp_history_append_nbytes")
            ),
        }
        materialized_nbytes = sum(bytes_delta.values())
        row = {
            "event": "decode_trace_bucket",
            "run_id": self.run_id,
            "label": self.label,
            "bucket_index": self.bucket_index,
            "final": bool(final),
            "t_start_s": self.last_emit_s - self.started_s,
            "t_end_s": now - self.started_s,
            "elapsed_s": elapsed_s,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "generated_tokens_total": int(totals["generated_tokens"]),
            "generated_tokens_delta": generated_delta,
            "tok_s_delta": generated_delta / elapsed_s if elapsed_s > 0 else None,
            "context_len": self.prompt_tokens + int(totals["generated_tokens"]),
            "speculative_depth": self.speculative_depth,
            "verify_calls_total": int(totals["verify_calls"]),
            "verify_calls_delta": verify_calls_delta,
            "accepted_drafts_total": int(totals["accepted_drafts"]),
            "accepted_drafts_delta": accepted_drafts_delta,
            "drafted_tokens_total": int(totals["drafted_tokens"]),
            "drafted_tokens_delta": drafted_tokens_delta,
            "accepted_per_verify_delta": (
                accepted_drafts_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "draft_acceptance_rate_delta": (
                accepted_drafts_delta / drafted_tokens_delta
                if drafted_tokens_delta
                else None
            ),
            "accepted_by_depth_total": [
                int(item) for item in totals["accepted_by_depth"]
            ],
            "accepted_by_depth_delta": accepted_by_depth_delta,
            "drafted_by_depth_total": [
                int(item) for item in totals["drafted_by_depth"]
            ],
            "drafted_by_depth_delta": drafted_by_depth_delta,
            "acceptance_rate_by_depth_delta": acceptance_rate_by_depth_delta,
            "mean_accept_probability_by_depth_delta": mean_accept_probability_by_depth_delta,
            "draft_confidence_width_stops_delta": draft_confidence_width_stops_delta,
            "draft_confidence_count_by_depth_delta": draft_confidence_count_delta,
            "draft_confidence_mean_by_depth_delta": draft_confidence_mean_delta,
            "draft_confidence_accepted_count_by_depth_delta": (
                draft_confidence_accepted_count_delta
            ),
            "draft_confidence_accepted_mean_by_depth_delta": (
                draft_confidence_accepted_mean_delta
            ),
            "draft_confidence_rejected_count_by_depth_delta": (
                draft_confidence_rejected_count_delta
            ),
            "draft_confidence_rejected_mean_by_depth_delta": (
                draft_confidence_rejected_mean_delta
            ),
            "draft_confidence_accepted_hist_flat_delta": (
                draft_confidence_accepted_hist_delta
            ),
            "draft_confidence_rejected_hist_flat_delta": (
                draft_confidence_rejected_hist_delta
            ),
            "rejected_drafts_delta": int(self._delta(totals, "rejected_drafts")),
            "correction_tokens_delta": int(self._delta(totals, "correction_tokens")),
            "bonus_tokens_delta": int(self._delta(totals, "bonus_tokens")),
            "verify_time_s_delta": verify_time_delta,
            "verify_forward_time_s_delta": verify_forward_time_delta,
            "verify_eval_time_s_delta": verify_eval_time_delta,
            "verify_logits_eval_time_s_delta": verify_logits_eval_time_delta,
            "verify_hidden_eval_time_s_delta": verify_hidden_eval_time_delta,
            "verify_joint_eval_time_s_delta": verify_joint_eval_time_delta,
            "verify_target_distribution_time_s_delta": verify_target_distribution_time_delta,
            "target_distribution_materialized_rows_delta": target_distribution_rows_delta,
            "target_distribution_materialized_windows_delta": target_distribution_windows_delta,
            "target_distribution_rows_per_window_delta": (
                target_distribution_rows_delta / target_distribution_windows_delta
                if target_distribution_windows_delta
                else None
            ),
            "verify_target_distribution_ms_per_row_delta": (
                1000.0
                * verify_target_distribution_time_delta
                / target_distribution_rows_delta
                if target_distribution_rows_delta
                else None
            ),
            "lazy_bonus_verify_calls_delta": lazy_bonus_verify_calls_delta,
            "lazy_bonus_commit_time_s_delta": lazy_bonus_commit_time_delta,
            "lazy_bonus_commit_ms_per_call_delta": (
                1000.0 * lazy_bonus_commit_time_delta / lazy_bonus_verify_calls_delta
                if lazy_bonus_verify_calls_delta
                else None
            ),
            "verify_eval_unattributed_time_s_delta": verify_eval_unattributed_time_delta,
            "draft_time_s_delta": draft_time_delta,
            "accept_time_s_delta": float(self._delta(totals, "accept_time_s")),
            "repair_time_s_delta": float(self._delta(totals, "repair_time_s")),
            "commit_time_s_delta": float(self._delta(totals, "commit_time_s")),
            "capture_commit_time_s_delta": float(
                self._delta(totals, "capture_commit_time_s")
            ),
            "snapshot_time_s_delta": float(self._delta(totals, "snapshot_time_s")),
            "bonus_time_s_delta": float(self._delta(totals, "bonus_time_s")),
            "verify_ms_per_call_delta": (
                1000.0 * verify_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_forward_ms_per_call_delta": (
                1000.0 * verify_forward_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_eval_ms_per_call_delta": (
                1000.0 * verify_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_logits_eval_ms_per_call_delta": (
                1000.0 * verify_logits_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_hidden_eval_ms_per_call_delta": (
                1000.0 * verify_hidden_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_joint_eval_ms_per_call_delta": (
                1000.0 * verify_joint_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_target_distribution_ms_per_call_delta": (
                1000.0 * verify_target_distribution_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_eval_unattributed_ms_per_call_delta": (
                1000.0 * verify_eval_unattributed_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "draft_ms_per_token_delta": (
                1000.0 * draft_time_delta / drafted_tokens_delta
                if drafted_tokens_delta
                else None
            ),
            **bytes_delta,
            "estimated_materialized_nbytes_delta": materialized_nbytes,
            "estimated_materialized_gib_s": (
                (materialized_nbytes / (1024**3)) / elapsed_s if elapsed_s > 0 else None
            ),
            "cache_state_nbytes": _tree_nbytes(cache),
            "mtp_cache_state_nbytes": _tree_nbytes(mtp_cache),
            "mlx_memory": _mlx_memory_stats(),
            "lazy_verify_logits": _env_truthy("MTPLX_LAZY_VERIFY_LOGITS"),
            "defer_verify_hidden_eval": _defer_verify_hidden_eval_enabled(),
            "verify_hidden_mode": _verify_hidden_mode(),
            "split_verify_eval": _env_truthy("MTPLX_SPLIT_VERIFY_EVAL"),
            "lazy_mtp_history_append": _env_truthy("MTPLX_LAZY_MTP_HISTORY_APPEND"),
            "batch_target_arrays": _batch_target_arrays_enabled(),
            "drop_events": _env_truthy("MTPLX_DROP_EVENTS"),
            # Trio ports (#313/#315/#318): receipts prove which lane ran —
            # the #314 dead-switch antidote.
            "greedy_draft_chain": _env_enabled_default_on("MTPLX_GREEDY_DRAFT_CHAIN"),
            "batched_greedy_accept": _env_enabled_default_on("MTPLX_BATCHED_GREEDY_ACCEPT"),
            # Env resolution above; the per-request truth is the fence stamp —
            # a >fence prompt runs all three knobs OFF regardless of env.
            "greedy_trio_max_context": _trio_max_context(),
            "trio_context_ok": _paged_offsets_context_ok(),
            "skip_verify_snapshot": _skip_verify_snapshot(),
            "mtp_history_materialize_every": int(mtp_history_materialize_every),
            "mtp_history_materialize_events": int(mtp_history_materialize_events),
            "clear_cache_every": int(_clear_cache_every()),
            "clear_cache_events_total": int(totals["clear_cache_events"]),
            "clear_cache_events_delta": clear_cache_events_delta,
            "clear_cache_time_s_total": float(totals["clear_cache_time_s"]),
            "clear_cache_time_s_delta": clear_cache_time_delta,
            "trunk_cache_materialize_every": int(
                os.environ.get("MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY") or 0
            ),
            "trunk_cache_materialize_events_total": int(
                totals["trunk_cache_materialize_events"]
            ),
            "trunk_cache_materialize_events_delta": trunk_cache_materialize_events_delta,
            "trunk_cache_materialize_time_s_total": float(
                totals["trunk_cache_materialize_time_s"]
            ),
            "trunk_cache_materialize_time_s_delta": trunk_cache_materialize_time_delta,
            "dirty_detach_components": os.environ.get("MTPLX_DETACH_COMPONENTS"),
            "dirty_detach_mode": os.environ.get("MTPLX_DETACH_MODE"),
            "dirty_detach_gdn_every": int(
                os.environ.get("MTPLX_DETACH_GDN_EVERY") or 0
            ),
            "dirty_detach_conv_every": int(
                os.environ.get("MTPLX_DETACH_CONV_EVERY") or 0
            ),
            "dirty_detach_attn_every": int(
                os.environ.get("MTPLX_DETACH_ATTN_EVERY") or 0
            ),
            "dirty_detach_events_total": int(totals["dirty_detach_events"]),
            "dirty_detach_events_delta": dirty_detach_events_delta,
            "dirty_detach_time_s_total": float(totals["dirty_detach_time_s"]),
            "dirty_detach_time_s_delta": dirty_detach_time_delta,
            "dirty_detach_arrays_total": int(totals["dirty_detach_arrays"]),
            "dirty_detach_arrays_delta": dirty_detach_arrays_delta,
            "dirty_detach_bytes_total": int(totals["dirty_detach_bytes"]),
            "dirty_detach_bytes_delta": dirty_detach_bytes_delta,
            "live_output_detach_enabled": bool(
                os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS")
            ),
            "live_output_detach_mode": os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS_MODE"),
            "live_output_detach_events_total": int(totals["live_output_detach_events"]),
            "live_output_detach_events_delta": live_output_detach_events_delta,
            "live_output_detach_time_s_total": float(
                totals["live_output_detach_time_s"]
            ),
            "live_output_detach_time_s_delta": live_output_detach_time_delta,
            "live_output_detach_arrays_total": int(totals["live_output_detach_arrays"]),
            "live_output_detach_arrays_delta": live_output_detach_arrays_delta,
            "live_output_detach_bytes_total": int(totals["live_output_detach_bytes"]),
            "live_output_detach_bytes_delta": live_output_detach_bytes_delta,
            "state_rebase_every": int(os.environ.get("MTPLX_STATE_REBASE_EVERY") or 0),
            "state_rebase_events_total": int(totals["state_rebase_events"]),
            "state_rebase_events_delta": state_rebase_events_delta,
            "state_rebase_time_s_total": float(totals["state_rebase_time_s"]),
            "state_rebase_time_s_delta": state_rebase_time_delta,
            "state_root_eval_enabled": bool(
                os.environ.get("MTPLX_EVAL_STATE_ROOTS_ON_COMMIT")
            ),
            "state_root_eval_include_mtp": bool(
                os.environ.get("MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP", "1")
                .strip()
                .lower()
                not in {"0", "false", "no", "off"}
            ),
            "state_root_eval_events_total": int(totals["state_root_eval_events"]),
            "state_root_eval_events_delta": state_root_eval_events_delta,
            "state_root_eval_time_s_total": float(totals["state_root_eval_time_s"]),
            "state_root_eval_time_s_delta": state_root_eval_time_delta,
            "state_root_eval_arrays_total": int(totals["state_root_eval_arrays"]),
            "state_root_eval_arrays_delta": state_root_eval_arrays_delta,
            "trace_accounting_time_s_total": float(totals["trace_accounting_time_s"]),
            "trace_accounting_time_s_delta": trace_accounting_time_delta,
            "verify_strategy": self.verify_strategy,
            "verify_core": self.verify_core,
            "mtp_history_policy": self.mtp_history_policy,
            "mtp_cache_policy": self.mtp_cache_policy,
            "sampler": {
                "temperature": float(self.sampler.temperature),
                "top_p": float(self.sampler.top_p),
                "top_k": int(self.sampler.top_k)
                if self.sampler.top_k is not None
                else None,
            },
            "metadata": self.metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.bucket_index += 1
        self.last_emit_s = now
        self.last_totals = {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in totals.items()
        }


_AR_FORWARD_PROFILE: Any = None


def _ar_forward_profiler(step: int) -> Any:
    """Diagnostic lane: MTPLX_AR_PROFILE_TOKENS=N cProfiles decode forwards
    for steps [8, 8+N) and dumps pstats to MTPLX_AR_PROFILE_PATH at the
    last profiled step. Off (None) unless the env is set; throughput
    measured with this enabled is not promotion evidence."""

    global _AR_FORWARD_PROFILE
    raw = os.environ.get("MTPLX_AR_PROFILE_TOKENS")
    if not raw:
        return None
    try:
        budget = int(raw)
    except ValueError:
        return None
    first, last = 8, 8 + budget
    if not first <= step < last:
        return None
