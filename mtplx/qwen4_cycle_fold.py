"""Early submission of the exact Qwen4 depth-one MTP draft."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import mlx.core as mx

from .attention_context import attention_phase
from .models.qwen4_omlx import QSAKVCache


class Qwen4CycleFoldConfigError(RuntimeError):
    """The exact Qwen4 cycle ticket cannot be installed."""


@dataclass(frozen=True)
class Qwen4CycleFoldTicket:
    primary: int
    logits: Any
    hidden: Any
    compiled_aux_prefetch: Any | None


def qwen4_cycle_fold_enabled() -> bool:
    """Resolve the experimental lane once, at runtime construction."""

    import os

    return (os.environ.get("MTPLX_QWEN4_CYCLE_FOLD") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _qsa_state_roots(cache: Any) -> tuple[mx.array, ...]:
    roots: list[mx.array] = []
    for entry in cache:
        for leaf in (
            entry.keys,
            entry.values,
            entry.indexer.keys,
            entry.indexer._pooled_keys,
        ):
            if leaf is not None:
                roots.append(leaf)
    return tuple(roots)


def _issue_qwen4_cycle_fold(
    self,
    *,
    hidden: Any,
    primary: int,
    mtp_cache: Any,
    mtp_hidden_variant: str,
    position_offset: int | None = None,
    compiled_aux_prefetch: Any | None = None,
) -> Qwen4CycleFoldTicket:
    # The accepted-history update, when present, has already rebound this
    # request-owned cache under ar_decode. Building the ordinary draft against
    # those lazy leaves preserves the exact M1 -> M1 dependency chain.
    with attention_phase(None):
        logits, next_hidden = self.draft_mtp(
            hidden,
            mx.array([[int(primary)]]),
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=1,
            position_offset=position_offset,
        )
    mx.async_eval(logits, next_hidden, *_qsa_state_roots(mtp_cache))
    return Qwen4CycleFoldTicket(
        primary=int(primary),
        logits=logits,
        hidden=next_hidden,
        compiled_aux_prefetch=compiled_aux_prefetch,
    )


def install_qwen4_cycle_fold(runtime: Any, *, config: dict[str, Any]) -> dict[str, Any]:
    """Prove the native one-layer MTP QSA topology and bind the direct issuer."""

    from .qwen4_capture import is_exact_qwen4_capture_config

    if not is_exact_qwen4_capture_config(config):
        raise Qwen4CycleFoldConfigError("cycle fold requires exact Qwen4 config")
    if not bool(getattr(runtime, "qwen4_depth1_batched_target_arrays", False)):
        raise Qwen4CycleFoldConfigError(
            "cycle fold requires the exact depth-one runtime to be bound"
        )
    cache = runtime.make_mtp_cache()
    if len(cache) != 1 or type(cache[0]) is not QSAKVCache:
        raise Qwen4CycleFoldConfigError(
            "cycle fold requires exactly one QSA cache entry"
        )
    runtime.qwen4_cycle_fold_issue = MethodType(_issue_qwen4_cycle_fold, runtime)
    return {"installed": True, "ticket_rows": 1, "qsa_layers": len(cache)}


__all__ = [
    "Qwen4CycleFoldConfigError",
    "Qwen4CycleFoldTicket",
    "install_qwen4_cycle_fold",
    "qwen4_cycle_fold_enabled",
]
