"""FR-Spec: frequency-ranked pruned DRAFT LM head (target verifies the full vocab).

Port of the Y-PC 3090 Session-2 lever (qwen38-int8-maxtps, 2026-08-15): drafting
from the top-65,536 frequency-ranked vocab rows cut step time 12.9% there while
the target kept verifying over all 248,320 rows. Exactness: probability-ratio
acceptance with residual correction is valid for ANY proposal distribution q —
pruning the draft support can only move the acceptance rate, never the emitted
distribution. The correction path already treats tokens outside the stored
draft top-k as q=0, which subsumes the pruned rows.

Coverage receipts for the ranked list (Y-PC runs/draft_vocab.json, generic
corpus deliberately NOT workload-fit — see the rig's LOSSES.md S2-L9):
0.99487 at n=65536 build-time out-of-sample, 0.99728 measured on real traces;
acceptance-length ceiling cost <=2.2% over 8 draft positions.

Env contract (all default-off):
- ``MTPLX_FRSPEC_DRAFT=1`` enables the pruned draft head.
- ``MTPLX_FRSPEC_VOCAB=<path>`` JSON carrying ``{"ids": [...]}`` ranked
  most-frequent-first (the Y-PC artifact loads unchanged — same tokenizer).
- ``MTPLX_FRSPEC_N`` optional cap; takes the first N ids of the ranking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def frspec_enabled() -> bool:
    return (os.environ.get("MTPLX_FRSPEC_DRAFT", "").strip().lower()
            in {"1", "true", "yes", "on"})


def _vocab_path() -> Path | None:
    raw = (os.environ.get("MTPLX_FRSPEC_VOCAB") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


def load_frspec_ids() -> list[int] | None:
    path = _vocab_path()
    if path is None:
        logger.warning("[frspec] MTPLX_FRSPEC_DRAFT set but MTPLX_FRSPEC_VOCAB missing/not found")
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[frspec] failed to read %s: %s", path, exc)
        return None
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not ids:
        logger.warning("[frspec] %s carries no ids list", path)
        return None
    raw_n = (os.environ.get("MTPLX_FRSPEC_N") or "").strip()
    if raw_n:
        try:
            n = int(raw_n)
        except ValueError:
            n = 0
        if n > 0:
            ids = ids[:n]
    return [int(i) for i in ids]


def install_frspec_draft_head(text: Any) -> dict[str, Any]:
    """Swap ``text._mtplx_draft_lm_head`` for a row-pruned copy.

    Call AFTER the normal draft head install. Returns a report dict; on any
    contract miss it leaves the full head in place (fail-open to correctness).
    """
    import mlx.core as mx
    import mlx.nn as nn

    started = time.perf_counter()
    head = getattr(text, "_mtplx_draft_lm_head", None)
    if head is None:
        return {"installed": False, "reason": "no_draft_lm_head"}
    if not isinstance(head, nn.QuantizedLinear):
        return {"installed": False, "reason": f"head_type_{type(head).__name__}"}

    ids = load_frspec_ids()
    if not ids:
        return {"installed": False, "reason": "no_ids"}

    vocab_rows = int(head.weight.shape[0])
    if max(ids) >= vocab_rows or min(ids) < 0:
        return {"installed": False, "reason": "ids_out_of_range", "vocab_rows": vocab_rows}
    n = len(ids)
    if n >= vocab_rows:
        return {"installed": False, "reason": "not_actually_pruned", "n": n}

    ids_arr = mx.array(ids, dtype=mx.int32)
    group_size = int(getattr(head, "group_size", 64))
    bits = int(getattr(head, "bits", 4))
    in_dims = int(head.scales.shape[1]) * group_size

    pruned = nn.QuantizedLinear(
        in_dims,
        n,
        bias="bias" in head,
        group_size=group_size,
        bits=bits,
    )
    pruned.weight = mx.take(head.weight, ids_arr, axis=0)
    pruned.scales = mx.take(head.scales, ids_arr, axis=0)
    if "biases" in head:
        pruned.biases = mx.take(head.biases, ids_arr, axis=0)
    if "bias" in head:
        pruned.bias = mx.take(head.bias, ids_arr, axis=0)
    mx.eval(pruned.parameters())

    # Side-stamped only: the device draft core swaps this head in around its
    # own trace window. The global _mtplx_draft_lm_head stays full-vocab so
    # legacy draft paths (dense draft_q consumers index by real token id)
    # keep their contract untouched.
    text._mtplx_frspec_draft_head = pruned
    text._mtplx_frspec_full_vocab = vocab_rows
    text._mtplx_frspec_ids = ids_arr
    report = {
        "installed": True,
        "n": n,
        "vocab_rows": vocab_rows,
        "bits": bits,
        "group_size": group_size,
        "bytes_ratio": round(n / vocab_rows, 4),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    logger.info("[frspec] pruned draft lm_head installed: %s", report)
    return report
