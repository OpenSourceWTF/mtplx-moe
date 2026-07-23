"""Multi-stream (cross-request) batched greedy decode for A3B — Phase 1.

WHY THIS EXISTS
---------------
Single-stream A3B decode is latency-bound and kernel-closed (~167 tok/s, §41 of
``claude-s3-serving-integration-build-20260721.md``).  The one remaining
throughput lever is CROSS-REQUEST BATCHING: run ``B`` concurrent decode requests
as ONE ``[B, ·]`` forward per cycle so the ~1054.8 MB of dense weights
(attn/GDN/router/shared-expert/lm_head) are read ONCE and amortized across all
``B`` streams.  The eager probe ``a3b_174_batch_upside_bench.py`` measured this
amortization at ×2.49 ideal / ×2.21 net-ragged @ B=8 (§42/§43).  This module is
the *running decode* that realizes it (the probe timed a bare ``forward_ar``; it
never decoded).

WHAT THIS IS (Phase 1) vs WHAT IT IS NOT (Phase 2)
--------------------------------------------------
This is a GREEDY, uniform-commit multi-stream driver on the BATCH-GENERIC cache
lane (stock KV / GDN caches + stock attention — NOT the served
``VllmMetalPagedKVCache``, which hard-raises at batch>1, ``cache_state.py:955``).
Each cycle:

  1. ``x0_b = argmax(logits_b)``         — the next greedy token per stream.
  2. draft ``d_b`` from the MTP head     — one ``[B,1]`` draft forward.
  3. VERIFY ``forward_ar([B,2])`` on ``[x0_b, d_b]`` — the single amortized
     weight read the probe measured; advances every stream's cache by 2.
  4. ``x1_b = argmax(verify[:,0])``      — the true 2nd greedy token per stream.
     ``accept_b = (d_b == x1_b)``.
  5. If EVERY stream accepted: the verify already put ``x1`` at position O+1 for
     all, so keep it — 1 forward committed 2 tokens for all B (the speculative
     win).  Otherwise: roll the WHOLE batch back to O and re-forward
     ``[B,2] = [x0_b, x1_b]`` (the correct 2 greedy tokens) — a UNIFORM full-B
     repair that keeps the single shared cache offset (Phase-1 constraint).

Because sampling is greedy, the committed sequence per stream is exactly the
target model's greedy-argmax continuation ``x0, x1, x2, …`` — the SAME sequence
regardless of the accept pattern, and byte-identical to that stream run alone
through this driver.  Crucially, for a stream that WOULD have accepted, the
repair re-forward of ``[x0, x1]`` is bit-identical to the verify it replaces
(same tokens, same prefix, same weights, deterministic forward), so a rejecting
neighbour never perturbs an accepting stream.  **That determinism is the Phase-1
correctness contract** (proved on CPU with a fake runtime; the per-stream sha
gate on the real model is fable-main's GPU window).

Phase-1 SCOPE HONESTY: the uniform full-B repair is CORRECT but pays the full
``[B,2]`` weight read again whenever ANY stream rejects — with independent
streams that is most cycles, so this realizes the cross-request amortization but
NOT the §43 compacted-repair economics (repair only the rejecting rows).  The
COMPACTED repair sub-batch (``filter``/merge on the batch-generic cache) is the
plan's hard Phase 2 and is deliberately NOT built here — see the module-level
``PHASE2_REMAINING`` note.  Greedy-only; a p/q ratio-accept (temperature>0) lane
is also Phase 2.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Env gate (fail-closed).  Phase 1 calls ``generate_greedy_batched`` directly
# from the bench; this flag is the seam a future served path (Phase 3) checks
# before routing a cohort here.  OFF => callers never touch this module, so a
# gate-off run is byte-identical to single-stream ``generate_mtpk``.
# --------------------------------------------------------------------------- #
BATCHED_DECODE_ENV = "MTPLX_A3B_BATCHED_DECODE"
_TRUTHY = {"1", "true", "yes", "on"}

# What a real cross-request serving build still needs beyond this module
# (recorded so the Phase-1/Phase-2 boundary is unambiguous):
PHASE2_REMAINING = (
    "compacted repair sub-batch (filter rejecting rows -> [B_reject,2] repair -> "
    "scatter KV back, vs the uniform full-B repair here); per-stream staggered "
    "offsets / ragged-KV for long context (this driver holds ONE shared cache "
    "offset, so all prompts must be equal length); dynamic admission/departure "
    "(mtplx/batching scheduler); and a p/q ratio-accept (temperature>0) lane."
)


def batched_decode_enabled(environ: dict[str, str] | None = None) -> bool:
    """True iff ``MTPLX_A3B_BATCHED_DECODE`` is set truthy.  Fail-closed."""
    env = os.environ if environ is None else environ
    return str(env.get(BATCHED_DECODE_ENV, "")).strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class BatchedStreamResult:
    index: int
    prompt_len: int
    tokens: list[int]
    finish_reason: str
    sha: str


@dataclass
class BatchedDecodeResult:
    batch_size: int
    streams: list[BatchedStreamResult]
    cycles: int
    forwards: int
    all_accept_cycles: int
    repair_cycles: int
    prefill_s: float
    decode_s: float
    generated_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_decode_tokps(self) -> float:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def shas(self) -> list[str]:
        return [s.sha for s in self.streams]


# --------------------------------------------------------------------------- #
# Pure helpers (no MLX — unit-drivable)
# --------------------------------------------------------------------------- #
def token_sha(tokens: list[int]) -> str:
    """Stable 16-hex digest of a committed token sequence (per-stream gate key)."""
    payload = json.dumps([int(t) for t in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def left_pad_prompts(
    prompts: list[list[int]], pad_id: int
) -> tuple[list[list[int]], list[int]]:
    """Left-pad a ragged prompt batch to a shared length with ``pad_id``.

    The batch-generic decode cache carries ONE shared offset, so every stream
    must enter the loop at the same length.  Left-padding keeps each stream's
    TRUE last token at the final position (so its next-token logits are its own),
    at the cost of the model attending to the pad prefix — acceptable for the
    Phase-1 throughput/correctness gate because the single-stream reference is
    fed the IDENTICAL padded prompt (apples-to-apples; the gate tests batching
    isolation, not prompt semantics).  Returns ``(padded, true_lengths)``.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    lengths = [len(p) for p in prompts]
    width = max(lengths)
    if width < 1:
        raise ValueError("each prompt needs at least one token")
    padded = [[int(pad_id)] * (width - len(p)) + [int(t) for t in p] for p in prompts]
    return padded, lengths


def diff_streams(
    batched: list[list[int]], reference: list[list[int]]
) -> list[dict[str, Any]]:
    """Per-stream sha comparison of a batched run vs its single-stream reference.

    Returns one record per stream with ``match`` and, on mismatch, the first
    differing position + a short window around it (so a GPU divergence is
    localized, not just pass/fail).  This is the Phase-1 correctness gate.
    """
    if len(batched) != len(reference):
        raise ValueError(
            f"stream count mismatch: batched {len(batched)} vs reference "
            f"{len(reference)}"
        )
    records: list[dict[str, Any]] = []
    for idx, (bt, rt_) in enumerate(zip(batched, reference)):
        match = bt == rt_
        record: dict[str, Any] = {
            "index": idx,
            "match": match,
            "batched_sha": token_sha(bt),
            "reference_sha": token_sha(rt_),
            "batched_len": len(bt),
            "reference_len": len(rt_),
        }
        if not match:
            first = next(
                (
                    i
                    for i in range(min(len(bt), len(rt_)))
                    if bt[i] != rt_[i]
                ),
                min(len(bt), len(rt_)),
            )
            lo = max(0, first - 2)
            record["first_divergence"] = first
            record["batched_window"] = bt[lo : first + 3]
            record["reference_window"] = rt_[lo : first + 3]
        records.append(record)
    return records


def streams_all_match(records: list[dict[str, Any]]) -> bool:
    return all(bool(r.get("match")) for r in records)


# --------------------------------------------------------------------------- #
# The driver (MLX; lazy imports keep module import cheap)
# --------------------------------------------------------------------------- #
def _argmax_ids(logits_2d: Any) -> list[int]:
    """Greedy argmax over a ``[B, V]`` logits tensor -> list of B python ints."""
    import mlx.core as mx

    ids = mx.argmax(logits_2d, axis=-1)
    mx.eval(ids)
    return [int(t) for t in ids.tolist()]


def generate_greedy_batched(
    rt: Any,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_token_ids: set[int] | None = None,
    use_mtp_draft: bool = True,
    collect_stats: bool = True,
) -> BatchedDecodeResult:
    """Greedy multi-stream batched decode (Phase 1).

    ``prompts`` is a list of ``B`` token-id sequences that MUST share a length
    (use :func:`left_pad_prompts` first for a ragged batch).  Every stream is
    decoded to ``max_new_tokens`` greedy tokens (or an earlier stop token),
    committing 2 greedy tokens per cycle via one ``[B,2]`` verify forward and, on
    any reject, one uniform ``[B,2]`` full-B repair forward.

    The committed sequence of stream ``b`` is byte-identical to running
    ``[prompts[b]]`` alone through this same function (the Phase-1 correctness
    contract); assert it with :func:`diff_streams`.
    """
    import mlx.core as mx

    from mtplx.attention_context import attention_phase
    from mtplx.cache_state import (
        rollback_after_verify,
        snapshot_untrimmable_cache,
    )

    if not rt.mtp_enabled:
        raise RuntimeError("generate_greedy_batched requires an MTP-enabled runtime")
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    batch = len(prompts)
    prompt_len = len(prompts[0])
    if prompt_len < 1:
        raise ValueError("each prompt needs at least one token")
    if any(len(p) != prompt_len for p in prompts):
        raise ValueError(
            "all prompts must share a length (shared-offset cache, Phase 1); "
            "left_pad_prompts() equalizes a ragged batch"
        )
    stop = {int(t) for t in (stop_token_ids or set())}

    started_all = time.perf_counter()
    cache = rt.make_cache()

    # --- batched prefill: one [B, prompt_len] forward -> per-stream last logits.
    started = time.perf_counter()
    with attention_phase("prefill"):
        logits, hidden = rt.forward_ar(
            mx.array([[int(t) for t in p] for p in prompts]),
            cache=cache,
            return_hidden=True,
        )
    mx.eval(logits, hidden)
    if int(logits.shape[0]) != batch or int(hidden.shape[0]) != batch:
        raise RuntimeError(
            f"prefill collapsed the batch dim: logits {tuple(logits.shape)} "
            f"hidden {tuple(hidden.shape)} for B={batch}"
        )
    logits_last = logits[:, -1, :]  # [B, V]
    hidden_last = hidden[:, -1:, :]  # [B, 1, H]
    prefill_s = time.perf_counter() - started

    tokens: list[list[int]] = [[] for _ in range(batch)]
    finish: list[str | None] = [None] * batch
    done = [False] * batch

    def _commit(b: int, tok: int) -> None:
        """Record one committed token for stream b, applying stop/length."""
        if done[b]:
            return
        tokens[b].append(int(tok))
        if int(tok) in stop:
            done[b] = True
            finish[b] = "stop"
        elif len(tokens[b]) >= max_new_tokens:
            done[b] = True
            finish[b] = "length"

    cycles = 0
    forwards = 0
    all_accept_cycles = 0
    repair_cycles = 0
    # Hard cap: even all-2-token cycles cannot exceed this; guards a runaway.
    max_cycles = max_new_tokens + 1
    started_decode = time.perf_counter()

    while not all(done) and cycles < max_cycles:
        # 1. next greedy token per stream (garbage for done streams; unused).
        x0 = _argmax_ids(logits_last)

        # 2. MTP draft (one [B,1] amortized forward), or a trivial self-draft.
        if use_mtp_draft:
            draft_logits = rt.draft_mtp(
                hidden_last,
                mx.array([[int(t)] for t in x0]),
                mtp_cache=rt.make_mtp_cache(),
            )
            draft = _argmax_ids(draft_logits[:, -1, :])
        else:
            draft = list(x0)

        # 3. snapshot (pre-verify offset O) + [B,2] verify of [x0, draft].
        snapshot = snapshot_untrimmable_cache(cache)
        with attention_phase("decode_verify"):
            v_logits, v_hidden = rt.forward_ar(
                mx.array([[x0[b], draft[b]] for b in range(batch)]),
                cache=cache,
                return_hidden=True,
            )
        mx.eval(v_logits, v_hidden)
        forwards += 1
        if (
            int(v_logits.shape[0]) != batch
            or int(v_hidden.shape[0]) != batch
            or int(v_hidden.shape[1]) != 2
        ):
            raise RuntimeError(
                f"verify collapsed shape: logits {tuple(v_logits.shape)} "
                f"hidden {tuple(v_hidden.shape)} for [B={batch}, rows=2]"
            )

        # 4. true 2nd greedy token per stream + accept decision.
        x1 = _argmax_ids(v_logits[:, 0, :])
        all_accept = all(draft[b] == x1[b] for b in range(batch))

        if all_accept:
            # Verify already committed the correct [x0, x1] KV for all streams.
            logits_last = v_logits[:, 1, :]
            hidden_last = v_hidden[:, 1:2, :]
            all_accept_cycles += 1
        else:
            # Uniform full-B repair: roll the whole batch back to O, re-forward
            # the correct [x0, x1].  For an accepting stream this reproduces the
            # verify bit-for-bit, so it never perturbs that stream (determinism
            # = the Phase-1 correctness contract).
            rollback_after_verify(cache, snapshot, verified_tokens=2)
            with attention_phase("decode_verify"):
                r_logits, r_hidden = rt.forward_ar(
                    mx.array([[x0[b], x1[b]] for b in range(batch)]),
                    cache=cache,
                    return_hidden=True,
                )
            mx.eval(r_logits, r_hidden)
            forwards += 1
            repair_cycles += 1
            logits_last = r_logits[:, 1, :]
            hidden_last = r_hidden[:, 1:2, :]

        # 5. commit the two greedy tokens per (still-live) stream.
        for b in range(batch):
            _commit(b, x0[b])
            _commit(b, x1[b])
        cycles += 1

    decode_s = time.perf_counter() - started_decode

    for b in range(batch):
        if finish[b] is None:
            finish[b] = "length" if len(tokens[b]) >= max_new_tokens else "cycle_cap"

    streams = [
        BatchedStreamResult(
            index=b,
            prompt_len=prompt_len,
            tokens=tokens[b],
            finish_reason=str(finish[b]),
            sha=token_sha(tokens[b]),
        )
        for b in range(batch)
    ]
    generated = sum(len(t) for t in tokens)
    meta: dict[str, Any] = {}
    if collect_stats:
        meta = {
            "elapsed_s": time.perf_counter() - started_all,
            "use_mtp_draft": bool(use_mtp_draft),
            "shared_offset_lane": "batch_generic_kv+stock_attention",
            "scheme": "uniform_+2_per_cycle_full_B_repair",
            "phase": 1,
            "phase2_remaining": PHASE2_REMAINING,
        }
    return BatchedDecodeResult(
        batch_size=batch,
        streams=streams,
        cycles=cycles,
        forwards=forwards,
        all_accept_cycles=all_accept_cycles,
        repair_cycles=repair_cycles,
        prefill_s=prefill_s,
        decode_s=decode_s,
        generated_tokens=generated,
        meta=meta,
    )
