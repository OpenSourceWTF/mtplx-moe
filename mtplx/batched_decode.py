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
# Build-1 fallback: force the exact Phase-1 SERIAL loop (4-5 blocking syncs per
# cycle) instead of the default single-sync pipelined loop.  This is the A/B
# switch — set it to compare the parallelized scheduling against the serial
# baseline WITHOUT any change to the committed token stream (the loop is a pure
# scheduling change; both commit the identical greedy sequence).  The ``serial=``
# argument overrides this env when passed explicitly.
BATCHED_DECODE_SERIAL_ENV = "MTPLX_A3B_BATCHED_DECODE_SERIAL"
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


def batched_decode_serial(environ: dict[str, str] | None = None) -> bool:
    """True iff ``MTPLX_A3B_BATCHED_DECODE_SERIAL`` is set truthy.  Fail-closed.

    The default (False) selects the Build-1 single-sync pipelined loop.
    """
    env = os.environ if environ is None else environ
    return str(env.get(BATCHED_DECODE_SERIAL_ENV, "")).strip().lower() in _TRUTHY


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


def _eval_bundle(bundle: Any) -> tuple[list[int], list[int], list[int], list[int]]:
    """THE single per-cycle critical-path sync of the pipelined loop.

    ``bundle`` is the in-graph decision tensor ``[4, B]`` stacking, per cycle,
    ``(x0, draft, x1, accept)`` — all computed on device with no host round-trip.
    One :func:`mx.eval` + one ``tolist`` reads the whole decision for the cycle;
    it is the ONLY blocking eval on the steady-state critical path (the accept
    mask must reach the host to drive the uniform full-B repair branch — that is
    why the budget is 1 sync/cycle, not 0; folding the branch on-device is
    Build-2).  Kept as a module-level seam so a test can monkeypatch it and count
    exactly one call per cycle.
    """
    import mlx.core as mx

    mx.eval(bundle)
    rows = bundle.tolist()  # [[x0...],[draft...],[x1...],[accept...]]
    return (
        [int(t) for t in rows[0]],
        [int(t) for t in rows[1]],
        [int(t) for t in rows[2]],
        [int(t) for t in rows[3]],
    )


def generate_greedy_batched(
    rt: Any,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_token_ids: set[int] | None = None,
    use_mtp_draft: bool = True,
    collect_stats: bool = True,
    cohort_slots: int | None = None,
    pad_id: int = 0,
    serial: bool | None = None,
) -> BatchedDecodeResult:
    """Greedy multi-stream batched decode.

    ``prompts`` is a list of REAL token-id sequences that MUST share a length
    (use :func:`left_pad_prompts` first for a ragged batch).  Every real stream is
    decoded to ``max_new_tokens`` greedy tokens (or an earlier stop token),
    committing 2 greedy tokens per cycle via one ``[B,2]`` verify forward and, on
    any real-stream reject, one uniform ``[B,2]`` full-B repair forward.

    FIXED-SHAPE COHORT MODE (``cohort_slots``).  When set (e.g. ``8``), the prompt
    list is padded to exactly ``cohort_slots`` streams with DUMMY prompts
    (``[pad_id] * prompt_len``).  Dummy slots — like finished streams — keep
    occupying their row in every forward but commit nothing and never trigger the
    repair branch (they are masked out of the all-accept decision).  EVERY forward
    therefore has identical ``[cohort_slots, ·]`` shapes regardless of how many
    real streams exist, which is what makes the per-stream sha gate FIXED-SHAPE:
    stream ``b`` batched among other real prompts vs stream ``b`` alone in a cohort
    of the same slot count differ only in the OTHER rows' content, so a bitwise
    match rests solely on per-row forward independence (``do_sort`` pinned via
    ``MTPLX_A3B_MOE_FORCE_UNSORTED``).  Results report REAL streams only.

    LOOP (``serial``).  Default (``serial=False``, or the
    ``MTPLX_A3B_BATCHED_DECODE_SERIAL`` env fallback unset) runs the Build-1
    single-sync PIPELINED loop: the per-cycle decision (x0, draft, x1, accept) is
    computed on-device and read back with ONE :func:`mx.eval` (:func:`_eval_bundle`)
    — the only blocking sync on the steady-state critical path — while the previous
    cycle's commit/stop bookkeeping drains one cycle behind (a stopped stream
    over-runs a bounded ``<=2`` cycles of uncommitted garbage).  ``serial=True``
    runs the exact Phase-1 serial loop (4-5 blocking syncs/cycle) for A/B; both
    commit the IDENTICAL greedy sequence — the parallelization is a pure scheduling
    change.

    The committed sequence of real stream ``b`` is byte-identical across loops and
    to running ``[prompts[b]]`` alone through this function in a cohort of the same
    slot count (the correctness contract); assert it with :func:`diff_streams`.
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
    n_real = len(prompts)
    prompt_len = len(prompts[0])
    if prompt_len < 1:
        raise ValueError("each prompt needs at least one token")
    if any(len(p) != prompt_len for p in prompts):
        raise ValueError(
            "all prompts must share a length (shared-offset cache); "
            "left_pad_prompts() equalizes a ragged batch"
        )

    # --- fixed-shape cohort padding: append dummy slots so every forward is the
    # same shape regardless of the real-stream count.  Dummy content is a fixed
    # pad-token prompt of the shared length.
    slots: list[list[int]] = [[int(t) for t in p] for p in prompts]
    if cohort_slots is not None:
        cohort_slots = int(cohort_slots)
        if cohort_slots < n_real:
            raise ValueError(
                f"cohort_slots ({cohort_slots}) must be >= the real prompt count "
                f"({n_real})"
            )
        dummy = [int(pad_id)] * prompt_len
        slots.extend([list(dummy) for _ in range(cohort_slots - n_real)])
    batch = len(slots)  # total rows in every forward (real + dummy) = FIXED shape
    real_slots = range(n_real)  # only these commit / are reported / gate the repair

    if serial is None:
        serial = batched_decode_serial()
    stop = {int(t) for t in (stop_token_ids or set())}

    started_all = time.perf_counter()
    cache = rt.make_cache()

    # --- batched prefill: one [batch, prompt_len] forward -> per-stream last logits.
    started = time.perf_counter()
    with attention_phase("prefill"):
        logits, hidden = rt.forward_ar(
            mx.array(slots),
            cache=cache,
            return_hidden=True,
        )
    mx.eval(logits, hidden)
    if int(logits.shape[0]) != batch or int(hidden.shape[0]) != batch:
        raise RuntimeError(
            f"prefill collapsed the batch dim: logits {tuple(logits.shape)} "
            f"hidden {tuple(hidden.shape)} for B={batch}"
        )
    logits_last = logits[:, -1, :]  # [batch, V]
    hidden_last = hidden[:, -1:, :]  # [batch, 1, H]
    prefill_s = time.perf_counter() - started

    tokens: list[list[int]] = [[] for _ in range(batch)]
    finish: list[str | None] = [None] * batch
    done = [False] * batch
    # Dummy slots occupy their row but are inert: pre-marked done so they commit
    # nothing and drop out of the ``all(done)`` termination test.
    for b in range(n_real, batch):
        done[b] = True
        finish[b] = "dummy"

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

    def _commit_pair(x0: list[int], x1: list[int]) -> None:
        for b in range(batch):
            _commit(b, x0[b])
            _commit(b, x1[b])

    def _verify_shape_ok(v_logits: Any, v_hidden: Any) -> None:
        if (
            int(v_logits.shape[0]) != batch
            or int(v_hidden.shape[0]) != batch
            or int(v_hidden.shape[1]) != 2
        ):
            raise RuntimeError(
                f"verify collapsed shape: logits {tuple(v_logits.shape)} "
                f"hidden {tuple(v_hidden.shape)} for [B={batch}, rows=2]"
            )

    cycles = 0
    forwards = 0
    all_accept_cycles = 0
    repair_cycles = 0
    started_decode = time.perf_counter()

    if serial:
        # ================= EXACT PHASE-1 SERIAL LOOP (A/B baseline) =============
        # 4-5 blocking syncs per cycle; behaviour identical to the original driver
        # except dummy slots are masked out of the all-accept decision.
        max_cycles = max_new_tokens + 1  # guards a runaway
        while not all(done) and cycles < max_cycles:
            x0 = _argmax_ids(logits_last)
            if use_mtp_draft:
                draft_logits = rt.draft_mtp(
                    hidden_last,
                    mx.array([[int(t)] for t in x0]),
                    mtp_cache=rt.make_mtp_cache(),
                )
                draft = _argmax_ids(draft_logits[:, -1, :])
            else:
                draft = list(x0)

            snapshot = snapshot_untrimmable_cache(cache)
            with attention_phase("decode_verify"):
                v_logits, v_hidden = rt.forward_ar(
                    mx.array([[x0[b], draft[b]] for b in range(batch)]),
                    cache=cache,
                    return_hidden=True,
                )
            mx.eval(v_logits, v_hidden)
            forwards += 1
            _verify_shape_ok(v_logits, v_hidden)

            x1 = _argmax_ids(v_logits[:, 0, :])
            all_accept = all(draft[b] == x1[b] for b in real_slots)

            if all_accept:
                logits_last = v_logits[:, 1, :]
                hidden_last = v_hidden[:, 1:2, :]
                all_accept_cycles += 1
            else:
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

            _commit_pair(x0, x1)
            cycles += 1
    else:
        # ============ BUILD-1 SINGLE-SYNC PIPELINED LOOP (default) ==============
        # ONE blocking eval per cycle (the decision bundle); the previous cycle's
        # commit/stop bookkeeping drains one cycle behind the GPU submission.
        max_cycles = max_new_tokens + 3  # +lag headroom over the serial guard

        def _submit(ll: Any, hl: Any) -> tuple[Any, Any, Any, Any]:
            """Build one cycle's device graph (no host sync).

            Returns lazy ``(snapshot, v_logits, v_hidden, bundle)`` where
            ``bundle`` is ``[4, batch]`` = stack(x0, draft, x1, accept), all
            computed on-device.
            """
            snapshot = snapshot_untrimmable_cache(cache)
            x0_ids = mx.argmax(ll, axis=-1)  # [batch]
            if use_mtp_draft:
                draft_logits = rt.draft_mtp(
                    hl,
                    mx.expand_dims(x0_ids, axis=1),
                    mtp_cache=rt.make_mtp_cache(),
                )
                draft_ids = mx.argmax(draft_logits[:, -1, :], axis=-1)
            else:
                draft_ids = x0_ids
            with attention_phase("decode_verify"):
                v_logits, v_hidden = rt.forward_ar(
                    mx.stack([x0_ids, draft_ids], axis=1),
                    cache=cache,
                    return_hidden=True,
                )
            x1_ids = mx.argmax(v_logits[:, 0, :], axis=-1)  # [batch]
            accept = (draft_ids == x1_ids).astype(mx.int32)  # [batch]
            bundle = mx.stack([x0_ids, draft_ids, x1_ids, accept], axis=0)  # [4,batch]
            return snapshot, v_logits, v_hidden, bundle

        def _read_repair(
            snapshot: Any, v_logits: Any, v_hidden: Any, bundle: Any
        ) -> tuple[list[int], list[int], Any, Any]:
            """The single per-cycle sync + the host-side accept/repair branch.

            Returns ``(x0, x1, next_logits_last, next_hidden_last)``.
            """
            nonlocal forwards, all_accept_cycles, repair_cycles
            _verify_shape_ok(v_logits, v_hidden)
            x0, draft, x1, accept = _eval_bundle(bundle)  # THE one blocking sync
            forwards += 1
            all_accept = all(accept[b] for b in real_slots)  # dummies masked out
            if all_accept:
                next_ll = v_logits[:, 1, :]
                next_hl = v_hidden[:, 1:2, :]
                all_accept_cycles += 1
            else:
                rollback_after_verify(cache, snapshot, verified_tokens=2)
                with attention_phase("decode_verify"):
                    r_logits, r_hidden = rt.forward_ar(
                        mx.array([[x0[b], x1[b]] for b in range(batch)]),
                        cache=cache,
                        return_hidden=True,
                    )
                forwards += 1
                repair_cycles += 1
                next_ll = r_logits[:, 1, :]
                next_hl = r_hidden[:, 1:2, :]
            return x0, x1, next_ll, next_hl

        # Prologue: submit + read cycle 0 so the loop always holds one committed-
        # but-not-yet-drained cycle in ``pending``.
        snapshot, v_logits, v_hidden, bundle = _submit(logits_last, hidden_last)
        mx.async_eval(bundle, v_logits, v_hidden)
        x0, x1, logits_last, hidden_last = _read_repair(
            snapshot, v_logits, v_hidden, bundle
        )
        pending: tuple[list[int], list[int]] = (x0, x1)
        cycles = 1

        while not all(done) and cycles < max_cycles:
            # 1. Submit the NEXT cycle's forward + decision bundle (kick the GPU).
            snapshot, v_logits, v_hidden, bundle = _submit(logits_last, hidden_last)
            mx.async_eval(bundle, v_logits, v_hidden)
            # 2. Drain the PREVIOUS cycle's commit while the GPU runs this cycle
            #    (one-cycle-lag; done-flags trail by a bounded <=2 cycles).
            _commit_pair(*pending)
            # 3. The single blocking sync for this cycle + accept/repair branch.
            x0, x1, logits_last, hidden_last = _read_repair(
                snapshot, v_logits, v_hidden, bundle
            )
            pending = (x0, x1)
            cycles += 1

        # Flush the final cycle's deferred commit.
        _commit_pair(*pending)

    decode_s = time.perf_counter() - started_decode

    for b in real_slots:
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
        for b in real_slots
    ]
    generated = sum(len(tokens[b]) for b in real_slots)
    meta: dict[str, Any] = {}
    if collect_stats:
        meta = {
            "elapsed_s": time.perf_counter() - started_all,
            "use_mtp_draft": bool(use_mtp_draft),
            "shared_offset_lane": "batch_generic_kv+stock_attention",
            "scheme": "uniform_+2_per_cycle_full_B_repair",
            "loop": "serial" if serial else "pipelined_single_sync",
            "cohort_slots": None if cohort_slots is None else int(cohort_slots),
            "real_streams": n_real,
            "syncs_per_cycle": None if serial else 1,
            "phase": 1,
            "phase2_remaining": PHASE2_REMAINING,
        }
    return BatchedDecodeResult(
        batch_size=n_real,
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
