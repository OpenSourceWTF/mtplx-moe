"""Fork-EV shadow telemetry — HYPER-PLAN Phase-3 H2 gate pricing (2026-08-26).

Answers "does margin-triggered B2 fork EV clear +10% whole-decode?" WITHOUT
building the tree: an observe-only accounting layer on the linear mtpk accept
loop (env ``MTPLX_FORKEV_TELEMETRY=1``, default off). It extends the
``MTPLX_DELTA_TELEMETRY`` instrument (commit 2010018b: correction-rank-in-
draft-candidates on every rejection) from unconditioned hit counting into a
margin-conditioned fork-EV simulator with a threshold policy sweep.

Trajectory neutrality
---------------------
The recorder never touches the RNG, never mutates draft/target state, and only
reads host-side NumPy arrays the lane already materialized
(``SparseDistribution.probs`` / ``.token_ids`` — the same arrays the
Δ-telemetry reads). Disabled: a single env read per generate call and one
``is not None`` check per round. Enabled: pure-NumPy top-2 on <=top-k-sized
sparse supports — zero extra MLX evals, zero graph changes. Draft
distributions that are NOT sparse-materialized (greedy lanes, dense arrays)
are never inspected; they count as ``margin_unavailable``.

The shadow B2 tree being priced
-------------------------------
At a low-margin drafted position r (1-indexed, K drafts in the round) the real
H2 design forks the draft into its #1 and #2 candidates and verifies both
branches in one forward (NAX extended window / shared walks). The shadow
prices a SAME-DEPTH tree: the linear chain keeps its K positions; branch2
enters at r and continues to the same total depth K, i.e. it owns
``h = K - r`` continuation slots. A real implementation may grant branch2 more
window — the same-depth choice is the conservative floor.

Accounting semantics (what counts as saved tokens)
--------------------------------------------------
Linear lane at a rejection r: commits (r-1) accepted drafts + the
residual-sampled correction c, round ends.

* hit := (c == draft's #2 candidate at r). The residual sample landing on the
  draft's #2 candidate is the observable proxy for "the tree's second branch
  would have been walked" — the identical proxy behind the Δ-telemetry
  receipts (depth1 .087 / depth2 .286 / depth3 .429, n=21), so the
  conditioned numbers stay comparable to the unconditioned ones.
* The hit token itself is NEVER counted as saved: the linear lane also
  commits a token at position r (the correction); a fork changes which
  mechanism commits it, not the count.
* On a hit with h >= 1 the tree walks branch2's continuation. Every branch
  node's target row is inside the same verify forward, so the walk commits
  its accepted run j plus one terminal token (residual correction at the
  first branch rejection, or the leaf bonus when all h accept):
  ``j + 1`` extra tokens, ``j in [0, h]``.
* j is unobservable (branch2's continuation was never drafted). Proxy: the
  NEXT linear round starts from the identical committed state ([..., c] with
  c == c2) and its accepted-draft run a' measures model/draft agreement from
  that exact state. Known bias: a' is measured at shallow MTP depths (1..K')
  while branch2's continuation sits at deeper recursive depths (r+1..K)
  where acceptance decays, so a' over-estimates j.
* Estimators (both streamed):
    - ``saved_hi = 0 if h == 0 else 1 + min(a', h)``
      the true tree shape (run proxy + terminal token, <= h+1). Upper
      bracket.
    - ``saved_lo = 0 if h == 0 else 1 + min(a', h - 1)``   [PRIMARY]
      keeps the structural +1 — the row after c2 is in the forward and is
      exactly the row the linear lane's next-round primary samples, so
      committing one token from it requires no draft to be right — then
      burns one continuation slot: forgoes the leaf bonus and tightens the
      depth-mismatched run proxy by one. Bound: ``saved_lo <= h = K - r``
      ("a fork can never save more than the slots it had").
* h == 0 (rejection at the deepest draft position): a same-depth tree has no
  continuation slots there — saved = 0 even on a hit.
* Resolution timing: saved needs a', so a hit parks a pending record resolved
  by the next observed round. A pending still open at generation end
  resolves to saved = 0 (the stream ended; nothing followed to save) and is
  counted in ``pending_unresolved``.

Policy sweep (per threshold T over the draft margin)
----------------------------------------------------
margin := q1 - q2 of the draft sampler's SHAPED sparse distribution — the
exact q the acceptance test uses, already on host. Support of 1 => margin 1.0
(fully confident, never triggers). At draft-temp 0.1 expect margins
concentrated near 1.0; hence the sweep reaches down to 0.05.

* fired(T): the round has >= 1 drafted position with margin < T (fork cost
  side — how often the trigger spends window budget).
* "first" variant — single fork budget: fork at the FIRST low-margin
  position; benefit only when the round's model rejection lands exactly
  there AND hits.
* "at_rejection" variant — fork at every low-margin position (window budget
  permitting): benefit whenever the rejection's own margin < T and hits.
  This is the margin-CONDITIONED analogue of the unconditioned n=21
  receipts — the number report 6 §3's arithmetic needs.
* EV tokens/round = saved_sum / rounds_observed  (lo and hi).

Exclusions (each separately counted, never silently dropped)
------------------------------------------------------------
* grammar-clamped rejections (``constraint_clamped``): the branch would face
  the same grammar mask; pricing them as fork wins would be fiction.
* rejections without a sparse draft distribution: ``margin_unavailable``.
* rounds that bypass the standard accept loop (context-copy block commits,
  pre-verify continues): simply never observed — the denominator is
  "verify rounds through the standard accept path".

Env knobs
---------
* ``MTPLX_FORKEV_TELEMETRY=1``  master gate (mirrors ``_env_truthy``).
* ``MTPLX_FORKEV_THRESHOLDS="0.05,0.1,0.2,0.3,0.5"``  optional sweep
  override (comma-separated floats in (0, 1]).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from .sampling import SparseDistribution

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)
_DECILES = 10


def _env_truthy(name: str) -> bool:
    # Mirrors generation._env_truthy exactly (kept local so this module never
    # imports the runtime).
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _decile(margin: float) -> int:
    return min(_DECILES - 1, max(0, int(float(margin) * _DECILES)))


class _PolicyCell:
    """Streaming accumulators for one margin threshold."""

    __slots__ = (
        "threshold",
        "fired_rounds",
        "first_fork_rejections",
        "first_hits",
        "first_saved_lo",
        "first_saved_hi",
        "at_rejection_forks",
        "at_rejection_hits",
        "at_rejection_saved_lo",
        "at_rejection_saved_hi",
    )

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.fired_rounds = 0
        self.first_fork_rejections = 0
        self.first_hits = 0
        self.first_saved_lo = 0
        self.first_saved_hi = 0
        self.at_rejection_forks = 0
        self.at_rejection_hits = 0
        self.at_rejection_saved_lo = 0
        self.at_rejection_saved_hi = 0


class _Pending:
    """A hit awaiting its saved-token resolution from the next round."""

    __slots__ = ("headroom", "depth", "decile", "first_matched", "any_matched")

    def __init__(
        self,
        headroom: int,
        depth: int,
        decile: int,
        first_matched: list[int],
        any_matched: list[int],
    ) -> None:
        self.headroom = int(headroom)
        self.depth = int(depth)
        self.decile = int(decile)
        # Threshold indices whose policy variant this hit credits on resolve.
        self.first_matched = first_matched
        self.any_matched = any_matched


class ForkEVRecorder:
    """Observe-only fork-EV shadow simulator for the mtpk accept loop.

    One instance per generate call. All state is host-side Python/NumPy;
    ``observe_round`` is the only hot-path entry (once per verify round).
    """

    def __init__(self, thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> None:
        cleaned: list[float] = []
        for value in thresholds:
            value = float(value)
            if 0.0 < value <= 1.0 and value not in cleaned:
                cleaned.append(value)
        if not cleaned:
            cleaned = list(DEFAULT_THRESHOLDS)
        cleaned.sort()
        self.thresholds: tuple[float, ...] = tuple(cleaned)
        self.policy: list[_PolicyCell] = [_PolicyCell(t) for t in self.thresholds]
        self.rounds = 0
        self.rejections = 0
        self.clamped_rejections = 0
        self.margin_unavailable_rejections = 0
        self.hits = 0
        self.pending_unresolved = 0
        self.errors = 0
        self._pending: _Pending | None = None
        self._finalized = False
        # margin decile x depth sufficient statistics; depth keys grow lazily.
        self._positions: dict[int, list[int]] = {}
        self._bin_rejections: dict[int, list[int]] = {}
        self._bin_hits: dict[int, list[int]] = {}
        self._bin_saved_lo: dict[int, list[int]] = {}
        self._bin_saved_hi: dict[int, list[int]] = {}

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls) -> "ForkEVRecorder | None":
        """One env read; returns None (fully inert) unless the gate is set."""
        if not _env_truthy("MTPLX_FORKEV_TELEMETRY"):
            return None
        raw = os.environ.get("MTPLX_FORKEV_THRESHOLDS", "").strip()
        if not raw:
            return cls()
        try:
            thresholds = [float(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            thresholds = []
        return cls(thresholds or DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------ margin math
    @staticmethod
    def margins_and_top2(
        draft_probs: Sequence[Any],
    ) -> tuple[list[float | None], list[int | None]]:
        """Per-position (top-2 margin, #2 candidate id) from the draft's own
        shaped sparse distributions. Positions without a SparseDistribution
        yield (None, None) — never inspected further (no dense scans, no
        device evals). Support of 1 yields margin 1.0 with no #2 candidate.
        """
        margins: list[float | None] = []
        top2: list[int | None] = []
        for q in draft_probs:
            if isinstance(q, SparseDistribution):
                probs = q.probs
                if probs.shape[0] >= 2:
                    order = np.argsort(-probs)
                    margins.append(float(probs[order[0]] - probs[order[1]]))
                    top2.append(int(q.token_ids[order[1]]))
                else:
                    margins.append(1.0)
                    top2.append(None)
            else:
                margins.append(None)
                top2.append(None)
        return margins, top2

    # -------------------------------------------------------------- bin utils
    def _bin_row(self, table: dict[int, list[int]], depth: int) -> list[int]:
        row = table.get(depth)
        if row is None:
            row = [0] * _DECILES
            table[depth] = row
        return row

    # ---------------------------------------------------------------- observe
    def observe_round(
        self,
        *,
        draft_probs: Sequence[Any],
        attempted: int,
        accepted: int,
        rejection_index: int | None,
        correction: int | None,
        clamped: bool = False,
    ) -> None:
        """Record one verify round of the standard accept loop.

        attempted        len(draft_tokens) for the round (may be 0).
        accepted         accepted-draft run before the round ended.
        rejection_index  0-indexed draft position of the round's rejection
                         (``event["rejected_at_depth"] - 1``), None when the
                         round ended without one (all-accept, accepted stop).
        correction       the committed/model correction token at the
                         rejection, None when rejection_index is None.
        clamped          True when the rejection was a grammar-constraint
                         clamp, not a model disagreement.
        """
        # 1) resolve last round's pending hit with THIS round's accepted run.
        if self._pending is not None:
            self._resolve_pending(accepted)

        self.rounds += 1
        if attempted <= 0:
            return

        margins, top2 = self.margins_and_top2(draft_probs[:attempted])

        # 2) drafted-position census (margin decile x depth) + trigger firing.
        first_low: list[int | None] = [None] * len(self.thresholds)
        for index, margin in enumerate(margins):
            if margin is None:
                continue
            row = self._bin_row(self._positions, index + 1)
            row[_decile(margin)] += 1
            for t_index, threshold in enumerate(self.thresholds):
                if first_low[t_index] is None and margin < threshold:
                    first_low[t_index] = index
        for t_index, position in enumerate(first_low):
            if position is not None:
                self.policy[t_index].fired_rounds += 1

        # 3) the round's rejection, if any.
        if rejection_index is None or correction is None:
            return
        self.rejections += 1
        if clamped:
            self.clamped_rejections += 1
            return
        if rejection_index >= len(margins) or margins[rejection_index] is None:
            self.margin_unavailable_rejections += 1
            return

        margin = float(margins[rejection_index])
        depth = rejection_index + 1
        decile = _decile(margin)
        self._bin_row(self._bin_rejections, depth)[decile] += 1

        hit = top2[rejection_index] is not None and int(correction) == int(
            top2[rejection_index]
        )
        first_matched = [
            t_index
            for t_index, position in enumerate(first_low)
            if position == rejection_index
        ]
        any_matched = [
            t_index
            for t_index, threshold in enumerate(self.thresholds)
            if margin < threshold
        ]
        for t_index in first_matched:
            self.policy[t_index].first_fork_rejections += 1
        for t_index in any_matched:
            self.policy[t_index].at_rejection_forks += 1
        if not hit:
            return

        self.hits += 1
        self._bin_row(self._bin_hits, depth)[decile] += 1
        for t_index in first_matched:
            self.policy[t_index].first_hits += 1
        for t_index in any_matched:
            self.policy[t_index].at_rejection_hits += 1
        # Saved tokens need the NEXT round's accepted run; park the hit.
        self._pending = _Pending(
            headroom=attempted - depth,
            depth=depth,
            decile=decile,
            first_matched=first_matched,
            any_matched=any_matched,
        )

    def _resolve_pending(self, next_round_accepted: int) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        h = pending.headroom
        a_next = max(0, int(next_round_accepted))
        if h <= 0:
            saved_lo = saved_hi = 0
        else:
            saved_lo = 1 + min(a_next, h - 1)
            saved_hi = 1 + min(a_next, h)
        self._bin_row(self._bin_saved_lo, pending.depth)[pending.decile] += saved_lo
        self._bin_row(self._bin_saved_hi, pending.depth)[pending.decile] += saved_hi
        for t_index in pending.first_matched:
            self.policy[t_index].first_saved_lo += saved_lo
            self.policy[t_index].first_saved_hi += saved_hi
        for t_index in pending.any_matched:
            self.policy[t_index].at_rejection_saved_lo += saved_lo
            self.policy[t_index].at_rejection_saved_hi += saved_hi

    # --------------------------------------------------------------- finalize
    def finalize(self) -> None:
        """End of generation: a still-open pending resolves to saved = 0 —
        the stream ended, nothing followed to save (conservative; counted)."""
        if self._finalized:
            return
        self._finalized = True
        if self._pending is not None:
            self.pending_unresolved += 1
            pending, self._pending = self._pending, None
            # Explicit zero-resolution: bins/policy get 0 saved; the hit
            # itself stays counted in hits / *_hits.
            self._bin_row(self._bin_saved_lo, pending.depth)[pending.decile] += 0
            self._bin_row(self._bin_saved_hi, pending.depth)[pending.decile] += 0

    # --------------------------------------------------------------- snapshot
    @staticmethod
    def _rate(numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return numerator / denominator

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe aggregate — the numbers report 6 §3 slots in directly:
        per threshold T: fire rate, hit rate, EV extra committed tokens/round
        (lo = conservative primary, hi = bracket)."""
        rounds = self.rounds
        policy_rows: list[dict[str, Any]] = []
        for cell in self.policy:
            policy_rows.append(
                {
                    "threshold": cell.threshold,
                    "fired_rounds": cell.fired_rounds,
                    "fire_rate": self._rate(cell.fired_rounds, rounds),
                    "first_fork_rejections": cell.first_fork_rejections,
                    "first_hits": cell.first_hits,
                    "first_hit_rate": self._rate(
                        cell.first_hits, cell.first_fork_rejections
                    ),
                    "first_saved_tokens_lo": cell.first_saved_lo,
                    "first_saved_tokens_hi": cell.first_saved_hi,
                    "first_ev_tokens_per_round_lo": self._rate(
                        cell.first_saved_lo, rounds
                    ),
                    "first_ev_tokens_per_round_hi": self._rate(
                        cell.first_saved_hi, rounds
                    ),
                    "at_rejection_forks": cell.at_rejection_forks,
                    "at_rejection_hits": cell.at_rejection_hits,
                    "at_rejection_hit_rate": self._rate(
                        cell.at_rejection_hits, cell.at_rejection_forks
                    ),
                    "at_rejection_saved_tokens_lo": cell.at_rejection_saved_lo,
                    "at_rejection_saved_tokens_hi": cell.at_rejection_saved_hi,
                    "at_rejection_ev_tokens_per_round_lo": self._rate(
                        cell.at_rejection_saved_lo, rounds
                    ),
                    "at_rejection_ev_tokens_per_round_hi": self._rate(
                        cell.at_rejection_saved_hi, rounds
                    ),
                }
            )

        def _bins(table: dict[int, list[int]]) -> dict[str, list[int]]:
            return {f"d{depth}": list(row) for depth, row in sorted(table.items())}

        return {
            "enabled": True,
            "rounds": rounds,
            "rejections": self.rejections,
            "clamped_rejections": self.clamped_rejections,
            "margin_unavailable_rejections": self.margin_unavailable_rejections,
            "hits": self.hits,
            "pending_unresolved": self.pending_unresolved,
            "errors": self.errors,
            "thresholds": list(self.thresholds),
            "policy": policy_rows,
            "bins": {
                "positions": _bins(self._positions),
                "rejections": _bins(self._bin_rejections),
                "hits": _bins(self._bin_hits),
                "saved_lo": _bins(self._bin_saved_lo),
                "saved_hi": _bins(self._bin_saved_hi),
            },
        }

    def stderr_summary(self) -> str:
        """One greppable line per request, Δ-telemetry style."""
        parts = [
            f"[forkev-telemetry] rounds={self.rounds}",
            f"rejections={self.rejections}",
            f"hits={self.hits}",
            f"unavailable={self.margin_unavailable_rejections}",
            f"errors={self.errors}",
        ]
        for cell in self.policy:
            fire = self._rate(cell.fired_rounds, self.rounds)
            hit = self._rate(cell.at_rejection_hits, cell.at_rejection_forks)
            ev = self._rate(cell.at_rejection_saved_lo, self.rounds)
            parts.append(
                f"T={cell.threshold:g}:fire={fire if fire is None else round(fire, 4)}"
                f",hit={hit if hit is None else round(hit, 4)}"
                f",ev_lo={ev if ev is None else round(ev, 4)}"
            )
        return " ".join(parts)
