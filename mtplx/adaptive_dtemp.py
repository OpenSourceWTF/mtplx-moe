"""Acceptance-EMA adaptive draft-temperature controller (``MTPLX_ADAPTIVE_DTEMP``).

HYPER-PLAN §15 ship shape for the dtemp lever: *"register-dependent, not
ctx-dependent … ship as acceptance-EMA adaptive controller, never a fixed
default."*  Env-gated, default OFF.

Receipts (MEASUREMENTS.md, 2026-08-25 flat-decode marathon; serve arms drafted
at base t1.0 vs the dtemp-0.85 arm, target t0.6 seed 17):

===================  ==========  ==============  ==============  =========
cell                 tps delta   base pos-1      0.85-arm pos-1  verdict
===================  ==========  ==============  ==============  =========
synthetic 71k        +5.1%       .71             .77             helps
synthetic 100k       +3.7%       ~.71            .717-.719       helps
synthetic 120k       -11%        .802            .683            HURTS
natural 184.6k       +13.0%      .750            .889            helps
===================  ==========  ==============  ==============  =========

Mechanism (10:50 PDT entry): sharpening toward 0.85 pays when the MTP head
ranks right-but-diffuse (natural/formulaic register — base pos-1 acceptance
sags into the ~.70-.76 band) and costs when the head is confidently matching
already (duplication-heavy register — base pos-1 >= ~.80, where min(1, p/q)
starts rejecting agreeing tokens as q sharpens past p).

Exactness: the draft temperature shapes only the PROPOSAL distribution q; the
house probability-ratio acceptance with residual correction derives p and q
independently, so the output marginal is exact for any q — including a q whose
temperature moves mid-stream.  Changing dtemp mid-stream changes trajectories
(documented, fine: this is a speed/tuning lane, not a correctness lane).

The schedule — two set-points, state-dependent hysteresis
---------------------------------------------------------

Observation per round: the pos-1 accept *probability* ``min(1, p/q)`` of the
round's first chained draft position (continuous in [0, 1]; same expectation
as the receipt ladders' pos-1 acceptance rate, far lower variance than the
0/1 accept coin).  Rounds that did not draft through the sampled MTP lane
(context-copy streak/block rounds, no-draft rounds) contribute nothing.

* SEED: the first ``seed_rounds`` observed rounds run at the request-resolved
  base temperature; their plain mean seeds the EMA ("seed from the first N
  rounds at the current default temp").
* EMA: ``ema <- (1 - alpha) * ema + alpha * x`` per observed round.
* State DEFAULT (dtemp = resolved base):
  RAISE -> BOOST (0.85) when ``floor <= ema <= raise_threshold`` (0.45..0.78).
  Receipts: 0.85 helps at base pos-1 .71-.758, hurts at .802 — the band edge
  sits between .758 and .802, so raise at <= .78.  Below the .45 floor there
  are no receipts at all: hold the shipped base in unmeasured territory.
* State BOOST (dtemp = 0.85):
  DROP -> DEFAULT when ``ema < drop_threshold`` (0.70).  The thresholds are
  state-dependent because the lever itself moves the operating point: under
  0.85 the hurt case reads .683 while every help case reads .717/.77/.889 —
  the boundary between them is ~.70.
* Hysteresis, three layers:
  1. the state-dependent bands (raise at <= .78 from DEFAULT, drop at < .70
     from BOOST — an 8-point gap);
  2. the plant's own response reinforces both decisions (boost SHIFTS pos-1
     acceptance +.06..+.14 where it helps — away from the drop edge — and
     -.12 where it hurts — decisively through it);
  3. a dwell floor: at least ``dwell_rounds`` observed rounds between
     transitions, which caps flip rate at band edges under EMA noise.

Interaction contract (documented, enforced by the build gate in
``generation.generate_mtpk``):

* Greedy draft chain / greedy coupling: the chain requires BOTH the target
  and draft samplers at temperature <= 0 and is pre-bound per request.  The
  controller activates only when both are > 0 and its set-points are all
  > 0, so the pre-bound chain decision can never go stale and the controller
  never un-greedies a greedy draft.
* Device draft cores (``draft_core`` "device"/"device-d2"): those compile the
  draft sampler INTO the core (temperature baked into the sampling graph,
  and the rebuild signature does not include it), so a mid-stream change
  would silently draft at a stale temperature.  The controller refuses to
  run there (``draft_core != "stock"`` blocker) instead of desyncing.
* fsleg / FR-Spec legacy (``MTPLX_FRSPEC_LEGACY``): composes freely — the
  pruned-head remap happens AFTER the draft sampler shapes and samples q, so
  the controller's temperature applies to the shortlisted logits the same
  way (08-25 attribution matrix: zero interaction between the two levers;
  the fsleg 120k acceptance drop was dtemp's alone).
* Research draft lanes that replace the sampled proposal (adaptive width
  policy's greedy d1/d2, draft margin threshold, adapter ensemble, top-k
  reranker, target-prefix verify, grammar constraints): blocked — pos-1 is
  either not sampled from the draft sampler there or acceptance is clamped,
  so both the lever and its observable are invalid.

Telemetry: the per-request summary (``summary()``) rides GenerationStats as
``draft_sampler_adaptive_dtemp`` and is stamped into the public mtplx_stats
and the request-log envelope only when non-empty (quiet-envelope idiom), so
disabled runs stay byte-stable.  Each transition also lands in the round
event stream as ``adaptive_dtemp_transition``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

ENV_GATE = "MTPLX_ADAPTIVE_DTEMP"
ENV_BOOST = "MTPLX_ADAPTIVE_DTEMP_BOOST"
ENV_RAISE = "MTPLX_ADAPTIVE_DTEMP_RAISE"
ENV_DROP = "MTPLX_ADAPTIVE_DTEMP_DROP"
ENV_FLOOR = "MTPLX_ADAPTIVE_DTEMP_FLOOR"
ENV_ALPHA = "MTPLX_ADAPTIVE_DTEMP_ALPHA"
ENV_SEED_ROUNDS = "MTPLX_ADAPTIVE_DTEMP_SEED_ROUNDS"
ENV_DWELL = "MTPLX_ADAPTIVE_DTEMP_DWELL"

# Transitions past the cap keep counting in `transitions`; only the log stops
# growing (bounded request-log rows even on a pathological flip-flop).
TRANSITION_LOG_CAP = 32


def adaptive_dtemp_enabled(env: Mapping[str, str] | None = None) -> bool:
    """The master gate. Default OFF — founder-gated tuning lane."""

    source = os.environ if env is None else env
    return str(source.get(ENV_GATE, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_float(
    source: Mapping[str, str], name: str, default: float
) -> float:
    raw = str(source.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = str(source.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class AdaptiveDtempConfig:
    """The step schedule's knobs. Defaults = the 08-25 receipt-derived bands."""

    # The measured lever value (+5.1/+3.7/+13% on favorable registers).
    boost_temperature: float = 0.85
    # DEFAULT-state raise edge: helps at base pos-1 .71-.758, hurts at .802.
    raise_threshold: float = 0.78
    # BOOST-state drop edge: under 0.85 the hurt case reads .683, helps .717+.
    drop_threshold: float = 0.70
    # No receipts below ~.45 base pos-1 — hold the shipped default there.
    floor_threshold: float = 0.45
    # Half-life ~34 observed rounds: fast enough to catch a register shift
    # inside a normal response, slow enough that the EMA's Bernoulli-ish
    # noise floor stays well under the 8-point raise/drop gap.
    ema_alpha: float = 0.02
    # Seed mean over the first N observed rounds at the base temperature.
    seed_rounds: int = 12
    # Minimum observed rounds between transitions (chatter cap).
    dwell_rounds: int = 24

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "AdaptiveDtempConfig":
        """Per-knob parse; a missing or unparsable value keeps its default."""

        source = os.environ if env is None else env
        defaults = cls()
        return cls(
            boost_temperature=_env_float(
                source, ENV_BOOST, defaults.boost_temperature
            ),
            raise_threshold=_env_float(
                source, ENV_RAISE, defaults.raise_threshold
            ),
            drop_threshold=_env_float(
                source, ENV_DROP, defaults.drop_threshold
            ),
            floor_threshold=_env_float(
                source, ENV_FLOOR, defaults.floor_threshold
            ),
            ema_alpha=_env_float(source, ENV_ALPHA, defaults.ema_alpha),
            seed_rounds=_env_int(
                source, ENV_SEED_ROUNDS, defaults.seed_rounds
            ),
            dwell_rounds=_env_int(source, ENV_DWELL, defaults.dwell_rounds),
        )

    def validation_error(self) -> str | None:
        """A reason string when the schedule cannot steer safely, else None.

        An invalid schedule makes the controller INERT (never a crash, never
        a guess): generation policy must fail toward the shipped default.
        """

        if not self.boost_temperature > 0.0:
            # A non-positive set-point would flip the greedy-chain /
            # device-core eligibility terms mid-request — never allowed.
            return "boost_temperature_not_positive"
        if not 0.0 < self.ema_alpha <= 1.0:
            return "ema_alpha_out_of_range"
        if not 0.0 < self.drop_threshold < self.raise_threshold < 1.0:
            # drop < raise is the hysteresis gap itself.
            return "thresholds_not_ordered"
        if not 0.0 <= self.floor_threshold < self.raise_threshold:
            return "floor_above_raise"
        if self.seed_rounds < 1:
            return "seed_rounds_below_one"
        if self.dwell_rounds < 0:
            return "dwell_rounds_negative"
        return None


class AdaptiveDraftTemperatureController:
    """Two-set-point flip-flop on the pos-1 accept-probability EMA.

    States: ``seed`` -> ``default`` <-> ``boost``.  ``observe_round`` returns
    the NEW draft temperature when a set-point transition fired (the caller
    rebinds its draft sampler), else None.
    """

    __slots__ = (
        "config",
        "base_temperature",
        "boost_temperature",
        "current_temperature",
        "state",
        "ema",
        "observed_rounds",
        "boost_rounds",
        "transitions",
        "transition_log",
        "_seed_sum",
        "_rounds_since_transition",
    )

    def __init__(
        self,
        *,
        base_temperature: float,
        config: AdaptiveDtempConfig | None = None,
    ) -> None:
        self.config = config if config is not None else AdaptiveDtempConfig.from_env()
        self.base_temperature = float(base_temperature)
        self.boost_temperature = float(self.config.boost_temperature)
        self.current_temperature = self.base_temperature
        self.state = "seed"
        self.ema: float | None = None
        self.observed_rounds = 0
        self.boost_rounds = 0
        self.transitions = 0
        self.transition_log: list[tuple[int, float]] = []
        self._seed_sum = 0.0
        self._rounds_since_transition = 0

    def observe_round(self, pos1_accept_probability: float) -> float | None:
        """Feed one round's pos-1 accept probability; maybe move a set-point.

        Returns the new draft temperature iff a transition fired this round.
        """

        x = float(pos1_accept_probability)
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if self.state == "boost":
            # The observed round was DRAFTED at the pre-decision set-point.
            self.boost_rounds += 1
        self.observed_rounds += 1
        cfg = self.config
        if self.ema is None:
            self._seed_sum += x
            if self.observed_rounds >= cfg.seed_rounds:
                self.ema = self._seed_sum / float(self.observed_rounds)
                self.state = "default"
                # Seeding satisfies the dwell: the first steer may fire now.
                self._rounds_since_transition = cfg.dwell_rounds
                return self._decide()
            return None
        self.ema = (1.0 - cfg.ema_alpha) * self.ema + cfg.ema_alpha * x
        self._rounds_since_transition += 1
        return self._decide()

    def _decide(self) -> float | None:
        cfg = self.config
        assert self.ema is not None
        if self._rounds_since_transition < cfg.dwell_rounds:
            return None
        if self.state == "default":
            if cfg.floor_threshold <= self.ema <= cfg.raise_threshold:
                return self._transition("boost", self.boost_temperature)
            return None
        if self.state == "boost":
            if self.ema < cfg.drop_threshold:
                return self._transition("default", self.base_temperature)
        return None

    def _transition(self, state: str, temperature: float) -> float:
        self.state = state
        self.current_temperature = float(temperature)
        self.transitions += 1
        self._rounds_since_transition = 0
        if len(self.transition_log) < TRANSITION_LOG_CAP:
            self.transition_log.append(
                (self.observed_rounds, self.current_temperature)
            )
        return self.current_temperature

    def summary(self) -> dict[str, object]:
        """The per-request telemetry block (GenerationStats / request log)."""

        cfg = self.config
        return {
            "enabled": True,
            "active": True,
            "state": self.state,
            "base_temperature": round(self.base_temperature, 6),
            "boost_temperature": round(self.boost_temperature, 6),
            "current_temperature": round(self.current_temperature, 6),
            "ema": None if self.ema is None else round(self.ema, 4),
            "observed_rounds": int(self.observed_rounds),
            "boost_rounds": int(self.boost_rounds),
            "transitions": int(self.transitions),
            "transition_log": [
                [int(round_index), float(temperature)]
                for round_index, temperature in self.transition_log
            ],
            "config": {
                "raise_threshold": cfg.raise_threshold,
                "drop_threshold": cfg.drop_threshold,
                "floor_threshold": cfg.floor_threshold,
                "ema_alpha": cfg.ema_alpha,
                "seed_rounds": int(cfg.seed_rounds),
                "dwell_rounds": int(cfg.dwell_rounds),
            },
        }


def build_adaptive_dtemp_controller(
    *,
    base_temperature: float,
    blockers: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> tuple[AdaptiveDraftTemperatureController | None, dict[str, object]]:
    """The single per-request build gate.

    Returns ``(controller, telemetry)``:

    * env gate off -> ``(None, {})`` — the no-op proof: with the gate unset
      the engine holds no controller object and the stats block stays empty,
      so disabled behavior is byte-identical to before this feature existed.
    * env gate on but blocked (lane blockers from the caller, an invalid
      schedule, or a base the schedule cannot steer) -> ``(None, summary)``
      with ``active: False`` and the reasons — visible, never silent.
    * env gate on and clean -> ``(controller, {})``; the caller reads
      ``controller.summary()`` at stats time.
    """

    if not adaptive_dtemp_enabled(env):
        return None, {}
    config = AdaptiveDtempConfig.from_env(env)
    reasons = [str(reason) for reason in blockers]
    config_error = config.validation_error()
    if config_error is not None:
        reasons.append(f"invalid_config:{config_error}")
    base = float(base_temperature)
    if not reasons:
        if base <= 0.0:
            # Defense in depth — the caller's greedy blockers should have
            # caught this already (greedy chain / coupled drafts).
            reasons.append("greedy_draft_base")
        elif abs(base - float(config.boost_temperature)) < 1e-9:
            reasons.append("base_equals_boost")
    if reasons:
        return None, {
            "enabled": True,
            "active": False,
            "inactive_reasons": reasons,
            "base_temperature": round(base, 6),
            "boost_temperature": round(float(config.boost_temperature), 6),
        }
    return (
        AdaptiveDraftTemperatureController(
            base_temperature=base, config=config
        ),
        {},
    )
