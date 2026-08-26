"""Acceptance-EMA adaptive draft-temperature controller (MTPLX_ADAPTIVE_DTEMP).

CPU-only: EMA/schedule/hysteresis logic on synthetic acceptance streams,
env-config parsing, and the no-op proof when the gate is off.  The synthetic
streams are named for the 2026-08-25 receipts they model (MEASUREMENTS.md
flat-decode marathon; schedule derivation in mtplx/adaptive_dtemp.py).
"""

from __future__ import annotations

import pytest

from mtplx.adaptive_dtemp import (
    ENV_ALPHA,
    ENV_BOOST,
    ENV_DROP,
    ENV_DWELL,
    ENV_FLOOR,
    ENV_GATE,
    ENV_RAISE,
    ENV_SEED_ROUNDS,
    TRANSITION_LOG_CAP,
    AdaptiveDraftTemperatureController,
    AdaptiveDtempConfig,
    adaptive_dtemp_enabled,
    build_adaptive_dtemp_controller,
)

BASE = 1.0  # the 08-25 serve arms drafted at base t1.0


def _controller(
    base: float = BASE, **overrides: float | int
) -> AdaptiveDraftTemperatureController:
    return AdaptiveDraftTemperatureController(
        base_temperature=base, config=AdaptiveDtempConfig(**overrides)
    )


def _feed(
    controller: AdaptiveDraftTemperatureController,
    values: list[float],
) -> list[float]:
    """Feed a stream; return the non-None transition temperatures in order."""

    fired: list[float] = []
    for value in values:
        new_temperature = controller.observe_round(value)
        if new_temperature is not None:
            fired.append(new_temperature)
    return fired


# ---------------------------------------------------------------------------
# gate + config parsing


def test_gate_off_by_default_and_truthy_forms():
    assert not adaptive_dtemp_enabled({})
    assert not adaptive_dtemp_enabled({ENV_GATE: ""})
    assert not adaptive_dtemp_enabled({ENV_GATE: "0"})
    assert not adaptive_dtemp_enabled({ENV_GATE: "off"})
    for truthy in ("1", "true", "YES", " on "):
        assert adaptive_dtemp_enabled({ENV_GATE: truthy})


def test_config_defaults_are_the_receipt_bands():
    config = AdaptiveDtempConfig()
    assert config.boost_temperature == 0.85
    assert config.raise_threshold == 0.78
    assert config.drop_threshold == 0.70
    assert config.floor_threshold == 0.45
    assert config.ema_alpha == 0.02
    assert config.seed_rounds == 12
    assert config.dwell_rounds == 24
    assert config.validation_error() is None


def test_config_from_env_overrides_and_garbage_falls_back():
    config = AdaptiveDtempConfig.from_env(
        {
            ENV_BOOST: "0.9",
            ENV_RAISE: "0.75",
            ENV_DROP: "0.6",
            ENV_FLOOR: "0.4",
            ENV_ALPHA: "not-a-float",  # falls back per-knob
            ENV_SEED_ROUNDS: "5",
            ENV_DWELL: "banana",  # falls back per-knob
        }
    )
    assert config.boost_temperature == 0.9
    assert config.raise_threshold == 0.75
    assert config.drop_threshold == 0.6
    assert config.floor_threshold == 0.4
    assert config.ema_alpha == AdaptiveDtempConfig().ema_alpha
    assert config.seed_rounds == 5
    assert config.dwell_rounds == AdaptiveDtempConfig().dwell_rounds
    assert config.validation_error() is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"boost_temperature": 0.0}, "boost_temperature_not_positive"),
        ({"boost_temperature": -0.85}, "boost_temperature_not_positive"),
        ({"ema_alpha": 0.0}, "ema_alpha_out_of_range"),
        ({"ema_alpha": 1.5}, "ema_alpha_out_of_range"),
        # drop < raise IS the hysteresis gap — an inverted pair cannot steer.
        ({"raise_threshold": 0.6, "drop_threshold": 0.7}, "thresholds_not_ordered"),
        ({"raise_threshold": 0.7, "drop_threshold": 0.7}, "thresholds_not_ordered"),
        ({"drop_threshold": 0.0}, "thresholds_not_ordered"),
        ({"raise_threshold": 1.0}, "thresholds_not_ordered"),
        ({"floor_threshold": 0.79}, "floor_above_raise"),
        ({"floor_threshold": -0.1}, "floor_above_raise"),
        ({"seed_rounds": 0}, "seed_rounds_below_one"),
        ({"dwell_rounds": -1}, "dwell_rounds_negative"),
    ],
)
def test_config_validation_errors(overrides, reason):
    assert AdaptiveDtempConfig(**overrides).validation_error() == reason


# ---------------------------------------------------------------------------
# build gate — the no-op proof when disabled, visible reasons when blocked


def test_build_is_a_noop_when_gate_off():
    controller, telemetry = build_adaptive_dtemp_controller(
        base_temperature=BASE, blockers=[], env={}
    )
    assert controller is None
    assert telemetry == {}  # empty {} => never stamped => byte-stable envelopes


def test_build_reports_lane_blockers_instead_of_running():
    controller, telemetry = build_adaptive_dtemp_controller(
        base_temperature=BASE,
        blockers=["draft_core:device", "constraint"],
        env={ENV_GATE: "1"},
    )
    assert controller is None
    assert telemetry["enabled"] is True
    assert telemetry["active"] is False
    assert telemetry["inactive_reasons"] == ["draft_core:device", "constraint"]


def test_build_refuses_greedy_base_and_base_equals_boost():
    controller, telemetry = build_adaptive_dtemp_controller(
        base_temperature=0.0, blockers=[], env={ENV_GATE: "1"}
    )
    assert controller is None
    assert telemetry["inactive_reasons"] == ["greedy_draft_base"]

    controller, telemetry = build_adaptive_dtemp_controller(
        base_temperature=0.85, blockers=[], env={ENV_GATE: "1"}
    )
    assert controller is None
    assert telemetry["inactive_reasons"] == ["base_equals_boost"]


def test_build_refuses_invalid_env_config_with_reason():
    controller, telemetry = build_adaptive_dtemp_controller(
        base_temperature=BASE,
        blockers=[],
        env={ENV_GATE: "1", ENV_RAISE: "0.6", ENV_DROP: "0.7"},
    )
    assert controller is None
    assert telemetry["inactive_reasons"] == [
        "invalid_config:thresholds_not_ordered"
    ]


def test_build_clean_returns_controller_at_base_set_point():
    controller, telemetry = build_adaptive_dtemp_controller(
        base_temperature=BASE, blockers=[], env={ENV_GATE: "1"}
    )
    assert controller is not None
    assert telemetry == {}
    assert controller.current_temperature == BASE
    assert controller.state == "seed"


# ---------------------------------------------------------------------------
# seeding


def test_seed_rounds_run_at_base_and_seed_the_ema_with_the_plain_mean():
    controller = _controller(seed_rounds=4, dwell_rounds=0, ema_alpha=0.5)
    for value in (0.9, 0.8, 0.9):  # 3 of 4 seed rounds
        assert controller.observe_round(value) is None
        assert controller.ema is None
        assert controller.current_temperature == BASE
        assert controller.state == "seed"
    # 4th observation completes the seed: EMA = plain mean = .85 (> raise
    # threshold => duplication register => stays at base).
    assert controller.observe_round(0.8) is None
    assert controller.ema == pytest.approx((0.9 + 0.8 + 0.9 + 0.8) / 4)
    assert controller.state == "default"
    assert controller.current_temperature == BASE
    assert controller.transitions == 0


def test_seed_completion_may_steer_immediately():
    controller = _controller(seed_rounds=3, dwell_rounds=8)
    fired = _feed(controller, [0.75, 0.75, 0.75])  # mean .75 in [.45, .78]
    assert fired == [0.85]
    assert controller.state == "boost"
    assert controller.observed_rounds == 3
    assert controller.transition_log == [(3, 0.85)]


def test_ema_recurrence_is_exact_after_seeding():
    controller = _controller(seed_rounds=1, dwell_rounds=10**6, ema_alpha=0.25)
    controller.observe_round(0.8)  # seed: ema = .8
    controller.observe_round(0.4)
    controller.observe_round(1.0)
    expected = 0.8
    expected = 0.75 * expected + 0.25 * 0.4
    expected = 0.75 * expected + 0.25 * 1.0
    assert controller.ema == pytest.approx(expected)


def test_observations_are_clamped_to_unit_interval():
    controller = _controller(seed_rounds=2, dwell_rounds=0)
    controller.observe_round(-3.0)
    controller.observe_round(7.0)
    assert controller.ema == pytest.approx(0.5)  # mean of clamped 0.0, 1.0


# ---------------------------------------------------------------------------
# the schedule on receipt-shaped streams


def test_duplication_register_stays_at_base():
    # synthetic-120k receipt: base pos-1 .802 — the register where 0.85 cost
    # -11%. The controller must hold the base set-point throughout.
    controller = _controller(seed_rounds=8, dwell_rounds=4, ema_alpha=0.1)
    fired = _feed(controller, [0.80, 0.82] * 30)
    assert fired == []
    assert controller.state == "default"
    assert controller.current_temperature == BASE
    assert controller.transitions == 0


def test_natural_register_boosts_and_holds():
    # natural-184.6k receipt: base pos-1 .750 => raise; under 0.85 the ladder
    # reads .889 => the boost state must HOLD (the state-dependent drop
    # threshold is what prevents the naive high-band oscillation).
    controller = _controller(seed_rounds=8, dwell_rounds=4, ema_alpha=0.1)
    fired = _feed(controller, [0.75] * 8)
    assert fired == [0.85]
    fired = _feed(controller, [0.89] * 40)
    assert fired == []
    assert controller.state == "boost"
    assert controller.current_temperature == 0.85
    assert controller.transitions == 1
    assert controller.boost_rounds == 40


def test_hurt_case_drops_back_and_does_not_rearm():
    # A mid-band seed raises; the boost arm then reads the 120k hurt
    # signature (.60-.68 band, receipt .683) => drop back to base after the
    # dwell; the default arm then reads .82 (> raise) => no re-raise.
    controller = _controller(seed_rounds=8, dwell_rounds=4, ema_alpha=0.3)
    fired = _feed(controller, [0.77] * 8)
    assert fired == [0.85]
    fired = _feed(controller, [0.60] * 10)
    assert fired == [BASE]
    assert controller.state == "default"
    fired = _feed(controller, [0.82] * 30)
    assert fired == []
    assert controller.transitions == 2
    assert controller.current_temperature == BASE


def test_low_band_floor_never_raises():
    # No receipts below ~.45 base pos-1 — unmeasured territory holds base.
    controller = _controller(seed_rounds=8, dwell_rounds=0, ema_alpha=0.1)
    fired = _feed(controller, [0.30] * 60)
    assert fired == []
    assert controller.state == "default"
    assert controller.current_temperature == BASE


def test_band_edges_raise_inclusive_drop_exclusive():
    # alpha=1 makes the EMA equal the last observation — exact edge probes.
    controller = _controller(seed_rounds=1, dwell_rounds=0, ema_alpha=1.0)
    assert controller.observe_round(0.78) == 0.85  # ema == raise => fires
    assert controller.observe_round(0.70) is None  # ema == drop => holds
    assert controller.observe_round(0.699) == BASE  # ema < drop => drops

    floor_edge = _controller(seed_rounds=1, dwell_rounds=0, ema_alpha=1.0)
    assert floor_edge.observe_round(0.45) == 0.85  # ema == floor => fires

    below_floor = _controller(seed_rounds=1, dwell_rounds=0, ema_alpha=1.0)
    assert below_floor.observe_round(0.4499) is None  # below floor => holds


def test_rearm_after_recovery_allows_a_genuine_register_change():
    # After a drop the raise trigger is DISARMED (do not re-boost into the
    # register that just hurt); it re-arms only once the default-state EMA
    # recovers ABOVE the raise threshold, so a LATER sag into the band is a
    # genuine register change and raises legitimately.
    controller = _controller(seed_rounds=4, dwell_rounds=2, ema_alpha=0.5)
    assert _feed(controller, [0.75] * 4) == [0.85]  # natural seed => boost
    assert _feed(controller, [0.55] * 6) == [BASE]  # hurt => drop + disarm
    assert controller.raise_armed is False
    # Mid-band forever while disarmed: the post-drop EMA transiting the
    # raise band must NOT re-boost (the oscillation guard).
    assert _feed(controller, [0.72] * 40) == []
    assert _feed(controller, [0.90] * 6) == []  # recovers > .78 => re-arms
    assert controller.raise_armed is True
    assert _feed(controller, [0.72] * 40) == [0.85]  # genuine sag => raise
    assert controller.transitions == 3


# ---------------------------------------------------------------------------
# hysteresis: dwell floor caps the transition rate


def test_dwell_blocks_transitions_after_a_flip():
    controller = _controller(seed_rounds=1, dwell_rounds=5, ema_alpha=1.0)
    assert controller.observe_round(0.75) == 0.85  # seed => raise
    # EMA sits decisively below the drop edge every round, but the dwell
    # floor holds the set-point for 4 more observed rounds.
    for _ in range(4):
        assert controller.observe_round(0.50) is None
    assert controller.observe_round(0.50) == BASE  # 5th round => drop fires
    assert controller.transitions == 2
    assert controller.observed_rounds == 6


def test_dwell_counts_observed_rounds_not_wall_time():
    controller = _controller(seed_rounds=1, dwell_rounds=3, ema_alpha=1.0)
    assert controller.observe_round(0.75) == 0.85
    assert controller.observe_round(0.5) is None
    assert controller.observe_round(0.5) is None
    assert controller.observe_round(0.5) == BASE


def test_transition_log_caps_but_count_keeps_counting():
    controller = _controller(seed_rounds=1, dwell_rounds=0, ema_alpha=1.0)
    controller.observe_round(0.75)  # seed => raise
    # alpha=1, dwell=0: drop (.60 < .70), recover above raise (.79 re-arms),
    # sag into the band (.75 raises) — two transitions per iteration.
    for _ in range(TRANSITION_LOG_CAP + 20):
        controller.observe_round(0.60)
        controller.observe_round(0.79)
        controller.observe_round(0.75)
    assert controller.transitions > TRANSITION_LOG_CAP
    assert len(controller.transition_log) == TRANSITION_LOG_CAP


# ---------------------------------------------------------------------------
# telemetry summary


def test_summary_carries_the_trajectory_fields():
    controller = _controller(seed_rounds=2, dwell_rounds=1, ema_alpha=0.5)
    _feed(controller, [0.75, 0.75, 0.89, 0.89])
    summary = controller.summary()
    assert summary["enabled"] is True
    assert summary["active"] is True
    assert summary["state"] == "boost"
    assert summary["base_temperature"] == BASE
    assert summary["boost_temperature"] == 0.85
    assert summary["current_temperature"] == 0.85
    assert summary["observed_rounds"] == 4
    assert summary["boost_rounds"] == 2
    assert summary["transitions"] == 1
    assert summary["raise_armed"] is True
    assert summary["transition_log"] == [[2, 0.85]]
    assert summary["ema"] == pytest.approx(
        0.5 * (0.5 * 0.75 + 0.5 * 0.89) + 0.5 * 0.89, abs=1e-4
    )
    assert summary["config"]["raise_threshold"] == 0.78
    assert summary["config"]["drop_threshold"] == 0.70
    assert summary["config"]["seed_rounds"] == 2


# ---------------------------------------------------------------------------
# engine/server integration surfaces (CPU only — no model, no GPU)


def test_sampler_config_rebind_matches_engine_pattern():
    # generate_mtpk rebinds its loop-local draft sampler via dataclasses
    # .replace on a transition; pin that SamplerConfig supports it and that
    # only the temperature moves.
    from dataclasses import replace

    from mtplx.sampling import SamplerConfig

    sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    rebound = replace(sampler, temperature=0.85)
    assert rebound.temperature == 0.85
    assert rebound.top_p == sampler.top_p
    assert rebound.top_k == sampler.top_k
    assert sampler.temperature == 1.0  # frozen original untouched


def test_generation_stats_field_defaults_empty_and_serializes():
    from mtplx.generation import GenerationStats

    stats = GenerationStats(
        mode="mtpk", generated_tokens=0, elapsed_s=0.0, tok_s=0.0
    )
    assert stats.draft_sampler_adaptive_dtemp == {}
    assert stats.to_dict()["draft_sampler_adaptive_dtemp"] == {}


def test_public_stats_stamp_is_quiet_when_empty_and_present_when_not():
    from mtplx.server.openai import _public_mtplx_stats

    quiet = _public_mtplx_stats({"stats": {"draft_sampler_adaptive_dtemp": {}}})
    assert "draft_sampler_adaptive_dtemp" not in quiet

    summary = {"enabled": True, "active": True, "transitions": 1}
    loud = _public_mtplx_stats(
        {"stats": {"draft_sampler_adaptive_dtemp": summary}}
    )
    assert loud["draft_sampler_adaptive_dtemp"] == summary
