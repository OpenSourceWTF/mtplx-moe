"""CLI-lane greedy draft coupling (_greedy_coupled_draft_spec).

The daemon resolver couples the draft sampler to a greedy target on the
server path (test_greedy_draft_coupling.py); the CLI lanes that call
generate_mtpk directly (one-shot generate, quickstart terminal chat, tune
candidate) bypass it. A stamped sampled draft under a greedy target
collapses acceptance by depth ([79/65/42]% vs [96/87/76]% coupled on the
coding suite), so those lanes apply _greedy_coupled_draft_spec before
handing the spec to the engine.
"""

from types import SimpleNamespace

from mtplx.commands.public import (
    _draft_sampler_from_spec,
    _greedy_coupled_draft_spec,
)

STAMPED = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def _args(cli_flags=(), injected=()):
    return SimpleNamespace(
        _cli_flags=set(cli_flags),
        _injected_default_flags=set(injected),
    )


def test_none_spec_passes_through():
    # None already mirrors the target sampler downstream
    # (_effective_draft_sampler), so coupling must not invent a spec.
    assert _greedy_coupled_draft_spec(None, _args(), 0.0) is None


def test_sampled_target_untouched():
    out = _greedy_coupled_draft_spec(dict(STAMPED), _args(), 0.6)
    assert out == STAMPED


def test_greedy_target_couples_stamped_draft():
    spec = dict(STAMPED)
    out = _greedy_coupled_draft_spec(spec, _args(), 0.0)
    assert out == {"temperature": 0.0, "top_p": 0.95, "top_k": 20}
    # The stamped spec itself must stay intact for reporting surfaces.
    assert spec == STAMPED


def test_user_typed_draft_temperature_wins():
    out = _greedy_coupled_draft_spec(
        dict(STAMPED), _args(cli_flags={"draft-temperature"}), 0.0
    )
    assert out == STAMPED


def test_injected_default_flag_does_not_block_coupling():
    # Only a user-typed flag is an override; a default injected by the
    # _apply_*_defaults helpers carries no user intent.
    out = _greedy_coupled_draft_spec(
        dict(STAMPED), _args(injected={"draft-temperature"}), 0.0
    )
    assert out["temperature"] == 0.0


def test_already_greedy_draft_untouched():
    spec = {"temperature": 0.0, "top_p": 0.9, "top_k": 10}
    assert _greedy_coupled_draft_spec(dict(spec), _args(), 0.0) == spec


def test_args_without_flag_tracking_still_couple():
    # Lanes whose args never went through flag tracking (bare Namespace)
    # must not crash and must still couple.
    out = _greedy_coupled_draft_spec(dict(STAMPED), SimpleNamespace(), 0.0)
    assert out["temperature"] == 0.0


def test_coupled_spec_converts_to_sampler_config():
    coupled = _greedy_coupled_draft_spec(dict(STAMPED), _args(), 0.0)
    cfg = _draft_sampler_from_spec(coupled)
    assert cfg.temperature == 0.0
    assert cfg.top_p == 0.95
    assert cfg.top_k == 20
