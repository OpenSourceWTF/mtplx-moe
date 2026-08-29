"""Greedy target ⇒ greedy draft coupling (speed-only, output-invariant)."""

from __future__ import annotations

from mtplx.sampling import SamplerConfig
from mtplx.server.openai import _couple_draft_sampler_to_greedy_target


def _launch_draft():
    return SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)


def test_couples_at_temp0():
    obs: dict = {}
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=0.0,
        request_observability=obs,
    )
    assert out.temperature == 0.0
    assert out.top_p == 0.95 and out.top_k == 20  # only greediness changes
    assert obs["draft_sampler_greedy_coupled"] is True


def test_untouched_at_sampled_target():
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=0.6,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_explicit_draft_sampler_wins():
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=True,
        target_temperature=0.0,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_none_target_temperature_untouched():
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=None,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_env_off_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_COUPLING", "off")
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=0.0,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_already_greedy_draft_passthrough():
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    obs: dict = {}
    out = _couple_draft_sampler_to_greedy_target(
        greedy,
        explicit_draft_sampler=False,
        target_temperature=0.0,
        request_observability=obs,
    )
    assert out is greedy
    assert "draft_sampler_greedy_coupled" not in obs


def _stub_state(draft_temperature: float = 1.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        draft_sampler=SamplerConfig(
            temperature=draft_temperature, top_p=0.95, top_k=20
        ),
        draft_sampler_pinned=False,
        draft_temperature_curve=None,
    )


def test_resolver_couples_greedy_launch_default_regression():
    """Call sites pass the EFFECTIVE target temperature, never the raw field.

    Regression: a server launched greedy (--temperature 0) serving a request
    that omits `temperature` decoded greedily while the family draft stayed
    at 1.0 — the resolver saw target_temperature=None and skipped coupling
    (silent sampled-draft collapse, [79/65/42]% vs [96/87/76]% by depth).
    Both serve call sites now derive target_temperature from the resolved
    sampler, so the resolver must couple whenever that value is 0.
    """
    from mtplx.server.openai import _resolve_draft_sampler_for_request

    obs: dict = {}
    out = _resolve_draft_sampler_for_request(
        _stub_state(draft_temperature=1.0),
        request_draft_sampler=None,
        target_temperature=0.0,  # what the fixed call sites now pass
        target_sampler=SamplerConfig(temperature=0.0, top_p=0.95, top_k=20),
        request_observability=obs,
    )
    assert out.temperature == 0.0
    assert obs["draft_sampler_policy_source"] == "family_default+greedy_coupled"
    assert obs["draft_sampler_resolved_temperature"] == 0.0
