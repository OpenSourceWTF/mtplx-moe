from __future__ import annotations

from types import SimpleNamespace

from mtplx import default_models
from mtplx.commands import public

HY3_STREAMING_REPO_ID = "OpensourceWTF/Hy3-oQ2e-MTPLX-streaming"


def _args(**overrides):
    """A minimal args namespace for the auto-detect / rewrite helpers."""

    base = {
        "expert_streaming": False,
        "expert_manifest": None,
        "expert_streaming_config": None,
        "_cli_flags": set(),
        "model": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# CHANGE 1 -- auto-detect streaming from a resident baked layout
# ---------------------------------------------------------------------------


def test_auto_enables_streaming_when_artifact_status_ok(tmp_path, monkeypatch):
    """(a) A valid baked streaming layout flips expert_streaming on."""

    monkeypatch.setattr(
        "mtplx.hf_loader.expert_artifact_status",
        lambda path: {"ok": True, "streamed_experts": True, "reason": None},
    )
    args = _args()
    flipped = public._maybe_enable_expert_streaming(args, str(tmp_path))
    assert flipped is True
    assert args.expert_streaming is True


def test_does_not_enable_when_status_not_ok(tmp_path, monkeypatch):
    """(b) A stray/partial manifest (ok False) is left as a normal load."""

    monkeypatch.setattr(
        "mtplx.hf_loader.expert_artifact_status",
        lambda path: {
            "ok": False,
            "streamed_experts": True,
            "reason": "expert bank is truncated",
        },
    )
    args = _args()
    flipped = public._maybe_enable_expert_streaming(args, str(tmp_path))
    assert flipped is False
    assert args.expert_streaming is False


def test_plain_model_dir_stays_normal_load(tmp_path):
    """(b) A dir with no manifest -- real expert_artifact_status -- stays off.

    Guards against the trap that ``status["ok"]`` is True for ordinary models,
    so gating on ``ok`` alone would flip every normal load into streaming.
    """

    args = _args()
    flipped = public._maybe_enable_expert_streaming(args, str(tmp_path))
    assert flipped is False
    assert args.expert_streaming is False


def test_explicit_streaming_flag_not_overridden(tmp_path, monkeypatch):
    """(c) A user who already asked for streaming is left as-is."""

    monkeypatch.setattr(
        "mtplx.hf_loader.expert_artifact_status",
        lambda path: {"ok": True, "streamed_experts": True, "reason": None},
    )
    args = _args(expert_streaming=True)
    flipped = public._maybe_enable_expert_streaming(args, str(tmp_path))
    assert flipped is False
    assert args.expert_streaming is True


def test_explicit_opt_out_not_overridden(tmp_path, monkeypatch):
    """(c) An explicit opt-out recorded in _cli_flags is honored over the layout."""

    monkeypatch.setattr(
        "mtplx.hf_loader.expert_artifact_status",
        lambda path: {"ok": True, "streamed_experts": True, "reason": None},
    )
    args = _args(expert_streaming=False, _cli_flags={"expert_streaming"})
    flipped = public._maybe_enable_expert_streaming(args, str(tmp_path))
    assert flipped is False
    assert args.expert_streaming is False


def test_missing_model_path_is_safe(monkeypatch):
    """A missing/empty resolved path never flips streaming on."""

    monkeypatch.setattr(
        "mtplx.hf_loader.expert_artifact_status",
        lambda path: {"ok": True, "streamed_experts": True, "reason": None},
    )
    args = _args()
    assert public._maybe_enable_expert_streaming(args, None) is False
    assert args.expert_streaming is False


# ---------------------------------------------------------------------------
# CHANGE 2 -- published streaming alias/id -> HF repo id rewrite
# ---------------------------------------------------------------------------


def test_catalog_model_matching_resolves_hy3_alias():
    """(d) The short alias maps to the OpensourceWTF Hy3 streaming repo id."""

    matched = default_models.catalog_model_matching("hy3-oq2e")
    assert matched is not None
    assert matched.hf_model_id == HY3_STREAMING_REPO_ID


def test_rewrite_maps_alias_to_hf_model_id():
    """(d) serve ref-rewrite maps the alias to the entry's hf_model_id."""

    args = _args(model="hy3-oq2e")
    rewritten = public._maybe_rewrite_streaming_model_ref(args)
    assert rewritten == HY3_STREAMING_REPO_ID
    assert args.model == HY3_STREAMING_REPO_ID


def test_rewrite_is_noop_for_full_repo_id():
    """The full repo id is already resolvable; the ref is left unchanged."""

    args = _args(model=HY3_STREAMING_REPO_ID)
    rewritten = public._maybe_rewrite_streaming_model_ref(args)
    assert rewritten is None
    assert args.model == HY3_STREAMING_REPO_ID


def test_rewrite_ignores_non_streaming_ref():
    """A non-streaming model ref is never rewritten."""

    args = _args(model="some/unrelated-non-streaming-model")
    rewritten = public._maybe_rewrite_streaming_model_ref(args)
    assert rewritten is None
    assert args.model == "some/unrelated-non-streaming-model"


def test_rewrite_leaves_existing_local_path_alone(tmp_path):
    """A concrete local path wins over any alias/basename match."""

    local = tmp_path / "OpensourceWTF--Hy3-oQ2e-MTPLX-streaming"
    local.mkdir()
    args = _args(model=str(local))
    rewritten = public._maybe_rewrite_streaming_model_ref(args)
    assert rewritten is None
    assert args.model == str(local)
