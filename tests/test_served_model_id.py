"""The served model identity must describe the model that was loaded.

``--model-id`` is what ``/v1/models``, ``/health``, and every completion report.
It used to default to one specific model's name, so serving anything else
advertised the wrong identity. Clients select prompt formats and tool modes by
matching on this string, so a stale value silently changes client behaviour.
"""

from __future__ import annotations

import pytest

from mtplx.profiles import DEFAULT_HF_MODEL_ID
from mtplx.server.openai import DEFAULT_SERVED_MODEL_ID, _derive_served_model_id


@pytest.mark.parametrize("model_ref", [None, "", "   ", DEFAULT_HF_MODEL_ID])
def test_stock_model_keeps_its_established_id(model_ref):
    """Existing clients pointed at the stock model must not see a rename."""

    assert _derive_served_model_id(model_ref) == DEFAULT_SERVED_MODEL_ID


@pytest.mark.parametrize(
    ("model_ref", "expected"),
    [
        ("OpensourceWTF/Hy3-oQ2e-MTPLX-streaming", "hy3-oq2e-mtplx-streaming"),
        ("OpensourceWTF/GLM-5.2-t158-MTPLX-streaming", "glm-5-2-t158-mtplx-streaming"),
        (
            "/Users/x/.mtplx/models/GrEarl--Kimi-K3-Q2_K-t158-MTPLX-streaming",
            "kimi-k3-q2-k-t158-mtplx-streaming",
        ),
    ],
)
def test_other_models_are_named_after_themselves(model_ref, expected):
    assert _derive_served_model_id(model_ref) == expected


def test_repo_id_and_local_snapshot_agree():
    """A model keeps one identity whether served from the hub or from disk."""

    remote = _derive_served_model_id("OpensourceWTF/Hy3-oQ2e-MTPLX-streaming")
    local = _derive_served_model_id(
        "/Users/x/.mtplx/models/OpensourceWTF--Hy3-oQ2e-MTPLX-streaming"
    )
    assert remote == local == "hy3-oq2e-mtplx-streaming"


def test_no_model_is_advertised_as_the_stock_qwen_by_accident():
    """The regression itself: Hy3 was reported as the stock Qwen id."""

    assert (
        _derive_served_model_id("OpensourceWTF/Hy3-oQ2e-MTPLX-streaming")
        != DEFAULT_SERVED_MODEL_ID
    )


@pytest.mark.parametrize(
    "model_ref",
    ["org/Repo/", "/tmp/models/Repo//", "Repo"],
)
def test_trailing_separators_do_not_leak_into_the_id(model_ref):
    slug = _derive_served_model_id(model_ref)
    assert slug == "repo"


def test_ids_are_client_safe_slugs():
    """Lowercase, no path separators, no characters needing URL escaping."""

    for ref in (
        "OpensourceWTF/Hy3-oQ2e-MTPLX-streaming",
        "OpensourceWTF/GLM-5.2-t158-MTPLX-streaming",
        "/Users/x/.mtplx/models/GrEarl--Kimi-K3-Q2_K-t158-MTPLX-streaming",
    ):
        slug = _derive_served_model_id(ref)
        assert slug == slug.lower()
        assert "/" not in slug and " " not in slug
        assert not slug.startswith("-") and not slug.endswith("-")
