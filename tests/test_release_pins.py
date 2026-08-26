"""Release-pin regression guards.

Added for 2.9.3 (founder directive, 2026-08-26) against two burn classes:

1. "We weren't accidentally running without turbo" — a shipped profile
   silently losing a fast-lane knob (the compiled-verify ceiling, the packed
   verify kernel, the dense-decode ceiling), or a profile stomping an
   operator's exported env back to the profile value. Both have happened:
   the 2026-07-17 compiled-verify sweep needed a site-packages patch because
   the profile stomped the env, and the first 2026-08-26 cliff-fix arm was a
   NULL because the profile stomped MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT.
   These tests pin the shipped values and the precedence so a regression is
   a test failure, not a silent lane loss found weeks later in a benchmark.

2. mlx pin drift — the wheel's floor carries receipts (0.32.0 -> 0.32.2 =
   +29% decode / +41% prefill @88.4k, clean shipped-wheel A/B, MEASUREMENTS
   2026-08-26 07:20) that an installed venv silently forfeits by sitting on
   an older mlx (the app bootstrapper's -U uses pip's only-if-needed
   strategy, so old venvs never converge on their own).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mtplx.profiles import apply_profile_env, get_profile

CEILING = "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"

# The product profiles users actually run. "stable"/"exact" deliberately do
# not carry the sustained fast-path env and are not pinned here.
PRODUCT_PROFILES = ("turbo", "sustained")


# ---------------------------------------------------------------------------
# 1a. Shipped fast-lane values stay shipped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", PRODUCT_PROFILES)
def test_product_profiles_ship_the_auto_dense_ceiling(profile):
    """2.9.3 flipped the dense-decode ceiling literal (131072) to "auto".

    The literal was the 147.4k decode cliff: past 2^17 tokens decode repaged
    and the packed fast-SDPA lane was structurally excluded (12.0 -> 18.44
    tok/s once dense decode holds, MEASUREMENTS 2026-08-26 07:58/08:24).
    Reverting this to a number is a founder-gated decision, not a refactor.
    """
    assert get_profile(profile).env_dict()[CEILING] == "auto"


def test_turbo_ships_the_32k_compiled_verify_ceiling():
    """12288 -> 32768 landed 2026-08-14 (dropday verify-wall calibration,
    ABBA-receipted: compiled-at-32k beat eager at 20k and 30k). Losing this
    silently costs every 6k-32k request the compiled verify bank."""
    assert get_profile("turbo").env_dict()["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"] == "32768"


def test_turbo_ships_the_packed_verify_kernel():
    """The packed GQA verify kernel is the decode wedge; turbo without it is
    the "we weren't running turbo" failure mode."""
    assert get_profile("turbo").env_dict()["MTPLX_GQA_PACKED_SDPA"] == "1"


# ---------------------------------------------------------------------------
# 1b. Operator env beats the profile for the ceiling key (the stomp class).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", PRODUCT_PROFILES)
def test_operator_dense_ceiling_export_survives_profile_apply(profile):
    env = {CEILING: "196608"}
    apply_profile_env(profile, environ=env)
    assert env[CEILING] == "196608", (
        "profile stomped an exported dense-decode ceiling — this exact stomp "
        "nulled the first cliff-fix arm on 2026-08-26"
    )


def test_operator_compiled_verify_ceiling_survives_profile_apply():
    env = {"MTPLX_COMPILED_VERIFY_MAX_CONTEXT": "16384"}
    apply_profile_env("turbo", environ=env)
    assert env["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"] == "16384"


# ---------------------------------------------------------------------------
# 2. The installed mlx honors the pyproject floor.
# ---------------------------------------------------------------------------


def _pyproject_mlx_floor() -> tuple[int, ...]:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    match = re.search(r'"mlx>=([0-9.]+),<', text)
    assert match, "pyproject no longer declares an mlx floor requirement"
    return tuple(int(part) for part in match.group(1).split("."))


def test_pyproject_mlx_floor_is_the_receipted_version():
    """The floor itself is a release decision with measurements behind it;
    lowering it must be deliberate, not a merge accident."""
    assert _pyproject_mlx_floor() >= (0, 32, 2)


def test_installed_mlx_meets_the_pyproject_floor():
    """Catches venv drift: an environment (the app runtime included) running
    an mlx older than the wheel's floor forfeits the receipts the floor was
    raised for, silently."""
    mlx = pytest.importorskip("mlx.core")
    installed = tuple(
        int(part) for part in re.match(r"(\d+)\.(\d+)\.(\d+)", mlx.__version__).groups()
    )
    assert installed >= _pyproject_mlx_floor(), (
        f"installed mlx {mlx.__version__} is below the pyproject floor "
        f"{'.'.join(map(str, _pyproject_mlx_floor()))} — this venv is running "
        "a stack the release's receipts do not cover"
    )
