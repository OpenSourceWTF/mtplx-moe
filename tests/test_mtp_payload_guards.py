"""MTP payload guards reject stray-key checkpoints (Tier-1 audit findings).

An appended-layer checkpoint whose weight map is non-empty but carries no
real MTP tensors previously passed the `if not mapped:` completeness check
and injected a headless draft surface; each backend guard must demand its
layer's actual marker tensors, per declared layer count.
"""
from __future__ import annotations

def test_deepseek_payload_guard_rejects_stray_keys() -> None:
    from mtplx.deepseek_mtp_patch import _has_complete_deepseek_mtp_payload

    complete = {
        "layers.0.enorm.weight": 1,
        "layers.0.hnorm.weight": 1,
        "layers.0.eh_proj.weight": 1,
        "layers.0.mtp_block.self_attn.q_proj.weight": 1,
    }
    assert _has_complete_deepseek_mtp_payload(complete, num_mtp_layers=1)
    # Non-empty, but no real MTP tensors -- the old `if not mapped:` passed.
    assert not _has_complete_deepseek_mtp_payload(
        {"layers.0.something_else": 1}, num_mtp_layers=1
    )
    # Projections present but the draft block missing.
    missing_block = {k: v for k, v in complete.items() if "mtp_block" not in k}
    assert not _has_complete_deepseek_mtp_payload(missing_block, num_mtp_layers=1)
    # Declared two layers, only one supplied.
    assert not _has_complete_deepseek_mtp_payload(complete, num_mtp_layers=2)


def test_mimo_payload_guard_rejects_stray_keys() -> None:
    from mtplx.mimo_mtp_patch import _has_complete_mimo_mtp_payload

    complete = {
        "layers.0.token_layernorm.weight": 1,
        "layers.0.hidden_layernorm.weight": 1,
        "layers.0.input_proj.weight": 1,
        "layers.0.final_layernorm.weight": 1,
        "layers.0.mtp_block.self_attn.q_proj.weight": 1,
    }
    assert _has_complete_mimo_mtp_payload(complete, num_mtp_layers=1)
    assert not _has_complete_mimo_mtp_payload(
        {"lm_head.weight": 1}, num_mtp_layers=1
    )
    missing_block = {k: v for k, v in complete.items() if "mtp_block" not in k}
    assert not _has_complete_mimo_mtp_payload(missing_block, num_mtp_layers=1)


def test_nemotron_h_payload_guard_rejects_stray_keys() -> None:
    from mtplx.nemotron_h_mtp_patch import _has_complete_nemotron_h_mtp_payload

    complete = {
        "layers.0.norm.weight": 1,
        "layers.0.mixer.in_proj.weight": 1,
    }
    assert _has_complete_nemotron_h_mtp_payload(complete, physical_layers=1)
    assert not _has_complete_nemotron_h_mtp_payload(
        {"layers.0.block_type": 1}, physical_layers=1
    )
    assert not _has_complete_nemotron_h_mtp_payload(
        {"layers.0.norm.weight": 1}, physical_layers=1
    )
    assert not _has_complete_nemotron_h_mtp_payload(complete, physical_layers=2)


# --- Nemotron-H routing gate (issue #341) -----------------------------------
#
# Official NVIDIA configs (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
# describe the MTP stack as ``mtp_layers_block_type: ["attention", "moe"]``
# with no ``mtp_hybrid_override_pattern``. The gate previously ignored the
# block-type list, returned False, and the runtime fell through to the
# generic injector, which crashes on Nemotron-H
# (AttributeError: '_MTPLXTextModel' object has no attribute 'model').
# Char mapping is the mlx_lm nemotron_h convention:
# attention -> "*", moe -> "E", mamba -> "M", mlp -> "-".


def _official_nemotron_h_config() -> dict:
    """The reporter's exact official-config shape (no override pattern)."""

    return {
        "model_type": "nemotron_h",
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
    }


def test_nemotron_h_routes_official_block_type_list_config() -> None:
    from mtplx.nemotron_h_mtp_patch import _mtp_pattern, is_nemotron_h_mtp_config

    config = _official_nemotron_h_config()
    assert _mtp_pattern(config) == "*E"
    assert is_nemotron_h_mtp_config(config)


def test_nemotron_h_block_type_list_outranks_backbone_pattern() -> None:
    """The backbone hybrid_override_pattern (M/- chars) must not shadow the
    MTP block-type list; before the fix it produced a non-{*,E} pattern and
    the gate returned False."""

    from mtplx.nemotron_h_mtp_patch import _mtp_pattern, is_nemotron_h_mtp_config

    config = _official_nemotron_h_config()
    config["hybrid_override_pattern"] = "M-M-M*-M-M-M*-E"
    assert _mtp_pattern(config) == "*E"
    assert is_nemotron_h_mtp_config(config)


def test_nemotron_h_explicit_override_pattern_wins() -> None:
    from mtplx.nemotron_h_mtp_patch import _mtp_pattern, is_nemotron_h_mtp_config

    config = _official_nemotron_h_config()
    # Contradicting list order: the explicit override must win.
    config["mtp_layers_block_type"] = ["moe", "attention"]
    config["mtp_hybrid_override_pattern"] = "*E"
    assert _mtp_pattern(config) == "*E"
    assert is_nemotron_h_mtp_config(config)


def test_nemotron_h_unknown_block_types_stay_unrouted_without_crash() -> None:
    from mtplx.nemotron_h_mtp_patch import is_nemotron_h_mtp_config

    unknown = _official_nemotron_h_config()
    unknown["mtp_layers_block_type"] = ["attention", "linear_moe"]
    assert not is_nemotron_h_mtp_config(unknown)

    # Known chars outside the supported {*, E} MTP subset stay rejected too.
    mamba = _official_nemotron_h_config()
    mamba["mtp_layers_block_type"] = ["attention", "mamba"]
    assert not is_nemotron_h_mtp_config(mamba)

    # No pattern source at all: unchanged pre-fix behavior.
    assert not is_nemotron_h_mtp_config(
        {"model_type": "nemotron_h", "num_nextn_predict_layers": 1}
    )
