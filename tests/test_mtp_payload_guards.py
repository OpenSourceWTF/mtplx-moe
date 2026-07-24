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


