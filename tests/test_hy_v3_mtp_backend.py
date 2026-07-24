"""Regression tests for the hy_v3 MTP backend (audit-driven).

mlx-lm ships models/hy_v3.py on main but not in any release yet (latest is
0.31.3, checked 2026-07-11); the backend is inert until it lands, so these
tests skip rather than break collection on released mlx-lm.
"""
import pytest

hy_v3 = pytest.importorskip(
    "mlx_lm.models.hy_v3",
    reason="mlx-lm does not ship models/hy_v3 yet (unreleased upstream)",
)

import mlx.core as mx
from pathlib import Path
from mtplx.hy_v3_mtp_patch import inject_hy_v3_mtp_support, is_hy_v3_mtp_config
from mtplx.mtp_patch import validate_mtp_support, MTPContract
from mtplx.backends.registry import SUPPORTED_ARCH_IDS


def _tiny():
    args = hy_v3.ModelArgs(
        model_type="hy_v3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, expert_hidden_dim=64,
        first_k_dense_replace=1, rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=1)
    return hy_v3.Model(args)


def test_hy_v3_in_supported_arch_ids():
    assert "hy-v3-mtp" in SUPPORTED_ARCH_IDS


def _checkpoint_dir(tmp_path):
    """Tiny standard checkpoint carrying the appended NextN layer (see
    test_hy_v3_mtp_graft.py for the full graft matrix)."""
    import json
    from mlx.utils import tree_flatten

    args = _tiny().args
    donor = hy_v3.DecoderLayer(args, layer_idx=2)
    tensors = {f"model.layers.2.{k}": v for k, v in tree_flatten(donor.parameters())}
    for name in ("enorm", "hnorm", "final_layernorm"):
        tensors[f"model.layers.2.{name}.weight"] = mx.ones((64,))
    tensors["model.layers.2.eh_proj.weight"] = 0.02 * mx.random.normal((64, 128))
    mx.save_safetensors(str(tmp_path / "model-mtp.safetensors"), tensors)
    json.dump({"metadata": {}, "weight_map": {k: "model-mtp.safetensors" for k in tensors}},
              open(tmp_path / "model.safetensors.index.json", "w"))
    return tmp_path


def _cfg():
    return {
        "model_type": "hy_v3", "num_nextn_predict_layers": 1,
        "num_hidden_layers": 2, "vocab_size": 128, "hidden_size": 64,
        "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "num_experts": 4,
        "num_experts_per_tok": 2, "num_shared_experts": 1,
        "expert_hidden_dim": 64, "first_k_dense_replace": 1,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    }


def test_inject_and_validate(tmp_path):
    m = _tiny()
    cfg = _cfg()
    assert is_hy_v3_mtp_config(cfg)
    assert inject_hy_v3_mtp_support(m, _checkpoint_dir(tmp_path), cfg, None)
    # validate_mtp_support needs model.mtp.layers -> alias must exist
    assert validate_mtp_support(m)


def test_post_norm_contract_default_does_not_crash(tmp_path):
    m = _tiny()
    inject_hy_v3_mtp_support(m, _checkpoint_dir(tmp_path), _cfg(), None)
    x = mx.array([[1, 2, 3, 4]])
    # the bare-contract default is post_norm; the backend must tolerate it
    assert MTPContract().hidden_variant == "post_norm"
    logits, hidden = m(x, return_hidden=True, hidden_variant="post_norm")
    assert logits.shape == (1, 4, 128) and hidden.shape == (1, 4, 64)
    d = m.mtp_forward(hidden[:, -1:, :], mx.array([[5]]),
                      mtp_hidden_variant="post_norm", concat_order="embedding_hidden")
    assert d.shape == (1, 1, 128)


def test_ar_only_export_raises_clearly():
    args = hy_v3.ModelArgs(
        model_type="hy_v3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, expert_hidden_dim=64,
        first_k_dense_replace=1, rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=0)
    m = hy_v3.Model(args)  # no mtp submodule
    try:
        inject_hy_v3_mtp_support(m, Path("t"), {"model_type": "hy_v3", "num_nextn_predict_layers": 1}, None)
    except RuntimeError as e:
        assert "no MTP submodule" in str(e) or "AR-only" in str(e)
    else:
        raise AssertionError("expected RuntimeError on AR-only export")
