"""forward_ar engages the compiled AR path when flagged (wiring test)."""
import json

import pytest

hy_v3 = pytest.importorskip(
    "mlx_lm.models.hy_v3",
    reason="mlx-lm does not ship models/hy_v3 yet (unreleased upstream)",
)

import mlx.core as mx
from pathlib import Path

from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime


def _tiny_model():
    args = hy_v3.ModelArgs(
        model_type="hy_v3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, expert_hidden_dim=64,
        first_k_dense_replace=1, rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=0)
    return hy_v3.Model(args)


def _runtime(model):
    return MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("t"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _primed_cache(model):
    from mlx_lm.models.cache import KVCache

    cache = [KVCache() for _ in model.model.layers]
    _ = model(mx.array([[1, 2, 3, 4]]), cache=cache)
    mx.eval(cache[0].keys)
    return cache


def test_compiled_path_engages_and_matches_eager(monkeypatch):
    model = _tiny_model()
    rt = _runtime(model)
    cache = _primed_cache(model)

    monkeypatch.delenv("MTPLX_COMPILE_AR_FORWARD", raising=False)
    eager = rt.forward_ar(mx.array([[5]]), cache=cache)
    mx.eval(eager)
    assert rt.diagnostic_counters.get("compiled_forward_calls", 0) == 0

    model2 = _tiny_model()
    model2.update(model.parameters())
    rt2 = _runtime(model2)
    cache2 = _primed_cache(model2)
    monkeypatch.setenv("MTPLX_COMPILE_AR_FORWARD", "1")
    compiled = rt2.forward_ar(mx.array([[5]]), cache=cache2)
    mx.eval(compiled)
    assert rt2.diagnostic_counters.get("compiled_forward_calls", 0) == 1, (
        "compiled AR path did not engage"
    )
    assert mx.allclose(
        eager.astype(mx.float32), compiled.astype(mx.float32), atol=1e-4
    ).item()
    # steps advance through the compiled path and stay engaged
    again = rt2.forward_ar(mx.array([[6]]), cache=cache2)
    mx.eval(again)
    assert rt2.diagnostic_counters["compiled_forward_calls"] == 2


def _grafted_runtime(tmp_path):
    """MTP-wrapped runtime (the live serving shape that missed engagement)."""
    from mlx.utils import tree_flatten

    from mtplx.hy_v3_mtp_patch import inject_hy_v3_mtp_support

    args = hy_v3.ModelArgs(
        model_type="hy_v3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, expert_hidden_dim=64,
        first_k_dense_replace=1, rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=1)
    model = hy_v3.Model(args)
    donor = hy_v3.DecoderLayer(args, layer_idx=2)
    tensors = {f"model.layers.2.{k}": v for k, v in tree_flatten(donor.parameters())}
    for name in ("enorm", "hnorm", "final_layernorm"):
        tensors[f"model.layers.2.{name}.weight"] = mx.ones((64,))
    tensors["model.layers.2.eh_proj.weight"] = 0.02 * mx.random.normal((64, 128))
    mx.save_safetensors(str(tmp_path / "model-mtp.safetensors"), tensors)
    json.dump({"metadata": {}, "weight_map": {k: "model-mtp.safetensors" for k in tensors}},
              open(tmp_path / "model.safetensors.index.json", "w"))
    cfg = {"model_type": "hy_v3", "num_nextn_predict_layers": 1, "num_hidden_layers": 2}
    assert inject_hy_v3_mtp_support(model, tmp_path, cfg, None)
    rt = MTPLXRuntime(
        model=model, tokenizer=None, model_path=tmp_path,
        mtp_enabled=True, contract=MTPContract(),
    )
    return rt


def test_compiled_path_engages_on_mtp_wrapped_runtime(tmp_path, monkeypatch):
    rt = _grafted_runtime(tmp_path)
    cache = _primed_cache(rt.model)
    monkeypatch.setenv("MTPLX_COMPILE_AR_FORWARD", "1")
    out = rt.forward_ar(mx.array([[5]]), cache=cache)
    mx.eval(out)
    assert rt.diagnostic_counters.get("compiled_forward_calls", 0) == 1, (
        "compiled AR path must engage on MTP-wrapped runtimes (the serving shape)"
    )
