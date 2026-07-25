"""Rejection repair without a verify snapshot (all-trimmable caches).

Profiles set MTPLX_SKIP_VERIFY_SNAPSHOT=1 for the product lanes; under plain
batched verify a rejected draft then had no repair path (no snapshot, no
trim/capture lane) and generate_mtpk raised — killing the first request that
ever saw a rejection. Attention-only models (every cache entry trimmable) can
always repair by trimming the uncommitted verify tail, snapshot or not.
"""
import json

import pytest

hy_v3 = pytest.importorskip(
    "mlx_lm.models.hy_v3",
    reason="mlx-lm does not ship models/hy_v3 yet (unreleased upstream)",
)

import mlx.core as mx
from mlx.utils import tree_flatten

from mtplx.cache_state import trim_verified_window_without_snapshot
from mtplx.generation import generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class _FixedTokenizer:
    eos_token_id = None
    eos_token_ids: set[int] = set()

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


def _grafted_runtime(tmp_path):
    from pathlib import Path

    from mtplx.hy_v3_mtp_patch import inject_hy_v3_mtp_support

    args = hy_v3.ModelArgs(
        model_type="hy_v3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, expert_hidden_dim=64,
        first_k_dense_replace=1, rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=1)
    mx.random.seed(7)
    model = hy_v3.Model(args)
    donor = hy_v3.DecoderLayer(args, layer_idx=2)
    tensors = {f"model.layers.2.{k}": v for k, v in tree_flatten(donor.parameters())}
    for name in ("enorm", "hnorm", "final_layernorm"):
        tensors[f"model.layers.2.{name}.weight"] = mx.ones((64,))
    tensors["model.layers.2.eh_proj.weight"] = 0.02 * mx.random.normal((64, 128))
    mx.save_safetensors(str(tmp_path / "model-mtp.safetensors"), tensors)
    json.dump(
        {"metadata": {}, "weight_map": {k: "model-mtp.safetensors" for k in tensors}},
        open(tmp_path / "model.safetensors.index.json", "w"),
    )
    cfg = {"model_type": "hy_v3", "num_nextn_predict_layers": 1, "num_hidden_layers": 2}
    assert inject_hy_v3_mtp_support(model, Path(tmp_path), cfg, None)
    return MTPLXRuntime(
        model=model, tokenizer=_FixedTokenizer(), model_path=Path(tmp_path),
        mtp_enabled=True, contract=MTPContract(),
    )


def _generate(rt, *, skip_snapshot, monkeypatch):
    if skip_snapshot:
        monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "1")
    else:
        monkeypatch.delenv("MTPLX_SKIP_VERIFY_SNAPSHOT", raising=False)
    out = generate_mtpk(
        rt,
        [1, 2, 3, 4, 5],
        max_tokens=24,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=1,
        mtp_history_policy="committed",
        stop_token_ids=set(),
    )
    return out


def test_unit_snapshot_free_trim_on_plain_kv_cache():
    from mlx_lm.models.cache import KVCache

    cache = [KVCache() for _ in range(2)]
    for entry in cache:
        entry.update_and_fetch(
            mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4))
        )
    assert all(entry.offset == 8 for entry in cache)
    # verified window of 4, keep 1 committed token -> trim 3
    assert trim_verified_window_without_snapshot(
        cache, verified_tokens=4, keep_tokens=1
    )
    assert all(entry.offset == 5 for entry in cache)

    class _Recurrent:  # no trim() => not trimmable
        offset = 5
        state = object()

    assert not trim_verified_window_without_snapshot(
        [cache[0], _Recurrent()], verified_tokens=2, keep_tokens=1
    )


def test_skip_snapshot_rejections_complete_and_match_snapshot_arm(
    tmp_path, monkeypatch
):
    rt = _grafted_runtime(tmp_path)
    baseline = _generate(rt, skip_snapshot=False, monkeypatch=monkeypatch)
    stats = baseline.stats.to_dict()
    assert stats.get("rejected_drafts", 0) > 0, (
        "test premise: the random draft head must produce rejections"
    )

    rt2 = _grafted_runtime(tmp_path)
    repaired = _generate(rt2, skip_snapshot=True, monkeypatch=monkeypatch)
    assert repaired.tokens == baseline.tokens, (
        "snapshot-free repair must reproduce the snapshot arm token-for-token"
    )
