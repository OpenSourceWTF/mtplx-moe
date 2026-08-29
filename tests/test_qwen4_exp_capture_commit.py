"""Family layer-owned capture-commit: repair-free verify rollback.

A rejected speculative window must commit by replaying ONLY the GDN
recurrences (and trimming trimmable entries) from the pre-verify snapshot,
matching a run that never saw the rejected rows to fp32 ulp-class tolerance.
Not bitwise: the chunked gated-delta scan reassociates when the kept rows
ride a wider verify window, so captured activations differ from a fresh
narrow forward's at the last float — the same noise class the fallback
path's own rollback+re-forward produces relative to the verify pass. The
acceptance decision itself always uses the verify pass's own logits, so
this tolerance never touches sampling exactness.

CPU-only (parity surface).
"""

import mlx.core as mx
import pytest

from mtplx.cache_state import snapshot_untrimmable_cache_lazy
from mtplx.models.qwen4_exp import (
    TextArgs,
    TextModel,
    verify_capture_scope,
)


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
    )


@pytest.fixture()
def tm():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = TextModel(_tiny_args())
    mx.eval(model.parameters())
    yield model
    mx.set_default_device(prev)


def _ids(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, 128, (1, tokens))


PREFILL = 12
WINDOW = 4
KEEP = 2


def _run(tm, chunks, cache):
    out = None
    for ids in chunks:
        out = tm.model(ids, cache)
    return out


def test_capture_commit_matches_fresh_run_eager(tm):
    ids_pre = _ids(PREFILL, seed=1)
    ids_verify = _ids(WINDOW, seed=2)
    ids_next = _ids(3, seed=3)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_verify, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [ids_next], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_verify[:, :KEEP], golden_cache)
    golden = _run(tm, [ids_next], golden_cache)

    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_capture_commit_matches_fresh_run_compiled(tm):
    tm.model._gdn_compiled_env = True

    ids_pre = _ids(PREFILL, seed=4)
    ids_verify = _ids(WINDOW, seed=5)
    ids_next = _ids(3, seed=6)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_verify, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [ids_next], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_verify[:, :KEEP], golden_cache)
    golden = _run(tm, [ids_next], golden_cache)

    # The compiled/eager boundary may reorder float ops; the commit itself
    # must still be exact relative to the same-lane golden (golden ran
    # eager S=2 which the compiled gate also serves) — compare through the
    # compiled lane end to end.
    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_commit_refuses_without_capture_and_leaves_cache_intact(tm):
    ids_pre = _ids(PREFILL, seed=7)
    ids_verify = _ids(WINDOW, seed=8)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    tm.model(ids_verify, cache)  # NOT captured
    offsets_before = [getattr(c, "offset", None) for c in cache]
    assert not tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    assert [getattr(c, "offset", None) for c in cache] == offsets_before


def test_full_accept_needs_no_commit_and_next_round_overwrites_rows(tm):
    ids_pre = _ids(PREFILL, seed=9)
    ids_v1 = _ids(WINDOW, seed=10)
    ids_v2 = _ids(WINDOW, seed=11)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    with verify_capture_scope():
        tm.model(ids_v1, cache)  # full accept: no commit
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_v2, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [_ids(2, seed=12)], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_v1, golden_cache)
    tm.model(ids_v2[:, :KEEP], golden_cache)
    golden = _run(tm, [_ids(2, seed=12)], golden_cache)

    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()
