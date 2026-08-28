"""n-gram sidecar memory contract: streamed-table accounting + hot-row LRU.

Two product invariants (2026-08-28):

* The streamed table is NOT weight. Counting ngram-table.safetensors as
  wired weights gave a 128G Mac a ~99G "weights" plan, a false MODEL DOES
  NOT FIT, and a 30G-pessimistic context window and session bank — while
  the engine actually served fine at ~69G resident.
* The hot-row LRU must be invisible in values: every gather, hit or miss,
  returns byte-identical rows to the plain memmap path.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mtplx.engine_session import model_weights_bytes, ngram_table_bytes
from mtplx.memory_plan import (
    GIB,
    NGRAM_TABLE_FILENAME,
    describe_plan,
    ngram_table_resident_policy,
    plan_memory,
)
from mtplx.models.qwen4_exp import NGramTable, _ngram_resident_policy

ROWS, DIM, GROUP, BITS = 64, 64, 32, 4


def _write_quantized_table(path):
    w = mx.random.normal((ROWS, DIM)).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(wq, scales, biases)
    mx.save_safetensors(
        str(path),
        {"ngram.weight": wq, "ngram.scales": scales, "ngram.biases": biases},
        metadata={"ngram_bits": str(BITS), "ngram_group_size": str(GROUP)},
    )
    ref = mx.dequantize(wq, scales, biases, group_size=GROUP, bits=BITS)
    mx.eval(ref)
    return ref


def _write_raw_table(path):
    w = mx.random.normal((ROWS, DIM)).astype(mx.bfloat16)
    mx.eval(w)
    mx.save_safetensors(
        str(path),
        {"ngram.weight": w},
        metadata={"ngram_bits": "0", "ngram_group_size": str(GROUP)},
    )
    return w


def _attached_table(path):
    table = NGramTable(ROWS, DIM, sidecar=True)
    table.attach_sidecar(path)
    return table


def _gather(table, ids):
    out = table(mx.array(np.asarray(ids, dtype=np.int64)))
    mx.eval(out)
    return np.asarray(out.astype(mx.float32))


def test_sidecar_gather_matches_reference_and_hot_cache_is_value_invisible(
    tmp_path,
):
    path = tmp_path / NGRAM_TABLE_FILENAME
    ref = _write_quantized_table(path)
    table = _attached_table(path)
    sidecar = table._sidecar
    assert sidecar._hot_cap_rows > 0  # default 1024M cap is on

    ids = np.array([0, 3, 3, 17, ROWS - 1, 5], dtype=np.int64)
    ref_rows = np.asarray(ref.astype(mx.float32))[ids]

    cold = _gather(table, ids)
    assert np.array_equal(cold, ref_rows)
    assert sidecar.hot_misses == 5 and sidecar.hot_hits == 0  # 5 unique rows

    warm = _gather(table, ids)  # all hits now
    assert np.array_equal(warm, cold)
    assert sidecar.hot_hits == 5 and sidecar.hot_misses == 5


def test_hot_cache_eviction_keeps_bound_and_values(tmp_path):
    path = tmp_path / NGRAM_TABLE_FILENAME
    ref = _write_quantized_table(path)
    table = _attached_table(path)
    sidecar = table._sidecar
    sidecar._hot_cap_rows = 4

    ids = np.arange(12, dtype=np.int64)
    out = _gather(table, ids)
    assert np.array_equal(out, np.asarray(ref.astype(mx.float32))[ids])
    assert len(sidecar._hot) <= 4

    # Evicted rows re-fetch correctly.
    again = _gather(table, ids[:3])
    assert np.array_equal(again, np.asarray(ref.astype(mx.float32))[ids[:3]])


def test_hot_cache_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_NGRAM_HOT_MB", "0")
    path = tmp_path / NGRAM_TABLE_FILENAME
    ref = _write_quantized_table(path)
    table = _attached_table(path)
    sidecar = table._sidecar

    ids = np.array([1, 2, 2, 9], dtype=np.int64)
    out = _gather(table, ids)
    assert np.array_equal(out, np.asarray(ref.astype(mx.float32))[ids])
    assert not sidecar._hot and sidecar.hot_hits == 0 and sidecar.hot_misses == 0


def test_big_gathers_bypass_hot_cache(tmp_path):
    path = tmp_path / NGRAM_TABLE_FILENAME
    ref = _write_quantized_table(path)
    table = _attached_table(path)
    sidecar = table._sidecar
    sidecar._HOT_PATH_MAX_ROWS = 4  # instance override: 5+ unique rows bypass

    ids = np.arange(10, dtype=np.int64)
    out = _gather(table, ids)
    assert np.array_equal(out, np.asarray(ref.astype(mx.float32))[ids])
    assert not sidecar._hot  # bypassed: nothing cached, values still exact


def test_raw_bf16_sidecar_mode(tmp_path):
    path = tmp_path / NGRAM_TABLE_FILENAME
    ref = _write_raw_table(path)
    table = _attached_table(path)

    ids = np.array([0, 7, 7, ROWS - 1], dtype=np.int64)
    out = _gather(table, ids)
    assert np.array_equal(out, np.asarray(ref.astype(mx.float32))[ids])


def test_model_weights_bytes_excludes_streamed_table(tmp_path):
    shard = mx.zeros((256, 8), dtype=mx.float16)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"w": shard})
    _write_quantized_table(tmp_path / NGRAM_TABLE_FILENAME)

    weights = model_weights_bytes(tmp_path)
    table = ngram_table_bytes(tmp_path)
    assert weights == (tmp_path / "model.safetensors").stat().st_size
    assert table == (tmp_path / NGRAM_TABLE_FILENAME).stat().st_size
    assert ngram_table_bytes(tmp_path / "missing") == 0


def test_resident_policy_env_pins_and_single_source(monkeypatch):
    monkeypatch.setenv("MTPLX_NGRAM_RESIDENT", "1")
    assert ngram_table_resident_policy() is True
    assert _ngram_resident_policy() is True
    monkeypatch.setenv("MTPLX_NGRAM_RESIDENT", "0")
    assert ngram_table_resident_policy() is False
    assert _ngram_resident_policy() is False
    monkeypatch.setenv("MTPLX_NGRAM_RESIDENT", "auto")
    assert isinstance(ngram_table_resident_policy(), bool)


def test_plan_streams_table_as_note_not_commitment():
    # 128G Mac, 69G weights, 30G table streamed: fits, and the banner says
    # where the other 30G of the pack lives.
    plan = plan_memory(
        total_ram_bytes=128 * GIB,
        model_weights_bytes=69 * GIB,
        ngram_table_streamed_bytes=30 * GIB,
        kv_bytes_per_token=65536,
        model_max_context=262144,
    )
    assert plan.available and plan.model_fits
    assert plan.ngram_table_streamed_bytes == 30 * GIB
    assert plan.to_dict()["ngram_table_streamed_bytes"] == 30 * GIB
    line = describe_plan(plan)
    assert "streamed from SSD" in line and "DOES NOT FIT" not in line

    # The old accounting (table folded into weights on a 128G box) is the
    # regression this pins against: 99G "weights" breaks the fit.
    old = plan_memory(
        total_ram_bytes=128 * GIB,
        model_weights_bytes=99 * GIB,
        kv_bytes_per_token=65536,
        model_max_context=262144,
    )
    assert not old.model_fits


@pytest.mark.parametrize("value", ["1024", "not-a-number"])
def test_hot_cache_env_parse_is_forgiving(tmp_path, monkeypatch, value):
    monkeypatch.setenv("MTPLX_NGRAM_HOT_MB", value)
    path = tmp_path / NGRAM_TABLE_FILENAME
    _write_quantized_table(path)
    table = _attached_table(path)
    assert table._sidecar._hot_cap_rows > 0  # bad value falls back to default


@pytest.mark.parametrize("resident", [False, True])
def test_memory_attribution_counts_table_only_when_resident(
    tmp_path, monkeypatch, resident
):
    from types import SimpleNamespace

    from mtplx.server.openai import _memory_attribution

    shard = mx.zeros((256, 8), dtype=mx.float16)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"w": shard})
    _write_quantized_table(tmp_path / NGRAM_TABLE_FILENAME)
    monkeypatch.setenv("MTPLX_NGRAM_RESIDENT", "1" if resident else "0")

    state = SimpleNamespace(args=SimpleNamespace(model=str(tmp_path)))
    weights = _memory_attribution(state)["model_weights_bytes"]
    shard_bytes = (tmp_path / "model.safetensors").stat().st_size
    table_bytes = (tmp_path / NGRAM_TABLE_FILENAME).stat().st_size
    assert weights == shard_bytes + (table_bytes if resident else 0)
