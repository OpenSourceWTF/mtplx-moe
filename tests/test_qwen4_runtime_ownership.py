from __future__ import annotations

from inspect import signature
from pathlib import Path

import mtplx.runtime as runtime
from mtplx.mtp_patch import MTPContract


class _Owner:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def close(self):
        self.events.append(self.name)


def test_runtime_close_restores_qwen4_process_cache_limit(monkeypatch) -> None:
    events = []
    cache = _Owner(events, "cache")
    artifact = _Owner(events, "artifact")
    monkeypatch.setattr(
        runtime,
        "_unbind_qwen4_ngram_resources",
        lambda _model, _cache: events.append("unbind") or 1,
    )
    loaded = runtime.MTPLXRuntime(
        model=object(),
        tokenizer=object(),
        model_path=Path("/model"),
        mtp_enabled=True,
        contract=MTPContract(),
        ngram_cache=cache,
        ngram_artifact=artifact,
    )
    loaded.ngram_cache_limit_restore = lambda: events.append("restore-limit")

    loaded.close()

    assert events == ["unbind", "cache", "artifact", "restore-limit"]
    assert loaded.ngram_cache_limit_restore is None


def test_construction_reserves_largest_auto_prefill_chunk(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "4096")

    assert runtime._construction_prefill_chunk_tokens(None) == 4_096


def test_runtime_ngram_cache_limit_defaults_to_one_gib() -> None:
    assert signature(runtime.load).parameters["ngram_cache_limit_bytes"].default == (
        1024**3
    )
