from __future__ import annotations

from mtplx.attention_split import (
    _install_split_attention_hook,
    configure_split_full_attention,
)


class DummyProjection:
    weight = None


class DummyAttention:
    q_proj = DummyProjection()
    q_norm = DummyProjection()

    def __call__(self, x, mask=None, cache=None):
        return x


class DummyLayer:
    is_linear = False

    def __init__(self):
        self.self_attn = DummyAttention()


class DummyInner:
    def __init__(self):
        self.layers = [DummyLayer()]


class DummyModel:
    def __init__(self):
        self.model = DummyInner()


class Qwen4StyleAttention:
    """Qwen4 owns RoPE and indexer cache outside the legacy split contract."""

    def __call__(self, x, rope, mask, cache, idx_cache):
        return x


def test_split_hook_rejects_qwen4_external_rope_and_index_cache_contract():
    attention = Qwen4StyleAttention()
    native_call = type(attention).__call__

    assert _install_split_attention_hook(attention) is False
    assert type(attention).__call__ is native_call


def test_vllm_paged_hook_does_not_enable_split_full_attention(monkeypatch):
    monkeypatch.delenv("MTPLX_SPLIT_FULL_ATTN", raising=False)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")

    model = DummyModel()
    stats = configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    assert stats["enabled"] is True
    assert stats["split_full_attn_enabled"] is False
    assert attn._mtplx_vllm_metal_paged_enabled is True
    assert attn._mtplx_split_full_attention_explicit_enabled is False


def test_explicit_split_full_attention_chunk_one_gets_safe_default(monkeypatch):
    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN", "1")
    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE", "1")

    model = DummyModel()
    stats = configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    assert stats["split_full_attn_enabled"] is True
    assert stats["split_full_attn_chunk_size"] == 2048
    assert stats["split_full_attn_chunk_size_defaulted"] is True
    assert attn._mtplx_split_full_attention_explicit_enabled is True
    assert attn._mtplx_split_full_attention_chunk_size == 2048
