from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.runtime import _load_tokenizer_resilient
from mtplx.server.openai import ChatMessage, _encode_messages


def _kimi_k3_config() -> dict[str, object]:
    return {
        "model_type": "kimi_linear",
        "hidden_act": "situ",
        "num_hidden_layers": 93,
        "num_experts": 896,
        "eos_token_id": 163586,
    }


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool, dict[str, object]]] = []

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool = False,
        **kwargs,
    ):
        self.calls.append((conversation, tokenize, dict(kwargs)))
        return [17, 23] if tokenize else "rendered"


def test_kimi_k3_loads_remote_tiktoken_code_without_tokenizer_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, dict[str, object] | None, list[int] | None]] = []
    inner = _RecordingTokenizer()
    wrapped = SimpleNamespace(_tokenizer=inner, _chat_template=None)

    def fake_load_tokenizer(
        model_path: Path,
        tokenizer_config_extra: dict[str, object] | None = None,
        eos_token_ids: list[int] | None = None,
    ):
        calls.append((model_path, tokenizer_config_extra, eos_token_ids))
        return wrapped

    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", fake_load_tokenizer)

    tokenizer = _load_tokenizer_resilient(tmp_path, _kimi_k3_config())

    assert calls == [(tmp_path, {"trust_remote_code": True}, [163586])]
    assert tokenizer.eos_token_ids == {163586}


def test_kimi_k3_chat_controls_are_bound_to_native_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inner = _RecordingTokenizer()
    wrapped = SimpleNamespace(_tokenizer=inner, _chat_template=None)
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *args, **kwargs: wrapped)
    tokenizer = _load_tokenizer_resilient(tmp_path, _kimi_k3_config())
    messages = [{"role": "user", "content": "hello"}]

    assert (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
        == "rendered"
    )
    assert inner.calls[-1] == (
        messages,
        False,
        {
            "add_generation_prompt": True,
            "return_dict": False,
            "thinking": False,
        },
    )

    assert tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        enable_thinking=True,
        reasoning_effort="high",
    ) == [17, 23]
    assert inner.calls[-1] == (
        messages,
        True,
        {
            "return_dict": False,
            "thinking": True,
            "thinking_effort": "high",
        },
    )


def test_kimi_k3_thinking_effort_is_not_synthesized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inner = _RecordingTokenizer()
    wrapped = SimpleNamespace(_tokenizer=inner, _chat_template=None)
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *args, **kwargs: wrapped)
    tokenizer = _load_tokenizer_resilient(tmp_path, _kimi_k3_config())

    tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        enable_thinking=True,
    )

    assert "thinking_effort" not in inner.calls[-1][2]


def test_openai_chat_encoding_reaches_k3_native_thinking_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inner = _RecordingTokenizer()
    wrapped = SimpleNamespace(_tokenizer=inner, _chat_template=None)
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *args, **kwargs: wrapped)
    tokenizer = _load_tokenizer_resilient(tmp_path, _kimi_k3_config())

    token_ids = _encode_messages(
        tokenizer,
        [ChatMessage(role="user", content="hello")],
        enable_thinking=True,
        reasoning_effort="high",
        add_generation_prompt=True,
    )

    assert token_ids == [17, 23]
    _conversation, tokenize, kwargs = inner.calls[-1]
    assert tokenize is True
    assert kwargs == {
        "add_generation_prompt": True,
        "return_dict": False,
        "thinking": True,
        "thinking_effort": "high",
    }


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "kimi_linear", "num_experts": 384},
        {"model_type": "kimi_linear", "num_experts": 896, "num_hidden_layers": 92},
        {"model_type": "other", "num_experts": 896, "num_hidden_layers": 93},
    ],
)
def test_non_k3_tokenizers_keep_the_existing_loader_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: dict[str, object],
) -> None:
    sentinel = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", fake_load_tokenizer)

    assert _load_tokenizer_resilient(tmp_path, config) is sentinel
    assert calls == [((tmp_path,), {})]
