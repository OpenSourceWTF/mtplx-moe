"""Offline tests for the hy3 q2 LiteLLM handler.

These validate everything that does NOT require loading the model:
  * the module imports without mtplx / MLX present,
  * chat-template application produces the expected prompt token ids,
  * the raw-prompt (text_completion) path bypasses the chat template,
  * the sampler is built from OpenAI params (seed goes to generate, not sampler),
  * completion()/acompletion() shape a valid OpenAI ModelResponse,
  * string ``stop`` truncation works,
  * a full round-trip through ``litellm.completion`` via ``custom_provider_map``,
  * config.yaml parses AND its ``custom_handler`` string resolves via the proxy's
    own ``get_instance_fn`` to our handler instance.

Run from the THROWAWAY probe venv (litellm installed, mtplx NOT installed):
    litellm-probe-venv/bin/python evals/litellm_hy3/test_handler_offline.py

The generate_mtpk path is stubbed here; it is validated only in the guarded
GPU window (see README).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import handler as H  # noqa: E402
from litellm.types.utils import ModelResponse  # noqa: E402


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic, inspectable stand-in for mlx-lm's TokenizerWrapper."""

    def __init__(self) -> None:
        self.eos_token_id = 120025
        self.pad_token_id = 120002
        self.apply_calls: list[dict] = []
        self.encode_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.apply_calls.append({"messages": messages, "kwargs": kwargs})
        # A rendered template string that is clearly distinct from raw text.
        joined = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        gen = "<GEN>" if kwargs.get("add_generation_prompt") else ""
        return f"<TMPL>{joined}{gen}"

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append({"text": text, "add_special_tokens": add_special_tokens})
        # Encode as one id per character; BOS(=1) prepended only when specials on.
        ids = [ord(c) % 5000 for c in text]
        return ([1] + ids) if add_special_tokens else ids

    def decode(self, tokens):
        return "".join(chr((int(t) % 26) + 97) for t in tokens)


def fake_sampler_factory(**kwargs):
    return SimpleNamespace(kind="sampler", **kwargs)


def make_state(*, generate_impl, tokenizer=None, stop_token_ids=None):
    tok = tokenizer or FakeTokenizer()
    return H._State(
        runtime=SimpleNamespace(tokenizer=tok),
        tokenizer=tok,
        generate_mtpk=generate_impl,
        sampler_factory=fake_sampler_factory,
        stop_token_ids=stop_token_ids or {120025, 120002},
        speculative_depth=2,
    )


def install_state(state):
    H._STATE = state


def reset_state():
    H._STATE = None


def new_model_response():
    # Mirror what litellm hands the handler: choices[0].message already present.
    return ModelResponse(
        choices=[{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": None}]
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_module_imports_and_instance():
    assert isinstance(H.instance, H.Hy3StreamedLLM)
    assert hasattr(H.instance, "completion") and hasattr(H.instance, "acompletion")


def test_render_chat_prompt_ids_uses_template():
    tok = FakeTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    ids = H.render_prompt_ids(tok, messages, text_completion=False)
    # Chat template was applied with add_generation_prompt=True ...
    assert tok.apply_calls and tok.apply_calls[0]["kwargs"]["add_generation_prompt"] is True
    # ... and the rendered string was encoded WITHOUT re-adding special tokens.
    assert tok.encode_calls[-1]["add_special_tokens"] is False
    rendered = "<TMPL>system:sys|user:hi<GEN>"
    assert ids == [ord(c) % 5000 for c in rendered]


def test_render_raw_prompt_bypasses_template():
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "def add(a,b):"}]
    ids = H.render_prompt_ids(tok, messages, text_completion=True)
    assert tok.apply_calls == []  # chat template NOT used on the raw path
    # Raw path uses default encode (specials on -> BOS prepended).
    assert ids[0] == 1
    assert ids == [1] + [ord(c) % 5000 for c in "def add(a,b):"]


def test_coerce_token_ids_shapes():
    assert H._coerce_token_ids([1, 2, 3]) == [1, 2, 3]
    assert H._coerce_token_ids({"input_ids": [4, 5]}) == [4, 5]
    assert H._coerce_token_ids(SimpleNamespace(ids=[7, 8])) == [7, 8]
    assert H._coerce_token_ids([[1, 2], [3]]) == [1, 2, 3]


def test_sampler_maps_openai_params_seed_excluded():
    sampler = H._build_sampler(
        fake_sampler_factory,
        {"temperature": 0.3, "top_p": 0.8, "seed": 99, "presence_penalty": 0.5},
    )
    assert sampler.temperature == 0.3
    assert sampler.top_p == 0.8
    assert sampler.presence_penalty == 0.5
    # seed is a generate_mtpk arg, NOT a sampler field.
    assert not hasattr(sampler, "seed")


def test_completion_shapes_modelresponse():
    captured = {}

    def gen(rt, prompt_ids, **kw):
        captured["prompt_ids"] = list(prompt_ids)
        captured["kwargs"] = kw
        # 3 generated tokens then eos(120025); text is the assistant answer.
        return SimpleNamespace(
            tokens=[104, 105, 106, 120025],
            text="def foo():\n    return 42",
            finish_reason="stop",
            stats=None,
        )

    install_state(make_state(generate_impl=gen))
    try:
        mr = new_model_response()
        out = H.instance.completion(
            model="hy3-q2",
            messages=[{"role": "user", "content": "write foo"}],
            model_response=mr,
            optional_params={"temperature": 0.0, "max_tokens": 64, "seed": 7},
            litellm_params={},
        )
        assert out.choices[0].message.content == "def foo():\n    return 42"
        assert out.choices[0].message.role == "assistant"
        assert out.choices[0].finish_reason == "stop"
        # terminal eos stripped from the completion token count
        assert out.usage.completion_tokens == 3
        assert out.usage.prompt_tokens == len(captured["prompt_ids"])
        assert out.usage.total_tokens == out.usage.prompt_tokens + 3
        # correct plumbing into generate_mtpk
        assert captured["kwargs"]["speculative_depth"] == 2
        assert captured["kwargs"]["seed"] == 7
        assert captured["kwargs"]["max_tokens"] == 64
        assert 120025 in captured["kwargs"]["stop_token_ids"]
    finally:
        reset_state()


def test_text_completion_marker_routes_raw_path():
    seen = {}

    def gen(rt, prompt_ids, **kw):
        seen["prompt_ids"] = list(prompt_ids)
        return SimpleNamespace(tokens=[97, 98], text="xy", finish_reason="length", stats=None)

    tok = FakeTokenizer()
    install_state(make_state(generate_impl=gen, tokenizer=tok))
    try:
        H.instance.completion(
            model="hy3-q2",
            messages=[{"role": "user", "content": "raw-prompt"}],
            model_response=new_model_response(),
            optional_params={"max_tokens": 8},
            litellm_params={"text_completion": True},
        )
        assert tok.apply_calls == []  # raw path, no chat template
        assert seen["prompt_ids"] == [1] + [ord(c) % 5000 for c in "raw-prompt"]
    finally:
        reset_state()


def test_string_stop_truncation():
    def gen(rt, prompt_ids, **kw):
        return SimpleNamespace(
            tokens=[1, 2, 3, 4],
            text="good\nSTOPHEREbad",
            finish_reason="length",
            stats=None,
        )

    install_state(make_state(generate_impl=gen))
    try:
        mr = new_model_response()
        out = H.instance.completion(
            model="hy3-q2",
            messages=[{"role": "user", "content": "x"}],
            model_response=mr,
            optional_params={"max_tokens": 128, "stop": ["STOPHERE"]},
            litellm_params={},
        )
        assert out.choices[0].message.content == "good\n"
        # string stop forces finish_reason=stop even though generate hit length
        assert out.choices[0].finish_reason == "stop"
    finally:
        reset_state()


def test_finish_reason_length_when_no_stop():
    def gen(rt, prompt_ids, **kw):
        return SimpleNamespace(tokens=[10, 11, 12], text="abc", finish_reason="length", stats=None)

    install_state(make_state(generate_impl=gen))
    try:
        mr = new_model_response()
        out = H.instance.completion(
            model="hy3-q2",
            messages=[{"role": "user", "content": "x"}],
            model_response=mr,
            optional_params={"max_tokens": 3},
            litellm_params={},
        )
        assert out.choices[0].finish_reason == "length"
        assert out.usage.completion_tokens == 3
    finally:
        reset_state()


def test_acompletion_matches_completion():
    def gen(rt, prompt_ids, **kw):
        return SimpleNamespace(tokens=[1, 2], text="ok", finish_reason="stop", stats=None)

    install_state(make_state(generate_impl=gen))
    try:
        out = asyncio.run(
            H.instance.acompletion(
                model="hy3-q2",
                messages=[{"role": "user", "content": "x"}],
                model_response=new_model_response(),
                optional_params={},
                litellm_params={},
            )
        )
        assert out.choices[0].message.content == "ok"
    finally:
        reset_state()


def test_end_to_end_through_litellm_completion():
    import litellm

    def gen(rt, prompt_ids, **kw):
        return SimpleNamespace(
            tokens=[1, 2, 3, 120025],
            text="RESULT",
            finish_reason="stop",
            stats=None,
        )

    install_state(make_state(generate_impl=gen))
    litellm.custom_provider_map = [{"provider": "hy3", "custom_handler": H.instance}]
    try:
        resp = litellm.completion(
            model="hy3/hy3-q2",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=32,
        )
        assert type(resp).__name__ == "ModelResponse"
        assert resp.choices[0].message.content == "RESULT"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.completion_tokens == 3
    finally:
        litellm.custom_provider_map = []
        reset_state()


def test_config_yaml_parses_and_resolves_handler():
    import yaml
    from litellm.proxy.types_utils.utils import get_instance_fn

    config_path = _HERE / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    # model_list exposes hy3-q2 routed to provider hy3
    models = {m["model_name"]: m["litellm_params"]["model"] for m in cfg["model_list"]}
    assert models.get("hy3-q2") == "hy3/hy3-q2"
    cpm = cfg["litellm_settings"]["custom_provider_map"]
    entry = next(e for e in cpm if e["provider"] == "hy3")
    assert entry["custom_handler"] == "handler.instance"
    # The proxy resolves the string exactly this way (loads handler.py from the
    # config dir and reads ``instance``).
    resolved = get_instance_fn(value=entry["custom_handler"], config_file_path=str(config_path))
    assert type(resolved).__name__ == "Hy3StreamedLLM"
    assert hasattr(resolved, "completion")


# --------------------------------------------------------------------------
# Minimal runner (probe venv has no pytest).
# --------------------------------------------------------------------------


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            reset_state()
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {t.__name__}: {exc!r}")
            traceback.print_exc()
        finally:
            reset_state()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
