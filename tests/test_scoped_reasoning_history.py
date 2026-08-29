"""Scoped reasoning history - the loop root-cause fix.

Qwen3.6/3.5 chat templates carry a "rolling checkpoint" (``last_query_index``):
they keep ``<think>`` blocks only for assistant messages after the last real
user query. MTPLX historically forced ``preserve_thinking=True``, overriding
that checkpoint. Measured legacy behavior (decoded from a live SSD-banked
loop prompt): structured ``reasoning_content`` history fields were dropped at
``_message_to_template_dict`` before the template, so preserve-all rendered an
EMPTY ``<think>\\n\\n</think>`` scaffold for every completed assistant turn -
off-contract scaffolding the model never saw in training - while inline
``<think>`` text in replayed content was preserved verbatim across all turns.

Scoped mode restores Qwen's trained contract in both directions:
- completed turns render with no think scaffold at all (inline think in
  replayed content is scoped out by the template's own split logic);
- the active agent round (assistant -> tool -> assistant chains after the
  last real user query) keeps its reasoning, including the structured
  ``reasoning_content`` fields OpenCode sends - the interleaved-thinking
  continuity every provider preserves, which legacy preserve-all silently
  dropped.

``on``/``off`` remain byte-identical to the legacy renderings (rollback path).

The golden rendering tests run against the byte-identical template shipped
with the Qwen3.6 Optimized Speed/Quality artifacts
(``tests/fixtures/qwen36_rolling_checkpoint_chat_template.jinja``).
"""

from pathlib import Path
from types import SimpleNamespace

import jinja2
import jinja2.sandbox
import pytest

from mtplx.server.openai import (
    ChatMessage,
    _REASONING_HISTORY_PRESERVE,
    _REASONING_HISTORY_SCOPED,
    _REASONING_HISTORY_STRIP,
    _encode_messages,
    _normalize_preserve_thinking_policy,
    _policy_fingerprint,
    _preserve_thinking_effective,
    _reasoning_history_fingerprint_component,
    _reasoning_history_mode,
    _reasoning_history_preserve_echo_active,
    _reasoning_history_scoped_active,
    _template_supports_scoped_reasoning,
    parse_args,
)

FIXTURE_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen36_rolling_checkpoint_chat_template.jinja"
).read_text(encoding="utf-8")


class Qwen36TemplateTokenizer:
    """Renders the real shipped Qwen3.6 chat template via jinja2.

    Mirrors the HF chat-template environment closely enough for golden
    rendering assertions (raise_exception, loop controls, string methods).
    Captures the last rendered text so tests can assert on prompt bytes.
    """

    def __init__(self, template: str = FIXTURE_TEMPLATE):
        self.chat_template = template
        self.last_rendered: str | None = None
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, **kwargs):
        render_kwargs = {
            key: value for key, value in kwargs.items() if key != "tokenize"
        }
        rendered = self._template.render(messages=messages, **render_kwargs)
        self.last_rendered = rendered
        if kwargs.get("tokenize"):
            return [ord(char) for char in rendered]
        return rendered

    def encode(self, text, **_kwargs):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


def _render_history(
    messages,
    *,
    strip_assistant_reasoning_history=False,
    scoped_reasoning_history=False,
    preserve_reasoning_history=False,
    enable_thinking=True,
):
    tokenizer = Qwen36TemplateTokenizer()
    _encode_messages(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        strip_assistant_reasoning_history=strip_assistant_reasoning_history,
        scoped_reasoning_history=scoped_reasoning_history,
        preserve_reasoning_history=preserve_reasoning_history,
    )
    assert tokenizer.last_rendered is not None
    return tokenizer.last_rendered


def _completed_turn_history():
    """Two completed rounds plus a fresh user query (the OpenCode shape)."""

    return [
        ChatMessage(role="system", content="You are a coding agent."),
        ChatMessage(role="user", content="Plan the chess engine."),
        ChatMessage(
            role="assistant",
            content="Here is the plan.",
            reasoning_content="THINK_TURN_ONE planning the chess engine",
        ),
        ChatMessage(role="user", content="Now execute the plan."),
        ChatMessage(
            role="assistant",
            content="Executed step one.",
            reasoning_content="THINK_TURN_TWO executing the plan",
        ),
        ChatMessage(role="user", content="Continue with step two."),
    ]


def _active_round_history():
    """An in-flight agent round: user -> assistant+tool_calls -> tool result."""

    return [
        ChatMessage(role="system", content="You are a coding agent."),
        ChatMessage(role="user", content="List the project files."),
        ChatMessage(
            role="assistant",
            content="",
            reasoning_content="THINK_ACTIVE_ROUND choosing the ls tool",
            tool_calls=[
                {
                    "id": "call_ls",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_ls", content="src\npackage.json"),
    ]


# ---------------------------------------------------------------------------
# Golden rendering: scoped mode against the real shipped template
# ---------------------------------------------------------------------------


def test_scoped_strips_completed_turn_reasoning_from_history():
    rendered = _render_history(
        _completed_turn_history(),
        scoped_reasoning_history=True,
    )
    assert "THINK_TURN_ONE" not in rendered
    assert "THINK_TURN_TWO" not in rendered
    # The visible answers survive untouched.
    assert "Here is the plan." in rendered
    assert "Executed step one." in rendered
    # No empty think scaffolds on completed turns either - the only <think>
    # left is the generation prompt's opening tag.
    assert rendered.count("<think>") == 1
    assert rendered.rstrip().endswith("<think>")


def test_scoped_keeps_active_round_reasoning():
    rendered = _render_history(
        _active_round_history(),
        scoped_reasoning_history=True,
    )
    # The assistant message sits after the last real user query (the tool
    # result is not a query), so the rolling checkpoint keeps its reasoning.
    assert "THINK_ACTIVE_ROUND" in rendered
    assert "<tool_response>" in rendered


def test_scoped_active_round_matches_preserve_for_inline_reasoning():
    """Quality-retention proof: inside the active round, scoped == preserve
    for inline think - the only form legacy preserve-all actually rendered."""

    messages = [
        ChatMessage(role="user", content="List the project files."),
        ChatMessage(
            role="assistant",
            content=(
                "<think>\nTHINK_ACTIVE_INLINE choosing the ls tool\n</think>\n\n"
            ),
            tool_calls=[
                {
                    "id": "call_ls",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_ls", content="src\npackage.json"),
    ]
    scoped = _render_history(messages, scoped_reasoning_history=True)
    preserve = _render_history(messages, scoped_reasoning_history=False)
    assert "THINK_ACTIVE_INLINE" in scoped
    assert scoped == preserve


def test_scoped_carries_structured_reasoning_legacy_preserve_dropped():
    """Legacy preserve-all silently DROPPED OpenCode's structured
    reasoning_content fields (measured on a live SSD-banked loop prompt:
    every history think block rendered empty). Scoped carries them for the
    active round - strictly more in-round continuity than today's default."""

    preserve = _render_history(
        _active_round_history(),
        scoped_reasoning_history=False,
    )
    assert "THINK_ACTIVE_ROUND" not in preserve
    assert "<think>\n\n</think>" in preserve


# ---------------------------------------------------------------------------
# Preserve-mode echo-carry (2026-08-21): postcommit starvation froze committed
# streams (15,389 tokens while the true stream passed 76k) and legacy preserve
# dropped the client's echoed reasoning, so the model re-derived a 57.8k-token
# turn from scratch. Preserve now carries the echo as the BASE render
# (committed-think substitution still overwrites covered turns with KV-exact
# bytes); explicit policy `on` keeps the legacy drop as the rollback lane.
# ---------------------------------------------------------------------------


def test_preserve_echo_renders_structured_reasoning_history():
    rendered = _render_history(
        _completed_turn_history(),
        preserve_reasoning_history=True,
    )
    assert "THINK_TURN_ONE" in rendered
    assert "THINK_TURN_TWO" in rendered
    assert "Here is the plan." in rendered
    assert "Executed step one." in rendered


def test_preserve_echo_flag_off_keeps_legacy_drop():
    """The bare-function default (and policy `on`) stays byte-identical
    legacy: structured echoes drop and completed turns render the empty
    scaffold."""
    rendered = _render_history(_completed_turn_history())
    assert "THINK_TURN_ONE" not in rendered
    assert "<think>\n\n</think>" in rendered


def test_preserve_echo_never_renders_with_thinking_off():
    rendered = _render_history(
        _completed_turn_history(),
        preserve_reasoning_history=True,
        enable_thinking=False,
    )
    assert "THINK_TURN_ONE" not in rendered
    assert "THINK_TURN_TWO" not in rendered


def test_preserve_echo_never_renders_under_strip():
    rendered = _render_history(
        _completed_turn_history(),
        preserve_reasoning_history=True,
        strip_assistant_reasoning_history=True,
    )
    assert "THINK_TURN_ONE" not in rendered


def test_preserve_echo_inline_think_content_is_single_source():
    messages = [
        ChatMessage(role="user", content="List the project files."),
        ChatMessage(
            role="assistant",
            content="<think>\nTHINK_INLINE original\n</think>\n\nDone.",
            reasoning_content="THINK_FIELD duplicate copy",
        ),
        ChatMessage(role="user", content="Thanks."),
    ]
    rendered = _render_history(messages, preserve_reasoning_history=True)
    assert "THINK_INLINE" in rendered
    assert "THINK_FIELD" not in rendered, (
        "content already starts with an inline think block - including the "
        "echoed field too would render the turn's thinking twice"
    )


def _echo_state(**arg_overrides):
    defaults = dict(
        strip_assistant_reasoning_history=False,
        preserve_thinking="auto",
    )
    defaults.update(arg_overrides)
    return SimpleNamespace(
        args=SimpleNamespace(**defaults),
        reasoning_history_scoped_capable=False,
    )


def test_preserve_echo_active_resolution_and_fingerprint():
    auto_state = _echo_state()
    assert _reasoning_history_preserve_echo_active(auto_state) is True
    assert (
        _reasoning_history_fingerprint_component(auto_state)
        == "reasoning_history=preserve_echo"
    )

    on_state = _echo_state(preserve_thinking="on")
    assert _reasoning_history_preserve_echo_active(on_state) is False
    assert (
        _reasoning_history_fingerprint_component(on_state) == "strip_reasoning=0"
    ), "policy `on` must keep the exact legacy fingerprint (rollback lane)"

    off_state = _echo_state(preserve_thinking="off")
    assert _reasoning_history_preserve_echo_active(off_state) is False
    assert (
        _reasoning_history_fingerprint_component(off_state) == "strip_reasoning=1"
    )

    scoped_state = _echo_state(preserve_thinking="scoped")
    scoped_state.reasoning_history_scoped_capable = True
    assert _reasoning_history_preserve_echo_active(scoped_state) is False
    assert (
        _reasoning_history_fingerprint_component(scoped_state)
        == "reasoning_history=scoped"
    )
    scoped = _render_history(
        _active_round_history(),
        scoped_reasoning_history=True,
    )
    assert "THINK_ACTIVE_ROUND" in scoped


def test_scoped_scopes_inline_think_history_via_template_split():
    """Replayed assistant content with inline <think> is scoped too."""

    messages = [
        ChatMessage(role="user", content="First question."),
        ChatMessage(
            role="assistant",
            content=(
                "<think>\nTHINK_INLINE_OLD stale plan\n</think>\n\n"
                "Old inline answer."
            ),
        ),
        ChatMessage(role="user", content="Second question."),
    ]
    rendered = _render_history(messages, scoped_reasoning_history=True)
    assert "THINK_INLINE_OLD" not in rendered
    assert "Old inline answer." in rendered
    preserved = _render_history(messages, scoped_reasoning_history=False)
    assert "THINK_INLINE_OLD" in preserved


def test_preserve_all_keeps_legacy_rendering_including_empty_scaffolds():
    """`on` keeps today's exact behavior: structured reasoning_content stays
    dropped and every completed assistant turn gets the empty think scaffold
    (the measured legacy rendering this mode exists to roll back to)."""

    rendered = _render_history(
        _completed_turn_history(),
        scoped_reasoning_history=False,
    )
    assert "THINK_TURN_ONE" not in rendered
    assert "THINK_TURN_TWO" not in rendered
    assert rendered.count("<think>\n\n</think>") == 2


def test_preserve_all_keeps_inline_think_across_completed_turns():
    """`on` preserves inline think in replayed content across all turns -
    the form the legacy mode actually rendered."""

    messages = [
        ChatMessage(role="user", content="First question."),
        ChatMessage(
            role="assistant",
            content="<think>\nTHINK_INLINE_OLD\n</think>\n\nOld answer.",
        ),
        ChatMessage(role="user", content="Second question."),
    ]
    rendered = _render_history(messages, scoped_reasoning_history=False)
    assert "THINK_INLINE_OLD" in rendered


def test_full_strip_renders_no_history_think_block():
    """`off` keeps today's exact behavior - all history reasoning out."""

    rendered = _render_history(
        _completed_turn_history(),
        strip_assistant_reasoning_history=True,
    )
    assert "THINK_TURN_ONE" not in rendered
    assert "THINK_TURN_TWO" not in rendered
    assert "Here is the plan." in rendered


# ---------------------------------------------------------------------------
# Capability probe + policy resolution
# ---------------------------------------------------------------------------


def test_template_probe_detects_rolling_checkpoint():
    assert _template_supports_scoped_reasoning(
        SimpleNamespace(chat_template=FIXTURE_TEMPLATE)
    )


def test_template_probe_rejects_templates_without_checkpoint():
    assert not _template_supports_scoped_reasoning(
        SimpleNamespace(chat_template="{% for message in messages %}...{% endfor %}")
    )
    assert not _template_supports_scoped_reasoning(SimpleNamespace(chat_template=None))
    assert not _template_supports_scoped_reasoning(SimpleNamespace())


def test_template_probe_rejects_gemma4_tokenizer():
    gemma = SimpleNamespace(
        chat_template="last_query_index",
        model_specific_special_tokens={
            "think_token": "<|think|>",
            "soc_token": "<|channel>",
            "eoc_token": "<channel|>",
        },
    )
    assert not _template_supports_scoped_reasoning(gemma)


def _state(
    policy: str,
    *,
    capable: bool,
    strip_flag: bool = False,
    model: str | None = None,
):
    return SimpleNamespace(
        args=SimpleNamespace(
            preserve_thinking=policy,
            strip_assistant_reasoning_history=strip_flag,
            model=model,
        ),
        reasoning_history_scoped_capable=capable,
    )


def test_auto_resolves_to_scoped_only_when_template_is_capable():
    assert (
        _reasoning_history_mode(_state("auto", capable=True))
        == _REASONING_HISTORY_SCOPED
    )
    assert (
        _reasoning_history_mode(_state("auto", capable=False))
        == _REASONING_HISTORY_PRESERVE
    )


def test_explicit_policies_resolve_exactly():
    assert (
        _reasoning_history_mode(_state("on", capable=True))
        == _REASONING_HISTORY_PRESERVE
    )
    assert (
        _reasoning_history_mode(_state("off", capable=True))
        == _REASONING_HISTORY_STRIP
    )
    assert (
        _reasoning_history_mode(_state("scoped", capable=True))
        == _REASONING_HISTORY_SCOPED
    )
    # Explicit scoped on a checkpoint-free template falls back to preserve:
    # sending preserve_thinking=False there would strip everything instead
    # of scoping (e.g. the froggeric profile), which is not what was asked.
    assert (
        _reasoning_history_mode(_state("scoped", capable=False))
        == _REASONING_HISTORY_PRESERVE
    )


def test_legacy_strip_flag_still_wins():
    assert (
        _reasoning_history_mode(_state("auto", capable=True, strip_flag=True))
        == _REASONING_HISTORY_STRIP
    )
    assert not _reasoning_history_scoped_active(
        _state("auto", capable=True, strip_flag=True)
    )


def test_auto_preserves_for_qwen38_and_flash_next_even_on_capable_templates():
    """The 3.8-generation trained contract: preserve_thinking on by default.

    Both families ship templates that CARRY the rolling checkpoint, so the
    capability probe alone would resolve auto -> scoped — which made every
    Flash-Next agent round abort postcommit with
    reasoning_history_scoping_mismatch and re-prefill the whole assistant
    turn (OpenCode wall-clock rig, 2026-08-28). Auto must resolve to
    preserve for qwen3_8 and qwen4_exp on every harness; only an explicit
    off/scoped policy changes it.
    """
    for model in (
        "/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
        "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
    ):
        assert (
            _reasoning_history_mode(_state("auto", capable=True, model=model))
            == _REASONING_HISTORY_PRESERVE
        ), model
        assert (
            _reasoning_history_mode(_state("off", capable=True, model=model))
            == _REASONING_HISTORY_STRIP
        ), model


def test_normalize_policy_accepts_scoped_and_rejects_junk():
    assert _normalize_preserve_thinking_policy("scoped") == "scoped"
    assert _normalize_preserve_thinking_policy(" SCOPED ") == "scoped"
    with pytest.raises(ValueError):
        _normalize_preserve_thinking_policy("sometimes")


def test_preserve_thinking_effective_counts_scoped_as_preserving():
    args = SimpleNamespace(preserve_thinking="scoped")
    assert _preserve_thinking_effective(args) is True
    args = SimpleNamespace(preserve_thinking="off")
    assert _preserve_thinking_effective(args) is False


def test_parse_args_accepts_scoped_and_keeps_strip_flag_off():
    args = parse_args(["--preserve-thinking", "scoped", "--warmup-tokens", "0"])
    assert args.preserve_thinking == "scoped"
    assert args.strip_assistant_reasoning_history is False


# ---------------------------------------------------------------------------
# Cache identity: legacy fingerprints pinned, scoped mints a new one
# ---------------------------------------------------------------------------


def test_fingerprint_component_pins_legacy_strings_for_on_and_off():
    # Explicit on/off users must keep their warm session banks: the emitted
    # component strings are byte-identical to the pre-scoped release.
    assert (
        _reasoning_history_fingerprint_component(_state("on", capable=True))
        == "strip_reasoning=0"
    )
    assert (
        _reasoning_history_fingerprint_component(_state("off", capable=True))
        == "strip_reasoning=1"
    )


def test_fingerprint_component_mints_new_identity_for_scoped_and_echo():
    assert (
        _reasoning_history_fingerprint_component(_state("auto", capable=True))
        == "reasoning_history=scoped"
    )
    # Auto-preserve carries the client's echoed reasoning since 2026-08-21
    # (postcommit-starvation fix), so its rendered bytes genuinely differ
    # from legacy preserve - same honesty rule as scoped. Explicit `on`
    # stays the byte-identical legacy lane above.
    assert (
        _reasoning_history_fingerprint_component(_state("auto", capable=False))
        == "reasoning_history=preserve_echo"
    )


def _fingerprint_state(policy: str, *, capable: bool):
    args = SimpleNamespace(
        preserve_thinking=policy,
        strip_assistant_reasoning_history=False,
        generation_mode="mtp",
        depth=3,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.0,
        online_hidden_corrector_decay=0.8,
        online_hidden_corrector_warmup=1,
        online_hidden_corrector_max_feed_depth=None,
        online_hidden_corrector_key="global",
        tool_prompt_mode="hybrid",
    )
    return SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
        reasoning_history_scoped_capable=capable,
    )


def test_policy_fingerprint_scoped_differs_but_on_matches_legacy():
    scoped = _policy_fingerprint(
        _fingerprint_state("auto", capable=True), thinking_enabled=True
    )
    preserve = _policy_fingerprint(
        _fingerprint_state("on", capable=True), thinking_enabled=True
    )
    legacy_preserve = _policy_fingerprint(
        _fingerprint_state("auto", capable=False), thinking_enabled=True
    )
    assert "reasoning_history=scoped" in scoped
    assert "strip_reasoning=0" in preserve
    # `on` keeps the exact legacy component so its warm banks stay valid;
    # auto-preserve mints preserve_echo (echo-carry renders different bytes,
    # 2026-08-21 postcommit-starvation fix) so it must NOT share the legacy
    # identity - a shared fingerprint would invite guaranteed prefix misses.
    assert "reasoning_history=preserve_echo" in legacy_preserve
    assert preserve != legacy_preserve
    assert scoped != preserve


# ---------------------------------------------------------------------------
# _reasoning_parser_for_state: the parent-resolved args value is
# authoritative; the backend codec is only a fallback for states that carry
# no parser. The old backend-wins-on-mismatch rule silently discarded an
# operator's --reasoning-parser on shared lanes whose codec pins "none"
# (llama-ar), which force-closed thinking with no possible override.
# ---------------------------------------------------------------------------


def _parser_state(parser_value, *, backend_parser="none", omit_attr=False):
    from mtplx.server.openai import _reasoning_parser_for_state  # noqa: F401

    args = SimpleNamespace() if omit_attr else SimpleNamespace(
        reasoning_parser=parser_value
    )
    backend = SimpleNamespace(
        reasoning_codec=SimpleNamespace(parser=backend_parser)
    )
    return SimpleNamespace(args=args, backend_descriptor=backend)


def test_operator_parser_wins_over_backend_none_codec():
    from mtplx.server.openai import _reasoning_parser_for_state

    state = _parser_state("qwen3", backend_parser="none")
    assert _reasoning_parser_for_state(state) == "qwen3"


def test_explicit_none_parser_stays_none():
    from mtplx.server.openai import _reasoning_parser_for_state

    state = _parser_state("none", backend_parser="qwen3")
    assert _reasoning_parser_for_state(state) == "none"


def test_missing_parser_attr_falls_back_to_backend_codec():
    from mtplx.server.openai import _reasoning_parser_for_state

    state = _parser_state(None, backend_parser="lfm2", omit_attr=True)
    assert _reasoning_parser_for_state(state) == "lfm2"


def test_null_parser_value_falls_back_to_backend_codec():
    from mtplx.server.openai import _reasoning_parser_for_state

    state = _parser_state(None, backend_parser="step3p5")
    assert _reasoning_parser_for_state(state) == "step3p5"
