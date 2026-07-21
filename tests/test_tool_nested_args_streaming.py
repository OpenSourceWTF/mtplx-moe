"""Issue #170: nested edit_file arguments intermittently collapse to {}.

Root-caused streaming/final parser contract asymmetry: the streaming Qwen-XML
tool parser silently discarded any function-body text that was not wrapped in
<parameter=> blocks, while the final (non-stream) parser raises a protocol
error for the same input. A model that emits the mixed form

    <tool_call>
    <function=edit_file>
    {"path": "...", "edits": [{"search": "...", "replace": "..."}]}
    </function>
    </tool_call>

(the natural slip on deeply nested payloads for a family trained on both the
XML and the JSON tool dialects) therefore produced a schema-valid call with
EMPTY arguments on requiredless tools, and a silently-vanished call on tools
with required fields. Both are the measured #170 shapes.

Contract after the fix, identical for the streaming and the final parser:
- a function body that is a single JSON object and has no <parameter=> blocks
  IS the arguments payload (unambiguous model intent, normal schema validation);
- any other unwrapped, lead, or trailing non-whitespace text is a loud
  protocol fallback, never a silent drop.
"""

import json

import pytest
from fastapi import HTTPException

from mtplx.server.openai import (
    _QwenXMLToolCallStreamParser,
    _ToolAwareContentStreamTranslator,
    _parse_generated_tool_calls,
    _tool_call_example,
)


# The exact asiai #170 suite schema: edits is the array-of-objects probe.
EDIT_FILE_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search": {"type": "string"},
                                "replace": {"type": "string"},
                            },
                            "required": ["search", "replace"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    }
]

# A requiredless tool: schema validation cannot save us here, so the silent
# streaming drop used to deliver arguments == {} to the client (the literal
# count_empty_object_bug shape).
LOOKUP_TOOL_SPECS = [
    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
]

# Adversarial payload: braces and quotes inside string values, three nested
# objects — the #170 "add three fields" turn shape.
NESTED_EDITS = [
    {"search": "retries = 3", "replace": 'retries = int(os.environ.get("RETRIES", "3"))'},
    {"search": "backoff = 1.0", "replace": 'backoff = {"base": 1.0, "max": 30.0}'},
    {"search": "tls = False", "replace": "tls = True"},
]
NESTED_ARGUMENTS = {"path": "config.py", "edits": NESTED_EDITS}

JSON_BODY_CALL = (
    "<tool_call>\n"
    "<function=edit_file>\n"
    + json.dumps(NESTED_ARGUMENTS, ensure_ascii=False)
    + "\n</function>\n</tool_call>"
)


def _make(tools):
    return _ToolAwareContentStreamTranslator(
        tools=tools,
        argument_chunk_chars=64,
        tokenizer=None,
    )


def _argument_text(deltas):
    return "".join(
        item.get("function", {}).get("arguments", "")
        for delta in deltas
        for item in delta.get("tool_calls", [])
    )


def _content_text(deltas):
    return "".join(delta.get("content", "") for delta in deltas)


def _feed_bytewise(translator, text):
    deltas = []
    for ch in text:
        deltas.extend(translator.feed("content", ch))
    deltas.extend(translator.finish())
    return deltas


# ---------- the #170 mixed form: JSON object body inside <function=> ----------

def test_json_body_in_function_envelope_streams_nested_args():
    t = _make(EDIT_FILE_TOOL_SPECS)
    deltas = t.feed("content", JSON_BODY_CALL)
    deltas.extend(t.finish())
    assert t.has_tool_calls is True, t.fallback_reason
    assert json.loads(_argument_text(deltas)) == NESTED_ARGUMENTS
    assert _content_text(deltas) == ""


def test_json_body_in_function_envelope_byte_stream():
    """Marker boundaries split at every possible position."""
    t = _make(EDIT_FILE_TOOL_SPECS)
    deltas = _feed_bytewise(t, JSON_BODY_CALL)
    assert t.has_tool_calls is True, t.fallback_reason
    assert json.loads(_argument_text(deltas)) == NESTED_ARGUMENTS


def test_json_body_in_function_envelope_final_parser():
    """Non-stream parity: the final parser accepts the same envelope."""
    calls = _parse_generated_tool_calls(
        JSON_BODY_CALL, tools=EDIT_FILE_TOOL_SPECS
    )
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "edit_file"
    assert json.loads(calls[0]["function"]["arguments"]) == NESTED_ARGUMENTS


def test_json_body_requiredless_tool_never_collapses_to_empty_args():
    """The literal count_empty_object_bug lane: before the fix the streaming
    parser dropped the JSON body and delivered arguments == {}."""
    payload = {"query": "hybrid cache", "filters": [{"kind": "code"}]}
    text = (
        "<tool_call>\n<function=lookup>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</function>\n</tool_call>"
    )
    t = _make(LOOKUP_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is True, t.fallback_reason
    arguments = json.loads(_argument_text(deltas))
    assert arguments == payload, (
        f"arguments must carry the model's payload, got {arguments!r}"
    )


def test_multiple_calls_second_with_json_body():
    text = (
        "<tool_call>\n<function=lookup>\n"
        "<parameter=query>\nwhere is Config\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=lookup>\n"
        + json.dumps({"query": "rename Config"}, ensure_ascii=False)
        + "\n</function>\n</tool_call>"
    )
    t = _make(LOOKUP_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.tool_calls is not None and len(t.tool_calls) == 2
    assert json.loads(t.tool_calls[0]["function"]["arguments"]) == {
        "query": "where is Config"
    }
    assert json.loads(t.tool_calls[1]["function"]["arguments"]) == {
        "query": "rename Config"
    }


# ---------- silent-drop lanes become loud (parity with the final parser) ------

def test_unwrapped_garbage_body_falls_back_loud():
    """Non-JSON unwrapped body: the final parser raises 'unwrapped parameter
    text'; streaming must fall back (no call), never deliver empty args."""
    text = (
        "<tool_call>\n<function=lookup>\n"
        "just prose, not a payload\n"
        "</function>\n</tool_call>"
    )
    t = _make(LOOKUP_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is False
    assert t.fallback_reason
    assert _argument_text(deltas) == ""
    with pytest.raises(HTTPException):
        _parse_generated_tool_calls(text, tools=LOOKUP_TOOL_SPECS)


def test_lead_text_before_parameter_falls_back_loud():
    """Text between <function=> and the first <parameter=> used to be silently
    discarded; the surviving call then carried partial arguments."""
    text = (
        "<tool_call>\n<function=lookup>\n"
        "stray words\n"
        "<parameter=query>\nhello\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    t = _make(LOOKUP_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is False
    assert t.fallback_reason
    assert _argument_text(deltas) == ""
    with pytest.raises(HTTPException):
        _parse_generated_tool_calls(text, tools=LOOKUP_TOOL_SPECS)


def test_junk_between_function_close_and_tool_close_falls_back():
    """Text between </function> and </tool_call> used to be silently dropped
    while the final parser rejects the same envelope."""
    text = (
        "<tool_call>\n<function=lookup>\n"
        "<parameter=query>\nhello\n</parameter>\n"
        "</function>\nleftover\n</tool_call>"
    )
    t = _make(LOOKUP_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is False
    assert t.fallback_reason
    assert _argument_text(deltas) == ""
    with pytest.raises(HTTPException):
        _parse_generated_tool_calls(text, tools=LOOKUP_TOOL_SPECS)


def test_mixed_parameter_blocks_and_json_residue_stays_loud():
    """Ambiguous mix (parameter blocks + stray JSON) stays a protocol error in
    both parsers — JSON-body acceptance applies only to the pure-body form."""
    text = (
        "<tool_call>\n<function=lookup>\n"
        "<parameter=query>\nhello\n</parameter>\n"
        '{"query": "shadow"}\n'
        "</function>\n</tool_call>"
    )
    t = _make(LOOKUP_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is False
    assert t.fallback_reason
    assert _argument_text(deltas) == ""
    with pytest.raises(HTTPException):
        _parse_generated_tool_calls(text, tools=LOOKUP_TOOL_SPECS)


# ---------- canonical form must not move ----------

def test_canonical_parameter_form_with_nested_array_unchanged():
    edits_json = json.dumps(NESTED_EDITS, ensure_ascii=False)
    text = (
        "<tool_call>\n<function=edit_file>\n"
        "<parameter=path>\nconfig.py\n</parameter>\n"
        f"<parameter=edits>\n{edits_json}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    t = _make(EDIT_FILE_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is True, t.fallback_reason
    assert json.loads(_argument_text(deltas)) == NESTED_ARGUMENTS

    calls = _parse_generated_tool_calls(text, tools=EDIT_FILE_TOOL_SPECS)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == NESTED_ARGUMENTS


def test_empty_required_array_is_schema_legal_and_passes():
    """edits: [] satisfies the declared schema (type array, present). The
    server must not invent a minItems policy; the exemplar fix addresses the
    degenerate-emission side (see _tool_call_example tests)."""
    text = (
        "<tool_call>\n<function=edit_file>\n"
        "<parameter=path>\nconfig.py\n</parameter>\n"
        "<parameter=edits>\n[]\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    t = _make(EDIT_FILE_TOOL_SPECS)
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is True
    assert json.loads(_argument_text(deltas)) == {"path": "config.py", "edits": []}


# ---------- streaming parser unit surface (kept independent of translator) ---

def test_stream_parser_json_body_direct():
    p = _QwenXMLToolCallStreamParser(tools=EDIT_FILE_TOOL_SPECS)
    deltas = p.feed(
        "<function=edit_file>\n"
        + json.dumps(NESTED_ARGUMENTS, ensure_ascii=False)
        + "\n</function>\n</tool_call>"
    )
    deltas.extend(p.finish())
    assert p.fallback_reason is None
    assert p.tool_calls is not None
    assert json.loads(p.tool_calls[0]["function"]["arguments"]) == NESTED_ARGUMENTS


# ---------- exemplar fix: no degenerate [] / {} examples for nested schemas --

def test_tool_call_example_populates_array_of_objects():
    example = _tool_call_example(EDIT_FILE_TOOL_SPECS)
    assert "<parameter=edits>" in example
    assert "\n[]\n" not in example, "degenerate empty-array exemplar (#170)"
    assert '"search"' in example and '"replace"' in example


def test_tool_call_example_populates_plain_array():
    specs = [
        {
            "type": "function",
            "function": {
                "name": "question",
                "parameters": {
                    "type": "object",
                    "properties": {"questions": {"type": "array"}},
                    "required": ["questions"],
                },
            },
        }
    ]
    example = _tool_call_example(specs)
    assert "\n[]\n" not in example
    assert '["ARGUMENT_VALUE"]' in example


def test_tool_call_example_string_params_unchanged():
    specs = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        }
    ]
    example = _tool_call_example(specs)
    assert "<parameter=path>\nARGUMENT_VALUE\n</parameter>" in example
