"""Issue #160: raw tool-call protocol markup must not reach no-tools chats.

Small models answer "look it up" prompts by emitting their trained tool
format even when the request declares no tools; the app chat then rendered
raw `<tool_call><function=web_search>...` XML (often UNCLOSED — the
reporter's transcript opened `<tool_call>` twice and never closed it).
"""

import mtplx.server.openai as srv


def test_strip_well_formed_block():
    text = "Let me check.\n<tool_call>\n{\"name\": \"web_search\"}\n</tool_call>\nDone."
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 1
    assert "<tool_call>" not in cleaned
    assert "Let me check." in cleaned and "Done." in cleaned


def test_strip_unclosed_block_reporter_shape():
    # The exact #160 shape: two openers, function/parameter body, no closer.
    text = (
        "Let me search Wikipedia directly for accurate specs.\n\n"
        "<tool_call>\n<function=web_search>\n<parameter=query>\n"
        "site:wikipedia.org Apple M1 M2 memory bandwidth\n"
        "</parameter>\n</function>\n<tool_call>"
    )
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count >= 1
    assert "<tool_call" not in cleaned
    assert "<function=" not in cleaned
    assert cleaned.startswith("Let me search Wikipedia")


def test_strip_preserves_code_fences():
    text = (
        "Here is the syntax you asked about:\n"
        "```xml\n<tool_call>\n{\"name\": \"demo\"}\n</tool_call>\n```\n"
        "That is how agents call tools."
    )
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 0
    assert cleaned == text


def test_strip_no_markup_untouched():
    text = "Plain answer with a < comparison and no protocol tags."
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 0
    assert cleaned == text


def test_stream_splitter_suppresses_orphan_spans():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=False,
        suppress_orphan_tool_markup=True,
    )
    chunks = []
    for piece in (
        "Checking now.\n",
        "<tool_call>\n<function=web_search>q</function>\n",
        "</tool_call>",
        "\nAll done.",
    ):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    content = "".join(text for field, text in chunks if field == "content")
    assert "<tool_call" not in content
    assert "web_search" not in content
    assert "Checking now." in content
    assert "All done." in content
    assert splitter.suppressed_tool_markup_chars > 0


def test_stream_splitter_passthrough_when_tools_active():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=False,
        suppress_orphan_tool_markup=False,
    )
    chunks = []
    for piece in ("Hi ", "<tool_call>x</tool_call>", " bye"):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    content = "".join(text for field, text in chunks if field == "content")
    assert "<tool_call>x</tool_call>" in content


# --- Issue #349: suppressed tool calls must leave a visible, truthful ---
# --- notice instead of silence ("tool calls going out into the void") ---


def test_tool_markup_call_names_extracts_from_spans():
    text = (
        "Let me look.\n"
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls ~/Dev"}}\n'
        "</tool_call>\n"
        "<function=search_files>\n<parameter=query>readme</parameter>\n</function>\n"
    )
    assert srv._tool_markup_call_names(text) == ["terminal", "search_files"]


def test_tool_markup_call_names_ignores_prose_json():
    # A JSON object the model wrote as prose (outside any tool-call span)
    # must not be misread as a call.
    text = 'Here is the config: {"name": "value"} — nothing else.'
    assert srv._tool_markup_call_names(text) == []


def test_unknown_tool_name_from_reason():
    assert srv._unknown_tool_name_from_reason("unknown tool 'terminal'") == "terminal"
    assert srv._unknown_tool_name_from_reason("something else entirely") == ""


def test_no_tools_notice_is_truthful_and_names_tools():
    notice = srv._no_tools_unexecuted_call_notice(["terminal", "read_file"])
    assert notice
    assert '"terminal"' in notice
    assert '"read_file"' in notice
    assert "nothing was executed" in notice
    # Nameless turns still get a non-empty truthful notice.
    assert "nothing was executed" in srv._no_tools_unexecuted_call_notice([])


def test_unknown_tool_notice_names_available_tools():
    notice = srv._unknown_tool_unexecuted_call_notice(
        "terminal", ["web_search", "fetch_url"]
    )
    assert notice
    assert '"terminal"' in notice
    assert "web_search, fetch_url" in notice
    assert "was not executed" in notice
    # No declared tools at all still yields a truthful, non-empty notice.
    assert "available: none" in srv._unknown_tool_unexecuted_call_notice("x", [])


def test_append_notice_is_idempotent_and_never_empty():
    notice = srv._no_tools_unexecuted_call_notice(["ls"])
    once = srv._append_unexecuted_tool_notice("prose", notice)
    assert once.startswith("prose")
    assert notice in once
    # Idempotent: appending again does not duplicate the notice.
    assert srv._append_unexecuted_tool_notice(once, notice) == once
    # A turn with no salvageable prose becomes the notice, never "".
    assert srv._append_unexecuted_tool_notice("", notice) == notice
    assert srv._append_unexecuted_tool_notice("   \n", notice) == notice


def test_no_tools_turn_that_is_only_a_tool_call_yields_visible_truth():
    # The exact #349 shape: the whole turn is one dead tool call. Before
    # the fix, the strip returned "" and the user + model both got a blank.
    # This mirrors the non-stream call-site sequence (strip, then notice).
    text = '<tool_call>\n{"name": "ls", "arguments": {"path": "~/Dev"}}\n</tool_call>'
    names = srv._tool_markup_call_names(text)
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 1
    assert cleaned == ""
    final = srv._append_unexecuted_tool_notice(
        cleaned, srv._no_tools_unexecuted_call_notice(names)
    )
    assert final
    assert '"ls"' in final
    assert "nothing was executed" in final


def test_stream_suppression_counter_drives_the_notice_gate():
    # The streaming call site fires the notice when the splitter suppressed
    # markup on a no-tools turn; pin the counter + composed notice together.
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=False,
        suppress_orphan_tool_markup=True,
    )
    raw = '<tool_call>\n{"name": "search_files", "arguments": {"query": "x"}}\n</tool_call>'
    chunks = []
    for piece in (raw[: len(raw) // 2], raw[len(raw) // 2 :]):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    content = "".join(text for field, text in chunks if field == "content")
    assert "search_files" not in content
    assert splitter.suppressed_tool_markup_chars > 0
    notice = srv._no_tools_unexecuted_call_notice(srv._tool_markup_call_names(raw))
    assert '"search_files"' in notice
    assert "nothing was executed" in notice
