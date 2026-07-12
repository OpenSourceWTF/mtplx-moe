import XCTest
@testable import MTPLXAppCore

// Covers the 2026-07-02 chat-UX pass: SourceRecord extraction/dedupe,
// transcript turn-grouping (incl. legacy nil-groupID fallback), the
// combined-reasoning fold, and ChatTurnStats backward compatibility.

final class ChatTurnGroupingTests: XCTestCase {

    // MARK: - SourceRecord extraction

    func testWebSearchResultExtraction() throws {
        let result = """
        {"results": [
          {"url": "https://www.anthropic.com/news/claude", "title": "Claude"},
          {"url": "https://openai.com/blog/gpt", "title": "GPT"},
          {"url": "not a url"},
          {"title": "no url at all"}
        ]}
        """
        let records = SourceRecord.extract(
            toolName: "web_search",
            argumentsJSON: #"{"query": "claude vs gpt"}"#,
            resultJSON: result
        )
        XCTAssertEqual(records.count, 2)
        XCTAssertEqual(records[0].domain, "anthropic.com")
        XCTAssertEqual(records[0].title, "Claude")
        XCTAssertEqual(records[1].domain, "openai.com")
    }

    func testFetchUrlExtractionPrefersArgumentsUrlAndResultTitle() throws {
        let records = SourceRecord.extract(
            toolName: "fetch_url",
            argumentsJSON: #"{"url": "https://example.com/page"}"#,
            resultJSON: #"{"title": "Example Page"}"#
        )
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records[0].url, "https://example.com/page")
        XCTAssertEqual(records[0].title, "Example Page")
        XCTAssertEqual(records[0].domain, "example.com")
    }

    func testExtractionToleratesMalformedJSON() throws {
        XCTAssertTrue(
            SourceRecord.extract(
                toolName: "web_search",
                argumentsJSON: nil,
                resultJSON: "{not json"
            ).isEmpty
        )
        XCTAssertTrue(
            SourceRecord.extract(
                toolName: "unknown_tool",
                argumentsJSON: nil,
                resultJSON: #"{"results": []}"#
            ).isEmpty
        )
    }

    func testDedupeNormalizesSchemeWwwAndTrailingSlash() throws {
        let records = [
            SourceRecord(url: "https://www.anthropic.com/news/", title: "", domain: "anthropic.com"),
            SourceRecord(url: "http://anthropic.com/news", title: "News", domain: "anthropic.com"),
            SourceRecord(url: "https://openai.com", title: "OpenAI", domain: "openai.com"),
        ]
        let deduped = SourceRecord.dedupe(records)
        XCTAssertEqual(deduped.count, 2)
        // First-seen record wins, but inherits the later duplicate's
        // title when it had none.
        XCTAssertEqual(deduped[0].url, "https://www.anthropic.com/news/")
        XCTAssertEqual(deduped[0].title, "News")
    }

    func testSourcesJSONRoundTrip() throws {
        let records = [
            SourceRecord(url: "https://a.com", title: "A", domain: "a.com"),
            SourceRecord(url: "https://b.com", title: "", domain: "b.com"),
        ]
        let json = try XCTUnwrap(SourceRecord.encodeJSON(records))
        XCTAssertEqual(SourceRecord.decodeJSON(json), records)
        XCTAssertNil(SourceRecord.encodeJSON([]))
        XCTAssertTrue(SourceRecord.decodeJSON(nil).isEmpty)
        XCTAssertTrue(SourceRecord.decodeJSON("{broken").isEmpty)
    }

    // MARK: - Turn grouping

    func testConsecutiveAssistantMessagesWithSharedGroupIDFoldIntoOneItem() throws {
        let turnID = UUID()
        let user = ChatMessage(role: .user, visibleContent: "compare the models")
        let round1 = ChatMessage(
            role: .assistant,
            visibleContent: "Let me search for that.",
            reasoningContent: "Need fresh benchmarks.",
            toolCallsJSON: #"[{"id":"c1","name":"web_search","arguments":"{}"}]"#,
            finishReason: "tool_calls",
            turnGroupID: turnID
        )
        let toolResult = ChatMessage(
            role: .tool,
            visibleContent: #"{"results": []}"#,
            toolCallId: "c1",
            turnGroupID: turnID
        )
        let final = ChatMessage(
            role: .assistant,
            visibleContent: "Here's the comparison.",
            reasoningContent: "Synthesizing sources.",
            finishReason: "stop",
            turnGroupID: turnID
        )
        let items = ChatTranscriptGrouping.items(from: [user, round1, toolResult, final])

        XCTAssertEqual(items.count, 2)
        guard case .user(let u) = items[0] else { return XCTFail("expected user item") }
        XCTAssertEqual(u.id, user.id)
        guard case .assistantTurn(let group) = items[1] else {
            return XCTFail("expected assistant turn")
        }
        XCTAssertEqual(group.id, turnID)
        XCTAssertEqual(group.members.map(\.id), [round1.id, final.id])
        XCTAssertEqual(group.finalMessage.id, final.id)
        XCTAssertFalse(group.isSingleton)
        // Intermediate narration joins the reasoning stream, in order;
        // the final answer does NOT.
        XCTAssertEqual(
            group.combinedReasoning,
            "Need fresh benchmarks.\n\nLet me search for that.\n\nSynthesizing sources."
        )
    }

    func testLegacyMessagesWithoutGroupIDStaySingletons() throws {
        let a = ChatMessage(role: .assistant, visibleContent: "old turn one")
        let b = ChatMessage(role: .assistant, visibleContent: "old turn two")
        let items = ChatTranscriptGrouping.items(from: [a, b])
        XCTAssertEqual(items.count, 2)
        for (item, message) in zip(items, [a, b]) {
            guard case .assistantTurn(let group) = item else {
                return XCTFail("expected assistant turn")
            }
            XCTAssertTrue(group.isSingleton)
            XCTAssertEqual(group.id, message.id)
        }
    }

    func testDistinctGroupIDsDoNotMerge() throws {
        let first = ChatMessage(
            role: .assistant, visibleContent: "answer one", turnGroupID: UUID()
        )
        let second = ChatMessage(
            role: .assistant, visibleContent: "answer two", turnGroupID: UUID()
        )
        let items = ChatTranscriptGrouping.items(from: [first, second])
        XCTAssertEqual(items.count, 2)
    }

    func testUserMessageClosesAnOpenGroup() throws {
        let turnID = UUID()
        let round1 = ChatMessage(
            role: .assistant, visibleContent: "", finishReason: "tool_calls",
            turnGroupID: turnID
        )
        let user = ChatMessage(role: .user, visibleContent: "actually stop")
        // Same group id APPEARING after an interposed user message must
        // not merge backwards (defensive; the loop never produces this).
        let stray = ChatMessage(
            role: .assistant, visibleContent: "answer", turnGroupID: turnID
        )
        let items = ChatTranscriptGrouping.items(from: [round1, user, stray])
        XCTAssertEqual(items.count, 3)
    }

    func testGroupSourcesPreferPersistedJSONOverTraceDerivation() throws {
        let turnID = UUID()
        let persisted = [
            SourceRecord(url: "https://a.com", title: "A", domain: "a.com")
        ]
        let final = ChatMessage(
            role: .assistant,
            visibleContent: "answer",
            turnGroupID: turnID,
            sourcesJSON: SourceRecord.encodeJSON(persisted)
        )
        let group = AssistantTurnGroup(id: turnID, members: [final])
        XCTAssertEqual(group.sources, persisted)
    }

    // MARK: - Stats backward compatibility

    func testChatTurnStatsDecodesLegacyJSONWithoutThinkingTime() throws {
        let legacy = #"{"rawDecodeTokS": 61.5, "completionTokens": 420}"#
        let stats = try JSONDecoder().decode(
            ChatTurnStats.self, from: Data(legacy.utf8)
        )
        XCTAssertEqual(stats.rawDecodeTokS, 61.5)
        XCTAssertNil(stats.thinkingTimeMs)
    }
}

// MARK: - Streaming markdown block safety (2026-07-03 turbo release)

final class StreamingMarkdownBlockSafetyTests: XCTestCase {

    func testLastBlockIsAlwaysUnsafe() {
        XCTAssertEqual(StreamingMarkdownBlockSafety.classify(["hello"]), [false])
        XCTAssertEqual(
            StreamingMarkdownBlockSafety.classify(["# Done", "growing tail"]),
            [true, false]
        )
    }

    func testFenceInteriorBlocksStayPlainUntilClosed() {
        // Block 0 opens a fence, block 1 is interior, block 2 closes it,
        // block 3 grows. Nothing before the close is markdown-safe.
        let flags = StreamingMarkdownBlockSafety.classify([
            "```python\ndef f():",
            "    return 1",
            "```",
            "And that's how",
        ])
        XCTAssertEqual(flags, [false, false, false, false])
    }

    func testSelfContainedFencedBlockIsSafeOnceFrozen() {
        let flags = StreamingMarkdownBlockSafety.classify([
            "```python\nprint(1)\n```",
            "closing prose",
            "tail",
        ])
        XCTAssertEqual(flags, [true, true, false])
    }

    func testEmptyInput() {
        XCTAssertEqual(StreamingMarkdownBlockSafety.classify([]), [])
    }
}
