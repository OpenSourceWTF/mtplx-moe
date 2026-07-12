import XCTest
@testable import MTPLXAppCore

// Covers the 2026-07-03 chat-UX redesign: TurnActivityModel chip
// composition for the live and settled activity strips, the
// phase→well auto-follow rules, and the transcript's exclusion of the
// in-flight turn's persisted rounds (the live surface is the turn's
// one representation while streaming).

final class TurnActivityModelTests: XCTestCase {

    // MARK: - Live chips

    func testPreTokenThinkingShowsOnlyLiveThoughtChip() {
        let model = TurnActivityModel.live(phase: .thinking, hasReasoning: false, traces: [])
        XCTAssertEqual(model.chips.count, 1)
        XCTAssertEqual(model.chips[0].kind, .thought)
        XCTAssertEqual(model.chips[0].label, "Thinking")
        XCTAssertTrue(model.chips[0].isLive)
        XCTAssertNil(model.chips[0].caption)
    }

    func testReasoningDisabledPreTokenShowsGeneratingChip() {
        let model = TurnActivityModel.live(phase: .generating, hasReasoning: false, traces: [])
        XCTAssertEqual(model.chips.map(\.label), ["Generating"])
        XCTAssertTrue(model.chips[0].isLive)
    }

    func testSearchingShowsBothChipsWithLiveSearch() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "web_search", status: .pending)
        ]
        let model = TurnActivityModel.live(phase: .searching, hasReasoning: true, traces: traces)
        XCTAssertEqual(model.chips.map(\.kind), [.thought, .search])
        // Thought settles while the search is the live activity.
        XCTAssertEqual(model.chips[0].label, "Thought")
        XCTAssertFalse(model.chips[0].isLive)
        XCTAssertEqual(model.chips[1].label, "Searching")
        XCTAssertTrue(model.chips[1].isLive)
    }

    func testBackToThinkingSettlesSearchChipWithCount() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "web_search", status: .success),
            PendingToolTrace(id: "r1-c2", name: "web_search", status: .success),
            PendingToolTrace(id: "r1-c3", name: "web_search", status: .success),
        ]
        let model = TurnActivityModel.live(phase: .thinking, hasReasoning: true, traces: traces)
        XCTAssertEqual(model.chips.map(\.kind), [.thought, .search])
        XCTAssertTrue(model.chips[0].isLive)
        XCTAssertEqual(model.chips[1].label, "Searched")
        XCTAssertEqual(model.chips[1].caption, "×3")
        XCTAssertFalse(model.chips[1].isLive)
    }

    func testAnsweringSettlesEveryChip() {
        let traces = [
            PendingToolTrace(id: "r1-c1", name: "web_search", status: .success)
        ]
        let model = TurnActivityModel.live(phase: .answering, hasReasoning: true, traces: traces)
        XCTAssertEqual(model.chips.map(\.isLive), [false, false])
        XCTAssertEqual(model.chips[0].label, "Thought")
        XCTAssertEqual(model.chips[1].label, "Searched")
        XCTAssertNil(model.chips[1].caption)
    }

    func testPlainNonReasoningAnswerHasNoChips() {
        let model = TurnActivityModel.live(phase: .answering, hasReasoning: false, traces: [])
        XCTAssertTrue(model.isEmpty)
    }

    // MARK: - Settled chips

    func testSettledThoughtAndSearchLabels() {
        let model = TurnActivityModel.settled(
            hasThought: true,
            thinkingTimeMs: 12_400,
            searchCount: 3,
            fetchedPageCount: 0,
            hasOtherToolActivity: true
        )
        XCTAssertEqual(model.chips.map(\.kind), [.thought, .search])
        XCTAssertEqual(model.chips[0].label, "Thought")
        XCTAssertEqual(model.chips[0].caption, "12.4s")
        XCTAssertEqual(model.chips[1].label, "Searched")
        XCTAssertEqual(model.chips[1].caption, "×3")
    }

    func testSettledSingleSearchOmitsCountCaption() {
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: 1,
            fetchedPageCount: 2,
            hasOtherToolActivity: true
        )
        XCTAssertEqual(model.chips.map(\.kind), [.search])
        XCTAssertEqual(model.chips[0].label, "Searched")
        XCTAssertNil(model.chips[0].caption)
    }

    func testSettledFetchOnlyTurnReadsAsPages() {
        let model = TurnActivityModel.settled(
            hasThought: true,
            thinkingTimeMs: 640,
            searchCount: 0,
            fetchedPageCount: 2,
            hasOtherToolActivity: true
        )
        XCTAssertEqual(model.chips[0].caption, "640 ms")
        XCTAssertEqual(model.chips[1].label, "Read pages")
        XCTAssertEqual(model.chips[1].caption, "×2")
    }

    func testSettledUnknownToolFallsBackToUsedTools() {
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: 0,
            fetchedPageCount: 0,
            hasOtherToolActivity: true
        )
        XCTAssertEqual(model.chips.map(\.label), ["Used tools"])
    }

    func testSettledPlainTurnIsEmpty() {
        let model = TurnActivityModel.settled(
            hasThought: false,
            thinkingTimeMs: nil,
            searchCount: 0,
            fetchedPageCount: 0,
            hasOtherToolActivity: false
        )
        XCTAssertTrue(model.isEmpty)
    }

    // MARK: - Auto-follow

    func testAutoDetailFollowsActiveTool() {
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .thinking), .thought)
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .generating), .thought)
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .searching), .search)
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .reading), .search)
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .answering), TurnActivityModel.Detail.none)
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .finalizing), TurnActivityModel.Detail.none)
        XCTAssertEqual(TurnActivityModel.autoDetail(for: .idle), TurnActivityModel.Detail.none)
    }

    // MARK: - In-flight turn exclusion from the settled transcript

    func testGroupingExcludesInFlightTurnRounds() {
        let liveTurnID = UUID()
        let settledTurnID = UUID()
        let earlierUser = ChatMessage(role: .user, visibleContent: "earlier question")
        let earlierTurn = ChatMessage(
            role: .assistant, visibleContent: "earlier answer", turnGroupID: settledTurnID
        )
        let user = ChatMessage(role: .user, visibleContent: "compare the models")
        let liveRound1 = ChatMessage(
            role: .assistant,
            visibleContent: "",
            reasoningContent: "Need fresh benchmarks.",
            finishReason: "tool_calls",
            turnGroupID: liveTurnID
        )
        let liveToolResult = ChatMessage(
            role: .tool,
            visibleContent: #"{"results": []}"#,
            toolCallId: "c1",
            turnGroupID: liveTurnID
        )
        let messages = [earlierUser, earlierTurn, user, liveRound1, liveToolResult]

        let streaming = ChatTranscriptGrouping.items(
            from: messages, excludingTurnGroupID: liveTurnID
        )
        // Earlier turn + both user messages render; the in-flight
        // turn's persisted rounds do NOT (the live surface shows them).
        XCTAssertEqual(streaming.count, 3)
        guard case .assistantTurn(let visibleGroup) = streaming[1] else {
            return XCTFail("expected earlier assistant turn")
        }
        XCTAssertEqual(visibleGroup.id, settledTurnID)
        guard case .user(let lastUser) = streaming[2] else {
            return XCTFail("expected trailing user message")
        }
        XCTAssertEqual(lastUser.id, user.id)

        // Exclusion lifted (turn finished): the whole turn renders.
        let settled = ChatTranscriptGrouping.items(from: messages)
        XCTAssertEqual(settled.count, 4)
        guard case .assistantTurn(let group) = settled[3] else {
            return XCTFail("expected in-flight turn to render once settled")
        }
        XCTAssertEqual(group.id, liveTurnID)
    }
}
