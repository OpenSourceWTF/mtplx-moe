import Foundation

// MARK: - AssistantTurnGroup
//
// The transcript's render unit for one logical assistant TURN. A turn
// with web search persists as several ChatMessages (assistant-with-
// tool_calls, tool results, ... , final assistant answer); rendering
// them individually is what produced the "wall of thinking bubbles"
// the 2026-07-02 chat-UX pass removed. This grouping folds a whole
// turn back into: one combined reasoning stream, one set of tool
// traces, one answer message, one deduped sources list.
//
// Grouping is presentation-only — persistence and the request wire
// shape are untouched, so replays/retries see exactly what the model
// actually emitted.

public struct AssistantTurnGroup: Identifiable, Equatable {
    /// Stable identity for SwiftUI: the shared turnGroupID when the
    /// turn was persisted by the grouped-loop code, else the (single)
    /// message's own id for legacy rows.
    public let id: UUID
    /// Every assistant message of the turn, oldest first. The last one
    /// is the user-facing answer.
    public let members: [ChatMessage]

    public init(id: UUID, members: [ChatMessage]) {
        self.id = id
        self.members = members
    }

    public var finalMessage: ChatMessage { members[members.count - 1] }

    /// True when this group is a plain single-message turn (no tool
    /// rounds) — the renderer can keep the exact legacy layout.
    public var isSingleton: Bool { members.count == 1 }

    /// All reasoning the model produced across the turn, in order.
    /// Intermediate rounds' visible narration ("Let me search for…")
    /// is process talk, not the answer, so it joins the thinking
    /// stream rather than rendering as a stray half-answer bubble.
    public var combinedReasoning: String {
        var parts: [String] = []
        for (index, message) in members.enumerated() {
            if let reasoning = message.reasoningContent?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                !reasoning.isEmpty
            {
                parts.append(reasoning)
            }
            let isFinal = index == members.count - 1
            if !isFinal {
                let narration = message.visibleContent
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !narration.isEmpty {
                    parts.append(narration)
                }
            }
        }
        return parts.joined(separator: "\n\n")
    }

    /// Every tool trace of the turn, oldest first.
    public var traces: [ToolTraceRecord] {
        members
            .flatMap { $0.toolTraces }
            .sorted { $0.startedAt < $1.startedAt }
    }

    /// Deduped sources for the footer. Prefers the JSON persisted on
    /// the final message (grouped-loop turns); falls back to deriving
    /// from tool traces so pre-existing conversations gain the footer
    /// retroactively.
    public var sources: [SourceRecord] {
        let persisted = SourceRecord.decodeJSON(finalMessage.sourcesJSON)
        if !persisted.isEmpty { return persisted }
        let derived = traces.flatMap { trace in
            SourceRecord.extract(
                toolName: trace.name,
                argumentsJSON: trace.argumentsJSON,
                resultJSON: trace.resultJSON
            )
        }
        return SourceRecord.dedupe(derived)
    }

    /// Total think time across the turn, from the final message's
    /// stats (written by the grouped loop). Nil for legacy turns.
    public var thinkingTimeMs: Int? {
        guard let json = finalMessage.statsJSON,
            let data = json.data(using: .utf8),
            let stats = try? JSONDecoder().decode(ChatTurnStats.self, from: data)
        else { return nil }
        return stats.thinkingTimeMs
    }

    /// The searches the turn ran, for the compact activity chip
    /// ("Searched: <query>" lines). Order preserved.
    public var searchQueries: [String] {
        traces.compactMap { trace in
            guard trace.name == "web_search",
                let json = trace.argumentsJSON,
                let data = json.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let query = dict["query"] as? String,
                !query.isEmpty
            else { return nil }
            return query
        }
    }

    public var fetchedPageCount: Int {
        traces.filter { $0.name == "fetch_url" }.count
    }

    public static func == (lhs: AssistantTurnGroup, rhs: AssistantTurnGroup) -> Bool {
        lhs.id == rhs.id && lhs.members.map(\.id) == rhs.members.map(\.id)
    }
}

// MARK: - ChatTranscriptItem

/// One row of the conversation column after grouping.
public enum ChatTranscriptItem: Identifiable, Equatable {
    case user(ChatMessage)
    case assistantTurn(AssistantTurnGroup)

    public var id: UUID {
        switch self {
        case .user(let message): return message.id
        case .assistantTurn(let group): return group.id
        }
    }

    public static func == (lhs: ChatTranscriptItem, rhs: ChatTranscriptItem) -> Bool {
        switch (lhs, rhs) {
        case (.user(let a), .user(let b)):
            return a.id == b.id
        case (.assistantTurn(let a), .assistantTurn(let b)):
            return a == b
        default:
            return false
        }
    }
}

public enum ChatTranscriptGrouping {
    /// Fold an ordered message list into transcript items. CONSECUTIVE
    /// assistant messages sharing a non-nil `turnGroupID` collapse into
    /// one `AssistantTurnGroup`; assistant messages without a group id
    /// (every message persisted before 2026-07-02) become singleton
    /// groups, so old conversations render exactly as they used to.
    /// Tool/system rows never render and never split a group.
    ///
    /// `excludingTurnGroupID` drops the IN-FLIGHT turn's already-
    /// persisted rounds: while the tool loop is streaming, the live
    /// surface is the turn's ONE representation, so its partial rounds
    /// must not also render as a settled bubble above it. The moment
    /// streaming ends the exclusion lifts and the whole turn renders
    /// settled.
    public static func items(
        from messages: [ChatMessage],
        excludingTurnGroupID excluded: UUID? = nil
    ) -> [ChatTranscriptItem] {
        var items: [ChatTranscriptItem] = []
        items.reserveCapacity(messages.count)
        var openGroupID: UUID?
        var openMembers: [ChatMessage] = []

        func closeOpenGroup() {
            guard let groupID = openGroupID, !openMembers.isEmpty else {
                openGroupID = nil
                openMembers = []
                return
            }
            items.append(
                .assistantTurn(AssistantTurnGroup(id: groupID, members: openMembers))
            )
            openGroupID = nil
            openMembers = []
        }

        for message in messages {
            switch message.role {
            case .tool, .system:
                // Invisible plumbing rows; a tool result BETWEEN two
                // assistant rounds must not break the group.
                continue
            case .user:
                closeOpenGroup()
                items.append(.user(message))
            case .assistant:
                if let excluded, message.turnGroupID == excluded {
                    continue
                }
                if let groupID = message.turnGroupID {
                    if openGroupID == groupID {
                        openMembers.append(message)
                    } else {
                        closeOpenGroup()
                        openGroupID = groupID
                        openMembers = [message]
                    }
                } else {
                    closeOpenGroup()
                    items.append(
                        .assistantTurn(
                            AssistantTurnGroup(id: message.id, members: [message])
                        )
                    )
                }
            }
        }
        closeOpenGroup()
        return items
    }
}
