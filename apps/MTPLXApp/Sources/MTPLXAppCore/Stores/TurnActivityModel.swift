import Foundation

// MARK: - TurnActivityModel
//
// Pure presentation model for the assistant turn's activity strip —
// the row of equal-width chips ("Thought", "Searched the web") that
// sits above the answer, plus which detail well should be open. One
// model serves BOTH the live streaming surface and the settled
// transcript bubble, so the moment a turn finishes the strip doesn't
// change shape — only its captions settle (durations, ×N counts).
//
// Kept in AppCore (no SwiftUI) so chip composition and the
// phase→well auto-follow rules are unit-testable.

public struct TurnActivityModel: Equatable {

    /// Which detail well is open beneath the chip row. `.none` renders
    /// the strip as bare chips (the settled default, and the streaming
    /// state once answer tokens start).
    public enum Detail: Equatable {
        case none
        case thought
        case search
    }

    public struct Chip: Equatable, Identifiable {
        public enum Kind: String, Equatable {
            case thought
            case search

            public var detail: Detail {
                switch self {
                case .thought: return .thought
                case .search: return .search
                }
            }
        }

        public let kind: Kind
        public let systemName: String
        public let label: String
        /// Dim trailing annotation: "12.4s" on a settled thought chip,
        /// "×3" on a settled search chip. Nil while live.
        public let caption: String?
        /// Live chips pulse (indicator dots) and read brighter.
        public let isLive: Bool

        public var id: String { kind.rawValue }

        public init(
            kind: Kind,
            systemName: String,
            label: String,
            caption: String? = nil,
            isLive: Bool = false
        ) {
            self.kind = kind
            self.systemName = systemName
            self.label = label
            self.caption = caption
            self.isLive = isLive
        }
    }

    public let chips: [Chip]

    public var isEmpty: Bool { chips.isEmpty }

    public func hasChip(_ kind: Chip.Kind) -> Bool {
        chips.contains { $0.kind == kind }
    }

    public init(chips: [Chip]) {
        self.chips = chips
    }

    // MARK: - Live (in-flight turn)

    /// Chips for the streaming surface. The thought chip exists from
    /// the moment the request is in flight (so nothing pops in later);
    /// the search chip appears beside it when the first tool call of
    /// the turn dispatches and stays for the rest of the turn.
    public static func live(
        phase: StreamingPhase,
        hasReasoning: Bool,
        traces: [PendingToolTrace]
    ) -> TurnActivityModel {
        var chips: [Chip] = []

        let thoughtIsLive = phase == .thinking || phase == .generating
        if hasReasoning || thoughtIsLive {
            let label: String
            if thoughtIsLive {
                label = (phase == .generating && !hasReasoning) ? "Generating" : "Thinking"
            } else {
                label = "Thought"
            }
            chips.append(
                Chip(kind: .thought, systemName: "brain", label: label, isLive: thoughtIsLive)
            )
        }

        if !traces.isEmpty {
            let searches = traces.filter { $0.name == "web_search" }.count
            let fetches = traces.filter { $0.name == "fetch_url" }.count
            let searchIsLive = phase == .searching || phase == .reading
                || traces.contains { $0.status == .pending }
            let label: String
            var caption: String?
            if searchIsLive {
                label = phase == .reading ? "Reading" : "Searching"
            } else {
                (label, caption) = Self.settledSearchLabel(
                    searchCount: searches,
                    fetchedPageCount: fetches,
                    hasOtherToolActivity: true
                )
            }
            chips.append(
                Chip(
                    kind: .search,
                    systemName: searchIsLive && phase == .reading ? "doc.text" : "globe",
                    label: label,
                    caption: caption,
                    isLive: searchIsLive
                )
            )
        }

        return TurnActivityModel(chips: chips)
    }

    // MARK: - Settled (persisted turn)

    public static func settled(
        hasThought: Bool,
        thinkingTimeMs: Int?,
        searchCount: Int,
        fetchedPageCount: Int,
        hasOtherToolActivity: Bool
    ) -> TurnActivityModel {
        var chips: [Chip] = []
        if hasThought {
            chips.append(
                Chip(
                    kind: .thought,
                    systemName: "brain",
                    label: "Thought",
                    caption: thinkingTimeMs.map(Self.formatDuration)
                )
            )
        }
        if searchCount > 0 || fetchedPageCount > 0 || hasOtherToolActivity {
            let (label, caption) = Self.settledSearchLabel(
                searchCount: searchCount,
                fetchedPageCount: fetchedPageCount,
                hasOtherToolActivity: hasOtherToolActivity
            )
            chips.append(
                Chip(kind: .search, systemName: "globe", label: label, caption: caption)
            )
        }
        return TurnActivityModel(chips: chips)
    }

    private static func settledSearchLabel(
        searchCount: Int,
        fetchedPageCount: Int,
        hasOtherToolActivity: Bool
    ) -> (label: String, caption: String?) {
        if searchCount > 0 {
            return ("Searched", searchCount > 1 ? "×\(searchCount)" : nil)
        }
        if fetchedPageCount > 0 {
            return (
                fetchedPageCount == 1 ? "Read a page" : "Read pages",
                fetchedPageCount > 1 ? "×\(fetchedPageCount)" : nil
            )
        }
        return ("Used tools", nil)
    }

    // MARK: - Auto-follow
    //
    // While streaming, the open well tracks what the model is doing:
    // thinking expands the thought well (collapsing search), a tool
    // round expands the search well (collapsing thought), and the
    // first answer token closes both so the reply gets the stage.

    public static func autoDetail(for phase: StreamingPhase) -> Detail {
        switch phase {
        case .thinking, .generating:
            return .thought
        case .searching, .reading:
            return .search
        case .answering, .finalizing, .idle:
            return .none
        }
    }

    public static func formatDuration(_ ms: Int) -> String {
        let seconds = Double(ms) / 1000.0
        if seconds < 1.0 { return "\(ms) ms" }
        return String(format: "%.1fs", seconds)
    }
}
