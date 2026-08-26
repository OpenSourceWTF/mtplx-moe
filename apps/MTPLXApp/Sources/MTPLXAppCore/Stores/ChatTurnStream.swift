import Foundation

// MARK: - ChatTurnStream
//
// The complete accumulation state of ONE in-flight assistant turn,
// owned by the conversation that asked for it (issue #324).
//
// Before this type existed, `ChatViewModel` kept a single app-wide set
// of streaming fields and `select(_:)` reset them on every conversation
// switch. Switching away mid-stream therefore (a) destroyed the visible
// partial answer, and (b) orphaned the still-running request: its
// events kept folding into state that no longer belonged to any
// conversation, a follow-up send elsewhere superseded the generation
// token, and the finished turn was never persisted — the server did all
// the work and the transcript kept the user prompt with no reply.
//
// Keying this state by conversation makes switching a pure view change:
// the stream keeps accumulating HERE regardless of what is visible, the
// view model's published surface mirrors whichever conversation is
// selected, and persistence reads this object — never the visible
// conversation's state. It also means a turn in conversation A and a
// turn in conversation B can be in flight at the same time without
// sharing a byte (each holds its own server session id via the
// conversation, so SessionBank warm-prefix reuse is unaffected).
//
// @MainActor like the view model that owns it; SSE events reach it only
// through `ChatViewModel.handleEvent`, which resolves the stream by
// `(conversationID, turnID)` — Sendable identity that can cross the
// stream callback boundary, unlike this class.

@MainActor
final class ChatTurnStream {
    /// The conversation this turn answers. Held strongly for the turn's
    /// lifetime (the tool-loop task already retained it before this
    /// refactor); `delete(_:)` cancels the turn before deleting the
    /// model, so the reference cannot outlive the store row.
    let conversation: ChatConversation
    let conversationID: UUID
    /// Identity shared by every assistant/tool message this turn's tool
    /// loop persists, AND the token SSE events are routed by: a
    /// replaced or cancelled turn's late events resolve against the
    /// registry with a stale `turnID` and are dropped — the per-turn
    /// successor of the old view-model-wide `streamGeneration` counter.
    let turnID = UUID()
    let startedAt = Date()

    // Live documents. Fresh per turn; the view model exposes the
    // CURRENT conversation's pair, so the live wells re-bind on switch.
    let reasoningDocument = StreamingDocumentStore(mode: .plainLines)
    let contentDocument = StreamingDocumentStore(mode: .plainLines)

    // Published-surface state (mirrored by ChatViewModel's computed
    // properties for whichever conversation is visible).
    var phase: StreamingPhase
    var hasReasoning = false
    var hasContent = false
    var pendingToolTraces: [PendingToolTrace] = []
    var liveTurnSources: [SourceRecord] = []
    var decodeReading: HeadlineDecodeReading = .absent
    var handoffAssistantMessageID: UUID?

    // Coalescing buffers ahead of the documents (small, CoW-shared).
    var reasoningBuffer = ""
    var contentBuffer = ""

    // Request plumbing.
    var requestId: String?
    var task: Task<Void, Never>?

    // Per-round accumulation for the tool loop.
    var roundToolCalls: [Int: AccumulatingToolCall] = [:]
    var roundFinishReason = "stop"
    var roundUsage: ChatUsage?
    var roundStats: ChatStreamStats?

    // Think-span accounting (multi-round "Thought · Ns" chip).
    var reasoningStartedAt: Date?
    var completedThinkingMs = 0
    /// Character offset into the accumulated live reasoning document
    /// where the CURRENT round's reasoning begins. The document only
    /// ever appends, so a count-based offset stays valid.
    var roundReasoningStartOffset = 0

    // Sources gathered across the turn's tool calls (raw + deduped).
    var turnSourceAccumulator: [SourceRecord] = []

    var leakedThinkingSplitter = ChatThinkingTagSplitter()

    // Live decode chip: sliding-window samples + throttle.
    var decodeWindowSamples: [(t: Double, tokens: Double)] = []
    var lastLiveDecodeUpdateAt: Date = .distantPast

    // Typewriter pacing (rate-based reveal) state.
    var contentArrivedCharsTotal = 0
    var lastArrivedCharsTotal = 0
    var revealRateCharsPerSecond: Double = 0
    var lastRevealTickUptime: Double = 0

    init(conversation: ChatConversation, phase: StreamingPhase) {
        self.conversation = conversation
        self.conversationID = conversation.id
        self.phase = phase
    }

    /// Full accumulated reasoning (document + unflushed buffer).
    /// O(reasoning) per access — turn-boundary use only.
    var reasoningText: String { reasoningDocument.rawText + reasoningBuffer }
    /// Full accumulated answer (document + unflushed buffer).
    /// O(answer) per access — turn-boundary use only.
    var contentText: String { contentDocument.rawText + contentBuffer }
}
