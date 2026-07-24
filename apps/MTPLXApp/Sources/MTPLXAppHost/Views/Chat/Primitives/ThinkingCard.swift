import SwiftUI
import MarkdownUI
import MTPLXAppCore

// MARK: - Layout constants
//
// 72pt viewport split into 3 lines of 24pt height each — matching
// Aphanes V2's `NativeChromeReference.Chat.thoughtViewportHeight` /
// `thoughtLineHeight`. The line-by-line opacity ramp (0.95 / 0.5 / 0.3)
// and the asymmetric insets (0/6/12 px leading, 0/10/18 px trailing)
// were tuned in Aphanes; ported verbatim.

enum ThinkingCardMetrics {
    static let collapsedViewportHeight: CGFloat = 72
    static let collapsedLineHeight: CGFloat = 24
    static let expandedMaxHeight: CGFloat = 360
    static let cornerRadius: CGFloat = 16
    static let collapsedTailCharacterLimit: Int = 512
    static let collapsedWrapColumn: Int = 64
}

// MARK: - ThinkingCard
//
// Port of Aphanes V2's `ThinkingCard` (AppViews.swift ~6249-6595).
// Two modes:
//   - Full card (default, used while streaming): header with "Thinking"
//     pulse + chevron, collapsed body shows the last 3 streamed lines
//     with the fade-mask viewport, expanded body shows full markdown.
//   - Compact chip (used inline above completed assistant messages):
//     a single "Thought · 4.3s" capsule that expands in place.
//
// Re-themed against MTPLX `Brand` tokens. The complex popover
// disclosure from Aphanes is replaced with a simple inline expand
// because MTPLX chat is one-conversation-at-a-time and popovers feel
// out of place against the existing dark dashboard chrome.

struct ThinkingCard: View {
    let content: String
    var isStreaming: Bool = false
    var thinkingTimeMs: Int? = nil
    var isCompact: Bool = false
    var expansionState: Binding<Bool>? = nil
    var collapsedContent: String? = nil

    @State private var localIsExpanded: Bool = false

    private var isExpandedBinding: Binding<Bool> {
        expansionState ?? $localIsExpanded
    }

    private var disclosureAnimation: Animation {
        .spring(response: 0.34, dampingFraction: 0.88, blendDuration: 0.12)
    }

    private func toggleExpanded() {
        withAnimation(disclosureAnimation) {
            isExpandedBinding.wrappedValue.toggle()
        }
    }

    var body: some View {
        Group {
            if isCompact {
                compactChip
            } else {
                fullCard
            }
        }
        .animation(isStreaming ? nil : disclosureAnimation, value: isCompact)
    }

    // MARK: - Compact chip

    private var compactChip: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                toggleExpanded()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "brain")
                        .font(.system(size: 11, weight: .medium))
                    Text("Thought")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                    if let thinkingTimeMs {
                        Text(Self.formatDuration(thinkingTimeMs))
                            .font(.system(size: 11, design: .rounded))
                            .foregroundStyle(Brand.typeTertiary)
                    }
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                        .rotationEffect(.degrees(isExpandedBinding.wrappedValue ? 90 : 0))
                }
                .foregroundStyle(Brand.typeSecondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .contentShape(Capsule())
                .background(
                    Capsule(style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )
            }
            .buttonStyle(.plain)

            if isExpandedBinding.wrappedValue {
                expandedThoughtBody
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: ThinkingCardMetrics.cornerRadius, style: .continuous)
                            .fill(Brand.cardSurface)
                            .overlay(
                                RoundedRectangle(cornerRadius: ThinkingCardMetrics.cornerRadius, style: .continuous)
                                    .stroke(Brand.separator, lineWidth: 1)
                            )
                    )
                    .transition(.opacity.combined(with: .offset(y: -4)))
            }
        }
    }

    // MARK: - Full card

    private var fullCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                toggleExpanded()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "brain")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Brand.typeSecondary.opacity(isStreaming ? 0.9 : 0.55))

                    Text(isStreaming ? "Thinking" : "Thought process")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(Brand.typeHi.opacity(isStreaming ? 0.82 : 0.6))

                    if !isStreaming, let thinkingTimeMs {
                        Text(Self.formatDuration(thinkingTimeMs))
                            .font(.system(size: 11, design: .rounded))
                            .foregroundStyle(Brand.typeTertiary)
                    }

                    Spacer(minLength: 8)

                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                        .rotationEffect(.degrees(isExpandedBinding.wrappedValue ? 90 : 0))
                }
                .padding(.horizontal, 14)
                .padding(.top, 12)
                .padding(.bottom, isExpandedBinding.wrappedValue ? 10 : 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Group {
                if isExpandedBinding.wrappedValue {
                    expandedThoughtBody
                        .padding(.horizontal, 14)
                        .padding(.bottom, 14)
                } else {
                    collapsedThoughtViewport
                        .padding(.horizontal, 14)
                        .padding(.bottom, 14)
                }
            }
            .clipped()
        }
        .clipped()
        .background(
            RoundedRectangle(cornerRadius: ThinkingCardMetrics.cornerRadius, style: .continuous)
                .fill(Color.white.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: ThinkingCardMetrics.cornerRadius, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 1)
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: ThinkingCardMetrics.cornerRadius, style: .continuous))
    }

    // MARK: - Bodies

    private var collapsedThoughtViewport: some View {
        let collapsedLineLimit = max(
            1,
            Int(ThinkingCardMetrics.collapsedViewportHeight / ThinkingCardMetrics.collapsedLineHeight)
        )
        let lines = Self.visibleCollapsedLines(from: collapsedContent ?? content)
        let paddedLines =
            Array(repeating: "", count: max(0, collapsedLineLimit - lines.count))
            + Array(lines.suffix(collapsedLineLimit))

        return VStack(alignment: .leading, spacing: 0) {
            if lines.isEmpty {
                Text(isStreaming ? "Processing…" : "No thought content captured.")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                    .frame(
                        height: ThinkingCardMetrics.collapsedViewportHeight,
                        alignment: .topLeading
                    )
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(0..<collapsedLineLimit, id: \.self) { slot in
                        let line = paddedLines[slot]
                        let visualIndex = collapsedLineLimit - 1 - slot
                        Text(line.isEmpty ? " " : line)
                            .font(.system(
                                size: Self.thoughtFontSize(for: visualIndex),
                                weight: visualIndex == 0 ? .medium : .regular,
                                design: .monospaced
                            ))
                            .foregroundStyle(
                                Brand.typeHi.opacity(
                                    line.isEmpty ? 0 : Self.thoughtOpacity(for: visualIndex)
                                )
                            )
                            .lineLimit(1)
                            .padding(.leading, Self.thoughtLeadingInset(for: visualIndex))
                            .padding(.trailing, Self.thoughtTrailingInset(for: visualIndex))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .frame(height: ThinkingCardMetrics.collapsedLineHeight)
                    }
                }
                .frame(height: ThinkingCardMetrics.collapsedViewportHeight, alignment: .bottom)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .mask(
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .black.opacity(0.9), location: 0.18),
                    .init(color: .black, location: 0.5),
                    .init(color: .black.opacity(0.92), location: 0.82),
                    .init(color: .clear, location: 1),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    @ViewBuilder
    private var expandedThoughtBody: some View {
        ScrollView {
            if isStreaming {
                Text(content)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineSpacing(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            } else {
                Markdown(content)
                    .markdownTheme(.mtplxChat)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
        }
        .frame(maxHeight: ThinkingCardMetrics.expandedMaxHeight)
    }

    // MARK: - Line shaping (ported verbatim)

    private static func visibleCollapsedLines(from text: String) -> [String] {
        let collapsedLineLimit = Int(
            ThinkingCardMetrics.collapsedViewportHeight / ThinkingCardMetrics.collapsedLineHeight
        )
        let tailSize = ThinkingCardMetrics.collapsedTailCharacterLimit
        let tail: Substring
        if text.count > tailSize,
            let idx = text.index(text.endIndex, offsetBy: -tailSize, limitedBy: text.startIndex)
        {
            tail = text[idx...]
        } else {
            tail = text[...]
        }
        let words = tail.split(whereSeparator: \.isNewline).flatMap { segment -> [String] in
            let trimmedSegment = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedSegment.isEmpty else { return [] }
            let stripped =
                trimmedSegment
                .replacingOccurrences(of: "**", with: "")
                .replacingOccurrences(of: "__", with: "")
                .replacingOccurrences(of: "*", with: "")
                .replacingOccurrences(of: "_", with: " ")
                .replacingOccurrences(of: "###", with: "")
                .replacingOccurrences(of: "##", with: "")
                .replacingOccurrences(of: "#", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !stripped.isEmpty else { return [] }
            return wrapThoughtLine(stripped, maxCharacters: ThinkingCardMetrics.collapsedWrapColumn)
        }
        return Array(words.suffix(collapsedLineLimit))
    }

    private static func wrapThoughtLine(_ text: String, maxCharacters: Int) -> [String] {
        guard text.count > maxCharacters else { return [text] }
        var lines: [String] = []
        var currentLine = ""
        for word in text.split(whereSeparator: \.isWhitespace) {
            let candidate = currentLine.isEmpty ? String(word) : "\(currentLine) \(word)"
            if candidate.count > maxCharacters, !currentLine.isEmpty {
                lines.append(currentLine)
                currentLine = String(word)
            } else {
                currentLine = candidate
            }
        }
        if !currentLine.isEmpty {
            lines.append(currentLine)
        }
        return lines
    }

    private static func thoughtOpacity(for visualIndex: Int) -> Double {
        switch visualIndex {
        case 0: return 0.95
        case 1: return 0.5
        default: return 0.3
        }
    }

    private static func thoughtLeadingInset(for visualIndex: Int) -> CGFloat {
        switch visualIndex {
        case 0: return 0
        case 1: return 6
        default: return 12
        }
    }

    private static func thoughtTrailingInset(for visualIndex: Int) -> CGFloat {
        switch visualIndex {
        case 0: return 0
        case 1: return 10
        default: return 18
        }
    }

    private static func thoughtFontSize(for visualIndex: Int) -> CGFloat {
        switch visualIndex {
        case 0: return 14
        case 1: return 13
        default: return 12
        }
    }

    private static func formatDuration(_ ms: Int) -> String {
        let seconds = Double(ms) / 1000.0
        if seconds < 1.0 { return "\(ms) ms" }
        return String(format: "%.1fs", seconds)
    }
}

// MARK: - StreamingThinkingCard

struct StreamingThinkingCard: View {
    @ObservedObject var document: StreamingDocumentStore
    var contentOverride: String?
    var isStreaming: Bool = true
    var thinkingTimeMs: Int? = nil
    var isCompact: Bool = false
    var expansionState: Binding<Bool>? = nil

    private var liveContent: String {
        document.rawText.isEmpty ? (contentOverride ?? "") : document.rawText
    }

    private var collapsedLiveContent: String {
        let recent = document.recentText(characterLimit: ThinkingCardMetrics.collapsedTailCharacterLimit)
        if !recent.isEmpty {
            return recent
        }
        return String((contentOverride ?? "").suffix(ThinkingCardMetrics.collapsedTailCharacterLimit))
    }

    var body: some View {
        ThinkingCard(
            content: liveContent,
            isStreaming: isStreaming,
            thinkingTimeMs: thinkingTimeMs,
            isCompact: isCompact,
            expansionState: expansionState,
            collapsedContent: collapsedLiveContent
        )
    }
}
