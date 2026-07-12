import SwiftUI
import MarkdownUI
import MTPLXAppCore

// MARK: - TurnActivityStrip
//
// THE surface for an assistant turn's activity (2026-07-03 chat-UX
// redesign, founder-directed). One strip serves both the streaming
// turn and the settled transcript bubble:
//
//   [ 🧠 Thinking ⋯ ]  [ 🌐 Searched ×3 ]              ← content-hugging chips
//   ┌───────────────────────────────────────────┐
//   │ …one detail well, only ever one open…     │     ← morphs per phase
//   └───────────────────────────────────────────┘
//
// Rules:
//   - Chips sit BESIDE each other (never stacked) and hug their
//     content — the classic chunky capsules (founder reverted the
//     brief equal-width experiment on sight, 2026-07-03 ~02:50).
//   - Exactly one well below the row. While streaming it auto-follows
//     the active tool (thinking → thought lines, searching → query
//     rows); the first answer token closes it. After the turn settles
//     the chips become manual toggles.
//   - The strip renders in the SAME geometry live and settled, so the
//     handoff from streaming surface to persisted bubble doesn't jump.
//
// This replaces the ThinkingCard + AssistantTraceSurface pair, whose
// separate stacked cards (plus the mid-turn persisted rounds rendering
// as a second, settled copy of the same turn) produced the cluttered
// transcript in the founder's 03:00 screenshots.

struct TurnActivityStrip<ThoughtWell: View, SearchWell: View>: View {
    let model: TurnActivityModel
    @Binding var expandedDetail: TurnActivityModel.Detail
    @ViewBuilder var thoughtWell: () -> ThoughtWell
    @ViewBuilder var searchWell: () -> SearchWell

    private var disclosureAnimation: Animation {
        .spring(response: 0.34, dampingFraction: 0.88, blendDuration: 0.12)
    }

    var body: some View {
        if !model.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    ForEach(model.chips) { chip in
                        chipView(chip)
                            .transition(.opacity.combined(with: .scale(scale: 0.96)))
                    }
                }

                switch activeDetail {
                case .thought:
                    wellContainer { thoughtWell() }
                case .search:
                    wellContainer { searchWell() }
                case .none:
                    EmptyView()
                }
            }
            // Value-scoped so these survive the transcript's
            // streaming-wide `transaction { animation = nil }` guard:
            // chip arrivals and well swaps animate, token repaints
            // never do.
            .animation(disclosureAnimation, value: model)
            .animation(disclosureAnimation, value: expandedDetail)
        }
    }

    /// A well only opens for a chip that exists — a stale `.search`
    /// selection on a turn that never searched renders as closed.
    private var activeDetail: TurnActivityModel.Detail {
        switch expandedDetail {
        case .thought: return model.hasChip(.thought) ? .thought : .none
        case .search: return model.hasChip(.search) ? .search : .none
        case .none: return .none
        }
    }

    private func toggle(_ chip: TurnActivityModel.Chip) {
        withAnimation(disclosureAnimation) {
            expandedDetail = expandedDetail == chip.kind.detail ? .none : chip.kind.detail
        }
    }

    private func chipView(_ chip: TurnActivityModel.Chip) -> some View {
        let isOpen = activeDetail == chip.kind.detail
        return Button {
            toggle(chip)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: chip.systemName)
                    .font(.system(size: 11, weight: .medium))
                Text(chip.label)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .contentTransition(.opacity)
                if let caption = chip.caption {
                    Text(caption)
                        .font(.system(size: 11, design: .rounded))
                        .foregroundStyle(Brand.typeTertiary)
                }
                if chip.isLive {
                    ThinkingIndicatorDots()
                }
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(Brand.typeTertiary)
                    .rotationEffect(.degrees(isOpen ? 90 : 0))
            }
            .foregroundStyle(chip.isLive || isOpen ? Brand.typeHi.opacity(0.85) : Brand.typeSecondary)
            .lineLimit(1)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(chip.isLive || isOpen ? 0.10 : 0.06))
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            isOpen ? "Collapse \(chip.label)" : "Expand \(chip.label)"
        )
    }

    private func wellContainer<Content: View>(
        @ViewBuilder _ content: () -> Content
    ) -> some View {
        content()
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(0.04))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(Brand.separator, lineWidth: 1)
                    )
            )
            .transition(.opacity.combined(with: .offset(y: -4)))
    }
}

// MARK: - Search well

/// One line of tool activity inside the search well
/// ("claude opus 4.8 release · Found 5 results").
struct ThinkingActivityRow: Identifiable, Equatable {
    let id: String
    var systemName: String
    var text: String
    var detail: String
    var isLive: Bool
}

struct SearchActivityWell: View {
    let rows: [ThinkingActivityRow]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(rows) { row in
                HStack(spacing: 7) {
                    Image(systemName: row.systemName)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Brand.typeTertiary)
                        .frame(width: 12)
                    Text(row.text)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Brand.typeSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    if row.isLive {
                        ThinkingIndicatorDots()
                    } else if !row.detail.isEmpty {
                        Text(row.detail)
                            .font(.system(size: 11))
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .transition(.opacity.combined(with: .offset(y: -2)))
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.9), value: rows)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Thought wells
//
// The 72pt / 3-line fading tail viewport and its line shaping are the
// Aphanes V2 port previously hosted in ThinkingCard — constants and
// ramp values carried over verbatim (they were hand-tuned there).

enum ThoughtViewportMetrics {
    static let viewportHeight: CGFloat = 72
    static let lineHeight: CGFloat = 24
    static let tailCharacterLimit: Int = 512
    static let wrapColumn: Int = 64
    static let settledMaxHeight: CGFloat = 360
}

/// Live thought well: the last three streamed lines in a fade-masked
/// viewport. Observes the reasoning document so token appends repaint
/// ONLY this view, not the strip around it.
///
/// `fallback` is the buffer-INCLUSIVE text (document + unflushed
/// coalescing buffer) captured at the parent's last render. The tail
/// reads whichever of the two is longer, so even if the 16 ms flush
/// loop ever stalls mid-turn, the viewport keeps advancing on parent
/// repaints instead of freezing on the last flushed state (the
/// 2026-07-03 "frozen after search→thinking" report).
struct StreamingThoughtWell: View {
    @ObservedObject var document: StreamingDocumentStore
    var fallback: String = ""

    private var tail: String {
        let flushed = document.rawText
        let live = fallback.count > flushed.count ? fallback : flushed
        return String(live.suffix(ThoughtViewportMetrics.tailCharacterLimit))
    }

    var body: some View {
        ThoughtStreamViewport(text: tail)
    }
}

/// Settled thought well: the whole turn's reasoning as markdown,
/// scrollable past `settledMaxHeight`.
struct SettledThoughtWell: View {
    let content: String

    var body: some View {
        ScrollView {
            Markdown(content)
                .markdownTheme(.mtplxChat)
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .frame(maxHeight: ThoughtViewportMetrics.settledMaxHeight)
    }
}

struct ThoughtStreamViewport: View {
    let text: String

    var body: some View {
        let lineLimit = max(
            1,
            Int(ThoughtViewportMetrics.viewportHeight / ThoughtViewportMetrics.lineHeight)
        )
        let lines = Self.visibleLines(from: text)
        let paddedLines =
            Array(repeating: "", count: max(0, lineLimit - lines.count))
            + Array(lines.suffix(lineLimit))

        return VStack(alignment: .leading, spacing: 0) {
            if lines.isEmpty {
                Text("Processing…")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                    .frame(
                        height: ThoughtViewportMetrics.viewportHeight,
                        alignment: .topLeading
                    )
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(0..<lineLimit, id: \.self) { slot in
                        let line = paddedLines[slot]
                        let visualIndex = lineLimit - 1 - slot
                        Text(line.isEmpty ? " " : line)
                            .font(.system(
                                size: Self.fontSize(for: visualIndex),
                                weight: visualIndex == 0 ? .medium : .regular,
                                design: .monospaced
                            ))
                            .foregroundStyle(
                                Brand.typeHi.opacity(
                                    line.isEmpty ? 0 : Self.opacity(for: visualIndex)
                                )
                            )
                            .lineLimit(1)
                            .padding(.leading, Self.leadingInset(for: visualIndex))
                            .padding(.trailing, Self.trailingInset(for: visualIndex))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .frame(height: ThoughtViewportMetrics.lineHeight)
                    }
                }
                .frame(height: ThoughtViewportMetrics.viewportHeight, alignment: .bottom)
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

    // MARK: Line shaping (ported verbatim)

    private static func visibleLines(from text: String) -> [String] {
        let lineLimit = Int(
            ThoughtViewportMetrics.viewportHeight / ThoughtViewportMetrics.lineHeight
        )
        let tailSize = ThoughtViewportMetrics.tailCharacterLimit
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
            return wrapLine(stripped, maxCharacters: ThoughtViewportMetrics.wrapColumn)
        }
        return Array(words.suffix(lineLimit))
    }

    private static func wrapLine(_ text: String, maxCharacters: Int) -> [String] {
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

    private static func opacity(for visualIndex: Int) -> Double {
        switch visualIndex {
        case 0: return 0.95
        case 1: return 0.5
        default: return 0.3
        }
    }

    private static func leadingInset(for visualIndex: Int) -> CGFloat {
        switch visualIndex {
        case 0: return 0
        case 1: return 6
        default: return 12
        }
    }

    private static func trailingInset(for visualIndex: Int) -> CGFloat {
        switch visualIndex {
        case 0: return 0
        case 1: return 10
        default: return 18
        }
    }

    private static func fontSize(for visualIndex: Int) -> CGFloat {
        switch visualIndex {
        case 0: return 14
        case 1: return 13
        default: return 12
        }
    }
}
