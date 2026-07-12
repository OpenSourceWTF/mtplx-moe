import AppKit
import SwiftUI
import MarkdownUI
import MTPLXAppCore

// MARK: - AssistantMarkdownView
//
// Markdown renderer for assistant bubbles. Settled bubbles render fully;
// the live streaming path promotes frozen fence-safe blocks to the same
// settled pipeline (rendered once, cached by Equatable text) while the
// growing tail stays plain text — markdown during streaming with no
// per-token parsing cost.

struct AssistantMarkdownView: View {
    let content: String
    let isStreaming: Bool

    init(_ content: String, isStreaming: Bool = false) {
        self.content = content
        self.isStreaming = isStreaming
    }

    var body: some View {
        if isStreaming {
            StreamingPlainTextView(text: content)
        } else {
            SettledAssistantMarkdownView(content: content)
        }
    }
}

private struct SettledAssistantMarkdownView: View {
    let content: String

    private var blocks: [SettledMarkdownBlock] {
        CachedSettledMarkdownBlocks.blocks(for: content)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(blocks) { block in
                switch block.kind {
                case .prose(let text):
                    if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        AssistantProseMarkdownView(text: text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                case .code(let language, let code, let lineCount):
                    AssistantCodeBlockView(language: language, code: code, lineCount: lineCount)
                        .padding(.vertical, 4)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct AssistantProseMarkdownView: View {
    let text: String

    private var lines: [AssistantProseLine] {
        AssistantProseLine.parse(text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(lines) { line in
                lineView(line)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func lineView(_ line: AssistantProseLine) -> some View {
        switch line.kind {
        case .blank:
            Color.clear
                .frame(height: 2)
                .accessibilityHidden(true)
        case .heading(let level, let text):
            Text(Self.inlineAttributed(text))
                .font(Self.headingFont(level: level))
                .foregroundStyle(Brand.typeHi)
                .padding(.top, level <= 2 ? 8 : 5)
                .padding(.bottom, 1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        case .bullet(let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("•")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(Brand.typeSecondary)
                    .frame(width: 12, alignment: .trailing)
                Text(Self.inlineAttributed(text))
                    .font(.system(size: 14))
                    .foregroundStyle(Brand.typeHi)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .ordered(let marker, let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(marker)
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .frame(width: 28, alignment: .trailing)
                Text(Self.inlineAttributed(text))
                    .font(.system(size: 14))
                    .foregroundStyle(Brand.typeHi)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .quote(let text):
            Text(Self.inlineAttributed(text))
                .font(.system(size: 14))
                .foregroundStyle(Brand.typeSecondary)
                .padding(.leading, 12)
                .overlay(alignment: .leading) {
                    Rectangle()
                        .fill(Brand.separatorStrong)
                        .frame(width: 2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        case .paragraph(let text):
            Text(Self.inlineAttributed(text))
                .font(.system(size: 14))
                .foregroundStyle(Brand.typeHi)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        case .math(let latex):
            // Display equations render as readable math text (the AIME
            // formatter), centered like ChatMathMarkdown's display rows.
            // The Jun-6 perf rewrite of this view dropped the math path
            // and chat regressed to raw $$...$$ source (QA-108).
            Text(StreamingMathTextFormatter.readableText(from: latex))
                .font(.system(size: 15, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeHi)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 4)
        }
    }

    private static func headingFont(level: Int) -> Font {
        switch level {
        case 1: .system(size: 21, weight: .heavy)
        case 2: .system(size: 17, weight: .heavy)
        case 3: .system(size: 15, weight: .bold)
        default: .system(size: 14, weight: .bold)
        }
    }

    private static func inlineAttributed(_ text: String) -> AttributedString {
        let readable = Self.withReadableInlineMath(text)
        return (try? AttributedString(
            markdown: readable,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .inlineOnlyPreservingWhitespace
            )
        )) ?? AttributedString(readable)
    }

    /// Replaces inline LaTeX spans (`$...$`, `\(...\)`, and any `$$...$$`
    /// embedded mid-line) with the readable math text the AIME surface
    /// uses, before inline-markdown attribution. Pure text-in/text-out so
    /// the settled-prose perf path stays line-based (QA-108).
    private static func withReadableInlineMath(_ text: String) -> String {
        guard text.contains("$") || text.contains("\\(") else { return text }
        let runs = StreamingDocumentStore.mathRuns(in: text)
        guard runs.contains(where: { $0.kind != .text }) else { return text }
        return runs.map { run in
            run.kind == .text
                ? run.text
                : StreamingMathTextFormatter.readableText(from: run.text)
        }.joined()
    }
}

private struct AssistantProseLine: Identifiable, Equatable {
    enum Kind: Equatable {
        case blank
        case heading(level: Int, text: String)
        case bullet(String)
        case ordered(marker: String, text: String)
        case quote(String)
        case paragraph(String)
        case math(String)
    }

    let id: Int
    let kind: Kind

    static func parse(_ source: String) -> [AssistantProseLine] {
        let rawLines = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var lines: [AssistantProseLine] = []
        var index = 0
        var cursor = 0
        while cursor < rawLines.count {
            let trimmed = rawLines[cursor].trimmingCharacters(in: .whitespaces)
            // Block-form display math: a bare $$ (or \[) line opens a
            // block that runs to the matching closer; the interior is
            // one centered math row (QA-108).
            if trimmed == "$$" || trimmed == "\\[" {
                let closer = trimmed == "$$" ? "$$" : "\\]"
                var body: [String] = []
                var lookahead = cursor + 1
                while lookahead < rawLines.count,
                      rawLines[lookahead].trimmingCharacters(in: .whitespaces) != closer
                {
                    body.append(rawLines[lookahead])
                    lookahead += 1
                }
                if lookahead < rawLines.count, !body.isEmpty {
                    lines.append(AssistantProseLine(
                        id: index,
                        kind: .math(body.joined(separator: " ").trimmingCharacters(in: .whitespaces))
                    ))
                    index += 1
                    cursor = lookahead + 1
                    continue
                }
            }
            lines.append(AssistantProseLine(id: index, kind: Self.kind(for: rawLines[cursor])))
            index += 1
            cursor += 1
        }
        return lines
    }

    private static func kind(for rawLine: String) -> Kind {
        let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return .blank }
        if let math = displayMath(in: trimmed) {
            return .math(math)
        }
        if let heading = heading(in: trimmed) {
            return heading
        }
        if let bullet = bullet(in: trimmed) {
            return .bullet(bullet)
        }
        if let ordered = ordered(in: trimmed) {
            return ordered
        }
        if trimmed.hasPrefix(">") {
            let quote = trimmed
                .dropFirst()
                .trimmingCharacters(in: .whitespaces)
            return .quote(quote)
        }
        return .paragraph(trimmed)
    }

    /// Single-line display math: `$$...$$` or `\[...\]` filling the
    /// whole line (the shape chat models emit most).
    private static func displayMath(in line: String) -> String? {
        for (open, close) in [("$$", "$$"), ("\\[", "\\]")] {
            if line.hasPrefix(open), line.hasSuffix(close),
               line.count > open.count + close.count
            {
                let body = line
                    .dropFirst(open.count)
                    .dropLast(close.count)
                    .trimmingCharacters(in: .whitespaces)
                if !body.isEmpty { return body }
            }
        }
        return nil
    }

    private static func heading(in line: String) -> Kind? {
        var cursor = line.startIndex
        var level = 0
        while cursor < line.endIndex, line[cursor] == "#", level < 6 {
            level += 1
            cursor = line.index(after: cursor)
        }
        guard level > 0, cursor < line.endIndex, line[cursor].isWhitespace else {
            return nil
        }
        let text = line[cursor...]
            .trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "#").union(.whitespaces))
        return text.isEmpty ? nil : .heading(level: level, text: text)
    }

    private static func bullet(in line: String) -> String? {
        guard let marker = line.first,
              marker == "*" || marker == "-" || marker == "+",
              line.count >= 2
        else { return nil }
        let bodyStart = line.index(after: line.startIndex)
        guard line[bodyStart].isWhitespace else { return nil }
        let text = line[bodyStart...].trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : text
    }

    private static func ordered(in line: String) -> Kind? {
        var cursor = line.startIndex
        while cursor < line.endIndex, line[cursor].isNumber {
            cursor = line.index(after: cursor)
        }
        guard cursor > line.startIndex,
              cursor < line.endIndex,
              line[cursor] == "." || line[cursor] == ")"
        else { return nil }
        let markerEnd = line.index(after: cursor)
        guard markerEnd < line.endIndex, line[markerEnd].isWhitespace else { return nil }
        let marker = String(line[line.startIndex...cursor])
        let text = line[markerEnd...].trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : .ordered(marker: marker, text: text)
    }
}

private struct SettledMarkdownBlock: Identifiable, Equatable {
    enum Kind: Equatable {
        case prose(String)
        case code(language: String?, code: String, lineCount: Int)
    }

    let id: Int
    let kind: Kind
}

private final class SettledMarkdownBlocksBox: NSObject {
    let blocks: [SettledMarkdownBlock]

    init(blocks: [SettledMarkdownBlock]) {
        self.blocks = blocks
    }
}

private enum CachedSettledMarkdownBlocks {
    nonisolated(unsafe) private static let cache: NSCache<NSString, SettledMarkdownBlocksBox> = {
        let cache = NSCache<NSString, SettledMarkdownBlocksBox>()
        cache.countLimit = 256
        cache.totalCostLimit = 16_000_000
        return cache
    }()

    static func blocks(for source: String) -> [SettledMarkdownBlock] {
        let key = source as NSString
        if let cached = cache.object(forKey: key) {
            return cached.blocks
        }

        let parsed = parse(source)
        cache.setObject(
            SettledMarkdownBlocksBox(blocks: parsed),
            forKey: key,
            cost: source.utf8.count
        )
        return parsed
    }

    static func removeAllObjects() {
        cache.removeAllObjects()
    }

    private static func parse(_ source: String) -> [SettledMarkdownBlock] {
        guard !source.isEmpty else { return [] }
        var blocks: [SettledMarkdownBlock] = []
        var cursor = source.startIndex

        func appendProse(upTo fenceStart: String.Index) {
            guard cursor < fenceStart else { return }
            let text = String(source[cursor..<fenceStart])
            blocks.append(SettledMarkdownBlock(id: blocks.count, kind: .prose(text)))
        }

        while cursor < source.endIndex {
            guard let fence = source.range(of: "```", range: cursor..<source.endIndex) else {
                appendProse(upTo: source.endIndex)
                break
            }

            appendProse(upTo: fence.lowerBound)

            let languageAndBodyStart = fence.upperBound
            let bodyStart: String.Index
            let language: String?
            if let newline = source[languageAndBodyStart...].firstIndex(of: "\n") {
                let rawLanguage = source[languageAndBodyStart..<newline]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                language = rawLanguage.isEmpty ? nil : rawLanguage
                bodyStart = source.index(after: newline)
            } else {
                let rawLanguage = source[languageAndBodyStart...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                blocks.append(
                    SettledMarkdownBlock(
                        id: blocks.count,
                        kind: .code(
                            language: rawLanguage.isEmpty ? nil : rawLanguage,
                            code: "",
                            lineCount: 1
                        )
                    )
                )
                cursor = source.endIndex
                break
            }

            if let closingFence = source.range(of: "```", range: bodyStart..<source.endIndex) {
                let code = String(source[bodyStart..<closingFence.lowerBound])
                blocks.append(
                    SettledMarkdownBlock(
                        id: blocks.count,
                        kind: .code(
                            language: language,
                            code: code,
                            lineCount: AssistantCodeMetrics.lineCount(in: code)
                        )
                    )
                )
                cursor = closingFence.upperBound
            } else {
                let code = String(source[bodyStart...])
                blocks.append(
                    SettledMarkdownBlock(
                        id: blocks.count,
                        kind: .code(
                            language: language,
                            code: code,
                            lineCount: AssistantCodeMetrics.lineCount(in: code)
                        )
                    )
                )
                cursor = source.endIndex
            }
        }

        return blocks
    }
}

enum ChatRenderCaches {}

extension ChatRenderCaches {
    static func clearMemoryPressureSensitiveCaches() {
        clearSettledMarkdownBlocks()
        clearMarkdownDocuments()
    }

    static func clearSettledMarkdownBlocks() {
        CachedSettledMarkdownBlocks.removeAllObjects()
    }
}

private enum AssistantCodeMetrics {
    static func lineCount(in code: String) -> Int {
        max(1, code.reduce(1) { count, character in
            character == "\n" ? count + 1 : count
        })
    }
}

// MARK: - StreamingAssistantMarkdownView

struct StreamingAssistantMarkdownView: View {
    @ObservedObject var document: StreamingDocumentStore
    var fallbackText: String = ""

    var body: some View {
        Group {
            if document.blocks.isEmpty {
                StreamingPlainTextView(text: fallbackText)
            } else {
                // Frozen fence-safe blocks render as full markdown ONCE
                // (Equatable on text, so they never repaint as later
                // tokens arrive); only the growing tail block — and any
                // open-fence interior — stays a plain Text. Per-token
                // cost is one linear safety pass plus the tail repaint,
                // so streaming TPS is untouched (2026-07-03).
                let blocks = document.blocks
                let safety = StreamingMarkdownBlockSafety.classify(blocks.map(\.text))
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(blocks.enumerated()), id: \.element.id) { index, block in
                        if index < safety.count, safety[index] {
                            StreamingSettledBlockView(text: block.text)
                                .equatable()
                        } else {
                            StreamingPlainBlockView(block: block)
                                .equatable()
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .transaction { tx in
            tx.animation = nil
        }
    }
}

/// A frozen streaming block promoted to the settled markdown pipeline.
/// Equatable on its text: SwiftUI evaluates the body once when the
/// block freezes and never again during the rest of the stream.
private struct StreamingSettledBlockView: View, Equatable {
    let text: String

    nonisolated static func == (lhs: StreamingSettledBlockView, rhs: StreamingSettledBlockView) -> Bool {
        lhs.text == rhs.text
    }

    var body: some View {
        SettledAssistantMarkdownView(content: text)
            .padding(.bottom, 6)
    }
}

private struct StreamingPlainTextView: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 14))
            .foregroundStyle(Brand.typeHi)
            .textSelection(.disabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct StreamingPlainBlockView: View, Equatable {
    let block: StreamingDocumentBlock

    nonisolated static func == (lhs: StreamingPlainBlockView, rhs: StreamingPlainBlockView) -> Bool {
        lhs.block == rhs.block
    }

    var body: some View {
        Text(block.text.isEmpty ? " " : block.text)
            .font(.system(size: 14))
            .foregroundStyle(Brand.typeHi)
            .textSelection(.disabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - Brand-themed MarkdownUI Theme

extension Theme {
    /// Brand-themed MarkdownUI theme for the in-app chat. Mirrors the
    /// shape of Aphanes' theme but rewires every color/font to MTPLX
    /// tokens. MainActor-scoped because the `markdownMargin` /
    /// `markdownTextStyle` SwiftUI modifiers used inside the block
    /// closures are main-actor isolated under Swift 6.
    @MainActor
    static var mtplxChat: Theme {
        Theme()
            .text {
                ForegroundColor(Brand.typeHi)
                FontSize(14)
                FontFamilyVariant(.normal)
            }
            .code {
                FontFamilyVariant(.monospaced)
                FontSize(13)
                ForegroundColor(Brand.typeHi)
                BackgroundColor(Color.white.opacity(0.06))
            }
            .link {
                ForegroundColor(Brand.accentChrome)
                UnderlineStyle(.single)
            }
            .strong { FontWeight(.semibold) }
            .emphasis { FontStyle(.italic) }
            .strikethrough { StrikethroughStyle(.single) }
            .heading1 { configuration in
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Brand.typeHi)
                        FontWeight(.heavy)
                        FontSize(22)
                    }
                    .markdownMargin(top: 18, bottom: 8)
            }
            .heading2 { configuration in
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Brand.typeHi)
                        FontWeight(.heavy)
                        FontSize(18)
                    }
                    .markdownMargin(top: 16, bottom: 6)
            }
            .heading3 { configuration in
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Brand.typeHi)
                        FontWeight(.bold)
                        FontSize(15)
                    }
                    .markdownMargin(top: 14, bottom: 4)
            }
            .paragraph { configuration in
                configuration.label
                    .markdownMargin(top: 0, bottom: 8)
            }
            .blockquote { configuration in
                configuration.label
                    .padding(.leading, 12)
                    .overlay(alignment: .leading) {
                        Rectangle()
                            .fill(Brand.separatorStrong)
                            .frame(width: 2)
                    }
                    .foregroundStyle(Brand.typeSecondary)
            }
            .codeBlock { configuration in
                AssistantCodeBlockView(
                    language: configuration.language,
                    code: configuration.content
                )
                    .markdownMargin(top: 8, bottom: 8)
            }
            .listItem { configuration in
                configuration.label
                    .markdownMargin(top: 4)
            }
            .table { configuration in
                configuration.label
                    .markdownMargin(top: 8, bottom: 8)
            }
            .tableCell { configuration in
                configuration.label
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .foregroundStyle(Brand.typeHi)
            }
    }
}

private struct AssistantCodeBlockView: View {
    let language: String?
    let code: String
    let lineCount: Int

    @State private var showCopied = false

    init(language: String?, code: String, lineCount: Int? = nil) {
        self.language = language
        self.code = code
        self.lineCount = lineCount ?? AssistantCodeMetrics.lineCount(in: code)
    }

    private var languageLabel: String? {
        let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed?.isEmpty == false ? trimmed : nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                if let languageLabel {
                    Text(languageLabel.uppercased())
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundStyle(Brand.typeTertiary)
                        .lineLimit(1)
                }

                Spacer(minLength: 12)

                copyButton
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 5)
            .background(Color.white.opacity(0.035))

            CodeTextViewport(code: code)
                .frame(height: codeViewportHeight)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(codeAccessibilitySummary)
            .accessibilityHint("Use the Copy button to copy the full code.")
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.bgInner)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var codeAccessibilitySummary: String {
        let label = languageLabel ?? "Plain text"
        return "\(label) code block, \(code.count) characters"
    }

    private var codeViewportHeight: CGFloat {
        let unclamped = CGFloat(lineCount) * 17 + 22
        return min(420, max(78, unclamped))
    }

    private var copyButton: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(code, forType: .string)
            showCopied = true

            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1.4))
                showCopied = false
            }
        } label: {
            Label(showCopied ? "Copied" : "Copy", systemImage: showCopied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 11, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .foregroundStyle(showCopied ? Brand.success : Brand.typeTertiary)
        }
        .buttonStyle(.plain)
        .help(showCopied ? "Copied" : "Copy code")
    }
}

private struct CodeTextViewport: NSViewRepresentable {
    let code: String

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true

        let textView = NSTextView()
        textView.drawsBackground = false
        textView.isEditable = false
        textView.isSelectable = false
        textView.isRichText = false
        textView.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.textColor = NSColor(calibratedWhite: 0.88, alpha: 1.0)
        textView.textContainerInset = NSSize(width: 12, height: 10)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.heightTracksTextView = false
        textView.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isHorizontallyResizable = true
        textView.isVerticallyResizable = true
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.setAccessibilityElement(false)
        textView.string = code

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        if textView.string != code {
            textView.string = code
        }
    }
}
