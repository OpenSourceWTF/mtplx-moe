import SwiftUI
import MTPLXAppCore

// MARK: - AssistantTraceSurface
//
// Port of Aphanes V2's `AssistantTraceSurface`. Used to render tool
// activity inline inside (or beneath) an assistant bubble: globe icon
// + "Web Search" title + subtitle showing the query + expandable
// detail showing results and a 4-line activity log.
//
// `isLive == true` shows the pulsing 3-dot indicator next to the title
// and forces expanded state. After the tool call settles, the trace
// shrinks back to a capsule that expands on click.
//
// Re-themed against MTPLX `Brand`. The Aphanes `CompactDisclosurePopover`
// path is dropped — chat is a single-conversation surface and inline
// expand is enough.

struct AssistantTraceSurface: View {
    let title: String
    let subtitle: String
    let detail: String
    var activityLog: [String] = []
    let systemName: String
    var isCompact: Bool = false
    var isLive: Bool = false
    var defaultExpanded: Bool = false
    var syncExpansionToDefault: Bool = false

    @State private var isExpanded = false

    private var disclosureAnimation: Animation {
        .spring(response: 0.34, dampingFraction: 0.88, blendDuration: 0.12)
    }

    private var visibleActivityLog: [String] {
        Array(activityLog.suffix(4))
    }

    private var visibleSupplementaryLines: [String] {
        let trimmedDetail = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedSubtitle = subtitle.trimmingCharacters(in: .whitespacesAndNewlines)
        var seen = Set<String>()
        return visibleActivityLog.compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                trimmed != trimmedDetail,
                trimmed != trimmedSubtitle,
                !seen.contains(trimmed)
            else { return nil }
            seen.insert(trimmed)
            return line
        }
    }

    private func toggleExpanded() {
        withAnimation(disclosureAnimation) {
            isExpanded.toggle()
        }
    }

    var body: some View {
        Group {
            if isCompact {
                compactView
            } else {
                fullView
            }
        }
        .onAppear {
            if syncExpansionToDefault {
                isExpanded = defaultExpanded
            } else if defaultExpanded {
                isExpanded = true
            }
        }
        .onChange(of: defaultExpanded) { _, expanded in
            if syncExpansionToDefault {
                isExpanded = expanded
            } else if expanded {
                isExpanded = true
            }
        }
    }

    // MARK: - Compact (capsule above a finished assistant bubble)

    private var compactView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                toggleExpanded()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: systemName)
                        .font(.system(size: 11, weight: .medium))
                    Text(title)
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                    if isLive {
                        ThinkingIndicatorDots()
                    }
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                }
                .foregroundStyle(Brand.typeSecondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(
                    Capsule(style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(isExpanded ? "Collapse reasoning" : "Expand reasoning")

            if isExpanded {
                expandedBody
                    .transition(.opacity.combined(with: .offset(y: -4)))
            }
        }
    }

    // MARK: - Full (used during streaming and as the always-on card)

    private var fullView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: systemName)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
                Text(title)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Brand.typeHi.opacity(0.8))
                if isLive {
                    ThinkingIndicatorDots()
                }
                Spacer(minLength: 8)
                Button {
                    toggleExpanded()
                } label: {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                        .padding(4)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }

            if !subtitle.isEmpty {
                Text(subtitle)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Brand.typeHi)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if !detail.isEmpty {
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(isExpanded ? nil : (isLive ? 4 : 3))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if isExpanded, !visibleSupplementaryLines.isEmpty {
                supplementaryLog
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 1)
                )
        )
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Shared bodies

    private var expandedBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !subtitle.isEmpty {
                Text(subtitle)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Brand.typeHi)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if !detail.isEmpty {
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if !visibleSupplementaryLines.isEmpty {
                supplementaryLog
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 1)
                )
        )
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var supplementaryLog: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(visibleSupplementaryLines.enumerated()), id: \.offset) { _, line in
                HStack(spacing: 6) {
                    Circle()
                        .fill(Brand.typeTertiary)
                        .frame(width: 3, height: 3)
                    Text(line)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(Brand.typeSecondary.opacity(0.7))
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}
