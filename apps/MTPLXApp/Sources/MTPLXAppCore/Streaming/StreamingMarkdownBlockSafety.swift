import Foundation

// MARK: - StreamingMarkdownBlockSafety
//
// Decides, per streaming-document block, whether it is safe to render
// through the settled markdown pipeline while the stream is still
// running. The contract that keeps this near-zero-cost (2026-07-03
// turbo release, founder-directed):
//
//   - The LAST block is always unsafe: it is still growing and must
//     stay a plain `Text` that repaints per snapshot.
//   - A frozen block is safe only when it is fence-neutral: it does
//     not start inside an open ``` fence, and any fences it opens it
//     also closes. Fence-interior blocks stay plain until the closing
//     fence freezes them (matching today's plain-stream behavior).
//
// One linear pass per snapshot over block texts (~tens of KB at 10 Hz
// worst case); frozen safe blocks then render exactly once through the
// cached settled machinery because their views are Equatable on text.
public enum StreamingMarkdownBlockSafety {

    /// Returns one flag per block: `true` = render as settled markdown.
    public static func classify(_ blockTexts: [String]) -> [Bool] {
        guard !blockTexts.isEmpty else { return [] }
        var flags = [Bool](repeating: false, count: blockTexts.count)
        var insideFence = false
        for (index, text) in blockTexts.enumerated() {
            let fences = fenceCount(in: text)
            let opensOrCloses = fences % 2 != 0
            let startsInsideFence = insideFence
            if index < blockTexts.count - 1 {
                flags[index] = !startsInsideFence && !opensOrCloses
            }
            if opensOrCloses {
                insideFence.toggle()
            }
        }
        return flags
    }

    static func fenceCount(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        var count = 0
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("```") {
                count += 1
                index = text.index(index, offsetBy: 3)
                continue
            }
            index = text.index(after: index)
        }
        return count
    }
}
