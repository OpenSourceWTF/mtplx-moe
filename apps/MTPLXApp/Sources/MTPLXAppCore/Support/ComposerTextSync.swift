import Foundation

// MARK: - ComposerTextSync

/// Pure decision logic for bridging a SwiftUI `@Binding<String>` with an AppKit
/// `NSTextView` in the chat composer.
///
/// The composer is an `NSViewRepresentable`. Two flows can both mutate the
/// string, and they race during IME (input method) composition:
///
///   * `textDidChange` publishes the text view's string up into the binding.
///   * `updateNSView` writes the binding's value back down into the text view.
///
/// While an IME composition is in flight — any input method that builds a
/// character from intermediate keystrokes: CJK (pinyin, kana-kanji, hangul),
/// dead-key accents (`´`+`e` → `é`), and more — the text view holds *marked
/// text* (the uncommitted preedit, e.g. `にほんご` mid-conversion to `日本語`).
/// Marked text mutates on every keystroke, but SwiftUI binding propagation lags
/// by a render cycle. If `updateNSView` writes a stale binding value back into
/// the text view while marked text is active, AppKit tears down the composition
/// session and the in-progress characters vanish — which is why CJK and other
/// IME-composed input "disappears" in the composer.
///
/// These helpers isolate the "should I touch the string?" decisions so they can
/// be unit-tested without a live `NSTextView`, keeping the AppKit bridge a thin
/// caller. The single rule both encode: never move provisional preedit across
/// the bridge — defer until the composition commits.
public enum ComposerTextSync {
    /// `updateNSView`: should the SwiftUI binding value be written back into the
    /// text view's `string`?
    ///
    /// Only when the value genuinely diverged *and* no IME composition is
    /// active. Writing during marked text destroys the composition.
    public static func shouldApplyBindingToTextView(
        hasMarkedText: Bool,
        textViewString: String,
        binding: String
    ) -> Bool {
        !hasMarkedText && textViewString != binding
    }

    /// `textDidChange`: should this text-view edit be published up into the
    /// SwiftUI binding?
    ///
    /// Not while marked text is active: that "string" is provisional preedit,
    /// and round-tripping it through SwiftUI races the live composition. The
    /// committed text arrives in a later `textDidChange` with no marked text.
    public static func shouldPublishEdit(hasMarkedText: Bool) -> Bool {
        !hasMarkedText
    }
}
