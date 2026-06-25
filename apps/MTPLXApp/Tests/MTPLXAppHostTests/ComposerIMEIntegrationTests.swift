import AppKit
import SwiftUI
import XCTest
@testable import MTPLXAppHost

/// TEMPORARY verification harness — added and reverted within this PR.
///
/// This target (`MTPLXAppHostTests`) and its Package.swift entry are added by one
/// commit and removed by the final commit, so the PR's net diff is the fix alone.
/// They drive the *real* `ComposerInputTextView.Coordinator` against a live
/// `NSTextView`, exercising the actual AppKit IME mechanism (`setMarkedText` /
/// `insertText`) so the fix is verifiable in-tree. Shippable regression coverage
/// lives in `MTPLXAppCoreTests/ComposerTextSyncTests` (the pure decision logic).
///
/// To see the before/after, check out the harness commit (fails) then the fix
/// commit (passes), running:
///   cd apps/MTPLXApp && swift test --filter ComposerIMEIntegrationTests
final class ComposerIMEIntegrationTests: XCTestCase {
    @MainActor
    private func makeComposer(boundTo text: Binding<String>) -> ComposerInputTextView {
        ComposerInputTextView(
            text: text,
            measuredHeight: .constant(40),
            minHeight: 40,
            maxHeight: 144,
            onSubmit: {},
            onFileDrop: { _ in }
        )
    }

    /// The bug: IME composition (here Japanese kana-kanji, but the same holds
    /// for any composed input — pinyin, hangul, dead-key accents) leaked into /
    /// got wiped from the composer. With the fix, provisional preedit must NOT
    /// reach the binding, and the committed text must arrive intact once
    /// the composition commits.
    @MainActor
    func testPreeditStaysOutOfBindingUntilCommitted() {
        var bound = ""
        let composer = makeComposer(boundTo: Binding(get: { bound }, set: { bound = $0 }))
        let coordinator = ComposerInputTextView.Coordinator(parent: composer)

        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 240, height: 40))
        textView.delegate = coordinator

        // Simulate an in-flight IME composition (the romaji-to-kana preedit).
        textView.setMarkedText(
            "にほんご",
            selectedRange: NSRange(location: 4, length: 0),
            replacementRange: NSRange(location: 0, length: 0)
        )
        XCTAssertTrue(textView.hasMarkedText(), "expected an active IME composition")

        // The delegate fires on every preedit keystroke. The binding must stay
        // empty: publishing the provisional text round-trips through SwiftUI and
        // races (and historically destroyed) the live composition.
        coordinator.textDidChange(Notification(name: NSText.didChangeNotification, object: textView))
        XCTAssertEqual(bound, "", "IME preedit must not leak into the SwiftUI binding")

        // Confirm the conversion: the IME replaces the marked range with kanji.
        // (Headless, off-window, NSTextView's insertText may append rather than
        // replace the marked range, so assert against the text view's own settled
        // string instead of a hard-coded literal — the production contract is
        // "publish textView.string verbatim once marked text clears".)
        textView.insertText("日本語", replacementRange: textView.markedRange())
        XCTAssertFalse(textView.hasMarkedText(), "composition should be committed")

        // Now the delegate may publish the settled text up into the binding.
        coordinator.textDidChange(Notification(name: NSText.didChangeNotification, object: textView))
        XCTAssertEqual(bound, textView.string, "binding must catch up to the committed text")
        XCTAssertTrue(bound.contains("日本語"), "committed kanji must reach the binding")
        XCTAssertFalse(bound.isEmpty, "committed text must not be empty")
    }

    /// Plain ASCII never goes through a marked-text phase, so it should publish
    /// immediately — proving the guard does not regress normal typing.
    @MainActor
    func testAsciiTypingPublishesImmediately() {
        var bound = ""
        let composer = makeComposer(boundTo: Binding(get: { bound }, set: { bound = $0 }))
        let coordinator = ComposerInputTextView.Coordinator(parent: composer)

        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 240, height: 40))
        textView.delegate = coordinator

        textView.insertText("hello", replacementRange: NSRange(location: 0, length: 0))
        XCTAssertFalse(textView.hasMarkedText())

        coordinator.textDidChange(Notification(name: NSText.didChangeNotification, object: textView))
        XCTAssertEqual(bound, "hello")
    }

    /// Documents the hazard the `updateNSView` guard prevents: writing into the
    /// text view's `string` while marked text is active tears down the IME
    /// composition. This is why `ComposerTextSync.shouldApplyBindingToTextView`
    /// returns false during composition.
    @MainActor
    func testWritingStringDuringCompositionDestroysIt() {
        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 240, height: 40))
        textView.setMarkedText(
            "にほんご",
            selectedRange: NSRange(location: 4, length: 0),
            replacementRange: NSRange(location: 0, length: 0)
        )
        XCTAssertTrue(textView.hasMarkedText())

        // The pre-fix behaviour: a stale programmatic write-back nukes the session.
        textView.string = "に"
        XCTAssertFalse(
            textView.hasMarkedText(),
            "writing string mid-composition drops the IME session — the guard must avoid this"
        )
    }
}
