import XCTest
@testable import MTPLXAppCore

final class ComposerTextSyncTests: XCTestCase {
    // MARK: shouldApplyBindingToTextView (updateNSView write-back)

    func testDoesNotWriteBindingBackWhileComposing() {
        // An IME preedit (here `にほんご`, but the same holds for any composed
        // input) has advanced past the lagging binding value (`に`). Writing the
        // stale binding back here would tear down the marked-text session and
        // drop the input. It must not happen.
        XCTAssertFalse(
            ComposerTextSync.shouldApplyBindingToTextView(
                hasMarkedText: true,
                textViewString: "にほんご",
                binding: "に"
            )
        )
    }

    func testWritesBindingBackWhenDivergedAndNotComposing() {
        XCTAssertTrue(
            ComposerTextSync.shouldApplyBindingToTextView(
                hasMarkedText: false,
                textViewString: "old",
                binding: "new"
            )
        )
    }

    func testDoesNotWriteBindingBackWhenAlreadyInSync() {
        XCTAssertFalse(
            ComposerTextSync.shouldApplyBindingToTextView(
                hasMarkedText: false,
                textViewString: "same",
                binding: "same"
            )
        )
    }

    // MARK: shouldPublishEdit (textDidChange publish-up)

    func testDoesNotPublishEditWhileComposing() {
        XCTAssertFalse(ComposerTextSync.shouldPublishEdit(hasMarkedText: true))
    }

    func testPublishesEditOnceCompositionCommits() {
        XCTAssertTrue(ComposerTextSync.shouldPublishEdit(hasMarkedText: false))
    }
}
