import XCTest

@testable import MTPLXAppCore

/// Pins the daemon → app contract for the machine memory plan (issue #305),
/// including the streamed n-gram table field (2026-08-28): the Memory
/// Detail card sources it to explain where the other ~30G of a Flash-Next
/// pack lives. Older daemons omit the key — decode must not fail.
final class MemoryPlanDecodeTests: XCTestCase {
    func testDecodesStreamedNgramTableBytes() throws {
        let json = Data("""
        {
          "available": true,
          "total_ram_bytes": 137438953472,
          "usable_bytes": 103079215104,
          "model_weights_bytes": 74302880768,
          "ngram_table_streamed_bytes": 31998345216,
          "context_window_resolved": 262144,
          "model_fits": true
        }
        """.utf8)
        let plan = try JSONDecoder().decode(MemoryPlanStatus.self, from: json)
        XCTAssertEqual(plan.ngramTableStreamedBytes, 31_998_345_216)
        XCTAssertEqual(plan.modelWeightsBytes, 74_302_880_768)
    }

    func testOlderDaemonWithoutTableFieldStillDecodes() throws {
        let json = Data("""
        {"available": true, "model_weights_bytes": 20401094656}
        """.utf8)
        let plan = try JSONDecoder().decode(MemoryPlanStatus.self, from: json)
        XCTAssertNil(plan.ngramTableStreamedBytes)
    }
}
