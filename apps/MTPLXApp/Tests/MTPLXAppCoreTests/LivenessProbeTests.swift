import XCTest
@testable import MTPLXAppCore

/// Liveness-probe truth table (2026-07-06 incident: the health watchdog
/// reaped a daemon that was answering /health 200 in 14 ms because the
/// payload stopped matching the app's Codable schema and `try?` turned the
/// DecodingError into a "miss"). A daemon that answers 2xx is alive; only
/// transport failures, timeouts, and non-2xx may count toward reaping.
@MainActor
final class LivenessProbeTests: XCTestCase {

    private final class StubURLProtocol: URLProtocol {
        nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            guard let handler = Self.handler else {
                client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
                return
            }
            do {
                let (response, data) = try handler(request)
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }

        override func stopLoading() {}
    }

    private func makeClient() -> MTPLXAPIClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        return MTPLXAPIClient(
            baseURL: URL(string: "http://127.0.0.1:9")!,
            apiKey: nil,
            session: URLSession(configuration: config)
        )
    }

    private func httpResponse(_ status: Int) -> HTTPURLResponse {
        HTTPURLResponse(
            url: URL(string: "http://127.0.0.1:9/health")!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
    }

    /// Minimal payload satisfying HealthPayload's non-optional fields.
    private var goodHealthJSON: String {
        """
        {
          "ok": true,
          "model": "m",
          "model_path": "/tmp/m",
          "generation_mode": "mtp",
          "load_mtp": true,
          "mtp_enabled": true,
          "depth": 3,
          "profile": {"name": "turbo"},
          "context_window": 262144,
          "active_requests": 0,
          "reasoning_parser": "qwen3"
        }
        """
    }

    override func tearDown() {
        StubURLProtocol.handler = nil
        super.tearDown()
    }

    func testHealthyPayloadDecodes() async {
        StubURLProtocol.handler = { [goodHealthJSON] _ in
            (self.httpResponse(200), Data(goodHealthJSON.utf8))
        }
        let result = await makeClient().livenessWithinDeadline(seconds: 5)
        guard case .healthy(let payload) = result else {
            return XCTFail("expected healthy, got \(result)")
        }
        XCTAssertTrue(payload.ok)
        XCTAssertEqual(payload.model, "m")
    }

    func testAliveDaemonWithUndecodablePayloadIsNotAMiss() async {
        // 200 + JSON that violates the schema (depth as a string): the exact
        // failure shape that killed a healthy daemon. Must be reported alive.
        let poisoned = goodHealthJSON.replacingOccurrences(
            of: "\"depth\": 3",
            with: "\"depth\": \"three\""
        )
        StubURLProtocol.handler = { _ in
            (self.httpResponse(200), Data(poisoned.utf8))
        }
        let result = await makeClient().livenessWithinDeadline(seconds: 5)
        guard case .aliveUndecodable = result else {
            return XCTFail("expected aliveUndecodable, got \(result)")
        }
    }

    func testTransportFailureIsUnreachable() async {
        StubURLProtocol.handler = nil  // startLoading fails with cannotConnectToHost
        let result = await makeClient().livenessWithinDeadline(seconds: 5)
        guard case .unreachable = result else {
            return XCTFail("expected unreachable, got \(result)")
        }
    }

    func testNon2xxIsUnreachable() async {
        StubURLProtocol.handler = { _ in
            (self.httpResponse(503), Data("{}".utf8))
        }
        let result = await makeClient().livenessWithinDeadline(seconds: 5)
        guard case .unreachable = result else {
            return XCTFail("expected unreachable for 503, got \(result)")
        }
    }

    func testAuthRejectionIsAliveNotAMiss() async {
        // /health requires the API key on LAN daemons; a 401/403 proves a
        // live daemon speaking HTTP. A key mismatch is a configuration
        // problem, never grounds for the watchdog to reap (issue #109).
        for status in [401, 403] {
            StubURLProtocol.handler = { _ in
                (self.httpResponse(status), Data("{\"detail\": \"missing or invalid api key\"}".utf8))
            }
            let result = await makeClient().livenessWithinDeadline(seconds: 5)
            guard case .aliveUnauthorized = result else {
                return XCTFail("expected aliveUnauthorized for \(status), got \(result)")
            }
        }
    }

    func testBackCompatShimReturnsPayloadOnlyWhenHealthy() async {
        StubURLProtocol.handler = { [goodHealthJSON] _ in
            (self.httpResponse(200), Data(goodHealthJSON.utf8))
        }
        let healthy = await makeClient().healthWithinDeadline(seconds: 5)
        XCTAssertNotNil(healthy)

        let poisoned = goodHealthJSON.replacingOccurrences(
            of: "\"depth\": 3",
            with: "\"depth\": null"
        )
        StubURLProtocol.handler = { _ in
            (self.httpResponse(200), Data(poisoned.utf8))
        }
        let undecodable = await makeClient().healthWithinDeadline(seconds: 5)
        XCTAssertNil(undecodable)
    }
}
