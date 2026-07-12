import XCTest
@testable import MTPLXAppCore

/// Bind-vs-connect address resolution (issue #109: the app probed
/// http://0.0.0.0:PORT for preflight/health/watchdog, so LAN binds
/// misreported free ports as occupied and the app killed its own healthy
/// daemon after the health-wait timeout).
///
/// SYNC PAIR: mtplx/server_urls.py is the Python twin.
final class ServerURLsTests: XCTestCase {

    func testWildcardAndLocalhostBindsResolveToLoopbackConnectHost() {
        for bind in ["0.0.0.0", "::", "[::]", "", "localhost", "  0.0.0.0  "] {
            XCTAssertEqual(
                MTPLXServerURLs.connectHost(forBind: bind),
                "127.0.0.1",
                "bind \(bind) must connect via loopback"
            )
        }
    }

    func testSpecificHostsPassThroughVerbatim() {
        XCTAssertEqual(MTPLXServerURLs.connectHost(forBind: "127.0.0.1"), "127.0.0.1")
        XCTAssertEqual(MTPLXServerURLs.connectHost(forBind: "192.168.1.20"), "192.168.1.20")
        XCTAssertEqual(MTPLXServerURLs.connectHost(forBind: "my-mac.local"), "my-mac.local")
    }

    func testBaseURLResolvesWildcardAndBracketsIPv6() {
        XCTAssertEqual(
            MTPLXServerURLs.baseURL(bindHost: "0.0.0.0", port: 12321).absoluteString,
            "http://127.0.0.1:12321"
        )
        XCTAssertEqual(
            MTPLXServerURLs.baseURL(bindHost: "192.168.1.20", port: 8000).absoluteString,
            "http://192.168.1.20:8000"
        )
        XCTAssertEqual(
            MTPLXServerURLs.baseURL(bindHost: "fe80::1", port: 8000).absoluteString,
            "http://[fe80::1]:8000"
        )
    }

    func testLoopbackAndWildcardPredicatesMirrorThePythonSets() {
        for host in ["", "127.0.0.1", "::1", "localhost", "[::1]"] {
            XCTAssertTrue(MTPLXServerURLs.isLoopbackBind(host), host)
        }
        for host in ["0.0.0.0", "::", "192.168.1.20"] {
            XCTAssertFalse(MTPLXServerURLs.isLoopbackBind(host), host)
        }
        XCTAssertTrue(MTPLXServerURLs.isWildcardBind("0.0.0.0"))
        XCTAssertTrue(MTPLXServerURLs.isWildcardBind("::"))
        XCTAssertFalse(MTPLXServerURLs.isWildcardBind("127.0.0.1"))
    }

    func testOpenCodeBaseURLStringUsesSharedResolver() {
        XCTAssertEqual(
            OpenCodeIntegration.baseURLString(host: "0.0.0.0", port: 12321),
            "http://127.0.0.1:12321/v1"
        )
        XCTAssertEqual(
            OpenCodeIntegration.baseURLString(host: "192.168.1.20", port: 8000),
            "http://192.168.1.20:8000/v1"
        )
    }

    @MainActor
    func testLaunchBlockedForNonLoopbackBindWithoutAPIKey() async throws {
        // `mtplx serve` exits at argparse for non-loopback binds without an
        // API key; the app must surface the actionable sentence instead of
        // spawning a doomed daemon and reporting a generic "Degraded".
        let settingsURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("lan-guard-\(UUID().uuidString).json")
        let store = MTPLXBackendStore(
            settingsStore: MTPLXSettingsStore(settingsURL: settingsURL)
        )
        var configuration = store.configuration
        configuration.host = "0.0.0.0"
        configuration.apiKey = nil
        try await store.applyConfiguration(configuration, restartIfRunning: false)

        await store.startDaemon(target: nil)

        guard case .degraded(let reason) = store.daemonState else {
            return XCTFail("expected degraded, got \(store.daemonState)")
        }
        XCTAssertTrue(reason.contains("API key"), reason)
    }

    func testPortIsBindableChecksTheWildcardFamilyForWildcardBinds() throws {
        // Occupy a port on INADDR_ANY: a loopback-only check would call it
        // bindable (SO_REUSEADDR + specific-vs-wildcard overlap), but the
        // daemon's own 0.0.0.0 bind would fail.
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { Darwin.close(descriptor) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: INADDR_ANY)
        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bindResult, 0)
        XCTAssertEqual(Darwin.listen(descriptor, 1), 0)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var boundAddress = sockaddr_in()
        _ = withUnsafeMutablePointer(to: &boundAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.getsockname(descriptor, $0, &length)
            }
        }
        let boundPort = Int(UInt16(bigEndian: boundAddress.sin_port))

        XCTAssertFalse(
            PortPreflight.portIsBindable(boundPort, bindHost: "0.0.0.0"),
            "wildcard-family check must see the INADDR_ANY listener"
        )
        // And nextFreePort for a wildcard config must skip it.
        let next = PortPreflight.nextFreePort(after: boundPort - 1, bindHost: "0.0.0.0")
        XCTAssertNotNil(next)
        XCTAssertNotEqual(next, boundPort)
    }
}
