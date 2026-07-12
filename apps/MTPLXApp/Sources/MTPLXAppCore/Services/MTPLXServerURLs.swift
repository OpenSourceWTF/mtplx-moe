import Foundation

// MARK: - MTPLXServerURLs
//
// Separates the daemon's BIND address from the address the app must CONNECT
// to. A user who serves on the LAN sets host 0.0.0.0 (or ::), which is a
// valid bind address but not a connectable one: URLSession does not treat
// http://0.0.0.0 as loopback, so every app-side probe (port preflight,
// startup health wait, watchdog, chat, metrics) stalls or fails against a
// perfectly healthy daemon (issue #109 — the app killed its own daemon
// after the 300 s health timeout and misreported free ports as occupied).
//
// SYNC PAIR: mtplx/server_urls.py (connect_host_for_bind / url_host) is the
// Python twin used by the CLI. Update both sides together.

public enum MTPLXServerURLs {
    /// Hosts that mean "this machine, loopback reachable without an API key"
    /// — mirrors LOCALHOST_BINDS in mtplx/commands/public.py and
    /// mtplx/server/openai.py.
    public static func isLoopbackBind(_ host: String) -> Bool {
        let raw = unbracketed(host).lowercased()
        return raw.isEmpty || raw == "127.0.0.1" || raw == "::1" || raw == "localhost"
    }

    /// True for wildcard binds that listen on every interface.
    public static func isWildcardBind(_ host: String) -> Bool {
        let raw = unbracketed(host).lowercased()
        return raw == "0.0.0.0" || raw == "::"
    }

    /// The address the app should CONNECT to for a given bind host.
    /// Wildcards and localhost aliases resolve to 127.0.0.1 (the app always
    /// runs on the same machine as the daemon it launched); anything else is
    /// taken verbatim.
    public static func connectHost(forBind host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let raw = unbracketed(trimmed).lowercased()
        switch raw {
        case "", "0.0.0.0", "::", "localhost":
            return "127.0.0.1"
        default:
            return trimmed
        }
    }

    /// Host formatted for a URL: bare IPv6 literals gain brackets.
    public static func urlHost(_ host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.contains(":"), !trimmed.hasPrefix("[") {
            return "[\(trimmed)]"
        }
        return trimmed
    }

    /// Connectable base URL for a daemon bound to `bindHost`:`port`.
    public static func baseURL(bindHost: String, port: Int) -> URL {
        let host = urlHost(connectHost(forBind: bindHost))
        return URL(string: "http://\(host):\(port)")!
    }

    private static func unbracketed(_ host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("["), trimmed.hasSuffix("]") {
            return String(trimmed.dropFirst().dropLast())
        }
        return trimmed
    }
}
