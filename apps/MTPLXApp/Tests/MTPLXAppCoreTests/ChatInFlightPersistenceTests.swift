import Foundation
import SwiftData
import XCTest

@testable import MTPLXAppCore

// MARK: - ChatInFlightPersistenceTests (issue #324)
//
// Pins the per-conversation ownership of in-flight turns:
//   1. Switching conversations mid-stream must NOT destroy the
//      accumulating response, must NOT cancel the server request, and
//      the finished turn must persist into ITS conversation while
//      another conversation is visible.
//   2. Switching back mid-stream re-attaches the live surface with the
//      partial content intact.
//   3. Sending in conversation B while A still streams works, and each
//      turn persists into its own conversation.
//
// Fake-daemon pattern mirrors MTPLXAppCoreTests: a Python SSE server
// per test. Conversation A's stream emits a first fragment, then holds
// until the test creates a RELEASE file — so "the user switched away
// mid-stream" is deterministic, not timing-lucky. Any hit on the
// cancel endpoint drops a CANCEL marker file; its absence proves the
// switch did not cancel generation server-side.

final class ChatInFlightPersistenceTests: XCTestCase {

    // MARK: Scenario 1 — switch away, let the stream finish in the background

    @MainActor
    func testSwitchAwayMidStreamKeepsRequestAliveAndPersistsIntoOwningConversation() async throws {
        let daemon = try await Self.startHoldableChatDaemon()
        defer { daemon.process.terminate() }
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: daemon.baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("prompt-a")
        let conversationA = try XCTUnwrap(viewModel.current)
        try await poll("first fragment reaches A's live surface") {
            viewModel.streamingContent.contains("Alpha part one.")
        }
        XCTAssertTrue(viewModel.isStreaming)

        // Switch away mid-stream.
        let conversationB = viewModel.createNewConversation()
        XCTAssertEqual(viewModel.current?.id, conversationB.id)

        // B's surface is idle — A's in-flight turn must not leak into it.
        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertFalse(viewModel.shouldRenderStreamingAssistant)
        XCTAssertFalse(viewModel.hasStreamingContent)
        XCTAssertEqual(viewModel.streamingContent, "")
        XCTAssertEqual(viewModel.streamingPhase, .idle)
        XCTAssertNil(viewModel.currentTurnGroupID)

        // The server request must keep running: no cancel call fired.
        XCTAssertFalse(FileManager.default.fileExists(atPath: daemon.cancelMarkerURL.path))

        // Let the held stream finish while B is visible.
        try daemon.release()
        try await poll("A's finished turn is persisted while B is visible") {
            let persisted = try Self.assistantMessages(in: container, conversationID: conversationA.id)
            return persisted.contains { $0.visibleContent == "Alpha part one. Alpha part two." }
        }

        let persisted = try Self.assistantMessages(in: container, conversationID: conversationA.id)
        XCTAssertEqual(persisted.count, 1)
        XCTAssertEqual(persisted.first?.visibleContent, "Alpha part one. Alpha part two.")
        XCTAssertEqual(persisted.first?.finishReason, "stop")
        XCTAssertFalse(FileManager.default.fileExists(atPath: daemon.cancelMarkerURL.path))
        // Nothing bled into B.
        XCTAssertTrue(try Self.assistantMessages(in: container, conversationID: conversationB.id).isEmpty)
        XCTAssertFalse(viewModel.isStreaming)

        // Returning to A shows the settled turn.
        viewModel.select(conversationA)
        XCTAssertEqual(viewModel.visibleMessages.map(\.role), [.user, .assistant])
        XCTAssertEqual(
            viewModel.visibleMessages.last?.visibleContent,
            "Alpha part one. Alpha part two."
        )
        XCTAssertFalse(viewModel.shouldRenderStreamingAssistant)
    }

    // MARK: Scenario 2 — switch away and back: the partial survives, then completes

    @MainActor
    func testSwitchBackMidStreamShowsSurvivingPartialAndCompletion() async throws {
        let daemon = try await Self.startHoldableChatDaemon()
        defer { daemon.process.terminate() }
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: daemon.baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("prompt-a")
        let conversationA = try XCTUnwrap(viewModel.current)
        try await poll("first fragment reaches A's live surface") {
            viewModel.streamingContent.contains("Alpha part one.")
        }
        let liveTurnID = try XCTUnwrap(viewModel.currentTurnGroupID)

        _ = viewModel.createNewConversation()
        XCTAssertFalse(viewModel.isStreaming)

        // Back to A: the live surface re-attaches with the partial intact
        // and the SAME turn identity — nothing was torn down.
        viewModel.select(conversationA)
        XCTAssertTrue(viewModel.isStreaming)
        XCTAssertTrue(viewModel.shouldRenderStreamingAssistant)
        XCTAssertTrue(viewModel.hasStreamingContent)
        XCTAssertTrue(viewModel.streamingContent.contains("Alpha part one."))
        XCTAssertEqual(viewModel.currentTurnGroupID, liveTurnID)
        XCTAssertFalse(
            viewModel.streamingContentDocument.rawText.isEmpty
                && viewModel.streamingContentPending.isEmpty,
            "the re-attached live document/buffer must still hold the partial"
        )

        // The held stream keeps growing into the SAME visible surface.
        try daemon.release()
        try await poll("stream completes and settles into A's transcript") {
            viewModel.visibleMessages.last?.role == .assistant
                && viewModel.visibleMessages.last?.visibleContent == "Alpha part one. Alpha part two."
        }
        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.visibleMessages.last?.turnGroupID, liveTurnID)
        XCTAssertFalse(FileManager.default.fileExists(atPath: daemon.cancelMarkerURL.path))
    }

    // MARK: Scenario 3 — a second conversation streams while the first is in flight

    @MainActor
    func testSendingInSecondConversationWhileFirstStreamsPersistsBothIntoOwnConversations() async throws {
        let daemon = try await Self.startHoldableChatDaemon()
        defer { daemon.process.terminate() }
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: daemon.baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("prompt-a")
        let conversationA = try XCTUnwrap(viewModel.current)
        try await poll("first fragment reaches A's live surface") {
            viewModel.streamingContent.contains("Alpha part one.")
        }

        let conversationB = viewModel.createNewConversation()
        viewModel.send("prompt-b")
        XCTAssertTrue(viewModel.isStreaming, "B's own turn starts while A's is still in flight")

        try await poll("B's turn settles into B") {
            viewModel.visibleMessages.last?.role == .assistant
                && viewModel.visibleMessages.last?.visibleContent == "Beta reply."
        }
        // A is still generating server-side the whole time.
        XCTAssertTrue(try Self.assistantMessages(in: container, conversationID: conversationA.id).isEmpty)

        try daemon.release()
        try await poll("A's turn settles into A") {
            let persisted = try Self.assistantMessages(in: container, conversationID: conversationA.id)
            return persisted.contains { $0.visibleContent == "Alpha part one. Alpha part two." }
        }

        let inA = try Self.assistantMessages(in: container, conversationID: conversationA.id)
        let inB = try Self.assistantMessages(in: container, conversationID: conversationB.id)
        XCTAssertEqual(inA.map(\.visibleContent), ["Alpha part one. Alpha part two."])
        XCTAssertEqual(inB.map(\.visibleContent), ["Beta reply."])
        XCTAssertFalse(FileManager.default.fileExists(atPath: daemon.cancelMarkerURL.path))

        viewModel.select(conversationA)
        XCTAssertEqual(viewModel.visibleMessages.map(\.role), [.user, .assistant])
        XCTAssertEqual(
            viewModel.visibleMessages.last?.visibleContent,
            "Alpha part one. Alpha part two."
        )
    }

    // MARK: - Fake daemon

    /// A running fake chat daemon whose "prompt-a" stream holds after
    /// its first content fragment until `release()` is called.
    /// "prompt-b" streams to completion immediately. Scratch files live
    /// under the OS temp directory (purged by the system).
    private struct HoldableChatDaemon {
        let process: Process
        let baseURL: URL
        let releaseURL: URL
        let cancelMarkerURL: URL

        func release() throws {
            try Data("go".utf8).write(to: releaseURL)
        }
    }

    private static func startHoldableChatDaemon() async throws -> HoldableChatDaemon {
        let port = try freeTCPPort()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-inflight-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let releaseURL = directory.appendingPathComponent("release-stream-a")
        let cancelMarkerURL = directory.appendingPathComponent("cancel-was-called")
        let script = directory.appendingPathComponent("fake-holdable-chat-daemon")
        try """
        #!/bin/sh
        exec python3 -u - <<'PY'
        import json
        import os
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        PORT = \(port)
        RELEASE = r'''\(releaseURL.path)'''
        CANCEL_MARKER = r'''\(cancelMarkerURL.path)'''

        def sse(payload):
            return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length else b""
                if self.path.startswith("/v1/mtplx/cancel/"):
                    with open(CANCEL_MARKER, "wb") as f:
                        f.write(b"cancelled")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b"{\\"ok\\": true}")
                    return
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    if b"prompt-b" in body:
                        self.stream_b()
                    else:
                        self.stream_a()
                except BrokenPipeError:
                    return

            def stream_a(self):
                self.wfile.write(sse({
                    "id": "chatcmpl-a",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                }))
                self.wfile.write(sse({
                    "id": "chatcmpl-a",
                    "choices": [{"index": 0, "delta": {"content": "Alpha part one. "}}],
                }))
                self.wfile.flush()
                for _ in range(1200):
                    if os.path.exists(RELEASE):
                        break
                    time.sleep(0.025)
                self.wfile.write(sse({
                    "id": "chatcmpl-a",
                    "choices": [{"index": 0, "delta": {"content": "Alpha part two."}}],
                }))
                self.wfile.write(sse({
                    "id": "chatcmpl-a",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
                    "mtplx_stats": {"raw_decode_tok_s": 41.0},
                }))
                self.wfile.write(b"data: [DONE]\\n\\n")
                self.wfile.flush()

            def stream_b(self):
                self.wfile.write(sse({
                    "id": "chatcmpl-b",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                }))
                self.wfile.write(sse({
                    "id": "chatcmpl-b",
                    "choices": [{"index": 0, "delta": {"content": "Beta reply."}}],
                }))
                self.wfile.write(sse({
                    "id": "chatcmpl-b",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    "mtplx_stats": {"raw_decode_tok_s": 44.0},
                }))
                self.wfile.write(b"data: [DONE]\\n\\n")
                self.wfile.flush()

        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
        PY
        """.data(using: .utf8)!.write(to: script)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path
        )

        let process = Process()
        process.executableURL = script
        try process.run()

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }
        return HoldableChatDaemon(
            process: process,
            baseURL: baseURL,
            releaseURL: releaseURL,
            cancelMarkerURL: cancelMarkerURL
        )
    }

    // MARK: - Helpers

    /// Polls a MainActor condition until it holds or the deadline
    /// passes, then asserts it.
    @MainActor
    private func poll(
        _ label: String,
        timeout: TimeInterval = 10,
        until condition: @MainActor () throws -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if try condition() { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(try condition(), "timed out waiting for: \(label)", file: file, line: line)
    }

    @MainActor
    private static func assistantMessages(
        in container: ModelContainer,
        conversationID: UUID
    ) throws -> [ChatMessage] {
        let descriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate<ChatMessage> { message in
                message.conversationID == conversationID
            },
            sortBy: [SortDescriptor(\.createdAt)]
        )
        return try container.mainContext.fetch(descriptor)
            .filter { $0.role == .assistant }
    }

    private static func freeTCPPort() throws -> Int {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        defer { Darwin.close(socketFD) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }

        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var boundAddress = sockaddr_in()
        let nameResult = withUnsafeMutablePointer(to: &boundAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &length)
            }
        }
        guard nameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        return Int(UInt16(bigEndian: boundAddress.sin_port))
    }
}
