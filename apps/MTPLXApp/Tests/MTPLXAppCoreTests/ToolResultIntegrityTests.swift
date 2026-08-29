import XCTest
@testable import MTPLXAppCore

// Issue #349: "No response from any tools after install." A fresh-install
// user asked the built-in chat about a project in ~/Dev; the model invoked
// tools (`ls`, `find`, `search_files`, `read_file`, `echo`) and received
// NOTHING back — no output, no error. These tests pin the app-side
// truthfulness invariants:
//
//   1. Any tool call the app cannot execute produces a NON-EMPTY,
//      explanatory result string (never a blank the model reads as "the
//      call went into the void").
//   2. That result survives request-payload compaction and reaches the
//      conversation payload for the next turn.
//   3. The Settings "Agent workspace" folder actually reaches the hermes
//      subprocess surfaces (launch environment + terminal.cwd config) —
//      the working-folder plumbing the reporter assumed was broken.
final class ToolResultIntegrityTests: XCTestCase {

    // MARK: - 1. Unknown / unexecuted tools never yield a blank result

    func testUnknownToolDispatchReturnsNonEmptyTruthfulResult() async {
        let factory = MTPLXChatToolFactory()
        for name in ["terminal", "search_files", "read_file", "ls", "echo"] {
            let result = await factory.dispatch(
                name: name,
                argumentsJSON: #"{"command":"ls ~/Dev"}"#
            )
            XCTAssertFalse(
                result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "dispatch(\(name)) returned a blank tool result"
            )
            XCTAssertTrue(result.contains("unknown_tool"), result)
            XCTAssertTrue(result.contains(name), result)
        }
    }

    func testFetchURLWithInvalidURLReturnsNonEmptyError() async {
        let factory = MTPLXChatToolFactory()
        let result = await factory.dispatch(
            name: "fetch_url",
            argumentsJSON: #"{"url":"not a url"}"#
        )
        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.contains("invalid_url"), result)
    }

    @MainActor
    func testUnexecutedToolResultJSONIsNonEmptyAndNamesTheTool() throws {
        let json = ChatViewModel.unexecutedToolResultJSON(toolName: "terminal")
        XCTAssertTrue(json.contains("tool_not_executed"), json)
        XCTAssertTrue(json.contains("terminal"), json)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let note = try XCTUnwrap(object["note"] as? String)
        XCTAssertFalse(note.isEmpty)
        XCTAssertTrue(note.contains("did not execute"))

        // A nameless call still yields a truthful, non-empty result.
        let nameless = ChatViewModel.unexecutedToolResultJSON(toolName: "")
        XCTAssertFalse(nameless.isEmpty)
        XCTAssertTrue(nameless.contains("tool_not_executed"), nameless)
    }

    // MARK: - 2. Results survive compaction and reach the request payload

    @MainActor
    func testCompactToolResultContentNeverEmptiesErrorResults() {
        let short = ChatViewModel.unexecutedToolResultJSON(toolName: "read_file")
        XCTAssertEqual(ChatViewModel.compactToolResultContent(short), short)

        let overLimit =
            "{\"error\":\"tool_not_executed\",\"content\":\""
            + String(repeating: "a", count: ChatViewModel.requestToolResultContentLimit + 8_000)
            + "\"}"
        let compacted = ChatViewModel.compactToolResultContent(overLimit)
        XCTAssertFalse(
            compacted.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "compaction turned an oversized tool result into a blank"
        )

        let overLimitNonJSON = String(
            repeating: "x",
            count: ChatViewModel.requestToolResultContentLimit + 5_000
        )
        let clamped = ChatViewModel.compactToolResultContent(overLimitNonJSON)
        XCTAssertFalse(clamped.isEmpty)
    }

    @MainActor
    func testBuildRequestMessagesCarriesNonEmptyToolResultForEveryCall() throws {
        let conversation = ChatConversation(title: "Dangling repair")
        let user = ChatMessage(
            role: .user,
            visibleContent: "look at ~/Dev",
            createdAt: Date(timeIntervalSince1970: 100),
            conversation: conversation
        )
        let records = [
            ToolCallRecord(
                id: "call_term",
                name: "terminal",
                arguments: #"{"command":"ls ~/Dev"}"#
            )
        ]
        let toolCallsJSON = try XCTUnwrap(
            String(data: JSONEncoder().encode(records), encoding: .utf8)
        )
        let assistant = ChatMessage(
            role: .assistant,
            visibleContent: "",
            toolCallsJSON: toolCallsJSON,
            finishReason: "tool_calls",
            createdAt: Date(timeIntervalSince1970: 101),
            conversation: conversation
        )
        let toolResult = ChatMessage(
            role: .tool,
            visibleContent: ChatViewModel.unexecutedToolResultJSON(toolName: "terminal"),
            toolCallId: "call_term",
            createdAt: Date(timeIntervalSince1970: 102),
            conversation: conversation
        )

        let request = ChatViewModel.buildRequestMessages(
            from: [user, assistant, toolResult],
            overrideLastUserContent: nil
        )

        let assistantEntry = try XCTUnwrap(
            request.first { $0.role == "assistant" }
        )
        XCTAssertEqual(assistantEntry.toolCalls?.first?.id, "call_term")

        // The load-bearing pin: every replayed assistant tool call has a
        // matching tool message whose content is non-empty. An empty
        // content here is exactly issue #349's "no result or output back".
        let toolEntry = try XCTUnwrap(
            request.first { $0.role == "tool" && $0.toolCallId == "call_term" }
        )
        let content = try XCTUnwrap(toolEntry.content)
        XCTAssertFalse(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertTrue(content.contains("tool_not_executed"), content)
    }

    // MARK: - 3. The Settings workspace folder reaches the hermes subprocess

    func testAgentWorkspaceFromSettingsReachesHermesLaunchEnvironment() throws {
        let workspace = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-349-workspace-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)

        let configuration = MTPLXAppConfiguration(hermesWorkspacePath: workspace.path)
        let integration = HermesIntegration(
            hermesHome: FileManager.default.temporaryDirectory
                .appendingPathComponent("mtplx-349-hermes-home-\(UUID().uuidString)", isDirectory: true)
        )

        XCTAssertEqual(
            HermesIntegration.resolvedWorkspacePath(configuration: configuration),
            workspace.path
        )
        let env = integration.launchEnvironment(configuration: configuration)
        XCTAssertEqual(env["HERMES_WORKSPACE"], workspace.path)
        XCTAssertEqual(env["TERMINAL_CWD"], workspace.path)
    }

    func testAgentWorkspaceFromSettingsReachesHermesTerminalCwdConfig() throws {
        let workspace = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-349-workspace-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let home = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-349-hermes-home-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)

        let configuration = MTPLXAppConfiguration(hermesWorkspacePath: workspace.path)
        let integration = HermesIntegration(hermesHome: home)
        let result = try integration.sync(configuration: configuration)

        XCTAssertEqual(result.workspacePath, workspace.path)
        let configText = try String(
            contentsOf: URL(fileURLWithPath: result.configPath),
            encoding: .utf8
        )
        XCTAssertTrue(
            configText.contains("cwd: '\(workspace.path)'"),
            "terminal.cwd missing from hermes config:\n\(configText)"
        )
        let envText = try String(
            contentsOf: URL(fileURLWithPath: result.envPath),
            encoding: .utf8
        )
        XCTAssertTrue(envText.contains("TERMINAL_CWD="), envText)
        XCTAssertTrue(envText.contains(workspace.path), envText)
    }
}
