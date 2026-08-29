import Foundation

// MARK: - RuntimeSetupRow
//
// One checklist row on the onboarding "Setting up MTPLX" step. The
// service publishes full row-set snapshots so the view renders state
// without tracking deltas, and tests assert on the same rows.

public enum RuntimeSetupRowID: String, CaseIterable, Equatable, Sendable {
    case engine
    case fanControl = "fan_control"
    case globalCLI = "global_cli"

    public var title: String {
        switch self {
        case .engine: return "MTPLX engine"
        case .fanControl: return "Fan control"
        case .globalCLI: return "Terminal command line"
        }
    }
}

public enum RuntimeSetupRowState: Equatable, Sendable {
    case pending
    case running
    case done
    /// Non-blocking problem: setup continues, the row explains.
    case warning
    /// Blocking problem: only the engine row can reach this state.
    case failed
}

public struct RuntimeSetupRow: Equatable, Sendable, Identifiable {
    public var id: RuntimeSetupRowID
    public var state: RuntimeSetupRowState
    public var detail: String
    /// Copyable terminal command rendered under the detail (e.g. the
    /// manual pip upgrade for a pip-installed global CLI).
    public var command: String?

    public init(
        id: RuntimeSetupRowID,
        state: RuntimeSetupRowState = .pending,
        detail: String = "",
        command: String? = nil
    ) {
        self.id = id
        self.state = state
        self.detail = detail
        self.command = command
    }

    public var title: String { id.title }
}

// MARK: - RuntimeSetupOutcome

public struct RuntimeSetupOutcome: Equatable, Sendable {
    public var rows: [RuntimeSetupRow]
    /// True when the app-usable runtime is installed and satisfies
    /// the version floor — the only hard requirement to continue.
    public var engineReady: Bool
    public var executablePath: String?

    public init(rows: [RuntimeSetupRow], engineReady: Bool, executablePath: String?) {
        self.rows = rows
        self.engineReady = engineReady
        self.executablePath = executablePath
    }
}

// MARK: - RuntimeSetupEvent

public enum RuntimeSetupEvent: Equatable, Sendable {
    /// Full row-set snapshot; replaces any previous one.
    case rows([RuntimeSetupRow])
    case finished(RuntimeSetupOutcome)
}

// MARK: - RuntimeSetupService
//
// Runs the onboarding "Setting up MTPLX" step: installs the
// app-owned engine from the bundled wheel, makes fan control
// available for honest tuning, and syncs a pre-existing global CLI.
// Idempotent and fast when everything is already in place — the
// engine check is one `mtplx --version`, fan control one
// `mtplx max --status`.
//
// Policy:
// - Engine install is the only blocking phase. Its failure modes are
//   the bootstrapper's actionable errors.
// - Fan control failure degrades to a warning (tuning falls back to
//   safe defaults; the tuner re-checks before measuring anyway).
// - Terminal CLI: the user's terminal always ends up with a current
//   `mtplx` — never a suggestion to fix it themselves. No CLI →
//   install the shim (`~/.mtplx/bin/mtplx` wrapper around the app engine
//   plus a PATH line in `~/.zshrc`, no sudo, LM Studio-style). Stale
//   Homebrew → upgraded through brew; if brew fails, the shim
//   shadows it. Stale anything else (pip, custom, unreadable) → the
//   shim shadows it in place; their file is never touched. The one
//   hands-off case is a CLI *newer* than the app — that's theirs.
//   Source checkouts are dev setups and stay untouched. CLI problems
//   never block — the app itself always resolves its own venv first.

public struct RuntimeSetupService: Sendable {
    public typealias EngineInstaller = @Sendable (@escaping @Sendable (String) -> Void) throws -> URL
    public typealias FanControlEnsurer = @Sendable (URL, @escaping @Sendable (String) -> Void) -> FanControlSetupResult
    public typealias HomebrewUpgrader = @Sendable () throws -> URL

    private let processEnvironment: [String: String]
    private let appVersion: String?
    private let engineInstaller: EngineInstaller
    private let fanControlEnsurer: FanControlEnsurer?
    private let homebrewUpgrader: HomebrewUpgrader?
    private let interruptBox = SubprocessInterruptBox()

    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        appVersion: String? = nil,
        engineInstaller: EngineInstaller? = nil,
        fanControlEnsurer: FanControlEnsurer? = nil,
        homebrewUpgrader: HomebrewUpgrader? = nil
    ) {
        self.processEnvironment = processEnvironment
        self.appVersion = appVersion
            ?? processEnvironment["MTPLX_APP_REQUIRED_RUNTIME_VERSION"]
            ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)
        let environment = processEnvironment
        self.engineInstaller = engineInstaller ?? { status in
            try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate(status: status)
        }
        self.fanControlEnsurer = fanControlEnsurer
        self.homebrewUpgrader = homebrewUpgrader
    }

    // MARK: Stream

    public func stream() -> AsyncStream<RuntimeSetupEvent> {
        AsyncStream { continuation in
            let box = interruptBox
            let rows = RuntimeSetupRowsBox()
            let service = self
            let worker = Task.detached(priority: .userInitiated) {
                continuation.yield(.rows(rows.ordered()))

                // Phase 1 — engine (blocking).
                rows.update(.engine, .running, "Checking MTPLX runtime")
                continuation.yield(.rows(rows.ordered()))
                let executable: URL
                do {
                    executable = try service.engineInstaller { message in
                        rows.update(.engine, .running, message)
                        continuation.yield(.rows(rows.ordered()))
                    }
                } catch {
                    let message = (error as? LocalizedError)?.errorDescription
                        ?? error.localizedDescription
                    rows.update(.engine, .failed, message)
                    continuation.yield(.rows(rows.ordered()))
                    continuation.yield(.finished(RuntimeSetupOutcome(
                        rows: rows.ordered(),
                        engineReady: false,
                        executablePath: nil
                    )))
                    continuation.finish()
                    return
                }
                let engineVersion = MTPLXRuntimeUpdateService.runtimeVersion(
                    executableURL: executable,
                    environment: service.processEnvironment
                )
                rows.update(.engine, .done, Self.engineReadyDetail(version: engineVersion))
                continuation.yield(.rows(rows.ordered()))

                // Phase 2 — fan control (warning-only).
                if Task.isCancelled {
                    continuation.finish()
                    return
                }
                rows.update(.fanControl, .running, "Checking fan control")
                continuation.yield(.rows(rows.ordered()))
                let ensure: FanControlEnsurer = service.fanControlEnsurer ?? { executable, status in
                    FanControlInstaller(processEnvironment: service.processEnvironment)
                        .ensureReady(executable: executable, subprocess: box, status: status)
                }
                let fanControl = ensure(executable) { message in
                    rows.update(.fanControl, .running, message)
                    continuation.yield(.rows(rows.ordered()))
                }
                if fanControl.ok {
                    rows.update(.fanControl, .done, "Fan control ready")
                } else {
                    rows.update(
                        .fanControl,
                        .warning,
                        Self.fanControlWarningDetail(message: fanControl.message)
                    )
                }
                continuation.yield(.rows(rows.ordered()))

                // Phase 3 — terminal CLI install/sync (best-effort).
                if Task.isCancelled {
                    continuation.finish()
                    return
                }
                service.syncGlobalCLI(engineExecutable: executable, rows: rows) {
                    continuation.yield(.rows(rows.ordered()))
                }

                continuation.yield(.finished(RuntimeSetupOutcome(
                    rows: rows.ordered(),
                    engineReady: true,
                    executablePath: executable.path
                )))
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                worker.cancel()
                box.interrupt()
            }
        }
    }

    // MARK: Global CLI

    private func syncGlobalCLI(
        engineExecutable: URL,
        rows: RuntimeSetupRowsBox,
        publish: () -> Void
    ) {
        rows.update(.globalCLI, .running, "Checking for an existing mtplx command")
        publish()

        // The user's login shell is the only honest oracle for which
        // executable `mtplx` actually runs in their terminal — the app's
        // Finder-launched PATH lacks the shell rc's /opt/homebrew/bin
        // ordering and certified a false green (2026-08-28 receipt: setup
        // said "Up to date (2.10.0)" while `command -v MTPLX` served the
        // Homebrew 2.9.2). Fall back to the in-process scan when the probe
        // fails.
        guard let globalCLI = MTPLXCommandBuilder.detectShellWinningCLIExecutable(
            environment: processEnvironment
        ) ?? MTPLXCommandBuilder.detectGlobalCLIExecutable(
            environment: processEnvironment
        ) else {
            // No user-managed CLI anywhere — install the terminal
            // command ourselves (symlink + PATH line, no sudo).
            do {
                let installedNow = try installTerminalShim(engineExecutable: engineExecutable)
                rows.update(
                    .globalCLI,
                    .done,
                    installedNow
                        ? "Installed the mtplx command — open a new terminal to use it."
                        : "mtplx command ready."
                )
            } catch {
                rows.update(
                    .globalCLI,
                    .warning,
                    "Couldn't install the mtplx terminal command (\(error.localizedDescription)). The app is unaffected.",
                    command: MTPLXCommandBuilder.homebrewInstallCommand
                )
            }
            publish()
            return
        }

        let kind = MTPLXRuntimeUpdateService.installKind(
            for: globalCLI,
            environment: processEnvironment
        )
        let rawVersion = MTPLXRuntimeUpdateService.runtimeVersion(
            executableURL: globalCLI,
            environment: processEnvironment
        )
        guard let version = rawVersion.flatMap(MTPLXSemanticVersion.init) else {
            // A CLI we can't even version is broken for the user too —
            // shadow it with the shim so their terminal serves the
            // current engine. Their file stays where it is.
            do {
                try installTerminalShim(engineExecutable: engineExecutable)
                rows.update(
                    .globalCLI,
                    .done,
                    "Replaced an unreadable mtplx at \(globalCLI.path) — open a new terminal to use the updated command."
                )
            } catch {
                rows.update(
                    .globalCLI,
                    .warning,
                    "Found \(globalCLI.path) but couldn't read its version. The app uses its own runtime either way."
                )
            }
            publish()
            return
        }

        let latest = appVersion.flatMap(MTPLXSemanticVersion.init)
        guard let latest, version < latest else {
            rows.update(.globalCLI, .done, "Up to date (\(version)) — \(kind.displayName)")
            publish()
            return
        }

        switch kind {
        case .homebrew:
            guard let upgrade = homebrewUpgrader ?? defaultHomebrewUpgrader() else {
                shimOverStaleCLI(
                    engineExecutable: engineExecutable,
                    rows: rows,
                    oldVersion: version,
                    latest: latest,
                    detailWhenShimmed: "Homebrew was not found, so your terminal now uses the app's CLI (\(latest), was \(version)). Open a new terminal."
                )
                publish()
                return
            }
            rows.update(.globalCLI, .running, "Updating your Homebrew CLI (\(version) → \(latest))")
            publish()
            do {
                let upgraded = try upgrade()
                let upgradedVersion = MTPLXRuntimeUpdateService.runtimeVersion(
                    executableURL: upgraded,
                    environment: processEnvironment
                ) ?? "\(latest)"
                rows.update(.globalCLI, .done, "Homebrew CLI updated to \(upgradedVersion)")
            } catch {
                let message = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
                shimOverStaleCLI(
                    engineExecutable: engineExecutable,
                    rows: rows,
                    oldVersion: version,
                    latest: latest,
                    detailWhenShimmed: "Homebrew didn't update (\(message)), so your terminal now uses the app's CLI (\(latest)). Open a new terminal."
                )
            }
            publish()
        case .sourceCheckout:
            rows.update(
                .globalCLI,
                .done,
                "Source checkout on PATH (\(version)). The app uses its own runtime."
            )
            publish()
        case .pipLike, .appOwned, .custom, .missing:
            // Don't tell the user their CLI is stale — make it current.
            // The shim shadows the old install on PATH; their file is
            // never modified or removed.
            shimOverStaleCLI(
                engineExecutable: engineExecutable,
                rows: rows,
                oldVersion: version,
                latest: latest,
                detailWhenShimmed: "Updated the mtplx command to \(latest) (was \(version)). Open a new terminal to use it."
            )
            publish()
        }
    }

    /// Stale-CLI remediation: put the app engine in front of the old
    /// install on PATH via the terminal shim. Falls back to an honest
    /// warning with the Homebrew command only when the shim itself
    /// cannot be written.
    private func shimOverStaleCLI(
        engineExecutable: URL,
        rows: RuntimeSetupRowsBox,
        oldVersion: MTPLXSemanticVersion,
        latest: MTPLXSemanticVersion,
        detailWhenShimmed: String
    ) {
        do {
            try installTerminalShim(engineExecutable: engineExecutable)
            rows.update(.globalCLI, .done, detailWhenShimmed)
        } catch {
            rows.update(
                .globalCLI,
                .warning,
                "Your mtplx CLI is \(oldVersion); the app ships \(latest). It couldn't be updated automatically (\(error.localizedDescription)).",
                command: MTPLXCommandBuilder.homebrewInstallCommand
            )
        }
    }

    /// Expose the app-owned engine as a terminal command without sudo:
    /// `~/.mtplx/bin/mtplx` wraps the venv binary (a stable path across
    /// app updates) and pins Python bytecode outside the signed application.
    /// `~/.zshrc` gains one guarded PATH line.
    /// Returns true when anything was newly written so the row can say
    /// "open a new terminal" only when it actually changed the shell.
    @discardableResult
    private func installTerminalShim(engineExecutable: URL) throws -> Bool {
        try Self.installTerminalShim(
            engineExecutable: engineExecutable,
            processEnvironment: processEnvironment
        )
    }

    /// Upgrade the direct app-runtime symlink shipped by older builds even
    /// when onboarding is already complete. This is intentionally narrow:
    /// custom, Homebrew, and source-checkout launchers remain untouched.
    @discardableResult
    public static func migrateLegacyTerminalShimIfNeeded(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Bool {
        let home = processEnvironment["HOME"] ?? NSHomeDirectory()
        let binDir = URL(fileURLWithPath: home)
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
        let appRuntimeBin = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeBinDirectory(
                environment: processEnvironment
            )
        ).resolvingSymlinksInPath().path
        let fileManager = FileManager.default

        for commandName in ["mtplx", "MTPLX"] {
            let shim = binDir.appendingPathComponent(commandName)
            guard let destination = try? fileManager.destinationOfSymbolicLink(
                atPath: shim.path
            ) else { continue }
            let destinationURL = destination.hasPrefix("/")
                ? URL(fileURLWithPath: destination)
                : shim.deletingLastPathComponent().appendingPathComponent(destination)
            let resolved = destinationURL.standardizedFileURL.resolvingSymlinksInPath()
            guard resolved.path.hasPrefix(appRuntimeBin + "/") else { continue }
            return try installTerminalShim(
                engineExecutable: resolved,
                processEnvironment: processEnvironment
            )
        }
        return false
    }

    @discardableResult
    private static func installTerminalShim(
        engineExecutable: URL,
        processEnvironment: [String: String]
    ) throws -> Bool {
        let home = processEnvironment["HOME"] ?? NSHomeDirectory()
        let binDir = URL(fileURLWithPath: home)
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: binDir, withIntermediateDirectories: true)

        var changed = false
        let safeEnvironment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
            environment: processEnvironment
        )
        let bytecodeCache = safeEnvironment["PYTHONPYCACHEPREFIX"]!
        let wrapper = """
        #!/bin/sh
        export PYTHONPYCACHEPREFIX=\(Self.shellSingleQuoted(bytecodeCache))
        exec \(Self.shellSingleQuoted(engineExecutable.path)) "$@"
        """ + "\n"

        // Default macOS volumes are case-insensitive, so these names usually
        // resolve to one file. Writing both also protects users who install on
        // a case-sensitive volume and invoke the documented uppercase alias.
        for commandName in ["mtplx", "MTPLX"] {
            let shim = binDir.appendingPathComponent(commandName)
            let existingDestination = try? fileManager.destinationOfSymbolicLink(
                atPath: shim.path
            )
            let existingWrapper = existingDestination == nil
                ? try? String(contentsOf: shim, encoding: .utf8)
                : nil
            if existingDestination != nil || existingWrapper != wrapper {
                if fileManager.fileExists(atPath: shim.path) || existingDestination != nil {
                    let backup = binDir.appendingPathComponent(
                        "\(commandName).pre-wrapper-\(UUID().uuidString)"
                    )
                    try fileManager.moveItem(at: shim, to: backup)
                }
                try wrapper.write(to: shim, atomically: true, encoding: .utf8)
                try fileManager.setAttributes(
                    [.posixPermissions: 0o755],
                    ofItemAtPath: shim.path
                )
                changed = true
            } else if !fileManager.isExecutableFile(atPath: shim.path) {
                try fileManager.setAttributes(
                    [.posixPermissions: 0o755],
                    ofItemAtPath: shim.path
                )
                changed = true
            }
        }

        let zshrc = URL(fileURLWithPath: home).appendingPathComponent(".zshrc")
        let existing = (try? String(contentsOf: zshrc, encoding: .utf8)) ?? ""
        if !existing.contains(".mtplx/bin") {
            let block = """

            # Added by MTPLX.app — terminal command
            export PATH="$HOME/.mtplx/bin:$PATH"
            """
            // Append through a file handle, never an atomic rewrite (#292):
            // atomic write is write-temp-then-rename, which replaces a
            // symlinked ~/.zshrc with a plain file and silently detaches the
            // user's dotfile repo. Appending through the handle follows the
            // link, preserves the inode (hard links and concurrent editors
            // survive), and only ever adds bytes the app authored.
            let payload = Data((block + "\n").utf8)
            if fileManager.fileExists(atPath: zshrc.path) {
                let handle = try FileHandle(forWritingTo: zshrc)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: payload)
            } else {
                try payload.write(to: zshrc)
            }
            changed = true
        }
        return changed
    }

    private static func shellSingleQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private func defaultHomebrewUpgrader() -> HomebrewUpgrader? {
        guard MTPLXCommandBuilder.resolveHomebrewExecutable(environment: processEnvironment) != nil else {
            return nil
        }
        let environment = processEnvironment
        return {
            try MTPLXRuntimeBootstrapper(environment: environment).upgradeHomebrewRuntime()
        }
    }

    // MARK: Helpers

    private static func engineReadyDetail(version: String?) -> String {
        if let version, !version.isEmpty {
            return "MTPLX \(version) ready"
        }
        return "MTPLX runtime ready"
    }

    private static func fanControlWarningDetail(message: String) -> String {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Fan control unavailable — tuning will use safe defaults."
        }
        return "Fan control unavailable — tuning will use safe defaults. (\(trimmed))"
    }
}

// MARK: - RuntimeSetupRowsBox
//
// Lock-guarded row storage so the engine/fan-control status callbacks
// (which Swift 6 treats as concurrently-executing) can update rows
// without capturing mutable state.

private final class RuntimeSetupRowsBox: @unchecked Sendable {
    private let lock = NSLock()
    private var rows: [RuntimeSetupRowID: RuntimeSetupRow]

    init() {
        var initial: [RuntimeSetupRowID: RuntimeSetupRow] = [:]
        for id in RuntimeSetupRowID.allCases {
            initial[id] = RuntimeSetupRow(id: id)
        }
        rows = initial
    }

    func update(
        _ id: RuntimeSetupRowID,
        _ state: RuntimeSetupRowState,
        _ detail: String,
        command: String? = nil
    ) {
        lock.lock()
        rows[id] = RuntimeSetupRow(id: id, state: state, detail: detail, command: command)
        lock.unlock()
    }

    func ordered() -> [RuntimeSetupRow] {
        lock.lock()
        defer { lock.unlock() }
        return RuntimeSetupRowID.allCases.compactMap { rows[$0] }
    }
}
