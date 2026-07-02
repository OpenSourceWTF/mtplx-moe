import Combine
import Foundation
#if canImport(AppKit)
import AppKit
#endif

public enum PendingModelDownloadLaunchAction: String, Equatable, Sendable {
    case start
    case restart
}

public struct PendingModelDownload: Identifiable, Equatable, Sendable {
    public var id: String
    public var repoID: String
    public var displayName: String
    public var shortName: String
    public var target: LaunchTarget?
    public var launchAction: PendingModelDownloadLaunchAction
    public var totalBytes: Int64?
    public var destinationPath: String

    public init(
        repoID: String,
        displayName: String,
        shortName: String,
        target: LaunchTarget?,
        launchAction: PendingModelDownloadLaunchAction,
        totalBytes: Int64?,
        destinationPath: String
    ) {
        self.id = "\(repoID)|\(target?.rawValue ?? "default")|\(launchAction.rawValue)"
        self.repoID = repoID
        self.displayName = displayName
        self.shortName = shortName
        self.target = target
        self.launchAction = launchAction
        self.totalBytes = totalBytes
        self.destinationPath = destinationPath
    }
}

public struct PendingModelTune: Identifiable, Equatable, Sendable {
    public var id: String
    public var repoID: String
    public var installedPath: String
    public var displayName: String
    public var shortName: String
    public var modelFamily: String
    public var target: LaunchTarget?
    public var launchAction: PendingModelDownloadLaunchAction
    public var candidates: [TuneCandidate]

    public init(
        repoID: String,
        installedPath: String,
        displayName: String,
        shortName: String,
        modelFamily: String,
        target: LaunchTarget?,
        launchAction: PendingModelDownloadLaunchAction,
        candidates: [TuneCandidate]
    ) {
        self.id = "\(repoID)|\(installedPath)|\(launchAction.rawValue)"
        self.repoID = repoID
        self.installedPath = installedPath
        self.displayName = displayName
        self.shortName = shortName
        self.modelFamily = modelFamily
        self.target = target
        self.launchAction = launchAction
        self.candidates = candidates
    }
}

public enum BenchmarkDaemonReadinessError: Error, Equatable, LocalizedError {
    case modelDownloadRequired(String)
    case startupFailed(String)
    case unreachable(URL)

    public var errorDescription: String? {
        switch self {
        case .modelDownloadRequired(let model):
            return "Download \(model) before running the benchmark."
        case .startupFailed(let reason):
            return "Couldn't start MTPLX for the benchmark: \(reason)"
        case .unreachable(let url):
            return "Can't reach MTPLX at \(url.absoluteString)."
        }
    }
}

public struct ClientHandoffNotice: Equatable, Sendable {
    public let target: LaunchTarget
    public let status: String
    public let detail: String
    public let isWarning: Bool

    public static func openCode(result: OpenCodeDesktopResult) -> ClientHandoffNotice {
        let status: String
        let detail: String
        let isWarning: Bool

        switch result.action {
        case .unavailable:
            status = "Needs OpenCode Desktop"
            detail = "MTPLX is running, but OpenCode Desktop was not found at /Applications/OpenCode.app."
            isWarning = true
        case .opened:
            status = "OpenCode opened"
            detail = result.detail
            isWarning = false
        case .relaunched:
            status = "OpenCode reloaded"
            detail = result.detail
            isWarning = false
        case .focused:
            status = "OpenCode focused"
            detail = result.detail
            isWarning = false
        }

        return ClientHandoffNotice(
            target: .openCode,
            status: status,
            detail: detail,
            isWarning: isWarning
        )
    }

    public static func pi(result: PiLaunchResult) -> ClientHandoffNotice? {
        if result.action == .launched && !result.launchedProcessIDs.isEmpty {
            return nil
        }
        let detail = result.action == .unavailable
            ? result.detail
            : "MTPLX opened Terminal, but no Pi agent process was detected. Install Pi, then pick Pi again."
        return ClientHandoffNotice(
            target: .pi,
            status: result.action == .unavailable ? "Pi handoff unavailable" : "Pi not detected",
            detail: detail,
            isWarning: true
        )
    }

    public static func hermes(result: HermesLaunchResult) -> ClientHandoffNotice? {
        guard result.action == .unavailable else { return nil }
        return ClientHandoffNotice(
            target: .hermes,
            status: "Hermes unavailable",
            detail: result.detail,
            isWarning: true
        )
    }
}

@MainActor
public final class MTPLXBackendStore: ObservableObject {
    @Published public private(set) var daemonState: DaemonState = .stopped
    @Published public private(set) var connectionState: MetricsConnectionState = .idle
    @Published public private(set) var startupPhase: DaemonStartupPhase = .idle
    @Published public private(set) var health: HealthPayload?
    @Published public private(set) var capabilities: AppCapabilities?
    @Published public private(set) var snapshot: DashboardSnapshot?
    @Published public private(set) var latest: MetricsLatest?
    @Published public private(set) var rolling: RollingMetrics?
    @Published public private(set) var inFlight: [InFlightRequest] = []
    @Published public private(set) var sessions: SessionsPayload?
    @Published public private(set) var sessionBank: SessionBank?
    @Published public private(set) var mem: MemSnapshot?
    @Published public private(set) var thermal: ThermalSnapshot?
    @Published public private(set) var settings: MutableSettings?
    @Published public private(set) var scheduler: DynamicObject?
    @Published public private(set) var prefillStatus: DynamicObject?
    @Published public private(set) var logs: [LogEntry] = []
    /// Current fan mode reported by the daemon. `nil`
    /// when no fan_mode endpoint has been called yet this session.
    @Published public private(set) var currentFanMode: String? = nil
    /// Last `/v1/mtplx/thermal/status` response — includes detection
    /// (whether thermalforge is installed) and the underlying fan
    /// summary. UI uses `detection.available` to decide whether to
    /// show the FanModeToggle at all.
    @Published public private(set) var thermalStatus: DynamicObject? = nil
    /// Recent prefill envelopes returned by `/v1/mtplx/prefill_history`.
    /// Surfaced in the Cache tab + LiveTab idle-prefill caption.
    @Published public private(set) var prefillHistory: PrefillHistoryPayload? = nil
    /// `/v1/models` list, populated lazily for the About sheet.
    @Published public private(set) var models: ModelsResponse? = nil
    /// Number of `.completed` SSE events we've actually observed since
    /// the current daemon started. The daemon's own
    /// `lifetime.requestsTotal` counts the model warm-up as a request
    /// (it does emit decode tokens), so we can't trust it to gate "is
    /// this a real user request?" — we count our own.
    @Published public private(set) var observedCompletionCount: Int = 0
    @Published public private(set) var observedUserMetricEventCount: Int = 0
    @Published public private(set) var piTerminalAgentRunning: Bool = false
    @Published public private(set) var piTerminalAgentProcessIDs: [Int] = []
    @Published public private(set) var piTerminalLaunchCommand: String?
    @Published public private(set) var piTerminalLaunchDetail: String?
    @Published public private(set) var clientHandoffNotice: ClientHandoffNotice?
    /// One-line banner shown after the configured port was occupied and the
    /// daemon moved to the next free port (persisted to settings).
    @Published public private(set) var portFallbackNotice: String?
    @Published public private(set) var pendingModelDownload: PendingModelDownload?
    @Published public private(set) var modelDownloadProgress: DownloadProgressSnapshot?
    @Published public private(set) var modelDownloadFailure: String?
    @Published public private(set) var isModelDownloading: Bool = false
    @Published public private(set) var pendingModelTune: PendingModelTune?
    @Published public private(set) var modelTuneCandidatesLanded: [TuneCandidate: TuneCandidateResult] = [:]
    @Published public private(set) var modelTuneResult: TuneResult?
    @Published public private(set) var modelTuneFailure: String?
    @Published public private(set) var modelTuneStatusMessage: String?
    @Published public private(set) var isModelTuning: Bool = false
    @Published public private(set) var runtimeUpdateSnapshot: MTPLXRuntimeUpdateSnapshot?
    @Published public private(set) var runtimeUpdateFailure: String?
    /// Lifecycle-aware headline decode reading driven by progress,
    /// completion, and snapshot events. The gauge reads this instead
    /// of probing `latest.decode_tok_s` directly, which means the
    /// displayed value: (a) is driven by the live request's current
    /// `decode_tok_s`, (b) holds at the last request's final average
    /// after completion instead of falling to zero, and (c) is reset
    /// cleanly on daemon stop/start so the next session begins from
    /// `.absent` rather than carrying the previous run's number.
    @Published public private(set) var headlineDecode: HeadlineDecodeReading = .absent
    /// Spring-friendly mirrors of the noisiest hot-path metrics.
    /// Acceptance bars, depth count, verify count, and cached tokens
    /// flicker on raw SSE data because each progress frame slightly
    /// shifts the numerator/denominator. A short EMA over each value
    /// removes the flicker without lying about steady-state magnitudes
    /// — the headline value still settles to the real number, it just
    /// gets there without strobing between two integers per frame.
    @Published public private(set) var smoothedMetrics: SmoothedMetrics = SmoothedMetrics()
    /// Monotonic per-request token. Bumped whenever a brand-new request
    /// identity is first observed, so views that keep their own
    /// per-request UI state (most notably the acceptance bars' last-good
    /// rows) can clear cleanly instead of bleeding the previous request's
    /// values into the next request's prefill window — the "acceptance
    /// shows prematurely" symptom.
    @Published public private(set) var metricsRequestGeneration: Int = 0
    /// Identity (request_id ?? session_id) of the request the live
    /// metrics currently describe. Drives the reset above.
    private var trackedRequestKey: String? = nil
    /// Once a request completes, its live metrics are frozen at the final
    /// values until a new request arrives. Idle snapshot polls keep
    /// arriving carrying the just-finished request's counters; without
    /// this freeze the EMA would keep converging after generation ended,
    /// which reads as acceptance "creeping up" once the answer is done.
    private var smoothedFrozen: Bool = false
    private let smoothedAlpha: Double = 0.25
    /// One coherent (generated-tokens, decode-elapsed) sample for the
    /// current request, used to compute the headline decode rate the
    /// Open WebUI way (tokens ÷ time, a single converging average). Held
    /// in the store — not recomputed per frame — because snapshot frames
    /// routinely omit `decode_elapsed_s` while progress frames carry it,
    /// so a per-frame recompute flipped the gauge between the cumulative
    /// value and the raw `decode_tok_s` fallback (the two-value bounce).
    /// Both fields are monotonic within a request; reset per request.

    public var hasObservedCurrentRunMetrics: Bool {
        observedUserMetricEventCount > 0 || observedCompletionCount > 0
    }

    public var settingsURL: URL {
        settingsStore.settingsURL
    }

    public private(set) var configuration: MTPLXAppConfiguration

    /// Host-supplied hook invoked on the main actor immediately after a
    /// daemon launch reaches `running` for a specific target. The host
    /// (`MTPLXApp`) wires this to `router.showChat()` when the target is
    /// `.chat`, so the app flips into the in-app chat surface as soon as
    /// the daemon is ready. Kept as a single callback rather than a
    /// hard dependency on `AppRouter` so MTPLXAppCore does not need to
    /// import MTPLXAppHost. Target is optional because callers like
    /// `applyConfiguration` derive it from persisted settings and may
    /// pass `nil` if the persisted value doesn't parse.
    public var onDaemonReady: (@MainActor (LaunchTarget?) -> Void)?

    private let settingsStore: MTPLXSettingsStore
    private let commandBuilder: MTPLXCommandBuilder
    private let supervisor: DaemonSupervisor
    private let openCodeIntegration: OpenCodeIntegration
    private let piIntegration: PiIntegration
    private let hermesIntegration: HermesIntegration
    private let modelDownloader: ModelDownloader
    private let autoTuner: AutoTuner
    private let runtimeUpdateService: MTPLXRuntimeUpdateService
    private let localFanRestorer: @Sendable () async -> Bool
    private var launchedPiAgentPIDs: Set<Int> = []
    private var streamTask: Task<Void, Never>?
    private var healthWatchTask: Task<Void, Never>?
    private var modelDownloadTask: Task<Void, Never>?
    private var modelTuneTask: Task<Void, Never>?
    private var lateHealthRecoveryTask: Task<Void, Never>?
    /// The slow part of stopping the daemon (fan restore + graceful
    /// SIGTERM/reap of the serve child) runs here, detached from the
    /// instant UI flip, so the Stop/Play control isn't frozen for the ~5s
    /// the process takes to exit. Start paths and app-termination await
    /// this so a new launch never races a half-finished teardown
    /// (`.alreadyRunning`) and quitting never orphans the daemon.
    private var daemonTeardownTask: Task<Void, Never>?
    private var pendingLiveSettings: MutableSettings?
    private var pendingLiveSettingsModel: String?
    private var liveSettingsModel: String?
    private var lastProgressPublishS: TimeInterval = 0
    private var fanRestoreRequiredOnStop: Bool = false
    private var activeLaunchID: String?
    private var cancelledLaunchIDs: Set<String> = []
    private let daemonStartupTimeoutSeconds: TimeInterval = 600

    public init(
        configuration: MTPLXAppConfiguration = MTPLXAppConfiguration(),
        settingsStore: MTPLXSettingsStore = MTPLXSettingsStore(),
        commandBuilder: MTPLXCommandBuilder = MTPLXCommandBuilder(),
        supervisor: DaemonSupervisor = DaemonSupervisor(),
        openCodeIntegration: OpenCodeIntegration = OpenCodeIntegration(),
        piIntegration: PiIntegration = PiIntegration(),
        hermesIntegration: HermesIntegration = HermesIntegration(),
        modelDownloader: ModelDownloader = ModelDownloader(),
        autoTuner: AutoTuner = AutoTuner(),
        runtimeUpdateService: MTPLXRuntimeUpdateService? = nil,
        localFanRestorer: (@Sendable () async -> Bool)? = nil
    ) {
        self.configuration = configuration
        self.settingsStore = settingsStore
        self.commandBuilder = commandBuilder
        self.supervisor = supervisor
        self.openCodeIntegration = openCodeIntegration
        self.piIntegration = piIntegration
        self.hermesIntegration = hermesIntegration
        self.modelDownloader = modelDownloader
        self.autoTuner = autoTuner
        self.runtimeUpdateService = runtimeUpdateService
            ?? MTPLXRuntimeUpdateService(environment: commandBuilder.environment)
        self.localFanRestorer = localFanRestorer ?? {
            await MTPLXBackendStore.restoreFanModeWithLocalThermalforge()
        }
    }

    public func loadPersistedSettings() {
        if var loaded = try? settingsStore.load() {
            if shouldPromoteStaleOpenCodeTarget(loaded) {
                loaded.lastLaunchTarget = LaunchTarget.openCode.rawValue
                try? settingsStore.save(loaded)
            }
            configuration = loaded
            seedLiveSettingsFromConfiguration(loaded)
        }
    }

    public func saveSettings(_ next: MTPLXAppConfiguration) throws {
        configuration = next
        seedLiveSettingsFromConfiguration(next)
        try settingsStore.save(next)
    }

    public func applyConfiguration(
        _ next: MTPLXAppConfiguration,
        restartIfRunning: Bool = true
    ) async throws {
        let shouldRestart = restartIfRunning && supervisor.isRunning()
        let target = LaunchTarget(rawValue: next.lastLaunchTarget)
        configuration = next
        try settingsStore.save(next)
        guard shouldRestart else { return }
        if promptForModelDownloadIfNeeded(
            configuration: next,
            target: target,
            launchAction: .restart
        ) {
            return
        }

        // Don't let a backgrounded Stop teardown race the restart's own
        // stop()->start() and surface as `.alreadyRunning`.
        await awaitDaemonTeardown()
        let launchID = UUID().uuidString
        let recoveryTarget = target
        do {
            try await prepareRuntimeForDaemonStart()
            if target == .openCode {
                let result = try openCodeIntegration.sync(configuration: next)
                await supervisor.logs.append(
                    "synced OpenCode config \(result.configPath) -> \(result.baseURL) \(result.modelReference)",
                    stream: .system
                )
            }
            if target == .pi {
                let result = try piIntegration.sync(configuration: next)
                await supervisor.logs.append(
                    "synced Pi config \(result.configPath) -> \(result.baseURL) \(result.modelReference)",
                    stream: .system
                )
            }
            if target == .hermes {
                let result = try hermesIntegration.sync(configuration: next)
                await supervisor.logs.append(
                    "synced Hermes profile \(result.profileName) \(result.configPath) -> \(result.baseURL) \(result.modelReference)",
                    stream: .system
                )
            }
            streamTask?.cancel()
            streamTask = nil
            lateHealthRecoveryTask?.cancel()
            lateHealthRecoveryTask = nil
            healthWatchTask?.cancel()
            healthWatchTask = nil
            connectionState = .connecting
            daemonState = .stopping
            let command = try commandBuilder.buildServeCommand(
                configuration: next,
                target: target,
                launchID: launchID
            )
            startupPhase = .launching
            let startupHealth = try await supervisor.restart(
                command: command,
                healthBaseURL: baseURL,
                apiKey: next.apiKey,
                probeHealth: true,
                timeoutSeconds: daemonStartupTimeoutSeconds,
                expectedLaunchID: launchID,
                requireActualFanRamp: requiresStartupFanRamp(next),
                onPhase: { phase in
                    Task { @MainActor [weak self] in
                        self?.startupPhase = phase
                    }
                }
            )
            if let startupHealth {
                health = startupHealth
                currentFanMode = verifiedFanMode(from: startupHealth)
                fanRestoreRequiredOnStop = fanRestoreRequiredOnStop
                    || modeRequiresFanRestore(currentFanMode)
            }
            await finishReadyDaemon(
                target: target,
                configuration: next,
                replaceExistingClient: true
            )
        } catch {
            let failureDescription = Self.humanizedStartFailure(
                error,
                port: configuration.port
            )
            daemonState = .degraded(failureDescription)
            startupPhase = .failed(failureDescription)
            await refreshLogs()
            scheduleLateHealthRecovery(launchID: launchID, target: recoveryTarget)
            throw error
        }
    }

    public func startDaemon() async {
        await startDaemon(target: defaultLaunchTarget(for: configuration))
    }

    public func attachExistingDaemonIfOwned() async {
        await awaitDaemonTeardown()
        guard !supervisor.isRunning() else { return }
        do {
            let target = defaultLaunchTarget(for: configuration)
            let command = (try? commandBuilder.buildServeCommand(
                configuration: configuration,
                target: target,
                launchID: UUID().uuidString
            )) ?? DaemonCommand(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: ["--model", configuration.model]
            )
            guard let adoptedHealth = try await supervisor.adoptExistingIfAppOwned(
                command: command,
                healthBaseURL: baseURL,
                apiKey: configuration.apiKey,
                requireActualFanRamp: requiresStartupFanRamp(configuration)
            ) else {
                return
            }
            clearLiveMetricsState(target: target)
            health = adoptedHealth
            currentFanMode = verifiedFanMode(from: adoptedHealth)
            fanRestoreRequiredOnStop = fanRestoreRequiredOnStop
                || modeRequiresFanRestore(currentFanMode)
            daemonState = .running
            startupPhase = .ready
            try await refreshStaticState()
            try await flushPendingLiveSettingsIfNeeded(target: target)
            startMetricsStream()
            await refreshThermalStatus()
        } catch {
            await refreshLogs()
        }
    }

    /// Start the daemon with the per-`mtplx-start-<target>` preset
    /// merged onto the user's Settings. Picking a target persists it as
    /// `lastLaunchTarget` so a subsequent click can skip the picker.
    public func startDaemon(target: LaunchTarget?) async {
        clientHandoffNotice = nil
        portFallbackNotice = nil
        await startDaemon(target: target, attemptedPortRemediation: false)
    }

    /// `attemptedPortRemediation` bounds the relaunch to one retry: a
    /// port-conflict failure remediates (replace our own stale daemon,
    /// or move to a free port) and relaunches once; a second failure
    /// surfaces through the normal degraded path.
    private func startDaemon(
        target: LaunchTarget?,
        attemptedPortRemediation: Bool
    ) async {
        let target = target ?? defaultLaunchTarget(for: configuration)
        if let target {
            var next = configuration
            next.lastLaunchTarget = target.rawValue
            configuration = next
            try? settingsStore.save(next)
        }
        if promptForModelDownloadIfNeeded(
            configuration: configuration,
            target: target,
            launchAction: .start
        ) {
            return
        }
        // Wipe stale snapshot/lifetime/rolling/sticky-max state from
        // the previous daemon session BEFORE handing off to the new
        // one. Without this the gauge shows the previous run's last
        // decode reading the moment the user clicks Play.
        clearLiveMetricsState(target: target)
        lateHealthRecoveryTask?.cancel()
        lateHealthRecoveryTask = nil
        // A just-issued Stop may still be reaping the previous serve
        // process in the background; wait for it so supervisor.start()
        // doesn't trip `.alreadyRunning`.
        await awaitDaemonTeardown()
        healthWatchTask?.cancel()
        healthWatchTask = nil
        let launchID = UUID().uuidString
        do {
            try await prepareRuntimeForDaemonStart()
            // Pre-flight the configured port before any integration writes
            // its config: adoptable app-owned daemons are left for the
            // supervisor, stale app-owned daemons are replaced in place,
            // and everything else moves us to the next free port so the
            // user never sees a raw "port occupied" failure.
            await preflightConfiguredPort(target: target, launchID: launchID)
            if target == .openCode {
                let result = try openCodeIntegration.sync(configuration: configuration)
                await supervisor.logs.append(
                    "synced OpenCode config \(result.configPath) -> \(result.baseURL) \(result.modelReference)",
                    stream: .system
                )
            }
            if target == .pi {
                let result = try piIntegration.sync(configuration: configuration)
                await supervisor.logs.append(
                    "synced Pi config \(result.configPath) -> \(result.baseURL) \(result.modelReference)",
                    stream: .system
                )
            }
            if target == .hermes {
                let result = try hermesIntegration.sync(configuration: configuration)
                await supervisor.logs.append(
                    "synced Hermes profile \(result.profileName) \(result.configPath) -> \(result.baseURL) \(result.modelReference)",
                    stream: .system
                )
            }
            let command = try commandBuilder.buildServeCommand(
                configuration: configuration,
                target: target,
                launchID: launchID
            )
            daemonState = .starting
            startupPhase = .launching
            activeLaunchID = launchID
            let startupHealth = try await supervisor.start(
                command: command,
                healthBaseURL: baseURL,
                apiKey: configuration.apiKey,
                probeHealth: true,
                timeoutSeconds: daemonStartupTimeoutSeconds,
                expectedLaunchID: launchID,
                requireActualFanRamp: requiresStartupFanRamp(configuration),
                adoptExistingAppOwnedDaemon: true,
                onPhase: { phase in
                    Task { @MainActor [weak self] in
                        self?.startupPhase = phase
                    }
                }
            )
            if let startupHealth {
                health = startupHealth
                currentFanMode = verifiedFanMode(from: startupHealth)
                fanRestoreRequiredOnStop = fanRestoreRequiredOnStop
                    || modeRequiresFanRestore(currentFanMode)
            }
            await finishReadyDaemon(
                target: target,
                configuration: configuration,
                replaceExistingClient: true
            )
            if activeLaunchID == launchID {
                activeLaunchID = nil
            }
        } catch {
            if !attemptedPortRemediation,
               !cancelledLaunchIDs.contains(launchID),
               Self.failureIndicatesPortConflict(error),
               await remediatePortConflict(target: target, launchID: launchID) {
                if activeLaunchID == launchID {
                    activeLaunchID = nil
                }
                await supervisor.logs.append(
                    "retrying daemon launch after port remediation",
                    stream: .system
                )
                await startDaemon(target: target, attemptedPortRemediation: true)
                return
            }
            let failureDescription = Self.humanizedStartFailure(
                error,
                port: configuration.port
            )
            let failedPhase = startupPhase
            let startupWasCancelled = cancelledLaunchIDs.remove(launchID) != nil
            if activeLaunchID == launchID {
                activeLaunchID = nil
            }

            if startupWasCancelled {
                if requiresStartupFanRamp(configuration) {
                    let restored = await restoreFansLocally(
                        successLog: "fan profile restored after canceled startup"
                    )
                    if !restored {
                        await supervisor.logs.append(
                            "fan restore fallback failed after canceled startup",
                            stream: .system
                        )
                    }
                }
                daemonState = .stopped
                startupPhase = .idle
                connectionState = .idle
                await refreshLogs()
                return
            }

            if shouldRestoreFansAfterFailedStartup(phase: failedPhase) {
                let restored = await restoreFansLocally(
                    successLog: "fan profile restored after failed startup"
                )
                if !restored {
                    await supervisor.logs.append(
                        "fan restore fallback failed after failed startup",
                        stream: .system
                    )
                }
            }
            daemonState = .degraded(failureDescription)
            startupPhase = .failed(failureDescription)
            await refreshLogs()
            scheduleLateHealthRecovery(launchID: launchID, target: target)
        }
    }

    /// Resolve who owns the configured port before launching.
    ///
    /// - Adoptable app-owned daemon: do nothing; `supervisor.start`
    ///   adopts it on the same port.
    /// - Stale app-owned daemon (ours, but a different model/config):
    ///   replace it in place. Bumping ports here would strand a model in
    ///   memory next to the new one — the exact double-load this app
    ///   promises to prevent.
    /// - CLI-started MTPLX server or a foreign app: never touch someone
    ///   else's process; move to the next free port, persist it, and
    ///   surface a one-line banner.
    func preflightConfiguredPort(
        target: LaunchTarget?,
        launchID: String
    ) async {
        let occupant = await PortPreflight.classify(
            baseURL: baseURL,
            apiKey: configuration.apiKey
        )
        let occupantDescription: String
        switch occupant {
        case .free:
            return
        case .mtplxServer(let existing):
            let probeCommand = try? commandBuilder.buildServeCommand(
                configuration: configuration,
                target: target,
                launchID: launchID
            )
            if let probeCommand,
               supervisor.canAdoptExisting(
                   existing,
                   for: probeCommand,
                   requireActualFanRamp: requiresStartupFanRamp(configuration)
               ) {
                return
            }
            if existing.startup?.launchId?.isEmpty == false,
               let stalePID = existing.startup?.pid {
                await supervisor.logs.append(
                    "replacing stale app-owned daemon pid \(stalePID) on port \(configuration.port)",
                    stream: .system
                )
                await supervisor.terminateExternalDaemon(rootPID: pid_t(stalePID))
                return
            }
            occupantDescription = "an MTPLX server started outside the app"
        case .foreign:
            occupantDescription = "another app"
        }
        let occupiedPort = configuration.port
        guard let freePort = PortPreflight.nextFreePort(after: occupiedPort) else {
            // No port available; let supervisor.start surface the failure.
            return
        }
        var next = configuration
        next.port = freePort
        configuration = next
        try? settingsStore.save(next)
        portFallbackNotice =
            "Port \(occupiedPort) was in use by \(occupantDescription). MTPLX now uses port \(freePort)."
        await supervisor.logs.append(
            "port preflight: \(occupiedPort) occupied by \(occupantDescription); switched to \(freePort)",
            stream: .system
        )
    }

    /// Test seam: run the port pre-flight and report the resulting port and
    /// fallback notice as Sendable values.
    func preflightOutcomeForTest(
        target: LaunchTarget?,
        launchID: String
    ) async -> (port: Int, notice: String?) {
        await preflightConfiguredPort(target: target, launchID: launchID)
        return (configuration.port, portFallbackNotice)
    }

    /// One-shot remediation between launch attempts after the bind
    /// itself failed. The standard preflight handles occupants it can
    /// see (replace our own stale daemon in place, sidestep foreign
    /// listeners). When the probe sees nothing yet the bind still
    /// failed, trust the bind error over the probe: IPv6-only and
    /// other-user wildcard listeners are invisible to a localhost HTTP
    /// probe but still collide, so move to the next free port outright.
    func remediatePortConflict(
        target: LaunchTarget?,
        launchID: String
    ) async -> Bool {
        let occupiedPort = configuration.port
        let occupant = await PortPreflight.classify(
            baseURL: baseURL,
            apiKey: configuration.apiKey
        )
        switch occupant {
        case .mtplxServer, .foreign:
            await preflightConfiguredPort(target: target, launchID: launchID)
            return true
        case .free:
            guard let freePort = PortPreflight.nextFreePort(after: occupiedPort) else {
                return false
            }
            var next = configuration
            next.port = freePort
            configuration = next
            try? settingsStore.save(next)
            portFallbackNotice =
                "Port \(occupiedPort) was busy. MTPLX now uses port \(freePort)."
            await supervisor.logs.append(
                "launch hit a port conflict on \(occupiedPort) the probe could not see; switched to \(freePort)",
                stream: .system
            )
            return true
        }
    }

    /// True when a launch failure means the port itself was lost:
    /// the supervisor's pre-spawn probe found a foreign MTPLX daemon,
    /// or the serve process exited with its own port-busy error.
    nonisolated static func failureIndicatesPortConflict(_ error: Error) -> Bool {
        if case DaemonSupervisorError.portOccupied = error {
            return true
        }
        if case DaemonSupervisorError.launchFailed(let detail) = error {
            let lowered = detail.lowercased()
            return lowered.contains("already in use") || lowered.contains("errno 48")
        }
        return false
    }

    /// Occupant-aware copy for startup failures. Port collisions get a
    /// sentence a user can act on instead of the raw error description.
    nonisolated static func humanizedStartFailure(_ error: Error, port: Int) -> String {
        if case DaemonSupervisorError.portOccupied(_, let launchID) = error {
            if launchID != nil {
                return "Port \(port) is running another MTPLX server. "
                    + "Stop it where it was started, or run `mtplx stop --port \(port)` "
                    + "in Terminal, then press Play."
            }
            return "Port \(port) is held by an MTPLX server started outside the app. "
                + "Press Ctrl-C in its terminal or run `mtplx stop --port \(port)`, "
                + "then press Play."
        }
        return String(describing: error)
    }

    public func refreshRuntimeUpdateStatus() async {
        runtimeUpdateSnapshot = await runtimeUpdateService.refreshSnapshot()
        runtimeUpdateFailure = nil
    }

    public func updateRuntimeWithHomebrew() async {
        do {
            let bootstrapper = MTPLXRuntimeBootstrapper(environment: commandBuilder.environment)
            let executable: URL
            switch runtimeUpdateSnapshot?.action {
            case .updateBundledRequired:
                // App-owned runtimes refresh from the bundled wheel, not brew.
                executable = try bootstrapper.installOrUpdate()
            default:
                executable = try bootstrapper.upgradeHomebrewRuntime()
            }
            runtimeUpdateSnapshot = MTPLXRuntimeUpdateService.snapshot(
                manifest: try? await runtimeUpdateService.fetchManifest(),
                environment: commandBuilder.environment.merging(["PATH": executable.deletingLastPathComponent().path]) { current, _ in current }
            )
            runtimeUpdateFailure = nil
        } catch {
            runtimeUpdateFailure = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
        }
    }

    private func prepareRuntimeForDaemonStart() async throws {
        startupPhase = .launching
        do {
            _ = try await runtimeUpdateService.prepareRuntimeForLaunch()
            runtimeUpdateSnapshot = await runtimeUpdateService.refreshSnapshot()
            runtimeUpdateFailure = nil
        } catch {
            runtimeUpdateSnapshot = await runtimeUpdateService.refreshSnapshot()
            runtimeUpdateFailure = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            throw error
        }
    }

    /// Old browser chat surface for the Open WebUI launch target.
    public var webChatURL: URL {
        authenticatedBrowserURL(nextPath: "/", fallback: baseURL)
    }

    /// Browser dashboard URL used by explicit dashboard actions.
    public var browserDashboardURL: URL {
        authenticatedBrowserURL(
            nextPath: "/dashboard/",
            fallback: baseURL.appendingPathComponent("dashboard")
        )
    }

    private func authenticatedBrowserURL(nextPath: String, fallback: URL) -> URL {
        guard let apiKey = configuration.apiKey, !apiKey.isEmpty else {
            return fallback
        }
        let authURL = baseURL
            .appendingPathComponent("mtplx")
            .appendingPathComponent("browser-auth")
        guard var components = URLComponents(url: authURL, resolvingAgainstBaseURL: false) else {
            return fallback
        }
        components.queryItems = [
            URLQueryItem(name: "mtplx_api_key", value: apiKey),
            URLQueryItem(name: "next", value: nextPath)
        ]
        return components.url ?? fallback
    }

    /// Open the old browser chat surface in the user's default browser.
    public func openWebChat() {
        #if canImport(AppKit)
        AppKit.NSWorkspace.shared.open(webChatURL)
        #endif
    }

    /// Open the React live dashboard in the user's default browser.
    /// Exposed for users who explicitly ask for the browser dashboard
    /// (menu bar / About sheet) — no longer wired to a LaunchTarget.
    public func openBrowserDashboard() {
        #if canImport(AppKit)
        AppKit.NSWorkspace.shared.open(browserDashboardURL)
        #endif
    }

    public func stopDaemon() async {
        // Stop is a real product boundary: when the top-right button says
        // Stop, no app-owned server should keep listening in the background.
        // Show Stopping while the process family is being reaped, then only
        // return to Stopped after the wrapper, model server child, and thermal
        // sidecar have been signalled and verified gone.
        if let launchID = activeLaunchID {
            cancelledLaunchIDs.insert(launchID)
            activeLaunchID = nil
        }
        let shouldWarnIfFanRestoreFails = shouldRestoreFanModeOnStop()
        let startupPID = health?.startup?.pid.map(pid_t.init)
        let piPIDsToStop = Array(launchedPiAgentPIDs)
        modelDownloadTask?.cancel()
        modelDownloadTask = nil
        isModelDownloading = false
        lateHealthRecoveryTask?.cancel()
        lateHealthRecoveryTask = nil
        healthWatchTask?.cancel()
        healthWatchTask = nil
        streamTask?.cancel()
        streamTask = nil
        launchedPiAgentPIDs.removeAll()
        piTerminalAgentRunning = false
        piTerminalAgentProcessIDs = []
        piTerminalLaunchCommand = nil
        piTerminalLaunchDetail = nil
        clientHandoffNotice = nil
        daemonState = .stopping
        startupPhase = .idle
        connectionState = .idle
        clearLiveMetricsState()

        // Serialize against any in-flight teardown so Stop and Restart never
        // race each other, then await the new teardown. This keeps the chrome
        // disabled until the server is really gone.
        let previousTeardown = daemonTeardownTask
        daemonTeardownTask = Task { @MainActor [self] in
            var fanRestoreSucceeded = await restoreFansLocally(
                successLog: "fan profile restored locally on stop"
            )
            if !fanRestoreSucceeded, shouldWarnIfFanRestoreFails {
                await supervisor.logs.append(
                    "local fan restore failed before daemon stop; will retry after process teardown",
                    stream: .system
                )
            }
            await previousTeardown?.value
            let stoppedHermes = hermesIntegration.stopLaunchedTerminalAgents()
            if stoppedHermes > 0 {
                await supervisor.logs.append(
                    "stopped \(stoppedHermes) Hermes Terminal handoff(s)",
                    stream: .system
                )
            }
            let stoppedPi = piIntegration.stopLaunchedAgents(processIDs: piPIDsToStop)
            if stoppedPi > 0 {
                await supervisor.logs.append(
                    "stopped \(stoppedPi) Pi Terminal handoff(s)",
                    stream: .system
                )
            }
            let additionalPIDs = startupPID.map { [$0] } ?? []
            await supervisor.stop(graceSeconds: 1.5, additionalProcessIDs: additionalPIDs)
            let localFanRestoreSucceeded = await restoreFansLocally(
                successLog: "fan profile restored with local ThermalForge after daemon stop"
            )
            if localFanRestoreSucceeded {
                fanRestoreSucceeded = true
            } else if !fanRestoreSucceeded, shouldWarnIfFanRestoreFails {
                await supervisor.logs.append(
                    "fan restore fallback failed; check ThermalForge status",
                    stream: .system
                )
            }
            daemonState = .stopped
            startupPhase = .idle
            connectionState = .idle
            if !fanRestoreSucceeded, shouldWarnIfFanRestoreFails {
                currentFanMode = nil
            }
            await refreshLogs()
        }
        await daemonTeardownTask?.value
    }

    /// Wait for the detached daemon teardown (fan restore + process reap)
    /// to finish. Start paths await this before launching so a new daemon
    /// never collides with the previous one's still-terminating process,
    /// and app-termination awaits it so quitting never orphans the daemon.
    public func awaitDaemonTeardown() async {
        await daemonTeardownTask?.value
    }

    @discardableResult
    public func ensureDaemonReadyForBenchmark() async throws -> HealthPayload {
        if let existing = try? await apiClient.health(), existing.ok {
            health = existing
            currentFanMode = verifiedFanMode(from: existing)
            fanRestoreRequiredOnStop = fanRestoreRequiredOnStop
                || modeRequiresFanRestore(currentFanMode)
            try await flushPendingLiveSettingsIfNeeded(target: .benchmark)
            return existing
        }

        await startDaemon(target: .benchmark)

        if let pendingModelDownload {
            throw BenchmarkDaemonReadinessError.modelDownloadRequired(
                pendingModelDownload.shortName
            )
        }

        if case .degraded(let reason) = daemonState {
            throw BenchmarkDaemonReadinessError.startupFailed(reason)
        }

        if let ready = try? await apiClient.health(), ready.ok {
            health = ready
            currentFanMode = verifiedFanMode(from: ready)
            fanRestoreRequiredOnStop = fanRestoreRequiredOnStop
                || modeRequiresFanRestore(currentFanMode)
            try await flushPendingLiveSettingsIfNeeded(target: .benchmark)
            return ready
        }

        throw BenchmarkDaemonReadinessError.unreachable(baseURL)
    }

    public func dismissModelDownloadPrompt() {
        guard !isModelDownloading, !isModelTuning else { return }
        pendingModelDownload = nil
        modelDownloadProgress = nil
        modelDownloadFailure = nil
        clearModelTuneState()
    }

    /// Present the shared model-download sheet for an arbitrary Hugging
    /// Face repo. Used by Forge Discover's Install so it goes through the
    /// exact same proven flow as the top-left picker download: a progress
    /// sheet with %/ETA/cancel and friendly auth/network errors, and on
    /// completion the model is remembered as a custom model (so it shows
    /// up in the top-left picker) and the daemon starts/restarts against
    /// it. If the weights are already on disk, it just registers + selects
    /// the model instead of re-pulling.
    public func presentModelDownload(
        repoID: String,
        displayName: String? = nil,
        shortName: String? = nil,
        totalBytes: Int64? = nil
    ) {
        let trimmed = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isModelDownloading, !isModelTuning else { return }
        let option = MTPLXModelOption.option(matching: trimmed)
            ?? MTPLXModelOption.customHuggingFaceModel(repoID: trimmed)
        let target = defaultLaunchTarget(for: configuration)
        let launchAction: PendingModelDownloadLaunchAction = supervisor.isRunning() ? .restart : .start
        if let installedPath = option?.installedLocalPath {
            Task { @MainActor [weak self] in
                do {
                    try await self?.finishModelInstall(
                        repoID: trimmed,
                        installedPath: installedPath,
                        target: target,
                        launchAction: launchAction
                    )
                } catch {
                    self?.modelDownloadFailure = self?.friendlyDownloadFailure(String(describing: error))
                }
            }
            return
        }
        let resolvedBytes: Int64? = {
            if let totalBytes, totalBytes > 0 { return totalBytes }
            if let option, option.sizeBytes > 0 { return option.sizeBytes }
            return nil
        }()
        pendingModelDownload = PendingModelDownload(
            repoID: trimmed,
            displayName: displayName ?? option?.displayName ?? trimmed,
            shortName: shortName ?? option?.shortName ?? trimmed,
            target: target,
            launchAction: launchAction,
            totalBytes: resolvedBytes,
            destinationPath: modelDownloader.cachedModelPath(for: trimmed).path
        )
        modelDownloadProgress = nil
        modelDownloadFailure = nil
        clearModelTuneState()
    }

    public func cancelModelDownload() {
        modelDownloadTask?.cancel()
        modelDownloadTask = nil
        isModelDownloading = false
        if var snapshot = modelDownloadProgress, !snapshot.isComplete {
            snapshot.bytesPerSecond = 0
            snapshot.etaSeconds = nil
            snapshot.stalledSeconds = 0
            snapshot.statusMessage = "Paused"
            modelDownloadProgress = snapshot
        }
    }

    public func cancelPendingModelTune() {
        modelTuneTask?.cancel()
        modelTuneTask = nil
        isModelTuning = false
        modelTuneStatusMessage = nil
    }

    public func downloadPendingModelAndStart() {
        guard let request = pendingModelDownload, !isModelDownloading, !isModelTuning else { return }
        modelDownloadTask?.cancel()
        modelDownloadFailure = nil
        modelTuneFailure = nil
        if let progress = modelDownloadProgress,
           progress.isComplete,
           MTPLXModelOption.hasCompleteInstall(at: progress.destinationPath)
        {
            Task { @MainActor [weak self] in
                do {
                    guard let self else { return }
                    if self.beginPostDownloadTuneIfSupported(
                        request: request,
                        installedPath: progress.destinationPath
                    ) {
                        return
                    }
                    try await self.finishModelInstall(
                        repoID: request.repoID,
                        installedPath: progress.destinationPath,
                        target: request.target,
                        launchAction: request.launchAction
                    )
                    self.clearModelDownloadAndTuneState()
                } catch {
                    self?.modelDownloadFailure = self?.friendlyDownloadFailure(String(describing: error))
                }
            }
            return
        }
        modelDownloadProgress = nil
        isModelDownloading = true
        let downloader = modelDownloader
        let extraEnvironment =
            MTPLXAppConfiguration.hfMirrorEnvironment(configuration.hfEndpoint) ?? [:]
        modelDownloadTask = Task.detached(priority: .userInitiated) { [weak self, downloader, request, extraEnvironment] in
            for await event in downloader.stream(
                repo: request.repoID,
                totalBytes: request.totalBytes,
                extraEnvironment: extraEnvironment
            ) {
                if Task.isCancelled { break }
                await self?.handleModelDownloadEvent(event, request: request)
            }
        }
    }

    /// Zero every `@Published` metric that came from the daemon's
    /// metrics stream / snapshot endpoint. Called on stop AND at the
    /// start of `startDaemon(target:)`. Live generation settings belong
    /// to MTPLX, so every launch target carries the compatible sampler
    /// and reasoning policy selected in the app.
    public func clearLiveMetricsState(target: LaunchTarget? = nil) {
        let carriedSettings = Self.targetCarriesSettingsSampler(target)
            ? carryableLiveSettingsForCurrentModel(target: target)
            : nil
        health = nil
        capabilities = nil
        snapshot = nil
        latest = nil
        rolling = nil
        inFlight = []
        sessions = nil
        sessionBank = nil
        mem = nil
        thermal = nil
        settings = carriedSettings
        pendingLiveSettings = carriedSettings
        pendingLiveSettingsModel = carriedSettings == nil ? nil : configuration.model
        liveSettingsModel = carriedSettings == nil ? nil : configuration.model
        scheduler = nil
        prefillStatus = nil
        prefillHistory = nil
        thermalStatus = nil
        models = nil
        observedCompletionCount = 0
        observedUserMetricEventCount = 0
        lastProgressPublishS = 0
        headlineDecode = .absent
        smoothedMetrics = SmoothedMetrics()
        trackedRequestKey = nil
        smoothedFrozen = false
        metricsRequestGeneration &+= 1
    }

    public func refreshStaticState() async throws {
        let client = apiClient
        do {
            async let health = client.health()
            async let capabilities = client.capabilities()
            async let sessions = client.sessions()
            self.health = try await health
            self.capabilities = try await capabilities
            self.sessions = try await sessions
            await refreshPrefillHistory()
            await refreshModels()
            await refreshLogs()
        } catch {
            markDaemonUnreachableIfNeeded(
                reason: "MTPLX lost contact with the model server. Start it again."
            )
            throw error
        }
    }

    public func refreshSnapshot() async throws {
        do {
            apply(snapshot: try await apiClient.snapshot())
        } catch {
            markDaemonUnreachableIfNeeded(
                reason: "MTPLX lost contact with live metrics. Start it again."
            )
            throw error
        }
    }

    public func updateLiveSettings(_ next: MutableSettings) async throws {
        let merged = mergedLiveSettingsPatch(next)
        let livePatch = Self.liveMutableSettingsPatch(from: next)
        // Only the caller's own patch counts as a depth choice; the
        // merged snapshot always carries the daemon's current depth.
        let depthIsExplicitSelection = next.depth != nil
        let daemonIsRunning = daemonState == .running || supervisor.isRunning()
        if !daemonIsRunning {
            settings = merged
            liveSettingsModel = configuration.model
            pendingLiveSettings = merged
            pendingLiveSettingsModel = configuration.model
            try? persistLiveSettings(merged, depthIsExplicitSelection: depthIsExplicitSelection)
        }
        do {
            settings = try await apiClient.updateSettings(livePatch)
            liveSettingsModel = configuration.model
            pendingLiveSettings = nil
            pendingLiveSettingsModel = nil
            try? persistLiveSettings(merged, depthIsExplicitSelection: depthIsExplicitSelection)
        } catch {
            if daemonIsRunning {
                if let daemonSettings = try? await apiClient.settings() {
                    settings = daemonSettings
                    liveSettingsModel = configuration.model
                }
                pendingLiveSettings = nil
                pendingLiveSettingsModel = nil
                throw error
            }
        }
    }

    public func refreshLiveSettingsFromDaemon(persist: Bool = false) async throws {
        guard daemonState == .running || supervisor.isRunning() else { return }
        adoptDaemonSettings(try await apiClient.settings(), persist: persist)
    }

    private func adoptDaemonSettings(_ daemonSettings: MutableSettings, persist: Bool) {
        let previous = settings
        settings = daemonSettings
        liveSettingsModel = configuration.model
        pendingLiveSettings = nil
        pendingLiveSettingsModel = nil
        if persist && previous != daemonSettings {
            try? persistLiveSettings(daemonSettings)
        }
    }

    nonisolated static func liveMutableSettingsPatch(from settings: MutableSettings) -> MutableSettings {
        MutableSettings(
            generationMode: settings.generationMode,
            depth: settings.depth,
            temperature: settings.temperature,
            topP: settings.topP,
            topK: settings.topK,
            presencePenalty: settings.presencePenalty,
            maxResponseTokens: settings.maxResponseTokens,
            streamInterval: settings.streamInterval,
            enableThinking: settings.enableThinking,
            reasoningParser: settings.reasoningParser,
            reasoning: settings.reasoning,
            reasoningEffort: settings.reasoningEffort,
            prefillChunkTokens: settings.prefillChunkTokens
        )
    }

    private func flushPendingLiveSettingsIfNeeded(target: LaunchTarget? = nil) async throws {
        guard let pending = pendingLiveSettings else { return }
        guard Self.targetCarriesSettingsSampler(target) else {
            pendingLiveSettings = nil
            pendingLiveSettingsModel = nil
            return
        }
        guard pendingLiveSettingsModel == nil || pendingLiveSettingsModel == configuration.model else {
            pendingLiveSettings = nil
            pendingLiveSettingsModel = nil
            return
        }
        guard let livePatch = Self.liveSettingsCarriedIntoTarget(
            pending,
            target: target,
            configuration: configuration
        ) else {
            pendingLiveSettings = nil
            pendingLiveSettingsModel = nil
            return
        }
        settings = try await apiClient.updateSettings(livePatch)
        liveSettingsModel = configuration.model
        pendingLiveSettings = nil
        pendingLiveSettingsModel = nil
    }

    private func flushFreshLaunchLiveOnlySettingsIfNeeded() async throws {
        guard let pending = pendingLiveSettings else { return }
        guard pendingLiveSettingsModel == nil || pendingLiveSettingsModel == configuration.model else {
            pendingLiveSettings = nil
            pendingLiveSettingsModel = nil
            return
        }
        guard let liveOnlyPatch = Self.liveOnlySettingsPatchAfterFreshLaunch(from: pending) else {
            pendingLiveSettings = nil
            pendingLiveSettingsModel = nil
            return
        }
        settings = try await apiClient.updateSettings(liveOnlyPatch)
        liveSettingsModel = configuration.model
        pendingLiveSettings = nil
        pendingLiveSettingsModel = nil
    }

    private nonisolated static func liveOnlySettingsPatchAfterFreshLaunch(
        from settings: MutableSettings
    ) -> MutableSettings? {
        let patch = MutableSettings(
            maxResponseTokens: settings.maxResponseTokens,
            streamInterval: settings.streamInterval
        )
        return patch.hasMutableLiveValue ? patch : nil
    }

    private func carryableLiveSettingsForCurrentModel(target: LaunchTarget? = nil) -> MutableSettings? {
        func compatible(_ settings: MutableSettings) -> MutableSettings? {
            Self.liveSettingsCarriedIntoTarget(
                settings,
                target: target,
                configuration: configuration
            )
        }
        if let pending = pendingLiveSettings,
           pendingLiveSettingsModel == nil || pendingLiveSettingsModel == configuration.model {
            return compatible(pending)
        }
        if let settings,
           liveSettingsModel == nil || liveSettingsModel == configuration.model {
            return compatible(settings)
        }
        return persistedLiveSettings(from: configuration, target: target)
    }

    private func seedLiveSettingsFromConfiguration(_ configuration: MTPLXAppConfiguration) {
        guard let persisted = persistedLiveSettings(from: configuration) else { return }
        settings = persisted
        pendingLiveSettings = persisted
        pendingLiveSettingsModel = configuration.model
        liveSettingsModel = configuration.model
    }

    private func persistLiveSettings(
        _ settings: MutableSettings,
        depthIsExplicitSelection: Bool = false
    ) throws {
        var next = configuration
        let family = MTPLXModelOption.modelFamily(for: configuration.model)
        next.liveSettingsModelFamily = family
        if let generationMode = normalizedGenerationMode(settings.generationMode) {
            next.generationMode = generationMode
        }
        persistDraftControlSelection(
            into: &next,
            settings: settings,
            family: family,
            depthIsExplicitSelection: depthIsExplicitSelection
        )
        if let temperature = settings.temperature { next.temperature = temperature }
        if let topP = settings.topP { next.topP = topP }
        if let topK = settings.topK { next.topK = topK }
        if let presencePenalty = settings.presencePenalty {
            next.presencePenalty = presencePenalty
        }
        if let reasoning = normalizedReasoning(settings.reasoning) {
            next.reasoning = reasoning
        }
        if let reasoningEffort = normalizedReasoningEffort(settings.reasoningEffort) {
            next.reasoningEffort = reasoningEffort
        }
        if let prefillChunkTokens = settings.prefillChunkTokens {
            next.prefillChunkTokens = prefillChunkTokens
        }
        configuration = next
        try settingsStore.save(next)
    }

    private func persistDraftControlSelection(
        into configuration: inout MTPLXAppConfiguration,
        settings: MutableSettings,
        family: String,
        depthIsExplicitSelection: Bool
    ) {
        // The tuned record is a per-model measurement (onboarding tune
        // or an explicit user depth choice). A merged live-settings
        // snapshot always carries the daemon's current depth, so
        // persisting it from reasoning/sampler riders would silently
        // erase the user's tune with whatever depth the launch preset
        // happened to run (QA-104 leg B).
        guard depthIsExplicitSelection,
              normalizedGenerationMode(settings.generationMode) != "ar",
              let value = settings.depth
        else {
            return
        }
        let controlField = Self.draftControlField(from: settings, family: family)
        guard Self.draftControlValueIsValid(
            value,
            family: family,
            controlField: controlField
        ) else {
            return
        }
        configuration.tunedControlRecord = TunedControlRecord(
            modelID: configuration.model,
            modelFamily: family,
            backendID: Self.backendID(for: family),
            controlField: controlField,
            controlValue: value,
            candidates: Self.draftControlCandidates(for: family, controlField: controlField),
            tunedAt: Date()
        )
        if controlField == "depth" {
            configuration.lastTunedDepth = value
        }
    }

    private func persistedLiveSettings(
        from configuration: MTPLXAppConfiguration,
        target: LaunchTarget? = nil
    ) -> MutableSettings? {
        guard persistedLiveSettingsCompatible(with: configuration) else { return nil }
        var persisted = MutableSettings()
        var hasValue = false
        if let generationMode = normalizedGenerationMode(configuration.generationMode),
           generationMode == "ar" || configuration.liveSettingsModelFamily != nil
        {
            persisted.generationMode = generationMode
            hasValue = true
        }
        if let temperature = configuration.temperature {
            persisted.temperature = temperature
            hasValue = true
        }
        if let topP = configuration.topP {
            persisted.topP = topP
            hasValue = true
        }
        if let topK = configuration.topK {
            persisted.topK = topK
            hasValue = true
        }
        if let presencePenalty = configuration.presencePenalty {
            persisted.presencePenalty = presencePenalty
            hasValue = true
        }
        if let reasoning = normalizedReasoning(configuration.reasoning) {
            persisted.reasoning = reasoning
            persisted.enableThinking = ChatReasoningPolicy.enableThinking(
                explicitMode: reasoning,
                modelFamily: MTPLXModelOption.modelFamily(for: configuration.model)
            )
            hasValue = true
        }
        if let reasoningEffort = normalizedReasoningEffort(configuration.reasoningEffort) {
            persisted.reasoningEffort = reasoningEffort
            hasValue = true
        }
        if persisted.generationMode != "ar",
           let controlValue = Self.persistedDraftControlValue(from: configuration)
        {
            persisted.depth = controlValue
            hasValue = true
        }
        if let prefillChunkTokens = configuration.prefillChunkTokens {
            persisted.prefillChunkTokens = prefillChunkTokens
            hasValue = true
        }
        return hasValue
            ? Self.liveSettingsCarriedIntoTarget(
                persisted,
                target: target,
                configuration: configuration
            )
            : nil
    }

    private func persistedLiveSettingsCompatible(with configuration: MTPLXAppConfiguration) -> Bool {
        let currentFamily = MTPLXModelOption.modelFamily(for: configuration.model)
        if let storedFamily = configuration.liveSettingsModelFamily,
           !storedFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return MTPLXModelOption.settingsFamiliesCompatible(
                stored: storedFamily,
                current: currentFamily
            )
        }
        return MTPLXModelOption.supportsTune(family: currentFamily)
    }

    private func normalizedReasoning(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "auto", "on", "off":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            return nil
        }
    }

    private func normalizedGenerationMode(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "mtp", "ar":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            return nil
        }
    }

    private func normalizedReasoningEffort(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "auto", "low", "medium", "high":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            return nil
        }
    }

    private static func targetCarriesSettingsSampler(_ target: LaunchTarget?) -> Bool {
        _ = target
        return true
    }

    nonisolated static func liveSettingsCarriedIntoTarget(
        _ settings: MutableSettings,
        target: LaunchTarget?,
        configuration: MTPLXAppConfiguration
    ) -> MutableSettings? {
        var carried = settings
        if !targetCarriesSettingsReasoning(target, model: configuration.model) {
            carried.enableThinking = nil
            carried.reasoning = nil
            carried.reasoningEffort = nil
        }
        return carried.hasMutableLiveValue ? carried : nil
    }

    private nonisolated static func targetCarriesSettingsReasoning(
        _ target: LaunchTarget?,
        model: String
    ) -> Bool {
        _ = target
        _ = model
        return true
    }

    private static func persistedDraftControlValue(from configuration: MTPLXAppConfiguration) -> Int? {
        let family = MTPLXModelOption.modelFamily(for: configuration.model)
        let controlField = family == "gemma4" ? "draft_block_size" : "depth"
        if controlField == "depth" {
            return configuration.compatibleTunedDepth()
        }
        return configuration.compatibleTunedControlValue(controlField: controlField)
    }

    private static func draftControlField(from settings: MutableSettings, family: String) -> String {
        if let field = settings.draftControl?.requestField?.trimmingCharacters(in: .whitespacesAndNewlines),
           !field.isEmpty
        {
            return field
        }
        return family == "gemma4" ? "draft_block_size" : "depth"
    }

    private static func draftControlValueIsValid(
        _ value: Int,
        family: String,
        controlField: String
    ) -> Bool {
        switch (family, controlField) {
        case ("qwen3_5", "depth"), ("qwen3_6", "depth"), ("step", "depth"):
            return (1...3).contains(value)
        case ("gemma4", "draft_block_size"):
            return (2...8).contains(value)
        default:
            return false
        }
    }

    private static func backendID(for family: String) -> String {
        switch family {
        case "gemma4": return "gemma4_assistant"
        case "step": return "step3p5_mtp"
        case "deepseek": return "deepseek_v3_mtp"
        case "glm": return "glm4_moe_mtp"
        default: return "qwen3_next"
        }
    }

    private static func draftControlCandidates(for family: String, controlField: String) -> [String] {
        if family == "gemma4" || controlField == "draft_block_size" {
            return ["2", "3", "4", "5", "6", "7", "8"]
        }
        return ["1", "2", "3"]
    }

    private func mergedLiveSettingsPatch(_ patch: MutableSettings) -> MutableSettings {
        var merged = carryableLiveSettingsForCurrentModel() ?? MutableSettings()
        if let generationMode = patch.generationMode { merged.generationMode = generationMode }
        if let depth = patch.depth { merged.depth = depth }
        if let temperature = patch.temperature { merged.temperature = temperature }
        if let topP = patch.topP { merged.topP = topP }
        if let topK = patch.topK { merged.topK = topK }
        if let presencePenalty = patch.presencePenalty {
            merged.presencePenalty = presencePenalty
        }
        merged.maxResponseTokens = patch.maxResponseTokens
        if let streamInterval = patch.streamInterval { merged.streamInterval = streamInterval }
        if let enableThinking = patch.enableThinking { merged.enableThinking = enableThinking }
        if let reasoningParser = patch.reasoningParser { merged.reasoningParser = reasoningParser }
        if let reasoning = patch.reasoning { merged.reasoning = reasoning }
        if let reasoningEffort = patch.reasoningEffort { merged.reasoningEffort = reasoningEffort }
        if let prefillChunkTokens = patch.prefillChunkTokens {
            merged.prefillChunkTokens = prefillChunkTokens
        }
        return merged
    }

    public func cancel(requestId: String) async throws {
        _ = try await apiClient.cancel(requestId: requestId)
    }

    public func clearCache() async throws {
        _ = try await apiClient.clearCache()
        self.sessions = try await apiClient.sessions()
    }

    public func clearSession(sessionId: String) async throws {
        _ = try await apiClient.clearSession(sessionId: sessionId)
        self.sessions = try await apiClient.sessions()
    }

    public func startMetricsStream() {
        streamTask?.cancel()
        let client = MetricsStreamClient(apiClient: apiClient)
        let interval = configuration.performanceLock ? 1000 : configuration.streamSnapshotIntervalMs
        streamTask = Task { [weak self] in
            await client.connect(
                snapshotIntervalMs: interval,
                onState: { state in
                    await MainActor.run { self?.connectionState = state }
                },
                onEvent: { event in
                    await MainActor.run { self?.apply(event: event) }
                }
            )
        }
        startDaemonHealthWatchdog()
    }

    public func markDaemonUnreachable(reason: String) {
        markDaemonUnreachableIfNeeded(reason: reason)
    }

    private var shouldProbeDaemonHealth: Bool {
        switch daemonState {
        case .running, .warming:
            return true
        case .stopped, .starting, .degraded, .stopping, .crashed:
            return false
        }
    }

    /// Per-probe budget for the running-daemon watchdog. A healthy
    /// daemon answers /health in milliseconds even mid-decode; one
    /// that can't answer twice in a row within this window is wedged
    /// for UI purposes (QA-113's alive-but-unresponsive shape), not
    /// merely slow.
    private static let watchdogProbeDeadlineSeconds: TimeInterval = 10

    private func startDaemonHealthWatchdog() {
        healthWatchTask?.cancel()
        let probeClient = MTPLXAPIClient.livenessProbe(
            baseURL: baseURL,
            apiKey: configuration.apiKey
        )
        healthWatchTask = Task { @MainActor [weak self] in
            defer { probeClient.session.finishTasksAndInvalidate() }
            var consecutiveMisses = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                guard let self, !Task.isCancelled else { return }
                guard self.shouldProbeDaemonHealth else {
                    consecutiveMisses = 0
                    continue
                }
                if let health = await probeClient.healthWithinDeadline(
                    seconds: Self.watchdogProbeDeadlineSeconds
                ), health.ok {
                    consecutiveMisses = 0
                    self.health = health
                    self.currentFanMode = self.verifiedFanMode(from: health)
                    continue
                }
                consecutiveMisses += 1
                guard consecutiveMisses >= 2 else { continue }
                self.markDaemonUnreachableIfNeeded(
                    reason: "MTPLX lost contact with the model server. Start it again."
                )
                return
            }
        }
    }

    private func markDaemonUnreachableIfNeeded(reason: String) {
        switch daemonState {
        case .running, .warming, .starting:
            break
        case .stopped, .degraded, .stopping, .crashed:
            return
        }

        let startupPID = health?.startup?.pid.map(pid_t.init)
        healthWatchTask?.cancel()
        healthWatchTask = nil
        streamTask?.cancel()
        streamTask = nil
        lateHealthRecoveryTask?.cancel()
        lateHealthRecoveryTask = nil
        connectionState = .failed(reason)
        daemonState = .degraded(reason)
        startupPhase = .failed(reason)
        clearLiveMetricsState()

        let previousTeardown = daemonTeardownTask
        daemonTeardownTask = Task { @MainActor [self] in
            await previousTeardown?.value
            await supervisor.logs.append(
                "daemon became unreachable; reaping stale process family",
                stream: .system
            )
            let additionalPIDs = startupPID.map { [$0] } ?? []
            await supervisor.stop(graceSeconds: 1.0, additionalProcessIDs: additionalPIDs)
            await refreshLogs()
        }
    }

    public func refreshLogs() async {
        logs = await supervisor.logs.snapshot()
    }

    private func promptForModelDownloadIfNeeded(
        configuration: MTPLXAppConfiguration,
        target: LaunchTarget?,
        launchAction: PendingModelDownloadLaunchAction
    ) -> Bool {
        guard let option = downloadableModelOption(for: configuration.model) else {
            return false
        }
        let selectedPath = NSString(string: configuration.model).expandingTildeInPath
        if FileManager.default.fileExists(atPath: selectedPath),
           MTPLXModelOption.hasCompleteInstall(at: selectedPath)
        {
            return false
        }
        if let installedPath = option.installedLocalPath {
            var next = configuration
            if next.model != installedPath {
                next.model = installedPath
                self.configuration = next
                try? settingsStore.save(next)
            }
            return false
        }

        pendingModelDownload = PendingModelDownload(
            repoID: option.hfModelID,
            displayName: option.displayName,
            shortName: option.shortName,
            target: target,
            launchAction: launchAction,
            totalBytes: option.sizeBytes > 0 ? option.sizeBytes : nil,
            destinationPath: modelDownloader.cachedModelPath(for: option.hfModelID).path
        )
        modelDownloadProgress = nil
        modelDownloadFailure = nil
        daemonState = .stopped
        startupPhase = .idle
        return true
    }

    private func downloadableModelOption(for model: String) -> MTPLXModelOption? {
        let rows = MTPLXModelOption.pickerCatalog(
            customModels: configuration.customModels,
            currentModel: model
        )
        if let match = rows.first(where: { $0.matches(model) }) {
            return match
        }
        return MTPLXModelOption.customHuggingFaceModel(repoID: model)
    }

    func handleModelDownloadEvent(
        _ event: DownloadEvent,
        request: PendingModelDownload
    ) async {
        switch event {
        case .started(let path):
            modelDownloadFailure = nil
            modelDownloadProgress = DownloadProgressSnapshot(
                destinationPath: path,
                bytesOnDisk: 0,
                totalBytes: request.totalBytes,
                bytesPerSecond: 0,
                etaSeconds: nil,
                stalledSeconds: 0,
                isComplete: false,
                statusMessage: "Resolving files"
            )
        case .status(let message, let bytes, let total, let path):
            modelDownloadFailure = nil
            var snapshot = modelDownloadProgress ?? DownloadProgressSnapshot(
                destinationPath: path ?? request.destinationPath,
                bytesOnDisk: bytes ?? 0,
                totalBytes: total ?? request.totalBytes,
                bytesPerSecond: 0,
                etaSeconds: nil,
                stalledSeconds: 0,
                isComplete: false,
                statusMessage: message
            )
            if let path { snapshot.destinationPath = path }
            if let bytes { snapshot.bytesOnDisk = bytes }
            if let total { snapshot.totalBytes = total }
            snapshot.statusMessage = message
            modelDownloadProgress = snapshot
        case .progress(let bytes, let total, let smoothed, let eta):
            modelDownloadFailure = nil
            modelDownloadProgress = DownloadProgressSnapshot(
                destinationPath: modelDownloadProgress?.destinationPath ?? request.destinationPath,
                bytesOnDisk: bytes,
                totalBytes: total,
                bytesPerSecond: smoothed,
                etaSeconds: eta,
                stalledSeconds: 0,
                isComplete: false,
                statusMessage: "Downloading"
            )
        case .stalled(let seconds):
            modelDownloadFailure = nil
            if var snapshot = modelDownloadProgress {
                snapshot.stalledSeconds = seconds
                snapshot.bytesPerSecond = 0
                snapshot.etaSeconds = nil
                snapshot.statusMessage = "Waiting on Hugging Face"
                modelDownloadProgress = snapshot
            }
        case .complete(let bytes, let path):
            guard MTPLXModelOption.hasCompleteInstall(at: path) else {
                modelDownloadProgress = DownloadProgressSnapshot(
                    destinationPath: path,
                    bytesOnDisk: bytes,
                    totalBytes: request.totalBytes ?? bytes,
                    bytesPerSecond: 0,
                    etaSeconds: nil,
                    stalledSeconds: 0,
                    isComplete: false,
                    statusMessage: "Incomplete"
                )
                modelDownloadFailure = "Download finished, but the model folder is missing required MTPLX files. Press Retry to resume the Hugging Face download."
                isModelDownloading = false
                modelDownloadTask = nil
                return
            }
            modelDownloadFailure = nil
            modelDownloadProgress = DownloadProgressSnapshot(
                destinationPath: path,
                bytesOnDisk: bytes,
                totalBytes: request.totalBytes ?? bytes,
                bytesPerSecond: 0,
                etaSeconds: 0,
                stalledSeconds: 0,
                isComplete: true,
                statusMessage: "Ready"
            )
            isModelDownloading = false
            modelDownloadTask = nil
            if beginPostDownloadTuneIfSupported(request: request, installedPath: path) {
                return
            }
            do {
                try await finishModelInstall(
                    repoID: request.repoID,
                    installedPath: path,
                    target: request.target,
                    launchAction: request.launchAction
                )
                clearModelDownloadAndTuneState()
            } catch {
                modelDownloadFailure = friendlyDownloadFailure(String(describing: error))
            }
        case .failed(_, let stderrTail):
            modelDownloadFailure = friendlyDownloadFailure(stderrTail)
            isModelDownloading = false
            modelDownloadTask = nil
        case .cancelled:
            isModelDownloading = false
            modelDownloadTask = nil
            if var snapshot = modelDownloadProgress, !snapshot.isComplete {
                snapshot.bytesPerSecond = 0
                snapshot.etaSeconds = nil
                snapshot.stalledSeconds = 0
                snapshot.statusMessage = "Paused"
                modelDownloadProgress = snapshot
            }
        }
    }

    public func runPendingModelTune() {
        guard let request = pendingModelTune, !isModelTuning else { return }
        modelTuneTask?.cancel()
        modelTuneFailure = nil
        modelTuneResult = nil
        modelTuneStatusMessage = "Preparing max fans and loading model"
        modelTuneCandidatesLanded = [:]
        isModelTuning = true
        let tuner = autoTuner
        modelTuneTask = Task { [weak self] in
            for await event in tuner.stream(modelPath: request.installedPath, candidates: request.candidates) {
                if Task.isCancelled { break }
                await MainActor.run {
                    self?.handleModelTuneEvent(event, request: request)
                }
            }
        }
    }

    public func skipPendingModelTune() {
        guard let request = pendingModelTune, !isModelTuning else { return }
        modelTuneFailure = nil
        modelTuneStatusMessage = nil
        modelTuneResult = nil
        modelTuneCandidatesLanded = [:]
        Task { @MainActor [weak self] in
            do {
                try self?.saveDownloadedModelInstall(
                    repoID: request.repoID,
                    installedPath: request.installedPath,
                    tuneResult: nil,
                    useSafeTuneDefault: true
                )
                try await self?.continueDownloadedModelLaunch(request)
                self?.clearModelDownloadAndTuneState()
            } catch {
                self?.modelTuneFailure = self?.friendlyDownloadFailure(String(describing: error))
            }
        }
    }

    public func startPendingTunedModel() {
        guard let request = pendingModelTune, !isModelTuning else { return }
        Task { @MainActor [weak self] in
            do {
                try await self?.continueDownloadedModelLaunch(request)
                self?.clearModelDownloadAndTuneState()
            } catch {
                self?.modelTuneFailure = self?.friendlyDownloadFailure(String(describing: error))
            }
        }
    }

    private func handleModelTuneEvent(_ event: TuneEvent, request: PendingModelTune) {
        guard pendingModelTune == request else { return }
        switch event {
        case .installingFanControl(let message):
            modelTuneStatusMessage = message
        case .started:
            modelTuneStatusMessage = "Preparing max fans and loading model"
        case .candidateLanded(let result):
            modelTuneStatusMessage = nil
            modelTuneCandidatesLanded[result.candidate] = result
        case .completed(let result):
            modelTuneStatusMessage = "Saved"
            modelTuneResult = result
            for entry in result.allCandidates {
                modelTuneCandidatesLanded[entry.candidate] = entry
            }
            isModelTuning = false
            modelTuneTask = nil
            do {
                try saveDownloadedModelInstall(
                    repoID: request.repoID,
                    installedPath: request.installedPath,
                    tuneResult: result,
                    useSafeTuneDefault: true
                )
            } catch {
                modelTuneFailure = friendlyDownloadFailure(String(describing: error))
            }
        case .failed(_, let stderrTail):
            modelTuneStatusMessage = nil
            modelTuneFailure = stderrTail.isEmpty ? "Tuning failed." : stderrTail
            isModelTuning = false
            modelTuneTask = nil
        case .cancelled:
            modelTuneStatusMessage = nil
            isModelTuning = false
            modelTuneTask = nil
        }
    }

    private func beginPostDownloadTuneIfSupported(
        request: PendingModelDownload,
        installedPath: String
    ) -> Bool {
        let pathFamily = MTPLXModelOption.modelFamily(for: installedPath)
        let repoFamily = MTPLXModelOption.modelFamily(for: request.repoID)
        let family = pathFamily == "unknown" ? repoFamily : pathFamily
        let candidates = TuneCandidate.candidates(forFamily: family)
        guard !candidates.isEmpty else { return false }
        pendingModelTune = PendingModelTune(
            repoID: request.repoID,
            installedPath: installedPath,
            displayName: request.displayName,
            shortName: request.shortName,
            modelFamily: family,
            target: request.target,
            launchAction: request.launchAction,
            candidates: candidates
        )
        modelTuneCandidatesLanded = [:]
        modelTuneResult = nil
        modelTuneFailure = nil
        modelTuneStatusMessage = nil
        return true
    }

    private func finishModelInstall(
        repoID: String,
        installedPath: String,
        target: LaunchTarget?,
        launchAction: PendingModelDownloadLaunchAction
    ) async throws {
        try saveDownloadedModelInstall(
            repoID: repoID,
            installedPath: installedPath,
            tuneResult: nil,
            useSafeTuneDefault: false
        )
        switch launchAction {
        case .restart:
            try await applyConfiguration(configuration, restartIfRunning: true)
        case .start:
            await startDaemon(target: target)
        }
    }

    @discardableResult
    private func saveDownloadedModelInstall(
        repoID: String,
        installedPath: String,
        tuneResult: TuneResult?,
        useSafeTuneDefault: Bool
    ) throws -> MTPLXAppConfiguration {
        var next = configuration
        next.model = installedPath
        next.rememberCustomModel(repoID: repoID)
        let pathFamily = MTPLXModelOption.modelFamily(for: installedPath)
        let repoFamily = MTPLXModelOption.modelFamily(for: repoID)
        let family = pathFamily == "unknown" ? repoFamily : pathFamily
        if let tuneResult {
            next.saveTuneResult(
                modelPath: installedPath,
                repoID: repoID,
                family: family,
                result: tuneResult
            )
        } else if useSafeTuneDefault {
            next.saveSafeTunedDefault(
                modelPath: installedPath,
                repoID: repoID,
                family: family
            )
        }
        try saveSettings(next)
        return next
    }

    private func continueDownloadedModelLaunch(_ request: PendingModelTune) async throws {
        switch request.launchAction {
        case .restart:
            try await applyConfiguration(configuration, restartIfRunning: true)
        case .start:
            await startDaemon(target: request.target)
        }
    }

    private func clearModelDownloadAndTuneState() {
        pendingModelDownload = nil
        modelDownloadProgress = nil
        modelDownloadFailure = nil
        clearModelTuneState()
    }

    private func clearModelTuneState() {
        modelTuneTask?.cancel()
        modelTuneTask = nil
        pendingModelTune = nil
        modelTuneCandidatesLanded = [:]
        modelTuneResult = nil
        modelTuneFailure = nil
        modelTuneStatusMessage = nil
        isModelTuning = false
    }

    private func friendlyDownloadFailure(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        if lower.contains("401") || lower.contains("403") || lower.contains("gated") || lower.contains("private") {
            return "This Hugging Face repo is private or gated. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN, then try again."
        }
        if lower.contains("no space left") || lower.contains("not enough free disk") {
            return "There is not enough free disk space to finish this download."
        }
        if lower.contains("timed out") || lower.contains("network") || lower.contains("connection") {
            if MTPLXAppConfiguration.hfMirrorEnvironment(configuration.hfEndpoint) == nil {
                return "The download could not reach Hugging Face. Check the network connection and try again. "
                    + "If huggingface.co is blocked on your network, set an HF download mirror in "
                    + "Settings under Advanced, for example https://hf-mirror.com."
            }
            return "The download could not reach Hugging Face. Check the network connection and try again."
        }
        return trimmed.isEmpty ? "Download failed. Try again." : trimmed
    }

    private func fanMode(for configuration: MTPLXAppConfiguration) -> MTPLXFanMode {
        MTPLXFanMode.normalized(configuration.fanMode)
    }

    private func requiresStartupFanRamp(_ configuration: MTPLXAppConfiguration) -> Bool {
        fanMode(for: configuration) == .max
    }

    private func modeRequiresFanRestore(_ mode: String?) -> Bool {
        switch MTPLXFanMode.normalized(mode) {
        case .max, .smart: return true
        case .default: return false
        }
    }

    private var canApplyFanModeLive: Bool {
        switch daemonState {
        case .running, .warming:
            return true
        default:
            return false
        }
    }

    /// POST `/v1/mtplx/thermal/fan_mode {"mode": mode}` and update
    /// `currentFanMode` to whatever the daemon verified. Optimistically
    /// pre-set the mode so the UI flips immediately; on failure
    /// `currentFanMode` is rolled back to the previous state.
    public func setFanMode(_ mode: String) async throws {
        let previous = currentFanMode
        let previousConfiguration = configuration
        let fanMode = MTPLXFanMode.normalized(mode)
        let canonicalMode = fanMode.rawValue
        if !canApplyFanModeLive {
            var next = configuration
            next.fanMode = canonicalMode
            next.pinFansAtMaxOnStart = fanMode == .max
            try saveSettings(next)
            currentFanMode = nil
            fanRestoreRequiredOnStop = false
            return
        }
        do {
            let result = try await apiClient.setFanMode(
                canonicalMode,
                requireActualRamp: fanMode == .max,
                timeoutS: fanMode == .max ? 25 : nil
            )
            currentFanMode = MTPLXFanMode.normalized(result.currentMode ?? canonicalMode).rawValue
            fanRestoreRequiredOnStop = modeRequiresFanRestore(currentFanMode)
            var next = configuration
            next.fanMode = currentFanMode ?? canonicalMode
            next.pinFansAtMaxOnStart = MTPLXFanMode.normalized(next.fanMode) == .max
            try saveSettings(next)
        } catch {
            currentFanMode = previous
            configuration = previousConfiguration
            throw error
        }
    }

    /// Pull thermal detection + current mode + fan summary. Used after
    /// daemon start so `FanModeToggle` can decide whether to render.
    public func refreshThermalStatus() async {
        thermalStatus = try? await apiClient.thermalStatus()
        if let mode = thermalStatus?.values["current_mode"]?.stringValue, !mode.isEmpty {
            currentFanMode = MTPLXFanMode.normalized(mode).rawValue
            fanRestoreRequiredOnStop = modeRequiresFanRestore(currentFanMode)
        }
    }

    private func shouldRestoreFanModeOnStop() -> Bool {
        if fanRestoreRequiredOnStop {
            return true
        }
        if modeRequiresFanRestore(currentFanMode) {
            return true
        }
        if health?.thermal?.actualRampVerified == true {
            return true
        }
        if let mode = thermalStatus?.values["current_mode"]?.stringValue?.lowercased(),
           mode == "max" || mode == "performance" {
            return true
        }
        return modeRequiresFanRestore(configuration.fanMode)
    }

    private func shouldRestoreFansAfterFailedStartup(phase: DaemonStartupPhase) -> Bool {
        guard requiresStartupFanRamp(configuration) else { return false }
        switch phase {
        case .waitingForOwnedHealth, .rampingFans, .warming:
            return true
        case .idle, .launching, .ready, .failed:
            return fanRestoreRequiredOnStop || modeRequiresFanRestore(currentFanMode)
        }
    }

    @discardableResult
    private func restoreFansLocally(successLog: String) async -> Bool {
        let restored = await localFanRestorer()
        if restored {
            fanRestoreRequiredOnStop = false
            currentFanMode = MTPLXFanMode.default.rawValue
            await supervisor.logs.append(successLog, stream: .system)
        }
        return restored
    }

    private static func restoreFanModeWithLocalThermalforge() async -> Bool {
        await Task.detached(priority: .utility) {
            let home = FileManager.default.homeDirectoryForCurrentUser.path
            let marker = URL(fileURLWithPath: "\(home)/.mtplx/max-active.json")
            let candidates = [
                "\(home)/.mtplx/bin/thermalforge",
                "/opt/homebrew/bin/thermalforge",
                "/usr/local/bin/thermalforge"
            ]
            for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
                if restoreFanMode(withThermalforge: path, marker: marker) {
                    return true
                }
            }
            return false
        }.value
    }

    nonisolated private static func restoreFanMode(
        withThermalforge path: String,
        marker: URL
    ) -> Bool {
        let invocations: [(String, [String])] = [
            (path, ["auto"]),
            ("/usr/bin/sudo", ["-n", path, "auto"])
        ].filter { executable, _ in
            FileManager.default.isExecutableFile(atPath: executable)
        }

        for attempt in 0..<4 {
            for (executable, arguments) in invocations {
                let result = runFanCommand(executable: executable, arguments: arguments)
                guard result.exitCode == 0 else { continue }
                if thermalforgeReportsAuto(path) {
                    try? FileManager.default.removeItem(at: marker)
                    return true
                }
            }
            if attempt < 3 {
                Thread.sleep(forTimeInterval: 0.25)
            }
        }
        return false
    }

    nonisolated private static func thermalforgeReportsAuto(_ path: String) -> Bool {
        let result = runFanCommand(executable: path, arguments: ["status"])
        guard result.exitCode == 0,
              let data = result.stdout.data(using: .utf8),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return false
        }
        guard let fans = payload["fans"] as? [[String: Any]], !fans.isEmpty else {
            return false
        }
        return fans.allSatisfy { fan in
            let mode = (fan["mode"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let target = doubleValue(fan["target_rpm"])
            return mode == "auto" || (target.map { $0 <= 100 } ?? false)
        }
    }

    nonisolated private static func doubleValue(_ raw: Any?) -> Double? {
        if let number = raw as? NSNumber {
            return number.doubleValue
        }
        if let string = raw as? String {
            return Double(string)
        }
        return raw as? Double
    }

    private struct FanCommandResult {
        var exitCode: Int32
        var stdout: String
    }

    nonisolated private static func runFanCommand(
        executable: String,
        arguments: [String]
    ) -> FanCommandResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            return FanCommandResult(
                exitCode: process.terminationStatus,
                stdout: String(decoding: data, as: UTF8.self)
            )
        } catch {
            return FanCommandResult(exitCode: 127, stdout: "")
        }
    }

    private func finishReadyDaemon(
        target: LaunchTarget?,
        configuration: MTPLXAppConfiguration,
        replaceExistingClient: Bool
    ) async {
        daemonState = .running
        startupPhase = .ready
        await launchClientHandoff(
            target: target,
            configuration: configuration,
            replaceExisting: replaceExistingClient
        )
        await refreshPostStartState(target: target, configuration: configuration)
    }

    private func launchClientHandoff(
        target: LaunchTarget?,
        configuration: MTPLXAppConfiguration,
        replaceExisting: Bool
    ) async {
        if target == .hermes {
            await launchHermesTerminalHandoff(
                configuration: configuration,
                replaceExisting: replaceExisting
            )
        }
        if target == .openCode {
            let desktop = await openCodeIntegration.reloadDesktopAfterDaemonReady()
            clientHandoffNotice = ClientHandoffNotice.openCode(result: desktop)
            await supervisor.logs.append(
                "OpenCode Desktop handoff \(desktop.action.rawValue): \(desktop.detail)",
                stream: .system
            )
        }
        if target == .pi {
            await launchPiTerminalHandoff(
                configuration: configuration,
                replaceExisting: replaceExisting
            )
        }
        onDaemonReady?(target)
    }

    private func refreshPostStartState(
        target: LaunchTarget?,
        configuration: MTPLXAppConfiguration
    ) async {
        do {
            try await refreshStaticState()
        } catch {
            await supervisor.logs.append(
                "post-start state refresh failed: \(String(describing: error))",
                stream: .system
            )
        }
        do {
            try await flushFreshLaunchLiveOnlySettingsIfNeeded()
        } catch {
            await supervisor.logs.append(
                "post-start settings sync failed: \(String(describing: error))",
                stream: .system
            )
        }
        startMetricsStream()
        await refreshThermalStatus()
        do {
            try await verifyPinnedFansAfterStartup(configuration: configuration)
        } catch {
            startupPhase = .ready
            await supervisor.logs.append(
                "post-start fan verification failed: \(String(describing: error))",
                stream: .system
            )
        }
    }

    private func verifyPinnedFansAfterStartup(configuration: MTPLXAppConfiguration) async throws {
        guard requiresStartupFanRamp(configuration) else { return }
        startupPhase = .rampingFans
        try await setFanMode(MTPLXFanMode.max.rawValue)
        await refreshThermalStatus()
        startupPhase = .ready
    }

    private func launchHermesTerminalHandoff(
        configuration: MTPLXAppConfiguration,
        replaceExisting: Bool
    ) async {
        if replaceExisting {
            let stopped = hermesIntegration.stopLaunchedTerminalAgents()
            if stopped > 0 {
                await supervisor.logs.append(
                    "stopped \(stopped) previous Hermes Terminal handoff(s)",
                    stream: .system
                )
            }
        } else if hermesIntegration.hasLaunchedTerminalAgent() {
            return
        }

        let launch = hermesIntegration.launchInTerminal(configuration: configuration)
        clientHandoffNotice = ClientHandoffNotice.hermes(result: launch)
        await supervisor.logs.append(
            "Hermes handoff \(launch.action.rawValue): \(launch.detail)",
            stream: .system
        )
    }

    private func launchPiTerminalHandoff(
        configuration: MTPLXAppConfiguration,
        replaceExisting: Bool
    ) async {
        if replaceExisting && !launchedPiAgentPIDs.isEmpty {
            let stopped = piIntegration.stopLaunchedAgents(processIDs: Array(launchedPiAgentPIDs))
            launchedPiAgentPIDs.removeAll()
            piTerminalAgentRunning = false
            piTerminalAgentProcessIDs = []
            if stopped > 0 {
                await supervisor.logs.append(
                    "stopped \(stopped) previous Pi Terminal handoff(s)",
                    stream: .system
                )
            }
        } else if !replaceExisting && !launchedPiAgentPIDs.isEmpty {
            return
        }

        let launch = piIntegration.launchInTerminal(configuration: configuration)
        launchedPiAgentPIDs.formUnion(launch.launchedProcessIDs)
        piTerminalAgentRunning = !launchedPiAgentPIDs.isEmpty
        piTerminalAgentProcessIDs = Array(launchedPiAgentPIDs).sorted()
        piTerminalLaunchCommand = launch.command
        piTerminalLaunchDetail = launch.detail
        clientHandoffNotice = ClientHandoffNotice.pi(result: launch)
        await supervisor.logs.append(
            "Pi handoff \(launch.action.rawValue): \(launch.detail)",
            stream: .system
        )
    }

    public func refreshPrefillHistory() async {
        prefillHistory = try? await apiClient.prefillHistory()
    }

    public func refreshModels() async {
        models = try? await apiClient.models()
    }

    /// Up-to-date `MTPLXAPIClient` derived from the current
    /// configuration. Public so adjacent stores (e.g. `ChatViewModel`)
    /// can stay in sync with port / API-key changes without holding a
    /// stale reference.
    public var apiClient: MTPLXAPIClient {
        MTPLXAPIClient(baseURL: baseURL, apiKey: configuration.apiKey)
    }

    /// Daemon base URL (`http://<host>:<port>`).
    public var baseURL: URL {
        URL(string: "http://\(configuration.host):\(configuration.port)")!
    }

    private func scheduleLateHealthRecovery(launchID: String, target: LaunchTarget?) {
        guard supervisor.isRunning() else { return }
        lateHealthRecoveryTask?.cancel()
        lateHealthRecoveryTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                self.startupPhase = .waitingForOwnedHealth
                let recoveredHealth = try await self.supervisor.waitForExistingHealth(
                    healthBaseURL: self.baseURL,
                    apiKey: self.configuration.apiKey,
                    timeoutSeconds: self.daemonStartupTimeoutSeconds,
                    expectedLaunchID: launchID,
                    requireActualFanRamp: self.requiresStartupFanRamp(self.configuration),
                    onPhase: { phase in
                        Task { @MainActor [weak self] in
                            self?.startupPhase = phase
                        }
                    }
                )
                guard !Task.isCancelled else { return }
                self.health = recoveredHealth
                self.currentFanMode = self.verifiedFanMode(from: recoveredHealth)
                self.fanRestoreRequiredOnStop = self.fanRestoreRequiredOnStop
                    || self.modeRequiresFanRestore(self.currentFanMode)
                await self.finishReadyDaemon(
                    target: target,
                    configuration: self.configuration,
                    replaceExistingClient: true
                )
            } catch {
                guard !Task.isCancelled else { return }
                await self.refreshLogs()
            }
        }
    }

    private func apply(snapshot: DashboardSnapshot) {
        self.snapshot = snapshot
        // Merge the snapshot's `latest` into the existing one rather than
        // replacing it wholesale. Snapshot frames interleave with the SSE
        // `.progress`/`.completed` frames, and a snapshot's in-flight
        // `latest` routinely omits fields the stream already populated —
        // most visibly `accepted_by_depth`. The old wholesale replace
        // nulled those fields on every poll, so any view that keys off a
        // field's presence (the per-depth acceptance bars above all)
        // strobed present/absent roughly twice a second. `mergeLatestValues`
        // keeps stable cross-frame fields alive while still resetting
        // cleanly when a new request_id / session_id arrives, and a
        // latest-less snapshot no longer wipes the live readout.
        if let incoming = snapshot.latest, shouldAcceptSnapshotLatest(snapshot) {
            let merged = Self.mergeLatestValues(existing: latest?.values ?? [:],
                                                incoming: incoming.values)
            let mergedLatest = MetricsLatest(values: merged)
            latest = mergedLatest
            updateSmoothedMetrics(mergedLatest)
        }
        rolling = snapshot.rolling
        inFlight = snapshot.inFlight
        if !snapshot.inFlight.contains(where: Self.hasActivePrefill) {
            prefillStatus = nil
        }
        sessions = snapshot.sessions
        sessionBank = snapshot.sessionBank
        mem = snapshot.mem
        thermal = snapshot.thermal
        if daemonState == .running || supervisor.isRunning() {
            adoptDaemonSettings(snapshot.settings, persist: true)
        } else {
            settings = carryableLiveSettingsForCurrentModel() ?? snapshot.settings
            liveSettingsModel = settings == nil ? nil : configuration.model
        }
        scheduler = snapshot.scheduler
        // A dashboard snapshot's `latest` is the daemon's most recent
        // completed request, while current request progress lives under
        // `in_flight[].last_progress`. During generation, accepting that
        // completed `latest` would make the hero gauge alternate between
        // the live current TPS and the previous/peak-like completed TPS.
        if shouldAcceptSnapshotLatest(snapshot), let value = headlineDecodeTPS(from: latest) {
            updateHeadlineDecode(value: value, isCompletion: snapshot.inFlight.isEmpty)
        }
        recordAIMEBackendMetrics(
            "backend_snapshot",
            values: snapshot.latest?.values,
            extra: [
                "published": .bool(true),
                "source": .string("snapshot")
            ]
        )
    }

    private func apply(event: MetricsStreamEvent) {
        switch event {
        case .snapshot(let snapshot):
            apply(snapshot: snapshot)
        case .progress(let payload):
            let progressValues = payload.values["progress"]?.objectValue ?? payload.values
            AIMEDiagnostics.signpost(.backendMetricsReceive)
            recordAIMEBackendMetrics(
                "backend_progress_received",
                values: progressValues,
                extra: [
                    "published": .bool(false),
                    "source": .string("progress")
                ]
            )
            guard shouldPublishProgressFrame() else {
                recordAIMEBackendMetrics(
                    "backend_progress_throttled",
                    values: progressValues,
                    extra: [
                        "published": .bool(false),
                        "source": .string("progress")
                    ]
                )
                return
            }
            if let progress = payload.values["progress"]?.objectValue {
                // Merge into the existing latest payload rather than
                // replacing it. The progress sub-payload often omits
                // fields the snapshot pre-populated (e.g.
                // `display_decode_tok_s`, the sliding decode window
                // keys). Replacing wholesale caused the decode hero to
                // alternate between two integers as the
                // `displayDecodeTokS` cascade picked a different source
                // each frame.
                let merged = Self.mergeLatestValues(existing: latest?.values ?? [:],
                                                   incoming: progress)
                let mergedLatest = MetricsLatest(values: merged)
                latest = mergedLatest
                updateSmoothedMetrics(mergedLatest)
                if let value = headlineDecodeTPS(from: mergedLatest) {
                    updateHeadlineDecode(value: value, isCompletion: false)
                }
                if Self.hasDecodeProgress(progress) {
                    prefillStatus = nil
                }
                observedUserMetricEventCount += 1
                recordAIMEBackendMetrics(
                    "backend_progress_published",
                    values: merged,
                    extra: [
                        "published": .bool(true),
                        "source": .string("progress")
                    ]
                )
            }
        case .completed(let payload):
            // Completion frames carry the full envelope so they can
            // safely replace `latest`.
            let envelope = MetricsLatest(values: payload.values["envelope"]?.objectValue ?? payload.values)
            latest = envelope
            // Snap the smoothed metrics to the request's exact final
            // values, then freeze them so subsequent idle snapshot polls
            // can't keep nudging acceptance/cached upward after the
            // request has already finished.
            updateSmoothedMetrics(envelope, snapToFinal: true)
            smoothedFrozen = true
            if let value = headlineDecodeTPS(from: envelope) {
                updateHeadlineDecode(value: value, isCompletion: true)
            }
            prefillStatus = nil
            observedUserMetricEventCount += 1
            observedCompletionCount += 1
            AIMEDiagnostics.signpost(.backendMetricsReceive)
            recordAIMEBackendMetrics(
                "backend_completed",
                values: envelope.values,
                extra: [
                    "published": .bool(true),
                    "source": .string("completed")
                ]
            )
        case .thermal(let payload):
            if let thermalValues = payload.values["thermal"]?.objectValue,
               let value = try? Self.decode(ThermalSnapshot.self, from: DynamicObject(values: thermalValues)) {
                thermal = value
            } else if let value = try? Self.decode(ThermalSnapshot.self, from: payload) {
                thermal = value
            }
        case .prefill(let payload):
            if Self.prefillPhase(in: payload) == "completed" {
                prefillStatus = nil
            } else {
                prefillStatus = payload
            }
        default:
            break
        }
    }

    private func recordAIMEBackendMetrics(
        _ name: String,
        values: [String: JSONValue]?,
        extra: [String: AIMEDiagnosticValue] = [:]
    ) {
        guard AIMEDiagnostics.isEnabled else { return }
        let latestValues = latest?.values
        let requestID = values?["request_id"]?.stringValue
            ?? latestValues?["request_id"]?.stringValue
        let tokenCount = values?["completion_tokens"]?.intValue
            ?? values?["generated_tokens"]?.intValue
            ?? latestValues?["completion_tokens"]?.intValue
            ?? latestValues?["generated_tokens"]?.intValue
        let intervalS: TimeInterval
        switch name {
        case "backend_snapshot":
            intervalS = 2
        case "backend_progress_received":
            intervalS = 2
        case "backend_progress_throttled":
            intervalS = 5
        default:
            intervalS = 1
        }
        let force = name == "backend_completed"
        guard AIMEDiagnostics.shouldRecordCadenced(
            name,
            intervalS: intervalS,
            tokenCount: tokenCount,
            identity: requestID,
            force: force
        ) else { return }

        var fields = values.map { AIMEDiagnostics.metricFields(from: $0) } ?? [:]
        if let latest {
            fields.merge(AIMEDiagnostics.metricFields(from: latest.values, prefix: "latest_")) { _, new in new }
        }
        fields.merge(AIMEDiagnostics.inFlightFields(inFlight)) { _, new in new }
        fields.merge(extra) { _, new in new }
        AIMEDiagnostics.record(name, fields: fields)
    }

    private static func decode<T: Decodable>(_ type: T.Type, from object: DynamicObject) throws -> T {
        let data = try JSONEncoder().encode(object.values)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func shouldPublishProgressFrame() -> Bool {
        let now = Date().timeIntervalSince1970
        let intervalMs = configuration.performanceLock
            ? 1000
            : max(100, configuration.streamSnapshotIntervalMs)
        let intervalS = Double(intervalMs) / 1000.0
        guard now - lastProgressPublishS >= intervalS else { return false }
        lastProgressPublishS = now
        return true
    }

    private static func hasActivePrefill(_ request: InFlightRequest) -> Bool {
        guard let phase = request.prefillState?.phase else { return false }
        return phase == "started" || phase == "chunk"
    }

    /// Headline decode rate for the hero gauge.
    ///
    /// The center gauge means "current decode TPS", not best, rolling,
    /// display-window, or cumulative average. Those other values have
    /// their own UI homes. Keeping this source to `decode_tok_s` prevents
    /// the gauge from bouncing between the current sample and an earlier
    /// high/peak-like value.
    private func headlineDecodeTPS(from latest: MetricsLatest?) -> Double? {
        guard let latest else { return nil }
        if let raw = latest.values["decode_tok_s"]?.doubleValue, raw.isFinite, raw > 0 {
            return raw
        }
        return nil
    }

    /// Push a new headline reading through the lifecycle state machine.
    /// `isCompletion == true` (from `.completed` events or post-request
    /// snapshots) promotes to `.held`; otherwise stays `.live`. Identical
    /// held values are dropped so repeated idle snapshots don't churn the
    /// publisher with a fresh `completedAt` every poll.
    private func updateHeadlineDecode(value: Double, isCompletion: Bool) {
        if isCompletion {
            if case .held(let held, _) = headlineDecode, abs(held - value) < 0.01 {
                return
            }
            headlineDecode = .held(value: value, completedAt: Date())
        } else {
            headlineDecode = .live(value)
        }
    }

    /// Observe the identity of the request the latest metrics describe.
    /// When it changes, reset all per-request live metrics so the next
    /// request starts clean (no stale acceptance bars, no carried cached
    /// count) and bump `metricsRequestGeneration` so views can clear
    /// their own held UI state. Single chokepoint for per-request reset.
    private func noteRequestIdentity(_ key: String?) {
        guard let key, !key.isEmpty, key != trackedRequestKey else { return }
        trackedRequestKey = key
        // New request: reset the per-request scalars (cached tokens,
        // verify count, decode-rate sample) and unfreeze — but KEEP the
        // acceptance arrays. The bars should keep showing the previous
        // request's values and smoothly EMA-transition into the new ones
        // once they arrive, instead of blanking/flickering on every send.
        var next = smoothedMetrics
        next.cachedTokens = nil
        next.verifyCalls = nil
        smoothedMetrics = next
        smoothedFrozen = false
        metricsRequestGeneration &+= 1
    }

    /// Update `smoothedMetrics` from the latest payload. Resets on a new
    /// request identity (via `noteRequestIdentity`), freezes after the
    /// request completes (so idle polls can't keep nudging the EMA), and
    /// — when `snapToFinal` is set, on the `.completed` envelope — writes
    /// the exact final values rather than an EMA step.
    private func updateSmoothedMetrics(_ latest: MetricsLatest, snapToFinal: Bool = false) {
        // Track identity by request_id ONLY. Falling back to session_id
        // made the key flip between progress frames (which carry
        // request_id) and dashboard snapshots (which may carry only
        // session_id), so the per-request reset fired spuriously every
        // frame and the acceptance bars flickered appear/disappear within
        // a single request.
        noteRequestIdentity(latest.values["request_id"]?.stringValue)
        // After completion the metrics are frozen at the final values
        // until the next request arrives. Snap-to-final is allowed
        // through because it *is* the completion write.
        if smoothedFrozen && !snapToFinal { return }

        var next = smoothedMetrics
        let alpha = snapToFinal ? 1.0 : smoothedAlpha

        if let verifyCalls = Self.doubleField(latest, "verify_calls") {
            next.verifyCalls = Self.ema(previous: next.verifyCalls,
                                         incoming: verifyCalls,
                                         alpha: alpha)
        }
        if let cachedTokens = Self.doubleField(latest, "cached_tokens"), cachedTokens > 0 {
            // Cached tokens is a per-request prefill constant, not a
            // noisy rate. EMA-ing it made it strobe between two values
            // whenever a snapshot and a progress frame disagreed (one
            // carries the count, the other carries 0). Hold the
            // per-request max instead — monotonic and flicker-proof.
            next.cachedTokens = max(next.cachedTokens ?? 0, cachedTokens)
        }

        let acceptanceRows = latest.acceptanceCounterRows()
        if !acceptanceRows.isEmpty {
            let rates = acceptanceRows.map(\.rate)
            next.acceptanceRateByDepth = snapToFinal
                ? rates
                : Self.emaArray(
                    previous: next.acceptanceRateByDepth,
                    incoming: rates,
                    alpha: alpha
                )
        }

        let means = Self.doubleArrayField(latest, "mean_accept_probability_by_depth")
        if !means.isEmpty {
            next.meanAcceptByDepth = snapToFinal
                ? means
                : Self.emaArray(
                    previous: next.meanAcceptByDepth,
                    incoming: means,
                    alpha: alpha
                )
        }

        if next != smoothedMetrics {
            smoothedMetrics = next
        }
    }

    private static func doubleField(_ latest: MetricsLatest, _ key: String) -> Double? {
        latest.values[key]?.doubleValue
    }

    private static func doubleArrayField(_ latest: MetricsLatest, _ key: String) -> [Double] {
        guard case let .array(items)? = latest.values[key] else { return [] }
        return items.compactMap { $0.doubleValue }
    }

    private static func ema(previous: Double?, incoming: Double, alpha: Double) -> Double {
        guard let previous else { return incoming }
        return previous + alpha * (incoming - previous)
    }

    private static func emaArray(
        previous: [Double],
        incoming: [Double],
        alpha: Double
    ) -> [Double] {
        if previous.isEmpty || previous.count != incoming.count {
            return incoming
        }
        return zip(previous, incoming).map { prev, next in
            prev + alpha * (next - prev)
        }
    }

    /// Merge an incoming progress payload into the existing latest
    /// values without dropping stable cross-frame fields. Incoming keys
    /// always win for their own slot, but any key not present in the
    /// incoming payload survives from the previous frame. This is what
    /// keeps `display_decode_tok_s` and the sliding decode window keys
    /// alive across progress frames that only carry raw `decode_tok_s`,
    /// so the gauge cascade does not flip between integer values.
    private static func mergeLatestValues(
        existing: [String: JSONValue],
        incoming: [String: JSONValue]
    ) -> [String: JSONValue] {
        var merged = existing
        // If the incoming payload represents a new request (different
        // `request_id` or `session_id`), drop the stale carryover so
        // smoothing does not pull from an old session.
        if let newRequestId = incoming["request_id"]?.stringValue,
           let oldRequestId = existing["request_id"]?.stringValue,
           newRequestId != oldRequestId {
            merged = [:]
        } else if let newSessionId = incoming["session_id"]?.stringValue,
                  let oldSessionId = existing["session_id"]?.stringValue,
                  newSessionId != oldSessionId {
            merged = [:]
        }
        for (key, value) in incoming {
            merged[key] = value
        }
        return merged
    }

    private func shouldAcceptSnapshotLatest(_ snapshot: DashboardSnapshot) -> Bool {
        guard let incoming = snapshot.latest else { return false }
        guard !snapshot.inFlight.isEmpty else { return true }
        guard let incomingRequestID = incoming.values["request_id"]?.stringValue,
              !incomingRequestID.isEmpty else {
            return false
        }
        return snapshot.inFlight.contains { $0.requestId == incomingRequestID }
    }

    private static func prefillPhase(in payload: DynamicObject) -> String? {
        if let phase = payload.values["phase"]?.stringValue {
            return phase
        }
        if let nested = payload.values["prefill"]?.objectValue {
            return nested["phase"]?.stringValue
        }
        return nil
    }

    private static func hasDecodeProgress(_ values: [String: JSONValue]) -> Bool {
        if let completion = values["completion_tokens"]?.doubleValue, completion > 0 {
            return true
        }
        if let generated = values["generated_tokens"]?.doubleValue, generated > 0 {
            return true
        }
        if let decodeElapsed = values["decode_elapsed_s"]?.doubleValue, decodeElapsed > 0 {
            return true
        }
        if let decodeTPS = values["decode_tok_s"]?.doubleValue, decodeTPS > 0 {
            return true
        }
        return false
    }

    private func verifiedFanMode(from health: HealthPayload) -> String? {
        if let fanMode = health.fanMode, !fanMode.isEmpty {
            return MTPLXFanMode.normalized(fanMode).rawValue
        }
        if health.thermal?.actualRampVerified == true {
            return MTPLXFanMode.max.rawValue
        }
        return nil
    }
}

private func defaultLaunchTarget(for configuration: MTPLXAppConfiguration) -> LaunchTarget? {
    if shouldPromoteStaleOpenCodeTarget(configuration) {
        return .openCode
    }
    return LaunchTarget(rawValue: configuration.lastLaunchTarget)
}

private func shouldPromoteStaleOpenCodeTarget(_ configuration: MTPLXAppConfiguration) -> Bool {
    configuration.lastLaunchTarget == LaunchTarget.chat.rawValue
        && configuration.port == 18083
        && configuration.schedulerMode == "ar_batch"
        && configuration.batchingPreset == "agent"
        && configuration.ssdSessionCache == "on"
}

private extension MutableSettings {
    var hasMutableLiveValue: Bool {
        generationMode != nil
            || depth != nil
            || temperature != nil
            || topP != nil
            || topK != nil
            || presencePenalty != nil
            || maxResponseTokens != nil
            || streamInterval != nil
            || enableThinking != nil
            || reasoningParser != nil
            || reasoning != nil
            || reasoningEffort != nil
            || prefillChunkTokens != nil
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }
}
