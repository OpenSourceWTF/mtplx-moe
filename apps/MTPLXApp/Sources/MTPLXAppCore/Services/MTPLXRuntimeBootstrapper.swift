import CryptoKit
import Foundation

public enum MTPLXRuntimeBootstrapperError: Error, LocalizedError, Sendable {
    case homebrewNotFound
    case pythonNotFound
    case commandFailed(command: String, exitCode: Int32, output: String)
    case runtimeStillMissing(output: String)

    public var errorDescription: String? {
        switch self {
        case .homebrewNotFound:
            return "Homebrew was not found, so MTPLX could not install its command-line runtime automatically. Install Homebrew from brew.sh, then press Retry."
        case .pythonNotFound:
            return "Python 3.11 or newer was not found, so MTPLX could not prepare its command-line runtime. Install Homebrew from brew.sh, then press Retry."
        case .commandFailed(let command, let exitCode, let output):
            let detail = output.trimmingCharacters(in: .whitespacesAndNewlines)
            if detail.isEmpty {
                return "\(command) failed with exit code \(exitCode)."
            }
            return "\(command) failed with exit code \(exitCode): \(detail)"
        case .runtimeStillMissing(let output):
            let detail = output.trimmingCharacters(in: .whitespacesAndNewlines)
            if detail.isEmpty {
                return "Homebrew finished, but MTPLX still was not available on PATH."
            }
            return "Homebrew finished, but MTPLX still was not available on PATH: \(detail)"
        }
    }
}

public struct MTPLXRuntimeBootstrapper: Sendable {
    public static let formula = "youssofal/mtplx/mtplx"

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.environment = environment
    }

    private let environment: [String: String]

    public func installOrUpdate(status: (@Sendable (String) -> Void)? = nil) throws -> URL {
        status?("Checking MTPLX runtime")
        let minimumVersion = minimumRuntimeVersion()
        let bundledWheel = MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment)
        // When this bundle ships a wheel, the engine is always the
        // app-owned venv. A user-managed mtplx on PATH (pip user-site,
        // old source install) that happens to satisfy the version
        // floor must never be adopted as the engine: its Python and
        // dependency state are unknown, and on first run — before the
        // app venv exists — it would otherwise win the PATH walk and
        // every later model load runs in an environment we never
        // installed. PATH installs remain the user's; onboarding's
        // CLI row reports them separately.
        if let existing = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment),
           runtime(existing, satisfies: minimumVersion),
           bundledWheel == nil || isAppManagedRuntime(existing),
           installedRuntimeMatchesBundledWheel(installedExecutable: existing) {
            return existing
        }
        if let wheel = bundledWheel {
            status?("Installing MTPLX runtime")
            return try installBundledRuntime(wheel: URL(fileURLWithPath: wheel))
        }
        if let existing = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment),
           minimumVersion == nil {
            return existing
        }
        status?("Installing MTPLX runtime")
        return try installHomebrewRuntime()
    }

    /// Whether `installedExecutable` can be reused as-is for this app
    /// bundle.
    ///
    /// The version floor alone cannot see same-version rebuilds:
    /// 1.0.0 build N and build N+1 ship different wheels under one
    /// semantic version, so after an auto-update the app-managed venv
    /// would silently keep serving the old code forever. For the venv
    /// the app installed itself, the install-time fingerprint marker
    /// must match the wheel this bundle ships; runtimes the app does
    /// not manage (Homebrew/system installs) keep the version-floor
    /// contract unchanged.
    func installedRuntimeMatchesBundledWheel(installedExecutable: URL) -> Bool {
        guard let wheelPath = MTPLXCommandBuilder.bundledRuntimeWheelPath(
            environment: environment
        ) else {
            return true
        }
        let runtimeDir = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment)
        )
        guard isAppManagedRuntime(installedExecutable) else {
            return true
        }
        guard let bundled = try? Self.wheelFingerprint(
            of: URL(fileURLWithPath: wheelPath)
        ) else {
            return true
        }
        return bundled == Self.recordedWheelFingerprint(runtimeDir: runtimeDir)
    }

    /// Whether `executable` resolves into the app-owned runtime venv.
    func isAppManagedRuntime(_ executable: URL) -> Bool {
        let managed = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment)
        )
        .appendingPathComponent("bin")
        .appendingPathComponent("mtplx")
        .resolvingSymlinksInPath()
        return executable.resolvingSymlinksInPath().path == managed.path
    }

    /// SHA-256 of the wheel file, hex-encoded.
    static func wheelFingerprint(of wheel: URL) throws -> String {
        let data = try Data(contentsOf: wheel, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Marker recording which bundled wheel the app-managed venv was
    /// installed from. It lives inside the venv so it travels and dies
    /// with the install it describes.
    static func wheelFingerprintMarkerURL(runtimeDir: URL) -> URL {
        runtimeDir.appendingPathComponent("bundled-wheel.sha256")
    }

    static func recordedWheelFingerprint(runtimeDir: URL) -> String? {
        guard let raw = try? String(
            contentsOf: wheelFingerprintMarkerURL(runtimeDir: runtimeDir),
            encoding: .utf8
        ) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Best-effort: a missing marker only costs one redundant
    /// reinstall on the next launch, while failing the install over it
    /// would cost a working runtime.
    static func recordWheelFingerprint(for wheel: URL, runtimeDir: URL) {
        guard let fingerprint = try? wheelFingerprint(of: wheel) else { return }
        try? fingerprint.write(
            to: wheelFingerprintMarkerURL(runtimeDir: runtimeDir),
            atomically: true,
            encoding: .utf8
        )
    }

    public func upgradeHomebrewRuntime() throws -> URL {
        guard MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) != nil else {
            throw MTPLXRuntimeBootstrapperError.homebrewNotFound
        }
        return try runHomebrewInstallSequence(allowExistingRuntime: true)
    }

    private func installHomebrewRuntime() throws -> URL {
        guard let brew = MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) else {
            throw MTPLXRuntimeBootstrapperError.homebrewNotFound
        }
        _ = brew
        return try runHomebrewInstallSequence(allowExistingRuntime: false)
    }

    private func runHomebrewInstallSequence(allowExistingRuntime: Bool) throws -> URL {
        guard let brew = MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) else {
            throw MTPLXRuntimeBootstrapperError.homebrewNotFound
        }

        var lastOutput = ""
        lastOutput = try run(brew: brew, arguments: ["update"])
        if allowExistingRuntime {
            if let upgradeOutput = try? run(brew: brew, arguments: ["upgrade", Self.formula]) {
                lastOutput = upgradeOutput
            } else {
                lastOutput = try run(brew: brew, arguments: ["install", Self.formula])
            }
        } else {
            lastOutput = try run(brew: brew, arguments: ["install", Self.formula])
        }
        if let upgradeOutput = try? run(brew: brew, arguments: ["upgrade", Self.formula]) {
            lastOutput = upgradeOutput
        }
        if let linkOutput = try? run(brew: brew, arguments: ["link", "--overwrite", "mtplx"]) {
            lastOutput = linkOutput
        }

        let minimumVersion = minimumRuntimeVersion()
        if let resolved = try? resolvedInstalledRuntime(minimumVersion: minimumVersion, output: lastOutput) {
            return resolved
        }

        if let unlinkOutput = try? run(brew: brew, arguments: ["unlink", "mtplx"]) {
            lastOutput = unlinkOutput
        }
        lastOutput = try run(brew: brew, arguments: ["link", "--overwrite", "mtplx"])
        if let resolved = try? resolvedInstalledRuntime(minimumVersion: minimumVersion, output: lastOutput) {
            return resolved
        }
        throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(output: lastOutput)
    }

    private func run(brew: URL, arguments: [String]) throws -> String {
        try run(
            executable: brew,
            arguments: arguments,
            displayCommand: brewCommand(arguments)
        )
    }

    private func run(
        executable: URL,
        arguments: [String],
        displayCommand: String
    ) throws -> String {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = Self.hermeticSubprocessEnvironment(from: environment)

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        let output = RuntimeInstallTailBuffer(capacity: 4096)
        stdout.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { output.append(chunk) }
        }
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { output.append(chunk) }
        }
        defer {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
        }

        do {
            try process.run()
        } catch {
            throw MTPLXRuntimeBootstrapperError.commandFailed(
                command: displayCommand,
                exitCode: -1,
                output: error.localizedDescription
            )
        }
        process.waitUntilExit()
        let tail = output.snapshot()
        guard process.terminationStatus == 0 else {
            throw MTPLXRuntimeBootstrapperError.commandFailed(
                command: displayCommand,
                exitCode: process.terminationStatus,
                output: tail
            )
        }
        return tail
    }

    private func brewCommand(_ arguments: [String]) -> String {
        ("brew " + arguments.joined(separator: " "))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Environment for every bootstrap subprocess. Beyond the standard
    /// app filtering, pip must not see the user's pip configuration at
    /// all: pip reads ~/.config/pip/pip.conf (and friends) from disk,
    /// which no env blocklist can reach, and a common `user = true`
    /// there aborts every venv install with "Can not perform a
    /// '--user' install. User site-packages are not visible in this
    /// virtualenv." Pointing PIP_CONFIG_FILE at /dev/null disables
    /// config-file loading — the same technique CPython's own
    /// ensurepip uses — and PIP_USER=0 pins user installs off at the
    /// env layer, which outranks any config file pip might still find.
    static func hermeticSubprocessEnvironment(
        from environment: [String: String]
    ) -> [String: String] {
        var env = MTPLXCommandBuilder.appSubprocessEnvironment(environment: environment)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_CONFIG_FILE"] = "/dev/null"
        env["PIP_USER"] = "0"
        return env
    }

    private func installBundledRuntime(wheel: URL) throws -> URL {
        let runtimeDir = URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment))
        try FileManager.default.createDirectory(
            at: runtimeDir.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let python = try resolvePythonExecutable()
        try createRuntimeVenv(python: python, runtimeDir: runtimeDir)
        let venvPython = runtimeDir.appendingPathComponent("bin").appendingPathComponent("python")
        // Best effort: the venv's ensurepip pip is already new enough
        // to install the bundled wheel, so a PyPI hiccup or blocked
        // network here must not fail first-run setup — the wheel
        // install below is the step that actually gates readiness.
        _ = try? run(
            executable: venvPython,
            arguments: ["-m", "pip", "install", "-U", "pip"],
            displayCommand: "runtime python -m pip install -U pip"
        )
        _ = try run(
            executable: venvPython,
            arguments: ["-m", "pip", "install", "-U", "\(wheel.path)[server]"],
            displayCommand: "runtime python -m pip install -U bundled MTPLX"
        )
        // pip skips a wheel whose version matches the installed one,
        // so -U alone is a no-op for same-version rebuilds — exactly
        // the case the fingerprint marker exists to catch. Force the
        // package itself back to this bundle's bytes; dependencies are
        // already satisfied by the install above, so this step needs
        // no network and unpacks one wheel.
        _ = try run(
            executable: venvPython,
            arguments: ["-m", "pip", "install", "--force-reinstall", "--no-deps", wheel.path],
            displayCommand: "runtime python -m pip install --force-reinstall bundled MTPLX"
        )

        let executable = runtimeDir.appendingPathComponent("bin").appendingPathComponent("mtplx")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "Bundled runtime install finished, but \(executable.path) was not created."
            )
        }
        let minimumVersion = minimumRuntimeVersion()
        guard runtime(executable, satisfies: minimumVersion) else {
            let observed = MTPLXRuntimeUpdateService.runtimeVersion(
                executableURL: executable,
                environment: environment
            ) ?? "unknown"
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "Bundled runtime installed \(observed), but \(minimumVersion?.description ?? "the required version") is required."
            )
        }
        Self.recordWheelFingerprint(for: wheel, runtimeDir: runtimeDir)
        return executable
    }

    /// Create (or repair) the app-owned runtime venv.
    ///
    /// The venv's `bin/python3` is a symlink chain into the interpreter
    /// it was built from — usually the one shipped inside the app
    /// bundle. After an app update replaces the bundle, that chain can
    /// dangle, and a plain `python -m venv` over the corpse exits 1
    /// with "[Errno 2] No such file or directory: …/bin/python3"
    /// (issue #139). Reinstalling the app cannot fix it because the
    /// venv lives in Application Support, outside the bundle. So:
    /// rebuild with `--clear` when the existing venv python is broken,
    /// and retry once with `--clear` on any other creation failure.
    /// A healthy venv keeps the plain no-clear path so same-venv
    /// updates reuse installed dependencies (fast and offline-safe).
    func createRuntimeVenv(python: URL, runtimeDir: URL) throws {
        let fileManager = FileManager.default
        // Probe bin/python — the executable the install steps below
        // actually invoke. isExecutableFile resolves symlinks, so a
        // dangling chain reads as not-executable — exactly the broken
        // state.
        let venvPython = runtimeDir
            .appendingPathComponent("bin")
            .appendingPathComponent("python")
        let venvBroken = fileManager.fileExists(atPath: runtimeDir.path)
            && !fileManager.isExecutableFile(atPath: venvPython.path)
        if venvBroken {
            _ = try run(
                executable: python,
                arguments: ["-m", "venv", "--clear", runtimeDir.path],
                displayCommand: "python -m venv --clear \(runtimeDir.path)"
            )
            return
        }
        do {
            _ = try run(
                executable: python,
                arguments: ["-m", "venv", runtimeDir.path],
                displayCommand: "python -m venv \(runtimeDir.path)"
            )
        } catch {
            _ = try run(
                executable: python,
                arguments: ["-m", "venv", "--clear", runtimeDir.path],
                displayCommand: "python -m venv --clear \(runtimeDir.path)"
            )
        }
    }

    private func resolvedInstalledRuntime(
        minimumVersion: MTPLXSemanticVersion?,
        output: String
    ) throws -> URL {
        guard let resolved = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment) else {
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(output: output)
        }
        guard runtime(resolved, satisfies: minimumVersion) else {
            let observed = MTPLXRuntimeUpdateService.runtimeVersion(
                executableURL: resolved,
                environment: environment
            ) ?? "unknown"
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "\(resolved.path) is \(observed), but \(minimumVersion?.description ?? "the required version") is required.\n\(output)"
            )
        }
        return resolved
    }

    private func minimumRuntimeVersion() -> MTPLXSemanticVersion? {
        if let raw = environment["MTPLX_APP_REQUIRED_RUNTIME_VERSION"],
           let version = MTPLXSemanticVersion(raw) {
            return version
        }
        if Bundle.main.bundleURL.pathExtension == "app",
           let raw = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
           let version = MTPLXSemanticVersion(raw) {
            return version
        }
        return nil
    }

    private func runtime(_ executable: URL, satisfies minimumVersion: MTPLXSemanticVersion?) -> Bool {
        guard let minimumVersion else { return true }
        guard let raw = MTPLXRuntimeUpdateService.runtimeVersion(
            executableURL: executable,
            environment: environment
        ),
            let current = MTPLXSemanticVersion(raw)
        else { return false }
        return current >= minimumVersion
    }

    func resolvePythonExecutable() throws -> URL {
        if let explicit = environment["MTPLX_APP_PYTHON_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty,
           FileManager.default.isExecutableFile(atPath: explicit),
           pythonVersionOK(URL(fileURLWithPath: explicit)) {
            return URL(fileURLWithPath: explicit)
        }

        // The interpreter shipped in Contents/Resources/PythonRuntime wins
        // over anything on the system: it is version-pinned, signed with
        // the app, and exists on Macs with no Homebrew or Xcode at all. A
        // venv built from it self-heals after app moves/updates via the
        // existing version-floor reinstall.
        if let bundled = MTPLXCommandBuilder.bundledPythonExecutablePath(
            environment: environment
        ) {
            let url = URL(fileURLWithPath: bundled)
            if pythonVersionOK(url) {
                return url
            }
        }

        let names = ["python3.14", "python3.13", "python3.12", "python3.11", "python3"]
        let fixedPaths = [
            "/opt/homebrew/bin/python3.14",
            "/opt/homebrew/bin/python3.13",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/usr/local/bin/python3.14",
            "/usr/local/bin/python3.13",
            "/usr/local/bin/python3.12",
            "/usr/local/bin/python3.11",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
        ]
        for path in fixedPaths {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.isExecutableFile(atPath: path), pythonVersionOK(url) {
                return url
            }
        }
        for name in names {
            for directory in MTPLXCommandBuilder.expandedPATH(environment: environment).split(separator: ":").map(String.init) {
                let url = URL(fileURLWithPath: directory).appendingPathComponent(name)
                if FileManager.default.isExecutableFile(atPath: url.path), pythonVersionOK(url) {
                    return url
                }
            }
        }
        throw MTPLXRuntimeBootstrapperError.pythonNotFound
    }

    private func pythonVersionOK(_ executable: URL) -> Bool {
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--version"]
        process.environment = MTPLXCommandBuilder.appSubprocessEnvironment(environment: environment)
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        do {
            try process.run()
        } catch {
            return false
        }
        process.waitUntilExit()
        var data = stdout.fileHandleForReading.readDataToEndOfFile()
        data.append(stderr.fileHandleForReading.readDataToEndOfFile())
        let output = String(data: data, encoding: .utf8) ?? ""
        guard let version = MTPLXSemanticVersion(output) else { return false }
        return version >= MTPLXSemanticVersion("3.11")!
    }
}

private final class RuntimeInstallTailBuffer: @unchecked Sendable {
    private let capacity: Int
    private let lock = NSLock()
    private var data = Data()

    init(capacity: Int) {
        self.capacity = max(256, capacity)
    }

    func append(_ chunk: Data) {
        lock.lock()
        data.append(chunk)
        if data.count > capacity {
            data.removeFirst(data.count - capacity)
        }
        lock.unlock()
    }

    func snapshot() -> String {
        lock.lock()
        let copy = data
        lock.unlock()
        return String(data: copy, encoding: .utf8) ?? ""
    }
}
