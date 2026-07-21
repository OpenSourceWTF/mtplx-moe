import Darwin
import Foundation

// MARK: - HardwareInspector
//
// Detects the user's Mac chip, RAM, GPU cores, and machine model so
// onboarding can:
//   1. Show "Apple M5 Max · 36 GB · 30 GPU Cores" on the scan step.
//   2. Route M1/M2 to the FP16 Speed variant automatically.
//   3. Gate the model-pick step with memory feasibility verdicts.
//
// Primary path: spawn `mtplx hardware inspect --json` and decode the
// schema documented at `mtplx/hardware.py:186-215`. That subprocess
// already does the heavy lifting (sysctl + system_profiler + Apple-
// silicon-generation classification) and is the daemon's own source
// of truth — using it keeps the app's view of the hardware identical
// to what `mtplx serve` will see at runtime.
//
// Fallback: if the CLI is missing or fails, derive `chipName` and
// `unifiedMemoryBytes` from `sysctlbyname` directly. This loses GPU
// core count and machine-model identifier but preserves the only two
// fields the model-pick step strictly needs to function (chip family
// for FP16 routing, RAM for feasibility).

public struct HardwareInspector: Sendable {
    private let processEnvironment: [String: String]
    private let runner: @Sendable (URL, [String]) async throws -> (Int32, Data, Data)

    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        runner: (@Sendable (URL, [String]) async throws -> (Int32, Data, Data))? = nil
    ) {
        self.processEnvironment = processEnvironment
        self.runner = runner ?? { executable, arguments in
            try await Self.runSubprocess(
                executable: executable,
                arguments: arguments,
                environment: processEnvironment
            )
        }
    }

    public enum InspectorError: Error, Sendable {
        case executableNotFound
        case subprocessFailed(exitCode: Int32, stderr: String)
        case decodeFailed(String)
    }

    /// Returns the user's hardware. Never throws — falls back to a
    /// sysctl-derived best effort and logs the original CLI error
    /// silently. The reason is product-driven: onboarding must not
    /// fail closed on an `mtplx hardware inspect` regression because
    /// the user's chip family / RAM is required to pick a model at all.
    public func detect() async -> DetectedHardware {
        if let cliResult = try? await detectViaCLI() {
            return cliResult
        }
        return detectViaSysctl()
    }

    // MARK: - CLI path

    private func detectViaCLI() async throws -> DetectedHardware {
        let executable = try resolveMtplxExecutable()
        let (exitCode, stdout, _) = try await runner(executable, ["hardware", "inspect", "--json"])
        guard exitCode == 0 else {
            throw InspectorError.subprocessFailed(
                exitCode: exitCode,
                stderr: "exit=\(exitCode)"
            )
        }
        guard
            let raw = try? JSONSerialization.jsonObject(with: stdout) as? [String: Any]
        else {
            throw InspectorError.decodeFailed("invalid JSON")
        }
        let chipName = (raw["chip"] as? String) ?? "Apple Silicon"
        let appleSiliconGeneration = raw["apple_silicon_generation"] as? String
        let modelIdentifier = raw["model_identifier"] as? String
        let unifiedMemoryBytes = (raw["unified_memory_bytes"] as? Int64)
            ?? Int64((raw["unified_memory_bytes"] as? Int) ?? 0)
        let gpuCoreCount = raw["gpu_cores"] as? Int
        let cpuCoreCount = raw["cpu_cores"] as? Int
            ?? raw["logical_cpu_cores"] as? Int
            ?? raw["physical_cpu_cores"] as? Int
        return DetectedHardware(
            chipName: chipName,
            appleSiliconGeneration: appleSiliconGeneration,
            modelIdentifier: modelIdentifier?.isEmpty == false ? modelIdentifier : nil,
            unifiedMemoryBytes: unifiedMemoryBytes > 0
                ? unifiedMemoryBytes
                : Int64(ProcessInfo.processInfo.physicalMemory),
            gpuCoreCount: gpuCoreCount,
            cpuCoreCount: cpuCoreCount
        )
    }

    // MARK: - sysctl fallback

    /// Last-resort path: read `machdep.cpu.brand_string` and
    /// `ProcessInfo.physicalMemory`. Loses GPU cores and machine
    /// identifier but is sufficient for chip-tier classification and
    /// memory feasibility.
    private func detectViaSysctl() -> DetectedHardware {
        let chipName = Self.sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"
        let generation = Self.parseAppleSiliconGeneration(from: chipName)
        return DetectedHardware(
            chipName: chipName,
            appleSiliconGeneration: generation,
            modelIdentifier: Self.sysctlString("hw.model"),
            unifiedMemoryBytes: Int64(ProcessInfo.processInfo.physicalMemory),
            gpuCoreCount: nil,
            cpuCoreCount: ProcessInfo.processInfo.activeProcessorCount
        )
    }

    /// FP16 beats BF16 for prompt processing on chips without native BF16
    /// (M1/M2). Forge's Auto precision label uses this; the engine
    /// re-resolves independently at build time from the same signal.
    public static func hostPrefersFP16Forge() -> Bool {
        let chip = sysctlString("machdep.cpu.brand_string") ?? ""
        guard let generation = parseAppleSiliconGeneration(from: chip) else { return false }
        return generation == "m1" || generation == "m2"
    }

    private static func parseAppleSiliconGeneration(from chip: String) -> String? {
        let lower = chip.lowercased()
        if lower.range(of: #"\bm1\b"#, options: .regularExpression) != nil { return "m1" }
        if lower.range(of: #"\bm2\b"#, options: .regularExpression) != nil { return "m2" }
        if lower.range(of: #"\bm3\b"#, options: .regularExpression) != nil { return "m3" }
        if lower.range(of: #"\bm4\b"#, options: .regularExpression) != nil { return "m4" }
        if lower.range(of: #"\bm5\b"#, options: .regularExpression) != nil { return "m5" }
        return nil
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }

    // MARK: - Executable resolution
    //
    // Mirrors the daemon launcher so the hardware probe shells the same
    // installed `mtplx` binary the app will later launch. If that binary is
    // missing, `detect()` falls back to sysctl-derived chip and RAM values.

    private func resolveMtplxExecutable() throws -> URL {
        do {
            return try MTPLXCommandBuilder.resolveInstalledExecutable(
                environment: processEnvironment
            )
        } catch {
            throw InspectorError.executableNotFound
        }
    }

    // MARK: - Default subprocess runner

    @Sendable
    public static func defaultRunner(
        executable: URL,
        arguments: [String]
    ) async throws -> (Int32, Data, Data) {
        try await runSubprocess(
            executable: executable,
            arguments: arguments,
            environment: ProcessInfo.processInfo.environment
        )
    }

    private static func runSubprocess(
        executable: URL,
        arguments: [String],
        environment: [String: String]
    ) async throws -> (Int32, Data, Data) {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = executable
                process.arguments = arguments
                process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
                    environment: environment
                )
                let outPipe = Pipe()
                let errPipe = Pipe()
                process.standardOutput = outPipe
                process.standardError = errPipe
                let watchdog = SubprocessWatchdog(process)
                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                // Drain as data arrives and bound the wait (#158): the
                // old read-after-exit could deadlock on a chatty child,
                // and a wedged CLI (Gatekeeper-stalled first exec)
                // parked onboarding forever. A timeout throws, which
                // detect() already degrades to the sysctl fallback.
                let stdoutDrain = SubprocessPipeDrain(outPipe)
                let stderrDrain = SubprocessPipeDrain(errPipe)
                let timeout: TimeInterval = 60
                guard watchdog.wait(for: process, timeout: timeout) else {
                    continuation.resume(throwing: InspectorError.subprocessFailed(
                        exitCode: -2,
                        stderr: "timed out after \(Int(timeout))s and was terminated"
                    ))
                    return
                }
                stdoutDrain.join()
                stderrDrain.join()
                continuation.resume(returning: (
                    process.terminationStatus,
                    stdoutDrain.snapshotData(),
                    stderrDrain.snapshotData()
                ))
            }
        }
    }
}
