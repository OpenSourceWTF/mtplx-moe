import Foundation

// MARK: - ForgeStep
//
// Canonical wizard order for the MTP Forge Create flow. `goNext` /
// `goBack` walk the case array in declaration order so the enum
// doubles as the progress-capsule's source of truth.
//
// The first six steps are the linear pipeline (Source → Plan →
// Convert → Calibrate → Verify → Brand). `registered` is the
// celebration card with three CTAs (Use it now / Publish / Build
// another). `publishing` is reachable only from `registered` when
// the user clicks Publish; going back from `publishing` returns to
// `registered`, going forward exits the wizard.

public enum ForgeStep: String, CaseIterable, Equatable, Sendable {
    case source
    case plan
    case convert
    case calibrate
    case verify
    case brand
    case registered
    case publishing
}

// MARK: - ForgeSubMode
//
// Top-of-tab segmented selection. Independent of `ForgeStep`; the
// user can switch sub-modes any time (mid-build the orchestrator
// surfaces a "build in progress — stay on Create or pause" warning).

public enum ForgeSubMode: String, CaseIterable, Equatable, Sendable {
    case create
    case discover
    case mine
}

// MARK: - ForgeSourceFormat
//
// What the probe detected at the source repo. Drives the recipe
// auto-pick on the Plan stage and the build-pipeline routing on the
// Convert stage. Cases:
//
// - `bf16Native`: stock HF safetensors with BF16 MTP head. Cleanest
//   path — Forge body-quantizes + repacks the MTP sidecar untouched.
// - `mlxAffine`: already MLX-affine quantized body. Skip body
//   conversion; may still need MTP sidecar extraction/calibration.
// - `mlxAffineWithMtp`: full MLX-affine artifact with the MTP sidecar
//   already published. Verify-only path.
// - `compressedTensorsAwq`: cyankiwi-style AWQ-4bit using vLLM's
//   compressed-tensors layout. Needs the AWQ → MLX-affine converter
//   the Python agent is writing; non-trivial.
// - `hfVllm`: any other HF model that vLLM/SGLang can serve. May or
//   may not have MTP — probe reports separately.
// - `unknown`: probe completed but couldn't classify; surface as a
//   warning in PlanStage, let the user override or abort.

public enum ForgeSourceFormat: String, Equatable, Sendable, Codable {
    case bf16Native = "bf16_native"
    case mlxAffine = "mlx_affine"
    case mlxAffineWithMtp = "mlx_affine_with_mtp"
    case compressedTensorsAwq = "compressed_tensors_awq"
    case hfVllm = "hf_vllm"
    case unknown
}

// MARK: - ForgeRecipe
//
// The knobs the user can (optionally) override on the Plan stage.
// Defaults are auto-picked by `ForgeRecipe.defaultFor(format:)` and
// rendered read-only unless the user opens the "Advanced" disclosure.
//
// MTP policy default is `.keepBf16` — mlx-lm PR #990 review evidence
// shows quantizing MTP weights collapses MoE acceptance to 5-11%.
// `.requantize` is allowed but gated by the user explicitly
// acknowledging the warning in PlanStage, which sets
// `hasAcknowledgedDegradedMTP = true` on the orchestrator.

public struct ForgeRecipe: Equatable, Sendable, Codable {
    public enum MTPPolicy: String, Equatable, Sendable, Codable {
        case keepBf16 = "keep_bf16"
        case extractFromSidecar = "extract_from_sidecar"
        case requantize = "requantize"
    }

    public enum QuantMode: String, Equatable, Sendable, Codable {
        case affine
    }

    /// Dtype of the non-quantized parameters (#166). `auto` is resolved by
    /// the engine at build time: FP16 on M1/M2 (no native BF16, faster
    /// prompt processing), BF16 on M3 and newer. Engines that predate the
    /// key ignore it and keep today's BF16 behaviour.
    public enum BodyDtype: String, Equatable, Sendable, Codable {
        case auto
        case bf16
        case fp16
    }

    public var bodyBits: Int
    public var bodyGroupSize: Int
    public var bodyMode: QuantMode
    public var mtpPolicy: MTPPolicy
    public var bodyDtype: BodyDtype

    public init(
        bodyBits: Int = 4,
        bodyGroupSize: Int = 64,
        bodyMode: QuantMode = .affine,
        mtpPolicy: MTPPolicy = .keepBf16,
        bodyDtype: BodyDtype = .auto
    ) {
        self.bodyBits = bodyBits
        self.bodyGroupSize = bodyGroupSize
        self.bodyMode = bodyMode
        self.mtpPolicy = mtpPolicy
        self.bodyDtype = bodyDtype
    }

    enum CodingKeys: String, CodingKey {
        case bodyBits = "body_bits"
        case bodyGroupSize = "body_group_size"
        case bodyMode = "body_mode"
        case mtpPolicy = "mtp_policy"
        case bodyDtype = "body_dtype"
    }

    // Persisted recipes from builds that predate body_dtype must keep
    // decoding; they resolve to `auto` like a fresh recipe would.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        bodyBits = try container.decodeIfPresent(Int.self, forKey: .bodyBits) ?? 4
        bodyGroupSize = try container.decodeIfPresent(Int.self, forKey: .bodyGroupSize) ?? 64
        bodyMode = try container.decodeIfPresent(QuantMode.self, forKey: .bodyMode) ?? .affine
        mtpPolicy = try container.decodeIfPresent(MTPPolicy.self, forKey: .mtpPolicy) ?? .keepBf16
        bodyDtype = try container.decodeIfPresent(BodyDtype.self, forKey: .bodyDtype) ?? .auto
    }

    /// Sensible default picked based on the detected source format.
    /// PlanStage shows this read-only unless the user expands the
    /// Advanced disclosure.
    public static func defaultFor(format: ForgeSourceFormat) -> ForgeRecipe {
        switch format {
        case .bf16Native, .hfVllm, .unknown:
            // 4-bit g64 affine matches the existing
            // Qwen3.6-27B-MTPLX-Flat4 recipe; conservative and known
            // to land speed-equivalent on the existing flagships.
            return ForgeRecipe(bodyBits: 4, bodyGroupSize: 64, bodyMode: .affine, mtpPolicy: .keepBf16)
        case .mlxAffine:
            // Body is already quantized; only MTP work needed. Pick
            // the bits the source artifact reports — defaulting to 4
            // is a safe fall-back for the schema, the Convert stage
            // will pass through unchanged when the source bits match.
            return ForgeRecipe(bodyBits: 4, bodyGroupSize: 64, bodyMode: .affine, mtpPolicy: .extractFromSidecar)
        case .mlxAffineWithMtp:
            // Verify-only. Recipe is moot but kept for the
            // mtplx_runtime.json provenance stamp.
            return ForgeRecipe(bodyBits: 4, bodyGroupSize: 64, bodyMode: .affine, mtpPolicy: .extractFromSidecar)
        case .compressedTensorsAwq:
            // AWQ is intrinsically group-128 4-bit; we requantize the
            // body to MLX-affine 4-bit g64 to match the runtime path.
            // The MTP sidecar is extracted as-is.
            return ForgeRecipe(bodyBits: 4, bodyGroupSize: 64, bodyMode: .affine, mtpPolicy: .extractFromSidecar)
        }
    }

    /// Whether this recipe would degrade MTP acceptance (per mlx-lm
    /// PR #990 evidence). PlanStage uses this to drive the warning
    /// chip + override gate.
    public var degradesMtp: Bool {
        mtpPolicy == .requantize
    }
}

// MARK: - ForgeSourceProbe
//
// Outcome of probing a source HF repo before downloading multi-GB
// weights. Extends the onboarding `OtherModelProbe` semantics with
// forge-specific signal: source format detection, whether MTP weights
// are present, and whether the repo is already an MTPLX-branded
// model (which short-circuits to "install instead").

public struct ForgeSourceProbe: Equatable, Sendable {
    public enum Verdict: String, Equatable, Sendable {
        /// Ready to forge — source format detected, MTP present, all
        /// gates clear.
        case forgeable
        /// Already an MTPLX-branded artifact — the user should
        /// install it, not rebuild it. SourceStage offers an
        /// "Install instead" CTA that hands off to ModelDownloader.
        case alreadyMTPLX
        /// Architecture has no MTP heads at all. Refuse to forge —
        /// the resulting artifact would have nothing to speculate
        /// with.
        case noMtpHeads
        /// Probe failed (network / 404 / 401 / malformed config).
        case probeFailed
    }

    public var verdict: Verdict
    public var hfRepo: String
    public var sourceFormat: ForgeSourceFormat
    public var hasMtpWeights: Bool
    public var estimatedSizeBytes: Int64?
    public var estimatedPeakGiB: Double?
    /// One-sentence explanation rendered under the result strip.
    public var message: String
    /// Surfaced when the probe failed (network error / status code).
    public var diagnostic: String?

    public init(
        verdict: Verdict,
        hfRepo: String,
        sourceFormat: ForgeSourceFormat = .unknown,
        hasMtpWeights: Bool = false,
        estimatedSizeBytes: Int64? = nil,
        estimatedPeakGiB: Double? = nil,
        message: String,
        diagnostic: String? = nil
    ) {
        self.verdict = verdict
        self.hfRepo = hfRepo
        self.sourceFormat = sourceFormat
        self.hasMtpWeights = hasMtpWeights
        self.estimatedSizeBytes = estimatedSizeBytes
        self.estimatedPeakGiB = estimatedPeakGiB
        self.message = message
        self.diagnostic = diagnostic
    }
}

// MARK: - ForgeVerification
//
// Recorded after VerifyStage's `mtplx tune` run completes. Drives
// the BrandStage's runtime-metadata preview and ultimately lands in
// the published `mtplx_runtime.json`'s existing `speed_evidence`
// block (NOT in the new `forge_provenance` block — verification
// data already has a home in the spine).

public struct ForgeVerification: Equatable, Sendable {
    public var arTokS: Double
    /// Map of depth → measured tok/s. Always includes the depth that
    /// was picked as the recommended product depth.
    public var tokSByDepth: [Int: Double]
    /// Per-depth acceptance fractions (length == depth for each entry).
    public var acceptanceByDepth: [Int: [Double]]
    public var bestDepth: Int
    public var multiplierVsAr: Double
    public var verifiedOnHardware: String
    public var sampler: ForgeSampler

    public init(
        arTokS: Double,
        tokSByDepth: [Int: Double],
        acceptanceByDepth: [Int: [Double]],
        bestDepth: Int,
        multiplierVsAr: Double,
        verifiedOnHardware: String,
        sampler: ForgeSampler
    ) {
        self.arTokS = arTokS
        self.tokSByDepth = tokSByDepth
        self.acceptanceByDepth = acceptanceByDepth
        self.bestDepth = bestDepth
        self.multiplierVsAr = multiplierVsAr
        self.verifiedOnHardware = verifiedOnHardware
        self.sampler = sampler
    }

    /// Extracts the product-facing verification winner from the
    /// stamped `mtplx_runtime.json` payload. Forge writes the canonical
    /// winning depth into `speed_evidence.depth`; `mtp_depth_max` only
    /// says what the backend can run and must not be used as the UX
    /// recommendation.
    public static func fromRuntimeMetadata(_ runtimeMeta: [String: Any]) -> ForgeVerification? {
        guard let evidence = runtimeMeta["speed_evidence"] as? [String: Any] else { return nil }
        let rows = evidence["forge_verify_rows"] as? [[String: Any]] ?? []
        let arTokS = rows.compactMap { row -> Double? in
            guard (row["depth"] as? Int) == 0 else { return nil }
            return row["tok_s"] as? Double
        }.first ?? ((evidence["greedy_diagnostic"] as? [String: Any])?["tok_s"] as? Double) ?? 0
        var tokSByDepth: [Int: Double] = [:]
        var acceptanceByDepth: [Int: [Double]] = [:]
        for row in rows {
            guard let depth = row["depth"] as? Int, depth > 0 else { continue }
            if let tokS = row["tok_s"] as? Double {
                tokSByDepth[depth] = tokS
            }
            if let acceptance = row["acceptance_by_position"] as? [Double] {
                acceptanceByDepth[depth] = acceptance
            }
        }
        if tokSByDepth.isEmpty,
           let depth = evidence["depth"] as? Int,
           let bestTokS = (evidence["tok_s"] as? [Double])?.max()
        {
            tokSByDepth[depth] = bestTokS
            acceptanceByDepth[depth] = (evidence["acceptance_by_depth"] as? [Double]) ?? []
        }
        let bestDepth = (evidence["depth"] as? Int)
            ?? tokSByDepth.max(by: { $0.value < $1.value })?.key
            ?? 0
        let bestTokS = tokSByDepth[bestDepth] ?? tokSByDepth.values.max() ?? 0
        let multiplier = arTokS > 0 ? bestTokS / arTokS : 1.0
        let sampler = MTPLXForgeProvenance.parseSampler(runtimeMeta["sampler"])
        let hardware = ((runtimeMeta["verified_on"] as? [String: Any])?["hardware"] as? String) ?? "Apple Silicon"
        return ForgeVerification(
            arTokS: arTokS,
            tokSByDepth: tokSByDepth,
            acceptanceByDepth: acceptanceByDepth,
            bestDepth: bestDepth,
            multiplierVsAr: multiplier,
            verifiedOnHardware: hardware,
            sampler: sampler
        )
    }
}

public struct ForgeSampler: Equatable, Sendable, Codable {
    public var temperature: Double
    public var topP: Double
    public var topK: Int

    public init(temperature: Double = 0.6, topP: Double = 0.95, topK: Int = 20) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }
}

// MARK: - ForgeBrandInfo
//
// Set before Forge starts so the backend writes the artifact into
// the user-chosen folder. The app owns a single hard rule: whatever
// the user names the model, the final filesystem / picker name ends
// in `-MTPLX`.

public struct ForgeBrandInfo: Equatable, Sendable {
    public static let suffix = "MTPLX"

    // Legacy role values are kept so older settings/tests can decode
    // and compare state, but the app no longer exposes role tags.
    public enum Role: String, CaseIterable, Equatable, Sendable, Codable {
        case speed = "Speed"
        case quality = "Quality"
        case balanced = "Balanced"
        case custom = "Custom"
    }

    public var role: Role
    /// Free-form label shown when `role == .custom`. Ignored otherwise.
    public var customRoleLabel: String
    /// Fully-resolved branded name, e.g. "Qwen3.6-35B-A3B-MTPLX".
    /// The suffix is locked by `resolvedBrandedName(...)`.
    public var brandedName: String

    public init(
        role: Role = .speed,
        customRoleLabel: String = "",
        brandedName: String = ""
    ) {
        self.role = role
        self.customRoleLabel = customRoleLabel
        self.brandedName = brandedName
    }

    /// Legacy role label for old state snapshots. New Forge naming
    /// does not append this value to the model name.
    public var effectiveRoleLabel: String {
        if role == .custom {
            let trimmed = customRoleLabel.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return role.rawValue
    }

    /// Default branded name for a given source repo. Sanitizes the
    /// source owner/repo so the resulting name is a valid HF repo
    /// segment and a valid filesystem directory name.
    public static func derivedBrandedName(sourceRepo: String, role: Role, customRoleLabel: String = "") -> String {
        resolvedBrandedName(userName: defaultBaseName(sourceRepo: sourceRepo), fallbackSourceRepo: sourceRepo)
    }

    public static func defaultBaseName(sourceRepo: String) -> String {
        sanitizedBaseName(sourceRepo.replacingOccurrences(of: "/", with: "-"))
    }

    public static func baseName(fromBrandedName brandedName: String) -> String {
        sanitizedBaseName(brandedName)
    }

    public static func resolvedBrandedName(userName: String, fallbackSourceRepo: String? = nil) -> String {
        let fallback = fallbackSourceRepo.map(defaultBaseName(sourceRepo:)) ?? "Model"
        let base = sanitizedBaseName(userName, fallback: fallback)
        return "\(base)-\(suffix)"
    }

    public static func sanitizedBaseName(_ raw: String, fallback: String = "Model") -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let dashedScalars = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .unicodeScalars
            .map { allowed.contains($0) ? String($0) : "-" }
            .joined()
        let parts = dashedScalars
            .split(separator: "-", omittingEmptySubsequences: true)
            .map(String.init)
            .filter { $0.caseInsensitiveCompare(suffix) != .orderedSame }
        let cleaned = parts.joined(separator: "-").trimmingCharacters(in: CharacterSet(charactersIn: ".-_"))
        if !cleaned.isEmpty { return cleaned }
        let fallbackCleaned = fallback
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .unicodeScalars
            .map { allowed.contains($0) ? String($0) : "-" }
            .joined()
            .split(separator: "-", omittingEmptySubsequences: true)
            .map(String.init)
            .filter { $0.caseInsensitiveCompare(suffix) != .orderedSame }
            .joined(separator: "-")
            .trimmingCharacters(in: CharacterSet(charactersIn: ".-_"))
        return fallbackCleaned.isEmpty ? "Model" : fallbackCleaned
    }
}

// MARK: - ForgePublishOptions
//
// Captured on the optional PublishStage. The repo name defaults to
// `<user-handle>/<branded-name>` once the user has supplied (or
// imported) an HF handle. License options are intentionally a small
// fixed list — every "other" license forces the user to type an
// SPDX id manually so we never accidentally publish under a fake
// license.

public struct ForgePublishOptions: Equatable, Sendable {
    public enum Visibility: String, CaseIterable, Equatable, Sendable, Codable {
        case publicRepo = "public"
        case privateRepo = "private"
    }

    public var repoName: String
    public var visibility: Visibility
    public var licenseSPDX: String
    public var readmeBody: String

    public init(
        repoName: String = "",
        visibility: Visibility = .publicRepo,
        licenseSPDX: String = "apache-2.0",
        readmeBody: String = ""
    ) {
        self.repoName = repoName
        self.visibility = visibility
        self.licenseSPDX = licenseSPDX
        self.readmeBody = readmeBody
    }
}

// MARK: - ForgeFeatureState
//
// Pure value state machine for the Forge tab. Mirrors the shape of
// `OnboardingFeatureState`: no `ObservableObject`, no `Combine`, no
// service dependencies — trivial to construct, mutate from a test,
// and compare for equality.
//
// Live progress (download bytes, calibration loss, verify candidate
// landings, publish bytes) is intentionally NOT held here. That data
// lives on `ForgeOrchestrator` as `@Published` properties because it
// changes on a sub-second cadence and would invalidate state
// equality checks otherwise.

public struct ForgeFeatureState: Equatable, Sendable {
    public var step: ForgeStep
    public var subMode: ForgeSubMode
    public var sourceRepoInput: String
    public var sourceProbe: ForgeSourceProbe?
    public var recipe: ForgeRecipe
    public var hardware: DetectedHardware?
    public var verification: ForgeVerification?
    public var brand: ForgeBrandInfo
    public var publish: ForgePublishOptions
    /// User explicitly opted into a recipe with `mtpPolicy ==
    /// .requantize`. Resets to false whenever the source changes.
    public var hasAcknowledgedDegradedMTP: Bool

    public init(
        step: ForgeStep = .source,
        subMode: ForgeSubMode = .create,
        sourceRepoInput: String = "",
        sourceProbe: ForgeSourceProbe? = nil,
        recipe: ForgeRecipe = ForgeRecipe(),
        hardware: DetectedHardware? = nil,
        verification: ForgeVerification? = nil,
        brand: ForgeBrandInfo = ForgeBrandInfo(),
        publish: ForgePublishOptions = ForgePublishOptions(),
        hasAcknowledgedDegradedMTP: Bool = false
    ) {
        self.step = step
        self.subMode = subMode
        self.sourceRepoInput = sourceRepoInput
        self.sourceProbe = sourceProbe
        self.recipe = recipe
        self.hardware = hardware
        self.verification = verification
        self.brand = brand
        self.publish = publish
        self.hasAcknowledgedDegradedMTP = hasAcknowledgedDegradedMTP
    }

    // MARK: Derived

    /// Repo id ultimately handed to `mtplx forge build`. Trims and
    /// normalises whitespace; `nil` when the user hasn't typed
    /// anything meaningful.
    public var resolvedRepoID: String? {
        let trimmed = sourceRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    public var hasSpeedWinningVerification: Bool {
        guard let verification else { return false }
        return verification.bestDepth > 0 && verification.multiplierVsAr > 1.0
    }

    // MARK: Transitions

    public mutating func goNext() {
        let all = ForgeStep.allCases
        guard let i = all.firstIndex(of: step), i < all.count - 1 else { return }
        step = all[i + 1]
    }

    public mutating func goBack() {
        let all = ForgeStep.allCases
        guard let i = all.firstIndex(of: step), i > 0 else { return }
        step = all[i - 1]
    }

    public mutating func selectSubMode(_ mode: ForgeSubMode) {
        guard subMode != mode else { return }
        subMode = mode
    }

    public mutating func resetWizard() {
        step = .source
        subMode = .create
        sourceRepoInput = ""
        sourceProbe = nil
        recipe = ForgeRecipe()
        verification = nil
        brand = ForgeBrandInfo()
        publish = ForgePublishOptions()
        hasAcknowledgedDegradedMTP = false
    }

    public mutating func setSourceRepo(_ input: String) {
        let normalised = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalised != sourceRepoInput {
            sourceRepoInput = normalised
            sourceProbe = nil
            hasAcknowledgedDegradedMTP = false
            // Reset downstream state so a fresh source can't inherit
            // a stale recipe / brand / verification by accident.
            recipe = ForgeRecipe()
            verification = nil
            brand = ForgeBrandInfo()
        }
    }

    public mutating func recordProbe(_ probe: ForgeSourceProbe) {
        sourceProbe = probe
        // Default the recipe to whatever the source format suggests
        // and the brand name to the locked MTPLX suffix. Both can be
        // overridden before build starts in PlanStage.
        recipe = ForgeRecipe.defaultFor(format: probe.sourceFormat)
        brand = ForgeBrandInfo(
            role: .speed,
            customRoleLabel: "",
            brandedName: ForgeBrandInfo.derivedBrandedName(sourceRepo: probe.hfRepo, role: .speed)
        )
        hasAcknowledgedDegradedMTP = false
    }

    public mutating func updateRecipe(_ recipe: ForgeRecipe) {
        self.recipe = recipe
        // If the user moved the policy back to a safe option, drop
        // the prior degradation acknowledgement so a future override
        // requires a fresh confirmation.
        if !recipe.degradesMtp {
            hasAcknowledgedDegradedMTP = false
        }
    }

    public mutating func acknowledgeDegradedMTP() {
        hasAcknowledgedDegradedMTP = true
    }

    public mutating func recordVerification(_ v: ForgeVerification) {
        verification = v
    }

    public mutating func updateBrand(_ info: ForgeBrandInfo) {
        brand = info
    }

    public mutating func updatePublish(_ options: ForgePublishOptions) {
        publish = options
    }

    // MARK: canAdvance gates

    /// Whether the primary "Next" / "Start" button should be enabled
    /// on the current step. Service-driven steps (convert, calibrate,
    /// verify, publishing) gate themselves via the orchestrator and
    /// ignore this flag — it covers the user-input-driven cases only.
    public var canAdvance: Bool {
        switch step {
        case .source:
            guard resolvedRepoID != nil, let probe = sourceProbe else { return false }
            switch probe.verdict {
            case .forgeable:
                return true
            case .alreadyMTPLX:
                // SourceStage swaps the primary CTA for "Install
                // instead" — the orchestrator handles the install
                // path; the wizard does not advance to Plan.
                return false
            case .noMtpHeads, .probeFailed:
                return false
            }
        case .plan:
            // Recipe is always derived from the probe so we have a
            // safe default. The only thing that blocks advancing is
            // an unacknowledged degraded-MTP override.
            if recipe.degradesMtp && !hasAcknowledgedDegradedMTP { return false }
            return sourceProbe?.verdict == .forgeable
        case .convert, .calibrate, .verify, .publishing:
            // Orchestrator-driven; the relevant stage view enables
            // its primary CTA only when the corresponding service
            // signals completion.
            return false
        case .brand:
            let trimmed = brand.brandedName.trimmingCharacters(in: .whitespacesAndNewlines)
            return !trimmed.isEmpty && hasSpeedWinningVerification
        case .registered:
            // Registered is terminal — Next exits the wizard.
            return true
        }
    }
}
