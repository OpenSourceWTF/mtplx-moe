// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MTPLXApp",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MTPLXAppCore", targets: ["MTPLXAppCore"]),
        .executable(name: "MTPLXApp", targets: ["MTPLXAppHost"]),
    ],
    dependencies: [
        // Markdown renderer for settled chat replies. Live generation stays
        // plain text so token streaming does not pay a per-delta parse cost.
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.1"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.9.2"),
    ],
    targets: [
        .target(
            name: "MTPLXAppCore",
            path: "Sources/MTPLXAppCore",
            exclude: ["Resources"]
        ),
        .executableTarget(
            name: "MTPLXAppHost",
            dependencies: [
                "MTPLXAppCore",
                .product(name: "MarkdownUI", package: "swift-markdown-ui"),
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/MTPLXAppHost",
            exclude: ["Resources"]
        ),
        .testTarget(
            name: "MTPLXAppCoreTests",
            dependencies: ["MTPLXAppCore"],
            path: "Tests/MTPLXAppCoreTests"
        ),
        // TEMPORARY verification harness — added by one commit and reverted by
        // the final commit of this PR, so the net diff is the fix alone. It lets
        // a reviewer `git checkout` the harness commit and run
        // `swift test --filter ComposerIMEIntegrationTests` to see the before
        // (fails) / after (passes). Shippable coverage lives in
        // MTPLXAppCoreTests/ComposerTextSyncTests.
        .testTarget(
            name: "MTPLXAppHostTests",
            dependencies: ["MTPLXAppHost"],
            path: "Tests/MTPLXAppHostTests"
        ),
    ]
)
