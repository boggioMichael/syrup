// swift-tools-version: 5.9
// LipReaderCore: the analysis pipeline of the LipReader iOS example app,
// as a library so the app, the share extension and the tests share it.
// No external dependencies: Vision, AVFoundation, Core ML, Accelerate and
// Speech are all system frameworks.
import PackageDescription

let package = Package(
    name: "LipReaderCore",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "LipReaderCore", targets: ["LipReaderCore"]),
    ],
    targets: [
        .target(
            name: "LipReaderCore",
            path: "Sources/LipReaderCore"
        ),
        .testTarget(
            name: "LipReaderCoreTests",
            dependencies: ["LipReaderCore"],
            path: "Tests/LipReaderCoreTests"
        ),
    ]
)
