// LipReaderApp.swift — the app, its model, and the small persistence it
// needs: where imported videos and results live, the pending analysis a
// BGProcessingTask resumes, the share-extension inbox, the API token.
//
// Everything runs on the device unless the user configured an inference
// API and the device cannot read the chosen language; where a result was
// produced is always shown (D6).
import AVFoundation
import BackgroundTasks
import LipReaderCore
import Security
import SwiftUI

enum AppGroup {
    /// PLACEHOLDER: the App Group both the app and the share extension are
    /// entitled to (Signing & Capabilities -> App Groups). Keep the two in step.
    static let identifier = "group.dev.lipreader.shared"
}

@main
struct LipReaderApp: App {
    @StateObject private var model = AppModel.shared
    @Environment(\.scenePhase) private var scenePhase

    init() { BackgroundAnalysis.register() }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .onOpenURL { url in
                    if url.scheme == "lipreader" { model.pickUpSharedInbox() } else { model.open(url) }
                }
                .task { model.pickUpSharedInbox() }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: model.pickUpSharedInbox()
            case .background: model.scheduleBackgroundContinuation()
            default: break
            }
        }
    }
}

// MARK: - Model

@MainActor
final class AppModel: ObservableObject {
    static let shared = AppModel()

    enum Phase: Equatable {
        case idle
        case running(Double, String)
        case done
        case failed(String)
    }

    @Published var videoURL: URL?
    @Published var probe: VideoProbe?
    @Published var options = Options()
    @Published var phase: Phase = .idle
    @Published var result: AnalysisResult?
    /// Where the current result came from; "server" is shown prominently.
    @Published var processing: Processing = .local
    @Published var remoteJob: Job?
    @Published var selectedTrack: Int?
    @Published var apiURL: String = UserDefaults.standard.string(forKey: "apiURL") ?? "" {
        didSet { UserDefaults.standard.set(apiURL, forKey: "apiURL") }
    }
    @Published var preferOnDevice: Bool = UserDefaults.standard.object(forKey: "preferOnDevice") as? Bool ?? true {
        didSet { UserDefaults.standard.set(preferOnDevice, forKey: "preferOnDevice") }
    }
    let registry = LanguageRegistry.shared
    private var task: Task<Void, Never>?

    init() {
        if let saved = ResultStore.loadLatest() { result = saved.result; processing = saved.processing; phase = .done }
    }

    var apiToken: String {
        get { Keychain.read() ?? "" }
        set { Keychain.write(newValue) }
    }

    var remote: RemoteLipReader? {
        guard let url = URL(string: apiURL), url.scheme != nil else { return nil }
        return RemoteLipReader(baseURL: url, token: apiToken)
    }

    /// Whether this device can read the chosen language in the chosen mode.
    var canRunLocally: Bool {
        guard options.mode.readsLips else { return true }
        if options.language == "auto" { return !registry.availableLanguages().isEmpty }
        return (try? registry.resolve(options.language)) != nil
    }

    /// The local copy of a video the user opened, probed. Security-scoped
    /// files (Files app) are copied while the scope is open.
    func open(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let copy = try Store.imports.appending(path: "\(UUID().uuidString).\(url.pathExtension.isEmpty ? "mov" : url.pathExtension)")
            try FileManager.default.copyItem(at: url, to: copy)
            videoURL = copy
            result = nil
            selectedTrack = nil
            phase = .idle
            Task { probe = try? await FrameSource.probe(copy) }
        } catch {
            phase = .failed("could not import the video: \(error.localizedDescription)")
        }
    }

    func analyze() {
        guard let url = videoURL else { return }
        task?.cancel()
        let options = options
        let useRemote = !(preferOnDevice && canRunLocally) && remote != nil
        let remote = remote
        let preferOnDeviceSpeech = preferOnDevice
        phase = .running(0, useRemote ? "uploading to \(remote?.baseURL.host() ?? "server")" : "on this device")
        task = Task.detached(priority: .userInitiated) { [weak self] in
            let report: @Sendable (Double, String) -> Void = { fraction, place in
                Task { @MainActor in self?.phase = .running(fraction, place) }
            }
            do {
                if useRemote, let remote {
                    let job = try await remote.analyze(fileURL: url, options: options) { report($0, "on the server") }
                    await MainActor.run { self?.finish(job.result!, processing: job.processing ?? .server, job: job) }
                } else {
                    if options.mode.usesAudio { _ = await SpeechTranscriber.requestAuthorization() }
                    let result = try await LipReaderPipeline.analyze(url: url, options: options, preferOnDeviceSpeech: preferOnDeviceSpeech) {
                        report($0, "on this device")
                    }
                    await MainActor.run { self?.finish(result, processing: .local, job: nil) }
                }
            } catch is CancellationError {
                await MainActor.run { self?.phase = .idle }
            } catch {
                await MainActor.run { self?.phase = .failed(error.localizedDescription) }
            }
            PendingStore.clear()
        }
    }

    private func finish(_ result: AnalysisResult, processing: Processing, job: Job?) {
        self.result = result
        self.processing = processing
        remoteJob = job
        phase = .done
        try? ResultStore.save(result, processing: processing)
    }

    func cancel() {
        task?.cancel()
        phase = .idle
    }

    /// "Delete now": the result here, the job on the server, the imported copies.
    func deleteEverything() async {
        cancel()
        if let job = remoteJob, let remote { try? await remote.delete(job.id) }
        remoteJob = nil
        result = nil
        probe = nil
        videoURL = nil
        selectedTrack = nil
        ResultStore.clear()
        PendingStore.clear()
        try? FileManager.default.removeItem(at: Store.imports)
    }

    /// A long local analysis interrupted by backgrounding is re-run by a
    /// BGProcessingTask from the saved request (the in-flight Task resumes if
    /// the app merely suspends; the task covers the case where it is killed).
    func scheduleBackgroundContinuation() {
        guard case .running = phase, let url = videoURL, remoteJob == nil else { return }
        PendingStore.save(PendingAnalysis(videoURL: url, options: options, preferOnDevice: preferOnDevice))
        BackgroundAnalysis.schedule()
    }

    func pickUpSharedInbox() {
        if let url = SharedInbox.takeNewest() { open(url) }
    }
}

// MARK: - Storage

enum Store {
    static var root: URL {
        get throws {
            let base = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let dir = base.appending(path: "LipReader")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        }
    }
    static var imports: URL {
        get throws {
            let dir = try root.appending(path: "Imports")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        }
    }
}

struct SavedResult: Codable {
    var result: AnalysisResult
    var processing: Processing
}

enum ResultStore {
    static var url: URL? { try? Store.root.appending(path: "latest-result.json") }
    static func save(_ result: AnalysisResult, processing: Processing) throws {
        guard let url else { return }
        try JSONCoding.encoder.encode(SavedResult(result: result, processing: processing)).write(to: url, options: .atomic)
    }
    static func loadLatest() -> SavedResult? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONCoding.decoder.decode(SavedResult.self, from: data)
    }
    static func clear() { if let url { try? FileManager.default.removeItem(at: url) } }
}

struct PendingAnalysis: Codable {
    var videoURL: URL
    var options: Options
    var preferOnDevice: Bool
}

enum PendingStore {
    static var url: URL? { try? Store.root.appending(path: "pending-analysis.json") }
    static func save(_ pending: PendingAnalysis) { if let url { try? JSONCoding.encoder.encode(pending).write(to: url, options: .atomic) } }
    static func load() -> PendingAnalysis? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONCoding.decoder.decode(PendingAnalysis.self, from: data)
    }
    static func clear() { if let url { try? FileManager.default.removeItem(at: url) } }
}

// MARK: - Background processing

enum BackgroundAnalysis {
    /// Listed under BGTaskSchedulerPermittedIdentifiers in Info.plist.
    static let identifier = "dev.lipreader.analysis"

    static func register() {
        // VERIFY: register(forTaskWithIdentifier:using:launchHandler:) must run before the app finishes launching.
        _ = BGTaskScheduler.shared.register(forTaskWithIdentifier: identifier, using: nil) { task in
            guard let processing = task as? BGProcessingTask else { task.setTaskCompleted(success: false); return }
            handle(processing)
        }
    }

    static func schedule() {
        let request = BGProcessingTaskRequest(identifier: identifier)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        try? BGTaskScheduler.shared.submit(request)
    }

    static func handle(_ task: BGProcessingTask) {
        guard let pending = PendingStore.load() else { task.setTaskCompleted(success: true); return }
        let work = Task.detached(priority: .background) {
            do {
                let result = try await LipReaderPipeline.analyze(url: pending.videoURL, options: pending.options, preferOnDeviceSpeech: pending.preferOnDevice)
                try ResultStore.save(result, processing: .local)
                PendingStore.clear()
                task.setTaskCompleted(success: true)
            } catch {
                task.setTaskCompleted(success: false)
            }
        }
        task.expirationHandler = { work.cancel() }
    }
}

// MARK: - Share extension inbox

enum SharedInbox {
    static var directory: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: AppGroup.identifier)?.appending(path: "Inbox")
    }

    /// The newest video the extension left, moved out of the inbox.
    static func takeNewest() -> URL? {
        guard let directory,
              let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey]),
              let newest = files.max(by: { modified($0) < modified($1) }) else { return nil }
        let destination = FileManager.default.temporaryDirectory.appending(path: newest.lastPathComponent)
        try? FileManager.default.removeItem(at: destination)
        guard (try? FileManager.default.moveItem(at: newest, to: destination)) != nil else { return nil }
        return destination
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}

// MARK: - Keychain (API token)

enum Keychain {
    static let service = "dev.lipreader.api-token"

    static func read() -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func write(_ value: String) {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service]
        _ = SecItemDelete(base as CFDictionary)
        guard !value.isEmpty else { return }
        var add = base
        add[kSecValueData as String] = Data(value.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        _ = SecItemAdd(add as CFDictionary, nil)
    }
}
