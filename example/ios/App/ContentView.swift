// ContentView.swift — pick a video (Photos, Files, or the camera), analyse
// it, and open the result screens.
import AVFoundation
import LipReaderCore
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var pickerItem: PhotosPickerItem?
    @State private var showingFiles = false
    @State private var showingRecorder = false
    @State private var showingSettings = false
    @State private var importError: String?

    var body: some View {
        NavigationStack {
            List {
                Section("Video") {
                    PhotosPicker(selection: $pickerItem, matching: .videos) { Label("Choose from Photos", systemImage: "photo.on.rectangle") }
                    Button { showingFiles = true } label: { Label("Open from Files", systemImage: "folder") }
                    Button { showingRecorder = true } label: { Label("Record", systemImage: "video") }
                    if let probe = model.probe {
                        Text("\(probe.info.source ?? "video") · \(probe.info.width)×\(probe.info.height) · \(probe.info.fps, specifier: "%.2f") fps · \(probe.info.duration, specifier: "%.1f") s · \(probe.hasAudio ? "with sound" : "no audio track")")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    if let importError { Text(importError).font(.footnote).foregroundStyle(.red) }
                }
                Section {
                    NavigationLink("Analyse") { AnalysisView() }.disabled(model.videoURL == nil)
                }
                if let result = model.result, let url = model.videoURL {
                    Section("Result · processed \(model.processing == .local ? "on this device" : "on the server")") {
                        NavigationLink("Open result") { ResultView(result: result, videoURL: url) }
                        Text("\(result.tracks.count) people, \(result.segments.count) segments, mode \(result.mode.rawValue)")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }
                Section { NavigationLink("Privacy and where processing happens") { PrivacyNoticeView() } }
            }
            .navigationTitle("LipReader")
            .toolbar { Button { showingSettings = true } label: { Image(systemName: "gear") } }
            .fileImporter(isPresented: $showingFiles, allowedContentTypes: [.movie, .video, .mpeg4Movie, .quickTimeMovie]) { outcome in
                switch outcome {
                case .success(let url): model.open(url)
                case .failure(let error): importError = error.localizedDescription
                }
            }
            .sheet(isPresented: $showingRecorder) { RecorderView { url in model.open(url) } }
            .sheet(isPresented: $showingSettings) { SettingsView() }
            .onChange(of: pickerItem) { _, item in
                guard let item else { return }
                Task {
                    do {
                        if let movie = try await item.loadTransferable(type: Movie.self) { model.open(movie.url) } else { importError = "that item is not a video" }
                    } catch { importError = error.localizedDescription }
                    pickerItem = nil
                }
            }
        }
    }
}

/// A picked video copied out of the Photos sandbox as a file.
struct Movie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let copy = FileManager.default.temporaryDirectory.appending(path: "photos-\(UUID().uuidString).\(received.file.pathExtension)")
            try FileManager.default.copyItem(at: received.file, to: copy)
            return Movie(url: copy)
        }
    }
}

// MARK: - Result screens

struct ResultView: View {
    let result: AnalysisResult
    let videoURL: URL
    @StateObject private var player: PlayerController

    init(result: AnalysisResult, videoURL: URL) {
        self.result = result
        self.videoURL = videoURL
        _player = StateObject(wrappedValue: PlayerController(url: videoURL))
    }

    var body: some View {
        TabView {
            PlayerView(result: result).tabItem { Label("Player", systemImage: "play.rectangle") }
            SpeakerTimelineView(result: result).tabItem { Label("Timeline", systemImage: "waveform.path") }
            TranscriptView(result: result).tabItem { Label("Transcript", systemImage: "text.quote") }
            ExportView(result: result).tabItem { Label("Export", systemImage: "square.and.arrow.up") }
        }
        .environmentObject(player)
        .navigationTitle("Result")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Camera

/// Records with the front camera. Sound is off by default: a clip recorded
/// for lip reading needs no microphone, and the visual mode would not read
/// it anyway; the toggle turns it on for the audio modes.
final class CameraRecorder: NSObject, ObservableObject, AVCaptureFileOutputRecordingDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureMovieFileOutput()
    private let queue = DispatchQueue(label: "dev.lipreader.camera")
    @Published var isRecording = false
    @Published var failure: String?
    var finished: ((URL) -> Void)?

    func start(withSound: Bool) async {
        guard await AVCaptureDevice.requestAccess(for: .video) else { failure = "camera access was not granted"; return }
        if withSound, !(await AVCaptureDevice.requestAccess(for: .audio)) { failure = "microphone access was not granted"; return }
        queue.async { [self] in
            session.beginConfiguration()
            session.inputs.forEach(session.removeInput)
            session.outputs.forEach(session.removeOutput)
            session.sessionPreset = .hd1280x720
            if let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
               let input = try? AVCaptureDeviceInput(device: camera), session.canAddInput(input) { session.addInput(input) }
            if withSound, let mic = AVCaptureDevice.default(for: .audio), let input = try? AVCaptureDeviceInput(device: mic), session.canAddInput(input) {
                session.addInput(input)
            }
            if session.canAddOutput(output) { session.addOutput(output) }
            session.commitConfiguration()
            session.startRunning()
        }
    }

    func toggle() {
        if isRecording {
            output.stopRecording()
        } else {
            let url = FileManager.default.temporaryDirectory.appending(path: "recording-\(UUID().uuidString).mov")
            output.startRecording(to: url, recordingDelegate: self)
            isRecording = true
        }
    }

    func stop() { queue.async { [self] in if session.isRunning { session.stopRunning() } } }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        DispatchQueue.main.async {
            self.isRecording = false
            if let error, !FileManager.default.fileExists(atPath: outputFileURL.path) { self.failure = error.localizedDescription; return }
            self.finished?(outputFileURL)
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}

struct RecorderView: View {
    let onRecorded: (URL) -> Void
    @StateObject private var recorder = CameraRecorder()
    @State private var withSound = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 12) {
            CameraPreview(session: recorder.session).clipShape(RoundedRectangle(cornerRadius: 12))
            Toggle("Record sound (needed only for the audio modes)", isOn: $withSound).disabled(recorder.isRecording).font(.footnote)
            if let failure = recorder.failure { Text(failure).foregroundStyle(.red).font(.footnote) }
            HStack {
                Button("Cancel") { recorder.stop(); dismiss() }
                Spacer()
                Button(recorder.isRecording ? "Stop" : "Record") { recorder.toggle() }.buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .task(id: withSound) { recorder.stop(); await recorder.start(withSound: withSound) }
        .onAppear {
            recorder.finished = { url in
                recorder.stop()
                onRecorded(url)
                dismiss()
            }
        }
    }
}
