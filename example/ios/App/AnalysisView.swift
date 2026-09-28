// AnalysisView.swift — the three modes with what each one does, the
// language with its honest availability, where the work will run, progress.
import LipReaderCore
import SwiftUI

struct AnalysisView: View {
    @EnvironmentObject var model: AppModel
    @State private var audioStatus: [String: AudioStatus] = [:]

    static let modeExplanations: [Mode: String] = [
        .visual: "Reads lips only. The audio track is never opened. English only on this device (LipNet, a 51-word vocabulary); other languages have no runnable visual model.",
        .audiovisual: "Reads lips and listens. Sound wins where it exists; the visual reading fills the rest. Uses Apple's speech recogniser.",
        .audioAttributed: "Listens, and uses the faces only to decide who said each word. Hebrew is reachable this way (no visual model exists for it).",
    ]

    var body: some View {
        Form {
            Section("Mode") {
                Picker("Mode", selection: $model.options.mode) {
                    ForEach(Mode.allCases) { mode in Text(title(mode)).tag(mode) }
                }
                .pickerStyle(.segmented)
                Text(Self.modeExplanations[model.options.mode] ?? "").font(.footnote).foregroundStyle(.secondary)
            }
            Section("Language") {
                Picker("Language", selection: $model.options.language) {
                    Text("Auto").tag("auto")
                    ForEach(model.registry.languages()) { status in
                        Text("\(status.name) \(availabilityMark(status))").tag(status.code)
                    }
                }
                Text(languageNote).font(.footnote).foregroundStyle(.secondary)
            }
            Section("Where") {
                if model.preferOnDevice && model.canRunLocally || model.remote == nil {
                    Label("On this device", systemImage: "iphone")
                    if !model.canRunLocally {
                        Text("No visual model for this language is bundled; the visual stage will be skipped with a warning (no text is ever guessed).")
                            .font(.footnote).foregroundStyle(.orange)
                    }
                } else {
                    Label("On the server at \(model.apiURL) — the video is uploaded", systemImage: "network").foregroundStyle(.orange)
                }
            }
            Section {
                switch model.phase {
                case .running(let fraction, let place):
                    ProgressView(value: fraction) { Text("Analysing \(place)…") }
                    Button("Cancel", role: .cancel) { model.cancel() }
                case .failed(let message):
                    Text(message).foregroundStyle(.red).font(.footnote)
                    Button("Analyse") { model.analyze() }
                case .done:
                    if let result = model.result, let url = model.videoURL {
                        NavigationLink("Open result") { ResultView(result: result, videoURL: url) }
                        ForEach(result.warnings, id: \.self) { Text($0).font(.footnote).foregroundStyle(.orange) }
                    }
                    Button("Analyse again") { model.analyze() }
                case .idle:
                    Button("Analyse") { model.analyze() }.disabled(model.videoURL == nil)
                }
            }
        }
        .navigationTitle("Analyse")
        .task(id: model.options.mode) {
            // The audio side is asked only when an audio mode is chosen: visual mode never touches Speech.
            guard model.options.mode.usesAudio else { audioStatus = [:]; return }
            audioStatus = Dictionary(uniqueKeysWithValues: LanguageRegistry.order.map { ($0, SpeechTranscriber.status(for: $0)) })
        }
    }

    private func title(_ mode: Mode) -> String {
        switch mode {
        case .visual: return "Visual"
        case .audiovisual: return "Audiovisual"
        case .audioAttributed: return "Audio-attributed"
        }
    }

    /// ✓ a visual model runs here, ♪ Apple's recogniser has the language, — neither.
    private func availabilityMark(_ status: LanguageStatus) -> String {
        let visual = status.visualAvailable ? "✓" : "—"
        let audio = (audioStatus[status.code]?.available ?? false) ? "♪" : "—"
        switch model.options.mode {
        case .visual: return visual
        case .audioAttributed: return audio
        case .audiovisual: return visual + audio
        }
    }

    private var languageNote: String {
        let code = model.options.language
        if code == "auto" {
            return model.options.mode.readsLips
                ? "No visual language identification exists: \"auto\" means the one visual model available (English), reported as \"assumed\"."
                : "Apple's speech recogniser needs a locale: \"auto\" uses the device language, reported as \"assumed\"."
        }
        guard let status = model.registry.languages().first(where: { $0.code == code }) else { return "" }
        var lines: [String] = []
        if model.options.mode.readsLips {
            lines.append(status.visualAvailable ? "Visual: \(status.visualModel ?? "") — \(status.visualLicense ?? "")"
                                                : "Visual: not available. \(status.visualNote)")
        }
        if model.options.mode.usesAudio, let audio = audioStatus[code] {
            lines.append("Audio: \(audio.available ? "" : "not available. ")\(audio.note)")
        }
        return lines.joined(separator: "\n")
    }
}
