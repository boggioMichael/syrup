// SettingsView.swift — the optional inference API (URL and bearer token),
// and whether to prefer the device over it.
import LipReaderCore
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var token = ""
    @State private var health: HealthResponse?
    @State private var checkFailure: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("On-device") {
                    Toggle("Prefer on-device processing", isOn: $model.preferOnDevice)
                    Text("When on, the server is used only for a language this device cannot read. When off, every analysis goes to the server if one is configured.")
                        .font(.footnote).foregroundStyle(.secondary)
                    LabeledContent("Visual models here", value: model.registry.availableLanguages().isEmpty ? "none bundled" : model.registry.availableLanguages().joined(separator: ", "))
                }
                Section("Inference API (docs/api.md)") {
                    TextField("http://127.0.0.1:8765", text: $model.apiURL)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Bearer token (if the service requires one)", text: $token)
                        .onChange(of: token) { _, value in model.apiToken = value }
                    Button("Check server") { Task { await check() } }.disabled(model.remote == nil)
                    if let health {
                        LabeledContent("Processing", value: health.processing.rawValue)
                        LabeledContent("Version", value: health.version)
                        ForEach(health.languages) { language in
                            let visual = language.visual.available ? "yes" : "no"
                            let audio = language.audio.map { $0.available ? ", audio yes" : ", audio no" } ?? ""
                            Text("\(language.name): visual \(visual)\(audio)").font(.footnote)
                        }
                    }
                    if let checkFailure { Text(checkFailure).foregroundStyle(.red).font(.footnote) }
                    Text("Uploaded video is deleted by the service once its frames have been read; results expire and can be deleted from the Privacy screen.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .toolbar { Button("Done") { dismiss() } }
            .onAppear { token = model.apiToken }
        }
    }

    private func check() async {
        guard let remote = model.remote else { return }
        do {
            health = try await remote.health()
            checkFailure = nil
        } catch {
            health = nil
            checkFailure = error.localizedDescription
        }
    }
}
