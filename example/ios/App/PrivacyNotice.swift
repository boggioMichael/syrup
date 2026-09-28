// PrivacyNotice.swift — the notice every transcript screen carries, what
// the app keeps and where processing happens, and "Delete now".
import LipReaderCore
import SwiftUI

/// One line under every transcript view (schema.py NOTICE, D4).
struct NoticeBanner: View {
    var body: some View {
        Text(NOTICE)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .padding(.vertical, 4)
            .background(Color.secondary.opacity(0.08))
            .accessibilityLabel("Notice: \(NOTICE)")
    }
}

struct PrivacyNoticeView: View {
    @EnvironmentObject var model: AppModel
    @State private var deleted = false

    var body: some View {
        List {
            Section("About the transcript") { Text(NOTICE) }
            Section("What is kept") {
                Text("People are numbered within one video only (Person 1, Person 2…). No face embedding or identity is computed or stored; a result holds boxes, speaking times and words.")
                Text("The imported copy of the video and the latest result stay in the app's own storage until you delete them here.")
            }
            Section("Where processing happens") {
                Label("On this device: frames are read by Vision and Core ML and discarded. Visual mode never opens the audio track.", systemImage: "iphone")
                Label("Audio modes use Apple's speech recogniser: on device where the language supports it, otherwise Apple's servers (the screen says which).", systemImage: "waveform")
                if model.remote != nil {
                    Label("Server at \(model.apiURL): the whole video is uploaded when the device cannot read the chosen language (or on-device is off). The service deletes the upload once read and keeps the result until it expires or you delete it.", systemImage: "network")
                } else {
                    Label("No inference API is configured: nothing leaves the device except speech recognition, when it is not on-device.", systemImage: "network.slash")
                }
                if let job = model.remoteJob {
                    LabeledContent("Last remote job", value: "\(job.id) · processing: \(job.processing?.rawValue ?? "?") · expires \(job.expiresAt ?? "-")")
                        .font(.footnote)
                }
            }
            Section {
                Button("Delete now", role: .destructive) { Task { await model.deleteEverything(); deleted = true } }
                Text("Removes the imported video, the saved result, and the job on the server if there is one.").font(.footnote).foregroundStyle(.secondary)
                if deleted { Text("Deleted.").font(.footnote).foregroundStyle(.green) }
            }
        }
        .navigationTitle("Privacy")
    }
}
