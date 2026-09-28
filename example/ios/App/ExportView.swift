// ExportView.swift — the transcript as TXT, JSON, SRT or VTT through the
// share sheet. Uncertain words stay marked in every format; TXT and VTT
// carry the notice.
import LipReaderCore
import SwiftUI

struct ExportView: View {
    let result: AnalysisResult
    @EnvironmentObject var model: AppModel
    @State private var format: ExportFormat = .srt
    @State private var onlySelected = true
    @State private var fileURL: URL?
    @State private var failure: String?

    private var tracks: [Int]? { onlySelected ? model.selectedTrack.map { [$0] } : nil }

    var body: some View {
        Form {
            Section("Format") {
                Picker("Format", selection: $format) {
                    ForEach(ExportFormat.allCases) { Text($0.rawValue.uppercased()).tag($0) }
                }
                .pickerStyle(.segmented)
                if let selected = model.selectedTrack {
                    Toggle("Only \(result.label(forTrack: selected))", isOn: $onlySelected)
                }
            }
            Section("Preview") {
                ScrollView(.horizontal) {
                    Text(preview).font(.system(.caption2, design: .monospaced)).textSelection(.enabled)
                }
                .frame(maxHeight: 220)
            }
            Section {
                if let fileURL {
                    ShareLink(item: fileURL) { Label("Share \(fileURL.lastPathComponent)", systemImage: "square.and.arrow.up") }
                }
                if let failure { Text(failure).foregroundStyle(.red).font(.footnote) }
            }
            Section { Text(NOTICE).font(.footnote).foregroundStyle(.secondary) }
        }
        .navigationTitle("Export")
        .task(id: "\(format.rawValue)-\(tracks ?? [])") { prepare() }
    }

    private var preview: String {
        (try? Exporters.export(result, format: format, tracks: tracks)).map { String($0.prefix(4000)) } ?? ""
    }

    private func prepare() {
        do {
            let base = (result.video.source.map { ($0 as NSString).deletingPathExtension } ?? "transcript") + "-lipreader"
            fileURL = try Exporters.write(result, format: format, tracks: tracks, baseName: base)
            failure = nil
        } catch {
            fileURL = nil
            failure = error.localizedDescription
        }
    }
}
