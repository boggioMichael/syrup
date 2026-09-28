// TranscriptView.swift — every segment with its speaker, time, language,
// mode, model and confidence; uncertain words marked; tap to seek.
import LipReaderCore
import SwiftUI

struct TranscriptView: View {
    let result: AnalysisResult
    @EnvironmentObject var player: PlayerController
    @EnvironmentObject var model: AppModel

    private var segments: [TranscriptSegment] {
        result.segments.filter { model.selectedTrack == nil || $0.trackId == model.selectedTrack }
    }

    var body: some View {
        VStack(spacing: 0) {
            NoticeBanner()
            List {
                Section {
                    LabeledContent("Language", value: languageLine)
                    if let note = result.language.note { Text(note).font(.footnote).foregroundStyle(.secondary) }
                    if let vsr = result.models.vsr { LabeledContent("Visual model", value: "\(vsr.name) — \(vsr.license)") }
                    if let asr = result.models.asr { LabeledContent("Speech model", value: "\(asr.name) — \(asr.license)") }
                    LabeledContent("Processed", value: model.processing == .local ? "on this device" : "on the server")
                    ForEach(result.warnings, id: \.self) { Text($0).font(.footnote).foregroundStyle(.orange) }
                }
                if segments.isEmpty {
                    Text("No words were read. A language without a model, or no speech seen, produces no text; see the warnings above.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                ForEach(segments) { segment in
                    Button { player.seek(to: segment.start) } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(result.label(forTrack: segment.trackId)).bold()
                                Spacer()
                                Text("\(Exporters.timestamp(segment.start, srt: false).dropLast(4))–\(Exporters.timestamp(segment.end, srt: false).dropLast(4))")
                                    .monospacedDigit()
                            }
                            .font(.subheadline)
                            WordsText(words: segment.words, time: player.time).font(.body)
                            Text("\(segment.language) · \(segment.mode.rawValue) · \(segment.model) · confidence \(segment.confidence, specifier: "%.2f")")
                                .font(.caption2).foregroundStyle(.secondary)
                            if segment.words.contains(where: { $0.raw != nil }) {
                                Text("raw: " + segment.words.map { $0.raw ?? $0.text }.joined(separator: " ")).font(.caption2).foregroundStyle(.tertiary)
                            }
                        }
                        .foregroundStyle(.primary)
                    }
                    .listRowBackground(player.time >= segment.start && player.time <= segment.end ? Color.yellow.opacity(0.12) : nil)
                }
            }
        }
    }

    private var languageLine: String {
        let used = result.language.used ?? "none"
        return "\(used) (\(result.language.detection.rawValue); requested \(result.language.requested))"
    }
}
