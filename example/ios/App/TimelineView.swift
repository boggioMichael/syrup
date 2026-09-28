// TimelineView.swift — a lane per person: speaking spans (pale) and
// transcript segments (solid) over time, a playhead, and scrubbing by drag.
// Named SpeakerTimelineView because SwiftUI already has a TimelineView.
import LipReaderCore
import SwiftUI

struct SpeakerTimelineView: View {
    let result: AnalysisResult
    @EnvironmentObject var player: PlayerController
    @EnvironmentObject var model: AppModel

    private var duration: Double { max(result.video.duration, 0.001) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Pale bars: the mouth was moving. Solid bars: words were read. Drag to scrub; tap a name to isolate that person.")
                .font(.footnote).foregroundStyle(.secondary).padding(.horizontal)
            ScrollView {
                VStack(spacing: 6) {
                    ForEach(result.tracks) { track in lane(track) }
                    if result.segments.contains(where: { $0.trackId < 0 }) { unattributedLane }
                }
                .padding(.horizontal)
            }
            NoticeBanner()
        }
        .padding(.vertical)
    }

    private func lane(_ track: PersonTrack) -> some View {
        HStack(spacing: 8) {
            Button(track.label) { model.selectedTrack = model.selectedTrack == track.trackId ? nil : track.trackId }
                .font(.caption.bold())
                .foregroundStyle(model.selectedTrack == track.trackId ? Color.yellow : Color.primary)
                .frame(width: 70, alignment: .leading)
            bars(spans: track.speaking, segments: result.segments.filter { $0.trackId == track.trackId }, presence: (track.firstSeen, track.lastSeen))
        }
        .opacity(model.selectedTrack == nil || model.selectedTrack == track.trackId ? 1 : 0.35)
    }

    private var unattributedLane: some View {
        HStack(spacing: 8) {
            Text("Unattributed").font(.caption).frame(width: 70, alignment: .leading)
            bars(spans: [], segments: result.segments.filter { $0.trackId < 0 }, presence: nil)
        }
    }

    private func bars(spans: [SpeakingSpan], segments: [TranscriptSegment], presence: (Double, Double)?) -> some View {
        GeometryReader { geo in
            let w = geo.size.width
            let x: (Double) -> CGFloat = { t in CGFloat(t / duration) * w }
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4).fill(Color.secondary.opacity(0.12))
                if let presence {
                    Rectangle().fill(Color.secondary.opacity(0.15)).frame(width: max(1, x(presence.1) - x(presence.0))).offset(x: x(presence.0))
                }
                ForEach(Array(spans.enumerated()), id: \.offset) { _, span in
                    RoundedRectangle(cornerRadius: 3).fill(Color.green.opacity(0.35))
                        .frame(width: max(2, x(span.end) - x(span.start))).offset(x: x(span.start))
                }
                ForEach(segments) { segment in
                    RoundedRectangle(cornerRadius: 3).fill(segment.mode == .visual ? Color.blue : Color.orange)
                        .frame(width: max(2, x(segment.end) - x(segment.start)), height: 12).offset(x: x(segment.start))
                        .onTapGesture { player.seek(to: segment.start) }
                }
                Rectangle().fill(Color.red).frame(width: 1.5).offset(x: x(player.time))
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                player.seek(to: Double(min(max(0, value.location.x), w) / w) * duration)
            })
        }
        .frame(height: 28)
    }
}
