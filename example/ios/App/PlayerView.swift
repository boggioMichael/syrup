// PlayerView.swift — the video with subtitles and a subtle box per person,
// drawn from the result's keyframes (boxAt) on every time tick. Tapping a
// face isolates that person's transcript everywhere in the result screens.
import AVFoundation
import LipReaderCore
import SwiftUI

@MainActor
final class PlayerController: ObservableObject {
    let player: AVPlayer
    @Published var time: Double = 0
    @Published var isPlaying = false
    private var observer: Any?

    init(url: URL) {
        player = AVPlayer(url: url)
        observer = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 30), queue: .main) { [weak self] t in
            self?.time = t.seconds
            self?.isPlaying = (self?.player.rate ?? 0) != 0
        }
    }

    deinit { if let observer { player.removeTimeObserver(observer) } }

    func seek(to seconds: Double) {
        let target = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        time = max(0, seconds)
    }

    func togglePlay() {
        if player.rate != 0 { player.pause() } else { player.play() }
        isPlaying = player.rate != 0
    }
}

/// AVPlayerLayer in a UIView, aspect-fit so the view's size is the video's.
struct PlayerLayerView: UIViewRepresentable {
    let player: AVPlayer

    final class PlayerUIView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    func makeUIView(context: Context) -> PlayerUIView {
        let view = PlayerUIView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PlayerUIView, context: Context) {}
}

struct PlayerView: View {
    let result: AnalysisResult
    @EnvironmentObject var player: PlayerController
    @EnvironmentObject var model: AppModel

    private var aspect: CGFloat { CGFloat(max(1, result.video.width)) / CGFloat(max(1, result.video.height)) }

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                PlayerLayerView(player: player.player)
                overlay
            }
            .aspectRatio(aspect, contentMode: .fit)
            HStack {
                Button { player.togglePlay() } label: { Image(systemName: player.isPlaying ? "pause.fill" : "play.fill") }
                Text(Exporters.timestamp(player.time, srt: false).dropLast(4)).monospacedDigit()
                Spacer()
                if let selected = model.selectedTrack {
                    Button("Showing \(result.label(forTrack: selected)) only — show all") { model.selectedTrack = nil }.font(.footnote)
                } else {
                    Text("Tap a face to isolate a person").font(.footnote).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)
            subtitle
            NoticeBanner()
        }
        .padding(.vertical)
    }

    /// Boxes and labels for every person on screen now, in view coordinates:
    /// the ZStack has the video's aspect ratio, so source pixels scale evenly.
    private var overlay: some View {
        GeometryReader { geo in
            let scale = geo.size.width / CGFloat(max(1, result.video.width))
            Canvas { context, _ in
                for track in result.tracks {
                    guard player.time >= track.firstSeen - 0.2, player.time <= track.lastSeen + 0.2, let box = track.boxAt(player.time) else { continue }
                    let rect = CGRect(x: box.x * scale, y: box.y * scale, width: box.w * scale, height: box.h * scale)
                    let isSelected = model.selectedTrack == track.trackId
                    let speaking = track.speaking.contains { player.time >= $0.start && player.time <= $0.end }
                    let colour: Color = isSelected ? .yellow : (speaking ? .green : .white)
                    context.stroke(Path(roundedRect: rect, cornerRadius: 6), with: .color(colour.opacity(isSelected ? 0.95 : 0.6)), lineWidth: isSelected ? 2.5 : 1.5)
                    context.draw(Text(track.label).font(.caption2).foregroundStyle(colour), at: CGPoint(x: rect.minX + 4, y: rect.minY - 8), anchor: .leading)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { location in // VERIFY: the (CGPoint) -> Void form of onTapGesture (iOS 16; iOS 17 adds a CoordinateSpaceProtocol overload)
                let px = location.x / scale, py = location.y / scale
                let hit = result.tracks.filter { $0.boxAt(player.time)?.contains(px, py) ?? false }
                    .min { ($0.boxAt(player.time)?.area ?? .infinity) < ($1.boxAt(player.time)?.area ?? .infinity) }
                model.selectedTrack = hit.map { $0.trackId == model.selectedTrack ? nil : $0.trackId } ?? nil
            }
        }
    }

    /// The segments spoken now, one line per person, uncertain words marked.
    private var subtitle: some View {
        let now = result.segments.filter { player.time >= $0.start && player.time <= $0.end }
            .filter { model.selectedTrack == nil || $0.trackId == model.selectedTrack }
        return VStack(alignment: .leading, spacing: 2) {
            ForEach(now) { segment in
                HStack(alignment: .top, spacing: 6) {
                    Text(result.label(forTrack: segment.trackId) + ":").bold()
                    WordsText(words: segment.words, time: player.time)
                }
            }
            if now.isEmpty { Text(" ").font(.body) }
        }
        .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
        .padding(.horizontal)
    }
}

/// A segment's words; uncertain ones as [word?] in a lighter italic, the
/// word being spoken now underlined.
struct WordsText: View {
    let words: [Word]
    var time: Double? = nil

    var body: some View {
        words.enumerated().map { i, w -> Text in
            var t = Text(w.rendered)
            if w.uncertain { t = t.italic().foregroundStyle(.secondary) } // VERIFY: Text.foregroundStyle(_:) -> Text is iOS 17
            if let time, time >= w.start, time <= w.end { t = t.underline() }
            return i == 0 ? t : Text(" ") + t
        }.reduce(Text(""), +)
    }
}
