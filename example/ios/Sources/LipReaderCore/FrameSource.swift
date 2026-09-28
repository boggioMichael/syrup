// FrameSource.swift — frames in, with their timestamps (video.py).
//
// The reader is created with the video track only: no audio output is ever
// added to it, in any mode. `probe` looks at track *metadata* (is there an
// audio track, what size, what rate), which is what ffprobe does for the
// Python pipeline; the audio modes read sound elsewhere (SpeechTranscriber),
// and visual mode never does.
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

public enum FrameSourceError: Error, LocalizedError {
    case noVideoTrack
    case cannotRead(String)

    public var errorDescription: String? {
        switch self {
        case .noVideoTrack: return "the file has no video track"
        case .cannotRead(let why): return "cannot read the video: \(why)"
        }
    }
}

/// One sampled frame: index in the sampled stream, seconds, BGRA pixels
/// already rotated upright (so boxes match what a player shows).
public struct SampledFrame {
    public let index: Int
    public let time: Double
    public let pixelBuffer: CVPixelBuffer
}

/// What `probe` learns without decoding a single sample.
public struct VideoProbe: Sendable {
    public var info: VideoInfo
    public var hasAudio: Bool
    /// The display transform of the video track (portrait phone videos are stored rotated).
    public var transform: CGAffineTransform
    public var naturalSize: CGSize
}

/// Iterates a video's frames at a target rate. When the source rate is
/// close to the target the frames pass through; otherwise the nearest
/// source frame to each target instant is used, never a duplicate and never
/// an interpolation (lip readers were trained on real frames).
public final class FrameSource {
    public private(set) var info: VideoInfo
    public let sampleFps: Double
    public let sourceFps: Double
    private let reader: AVAssetReader
    private let output: AVAssetReaderOutput
    private let start: Double
    private let end: Double?
    private var nextPick: Double
    private var outIndex = 0
    private var started = false
    private var finished = false

    /// Duration, frame rate, upright size and whether an audio track exists.
    public static func probe(_ url: URL) async throws -> VideoProbe {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw FrameSourceError.noVideoTrack }
        // VERIFY: AVAsynchronousKeyValueLoading's multi-property `load(_:_:_:)` returns a tuple (iOS 16+).
        let (naturalSize, transform, nominalFps) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let upright = CGRect(origin: .zero, size: naturalSize).applying(transform)
        let fps = nominalFps > 0 ? Double(nominalFps) : 25.0
        let info = VideoInfo(source: url.lastPathComponent, duration: max(0, duration.seconds), fps: fps,
                             width: max(1, Int(abs(upright.width).rounded())), height: max(1, Int(abs(upright.height).rounded())))
        return VideoProbe(info: info, hasAudio: !audioTracks.isEmpty, transform: transform, naturalSize: naturalSize)
    }

    public init(url: URL, probe: VideoProbe, sampleFps: Double? = nil, start: Double = 0, end: Double? = nil) async throws {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw FrameSourceError.noVideoTrack }
        sourceFps = probe.info.fps > 0 ? probe.info.fps : 25.0
        self.sampleFps = sampleFps ?? sourceFps
        self.start = max(0, start)
        self.end = end
        nextPick = self.start
        info = probe.info
        info.sampledFps = self.sampleFps
        let settings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        do { reader = try AVAssetReader(asset: asset) } catch { throw FrameSourceError.cannotRead(error.localizedDescription) }
        if probe.transform.isIdentity {
            let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
            trackOutput.alwaysCopiesSampleData = false
            output = trackOutput
        } else {
            // A rotated track is rendered upright through a composition so that
            // the frames, the boxes in the result and AVPlayer's picture share one
            // coordinate system.
            // VERIFY: AVMutableVideoComposition.videoComposition(withPropertiesOf:) is the async form (iOS 16+).
            let composition = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset)
            let compositionOutput = AVAssetReaderVideoCompositionOutput(videoTracks: [track], videoSettings: settings)
            compositionOutput.videoComposition = composition
            compositionOutput.alwaysCopiesSampleData = false
            output = compositionOutput
        }
        guard reader.canAdd(output) else { throw FrameSourceError.cannotRead("reader refuses the video output") }
        reader.add(output)
        let from = CMTime(seconds: self.start, preferredTimescale: 600)
        let to = end.map { CMTime(seconds: $0, preferredTimescale: 600) } ?? .positiveInfinity
        reader.timeRange = CMTimeRange(start: from, end: to)
    }

    /// The next sampled frame, nil at the end. Pull-based on purpose: the
    /// caller finishes with a frame before the next one is decoded, so memory
    /// stays at a few buffers however long the video.
    public func next() throws -> SampledFrame? {
        if finished { return nil }
        if !started {
            started = true
            guard reader.startReading() else { throw FrameSourceError.cannotRead(reader.error?.localizedDescription ?? "startReading failed") }
        }
        let sourceInterval = 1.0 / sourceFps
        let step = 1.0 / sampleFps
        let passthrough = abs(step - sourceInterval) < 1e-6
        while let sample = output.copyNextSampleBuffer() {
            let t = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            if let end, t > end + 0.5 * sourceInterval { break }
            if t < start - 0.5 * sourceInterval { continue }
            // The first source frame at or past a target instant (less half a
            // source interval) is the nearest one to it.
            let take = passthrough || t >= nextPick - 0.5 * sourceInterval
            if !take { continue }
            nextPick += step
            while nextPick <= t { nextPick += step } // a slow source is passed through, never duplicated
            let frame = SampledFrame(index: outIndex, time: t, pixelBuffer: pixels)
            outIndex += 1
            return frame
        }
        finished = true
        if reader.status == .failed { throw FrameSourceError.cannotRead(reader.error?.localizedDescription ?? "reader failed") }
        return nil
    }

    public func cancel() {
        finished = true
        reader.cancelReading()
    }
}
