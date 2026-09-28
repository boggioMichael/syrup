// FaceTracker.swift — faces per frame (Vision), the same person from frame
// to frame (track.py), and shot cuts (cuts.py).
//
// Detections are matched to live tracks by intersection-over-union with the
// Hungarian assignment; a track survives a few frames without a detection
// (its box is held) and ends after that. A shot cut ends every track at
// once: after a cut nothing on screen is the same person, whatever the boxes
// say. Nothing about a face is kept beyond its boxes and lip points: no
// embeddings, no identity (tracks are numbered within the video only).
import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// A face on one frame. `held` marks a live track's box carried over a frame
/// the detector skipped: not a detection, and never counted as one.
public struct FaceDetection: Sendable {
    public var box: Box
    public var score: Double
    public var landmarks: MouthLandmarks?
    public var held = false

    public init(box: Box, score: Double, landmarks: MouthLandmarks? = nil, held: Bool = false) {
        self.box = box; self.score = score; self.landmarks = landmarks; self.held = held
    }
}

// MARK: - Vision

/// Face rectangles, then landmarks for those rectangles, on one BGRA frame.
public final class VisionFaceDetector {
    public let name = "vision (VNDetectFaceRectanglesRequest + VNDetectFaceLandmarksRequest)"
    public init() {}

    public func detect(_ pixelBuffer: CVPixelBuffer) throws -> [FaceDetection] {
        let width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer)
        let size = CGSize(width: width, height: height)
        // Frames arrive upright from FrameSource, so the orientation is .up.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        let rectangles = VNDetectFaceRectanglesRequest()
        try handler.perform([rectangles])
        guard let faces = rectangles.results, !faces.isEmpty else { return [] }
        let landmarks = VNDetectFaceLandmarksRequest()
        landmarks.inputFaceObservations = faces // VERIFY: landmarks are computed for exactly these faces, in order
        try handler.perform([landmarks])
        let observed = landmarks.results ?? faces
        return observed.map { face in
            // Vision boxes are normalized with the origin bottom-left; the schema wants pixels, y down.
            let r = VNImageRectForNormalizedRect(face.boundingBox, width, height)
            let box = Box(x: r.minX, y: Double(height) - r.maxY, w: r.width, h: r.height)
            return FaceDetection(box: box, score: Double(face.confidence), landmarks: Self.mouthLandmarks(face, size: size))
        }
    }

    private static func mouthLandmarks(_ face: VNFaceObservation, size: CGSize) -> MouthLandmarks? {
        guard let all = face.landmarks, let outer = all.outerLips else { return nil }
        func flipped(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            // VERIFY: pointsInImage(imageSize:) returns image pixels with y up.
            (region?.pointsInImage(imageSize: size) ?? []).map { CGPoint(x: $0.x, y: size.height - $0.y) }
        }
        func centre(_ points: [CGPoint]) -> CGPoint? {
            guard !points.isEmpty else { return nil }
            return CGPoint(x: points.map(\.x).reduce(0, +) / Double(points.count), y: points.map(\.y).reduce(0, +) / Double(points.count))
        }
        return MouthLandmarks(outerLips: flipped(outer), innerLips: flipped(all.innerLips),
                              leftEye: centre(flipped(all.leftEye)), rightEye: centre(flipped(all.rightEye)))
    }
}

// MARK: - Shot cuts

/// A frame that looks nothing like the previous one. Coarse colour
/// histograms are compared frame to frame; a jump far above the running
/// level of change is a cut. Pans and people moving change a histogram
/// slowly, a cut changes it at once.
public struct CutDetector {
    let threshold: Float
    let minGap: Int
    private var previous: [Float]?
    private var sinceCut = Int.max / 2
    public private(set) var lastDistance: Float = 0

    public init(threshold: Float = 0.25, minGapFrames: Int = 5) { self.threshold = threshold; minGap = minGapFrames }

    /// True when this frame starts a new shot.
    public mutating func update(histogram: [Float]) -> Bool {
        var cut = false
        if let previous {
            var distance: Float = 0
            for i in 0..<histogram.count { distance += abs(histogram[i] - previous[i]) }
            lastDistance = 0.5 * distance // 0 identical ... 1 disjoint
            if lastDistance > threshold, sinceCut >= minGap {
                cut = true
                sinceCut = 0
            }
        }
        previous = histogram
        sinceCut += 1
        return cut
    }
}

// MARK: - Tracks

public final class TrackState {
    public let trackId: Int
    public internal(set) var box: Box
    public internal(set) var firstT: Double
    public internal(set) var lastT: Double
    public internal(set) var lastSeenT: Double
    var framesMissing = 0
    var detections = 0
    var attempts = 0 // frames on which the detector actually ran
    var frames = 0
    public internal(set) var keyframes: [Keyframe] = []
    /// Per sampled frame index: the box (held while missing) and landmarks if any.
    public internal(set) var boxes: [Int: Box] = [:]
    public internal(set) var landmarksByFrame: [Int: MouthLandmarks] = [:]
    var detected: Set<Int> = []

    init(trackId: Int, box: Box, t: Double) {
        self.trackId = trackId; self.box = box; firstT = t; lastT = t; lastSeenT = t
    }

    public func toPersonTrack() -> PersonTrack {
        let confidence = attempts > 0 ? Double(detections) / Double(attempts) : 0
        return PersonTrack(trackId: trackId, firstSeen: firstT, lastSeen: lastT, keyframes: keyframes, confidence: min(1, confidence))
    }
}

/// Minimum-cost assignment of rows to columns (Hungarian algorithm), the
/// column of each row or -1. Small matrices only: faces on a frame.
func hungarian(_ cost: [[Double]]) -> [Int] {
    let n = cost.count
    guard n > 0, let m = cost.first?.count, m > 0 else { return Array(repeating: -1, count: n) }
    if n > m {
        let transposed = (0..<m).map { j in (0..<n).map { i in cost[i][j] } }
        var out = Array(repeating: -1, count: n)
        for (j, i) in hungarian(transposed).enumerated() where i >= 0 { out[i] = j }
        return out
    }
    var u = [Double](repeating: 0, count: n + 1), v = [Double](repeating: 0, count: m + 1)
    var p = [Int](repeating: 0, count: m + 1), way = [Int](repeating: 0, count: m + 1)
    for i in 1...n {
        p[0] = i
        var j0 = 0
        var minv = [Double](repeating: .infinity, count: m + 1)
        var used = [Bool](repeating: false, count: m + 1)
        repeat {
            used[j0] = true
            let i0 = p[j0]
            var delta = Double.infinity, j1 = 0
            for j in 1...m where !used[j] {
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j] { minv[j] = cur; way[j] = j0 }
                if minv[j] < delta { delta = minv[j]; j1 = j }
            }
            for j in 0...m {
                if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
            }
            j0 = j1
        } while p[j0] != 0
        repeat { let j1 = way[j0]; p[j0] = p[j1]; j0 = j1 } while j0 != 0
    }
    var result = Array(repeating: -1, count: n)
    for j in 1...m where p[j] != 0 { result[p[j] - 1] = j - 1 }
    return result
}

/// track.py's Tracker: pure bookkeeping, no Vision, so the tests can drive it.
public final class Tracker {
    let iouThreshold: Double
    let graceFrames: Int
    let keyframeEvery: Int
    let smoothing: Double
    public private(set) var live: [TrackState] = []
    public private(set) var finished: [TrackState] = []
    private var nextId = 1

    public init(iouThreshold: Double = 0.3, graceFrames: Int = 12, keyframeEvery: Int = 5, smoothing: Double = 0.5) {
        self.iouThreshold = iouThreshold; self.graceFrames = graceFrames; self.keyframeEvery = keyframeEvery; self.smoothing = smoothing
    }

    /// A shot change: every live track ends here.
    public func cut() {
        finished.append(contentsOf: live)
        live = []
    }

    /// Assigns this frame's detections; returns the tracks alive after it.
    @discardableResult
    public func update(frameIndex: Int, t: Double, detections: [FaceDetection]) -> [TrackState] {
        var matchedTracks = Set<Int>(), matchedDetections = Set<Int>()
        if !live.isEmpty, !detections.isEmpty {
            let cost = live.map { track in detections.map { 1.0 - track.box.iou($0.box) } }
            for (i, j) in hungarian(cost).enumerated() where j >= 0 && cost[i][j] <= 1.0 - iouThreshold {
                observe(live[i], detections[j], frameIndex: frameIndex, t: t)
                matchedTracks.insert(i); matchedDetections.insert(j)
            }
        }
        let heldFrame = !detections.isEmpty && detections.allSatisfy(\.held)
        for (i, track) in live.enumerated() where !matchedTracks.contains(i) {
            track.framesMissing += 1
            track.frames += 1
            if !heldFrame { track.attempts += 1 }
            track.boxes[frameIndex] = track.box
        }
        for (j, det) in detections.enumerated() where !matchedDetections.contains(j) && !det.held {
            let track = TrackState(trackId: nextId, box: det.box, t: t)
            nextId += 1
            live.append(track)
            observe(track, det, frameIndex: frameIndex, t: t, new: true)
        }
        let ended = live.filter { $0.framesMissing > graceFrames }
        for track in ended { trim(track); finished.append(track) }
        live.removeAll { $0.framesMissing > graceFrames }
        return live
    }

    public func finish() -> [TrackState] {
        for track in live { trim(track); finished.append(track) }
        live = []
        return finished.sorted { $0.trackId < $1.trackId }
    }

    private func observe(_ track: TrackState, _ det: FaceDetection, frameIndex: Int, t: Double, new: Bool = false) {
        if new {
            track.box = det.box
        } else {
            let a = smoothing, b = track.box
            track.box = Box(x: b.x + (det.box.x - b.x) * a, y: b.y + (det.box.y - b.y) * a,
                            w: b.w + (det.box.w - b.w) * a, h: b.h + (det.box.h - b.h) * a)
        }
        track.frames += 1
        track.lastT = t
        track.boxes[frameIndex] = track.box
        if det.held { return }
        track.framesMissing = 0
        track.detections += 1
        track.attempts += 1
        track.lastSeenT = t
        track.detected.insert(frameIndex)
        if let lm = det.landmarks { track.landmarksByFrame[frameIndex] = lm }
        if track.keyframes.isEmpty || track.detections % keyframeEvery == 0 {
            track.keyframes.append(Keyframe(t: t, box: track.box))
        }
    }

    /// Drops the frames held after the last detection.
    private func trim(_ track: TrackState) {
        let last = track.detected.max() ?? -1
        for i in track.boxes.keys where i > last { track.boxes.removeValue(forKey: i) }
        track.frames = track.boxes.count
        track.lastT = track.lastSeenT
        if let k = track.keyframes.last, k.t < track.lastSeenT {
            track.keyframes.append(Keyframe(t: track.lastSeenT, box: track.box))
        }
    }
}

// MARK: - Per frame

public struct FrameTracking {
    public var cut: Bool
    public var live: [TrackState]
    /// Tracks that ended on this frame, by cut or by grace.
    public var ended: [TrackState]
}

/// Detection on a cadence (every `detectEvery`-th frame, or whenever no
/// track is alive), boxes held in between, cuts ending everything.
public final class FaceTracker {
    public let detector = VisionFaceDetector()
    public let tracker = Tracker()
    let detectEvery: Int
    let maxFaces: Int
    private var cuts = CutDetector()
    private var framesSeen = 0
    public private(set) var detectSeconds = 0.0
    public private(set) var trackSeconds = 0.0

    public init(detectEvery: Int = 2, maxFaces: Int = 8) {
        self.detectEvery = max(1, detectEvery)
        self.maxFaces = max(1, maxFaces)
    }

    public func process(_ frame: SampledFrame, view: BGRAFrame) throws -> FrameTracking {
        var ended: [TrackState] = []
        let cut = cuts.update(histogram: view.coarseHistogram())
        if cut {
            ended.append(contentsOf: tracker.live)
            tracker.cut()
        }
        var t0 = Date()
        var detections: [FaceDetection]
        if framesSeen % detectEvery == 0 || tracker.live.isEmpty {
            detections = try detector.detect(frame.pixelBuffer)
            detections.sort { $0.box.area > $1.box.area }
            detections = Array(detections.prefix(maxFaces))
        } else {
            detections = tracker.live.map { FaceDetection(box: $0.box, score: 1, landmarks: nil, held: true) }
        }
        detectSeconds += Date().timeIntervalSince(t0)
        t0 = Date()
        let before = Set(tracker.live.map(\.trackId))
        let live = tracker.update(frameIndex: frame.index, t: frame.time, detections: detections)
        let alive = Set(live.map(\.trackId))
        ended.append(contentsOf: tracker.finished.filter { before.contains($0.trackId) && !alive.contains($0.trackId) })
        trackSeconds += Date().timeIntervalSince(t0)
        framesSeen += 1
        return FrameTracking(cut: cut, live: live, ended: ended)
    }

    public func finish() -> [TrackState] { tracker.finish() }
}
