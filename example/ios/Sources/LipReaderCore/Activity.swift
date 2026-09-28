// Activity.swift — is this person speaking right now? (activity.py), and
// which frames to hand the lip reader (pipeline.py _windows).
//
// Two signals per tracked mouth, both from pixels: how the openness of the
// mouth varies over a short window (speech opens and closes the mouth
// several times a second; a still face does not), and how much the mouth
// crop changes from frame to frame relative to the face as a whole (so that
// a head turning does not count as talking). The two are combined into a
// probability with a fixed, documented calibration; spans of speech are cut
// where it drops. Constants are those of activity.py.
import Foundation

public enum Activity {
    /// Frames each side of the rolling window; at 25 fps ±0.36 s.
    public static let window = 9

    public struct Config: Sendable {
        /// Rolling std of openness that counts as "clearly speaking".
        public var opennessScale = 0.03
        /// Mouth-vs-face motion excess that counts as "clearly speaking".
        public var motionScale = 0.05
        /// Probability above which a frame is speech.
        public var threshold = 0.5
        public var minSpanSeconds = 0.2
        /// Pauses inside a sentence are shorter than this.
        public var mergeGapSeconds = 0.6
        public var padSeconds = 0.16
        public init() {}
    }

    /// Population standard deviation over ±window frames (numpy's default ddof=0).
    static func rollingStd(_ values: [Double], window: Int = window) -> [Double] {
        let n = values.count
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let lo = max(0, i - window), hi = min(n, i + window + 1)
            let slice = values[lo..<hi]
            let mean = slice.reduce(0, +) / Double(slice.count)
            let variance = slice.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(slice.count)
            out[i] = variance.squareRoot()
        }
        return out
    }

    static func rollingMean(_ values: [Double], window: Int = window) -> [Double] {
        let n = values.count
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let lo = max(0, i - window), hi = min(n, i + window + 1)
            out[i] = values[lo..<hi].reduce(0, +) / Double(hi - lo)
        }
        return out
    }

    /// Per-frame probability of speech from the three per-frame signals (all
    /// the same length; motions are mean absolute frame differences).
    public static func speakingProbability(openness: [Double], mouthMotion: [Double], faceMotion: [Double],
                                           config: Config = Config()) -> [Double] {
        let n = openness.count
        guard n > 0, mouthMotion.count == n, faceMotion.count == n else { return [] }
        let variation = rollingStd(openness)
        let excess = (0..<n).map { max(0, mouthMotion[$0] - faceMotion[$0]) }
        let motion = rollingMean(excess)
        return (0..<n).map { i in
            let a = variation[i] / config.opennessScale, b = motion[i] / config.motionScale
            let evidence = max(a, b) + 0.5 * min(a, b)
            return 1 - exp(-evidence)
        }
    }

    /// Continuous stretches of speech, short gaps bridged, each padded a
    /// little so the first and last visemes are inside.
    public static func spans(probability: [Double], times: [Double], config: Config = Config()) -> [SpeakingSpan] {
        let n = probability.count
        guard n > 0, times.count == n else { return [] }
        let active = probability.map { $0 >= config.threshold }
        var runs: [(Int, Int)] = []
        var start: Int?
        for (i, on) in active.enumerated() {
            if on, start == nil { start = i } else if !on, let s = start { runs.append((s, i - 1)); start = nil }
        }
        if let s = start { runs.append((s, n - 1)) }
        let fps = frameRate(times)
        var merged: [(Int, Int)] = []
        for (s, e) in runs {
            if let last = merged.last, Double(s - last.1) / fps <= config.mergeGapSeconds {
                merged[merged.count - 1] = (last.0, e)
            } else {
                merged.append((s, e))
            }
        }
        let pad = Int((config.padSeconds * fps).rounded())
        return merged.compactMap { (s, e) in
            guard Double(e - s + 1) / fps >= config.minSpanSeconds else { return nil }
            let s2 = max(0, s - pad), e2 = min(n - 1, e + pad)
            let mean = probability[s...e].reduce(0, +) / Double(e - s + 1)
            return SpeakingSpan(start: times[s2], end: times[e2], probability: mean)
        }
    }

    /// 1 / median frame interval; 25 when there is only one frame.
    public static func frameRate(_ times: [Double]) -> Double {
        guard times.count > 1 else { return 25 }
        let gaps = zip(times.dropFirst(), times).map { $0 - $1 }.sorted()
        let median = gaps.count % 2 == 1 ? gaps[gaps.count / 2] : (gaps[gaps.count / 2 - 1] + gaps[gaps.count / 2]) / 2
        return 1 / max(median, 1e-6)
    }

    /// First index whose time is >= t (numpy searchsorted, side="left").
    public static func position(of t: Double, in times: [Double]) -> Int {
        var lo = 0, hi = times.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if times[mid] < t { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    /// The shortest clip worth reading.
    public static let minChunkFrames = 8

    public struct Window: Equatable, Sendable {
        public var first: Int
        public var last: Int
        public var probability: Double
        public init(first: Int, last: Int, probability: Double) { self.first = first; self.last = last; self.probability = probability }
    }

    /// Frame windows to read, as (first, last, probability), from pipeline.py
    /// _windows. A short span is widened with the silence around it up to the
    /// model's clip length, because the model was trained on utterances with
    /// their silences and reads a bare fragment worse; it never takes another
    /// span's speech. Neighbouring spans that fit in one clip are read
    /// together: a pause of up to a second is still inside one sentence for
    /// the model. A long span is cut into clips of the model's length.
    public static func readWindows(spans: [SpeakingSpan], times: [Double], readUntil: Int, maxFrames: Int, fps: Double) -> [Window] {
        guard !times.isEmpty, maxFrames > 0 else { return [] }
        let limit = times.count - 1
        var bounds: [(Int, Int)] = []
        var merged: [SpeakingSpan] = []
        for span in spans {
            let s = max(position(of: span.start, in: times), readUntil), e = min(position(of: span.end, in: times), limit)
            if let last = bounds.last, Double(s - last.1) / fps <= 1.0, e - last.0 + 1 <= maxFrames {
                bounds[bounds.count - 1] = (last.0, e)
                let prev = merged[merged.count - 1]
                merged[merged.count - 1] = SpeakingSpan(start: prev.start, end: span.end, probability: max(prev.probability, span.probability))
            } else {
                bounds.append((s, e))
                merged.append(span)
            }
        }
        var out: [Window] = []
        for (k, ((s, e), span)) in zip(bounds, merged).enumerated() {
            let length = e - s + 1
            if length < minChunkFrames { continue }
            if length < maxFrames {
                let lo = k > 0 ? bounds[k - 1].1 + 1 : readUntil
                let hi = k + 1 < bounds.count ? bounds[k + 1].0 - 1 : limit
                let room = maxFrames - length
                var before = min(room / 2, s - lo)
                let after = min(room - before, hi - e)
                before = min(room - after, s - lo)
                out.append(Window(first: s - before, last: e + after, probability: span.probability))
            } else {
                var c0 = s
                while c0 <= e {
                    let c1 = min(c0 + maxFrames - 1, e)
                    if c1 - c0 + 1 >= minChunkFrames { out.append(Window(first: c0, last: c1, probability: span.probability)) }
                    c0 += maxFrames
                }
            }
        }
        return out
    }
}
