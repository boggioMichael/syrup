// Schema.swift — the data every part of LipReader exchanges.
//
// Mirrors example/shared-types/schema/analysis.schema.json and
// job.schema.json field for field (camelCase keys, the same optionality),
// the way schema.py and index.ts do on the other platforms. Numbers are
// kept at full precision while a result is built; `rounded()` applies the
// same 3-decimal (2 for boxes) rounding schema.py applies when it writes
// JSON, so the three implementations produce the same files.
import Foundation

/// Shown wherever a transcript is shown. Identical to schema.py NOTICE.
public let NOTICE = "Lip-reading output is probabilistic. Many sounds look identical on the lips, so this transcript is a best guess, not a verbatim record; words marked [like this?] are uncertain."

/// Rounds like Python's round(x, places) for the export formats.
func roundTo(_ value: Double, _ places: Int) -> Double {
    let f = pow(10.0, Double(places))
    return (value * f).rounded() / f
}

public enum Mode: String, Codable, CaseIterable, Sendable, Identifiable {
    case visual
    case audiovisual
    case audioAttributed = "audio-attributed"
    public var id: String { rawValue }
    /// The two modes that open the audio track. `visual` never does.
    public var usesAudio: Bool { self != .visual }
    /// The two modes that read lips. `audio-attributed` only watches who moves.
    public var readsLips: Bool { self != .audioAttributed }
}

// MARK: - Geometry

/// Pixels in the source frame, x/y top-left (never Vision's bottom-left).
public struct Box: Codable, Equatable, Sendable {
    public var x: Double, y: Double, w: Double, h: Double

    public init(x: Double, y: Double, w: Double, h: Double) {
        self.x = x; self.y = y; self.w = w; self.h = h
    }

    public var x2: Double { x + w }
    public var y2: Double { y + h }
    public var centre: (x: Double, y: Double) { (x + w / 2, y + h / 2) }
    public var area: Double { max(w, 0) * max(h, 0) }

    public func iou(_ other: Box) -> Double {
        let ix = max(0, min(x2, other.x2) - max(x, other.x))
        let iy = max(0, min(y2, other.y2) - max(y, other.y))
        let inter = ix * iy
        let union = area + other.area - inter
        return union > 0 ? inter / union : 0
    }

    public func scaled(_ factor: Double) -> Box { Box(x: x * factor, y: y * factor, w: w * factor, h: h * factor) }
    public func contains(_ px: Double, _ py: Double) -> Bool { px >= x && px <= x2 && py >= y && py <= y2 }
    func rounded() -> Box { Box(x: roundTo(x, 2), y: roundTo(y, 2), w: roundTo(w, 2), h: roundTo(h, 2)) }
}

// MARK: - Transcript

public struct Word: Codable, Equatable, Sendable {
    public var text: String
    public var start: Double
    public var end: Double
    public var confidence: Double
    /// Below the threshold: rendered marked, never as certain text (D4).
    public var uncertain: Bool
    /// What the model emitted before dictionary correction, when different.
    public var raw: String?

    public init(text: String, start: Double, end: Double, confidence: Double, uncertain: Bool, raw: String? = nil) {
        self.text = text; self.start = start; self.end = end; self.confidence = confidence; self.uncertain = uncertain
        self.raw = (raw == text) ? nil : raw
    }

    /// `[word?]` for an uncertain word, the word otherwise.
    public var rendered: String { uncertain ? "[\(text)?]" : text }

    func rounded() -> Word {
        Word(text: text, start: roundTo(start, 3), end: roundTo(end, 3), confidence: roundTo(confidence, 3), uncertain: uncertain, raw: raw)
    }
}

public struct TranscriptSegment: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var trackId: Int
    public var start: Double
    public var end: Double
    /// Words joined; uncertain words wrapped as [word?].
    public var text: String
    public var words: [Word]
    public var language: String
    public var confidence: Double
    public var mode: Mode
    public var model: String

    /// `text` and `confidence` are derived from the words, as in schema.py.
    public init(id: String, trackId: Int, words: [Word], language: String, mode: Mode, model: String) {
        self.id = id; self.trackId = trackId; self.words = words; self.language = language; self.mode = mode; self.model = model
        start = words.first?.start ?? 0
        end = words.last?.end ?? 0
        text = TranscriptSegment.render(words)
        confidence = words.isEmpty ? 0 : words.map(\.confidence).reduce(0, +) / Double(words.count)
    }

    public static func render(_ words: [Word]) -> String { words.map(\.rendered).joined(separator: " ") }

    func rounded() -> TranscriptSegment {
        var s = self
        s.start = roundTo(start, 3); s.end = roundTo(end, 3); s.confidence = roundTo(confidence, 3)
        s.words = words.map { $0.rounded() }
        return s
    }
}

public struct SpeakingSpan: Codable, Equatable, Sendable {
    public var start: Double
    public var end: Double
    public var probability: Double
    public init(start: Double, end: Double, probability: Double) { self.start = start; self.end = end; self.probability = probability }
    func rounded() -> SpeakingSpan { SpeakingSpan(start: roundTo(start, 3), end: roundTo(end, 3), probability: roundTo(probability, 3)) }
}

public struct Keyframe: Codable, Equatable, Sendable {
    public var t: Double
    public var box: Box
    public var mouth: Box?
    public init(t: Double, box: Box, mouth: Box? = nil) { self.t = t; self.box = box; self.mouth = mouth }
    func rounded() -> Keyframe { Keyframe(t: roundTo(t, 3), box: box.rounded(), mouth: mouth?.rounded()) }
}

/// One visible person, numbered within this video only: no identity, no
/// embedding that could be matched elsewhere.
public struct PersonTrack: Codable, Equatable, Sendable, Identifiable {
    public var trackId: Int
    public var label: String
    public var firstSeen: Double
    public var lastSeen: Double
    /// Where the face was, sampled; enough to draw a box that follows the person.
    public var keyframes: [Keyframe]
    public var speaking: [SpeakingSpan]
    public var detectedLanguage: String?
    public var segmentIds: [String]
    /// How sure the tracker is this was one person throughout.
    public var confidence: Double

    public var id: Int { trackId }

    public init(trackId: Int, firstSeen: Double, lastSeen: Double, keyframes: [Keyframe] = [], speaking: [SpeakingSpan] = [],
                detectedLanguage: String? = nil, segmentIds: [String] = [], confidence: Double = 1.0) {
        self.trackId = trackId; self.label = "Person \(trackId)"; self.firstSeen = firstSeen; self.lastSeen = lastSeen
        self.keyframes = keyframes; self.speaking = speaking; self.detectedLanguage = detectedLanguage
        self.segmentIds = segmentIds; self.confidence = confidence
    }

    enum CodingKeys: String, CodingKey { case trackId, label, firstSeen, lastSeen, keyframes, speaking, detectedLanguage, segmentIds, confidence }

    /// `segmentIds` is optional in index.ts; every producer writes it.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        trackId = try c.decode(Int.self, forKey: .trackId)
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? "Person \(trackId)"
        firstSeen = try c.decode(Double.self, forKey: .firstSeen)
        lastSeen = try c.decode(Double.self, forKey: .lastSeen)
        keyframes = try c.decode([Keyframe].self, forKey: .keyframes)
        speaking = try c.decode([SpeakingSpan].self, forKey: .speaking)
        detectedLanguage = try c.decodeIfPresent(String.self, forKey: .detectedLanguage)
        segmentIds = try c.decodeIfPresent([String].self, forKey: .segmentIds) ?? []
        confidence = try c.decode(Double.self, forKey: .confidence)
    }

    /// The box at time t, interpolated between the two nearest keyframes;
    /// the same binary search as `boxAt` in index.ts and schema.py.
    public func boxAt(_ t: Double) -> Box? {
        let k = keyframes
        guard let first = k.first, let last = k.last else { return nil }
        if t <= first.t { return first.box }
        if t >= last.t { return last.box }
        var lo = 0, hi = k.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if k[mid].t <= t { lo = mid } else { hi = mid }
        }
        let a = k[lo], b = k[hi]
        let f = b.t == a.t ? 0 : (t - a.t) / (b.t - a.t)
        return Box(x: a.box.x + (b.box.x - a.box.x) * f, y: a.box.y + (b.box.y - a.box.y) * f,
                   w: a.box.w + (b.box.w - a.box.w) * f, h: a.box.h + (b.box.h - a.box.h) * f)
    }

    func rounded() -> PersonTrack {
        var p = self
        p.firstSeen = roundTo(firstSeen, 3); p.lastSeen = roundTo(lastSeen, 3); p.confidence = roundTo(confidence, 3)
        p.keyframes = keyframes.map { $0.rounded() }; p.speaking = speaking.map { $0.rounded() }
        return p
    }
}

// MARK: - Models, video, language

public struct ModelInfo: Codable, Equatable, Sendable {
    public var name: String
    public var license: String
    public var languages: [String]
    /// "open" or a description of a closed vocabulary.
    public var vocabulary: String
    public var note: String?

    public init(name: String, license: String, languages: [String] = [], vocabulary: String = "open", note: String? = nil) {
        self.name = name; self.license = license; self.languages = languages; self.vocabulary = vocabulary
        self.note = (note?.isEmpty ?? true) ? nil : note
    }

    enum CodingKeys: String, CodingKey { case name, license, languages, vocabulary, note }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        license = try c.decode(String.self, forKey: .license)
        languages = try c.decodeIfPresent([String].self, forKey: .languages) ?? []
        vocabulary = try c.decodeIfPresent(String.self, forKey: .vocabulary) ?? "open"
        note = try c.decodeIfPresent(String.self, forKey: .note)
    }
}

public struct VideoInfo: Codable, Equatable, Sendable {
    /// File name as given; never a path.
    public var source: String?
    public var duration: Double
    public var fps: Double
    public var width: Int
    public var height: Int
    /// The frame rate the pipeline actually looked at.
    public var sampledFps: Double?

    public init(source: String? = nil, duration: Double, fps: Double, width: Int, height: Int, sampledFps: Double? = nil) {
        self.source = source; self.duration = duration; self.fps = fps; self.width = width; self.height = height; self.sampledFps = sampledFps
    }

    func rounded() -> VideoInfo {
        VideoInfo(source: source, duration: roundTo(duration, 3), fps: roundTo(fps, 3), width: width, height: height,
                  sampledFps: sampledFps.map { roundTo($0, 3) })
    }
}

public enum LanguageDetection: String, Codable, Sendable {
    case requested
    /// No visual language identification exists; the only available model's language was used.
    case assumed
    /// An ASR model identified it from sound (never in visual-only mode).
    case audioDetected = "audio-detected"
    case unavailable
}

public struct LanguageInfo: Codable, Equatable, Sendable {
    /// BCP-47 code or "auto".
    public var requested: String
    /// The language the models actually read; nil when nothing could be read.
    public var used: String?
    public var detection: LanguageDetection
    public var note: String?

    public init(requested: String, used: String?, detection: LanguageDetection, note: String? = nil) {
        self.requested = requested; self.used = used; self.detection = detection
        self.note = (note?.isEmpty ?? true) ? nil : note
    }

    enum CodingKeys: String, CodingKey { case requested, used, detection, note }

    /// `used` is required and nullable in the schema, so nil is written as null.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(requested, forKey: .requested)
        try c.encode(used, forKey: .used)
        try c.encode(detection, forKey: .detection)
        try c.encodeIfPresent(note, forKey: .note)
    }
}

public struct Models: Codable, Equatable, Sendable {
    public var faceDetector: String?
    public var vsr: ModelInfo?
    public var asr: ModelInfo?
    public init(faceDetector: String? = nil, vsr: ModelInfo? = nil, asr: ModelInfo? = nil) {
        self.faceDetector = faceDetector; self.vsr = vsr; self.asr = asr
    }
}

public struct Timing: Codable, Equatable, Sendable {
    public var processingSeconds: Double
    /// Processing seconds per second of video; < 1 is faster than real time.
    public var realtimeFactor: Double
    public var stages: [String: Double]

    public init(processingSeconds: Double, realtimeFactor: Double, stages: [String: Double] = [:]) {
        self.processingSeconds = processingSeconds; self.realtimeFactor = realtimeFactor; self.stages = stages
    }

    enum CodingKeys: String, CodingKey { case processingSeconds, realtimeFactor, stages }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processingSeconds = try c.decode(Double.self, forKey: .processingSeconds)
        realtimeFactor = try c.decode(Double.self, forKey: .realtimeFactor)
        stages = try c.decodeIfPresent([String: Double].self, forKey: .stages) ?? [:]
    }

    func rounded() -> Timing {
        Timing(processingSeconds: roundTo(processingSeconds, 3), realtimeFactor: roundTo(realtimeFactor, 3),
               stages: stages.mapValues { roundTo($0, 3) })
    }
}

// MARK: - The result

public struct AnalysisResult: Codable, Equatable, Sendable {
    public var version: String
    public var video: VideoInfo
    public var mode: Mode
    public var language: LanguageInfo
    public var tracks: [PersonTrack]
    public var segments: [TranscriptSegment]
    public var models: Models
    public var timing: Timing
    public var notice: String
    public var warnings: [String]

    public init(video: VideoInfo, mode: Mode, language: LanguageInfo, tracks: [PersonTrack], segments: [TranscriptSegment],
                models: Models, timing: Timing, warnings: [String] = []) {
        version = "1"; self.video = video; self.mode = mode; self.language = language; self.tracks = tracks
        self.segments = segments; self.models = models; self.timing = timing; notice = NOTICE; self.warnings = warnings
    }

    enum CodingKeys: String, CodingKey { case version, video, mode, language, tracks, segments, models, timing, notice, warnings }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decode(String.self, forKey: .version)
        video = try c.decode(VideoInfo.self, forKey: .video)
        mode = try c.decode(Mode.self, forKey: .mode)
        language = try c.decode(LanguageInfo.self, forKey: .language)
        tracks = try c.decode([PersonTrack].self, forKey: .tracks)
        segments = try c.decode([TranscriptSegment].self, forKey: .segments)
        models = try c.decodeIfPresent(Models.self, forKey: .models) ?? Models()
        timing = try c.decode(Timing.self, forKey: .timing)
        notice = try c.decode(String.self, forKey: .notice)
        warnings = try c.decodeIfPresent([String].self, forKey: .warnings) ?? []
    }

    public func track(_ id: Int) -> PersonTrack? { tracks.first { $0.trackId == id } }

    /// "Person n", or "Unattributed" for the -1 of words nobody's mouth claimed.
    public func label(forTrack id: Int) -> String {
        track(id)?.label ?? (id < 0 ? "Unattributed" : "Person \(id)")
    }

    /// The same rounding schema.py applies in to_dict.
    public func rounded() -> AnalysisResult {
        var r = self
        r.video = video.rounded(); r.tracks = tracks.map { $0.rounded() }; r.segments = segments.map { $0.rounded() }
        r.timing = timing.rounded()
        return r
    }
}

// MARK: - Options and jobs (job.schema.json)

/// "all", "current" (the largest face when analysis starts) or explicit track ids.
public enum Speakers: Codable, Equatable, Sendable {
    case all, current
    case tracks([Int])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            switch s {
            case "all": self = .all
            case "current": self = .current
            default: throw DecodingError.dataCorruptedError(in: c, debugDescription: "speakers must be \"all\", \"current\" or a list of track ids")
            }
        } else {
            self = .tracks(try c.decode([Int].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .all: try c.encode("all")
        case .current: try c.encode("current")
        case .tracks(let ids): try c.encode(ids)
        }
    }
}

public struct Options: Codable, Equatable, Sendable {
    public var mode: Mode = .visual
    public var language: String = "auto"
    public var speakers: Speakers = .all
    public var sampleFps: Double?
    public var maxFaces: Int = 8
    public var startTime: Double = 0
    public var endTime: Double?
    public var uncertainBelow: Double = 0.5

    public init(mode: Mode = .visual, language: String = "auto", speakers: Speakers = .all, sampleFps: Double? = nil,
                maxFaces: Int = 8, startTime: Double = 0, endTime: Double? = nil, uncertainBelow: Double = 0.5) {
        self.mode = mode; self.language = language.lowercased(); self.speakers = speakers; self.sampleFps = sampleFps
        self.maxFaces = maxFaces; self.startTime = startTime; self.endTime = endTime; self.uncertainBelow = uncertainBelow
    }

    enum CodingKeys: String, CodingKey { case mode, language, speakers, sampleFps, maxFaces, startTime, endTime, uncertainBelow }

    /// Every field has a default in the schema, so every key is optional.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        mode = try c.decodeIfPresent(Mode.self, forKey: .mode) ?? .visual
        language = (try c.decodeIfPresent(String.self, forKey: .language) ?? "auto").lowercased()
        speakers = try c.decodeIfPresent(Speakers.self, forKey: .speakers) ?? .all
        sampleFps = try c.decodeIfPresent(Double.self, forKey: .sampleFps)
        maxFaces = try c.decodeIfPresent(Int.self, forKey: .maxFaces) ?? 8
        startTime = try c.decodeIfPresent(Double.self, forKey: .startTime) ?? 0
        endTime = try c.decodeIfPresent(Double.self, forKey: .endTime)
        uncertainBelow = try c.decodeIfPresent(Double.self, forKey: .uncertainBelow) ?? 0.5
    }
}

public enum JobStatus: String, Codable, Sendable { case queued, running, done, failed, deleted }

/// Where the frames were processed; shown to the user (D6).
public enum Processing: String, Codable, Sendable { case local, server }

public struct Job: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var status: JobStatus
    public var progress: Double
    public var createdAt: String
    public var finishedAt: String?
    public var expiresAt: String?
    public var options: Options
    public var result: AnalysisResult?
    public var error: String?
    public var processing: Processing?
    /// The API adds the file name or URL it was given; not part of the schema.
    public var source: String?
}

/// One row of the honest availability table (`GET /languages`, index.ts LanguageAvailability).
public struct LanguageAvailability: Codable, Equatable, Sendable, Identifiable {
    public struct Visual: Codable, Equatable, Sendable {
        public var available: Bool
        public var model: String?
        public var license: String?
        public var note: String?
        public init(available: Bool, model: String? = nil, license: String? = nil, note: String? = nil) {
            self.available = available; self.model = model; self.license = license; self.note = note
        }
    }
    public struct Audio: Codable, Equatable, Sendable {
        public var available: Bool
        public var model: String?
        public var note: String?
        public init(available: Bool, model: String? = nil, note: String? = nil) { self.available = available; self.model = model; self.note = note }
    }
    public var code: String
    public var name: String
    public var visual: Visual
    public var audio: Audio?
    public var id: String { code }
    public init(code: String, name: String, visual: Visual, audio: Audio? = nil) { self.code = code; self.name = name; self.visual = visual; self.audio = audio }
}

public struct HealthResponse: Codable, Equatable, Sendable {
    public var ok: Bool
    public var version: String
    public var processing: Processing
    public var languages: [LanguageAvailability]
    public var fetchers: [String]
}

// MARK: - JSON

public enum JSONCoding {
    /// The encoder the exporters and the API client share: stable key order, readable.
    public static var encoder: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return e
    }
    public static var decoder: JSONDecoder { JSONDecoder() }
}
