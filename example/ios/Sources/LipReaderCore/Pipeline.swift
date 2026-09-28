// Pipeline.swift — the pipeline, end to end (pipeline.py):
//
//     frames -> face detection -> tracking -> mouth -> speaking activity
//            -> visual speech recognition (batched per model)
//            -> words with time and confidence -> segments per person
//            -> (audio modes) ASR words attributed to the person whose mouth moved
//
// `Analyzer` takes frames one at a time and produces an AnalysisResult;
// `LipReaderPipeline.analyze(url:options:)` wraps it for a file. Visual mode
// never touches the audio track: the audio stage is reached only through
// `Mode.usesAudio`, and the frame source never opens audio in any mode.
import Foundation

/// Everything remembered about one track while it is alive.
final class PersonState {
    let mouth = MouthTracker()
    var indices: [Int] = []
    var times: [Double] = []
    var openness: [Double] = []
    var mouthMotion: [Double] = []
    var faceMotion: [Double] = []
    /// 100x50x3 bytes per frame; nil once a frame's speech has been read.
    var crops: [[UInt8]?] = []
    var mouthBoxes: [Int: Box] = [:]
    var lastCropGray: [Float]?
    var lastFaceGray: [Float]?
    var probability: [Double]?
    var spans: [SpeakingSpan] = []
    /// Frames before this position have been read already.
    var readUntil = 0
}

struct LanguageResolution {
    var spec: ModelSpec?
    var used: String?
    var detection: LanguageDetection
    var note = ""
    var warning: String?

    /// pipeline.py _resolve_language: which visual model, if any, and how the
    /// language was decided. "auto" in a lip-reading mode means "the one
    /// visual model available", reported as `assumed` (D4).
    static func resolve(options: Options, registry: LanguageRegistry, given: ModelSpec?) -> LanguageResolution {
        let requested = options.language
        guard options.mode.readsLips else {
            return LanguageResolution(spec: nil, used: requested == "auto" ? nil : requested, detection: requested == "auto" ? .assumed : .requested)
        }
        do {
            if let given {
                return LanguageResolution(spec: given, used: given.info.languages.first ?? requested, detection: requested == "auto" ? .assumed : .requested)
            }
            if requested == "auto" {
                guard let first = registry.availableLanguages().first else {
                    throw LanguageUnavailable(language: "auto", candidates: registry.candidates("en"))
                }
                return LanguageResolution(spec: try registry.resolve(first), used: first, detection: .assumed,
                                          note: "no visual language-identification model exists publicly; the only visual model available here reads \(first), so that is what was used")
            }
            return LanguageResolution(spec: try registry.resolve(requested), used: requested, detection: .requested)
        } catch {
            return LanguageResolution(spec: nil, used: nil, detection: .unavailable, warning: "visual reading disabled: \(error.localizedDescription)")
        }
    }
}

public actor Analyzer {
    /// Per track: read closed speech spans and drop their crops beyond this.
    public static let flushFrames = 1500

    let options: Options
    let video: VideoInfo
    let registry: LanguageRegistry
    let faces: FaceTracker
    let activityConfig: Activity.Config
    let preferOnDeviceSpeech: Bool
    let progress: (@Sendable (Double) -> Void)?
    let started = Date()
    private var people: [Int: PersonState] = [:]
    private var ended: [Int: PersonState] = [:]
    private var framesSeen = 0
    private var warnings: [String] = []
    private var stages: [String: Double] = ["detect": 0, "track": 0, "mouth": 0, "vsr": 0, "asr": 0]
    private var segments: [TranscriptSegment] = []
    private var segmentCount = 0
    private var languageUsed: String?
    private var languageDetection: LanguageDetection
    private var languageNote: String
    private var models: Models
    private var vsr: VisualSpeechModel?
    private var vsrSpec: ModelSpec?
    private var currentTrack: Int?

    /// `vsrSpec`/`model` let a caller (or a test) inject the visual model;
    /// otherwise the registry resolves the language.
    public init(options: Options, video: VideoInfo, registry: LanguageRegistry = .shared, vsrSpec: ModelSpec? = nil,
                model: VisualSpeechModel? = nil, activityConfig: Activity.Config = Activity.Config(), detectEvery: Int = 2,
                preferOnDeviceSpeech: Bool = true, progress: (@Sendable (Double) -> Void)? = nil) {
        self.options = options
        self.video = video
        self.registry = registry
        self.activityConfig = activityConfig
        self.preferOnDeviceSpeech = preferOnDeviceSpeech
        self.progress = progress
        let tracker = FaceTracker(detectEvery: detectEvery, maxFaces: options.maxFaces)
        faces = tracker
        let resolved = LanguageResolution.resolve(options: options, registry: registry, given: vsrSpec)
        self.vsrSpec = resolved.spec
        vsr = model
        languageUsed = resolved.used
        languageDetection = resolved.detection
        languageNote = resolved.note
        models = Models(faceDetector: tracker.detector.name, vsr: resolved.spec?.info)
        warnings = resolved.warning.map { [$0] } ?? []
    }

    // MARK: Frames

    public func push(_ frame: SampledFrame) async throws {
        let endedTracks: [TrackState] = try BGRAFrame.withLocked(frame.pixelBuffer) { view in
            let tracking = try faces.process(frame, view: view)
            let t0 = Date()
            for track in tracking.live where track.boxes[frame.index] != nil {
                let person: PersonState
                if let existing = people[track.trackId] { person = existing } else { person = PersonState(); people[track.trackId] = person }
                observe(person, track: track, frame: frame, view: view)
            }
            if options.speakers == .current, currentTrack == nil, let largest = tracking.live.max(by: { $0.box.area < $1.box.area }) {
                currentTrack = largest.trackId
            }
            stages["mouth", default: 0] += Date().timeIntervalSince(t0)
            return tracking.ended
        }
        for track in endedTracks { try await endTrack(track) }
        framesSeen += 1
        if let progress, video.duration > 0 { progress(min(0.95, frame.time / video.duration)) }
        for (trackId, person) in people where person.indices.count - person.readUntil > Self.flushFrames {
            try await readClosed(person, trackId: trackId, keepTail: 40)
        }
    }

    private func observe(_ person: PersonState, track: TrackState, frame: SampledFrame, view: BGRAFrame) {
        let estimate = person.mouth.update(view, face: track.box, landmarks: track.landmarksByFrame[frame.index])
        let crop = person.mouth.crop(view)
        let faceSmall = FaceMotion.sample(view, face: track.box)
        let mouthMotion = person.lastCropGray.map { FaceMotion.meanAbsDiff(crop.gray, $0) } ?? 0
        let faceMotion = person.lastFaceGray.map { FaceMotion.meanAbsDiff(faceSmall, $0) } ?? 0
        person.lastCropGray = crop.gray
        person.lastFaceGray = faceSmall
        person.indices.append(frame.index)
        person.times.append(frame.time)
        person.openness.append(estimate.openness)
        person.mouthMotion.append(mouthMotion)
        person.faceMotion.append(faceMotion)
        person.crops.append(crop.rgb)
        if let mouthBox = person.mouth.box() {
            person.mouthBoxes[frame.index] = mouthBox
            if let last = track.keyframes.last, last.t == frame.time {
                track.keyframes[track.keyframes.count - 1].mouth = mouthBox
            }
        }
    }

    // MARK: Reading

    private func selected(_ trackId: Int) -> Bool {
        switch options.speakers {
        case .all: return true
        case .current: return trackId == currentTrack
        case .tracks(let ids): return ids.contains(trackId)
        }
    }

    private func speakingProbability(_ person: PersonState) -> [Double] {
        Activity.speakingProbability(openness: person.openness, mouthMotion: person.mouthMotion, faceMotion: person.faceMotion, config: activityConfig)
    }

    /// Reads the speech spans that are safely over and forgets their crops.
    private func readClosed(_ person: PersonState, trackId: Int, keepTail: Int) async throws {
        let p = speakingProbability(person)
        let spans = Activity.spans(probability: p, times: person.times, config: activityConfig)
        let limit = person.indices.count - keepTail
        let closed = spans.filter { Activity.position(of: $0.end, in: person.times) < limit }
        try await readSpans(person, trackId: trackId, spans: closed)
        if let lastSpan = closed.last {
            let last = Activity.position(of: lastSpan.end, in: person.times)
            if person.readUntil <= last {
                for i in person.readUntil...last { person.crops[i] = nil }
            }
            person.readUntil = last + 1
        }
    }

    private func readSpans(_ person: PersonState, trackId: Int, spans: [SpeakingSpan]) async throws {
        person.spans.append(contentsOf: spans)
        guard let spec = vsrSpec, options.mode != .audioAttributed, selected(trackId) else { return }
        let fps = video.sampledFps ?? (video.fps > 0 ? video.fps : 25)
        var clips: [MouthClip] = []
        for w in Activity.readWindows(spans: spans, times: person.times, readUntil: person.readUntil, maxFrames: spec.maxFrames, fps: fps) {
            let frames = person.crops[w.first...w.last].compactMap { $0 }
            guard frames.count == w.last - w.first + 1 else { continue }
            clips.append(MouthClip(trackId: trackId, frameIndices: Array(person.indices[w.first...w.last]),
                                   times: Array(person.times[w.first...w.last]), frames: frames, spanProbability: w.probability))
        }
        if !clips.isEmpty { try await runVSR(clips, spec: spec) }
    }

    /// One model call per batch (D9). A model that cannot be loaded disables
    /// visual reading with a warning rather than failing the analysis: the
    /// result then has no visual text, never a guess.
    private func runVSR(_ clips: [MouthClip], spec: ModelSpec) async throws {
        if vsr == nil {
            do { vsr = try await registry.load(spec) } catch {
                vsrSpec = nil
                models.vsr = nil
                languageUsed = nil
                languageDetection = .unavailable
                warnings.append("visual reading disabled: the model could not be loaded (\(error.localizedDescription))")
                return
            }
        }
        guard let vsr else { return }
        let t0 = Date()
        let decoded = try await vsr.predict(batch: clips)
        stages["vsr", default: 0] += Date().timeIntervalSince(t0)
        for (clip, utterance) in zip(clips, decoded) { addSegment(clip, utterance, spec: spec) }
    }

    private func addSegment(_ clip: MouthClip, _ decoded: DecodedUtterance, spec: ModelSpec) {
        guard !decoded.words.isEmpty else { return }
        let fps = video.sampledFps ?? (video.fps > 0 ? video.fps : 25)
        let words = decoded.words.map { w -> Word in
            let start = clip.times[min(w.startFrame, clip.times.count - 1)]
            let end = clip.times[min(w.endFrame, clip.times.count - 1)] + 1 / fps
            let confidence = max(0, min(1, w.confidence))
            return Word(text: w.text, start: start, end: end, confidence: confidence, uncertain: confidence < options.uncertainBelow, raw: w.raw)
        }
        segmentCount += 1
        segments.append(TranscriptSegment(id: "s\(segmentCount)", trackId: clip.trackId, words: words,
                                          language: languageUsed ?? "?", mode: .visual, model: spec.key))
    }

    private func endTrack(_ track: TrackState) async throws {
        guard let person = people.removeValue(forKey: track.trackId) else { return }
        let p = speakingProbability(person)
        person.probability = p
        let spans = Activity.spans(probability: p, times: person.times, config: activityConfig)
            .filter { Activity.position(of: $0.end, in: person.times) >= person.readUntil }
        try await readSpans(person, trackId: track.trackId, spans: spans)
        person.crops = []
        ended[track.trackId] = person
    }

    // MARK: Finish

    /// `audioURL` is the video file again, passed only by the audio modes and
    /// only when the file has an audio track; nil means the audio stage has
    /// nothing to read.
    public func finish(audioURL: URL?) async throws -> AnalysisResult {
        let tracks = faces.finish()
        for track in tracks { try await endTrack(track) }
        var personTracks = tracks.map { track -> PersonTrack in
            var pt = track.toPersonTrack()
            if let person = ended[track.trackId] { pt.speaking = person.spans.sorted { $0.start < $1.start } }
            pt.detectedLanguage = segments.contains { $0.trackId == track.trackId } ? languageUsed : nil
            return pt
        }
        if options.mode.usesAudio { await audioStage(audioURL, &personTracks) }
        for i in personTracks.indices {
            personTracks[i].segmentIds = segments.filter { $0.trackId == personTracks[i].trackId }.map(\.id)
        }
        segments.sort { ($0.start, $0.trackId) < ($1.start, $1.trackId) }
        stages["detect"] = faces.detectSeconds
        stages["track"] = faces.trackSeconds
        progress?(1)
        let seconds = Date().timeIntervalSince(started)
        let timing = Timing(processingSeconds: seconds, realtimeFactor: video.duration > 0 ? seconds / video.duration : 0, stages: stages)
        let language = LanguageInfo(requested: options.language, used: languageUsed, detection: languageDetection, note: languageNote)
        return AnalysisResult(video: video, mode: options.mode, language: language, tracks: personTracks, segments: segments,
                              models: models, timing: timing, warnings: warnings).rounded()
    }

    // MARK: Audio modes

    private func audioStage(_ url: URL?, _ personTracks: inout [PersonTrack]) async {
        func skipped(_ reason: String) {
            warnings.append(reason + (options.mode == .audiovisual ? "; the result is visual-only" : ""))
            if options.mode == .audioAttributed { languageUsed = nil; languageDetection = .unavailable }
        }
        guard let url else { skipped("no audio track: the audio stage had nothing to read"); return }
        let requested: String? = options.language == "auto" ? nil : options.language
        let t0 = Date()
        let asr: AsrResult
        do {
            asr = try await SpeechTranscriber.transcribe(url: url, language: requested, preferOnDevice: preferOnDeviceSpeech)
        } catch {
            skipped("speech recognition failed (\(error.localizedDescription)); the audio stage was skipped")
            return
        }
        stages["asr", default: 0] += Date().timeIntervalSince(t0)
        models.asr = asr.info
        if options.language == "auto" {
            languageUsed = asr.language
            languageDetection = .assumed
            languageNote = "Apple's speech recogniser does not identify languages; the device locale (\(asr.language)) was assumed"
        }
        if !asr.onDevice, video.duration > 60 {
            warnings.append("server-based speech recognition reads about one minute of audio per request; later words may be missing")
        }
        let end = options.endTime ?? .infinity
        let words = asr.words.filter { $0.end >= options.startTime && $0.start <= end }
        if words.isEmpty { warnings.append("the speech recogniser returned no words for this audio") }
        var newSegments: [TranscriptSegment] = []
        for (trackId, group) in attribute(words) {
            segmentCount += 1
            let ws = group.map { Word(text: $0.text, start: $0.start, end: $0.end, confidence: $0.confidence, uncertain: $0.confidence < options.uncertainBelow) }
            newSegments.append(TranscriptSegment(id: "s\(segmentCount)", trackId: trackId, words: ws, language: asr.language,
                                                 mode: options.mode, model: asr.info.name))
        }
        if options.mode == .audioAttributed {
            segments = newSegments
        } else {
            // Fusion: sound is the more reliable reading where it exists; the
            // visual reading stays where the audio said nothing for that person.
            let kept = segments.filter { s in !newSegments.contains { n in n.trackId == s.trackId && n.start < s.end && s.start < n.end } }
            segments = kept + newSegments
        }
        for i in personTracks.indices where segments.contains(where: { $0.trackId == personTracks[i].trackId }) {
            personTracks[i].detectedLanguage = asr.language
        }
    }

    /// Each spoken word goes to the person whose mouth was moving most while
    /// it was said; nobody moving enough means unattributed (-1). Consecutive
    /// words of one person form a segment.
    private func attribute(_ words: [AsrWord]) -> [(Int, [AsrWord])] {
        var groups: [(Int, [AsrWord])] = []
        for w in words {
            var best = -1, bestP = 0.0
            for (trackId, person) in ended.sorted(by: { $0.key < $1.key }) {
                guard let p = person.probability, !person.times.isEmpty else { continue }
                let lo = Activity.position(of: w.start - 0.1, in: person.times)
                let hi = Activity.position(of: w.end + 0.1, in: person.times)
                guard hi > lo else { continue }
                let mean = p[lo..<hi].reduce(0, +) / Double(hi - lo)
                if mean > bestP { best = trackId; bestP = mean }
            }
            if bestP < 0.35 { best = -1 }
            if let last = groups.last, last.0 == best, let previous = last.1.last, w.start - previous.end <= 0.6 {
                groups[groups.count - 1].1.append(w)
            } else {
                groups.append((best, [w]))
            }
        }
        return groups
    }
}

// MARK: - A file, start to finish

public enum LipReaderPipeline {
    /// Probes the file, samples frames at the model's native rate (25 fps
    /// for LipNet) unless `options.sampleFps` says otherwise, pushes every
    /// frame through the analyzer and finishes with the audio stage when the
    /// mode asks for it and the file has sound.
    public static func analyze(url: URL, options: Options, registry: LanguageRegistry = .shared, preferOnDeviceSpeech: Bool = true,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> AnalysisResult {
        let probe = try await FrameSource.probe(url)
        var native = 25.0
        if options.mode.readsLips {
            let code = options.language == "auto" ? registry.availableLanguages().first : options.language
            if let code, let spec = try? registry.resolve(code) { native = spec.nativeFps }
        }
        let source = try await FrameSource(url: url, probe: probe, sampleFps: options.sampleFps ?? native,
                                           start: options.startTime, end: options.endTime)
        let analyzer = Analyzer(options: options, video: source.info, registry: registry, preferOnDeviceSpeech: preferOnDeviceSpeech, progress: progress)
        do {
            while let frame = try source.next() {
                try Task.checkCancellation()
                try await analyzer.push(frame)
            }
        } catch {
            source.cancel()
            throw error
        }
        let audioURL: URL? = options.mode.usesAudio && probe.hasAudio ? url : nil
        return try await analyzer.finish(audioURL: audioURL)
    }
}
