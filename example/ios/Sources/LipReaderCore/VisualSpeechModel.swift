// VisualSpeechModel.swift — the visual speech recogniser interface
// (vsr/base.py), CTC decoding kept honest (decode.py), LipNet through Core
// ML (vsr/lipnet.py) and the inference API as a remote reader (docs/api.md).
//
// Every word remembers which frames emitted its letters (its timing) and
// how sure the network was at those frames (its confidence). A dictionary
// correction may replace a word; the replaced word keeps its raw form and
// loses some confidence, because a correction is a guess about a guess.
import CoreML
import Foundation

/// A mouth clip for the model: T frames of the 100x50 crop, bytes 0...255.
public struct MouthClip: Sendable {
    public var trackId: Int
    public var frameIndices: [Int]
    public var times: [Double]
    public var frames: [[UInt8]]
    public var spanProbability: Double

    public init(trackId: Int, frameIndices: [Int], times: [Double], frames: [[UInt8]], spanProbability: Double) {
        self.trackId = trackId; self.frameIndices = frameIndices; self.times = times; self.frames = frames; self.spanProbability = spanProbability
    }
}

public struct DecodedWord: Equatable, Sendable {
    public var text: String
    public var startFrame: Int
    /// Inclusive.
    public var endFrame: Int
    public var confidence: Double
    public var raw: String?

    public init(text: String, startFrame: Int, endFrame: Int, confidence: Double, raw: String? = nil) {
        self.text = text; self.startFrame = startFrame; self.endFrame = endFrame; self.confidence = confidence; self.raw = raw
    }
}

public struct DecodedUtterance: Equatable, Sendable {
    public var words: [DecodedWord]
    public init(words: [DecodedWord] = []) { self.words = words }
    public var text: String { words.map(\.text).joined(separator: " ") }
}

/// A model reads a batch of mouth clips and returns timed, scored words for
/// each; everything else in the pipeline is the same whatever the model.
public protocol VisualSpeechModel: AnyObject {
    var key: String { get }
    var info: ModelInfo { get }
    var nativeFps: Double { get }
    /// The longest clip the model should see at once.
    var maxFrames: Int { get }
    func predict(batch: [MouthClip]) async throws -> [DecodedUtterance]
}

// MARK: - CTC

public enum CTCDecoder {
    /// Greedy CTC over `probabilities` (frames x classes, row-major): argmax
    /// per frame, repeats collapsed, blanks dropped. `alphabet[i]` is the
    /// symbol for class i; `space` is the class that separates words (nil
    /// when the alphabet has word pieces).
    public static func bestPath(probabilities: [Float], frames: Int, classes: Int, alphabet: [String], blank: Int, space: Int?) -> DecodedUtterance {
        var words: [DecodedWord] = []
        var letters: [String] = [], frameIds: [Int] = [], confidences: [Double] = []
        var previous = -1
        func flush() {
            if !letters.isEmpty {
                words.append(DecodedWord(text: letters.joined(), startFrame: frameIds[0], endFrame: frameIds[frameIds.count - 1],
                                         confidence: confidences.reduce(0, +) / Double(confidences.count)))
            }
            letters.removeAll(); frameIds.removeAll(); confidences.removeAll()
        }
        for t in 0..<frames {
            var best = 0
            var peak = -Float.infinity
            for c in 0..<classes where probabilities[t * classes + c] > peak { peak = probabilities[t * classes + c]; best = c }
            if best != previous, best != blank {
                if let space, best == space {
                    flush()
                } else {
                    letters.append(alphabet[best]); frameIds.append(t); confidences.append(Double(peak))
                }
            }
            previous = best
        }
        flush()
        return DecodedUtterance(words: words)
    }

    /// Applies a word-level dictionary correction, remembering the raw word.
    /// A word the dictionary still does not know after correction (a closed
    /// vocabulary model emitting letters that spell nothing) loses more.
    public static func correct(_ decoded: DecodedUtterance, corrector: (String) -> String, known: ((String) -> Bool)? = nil,
                               penalty: Double = 0.8, unknownPenalty: Double = 0.5) -> DecodedUtterance {
        DecodedUtterance(words: decoded.words.map { w in
            let fixed = corrector(w.text)
            var confidence = w.confidence
            if fixed != w.text { confidence *= penalty }
            if let known, !known(fixed) { confidence *= unknownPenalty }
            return DecodedWord(text: fixed, startFrame: w.startFrame, endFrame: w.endFrame, confidence: confidence,
                               raw: fixed != w.text ? w.text : nil)
        })
    }
}

// MARK: - GRID dictionary

/// Norvig's corrector over the GRID dictionary, as the original LipNet code
/// uses it (lipnet/utils/spell.py). The counts are the word frequencies of
/// LipNet's common/dictionaries/grid.txt (33,000 sentences), so a tie between
/// candidates resolves the same way.
public struct GridDictionary: Sendable {
    public static let letters = "abcdefghijklmnopqrstuvwxyz"
    public let counts: [String: Int]
    let total: Int

    public static let grid: GridDictionary = {
        var c: [String: Int] = ["bin": 8203, "lay": 8409, "place": 8284, "set": 8104,
                                "blue": 8250, "green": 8250, "red": 8250, "white": 8250,
                                "at": 8256, "by": 8244, "in": 8256, "with": 8244,
                                "again": 8250, "now": 8250, "please": 8250, "soon": 8250,
                                "zero": 3300, "one": 3300, "two": 3300, "three": 3300, "four": 3300,
                                "five": 3300, "six": 3300, "seven": 3300, "eight": 3300, "nine": 3300]
        for letter in letters where letter != "w" { c[String(letter)] = 1320 } // GRID spells no "w"
        return GridDictionary(counts: c)
    }()

    public init(counts: [String: Int]) {
        self.counts = counts
        total = counts.values.reduce(0, +)
    }

    public var words: [String] { counts.keys.sorted() }
    public func known(_ word: String) -> Bool { counts[word] != nil }

    /// The most frequent known candidate; ties break alphabetically.
    public func correction(_ word: String) -> String {
        candidates(word).max { a, b in
            let pa = counts[a] ?? 0, pb = counts[b] ?? 0
            return pa != pb ? pa < pb : a > b
        } ?? word
    }

    func candidates(_ word: String) -> [String] {
        if known(word) { return [word] }
        let one = edits1(word)
        let knownOne = one.filter(known)
        if !knownOne.isEmpty { return Array(knownOne) }
        var knownTwo = Set<String>()
        for e1 in one { for e2 in edits1(e1) where known(e2) { knownTwo.insert(e2) } }
        return knownTwo.isEmpty ? [word] : Array(knownTwo)
    }

    func edits1(_ word: String) -> Set<String> {
        let chars = Array(word)
        var out = Set<String>()
        for i in 0...chars.count {
            let l = chars[0..<i], r = chars[i...]
            if !r.isEmpty { out.insert(String(l + r.dropFirst())) }
            if r.count > 1 { out.insert(String(l + [r[r.startIndex + 1], r[r.startIndex]] + r.dropFirst(2))) }
            for c in Self.letters {
                if !r.isEmpty { out.insert(String(l + [c] + r.dropFirst())) }
                out.insert(String(l + [c] + r))
            }
        }
        return out
    }
}

// MARK: - LipNet through Core ML

/// LipNet (Assael et al. 2016), weights from rizkiarm/LipNet (MIT, trained
/// on GRID), converted to Core ML by ml/conversion/lipnet_to_coreml.py.
/// Input [1, T, 100, 50, 3] float32 in [0, 1], output [1, T, 28]: 26 letters,
/// space (26), CTC blank (27). English, 51-word vocabulary, 25 fps, clips of
/// up to 75 frames.
public final class LipNetCoreML: VisualSpeechModel {
    public static let modelName = "LipNetGRID"
    public static let info = ModelInfo(
        name: "lipnet-grid",
        license: "MIT (code and weights, github.com/rizkiarm/LipNet); GRID corpus CC BY 4.0",
        languages: ["en"],
        vocabulary: "GRID: 51 words in a fixed sentence shape (command colour preposition letter digit adverb)",
        note: "reads GRID's vocabulary only; open English needs a larger model (see docs/models.md); Core ML conversion of the Keras weights")
    public static let howToGet = "python3 ../ml/conversion/lipnet_to_coreml.py, then add LipNetGRID.mlpackage to the app target"

    public let key = "lipnet-grid"
    public let info = LipNetCoreML.info
    public let nativeFps = 25.0
    public let maxFrames = 75
    public let dictionary = GridDictionary.grid
    let alphabet = GridDictionary.letters.map { String($0) } + [" ", ""]
    let blank = 27, space = 26
    private let model: MLModel
    private let inputName: String
    private let outputName: String

    /// The compiled model in the bundle (Xcode compiles an .mlpackage into
    /// .mlmodelc), or the package itself when it was added as a resource.
    public static func bundledModelURL(in bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: modelName, withExtension: "mlmodelc") ?? bundle.url(forResource: modelName, withExtension: "mlpackage")
    }

    public init(modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws {
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try await MLModel.compileModel(at: modelURL)
        model = try MLModel(contentsOf: compiled, configuration: configuration)
        let description = model.modelDescription
        guard let input = description.inputDescriptionsByName.keys.sorted().first,
              let output = description.outputDescriptionsByName.keys.sorted().first else {
            throw LipNetError.unexpectedModel("no input or output feature")
        }
        inputName = input
        outputName = output
    }

    public func predict(batch: [MouthClip]) async throws -> [DecodedUtterance] {
        var out: [DecodedUtterance] = []
        for clip in batch {
            let t = clip.frames.count
            guard t >= 2 else { out.append(DecodedUtterance()); continue }
            let probabilities = try run(clip)
            let decoded = CTCDecoder.bestPath(probabilities: probabilities, frames: t, classes: 28, alphabet: alphabet, blank: blank, space: space)
            out.append(CTCDecoder.correct(decoded, corrector: dictionary.correction, known: dictionary.known))
        }
        return out
    }

    /// One clip through the network: (T, 28) probabilities, row-major.
    private func run(_ clip: MouthClip) throws -> [Float] {
        let t = clip.frames.count
        let perFrame = MouthCrop.width * MouthCrop.height * 3
        let input = try MLMultiArray(shape: [1, t, MouthCrop.width, MouthCrop.height, 3] as [NSNumber], dataType: .float32)
        // VERIFY: MLMultiArray.withUnsafeMutableBytes(_:) passes (buffer, strides in elements), iOS 15.4+.
        input.withUnsafeMutableBytes { buffer, strides in
            let floats = buffer.bindMemory(to: Float.self)
            for (frame, bytes) in clip.frames.enumerated() {
                precondition(bytes.count == perFrame, "crop must be 100x50x3")
                for x in 0..<MouthCrop.width {
                    for y in 0..<MouthCrop.height {
                        let src = (x * MouthCrop.height + y) * 3
                        let dst = frame * strides[1] + x * strides[2] + y * strides[3]
                        floats[dst] = Float(bytes[src]) / 255
                        floats[dst + strides[4]] = Float(bytes[src + 1]) / 255
                        floats[dst + 2 * strides[4]] = Float(bytes[src + 2]) / 255
                    }
                }
            }
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
        let prediction = try model.prediction(from: features)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw LipNetError.unexpectedModel("output \(outputName) is not a multi-array")
        }
        guard output.count == t * 28 else { throw LipNetError.unexpectedModel("output has \(output.count) values, expected \(t * 28)") }
        if output.dataType == .float32 {
            // VERIFY: withUnsafeBufferPointer(ofType:) reads a contiguous float32 array, iOS 15.4+.
            return output.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
        }
        return (0..<output.count).map { output[$0].floatValue } // float16/double outputs, element by element
    }
}

public enum LipNetError: Error, LocalizedError {
    case unexpectedModel(String)
    public var errorDescription: String? {
        switch self { case .unexpectedModel(let why): return "LipNetGRID model: \(why)" }
    }
}

// MARK: - Remote reader

public enum RemoteError: Error, LocalizedError {
    case http(Int, String)
    case notConfigured
    case failed(String)

    public var errorDescription: String? {
        switch self {
        case .http(let status, let body): return "server answered \(status): \(body)"
        case .notConfigured: return "no inference API URL is configured (Settings)"
        case .failed(let why): return why
        }
    }
}

/// The inference API (docs/api.md): the whole video goes to the service,
/// which runs the same pipeline and answers with the schema's JSON. The
/// job's `processing` field ("local" for a service on the user's own machine,
/// "server" otherwise) is shown to the user; DELETE removes the result.
public final class RemoteLipReader {
    public let baseURL: URL
    public let token: String?
    let session: URLSession

    public init(baseURL: URL, token: String? = nil, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = (token?.isEmpty ?? true) ? nil : token
        self.session = session
    }

    private func request(_ path: String, method: String = "GET") -> URLRequest {
        var r = URLRequest(url: baseURL.appending(path: path))
        r.httpMethod = method
        if let token { r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        return r
    }

    private func check(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw RemoteError.http(http.statusCode, String(data: data.prefix(300), encoding: .utf8) ?? "")
        }
    }

    public func health() async throws -> HealthResponse {
        let (data, response) = try await session.data(for: request("health"))
        try check(response, data)
        return try JSONCoding.decoder.decode(HealthResponse.self, from: data)
    }

    public func languages() async throws -> [LanguageAvailability] {
        let (data, response) = try await session.data(for: request("languages"))
        try check(response, data)
        return try JSONCoding.decoder.decode([LanguageAvailability].self, from: data)
    }

    /// `POST /jobs` as multipart form data with a `file` field and the options
    /// JSON in `options`. The body is streamed from a temporary file so a long
    /// video is never held in memory.
    public func createJob(fileURL: URL, options: Options) async throws -> Job {
        let boundary = "lipreader-\(UUID().uuidString)"
        var req = request("jobs", method: "POST")
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let bodyURL = FileManager.default.temporaryDirectory.appending(path: "upload-\(UUID().uuidString).body")
        _ = FileManager.default.createFile(atPath: bodyURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: bodyURL)
        defer { try? FileManager.default.removeItem(at: bodyURL) }
        func write(_ s: String) throws { try handle.write(contentsOf: Data(s.utf8)) }
        try write("--\(boundary)\r\nContent-Disposition: form-data; name=\"options\"\r\nContent-Type: application/json\r\n\r\n")
        try handle.write(contentsOf: try JSONCoding.encoder.encode(options))
        try write("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\nContent-Type: video/mp4\r\n\r\n")
        let source = try FileHandle(forReadingFrom: fileURL)
        while let chunk = try source.read(upToCount: 1 << 20), !chunk.isEmpty { try handle.write(contentsOf: chunk) }
        try source.close()
        try write("\r\n--\(boundary)--\r\n")
        try handle.close()
        let (data, response) = try await session.upload(for: req, fromFile: bodyURL)
        try check(response, data)
        return try JSONCoding.decoder.decode(Job.self, from: data)
    }

    /// `GET /jobs/{id}?wait=n`: long-polls up to n seconds (the API caps it at 30).
    public func job(_ id: String, wait: Int = 0) async throws -> Job {
        var req = request("jobs/\(id)")
        if wait > 0 { req.url = req.url?.appending(queryItems: [URLQueryItem(name: "wait", value: String(min(wait, 30)))]) }
        let (data, response) = try await session.data(for: req)
        try check(response, data)
        return try JSONCoding.decoder.decode(Job.self, from: data)
    }

    public func delete(_ id: String) async throws {
        let (data, response) = try await session.data(for: request("jobs/\(id)", method: "DELETE"))
        try check(response, data)
    }

    /// Upload, then poll until the job is done or failed; `progress` gets the
    /// server's 0...1. The returned job carries `result` and `processing`.
    public func analyze(fileURL: URL, options: Options, progress: (@Sendable (Double) -> Void)? = nil) async throws -> Job {
        var job = try await createJob(fileURL: fileURL, options: options)
        while job.status == .queued || job.status == .running {
            try Task.checkCancellation()
            progress?(job.progress)
            job = try await self.job(job.id, wait: 30)
        }
        progress?(1)
        switch job.status {
        case .done where job.result != nil: return job
        case .failed: throw RemoteError.failed(job.error ?? "the server failed without a reason")
        case .deleted: throw RemoteError.failed("the job was deleted before it finished")
        default: throw RemoteError.failed("the job ended without a result")
        }
    }
}
