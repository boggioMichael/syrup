// LanguageRegistry.swift — which visual speech model reads which language,
// whether it can run on this device, and under what licence (vsr/registry.py).
//
// The table is the honest one: a language whose only models are
// non-commercial research checkpoints that need PyTorch and a download is
// listed as such, and asking for it throws LanguageUnavailable with the
// reasons. Hebrew has no public visual model or corpus; it is reachable only
// through the audio modes, and the UI says so.
//
// The audio side lives here too: SFSpeechRecognizer availability per
// language, and the transcriber the audio modes use. Nothing in this file is
// called by the visual mode's code path (Pipeline gates on `Mode.usesAudio`).
import Foundation
import Speech

public struct ModelSpec: Sendable {
    public var key: String
    public var info: ModelInfo
    public var nativeFps: Double
    /// Longest clip the model should see at once.
    public var maxFrames: Int
    /// "lipnet-100x50" | "mouth-96" ...
    public var crop: String
    public var available: Bool
    /// Why not available, or how it was found.
    public var reason: String
    /// Download/licence instructions, shown to users.
    public var howToGet: String
}

public struct LanguageStatus: Equatable, Sendable, Identifiable {
    public var code: String
    public var name: String
    public var visualAvailable: Bool
    public var visualModel: String?
    public var visualLicense: String?
    public var visualNote: String
    public var id: String { code }

    public var availability: LanguageAvailability {
        LanguageAvailability(code: code, name: name,
                             visual: .init(available: visualAvailable, model: visualModel, license: visualLicense, note: visualNote))
    }
}

/// No visual speech model for the requested language can run here; the
/// message says which models exist, under what licence, and what is needed.
public struct LanguageUnavailable: Error, LocalizedError {
    public let language: String
    public let candidates: [ModelSpec]

    public var errorDescription: String? {
        guard !candidates.isEmpty else {
            return "no public visual speech model or corpus for '\(language)' is known; the language can only be read through the audio modes"
        }
        var lines = ["no visual speech model for '\(language)' can run here:"]
        for c in candidates {
            lines.append("  - \(c.info.name) (\(c.info.license)): \(c.reason.isEmpty ? "not available" : c.reason)")
            if !c.howToGet.isEmpty { lines.append("      \(c.howToGet)") }
        }
        return lines.joined(separator: "\n")
    }
}

public final class LanguageRegistry {
    public static let shared = LanguageRegistry()

    public static let names: [String: String] = [
        "en": "English", "he": "Hebrew", "es": "Spanish", "ar": "Arabic", "fr": "French", "de": "German", "it": "Italian",
        "pt": "Portuguese", "el": "Greek", "ru": "Russian", "zh": "Mandarin Chinese",
    ]
    public static let order = ["en", "he", "es", "ar", "fr", "de", "it", "pt", "el", "ru", "zh"]

    /// Where the LipNet Core ML model is, if the app bundled it.
    public let lipNetURL: URL?

    public init(lipNetURL: URL? = LipNetCoreML.bundledModelURL()) { self.lipNetURL = lipNetURL }

    // MARK: Candidates

    func lipNetSpec() -> ModelSpec {
        ModelSpec(key: "lipnet-grid", info: LipNetCoreML.info, nativeFps: 25, maxFrames: 75, crop: "lipnet-100x50",
                  available: lipNetURL != nil,
                  reason: lipNetURL.map { "model at \($0.lastPathComponent)" } ?? "LipNetGRID.mlpackage is not in the app bundle",
                  howToGet: LipNetCoreML.howToGet)
    }

    /// Ma, Petridis, Pantic — Visual Speech Recognition for Multiple Languages
    /// in the Wild (2022). Checkpoints on Google Drive; PyTorch + ESPnet code.
    static func mpc001Spec(_ language: String, wer: String) -> ModelSpec {
        ModelSpec(key: "mpc001-vsr-\(language)",
                  info: ModelInfo(name: "mpc001/Visual_Speech_Recognition_for_Multiple_Languages (\(language))",
                                  license: "non-commercial: \"comparative or benchmarking purposes\" only (repository LICENSE)",
                                  languages: [language], vocabulary: "open", note: "reported visual-only WER \(wer)"),
                  nativeFps: 25, maxFrames: 600, crop: "mouth-96", available: false,
                  reason: "not runnable here: PyTorch/ESPnet research code, no Core ML conversion, and a Google Drive download; non-commercial licence",
                  howToGet: "git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages; download the \(language) model from its README (Google Drive); convert with ml/conversion (not written); non-commercial use only")
    }

    /// Meta MuAViC AVSR checkpoints (AV-HuBERT, fairseq), CC BY-NC 4.0.
    static func muavicSpec(_ language: String) -> ModelSpec {
        ModelSpec(key: "muavic-avhubert-\(language)",
                  info: ModelInfo(name: "MuAViC AV-HuBERT AVSR (\(language))", license: "CC BY-NC 4.0 (facebookresearch/muavic)",
                                  languages: [language], vocabulary: "open",
                                  note: "audio-visual checkpoint; video-only accuracy is not reported by the authors"),
                  nativeFps: 25, maxFrames: 500, crop: "mouth-96", available: false,
                  reason: "not runnable here: PyTorch + fairseq + av_hubert code and a download from dl.fbaipublicfiles.com; non-commercial licence",
                  howToGet: "see https://github.com/facebookresearch/muavic#models (\(language)_avsr checkpoint, dict, tokenizer); non-commercial use only")
    }

    static func autoAvsrSpec() -> ModelSpec {
        ModelSpec(key: "auto-avsr-en",
                  info: ModelInfo(name: "Auto-AVSR VSR base (vsr_trlrs2lrs3vox2avsp_base)",
                                  license: "Apache-2.0 code; weights carry the training data's terms (LRS2/LRS3: non-commercial research)",
                                  languages: ["en"], vocabulary: "open", note: "reported visual-only WER 20.3% on LRS3; 250M parameters"),
                  nativeFps: 25, maxFrames: 600, crop: "mouth-96", available: false,
                  reason: "not runnable here: PyTorch research code, no Core ML conversion, and a Google Drive download",
                  howToGet: "git clone https://github.com/mpc001/auto_avsr; download the checkpoint from its README (Google Drive)")
    }

    /// Every known visual model for a language, best first.
    public func candidates(_ language: String) -> [ModelSpec] {
        switch language.lowercased() {
        case "en": return [lipNetSpec(), Self.autoAvsrSpec(), Self.mpc001Spec("en", wer: "32.3% (LRS3)"), Self.muavicSpec("en")]
        case "es": return [Self.mpc001Spec("es", wer: "44.5% (CMU-MOSEAS)"), Self.muavicSpec("es")]
        case "fr": return [Self.mpc001Spec("fr", wer: "58.6% (CMU-MOSEAS)"), Self.muavicSpec("fr")]
        case "pt": return [Self.mpc001Spec("pt", wer: "51.4% (CMU-MOSEAS)"), Self.muavicSpec("pt")]
        case "it", "ar", "de", "el", "ru": return [Self.muavicSpec(language.lowercased())]
        case "zh": return [Self.mpc001Spec("zh", wer: "8.0% CER (CMLR)")]
        default: return [] // Hebrew and everything else: nothing public exists
        }
    }

    // MARK: The table

    public func languages() -> [LanguageStatus] {
        Self.order.map { code in
            let name = Self.names[code] ?? code
            let specs = candidates(code)
            if let ready = specs.first(where: \.available) {
                return LanguageStatus(code: code, name: name, visualAvailable: true, visualModel: ready.info.name,
                                      visualLicense: ready.info.license, visualNote: ready.info.vocabulary)
            }
            if let first = specs.first {
                return LanguageStatus(code: code, name: name, visualAvailable: false, visualModel: first.info.name,
                                      visualLicense: first.info.license, visualNote: "\(first.reason). \(first.howToGet)")
            }
            return LanguageStatus(code: code, name: name, visualAvailable: false, visualModel: nil, visualLicense: nil,
                                  visualNote: "no public visual speech model or corpus is known for this language; audio modes only")
        }
    }

    public func availableLanguages() -> [String] { languages().filter(\.visualAvailable).map(\.code) }

    /// The model to use for a language, or LanguageUnavailable.
    public func resolve(_ language: String) throws -> ModelSpec {
        let specs = candidates(language)
        if let ready = specs.first(where: \.available) { return ready }
        throw LanguageUnavailable(language: language, candidates: specs)
    }

    public func load(_ spec: ModelSpec) async throws -> VisualSpeechModel {
        guard spec.key == "lipnet-grid", let url = lipNetURL else {
            throw LanguageUnavailable(language: spec.info.languages.first ?? "?", candidates: [spec])
        }
        return try await LipNetCoreML(modelURL: url)
    }
}

// MARK: - Audio modes: SFSpeechRecognizer

public struct AudioStatus: Equatable, Sendable {
    public var code: String
    public var available: Bool
    public var onDevice: Bool
    public var localeIdentifier: String?
    public var note: String
}

public struct AsrWord: Equatable, Sendable {
    public var text: String
    public var start: Double
    public var end: Double
    public var confidence: Double
}

public struct AsrResult: Sendable {
    public var words: [AsrWord]
    /// The locale's language code; Apple's recogniser never identifies a language itself.
    public var language: String
    public var onDevice: Bool
    public var info: ModelInfo
}

public enum SpeechError: Error, LocalizedError {
    case notAuthorized
    case unsupportedLanguage(String)
    case recognizerUnavailable
    public var errorDescription: String? {
        switch self {
        case .notAuthorized: return "speech recognition is not authorised for this app"
        case .unsupportedLanguage(let code): return "Apple's speech recogniser has no locale for '\(code)'"
        case .recognizerUnavailable: return "the speech recogniser is unavailable right now (network or device state)"
        }
    }
}

/// Apple's speech recogniser, used by `audiovisual` and `audio-attributed`
/// only. On-device recognition is preferred where the locale supports it
/// (Hebrew is among the supported locales; whether it runs on device depends
/// on the iOS version and the downloaded assets, so `status` asks).
public enum SpeechTranscriber {
    public static let modelName = "apple-sfspeechrecognizer"

    /// The locale Apple's recogniser has for a language code, if any.
    public static func locale(for code: String) -> Locale? {
        let wanted = code.lowercased()
        return SFSpeechRecognizer.supportedLocales().first { ($0.language.languageCode?.identifier ?? "").lowercased() == wanted }
    }

    public static func status(for code: String) -> AudioStatus {
        guard let locale = locale(for: code), let recognizer = SFSpeechRecognizer(locale: locale) else {
            return AudioStatus(code: code, available: false, onDevice: false, localeIdentifier: nil,
                               note: "Apple's speech recogniser has no locale for this language")
        }
        let onDevice = recognizer.supportsOnDeviceRecognition
        return AudioStatus(code: code, available: true, onDevice: onDevice, localeIdentifier: locale.identifier,
                           note: onDevice ? "on-device recognition" : "server-based recognition (audio leaves the device; about one minute per request)")
    }

    public static func requestAuthorization() async -> Bool {
        if SFSpeechRecognizer.authorizationStatus() == .authorized { return true }
        return await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        }
    }

    /// Words with timing and confidence for the whole file. `language` is a
    /// code such as "he" or "en"; nil uses the device locale.
    public static func transcribe(url: URL, language: String?, preferOnDevice: Bool) async throws -> AsrResult {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else { throw SpeechError.notAuthorized }
        let locale: Locale
        if let language {
            guard let found = locale(for: language) else { throw SpeechError.unsupportedLanguage(language) }
            locale = found
        } else {
            locale = Locale.current
        }
        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else { throw SpeechError.recognizerUnavailable }
        let onDevice = preferOnDevice && recognizer.supportsOnDeviceRecognition
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.shouldReportPartialResults = false
        request.requiresOnDeviceRecognition = onDevice
        request.taskHint = .dictation
        let final: SFSpeechRecognitionResult = try await withCheckedThrowingContinuation { continuation in
            var resumed = false
            _ = recognizer.recognitionTask(with: request) { result, error in
                guard !resumed else { return }
                if let error { resumed = true; continuation.resume(throwing: error); return }
                if let result, result.isFinal { resumed = true; continuation.resume(returning: result) }
            }
        }
        // VERIFY: SFTranscriptionSegment.confidence is 0...1 in final results; on-device results may report 0, which stays honest (uncertain).
        let words = final.bestTranscription.segments.map {
            AsrWord(text: $0.substring.trimmingCharacters(in: .whitespaces), start: $0.timestamp, end: $0.timestamp + $0.duration,
                    confidence: Double($0.confidence))
        }
        let code = locale.language.languageCode?.identifier ?? locale.identifier
        let info = ModelInfo(name: "\(modelName) (\(locale.identifier), \(onDevice ? "on-device" : "server"))",
                             license: "Apple system framework (Speech)", languages: [code], vocabulary: "open")
        return AsrResult(words: words, language: code, onDevice: onDevice, info: info)
    }
}
