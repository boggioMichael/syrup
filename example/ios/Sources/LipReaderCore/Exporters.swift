// Exporters.swift — transcripts as files: TXT, JSON, SRT, VTT (export.py).
// Uncertain words stay marked in every format; the notice is written into
// the text formats (a TXT header line, a VTT NOTE block).
import Foundation

public enum ExportFormat: String, CaseIterable, Identifiable, Sendable {
    case txt, json, srt, vtt
    public var id: String { rawValue }
    public var fileExtension: String { rawValue }
    public var mimeType: String {
        switch self {
        case .txt: return "text/plain; charset=utf-8"
        case .json: return "application/json"
        case .srt: return "application/x-subrip"
        case .vtt: return "text/vtt"
        }
    }
}

public enum Exporters {
    /// HH:MM:SS,mmm for SRT, HH:MM:SS.mmm otherwise.
    public static func timestamp(_ seconds: Double, srt: Bool) -> String {
        let ms = Int((max(seconds, 0) * 1000).rounded())
        let h = ms / 3_600_000, m = (ms % 3_600_000) / 60_000, s = (ms % 60_000) / 1000, rem = ms % 1000
        return String(format: "%02d:%02d:%02d", h, m, s) + (srt ? "," : ".") + String(format: "%03d", rem)
    }

    static func selected(_ result: AnalysisResult, _ tracks: [Int]?) -> [TranscriptSegment] {
        let wanted: Set<Int>? = tracks.map { Set($0) }
        return result.segments
            .filter { wanted?.contains($0.trackId) ?? true }
            .sorted { ($0.start, $0.trackId) < ($1.start, $1.trackId) }
    }

    public static func toTXT(_ result: AnalysisResult, tracks: [Int]? = nil) -> String {
        var lines = ["# \(NOTICE)", ""]
        for s in selected(result, tracks) {
            let stamp = String(timestamp(s.start, srt: false).dropLast(4))
            lines.append("[\(stamp)] \(result.label(forTrack: s.trackId)): \(s.text)   (confidence \(String(format: "%.2f", s.confidence)), \(s.language), \(s.mode.rawValue))")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    public static func toJSON(_ result: AnalysisResult, tracks: [Int]? = nil) throws -> String {
        var data = result.rounded()
        if let tracks {
            let wanted = Set(tracks)
            data.segments = data.segments.filter { wanted.contains($0.trackId) }
            data.tracks = data.tracks.filter { wanted.contains($0.trackId) }
        }
        return String(decoding: try JSONCoding.encoder.encode(data), as: UTF8.self)
    }

    public static func toSRT(_ result: AnalysisResult, tracks: [Int]? = nil, labels: Bool = true) -> String {
        selected(result, tracks).enumerated().map { i, s in
            let text = labels ? "\(result.label(forTrack: s.trackId)): \(s.text)" : s.text
            return "\(i + 1)\n\(timestamp(s.start, srt: true)) --> \(timestamp(s.end, srt: true))\n\(text)\n"
        }.joined(separator: "\n")
    }

    public static func toVTT(_ result: AnalysisResult, tracks: [Int]? = nil, labels: Bool = true) -> String {
        var out = ["WEBVTT", "NOTE \(NOTICE)", ""]
        for s in selected(result, tracks) {
            let text = labels ? "<v \(result.label(forTrack: s.trackId))>\(s.text)" : s.text
            out.append("\(timestamp(s.start, srt: false)) --> \(timestamp(s.end, srt: false))\n\(text)\n")
        }
        return out.joined(separator: "\n")
    }

    public static func export(_ result: AnalysisResult, format: ExportFormat, tracks: [Int]? = nil) throws -> String {
        switch format {
        case .txt: return toTXT(result, tracks: tracks)
        case .json: return try toJSON(result, tracks: tracks)
        case .srt: return toSRT(result, tracks: tracks)
        case .vtt: return toVTT(result, tracks: tracks)
        }
    }

    /// The export written to a temporary file, for a share sheet.
    public static func write(_ result: AnalysisResult, format: ExportFormat, tracks: [Int]? = nil, baseName: String = "transcript") throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(baseName).\(format.fileExtension)")
        try export(result, format: format, tracks: tracks).write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
