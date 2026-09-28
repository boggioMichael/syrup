import XCTest
@testable import LipReaderCore

/// Mirrors test_export_formats in example/lipreader/tests/test_lipreader.py.
final class ExportersTests: XCTestCase {
    static func sampleResult() -> AnalysisResult {
        let words = [Word(text: "set", start: 1.0, end: 1.3, confidence: 0.9, uncertain: false),
                     Word(text: "blue", start: 1.4, end: 2.5, confidence: 0.3, uncertain: true, raw: "bleu")]
        let segment = TranscriptSegment(id: "s1", trackId: 1, words: words, language: "en", mode: .visual, model: "lipnet-grid")
        return AnalysisResult(video: VideoInfo(duration: 3, fps: 25, width: 360, height: 288), mode: .visual,
                              language: LanguageInfo(requested: "en", used: "en", detection: .requested),
                              tracks: [PersonTrack(trackId: 1, firstSeen: 0, lastSeen: 3)], segments: [segment],
                              models: Models(faceDetector: "vision"), timing: Timing(processingSeconds: 1, realtimeFactor: 1 / 3))
    }

    func testTimestamps() {
        XCTAssertEqual(Exporters.timestamp(3661.5, srt: true), "01:01:01,500")
        XCTAssertEqual(Exporters.timestamp(0.0004, srt: false), "00:00:00.000")
        XCTAssertEqual(Exporters.timestamp(-2, srt: false), "00:00:00.000")
    }

    func testFormatsKeepUncertainWordsMarked() throws {
        let result = Self.sampleResult()
        let srt = Exporters.toSRT(result)
        XCTAssertTrue(srt.contains("00:00:01,000 --> 00:00:02,500"))
        XCTAssertTrue(srt.contains("Person 1: set [blue?]"))
        XCTAssertTrue(srt.hasPrefix("1\n"))
        let vtt = Exporters.toVTT(result)
        XCTAssertTrue(vtt.hasPrefix("WEBVTT\nNOTE \(NOTICE)\n"))
        XCTAssertTrue(vtt.contains("<v Person 1>set [blue?]"))
        let txt = Exporters.toTXT(result)
        XCTAssertTrue(txt.hasPrefix("# \(NOTICE)\n"))
        XCTAssertTrue(txt.contains("[00:00:01] Person 1: set [blue?]   (confidence 0.60, en, visual)"))
        let json = try Exporters.toJSON(result)
        let decoded = try JSONCoding.decoder.decode(AnalysisResult.self, from: Data(json.utf8))
        XCTAssertEqual(decoded.segments[0].words[1].uncertain, true)
        XCTAssertEqual(decoded.segments[0].words[1].raw, "bleu")
        XCTAssertEqual(decoded.segments[0].text, "set [blue?]")
        XCTAssertEqual(decoded.notice, NOTICE)
        XCTAssertEqual(Exporters.toSRT(result, tracks: [2]).trimmingCharacters(in: .whitespacesAndNewlines), "")
    }

    func testUnattributedLabel() {
        var result = Self.sampleResult()
        result.segments[0].trackId = -1
        XCTAssertTrue(Exporters.toSRT(result).contains("Unattributed: set [blue?]"))
    }
}
