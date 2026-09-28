import XCTest
@testable import LipReaderCore

final class SchemaTests: XCTestCase {
    /// A job as the inference API writes it (job.schema.json wrapping analysis.schema.json).
    static let jobJSON = """
    {
      "id": "abc123", "status": "done", "progress": 1, "createdAt": "2026-09-28T10:00:00Z",
      "finishedAt": "2026-09-28T10:00:05Z", "expiresAt": null, "processing": "server", "error": null,
      "options": {"mode": "audiovisual", "language": "auto", "speakers": [1, 2], "maxFaces": 8, "startTime": 0, "uncertainBelow": 0.5},
      "result": {
        "version": "1",
        "video": {"duration": 3.0, "fps": 25.0, "width": 360, "height": 288, "source": "one_speaker.mp4", "sampledFps": 25.0},
        "mode": "audiovisual",
        "language": {"requested": "auto", "used": null, "detection": "unavailable", "note": "nothing could be read"},
        "tracks": [{"trackId": 1, "label": "Person 1", "firstSeen": 0.0, "lastSeen": 2.96,
                    "keyframes": [{"t": 0.0, "box": {"x": 100, "y": 50, "w": 120, "h": 120}, "mouth": {"x": 130, "y": 130, "w": 60, "h": 30}},
                                  {"t": 2.0, "box": {"x": 140, "y": 50, "w": 120, "h": 120}}],
                    "speaking": [{"start": 0.52, "end": 1.92, "probability": 0.91}],
                    "detectedLanguage": null, "segmentIds": ["s1"], "confidence": 0.98}],
        "segments": [{"id": "s1", "trackId": 1, "start": 0.6, "end": 1.9, "text": "set [blue?]",
                      "words": [{"text": "set", "start": 0.6, "end": 0.9, "confidence": 0.9, "uncertain": false},
                                {"text": "blue", "start": 1.0, "end": 1.9, "confidence": 0.3, "uncertain": true, "raw": "bleu"}],
                      "language": "en", "confidence": 0.6, "mode": "visual", "model": "lipnet-grid"}],
        "models": {"faceDetector": "opencv-haar", "vsr": {"name": "lipnet-grid", "license": "MIT", "languages": ["en"], "vocabulary": "GRID"}},
        "timing": {"processingSeconds": 1.5, "realtimeFactor": 0.5, "stages": {"detect": 0.4, "vsr": 0.9}},
        "notice": "\(NOTICE)",
        "warnings": ["no audio track: the audio stage had nothing to read; the result is visual-only"]
      }
    }
    """

    func testJobAndResultRoundTrip() throws {
        let job = try JSONCoding.decoder.decode(Job.self, from: Data(Self.jobJSON.utf8))
        XCTAssertEqual(job.status, .done)
        XCTAssertEqual(job.processing, .server)
        XCTAssertEqual(job.options.speakers, .tracks([1, 2]))
        XCTAssertEqual(job.options.mode, .audiovisual)
        let result = try XCTUnwrap(job.result)
        XCTAssertNil(result.language.used)
        XCTAssertEqual(result.language.detection, .unavailable)
        XCTAssertEqual(result.tracks[0].segmentIds, ["s1"])
        XCTAssertEqual(result.tracks[0].keyframes[0].mouth?.w, 60)
        XCTAssertEqual(result.segments[0].words[1].raw, "bleu")
        XCTAssertEqual(result.models.vsr?.name, "lipnet-grid")
        XCTAssertEqual(result.timing.stages["vsr"], 0.9)
        XCTAssertEqual(result.warnings.count, 1)

        let encoded = try JSONCoding.encoder.encode(result)
        let again = try JSONCoding.decoder.decode(AnalysisResult.self, from: encoded)
        XCTAssertEqual(again, result)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["version", "video", "mode", "language", "tracks", "segments", "models", "timing", "notice", "warnings"])
        let language = try XCTUnwrap(object["language"] as? [String: Any])
        XCTAssertTrue(language["used"] is NSNull, "`used` is required and nullable: nil must be written as null")
        let word = try XCTUnwrap(((object["segments"] as? [[String: Any]])?[0]["words"] as? [[String: Any]])?[0])
        XCTAssertNil(word["raw"], "raw is only written when it differs from the text")
    }

    func testOptionsDefaultsAndSpeakers() throws {
        let options = try JSONCoding.decoder.decode(Options.self, from: Data("{}".utf8))
        XCTAssertEqual(options, Options())
        XCTAssertEqual(options.mode, .visual)
        XCTAssertEqual(options.language, "auto")
        XCTAssertEqual(options.uncertainBelow, 0.5)
        let current = try JSONCoding.decoder.decode(Options.self, from: Data(#"{"speakers": "current", "language": "HE"}"#.utf8))
        XCTAssertEqual(current.speakers, .current)
        XCTAssertEqual(current.language, "he")
        XCTAssertThrowsError(try JSONCoding.decoder.decode(Options.self, from: Data(#"{"speakers": "someone"}"#.utf8)))
        XCTAssertThrowsError(try JSONCoding.decoder.decode(Options.self, from: Data(#"{"mode": "psychic"}"#.utf8)))
        let encoded = String(decoding: try JSONCoding.encoder.encode(Options(speakers: .tracks([3]))), as: UTF8.self)
        XCTAssertTrue(encoded.contains("\"speakers\" : [\n    3\n  ]") || encoded.contains("\"speakers\":[3]"))
        XCTAssertFalse(encoded.contains("endTime"), "absent options stay absent")
    }

    func testRenderingNeverAssertsUncertainWords() {
        XCTAssertEqual(Word(text: "five", start: 0, end: 0.2, confidence: 0.3, uncertain: true).rendered, "[five?]")
        XCTAssertEqual(Word(text: "five", start: 0, end: 0.2, confidence: 0.9, uncertain: false).rendered, "five")
        let segment = TranscriptSegment(id: "s1", trackId: 1, words: [Word(text: "a", start: 0, end: 1, confidence: 0.2, uncertain: true)],
                                        language: "en", mode: .visual, model: "m")
        XCTAssertEqual(segment.text, "[a?]")
        XCTAssertEqual(segment.confidence, 0.2)
    }

    func testBoxAtInterpolatesBetweenKeyframes() {
        let track = PersonTrack(trackId: 1, firstSeen: 0, lastSeen: 2, keyframes: [
            Keyframe(t: 0, box: Box(x: 0, y: 0, w: 100, h: 100)),
            Keyframe(t: 1, box: Box(x: 100, y: 50, w: 100, h: 100)),
            Keyframe(t: 2, box: Box(x: 100, y: 50, w: 200, h: 100)),
        ])
        XCTAssertEqual(track.boxAt(-1), Box(x: 0, y: 0, w: 100, h: 100), "before the first keyframe: the first box")
        XCTAssertEqual(track.boxAt(5), Box(x: 100, y: 50, w: 200, h: 100), "after the last: the last box")
        XCTAssertEqual(track.boxAt(0.5), Box(x: 50, y: 25, w: 100, h: 100))
        XCTAssertEqual(track.boxAt(1.25), Box(x: 100, y: 50, w: 125, h: 100))
        XCTAssertEqual(track.boxAt(1), Box(x: 100, y: 50, w: 100, h: 100))
        XCTAssertNil(PersonTrack(trackId: 2, firstSeen: 0, lastSeen: 0).boxAt(0))
    }

    func testBoxGeometry() {
        let a = Box(x: 0, y: 0, w: 10, h: 10), b = Box(x: 5, y: 5, w: 10, h: 10)
        XCTAssertEqual(a.iou(b), 25.0 / 175.0, accuracy: 1e-9)
        XCTAssertEqual(a.iou(Box(x: 20, y: 20, w: 1, h: 1)), 0)
        XCTAssertEqual(Box(x: 1.234, y: 2.346, w: 3.456, h: 4.567).rounded(), Box(x: 1.23, y: 2.35, w: 3.46, h: 4.57))
    }
}
