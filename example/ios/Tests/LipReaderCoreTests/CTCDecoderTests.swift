import XCTest
@testable import LipReaderCore

/// Mirrors test_ctc_best_path_keeps_timing_and_confidence in
/// example/lipreader/tests/test_lipreader.py.
final class CTCDecoderTests: XCTestCase {
    let alphabet = GridDictionary.letters.map { String($0) } + [" ", ""]
    let a = 0, b = 1, space = 26, blank = 27

    func testBestPathKeepsTimingAndConfidence() {
        let path = [a, a, blank, b, blank, space, space, b, a]
        var probs = [Float](repeating: 0.01, count: path.count * 28)
        for (t, c) in path.enumerated() { probs[t * 28 + c] = t < 5 ? 0.9 : 0.6 }
        let decoded = CTCDecoder.bestPath(probabilities: probs, frames: path.count, classes: 28, alphabet: alphabet, blank: blank, space: space)
        XCTAssertEqual(decoded.words.map(\.text), ["ab", "ba"])
        XCTAssertEqual(decoded.words[0].startFrame, 0)
        XCTAssertEqual(decoded.words[0].endFrame, 3)
        XCTAssertEqual(decoded.words[0].confidence, 0.9, accuracy: 1e-6)
        XCTAssertEqual(decoded.words[1].confidence, 0.6, accuracy: 1e-6)

        let fixed = CTCDecoder.correct(decoded, corrector: { ["ab": "abe"][$0] ?? $0 }, known: { $0 == "abe" })
        XCTAssertEqual(fixed.words[0].text, "abe")
        XCTAssertEqual(fixed.words[0].raw, "ab")
        XCTAssertEqual(fixed.words[0].confidence, 0.9 * 0.8, accuracy: 1e-6)
        XCTAssertEqual(fixed.words[1].confidence, 0.6 * 0.5, accuracy: 1e-6, "unknown word: doubly doubtful")
        XCTAssertNil(fixed.words[1].raw)
    }

    func testBlankOnlyPathYieldsNoWords() {
        let probs = [Float](repeating: 0, count: 5 * 28).enumerated().map { $0.offset % 28 == blank ? Float(1) : 0 }
        let decoded = CTCDecoder.bestPath(probabilities: probs, frames: 5, classes: 28, alphabet: alphabet, blank: blank, space: space)
        XCTAssertTrue(decoded.words.isEmpty, "no letters means no text, never a guess")
    }

    func testGridDictionaryCorrectsLikeNorvig() {
        let d = GridDictionary.grid
        XCTAssertEqual(d.words.count, 51)
        XCTAssertEqual(d.correction("blue"), "blue")
        XCTAssertEqual(d.correction("bleu"), "blue")
        XCTAssertEqual(d.correction("sevn"), "seven")
        XCTAssertEqual(d.correction("agian"), "again")
        XCTAssertFalse(d.known("w"), "GRID has no letter w")
        XCTAssertEqual(d.correction("zzqxxj"), "zzqxxj", "nothing within two edits: the raw word stays, and the unknown penalty applies")
    }
}
