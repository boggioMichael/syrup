import XCTest
@testable import LipReaderCore

final class ActivityTests: XCTestCase {
    func testStillFaceIsSilentAndMovingMouthSpeaks() {
        let still = [Double](repeating: 0, count: 50)
        XCTAssertTrue(Activity.speakingProbability(openness: still, mouthMotion: still, faceMotion: still).allSatisfy { $0 == 0 })
        let moving = (0..<50).map { 0.1 + 0.08 * sin(Double($0) * 12 / 49) }
        let p = Activity.speakingProbability(openness: moving, mouthMotion: still, faceMotion: still)
        XCTAssertGreaterThan(p[25], 0.5)
        let spans = Activity.spans(probability: p, times: (0..<50).map { Double($0) / 25 })
        XCTAssertEqual(spans.count, 1)
        XCTAssertGreaterThan(spans[0].probability, 0.5)
    }

    func testHeadTurningDoesNotCountAsTalking() {
        let still = [Double](repeating: 0, count: 50)
        let motion = [Double](repeating: 0.2, count: 50)
        // The mouth crop changes exactly as much as the whole face: no excess, no speech.
        let p = Activity.speakingProbability(openness: still, mouthMotion: motion, faceMotion: motion)
        XCTAssertTrue(p.allSatisfy { $0 == 0 })
    }

    func testSpansMergeShortGapsAndDropShortRuns() {
        let times = (0..<100).map { Double($0) / 25 }
        var p = [Double](repeating: 0, count: 100)
        for i in 10..<30 { p[i] = 0.9 }
        for i in 40..<60 { p[i] = 0.9 } // gap of 10 frames = 0.4 s <= merge gap 0.6 s
        p[80] = 0.9 // one frame: shorter than min span
        let spans = Activity.spans(probability: p, times: times)
        XCTAssertEqual(spans.count, 1)
        XCTAssertEqual(spans[0].start, times[6], accuracy: 1e-9, "padded by 0.16 s = 4 frames")
        XCTAssertEqual(spans[0].end, times[63], accuracy: 1e-9)
    }

    func testWindowsWidenWithSilenceAndNeverTakeAnotherSpansSpeech() {
        let times = (0..<200).map { Double($0) / 25 }
        // Two short spans far apart: each is widened towards 75 frames with the silence around it.
        let spans = [SpeakingSpan(start: times[50], end: times[70], probability: 0.9), SpeakingSpan(start: times[150], end: times[170], probability: 0.8)]
        let windows = Activity.readWindows(spans: spans, times: times, readUntil: 0, maxFrames: 75, fps: 25)
        XCTAssertEqual(windows.count, 2)
        for w in windows { XCTAssertEqual(w.last - w.first + 1, 75) }
        XCTAssertLessThanOrEqual(windows[0].last, 149, "the first window stops before the second span")
        XCTAssertGreaterThanOrEqual(windows[1].first, 71, "the second window starts after the first span")
        XCTAssertEqual(windows[0].probability, 0.9)
    }

    func testWindowsMergeNeighboursWithinASecondAndSplitLongSpans() {
        let times = (0..<400).map { Double($0) / 25 }
        // 20 frames, a 15-frame pause (0.6 s), 20 frames: one clip.
        let close = [SpeakingSpan(start: times[10], end: times[29], probability: 0.7), SpeakingSpan(start: times[45], end: times[64], probability: 0.9)]
        let merged = Activity.readWindows(spans: close, times: times, readUntil: 0, maxFrames: 75, fps: 25)
        XCTAssertEqual(merged.count, 1)
        XCTAssertEqual(merged[0].probability, 0.9, "the merged span keeps the higher probability")
        // 200 frames of speech: chunks of 75, 75, 50.
        let long = [SpeakingSpan(start: times[100], end: times[299], probability: 0.95)]
        let chunks = Activity.readWindows(spans: long, times: times, readUntil: 0, maxFrames: 75, fps: 25)
        XCTAssertEqual(chunks.map { $0.last - $0.first + 1 }, [75, 75, 50])
        XCTAssertEqual(chunks[0].first, 100)
        // Fewer than 8 frames is not worth reading.
        let tiny = [SpeakingSpan(start: times[10], end: times[14], probability: 0.9)]
        XCTAssertTrue(Activity.readWindows(spans: tiny, times: times, readUntil: 0, maxFrames: 75, fps: 25).isEmpty)
    }

    func testPositionIsSearchsortedLeft() {
        let times = [0.0, 0.04, 0.08, 0.12]
        XCTAssertEqual(Activity.position(of: 0.04, in: times), 1)
        XCTAssertEqual(Activity.position(of: 0.05, in: times), 2)
        XCTAssertEqual(Activity.position(of: -1, in: times), 0)
        XCTAssertEqual(Activity.position(of: 1, in: times), 4)
    }
}
