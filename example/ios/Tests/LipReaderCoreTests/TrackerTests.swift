import XCTest
@testable import LipReaderCore

/// Mirrors test_tracker_keeps_ids_and_cuts_end_tracks in test_lipreader.py;
/// pure bookkeeping, no Vision.
final class TrackerTests: XCTestCase {
    func testTrackerKeepsIdsAndCutsEndTracks() {
        let tracker = Tracker(graceFrames: 2)
        let a = Box(x: 10, y: 10, w: 50, h: 50), b = Box(x: 200, y: 10, w: 50, h: 50)
        var live: [TrackState] = []
        for i in 0..<5 {
            live = tracker.update(frameIndex: i, t: Double(i) / 25, detections: [
                FaceDetection(box: Box(x: a.x + Double(i), y: a.y, w: a.w, h: a.h), score: 1), FaceDetection(box: b, score: 1),
            ])
        }
        XCTAssertEqual(live.map(\.trackId).sorted(), [1, 2])
        // A disappears for longer than the grace: its track ends, B keeps its id.
        for i in 5..<10 { live = tracker.update(frameIndex: i, t: Double(i) / 25, detections: [FaceDetection(box: b, score: 1)]) }
        XCTAssertEqual(live.map(\.trackId), [2])
        tracker.cut()
        live = tracker.update(frameIndex: 10, t: 10 / 25, detections: [FaceDetection(box: b, score: 1)])
        XCTAssertEqual(live.map(\.trackId), [3], "after a cut nothing is the same person")
        let finished = tracker.finish()
        XCTAssertEqual(finished.map(\.trackId), [1, 2, 3])
        XCTAssertLessThanOrEqual(finished[0].lastT, 4 / 25 + 1e-9, "held frames after the last detection are trimmed")
        XCTAssertEqual(finished[0].toPersonTrack().confidence, 5.0 / 8.0, accuracy: 1e-9, "5 detections on 8 attempts (3 missed frames)")
    }

    func testHeldFramesNeitherCountAsDetectionsNorStartTracks() {
        let tracker = Tracker()
        let box = Box(x: 0, y: 0, w: 40, h: 40)
        tracker.update(frameIndex: 0, t: 0, detections: [FaceDetection(box: box, score: 1)])
        tracker.update(frameIndex: 1, t: 0.04, detections: [FaceDetection(box: box, score: 1, held: true)])
        let track = tracker.finish()[0]
        XCTAssertEqual(track.detections, 1)
        XCTAssertEqual(track.attempts, 1)
        XCTAssertEqual(tracker.finished.count, 1)
        XCTAssertEqual(track.keyframes.count, 1)
    }

    func testHungarianPicksTheCheapestAssignment() {
        XCTAssertEqual(hungarian([[0.9, 0.1], [0.1, 0.9]]), [1, 0])
        XCTAssertEqual(hungarian([[0.1, 0.2, 0.3]]), [0])
        XCTAssertEqual(hungarian([[0.5], [0.1], [0.3]]), [-1, 0, -1])
        XCTAssertEqual(hungarian([]), [])
    }

    func testCutDetectorFiresOnADifferentPictureOnly() {
        var blue = [Float](repeating: 0, count: 512), green = blue, shifted = blue
        blue[(200 >> 5) * 0 + 0 * 8 + (200 >> 5)] = 1.0          // everything in the blue bin
        shifted[(200 >> 5) * 0 + 0 * 8 + (200 >> 5)] = 0.9        // the same picture, a little different
        shifted[(180 >> 5) * 64 + (150 >> 5) * 8 + (120 >> 5)] = 0.1
        green[(160 >> 5) * 8] = 1.0                               // a green room
        var detector = CutDetector()
        XCTAssertFalse(detector.update(histogram: blue))
        XCTAssertFalse(detector.update(histogram: shifted))
        XCTAssertTrue(detector.update(histogram: green))
        XCTAssertFalse(detector.update(histogram: blue), "a second cut within the minimum gap is ignored")
    }
}
