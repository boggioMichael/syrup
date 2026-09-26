//! Frame differencing and moving-blob detection.
//!
//! With a static camera, anything that changes between two frames is a
//! moving object or an animation. The detector diffs consecutive frames,
//! extracts contiguous changed regions ("blobs"), and feeds them through the
//! [`crate::tracking::ObjectTracker`] so callers get temporally-stable IDs
//! instead of raw, unlabeled regions every call.
//!
//! The detector reports moving regions with position, size, velocity and a
//! track-age-based confidence. It does **not** classify what moved — that
//! interpretation belongs to the consumer.

use image::{GrayImage, Luma, RgbaImage};

use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::{Rect, group_segments};
use crate::scan::for_each_run;
use crate::tracking::{ObjectTracker, Track};

/// Compute a binary motion mask between two same-sized frames in one pass:
/// a pixel is "moved" when the luminance of its per-channel absolute
/// difference crosses `threshold`. Returns `None` on a size mismatch.
pub fn motion_mask(a: &RgbaImage, b: &RgbaImage, threshold: u8) -> Option<GrayImage> {
    if a.dimensions() != b.dimensions() {
        return None;
    }
    let (w, h) = a.dimensions();
    let mut out = GrayImage::new(w, h);
    for (mask, (pa, pb)) in out.pixels_mut().zip(a.pixels().zip(b.pixels())) {
        *mask = Luma([if moved(pa.0, pb.0, threshold) { 255 } else { 0 }]);
    }
    Some(out)
}

/// Rec. 709 luma weights in 16-bit fixed point (0.2126, 0.7152, 0.0722).
/// They sum to exactly 65536, so a grey change of `d` levels weighs exactly
/// `d`; floating-point weights can fall a hair short and miss a change of
/// exactly the threshold.
const LUMA_WEIGHTS: [u32; 3] = [13933, 46871, 4732];

/// The luminance of the per-channel absolute difference of two pixels, in
/// 16-bit fixed point (`256 << 16` would be a change of 256 levels).
///
/// Taking the pixels as arrays rather than slices lets the compiler drop
/// every bounds check, which is what allows the row loop to vectorise.
#[inline]
fn weighted_difference(a: [u8; 4], b: [u8; 4]) -> u32 {
    LUMA_WEIGHTS[0] * u32::from(a[0].abs_diff(b[0]))
        + LUMA_WEIGHTS[1] * u32::from(a[1].abs_diff(b[1]))
        + LUMA_WEIGHTS[2] * u32::from(a[2].abs_diff(b[2]))
}

/// Whether one pixel changed enough to count as motion: the luminance of
/// the per-channel absolute difference reaches `threshold`. The single
/// definition shared by [`motion_mask`] and the detector's fused scan, so
/// the two cannot drift apart.
#[inline]
fn moved(a: [u8; 4], b: [u8; 4], threshold: u8) -> bool {
    weighted_difference(a, b) >= u32::from(threshold) << 16
}

/// A four-byte chunk of a row as one pixel.
#[inline]
fn as_pixel(chunk: &[u8]) -> [u8; 4] {
    chunk.try_into().expect("rows are split into 4-byte pixels")
}

/// Horizontal runs of moved pixels, and how many pixels moved in total,
/// computed in one pass over the two frames.
///
/// Equivalent to building [`motion_mask`] and then scanning it for runs,
/// without allocating the mask or reading it back: at 1366x768 that is a
/// megabyte written and read again every frame for nothing. Returns
/// `(y, start_x, end_x)` runs of at least `min_run_width`, in scan order.
fn motion_runs(
    a: &RgbaImage,
    b: &RgbaImage,
    threshold: u8,
    min_run_width: u32,
) -> (Vec<(u32, u32, u32)>, u64) {
    let (width, _) = a.dimensions();
    let row_bytes = width as usize * 4;
    let mut runs = Vec::new();
    let mut moved_pixels = 0u64;
    if row_bytes == 0 {
        return (runs, 0);
    }
    let limit = u32::from(threshold) << 16;
    let min_run = (min_run_width as usize).max(1);
    let mut verdicts = vec![0u8; width as usize];
    for (y, (row_a, row_b)) in a
        .as_raw()
        .chunks_exact(row_bytes)
        .zip(b.as_raw().chunks_exact(row_bytes))
        .enumerate()
    {
        // A row in which no byte changed by `threshold` holds no motion (the
        // luminance of a difference never exceeds its largest channel).
        // This reduction has no branches, so it compiles to wide SIMD and
        // lets the static parts of a frame cost almost nothing.
        let largest_change = row_a
            .iter()
            .zip(row_b)
            .map(|(pa, pb)| pa.abs_diff(*pb))
            .fold(0u8, u8::max);
        if largest_change < threshold {
            continue;
        }
        // One verdict byte per pixel, computed without branches so the loop
        // vectorises; runs are then read off the verdicts.
        let mut moved_in_row = 0u32;
        for (verdict, (pa, pb)) in verdicts
            .iter_mut()
            .zip(row_a.chunks_exact(4).zip(row_b.chunks_exact(4)))
        {
            let (pa, pb) = (as_pixel(pa), as_pixel(pb));
            let moved = u8::from(weighted_difference(pa, pb) >= limit);
            *verdict = moved;
            moved_in_row += u32::from(moved);
        }
        moved_pixels += u64::from(moved_in_row);
        // Too few moved pixels for any run to be long enough.
        if (moved_in_row as usize) < min_run {
            continue;
        }
        let y = y as u32;
        for_each_run(&verdicts, |start, end| {
            if end - start >= min_run {
                runs.push((y, start as u32, end as u32 - 1));
            }
        });
    }
    (runs, moved_pixels)
}

/// Configuration for the motion detector; exposed so callers can retune it
/// per resolution/scene without touching detection code.
#[derive(Debug, Clone, Copy)]
pub struct MotionConfig {
    /// Luminance-diff threshold (0-255) to count a pixel as moved.
    pub diff_threshold: u8,
    /// Minimum contiguous run width, in pixels, to consider a row segment.
    pub min_run_width: u32,
    /// Minimum blob height, in rows, after grouping.
    pub min_blob_height: u32,
    /// Minimum blob area, in pixels, to keep a candidate (rejects noise).
    pub min_blob_area: u32,
    /// Maximum matching distance (pixels) for the object tracker.
    pub track_match_distance: f32,
    /// How many consecutive missed frames a track survives (occlusion grace).
    pub track_grace_frames: u32,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            diff_threshold: 28,
            min_run_width: 4,
            min_blob_height: 4,
            min_blob_area: 24,
            track_match_distance: 48.0,
            track_grace_frames: 5,
        }
    }
}

/// A single moving blob tracked across frames.
#[derive(Debug, Clone, Copy)]
pub struct MovingBlob {
    /// Stable identity assigned by the tracker.
    pub id: u64,
    pub bounds: Rect,
    /// Per-frame displacement in pixels.
    pub velocity: (f32, f32),
    /// Consecutive frames this blob has been alive.
    pub age_frames: u32,
    /// True when the position is predicted rather than observed this frame.
    pub is_predicted: bool,
}

/// Stateful motion detector: owns the previous frame and an object tracker
/// so it can report temporally-consistent blobs rather than raw regions.
pub struct MotionDetector {
    config: MotionConfig,
    previous_frame: Option<RgbaImage>,
    tracker: ObjectTracker,
    last_diff_magnitude: f32,
}

impl MotionDetector {
    pub fn new(config: MotionConfig) -> Self {
        Self {
            tracker: ObjectTracker::new(config.track_match_distance, config.track_grace_frames),
            config,
            previous_frame: None,
            last_diff_magnitude: 0.0,
        }
    }

    /// Run motion detection for the current frame. The first call for a new
    /// detector instance always reports "missing" (no previous frame yet) —
    /// a normal warm-up condition, reported via the failure reason instead
    /// of an empty result that could be mistaken for "confirmed no motion".
    pub fn detect(&mut self, image: &RgbaImage) -> Detection<Vec<MovingBlob>> {
        let Some(previous) = self.previous_frame.as_ref() else {
            self.previous_frame = Some(image.clone());
            self.last_diff_magnitude = 0.0;
            return Detection::missing(
                "motion",
                "warming up: no previous frame to diff against yet",
            );
        };

        if previous.dimensions() != image.dimensions() {
            // Capture resolution changed between frames (e.g. window
            // resize); restart the baseline rather than reporting stale
            // motion computed against mismatched dimensions.
            self.previous_frame = Some(image.clone());
            self.last_diff_magnitude = 0.0;
            return Detection::missing("motion", "frame size changed since previous frame");
        }

        let (runs, moved_pixels) = motion_runs(
            previous,
            image,
            self.config.diff_threshold,
            self.config.min_run_width,
        );
        let total_pixels = image.width() as u64 * image.height() as u64;
        self.last_diff_magnitude = if total_pixels == 0 {
            0.0
        } else {
            moved_pixels as f32 / total_pixels as f32
        };
        let blobs = blobs_from_runs(runs, &self.config);

        // Keep this frame as the next baseline by copying into the buffer
        // already held: same size, so no allocation per frame.
        if let Some(previous) = self.previous_frame.as_mut() {
            previous.copy_from_slice(image.as_raw());
        }

        let detections: Vec<(f32, f32, f32, f32)> = blobs
            .iter()
            .map(|rect| {
                let (cx, cy) = rect.center();
                (cx, cy, rect.w as f32, rect.h as f32)
            })
            .collect();
        let tracks = self.tracker.update(&detections);

        let moving: Vec<MovingBlob> = tracks.iter().map(track_to_blob).collect();
        if moving.is_empty() {
            // A real "no motion this frame" result: previous frame existed,
            // diffed successfully, but nothing crossed the threshold.
            let mut detection = Detection::found(
                Vec::new(),
                Confidence::new(0.6),
                "motion",
                Reliability::Heuristic,
            );
            detection.failure_reason = Some("no motion above threshold".to_string());
            detection
        } else {
            let confidence = average_confidence(tracks);
            // Every blob being predicted means nothing was actually
            // redetected this frame: the positions come from the tracker's
            // linear extrapolation, not from pixels. Reporting that as
            // observed evidence would overstate what the frame showed.
            let reliability = if moving.iter().all(|blob| blob.is_predicted) {
                Reliability::Predicted
            } else if tracks.iter().any(|track| track.age_frames > 3) {
                Reliability::Corroborated
            } else {
                Reliability::Heuristic
            };
            Detection::found(moving, confidence, "motion", reliability)
        }
    }

    /// Fraction of pixels that moved in the most recent frame, in `[0, 1]`.
    pub fn last_diff_magnitude(&self) -> f32 {
        self.last_diff_magnitude
    }

    pub fn tracked_blob_count(&self) -> usize {
        self.tracker.tracks().len()
    }
}

fn track_to_blob(track: &Track) -> MovingBlob {
    MovingBlob {
        id: track.id,
        bounds: Rect {
            x: (track.position.x - track.width / 2.0).max(0.0) as u32,
            y: (track.position.y - track.height / 2.0).max(0.0) as u32,
            w: track.width.max(1.0) as u32,
            h: track.height.max(1.0) as u32,
        },
        velocity: (track.velocity.x, track.velocity.y),
        age_frames: track.age_frames,
        is_predicted: track.is_predicted(),
    }
}

fn average_confidence(tracks: &[Track]) -> Confidence {
    if tracks.is_empty() {
        return Confidence::NONE;
    }
    let total: f32 = tracks.iter().map(|t| t.confidence.value()).sum();
    Confidence::new(total / tracks.len() as f32)
}

fn blobs_from_runs(runs: Vec<(u32, u32, u32)>, config: &MotionConfig) -> Vec<Rect> {
    group_segments(runs, config.min_blob_height, 1)
        .into_iter()
        .filter(|rect| rect.area() >= config.min_blob_area)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn frame_with_square(w: u32, h: u32, at: (u32, u32), size: u32) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(w, h, Rgba([20, 20, 20, 255]));
        for y in at.1..(at.1 + size).min(h) {
            for x in at.0..(at.0 + size).min(w) {
                image.put_pixel(x, y, Rgba([230, 230, 230, 255]));
            }
        }
        image
    }

    #[test]
    fn motion_mask_flags_only_changed_pixels() {
        let a = frame_with_square(32, 32, (4, 4), 4);
        let b = frame_with_square(32, 32, (20, 4), 4);
        let mask = motion_mask(&a, &b, 28).unwrap();
        assert!(mask.get_pixel(5, 5).0[0] > 0, "vacated pixels are motion");
        assert!(mask.get_pixel(21, 5).0[0] > 0, "entered pixels are motion");
        assert_eq!(mask.get_pixel(15, 15).0[0], 0, "unchanged pixel is still");
    }

    /// A grey change of exactly the threshold counts; one level less does
    /// not. (Floating-point weights summing to a hair under 1 got this
    /// wrong for some levels.)
    #[test]
    fn grey_changes_are_measured_exactly() {
        let base = [0u8, 0, 0, 255];
        for threshold in 1..=255u8 {
            let at = [threshold, threshold, threshold, 255];
            let below = [threshold - 1, threshold - 1, threshold - 1, 255];
            assert!(moved(base, at, threshold), "grey {threshold}");
            assert!(!moved(base, below, threshold), "grey {}", threshold - 1);
        }
        assert!(moved(base, base, 0), "threshold 0 counts every pixel");
    }

    /// The fixed-point verdict agrees with Rec. 709 luminance everywhere
    /// except within rounding distance of the threshold.
    #[test]
    fn verdict_follows_rec709_luminance() {
        let levels: Vec<u8> = (0..=70u8).chain((71..=255u8).step_by(7)).collect();
        for &threshold in &[1u8, 2, 28, 29, 100, 254, 255] {
            for &r in &levels {
                for &g in &levels {
                    for &b in &levels {
                        let luminance = 0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64;
                        if (luminance - threshold as f64).abs() < 0.01 {
                            continue;
                        }
                        assert_eq!(
                            moved([0, 0, 0, 255], [r, g, b, 255], threshold),
                            luminance >= threshold as f64,
                            "diff {r},{g},{b} at threshold {threshold}"
                        );
                    }
                }
            }
        }
    }

    /// The fused scan must find exactly the runs (and moved-pixel count)
    /// that scanning the explicit mask finds.
    #[test]
    fn fused_runs_match_scanning_the_mask() {
        let mut a = RgbaImage::from_pixel(97, 61, Rgba([20, 20, 20, 255]));
        let mut b = a.clone();
        // A pseudo-random speckle of changes, plus runs touching both edges.
        let mut state = 12345u32;
        for y in 0..61 {
            for x in 0..97 {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let v = (state >> 16) as u8;
                if v > 170 {
                    b.put_pixel(x, y, Rgba([v, v / 2, 255 - v, 255]));
                }
            }
        }
        for x in 0..97 {
            a.put_pixel(x, 30, Rgba([250, 250, 250, 255]));
        }
        let mask = motion_mask(&a, &b, 28).unwrap();
        let mut expected = Vec::new();
        let mut expected_moved = 0u64;
        for y in 0..61u32 {
            let mut start = None;
            for x in 0..97u32 {
                if mask.get_pixel(x, y).0[0] > 0 {
                    expected_moved += 1;
                    start.get_or_insert(x);
                } else if let Some(begin) = start.take()
                    && x - begin >= 3
                {
                    expected.push((y, begin, x - 1));
                }
            }
            if let Some(begin) = start
                && 97 - begin >= 3
            {
                expected.push((y, begin, 96));
            }
        }
        let (runs, moved) = motion_runs(&a, &b, 28, 3);
        assert_eq!(runs, expected);
        assert_eq!(moved, expected_moved);
    }

    #[test]
    fn motion_mask_rejects_size_mismatch() {
        let a = RgbaImage::new(10, 10);
        let b = RgbaImage::new(12, 10);
        assert!(motion_mask(&a, &b, 28).is_none());
    }

    #[test]
    fn first_call_reports_warmup_not_failure_masquerading_as_empty() {
        let mut detector = MotionDetector::new(MotionConfig::default());
        let frame = frame_with_square(64, 64, (10, 10), 8);
        let detection = detector.detect(&frame);
        assert!(!detection.is_present());
        assert!(detection.failure_reason.unwrap().contains("warming up"));
    }

    #[test]
    fn moving_square_is_tracked_with_stable_id() {
        let mut detector = MotionDetector::new(MotionConfig::default());
        let frame1 = frame_with_square(80, 80, (10, 10), 10);
        let frame2 = frame_with_square(80, 80, (20, 10), 10);
        let frame3 = frame_with_square(80, 80, (30, 10), 10);

        detector.detect(&frame1);
        let second = detector.detect(&frame2);
        assert!(second.is_present());

        let third = detector.detect(&frame3);
        assert!(third.is_present());
        assert!(!third.value.unwrap().is_empty());
    }

    #[test]
    fn static_scene_reports_no_motion_confidently() {
        let mut detector = MotionDetector::new(MotionConfig::default());
        let frame = frame_with_square(64, 64, (10, 10), 8);
        detector.detect(&frame);
        let detection = detector.detect(&frame);
        assert!(detection.is_present());
        assert!(detection.value.unwrap().is_empty());
    }

    /// While an object is occluded its position is extrapolated, not seen.
    /// The detection must say so rather than presenting a predicted
    /// position as observed evidence.
    #[test]
    fn blobs_carried_through_occlusion_are_reported_as_predicted() {
        let mut detector = MotionDetector::new(MotionConfig::default());
        detector.detect(&frame_with_square(80, 80, (10, 10), 10));
        let observed = detector.detect(&frame_with_square(80, 80, (25, 10), 10));
        assert_eq!(observed.reliability, Reliability::Heuristic);

        // A frame identical to the previous one produces no new detections,
        // so every surviving track is coasting on prediction.
        let predicted = detector.detect(&frame_with_square(80, 80, (25, 10), 10));
        let blobs = predicted.value.as_ref().expect("tracks survive occlusion");
        assert!(!blobs.is_empty());
        assert!(blobs.iter().all(|blob| blob.is_predicted));
        assert_eq!(predicted.reliability, Reliability::Predicted);
    }

    #[test]
    fn resize_between_frames_restarts_the_baseline() {
        let mut detector = MotionDetector::new(MotionConfig::default());
        detector.detect(&RgbaImage::new(64, 64));
        let detection = detector.detect(&RgbaImage::new(32, 32));
        assert!(!detection.is_present());
        assert!(detection.failure_reason.unwrap().contains("size changed"));
    }
}
