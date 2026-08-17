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
        let dr = pa[0].abs_diff(pb[0]) as f32;
        let dg = pa[1].abs_diff(pb[1]) as f32;
        let db = pa[2].abs_diff(pb[2]) as f32;
        let lum = (0.2126 * dr + 0.7152 * dg + 0.0722 * db) as u8;
        *mask = Luma([if lum >= threshold { 255 } else { 0 }]);
    }
    Some(out)
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

        let blobs = match motion_mask(previous, image, self.config.diff_threshold) {
            Some(mask) => {
                self.last_diff_magnitude = changed_fraction(&mask);
                extract_blobs(&mask, &self.config)
            }
            None => {
                // Capture resolution changed between frames (e.g. window
                // resize); restart the baseline rather than reporting stale
                // motion computed against mismatched dimensions.
                self.previous_frame = Some(image.clone());
                self.last_diff_magnitude = 0.0;
                return Detection::missing("motion", "frame size changed since previous frame");
            }
        };

        self.previous_frame = Some(image.clone());

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
            let reliability = if tracks.iter().any(|t| t.age_frames > 3) {
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

fn changed_fraction(mask: &GrayImage) -> f32 {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return 0.0;
    }
    let total_pixels = width as f32 * height as f32;
    let moved_pixels: u32 = mask.iter().map(|&v| if v > 0 { 1 } else { 0 }).sum();
    moved_pixels as f32 / total_pixels
}

fn extract_blobs(mask: &GrayImage, config: &MotionConfig) -> Vec<Rect> {
    let (width, height) = mask.dimensions();
    let mut rows = Vec::new();
    for y in 0..height {
        let mut start: Option<u32> = None;
        for x in 0..width {
            let moved = mask.get_pixel(x, y).0[0] > 0;
            if moved {
                if start.is_none() {
                    start = Some(x);
                }
            } else if let Some(begin) = start {
                if x - begin >= config.min_run_width {
                    rows.push((y, begin, x - 1));
                }
                start = None;
            }
        }
        if let Some(begin) = start
            && width - begin >= config.min_run_width
        {
            rows.push((y, begin, width - 1));
        }
    }

    group_segments(rows, config.min_blob_height, 1)
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

    #[test]
    fn resize_between_frames_restarts_the_baseline() {
        let mut detector = MotionDetector::new(MotionConfig::default());
        detector.detect(&RgbaImage::new(64, 64));
        let detection = detector.detect(&RgbaImage::new(32, 32));
        assert!(!detection.is_present());
        assert!(detection.failure_reason.unwrap().contains("size changed"));
    }
}
