//! A minimal centroid-distance multi-object tracker.
//!
//! Treating each frame independently throws away one of the strongest
//! signals in a video feed: continuity. The tracker assigns stable IDs to
//! detections across frames using nearest-centroid matching, with a grace
//! period so briefly-occluded objects are not immediately dropped and a
//! linear predictor for their position on frames where they were not
//! redetected.

use std::collections::HashMap;

use crate::detection::Confidence;

/// A 2D point used for tracker centroids/velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn distance(self, other: Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A single tracked object with a stable identity across frames.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub position: Point,
    pub velocity: Point,
    pub width: f32,
    pub height: f32,
    /// Frame index this track was last matched to a real detection.
    pub last_seen_frame: u64,
    /// Number of consecutive frames without a redetection. Used for
    /// occlusion-recovery grace periods.
    pub missed_frames: u32,
    /// Number of consecutive frames this track has been alive, used to grow
    /// confidence for stable tracks.
    pub age_frames: u32,
    pub confidence: Confidence,
}

impl Track {
    /// True while the object is still redetected or within the tracker's
    /// grace period; the tracker itself owns the grace threshold.
    pub fn is_predicted(&self) -> bool {
        self.missed_frames > 0
    }
}

/// A minimal centroid-distance multi-object tracker.
///
/// This is intentionally simple (no Kalman filter, no Hungarian assignment)
/// because the goal is robust "is this the same blob as last frame"
/// continuity for UI/gameplay blobs at a fixed camera, not general-purpose
/// SLAM-grade tracking. Complexity is O(existing_tracks * new_detections)
/// per frame, which is negligible at the scale of on-screen entities.
pub struct ObjectTracker {
    tracks: Vec<Track>,
    next_id: u64,
    max_match_distance: f32,
    grace_frames: u32,
    frame_index: u64,
}

impl ObjectTracker {
    pub fn new(max_match_distance: f32, grace_frames: u32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_match_distance,
            grace_frames,
            frame_index: 0,
        }
    }

    /// Advance the tracker by one frame given the raw detections observed
    /// this frame (center x, center y, width, height). Returns the current
    /// set of live tracks (both redetected and predicted-through-occlusion).
    pub fn update(&mut self, detections: &[(f32, f32, f32, f32)]) -> &[Track] {
        self.frame_index += 1;
        let mut used = vec![false; detections.len()];
        let mut matched_ids: HashMap<usize, usize> = HashMap::new();

        // Greedy nearest-neighbor assignment: for each existing track, pick
        // the closest not-yet-claimed detection within the match radius.
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let predicted = Point {
                x: track.position.x + track.velocity.x,
                y: track.position.y + track.velocity.y,
            };
            let mut best: Option<(usize, f32)> = None;
            for (det_idx, &(x, y, _, _)) in detections.iter().enumerate() {
                if used[det_idx] {
                    continue;
                }
                let dist = predicted.distance(Point { x, y });
                if dist <= self.max_match_distance && best.map(|(_, d)| dist < d).unwrap_or(true) {
                    best = Some((det_idx, dist));
                }
            }
            if let Some((det_idx, _)) = best {
                used[det_idx] = true;
                matched_ids.insert(det_idx, track_idx);
            }
        }

        let frame_index = self.frame_index;
        for (det_idx, &(x, y, w, h)) in detections.iter().enumerate() {
            if let Some(&track_idx) = matched_ids.get(&det_idx) {
                let track = &mut self.tracks[track_idx];
                let new_position = Point { x, y };
                track.velocity = Point {
                    x: new_position.x - track.position.x,
                    y: new_position.y - track.position.y,
                };
                track.position = new_position;
                track.width = w;
                track.height = h;
                track.last_seen_frame = frame_index;
                track.missed_frames = 0;
                track.age_frames = track.age_frames.saturating_add(1);
                // Stable, repeatedly redetected tracks become more trustworthy.
                track.confidence = track.confidence.combine(Confidence::new(0.35)).decay(0.995);
            } else {
                self.tracks.push(Track {
                    id: self.next_id,
                    position: Point { x, y },
                    velocity: Point { x: 0.0, y: 0.0 },
                    width: w,
                    height: h,
                    last_seen_frame: frame_index,
                    missed_frames: 0,
                    age_frames: 1,
                    confidence: Confidence::new(0.3),
                });
                self.next_id += 1;
            }
        }

        // Any track not matched this frame either enters (or continues) its
        // occlusion grace period with a linearly predicted position, or is
        // dropped once the grace period elapses.
        self.tracks.retain_mut(|track| {
            if track.last_seen_frame == frame_index {
                return true;
            }
            track.missed_frames += 1;
            if track.missed_frames > self.grace_frames {
                return false;
            }
            track.position = Point {
                x: track.position.x + track.velocity.x,
                y: track.position.y + track.velocity.y,
            };
            track.confidence = track.confidence.decay(0.6);
            true
        });

        &self.tracks
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_assigns_stable_ids_across_frames() {
        let mut tracker = ObjectTracker::new(15.0, 3);
        tracker.update(&[(10.0, 10.0, 4.0, 4.0)]);
        let ids_after_first: Vec<u64> = tracker.tracks().iter().map(|t| t.id).collect();
        // Object moves slightly next frame; should keep the same id.
        tracker.update(&[(12.0, 11.0, 4.0, 4.0)]);
        let ids_after_second: Vec<u64> = tracker.tracks().iter().map(|t| t.id).collect();
        assert_eq!(ids_after_first, ids_after_second);
    }

    #[test]
    fn tracker_survives_brief_occlusion() {
        let mut tracker = ObjectTracker::new(15.0, 2);
        tracker.update(&[(10.0, 10.0, 4.0, 4.0)]);
        let id = tracker.tracks()[0].id;
        // Missing for one frame (occlusion) should not drop the track.
        tracker.update(&[]);
        assert_eq!(tracker.tracks().len(), 1);
        assert!(tracker.tracks()[0].is_predicted());
        tracker.update(&[(11.0, 10.0, 4.0, 4.0)]);
        assert_eq!(tracker.tracks()[0].id, id);
    }

    #[test]
    fn tracker_drops_object_after_grace_period() {
        let mut tracker = ObjectTracker::new(15.0, 1);
        tracker.update(&[(10.0, 10.0, 4.0, 4.0)]);
        tracker.update(&[]);
        tracker.update(&[]);
        assert!(tracker.tracks().is_empty());
    }
}
