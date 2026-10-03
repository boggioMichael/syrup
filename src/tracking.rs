//! A minimal centroid-distance multi-object tracker.
//!
//! Treating each frame independently throws away one of the strongest
//! signals in a video feed: continuity. The tracker assigns stable IDs to
//! detections across frames by matching them to where each track is
//! predicted to be, with a grace period so briefly-occluded objects are not
//! immediately dropped and a linear predictor for their position on frames
//! where they were not redetected.

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

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
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

    /// Where the track is expected to be on the next frame.
    fn predicted(&self) -> Point {
        Point {
            x: self.position.x + self.velocity.x,
            y: self.position.y + self.velocity.y,
        }
    }
}

/// A minimal centroid-distance multi-object tracker.
///
/// Each frame, detections are matched to where every track is predicted to
/// be by a globally optimal assignment: no pair further apart than the
/// match distance is ever matched, as many tracks as possible are matched,
/// and among those matchings the one with the smallest total distance wins.
/// The result does not depend on the order tracks or detections are listed
/// in, so two nearby objects cannot swap identities just because one track
/// happened to claim its nearest detection first.
///
/// There is no Kalman filter: the goal is robust "is this the same blob as
/// last frame" continuity for UI/gameplay blobs at a fixed camera, not
/// general-purpose tracking. Matching is solved separately for each group
/// of tracks and detections within reach of each other, so its cost stays
/// negligible at the scale of on-screen entities.
#[derive(Debug, Clone)]
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
    ///
    /// Detections with a non-finite coordinate are ignored.
    pub fn update(&mut self, detections: &[(f32, f32, f32, f32)]) -> &[Track] {
        self.assign(detections);
        &self.tracks
    }

    /// Like [`ObjectTracker::update`], but returns, for each detection in
    /// order, the id of the track it was matched to or opened: `None` for a
    /// detection with a non-finite coordinate, which is ignored.
    pub fn assign(&mut self, detections: &[(f32, f32, f32, f32)]) -> Vec<Option<u64>> {
        self.frame_index += 1;
        let finite: Vec<usize> = detections
            .iter()
            .enumerate()
            .filter(|(_, (x, y, w, h))| {
                x.is_finite() && y.is_finite() && w.is_finite() && h.is_finite()
            })
            .map(|(index, _)| index)
            .collect();
        let mut ids = vec![None; detections.len()];
        let detections: Vec<(f32, f32, f32, f32)> =
            finite.iter().map(|&index| detections[index]).collect();

        let predicted: Vec<Point> = self.tracks.iter().map(Track::predicted).collect();
        let observed: Vec<Point> = detections
            .iter()
            .map(|&(x, y, _, _)| Point { x, y })
            .collect();
        let track_for_detection = assign(&predicted, &observed, self.max_match_distance);

        let frame_index = self.frame_index;
        for ((&(x, y, w, h), matched), &index) in
            detections.iter().zip(track_for_detection).zip(&finite)
        {
            if let Some(track_idx) = matched {
                let track = &mut self.tracks[track_idx];
                ids[index] = Some(track.id);
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
                ids[index] = Some(self.next_id);
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
            track.position = track.predicted();
            track.confidence = track.confidence.decay(0.6);
            true
        });

        ids
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

/// Optimal one-to-one matching of `tracks` to `detections`, returned as the
/// matched track (if any) for each detection.
///
/// Pairs further apart than `max_distance` are never matched. Among the
/// others, the matching pairs as many as possible, then minimises the total
/// distance. Tracks and detections only compete with those within reach, so
/// the problem splits into independent groups — connected components of the
/// "within reach" graph — each solved exactly and usually tiny.
fn assign(tracks: &[Point], detections: &[Point], max_distance: f32) -> Vec<Option<usize>> {
    let mut matched = vec![None; detections.len()];
    let within = |t: usize, d: usize| {
        let distance = tracks[t].distance(detections[d]);
        (distance <= max_distance).then_some(distance)
    };

    // Group tracks and detections that can reach each other. Nodes
    // 0..tracks.len() are tracks; the rest are detections.
    let mut groups = UnionFind::new(tracks.len() + detections.len());
    let mut any_pair = false;
    for (t, track) in tracks.iter().enumerate() {
        if !track.is_finite() {
            continue;
        }
        for d in 0..detections.len() {
            if within(t, d).is_some() {
                groups.union(t, tracks.len() + d);
                any_pair = true;
            }
        }
    }
    if !any_pair {
        return matched;
    }

    let mut members: std::collections::BTreeMap<usize, (Vec<usize>, Vec<usize>)> =
        std::collections::BTreeMap::new();
    for t in 0..tracks.len() {
        members.entry(groups.find(t)).or_default().0.push(t);
    }
    for d in 0..detections.len() {
        members
            .entry(groups.find(tracks.len() + d))
            .or_default()
            .1
            .push(d);
    }

    for (group_tracks, group_detections) in members.values() {
        if group_tracks.is_empty() || group_detections.is_empty() {
            continue;
        }
        // A cost larger than any sum of real distances in the group, so the
        // solver first maximises the number of real pairs.
        let unmatchable = f64::from(max_distance.max(0.0))
            * (group_tracks.len() + group_detections.len()) as f64
            + 1.0;
        let cost = |t: usize, d: usize| within(t, d).map_or(unmatchable, f64::from);
        // The solver wants no more rows than columns.
        if group_tracks.len() <= group_detections.len() {
            let costs: Vec<Vec<f64>> = group_tracks
                .iter()
                .map(|&t| group_detections.iter().map(|&d| cost(t, d)).collect())
                .collect();
            for (row, column) in min_cost_assignment(&costs).into_iter().enumerate() {
                let (t, d) = (group_tracks[row], group_detections[column]);
                if within(t, d).is_some() {
                    matched[d] = Some(t);
                }
            }
        } else {
            let costs: Vec<Vec<f64>> = group_detections
                .iter()
                .map(|&d| group_tracks.iter().map(|&t| cost(t, d)).collect())
                .collect();
            for (row, column) in min_cost_assignment(&costs).into_iter().enumerate() {
                let (t, d) = (group_tracks[column], group_detections[row]);
                if within(t, d).is_some() {
                    matched[d] = Some(t);
                }
            }
        }
    }
    matched
}

/// Minimum-cost assignment of every row to a distinct column (the Hungarian
/// method with potentials, O(rows² × columns)). Requires at least as many
/// columns as rows; returns the column chosen for each row.
fn min_cost_assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let rows = costs.len();
    if rows == 0 {
        return Vec::new();
    }
    let columns = costs[0].len();
    debug_assert!(rows <= columns, "more rows than columns");

    // 1-based, with row/column 0 as the sentinel of the textbook algorithm.
    let mut row_potential = vec![0.0f64; rows + 1];
    let mut column_potential = vec![0.0f64; columns + 1];
    let mut row_of_column = vec![0usize; columns + 1];
    let mut previous_column = vec![0usize; columns + 1];
    for row in 1..=rows {
        row_of_column[0] = row;
        let mut column = 0usize;
        let mut slack = vec![f64::INFINITY; columns + 1];
        let mut visited = vec![false; columns + 1];
        loop {
            visited[column] = true;
            let current_row = row_of_column[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for j in 1..=columns {
                if visited[j] {
                    continue;
                }
                let reduced = costs[current_row - 1][j - 1]
                    - row_potential[current_row]
                    - column_potential[j];
                if reduced < slack[j] {
                    slack[j] = reduced;
                    previous_column[j] = column;
                }
                if slack[j] < delta {
                    delta = slack[j];
                    next_column = j;
                }
            }
            for j in 0..=columns {
                if visited[j] {
                    row_potential[row_of_column[j]] += delta;
                    column_potential[j] -= delta;
                } else {
                    slack[j] -= delta;
                }
            }
            column = next_column;
            if row_of_column[column] == 0 {
                break;
            }
        }
        // Flip the augmenting path.
        while column != 0 {
            let previous = previous_column[column];
            row_of_column[column] = row_of_column[previous];
            column = previous;
        }
    }

    let mut column_of_row = vec![0usize; rows];
    for (column, &row) in row_of_column.iter().enumerate().skip(1) {
        if row != 0 {
            column_of_row[row - 1] = column - 1;
        }
    }
    column_of_row
}

/// Disjoint sets with path halving; enough for grouping a few hundred nodes.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a.max(b)] = a.min(b);
        }
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
    fn assign_names_each_detection_s_track_in_order() {
        let mut tracker = ObjectTracker::new(20.0, 2);
        let first = tracker.assign(&[(10.0, 10.0, 4.0, 4.0), (80.0, 10.0, 4.0, 4.0)]);
        assert_eq!(first, vec![Some(1), Some(2)]);
        // Listed the other way round, with an unusable detection between.
        let second = tracker.assign(&[
            (82.0, 11.0, 4.0, 4.0),
            (f32::NAN, 0.0, 1.0, 1.0),
            (12.0, 10.0, 4.0, 4.0),
        ]);
        assert_eq!(second, vec![Some(2), None, Some(1)]);
        assert_eq!(tracker.tracks().len(), 2);
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

    fn position_of(tracker: &ObjectTracker, id: u64) -> f32 {
        tracker
            .tracks()
            .iter()
            .find(|track| track.id == id)
            .expect("track alive")
            .position
            .x
    }

    /// Two objects at x=6 and x=0 both step right, to 12 and 5. Letting the
    /// first track claim its nearest detection (5) would force the other
    /// to jump across it to 12; the optimal matching keeps both steps short.
    #[test]
    fn nearby_objects_keep_their_identities() {
        let mut tracker = ObjectTracker::new(15.0, 2);
        tracker.update(&[(6.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)]);
        tracker.update(&[(5.0, 0.0, 2.0, 2.0), (12.0, 0.0, 2.0, 2.0)]);
        assert_eq!(tracker.tracks().len(), 2, "no identity was lost");
        assert_eq!(position_of(&tracker, 1), 12.0);
        assert_eq!(position_of(&tracker, 2), 5.0);
    }

    /// The matching must not depend on the order detections arrive in.
    #[test]
    fn assignment_ignores_detection_order() {
        let tracks = [
            Point { x: 6.0, y: 0.0 },
            Point { x: 0.0, y: 0.0 },
            Point { x: 30.0, y: 4.0 },
        ];
        let detections = [
            Point { x: 5.0, y: 0.0 },
            Point { x: 12.0, y: 0.0 },
            Point { x: 33.0, y: 3.0 },
        ];
        let forward = assign(&tracks, &detections, 15.0);
        let reversed_detections: Vec<Point> = detections.iter().rev().copied().collect();
        let mut backward = assign(&tracks, &reversed_detections, 15.0);
        backward.reverse();
        assert_eq!(forward, backward);
        assert_eq!(forward, vec![Some(1), Some(0), Some(2)]);
    }

    /// Matching one track to its nearest detection (2 px) would strand the
    /// other track, which then has nothing within reach. Matching both is
    /// preferred over the single shortest pair.
    #[test]
    fn as_many_tracks_as_possible_are_matched() {
        let tracks = [Point { x: 0.0, y: 0.0 }, Point { x: 10.0, y: 0.0 }];
        let detections = [Point { x: 8.0, y: 0.0 }, Point { x: 20.0, y: 0.0 }];
        assert_eq!(assign(&tracks, &detections, 15.0), vec![Some(0), Some(1)]);
    }

    #[test]
    fn pairs_beyond_reach_are_never_matched() {
        let tracks = [Point { x: 0.0, y: 0.0 }];
        let detections = [Point { x: 16.0, y: 0.0 }];
        assert_eq!(assign(&tracks, &detections, 15.0), vec![None]);
        assert_eq!(assign(&[], &detections, 15.0), vec![None]);
        assert!(assign(&tracks, &[], 15.0).is_empty());
    }

    /// Exhaustive check against brute force on small random scenes.
    #[test]
    fn assignment_is_optimal() {
        fn brute_force(tracks: &[Point], detections: &[Point], reach: f32) -> (usize, f32) {
            fn go(
                t: usize,
                used: &mut Vec<bool>,
                tracks: &[Point],
                detections: &[Point],
                reach: f32,
            ) -> (usize, f32) {
                if t == tracks.len() {
                    return (0, 0.0);
                }
                let mut best = go(t + 1, used, tracks, detections, reach);
                for d in 0..detections.len() {
                    let distance = tracks[t].distance(detections[d]);
                    if used[d] || distance > reach {
                        continue;
                    }
                    used[d] = true;
                    let (count, total) = go(t + 1, used, tracks, detections, reach);
                    used[d] = false;
                    let candidate = (count + 1, total + distance);
                    if candidate.0 > best.0 || (candidate.0 == best.0 && candidate.1 < best.1) {
                        best = candidate;
                    }
                }
                best
            }
            go(
                0,
                &mut vec![false; detections.len()],
                tracks,
                detections,
                reach,
            )
        }

        let mut state = 0x2545_f491u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 1000) as f32 / 10.0
        };
        for _ in 0..400 {
            let track_count = (next() as usize) % 6;
            let detection_count = (next() as usize) % 6;
            let tracks: Vec<Point> = (0..track_count)
                .map(|_| Point {
                    x: next() / 2.0,
                    y: next() / 4.0,
                })
                .collect();
            let detections: Vec<Point> = (0..detection_count)
                .map(|_| Point {
                    x: next() / 2.0,
                    y: next() / 4.0,
                })
                .collect();
            let matched = assign(&tracks, &detections, 20.0);
            let mut used = vec![false; tracks.len()];
            let mut count = 0;
            let mut total = 0.0;
            for (d, track) in matched.iter().enumerate() {
                if let Some(t) = *track {
                    assert!(!used[t], "track matched twice");
                    used[t] = true;
                    let distance = tracks[t].distance(detections[d]);
                    assert!(distance <= 20.0);
                    count += 1;
                    total += distance;
                }
            }
            let (best_count, best_total) = brute_force(&tracks, &detections, 20.0);
            assert_eq!(count, best_count, "{tracks:?} {detections:?}");
            assert!((total - best_total).abs() < 1e-3, "{total} vs {best_total}");
        }
    }

    #[test]
    fn non_finite_detections_are_ignored() {
        let mut tracker = ObjectTracker::new(15.0, 1);
        tracker.update(&[(f32::NAN, 1.0, 2.0, 2.0), (4.0, 4.0, 2.0, 2.0)]);
        assert_eq!(tracker.tracks().len(), 1);
        tracker.update(&[(5.0, 4.0, 2.0, f32::INFINITY), (5.0, 4.0, 2.0, 2.0)]);
        assert_eq!(tracker.tracks().len(), 1);
        assert_eq!(tracker.tracks()[0].position.x, 5.0);
    }

    #[test]
    fn many_objects_in_one_cluster_stay_fast_and_consistent() {
        let mut tracker = ObjectTracker::new(40.0, 2);
        let row = |shift: f32| -> Vec<(f32, f32, f32, f32)> {
            (0..60)
                .map(|i| (i as f32 * 10.0 + shift, 0.0, 4.0, 4.0))
                .collect()
        };
        tracker.update(&row(0.0));
        let started = std::time::Instant::now();
        tracker.update(&row(3.0));
        assert!(started.elapsed().as_millis() < 500);
        assert_eq!(tracker.tracks().len(), 60);
        for track in tracker.tracks() {
            assert_eq!(track.velocity.x, 3.0, "track {} jumped", track.id);
        }
    }
}
