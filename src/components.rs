//! Connected-component labelling of binary masks.
//!
//! Grouping pixels row by row (see [`crate::geometry::group_segments`]) is
//! fast and tolerant of small gaps, but it only ever produces rectangles and
//! it cannot tell two touching regions from one. Labelling finds the exact
//! pixel sets instead: every set pixel gets the id of the region it belongs
//! to, and each region reports its bounds, area and centroid.
//!
//! The implementation labels horizontal runs of set pixels rather than
//! single pixels: each row is split into runs, each run is joined (with a
//! union-find) to the runs it touches in the row above, and the label map is
//! then filled run by run. Work beyond the one scan of the input is
//! proportional to the number of runs, not pixels, which on UI and game
//! masks — long spans of one value — is a small fraction.

use image::{GrayImage, RgbaImage};

use crate::geometry::{Rect, row_pixels};
use crate::scan::for_each_run;

/// Which neighbours count as touching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Up, down, left, right.
    Four,
    /// The four edge neighbours plus the diagonals. Thin diagonal strokes
    /// (a slash, the arm of a 7) stay in one piece.
    Eight,
}

/// One connected region of set pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Component {
    /// Label in the [`Labels`] map; labels start at 1 and follow the raster
    /// order of each component's first pixel.
    pub label: u32,
    pub bounds: Rect,
    /// Number of pixels in the component.
    pub area: u32,
    /// Mean pixel position `(x, y)`.
    pub centroid: (f32, f32),
}

/// Per-pixel component labels for a labelled area (0 is background).
#[derive(Debug, Clone)]
pub struct Labels {
    origin: (u32, u32),
    width: u32,
    height: u32,
    labels: Vec<u32>,
}

impl Labels {
    /// The label at image coordinates `(x, y)`; 0 outside the labelled area
    /// or on background.
    pub fn at(&self, x: u32, y: u32) -> u32 {
        let (ox, oy) = self.origin;
        if x < ox || y < oy || x - ox >= self.width || y - oy >= self.height {
            return 0;
        }
        self.labels[((y - oy) * self.width + (x - ox)) as usize]
    }

    /// The labelled area in image coordinates.
    pub fn area(&self) -> Rect {
        Rect {
            x: self.origin.0,
            y: self.origin.1,
            w: self.width,
            h: self.height,
        }
    }
}

/// Label the non-zero pixels of `mask`.
pub fn label_mask(mask: &GrayImage, connectivity: Connectivity) -> (Labels, Vec<Component>) {
    let (width, height) = mask.dimensions();
    let raw = mask.as_raw();
    let row_len = width as usize;
    label_runs(
        (0, 0),
        width,
        height,
        |y, runs| {
            let start = y as usize * row_len;
            push_byte_runs(y, &raw[start..start + row_len], runs);
        },
        connectivity,
    )
}

/// Label the pixels of `region` for which `predicate` holds.
///
/// The predicate is evaluated exactly once per pixel.
pub fn label_pixels<F>(
    image: &RgbaImage,
    region: Rect,
    connectivity: Connectivity,
    predicate: F,
) -> (Labels, Vec<Component>)
where
    F: Fn(&image::Rgba<u8>) -> bool,
{
    let x_end = region.x.saturating_add(region.w).min(image.width());
    let y_end = region.y.saturating_add(region.h).min(image.height());
    if region.x >= x_end || region.y >= y_end {
        return (
            Labels {
                origin: (region.x, region.y),
                width: 0,
                height: 0,
                labels: Vec::new(),
            },
            Vec::new(),
        );
    }
    let (width, height) = (x_end - region.x, y_end - region.y);
    label_runs(
        (region.x, region.y),
        width,
        height,
        |y, runs| {
            let pixels = row_pixels(image, region.y + y, region.x, x_end - 1);
            push_runs(y, pixels.map(|(_, pixel)| predicate(pixel)), runs);
        },
        connectivity,
    )
}

/// A horizontal run of set pixels: row `y`, columns `x0..=x1`, with its
/// provisional label.
#[derive(Debug, Clone, Copy)]
struct Run {
    y: u32,
    x0: u32,
    x1: u32,
    label: u32,
}

/// Append the runs of `true` in one row.
fn push_runs(y: u32, row: impl Iterator<Item = bool>, runs: &mut Vec<Run>) {
    let mut start: Option<u32> = None;
    let mut x = 0u32;
    for set in row {
        match (set, start) {
            (true, None) => start = Some(x),
            (false, Some(x0)) => {
                runs.push(Run {
                    y,
                    x0,
                    x1: x - 1,
                    label: 0,
                });
                start = None;
            }
            _ => {}
        }
        x += 1;
    }
    if let Some(x0) = start {
        runs.push(Run {
            y,
            x0,
            x1: x - 1,
            label: 0,
        });
    }
}

/// Append the runs of non-zero bytes in one mask row.
fn push_byte_runs(y: u32, row: &[u8], runs: &mut Vec<Run>) {
    for_each_run(row, |x0, x1| {
        runs.push(Run {
            y,
            x0: x0 as u32,
            x1: (x1 - 1) as u32,
            label: 0,
        })
    });
}

/// Union-find over provisional labels with path halving.
struct DisjointSet {
    parent: Vec<u32>,
}

impl DisjointSet {
    fn new() -> Self {
        // Index 0 is the background and never merges.
        Self { parent: vec![0] }
    }

    fn make(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Keep the smaller root so the final labels follow raster order.
            let (keep, merge) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[merge as usize] = keep;
        }
    }
}

/// Label the runs produced row by row by `row_runs`, which appends the runs
/// of row `y` (in increasing `x`) to the vector it is given.
fn label_runs(
    origin: (u32, u32),
    width: u32,
    height: u32,
    mut row_runs: impl FnMut(u32, &mut Vec<Run>),
    connectivity: Connectivity,
) -> (Labels, Vec<Component>) {
    // How far past its ends a run reaches into the row above: diagonal
    // neighbours touch under eight-connectivity.
    let reach = u32::from(connectivity == Connectivity::Eight);
    let mut runs: Vec<Run> = Vec::new();
    let mut sets = DisjointSet::new();
    let mut previous = 0..0;

    // First pass: provisional labels per run, joining runs that touch the
    // row above. Both rows are sorted by x, so one sweep pairs them up.
    for y in 0..height {
        let start = runs.len();
        row_runs(y, &mut runs);
        let mut candidate = previous.start;
        for i in start..runs.len() {
            let (x0, x1) = (runs[i].x0, runs[i].x1);
            // Runs above that end before this one's reach cannot touch it,
            // nor any later run in this row.
            while candidate < previous.end && runs[candidate].x1 + reach < x0 {
                candidate += 1;
            }
            let mut label = 0;
            let mut above = candidate;
            while above < previous.end && runs[above].x0 <= x1 + reach {
                let other = runs[above].label;
                if label == 0 {
                    label = other;
                } else {
                    sets.union(label, other);
                }
                above += 1;
            }
            runs[i].label = if label == 0 { sets.make() } else { label };
        }
        previous = start..runs.len();
    }

    // Second pass: resolve each run to its root, renumber roots densely in
    // raster order (runs already are), fill the map and gather statistics.
    let mut labels = vec![0u32; (width as usize) * (height as usize)];
    let mut dense = vec![0u32; sets.parent.len()];
    let mut stats: Vec<RegionStats> = Vec::new();
    for run in &runs {
        let root = sets.find(run.label) as usize;
        if dense[root] == 0 {
            stats.push(RegionStats::starting_at(run));
            dense[root] = stats.len() as u32;
        }
        let label = dense[root];
        let row = run.y as usize * width as usize;
        labels[row + run.x0 as usize..=row + run.x1 as usize].fill(label);
        stats[(label - 1) as usize].add(run);
    }

    let components = stats
        .into_iter()
        .enumerate()
        .map(|(i, stats)| stats.component(i as u32 + 1, origin))
        .collect();

    (
        Labels {
            origin,
            width,
            height,
            labels,
        },
        components,
    )
}

/// Running bounds, area and coordinate sums of one component.
struct RegionStats {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    sum_x: u64,
    sum_y: u64,
    area: u64,
}

impl RegionStats {
    fn starting_at(run: &Run) -> Self {
        Self {
            x0: run.x0,
            y0: run.y,
            x1: run.x1,
            y1: run.y,
            sum_x: 0,
            sum_y: 0,
            area: 0,
        }
    }

    fn add(&mut self, run: &Run) {
        let len = u64::from(run.x1 - run.x0 + 1);
        self.x0 = self.x0.min(run.x0);
        self.x1 = self.x1.max(run.x1);
        self.y0 = self.y0.min(run.y);
        self.y1 = self.y1.max(run.y);
        // Sum of x0..=x1; the product is always even.
        self.sum_x += (u64::from(run.x0) + u64::from(run.x1)) * len / 2;
        self.sum_y += u64::from(run.y) * len;
        self.area += len;
    }

    fn component(&self, label: u32, (ox, oy): (u32, u32)) -> Component {
        let area = self.area as f64;
        Component {
            label,
            bounds: Rect {
                x: ox + self.x0,
                y: oy + self.y0,
                w: self.x1 - self.x0 + 1,
                h: self.y1 - self.y0 + 1,
            },
            area: self.area as u32,
            centroid: (
                (f64::from(ox) + self.sum_x as f64 / area) as f32,
                (f64::from(oy) + self.sum_y as f64 / area) as f32,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    fn mask_from(rows: &[&str]) -> GrayImage {
        let height = rows.len() as u32;
        let width = rows[0].len() as u32;
        let mut mask = GrayImage::new(width, height);
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    mask.put_pixel(x as u32, y as u32, Luma([255]));
                }
            }
        }
        mask
    }

    #[test]
    fn separate_blobs_get_separate_labels_in_raster_order() {
        let mask = mask_from(&["##..#", "##..#", ".....", "..###"]);
        let (labels, comps) = label_mask(&mask, Connectivity::Four);
        assert_eq!(comps.len(), 3);
        assert_eq!(
            comps[0].bounds,
            Rect {
                x: 0,
                y: 0,
                w: 2,
                h: 2
            }
        );
        assert_eq!(comps[0].area, 4);
        assert_eq!(
            comps[1].bounds,
            Rect {
                x: 4,
                y: 0,
                w: 1,
                h: 2
            }
        );
        assert_eq!(
            comps[2].bounds,
            Rect {
                x: 2,
                y: 3,
                w: 3,
                h: 1
            }
        );
        assert_eq!(labels.at(1, 1), 1);
        assert_eq!(labels.at(4, 1), 2);
        assert_eq!(labels.at(3, 3), 3);
        assert_eq!(labels.at(2, 2), 0);
    }

    #[test]
    fn diagonal_touch_depends_on_connectivity() {
        let mask = mask_from(&["#...", ".#..", "..#.", "...#"]);
        assert_eq!(label_mask(&mask, Connectivity::Four).1.len(), 4);
        let (_, eight) = label_mask(&mask, Connectivity::Eight);
        assert_eq!(eight.len(), 1);
        assert_eq!(eight[0].area, 4);
        assert_eq!(eight[0].centroid, (1.5, 1.5));
    }

    /// A U shape joins two provisional labels on its last row; both arms
    /// must end up in one component.
    #[test]
    fn u_shapes_merge_into_one_component() {
        let mask = mask_from(&["#...#", "#...#", "#...#", "#####"]);
        let (labels, comps) = label_mask(&mask, Connectivity::Four);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].area, 11);
        assert_eq!(labels.at(0, 0), labels.at(4, 0));
    }

    /// A staircase of merges exercises the union-find beyond a single join.
    #[test]
    fn chained_merges_resolve_to_one_root() {
        let mask = mask_from(&["#.#.#.#", "#######"]);
        let (_, comps) = label_mask(&mask, Connectivity::Four);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].area, 11);
    }

    #[test]
    fn label_pixels_reports_image_coordinates() {
        let mut image = RgbaImage::from_pixel(20, 10, Rgba([0, 0, 0, 255]));
        for y in 4..6 {
            for x in 12..15 {
                image.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        let region = Rect {
            x: 10,
            y: 2,
            w: 8,
            h: 6,
        };
        let (labels, comps) = label_pixels(&image, region, Connectivity::Eight, |p| p[0] > 128);
        assert_eq!(comps.len(), 1);
        assert_eq!(
            comps[0].bounds,
            Rect {
                x: 12,
                y: 4,
                w: 3,
                h: 2
            }
        );
        assert_eq!(comps[0].centroid, (13.0, 4.5));
        assert_eq!(labels.at(13, 5), 1);
        assert_eq!(labels.at(0, 0), 0, "outside the labelled area");
        assert_eq!(labels.area(), region);
    }

    #[test]
    fn empty_and_out_of_bounds_regions_are_safe() {
        let image = RgbaImage::new(4, 4);
        let (_, comps) = label_pixels(
            &image,
            Rect {
                x: 10,
                y: 10,
                w: 5,
                h: 5,
            },
            Connectivity::Four,
            |_| true,
        );
        assert!(comps.is_empty());
        let (_, comps) = label_mask(&GrayImage::new(0, 0), Connectivity::Eight);
        assert!(comps.is_empty());
    }

    /// Cross-check against a flood fill on pseudo-random masks.
    #[test]
    fn matches_a_flood_fill_on_random_masks() {
        let mut state = 0x2545_f491u32;
        for connectivity in [Connectivity::Four, Connectivity::Eight] {
            for _ in 0..20 {
                let (w, h) = (23u32, 17u32);
                let mut mask = GrayImage::new(w, h);
                for p in mask.pixels_mut() {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    if state % 100 < 45 {
                        *p = Luma([1]);
                    }
                }
                let (labels, comps) = label_mask(&mask, connectivity);
                // Flood fill reference.
                let mut seen = vec![false; (w * h) as usize];
                let mut areas = Vec::new();
                for y in 0..h {
                    for x in 0..w {
                        if mask.get_pixel(x, y).0[0] == 0 || seen[(y * w + x) as usize] {
                            continue;
                        }
                        let mut stack = vec![(x, y)];
                        seen[(y * w + x) as usize] = true;
                        let mut area = 0;
                        let label = labels.at(x, y);
                        while let Some((cx, cy)) = stack.pop() {
                            area += 1;
                            assert_eq!(labels.at(cx, cy), label, "one region, two labels");
                            for (dx, dy) in [
                                (-1i32, 0i32),
                                (1, 0),
                                (0, -1),
                                (0, 1),
                                (-1, -1),
                                (1, -1),
                                (-1, 1),
                                (1, 1),
                            ] {
                                if connectivity == Connectivity::Four && dx != 0 && dy != 0 {
                                    continue;
                                }
                                let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                                    continue;
                                }
                                let (nx, ny) = (nx as u32, ny as u32);
                                if mask.get_pixel(nx, ny).0[0] != 0 && !seen[(ny * w + nx) as usize]
                                {
                                    seen[(ny * w + nx) as usize] = true;
                                    stack.push((nx, ny));
                                }
                            }
                        }
                        areas.push(area);
                    }
                }
                let mut expected: Vec<u32> = areas;
                let mut actual: Vec<u32> = comps.iter().map(|c| c.area).collect();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected);
            }
        }
    }

    /// The textbook pixel-by-pixel two-pass labelling the run-based version
    /// replaced; the two must agree exactly.
    fn reference_labels(
        mask: &GrayImage,
        connectivity: Connectivity,
    ) -> (Vec<u32>, Vec<Component>) {
        let (width, height) = mask.dimensions();
        let is_set = |x: u32, y: u32| mask.get_pixel(x, y).0[0] != 0;
        let index = |x: u32, y: u32| (y * width + x) as usize;
        let mut labels = vec![0u32; (width * height) as usize];
        let mut sets = DisjointSet::new();
        for y in 0..height {
            for x in 0..width {
                if !is_set(x, y) {
                    continue;
                }
                let mut neighbours = Vec::new();
                if x > 0 {
                    neighbours.push(labels[index(x - 1, y)]);
                }
                if y > 0 {
                    neighbours.push(labels[index(x, y - 1)]);
                    if connectivity == Connectivity::Eight {
                        if x > 0 {
                            neighbours.push(labels[index(x - 1, y - 1)]);
                        }
                        if x + 1 < width {
                            neighbours.push(labels[index(x + 1, y - 1)]);
                        }
                    }
                }
                neighbours.retain(|&l| l != 0);
                let label = match neighbours.iter().min() {
                    None => sets.make(),
                    Some(&smallest) => {
                        for &other in &neighbours {
                            sets.union(smallest, other);
                        }
                        smallest
                    }
                };
                labels[index(x, y)] = label;
            }
        }
        let mut dense = vec![0u32; sets.parent.len()];
        let mut stats: Vec<(u32, u32, u32, u32, u64, u64, u32)> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let provisional = labels[index(x, y)];
                if provisional == 0 {
                    continue;
                }
                let root = sets.find(provisional) as usize;
                if dense[root] == 0 {
                    stats.push((x, y, x, y, 0, 0, 0));
                    dense[root] = stats.len() as u32;
                }
                let label = dense[root];
                labels[index(x, y)] = label;
                let s = &mut stats[(label - 1) as usize];
                s.0 = s.0.min(x);
                s.1 = s.1.min(y);
                s.2 = s.2.max(x);
                s.3 = s.3.max(y);
                s.4 += x as u64;
                s.5 += y as u64;
                s.6 += 1;
            }
        }
        let components = stats
            .into_iter()
            .enumerate()
            .map(|(i, (x0, y0, x1, y1, sx, sy, area))| Component {
                label: i as u32 + 1,
                bounds: Rect {
                    x: x0,
                    y: y0,
                    w: x1 - x0 + 1,
                    h: y1 - y0 + 1,
                },
                area,
                centroid: (sx as f32 / area as f32, sy as f32 / area as f32),
            })
            .collect();
        (labels, components)
    }

    #[test]
    fn runs_agree_with_pixel_labelling_exactly() {
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for connectivity in [Connectivity::Four, Connectivity::Eight] {
            for density in [5u32, 30, 50, 70, 95] {
                for (w, h) in [(1u32, 1u32), (1, 9), (9, 1), (13, 7), (40, 31)] {
                    let mut mask = GrayImage::new(w, h);
                    for p in mask.pixels_mut() {
                        if next() % 100 < density {
                            *p = Luma([next() as u8 | 1]);
                        }
                    }
                    let (labels, comps) = label_mask(&mask, connectivity);
                    let (want_labels, want_comps) = reference_labels(&mask, connectivity);
                    assert_eq!(labels.labels, want_labels, "{w}x{h} at {density}%");
                    assert_eq!(comps.len(), want_comps.len());
                    for (got, want) in comps.iter().zip(&want_comps) {
                        assert_eq!(
                            (got.label, got.bounds, got.area),
                            (want.label, want.bounds, want.area)
                        );
                        assert!((got.centroid.0 - want.centroid.0).abs() < 1e-3);
                        assert!((got.centroid.1 - want.centroid.1).abs() < 1e-3);
                    }

                    // The predicate path must label identically.
                    let image = RgbaImage::from_fn(w, h, |x, y| {
                        let v = mask.get_pixel(x, y).0[0];
                        Rgba([v, 0, 0, 255])
                    });
                    let region = Rect { x: 0, y: 0, w, h };
                    let (pixel_labels, pixel_comps) =
                        label_pixels(&image, region, connectivity, |p| p[0] != 0);
                    assert_eq!(pixel_labels.labels, want_labels);
                    assert_eq!(pixel_comps, comps);
                }
            }
        }
    }

    #[test]
    fn byte_runs_are_inclusive_and_carry_their_row() {
        let mut row = vec![0u8; 37];
        row[8] = 3;
        row[20..30].fill(1);
        row[36] = 9;
        let mut runs = Vec::new();
        push_byte_runs(4, &row, &mut runs);
        let spans: Vec<(u32, u32, u32)> = runs.iter().map(|r| (r.y, r.x0, r.x1)).collect();
        assert_eq!(spans, vec![(4, 8, 8), (4, 20, 29), (4, 36, 36)]);
    }
}
