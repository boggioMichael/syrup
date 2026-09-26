//! Connected-component labelling of binary masks.
//!
//! Grouping pixels row by row (see [`crate::geometry::group_segments`]) is
//! fast and tolerant of small gaps, but it only ever produces rectangles and
//! it cannot tell two touching regions from one. Labelling finds the exact
//! pixel sets instead: every set pixel gets the id of the region it belongs
//! to, and each region reports its bounds, area and centroid.
//!
//! The implementation is the classic two-pass algorithm with a union-find
//! over provisional labels, so it is linear in the number of pixels and
//! allocates only the label buffer and the component list.

use image::{GrayImage, RgbaImage};

use crate::geometry::{Rect, row_pixels};

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
    label_with(
        (0, 0),
        width,
        height,
        |x, y| raw[(y * width + x) as usize] != 0,
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
    let mut set = vec![false; (width * height) as usize];
    for y in region.y..y_end {
        let row = ((y - region.y) * width) as usize;
        for (x, pixel) in row_pixels(image, y, region.x, x_end - 1) {
            set[row + (x - region.x) as usize] = predicate(pixel);
        }
    }
    label_with(
        (region.x, region.y),
        width,
        height,
        |x, y| set[(y * width + x) as usize],
        connectivity,
    )
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

fn label_with(
    origin: (u32, u32),
    width: u32,
    height: u32,
    is_set: impl Fn(u32, u32) -> bool,
    connectivity: Connectivity,
) -> (Labels, Vec<Component>) {
    let mut labels = vec![0u32; (width as usize) * (height as usize)];
    let mut sets = DisjointSet::new();
    let index = |x: u32, y: u32| (y * width + x) as usize;

    // First pass: provisional labels, recording equivalences.
    for y in 0..height {
        for x in 0..width {
            if !is_set(x, y) {
                continue;
            }
            let mut neighbours = [0u32; 4];
            let mut count = 0;
            let mut push = |label: u32| {
                if label != 0 {
                    neighbours[count] = label;
                    count += 1;
                }
            };
            if x > 0 {
                push(labels[index(x - 1, y)]);
            }
            if y > 0 {
                push(labels[index(x, y - 1)]);
                if connectivity == Connectivity::Eight {
                    if x > 0 {
                        push(labels[index(x - 1, y - 1)]);
                    }
                    if x + 1 < width {
                        push(labels[index(x + 1, y - 1)]);
                    }
                }
            }
            let label = match neighbours[..count].iter().min() {
                None => sets.make(),
                Some(&smallest) => {
                    for &other in &neighbours[..count] {
                        sets.union(smallest, other);
                    }
                    smallest
                }
            };
            labels[index(x, y)] = label;
        }
    }

    // Second pass: resolve each provisional label to its root, then renumber
    // roots densely in raster order and accumulate statistics.
    let mut dense = vec![0u32; sets.parent.len()];
    let mut stats: Vec<(u32, u32, u32, u32, u64, u64, u32)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let provisional = labels[index(x, y)];
            if provisional == 0 {
                continue;
            }
            let root = sets.find(provisional);
            if dense[root as usize] == 0 {
                stats.push((x, y, x, y, 0, 0, 0));
                dense[root as usize] = stats.len() as u32;
            }
            let label = dense[root as usize];
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

    let (ox, oy) = origin;
    let components = stats
        .into_iter()
        .enumerate()
        .map(|(i, (x0, y0, x1, y1, sx, sy, area))| Component {
            label: i as u32 + 1,
            bounds: Rect {
                x: ox + x0,
                y: oy + y0,
                w: x1 - x0 + 1,
                h: y1 - y0 + 1,
            },
            area,
            centroid: (
                ox as f32 + sx as f32 / area as f32,
                oy as f32 + sy as f32 / area as f32,
            ),
        })
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
}
