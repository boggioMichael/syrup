//! Boosted cascades of Haar-like features: the Viola–Jones object detector.
//!
//! A cascade is a sequence of stages, each a small boosted ensemble of
//! decision trees over rectangle-sum features. A window slides over the
//! image at every scale; each stage either rejects the window or passes it
//! to the next, so the overwhelming majority of windows are dismissed after
//! a handful of features and only the few that look like the object pay
//! for the whole classifier (Viola & Jones 2001; Lienhart & Maydt 2002).
//!
//! The cascades themselves are trained models. This module ships OpenCV's
//! frontal-face cascade (`assets/cascades`, with its licence) in a compact
//! binary form and evaluates it exactly as OpenCV does — same features,
//! same variance normalisation, same tree encoding — so its behaviour is
//! cross-checked against a widely used reference rather than invented.
//!
//! ```text
//!   grey image ─▶ scale pyramid ─▶ integral + squared integral
//!                                   │
//!                                   ▼ every window, every scale
//!                          stage 1 → stage 2 → … → stage n   (reject early)
//!                                   │
//!                                   ▼ surviving windows
//!                          cluster overlapping windows ─▶ matches + support
//! ```

use std::sync::OnceLock;

use image::{GrayImage, RgbaImage, imageops};

use crate::geometry::Rect;
use crate::threshold::{Channel, channel_image};

/// OpenCV's cascades, converted by `tools/cascade2bin.py`. See
/// `assets/cascades/LICENSE-opencv-cascades.txt` for their notices.
const FRONTAL_FACE: &[u8] = include_bytes!("../assets/cascades/frontalface_alt2.bin");
const EYE: &[u8] = include_bytes!("../assets/cascades/eye.bin");
const PROFILE_FACE: &[u8] = include_bytes!("../assets/cascades/profileface.bin");

/// Why a cascade file could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeError {
    /// Not a syrup cascade file.
    BadMagic,
    /// A file version this build does not read.
    UnsupportedVersion(u8),
    /// The file ended, or an index pointed outside the tables.
    Truncated,
}

impl std::fmt::Display for CascadeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CascadeError::BadMagic => write!(f, "not a syrup cascade file"),
            CascadeError::UnsupportedVersion(v) => {
                write!(f, "cascade file version {v} is not supported")
            }
            CascadeError::Truncated => write!(f, "cascade file is truncated or inconsistent"),
        }
    }
}

impl std::error::Error for CascadeError {}

/// One rectangle of a Haar feature, in window coordinates.
#[derive(Debug, Clone, Copy)]
struct WeightedRect {
    x: u8,
    y: u8,
    w: u8,
    h: u8,
    weight: f32,
}

/// A Haar-like feature: the weighted sum of two or three rectangles.
#[derive(Debug, Clone, Copy)]
struct Feature {
    rects: [WeightedRect; 3],
    count: u8,
}

/// A node of a weak classifier's decision tree. `left`/`right` follow
/// OpenCV: a positive value is the next node's index within the tree,
/// zero or negative is `-(leaf index)`.
#[derive(Debug, Clone, Copy)]
struct Node {
    left: i32,
    right: i32,
    feature: u32,
    threshold: f32,
}

#[derive(Debug, Clone, Copy)]
struct Weak {
    first_node: u32,
    first_leaf: u32,
}

#[derive(Debug, Clone)]
struct Stage {
    threshold: f32,
    /// Indices into `Cascade::weak`.
    weak: std::ops::Range<usize>,
}

/// A trained cascade classifier.
#[derive(Debug, Clone)]
pub struct Cascade {
    window: (u32, u32),
    features: Vec<Feature>,
    stages: Vec<Stage>,
    weak: Vec<Weak>,
    nodes: Vec<Node>,
    leaves: Vec<f32>,
}

/// A little-endian cursor over the binary cascade format.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CascadeError> {
        let end = self.at.checked_add(n).ok_or(CascadeError::Truncated)?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or(CascadeError::Truncated)?;
        self.at = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, CascadeError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, CascadeError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, CascadeError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn i32(&mut self) -> Result<i32, CascadeError> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn f32(&mut self) -> Result<f32, CascadeError> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
}

impl Cascade {
    /// The bundled frontal-face detector (`haarcascade_frontalface_alt2`).
    pub fn frontal_face() -> &'static Cascade {
        static CASCADE: OnceLock<Cascade> = OnceLock::new();
        CASCADE.get_or_init(|| Cascade::from_bytes(FRONTAL_FACE).expect("bundled cascade is valid"))
    }

    /// The bundled eye detector (`haarcascade_eye`), meant to run inside a
    /// face rather than over a whole frame.
    pub fn eye() -> &'static Cascade {
        static CASCADE: OnceLock<Cascade> = OnceLock::new();
        CASCADE.get_or_init(|| Cascade::from_bytes(EYE).expect("bundled cascade is valid"))
    }

    /// The bundled profile-face detector (`haarcascade_profileface`): faces
    /// turned to the side, one direction only, as OpenCV ships it.
    pub fn profile_face() -> &'static Cascade {
        static CASCADE: OnceLock<Cascade> = OnceLock::new();
        CASCADE.get_or_init(|| Cascade::from_bytes(PROFILE_FACE).expect("bundled cascade is valid"))
    }

    /// Load a cascade in syrup's binary form (see `tools/cascade2bin.py`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CascadeError> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != b"SYRC" {
            return Err(CascadeError::BadMagic);
        }
        let version = r.u8()?;
        if version != 1 {
            return Err(CascadeError::UnsupportedVersion(version));
        }
        let window = (u32::from(r.u16()?), u32::from(r.u16()?));
        if window.0 < 3 || window.1 < 3 {
            return Err(CascadeError::Truncated);
        }

        let n_features = r.u32()? as usize;
        let mut features = Vec::with_capacity(n_features.min(1 << 16));
        for _ in 0..n_features {
            let count = r.u8()?;
            if count == 0 || count > 3 {
                return Err(CascadeError::Truncated);
            }
            let mut rects = [WeightedRect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
                weight: 0.0,
            }; 3];
            for rect in rects.iter_mut().take(count as usize) {
                let (x, y, w, h) = (r.u8()?, r.u8()?, r.u8()?, r.u8()?);
                if u32::from(x) + u32::from(w) > window.0 || u32::from(y) + u32::from(h) > window.1
                {
                    return Err(CascadeError::Truncated);
                }
                *rect = WeightedRect {
                    x,
                    y,
                    w,
                    h,
                    weight: r.f32()?,
                };
            }
            features.push(Feature { rects, count });
        }

        let n_stages = r.u32()? as usize;
        let mut stages = Vec::with_capacity(n_stages.min(1 << 10));
        let mut weak = Vec::new();
        let mut nodes = Vec::new();
        let mut leaves = Vec::new();
        for _ in 0..n_stages {
            let threshold = r.f32()?;
            let n_weak = r.u32()? as usize;
            let first = weak.len();
            for _ in 0..n_weak {
                let n_nodes = r.u8()? as usize;
                if n_nodes == 0 {
                    return Err(CascadeError::Truncated);
                }
                let first_node = nodes.len() as u32;
                let first_leaf = leaves.len() as u32;
                for _ in 0..n_nodes {
                    let node = Node {
                        left: r.i32()?,
                        right: r.i32()?,
                        feature: r.u32()?,
                        threshold: r.f32()?,
                    };
                    let valid = |link: i32| {
                        (link > 0 && (link as usize) < n_nodes)
                            || (link <= 0 && (-link as usize) <= n_nodes)
                    };
                    if node.feature as usize >= features.len()
                        || !valid(node.left)
                        || !valid(node.right)
                    {
                        return Err(CascadeError::Truncated);
                    }
                    nodes.push(node);
                }
                for _ in 0..=n_nodes {
                    leaves.push(r.f32()?);
                }
                weak.push(Weak {
                    first_node,
                    first_leaf,
                });
            }
            stages.push(Stage {
                threshold,
                weak: first..weak.len(),
            });
        }
        if stages.is_empty() {
            return Err(CascadeError::Truncated);
        }
        Ok(Self {
            window,
            features,
            stages,
            weak,
            nodes,
            leaves,
        })
    }

    /// The window the cascade was trained on, `(width, height)`.
    pub fn window(&self) -> (u32, u32) {
        self.window
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Run the cascade on the window whose top-left corner is at index
    /// `base` of the integral image. Returns how many stages the window
    /// passed; a window is accepted when it passes all of them.
    fn stages_passed(&self, integral: &Integral, level: &Level, base: usize) -> usize {
        // Variance normalisation over the window with a one-pixel border
        // trimmed, exactly as the reference implementation does.
        let sum = integral.sum_at(base, &level.norm) as f64;
        let squares = integral.squares_at(base, &level.norm) as f64;
        let variance = level.norm_area * squares - sum * sum;
        let inverse_deviation = if variance > 0.0 {
            (1.0 / variance.sqrt()) as f32
        } else {
            1.0
        };

        let sums = &integral.sums;
        let mut passed = 0;
        for stage in &self.stages {
            let mut total = 0.0f32;
            for weak in &self.weak[stage.weak.clone()] {
                let mut index = 0i32;
                loop {
                    let node = &self.nodes[weak.first_node as usize + index as usize];
                    let mut value = 0.0f32;
                    for rect in &level.rects[level.first_rect[node.feature as usize]
                        ..level.first_rect[node.feature as usize + 1]]
                    {
                        let area = sums[base + rect.offsets[0]] + sums[base + rect.offsets[3]]
                            - sums[base + rect.offsets[1]]
                            - sums[base + rect.offsets[2]];
                        value += rect.weight * area as f32;
                    }
                    index = if value * inverse_deviation < node.threshold {
                        node.left
                    } else {
                        node.right
                    };
                    if index <= 0 {
                        break;
                    }
                }
                total += self.leaves[weak.first_leaf as usize + (-index) as usize];
            }
            if total < stage.threshold {
                return passed;
            }
            passed += 1;
        }
        passed
    }

    /// Feature rectangles as offsets into an integral image of the given
    /// stride, so evaluating a window is pure loads and adds.
    fn level(&self, stride: usize) -> Level {
        let (ww, wh) = self.window;
        let corners = |x: u32, y: u32, w: u32, h: u32| -> [usize; 4] {
            let (x0, y0, x1, y1) = (x as usize, y as usize, (x + w) as usize, (y + h) as usize);
            [
                y0 * stride + x0,
                y0 * stride + x1,
                y1 * stride + x0,
                y1 * stride + x1,
            ]
        };
        let mut rects = Vec::with_capacity(self.features.len() * 3);
        let mut first_rect = Vec::with_capacity(self.features.len() + 1);
        for feature in &self.features {
            first_rect.push(rects.len());
            for rect in &feature.rects[..feature.count as usize] {
                rects.push(OffsetRect {
                    offsets: corners(rect.x.into(), rect.y.into(), rect.w.into(), rect.h.into()),
                    weight: rect.weight,
                });
            }
        }
        first_rect.push(rects.len());
        Level {
            rects,
            first_rect,
            norm: corners(1, 1, ww - 2, wh - 2),
            norm_area: f64::from((ww - 2) * (wh - 2)),
        }
    }

    /// Every window of `gray`, at every scale, that passes the whole
    /// cascade — before neighbouring windows are merged.
    pub fn raw_windows(&self, gray: &GrayImage, options: &CascadeOptions) -> Vec<Rect> {
        let (ww, wh) = self.window;
        let (width, height) = gray.dimensions();
        let mut windows = Vec::new();
        if width < ww || height < wh {
            return windows;
        }
        let factor = options.scale_factor.max(1.01);
        let min_side = options.min_size.max(ww.min(wh)) as f32;
        let max_side = options.max_size.map(|m| m as f32).unwrap_or(f32::MAX);
        let mut scale = min_side / ww.min(wh) as f32;
        loop {
            let window_w = (ww as f32 * scale).round() as u32;
            let window_h = (wh as f32 * scale).round() as u32;
            if window_w > width || window_h > height || window_w.min(window_h) as f32 > max_side {
                break;
            }
            let scaled_w = (width as f32 / scale).round() as u32;
            let scaled_h = (height as f32 / scale).round() as u32;
            if scaled_w < ww || scaled_h < wh {
                break;
            }
            let level = if (scaled_w, scaled_h) == (width, height) {
                std::borrow::Cow::Borrowed(gray)
            } else {
                std::borrow::Cow::Owned(imageops::resize(
                    gray,
                    scaled_w,
                    scaled_h,
                    imageops::FilterType::Triangle,
                ))
            };
            let integral = Integral::new(&level);
            let offsets = self.level(integral.stride);
            // The reference implementation slides two pixels at a time until
            // the window has doubled, then one.
            let step = if scale > 2.0 { 1 } else { 2 };
            let mut y = 0;
            while y + wh <= scaled_h {
                let mut x = 0;
                while x + ww <= scaled_w {
                    let base = y as usize * integral.stride + x as usize;
                    if self.stages_passed(&integral, &offsets, base) == self.stages.len() {
                        windows.push(Rect {
                            x: (x as f32 * scale).round() as u32,
                            y: (y as f32 * scale).round() as u32,
                            w: window_w,
                            h: window_h,
                        });
                    }
                    x += step;
                }
                y += step;
            }
            scale *= factor;
        }
        windows
    }

    /// Detect objects in a grey image: the raw windows, clustered, with
    /// clusters supported by more than `min_neighbors` windows kept.
    pub fn detect(&self, gray: &GrayImage, options: &CascadeOptions) -> Vec<CascadeMatch> {
        group_windows(&self.raw_windows(gray, options), options.min_neighbors)
    }

    /// Detect objects in `region` of a colour frame, in frame coordinates.
    pub fn detect_in(
        &self,
        image: &RgbaImage,
        region: Rect,
        options: &CascadeOptions,
    ) -> Vec<CascadeMatch> {
        let gray = channel_image(image, region, Channel::Luma);
        let mut matches = self.detect(&gray, options);
        for m in &mut matches {
            m.bounds.x += region.x;
            m.bounds.y += region.y;
        }
        matches
    }
}

/// Detection parameters, with the reference implementation's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CascadeOptions {
    /// How much the window grows between scales; 1.1 is thorough, 1.3 fast.
    pub scale_factor: f32,
    /// A cluster of windows must be larger than this to count.
    pub min_neighbors: u32,
    /// Smallest object side to look for, in pixels (0: the cascade's window).
    pub min_size: u32,
    /// Largest object side to look for, in pixels.
    pub max_size: Option<u32>,
}

impl Default for CascadeOptions {
    fn default() -> Self {
        Self {
            scale_factor: 1.1,
            min_neighbors: 3,
            min_size: 0,
            max_size: None,
        }
    }
}

/// An object found by a cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeMatch {
    pub bounds: Rect,
    /// How many overlapping windows agreed; the classic confidence proxy.
    pub neighbors: u32,
}

/// Integral image with sums of squares, over a grey image, with a zero
/// first row and column so any rectangle is four lookups.
struct Integral {
    stride: usize,
    sums: Vec<u32>,
    squares: Vec<u64>,
}

impl Integral {
    fn new(gray: &GrayImage) -> Self {
        let (width, height) = gray.dimensions();
        let stride = width as usize + 1;
        let mut sums = vec![0u32; stride * (height as usize + 1)];
        let mut squares = vec![0u64; stride * (height as usize + 1)];
        let raw = gray.as_raw();
        for y in 0..height as usize {
            let (mut row_sum, mut row_squares) = (0u32, 0u64);
            let above = y * stride;
            let here = (y + 1) * stride;
            for x in 0..width as usize {
                let v = raw[y * width as usize + x];
                row_sum += u32::from(v);
                row_squares += u64::from(v) * u64::from(v);
                sums[here + x + 1] = sums[above + x + 1] + row_sum;
                squares[here + x + 1] = squares[above + x + 1] + row_squares;
            }
        }
        Self {
            stride,
            sums,
            squares,
        }
    }

    #[inline]
    fn sum_at(&self, base: usize, corners: &[usize; 4]) -> u32 {
        let s = &self.sums;
        s[base + corners[0]] + s[base + corners[3]] - s[base + corners[1]] - s[base + corners[2]]
    }

    #[inline]
    fn squares_at(&self, base: usize, corners: &[usize; 4]) -> u64 {
        let s = &self.squares;
        s[base + corners[0]] + s[base + corners[3]] - s[base + corners[1]] - s[base + corners[2]]
    }
}

/// A feature rectangle resolved to integral-image offsets for one stride.
#[derive(Debug, Clone, Copy)]
struct OffsetRect {
    /// Top-left, top-right, bottom-left, bottom-right.
    offsets: [usize; 4],
    weight: f32,
}

/// The cascade's features laid out for one pyramid level.
struct Level {
    rects: Vec<OffsetRect>,
    /// `rects[first_rect[f]..first_rect[f + 1]]` belong to feature `f`.
    first_rect: Vec<usize>,
    /// The variance-normalisation window.
    norm: [usize; 4],
    norm_area: f64,
}

/// Fraction of a window's size within which two windows count as the same
/// object (the reference implementation's `eps`).
const SIMILARITY: f32 = 0.2;

fn similar(a: Rect, b: Rect) -> bool {
    let delta = SIMILARITY * (a.w.min(b.w) + a.h.min(b.h)) as f32 * 0.5;
    let close = |p: u32, q: u32| (p as f32 - q as f32).abs() <= delta;
    close(a.x, b.x) && close(a.y, b.y) && close(a.x + a.w, b.x + b.w) && close(a.y + a.h, b.y + b.h)
}

/// Merge overlapping windows into one match per object, the way the
/// reference implementation's `groupRectangles` does: windows are
/// clustered by similarity (transitively), each cluster with more than
/// `min_neighbors` members becomes its average rectangle, and a cluster
/// lying inside a better-supported one is dropped as a fragment.
pub fn group_windows(windows: &[Rect], min_neighbors: u32) -> Vec<CascadeMatch> {
    if windows.is_empty() {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..windows.len()).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        let mut at = i;
        while parent[at] != root {
            let next = parent[at];
            parent[at] = root;
            at = next;
        }
        root
    }
    for i in 0..windows.len() {
        for j in (i + 1)..windows.len() {
            if similar(windows[i], windows[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut clusters: Vec<(f32, f32, f32, f32, u32)> = Vec::new();
    let mut cluster_of = vec![usize::MAX; windows.len()];
    for (i, &r) in windows.iter().enumerate() {
        let root = find(&mut parent, i);
        if cluster_of[root] == usize::MAX {
            cluster_of[root] = clusters.len();
            clusters.push((0.0, 0.0, 0.0, 0.0, 0));
        }
        let c = &mut clusters[cluster_of[root]];
        c.0 += r.x as f32;
        c.1 += r.y as f32;
        c.2 += r.w as f32;
        c.3 += r.h as f32;
        c.4 += 1;
    }
    let averaged: Vec<(Rect, u32)> = clusters
        .iter()
        .map(|&(x, y, w, h, n)| {
            let s = 1.0 / n as f32;
            (
                Rect {
                    x: (x * s).round() as u32,
                    y: (y * s).round() as u32,
                    w: (w * s).round() as u32,
                    h: (h * s).round() as u32,
                },
                n,
            )
        })
        .collect();

    let mut kept = Vec::new();
    for (i, &(r1, n1)) in averaged.iter().enumerate() {
        if n1 <= min_neighbors {
            continue;
        }
        let inside_a_better_one = averaged.iter().enumerate().any(|(j, &(r2, n2))| {
            if i == j || n2 <= min_neighbors {
                return false;
            }
            let dx = r2.w as f32 * SIMILARITY;
            let dy = r2.h as f32 * SIMILARITY;
            r1.x as f32 >= r2.x as f32 - dx
                && r1.y as f32 >= r2.y as f32 - dy
                && (r1.x + r1.w) as f32 <= (r2.x + r2.w) as f32 + dx
                && (r1.y + r1.h) as f32 <= (r2.y + r2.h) as f32 + dy
                && (n2 > n1.max(3) || n1 < 3)
        });
        if !inside_a_better_one {
            kept.push(CascadeMatch {
                bounds: r1,
                neighbors: n1,
            });
        }
    }
    kept.sort_by_key(|m| std::cmp::Reverse(m.neighbors));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn astronaut() -> RgbaImage {
        image::load_from_memory(include_bytes!("../tests/fixtures/astronaut_320.jpg"))
            .expect("fixture decodes")
            .to_rgba8()
    }

    #[test]
    fn the_bundled_cascade_loads() {
        let cascade = Cascade::frontal_face();
        assert_eq!(cascade.window(), (20, 20));
        assert_eq!(cascade.stage_count(), 20);
        assert_eq!(cascade.features.len(), 2094);
    }

    #[test]
    fn every_bundled_cascade_loads() {
        assert_eq!(Cascade::eye().window(), (20, 20));
        assert_eq!(Cascade::eye().stage_count(), 24);
        assert_eq!(Cascade::profile_face().window(), (20, 20));
        assert_eq!(Cascade::profile_face().stage_count(), 26);
    }

    /// OpenCV 4.13 on the astronaut's face crop finds two eyes,
    /// (115, 51, 23x23) and (142, 53, 22x22) in image coordinates, with
    /// 28 and 16 supporting windows.
    #[test]
    fn finds_both_eyes_inside_the_astronauts_face() {
        let image = astronaut();
        let face = Rect {
            x: 109,
            y: 40,
            w: 62,
            h: 62,
        };
        let mut eyes = Cascade::eye().detect_in(&image, face, &CascadeOptions::default());
        eyes.sort_by_key(|e| e.bounds.x);
        assert_eq!(eyes.len(), 2, "{eyes:?}");
        let left = intersection_over_union(
            eyes[0].bounds,
            Rect {
                x: 115,
                y: 51,
                w: 23,
                h: 23,
            },
        );
        let right = intersection_over_union(
            eyes[1].bounds,
            Rect {
                x: 142,
                y: 53,
                w: 22,
                h: 22,
            },
        );
        assert!(
            left > 0.6 && right > 0.6,
            "{eyes:?}: IoU {left:.2} / {right:.2}"
        );
    }

    #[test]
    fn corrupt_files_are_refused() {
        assert_eq!(
            Cascade::from_bytes(b"nope").unwrap_err(),
            CascadeError::BadMagic
        );
        let mut bytes = FRONTAL_FACE.to_vec();
        bytes[4] = 9;
        assert_eq!(
            Cascade::from_bytes(&bytes).unwrap_err(),
            CascadeError::UnsupportedVersion(9)
        );
        assert_eq!(
            Cascade::from_bytes(&FRONTAL_FACE[..1000]).unwrap_err(),
            CascadeError::Truncated
        );
    }

    /// OpenCV 4.13 on the same JPEG bytes: one face at (109, 40, 62, 62)
    /// with 37 neighbouring windows, from 40 raw windows.
    #[test]
    fn finds_the_astronauts_face_where_opencv_does() {
        let image = astronaut();
        let region = Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        };
        let cascade = Cascade::frontal_face();
        let matches = cascade.detect_in(&image, region, &CascadeOptions::default());
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one face, got {matches:?}"
        );
        let face = matches[0].bounds;
        let reference = Rect {
            x: 109,
            y: 40,
            w: 62,
            h: 62,
        };
        let iou = intersection_over_union(face, reference);
        assert!(
            iou > 0.8,
            "face {face:?} vs OpenCV {reference:?}: IoU {iou:.2}"
        );
        assert!(
            matches[0].neighbors >= 20,
            "support {} is far below OpenCV's 37",
            matches[0].neighbors
        );
    }

    #[test]
    fn a_face_free_image_yields_nothing() {
        let image = astronaut();
        // The astronaut's shoulder patch and the shuttle model: texture, no face.
        let region = Rect {
            x: 0,
            y: 200,
            w: 320,
            h: 120,
        };
        let matches = Cascade::frontal_face().detect_in(&image, region, &CascadeOptions::default());
        assert!(matches.is_empty(), "false positives: {matches:?}");
    }

    #[test]
    fn size_limits_are_honoured() {
        let image = astronaut();
        let region = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 320,
        };
        let small_only = CascadeOptions {
            max_size: Some(40),
            ..CascadeOptions::default()
        };
        assert!(
            Cascade::frontal_face()
                .detect_in(&image, region, &small_only)
                .is_empty()
        );
        let large_only = CascadeOptions {
            min_size: 50,
            ..CascadeOptions::default()
        };
        assert_eq!(
            Cascade::frontal_face()
                .detect_in(&image, region, &large_only)
                .len(),
            1
        );
    }

    #[test]
    fn grouping_merges_neighbours_and_drops_lonely_windows() {
        let cluster: Vec<Rect> = (0..5)
            .map(|i| Rect {
                x: 100 + i,
                y: 100 + i,
                w: 60,
                h: 60,
            })
            .collect();
        let lonely = Rect {
            x: 10,
            y: 10,
            w: 60,
            h: 60,
        };
        let mut windows = cluster.clone();
        windows.push(lonely);
        let groups = group_windows(&windows, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].neighbors, 5);
        assert_eq!(
            groups[0].bounds,
            Rect {
                x: 102,
                y: 102,
                w: 60,
                h: 60
            }
        );
        assert!(group_windows(&windows, 5).is_empty());
    }

    #[test]
    fn a_small_cluster_inside_a_bigger_one_is_a_fragment() {
        let mut windows: Vec<Rect> = (0..8)
            .map(|i| Rect {
                x: 100 + i,
                y: 100,
                w: 80,
                h: 80,
            })
            .collect();
        windows.extend((0..4).map(|i| Rect {
            x: 120 + i,
            y: 120,
            w: 30,
            h: 30,
        }));
        let groups = group_windows(&windows, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].neighbors, 8);
    }

    fn intersection_over_union(a: Rect, b: Rect) -> f32 {
        let x0 = a.x.max(b.x);
        let y0 = a.y.max(b.y);
        let x1 = (a.x + a.w).min(b.x + b.w);
        let y1 = (a.y + a.h).min(b.y + b.h);
        let inter = x1.saturating_sub(x0) as f32 * y1.saturating_sub(y0) as f32;
        inter / (a.area() as f32 + b.area() as f32 - inter)
    }
}
