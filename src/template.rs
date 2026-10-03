//! Finding a known picture — an icon, a marker, a button — in a frame.
//!
//! Matching is by zero-mean normalised cross-correlation on one channel:
//! the score of a position is the Pearson correlation between the template
//! and the window of the frame under it. It is 1 for an exact copy whatever
//! the window's brightness and contrast, near 0 for unrelated content, and
//! negative for an inverted copy.
//!
//! Scoring every position at full resolution costs the template's area per
//! position. Large searches therefore run on a pyramid of copies halved in
//! size: every position is scored at the coarsest level, the most promising
//! are refined level by level — fewer at each, as the rankings get more
//! reliable — and only their neighbourhoods are scored at full resolution.
//! Window sums slide along the rows, and the correlations are integer
//! multiply-adds the compiler vectorises.
//!
//! ```text
//!   frame ─▶ channel ─▶ ½ ─▶ ¼         template ─▶ ½ ─▶ ¼
//!                             │                         │
//!                             └──── score everywhere ───┘
//!                                          │ 256 best positions
//!                     refine at ½ (keep 32), then at full size:
//!                     every position within 2 px, then uphill
//!                                          ▼
//!                           matches (+ sub-pixel centre)
//! ```
//!
//! A coarse level is only used when the template still looks like itself
//! there wherever the halving grid falls, and still has at least 10x10
//! pixels of structure to rank positions by. Templates of fine detail —
//! small text, one-pixel lines — are therefore scored at every position at
//! full resolution, which costs the search area times the template area:
//! give those a search region near where they are expected.
//! [`Template::coarse_levels`] tells which case a template is in.
//!
//! A thing that is seen in several poses, or facing either way, is a
//! [`TemplateSet`]: its pictures (and their mirror images, when asked) are
//! searched for together in one pass over the frame's pyramid
//! ([`find_set`]), the matches of all of them ranked as one, and an
//! optional colour check drops a match whose colours are nothing like the
//! picture's — correlation on one channel cannot tell a red thing from a
//! green one of the same shape.

use image::{GrayImage, RgbaImage};

use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::Rect;
use crate::threshold::{Channel, channel_image};

/// Smallest side, in pixels, of the part of a template matched at a coarse
/// pyramid level. Smaller coarse templates of interface content reduce to
/// a blurred edge or two, which match every panel edge in the frame as
/// well as they match the real copy.
const MIN_COARSE_SIDE: usize = 10;

/// Deepest pyramid level: 1/8 scale.
const MAX_LEVELS: usize = 3;

/// Multiply-accumulates below which a search skips the pyramid and scores
/// every position exactly: about a millisecond of work.
const EXHAUSTIVE_BUDGET: u64 = 2 << 20;

/// Candidates kept at pyramid level `L >= 1` when looking for the best
/// match: `KEEP_AT_LEVEL_ONE << 2 * (L - 1)`, up to [`MAX_CANDIDATES`].
/// Each level is a quarter of the work per candidate of the one below, so
/// every level of refinement costs about the same.
const KEEP_AT_LEVEL_ONE: usize = 32;

/// How closely a halved template must agree with itself halved on a grid
/// shifted by one pixel for that coarse level to be searched.
const PHASE_AGREEMENT: f64 = 0.75;

/// Most candidates carried from one level to the next.
const MAX_CANDIDATES: usize = 256;

/// How far below the requested score a coarse position may score and still
/// be refined: fine detail lost to downscaling lowers coarse scores.
const COARSE_SLACK: f32 = 0.25;

/// Candidates to keep at `level` of the pyramid.
fn keep_at(level: usize) -> usize {
    if level == 0 {
        return usize::MAX;
    }
    KEEP_AT_LEVEL_ONE
        .saturating_mul(1 << (2 * (level - 1)).min(16))
        .min(MAX_CANDIDATES)
}

/// Search radius, in pixels, when refining a position at a finer level.
const REFINE_RADIUS: usize = 2;

/// Most uphill steps taken past the refinement window.
const MAX_CLIMB: usize = 8;

/// Candidates closer than this, in pixels of their level, count as one.
const PEAK_SPACING: usize = 2;

/// A picture to look for, prepared for matching.
#[derive(Debug, Clone)]
pub struct Template {
    channel: Channel,
    /// Full size first, then successively halved copies.
    levels: Vec<TemplateLevel>,
    /// The mean colour of the picture's opaque pixels, for a colour check
    /// after matching.
    mean_rgb: [f32; 3],
}

impl Template {
    /// The template is `region` of `image`, compared through `channel`
    /// (clipped to the image). `None` when the region is empty or uniform:
    /// a template with no variation matches every flat patch equally.
    pub fn new(image: &RgbaImage, region: Rect, channel: Channel) -> Option<Self> {
        let plane = Plane::from_gray(channel_image(image, region, channel));
        if plane.width == 0 || plane.height == 0 {
            return None;
        }
        let mean_rgb = mean_rgb(image, region);
        let mut levels = vec![TemplateLevel::new(plane, 0)?];
        while levels.len() <= MAX_LEVELS {
            let last = &levels[levels.len() - 1].plane;
            if last.width / 2 < MIN_COARSE_SIDE + 2 || last.height / 2 < MIN_COARSE_SIDE + 2 {
                break;
            }
            // A template whose detail is too fine to survive halving (a
            // checkerboard, one-pixel text) is only searched at the levels
            // where it still looks like itself.
            let half = last.half();
            if !survives_misalignment(last, &half) {
                break;
            }
            match TemplateLevel::new(half, 1) {
                Some(level) => levels.push(level),
                None => break,
            }
        }
        Some(Self {
            channel,
            levels,
            mean_rgb,
        })
    }

    /// The same picture seen in a mirror, left to right.
    pub fn mirrored(image: &RgbaImage, region: Rect, channel: Channel) -> Option<Self> {
        let region = clip(region, image.width(), image.height());
        if region.w == 0 || region.h == 0 {
            return None;
        }
        let crop = image::imageops::crop_imm(image, region.x, region.y, region.w, region.h);
        let flipped = image::imageops::flip_horizontal(&*crop);
        Self::from_image(&flipped, channel)
    }

    /// The whole of `image` as a template.
    pub fn from_image(image: &RgbaImage, channel: Channel) -> Option<Self> {
        let region = Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        };
        Self::new(image, region, channel)
    }

    pub fn width(&self) -> u32 {
        self.levels[0].plane.width as u32
    }

    pub fn height(&self) -> u32 {
        self.levels[0].plane.height as u32
    }

    pub fn channel(&self) -> Channel {
        self.channel
    }

    /// How many halved copies a large search can use. 0 means every
    /// position is scored at full resolution, so a search costs the search
    /// area times the template's area.
    pub fn coarse_levels(&self) -> usize {
        self.levels.len() - 1
    }

    /// The mean colour (red, green, blue, 0–255) of the picture's opaque
    /// pixels.
    pub fn mean_rgb(&self) -> [f32; 3] {
        self.mean_rgb
    }
}

/// `region` clipped to a `width`×`height` image.
fn clip(region: Rect, width: u32, height: u32) -> Rect {
    let x = region.x.min(width);
    let y = region.y.min(height);
    Rect {
        x,
        y,
        w: region.w.min(width - x),
        h: region.h.min(height - y),
    }
}

/// The mean colour of the opaque pixels of `region` (clipped), 0 when there
/// are none.
fn mean_rgb(image: &RgbaImage, region: Rect) -> [f32; 3] {
    let region = clip(region, image.width(), image.height());
    let mut sum = [0f64; 3];
    let mut count = 0f64;
    for y in region.y..region.y + region.h {
        for x in region.x..region.x + region.w {
            let p = image.get_pixel(x, y).0;
            if p[3] < 128 {
                continue;
            }
            for (s, &v) in sum.iter_mut().zip(&p[..3]) {
                *s += f64::from(v);
            }
            count += 1.0;
        }
    }
    if count == 0.0 {
        return [0.0; 3];
    }
    sum.map(|v| (v / count) as f32)
}

/// Several pictures of one thing — its poses, and when asked their mirror
/// images too — looked for together.
#[derive(Debug, Clone)]
pub struct TemplateSet {
    channel: Channel,
    mirrored: bool,
    pictures: Vec<Picture>,
}

#[derive(Debug, Clone)]
struct Picture {
    template: Template,
    mirror: Option<Template>,
    width: u32,
    height: u32,
}

impl TemplateSet {
    /// An empty set; with `mirrored`, every picture added is also looked
    /// for facing the other way.
    pub fn new(channel: Channel, mirrored: bool) -> Self {
        Self {
            channel,
            mirrored,
            pictures: Vec::new(),
        }
    }

    /// Adds a picture. `false` (and nothing added) when it is empty or
    /// uniform, like [`Template::new`].
    pub fn add(&mut self, image: &RgbaImage) -> bool {
        let Some(template) = Template::from_image(image, self.channel) else {
            return false;
        };
        let region = Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        };
        let mirror = self
            .mirrored
            .then(|| Template::mirrored(image, region, self.channel))
            .flatten();
        self.pictures.push(Picture {
            template,
            mirror,
            width: image.width(),
            height: image.height(),
        });
        true
    }

    pub fn len(&self) -> usize {
        self.pictures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pictures.is_empty()
    }

    pub fn channel(&self) -> Channel {
        self.channel
    }

    pub fn mirrored(&self) -> bool {
        self.mirrored
    }

    /// The pictures, in the order added.
    pub fn templates(&self) -> impl Iterator<Item = &Template> + '_ {
        self.pictures.iter().map(|p| &p.template)
    }

    /// How deep a pyramid any picture in the set can use.
    fn depth(&self) -> usize {
        self.pictures
            .iter()
            .flat_map(|p| std::iter::once(&p.template).chain(p.mirror.as_ref()))
            .map(|t| t.coarse_levels())
            .max()
            .unwrap_or(0)
    }
}

/// The part of `picture` that stands out from its edges — the thing, not
/// the ground and sky around it — when that is a fair part of the picture
/// and big enough to match; `None` means the whole picture is the thing.
///
/// The background is the most common colours along the border (a few, as a
/// creature standing on the ground has sky behind its head); a pixel near
/// none of them is foreground, and the result is the box around the
/// foreground with a pixel or two of margin.
pub fn foreground(picture: &RgbaImage) -> Option<Rect> {
    let (w, h) = picture.dimensions();
    if w < 8 || h < 8 {
        return None;
    }
    // The border's colours, in 5-bit buckets, each with its mean.
    let mut buckets: std::collections::HashMap<u16, (u32, [u32; 3])> = Default::default();
    let mut border = 0u32;
    let mut add = |x: u32, y: u32| {
        let p = picture.get_pixel(x, y).0;
        let key = ((p[0] as u16 >> 3) << 10) | ((p[1] as u16 >> 3) << 5) | (p[2] as u16 >> 3);
        let e = buckets.entry(key).or_insert((0, [0; 3]));
        e.0 += 1;
        for (sum, &v) in e.1.iter_mut().zip(&p[..3]) {
            *sum += v as u32;
        }
        border += 1;
    };
    for x in 0..w {
        add(x, 0);
        add(x, h - 1);
    }
    for y in 1..h - 1 {
        add(0, y);
        add(w - 1, y);
    }
    let mut common: Vec<(u32, [u32; 3])> = buckets.into_values().collect();
    common.sort_by_key(|c| std::cmp::Reverse(c.0));
    let background: Vec<[i32; 3]> = common
        .iter()
        .take(4)
        .filter(|(n, _)| n * 10 >= border)
        .map(|(n, sum)| sum.map(|v| (v / n) as i32))
        .collect();
    if background.is_empty() {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    let mut count = 0u32;
    for y in 0..h {
        for x in 0..w {
            let p = picture.get_pixel(x, y).0;
            let near = background.iter().any(|bg| {
                (0..3)
                    .map(|c| (p[c] as i32 - bg[c]).abs())
                    .max()
                    .unwrap_or(0)
                    <= 40
            });
            if !near {
                count += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if count < 16 || x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (bw, bh) = (x1 - x0, y1 - y0);
    // Only when it is a fair part of the picture and big enough to match.
    if bw * bh * 10 < w * h || bw < 8 || bh < 8 {
        return None;
    }
    let (x0, y0) = (x0.saturating_sub(2), y0.saturating_sub(2));
    let (x1, y1) = ((x1 + 2).min(w), (y1 + 2).min(h));
    Some(Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

/// How a [`TemplateSet`] is searched for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetSearch {
    /// Normalised correlation a match needs.
    pub min_score: f32,
    /// At most this many matches, strongest first.
    pub limit: usize,
    /// When set, a match whose window's mean colour differs from the
    /// picture's by more than this on any channel (0–255) is dropped.
    pub max_colour_shift: Option<f32>,
}

impl Default for SetSearch {
    fn default() -> Self {
        Self {
            min_score: 0.7,
            limit: 16,
            max_colour_shift: None,
        }
    }
}

/// Where a picture of a set matched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetMatch {
    /// The matched window, in image coordinates.
    pub bounds: Rect,
    /// Normalised correlation with the picture, in `[-1, 1]`.
    pub score: f32,
    /// Centre of the match to a fraction of a pixel.
    pub centre: (f32, f32),
    /// Which picture of the set matched (its index in the order added).
    pub picture: usize,
    /// Whether it was the picture's mirror image that matched.
    pub mirrored: bool,
}

/// Every place in `search` (clipped to the image) where a picture of `set`
/// scores at least `options.min_score`, strongest first; where matches of
/// different pictures overlap by more than a quarter, only the strongest
/// is kept. The frame's pyramid is built once for all the pictures.
pub fn find_set(
    image: &RgbaImage,
    search: Rect,
    set: &TemplateSet,
    options: SetSearch,
) -> Vec<SetMatch> {
    if set.is_empty() || options.limit == 0 {
        return Vec::new();
    }
    let pyramid = Pyramid::new(image, search, set.channel, set.depth());
    let mut found: Vec<(usize, bool, usize, usize, f32)> = Vec::new();
    for (index, picture) in set.pictures.iter().enumerate() {
        let variants = std::iter::once((&picture.template, false))
            .chain(picture.mirror.as_ref().map(|m| (m, true)));
        for (template, mirrored) in variants {
            for (x, y, score) in candidates(&pyramid, template, Some(options.min_score), true) {
                found.push((index, mirrored, x, y, score));
            }
        }
    }
    found.sort_by(|a, b| b.4.total_cmp(&a.4));
    let search = pyramid.search;
    let mut kept: Vec<SetMatch> = Vec::new();
    for (index, mirrored, x, y, score) in found {
        let picture = &set.pictures[index];
        let (w, h) = (picture.width as usize, picture.height as usize);
        let bounds = Rect {
            x: search.x + x as u32,
            y: search.y + y as u32,
            w: picture.width,
            h: picture.height,
        };
        if let Some(max_shift) = options.max_colour_shift {
            let template = if mirrored {
                picture.mirror.as_ref().unwrap_or(&picture.template)
            } else {
                &picture.template
            };
            let window = mean_rgb(image, bounds);
            let shift = (0..3)
                .map(|c| (window[c] - template.mean_rgb[c]).abs())
                .fold(0f32, f32::max);
            if shift > max_shift {
                continue;
            }
        }
        let overlaps = kept.iter().any(|k| {
            let (kw, kh) = (k.bounds.w as usize, k.bounds.h as usize);
            2 * (k.bounds.x as usize).abs_diff(bounds.x as usize) < w.min(kw)
                && 2 * (k.bounds.y as usize).abs_diff(bounds.y as usize) < h.min(kh)
        });
        if overlaps {
            continue;
        }
        let template = if mirrored {
            picture.mirror.as_ref().unwrap_or(&picture.template)
        } else {
            &picture.template
        };
        let described = describe(&pyramid.planes[0], &template.levels[0], search, x, y, score);
        kept.push(SetMatch {
            bounds: described.bounds,
            score,
            centre: described.centre,
            picture: index,
            mirrored,
        });
        if kept.len() >= options.limit {
            break;
        }
    }
    kept
}

/// Where a template matched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemplateMatch {
    /// The matched window, in image coordinates.
    pub bounds: Rect,
    /// Normalised correlation with the template, in `[-1, 1]`.
    pub score: f32,
    /// Centre of the match to a fraction of a pixel, from the shape of the
    /// score around the best position; for tracking small markers.
    pub centre: (f32, f32),
}

/// The best-scoring position of `template` within `search` (clipped to
/// the image), or `None` when the template does not fit in the search.
pub fn find_best(image: &RgbaImage, search: Rect, template: &Template) -> Option<TemplateMatch> {
    run(image, search, template, None).into_iter().next()
}

/// Every match scoring at least `min_score`, strongest first. Of matches
/// overlapping by more than a quarter of the template, only the strongest
/// is kept.
///
/// With a pyramid, coarse positions scoring more than 0.25 below
/// `min_score` are not followed up, so a copy that only just clears
/// `min_score` at full resolution can, rarely, be missed.
pub fn find_all(
    image: &RgbaImage,
    search: Rect,
    template: &Template,
    min_score: f32,
) -> Vec<TemplateMatch> {
    run(image, search, template, Some(min_score))
}

/// The best match as a detection: found when it scores at least
/// `min_score`, otherwise missing with the best score in the reason.
///
/// Confidence rises from 0.5 at `min_score` to 1 for an exact match.
pub fn locate(
    image: &RgbaImage,
    search: Rect,
    template: &Template,
    min_score: f32,
) -> Detection<TemplateMatch> {
    match find_best(image, search, template) {
        None => Detection::missing("template", "the template does not fit in the search region"),
        Some(found) if found.score < min_score => Detection::missing(
            "template",
            format!(
                "best match scored {:.2} at ({}, {}), below {min_score:.2}",
                found.score, found.bounds.x, found.bounds.y
            ),
        ),
        Some(found) => {
            let headroom = (1.0 - min_score).max(f32::EPSILON);
            let confidence = 0.5 + 0.5 * ((found.score - min_score) / headroom).clamp(0.0, 1.0);
            Detection::found(
                found,
                Confidence::new(confidence),
                "template",
                Reliability::Heuristic,
            )
        }
    }
}

/// One channel of an image: a byte per pixel, row-major.
#[derive(Debug, Clone)]
struct Plane {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Plane {
    fn from_gray(gray: GrayImage) -> Self {
        let (width, height) = (gray.width() as usize, gray.height() as usize);
        Self {
            width,
            height,
            pixels: gray.into_raw(),
        }
    }

    fn row(&self, y: usize, x: usize, len: usize) -> &[u8] {
        let start = y * self.width + x;
        &self.pixels[start..start + len]
    }

    /// Half the size. Each output pixel is centred where a 2x2 block of
    /// the input is, and averages the 4x4 pixels around it with weights
    /// 1-3-3-1 in each direction: a plain 2x2 mean lets fine detail alias
    /// into patterns that change with the grid's alignment. Borders repeat
    /// the edge pixel; an odd last row or column is dropped.
    fn half(&self) -> Plane {
        let (width, height) = (self.width / 2, self.height / 2);
        let last_x = self.width as isize - 1;
        let last_y = self.height as isize - 1;

        // Horizontal pass: weights sum to 8. The taps of output `i` are the
        // input pixels 2i-1 .. 2i+2; splitting the (edge-extended) row into
        // its even and odd pixels turns that into four contiguous streams.
        let mut horizontal = vec![0u16; width * self.height];
        // even[k] is input pixel 2k-1 and odd[k] input pixel 2k, so output
        // i reads even[i], odd[i], even[i+1], odd[i+1].
        let (mut even, mut odd) = (vec![0u16; width + 1], vec![0u16; width + 1]);
        for (y, out) in horizontal
            .chunks_exact_mut(width.max(1))
            .enumerate()
            .take(self.height)
        {
            let row = self.row(y, 0, self.width);
            for (k, pair) in row.as_chunks::<2>().0.iter().enumerate() {
                odd[k] = u16::from(pair[0]);
                even[k + 1] = u16::from(pair[1]);
            }
            even[0] = u16::from(row[0]);
            odd[width] = u16::from(row[(2 * width).min(last_x as usize)]);
            for ((value, (e0, o0)), (e1, o1)) in out
                .iter_mut()
                .zip(even.iter().zip(&odd))
                .zip(even[1..].iter().zip(&odd[1..]))
            {
                *value = e0 + 3 * o0 + 3 * e1 + o1;
            }
        }

        // Vertical pass: weights sum to 8, so 64 in all, and the largest
        // total, 64 * 255, still fits 16 bits.
        let mut pixels = Vec::with_capacity(width * height);
        let row_of = |y: isize| {
            let y = y.clamp(0, last_y) as usize;
            &horizontal[y * width..(y + 1) * width]
        };
        for j in 0..height as isize {
            let (a, b, c, d) = (
                row_of(2 * j - 1),
                row_of(2 * j),
                row_of(2 * j + 1),
                row_of(2 * j + 2),
            );
            pixels.extend(
                a.iter()
                    .zip(b)
                    .zip(c.iter().zip(d))
                    .map(|((&a, &b), (&c, &d))| ((a + 3 * b + 3 * c + d + 32) >> 6) as u8),
            );
        }
        Plane {
            width,
            height,
            pixels,
        }
    }

    /// The `width x height` part of the plane starting at `(x, y)`.
    fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Plane {
        let mut pixels = Vec::with_capacity(width * height);
        for row in y..y + height {
            pixels.extend_from_slice(self.row(row, x, width));
        }
        Plane {
            width,
            height,
            pixels,
        }
    }
}

/// Pearson correlation of the `width x height` pixels of two planes
/// starting at `(x0, y0)` in each; 0 when either is flat there.
fn plane_correlation(
    a: &Plane,
    b: &Plane,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> f64 {
    let n = (width * height) as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for y in y0..y0 + height {
        for (&va, &vb) in a.row(y, x0, width).iter().zip(b.row(y, x0, width)) {
            let (va, vb) = (f64::from(va), f64::from(vb));
            sa += va;
            sb += vb;
            saa += va * va;
            sbb += vb * vb;
            sab += va * vb;
        }
    }
    let spread = (saa - sa * sa / n) * (sbb - sb * sb / n);
    if spread <= 0.0 {
        return 0.0;
    }
    (sab - sa * sb / n) / spread.sqrt()
}

/// Whether halving `fine` gives much the same picture wherever the
/// halving grid falls. A copy of the template in a frame can sit at any
/// offset from the frame's own grid, so a coarse level that changes with
/// the offset would rank the copy poorly and the search would miss it.
/// Only the interior is compared: the outer ring is never matched.
fn survives_misalignment(fine: &Plane, coarse: &Plane) -> bool {
    [(1, 0), (0, 1), (1, 1)].into_iter().all(|(dx, dy)| {
        let shifted = fine.crop(dx, dy, fine.width - dx, fine.height - dy).half();
        let width = shifted.width.min(coarse.width).saturating_sub(2);
        let height = shifted.height.min(coarse.height).saturating_sub(2);
        width > 0
            && height > 0
            && plane_correlation(coarse, &shifted, 1, 1, width, height) >= PHASE_AGREEMENT
    })
}

/// A template at one scale, with the sums its correlation needs.
///
/// Coarse levels are matched without their outermost ring of pixels: the
/// halving filter reaches one pixel past the template's edge, so in a
/// frame that ring mixes in whatever surrounds the copy, and matching on it
/// would mark down every real copy.
#[derive(Debug, Clone)]
struct TemplateLevel {
    /// The whole template at this scale; the next level is made from it.
    plane: Plane,
    /// What is matched: `plane` without `margin` pixels on every side.
    inner: Plane,
    margin: usize,
    /// Sum of the matched pixels.
    sum: i64,
    /// `n * Σt² - (Σt)²` over the `n` matched pixels: `n²` times their
    /// variance, kept exact in integers.
    spread: i64,
}

impl TemplateLevel {
    fn new(plane: Plane, margin: usize) -> Option<Self> {
        let width = plane.width.checked_sub(2 * margin).filter(|&w| w > 0)?;
        let height = plane.height.checked_sub(2 * margin).filter(|&h| h > 0)?;
        let inner = plane.crop(margin, margin, width, height);
        let n = inner.pixels.len() as i64;
        let sum: i64 = inner.pixels.iter().map(|&v| i64::from(v)).sum();
        let squares: i64 = inner
            .pixels
            .iter()
            .map(|&v| i64::from(v) * i64::from(v))
            .sum();
        let spread = n * squares - sum * sum;
        // Less than a quarter of a level of variance: flat.
        (4 * spread > n * n).then_some(Self {
            plane,
            inner,
            margin,
            sum,
            spread,
        })
    }

    fn area(&self) -> usize {
        self.inner.pixels.len()
    }
}

/// A position of the matched part at a coarser level, carried to the next
/// finer one: twice as far, adjusted for the two levels' margins.
fn to_finer(position: usize, coarse_margin: usize, fine_margin: usize) -> usize {
    (2 * position + fine_margin).saturating_sub(2 * coarse_margin)
}

/// The correlation of `template` with the window of `plane` at `(x, y)`,
/// given the window's pixel sum and sum of squares.
fn correlation(
    plane: &Plane,
    template: &TemplateLevel,
    x: usize,
    y: usize,
    sum: u64,
    squares: u64,
) -> f32 {
    let (w, h) = (template.inner.width, template.inner.height);
    let mut products = 0u64;
    for v in 0..h {
        products += u64::from(dot(plane.row(y + v, x, w), template.inner.row(v, 0, w)));
    }
    score_from(template, products, sum, squares)
}

/// The correlation of `template` with a window, from the window's sum of
/// products with the template, pixel sum and sum of squares.
///
/// Everything up to the final ratio is exact integer arithmetic: the
/// variance-like terms are differences of large, nearly equal numbers,
/// which floating point would round away on flat windows.
#[inline]
fn score_from(template: &TemplateLevel, products: u64, sum: u64, squares: u64) -> f32 {
    // Products reach 255² per pixel, so n * products stays within i64 for
    // any template up to ~11 million pixels, far beyond a screen.
    let n = template.area() as i64;
    let (products, sum, squares) = (products as i64, sum as i64, squares as i64);
    let spread = n * squares - sum * sum;
    if 4 * spread <= n * n {
        // A flat window correlates with nothing.
        return 0.0;
    }
    let covariance = n * products - sum * template.sum;
    let denominator = (spread as f32 * template.spread as f32).sqrt();
    (covariance as f32 / denominator).clamp(-1.0, 1.0)
}

/// Integer dot product of two rows. Integer sums can be reordered freely,
/// so this compiles to SIMD multiply-adds, where a float sum would not.
#[inline]
fn dot(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u32::from(x) * u32::from(y))
        .sum()
}

/// The correlation of `template` with the window at `(x, y)`, computing
/// the window's sums, sum of squares and products in one pass per row.
fn score_at(plane: &Plane, template: &TemplateLevel, x: usize, y: usize) -> f32 {
    let (w, h) = (template.inner.width, template.inner.height);
    let (mut sum, mut squares, mut products) = (0u64, 0u64, 0u64);
    for v in 0..h {
        let (row, weights) = (plane.row(y + v, x, w), template.inner.row(v, 0, w));
        let (mut s, mut q, mut p) = (0u32, 0u32, 0u32);
        for (&pixel, &weight) in row.iter().zip(weights) {
            let pixel = u32::from(pixel);
            s += pixel;
            q += pixel * pixel;
            p += pixel * u32::from(weight);
        }
        sum += u64::from(s);
        squares += u64::from(q);
        products += u64::from(p);
    }
    score_from(template, products, sum, squares)
}

/// Largest template area whose sums of products fit in 32 bits.
const MAX_AREA_32: usize = (u32::MAX / (255 * 255)) as usize;

/// Correlation at every position of `plane`, row-major over the
/// `(width - w + 1) x (height - h + 1)` positions. Window sums slide: each
/// column's sum over the template's height is updated one row at a time,
/// and each window's sum one column at a time.
fn score_everywhere(plane: &Plane, template: &TemplateLevel) -> Vec<f32> {
    let (w, h) = (template.inner.width, template.inner.height);
    let (columns, rows) = (plane.width - w + 1, plane.height - h + 1);
    let products = (template.area() <= MAX_AREA_32).then(|| products_everywhere(plane, template));
    let mut scores = Vec::with_capacity(columns * rows);
    let mut column_sums = vec![0u64; plane.width];
    let mut column_squares = vec![0u64; plane.width];
    for y in 0..h {
        for (x, &p) in plane.row(y, 0, plane.width).iter().enumerate() {
            column_sums[x] += u64::from(p);
            column_squares[x] += u64::from(p) * u64::from(p);
        }
    }
    for y in 0..rows {
        let mut sum: u64 = column_sums[..w].iter().sum();
        let mut squares: u64 = column_squares[..w].iter().sum();
        for x in 0..columns {
            if x > 0 {
                sum = sum + column_sums[x + w - 1] - column_sums[x - 1];
                squares = squares + column_squares[x + w - 1] - column_squares[x - 1];
            }
            scores.push(match &products {
                Some(products) => {
                    score_from(template, u64::from(products[y * columns + x]), sum, squares)
                }
                None => correlation(plane, template, x, y, sum, squares),
            });
        }
        if y + 1 < rows {
            let (leaving, entering) = (
                plane.row(y, 0, plane.width),
                plane.row(y + h, 0, plane.width),
            );
            for x in 0..plane.width {
                let (old, new) = (u64::from(leaving[x]), u64::from(entering[x]));
                column_sums[x] = column_sums[x] + new - old;
                column_squares[x] = column_squares[x] + new * new - old * old;
            }
        }
    }
    scores
}

/// The sum of products of the template with the window at every position.
///
/// Rather than a short dot product per position, each template pixel is
/// multiplied into a whole row of positions at once: long, independent
/// integer multiply-adds that vectorise well. Each product is at most
/// 255 * 255, which fits 16 bits exactly, so the multiply is a 16-bit one.
/// Callers must keep the template area at or below [`MAX_AREA_32`].
fn products_everywhere(plane: &Plane, template: &TemplateLevel) -> Vec<u32> {
    let (w, h) = (template.inner.width, template.inner.height);
    let (columns, rows) = (plane.width - w + 1, plane.height - h + 1);
    let mut out = vec![0u32; columns * rows];
    for (y, accumulator) in out.chunks_exact_mut(columns).enumerate() {
        for v in 0..h {
            let source = plane.row(y + v, 0, plane.width);
            for (u, &weight) in template.inner.row(v, 0, w).iter().enumerate() {
                if weight == 0 {
                    continue;
                }
                let weight = u16::from(weight);
                for (total, &pixel) in accumulator.iter_mut().zip(&source[u..u + columns]) {
                    *total += u32::from(weight * u16::from(pixel));
                }
            }
        }
    }
    out
}

/// Positions whose score is at least `floor` and no lower than any of their
/// eight neighbours, strongest first, at most `limit` of them.
fn local_maxima(
    scores: &[f32],
    columns: usize,
    rows: usize,
    floor: f32,
    limit: usize,
) -> Vec<(usize, usize, f32)> {
    // The largest of each score and its left and right neighbours, so that
    // a peak is a score no lower than the row maxima above, at and below.
    let mut across = vec![f32::NEG_INFINITY; scores.len()];
    for (row, out) in scores
        .chunks_exact(columns)
        .zip(across.chunks_exact_mut(columns))
    {
        out[0] = row[0].max(row.get(1).copied().unwrap_or(f32::NEG_INFINITY));
        if columns > 1 {
            out[columns - 1] = row[columns - 1].max(row[columns - 2]);
        }
        if columns > 2 {
            for (value, ((&a, &b), &c)) in out[1..columns - 1]
                .iter_mut()
                .zip(row.iter().zip(&row[1..]).zip(&row[2..]))
            {
                *value = a.max(b).max(c);
            }
        }
    }
    let mut peaks = Vec::new();
    for y in 0..rows {
        let row = &scores[y * columns..(y + 1) * columns];
        let neighbours = |dy: isize| {
            let ny = y as isize + dy;
            (0..rows as isize)
                .contains(&ny)
                .then(|| &across[ny as usize * columns..(ny as usize + 1) * columns])
        };
        let (above, here, below) = (
            neighbours(-1),
            neighbours(0).expect("own row"),
            neighbours(1),
        );
        for (x, &score) in row.iter().enumerate() {
            if score >= floor
                && score >= here[x]
                && above.is_none_or(|a| score >= a[x])
                && below.is_none_or(|b| score >= b[x])
            {
                peaks.push((x, y, score));
            }
        }
    }
    // Along a straight edge every position ties, and each counts as a
    // peak; keep one per neighbourhood so ties cannot use up the budget.
    peaks.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut kept: Vec<(usize, usize, f32)> = Vec::new();
    for peak in peaks {
        if kept.len() >= limit {
            break;
        }
        let crowded = kept.iter().any(|&(x, y, _)| {
            x.abs_diff(peak.0) <= PEAK_SPACING && y.abs_diff(peak.1) <= PEAK_SPACING
        });
        if !crowded {
            kept.push(peak);
        }
    }
    kept
}

/// The best position near `(cx, cy)` at one level: every position within
/// `radius`, then uphill from the best of those while a neighbour scores
/// higher. A template that is smooth along one axis has a flat coarse peak
/// along it, so the coarse position can be off by more than the radius.
fn refine(
    plane: &Plane,
    template: &TemplateLevel,
    cx: usize,
    cy: usize,
    radius: usize,
) -> (usize, usize, f32) {
    let (w, h) = (template.inner.width, template.inner.height);
    let (max_x, max_y) = (plane.width - w, plane.height - h);
    let score_at = |x: usize, y: usize| score_at(plane, template, x, y);
    let (xs, ys) = (
        cx.saturating_sub(radius).min(max_x)..=(cx + radius).min(max_x),
        cy.saturating_sub(radius).min(max_y)..=(cy + radius).min(max_y),
    );
    let mut best = (cx.min(max_x), cy.min(max_y), f32::NEG_INFINITY);
    for y in ys.clone() {
        for x in xs.clone() {
            let score = score_at(x, y);
            if score > best.2 {
                best = (x, y, score);
            }
        }
    }
    for _ in 0..MAX_CLIMB {
        let (bx, by, _) = best;
        let mut next = best;
        for y in by.saturating_sub(1)..=(by + 1).min(max_y) {
            for x in bx.saturating_sub(1)..=(bx + 1).min(max_x) {
                if xs.contains(&x) && ys.contains(&y) {
                    continue; // already scored
                }
                let score = score_at(x, y);
                if score > next.2 {
                    next = (x, y, score);
                }
            }
        }
        if next.2 <= best.2 {
            break;
        }
        best = next;
    }
    best
}

/// Offset of a peak from the middle of three samples, by fitting a
/// parabola through them, within half a pixel.
fn parabolic_offset(before: f32, at: f32, after: f32) -> f32 {
    let curvature = before - 2.0 * at + after;
    if curvature >= 0.0 {
        return 0.0;
    }
    (0.5 * (before - after) / curvature).clamp(-0.5, 0.5)
}

fn run(
    image: &RgbaImage,
    search: Rect,
    template: &Template,
    min_score: Option<f32>,
) -> Vec<TemplateMatch> {
    run_with(image, search, template, min_score, true)
}

fn run_with(
    image: &RgbaImage,
    search: Rect,
    template: &Template,
    min_score: Option<f32>,
    pyramid: bool,
) -> Vec<TemplateMatch> {
    let planes = Pyramid::new(image, search, template.channel, template.coarse_levels());
    let base = &template.levels[0];
    candidates(&planes, template, min_score, pyramid)
        .into_iter()
        .map(|(x, y, score)| describe(&planes.planes[0], base, planes.search, x, y, score))
        .collect()
}

/// The search region of a frame on one channel, at full size and halved
/// `depth` times: built once, shared by every template of a search.
struct Pyramid {
    /// The search region as clipped to the image.
    search: Rect,
    planes: Vec<Plane>,
}

impl Pyramid {
    fn new(image: &RgbaImage, search: Rect, channel: Channel, depth: usize) -> Pyramid {
        let search = clip(search, image.width(), image.height());
        let full = Plane::from_gray(channel_image(image, search, channel));
        let mut planes = vec![full];
        while planes.len() <= depth {
            let last = &planes[planes.len() - 1];
            if last.width < 2 || last.height < 2 {
                break;
            }
            let next = last.half();
            planes.push(next);
        }
        Pyramid { search, planes }
    }
}

/// Where `template` matches in `pyramid`, as positions of its top-left
/// corner at full resolution with their scores: every position scoring at
/// least `min_score`, or just the best when `None`; overlapping matches
/// reduced to the strongest.
fn candidates(
    pyramid: &Pyramid,
    template: &Template,
    min_score: Option<f32>,
    use_pyramid: bool,
) -> Vec<(usize, usize, f32)> {
    let full = &pyramid.planes[0];
    let base = &template.levels[0];
    if full.width < base.inner.width || full.height < base.inner.height {
        return Vec::new();
    }

    // Use the pyramid only when scoring every position exactly would be
    // expensive, and only as deep as both the template and the search
    // region allow.
    let positions =
        ((full.width - base.inner.width + 1) * (full.height - base.inner.height + 1)) as u64;
    let mut depth = 0;
    if use_pyramid && positions * base.area() as u64 > EXHAUSTIVE_BUDGET {
        while depth + 1 < template.levels.len() && depth + 1 < pyramid.planes.len() {
            let next = &pyramid.planes[depth + 1];
            let level = &template.levels[depth + 1];
            if next.width < level.inner.width || next.height < level.inner.height {
                break;
            }
            depth += 1;
        }
    }
    let planes = &pyramid.planes[..=depth];

    // Positions below are of each level's matched part (its interior).
    // Score everywhere at the coarsest level, then carry the strongest
    // candidates down, refining each at every level and keeping fewer as
    // the rankings get more reliable.
    let coarsest = depth;
    let (plane, level) = (&planes[coarsest], &template.levels[coarsest]);
    let scores = score_everywhere(plane, level);
    let (columns, rows) = (
        plane.width - level.inner.width + 1,
        plane.height - level.inner.height + 1,
    );
    // Scores at full resolution are exact, so nothing below the requested
    // score can qualify there; coarser scores get some slack.
    let floor_at = |level: usize| match min_score {
        None => f32::NEG_INFINITY,
        Some(min) if level == 0 => min,
        Some(min) => min - COARSE_SLACK,
    };
    // The coarsest ranking is the least reliable, so it passes on the most
    // candidates; refining them one level down still costs less than the
    // search that ranked them.
    let first_cut = if coarsest == 0 {
        keep_at(0)
    } else {
        MAX_CANDIDATES
    };
    let mut candidates = local_maxima(&scores, columns, rows, floor_at(coarsest), first_cut);
    for finer in (0..coarsest).rev() {
        let (coarse_margin, fine_margin) = (
            template.levels[finer + 1].margin,
            template.levels[finer].margin,
        );
        candidates = candidates
            .into_iter()
            .map(|(x, y, _)| {
                refine(
                    &planes[finer],
                    &template.levels[finer],
                    to_finer(x, coarse_margin, fine_margin),
                    to_finer(y, coarse_margin, fine_margin),
                    REFINE_RADIUS,
                )
            })
            .filter(|&(_, _, score)| score >= floor_at(finer))
            .collect();
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
        // Two candidates can converge on one position.
        candidates.dedup_by(|a, b| (a.0, a.1) == (b.0, b.1));
        candidates.truncate(keep_at(finer));
    }

    // Keep the strongest of any matches overlapping by more than a quarter.
    // Full resolution has no margin, so these are the template's corners.
    let (w, h) = (base.inner.width, base.inner.height);
    let mut kept: Vec<(usize, usize, f32)> = Vec::new();
    for candidate in candidates {
        let overlaps = kept
            .iter()
            .any(|&(x, y, _)| 2 * x.abs_diff(candidate.0) < w && 2 * y.abs_diff(candidate.1) < h);
        if !overlaps {
            kept.push(candidate);
        }
        if min_score.is_none() {
            break;
        }
    }
    kept
}

/// A match at full resolution, with its sub-pixel centre.
fn describe(
    plane: &Plane,
    template: &TemplateLevel,
    search: Rect,
    x: usize,
    y: usize,
    score: f32,
) -> TemplateMatch {
    let (w, h) = (template.inner.width, template.inner.height);
    let score_at = |x: usize, y: usize| score_at(plane, template, x, y);
    let dx = if x > 0 && x + w < plane.width {
        parabolic_offset(score_at(x - 1, y), score, score_at(x + 1, y))
    } else {
        0.0
    };
    let dy = if y > 0 && y + h < plane.height {
        parabolic_offset(score_at(x, y - 1), score, score_at(x, y + 1))
    } else {
        0.0
    };
    let (left, top) = (search.x + x as u32, search.y + y as u32);
    TemplateMatch {
        bounds: Rect {
            x: left,
            y: top,
            w: w as u32,
            h: h as u32,
        },
        score,
        centre: (
            left as f32 + w as f32 / 2.0 + dx,
            top as f32 + h as f32 / 2.0 + dy,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    /// Deterministic texture with detail at several scales.
    fn scene(width: u32, height: u32, seed: u32) -> RgbaImage {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let blobs: Vec<(f32, f32, f32, f32)> = (0..40)
            .map(|_| {
                (
                    (noise() % width) as f32,
                    (noise() % height) as f32,
                    4.0 + (noise() % 30) as f32,
                    (noise() % 200) as f32,
                )
            })
            .collect();
        RgbaImage::from_fn(width, height, |x, y| {
            let mut v = 30.0;
            for &(bx, by, r, level) in &blobs {
                let d2 = (x as f32 - bx).powi(2) + (y as f32 - by).powi(2);
                if d2 < r * r {
                    v += level * 0.4;
                }
            }
            let v = (v as u32 + noise() % 9).min(255) as u8;
            Rgba([v, v / 2 + 20, 255 - v, 255])
        })
    }

    /// Flat panels, bars and labels, like an application's interface.
    fn ui_scene(width: u32, height: u32, seed: u32) -> RgbaImage {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        let mut next = move |limit: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % limit.max(1)
        };
        let mut image = RgbaImage::from_pixel(width, height, Rgba([28, 30, 36, 255]));
        for _ in 0..25 {
            let (x, y) = (next(width), next(height));
            let (w, h) = (20 + next(160), 10 + next(80));
            let colour = Rgba([next(256) as u8, next(256) as u8, next(256) as u8, 255]);
            for yy in y..(y + h).min(height) {
                for xx in x..(x + w).min(width) {
                    image.put_pixel(xx, yy, colour);
                }
            }
        }
        for _ in 0..30 {
            let text: String = (0..3 + next(8))
                .map(|_| char::from(b'A' + next(26) as u8))
                .collect();
            let (x, y, scale) = (next(width) as i64, next(height) as i64, 1 + next(2));
            let ink = Rgba([230, 230, 220, 255]);
            crate::draw::draw_text(&text, x, y, scale, |px, py| {
                if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                    image.put_pixel(px as u32, py as u32, ink);
                }
            });
        }
        image
    }

    /// An icon with structure at several scales, like a real one: a bright
    /// disc, a dark bar, a diagonal stroke and faint fine rings.
    fn icon(size: u32) -> RgbaImage {
        let s = size as f32;
        RgbaImage::from_fn(size, size, |x, y| {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let d = ((fx - s * 0.4).powi(2) + (fy - s * 0.45).powi(2)).sqrt();
            let mut v = 70.0;
            if d < s * 0.3 {
                v = 200.0
                    + if (d as u32 / 2).is_multiple_of(2) {
                        25.0
                    } else {
                        0.0
                    };
            }
            if fx > s * 0.72 && fx < s * 0.86 {
                v = 20.0;
            }
            if (fx - fy).abs() < 1.5 && fx > s * 0.5 {
                v = 245.0;
            }
            let v = v as u8;
            Rgba([v, 255 - v / 2, v / 3, 255])
        })
    }

    fn paste(target: &mut RgbaImage, source: &RgbaImage, at: (u32, u32)) {
        image::imageops::replace(target, source, i64::from(at.0), i64::from(at.1));
    }

    fn whole(image: &RgbaImage) -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        }
    }

    /// A creature with a face, facing left: eyes on the left, a dark mouth.
    fn creature(flip: bool) -> RgbaImage {
        let mut m = RgbaImage::from_pixel(40, 36, Rgba([240, 140, 40, 255]));
        for y in 0..36 {
            for x in 0..40 {
                let (fx, fy) = (x as f32 - 20.0, y as f32 - 18.0);
                if fx * fx / 300.0 + fy * fy / 250.0 > 1.0 {
                    m.put_pixel(x, y, Rgba([90, 160, 90, 255]));
                }
            }
        }
        let eye = if flip { [26, 30] } else { [10, 14] };
        for y in 10..16 {
            for x in eye[0]..eye[1] {
                m.put_pixel(x, y, Rgba([20, 20, 20, 255]));
            }
        }
        for y in 24..27 {
            for x in 8..32 {
                m.put_pixel(x, y, Rgba([120, 40, 20, 255]));
            }
        }
        m
    }

    /// A striped field with the creature twice: as taught, and facing the
    /// other way.
    fn field() -> RgbaImage {
        let mut s = RgbaImage::from_fn(640, 360, |x, y| {
            let v = ((x / 23 + y / 31) % 3) as u8 * 12;
            Rgba([80 + v, 150 + v, 80 + v, 255])
        });
        paste(&mut s, &creature(false), (100, 200));
        paste(
            &mut s,
            &image::imageops::flip_horizontal(&creature(false)),
            (420, 90),
        );
        s
    }

    #[test]
    fn a_set_finds_a_picture_facing_either_way_in_one_search() {
        let mut set = TemplateSet::new(Channel::Luma, true);
        assert!(set.add(&creature(false)));
        assert_eq!(set.len(), 1);
        let frame = field();
        let options = SetSearch {
            min_score: 0.75,
            limit: 5,
            max_colour_shift: Some(60.0),
        };
        let found = find_set(&frame, whole(&frame), &set, options);
        assert_eq!(found.len(), 2, "{found:?}");
        let mut places: Vec<(u32, u32, bool)> = found
            .iter()
            .map(|f| (f.bounds.x, f.bounds.y, f.mirrored))
            .collect();
        places.sort();
        assert_eq!(places, vec![(100, 200, false), (420, 90, true)]);
        assert!(found.iter().all(|f| f.score > 0.9 && f.picture == 0));
        assert!(found.iter().all(|f| f.bounds.w == 40 && f.bounds.h == 36));
        // Not mirrored: only the one as taught.
        let mut plain = TemplateSet::new(Channel::Luma, false);
        plain.add(&creature(false));
        let found = find_set(&frame, whole(&frame), &plain, options);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!((found[0].bounds.x, found[0].bounds.y), (100, 200));
        // Within a region that holds neither.
        let region = Rect {
            x: 200,
            y: 0,
            w: 200,
            h: 360,
        };
        assert!(find_set(&frame, region, &set, options).is_empty());
        // A limit of one keeps the strongest.
        let one = find_set(
            &frame,
            whole(&frame),
            &set,
            SetSearch {
                limit: 1,
                ..options
            },
        );
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn a_set_of_poses_ranks_every_picture_together() {
        // Two poses: the creature, and the creature with its mouth closed.
        let mut closed = creature(false);
        for y in 24..27 {
            for x in 8..32 {
                closed.put_pixel(x, y, Rgba([240, 140, 40, 255]));
            }
        }
        let mut frame = field();
        paste(&mut frame, &closed, (300, 250));
        let mut set = TemplateSet::new(Channel::Luma, false);
        set.add(&creature(false));
        set.add(&closed);
        let found = find_set(
            &frame,
            whole(&frame),
            &set,
            SetSearch {
                min_score: 0.9,
                limit: 8,
                max_colour_shift: None,
            },
        );
        let mut places: Vec<(u32, u32, usize)> = found
            .iter()
            .map(|f| (f.bounds.x, f.bounds.y, f.picture))
            .collect();
        places.sort();
        // Each copy is reported once, by the pose that fits it best.
        assert_eq!(places, vec![(100, 200, 0), (300, 250, 1)], "{found:?}");
    }

    #[test]
    fn the_colour_check_drops_a_lookalike_of_another_colour() {
        // The same shape, as light, in green instead of orange: the same
        // picture on the luma channel, another colour.
        let original = creature(false);
        let green = RgbaImage::from_fn(40, 36, |x, y| {
            let p = original.get_pixel(x, y).0;
            if p == [240, 140, 40, 255] {
                Rgba([120, 190, 60, 255])
            } else {
                Rgba(p)
            }
        });
        let mut frame = field();
        paste(&mut frame, &green, (500, 250));
        let mut set = TemplateSet::new(Channel::Luma, false);
        set.add(&creature(false));
        let find = |shift| {
            find_set(
                &frame,
                whole(&frame),
                &set,
                SetSearch {
                    min_score: 0.6,
                    limit: 8,
                    max_colour_shift: shift,
                },
            )
        };
        let without = find(None);
        assert!(
            without
                .iter()
                .any(|f| (f.bounds.x, f.bounds.y) == (500, 250)),
            "{without:?}"
        );
        let with = find(Some(60.0));
        assert_eq!(with.len(), 1, "{with:?}");
        assert_eq!((with[0].bounds.x, with[0].bounds.y), (100, 200));
    }

    #[test]
    fn a_set_with_one_picture_agrees_with_find_all() {
        let mut frame = scene(420, 300, 9);
        for at in [(20, 30), (200, 100), (330, 220)] {
            paste(&mut frame, &icon(32), at);
        }
        let template = Template::from_image(&icon(32), Channel::Luma).unwrap();
        let all = find_all(&frame, whole(&frame), &template, 0.9);
        let mut set = TemplateSet::new(Channel::Luma, false);
        set.add(&icon(32));
        let found = find_set(
            &frame,
            whole(&frame),
            &set,
            SetSearch {
                min_score: 0.9,
                limit: 16,
                max_colour_shift: None,
            },
        );
        let a: Vec<(u32, u32)> = all.iter().map(|m| (m.bounds.x, m.bounds.y)).collect();
        let b: Vec<(u32, u32)> = found.iter().map(|m| (m.bounds.x, m.bounds.y)).collect();
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
        assert_eq!(
            template.mean_rgb(),
            set.templates().next().unwrap().mean_rgb()
        );
    }

    #[test]
    fn the_foreground_is_what_stands_out_from_the_edges() {
        // Sky over grass, with a dark creature in the middle.
        let mut crop = RgbaImage::from_fn(60, 50, |_, y| {
            if y < 25 {
                Rgba([120, 180, 240, 255])
            } else {
                Rgba([70, 150, 60, 255])
            }
        });
        for y in 15..40 {
            for x in 20..44 {
                crop.put_pixel(x, y, Rgba([30, 20, 20, 255]));
            }
        }
        let fg = foreground(&crop).unwrap();
        assert_eq!((fg.x, fg.y, fg.w, fg.h), (18, 13, 28, 29));
        // Nothing stands out of a flat picture, and a tiny crop is kept whole.
        assert!(foreground(&RgbaImage::from_pixel(60, 50, Rgba([9, 9, 9, 255]))).is_none());
        assert!(foreground(&image::imageops::crop_imm(&crop, 0, 0, 6, 6).to_image()).is_none());
        // A picture that is all thing has no background to take away.
        assert!(foreground(&scene(40, 40, 1)).is_none());
    }

    #[test]
    fn finds_a_planted_icon_exactly() {
        for (size, at) in [(12u32, (7u32, 5u32)), (32, (301, 97)), (64, (40, 200))] {
            let mut frame = scene(420, 300, size);
            paste(&mut frame, &icon(size), at);
            let template = Template::from_image(&icon(size), Channel::Luma).unwrap();
            let found = find_best(&frame, whole(&frame), &template).expect("fits");
            assert_eq!((found.bounds.x, found.bounds.y), at, "{size}px icon");
            assert!(found.score > 0.999, "score {}", found.score);
            assert_eq!((found.bounds.w, found.bounds.h), (size, size));
        }
    }

    /// Scores ignore brightness and contrast: a dimmer, flatter copy of the
    /// icon still correlates perfectly.
    #[test]
    fn brightness_and_contrast_do_not_matter() {
        let mut frame = scene(200, 150, 3);
        let mut dim = icon(24);
        for p in dim.pixels_mut() {
            for c in 0..3 {
                p[c] = (20.0 + p[c] as f32 * 0.5) as u8;
            }
        }
        paste(&mut frame, &dim, (100, 60));
        let template = Template::from_image(&icon(24), Channel::Max).unwrap();
        let found = find_best(&frame, whole(&frame), &template).unwrap();
        assert_eq!((found.bounds.x, found.bounds.y), (100, 60));
        assert!(found.score > 0.99, "score {}", found.score);
    }

    /// The pyramid must find what exhaustive search finds.
    #[test]
    fn coarse_to_fine_agrees_with_exhaustive_search() {
        for seed in 0..12u32 {
            let size = 24 + (seed % 3) * 16;
            let mut frame = scene(640, 360, seed + 100);
            let at = (37 + seed * 41 % 500, 11 + seed * 23 % 300);
            paste(&mut frame, &icon(size), at);
            let template = Template::from_image(&icon(size), Channel::Luma).unwrap();
            assert!(template.levels.len() > 1, "a pyramid is in use");
            let found = find_best(&frame, whole(&frame), &template).unwrap();
            assert_eq!((found.bounds.x, found.bounds.y), at, "seed {seed}");
        }
    }

    #[test]
    fn find_all_returns_every_copy_once_strongest_first() {
        let mut frame = scene(500, 320, 9);
        let spots = [(20u32, 30u32), (200, 40), (410, 250), (60, 260)];
        for &at in &spots {
            paste(&mut frame, &icon(28), at);
        }
        // One copy slightly degraded, so the order is known.
        for y in 250..278 {
            for x in 410..438 {
                let p = frame.get_pixel_mut(x, y);
                p[0] = p[0].saturating_add(((x * 7 + y * 3) % 23) as u8);
            }
        }
        let template = Template::from_image(&icon(28), Channel::Luma).unwrap();
        let found = find_all(&frame, whole(&frame), &template, 0.8);
        let mut positions: Vec<(u32, u32)> =
            found.iter().map(|m| (m.bounds.x, m.bounds.y)).collect();
        assert_eq!(found.len(), 4, "{found:?}");
        assert_eq!(positions[3], (410, 250), "the degraded copy scores lowest");
        positions.sort();
        let mut expected = spots.to_vec();
        expected.sort();
        assert_eq!(positions, expected);
        assert!(found.windows(2).all(|pair| pair[0].score >= pair[1].score));
    }

    #[test]
    fn find_all_in_a_small_search_is_exhaustive() {
        let mut frame = scene(90, 60, 4);
        paste(&mut frame, &icon(10), (5, 5));
        paste(&mut frame, &icon(10), (70, 40));
        let template = Template::from_image(&icon(10), Channel::Luma).unwrap();
        let found = find_all(&frame, whole(&frame), &template, 0.95);
        let mut positions: Vec<(u32, u32)> =
            found.iter().map(|m| (m.bounds.x, m.bounds.y)).collect();
        positions.sort();
        assert_eq!(positions, vec![(5, 5), (70, 40)]);
    }

    #[test]
    fn search_is_restricted_to_the_region_and_clipped() {
        let mut frame = scene(300, 200, 5);
        paste(&mut frame, &icon(20), (30, 30));
        paste(&mut frame, &icon(20), (250, 150));
        let template = Template::from_image(&icon(20), Channel::Luma).unwrap();
        // A region reaching past the image still searches what is inside.
        let region = Rect {
            x: 200,
            y: 100,
            w: 500,
            h: 500,
        };
        let found = find_best(&frame, region, &template).unwrap();
        assert_eq!((found.bounds.x, found.bounds.y), (250, 150));
        let tiny = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        assert!(find_best(&frame, tiny, &template).is_none());
    }

    #[test]
    fn uniform_or_empty_templates_are_refused() {
        let flat = RgbaImage::from_pixel(16, 16, Rgba([90, 90, 90, 255]));
        assert!(Template::from_image(&flat, Channel::Luma).is_none());
        assert!(Template::from_image(&RgbaImage::new(0, 0), Channel::Luma).is_none());
        let region = Rect {
            x: 50,
            y: 50,
            w: 4,
            h: 4,
        };
        assert!(Template::new(&icon(16), region, Channel::Luma).is_none());
    }

    #[test]
    fn sub_pixel_centre_follows_a_half_pixel_shift() {
        // Blend the icon at x and x+1 to draw it half a pixel to the right.
        let base = icon(24);
        let mut frame = RgbaImage::from_pixel(120, 80, Rgba([30, 225, 10, 255]));
        let mut a = frame.clone();
        let mut b = frame.clone();
        paste(&mut a, &base, (50, 30));
        paste(&mut b, &base, (51, 30));
        for (out, (pa, pb)) in frame.pixels_mut().zip(a.pixels().zip(b.pixels())) {
            for c in 0..4 {
                out[c] = (u16::from(pa[c]) + u16::from(pb[c])).div_ceil(2) as u8;
            }
        }
        let template = Template::from_image(&base, Channel::Luma).unwrap();
        let found = find_best(&frame, whole(&frame), &template).unwrap();
        let expected = 50.5 + 12.0;
        assert!(
            (found.centre.0 - expected).abs() < 0.25,
            "centre x {} vs {expected}",
            found.centre.0
        );
        assert!(
            (found.centre.1 - 42.0).abs() < 0.25,
            "centre y {}",
            found.centre.1
        );
    }

    #[test]
    fn locate_explains_a_miss() {
        let frame = scene(200, 120, 6);
        let template = Template::from_image(&icon(20), Channel::Luma).unwrap();
        let detection = locate(&frame, whole(&frame), &template, 0.9);
        assert!(!detection.is_present());
        let reason = detection.failure_reason.unwrap();
        assert!(reason.contains("below 0.90"), "{reason}");

        let mut frame = frame;
        paste(&mut frame, &icon(20), (90, 50));
        let detection = locate(&frame, whole(&frame), &template, 0.9);
        assert!(detection.is_present());
        assert!(detection.confidence.value() > 0.95);
    }

    #[test]
    fn halving_filters_and_drops_odd_edges() {
        let plane = Plane {
            width: 7,
            height: 5,
            pixels: (0..35).map(|i| (i * 37 % 251) as u8).collect(),
        };
        let half = plane.half();
        assert_eq!((half.width, half.height), (3, 2));
        // Reference: the 1-3-3-1 weights applied directly, edges repeated.
        let at = |x: isize, y: isize| {
            let (x, y) = (x.clamp(0, 6) as usize, y.clamp(0, 4) as usize);
            f64::from(plane.pixels[y * 7 + x])
        };
        let weights = [1.0, 3.0, 3.0, 1.0];
        for j in 0..2isize {
            for i in 0..3isize {
                let mut total = 0.0;
                for (dy, wy) in weights.iter().enumerate() {
                    for (dx, wx) in weights.iter().enumerate() {
                        total += wx * wy * at(2 * i - 1 + dx as isize, 2 * j - 1 + dy as isize);
                    }
                }
                let want = (total / 64.0).round() as u8;
                assert_eq!(half.pixels[(j * 3 + i) as usize], want, "pixel {i},{j}");
            }
        }
        let flat = Plane {
            width: 6,
            height: 6,
            pixels: vec![77; 36],
        };
        assert!(flat.half().pixels.iter().all(|&v| v == 77));
    }

    #[test]
    fn detail_too_fine_to_halve_keeps_the_search_exact() {
        // A one-pixel checkerboard is flat once halved and changes
        // completely with the grid's alignment: no coarse level may be used.
        let checker = RgbaImage::from_fn(24, 24, |x, y| {
            let v = if (x + y) % 2 == 0 { 30 } else { 220 };
            Rgba([v, v, v, 255])
        });
        let template = Template::from_image(&checker, Channel::Luma).unwrap();
        assert_eq!(template.levels.len(), 1);
        let mut frame = scene(300, 200, 12);
        paste(&mut frame, &checker, (131, 77));
        let found = find_best(&frame, whole(&frame), &template).unwrap();
        assert_eq!((found.bounds.x, found.bounds.y), (131, 77));
    }

    #[test]
    fn icons_with_coarse_structure_get_a_pyramid() {
        for (size, levels) in [
            (12u32, 1usize),
            (16, 1),
            (24, 2),
            (32, 2),
            (64, 3),
            (128, 4),
        ] {
            let template = Template::from_image(&icon(size), Channel::Luma).unwrap();
            assert_eq!(template.levels.len(), levels, "{size}px icon");
        }
    }

    #[test]
    fn sliding_sums_match_scoring_each_window_directly() {
        let frame = scene(61, 43, 8);
        let plane = Plane::from_gray(channel_image(&frame, whole(&frame), Channel::Luma));
        let template = TemplateLevel::new(
            Plane::from_gray(channel_image(
                &icon(9),
                Rect {
                    x: 0,
                    y: 0,
                    w: 9,
                    h: 7,
                },
                Channel::Luma,
            )),
            0,
        )
        .unwrap();
        let scores = score_everywhere(&plane, &template);
        let columns = 61 - 9 + 1;
        for (i, &score) in scores.iter().enumerate() {
            let (x, y) = (i % columns, i / columns);
            assert_eq!(score, score_at(&plane, &template, x, y));
        }
        assert_eq!(scores.len(), columns * (43 - 7 + 1));
    }

    /// Templates cut from real-looking content at random sizes and
    /// alignments: the pyramid must find what exhaustive search finds.
    #[test]
    #[ignore = "slow in debug builds; run with --release -- --ignored"]
    fn pyramid_recall_on_random_crops() {
        let mut misses = Vec::new();
        let trials: u32 = std::env::var("TRIALS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);
        for trial in 0..trials {
            let frame = if trial % 2 == 0 {
                scene(480, 320, 1000 + trial * 7919)
            } else {
                ui_scene(480, 320, 1000 + trial * 7919)
            };
            let size = 12 + trial % 53;
            let (x, y) = ((trial * 97) % (480 - size), (trial * 61) % (320 - size));
            let region = Rect {
                x,
                y,
                w: size,
                h: size,
            };
            let Some(template) = Template::new(&frame, region, Channel::Luma) else {
                continue;
            };
            let fast = find_best(&frame, whole(&frame), &template).unwrap();
            let exact = run_with(&frame, whole(&frame), &template, None, false)
                .into_iter()
                .next()
                .unwrap();
            // On smooth content a small template can have near-identical
            // copies; settling on one of those is not a miss.
            if fast.score < exact.score - 0.05 {
                misses.push((trial, size, template.levels.len(), fast.score, exact.score));
            }
        }
        assert!(misses.is_empty(), "{} misses: {misses:?}", misses.len());
    }
}
