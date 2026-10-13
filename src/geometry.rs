//! Rectangles, pixel-run segmentation, region grouping, and bar-fill
//! measurement.
//!
//! Every function operates on a borrowed `&RgbaImage` and avoids copying
//! pixels; the only allocations are the result buffers, whose size is
//! data-dependent.

use image::{Pixel, Rgba, RgbaImage};

/// Pixels `x0..=x1` of row `y`, with their x coordinate, read straight from
/// the image buffer.
///
/// `get_pixel` bounds-checks every access and recomputes the row offset
/// each time; scanning a row through one slice does neither, which is most
/// of the cost of a full-frame scan. The caller guarantees `y < height` and
/// `x0 <= x1 < width`.
#[inline]
pub(crate) fn row_pixels(
    image: &RgbaImage,
    y: u32,
    x0: u32,
    x1: u32,
) -> impl Iterator<Item = (u32, &Rgba<u8>)> {
    let width = image.width() as usize;
    let start = (y as usize * width + x0 as usize) * 4;
    let end = (y as usize * width + x1 as usize + 1) * 4;
    image.as_raw()[start..end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| Rgba::from_slice(pixel.as_slice()))
        .enumerate()
        .map(move |(offset, pixel)| (x0 + offset as u32, pixel))
}

/// A generic rectangle in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    /// End X coordinate inclusive.
    pub fn x2(&self) -> u32 {
        self.x.saturating_add(self.w.saturating_sub(1))
    }

    /// End Y coordinate inclusive.
    pub fn y2(&self) -> u32 {
        self.y.saturating_add(self.h.saturating_sub(1))
    }

    /// Rectangle area in pixels.
    pub fn area(&self) -> u32 {
        self.w.saturating_mul(self.h)
    }

    /// Center point in pixel coordinates.
    pub fn center(&self) -> (f32, f32) {
        (
            self.x as f32 + self.w as f32 / 2.0,
            self.y as f32 + self.h as f32 / 2.0,
        )
    }
}

/// A rectangle as fractions of the frame's width and height (0 to 1, from
/// the top left), so it means the same place after the window is resized.
///
/// Constructed through [`NormRect::new`], which orders and clamps the
/// corners, so `x0 <= x1` and `y0 <= y1` always hold.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl NormRect {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> NormRect {
        NormRect {
            x0: x0.min(x1).clamp(0.0, 1.0),
            y0: y0.min(y1).clamp(0.0, 1.0),
            x1: x0.max(x1).clamp(0.0, 1.0),
            y1: y0.max(y1).clamp(0.0, 1.0),
        }
    }

    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }

    /// Grown by `fx` of its width and `fy` of its height on each side.
    pub fn grown(&self, fx: f32, fy: f32) -> NormRect {
        let (dx, dy) = (self.width() * fx, self.height() * fy);
        NormRect::new(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)
    }

    /// The smallest box holding both.
    pub fn union(&self, other: &NormRect) -> NormRect {
        NormRect::new(
            self.x0.min(other.x0),
            self.y0.min(other.y0),
            self.x1.max(other.x1),
            self.y1.max(other.y1),
        )
    }

    /// In pixels of a `width`×`height` frame: x, y, w, h (at least 1×1).
    pub fn pixels(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        // A hair of slack, so a box made from whole pixels comes back exact.
        let lo = |v: f32, n: u32| {
            ((v * n as f32 + 0.01).floor().max(0.0) as u32).min(n.saturating_sub(1))
        };
        let hi = |v: f32, n: u32| (v * n as f32 - 0.01).ceil().max(0.0) as u32;
        let (x0, y0) = (lo(self.x0, width), lo(self.y0, height));
        let x1 = hi(self.x1, width).clamp(x0 + 1, width.max(1));
        let y1 = hi(self.y1, height).clamp(y0 + 1, height.max(1));
        (x0, y0, x1 - x0, y1 - y0)
    }

    /// The same, as a [`Rect`].
    pub fn rect(&self, width: u32, height: u32) -> Rect {
        let (x, y, w, h) = self.pixels(width, height);
        Rect { x, y, w, h }
    }

    /// A box in pixels of a `width`×`height` frame, as fractions.
    pub fn from_pixels(x: u32, y: u32, w: u32, h: u32, width: u32, height: u32) -> NormRect {
        let (fw, fh) = (width.max(1) as f32, height.max(1) as f32);
        NormRect::new(
            x as f32 / fw,
            y as f32 / fh,
            (x + w) as f32 / fw,
            (y + h) as f32 / fh,
        )
    }

    /// A [`Rect`] of a `width`×`height` frame, as fractions.
    pub fn from_rect(rect: Rect, width: u32, height: u32) -> NormRect {
        NormRect::from_pixels(rect.x, rect.y, rect.w, rect.h, width, height)
    }

    /// This box (given in fractions of `inner`, which sits at `inner` within
    /// a larger frame) in fractions of the larger frame.
    pub fn within(&self, inner: &NormRect) -> NormRect {
        NormRect::new(
            inner.x0 + self.x0 * inner.width(),
            inner.y0 + self.y0 * inner.height(),
            inner.x0 + self.x1 * inner.width(),
            inner.y0 + self.y1 * inner.height(),
        )
    }
}

/// Find runs of `min_width`-or-longer consecutive pixels along row `y` in
/// `[x0, x1]` matching `predicate`. Returns inclusive `(start_x, end_x)` pairs.
///
/// A run at least `min_width` long holds one of every `min_width`-th pixel
/// along the row, so (for a `min_width` of 3 or more) only those are asked
/// about at first, and each that matches is followed out to both ends of
/// its run; the rest of the row is never asked about. The same runs: the
/// predicate is asked about fewer pixels, so it must say the same of a
/// pixel whenever it is asked.
pub fn segment_row<F>(
    image: &RgbaImage,
    y: u32,
    x0: u32,
    x1: u32,
    min_width: u32,
    predicate: F,
) -> Vec<(u32, u32)>
where
    F: Fn(&Rgba<u8>) -> bool,
{
    let mut result = Vec::new();
    if y >= image.height() || image.width() == 0 {
        return result;
    }
    let mut start = None;
    let x1 = x1.min(image.width() - 1);
    if x0 > x1 {
        return result;
    }
    let step = min_width as usize;
    if step > 2 {
        let width = image.width() as usize;
        let at = (y as usize * width + x0 as usize) * 4;
        let count = (x1 - x0 + 1) as usize;
        let row = image.as_raw()[at..at + 4 * count].as_chunks::<4>().0;
        let matches = |i: usize| predicate(Rgba::from_slice(&row[i]));
        // Where the last run found ended (one past it: a miss, or the end).
        let mut done = 0;
        for sample in (step - 1..count).step_by(step) {
            if sample < done || !matches(sample) {
                continue;
            }
            let mut begin = sample;
            while begin > 0 && matches(begin - 1) {
                begin -= 1;
            }
            let mut end = sample + 1;
            while end < count && matches(end) {
                end += 1;
            }
            if end - begin >= step {
                result.push((x0 + begin as u32, x0 + end as u32 - 1));
            }
            done = end;
        }
        return result;
    }

    for (x, pixel) in row_pixels(image, y, x0, x1) {
        let is_target = predicate(pixel);
        if is_target {
            if start.is_none() {
                start = Some(x);
            }
        } else if let Some(begin) = start {
            let len = x - begin;
            if len >= min_width {
                result.push((begin, x.saturating_sub(1)));
            }
            start = None;
        }
    }

    if let Some(begin) = start {
        let len = x1.saturating_add(1).saturating_sub(begin);
        if len >= min_width {
            result.push((begin, x1));
        }
    }

    result
}

/// Number of overlapping units between two inclusive 1-D spans.
fn overlap(a: (u32, u32), b: (u32, u32)) -> u32 {
    if a.1 < b.0 || b.1 < a.0 {
        return 0;
    }
    let left = a.0.max(b.0);
    let right = a.1.min(b.1);
    right.saturating_sub(left).saturating_add(1)
}

/// Horizontal edges: runs of at least `min_run` adjacent pixels whose
/// luminance differs from the pixel below by `threshold` (0–255) or more,
/// grouped row to row into rectangles. A ledge, a floor, the edge of a
/// panel: anything drawn as a line where one shade meets another.
///
/// The frame's luminance is taken once (`threshold::channel_image`), and
/// the rows compared as bytes, so a frame costs one pass.
pub fn horizontal_edges(image: &RgbaImage, threshold: u8, min_run: u32) -> Vec<Rect> {
    let (width, height) = image.dimensions();
    if width == 0 || height < 2 {
        return Vec::new();
    }
    let luma = crate::threshold::channel_image(
        image,
        Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        },
        crate::threshold::Channel::Luma,
    );
    let plane = luma.as_raw();
    let width = width as usize;
    let mut rows: Vec<(u32, u32, u32)> = Vec::new();
    // Each pair of rows marked a byte a column (a loop the compiler
    // vectorises), then the marks read eight at a time: a row with no edge
    // in eight columns, or edge all along them, is passed in one step.
    let mut marks = vec![0u8; width.next_multiple_of(8)];
    for y in 0..height as usize - 1 {
        let (above, below) = (
            &plane[y * width..(y + 1) * width],
            &plane[(y + 1) * width..(y + 2) * width],
        );
        for (mark, (&a, &b)) in marks.iter_mut().zip(above.iter().zip(below)) {
            *mark = u8::from(a.abs_diff(b) >= threshold);
        }
        let words = marks.as_chunks::<8>().0;
        // The first column at or after `x` marked `edge` (or not), or the
        // row's width.
        let next = |x: usize, edge: bool| -> usize {
            let mut x = x;
            while x < width {
                let word = u64::from_le_bytes(words[x / 8]) >> (8 * (x % 8));
                let wanted = if edge {
                    word
                } else {
                    !word & 0x0101_0101_0101_0101 >> (8 * (x % 8))
                };
                if wanted != 0 {
                    return (x + wanted.trailing_zeros() as usize / 8).min(width);
                }
                x = (x / 8 + 1) * 8;
            }
            width
        };
        let mut x = 0;
        while x < width {
            let begin = next(x, true);
            if begin >= width {
                break;
            }
            let end = next(begin, false);
            if (end - begin) as u32 >= min_run {
                rows.push((y as u32, begin as u32, end as u32 - 1));
            }
            x = end;
        }
    }
    group_segments(rows, 1, 0)
}

/// Group per-row horizontal segments into rectangles by greedily continuing
/// the best-overlapping active rectangle from the previous row, closing out
/// rectangles once a vertical gap larger than `max_gap` rows is seen.
///
/// `rows` holds `(y, start_x, end_x)` segments in top-to-bottom scan order;
/// a row may contribute any number of segments. All segments of one row are
/// matched against the active rectangles *together* — matching them one at a
/// time would make side-by-side regions evict each other from the active
/// set, which silently dropped every multi-region row.
pub fn group_segments(rows: Vec<(u32, u32, u32)>, min_height: u32, max_gap: u32) -> Vec<Rect> {
    let mut active = Vec::<Rect>::new();
    let mut completed = Vec::<Rect>::new();
    let mut prev_y: Option<u32> = None;

    let mut index = 0;
    while index < rows.len() {
        // Collect every segment that belongs to this row.
        let y = rows[index].0;
        let row_end = rows[index..]
            .iter()
            .position(|&(row_y, _, _)| row_y != y)
            .map(|offset| index + offset)
            .unwrap_or(rows.len());
        let segments = &rows[index..row_end];
        index = row_end;

        if let Some(prev_y) = prev_y
            && y > prev_y.saturating_add(max_gap).saturating_add(1)
        {
            completed.extend(active.drain(..).filter(|rect| rect.h >= min_height));
        }
        prev_y = Some(y);

        let mut next_active: Vec<Rect> = Vec::new();
        // Index into `next_active` for each already-matched active rect, so
        // a region that splits into several segments extends one rectangle.
        let mut continuation: Vec<Option<usize>> = vec![None; active.len()];
        for &(_, segment_x0, segment_x1) in segments {
            let mut best_match = None;
            let mut best_overlap = 0;
            for (idx, rect) in active.iter().enumerate() {
                let overlap = overlap((rect.x, rect.x2()), (segment_x0, segment_x1));
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_match = Some(idx);
                }
            }
            if let Some(idx) = best_match
                && best_overlap * 3 >= (segment_x1 - segment_x0 + 1).max(active[idx].w)
            {
                let target = match continuation[idx] {
                    Some(position) => position,
                    None => {
                        next_active.push(active[idx]);
                        continuation[idx] = Some(next_active.len() - 1);
                        next_active.len() - 1
                    }
                };
                let rect = &mut next_active[target];
                rect.x = rect.x.min(segment_x0);
                rect.w = rect
                    .x2()
                    .max(segment_x1)
                    .saturating_sub(rect.x)
                    .saturating_add(1);
                rect.h = y.saturating_sub(rect.y).saturating_add(1);
                continue;
            }
            next_active.push(Rect {
                x: segment_x0,
                y,
                w: segment_x1.saturating_sub(segment_x0).saturating_add(1),
                h: 1,
            });
        }

        for (idx, rect) in active.into_iter().enumerate() {
            if continuation[idx].is_none() && rect.h >= min_height {
                completed.push(rect);
            }
        }
        active = next_active;
    }

    completed.extend(active.into_iter().filter(|rect| rect.h >= min_height));
    completed
}

/// Fewer text pixels than this in a region is noise, not a line of text.
const MIN_TEXT_PIXELS: u32 = 6;

/// Padding added around a located text block, so glyph edges are not
/// clipped by the returned rectangle.
const TEXT_BLOCK_PADDING: u32 = 2;

/// Find the bounding box of the text-like pixels within `region`, padded
/// slightly. Returns `None` when the region holds too few to be text.
pub fn find_text_block(image: &RgbaImage, region: Rect) -> Option<Rect> {
    let x_end = region.x.saturating_add(region.w).min(image.width());
    let y_end = region.y.saturating_add(region.h).min(image.height());

    let mut bounds: Option<(u32, u32, u32, u32)> = None;
    let mut count = 0u32;
    if region.x >= x_end {
        return None;
    }

    // `is_text_pixel` from tables (`color::TextTables`): the same answers.
    let tables = crate::color::TextTables::get();
    let width = image.width() as usize;
    for y in region.y..y_end {
        let at = (y as usize * width + region.x as usize) * 4;
        let row = image.as_raw()[at..at + 4 * (x_end - region.x) as usize]
            .as_chunks::<4>()
            .0;
        for (i, pixel) in row.iter().enumerate() {
            if !tables.matches(pixel) {
                continue;
            }
            let x = region.x + i as u32;
            count += 1;
            bounds = Some(match bounds {
                None => (x, x, y, y),
                Some((min_x, max_x, min_y, max_y)) => {
                    (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
                }
            });
        }
    }

    let (min_x, max_x, min_y, max_y) = bounds?;
    if count < MIN_TEXT_PIXELS {
        return None;
    }

    let padding = TEXT_BLOCK_PADDING;
    Some(Rect {
        x: min_x.saturating_sub(padding),
        y: min_y.saturating_sub(padding),
        w: max_x
            .saturating_sub(min_x)
            .saturating_add(1)
            .saturating_add(padding.saturating_mul(2)),
        h: max_y
            .saturating_sub(min_y)
            .saturating_add(1)
            .saturating_add(padding.saturating_mul(2)),
    })
}

pub fn find_text_block_in_regions(image: &RgbaImage, regions: &[Rect]) -> Option<Rect> {
    regions
        .iter()
        .filter_map(|region| find_text_block(image, *region))
        .next()
}

/// How full a horizontal bar is, as a percentage of its own track.
///
/// Measuring a bar means comparing the filled span against the track it
/// sits in, and the track's width is not known in advance: it changes with
/// resolution, UI scale, and skin. So rather than dividing by a guessed
/// reference width, this walks outward from the filled span along the bar's
/// own rows and classifies each column as filled (the bar colour) or empty
/// (the groove behind it), stopping at the first column that is neither.
/// The result normalises itself:
///
/// ```text
///   [############--------]   filled=12 empty=8  -> 60%
///    ^ track found by walking out from the filled span
/// ```
///
/// The groove's colour is *learned* from the pixels just past the fill
/// rather than assumed. Assuming is what breaks: real UIs mix light and
/// dark grooves in the same screen, so any fixed "empty is dark" rule
/// silently measures the light-grooved bars as permanently full. Sampling
/// adapts to whatever the skin uses and stops at the bar's border, which is
/// a different colour again.
///
/// A column votes by majority over the rows it spans, so antialiasing and
/// the bar's border do not swing the result. Returns `None` when no track
/// can be established, which is honest about "could not measure" instead of
/// reporting a fabricated 0%.
pub fn measure_bar_fill<F>(image: &RgbaImage, fill: Rect, search: Rect, is_filled: F) -> Option<f32>
where
    F: Fn(&Rgba<u8>) -> bool,
{
    let groove = match sample_groove_color(image, fill, search, &is_filled) {
        Groove::Colour(colour) => Some(colour),
        // Past the fill lies the world outside the bar: the bar is full.
        Groove::Background => None,
        // Nothing past the fill to sample: nothing can be said about the
        // track from here.
        Groove::Unknown => return None,
    };
    let is_empty = |pixel: &Rgba<u8>| match groove {
        Some(reference) => is_similar_color(pixel, reference, GROOVE_TOLERANCE),
        None => false,
    };
    if fill.w == 0 || fill.h == 0 {
        return None;
    }
    let (y_start, y_end) = bar_core_rows(image, fill)?;

    // Classify one column by majority vote across the bar's rows.
    let classify = |x: u32| -> ColumnKind {
        if x >= image.width() {
            return ColumnKind::Neither;
        }
        let (mut filled, mut empty) = (0u32, 0u32);
        for y in y_start..y_end {
            let pixel = image.get_pixel(x, y);
            if is_filled(pixel) {
                filled += 1;
            } else if is_empty(pixel) {
                empty += 1;
            }
        }
        column_kind(filled, empty, y_end - y_start)
    };
    walk_bar(image, fill, search, classify)
}

/// [`measure_bar_fill`] for a bar of a colour, `is_filled` being
/// [`crate::color::is_color_pixel`] with `colour` (its hue range, least
/// saturation and least value): the same reading, the columns' votes taken
/// 64 columns at a time from each row's pixels tested together
/// (`kernels::color_mask`), where the general one tests a pixel at a time.
pub fn measure_color_bar_fill(
    image: &RgbaImage,
    fill: Rect,
    search: Rect,
    colour: ((f32, f32), f32, f32),
) -> Option<f32> {
    let (hue_range, min_saturation, min_value) = colour;
    let is_filled = |pixel: &Rgba<u8>| {
        crate::color::is_color_pixel(pixel, hue_range, min_saturation, min_value)
    };
    let groove = match sample_groove_color(image, fill, search, &is_filled) {
        Groove::Colour(colour) => Some(colour),
        Groove::Background => None,
        Groove::Unknown => return None,
    };
    if fill.w == 0 || fill.h == 0 {
        return None;
    }
    let (y_start, y_end) = bar_core_rows(image, fill)?;
    let width = image.width();
    let raw = image.as_raw();
    // Each block of 64 columns' kinds, once a column of it is asked for.
    let mut blocks: Vec<Option<[ColumnKind; 64]>> = vec![None; width.div_ceil(64) as usize];
    let mut mask = Vec::new();
    let mut classify = |x: u32| -> ColumnKind {
        if x >= width {
            return ColumnKind::Neither;
        }
        let block = (x / 64) as usize;
        if blocks[block].is_none() {
            let (x0, x1) = (64 * block as u32, (64 * block as u32 + 64).min(width));
            let n = (x1 - x0) as usize;
            let (mut filled, mut empty) = ([0u32; 64], [0u32; 64]);
            for y in y_start..y_end {
                let at = (y as usize * width as usize + x0 as usize) * 4;
                let row = &raw[at..at + 4 * n];
                crate::kernels::color_mask(row, hue_range, min_saturation, min_value, &mut mask);
                for (c, pixel) in row.as_chunks::<4>().0.iter().enumerate() {
                    if mask[0] & (1 << c) != 0 {
                        filled[c] += 1;
                    } else if let Some(reference) = groove
                        && is_similar_color(&Rgba(*pixel), reference, GROOVE_TOLERANCE)
                    {
                        empty[c] += 1;
                    }
                }
            }
            let rows = y_end - y_start;
            let mut kinds = [ColumnKind::Neither; 64];
            for c in 0..n {
                kinds[c] = column_kind(filled[c], empty[c], rows);
            }
            blocks[block] = Some(kinds);
        }
        blocks[block].expect("filled above")[(x % 64) as usize]
    };
    walk_bar(image, fill, search, &mut classify)
}

/// A column of a bar by its votes over `rows` rows: filled, empty or
/// neither. A real majority is required, so a stray matching row cannot
/// claim a column that is mostly background.
fn column_kind(filled: u32, empty: u32, rows: u32) -> ColumnKind {
    if filled * 2 > rows {
        ColumnKind::Filled
    } else if empty * 2 > rows {
        ColumnKind::Empty
    } else {
        ColumnKind::Neither
    }
}

/// The walk of [`measure_bar_fill`] out from a bar's fill, its columns
/// classified by `classify`: the fill's percentage of its track.
fn walk_bar(
    image: &RgbaImage,
    fill: Rect,
    search: Rect,
    mut classify: impl FnMut(u32) -> ColumnKind,
) -> Option<f32> {
    let search_left = search.x;
    let search_right = search.x.saturating_add(search.w).min(image.width());

    let mut filled_columns = 0u32;
    let mut track_columns = 0u32;

    // Walk outward from the filled span in both directions, stopping at the
    // edge of the bar's track.
    //
    // The stop condition tolerates a short run of ambiguous columns. Where
    // the fill meets the groove there is an antialiased boundary that is
    // neither colour, and treating the first such column as the end of the
    // track stops the walk before it ever reaches the groove — which read
    // as "always full" no matter how empty the bar was. Ambiguous columns
    // are held back and only counted once real track is found beyond them.
    let mut walk = |mut step: Box<dyn FnMut() -> Option<u32> + '_>| {
        let mut pending = 0u32;
        while let Some(x) = step() {
            match classify(x) {
                ColumnKind::Filled => {
                    filled_columns += 1 + pending;
                    track_columns += 1 + pending;
                    pending = 0;
                }
                ColumnKind::Empty => {
                    track_columns += 1 + pending;
                    pending = 0;
                }
                ColumnKind::Neither => {
                    pending += 1;
                    if pending > BOUNDARY_TOLERANCE {
                        break;
                    }
                }
            }
        }
    };

    let mut left = Some(fill.x);
    walk(Box::new(move || {
        let current = left?;
        // Column 0 ends the walk too, for a fill left of the search: the
        // walk went on there into columns past the image's width (by
        // wrapping), which are neither filled nor empty and were never
        // counted.
        left = if current == search_left || current == 0 {
            None
        } else {
            Some(current - 1)
        };
        Some(current)
    }));

    let mut right = fill.x.saturating_add(1);
    walk(Box::new(move || {
        if right >= search_right {
            return None;
        }
        let current = right;
        right += 1;
        Some(current)
    }));

    if track_columns == 0 {
        return None;
    }
    Some((filled_columns as f32 / track_columns as f32 * 100.0).clamp(0.0, 100.0))
}

/// How many consecutive ambiguous columns may sit inside a bar's track
/// before the walk concludes it has left the bar. Covers the antialiased
/// fill/groove boundary and thin separator pixels.
const BOUNDARY_TOLERANCE: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    Filled,
    Empty,
    Neither,
}

/// The rows to sample a bar along: its central band, not its full height.
///
/// A detected fill rectangle is taller than the bar's coloured core — it
/// picks up the border and a little surrounding UI. Those extra rows are
/// harmless over the fill itself, but past it they outvote the groove and
/// make the very first empty column look like "neither", ending the walk
/// immediately and reporting every bar as full. Sampling the middle half
/// keeps the vote on pixels that actually belong to the bar.
fn bar_core_rows(image: &RgbaImage, fill: Rect) -> Option<(u32, u32)> {
    let height = image.height();
    if fill.y >= height {
        return None;
    }
    let quarter = fill.h / 4;
    let start = fill.y.saturating_add(quarter).min(height.saturating_sub(1));
    let end = fill
        .y
        .saturating_add(fill.h.saturating_sub(quarter))
        .min(height);
    if end <= start {
        // Too thin to trim; use whatever single row is available.
        return Some((start, start.saturating_add(1).min(height)));
    }
    Some((start, end))
}

/// Per-channel tolerance when matching pixels against the sampled groove.
///
/// Wide enough to absorb the track's gradient and JPEG-ish artefacts in a
/// recorded frame, narrow enough that the bar's border and the surrounding
/// UI do not pass as groove.
const GROOVE_TOLERANCE: i32 = 38;

fn is_similar_color(pixel: &Rgba<u8>, reference: [u8; 3], tolerance: i32) -> bool {
    (0..3).all(|channel| (pixel[channel] as i32 - reference[channel] as i32).abs() <= tolerance)
}

/// Learn the groove colour from the columns immediately after the fill.
///
/// Returns `None` for a bar that is completely full, where there is no
/// groove to sample — the walk then measures only filled columns and
/// correctly reports 100%.
/// What lies just past a bar's fill.
enum Groove {
    /// The empty track, in this colour.
    Colour([u8; 3]),
    /// The world outside the bar: the bar is full.
    Background,
    /// Nothing to sample (the bar fills its box to the edge).
    Unknown,
}

fn sample_groove_color<F>(image: &RgbaImage, fill: Rect, search: Rect, is_filled: &F) -> Groove
where
    F: Fn(&Rgba<u8>) -> bool,
{
    let Some((y_start, y_end)) = bar_core_rows(image, fill) else {
        return Groove::Unknown;
    };
    let search_right = search.x.saturating_add(search.w).min(image.width());

    // Step just past the fill, skipping a couple of columns of antialiasing
    // at the fill's leading edge.
    let first = fill.x.saturating_add(fill.w).saturating_add(2);
    let last = first.saturating_add(6).min(search_right);

    let mut samples: Vec<[u8; 3]> = Vec::new();
    for x in first..last {
        for y in y_start..y_end {
            let pixel = image.get_pixel(x, y);
            if pixel[3] < 128 || is_filled(pixel) {
                continue;
            }
            samples.push([pixel[0], pixel[1], pixel[2]]);
        }
    }
    if samples.is_empty() {
        return Groove::Unknown;
    }

    // The median per channel resists the border pixels caught at the edges
    // of the sample window.
    let median = |channel: usize| {
        let mut values: Vec<u8> = samples.iter().map(|s| s[channel]).collect();
        values.sort_unstable();
        values[values.len() / 2]
    };
    let candidate = [median(0), median(1), median(2)];

    // Sampling past the fill is ambiguous: for a bar that is completely
    // full, what lies beyond the fill is not groove at all but the world
    // outside the bar. Distinguish them by looking just above the bar,
    // which is definitionally outside it — if the candidate matches that,
    // it is background and there is no groove, so the bar reads as full.
    if let Some(outside) = sample_outside_color(image, fill)
        && is_similar_color(
            &Rgba([candidate[0], candidate[1], candidate[2], 255]),
            outside,
            GROOVE_TOLERANCE,
        )
    {
        return Groove::Background;
    }
    Groove::Colour(candidate)
}

/// Sample the colour just above the bar, which is outside its track.
fn sample_outside_color(image: &RgbaImage, fill: Rect) -> Option<[u8; 3]> {
    // Step above the bar far enough to clear its border.
    let y = fill.y.checked_sub(3)?;
    let x_end = fill.x.saturating_add(fill.w).min(image.width());
    if fill.x >= x_end {
        return None;
    }

    let mut reds: Vec<u8> = Vec::new();
    let mut greens: Vec<u8> = Vec::new();
    let mut blues: Vec<u8> = Vec::new();
    for x in fill.x..x_end {
        let pixel = image.get_pixel(x, y);
        reds.push(pixel[0]);
        greens.push(pixel[1]);
        blues.push(pixel[2]);
    }
    if reds.is_empty() {
        return None;
    }
    let median = |mut values: Vec<u8>| {
        values.sort_unstable();
        values[values.len() / 2]
    };
    Some([median(reds), median(greens), median(blues)])
}

pub fn find_color_bar(
    image: &RgbaImage,
    region: Rect,
    hue_range: (f32, f32),
    min_saturation: f32,
    min_value: f32,
) -> Option<Rect> {
    // Runs of the colour on each row (each row's pixels tested together,
    // `color_runs`), grouped into regions: the largest is the bar.
    let min_width = (region.w / 20).max(8);
    let mut rows = Vec::new();
    let x1 = region.x.saturating_add(region.w).saturating_sub(1);
    let colour = (hue_range, min_saturation, min_value);
    let y1 = region.y.saturating_add(region.h).min(image.height());
    color_runs_rows(
        image,
        (region.y, y1),
        (region.x, x1),
        min_width,
        colour,
        &mut rows,
    );
    group_segments(rows, 2, 2)
        .into_iter()
        .max_by_key(|rect| rect.area())
}

thread_local! {
    /// The pixels `color_runs` samples along a row, kept between rows.
    static SAMPLES: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// [`segment_row`] with [`crate::color::is_color_pixel`] as the test: the
/// same runs, from pixels tested together into bits
/// (`kernels::color_mask`, in `mask`).
///
/// A run at least `min_width` long holds one of every `min_width`-th pixel
/// along the row, so only those are tested at first, and each that
/// matches is followed out to both ends of its run, 64 pixels a test; the
/// rest of the row is never looked at. (For runs of one or two pixels,
/// the whole row is tested.)
fn color_runs(
    image: &RgbaImage,
    y: u32,
    (x0, x1): (u32, u32),
    min_width: u32,
    (hue_range, min_saturation, min_value): ((f32, f32), f32, f32),
    mask: &mut Vec<u64>,
) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    if y >= image.height() || image.width() == 0 {
        return result;
    }
    let x1 = x1.min(image.width() - 1);
    if x0 > x1 {
        return result;
    }
    let width = image.width() as usize;
    let start = (y as usize * width + x0 as usize) * 4;
    let end = (y as usize * width + x1 as usize + 1) * 4;
    let row = &image.as_raw()[start..end];
    let count = (x1 - x0 + 1) as usize;
    let test = |from: usize, to: usize, mask: &mut Vec<u64>| {
        crate::kernels::color_mask(
            &row[4 * from..4 * to],
            hue_range,
            min_saturation,
            min_value,
            mask,
        );
    };
    let step = min_width as usize;
    if step <= 2 {
        test(0, count, mask);
        // From bit `i`, where the run of ones (or of zeros) it starts ends.
        let run_end = |i: usize, ones: bool| -> usize {
            let mut i = i;
            while i < count {
                let word = (if ones { !mask[i / 64] } else { mask[i / 64] }) >> (i % 64);
                if word != 0 {
                    return (i + word.trailing_zeros() as usize).min(count);
                }
                i = (i / 64 + 1) * 64;
            }
            count
        };
        let mut i = 0;
        while i < count {
            let begin = run_end(i, false);
            if begin >= count {
                break;
            }
            let end = run_end(begin, true);
            if (end - begin) as u32 >= min_width {
                result.push((x0 + begin as u32, x0 + end as u32 - 1));
            }
            i = end;
        }
        return result;
    }
    // The pixels `step - 1`, `2 step - 1`, … of the row, together.
    let hits: Vec<usize> = SAMPLES.with_borrow_mut(|samples| {
        samples.clear();
        for i in (step - 1..count).step_by(step) {
            samples.extend_from_slice(&row[4 * i..4 * i + 4]);
        }
        crate::kernels::color_mask(samples, hue_range, min_saturation, min_value, mask);
        let found = mask.iter().map(|w| w.count_ones() as usize).sum();
        let mut hits = Vec::with_capacity(found);
        for (w, &word) in mask.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                hits.push((64 * w + bits.trailing_zeros() as usize) * step + step - 1);
                bits &= bits - 1;
            }
        }
        hits
    });
    let colour = (hue_range, min_saturation, min_value);
    follow_runs(row, (x0, count), step, colour, hits, mask, |a, b| {
        result.push((a, b))
    });
    result
}

/// [`color_runs`] of rows `y0 .. y1`, into `out` as `(y, first, last)` in
/// row order: the same runs, with the samples of every row tested together
/// (rows whose samples all miss are done with then).
fn color_runs_rows(
    image: &RgbaImage,
    (y0, y1): (u32, u32),
    (x0, x1): (u32, u32),
    min_width: u32,
    colour: ((f32, f32), f32, f32),
    out: &mut Vec<(u32, u32, u32)>,
) {
    let mut mask = Vec::new();
    let y1 = y1.min(image.height());
    let step = min_width as usize;
    if image.width() == 0 || y0 >= y1 || step <= 2 || x0 > x1.min(image.width() - 1) {
        for y in y0..y1 {
            for (a, b) in color_runs(image, y, (x0, x1), min_width, colour, &mut mask) {
                out.push((y, a, b));
            }
        }
        return;
    }
    let x1 = x1.min(image.width() - 1);
    let width = image.width() as usize;
    let count = (x1 - x0 + 1) as usize;
    let row_of = |y: u32| {
        let start = (y as usize * width + x0 as usize) * 4;
        &image.as_raw()[start..start + 4 * count]
    };
    // Every row's samples, row after row, `per_row` each.
    let per_row = count / step;
    let found: Vec<u64> = SAMPLES.with_borrow_mut(|samples| {
        samples.clear();
        for y in y0..y1 {
            let row = row_of(y);
            for i in (step - 1..count).step_by(step) {
                samples.extend_from_slice(&row[4 * i..4 * i + 4]);
            }
        }
        let ((low, high), saturation, value) = colour;
        crate::kernels::color_mask(samples, (low, high), saturation, value, &mut mask);
        std::mem::take(&mut mask)
    });
    for (r, y) in (y0..y1).enumerate() {
        let (from, to) = (r * per_row, (r + 1) * per_row);
        let mut hits = Vec::new();
        let mut i = from;
        while i < to {
            let word = found[i / 64] >> (i % 64);
            if word == 0 {
                i = (i / 64 + 1) * 64;
                continue;
            }
            let k = i + word.trailing_zeros() as usize;
            if k >= to {
                break;
            }
            hits.push((k - from) * step + step - 1);
            i = k + 1;
        }
        if !hits.is_empty() {
            follow_runs(
                row_of(y),
                (x0, count),
                step,
                colour,
                hits,
                &mut mask,
                |a, b| out.push((y, a, b)),
            );
        }
    }
}

/// The rest of [`color_runs`]: each sample that matched (`hits`, pixel
/// indexes along `row` in increasing order) followed out to both ends of
/// its run, and each run at least `step` long given to `push` as its
/// first and last column (from `x0`).
fn follow_runs(
    row: &[u8],
    (x0, count): (u32, usize),
    step: usize,
    (hue_range, min_saturation, min_value): ((f32, f32), f32, f32),
    hits: Vec<usize>,
    mask: &mut Vec<u64>,
    mut push: impl FnMut(u32, u32),
) {
    let test = |from: usize, to: usize, mask: &mut Vec<u64>| {
        crate::kernels::color_mask(
            &row[4 * from..4 * to],
            hue_range,
            min_saturation,
            min_value,
            mask,
        );
    };
    // Where the run of matching pixels through `at` ends, to the right (one
    // past its last) and to the left (its first): stretches of 64 pixels
    // tested together, then twice as many each time while the run goes on.
    let first_miss = |mask: &[u64], n: usize| -> Option<usize> {
        (0..n.div_ceil(64)).find_map(|w| {
            let valid = (n - 64 * w).min(64);
            let misses = !mask[w]
                & if valid == 64 {
                    u64::MAX
                } else {
                    (1 << valid) - 1
                };
            (misses != 0).then(|| 64 * w + misses.trailing_zeros() as usize)
        })
    };
    let last_miss = |mask: &[u64], n: usize| -> Option<usize> {
        (0..n.div_ceil(64)).rev().find_map(|w| {
            let valid = (n - 64 * w).min(64);
            let misses = !mask[w]
                & if valid == 64 {
                    u64::MAX
                } else {
                    (1 << valid) - 1
                };
            (misses != 0).then(|| 64 * w + 63 - misses.leading_zeros() as usize)
        })
    };
    let run_right = |at: usize, mask: &mut Vec<u64>| -> usize {
        let (mut i, mut stretch) = (at + 1, 64);
        while i < count {
            let n = (count - i).min(stretch);
            test(i, i + n, mask);
            if let Some(miss) = first_miss(mask, n) {
                return i + miss;
            }
            i += n;
            stretch *= 2;
        }
        count
    };
    let mut done = 0;
    for at in hits {
        if at < done {
            continue; // within the run just found
        }
        let (mut begin, mut stretch) = (at, 64);
        while begin > 0 {
            let n = begin.min(stretch);
            test(begin - n, begin, mask);
            if let Some(miss) = last_miss(mask, n) {
                // One past the last miss.
                begin = begin - n + miss + 1;
                break;
            }
            begin -= n;
            stretch *= 2;
        }
        let end = run_right(at, mask);
        if end - begin >= step {
            push(x0 + begin as u32, x0 + end as u32 - 1);
        }
        done = end;
    }
}

pub fn find_color_regions(
    image: &RgbaImage,
    region: Rect,
    hue_range: (f32, f32),
    min_saturation: f32,
    min_value: f32,
) -> Vec<Rect> {
    let mut rows = Vec::new();
    let x1 = region.x.saturating_add(region.w).saturating_sub(1);
    let colour = (hue_range, min_saturation, min_value);
    let y1 = region.y.saturating_add(region.h).min(image.height());
    color_runs_rows(image, (region.y, y1), (region.x, x1), 20, colour, &mut rows);
    group_segments(rows, 4, 2)
        .into_iter()
        .filter(|rect| rect.w >= 20 && rect.h >= 4)
        .collect()
}

/// Quantize colors in `region` into `(r,g,b)/step` buckets and return the
/// bucket with the most pixels ("mode color"). Ignores near-transparent
/// pixels so alpha-blended overlays do not skew the histogram. Shared by
/// every detector that looks for a solid-color UI panel so they don't each
/// reimplement the same histogram.
///
/// Ties go to the lowest bucket in `(r, g, b)` order, so the answer is
/// deterministic (a hash map made it depend on iteration order). A `step`
/// of zero has no buckets and returns `None`.
pub fn dominant_color_bucket(image: &RgbaImage, region: Rect, step: u8) -> Option<(u8, u8, u8)> {
    if step == 0 {
        return None;
    }
    let x_end = region.x.saturating_add(region.w).min(image.width());
    let y_end = region.y.saturating_add(region.h).min(image.height());
    if region.x >= x_end || region.y >= y_end {
        return None;
    }

    // A dense histogram indexed by bucket: one increment per pixel, no
    // hashing. Buckets per channel is 256/step rounded up, so a step of 8
    // or more fits in at most 32^3 counters.
    let per_channel = 255 / step as usize + 1;

    if per_channel.pow(3) <= DENSE_HISTOGRAM_LIMIT {
        let size = per_channel.pow(3);
        // Each channel's share of the index from a table (a division a
        // channel a pixel otherwise), and, while the histogram is small,
        // the pixels counted into four histograms in turn and summed after:
        // a panel of one colour then does not wait on one counter. The same
        // counts.
        let weights = [(per_channel * per_channel) as u32, per_channel as u32, 1];
        let tables = weights.map(|weight| {
            std::array::from_fn::<u32, 256, _>(|v| (v as u32 / u32::from(step)) * weight)
        });
        let ways = if size <= 1 << 16 { 4 } else { 1 };
        let mut counts = vec![0u32; ways * size];
        let mut any = false;
        let width = image.width() as usize;
        for y in region.y..y_end {
            let at = (y as usize * width + region.x as usize) * 4;
            let row = &image.as_raw()[at..at + 4 * (x_end - region.x) as usize];
            let mut turn = 0usize;
            for p in row.as_chunks::<4>().0 {
                if p[3] >= 200 {
                    let i = (tables[0][p[0] as usize]
                        + tables[1][p[1] as usize]
                        + tables[2][p[2] as usize]) as usize;
                    counts[(turn % ways) * size + i] += 1;
                    turn += 1;
                }
            }
            any |= turn > 0;
        }
        if !any {
            return None;
        }
        let (counts, others) = counts.split_at_mut(size);
        for other in others.chunks_exact(size) {
            for (count, &more) in counts.iter_mut().zip(other) {
                *count += more;
            }
        }
        // First maximum in index order, i.e. the lowest bucket on ties.
        let (best, _) =
            counts.iter().enumerate().fold(
                (0, 0u32),
                |best, (i, &c)| if c > best.1 { (i, c) } else { best },
            );
        let b = best % per_channel;
        let g = (best / per_channel) % per_channel;
        let r = best / (per_channel * per_channel);
        return Some((r as u8, g as u8, b as u8));
    }

    // Very fine steps would need millions of counters; count sparsely.
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<(u8, u8, u8), u32> = BTreeMap::new();
    for y in region.y..y_end {
        for (_, pixel) in row_pixels(image, y, region.x, x_end - 1) {
            if pixel[3] >= 200 {
                *counts
                    .entry((pixel[0] / step, pixel[1] / step, pixel[2] / step))
                    .or_insert(0) += 1;
            }
        }
    }
    counts
        .into_iter()
        .fold(
            None,
            |best: Option<((u8, u8, u8), u32)>, (bucket, count)| match best {
                Some((_, top)) if top >= count => best,
                _ => Some((bucket, count)),
            },
        )
        .map(|(bucket, _)| bucket)
}

/// Largest dense histogram [`dominant_color_bucket`] will allocate
/// (a step of 3 or more stays under it).
const DENSE_HISTOGRAM_LIMIT: usize = 1 << 21;

/// Find the largest rectangle within `region` whose pixels quantize to
/// `bucket` (see [`dominant_color_bucket`]). Used to locate a solid-color
/// UI panel without hardcoding any specific skin's exact color.
pub fn find_uniform_color_panel(
    image: &RgbaImage,
    region: Rect,
    bucket: (u8, u8, u8),
    step: u8,
) -> Option<Rect> {
    if step == 0 {
        return None;
    }
    // Whether each channel's value falls in the bucket's, from a table: a
    // division a channel a pixel otherwise.
    let tables = [bucket.0, bucket.1, bucket.2]
        .map(|b| std::array::from_fn::<bool, 256, _>(|v| v as u8 / step == b));
    let matches = |pixel: &Rgba<u8>| {
        pixel[3] >= 200
            && tables[0][pixel[0] as usize]
            && tables[1][pixel[1] as usize]
            && tables[2][pixel[2] as usize]
    };

    let mut rows = Vec::new();
    let x_end = region.x.saturating_add(region.w).min(image.width());
    let y_end = region.y.saturating_add(region.h).min(image.height());
    for y in region.y..y_end {
        for (x0, x1) in segment_row(image, y, region.x, x_end.saturating_sub(1), 20, matches) {
            rows.push((y, x0, x1));
        }
    }

    group_segments(rows, 12, 2)
        .into_iter()
        .max_by_key(|rect| rect.area())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::is_color_pixel;

    /// The colour's runs found from bits are `segment_row`'s with the
    /// colour test, on rows of every length, with runs touching both ends.
    #[test]
    fn colour_runs_are_the_runs_segment_row_finds() {
        let mut state = 7u32;
        let image = RgbaImage::from_fn(300, 40, |x, y| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Long stretches of red, of blue, and of noise; every fifth
            // row red from end to end but for a few gaps.
            if y % 10 == 0 {
                return Rgba([220, 20, 30, 255]);
            }
            if y % 5 == 0 {
                return if x % 97 == 13 || x % 151 == 0 {
                    Rgba([20, 40, 230, 255])
                } else {
                    Rgba([220, 20, 30, 255])
                };
            }
            match (x / 23 + y) % 4 {
                0 => Rgba([220, 20, 30, 255]),
                1 => Rgba([20, 40, 230, 255]),
                2 => Rgba(state.to_le_bytes()),
                _ => Rgba([(state & 0xff) as u8 | 0xc0, 10, 10, 255]),
            }
        });
        let mut mask = Vec::new();
        for (hue, saturation, value) in [((340.0, 30.0), 0.35, 0.30), ((190.0, 250.0), 0.30, 0.30)]
        {
            for y in 0..40 {
                for (x0, x1) in [
                    (0, 299),
                    (0, 63),
                    (5, 70),
                    (64, 191),
                    (250, 400),
                    (299, 299),
                    (10, 9),
                ] {
                    for min_width in [1, 2, 3, 7, 8, 20, 23, 24, 64, 144] {
                        let expected = segment_row(&image, y, x0, x1, min_width, |p| {
                            is_color_pixel(p, hue, saturation, value)
                        });
                        let got = color_runs(
                            &image,
                            y,
                            (x0, x1),
                            min_width,
                            (hue, saturation, value),
                            &mut mask,
                        );
                        assert_eq!(got, expected, "row {y}, {x0}..={x1}, at least {min_width}");
                    }
                }
            }
            // Many rows at once: the same runs, row after row.
            for (y0, y1) in [(0, 40), (3, 17), (39, 40), (5, 5), (30, 90)] {
                for (x0, x1) in [(0, 299), (5, 70), (250, 400), (10, 9)] {
                    for min_width in [1, 3, 7, 8, 20, 23, 64, 144] {
                        let colour = (hue, saturation, value);
                        let mut expected = Vec::new();
                        for y in y0..y1.min(40) {
                            for (a, b) in
                                color_runs(&image, y, (x0, x1), min_width, colour, &mut mask)
                            {
                                expected.push((y, a, b));
                            }
                        }
                        let mut got = Vec::new();
                        color_runs_rows(&image, (y0, y1), (x0, x1), min_width, colour, &mut got);
                        assert_eq!(
                            got, expected,
                            "rows {y0}..{y1}, {x0}..={x1}, at least {min_width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn horizontal_edges_are_where_one_shade_meets_another() {
        // A dark scene with a light floor from row 60 down, and a short
        // light step from x 20 to 50 at row 30 that is too short to count.
        let mut image = RgbaImage::from_pixel(200, 100, Rgba([20, 20, 20, 255]));
        for y in 60..100 {
            for x in 0..200 {
                image.put_pixel(x, y, Rgba([200, 200, 200, 255]));
            }
        }
        for x in 20..50 {
            image.put_pixel(x, 30, Rgba([200, 200, 200, 255]));
        }
        let edges = horizontal_edges(&image, 40, 60);
        assert_eq!(edges.len(), 1, "{edges:?}");
        assert_eq!((edges[0].y, edges[0].x, edges[0].w), (59, 0, 200));
        // The short step counts once the run may be short: its top and
        // bottom edges, one above the other, grouped into one.
        let edges = horizontal_edges(&image, 40, 20);
        assert_eq!(edges.len(), 2, "{edges:?}");
        assert!(
            edges
                .iter()
                .any(|e| e.y == 29 && e.x == 20 && e.w == 30 && e.h == 2),
            "{edges:?}"
        );
        // Flat: nothing.
        assert!(
            horizontal_edges(&RgbaImage::from_pixel(50, 50, Rgba([9, 9, 9, 255])), 40, 8)
                .is_empty()
        );
        assert!(horizontal_edges(&RgbaImage::new(50, 1), 40, 8).is_empty());
    }
    use image::Rgba;

    /// Build a frame containing one horizontal bar whose track runs from
    /// `track_x` for `track_w` pixels, filled `fill_ratio` of the way.
    fn bar_frame(track_x: u32, track_w: u32, fill_ratio: f32) -> (RgbaImage, Rect) {
        let mut image = RgbaImage::from_pixel(400, 40, Rgba([120, 180, 90, 255]));
        let filled_w = (track_w as f32 * fill_ratio).round() as u32;
        let (y0, y1) = (14, 22);
        for y in y0..y1 {
            for x in track_x..track_x + track_w {
                let pixel = if x < track_x + filled_w {
                    Rgba([230, 30, 30, 255]) // bar colour
                } else {
                    Rgba([16, 16, 16, 255]) // dark groove
                };
                image.put_pixel(x, y, pixel);
            }
        }
        let fill = Rect {
            x: track_x,
            y: y0,
            w: filled_w.max(1),
            h: y1 - y0,
        };
        (image, fill)
    }

    fn red_filled(pixel: &Rgba<u8>) -> bool {
        is_color_pixel(pixel, (340.0, 30.0), 0.35, 0.30)
    }

    #[test]
    fn measure_bar_fill_reports_true_fill_ratio() {
        let search = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 40,
        };
        for ratio in [0.25f32, 0.5, 0.75, 1.0] {
            let (image, fill) = bar_frame(100, 200, ratio);
            let percent = measure_bar_fill(&image, fill, search, red_filled)
                .expect("track should be measurable");
            let expected = ratio * 100.0;
            assert!(
                (percent - expected).abs() <= 1.0,
                "fill {ratio} should read ~{expected}%, got {percent}%"
            );
        }
    }

    #[test]
    fn measure_bar_fill_is_independent_of_track_width_and_position() {
        // The old implementation divided by a fixed reference width, so the
        // same 50%-full bar read differently at different sizes.
        let search = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 40,
        };
        let (narrow_image, narrow_fill) = bar_frame(20, 80, 0.5);
        let (wide_image, wide_fill) = bar_frame(150, 240, 0.5);

        let narrow = measure_bar_fill(&narrow_image, narrow_fill, search, red_filled).unwrap();
        let wide = measure_bar_fill(&wide_image, wide_fill, search, red_filled).unwrap();

        assert!((narrow - 50.0).abs() <= 1.5, "narrow bar read {narrow}%");
        assert!((wide - 50.0).abs() <= 1.5, "wide bar read {wide}%");
    }

    #[test]
    fn measure_bar_fill_handles_a_light_groove_not_only_a_dark_one() {
        // Some UIs draw a light grey groove behind a bar instead of a dark
        // one. A fixed "empty is dark" rule reports such bars as permanently
        // 100% full; the groove colour is sampled instead.
        let mut image = RgbaImage::from_pixel(400, 40, Rgba([25, 25, 28, 255]));
        for y in 14..22 {
            for x in 100..300 {
                let pixel = if x < 100 + 75 {
                    Rgba([225, 225, 40, 255]) // yellow fill
                } else {
                    Rgba([210, 210, 210, 255]) // light grey groove
                };
                image.put_pixel(x, y, pixel);
            }
        }
        let fill = Rect {
            x: 100,
            y: 14,
            w: 75,
            h: 8,
        };
        let search = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 40,
        };
        let percent = measure_bar_fill(&image, fill, search, |pixel| {
            is_color_pixel(pixel, (40.0, 80.0), 0.25, 0.25)
        })
        .expect("light groove should still yield a track");
        assert!(
            (percent - 37.5).abs() <= 1.5,
            "75/200 of the track is filled, got {percent}%"
        );
    }

    #[test]
    fn a_completely_full_bar_reads_full_not_diluted_by_background() {
        // With no groove to sample, what lies past the fill is the world
        // outside the bar. Counting it as empty track would under-report a
        // full bar.
        let search = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 40,
        };
        let (image, fill) = bar_frame(100, 200, 1.0);
        let percent = measure_bar_fill(&image, fill, search, red_filled).unwrap();
        assert!((percent - 100.0).abs() <= 1.0, "full bar read {percent}%");
    }

    #[test]
    fn measure_bar_fill_returns_none_without_a_track() {
        // Uniform background: nothing is either filled or empty, so there is
        // no track to measure and the honest answer is "unknown".
        let image = RgbaImage::from_pixel(120, 30, Rgba([120, 180, 90, 255]));
        let fill = Rect {
            x: 40,
            y: 10,
            w: 10,
            h: 6,
        };
        let search = Rect {
            x: 0,
            y: 0,
            w: 120,
            h: 30,
        };
        assert!(measure_bar_fill(&image, fill, search, red_filled).is_none());
    }

    #[test]
    fn measure_bar_fill_handles_a_bar_touching_the_frame_edge() {
        let search = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 40,
        };
        // Track starts at x=0, so walking left immediately hits the bound.
        let (image, fill) = bar_frame(0, 120, 0.5);
        let percent = measure_bar_fill(&image, fill, search, red_filled).unwrap();
        assert!((percent - 50.0).abs() <= 1.5, "edge bar read {percent}%");
    }

    /// The colour bar's reading, its votes taken from row masks, is the
    /// general reading with the colour test, on bars of every kind: full,
    /// partial and empty, dark and light grooves, noisy edges, at the
    /// frame's edges, with searches cut short.
    #[test]
    fn a_colour_bar_reads_as_the_general_reading_with_the_colour_test() {
        let mut state = 99u32;
        let mut next = move |n: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % n
        };
        let colours = [
            ((340.0, 30.0), 0.35, 0.30),
            ((40.0, 80.0), 0.25, 0.25),
            ((190.0, 250.0), 0.30, 0.30),
        ];
        let fills = [[230, 30, 30, 255], [225, 225, 40, 255], [40, 90, 230, 255]];
        let (mut read, mut unread) = (0, 0);
        for trial in 0..300u32 {
            let k = (trial % 3) as usize;
            let (width, height) = (150 + next(400), 40);
            let mut image =
                RgbaImage::from_pixel(width, height, Rgba([25 + next(100) as u8, 25, 28, 255]));
            let track_x = next(width / 2);
            let track_w = (1 + next(width - track_x)).min(width - track_x);
            let filled_w = next(track_w + 1);
            let groove = if trial % 2 == 0 {
                [16u8, 16, 16, 255]
            } else {
                [210, 210, 210, 255]
            };
            let (y0, y1) = (10 + next(4), 20 + next(6));
            for y in y0..y1 {
                for x in track_x..track_x + track_w {
                    let mut pixel = if x < track_x + filled_w {
                        fills[k]
                    } else {
                        groove
                    };
                    // Noise: a stray pixel here and there, and soft edges.
                    if next(13) == 0 {
                        pixel = [next(256) as u8, next(256) as u8, next(256) as u8, 255];
                    }
                    image.put_pixel(x, y, Rgba(pixel));
                }
            }
            let fill = Rect {
                x: track_x,
                y: y0,
                w: filled_w.max(1),
                h: y1 - y0,
            };
            for search in [
                Rect {
                    x: 0,
                    y: 0,
                    w: width,
                    h: height,
                },
                Rect {
                    x: track_x.saturating_sub(3),
                    y: 0,
                    w: track_w / 2 + 5,
                    h: height,
                },
                Rect {
                    x: track_x + track_w / 3,
                    y: 0,
                    w: width,
                    h: height,
                },
            ] {
                let (hue, saturation, value) = colours[k];
                let general = measure_bar_fill(&image, fill, search, |p| {
                    is_color_pixel(p, hue, saturation, value)
                });
                let colour = measure_color_bar_fill(&image, fill, search, colours[k]);
                if general.is_some() {
                    read += 1;
                } else {
                    unread += 1;
                }
                assert_eq!(
                    colour.map(f32::to_bits),
                    general.map(f32::to_bits),
                    "trial {trial}: {fill:?} in {search:?}: {colour:?} vs {general:?}"
                );
            }
        }
        // Both outcomes were met, many times.
        assert!(read > 200 && unread > 50, "{read} read, {unread} not");
    }

    /// [`segment_row`] as it was first written: every pixel asked about.
    fn segment_row_one_by_one(
        image: &RgbaImage,
        y: u32,
        x0: u32,
        x1: u32,
        min_width: u32,
        predicate: impl Fn(&Rgba<u8>) -> bool,
    ) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        if y >= image.height() || image.width() == 0 {
            return result;
        }
        let x1 = x1.min(image.width() - 1);
        if x0 > x1 {
            return result;
        }
        let mut start = None;
        for (x, pixel) in row_pixels(image, y, x0, x1) {
            if predicate(pixel) {
                start.get_or_insert(x);
            } else if let Some(begin) = start.take()
                && x - begin >= min_width
            {
                result.push((begin, x - 1));
            }
        }
        if let Some(begin) = start
            && x1 + 1 - begin >= min_width
        {
            result.push((begin, x1));
        }
        result
    }

    /// [`dominant_color_bucket`] as it was first written: a division a
    /// channel a pixel, one histogram.
    fn dominant_one_by_one(image: &RgbaImage, region: Rect, step: u8) -> Option<(u8, u8, u8)> {
        if step == 0 {
            return None;
        }
        let x_end = region.x.saturating_add(region.w).min(image.width());
        let y_end = region.y.saturating_add(region.h).min(image.height());
        if region.x >= x_end || region.y >= y_end {
            return None;
        }
        let per_channel = 255 / step as usize + 1;
        let mut counts = vec![0u32; per_channel.pow(3)];
        let mut any = false;
        for y in region.y..y_end {
            for (_, p) in row_pixels(image, y, region.x, x_end - 1) {
                if p[3] >= 200 {
                    let i = ((p[0] / step) as usize * per_channel + (p[1] / step) as usize)
                        * per_channel
                        + (p[2] / step) as usize;
                    counts[i] += 1;
                    any = true;
                }
            }
        }
        if !any {
            return None;
        }
        let (best, _) =
            counts.iter().enumerate().fold(
                (0, 0u32),
                |best, (i, &c)| {
                    if c > best.1 { (i, c) } else { best }
                },
            );
        Some((
            (best / (per_channel * per_channel)) as u8,
            ((best / per_channel) % per_channel) as u8,
            (best % per_channel) as u8,
        ))
    }

    #[test]
    fn runs_from_samples_are_the_runs_of_every_pixel() {
        let mut state = 11u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        // Stretches of one colour of every length, broken by single odd
        // pixels and by noise.
        let image = RgbaImage::from_fn(420, 30, |x, y| {
            let r = next();
            if (x / (3 + y % 17)) % 3 == 0 || r % 31 == 0 {
                Rgba([200, 30, 30, 255])
            } else if r % 5 == 0 {
                Rgba(r.to_le_bytes())
            } else {
                Rgba([30, 30, 200, 255])
            }
        });
        let red = |p: &Rgba<u8>| p[0] > 150 && p[2] < 100;
        let blue = |p: &Rgba<u8>| p[2] > 150;
        for y in 0..31 {
            for (x0, x1) in [(0, 419), (0, 0), (3, 64), (100, 419), (400, 900), (9, 8)] {
                for min_width in [0, 1, 2, 3, 4, 7, 8, 20, 21, 64, 419, 500] {
                    for (name, test) in
                        [("red", &red as &dyn Fn(&Rgba<u8>) -> bool), ("blue", &blue)]
                    {
                        assert_eq!(
                            segment_row(&image, y, x0, x1, min_width, test),
                            segment_row_one_by_one(&image, y, x0, x1, min_width, test),
                            "{name} row {y}, {x0}..={x1}, at least {min_width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_dominant_bucket_from_tables_is_the_one_counted_by_division() {
        let mut state = 5u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..40u32 {
            let (w, h) = (1 + next() % 90, 1 + next() % 60);
            // A few colours in patches, noise, and some translucent pixels.
            let palette: Vec<[u8; 4]> = (0..4).map(|_| next().to_le_bytes()).collect();
            let image = RgbaImage::from_fn(w, h, |x, y| {
                let r = next();
                let mut p = if r % 7 == 0 {
                    r.to_le_bytes()
                } else {
                    palette[((x / 9 + y / 7) % 4) as usize]
                };
                p[3] = if r % 11 == 0 { (r >> 8) as u8 } else { 255 };
                Rgba(p)
            });
            for step in [0u8, 1, 2, 3, 7, 8, 10, 12, 16, 33, 128, 255] {
                for region in [
                    Rect { x: 0, y: 0, w, h },
                    Rect {
                        x: w / 3,
                        y: h / 4,
                        w: w / 2 + 1,
                        h: h / 2 + 1,
                    },
                    Rect {
                        x: w,
                        y: 0,
                        w: 5,
                        h: 5,
                    },
                ] {
                    assert_eq!(
                        dominant_color_bucket(&image, region, step),
                        dominant_one_by_one(&image, region, step),
                        "trial {trial}, step {step}, {region:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_uniform_panel_from_tables_and_samples_is_the_one_found_pixel_by_pixel() {
        let mut state = 17u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut found = 0;
        for trial in 0..30u32 {
            let (w, h) = (40 + next() % 200, 30 + next() % 120);
            let panel = [
                (next() % 200) as u8 + 20,
                (next() % 200) as u8 + 20,
                90,
                255,
            ];
            let (px, py) = (next() % (w / 2), next() % (h / 2));
            let (pw, ph) = (10 + next() % (w - px - 9), 5 + next() % (h - py - 4));
            let image = RgbaImage::from_fn(w, h, |x, y| {
                let r = next();
                if (px..px + pw).contains(&x) && (py..py + ph).contains(&y) && r % 23 != 0 {
                    // The panel, a level or two off here and there.
                    let d = if trial % 2 == 0 { 0 } else { (r % 3) as u8 };
                    Rgba([
                        panel[0] + d,
                        panel[1],
                        panel[2] - d,
                        if r % 41 == 0 { 100 } else { 255 },
                    ])
                } else {
                    Rgba(r.to_le_bytes())
                }
            });
            for step in [1u8, 2, 7, 10, 12, 64] {
                let bucket = (panel[0] / step, panel[1] / step, panel[2] / step);
                let region = Rect { x: 0, y: 0, w, h };
                let expected = {
                    let matches = |p: &Rgba<u8>| {
                        p[3] >= 200
                            && p[0] / step == bucket.0
                            && p[1] / step == bucket.1
                            && p[2] / step == bucket.2
                    };
                    let mut rows = Vec::new();
                    for y in 0..h {
                        for (a, b) in segment_row_one_by_one(&image, y, 0, w - 1, 20, matches) {
                            rows.push((y, a, b));
                        }
                    }
                    group_segments(rows, 12, 2)
                        .into_iter()
                        .max_by_key(|r| r.area())
                };
                found += usize::from(expected.is_some());
                assert_eq!(
                    find_uniform_color_panel(&image, region, bucket, step),
                    expected,
                    "trial {trial}, step {step}"
                );
            }
        }
        // Panels were there to find, most of the time.
        assert!(found > 40, "{found} found");
    }

    #[test]
    fn edges_read_eight_marks_at_a_time_are_the_edges_read_one_by_one() {
        // `horizontal_edges` as it was first written: each column compared
        // and followed on its own.
        let one_by_one = |image: &RgbaImage, threshold: u8, min_run: u32| {
            let (width, height) = image.dimensions();
            let luma = crate::threshold::channel_image(
                image,
                Rect {
                    x: 0,
                    y: 0,
                    w: width,
                    h: height,
                },
                crate::threshold::Channel::Luma,
            );
            let plane = luma.as_raw();
            let width = width as usize;
            let mut rows: Vec<(u32, u32, u32)> = Vec::new();
            for y in 0..height as usize - 1 {
                let (above, below) = (
                    &plane[y * width..(y + 1) * width],
                    &plane[(y + 1) * width..(y + 2) * width],
                );
                let mut start: Option<usize> = None;
                for (x, (&a, &b)) in above.iter().zip(below).enumerate() {
                    if a.abs_diff(b) >= threshold {
                        start.get_or_insert(x);
                    } else if let Some(begin) = start.take()
                        && (x - begin) as u32 >= min_run
                    {
                        rows.push((y as u32, begin as u32, x as u32 - 1));
                    }
                }
                if let Some(begin) = start
                    && (width - begin) as u32 >= min_run
                {
                    rows.push((y as u32, begin as u32, width as u32 - 1));
                }
            }
            group_segments(rows, 1, 0)
        };
        let mut state = 23u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for (w, h) in [
            (1u32, 2u32),
            (7, 5),
            (8, 3),
            (9, 9),
            (63, 20),
            (64, 8),
            (65, 11),
            (300, 40),
        ] {
            // Bands of shades with steps between them, broken in places,
            // and noise.
            let image = RgbaImage::from_fn(w, h, |x, y| {
                let r = next();
                let v = if r % 9 == 0 {
                    r as u8
                } else {
                    ((y / 3) * 40 + (x / 17) % 2 * 9) as u8
                };
                Rgba([v, v, v, 255])
            });
            for threshold in [0u8, 1, 8, 30, 255] {
                for min_run in [0u32, 1, 3, 8, 9, 64] {
                    assert_eq!(
                        horizontal_edges(&image, threshold, min_run),
                        one_by_one(&image, threshold, min_run),
                        "{w}x{h}, at least {threshold}, runs of {min_run}"
                    );
                }
            }
        }
    }

    #[test]
    fn segment_row_finds_runs_above_min_width() {
        let mut image = RgbaImage::from_pixel(40, 4, Rgba([0, 0, 0, 255]));
        for x in 5..20 {
            image.put_pixel(x, 1, Rgba([255, 0, 0, 255]));
        }
        let segments = segment_row(&image, 1, 0, 39, 10, |p| p[0] > 200);
        assert_eq!(segments, vec![(5, 19)]);
    }

    /// Regression: rows containing several separate segments (two regions
    /// side by side) used to be processed one segment at a time, so the
    /// side-by-side regions evicted each other from the active set and *no*
    /// rectangle ever reached `min_height`.
    #[test]
    fn group_segments_keeps_side_by_side_regions_apart() {
        let mut rows = Vec::new();
        for y in 10..20 {
            rows.push((y, 5, 24)); // left region
            rows.push((y, 40, 59)); // right region
        }
        let mut rects = group_segments(rows, 4, 1);
        rects.sort_by_key(|rect| rect.x);
        assert_eq!(rects.len(), 2, "expected two rectangles, got {rects:?}");
        assert_eq!((rects[0].x, rects[0].w, rects[0].h), (5, 20, 10));
        assert_eq!((rects[1].x, rects[1].w, rects[1].h), (40, 20, 10));
    }

    /// A region that momentarily splits into two segments on one row should
    /// continue as a single rectangle, not fork into duplicates.
    #[test]
    fn group_segments_absorbs_a_momentary_split() {
        let mut rows = Vec::new();
        for y in 0..4 {
            rows.push((y, 10, 39));
        }
        // One row where the middle pixel column is missing (antialiasing).
        rows.push((4, 10, 23));
        rows.push((4, 26, 39));
        for y in 5..8 {
            rows.push((y, 10, 39));
        }
        let rects = group_segments(rows, 4, 1);
        assert_eq!(rects.len(), 1, "split row forked the region: {rects:?}");
        assert_eq!((rects[0].x, rects[0].w, rects[0].h), (10, 30, 8));
    }

    #[test]
    fn segment_row_is_safe_outside_the_image() {
        let image = RgbaImage::from_pixel(10, 4, Rgba([255, 0, 0, 255]));
        assert!(
            segment_row(&image, 99, 0, 9, 1, |_| true).is_empty(),
            "row past the bottom"
        );
        assert!(
            segment_row(&image, 1, 8, 3, 1, |_| true).is_empty(),
            "x0 past x1"
        );
        assert_eq!(
            segment_row(&image, 1, 0, 500, 1, |_| true),
            vec![(0, 9)],
            "x1 clamped"
        );
    }

    #[test]
    fn dominant_color_bucket_is_deterministic_and_rejects_zero_step() {
        let mut image = RgbaImage::from_pixel(20, 10, Rgba([200, 10, 10, 255]));
        // Exactly half the region is blue: a tie that the answer must break
        // the same way every time (lowest bucket first).
        for y in 0..10 {
            for x in 10..20 {
                image.put_pixel(x, y, Rgba([10, 10, 200, 255]));
            }
        }
        let region = Rect {
            x: 0,
            y: 0,
            w: 20,
            h: 10,
        };
        for _ in 0..5 {
            assert_eq!(dominant_color_bucket(&image, region, 10), Some((1, 1, 20)));
        }
        assert_eq!(dominant_color_bucket(&image, region, 0), None);
        // A fine step takes the sparse path and must agree with the dense one.
        assert_eq!(
            dominant_color_bucket(&image, region, 1),
            Some((10, 10, 200))
        );
        let transparent = RgbaImage::new(8, 8);
        assert_eq!(
            dominant_color_bucket(
                &transparent,
                Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8
                },
                10
            ),
            None
        );
    }

    #[test]
    fn find_color_bar_locates_solid_block() {
        let mut image = RgbaImage::from_pixel(200, 40, Rgba([10, 10, 10, 255]));
        for y in 10..20 {
            for x in 20..160 {
                image.put_pixel(x, y, Rgba([220, 20, 20, 255]));
            }
        }
        let region = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 40,
        };
        let bar = find_color_bar(&image, region, (340.0, 30.0), 0.35, 0.30).expect("bar found");
        assert!(bar.w >= 100);
    }

    #[test]
    fn a_normalised_rect_round_trips_through_pixels() {
        let rect = Rect {
            x: 520,
            y: 652,
            w: 340,
            h: 10,
        };
        let norm = NormRect::from_rect(rect, 1280, 720);
        assert_eq!(norm.rect(1280, 720), rect);
        // Corners given backwards are put in order and clamped.
        let b = NormRect::new(1.2, 0.5, 0.25, -0.1);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0.25, 0.0, 1.0, 0.5));
        // Grown, united, placed within another.
        let g = NormRect::new(0.4, 0.4, 0.6, 0.6).grown(0.5, 0.0);
        assert!((g.x0 - 0.3).abs() < 1e-6 && (g.x1 - 0.7).abs() < 1e-6 && g.y0 == 0.4);
        let u = NormRect::new(0.0, 0.0, 0.2, 0.2).union(&NormRect::new(0.5, 0.5, 0.6, 0.9));
        assert_eq!((u.x0, u.y0, u.x1, u.y1), (0.0, 0.0, 0.6, 0.9));
        let inner = NormRect::new(0.5, 0.5, 1.0, 1.0);
        let w = NormRect::new(0.0, 0.0, 0.5, 0.5).within(&inner);
        assert_eq!((w.x0, w.y0, w.x1, w.y1), (0.5, 0.5, 0.75, 0.75));
        // Never smaller than a pixel.
        assert_eq!(
            NormRect::new(0.5, 0.5, 0.5, 0.5).pixels(100, 100),
            (50, 50, 1, 1)
        );
    }
}
