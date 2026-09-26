//! Reading short lines of pixel-font text by template matching.
//!
//! On-screen counters, HUD values and labels are drawn in one fixed font at
//! one of a handful of sizes. A general OCR engine is trained on documents
//! and misreads such text badly, and it costs a process launch per call.
//! This reader learns the font from labelled examples instead — show it a
//! crop and tell it what the crop says — and then reads new crops by
//! matching each glyph against what it learned:
//!
//! ```text
//!   crop ─▶ text evidence (brighter than its row's background)
//!        ─▶ text band (the rows the line occupies)
//!        ─▶ glyph columns (runs of inked columns; wide runs are split)
//!        ─▶ each glyph resized to a fixed cell ─▶ best template + margin
//! ```
//!
//! Every glyph reports its score and its margin over the best *other*
//! character. [`GlyphSet::read`] refuses to answer when any glyph is not
//! clearly one character, because a confident wrong number is worse than an
//! admitted "unknown"; [`GlyphSet::read_all`] returns the full breakdown
//! for callers that want to inspect uncertain readings themselves.
//!
//! Glyphs are normalised to the line height, so a font learned at one size
//! also reads the same font rendered at another.

use std::collections::HashMap;

use image::{GrayImage, RgbaImage};

use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::Rect;
use crate::threshold::{Channel, Polarity, text_evidence};

/// Height of the normalised glyph cell, in cells.
const CELL_H: usize = 16;
/// Width of the normalised glyph cell; wide enough for any glyph whose
/// width is at most its height.
const CELL_W: usize = 16;

/// Tuning for how text is separated from its background and how sure a
/// match must be.
#[derive(Debug, Clone, Copy)]
pub struct GlyphOptions {
    /// Which view of the pixel to measure. [`Channel::Min`] suits light text
    /// on coloured bars; [`Channel::Luma`] suits text on grey panels.
    pub channel: Channel,
    pub polarity: Polarity,
    /// Levels above the row background at which a stroke counts fully.
    pub evidence_span: u8,
    /// Evidence (0-255) at or above which a pixel is ink.
    pub ink_threshold: u8,
    /// Minimum normalised correlation for a glyph to count as recognised.
    pub min_score: f32,
    /// Minimum lead of the best character over the best other character.
    pub min_margin: f32,
}

impl Default for GlyphOptions {
    fn default() -> Self {
        Self {
            channel: Channel::Min,
            polarity: Polarity::LightText,
            evidence_span: 60,
            ink_threshold: 100,
            min_score: 0.72,
            min_margin: 0.06,
        }
    }
}

/// One recognised (or unrecognised) glyph.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphMatch {
    /// The best-matching character.
    pub ch: char,
    /// Where the glyph was found, in image coordinates.
    pub bounds: Rect,
    /// Normalised correlation with the best template, in `[-1, 1]`.
    pub score: f32,
    /// `score` minus the best score of any other character.
    pub margin: f32,
}

/// A line read glyph by glyph.
#[derive(Debug, Clone, PartialEq)]
pub struct TextReading {
    /// The characters in reading order; a space marks a wide gap.
    pub text: String,
    pub glyphs: Vec<GlyphMatch>,
}

/// Why a labelled example could not be learned from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearnError {
    /// No text-like pixels were found in the region.
    NoText,
    /// The crop splits into a different number of glyphs than the label
    /// has characters.
    GlyphCountMismatch { expected: usize, found: usize },
}

#[derive(Debug, Clone)]
struct Template {
    ch: char,
    /// Running mean of the normalised cells learned for this character.
    cell: Vec<f32>,
    /// `cell` centred and scaled to unit length, so that matching a glyph
    /// against it is a single dot product (see [`Standardised`]).
    unit: Standardised,
    samples: u32,
}

/// A font learned from labelled examples.
#[derive(Debug, Clone)]
pub struct GlyphSet {
    options: GlyphOptions,
    templates: Vec<Template>,
}

impl GlyphSet {
    pub fn new(options: GlyphOptions) -> Self {
        Self {
            options,
            templates: Vec::new(),
        }
    }

    pub fn options(&self) -> &GlyphOptions {
        &self.options
    }

    /// Characters learned so far.
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        self.templates.iter().map(|t| t.ch)
    }

    /// Learn from `region` of `image`, which shows exactly `label` (spaces
    /// in the label are ignored). Returns the number of glyphs learned.
    ///
    /// Repeated examples of a character are averaged, so learning from
    /// several captures makes the templates robust to the noise of any one.
    ///
    /// Glyphs are measured against the height of their line, which is what
    /// tells a dash from an underscore. Learn from whole lines as they
    /// appear on screen: a character that does not reach the full height
    /// (a slash, a dot, a dash) learned from a crop of its own is measured
    /// against its own height instead, and will not match in a real line.
    pub fn learn(
        &mut self,
        image: &RgbaImage,
        region: Rect,
        label: &str,
    ) -> Result<usize, LearnError> {
        let expected: Vec<char> = label.chars().filter(|c| !c.is_whitespace()).collect();
        let line = Line::extract(image, region, &self.options).ok_or(LearnError::NoText)?;
        let spans = fit_span_count(&line, line.spans(), expected.len());
        if spans.len() != expected.len() {
            return Err(LearnError::GlyphCountMismatch {
                expected: expected.len(),
                found: spans.len(),
            });
        }
        for (&(x0, x1), &ch) in spans.iter().zip(&expected) {
            let cell = line.cell(x0, x1);
            match self.templates.iter_mut().find(|t| t.ch == ch) {
                Some(template) => {
                    template.samples += 1;
                    let n = template.samples as f32;
                    for (mean, value) in template.cell.iter_mut().zip(&cell) {
                        *mean += (value - *mean) / n;
                    }
                    template.unit = Standardised::new(&template.cell);
                }
                None => self.templates.push(Template {
                    ch,
                    unit: Standardised::new(&cell),
                    cell,
                    samples: 1,
                }),
            }
        }
        Ok(expected.len())
    }

    /// Read `region`, glyph by glyph, whatever the confidence.
    pub fn read_all(&self, image: &RgbaImage, region: Rect) -> Option<TextReading> {
        if self.templates.is_empty() {
            return None;
        }
        let line = Line::extract(image, region, &self.options)?;
        let mut glyphs = Vec::new();
        let mut text = String::new();
        let mut previous_end: Option<u32> = None;
        let space_gap = (line.height as f32 * SPACE_GAP_FRACTION).ceil() as u32;

        for (x0, x1) in line.spans() {
            // A run that matches no single glyph well may be two or three
            // glyphs touching; take whichever split reads best.
            let pieces = self.best_split(&line, x0, x1);
            for (a, b) in pieces {
                if let Some(end) = previous_end
                    && a.saturating_sub(end) >= space_gap
                {
                    text.push(' ');
                }
                let found = self.classify(&line.cell(a, b));
                text.push(found.0);
                glyphs.push(GlyphMatch {
                    ch: found.0,
                    bounds: Rect {
                        x: line.origin.0 + a,
                        y: line.origin.1 + line.top,
                        w: b - a,
                        h: line.height,
                    },
                    score: found.1,
                    margin: found.2,
                });
                previous_end = Some(b);
            }
        }
        (!glyphs.is_empty()).then_some(TextReading { text, glyphs })
    }

    /// Read `region`, or explain why it could not be read with confidence.
    ///
    /// The confidence of a successful reading is that of its least certain
    /// glyph. Any glyph below [`GlyphOptions::min_score`] or
    /// [`GlyphOptions::min_margin`] makes the whole reading missing: one
    /// wrong digit makes a number wrong.
    pub fn read(&self, image: &RgbaImage, region: Rect) -> Detection<TextReading> {
        let Some(reading) = self.read_all(image, region) else {
            return Detection::missing("glyphs", "no glyphs found in the region");
        };
        let weakest =
            reading.glyphs.iter().enumerate().find(|(_, g)| {
                g.score < self.options.min_score || g.margin < self.options.min_margin
            });
        if let Some((index, glyph)) = weakest {
            return Detection::missing(
                "glyphs",
                format!(
                    "glyph {index} is not clearly one character (best {:?}, score {:.2}, margin {:.2}); read {:?}",
                    glyph.ch, glyph.score, glyph.margin, reading.text
                ),
            );
        }
        let confidence = reading
            .glyphs
            .iter()
            .map(|g| {
                let score = (g.score - self.options.min_score) / (1.0 - self.options.min_score);
                let margin = g.margin / (2.0 * self.options.min_margin);
                (0.5 + 0.5 * score.min(margin)).clamp(0.0, 1.0)
            })
            .fold(1.0f32, f32::min);
        Detection::found(
            reading,
            Confidence::new(confidence),
            "glyphs",
            Reliability::Heuristic,
        )
    }

    /// Best character for a normalised cell: `(char, score, margin)`.
    fn classify(&self, cell: &[f32]) -> (char, f32, f32) {
        let cell = Standardised::new(cell);
        let mut best = (' ', -1.0f32);
        let mut second = -1.0f32;
        for template in &self.templates {
            let score = cell.correlation(&template.unit);
            if score > best.1 {
                if best.0 != template.ch {
                    second = second.max(best.1);
                }
                best = (template.ch, score);
            } else if template.ch != best.0 {
                second = second.max(score);
            }
        }
        (best.0, best.1, best.1 - second.max(-1.0))
    }

    /// Split `x0..x1` into one, two or three glyphs, whichever makes the
    /// weakest piece score highest. A single glyph is kept unless a split
    /// is clearly better, so a well-matched wide glyph is never cut up.
    ///
    /// Cut positions are chosen by the templates, not by the ink profile:
    /// where two glyphs touch, the least-inked column is usually *inside* a
    /// glyph (the waist of an 8), and cutting there reads neither half.
    ///
    /// The search only considers pieces a glyph could fill — at most
    /// [`MAX_GLYPH_ASPECT`] times the line height wide — and scores each
    /// distinct piece once, so a wide smear of unreadable ink costs about as
    /// much as a few glyphs rather than growing with the square of its width.
    fn best_split(&self, line: &Line, x0: u32, x1: u32) -> Vec<(u32, u32)> {
        let whole = self.classify(&line.cell(x0, x1));
        if whole.1 >= self.options.min_score || x1 - x0 < 4 {
            return vec![(x0, x1)];
        }
        let widest = ((line.height as f32 * MAX_GLYPH_ASPECT).ceil() as u32).max(2);
        let mut scores: HashMap<(u32, u32), Option<ScoredPiece>> = HashMap::new();
        let mut score = |a: u32, b: u32| -> Option<ScoredPiece> {
            if b - a > widest {
                return None;
            }
            *scores.entry((a, b)).or_insert_with(|| {
                let (a, b) = line.trim_columns(a, b)?;
                Some(((a, b), self.classify(&line.cell(a, b)).1))
            })
        };
        let mut best = (vec![(x0, x1)], whole.1);

        // A split is only believed when its weakest piece beats the best so
        // far by SPLIT_ADVANTAGE, so any piece scoring below that bar rules
        // out every split containing it without scoring the rest.
        let bar = |best: f32| best + SPLIT_ADVANTAGE;

        // Two pieces, each at least two columns wide.
        if x1 - x0 <= 2 * widest {
            for cut in (x0 + 2).max(x1.saturating_sub(widest))..=(x1 - 2).min(x0 + widest) {
                let Some((p, sp)) = score(x0, cut) else {
                    continue;
                };
                if sp <= bar(best.1) {
                    continue;
                }
                if let Some((q, sq)) = score(cut, x1)
                    && sp.min(sq) > bar(best.1)
                {
                    best = (vec![p, q], sp.min(sq));
                }
            }
        }
        if best.1 >= self.options.min_score || x1 - x0 < 6 || x1 - x0 > 3 * widest {
            return best.0;
        }

        // Three pieces, only when two did not already explain the run.
        for first in x0 + 2..=(x1 - 4).min(x0 + widest) {
            let Some((p, sp)) = score(x0, first) else {
                continue;
            };
            for second in (first + 2).max(x1.saturating_sub(widest))..=(x1 - 2).min(first + widest)
            {
                if sp <= bar(best.1) {
                    break;
                }
                let Some((q, sq)) = score(first, second) else {
                    continue;
                };
                if sq <= bar(best.1) {
                    continue;
                }
                if let Some((r, sr)) = score(second, x1)
                    && sp.min(sq).min(sr) > bar(best.1)
                {
                    best = (vec![p, q, r], sp.min(sq).min(sr));
                }
            }
        }
        best.0
    }
}

/// A candidate glyph after trimming — its columns `(x0, x1)` — and how well
/// it matches its best character.
type ScoredPiece = ((u32, u32), f32);

/// Widest glyph a split may produce, as a multiple of the line height.
/// Glyph cells are square, so wider pieces would be squeezed to fit anyway.
const MAX_GLYPH_ASPECT: f32 = 1.5;

/// A gap of at least this fraction of the line height reads as a space.
const SPACE_GAP_FRACTION: f32 = 0.5;

/// How much better the weakest piece of a split must score than the
/// unsplit run before the split is believed.
const SPLIT_ADVANTAGE: f32 = 0.05;

/// The text band of a region: evidence rows the line occupies, and which
/// columns hold ink.
struct Line {
    origin: (u32, u32),
    /// Evidence for the whole region, 0-255.
    evidence: GrayImage,
    top: u32,
    height: u32,
    ink_threshold: u8,
}

impl Line {
    fn extract(image: &RgbaImage, region: Rect, options: &GlyphOptions) -> Option<Self> {
        let evidence = text_evidence(
            image,
            region,
            options.channel,
            options.polarity,
            options.evidence_span,
        );
        let (width, height) = evidence.dimensions();
        if width == 0 || height == 0 {
            return None;
        }
        let threshold = options.ink_threshold;
        // Rows carrying ink; the line is the tallest run of them, allowing a
        // one-row gap (the dot of an i, the gap in a colon).
        let inked: Vec<bool> = (0..height)
            .map(|y| (0..width).any(|x| evidence.get_pixel(x, y).0[0] >= threshold))
            .collect();
        let mut best: Option<(u32, u32)> = None;
        let mut y = 0;
        while y < height {
            if !inked[y as usize] {
                y += 1;
                continue;
            }
            let start = y;
            let mut end = y + 1;
            while end < height
                && (inked[end as usize] || (end + 1 < height && inked[end as usize + 1]))
            {
                end += 1;
            }
            if best.is_none_or(|(s, e)| end - start > e - s) {
                best = Some((start, end));
            }
            y = end;
        }
        let (top, bottom) = best?;
        Some(Self {
            origin: (region.x, region.y),
            evidence,
            top,
            height: bottom - top,
            ink_threshold: threshold,
        })
    }

    fn is_ink(&self, x: u32, y: u32) -> bool {
        self.evidence.get_pixel(x, y).0[0] >= self.ink_threshold
    }

    /// Ink pixels per column over `x0..x1`, within the band.
    fn column_ink(&self, x0: u32, x1: u32) -> Vec<u32> {
        (x0..x1)
            .map(|x| {
                (self.top..self.top + self.height)
                    .filter(|&y| self.is_ink(x, y))
                    .count() as u32
            })
            .collect()
    }

    /// Runs of inked columns within the band, as half-open `(x0, x1)`.
    fn spans(&self) -> Vec<(u32, u32)> {
        let ink = self.column_ink(0, self.evidence.width());
        let mut spans = Vec::new();
        let mut start = None;
        for (x, &count) in ink.iter().enumerate() {
            match (count > 0, start) {
                (true, None) => start = Some(x as u32),
                (false, Some(s)) => {
                    spans.push((s, x as u32));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            spans.push((s, ink.len() as u32));
        }
        self.main_cluster(spans, &ink)
    }

    /// Keep only the line itself: the group of runs with the most ink,
    /// where groups are separated by gaps wider than twice the line height.
    ///
    /// A region drawn around a value usually also catches the ends of the
    /// bar or panel it sits on — short bright slivers well away from the
    /// text. Word spacing is far narrower than that gap, so the text stays
    /// together while the debris at the edges is dropped.
    fn main_cluster(&self, spans: Vec<(u32, u32)>, ink: &[u32]) -> Vec<(u32, u32)> {
        let limit = self.height.saturating_mul(2).max(4);
        let mut groups: Vec<Vec<(u32, u32)>> = Vec::new();
        for span in spans {
            match groups.last_mut() {
                Some(group) if span.0 - group.last().expect("non-empty").1 <= limit => {
                    group.push(span)
                }
                _ => groups.push(vec![span]),
            }
        }
        let weight = |group: &Vec<(u32, u32)>| -> u32 {
            group
                .iter()
                .map(|&(a, b)| ink[a as usize..b as usize].iter().sum::<u32>())
                .sum()
        };
        groups
            .into_iter()
            .max_by_key(|g| weight(g))
            .unwrap_or_default()
    }

    /// Shrink `a..b` to its inked columns, or `None` if it holds no ink.
    fn trim_columns(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let ink = self.column_ink(a, b);
        let first = ink.iter().position(|&c| c > 0)? as u32;
        let last = ink.iter().rposition(|&c| c > 0)? as u32;
        Some((a + first, a + last + 1))
    }

    /// The glyph in columns `x0..x1`, scaled so the band fills the cell's
    /// height (keeping its aspect), centred horizontally. Values are the
    /// evidence in `[0, 1]`, sampled with box filtering: each cell pixel is
    /// the area-weighted mean of the source pixels it covers.
    ///
    /// The box filter is separable, so it runs as a horizontal pass over
    /// the band's rows and then a vertical pass over the result.
    fn cell(&self, x0: u32, x1: u32) -> Vec<f32> {
        let mut cell = vec![0.0f32; CELL_W * CELL_H];
        let glyph_w = x1 - x0;
        let scale = CELL_H as f32 / self.height as f32;
        let out_w = (glyph_w as f32 * scale).round().clamp(1.0, CELL_W as f32) as usize;
        let left = (CELL_W - out_w) / 2;
        let available = self.evidence.width().saturating_sub(x0);
        let columns = BoxTaps::new(glyph_w, out_w, available);
        let rows = BoxTaps::new(self.height, CELL_H, self.height);

        let evidence = self.evidence.as_raw();
        let stride = self.evidence.width() as usize;
        let mut horizontal = vec![0.0f32; self.height as usize * out_w];
        for sy in 0..self.height as usize {
            let row = (self.top as usize + sy) * stride + x0 as usize;
            for cx in 0..out_w {
                horizontal[sy * out_w + cx] = columns
                    .of(cx)
                    .iter()
                    .map(|&(sx, w)| w * f32::from(evidence[row + sx]))
                    .sum();
            }
        }
        for cy in 0..CELL_H {
            for cx in 0..out_w {
                let weight = rows.weight(cy) * columns.weight(cx);
                if weight <= 0.0 {
                    continue;
                }
                let total: f32 = rows
                    .of(cy)
                    .iter()
                    .map(|&(sy, w)| w * horizontal[sy * out_w + cx])
                    .sum();
                cell[cy * CELL_W + left + cx] = total / weight / 255.0;
            }
        }
        cell
    }
}

/// Box-filter taps resampling `len` source pixels to `out` pixels: for each
/// output pixel, the source pixels its span covers and how much of each,
/// skipping source pixels at or beyond `limit`. Stored flat, one
/// allocation for all output pixels.
struct BoxTaps {
    /// `taps[starts[i]..starts[i + 1]]` belong to output pixel `i`.
    starts: Vec<usize>,
    taps: Vec<(usize, f32)>,
    weights: Vec<f32>,
}

impl BoxTaps {
    fn new(len: u32, out: usize, limit: u32) -> Self {
        let step = len as f32 / out as f32;
        let mut starts = Vec::with_capacity(out + 1);
        let mut taps = Vec::with_capacity(out * (step.ceil() as usize + 1));
        let mut weights = Vec::with_capacity(out);
        for i in 0..out {
            starts.push(taps.len());
            let (start, end) = (i as f32 * step, (i + 1) as f32 * step);
            let mut total = 0.0;
            let mut source = start.floor() as u32;
            while (source as f32) < end && source < limit {
                let weight = end.min(source as f32 + 1.0) - start.max(source as f32);
                if weight > 0.0 {
                    taps.push((source as usize, weight));
                    total += weight;
                }
                source += 1;
            }
            weights.push(total);
        }
        starts.push(taps.len());
        Self {
            starts,
            taps,
            weights,
        }
    }

    fn of(&self, i: usize) -> &[(usize, f32)] {
        &self.taps[self.starts[i]..self.starts[i + 1]]
    }

    /// Total weight of output pixel `i`'s taps.
    fn weight(&self, i: usize) -> f32 {
        self.weights[i]
    }
}

/// Pick `parts - 1` cut columns in a run's ink profile: each near its
/// share of the width, at the least inked column within a small window.
fn cut_points(ink: &[u32], parts: u32) -> Vec<u32> {
    let width = ink.len() as u32;
    let window = (width / (parts * 3)).max(1);
    (1..parts)
        .map(|k| {
            let target = k * width / parts;
            let lo = target.saturating_sub(window).max(1);
            let hi = (target + window).min(width - 1);
            (lo..=hi)
                .min_by_key(|&c| (ink[c as usize], c.abs_diff(target)))
                .unwrap_or(target)
        })
        .collect()
}

/// When a labelled example has touching characters, split the widest runs
/// until there is one run per character. Extra runs are left alone: gluing
/// runs together to fit a label would teach a template made of two glyphs,
/// so the caller reports the mismatch instead.
fn fit_span_count(line: &Line, mut spans: Vec<(u32, u32)>, count: usize) -> Vec<(u32, u32)> {
    while spans.len() < count && !spans.is_empty() {
        let (i, &(a, b)) = spans
            .iter()
            .enumerate()
            .max_by_key(|(_, (a, b))| b - a)
            .expect("at least one span");
        if b - a < 2 {
            break;
        }
        let cut = a + cut_points(&line.column_ink(a, b), 2)[0];
        let pieces: Vec<(u32, u32)> = [(a, cut), (cut, b)]
            .into_iter()
            .filter_map(|(p, q)| line.trim_columns(p, q))
            .collect();
        if pieces.len() != 2 {
            break;
        }
        spans.splice(i..=i, pieces);
    }
    spans
}

/// A cell centred on its mean and scaled to unit length. The Pearson
/// correlation of two cells is then the dot product of their standardised
/// forms, so each template is standardised once when learned and each
/// glyph once when read, instead of for every comparison.
#[derive(Debug, Clone)]
struct Standardised(Vec<f32>);

impl Standardised {
    fn new(cell: &[f32]) -> Self {
        let n = cell.len().max(1) as f32;
        let mean = cell.iter().sum::<f32>() / n;
        let norm = cell
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            .sqrt();
        // A flat cell has no shape to correlate; all zeros makes every
        // correlation with it 0.
        let scale = if norm > 1e-6 { 1.0 / norm } else { 0.0 };
        Self(cell.iter().map(|v| (v - mean) * scale).collect())
    }

    /// Pearson correlation with another standardised cell of the same size.
    fn correlation(&self, other: &Standardised) -> f32 {
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{self, GLYPH_WIDTH};
    use image::Rgba;

    const BAR: Rgba<u8> = Rgba([228, 70, 104, 255]);
    const INK: Rgba<u8> = Rgba([250, 238, 240, 255]);

    /// Render `text` with the built-in bitmap font onto a coloured bar.
    fn render(text: &str, scale: u32, spacing: Option<u32>) -> (RgbaImage, Rect) {
        let advance = spacing.map(|s| GLYPH_WIDTH * scale + s);
        let width = match advance {
            Some(a) => a * text.chars().count() as u32,
            None => draw::text_width(text, scale),
        } + 24;
        let height = draw::text_height(scale) + 12;
        let mut image = RgbaImage::from_pixel(width, height, BAR);
        let mut plot = |x: i64, y: i64| {
            if x >= 0 && y >= 0 && (x as u32) < width && (y as u32) < height {
                image.put_pixel(x as u32, y as u32, INK);
            }
        };
        match advance {
            None => draw::draw_text(text, 12, 6, scale, &mut plot),
            Some(a) => {
                for (i, ch) in text.chars().enumerate() {
                    draw::draw_text(
                        &ch.to_string(),
                        12 + (i as u32 * a) as i64,
                        6,
                        scale,
                        &mut plot,
                    );
                }
            }
        }
        let region = Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        (image, region)
    }

    fn digits_font(scale: u32) -> GlyphSet {
        let mut set = GlyphSet::new(GlyphOptions::default());
        let (image, region) = render("0123456789/", scale, None);
        assert_eq!(set.learn(&image, region, "0123456789/"), Ok(11));
        set
    }

    fn read_text(set: &GlyphSet, text: &str, scale: u32) -> Detection<TextReading> {
        let (image, region) = render(text, scale, None);
        set.read(&image, region)
    }

    #[test]
    fn reads_back_numbers_in_the_learned_font() {
        let set = digits_font(2);
        for value in ["65132/71867", "25896/29906", "0/1", "400/408", "9876543210"] {
            let reading = read_text(&set, value, 2);
            let text = reading.value.as_ref().map(|r| r.text.as_str());
            assert_eq!(text, Some(value), "{:?}", reading.failure_reason);
            assert!(reading.confidence.value() > 0.5);
        }
    }

    #[test]
    fn a_font_learned_at_one_size_reads_other_sizes() {
        let set = digits_font(2);
        for scale in [1, 3, 4] {
            let reading = read_text(&set, "71867/71867", scale);
            assert_eq!(
                reading.value.map(|r| r.text),
                Some("71867/71867".to_string()),
                "scale {scale}: {:?}",
                reading.failure_reason
            );
        }
    }

    #[test]
    fn wide_gaps_read_as_spaces() {
        let set = digits_font(2);
        let (image, region) = render("12 / 34", 2, None);
        let reading = set.read(&image, region).value.expect("readable");
        assert_eq!(reading.text, "12 / 34");
        assert_eq!(reading.glyphs.len(), 5, "spaces are not glyphs");
    }

    #[test]
    fn touching_glyphs_are_split() {
        let set = digits_font(2);
        // No spacing at all: adjacent digits share columns of ink.
        let (image, region) = render("188", 2, Some(0));
        let reading = set.read(&image, region);
        assert_eq!(
            reading.value.map(|r| r.text),
            Some("188".to_string()),
            "{:?}",
            reading.failure_reason
        );
    }

    #[test]
    fn noise_does_not_change_the_reading() {
        let set = digits_font(2);
        let (mut image, region) = render("29456/29906", 2, None);
        let mut state = 7u32;
        for pixel in image.pixels_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let jitter = (state >> 24) as i32 % 31 - 15;
            for c in 0..3 {
                pixel[c] = (pixel[c] as i32 + jitter).clamp(0, 255) as u8;
            }
        }
        let reading = set.read(&image, region);
        assert_eq!(
            reading.value.map(|r| r.text),
            Some("29456/29906".to_string()),
            "{:?}",
            reading.failure_reason
        );
    }

    /// A character the set never learned must not be passed off as the
    /// nearest digit.
    #[test]
    fn unknown_characters_make_the_reading_missing() {
        let set = digits_font(2);
        let reading = read_text(&set, "12A4", 2);
        assert!(!reading.is_present());
        let reason = reading.failure_reason.unwrap();
        assert!(reason.contains("glyph 2"), "{reason}");
        // The breakdown is still available for inspection.
        let (image, region) = render("12A4", 2, None);
        let all = set.read_all(&image, region).unwrap();
        assert_eq!(all.glyphs.len(), 4);
        assert!(
            all.glyphs[2].score < set.options().min_score
                || all.glyphs[2].margin < set.options().min_margin
        );
    }

    #[test]
    fn learning_requires_the_label_to_match_the_glyphs() {
        let mut set = GlyphSet::new(GlyphOptions::default());
        let (image, region) = render("123", 2, None);
        // Two labels for three separate glyphs: gluing two of them into one
        // template would poison the set, so the mismatch is reported.
        assert_eq!(
            set.learn(&image, region, "12"),
            Err(LearnError::GlyphCountMismatch {
                expected: 2,
                found: 3
            })
        );
        assert_eq!(
            set.chars().count(),
            0,
            "nothing was learned from the bad example"
        );
        let blank = RgbaImage::from_pixel(40, 20, BAR);
        assert_eq!(
            set.learn(
                &blank,
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 20
                },
                "1"
            ),
            Err(LearnError::NoText)
        );
        assert!(
            set.read_all(
                &blank,
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 20
                }
            )
            .is_none()
        );
    }

    #[test]
    fn dark_text_on_a_light_panel_is_read_with_dark_polarity() {
        let options = GlyphOptions {
            channel: Channel::Luma,
            polarity: Polarity::DarkText,
            ..GlyphOptions::default()
        };
        let mut set = GlyphSet::new(options);
        let paper = Rgba([235, 232, 220, 255]);
        let ink = Rgba([30, 30, 40, 255]);
        let draw_on = |text: &str| {
            let width = draw::text_width(text, 2) + 20;
            let mut image = RgbaImage::from_pixel(width, 30, paper);
            draw::draw_text(text, 10, 8, 2, |x, y| {
                image.put_pixel(x as u32, y as u32, ink)
            });
            (
                image,
                Rect {
                    x: 0,
                    y: 0,
                    w: width,
                    h: 30,
                },
            )
        };
        let (image, region) = draw_on("0123456789");
        set.learn(&image, region, "0123456789").unwrap();
        let (image, region) = draw_on("2026");
        assert_eq!(
            set.read(&image, region).value.map(|r| r.text),
            Some("2026".to_string())
        );
    }
}
