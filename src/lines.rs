//! Reading a line of text whose glyphs are not pixel-exact.
//!
//! [`crate::glyphs`] cuts a line at its blank columns and matches each piece
//! against the font it learned. That is right for a HUD captured from the
//! screen, where every frame draws the same pixels. A video of that HUD, a
//! stream, or a HUD the game stretches by a fraction blurs neighbouring
//! glyphs into each other and breaks single glyphs apart, and the cutting
//! fails: on the EXP lines of a compressed MapleStory video, a digit falls
//! into two pieces two or three columns wide, and the glyph reader learned
//! from 6 of 2,300 labelled lines and refused every line it was shown. This
//! reader cuts nothing:
//!
//! ```text
//!   line ─▶ 15 rows, grey: each colour less its mean, over the line's spread
//!        ─▶ the cheapest cover of its columns by the band and by character
//!           pictures (dynamic programming over the columns: touching
//!           glyphs need no cut, a glyph in two pieces is one picture)
//!        ─▶ each placed glyph read again by linear templates over the
//!           window round it (a digit stays a digit, a mark a mark)
//!        ─▶ the text, and the smallest margin of any glyph
//! ```
//!
//! [`LineFont::learn`] makes the pictures and templates from labelled lines,
//! with no model: the lines whose ink falls into as many runs as their label
//! has characters start the pictures; every line is then placed against its
//! own label by the same dynamic programming, and the pictures made again,
//! until they settle. The templates are a logistic regression on the windows
//! round the placed glyphs, and on the same windows a column to either side.
//! Every fifth line is held out to fit the margin under which a reading is
//! refused. It is the explicit reader of the extraction pilot
//! (`bench/extract`): on that video it read 859 of 872 frames right, where a
//! 93,857-parameter network read 734.

use image::RgbaImage;

use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::Rect;

/// The rows a line is resampled to: pictures and templates are learned, and
/// lines read, at this height.
pub const ROWS: usize = 15;

/// How the font is learned and lines are read.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineOptions {
    /// What placing one glyph costs over its distance from its picture:
    /// fewer, surer glyphs over a scatter of small ones.
    pub penalty: f32,
    /// Columns either side of a glyph's centre its templates read.
    pub window: usize,
    /// Columns either way the templates also learn each glyph moved by: a
    /// glyph placed as a look-alike sits a column off from where its own
    /// picture would put it, and is still read.
    pub jitter: usize,
    /// Rounds of placing and averaging when learning.
    pub rounds: usize,
    /// Windows the templates learn from, at most (an even sample of all).
    pub max_windows: usize,
    /// Steps of the templates' training, and its rate.
    pub steps: usize,
    pub learning_rate: f32,
    /// Weight decay of the templates' training.
    pub decay: f32,
    /// A column is ink when its distance from the band is over this share
    /// of the line's most inked column's.
    pub ink: f32,
    /// Of the held-out lines a reading is given for, the share that may be
    /// wrong at the refusal level.
    pub max_wrong: f32,
}

impl Default for LineOptions {
    fn default() -> Self {
        Self {
            penalty: 2.0,
            window: 4,
            jitter: 1,
            rounds: 6,
            max_windows: 40_000,
            steps: 150,
            learning_rate: 0.1,
            decay: 1e-3,
            ink: 0.2,
            max_wrong: 0.01,
        }
    }
}

/// Why a font could not be learned.
#[derive(Debug, Clone, PartialEq)]
pub enum LineLearnError {
    /// Fewer labelled lines than learning needs.
    TooFew { needed: usize, given: usize },
    /// No line's ink fell into as many runs as its label has characters,
    /// so nothing could start the pictures.
    NoStart,
    /// These characters were never placed, so they have no picture.
    Unplaced(Vec<char>),
}

impl std::fmt::Display for LineLearnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LineLearnError::TooFew { needed, given } => {
                write!(f, "{given} labelled lines, {needed} needed")
            }
            LineLearnError::NoStart => write!(
                f,
                "no line's ink fell into as many runs as its label has characters"
            ),
            LineLearnError::Unplaced(chars) => {
                write!(f, "never placed, so no picture: {chars:?}")
            }
        }
    }
}

impl std::error::Error for LineLearnError {}

/// A labelled line to learn from: `region` of `image` shows `label`
/// (whitespace in the label is ignored).
#[derive(Debug, Clone, Copy)]
pub struct LabelledLine<'a> {
    pub image: &'a RgbaImage,
    pub region: Rect,
    pub label: &'a str,
}

/// One glyph of a reading.
#[derive(Debug, Clone, PartialEq)]
pub struct LineGlyph {
    pub ch: char,
    /// Where it was placed, in image coordinates.
    pub bounds: Rect,
    /// Its best template's score less the next best of its kind.
    pub margin: f32,
}

/// A line read.
#[derive(Debug, Clone, PartialEq)]
pub struct LineReading {
    pub text: String,
    pub glyphs: Vec<LineGlyph>,
    /// The smallest margin of any glyph: the reading's confidence.
    pub margin: f32,
}

/// The fewest labelled lines [`LineFont::learn`] takes.
pub const MIN_LINES: usize = 10;

/// From this many labelled lines on, every fifth is held out to fit the
/// refusal level; with fewer, nothing is refused on the margin.
const HOLD_OUT_FROM: usize = 20;

/// A font learned for reading lines whole: a picture and a width for each
/// character, the band's colour, the reading templates, and the margin
/// under which a reading is refused.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineFont {
    options: LineOptions,
    chars: Vec<char>,
    widths: Vec<usize>,
    /// `ROWS` x width, row-major, one per character.
    pictures: Vec<Vec<f32>>,
    band: [f32; ROWS],
    /// (`ROWS` * (2 * window + 1) + 1) x characters, row-major.
    weights: Vec<f32>,
    level: f32,
    /// What a reading must be to be given (the caller's format), if anything.
    #[cfg_attr(feature = "serde", serde(skip))]
    accept: Option<fn(&str) -> bool>,
}

/// A line as the reader sees it: `ROWS` rows of `w` columns, grey and
/// normalised; `x0`, `y0`, `sx` and `sy` map its columns and rows back to
/// the image.
#[derive(Debug, Clone)]
struct Grey {
    w: usize,
    px: Vec<f32>,
    x0: u32,
    y0: u32,
    sx: f32,
    sy: f32,
}

impl LineFont {
    /// Learn a font from labelled lines (see the module's description).
    /// Every character in the labels must be placed at least once.
    pub fn learn(lines: &[LabelledLine], options: LineOptions) -> Result<LineFont, LineLearnError> {
        Self::learn_inner(lines, options, None)
    }

    /// [`LineFont::learn`], for lines whose text has a known form (a
    /// number, `HP[a/b]`): a reading `accept` turns down is refused, and the
    /// refusal level is fitted on the held-out readings it accepts. A glyph
    /// one column wide that the cover leaves out leaves every other glyph
    /// sure, so the margin cannot tell that reading from a right one; the
    /// form usually can (on the pilot's EXP lines, it turned down 83% of
    /// the wrong readings).
    pub fn learn_checked(
        lines: &[LabelledLine],
        options: LineOptions,
        accept: fn(&str) -> bool,
    ) -> Result<LineFont, LineLearnError> {
        Self::learn_inner(lines, options, Some(accept))
    }

    /// What a reading must be to be given, as after
    /// [`LineFont::learn_checked`] (a font read back from storage has none).
    pub fn set_accept(&mut self, accept: Option<fn(&str) -> bool>) {
        self.accept = accept;
    }

    fn learn_inner(
        lines: &[LabelledLine],
        options: LineOptions,
        accept: Option<fn(&str) -> bool>,
    ) -> Result<LineFont, LineLearnError> {
        if lines.len() < MIN_LINES {
            return Err(LineLearnError::TooFew {
                needed: MIN_LINES,
                given: lines.len(),
            });
        }
        let mut chars: Vec<char> = Vec::new();
        let mut greys = Vec::new();
        let mut labels: Vec<Vec<usize>> = Vec::new();
        for line in lines {
            let Some(g) = grey(line.image, line.region) else {
                continue;
            };
            let label: Vec<usize> = line
                .label
                .chars()
                .filter(|c| !c.is_whitespace())
                .map(|c| match chars.iter().position(|&k| k == c) {
                    Some(k) => k,
                    None => {
                        chars.push(c);
                        chars.len() - 1
                    }
                })
                .collect();
            greys.push(g);
            labels.push(label);
        }
        let held_out = |i: usize| greys.len() >= HOLD_OUT_FROM && i % 5 == 4;
        let learn: Vec<usize> = (0..greys.len()).filter(|&i| !held_out(i)).collect();
        let band = band(learn.iter().map(|&i| &greys[i]));

        // The start: lines whose ink falls into as many runs as their label
        // has characters, cut there; each character's width, the most
        // common width of its pieces.
        let mut placed: Vec<Vec<(usize, usize)>> = vec![Vec::new(); greys.len()];
        let mut seen: Vec<Vec<usize>> = vec![Vec::new(); chars.len()];
        for &i in &learn {
            let runs = ink_runs(&greys[i], &band, options.ink);
            if runs.len() == labels[i].len() && !runs.is_empty() {
                placed[i] = labels[i]
                    .iter()
                    .zip(&runs)
                    .map(|(&k, &(a, _))| (k, a))
                    .collect();
                for (&k, &(_, w)) in labels[i].iter().zip(&runs) {
                    seen[k].push(w);
                }
            }
        }
        if placed.iter().all(Vec::is_empty) {
            return Err(LineLearnError::NoStart);
        }
        let unplaced: Vec<char> = (0..chars.len())
            .filter(|&k| seen[k].is_empty())
            .map(|k| chars[k])
            .collect();
        if !unplaced.is_empty() {
            return Err(LineLearnError::Unplaced(unplaced));
        }
        let widths: Vec<usize> = seen.iter().map(|w| mode(w)).collect();

        // Pictures from the placements, the placements again from the
        // pictures (every line against its own label), until nothing moves.
        let mut font = LineFont {
            options,
            chars,
            widths,
            pictures: Vec::new(),
            band,
            weights: Vec::new(),
            level: 0.0,
            accept,
        };
        for _ in 0..options.rounds {
            font.pictures = font.mean_pictures(&greys, &placed);
            let mut moved = 0;
            for &i in &learn {
                let label = &labels[i];
                if label.iter().any(|&k| font.pictures[k].is_empty()) {
                    continue;
                }
                let out = font.cover(&greys[i], Some(label));
                if out.len() != label.len() {
                    continue;
                }
                if placed[i].is_empty() || placed[i].iter().zip(&out).any(|(a, b)| a.1 != b.1) {
                    moved += 1;
                }
                placed[i] = out;
            }
            if moved == 0 {
                break;
            }
        }
        font.pictures = font.mean_pictures(&greys, &placed);
        let unplaced: Vec<char> = (0..font.chars.len())
            .filter(|&k| font.pictures[k].is_empty())
            .map(|k| font.chars[k])
            .collect();
        if !unplaced.is_empty() {
            return Err(LineLearnError::Unplaced(unplaced));
        }
        font.weights = font.train_templates(&greys, &placed);

        // The refusal level, on the lines held out (the readings the form
        // turns down are refused before any level).
        let found: Vec<(f32, bool)> = (0..greys.len())
            .filter(|&i| held_out(i))
            .filter_map(|i| {
                let reading = font.read_grey(&greys[i])?;
                if accept.is_some_and(|accept| !accept(&reading.text)) {
                    return None;
                }
                let right = reading
                    .text
                    .chars()
                    .eq(labels[i].iter().map(|&k| font.chars[k]));
                Some((reading.margin, right))
            })
            .collect();
        font.level = if found.is_empty() {
            0.0
        } else {
            fit_level(&found, options.max_wrong)
        };
        Ok(font)
    }

    pub fn options(&self) -> &LineOptions {
        &self.options
    }

    /// The characters it knows.
    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// The margin under which [`LineFont::read`] refuses.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Read `region`, whatever the confidence.
    pub fn read_all(&self, image: &RgbaImage, region: Rect) -> Option<LineReading> {
        self.read_grey(&grey(image, region)?)
    }

    /// Read `region`, or say why not: no glyph in it, a reading the form
    /// turns down ([`LineFont::learn_checked`]), or one whose smallest
    /// margin is under the level fitted when learning.
    pub fn read(&self, image: &RgbaImage, region: Rect) -> Detection<LineReading> {
        let Some(reading) = self.read_all(image, region) else {
            return Detection::missing("lines", "no glyphs found in the region");
        };
        if self.accept.is_some_and(|accept| !accept(&reading.text)) {
            return Detection::missing(
                "lines",
                format!(
                    "read {:?}, which is not of the form the line's text has",
                    reading.text
                ),
            );
        }
        if reading.margin < self.level {
            return Detection::missing(
                "lines",
                format!(
                    "read {:?}, but its least sure glyph leads by {:.2}, under {:.2}",
                    reading.text, reading.margin, self.level
                ),
            );
        }
        let sure = (reading.margin - self.level) / (reading.margin.abs() + 1.0);
        let confidence = (0.5 + 0.5 * sure).clamp(0.5, 1.0);
        Detection::found(
            reading,
            Confidence::new(confidence),
            "lines",
            Reliability::Heuristic,
        )
    }

    fn read_grey(&self, g: &Grey) -> Option<LineReading> {
        let placed = self.cover(g, None);
        if placed.is_empty() {
            return None;
        }
        let hw = self.options.window as isize;
        let span = 2 * self.options.window + 1;
        let nc = self.chars.len();
        let mut text = String::new();
        let mut glyphs = Vec::with_capacity(placed.len());
        let mut least = f32::INFINITY;
        let mut scores = vec![0f32; nc];
        for &(k, x) in &placed {
            let centre = (x + self.widths[k] / 2) as isize;
            scores.copy_from_slice(&self.weights[ROWS * span * nc..][..nc]);
            for r in 0..ROWS {
                for j in 0..span {
                    let col = centre - hw + j as isize;
                    if col < 0 || col >= g.w as isize {
                        continue;
                    }
                    let v = g.px[r * g.w + col as usize];
                    let row = &self.weights[(r * span + j) * nc..][..nc];
                    for (s, &wv) in scores.iter_mut().zip(row) {
                        *s += v * wv;
                    }
                }
            }
            // a digit stays a digit, anything else stays not a digit
            let digit = self.chars[k].is_ascii_digit();
            let (mut first, mut second) = (usize::MAX, usize::MAX);
            for c in (0..nc).filter(|&c| self.chars[c].is_ascii_digit() == digit) {
                if first == usize::MAX || scores[c] > scores[first] {
                    second = first;
                    first = c;
                } else if second == usize::MAX || scores[c] > scores[second] {
                    second = c;
                }
            }
            let margin = if second == usize::MAX {
                f32::INFINITY
            } else {
                scores[first] - scores[second]
            };
            least = least.min(margin);
            text.push(self.chars[first]);
            glyphs.push(LineGlyph {
                ch: self.chars[first],
                bounds: Rect {
                    x: g.x0 + (x as f32 * g.sx) as u32,
                    y: g.y0,
                    w: ((self.widths[k] as f32 * g.sx).round() as u32).max(1),
                    h: ((ROWS as f32 * g.sy).round() as u32).max(1),
                },
                margin,
            });
        }
        Some(LineReading {
            text,
            glyphs,
            margin: least,
        })
    }

    /// The cheapest cover of the line's columns by the band and the
    /// characters' pictures: [(character, first column)]. With `label`, the
    /// cover must place exactly those characters in that order.
    fn cover(&self, g: &Grey, label: Option<&[usize]>) -> Vec<(usize, usize)> {
        let w = g.w;
        let mut bg = vec![0f32; w];
        for r in 0..ROWS {
            for (c, b) in bg.iter_mut().enumerate() {
                let d = g.px[r * w + c] - self.band[r];
                *b += d * d;
            }
        }
        // each picture against every stretch of the line
        let cost: Vec<Vec<f32>> = self
            .pictures
            .iter()
            .zip(&self.widths)
            .map(|(pic, &pw)| {
                if pic.is_empty() || pw > w {
                    return Vec::new();
                }
                let n = w - pw + 1;
                let mut cost = vec![0f32; n];
                for r in 0..ROWS {
                    for j in 0..pw {
                        let t = pic[r * pw + j];
                        let row = &g.px[r * w + j..][..n];
                        for (c, &v) in cost.iter_mut().zip(row) {
                            let d = v - t;
                            *c += d * d;
                        }
                    }
                }
                cost
            })
            .collect();
        let penalty = self.options.penalty;
        match label {
            None => {
                // best[x]: the cheapest cover of columns 0..x
                let mut best = vec![0f32; w + 1];
                let mut back = vec![(usize::MAX, 0usize); w + 1];
                for x in 1..=w {
                    let mut v = best[x - 1] + bg[x - 1];
                    let mut arg = (usize::MAX, x - 1);
                    for (k, c) in cost.iter().enumerate() {
                        let pw = self.widths[k];
                        if c.is_empty() || x < pw {
                            continue;
                        }
                        let s = x - pw;
                        let total = best[s] + c[s] + penalty;
                        if total < v {
                            v = total;
                            arg = (k, s);
                        }
                    }
                    best[x] = v;
                    back[x] = arg;
                }
                let mut out = Vec::new();
                let mut x = w;
                while x > 0 {
                    let (k, s) = back[x];
                    if k != usize::MAX {
                        out.push((k, s));
                    }
                    x = s;
                }
                out.reverse();
                out
            }
            Some(label) => {
                // best[i][x]: the first i characters placed within columns 0..x
                let n = label.len();
                let stride = w + 1;
                let mut best = vec![f32::INFINITY; (n + 1) * stride];
                let mut back = vec![(0usize, 0usize); (n + 1) * stride];
                best[0] = 0.0;
                for x in 1..=w {
                    for i in 0..=n {
                        let mut v = best[i * stride + x - 1] + bg[x - 1];
                        let mut arg = (i, x - 1);
                        if i > 0 {
                            let k = label[i - 1];
                            let pw = self.widths[k];
                            if !cost[k].is_empty() && x >= pw {
                                let s = x - pw;
                                let total = best[(i - 1) * stride + s] + cost[k][s] + penalty;
                                if total < v {
                                    v = total;
                                    arg = (i - 1, s);
                                }
                            }
                        }
                        best[i * stride + x] = v;
                        back[i * stride + x] = arg;
                    }
                }
                if !best[n * stride + w].is_finite() {
                    return Vec::new();
                }
                let mut out = Vec::new();
                let (mut i, mut x) = (n, w);
                while x > 0 {
                    let (pi, s) = back[i * stride + x];
                    if pi + 1 == i {
                        out.push((label[i - 1], s));
                    }
                    i = pi;
                    x = s;
                }
                out.reverse();
                out
            }
        }
    }

    /// Each character's picture: the mean of the line at its placements,
    /// at its width (empty for one never placed).
    fn mean_pictures(&self, greys: &[Grey], placed: &[Vec<(usize, usize)>]) -> Vec<Vec<f32>> {
        let nc = self.chars.len();
        let mut sums: Vec<Vec<f32>> = self.widths.iter().map(|&w| vec![0f32; ROWS * w]).collect();
        let mut counts = vec![0usize; nc];
        for (g, pl) in greys.iter().zip(placed) {
            for &(k, x) in pl {
                let pw = self.widths[k];
                if x + pw > g.w {
                    continue;
                }
                counts[k] += 1;
                for r in 0..ROWS {
                    for j in 0..pw {
                        sums[k][r * pw + j] += g.px[r * g.w + x + j];
                    }
                }
            }
        }
        sums.into_iter()
            .zip(counts)
            .map(|(s, n)| {
                if n == 0 {
                    Vec::new()
                } else {
                    s.into_iter().map(|v| v / n as f32).collect()
                }
            })
            .collect()
    }

    /// The reading templates: a logistic regression on the windows round
    /// the placed glyphs' centres and those moved by up to `jitter`
    /// columns, an even sample of at most `max_windows`, trained with Adam.
    fn train_templates(&self, greys: &[Grey], placed: &[Vec<(usize, usize)>]) -> Vec<f32> {
        let o = &self.options;
        let hw = o.window as isize;
        let span = 2 * o.window + 1;
        let d = ROWS * span + 1;
        let nc = self.chars.len();
        let mut spots: Vec<(usize, isize, usize)> = Vec::new(); // line, centre, class
        for (i, pl) in placed.iter().enumerate() {
            for &(k, x) in pl {
                let centre = (x + self.widths[k] / 2) as isize;
                for dx in -(o.jitter as isize)..=o.jitter as isize {
                    let c = centre + dx;
                    if c >= 0 && c < greys[i].w as isize {
                        spots.push((i, c, k));
                    }
                }
            }
        }
        if spots.len() > o.max_windows {
            let step = spots.len() as f64 / o.max_windows as f64;
            spots = (0..o.max_windows)
                .map(|j| spots[(j as f64 * step) as usize])
                .collect();
        }
        let n = spots.len();
        let mut xs = vec![0f32; n * d];
        let ys: Vec<usize> = spots.iter().map(|s| s.2).collect();
        for (row, &(i, centre, _)) in xs.chunks_exact_mut(d).zip(&spots) {
            let g = &greys[i];
            for r in 0..ROWS {
                for j in 0..span {
                    let col = centre - hw + j as isize;
                    if col >= 0 && col < g.w as isize {
                        row[r * span + j] = g.px[r * g.w + col as usize];
                    }
                }
            }
            row[d - 1] = 1.0;
        }
        let mut w = vec![0f32; d * nc];
        let (mut m, mut v) = (vec![0f32; d * nc], vec![0f32; d * nc]);
        let mut grad = vec![0f32; d * nc];
        let mut z = vec![0f32; nc];
        for step in 1..=o.steps {
            grad.fill(0.0);
            for (x, &y) in xs.chunks_exact(d).zip(&ys) {
                z.fill(0.0);
                for (j, &xv) in x.iter().enumerate() {
                    if xv == 0.0 {
                        continue;
                    }
                    let wr = &w[j * nc..][..nc];
                    for (zc, &wc) in z.iter_mut().zip(wr) {
                        *zc += xv * wc;
                    }
                }
                let top = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for zc in z.iter_mut() {
                    *zc = (*zc - top).exp();
                    sum += *zc;
                }
                for zc in z.iter_mut() {
                    *zc /= sum;
                }
                z[y] -= 1.0;
                for (j, &xv) in x.iter().enumerate() {
                    if xv == 0.0 {
                        continue;
                    }
                    let gr = &mut grad[j * nc..][..nc];
                    for (g, &p) in gr.iter_mut().zip(&z) {
                        *g += xv * p;
                    }
                }
            }
            let t = step as i32;
            let (b1, b2) = (1.0 - 0.9f32.powi(t), 1.0 - 0.999f32.powi(t));
            for ((wv, g), (mv, vv)) in w.iter_mut().zip(&grad).zip(m.iter_mut().zip(v.iter_mut())) {
                let g = g / n as f32 + o.decay * *wv;
                *mv = 0.9 * *mv + 0.1 * g;
                *vv = 0.999 * *vv + 0.001 * g * g;
                *wv -= o.learning_rate * (*mv / b1) / ((*vv / b2).sqrt() + 1e-8);
            }
        }
        w
    }
}

/// The region as the reader sees it (see [`Grey`]), or None for an empty one.
fn grey(image: &RgbaImage, region: Rect) -> Option<Grey> {
    let (iw, ih) = image.dimensions();
    let (x0, y0) = (region.x.min(iw), region.y.min(ih));
    let (x1, y1) = (
        region.x.saturating_add(region.w).min(iw),
        region.y.saturating_add(region.h).min(ih),
    );
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let raw = image.as_raw();
    let stride = iw as usize * 4;
    let rows =
        || (0..h).map(move |y| &raw[(y0 as usize + y) * stride + x0 as usize * 4..][..w * 4]);
    if h == ROWS {
        // At the reader's height: sums of each colour and of its squares
        // (integers), then the grey line straight from the pixels.
        let (mut sum, mut squares) = ([0u64; 3], [0u64; 3]);
        for row in rows() {
            for p in row.chunks_exact(4) {
                for c in 0..3 {
                    let v = u64::from(p[c]);
                    sum[c] += v;
                    squares[c] += v * v;
                }
            }
        }
        let n = (w * h) as f64;
        let mut var = 0f64;
        for c in 0..3 {
            let m = sum[c] as f64 / n;
            var += squares[c] as f64 / n - m * m;
        }
        let spread = (var.max(0.0) / 3.0).sqrt() / 255.0;
        let scale = (1.0 / (spread + 0.02) / (3.0 * 255.0)) as f32;
        let offset = ((sum[0] + sum[1] + sum[2]) as f64 / n) as f32;
        let mut px = Vec::with_capacity(w * h);
        for row in rows() {
            px.extend(
                row.chunks_exact(4).map(|p| {
                    (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2]) - offset) * scale
                }),
            );
        }
        return Some(Grey {
            w,
            px,
            x0,
            y0,
            sx: 1.0,
            sy: 1.0,
        });
    }
    // Any other height: each colour resampled to the reader's height (the
    // width in proportion), then the same.
    let mut planes = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
    for (y, row) in rows().enumerate() {
        for (x, p) in row.chunks_exact(4).enumerate() {
            for c in 0..3 {
                planes[c][y * w + x] = p[c] as f32 / 255.0;
            }
        }
    }
    let gw = ((w as f32 * ROWS as f32 / h as f32).round() as usize).max(1);
    let planes = planes.map(|p| resize(&p, w, h, gw, ROWS));
    let n = (gw * ROWS) as f64;
    let means: Vec<f64> = planes
        .iter()
        .map(|p| p.iter().map(|&v| v as f64).sum::<f64>() / n)
        .collect();
    let mut var = 0f64;
    for (p, m) in planes.iter().zip(&means) {
        for &v in p {
            let d = v as f64 - m;
            var += d * d;
        }
    }
    let scale = 1.0 / ((var / (3.0 * n)).sqrt() + 0.02);
    let mean = (means[0] + means[1] + means[2]) / 3.0;
    let px = (0..gw * ROWS)
        .map(|i| {
            (((planes[0][i] + planes[1][i] + planes[2][i]) as f64 / 3.0 - mean) * scale) as f32
        })
        .collect();
    Some(Grey {
        w: gw,
        px,
        x0,
        y0,
        sx: w as f32 / gw as f32,
        sy: h as f32 / ROWS as f32,
    })
}

/// A plane resampled with the triangle filter, widened when shrinking (as
/// `image`'s `FilterType::Triangle`).
fn resize(src: &[f32], w: usize, h: usize, w2: usize, h2: usize) -> Vec<f32> {
    let taps = |from: usize, to: usize| -> Vec<(usize, Vec<f32>)> {
        let ratio = from as f32 / to as f32;
        let support = ratio.max(1.0);
        (0..to)
            .map(|o| {
                let centre = (o as f32 + 0.5) * ratio;
                let left = ((centre - support).floor().max(0.0)) as usize;
                let right = ((centre + support).ceil() as usize).min(from).max(left + 1);
                let mut ws: Vec<f32> = (left..right)
                    .map(|i| {
                        let t = (i as f32 + 0.5 - centre).abs() / support;
                        (1.0 - t).max(0.0)
                    })
                    .collect();
                let sum: f32 = ws.iter().sum();
                if sum > 0.0 {
                    ws.iter_mut().for_each(|v| *v /= sum);
                } else {
                    ws.iter_mut().for_each(|v| *v = 1.0 / (right - left) as f32);
                }
                (left, ws)
            })
            .collect()
    };
    let (tx, ty) = (taps(w, w2), taps(h, h2));
    let mut across = vec![0f32; w2 * h];
    for y in 0..h {
        for (x, (left, ws)) in tx.iter().enumerate() {
            across[y * w2 + x] = ws
                .iter()
                .enumerate()
                .map(|(i, &k)| k * src[y * w + left + i])
                .sum();
        }
    }
    let mut out = vec![0f32; w2 * h2];
    for (y, (top, ws)) in ty.iter().enumerate() {
        for x in 0..w2 {
            out[y * w2 + x] = ws
                .iter()
                .enumerate()
                .map(|(i, &k)| k * across[(top + i) * w2 + x])
                .sum();
        }
    }
    out
}

/// The band's colour, row by row: the median of every column of the lines.
fn band<'a>(greys: impl Iterator<Item = &'a Grey>) -> [f32; ROWS] {
    let greys: Vec<&Grey> = greys.collect();
    let mut out = [0f32; ROWS];
    for (r, o) in out.iter_mut().enumerate() {
        let mut values: Vec<f32> = greys
            .iter()
            .flat_map(|g| g.px[r * g.w..][..g.w].iter().copied())
            .collect();
        if values.is_empty() {
            continue;
        }
        let mid = values.len() / 2;
        let (_, m, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
        *o = *m;
    }
    out
}

/// The runs of columns that stand out from the band: [(first, width)].
fn ink_runs(g: &Grey, band: &[f32; ROWS], level: f32) -> Vec<(usize, usize)> {
    let d: Vec<f32> = (0..g.w)
        .map(|x| {
            let s: f32 = (0..ROWS)
                .map(|r| {
                    let v = g.px[r * g.w + x] - band[r];
                    v * v
                })
                .sum();
            (s / ROWS as f32).sqrt()
        })
        .collect();
    let top = d.iter().cloned().fold(0.0f32, f32::max);
    if top <= 0.0 {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut x = 0;
    while x < d.len() {
        if d[x] > level * top {
            let a = x;
            while x < d.len() && d[x] > level * top {
                x += 1;
            }
            runs.push((a, x - a));
        } else {
            x += 1;
        }
    }
    runs
}

fn mode(values: &[usize]) -> usize {
    let mut counts = std::collections::BTreeMap::new();
    for &v in values {
        *counts.entry(v).or_insert(0usize) += 1;
    }
    // the most common; of equals, the smallest
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map_or(1, |(v, _)| v)
}

/// The lowest margin at which, of the readings at or over it, at most
/// `max_wrong` are wrong (infinity when even the surest one is wrong).
fn fit_level(found: &[(f32, bool)], max_wrong: f32) -> f32 {
    let mut sorted: Vec<(f32, bool)> = found.to_vec();
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut level, mut wrong) = (f32::INFINITY, 0usize);
    for (i, &(margin, right)) in sorted.iter().enumerate() {
        wrong += usize::from(!right);
        if wrong as f32 <= max_wrong * (i + 1) as f32 {
            level = margin;
        }
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3x5 pixel font for digits and the marks of a HUD number.
    fn bitmap(c: char) -> [&'static str; 5] {
        match c {
            '0' => ["###", "#.#", "#.#", "#.#", "###"],
            '1' => [".#.", "##.", ".#.", ".#.", "###"],
            '2' => ["###", "..#", "###", "#..", "###"],
            '3' => ["###", "..#", ".##", "..#", "###"],
            '4' => ["#.#", "#.#", "###", "..#", "..#"],
            '5' => ["###", "#..", "###", "..#", "###"],
            '6' => ["###", "#..", "###", "#.#", "###"],
            '7' => ["###", "..#", ".#.", ".#.", ".#."],
            '8' => ["###", "#.#", "###", "#.#", "###"],
            '9' => ["###", "#.#", "###", "..#", "###"],
            ',' => ["...", "...", "...", ".#.", "#.."],
            '/' => ["..#", "..#", ".#.", "#..", "#.."],
            _ => unreachable!(),
        }
    }

    /// `text` drawn 3x, light on a dark band, its glyphs three or four
    /// columns apart, blurred across and noisy, as a compressed video would
    /// draw it, on a band twice as long: 15 rows.
    fn line(text: &str, seed: u64) -> RgbaImage {
        let mut rng = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / (1u64 << 31) as f32
        };
        // the band runs on past the text, as a HUD's does: most of each
        // row is band
        let width = 8 + text.chars().count() * 14 * 2 + 8;
        let mut ink = vec![0f32; width * ROWS];
        let mut x = 6 + (next() * 4.0) as usize;
        for c in text.chars() {
            let rows = bitmap(c);
            for (r, row) in rows.iter().enumerate() {
                for (j, b) in row.bytes().enumerate() {
                    if b == b'#' {
                        for dy in 0..3 {
                            for dx in 0..3 {
                                ink[(r * 3 + dy) * width + x + j * 3 + dx] = 1.0;
                            }
                        }
                    }
                }
            }
            x += 9 + 3 + (next() * 2.0) as usize;
        }
        // blur: a 3-tap box across, then down
        let mut blurred = ink.clone();
        for y in 0..ROWS {
            for x in 1..width - 1 {
                blurred[y * width + x] =
                    (ink[y * width + x - 1] + ink[y * width + x] + ink[y * width + x + 1]) / 3.0;
            }
        }
        let mut img = RgbaImage::new(width as u32, ROWS as u32);
        for y in 0..ROWS {
            for x in 0..width {
                let v = 40.0 + 180.0 * blurred[y * width + x] + (next() - 0.5) * 30.0;
                let v = v.clamp(0.0, 255.0) as u8;
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgba([v, v, (v as f32 * 0.9) as u8, 255]),
                );
            }
        }
        img
    }

    fn text(seed: u64) -> String {
        let mut rng = seed;
        let mut next = move |n: u64| {
            rng = rng
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            (rng >> 33) % n
        };
        let a = 1 + next(999);
        let b = a + next(9000);
        format!("{a},{}/{b}", next(10))
    }

    fn whole(img: &RgbaImage) -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: img.width(),
            h: img.height(),
        }
    }

    #[test]
    fn learns_a_blurred_font_from_labelled_lines_and_reads_new_ones() {
        let images: Vec<(RgbaImage, String)> =
            (0..120).map(|i| (line(&text(i), i), text(i))).collect();
        let labelled: Vec<LabelledLine> = images
            .iter()
            .map(|(img, t)| LabelledLine {
                image: img,
                region: whole(img),
                label: t,
            })
            .collect();
        let font = LineFont::learn(&labelled, LineOptions::default()).expect("learns");
        assert_eq!(font.chars().len(), 12, "ten digits, a comma and a slash");
        let (mut right, mut wrong) = (0, 0);
        for i in 1000..1060 {
            let t = text(i);
            let img = line(&t, i);
            match font.read(&img, whole(&img)).value {
                Some(r) if r.text == t => right += 1,
                Some(_) => wrong += 1,
                None => {}
            }
        }
        assert!(right >= 55, "{right} of 60 read right");
        assert!(wrong <= 1, "{wrong} read wrong");
    }

    #[test]
    fn reads_a_line_drawn_bigger_than_it_learned() {
        let images: Vec<(RgbaImage, String)> =
            (0..60).map(|i| (line(&text(i), i), text(i))).collect();
        let labelled: Vec<LabelledLine> = images
            .iter()
            .map(|(img, t)| LabelledLine {
                image: img,
                region: whole(img),
                label: t,
            })
            .collect();
        let font = LineFont::learn(&labelled, LineOptions::default()).expect("learns");
        let t = text(5000);
        let img = line(&t, 5000);
        let big = image::imageops::resize(
            &img,
            img.width() * 2,
            img.height() * 2,
            image::imageops::FilterType::Triangle,
        );
        let reading = font.read_all(&big, whole(&big)).expect("reads");
        assert_eq!(reading.text, t);
        // its glyphs in the big image's coordinates: twice as far along as
        // at the size it learned
        let small = font.read_all(&img, whole(&img)).expect("reads");
        for (b, s) in reading.glyphs.iter().zip(&small.glyphs) {
            assert!(
                b.bounds.x.abs_diff(2 * s.bounds.x) <= 2,
                "{} vs 2 x {}",
                b.bounds.x,
                s.bounds.x
            );
            assert_eq!(b.bounds.h, 30);
        }
    }

    #[test]
    fn too_few_lines_or_nothing_to_start_from_is_said() {
        let img = line("12", 1);
        let few = vec![
            LabelledLine {
                image: &img,
                region: whole(&img),
                label: "12"
            };
            3
        ];
        assert_eq!(
            LineFont::learn(&few, LineOptions::default()).unwrap_err(),
            LineLearnError::TooFew {
                needed: MIN_LINES,
                given: 3
            }
        );
        // labels that never match the line's runs: nothing starts
        let wrong = vec![
            LabelledLine {
                image: &img,
                region: whole(&img),
                label: "1234567"
            };
            12
        ];
        assert_eq!(
            LineFont::learn(&wrong, LineOptions::default()).unwrap_err(),
            LineLearnError::NoStart
        );
    }

    #[test]
    fn the_refusal_level_is_the_lowest_with_few_enough_wrong() {
        let found = [
            (5.0, true),
            (4.0, true),
            (3.0, false),
            (2.0, true),
            (1.0, true),
        ];
        assert_eq!(fit_level(&found, 0.0), 4.0);
        assert_eq!(fit_level(&found, 0.25), 1.0);
        assert_eq!(fit_level(&[(1.0, false)], 0.0), f32::INFINITY);
    }
}
