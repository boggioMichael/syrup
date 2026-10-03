//! A bar that fills and empties (health, a timer, a progress bar): learned
//! once from roughly where it is, then measured on every frame.
//!
//! [`BarModel::learn`] finds the bar's colour (the most common saturated
//! hue in the box, or the one nearest the colour expected) and the rows it
//! fills, which pins the box down to the pixel. How far the fill can go —
//! the track's right end — is first taken from the box, then worked out
//! from readings: when the application says the bar is at 80% and the fill
//! ends at pixel 400, the track is (400 − start) / 0.8 long
//! ([`BarModel::reading`]); every reading makes it more exact, by least
//! squares through the track's start.
//!
//! The model is a few numbers as fractions of the frame, so it survives a
//! window resize and (with the `serde` feature) a trip to disk.
//!
//! ```text
//!   learn(frame, ≈box, hue?)  ─▶  BarModel { band, hue, tolerance }
//!   measure(frame)            ─▶  0..100 %, or None when the bar is not to be seen
//!   reading(frame, 80%)       ─▶  the track's end, refitted
//! ```

use image::RgbaImage;

use crate::color::{hsv_from_rgb, is_color_pixel};
use crate::geometry::{NormRect, Rect};

/// A learned bar: where its fill runs and what colour it is.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BarModel {
    /// The fill's rows (`y0`..`y1`), where the fill starts (`x0`) and where
    /// a full bar would end (`x1`), as fractions of the frame.
    pub band: NormRect,
    /// The fill's hue (degrees) and how far from it a pixel may be (capped
    /// at [`HUE_TOLERANCE`] when measuring, whatever was learned).
    pub hue: f32,
    pub hue_tolerance: f32,
    /// The least saturation and brightness a pixel of the fill has, learned
    /// from the fill itself (0: not learned; the usual least is used). A
    /// track that shows the scene through it — lava behind the EXP bar —
    /// is of the fill's hue but duller and darker, and would otherwise
    /// read as fill.
    #[cfg_attr(feature = "serde", serde(default))]
    pub min_saturation: f32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub min_value: f32,
    /// Readings that fixed where the track ends: (where the fill ended, as
    /// a fraction of the frame width; the percent the application showed).
    #[cfg_attr(feature = "serde", serde(default))]
    pub readings: Vec<(f32, f32)>,
}

/// Saturation and brightness a fill pixel needs when the bar's own are not
/// known (the dark empty track and white text do not have them).
const MIN_SATURATION: f32 = 0.35;
const MIN_VALUE: f32 = 0.28;
/// A fill pixel has at least this share of the fill's typical saturation
/// and brightness (the fill is shaded towards its edges; a translucent
/// track showing the scene behind it is duller and darker than that).
const FILL_SHARE: f32 = 0.6;
/// How far a pixel's hue may be from the fill's and count as fill, in
/// degrees. (Hue barely moves with shading; a wider tolerance took a lava
/// background, 20° off a yellow bar, for its fill.)
pub const HUE_TOLERANCE: f32 = 16.0;
/// How far the colour found may be from the colour expected of the bar
/// before it is not that bar at all, in degrees (a yellow bar's lava
/// background is 27° off it).
pub const EXPECTED_WITHIN: f32 = 25.0;
/// Readings kept for the fit.
const MAX_READINGS: usize = 12;

/// The shorter way round the hue circle between two hues, in degrees.
pub fn hue_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

impl BarModel {
    /// The least saturation and brightness of a fill pixel: the bar's own,
    /// when learned, else the usual.
    fn least(&self) -> (f32, f32) {
        let s = if self.min_saturation > 0.0 {
            self.min_saturation
        } else {
            MIN_SATURATION
        };
        let v = if self.min_value > 0.0 {
            self.min_value
        } else {
            MIN_VALUE
        };
        (s, v)
    }

    #[inline]
    fn is_fill(&self, p: &image::Rgba<u8>) -> bool {
        let (h, s, v) = hsv_from_rgb(p.0[0], p.0[1], p.0[2]);
        let (min_s, min_v) = self.least();
        s >= min_s
            && v >= min_v
            && hue_distance(h, self.hue) <= self.hue_tolerance.min(HUE_TOLERANCE)
    }

    /// Learn a bar from `approx` (roughly where it is). `expected_hue`: the
    /// colour it should be (red for health, say), when known. `None` when
    /// no coloured bar is there.
    pub fn learn(
        frame: &RgbaImage,
        approx: &NormRect,
        expected_hue: Option<f32>,
    ) -> Option<BarModel> {
        let (fw, fh) = frame.dimensions();
        let (bx, by, bw, bh) = approx.pixels(fw, fh);
        // The hue: a histogram of the saturated pixels in the box, in 10° bins.
        let mut bins = [0u32; 36];
        for y in by..(by + bh).min(fh) {
            for x in bx..(bx + bw).min(fw) {
                let p = frame.get_pixel(x, y).0;
                let (h, s, v) = hsv_from_rgb(p[0], p[1], p[2]);
                if s >= MIN_SATURATION && v >= MIN_VALUE {
                    bins[(h / 10.0) as usize % 36] += 1;
                }
            }
        }
        let most = *bins.iter().max()?;
        if most < 12 {
            return None;
        }
        // With a colour expected, the nearest hue with any presence in the
        // box wins, however small: the box is rough, and the scenery in
        // it (lava, sky) can outnumber the bar fifty to one.
        let bin = match expected_hue {
            Some(expected) => (0..36)
                .filter(|&b| bins[b] >= 12 && bins[b] * 50 >= most)
                .min_by(|&a, &b| {
                    hue_distance(a as f32 * 10.0 + 5.0, expected)
                        .total_cmp(&hue_distance(b as f32 * 10.0 + 5.0, expected))
                })?,
            None => (0..36).max_by_key(|&b| bins[b])?,
        };
        // Nothing of the colour expected: not this bar (the nearest colour
        // in the box may be the scenery).
        if let Some(expected) = expected_hue
            && hue_distance(bin as f32 * 10.0 + 5.0, expected) > EXPECTED_WITHIN
        {
            return None;
        }
        let mut model = BarModel {
            band: *approx,
            hue: bin as f32 * 10.0 + 5.0,
            hue_tolerance: HUE_TOLERANCE,
            min_saturation: 0.0,
            min_value: 0.0,
            readings: Vec::new(),
        };
        // A finer hue: the circular mean of the pixels near the bin; and
        // how saturated and bright the fill is, to tell it from a track
        // of its hue (the scene showing through) later.
        let (mut sx, mut sy) = (0f64, 0f64);
        let (mut sats, mut vals) = (Vec::new(), Vec::new());
        for y in by..(by + bh).min(fh) {
            for x in bx..(bx + bw).min(fw) {
                let p = frame.get_pixel(x, y);
                if model.is_fill(p) {
                    let (h, s, v) = hsv_from_rgb(p.0[0], p.0[1], p.0[2]);
                    sx += (h as f64).to_radians().cos();
                    sy += (h as f64).to_radians().sin();
                    sats.push(s);
                    vals.push(v);
                }
            }
        }
        model.hue = (sy.atan2(sx).to_degrees() as f32).rem_euclid(360.0);
        sats.sort_by(f32::total_cmp);
        vals.sort_by(f32::total_cmp);
        if !sats.is_empty() {
            model.min_saturation = (sats[sats.len() / 2] * FILL_SHARE).max(MIN_SATURATION);
            model.min_value = (vals[vals.len() / 2] * FILL_SHARE).max(MIN_VALUE);
        }
        // Boxes are rough about a thin bar's height: look well above and
        // below. Sideways, a little past the box's ends.
        let pad_y = bh.max(8);
        let y_from = by.saturating_sub(pad_y);
        let y_to = (by + bh + pad_y).min(fh);
        let x_from = bx.saturating_sub(bw / 12);
        let x_to = (bx + bw + bw / 50 + 2).min(fw);
        let gap = (bw / 8).max(3);
        // Each row's longest run of the colour (gaps for text allowed).
        let runs: Vec<Option<(u32, u32)>> = (y_from..y_to)
            .map(|y| model.longest_run(frame, y, x_from, x_to, gap))
            .collect();
        let min_run = (bw / 60).max(3);
        // Bands: rows in a row with a run, starting at about the same place.
        let mut bands: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < runs.len() {
            match runs[i] {
                Some((a, b)) if b - a >= min_run => {
                    let mut j = i + 1;
                    while j < runs.len()
                        && runs[j].is_some_and(|(c, d)| d - c >= min_run && c.abs_diff(a) <= gap)
                    {
                        j += 1;
                    }
                    bands.push((i, j));
                    i = j;
                }
                _ => i += 1,
            }
        }
        // The bar: a band no taller than a bar can be, nearest the box's
        // middle.
        let tallest = (bh * 5 / 2).max(30) as usize;
        let middle = (by + bh / 2) as i64 - y_from as i64;
        let (top, bottom) = bands
            .into_iter()
            .filter(|(a, b)| b - a >= 2 && b - a <= tallest)
            .min_by_key(|(a, b)| ((*a + *b) as i64 / 2 - middle).abs())?;
        let mut starts: Vec<u32> = runs[top..bottom].iter().flatten().map(|r| r.0).collect();
        let mut ends: Vec<u32> = runs[top..bottom].iter().flatten().map(|r| r.1).collect();
        starts.sort_unstable();
        ends.sort_unstable();
        let x0 = starts[starts.len() / 2];
        let x_end = ends[ends.len() / 2];
        let (y0, y1) = (y_from + top as u32, y_from + bottom as u32);
        // Until a reading says otherwise, the track ends where the box does
        // (or where the fill does, if it goes further). A fill that nearly
        // reaches the box's end is a full bar: it ends there.
        let box_end = bx + bw;
        let x1 = if box_end.saturating_sub(x_end) * 25 <= box_end.saturating_sub(x0) {
            x_end
        } else {
            box_end.max(x_end)
        }
        .min(fw);
        model.band = NormRect::from_pixels(x0, y0, x1.saturating_sub(x0).max(1), y1 - y0, fw, fh);
        Some(model)
    }

    /// The longest run of fill pixels in row `y` between `from` and `to`,
    /// bridging gaps of up to `gap` (text over the bar): start and end
    /// (exclusive).
    fn longest_run(
        &self,
        frame: &RgbaImage,
        y: u32,
        from: u32,
        to: u32,
        gap: u32,
    ) -> Option<(u32, u32)> {
        let mut best: Option<(u32, u32)> = None;
        let mut current: Option<(u32, u32)> = None;
        for x in from..to {
            if self.is_fill(frame.get_pixel(x, y)) {
                current = Some(match current {
                    Some((a, b)) if x - b <= gap => (a, x + 1),
                    _ => (x, x + 1),
                });
                if let Some((a, b)) = current
                    && best.is_none_or(|(c, d)| b - a > d - c)
                {
                    best = Some((a, b));
                }
            }
        }
        best
    }

    /// Where row `y`'s fill ends: the run that starts at the track's start
    /// (`x0`, give or take `slack`), bridging gaps up to `gap`.
    fn row_end(
        &self,
        frame: &RgbaImage,
        y: u32,
        x0: u32,
        to: u32,
        slack: u32,
        gap: u32,
    ) -> Option<u32> {
        let mut end: Option<u32> = None;
        for x in x0.saturating_sub(slack)..to {
            if self.is_fill(frame.get_pixel(x, y)) {
                match end {
                    Some(e) if x - e > gap => break,
                    _ => end = Some(x + 1),
                }
            } else if end.is_none() && x > x0 + slack {
                // Nothing at the start: no fill in this row.
                return None;
            } else if let Some(e) = end
                && x - e > gap
            {
                break;
            }
        }
        end
    }

    /// Where the fill ends now, in pixels, if the bar is to be seen: the
    /// track's start and end, and the fill's end.
    fn fill_end(&self, frame: &RgbaImage) -> Option<(u32, u32, u32)> {
        let (fw, fh) = frame.dimensions();
        let (x0, y0, w, h) = self.band.pixels(fw, fh);
        if y0 >= fh || x0 >= fw {
            return None;
        }
        // A little past the track's end, in case it was set short.
        let to = (x0 + w + w / 50 + 2).min(fw);
        let slack = (w / 50).max(3);
        // Gaps in a row's fill come from text drawn over the bar, so they
        // are at most a few glyphs wide: a few times the bar's height.
        // Measured against the width alone, a bar that runs the whole
        // width of a large screen (an EXP bar at 4K: 3,840 px) would
        // bridge hundreds of pixels of nothing to the next thing of its
        // colour, and read 60% when it was at 19%.
        let gap = (w / 8).min(3 * h).max(3);
        let mut ends: Vec<u32> = (y0..(y0 + h).min(fh))
            .filter_map(|y| self.row_end(frame, y, x0, to, slack, gap))
            .collect();
        if ends.is_empty() || ends.len() * 10 < h as usize * 4 {
            return None;
        }
        ends.sort_unstable();
        Some((x0, x0 + w, ends[ends.len() / 2]))
    }

    /// How full the bar is, 0 to 100, if it can be seen. An empty bar and
    /// one covered up look the same, so neither gives a number.
    pub fn measure(&self, frame: &RgbaImage) -> Option<f32> {
        let (x0, x1, end) = self.fill_end(frame)?;
        if x1 <= x0 {
            return None;
        }
        Some(((end.saturating_sub(x0)) as f32 / (x1 - x0) as f32 * 100.0).clamp(0.0, 100.0))
    }

    /// The application shows `percent` now: work out from it where the
    /// track ends. Readings of a nearly empty bar say too little about the
    /// track and are not used.
    pub fn reading(&mut self, frame: &RgbaImage, percent: f32) {
        let Some((x0, _, end)) = self.fill_end(frame) else {
            return;
        };
        let fw = frame.width() as f32;
        if percent < 8.0 || end <= x0 {
            return;
        }
        self.readings.push((end as f32 / fw, percent.min(100.0)));
        if self.readings.len() > MAX_READINGS {
            self.readings.remove(0);
        }
        // Least squares through the start: length = Σ(d·p) / Σ(p²).
        let start = self.band.x0;
        let (mut dp, mut pp) = (0f64, 0f64);
        for &(e, p) in &self.readings {
            let (d, p) = ((e - start) as f64, p as f64 / 100.0);
            dp += d * p;
            pp += p * p;
        }
        if pp > 0.0 {
            let length = (dp / pp) as f32;
            if length > 0.0 {
                self.band.x1 = (start + length).min(1.0);
            }
        }
    }
}

/// A bar of a colour in `region` of `frame`, where it is not yet known: the
/// fill's rows and extent. Rows are read for their longest run of the
/// colour (bridging the gaps text leaves), and a band of rows whose runs
/// start at about the same place and are about as long is a bar; the
/// tallest such band of at least [`BAR_ROWS`] rows wins — not the biggest
/// blob of the colour, which on a lava map is the lava, nor a one-pixel
/// line of it, which is a border. `hue_range` wraps (340 to 20 is red).
pub fn find_bar(
    frame: &RgbaImage,
    region: Rect,
    hue_range: (f32, f32),
    min_saturation: f32,
    min_value: f32,
) -> Option<Rect> {
    let (fw, fh) = frame.dimensions();
    let x_to = region.x.saturating_add(region.w).min(fw);
    let y_to = region.y.saturating_add(region.h).min(fh);
    if region.x >= x_to || region.y >= y_to {
        return None;
    }
    let gap = (region.w / 100).max(3);
    let min_run = (region.w / 60).max(3);
    let is_fill = |p: &image::Rgba<u8>| is_color_pixel(p, hue_range, min_saturation, min_value);
    // Each row's runs of the colour (start, end), gaps bridged, short ones
    // dropped.
    let runs: Vec<Vec<(u32, u32)>> = (region.y..y_to)
        .map(|y| {
            let mut found = Vec::new();
            let mut current: Option<(u32, u32)> = None;
            for x in region.x..x_to {
                if is_fill(frame.get_pixel(x, y)) {
                    current = Some(match current {
                        Some((a, b)) if x - b <= gap => (a, x + 1),
                        Some(done) => {
                            found.push(done);
                            (x, x + 1)
                        }
                        None => (x, x + 1),
                    });
                }
            }
            found.extend(current);
            found.retain(|(a, b)| b - a >= min_run);
            found
        })
        .collect();
    // Bands: a run, and the runs in the rows below that start where it did
    // and are about as long, each used once.
    let mut used: Vec<Vec<bool>> = runs.iter().map(|r| vec![false; r.len()]).collect();
    let mut best: Option<(u32, Rect)> = None;
    for i in 0..runs.len() {
        for k in 0..runs[i].len() {
            if used[i][k] {
                continue;
            }
            let (a, b) = runs[i][k];
            let len = b - a;
            let mut members = vec![(a, len)];
            used[i][k] = true;
            let mut j = i + 1;
            while j < runs.len() {
                let Some(m) = runs[j]
                    .iter()
                    .position(|(c, d)| c.abs_diff(a) <= gap && (d - c).abs_diff(len) * 5 <= len)
                else {
                    break;
                };
                if used[j][m] {
                    break;
                }
                used[j][m] = true;
                members.push((runs[j][m].0, runs[j][m].1 - runs[j][m].0));
                j += 1;
            }
            let rows = members.len() as u32;
            // At least a few rows, and thin: a block of the colour taller
            // than a third of the region is a panel or the scenery.
            if rows < BAR_ROWS || rows * 3 > region.h {
                continue;
            }
            let mut starts: Vec<u32> = members.iter().map(|m| m.0).collect();
            let mut lens: Vec<u32> = members.iter().map(|m| m.1).collect();
            starts.sort_unstable();
            lens.sort_unstable();
            let found = Rect {
                x: starts[starts.len() / 2],
                y: region.y + i as u32,
                w: lens[lens.len() / 2],
                h: rows,
            };
            let score = found.w * found.h;
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, found));
            }
        }
    }
    best.map(|(_, r)| r)
}

/// A bar is at least this many rows tall; fewer is a line (a border).
pub const BAR_ROWS: u32 = 3;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use image::Rgba;

    /// A status bar: dark background, a red bar and a blue bar with white
    /// text over them and a gradient on each, and a thin yellow-green bar
    /// along the bottom.
    pub(crate) fn status_bar(red: f32, blue: f32, bottom: f32) -> RgbaImage {
        let mut f = RgbaImage::from_pixel(1280, 720, Rgba([60, 90, 140, 255]));
        for y in 640..700 {
            for x in 400..880 {
                f.put_pixel(x, y, Rgba([28, 28, 34, 255]));
            }
        }
        let bar = |f: &mut RgbaImage, y0: u32, fill: f32, rgb: [u8; 3]| {
            let (x0, x1) = (520u32, 860u32);
            let end = x0 + ((x1 - x0) as f32 * fill / 100.0) as u32;
            for y in y0..y0 + 10 {
                let shade = 1.0 - (y - y0) as f32 * 0.04;
                for x in x0..x1 {
                    let p = if x < end {
                        Rgba([
                            (rgb[0] as f32 * shade) as u8,
                            (rgb[1] as f32 * shade) as u8,
                            (rgb[2] as f32 * shade) as u8,
                            255,
                        ])
                    } else {
                        Rgba([45, 45, 50, 255])
                    };
                    f.put_pixel(x, y, p);
                }
            }
            // "1234 / 2000" in white over the middle.
            for y in y0 + 2..y0 + 8 {
                for x in (660..720).step_by(3) {
                    f.put_pixel(x, y, Rgba([250, 250, 250, 255]));
                }
            }
        };
        bar(&mut f, 652, red, [230, 40, 50]);
        bar(&mut f, 670, blue, [40, 110, 235]);
        let end = (1280.0 * bottom / 100.0) as u32;
        for y in 708..714 {
            for x in 0..1280 {
                let p = if x < end {
                    Rgba([200, 220, 40, 255])
                } else {
                    Rgba([30, 30, 30, 255])
                };
                f.put_pixel(x, y, p);
            }
        }
        f
    }

    #[test]
    fn a_bar_is_learned_from_a_rough_box_and_measured() {
        let frame = status_bar(60.0, 100.0, 50.0);
        // The box: a bit off in every direction.
        let rough = NormRect::new(512.0 / 1280.0, 648.0 / 720.0, 850.0 / 1280.0, 664.0 / 720.0);
        let red = BarModel::learn(&frame, &rough, Some(0.0)).expect("learned");
        let (x, y, _, h) = red.band.pixels(1280, 720);
        assert_eq!((x, y, h), (520, 652, 10), "{:?}", red.band);
        assert!(hue_distance(red.hue, 357.0) < 8.0, "{}", red.hue);
        // The track's end comes from the rough box until a reading fixes it.
        let first = red.measure(&frame).unwrap();
        assert!((first - 61.0).abs() < 3.0, "{first}");
        let mut red = red;
        red.reading(&frame, 60.0);
        let (_, _, w, _) = red.band.pixels(1280, 720);
        assert!(w.abs_diff(340) <= 2, "{w}");
        // Now exact on other frames.
        for fill in [25.0, 80.0, 100.0] {
            let got = red.measure(&status_bar(fill, 100.0, 50.0)).unwrap();
            assert!((got - fill).abs() < 1.5, "{fill}: {got}");
        }
        // The blue bar next to it is not mistaken for it.
        let blue = BarModel::learn(&frame, &rough.grown(0.0, 1.5), Some(220.0)).unwrap();
        assert!(hue_distance(blue.hue, 220.0) < 10.0, "{}", blue.hue);
        let (_, my, _, _) = blue.band.pixels(1280, 720);
        assert_eq!(my, 670);
        // The thin bar along the bottom, from a box taller than it.
        let bottom = NormRect::new(0.0, 706.0 / 720.0, 1.0, 716.0 / 720.0);
        let thin = BarModel::learn(&frame, &bottom, Some(55.0)).unwrap();
        let got = thin.measure(&frame).unwrap();
        assert!((got - 50.0).abs() < 1.5, "{got}");
        // With no colour expected, the most common saturated colour in the
        // box is taken for the bar: here the blue background around it.
        let blue_too = BarModel::learn(&frame, &bottom, None).unwrap();
        assert!(hue_distance(blue_too.hue, 217.0) < 10.0, "{}", blue_too.hue);
        // A box on the bar itself needs no hint.
        let snug = NormRect::new(0.0, 708.0 / 720.0, 1.0, 714.0 / 720.0);
        let thin = BarModel::learn(&frame, &snug, None).unwrap();
        assert!((thin.measure(&frame).unwrap() - 50.0).abs() < 1.5);
    }

    #[test]
    fn a_covered_or_empty_bar_gives_no_number() {
        let frame = status_bar(60.0, 100.0, 50.0);
        let rough = NormRect::new(520.0 / 1280.0, 652.0 / 720.0, 860.0 / 1280.0, 662.0 / 720.0);
        let red = BarModel::learn(&frame, &rough, Some(0.0)).unwrap();
        assert_eq!(red.measure(&status_bar(0.0, 100.0, 50.0)), None);
        let covered = RgbaImage::from_pixel(1280, 720, Rgba([200, 200, 200, 255]));
        assert_eq!(red.measure(&covered), None);
        // Nothing to learn where there is no bar.
        assert!(BarModel::learn(&covered, &rough, None).is_none());
        // A frame too small for the band: nothing, not a panic.
        assert_eq!(red.measure(&RgbaImage::new(64, 32)), None);
    }

    #[test]
    fn a_thing_of_the_bars_colour_far_along_the_track_is_not_its_fill() {
        // The bottom bar runs the whole width: 19% full, with a patch of
        // its own colour (an icon, a button) 400 px past the fill's end.
        let mut frame = status_bar(60.0, 100.0, 19.0);
        let end = (1280.0 * 0.19) as u32;
        for y in 708..714 {
            for x in end + 400..end + 440 {
                frame.put_pixel(x, y, Rgba([200, 220, 40, 255]));
            }
        }
        let rough = NormRect::new(0.0, 708.0 / 720.0, 1.0, 714.0 / 720.0);
        let bar = BarModel::learn(&status_bar(60.0, 100.0, 19.0), &rough, Some(55.0)).unwrap();
        let got = bar.measure(&frame).unwrap();
        assert!((got - 19.0).abs() < 1.5, "{got}");
        // Text over the fill — gaps of a glyph or two — is still bridged.
        let mut written = status_bar(60.0, 100.0, 50.0);
        for y in 708..714 {
            for x in (100..180).step_by(4) {
                written.put_pixel(x, y, Rgba([250, 250, 250, 255]));
            }
        }
        let got = bar.measure(&written).unwrap();
        assert!((got - 50.0).abs() < 1.5, "{got}");
    }

    /// The status bar on a lava map: the scene shows through the bottom
    /// bar's track (dull orange, the fill's hue but duller and darker),
    /// bright orange runs above and below it, and a thin red line of a
    /// panel's border sits right of the red bar.
    fn lava_status_bar(red: f32, bottom: f32) -> RgbaImage {
        let mut f = status_bar(red, 100.0, bottom);
        for y in 700..720 {
            for x in 0..1280 {
                f.put_pixel(x, y, Rgba([179, 106, 0, 255]));
            }
        }
        let end = (1280.0 * bottom / 100.0) as u32;
        for y in 708..714 {
            for x in 0..1280 {
                let p = if x < end {
                    Rgba([225, 240, 0, 255])
                } else {
                    // The track, lava behind it: warmer towards the bottom.
                    let warmth = (y - 708) as u8 * 3;
                    Rgba([76 + warmth, 65 + warmth / 2, 48 - warmth / 2, 255])
                };
                f.put_pixel(x, y, p);
            }
        }
        for y in [646u32, 647, 666, 667] {
            for x in 900..1240 {
                f.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        f
    }

    #[test]
    fn a_track_showing_the_scene_through_it_is_not_fill() {
        let frame = lava_status_bar(60.0, 19.0);
        let bottom = NormRect::new(0.0, 708.0 / 720.0, 1.0, 714.0 / 720.0);
        let bar = BarModel::learn(&frame, &bottom, Some(62.0)).expect("learned");
        assert!(hue_distance(bar.hue, 64.0) < 4.0, "{}", bar.hue);
        assert!(bar.min_saturation > 0.5 && bar.min_value > 0.5, "{bar:?}");
        let got = bar.measure(&frame).unwrap();
        assert!(
            (got - 19.0).abs() < 1.5,
            "{got} {bar:?} {:?}",
            bar.band.pixels(1280, 720)
        );
        // A box that takes in the orange above and below reads the same.
        let loose = NormRect::new(0.0, 702.0 / 720.0, 1.0, 720.0 / 720.0);
        let bar = BarModel::learn(&frame, &loose, Some(62.0)).expect("learned");
        let got = bar.measure(&frame).unwrap();
        assert!((got - 19.0).abs() < 1.5, "{got}");
        // A model learned before the fill's own least was kept measures
        // with the usual least — and the tolerance capped — so it still
        // does not take the bright orange for fill.
        let mut old = bar.clone();
        old.min_saturation = 0.0;
        old.min_value = 0.0;
        old.hue_tolerance = 24.0;
        let got = old.measure(&frame).unwrap();
        assert!(got < 60.0, "{got}");
        // Nothing of the expected colour in the box: not that bar.
        let orange = NormRect::new(0.0, 700.0 / 720.0, 1.0, 706.0 / 720.0);
        assert!(BarModel::learn(&frame, &orange, Some(200.0)).is_none());
        assert!(BarModel::learn(&frame, &orange, Some(62.0)).is_none());
    }

    #[test]
    fn a_bar_is_found_as_a_band_of_rows_not_the_biggest_blob() {
        let frame = lava_status_bar(60.0, 19.0);
        let strip = Rect {
            x: 0,
            y: 640,
            w: 960,
            h: 80,
        };
        // The red bar, not the lava (20° off, kept out by the range) nor
        // the red border lines (two rows each).
        let red = find_bar(&frame, strip, (325.0, 15.0), 0.35, 0.3).expect("the red bar");
        assert_eq!((red.y, red.h), (652, 10), "{red:?}");
        assert!(
            red.x.abs_diff(520) <= 2 && red.w.abs_diff(204) <= 4,
            "{red:?}"
        );
        // The blue bar: the blue background around the status bar is cut
        // off by the strip; only the bar is a band of equal rows.
        let blue = find_bar(&frame, strip, (180.0, 235.0), 0.3, 0.3).expect("the blue bar");
        assert_eq!((blue.y, blue.h), (670, 10), "{blue:?}");
        // The bottom bar: its fill, not the orange runs above and below.
        let yellow = find_bar(&frame, strip, (48.0, 80.0), 0.25, 0.25).expect("the yellow bar");
        assert_eq!((yellow.y, yellow.h), (708, 6), "{yellow:?}");
        assert!(yellow.w.abs_diff(243) <= 3, "{yellow:?}");
        // Nothing green in it.
        assert!(find_bar(&frame, strip, (100.0, 150.0), 0.3, 0.3).is_none());
    }

    #[test]
    fn readings_refit_the_track_and_the_oldest_are_dropped() {
        let frame = status_bar(60.0, 100.0, 50.0);
        let rough = NormRect::new(512.0 / 1280.0, 648.0 / 720.0, 700.0 / 1280.0, 664.0 / 720.0);
        let mut red = BarModel::learn(&frame, &rough, Some(0.0)).unwrap();
        // The box ended short of the track, so the bar reads as over-full.
        assert_eq!(red.measure(&frame), Some(100.0));
        for fill in [60.0, 30.0, 90.0, 45.0] {
            red.reading(&status_bar(fill, 100.0, 50.0), fill);
        }
        let got = red.measure(&status_bar(70.0, 100.0, 50.0)).unwrap();
        assert!((got - 70.0).abs() < 1.5, "{got}");
        for _ in 0..20 {
            red.reading(&frame, 60.0);
        }
        assert_eq!(red.readings.len(), MAX_READINGS);
        // A nearly empty bar teaches nothing.
        let before = red.readings.len();
        red.reading(&status_bar(5.0, 100.0, 50.0), 5.0);
        assert_eq!(red.readings.len(), before);
    }
}
