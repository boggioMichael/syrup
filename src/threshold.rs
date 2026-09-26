//! Turning colour pixels into a foreground/background decision.
//!
//! Fixed thresholds break as soon as the skin, the lighting or the capture
//! path changes, so this module derives them from the pixels themselves:
//!
//! - [`otsu_threshold`] picks the cut that best separates a bimodal
//!   histogram (text versus the panel behind it).
//! - [`IntegralImage`] answers "what is the mean brightness around here?"
//!   in constant time, for local rather than global decisions.
//! - [`Channel`] chooses which view of a colour pixel to threshold. The
//!   minimum channel is the useful one for light text drawn on a saturated
//!   bar: the bar is dark in at least one channel, the text is bright in
//!   all of them.
//! - [`text_evidence`] combines the two: how much brighter each pixel is
//!   than the typical pixel of its own row, which lifts glyph strokes off
//!   bars and gradients whose overall brightness varies.

use image::{GrayImage, Luma, RgbaImage};

use crate::geometry::{Rect, row_pixels};

/// Which single value to read from a colour pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Rec. 601 luma, the usual brightness.
    Luma,
    /// The smallest of R, G and B: high only where a pixel is bright in
    /// every channel, i.e. white or light-grey text, whatever the bar colour.
    Min,
    /// The largest of R, G and B (HSV value).
    Max,
}

impl Channel {
    #[inline]
    fn of(self, p: &image::Rgba<u8>) -> u8 {
        match self {
            Channel::Luma => {
                ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000) as u8
            }
            Channel::Min => p[0].min(p[1]).min(p[2]),
            Channel::Max => p[0].max(p[1]).max(p[2]),
        }
    }
}

/// The chosen channel of `region` as a grey image the size of the region
/// (clipped to the image).
pub fn channel_image(image: &RgbaImage, region: Rect, channel: Channel) -> GrayImage {
    let x_end = region.x.saturating_add(region.w).min(image.width());
    let y_end = region.y.saturating_add(region.h).min(image.height());
    if region.x >= x_end || region.y >= y_end {
        return GrayImage::new(0, 0);
    }
    let mut out = GrayImage::new(x_end - region.x, y_end - region.y);
    for y in region.y..y_end {
        for (x, pixel) in row_pixels(image, y, region.x, x_end - 1) {
            out.put_pixel(x - region.x, y - region.y, Luma([channel.of(pixel)]));
        }
    }
    out
}

/// A 256-bin histogram of a grey image.
pub fn histogram(gray: &GrayImage) -> [u32; 256] {
    let mut bins = [0u32; 256];
    for &v in gray.as_raw() {
        bins[v as usize] += 1;
    }
    bins
}

/// Otsu's threshold: the value `t` that maximises the between-class
/// variance of `{v <= t}` and `{v > t}`. Returns `None` when the histogram
/// has fewer than two distinct values (nothing to separate).
pub fn otsu_threshold(histogram: &[u32; 256]) -> Option<u8> {
    let total: u64 = histogram.iter().map(|&c| c as u64).sum();
    if total == 0 || histogram.iter().filter(|&&c| c > 0).count() < 2 {
        return None;
    }
    let sum_all: f64 = histogram
        .iter()
        .enumerate()
        .map(|(v, &c)| v as f64 * c as f64)
        .sum();
    let (mut weight_below, mut sum_below) = (0u64, 0f64);
    let (mut best, mut best_t) = (-1.0f64, 0u8);
    // The last bin cannot be a cut: nothing would lie above it.
    for (t, &count) in histogram.iter().enumerate().take(255) {
        weight_below += count as u64;
        if weight_below == 0 {
            continue;
        }
        let weight_above = total - weight_below;
        if weight_above == 0 {
            break;
        }
        sum_below += t as f64 * count as f64;
        let mean_below = sum_below / weight_below as f64;
        let mean_above = (sum_all - sum_below) / weight_above as f64;
        let between = weight_below as f64 * weight_above as f64 * (mean_below - mean_above).powi(2);
        if between > best {
            best = between;
            best_t = t as u8;
        }
    }
    Some(best_t)
}

/// Summed-area table: the sum of any axis-aligned rectangle in O(1).
#[derive(Debug, Clone)]
pub struct IntegralImage {
    width: u32,
    height: u32,
    /// `(width + 1) * (height + 1)` running sums with a zero first row and
    /// column, so rectangle sums need no edge cases.
    sums: Vec<u64>,
}

impl IntegralImage {
    pub fn new(gray: &GrayImage) -> Self {
        let (width, height) = gray.dimensions();
        let stride = width as usize + 1;
        let mut sums = vec![0u64; stride * (height as usize + 1)];
        for y in 0..height as usize {
            let mut row_sum = 0u64;
            for x in 0..width as usize {
                row_sum += gray.as_raw()[y * width as usize + x] as u64;
                sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + row_sum;
            }
        }
        Self {
            width,
            height,
            sums,
        }
    }

    /// Sum of the pixels of `rect`, clipped to the image.
    pub fn sum(&self, rect: Rect) -> u64 {
        let x0 = rect.x.min(self.width) as usize;
        let y0 = rect.y.min(self.height) as usize;
        let x1 = rect.x.saturating_add(rect.w).min(self.width) as usize;
        let y1 = rect.y.saturating_add(rect.h).min(self.height) as usize;
        if x1 <= x0 || y1 <= y0 {
            return 0;
        }
        let stride = self.width as usize + 1;
        self.sums[y1 * stride + x1] + self.sums[y0 * stride + x0]
            - self.sums[y0 * stride + x1]
            - self.sums[y1 * stride + x0]
    }

    /// Mean of the pixels of `rect`, clipped to the image; `None` when the
    /// clipped rectangle is empty.
    pub fn mean(&self, rect: Rect) -> Option<f32> {
        let x1 = rect.x.saturating_add(rect.w).min(self.width);
        let y1 = rect.y.saturating_add(rect.h).min(self.height);
        let area = x1.saturating_sub(rect.x.min(self.width)) as u64
            * y1.saturating_sub(rect.y.min(self.height)) as u64;
        (area > 0).then(|| self.sum(rect) as f32 / area as f32)
    }
}

/// Whether text is lighter or darker than what it is drawn on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Light glyphs on a darker (or more saturated) background: values on
    /// bars, white labels on panels.
    LightText,
    /// Dark glyphs on a lighter background.
    DarkText,
}

/// How strongly each pixel of `region` stands out as text: how far its
/// `channel` value lies from the median of its row, in the direction given
/// by `polarity`, scaled so a stroke `span` levels away from the row's
/// typical pixel reads 255 (clamped to `0..=255`).
///
/// Text occupies a minority of any row that crosses it, so the row median
/// is the background behind the text — the bar's fill, a gradient, a panel
/// — and subtracting it removes that background whatever its colour or
/// brightness. Returns an image the size of the clipped region.
pub fn text_evidence(
    image: &RgbaImage,
    region: Rect,
    channel: Channel,
    polarity: Polarity,
    span: u8,
) -> GrayImage {
    let values = channel_image(image, region, channel);
    let (width, height) = values.dimensions();
    let mut out = GrayImage::new(width, height);
    let span = span.max(1) as f32;
    if width == 0 {
        return out;
    }
    let row_len = width as usize;
    let mut scratch = Vec::with_capacity(row_len);
    for (row, out_row) in values
        .as_raw()
        .chunks_exact(row_len)
        .zip(out.chunks_exact_mut(row_len))
    {
        scratch.clear();
        scratch.extend_from_slice(row);
        // The value at the middle rank, without sorting the whole row.
        let median = *scratch.select_nth_unstable(row_len / 2).1 as f32;
        for (out, &v) in out_row.iter_mut().zip(row) {
            let away = match polarity {
                Polarity::LightText => v as f32 - median,
                Polarity::DarkText => median - v as f32,
            };
            *out = (away / span * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Binarise `gray`: 255 where the value is strictly above `threshold`.
pub fn binarize(gray: &GrayImage, threshold: u8) -> GrayImage {
    let mut out = gray.clone();
    for v in out.iter_mut() {
        *v = if *v > threshold { 255 } else { 0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn otsu_splits_a_bimodal_histogram_between_the_modes() {
        let mut hist = [0u32; 256];
        hist[40] = 500;
        hist[45] = 300;
        hist[200] = 120;
        hist[210] = 80;
        let t = otsu_threshold(&hist).unwrap();
        assert!(
            (45..200).contains(&t),
            "threshold {t} should fall between the modes"
        );
    }

    #[test]
    fn otsu_needs_two_distinct_values() {
        let mut hist = [0u32; 256];
        assert_eq!(otsu_threshold(&hist), None);
        hist[100] = 10;
        assert_eq!(otsu_threshold(&hist), None);
        hist[101] = 1;
        assert_eq!(otsu_threshold(&hist), Some(100));
    }

    #[test]
    fn integral_image_matches_direct_sums() {
        let mut gray = GrayImage::new(13, 9);
        for (i, v) in gray.iter_mut().enumerate() {
            *v = (i * 37 % 251) as u8;
        }
        let integral = IntegralImage::new(&gray);
        for rect in [
            Rect {
                x: 0,
                y: 0,
                w: 13,
                h: 9,
            },
            Rect {
                x: 3,
                y: 2,
                w: 5,
                h: 4,
            },
            Rect {
                x: 12,
                y: 8,
                w: 1,
                h: 1,
            },
            Rect {
                x: 10,
                y: 5,
                w: 50,
                h: 50,
            },
        ] {
            let mut direct = 0u64;
            for y in rect.y..(rect.y + rect.h).min(9) {
                for x in rect.x..(rect.x + rect.w).min(13) {
                    direct += gray.get_pixel(x, y).0[0] as u64;
                }
            }
            assert_eq!(integral.sum(rect), direct, "{rect:?}");
        }
        assert_eq!(
            integral.sum(Rect {
                x: 20,
                y: 0,
                w: 5,
                h: 5
            }),
            0
        );
        assert_eq!(
            integral.mean(Rect {
                x: 20,
                y: 0,
                w: 5,
                h: 5
            }),
            None
        );
        assert_eq!(
            integral.mean(Rect {
                x: 0,
                y: 0,
                w: 1,
                h: 1
            }),
            Some(0.0)
        );
    }

    /// White text on a pink bar: luma barely separates them, the minimum
    /// channel separates them cleanly.
    #[test]
    fn min_channel_separates_light_text_from_a_saturated_bar() {
        let bar = Rgba([235, 75, 105, 255]);
        let text = Rgba([250, 185, 195, 255]);
        let mut image = RgbaImage::from_pixel(10, 1, bar);
        image.put_pixel(4, 0, text);
        let region = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 1,
        };
        let min = channel_image(&image, region, Channel::Min);
        assert_eq!(min.get_pixel(0, 0).0[0], 75);
        assert_eq!(min.get_pixel(4, 0).0[0], 185);
        let evidence = text_evidence(&image, region, Channel::Min, Polarity::LightText, 60);
        assert_eq!(evidence.get_pixel(0, 0).0[0], 0, "the bar is background");
        assert_eq!(
            evidence.get_pixel(4, 0).0[0],
            255,
            "the stroke stands out fully"
        );
    }

    #[test]
    fn text_evidence_follows_each_rows_own_background() {
        // Two rows with very different backgrounds, same stroke contrast.
        let mut image = RgbaImage::new(9, 2);
        for x in 0..9 {
            image.put_pixel(x, 0, Rgba([30, 30, 30, 255]));
            image.put_pixel(x, 1, Rgba([150, 150, 150, 255]));
        }
        image.put_pixel(3, 0, Rgba([90, 90, 90, 255]));
        image.put_pixel(3, 1, Rgba([210, 210, 210, 255]));
        let evidence = text_evidence(
            &image,
            Rect {
                x: 0,
                y: 0,
                w: 9,
                h: 2,
            },
            Channel::Luma,
            Polarity::LightText,
            60,
        );
        assert_eq!(evidence.get_pixel(3, 0).0[0], 255);
        assert_eq!(evidence.get_pixel(3, 1).0[0], 255);
        assert_eq!(evidence.get_pixel(0, 1).0[0], 0);
        // The same strokes are no evidence at all of dark text.
        let dark = text_evidence(
            &image,
            Rect {
                x: 0,
                y: 0,
                w: 9,
                h: 2,
            },
            Channel::Luma,
            Polarity::DarkText,
            60,
        );
        assert_eq!(dark.get_pixel(3, 0).0[0], 0);
        assert_eq!(dark.get_pixel(3, 1).0[0], 0);
    }

    #[test]
    fn binarize_keeps_values_strictly_above_the_threshold() {
        let gray = GrayImage::from_raw(4, 1, vec![0, 100, 101, 255]).unwrap();
        assert_eq!(binarize(&gray, 100).as_raw(), &vec![0, 0, 255, 255]);
    }

    #[test]
    fn regions_outside_the_image_yield_empty_images() {
        let image = RgbaImage::new(4, 4);
        let out = channel_image(
            &image,
            Rect {
                x: 9,
                y: 9,
                w: 2,
                h: 2,
            },
            Channel::Luma,
        );
        assert_eq!(out.dimensions(), (0, 0));
        let evidence = text_evidence(
            &image,
            Rect {
                x: 9,
                y: 9,
                w: 2,
                h: 2,
            },
            Channel::Min,
            Polarity::LightText,
            40,
        );
        assert_eq!(evidence.dimensions(), (0, 0));
    }
}
