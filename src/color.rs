//! Color-space conversion and the pixel predicates shared by detectors.

use image::Rgba;

/// HSV tuple with H in `[0, 360)`, S and V in `[0, 1]`.
pub type Hsv = (f32, f32, f32);

/// Convert RGB to HSV. H in degrees `[0, 360)`, S and V in `[0, 1]`.
pub fn hsv_from_rgb(r: u8, g: u8, b: u8) -> Hsv {
    let parts = HsvParts::new(r, g, b);
    (parts.hue(), parts.saturation(), parts.value())
}

/// The intermediate values of an RGB→HSV conversion, so predicates can test
/// value and saturation — cheap — and compute the hue, which needs a
/// division and branches, only for the few pixels that pass both. On a
/// typical frame most pixels are too dark or too grey to match any colour.
///
/// Every quantity is computed with exactly the arithmetic [`hsv_from_rgb`]
/// uses, so the lazy path and the full conversion can never disagree.
struct HsvParts {
    rf: f32,
    gf: f32,
    bf: f32,
    max: f32,
    delta: f32,
}

impl HsvParts {
    #[inline]
    fn new(r: u8, g: u8, b: u8) -> Self {
        let rf = r as f32 / 255.0;
        let gf = g as f32 / 255.0;
        let bf = b as f32 / 255.0;
        let max = rf.max(gf).max(bf);
        let min = rf.min(gf).min(bf);
        Self {
            rf,
            gf,
            bf,
            max,
            delta: max - min,
        }
    }

    #[inline]
    fn value(&self) -> f32 {
        self.max.clamp(0.0, 1.0)
    }

    #[inline]
    fn saturation(&self) -> f32 {
        let s = if self.max == 0.0 {
            0.0
        } else {
            self.delta / self.max
        };
        s.clamp(0.0, 1.0)
    }

    #[inline]
    fn hue(&self) -> f32 {
        let (rf, gf, bf, max, delta) = (self.rf, self.gf, self.bf, self.max, self.delta);
        let h = if delta == 0.0 {
            0.0
        } else if max == rf {
            60.0 * ((gf - bf) / delta % 6.0)
        } else if max == gf {
            60.0 * ((bf - rf) / delta + 2.0)
        } else {
            60.0 * ((rf - gf) / delta + 4.0)
        };
        if h < 0.0 { h + 360.0 } else { h }
    }
}

/// Is `hue` within `[min, max]`, where a range with `min > max` wraps
/// around 360° (e.g. `(340, 30)` covers red)?
fn hue_in_range(hue: f32, min: f32, max: f32) -> bool {
    if min <= max {
        hue >= min && hue <= max
    } else {
        hue >= min || hue <= max
    }
}

/// Does this pixel match a saturated color in the given hue range?
/// Near-transparent pixels never match.
pub fn is_color_pixel(
    pixel: &Rgba<u8>,
    hue_range: (f32, f32),
    min_saturation: f32,
    min_value: f32,
) -> bool {
    let alpha = pixel[3] as f32 / 255.0;
    if alpha < 0.5 {
        return false;
    }
    // Cheapest rejections first; the hue is computed last and only when the
    // pixel is already bright and saturated enough to matter.
    let parts = HsvParts::new(pixel[0], pixel[1], pixel[2]);
    parts.value() >= min_value
        && parts.saturation() >= min_saturation
        && hue_in_range(parts.hue(), hue_range.0, hue_range.1)
}

/// Does this pixel plausibly belong to rendered UI text?
///
/// UI text is bright, fairly desaturated, and opaque; the thresholds were
/// tuned against real interface captures.
pub fn is_text_pixel(pixel: &Rgba<u8>) -> bool {
    if !alpha_is_high(pixel) {
        return false;
    }
    let brightness =
        0.299 * (pixel[0] as f32) + 0.587 * (pixel[1] as f32) + 0.114 * (pixel[2] as f32);
    if brightness < 90.0 {
        return false;
    }
    // The hue plays no part here, so it is never computed.
    let parts = HsvParts::new(pixel[0], pixel[1], pixel[2]);
    parts.saturation() <= 0.55 && parts.value() >= 0.45
}

/// Is this pixel opaque enough to be foreground rather than a blend edge?
fn alpha_is_high(pixel: &Rgba<u8>) -> bool {
    pixel[3] as f32 / 255.0 >= 0.45
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_colors_convert_correctly() {
        assert_eq!(hsv_from_rgb(255, 0, 0), (0.0, 1.0, 1.0));
        assert_eq!(hsv_from_rgb(0, 255, 0), (120.0, 1.0, 1.0));
        assert_eq!(hsv_from_rgb(0, 0, 255), (240.0, 1.0, 1.0));
        assert_eq!(hsv_from_rgb(255, 255, 0), (60.0, 1.0, 1.0));
    }

    #[test]
    fn grayscale_has_no_saturation() {
        let (_, s, v) = hsv_from_rgb(128, 128, 128);
        assert_eq!(s, 0.0);
        assert!((v - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn hue_ranges_wrap_around_zero() {
        assert!(hue_in_range(350.0, 340.0, 30.0));
        assert!(hue_in_range(10.0, 340.0, 30.0));
        assert!(!hue_in_range(180.0, 340.0, 30.0));
        assert!(hue_in_range(180.0, 100.0, 200.0));
    }

    /// The original, eager implementation, kept to prove the lazy predicates
    /// accept exactly the same pixels.
    fn reference_is_color_pixel(
        pixel: &Rgba<u8>,
        hue_range: (f32, f32),
        min_saturation: f32,
        min_value: f32,
    ) -> bool {
        let (h, s, v) = hsv_from_rgb(pixel[0], pixel[1], pixel[2]);
        let alpha = pixel[3] as f32 / 255.0;
        alpha >= 0.5
            && hue_in_range(h, hue_range.0, hue_range.1)
            && s >= min_saturation
            && v >= min_value
    }

    fn reference_is_text_pixel(pixel: &Rgba<u8>) -> bool {
        let (_, s, v) = hsv_from_rgb(pixel[0], pixel[1], pixel[2]);
        let brightness =
            0.299 * (pixel[0] as f32) + 0.587 * (pixel[1] as f32) + 0.114 * (pixel[2] as f32);
        alpha_is_high(pixel) && brightness >= 90.0 && s <= 0.55 && v >= 0.45
    }

    #[test]
    fn lazy_predicates_match_the_eager_reference_exactly() {
        // Every 5th value per channel plus both extremes covers the boundaries
        // of all thresholds used below without taking 16M iterations.
        let levels: Vec<u8> = (0..=255u16)
            .step_by(5)
            .map(|v| v as u8)
            .chain([255])
            .collect();
        let ranges = [
            ((340.0, 30.0), 0.35, 0.30),
            ((190.0, 250.0), 0.30, 0.30),
            ((40.0, 80.0), 0.25, 0.25),
        ];
        for &r in &levels {
            for &g in &levels {
                for &b in &levels {
                    for alpha in [0u8, 127, 128, 255] {
                        let pixel = Rgba([r, g, b, alpha]);
                        for (hue, s, v) in ranges {
                            assert_eq!(
                                is_color_pixel(&pixel, hue, s, v),
                                reference_is_color_pixel(&pixel, hue, s, v),
                                "{pixel:?} {hue:?}"
                            );
                        }
                        assert_eq!(
                            is_text_pixel(&pixel),
                            reference_is_text_pixel(&pixel),
                            "{pixel:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn transparent_pixels_never_match_color() {
        let red_transparent = Rgba([255, 0, 0, 10]);
        assert!(!is_color_pixel(&red_transparent, (340.0, 30.0), 0.3, 0.3));
        let red_opaque = Rgba([255, 0, 0, 255]);
        assert!(is_color_pixel(&red_opaque, (340.0, 30.0), 0.3, 0.3));
    }
}
