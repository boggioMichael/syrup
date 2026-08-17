//! Color-space conversion and the pixel predicates shared by detectors.

use image::Rgba;

/// HSV tuple with H in `[0, 360)`, S and V in `[0, 1]`.
pub type Hsv = (f32, f32, f32);

/// Convert RGB to HSV. H in degrees `[0, 360)`, S and V in `[0, 1]`.
pub fn hsv_from_rgb(r: u8, g: u8, b: u8) -> Hsv {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == rf {
        60.0 * ((gf - bf) / delta % 6.0)
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s.clamp(0.0, 1.0), max.clamp(0.0, 1.0))
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
    let (h, s, v) = hsv_from_rgb(pixel[0], pixel[1], pixel[2]);
    let alpha = pixel[3] as f32 / 255.0;
    alpha >= 0.5
        && hue_in_range(h, hue_range.0, hue_range.1)
        && s >= min_saturation
        && v >= min_value
}

/// Does this pixel plausibly belong to rendered UI text?
///
/// UI text is bright, fairly desaturated, and opaque; the thresholds were
/// tuned against real interface captures.
pub fn is_text_pixel(pixel: &Rgba<u8>) -> bool {
    let (_, s, v) = hsv_from_rgb(pixel[0], pixel[1], pixel[2]);
    let brightness =
        0.299 * (pixel[0] as f32) + 0.587 * (pixel[1] as f32) + 0.114 * (pixel[2] as f32);
    let is_bright = brightness >= 90.0;
    let is_desaturated = s <= 0.55;
    let is_light = v >= 0.45;
    alpha_is_high(pixel) && is_bright && is_desaturated && is_light
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

    #[test]
    fn transparent_pixels_never_match_color() {
        let red_transparent = Rgba([255, 0, 0, 10]);
        assert!(!is_color_pixel(&red_transparent, (340.0, 30.0), 0.3, 0.3));
        let red_opaque = Rgba([255, 0, 0, 255]);
        assert!(is_color_pixel(&red_opaque, (340.0, 30.0), 0.3, 0.3));
    }
}
