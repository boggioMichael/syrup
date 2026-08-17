//! Debug-overlay drawing primitives: rectangles and a small bitmap font.
//!
//! Annotating a frame with what a detector saw should not require a font
//! file and a text-shaping stack, so this module carries a compact 5x7
//! glyph table instead. Text is uppercased before lookup, which halves the
//! table for no practical loss on debug overlays.
//!
//! Each glyph is 5 columns stored as one byte per column; bit `n` of a
//! column is the pixel in row `n`, top to bottom.

use image::{Rgba, RgbaImage};

/// Draw a rectangle outline onto a mutable RGBA image, clipped to its bounds.
pub fn draw_rect(
    img: &mut RgbaImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    color: Rgba<u8>,
    thickness: u32,
) {
    let (iw, ih) = (img.width(), img.height());
    for t in 0..thickness {
        let yy_top = y.saturating_add(t);
        let yy_bottom = y.saturating_add(h.saturating_sub(1)).saturating_sub(t);
        for px in x..(x + w).min(iw) {
            if yy_top < ih {
                img.put_pixel(px, yy_top, color);
            }
            if yy_bottom < ih {
                img.put_pixel(px, yy_bottom, color);
            }
        }
        let xx_left = x.saturating_add(t);
        let xx_right = x.saturating_add(w.saturating_sub(1)).saturating_sub(t);
        for py in y..(y + h).min(ih) {
            if xx_left < iw {
                img.put_pixel(xx_left, py, color);
            }
            if xx_right < iw {
                img.put_pixel(xx_right, py, color);
            }
        }
    }
}

/// Width of a glyph in pixels, excluding inter-character spacing.
pub const GLYPH_WIDTH: u32 = 5;
/// Height of a glyph in pixels.
pub const GLYPH_HEIGHT: u32 = 7;
/// Pixels between glyph cells.
pub const GLYPH_SPACING: u32 = 1;

/// Column bitmaps for a character, or `None` when unmapped.
fn glyph(ch: char) -> Option<[u8; 5]> {
    let bitmap = match ch.to_ascii_uppercase() {
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00],
        '0' => [0x3E, 0x51, 0x49, 0x45, 0x3E],
        '1' => [0x00, 0x42, 0x7F, 0x40, 0x00],
        '2' => [0x42, 0x61, 0x51, 0x49, 0x46],
        '3' => [0x21, 0x41, 0x45, 0x4B, 0x31],
        '4' => [0x18, 0x14, 0x12, 0x7F, 0x10],
        '5' => [0x27, 0x45, 0x45, 0x45, 0x39],
        '6' => [0x3C, 0x4A, 0x49, 0x49, 0x30],
        '7' => [0x01, 0x71, 0x09, 0x05, 0x03],
        '8' => [0x36, 0x49, 0x49, 0x49, 0x36],
        '9' => [0x06, 0x49, 0x49, 0x29, 0x1E],
        'A' => [0x7E, 0x11, 0x11, 0x11, 0x7E],
        'B' => [0x7F, 0x49, 0x49, 0x49, 0x36],
        'C' => [0x3E, 0x41, 0x41, 0x41, 0x22],
        'D' => [0x7F, 0x41, 0x41, 0x22, 0x1C],
        'E' => [0x7F, 0x49, 0x49, 0x49, 0x41],
        'F' => [0x7F, 0x09, 0x09, 0x09, 0x01],
        'G' => [0x3E, 0x41, 0x49, 0x49, 0x7A],
        'H' => [0x7F, 0x08, 0x08, 0x08, 0x7F],
        'I' => [0x00, 0x41, 0x7F, 0x41, 0x00],
        'J' => [0x20, 0x40, 0x41, 0x3F, 0x01],
        'K' => [0x7F, 0x08, 0x14, 0x22, 0x41],
        'L' => [0x7F, 0x40, 0x40, 0x40, 0x40],
        'M' => [0x7F, 0x02, 0x0C, 0x02, 0x7F],
        'N' => [0x7F, 0x04, 0x08, 0x10, 0x7F],
        'O' => [0x3E, 0x41, 0x41, 0x41, 0x3E],
        'P' => [0x7F, 0x09, 0x09, 0x09, 0x06],
        'Q' => [0x3E, 0x41, 0x51, 0x21, 0x5E],
        'R' => [0x7F, 0x09, 0x19, 0x29, 0x46],
        'S' => [0x46, 0x49, 0x49, 0x49, 0x31],
        'T' => [0x01, 0x01, 0x7F, 0x01, 0x01],
        'U' => [0x3F, 0x40, 0x40, 0x40, 0x3F],
        'V' => [0x1F, 0x20, 0x40, 0x20, 0x1F],
        'W' => [0x3F, 0x40, 0x38, 0x40, 0x3F],
        'X' => [0x63, 0x14, 0x08, 0x14, 0x63],
        'Y' => [0x07, 0x08, 0x70, 0x08, 0x07],
        'Z' => [0x61, 0x51, 0x49, 0x45, 0x43],
        '.' => [0x00, 0x60, 0x60, 0x00, 0x00],
        ',' => [0x00, 0x50, 0x30, 0x00, 0x00],
        ':' => [0x00, 0x36, 0x36, 0x00, 0x00],
        ';' => [0x00, 0x56, 0x36, 0x00, 0x00],
        '/' => [0x20, 0x10, 0x08, 0x04, 0x02],
        '\\' => [0x02, 0x04, 0x08, 0x10, 0x20],
        '%' => [0x23, 0x13, 0x08, 0x64, 0x62],
        '(' => [0x00, 0x1C, 0x22, 0x41, 0x00],
        ')' => [0x00, 0x41, 0x22, 0x1C, 0x00],
        '[' => [0x00, 0x7F, 0x41, 0x41, 0x00],
        ']' => [0x00, 0x41, 0x41, 0x7F, 0x00],
        '-' => [0x08, 0x08, 0x08, 0x08, 0x08],
        '_' => [0x40, 0x40, 0x40, 0x40, 0x40],
        '=' => [0x14, 0x14, 0x14, 0x14, 0x14],
        '+' => [0x08, 0x08, 0x3E, 0x08, 0x08],
        '*' => [0x14, 0x08, 0x3E, 0x08, 0x14],
        '#' => [0x14, 0x7F, 0x14, 0x7F, 0x14],
        '?' => [0x02, 0x01, 0x51, 0x09, 0x06],
        '!' => [0x00, 0x00, 0x5F, 0x00, 0x00],
        '<' => [0x08, 0x14, 0x22, 0x41, 0x00],
        '>' => [0x00, 0x41, 0x22, 0x14, 0x08],
        '\'' => [0x00, 0x07, 0x00, 0x00, 0x00],
        _ => return None,
    };
    Some(bitmap)
}

/// Pixel width of `text` at `scale`, including inter-character spacing.
pub fn text_width(text: &str, scale: u32) -> u32 {
    let count = text.chars().count() as u32;
    if count == 0 {
        return 0;
    }
    // Trailing spacing is not part of the drawn text.
    (count * (GLYPH_WIDTH + GLYPH_SPACING) * scale).saturating_sub(GLYPH_SPACING * scale)
}

/// Pixel height of a line at `scale`.
pub fn text_height(scale: u32) -> u32 {
    GLYPH_HEIGHT * scale
}

/// Call `plot` for every lit pixel of `text` drawn at `(x, y)`.
///
/// Kept as a callback rather than taking an image so the caller decides how
/// pixels land (bounds checks, blending, target buffer) without this module
/// depending on an image type.
pub fn draw_text(text: &str, x: i64, y: i64, scale: u32, mut plot: impl FnMut(i64, i64)) {
    let scale = scale.max(1) as i64;
    let advance = (GLYPH_WIDTH + GLYPH_SPACING) as i64 * scale;

    for (index, ch) in text.chars().enumerate() {
        let Some(bitmap) = glyph(ch) else {
            continue;
        };
        let origin_x = x + index as i64 * advance;
        for (column_index, column) in bitmap.iter().enumerate() {
            for row in 0..GLYPH_HEIGHT {
                if column & (1 << row) == 0 {
                    continue;
                }
                // Expand one font pixel into a scale x scale block.
                for dy in 0..scale {
                    for dx in 0..scale {
                        plot(
                            origin_x + column_index as i64 * scale + dx,
                            y + row as i64 * scale + dy,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_rect_strokes_the_outline_and_clips_to_bounds() {
        let black = Rgba([0, 0, 0, 255]);
        let red = Rgba([255, 0, 0, 255]);
        let mut img = RgbaImage::from_pixel(20, 20, black);

        // Partially off-image: must not panic, must stroke what is visible.
        draw_rect(&mut img, 15, 15, 10, 10, red, 1);
        assert_eq!(*img.get_pixel(15, 15), red);

        draw_rect(&mut img, 2, 2, 5, 5, red, 1);
        assert_eq!(*img.get_pixel(2, 2), red); // corner
        assert_eq!(*img.get_pixel(4, 4), black); // interior untouched
    }

    #[test]
    fn known_characters_have_glyphs() {
        for ch in "ABCXYZ0189:/%.,()-".chars() {
            assert!(glyph(ch).is_some(), "expected a glyph for {ch:?}");
        }
    }

    #[test]
    fn lowercase_falls_back_to_uppercase_glyph() {
        assert_eq!(glyph('a'), glyph('A'));
        assert_eq!(glyph('z'), glyph('Z'));
    }

    #[test]
    fn unmapped_characters_are_skipped_not_panicking() {
        assert!(glyph('☃').is_none());
        // Drawing must survive characters outside the table.
        let mut plotted = 0;
        draw_text("☃", 0, 0, 1, |_, _| plotted += 1);
        assert_eq!(plotted, 0);
    }

    #[test]
    fn text_width_accounts_for_spacing() {
        assert_eq!(text_width("", 1), 0);
        assert_eq!(text_width("A", 1), GLYPH_WIDTH);
        // Two glyphs plus one gap between them.
        assert_eq!(text_width("AB", 1), GLYPH_WIDTH * 2 + GLYPH_SPACING);
        assert_eq!(text_width("A", 2), GLYPH_WIDTH * 2);
    }

    #[test]
    fn space_draws_nothing_but_still_advances() {
        let mut plotted = 0;
        draw_text(" ", 0, 0, 1, |_, _| plotted += 1);
        assert_eq!(plotted, 0);
        assert_eq!(text_width(" ", 1), GLYPH_WIDTH);
    }

    #[test]
    fn scale_expands_pixel_count_quadratically() {
        let mut small = 0;
        draw_text("A", 0, 0, 1, |_, _| small += 1);
        let mut large = 0;
        draw_text("A", 0, 0, 2, |_, _| large += 1);
        assert_eq!(large, small * 4);
    }

    #[test]
    fn glyphs_stay_within_their_cell() {
        let mut max_x = 0i64;
        let mut max_y = 0i64;
        draw_text("W", 0, 0, 1, |x, y| {
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        });
        assert!(max_x < GLYPH_WIDTH as i64);
        assert!(max_y < GLYPH_HEIGHT as i64);
    }
}
