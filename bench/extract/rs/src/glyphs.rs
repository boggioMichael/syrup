//! Syrup's glyph reader, as MapleSyrup reads the modern HUD's numbers with
//! it (`ms/src/sight/numbers.rs`: the default options, `min_score` 0.95),
//! learning the font from the same training lines as the explicit readers.
//! Pilot A's reader 4, "Syrup today".

use image::RgbaImage;
use syrup::geometry::Rect;
use syrup::glyphs::{GlyphOptions, GlyphSet, LearnError};

/// How surely a glyph must match, MapleSyrup's for the modern HUD font.
pub const MIN_SCORE: f32 = 0.95;

/// An RGB line as the RGBA frame MapleSyrup hands Syrup.
pub fn rgba(px: &[u8], w: usize, rows: usize) -> RgbaImage {
    let mut img = RgbaImage::new(w as u32, rows as u32);
    for (to, p) in img.as_mut().chunks_exact_mut(4).zip(px.chunks_exact(3)) {
        to.copy_from_slice(&[p[0], p[1], p[2], 255]);
    }
    img
}

pub fn whole(w: usize, rows: usize) -> Rect {
    Rect { x: 0, y: 0, w: w as u32, h: rows as u32 }
}

pub struct Glyphs {
    pub set: GlyphSet,
    /// Lines learned from, and those it refused to learn, by why.
    pub learned: usize,
    pub refused: [usize; 4],
}

impl Glyphs {
    /// The font learned from labelled lines, with MapleSyrup's options
    /// (`min_score` = [`MIN_SCORE`]) or, with `min_score` given, the
    /// default options at that score.
    pub fn learn<'a>(lines: impl Iterator<Item = (&'a [u8], &'a str)>, w: usize, rows: usize, min_score: f32) -> Glyphs {
        let mut set = GlyphSet::new(GlyphOptions { min_score, ..GlyphOptions::default() });
        let (mut learned, mut refused) = (0, [0usize; 4]);
        for (px, label) in lines {
            match set.learn(&rgba(px, w, rows), whole(w, rows), label) {
                Ok(_) => learned += 1,
                Err(LearnError::NoText) => refused[0] += 1,
                Err(LearnError::GlyphCountMismatch { .. }) => refused[1] += 1,
                Err(LearnError::Inconsistent { .. }) => refused[2] += 1,
                Err(LearnError::DoesNotReadBack { .. }) => refused[3] += 1,
            }
        }
        Glyphs { set, learned, refused }
    }

    /// The line's text (without the spaces the reader marks wide gaps
    /// with), or None when it refuses: some glyph not surely one character.
    pub fn read(&self, img: &RgbaImage, region: Rect) -> Option<String> {
        self.set.read(img, region).value.map(|r| r.text.chars().filter(|c| !c.is_whitespace()).collect())
    }
}
