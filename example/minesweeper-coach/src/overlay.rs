//! The marks laid over the board, as a picture: transparent but for the marks,
//! in premultiplied RGBA — what a layered window on Windows takes, and what
//! can be laid over a screenshot anywhere to look at.
//!
//! Sure cells get a green box (the one the coach is talking about, a bolder
//! one), proven mines a red cross, a wrong flag a magenta slash, the numbers
//! behind the coach's reason a blue ring, and the best guess, when nothing
//! is sure, a yellow box with its chance of a mine written in it. Below the
//! board (above it, near the bottom of the screen) a dark box for the words,
//! which the window writes in with the system's own font.

use image::{Rgba, RgbaImage};

use crate::coach::{Mark, Marks};
use crate::reader::Grid;
use crate::render::glyph;

pub const GREEN: [u8; 3] = [24, 196, 84];
pub const RED: [u8; 3] = [232, 48, 40];
pub const YELLOW: [u8; 3] = [255, 196, 0];
pub const BLUE: [u8; 3] = [32, 136, 255];
pub const MAGENTA: [u8; 3] = [224, 0, 200];
pub const INK: [u8; 3] = [16, 16, 16];
/// The box behind the words: its colour and how opaque.
pub const CAPTION_BOX: ([u8; 3], f32) = ([22, 22, 26], 0.84);

/// Where the overlay sits, and what is where in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    /// The overlay's top left, in the picture the board was read from.
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    /// The board's cells, in the overlay.
    pub grid: Grid,
    /// The box for the words, in the overlay: left, top, right, bottom.
    pub caption: (i32, i32, i32, i32),
    /// The words' height, in pixels.
    pub font: i32,
}

/// The overlay for a board read at `grid` in a picture of `size`: the board with
/// a margin round it, and a box for the words below it — above it when the
/// board is too near the bottom.
pub fn layout(grid: &Grid, size: (u32, u32)) -> Layout {
    let p = grid.pitch;
    let margin = (0.5 * p).max(8.0).round() as i32;
    let font = (0.42 * p).clamp(15.0, 24.0).round() as i32;
    let cap_h = (font as f32 * 2.7).round() as i32 + 12;
    let (bl, bt, br, bb) = grid.bounds();
    let (bl, bt, br, bb) = (
        bl.round() as i32,
        bt.round() as i32,
        br.round() as i32,
        bb.round() as i32,
    );
    let mid = (bl + br) / 2;
    let cap_w = (br - bl).max(22 * font);
    let below = bb + margin / 2 + cap_h <= size.1 as i32;
    let left = (bl - margin).min(mid - cap_w / 2);
    let right = (br + margin).max(mid + cap_w / 2);
    let (top, bottom) = if below {
        (bt - margin, bb + margin / 2 + cap_h)
    } else {
        (bt - margin / 2 - cap_h, bb + margin)
    };
    let cap_top = if below { bb + margin / 2 - top } else { 0 };
    Layout {
        left,
        top,
        width: (right - left) as u32,
        height: (bottom - top) as u32,
        grid: Grid {
            x: grid.x - left as f32,
            y: grid.y - top as f32,
            ..*grid
        },
        caption: (
            mid - cap_w / 2 - left,
            cap_top,
            mid + cap_w / 2 - left,
            cap_top + cap_h,
        ),
        font,
    }
}

/// Premultiplied "over": `c` at opacity `a` on top of what is there.
fn over(px: &mut Rgba<u8>, c: [u8; 3], a: f32) {
    if a <= 0.0 {
        return;
    }
    let a = a.min(1.0);
    let inv = 1.0 - a;
    for (d, &s) in px.0.iter_mut().zip(&c) {
        *d = (s as f32 * a + *d as f32 * inv).round() as u8;
    }
    px.0[3] = (255.0 * a + px.0[3] as f32 * inv).round() as u8;
}

/// Distance from `(x, y)` to a box with rounded corners (negative inside).
fn rounded_box(x: f32, y: f32, (x0, y0, x1, y1): (f32, f32, f32, f32), r: f32) -> f32 {
    let (cx, cy, hw, hh) = (
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        (x1 - x0) / 2.0,
        (y1 - y0) / 2.0,
    );
    let r = r.min(hw).min(hh);
    let (qx, qy) = ((x - cx).abs() - (hw - r), (y - cy).abs() - (hh - r));
    let (ox, oy) = (qx.max(0.0), qy.max(0.0));
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0) - r
}

/// A rounded box: filled at opacity `fill.1`, and a line of width `stroke.2` just inside its edge.
fn rbox(
    img: &mut RgbaImage,
    b: (f32, f32, f32, f32),
    r: f32,
    fill: Option<([u8; 3], f32)>,
    stroke: Option<([u8; 3], f32, f32)>,
) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    for y in (b.1.floor() as i64).max(0)..(b.3.ceil() as i64 + 1).min(h) {
        for x in (b.0.floor() as i64).max(0)..(b.2.ceil() as i64 + 1).min(w) {
            let d = rounded_box(x as f32 + 0.5, y as f32 + 0.5, b, r);
            let px = img.get_pixel_mut(x as u32, y as u32);
            if let Some((c, a)) = fill {
                over(px, c, a * (0.5 - d).clamp(0.0, 1.0));
            }
            if let Some((c, a, sw)) = stroke {
                over(px, c, a * (0.5 - d).min(d + sw + 0.5).clamp(0.0, 1.0));
            }
        }
    }
}

/// A straight stroke of width `sw` with round ends.
fn stroke(
    img: &mut RgbaImage,
    (ax, ay): (f32, f32),
    (bx, by): (f32, f32),
    sw: f32,
    c: [u8; 3],
    a: f32,
) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, x1) = (
        (ax.min(bx) - sw).floor() as i64,
        (ax.max(bx) + sw).ceil() as i64,
    );
    let (y0, y1) = (
        (ay.min(by) - sw).floor() as i64,
        (ay.max(by) + sw).ceil() as i64,
    );
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = (dx * dx + dy * dy).max(1e-6);
    for y in y0.max(0)..(y1 + 1).min(h) {
        for x in x0.max(0)..(x1 + 1).min(w) {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
            let (qx, qy) = (ax + t * dx - px, ay + t * dy - py);
            let d = (qx * qx + qy * qy).sqrt() - sw / 2.0;
            over(
                img.get_pixel_mut(x as u32, y as u32),
                c,
                a * (0.5 - d).clamp(0.0, 1.0),
            );
        }
    }
}

/// Digits (and `%`, `<`) in the 5x7 font, `h` pixels tall, centred on `(cx, cy)`.
fn text(img: &mut RgbaImage, s: &str, cx: f32, cy: f32, h: f32, c: [u8; 3]) {
    let u = h / 7.0;
    let n = s.chars().count() as f32;
    let width = n * 5.0 * u + (n - 1.0).max(0.0) * u;
    let (x0, y0) = (cx - width / 2.0, cy - h / 2.0);
    for (k, ch) in s.chars().enumerate() {
        let Some(g) = glyph(ch) else { continue };
        let gx = x0 + k as f32 * 6.0 * u;
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits & (0b10000 >> col) != 0 {
                    let (ax, ay) = (gx + col as f32 * u, y0 + row as f32 * u);
                    rbox(img, (ax, ay, ax + u, ay + u), 0.0, Some((c, 1.0)), None);
                }
            }
        }
    }
}

/// How a chance of a mine is written on the guess: "12%", or "<1%".
pub fn percent_label(risk: f64) -> String {
    let n = (risk * 100.0).round() as u32;
    if n < 1 { "<1%".into() } else { format!("{n}%") }
}

/// The overlay's picture: every mark, and the box for the words.
pub fn paint(l: &Layout, marks: &Marks) -> RgbaImage {
    let mut img = RgbaImage::new(l.width, l.height);
    let g = &l.grid;
    let p = g.pitch;
    let line = (0.075 * p).max(2.0);
    for &((cx, cy), m) in &marks.cells {
        let (x0, y0) = g.cell_origin(cx, cy);
        let inset = |k: f32| {
            (
                x0 + k * p,
                y0 + k * p,
                x0 + (1.0 - k) * p,
                y0 + (1.0 - k) * p,
            )
        };
        let r = 0.16 * p;
        match m {
            Mark::Safe => rbox(
                &mut img,
                inset(0.1),
                r,
                Some((GREEN, 0.2)),
                Some((GREEN, 0.85, line)),
            ),
            Mark::Next => {
                rbox(
                    &mut img,
                    inset(0.0),
                    r,
                    Some((GREEN, 0.72)),
                    Some((GREEN, 1.0, line * 1.7)),
                );
                let c = (x0 + 0.5 * p, y0 + 0.5 * p);
                let d = 0.13 * p;
                rbox(
                    &mut img,
                    (c.0 - d, c.1 - d, c.0 + d, c.1 + d),
                    d,
                    Some(([255, 255, 255], 0.95)),
                    None,
                );
            }
            Mark::Start => rbox(
                &mut img,
                inset(0.08),
                r,
                Some((GREEN, 0.14)),
                Some((GREEN, 0.8, line)),
            ),
            Mark::Mine => {
                rbox(&mut img, inset(0.08), r, Some((RED, 0.22)), None);
                let (a, b) = (0.3 * p, 0.7 * p);
                stroke(
                    &mut img,
                    (x0 + a, y0 + a),
                    (x0 + b, y0 + b),
                    line * 1.3,
                    RED,
                    0.95,
                );
                stroke(
                    &mut img,
                    (x0 + b, y0 + a),
                    (x0 + a, y0 + b),
                    line * 1.3,
                    RED,
                    0.95,
                );
            }
            Mark::WrongFlag => {
                rbox(
                    &mut img,
                    inset(0.04),
                    r,
                    None,
                    Some((MAGENTA, 1.0, line * 1.4)),
                );
                stroke(
                    &mut img,
                    (x0 + 0.2 * p, y0 + 0.8 * p),
                    (x0 + 0.8 * p, y0 + 0.2 * p),
                    line * 1.2,
                    MAGENTA,
                    0.95,
                );
            }
            Mark::Guess(risk) => {
                rbox(
                    &mut img,
                    inset(0.04),
                    r,
                    Some((YELLOW, 0.55)),
                    Some((YELLOW, 1.0, line * 1.3)),
                );
                text(
                    &mut img,
                    &percent_label(risk),
                    x0 + 0.5 * p,
                    y0 + 0.5 * p,
                    (0.3 * p).max(7.0),
                    INK,
                );
            }
            Mark::Reason => rbox(
                &mut img,
                inset(0.0),
                0.5 * p,
                None,
                Some((BLUE, 0.95, line)),
            ),
        }
    }
    if !marks.caption.is_empty() {
        let (a, b, c, d) = l.caption;
        rbox(
            &mut img,
            (a as f32, b as f32, c as f32, d as f32),
            (d - b) as f32 * 0.3,
            Some(CAPTION_BOX),
            None,
        );
    }
    img
}

/// A screenshot with the overlay laid on it at `(left, top)`: to look at the marks.
pub fn composite(screen: &RgbaImage, overlay: &RgbaImage, left: i32, top: i32) -> RgbaImage {
    let mut out = screen.clone();
    for (x, y, o) in overlay.enumerate_pixels() {
        let (sx, sy) = (left + x as i32, top + y as i32);
        if sx < 0 || sy < 0 || sx >= out.width() as i32 || sy >= out.height() as i32 || o.0[3] == 0
        {
            continue;
        }
        let px = out.get_pixel_mut(sx as u32, sy as u32);
        let inv = 1.0 - o.0[3] as f32 / 255.0;
        for k in 0..3 {
            px.0[k] = (o.0[k] as f32 + px.0[k] as f32 * inv).round().min(255.0) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid {
            x: 100.0,
            y: 80.0,
            pitch: 40.0,
            cols: 9,
            rows: 9,
        }
    }

    #[test]
    fn the_overlay_covers_the_board_and_its_words() {
        let l = layout(&grid(), (1920, 1080));
        assert!(l.left <= 80 && l.top <= 60, "{l:?}");
        let (x1, y1) = (l.left + l.width as i32, l.top + l.height as i32);
        assert!(
            x1 >= 100 + 360 + 20 && y1 >= 80 + 360 + (l.caption.3 - l.caption.1),
            "{l:?}"
        );
        assert!(
            l.caption.1 > (l.grid.y + 360.0) as i32,
            "the words go below: {l:?}"
        );
        // Near the bottom of the screen they go above.
        let high = layout(&Grid { y: 700.0, ..grid() }, (1920, 1080));
        assert!(high.caption.3 <= high.grid.y as i32, "{high:?}");
    }

    #[test]
    fn marks_are_drawn_where_the_cells_are_and_nowhere_else() {
        let l = layout(&grid(), (1920, 1080));
        let marks = Marks {
            cells: vec![
                ((0, 0), Mark::Next),
                ((2, 0), Mark::Mine),
                ((4, 4), Mark::Guess(0.12)),
                ((5, 5), Mark::Reason),
            ],
            caption: "Hello".into(),
        };
        let img = paint(&l, &marks);
        let at = |cx: usize, cy: usize, fx: f32, fy: f32| {
            let (x, y) = l.grid.cell_origin(cx, cy);
            *img.get_pixel(
                (x + fx * l.grid.pitch) as u32,
                (y + fy * l.grid.pitch) as u32,
            )
        };
        // The sure cell's box: green and opaque at its edge, half see-through inside.
        let edge = at(0, 0, 0.05, 0.5);
        assert!(
            edge.0[3] > 240 && edge.0[1] > 150 && edge.0[0] < 80,
            "{edge:?}"
        );
        let inside = at(0, 0, 0.5, 0.5);
        assert!((100..200).contains(&inside.0[3]), "{inside:?}");
        // The mine's cross runs through its centre.
        assert!(at(2, 0, 0.5, 0.5).0[0] > 180);
        // The guess has its number written in it, in dark ink.
        let dark = (0..40)
            .flat_map(|y| (0..40).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let (ox, oy) = l.grid.cell_origin(4, 4);
                let p = img.get_pixel(ox as u32 + x, oy as u32 + y).0;
                p[3] > 200 && p[0] < 60 && p[1] < 60
            });
        assert!(dark.count() > 20);
        // An unmarked cell stays clear, and so does the margin.
        assert_eq!(at(8, 8, 0.5, 0.5).0[3], 0);
        assert_eq!(img.get_pixel(2, 2).0[3], 0);
        // The box for the words.
        let (a, b, c, d) = l.caption;
        assert!(img.get_pixel(((a + c) / 2) as u32, ((b + d) / 2) as u32).0[3] > 200);
    }

    #[test]
    fn a_risk_is_written_as_a_percentage() {
        assert_eq!(percent_label(0.123), "12%");
        assert_eq!(percent_label(0.5), "50%");
        assert_eq!(percent_label(0.001), "<1%");
    }
}
