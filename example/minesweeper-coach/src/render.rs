//! Boards drawn in the classic look (grey bevelled cells, coloured numbers) —
//! the look of Windows' Minesweeper and of minesweeper.online's default skin,
//! drawn here from scratch — so that the reader can be tested on pictures
//! whose every cell is known, at any size, and with a page around them.

use image::{Rgba, RgbaImage};

use crate::board::{Board, Cell};
use crate::reader::Grid;

pub const FILL: [u8; 3] = [198, 198, 198];
pub const LINE: [u8; 3] = [128, 128, 128];
pub const WHITE: [u8; 3] = [255, 255, 255];
pub const RED: [u8; 3] = [255, 0, 0];
/// The numbers' colours, 1 to 8 (index 0 unused).
pub const DIGIT_COLORS: [[u8; 3]; 9] = [
    [0, 0, 0],
    [0, 0, 247],
    [0, 119, 0],
    [236, 0, 0],
    [0, 0, 128],
    [128, 0, 0],
    [0, 128, 128],
    [0, 0, 0],
    [112, 112, 112],
];

/// 5x7 glyphs for 0-9, row by row, the top bit on the left.
const GLYPHS: [[u8; 7]; 10] = [
    [
        0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
    ],
    [
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ],
    [
        0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
    ],
    [
        0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
    ],
    [
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
    ],
    [
        0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
    ],
    [
        0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
    ],
    [
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
    ],
];

/// 5x7 glyphs for the few other characters the overlay writes.
const PERCENT: [u8; 7] = [
    0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
];
const LESS: [u8; 7] = [
    0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010,
];

/// The 5x7 glyph of a digit, `%` or `<`: rows from the top, the top bit on the left.
pub fn glyph(c: char) -> Option<[u8; 7]> {
    match c {
        '0'..='9' => Some(GLYPHS[c as usize - '0' as usize]),
        '%' => Some(PERCENT),
        '<' => Some(LESS),
        _ => None,
    }
}

fn put(img: &mut RgbaImage, x: i64, y: i64, c: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, Rgba([c[0], c[1], c[2], 255]));
    }
}

pub fn fill_rect(img: &mut RgbaImage, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 3]) {
    for y in y0..y1 {
        for x in x0..x1 {
            put(img, x, y, c);
        }
    }
}

/// A digit 0-9 in a box of `w` x `h` pixels at `(x0, y0)`.
pub fn draw_digit(img: &mut RgbaImage, d: usize, x0: i64, y0: i64, w: i64, h: i64, c: [u8; 3]) {
    let g = GLYPHS[d.min(9)];
    for (gy, bits) in g.iter().enumerate() {
        for gx in 0..5 {
            if bits & (0b10000 >> gx) != 0 {
                let (ax, bx) = (x0 + gx as i64 * w / 5, x0 + (gx as i64 + 1) * w / 5);
                let (ay, by) = (y0 + gy as i64 * h / 7, y0 + (gy as i64 + 1) * h / 7);
                fill_rect(img, ax, ay, bx.max(ax + 1), by.max(ay + 1), c);
            }
        }
    }
}

fn closed(img: &mut RgbaImage, x0: i64, y0: i64, s: i64, inner: [u8; 3]) {
    let b = ((s as f64 * 0.12).round() as i64).max(1);
    fill_rect(img, x0, y0, x0 + s, y0 + s, inner);
    for i in 0..s {
        for j in 0..b {
            // top and left: white; bottom and right: grey (the corners split on the diagonal)
            put(img, x0 + i, y0 + j, if i < s - j { WHITE } else { LINE });
            put(img, x0 + j, y0 + i, if i < s - j { WHITE } else { LINE });
            put(
                img,
                x0 + i,
                y0 + s - 1 - j,
                if i > j { LINE } else { WHITE },
            );
            put(
                img,
                x0 + s - 1 - j,
                y0 + i,
                if i > j { LINE } else { WHITE },
            );
        }
    }
}

fn opened(img: &mut RgbaImage, x0: i64, y0: i64, s: i64, inner: [u8; 3]) {
    let t = ((s as f64 * 0.06).round() as i64).max(1);
    fill_rect(img, x0, y0, x0 + s, y0 + s, inner);
    fill_rect(img, x0, y0, x0 + s, y0 + t, LINE);
    fill_rect(img, x0, y0, x0 + t, y0 + s, LINE);
}

fn flag(img: &mut RgbaImage, x0: i64, y0: i64, s: i64) {
    let f = |v: f64| (v * s as f64).round() as i64;
    // pole and base
    fill_rect(
        img,
        x0 + f(0.52),
        y0 + f(0.2),
        x0 + f(0.58),
        y0 + f(0.7),
        [0, 0, 0],
    );
    fill_rect(
        img,
        x0 + f(0.38),
        y0 + f(0.66),
        x0 + f(0.68),
        y0 + f(0.73),
        [0, 0, 0],
    );
    fill_rect(
        img,
        x0 + f(0.28),
        y0 + f(0.73),
        x0 + f(0.78),
        y0 + f(0.8),
        [0, 0, 0],
    );
    // the cloth: a triangle pointing left
    for y in f(0.2)..f(0.5) {
        let t = (y - f(0.2)) as f64 / (f(0.5) - f(0.2)).max(1) as f64;
        let half = 1.0 - (2.0 * t - 1.0).abs();
        let left = x0 + f(0.55) - (half * 0.3 * s as f64).round() as i64;
        fill_rect(img, left, y0 + y, x0 + f(0.55), y0 + y + 1, [230, 20, 30]);
    }
}

fn mine(img: &mut RgbaImage, x0: i64, y0: i64, s: i64) {
    let (cx, cy, r) = (
        x0 as f64 + s as f64 * 0.53,
        y0 as f64 + s as f64 * 0.53,
        s as f64 * 0.26,
    );
    for y in y0..y0 + s {
        for x in x0..x0 + s {
            let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            let d = (dx * dx + dy * dy).sqrt();
            let spike = (dx.abs() < s as f64 * 0.04
                || dy.abs() < s as f64 * 0.04
                || (dx.abs() - dy.abs()).abs() < s as f64 * 0.05)
                && d < r * 1.45;
            if d < r || spike {
                put(img, x, y, [0, 0, 0]);
            }
        }
    }
    let hl = (s as f64 * 0.08).max(1.0) as i64;
    fill_rect(
        img,
        (cx - r * 0.45) as i64,
        (cy - r * 0.45) as i64,
        (cx - r * 0.45) as i64 + hl,
        (cy - r * 0.45) as i64 + hl,
        WHITE,
    );
}

/// One cell, `s` pixels square, with its top left at `(x0, y0)`.
pub fn draw_cell(img: &mut RgbaImage, x0: i64, y0: i64, s: i64, cell: Cell) {
    match cell {
        Cell::Hidden => closed(img, x0, y0, s, FILL),
        Cell::Flag => {
            closed(img, x0, y0, s, FILL);
            flag(img, x0, y0, s);
        }
        Cell::WrongFlag => {
            closed(img, x0, y0, s, [255, 160, 160]);
            flag(img, x0, y0, s);
        }
        Cell::Open(n) => {
            opened(img, x0, y0, s, FILL);
            if n > 0 {
                let (w, h) = ((s as f64 * 0.46) as i64, (s as f64 * 0.64) as i64);
                draw_digit(
                    img,
                    n as usize,
                    x0 + (s - w) / 2 + 1,
                    y0 + (s - h) / 2 + 1,
                    w,
                    h,
                    DIGIT_COLORS[n as usize],
                );
            }
        }
        Cell::Mine => {
            opened(img, x0, y0, s, FILL);
            mine(img, x0, y0, s);
        }
        Cell::Exploded => {
            opened(img, x0, y0, s, RED);
            mine(img, x0, y0, s);
        }
    }
}

/// Just the cells, `s` pixels each.
pub fn draw_board(board: &Board, s: u32) -> RgbaImage {
    let mut img = RgbaImage::new(board.width() as u32 * s, board.height() as u32 * s);
    for ((x, y), c) in board.iter() {
        draw_cell(
            &mut img,
            (x as u32 * s) as i64,
            (y as u32 * s) as i64,
            s as i64,
            c,
        );
    }
    img
}

/// A game as a page shows it: a white page, a grey frame, the counters and the
/// face above the cells, the cells sunk in the frame (a dark band above and to
/// the left of them, a white one below and to the right). Returns the picture
/// and where the cells are.
pub fn draw_page(
    board: &Board,
    s: u32,
    left: u32,
    top: u32,
    extra: (u32, u32),
) -> (RgbaImage, Grid) {
    let (bw, bh) = (board.width() as u32 * s, board.height() as u32 * s);
    let border = ((s as f64 * 0.72).round() as u32).max(8);
    let band = ((s as f64 * 0.17).round() as i64).max(3);
    let edge = ((s as f64 * 0.08).round() as i64).max(2);
    let panel = s * 2;
    let (w, h) = (
        left + bw + 2 * border + extra.0,
        top + panel + bh + 3 * border + extra.1,
    );
    let mut img = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));
    let (fx, fy) = (left as i64, top as i64);
    let (fw, fh) = ((bw + 2 * border) as i64, (panel + bh + 3 * border) as i64);
    // the frame: a raised grey box
    fill_rect(&mut img, fx, fy, fx + fw, fy + fh, [192, 192, 192]);
    fill_rect(&mut img, fx, fy, fx + fw, fy + edge, WHITE);
    fill_rect(&mut img, fx, fy, fx + edge, fy + fh, WHITE);
    fill_rect(&mut img, fx, fy + fh - edge, fx + fw, fy + fh, LINE);
    fill_rect(&mut img, fx + fw - edge, fy, fx + fw, fy + fh, LINE);
    // counters: black boxes with red digits; the face: a yellow circle on a bevelled button
    let (py, ph) = (fy + border as i64, panel as i64);
    for (i, bx) in [fx + border as i64, fx + fw - border as i64 - (s as i64 * 2)]
        .into_iter()
        .enumerate()
    {
        fill_rect(
            &mut img,
            bx,
            py + ph / 6,
            bx + s as i64 * 2,
            py + ph - ph / 6,
            [0, 0, 0],
        );
        for k in 0..3 {
            let d = if i == 0 { [0, 1, 0][k] } else { [0, 0, 7][k] };
            draw_digit(
                &mut img,
                d,
                bx + 2 + k as i64 * (s as i64 * 2 - 4) / 3,
                py + ph / 6 + 3,
                (s as i64 * 2 - 4) / 3 - 2,
                ph * 2 / 3 - 6,
                [255, 0, 0],
            );
        }
    }
    let fs = (ph as f64 * 0.8) as i64;
    let (bx, by) = (fx + fw / 2 - fs / 2, py + (ph - fs) / 2);
    closed(&mut img, bx, by, fs, FILL);
    let (cx, cy, r) = (
        bx as f64 + fs as f64 / 2.0,
        by as f64 + fs as f64 / 2.0,
        fs as f64 * 0.32,
    );
    for y in by..by + fs {
        for x in bx..bx + fs {
            let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            if dx * dx + dy * dy < r * r {
                put(&mut img, x, y, [255, 230, 0]);
            }
        }
    }
    // the cells, sunk in the frame
    let (gx, gy) = (fx + border as i64, py + ph + border as i64);
    fill_rect(
        &mut img,
        gx - band,
        gy - band,
        gx + bw as i64 + band,
        gy + bh as i64 + band,
        WHITE,
    );
    fill_rect(&mut img, gx - band, gy - band, gx + bw as i64, gy, LINE);
    fill_rect(&mut img, gx - band, gy - band, gx, gy + bh as i64, LINE);
    for ((x, y), c) in board.iter() {
        draw_cell(
            &mut img,
            gx + (x as u32 * s) as i64,
            gy + (y as u32 * s) as i64,
            s as i64,
            c,
        );
    }
    let grid = Grid {
        x: gx as f32,
        y: gy as f32,
        pitch: s as f32,
        cols: board.width(),
        rows: board.height(),
    };
    (img, grid)
}
